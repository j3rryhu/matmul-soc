# Standalone calibrated INT8 LayerNorm RTL

`layernorm_row.sv` implements the calibrated `layernorm_fixed.py` code contract.
It has no dependency on the matmul, GELU or softmax RTL and is not instantiated
by the SoC. No existing Qsys or Quartus project files were edited.

## Interface

| Signal | Meaning |
| --- | --- |
| `clk` | Processing clock |
| `reset` | Synchronous active-high reset; discard the current row/output |
| `in_valid`, `in_ready` | Accept a channel on a rising edge when both are high |
| `in_data` | Signed INT8 input channel code |
| `in_gamma`, `in_beta` | Signed INT8 affine codes paired with that channel |
| `row_length[9:0]` | 1..576 channels; sampled only on first accepted channel |
| `out_valid`, `out_ready` | Transfer an output on a rising edge when both are high |
| `out_data` | Signed INT8 output code |
| `out_last` | High with the final valid output of the row |

Hold all input values and `in_valid` stable until an input handshake. No input
`last` is needed: exactly the latched number of channels completes the row.
Subsequent `row_length` changes are ignored for that row. An invalid length blocks
the first input handshake. Input ready is low while computing or emitting outputs.
The next row can start after the final output handshake. No simultaneous rows
or parameter-loading channel is implemented: send gamma/beta again with each row.

Output data, valid and last remain stable under backpressure. Reset clears pending
valid and partial computations; it does not clear memories. Deassert reset before
starting a fresh complete row. Input/output arrays must use token channel order;
this core performs no matmul canvas address remapping.

## Fixed numerical contract

The four real-value scales are fixed by `../example/int8_config.json`:

```text
input_scale  = 0.14372939763106699
output_scale = 0.014901076910004555
gamma_scale  = 0.010405473821745144
beta_scale   = 0.012730173238619106
```

Zero points are zero. Epsilon is `1e-5`. Storage is INT8, including gamma/beta.
Internal reduction, square root, normalization and affine arithmetic are wider.

```text
S = sum(input_code)
Q = sum(input_code^2)
D = N*Q - S^2
E = max(1, round_ties_even(eps*N*N/input_scale^2))
denominator = floor_sqrt((D+E) << 16)
z = round_ties_even(((N*input_code-S) << 22) / denominator)
affine = z*gamma_code*45764 + ((beta_code*55988) << 14)
output_code = sat_int8(round_ties_even(affine / 2^30))
```

The epsilon ROM supplies E for each legal N. Gamma/beta scale ratios are the
Q16 constants 45764 and 55988. Normalization has 14 fractional bits, and square
root has 8 extra fractional bits. Both signed divisions round nearest with ties
to even by applying the rule to magnitude before restoring sign. Square root is
floor, exactly as in software. All saturation occurs after the affine stage.

Important widths include signed 18-bit S, unsigned 24-bit Q, unsigned 33-bit D,
49-bit square-root argument (padded to 50 bits), 25-bit denominator and signed
41-bit normalization numerator. A 25-step digit-by-digit square root and a
41-step restoring divider avoid combinational `/` or `$sqrt`. Normalized results
use signed 21-bit storage, the gamma product signed 29-bit storage, and affine
arithmetic signed 46-bit storage; these cover the software bounds for this
fixed config. Coefficients are captured into channel registers during the
existing `WAIT_READ` state, separating M10K read paths from multipliers without
adding a cycle. Memories hold 576 x 24 input/affine bits plus a 1024 x 8 epsilon ROM.
Quartus inferred four M10K blocks and eight DSP blocks for the optimized
standalone benchmark; see `../quartus/README.md` for measured resources/timing.

The constants and epsilon ROM form one numerical contract. Changing scales or
epsilon requires updating/regenerating both and rerunning verification. The
`EPS_FILE` parameter changes only the file location, not the numerical format.
Runtime scale changes, coefficient banking, reciprocal-square-root LUTs and
multiple channel lanes are not implemented.

The normalized-code width uses a conservative bound. With N<=576 and K=14,
the exact normalized magnitude is at most `sqrt(575)*2^14`. Floor square root
can enlarge it by at most a factor of two, and nearest rounding is covered by
`2*(floor(sqrt(575*2^28))+1) < 2^20`. Signed 21 bits therefore preserve the
integer result, including signed INT8 extrema; its product with an 8-bit gamma
fits 29 bits. This removes unnecessary width from the former 32x8 multiplier.

For timing, the constant gamma multiplication is split exactly: write the signed
29-bit `norm_gamma = signed_high13*65536 + unsigned_low16`. Multiply each part
by 45764 into separate signed 30-bit / unsigned 32-bit registers, then recombine
in a separate 46-bit addition stage. Beta multiplication has its own signed
25-bit register. Adding shifted beta and computing output magnitude are further
registered stages. This avoids a chain of multipliers and additions in one
clock, and introduces no extra numerical approximation or rounding step.

## Simulation

Run from repository root:

```powershell
vsim -c -do layernorm/rtl/run_modelsim.do
```

The script uses a separate `layernorm/rtl/work` library without modifying library
mappings. Compile/simulation errors or a missing testbench pass flag produce
a nonzero exit code. No board is programmed. `epsilon.hex`, `rows.txt` and
`manifest.json` are supplied so simulation needs no Python environment, model
checkpoint or original test images.

Regenerate golden vectors with the calibrated model captures available:

```powershell
& '../.venv/Scripts/python.exe' layernorm/generate_rtl_vectors.py
```

The manifest records hashes, scale/width metadata, counts and the random seed.
Rows contain signed decimal `input gamma beta expected_output` values, with
row-length headers and a file-level row/output-count header.

The testbench checks every output byte and `out_last`, randomized input bubbles,
output stalls, busy input pressure, invalid lengths, and row-length latching.
It resets during partial reception, square-root iteration, divider iteration,
scale multiplication, output magnitude and stalled output, then verifies
recovery. Synthetic vectors cover every legal
length, all input extrema, constant and near-zero variance rows, random signed
affine codes and output saturation. Real vectors sample eight rows per captured
image, including the scale-search validation images; no scales are tuned here.

The directed testbench additionally sends two real-interface rows whose exact
affine outputs are +24.5 and -24.5 codes; both must round to the even code +/-24.
It measures first-valid latency and the unstalled output interval. The default
simulation script runs both testbenches. To run only the short directed test:

```powershell
vsim -c -do "do layernorm/rtl/run_modelsim.do directed"
```

ModelSim Intel FPGA Edition 10.5b passed 744 golden rows / 200,296 outputs in
11,341,664 testbench clocks, plus the four directed tie outputs. Compilation and
simulation reported zero errors/warnings. The vector set includes 80 real rows,
664 synthetic rows and 356 saturated outputs. `verification.json` records source
and fixture hashes. On this Windows setup the legacy 32-bit simulator required
execution outside the sandbox because sandbox startup failed with a DLL
relocation error; the simulation itself ran successfully.

## Latency and current limits

With no stalls, `out_valid` first asserts 78 clocks after the final input
handshake. Subsequent outputs can transfer every 52 clocks. This is a serialized
correctness baseline, not a one-channel-per-clock accelerator. The 25-step square
root is shared across the row; the 41-step division repeats for each channel.
The original unpipelined affine stage used 75/49 clocks; three registered
arithmetic stages now trade extra latency for shorter timing paths.

ModelSim bit-exact verification establishes the selected integer behavior only.
Standalone Quartus synthesis and fitting succeeded, using 510 ALMs, 4 M10Ks and
8 DSP blocks. The optimized core meets the 90 MHz constraint with worst setup
slack +0.321 ns, hold +0.116 ns and pulse-width +4.337 ns. Lowest reported
slow-corner Fmax is 92.68 MHz. Physical board testing and integrated-SoC timing
remain pending. No inference speedup is claimed. See `../quartus/README.md`.
The selected scales were checked at one TinyViT LayerNorm site; other sites need
their own range/accuracy validation before use.
