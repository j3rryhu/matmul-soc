# Standalone INT8 LayerNorm

Branch: `LayerNorm`. Software and standalone RTL; no connection to matmul, GELU,
softmax, Qsys, or the Quartus project. See `rtl/README.md` for the RTL interface,
integer arithmetic, simulation commands and implementation limits.

The interface follows the existing matmul/GELU design:

- Input, gamma, beta and output are signed INT8 codes, zero point zero.
- Real values are `code * scale`, as in `activation/gelu_lut.py`.
- Float-to-code and integer division round nearest, ties to even. Output
  saturates to [-128,127]; raw bytes use two's-complement encoding.
- Default input/output scales are `1/32`, matching
  `activation/example/gelu_metadata.json`. Gamma/beta also default to `1/32`.
  All four scales are explicit configuration values, not learned automatically.
- The matmul `OUTPUT_SCALE` Q0.16 register is its accumulator-to-INT8 multiplier,
  **not** this real-value input scale. The producer's actual output scale must
  agree with LayerNorm's input scale.

There is one INT8 storage contract, with explicit real-value scales.

## Test

From the repository root in PowerShell:

```powershell
& '../.venv/Scripts/python.exe' -m unittest discover -s layernorm -p 'test_*.py' -v
& '../.venv/Scripts/python.exe' layernorm/evaluate_model.py
```

For the calibrated first LayerNorm target, use the saved recommended config:

```powershell
& '../.venv/Scripts/python.exe' layernorm/evaluate_model.py --config layernorm/example/int8_config.json --out layernorm/runs/calibrated_21m
```

Its input/output scales are approximately `0.1437294` and `0.01490108`;
gamma/beta scales are `0.01040547` and `0.01273017`. This keeps all storage INT8
and improves scale-search validation relative RMSE from 49.30% to 3.49% at
`layers.1.blocks.0.attn.norm`. It is target-specific, not a global model scale.
The default `Config()` remains the GELU demonstration-scale baseline so older
experiments remain reproducible; the calibrated values are loaded explicitly.

To reproduce scale selection from the original captured INT8-baseline run:

```powershell
& '../.venv/Scripts/python.exe' layernorm/calibrate_int8.py
```

The filename-sorted first seven images select scales; the remaining three are
excluded from selection and used to evaluate the frozen config. Gamma/beta
scales minimize parameter quantization MSE. A 121-pair input/output search then
minimizes calibration LayerNorm RMSE with exact integer arithmetic. Candidate
scales include powers of two and ranges derived only from calibration tensors.
Every final calibration/validation output must match the scalar integer kernel
bit-for-bit. Reports, split identities, candidate scores and INT8 vectors are
written under `runs/calibration_21m/`. These images were used in earlier
experiments, so this is not a new independent accuracy benchmark.

The existing environment has NumPy and the optional PyTorch oracle. The model
runner additionally uses local TinyViT source/checkpoint/images and existing
torchvision/timm/Pillow dependencies; it does not download or modify model files.
Options: `--model-root`, `--checkpoint`, `--images`, `--layer`, `--out`, `--config`,
`--input-scale`, `--output-scale`, `--gamma-scale`, `--beta-scale`.
Default fixed target: `layers.1.blocks.0.attn.norm`.

The default `1/32` scales are the existing GELU **demonstration** scales, not
calibrated TinyViT deployment scales. Values outside their representable range
will clip. The report includes that loss rather than silently selecting a
different scale or wider interface.

## Computation

Normalize across the last/channel axis using population variance:

```text
y = gamma * (x - mean(x)) / sqrt(mean((x - mean(x))^2) + eps) + beta
```

`layernorm_reference.py` is the independent float64 oracle.
`layernorm_fixed.py` quantizes inputs/parameters to INT8 and exposes a code-level
kernel accepting and returning signed bytes. Wider internal integers are required
for reductions; they do not change the INT8 interface.

For N channels, input codes q and scales sx, sy, sg, sb:

```text
S = sum(q)
Q = sum(q*q)
D = N*Q - S*S
E = max(1, round_ties_even(eps * N*N / sx^2))
root = floor_sqrt((D + E) << 16)
z = round_ties_even(((N*q - S) << 22) / root)     # 14 fractional bits
Mg = round_ties_even((sg/sy) * 65536)
Mb = round_ties_even((sb/sy) * 65536)
output = sat_int8(round_ties_even((z*g*Mg + (b*Mb << 14)) / 2^30))
```

E, Mg and Mb are host-prepared constants for the configuration/row length.
All vector arithmetic uses integer codes. The positive E floor prevents division
by zero, but changes the effective epsilon if the requested epsilon is too small
to represent. For the default scales, Mg=Mb=65536 exactly. Other scale ratios
have the documented Q16 multiplier approximation.

For N<=576 and default scales/epsilon, conservative intermediate widths include
18-bit signed sum, 24-bit unsigned sum-of-squares, 33-bit unsigned variance
numerator and 49-bit unsigned square-root argument. `Config.metadata()` reports
the remaining bounds and scale multipliers. Python uses wide integers; finite
register widths and overflow behavior are checked by standalone RTL simulation;
physical implementation and timing still need validation. Exact integer
square root/division are the current baseline;
no reciprocal-square-root LUT approximation has been tested. The standalone RTL
now implements this baseline using iterative square root and division.

## Validation scope and outputs

Nine focused unit tests cover all INT8 byte values, the existing GELU scales,
ties-to-even rounding, saturation, population variance, last-axis normalization,
PyTorch on dequantized INT8 inputs, an independent Decimal integer oracle,
constants/near-zero variance, channel permutations, invalid inputs and
192/384/576-channel rows. A further test checks the calibration search against
the scalar integer kernel, including non-power-of-two scales and corner cases.
Boundary tests emulate matmul's INT32 multiply/shift/
saturate conversion and feed the existing GELU ROM bytes into LayerNorm. They
do not execute the matmul or GELU hardware or connect the modules.

The model runner checks all floating LayerNorm outputs, then measures one INT8
replacement. It reports three separate comparisons:

1. End-to-end layer error vs the original floating model output.
2. Loss from quantizing/clipping the upstream values and gamma/beta.
3. Kernel/output rounding error vs a float oracle on those quantized values,
   including byte differences against the ideally requantized INT8 result.

These error components do not add linearly. Capture hooks and removal of the
replacement must restore bit-exact baseline logits. The real-image inputs are
quantized at the proposed upstream INT8 boundary; they are not FPGA captures or
an emulation of the full quantized matmul/residual network.

`runs/int8_21m/` is ignored by Git. Its report contains config, source/checkpoint/
image hashes, clipping, errors and logits. Per-image NPZ files contain INT8
input/gamma/beta/output vectors, ideal INT8 outputs, floating references and
baseline/replacement logits. See `RESULTS.md` for the completed retest.

## Standalone RTL verification

```powershell
vsim -c -do layernorm/rtl/run_modelsim.do
```

The calibrated core passed bit-exact ModelSim checks on 744 rows / 200,296
golden outputs, plus directed signed half-tie rounding and latency checks.
All 576 legal row lengths, reset recovery and output backpressure are covered.
The optimized core passes Quartus synthesis/fitting and 90 MHz timing on the
DE1-SoC target device (worst setup +0.321 ns, reported slow-corner Fmax 92.68 MHz).
All golden outputs remain unchanged; first-valid/output-interval latency is now
78/52 clocks, trading three extra stages for shorter timing paths.
See `rtl/README.md` and `quartus/README.md`. No board execution is claimed.
