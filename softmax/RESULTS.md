# Signed INT8 softmax results

## Overlapped RAM reads

The shared RAM fetches the next score while writing the previous score's
exponential. The core also fetches the next exponential on each accepted
output. This keeps the single RAM and reduces the unstalled output interval
from 8 to 7 clocks.

ModelSim measured **606 cycles** from the accepted OBUF command to the last
Avalon result for 49 keys, down from 654. This is **8.08 µs at 75 MHz**
before the HPS bridge and host polling. The SoC regression passed and all
691,488 reference output bytes matched.

The standalone fit passed 90 MHz across all four corners with +4.223 ns
worst setup and +0.132 ns worst hold slack. It used 225 ALMs, 193
registers, one M10K (8,192 bits), and zero DSP blocks.

## Shortened divider and shared RAM

Because each exponential is at most the sum, the exact softmax quotient is
at most 127. Loading the high numerator bits as the initial remainder
reduces the radix-4 divide from 12 to 4 cycles and the unstalled output
interval from 16 to 8 cycles. Rounding uses the exact final remainder.

ModelSim measured **654 cycles** from the accepted OBUF command to the last
Avalon result for 49 keys, down from 1,092. That projects to **8.72 µs at
75 MHz**, excluding the HPS bridge and host polling. The golden regression
and integrated test passed, and the comparison matched all 691,488 bytes
from the saved Cortex-A9 output.

The score row, intermediate exponentials and LUT share one 512 × 16 RAM.
Each exponential replaces a score after its final use. The current standalone
fit used **1 M10K (8,192 bits)**, 219 ALMs, 197 registers and zero DSP blocks.
It passed 90 MHz with +3.566 ns worst setup slack. All 256 exponential LUT
entries matched the Quartus RAM initialization. The shortened divider with
separate memories used 3 M10Ks and took the same 654 cycles. The physical
board timing below is for the earlier one-bit-divider build.

## First radix-4 implementation

The first radix-4 RTL processes two quotient bits per clock. Exact division takes
12 cycles per output, down from 23. Ties-to-even rounding uses the exact
remainder. The unstalled output interval is 16 cycles, down from 27.

The integrated ModelSim test measured **1,092 cycles** from the accepted
direct OBUF command to the last Avalon result for a 49-key row, down from
1,536 cycles. That is **14.56 µs at 75 MHz** in simulation. This excludes the
HPS bridge and host polling, and the physical board latency below is from
the earlier one-bit-divider build.

The 578-row golden regression matched all 25,999 output bytes. The benchmark
regression also matched all 691,488 bytes from the saved Cortex-A9 output.
The integrated test passed for raw QK, biased QK, GELU output and P×V readback.
The standalone Cyclone V fit used 214 ALMs, 213 registers, 3 M10Ks
(8,800 bits), and no DSP blocks. It passed 90 MHz timing with +4.418 ns
worst setup slack. The revised full SoC timing and board performance
remain unmeasured.

## Matched TinyViT comparison for the one-bit divider

The benchmark uses all 14,112 rows of 49 keys from the three validation
images at `tiny_vit_21m_224/layers.1.blocks.0.attn`. Each score already
includes query-key scaling and learned relative-position bias. The INT8
comparison uses the calibrated input scale and decodes probabilities as
`code / 127`. The floating-point reference comes from TinyViT's captured
FP32 probabilities. The FPGA implements the INT8 path.

| Comparison | Relative RMSE | Mean absolute probability error | Maximum absolute probability error |
| --- | ---: | ---: | ---: |
| INT8 vs. FP32 on original scores | 6.562% | 0.184 percentage points | 29.338 percentage points |
| INT8 vs. floating-point softmax on the same INT8 scores | 4.717% | 0.166 percentage points | 0.395 percentage points |

Relative RMSE is `100 × norm(INT8 - reference) / norm(reference)`. The
first row includes score quantization and clipping. In the worst row,
three scores clipped at +127. With the original floating-point values
held fixed, INT8 probabilities change `P @ V` by 2.601% relative RMSE.
These figures measure numerical error at one attention block.

The physical DE1-SoC and the native Cortex-A9 C implementation produced
the same 691,488 probability bytes, with zero mismatches (0%). The board
test also verified that the scores staged in OBUF were byte-identical to
the CPU inputs. ModelSim independently matched the same C vectors.

| 49-key row | Latency |
| --- | ---: |
| Pure HPS INT8, scalar C | 5.512 µs mean |
| Pure HPS FP32, scalar C with `expf` | 13.124 µs mean |
| Physical fabric at 75 MHz, with HPS MMIO | 31.853 µs mean |
| Fabric RTL at 75 MHz, simulated Avalon polling | 20.480 µs to last result |

The board timings come from the same run. At this row length, the
fabric path took 5.78× as long as the HPS INT8 implementation. The HPS
measurements use GCC 4.6.3, `-O3`, one warmup and twenty timed passes over
the same quantized scores. The RTL test measures 1,536 cycles
from acceptance of the direct OBUF command to the last probability read
at the Avalon slave. The first probability read occurs at 240 cycles
(3.200 µs). The simulated measurement excludes Cortex-A9 and HPS bridge
latency.

### Reproduce

Prepare the fixture from the repository root, using the existing captures.

```sh
python softmax/benchmark_compare.py prepare
gcc -O3 -std=gnu99 -mcpu=cortex-a9 -mfpu=neon softmax/benchmark_hps.c -o benchmark_hps -lm -lrt
./benchmark_hps softmax/runs/benchmark/scores.txt softmax/example/exp_lut.hex softmax/runs/benchmark/cpu_vectors.txt
python softmax/benchmark_compare.py report
SOFTMAX_ROW_VECTORS=softmax/runs/benchmark/cpu_vectors.txt vsim -c -do softmax/rtl/run_modelsim.do
vsim -c -do softmax/rtl/run_soc_modelsim.do
```

Build with `-DSOFTMAX_BOARD` and run on the DE1-SoC with the direct-softmax
bitstream and HPS bridge enabled to measure fabric wall time. The board
mode stages exact score codes via an identity matmul, verifies the OBUF
contents, and times the softmax command plus HPS polling. Its timing
excludes the staging matmul, output comparison, and subsequent P×V.

The physical test used the DE1-SoC 32-bit DDR profile. The fitted fabric
PLL passed all four corners at 75 MHz, with worst setup slack +2.896 ns
and hold slack +0.114 ns. In that fit, the softmax core used 8,800 memory
bits in 3 M10Ks. The entire SoC used 375 of the device's 397 M10Ks.
This SD image loads its stock FPGA image from
`/etc/init.d/programfpga`. The test loaded the accelerator temporarily
with USB-Blaster, booted Linux using `init=/bin/bash`, and enabled
`hps2fpga` through the kernel's FPGA bridge interface.

## Earlier INT8 migration

Work is on the `softmax` branch. The current boundary is signed INT8 input and
output, zero points zero, matching matmul/GELU/LayerNorm storage conventions.
Internal 16-bit exponentials, 24-bit sums and 23-bit division numerators retain
precision. No accelerator integration or SoC project edits were made.

## Calibrated contract and model results

- Input scale: 0.11335523860661063 (real value = signed code times scale).
- Output scale: 1/127, using signed codes 0..127; sign bit always zero.
- 256 x 16-bit exponential ROM covers all INT8 score differences.
- Input/ROM/normalization rounding: nearest ties to even.
- Exact 23-cycle restoring division; no reciprocal approximation.

Ten candidate input scales were evaluated on seven calibration images, minimizing
mean per-image P@V RMSE. The three remaining images did not influence selection.
The output scale is fixed to cover [0,1] with nonnegative signed INT8. These same
local images have been examined previously; this is not a new independent dataset.
The target is TinyViT-21M `layers.1.blocks.0.attn`, score shape `[16,6,49,49]`.

| Metric | Calibration (7 images) | Validation (3 images) |
| --- | --- | --- |
| Mean per-image probability RMSE | 0.00261026 | 0.00286682 |
| Maximum probability error | 0.380650 | 0.293380 |
| Mean per-image P@V RMSE | 0.0110348 | 0.0127286 |
| Maximum P@V error | 0.396605 | 0.441886 |
| Maximum probability row-sum error | 0.0944882 | 0.0708661 |
| Input clipping count | 1,500 | 2,554 |
| Top-1 agreement | 7/7 | 3/3 |
| Top-5 set agreement | 7/7 | 3/3 |

Across all 10 images, relative probability RMSE is 6.63406% and relative P@V RMSE
is 2.53168%, computed as `100*sqrt(sum(error^2)/sum(reference^2))`. These are
numerical errors, not classification error. Worst final-logit absolute change
is 0.0218391. Baseline logits matched captured reference exactly; removing the
replacement restored baseline logits exactly for each image.

The former 16-bit boundary had validation probability RMSE 4.03157e-5 and P@V
RMSE 1.10274e-4, with maximum row-sum error 1.52590e-4. INT8 format compatibility
has a measurable precision cost. The selected scale minimizes average calibration
context error; it permits some input clipping and rare larger local errors.
There is no application error budget or ground-truth accuracy benchmark here.

Uniform rows expose the coarse output resolution: 49 equal scores become codes
3 (decoded sum 147/127); 196 equal scores become codes 1 (sum 196/127). Independent
rounding is not sum-preserving. The real-image model check covers one 49-key site;
other sites, especially 196-key attention, need separate accuracy evaluation.
The rest of the model remains floating point, including V and P@V/projection.

## Software and RTL verification

Nine unit tests passed. They cover a Decimal normalization oracle, signed INT8
rounding/saturation/byte representation, widened subtraction, ROM bounds, matmul
output-byte emulation, permutation invariance and explicit probability-mass
limits. The independent stable float64 softmax reference tests also pass.

ModelSim Intel FPGA Edition 10.5b passed 578 rows / 25,999 outputs bit-for-bit
in 897,599 main-testbench clocks. All 256 score differences and all 196 legal row
lengths are covered, including 100 real-image rows and 478 synthetic rows.
Protocol tests cover bubbles, backpressure, invalid lengths, last markers,
row-length latching and resets during reception/division/stalled output.
Directed tests check ties and uniform rows, `3*N+26` clocks to first valid and
27 clocks per subsequent unstalled output. Compile/simulation had zero warnings
or errors. Quartus device/timing results are recorded in `quartus/README.md`.
The fixture also contains a 26-key row with exact 0.5-code outputs rounding to
zero, explicitly distinguishing ties-to-even from the previous half-up rule.

## Reproduce and limitations

```powershell
& '../.venv/Scripts/python.exe' -m unittest discover -s softmax -p 'test_*.py' -v
& '../.venv/Scripts/python.exe' softmax/evaluate_fixed.py
& '../.venv/Scripts/python.exe' softmax/generate_rtl_vectors.py
vsim -c -do softmax/rtl/run_modelsim.do
```

Existing `runs/reference_21m/` captures are required for software reevaluation;
`capture_reference.py` can regenerate them from local assets. Reports, signed
INT8 per-image arrays, ROM/config and logits are in ignored `runs/int8_21m/`.
The hardware ROM, representative fixtures and metadata are in `example/`.
Matching data width is not sufficient to connect blocks: upstream attention
scale/bias and downstream matmul operand scales still need explicit agreement.

The INT8 RTL also passed standalone Quartus synthesis/fitting and four-corner
90 MHz timing on `5CSEMA5F31C6`: 175 ALMs, 192 registers, 3 M10Ks, 8,800 memory
bits and zero DSPs. Worst setup slack +3.894 ns, hold +0.113 ns and pulse-width
+4.344 ns; lowest reported slow-corner Fmax 138.56 MHz. All unconstrained counts
are zero. ModelSim and Quartus used identical core/ROM snapshots; source and
fixture hashes are in `rtl/verification.json` and `quartus/metrics.json`.

No board was programmed, and no integration or full-model INT8 accuracy is
claimed. Model evaluation exited successfully; existing timm/wandb dependencies
emitted sandbox temporary-directory cleanup warnings at exit. The legacy
ModelSim run required execution outside the sandbox due to DLL relocation.
