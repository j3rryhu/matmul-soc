# TinyViT 21M fixed-point softmax baseline

Run date: 2026-11-02. Target: `layers.1.blocks.0.attn`.
Seven calibration images and three held-out images from the local example set.

Selection: minimum mean per-image P @ V RMSE over calibration only, with
ROM size breaking exact ties. Nine configurations were tested: score
fractional bits 4/6/8 and exponential cutoff 8/12/16. Validation did not
participate in selection.

Selected contract:

- Signed 16-bit scores, real scale 1/256; nearest ties-to-even quantization.
- 3073-entry, 16-bit exponential ROM for differences 0 through 12, step 1/256.
- ROM values are round(exp(-difference) * 65535); larger differences output zero.
- Signed score subtraction widened to 17 bits; sum width 24 bits for 196 keys.
- Unsigned 16-bit probability codes, scale 1/65535.
- Exact integer normalization with nearest ties-up rounding; rounded numerator
  uses a conservative 33-bit temporary. Independent rounding permits row sums
  to differ slightly from one.

| Metric | Calibration (7 images) | Validation (3 images) |
| --- | ---: | ---: |
| Mean per-image probability RMSE | 3.79884e-5 | 4.03157e-5 |
| Maximum probability error | 8.17924e-4 | 7.96226e-4 |
| Mean per-image P @ V RMSE | 9.91922e-5 | 1.10274e-4 |
| Maximum P @ V error | 1.28132e-3 | 1.48496e-3 |
| Maximum probability row-sum error | 1.37331e-4 | 1.52590e-4 |
| Input saturation count | 0 | 0 |
| Top-1 agreement with float model | 7/7 | 3/3 |
| Top-5 set agreement with float model | 7/7 | 3/3 |
| Maximum final logit error | 2.27690e-4 | 1.47343e-4 |

All six unit tests passed. Model baseline logits matched the captured
reference exactly. Removing the replacement hooks restored original logits
exactly for every image. Only one attention block's probabilities were replaced;
all other model arithmetic remained floating point. The model test dequantizes
probabilities to float32 before P @ V; standalone context metrics use float64.

This is not ground-truth classification accuracy, full-model quantization,
or hardware validation. No acceptance threshold was supplied. The 10 example
images are insufficient for a deployment accuracy claim. Integer division is
the correctness baseline; reciprocal approximation remains future work.

Reproduce using the commands in README.md. Full metrics, integer vectors,
ROM HEX and baseline/modified logits are in `runs/fixed_21m/` (Git-ignored).
The installed timm/wandb dependency emitted sandbox temporary-directory cleanup
warnings at exit; computation passed and the process exited with code 0.

## RTL verification

The standalone `rtl/softmax_row.sv` now implements this contract. ModelSim
Intel FPGA Edition 10.5b passed all 147 rows (8666 output codes) bit-for-bit
against the Python integer model, with zero errors and warnings. Tests cover
input gaps, output backpressure/stability, last markers, invalid lengths,
signed limits and reset recovery. See `rtl/README.md` for reproduction.

## Quartus synthesis and timing

Standalone routed results for Cyclone V `5CSEMA5F31C6`, approximately 90 MHz:
the original combinational-divider implementation used 2665 ALMs and failed
setup timing at -69.583 ns. The revised exact iterative divider and synchronous
memory reads use 222 ALMs, 264 registers, 10 M10K blocks and no DSPs. Worst
setup slack is +3.188 ns and hold slack +0.129 ns over all four analyzed
corners. Lowest reported slow-corner Fmax is 126.21 MHz.

The revised RTL still matches all 8666 integer outputs bit-for-bit. Output
interval increases to 35 cycles. See `quartus/README.md` for constraints,
warnings, resource/latency tradeoffs and reproduction. Matmul integration
and integrated-SoC timing have not been performed.
