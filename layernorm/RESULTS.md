# INT8 LayerNorm results

See [shared reciprocal](#shared-reciprocal) for the current RTL and
the [integrated canvas benchmark](#integrated-canvas-and-cortex-a9-comparison)
for the earlier controller fit and HPS comparison.

## Shared reciprocal

The row core registers the input and square before accumulation, and
registers the two variance products before subtracting them. It computes
one fixed-point reciprocal of the integer standard deviation per token.
A per-channel multiply estimates the quotient, then an integer residual
gives exact ties-to-even rounding. The output interval fell from 21 to
14 clocks. The first output takes 85 clocks after the last input, up
from 47 clocks, because reciprocal setup is shared across the token.

ModelSim measured **3,337 cycles** from the accepted canvas command to
writeback for 192 channels, down from 4,636. That corresponds to
**44.49 µs at 75 MHz** before HPS bridge traffic and polling. The
standalone regression matched all 200,296 reference INT8 outputs. The
canvas regression passed with GELU enabled and bypassed.

The isolated canvas fit passed 75 MHz across all four corners with
+2.314 ns worst setup and +0.131 ns worst hold slack. It used 950 ALMs,
867 registers, six M10Ks (22,016 bits), and 13 DSP blocks.

## Shared coefficient memory

The canvas controller reads gamma and beta from its host-loaded RAM in the
core's existing read stage. The integrated core buffers activations only
and no longer stores a second copy of the coefficients. The canvas fit uses
**6 M10Ks (22,016 bits)**, down from 8 M10Ks (31,232 bits). ALMs went from
872 to 874. The fit uses 10 DSP blocks and 634 registers. The canvas fit
reports a 69.20 MHz slow-corner Fmax and misses 75 MHz setup by 1.118 ns.

ModelSim still measured **4,636 cycles** for a 192-channel token. This
projects to **66.99 µs** at the standalone fit's 69.20 MHz Fmax, before
bridge traffic and host polling. Full SoC timing and physical fabric
latency for this revision remain unmeasured. The
matmul-to-LayerNorm tests passed with GELU enabled and bypassed. The
standalone interface still accepts gamma and beta with each input, and its
golden regression matched all 200,296 bytes. The standalone fit uses 4 M10Ks
and passes 90 MHz with +0.475 ns worst setup slack.

## Skip leading zero quotient bits

The normalized quotient magnitude fits in 20 bits for all supported channel
counts. Seeding the divider remainder with the upper numerator bits reduces
the radix-4 divide from 21 to 10 cycles per channel and the unstalled output
interval from 32 to 21 cycles. Rounding uses the exact final remainder.

ModelSim measured **4,636 cycles** from the accepted canvas command to the
last writeback for a 192-channel token, down from 6,748. The golden regression
matched all 200,296 INT8 outputs, and the integrated test passed with GELU
enabled and bypassed. The standalone core fit used 537 ALMs and 4 M10Ks
(22,016 bits), passing 90 MHz with +0.027 ns worst setup slack.
The separate canvas fit used 872 ALMs and 8 M10Ks. It missed 75 MHz setup
by 1.171 ns and reported 68.95 MHz slow-corner Fmax. At that reported Fmax,
4,636 cycles project to 67.24 µs before bridge traffic and host polling.

## Radix-4 divider

The divider consumes two bits per clock and retains the exact remainder for
ties-to-even rounding. Division takes 21 clocks per channel, down from 41.
First valid is 58 clocks after the final input, down from 78, and the unstalled
output interval is 32 clocks, down from 52.

ModelSim measured **6,748 cycles** from the accepted canvas command to the last
writeback for one 192-channel token, down from 10,588. That projects to
**89.97 µs** at the configured 75 MHz, excluding bridge traffic and host polling.
The separate canvas-controller fit missed 75 MHz by 1.413 ns and reported
a slow-corner Fmax of 67.82 MHz, corresponding to **99.50 µs** for the same
cycle count. It used 883 ALMs and 8 M10Ks (31,232 bits). Full SoC timing and
board latency for this revision remain unmeasured.

The standalone golden regression matched all 200,296 INT8 outputs. The
integrated canvas test passed with GELU enabled and bypassed. The standalone core fit
used 548 ALMs, 662 registers, 4 M10Ks (22,016 bits), and 8 DSP blocks.
It passed 90 MHz timing with +0.465 ns worst setup slack.

The first section records the demonstration-scale baseline. The calibrated
INT8 scale results follow below. Input, gamma, beta and
output storage is signed INT8 with zero point zero. Float conversion and integer
rounding use nearest ties to even, with saturation to [-128,127]. Default scales
are all `1/32`; input/output scales match the existing GELU example. Wider
accumulators are internal only. No matmul/GELU/softmax integration was added.

The previous width-sweep script, alternative storage-width tests and reports
were removed. Generated current artifacts are in `runs/int8_21m/` (ignored).

## Reproduce

```powershell
& '../.venv/Scripts/python.exe' -m unittest discover -s layernorm -p 'test_*.py' -v
& '../.venv/Scripts/python.exe' layernorm/evaluate_model.py
```

The existing local TinyViT-21M checkpoint, source and 10 images were used.
The report records source/checkpoint/image hashes and preprocessing. NumPy
2.5.1, PyTorch 2.12.1+cpu; CPU inference, 224x224 images. No assets were downloaded
or modified. The target was `layers.1.blocks.0.attn.norm`, shape `[16,49,192]`.

## Results

| Check | Observed result |
| --- | --- |
| Focused INT8 unit tests | 8 passed, none skipped |
| Floating LayerNorm reference checks | 210 passed (21 sites x 10 images) |
| INT8 target outputs tested | 1,505,280 |
| Kernel vs ideal INT8 output: exact matches | 1,505,071 (99.9861%) |
| Kernel vs ideal INT8 output: differing outputs | 209, all differ by 1 code |
| Worst kernel vs ideal INT8 code difference | 1 code = 0.03125 real units |
| Kernel plus output rounding vs float on quantized operands: RMSE | 0.00742228 |
| Kernel plus output rounding vs float on quantized operands: max error | 0.0156531 |
| Input/affine quantization vs original float output: RMSE | 0.110494 |
| Total INT8 output vs original float output: RMSE | 0.110747 |
| Total INT8 output vs original float output: max error | 1.42369 |
| Input values clipped | 472,972 / 1,505,280 (31.4209%) |
| Gamma/beta clipping, output saturation | All zero |
| Worst absolute final-logit change | 0.491375 |
| Baseline/replacement top-1 agreement | 10/10 |
| Capture hooks and model restoration | Final logits bit-exact with baseline |

RMSE values pool equally sized image tensors. The error components do not add
linearly. The ideal INT8 comparator runs independent floating LayerNorm on the
dequantized INT8 input/gamma/beta, then rounds and saturates its output using the
configured output scale. It isolates kernel error from upstream quantization.

The default example scale has an input range of [-4,3.96875], while original
inputs at this site extend approximately from -31.6613 to 27.5311. Clipping
therefore dominates the total error. This run establishes the byte contract and
numerical baseline; it does **not** establish suitable TinyViT deployment scales.
Keeping the same INT8 interface, deployment needs explicit producer/consumer
scales calibrated for the actual layer. The matmul Q0.16 rescale multiplier
cannot be substituted directly for a real-value input scale.

## Scope

Unit tests include the exact GELU example scale and two's-complement encoding,
all 256 input bytes, rounding ties, saturation, signed extrema, constants,
near-zero variance, last-axis behavior and 192/384/576 channels. A 100-digit
Decimal oracle checks integer arithmetic independently; PyTorch checks the
floating formula on dequantized INT8 operands. Matmul boundary tests emulate
INT32 accumulation outputs scaled by a Q0.16 multiplier, arithmetic-shifted and
saturated, and also test bytes from the existing GELU ROM.

Those boundary tests are software emulation, not FPGA captures. The model run
quantizes the captured floating inputs at the proposed INT8 boundary; the rest
of the network remains floating point. One LayerNorm is replaced per inference.
The 10 unlabeled images are a regression set, not a classification accuracy
benchmark, and no application error acceptance threshold has been supplied.

Default conservative intermediate bounds for up to 576 channels are: signed
sum 18 bits, unsigned sum-of-squares 24 bits, unsigned variance numerator 33 bits,
unsigned square-root argument 49 bits, signed normalization numerator 41 bits,
signed affine accumulator 44 bits. Python uses wide integers; RTL register widths,
overflow behavior, hardware square root/division and timing still need testing.
At this software-only stage no reciprocal-square-root LUT or RTL was implemented.
The subsequent standalone RTL implementation is recorded below.

The model evaluation exited successfully. Existing timm/wandb dependencies
emitted two sandbox temporary-directory cleanup warnings at interpreter exit,
after all assertions and artifact writes completed.

## Calibrated INT8 scales

All storage remains INT8. The saved `example/int8_config.json` is the recommended
starting configuration for `layers.1.blocks.0.attn.norm` only:

| Tensor | Scale (real value per integer code) | Approximate real range |
| --- | --- | --- |
| Input | 0.14372939763106699 | [-18.3974, 18.2536] |
| Output | 0.014901076910004555 | [-1.90734, 1.89244] |
| Gamma | 0.010405473821745144 | [-1.33190, 1.32150] |
| Beta | 0.012730173238619106 | [-1.62946, 1.61673] |

Integer scale multipliers are `Mg=45764`, `Mb=55988` in Q16. Scale selection
does not add floating-point vector arithmetic to the proposed hardware kernel.
Epsilon constants are prepared for the selected input scale and channel count.
The existing matmul/GELU hardware remains unchanged; its producer scale must
eventually agree with the selected LayerNorm input scale.

Reproduce after generating the baseline captures:

```powershell
& '../.venv/Scripts/python.exe' layernorm/calibrate_int8.py
& '../.venv/Scripts/python.exe' layernorm/evaluate_model.py --config layernorm/example/int8_config.json --out layernorm/runs/calibrated_21m
```

The first seven filename-sorted images are calibration; the final three
(Chihuahua, Pekinese, redbone) are excluded from scale selection. Their identities
and hashes are recorded. Gamma/beta scales minimize parameter quantization MSE
over a powers-of-two/data-range candidate set. Then 121 input/output scale pairs
are scored using integer LayerNorm calibration RMSE. The chosen pair is frozen
before validation. This is the best tested pair for those fixed affine scales,
not proof of a globally optimal combination.

| Metric | Original 1/32 scales | Calibrated scales |
| --- | --- | --- |
| Validation relative RMSE | 49.2999% | 3.48513% |
| Validation RMSE in real output units | 0.116856 | 0.00826083 |
| Validation maximum absolute error | 1.28959 | 0.610102 |
| Validation input clipping | 32.9983% | 0.0961062% |
| Validation gamma/beta clipping and output saturation | 0 | 0 |

Calibration relative RMSE for the selected config is 3.23718%. Relative RMSE is
`100 * sqrt(sum((fixed-reference)^2) / sum(reference^2))`, not classification
error. The validation worst-case error remains substantially larger than average:
a few input outliers still clip. An application-specific error budget may require
a more conservative input scale, even if its average RMSE is slightly higher.

All final calibration/validation vectorized-search outputs matched the scalar
integer kernel bit-for-bit. Nine unit tests passed, including the new
vectorized-vs-scalar comparison on arbitrary scales and corner cases. The full
search report and selected config are in `runs/calibration_21m/`.

The frozen-config model rerun also completed on all 10 images. All 210 floating
reference checks passed; capture/removal restored bit-exact baseline logits.
The rerun INT8 output codes matched every saved calibration/validation vector
exactly. Top-1 predictions agreed with baseline on 10/10 images (3/3 validation).
Worst absolute logit change was 0.0267665, compared with 0.491375 for the initial
demonstration-scale run. Overall relative RMSE across all 10 images was 3.31919%.
Against ideal INT8 outputs on quantized operands, only 400 of 1,505,280 codes
differed, each by one code. Artifacts are in `runs/calibrated_21m/`.

The three validation images were not used in this search, but were inspected in
earlier experiments. Ten unlabeled local images, one LayerNorm site, and a
single-layer replacement cannot establish model accuracy or deployment readiness.
Other LayerNorm sites need separate scale checks. FPGA integration, timing and
reciprocal-square-root approximation remain unvalidated.

## Standalone RTL completed

`rtl/layernorm_row.sv` now implements the calibrated integer contract without
connecting to other accelerator modules or editing the Quartus/Qsys project.
Paired INT8 input/gamma/beta values stream into a one-row buffer. The core
computes population variance, exact iterative integer square root and division,
applies affine scaling, and emits saturated INT8 outputs with ready/valid/last.

ModelSim Intel FPGA Edition 10.5b results on 2026-11-14 (America/Toronto):

| Check | Result |
| --- | --- |
| RTL compile and simulation errors / warnings | 0 / 0 |
| Golden-vector rows / outputs | 744 / 200,296, all bit-exact |
| Real-image rows / synthetic rows | 80 / 664 |
| Legal row lengths checked | Every length 1..576 |
| Saturated golden outputs | 356, all matched |
| Random input bubbles and output backpressure | Passed |
| Invalid lengths 0/577/1023; length latched only at row start | Passed |
| Reset during reception, sqrt, division and stalled output | Passed |
| Directed +24.5/-24.5 output-code ties | All four outputs correctly rounded to +/-24 |
| First `out_valid` after last input handshake, no stalls | 75 clocks |
| Subsequent output handshake interval, no stalls | 49 clocks |
| Complete main-testbench clocks including stalls/reset cases | 10,740,486 |
| Existing software tests after RTL/vector generation | 9 passed |

Run `vsim -c -do layernorm/rtl/run_modelsim.do` from repository root. The supplied
epsilon ROM and golden vectors make simulation independent of the checkpoint
and Python dependencies. `generate_rtl_vectors.py` regenerates them from saved
calibrated captures; the manifest records configuration, provenance and hashes.

This verifies the finite-width RTL against the chosen integer software model;
the original floating-model approximation error remains the calibrated result
above. This first implementation is serialized (49 clocks per output without
stalls), not a high-throughput pipeline. No Quartus synthesis, routed timing,
FPGA board test or performance benefit is claimed. See `rtl/README.md` for
interface and arithmetic details. The legacy simulator needed execution outside
the sandbox due to its Windows DLL relocation startup error.

## DE1-SoC target synthesis and fit

A subsequent standalone Quartus Prime Lite 18.1 run on `5CSEMA5F31C6` succeeded
in synthesis and fitting: 540 ALMs, 488 registers, 4 M10Ks (22,016 memory bits)
and 9 DSP blocks. The registered virtual-I/O benchmark includes the unchanged
ModelSim-verified core. Source/ROM hashes are recorded in `quartus/metrics.json`.

The 90 MHz constraint was **not met**: worst setup slack -0.686 ns; worst hold
+0.121 ns; worst pulse-width slack +4.338 ns. Lowest reported slow-corner Fmax
was 84.77 MHz. All unconstrained clock/input/output counts were zero. The build
script intentionally reports failure for the negative slack. No board was
programmed and no integrated-SoC timing or speedup is claimed. This establishes
device fit only; timing optimization or a validated lower clock is still needed.
See `quartus/README.md` for reproduction and warning details.

## Timing optimization: 90 MHz passed

The same device (`5CSEMA5F31C6`), seed 1, registered virtual-I/O wrapper and
11.111 ns SDC constraints were retained. No multicycle/false-path exceptions or
clock relaxation were introduced. The SDC hash is unchanged from the baseline.

The original critical path ran from coefficient RAM through the wide gamma
multiplier. Coefficients are now captured during the existing read-wait cycle.
The normalized value was reduced from signed 32 to signed 21 bits using its
conservative <2^20 magnitude bound; the gamma product is correspondingly 29 bits.
The constant multiply is split into registered signed-high/unsigned-low products,
then recombination, beta addition and output magnitude have separate registers.
Affine arithmetic uses 46 bits, exceeding its proven 44-bit requirement.

| Metric | Original | Optimized |
| --- | --- | --- |
| 90 MHz timing | Failed | Passed |
| Worst setup slack, all corners | -0.686 ns | +0.321 ns |
| Worst hold slack, all corners | +0.121 ns | +0.116 ns |
| Worst minimum pulse-width slack | +4.338 ns | +4.337 ns |
| Lowest slow-corner Fmax | 84.77 MHz | 92.68 MHz |
| ALMs | 540 | 510 |
| Registers | 488 | 639 |
| DSP blocks | 9 | 8 |
| M10K blocks | 4 | 4 |
| First-valid latency / unstalled output interval | 75 / 49 clocks | 78 / 52 clocks |

The complete ModelSim regression passed again: 744 rows, 200,296 bit-exact
outputs, four directed signed half-tie outputs, stalls/bubbles/invalid lengths
and resets in reception, square root, division, the new scale multiplication and
output-magnitude stages, and stalled output. The main testbench took 11,341,664
clocks; compilation and simulation had zero errors/warnings. Scale values,
golden vectors and numerical error relative to the floating model are unchanged.

Quartus's four-corner STA had zero warnings, no violations and zero unconstrained
clock/input/output counts. The full build exited successfully with six warnings
(three inferred ROM write-side defaults and three benchmark/license/I/O-placement
warnings). Current metrics and hashes are in `quartus/metrics.json`; original
failing metrics are retained in `quartus/baseline_metrics.json`. Default build
output is now `quartus/build_timing2/` (ignored). `rtl/verification.json` records
the passing RTL regression and matching source hashes.

This meets the standalone 90 MHz target while adding three cycles per output.
It does not establish integrated-SoC timing, physical I/O timing, board execution
or an end-to-end inference speedup. The core remains independent of other modules.
## Integrated canvas and Cortex-A9 comparison

The canvas controller reads completed matmul output with optional GELU,
normalizes one token across its channels, and writes the result back to the
same output buffer. See [the host interface](rtl/CANVAS.md).

Quartus Prime Lite 18.1 fitted the canvas controller and core in isolation on
the Cyclone V `5CSEMA5F31C6`. The row buffers and epsilon ROM use 22,016 bits.
Coefficient RAM adds 9,216 bits. The fitted total is **31,232 bits in eight
M10Ks** (2.02% of 397), 844 ALMs, 630 registers and 10 DSP blocks. The design
reuses the existing matmul output canvas. The standalone controller missed
75 MHz setup timing by 1.025 ns at the worst corner, with a reported
slow-corner Fmax of 69.65 MHz. Full SoC timing and fabric board latency
remain unverified.

| Implementation | Time per 192-channel token |
| --- | ---: |
| DE1-SoC HPS FP32, scalar C | 7.76 µs |
| DE1-SoC HPS INT8, VFP reciprocal estimate with exact integer correction | 40.43 µs |
| DE1-SoC HPS INT8, exact integer division | 158.13 µs |
| Fabric INT8, projected at 75 MHz | 141.17 µs |
| Fabric INT8, projected at the standalone fit's 69.65 MHz Fmax | 152.02 µs |

The HPS medians use seven trials of 12,000 row executions each with GCC 4.6.3
and `-O3 -mcpu=cortex-a9 -mfpu=neon`. The test uses 24 captured rows, eight
from each of three validation images, with operands prepared before timing.
Both INT8 CPU kernels matched all 4,608 expected codes. ModelSim measured
10,588 cycles from an accepted canvas start to done for one token. The
projections exclude matmul, HPS bridge traffic and host polling.

The calibrated INT8 replacement at `layers.1.blocks.0.attn.norm` had
3.485% relative output RMSE against FP32 on the three validation images and
3.319% over all ten images. Scoring the saved logits against the ImageNet
classes in those filenames gave 10/10 top-1 and 10/10 top-5 for both FP32 and
the model with one INT8 LayerNorm replacement. The validation subset gave
3/3 top-1 for both. Dataset-wide accuracy and a fully quantized model were
outside this measurement. ModelSim matched all 200,296 standalone INT8
reference bytes, and the integrated regression passed with GELU enabled
and bypassed.
