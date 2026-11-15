# INT8 software retest

Latest hardware status: the optimized standalone RTL meets the unchanged 90 MHz
constraint on the DE1-SoC target device. See the final timing-optimization section.

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
