# Signed INT8 softmax migration

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
