# Standalone signed INT8 attention softmax

The `softmax` branch now uses the same external storage convention as the
matmul, GELU and standalone LayerNorm blocks: signed INT8, zero point zero,
explicit real-value scales. This replaces the former signed INT16 input and
unsigned UINT16 probability interface. No SoC, matmul or activation wiring changes
are included. The existing activation implementation is GELU.

| Boundary | Code range | Real value |
| --- | --- | --- |
| Score input | signed INT8, -128..127 | `code * 0.11335523860661063` |
| Probability output | signed INT8, 0..127 | `code / 127` |

The output sign bit is always zero because probabilities are nonnegative.
Using unsigned 0..255 would conflict with the existing signed matmul operands:
codes above 127 would be interpreted as negative. Gamma/LayerNorm and GELU scales
are not reused as probability scales; matching byte types does not imply equal
tensor units. The producing matmul's real output scale must agree with this
score scale after attention scaling and learned attention bias have been applied.
The matmul `OUTPUT_SCALE` Q0.16 multiplier is not itself a real tensor scale.

## Integer computation

```text
q = sat_int8(round_ties_even(score_real / score_scale))
d = max(q) - q                         # 9-bit widened subtraction, 0..255
e = ROM[d]                            # round(exp(-d*score_scale)*65535)
S = sum(e)                            # 24 bits for up to 196 scores
output_code = round_ties_even(e*127/S) # integer division; result 0..127
probability_real = output_code/127
```

ROM has 256 x 16-bit entries, one for every possible INT8 difference. There is
no cutoff address; tiny exponentials naturally round to zero. At least one
exponential equals 65535, so S cannot be zero. `e*127` needs 23 bits and is
implemented in RTL as `(e<<7)-e`. Internal exponentials/reductions are wider
than the INT8 interface, as with matmul/LayerNorm accumulators. Input conversion,
ROM creation and probability normalization all use nearest ties-to-even rounding.

## Run and calibrate

From repository root in PowerShell:

```powershell
& '../.venv/Scripts/python.exe' -m unittest discover -s softmax -p 'test_*.py' -v
& '../.venv/Scripts/python.exe' softmax/capture_reference.py
& '../.venv/Scripts/python.exe' softmax/evaluate_fixed.py
& '../.venv/Scripts/python.exe' softmax/generate_rtl_vectors.py
vsim -c -do softmax/rtl/run_modelsim.do
```

Existing reference captures can be reused; recapture only if model assets or
baseline inference change. The scripts never download weights or modify model
source/checkpoints. Model evaluation requires the existing local TinyViT/PyTorch
environment. Core software tests require NumPy; their Decimal oracle uses stdlib.

The target remains `tiny_vit_21m_224`, `layers.1.blocks.0.attn`, input score shape
`[16,6,49,49]` at batch size one. Softmax acts on each final-axis key-score row.
Scores are captured after QK multiplication, key-dimension scaling and attention
bias. The model source and image preprocessing are recorded by the capture script.

`evaluate_fixed.py` tests 10 score scales (powers of two and calibration-data
ranges), using only the seven calibration images. It minimizes mean per-image
P@V RMSE, freezes the chosen scale, then evaluates the three validation images.
Output remains fixed at scale 1/127 to represent the complete [0,1] range with
signed INT8. The selected score scale is in `example/int8_config.json` and the
ROM is generated from it. `Config()` defaults to the documented provisional
scale 1/8; hardware uses the calibrated supplied ROM. No runtime scale register
is implemented.

Generated `runs/int8_21m/` contains per-image **signed INT8** score/probability
codes, selection/error reports and baseline/replacement logits. Model replacement
dequantizes probabilities before the original floating V multiplication and
projection, checks baseline logits against capture, and verifies exact model
restoration. It replaces only one attention block. No actual upstream integer
QK/V matmul or board captures are simulated.

## Precision limits

There are only 128 nonnegative signed INT8 levels. Independent output rounding
does not preserve an exact probability sum. Uniform rows illustrate the limit:

| Row length | Ideal probability | INT8 code per element | Decoded row sum |
| --- | --- | --- | --- |
| 2 | 0.5 | 64 | 128/127 = 1.00787 |
| 49 | 1/49 | 3 | 147/127 = 1.15748 |
| 196 | 1/196 | 1 | 196/127 = 1.54331 |

This is an explicit format/accuracy tradeoff, not a hidden row-sum correction.
Unit tests and RTL vectors cover these cases. Real-image error and P@V error
are measured separately in `RESULTS.md`. Calibrating a score scale cannot remove
the finite probability-resolution limit. The current scale/accuracy evidence is
for one 49-key attention site, not every 196-key site or full-model quantization.
Ten unlabeled images and matching predictions do not establish classification
accuracy. These images were also inspected in prior experiments.

See `rtl/README.md` for the handshake and iterative divider. Quartus device
resource/timing results are recorded separately in `quartus/README.md`.
