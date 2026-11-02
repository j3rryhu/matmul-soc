# TinyViT attention softmax

## Step 1: Locate the model operation

Verified against the local model source on 2026-11-02:

`../TinyVit-on-FPGA/tiny-vit/models/tiny_vit.py`

- `Attention` starts at line 216.
- `Attention.forward()` starts at line 262.
- Lines 277-282 compute scores: `Q @ K.transpose(-2, -1)`, multiplied
  by `key_dim ** -0.5`, then added to learned attention bias.
- Line 283 applies `attn.softmax(dim=-1)`.
- Line 284 multiplies the resulting probabilities by `V`.

```text
Q, K -> QK^T -> scale -> attention bias -> SOFTMAX -> P
                                                    |
V ------------------------------------------------> P @ V -> projection
```

This is attention softmax inside the transformer blocks. The hardware input
boundary should include the completed scaling and attention-bias addition.

## First target

Use `tiny_vit_21m_224` and target `layers.1.blocks.0.attn` first, consistent
with the stage/block used for the existing GELU experiment. Its constructor
is at lines 667-676. The configuration has embedding dimensions
`[96, 192, 384, 576]`, heads `[3, 6, 12, 18]`, and window sizes
`[7, 7, 14, 7]`.

For the first target, the source-derived expected shapes are:

| Tensor | Shape |
| --- | --- |
| Attention module input | `[B_windows, 49, 192]` |
| Q, K, V | `[B_windows, 6, 49, 32]` |
| Scores and probabilities | `[B_windows, 6, 49, 49]` |
| P @ V, before reshaping | `[B_windows, 6, 49, 32]` |

`B_windows` includes the image batch and partitioned windows. Softmax
normalizes each row of 49 key scores independently for each query and head.
Other stages require checking their window sizes; do not hardcode 49 for
the complete model (the 14x14 windows have 196 keys).

These shapes have now been checked by inference on all 10 local example
images: `B_windows=16`, score shape `[16, 6, 49, 49]`.

## Next step: Capture a floating-point reference

Run the existing local 21M checkpoint in evaluation mode with the same image
preprocessing used by the GELU experiment. Capture the scores immediately
before line 283, the probabilities immediately after it, and V. Also save
the baseline final logits and input image identity.

Softmax is a tensor method here, not a separate `nn.Softmax` module, so a
module output hook alone cannot capture its internal inputs and outputs.
Instrument the target attention forward function, preserving its original
operations and checking that baseline logits remain unchanged.

Compare a stable floating-point reference against the captured probabilities:

```text
shifted = scores - max(scores, axis=-1, keepdims=True)
exp_values = exp(shifted)
probabilities = exp_values / sum(exp_values, axis=-1, keepdims=True)
```

Record observed score ranges, row lengths, probability error, row-sum error,
and P @ V error before selecting fixed-point widths or LUT ranges.

No model code, checkpoint, or FPGA datapath was changed during step 1.

## Completed: floating-point capture and reference

From the matmul-soc repository root in PowerShell:

```powershell
& '../.venv/Scripts/python.exe' -m unittest discover -s softmax -p 'test_*.py'
& '../.venv/Scripts/python.exe' softmax/capture_reference.py
```

The script accepts `--model-root`, `--checkpoint`, `--images`, `--layer`,
and `--out`. It never downloads weights. It loads the checkpoint strictly,
then enables evaluation mode to populate attention-bias caches from the
loaded weights.

Rather than replacing forward methods, the script hooks the actual QKV
linear output and reconstructs scores, PyTorch probabilities, and context.
The reconstructed projected attention output must match the original
attention output bit-for-bit. Full model logits with and without hooks
must also match bit-for-bit for every image. The independent NumPy
float64 softmax is checked against PyTorch float32 probabilities and P @ V.

Outputs are saved under `softmax/runs/reference_21m/` (ignored by Git):

- Per-image NPZ: scores, PyTorch probabilities, V, PyTorch P @ V,
  independent reference probabilities, and baseline logits.
- `report.json`: hashes, versions, preprocessing, shapes, score ranges,
  errors, and separate calibration/validation image assignments.
- `logits.npz`: all baseline logits and image filenames.

The 2026-11-02 run passed all 3 unit tests and all 10 real-image checks.
The filename-sorted split is 7 calibration images and 3 held-out images;
duplicate file contents are rejected. Worst probability absolute error
was approximately `1.84e-7`. Calibration scores ranged from `-24.1476`
to `17.9202`; validation scores ranged from `-24.8841` to `17.7919`.

This is a floating-point baseline for one attention block, not fixed-point
validation or a classification accuracy benchmark. Next, use only calibration
data to select score quantization, exponential LUT range/precision and
probability width, then evaluate the fixed choices on held-out data.

The installed timm dependency imports wandb, which emitted temporary-directory
cleanup permission warnings at process exit in the sandbox. Computation,
assertions and artifact generation completed successfully (exit code 0).

## Fixed-point software baseline

```powershell
& '../.venv/Scripts/python.exe' -m unittest discover -s softmax -p 'test_*.py'
& '../.venv/Scripts/python.exe' softmax/evaluate_fixed.py
```

`softmax_fixed.py` defines the integer contract. Float-to-score quantization
uses nearest ties-to-even rounding and signed saturation. All subsequent
softmax arithmetic operates on integer codes:

```text
d = max(score_codes) - score_code       # widened unsigned/nonnegative difference
e = ROM[d]                             # zero beyond ROM range
S = sum(e)                             # includes e(maximum) = 65535; never zero
p_code = (e * 65535 + S//2) // S        # nearest, ties up
p_real = p_code / 65535
```

The evaluator compares 9 configurations: 4/6/8 score fractional bits and
exponential cutoffs 8/12/16. All use signed 16-bit scores, 16-bit exponential
codes and unsigned 16-bit probability codes. It selects minimum mean
per-image context (P @ V) RMSE using calibration only, breaking exact ties
by ROM size, before evaluating validation data. This is an accuracy-oriented
baseline within the tested grid, not a hardware cost optimization.

The calibration selection is 8 fractional bits and cutoff 12 (3073 x 16-bit
ROM, about 6 KiB). A score code represents `code / 256`. Differences need
17 bits; sums for up to 196 keys need 24 bits. The `e * 65535` product fits
32 bits, but adding the rounding term requires a conservative 33-bit
temporary. Independently rounded probability codes need not sum exactly
to 65535; row-sum error is reported rather than silently corrected.

`runs/fixed_21m/` contains the ROM HEX, config metadata, per-image integer
test vectors, comparison report and baseline/replacement logits. The model
test replaces only the target attention output using the original QKV and
projection with fixed-point softmax probabilities. It checks baseline logits
against the captured run and verifies exact restoration after each image.

The software implementation uses integer division; a reciprocal approximation
remains future work. Input QK multiplication, scaling, bias addition,
V, P @ V and the rest of the model remain floating point. The input boundary
is therefore a proposed quantization contract, not a simulation of the
current FPGA matmul output. No acceptance thresholds or labeled accuracy
benchmark have been supplied.

## Standalone RTL

`rtl/softmax_row.sv` implements the selected integer contract and passed
bit-exact ModelSim verification on 147 rows / 8666 outputs. See
`rtl/README.md` for handshake details, limitations and simulation commands.
The generated ROM and vectors in `example/` allow simulation without the
local model checkpoint. Standalone Quartus synthesis and routed timing now
pass approximately 90 MHz using an exact iterative divider and block memories:
222 ALMs, 10 M10K blocks, worst setup slack +3.188 ns. See `quartus/README.md`.
Accelerator integration and full-SoC timing remain future work.
