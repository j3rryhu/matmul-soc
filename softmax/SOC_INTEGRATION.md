# SoC softmax and TinyViT attention

Softmax shares the accelerator's clock, reset and Avalon slave at HPS base
`0xC0000000`. The host passes one signed INT8 query row at a time. Matmul
continues to use main's row-major buffers and its `ACT_CFG` options, including
accumulator-unit bias and GELU.

| Byte offset | Access | Low bits of the 32-bit word |
| --- | --- | --- |
| `0x0100` | Write | Bits 7:0 configure a length from 1 to 196. Zero aborts the current row and clears the length. |
| `0x0110` | Write | Bits 7:0 submit one signed score. |
| `0x0120` | Read | Bits 7:0 carry the probability, bit 8 is VALID, bit 9 is LAST, and bit 10 is input ready. |

Use aligned, single-beat accesses. Writes require byte lane zero. Set the
length before writing the first score, then write exactly that many scores
before reading results. The core latches the length with the first score.
Writes to the score port while input ready is low are ignored. An empty result
read returns immediately with VALID clear. A read with VALID set consumes
one result and LAST identifies the final key. For a burst starting at the
result register only the first beat accesses the core. Subsequent beats are
zero. Serialize host access for each entire row.

`softmax_mmio.h` provides a polling row call for prequantized scores and a
TinyViT adapter accepting a QK row from OBUF and its real scale. The adapter
uses the signed INT8 scale of `0.11335523860661063` and ties-to-even rounding.
The header uses the existing accelerator device mapping and links with
`-lm` on the HPS.

## Attention data flow

1. For each window and head, compute `Q[queries, d] @ Kᵀ[d, keys]`.
   In the accelerator's `W @ X` layout, put Q in WBUF and K transposed in
   IBUF. Use `ACT_CFG=0` to produce INT8 QK scores. Read successive query rows
   from OBUF using main's 16-byte row alignment.
2. For each query and key, multiply the decoded QK value by
   `1/sqrt(key_dim)` and add the learned relative-position bias for that
   head/query/key. Pass this final score to softmax in the calibrated
   INT8 format. The adapter in `softmax_mmio.h` performs these conversions.
   Its `qk_scale` is the real scale of the QK output byte, which depends on
   the Q and K scales and the programmed multiplier.
3. Read the probability row in key order. Bytes are signed INT8
   with nonnegative codes 0–127 and scale `1/127`. For quantized
   `P[queries, keys] @ V[keys, d]`, put P in WBUF and V in IBUF. Include
   `(1/127) * V_scale` when selecting the matmul output multiplier for
   the desired context scale. Bias and GELU are disabled for this multiply.

The calibrated target is `tiny_vit_21m_224`, `layers.1.blocks.0.attn`, with
scores of shape `[16,6,49,49]` at batch size one. Normalize each 49-key
row independently across the last axis. The host stages the score and
probability rows through MMIO. See [RESULTS.md](RESULTS.md) for the calibration
coverage and measured model error.

## Validation

Run from the repository root so the ROM paths resolve.

```text
vsim -c -do softmax/rtl/run_soc_modelsim.do
qsys-generate adder_prototype.qsys --synthesis=VERILOG
quartus_sh --flow compile adder_prototype
```

The ModelSim test uses the real accelerator RAMs and Avalon transfers for a
synthetic 49-key QK → softmax → P×V path, and separately verifies GELU on
matmul output. The SoC PLL targets 75 MHz. Check
`output_files/adder_prototype.sta.summary` after a fresh fit.

The integrated Cyclone V build completed in Quartus Prime Lite 18.1
using a scratch copy with the Platform Designer IP version tags adapted
from 23.1 to 18.1. TimeQuest reported the fabric PLL at 75.0 MHz
(13.333 ns), with zero total negative slack in all four corners.

| Fabric PLL check | Worst slack across four corners |
| --- | ---: |
| Setup | +3.260 ns |
| Hold | +0.107 ns |
| Recovery | +5.344 ns |
| Removal | +0.311 ns |
| Minimum pulse width | +5.405 ns |

TimeQuest reported zero unconstrained clocks. It also listed external
HPS and JTAG I/O ports without delay constraints. The timing result covers
the FPGA clocked logic and the report's constrained paths. Board execution
and full TinyViT inference were outside this validation.
