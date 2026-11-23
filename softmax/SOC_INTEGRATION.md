# SoC softmax and TinyViT attention

Softmax shares the accelerator's clock, reset and Avalon slave at HPS base
`0xC0000000`. The RTL can fetch signed INT8 scores directly from the
matmul output buffer (OBUF), using the same row layout and read port as Avalon.
With GELU enabled, OBUF contains GELU output. With GELU bypassed, it contains
the final requantized matmul output. Probability reads use the same result
register for both sources.

| Byte offset | Access | Low bits of the 32-bit word |
| --- | --- | --- |
| `0x0100` | Write | Bits 7:0 configure a length from 1 to 196. Zero aborts the current row and clears the length. |
| `0x0110` | Write | Bits 7:0 submit one signed score. |
| `0x0120` | Read | Bits 7:0 carry the probability, bit 8 is VALID, bit 9 is LAST, bit 10 is manual input ready, bit 11 is OBUF ingest busy, and bit 12 reports a rejected OBUF command. |
| `0x0130` | Write | Bits 15:0 select a byte offset within OBUF and bits 23:16 give the row length. Requires byte lanes 0–2. |

For direct ingest, wait for matmul completion and write `0x0130` once per
row. Use the aligned OBUF row offset `query * round_up(canvas_width, 16)`
and pass the actual key count so padding is skipped. Select a completed row
with codes in the softmax input scale. An accepted command replaces the
previous softmax row. The hardware stalls host OBUF reads and new matmul
starts during the fetch phase. A rejected command sets bit 12 until the
next valid command or a zero-length abort. A command is rejected while a
fetch is active, before matmul finishes, after a raw-psum operation, or
for an invalid length, alignment or OBUF range.

For manual input, set the length before writing the first score, then write
exactly that many scores. This path uses byte lane zero and accepts scores
while bit 10 is set. Use aligned, single-beat MMIO. An empty result
read returns immediately with VALID clear. A read with VALID set consumes
one result and LAST identifies the final key. For a burst starting at the
result register only the first beat accesses the core. Subsequent beats are
zero. Serialize host access for each entire row.

`softmax_mmio.h` provides `softmax_obuf_row_mmio` for the direct path,
`softmax_row_mmio` for prequantized host scores, and
`tinyvit_softmax_row_mmio` for host-side score preparation. The header uses
the existing accelerator device mapping and links with `-lm` on the HPS.

## Attention data flow

1. For each window and head, compute `Q[queries, d] @ Kᵀ[d, keys]`.
   In the accelerator's `W @ X` layout, put Q in WBUF and K transposed in
   IBUF. Bypass GELU for attention scores.
2. For each query and key, multiply the decoded QK value by
   `1/sqrt(key_dim)` and add the learned relative-position bias for that
   head/query/key. The calibrated INT8 softmax input scale is
   `0.11335523860661063`. For direct ingest, prepare the OBUF scores at
   this scale before launching softmax. One integer option is to seed the
   canvas with per-key bias codes and issue QK with `TILE_CFG.use_prev_out`
   and `ACT_CFG.prev_int8`. Set the matmul output multiplier to include
   key-dimension scaling in softmax score units. The output loader adds the
   previous bias byte to the rescaled QK value before saturation. This
   option quantizes QK and bias separately. The host adapter can instead
   apply a combined floating-point scale and bias with ties-to-even
   quantization before submitting scores through the manual input.
3. Read the probability row in key order. Bytes are signed INT8
   with nonnegative codes 0–127 and scale `1/127`. For quantized
   `P[queries, keys] @ V[keys, d]`, put P in WBUF and V in IBUF. Include
   `(1/127) * V_scale` when selecting the matmul output multiplier for
   the desired context scale. Bias and GELU are disabled for this multiply.

The calibrated target is `tiny_vit_21m_224`, `layers.1.blocks.0.attn`, with
scores of shape `[16,6,49,49]` at batch size one. Normalize each 49-key
row independently across the last axis. With direct ingest, the host starts
each row and stages the probability bytes into WBUF for P×V.
See [RESULTS.md](RESULTS.md) for the calibration
coverage and measured model error.

## Validation

Run from the repository root so the ROM paths resolve.

```text
vsim -c -do softmax/rtl/run_soc_modelsim.do
qsys-generate adder_prototype.qsys --synthesis=VERILOG
quartus_sh --flow compile adder_prototype
```

The ModelSim test uses the real accelerator RAMs and exercises direct
49-key OBUF → softmax transfers for raw QK, bias-corrected QK, and GELU
output. It also checks a competing Avalon OBUF read and uses the returned
probabilities for P×V. The SoC PLL targets 75 MHz. Check
`output_files/adder_prototype.sta.summary` after a fresh fit.

The initial Avalon-only softmax integration completed in Quartus Prime Lite 18.1
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
