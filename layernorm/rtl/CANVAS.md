# Matmul, GELU and LayerNorm on the SoC

The matmul accelerator stores its finished INT8 output in the existing canvas
at `0x9000`, using the tiled layout from `main`. Bias and optional GELU are
applied by the output loader on the final contraction pass. An explicit
LayerNorm command then gathers each token's channels directly from the canvas
and writes the normalized codes back to their original locations. For a
matmul `W[K,N] @ X[N,M]`, canvas row `k` is a channel and column `m` is a
token. Each normalization operates on the `K` values in one column.

## Host interface

Offsets are relative to the accelerator base `0xC0000000`.

| Offset | Access | Meaning |
| --- | --- | --- |
| `0x0100–0x057f` | Write | Signed INT8 gamma and beta pairs. Channel `k` uses bytes `0x0100 + 2*k` (gamma) and `0x0101 + 2*k` (beta). Byte enables select the bytes written. |
| `0x0c00` | 32-bit write | Bit 31 starts LayerNorm. Bit 30 enables conversion from the default GELU output scale. Bits 22:10 hold the number of tokens. Bits 9:0 hold the number of channels. Write zero to the reserved bits. |
| `0x0c00` | Single-beat read | Bit 0 is busy, bit 1 is done and bit 2 is error. Done and error persist until the next start or reset. |

1. Upload the calibrated gamma and beta codes with enabled bytes for each active
   channel. For a 32-bit write at `0x0100 + 4*j`, the four bytes from least to
   most significant are gamma and beta for channels `2*j` and `2*j+1).
2. Run matmul with the desired bias/GELU settings in `ACT_CFG`. Wait for matmul
   done after each invocation. For tiling, finish every tile and the final
   contraction first, filling the intended rectangle from canvas row and column
   zero. The command uses the canvas width and padded pitch latched by the
   last matmul and requires that invocation to have emitted INT8 output.
3. Write `0x80000000 | (gelu_scale << 30) | (M << 10) | K` to `0x0c00`.
   The controller supports 1–576 channels within the available canvas depth.
   Token count must be positive and at most the configured canvas width.
   The padded footprint per bank, `ceil(K / 16) * row_pitch_bytes`, must fit
   in 4096 bytes on the current board.
4. Poll `0x0c00` until done or error. The Avalon slave stalls writes and
   canvas reads during the pass. On done, read the same canvas offsets using the
   usual padded row-major layout. The output scale is
   `0.014901076910004555`. After a reset during normalization, rerun matmul
   for the canvas before issuing LayerNorm again.

Set `gelu_scale=1` for the supplied GELU LUT, whose signed INT8 output scale is
`1/32`. The controller converts those codes to the core input scale of
`0.14372939763106699` before computing statistics. Set `gelu_scale=0` when
the producer already emits codes at the LayerNorm input scale. The calibrated
gamma and beta scales are `0.010405473821745144` and `0.012730173238619106`.
All zero points are zero. See [core RTL](README.md) for the fixed arithmetic.

## Build and verification

Regenerate `adder_prototype.qsys` synthesis HDL in Platform Designer using
the project's Quartus 23.1 version, then build the root Quartus project. The
accelerator PLL is configured for 75 MHz. The component fileset includes
both LayerNorm modules and the ROM initialization files. Run Quartus from
the repository root so the ROM paths resolve.

From the repository root, with Python/NumPy and ModelSim's Intel FPGA
simulation libraries available:

```powershell
python layernorm/rtl/generate_canvas_test.py
vsim -c -do layernorm/rtl/run_canvas_modelsim.do
```

The test performs matmul and bias, runs with and without GELU, checks the
pre-normalization canvas, then compares the normalized canvas against the
Python fixed-point reference. Its 17 channels and 17 tokens cross a bank
group and a 128-bit word boundary. It also checks canvas padding, invalid
commands, busy and done status, and a read burst held pending during LayerNorm.
