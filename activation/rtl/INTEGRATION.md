# Matmul readout integration

The current SoC uses main's GELU in `matmul_accel/output_loader.v`.
Its bias buffer stores signed INT32 values in accumulator units and
`ACT_CFG` at `0x0090` configures bias and GELU. For current HPS usage see
`test_program/new_tiling_test/test4_bias_gelu.c`. The readout design and
register map below describe the earlier implementation. The combined
Avalon regression runs through `softmax/rtl/run_soc_modelsim.do`.

Row mapping and bias read are shared by all bytes of a wide word. Row mapping
uses parallel comparisons against constant multiples of the row length in
words, replacing the runtime divider. The selected row index is registered
alongside the output RAM read, so mapping and bias/GELU occupy separate clock
paths without adding a read cycle. Addition is explicitly signed 9-bit. Enabled processing requires M
to be a multiple of DATA_WIDTH/8 (16 for the current 128-bit interface), so a
word cannot straddle output-row segments. Unsupported geometry bypasses the
activation; software must enforce the restriction. The existing M=32 test meets
it. Actual fit/timing improvement must be measured by recompiling Quartus.

New logic is isolated in `matmul_gelu_readout.v`. Existing edits are limited to
one instance and the output read mux in `accel_top.sv`, plus two component
fileset entries in `matmul_accel_hw.tcl`. The output loader and RAM are unchanged.

Why readout: the output RAM stores partial-block sums as well as final results.
Applying nonlinear GELU on intermediate writes would corrupt later accumulation.
The host must wait for matmul done, then read the final output. The same output
addresses now return bias/GELU when enabled, raw matmul bytes when disabled.
No additional read cycle is added. Processing is combinational across every
byte of the wide RAM read; registered row tracking matches RAM latency.
Full-system Quartus timing/resource checks are still required (bias selection,
parallel ROM lookups, and Avalon return routing remain combinational). This is not a streaming
connection to the next Linear layer and does not write activated values into RAM.

## New byte-addressed host writes

All offsets are relative to accelerator base `0xC0000000`, without --reg16 scaling.

| Offset | Write behavior |
|---|---|
| 0x0800..0x08ff | 256 signed int8 biases, indexed by output matrix row |
| 0x0900 | Bit 0 enables bias + GELU; zero bypasses both |

Reset disables processing and clears biases. Configuration writes are ignored
while matmul busy. These registers are write-only; reads are not implemented.
Do not change dimensions, biases or enable state during output reads. Bias is
constant across each output row in the existing W[K,N] @ X[N,M] layout.
Upstream software must transpose/map its model tensors to that layout.

Use 32-bit aligned host writes (Avalon byteenable selects the correct wide lane):

```c
/* accel is your existing mmap pointer; bias contains K same-scale int8 values. */
*(volatile uint32_t *)(accel + 0x0900) = 0; /* disable, while idle */
for (int row = 0; row < K; row += 4) {
    uint32_t packed = 0;
    for (int j = 0; j < 4 && row+j < K; j++)
        packed |= (uint32_t)(uint8_t)bias[row+j] << (8*j);
    *(volatile uint32_t *)(accel + 0x0800 + row) = packed;
}
__sync_synchronize();
*(volatile uint32_t *)(accel + 0x0900) = 1;
__sync_synchronize();
/* Configure/start matmul as usual, wait for done, then read OBUF. */
```

Default LUT input/output scales are both 1/32. Bias and matmul bytes MUST have
that input scale for this table. These remain demonstration scales, not selected
TinyViT deployment parameters. A different table requires rebuilding the FPGA.

Expected host result per element:
`table[clip(matmul_golden[k,m] + bias[k], -128, 127) & 255]`.
The old identity test expects raw matmul, so it must run with activation disabled.
For activation testing first read raw output with enable=0, then enable=1 while
idle and reread identical addresses; compare every byte against this reference.
Disable again to verify the original raw data has not changed.

## Build and verification

Regenerate Platform Designer synthesis HDL after refreshing the component.
Check that the generated fileset includes the new Verilog and LUT HEX and that
the ROM initialization path resolves. The current root-relative LUT path assumes
Quartus is launched from the project root. Compile Quartus and inspect timing
before programming. Existing generated Qsys HDL was not regenerated here.

Run `vsim -c -do activation/rtl/run_modelsim.do` from the project root.
Tests cover standalone GELU, all 65,536 int8 bias pairs, and 1,024 readout bytes
over both 16-row segments with distinct biases, byte-enabled writes, bypass and
reset. An additional 539,649 wide-word checks cover every aligned 16-bit column
count and all 128 output word addresses in nonsequential order, plus invalid
geometry bypass. Expected row selection uses an independent division reference.
Large column counts test mapper arithmetic only; they do not establish that
the complete matmul supports those dimensions. The readout test models synchronous RAM; it is not a full Avalon/HPS or
full matmul simulation. Board and full-system verification remain necessary.

## Merged main memory map

The integrated bias window moved to 0x0800..0x08ff and enable to 0x0900
because main expanded IBUF to 0x5000..0x8fff. The top level translates these
addresses to the standalone readout module's unchanged 0x8000/0x8100 interface.
Do not use the old integrated offsets: they now write input data.

OBUF now starts at 0x9000 and exposes a contiguous row-major canvas. GELU
uses the remapped bank address aligned with the synchronous RAM response,
and uses TILED_MAT_DIM (or INPUT_COLS when zero) for row-bias selection.
Enabled GELU requires that canvas width be a multiple of 16, and supports
biases for canvas rows 0..255. Host software must use main's new register map
and row-major input/weight layout. matmul_gelu_test.c follows this map.
The merged design requires fresh Platform Designer generation and Quartus
compilation; previous timing results do not validate this larger design.

Run `vsim -c -do activation/rtl/run_merge_modelsim.do` for the merged-path
regression. On 2026-11-14 it passed with zero errors or warnings: 3,072 bytes
across 192 Avalon burst beats through real output RAM and address remapping,
checking raw output, per-row bias/GELU, bypass restoration, canvas width
selection, and isolation from writes to the former enable address in IBUF.
It seeds RAM through its write ports; it does not run the matmul datapath.
The imported PE enable register was placed inside a generate-if to avoid an
invalid ROW[-1] reference when ModelSim elaborates the first PE row.
