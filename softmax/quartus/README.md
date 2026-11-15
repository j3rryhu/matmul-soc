# Signed INT8 standalone Cyclone V benchmark

Quartus Prime Lite 18.1 completed synthesis, placement, routing and four-corner
TimeQuest analysis for device `5CSEMA5F31C6`, seed 1, two processors. The target
period remains 11.111 ns, approximately 90 MHz. No existing SoC/Quartus project
files or clock requests were changed.

The current core has signed INT8 inputs and signed INT8 probability outputs
0..127. Exponentials/ROM remain 16-bit internal values. The benchmark wrapper is
updated to the same byte interface; probability MSB is correctly constant zero.

| Routed result | Previous 16-bit iterative core | Current INT8 core |
| --- | --- | --- |
| ALMs | 222 | 175 |
| Registers | 264 | 192 |
| Memory bits | 71,808 | 8,800 |
| M10K blocks | 10 | 3 |
| DSP blocks | 0 | 0 |
| Worst setup slack, all four corners | +3.188 ns | +3.894 ns |
| Worst hold slack | +0.129 ns | +0.113 ns |
| Worst pulse-width slack | +4.344 ns | +4.344 ns |
| Lowest slow-corner Fmax | 126.21 MHz | 138.56 MHz |
| Meets approximately 90 MHz constraints | Yes | Yes |
| Unstalled output interval | 35 clocks | 27 clocks |

All unconstrained clock/input/output port/path counts are zero. Current numeric
error relative to the floating model is higher than the old wider boundary;
this is a storage/precision change, not an equivalent floating-point algorithm.
Current and old resource/timing metadata are in `metrics.json` and
`baseline_metrics.json`. Raw reports are in the ignored build directories.

The exponential table is now 256x16 rather than a padded 4096x16 ROM. Scores use
196x8 bits and exponentials 196x16 bits, each in a separate M10K. The 23-cycle
restoring divider uses shift/compare/subtract and a separate ties-to-even rounding
state; no reciprocal table or combinational `/` is used. Multiplication by 127
uses `(e<<7)-e`. Weight reads are synchronous to infer block memory.

## Reproduce

From repository root:

```powershell
& 'C:/intelFPGA_lite/18.1/quartus/bin64/quartus_sh.exe' -t softmax/quartus/build.tcl
vsim -c -do softmax/rtl/run_modelsim.do
```

Default build output is `build_int8/`; an optional build-directory argument is
accepted. The script snapshots the RTL and ROM, runs map/fit/STA and rejects any
negative reported slack. Current build exits successfully. It does not run the
assembler, generate a board-programming bitstream or program a device.

`softmax_benchmark.sv` has registered virtual I/O to measure register paths.
It is a timing harness, not a protocol-preserving SoC wrapper. The clock is
auto-placed; virtual data ports have zero min/max I/O delay budgets. No board
pinout, PLL/reset integration, physical I/O timing or integrated-SoC timing is
validated. Placement with the other accelerators can change resource/timing.

Current synthesis has five warnings: three inferred ROM write-side defaults,
and two stuck-output messages for the intentionally zero probability sign bit.
Fitting has three warnings for the Lite LogicLock feature and incomplete/unspecified
clock I/O placement. TimeQuest has zero warnings and no violations. The complete
flow has zero errors and eight warnings. These are standalone results only.
