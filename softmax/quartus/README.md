# Standalone Cyclone V resource and timing benchmark

On 2026-11-02, Quartus Prime Lite 18.1 completed synthesis, placement, routing
and multicorner TimeQuest analysis for device `5CSEMA5F31C6`, seed 1, two
compilation processors. Target period is 11.111 ns (approximately 90 MHz).

The current working `adder_prototype.qsys` requests 90 MHz on `pll_0.outclk0`
and connects that output to `matmul_accel_0.clock`. The older exported
`adder_prototype.out.sdc` describes a 100 MHz generated clock. The standalone
benchmark uses the current Qsys request; the integrated project's clock/SDC
consistency must be resolved during integration. No original project files
were modified by this benchmark.

## Routed results

| Metric | Original combinational divider | Revised iterative divider |
| --- | ---: | ---: |
| ALMs (including benchmark wrapper) | 2665 | 222 |
| Registers | 3343 | 264 |
| Block memory bits | 3136 | 71808 |
| M10K blocks | 1 | 10 |
| DSP blocks | 0 | 0 |
| Worst setup slack, all four corners | -69.583 ns | +3.188 ns |
| Worst hold slack, all four corners | +0.171 ns | +0.129 ns |
| Worst minimum-pulse-width slack | +4.345 ns | +4.344 ns |
| Lowest reported slow-corner Fmax | 12.39 MHz | 126.21 MHz |
| Meets approximately 90 MHz constraints | No | Yes |
| Output interval without stalls | 2 cycles | 35 cycles |

The revised design reduces ALMs by approximately 91.7%. It uses exact
33-cycle restoring division, and synchronous reads infer the score buffer,
exponential buffer and exponential ROM as block memories. The ROM is padded
from 3073 useful words to 4096 physical words. Numerical results are unchanged.
All setup/hold unconstrained clock and input/output counts are zero in both
TimeQuest reports. These are standalone results, not integrated-SoC sign-off.

`softmax_benchmark.sv` adds registered virtual inputs and outputs to retain
the core and measure register-to-register paths. It is a timing harness,
not a protocol-preserving integration wrapper. The clock is auto-placed;
virtual data ports have zero min/max I/O delay budgets. No pinout or bitstream
for programming the board is provided. Actual integrated placement, clock
distribution and interface paths may change timing.

The successful revised run had four synthesis warnings (three unused ROM
write-side inference nets tied to zero, and inferred RAM read-during-write
pass-through logic), plus fitter warnings for the Lite LogicLock feature and
incomplete clock I/O/pin assignments. TimeQuest had zero warnings and no
timing violations. These do not establish board I/O timing.

## Reproduce

From the matmul-soc repository root:

```powershell
& 'C:/intelFPGA_lite/18.1/quartus/bin64/quartus_sh.exe' -t softmax/quartus/build.tcl
vsim -c -do softmax/rtl/run_modelsim.do
```

The default build directory is `softmax/quartus/build_current/`. An optional
second argument selects another build directory. The script copies the ROM
and core RTL snapshot into that directory, runs map, fit and sta, and rejects
negative timing slack. No assembler or FPGA programmer is invoked.

Recorded run directories (Git-ignored):

- `build/`: original implementation; `softmax_row_snapshot.sv` preserves its
  original source. TimeQuest analyzed its fitted netlist after detecting that
  the live source had changed; that netlist was not rebuilt during STA.
- `build_iterative/`: revised implementation. Its core source snapshot matches
  the current RTL. Quartus automatically inferred all three block memories.

Each directory contains `output_files/softmax_benchmark.fit.summary`,
`.fit.rpt`, `.sta.summary` and `.sta.rpt`. `metrics.json` records extracted
metrics and hashes of the underlying reports/source snapshots.

Functional regression after the changes passed 147 rows / 8666 outputs in
368721 testbench cycles, with zero ModelSim errors/warnings. The extra reset
test aborts active division; reception and stalled-output resets are also
checked. Row lengths 1/2/49/196 and signed/LUT boundaries remain covered.

The revised core's stall-free row initiation interval is 39N cycles, about
21.23 microseconds for N=49 or 84.93 microseconds for N=196 at 90 MHz.
Next work is accelerator interface integration and a full-SoC timing check;
throughput can be improved later with a pipelined divider or a row reciprocal.
