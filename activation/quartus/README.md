# GELU readout timing benchmark

From the repository root:

```powershell
& 'C:/intelFPGA_lite/18.1/quartus/bin64/quartus_sh.exe' -t activation/quartus/build.tcl
vsim -c -do activation/rtl/run_modelsim.do
```

The build runs synthesis, fit and multicorner TimeQuest on Cyclone V
`5CSEMA5F31C6`, seed 1, two processors, with an 11.111 ns target (90 MHz).
It snapshots the core and LUT into the ignored `build_current/` directory.
An optional first argument selects another build directory; a second selects
another core RTL file for comparison with the same harness and constraints.
Negative reported slack makes the build script fail, even if Quartus itself
returns success. The build never assembles or programs a bitstream.

`readout_benchmark.sv` registers virtual inputs and outputs to measure paths
through the core. It is a timing harness, not a protocol-preserving wrapper.
Virtual ports have zero min/max I/O delays; the clock pin is automatically
placed. These measurements do not establish board I/O timing or full-SoC
timing closure. Refresh Platform Designer RTL and rebuild the full project
before using this change on the board.

The registered output harness also allows Quartus to infer the 16 LUTs as
M10K ROMs. The integrated Avalon path may infer different hardware, so these
figures must not be treated as integrated resource or timing guarantees.

On 2026-11-03 the revised core passed all four timing corners with worst setup
slack +1.925 ns, hold slack +0.172 ns, and minimum pulse-width slack +4.373 ns.
Lowest reported slow-corner Fmax was 108.86 MHz. The harness used 957 ALMs,
1,493 registers, 16 M10Ks and no DSPs. All setup/hold unconstrained clock and
input/output counts were zero. TimeQuest reported zero errors and warnings.
Synthesis reported two integer-width truncation warnings (bounded row-index
arithmetic); fitter reported Lite LogicLock and unassigned clock-pin warnings.

The original core at commit `ff58737` was also fitted with this identical
harness, seed and constraints. Its build script correctly failed on negative
slack. Recorded results and source/report SHA-256 hashes are in `metrics.json`.
Portable copies of both runs' timing and fitter summaries are in `results/`.
Their bytes match the hashes in `metrics.json`; full generated build databases
and detailed reports remain excluded and can be regenerated with the script.

| Metric | Original divider | Registered boundary comparisons |
| --- | ---: | ---: |
| Worst setup slack (four corners) | -8.769 ns | +1.925 ns |
| ALMs, including harness | 1,707 | 957 |
| Registers, including harness | 2,509 | 1,493 |

The core retains its 16-lane throughput and existing RAM response latency.
The reduced register count includes synthesis removing unreachable bias rows
for the benchmark's fixed 16-bank, 128-byte-bank, 128-bit geometry; the source
still provides the same 256-byte configuration window.

The address-mapping regression covers every aligned 16-bit column count at
all 128 wide-word output addresses, invalid-geometry bypass, signed row biases,
saturation, enable/disable and reset. Large column counts exercise mapper
arithmetic only, not the full matmul's supported dimensions. The mapper
register advances with the existing RAM response, preserving read latency.
