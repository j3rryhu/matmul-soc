# Standalone DE1-SoC device check

Quartus Prime Lite 18.1 completed synthesis, fitting and multicorner timing for
the existing LayerNorm RTL on `5CSEMA5F31C6` (the device selected by the repository
DE1-SoC project), seed 1, two processors. No SoC project files were edited.

The target is 11.111 ns (approximately 90 MHz), matching the current matmul
clock request. The optimized standalone implementation **meets 90 MHz** across
all four analyzed voltage/temperature models. The SDC constraints, device,
wrapper and seed are unchanged from the baseline; no timing exceptions were added.

| Routed result | Baseline | Optimized |
| --- | --- | --- |
| Synthesis / fitting | Successful | Successful |
| Logic utilization (32,070 available) | 540 ALMs | 510 ALMs |
| Registers | 488 | 639 |
| Memory bits | 22,016 | 22,016 |
| M10K blocks (397 available) | 4 | 4 |
| DSP blocks (87 available) | 9 | 8 |
| Worst setup slack across four corners | -0.686 ns | +0.321 ns |
| Worst hold slack across four corners | +0.121 ns | +0.116 ns |
| Worst minimum pulse-width slack | +4.338 ns | +4.337 ns |
| Lowest slow-corner Fmax | 84.77 MHz | 92.68 MHz |
| Unconstrained clocks/input/output ports and paths | All zero | All zero |
| Meets approximately 90 MHz constraints | No | Yes |

These results establish synthesis/device fit, not successful board execution.
The resource counts include the registered benchmark wrapper;
integration alongside existing accelerators has not been fitted or tested.

## What changed and verification

The baseline's worst path was coefficient RAM read through a wide gamma
multiplier to `norm_gamma` (10.513 ns data delay, -0.686 ns slack). Coefficient
registers now capture during the existing read-wait slot, isolating the RAM path.
The proven normalized-code bound fits signed 21 bits instead of 32; multiplying
by signed INT8 gamma therefore requires 29 bits instead of 40.

The following affine path was also split. A signed 29-bit gamma product is
decomposed into signed high 13 bits and unsigned low 16 bits. Independent small
constant multiplications finish at registers, followed by separate recombination,
beta-add and magnitude stages. Affine storage is 46 bits, above its 44-bit bound.
The transformation is exact; scales, square-root/division precision, rounding
and INT8 saturation are unchanged.

All 200,296 golden outputs on 744 rows matched bit-for-bit after the change,
including all lengths 1..576. Reset recovery additionally covers the new scale
multiply and output-magnitude stages. Directed +/-24.5-code ties still round to
the even codes +/-24. ModelSim reported zero errors/warnings. The tradeoff is
three more clocks per output: first-valid latency 75 -> 78 clocks, subsequent
unstalled interval 49 -> 52 clocks. This improves clock feasibility, not a claim
of increased throughput at the same clock or end-to-end inference speedup.

The core implements a 41-cycle restoring divider from shifts, comparisons and
subtractions. It does not use a reciprocal/division lookup table or an inferred
combinational `/` operator. A separate 25-cycle shift/compare/subtract algorithm
computes integer square root once per row. The epsilon ROM is a channel-count
constant table, not a division table. FPGA ALMs/LUT logic still implement the
arithmetic/control gates; that is different from an algorithmic lookup table.

## Reproduce

From repository root:

```powershell
& 'C:/intelFPGA_lite/18.1/quartus/bin64/quartus_sh.exe' -t layernorm/quartus/build.tcl
```

The default output is `build_timing2/` (ignored). An optional argument selects
another build directory. The script snapshots RTL and epsilon ROM, runs map,
fit and STA, and deliberately returns failure for negative slack. The current
optimized run exits successfully. `metrics.json` retains the observed counts,
timing and hashes; `baseline_metrics.json` retains the original failing results.
The original local build is `build_current/`.

To inspect detailed critical paths of a completed build:

```powershell
& 'C:/intelFPGA_lite/18.1/quartus/bin64/quartus_sta.exe' -t layernorm/quartus/report_paths.tcl
```

An optional build-directory argument selects a different completed build.

The wrapper has registered virtual data I/O and an auto-placed clock. It measures
register paths; it is not a protocol-preserving integration wrapper. Virtual
ports have zero min/max I/O delay budgets. There is no board pinout, PLL, reset
integration, assembler/programming step, downloadable bitstream or physical I/O
timing sign-off. No device was programmed.

Optimized synthesis reported three inferred ROM write-side nets defaulted to
zero. Fitting reported the Lite
LogicLock feature warning plus incomplete/unspecified clock I/O placement.
TimeQuest reported zero warnings and no timing violations. The complete flow
reported six warnings, all described above. These standalone results are not
board I/O timing or integrated-SoC sign-off.
