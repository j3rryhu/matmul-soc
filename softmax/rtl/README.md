# Row softmax RTL baseline

`softmax_row.sv` implements the selected fixed-point contract in the parent
README. It buffers one row of 1..196 signed 16-bit score codes, computes the
row maximum, reads a 3073 x 16-bit exponential ROM, sums exponential codes
in 24 bits, and normalizes to unsigned 16-bit probability codes.

## Interface

- Rising-edge clock, synchronous active-high `reset`. Reset aborts any active
  row; the producer must resend that row from its first element afterwards.
- Input transfer: `in_valid && in_ready`. The source holds data stable until
  accepted. `in_score` has scale `1/256` (real score = code/256).
- `row_length` is sampled on the first accepted element only. Legal lengths
  are 1..196. An invalid first length keeps `in_ready` low. Later values of
  `row_length` do not affect the active row.
- Output transfer: `out_valid && out_ready`. `out_probability` represents
  real probability `code/65535`. `out_last` marks the final element.
- Output valid, probability and last remain stable during backpressure.
- The next row can start after the final output transfer. Rows do not overlap.

The input must already include the complete QK contraction, attention scale
and attention bias, quantized to this score format. This module does not
perform those operations or connect to the existing int8 matmul buffer.

## Architecture and limits

The state machine receives N elements, performs three cycles per score
read/exponential lookup/store, then performs integer division one bit per
cycle (33 cycles). Without output stalls, outputs transfer every 35 cycles.
The first output is available 3N+34 cycles after the final input transfer,
and can transfer on the following edge. Input stalls and output backpressure
extend elapsed time. A continuously supplied row takes 39N clock periods
from its first input to the following row's first input, with no stalls.

Differences are widened to 17 bits before ROM indexing. Differences greater
than 3072 produce zero. The maximum score always produces exponential code
65535, so a completed row has a nonzero denominator. The 32-bit product and
33-bit rounded numerator preserve integer Python normalization exactly.

The original combinational `/` operator has been replaced by an exact
33-cycle restoring unsigned divider. Synchronous score and ROM reads permit
block-memory inference. This trades output throughput for a shorter critical
path and less logic; no probability approximation was introduced.
The LUT path defaults to `softmax/example/exp_lut.hex`; tools must run from
the repository root or override `LUT_FILE`.

## Reproduce verification

The generated ROM and vectors are included in `softmax/example/`, so ModelSim
simulation does not require PyTorch, a checkpoint or the capture artifacts:

```powershell
vsim -c -do softmax/rtl/run_modelsim.do
```

To regenerate vectors after the Python capture and fixed-point evaluation:

```powershell
& '../.venv/Scripts/python.exe' softmax/generate_rtl_vectors.py
```

On 2026-11-02, ModelSim Intel FPGA Edition 10.5b passed 147 rows / 8666
elements for both the original and revised implementations, with zero
compiler/simulator errors or warnings.
The Python-generated expected codes matched RTL bit-for-bit.

Coverage includes 100 real attention rows (10 per image), 47 synthetic rows,
lengths 1/2/49/196, all-equal scores, signed extrema, dominant maxima, LUT
boundary differences, deterministic random rows, input gaps, output stalls,
output stability and last markers, invalid lengths, reset during partial
reception, active division, and a stalled output. Rows run consecutively without
reset between test vectors. This is functional simulation, not FPGA validation.

The ROM file has 4096 physical words: 3073 useful LUT entries followed by
zero padding. Valid differences beyond the useful range select address 3073
and output zero. Python's numerical LUT retains 3073 entries.

See `../quartus/README.md` for the standalone Cyclone V resource/timing build.
