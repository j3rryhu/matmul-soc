# Standalone signed INT8 softmax RTL

`softmax_row.sv` accepts signed INT8 scores and emits signed INT8 probability
codes 0..127, zero point zero. Decode an output as `code/127`. Input real scale
is 0.11335523860661063 for the supplied calibrated ROM. This replaces the former
16-bit input/16-bit output interface, so any external wrapper must update widths
and scale interpretation. The repository's benchmark and testbenches are updated.
The block is standalone; no matmul/GELU/LayerNorm/SoC connection is added.

## Interface and algorithm

- `reset` is synchronous active-high and discards a partial row/pending output.
- A channel is accepted at a rising edge with `in_valid && in_ready`.
- `row_length[7:0]` is sampled on the first accepted score; legal values 1..196.
  Later changes do not change that row. An invalid first length blocks input.
- Hold input data/valid until accepted. Exactly the latched length completes input.
- One row is buffered. Input ready is low during computation/output.
- Output transfers on `out_valid && out_ready`. Data, valid and `out_last` remain
  stable while stalled. `out_last` marks the final score's output.
- The next row starts after the previous final output is accepted. Memory is not
  reset; control ensures only the current complete row is used.

The core buffers scores, finds their maximum, and replays them through a 256x16
exponential ROM, accumulating a 24-bit sum. Differences widen to signed 9 bits,
covering 0..255. The internal exponential memory also has 16-bit entries; external
input/output storage remains INT8. Memories use synchronous reads.

Each numerator is `e*127`, implemented as shift/subtract, fitting unsigned 23 bits.
A shared restoring divider processes one bit per clock for **23 clocks** using
shifts, comparisons and subtraction, with a 25-bit remainder. It is not a
reciprocal lookup table or a combinational `/`. A separate state compares twice
the remainder with the sum and applies ties-to-even rounding. Output is bounded
by 127; its sign bit is zero. The exponential LUT approximates exp only.

For each output there is a weight-read state, normalization-load state, 23 divide
states, a rounding state and an output-transfer state. With no stalls, output
handshakes are **27 clocks apart**. First valid asserts `3*N+26` clocks after
the final input handshake. The directed test measures these timings. This is
a serialized correctness baseline, not a one-result-per-clock pipeline.

## Simulation and vectors

From repository root:

```powershell
vsim -c -do softmax/rtl/run_modelsim.do
```

The script uses the separate `softmax/rtl/work` library without changing library
mappings. It runs directed tests and the full golden-vector test; compile/runtime
failure or a missing pass flag produces nonzero exit status. It does not program
hardware. Supplied ROM/vectors allow simulation without Python/model assets.

`example/manifest.json` specifies calibrated scales, widths, hashes and counts.
The fixture header contains row/output counts; each row contains a length followed
by signed decimal `score_code expected_probability_code` pairs. Regenerate with:

```powershell
& '../.venv/Scripts/python.exe' softmax/generate_rtl_vectors.py
```

ModelSim passed **578 rows / 25,999 output codes** bit-for-bit in 897,599 main
testbench clocks. Coverage includes every difference 0..255, every row length
1..196, signed extrema, constant/one-hot-like/random rows, 100 real score rows,
input gaps, output stalls/stability, `out_last`, invalid lengths and reset
recovery. Directed tests additionally check equal rows of lengths 1/2/49/196,
the 63.5 -> 64 ties-to-even case and latency/interval. An additional 26-key
golden row uses exact exponential weights with total `254*318`, forcing 24
probabilities to exactly 0.5 output codes; each rounds to even zero, which
distinguishes the new rule from the former half-up implementation.
Compile/simulation reported
zero errors/warnings. The 32-bit legacy simulator needs execution outside the
sandbox on this Windows installation due to DLL relocation at sandbox startup.

Independent rounding can substantially change probability sums, especially for
196-element uniform rows; see the top-level README and RESULTS. Passing RTL
equivalence does not establish acceptable floating-model error or accuracy.
Quartus resources/timing and physical board testing are separate validations.
