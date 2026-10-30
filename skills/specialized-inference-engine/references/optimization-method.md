# Optimization method

Purpose: how to find where the time goes, decide what to try next, and record what each attempt bought. It covers
the dispatch map, floor arithmetic, classifying time, ranking opportunities, writing a mechanism down before
editing, committing and recording results, running explorations in parallel, and when to stop tuning locally.
Read it at the start of phases 3 and 4 of `workflow.md`, before choosing the next optimization, and when you write
up a result. It assumes the measurement rules of `measurement.md` and the bitwise rule of `correctness.md`.

Terms: a dispatch is one kernel launch; a threadgroup is a group of threads that share on-chip memory (a thread
block in CUDA); a GEMV is a matrix-vector product, the shape of every projection in batch-1 decode.

## The dispatch map

A dispatch map lists every GPU dispatch of one decode step (or of one prefill chunk) in order. It is the single
most useful document for optimization: it shows the fixed costs, the bytes and the dependencies at a glance.

| # | stage | kernel | per token | grid x threadgroup | bytes read | depends on | in-chain us | floor us | excess |
|---|---|---|---|---|---|---|---|---|---|
| 1 | start | embedding gather | 1 | 16 x 32 | 1 row | previous token | | | |
| 2 | layer 0 | input norm | 1 | 1 x 256 | 8 KB | 1 | | | |
| ... | | | | | | | | | |

How to build one:
1. Read the decode step's code top to bottom and write down each dispatch with its kernel, grid and threadgroup
   size, and how many times it runs per token (per layer type, times the number of such layers).
2. Fill "bytes read" from the model card's tensor table (weights at their stored size, including scales and
   biases) plus activations, caches and state at the context you study.
3. Fill "depends on": which earlier dispatch wrote the buffers it reads. Mark the pairs that are independent; in a
   decode step there are usually very few.
4. Count the dispatches with a counter in the engine and check the total against your count.
5. Add the measured times (next section), the floors and the excess.

`examples/lfm2.5/decode-map.md` is a complete map of a 40-layer MoE decode step.

## Put measured times on the map

- Measure each dispatch's GPU time **inside the real engine**, with real weights, real routing and real state.
  Isolated micro-benchmarks are misleadingly fast when their working set stays in an on-chip cache. Example (LFM2.5
  on an M5 Max, measured): an expert down-projection took 12.6 us in a micro-benchmark that cycled 57 MB of weights
  (largely cache-resident) and 15.7 us in the chain, where every weight comes from DRAM.
- Per-dispatch timestamps inflate each interval (`measurement.md`). Two cleaner methods: the **marginal cost by
  duplication** (add one extra copy of a dispatch class that reads another layer's weights into dummy outputs;
  the step-time increase divided by the extra dispatches is that class's in-chain cost), and the **empty-kernel
  probe** (an empty kernel of the same grid after every dispatch gives the fixed per-dispatch cost).
- Check that the per-kernel costs add up to the uninstrumented step time within about 1%. If they do not, the map
  is missing something (host gaps, startup, a mis-measured class).
- Platform mechanics (counters, timestamp sampling, its artifacts): see `backends/metal-rust/profiling.md`.

## Floor arithmetic

- Measure the **streaming bandwidth** with a plain read kernel over more data than the caches hold. Use the
  measured value, never the nominal one.
- **Decode floor per token** = bytes read per token / measured bandwidth. Per dispatch: floor = bytes / bandwidth +
  the measured fixed cost of an empty dispatch of that grid.
- **Prefill floor** = FLOPs per token x tokens / measured matrix throughput at your dtype, plus the attention FLOPs,
  which grow with context.
- **Excess** = in-chain time - floor. Sum it by category; that sum is the room you have.

Example (LFM2.5 on an M5 Max, decode at a 256-token context): about 1.46 GB per token (estimate, from the tensor
table) at a plain stream of about 590 GB/s (measured) is 2.48 ms. Adding the measured 1.95 us of an empty
2048-threadgroup dispatch to each of the 262 dispatches, the per-dispatch floors summed to 3006 us against 3280 us
measured.

## Classify the time

Put every microsecond of the step in one of four classes; each has different remedies.

1. **Streaming near the ceiling** (roughly 90% of the measured bandwidth or more). Only fewer bytes or fewer
   dispatches help, and the bytes are fixed when the weights are. Leave these kernels alone.
2. **Streaming below the ceiling.** Usual causes: a chain of dependent loads before the first weight address,
   loads issued late (after a barrier or a prologue), runtime shape arithmetic in the address computation, too few
   threadgroups to fill the machine, a dispatch too small for its ramp-up and drain. Bitwise-safe remedies: issue
   loads earlier, compile-time shapes, staging in threadgroup memory, launch order. Not bitwise-safe: changing
   lanes per row, splitting K differently, reassigning blocks to lanes. Example (LFM2.5, measured): compile-time K,
   group size and row strides in every decode GEMV, with the same lanes and order, gave +2.7-3.0% decode.
3. **Fixed per-dispatch cost.** Small kernels (norms, routing, top-k, argmax, position encoding) cost one to a few
   microseconds each for almost no work (LFM2.5, measured: an extra empty dispatch cost 0.86 us at one threadgroup
   and 1.95 us at 2048 threadgroups). The remedy is fusion into a neighbour's prologue or epilogue when the
   dependency allows it. A grid-wide dependency (the next step needs every threadgroup's result) needs
   synchronization whose price depends on the grid: measure it before you build on it.
4. **Host and submission.** GPU idle between command batches, the host waiting inside the token loop, request
   startup. Measure GPU busy time over the request; at about 99% busy little is left here.

Example (LFM2.5, measured, us per token at 256 tokens; in-chain time / excess over the floor): expert GEMVs 1641 /
+124, convolution projections 432 / +12, head and token tail 368 / +3, fused router 225 / +116, output projections
217 / -22, post-FFN norms 86 / +3; the GPU was busy 99.3% of the decode request.

## Rank the opportunities

For each candidate write: the estimated gain in us per token (saving per dispatch x dispatches per token), its
upper bound, your confidence, the risk and the cost. Rank by gain x confidence / risk and start with cheap,
high-confidence items.

- **Upper bounds** keep estimates honest: the excess over the floor; cold minus hot (the same kernel with its data
  already in cache); a "without math" probe (the kernel with its arithmetic removed: what remains is memory and
  launch). An estimate without an upper bound is a guess.
- **Risk** is mostly bitwise risk (does the change touch an expression, an order or a rounding point?) plus code
  shared with prefill and grid-wide synchronization.
- Expect estimates to be optimistic. Example (LFM2.5, measured): staging the router's weight row at kernel entry
  had a 116 us/token excess to attack and was estimated at 30-60 us/token; a micro-benchmark showed -1.25 us per
  layer (about 47 us/token); end to end it measured +0.42-0.47% (about 15 us/token). Record all three.

## Write the mechanism down before editing

In the engine's optimization notes, before the code:
- **Hypothesis**: what is slow and why, with the measured evidence.
- **Call site and parent behaviour**: the exact dispatch or code location, and whether the parent already does it.
- **Mechanism or calibration**: a structural change (different work, fewer dispatches, different submission) or a
  parameter change within an existing mechanism (a chunk size, a threadgroup count). Calibration alone does not
  explain a speedup.
- **Expected effect**: which work disappears or moves, the estimate per token, which segments and lengths.
- **Discriminating observation**: the smallest measurement that shows the mechanism works (dispatch count,
  per-kernel in-chain time, GPU idle time), separate from the end-to-end timing.
- **Falsifier**: the result that would show the hypothesis wrong.
- **Alternatives**: the other ways to the same effect, and why this one first.

## One mechanism per commit

Each commit holds one mechanism, so that its effect is attributable and it can be reverted alone. The message says:
- **what** changed and where, and what stays identical (expressions, orders, rounding points);
- **verification**: the bitwise check against the parent (prompt count, request shapes, result), any kernel
  harness or stress test, and the positive control if you wrote one;
- **measured effect**: tool, rounds, per-length ratios per segment, per-round spread, win counts, and the
  discriminating observation (for example dispatches removed per token);
- the estimate you made beforehand, labelled as such.

```
Fuse <operator> into <kernel> (-N dispatches/token)

<What the new code does, and what stays identical: expressions, rounding points, reduction orders.>

Verification: bitwise check vs parent <build>, <prompts> x <request shapes>: all identical; <harness or stress test>.
Measured: ab, <rounds> rounds: decode B speedup <r1> / <r2> / <r3> at <L1> / <L2> / <L3> (per-round sd <s>,
B faster in <k> of <n>); prefill unchanged within noise. Estimate beforehand: <x> us/token (estimate).
```

Instrumented, check-only and benchmark-only builds live on clearly named branches ("INSTRUMENTED BRANCH ONLY") and
are never merged. The full log of one engine's changes is in `examples/lfm2.5/`.

## Record neutral and negative results

Every attempt gets an entry with its numbers, the reason it failed and the condition under which it might be worth
revisiting; keep the patch. Search these entries (and `backends/<name>/counterexamples.md`) before trying something:
revisit a rejected mechanism only with a changed condition or new evidence. Examples (LFM2.5, measured):
- folding a norm into the last-finishing threadgroup of a 2048-threadgroup GEMV: identical tokens, decode 0.85-0.86
  of the parent, because the device-scope fence cost about 8 us per dispatch (the same pattern won 0.46 us per layer
  on a 64-threadgroup kernel);
- submitting the step's kernels for concurrent execution with an explicit barrier after each, instead of one serial
  stream: 2.3-2.7% slower;
- four query heads per threadgroup in decode attention: identical tokens, but decode at an 8192-token context went
  from 4.75 to 5.25 ms/token, because only 8 threadgroups remained;
- 32-bit row offsets in an expert GEMV: 3% faster in isolation, 0.997-0.999 end to end; rejected.

## Parallel explorations

Several explorers (agents or people) can work at once when the GPU is shared carefully:
- one work tree and branch per explorer and mechanism area, chosen so that areas do not overlap (for example GEMV
  internals, routing, the token tail, attention, profiling);
- one shared parent build that everyone compares against, a fresh build directory per build, log files prefixed
  with the explorer's name, nothing deleted;
- one GPU lock for every GPU process (`measurement.md`); explorers wait for it, never kill each other's processes;
- a profiling explorer whose branch is evidence only and never merged;
- merge in waves: merge the verified commits into a new base, rerun the bitwise check and the timing of the merged
  base against the old parent (separately measured gains do not add exactly), then start the next wave from the
  new base as everyone's new parent.

## When to stop local tuning

Stop tuning a kernel or an area when:
- the remaining estimates are below the noise floor, or their upper bounds are;
- the kernels are at the ceiling (a "without math" probe leaves a few percent at most);
- micro-benchmark wins stop showing up end to end;
- several attempts on the same mechanism were neutral.

Then look for a structural change: fewer dispatches by fusing across operator or layer boundaries, a different
decode loop (tokens resident on the GPU, steps encoded ahead, one wait per request), a load-time layout change that
keeps every dequantized value identical, a different launch order that shares cached data, or another segment or
length where the gap to the floor is larger (prefill, long-context attention). Example (LFM2.5, measured): once the
GEMVs streamed at about 590 GB/s, the remaining lever was the fixed cost of about 2 us per serial dispatch; fusions
took the decode step from about 355 to about 252 dispatches per token (262 in the profiled build, before the last
fusion).
