# Profiling on Metal

Purpose: how to find where GPU time goes inside the real engine on an M5-class GPU: per-dispatch timing that the
hardware allows, how much each method perturbs the run, how to check that the parts add up to the whole, why
isolated micro-benchmarks mislead, and how to compare each kernel with the streaming ceiling. This is the
platform half of phase 3 in `references/workflow.md`; timing A/B builds is `references/measurement.md`.

Numbers are measurements from the LFM2.5 worked engine on an Apple M5 Max unless they are marked as estimates; most
are from profiling its decode step (P = 256 and 4096, real weights, routing and state, every run token-identical to
its base).

## What to measure

- **Ground truth first:** the GPU time of each command buffer the engine already submits, with nothing inserted.
  `GPUStartTime` and `GPUEndTime` (seconds, read after completion) cost nothing and are the number every
  finer method must add up to.
- **Then per-dispatch time inside the real chain:** real weights, real routing, real cache and recurrent state, the
  real order of dependent dispatches. A kernel's cost depends on what ran before it (cache contents, ramp-up after
  the previous dispatch drains), so a dispatch measured anywhere else is a different measurement.
- **Then the check:** the per-dispatch costs summed over one step must match the ground-truth step time. Example
  (measured): 3272 us summed vs 3280 us measured at P = 256; 3483 vs 3518 at P = 4096. Until the sum matches, the
  table is not evidence.

```rust
// ground truth, after the command buffer completed (seconds -> us)
let gpu_us = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e6;
```

## Per-dispatch timestamps on M5

Probe what the GPU supports before designing the instrument. Example (M5 Max): counter
sampling is supported only `AtStageBoundary` (not at dispatch, draw, tile or blit boundaries); the `timestamp`
counter set has one counter (`GPUTimestamp`), in ns on the same time base as `GPUStartTime`, at 24 MHz (41.7 ns
resolution). So a timestamp between two dispatches of one encoder is impossible; the instrument is **one compute
encoder per dispatch, sampled at its start and end**.

```rust
// once: a shared timestamp sample buffer with two samples per dispatch
let d = MTLCounterSampleBufferDescriptor::new();
d.setCounterSet(Some(&timestamp_set));            // from device.counterSets(), name == "timestamp"
d.setStorageMode(MTLStorageMode::Shared);
unsafe { d.setSampleCount(2 * n_dispatches) };
let sb = device.newCounterSampleBufferWithDescriptor_error(&d)?;
// per dispatch i: its own encoder, sampled at encoder start and end
let desc = MTLComputePassDescriptor::computePassDescriptor();
let att = unsafe { desc.sampleBufferAttachments().objectAtIndexedSubscript(0) };
att.setSampleBuffer(Some(&sb));
unsafe { att.setStartOfEncoderSampleIndex(2 * i); att.setEndOfEncoderSampleIndex(2 * i + 1); }
let enc = cmd.computeCommandEncoderWithDescriptor(&desc).unwrap();
/* set pipeline, buffers, dispatch exactly as the engine does */
enc.endEncoding();
// after completion: u64 ns per sample
let raw = unsafe { sb.resolveCounterRange(NSRange::new(0, 2 * n_dispatches)) };
```

Cost of each method per dispatch (measured; empty 2048x32 kernel / 7.08 MB
streaming kernel, 400 dispatches per command buffer, medians):

| method | empty | streaming | verdict |
|---|---|---|---|
| one serial encoder (the engine) | 1.86 us | 13.99 us | reference |
| one encoder per dispatch, not sampled | 1.85 | 13.97 | free |
| one encoder per dispatch, sampled at start and end | 5.63 span (2.96 sampled interval) | 17.21 (14.96) | use this |
| one command buffer per dispatch | 13.6 | 18.8 | unusable |
| Metal 4 encoder, barrier per dispatch, `writeTimestamp` after each | 6.15 | 17.89 | no better; relaxed granularity returned zeros |

## Measure the perturbation, then correct for it

Build the instrument as switchable modes of the engine itself and measure each against the ground truth. Example
(measured; GPU us per decode step at P = 256 / 4096):

| mode | what it does | GPU time per step |
|---|---|---|
| `cb` | per-command-buffer start/end only | 3280 / 3518 (identical to the parent build) |
| `split` | one encoder per dispatch, not sampled | +2.5 / -4.3 (free) |
| `layer` | one sampled encoder per layer, head and tail (42 per step) | +371 / +339 |
| `enc` | one sampled encoder per dispatch (262 per step) | 4729 (+44%) / 4318 (+23%) |

- **Driver artifacts:** sampled encoders stalled the GPU for 300-400 us after every 80th sampled encoder, in every
  command buffer of every run (~1.1 ms per step in `enc` mode). Detect such gaps and exclude them; they fall between
  encoders, not inside a kernel.
- **Inflation is not uniform:** summed sampled intervals were 3604 us against 3280 true, 1.24 us per dispatch on
  average, but +2.7 us for a kernel that ends with device fences and +0.3-1.9 us for GEMVs. Use sampled intervals to
  rank and to see structure; do not subtract a constant and call the result a cost.
- **Per-layer segments** (one sampled encoder per layer) were each inflated by ~3.6-4.3 us; per-layer medians were
  uniform within 0.3 us across 40 layers, which is a useful sanity check for a model with repeated layers.

## In-chain marginal cost by duplication

The cleanest per-kernel cost comes from the `cb` mode alone: add one more dispatch of a class next to each real one,
interleave that build with the base build (3 engines loaded, rotating order, 6-8 rounds), and divide the extra GPU
time per step by the extra dispatches per step. Variants (timing only; outputs go to dummy buffers or the kernel is
idempotent):

- **hot**: an identical in-place copy, so it finds its weights in the cache: a lower bound;
- **cold**: the same kernel with another layer's weights, at least 8 layers back (beyond the system-level cache):
  the real cost of the dispatch;
- **empty**: an empty kernel of a given grid after every dispatch: the fixed cost of a dispatch boundary in this
  chain (Example: 0.86 us at 1 threadgroup, 1.95 us at 2048).

Noise: an unchanged arm read -8 to -11 us per step against base with +-10-20 us round spread, so a class with 38-40
dispatches per step is resolved to +-0.3-0.5 us, a class with 10 to +-1-2 us, and a class of 1-2 dispatches not at
all (use `enc` minus the average inflation there, and say so).

## Why isolated micro-benchmarks mislead

- **The system-level cache (SLC) holds a micro-benchmark's working set.** Example: a
  7.08 MB streaming dispatch cycling over 1 buffer took 7.9 us; over 2-6 buffers (14-42 MB) 8.6-8.8 us (~810 GB/s);
  over 8 buffers (57 MB) 11.4 us; over >= 10 buffers (>= 71 MB) 13.4-15 us (470-530 GB/s). Effective SLC ~48-56 MB.
  A decode step reads ~1.46 GB, so every real weight read comes from DRAM.
- Example (measured, isolated vs in chain, P = 256): expert down GEMV 12.57 vs 15.72 us (+120 us per token; its
  benchmark cycled 57 MB, mostly SLC-resident); fused router 3.93 vs 5.91 us (+75 us per token; its benchmark re-read
  the same 512 KB); add_rmsnorm 1.67 vs 2.05 us. Isolated sums 2986 us vs 3272 in chain.
- Example (prefill): a BF16 expert GEMM 2.1x faster in isolation was neutral in the
  engine (998 vs 998 ms), because its weight stream was cold there.
- An isolated gain is a hypothesis. Accept a change only on the interleaved end-to-end measurement
  (`references/measurement.md`). Example: 32-bit row offsets -3% isolated, 0.9969-0.9985 end to end (rejected).

## Effective bandwidth against the streaming ceiling

For each kernel: bytes it must read (weights incl. scales/biases, activations, cache rows) divided by its in-chain
time, and a floor of `bytes / ceiling + fixed dispatch cost`. Example (P = 256 unless noted; ceiling 590 GB/s, fixed
1.95 us):

| kernel | in chain | MB | GB/s | floor | excess |
|---|---|---|---|---|---|
| expert gate+up+act (4 experts) | 27.46 us | 14.16 | 516 | 25.95 | +1.51 |
| expert down+combine | 15.72 | 7.09 | 451 | 13.97 | +1.76 |
| fused router (norm + F32 GEMV + top-k) | 5.91 | 0.54 | 91 | 2.86 | +3.05 |
| 6-bit head + argmax epilogue | 363.3 | 213.0 | 586 | 362.96 | +0.34 |
| decode attention, 4097 keys | 26.27 | 8.40 | 320 | 16.18 | +10.09 |
| whole step (262 dispatches; GB/s derived) | 3272 | 1472 | 450 | 3006 | +267 |

Multiply the excess by dispatches per token and sort: that is the ranked opportunity list
(`references/optimization-method.md`). Kernels near the floor are done; kernels far above it with tiny byte counts
are latency chains (fences, serial top-k, dependent loads) or fixed cost, which only fusion or restructuring removes
(`kernels-fusion.md`).

## Instrumentation switches: off by default, proven neutral

- Gate every instrument by an environment variable read once at startup; with it unset the binary must run the
  parent's code path. Write the records at the next call or at exit, never inside the measured request.
- Prove neutrality, do not assume it: Example (measured): instrumented binary with no variable vs parent, decode
  1.0004 / 1.0005 / 1.0014 at 256 / 1024 / 4096 and 40 of 40 prompts token-identical.
- Keep heavy instruments on an evidence branch that is never merged; record the branch, commits and builds with the
  report so the numbers can be reproduced.
- Coarser tools for prefill: an operator-skip build (whole operator groups removed; wrong outputs, time attribution
  only) and a command-buffer split between the mixer and the FFN of every layer. Both change overlap and caching;
  use them for shares, never for absolute costs.

## A micro-benchmark harness that does not lie

Use one when the chain is too noisy to resolve a kernel variant, then confirm the result in the chain:

1. A small Rust binary linking the engine's kernel library (same MSL, same wrappers, same compile options).
2. Synthetic random weights and activations at the production shapes, cycled over more distinct copies than fit in
   the SLC (at least ~2x its size), or a large streaming dispatch in front of the kernel to evict it.
3. Many dispatches per command buffer; GPU time per dispatch from `GPUStartTime`/`GPUEndTime`; base and variant
   interleaved per trial; medians over trials.
4. A bitwise check of the variant against the base over millions of outputs, with fresh activations per round.
5. A positive control: a variant known to change rounding must be reported as different. Example: a compile-time
   combine loop flagged 78,517 of 327,680 values.
6. Probe variants for bounds only (no math, no dependent index load, loads from one address): they tell you how much a
   mechanism could at most buy. Example: a Q4 GEMV without math was only 4% faster, so arithmetic was not its limit.
7. Run it under the same GPU lock as everything else.
