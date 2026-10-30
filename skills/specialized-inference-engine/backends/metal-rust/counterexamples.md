# Counterexamples

Purpose: measured negative and neutral results on Metal + Rust, one line each with its number, grouped by topic, so
that no session repeats them without a new reason. Read it before trying an idea (phase 4 step 2 of
`references/workflow.md`). Each entry was measured on the LFM2.5 worked engine on an Apple M5 Max unless it says
otherwise; the topic files explain the mechanisms. A result here is a fact about that engine and machine, not a law:
retry only with a stated reason why your case differs, and record the new number.

Numbers are measurements from the LFM2.5 worked engine on an Apple M5 Max unless they are marked as estimates. The
general counterexamples at the end are derived from Lily's public source and tests (Apache-2.0), not reproduced
locally.

## Encoding, synchronization and submission

- **Concurrent encoder, barrier after every dispatch**: whole decode 0.9729 (4-step buffers) and 0.9773 (1:4) of one
  serial encoder, i.e. 2.3-2.7% slower.
- **Last-threadgroup fence pattern on a wide grid**: device-scope fence + atomic counter on a 2048-threadgroup down
  GEMV: 12.83 -> 24.72 us per dispatch (16.91 without the fences), ~8 us of fence cost; decode 0.861 / 0.847 / 0.856
  of the parent at 256 / 1K / 4K. The same pattern wins 0.46 us per layer at 64 threadgroups.
- **Hierarchical (two-level) completion counters**: 24.74 / 24.78 vs 24.72 us; contention is not the cost.
- **Epilogue run by the last threadgroup instead of a separate dispatch**: even with free synchronization the
  128-thread epilogue (16.7 us total) lost to a separate 256-thread norm (1.7 us after the GEMV); a one-simdgroup
  epilogue on the dense down GEMV took it from 19.1 to 53.9-61.3 us by cutting occupancy.
- **Decode steps per command buffer 8 or 16 instead of 4**: 1.0006 / 1.0013 vs 1.0014, the same.
- **Warm-up command buffer, spin-wait on the last buffer, unretained references**: neutral, 0.9985-1.0015 (one noisy
  run 0.98; details in `decode-loop.md`).
- **Prefill: host encodes chunk n+1 before chunk n's wait**: the host rewrote the token buffer before the GPU read
  it; every token of the second chunk diverged.

## Router and top-k

- **Top-k recomputed in every expert-gather threadgroup** (one dispatch per layer fewer): 239.0 vs 228.2 ms per 64
  tokens at P=256, 277.7 vs 255.0 at P=8192; 4-5% slower.
- **Router GEMV + top-k in one 1024-thread threadgroup**: 5-6% slower decode (5-7% in another measurement); one
  threadgroup holds one core.
- **Norm gain combined before the first barrier** (earlier operand availability): not bitwise, 26 of 40 prompts
  differ, 1 of 64 synthetic input sets; the IR before the GPU backend was identical.
- **Making that variant bitwise with an identity `simd_shuffle`**: clean, but 4.5 vs 4.4 us streamed: no gain, opaque
  code.
- **Dropping the router's device fences**: ~0.05-0.2 us per dispatch in a micro-benchmark; not allowed by the memory
  model, kept.

## Decode GEMV structure (details: `kernels-gemv-quant.md`)

- **2 or 4 rows per simdgroup or threadgroup** (shared activation loads, bitwise): decode 257.6 -> 264.9 / 266.0 ms
  per 64 tokens; with the expert gathers too 223.6 -> 260.3 ms; two rows per warp in the expert GEMVs 5-6% slower.
- **Precomputed activation sums** in the expert gate/up gather: 223.2 -> 226.2 ms, slower: not ALU-bound.
- **Rows per threadgroup**: gather 1/2/4/8 rows 26.09 / 25.57 / 25.62 / 25.85 vs 25.7-25.96 us; out_proj 2/4 rows
  5.451 / 5.461 vs 5.476 us: neutral.
- **Explicit fixed trip count plus tail; loads hoisted into arrays**: 6.196 and 6.079 vs 5.458 us.
- **16-byte activation loads**: 5.744 vs 5.468 us at N = 2048 (slower), neutral at N = 8192.
- **Dense down unrolled 2x by hand**: 28.297 vs 27.523 us; **conv projections in one simdgroup, B/C/X interleaved**:
  14.548 vs 14.097 us.
- **32-bit row offsets**: -3% isolated on the expert down GEMV; decode 0.9969 / 0.9985 / 0.9973 end to end, rejected.
- **Expert down combine weights loaded at kernel start**: 12.233 vs 12.216 us, neutral.

## Fast math and rounding (rules: `README.md` rule 3, `references/correctness.md`)

- **A fixed four-term combine** (compile-time trip count of the expert combine loop): fast math re-associated it;
  78,517 of 327,680 outputs differ.
- **Compile-time K and group size on the 6-bit head**: 388 of 2.56 M outputs differ, and slower, 366.2 -> 398.1 us.
- **A copied rotation without explicit `fma`** in a fused Q/K norm + RoPE: 2 of 409,600 values moved by one BF16 ulp;
  a token check then failed at step 56 of a one-token prompt; with explicit `fma` it was identical.
- **Positive result to copy**: the compile-time shapes that kept every lane's block order and expression changed no
  bit over 1.5-9.8 M outputs per kernel and bought ~2.8% decode.

## Measurement traps (details: `profiling.md`)

- **Isolated gains that vanish in the chain**: expert down 12.57 us isolated vs 15.72 in chain; fused router 3.93 vs
  5.91; isolated sums 2986 vs 3272 us per step. The isolated working sets sat in the SLC.
- **BF16 expert weights with a whole-K GEMM**: 2.1x faster isolated (77.8 vs 36.5 TFLOP/s), 998 vs 998 ms prefill at
  8192 tokens in the engine.
- **Persistent BF16 dense projections**: -3.8% prefill at 256 tokens, +0.4% / +1.1% at 2048 / 8192.
- **Run order masquerading as a speedup**: a tile A/B in which one arm always ran second showed 11-13% lower prefill
  time; reversing the order reversed the result.
- **Changes below the day's noise**: a router unroll measured 0.9894-1.0211 across reruns with +-5% per-round spread;
  report such a change as unresolved, not as a gain.
- **Sampled per-dispatch timestamps**: +1.24 us per dispatch on average, +2.7 us on a kernel with fences, and a
  300-400 us stall every 80th sampled encoder.

## Attention (details: `attention.md`)

- **Porting only the single-pass L=1 kernel**: decode slope ~103 vs ~55 ns per position for the reference, crossover
  near 21K positions, 0.70x the reference at 131,008 (an LFM2.5 candidate engine).
- **GQA-folded single-pass kernel** (8 threadgroups): decode at 8192 positions 4.75 -> 5.25 ms/token.
- **Two-pass below 8192 positions**: 274 vs 271 ms per 64 tokens at 4096; block counts 64 and 1024 slower than 128 /
  512.
- **Batched K/V prefetch in the two-pass stage 1**: slower below 64K positions (324.8 -> 384.4 us at 8192), faster
  only at 131,008 (2228.5 -> 2047.2 us).
- **Prefill attention launcher `BK` disagreeing with the template's**: the kernel covered only part of the key range.
  **Other tile shapes** (bq32k32 ... bq128k64): equal at 8192, 2% slower at 32,768.

## Prefill (details: `prefill.md`)

- **Chunk ladder instead of the reference partition**: the last token of a 512-token prompt landed in a 512-row chunk;
  mismatch at step 0, reference gap 0.94.
- **Launch geometry not matching the generated kernel** (128 threads for 256-thread templates): ran, plausible output,
  34 of 65 teacher-forced mismatches at 32,768 tokens.
- **Double-buffered threadgroup B staging** in the grouped Q4 GEMM: +3.5% slower, the extra 16 KB cut occupancy.
- **Wider expert tiles**: BN 128 8% slower; BM 128 with four simdgroups 12% slower at 2048 tokens; BK 128 neutral;
  128x128 4% slower at 8192. A later 128-row tile that stages the weights once for two sub-tiles and skips empty ones
  did win (`prefill.md`): tile results depend on the exact mechanism.
- **Materializing expert weights per chunk**: +50% prefill; per expert: 14 TFLOP/s, launch-starved.
- **Fused dequantizing GEMM for short chunks** (<= 512 rows): 65.0 / 100.8 -> 65.8 / 103.6 ms at 256 / 512 tokens,
  neutral; dense projections through the MLX `qmm_t` body: neutral.
- **Eight-element vectorized `silu_mul`**: 2.130 -> 2.293 ms, 7% slower.
- **Wider expert-row gather and combine (32 bytes per thread)**: bitwise, kept, but prefill ratio 0.996: no
  measurable gain.

## General counterexamples (source-derived, not measured here)

- **Concatenated projections change reduction routes.** Lily's test rejects uniform routing for M = 1, total width
  65,536, slices [65,535, 1]: the stacked matrix takes a different GEMV route than its slices. One larger matmul over
  the same weights is not bitwise the same as two; check the dispatched route after any shape change.
- **A cast in the source does not guarantee hardware rounding.** Lily's residual + norm test reports one BF16 ulp on
  g14 when the compiler keeps an unrounded F32 sum, and bit identity on g15-g17. Check stored residuals and normalized
  outputs separately, on the target GPU.
- **Never pad quantization groups.** Lily rejects K = 96 with group 64 before dispatch. Dispatch a fast path only when
  the packing contract holds; never pad or requantize checkpoint weights to fit one.
- **Double-buffer parallel history writes.** In-place prefill convolution history updates race across threadgroups;
  use two windows and flip. Scratch reuse needs a full live-extent write before any read, or a shorter chunk reads a
  longer one's stale tail.
- **Keep intermediate BF16 casts in fusions.** Removing `float(bfloat(gate_sum))` and `float(bfloat(up_sum))` before
  SwiGLU changes the function; keep the rounding in registers when the memory round trip goes.
