# Optimization log: from 0.56x to 1.28x the reference

How the LFM2.5 engine got fast, as a teaching example: what each step changed, how it was checked and what it
bought, including what did not work. Two stages:

| stage | when | versions | formal score | decode tok/s at 256 | TTFT at 4K |
|---|---|---|---|---|---|
| agent loop (a coding agent in a build / check / evaluate loop) | 2026-10-20 00:14 to 2026-10-25 01:52 | 1-31 | 0.5604 -> 1.2484 | 171.6 -> 286.7 | 869.7 -> 394.3 ms |
| final round (Claude with parallel sub-agents, decode only) | 2026-10-25, about 3 hours | 32 | 1.2484 -> 1.2826 | 286.7 -> 305.5 | 394.3 -> 393.8 ms |

- **Formal score**: the framework's whole-request speedup over mlx-lm (geometric mean over ten context lengths from
  256 to 131,008 tokens), measured by the framework when each version was accepted.
- **decode tok/s and TTFT**: measured on 2026-10-25 by rebuilding all 32 versions with the framework's build recipe
  and running each on the same machine: the split request `generate(P, 1)` (its time is the TTFT) then
  `generate((), 128)`, at P = 256 and 4096, on fixed nonce prompts, median of 3 after one warm-up. The final version
  was re-measured five times through the series: 304.0-304.9 tok/s at 256 and 285.2-285.9 at 4K, so drift was about
  0.3%. These are not the formal numbers; reference for scale: mlx-lm decodes at 190.0 tok/s at 256 and has a TTFT
  of 463.9 ms at 4K (measured, `docs/BENCHMARKS.md`).

## (a) The agent loop: 31 accepted versions

"Change" is taken from each session's own record (its delivery summary, or its notes for v28). A version is the best engine so far across several runs, and a new run can restart from an earlier
candidate, so a row's parent is not always the row above it.

| v | day | session | formal | tok/s 256 | tok/s 4K | TTFT 4K ms | change (session record) |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.13 | 1 | 0.5604 | 171.6 | 136.7 | 869.7 | first complete engine: all 40 layers on Metal, the protocol, GPU-resident greedy loop |
| 2 | 0.23 | 2 | 0.5887 | 173.7 | 138.0 | 869.4 | MoE gate/up streams interleaved in one loop; 4-expert down GEMV with fused combine |
| 3 | 0.38 | 5 | 0.6239 | 195.3 | 169.7 | 869.5 | router GEMV on 4 simdgroups (it was latency-bound); split-K decode attention (changed a few tokens at long prompts) |
| 4 | 1.03 | 12 | 0.7266 | 178.2 | 166.4 | 624.5 | **correctness repair** after a failed audit: L=1 attention ported from MLX's `sdpa_vector` (it had used the prefill tile); prefill softmax with `exp2`, new prefill tile |
| 5 | 1.43 | 17 | 0.7482 | 178.1 | 166.4 | 570.8 | prefill chunk 2048 -> 8192; 8-simdgroup prefill attention tile |
| 6 | 1.53 | 18 | 0.7755 | 188.5 | 175.3 | 570.3 | 6-bit head GEMV with 16 rows per threadgroup, input shared by 4 rows |
| 7 | 1.79 | 21 | 0.8047 | 188.2 | 175.2 | 521.6 | **correctness repair**: decode Q4 dots round each dequantized weight to bf16 |
| 8 | 1.86 | 22 | 0.9014 | 188.3 | 175.0 | 521.8 | MLX steel/NAX flash attention for prefill above 8192 positions |
| 9 | 2.04 | 25 | 0.9221 | 187.9 | 174.7 | 521.4 | a regenerated tree; one MoE tile launch fix |
| 10 | 2.13 | 26 | 0.9578 | 207.8 | 192.0 | 498.1 | decode router GEMV with staged 8-element loads (the router was about 1.9 ms of a 4.7 ms step); 4-token prefill router |
| 11 | 2.33 | 29 | 0.9603 | 211.5 | 195.0 | 505.0 | **correctness repair**: prefill chunk back to 2048, the reference's |
| 12 | 2.40 | 30 | 0.9888 | 211.7 | 195.2 | 466.1 | 4096-row chunks for prompts of 4096 or more |
| 13 | 2.46 | 31 | 1.0313 | 242.2 | 220.9 | 465.7 | top-k: a reduction tree instead of one thread scanning 64 experts 4 times (25-30 us -> 5.3 us per call) |
| 14 | 2.63 | 33 | 1.0392 | 248.2 | 225.9 | 464.4 | 8192-row chunk ladder; decode submitted in batches of 4 instead of all at once (host encoding overlaps the GPU) |
| 15 | 2.72 | 35 | 1.0449 | 248.6 | 226.1 | 464.6 | **none**: engine source identical to v14 (only notes changed) |
| 16 | 2.83 | 37 | 1.0483 | 248.8 | 226.6 | 464.7 | a regenerated tree |
| 17 | 2.95 | 40 | 1.0535 | 252.3 | 228.8 | 465.4 | Q/K/V in one dispatch with V into the cache; B/C/X projection + conv in one kernel |
| 18 | 3.01 | 41 | 1.0611 | 254.3 | 230.7 | 464.7 | short-prefill MoE tile |
| 19 | 3.09 | 42 | 1.0667 | 253.4 | 230.2 | 465.3 | two-pass L=1 attention from 8192 positions (64 decode tokens at 131,008: 1147 -> 640 ms) |
| 20 | 3.17 | 43 | 1.0865 | 256.9 | 232.9 | 445.9 | steel/NAX attention for every prefill chunk; Q/K norm + RoPE in 2 dispatches instead of 5 |
| 21 | 3.25 | 44 | 1.0945 | 266.8 | 241.2 | 445.7 | one-warp shuffle top-k; conv B/C/X rows one simdgroup each |
| 22 | 3.34 | 45 | 1.1049 | 271.9 | 245.3 | 446.1 | vectorized add_rmsnorm; top-k recomputed inside the expert gathers (-30 dispatches) |
| 23 | 3.53 | 47 | 1.1376 | 283.9 | 255.7 | 445.7 | top-k once per token again instead of in every gather threadgroup (+4-5% decode); 128-row tile for 8192-row chunks |
| 24 | 3.78 | 51 | 1.1422 | 284.8 | 255.5 | 447.2 | MLX's `qmm_t` NAX body for the prefill expert GEMM |
| 25 | 3.92 | 54 | 1.1645 | 286.3 | 257.5 | 431.8 | 128-row shared-weight prefill expert tile |
| 26 | 4.04 | 56 | 1.1718 | 287.3 | 269.9 | 431.8 | decode attention prefetches 8 positions' K/V ahead of the softmax |
| 27 | 4.10 | 57 | 1.1867 | 286.8 | 269.7 | 431.5 | fused gate/up prefill GEMM for chunks of 512 rows or fewer |
| 28 | 4.84 | 64 | 1.2092 | 285.3 | 268.2 | 398.1 | (a new run's seed) consecutive reference chunks merged into one submission of up to 8192 rows, partition unchanged |
| 29 | 4.91 | 65 | 1.2161 | 287.1 | 270.2 | 398.2 | 128 x 128 tensor-op tile for the dense prefill projections at 4096+ rows |
| 30 | 5.10 | 68 | 1.2297 | 287.8 | 270.7 | 393.4 | prefill expert GEMM stages its activation tile once per 128-row block |
| 31 | 5.20 | 70 | 1.2484 | 286.7 | 269.7 | 394.3 | prefill attention launched so the 4 query heads of a KV head run adjacently (1.09x prefill at 131,008) |
| 32 | 5.70 | final | 1.2826 | 305.5 | 286.3 | 393.8 | the final round, section (b) |

What the curve teaches:
- **The largest decode steps came from a 0.5 MB kernel.** v13 (+14.4%) and v10 (+10.6%) fixed the MoE router (an
  F32 GEMV plus a top-4 over 64 experts, 1.4% of the bytes per token), and v3 (+12.4%) fixed it together with the
  decode attention. It was latency-bound: a serial scan, a dependent FMA chain, dependent loads. The other big decode
  steps: the final round (+6.6%), the head GEMV in v6 (+5.8%), v23 (+4.4%) and v21 (+3.9%).
- **Correctness repairs come back.** v4 (L=1 attention), v7 (rounding of the dequantized weights) and v11 (the
  partition) were repairs after failed audits. A later repair (in the notes, between v23 and v28) replaced v7's
  per-weight bf16 rounding with MLX's `qmv` sums: v7 had reconstructed the reference's formula instead of porting
  its kernel. v4's record says its parent ran L=1 attention through the prefill tile, which v3 did not, so v4 descends
  from an earlier candidate: it decodes 8.8% slower than v3 and was accepted on a faster whole request (TTFT -28%).
- **The prefill partition is part of the specification.** Changing the chunking (v5 also stopped running the last
  prompt token as its own L=1 step; v12 and v14 used 4096/8192 ladders) sped up prefill and failed audits twice: v11
  went back to 2048, and a later repair restored the exact partition after a 512-token prompt's last token had been
  computed inside a 512-row chunk. The safe form came at v28: keep the reference's chunk boundaries and only merge
  whole chunks into one submission, which leaves every row's arithmetic unchanged (its record: request time 1.15x at
  4K and 1.24x at 8K, tokens identical).
- **Formal scores carry noise.** v15's engine source is v14's, yet it scored 1.0449 against 1.0392.

## (b) The final round: decode fusion, one mechanism at a time

Parent: v31. Sub-agents worked on separate branches against a common parent build, each change bitwise-checked with
`scripts/same.py` (40 prompts x 3 request shapes: `generate(P, 65)`, the split request, a continuation chain) and
timed with `scripts/ab.py` (interleaved ABBA). Noise floor, A against A, 6 rounds: decode 1.0000 / 1.0032 / 1.0021 at
256 / 1024 / 4096. Ratios below are B's decode speedup over its parent at 256 / 1024 / 4096 tokens.

| # | mechanism | dispatches per token | verification | measured effect |
|---|---|---:|---|---|
| 0 | two later candidates of the stopped agent run, not part of v31: residual add + RMSNorm in the router GEMV's prologue (each of the 64 threadgroups recomputes the norm row; group 0 writes it), and Q/K norm + RoPE + K-cache write in one kernel | -48 | same.py 40/40 | 1.0154 / 1.0166 / 1.0142 (6 rounds) |
| 1 | compile-time K, group size and rows in every decode Q4 GEMV: the block loop's divisions become shifts, lanes walk the same blocks in the same order | 0 | same.py 40/40; micro-benchmark bitwise over 1.5-9.8 M outputs per kernel | 1.0268 / 1.0287 / 1.0303 (10 rounds, quiet machine), 283.6 -> 291.7 tok/s; per dispatch e.g. out_proj 6.27 -> 5.48 us, expert down 13.47 -> 12.57 us |
| 2 | the top-4 in the router's last-finishing threadgroup: each group writes its logit, a device-scope fence, an atomic increment; the group that sees 63 runs the unchanged warp top-k and resets the counter. No group waits for another, so nothing depends on co-residency (needs MSL 3.2 fences) | -38 | same.py 40/40; 20,000 back-to-back synthetic dispatches and 38,912 replays on captured real activations bitwise equal to the separate kernels, counter back at 0 | 1.0083 / 1.0109 / 1.0028 (6 rounds) |
| 2b | the router's lane sum unrolled for K = 2048 (same add order) | 0 | as 2 | below that day's noise (16 rounds 1.0105 / 1.0102 / 1.0211 with a +-5% per-round spread) |
| 3 | token tail: the head GEMV folds each threadgroup's best bf16 logit into one 64-bit `atomic_max` key (value descending, index ascending; -0 ties +0; NaN never wins) and stores no logits; one 256-thread kernel writes the token, the next embedding row and its RMSNorm | -3 | same.py 40/40; a check build ran the parent's kernels into shadow buffers every step and failed on any differing word | 1.0035 / 1.0045 / 1.0084 |
| 4 | dense FFN gate + up + SwiGLU in one kernel (two dot chains per lane, the parent's expression on the bf16 sums) | -4 | same.py 40/40; check build: 0 differing words in 7,680 steps | 1.0030 / 1.0016 / 1.0058; micro-benchmark 79.9 -> 76.6 us per dense layer |
| 5 | the first command buffer of each `generate` holds one step, then 4 | 0 | same.py 40/40 | GPU start-up idle 585 -> 348 us per call (instrumented); 30 rounds at 256: median 1.0008 (expected about 0.1%) |
| 6 | Q/K norm + RoPE in the single-pass attention kernel's prologue; the lanes that own the current position take the new key from threadgroup memory, so no threadgroup reads the cache row this dispatch writes | -10 | same.py 40/40, plus 45 long and 75 dev-length prompts; kernel harness: 5,228 random cases at lengths 1-8191, output and whole K cache identical | -1.1 us per layer in the harness (expected about +0.3%); 40 rounds 1.0071 / 1.0064 / 1.0077 on a noisy machine |
| 7 | the router issues its 8 KB F32 weight row and bias loads at kernel entry, overlapping the norm prologue | 0 | same.py 40/40; the stress tests of 2 | 6.00 -> 4.75 us per layer behind a DRAM stream (micro-benchmark); 24 rounds 1.0042 / 1.0047 / 1.0046 (a 6-round run read 0.993-0.995: noise) |

Totals:
- **Bitwise**: the final engine against v31 on 84 prompts (Lily nonce prompts of 1 to 32,768 tokens, random-token
  prompts, and the 39 public dev cases up to 131,008 tokens) in the three request shapes: identical.
- **Dispatches**: 355 -> 252 per token below 8,192 positions (`decode-map.md`).
- **Speed**: mechanisms 0-5 together (base2) against v31, 10 rounds: decode 1.040 / 1.041 / 1.057 / 1.054 / 1.051 /
  1.043 at 256 / 512 / 1K / 2K / 4K / 8K. All of them: 286.7 -> 305.5 tok/s at 256 (+6.6%) and 269.7 -> 286.3 at 4K
  (+6.2%) in the version re-measurement; TTFT unchanged (prefill was not touched); formal score +2.7%, the
  whole-request score diluted by the unchanged prefill. Formal paired decode at 256: 303 tok/s against mlx-lm's 198
  (1.53x).
- **The profile drove wave 2.** Between the two waves a per-kernel profile inside the real engine
  (`decode-map.md`) ranked the router (excess about 116 us per token) and the expert down GEMV (about 67 us) as the
  largest recoverable items. Wave 2 got mechanism 7 from the router and nothing from the down GEMV (below).

## (c) Tried and rejected, with numbers

Final round:
- **The post-FFN add_rmsnorm in the last threadgroup of the down GEMVs** (the pattern of mechanism 2 on a wide
  grid): bitwise identical but decode 0.861 / 0.847 / 0.856 of the parent, every round slower. The fence costs about
  8 us on a 2048-threadgroup dispatch (expert down + fused norm 24.72 us against 14.51 for down + a separate norm;
  16.9 us even without the fences, because a one-threadgroup epilogue loses to a 1.7 us norm dispatch).
- **Concurrent encoder with an explicit barrier after every dispatch**: 0.973 and 0.977 of the serial encoder.
- **Command-buffer shapes** (relative to 4 steps per buffer, instrumented sweep at 256): 1-then-4 1.0014 (kept),
  1-then-8 1.0006, 1-then-16 1.0013; a warm-up command buffer, spinning on the last buffer, unretained references:
  within +-0.2% (one throttled run read 0.980 for the warm-up buffer).
- **GEMV internals after compile-time shapes** (micro-benchmark, us per dispatch): an explicit trip-count loop 6.20
  against 5.46; loads hoisted into arrays 6.08; 16-byte activation loads 5.74 against 5.47; 2 or 4 rows per
  threadgroup neutral; 32-bit row offsets -3% on the expert down kernel alone but 0.9969 / 0.9985 / 0.9973 end to end.
  Not bitwise: compile-time shapes in the 6-bit head (388 of 2.56 M outputs differ, and slower: 366 -> 398 us) and a
  fixed four-term expert combine (78,517 of 327,680 differ): fast-math re-associated both.
- **The expert down GEMV, in the chain**: loading both blocks early was 0.7 us faster in a micro-benchmark that
  streamed from DRAM, but +0.4 to +8.2 us per step in the engine (A/B 0.98-1.005: neutral); two rows per simdgroup
  +24 us per step and a token mismatch; lane-balanced loads +57 us per step; prefetching the routed experts' first
  down rows from the gate/up kernel +12 and +72 us per step. The micro-benchmark did not predict the chain.
- **Router prologue**: computing the norm gain before the first barrier was not bitwise (same.py 26/40) even though
  the front-end IR was identical: the GPU back end re-associated `(x · inv_rms) · gain` once the gain was available
  early. The bitwise variants (one barrier fewer; an identity shuffle to pin the order) gained 0.1 us or less.
- Removing the router's two device fences saves about 0.2 us per dispatch; kept, they are the memory-model
  guarantee.

Agent loop (from the session records):
- Top-k recomputed in every expert-gather threadgroup (v22, -30 dispatches) lost to one top-k dispatch per layer
  (v23, +4-5% decode; an earlier measurement 239.0 against 228.2 ms). One 1024-thread threadgroup doing the router
  GEMV and the top-k was 5-7% slower than 64 threadgroups plus a top-k dispatch.
- Four-row and two-row Q4 GEMVs for the dense decode projections: 64-token decode at 256 went 257.6 -> 266.0 ms
  (x4) and 264.9 ms (x2).
- A prefill submission cap of 16,384 rows instead of 8,192: 8,192 was faster by 2.5% at 32K and 3.7% at 64K.

## (d) Lessons

- **Port the reference's kernels, not your reading of its formula.** Rounding comes from the kernel and its per-shape
  choice: v7's reconstructed rounding had to be replaced by MLX's `qmv` sums, and v4 had to put the reference's own
  `sdpa_vector` on the L=1 path.
- **Keep the reference's partition and change only how you submit it.** Merging whole chunks gave 1.15-1.24x request
  time at 4K-8K with identical tokens; changing the chunk sizes failed audits twice.
- **Look for latency-bound small kernels first.** The router, 1.4% of the bytes, gave two of the loop's three largest
  decode gains and part of the third: serial scans and dependent loads, not bandwidth.
- **Once the GEMVs stream near the ceiling, count dispatches.** With every weight kernel at 425-586 GB/s, removing
  103 of 355 dispatches and fixing compile-time shapes gave +6.6% decode, bitwise.
- **Grid-wide dependencies are cheap only on narrow grids.** The last-threadgroup pattern wins on a 64-threadgroup
  router and costs 14% on 2048-threadgroup GEMVs.
- **Measure inside the real chain.** Isolated micro-benchmarks whose weights stayed in the cache were off by up to
  25% (expert down 12.6 against 15.7 us), and one predicted a gain the chain did not show; in-chain costs summed to
  the measured step within 0.3%.
- **Verify bits even for "pure" refactors.** Fast-math re-association broke two of them with identical source
  expressions; check builds that run the parent's kernels into shadow buffers every step catch it at once.
- **Run enough rounds.** Day-to-day spread reached +-5% per round on a throttling laptop; sub-1% changes needed 24-40
  ABBA rounds, and even a formal score moved 0.5% on identical code.
