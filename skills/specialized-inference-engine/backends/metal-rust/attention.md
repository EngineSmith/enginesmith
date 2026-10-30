# Attention

Purpose: how to build decode attention (one query row against the KV cache) and prefill attention for a GQA decoder
on Apple Silicon, how to lay out and write the KV cache, and which of the reference's attention routes you must
reproduce. Decode attention is latency-bound at short context and bound by the KV read at long context; prefill
attention is a compute-bound flash tile. Read it before touching any attention kernel, the cache, or the threshold
between attention routes. Fusions around attention are in `kernels-fusion.md`; the prefill pipeline around the tile
is in `prefill.md`.

Numbers are measurements from the LFM2.5 worked engine on an Apple M5 Max unless they are marked as estimates; MLX's
routes are from MLX's `mlx/backend/metal/scaled_dot_product_attention.cpp`.

## Know the cost first

- **KV bytes per decode token** at context C: `C x attention_layers x kv_heads x head_dim x 2 (K and V) x bytes per
  element`. Put it in the model card (section 6) beside the weight bytes. Above some context it is the larger term,
  and from there decode speed is attention speed.
- Example (LFM2.5: 10 attention layers, 8 KV heads, D = 64, BF16; estimate): 20,480 B per position, so 2.68 GB per
  decode token at 131,072 positions against 1.46 GB of weights. At the ~590 GB/s streaming ceiling (`README.md`)
  that is a floor of ~4.5 ms per token, ~35 ns per position, for the KV read alone.
- **At short context the kernel is latency-bound.** Example (measured in the real decode chain): single-pass attention
  4.95 us per layer at 256 keys (108 GB/s effective) and 26.3 us at 4097 keys (320 GB/s; floor 16.2 us). Isolated,
  before load pipelining, it reached only 15-59 GB/s at 256-8192 keys.

## Reproduce the reference's routes

A framework usually has several L=1 attention kernels and picks one by shape and device. Each route rounds
differently (the split form combines partial softmaxes in another order), so read the dispatch code at the pinned
version, list every condition, and record it in the model card (section 7). See `references/correctness.md`.

Example (MLX, read in the source of 0.31.2 and 0.32.3): for query length <= 8 it calls `sdpa_vector_2pass` when
`(arch suffix is 'd' or 's') && keys >= 1024`, or when `kv_heads < q_heads && keys >= 4096`; otherwise the single-pass
`sdpa_vector`. The two-pass block count is a function of the suffix, `n_simds = gqa_factor x q_len` and the key count
(on 's' it is 64 unless `n_simds > 4`). The suffix is the last character of the device's architecture name
(`MTLDevice.architecture().name()`); read it on the target machine and record it. The LFM2.5 notes assumed the 1024
threshold applies on the M5 Max; the architecture string was not recorded.

Example (LFM2.5 engines, measured): the engines switch at 8192 positions
with 128 blocks up to 49,152 positions and 512 above, chosen by speed (two-pass at 4096 positions: 274 vs 271 ms per
64 tokens; 64 and 1024 blocks slower than 128/512). That is not MLX's rule: it is a rounding route the reference does
not take, accepted because the measured mismatch counts stayed within that task's tolerance. Prefer the reference's
threshold and block count; treat any other choice as a correctness change and check it against the reference.

## Single-pass vector kernel (short context)

Layout of MLX's `sdpa_vector` for D = 64: one threadgroup per query head, 32 simdgroups of 32 lanes; lane `l` holds
D/32 elements of q, K and V; simdgroup `s` walks positions `s, s+32, s+64, ...` with an F32 online softmax; the 32
partial (max, sum, output) triples are combined through threadgroup memory; the output is rounded to BF16 once.

The loop is a dependent chain per position (K load, `simd_sum`, `exp`, V load). Issue several positions' loads
first, then fold them in the original order; every value stays bitwise the same:

```metal
// p = sg; stride BN = 32 positions; lane holds QPT = VPT = 2 of D = 64 elements
constexpr uint U = 8;
for (; p + (U - 1) * BN < len; p += U * BN) {
    float kk[U][QPT], vv[U][VPT];
    for (uint u = 0; u < U; ++u) {                       // loads for U positions in flight
        const ulong off = (ulong)(p + u * BN) * D;
        for (uint j = 0; j < QPT; ++j) { kk[u][j] = float(kb[off + j]); vv[u][j] = float(vb[off + j]); }
    }
    for (uint u = 0; u < U; ++u) {                       // the parent's body, verbatim, same position order
        float s = 0.0f;
        for (uint j = 0; j < QPT; ++j) { s += qv[j] * kk[u][j]; }
        s = simd_sum(s);
        const float new_max = max(max_score, s);
        const float factor = metal::fast::exp(max_score - new_max);
        const float exp_score = metal::fast::exp(s - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        for (uint j = 0; j < VPT; ++j) { ov[j] = ov[j] * factor + exp_score * vv[u][j]; }
    }
}
for (; p < len; p += BN) { /* the parent's one-position loop, unchanged */ }
```

- Example (measured, isolated, 10 dispatches): U = 8 took 1024 keys 67.8 -> 47.0 us,
  4096 keys 155.5 -> 117.5 us, 8192 keys 282.9 -> 134.2 us; 256 keys unchanged (the fixed cost dominates). End to end
  about -1% decode time, within that day's noise.
- The grid is the number of query heads, and each KV head is read once per query head of its group. Example (LFM2.5):
  32 threadgroups on a 40-core GPU. Acceptable only while the KV read is small.
- **Do not fold the group's query heads into one threadgroup here**: that cuts the grid to the number of KV heads.
  Example (measured): a GQA-folded single-pass kernel with 8 threadgroups,
  bitwise identical, raised decode at 8192 positions from 4.75 to 5.25 ms/token.

## Split (two-pass) form (long context)

Stage 1, grid `(kv_heads, blocks)`: block `b` takes positions `b, b + blocks, b + 2 blocks, ...` (interleaved, so
every block has work at every length the path serves) and keeps one F32 online softmax per query head; it writes
the partial output (BF16 in MLX) and its F32 max and sum. Stage 2, one threadgroup per query head: rescale each
block's partial by `exp(max_b - max)`, sum in block order, divide once. Share K/V between the group's query heads
inside a block: one simdgroup owns a block and accumulates all G heads, each with its own statistics, so each K/V
row is loaded once per block instead of G times; the partial layout is the generic one, so the same combine runs.

```metal
// stage 1, GQA group G = 4; one simdgroup per threadgroup; tg = (kv head, block)
for (uint t = block; t < len; t += blocks) {
    const float k0 = float(kp[0]), k1 = float(kp[1]);
    const float v0 = float(vp[0]), v1 = float(vp[1]);
    for (uint h = 0; h < G; ++h) {                       // per-head max/sum/output, unchanged per head
        float s = simd_sum(q0[h] * k0 + q1[h] * k1);
        const float nm = max(mx[h], s);
        const float f = metal::fast::exp(mx[h] - nm), e = metal::fast::exp(s - nm);
        mx[h] = nm; sm[h] = sm[h] * f + e;
        o0[h] = o0[h] * f + e * v0; o1[h] = o1[h] * f + e * v1;
    }
    kp += (ulong)blocks * D; vp += (ulong)blocks * D;
}
// partials[(hq * blocks + block) * D + 2 * lane + {0,1}] = bfloat(o); sums/maxs[hq * blocks + block] = sm/mx
```

- Example (measured, decode ms/token, single-pass -> two-pass): 32,768 positions 7.42 -> 5.33; 65,536 10.86 -> 7.48;
  131,008 17.86 -> 10.06.
- Counterexample (measured, an LFM2.5 candidate engine): an engine that ported only the single-pass
  kernel had a decode slope of ~103 ns per position against the reference's ~55, crossed the reference near 21K
  positions and ran decode at 0.70x the reference at 131,008. Its code comment claimed the reference used the
  single-pass kernel for every L=1 call. Port every route.
- Block counts are constants of the length range and multiples of the combine width (32). Partial traffic is small:
  `q_heads x blocks x D x 2 B` per layer (Example: 2.1 MB at 512 blocks vs 268 MB of K+V per layer at 131K; estimate).
- Measured and rejected: the single-pass load pipelining applied to stage 1 was slower below 64K positions;
  two-pass below 8192 positions was slower than single-pass (above).

## KV cache layout and writes

- **Layout** `[kv_heads, max_seq, D]` in the reference's cache dtype, per layer, allocated at the maximum context at
  load. One head's position row is D contiguous elements and one head's prefix is one contiguous range, which is
  what both the split blocks and the prefill tile stream. Example (LFM2.5): 20 KB per token of context in all.
- **Write V from the projection.** The fused Q/K/V GEMV writes its V rows straight into the cache at the current
  position; a separate scatter is a pure copy and one more dispatch (`kernels-fusion.md`). K is written after its
  norm and RoPE by the kernel that computes them.
- **No intra-dispatch read of a row being written.** In a fused norm/RoPE + attention kernel, take the current
  position's key from threadgroup memory and let exactly one threadgroup per KV head (the first query head of the
  group) write it to the cache; the other threadgroups never read that cache row in this dispatch.
- **Rows beyond the position are stale, not zero.** Every kernel bounds its loop by the live length, so a rewind is a
  position change with no clearing (`decode-loop.md`), and prefix reuse keeps rows in place (`serving.md`).
- **The stored dtype is arithmetic.** Quantizing or re-laying-out cached values changes what attention computes;
  treat it as a correctness change against the reference, not an optimization.

## Q/K norm and RoPE in the attention prologue

With per-head Q/K norms before attention, the norm + RoPE kernel is tiny and costs a dispatch per attention layer.
Fold it into the single-pass kernel (mechanics and verification: `kernels-fusion.md`): in threadgroup `hq`,
simdgroup 0 normalizes and rotates query head `hq`, simdgroup 1 the key row of its KV head at the current position;
both rows are rounded to BF16 and staged in threadgroup memory exactly as the separate kernel stored them, with
explicit `fma` in the rotation. Only the simdgroup that owns the current position takes that key from threadgroup
memory, in a copy of the block or tail step that holds it; the loop partition is unchanged.

Example (measured): -10 dispatches per token; -1.1 us per layer in a 10-layer kernel harness; decode +0.64% to +0.77%
at 256-4096 positions over 40 interleaved rounds (day noise +-0.2-0.5%); bitwise: 5,228 random cases at lengths
1..8191, output and the whole K cache identical. The two-pass path kept the separate kernel; folding it into stage 1
was not tried.

## Prefill attention

- **Use the reference's flash tile at your head dimension** (MLX: steel attention; `steel_attention_nax.h` uses the
  matrix units of GPU family 10, `prefill.md`). Generate the MSL from the pinned headers with a script; never hand-edit
  the generated file. Example (measured): a hand port whose launcher's `BK` disagreed with the instantiated template
  covered only part of the key range; check launcher constants against the template's at pipeline creation.
- **Causal tiles:** each query tile visits key blocks from 0 to its diagonal; only the diagonal block needs the
  mask. A chunk after the first, or a continuation, offsets the query rows by the cached length.
- **Launch order shares K/V across the group.** Make the G query heads of one KV head adjacent in dispatch order
  (grid axes `(head within group, query block, kv head)`), so they read the same K/V blocks at nearly the same time
  and hit the cache. Every threadgroup computes the same rows in the same order. Example (measured): the tile went
  from ~35 to ~61 TFLOP/s at long context; prefill at 131,008 tokens 33.0 -> 30.2 s (1.09x), at 65,536 1.04x, at <=
  8192 unchanged; tokens identical.
- **Tile shape** is a measured constant. Example (D = 64, M5 Max): BQ 64, BK 32, two simdgroups; bq32k32, bq32k64,
  bq64k64, bq128k32 and bq128k64 were equal at 8192 and 2% slower at 32,768.
- **The last prompt token** follows the reference's partition: when it is an L=1 step there, it takes the decode
  kernels, not the tile (`prefill.md`).

## Checks specific to attention

- A kernel harness that compares a new attention kernel with its parent on random lengths over its whole range
  (1, 31, 32, 33, ..., both sides of every route threshold), comparing the output and the whole cache bytes.
- Prompts and continuations whose decode steps cross each route threshold (`references/correctness.md`, boundary
  prompts), and at least one request at the task's maximum context for every change that touches attention.
- Throughput at several contexts (for example 256, 4K, 32K, the maximum): a short-context gain can hide a
  long-context loss, and the reverse.
