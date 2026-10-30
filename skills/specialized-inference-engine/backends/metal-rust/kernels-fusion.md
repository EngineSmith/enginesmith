# Kernel fusion for decode

Purpose: a catalog of the fusions that remove dispatches from a batch-1 decode step without changing a single output
bit: for each, what is fused, its preconditions, how to keep it bitwise identical, and its measured effect in the
worked engine. Also the fusions that were measured and rejected, and why. Read it once the weight-streaming kernels
are near the bandwidth ceiling (`kernels-gemv-quant.md`): from there on, the fixed cost per dispatch (0.86-1.95 us in
the chain, `README.md`) is the largest lever left.

## Rules that keep a fusion bitwise identical

1. **Keep every rounding point.** Where the parent stored a BF16 tensor and the next kernel reloaded it, round to BF16
   at the same point in a register or in threadgroup memory: `float(bfloat(x))`. Removing a cast removes a rounding.
2. **Copy each producer's arithmetic verbatim:** the same expressions, statement order, lane-to-element mapping,
   reduction tree and `simd_sum`s. Call the same inline helper from the fused and the separate kernel where you can.
3. **Write `fma` explicitly where the parent's compiled code contracted** (README rule 3), and do not hoist an operand
   ahead of the code it used to follow without a bitwise check.
4. **No in-place writes that other threadgroups of the same dispatch read.** Write to a second buffer and swap
   (ping-pong). Let exactly one threadgroup write each shared output.
5. **A grid-wide dependency costs in proportion to the grid.** If a fused epilogue needs every threadgroup's result,
   use the last-finishing-threadgroup pattern below, and only on small grids.
6. **Verify three ways, then end to end.** (a) A kernel harness: fused kernel against the parent kernels on random
   inputs over many rounds and edge shapes (lengths 1..max, last rows, last groups). (b) Captured real activations
   replayed through both. (c) A check build that runs the parent kernels into shadow buffers every decode step and
   fails on any differing word. Then token-for-token comparison on many prompts and request shapes. Random inputs
   alone are not enough: one Q/K norm+RoPE fusion was identical on random tensors and on four prompts and still
   changed tokens on others.

## Catalog (effects measured on LFM2.5, M5 Max)

Decode effects are interleaved A/B speedups against the previous build at 256 / 1024 / 4096 context unless noted.

| fusion | dispatches removed per token | measured decode effect |
|---|---|---|
| residual add + RMSNorm into the router prologue | 38 | +1.5% together with the Q/K norm+RoPE kernel below (-48 in all) |
| router + top-k in the last-finishing threadgroup | 38 | +0.83% / +1.09% / +0.28% |
| router lane sum with compile-time K; router row staged at entry | 0 | below noise; then +0.42% / +0.47% / +0.46% (24 rounds) |
| Q/K norm + RoPE + K-cache write as one kernel | 10 | in the +1.5% above |
| Q/K norm + RoPE folded into the attention prologue | 10 | +0.71% / +0.64% / +0.77% (40 rounds; harness predicted ~+0.3%) |
| Q/K/V projections in one dispatch, V written to the cache; conv projections + conv | 1 per conv layer, 2 per attention layer | decode 4.02 -> 3.96 ms/token at P=256 (with a router re-schedule) |
| expert gate/up + activation; expert down + combine | (in the incumbent) | not measured separately |
| dense gate + up + SwiGLU | 4 | +0.30% / +0.16% / +0.58% |
| head GEMV with argmax epilogue + token tail | 3 | +0.35% / +0.45% / +0.84% |

Together with compile-time GEMV shapes and the one-step first command buffer, these took the worked engine from
~355 to ~252 dispatches per token, and decode at 256 context from ~285 tokens/s (agents' harness) to 304.5 tokens/s
in the release benchmark, 1.60x the reference's 190.0 on the same machine (`docs/BENCHMARKS.md`; different harnesses,
so the two speeds are not a paired ratio).

## Residual add + RMSNorm (one kernel, then into the next consumer)

**What.** A pre-norm block computes `h = h + branch` then `normed = RMSNorm(h)`. One kernel writes both the rounded
residual and the normalized row. The FFN's tail add feeds the next layer's input norm (the final norm after the last
layer); the mixer's tail add feeds the FFN norm.
**Bitwise.** Round the sum to BF16 before the statistics if the reference does; same stride, partial sums, `simd_sum`
and combine order as the parent; vector loads (`bfloat4`) only change memory access, not the arithmetic.
**Effect.** As a standalone kernel, vectorizing it made a prefill-sized call 2.19x faster (8192 x 2048, bitwise).
In decode it is still a one-threadgroup dispatch of ~2.05 us (42 per token, 86 us).
**Into a consumer's prologue.** If the consumer has few threadgroups, let each one recompute the whole norm row in
threadgroup memory and let threadgroup 0 write the residual and the normalized row for later kernels; the residual
goes to the other buffer of a ping-pong pair. Example (LFM2.5): the 64-threadgroup router GEMV recomputes a
2048-element norm in each threadgroup (256 threads, eight elements each). The same trick on a 2048-threadgroup GEMV
would repeat the norm 2048 times; folding the norm into the end of the previous GEMV instead needs a grid-wide
dependency, which failed (Rejected, below).

## Router + top-k in the last-finishing threadgroup

**What.** Every threadgroup of the router GEMV computes its expert's logit; the threadgroup that finishes last runs
the unchanged top-k on all logits. No threadgroup ever waits for another.
**Preconditions.** MSL >= 3.2 (`atomic_thread_fence`); a zeroed counter word in device memory; a small grid.

```metal
if (sg == 0) {
    float sum = simd_sum(router_lane_sum(wrow, normed_row, lane));
    uint last = 0u;
    if (lane == 0) {
        logits[expert] = sum;                                        // publish
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        last = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed) == E - 1u ? 1u : 0u;
    }
    if (simd_shuffle(last, 0u)) {                                    // every other logit is visible now
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        topk_warp(logits, bias_lo, bias_hi, lane, chosen, weight);   // same comparisons, ties, rounding
        if (lane == 0) { /* write ids and BF16 weights */ atomic_store_explicit(counter, 0u, memory_order_relaxed); }
    }
}
```

**Bitwise.** The top-k reads the same F32 logits with the same code as the separate kernel. Keep both fences: removing
them saved ~0.2 us in a micro-benchmark and gives up the memory-model guarantee.
**Verification used.** 20,000 back-to-back dispatches (500 per command buffer, no host waits) over 64 random input
sets with 64 distinct top-k results, every output equal to the parent kernels and the counter back at 0; plus 2,432
captured real layer-steps replayed 16 times each (38,912 dispatches), all equal.
**Cost by grid size.** At 64 threadgroups the fused kernel is 0.46 us faster than router + separate top-k (3.93 vs
4.39 us). At 2048 threadgroups the fence and counter cost ~8 us per dispatch; hierarchical counters did not help.
Use this pattern on small grids only.

## Staging rows in threadgroup memory to overlap latency

A latency-bound kernel that loads its weights only after a prologue pays the DRAM latency serially. Issue the loads
at kernel entry with all threads, into threadgroup memory, and let the prologue's barriers publish them; then read
them with the original lane order. Example (LFM2.5): the router's 8 KB F32 row and its two top-k bias values
staged at entry: behind a 7 MB DRAM stream 6.00 -> 4.75 us per layer, decode +0.4-0.5%. Staging also lets a
multi-threadgroup kernel keep the norm's input and output rows (`srow`, `orow`) local, so only threadgroup 0 writes
them. It does not always help: issuing all of the expert down GEMV's loads early was neutral in the chain.

## Q/K norm + RoPE into the attention prologue

**What.** Per-head Q and K RMSNorm, RoPE and the K-cache write, first as one kernel, then inside the single-pass
decode attention kernel. Example (LFM2.5): in threadgroup `hq`, simdgroup 0 normalizes and rotates query head
`hq`, simdgroup 1 the key row of its KV head; both rows go to threadgroup memory as BF16; the first query head of each
KV group writes the key into the cache. The attention loops run unchanged over cached positions; the position being
written is folded from threadgroup memory by a copy of the same code, so no threadgroup reads a cache row that this
dispatch writes.
**Bitwise.** Keep the BF16 rounding of the normalized value before the rotation; the rotation `lo*c - hi*s`,
`hi*c + lo*s` compiled to a multiply plus an `fma` in the parent, so write `fma(n0, c, -n1 * s)` and
`fma(n1, c, n0 * s)` explicitly. Without that, 2 of 409,600 values differed by one BF16 ulp.
**Verification used.** 5,228 random cases at lengths 1..8191, attention output and the whole K cache identical; 40/40
prompts identical. The prologue costs ~0.5 us of the ~1.6 us dispatch it removes.
**Precondition.** Only where one threadgroup per query head exists (the single-pass kernel); the two-pass long-context
kernel keeps the separate kernel (`attention.md`).

## Projections that share an input

**What.** Rows of several projections that read the same activation run in one dispatch: the threadgroup index picks
the weight set (Q, K, V, or a conv layer's B, C, X), each row keeps its own lane walk and `simd_sum`. Outputs whose
next use is a copy are written to their destination directly (V into the KV cache at the current position). The short
conv's elementwise gates and its three-tap convolution run in the same kernel: one threadgroup per channel, one
simdgroup per projection row, then one thread applies the parent's `bf16(b * x)`, tap order, `bf16(z)` and the state
shift.
**Precondition.** Same quantization format and group size; every row's arithmetic unchanged. Stacking rows into one
bigger matrix for a library GEMM is a different matter: it can move the shape across the library's kernel-selection
thresholds and change the reduction route.
**Effect.** Example (LFM2.5): giving each of the conv's three rows its own simdgroup (instead of one simdgroup
serializing them) and a one-warp top-k took decode from 3.90 to 3.74 ms/token.

## Expert gate/up + activation, and expert down + combine

Fuse the routed experts' gate and up GEMVs with the activation, and the down GEMV with the weighted combine
(`kernels-gemv-quant.md`, Expert gather GEMVs). Keep the reference's roundings: each of gate and up rounded to BF16,
the activation rounded, the product rounded; for the combine, each `bf16(weight * expert_out)` summed in slot order,
rounded once. Do not use float atomics to combine experts: the order would vary.

## Dense FFN gate + up + SwiGLU

**What.** One 32-thread threadgroup per row with two chains per lane (gate and up), the plain GEMV's block walk for
each, and the separate SwiGLU kernel's expression on the BF16-rounded sums. Example (LFM2.5, two dense
layers): gate, up and SwiGLU become one dispatch (-4 per token); micro-benchmark 79.9 -> 76.6 us per layer; a check
build compared the fused activation with the separate kernels' every step: 0 differing words over 7,680 steps.

## LM head with an argmax epilogue, and the token tail

**What.** During greedy decode nobody reads the logits, so the head GEMV does not store them. Each threadgroup merges
its rows' BF16-rounded logits under "value descending, index ascending" and folds the winner into one 64-bit key with
`atomic_max`. A one-threadgroup tail kernel reads and clears the key, writes the token into the ring, and, unless this
is the call's last step, writes the next token's embedding row and its first-layer RMSNorm (`decode-loop.md`).

```metal
static inline ulong argmax_key(float v, uint i) {      // larger key: larger value, or equal value at lower index
    uint u = as_type<uint>(v);
    if ((u << 1) == 0u) u = 0u;                          // -0 ties +0, as float == does
    const uint o = (u & 0x80000000u) ? ~u : (u | 0x80000000u);   // order-preserving float -> uint
    return (ulong(o) << 32) | ulong(~i);
}
// per threadgroup: bv = -INFINITY; merge with (v > bv || (v == bv && i < bi)); then
atomic_max_explicit(best, argmax_key(bv, bi), memory_order_relaxed);
```

**Bitwise.** "Value desc, index asc" is a total order on non-NaN values, so the winner does not depend on how rows are
partitioned; a NaN never wins because both comparisons are false for it (structural, not `isnan`). Round each logit to
BF16 before comparing if the reference's argmax sees BF16 logits. The tail uses the embedding gather's dequantization
expression and the norm kernel's body (same staging, order, `simd_sum`, rounding). Sampling paths keep the plain head.
**Effect.** 5 -> 2 dispatches per token; the tail alone costs 2.78 us isolated; decode +0.35% / +0.45% / +0.84%. A check
build also compared the key's value with the parent's BF16 logit at the chosen token, which catches an elided
rounding.

## Rejected fusions (measured)

| attempt | result | why |
|---|---|---|
| post-FFN add + RMSNorm in the last threadgroup of the expert down GEMV (2048 TGs) and the dense down GEMV | bitwise, but decode 0.861 / 0.847 / 0.856 of the parent | fence + counter ~8 us at 2048 TGs; even with free synchronization the 128-thread epilogue (16.7 us total) lost to a separate 256-thread norm (1.7 us after the GEMV); the dense epilogue wrecked the GEMV's occupancy (54-61 vs 19 us) |
| top-k recomputed inside every expert-gather threadgroup (one dispatch per MoE layer removed) | 239.0 vs 228.2 ms per 64 tokens for one top-k per layer (4-5% slower) | duplicating work across ~3,600 threadgroups per layer costs more than one dispatch |
| router + top-k in one 1024-thread threadgroup | 5-7% slower | likely: 512 KB of weights through one threadgroup |
| norm gain `w_bias + w` computed before the first barrier | not bitwise (26/40 prompts) | backend reassociation with identical IR |
| Q/K norm + RoPE fused without explicit `fma` | not bitwise (tokens changed on real prompts) | diagnosed as different FMA contraction; disabling contraction did not fix it, an explicit `fma` did |
| last-threadgroup router under MSL 3.1 | not delivered | relaxed atomics only, no fence: visibility not guaranteed |

What remains after these (LFM2.5 estimates): the 42 one-threadgroup norms (~86 us/token)
only fold away with a cheap grid-wide dependency, which this platform does not offer on wide grids; the router's
excess over its floor (~116 us/token before the row staging recovered an estimated ~47) and the expert down GEMV's
(~67 us/token) are latency and kernel-shape problems, not fusion problems.
