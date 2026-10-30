# Quantized GEMV kernels for decode

Purpose: how to write the matrix-vector kernels that stream the weights in batch-1 decode, using MLX's affine 4-bit
and 6-bit formats as the worked format: the lane layout, compile-time shapes, the bandwidth they reach inside the real
chain, which changes keep the bits and which do not, the variants measured and rejected, and the expert-gather and
F32 router GEMVs of a mixture-of-experts layer. Read it before writing or tuning any decode GEMV.

## The format (worked example: MLX affine quantization)

- **4-bit, group 64.** Codes are U32 `[out, in/8]`, eight 4-bit codes per word, lowest nibble first. Scales and biases
  are BF16 `[out, in/64]`. Element `k` of row `r` is `w = scale[r][k/64] * q + bias[r][k/64]`.
- **6-bit, group 64.** Four values in three bytes (16 values per three words), 48 bytes per group, `in * 6 / 8` bytes
  per row. Decode `q0 = b0 & 63`, `q1 = (b0 >> 6) | ((b1 & 15) << 2)`, `q2 = (b1 >> 4) | ((b2 & 3) << 4)`,
  `q3 = b2 >> 2`.
- **Arithmetic route.** Copy the reference's single-row kernel, not the formula. MLX's `qmv`/`qdot` accumulates the
  integer codes times the BF16 activations, and the activation sum, in F32 over a packed chunk and applies
  `scale * sum(q*x) + bias * sum(x)` to the chunk; it does not dequantize each weight to BF16 first. The worked kernel
  applies it per 32-bit word (eight codes). Example (LFM2.5): switching the L=1 dots from dequantize-then-dot to this
  form cut the teacher-forced steps whose token differed from the reference's, out of 65, from 3 / 4 to 2 / 2 at 256 /
  512 prompt tokens.

## The kernel layout

One simdgroup (32 lanes) per output row. Lane `l` walks 16-byte blocks (one `uint4` = four words = 32 codes) at
`l, l + 32, l + 64, ...`; each word contributes `s * dot(q, x) + b * dot(x, 1)` over its eight elements, the four word
terms of a block are added into the lane's sum, `simd_sum` reduces the lanes, and lane 0 rounds to BF16 and stores.

```metal
static inline float dot_word_q4(uint w, float s, float b, device const bfloat4* xv, uint i) {
    float4 xlo = float4(xv[2 * i]), xhi = float4(xv[2 * i + 1]);
    float4 qlo = float4(float(w & 0xF), float((w >> 4) & 0xF), float((w >> 8) & 0xF), float((w >> 12) & 0xF));
    float4 qhi = float4(float((w >> 16) & 0xF), float((w >> 20) & 0xF), float((w >> 24) & 0xF), float(w >> 28));
    float qx = dot(qlo, xlo) + dot(qhi, xhi);
    float xs = dot(xlo, float4(1.0f)) + dot(xhi, float4(1.0f));
    return s * qx + b * xs;
}
template <uint K> kernel void gemv_q4_k(/* codes, scales, biases, x, y */ uint row [[threadgroup_position_in_grid]],
                                        uint lane [[thread_index_in_threadgroup]]) {
    constexpr uint GS = 64, words = K / 8, blocks = words / 4, bpg = GS / 32, groups = K / GS;
    device const uint4* wrow = (device const uint4*)(codes + (ulong)row * words);
    device const bfloat4* xv = (device const bfloat4*)x;
    float sum = 0.0f;
    for (uint i = lane; i < blocks; i += 32) {
        const uint g = i / bpg;  const float s = float(scales[row * groups + g]), b = float(biases[row * groups + g]);
        const uint4 w4 = wrow[i];
        sum += dot_word_q4(w4.x, s, b, xv, 4*i) + dot_word_q4(w4.y, s, b, xv, 4*i+1)
             + dot_word_q4(w4.z, s, b, xv, 4*i+2) + dot_word_q4(w4.w, s, b, xv, 4*i+3);
    }
    sum = simd_sum(sum);
    if (lane == 0) y[row] = bfloat(sum);
}
```

The grid is one 32-thread threadgroup per row. The 6-bit head of the worked engine packs 16 rows into a 128-thread
threadgroup, four rows per simdgroup sharing each four-value activation packet; each row keeps its own lane and group
order and `simd_sum`, so its value is the single-row kernel's. It streams 213 MB at 586 GB/s in the chain.

A row's bits are a function of: lanes per row, which blocks each lane owns, the order of a lane's adds (including how
the four word terms are grouped), the `simd_sum` tree and every rounding point. Fix all of them at bring-up to match
the reference and never change them for speed.

## Compile-time shapes

Make every decode shape a compile-time constant: `template <uint K>` with one `[[host_name("gemv_q4_k2048")]]`
instantiation per K, `constexpr` group size, `constexpr` expert rows. The block count, the per-block group index and
the row strides become constants and shifts instead of integer divisions ahead of the first weight load; the loop and
its float arithmetic are unchanged. The Rust wrapper checks the shape and keeps the runtime-K kernel for every other
shape.

Measured (LFM2.5; isolated GPU time per dispatch, bitwise identical over 1.5-9.8 M outputs each):

| kernel | runtime shapes | compile-time |
|---|---|---|
| out_proj, K = N = 2048 | 6.27 us | 5.48 us |
| dense gate, K 2048 -> N 11776 | 25.11 | 24.69 |
| dense down, K 11776 -> N 2048 | 28.30 | 27.56 |
| conv in_proj (3 rows per channel) + conv | 14.66 | 14.20 |
| Q/K/V projections | 8.26 | 8.12 |
| expert gate/up gather (4 experts) | 26.05 | 25.71 |
| expert down + combine (4 experts) | 13.47 | 12.57 |

End to end: decode +2.7% / +2.9% / +3.0% at 256 / 1024 / 4096 context (10 interleaved rounds), dispatch count unchanged.

Do not fix the trip count of a float-sum loop whose structure the compiler would then change: fixing the 6-bit head's
16-packet group loop changed 388 of 2.56 M outputs and was slower (366 -> 398 us); a fixed four-term expert combine
loop changed 78,517 of 327,680 (fast-math reassociation, README rule 3). Fix integer arithmetic; keep runtime bounds on
float loops unless the bits are verified.

## Bandwidth: in the chain vs isolated

In-chain cost per dispatch (LFM2.5 at P=256; marginal cost of a cold duplicate inside the real step):

| kernel | MB read | us | GB/s |
|---|---|---|---|
| 6-bit head + argmax epilogue | 213.0 | 363.3 | 586 |
| dense gate + up + SwiGLU | 27.14 | 48.94 | 554 |
| expert gate + up + SwiGLU (4 experts) | 14.16 | 27.46 | 516 |
| conv in_proj + conv | 7.11 | 14.40 | 494 |
| dense down, K 11776 | 13.59 | 28.33 | 480 |
| expert down + combine (4 experts) | 7.09 | 15.72 | 451 |
| out_proj, K 2048 | 2.36 | 5.41 | 437 |
| Q/K/V projections | 3.54 | 8.34 | 425 |
| F32 router (+ norm + top-k) | 0.54 | 5.91 | 91 |

Bandwidth tracks bytes per dispatch: every serial dispatch pays a ramp and drain of roughly 1-3 us (estimate) that a
small GEMV cannot amortize. A useful floor per dispatch is `bytes / 590 GB/s + 1.95 us`.

Isolated micro-benchmarks mislead when the working set fits the system-level cache (~48-56 MB): the expert down GEMV
measured 12.57 us cycling 57 MB of weights but 15.72 us in the chain. With more than ~70 MB of distinct weights
(32 slots x 7 MB) the micro-benchmark matched the chain (15.44-15.52 us), yet its
deltas still did not predict in-chain deltas (early loads: -4.5% isolated, neutral in the chain). Use micro-benchmarks
to rank ideas and to probe bounds; accept changes only on an in-engine A/B.

Bound probes (wrong values, for bounds only): removing the math saves 4% on out_proj and 2% on the expert gather
isolated, 2.4 us per expert-down dispatch and ~1 us per expert gate/up dispatch in the chain; doubling the math costs
+3.3 us on expert down. So the in-chain expert down GEMV sits at a memory/ALU balance point, while gate/up is
memory-bound.

## What is not bitwise-safe

| change | evidence (LFM2.5) |
|---|---|
| lanes per row (16 or 64 instead of 32) | changes the per-lane partial sums and the `simd_sum` tree by construction |
| splitting K across simdgroups or threadgroups | changes the reduction tree by construction |
| several rows per simdgroup | identical in principle when each row keeps its lanes and order (the x4 head; the x2/x4 Q4 GEMVs), yet two expert-down variants differed: with early loads 3 of 19.7 M outputs; in a row loop 0 of 1.3 M synthetic outputs but a token mismatch in the real chain |
| block-to-lane assignment, or moving a block's partial result to another lane to balance lanes | 70 of 2.6 M differ (`bal2`) |
| unrolling a float loop to a fixed trip count | see above |
| evaluating an operand earlier (hoisting) | identical IR, different backend association |

The compiler front end factors each block as `(s * QX + sum) + b * XS`, with `QX` and `XS` chained over the eight
`dot` terms, so a block's contribution is not a separable value: moving raw
loads between lanes kept the bits (0 of 2.6 M, `lbal2`), moving computed partials did not.

Verified bitwise in the worked engine (still verify each time): compile-time integer constants with the same loop;
load issue order (all of a lane's loads first: 0 of 19.7 M); rows per threadgroup when each row's lanes and order are
unchanged; 32-bit row offsets. A clean synthetic check is necessary, not sufficient: run the real chain too.

## Rejected variants (measured; do not repeat without a new reason)

| variant | result |
|---|---|
| 2 or 4 rows per simdgroup or threadgroup, shared activation loads (bitwise identical) | decode 257.6 -> 264.9 ms (x2) and 266.0 ms (x4) per 64 tokens; in another session +2.2% to +10.2% slower; with the expert gathers too, 223.6 -> 260.3 ms. More accumulator chains cost occupancy |
| precomputed per-word activation sums in the expert gate/up gather (bitwise identical) | 223.2 -> 226.2 ms: not ALU-bound |
| rows per threadgroup 1/2/4/8 (expert gather), 2/4 (out_proj) | neutral (within +-0.8%) |
| 16-byte activation loads | 5.74 vs 5.47 us at N 2048 (slower), neutral at N 8192 |
| loads hoisted into arrays before the math; explicit fixed trip count plus tail | 6.08 and 6.20 vs 5.46 us |
| 32-bit row offsets in expert down | -3% isolated, decode 0.997-0.999 end to end |
| expert down: all loads issued early; combine weights loaded at start | -4.5% isolated; in the chain +0.4 to +8.2 us per step (neutral) |
| expert ids in the constant address space; magic-number nibble-to-float decode | neutral isolated, +12.5 us per step in the chain; 16.02 vs 15.49 us |
| last gate/up threadgroups prefetch the next routed down rows into the cache | +12 to +72 us per step |
| conv projections in one interleaved simdgroup; dense down manually unrolled 2x | 14.55 vs 14.10 us; 28.30 vs 27.52 us |

Conclusion of that work: after compile-time shapes, kernel-internal GEMV changes were exhausted; the remaining cost
is the fixed cost per serial dispatch, which only fewer dispatches reduce (`kernels-fusion.md`).

## Expert (MoE) gather GEMVs

The router writes the selected expert ids and their BF16 weights to a small device buffer; the gathers read them at
kernel start (`expert = ids[slot]`, weight row `expert * N + row`). The `ids` load sits in front of every weight
address; removing that dependency (a probe with wrong values) saved ~0.5 us on gate/up and ~0.9 us on down, isolated.

- **Gate + up + activation:** a grid of `top_k * N` 32-thread threadgroups, one row of one slot each. Each lane runs
  two chains (gate and up) over the same blocks with the plain GEMV's order; lane 0 applies the activation with the
  reference's roundings. Example (LFM2.5, SwiGLU): `gv = bf16(gsum); a = bf16(gv / (1 + exp(-gv)));
  act = bf16(a * bf16(usum))`.
- **Down + combine:** a grid of `N` threadgroups of `32 * top_k` threads; simdgroup `r` computes the row for slot `r`;
  the BF16-rounded partials go to threadgroup memory; after a barrier one thread sums `bf16(weight[r] * partial[r])`
  in slot order in F32 and rounds once. Keep `top_k` a runtime bound (a fixed four-term loop was reassociated).
- **Route once per layer.** Recomputing the top-k inside every gather threadgroup was slower (239.0 vs 228.2 ms per 64
  tokens). Put the top-k in the router (`kernels-fusion.md`).

Example (LFM2.5): the expert GEMVs are 76 dispatches and ~1641 us of a 3280 us step (50%), at 451-516 GB/s.

## The F32 router GEMV

A router matrix is small (example LFM2.5: 64 experts x 2048 F32 = 512 KB) and the kernel is latency-bound (91 GB/s in
the chain; floor 2.86 us, measured 5.91 us). Use one threadgroup per expert; let one simdgroup sum the row in
increasing `k` per lane; issue many loads per lane before the adds (16 in the worked engine) without changing the add
order; unroll with a compile-time K; and stage the row into threadgroup memory at kernel entry so its DRAM latency
overlaps the norm prologue (`kernels-fusion.md`). Keep logits, sigmoid and selection in F32 if the reference does.

## Checklist for a new decode GEMV

1. Copy the reference's per-group formula, accumulation dtype and rounding points.
2. Fix lanes per row and block-to-lane order to match the reference route; record them in the model card.
3. Compile-time K, group size and rows for every decode shape; the wrapper checks shapes, dtypes and alignment.
4. Bitwise check against the parent kernel: random inputs over many rounds (include the last group of a row and the
   last row of an expert), captured real activations, then end-to-end tokens.
5. Time it inside the engine, interleaved against the parent.
