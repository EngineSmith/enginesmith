# Attention

Purpose: how to build decode attention (one query row against the KV cache) and prefill attention for a GQA decoder
on gfx942: the cost model, the KV cache layout, the single-pass and split-KV (flash-decoding) forms, GQA grouping,
RoPE variants and QK-norm, prefill flash attention with MFMA, the reference's routes, and where to copy kernels from.
Read it before touching any attention kernel, the cache, or the threshold between attention routes. Fusions around
attention are in `kernels-fusion.md`; the prefill pipeline around the tile is in `prefill.md`.

No attention kernel has been measured on this platform yet; numbers are estimates from the model's shapes and the
measured bandwidth in `README.md`, or the reference's end-to-end timings.

## Know the cost first

- **KV bytes per decode token** at context C: `C x attention layers x KV heads x head_dim x 2 (K and V) x bytes per
  element`. Put it in the model card (section 6) beside the weight bytes.
- Example (Qwen3-0.6B: 28 layers, 8 KV heads, D = 128, bf16; estimate): 114,688 B per position; 0.47 GB at 4,097
  positions, 3.76 GB at 32,768, against 1.19 GB of weights. At ~4.4 TB/s the KV read alone is ~0.11 ms per token at
  4K and ~0.86 ms at 32K: above ~10K positions decode is mostly attention.
- Example (the reference, measured, `README.md`): SGLang decode goes from 1.72 ms per token at 63 prompt tokens to
  3.33 ms at 32,768, i.e. ~1.6 ms more for ~3.75 GB more KV read: ~2.3 TB/s effective on the increment (derived). An
  engine whose attention streams the cache near the measured bandwidth has room there too.
- **At short context the kernel is latency-bound**: a few KB of KV per head. Its cost is launch and fixed cost;
  fold it into neighbouring work where possible (`kernels-fusion.md`).

## Reproduce the reference's routes

SGLang on ROCm defaults to its aiter attention backend. Read the backend at the pinned version (in the container
image the reference runs from, not in some other checkout): which kernel serves decode (L = 1 per request), which
serves prefill and extend (a continuation's pending token is computed as a one-token extend on the cached prompt),
the KV-split or partition size of the decode kernel and its conditions, the dtype of the stored partials, and whether
the softmax probabilities are rounded to bf16 before the P.V product (flash-style kernels commonly do). Record every
route with its condition in the model card (section 7). Exact bitwise agreement with aiter's kernels is not
reachable without running them; agreement within near-ties is (`references/correctness.md`). Choosing the same
partition and rounding points narrows the noise.

## KV cache layout and writes

- **Layout** `[layer][kv_head][position][D]` in the reference's cache dtype (bf16 unless the reference quantizes
  it), allocated at the task's maximum context at load. One position of one head is D contiguous elements (256 B at
  D = 128), one head's prefix is one contiguous range: what both the split kernel and the prefill tile stream.
- **Write K and V from the kernels that produce them.** The fused QKV projection writes V rows straight into the cache
  at the current position; K is written after its QK-norm and RoPE by the kernel that computes them. A separate
  scatter is one more kernel.
- **No intra-kernel read of a row being written.** If one kernel writes the current position's K and other blocks of
  the same kernel read it, take it from registers or LDS in the writer and make sure no other block reads that row
  from memory in the same launch (blocks are not ordered).
- **Rows beyond the position are stale, not zero**; loops are bounded by the live length, so rewinds and resets are
  position changes.
- **The stored dtype is arithmetic**: quantizing the cache (fp8) is a correctness change against the reference, not an
  optimization.

## Single-pass decode kernel (short context)

Hypothesis to measure: one block per KV head (all G query heads of the group in it), waves splitting positions.

- **GQA grouping.** Each K/V row is read once for the G query heads that share it. Example (Qwen3-0.6B): 16 query
  heads, 8 KV heads, G = 2. On Metal, folding the group into one threadgroup made decode slower because it cut the
  grid to the KV head count (`backends/metal-rust/attention.md`); here, with 304 CUs, any decode attention grid of 8
  or 16 blocks leaves most of the GPU idle, so the grid comes from splitting positions (next section), and grouping
  is about bytes, not about the grid.
- **Lane mapping for D = 128 on wave64.** One key per wave puts 2 elements in each lane and costs a 6-step reduction
  per key. A 16-lane group per key (8 elements, one 16-byte load each for K and V) runs 4 keys per wave in parallel
  and reduces in 4 steps inside a DPP row. Candidates to measure; both are valid if the softmax is kept per query head.
- **Online softmax in fp32**: per position `s = q.k * scale`, `m' = max(m, s)`, `l = l * exp(m - m') + exp(s - m')`,
  `o = o * exp(m - m') + exp(s - m') * v`. Issue the loads of several positions before folding them in order, so the
  loop is not one dependent chain per position.
- **MFMA for decode** is an option: aiter's decode kernels use matrix instructions with the query rows padded to the
  instruction's 16 rows. With G = 2 most of the MFMA is padding, but the instruction may still be cheaper than shuffle
  reductions; measure.

## Split-KV (flash-decoding) form (long context)

Stage 1, grid `(KV heads, splits)`: split `b` takes a contiguous range of positions (or an interleaved one: choose
by measurement and keep it a constant of the length range), keeps one fp32 online softmax per query head of the
group, and writes its partial output with its max and sum. Stage 2 (a kernel, the next stage of a persistent kernel,
or the last-finishing block of stage 1, `kernels-fusion.md`): for each query head rescale each split's partial by
`exp(m_b - m)`, sum in split order, divide once, round to bf16.

- **Grid size.** Enough splits to fill the device: Example (Qwen3-0.6B, estimate): 8 KV heads x 64 splits = 512 blocks
  at 32K positions, each streaming 512 positions x 2 x 256 B = 256 KB per layer. The split count is a function of the
  length range only, never of timing.
- **Partials** in fp32 (or the dtype the reference stores them in, if you follow its route): `q_heads x splits x D x
  4 B` per layer, 0.5 MB at 64 splits (estimate): small next to the 134 MB of KV per layer at 32K.
- **The route switch** between single-pass and split-KV is a rounding route; put it where the reference switches (if
  it does) or measure both sides and check the mismatch counts at the switch length. Test decode steps that cross it
  during a continuation (`references/correctness.md`, boundary prompts).
- Counterexample from Metal (not measured here): an engine that ported only the single-pass kernel fell to 0.70x of
  its reference at long context. Port every route the workload's lengths need.

## Q/K norm and RoPE

- **QK-norm** (Qwen3 and others): an RMSNorm over D per head on q and k, with its own weight vector of length D,
  applied before RoPE. fp32 statistics; round to bf16 where the reference stores the normalized q and k.
- **RoPE variants**: NeoX style (rotate halves: element `i` pairs with `i + D/2`; Qwen3, Llama) versus GPT-J style
  (interleaved pairs `2i, 2i+1`); partial rotary dims; scaling variants (linear, NTK, YaRN, Llama-3 frequency bands).
  Take the variant, `rope_theta` and scaling from `config.json` and the reference's model code; put them in the model
  card.
- **The cos/sin table** is a rounding point: read whether the reference computes cos/sin in fp32 at run time or reads
  a precomputed cache stored in the model dtype (INFERENCE from vLLM-derived rotary code: often a cache converted to
  bf16; verify in the pinned SGLang). Precompute your table at load in the same dtype and with the same formula
  (`inv_freq = theta ^ (-2i / D)` in fp32, `t * inv_freq`, then cos and sin).
- **Where:** the QKV projection's epilogue or the attention prologue (`kernels-fusion.md`). Write the rotation with
  explicit `fmaf` so its rounding does not depend on contraction.

## Prefill attention

- **A flash-attention tile with MFMA.** Per block: a tile of query rows of one head (64 or 128 rows), K and V tiles
  streamed through LDS, `S = Q K^T` and `O += P V` with `v_mfma_f32_32x32x8bf16_1k` or
  `v_mfma_f32_16x16x16bf16_1k`, online softmax in fp32, P rounded to bf16 for the P.V product if the reference's kernel
  does. Operand layouts for MFMA are in `prefill.md`.
- **Causal tiles:** each query tile visits key tiles from 0 to its diagonal; only the diagonal tile needs the mask.
  A later chunk or a continuation offsets query positions by the cached length.
- **GQA sharing:** the G query heads of one KV head should read the same K/V tiles at nearly the same time: adjacent
  in launch order, or in the same block. Note that consecutive block ids are dispatched round-robin to the 8 XCDs
  (vendor documentation), so "adjacent" blocks sit on different XCDs and do not share an L2; remap `blockIdx` so that
  blocks sharing K/V land on the same XCD (`blockIdx.x % 8` is the XCD), and measure.
- **Cost (estimate):** attention FLOPs per prompt of T tokens are `2 x T^2 x D x q_heads x layers` for a causal mask.
  Example (Qwen3-0.6B): 123 TFLOP at T = 32,768 against 29 TFLOP for the projections; at 32K prefill is attention.
  The reference's 331-338 ms there is ~460 TFLOP/s overall (derived), a substantial fraction of the ~1.3 PFLOP/s
  vendor peak: beating it needs a good tile.

## Where to copy kernels from

Copy source, adapt it to this model's D, G and dtype, keep the licence notice and record the commit (`README.md`,
rule 2). Check each licence file at the commit you copy.

| source | what | licence (check) |
|---|---|---|
| aiter (ROCm/aiter) | the reference's own attention kernels: paged / ragged decode attention in HIP, prefill/extend kernels; some only as prebuilt assembly (`.co`), which you may read but not embed | MIT |
| Composable Kernel (ROCm/composable_kernel) | `ck_tile` FMHA forward (prefill) and split-KV decode, header-only templates you can instantiate for D = 128 | MIT |
| ROCm flash-attention fork | the CK-based backend of flash-attention on ROCm | BSD-3-Clause |
| vLLM ROCm kernels | HIP paged attention for decode (custom ROCm path) | Apache-2.0 |
| Triton kernels (aiter, SGLang, vLLM) | algorithm and tile choices only; port to HIP | MIT / Apache-2.0 |

CK templates are large; instantiate only the configuration you need, compile it in `build.rs`, and check that the
binary does not pick up a library dependency (`runtime.md`, Build).

## Checks specific to attention

- A kernel harness comparing a new attention kernel with its parent (bitwise) and with an fp32 scalar reference
  (tolerance) over lengths 1, 2, 63, 64, 65, ... on both sides of every split and tile boundary, comparing the output
  and the whole cache bytes.
- Prompts and continuations whose decode steps cross each route threshold, and at least one request at the task's
  longest length for every change that touches attention or the cache.
- Throughput at several contexts (for example 63, 4K, 32K): a short-context gain can hide a long-context loss.
