# Prefill

Purpose: how to make prompt processing fast without leaving the reference's rounding. Prefill is compute-bound
matrix work: use the M5 GPU's matrix units through Metal 4 tensor operations, keep the reference's chunk partition,
and group the routed experts' rows by expert so each expert's weights are read once per row tile. Read it before
changing a prefill GEMM, the chunking, or the MoE prefill path. Prefill attention (the flash tile and its launch
order) is in `attention.md`; timing methods are in `profiling.md`.

Numbers are measurements from the LFM2.5 worked engine on an Apple M5 Max unless they are marked as estimates.

## Judge prefill kernels in TFLOP/s

- FLOPs per prompt token are about `2 x active parameters` plus attention (`4 x t x D` per query head at position
  `t`, so `2 x T x D` per head per token on average over a causal prompt of T tokens). Compute them in the model card
  (section 6).
- Measure the ceiling yourself: the plain BF16 tensor-op GEMM tile on your production shapes. Judge every prefill
  kernel against it, and the elementwise kernels against the streaming bandwidth.
- Example (M5 Max, measured): a 128x128 dense BF16 tile 52 TFLOP/s on an 8191-row chunk; the conv mixers' GEMMs
  52-62; the grouped Q4 expert GEMM 47-50 in micro-benchmarks of the production shapes but ~37 in the engine at an
  8191-row chunk (before the one-stripe tile below); the flash attention tile ~35 before and ~61 after its
  launch-order fix (`attention.md`).
- Example (where the time went, measured with operator-skip or per-pass timing builds): at 2048 tokens (215 ms)
  expert GEMMs + combine 63.5%, conv projections 17.0%, MoE glue (router, counting sort, gather) 6.4%, attention
  4.8%, norms 1.4%; at an 8191-row chunk (778 ms) MoE 498 ms, conv 138, attention 80, dense FFN 25; at 131,008 tokens
  (34 s instrumented) attention ~20 s, MoE GEMMs ~10 s, conv 2.9 s, dense FFN 1 s.
- Elementwise kernels are not free at prefill sizes. Example (measured): a scalar `add_rmsnorm` over
  8192 rows moved 134 MB at 111 GB/s (1.2 ms per call, 80 calls per chunk); moving eight elements per thread as
  16-byte accesses, with the same reduction order, made it 0.548 ms (2.19x, bitwise identical). Measure each one: an
  eight-element `silu_mul` was 7% slower.

## Matrix units on M5 (Apple GPU family 10)

- Family 10 GPUs run Metal 4 tensor operations (MSL 4.0, `MetalPerformancePrimitives`, `matmul2d`) on native matrix
  units. MLX's "NAX" kernels use them: `quantized_nax.h` (`qmm_t_nax`), `steel/gemm/gemm_nax.h`,
  `steel/attn/kernels/steel_attention_nax.h`. The LFM2.5 work treated these as the reference's prefill route on the
  M5; read MLX's dispatch code at your pinned version to confirm which kernel it picks per shape.
- Detect with `supportsFamily(MTLGPUFamilyApple10)` and probe the MSL 4.0 compile at startup (`README.md`, rules 4-5).
  The worked engine refuses family < 10 with one message; its source notes say family 9 (M3/M4) runs tensor ops
  through emulation (not measured).
- **Keep a plain kernel beside each tensor-op kernel** (MLX's non-NAX steel templates on `simdgroup_matrix`): for
  bring-up, as a numerical cross-check, and as the fallback route. A fallback rounds differently, so it is a
  separately verified route chosen by device family only, never a silent substitute.
- **Generate, do not hand-port.** A script flattens the pinned MLX headers into one embedded MSL file and adds your
  `[[kernel]]` entry points (Example: `tools/gen_quant_nax.py`, `tools/gen_steel_attn.py`). Never edit the generated
  file; apply any change (launch-axis remaps, guards) in the generator so it is reviewable and reproducible.
- **Launch geometry must match the template.** Example (measured): a launcher gave 128 threads to 8-simdgroup
  (256-thread) kernels, so B-tile staging skipped every other 32-value chunk; the kernel ran and produced plausible
  output (34 of 65 teacher-forced mismatches at 32,768 tokens). Assert threads per threadgroup and tile constants
  against the template when you create the pipeline.
- **Copy the reference's tile mechanics, not only its math.** Example (measured): replacing a compact
  64x64 BF16 B tile with MLX's `qmm_t_nax_tgp_impl` body (padded B stride `BK + 16/sizeof(T)`, SK = 32
  `tile_matmad_nax` slices) gave 1.04-1.14x prefill at 256-8192 tokens with outputs bitwise identical (0 differing
  values on four synthetic shapes; 16 cases token-identical).

## Keep the reference's chunk partition

- Take the partition from the reference's driving loop, not its documentation. Example (mlx-lm `generate_step`):
  chunks of `min(2048, remaining - 1)` tokens, then the last prompt token alone as an L=1 step.
- **Departing from it changes rounding where it hurts.** Example (measured): an 8192/4096/2048 chunk ladder put the
  last token of a 512-token prompt into a 512-row chunk instead of an L=1 step, and the engine mismatched at step 0
  (reference gap 0.94). Restoring the partition lowered prefill speed; that was accepted for correctness. See
  `references/correctness.md`.
- **Merge chunks, do not change them.** Several consecutive reference chunks may share one submission when every
  row's arithmetic is independent of the submission size: the same causal key range and key-block order per query
  tile, the same K order per GEMM output, the same tile rows, and per-chunk state (convolution history, recurrent
  state) carried exactly across internal boundaries. Never merge the final L=1 step. Example (measured): merging
  up to 8192 rows was bitwise identical on 39 cases; whole request 1.15x at 4096 tokens, 1.24x at 8192, 1.10x at 16K,
  1.06x at 32K, 1.00x at 131K (attention-bound; loaded machine, paired).
- **Choices made by chunk size must change grouping only.** A tile picked by row count (for example a 32-row tile up
  to 512 rows) may change how rows are grouped, never the per-output K order; check it bitwise on both sides.
- **One input buffer per in-flight chunk.** Example (measured, rejected): deferring the per-chunk wait so the host
  could encode the next chunk rewrote the token buffer before the GPU had read it; every token of the second chunk
  diverged. Overlap host encode with GPU work only through buffers the GPU no longer
  needs.

## Grouped expert GEMMs (MoE)

Per chunk, all on the GPU, in this order:

1. Router GEMM and top-k per token, with the reference's selection rule, dtype and tie order.
2. Count routed rows per expert (integer histogram), exclusive prefix sum, scatter (token, slot) ids in expert order.
   Slot assignment by integer atomics is safe only if nothing downstream depends on slot order; otherwise use a
   stable scan (`references/correctness.md`).
3. Build a block map on the GPU: one entry per (expert, row tile, column tile) with its row range; zero the unused
   entries. The map's row granularity is the tile's row height.
4. Gather the routed rows; grouped GEMM gate and up over the map (each expert's weight tile is staged and dequantized
   once per row tile); the activation with the reference's intermediate BF16 roundings; grouped GEMM down.
5. Combine each token's k expert outputs in a fixed slot order with the reference's per-product rounding.

- **Match the reference's MoE route by token count.** Example (mlx-lm 0.32 `SwitchGLU`, read in source): it sorts by
  expert only from 64 indices (16 tokens at top-4); below that it runs an unsorted gather.
- **Tiles are a padding trade.** Each expert's last row tile is partly empty, and every tile stages a full `BN x K`
  weight tile. Example (LFM2.5, 64 experts, top-4, measured): at a 2048-row chunk each expert owns ~128 rows. A
  128-row tile that stages the weights once for two 64-row sub-tiles, and skips the MMA of an empty sub-tile, gave
  +3.8% to +5.8% prefill at 2048-16384 tokens; one 64-column stripe per threadgroup (the activation tile read once per
  tile) another +0.2% to +5.8%; chunks of at most 512 rows keep a 32-row tile. Wider or deeper tiles (BN 128, BK 128,
  128x128, 256 rows, 8 simdgroups) were neutral or slower (register pressure, occupancy).
- **Dense projections: dequantize once per chunk.** A fused dequantizing GEMM re-dequantized each 64x64 weight tile
  once per 64-row output tile (128 times at 8192 rows): ~10 TFLOP/s against ~42 for the expert GEMM. Bulk-dequantizing
  each weight once per chunk into a BF16 scratch, then the BF16 tensor-op GEMM, writes the same BF16 values and keeps
  the K loop, so it is bitwise identical. At <= 512 rows the fused form was neutral.
- **Persistent BF16 copies of the weights lost.** Experts in BF16 (36 GB) with a whole-K GEMM: 2.1x faster isolated
  (77.8 vs 36.5 TFLOP/s) but 998 vs 998 ms at 8192 tokens in the engine (cold weight stream); dense projections in BF16
  (1.5 GB): -3.8% at 256 tokens but +0.4% / +1.1% at 2048 / 8192 (the per-chunk scratch stays in cache).

## Measured prefill against the reference (Example)

LFM2.5-24B-A2B, MLX 4-bit, M5 Max; released engine vs mlx-lm 0.32.0 on MLX 0.32.3, same weights, same nonce prompts
(`docs/BENCHMARKS.md`; medians, two opposite-order passes):

| prompt tokens | 256 | 1K | 4K | 8K | 32K | 128K |
|---|---:|---:|---:|---:|---:|---:|
| engine, tokens/s | 4243 | 8032 | 10361 | 10360 | 8445 | 4662 |
| mlx-lm, tokens/s | 3800 | 6981 | 8829 | 8565 | 7124 | 4032 |
| ratio | 1.12 | 1.15 | 1.17 | 1.21 | 1.19 | 1.16 |

A separate paired evaluation (`generate(P, 1)` timed as prefill, five alternating rounds, 120 s cool-downs) measured
prefill speedups of 1.01x at 256 tokens, 1.12x at 1K, 1.23x at 8K and 16K, 1.12x at 128K. Inference (not measured):
the margin is small because both sides run MLX's own tensor-op tiles near the same ceiling; the gains came from
merged submissions, launch order, padding and glue kernels, not from faster matmul arithmetic.

## Not tried (no record of it in the LFM2.5 work)

- A 16-row simdgroup tile (`WM = 8, WN = 1`) for the grouped expert GEMM to halve its padding at 2048-row chunks:
  considered and left for a quieter machine (padding estimated at ~3% of prefill).
- Concurrent encoders for independent prefill work (for example the dense FFN's gate and up GEMMs).
- Host encode of chunk n+1 overlapped with chunk n through double-buffered inputs (the attempt above failed for lack
  of separate buffers, not on its merits).
- Fusing the expert row gather into the grouped GEMM's activation load, or the combine into the down GEMM's epilogue.
- Sharing K/V tiles between the group's query heads inside one prefill threadgroup (the launch-order change obtains
  the reuse through the cache instead).
