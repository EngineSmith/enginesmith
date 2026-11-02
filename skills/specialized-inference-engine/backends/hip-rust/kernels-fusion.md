# Kernel fusion for decode

Purpose: how to remove kernel boundaries from a batch-1 decode step on gfx942 without changing a single output bit
relative to the parent: the rules that keep a fusion bitwise identical, a catalog of fusions that need no grid-wide
dependency, the ones that do, how to build a grid-wide barrier correctly across the 8 XCDs, and the persistent
decode megakernel. Read it once the kernel count per token is known (`decode-loop.md`): on this machine every kernel
boundary costs on the order of the work of one small projection (`README.md`), so fusion is the main lever for a small
model, earlier than it was on Metal.

Nothing in this file has been measured on this platform yet. Mechanisms are from the Metal backend
(`backends/metal-rust/kernels-fusion.md`, measured there) or from the HIP and LLVM AMDGPU documentation; each is a
hypothesis here until you measure it, and every cost must be measured in your own chain.

## Rules that keep a fusion bitwise identical

1. **Keep every rounding point.** Where the parent stored a bf16 tensor and the next kernel reloaded it, round to
   bf16 at the same point in a register or in LDS (`f32_to_bf16`, then widen). Removing a cast removes a rounding.
2. **Copy each producer's arithmetic verbatim:** the same expressions, statement order, lane-to-element mapping,
   reduction tree and shuffles. Call the same `__device__` helper from the fused and the separate kernel.
3. **Write FMAs explicitly.** hipcc contracts `a*b + c` across statements by default (`README.md`, rule 4), and what
   it contracts depends on the surrounding code. Write `fmaf` where the parent's code computed a product-sum, so the
   fused copy cannot compile differently. Check the assembly when in doubt.
4. **No in-place writes that other blocks of the same kernel read.** Write to a second buffer and swap (ping-pong);
   let exactly one block write each shared output.
5. **Redundant work must be identical work.** If several blocks recompute the same value (a norm statistic), they
   run the same code on the same input, so they get the same bits; check it anyway.
6. **Verify three ways, then end to end:** a kernel harness (fused against parent kernels, random inputs, edge shapes,
   last rows), captured real activations replayed through both, and a check build that runs the parent kernels into
   shadow buffers every step and fails on any differing word. Then `scripts/same.py` on many prompts and shapes.
   Random inputs alone are not enough (on Metal one fusion passed random tensors and four prompts and still changed
   tokens on others).

## Fusions without a grid-wide dependency

The dense decoder's dependency chain per layer is: input norm (needs the whole residual row) -> QKV -> QK-norm, RoPE,
cache write -> attention (per KV head) -> O (needs every head) -> residual add -> post-attention norm (whole row) ->
gate/up -> SiLU * up -> down (needs every intermediate) -> residual add. Elementwise steps fold into the epilogue of
their producer; whole-row reductions of a short row fold into the prologue of their consumer.

| fusion | kernels removed per layer | mechanism | precondition |
|---|---|---|---|
| RMSNorm into the next GEMV's prologue | 1 per norm | each block recomputes the row's sum of squares and normalizes into LDS or registers | the row is short (example Qwen3-0.6B: 1024 elements, 2 KB) |
| residual add into the O and down epilogues | 1 per add | each output element: `bf16(residual + bf16(gemv))` with the reference's rounding order | elementwise; ping-pong the residual buffer |
| gate/up + SiLU * up | 1-2 | interleaved gate/up rows, two accumulators per lane group, epilogue applies the activation | gate and up rows owned by the same lanes |
| QK-norm + RoPE + K write into the attention prologue | 1-3 | each attention block normalizes and rotates its G query heads and the current K row in LDS | the head (D = 128) is computed by one block |
| V write into the QKV epilogue | 1 | V rows go straight to the cache at the current position | V needs no norm |
| LM head + argmax, token tail + next embedding + first norm | 2-3 per token | `decode-loop.md` | greedy path only |

Example (Qwen3-0.6B, estimate): these take a ~12-kernel layer to 5 kernels (norm+QKV+V write; attention with its
prologue; O+add; norm+gate/up+SiLU; down+add), ~340 to ~143 kernels per token, ~0.34 ms less boundary cost at ~1.7 us
per boundary (`decode-loop.md`). Not measured.

**RMSNorm in the prologue.** Each block reads the residual row (2 KB) from L2, sums squares in a fixed order (the same
helper in every block), and scales. The redundancy is cheap when the row is short and the block count moderate; for
the LM head (thousands of blocks) the repeated norm may cost more than a separate kernel: measure, or let the
token tail compute the final norm once. The reference's fused add + RMSNorm kernel decides the rounding order: read
it (aiter or sgl-kernel at the pinned version). Example of what to look for (vLLM's CUDA `fused_add_rms_norm`, read in
its source): it adds in fp32, stores the bf16-rounded sum as the new residual, takes the variance from the fp32 sum
and normalizes the rounded residual; HF's reference module rounds the normalized value to bf16 before multiplying by
the weight. Such details decide near-ties; copy the one SGLang runs.

**Gate/up + SiLU.** With rows interleaved at load (`kernels-gemv-quant.md`), a lane group computes gate row `i` and
up row `i` over the same activation chunks; the epilogue applies the reference's expression on the bf16-rounded sums,
for example `g = bf16(gsum); a = bf16(g / (1 + exp(-g))); out = bf16(a * bf16(usum))` if that is what the reference's
`silu_and_mul` kernel rounds (check which transcendental and which rounding it uses).

**QK-norm + RoPE in the attention prologue.** In each attention block: normalize and rotate the G query heads it serves
and the current position's K row (128 elements each), stage them in LDS as bf16 exactly as the separate kernel stored
them, use them for the current position, and let exactly one block per KV head (split 0) write K to the cache. No
block reads that cache row from memory in this launch. On Metal this fusion removed 10 kernels per token and gained
0.6-0.8% decode (`backends/metal-rust/kernels-fusion.md`); here the boundary is relatively more expensive.

## Fusions that need a grid-wide dependency

O needs every head's attention output, down needs every intermediate, and the next layer's norm needs every element of
the residual. Fusing across those points needs all blocks of one stage to finish before any block of the next starts
reading. Three ways, in order of risk:

1. **Last-finishing block.** Every block publishes its output, does a release fence at agent scope, increments a
   counter; the block that sees `count == nblocks - 1` acquires and runs the epilogue (combine split-KV partials, the
   argmax over block partials). No block waits for another, so forward progress never depends on co-residency. On
   Metal the fence-and-counter cost grew with the grid (~8 us at 2048 threadgroups, a win at 64); measure here by grid.
2. **Cooperative launch with a grid barrier.** `hipModuleLaunchCooperativeKernel` (or `hipLaunchCooperativeKernel`)
   guarantees that every block is resident, or fails the launch if the grid is too large. Size the grid with
   `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor x 304`. Then a barrier is legal: HIP's cooperative groups
   (`#include <hip/hip_cooperative_groups.h>`, `cooperative_groups::this_grid().sync()`, device code only, no host
   library) is the correct baseline; verify that it works with the module launch API on your ROCm version.
3. **A hand-written barrier** (after measuring the cooperative-groups one):

```cpp
// count and gen: zeroed device words. Every thread fences its own writes before the block barrier.
__device__ void grid_barrier(unsigned* count, unsigned* gen, unsigned nblocks) {
    __threadfence();                                              // agent scope: this thread's writes visible
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned g = __hip_atomic_load(gen, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        if (__hip_atomic_fetch_add(count, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) == nblocks - 1) {
            __hip_atomic_store(count, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(gen, g + 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        } else {
            while (__hip_atomic_load(gen, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) == g) __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();
}
```

   Only legal under a cooperative launch (or another proof that every block is resident). Stress it with thousands of
   back-to-back barriers and compare every output with the unfused chain.

**Cross-XCD visibility is the cost to understand.** The MI325X has 8 XCDs, each with its own L2; ordinary device
memory is not kept coherent between those L2s. Per the LLVM AMDGPU memory model (GFX942 section), an agent-scope
release writes back dirty L2 lines (`buffer_wbl2`) and an agent-scope acquire invalidates (`buffer_inv`), so a block on
one XCD sees another XCD's writes only through these fences. Their cost grows with dirty data and with the number of
blocks fencing; read-only weight traffic leaves nothing dirty. Options to measure: fence as above; allocate the small
activation and flag buffers as fine-grained or uncached memory (`runtime.md`) so they bypass the L2; keep each
stage's producer and consumer on the same XCD where the dependency allows (block id modulo 8 is the XCD, vendor
documentation).

## The decode megakernel (persistent kernel)

One cooperative launch per decode step (or per request), a grid of `occupancy x 304` blocks that walks the stages of
the whole step with grid barriers between them. Hypotheses, in the order to test them:

- **Barrier cost decides everything.** Example (Qwen3-0.6B, estimate): 4 all-to-all barriers per layer (before O,
  before gate/up, before down, before the next layer's QKV; QKV -> attention is a per-head dependency that per-head
  counters can serve) and ~2 for the head and tail: ~115 barriers per token. At 2 us per barrier that is 0.23 ms, no
  better than the fused launch design; at 0.5 us it is 0.06 ms. Measure a barrier-only persistent kernel at 304, 608
  and 1216 blocks first.
- **Weight loads do not depend on the barrier.** Each block knows which weight slice it streams in the next stage and
  can issue those loads before it waits. Example (Qwen3-0.6B, estimate): a block's share of one stage is small (8.4 MB
  of QKV over 304 CUs is ~28 KB per CU), so a whole stage's slice fits in registers or LDS (64 KiB per CU). If the
  loads for stage s + 1 are in flight during barrier s, the barrier hides behind HBM streaming and the step can
  approach the weight-streaming floor. This is the central idea of published decode megakernels on other GPUs; it is
  untested here.
- **Work assignment is a constant of the shape**: block `b` owns the same rows in every stage and every step, so the
  reduction order is fixed and the result is deterministic. Never assign work by atomics on a queue whose order
  depends on timing, unless the arithmetic of each item is independent of who computes it (it usually is: then
  dynamic assignment is safe for bits, but check).
- **Per-request persistence** (the kernel loops over all `max_tokens` steps, reading positions and writing tokens in
  device memory) removes the last launches but holds the GPU for the whole request; the host only waits once. Try it
  only after the per-step megakernel works.
- Keep the fused multi-kernel path as the reference for bits: the megakernel must produce exactly its outputs.

## Rejected fusions

None measured on this platform yet. Record each attempt with its numbers in `counterexamples.md`; the Metal backend's
rejected fusions (`backends/metal-rust/kernels-fusion.md`) are worth reading as mechanisms that can fail (occupancy
collapse from a heavy epilogue, fence cost on wide grids, duplicated work across thousands of blocks).
