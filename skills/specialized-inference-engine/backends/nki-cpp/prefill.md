# Prefill on Trainium2

Purpose: whole-model prefill NEFFs (GEMMs on the TensorE, flash attention, the KV cache written for decode).
Measured by the kernel lab on one trn2.48xlarge (nki 0.6.0, neuronx-cc 2.27.5334, logical NeuronCore config 2,
2026-11-03) for Qwen3-0.6B bf16, queued from C++ (whole model including embedding, LM head and argmax), against
HF transformers on the CPU; not yet through an EngineSmith evaluation. Public NKI API and nkilib (Apache-2.0) only.

## Result (Qwen3-0.6B)

| prompt bucket | prefill | the reference (NxD Inference, whole request) | TFLOP/s |
|---|---|---|---|
| 256 | 7.5 ms (one 28-layer NEFF: 7.3 ms) | 11 ms | 31 |
| 1,024 | 19.8 ms | 36 ms | 52 |
| 4,096 | 76.3 ms (layered: 3 NEFFs, the layer NEFF run 28 times) | 154 ms | 73 |

First token equal to HF fp32 at 256 and 1,024; prefill plus 31 decode steps reproduce HF fp32 greedy at 4,093
tokens.

## Structure

- **Feature-major activations** (`h^T [128, 8, S]`): GEMM = weight tile `[128 k, 128 n]` stationary, activation
  chunk `[128 k, 512 tokens]` moving, PSUM `[128 n, 512]`. No activation transposes between GEMMs; q^T and k^T come
  out in nkilib's `tp_q=False` / `tp_k=False` layouts and in the decode K^T cache layout; V needs one TensorE
  transpose per 128 x 128 block (a matmul against the identity, exact).
- Per layer, five phases over 512-token chunks with that phase's weights resident in SBUF, `nisa.core_barrier`
  between phases: A norm + QKV + q/k norm + RoPE + K/V cache writes; B attention; C o_proj (split along N) +
  residual; D norm + gate/up + SiLU x up; E down (split along N) + residual.
- **Attention: nkilib `core/attention/attention_cte.py` (`_attention_cte`)** inlined with `tp_q=False, tp_k=False,
  tp_out=True`; it splits the heads over the two cores itself. Standalone: 73 us at 256 (3.7 TFLOP/s), 250 us at
  1,024 (17), 1,384 us at 4,096 (49.6); max error 0.008 against fp32.
- **Layered prefill for long prompts**: an embedding NEFF, one layer NEFF executed for every layer with each
  per-layer tensor bound to a slice of the stacked tensor (`nrt_tensor_allocate_slice(src, l * size, size, ...)`), a
  head NEFF, all queued. It costs 0.1-0.2 ms more than one NEFF and builds in 37 s at 4,096, where unrolling 28
  layers did not finish tracing in 10 minutes.
- Measured GEMM rates at 4,096 tokens (queued, about 20 us fixed cost included): 130.5 TFLOP/s (1024 x 4096),
  129.7 (1024 x 6144), 103.5 (3072 x 1024), 95.8 (2048 x 1024).

## What mattered at 4,096 (neuron-explorer)

1. Fewer VectorE passes (squares on the ScalarE from PSUM, sums of squares as matmuls): no gain (88.8 ms); the
   bottleneck moved to the GpSimd.
2. The GpSimd was issuing about 48 small DMAs per chunk (the default `dma_copy` is a software DGE driven by it).
   Activation DMAs on hardware DGE (`dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync`): **88.8 -> 76.3 ms**
   at 4,096, 22.3 -> 19.8 ms at 1,024. (Keep big *weight* streams of the decode step on the default path: there
   hardware DGE was slower, 2.378 vs 2.235 ms.)

## Pitfalls

- Internal `nl.shared_hbm` scratch tensors were not shared between the two cores in the prefill kernel (each core
  read back only its own writes, despite `core_barrier`). Pass every cross-core scratch tensor as a kernel input
  (a caller-allocated device buffer).
- Wrapping a `@nki.jit` library kernel in another through a Python conditional traced to an empty kernel
  (`NCC_INKI003 ... got 0 outputs`): call the undecorated function.
- In nki 0.6 `nl.affine_range` and `nl.sequential_range` are `range` (fully unrolled); only `nl.dynamic_range` and
  `fori_loop` are device loops. Unrolling 28 layers x 8 chunks x the attention made a 190 MB MLIR (162 s build).

## Qwen3-8B: layered prefill (measured)

Embedding NEFF + 36 x one-layer NEFF on per-layer tensor slices + head NEFF, queued: **37.8 / 129.0 / 552.7 ms** at
256 / 1,024 / 4,096 tokens (112 TFLOP/s at the longer two; the reference: 67 / 243 / 606 ms); prefill plus 15-23
decode steps reproduce HF fp32 greedy. Token blocks of 512 (256 for gate/up and down) stream each weight slab once
per block, since one core's QKV weights for one layer do not fit SBUF; o_proj and down split along K with the decode
weight layouts and the per-core partial results exchanged through a caller-provided scratch tensor plus
`core_barrier`, so no extra N-split weight copies (+4.8 GB would not fit next to 16.4 GB of weights in 24 GiB).

## Qwen3-8B components

QKV 4096 -> 6144 at 1,024 tokens: 136.6 TFLOP/s; gate/up half: 145.6; o_proj: 129.4; down 12288 -> 4096 with resident
activations does not fit SBUF (`NCC_IBIR229`). nkilib attention, 32 q / 8 KV heads: 479 us at 1,024, 2,732 us at
4,096. One core's QKV weights for one layer (196 KB per partition) exceed SBUF, so phases must stream weight slabs
per token chunk. Estimate (not measured): about 125 ms at 1,024 and 520 ms at 4,096 (the reference: 243 and 606 ms).
