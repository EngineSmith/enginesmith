# Kernel fusion on Trainium2: a whole decode step in one NEFF

Purpose: how to run all layers of a decode step, the LM head and the argmax as one NEFF execution, the layout
that needs no transposes between layers, and how the two physical cores of a logical NeuronCore split a layer.
Measured by the kernel lab on one trn2.48xlarge (nki 0.6.0, neuronx-cc 2.27.5334, logical NeuronCore config 2,
2026-11-03) for Qwen3-0.6B bf16; public NKI API only. The lab measured it with its own C++ driver and a CPU
reference (HF transformers fp32 and bf16), not yet through an EngineSmith evaluation.

## Result

| NEFF | compile | size | queued ms/step from C++ | HBM read | MBU |
|---|---|---|---|---|---|
| 28 layers + LM head + argmax, one physical core (LNC=1), 512-position bucket | 27.5 s | 1.29 MB | 3.46 | 1.25 GB | 50% |
| **same, both physical cores (LNC=2), 512-position bucket** | 26-33 s | 1.48 MB | **2.23** | 1.25 GB | 72% |
| LNC=2, 1,024 / 2,048 / 4,608-position buckets | 29 / 31 / 42 s | 1.5-2.0 MB | 2.37 / 2.62 / 3.15 | +59 MB per 512 positions | - |
| the reference's decode step (NxD Inference) | | | 6.26 (whole request, 256 positions) | 1.50 GB | 38% |

At 2.23 ms the step streams 1.25 GB at 561 GB/s, the measured streaming ceiling of an LNC=2 core (a pure HBM-to-SBUF
stream reached 535 GB/s). Static attention buckets read the whole bucket of cache every step; each extra 512
positions costs about 0.13 ms, so use buckets every 512 or 1,024 positions and pick the NEFF from the step count
the host knows. Correctness: 11-, 983- and 4,000-token prompts with 32 to 40 generated tokens identical to HF fp32
and bf16; teacher-forced disagreements only at near-ties.

## One execution, no host work between steps

`pos` (a [1,1] int32 device tensor) selects the input token (`prompt[pos]` while `pos < plen`, else `toks[pos]`),
the embedding row is gathered with the token in a register (`embed.ap(..., scalar_offset=treg, indirect_dim=0)`,
landing directly in column layout), then 28 x {RMSNorm, fused QKV, q/k norm, RoPE, K/V write at `pos`
(`attention.md`), attention over the static bucket, o_proj, residual, RMSNorm, gate/up, SiLU x up, down,
residual}, final RMSNorm, LM head, argmax, `toks[pos + 1] = argmax` (in-place dynamic write) and
`output_0 = pos + 1`. The host builds two tensor sets that swap `pos` and `output_0`, queues every step of a request
with `nrta_execute_schedule` alternating them, waits once and reads `toks` once.

## Column layout, weight-stationary GEMV (no transposes between layers)

A vector of K lives in SBUF as `[128, K/128]` (`v[kt*128 + p]` at `[p, kt]`). Weight tile (nt, kt) of W^T,
`[128 k, 128 n]`, is the stationary operand and the column `x[:, kt]` the moving one: `y[:, nt]` lands in PSUM
already in column layout. Host re-layout at load: `out[p, (nt*KT + kt)*128 + j] = W[nt*128 + j, kt*128 + p]`, so
each partition reads 8 KB contiguous per DMA (4 n-tiles x 8 k-tiles, 1 MB per DMA). TensorE: one 128 x 128 tile
in about 30 ns; x stationary with 512-wide moving weights took twice as long for the same 2 MB. (This corrects the
counterexample about weights as the stationary operand, which used a `[1, K]` row of x as the moving operand.)

## Splitting a layer over the two physical cores (LNC=2)

Both cores run the same program; `nl.program_id(0)` (0 or 1) works as an index (`w[l, pid]`, `1 - pid`). Weights
are re-laid out per core on the host so each core streams one contiguous half:
- QKV split by heads (q heads 8c..8c+7, KV heads 4c..4c+3): attention runs locally on that core's heads and its
  half of the cache.
- o_proj and down split along K (gate/up along N over the same intermediate tiles): each core produces a partial
  `[128, 8]` fp32 hidden vector, summed with `nisa.sendrecv(src, dst, send_to_rank=1-pid, recv_from_rank=1-pid,
  pipe_id=0, dma_engine=nisa.dma_engine.gpsimd_dma)` plus an add; two exchanges per layer, 0.85 us each with
  `gpsimd_dma` (2.0 us with the default DMA engine), about 50 us per step.
- LM head split along the vocabulary (padded to a multiple of 128 with rows masked to -3e38), local argmax, one
  exchange of (max, lowest-id score).
- Embedding, norms and residual adds run redundantly on both cores; both write the same bytes (no
  read-modify-write, so no race).

## Argmax with ties to the lowest id, exactly

Logits in column layout `lg[p, c]` (id = c*128 + p, fp32). Per partition `m_p = max_c`, score =
`(lg == m_p) * (2^20 - id)`, max gives the lowest id reaching `m_p` (ids below 2^20 are exact in fp32). Across
partitions, transpose both columns to rows on the TensorE, `M = max`, then `max((row == M) * score)`.

## Pitfalls

- Test whole-model kernels on real token sequences against a real reference: with random KV caches attention is
  chaotic over 28 layers (relative error 0.14) even for a correct kernel.
- `nrt_load` of the 2 MB step NEFF takes 186-255 ms; uploading 1.4 GB of weights one `nrt_tensor_write` per tensor
  took 7-8 s (not optimized).
- Feeding a long prompt through the decode step costs a step per token: prefill needs its own NEFF (`prefill.md`).

## MoE over a multi-token group: one slot per distinct expert

When a group of T tokens is routed together (prefill in 4-token groups), the T x k (token, expert) pairs cover far
fewer distinct experts. On DeepSeek-V4.1-Flash, 24 pairs hit 16.2 distinct experts on average. Give each distinct
expert one slot, in ascending id order, and apply it to all T tokens at once. The moving operand is the tokens'
block-masked x, and tokens not routed to that expert get routing weight 0, which adds exact zeros. Each token keeps
the same instructions and rounding as the per-token kernel, so tokens are identical, provided the combine sums
experts in the same order the reference does (ascending id for the official DeepSeek code). Compile one NEFF per
distinct-count S and pick it at run time. MoE device time went 1842 -> 1403 us; prefill 25-30% faster.
