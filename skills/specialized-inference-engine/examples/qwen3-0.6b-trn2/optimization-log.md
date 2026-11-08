# Optimization log: Qwen3-0.6B on Trainium2

Times are from the engine's own measurements on spare NeuronCores (`host/chain`: queued executions of one NEFF;
A/B: interleaved runs of two builds) unless marked as a harness evaluation. Decode buckets are attention lengths.

## Round 001: the lab's kernels as an engine (harness: 1.75x; prefill 1.47x, decode 2.08x)

One NEFF per decode step (28 layers + LM head + argmax + token and position update, the two physical cores
splitting heads and MLP tiles), layered prefill (embedding, one layer NEFF per layer, head), static attention buckets
chosen by the host, every NEFF embedded with `.incbin`, built through NKI's documented route.

## Round 002 (harness: 2.21x; prefill 2.03x, decode 2.40x)

Decode (profile: DMA 97% busy but 520 GB/s; every weight DMA was a software DGE issued from GpSimd's in-order stream,
behind the q/k-norm, casts and core exchanges):

| change | effect |
|---|---|
| weight, LM-head and KV-cache DMAs on the Sync engine's hardware DGE queue (Sync runs nothing else) | b384 step 2,205 -> 2,078 us; b4224 3,052 -> 2,831 |
| cache writes through separate kernel inputs bound to the same device tensors (`kcw`/`vcw`): the compiler had serialized layer l+1's cache load behind layer l's dynamic-offset write | 2,078 -> 1,987 us |
| KV-cache loads with `dge_mode.none` for buckets of 3,072 and more (worse below) | b4224 2,819 -> 2,679 us |
| 2 MB weight slabs instead of 1 MB below 3,072 | 1,987 -> 1,958 us |
| more buckets (384, 1,152, 4,224) matching the timed positions | decode +0.8 / +2.9 / +2.6% |

The b384 step then moves 1.24 GB in 1.96 ms (633-655 GB/s), at the measured streaming ceiling. Rejected: caches or
weights on the GpSimd or Scalar queues, `dge_mode.none` for weights (b4224 3,947 us), slabs alternating over queues.

Prefill:

| change | effect (per layer unless noted) |
|---|---|
| 256: all four weight matrices loaded at the top of the layer on the Scalar HWDGE queue | 247 -> 231 us |
| own causal attention for buckets up to 1,024 (scores in PSUM, PE transposes, work items software-pipelined in program order) | 256: 231 -> 214 us; 1,024: 684 -> 590 us |
| last layer split: KV writes for every token + the decode layer code for the last token only, then LM head | 256: 734 -> 646 us for the tail |
| RMSNorm rsqrt on the ScalarE (LUT) instead of GpSimd, which made phase A GpSimd-bound | 4,096: 2,702 -> 2,550 us |
| 1,024: transposes by `nisa.dma_transpose` | attention 172 -> 141 us |
| 4,096: own attention with scores staged per query block and the row max chained in the VectorE accumulator | attention 1,311 (nkilib) -> 1,090 us |
| phase A software-pipelined (QKV matmuls issued ahead of the norm and RoPE epilogues) | 4,096: 2,422 -> 2,339 us |
| 256: the whole layer SBUF-resident and split over the cores like the decode step | 197 -> 153 us; prefill +21.7% (A/B) |
| 1,024: the same in 512-token chunks | 537 -> 497 us |
| host: spin on a non-blocking stdin for 10 s after the last input instead of blocking in read() | prefill +4.7 / +3.5 / +2.2% (A/B) |

Rejected: one NEFF for the whole 256-token prefill (6,655 vs 6,562 us for 30 layered executions), nkilib attention
DMAs on HWDGE (at most 2%; its output store does not compile with HWDGE), several layers per NEFF (no gain),
wider exp tiles in nkilib's attention.

What limits it now: decode is at the streaming ceiling (bf16 weights must stay); the 256 prefill is TensorE-bound
under power throttling (`throttle_avg_util_limit` 0.6-0.8 in every compute-heavy profile, not understood); the 4,096
prefill attention's copy + max pass costs ~1 ns per score element.
