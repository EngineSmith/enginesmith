# Worked example: Qwen3-0.6B on Trainium2, 2.21x NxD Inference

An engine for Qwen3-0.6B (bf16, Hugging Face `Qwen/Qwen3-0.6B`) on one logical NeuronCore of a trn2.48xlarge, built by
EngineSmith runs on 2026-11-03/04 (Neuron runtime 2.34.10, neuronx-cc 2.27.5334, nki 0.6.0; reference NxD Inference
0.10.18399, `enginesmith.references.nxdi`). Task: prompt lengths 256 / 1,024 / 4,096, 65 output tokens, prefill and
decode weighted equally.

## Result (independent held-out evaluation of a clean build from source)

| | prefill ms, reference / engine | decode ms per token, reference / engine |
|---|---|---|
| 256 | 11.7 / 5.2 (2.25x) | 4.24 / 1.97 (2.15x) |
| 1,024 | 36.5 / 14.8 (2.47x) | 6.85 / 2.14 (3.20x) |
| 4,096 | 97.1 / 63.9 (1.52x) | 5.41 / 2.70 (2.00x) |
| score | prefill 2.03x | decode 2.40x, **score 2.21x** |

Five paired rounds on held-out prompts from a seed drawn after the search; every stream passed the gates (worst
mismatching token: the reference's second choice). The same binary measured 2.199-2.212 in three further harness
evaluations. Task accuracy (100 MATH-500 and 100 IFEval items, greedy, 3,840 output tokens): 54% / 51% against the
reference's 56% / 55%, every paired difference within one standard error; outputs fork after a median of ~80-125
tokens, always at near-ties (logit gap under 0.3), and a float32 CPU copy of the model sides with the engine about
twice as often as with the reference.

## How it was built (the division of labour that worked)

1. **Kernel lab** (one long session, not judged by the harness): the hard shared problems, each measured against
   HF transformers on the CPU: decode attention with the KV cache written on the device at a run-time position,
   a whole decode step in one NEFF split over the two physical cores, whole-model prefill. Its findings are the
   backend files `attention.md`, `kernels-fusion.md`, `prefill.md`, `kernels-gemv-quant.md`.
2. **Round 001** (35 min, seeded with the lab's code): the lab's kernels behind the engine protocol: 1.75x.
3. **Round 002** (5.4 h): DMA queue placement, an own prefill attention, SBUF-resident short-prompt layers, a
   spinning host loop: 2.21x (`optimization-log.md`).

Independently written engines (other agents, no lab code) reached 1.79x on the same task; agents that started from
one NEFF per operator stayed below 0.2x: the per-execution cost (24 us queued) dominates a 28-layer step.

Files: `optimization-log.md` (what each change bought, including the rejected ones), `model-card.md`.
