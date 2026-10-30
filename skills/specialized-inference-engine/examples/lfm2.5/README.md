# Worked example: LFM2.5-24B-A2B on an Apple M5 Max ("Molten")

A complete, measured example of the skill applied end to end: a native Rust + Metal engine for one checkpoint,
`LiquidAI/LFM2.5-24B-A2B-MLX-4bit` @ `667408d4`, on one machine, an Apple M5 Max (128 GB). The reference is
unmodified mlx-lm 0.32.0 on MLX 0.32.3 with the same weights. The released engine is called Molten.

The model: 24B parameters, about 2.45B active per token. 40 layers (30 double-gated short-convolution layers and 10
GQA attention layers), two dense FFN layers then 38 MoE layers with 64 experts and sigmoid top-4 routing, a 6-bit tied
embedding/head, MLX affine 4-bit/64 weights. A decode token reads 1.46 GB of weights.

How it was built:
1. **An agent loop** (a coding agent working session by session from the best engine so far, with a fixed evaluator
   that rebuilds, checks tokens against mlx-lm and times both): the first correct engine was 0.56x the reference
   three hours in, passed it on day 2.5, and reached 1.25x after 31 accepted versions (day 5.2).
2. **A final fusion round** (Claude coordinating parallel sub-agents, about three hours): an in-engine profile, then
   decode-only fusions, each bitwise identical to its parent. Dispatches per decode token fell from 355 to 252.

Result (measured, the engine's `docs/BENCHMARKS.md`, 2026-10-25):
- **1.28x end to end**: the framework's whole-request score, the geometric mean over ten context lengths from 256 to
  131,008 tokens of the request speedup over mlx-lm.
- **Decode 1.53x mlx-lm** at 256 tokens of context in the paired evaluation (303 against 198 tok/s), 1.43-1.53x
  across all ten lengths.
- **300+ tok/s**: 304.5 tok/s at 256 tokens of context and 285.1 at 4K, against mlx-lm's 190.0 and 184.0; the
  decode floor at 256 is about 401 tok/s (1.47 GB at about 590 GB/s).
- Greedy tokens agree with mlx-lm except at near-ties: 4.5% of held-out teacher-forced steps differ (186 of 4,160),
  within the task's checks.

## Files

| file | what it holds | read it when |
|---|---|---|
| `model-card.md` | the template of `references/model-card.md` filled in for this checkpoint: identity, config, forward pass with dtypes and rounding points, tensor groups and formats, state, bytes and floors, the reference's rounding routes, where the sources disagree | onboarding a model of a similar family (MoE, hybrid conv/attention, MLX affine weights), or to see what a finished card contains |
| `decode-map.md` | the decode dispatch map of the final engine (stage, kernel, grid x threadgroup, bytes, dependency), how the dispatch count fell, the in-engine GPU time per kernel and per category, bytes per token by category | writing a dispatch map, or judging where a decode step's time goes |
| `optimization-log.md` | the history: all 32 accepted versions with decode speed and TTFT, what changed at the big steps; the final round's mechanisms with dispatches removed, verification and measured effect; what was rejected, with numbers; lessons | choosing the next optimization; before trying something the log may already have measured |

The tools these measurements were made with are `scripts/same.py` (bitwise A/B), `scripts/ab.py` (interleaved
timing), `scripts/gpu_lock.py` and `scripts/bytes_per_token.py`; the model card's section 6 has this model's spec for
the last one.

Numbers in these files are labelled with the build they come from (the agent loop's last version, the profiled
build base2, or the final engine) and with whether they were measured or estimated. Do not reuse them for another
model or machine: recompute the floors and re-measure.
