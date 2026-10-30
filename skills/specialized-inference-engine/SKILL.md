---
name: specialized-inference-engine
description: Build and optimize a specialized native inference engine for one fixed model checkpoint, matched token for token against a reference implementation and faster than it. Covers onboarding a model (model card, bytes and FLOPs per token, speed floor), bringing up a correct baseline, bitwise-safe optimization, interleaved timing, profiling, kernel fusion and packaging. Backend modules hold the platform techniques; the Metal + Rust backend for Apple Silicon (M5 family) is the one written today. Use for any session that writes, verifies or speeds up engine code.
---

# Specialized inference engine

You are building an inference engine for **one model checkpoint on one kind of machine**, not a framework. That
narrowness is the whole advantage: every shape is known at compile time, every kernel can be fused for this graph,
and the decode loop can be shaped around this model's dispatch chain. A general framework cannot do any of that.

The task contract you were given (engine protocol, correctness gates, budgets, sandbox, what you may run) is
provided separately by the framework. **It always wins over this skill.** This skill is about how to build and
speed up the engine; it never restates the contract.

## The rules that matter most

1. **Correct before fast.** No tuning until the whole model matches the reference (phase 1 below). A fast engine
   that fails the correctness gate is worth nothing, and debugging numerics inside an optimized engine is slow.
2. **Every optimization is bitwise identical to its parent.** Once a correct baseline exists, a change that alters
   any output token or logit is a correctness change, not an optimization. Keep the same arithmetic expressions,
   the same rounding points and the same reduction orders, and verify with a token-for-token comparison over many
   prompts and request shapes before timing anything. See `references/correctness.md`.
3. **Measure, never guess, and measure the right thing.** Time two builds interleaved (A, B, B, A) on fresh prompts,
   compare ratios rather than absolute numbers, and know your noise floor. Thermal throttling and other GPU users
   move absolute numbers by several percent. See `references/measurement.md`.
4. **Know the floor.** Decode time per token is bounded below by bytes read per token divided by the measured
   streaming bandwidth. Compute it on day one (`references/model-card.md`) and judge every kernel against it.
5. **One mechanism per change, with its measured effect.** Record what you changed, how you verified it and what
   it bought, including the attempts that were neutral or slower. Negative results with numbers are as valuable
   as wins: they stop the next session from repeating them.
6. **Label every number** as measured (with where and how) or estimated. A missing measurement is "not measured",
   never zero.

## Workflow

The six phases, their entry and exit criteria and the evidence each must leave behind are in
`references/workflow.md`. In short:

| phase | goal | done when |
|---|---|---|
| 0. Onboard the model | fill the model card: graph, tensors, formats, bytes and FLOPs per token, reference pinned | the decode floor and prefill floor are computed |
| 1. Correct baseline | port the nearest existing kernels, match the reference's partition and rounding | teacher-forced and free-running checks pass at every length |
| 2. Measurement harness | bitwise A/B tool, interleaved timing tool, GPU lock, noise floor | an A/A run shows the noise floor |
| 3. Find the time | dispatch map, per-kernel GPU time, compare with the floor | a ranked list of opportunities with estimates |
| 4. Optimize | one mechanism at a time, bitwise-checked and timed | measured speedup, tokens identical |
| 5. Package | self-contained build, reproducible, no host paths | a fresh clone builds and runs |

Most sessions resume in phase 4. Read the parent engine's notes and the last session's results first; do not
redo work that was measured and rejected.

## Where the time goes (the general shape)

For a batch-1 decoder on a GPU with unified memory:

- **Decode** is memory-bound: every token streams the active weights once. A good engine runs its weight-streaming
  kernels near the machine's streaming bandwidth and then spends the rest of its effort on **fixed per-dispatch
  costs**, because a decode step is hundreds of small, dependent dispatches. Removing dispatches by fusion is
  usually the largest remaining lever once the GEMVs are near the bandwidth limit.
- **Prefill** is compute-bound for long prompts: use the GPU's matrix units, tile for the shapes you have, and
  share K/V tiles between the query heads of a KV head.
- **Long-context decode** becomes bound by the KV-cache read, not by weights; attention needs a split (two-pass)
  form past a few thousand keys.
- **The host** must never sit in the token loop: keep tokens on the GPU between steps, encode steps ahead, and wait
  once per request.

The backend module gives the measured numbers for its platform.

## Backends

Platform techniques live in `backends/<name>/`. Read the backend's `README.md` first; it lists its files and the
platform's hard rules.

| backend | status | for |
|---|---|---|
| `backends/metal-rust/` | written, measured on an Apple M5 Max | Apple Silicon, Metal Shading Language kernels, Rust host code |
| others (e.g. CUDA) | not written | see `backends/README.md` for how to add one |

## Files

| file | read when |
|---|---|
| `references/workflow.md` | starting a task or a session; deciding what to do next |
| `references/model-card.md` | onboarding a model; computing floors; checking a semantic detail of the model |
| `references/correctness.md` | before any change to arithmetic; when a check fails; when choosing what "identical" means |
| `references/measurement.md` | before timing anything; when a result looks too good or too noisy |
| `references/optimization-method.md` | choosing the next optimization; writing up a result |
| `backends/metal-rust/README.md` | any Metal + Rust work (index of the backend files) |
| `examples/lfm2.5/` | a complete worked example: model card, decode dispatch map and the optimization log of an engine that reached 1.53x the reference's decode speed in a paired evaluation (300+ tokens/s on an M5 Max) |
| `scripts/README.md` | the A/B, timing and lock tools, and when the task contract lets you use them |
