# Backends

Purpose: how the platform-specific part of this skill is organized, what belongs in a backend and what does not, how
to add a backend for another platform (for example CUDA), and which backends exist today. Read it before writing or
extending a backend; for engine work on an existing backend, go straight to that backend's `README.md`.

## Organization

- One directory per platform and host stack: `backends/<name>/`, for example `backends/metal-rust/` (Metal Shading
  Language kernels, Rust host code, Apple Silicon).
- Each backend has a `README.md` with: its file index, its hard rules (language, toolchain and numerics constraints
  that every change must keep), and a table of measured platform numbers (streaming bandwidth, cache sizes,
  per-dispatch and synchronization costs, timing noise) with their sources.
- Topic files mirror the `metal-rust` set wherever the topic exists on the platform, so that a reader who knows one
  backend finds the same question in the same file of another:

| file | topic |
|---|---|
| `runtime.md` | weights, buffers and views, kernel compilation, submission, hazards, lifetimes, build |
| `decode-loop.md` | the GPU-resident decode loop, batching of steps, the dispatch budget, stopping and rewinding |
| `kernels-gemv-quant.md` | decode GEMVs over the quantized format, expert gathers, the router GEMV |
| `kernels-fusion.md` | merging dispatches without changing bits; grid-wide dependencies and their cost |
| `attention.md` | decode attention (single-pass and split forms), KV cache layout, prefill attention |
| `prefill.md` | matrix-unit GEMMs, the reference's chunk partition, grouped expert GEMMs |
| `profiling.md` | per-kernel GPU time inside the real chain, perturbation, micro-benchmarks that do not mislead |
| `serving.md` | streaming, stop conditions, prefix reuse, sampling, text layer, optional server |
| `counterexamples.md` | measured negative and neutral results, one line each with number and source |

A backend may add files for topics its platform has and Metal lacks; it should not drop a topic silently. If a topic
does not apply, say so in its `README.md`.

## What stays out of a backend

- **Model facts** go in the model card (`references/model-card.md`; a filled card per model under `examples/`).
  Backend files use a model's facts only as short, labelled examples ("Example (LFM2.5, M5 Max): ...").
- **Framework and task rules** (the engine protocol, the correctness checks, budgets, what may be run) come from the
  task contract and are never restated in the skill.
- **Platform-neutral method** goes in `references/`: the workflow, correctness, measurement and optimization method.
  If a backend needs a rule that a generic file contradicts, either the generic rule is platform-specific (move it
  into the backend) or the backend is wrong; do not keep both.
- **Unlabelled numbers.** Every number in a backend is measured (with its source: log file, commit, document) or
  marked as an estimate or inference. A vendor specification is not a measurement of your machine.

## Adding a backend (example: CUDA)

1. **Write the README first.** Hard rules for the platform: host language and toolchain as the task contract allows
   them; how kernels are compiled and embedded; the compiler's default numerics (for example whether FMA contraction
   or fast math is on, and per-kernel overrides); feature and version gates; device capability detection and the
   clean refusal on unsupported devices. Then measure the platform table yourself on the target machine.
2. **Write a counterpart for each `metal-rust` topic, in this order:** what you need for a correct baseline first
   (runtime, the quantized GEMV, attention, prefill), then what you need to find time (profiling), then speed
   (decode loop, fusion), then serving.

| `metal-rust` topic | what a CUDA counterpart must cover (to be written and measured; nothing here is measured) |
|---|---|
| `runtime.md` | device allocation and weight upload or mapping, streams and events, kernel compilation (embedded PTX/cubin or run-time compilation), hazards between streams, build reproducibility |
| `decode-loop.md` | the same GPU-resident token ring; the launch-overhead counterpart (for example capturing a decode step as a graph, or persistent kernels) with its measured cost per launch |
| `kernels-gemv-quant.md` | warp-level dequantizing GEMVs for the checkpoint's format, compile-time shapes, bandwidth against the measured ceiling |
| `kernels-fusion.md` | the same bitwise rules; grid-wide dependency options of the platform (cooperative launch, dependent launches, last-block completion counters) and their measured cost by grid size |
| `attention.md` | single-pass and split (split-K) decode attention matching the reference's routes and partition; KV layout; the prefill flash tile |
| `prefill.md` | tensor-core GEMMs for the device's architecture, the reference's chunk partition, grouped expert GEMMs |
| `profiling.md` | the platform's timers and profilers, the same in-chain method, sums checked against the step time, cache effects on micro-benchmarks |
| `serving.md` | mostly reusable: the pure-Rust text layer and server carry over; sampling kernels need a port with the same semantics and checks |
| `counterexamples.md` | starts empty; Metal results are hypotheses on another platform, never facts |

3. **Generic references apply unchanged:** `references/workflow.md` (phases), `references/model-card.md` (floors are
   bytes over measured bandwidth, FLOPs over measured throughput), `references/correctness.md` (reference routes,
   bitwise rule, self-checks), `references/measurement.md`, `references/optimization-method.md`, and the A/B, timing
   and lock tools in `scripts/` where the task contract allows them. The reference implementation for another
   platform may be a different framework; its per-shape routes and partition are then the ones to reproduce.
4. **Carry over mechanisms, not numbers.** Fusion targets, the dispatch map, the in-chain profiling method and the
   general counterexamples (route changes from concatenation, intermediate casts in fusions, never padding
   quantization groups, double-buffered history writes) transfer as ideas to test. Their costs do not.
5. **Register the backend** in the table below and in `SKILL.md`, with its status and the machine it was measured on.

## Status

| backend | status | measured on | notes |
|---|---|---|---|
| `metal-rust` | written: README and nine topic files | Apple M5 Max (40-core GPU, 128 GB) | one worked model (LFM2.5-24B-A2B, MLX 4-bit); numbers outside that engine and machine are not measured |
| others (CUDA, ...) | not written | - | follow "Adding a backend" |
