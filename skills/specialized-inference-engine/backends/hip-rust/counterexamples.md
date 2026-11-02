# Counterexamples

Purpose: measured negative and neutral results on HIP + Rust on an AMD Instinct MI325X, one entry each with its
number, grouped by topic, so that no session repeats them without a new reason. Read it before trying an idea (phase
4 step 2 of `references/workflow.md`). The topic files explain the mechanisms. A result here is a fact about the
engine and machine it was measured on, not a law: retry only with a stated reason why your case differs, and record
the new number.

This file starts nearly empty: the backend has no worked engine yet. The Metal backend's counterexamples
(`backends/metal-rust/counterexamples.md`) are hypotheses about mechanisms here, never facts; do not copy them into
this file unless you reproduce them on this platform.

## Format

One entry per attempt, under the topic it belongs to:

- **What was tried** (bold, one phrase): the measured result with both numbers (before -> after, or variant vs base),
  the unit, the context lengths and rounds, the machine and software (GPU, ROCm and hipcc version, date), whether
  tokens stayed identical, and the source (log file, commit, branch). Then one clause on why, labelled as measured
  (a probe confirmed it) or as a hypothesis.

The shape of an entry (placeholders, not a result):

- **<the change, in a phrase>**: decode <A> -> <B> ms/token at <lengths>, <n> interleaved rounds (noise floor
  <+-x%>), tokens identical / <k> mismatches; <machine, ROCm and hipcc version, date>, <log file or commit>. Why
  (measured / hypothesis): <mechanism>.

Neutral results belong here too (a change within the noise floor, with the floor stated). A change that altered
tokens belongs here with the mismatch count, even if it was faster.

## Launch, graphs and synchronization

- **hipGraph replay to remove per-kernel launch overhead** (one micro-benchmark, not yet reproduced in an engine): a
  100-kernel graph replayed at 1.72 us per kernel (171.7 us per graph) and a 400-kernel graph at 1.69 us per kernel
  (675.7 us), against 1.53 us per back-to-back stream launch of an empty kernel and 1.87 us for a 304-block tiny
  kernel; host wall time; one AMD Instinct MI325X, ROCm 7.14/HIP 7.15 hipcc, 2026-11-02. So graph capture did not
  remove the per-kernel cost here, unlike typical CUDA experience. Why (hypothesis): the cost is per kernel on the GPU
  side (dispatch and the inter-kernel barrier), which a graph still pays. Re-measure with your own decode-step graph
  and a rocprofv3 trace before relying on this either way (`decode-loop.md`).

## Decode GEMV

None yet.

## Fusion and grid-wide synchronization

None yet.

## Attention

None yet.

## Prefill

None yet.

## Numerics

None yet.

## Measurement traps

None yet.
