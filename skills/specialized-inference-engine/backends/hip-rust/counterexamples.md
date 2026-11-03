# Counterexamples

Purpose: measured negative and neutral results on HIP + Rust on an AMD Instinct MI325X, one entry each with its
number, grouped by topic, so that no session repeats them without a new reason. Read it before trying an idea (phase
4 step 2 of `references/workflow.md`). The topic files explain the mechanisms. A result here is a fact about the
engine and machine it was measured on, not a law: retry only with a stated reason why your case differs, and record
the new number.

The entries below come from one engine (Qwen3-0.6B, batch 1, 8 accepted rounds). The Metal backend's counterexamples
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
  side (dispatch and the inter-kernel barrier), which a graph still pays.
- **hipGraph replay of a whole captured decode step** (timing-only build of an engine with ~6 kernels per layer, one
  sync per request): 64 decode tokens 56.24 -> 56.20 ms at 4K, 99.89 -> 99.62 ms at 32K; no gain. A micro-benchmark
  in the same session: back-to-back dependent 256-block kernels 2.23 us with stream launches vs 1.91 us in a graph
  when empty, 4.57 vs 4.44 us with 3 us of work. one AMD Instinct MI325X, ROCm 7.14 hipcc, Qwen3-0.6B bf16 engine of an EngineSmith run (2026-11-02/03; `source/NOTES.md` of github.com/ChangyiYang/qwen3-0.6b-mi325x-engine), round 006. Why (measured): with the host already ahead of the
  GPU, the per-kernel cost is the GPU-side dependency between kernels, not host dispatch.

## Decode GEMV

- **A decode GEMV concentrated in fewer, fatter blocks** (the 8 MB QKV GEMV spread over 136 blocks inside the
  attention launch instead of 512 small blocks of its own): decode 64 tokens 4K 56.4 -> 64.9 ms (0.870x), 5K 0.846x,
  tokens identical; a build that skipped the inter-block wait was still 0.883x. one AMD Instinct MI325X, ROCm 7.14 hipcc, Qwen3-0.6B bf16 engine of an EngineSmith run (2026-11-02/03; `source/NOTES.md` of github.com/ChangyiYang/qwen3-0.6b-mi325x-engine), round 007. Why (measured):
  bandwidth-bound GEMVs need many small blocks in flight; the kernel boundary they save (~2-4 us) is smaller than
  what the lost parallelism costs.

## Fusion and grid-wide synchronization

- **A device-scope fence to merge decode-attention partitions in the last-finishing block** (atomic counter +
  `__threadfence()`, removing the reduce kernel): decode 1.14 -> 1.55 ms/token at 4K, 2.2 -> 8.1 ms at 40K. one AMD Instinct MI325X, ROCm 7.14 hipcc, Qwen3-0.6B bf16 engine of an EngineSmith run (2026-11-02/03; `source/NOTES.md` of github.com/ChangyiYang/qwen3-0.6b-mi325x-engine),
  round 001. Why (measured): on gfx942 a device-scope fence writes back L2 per block; far more than a launch.
- **The same merge with agent-scope atomics (`sc1` stores and loads, no L2 writeback)**: bitwise identical, decode 1K
  +2.0%, 4K -1.8%, 5K -2.4%. one AMD Instinct MI325X, ROCm 7.14 hipcc, Qwen3-0.6B bf16 engine of an EngineSmith run (2026-11-02/03; `source/NOTES.md` of github.com/ChangyiYang/qwen3-0.6b-mi325x-engine), round 005. Why (measured): `sc1` loads bypass L2 and the extra barriers cost more
  than the boundary once there are more than a few partitions.
- **The partition reduce and the O GEMV in one launch** (reduce blocks dispatched first, GEMV blocks issue their
  weight loads, then spin on a per-launch counter; `sc1` stores, atomic add, `sc1` polling with `s_sleep`; all 320
  blocks resident): tokens identical, decode 64 tokens 1K 52.7 -> 62.1 ms (0.850x), 4K 0.879x, 8K 0.887x, 16K
  0.887x, 40K 0.909x, about 5 us worse per layer. one AMD Instinct MI325X, ROCm 7.14 hipcc, Qwen3-0.6B bf16 engine of an EngineSmith run (2026-11-02/03; `source/NOTES.md` of github.com/ChangyiYang/qwen3-0.6b-mi325x-engine), round 007. Why (measured): on this GPU a cross-CU handoff
  inside one launch costs more than a kernel boundary, even when the GEMV keeps its full block count. Fusions that
  worked fold element-wise work into a GEMV or attention kernel (RMSNorm into the QKV and gate/up GEMVs, SiLU into
  the down GEMV, RoPE into decode attention) and need no inter-block synchronization.

## Attention

None yet.

## Prefill

- **GEMM with a double-buffered LDS and one barrier per K step**: bitwise equal, 10-50% slower than single-buffer
  tiles at every shape (M = 4K-40K; 128x128x32, 128x256x32, 64x128x64, 64x128x32, 256x128x32). one AMD Instinct MI325X, ROCm 7.14 hipcc, Qwen3-0.6B bf16 engine of an EngineSmith run (2026-11-02/03; `source/NOTES.md` of github.com/ChangyiYang/qwen3-0.6b-mi325x-engine), round 004.
  Why (measured): it halves the blocks per CU; the main loop is bound by global-load latency per K step and the
  store + two-barrier round trip, not by MFMA or LDS.
- **Where the hand-written GEMMs stand** (not a negative, the ceiling to aim at): hipBLASLt via `torch` in the
  reference's container (a development tool only; an engine may not link it), 50 hot calls, us at M = 4K: QKV 66.2,
  O 41.8, gate/up 97.9, down 48.9; the engine's tiles 70.0, 44.8, 129.9, 64.5. hipBLASLt is ~5-10% ahead on O and QKV
  at 4K and 25-45% ahead on gate/up and down at 8K-16K. one AMD Instinct MI325X, ROCm 7.14 hipcc, Qwen3-0.6B bf16 engine of an EngineSmith run (2026-11-02/03; `source/NOTES.md` of github.com/ChangyiYang/qwen3-0.6b-mi325x-engine), round 006.

## Numerics

None yet.

## Measurement traps

None yet.
