# Workflow

Purpose: the order of work for building and optimizing a specialized engine, with entry and exit criteria for each
phase and the evidence each phase leaves behind. Read it at the start of a task and whenever you are unsure what to
do next.

The phases are sequential for a new model. A session on an existing engine usually resumes in phase 4, but drops
back to an earlier phase whenever that phase's exit criterion stops holding (for example, a correctness check fails:
back to phase 1 for that path; timings stop being reproducible: back to phase 2).

## Phase 0: onboard the model

Goal: know exactly what you are computing and how fast it could possibly run.

Do:
- Fill the model card (`model-card.md`): identity and hashes of the checkpoint, the configuration, the forward pass
  as pseudo-code, every tensor with its stored format and size, the state the model carries between tokens, and the
  reference implementation pinned to an exact version.
- Compute **bytes read per decode token** (active weights plus KV and recurrent state at a given context) and
  **FLOPs per prefill token**. Divide by the measured streaming bandwidth and the measured matrix throughput of the
  machine to get the decode and prefill floors.
- Record where the sources disagree (config vs code vs paper) and which one the reference follows. The reference's
  behavior is the specification.

Done when: the floors are written down, and every operator of the forward has a known formula, dtype and rounding
point.

## Phase 1: a correct baseline

Goal: an engine whose outputs match the reference's, with no optimization yet.

Do:
- Port the nearest existing kernels (often the reference framework's own) rather than writing new numerics. The
  reference's rounding comes from its specific kernels and its per-shape kernel choice; matching the formula is not
  enough.
- Match the reference's prefill partition (chunk sizes and where L=1 steps start). The partition changes rounding.
- Keep a simple, slow path for every operator until the whole model passes; only then replace pieces.
- Check teacher-forced logits at every operator boundary you can (per-layer dumps help), then free-running greedy
  generation, across prompt lengths that straddle every chunk and tile boundary.

Done when: the framework's correctness checks pass at every length the task covers, and the self-checks in
`correctness.md` (repeatability across processes, split requests, checkpoints) pass.

## Phase 2: the measurement harness

Goal: tools you trust, before you trust any speedup.

Do:
- A bitwise A/B tool: two builds, the same prompts, every output token compared (`scripts/same.py` or the task's
  equivalent). Cover chunk and tile boundaries, several request shapes (one long request, a split request, a chain
  of continuations) and random prompts.
- An interleaved timing tool: both builds loaded, requests in A, B, B, A order on fresh prompts, ratios per segment
  (prefill, decode) (`scripts/ab.py`).
- A lock so that only one process uses the GPU at a time.
- Run A against A to learn the noise floor at each length.

Done when: an A/A run reports ratios within the noise floor you will use as your acceptance threshold (about
+-0.3% for decode on a quiet machine in the worked example).

## Phase 3: find the time

Goal: a ranked list of opportunities, each with an estimate.

Do:
- Write a **dispatch map** of one decode step: every kernel in order, its grid, the bytes it reads, and which
  dispatch depends on which (`optimization-method.md` has the template; `examples/lfm2.5/decode-map.md` is a full
  one).
- Measure GPU time per dispatch inside the real engine (not only in isolated micro-benchmarks, which can be
  misleadingly fast when their working set stays in the on-chip cache) and check that the per-kernel times add up
  to the measured step time.
- Compare each kernel's effective bandwidth with the streaming ceiling, and count the fixed per-dispatch cost.

Done when: you know how much of the step is weight streaming near the ceiling, streaming below it, and fixed
per-dispatch overhead, and you have ranked the opportunities by estimated gain and risk.

## Phase 4: optimize

Goal: measured, bitwise-identical speedups, one mechanism at a time.

Do, for each opportunity:
1. State the mechanism, the call site, the expected effect and the observation that would prove it wrong.
2. Check whether the parent already does it, and whether a previous session measured it (the optimization log and
   `backends/<name>/counterexamples.md`).
3. Implement it in a branch or work tree of its own.
4. Bitwise A/B against the parent. Any difference: stop and find the expression or order you changed.
5. Interleaved timing against the parent, enough rounds to beat the noise floor.
6. Commit with what it does, how it was verified and the measured effect. Record neutral and slower results too.

Done when: the remaining opportunities are all below the noise floor or above the risk you accept. Then look for a
structural change (a different loop, a different layout) rather than more local tuning.

## Phase 5: package

Goal: the engine builds and runs from a fresh checkout on another machine of the same kind.

Do:
- No absolute host paths in the source or binary; kernels embedded in the binary; pinned dependencies; offline,
  locked builds; a quiet build (no warnings).
- Reproducible output: the same source builds the same binary bytes regardless of the build directory.
- A clean failure, with one short message, on an unsupported machine.
- Third-party code attributed per file with its license.

Done when: a fresh clone builds, passes the bitwise A/B against the engine you measured, and runs.
