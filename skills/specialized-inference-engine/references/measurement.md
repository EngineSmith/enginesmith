# Measurement

Purpose: how to time two builds so that a difference of a fraction of a percent is real, how to learn the noise
floor, and how to report and read the results. Read it before you time anything, before you trust a speedup, and
whenever a result looks too good or too noisy. The task's own timing protocol, if it has one, decides acceptance;
your measurements are development evidence and are labelled as such.

## The rules

1. Compare **ratios measured side by side**, never absolute numbers from different runs.
2. Know the **noise floor** (an A/A run) before you interpret an A/B run.
3. Run **enough rounds** for the effect you are looking for, decided before you start.
4. **One GPU user at a time**, enforced by a lock.
5. **Time what the client sees**: the real request path, the real request shapes.
6. **Label every number** measured (tool, log, rounds) or estimated (how computed); missing is "not measured".

## Interleaved A/B timing (ABBA)

- Start both builds as separate processes and load both before timing (they must fit in memory together without
  swapping; if they do not, run one at a time per round with a cooldown between arms and expect more noise).
- Warm both: one or two untimed requests at every length you will time.
- For each round, for each length: build a **fresh prompt** (never seen by either process; the same prompt for
  both arms within the round), run arm A then arm B in even rounds and B then A in odd rounds, and record one
  sample per arm. The two samples of a pair run seconds apart, so they share the machine's thermal and clock
  state; the ratio cancels most of the drift, and alternating the order cancels a first-or-second-position bias.
- Per length and segment, report the **median of the per-pair ratios** (A time over B time = speedup of B), the
  per-round spread, and how many rounds favoured B.
- Compare the tokens of every pair. The timing tool doubles as a correctness check: a timing run whose tokens
  differ is void.

`scripts/ab.py` implements this. Interleave within one tool run; never compare a number from this morning's run
with one from this afternoon's.

Example (LFM2.5 on an M5 Max laptop, measured): the same parent binary decoded at 258.5, 232.1 and 220.8 tokens/s
at a 4096-token context in three A/B runs on one day. Only ratios taken within one run are comparable.

## Segments and first-call costs

- **Time prefill and decode separately.** Prefill is the request that consumes the prompt and returns the first
  token (time to first token); decode is a continuation that generates N more tokens. They have different
  bottlenecks and change for different reasons. Report both; a change can win one and lose the other.
- **Do not move work across the segment boundary.** Returning the first token before the prompt's state is
  complete, or starting the next call's work early, shortens one segment by lengthening the other. Each request
  completes its own work before it replies.
- **First-call costs** are real but separate: pipeline compilation, buffer allocation, page faults on the first
  touch of the weights, the reference's per-shape compilation on a new shape. Do them at startup where you can,
  warm every length in every process before timing, and measure cold behaviour separately if the task cares.
  Example (an MLX reference of another model, measured): its first request in a fresh process took 15.8-25.7 s
  against 4.0-4.6 s warm.
- **Per-request startup** counts in every request: the GPU starts only when the first batch of work is committed,
  and the host wakes up after the last. Example (LFM2.5, measured): about 0.11 ms host encode, 0.39 ms until the GPU
  started and 0.09 ms wake-up per decode request (about 9 us per token over 64 tokens); making the first command
  batch one step instead of four cut the idle start from 585 to 348 us per request.

## The noise floor (A/A)

Run the A/B tool with the same binary in both arms. The median ratio shows any order or position bias (it should
be 1.000); the per-round spread is your noise. Repeat it when the machine's state changes (another day, after a
long run, another power source).

Example (LFM2.5, measured, 6 rounds after 2 warmups): decode median ratios 1.000 / 1.003 / 1.002 at 256 / 1024 /
4096 tokens, per-round ratios 0.985-1.014; prefill medians 1.020 / 1.002 / 1.015, per-round 0.936-1.058. Prefill
at 256 tokens is a ~60 ms request, so its relative noise is much larger than that of a 64-token decode (~225 ms).
On a quiet day the per-round standard deviation of decode ratios was 0.3-0.5%; on noisy days 1.3-9.7%.

## Rounds needed

- Estimate (normal approximation, independent rounds): to resolve an effect `d` with per-round standard deviation
  `s`, you need roughly `n = (2.5 s / d)^2` rounds. `s` = 0.4% and `d` = 1%: a handful. `s` = 2% and `d` = 0.5%:
  about 100.
- Count wins: under no effect, B wins each round with probability 1/2. Two-sided sign-test p-values: 6 of 6 is
  0.03, 9 of 10 is 0.02, 18 of 24 is 0.02, 30 of 40 is 0.002. Five of six proves nothing.
- Fix the round count before you start. Stopping when the number looks good selects for noise.
- Raise the work per sample (more decode tokens per request) when the requests are short.

Examples (LFM2.5, measured): a +2.7-3.0% change was clear in 10 rounds on a quiet day (B faster in 10 of 10). A
~+0.45% change read 0.993-0.995 over 6 rounds and 1.0042-1.0047 over 24 rounds (B faster in 18-19 of 24). A ~+0.7%
change needed 40 rounds on a noisy day (B faster in 33 of 40, per-round sd 1.4-2.1%).

## Thermal state and the machine

- Laptops throttle under sustained load and absolute speed drifts by several percent within a day. Compare only
  paired ratios, and treat a long run's later rounds with suspicion if the per-round absolute times trend.
- Cool down between long runs and before a run you want to report (for example 2 minutes idle). Example (LFM2.5,
  measured): cross-engine benchmarks run as two passes in opposite order with 2-minute cooldowns agreed within
  1-2%, once 4%. That is the accuracy of a careful absolute number; ratios within a run do better.
- Record the thermal state, the power source and the other heavy processes (builds count) at the start and end of
  every run. How to read the thermal state on Apple Silicon: see `backends/metal-rust/profiling.md`.
- Keep the machine otherwise idle: no builds, no other model processes, no other GPU work while you time.

## One GPU user at a time

Every process that touches the GPU takes one shared lock first: timing runs, bitwise checks, micro-benchmarks,
profiler runs and any ad-hoc engine run. A check needs the lock too, because it uses GPU memory and heats the
machine for whoever times next. Parallel explorers share one lock file and wait for it; never kill another
process to get the GPU. The lock tool in `scripts/` (see `scripts/README.md`) wraps a command in the lock (`flock`
on a file). Bound every run you start with a timeout so that a hung engine cannot hold the lock forever.

## GPU time versus request time

- **Never sum overlapping GPU durations.** Per-dispatch timestamps include overlap with neighbours and the cost of
  the instrumentation; concurrent command batches overlap. Check any per-kernel table against the step time
  measured without instrumentation. Example (LFM2.5, measured): per-dispatch sampled intervals summed to 3604 us per
  decode step against a true 3280 us (about 1.24 us of inflation per dispatch); per-kernel costs measured in the
  chain by duplication summed to 3272 us.
- **Know what the instrumentation costs.** An instrumented build with its switch off must time as the parent
  (ratio 1.000 within noise) and give identical tokens. Each instrumentation mode has its own perturbation; measure
  it and say which mode a number came from.
- **Mirror the real request path.** Time the whole request as the client sees it: through the real protocol, after
  a reset, with the request shapes the task uses (for example a prefill request returning one token, then a
  64-token continuation), waiting for completion before the reply. No benchmark-only mode, no different code path.
- **A micro-benchmark is a diagnostic for one kernel**, never a speedup claim. It can be misleadingly fast when its
  working set stays in an on-chip cache (see `optimization-method.md`). Confirm every kernel-level gain end to end.

## Reporting

For each change and each length: segment, median ratio, per-round range or standard deviation, rounds and warmups,
win count, token identity, the machine state, the tool and the log file. Absolute times are for orientation only.

| length | segment | B speedup (median) | per-round sd | B wins | rounds | tokens |
|---|---|---|---|---|---|---|
| 256 | decode | 1.0042 | 1.3% | 19 / 24 | 24 (2 warmups) | identical |

- Label each number measured (with the log) or estimate (with the arithmetic). Do not combine the two into one
  figure.
- A missing measurement is "not measured" with the reason, never 0 and never left out of a table silently.
- Keep the raw samples, not only the summary; never delete a log, including those of failed or noisy runs.
- Report per-length effects, not only an average: a change can help short contexts and hurt long ones.

## Reading results that look wrong

**Too good:**
- Faster than the floor (`model-card.md` section 6) means a measurement error: work not waited for, work moved
  outside the timed window or into another segment, or a reused result.
- Tokens differ: the build computes something else; the timing is void.
- One length jumps alone. Example (LFM2.5, measured): 1.113 at 512 tokens between 1.028 and 1.061 at its neighbours,
  with a per-round sd of 6.5%: most likely noise; rerun before believing it.
- The parent arm was disadvantaged: warmed less, always in the same position, or throttled during its slots (look
  at the per-round absolute times of each arm).

**Too noisy:**
- Another GPU user, a build or other background load, thermal throttling, memory pressure or swap, an unwarmed
  shape, requests too short for the timer's resolution.
- Fix the cause first (lock, idle machine, cooldown, warmups); then add rounds or work per sample.
- A wrong-sign result at a few rounds is not evidence of a regression; neither is a right-sign one evidence of a
  gain. Run the planned number of rounds and the A/A.
