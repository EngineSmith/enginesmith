# How EngineSmith works

EngineSmith is a harness. Coding agents write and optimize the engine; the harness decides what counts: the
workload, the reference, the correctness gates, the timing, the acceptance rules and the budget. It is standard
library Python (`src/enginesmith`, about 3,800 lines); optional extras add mlx-lm (the Apple reference) and the task-accuracy scorers.

```
user <-> user's coding agent ──(skills/enginesmith-setup)──> task.toml + model card
                                                                │
                                      enginesmith explain: plan.md, shown to the user
                                                                │
                                                     enginesmith run
                                                                │
         ┌──────────────── one round per candidate ─────────────┴──────────────┐
         │ session: a fresh headless agent edits engine/ (brief: contract,      │
         │          facts, skill, model card, protocol)                         │
         │ snapshot: git commit of engine/                                      │
         │ build: clean checkout, the task's build command                      │
         │ evaluation: controls, then paired rounds vs the reference, judging   │
         │ decision: accept as the best engine, or reject                       │
         └──────────────────────────────────────────────────────────────────────┘
                                                                │
                        final: the best engine on held-out prompts drawn after the search
```

The user sees the rules before they spend hours on them: `enginesmith explain` generates the plan from the task and
the reference's measurements (each case's share of the score, the gates, the round kinds, the acceptance rules, the
budget) and writes it to `plan.md`, stamped with a digest of what it was generated from. `enginesmith run` refuses a
task whose plan is missing or stale, so the setup agent cannot start a run without having produced it.

The judging rules (the workload, the protocol and its timing, the gates, the score, the acceptance rules, the
references) carry a version, `EVALUATION_VERSION`, kept with the history of what each version changed in
`zones.toml`. Every run records the version it started under; the plan, the report and shared engines show it, and
`enginesmith verify` says when a package was judged under another one. A test fingerprints the judging code, so it
cannot change without a new version (CONTRIBUTING.md, "Zones").

## Rounds

| kind | when | starts from |
|---|---|---|
| seed | `engine.seed` is set and `evaluate_seed` | the seed, no session |
| generate | no candidate yet | the seed or an empty project |
| repair | no candidate has passed yet | the latest candidate |
| optimize | a best engine exists | the best engine |

A round records its progress in `rounds/NNN/round.json` after every step (session, snapshot, build, evaluate,
decide). `enginesmith resume` continues from there: an interrupted session is not restarted (its engine directory
is taken as it is), an interrupted evaluation is run again. A candidate whose source is identical to its starting
point is rejected as `unchanged` without an evaluation.

## Sessions

Each session is one fresh agent process in its own process group, started in `work/NNN/` with `engine/` (a git
worktree at the starting commit) and `brief/`:

- `README.md`: the contract, generated from the task (goal, deliverable, protocol, gates, timing, score, acceptance,
  tools, rules);
- `SESSION.md`: facts only: the best engine's per-case table, the failure of the candidate being repaired, the
  candidates rejected since the best engine was accepted and why, and every round's outcome;
- `skill/`, `model-card.md`, `protocol.md`.

The built-in launchers are `claude -p ... --permission-mode bypassPermissions`, `codex exec
--dangerously-bypass-approvals-and-sandbox`, `kimi -p` (Kimi Code's prompt mode, which asks for no approvals),
`pi --print --mode json` and `DSH_PERMISSION_MODE=danger-full-access dsh --profile headless --json` (DeepSeek
Harness); `kind = "command"` takes any argv. By default a task uses the agent the user is talking to:
`agents.current()` recognizes it by the variables each harness sets for the commands it runs (Kimi Code sets none),
and the session environment drops those variables so a session does not attach to the user's own session. A run
does not start inside an agent's sandbox (a shell that cannot write to the home directory, as Codex's and DeepSeek
Harness's are by default): its sessions' agent CLI, cargo and uv write there, so `run` and `resume` refuse and say how
to start it outside. None of them asks for approval or applies the agent's
own sandbox: a session builds and runs GPU code. The agent uses the user's own login or API key.
Inside the session `enginesmith dev-eval` builds `engine/` and runs the evaluation procedure on public prompts. When
the session ends (the agent exits, the timeout passes or the user stops the run), its process group is stopped and
every engine registered by `dev-eval` that is still running is killed.

Isolation is deliberately light: the agent runs as the user. What the design relies on instead:

- every timed request uses a prompt drawn fresh for that request, and the held-out prompts come from a seed that
  is created only after the last session, so no agent can have seen them;
- the harness builds every candidate from a clean checkout of its committed source and runs it with a clean
  environment, so the session's own files and processes do not take part;
- the harness hashes its own source when a run starts and refuses to evaluate if it changed.

These stop mistakes and casual shortcuts, not a determined adversary with the user's permissions.

## Workload

Prompts are raw token ids from the Lily formula, token i = ((i × 2654435761) mod 2^32) mod vocab, so they serve
every model without a tokenizer. The index period is split into a public range (development prompts start at 0,
so shorter ones are prefixes of longer ones), a held-out range and a nonce range (`src/enginesmith/workload.py`).
Lengths go up to 524,288 tokens. Seventeen control prompts from 1 to 4,097 tokens sit at the boundaries where tiled
and chunked kernels usually break.

## Platforms

Everything that depends on the operating system or the accelerator is in `src/enginesmith/platforms/<name>.py`: the
machine facts and checks (`doctor`), the memory budget (`probe`), the memory watchdog, sleep inhibition, the
native-executable check and the default reference. `macos.py` serves Apple Silicon; `generic.py` is the fallback
that keeps the harness and its tests working anywhere. References live in `src/enginesmith/references/` and are
selected by `[reference] kind` or the platform's default. CONTRIBUTING.md says how to add both.

## Reference

`python -m enginesmith.references.mlx_lm --model DIR` loads the model with `mlx_lm.utils.load_model` and runs mlx-lm's own
model and cache classes, unmodified. It reproduces `mlx_lm.generate.generate_step`: prefill in chunks of
min(2048, remaining − 1), the last prompt token alone, greedy decoding pipelined one step ahead (checked token for
token against `generate_step` at 1 to 8,192 tokens). It never computes a step past the last requested token, so no
work leaks between timed requests.

## Evaluation

1. **Controls.** The candidate generates every control prompt; the result is compared with the reference's (cached
   per run) and gated. A wrong engine fails here in minutes.
2. **Rounds.** For each round, both engines run every timed case on the same fresh prompt, the reference first in
   even rounds and the candidate first in odd ones. Each engine is a fresh process: load, warmup requests, then per
   case a timed prefill request generate(P, 1) and a timed decode request generate((), n − 1). One engine is
   resident at a time; each process starts after the cooldown. Round 0 is judged before the other rounds are timed.
3. **Judging.** A stream identical to the reference's agrees. A differing stream is teacher-forced through the
   reference; each step whose token is not the reference's top choice is a mismatch, reported with its margin and
   rank. A stream passes with at most `case_mismatch_percent` mismatching steps, each among the reference's top
   `mismatch_rank` tokens; the evaluation with at most `total_mismatch_percent` of all judged steps. The rank limit
   matters: on Lily prompts the model's output is repetitive, so teacher forcing alone is lenient. Measured on
   Qwen3-0.6B (4-bit): a stream that repeats one token was judged to mismatch at only 4 to 8 of 65 steps, but every
   such garbage stream had a token ranked 1,324th or further down. The mismatches of engines that were correct up to
   rounding ranked 2nd or 3rd on Qwen3-0.6B, and 2nd to 8th, with outliers at 13th and 51st, on the MoE model
   LFM2.5-8B-A1B, where a one-ulp difference can flip an expert choice. The default limit of 100 sits between.
4. **Speed.** Speedup per case and segment: the median over rounds of reference time / candidate time. The parent
   times each request from just before writing it until the reply line arrives. A candidate request that takes more
   than 50 times the reference's time for it (and more than 300 s) fails the candidate.

The memory watchdog kills registered engines when macOS reports critical memory pressure, and the run holds a
`caffeinate` assertion against idle sleep.

## Acceptance

`src/enginesmith/acceptance.py`: the first passing candidate is accepted; afterwards a passing candidate replaces the
best engine when it improves the score by `min_improvement`, or a segment mean by `segment_improvement` (the other
segment and the score at most `max_regression` worse), or one case's segment by `case_improvement` (the score at
most `max_regression` worse). Only weighted cases and segments count. A rejected candidate is kept (its commit
under `refs/rounds/NNN` in `engine.git`) and listed in the next sessions' facts.

## Final

After the search, the best engine is evaluated once more with `final_rounds` rounds on windows from the held-out
range, drawn from a seed created at that moment and recorded in `final/final.json`. Its result is the run's
reported result; it never feeds back into the search.

## Task accuracy

With a `[quality]` table, the best engine and the reference then answer MATH-500 and IFEval
(`src/enginesmith/quality.py`): the same fixed subset of items (MATH-500 stratified by difficulty level), the
model's own tokenizer and chat template, greedy decoding, until an end-of-sequence token or `max_tokens`. An engine
that is correct up to rounding gives mostly the same outputs as the reference, so the comparison is direct; the
greedy scores themselves are not the model card's, which are usually sampled. The answer is the text after the last
`</think>`, or the whole output. MATH-500 is scored with math-verify; IFEval
with Google's checker, copied unmodified into `src/enginesmith/third_party` (strict and loose, prompt and instruction
level; the headline is prompt-level strict). The datasets and NLTK's sentence-tokenizer data are downloaded once at
pinned revisions and checked against their SHA-256.

The result is reported (`quality/summary.json`, the report), never gated: each arm's accuracy, its standard error,
truncations, output lengths and speed, the number of items with identical outputs, and the paired difference over
the items both arms answered. Every generation is kept, and `enginesmith quality RUN` evaluates any round's engine
afterwards.

## Sharing

`enginesmith share RUN` writes a package (`src/enginesmith/share.py`): the engine's source at its round, the binary
(unless `--no-binary`), `enginesmith.json` (the model's source, revision and the SHA-256 of every model file, the
machine, the harness version and digest, the task's settings, the search, the final evaluation and the task accuracy)
and a README that doubles as a Discussions post, plus a zip of it all. It holds no host path, agent command or
credential, and no model weights. `enginesmith verify PACKAGE --model DIR` checks the model files against the
package, builds the engine from the package's source (`--binary` runs the shared binary instead), and runs the formal
evaluation's procedure against the local mlx-lm with the package's settings on fresh prompts.

## Run directory

```
RUN/
  task.json task.toml state.json events.jsonl report.json report.html history.csv run.log
  engine.git/            every snapshot, under refs/rounds/NNN
  build/                 the build worktree
  cache/                 the reference's control tokens
  rounds/NNN/            round.json session.log brief/ build.log engine evaluation/
  final/                 final.json evaluation/
  quality/               summary.json, one JSON-lines file per benchmark and arm
```

The run records its evolution as it goes, with nothing to switch on: after every round `report.json` (`history`,
`milestones`) and `history.csv` hold one row per decided round and one for the held-out final, with the minutes since
the start, the decision and its reason, the score and segment speedups, the incumbent's score, what the agent reported
about itself (model, input, output and cache tokens, cost; None when it reports nothing), the dev-evals it ran, the
engine's size in lines and the lines changed, and for every prompt length the speedups, the engine's and the
reference's time to first token and prefill and decode tokens per second (medians over the paired rounds). The run
also records the agent CLI's version and digests of the skill and the model card. `enginesmith share` puts the same
data in the package and its README.
