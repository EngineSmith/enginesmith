---
name: enginesmith-setup
description: Set up, launch and watch an EngineSmith run for a user who wants a specialized inference engine for one model on their own machine. Covers the conversation (model, machine, what to optimize), the task file and model card, checking the reference on this machine, launching the run, monitoring it and explaining the results. Use when a user asks to build, speed up or optimize an inference engine with EngineSmith, or asks about a run's progress.
---

# Setting up an EngineSmith run

EngineSmith builds a specialized inference engine for one model on one machine. It starts a fresh coding agent for
every session; that agent writes and optimizes the engine. The harness owns the rules: it builds every candidate,
checks its tokens against the reference (stock mlx-lm), times it against the reference, and keeps a candidate only
when it is correct and faster under the user's priorities.

**Your role** (the user's coding agent, in this repository) is the other side: agree with the user on what to
build, write the task, check that the reference works on this machine, launch the run, watch it and explain the
results. You do not write the engine.

Everything below runs from the repository root. The commands print JSON you can read.

**End every reply to the user with three short lines:**
- **Now:** what EngineSmith is doing at this moment (nothing yet; measuring the reference; round 3 of 12, session /
  build / evaluation, since when; the final evaluation; finished), from `enginesmith status` when a run exists;
- **Next:** the next step, and roughly when it happens;
- **Needed from you:** an answer, a confirmation, a login, or nothing.

The user is usually a practitioner: give exact numbers and the reasons behind them, not reassurance.

## 1. Check the machine

```
uv sync --extra mlx --extra quality --extra dev   # once: the harness, the mlx-lm reference, the benchmarks
uv run enginesmith doctor
```

Read `problems` and `notes`. EngineSmith supports macOS on Apple Silicon only for now. It needs `git`, Rust
(`cargo`) and one coding agent CLI that the user is logged in to: `claude` (Claude Code), `codex` (Codex CLI), `kimi`
(Kimi Code), `pi` or `dsh` (DeepSeek Harness). The repository does not install one. Ask before installing anything.

**The sessions run on you by default.** `current_agent` in the doctor's output is the agent this shell runs in: you,
detected from the variables your harness sets for its commands, even when your system prompt does not name your
harness. Use it unless the user wants another; it is null only for Kimi Code (which sets none) or a plain terminal,
and then you know which CLI you are. Each session starts that CLI headless with the user's own subscription or API
key, from the environment the run starts in; tell the user that a run uses it for many hours. Never write a key into
the task file.

- Claude Code, Codex, Kimi Code, Pi: the CLI's own login, or an API key in the environment (for example
  `ANTHROPIC_API_KEY`, `CODEX_API_KEY`, `DEEPSEEK_API_KEY` for Pi).
- DeepSeek Harness: the sessions use its headless profile, which ignores keys saved in the Web UI and reads
  `DEEPSEEK_API_KEY` (or another provider's key) from the environment, then `.env` in the working directory, then
  `$DSH_HOME/.env`; the user puts the key there themselves. It uses the default model chosen in its settings
  (`agent.model` does not apply). `dsh` must be on PATH: if the user starts it with `npx`, it needs
  `npm install -g @deepseek-ai/dsh` (ask first).

## 2. Agree on the task

Talk with the user until these are clear. Offer the defaults; do not interrogate.

1. **The model.** A Hugging Face repo id or a local directory, in MLX format (the reference is mlx-lm). If the user
   names a non-MLX checkpoint, look for its MLX release or an `mlx-community` conversion with the same quantization
   they want. Say how large the download is and ask before downloading it into `models/` (git-ignored):
   `uv run hf download REPO --local-dir models/NAME`.
2. **What to optimize.** Translate the user's goal into the objective (section 4):

   | the user wants | prefill | decode | lengths and weights |
   |---|---|---|---|
   | a balanced engine (default) | 0.5 | 0.5 | 256, 1024, 4096, 16384, equal |
   | fast chat or agent replies (decode speed) | 0.2 | 0.8 | 256 to 16384; `output_tokens` 129 or more |
   | fast time to first token, long documents, RAG | 0.8 | 0.2 | 1024 to 65536, more weight on the long ones |
   | long context | 0.5 | 0.5 | add 65536 to 524288 (if the model supports them), weight 2 to 4 |
   | one specific shape, e.g. decode at 32K | 0 | 1 | `[32768]` |

   Weights are relative. A segment or length with weight 0 is still checked for correctness but not scored.
3. **How strict the acceptance is.** The first candidate that passes the correctness gates becomes the best engine,
   however slow it is. After that, the defaults accept a candidate when its score improves by 1.5%, or its prefill
   or decode mean by 1.5% (the other at most 1% worse), or one case's prefill or decode by 4%. They sit above the
   timing noise of an unchanged engine (about 1% for means, up to 4% for single cases). Lower thresholds accept
   smaller steps and more noise; higher ones fewer, surer steps. Every timed request must also answer within 50
   times the reference's time for it (at least 300 s); a slower candidate fails.
4. **Correctness.** The defaults (at most 20% mismatching steps per token stream, each mismatching token among the
   reference's top 100 choices, at most 10% overall) accept the numerical near-ties of a correct engine, including
   the expert-routing flips of MoE models, and reject a wrong one. Explain them; change them only if the user asks.
5. **Budget.** Sessions (default 24), session timeout (default 2 h), wall time (default 48 h). Use the timing from
   step 5 to say what fits.
6. **Agent.** You, by default (`current_agent`); say which harness and model the sessions will use. Offer the
   others found on PATH as a short choice, yours first and marked as the default, and switch only if the user picks
   one. Optionally the model (`agent.model`, passed to the CLI's `--model`; not for `dsh`).
7. **Task accuracy.** Include `[quality]` (section 4) unless the user declines: after the final evaluation the best
   engine and mlx-lm both answer MATH-500 and IFEval with greedy decoding, so their accuracies compare directly.
   Size `samples`, `max_tokens` and `time_limit` with the reference's decode speed from step 5: both arms generate
   up to samples x 2 x max_tokens tokens. Greedy scores are not the model card's (those are sampled); say so.
8. **Starting point.** From scratch (default) or from an existing engine directory (`engine.seed`), for example the
   best engine of an earlier run (step 8 shows how to get its source).

## 3. Probe the model

```
uv run enginesmith probe models/NAME
```

It reports the architecture facts from `config.json`, the weight bytes, an ESTIMATE of the KV cache per token and
which prompt lengths should fit in memory. Lengths beyond `max_position_embeddings` are out. Tell the user which
long lengths are realistic before they choose them.

## 4. Write the task and the model card

Create `tasks/NAME/` with two files. The task (paths are relative to it):

```toml
name = "qwen3-8b-m5max"            # letters, digits, '.', '_', '-'

[model]
path = "../../models/Qwen3-8B-4bit"
source = "mlx-community/Qwen3-8B-4bit"   # the repository you downloaded it from
card = "model-card.md"

[engine]
# seed = "../../seeds/my-engine"   # optional: start from this engine source instead of an empty project
# build = ["cargo", "build", "--release"]      (default)
# executable = "target/release/engine"        (default)

[agent]
kind = "dsh"                       # current_agent from doctor; claude | codex | kimi | pi | dsh | command
# model = "opus"                   # passed to the CLI's --model (not for dsh)
session_timeout = "2h"

[workload]
lengths = [256, 1024, 4096, 16384]  # prompt tokens, at most 524288
output_tokens = 65                  # tokens generated per case (one prefill request + a decode request of n - 1)

[objective]
prefill = 0.5
decode = 0.5
# length_weights = { "16384" = 2 }

[acceptance]                       # the defaults; 0 turns a rule off
min_improvement = 0.015
segment_improvement = 0.015
case_improvement = 0.04
max_regression = 0.01

[gates]                            # the defaults
case_mismatch_percent = 20
total_mismatch_percent = 10
mismatch_rank = 100

[evaluation]                       # the defaults
rounds = 5
warmup = 1
cooldown = "15s"
final_rounds = 5

[budget]
sessions = 24
wall_time = "48h"

[quality]                          # MATH-500 and IFEval accuracy of the best engine and of mlx-lm, reported
samples = 100
max_tokens = 8192
time_limit = "2h"
```

`docs/task.md` documents every key. Leave out what keeps its default.

**The model card** (`model-card.md`) is the model-specific part of the coding agent's skill: every session reads it.
Write it from `config.json`, the probe and the reference's own source (`.venv/lib/python3.12/site-packages/mlx_lm/
models/<model_type>.py`), following the template in `skills/specialized-inference-engine/references/model-card.md`.
A complete example is `skills/specialized-inference-engine/examples/lfm2.5/model-card.md`. Cover:

- the graph: every layer kind and its order, attention (heads, KV heads, head dimension, RoPE variant and
  parameters, sliding windows, QK norms), MLP or MoE (experts, top-k, router function and normalization, shared
  experts), norms (kind, epsilon, placement), embeddings (tied or not), the final norm and LM head;
- the stored formats: dtypes, quantization (bits, group size, which tensors), tensor names and shapes;
- the reference's exact semantics where engines usually get it wrong (from the mlx-lm source: accumulation
  dtypes, where it casts, the prefill chunking of 2048 tokens);
- bytes read per decode token (`skills/specialized-inference-engine/scripts/bytes_per_token.py`), the KV cache bytes
  per token, and the decode floor (bytes per token divided by measured bandwidth; say it is an estimate);
- the reference's measured times on this machine from step 5.

Write facts with their source and label every estimate. Leave out strategy; the default skill holds that.

Then check the task:

```
uv run enginesmith validate tasks/NAME/task.toml
```

## 5. Measure the reference on this machine

```
uv run enginesmith prepare tasks/NAME/task.toml
```

It runs the reference once at every length and writes `tasks/NAME/prepare.json`: load time, prefill and decode
times, peak memory, and an ESTIMATE of one evaluation's duration (the run uses it until it has measured one). It
stops when memory pressure turns critical or a request takes more than an hour. Use it to settle the lengths, the
rounds and the budget with the user: every evaluation runs every timed length 2 x `rounds` times (reference and
candidate), so very long prompts dominate the time. A 512K prompt can take tens of minutes per prefill.

## 6. Show the plan, then launch

```
uv run enginesmith explain tasks/NAME/task.toml
```

It writes the plan to `tasks/NAME/plan.md` and prints it, generated from the task file and `prepare.json`: the model
and reference; every timed case with its share of the score and the reference's measured times; the score formula;
the correctness gates; one evaluation's procedure and duration; how the search evolves round by round and which agent
and model run the sessions; the acceptance rules with their thresholds; the final evaluation and task accuracy; the
budget.

**Show the user the whole plan before anything starts**: every section and every number, translated into their
language, not a summary. The user decides how many hours of their machine and their agent subscription the run takes,
so this is where they see exactly what will be measured and when a candidate counts as better. Add for each choice
why you made it from what the user said (why these lengths and weights, why these thresholds against the measured
noise, why this budget). Do not restate the rules from memory: the plan is generated from the code. Then ask for
confirmation; if the user changes something, edit the task and run `explain` again. `enginesmith run` refuses a task
whose `plan.md` is missing or older than the task or `prepare.json`.

The run occupies the GPU and the agent for up to `wall_time`. Check that the agent works headless, from this shell
(the run inherits its environment): `claude -p "reply ok"`, `codex exec "reply ok"`, `kimi -p "reply ok"`,
`pi --print "reply ok"` or `dsh --profile headless "reply ok"`. The user must log in or set the key themselves. Then:

```
uv run enginesmith run tasks/NAME/task.toml --detach
```

It prints the run directory (`tasks/NAME/runs/NAME-DATE-TIME/`) and continues in the background; the output goes to
`run.log` there. The run keeps the machine awake while it is active; tell the user to keep it plugged in.

## 7. Watch it

```
uv run enginesmith status RUN_DIR
```

It shows the phase, the current round and step, the best engine's score so far, the latest decisions and the time
left. A round (session, build, evaluation) usually takes one to a few hours, so check every 30 to 60 minutes, not
more often. `RUN_DIR/report.html` has the full table; `RUN_DIR/rounds/NNN/` has each round's session log
(`session.log`), brief, build log and evaluation.

- `uv run enginesmith stop RUN_DIR` ends the search after the current step; the best engine still gets its held-out
  final evaluation. `--now` stops everything at once.
- `uv run enginesmith resume RUN_DIR --detach` continues a run that stopped (a crash, a reboot, `stop --now`).
  `--continue-search` also withdraws a stop request.
- A run's settings are frozen. To change them, stop it and start a new run, seeded with its best engine if useful.
- Never edit files in a run directory or in `src/enginesmith` while a run is active: the harness checks its own
  source before every evaluation and stops if it changed.

## 8. Explain the results

- Speedups are reference time / engine time on this machine, per case and per segment (prefill, decode). The
  score is their weighted geometric mean under the task's objective. 1.25 means 25% faster than mlx-lm here.
- The search's numbers come from public-style prompts the agent never saw in advance. The final evaluation uses
  held-out prompts drawn from a seed created after the last session; it is the number to report. Say whether it
  passed the gates.
- The best engine: `RUN_DIR/rounds/NNN/engine` (the binary; it runs as `engine --model MODEL_DIR` and speaks
  `docs/protocol.md`). Its source: `git -C RUN_DIR/engine.git archive refs/rounds/NNN | tar -x -C DEST`. Its
  `NOTES.md` records what each session changed and measured.
- Task accuracy (`RUN_DIR/quality/summary.json`, when the task had `[quality]`): each arm's greedy accuracy, how
  many items had identical outputs, and the paired difference over the items both answered.
  `uv run enginesmith quality RUN_DIR [--round NNN] [--samples N]` evaluates an engine afterwards.
- Report what was measured, with the file it came from. Label estimates. If something was not measured, say so.

## 9. Share it, or check someone else's

- `uv run enginesmith share RUN_DIR` writes `RUN_DIR/share/<name>/` and `<name>.zip`: the engine's source, the binary,
  `enginesmith.json` and a README with the results. Offer it when a run produced an engine; the user posts the README
  and the zip in the repository's Discussions (category Engines). It contains no paths, keys or model weights.
- `uv run enginesmith verify PACKAGE.zip --model models/NAME` checks a shared engine on this machine: the model files
  must match the package's (download the repository and revision its README names), then it builds the engine from
  source and runs the correctness checks and paired timing against the local mlx-lm. Tell the user to read the
  source first: it is native code that another user's coding agent wrote.
