# The task file

A task is one TOML file. Relative paths are resolved from the file's directory. Unknown keys are errors. A run
copies the resolved task into `RUN/task.json` when it starts; nothing changes it afterwards.

| key | default | meaning |
|---|---|---|
| `name` | required | 1 to 64 letters, digits, `.`, `_` or `-`; names the run directories |
| **[model]** | | |
| `path` | required | the model directory, as downloaded (MLX format for the built-in reference) |
| `card` | none | the model card every session reads (`brief/model-card.md`) |
| `source` | none | the Hugging Face repository the model came from, e.g. `mlx-community/Qwen3-0.6B-4bit`; `enginesmith share` records it |
| **[reference]** | | |
| `kind` | the platform's default (`mlx-lm` on Apple Silicon, `nxdi` on AWS Trainium) | a reference of `src/enginesmith/references` |
| `command` | none | argv of another reference engine instead of `kind`; it must also answer `judge` (docs/protocol.md) |
| **[engine]** | | |
| `seed` | none | a directory with engine source to start from (copied without `.git` and `target/`) |
| `evaluate_seed` | true when `seed` is set | evaluate the seed before the first session; if it passes, it is the first best engine |
| `build` | `["cargo", "build", "--release"]` | the build command, run in a clean checkout of the candidate |
| `executable` | `target/release/engine` | the executable the build produces, relative to the source |
| `native` | true | require a native executable that links only the platform's allowed libraries ([libraries.md](libraries.md)) |
| `one_forward_per_token` | true | the session brief requires one forward pass per generated token, which rules out speculative decoding. The harness cannot check it; shared engines publish their source. `false` allows speculative decoding, and the plan, the report and the shared README say the speedups are not like for like |
| `skill` | `skills/specialized-inference-engine` | the skill directory copied into every session's brief |
| **[agent]** | | |
| `kind` | the agent this shell runs in | `claude`, `codex`, `kimi`, `pi`, `dsh` or `command` (docs/design.md, sessions). Left out, it is the agent the task is loaded from (`enginesmith doctor` reports it as `current_agent`); a shell in no recognized agent must name it |
| `model` | none | passed to the CLI as `--model`; not with `dsh`, which uses the default model of its own settings |
| `command` | none | with `kind = "command"`: the argv; `{prompt}`, `{workspace}` and `{model}` are replaced. No credentials: the task is copied into every session, so API keys go in the environment or the agent's own configuration (a command that looks like it holds one is refused) |
| `session_timeout` | `2h` | a session is stopped after this long |
| **[workload]** | | |
| `lengths` | 4,096 doubling up to the model's context | prompt lengths of the timed cases, 1 to 524,288. Left out: 4,096, 8,192, 16,384 and so on, then the longest prompt that fits the context of `config.json` (`max_position_embeddings`) with the generated tokens, at most 524,288; for a 262,144-token model 4,096 to 131,072 and 262,079 |
| `output_tokens` | 65 | tokens generated per case: a prefill request of 1 token and a decode request of the rest |
| `controls` | true | also check the correctness-only control prompts (1 to 4,097 tokens) |
| `vocab_size` | the model's `config.json` | the bound for prompt token ids |
| **[objective]** | | |
| `prefill`, `decode` | 0.5, 0.5 | segment weights of the score |
| `length_weights` | 1 for every length | a table `"LENGTH" = weight`; 0 keeps a length checked but unscored |
| **[acceptance]** | | 0 turns a rule off |
| `min_improvement` | 0.015 | accept when the score improves by this much |
| `segment_improvement` | 0.015 | accept when the prefill or decode mean improves by this much, the other one and the score at most `max_regression` worse |
| `case_improvement` | 0.04 | accept when one case's prefill or decode speedup improves by this much, the score at most `max_regression` worse |
| `max_regression` | 0.01 | see above |
| **[gates]** | | |
| `case_mismatch_percent` | 20 | mismatching steps allowed per token stream |
| `total_mismatch_percent` | 10 | mismatching steps allowed over the whole evaluation |
| `mismatch_rank` | 100 | a mismatching token must be among the reference's top N choices; 0: no limit |
| **[evaluation]** | | |
| `rounds` | 5 | paired rounds per evaluation (AB, BA, ...) |
| `warmup` | 1 | untimed requests at the start of every engine process |
| `cooldown` | `15s` | idle time before every engine process starts |
| `final_rounds` | 5 | rounds of the held-out final evaluation |
| **[budget]** | | |
| `sessions` | 24 | coding sessions in the run |
| `wall_time` | `48h` | the whole run, including the final evaluation |

| **[quality]** | absent: not run | task accuracy after the final evaluation (docs/design.md), reported, not gated |
| `benchmarks` | `["math500", "ifeval"]` | which benchmarks |
| `samples` | 100 | items per benchmark (a fixed subset) |
| `max_tokens` | 8192 | output tokens per item (greedy decoding) |
| `time_limit` | `2h` | for all benchmarks and both arms; the budget reserves it |

Durations are seconds (a number) or strings such as `"90s"`, `"45m"` and `"2h"`.

## Concurrent tasks

`[workload] mode = "concurrent"` judges an engine serving several independent requests at once (protocol 2 in
[protocol.md](protocol.md), the evaluation in [design.md](design.md)). Every value below must be written in the task
file; `preset` only records which starting point they were copied from, and the plan lists what differs from it.
`workload.lengths`, `workload.output_tokens`, `objective.prefill`, `objective.decode` and `objective.length_weights`
do not apply. Concurrent tasks run on Apple Silicon with the mlx-lm reference for now, without `[quality]`, the decode
probe or sharing.

| key | meaning |
|---|---|
| **[workload]** | |
| `mode` | `"concurrent"` |
| `preset` | optional: `batch`, `interactive`, `service` or `longdoc` (below) |
| `arrival` | `"closed"`: a client sends its next request as soon as its last one has all its tokens (the only kind for now) |
| `levels` | the concurrency levels, distinct integers in [1, 256]: level C runs C clients |
| `requests_per_level` | requests per level, at least the highest level; request i goes to client i mod C |
| `prompt_range`, `output_range` | `[low, high]`: each request's prompt and output lengths are drawn log-uniformly from them; outputs at least 2 tokens |
| `stagger` | the clients' first requests start at random times within this duration, e.g. `"0.2s"` |
| `controls` | as for single-request tasks: the control prompts, each sent alone |
| **[objective]** | |
| `throughput`, `ttft`, `latency` | segment weights: throughput (first request to last token), time to first token (90th percentile over the requests), normalized latency (end-to-end time / output tokens, 90th percentile) |
| `level_weights` | a table `"LEVEL" = weight` for every level |

The request plan (every request's lengths, its client, the clients' start times) is drawn once per run and used by
every round; every round draws fresh prompt windows; the final draws a new plan and held-out windows. The acceptance
rules are those of single-request tasks with levels for prompt lengths and the three segments for prefill and decode.

The presets:

| preset | levels | level weights | throughput / ttft / latency | prompt tokens | output tokens |
|---|---|---|---|---|---|
| `batch` (bulk processing: total throughput) | 4, 16 | 0.3, 0.7 | 1 / 0 / 0 | 256-4,096 | 64-512 |
| `interactive` (one user, single turns) | 1, 4 | 0.5, 0.5 | 0.4 / 0.3 / 0.3 | 256-4,096 | 64-512 |
| `service` (several users of one machine) | 4, 8, 16 | 1/3 each | 0.6 / 0.2 / 0.2 | 256-4,096 | 64-1,024 |
| `longdoc` (questions on long documents) | 1, 4 | 0.6, 0.4 | 0 / 0.75 / 0.25 | 8,192-65,536 | 64-256 |

```toml
[workload]
mode = "concurrent"
preset = "interactive"
arrival = "closed"
levels = [1, 4]
requests_per_level = 64
prompt_range = [256, 4096]
output_range = [64, 512]
stagger = "0.2s"

[objective]
throughput = 0.4
ttft = 0.3
latency = 0.3
level_weights = { "1" = 0.5, "4" = 0.5 }
```

## The score

For each timed case and segment, the speedup is the median over the rounds of reference time / candidate time. The
score is their weighted geometric mean: case weight (`length_weights`) times segment weight (`prefill`,
`decode`). The segment means use the case weights only.

## Budget

Before every session the run checks that the session timeout, one evaluation and the final evaluation still fit
before the wall time (the evaluation time is the longest one measured so far, or the estimate from
`prepare.json` next to the task, or one hour). If not, the search ends and the final evaluation runs.
