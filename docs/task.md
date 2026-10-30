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
| `kind` | the platform's default (`mlx-lm` on Apple Silicon) | a reference of `src/enginesmith/references` |
| `command` | none | argv of another reference engine instead of `kind`; it must also answer `judge` (docs/protocol.md) |
| **[engine]** | | |
| `seed` | none | a directory with engine source to start from (copied without `.git` and `target/`) |
| `evaluate_seed` | true when `seed` is set | evaluate the seed before the first session; if it passes, it is the first best engine |
| `build` | `["cargo", "build", "--release"]` | the build command, run in a clean checkout of the candidate |
| `executable` | `target/release/engine` | the executable the build produces, relative to the source |
| `native` | true | require a Mach-O executable that links no Python or MLX library |
| `skill` | `skills/specialized-inference-engine` | the skill directory copied into every session's brief |
| **[agent]** | | |
| `kind` | the agent this shell runs in | `claude`, `codex`, `kimi`, `pi`, `dsh` or `command` (docs/design.md, sessions). Left out, it is the agent the task is loaded from (`enginesmith doctor` reports it as `current_agent`); a shell in no recognized agent must name it |
| `model` | none | passed to the CLI as `--model`; not with `dsh`, which uses the default model of its own settings |
| `command` | none | with `kind = "command"`: the argv; `{prompt}`, `{workspace}` and `{model}` are replaced. No credentials: the task is copied into every session, so API keys go in the environment or the agent's own configuration (a command that looks like it holds one is refused) |
| `session_timeout` | `2h` | a session is stopped after this long |
| **[workload]** | | |
| `lengths` | `[256, 1024, 4096, 16384]` | prompt lengths of the timed cases, 1 to 524,288 |
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

## The score

For each timed case and segment, the speedup is the median over the rounds of reference time / candidate time. The
score is their weighted geometric mean: case weight (`length_weights`) times segment weight (`prefill`,
`decode`). The segment means use the case weights only.

## Budget

Before every session the run checks that the session timeout, one evaluation and the final evaluation still fit
before the wall time (the evaluation time is the longest one measured so far, or the estimate from
`prepare.json` next to the task, or one hour). If not, the search ends and the final evaluation runs.
