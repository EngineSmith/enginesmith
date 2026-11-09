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
| `kind` | the platform's default (`mlx-lm` on Apple Silicon, `nxdi` on AWS Trainium) | a reference of `src/enginesmith/references`; `transformers-cpu` is stock transformers on the host CPU, a correctness-only reference for models the platform's own reference cannot run |
| `command` | none | argv of another reference engine instead of `kind`; it must also answer `judge` (docs/protocol.md) |
| `correctness_only` | true for `transformers-cpu`, else false | the reference decides what is correct but is no speed baseline: nothing reports a speedup, and the score is the fraction of the hardware bound (below). Use it only when no framework runs the model correctly on this platform |
| `frameworks` | required with `correctness_only`, else not allowed | the frameworks tried on this platform and how each failed, as free text; the plan, the report and the shared README show it |
| `tensor_parallel` | 1 | the reference's tensor-parallel degree (`nxdi`), which is also the number of accelerator cores the engine gets; with a correctness-only reference, only the engine's cores |
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

## The score

For each timed case and segment, the speedup is the median over the rounds of reference time / candidate time. The
score is their weighted geometric mean: case weight (`length_weights`) times segment weight (`prefill`,
`decode`). The segment means use the case weights only. With a correctness-only reference the fractions of the
hardware bound (below) take the speedups' place.

## The hardware bound

`src/enginesmith/bound.py` estimates the least time this machine needs for each timed request, from the checkpoint and
the platform's figures (`peak()` of the platform module, each figure with its source; for Trainium2, 725 GB/s and
about 167 dense BF16 TFLOPS per logical NeuronCore, times the cores the engine gets):

- decode: bytes read per generated token / memory bandwidth: every weight tensor of the safetensors headers (the
  input embedding only when it is also the output head) plus the KV cache at that position (per layer K and V, KV
  heads x `head_dim` x the bytes of the config's dtype, at most the window of a sliding-window layer);
- prefill: FLOPs / peak compute: 2 x parameters x prompt tokens, the output head for the last token only, and the
  causal attention's 4 x heads x `head_dim` per (query, key) pair.

The fraction of the bound is bound time / engine time, the median over the rounds. It is computed for dense models
only; a mixture-of-experts or hybrid model, a quantized checkpoint (prefill), or a platform without figures gets "not
computed" and the reason. The report and the shared README show the fraction next to every case's speedup; it is an
estimate and not scored.

With a correctness-only reference it is the score: the weighted geometric mean of the fractions, with the same
weights as the speedups. Where the bound is not computed a nominal 1 ms per prefill request and per decode token
stands in, so the score stays a consistent relative measure (reported as "not computed"). The acceptance rules are
unchanged: they compare two candidates case by case, so each case's bound cancels.

## Budget

Before every session the run checks that the session timeout, one evaluation and the final evaluation still fit
before the wall time (the evaluation time is the longest one measured so far, or the estimate from
`prepare.json` next to the task, or one hour). If not, the search ends and the final evaluation runs.
