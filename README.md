# EngineSmith

EngineSmith builds a specialized inference engine for one model on your machine. You tell your coding agent which
model you want to run and what matters to you: fast decoding, fast prompt processing, long context, or one
particular shape. Your agent sets up the task. EngineSmith then runs coding sessions that write and optimize the
engine, and judges every candidate against a reference implementation: are the tokens right, and is it faster?

EngineSmith is a harness. The agents write the engine. EngineSmith owns everything that decides whether they
succeeded: the workload, the reference, the correctness gates, the timing, the acceptance rules and the budget.

Its first engine, for LFM2.5-24B-A2B on an Apple M5 Max, was released as
[Molten](https://github.com/zhangshuibai/molten). Over 78 agent sessions it went from 0.56x to 1.25x the
end-to-end speed of mlx-lm.

**Status: research code under active development.** Apple Silicon only for now; support for other platforms
(CUDA, ROCm, ...) plugs in without changing the core, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Quick start

```sh
git clone https://github.com/EngineSmith/enginesmith && cd enginesmith
uv sync --extra mlx --extra quality --extra dev
```

Then open your coding agent (Claude Code, Codex CLI, DeepSeek Harness, ...) in the repository and say what you want, for example:

> Build an engine for Qwen3-8B (4-bit) on this Mac. I mostly care about decode speed for chat.

The agent follows [`skills/enginesmith-setup`](skills/enginesmith-setup/SKILL.md). It checks the machine, downloads
the model with your permission, measures the reference here, writes the task, launches the run and keeps you
informed. Before anything starts it shows you the whole plan (`plan.md`): each prompt length's share of the score,
the correctness gates, how the search evolves and when a candidate is accepted. By default the coding sessions run on
the same agent you are talking to, headless, with your own login or API key: Claude Code (`claude`), Codex (`codex`),
Kimi Code (`kimi`), Pi (`pi`) or DeepSeek Harness (`dsh`), or any other through `agent.kind = "command"`.

Without an agent, the same steps by hand:

```sh
uv run enginesmith doctor                          # machine, toolchain and agent CLIs
uv run enginesmith probe models/NAME               # model facts, which prompt lengths fit
uv run enginesmith validate tasks/NAME/task.toml   # see docs/task.md
uv run enginesmith prepare tasks/NAME/task.toml    # the reference's times and memory on this machine
uv run enginesmith explain tasks/NAME/task.toml    # writes plan.md: cases, score, gates, acceptance, budget
uv run enginesmith run tasks/NAME/task.toml --detach
uv run enginesmith status tasks/NAME/runs/RUN
```

## How a run works

1. **Rounds.** Each round is one candidate. A fresh, headless coding agent edits the engine source, starting from
   the best engine so far (or from scratch). Its brief holds the contract, the facts from earlier rounds, the
   default skill and the model card. EngineSmith commits the source, builds a clean checkout and evaluates it.
2. **Correctness.** Control prompts first, then every timed request. A token stream identical to the reference's
   agrees. A stream that differs is replayed through the reference with the candidate's tokens forced; the gates
   cap the steps where the candidate's token is not the reference's top choice, and how far down the reference's
   ranking it may be.
3. **Speed.** Reference and candidate alternate in paired rounds (AB, BA, ...) on prompts drawn fresh for every
   request. Each case times a prefill request and a decode request. The score is the weighted geometric mean of
   the speedups, weighted the way you asked.
4. **Acceptance.** A correct candidate replaces the best engine only when it is faster by more than the timing
   noise. Rejected candidates and their reasons go to the next sessions as facts.
5. **Final.** After the last session, the best engine is evaluated on held-out prompts drawn from a seed created
   at that moment.
6. **Task accuracy.** Then the best engine and mlx-lm both answer MATH-500 and IFEval with greedy decoding, and the
   report shows both accuracies side by side.

The workload is raw token ids from a public formula, from 1 to 524,288 tokens, so it serves any model without a
tokenizer. The reference is stock mlx-lm, so a new model needs no framework code. [docs/design.md](docs/design.md)
has the details.

## Share an engine, check someone else's

```sh
uv run enginesmith share tasks/NAME/runs/RUN                     # a zip with the source, the binary and every result
uv run enginesmith verify PACKAGE.zip --model models/MODEL_DIR    # build it from source here, compare with your mlx-lm
```

Post the zip, with the README the package contains, in the repository's Discussions (category Engines). `verify`
checks the model files by SHA-256, builds the engine from the package's source and runs the same correctness checks
and paired timing on your machine, so a shared result never has to be taken on trust. Read the source before you
run an engine someone else wrote.

## Documentation

- [How EngineSmith works](docs/design.md)
- [Contributing](CONTRIBUTING.md): skill knowledge, worked examples, coding agents, other platforms and references
- [The task file](docs/task.md)
- [The engine protocol](docs/protocol.md)
- [The setup skill](skills/enginesmith-setup/SKILL.md), for the user's coding agent
- [The default engine skill](skills/specialized-inference-engine/SKILL.md), for the coding sessions: correctness,
  measurement, profiling and fusion, with an Apple Metal + Rust backend and the LFM2.5 engine as a worked example.
  Other backends can be added under its `backends/` directory.

## Development

```sh
uv run pytest            # CPU only; no model, GPU or agent needed
uv run enginesmith --help
```

## License

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). EngineSmith was created and is led by
[Shuibai Zhang](https://zhangshuibai.github.io/).
