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

**Status: research code under active development.** Apple Silicon (mlx-lm as the reference) and AMD Instinct GPUs
on Linux (ROCm, SGLang as the reference; see the setup skill); support for other platforms (CUDA, ...) plugs in
without changing the core, see [CONTRIBUTING.md](CONTRIBUTING.md).

## What a specialized engine is

A specialized engine is a native program written for one fixed target: one model checkpoint, one machine, one
workload. A general engine (mlx-lm, llama.cpp, vLLM) has to serve every model, shape and machine. A specialized
engine serves one, and is free to exploit everything it knows about it. That narrowness is the point of EngineSmith.

- **One fixed target.** The task fixes the checkpoint, the machine and the workload (the prompt lengths, and how much
  prefill counts against decode); a shared engine records the SHA-256 of every model file. The engine may treat all
  of them as constants: the layer count, the dimensions, the quantization group size, the maximum context, the
  chip's parameters. It need not run any other model or shape; outside its target it may refuse.
- **It owns the model's computation.** The kernels and the host code that schedules them are the engine's own
  source, written or adapted for this model (from any open-source project, under its license), so every step can be
  specialized and fused. It does not hand the model to a framework, graph compiler or inference runtime (MLX,
  PyTorch, Core ML, llama.cpp and the like).
- **Same model, same answers.** Specialization changes how the model is computed, not which model it is: the engine
  reads the checkpoint as downloaded, and its tokens must match the reference's within the correctness gates. By
  default it computes one forward pass per generated token: speculative decoding is a different algorithm from the
  reference's, so a task has to allow it (`[engine] one_forward_per_token = false`), and its results say so.
- **Native and self-contained.** The build produces one native executable that needs nothing but the model
  directory. At run time it loads no Python and starts no Python process. The tools used to develop and build it
  may be anything, Python included: only the executable counts.

**What an engine may link.** Only the operating system's and the C/C++ runtime's libraries, the accelerator's driver
and runtime layer (which allocates memory, loads the engine's own kernels and launches them: Metal on Apple Silicon,
the HIP runtime on AMD), and utilities that do no arithmetic on the model. Anything that computes on the model is
refused: frameworks, inference runtimes and precompiled numerical libraries (MLX, PyTorch, Core ML, llama.cpp, MPS,
Accelerate, rocBLAS); their open-source kernels may be copied into the engine's source instead. A library call is a
wall that no fusion or specialization crosses, and with a library doing the arithmetic the score would measure the
library, not the engine. Every platform checks its own list of allowed libraries on every build (`[engine] native` in
[docs/task.md](docs/task.md)); [docs/libraries.md](docs/libraries.md) explains each one, with a MacBook as the
example. Shared engines publish their source, so anyone can read how the rest is met.

**How it differs from a general engine.** A general engine assembles the model at run time: it reads the architecture
and the sizes from the model file, strings generic operators (matrix multiplication, normalization, attention, ...)
into a computation graph, and runs the graph node by node, choosing a kernel for each node's type and shape and
writing each result to memory for the next node to read. That is what lets one program serve hundreds of models. A
specialized engine has no graph: its code is this model's computation in a fixed order, so it can fuse any steps and
keep intermediate results on the chip.

| | A general engine (llama.cpp) | An EngineSmith engine |
|---|---|---|
| Serves | hundreds of models with one program | one model on one machine |
| Architecture and sizes | read from the model file at run time | constants in the code |
| Execution | a graph of generic operators, assembled at run time | a fixed sequence written for this model |
| Kernels | each handles many shapes and quantization formats | written for this model's shapes and format |
| Fusion | the combinations it implements generically | any steps, for example a whole layer |
| Tuned for | many chips | one chip |
| Written by | people, over years | coding agents, per model (Molten: 78 sessions) |
| A new model | runs at once | needs a new engine |

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
7. **Checks on the speed.** The report and the shared README show two facts next to the score, neither gated: the
   speed on those natural-text prompts, and a decode probe, the same decode asked for in one request and one token
   per request. One forward pass per generated token keeps the probe near 1; speculative decoding raises it.

The workload is raw token ids from a public formula, from 1 to 524,288 tokens, so it serves any model without a
tokenizer. The reference is stock mlx-lm, so a new model needs no framework code. [docs/design.md](docs/design.md)
has the details.

## Roadmap

Not implemented yet; each item has an issue for the design.

- **Batches.** Every request is one sequence today (batch size 1). Planned: batch sizes as a second workload
  dimension next to prompt lengths, each with its own weight in the score
  ([#14](https://github.com/EngineSmith/enginesmith/issues/14)).
- **Speculative decoding as part of the model.** A task that declares the algorithm and its draft weights, with the
  reference running the same, timed on real text ([#3](https://github.com/EngineSmith/enginesmith/issues/3)).
- **Judging.** A timed workload agents cannot learn ([#12](https://github.com/EngineSmith/enginesmith/issues/12)),
  and no acceptance for a rejected candidate resubmitted unchanged
  ([#13](https://github.com/EngineSmith/enginesmith/issues/13)).
- **More platforms.** AWS Trainium2 is in review ([#11](https://github.com/EngineSmith/enginesmith/pull/11)); NVIDIA
  GPUs plug in the same way ([CONTRIBUTING.md](CONTRIBUTING.md)).

## Share an engine, check someone else's

When a run is done, your agent asks whether to share the engine. You can also just say so at any time, in your own
words ("share my engine", "分享"): the agent tells you which engine it will share, exactly what will be published and
where, what is left out and what to check, and does nothing until you choose. Behind it is one command:

```sh
uv run enginesmith share tasks/NAME/runs/RUN                       # publish: your repository + a Discussions post
uv run enginesmith verify https://github.com/USER/REPO --model models/MODEL_DIR   # check someone else's engine here
```

`share` puts the engine's source and all the run's data (`enginesmith.json`, `history.csv`, a README with the results,
the speeds in tokens per second, how the search went and which agent, model and reasoning effort ran it) in a
repository of your GitHub account, and posts the README with a link in this repository's Discussions (category
Engines). The engine is published under the MIT license in your name (`--license Apache-2.0` or `none` to change it).
It uses the GitHub CLI and your login (`gh auth login`); `--private` makes the repository private, `--no-post` only
writes the package locally. The package holds no host paths, keys, session logs or model weights; `--binary` adds the
built binary, which holds paths of your machine such as your home directory. `verify` checks the model files by
SHA-256, builds the engine from source and runs the same correctness checks and paired timing on your machine, so a
shared result never has to be taken on trust. Read the source before you run an engine someone else wrote.

## Documentation

- [How EngineSmith works](docs/design.md)
- [Contributing](CONTRIBUTING.md): engines, skill knowledge, worked examples, coding agents, other platforms and references
- [What an engine may link](docs/libraries.md), with a MacBook as the example
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

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
