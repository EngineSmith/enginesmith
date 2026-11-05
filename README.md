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

**Status: research code under active development.** Apple Silicon (mlx-lm as the reference), AMD Instinct GPUs
on Linux (ROCm, SGLang as the reference; see the setup skill) and AWS Trainium2 (NxD Inference as the reference; see
"Trainium" below); support for other platforms (CUDA, ...) plugs in without changing the core, see
[CONTRIBUTING.md](CONTRIBUTING.md).

## This fork: AWS Trainium2

This fork ([xuanfeiren/enginesmith](https://github.com/xuanfeiren/enginesmith)) adds AWS Trainium2 as a third
platform, next to Apple Silicon and AMD Instinct, built the same way as the Apple path: coding agents write native
engines (NKI kernels compiled to NEFF at build time, embedded in a C++ executable that links only libnrt), judged
token by token against stock NxD Inference. Only public, open-source AWS material is used: the public Neuron SDK
packages and the documented NKI API.

### What it adds to upstream

| Part | Where |
|---|---|
| Platform module `linux-neuron`: neuron-ls facts, HBM budget, an ELF + ldd native check (refuses Python and framework libraries, and executables that do not link libnrt, so engines must compute on the NeuronCores), per-run NeuronCores (`ENGINESMITH_NEURON_CORE`, `ENGINESMITH_NEURON_CORES`) | `src/enginesmith/platforms/neuron.py`, optional `engine_environment()` in `platforms/__init__.py` and `protocol.py` |
| Reference `nxdi`: stock NxD Inference driven exactly like its `HuggingFaceGenerationAdapter`; judge in a worker process on separate cores; tensor parallel (`[reference] tensor_parallel`); mixture-of-experts models; a compile cache | `src/enginesmith/references/nxdi.py`, `references/__init__.py`, `task.py`, `brief.py` |
| Evaluation version 3 (adds the Trainium reference; nothing changes on Apple Silicon or AMD) | `zones.toml`, `evaluate.py` |
| Skill backend `nki-cpp`: hard rules, platform numbers, the public compile route, runtime, decode loop, kernel fusion, GEMV, attention, prefill, tensor parallel, profiling, counterexamples, and a working example | `skills/specialized-inference-engine/backends/nki-cpp/` |
| Worked example: Qwen3-0.6B at 2.2x NxD Inference | `skills/specialized-inference-engine/examples/qwen3-0.6b-trn2/` |
| Setup skill, task docs and README: Trainium sections | `skills/enginesmith-setup/SKILL.md`, `docs/task.md`, this README |
| Harness fixes found on the way: snapshots never include `target/` (a prebuilt kernel cache had reached a "clean" build); a build that fails on a busy accelerator is retried once; reference-neutral share and explain text; `--skip-git-repo-check` is no longer mistaken for an `sk-` key; a per-run GPU lock (`GPU_LOCK_FILE`); task accuracy reports the engine's speed on real prompts relative to the reference | `engine_repo.py`, `quality.py`, `share.py`, `explain.py`, `task.py`, `dev_eval.py` |
| The reviewer's scripts (clean rebuild + held-out check, task-accuracy drivers) and the raw results behind the numbers below | `contrib/trainium2/` |

### Results (trn2.48xlarge, Qwen3 bf16)

Every headline number is the reviewer's own check, not the run's score: the engine rebuilt from source with no cache,
then an independent held-out evaluation against NxD Inference 0.10.18399 (the latest public release), 5 paired rounds,
the seed drawn afterwards, every correctness gate passed. Decoding is token by token, the same algorithm as the
reference, on the same NeuronCores.

| Model | NeuronCores | Speedup | Prefill | Decode |
|---|---|---|---|---|
| Qwen3-0.6B | 1 | **2.28x** | 2.16x | 2.41x |
| Qwen3-8B | 1 | **1.51x** | 1.66x | 1.38x |
| Qwen3-14B | 2 (tensor parallel) | **1.75x** | 2.32x | 1.32x |
| Qwen3-30B-A3B (mixture of experts) | 4 (tensor parallel) | **2.87x** | 3.63x | 2.27x |

On 100 MATH-500 and 100 IFEval prompts the Qwen3-0.6B engine's task accuracy is statistically indistinguishable from
the reference's. The 8B decode reads its 16.4 GB of weights at about 95% of a logical NeuronCore's HBM bandwidth
(about 68% for the reference), and its prefill keeps the matrix engine 95-97% busy: that model is at the hardware
ceiling.

Side result, not like for like: with lossless speculative decoding (prompt-lookup drafts verified by the model, all
weights bf16; the reference does not speculate) generation on real prompts runs 5.05x (0.6B), 2.58x (8B), 1.78x
(14B) and 2.58x (30B-A3B) the reference's speed, accuracy within one standard error. Upstream's
`one_forward_per_token` rule (on by default) rules speculation out; these runs predate it.

The agents: Claude Opus 5.5, Claude Fable 5.1, GPT-6 astra and GPT-6 sol, in parallel runs on one machine, plus
kernel-lab sessions outside the harness that solved the shared problems first (KV cache written on the device, a whole
decode step in one NEFF, layered prefill, tensor parallelism, mixture of experts) and wrote the backend topics.

### Findings for the maintainers

- **The random-token timed workload rewards speculative decoding.** A small model's greedy output loops on random
  prompts, so drafts are easy: harness scores of 3.7-9.3 came with 2.6-5.4x on real text, and on Qwen3-30B-A3B a round
  scored 40% higher while running 4% slower on real text. (Upstream's `one_forward_per_token` addresses this.)
- **The timed prompt generator can be learned.** One agent recorded the model's greedy outputs on hundreds of seeds of
  the harness's random prompts and embedded the resulting n-gram table as a draft source; its harness score went from
  5.7 to 9.3 with no change on real text.
- **An unchanged resubmission can pass on timing noise.** The same code scored 2.591 and 2.466 on two draws (about
  5%), and one agent resubmitted rejected code until it was accepted. Refusing a candidate identical to a rejected one
  would close this.
- **The harness never asks for long outputs.** One speculative engine failed on long answers ("continuation exceeds
  the compiled positions") in five accepted rounds; only the task-accuracy run showed it.
- **Shared machines need process hygiene.** `pkill -f` by name killed another run's build, and a `pgrep -f` wait loop
  matched its own command line and never ended; both are rules in the skill now.
- NxD Inference answers a one-token prompt as the prediction after `[token, 0]`; engines reproduce it. neuronx-cc 2.27
  fails on some unaligned bucket sizes, so the reference rounds its compiled length to 512.

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

On every build EngineSmith checks that the engine is a native executable that links no Python or MLX library
(`[engine] native` in [docs/task.md](docs/task.md)). Shared engines publish their source, so anyone can read how the
rest is met.

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

## Trainium

On a Trainium2 instance (trn2, Neuron driver and runtime installed, as on the AWS Neuron deep learning AMI) the
platform module `linux-neuron` applies, the reference is stock NxD Inference (`references/nxdi.py`) on one logical
NeuronCore, and the coding sessions use the skill backend `backends/nki-cpp`: NKI kernels compiled to NEFF files at
build time and embedded in a C++ executable that drives them through libnrt.

```sh
uv sync --extra dev
uv pip install --index-strategy unsafe-best-match --extra-index-url https://pip.repos.neuron.amazonaws.com \
    "neuronx-distributed-inference==0.10.18399+ed62453e" "neuronx-cc==2.27.5334.0+f702b353" "islpy==2026.1" "numpy==2.3.5"
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:$PATH   # neuronx-cc and nki for engine builds
```

A task sets `[engine] build` and `executable` for the C++ build (for example `["bash", "build.sh"]` and
`"target/engine"`). Each run uses two logical NeuronCores, `ENGINESMITH_NEURON_CORE` and the next one (0 and 1 by
default), so several runs with different values can share one machine; give each its own `GPU_LOCK_FILE` as well (for example `/tmp/engine-gpu-8.lock`), the lock that `enginesmith dev-eval` and the skill's timing tools take. The reference compiles the model for the
task's longest request the first time (minutes) and caches it under `~/.cache/enginesmith/nxdi`.

## Documentation

- [How EngineSmith works](docs/design.md)
- [Contributing](CONTRIBUTING.md): engines, skill knowledge, worked examples, coding agents, other platforms and references
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
