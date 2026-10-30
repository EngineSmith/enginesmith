# Contributing to EngineSmith

This file is written for contributors and for their coding agents: read it before changing anything. Find your kind
of contribution in the table, then read its section. None of them changes how candidates are judged.

## Ways to contribute

| contribution | where | example | code? |
|---|---|---|---|
| Knowledge for an existing backend | `skills/specialized-inference-engine/backends/metal-rust/` | a 4-bit GEMV variant that is faster on M4 Pro, with the measurement; a fusion that did not pay off, as a counterexample | no |
| A worked example for a model family | `skills/specialized-inference-engine/examples/<model>/` | Qwen3.5 (gated DeltaNet and full attention) from a finished run: model card, decode map, optimization log | no |
| A whole skill | `skills/<name>/`, used through `engine.skill` | a skill for engines in C++ and Metal, or one specialized in long-context prefill | no |
| Another coding agent | `src/enginesmith/agents.py` and the docs | Gemini CLI, Qwen Code, OpenCode | a few lines |
| Another platform | `platforms/`, `references/` and a skill backend | Linux with NVIDIA GPUs and vLLM as the reference; AMD GPUs with ROCm | yes |
| Another reference on a supported platform | `src/enginesmith/references/` | llama.cpp's Metal backend as a second reference on Apple Silicon | yes |
| An engine | the Discussions, category Engines (not a pull request) | your Qwen3-8B engine for an M4 Max, packaged by `enginesmith share` | no |
| A fix or feature in the harness | `src/enginesmith/` | a bug fix; another accuracy benchmark next to MATH-500 and IFEval | yes |

Sections below: [skill knowledge](#contributing-skill-knowledge), [a coding agent](#supporting-another-coding-agent),
[a platform or a reference](#supporting-another-platform), [the harness](#changing-the-harness-itself). Sharing an
engine is described in the README ("Share an engine, check someone else's"). Open an issue first for anything that changes how candidates
are judged.

## Map of the code

Hardware-agnostic: the core. A new platform does not touch these files.

| module | what it owns |
|---|---|
| `src/enginesmith/task.py` | the task file schema (`docs/task.md`) |
| `src/enginesmith/workload.py` | the Lily prompts and the public / held-out / nonce index ranges |
| `src/enginesmith/protocol.py` | the engine protocol client and its timing (`docs/protocol.md`) |
| `src/enginesmith/evaluate.py` | controls, paired AB/BA rounds, teacher-forced judging, gates, speedups |
| `src/enginesmith/acceptance.py` | the score and the acceptance rules |
| `src/enginesmith/run.py`, `engine_repo.py`, `agents.py`, `brief.py` | rounds, snapshots, builds, sessions, the session brief |
| `src/enginesmith/dev_eval.py`, `explain.py`, `report.py`, `share.py`, `quality.py`, `setup_tools.py`, `cli.py` | the commands |

Platform-specific: where a new platform plugs in.

| place | what it holds | today |
|---|---|---|
| `src/enginesmith/platforms/<name>.py` | machine facts, checks, memory budget and watchdog, sleep inhibition, the native-executable check, the default reference | `macos.py` (Apple Silicon), `generic.py` (fallback) |
| `src/enginesmith/references/<name>.py` | a reference implementation behind the engine protocol | `mlx_lm.py` |
| `skills/specialized-inference-engine/backends/<name>/` | how to build fast engines on the platform, with measured numbers | `metal-rust/` |

## Supporting another platform

Example: Linux with an NVIDIA GPU. Do the three parts in this order; each has its own checks.

### 1. A platform module: `src/enginesmith/platforms/<name>.py`

Start from a copy of `platforms/generic.py`, which is the smallest complete implementation, and read the interface in
`platforms/__init__.py`. Then add the module to `PLATFORMS` there, before `generic`. Every module must define:

| name | contract | NVIDIA hint (not implemented, not measured) |
|---|---|---|
| `NAME` | a short id, e.g. `linux-cuda` | |
| `DEFAULT_REFERENCE` | a key of `references.KINDS`, or `None` | your reference's kind |
| `supported()` | true when the module applies to this machine; cheap, no side effects | Linux and `nvidia-smi` lists a GPU |
| `machine()` | dict with `platform`, `os`, `arch`, `chip`, `accelerator`, `accelerator_cores`, `memory_bytes` (the memory a model can use: unified memory or VRAM), plus anything useful | `nvidia-smi --query-gpu=name,memory.total,...` |
| `checks(facts)` | `(problems, notes)`: problems block a run (missing driver or compiler), notes inform | driver and `nvcc` present |
| `memory_budget()` | bytes the model and its KV cache may use; `enginesmith probe` uses it to say which prompt lengths fit | a fraction of free VRAM |
| `memory_critical()` | true when memory is about to run out; the watchdog then kills every engine it started | host `MemAvailable` (`/proc/meminfo`) below a few percent |
| `keep_awake()` | keep the machine from sleeping while the run's process lives, or `None` | `systemd-inhibit`, or `None` on servers |
| `native_problem(path)` | `None` when `path` is a self-contained native executable, else why not; it must refuse scripts and executables that load Python or an inference runtime, and may allow system and driver libraries | ELF magic, then `ldd`: allow `libcuda` / `libcudart`, refuse `libpython`, `libtorch` and the like |

`tests/test_platforms.py` checks every module in `PLATFORMS` against this interface. Add CPU tests for your parsing,
for example with recorded `nvidia-smi` output, so that the suite still passes on any machine.

### 2. A reference: `src/enginesmith/references/<name>.py`

A second reference on a platform that already has one needs only this part; a task chooses it with
`[reference] kind`.

The reference decides both what is correct and what "faster" means, so it is the most consequential choice. Use the
implementation that practitioners on that platform consider the fast standard (for example vLLM, llama.cpp or
TensorRT-LLM), run it unmodified (no patches, no monkey patching), and reproduce its own generation path exactly.
`references/mlx_lm.py` (about 150 lines) is the model to follow.

It runs as `python -m enginesmith.references.<name> --model MODEL_DIR` and answers, one JSON object per line
(`docs/protocol.md`):

- `hello`: load the model; reply `{"protocol": 1}`.
- `generate` (`prompt`, `max_tokens`, `reset`): greedy decoding with ties broken towards the lowest token id, exactly
  `max_tokens` tokens, no stop tokens. A continuation (`reset: false`, empty prompt) first consumes the last emitted
  token. Use the library's own prefill partition and step order; never compute a step past the last requested
  token, and finish all device work before replying.
- `judge` (`prompt`, `tokens`): teacher-force `tokens` after `prompt` along the same computation path as generation.
  For each step whose token's log-probability is below the row's maximum, report `[step, margin, the reference's
  token, the token's rank]` in `mismatches` (an exact tie is agreement; rank 2 is the second choice).
- `stats` (`reset_peak`): `peak_memory` and `active_memory` in bytes.

Register it in `references/KINDS` (module and Python package, whose version identifies it) and add its package as an
optional dependency extra in `pyproject.toml`, the way `mlx` is.

Validate it on the platform, and put the evidence in the pull request:

1. Its `generate` equals the library's own generation, token for token, at prompt lengths 1, 2, 63, 2047, 2048,
   2049, 4097 and 8192 (and at the platform's own chunk boundaries).
2. A split request (`generate(P, 1)` then `generate((), 64)`) and a chain of continuations equal one `generate(P, 65)`.
3. `judge` on its own tokens reports no mismatch; on a stream of one repeated token it reports early mismatches
   ranked in the hundreds or worse.
4. An evaluation of the reference against itself (`enginesmith.evaluate.Evaluator`, the reference command as both
   arms) passes, with every speedup within a few percent of 1.0.
5. `enginesmith prepare` runs at the long lengths you claim the platform supports.

If the library cannot be made deterministic for greedy decoding (batch-dependent kernels, atomics), say so: the gates
assume that the reference agrees with itself.

### 3. A skill backend: `skills/specialized-inference-engine/backends/<name>/`

The coding sessions learn the platform here. Follow "Adding a backend" in
`skills/specialized-inference-engine/backends/README.md`: a README with the platform's hard rules (toolchain,
compiler numerics, feature gates) and a table of numbers you measured on the target machine, then one topic file per
`metal-rust` topic, starting with what a correct baseline needs. Every number is measured on a named machine or
labelled as an estimate; Metal results are hypotheses on another platform, never facts. Register the backend in the
skill's `SKILL.md` and in the status table of `backends/README.md`. English only, no host paths;
`tests/test_skills.py` checks the files.

### 4. Then

- `skills/enginesmith-setup/SKILL.md` and `README.md`: installation and requirements for the platform.
- An end-to-end check before the pull request: `enginesmith doctor` reports no problem; `enginesmith prepare` on a
  small model; a run of two or three sessions on a small dense model (Qwen3-0.6B, for example) that ends with the
  final evaluation. Attach its `report.json`.

## Contributing skill knowledge

The default skill (`skills/specialized-inference-engine/`) is what every coding session reads before it writes a line
of the engine, so a good page there helps every later run. Pull requests here need no code.

- **Backend knowledge.** Add to the topic file it belongs to (`backends/metal-rust/attention.md`, `prefill.md`,
  `kernels-gemv-quant.md`, ...): what to do, why, and the measurement that shows it, with the machine, the model and
  how it was measured. A technique that did not pay off belongs in `counterexamples.md`, with the numbers.
- **A worked example.** A finished run on another model family becomes `examples/<model>/`, laid out like
  `examples/lfm2.5/`: a README (the checkpoint and revision, the machine, the reference version, the result from the
  run's held-out final evaluation), the model card, a decode map (what one decode token launches, in order) and an
  optimization log (what each accepted round changed and what it gained, from the run's report). Link it from the
  skill's `SKILL.md`.
- **A whole skill.** A task can use another skill with `engine.skill`; contribute it as `skills/<name>/` with a
  `SKILL.md` whose frontmatter names the directory, and add it to `test_both_skills_exist` in
  `tests/test_skills.py`. It must keep the contract the brief states (the protocol, the build, the native
  executable) and may change everything else.

For every skill file: every number is measured on a named machine or labelled as an estimate; English only; no host
paths or private files; each document small enough for one read (under 50 KB); helper scripts use the standard
library. `tests/test_skills.py` checks these. Strategy belongs here, not in the harness.

## Supporting another coding agent

The sessions run on the user's own coding agent CLI; `claude`, `codex`, `kimi`, `pi` and `dsh` are built in. Adding
one takes a few lines, plus a real run:

1. `TEMPLATES` in `src/enginesmith/agents.py`: the argv of its one-shot headless mode, with `{prompt}` (and
   `{workspace}` if it takes the directory as a flag). It must ask for no approval and apply no sandbox of its own
   (a session builds and runs GPU code), and should print JSON events if it can, so the session log is a full
   transcript. If it takes a model flag, check that `--model` works or adjust `argv()`.
2. `MARKERS` in the same file: the environment variables it sets for the commands it runs, so that `enginesmith
   doctor` reports it as `current_agent` and a task defaults to it. Add them to `PARENT_SESSION` too, so a session
   never attaches to the user's own session.
3. `AGENT_KINDS` in `src/enginesmith/task.py` and `AGENT_NAMES` in `src/enginesmith/explain.py`; `cost_usd()` in
   `agents.py` if its output reports the cost.
4. Docs: `docs/task.md` (`agent.kind`), `docs/design.md` (the launchers), and section 1 of the setup skill (how to
   install it, how it authenticates headless, the one-line headless check).
5. Tests: its argv in `test_the_agent_launchers` and its markers in `test_the_current_agent_comes_from_its_markers`
   (`tests/test_cli.py`).
6. A real check before the pull request, with the released version of the agent: its headless check answers; run
   inside it, `enginesmith doctor` reports it as `current_agent`; a run of two sessions on a small model ends with
   the final evaluation. Attach the `report.json` and say which version you tested. Released versions can lag the
   agent's documentation: DeepSeek Harness 0.2.1 had no permission flags yet, so its launcher sets
   `DSH_PERMISSION_MODE` instead.

## Changing the harness itself

- `uv run pytest` must pass (CPU only; no model, GPU or agent). Add tests for what you change.
- The harness (`src/enginesmith`) uses the standard library only; optional extras (`mlx`, `quality`) serve the
  reference and the accuracy benchmarks.
- What judges a candidate (the workload, the reference, the gates, the timing, the acceptance rules, the held-out
  prompts) must stay out of the coding agents' reach. Open an issue before changing how it judges.
- A session's brief carries facts only; strategy belongs in the skill.
- Keep observations apart from estimates; a missing measurement is null with a reason, never zero.
- Documentation follows the code: change `docs/` in the same pull request.
