# Contributing to EngineSmith

This file is written for contributors and for their coding agents: read it before changing anything. The most
common contribution is support for another platform (CUDA, ROCm, ...); it needs three new pieces and no change to the
core.

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

## Changing the harness itself

- `uv run pytest` must pass (CPU only; no model, GPU or agent). Add tests for what you change.
- The harness (`src/enginesmith`) uses the standard library only; optional extras (`mlx`, `quality`) serve the
  reference and the accuracy benchmarks.
- What judges a candidate (the workload, the reference, the gates, the timing, the acceptance rules, the held-out
  prompts) must stay out of the coding agents' reach. Open an issue before changing how it judges.
- A session's brief carries facts only; strategy belongs in the skill.
- Keep observations apart from estimates; a missing measurement is null with a reason, never zero.
- Documentation follows the code: change `docs/` in the same pull request.
