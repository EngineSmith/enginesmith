# Pinned external AgentX diagnostics

`enginesmith agentx-check` runs AgentX-Harness **1.0.14** as an external process against an existing endpoint.
EngineSmith itself still uses only the standard library. The benchmark owns session trees, appended histories,
cache priming, warmup, profiling, draining, request headers and SSE parsing. The adapter retains its artifacts and
checks the exported effective configuration. It does not add an automatic search score or measure coding correctness.

Install the client separately, using Python 3.11-3.13, and preserve its dependency lock for your experiment:

```sh
uv venv --python 3.12 .venv-agentx
uv pip install --python .venv-agentx/bin/python 'agentx-harness==1.0.14'
uv pip freeze --python .venv-agentx/bin/python > agentx-dependencies.txt
```

Create `agentx.toml` alongside that environment:

```toml
version = 1
python = ".venv-agentx/bin/python"
url = "http://127.0.0.1:8000"
model = "served-model"
tokenizer = "tokenizer-model-id"
dataset = "semianalysis_cc_traces_weka_062126"
concurrency = 8
duration_s = 3600
timeout_s = 9000
seed = 42
session_header = "dynamo"
trust_remote_code = false
inherit_env = ["PATH", "HOME"]
```

```sh
uv run enginesmith agentx-check agentx.toml --out results/agentx-01 --dry-run
uv run enginesmith agentx-check agentx.toml --out results/agentx-01
```

The dry run validates the manifest and prints argv; it does not check the installed client or contact the server.
The actual run verifies the client's distribution version before launching. `--out` must not already exist.
The interpreter path and client working directory are relative to the manifest. A bare interpreter name uses
the child's PATH. Local tokenizer names/paths are passed unchanged to AgentX, whose cwd is the manifest directory.

`concurrency` is AgentX replay concurrency, not GPU batch size. The pinned `agentx` scenario defaults to ten
warmup requests per lane, mid-trajectory starts in the 25-75% range, and seed 42. Preparation, warmup and drain
add to the measurement duration; `timeout_s` bounds the entire client. The scenario rejects profiling windows
shorter than 900 seconds, so this adapter does too. Do not label a fake-client unit test a real AgentX profile.
The dataset name is recorded, but does not pin mutable upstream dataset files: archive and hash the dataset
snapshot separately before making reproducibility or baseline-parity claims.

`session_header` accepts `none`, `session`, `dynamo`, or `both`. It sets AgentX's correlation-derived header options;
the Dynamo option also carries the parent session ID for subagents. The adapter does not invent or rewrite IDs.
`server_metrics` optionally lists metrics URLs. `trust_remote_code` is an explicit tokenizer option, not inferred.
Unknown manifest fields and caller-provided `AIPERF_*` overrides are rejected. Only explicitly named inherited
environment variables reach the client; the adapter's two affinity flags are recorded. Non-secret extra client
settings such as cache locations can use `[env]`. Keep credentials in explicitly inherited variables, and review
raw logs/artifacts before sharing. They may contain endpoint addresses, prompts and local paths.

## Results and limits

`result.json` records process lifecycle, `workload.log` contains the client log, `aiperf_artifacts/` is AgentX's own
output, and `agentx-result.json` contains the plan plus export consistency checks. Units and percentile values are
preserved; missing metrics remain null with a reason. Warmup, branch and error summaries are kept separately.
`completed` means the client exited successfully, was not cancelled, reported completed requests and exported
the expected configuration. It does **not** mean zero failed requests, a passed quality gate, or faster inference.
The raw error summary must still be evaluated. The adapter never changes the original export or computes a score.

This is a diagnostic adapter for an already managed service. It can itself be a `service-check` workload once
services have passed readiness. No GPU or remote deployment is started automatically, and no cache is flushed.
Use independently prepared servers/caches under the same recipe when comparing baseline and candidate.
Server speculation/precision settings are not attested by the client export; those need separate runtime evidence.
Trace replay measures serving performance, not whether a live coding agent successfully edits and tests software.

The source contract is the [AgentX 1.0.14 release](https://pypi.org/project/agentx-harness/1.0.14/) and its
`agentx_cli.py`, `aiperf/common/scenario/inferencex_agentx_mvp.py` and configuration/export modules.
Tracked in [#27](https://github.com/EngineSmith/enginesmith/issues/27), under the [#25 roadmap](https://github.com/EngineSmith/enginesmith/issues/25).
