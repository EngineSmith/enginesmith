# Local service diagnostics

`enginesmith service-check MANIFEST --out DIRECTORY` starts local foreground services, waits for each to become
ready, runs an external workload, and stops its process groups in reverse launch order. It retains lifecycle
results and logs in a fresh directory. This is an opt-in diagnostic foundation for long-lived service experiments.
It does not change the stdio engine protocol, task schema, scoring, acceptance, or evaluation version, and is not
called by `enginesmith run`.

## Try it without a model or GPU

Save this as `services.toml`. Choose an unused local port and run from a directory whose contents you can expose
to a local HTTP client. The Python interpreter must be on `PATH` (or use its absolute path in each command).

```toml
version = 1

[[services]]
name = "frontend"
command = ["python3", "-m", "http.server", "8765", "--bind", "127.0.0.1"]
ready = ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/', timeout=1).close()"]
timeout_s = 20

[workload]
command = ["python3", "-c", "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8765/', timeout=1).status)"]
timeout_s = 10
```

```sh
uv run enginesmith service-check services.toml --out service-results/first-check
```

The command prints JSON and writes `result.json`, `frontend.log`, `frontend.ready.log` and `workload.log`.
It refuses an existing output directory. Exit code 0 means the services stayed alive and the workload exited
successfully; 1 means a lifecycle/workload failure; 2 means invalid input; 130 means interruption. None of these
is an EngineSmith correctness gate or performance score. A benchmark can exit 0 while reporting failed requests;
the caller must interpret the benchmark's own artifacts.

## Manifest version 1

There must be at least one `[[services]]` table and one `[workload]`. Unknown fields are rejected before any process
starts. Services launch in table order; each readiness command must succeed before the next service launches.
Place dependencies first and their consumers last. Cleanup uses the reverse order.

| Field | Meaning |
|---|---|
| `command` | Required nonempty argv array; no implicit shell, interpolation or expansion. |
| `cwd` | Working directory, relative to the manifest's directory. Defaults to `"."`. Must exist. |
| `env` | Explicit string environment overrides, recorded verbatim in the local result. Defaults to `{}`. |
| `inherit_env` | Names to copy from the caller, default `["PATH"]`. Missing names and overlap with `env` are errors. Their values are never serialized by the supervisor. |
| `timeout_s` | Finite positive seconds. For a service: readiness deadline (default 300). For the workload: execution deadline (default 3600). |
| `name` | Services only. Unique name starting with a letter; letters, digits, `_` and `-` follow. `workload` is reserved. |
| `ready` | Services only. Required argv array; exit 0 signals readiness. Uses the service's cwd and environment. |

Readiness probes are retried until the deadline; each gets at most five seconds of execution and is cleaned up
before retry. A failed probe is not evidence of readiness. Every started service is checked for exit while
waiting on probes or the workload, including between dependent launches. Successful readiness is a startup check,
not a continuing HTTP health check: a live process that later hangs is bounded by the workload timeout.
Cleanup can add time beyond the readiness/workload deadline because the existing process helper allows up to
five seconds after TERM and another five after KILL. This is not a strict wall-time budget.

Environment inheritance is intentionally explicit. List necessary names such as `HOME`, `CUDA_VISIBLE_DEVICES`
or `LD_LIBRARY_PATH` in `inherit_env`, or supply non-secret values in `env` when their exact values should be in
the report. Inherited values are not a complete reproducibility record; record non-secret device, build, model,
benchmark and dataset versions separately. The command arrays and explicit `env` are recorded; do not put
credentials there. Pass secrets via named inherited variables and review service/workload logs before sharing:
the supervisor cannot redact arbitrary child output.

## Artifacts and ownership

`result.json` includes the manifest SHA-256, normalized commands and working directories, explicit environment
values and inherited names, process IDs, readiness attempts and durations, workload exit/duration, cleanup
outcomes, and any failure. It is updated atomically at phase transitions and after cleanup. Each service and
its probes have separate logs. The workload owns any metrics or other files it writes in its `cwd`; this command
does not find, upload, parse or score them. Durations here are lifecycle diagnostics, not inference latency.

The supervisor runs in the foreground on the main thread. Every command, including probes and the workload,
gets a new local process group. Normal exit, failure, SIGINT and SIGTERM clean up only the groups created by
this invocation. It does not kill by service name, port or a PID read from an earlier run. A caught interrupt
returns 130 after cleanup and restores the caller's signal handlers. A hard kill of the supervisor, host failure,
or a child that daemonizes/escapes its process group is not covered. Services must remain in the foreground;
do not use shell backgrounding, detached containers, `setsid`, or remote launchers with this command.

This slice supports local process supervision on POSIX hosts. It does not implement SSH ownership, cluster
resource allocation, GPU isolation, continuous health probes, recovery after supervisor death, or benchmark
scoring. In particular, killing a local SSH process does not establish that remote workers stopped. Multi-node
ownership, the pinned AgentX adapter, speculation/quality contracts and search integration are follow-ups in
[#25](https://github.com/EngineSmith/enginesmith/issues/25). Existing concurrency work in #18 is independent.
