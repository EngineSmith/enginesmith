# Profiling on ROCm

Purpose: how to find where GPU time goes inside the real engine on an MI325X: the ground truth to check everything
against, per-kernel times from rocprofv3 kernel traces, the in-chain marginal cost of a kernel, hardware counters
with rocprof-compute, hipEvent timing, how to attribute a decode step's time per kernel, why micro-benchmarks
mislead, the GPU lock, and timing noise. This is the platform half of phase 3 in `references/workflow.md`; timing
A/B builds is `references/measurement.md`.

Nothing here has been measured in an engine yet. Tool behaviour is from the ROCm 7.x tools on the measured host;
check flags with `rocprofv3 --help` on yours.

## What to measure, in this order

1. **Ground truth:** the request times the harness sees (prefill request, decode request) and, inside the engine,
   the GPU time of one request between two `hipEventRecord`s on the forward stream. Nothing inserted, nothing
   serialized. Every finer method must add up to this.
2. **Per-kernel times in the real chain:** real weights, real cache, real order of dependent kernels. A kernel's cost
   depends on what ran before it (caches, the previous kernel's drain).
3. **The check:** per-kernel times summed over one decode step must match the ground-truth step time. Until they do,
   the table is not evidence. On this platform the gap between kernels may be a large share of the step
   (`README.md`: ~1.5-1.9 us per launch against 1-3 us of work per layer GEMV), so a kernel trace that sums only
   kernel durations will undercount; the difference is the inter-kernel cost you are trying to remove.

```rust
// ground truth inside the engine, behind an environment switch read once at startup
hip::event_record(ev_start, stream)?;
run_decode_request(...)?;                 // the real path
hip::event_record(ev_stop, stream)?;
hip::event_synchronize(ev_stop)?;
let gpu_ms = hip::event_elapsed_ms(ev_start, ev_stop)?;  // write to stderr after the reply, never to stdout
```

## rocprofv3 kernel traces

Run the engine directly, with a file of protocol requests on stdin, under the GPU lock:

```
scripts/gpu_lock.py -- rocprofv3 --kernel-trace --stats --output-format csv -d prof/run1 -- \
    ./target/release/engine --model MODEL_DIR < requests.jsonl > replies.jsonl
```

- Do not profile through `scripts/engine_client.py`, `scripts/ab.py` or `scripts/same.py`: they start the engine
  with a clean environment, which drops the variables rocprofv3 uses to inject itself into the child process, and
  they take the lock themselves (`scripts/README.md`). Write the request file with the same nonce prompts the tools
  use (`engine_client.nonce_prompt`): `hello`, a reset `generate(P, 1)`, then a continuation `generate([], 64)`.
- The kernel trace CSV has one row per launch with GPU begin and end timestamps (ns), the kernel name, grid and
  block sizes; the stats CSV sums per kernel name. Extract one decode step: the kernels between two consecutive
  argmax kernels of the continuation.
- From the trace compute per step: kernel count (check against your own count), sum of kernel durations, and the
  span from the first kernel's begin to the last kernel's end. Span minus sum is the time the GPU spent between
  kernels.
- `--hip-runtime-trace` (API calls, host side) shows whether the host keeps ahead of the GPU; `--memory-copy-trace`
  shows copies. `--kernel-include-regex` limits the trace.
- **Measure the perturbation:** the decode step time with the profiler attached against the ground truth without it.
  If the profiler adds per-kernel cost, use its traces for structure and ranking, not for absolute costs.
- **No ROCTx markers in the judged binary:** `libroctx64` is not in the native check's allow-list. Use kernel names
  and event ranges instead, or keep markers in a separate profiling build.

## In-chain marginal cost (the number to trust)

Profilers report kernel durations; what fusion removes is the duration plus the boundary. Measure the boundary
directly, as the Metal backend did (`backends/metal-rust/profiling.md`):

- **empty probe:** a build that launches an empty kernel of a given grid after every real kernel of the step. (Its
  extra step time) / (extra kernels) is the in-chain cost of one kernel boundary at that grid. This is the number
  that decides how much fusion buys on this machine; it is not measured yet.
- **duplicate probe:** an extra copy of one kernel class reading another layer's weights into a dummy output. (Its
  extra step time) / (extra kernels) is that class's real in-chain cost, including its boundary.
- Time probe builds interleaved with the base build (`scripts/ab.py`, 6-10 rounds); a class with tens of kernels per
  step resolves to a fraction of a microsecond, a class of one or two kernels does not.

## Attributing a decode step

Build the dispatch map (`references/optimization-method.md`): every kernel in order, grid x block, bytes read,
dependency. Then put on it:

| column | source |
|---|---|
| kernel duration | rocprofv3 kernel trace (median over steps) |
| boundary cost | empty probe |
| floor | bytes / measured bandwidth + boundary cost |
| excess | duration + boundary - floor |

Multiply the excess by kernels per token and sort: that is the ranked opportunity list. Example (Qwen3-0.6B, an
estimate of what the map will show, not a measurement): a naive engine with ~10 kernels per layer runs ~285 kernels
per token; at ~1.7 us each that is ~0.48 ms of boundary cost against a ~0.27 ms streaming floor, so the first
opportunities are kernel count, then bytes per kernel.

## Hardware counters

- **rocprofv3 `--pmc`** collects named counters per kernel (list them with `rocprofv3-avail`): fetched bytes from HBM,
  L2 hit and miss counts, busy cycles. Use them to tell a kernel that streams from HBM from one that hits a cache, and
  to compute achieved bandwidth from bytes actually fetched.
- **rocprof-compute** (formerly Omniperf; if installed on your machine) collects many counter passes by replaying the
  workload and produces roofline, occupancy, LDS bank conflict and cache reports per kernel. Every pass reruns the
  program, so use it on a micro-benchmark harness or a short, deterministic request file, under the lock.
- Counter collection serializes and replays kernels: never take times from a counter run.

## Why micro-benchmarks mislead here

- **Caches hold a micro-benchmark's working set.** 4 MB of L2 per XCD and a 256 MB Infinity Cache: a GEMV that
  re-reads one 4-12 MB matrix runs from cache. The streaming benchmark itself shows the effect: 4796 GB/s at 256 MB
  against 4193-4376 GB/s at 1-4 GB (`README.md`). Cycle through more than 512 MB of distinct weights (or use the real
  1.19 GB weight set), or run a large streaming kernel in front to evict.
- **Launch-bound timing:** a micro-benchmark that launches one tiny kernel at a time measures the launch path, not the
  kernel. Launch many back to back and divide, or read GPU timestamps from the trace.
- **Isolated is a hypothesis.** Accept a change only on an interleaved end-to-end A/B through the real protocol
  (`references/measurement.md`).

## A micro-benchmark harness that does not lie

1. A small Rust binary that links the engine's kernel code object and launch wrappers (same hipcc flags).
2. Synthetic random weights at the production shapes, cycled over more distinct copies than the caches hold.
3. Many launches per measurement; GPU time from events around the batch or from a kernel trace; base and variant
   interleaved; medians over trials.
4. A bitwise check of the variant against the base over millions of outputs, with fresh inputs per round, and a
   positive control (a variant known to change rounding must be reported as different).
5. Bound probes (no math, no dependent loads) to see how much a mechanism could at most buy.
6. Run it under `scripts/gpu_lock.py`, like everything else that touches the GPU.

## The GPU lock and the machine

- Every GPU process takes the shared lock (`scripts/gpu_lock.py`, `references/measurement.md`); the tools that take it
  themselves must not run under it.
- The MI325X sits in a multi-GPU server: other jobs may run on the other GPUs and share the host's CPUs, memory and
  power. Check that your GPU is otherwise idle before timing (`rocm-smi --showpids` or `amd-smi process`; the
  harness also warns when more than 2 GiB of VRAM is already in use) and record what else ran.
- **Noise is not measured yet on this platform.** Run the parent against itself first (`scripts/ab.py PARENT
  PARENT`); its per-round spread is your acceptance threshold. Short requests (a 64-token decode of a small model is
  ~20-120 ms) have higher relative noise: add rounds or decode tokens rather than trusting one run.
- Record GPU clocks and temperature at the start and end of a long run (`rocm-smi --showclocks --showtemp`), and
  compare only paired ratios.

## Instrumentation switches: off by default, proven neutral

- Gate every instrument by an environment variable read once at startup; with it unset the binary runs the parent's
  code path. Write records to stderr or a scratch file after the reply, never inside the timed request.
- Prove neutrality: the instrumented binary with the switch off against the parent, timed (ratio 1.000 within noise)
  and token-identical (`scripts/same.py`).
