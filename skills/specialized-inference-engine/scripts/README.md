# Scripts: bitwise A/B, interleaved timing, GPU lock, bytes per token

Generic tools for any engine binary that speaks the framework's JSON-lines engine protocol (version 1: `hello`,
then `generate` with token-id prompts, `max_tokens` and `reset`). Python 3 standard library only, no host paths. They are the tools the
worked example (`examples/lfm2.5/`) was optimized with, with the paths turned into flags.

**Use them only when the task contract allows running the engine directly. The framework's own development
evaluation (`enginesmith dev-eval`) is authoritative**: these tools tell you whether a change is worth submitting, not whether it passes.

| tool | what it does | takes the GPU lock |
|---|---|---|
| `engine_client.py` | the driver (library and a small CLI): starts an engine, sends requests, times them; nonce prompts; the lock | yes (CLI) |
| `same.py` | bitwise A/B: two engines, ~40 prompts x 3 request shapes, every token compared | yes |
| `ab.py` | interleaved ABBA timing of two engines, prefill and decode ratios per length, token check | yes |
| `gpu_lock.py` | runs any other GPU command (a micro-benchmark, a profiler run) under the lock | yes |
| `bytes_per_token.py` | bytes read per decode token and the decode floor, from safetensors headers and a small spec | no (reads headers only) |

## Launch configuration

Every engine is started as `ENGINE --model MODEL_DIR`, with a clean environment (`HOME` and `TMPDIR` set to a fresh
scratch directory, `PATH=/usr/bin:/bin:/usr/sbin:/sbin`, Hugging Face offline), its working directory in that scratch
directory and its stderr in `<scratch>/stderr.log`. The model directory comes from, in order of precedence:

1. `--model DIR`;
2. `--launch FILE`, a launch file `{"model": "<dir>"}` (default `$ENGINE_LAUNCH`);
3. the EngineSmith session (`$ENGINESMITH_SESSION`, which every session sets), so inside a session no flag is needed.

The prompt vocabulary is `--vocab`, else the session's, else the model's `config.json` `vocab_size`. `--scratch-dir`
chooses where the per-engine scratch directories go (default: the temp directory).

## The GPU lock, and the deadlock to avoid

Only one process may use the GPU while you measure. The lock is an `flock` on one file: `--lock-file`, else
`$GPU_LOCK_FILE`, else `/tmp/engine-gpu.lock`. The default ignores `$TMPDIR` on purpose, so that processes with
different temp directories still share one lock; every process (and every sub-agent) must use the same path.

**A tool that takes the lock itself (`engine_client.py`, `same.py`, `ab.py`) must not be run under `gpu_lock.py`:
the child waits for the lock its parent holds, and both hang.** `gpu_lock.py` sets `$GPU_LOCK_HELD` for its child,
and these three tools stop with an error when they see it; any other program that takes an `flock` on the same file
still deadlocks. Run everything else that touches the GPU (your own engine run, a Metal micro-benchmark) through
`gpu_lock.py`. The tools have no timeout of their own; bound them yourself if your task requires it.

## same.py: bitwise A/B

```
same.py PARENT_ENGINE CANDIDATE_ENGINE --out logs/same-001.json [--long] \
        [--extra-prompts cases.jsonl [--max-extra-tokens 8192]]
```

For each prompt and each engine: `generate(P, 65)`; the split request `generate(P, 1)` + `generate((), 64)`; and a
chain `generate(P, 17)` + 3 x `generate((), 16)` (every request with a prompt starts from a fresh state). Prompts:
nonce prompts at 34 lengths from 1 to 4097 around every power-of-two boundary (`--long` adds 8191, 8192, 8193,
16384, 32768) and six random-token prompts with a fixed seed; `--extra-prompts` adds your own cases, one
`{"id": ..., "prompt": [ids]}` per line. A case is bad when the engines differ in any token, or when one engine's three
shapes disagree on their first 65 tokens. It prints `{"cases", "identical", "bad", ...}` and exits 1 when `bad` is
not empty; the full token streams go to `--out` (which must not exist). An optimization is only a candidate when
`bad` is `[]`. Extend the prompt set with the lengths where your model's chunk or tile boundaries fall.

## ab.py: interleaved timing

```
ab.py PARENT_ENGINE CANDIDATE_ENGINE --out logs/ab-001.json [--lengths 256,1024,4096] [--rounds 6] \
      [--warmup 2] [--decode 64]
```

Both engines stay loaded. Each round runs the split request (prefill = `generate(P, 1)`, decode =
`generate((), DECODE)`) on a fresh nonce prompt, the same prompt for both arms, A first on even
rounds and B first on odd rounds. It reports each arm's median and the median per-round ratio A/B (B's speedup) for
prefill and decode, the decode tokens per second, and any request whose tokens differ. Run the parent against itself
first: that ratio's spread is your noise floor (about +-0.3% for decode on a quiet machine in the worked example).
For changes near the floor, add rounds (24-40) rather than trusting one run; on a thermally throttling laptop, never
compare absolute times between runs.

## engine_client.py: the driver

As a command, for a smoke test (takes the lock):

```
engine_client.py ENGINE [--generate P,COUNT] [--split P,DECODE] [--offset N]
```

As a library: `Engine(path, launch, label, scratch_root)` with `request`, `generate(prompt, count)` ->
`(tokens, ms)`, `split(prompt, decode)` -> `(tokens, prefill_ms, decode_ms)`, `close()`; `load_launch(args)` after
`add_launch_args(parser)`; `gpu_lock(path)`; `nonce_prompt(length, offset, vocab)`, the deterministic prompt
`((i * 2654435761) mod 2**32) mod vocab` for `i` in `offset .. offset+length-1`.

## gpu_lock.py

```
gpu_lock.py [--lock-file PATH] -- CMD ARGS...
```

Waits for the lock, runs the command, releases the lock and exits with the command's status.

## bytes_per_token.py: the decode floor

```
bytes_per_token.py MODEL_DIR SPEC.json --bandwidth-gbs 590 --contexts 0,256,4096 [--json] [--unmatched error]
bytes_per_token.py --example
```

Reads each shard's header (`model.safetensors.index.json`, else every `*.safetensors`; never the tensor data),
assigns each tensor to the first spec group whose `fnmatch` pattern matches its name, multiplies the group's stored
bytes by its `fraction` (default 1; `top_k / n_experts` for stacked routed experts; 0 for tensors decode never
reads), adds `state_bytes` and `per_position_bytes` x context, and divides by the bandwidth you give. Use the
streaming read bandwidth you measured on the machine, not the vendor's peak. `--unmatched error` makes sure no
tensor was missed. The spec of the worked example is in `examples/lfm2.5/model-card.md` (section 6); it reproduces
that model's 1,464,000,512 weight bytes per token.
