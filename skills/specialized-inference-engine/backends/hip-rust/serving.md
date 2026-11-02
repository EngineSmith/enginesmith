# Serving

Purpose: the protocol loop of a HIP + Rust engine (stdin JSON lines in, stdout JSON lines out), with the model resident
on the GPU between requests, the reset and continuation semantics, the rule that all GPU work is finished before a
reply, diagnostics on stderr only, and the run-time environment the harness gives the engine. Then, briefly, what an
interactive front end would add. Read it before writing the engine's main loop or changing how requests are parsed,
answered or validated. The protocol itself is the task contract's (docs of the framework); this file is about
implementing it well on this platform.

Nothing here is measured on this platform; the per-request costs are from `README.md`.

## The process

```rust
fn main() -> ExitCode {
    let args = parse_args();                         // ENGINE --model MODEL_DIR; nothing else is given
    let (tx, rx) = std::sync::mpsc::channel::<Option<String>>();
    let eof = Arc::new(AtomicBool::new(false));
    spawn_stdin_reader(tx, eof.clone());             // one thread: lines -> channel, None + eof flag at end
    let mut out = std::io::stdout().lock();          // replies only
    let mut engine: Option<Engine> = None;           // loaded at hello
    while let Ok(Some(line)) = rx.recv() {
        let reply = match handle(&mut engine, &line, &eof) {
            Ok(v) => v,                              // {"id", "ok": true, ...}
            Err(e) => { eprintln!("{e:#}"); error_reply(&line, &e) }   // {"id", "ok": false, "error"}
        };
        writeln!(out, "{reply}").and_then(|_| out.flush()).ok();
    }
    if let Some(e) = engine.as_mut() { e.synchronize().ok(); }        // finish queued work, then exit 0
    ExitCode::SUCCESS
}
```

- **`hello` loads everything**: device check, code objects and functions, weights uploaded and relaid out, KV cache
  and scratch allocated, a warm-up forward if it removes first-request costs (first launches of each kernel, page
  faults on pinned buffers). Loading time is recorded but not scored. If anything fails, reply with an error or exit
  non-zero with one stderr line; never reply ready and then run another numeric path.
- **stdout carries replies only.** One JSON object per line, flushed after each reply. Everything else (warnings,
  timings, debug dumps) goes to stderr, bounded in size; the harness keeps its tail. Make sure no library you call
  prints to stdout (the HIP runtime's logging goes to stderr when enabled).
- **Requests come one at a time**: the next arrives only after the reply, so there is no queueing or concurrency in
  the protocol path. The stdin thread exists to notice end of input, not for parallelism.
- **End of input** (stdin closed) during a request: stop launching new steps (the decode loop checks the flag between
  steps), synchronize the stream, exit 0. An engine that ignores it keeps the GPU and its memory after its parent is
  gone.

## generate: reset and continuation

- `reset: true`: discard all earlier state (KV length 0, position 0; no clearing of the cache memory is needed, since
  rows beyond the live length are never read), prefill `prompt` (raw token ids: no BOS, no template), generate exactly
  `max_tokens` tokens greedily. The first token comes from the prefill's last row.
- `reset: false`: `prompt` is empty; first consume the pending token (the last one emitted, which the device ring
  still holds, `decode-loop.md`), then generate `max_tokens` more.
- After any reply the state has consumed the prompt and every emitted token except the last one.
- Greedy: the highest logit, ties to the lowest token id. No stop token: always exactly `max_tokens` ids, each below
  the vocabulary size.
- **Validate before launching anything**: ids below `vocab_size`, `max_tokens >= 1`, a continuation only after a
  generate, and `position + prompt length + max_tokens` within the KV cache you allocated. Allocate the cache for the
  model's context (example Qwen3-0.6B: 40,960 positions x 114,688 B = 4.7 GB, estimate), so a valid request never
  fails for lack of room. A request that fails validation gets an error reply and leaves the state unchanged.
- **Finish all GPU work before replying**: the tokens are copied to a pinned host buffer and the stream synchronized
  once per request (`decode-loop.md`); this is also what makes the harness's timing honest.

## Parsing long requests

Prompts can be as long as the task's longest workload length (the protocol allows up to 524,288 tokens; the line is
then a few megabytes). Read lines with a growing buffer (`BufRead::read_line` into a reused `String`), parse with
`serde_json` into a struct with a `Vec<u32>` prompt, and copy the prompt into a pinned host buffer for the
upload. Parsing cost is counted in the timed request; at a few MB it is milliseconds at most (estimate), small next to
the prefill of that many tokens, but do not parse twice or build intermediate `Value` trees for the prompt.

## The run-time environment

- `PATH=/usr/bin:/bin:/usr/sbin:/sbin`, `HOME` and `TMPDIR` set to the scratch directory, which is also the working
  directory and has a long path; Hugging Face offline; no network. Do not use the network, and do not write outside
  the scratch directory (not even a cache under the real home).
- Nothing may depend on `hipcc` being found (code objects are embedded) or on `ROCM_PATH`, `LD_LIBRARY_PATH` and `HIP_*`
  variables. If the engine wants a runtime setting (for example kernel arguments in device memory), it sets it itself
  before the first HIP call, records that in its notes, and measures that it helps.
- The long scratch path: never build a Unix-socket path from it (the 108-byte limit), and if you compile with hiprtc
  at run time, check that it works with `TMPDIR` and `HOME` pointing there (`runtime.md`).
- The GPU is the first visible one (`HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES` select it before the engine
  starts); use device 0 and do not enumerate further.

## Beyond the protocol (optional)

The judged path is the protocol above. If you add interactive serving (streaming tokens, stop strings, sampling, a
tokenizer and chat template, an HTTP server), the Metal backend's `serving.md` describes mechanisms that carry over
unchanged in kind: a look-ahead window of queued steps with readback per completed batch and an exact rewind on stop;
host mirrors updated from kept tokens only; sampling on the GPU with Gumbel-max noise from a counter-based generator
keyed by seed and position, written into the same argmax key the greedy path uses; a pure-Rust tokenizer and template
layer checked against transformers on a corpus. The rule for all of them: they reuse the protocol path's kernels and
leave its bytes and speed unchanged with the feature off. None of this is measured on this platform.

## Checks

- A fresh process: `hello`, then the shapes `scripts/same.py` sends (one long request, a split request, a chain of
  continuations) give identical tokens.
- A request after a longer one equals the same request in a fresh process (stale scratch, stale counters).
- An invalid request (an id at `vocab_size`, a continuation first) gets an error reply and the next valid request
  still works.
- stdin closed mid-request: the process exits 0 within a bounded time.
- Run the engine under `env -i PATH=/usr/bin:/bin HOME=$PWD TMPDIR=$PWD` from a directory with a long path, to
  reproduce the harness's environment before submitting.
