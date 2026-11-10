# Engine protocol

Version 1 serves single-request tasks (`[workload] mode = "single"`, the default). Version 2, at the end, serves
concurrent tasks (`mode = "concurrent"`).

## Version 1

An engine is one executable. The harness starts it as

```
ENGINE --model MODEL_DIR
```

with its working directory set to a fresh, empty scratch directory and a small environment: `PATH=/usr/bin:/bin:/usr/sbin:/sbin`,
`HOME` and `TMPDIR` set to the scratch directory, Hugging Face offline. `MODEL_DIR` holds the model exactly as
downloaded (`config.json`, the safetensors shards, tokenizer files). The engine must not write anywhere except its
scratch directory and must not use the network.

The engine reads requests from stdin and writes replies to stdout, one JSON object per line (UTF-8, ending in
`\n`). Every request has an integer `id` and an `op`; the reply repeats the `id` and has `"ok": true`. Stdout
carries replies only; write diagnostics to stderr (the harness keeps the tail of it). Requests come one at a time:
the next request is sent only after the previous reply has arrived.

An error reply is `{"id": 7, "ok": false, "error": "text"}`. During an evaluation it fails the candidate.

### hello

```json
{"id": 1, "op": "hello"}
{"id": 1, "ok": true, "protocol": 1}
```

Load the model and do any one-time preparation (pipelines, buffers) before replying. Loading time is recorded but
not scored.

### generate

```json
{"id": 2, "op": "generate", "prompt": [9707, 11, 1879], "max_tokens": 1, "reset": true}
{"id": 2, "ok": true, "tokens": [21806]}
{"id": 3, "op": "generate", "prompt": [], "max_tokens": 64, "reset": false}
{"id": 3, "ok": true, "tokens": [13, 576, 1191, "... 64 token ids in all"]}
```

- `reset: true` discards all earlier state, consumes `prompt` (raw token ids: no BOS is added, no chat template, no
  tokenizer) and generates exactly `max_tokens` tokens greedily.
- `reset: false` continues: `prompt` is empty, and the engine first consumes the last token it emitted, then
  generates `max_tokens` more.
- Greedy means the token with the highest logit; on an exact tie, the lowest token id. There is no stop token:
  always return exactly `max_tokens` ids, each below the vocabulary size. After a reply, the state has consumed
  the prompt and every emitted token except the last one.
- Finish all GPU work before replying.
- Under the task's default `[engine] one_forward_per_token`, decode computes one forward pass per generated token:
  no draft models, and no drafted, speculated or looked-up tokens verified in batches.

Prompts can be as long as the task's longest workload length (up to 524,288 tokens; the JSON line is then a few
megabytes). Token ids are only guaranteed to be below the model's `vocab_size`; they are not natural text.

### Timing

The harness times each request from just before it writes the request until the whole reply line has arrived. It
ignores any timing the engine reports. A formal evaluation times, per case, a prefill request
`generate(P, 1, reset)` and a decode request `generate([], n - 1, continue)`. Every timed request uses a prompt drawn
fresh for that request, so caching results between requests or processes cannot help.

### Correctness

A candidate's tokens are compared with the reference's for the same prompt. Identical streams agree. A stream that
differs is replayed through the reference with the candidate's tokens forced (teacher forcing); a step whose
token is not the reference's top choice is a mismatch. The task's gates cap mismatches per stream and in total.
The harness also runs short control prompts (1 to 4,097 tokens, at the lengths where tiled and chunked kernels
usually break) before any timing.

### Requests only the reference answers

The built-in reference (`python -m enginesmith.references.mlx_lm`) also answers `judge` (`prompt`, `tokens` → `mismatches`:
`[step, margin, reference token]` for each mismatching step) and `stats` (peak and active memory). A candidate
engine does not need them.

## Version 2: concurrent requests

A concurrent task's engine is started the same way, in the same environment, and answers `hello` with
`"protocol": 2`. After that the harness may send `submit` requests at any time, also while earlier ones are still
being computed: the engine must keep reading stdin while it works (for example with a reader thread). Each request is
independent: it starts from an empty state, consumes its prompt and generates exactly `max_tokens` tokens greedily
(the same greedy rule and decode rule as version 1). How the engine schedules the requests in flight (one at a time,
batched, prefill and decode in the same step) is its own choice.

```json
{"id": 1, "op": "hello"}
{"id": 1, "ok": true, "protocol": 2}
{"id": "c4-0000", "op": "submit", "prompt": [9707, 11, 1879], "max_tokens": 64}
{"id": "c4-0001", "op": "submit", "prompt": [17, 4, 220, 9], "max_tokens": 128}
{"tokens": [["c4-0000", 21806]]}
{"tokens": [["c4-0000", 13], ["c4-0001", 576]]}
```

- A `submit` gets no reply line. The engine streams tokens as lines `{"tokens": [[id, token], ...]}`: any number of
  pairs, from any requests in flight, each request's tokens in order. A request is complete when it has
  `max_tokens` tokens; a token for a request that is not in flight is a protocol failure.
- Send each token as soon as it is computed: the harness takes a token's time from the line that carries it, and
  holding tokens back only delays them.
- An error for one request is `{"id": "c4-0001", "ok": false, "error": "text"}`; it fails the candidate.
- Requests carry string ids. The prompts are raw token ids, as in version 1.

### Timing (version 2)

The harness runs closed-loop clients: per concurrency level C, C clients each send their requests one after another,
the next as soon as the last has all its tokens. It timestamps each request just before it writes it and each token
when its line arrives, and ignores any timing the engine reports. Per level and round it records the throughput (first
request to last token), the time to the first token of every request and its normalized latency (end-to-end time /
output tokens); see docs/design.md for the score. The correctness checks are those of version 1: every request's
stream against the single-request reference.

### The reference in version 2

`python -m enginesmith.references.mlx_lm --concurrent` answers `hello`, `submit` and `stats` in version 2; its tokens
come from mlx-lm's own continuous-batching generator (`BatchGenerator`, the settings of `mlx_lm.server`). Judging uses
the reference without `--concurrent`. A custom reference command (`[reference] command`) of a concurrent task must
accept `--concurrent` for its timed rounds.
