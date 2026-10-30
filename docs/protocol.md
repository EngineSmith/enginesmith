# Engine protocol, version 1

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

## hello

```json
{"id": 1, "op": "hello"}
{"id": 1, "ok": true, "protocol": 1}
```

Load the model and do any one-time preparation (pipelines, buffers) before replying. Loading time is recorded but
not scored.

## generate

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

Prompts can be as long as the task's longest workload length (up to 524,288 tokens; the JSON line is then a few
megabytes). Token ids are only guaranteed to be below the model's `vocab_size`; they are not natural text.

## Timing

The harness times each request from just before it writes the request until the whole reply line has arrived. It
ignores any timing the engine reports. A formal evaluation times, per case, a prefill request
`generate(P, 1, reset)` and a decode request `generate([], n - 1, continue)`. Every timed request uses a prompt drawn
fresh for that request, so caching results between requests or processes cannot help.

## Correctness

A candidate's tokens are compared with the reference's for the same prompt. Identical streams agree. A stream that
differs is replayed through the reference with the candidate's tokens forced (teacher forcing); a step whose
token is not the reference's top choice is a mismatch. The task's gates cap mismatches per stream and in total.
The harness also runs short control prompts (1 to 4,097 tokens, at the lengths where tiled and chunked kernels
usually break) before any timing.

## Requests only the reference answers

The built-in reference (`python -m enginesmith.references.mlx_lm`) also answers `judge` (`prompt`, `tokens` → `mismatches`:
`[step, margin, reference token]` for each mismatching step) and `stats` (peak and active memory). A candidate
engine does not need them.
