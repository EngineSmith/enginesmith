# Serving

Purpose: what an engine needs beyond the judged protocol path to be used interactively: streamed output without GPU
stalls, stop conditions, prompt-prefix reuse across chat turns, sampling on the GPU, a pure-Rust text layer that
matches transformers, and an OpenAI-compatible server as an optional outer layer. Read it before adding any of these.
The rule for all of them: they reuse the protocol path's kernels and leave its bytes unchanged; the greedy path with
sampling off must stay the same tokens at the same speed.

Numbers are measurements from the LFM2.5 worked engine on an Apple M5 Max unless they are marked as estimates.

## Streaming without GPU stalls

The loop is the one in `decode-loop.md` (streaming variant, stopping and rewinding): keep N steps committed beyond
the last token read back (Example: 16), read back each command buffer as it completes while the GPU runs the next
ones, encode only one step before the first token, check stop conditions per completed batch, and on a stop wait for
the in-flight buffers, keep the tokens up to the stop and rewind position and recurrent-state rings exactly. What
serving adds:

- **Host mirrors follow the kept tokens.** Anything the host keeps per sequence (the seen-token set of the repetition
  penalty, the pending token, the decoded text) is updated from the kept tokens only, after the rewind, so discarded
  steps leave no trace. Upload GPU copies at the start of the next call, when the GPU is idle.
- **Cancellation** (client disconnect, Ctrl-C) is a stop flag checked between encodes; it ends like any other stop.
- **Check** stop-then-continue against an uninterrupted generation for many stop points (0, 1, the batch size, the
  look-ahead, the token ring size and beyond), greedy and sampled. Example (measured): 84 such cases plus 18 prefix
  restores per run, 0 differences.
- Example (measured on a shared GPU): decode through the HTTP server's streaming path 155.7 tokens/s vs 158.5 on the
  protocol path.

## Stop conditions

- **End-of-sequence ids** from `generation_config.json` (it may list several), falling back to the tokenizer's;
  `stop_token_ids` given by the request; `max_tokens` and the context limit.
- **Stop strings** are text, not tokens: detokenize incrementally, hold back any text that could still be the prefix of
  a stop string, and end the output before the first match (the match is not returned). A token can end inside a
  multi-byte character: hold back incomplete UTF-8 until the next token completes it.
- Report the reason (`stop` for EOS, stop token or string; `length` for the count or the context) and count a final
  EOS or stop token in the completion tokens.

## Prompt-prefix reuse

A chat turn's prompt usually repeats the previous prompt. Keep, per sequence:

- a **snapshot** at a split point of the last prompt: the position, the recurrent state (convolution or SSM state),
  and the seen-token set; Example (LFM2.5): ~0.25 MB;
- the **KV rows** below the split point, left in place in the cache (they are never cleared, `attention.md`).

When a new prompt's token ids start with the snapshot's ids, restore the snapshot and prefill only the rest; any other
prompt starts from scratch and replaces the snapshot. To take the snapshot, cut the prefill at the split point (one
extra submission) and capture the state there; recurrent state at an arbitrary position is not recoverable later.

- **Where to split.** Where the next turn's rendering diverges from this one. Example (LFM2.5 chat template): the last
  `<|im_start|>`, because the template re-renders the previous assistant message (dropping its reasoning) in the next
  prompt; without one, the prompt's last token.
- **Numerics.** A cut changes the chunking, so greedy output can differ at near-ties from an uncut prefill. A cut at the
  prompt's last token coincides with the reference partition when the reference runs the last token as an L=1 step,
  so it changes nothing. Keep a switch that disables reuse and reproduces the protocol path exactly.
- Example (measured on a shared GPU, absolute times about twice idle; `docs/SERVER.md`): a six-turn chat with a
  3.6K-token system prompt: time to first token 103-108 ms with reuse vs 534-569 ms without, from turn 2 on.

## Sampling on the GPU

Sampling stays inside the GPU-resident loop: the head stores the logits (the greedy head's argmax epilogue is not
used), a sampler kernel draws the token into the same 64-bit key the greedy argmax uses, and the greedy token tail
continues from it (`kernels-fusion.md`). Semantics, matching Hugging Face and vLLM:

- **Processed logits** `w_i = pen(v_i) / T`, with `v_i` the logit as greedy compares it (BF16 widened to F32) and
  `pen(v) = v / p` for `v > 0`, `v * p` otherwise, applied to tokens in the sequence's seen set (prompt and generated
  tokens).
- **Kept set** `{w >= tau}`: top-k keeps every token at least the k-th largest (ties at the boundary kept); top-p then
  keeps, among those, the smallest set with probability at least `top_p` (a tie group at the boundary kept whole).
  Consequence: `top_k = 1` is not greedy when the top logits tie exactly. Example (measured): exact BF16 ties at the
  top occurred in 4 checked cases.
- **The draw** is the Gumbel-max trick: argmax over the kept set of `w_i + G_i`, an exact sample of the softmax over
  the kept set. `G_i` comes from a counter-based generator (Philox4x32-10) keyed by the seed with the counter (token
  index / 4, absolute position, a domain constant, 0), so the draw depends only on (seed, position, logits, kept set):
  not on how the vocabulary is split across threadgroups, nor on how a generation is split into requests.

```metal
// G = -log(E), E = -log(1 - V) ~ Exp(1), V = (x + 0.5) / 2^32; small E via the series of -log1p(-V)
static inline float gumbel(uint x) {
    const float v = (float(x) + 0.5f) * 2.3283064365386963e-10f;
    const float e = v < 0.015625f ? v * (1.0f + v * (0.5f + v * (0.33333334f + v * 0.25f)))
                                  : -log(max(1.0f - v, 2.9802322387695312e-08f));
    return -log(e);
}
// The greedy argmax's key: order-preserving float bits high, ~index low, so atomic_max picks
// "value desc, index asc" (-0 folded to +0; NaN never a candidate)
static inline ulong argmax_key(float v, uint i) {
    uint u = as_type<uint>(v);
    if ((u << 1) == 0u) { u = 0u; }
    const uint o = (u & 0x80000000u) != 0u ? ~u : (u | 0x80000000u);
    return (ulong(o) << 32) | ulong(~i);
}
```

- **Temperature only:** one wide kernel computes `w`, adds `G` and folds the best key per threadgroup.
- **Top-k / top-p: exact threshold search, no full sort.** A prep pass stores `w` and its max and min; a levels pass
  counts tokens and sums `exp(w - max)` above 32 cutoffs with integer counts and fixed-point masses and picks the band
  holding the threshold; up to two more rounds split a band of more than 256 tokens 32-fold; a gather copies the band
  and draws among the tokens above it; one threadgroup sorts the band (radix rounds first above 2048 tokens), fixes
  `tau` exactly and draws among the kept band tokens. Every sum in the decision is integer or in a fixed order, so the
  result is bitwise deterministic.
- **Repetition penalty:** a seen-token bitmap (vocabulary / 32 words) on the GPU; the host copy is authoritative and is
  uploaded at the start of each call; the token tail sets the drawn token's bit.
- **Cost** (measured, decode vs greedy at 256 / 1K / 4K context, paired rounds):
  temperature 0.6 alone -0.6% / +0.1% / 0.0% (noise); temperature 0.6 with top-k 50 and penalty 1.05 +0.6% / +1.0% /
  +1.2%. The greedy path kept its kernels, tokens and speed.
- **Checks:** a sampled draw replays exactly with the same seed, in one process and across processes; split requests
  and checkpoint/restore give the same tokens; the empirical distribution of 1000 draws at a fixed position matches
  the exact kept-set softmax (chi-square and total-variation distance against their null distribution); degenerate
  settings (`T -> 0`, `top_p -> 0`) equal greedy except at exact ties.

## Text layer in pure Rust

- **Tokenizer:** the checkpoint's `tokenizer.json` through the `tokenizers` crate with default features off and the
  pure-Rust `fancy-regex` backend (no Oniguruma, no esaxx C++, no HTTP client). Pin the exact version.
- **Chat template:** the checkpoint's `chat_template.jinja` through `minijinja` plus `minijinja-contrib`'s `pycompat`
  (Python string and dict methods). Reproduce transformers' environment: `trim_blocks`, `lstrip_blocks`, loop
  controls, its `{% generation %}` pass-through tag, `tojson` as `json.dumps(x, ensure_ascii=False)`,
  `raise_exception`, the special-token variables, `add_generation_prompt`, tools, documents and extra kwargs.
- **Defaults** from `generation_config.json` (temperature, top-k, top-p, repetition penalty, EOS ids); `do_sample:
  false` means greedy.
- **Parity, measured against transformers on a corpus, not on a few strings.** Example (measured; tokenizers 0.23.2,
  transformers 5.19.0, jinja2 3.1.6): 41,885 encodes, 86,848 decodes, 86,848 incremental (streamed) decodes and 46
  template renderings (tools, reasoning, multi-turn), 0 differences. Include ids that decode to partial UTF-8 (1,561 in
  that vocabulary).

## OpenAI-compatible server (optional)

An outer layer over the same session object the CLI uses; the engine core does not depend on it.

- `std::net` HTTP/1.1 with keep-alive and chunked bodies is enough; server-sent events for streaming; endpoints
  `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`.
- A batch-1 engine serves one request at a time in arrival order; accept connections concurrently and queue them.
- Reject what you do not implement (log-probabilities, logit bias, `n > 1`) with a 400, rather than ignoring it.
- Model-specific output parsing (reasoning blocks, tool-call syntax) belongs to the model's serving notes, not to the
  engine. Example (LFM2.5): text before `</think>` becomes `reasoning_content`; Pythonic tool-call lists between
  `<|tool_call_start|>` and `<|tool_call_end|>` become OpenAI `tool_calls`.
- Check end to end: the server binary's protocol path equals the engine it was built from (Example: 40 of 40 prompts
  token-identical); over HTTP, greedy and seeded sampled outputs equal the protocol path's (300 tokens per case); the
  decode rate is within noise of the protocol path's.
