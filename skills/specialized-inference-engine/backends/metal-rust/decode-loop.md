# The GPU-resident decode loop

Purpose: how to encode and submit batch-1 decode so that the GPU never waits for the host: tokens stay on the GPU,
steps are encoded ahead in a few command buffers, and the host waits once per request. Also: the per-dispatch cost
that remains once the loop has this shape, how to stop early and rewind state, the streaming form, and why any
per-token host work costs so much. Read it before changing how decode steps are encoded, batched, stopped or streamed.

## The shape

```rust
// generate(prompt, count) -> count tokens. Host bookkeeping only; every value-dependent step is a kernel.
let ring = &scratch.gen_tokens;                        // U32 slots on the GPU (example LFM2.5: 256)
let first = if prompt.is_empty() {
    ring.view(0, &[1])?.write_bytes(&pending.to_le_bytes())?; 0   // host write before any commit
} else { prefill(prompt)?; first_token_from_last_row(ring.view(0, &[1])?)?; 1 };
let (mut j, mut batch, mut inflight) = (first, 1, Vec::new());    // the first command buffer holds one step
while j < count {
    let pass = ctx.begin()?;                           // one serial compute encoder
    for s in j..(j + batch).min(count) {
        if eof.load(SeqCst) { break; }                 // stdin closed: stop encoding (runtime.md)
        let input = ring.view(s.saturating_sub(1), &[1])?;          // step s reads token s-1 ...
        let output = ring.view(s, &[1])?;                           // ... and writes token s
        decode_step_into(&pass, &input, &output, s == first, s + 1 < count)?;
        state.pos += 1; state.advance_recurrent_slot();             // advanced at encode time
    }
    inflight.push(pass.commit()?);                     // commit, do not wait
    j = (j + batch).min(count); batch = 4;
}
for p in inflight { p.wait()?; }                       // once per request; then check status (runtime.md)
let tokens = ring.view(0, &[count])?.to_u32()?;
```

## Tokens stay on the GPU

- **The token ring.** Step `s` reads token `s-1` from a device buffer (its embedding gather) and writes token `s`
  (the greedy argmax or the sampler). The host writes only the pending token of a continuation, before the first
  commit, and reads the ring once, after the last wait. A ring larger than any request (example LFM2.5: 256 slots,
  `count <= 256` is checked) needs no wrap logic in the protocol path; the streaming path indexes it modulo its size.
- **The step's end writes the next step's start.** With the argmax in the head GEMV's epilogue, one small tail
  kernel writes the token, the next token's embedding row and the first layer's input norm (`kernels-fusion.md`).
  The first step of a call embeds its input with the ordinary gather and norm kernels; the last step of a call skips
  the next-row work, so a later continuation embeds the pending token itself.
- **Position and recurrent-state slots advance on the host at encode time.** They describe the state after every
  committed step has finished, which is only true after the wait; anything that rewinds must account for that.

## Command buffers: how many steps each

The GPU starts a call's decode only when the first command buffer is committed, and encoding a step on the host takes
~85-110 us. Measured on LFM2.5 (P=256 split request, 64-token decode, one instrumented process cycling configurations):

| first CB : later CBs (steps) | startup idle per call | decode speed vs 4:4 |
|---|---|---|
| 4 : 4 | 568-591 us (encode of the first CB 337-352 us) | 1.0000 |
| 1 : 4 | 340-350 us (encode 107-112 us) | 1.0014-1.0023 |
| 1 : 8, 1 : 16 | 348-350 us | 1.0006-1.0018 |
| 2 : 4 | 421 us | 1.0006 |

So: one step in the first command buffer, then a few steps per buffer (4, 8 and 16 measured the same). After the
first buffer the host is ~30 times faster than the GPU (~0.1 ms encode vs ~3.3 ms GPU per step), so later batch sizes
do not matter. The end-to-end gain was small (expected ~0.1%). Measured neutral on the same harness: an empty or
one-dispatch warm-up command buffer before the first step (1.0001-1.0015, one noisy run 0.98), spin-waiting on the
last buffer (wake-up 95 -> 65 us, 0.9997-1.0015), and command buffers with unretained references (0.9985-0.9998).

## Where a decode request's time goes once the loop has this shape

Example (LFM2.5, M5 Max, 64-token decode at P=256): the GPU is busy 99.3% of the request; no measurable idle between
command buffers (median -0.25 us); per call ~0.11 ms first-buffer encode, ~0.23-0.39 ms from commit to GPU start and
~0.09 ms wake-up after the last buffer, i.e. ~9 us per token; request parsing and the pipe ~5 us per token.
Everything else is GPU time: the kernels and their fixed per-dispatch cost.

## The dispatch budget

A batch-1 decode step is a chain of a few hundred small dependent dispatches, and each costs a fixed amount on top
of its work. Measured in the chain: 0.86 us for a one-threadgroup dispatch, 1.95 us for a 2048-threadgroup one.
Isolated empty dispatches grow with the grid: 6.08 us at 8192 and 22.5 us at 32768 threadgroups. A one-threadgroup
norm costs ~2.05 us in the chain, of which ~0.9 us is the dispatch boundary.

Count your dispatches per token by kind (the dispatch map: `references/optimization-method.md`) and multiply. Example
(LFM2.5): ~355 dispatches per token in the incumbent, ~262 after the fusions in `kernels-fusion.md`, ~252 after the
attention-prologue fusion; at 0.86-1.95 us each that is ~0.25-0.5 ms of a 3.28 ms step (derived). One layer after the
fusions:

| layer kind | dispatches |
|---|---|
| short-conv mixer + MoE | conv projections+conv, out_proj, router (norm + GEMV + top-k), expert gate/up+act, expert down+combine, add+norm = 6 |
| attention mixer + MoE | Q/K/V projections (V to cache), attention with Q/K norm+RoPE prologue, out_proj, then the same 4 = 7 |
| short-conv mixer + dense FFN | conv, out_proj, add+norm, gate/up+act, down, add+norm = 6 |
| head and tail | head GEMV with argmax epilogue, token tail = 2 |

Once the weight-streaming kernels are near the bandwidth ceiling, removing dispatches is the main lever left; the
next fusions usually need a grid-wide dependency, whose cost depends on the grid size (`kernels-fusion.md`).

## Why per-token host work kills throughput

- **A host wait per token** pays the start latency and the wake-up every token: ~0.23-0.39 ms + ~0.1 ms, i.e. 10-15%
  of a 3.3 ms step (derived from the measured latencies above; not measured as a loop).
- **A command buffer per dispatch** costs 13.6-14.4 us instead of 1.86 us per dispatch:
  ~250 dispatches would add ~3 ms per token, about doubling the step (derived).
- **Any host decision on a token value** (host argmax or sampling, host embedding lookup, a stop check that gates the
  next encode) forces such a wait. The host may decide only what is known before the request: the count, the
  positions, the chunking. Everything value-dependent runs in kernels.
- **Host encode cost is hidden only while it stays below GPU step time.** Per-dispatch allocations, locks and
  hash lookups are harmless at ~0.1 ms per step; if host encode ever approaches the GPU step, it becomes the step.

## Stopping early and rewinding state

A request with a known count never stops early: encode every step. When a token-dependent condition can stop
generation (end-of-sequence policy, stop strings, cancellation), the host learns it only after reading back a
completed command buffer, by which time more steps are committed. Then:

1. Keep at most `AHEAD` steps committed beyond the last token read (example LFM2.5: 16).
2. On a stop, wait for every in-flight command buffer. Committed work cannot be recalled; do not drop handles and
   assume it was cancelled.
3. Keep the tokens up to the stop; discard the rest.
4. Rewind: `pos -= extra`; move each recurrent-state ring back by `extra` slots; the pending token becomes the last
   kept token. KV rows beyond `pos` need no action: they are never read and are overwritten before they become live.

Recurrent state must be a ring for this to work: each step reads slot `s` and writes slot `s + 1 mod N`, so the last
`N - 1` states stay intact. `N > AHEAD` must hold at compile time.

```rust
pub const CONV_SLOTS: usize = 32;   // example LFM2.5: [2, hidden] bf16 per conv layer per slot
pub const STREAM_AHEAD: usize = 16;
const _: () = assert!(STREAM_AHEAD < CONV_SLOTS && STREAM_AHEAD < TOKEN_RING);
fn rewind(&mut self, extra: usize) { self.pos -= extra; self.slot = (self.slot + CONV_SLOTS - extra) % CONV_SLOTS; }
```

Check: stop after k = 0, 1, ..., AHEAD-1, AHEAD, AHEAD+1, the ring size - 1, the ring size and beyond, then continue,
and compare with a non-streamed generation of the kept tokens followed by the same continuation. Example (LFM2.5):
all such cases passed for greedy and sampled modes, plus prefix restore.

## Streaming variant

The same steps, with the host reading tokens while the GPU runs ahead:

```rust
loop {
    while next < limit && next < delivered + STREAM_AHEAD {           // keep the queue full
        let end = (next + batch).min(limit).min(delivered + STREAM_AHEAD); batch = 4;
        let pass = ctx.begin()?;
        for j in next..end { decode_step_into(&pass, ring_at(j - 1), ring_at(j), ..)?; }
        inflight.push_back((pass.commit()?, next, end)); next = end;
        if with_prompt && delivered == 0 { break; }                      // only one step before token 0
    }
    let (s, e) = if with_prompt && delivered == 0 { (0, 1) }             // token 0 came from prefill: no wait
        else { let Some((p, s, e)) = inflight.pop_front() else { break }; p.wait()?; (s, e) };
    let toks = read_ring(s..e);                                          // indices modulo the ring size
    delivered = e;
    if on_tokens(&toks) == Flow::Stop(n) { stop_and_rewind(s + n); break; }
}
```

The kernels are the protocol path's, so the tokens are bitwise identical to it. Its GPU-time cost against the
non-streaming loop was not measured separately; the worked engine's streaming HTTP endpoint, in rounds interleaved
with the protocol path, measured 155.7 vs 158.5 tokens/s (medians, -1.8%, includes the HTTP layer).
Serving concerns (stop strings, prefix reuse, chat templates) are in `serving.md`.

## Sampling inside the loop

Non-greedy decoding stays GPU-resident too: the head stores logits, a sampler kernel draws the token (for example
with Gumbel-max noise from a counter-based generator keyed by seed and position) into the same key the greedy argmax
uses, and the token tail continues as for greedy. Example (LFM2.5, `docs/ARCHITECTURE.md`): temperature alone was
within noise of greedy; temperature with top-k 50 and a repetition penalty cost +0.6% / +1.0% / +1.2% at 256 / 1K / 4K
context. Keep the greedy path's kernels unchanged when you add sampling.

## Checks for any change to the loop

- Tokens identical to the parent for one long request, a split request (prompt with count 1, then a continuation),
  and a chain of continuations; counts 1, 2, a batch boundary (5 with 1:4 batching), and the ring size.
- A continuation after a request that ended exactly on a batch boundary; a request after a reset.
- End of input during a request: encoding stops, committed buffers finish, the process exits 0.
