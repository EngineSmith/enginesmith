# The GPU-resident decode loop

Purpose: how to run batch-1 decode on an MI325X so that the GPU never waits for the host and pays as few kernel
boundaries as possible: count kernels per token and price them, keep tokens on the GPU with a device-side argmax,
synchronize once per request, decide between stream launches, hipGraph replay and persistent kernels by
measurement, and handle continuations. Read it before changing how decode steps are launched, synchronized or
counted. The fusions that reduce the kernel count are in `kernels-fusion.md`.

Measured inputs are from `README.md` (one MI325X, ROCm 7.14/HIP 7.15 hipcc, 2026-11-02): 1.53 us per empty launch,
1.87 us per launch of a 304-block tiny kernel, 9.31 us for one launch + `hipStreamSynchronize`, 1.69-1.72 us per
kernel under hipGraph replay. Everything else here is an estimate or a hypothesis.

## The shape

```rust
// generate(prompt, count) -> count tokens. The host launches; every value-dependent step is a kernel.
if reset { state.clear(); prefill(&prompt)?; }               // writes the first token into ring[0] on the GPU
else     { /* the pending token is ring[last] from the previous request: already on the GPU */ }
let first = if reset { 1 } else { 0 };
for s in first..count {
    if eof.load(SeqCst) { break; }                           // stdin closed: stop launching (serving.md)
    let input = if s == 0 { state.pending_slot } else { ring_slot(s - 1) };
    decode_step(stream, input, ring_slot(s), state.pos)?;    // N launches, no waits
    state.pos += 1;
}
hip::memcpy_async_d2h(host_tokens, ring_slots(0..count), stream)?;  // pinned host buffer
hip::stream_synchronize(stream)?;                            // the only wait of the request
reply(host_tokens);
```

- **One synchronization per request.** The protocol requires all GPU work to be finished before the reply; the
  device-to-host copy of the tokens followed by `hipStreamSynchronize` does both. A host wait per token costs at
  least the measured 9.31 us round trip instead of ~1.5-1.9 us per launch, i.e. ~7.5 us more per token (derived),
  and it empties the GPU's queue every step, so the next step's first kernels wait for the host.
- **Tokens stay on the GPU.** Step `s` reads token `s - 1` from a device ring (its embedding gather) and writes token
  `s` (the argmax). The host never reads a token before the end of the request and never writes one: a
  continuation's pending token is the last slot the previous request wrote, still in device memory. Size the ring
  for the largest `max_tokens` the task sends, or index it modulo its size.
- **The host stays ahead.** Launches are asynchronous; the host can queue a whole request's steps before the GPU has
  finished the first. Whether it actually keeps ahead depends on host launch cost against GPU time per kernel:
  check with `rocprofv3 --hip-runtime-trace --kernel-trace` (`profiling.md`). If the trace shows the GPU idle between
  kernels while the host is inside launch calls, the loop is host-bound and kernel count is a host cost too.

## Count kernels per token, then price them

Write the dispatch map (`references/optimization-method.md`) and multiply. Example (Qwen3-0.6B, 28 layers,
estimates at ~1.7 us per kernel boundary, against a ~0.27 ms streaming floor):

| design | kernels per layer | kernels per token | boundary cost per token |
|---|---|---|---|
| one kernel per operator (norm, QKV, QK-norm, RoPE, cache write, attention, O, add, norm, gate/up, SiLU*mul, down, add) | ~12 | ~340 | ~0.58 ms |
| fused: norm+QKV; QK-norm+RoPE+cache write+attention; O+residual; norm+gate/up+SiLU; down+residual | 5 | ~143 | ~0.24 ms |
| one persistent kernel per step, grid barriers between stages | 0 launches (5 barriers) | 1 | 1 launch + barriers (not measured) |
| one persistent kernel per request | 0 | 1 per request | barriers only (not measured) |

With the per-operator design the boundaries alone cost twice the streaming floor; the reference sits at ~1.7 ms per
token (`README.md`), so even a modestly fused engine has a large margin, and a persistent design aims at the floor.
The price of a boundary in your chain is not measured yet: measure it with the empty probe (`profiling.md`) before
trusting this table.

## Stream launches, hipGraph, persistent kernels: decide by measurement

- **Stream launches** are the baseline: simplest, kernel arguments change freely per step (position, ring slot).
- **hipGraph replay.** Capture one decode step (or several) with `hipStreamBeginCapture` / `hipStreamEndCapture`,
  `hipGraphInstantiate` once, `hipGraphLaunch` per step. Kernel arguments are frozen at capture, so everything that
  changes per step (position, ring slot, KV length) must be read by the kernels from device memory, advanced by a
  kernel at the end of each step. Measured here (one micro-benchmark, `counterexamples.md`): replay cost 1.69-1.72 us
  per kernel against 1.53-1.87 us for stream launches, i.e. no saving. SGLang captures decode as a graph by default
  and still runs at ~1.7 ms per token for a 0.6B model. Re-measure with your own step graph before deciding; if replay
  does not help, keep stream launches.
- **Persistent kernels** remove launches, not dependencies: each stage boundary becomes a grid-wide barrier whose cost
  depends on grid size and on the cross-XCD fences it needs (`kernels-fusion.md`). Measure the barrier cost by grid
  size first; a persistent design wins only if a barrier is cheaper than a launch boundary, or if it lets the next
  stage's weight loads start before the barrier (they do not depend on activations).
- **Launch tuning knobs** to measure, not to assume (`README.md`, Not measured yet): kernel arguments in device memory
  (`HIP_FORCE_DEV_KERNARG=1`; an engine may set its own environment before the first HIP call, but it must not rely
  on the harness setting it), fewer kernel arguments (pass one pointer to a device-side parameter block), and
  `hipSetDeviceFlags(hipDeviceScheduleSpin)` versus blocking sync for the one wait per request.

## Device-side argmax and the token tail

- **Argmax in the LM head's epilogue.** Nobody reads the logits in greedy decode, so the head GEMV does not store
  them. Each block reduces its rows to one (value, index) pair under "value descending, index ascending" and either
  writes it to a per-block partials array (a one-block tail kernel then reduces the partials in index order) or folds
  it into one 64-bit key with `atomicMax` on `unsigned long long`:

```cpp
__device__ inline unsigned long long argmax_key(float v, unsigned idx) {  // larger key: larger value, or lower idx
    unsigned u = __float_as_uint(v);
    if ((u << 1) == 0u) u = 0u;                                  // -0 ties +0
    const unsigned o = (u & 0x80000000u) ? ~u : (u | 0x80000000u); // order-preserving float -> uint
    return ((unsigned long long)o << 32) | (unsigned long long)(~idx);
}
```

  Both forms are deterministic (a max over a total order does not depend on arrival order). A NaN never wins if the
  merge uses `v > best || (v == best && i < bi)`.
- **The reference's row.** Round each logit to bf16 before comparing if SGLang's argmax sees bf16 logits widened to
  fp32 (read its `LogitsProcessor` and sampler at the pinned version); otherwise compare fp32. Either way ties go to the
  lowest id (the protocol's rule), and the judge counts an exact tie as agreement.
- **The token tail.** One small kernel (or the last block of the head, `kernels-fusion.md`) reads the winner, writes
  it to the ring, resets the key or partials for the next step, advances the device-side position if kernels read
  it from memory, and, unless this is the request's last step, gathers the next token's embedding row and computes
  the first layer's input RMSNorm. The first step of a request embeds its input with the ordinary gather.

## Continuations and state

- `reset: true` clears the KV length and position, runs the prompt, and generates `max_tokens` tokens; the first
  comes from the prefill's last row.
- `reset: false` first consumes the pending token (the last one emitted, still in the ring), then generates
  `max_tokens` more. After any reply the state has consumed everything except the last emitted token.
- The protocol has no stop tokens: every request generates exactly `max_tokens`, so the protocol path never stops
  early and never rewinds. Early stopping for serving is in `serving.md` (and the Metal backend's `decode-loop.md`
  for the rewind mechanics, which carry over).
- Position, KV length and ring index are host bookkeeping advanced at launch time (or device counters advanced by
  the tail kernel). Either way they describe the state after the queued work has run, which is only true after the
  request's synchronization.

## Checks for any change to the loop

- Tokens identical to the parent for one long request, a split request (`generate(P, 1)` then a continuation), and a
  chain of continuations; counts 1, 2 and the ring size (`scripts/same.py` covers these shapes).
- A continuation after a request whose count ended exactly at the ring size; a request after a reset; a request
  after a longer one (stale scratch).
- End of input during a request: launching stops, the stream is synchronized, the process exits 0.
- The kernel count per token, from a rocprofv3 kernel trace, equals the count in your dispatch map.
