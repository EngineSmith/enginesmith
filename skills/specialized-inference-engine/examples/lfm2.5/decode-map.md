# Decode dispatch map

One L=1 greedy decode step of the final engine, in the form `references/optimization-method.md` asks for, with the
GPU time and the bytes behind it. Three builds appear here; every number says which one it is from.

| build | what it is | dispatches per token (context < 8,192) |
|---|---|---:|
| **incumbent** | the agent loop's last best engine (version 31), parent of the final round | 355 |
| **base2** | incumbent + mechanisms 0-5 of the final round (`optimization-log.md`); the profiled build | 262 |
| **final** | base2 + the Q/K norm+RoPE fold into attention + router row staging (version 32, the released engine's decode path) | **252** |

The counts come from the source of each build's decode step (`decode_step_into`). The released engine's
`docs/ARCHITECTURE.md` gives the same 252 below 8,192 tokens of context, and 272 from there on, where decode attention
switches to its two-pass form.

## The loop

- `generate` keeps tokens on the GPU: the head's argmax epilogue leaves a key, and the token tail turns it into the
  token, the next step's embedding row and that row's first RMSNorm. Only step 0 of each call embeds and normalizes
  separately (and the last step skips the next embedding).
- Steps are encoded into command buffers of 4 steps, except the first buffer of each call, which holds 1 step so
  the GPU starts sooner. Each buffer is one serial compute encoder; buffers are committed without waiting, and the
  host waits once per request.
- Measured on base2 (prof, P=256): host encoding about 0.11 ms per step; the GPU is busy 99.3% of the decode request;
  no measurable idle time between command buffers; about 9 us per token of per-call start-up and wake-up, plus about
  5 us per token of request overhead.

## Map of one step (final engine, context < 8,192)

Every dispatch reads what the one before it wrote (read after write): there is no independent pair left in a step.

| # | stage (count per token) | kernel | grid x TG | bytes read | depends on |
|---|---|---|---|---|---|
| 1 | conv layer (x30) | `lfm_conv3_gemv_split_bf16`: B/C/X GEMV + the conv + the state update | 2048 x 96 | 7.08 MB `in_proj` + 12 KB taps + 8 KB state | `normed` (#12 or #14) |
| 2 | | `lfm_gemv_q4_bf16_k2048`: `out_proj` | 2048 x 32 | 2.36 MB | #1 |
| 3 | attention layer (x10) | `lfm_proj3_gemv_scatter_bf16`: Q, K, V GEMVs; V written into the cache | 3072 x 32 | 3.54 MB | `normed` |
| 4 | | `lfm_sdpa_vector_norm_rope_bf16`: Q/K RMSNorm + RoPE in the prologue, K into the cache, single-pass attention | 32 x 1024 | K and V: 2 KB per position | #3 |
| 5 | | `lfm_gemv_q4_bf16_k2048`: `out_proj` | 2048 x 32 | 2.36 MB | #4 |
| 6 | MoE layer (x38) | `lfm_router_addnorm_route_f32`: residual add + RMSNorm + F32 router GEMV; the last-finishing TG runs the top-4 | 64 x 256 | 512 KB F32 + 8 KB | `h`, `branch` (#2 or #5) |
| 7 | | `lfm_gemv_q4_gather_act_bf16`: gate + up of the 4 experts + SiLU product | 6144 x 32 | 14.16 MB | #6 (`normed`, ids) |
| 8 | | `lfm_gemv_q4_gather_down_combine4_bf16`: down of the 4 experts + weighted combine | 2048 x 128 | 7.08 MB | #7, #6 (weights) |
| 9 | dense layer (x2) | `add_rmsnorm_bf16_v8`: residual + `ffn_norm` | 1 x 256 | 12 KB | #2 |
| 10 | | `lfm_gemv_q4_gu_act_bf16`: gate + up + SwiGLU | 11776 x 32 | 27.13 MB | #9 |
| 11 | | `lfm_gemv_q4_bf16_k11776`: down | 2048 x 32 | 13.57 MB | #10 |
| 12 | every layer (x40) | `add_rmsnorm_bf16_v8`: residual + the next `operator_norm` (the final norm after layer 39) | 1 x 256 | 12 KB | #8 or #11 |
| 13 | end (x1) | `lfm_gemv_q6_bf16_x4_argmax`: 6-bit head GEMV, logits not stored, 64-bit argmax key | 8000 x 128 | 213.0 MB | #12 |
| 14 | end (x1) | `lfm_token_tail`: the token, the next embedding row, its RMSNorm | 1 x 256 | 1.7 KB + 4 KB | #13 |

Per token: 30 x 2 + 10 x 3 + 38 x 3 + 2 x 3 + 40 + 2 = **252**. From 8,192 positions on, #4 becomes three
dispatches: `lfm_norm_rope_qk_scatter_bf16` (5 x 256), `lfm_sdpa_vector_2pass1_gqa4_bf16` ((8 x 128) x 32 up to
49,152 positions, (8 x 512) x 32 above) and `lfm_sdpa_vector_2pass2_bf16` (32 x 1024): 272 per token.

## How the count fell (incumbent -> final)

| per | incumbent | final | mechanism |
|---|---|---|---|
| conv layer | 2 | 2 | (B/C/X + conv had been fused in the agent loop) |
| attention layer | 5: proj3, q/k norm, RoPE + K scatter, sdpa, out_proj | 3 | norm and RoPE fused (an agent candidate), then folded into the attention prologue |
| MoE layer | 5: add_rmsnorm, router GEMV, top-k, gate/up gather, down combine | 3 | add_rmsnorm into the router's prologue (an agent candidate); top-k into the router's last-finishing TG |
| dense layer | 5: add_rmsnorm, gate, up, silu_mul, down | 3 | gate + up + SwiGLU in one kernel |
| every layer | 1 add_rmsnorm | 1 | (fusing it into the down GEMV's last TG was measured 14% slower) |
| step start + end | 2 (gather, rmsnorm) + 3 (head, 2 argmax kernels) | 2 | head argmax epilogue + one-TG token tail |
| **per token** | **355** | **252** | 355 -> 307 (c9, c11) -> 269 (top-k) -> 262 (tail, dense) -> 252 (attention fold) |

## GPU time per token by category (base2, P=256, measured in the real engine)

Method (prof): the engine's own command-buffer timestamps give the true step time (3,280 us at P=256, 3,518 us at
P=4096). Per-kernel costs inside the chain come from duplicate experiments: an arm that adds one more dispatch of a
kernel per real one (with another layer's weights at least 8 layers back, so it reads DRAM like the real one) is
interleaved with a base arm, and the step-time difference is divided by the added dispatches. Timestamps around
each dispatch were not used directly: they inflate every dispatch by about 1.24 us. The per-kernel costs sum to
3,272 us against the measured 3,280 us. "Floor" = bytes / 590 GB/s + 1.95 us per dispatch.

| category | dispatches | us per token | share | excess over floor (us) |
|---|---:|---:|---:|---:|
| expert GEMVs (gate+up, down) | 76 | 1,640.8 | 50.1% | +124.0 |
| conv3 (B/C/X GEMV + conv) | 30 | 432.1 | 13.2% | +12.1 |
| head + token tail | 2 | 368.4 | 11.3% | +3.4 |
| router (add_rmsnorm + F32 GEMV + top-k) | 38 | 224.7 | 6.9% | +116.0 |
| out_proj (30 conv + 10 attention) | 40 | 216.5 | 6.6% | -21.8 |
| dense FFN (layers 0-1) | 4 | 154.5 | 4.7% | +8.7 |
| add_rmsnorm (40 per layer + 2 dense `ffn_norm`) | 42 | 86.2 | 2.6% | +3.4 |
| proj3 (Q, K, V) | 10 | 83.4 | 2.5% | +3.9 |
| attention (sdpa + Q/K norm/RoPE) | 20 | 65.6 (276.4 at P=4096) | 2.0% | +17.4 (+94.9 at P=4096) |
| **total** | **262** | **3,272** | | **+267** |

Per kernel (base2, P=256; "hot" = an in-place duplicate whose weights are still in the cache):

| kernel | per token | us in chain | us hot | MB | GB/s in chain | floor us |
|---|---:|---:|---:|---:|---:|---:|
| expert gate+up+act | 38 | 27.46 | 16.51 | 14.16 | 516 | 25.95 |
| expert down+combine4 | 38 | 15.72 | 11.69 | 7.09 | 451 | 13.97 |
| conv3 | 30 | 14.40 | 9.96 | 7.11 | 494 | 14.00 |
| head + argmax epilogue | 1 | 363.30 | - | 213.0 | 586 | 362.96 |
| router (fused) | 38 | 5.91 | 4.38 | 0.54 | 91 | 2.86 |
| out_proj | 40 | 5.41 | 4.54 | 2.36 | 437 | 5.96 |
| add_rmsnorm | 42 | 2.05 | - | 0.01 | - | 1.97 |
| dense gate+up+act | 2 | 48.94 | - | 27.14 | 554 | 47.94 |
| proj3 | 10 | 8.34 | 6.10 | 3.54 | 425 | 7.96 |
| dense down | 2 | 28.33 | - | 13.59 | 480 | 24.98 |
| sdpa (1-pass, 257-320 keys) | 10 | 4.95 | 3.85 | 0.53 | 108 | 2.85 |
| Q/K norm + RoPE | 10 | 1.62 | 1.62 | 0.01 | - | 1.97 |
| token tail | 1 | 5.09 | - | 0.07 | - | 2.07 |

Reading: the weight-streaming kernels run at 425-586 GB/s against about 590 GB/s for a plain stream; the remaining
time is mostly fixed per-dispatch cost (0.86-1.95 us per extra dispatch, measured) and two latency-bound kernels,
the router (91 GB/s) and the attention at long context. The expert down GEMV costs 15.7 us in the chain but 12.6 us
in an isolated micro-benchmark whose 57 MB weight cycle mostly stayed in the cache: profile inside the real chain.

**After the profile (final build, not profiled).** Two changes followed; their effects were measured in isolation
only, so the per-token figures are estimates:
- the Q/K norm+RoPE fold into the single-pass attention: -10 dispatches; -1.1 us per layer in a 10-layer kernel
  harness (measured), about -11 us per token (estimate);
- staging the router's 8 KB F32 row at kernel entry: 6.00 -> 4.75 us per layer behind a 7 MB DRAM stream (measured,
  micro-benchmark), about -47 us per token (estimate).

The final build's measured decode on 2026-10-25 was 304.0-305.5 tok/s at 256 tokens of context (3.27-3.29 ms per
token including the host; five re-measurements during the version series in `optimization-log.md`). That run and
the profile are different occasions on a laptop whose speed drifts by about 1%, so do not subtract one from the
other.

## Bytes per token by category (exact, from the tensor table)

| category | bytes per token | share of weights | GB/s in chain (base2) |
|---|---:|---:|---|
| routed experts (4 of 64, gate + up + down) | 806,879,232 | 55.1% | gate+up 516, down 451 |
| short conv (`in_proj`, taps, `out_proj`) | 283,484,160 | 19.4% | conv3 494, out_proj 437 |
| head (6-bit tied embedding, read in full) | 212,992,000 | 14.5% | 586 |
| dense FFN (layers 0-1) | 81,395,712 | 5.6% | gate+up 554, down 480 |
| attention projections | 58,984,960 | 4.0% | proj3 425, out_proj 437 |
| router (F32 gate + bias) | 19,932,672 | 1.4% | 91 |
| norms | 331,776 | 0.0% | |
| **weights** | **1,464,000,512** | | about 446 average over the 3.28 ms step (estimate) |
| KV cache read at context C | 20,480·C: 5.2 MB at 256, 83.9 MB at 4,096 | | |
| conv state (read; the same amount is written) | 245,760 | | |
