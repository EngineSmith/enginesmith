# Model card: LFM2.5-24B-A2B, official MLX 4-bit checkpoint

The template of `references/model-card.md`, filled in for this example's engine. Sources, by the short names used
below:
- **hdr**: the checkpoint's `config.json`, `model.safetensors.index.json` and the three shard headers, read as text.
- **mlx-lm**: the reference's source, mlx-lm 0.32.0 (`models/lfm2_moe.py`, `models/switch_layers.py`,
  `models/cache.py`, `generate.py`); **MLX**: MLX 0.32.3's Metal backend source.
- **HF**, **llama.cpp**: Transformers v5.19.0 `models/lfm2_moe/` and llama.cpp 926862e5 (disagreement table only).
- **val**: a GPU validation run of the reference before the work began: 19 random-token prompts of 1 to 8128 tokens,
  65-token greedy `generate`, in two processes.
- **bench**, **arch**: the final engine's `docs/BENCHMARKS.md` and `docs/ARCHITECTURE.md` (2026-10-25).
- **prof**: the in-engine decode profile of build base2 (`decode-map.md`).
- **src**: the final engine's source; **notes**: the agent sessions' own records.

Labels: *measured* (with its source), *estimate* (with how it was computed), *inference* (read in source, not run).

## 1. Identity

- **Checkpoint**: `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` @ `667408d494bde191c2cb9dd4f4549827816b9275`. Converted with
  mlx-lm 0.32.0 from the base model `LiquidAI/LFM2.5-24B-A2B` (not the QAD checkpoint in LiquidAI's GGUF). Its model
  card says it requires mlx-lm 0.32.0 or later, because earlier versions route the experts incorrectly. License:
  LFM Open License v1.0.

  | file | bytes | sha256 (abbreviated) |
  |---|---:|---|
  | `model-00001-of-00003.safetensors` | 5,318,453,801 | `17e95345…11c0` |
  | `model-00002-of-00003.safetensors` | 5,344,895,125 | `c3ae09db…bb14` |
  | `model-00003-of-00003.safetensors` | 2,903,945,409 | `9814144a…9840` |
  | `config.json` | 2,148 | `4901707c…4244` |
  | `tokenizer.json` | 17,905,598 | `695be780…96f41` |
  | index, `tokenizer_config.json`, `generation_config.json`, `chat_template.jinja`, `LICENSE`, `README.md` | 80,192 / 363 / 329 / 5,496 / 10,574 / 3,688 | |

  870 tensors (358 + 331 + 181), 13,567,188,992 data bytes (the index's `total_size`), 23,971,585,280 parameters.
  sha256 of the sorted (name, dtype, shape) rows: `4d015ae76a51995d92e90ebfa6bec5a9d54b3d32dbe774275344563e73b4dc29`.
- **Reference**: unmodified mlx-lm 0.32.0 on MLX 0.32.3, loading with mlx-lm's own strict `load_model` and generating
  with `generate_step` (`lfm2_moe.py` sha256 `4234efaf…ff35`). **mlx-lm 0.31.3 and earlier, and mlx-vlm 0.7.0,
  compute a different model** (softmax routing, bias added to the probabilities, combine weights from the biased
  values); never take routing semantics from them.
- **Machine**: Apple M5 Max (`applegpu_g17s`), 128 GiB unified memory, macOS 26, 18 CPU cores; Metal recommended
  working set about 107.5 GiB, largest buffer about 80.6 GiB.
  - Streaming read bandwidth: **about 590 GB/s** (measured, arch: a plain streaming read). In the decode chain the
    213 MB head GEMV runs at 586 GB/s (measured, prof).
  - System-level cache: about 48-56 MB effective (measured, prof). A 7 MB stream cycled over a working set of 14-42 MB
    reads at about 810 GB/s, so a micro-benchmark must cycle more than about 70 MB of weights to see DRAM speed.
  - Fixed cost of one more dispatch in a serial chain: 0.86 us (1 x 32 threads, empty) to 1.95 us (2048 x 32)
    (measured, prof).
  - Matrix throughput: **not measured.** Estimate: the engine's measured prefill of 10,361 tok/s at 4K tokens
    (bench) times about 4.6 GFLOP per token there (section 6) is about 47 TFLOP/s achieved, a lower bound on the
    BF16 peak.

## 2. Configuration (hdr: `config.json`)

| field | value |
|---|---|
| `model_type`, `dtype` | `lfm2_moe`, `bfloat16` |
| `num_hidden_layers`, `layer_types` | 40: 30 `conv` + 10 `full_attention` at layers 2, 6, 10, ..., 38 (i % 4 == 2) |
| `hidden_size`, `vocab_size`, `max_position_embeddings` | 2048, 128000, 131072 |
| `num_attention_heads`, `num_key_value_heads`, head dim | 32, 8, 64 (= 2048 / 32); 4 query heads per KV head |
| `intermediate_size` (dense), `num_dense_layers` | 11776, 2 (layers 0 and 1, both conv layers) |
| `num_experts`, `num_experts_per_tok`, `moe_intermediate_size` | 64, 4, 1536 (MoE in layers 2-39, no shared expert) |
| `norm_topk_prob`, `use_expert_bias`, `routed_scaling_factor` | true, true, 1.0 |
| `conv_L_cache`, `conv_bias` | 3, false |
| `norm_eps`, `rope_parameters` | 1e-5; theta 1e7, type default |
| `tie_word_embeddings` | true: the head is the embedding table |
| `quantization` | affine, 4 bits, group 64; `model.embed_tokens` 6 bits, group 64; the router gate left in F32 |
| special ids | BOS 124894 `<\|startoftext\|>`, EOS 124900 `<\|im_end\|>`, PAD 124893, `<\|im_start\|>` 124899, `<think>` 124901, `</think>` 124902 |

- Tokenizer: byte-level BPE with ids 0..125016. Ids 125,017-127,999 have no tokenizer entry but still have rows in the
  embedding and logits; the argmax runs over all 128,000.
- Present but not used by the reference or this task: the sampling settings of `generation_config.json` (temperature
  0.1, top_k 50, repetition_penalty 1.05; the task is greedy); the BOS of `tokenizer_config.json` (mlx-lm adds none
  to token ids; prompts are token ids, no template); the `quant_predicate` of `lfm2_moe.py` (an 8-bit gate at
  conversion time; the published gate is F32); `routed_scaling_factor` 1.0 (a no-op multiply).

## 3. Forward pass (mlx-lm; HF and llama.cpp agree except where the table below says)

```
h = dequant(embed_tokens[token])                    # 6-bit row -> bf16; no scaling, no norm after it
for l in 0..39:
    u = RMSNorm(h, operator_norm[l])                # bf16
    h = h + (ShortConv_l(u) if conv layer else Attention_l(u))
    x = RMSNorm(h, ffn_norm[l])
    h = h + (DenseFFN_l(x) if l < 2 else MoE_l(x))
logits = quantized_matmul(RMSNorm(h, embedding_norm), embed_tokens^T)   # bf16, all 128000 rows
token  = first argmax of float32(logits)            # exact ties go to the lowest id
```

Pre-norm, two residual adds per layer, no post-norms, no layer scale. Every value between operators is bf16.

- **RMSNorm**: `y = x / sqrt(mean(x^2) + 1e-5) * w`, plain gain `w` (not `1 + w`); `mx.fast.rms_norm`, float32
  statistics inside (inference). Width 2048 for the operator, ffn and final norms; width 64 for the q/k norms (one
  `[64]` weight shared by all heads of a layer). `embedding_norm` is the final norm only.
- **Short conv** (30 layers), double-gated, no bias, no activation, no norm inside:
  ```
  B, C, X = split(in_proj · u, 3)                  # 6144 = 3 x 2048, in the order B, C, X
  v_t     = B * X                                   # first gate (bf16)
  z_t[c]  = w[c][0]·v_{t-2}[c] + w[c][1]·v_{t-1}[c] + w[c][2]·v_t[c]   # causal depthwise conv, 3 taps
  out     = out_proj · (C * z_t)                    # second gate, then the projection
  ```
  The carried state is the last two rows of `v` (after the first gate), bf16, zeros before the first token. mlx-lm
  concatenates the state and `v` and runs `nn.Conv1d` without padding.
- **Attention** (10 layers): `q = q_proj·u` (32 x 64), `k = k_proj·u`, `v = v_proj·u` (8 x 64 each); per-head RMSNorm
  of q and k, then RoPE (`nn.RoPE(64, base=1e7, traditional=False)`: NeoX half split over all 64 dims, the pair
  `(x_i, x_{i+32})` rotated by `pos · 1e7^(-2i/64)`, positions 0-based absolute, offset = the cache length);
  `mx.fast.scaled_dot_product_attention` with scale 1/8, causal mask for L > 1, no mask for L = 1; query head h
  reads KV head h / 4. No output gate, sinks, soft-cap, window or bias. The cache stores k after norm and RoPE.
- **Dense FFN** (layers 0-1): `down · (SiLU(gate·x) * (up·x))`, width 11776.
- **MoE** (layers 2-39):
  ```
  s   = sigmoid(W_router · x)                       # W_router F32 (64 x 2048): the one F32 matmul; s in F32
  sel = argpartition(s + expert_bias, kth=-4)[-4:]  # the bias is used ONLY for selection; unordered among the 4
  g   = bf16(s[sel] / (sum(s[sel]) + 1e-6))         # weights from the UNBIASED s, cast to bf16
  out = sum_i g_i · down_e(SiLU(gate_e·x) * (up_e·x)),  e = sel_i   # width 1536; summed over the selection axis
  ```
- **Head and greedy**: tied (`embed_tokens.as_linear`), bf16 logits. The argmax is the first maximum of the float32
  copy. mlx-lm's own `generate_step` takes the argmax of bf16 `logits - logsumexp(logits)`, where two distinct maxima
  can round together (inference; it did not happen on val's 19 paths).
- **Accumulation**: the quantized matmuls accumulate in float32 inside MLX's kernels and emit bf16 (inference).

### Where the sources disagree

| topic | mlx-lm 0.32.0 (the reference) | HF 5.19.0 | llama.cpp | effect |
|---|---|---|---|---|
| router | sigmoid; bias for selection only; unbiased weights | same | same | mlx-lm <= 0.31.3 and mlx-vlm 0.7.0: softmax, biased weights (a different model) |
| normalization guard | `sum + 1e-6` | same | sum clamped at >= 6.1035e-5 | rounding |
| combine-weight dtype | bf16 before weighting | bf16 | F32 | rounding |
| expert sum order | selection order | expert-id order | selection order | rounding |
| top-k ties | `argpartition` (unordered) | `torch.topk` | `argsort_top_k` | unknown order among equal `s + bias` |
| activations | bf16, F32 router | bf16, F32 norm statistics | F32 hidden | rounding |
| RoPE angle | `mx.fast.rope` at the cache offset | `pos · inv_freq` in F32, cos/sin to bf16 | CPU / Metal `pos · pow` | rounding, growing with position (about 1e-2 rad near 1.3e5, estimate) |
| conv state rows | 2 | 3 (the oldest unused) | 2 | none |
| BOS on plain text | not added | not added | added | none for token-id prompts |
| weights | this checkpoint | the base bf16 model | the QAD Q4_0 GGUF | a different checkpoint, not rounding |

No source disagrees on the layer composition, the B/C/X order, the tap order, the absence of a conv activation, q/k
norm before RoPE, full-width NeoX RoPE, the GQA grouping, the 1/8 scale, the dense layers 0-1, the final-norm
placement, the tied head, or plain-gain RMSNorm with eps 1e-5.

## 4. Tensors (hdr)

Formats (MLX 0.32.3 affine quantization):
- **4-bit/64** (every linear and expert weight): `.weight` U32 codes of shape `(rows, cols/8)`, eight codes per word,
  element j in bits `4·(j mod 8)` of word `j div 8`; `.scales` and `.biases` bf16 of shape `(rows, cols/64)`.
  `w = scale · q + bias`, q in 0..15. A 2048-wide row is 1,024 B of codes + 128 B of scales and biases = 1,152 B
  (4.5 bits per weight).
- **6-bit/64** (the tied embedding): 16 codes per 3 words (384 words per 2048-wide row), same scale/bias arithmetic,
  1,664 B per row. Take the exact cross-word packing from MLX's `quantized.h` (`bits = 6`) and check rows against
  `mx.dequantize`.
- The router gate and the expert bias are F32; every other small tensor is bf16.

| module (per layer unless global) | stored as | layers | bytes, all layers |
|---|---|---|---:|
| `model.embed_tokens` (6-bit; also the head) | U32 (128000, 384) + bf16 (128000, 32) x 2 | global | 212,992,000 |
| `model.layers.N.feed_forward.switch_mlp.{gate,up}_proj` | U32 (64, 1536, 256) + bf16 (64, 1536, 32) x 2, each | 2-39 | 8,606,711,808 |
| `model.layers.N.feed_forward.switch_mlp.down_proj` | U32 (64, 2048, 192) + bf16 (64, 2048, 24) x 2 | 2-39 | 4,303,355,904 |
| `model.layers.N.feed_forward.gate.weight` (router) | **F32** (64, 2048), unquantized | 2-39 | 19,922,944 |
| `model.layers.N.feed_forward.expert_bias` | **F32** (64,) | 2-39 | 9,728 |
| `model.layers.N.feed_forward.{gate,up,down}_proj` (dense) | 4-bit, (11776 x 2048) and (2048 x 11776) | 0, 1 | 81,395,712 |
| `model.layers.N.conv.in_proj` | 4-bit (6144 x 2048) | 30 conv | 212,336,640 |
| `model.layers.N.conv.out_proj` | 4-bit (2048 x 2048) | 30 conv | 70,778,880 |
| `model.layers.N.conv.conv.weight` | bf16 (2048, 3, 1) (MLX `Conv1d` layout) | 30 conv | 368,640 |
| `model.layers.N.self_attn.{q_proj,out_proj}` | 4-bit (2048 x 2048), each | 10 attn | 47,185,920 |
| `model.layers.N.self_attn.{k_proj,v_proj}` | 4-bit (512 x 2048), each | 10 attn | 11,796,480 |
| `model.layers.N.self_attn.{q,k}_layernorm.weight` | bf16 (64,) | 10 attn | 2,560 |
| `model.layers.N.{operator,ffn}_norm.weight`, `model.embedding_norm.weight` | bf16 (2048,) | all + global | 331,776 |
| **total** | 221 U32, 573 bf16, 76 F32 tensors | | **13,567,188,992** |

- The routed experts are 12,910,067,712 B (95.2%). No fused QKV, no shared expert, no bias except `expert_bias`, no
  `lm_head` tensor. Expert slice `e` of a stacked tensor is expert `e`.
- **Alignment**: the three header lengths are odd (43,041 / 40,333 / 21,945 B, unpadded), so the data sections start
  at file offsets 43,049 / 40,341 / 21,953 and every tensor's file offset is 1 mod 4. Metal buffer offsets must be
  4-byte aligned, so a shard cannot be bound in place; the final engine reads each tensor into its own Metal buffer
  at load, in the stored layout (src: `load_tensor`).

## 5. State between tokens

- **KV cache**: 10 layers x (K and V) x 8 heads x 64 x bf16 = 20,480 B per position; 2.68 GB at 131,072 positions.
  K is stored after the norm and RoPE. mlx-lm's `KVCache` grows in steps of 256 positions.
- **Conv state**: 30 layers x 2 rows x 2048 x bf16 = 245,760 B, the last two rows of `v = B * X`.
- **Position and pending token**: after generating N tokens from a P-token prompt, P + N - 1 tokens are consumed and
  the last emitted token is pending (fed by the next step). Reset clears the conv states, the KV cache and the
  position. There is no other state (no scan state, no token history).
- The reference's memory (measured, val): 13,567,189,000 B active after load; peak 14,746,949,779 B at 8128 tokens.

## 6. Costs and floors

| quantity | value | how computed |
|---|---|---|
| active weight bytes per decode token | **1,464,000,512 B**: experts 806.9 MB (55.1%), conv 283.5 MB (19.4%), head 213.0 MB (14.5%), dense FFN 81.4 MB (5.6%), attention 59.0 MB (4.0%), router 19.9 MB (1.4%), norms 0.3 MB | exact: every non-expert tensor in full plus 4/64 of the experts (`scripts/bytes_per_token.py` with the spec below) |
| KV and state bytes per token at context C | 20,480·C + 245,760 B: 5.5 MB at 256, 84.1 MB at 4096, 2.68 GB at 131,072 | section 5 |
| decode floor | 2.49 ms (401 tok/s) at 256; 2.62 ms (381 tok/s) at 4096; 7.03 ms (142 tok/s) at 131,072 | estimate: bytes / 590 GB/s |
| FLOPs per prefill token | about 4.39 GFLOP of matrix work + 81,920·T for attention at position T | estimate: 2 x the 2.19 B weights a token multiplies (the head excluded: it runs at the last position only); attention 10 layers x 32 heads x 64 x 2 x 2 per key |
| prefill floor | not computed | matrix throughput not measured |
| reference (mlx-lm) speed | decode 190.0 / 188.5 / 184.0 / 82.2 tok/s at 256 / 1K / 4K / 128K context; prefill 3,800 / 6,981 / 8,829 / 4,032 tok/s | measured, bench |
| final engine speed | decode 304.5 / 300.2 / 285.1 / 119.1 tok/s; prefill 4,243 / 8,032 / 10,361 / 4,662 tok/s | measured, bench |

Reading: at 256 tokens the reference decodes at 47% of the floor speed (190 of 401 tok/s) and the final engine at
76% (304.5). At 128K the ratios are 58% and 84% of 142 tok/s; the KV read is 65% of the bytes there. Of the profiled
build's 3.28 ms GPU step at 256 tokens (prof), about 2.50 ms is the byte floor, about 0.51 ms the fixed cost of 262
dispatches at 1.95 us each (an upper bound for the small kernels), and 0.27 ms the excess of individual kernels over
their floor.

The spec for `scripts/bytes_per_token.py` (it reproduces the 1,464,000,512 B; the first matching group wins):

```json
{"groups": [
  {"name": "routed experts (4 of 64)", "patterns": ["model.layers.*.feed_forward.switch_mlp.*"], "fraction": 0.0625},
  {"name": "router", "patterns": ["model.layers.*.feed_forward.gate.weight", "model.layers.*.feed_forward.expert_bias"]},
  {"name": "dense FFN (layers 0-1)", "patterns": ["model.layers.*.feed_forward.*"]},
  {"name": "short conv", "patterns": ["model.layers.*.conv.*"]},
  {"name": "attention", "patterns": ["model.layers.*.self_attn.*"]},
  {"name": "norms", "patterns": ["model.layers.*_norm.weight", "model.embedding_norm.weight"]},
  {"name": "tied embedding = head", "patterns": ["model.embed_tokens.*"]}],
 "per_position_bytes": 20480, "state_bytes": 245760}
```

## 7. Reference behavior that affects rounding

- **Prefill partition** (mlx-lm `generate_step`): chunks of `min(2048, remaining - 1)` tokens, each evaluated and
  followed by `mx.clear_cache()`, then the last prompt token alone as an L=1 step; teacher-forced forwards use the
  same partition. Decode is pipelined and clears the cache every 256 tokens. The final engine keeps this partition
  and merges consecutive reference chunks into one submission of up to 8,192 rows, which leaves every row's
  arithmetic unchanged (src: `next_chunk`, `next_group`).
- **The reference is not partition-invariant** (measured, val): feeding a prompt one token at a time instead of in
  its chunks moved the last logits row by up to 6.6 and flipped its argmax at 7 of 18 lengths.
- **Quantized matmul by shape**: an L=1 row takes MLX's `qmv`/`qdot` form (float32 sums of `scale·Σq·x + bias·Σx`
  per group); a prompt chunk takes `qmm_t` (dequantize to bf16, then a matrix multiply) (notes, a repair made after
  reading MLX's kernels). `gather_qmm` sorts tokens by expert when `indices.size >= 64`, i.e. from 16 tokens (mlx-lm).
- **Attention by shape**: L=1 rows use `sdpa_vector`. MLX 0.32.3 switches an L <= 8 query to `sdpa_vector_2pass`
  when (the architecture suffix is `d` or `s` and there are >= 1024 keys) or (GQA and >= 4096 keys) (MLX). This
  machine is `applegpu_g17s`, so by that rule the reference is two-pass from 1,024 keys (inference). The final engine
  is single-pass below 8,192 keys and two-pass from 8,192, and passes the task's token checks that way (open question
  below). Prompt chunks use MLX's steel flash-attention tile, whose softmax
  uses `scale·log2(e)` and `exp2`.
- **TF32**: `MLX_ENABLE_TF32` changed the generated tokens at 6 of the 19 val lengths (measured); the reference is run
  with MLX's default. The F32 router matmul is the likely place (inference).
- **Fast math**: Metal compiles with fast-math by default, so the compiler may contract or reassociate an expression
  you copied; pin the order (explicit `fma`) and verify bits (`optimization-log.md` has two cases).
- **Ties**: bf16 logits keep 8 significant bits; exact top-2 ties occurred on 8 of the 19 val paths, 0-2 per 65-step
  path (measured); the median top-2 margin per path was 0.75-10.4.
- **Consequence** (measured, bench): the final engine's greedy tokens differ from the reference's at 4.5% of held-out
  teacher-forced steps (186 of 4,160). Bit equality with the reference is not reachable, so every optimization was
  checked bit for bit against its parent engine instead.

## 8. Open questions

- The order among equal `s + bias` values in the top-4 (`argpartition` is unordered): never observed; test with
  synthetic ties if a mismatch points there.
- Whether a row without a tokenizer entry (ids >= 125,017) ever wins the argmax: not checked; the engine includes
  them, as the reference does.
- The RoPE angle rounding near position 1.3e5 across implementations: not measured.
- The L=1 attention route between 1,024 and 8,191 keys (two-pass by MLX's rule on this machine, single-pass in the
  engine): count teacher-forced mismatches both ways at those lengths before changing it.
- Which operation the TF32 setting changes: not isolated.
