# Model card template

Purpose: the one document that holds everything model-specific. The rest of the skill is model-agnostic; when you
start on a new model, copy this template into your work tree as `MODEL_CARD.md` and fill it in (phase 0 of
`workflow.md`). `examples/lfm2.5/model-card.md` is a complete filled-in card.

Rules for filling it in:
- Every fact names its source (config file, reference code with version and line, paper) and every number is
  labelled measured or estimated.
- Where sources disagree, record all of them and which one the reference follows. The reference is the
  specification, even where it disagrees with the paper.
- Keep it current: when a check reveals a semantic detail you had wrong, fix the card in the same change.

---

## 1. Identity

- Checkpoint: repository, revision (commit hash), file list with SHA-256.
- Reference implementation: package and exact version, plus the framework version underneath it. Note any version
  known to compute this model differently.
- Machine: chip, memory, OS version; measured streaming read bandwidth (GB/s) and matrix throughput (TFLOP/s) at the
  dtype you will use.

## 2. Configuration

The fields of the model's config that the engine uses, with values: hidden size, layers, heads, KV heads, head
dimension, vocabulary, context length, norm epsilon, rope parameters, expert count and top-k, intermediate sizes,
tied embeddings, special token ids. Note fields that exist but the reference ignores.

## 3. Forward pass

Pseudo-code of one token's forward, operator by operator, with the dtype of every input, output and accumulator and
every point where a value is rounded (for example to bf16). Include:
- the layer types and their order (if layers differ, a table of which layer is which);
- the norm formula (`w` versus `1 + w`, epsilon placement, accumulation dtype);
- the position encoding (variant, which dimensions rotate, how positions are counted);
- attention (scaling, masking, GQA grouping, softcapping if any);
- the MLP or mixture-of-experts (activation, gating, router dtype, top-k selection rule, score function, any bias
  used only for selection, how expert outputs are combined, shared experts);
- any recurrent or convolutional state (shape, dtype, update rule);
- the head (tied or not, final norm, logit dtype);
- the greedy rule, including how exact ties are broken.

## 4. Tensors

A table of every tensor: name, shape, stored dtype and quantization format, bytes. Describe each quantization format
precisely enough to dequantize one element by hand (bit layout, group size, scale and bias dtypes, formula). Note
alignment facts that matter for memory mapping (for example, offsets inside the file that are not page- or even
16-byte-aligned).

## 5. State between tokens

Everything the engine carries from one token to the next: KV cache (layout, dtype, bytes per token), recurrent or
convolution state, position, the pending token. Its size at the task's maximum context.

## 6. Costs and floors

| quantity | value | how computed |
|---|---|---|
| active weight bytes per decode token | | sum over the tensors a token reads (for MoE: the selected experts only) |
| KV / state bytes per decode token at context C | | |
| decode floor at context C (ms/token) | | bytes / measured streaming bandwidth |
| FLOPs per prefill token | | |
| prefill floor (tokens/s) | | FLOPs / measured matrix throughput |
| reference decode and prefill speed | | measured, same machine, same lengths |

The ratio between the reference's speed and the floor tells you how much room there is, and where.

## 7. Reference behavior that affects rounding

- The prefill partition the reference uses (chunk size, when it switches to L=1 steps).
- Per-shape kernel choices in the reference (for example a different attention kernel above some context length, or
  a different matrix kernel for small M); each choice is a rounding route you must reproduce.
- Settings that change results (fast-math, TF32 or similar reduced-precision modes, fused versus unfused paths).
- How often exact ties in the logits occur, so that tie-breaking is tested.

## 8. Open questions

Semantics you have not confirmed yet, and how you will confirm each one.
