# Qwen3-0.6B on Trainium2 (trn2.48xlarge, one logical NeuronCore)

Source: Hugging Face Qwen/Qwen3-0.6B, one file model.safetensors (1,503,300,328 bytes), bf16. Facts from config.json:

| field | value |
|---|---|
| architecture | Qwen3ForCausalLM (dense decoder) |
| layers | 28 |
| hidden size | 1024 |
| attention heads / KV heads / head dim | 16 / 8 / 128 (q projection 1024 -> 2048) |
| MLP | SwiGLU, intermediate 3072 |
| norms | RMSNorm, eps 1e-6; Qwen3 also RMS-normalizes q and k per head before RoPE (q_norm, k_norm, 128 weights each) |
| RoPE | theta 1,000,000, full head dim, no scaling |
| vocabulary | 151,936; embeddings tied to the LM head |
| context | 40,960 positions (this task uses at most 4,097 + 65) |

Reference: stock NxD Inference (neuronx-distributed-inference 0.10.18399) through `enginesmith.references.nxdi`:
tensor parallel 1, bf16, fused QKV, async mode, on-device greedy sampling, context-encoding buckets 128, 256, ...,
4,096 and the task's longest length (the prompt is padded to the next bucket), token-generation buckets the same.

Measured (`enginesmith.references.nxdi`, trn2.48xlarge, 2026-11-03): decode 6.26 ms/token at 256 positions, 5.10 at
4,096; prefill 11 ms at 256 tokens, 155 ms at 4,096. Device time of one reference decode step (512-position bucket):
6.42 ms, 1.50 GB read from HBM (neuron-explorer).

Bytes per decode token (estimate from the checkpoint): about 1.19 GB of weights other than the embedding table (the
LM head reads the tied 151,936 x 1024 bf16 matrix, 311 MB, once per token) plus the KV cache: 28 layers x 2 x 8 x 128
x 2 bytes = 114,688 bytes per position (470 MB at 4,096 positions).
