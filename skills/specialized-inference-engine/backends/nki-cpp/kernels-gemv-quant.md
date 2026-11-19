# GEMVs on Trainium2 (decode)

Purpose: decode GEMV rates at real model shapes and the rules that keep them at the streaming ceiling. Measured on
one trn2.48xlarge (nki 0.6.0, neuronx-cc 2.27.5334, LNC=2 unless noted) by the kernel lab, 2026-11-03.

- Use the weight-stationary column layout of `kernels-fusion.md`: 1.25 GB per Qwen3-0.6B step at 561 GB/s, the
  Qwen3-8B LM-head-shaped GEMV (1.24 GB) at 661 GB/s, a whole 36-layer Qwen3-8B step at 637 GB/s (23.86 ms, within
  5% of the 22.7 ms floor at 670 GB/s).
- **The PSUM accumulation-depth cliff**: keep at most about 8k accumulating matmuls per PSUM tile.

  | output tiles per core x K tiles, PSUM tile width | GB/s |
  |---|---|
  | 96 x 32, 512 | 589 |
  | 192 x 32, 512 | 600 |
  | 512 x 32, 512 (16,384 matmuls into one PSUM tile) | **176** |
  | 592 x 32, 512 | 292 |
  | 592 x 32, 256 or 128 | **661** |
  | 2368 x 8 (K = 1,024), 512 | 674 |

  In a Qwen3-8B step this made the LM head take 4.25 ms instead of 1.9. The HBM layout of the weight did not matter
  here (a 4.8 MB partition stride and contiguous 1 MB blocks both gave 292 GB/s).
- Where the weight stream is triggered matters: issuing the weight DMAs on hardware DGE from an engine that does
  no compute (`dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync`) took the Qwen3-0.6B step from 2.23 to
  2.10 ms and the Qwen3-8B step from 23.86 to 23.11 ms (658 GB/s); hardware DGE on the sync and scalar queues
  alternating was slower (2.378 ms), because the ScalarE also computes.
- The remaining loss of the Qwen3-0.6B step (2.235 ms against a ~1.9 ms floor) is serial latency at each layer
  (norms, the core exchanges, attention between GEMVs) with limited weight prefetch; a layer's weights per core
  (123 KB per partition) do not fit twice in SBUF.

## Quantization: not allowed for a bf16 checkpoint

Requantizing bf16 weights to fp8 (e4m3, per-output-channel scale) made the steps 1.7x faster (Qwen3-0.6B 2.23 ->
1.35 ms, Qwen3-8B 23.9 -> 13.8 ms) but changed the model: top-1 agreement with HF fp32 under teacher forcing fell from
478/479 to 447/479 (0.6B) and 197/199 to 184/199 (8B), while the first 16-24 generated tokens stayed identical (short
greedy checks hide it). Changing weight values is outside the engine rules; a quantized checkpoint is a different
task (and model).

Extra low-precision copies count the same way. An fp8 copy of the weights used as the draft model, or an fp8 LM
head that screens candidates before an exact rescore, are both reported separately and do not enter the project's
results: only engines whose every weight in use is the checkpoint's bf16 (in any layout) count. Draft with the model
itself (prompt lookup, n-gram) instead.

## Decode GEMVs: TensorE column tiling (gpt-oss-20b, measured)

An activation-stationary GEMV uses 1 of the 128 PE columns. `nc_matmul(..., tile_position=(0, 32 * c),
tile_size=(128, 32))` runs up to 4 matmuls with stationaries of at most 32 columns in parallel on the 4 column
groups, the technique of AWS nkilib's MoE token-generation kernels. Give each column tile different output columns and
accumulate over the contraction in the same order as before: every output stays bitwise identical. Pad the
stationary x to 32 columns with zeros (x in column 0) so all 32 PSUM rows are written; the PSUM copy and the
exact fp32 transposes back then never read uninitialised data. On gpt-oss-20b (4 logical cores, MXFP4 experts),
the 4 experts' gate/up went from 30.8 us to about 8 us of TensorE. Applied to gate/up, down, QKV and o_proj, decode
went from 300 to 248 ms per 64 tokens. Splitting the contraction over the tiles instead needs a cross-tile fp32 sum
and gave only 2x.

The LM head works the same way: the hidden vector stationary, 4 vocab tiles streaming as moving operands in the 4
column groups, one exact transpose per group. 4,600 one-column weight-stationary matmuls became 1,150 rounds of 4,
and the head became DMA-bound (about 620 -> 500 us).

For the router's top-k use `max8` (the 8 largest, descending) and `nc_find_index8` (duplicates get ascending
positions, so ties go to the lowest expert id, as a reduce/compare loop would). Router plus top-4 went from about 13
to 7 us per layer.
