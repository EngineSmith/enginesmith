# Counterexamples (measured on a trn2.48xlarge)

One line each: what was tried, what it did, the source. Ideas from the Metal backend are hypotheses here.

- GEMV 4096 x 4096 bf16, splitting the matmul into column tiles of 32 or 64 (`gemv_ct32`, `gemv_ct64`): no
  meaningful change against 2048-wide slabs. `example/`, 2026-11-03.
- Same GEMV, hardware DGE or `dge_mode.none` for the weight DMA instead of the default software DGE: no meaningful
  change. `example/`.
- Same GEMV, the whole weight kept in SBUF: little or no gain over streaming large slabs. `example/`.
- Weights as the stationary operand and a [1, K] row of x moving (`gemv_stat`): 100.7 us against 81.9 us for x
  stationary with the same slabs. `example/`. With x as a single moving *column* the weight-stationary form is the
  faster one (`kernels-fusion.md`).
- Pinning the host process to one CPU: synchronous `nrt_execute` 71 -> 45 us, but queued executions 24 -> 72 us.
  `example/`.
- Reference configuration (NxD Inference, Qwen3-0.6B): its NKI QKV, attention and MLP kernels made decode slower
  (7.42 against 6.50 ms/token at 256 positions); its token-generation attention kernels do not trace for Qwen3
  (`Invalid permutation specified`). Async mode plus fused QKV was fastest (6.26 ms). Reference configuration sweep, 2026-11-03.

What did help: weight slabs of [128 x 2048] instead of [128 x 512] (123.6 -> 81.9 us), loading x with one strided
DMA (77.0 us), and swizzling the weight at load time so each partition reads 16 KB contiguous per DMA (74.8 us,
448 GB/s). `example/`.

## Speculation: harness score up, real text down
On the random-token timed prompts a draft-and-verify decoder looks far better than it is. Measured on 100 IFEval /
MATH-500 prompts against NxD Inference: Qwen3-0.6B harness 3.01x but real text 2.6x (a token-by-token engine already
gives 2.5x); Qwen3-14B harness 4.65x -> 5.17x between two rounds while real-text speed fell 1.73x -> 1.64x. A
speculation change is kept only if real-text generation speed (task accuracy, quality.py) improves.

- **A DMA whose offset comes from SBUF (`scalar_offset` / `vector_offset`) can read the offset before the
  instruction producing it has written it**, even when the producer is another DMA from a kernel input. On
  DeepSeek-V4.1-Flash this read wrong expert weights, and caused NRT_EXEC_OOB crashes at random layers. Remove
  run-time offsets from kernels, or compile one NEFF per offset choice.

- **Sharing SBUF between tensors: use `.view`, not fixed addresses.** Regions declared at explicit SBUF addresses
  (`nl.ndarray(..., address=...)`) with another tensor at the same address gave deterministic wrong tokens on
  gpt-oss-20b: the auto-allocator does not avoid explicit addresses, and the compiler keeps internal constants
  (such as the transpose identity) at the top of SBUF. A `.view` of one region worked, with `instr.depends_on()` to
  order the DMAs that reuse it.
- **Stacked tensors past 2^31 elements**: DMA offsets beyond that are out of range, and only fail at run time
  (NRT_EXEC_OOB from layer 17 of a 24-layer expert stack). Split the stacks.
- **A position-indexed DMA into a 4-D cache must name the position dimension in `indirect_dim`.** With the index
  of a 3-D layout, a zero-input test at position 0 passed and the engine failed later with NRT_EXEC_OOB.
- **Chunk sizes must divide the length**: `CH = 512 if NK % 512 == 0 else 256` silently skipped the last 128 keys
  for NK = 384 (stale SBUF, NaN rows that changed between runs). Take the largest of 512 / 256 / 128 that divides,
  and assert it.
- **A strided DMA of 4-byte elements can be very slow**: a bias row gathered from [n, t] order took about 100 us per
  expert; a transposed copy made at load time made it one contiguous row.
