# Prefill

Purpose: how to make prompt processing fast on gfx942 without leaving the reference's numerics: what the reference
costs, bf16 GEMMs with MFMA intrinsics, tiling for 304 CUs and 64 KiB of LDS, double buffering, the small-M case,
chunking long prompts, and copying tile configurations from CK and aiter. Read it before changing a prefill GEMM or
the chunking. Prefill attention is in `attention.md`; timing methods are in `profiling.md`.

No prefill kernel has been measured on this platform yet. The reference's timings are measured (`README.md`); FLOP
counts and throughputs derived from them are estimates.

## Know what you are up against

- FLOPs per prompt token are about `2 x layer parameters` (the LM head runs only for the last row) plus attention
  (`2 x T^2 x D x q_heads x layers` per prompt of T tokens, causal). Put both in the model card (section 6).
- Example (Qwen3-0.6B, estimates from the shapes and the reference's measured request times):

| prompt tokens | projections | attention | reference request | effective |
|---:|---:|---:|---:|---:|
| <= 2048 | <= 1.8 TFLOP | <= 0.5 TFLOP | ~14 ms | framework-bound |
| 4097 | 3.6 TFLOP | 1.9 TFLOP | 23 ms | ~240 TFLOP/s |
| 8192 | 7.2 TFLOP | 7.7 TFLOP | 45 ms | ~330 TFLOP/s |
| 32768 | 28.9 TFLOP | 123 TFLOP | 331-338 ms | ~460 TFLOP/s |

  Up to ~2K tokens the reference's request time is fixed overhead (the same ~14 ms at 1 token): an engine that runs
  the forward without that overhead wins by a large factor there. At 32K the reference runs at a substantial fraction
  of the ~1.3 PFLOP/s vendor peak, mostly in attention; there you need tiles as good as aiter's.
- **Measure your ceiling** before judging a kernel: your own plain bf16 MFMA GEMM at the production shapes
  (M = 4096, N = 6144, K = 1024 and so on), in TFLOP/s. Vendor peak is not a measurement.

## Keep the reference's partition (and know what it is)

- **SGLang's driving loop** (read `server_args.py` and the scheduler in the image the reference runs from): chunked
  prefill splits a prompt longer than `chunked_prefill_size` into chunks (the default depends on the version and the
  GPU; read it, do not assume); the whole last chunk, including the last prompt token, is one extend batch whose last
  row gives the first token. Unlike mlx-lm, the last prompt token is not a separate L = 1 step.
- **A continuation's pending token** is computed as a one-token extend on the cached prompt (the harness's reference
  docstring); later tokens are decode steps. A one-token extend may take a different attention kernel than a decode
  step in the reference; in your engine the same L = 1 path may serve both.
- Your GEMMs are not the reference's (hipBLASLt or aiter tuned kernels picked per M), so bitwise agreement is not
  reachable; agreement within near-ties is. What you can control: fp32 accumulation, bf16 rounding of each operator's
  output where the reference stores it, and a partition whose rounding is independent of chunk size.
- **Make chunk size free of numerics in your own engine.** If every GEMM output's K order and every query row's key
  tile order is fixed regardless of how many rows the chunk has, chunks can be as large as scratch allows, and the
  teacher-forced forward equals generation (self-check 5 in `references/correctness.md`). Check it bitwise: the same
  prompt prefilled in one chunk and in several must give the same bytes.

## Small M (short prompts and the last chunk)

For M up to a few tens of rows a GEMM is still weight-streaming: read each weight once and compute all M rows from
it. Two candidates to measure against each other:
- a multi-row GEMV (the decode layout of `kernels-gemv-quant.md` with M accumulators per lane);
- an MFMA GEMM whose M dimension is padded to 16.

Find the crossover M by measurement. Keep the per-output K order the same as in the large-M path if you want
chunk-independent bits (often not possible across GEMV and MFMA layouts: then the route by M is a rounding route of
your own, fixed by shape, and checked against the reference at both sides of the switch).

## MFMA on gfx942

- Intrinsics (clang builtins; vendor ISA documentation):
  - `__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, cbsz, abid, blgp)`: `a`, `b` are 4 bf16 per lane (a 4 x
    `short` vector), `c` and the result 4 floats per lane; D = A (16 x 16) x B (16 x 16) + C, one wave.
  - `__builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a, b, c, ...)`: 4 bf16 per lane in, 16 floats per lane out, D = A
    (32 x 8) x B (8 x 32) + C.
  - Pass `0, 0, 0` for the broadcast controls unless you know you need them.
- **Operand layout** (16x16x16, as commonly documented; verify with a unit test or AMD's matrix instruction
  calculator before building on it): lane `l` holds A row `l % 16`, k = `4 * (l / 16) + 0..3`; B column `l % 16`, the
  same k; and receives C column `l % 16`, rows `4 * (l / 16) + 0..3`. Write a one-instruction test kernel against a
  scalar fp32 product first.
- **Accumulation is fp32** inside the instruction; the order in which an MFMA sums its K products is the hardware's.
  Your K loop order across instructions is yours: fix it.
- **Registers:** MFMA results can live in AGPRs (accumulation registers); the compiler allocates them. Watch the
  kernel-resource-usage remark for spills (`runtime.md`).

## Tiling for gfx942 (hypotheses to measure)

- **Block tile:** 256 threads (4 waves), output tile 128 x 128 or 256 x 128, K step 32 or 64; each wave computes a 64 x
  64 (or 32 x 128) sub-tile as a grid of 32 x 32 or 16 x 16 MFMAs. These are the shapes CK's and hipBLASLt's bf16
  configurations for MI300-class GPUs are built around; copy a configuration that CK ships for gfx942 and tune from
  it rather than inventing one.
- **LDS budget:** 64 KiB per CU. A 128 x 64 bf16 A tile and a 64 x 128 B tile take 16 KiB each: double-buffering both
  uses 64 KiB and allows one block per CU. Alternatives: single LDS buffer plus prefetch of the next K tile into
  registers (global -> VGPR while computing from LDS, then VGPR -> LDS), which CK's pipelines use; or K step 32.
  Measure occupancy against latency hiding.
- **LDS layout:** pad or swizzle rows so that the reads feeding MFMA operands are conflict-free (32 banks of 4 bytes);
  `ds_read_b128` / `ds_write_b128` for 16-byte accesses.
- **Grid:** enough output tiles to fill 304 CUs several times. Example (Qwen3-0.6B, M = 4096): gate+up N = 6144 gives
  32 x 48 = 1536 tiles of 128 x 128; down N = 1024 gives only 256 tiles (fewer than the CUs): use smaller tiles or
  split-K with a fixed-order reduction there (no float atomics).
- **XCD locality:** blocks are dispatched round-robin to the 8 XCDs (vendor documentation), each with its own 4 MB L2.
  Remapping block ids so that tiles sharing an A or B panel run on the same XCD is a known tuning lever on MI300-class
  GPUs; measure.

## Fusions in prefill

The decode fusions (`kernels-fusion.md`) have prefill counterparts with different trade-offs: RMSNorm can be applied
while staging the A tile (each row's inverse RMS computed in a small pre-pass), SiLU and the gate*up product in the
gate/up GEMM's epilogue (interleaved weight rows, `kernels-gemv-quant.md`), the residual add in the down and O
projection epilogues, QK-norm and RoPE in the QKV epilogue with the K/V cache write. Keep every bf16 rounding point
the reference has. Elementwise kernels at prefill sizes are bandwidth-bound: vectorize them (16-byte accesses) and
judge them against the streaming bandwidth.

## Chunking long prompts

- Scratch for one chunk: activations for M rows of the widest projection. Example (Qwen3-0.6B, estimate): gate+up
  output at 8192 rows is 8192 x 6144 x 2 B = 100 MB; the whole 32K prompt in one chunk is ~400 MB, small on a
  256 GB device. Fewer chunks mean fewer launches.
- If your partition is chunk-independent (above), choose chunk size for speed and memory only; otherwise treat it as
  a rounding route and check against the reference.
- One input buffer per in-flight chunk: never let the host rewrite a buffer that a queued kernel has not read yet
  (on Metal that bug made every token of a second chunk diverge, `backends/metal-rust/prefill.md`).

## Where to copy GEMM code from

| source | what | licence (check at the commit) |
|---|---|---|
| Composable Kernel (`ck_tile` GEMM, `DeviceGemm_Xdl_CShuffle` and successors) | MFMA GEMM pipelines and tile configs for gfx942 | MIT |
| aiter | tuned bf16 GEMMs and the shape-to-kernel tables SGLang may use | MIT |
| hipBLASLt / Tensile | the reference library's kernel configurations (logic YAML files list tile shapes per problem size) as reading material | MIT |
| Triton matmul tutorials and aiter's Triton GEMMs | tile and pipelining ideas only (port to HIP) | MIT |

## Not tried

Nothing on this platform yet. Candidates, in a reasonable order: a plain MFMA GEMM with a CK-derived 128 x 128 tile
(correctness and ceiling), chunk-independent partition with the whole prompt in one chunk, QKV epilogue with RoPE and
cache write, SiLU epilogue, XCD-aware block order, a flash-attention tile from CK or aiter (`attention.md`).
