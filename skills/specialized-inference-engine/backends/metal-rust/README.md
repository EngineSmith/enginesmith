# Metal + Rust backend (Apple Silicon)

Purpose: the entry point for any engine whose kernels are Metal Shading Language (MSL) and whose host code is Rust on
Apple Silicon (measured on an M5 Max). It lists the backend's files, the platform rules that every change must keep,
and the measured platform numbers that the other files and your floors rely on. Read it before the first change to a
Metal engine and whenever a number elsewhere looks off.

Numbers are measurements from the LFM2.5 worked engine (see `examples/lfm2.5/`) on an Apple M5 Max unless they are
marked as estimates. Where a number is derived from measured ones, it says so; where nothing was measured, it says
"not measured".

## Files

| file | read when |
|---|---|
| `README.md` | first; before any Metal change; when you need a platform number |
| `runtime.md` | loading weights, tensors and views, pipelines, command buffers and encoders, hazards, scratch lifetime, process start and shutdown, build settings |
| `decode-loop.md` | changing how decode steps are encoded, batched, submitted, stopped or streamed; counting the dispatch budget |
| `kernels-gemv-quant.md` | writing or tuning a decode GEMV over quantized weights, including expert (MoE) gather GEMVs and the F32 router |
| `kernels-fusion.md` | merging dispatches: norms into GEMV prologues, top-k into the router, epilogues, the token tail; what keeps a fusion bitwise identical |
| `attention.md` | decode and prefill attention, the KV cache layout, the single-pass vs two-pass decode switch |
| `prefill.md` | prefill chunks, matrix-unit (tensor op) GEMMs, grouped expert GEMMs, the reference's chunk partition |
| `profiling.md` | measuring GPU time per dispatch inside the real chain; counter sampling; micro-benchmarks that do not mislead |
| `serving.md` | streaming output, stop conditions, prefix reuse, chat or HTTP front ends over the decode loop |
| `counterexamples.md` | before trying an idea: measured failures and the mechanism behind each |

## Hard rules

1. **Pure Rust host, MSL kernels.** Host code uses `objc2`, `objc2-foundation` and `objc2-metal` (pinned exact
   versions). No C, C++ or Objective-C sources, no build script that compiles anything, no prebuilt `.metallib`, no
   interpreter at run time. Third-party kernel code (for example MLX's Metal headers) may be copied and adapted into
   your MSL with its license and a provenance record.
2. **MSL is embedded and compiled at startup.** Embed each source with `include_str!("kernels/x.metal")` (a path
   relative to the including file, never absolute, never built from `env!`). Compile with
   `newLibraryWithSource:options:error:`, set the language version explicitly, compile every pipeline the engine can
   dispatch before reporting ready, and treat any compile failure as a clean refusal (`runtime.md`).
3. **Fast math is on by default, and it decides your bits.** `MTLCompileOptions` enables fast math unless you turn it
   off; the worked engine sets only the language version, so all its kernels compile with it. The compiler may then
   contract `a*b + c` into `fma`, reassociate sums, use fast transcendentals (`exp`, `rsqrt`, `pow`, `sin`, `cos`) and
   assume no NaN or infinity. Measured consequences (worked engine):
   - Unrolling a float sum with a fixed trip count changed bits: the 6-bit head with compile-time K (388 of 2.56 M
     outputs differ) and a fixed four-term expert combine (78,517 of 327,680).
   - A copied rotation `lo*c - hi*s` compiled to a multiply plus a fused multiply-add in one kernel and differently in
     the fused copy (1 bf16 ulp on 2 of 409,600 values); writing the `fma` explicitly fixed it.
   - Computing a norm gain earlier changed the GPU backend's association although the IR before the backend was
     identical (26 of 40 prompts differed).

   So: copy expressions and statement order verbatim; write `fma` explicitly where the parent's code was contracted;
   keep runtime loop bounds on float reductions that the parent had; never trust source or IR equality, only GPU
   output equality. Do not depend on `isnan`/`isinf` or NaN propagation; make NaN and tie handling structural
   (comparisons with NaN are false). Decide fast math on or off once, at bring-up, by measuring parity with the
   reference, and never flip it on an existing engine: that changes every kernel's rounding at once. Whether the
   reference framework compiles with fast math is not verified here (INFERENCE that MLX turns it off).
4. **Respect MSL version gates.** Compile each source at the lowest version it needs and guard newer features:

   | feature | needs | note |
   |---|---|---|
   | `bfloat` | MSL 3.1 (macOS 14) | baseline of every kernel in the worked engine |
   | `atomic_thread_fence(mem_device, seq_cst, thread_scope_device)` | MSL 3.2 | 3.1 has relaxed atomics only, so a last-threadgroup epilogue cannot be made safe (the worked engine tried and stopped). The worked engine compiles that kernel at 4.0 because its version enum has no 3.2, inside `#if __METAL_VERSION__ >= 320` in a file also compiled at 3.1 |
   | tensor ops / MetalPerformancePrimitives (matrix units) | MSL 4.0 (macOS 26) | probe 4.0, then 4.1, at startup; keep a fallback |
   | MSL 4.1 | macOS 27 | `objc2-metal` 0.3.2 has no named constant: `MTLLanguageVersion((4 << 16) + 1)` |
   | 64-bit `atomic_max` on `atomic_ulong` | compiled at 3.1 on the M5 Max | used by the head's argmax epilogue; not checked on other GPU families |

5. **Matrix units need GPU family 10, and a plain kernel stays beside them.** Family 10 (M5) runs MSL 4.0 tensor ops
   on native matrix units; the worked engine's notes say family 9 (M3/M4) runs them through emulation. Check
   `supportsFamily(MTLGPUFamilyApple10)` at startup. Use matrix units for prefill-sized work (`prefill.md`); decode
   GEMVs are bandwidth-bound and do not use them. Keep a plain (non-tensor-op) kernel for every tensor-op kernel: for
   bring-up, as a numerical check, and as the fallback route. A fallback rounds differently, so it is a separately
   verified route, chosen by device family only, never a silent substitute. If you do not verify one, refuse on that
   device with one stderr line and a non-zero exit (the worked engine refuses family < 10).
6. **Determinism is structural.** No float atomics in any reduction. Integer atomics only where the result is
   order-independent (a max over a total-order key, a completion counter). Split counts, tile shapes and kernel choices
   are functions of the shape and the device family, never of timing or occupancy.
7. **No host paths; quiet, reproducible builds.** Offline, `--locked` builds of pinned crates; no warnings; no
   absolute path of the build machine in the source tree or the binary; the same tree builds the same bytes in any
   directory (`runtime.md`, Build).

## Measured platform numbers (Apple M5 Max, 128 GB, macOS 26)

Use these for floors and sanity checks; remeasure on a different machine. "In chain" means inside the real decode
step, with real weights and state; "isolated" means a micro-benchmark.

| quantity | value |
|---|---|
| Streaming read ceiling, large stream | ~586-590 GB/s: the 213 MB 6-bit head GEMV runs at 586 GB/s in chain; decode GEMVs reach ~590 GB/s in steady state after compile-time shapes |
| Same, from DRAM, per 7.08 MB dispatch | 13.4-15.1 us = 470-530 GB/s including the fixed dispatch cost (working set >= 71 MB) |
| System-level cache (SLC), effective | ~48-56 MB; a 7.08 MB stream cycling within it runs 7.9-8.8 us (~800-900 GB/s) |
| Empty dispatch, isolated (simd_sum + store) | 1.89 us (512 TGs), 1.95 us (2048 TGs), 6.08 us (8192 TGs), 22.5 us (32768 TGs) |
| Marginal cost of one more dispatch, in chain | 0.86 us (1 TG x 32), 1.95 us (2048 TGs x 32) |
| One command buffer per dispatch | 13.6-14.4 us per empty dispatch (vs 1.86 in one encoder) |
| One encoder per dispatch, unsampled | free: 1.85 vs 1.86 us empty, 13.97 vs 13.99 us streaming |
| Metal 4 encoder, barrier after every dispatch | 1.85 us empty, 14.64 us streaming (no better than the legacy serial encoder); without barriers 1.41 / 11.89 us, which only independent dispatches can use |
| Concurrent encoder + barrier after every dispatch, whole decode | 2.3-2.7% slower than one serial encoder |
| Device-scope fence + atomic counter, last-threadgroup epilogue | ~8 us extra per dispatch at 2048 TGs; a net 0.46 us win per call at 64 TGs |
| Decode GEMVs in chain | 425-586 GB/s by kernel; bandwidth tracks bytes per dispatch (table in `kernels-gemv-quant.md`) |
| GPU busy during a decode request | 99.3% |
| Host encode of one decode step (~250 dispatches) | ~85-110 us |
| Commit to GPU start of the first command buffer of a call | ~0.23-0.39 ms; wake-up after the last one ~0.1 ms |
| GPU timestamps | 24 MHz (41.7 ns); counter sampling only at stage boundaries |
| Largest buffer; recommended working set | ~80.6 GiB; ~107.5 GiB (observed on the same machine) |
| Load time with all weights copied into Metal buffers | 1.87 s for 13.6 GB, warm file cache, process start to one token |
| Decode timing noise, interleaved A/B, quiet machine | about +-0.3%; thermal throttling under sustained load moves absolute numbers by several percent |

Example (LFM2.5, M5 Max): with these numbers the decode step of a 24B-total, 2B-active MoE reads ~1.46 GB per token
(floor ~2.5 ms at 590 GB/s, estimate) and the profiled build measured 3.28 ms/token at a 256-token context, of which
~0.25-0.5 ms is fixed per-dispatch cost (derived from the marginal costs above and its ~262 dispatches per token).
