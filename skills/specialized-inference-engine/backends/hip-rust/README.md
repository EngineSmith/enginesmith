# HIP + Rust backend (AMD Instinct MI325X, gfx942)

Purpose: the entry point for any engine whose host code is Rust and whose kernels are HIP C++ on an AMD Instinct
MI325X (gfx942, CDNA3), judged against stock SGLang at batch size 1. It lists the backend's files, the platform rules
that every change must keep, the platform numbers measured so far, and what is still unmeasured. Read it before the
first change to a HIP engine and whenever a number elsewhere looks off.

Status: this backend has no worked engine yet. The numbers below come from micro-benchmarks and from the reference
measured through the harness on one MI325X; everything else is either a vendor specification (labelled as such), an
estimate (with its arithmetic), or a hypothesis to measure. Mechanisms carried over from `backends/metal-rust/` are
ideas to test here, never facts: their costs were measured on a different GPU with a different memory system.

## Files

| file | read when |
|---|---|
| `README.md` | first; before any HIP change; when you need a platform number |
| `runtime.md` | HIP FFI from Rust, building and loading code objects, streams, memory, weight upload and relayout, error checking, gfx942 device-code specifics, build reproducibility |
| `decode-loop.md` | counting kernels per token, the launch-overhead budget, hipGraph capture, device-side argmax, avoiding host syncs, persistent kernels |
| `kernels-gemv-quant.md` | writing or tuning a decode GEMV (bf16 or a quantized format) at the bandwidth limit |
| `kernels-fusion.md` | merging kernels without changing bits: norms into GEMV prologues, QKV + RoPE, gate/up + SiLU, residual adds, whole layers, grid-wide sync, a decode megakernel |
| `attention.md` | decode attention (single-pass and split-KV), GQA grouping, RoPE and QK-norm, the KV cache layout, prefill flash attention with MFMA, where to copy kernels from |
| `prefill.md` | bf16 GEMM with MFMA intrinsics, tiling for gfx942, LDS double buffering, chunking long prompts, copying tile configs |
| `profiling.md` | rocprofv3 kernel traces and stats, rocprof-compute, hipEvent timing, attributing decode time per kernel, the GPU lock, noise |
| `serving.md` | the protocol loop on this platform: stdin JSON lines, resident model, reset and continuation, finishing GPU work before replying, stderr diagnostics |
| `counterexamples.md` | before trying an idea: measured negative and neutral results on this platform |

## Hard rules

1. **The forward pass runs on the GPU.** Every operator of prefill and decode (embedding gather, norms, projections,
   attention, the LM head, the argmax) runs in HIP kernels. CPU-only optimization is a dead end: decode is bounded
   below by bytes read per token divided by bandwidth. Example (Qwen3-0.6B bf16, estimate): ~1.19 GB of weights per
   token over ~4.4 TB/s of measured HBM bandwidth is ~0.27 ms per token on the GPU; the same bytes over a host's
   memory system (a few hundred GB/s at best, not measured) are several milliseconds, already slower than the
   reference's 1.7-1.8 ms. The host only parses requests, launches kernels and copies tokens.
2. **Toolchain: Rust (cargo) host, HIP C++ kernels for gfx942.**
   - Kernels are HIP C++ compiled with hipcc `--offload-arch=gfx942`. Preferred: at build time, `build.rs` invokes
     hipcc (`/opt/rocm/bin/hipcc` on the build host, or `$HIPCC`) with `--genco` to produce a code object, which the
     binary embeds with `include_bytes!` and loads at startup with `hipModuleLoadData`, then `hipModuleGetFunction`
     and `hipModuleLaunchKernel` (`runtime.md`). Alternative: compile embedded source at run time with hiprtc
     (`libhiprtc`; it needs no `PATH`), then load the result the same way.
   - Rust calls the HIP runtime through FFI to `libamdhip64`: hand-written `extern "C"` bindings, or bindgen at build
     time against the installed headers. No crate that wraps a numerical library.
   - Link only system libraries and the HIP runtime layer: `libamdhip64`, `libhsa-runtime64`, `libhiprtc`,
     `libamd_comgr` (and what they need). The harness's native check reads the binary's direct dependencies and
     refuses rocBLAS, hipBLAS / hipBLASLt, MIOpen, Composable Kernel, aiter, rocPRIM / hipCUB, Triton, torch and
     anything else outside its allow-list (`docs/libraries.md`; RCCL is allowed for now).
   - Open-source kernels from those projects may be **copied and adapted into the engine's source**: aiter and CK
     kernels, the HIP source of attention kernels, vLLM's ROCm kernels. Respect each licence (keep its notice, record
     the project, commit and file you copied from in the engine's provenance notes). Triton kernels cannot run (they
     need Python at run time); port their algorithm, tile shapes and ideas into HIP. Do not embed prebuilt code
     objects from other projects (some aiter kernels ship only as assembled `.co` files): they are not source you can
     adapt to this model.
3. **No network; a minimal run-time environment.** The engine runs with `PATH=/usr/bin:/bin:/usr/sbin:/sbin`,
   `HOME` and `TMPDIR` set to its scratch directory, which is also its working directory and has a long path. It must
   not use the network or write outside the scratch directory. Nothing at run time may depend on `PATH` finding
   hipcc, on `ROCM_PATH`, `LD_LIBRARY_PATH` or `HIP_*` variables being set, or on files outside the model directory
   and the system install. On the measured host the ROCm libraries resolve through the loader cache
   (`/etc/ld.so.conf.d` lists `/opt/rocm/lib`); check yours with `env -i PATH=/usr/bin:/bin ldd ./engine`.
4. **Numerics: bf16 storage, fp32 arithmetic, the reference's rounding points.**
   - gfx942 has bf16 matrix instructions that accumulate in fp32 (`v_mfma_f32_16x16x16bf16_1k`,
     `v_mfma_f32_32x32x8bf16_1k`, and smaller shapes); use them for prefill (`prefill.md`). Decode GEMVs are
     bandwidth-bound and use ordinary fp32 FMAs on widened bf16 (`kernels-gemv-quant.md`).
   - Reproduce what SGLang computes where it matters: fp32 accumulation in every dot product, RMSNorm statistics and
     scaling in fp32, the reference's bf16 rounding points between operators (each projection output, the residual
     stream, the normalized rows), the LM head accumulated in fp32 and the argmax taken over the row as the
     reference stores it (read the pinned `LogitsProcessor`: if its row is bf16 widened to fp32, many exact ties
     occur), greedy ties to the lowest token id. Correctness is judged by teacher forcing against SGLang
     (`references/correctness.md`); an exact tie counts as agreement.
   - **hipcc contracts across statements by default** (`-ffp-contract=fast-honor-pragmas` for HIP in clang; verify
     with your hipcc version): `a*b + c` may become an FMA in one kernel and not in its fused copy. Decide the
     contraction mode once at bring-up, pass it explicitly on the hipcc command line, write `fmaf` explicitly where
     the order matters, and never flip it on an existing engine. Do not use `-ffast-math`; set denormal handling
     explicitly (`-fgpu-flush-denormals-to-zero` or not) and keep it.
   - Determinism is structural: no float atomics in any reduction (copied split-K and attention kernels often use
     `atomicAdd` on floats: replace it with partials and a fixed-order combine); split counts and tile shapes are
     functions of the shape only. SGLang greedy was bit-identical across repeated runs at every length tested on
     this setup, so any run-to-run difference in your engine is your bug.
5. **The reference to beat is SGLang 0.5.22 on ROCm with its defaults**: aiter attention, hipGraph capture for
   decode, the overlap scheduler, chunked prefill. Its batch-1 decode for a small model is far above the bandwidth
   floor (table below: ~1.7 ms per token against a ~0.27 ms floor for Qwen3-0.6B), i.e. dominated by per-kernel
   launch cost and framework overhead, not by weight streaming. That is the opening: fewer kernels per token,
   kernels that stream near the measured bandwidth, and no host in the token loop.
6. **Wavefronts are 64 lanes.** Every reduction, shuffle, lane-to-element mapping and `__launch_bounds__` assumes 64;
   a kernel copied from CUDA with 32-lane assumptions is wrong, not slow (`runtime.md`).
7. **Refuse cleanly on another device.** At startup check the device: the gfx942 code object must load (a load
   failure is the refusal), the CU count and wavefront size match what the kernels assume. On any failure write
   one line to stderr and exit non-zero before replying to `hello`.
8. **No host paths; quiet, reproducible builds.** Pinned crates, `Cargo.lock` committed, offline `--locked` builds;
   no warnings from your tree; no build-machine path in the source or the binary (`runtime.md`, Build).

## Measured platform numbers

Measured on one AMD Instinct MI325X, ROCm 7.14/HIP 7.15 hipcc, 2026-11-02, unless marked as an estimate or a vendor
specification. Re-measure on your machine; the micro-benchmarks are single runs of simple kernels.

| quantity | value | how |
|---|---|---|
| HBM streaming read, 64 MB | 3635 GB/s | float4 streaming-read kernel, grid = 8 x CUs, 256 threads |
| HBM streaming read, 256 MB | 4796 GB/s | same kernel; may be partly served by the 256 MB Infinity Cache (hypothesis) |
| HBM streaming read, 1 GB | 4193 GB/s | same kernel |
| HBM streaming read, 4 GB | 4376 GB/s | same kernel; use ~4.2-4.4 TB/s for floors of multi-GB streams |
| fixed cost of one streaming kernel (estimate) | ~3 us | derived: 64 MB took ~17.6 us where 4.4 TB/s predicts 14.6 us |
| back-to-back launches of an empty kernel, one stream | 1.53 us per launch | host wall time |
| back-to-back launches of a 304-block tiny kernel | 1.87 us per launch | host wall time |
| one launch + `hipStreamSynchronize` round trip | 9.31 us | host wall time |
| hipGraph replay, 100-kernel graph | 1.72 us per kernel (171.7 us per graph) | host wall time |
| hipGraph replay, 400-kernel graph | 1.69 us per kernel (675.7 us per graph) | host wall time |
| SGLang greedy determinism | bit-identical across repeated runs at every length tested | through the harness |

**Graph replay did not reduce the per-kernel cost** on this machine, unlike typical CUDA experience: first in the
micro-benchmark above, then in a real engine (a whole captured decode step: 56.24 -> 56.20 ms per 64 tokens at 4K;
`counterexamples.md`). Kernel count itself is the cost. **What worked** was folding element-wise work into the
kernels that need it, with no inter-block synchronization (RMSNorm into the QKV and gate/up GEMVs, SiLU into the
down GEMV, RoPE into decode attention, head + argmax), and loading the next stage's weights early. **What did not**
was handing data between blocks inside one launch: three measured attempts (device-scope fence, agent-scope `sc1`
atomics, a producer / consumer launch) were all slower than the kernel boundary they removed. Treat persistent and
megakernel designs as unproven here until a measured grid barrier beats a launch boundary.

The reference: SGLang 0.5.22, Qwen3-0.6B bf16, batch 1, greedy, 64 decode tokens, measured through the harness on the
same machine and date:

| prompt tokens | 1 | 2 | 63 | 2048 | 4097 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode, ms per token | 1.83 | 1.69 | 1.72 | 1.80 | 1.87 | 2.09 | 2.69 | 3.33 |
| prefill request (prompt + 1 token), ms | ~14 | ~14 | ~14 | ~14 | 23 | 45 | 108-126 | 331-338 |
| decode floor, ms per token (estimate) | 0.27 | 0.27 | 0.27 | 0.32 | 0.38 | 0.48 | 0.70 | 1.13 |

Floor arithmetic (estimate): Qwen3-0.6B reads 880.9 MB of layer weights plus the tied 151936 x 1024 bf16 LM head
(311.2 MB) per decode token, ~1.19 GB; at ~4.4 TB/s that is ~0.27 ms. The checkpoint file is 1.50 GB because it
stores the tied matrix twice (`model.embed_tokens.weight` and `lm_head.weight`, byte-identical); decode reads only one
embedding row. The KV cache adds 28 layers x 8 KV heads x 128 x 2 (K, V) x 2 B = 114,688 B per position, 3.76 GB at
32,768 positions. So the reference is ~6x above the floor at short context and ~3x at 32K: at short context the gap
is launch and framework overhead; at long context decode attention becomes the main term. The ~14 ms prefill request
at short prompts is framework overhead too; an engine that runs a short prompt in well under a millisecond is
plausible (estimate, not measured).

## Vendor specifications (not measured on this machine)

| quantity | value |
|---|---|
| compute units | 304 (8 XCDs of 38 CUs) |
| wavefront size | 64 |
| LDS per CU | 64 KiB |
| L2 cache | 4 MB per XCD (not coherent across XCDs for ordinary memory; see `kernels-fusion.md`) |
| Infinity Cache (last-level, shared) | 256 MB |
| HBM3E | 256 GB, ~6 TB/s peak (measured streaming read above: ~4.2-4.8 TB/s) |
| dense bf16 matrix peak | ~1.3 PFLOP/s (measure your own GEMM ceiling, `prefill.md`) |

## Not measured yet (measure before relying on the mechanism)

- The marginal cost of one more kernel inside a real decode chain, by grid size (the launch rates above are host
  wall time of empty kernels, which do not separate host submission from the GPU's per-dispatch gap).
- The cost of a grid-wide barrier inside a persistent kernel, by grid size, with device-scope fences across XCDs (an
  in-launch handoff between 64 and 256 blocks already cost ~5 us per layer more than a kernel boundary;
  `counterexamples.md`).
- Bandwidth of a GEMV as a function of bytes per kernel (1-12 MB, the size of one projection of a small model).
- Effect of `HIP_FORCE_DEV_KERNARG=1` (kernel arguments in device memory) and of `hipDeviceScheduleSpin` versus
  blocking sync on launch and sync latency.
- The MFMA GEMM ceiling at prefill shapes, and decode/prefill timing noise in an A/A run (`references/measurement.md`).
- Which kernels SGLang runs per decode step, and how many (a rocprofv3 trace of the reference, if you can run it).
