# Runtime: HIP from Rust, code objects, memory, weights, gfx942 device code, build

Purpose: the host-side machinery every HIP + Rust engine needs, and the device-code facts of gfx942 that every kernel
depends on: FFI bindings to the HIP runtime, building and embedding code objects, module loading and kernel launch,
streams and events, device and pinned memory, error checking, loading safetensors weights and relaying them out once
at load, wave64 reductions, LDS, launch bounds and builtins, and the build settings for a quiet, reproducible binary.
Read it when you write or change any of these. The decode loop built on top is in `decode-loop.md`.

Nothing in this file has been measured in an engine on this platform yet; numbers come from `README.md` or are
labelled as estimates. API facts are from the HIP headers and documentation of ROCm 7.x: check them against the
headers installed on your build host.

## FFI to the HIP runtime

Two ways, both allowed (`README.md`, rule 2):

1. **Hand-written bindings** for the dozen functions you call. Take every type, enum value and struct layout from the
   installed `hip_runtime_api.h` of the pinned ROCm version; never guess an enum value. Avoid binding
   `hipDeviceProp_t` by hand (large, version-dependent; in ROCm 6+ `hipGetDeviceProperties` is a macro for a
   versioned symbol): query integers with `hipDeviceGetAttribute` instead.
2. **bindgen at build time** against the installed headers (`-D__HIP_PLATFORM_AMD__`, an allow-list of the functions
   and types you use), if bindgen and libclang are in your vendored, pinned crate set.

```rust
use std::ffi::{c_char, c_int, c_uint, c_void};
pub type HipError = c_int;                       // hipSuccess = 0
#[repr(transparent)] #[derive(Clone, Copy)] pub struct Stream(pub *mut c_void);
#[repr(transparent)] #[derive(Clone, Copy)] pub struct Module(pub *mut c_void);
#[repr(transparent)] #[derive(Clone, Copy)] pub struct Function(pub *mut c_void);
#[repr(transparent)] #[derive(Clone, Copy)] pub struct Event(pub *mut c_void);
#[link(name = "amdhip64")]
extern "C" {
    pub fn hipInit(flags: c_uint) -> HipError;
    pub fn hipSetDevice(id: c_int) -> HipError;
    pub fn hipDeviceGetAttribute(v: *mut c_int, attr: c_int, dev: c_int) -> HipError;  // attr values from the header
    pub fn hipGetErrorString(e: HipError) -> *const c_char;
    pub fn hipMalloc(p: *mut *mut c_void, bytes: usize) -> HipError;
    pub fn hipHostMalloc(p: *mut *mut c_void, bytes: usize, flags: c_uint) -> HipError;
    pub fn hipFree(p: *mut c_void) -> HipError;
    pub fn hipMemcpyAsync(dst: *mut c_void, src: *const c_void, n: usize, kind: c_int, s: Stream) -> HipError;
    pub fn hipMemsetAsync(dst: *mut c_void, value: c_int, n: usize, s: Stream) -> HipError;
    pub fn hipStreamCreateWithFlags(s: *mut Stream, flags: c_uint) -> HipError;
    pub fn hipStreamSynchronize(s: Stream) -> HipError;
    pub fn hipModuleLoadData(m: *mut Module, image: *const c_void) -> HipError;
    pub fn hipModuleGetFunction(f: *mut Function, m: Module, name: *const c_char) -> HipError;
    pub fn hipModuleLaunchKernel(f: Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint,
        bz: c_uint, shared_bytes: c_uint, s: Stream, params: *mut *mut c_void, extra: *mut *mut c_void) -> HipError;
    pub fn hipEventCreate(e: *mut Event) -> HipError;
    pub fn hipEventRecord(e: Event, s: Stream) -> HipError;
    pub fn hipEventSynchronize(e: Event) -> HipError;
    pub fn hipEventElapsedTime(ms: *mut f32, start: Event, stop: Event) -> HipError;
}
```

Wrap each call in a function that turns a non-zero code into an error carrying the call name and
`hipGetErrorString`. Keep the raw pointers inside a few owning types (`DeviceBuf`, `Module`, `Stream`) with `Drop`
implementations; nothing else in the engine touches FFI.

## Building and embedding code objects

```rust
// build.rs: compile kernels/*.hip into one code object per source; never at run time, never into the source tree
let hipcc = std::env::var("HIPCC").unwrap_or_else(|_| "/opt/rocm/bin/hipcc".into());
let out = std::path::PathBuf::from(std::env::var("OUT_DIR")?).join("decode.co");
let status = std::process::Command::new(&hipcc)
    .args(["--genco", "--offload-arch=gfx942", "-O3", "-ffp-contract=fast-honor-pragmas", // explicit, see README
           "-fno-gpu-rdc", "-o"]).arg(&out).arg("kernels/decode.hip").status()?;
assert!(status.success(), "hipcc failed");
println!("cargo:rerun-if-changed=kernels");
println!("cargo:rustc-link-lib=dylib=amdhip64");
```

```rust
// engine: the code object is bytes in the binary
static DECODE_CO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/decode.co"));
let mut m = Module(std::ptr::null_mut());
check("hipModuleLoadData", unsafe { hipModuleLoadData(&mut m, DECODE_CO.as_ptr().cast()) })?;   // fails on non-gfx942
```

- Declare every kernel `extern "C" __global__` so `hipModuleGetFunction` finds it by its plain name. Resolve every
  function the engine can launch at startup, before replying to `hello`; a missing symbol is a clean refusal.
- `hipModuleLoadData` accepts a code object or an offload bundle (verify which `--genco` produces with your hipcc;
  both load). It is the device check: on another architecture it fails.
- **Inspect what the compiler made.** `-Rpass-analysis=kernel-resource-usage` prints VGPRs, SGPRs, AGPRs, scratch
  (spills), LDS and occupancy per kernel at build time; `--save-temps` keeps the `.s` assembly. Read them for every
  hot kernel: scratch use means spills to memory, and the assembly shows whether loads are `global_load_dwordx4`,
  whether a shuffle became DPP or `ds_bpermute`, and where FMAs were contracted.
- **Run-time compilation (hiprtc)** is the alternative: `hiprtcCreateProgram`, `hiprtcCompileProgram` with
  `--offload-arch=gfx942`, `hiprtcGetCode`, then `hipModuleLoadData`. Its compile time counts in load time (not
  scored, not measured). Check that it works with `TMPDIR` set to a long scratch path and `HOME` set to scratch
  (hypothesis: the compiler support library may place temporary files there).
- Specialize by compile-time constants: templates instantiated per decode shape behind `extern "C"` wrappers
  (`gemv_bf16_k1024_n4096`), so trip counts, strides and row counts are constants. The Rust wrapper checks the shape
  and dtype before launching.

## Launching

```rust
// kernel params: an array of pointers to each argument value, in declaration order
let (mut w, mut x, mut y, mut n) = (wt.ptr(), xin.ptr(), yout.ptr(), rows as i32);
let mut args = [&mut w as *mut _ as *mut c_void, &mut x as *mut _ as *mut c_void,
                &mut y as *mut _ as *mut c_void, &mut n as *mut _ as *mut c_void];
check("launch gemv", unsafe { hipModuleLaunchKernel(f, grid, 1, 1, 256, 1, 1, 0, stream,
                                                   args.as_mut_ptr(), std::ptr::null_mut()) })?;
```

- Argument types must match the kernel's declaration exactly (an `i32` passed where the kernel takes `size_t` reads
  garbage); keep one Rust wrapper per kernel with typed parameters.
- **One non-default stream** created with `hipStreamNonBlocking`; the null stream synchronizes implicitly with other
  streams. Kernels on one stream run in order and each sees the previous one's writes.
- A launch returns before the kernel runs; execution errors surface at the next synchronizing call. A GPU memory
  fault usually aborts the whole process with a "Memory access fault" message from the runtime rather than returning
  an error, so find out-of-bounds bugs in a kernel harness, not in the protocol path. For debugging only:
  `HIP_LAUNCH_BLOCKING=1` serializes launches; `AMD_LOG_LEVEL=3` or 4 logs every API call (very heavy: short
  repros only).

## Memory

- **Device memory:** `hipMalloc` (coarse-grained VRAM). Allocate everything at load: weights, KV cache at the
  context the task needs, scratch for the largest prefill chunk, the token ring, counters. No allocation per request
  or per token. New allocations have undefined contents: `hipMemsetAsync` anything that must start at zero.
- **Pinned host memory:** `hipHostMalloc` for the buffers you copy through: the prompt ids, the returned tokens,
  upload staging. `hipMemcpyAsync` from pageable memory is staged by the runtime and is not truly asynchronous.
- **Alignment:** `hipMalloc` returns at least 256-byte aligned pointers (documented minimum; check). Views into a
  buffer must keep 16-byte alignment for `uint4` / `float4` loads; check alignment when a view is made.
- **Fine-grained or uncached memory** (`hipExtMallocWithFlags` with `hipDeviceMallocFinegrained` or
  `hipDeviceMallocUncached`) changes cache behaviour and coherence; consider it only for small cross-workgroup sync
  words and flags (`kernels-fusion.md`), and measure.
- **Memory budget:** the harness reports the VRAM budget; the reference's KV pool takes a large fraction of the GPU,
  but your engine's footprint is its own. Weights (example Qwen3-0.6B: 1.19 GB once, with the head and embedding
  shared) plus the KV cache (114,688 B per position) are small on a 256 GB device.

## Weights from safetensors

**Format.** A shard is an 8-byte little-endian header length `n`, `n` bytes of JSON
`{name: {dtype, shape, data_offsets: [start, end]}}` (plus `__metadata__`), then the data section at file offset
`8 + n`. Sharded checkpoints list each tensor's shard in `model.safetensors.index.json`.

**Check everything at load.** Byte length equals `numel x dtype size`; the set of names, shapes and dtypes is exactly
the one in the model card (refuse missing or unexpected tensors); tied weights are handled as the reference handles
them. Example (Qwen3-0.6B, read from the header and the bytes): `tie_word_embeddings` is true and the checkpoint
stores both `model.embed_tokens.weight` and `lm_head.weight`, byte-identical; load one copy and use it for both. Its
data section starts at offset 35,560 (8-byte but not 16-byte aligned in the file), so a vector load straight from a
mapped file would be misaligned; copying into `hipMalloc` buffers removes the problem.

**Upload.** Load time is recorded but not scored, so the simple path is fine: `mmap` each shard read-only (or read
it), copy each tensor through a pinned staging buffer with `hipMemcpyAsync` (double-buffer two staging chunks of
tens of MB if load time matters), synchronize once at the end. Never write a derived file next to the checkpoint or
anywhere outside scratch.

**Relayout at load, once, bit-exact.** Your kernels may want another layout than the checkpoint's: Q, K and V rows
concatenated into one matrix; gate and up rows interleaved so one wave reads both; rows permuted or tiled so each
lane group reads contiguous 16-byte chunks; a quantized format repacked into the order a kernel consumes it. Do it
once after upload (a relayout kernel, or on the host before upload), keep every value bit-identical, and record the
cost in load time and memory. Never repack, requantize or dequantize per launch. A layout change that changes the
order in which your kernel sums a row is an arithmetic change (check it against the reference); a layout change
that only moves bytes is free of numerics.

## Device code on gfx942

**Wave64.** A wavefront is 64 lanes. `warpSize` is 64; write `constexpr int WAVE = 64;` and
`static_assert` it under `__gfx942__` rather than leaving 32 anywhere. Lane masks are 64-bit (`__ballot` returns
`unsigned long long`). A 256-thread block is 4 waves; a CU runs 4 SIMDs.

**Reductions.** The portable form is a butterfly with `__shfl_xor` (its default width is the whole wave, 64):

```cpp
__device__ inline float wave_sum(float v) {          // all 64 lanes end with the same bits
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off);
    return v;
}
__device__ inline float group16_sum(float v) {        // independent sums in each 16-lane group
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) v += __shfl_xor(v, off, 16);
    return v;
}
```

A butterfly gives every lane the same bits (each step adds the same two values in either order, and float addition
is commutative), so any lane may store. `__shfl_xor` may compile to `ds_bpermute_b32` (through the LDS hardware) or
to DPP / `ds_swizzle` moves; check the assembly. Faster forms exist as builtins (`__builtin_amdgcn_mov_dpp`,
`__builtin_amdgcn_update_dpp`, `__builtin_amdgcn_ds_swizzle`, `__builtin_amdgcn_readlane`), with DPP row operations
inside 16-lane rows and row broadcasts across them. Switching mechanism is bitwise-safe only if the pairing tree is
the same; a different tree (for example a tree of row shifts instead of a butterfly) changes rounding. Measure
before hand-writing DPP: the shuffle cost matters only in kernels that reduce often (attention), rarely in a GEMV
that reduces once per row.

**LDS.** 64 KiB per CU, at most 64 KiB per block (vendor spec; `hipDeviceAttributeMaxSharedMemoryPerBlock` reports
it). Static `__shared__` or dynamic (the launch's `shared_bytes`). Banks are 4 bytes wide (32 banks, vendor
documentation); pad tiles whose rows are a multiple of 128 bytes. LDS use limits occupancy: two blocks of 32 KiB fill
a CU.

**Launch bounds and occupancy.** Without `__launch_bounds__`, the compiler assumes up to 1024 threads per block and
limits registers accordingly. Give every kernel `__launch_bounds__(THREADS)` with its real block size. In HIP the
optional second argument is the minimum number of **waves per execution unit** (not CUDA's blocks per SM);
`__attribute__((amdgpu_waves_per_eu(min, max)))` is the direct form. Registers per lane and spills are in the
kernel-resource-usage remark; occupancy per SIMD is bounded by VGPRs (512 per SIMD lane shared with AGPRs, vendor
documentation), LDS and the block size. `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor` gives the resident
blocks per CU for a launch configuration, which a persistent kernel needs (`kernels-fusion.md`).

**Loads.** Use 16-byte vector loads (`uint4`, `float4`) for every weight stream: one `global_load_dwordx4` per lane,
1 KiB per wave per instruction. Mark weight pointers `const __restrict__`. Non-temporal hints
(`__builtin_nontemporal_load`) change cache allocation; whether they help on gfx942 is not measured
(`kernels-gemv-quant.md`).

**bf16.** A bf16 value is the upper 16 bits of an fp32. Widening is a shift; rounding fp32 to bf16 needs
round-to-nearest-even and NaN handling (gfx942 has no single conversion instruction for it, as far as the ISA
documentation shows; check the assembly). Either use `hip/hip_bf16.h` (header-only, part of HIP) or own helpers:

```cpp
__device__ inline float bf16_lo(uint32_t u) { return __uint_as_float(u << 16); }          // element 0 of a pair
__device__ inline float bf16_hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }  // element 1
__device__ inline uint16_t f32_to_bf16(float f) {                                           // RNE, NaN stays NaN
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
```

Use one helper everywhere, so every rounding point of the engine rounds the same way as PyTorch's (RNE).

**Transcendentals.** `expf`, `rsqrtf`, `__expf`, `__frsqrt_rn` and the `__builtin_amdgcn_*` hardware forms differ in
accuracy; pick one per operator at bring-up to match the reference closely (it is not bitwise reproducible against
PyTorch kernels anyway), then keep it.

## Process start and shutdown

- **Refuse cleanly at startup:** load the code objects (device check), resolve every function, check CU count and
  wavefront size with `hipDeviceGetAttribute`, load and check all weights, allocate everything, run one warm-up
  forward if it removes first-request costs, and only then reply to `hello`. On failure write one bounded line to
  stderr and exit non-zero.
- **Read stdin on its own thread** and set an end-of-input flag when it closes; the main loop finishes in-flight
  work, synchronizes the stream and exits 0 (`serving.md`).
- One process, one device (the first visible one), one stream for the forward pass. Do not fork.

## Build settings

```toml
[profile.release]
codegen-units = 1     # one codegen unit: the same bytes on every build
lto = false           # check reproducibility before enabling LTO
```

- Pin every crate exactly and commit `Cargo.lock`; build `--offline --locked` from a vendored crate set.
- `build.rs` uses only relative paths inside the crate, `OUT_DIR` and the hipcc location; it never writes into the
  source tree. Do not pass `-g` to hipcc unless you remap paths (`-ffile-prefix-map=<dir>=.`): debug info records
  source paths.
- **Remap Rust paths** with `--remap-path-prefix` for the build directory, `HOME`, the crate set and the toolchain.
  Then scan the binary and the embedded code objects for your home, scratch, crate-set and toolchain paths
  (`strings engine | grep -e /home -e /tmp`).
- **Verify reproducibility:** build the same tree in two directories of different path lengths and compare SHA-256
  of the binary. hipcc output is expected to be deterministic for the same inputs (not verified here; check it).
- **Check the dependencies:** `readelf -d target/release/engine | grep NEEDED` must list only system libraries and the
  HIP runtime layer (`README.md`, rule 2); this is what the harness's native check reads.
- **Quiet:** zero warnings from Rust and from hipcc; fix each or allow it narrowly with a reason.
