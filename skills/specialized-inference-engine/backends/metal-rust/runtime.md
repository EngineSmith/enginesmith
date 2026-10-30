# Runtime: weights, buffers, pipelines, submission, lifetimes, build

Purpose: the host-side machinery every Metal + Rust engine needs, and the mistakes that make it wrong or slow:
loading safetensors weights into Metal buffers, tensors as buffer views, pipeline compilation and caching, command
buffers and encoders, host/GPU hazards, scratch and state lifetimes, process start and shutdown, and the build
settings that give a quiet, byte-reproducible binary. Read it when you write or change any of these. The decode
loop built on top of this is in `decode-loop.md`.

## Weights from safetensors

**Format.** A shard is an 8-byte little-endian header length `n`, then `n` bytes of JSON
`{name: {dtype, shape, data_offsets: [start, end]}}` plus an optional `__metadata__` object, then the data section.
Offsets are relative to the data section, which starts at file offset `8 + n`. A sharded checkpoint has
`model.safetensors.index.json` whose `weight_map` names each tensor's shard.

```rust
let n = u64::from_le_bytes(first8) as usize;
let header: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&bytes[8..8 + n])?;
for (name, v) in header {
    if name == "__metadata__" { continue; }
    let e: Entry = serde_json::from_value(v)?;              // dtype, shape, data_offsets
    ensure!(e.data_offsets[1] >= e.data_offsets[0], "{name}: reversed offsets");
    let file_start = 8 + n as u64 + e.data_offsets[0];      // absolute offset in the shard
    // record (shard, file_start, len, dtype, shape)
}
```

**Check everything at load.** Every tensor's byte length equals `numel * dtype size`; no tensor appears in two
shards; the set of names, shapes and dtypes is exactly the one the model card lists (refuse missing and unexpected
tensors; do not skip unknown ones silently); quantized tensors have consistent codes, scales and biases (group count,
bits implied by the code width). A shape error caught here is one clear message; caught in a kernel it is wrong
numbers far from the cause.

**Alignment.** The data section starts at `8 + n`, and `n` can be anything, so tensor file offsets are only as aligned
as the header length makes them. Example (LFM2.5): the three shards' data sections start at 43,049 / 40,341 / 21,953,
so every tensor sits at an odd file offset, aligned to 4 bytes only relative to its data section.
Metal's alignment rules then decide what you can do:
- `setBuffer:offset:atIndex:` offsets must suit the data type the kernel reads, at least 4 bytes (the worked engine
  checks 4 on every view); 16-byte vector loads (`uint4`, `bfloat4`, `float4`) need 16-byte aligned addresses.
- A no-copy buffer over mapped memory (`newBufferWithBytesNoCopy`) needs a page-aligned address (16 KiB pages on Apple
  Silicon; query the page size, do not hard-code it). Apple's documentation also asks for a page-multiple length
  (INFERENCE from the API documentation; verify on your SDK).

**Choose one of two placements.**
1. **Copy each tensor into its own shared buffer at load** (the worked engine). Allocate
   `newBufferWithLength(len, StorageModeShared)`, read the tensor's bytes straight into `contents()`. Every tensor then
   starts at offset 0 of an aligned buffer. Measured cost (LFM2.5, 13.6 GB): 1.87 s from process start to the first
   token with a warm file cache. The weights are resident; no page faults
   in the timed path.
2. **Map the shards read-only and wrap page-aligned ranges** in no-copy buffers, binding each tensor as
   `(range buffer, file_start - range_start)`. Where that offset breaks the binding or vector-load alignment (odd
   offsets), pass the element offset to the kernel and load unaligned, or fall back to copying that tensor. Pages fault
   in on first GPU touch, which moves cost into the first request: measure load time and first-request time
   separately. Not measured in the worked engine.

Never repack, requantize, pad or dequantize weights per launch, and never write a derived file next to the
checkpoint. A one-time in-memory relayout at load is allowed only if every dequantized value stays bit-identical; pay
for it in load time and memory and record both. A single tensor fits one buffer easily (largest buffer ~80.6 GiB on
the M5 Max).

## Tensors and views

A tensor is a buffer plus shape, dtype and a byte offset. Views share storage; cloning a tensor clones the handle,
not the data.

```rust
pub struct Tensor { buf: Retained<ProtocolObject<dyn MTLBuffer>>, shape: Vec<usize>, dtype: DType, offset: usize }
// view(start, shape): offset = self.offset + start * dtype.size(); ensure!(offset % 4 == 0);
// binding() -> (&buffer, offset): every kernel binding passes the real offset.
```

Check alignment when a view is made, not when a kernel misreads it: 4 bytes for any binding (so a bf16 view starts at
an even element), 16 bytes for views a vector kernel reads, 8 bytes for 64-bit atomics. Wrappers check shapes, dtypes
and alignment with `ensure!` before encoding anything, so an error never leaves an encoder half-written.

New Metal buffers have undefined contents; zero-fill on allocation if any code relies on zeros.

## Pipelines

Compile once, at startup, every pipeline the engine can dispatch. Cache libraries by source identity and language
version (one compile per embedded source, not per function), and pipelines by function name, language version and
function-constant values:

```rust
libraries: HashMap<(usize /*src ptr*/, usize /*src len*/, MslVersion), Library>,
pipelines: HashMap<(&'static str, MslVersion, u64 /*function constants*/), Pipeline>,
```

Name-only keys are enough for sources embedded with `include_str!`; if any source can change inside one process (a
test that edits kernels), the key must include the source's identity, or a stale pipeline is reused. Read
`threadExecutionWidth` from the pipeline before relying on 32-wide SIMD groups, and `maxTotalThreadsPerThreadgroup`
before choosing a threadgroup size. Specialize by compile-time constants (templates with `[[host_name]]`
instantiations, or `[[function_constant]]`) rather than runtime branches on shape (`kernels-gemv-quant.md`). Startup
compile time of the worked engine's sources: not measured separately (inside the 1.87 s load).

## Command buffers and encoders

- **One serial compute encoder per command buffer** for decode and for each prefill chunk. Dispatches on a serial
  encoder execute in order, and buffer writes of one are visible to the next. A decode step's dispatches are almost all
  read-after-write dependent (example LFM2.5: the only independent pair was a dense gate/up), so there is nothing for
  concurrency to overlap.
- **Concurrent encoders need explicit barriers** (`memoryBarrierWithScope(Buffers)`) at every dependency, plus your own
  write-after-read and write-after-write analysis on reused buffers. Measured on the decode chain with a barrier after
  every dispatch: 2.3-2.7% slower. A Metal 4 encoder with a barrier per dispatch measured
  the same as the serial encoder. Use concurrency only for proven independent work.
- **Never one command buffer per dispatch** (13.6-14.4 us per empty dispatch vs 1.86 us).
- **Keep the host encode cheap and allocation-free.** Bind buffers with offsets, pass small parameters with
  `setBytes`, validate parameters into a fixed-size stack array rather than a `Vec`. The worked engine encodes a
  ~250-dispatch step in ~85-110 us, far below the GPU's ~3.3 ms, so host cost is hidden as long as steps are encoded
  ahead (`decode-loop.md`).
- **Commit without waiting; wait once.** `commit()` returns a pending handle; the host waits on the last one, then
  checks `status()` of every command buffer it relies on (`Error` means the outputs and any in-kernel counters are
  undefined: fail the request and reset that state).

```rust
fn encode(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, k: &Kernel, bufs: &[(&Buf, usize)], params: &[&[u8]], tg: (usize, usize)) {
    enc.setComputePipelineState(&k.pipeline);
    for (i, (b, off)) in bufs.iter().enumerate() { unsafe { enc.setBuffer_offset_atIndex(Some(b), *off, i) } }
    for (i, p) in params.iter().enumerate() { unsafe { enc.setBytes_length_atIndex(ptr(p), p.len(), bufs.len() + i) } }
    enc.dispatchThreadgroups_threadsPerThreadgroup(size(tg.0), size(tg.1));
}
```

## Hazards

- **Unified memory does not order host and GPU accesses.** The host must not write a buffer that a committed,
  unfinished command buffer reads or writes, and must not read results before the wait returns. Write host inputs
  (prompt ids, the pending token) before committing the first command buffer that reads them; read outputs (tokens,
  logits) after the final wait.
- **Threadgroup barriers do not order threadgroups.** Within one dispatch, a threadgroup must not write data another
  threadgroup of the same dispatch reads. Write updated values to a second buffer (ping-pong) instead of in place.
  Example (LFM2.5): the fused router's 64 threadgroups all read the residual and threadgroup 0 writes the updated
  residual, so the residual alternates between two buffers.
- **Cross-threadgroup handoff inside a dispatch needs device-scope fences** (MSL >= 3.2) and must never make one
  threadgroup wait for another (no spin on a flag: forward progress would depend on co-residency). The last-finishing
  pattern is in `kernels-fusion.md`.
- **State that kernels reset must start in the reset state.** Completion counters and argmax keys are zero before the
  first dispatch (zero-filled at allocation) and reset by the kernel that consumes them; after a failed command buffer,
  re-zero them on the host while the GPU is idle.
- **Shorter calls must not read longer calls' leftovers** (next section).

## Scratch, state and lifetimes

- **Allocate all scratch once**, at load, sized for the largest chunk the engine accepts; no allocation per operator or
  per token. Views over scratch are cut to the live rows of the current call.
- **Every read follows a write of the same live extent in the same call.** A buffer sized for an 8192-row chunk still
  holds rows of the previous long call when a 3-row call runs; a kernel that reads past the live rows reads stale data
  and is correct by accident until it is not. Test long -> short -> long sequences and compare with fresh processes.
- **Carried state** (KV cache, recurrent or convolution state) lives outside scratch, is zeroed on reset, and is
  advanced only by code that also knows how to rewind it (`decode-loop.md`). KV rows beyond the current position may be
  stale: they must never be read and are overwritten before they become live.
- **Parallel in-place history updates race.** A prefill kernel that reads a convolution history while other
  threadgroups write it needs two history buffers and a slot flip.
- **Isolation checks:** a split request equals one long request; a reset followed by the same prompt gives the same
  tokens; a request after a different request gives the same tokens as in a fresh process.

## Process start, stdin and shutdown

- **Refuse cleanly at startup.** Check the device (GPU family, required MSL version by compiling a probe), compile all
  pipelines, load and check all weights, and only then report ready. On any failure write one short line to stderr
  (cause plus a bounded excerpt of the compiler message) and exit non-zero. Never report ready and then fall back to
  another numeric path.
- **stdout carries protocol replies only;** diagnostics go to stderr, bounded in size.
- **Read stdin on its own thread.** It forwards lines to the main loop and sets an end-of-input flag when stdin closes.
  The decode loop checks the flag between steps and stops encoding; the main loop then waits for every committed
  command buffer and exits 0. An engine that ignores end of input keeps gigabytes of weights resident after its parent
  is gone.
- **Do not daemonize or fork;** one process, one Metal device, one command queue.

## Build settings

```toml
[profile.release]
lto = false          # thin LTO made LC_UUID and the code signature depend on the build directory's path length
codegen-units = 1    # one codegen unit: same bytes on every build
# leave debug and strip at the release defaults: debug info records build paths
```

- **Pin every crate exactly** (`objc2 = "=0.6.4"`) and commit `Cargo.lock`; build `--offline --locked` from a vendored
  crate set. No build script, no `.cargo/config` in the tree, no `env!` path as data or include path, no absolute
  `include_str!`.
- **Remap paths** so the binary names no host path: pass `--remap-path-prefix=<dir>=<placeholder>` for the build
  directory, `HOME`, the working directory, the source tree, the target directory, the crate set and the toolchain
  root, and the linker's `-oso_prefix <toolchain root>` so the Mach-O UUID does not depend on where the toolchain is.
- **Verify reproducibility:** build the same tree in two directories with different path lengths and compare SHA-256
  (the worked engine: identical at four paths, `DESIGN.md`; its fixed build recipe rebuilt a parent binary that had
  been built on another path to the same SHA-256). Then scan the binary and every source file for your home, scratch,
  crate-set and toolchain paths.
- **Quiet:** zero warnings from your tree; fix each one, or allow it narrowly at the item with a reason.
- The resulting binary links only libraries under `/usr/lib` and `/System/Library` (Metal, Foundation, CoreFoundation,
  libobjc, libSystem, libiconv) and has no `LC_RPATH` (worked engine, `DESIGN.md`).
