# Rust + CUDA runtime foundation

This small native executable demonstrates a Rust owner around a C ABI implemented with the CUDA runtime and a
kernel compiled from the included CUDA source. It allocates one device buffer, owns a nonblocking stream and
two events, captures an increment kernel into a CUDA graph, verifies direct launches and graph replay, and
destroys graph/stream/buffer resources in dependency order. It does not implement a model or the engine protocol.

Build on a Linux CUDA machine with `nvcc`, a C++ compiler, `ar`, Python 3 and Rust 1.82+ available on PATH:

```sh
python3 examples/cuda-rust-runtime/build.py --arch sm_103 --out build/cuda-example
build/cuda-example/cuda-runtime-check 1000
python3 examples/cuda-rust-runtime/validate.py --build build/cuda-example
```

Select an architecture supported by the actual device and toolkit; `sm_103` here is an example target, not
automatic device detection. Dynamic CUDA library discovery is the caller's responsibility. Use `CUDA_LIBRARY_DIR`
for a nonstandard toolkit library location. The example needs no Cargo packages, downloads or precompiled model
math libraries. `build.json` records compiler versions, architecture and source/binary hashes without host paths.
The source compiles without fast math and disables FMA contraction explicitly; choose and validate numerics
per kernel in a real model rather than assuming one compiler flag establishes model parity.

The runtime produces one JSON line with correctness and direct/graph event and host-wall timings. Its 4096
elements start at zero for each run and are incremented exactly once per launch. Positive repetition counts are
bounded by the exact-integer range used by the float check. Event timing covers the launch stream; host wall
time includes the synchronization, but not allocation, graph capture or the final host readback. Both paths are
warmed up. The fixed direct-then-graph order is a smoke diagnostic, not an unbiased benchmark or a performance
claim: use randomized/paired repeats, the full kernel chain, and profiler evidence before selecting an optimization.

`Runtime` is a unique owner and exposes mutating operations through `&mut self`. Its opaque pointer cannot be
shared across threads by this API. All asynchronous work is synchronized before readback and destruction. Failed
CUDA calls become Rust errors; a partial initialization tears down previously created objects. The allocation
survives every graph replay. A real decode graph also needs stable metadata/KV pointer lifetimes and explicit
handling of acceptance, rewind, cancellation and changing batch shapes.

This is an initial build/runtime foundation tracked in [#29](https://github.com/EngineSmith/enginesmith/issues/29).
It does not register a complete CUDA skill backend. Quantized GEMV, attention, prefill, model correctness and
platform performance tables still need implementation and measured evidence. A successful compile is distinct
from execution on a GPU; neither establishes a specialized inference engine or an improvement over a baseline.

Primary API reference: [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/), particularly stream/event,
memory and graph APIs. C declarations are compiled against the installed toolkit headers rather than handwritten
CUDA types in Rust.

## GPU regression checks

`validate.py` is opt-in: it uses the caller's visible GPU and requires the same source files as the build receipt.
It parses the executable's raw JSON, checks nine direct/graph runs (1, 17 and 1000 repetitions, three each), rejects
three invalid CLI inputs, and checks that hiding all GPUs reports initialization failure as a Rust error. It then
compiles `lifecycle.cu` separately and writes source/binary hashes plus results to `validation.json` in the build
directory. It does not alter the normal binary, judging rules or model workloads.

The lifecycle test uses 1, 255, 256, 257 and 4096 elements across 40 create/run/destroy cycles and 240 exact-output
checks. Test-only wrappers inject failures at six setup calls (stream, two events, allocation, capture start and
graph instantiation) and require all acquired resources to be released. These are controlled returned errors, not
real OOM or device faults. Invalid C ABI arguments must leave an otherwise usable runtime intact; the repetition
bound applies at the ABI as well as the CLI. Capture invalidation, asynchronous device faults and cross-thread
ownership are outside these tests.

For a separate memory check, with NVIDIA Compute Sanitizer installed:

```sh
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 build/cuda-example/cuda-runtime-check 100
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 build/cuda-example/cuda-lifecycle-check
```

On 2026-11-21, these checks passed on one NVIDIA B300 (compute capability 10.3), Linux x86-64, driver 595.71.05,
CUDA compiler 13.4.92, Rust 1.82.0 and GCC 13.3.0, targeting `sm_103`. Compute Sanitizer 2026.2.0.0 reported zero
errors and zero leaked bytes for both executables. `cuobjdump --list-elf` confirmed an `sm_103` cubin; ELF direct
dependencies were the CUDA runtime and system C/C++ runtime libraries. The sanitized receipt is
[`validation-b300.json`](validation-b300.json); raw logs and machine identifiers are intentionally excluded.

Other resident services held GPU memory and were idle before/after this small check. No exclusive-device timing,
model parity, inference throughput, concurrency or agentic-workload performance is established by these results.
The issue's full CUDA backend and model-kernel work remains separate from this runtime foundation.
