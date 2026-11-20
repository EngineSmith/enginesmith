# Rust + CUDA runtime foundation

This small native executable demonstrates a Rust owner around a C ABI implemented with the CUDA runtime and a
kernel compiled from the included CUDA source. It allocates one device buffer, owns a nonblocking stream and
two events, captures an increment kernel into a CUDA graph, verifies direct launches and graph replay, and
destroys graph/stream/buffer resources in dependency order. It does not implement a model or the engine protocol.

Build on a Linux CUDA machine with `nvcc`, a C++ compiler, `ar`, Python 3 and Rust 1.82+ available on PATH:

```sh
python3 examples/cuda-rust-runtime/build.py --arch sm_103 --out build/cuda-example
build/cuda-example/cuda-runtime-check 1000
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
