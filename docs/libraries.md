# What an engine may link

An EngineSmith engine is specialized: it serves one model on one machine, and every computation on the model is its
own source, from the accelerator's driver up. Which libraries it may link is what draws that line, so every platform
fixes them in a list and checks every build against it (`[engine] native`, on by default).

## The rule

An engine links only:

1. the operating system's and the C/C++ runtime's libraries;
2. the accelerator's driver and runtime layer, which allocates memory, loads the engine's own kernels and launches
   them;
3. utilities that do no arithmetic on the model (compression, file formats and the like).

Anything that computes on the model's tensors is refused: Python, frameworks (MLX, PyTorch), inference runtimes
(llama.cpp, Core ML, ONNX Runtime) and precompiled numerical libraries (a BLAS, MPS, Accelerate, rocBLAS, cuDNN).
Their open-source kernels may be copied into the engine's source and adapted, under their licenses; copied code is
the engine's own.

## Why

- **The speed comes from specialization, and a library call is a wall it cannot cross.** Around a call, inputs and
  outputs go through memory, and no step on one side can be fused with a step on the other. The library's kernels
  were tuned for other shapes, and the coding sessions cannot change them for this model.
- **The score measures the engine.** With a library doing the arithmetic, it would measure the library.
- **Everything that computes on the model is in the published source**, so anyone can read it and check it.
- **The driver layer is the exception** because there is no other way to use the accelerator, and it does not know
  what it computes.

## Whether a library belongs on a list

Ask three questions:

1. Does it do arithmetic on the model's numbers for the engine (matrix multiplication, attention, normalization,
   convolution, sampling)? Then it is refused: that is the engine's work.
2. Is it the only way to use the hardware (the driver and runtime layer)? Then it is allowed.
3. Is it unrelated to the model (memory, files, threads, text encoding)? Then it is allowed.

In short: if what the library does is the model's computation, the engine writes it; if it is the way to the operating
system or the hardware, the engine may link it.

## Example: a MacBook (`macos-metal`)

`otool -L` lists the libraries a Mach-O executable loads when it starts. For an engine EngineSmith built for
Qwen3.5-0.8B on an M5 Max, it prints:

```
/System/Library/Frameworks/Metal.framework/Versions/A/Metal
/System/Library/Frameworks/Foundation.framework/Versions/C/Foundation
/usr/lib/libSystem.B.dylib
/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation
/usr/lib/libobjc.A.dylib
```

The kernels are not among them: they are Metal Shading Language source embedded in the executable, which Metal
compiles when the engine starts. On every build the check reads this list. A framework is named by its framework
(`Metal`), any other library by its file name up to the first dot (`libSystem`); each must be the system's own copy
(under `/usr/lib` or `/System/Library`).

| Allowed | What it does | Why it is allowed |
|---|---|---|
| `libSystem` | the C runtime: memory, files, threads | the operating system |
| `libc++` | the C++ runtime | the C++ runtime, for an engine written in C++ |
| `libobjc` | the Objective-C runtime | Metal is an Objective-C API; any language calls it through this layer |
| `Foundation`, `CoreFoundation` | Apple's basic types: strings, arrays, errors | Metal's API takes and returns them |
| `libiconv` | text encoding conversion | a utility; no arithmetic on the model |
| `Metal` | the GPU: memory, compiling the engine's own kernel source, submitting work | the driver and runtime layer: there is no other way to the GPU, and it brings no kernels of its own |

| Refused | What it is | Why |
|---|---|---|
| `libpython`, `Python` | the Python interpreter | the engine must be self-contained |
| `libmlx` (MLX, mlx-lm) | a machine-learning framework | the framework would compute the model |
| `MetalPerformanceShaders`, `MetalPerformanceShadersGraph` (MPS) | Apple's ready-made GPU kernels: matrix multiplication, convolution, attention | precompiled numerical kernels |
| `Accelerate` (BLAS, BNNS, vDSP) | Apple's CPU numerical library | precompiled numerical kernels |
| `CoreML` | Apple's inference runtime | it would run the whole model |
| `libggml`, `libllama`, `libonnxruntime`, `libtorch` | other engines and frameworks | they would compute the model |

Metal and MPS are easy to confuse: both are Apple's, both are about the GPU, and both come with every Mac. Metal is
the stove; MPS is a ready meal. The engine may use the stove, and cooks the meal itself.

## AMD Instinct GPUs (`linux-rocm`)

The check reads the executable's direct dependencies (the ELF `DT_NEEDED` entries); a library is named by its file
name up to `.so`.

| Allowed | What it does | Why it is allowed |
|---|---|---|
| `libc`, `libm`, `libdl`, `libpthread`, `librt`, `libutil`, `libresolv`, `libatomic`, `libgcc_s`, `libstdc++` | the C and C++ runtimes | the operating system |
| `ld-linux-x86-64`, `ld-linux-aarch64`, `linux-vdso` | the dynamic loader and the kernel's system-call page | they start the program |
| `libamdhip64`, `libhsa-runtime64` | the HIP runtime and the HSA layer under it: device memory, loading and launching kernels | the driver and runtime layer |
| `libhiprtc`, `libhiprtc-builtins`, `libamd_comgr` | compile the engine's own kernel source at run time | the runtime layer: they compile the engine's kernels and bring none |
| `libdrm`, `libdrm_amdgpu` | the kernel driver's interface | the driver layer |
| `libz`, `libnuma`, `libelf` | compression, memory placement, reading ELF files, for the HIP runtime | utilities; no arithmetic on the model |
| `librccl` | communication between GPUs | allowed for now, see "Open questions" |

| Refused | What it is | Why |
|---|---|---|
| `librocblas`, `libhipblas`, `libhipblaslt` | matrix multiplication | precompiled numerical kernels |
| `libMIOpen` | neural-network kernels | precompiled numerical kernels |
| `libcomposable_kernel`, `libck`, `libaiter` | fused attention and matmul kernels | precompiled numerical kernels (their source may be copied) |
| `librocprim`, `libhipcub`, `librocsolver`, `librocfft`, `librocsparse`, `librocrand` and their `hip` versions | numerical primitives | precompiled numerical kernels |
| `libtorch`, `libc10`, `libmigraphx`, `libtriton`, `libvllm`, `libpython` | frameworks, compilers and runtimes | they would compute the model |

## Code compiled into the executable

A library linked statically, a Rust crate or code copied from another project becomes part of the executable and
does not appear in the list the check reads. That is allowed: it is the engine's source, published with it, and the
coding sessions can change it for this model (a crate that parses safetensors, kernels adapted from MLX, a header-only
template library). A framework compiled in the same way still breaks the rule, though the check cannot see it; the
published source shows it.

## A library that is not on the list

The build is refused with the library's name, and the session sees why. If the library passes the three questions
(for example a new system library after an operating-system update), add it to the platform's `ALLOWED` and to its
table here in the same change; `tests/test_platforms.py` keeps the two equal.

## Adding a platform

A platform module defines `ALLOWED` and a `native_problem` that refuses every library outside it (an allow-list, never
a list of refused names: a list of refused names misses the ones nobody thought of). Its pull request adds a section
here with an "Allowed" table, what each library does and why it passes the three questions, and the libraries it
refuses. CONTRIBUTING.md has the rest of the interface.

## Open questions

- **Communication between GPUs** (RCCL on AMD, NCCL on NVIDIA). Their collectives compute on the model's activations
  (an all-reduce sums them), and specialized communication has room in decode, where the messages are small and can
  be fused with the next step. They are allowed for now, to be decided when an engine needs several GPUs.
