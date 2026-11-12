# Runtime: NKI to NEFF, libnrt, device-resident state

Purpose: how a Trainium engine is built and how its host code drives the device: compiling NKI kernels to NEFF
files at build time, embedding them, loading and executing them through libnrt, keeping weights and the KV cache on
the device. Everything here was run on the trn2.48xlarge of (`README.md`); `example/` holds the working code.

## Build time: an NKI kernel to a NEFF

An NKI kernel is a Python function decorated with `@nki.jit`. NKI parses its source (it is not executed as Python),
traces it for the shapes and dtypes of the arguments it is called with, and neuronx-cc compiles the result into a
NEFF, the executable format of a NeuronCore.

- **The documented way to get a NEFF** (Neuron docs, "debug a NEFF with nrtpy"): set `NKI_ARTIFACTS_DIR` to an
  empty directory **before** `import nki`, then call the `@nki.jit` kernel with numpy arrays. That selects nki's
  framework-free mode: it compiles the kernel, runs it once on a NeuronCore and leaves `kernel.neff` in that
  directory. `kernel[2](...)` compiles for logical NeuronCore config 2 (one program per physical core;
  `nl.program_id(0)` and `nl.num_programs()` split the work). The build therefore needs a free NeuronCore: give the
  build process `NEURON_RT_VISIBLE_CORES` = one of your run's experiment cores (`README.md`, hard rule 6), and one
  directory per kernel (run kernels in a pool of processes to build in parallel).
- **Do not use nki's private entry points** (anything with a leading underscore or outside the documented API): they are not a public interface, and this project uses
  only AWS's public, documented material (`README.md`, "Allowed sources"). Engines that used them must switch to the
  documented route above; the NEFF is the same.
- **What runs underneath.** nki writes `penguin.py` and BIR and runs `neuronx-cc compile --framework XLA penguin.py
  --pipeline compile SaveTemps --target trn2 --lnc 2 --output=<dir>/kernel.neff` (`build/<name>/log-neuron-cc.txt`).
  Set `NEURON_PLATFORM_TARGET_OVERRIDE=trn2` and put the Neuron virtual environment's `bin/` on `PATH`, since nki
  calls `neuronx-cc` as a subprocess.
- **Tensor names.** A NEFF's inputs are named after the kernel's Python arguments (`x`, `wT`, `g`); its outputs are
  `output_0`, `output_1`, ... in return order. `nrt_get_model_tensor_info` (`nrt_experimental.h`) lists name, size,
  dtype and shape.
- **Build cache.** The harness cleans the build tree except `target/` before every build. Cache NEFFs under
  `target/`, keyed by the kernel source and its shapes, so a session's rebuilds do not recompile unchanged kernels.

Pitfalls measured in `example/`: kernels built from closures fail (`inner functions can only be used as
fori_loop/while_loop body arguments`), top-level helpers called with constant arguments work; `nisa.tensor_copy`
has no `dtype=` (it casts to the destination's type); a computation whose result never reaches an output is removed
by the compiler, which makes "DMA only" probes misleading.

## Run time: libnrt

Headers in `/opt/aws/neuron/include/nrt/`, library `/opt/aws/neuron/lib/libnrt.so`. Build with
`g++ -O2 -std=c++17 -I/opt/aws/neuron/include ... -L/opt/aws/neuron/lib -lnrt -Wl,-rpath,/opt/aws/neuron/lib`.

1. `nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", "")` once.
2. `nrt_load(neff_bytes, size, start_vnc, vnc_count, &model)` per NEFF (4 to 8 ms each): load every NEFF before
   replying to `hello`.
3. `nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, vnc, bytes, name, &t)` for every weight, activation buffer and
   cache, **once**; `nrt_tensor_write` the weights once (about 10 GB/s).
4. `nrt_allocate_tensor_set`, `nrt_add_tensor_to_tensor_set(set, name, t)` per NEFF input and output, once. Reuse
   the same sets for every call: the profile of a resident-weight GEMV shows only the 8 KB activation moving per call.
5. `nrt_execute(model, inputs, outputs)` is synchronous and noisy (p50 71 us for a tiny kernel, p90 up to 380 us).
   Prefer the async API below.

**Device-to-device chaining works.** Two tensor sets that swap two device buffers (call A reads `bufA` and writes
`bufB`, call B the reverse) ran 8 dependent GEMVs without touching the host and matched a CPU reference, also when
all 8 were queued at once: the compute queue runs in order. So a decode step's output token buffer can be the next
step's input buffer.

**In-place updates of inputs work.** A kernel that DMAs into one of its input HBM tensors (`cache[3] = row`)
changes the caller's buffer and nothing else (`example/nrt_inplace.cpp`): this is how a KV cache stays resident.
A position chosen at run time needs dynamic indexing in NKI; not tried yet (`README.md`, open questions).

## Async execution (`nrt_async.h`)

- `nrta_execute_schedule(model, inputs, outputs, queue, &ret, &seq)` queues an execution and returns at once;
  `NRT_QUEUE_FULL` means retry. `nrta_is_completed(seq, &done)` polls; `nrta_event_register_seq_id_completion` signals
  an eventfd.
- `nrta_tensor_write`, `nrta_tensor_read`, `nrta_tensor_copy` run on the tensor-op queue.
- With many executions in flight the fixed cost per execute drops to 24 us (from 53 us one at a time). Pinning the
  process to one CPU makes synchronous calls steadier (45 us) but queued ones slower (72 us): do not pin when queuing.
- `nrt_execute_repeat` is rejected (`Supports for loops on the host is deprecated`, NRT_INVALID).

## Cores

The runtime claims every NeuronCore it can see (`NEURON_RT_VISIBLE_CORES` limits that); a second process on the same
cores fails with `Logical Neuron Core(s) not available ... (cores busy, ret=-16)`. Which cores are yours: `README.md`,
hard rule 6. Never run two engines on one core.

## LNC=2: per-token work runs twice unless you split it

Measured on DeepSeek-V4.1-Flash (64 logical cores, LNC=2, 4-token prefill groups). A kernel compiled for LNC=2 runs
on both physical cores of the logical core. GEMVs split by output tiles use both, but every step that is not
split (indexer, attention core, norms, RoPE, residual rows) runs redundantly on both physical cores while the DMA
idles. Give each physical core half of the tokens for those steps, and swap the rows the GEMVs need on both cores
with `nisa.sendrecv`. Only the owner writes residual rows. Device time per layer kernel fell 16-26%
(1107 -> 818 us, 808 -> 681 us). The arithmetic is unchanged, so tokens stay bitwise identical. Check a profile
for long engine-busy, DMA-idle stretches that appear twice before assuming a kernel is compute-bound.

## Host threads: wait with umwait, not pause loops

With ~40 pipeline-stage threads idle at any moment, `pause` spin loops slowed the threads that were inside
runtime calls on the same socket. A schedule call took 18.6 us alone and 30.4 us with 89 spinning threads.
`umonitor`/`umwait` (`-mwaitpkg`, C0.1) on the queues the stage threads wait on cut decode by 10-12%
(46.7-50.3 -> 42.8-44.7 ms/token). Apply it per wait and measure each one: on the expert workers' queues it made
prefill slower. The cores that slept in C0.1 also made the first requests after idle slower, so add a self
warm-up (below).

## Warm up with real requests at load

The first prompt requests of a process were 5-30% slower than later ones, and the harness times the request right
after a single warm-up. Calling every NEFF once at load did not fix it. Two real requests before answering `hello`
(about 300 + 8 and 1,100 + 2 tokens), then a reset, did: the first p256 went from 252 ms to 231 ms.
