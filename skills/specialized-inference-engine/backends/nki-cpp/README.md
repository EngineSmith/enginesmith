# NKI + C++ backend (AWS Trainium2)

Purpose: the entry point for any engine whose kernels are NKI (Neuron Kernel Interface) compiled ahead of time to
NEFF files, and whose host code is C++ (or Rust through C bindings) driving them through libnrt, the Neuron runtime,
on Trainium2 (measured on a trn2.48xlarge). It lists the backend's files, the platform rules every change must keep,
and the measured platform numbers. Read it before the first change to a Trainium engine and whenever a number
elsewhere looks off.

Numbers are measurements on one trn2.48xlarge ( 16 Trainium2 devices, logical NeuronCore config 2, Neuron
runtime 2.34.10, neuronx-cc 2.27.5334, nki 0.6.0) unless marked as estimates. Sources are named next to each one.

## Files

| file | read when |
|---|---|
| `README.md` | first; before any Trainium change; when you need a platform number |
| `runtime.md` | the toolchain: NKI kernel to NEFF at build time, embedding NEFFs in the executable, libnrt load / tensors / execute / async, device-resident weights and KV cache |
| `decode-loop.md` | the per-execute cost floor and what it implies: one NEFF per decode step (or per layer group), queued asynchronously, tokens kept on the device |
| `kernels-fusion.md` | one NEFF per decode step: all layers, LM head and argmax; column layout; splitting a layer over the two physical cores |
| `prefill.md` | whole-model prefill NEFFs: feature-major GEMMs, nkilib flash attention, layered prefill for long prompts |
| `kernels-gemv-quant.md` | decode GEMV rates at real shapes, the PSUM accumulation-depth cliff, why fp8 requantization is not allowed |
| `tensor-parallel.md` | models that do not fit one logical NeuronCore: in-NEFF collectives, the SPMD load, the split, host threads per rank |
| `attention.md` | decode attention with the KV cache in HBM written at a position held on the device; layouts without transposes; what works inside `fori_loop` |
| `profiling.md` | device time of one NEFF with `neuron-explorer`; the metrics that matter; the reference's own profile |
| `counterexamples.md` | before trying an idea: measured negative and neutral results |
| `example/` | a working, minimal build: NKI kernels (bf16 GEMV, fused RMSNorm + GEMV, in-place cache write), the NEFF build script, a C++ runner on libnrt with a CPU check and timing |

Topics of the `metal-rust` set not written yet for this backend (no measurement exists; write them as you measure):
`serving.md`. The method in `references/` applies unchanged.

## Hard rules

0. **The model runs on the NeuronCores.** The target is the chip: every layer of the forward pass (the matmuls,
   attention, norms, the LM head and the argmax) executes as NKI kernels on the NeuronCores. The host CPU loads
   weights, schedules NEFFs and handles the protocol; it does not compute the model, not even as a temporary
   baseline. A CPU engine is not an engine for this target, whatever its speed. The harness refuses an executable
   that does not link libnrt.
1. **Everything that touches model tensors is engine source.** NKI kernels (written or copied and adapted from an
   open-source project under its license, with a provenance record in `THIRD_PARTY.md`) and the host code that
   schedules them. No torch-neuronx, libneuronxla, JAX, NxD / NxD Inference, vLLM or any other framework, graph
   compiler or inference runtime, at build time or run time, for the model's computation. Their NKI kernels may be
   copied into the engine and changed; then they are the engine's own.
2. **Build time may use Python, NKI and neuronx-cc; run time may not.** The build compiles each NKI kernel to a
   NEFF (`runtime.md`). The executable links only libnrt (and libc, libstdc++, libm, libgcc_s); the harness checks
   `ldd`. It starts no Python process.
3. **NEFFs are embedded in the executable.** The harness copies only the executable out of the build tree, and the
   engine may read nothing but the model directory at run time. Embed every NEFF (for example with `.incbin` in an
   assembler file, or `ld -r -b binary`) and load it from memory with `nrt_load(bytes, size, ...)`.
4. **The model's shapes are compile-time constants.** NKI traces a kernel for fixed shapes; a NEFF is valid for
   exactly the shapes it was compiled with. Requests outside the compiled range are refused with an error reply.
5. **Weights are read from the checkpoint as downloaded, values unchanged.** Re-laying them out at load time (for
   example pre-swizzling a weight so each partition reads contiguous bytes, `example/`) is allowed and was the
   largest GEMV win measured; changing a value (requantizing, casting to another type) is not.
6. **The engine gets exactly the cores the reference generates on.** With a reference of tensor-parallel degree N
   (1 unless the task sets `[reference] tensor_parallel`; more cores: `tensor-parallel.md`), the harness starts the
   engine with `NEURON_RT_VISIBLE_CORES` set to N logical NeuronCores (two physical cores each with logical NeuronCore
   config 2), starting at `ENGINESMITH_NEURON_CORE`; the reference sees those N and the next N, where it judges. Use
   every core you see; there are no others. Without the variable the Neuron runtime claims every core it can see.
   Experiments outside `enginesmith dev-eval` and kernel builds use the N cores after the run's 2N
   (`ENGINESMITH_NEURON_CORE` + 2N onward, set in your session's environment), so they never collide with an
   evaluation or with another run on the machine.
7. **Other runs share this machine.** Several EngineSmith runs (other agents) use other NeuronCores of the same
   host at the same time. Never wait for, signal or count processes by a machine-wide pattern (`pgrep -f "enginesmith
   dev-eval"` never becomes empty while another run evaluates, and inside `bash -c` it matches its own command line, so
   `until ! pgrep -f ...` never ends): wait for the process ids you started (`wait $PID`), or for your
   own output files. Never touch cores outside your run's.
   **Never kill by name**: `pkill -f build_neffs.py`, `pkill -f neuronx-cc`, `killall` and the like also kill other
   runs' builds and evaluations (one such command failed another run's harness build). Kill only the process ids or
   process group you started (`kill -- -$PGID`).
8. **Greedy means the largest logit, ties to the lowest token id**, computed on the device (an argmax kernel at the
   end of the step) so the host never reads logits in the token loop.
9. **Speculative decoding only where the task allows it, and then judged on real text.** With `[engine]
   one_forward_per_token` on (the default), the brief rules it out: one forward pass per generated token. Where a task
   sets it to `false`, a lossless draft-and-verify decoder (the emitted tokens are exactly the model's greedy argmaxes)
   is a legitimate optimization. But the harness's timed prompts
   are random token ids, on which a small model's greedy output falls into loops that a prompt-lookup draft
   predicts almost perfectly: a prompt-lookup engine scored decode 7.65x there and about 2x over the same engine
   without speculation on real prompts (IFEval, 841 against 410 tokens/s; NxD Inference 169). The project reports
   speed on real-text prompts (task-accuracy prompts), so optimize for those, and never let a verify step run past
   the compiled positions: every request whose prompt plus output fits the engine's limit must succeed (that
   engine failed on long MATH-500 answers with `continuation exceeds the compiled positions`).
   The timed workload never asks for long outputs, so the harness will not catch this. Check it before you finish:
   one request with a short prompt and `max_tokens` up to the engine's compiled limit minus the prompt (for example
   a 200-token prompt and 3,840 output tokens on a 4,096-position engine) must return every token, with no error,
   near the end of the cache as well as at the start. The same engine failed this way in three accepted rounds.
   Do not tune to the timed prompts: a draft table built from the model's outputs on prompts drawn like the harness's
   (random token ids from its generator, many seeds) predicts those loops and inflates the harness score without helping
   real text. Draft sources must come from the request itself (its prompt and output so far), not from the outputs of
   earlier requests kept across calls: the timed requests are related by construction, real ones are not.

## Allowed sources: public, open-source AWS material only

Use only what AWS has released publicly: the Neuron SDK packages from the public repositories
(pip.repos.neuron.amazonaws.com, apt.repos.neuron.amazonaws.com), the public NKI language and ISA APIs
(`nki.language`, `nki.isa`, `@nki.jit`) as documented at awsdocs-neuron.readthedocs-hosted.com, and open-source
repositories such as github.com/aws-neuron/nki-library and github.com/aws-neuron/nki-samples. Never use AWS-internal
or unreleased material: no undocumented or internal NKI APIs (only what the public `nki` package documents), no
private repositories, no internal reference packs, no other package or skill whose
open-source status you cannot show. When in doubt, do not use it, and say so in the notes.

- **`nkilib/`: AWS's NKI Library, open source (Apache-2.0)**, github.com/aws-neuron/nki-library; a copy validated
  against this compiler ships inside the neuronx-cc wheel
  (`$NEURON_VENV/lib/python3.12/site-packages/nkilib/`). `core/attention/attention_tkg.py` (decode attention),
  `core/attention/attention_cte.py` (prefill attention), `core/qkv/`, `core/mlp/`, `core/output_projection/` and more.
  Copying a kernel into the engine and adapting it is allowed: record it in `THIRD_PARTY.md` with its path,
  version (neuronx-cc 2.27.5334) and license, keep the license header, and copy only what you use. Check that the
  kernel uses only the public NKI API before you rely on it.

## Platform numbers

| quantity | value | source |
|---|---|---|
| HBM per logical NeuronCore | 24 GiB (96 GiB per device, 4 logical cores) | `neuron-ls --json-output` |
| bf16 GEMV 4096 x 4096 (32 MiB), best kernel, queued | 74.8 us per call, 448 GB/s (device execution 61.6 us, about 545 GB/s, MBU 61%) | `example/` `gemv_swz4`, `neuron-explorer` |
| same, naive [128 x 512] tiles | 123.6 us, 272 GB/s | `example/` `gemv` |
| fixed cost per execute, queued (`nrta_execute_schedule`, many in flight) | 24 us (about 19 us of it on the device for a 4 us kernel) | `example/` `tiny_copy` |
| per execute, one at a time (schedule, then spin) | 53 us | same |
| per execute, `nrt_execute` | p50 71 us, p90 up to 380 us; 45 us with the process pinned to one CPU | same |
| `nrt_load` of a small NEFF | 4 to 8 ms | same |
| host to device upload | about 10 GB/s (32 MiB in 2.4 to 3.5 ms) | same |
| small read back (2 KB) | 7 us | same |
| reference decode step, Qwen3-0.6B bf16 (NxD Inference, 512-token bucket) | 6.42 ms device time, 1.50 GB read from HBM, MBU 38% | `neuron-explorer` on its token-generation NEFF |
| reference, Qwen3-0.6B, whole requests | decode 6.26 ms/token at 256 positions, 5.10 at 4,096; prefill 11 ms at 256, 155 ms at 4,096 | `enginesmith.references.nxdi`, async + fused QKV |
| power throttling | the profiler reports the engines throttled to an average utilization limit of 50 to 60% during a decode step | `throttle_avg_util_limit_*` in `neuron-explorer` summaries; not understood yet |

Estimate (vendor figure, not measured here): about 2.9 TB/s of HBM bandwidth per Trainium2 device, so roughly
0.7 TB/s per logical NeuronCore. The best measured GEMV reaches about 545 GB/s of device execution time; use the
measured figure for floors.

## Open questions (not measured)

- Loops inside one NEFF over layers with different weights, and the largest NEFF the compiler accepts in reasonable
  time.
- Whether the throttling above is a property of the instance or of the workload.
