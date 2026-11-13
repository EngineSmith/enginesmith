# Tensor parallelism on Trainium2: one decode step over several logical NeuronCores

Purpose: how to run a model that does not fit in one logical NeuronCore (24 GiB of HBM), or that must decode
faster than one core's bandwidth allows, as an SPMD NEFF on W logical NeuronCores driven natively through libnrt:
which communication works, the measured costs, the Megatron-style split that reproduces HF greedy output, and the
host-side rules that keep it from hanging. Measured by kernel lab 2 on one trn2.48xlarge (nki 0.6.0, neuronx-cc
2.27.5334, Neuron runtime 2.34.10, aws-neuronx-collectives 2.34, logical NeuronCore config 2, 2026-11-04) on logical
cores 56-63 with its own C++ drivers and a CPU reference (HF transformers fp32). It has not been run through an
EngineSmith evaluation yet. Public NKI API (`nki.collectives`, `nki.isa`, `nki.language`) and documented runtime
API / environment variables only.

## Result

| Qwen3 decode step, bf16 weights, b512 attention bucket, one NEFF per core per step, queued | per-core HBM | ms/step | floor (estimate) |
|---|---|---|---|
| 8B on 1 logical core (`kernels-fusion.md`) | 17.1 GB | 23.1 | 23.1 |
| 8B, TP=2 | 9.2 GB | 12.46 | 11.6 |
| 8B, TP=4 | 5.2 GB | 7.10 | 5.8 |
| 14B (29.5 GB, does not fit one core), TP=2 / TP=4 | 15.9 / 8.8 GB | 22.41 / 12.17 | 21.3 / 10.7 |
| 32B (62 GB), TP=4 | 17.9 GB | 26.06 | 24.3 |
| 32B, TP=8 over two devices | 10.3 GB | 15.37 | 12.8 |

(floor = bytes streamed per core per step / 658 GB/s, the best single-core GEMV rate.) Every run gives HF fp32's
greedy tokens on a real prompt (teacher-forced 11/11, 24/24 generated), with the same token history on all ranks.
With a 470-token prompt, 8B TP=2 / TP=4 agree on 467 / 468 of 469 teacher-forced positions (near-ties, fp32 logit
gap <= 0.048). Collectives cost 0.5-1.3 ms per step (7-10 us per all-reduce, 2 per layer). The rest streams
weights at 580-650 GB/s per core, as on one core.

## Which communication to use

| option | verdict |
|---|---|
| `nki.collectives.all_reduce` (and `all_gather`, `reduce_scatter`, `collective_permute`, ...) inside the NEFF | **use this, with HBM buffers**. 16 KB all-reduce: 9.8 us (2 ranks), 11.3 us (4), 13.4 us (8 over two devices); SBUF buffers 11.6 / 14.9 / 22.3 us. 16 MB: 48 / 125 / 62 GB/s algbw. `collective_permute` is slower (16-20 us) |
| `nisa.sendrecv` | only between the two physical cores of one logical core (0.85 us); not across logical cores |
| host-driven `nrta_cc_prepare` / `nrta_cc_schedule` (marked work-in-progress) | works, ~0.6 ms per 4 KB all-reduce: too slow per layer |
| `nrt_tensor_copy` across logical cores | refused: `Tensors must be allocated on same HBM` |
| a NEFF given a tensor allocated on another logical core | refused: `Tensor ... allocated on HBM 2 was passed to a model loaded on lnc 0. That is not supported.` |
| host bounce (`nrt_tensor_read` + `nrt_tensor_write`) | works, 9 us per 4 KB, ~7 GB/s; a host round trip per layer |

## Build: one NEFF for all ranks, through the documented NKI route

```python
import nki.collectives as ncc
from nki.collectives import ReplicaGroup
red = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
ncc.all_reduce(srcs=[partial_sb], dsts=[red], op=nl.add, replica_group=rg)    # rg: kernel argument
...
kernel[2](*[[arr_rank0, arr_rank1, ...] for each tensor argument], rg=ReplicaGroup([[0, 1, ..., W-1]]), **consts)
```
- With `NKI_ARTIFACTS_DIR` set, calling the kernel with **lists of numpy arrays** (one per rank) compiles one NEFF
  and runs it once on the first W visible cores. The build process therefore needs W free cores in
  `NEURON_RT_VISIBLE_CORES` and `NEURON_RT_ROOT_COMM_ID=localhost:<free port>`. Without the latter: `nrt_load_collectives
  NEURON_RT_ROOT_COMM_ID is not set; cannot load NEFF with collective operations!`. Only shapes matter, so build
  with zero arrays.
- A Python list argument is always taken as "one value per rank". Pass non-tensor structures (e.g.
  `collective_permute`'s source-target pairs) as tuples: a list of pairs is split over the ranks
  (`expecting 'list or tuple', got 'int'`).
- **SBUF collective tiles need a free size of at least max(2, W)**. Smaller ones compile and then assert in
  `nrt_load`: `get_enc_tensor_from_collectives_tensor: Assertion 'coll_tensor->num_elem[0] > 0 && ...'`. For
  scalars (the argmax), replicate the value over 8 columns.
- Under LNC=2 an SBUF collective runs per physical core: each physical core's buffer is reduced with the same
  physical core of the other ranks. HBM collectives work on the logical core's single buffer, and nkilib puts a
  `core_barrier` before them.
- Collectives may not read or write the kernel's I/O tensors directly (nkilib: "Collective instruction cannot
  read/write IO tensors"); copy into an internal buffer first.
- Compile time drops with W (each core traces 1/W of the matmuls): 8B 36 layers 319 s on 1 core, 170 s at TP=2,
  103 s at TP=4; 14B 40 layers 298 s (TP=2), 161 s (TP=4); 32B 64 layers 361 s (TP=4), 209 s (TP=8).
  nrt_load: 2.2 s (8B) to 9.2 s (32B) for all ranks.

## Run: libnrt, one process, one thread per rank

1. `nrt_init`, then for each rank r: `nrt_load_collectives(neff, size, vnc = r, -1, ctx_device_id = r,
   ctx_device_count = W, &model[r])` (sequential is fine; 2.2-3.9 s for a 4-13 MB decode NEFF on all ranks).
2. Allocate every tensor of rank r on vnc r (`nrt_tensor_allocate(DEVICE, r, ...)`), upload that rank's weight
   shards, and build rank r's tensor sets. For the lab 1 decode chain, that is two sets per rank swapping `pos` and
   `output_0`.
3. **Schedule each rank's executions from its own thread.** `nrta_execute_schedule` of a collective NEFF blocks
   until every peer has scheduled the same execution, so a single thread looping over ranks hangs on its first call.
   With one thread per rank, each thread queues all its steps and then polls `nrta_is_completed`.
4. Set `NEURON_RT_DISABLE_EXECUTION_BARRIER=1` (documented: "valid only when the workload is genuinely SPMD").
   Without it, a NEFF holding one all-reduce took 82-900 us per execution (bimodal, median ~320-880 us at 4 ranks).
   With it, it takes a stable 82 us. For the long decode-step NEFFs it changed nothing (7.51 vs 7.53 ms). Two
   things that are not the cause: busy polling (sleeps did not change it), and device contention (4 shards in 4
   processes on one device each ran exactly as fast as one alone).
5. Warm up: the first chain after loading was 140-200 ms slower than the next identical one.

## Topology rules (measured failures)

- `NEURON_RT_VISIBLE_CORES` must be one contiguous range. `58,62` fails in `nrt_init` with `Logical Neuron cores
  are not contiguous`.
- Across devices, use whole devices (8 ranks on 56-63 worked) or 1 + 1 cores with HBM buffers (59-60 worked, but at
  30.9 us per 16 KB). 2 + 2 cores on 58-61 fail at the first execution: `ENC:enc_init_comm ... failed (2) to init a
  collective algorithm. reason: no_hier no_mesh`. SBUF buffers on 59-60 fail with an assertion in
  `enc_post_operation`.
- SBUF all_gather of 1 MB per rank at 4 ranks fails at the first execution (`ENC:enc_post_function ... Failed to post
  function 0`). Smaller sizes and 2 ranks work.

## The split (Megatron, generalised from the two-physical-core split of `kernels-fusion.md`)

2W "virtual cores" g = 2 x rank + pid. The lab 1 single-core layout is split over them:
- QKV N-split by heads (32/2W q heads, 8/2W kv heads per virtual core for 8B). When 2W exceeds the kv head count,
  replicate kv heads: each virtual core keeps its own copy of its kv head's cache. Attention and the KV cache
  stay local.
- o_proj and down K-split. Each virtual core produces a partial `[128, HT]` fp32 hidden vector. The two physical
  cores add theirs with `nisa.sendrecv` (0.85 us), both DMA the sum to an internal named `nl.shared_hbm` buffer, one
  HBM `all_reduce` runs over the ranks, and both cores DMA the result back. This was faster than an SBUF
  all-reduce per physical core followed by the sendrecv: 8B TP=4 7.44 -> 7.10 ms, 32B TP=8 17.05 -> 15.37 ms.
  SBUF all-reduces of the [128, 40] partial of 5120-wide models are especially slow (19-22 us). Per layer: 2
  all-reduces + 2 sendrecvs.
- Issue the first 4 weight slabs of the next GEMV before each all-reduce so they stream during it (-0.05 ms at
  8B TP=4; no gain at 32B).
- If the virtual-core count does not divide the intermediate size (32B TP=8: 200 tiles over 16), zero-pad gate/up
  rows and down columns. It is a layout change (SiLU(0) x 0 = 0), not a value change.
- gate/up N-split over the same intermediate tiles as down.
- LM head N-split over the vocabulary (padded to 2W x NVL tiles, padding masked to -3e38). Each core computes a
  local (max, BIGC - lowest id), combines it over the physical cores with a sendrecv, then over the ranks with
  `all_reduce(max)` of M = max, then `all_reduce(max)` of (gm == M) x score. This is exact; ties go to the lowest
  id. Put the scalars in row 0 of a [128, 8] tile filled with the op's identity: a 1-partition collective is
  ~2x slower.
- Embedding, norms, residual, token selection and the `toks` / `pos` writes run redundantly on every rank, each
  into its own tensors. All ranks run the same NEFF; rank-specific values (the vocab id base) come in a per-rank
  input tensor.

## Pitfalls

- Never edit a kernel source while a build of it is tracing (lab 1). With TP builds of several minutes this is easy
  to do by accident.
- `nrt_load` is per rank: ranks are the visible cores in order; NEFFs are not tied to particular cores, so the same
  NEFF ran on 56-59 and 60-63.
- The runtime's device profile (`NEURON_RT_INSPECT_DEVICE_PROFILE=1` / `model`) was not written for async
  multi-rank executions: only the NEFF copies (and, with the system profile on, a system trace) appeared. Profile a
  shard with the same kernel built for W=1 instead (`NOCC=1` in the lab's test).

## Small all-reduces in decode: all-gather and a fixed-order sum (gpt-oss-20b, measured)

For the small per-layer reductions of decode ([120, 24] and [120, 96] fp32 per rank over 4 ranks), `ncc.all_gather`
of the partials into SBUF followed by a fixed-order local sum ((r0 + r1) + (r2 + r3), the same on every rank) beat
`ncc.all_reduce`: about 2.5-9 us against 11.3 us (noisy micro-benchmark), and decode 248 -> 239 ms per 64 tokens.
The result is not bitwise equal to the all-reduce's (its fp32 order differs); check the error against float64 on
random data (6e-8 relative here) and with dev-eval. `ncc.all_reduce` also accepts SBUF source and destination,
which removes the HBM hop and a core barrier (278 -> 261 ms); for the larger [120, 96] it was slower than the HBM
form (20.8 against 18.9 us), so measure each size. Measured and not kept: one all-reduce per expert as soon as its
partial is ready (collectives in flight stalled the expert weight DMAs: 261 -> 322 ms).

The first collective of an execution pays a start and rank-skew delay (a 2.9 MB all-reduce took 149-174 us first,
33-41 us later). Fewer executions help: running all 24 one-chunk prefill layers in one NEFF instead of one per
layer took 256-token prefill from 32.5 to 30.6 ms.
