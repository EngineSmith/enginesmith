# Profiling on Trainium

Purpose: measuring the device time of a NEFF and finding where it goes. Use it before and after every optimization;
compare with the floors in `decode-loop.md`.

## One NEFF, device side

```sh
NEURON_RT_VISIBLE_CORES=<a free core> neuron-explorer capture -n kernel.neff -s prof.ntff --num-exec 3
neuron-explorer view -n kernel.neff -s prof_exec_3.ntff --output-format summary-text   # prints to the console
```

`capture` runs the NEFF with random inputs (it was called `neuron-profile` in older releases). Use the last
execution's trace (the first includes warm-up). Each trace of a decode-sized NEFF is a few hundred MB. `view`
refuses `--output-file`; redirect the console. `--output-format json` or `perfetto` gives per-instruction timelines.

Metrics that matter (names as printed):

| metric | meaning |
|---|---|
| `total_time` | device time of one execution, seconds |
| `hbm_read_bytes` | bytes read from HBM: compare with the weights plus KV bytes the step must read |
| `mbu_estimated_percent` | memory bandwidth utilization (0 to 1 despite the name) |
| `dma_active_time_percent` | how much of the time DMA is busy; near 1 with low MBU means many small or strided transfers |
| `software_dynamic_dma_*` | DMA descriptors generated on GpSimd at run time (gathers): costly when large |
| `tensor_engine_*`, `vector_engine_*`, `scalar_engine_*`, `gpsimd_engine_*` | per-engine active time |
| `throttle_avg_util_limit_nc*_percent` | power throttling of the engines (0 to 1) |

## Examples (measured)

- The reference (NxD Inference) token-generation NEFF of Qwen3-0.6B, 512-position bucket: `total_time` 6.42 ms,
  `hbm_read_bytes` 1.50 GB, MBU 0.38, DMA active 0.75, throttling limit 0.52 to 0.60.
- vllm-neuron 0.24's decode NEFF of the same model (8,192-position bucket): 58.9 ms, 6.7 GB read, 6.2 GB of it by
  software dynamic DMA (a paged-KV gather), throttling limit 0.5. That is why vllm-neuron is not the reference.
- The best GEMV of `example/`: `total_time` 76.6 us, execution 61.6 us, MBU 0.61, DMA-bound with software DGE.

## Inside the engine

Device time per NEFF from the profiler plus the fixed per-execute cost (`README.md`) should add up to the measured
step time; if not, the host is in the loop. Time whole requests the way the harness does (from writing the request
to reading the reply) with the interleaved method of `references/measurement.md`.

## Know when a model is at the hardware ceiling
Qwen3-8B (bf16, one logical NeuronCore) reached both limits: decode at about 23.5 ms per token reads the 16.4 GB of
weights at about 95% of the core's HBM bandwidth (floor about 22.6 ms), and prefill at 1,024 and 4,096 tokens keeps the
matrix engine 95-97% busy and power-throttled. Two independent runs stopped near the same score after that. Before a
long session, compute both floors for your model (weight bytes / HBM bandwidth for decode, FLOPs / matrix-engine peak
for prefill) and compare with your measured times: within about 10% of both, the remaining gain is an algorithmic one
(the speculation rule in README applies), not a kernel one.
