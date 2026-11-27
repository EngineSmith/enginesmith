# The decode loop on Trainium

Purpose: how to structure a decode step given the measured cost of an execution, and how to keep the host out of
the token loop. The numbers are in `README.md`; the mechanisms in `runtime.md`.

## The budget

A batch-1 decode step is memory-bound: it streams the active weights once. For Qwen3-0.6B in bf16 that is about
1.2 GB of weights plus the KV cache read (the reference reads 1.50 GB from HBM per step at a 512-position bucket).
At the best measured GEMV rate (about 545 GB/s of device execution) that is roughly 2.2 to 2.8 ms per token: the
working floor until a faster kernel is measured. The reference takes 6.42 ms of device time per step (MBU 38%).

Every NEFF execution costs at least 24 us when queued, 53 us when waited for one at a time and 45 to 70 us or more
through synchronous `nrt_execute`. A 28-layer model with ten small kernels per layer would spend 280 x 24 us = 6.7
ms on fixed costs alone, more than the reference's whole step. So:

1. **One NEFF per decode step** if the compiler accepts it, else one per group of layers. Inside a NEFF the
   compiler schedules DMA and the engines without host involvement; across NEFFs every boundary costs the fixed
   floor.
2. **Queue, do not wait.** Schedule the steps of a request with `nrta_execute_schedule`, feeding each step's
   on-device token buffer to the next (`runtime.md`, chaining), and read the tokens back once per request (a small
   read costs 7 us). The protocol needs the tokens only in the reply.
3. **Argmax on the device**, at the end of the step NEFF, with ties to the lowest id: the logits never leave HBM.
4. **Position and KV write inside the step.** The step NEFF reads the position from a device tensor, writes the new
   K and V at that position (dynamic indexing, not yet measured here) and increments it, so consecutive steps need
   no host writes.

## Prefill

Prefill is compute-bound for long prompts and runs as one or a few large NEFFs over the prompt in chunks (the
reference compiles power-of-two buckets from 128 up to the task's longest length and pads the prompt to the next
bucket). Match the reference's numerics first (`references/correctness.md`); the partition does not have to match
when the gates allow rounding-level differences, but every difference must be measured against the gates.

## What a tuned engine does (measured, `examples/qwen3-0.6b-trn2/optimization-log.md`)

- **Issue every weight and KV-cache DMA from an engine that computes nothing**: the Sync engine's hardware DGE
  queue (`dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync`). A software DGE DMA is issued from GpSimd's
  in-order stream, so the weight prefetch waits behind every norm, cast and core exchange GpSimd runs: 2,205 ->
  2,078 us per Qwen3-0.6B step.
- **Write the KV cache through a separate kernel input bound to the same device tensor** (`kcw` aliasing `kcache`):
  otherwise the compiler orders layer l+1's cache load after layer l's dynamic-offset write to the whole tensor.
  Safe when the step never reads what it writes (current token from SBUF): 2,078 -> 1,987 us.
- **Buckets at the timed positions** (prompt length + up to 64 steps): an attention bucket just above a timed
  position saves the reads of a larger one (+0.8 to +2.9% decode).
- **Keep the host spinning between requests**: a process that blocked in `read()` on stdin was measurably slower
  on the next request (prefill +2 to +5% after switching to a non-blocking read with a 10 s spin).
- **libnrt may print to stdout** (warnings): write protocol replies to a private copy of the original stdout and
  point file descriptor 1 at stderr before `nrt_init`.
- **Several decode steps in one NEFF: watch the per-partition buffers** (Gemma-3-4B, one core, near the bandwidth
  floor). A NEFF running two steps back to back was first 15% slower than two launches of the one-step NEFF (30.3 ms
  against 2 x 13.1 ms at a 4,608-key window); the cause was a persistent SBUF buffer for the K/V window (72 KB per
  partition) that the two-step body kept alive. Without it the two-step NEFF took 25.5 ms, 12.7 ms per step (-2.9%),
  and running 2 steps per NEFF at every bucket gave +0.5 to +2.2% decode. Going from 2 to 4 steps per NEFF added only
  0.2 to 0.3%: by then the launch overhead is already gone. Each step is still one forward pass per
  generated token (check with `enginesmith decode-probe`).
