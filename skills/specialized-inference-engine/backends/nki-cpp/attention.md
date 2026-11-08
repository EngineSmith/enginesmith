# Attention on Trainium2

Purpose: decode attention with the KV cache resident in HBM and the new K/V written at a position held in a device
tensor, so that consecutive decode steps need no host writes. Measured by the kernel lab on one trn2.48xlarge
(nki 0.6.0, neuronx-cc 2.27.5334, one physical core of a logical NeuronCore, 2026-11-03); Qwen3-0.6B shapes
(16 query heads, 8 KV heads, head dim 128, up to 4,608 positions). Public NKI API only (`nki.isa`, `nki.language`).

## Writing the cache at a run-time position (works, bitwise)

Load the position from a `[1, 1]` int32 input into SBUF, then into a register, and use it as `scalar_offset` of an
access pattern on the HBM side:

```python
p_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
nisa.dma_copy(dst=p_sb, src=pos)
preg = nisa.register_alloc(); nisa.register_load(preg, p_sb)
# V cache [TMAX, 1024]: row pos
nisa.dma_copy(dst=vcache.ap(pattern=[[1024, 1], [1, 1024]], offset=0, scalar_offset=preg, indirect_dim=0), src=vrow)
# K cache stored transposed [8, 128, TMAX]: column pos of every (head, d) row
nisa.dma_copy(dst=kcache.ap(pattern=[[TMAX, 128], [128 * TMAX, 8]], offset=0, scalar_offset=preg, indirect_dim=2),
              src=kT_new)
# a RoPE table row at pos, broadcast over partitions (partition step 0 on the HBM side)
nisa.dma_copy(dst=cs, src=cos.ap(pattern=[[0, 24], [1, 64]], offset=0, scalar_offset=preg, indirect_dim=0))
```

`scalar_offset` counts in units of the `indirect_dim` stride. The caches are ordinary kernel inputs, updated in
place: the written row and column equal a numpy reference bitwise and nothing else changes. The device can advance
the position itself (`tensor_scalar(p1, p_sb, add, 1)`, then a DMA to a second position tensor); with logical
NeuronCore config 2 both programs execute that increment, so write it to a different tensor (two tensor sets used
in turn), never in place.

## Layout that avoids transposes

- K transposed `[kv_head, d, T]`: a 128-position tile `[128 d, 128 t]` is the stationary operand of QK with q^T
  `[128 d, 2 heads]` moving; scores land as `[128 t, 2]`.
- V natural `[T, kv_head * d]` is the stationary operand of `O^T[d, h] = sum_t V[t, d] P[t, h]` with P moving: no
  transpose of P, and the output lands as `O^T [128 d, 16 heads]`, the stationary layout the o_proj GEMV wants. The
  price is a scatter of 1,024 two-byte elements per layer for the new K.
- Read the cache only over `[0, pos)` and take the current token's K/V from SBUF: the kernel never reads what it
  writes in the same execution, so no read-after-write ordering is needed (zero-initialize the cache once so the
  masked tail is finite).
- Mask with an additive -30000 before the max, exp in fp32 on ScalarE, probabilities rounded to bf16 for PV,
  normalize in fp32 after PV.

## Numbers

| variant | positions read | device total / exec (us) | queued us per execute |
|---|---|---|---|
| static bucket 512 | 512 | 33.3 / 21.3 | 34.1 |
| static bucket 4,608 | 4,608 | 76.7 / 64.2 | 73.9 (18.9 MB, ~300 GB/s on one physical core) |
| static 4,608, hardware DGE on 2 queues | 4,608 | 74.3 / 59.9 | 69.4 |
| dynamic `fori_loop`, 512 positions per iteration | ceil(pos/512) x 512 | - | 48 (pos 0) to 226 (pos 4,095) |

Issue the cache loads first in program order (they do not depend on q/k/v): bucket 512 52.6 -> 34.1 us, bucket
4,608 87.8 -> 73.9 us; each engine runs its instruction stream in order. Use static buckets chosen by the host
(it knows the step count): the dynamic loop is correct but about 10 us per iteration slower, because one
iteration's DMA and compute do not overlap the next.

## What works inside `nl.fori_loop` (measured)

| inside the body | result |
|---|---|
| DMA with a dynamic offset on the HBM side into a fixed SBUF tile | works |
| `tensor_copy` to or from SBUF at a dynamic offset | works |
| SBUF accumulation across iterations | works |
| DMA into SBUF at a dynamic offset | compiles; `nrt_load` fails (`TDRV:calculate_max_reg_offset OOB access detected in NEFF`) |
| matmul operand at a dynamic SBUF offset | compile error `NCC_IBIR040` (the whole dimension is assumed) |
| a PSUM tile accumulated across iterations | compile error `NCC_IBIR092` (live-in/live-out not allowed) |

## Pitfalls

- A Python process that ran a kernel through nki's own runner keeps its NeuronCore until it exits; a C++ runner on
  the same core then fails with `cores busy, ret=-16`. Compile in Python, run in C++.
- `nl.power` is not a valid `tensor_scalar` op; rsqrt on GpSimd (`tensor_scalar(op0=nl.rsqrt, engine=gpsimd)`) is
  more precise than the ScalarE table.
- A small NEFF has 12 to 23 us of fixed device time around its body (semaphores, activation-table load): another
  reason to fuse a whole layer or step into one NEFF.
