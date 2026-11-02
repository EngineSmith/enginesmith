# Decode GEMV kernels (bf16 and quantized)

Purpose: how to write the matrix-vector kernels that stream the weights in batch-1 decode on gfx942: 16-byte loads,
lane groups per row, rows per wave, split-K and when it is allowed, weight relayout, compile-time shapes, the
bandwidth to judge them against, and what changes if the checkpoint is quantized. Read it before writing or tuning
any decode GEMV.

No GEMV has been measured in an engine on this platform yet. The bandwidth ceiling is from `README.md` (~4.2-4.4
TB/s on multi-GB streams, 3.6 TB/s on a single 64 MB kernel); every layout below is a hypothesis to measure.

## Know the size of the problem

A batch-1 decode GEMV reads its weight matrix once and does 2 FLOPs per weight: at ~4.4 TB/s it needs ~4.4 TFLOP/s
of fp32 FMAs, a small fraction of the device's vector throughput, so it is bandwidth- and latency-bound, never
compute-bound. What decides its speed is how many bytes are in flight and how much fixed cost each kernel pays.

Example (Qwen3-0.6B bf16, estimates at 4.4 TB/s):

| projection | shape (N x K) | MB | streaming time |
|---|---|---:|---:|
| Q, K, V (concatenated) | 4096 x 1024 | 8.39 | 1.9 us |
| O | 1024 x 2048 | 4.19 | 0.95 us |
| gate + up (concatenated) | 6144 x 1024 | 12.58 | 2.9 us |
| down | 1024 x 3072 | 6.29 | 1.4 us |
| LM head (tied) | 151936 x 1024 | 311.2 | 71 us |

Every layer GEMV of a small model streams for 1-3 us, the same order as the measured ~1.5-1.9 us per launch and the
~3 us fixed cost estimated for a 64 MB streaming kernel (`README.md`). Two consequences:

- A separate kernel per projection cannot run near the bandwidth ceiling: its ramp-up and drain are as long as its
  work. Bytes per kernel are the lever; fewer, larger kernels (concatenated projections, fused stages, a persistent
  kernel that streams across stages: `kernels-fusion.md`) come first.
- Little's law (estimate): sustaining 4.4 TB/s with a loaded memory latency of 1-2 us needs ~4-9 MB in flight. One
  projection is about that size in total. Measure bandwidth as a function of bytes per kernel (1, 2, 4, 8, 16, 64 MB)
  with your GEMV before choosing a design; that curve is this platform's most important missing number.

## The kernel layout (start here)

Hypothesis to measure: a group of lanes per output row, 16-byte loads, rows packed into a wave.

```cpp
// y[N] = W[N, K] . x[K], bf16 in, fp32 accumulate, bf16 out. K, LANES compile-time.
template <int K, int LANES>                       // LANES lanes per row: 16 -> 4 rows per wave
__device__ void gemv_rows(const uint4* __restrict__ w, const uint4* __restrict__ x, uint16_t* __restrict__ y,
                          int n) {
    constexpr int CHUNKS = K / 8;                 // 8 bf16 per 16-byte chunk
    const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
    const int row = (blockIdx.x * (blockDim.x / 64) + wave) * (64 / LANES) + lane / LANES;
    const int l = lane % LANES;
    if (row >= n) return;                         // a whole lane group leaves together
    const uint4* wr = w + (size_t)row * CHUNKS;
    float acc = 0.0f;
    #pragma unroll
    for (int c = l; c < CHUNKS; c += LANES) {     // lane l owns chunks l, l + LANES, ...
        const uint4 a = wr[c], b = x[c];
        acc = fmaf(bf16_lo(a.x), bf16_lo(b.x), acc); acc = fmaf(bf16_hi(a.x), bf16_hi(b.x), acc);
        acc = fmaf(bf16_lo(a.y), bf16_lo(b.y), acc); acc = fmaf(bf16_hi(a.y), bf16_hi(b.y), acc);
        acc = fmaf(bf16_lo(a.z), bf16_lo(b.z), acc); acc = fmaf(bf16_hi(a.z), bf16_hi(b.z), acc);
        acc = fmaf(bf16_lo(a.w), bf16_lo(b.w), acc); acc = fmaf(bf16_hi(a.w), bf16_hi(b.w), acc);
    }
    #pragma unroll
    for (int off = LANES / 2; off > 0; off >>= 1) acc += __shfl_xor(acc, off, LANES);
    if (l == 0) y[row] = f32_to_bf16(acc);
}
extern "C" __global__ __launch_bounds__(256) void gemv_bf16_k1024(const uint4* w, const uint4* x, uint16_t* y, int n)
{ gemv_rows<1024, 16>(w, x, y, n); }
```

- **Explicit `fmaf` chains** fix the rounding of every partial sum regardless of the contraction flag (`README.md`,
  rule 4). A plain `acc += a * b` is contracted or not depending on flags and context.
- **Lanes per row.** With K = 1024 a row is 128 chunks: 16 lanes take 8 loads each, 64 lanes take 2. Fewer lanes
  per row give each lane more independent loads in flight and a cheaper reduction (inside a 16-lane DPP row);
  more lanes per row suit long rows (down projection, K = 3072). Choose per shape, by measurement.
- **The activation vector** (2-6 KB) is read by every row; it stays in L1/L2. Loading it into LDS once per block is
  an alternative to measure, not a default.
- **Grid.** Enough blocks to fill 304 CUs several times over, or a persistent grid that walks rows (`kernels-fusion.md`).
  Example (Qwen3-0.6B): the O projection has 1024 rows; at 16 rows per 256-thread block that is 64 blocks, a fifth
  of the CUs: such a kernel is latency-bound however well it is written. More, smaller blocks (64 threads, 4 rows) or
  split-K (below) restore parallelism.
- **Compile-time shapes.** One instantiation per (K, lanes per row), so loop bounds, strides and the unroll are
  constants. On Metal this bought ~3% with unchanged bits (`backends/metal-rust/kernels-gemv-quant.md`); here it is a
  hypothesis. With clang without fast math, unrolling does not reorder a float chain; still check bits.

A row's bits are a function of: lanes per row, which chunks each lane owns, the order of each lane's FMAs, the
reduction tree, and the final rounding. Fix them at bring-up and never change them for speed afterwards; any change
to them is a correctness change to be checked against the reference (`references/correctness.md`).

## Split-K

When N is small (1024 rows on 304 CUs), split each row's K range over several waves or blocks: each writes an fp32
partial, and a fixed-order combine (a second kernel, the next kernel's prologue, or the last-finishing block)
sums the partials in index order and rounds once. No float `atomicAdd` (`README.md`, rule 4). The split count is a
constant of the shape. Split-K changes the summation tree, so choose it at bring-up and keep it.

## Weight relayout

The checkpoint stores `[N, K]` row-major, which the layout above reads directly. Relayouts worth measuring (once at
load, bit-exact, `runtime.md`):

- **Concatenate** projections that read the same input: Q, K, V into one `[4096, 1024]`; gate and up into one
  `[6144, 1024]`. One kernel instead of two or three; per-row arithmetic unchanged.
- **Interleave gate and up rows** (row `2i` gate `i`, row `2i + 1` up `i`) so the wave that computes both can apply
  SiLU and the product in its epilogue (`kernels-fusion.md`).
- **Tile for the lane walk**: reorder the chunks of a group of rows so that each wave's loads for one step are one
  contiguous 1 KiB, if the assembly or counters show poor coalescing (row-major with 16 lanes per row already reads
  256 contiguous bytes per row group; measure before changing).

## Bandwidth: what to measure and how

- **The curve first:** effective GB/s of your GEMV at each production shape, alone and inside the decode chain
  (`profiling.md`), against bytes per kernel. A micro-benchmark that re-reads one matrix may be served from the 4 MB
  L2 per XCD or the 256 MB Infinity Cache; cycle through more distinct weight copies than 256 MB (the whole model's
  1.19 GB of weights already does) to measure HBM.
- **Floor per kernel:** `bytes / measured bandwidth + the measured fixed cost of a kernel of that grid`. Kernels near
  it are done; kernels far above it with few bytes are latency or launch problems that fusion fixes.
- **Bound probes** (wrong values, timing only): no math, loads from one address, no reduction. They tell you how much
  a mechanism could at most buy.

## Cache hints (hypothesis, not measured)

The whole weight set (1.19 GB for Qwen3-0.6B) cycles once per token through a 256 MB Infinity Cache, so with LRU-like
replacement nothing survives to the next token. If non-temporal loads bypass allocation in that cache (not
documented for gfx942 as far as this backend knows), streaming most weights non-temporally might keep a fixed
~200 MB subset resident and raise effective bandwidth. The 256 MB point of the streaming benchmark (4796 vs 4193-4376
GB/s) hints at what a cache-resident stream gains. Measure before building on it.

## Quantized checkpoints

If the model is quantized (int4/int8 weights with scales, fp8, MXFP4, GGUF-style blocks), the same layout holds with
a dequantizing inner loop; bytes per token shrink and the kernel's ALU work grows, so check that it stays
bandwidth-bound.

- **Copy the reference's dequantization formula and rounding points**, not a textbook one: whether it dequantizes
  each weight to bf16 first or applies `scale * sum(q * x)` per group in fp32 changes the result
  (`references/correctness.md`). Read the kernel SGLang dispatches for that format at the pinned version.
- **Never pad or requantize groups** to fit a kernel; dispatch a fast path only when the packing contract holds.
- **fp8 on gfx942** uses the FNUZ encodings (`e4m3fnuz`, `e5m2fnuz`), which differ from the OCP `e4m3fn` used on
  NVIDIA and newer AMD parts (bias and NaN encoding); a checkpoint stored in OCP fp8 must be converted exactly as the
  reference converts it on this device. gfx942 has fp8 conversion and MFMA instructions (vendor ISA documentation;
  not measured).
- **Sources to copy from** (check each licence at the commit you copy): vLLM's ROCm skinny GEMMs and quantized
  kernels (Apache-2.0), llama.cpp's HIP build of its CUDA quantized matrix-vector kernels (MIT), aiter's and CK's
  quantized GEMMs (MIT). Copy the HIP source and adapt it to your shapes; record the provenance.

## Rejected variants

None measured on this platform yet. Record each attempt in `counterexamples.md` with its numbers.

## Checklist for a new decode GEMV

1. Accumulate in fp32 with an explicit FMA order; round the output to bf16 once, with the shared helper.
2. Fix lanes per row, chunk-to-lane order, split-K and the reduction tree at bring-up; record them.
3. Compile-time K and lanes per row for every decode shape; the wrapper checks shape, dtype and 16-byte alignment.
4. Check the kernel against a scalar fp32 reference (tolerance) and against its parent (bitwise) on random inputs,
   including the last row and the last chunk.
5. Read `-Rpass-analysis=kernel-resource-usage` (no spills) and the assembly (`global_load_dwordx4`).
6. Time it inside the engine, interleaved against the parent (`references/measurement.md`).
