# Correctness

Purpose: what "matching the reference" means, how to reproduce the reference's rounding, the rule that keeps every
optimization bitwise identical to its parent, and the checks that catch the defects a quick comparison misses. Read
it before any change to arithmetic or scheduling, when a check fails, and when you plan how to verify a change. The
task contract defines the correctness checks you are judged by and their tolerances; it wins over this file.

Terms: a dispatch is one kernel launch; a threadgroup is a group of threads that share on-chip memory (a thread
block in CUDA); an L=1 step is a forward pass over a single token.

## Three standards, not one

- **Against the reference: the same greedy tokens, except at near-ties.** An engine with its own kernels is in
  general not bitwise equal to the reference, because the reference's own output moves when you change its
  partition or a reduction order, and with 16-bit logits that movement is as large as many top-2 margins. Matching
  means: along a teacher-forced path your greedy token equals the reference's at every step except where the
  reference's top two logits are (nearly) tied, the logits rows are close, and no difference is systematic (a
  wrong formula, a wrong rounding route, a wrong partition). Aim for zero mismatches.
- **Against the parent: bitwise identical.** Once a correct baseline exists, every optimization must produce the
  same bytes (tokens, logits, state) for every request sequence. You control both sides, so this is achievable and
  checkable exactly. A change that alters one byte is a correctness change: treat it as phase 1 work and check it
  against the reference again.
- **Against itself: deterministic.** The same request sequence gives the same bytes in any process, after any
  history, however the request is split (see the self-check recipe below).

Example (another MoE model under its MLX reference, measured): the reference run with a different partition
(prefill chunk or decode block size) flipped 1.3-4.6% of its own teacher-forced argmaxes. Example (LFM2.5 on an M5
Max, measured): a finished engine had 186 token mismatches in 4160 compared steps on held-out prompts (4.5%)
against mlx-lm.

## Reproduce the reference's rounding

The formula is not the result. The reference's outputs come from its specific kernels and the order it runs them:

- **Kernels.** Accumulation dtype, reduction tree, every point where a value is rounded to 16 bits, the exact
  dequantization expression, whether a norm keeps an F32 statistic. A quantized dot may be computed as
  `scale * sum(q*x) + bias * sum(x)` in F32 or by rounding each dequantized weight first; these differ.
- **Per-shape routes.** The reference picks kernels by shape, dtype and device: a matrix-vector or a matrix-matrix
  kernel for small M, a single-pass or a two-pass attention kernel above some key count, a MoE path that sorts
  tokens by expert above some token count, fused or unfused paths. Each route rounds differently; reproduce each
  one with the same threshold.
- **The prefill partition.** Chunk sizes, where L=1 steps begin, how the last prompt token is processed, what a
  continuation does with its pending token. Take it from the reference's driving loop, not its documentation.
  Example (LFM2.5, measured): feeding a prompt one token at a time instead of in the reference's chunks moved the
  last logits row by up to 6.6 and flipped its argmax at 7 of 18 lengths.
- **Settings.** Reduced-precision matrix modes, fast-math flags, the framework version. Example (LFM2.5,
  measured): toggling MLX's TF32 setting changed the generated tokens at 6 of 19 prompt lengths; the default is
  part of the reference's identity.
- **Casts between operators.** When you fuse two operators, keep the reference's intermediate rounding (for example
  a projection sum rounded to BF16 before the activation) in registers. Removing it changes the function.

How: read the reference's dispatch code at the pinned version and list every route with its condition; a code
comment or another engine's claim about a route is not evidence. Port the reference's kernels for each route
(copying and adapting them is the fastest way to the same rounding). Test prompts on both sides of every
threshold. Record the routes in the model card (section 7).

## Teacher-forced and free-running checks

- **Teacher-forced (the primary check).** Feed the reference's sequence (prompt plus the reference's generated
  tokens) through your forward pass and compare row by row. Per step record: your argmax; whether it is a mismatch
  (a useful definition: the reference's logit at your token is below its row maximum, so an exact tie agrees); the
  reference's top-2 margin; the largest absolute logit difference. Every step starts from the same context, so one
  difference does not propagate and each step is an independent sample.
- **Free-running.** Generate greedily and compare with the reference's generation. Report the first divergent step
  and the reference's margin there. After a divergence the two runs continue from different contexts, so later
  steps say nothing about correctness: one near-tie flip can lead to different text or a repetition loop. Never
  count later steps as errors, and never tune for free-running agreement.
- **Telling noise from a defect.** Near-tie noise: rare teacher-forced mismatches, each at a small reference
  margin, spread across lengths. A defect: mismatches at large margins, or clustered at one length, one layer type,
  one route or one side of a chunk boundary, or logits errors that grow with position.

Example (LFM2.5, measured): an engine that put the last prompt token of a 512-token prompt into a 512-row prefill
chunk, where the reference runs it as an L=1 step, mismatched at step 0 (reference gap 0.94). The repair restored
the reference partition and computed the L=1 quantized dot as the reference does (F32 `scale * sum(q*x) +
bias * sum(x)` instead of BF16-rounded weights); teacher-forced mismatches at 256 and 512 tokens went from 3 and 4
to 2 and 2 of 65. The lower prefill speed of the reference partition was accepted for correctness.

## Greedy, ties and near-ties

- Use the reference's exact greedy rule, usually the first maximum (lowest token id) over the full vocabulary of
  the logits as stored (for example BF16 widened to F32). Check whether the reference takes the argmax of the
  logits or of log-probabilities: two distinct 16-bit maxima can round to one log-probability.
- With 16-bit logits exact ties are common. Example (LFM2.5, measured): exact top-2 ties on 8 of 19 65-step greedy
  paths of the reference. Test the tie rule on real paths, not only on synthetic rows.
- A parallel argmax is partition-independent when it uses a total order (value descending, then index ascending;
  -0 equal to +0; NaN never wins), for example as an integer key combined with an integer atomic max.
- Near-ties make the engine sensitive to one-ulp changes: one reordered sum can flip a token many steps later.
  This is why the parent rule below is exact equality, not "close".

## The bitwise-identical rule for optimizations

An optimization may change where and when values are computed, never how. Keep:

- **The same expressions.** Copy arithmetic verbatim. GPU compilers often default to fast-math: they may contract
  `a*b + c` into a fused multiply-add and reassociate sums, and what they do depends on the surrounding code. Where
  the parent's compiled code contracted, write an explicit `fma` so the new context compiles the same; then verify.
- **The same rounding points.** Every 16-bit rounding of the parent stays, even when the value now lives in a
  register instead of memory.
- **The same reduction orders.** The same lanes sum the same elements in the same order, the same simd or tree
  reduction shape, the same order of combining partial results. Do not change lanes per row, how K is split, or
  which lane owns which block.
- **No float atomics** and no other order-dependent accumulation (combine a token's experts in a fixed order).
  Integer atomics that assign slots are safe only when nothing downstream depends on slot order; otherwise use a
  stable prefix sum. No CPU float reduction whose order depends on a thread pool.
- **Partition constants of the shape.** Threadgroup sizes, split counts, tile shapes and combine orders depend on
  the shape only, never on timing, occupancy, load, thread count or earlier requests. No autotuning, no kernel
  chosen by measured time.
- **One path per shape.** The same kernels and partition in every phase: prefill's L=1 steps and decode,
  teacher-forced forward and generate, the first request and the hundredth.
- **Grid-wide synchronization done right.** A last-finishing-threadgroup pattern needs device-scope fences and must
  not depend on threadgroups being co-resident; stress-test it with thousands of back-to-back dispatches.

Examples (LFM2.5, Metal default fast-math, measured): fusing two kernels let the compiler contract `lo*c - hi*s`
differently, so 2 of 409,600 values moved by one BF16 ulp and a token check failed at step 56 of a one-token
prompt; a later version with an explicit `fma` was identical. A compile-time trip count on a 4-term combine loop
changed 78,517 of 327,680 outputs. Loading a norm gain before a barrier instead of inside the loop gave the same
compiler IR, yet the GPU backend reassociated the product and 1 of 64 input sets differed. A fusion that was
identical on random tensors and on four prompts changed tokens on others, among them the 3-, 4-, 2050- and
2051-token prompts. Verify every schedule change on the full prompt set.

## Self-check recipe

Run these on every change that touches arithmetic, scheduling, buffers or state, comparing tokens, logits bytes
and state bytes:

1. **Two fresh processes**, the same request sequence: identical bytes.
2. **After other requests**: in one process, a long request, then every test length once with a reset before each,
   then the first case again: identical to a fresh process.
3. **Split versus one request**: one 65-token generate; a 1-token generate followed by a 64-token continuation; a
   chain of continuations (17, then 16 three times). The common prefixes must be identical.
4. **Checkpoint and restore**: generate 32, checkpoint, generate 1, restore, generate 33 equals one generate of 65,
   and the state bytes after restore equal those before.
5. **Teacher-force your own tokens**: feed your generated tokens through the forward pass; each row's first maximum
   must be your token at that step. This proves forward and generate run the same arithmetic.
6. **Repeat**: run each comparison several times. A difference in one pass out of many is a race, never noise.
7. **Positive control**: run the checker once against a build with one deliberate rounding change and confirm it
   reports differences. A checker that cannot fail proves nothing.

`scripts/same.py` compares a build with its parent on a fixed prompt set in the three request shapes of check 3,
many requests per process (check 2); give it the same build as both arms for check 1. Example (LFM2.5): every
decode optimization of the final round was checked this way on 84 prompts of 1 to 131008 tokens.

## Boundary prompts

Choose prompt lengths at every place the computation changes shape:
- 1-5 tokens (state initialization, convolution windows shorter than the kernel, the first L=1 step);
- powers of two and tile sizes, each with -1 and +1 (15, 16, 17 ... 4095, 4096, 4097);
- the prefill chunk size with -1 and +1, and the lengths that leave a one- or two-token chunk before the final
  token (for a 2048-token chunk: 2047, 2048, 2049, 2050, 2051);
- each route threshold (for example the key count where attention switches kernel), both for prompts and for
  decode steps that cross it during a continuation;
- long prompts up to the task's maximum context, at least once per change that touches attention or the cache;
- random-token prompts with fixed seeds and natural text, so that routing and attention patterns vary.

## Per-layer dump and diff

When an end-to-end check fails and the cause is not obvious:

1. Add an environment-gated dump that writes chosen intermediate tensors (each operator's output, chosen layers,
   chosen positions) to files, never to the protocol output. With the switch off, the build must be bitwise
   identical to the parent.
2. Capture the same tensors from the reference with the same partition (hooks, or running its modules one by one).
3. Diff in the reference's dtype: largest absolute and relative difference, the number of differing elements, and
   the first layer and operator where the difference exceeds the reference's own noise (run the reference with two
   partitions to measure that noise per layer).
4. Compare discrete decisions exactly: selected expert ids, block or top-k selections. A different id with a large
   score margin is a defect; with a tiny margin it is amplified rounding noise.
5. Bisect by operator inside the first bad layer: input norm, projections, position encoding, attention output,
   router logits, combine weights, residual add.

## Typical bugs and their symptoms

- **Missing wait or barrier** between dependent GPU work, or a host read before the GPU finished: results depend on
  timing; often only long requests or a busy machine show it.
- **Scratch reuse**: two live values in one buffer, or a read beyond the live extent (a shorter chunk reads the tail
  of a previous, longer one): wrong only after a longer request.
- **State left by reset**: recurrent or convolution state, the KV tail, position counters, atomic counters,
  "first step" flags: correct in a fresh process, wrong after other requests.
- **Races**: in-place updates that other threadgroups still read, grid-wide patterns without fences: one pass in
  many differs.
- **Uninitialized memory**: fresh allocations often read as zero, so the bug hides in fresh processes.
- **Off by one** at chunk boundaries, the pending-token rule of continuations, position counting after restore.
- **Two paths**: a decode kernel that differs from the prefill's L=1 kernel shows as a teacher-forced-own-tokens
  failure.
- **Unchecked fast paths**: a shape-specialized kernel without an eligibility check (partial quantization groups,
  tails) is silently wrong on the shapes it does not handle.

Platform-specific tools for capturing buffers and checking fences: see `backends/metal-rust/README.md`.
