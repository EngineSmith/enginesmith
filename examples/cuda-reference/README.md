# Experimental CUDA reference conformance

This draft example reuses EngineSmith's existing SGLang protocol implementation on **one CUDA GPU**, with stock
`sglang==0.5.22`, and replaces the AMD device-memory observer with the CUDA platform observer. It is not registered
as a built-in reference and does not change a judging file, default or evaluation version. It is not yet a
validated reference: the tests, results and review described below are required before registration.

The initial roadmap proposed vLLM. This experiment uses SGLang because the repository already has an explicit
generation/continuation/teacher-forcing implementation for it; reuse reduces the amount of new numerical judging
code. This is an implementation choice for correctness validation, not a claim that SGLang is the right speed
baseline for every model. The existing multi-node serving baseline remains a separate comparison.

In an isolated environment containing EngineSmith and the pinned, unmodified CUDA SGLang distribution, select
exactly one idle GPU through the platform's supported device selection. Do not run this against a live benchmark
GPU. Download a small supported dense model separately, record its revision and file hashes, then run:

```sh
python3 examples/cuda-reference/validate.py --model models/test-model --out conformance.json
```

This starts a stock offline engine and the wrapper sequentially, compares their generated tokens at the documented
control lengths, checks a split continuation and chained one-token continuations, self-judges the split stream,
and checks that a deliberately wrong stream has early, low-ranked mismatches. It retains failures; it does not
loosen the gates to hide batch/route-dependent differences. Custom processors are enabled for both stock and
wrapper, and generation requests carry no processor. Teacher forcing is the existing SGLang implementation.

`sglang_cuda.py --model MODEL` exposes the protocol for independent experiments. It checks the exact SGLang
version and one-device selection. Memory stats cover the selected device, including unrelated allocations;
the peak is sampled after requests, not a per-kernel allocator high-water mark. Missing telemetry is null.

## Required evidence before a built-in CUDA reference

- Stock generation parity at the control lengths, the actual runtime's chunk boundaries and declared long lengths.
- Split/chained parity, self-judge agreement, wrong-token rank behavior, and repeated deterministic generation.
- Formal reference-vs-reference paired evaluation near 1.0, with separate initialization/warmup and raw results.
- `prepare` at declared lengths and a small-model engine search finishing its final evaluation.
- Model/checkpoint/toolchain versions and numerical settings. A dense model smoke does not validate a large MoE,
  quantized checkpoint or multi-GPU deployment.
- Maintainer review of [#30](https://github.com/EngineSmith/enginesmith/issues/30), judging-zone registration,
  library version provenance and an evaluation version coordinated with other pending judging PRs.

CPU tests exercise the conformance control flow with a toy reference and check the wrapper's version/device/stats
boundaries. They are not model/CUDA correctness evidence. No stock library is patched or monkey-patched by the
example, and no GPU speed or accuracy number is claimed. The formal timing and long-context fields remain null
until those checks are actually run. Do not use this unvalidated example as the authority for accepting candidates.
