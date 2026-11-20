# Diagnostic service comparisons

`service-compare` checks that two experiment declarations describe comparable conditions, displays paired output
throughput ratios, and reports missing evidence, latency-limit violations and errors/cancellations. It does not
accept a candidate, alter task gates, or run inside `enginesmith run`. It is the diagnostic first slice of
[#31](https://github.com/EngineSmith/enginesmith/issues/31) and [#32](https://github.com/EngineSmith/enginesmith/issues/32).

```sh
uv run enginesmith service-compare baseline.json candidate.json \
  --max-ttft-ms 2000 --max-itl-ms 20 --max-error-fraction 0.01
```

The thresholds above are illustrative user choices, not recommended model SLOs or new EngineSmith judging defaults.
The input is an explicitly versioned JSON object. Each field below is required; measurements unavailable in a
source should be null, never zero. Do not fill placeholder hashes with invented values in a real experiment.

```json
{
  "schema": 1,
  "identity": {
    "model_revision": "checkpoint-commit",
    "weights_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "precision": "checkpoint-fp8",
    "kv_precision": "fp8",
    "hardware": "gpu-model",
    "gpu_count": 16,
    "topology": "P8+D8",
    "resource_budget": {"cpu_cores": 64, "host_memory_gb": 512, "nic_count": 8},
    "workload_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "client_version": "1.0.14",
    "dataset_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    "sampling": {"temperature": 1, "top_p": 1},
    "cache_policy": "independent-warmup",
    "concurrency": 2048,
    "duration_s": 3600
  },
  "speculation": {
    "mode": "actual",
    "draft_revision": "draft-checkpoint-commit",
    "draft_length": 5,
    "synthetic_rule": null
  },
  "evidence": {
    "runtime_verified": false,
    "runtime_artifact_sha256": null,
    "quality": null
  },
  "measurements": [
    {"pair_id": "pair-01", "output_tokens_per_s": 100, "ttft_p99_ms": 1000, "itl_p99_ms": 10,
     "requests": 1000, "failed_requests": 0, "cancelled_requests": 0}
  ]
}
```

The example is fabricated schema data, not a measured result. Weights, workload and dataset digests should come
from canonical manifests of the actual files/configuration. Resource-budget objects must include all resources
that matter to the comparison; exact equality alone cannot establish that an incomplete declaration is adequate.

`speculation.mode` is `none`, `actual`, or `synthetic`. Disabled speculation has null draft revision/rule and zero
draft length. Enabled modes require a pinned draft revision and positive length. Synthetic acceptance also needs
its rule parameters (including seed/distribution or schedule when relevant); a target mean alone may not identify
the algorithm. Any difference in these fields prevents a throughput-ratio report. The strict policy compares
optimization within the same algorithm; it does not evaluate changing from non-speculative to speculative decoding.

Quality is null or `{ "mode": "actual", "passed": true, "artifact_sha256": "..." }`, with a real SHA256.
Mode `none` is also valid for ordinary generation quality. Synthetic acceptance cannot supply quality evidence.
Average accepted length is a performance characteristic, not coding-task or answer correctness. A quality summary
must be traceable to the same checkpoint/precision settings through the underlying evidence.

Every measurement names a unique `pair_id`. Both inputs must have the same pair IDs. Record independently warmed
AB/BA or randomized paired runs under one workload contract. At least three repetitions are required to avoid a
"too few repetitions" limitation in this diagnostic, but three runs are not a statistical guarantee. The tool
reports individual throughput ratios and their median/min/max; these are observed values, not confidence bounds.
It reports latency limits for every repeat instead of pooling percentiles. Raw request data is required for a
different aggregation or inference. Held-out workload validation is not implemented in this command.

`requests` means total attempted requests; `failed_requests` and `cancelled_requests` are disjoint subsets.
The error fraction includes both. This differs from an export's count of successful/valid requests: reconcile
raw records before filling these fields. The tool does not infer counts from unrelated throughput totals.

Missing runtime verification, missing/failed quality, unavailable metrics, insufficient repeats and limit
violations remain visible in `limitations`. Identity/speculation mismatches set `compatible: false` and suppress
ratios. Even a compatible result with no limitations has `acceptance: null`: digest strings and verification
flags are user declarations here, not proof that the referenced artifacts were opened and checked. The future
judging integration must verify artifact contents, baseline self-parity, statistical rules and held-out evidence
before it can make an automatic candidate-acceptance decision.
