# Proposal: natural-text development evaluation for speculative decoding

Status: phase 1 implemented as opt-in core-only diagnostics; tracked in issue #21. See
[usage and validation](../natural-text-development.md). Formal judging rules are unchanged. Phases 2 and 3 remain proposals.

## Problem and source evidence

The current development and performance workloads are deterministic synthetic token-ID sequences. `workload.prompt()` generates IDs with an arithmetic formula; it does not tokenize natural text. The `dataset` command exports that same workload. `dev_eval.run()` evaluates its public prefix, while formal evaluation chooses separate nonce and held-out windows.

Speculative decoding proposes a model-generated continuation after that context. It does not predict the next value of the synthetic input generator. Synthetic contexts may induce repetitive or otherwise atypical continuations, so draft acceptance and decode throughput on them need not represent natural-text workloads. The evaluator's source comments explicitly describe repetitive output on these prompts. This establishes a coverage gap, not a measured natural-text regression or speedup.

The optional task-accuracy path tokenizes natural-language examples, but it does not provide a development performance suite with controlled context lengths and speculative-decoding accounting. It should remain a separate quality measurement.

## Recommended implementation sequence

### 1. Optional natural-text development evaluation

Provide an explicitly selected, public development corpus alongside the existing synthetic checks. This is diagnostic feedback first, not a replacement for the formal score or an implicit extra evaluation after every coding session.

Implemented interface, after explicit offline tokenizer preparation:

```text
dev-eval --text-dataset PREPARED.jsonl --samples-per-length 3 --dataset-seed 0
```

Each record contains a stable document ID and natural text. Any bundled corpus needs an explicit redistribution license, source revision and content checksum. Local corpora remain local. No automatic downloads or model-weight changes are part of this feature.

Prepare samples before timing with the task's fixed tokenizer. Record the corpus checksum, tokenizer fingerprint, sampling version, seed, document hash, token offset, exact prompt-token hash and template policy. The initial raw-text mode uses no chat template or added special tokens; conversational formatting requires a separately declared policy.

For every selected task length, select a configured number of distinct documents that contain enough tokens and take contiguous windows. Do not repeat a short passage, pad it with random tokens, or silently reduce the length. Insufficient coverage must fail clearly. Keep the selection reproducible. A later held-out suite must split by document before choosing windows to avoid development/final overlap.

Reference and candidate must receive identical token IDs for each sample and paired round. A differing generated stream must be judged against that exact natural-text prompt, not a regenerated synthetic prompt. Preserve synthetic boundary controls, engine isolation, cache/reset rules, cooldown, warmup, existing correctness gates and output length. Prefill includes the first output token; Decode covers the remaining requested tokens.

Report per-sample and per-length Prefill and Decode timings and paired ratios, sample counts, raw repetitions and correctness outcomes. Label the workload and provenance explicitly. Keep these results separate from synthetic development scores and formal incumbent selection. Repeated timings of one sample estimate timing variability, not diversity of input content.

Natural-text evaluation consumes the existing coding-session time and uses the same device lock and process registry. It must not extend a deadline, refund a slot, or start automatically in an existing run. Tokenization must stay outside timed engine requests. Keep the harness standard-library-only; agree on an isolated tokenizer-preparation interface or the appropriate optional dependency boundary before implementation.

### 2. Optional speculative-decoding telemetry

Agree on a capability-negotiated schema before changing the protocol or reference adapters. Desired counters are draft tokens proposed, draft tokens accepted, target verification invocations, generated tokens and tokens delivered to the caller. Accepted draft tokens exclude target-only bonus or correction tokens. Missing counters are null with a reason; acceptance cannot be reconstructed from throughput alone.

Distinguish a host request from a target verification step. Account explicitly for tokens carried across requests in a speculative FIFO, especially across the Prefill/Decode boundary. Per-step IDs and counter deltas must prevent double counting or attributing buffered tokens to fresh verification work.

Report acceptance, tokens per verification step and draft/verification timing where supported. GPU timing must have a documented synchronization boundary and measured instrumentation overhead. Keep uninstrumented end-to-end timing as the primary performance measurement. Do not patch the stock reference library or invent unavailable measurements.

### 3. Consider formal natural-text evaluation separately

Only after maintainer agreement, define corpus coverage, document-disjoint public/held-out splits, sampling, length weights and aggregation for an opt-in formal workload. Create the required evaluation version and judging fingerprint update for any judging-zone change, including shared evaluator changes made initially for development mode.

Existing runs keep their frozen workload, gates, score, model, agent and budgets. Reports under different workloads or evaluation versions are not directly comparable. Do not combine natural-text and synthetic scores into one headline result without an explicitly agreed objective.

## Acceptance criteria for implementation

- The existing synthetic suite and its behavior are preserved.
- Identical corpus, tokenizer and seed reproduce identical prompt tokens and metadata.
- Insufficient long-document coverage is rejected rather than padded or recycled.
- Paired reference/candidate execution and teacher-forced replay use the identical selected prompt.
- Natural-text results are clearly separated from formal candidate acceptance.
- CPU tests cover malformed corpora, invalid token IDs, duplicate IDs, coverage, deterministic sampling and mixed control/timed cases with fake engines.
- Capability tests cover unsupported telemetry and correct FIFO/bonus-token accounting before telemetry is advertised.
- No performance claim is made until paired GPU measurements on the declared natural-text suite are available.

## Repository process and decisions requested

`CONTRIBUTING.md` requires an issue before a core feature and maintainer agreement before judging changes. `workload.py`, `evaluate.py` and `protocol.py` are in the judging zone; `dev_eval.py` and the CLI are in the core zone. This proposal is documentation in the open zone.

Issue #21 tracks the diagnostic-first implementation and its isolated tokenizer-preparation boundary. The shared
judging files are reused without edits, and no evaluation version changes. Speculative telemetry and any formal
natural-text workload still require separate agreement and implementation.
