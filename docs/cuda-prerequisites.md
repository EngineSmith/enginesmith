# CUDA prerequisite support

This platform module has CPU tests. It supplies no inference
engine, built-in reference, model validation or GPU performance claim.

Select the GPUs with exact full UUIDs in `ENGINESMITH_CUDA_UUIDS`; the selection
must be within any inherited `CUDA_VISIBLE_DEVICES` restriction. Both evaluation
arms receive all selected UUIDs, in exactly the same order, with
`CUDA_DEVICE_ORDER=PCI_BUS_ID`. Numeric `CUDA_VISIBLE_DEVICES` values are accepted
only with that ordering explicitly set and refer to PCI order. Empty visibility,
unknown or repeated identifiers, UUID abbreviations and MIG are refused.

Configure tensor parallelism in the external reference command itself. The current
task schema does not accept `reference.tensor_parallel > 1` with an external
command. All selected devices remain visible to both arms even with the default
task-level degree of one; visibility alone does not implement or prove sharding.
Record the rank-to-UUID mapping in both reference and candidate evidence.

`doctor` requires a compatible NVIDIA driver, `nvcc` and a C++ compiler. The memory
budget is 85% of the smallest selected GPU's free memory times the GPU count,
assuming balanced sharding. This estimate does not prove shard, workspace or KV
cache fit. The watchdog checks every selected GPU (critical above 98% used), host
available memory (critical below 3%), and treats unavailable GPU telemetry as
critical. It can stop only process groups registered with the harness.

The external reference must implement protocol 1: hello, generation, continuation,
teacher-forced judge with four-field `[step, margin, reference_token, rank]` rows,
and memory stats. Validate stock-generation parity, split and chained continuation
parity, self-judge, repeated-token rejection, reference against itself, and all
claimed long lengths on the actual GPUs. Pin the implementation and preserve the
checkpoint's published precision. None of these checks follow from CPU tests.

The native check reads ELF metadata without executing the candidate. It accepts
the architecture's glibc loader only at standard system library locations, and
checks direct dependency names against the table in [libraries.md](libraries.md).
It does not verify the contents of those libraries, transitive dependencies,
statically embedded code or `dlopen` calls. NCCL remains provisionally allowed
under the repository's existing collective-communication policy; this addition
does not settle that policy question.
