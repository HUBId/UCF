# v0 Backend Traits (hardware-neutral, offline)

This document defines the v0 canonical backend abstraction layer for deterministic offline compute onboarding.

## Traits (V1)

Implemented in `runtime/ucf-compute/src/stage_v1.rs`.

- `LlmBackendV1`
- `WorldPredictorV1`
- `SaeExtractorV1`
- `SsmKernelV1`
- `LfmBackendV1`

All traits expose:
- `contract_version() -> u16` (currently `1`)
- `backend_id() -> u16`
- stage method (`infer`/`step`) returning `Result<..., StageError>`.

## Stable stage error codes

`StageErrorCode` provides stable, string-addressable codes:

- `BACKEND_DISABLED`
- `TIMEOUT`
- `BUDGET_EXCEEDED`
- `VALIDATION_FAILED`
- `INTERNAL`

## Deterministic CPU stubs

The following deterministic stubs are provided:

- `CpuLlmStubV1`
- `CpuWorldStubV1`
- `CpuSaeStubV1`
- `CpuSsmStubV1`
- `CpuLfmStubV1`

Rules:
- No RNG.
- Digest-driven mapping only.
- Bounded output sizes (`MAX_TEXT_BYTES`, `MAX_SAE_SPIKES`, bounded metadata).

## Uniform timeout/budget semantics

`StageRunner` enforces:
- work-unit charge through `WorkMeter`
- wall-clock timeout check
- deterministic degraded fallback behavior via returned `(output, Option<StageViolationRecord>)`

## ESS summary records (bounded)

`domains/ucf-ess/src/v1/record.rs` now includes:

- `WorldSummaryRecord`
- `SaeSummaryRecord`
- `SsmSummaryRecord`
- `LfmSummaryRecord` (existing)
- `LlmSummaryRecord`

Each includes contract/backend IDs and digest prefixes for policy/evidence linkage.

## Extending to Candle/Burn backends

Future backends can implement the same `*V1` traits and be wired without architecture changes:
- preserve bounded I/O,
- preserve stable errors,
- preserve deterministic digest contracts,
- keep `contract_version` stable unless schema-breaking changes are required.
