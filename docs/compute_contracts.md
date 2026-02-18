# Compute Contracts v1

This document defines formal stage contracts for the compute pipeline.

## Contract versioning

- `StageContractVersion::V1` is the initial compatibility version.
- Backends expose `contract_version()` and are checked against a runtime registry.
- Contract versioning is additive; future versions can be introduced without changing v1 behavior.

## Stage invariants (V1)

- **World**
  - `prediction_error`, `surprise`, `state_norm` in `[0,1]`.
  - Serialized output must fit `MAX_STAGE_ENCODED_BYTES`.
- **SAE**
  - `spike_count <= SAE_MAX_SPIKES` and `spikes.len() <= SAE_MAX_SPIKES`.
  - `sparsity`, `energy`, and spike magnitudes in `[0,1]`.
  - `spikes_digest` must match canonical digest of spikes.
  - Serialized size bounded.
- **SSM**
  - `pressure`, `readout`, `state_norm` in `[0,1]`.
  - `readout_digest` must match canonical derivation from readout and state digest.
  - soft envelope warning if pressure jump exceeds configured threshold.
  - serialized size bounded.
- **LFM**
  - key scalar outputs and stage input scalars in `[0,1]`.
  - `liquid_readout_digest` linkage check.
  - serialized size bounded.
- **LLM**
  - response digest must match canonical digest over status/text/token_count/finish_reason.
  - serialized size bounded.

## Evidence chain

- Evidence chain digest is revalidated from canonical encoding.
- Ordering is the canonical order encoded by `EvidenceChain::encode_canonical`.

## Runtime handling

- Hard violations => `ValidationStatus::Degraded`, safe fallback output, no panic.
- Soft violations => `ValidationStatus::Warned`, pipeline continues.
- Violation reasons are persisted as a bounded bitmask (`violation_reason_mask`).

## Persisted summary metadata

Compute summary includes:

- `contract_version`
- `backend_id`
- `validation_status`
- `violation_reason_mask`

These fields are propagated into decision/ESS records.

## Adding V2

1. Add `StageContractVersion::V2`.
2. Add `*ValidatorV2` with explicit invariants/limits.
3. Extend registry mapping stage+backend to supported versions.
4. Keep V1 logic untouched.
5. Add property and integration tests for V2 behavior.
