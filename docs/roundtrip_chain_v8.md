# Roundtrip Chain v8

`CanonicalRoundTripChainV1` is the end-to-end continuity proof for operator governance/readiness state to final export bundle artifacts.

## What it proves

A PASS means the same canonical authority inputs are continuous across:

1. `CanonicalGovernanceEntryV1`
2. `CanonicalReadinessSpineV1`
3. `OperatorReviewPacketV1`
4. `OperatorSignoffDecisionV1`
5. `OperatorWorkflowChainV1`
6. `OperatorExportAuthorityChainV1`
7. `CanonicalBundleSpineV1`

and reconstructed from the provided bundle without drift.

## How this differs from other checks

- `exports roundtrip-check`: validates bundle-local manifest/context roundtrip consistency.
- `exports bundle-spine-check`: validates canonical bundle spine continuity and governance/readiness references.
- `operator roundtrip-chain-check`: validates full operator->export->bundle continuity against current authoritative operator state.

## Canonical sequence

1. `cargo run -p ucf-ops -- operator workflow --latest --out ./out/operator_workflow_chain.json`
2. `cargo run -p ucf-ops -- operator export-chain-check --out ./out/operator_export_chain_check.json`
3. Build export bundle (`repro_pack` or `bugkit build`).
4. `cargo run -p ucf-ops -- operator roundtrip-chain-check --bundle <bundle.zip> --out ./out/operator_roundtrip_chain_check.json`

## Main mismatch categories

- `ROUNDTRIP_CHAIN_GOVERNANCE_ENTRY_MISMATCH`
- `ROUNDTRIP_CHAIN_READINESS_SPINE_MISMATCH`
- `ROUNDTRIP_CHAIN_WORKFLOW_MISMATCH`
- `ROUNDTRIP_CHAIN_SIGNOFF_MISMATCH`
- `ROUNDTRIP_CHAIN_REVIEW_PACKET_MISMATCH`

## Final continuity authority (v9)
Use `cargo run -p ucf-ops -- continuity-authority-check --bundle <path> --out ./out/continuity_authority_check.json` as final top-level proof that this surface aligns with canonical governance/readiness/operator/export/bundle continuity.


## v10 Continuity Position

`CanonicalRoundTripChainV1` ist ab v10 ein **SUBORDINATE_CONTINUITY_CONTRIBUTOR**.
Finale PASS/FAIL-Top-Level-Autorität erfolgt ausschließlich über `final-continuity-sweep`.
