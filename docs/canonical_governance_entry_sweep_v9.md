# Canonical Governance Entry Sweep v9

`ucf-ops governance-entry-sweep` is the final v9 consolidation proof that canonical governance surfaces consume a **single authoritative entrypoint**:

1. `AppliedSupportedSetContextV1`
2. `CanonicalGovernanceEntryV1`

The sweep emits `CanonicalGovernanceEntryAuthorityV2` and per-surface status records.

## Command

```bash
cargo run -p ucf-ops -- governance-entry-sweep --out ./out/governance_entry_sweep.json
```

## What this proves

- Canonical surfaces are checked against the same applied set digest and applied context digest.
- Canonical governance entry is required before downstream canonical consumers are evaluated.
- Secondary governance entry paths are flagged as mismatch/legacy and cannot produce PASS authority.

## Covered canonical surfaces

- `ActiveReviewSnapshot`
- `OperatorSignoff`
- `OperatorReviewPacket`
- `OperatorWorkflowChain`
- `InteropConsistencyMatrix`
- `GovernanceSurfacesCheck`

## Why secondary entry paths are disallowed

Secondary entry starts from raw evidence/signoff/export internals/gate artifacts can create parallel truths.
In v9 canonical flows fail closed when canonical governance entry authority is missing or mismatched.

## Mismatch categories

- `SURFACE_SKIPPED_CANONICAL_GOVERNANCE_ENTRY`
- `SURFACE_USED_SECONDARY_GOVERNANCE_ENTRY`
- `GOVERNANCE_ENTRY_SCOPE_MISMATCH`
- `GOVERNANCE_ENTRY_POLICY_MISMATCH`
- `LEGACY_GOVERNANCE_ENTRY_PRESENT`
