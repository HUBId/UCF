# Governance Primary Surfaces v6

v6 normalizes governance consumption around two canonical, read-only snapshots:

- `BackendEvidenceSnapshotV1`
- `AggregatedActiveReviewSnapshotV1`

These are the primary governance surfaces used by downstream review/signoff/export and gate-preparation helpers.

## Why

- Reduces semantic drift from duplicating lower-level evidence gathering.
- Keeps review, signoff, export, and gate helpers on the same evidence worldview.
- Preserves deterministic, offline, fail-closed behavior.

## Canonical handoff

`GovernancePrimarySurfacesV1` is the bounded consistency contract produced by validating the two snapshots.

Fields:

- `backend_evidence_snapshot_digest_prefix`
- `active_review_snapshot_digest_prefix`
- `supported_slot_set_digest_prefix`
- `policy_graph_digest_prefix`
- `manifest_digest_prefix`
- `consistency_ok`
- `governance_surfaces_digest`

## Deterministic consistency check

Run:

```bash
cargo run -p ucf-ops -- governance-surfaces-check --out ./out/governance_surfaces_check.json
```

Behavior:

1. Generates/loads backend evidence snapshot.
2. Generates/loads active review snapshot.
3. Validates cross-snapshot consistency.
4. Emits PASS/FAIL report and `GovernancePrimarySurfacesV1` on PASS.

Mismatch fails closed with stable code:

- `GOVERNANCE_SURFACE_MISMATCH`

## Applied scope authority (v6)
`GovernancePrimarySurfacesV1` validation additionally enforces alignment of backend/active snapshots to `AppliedSupportedSetContextV1` (membership + order + digest-prefix consistency).

## v7 applied scope authority

Canonical surfaces now require applied-scope authority from `AppliedSupportedSetContextV1`; legacy scope inference paths are blocked from canonical scope-authority checks.


See also canonical entrypoint rule: docs/canonical_governance_entry_v8.md

## v9 continuity

Governance primary surfaces are now enforced through `CanonicalGovernanceEntryV1` + `AppliedSupportedSetContextV1` as universal canonical entrypoint via `governance-entry-sweep`.
