# Readiness Spine v8

`CanonicalReadinessSpineV1` is the canonical auditable readiness substrate for the currently applied supported scope.

Flow:

Applied Scope (`AppliedSupportedSetContextV1`) → Canonical Governance Entry (`CanonicalGovernanceEntryV1`) → Slot Truths (`SlotReviewabilityTruthV1`) → Reduction (`ReviewabilityReductionV1`) → Active Review Snapshot / Operator Signoff / Operator Review Packet / Operator Workflow Chain.

The spine separates:

- readiness truth (slot truths + reduction)
- external governance blockers (gate/strict/health/interop/export checks)

Run check:

```bash
cargo run -p ucf-ops -- readiness-spine-check --out ./out/readiness_spine_check.json
```

Mismatch categories:

- `SLOT_TRUTH_MISMATCH`
- `REDUCTION_MISMATCH`
- `SIGNOFF_SPINE_DRIFT`
- `REVIEW_PACKET_SPINE_DRIFT`
- `WORKFLOW_SPINE_DRIFT`
- `APPLIED_SCOPE_SPINE_MISMATCH`
- `LEGACY_READINESS_FIELD`
- `LEGACY_READINESS_TRANSLATED`
- `LEGACY_READINESS_REJECTED`
