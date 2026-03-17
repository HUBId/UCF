# Reviewability Truth v7

`SlotReviewabilityTruthV1` ist die kanonische per-slot Reviewability-Wahrheit für den angewendeten Scope (`AppliedSupportedSetContextV1`).

## Kernkonzepte
- `SlotReviewabilityTruthV1`: probe/shadow/active + strict/drift/alert blocking + primary denial/remediation + evidence digest prefixes.
- `ReviewabilityReductionV1`: deterministische Reduktion über alle in-scope Slots in Scope-Reihenfolge.
  - `NONE_REVIEWABLE`
  - `PARTIAL_REVIEWABLE`
  - `ALL_REVIEWABLE`

## Konsistenzflächen
Diese Flächen müssen dieselbe Wahrheit nutzen:
- `AggregatedActiveReviewSnapshotV1`
- `OperatorSignoffDecisionV1`
- `OperatorReviewPacketV1`

Legacy-Felder werden nur additive und nicht-widersprüchlich toleriert; widersprüchliche Legacy-Reduktion wird fail-closed als `LEGACY_REDUCTION_REJECTED` markiert.

## Prüfung
```bash
cargo run -p ucf-ops -- operator review-truth-check --out ./out/review_truth_check.json
```

Mismatch-Kategorien:
- `PER_SLOT_REVIEWABILITY_MISMATCH`
- `AGGREGATE_REDUCTION_MISMATCH`
- `SIGNOFF_REVIEWABILITY_DRIFT`
- `REVIEW_PACKET_REVIEWABILITY_DRIFT`
- `APPLIED_SCOPE_SLOT_TRUTH_MISSING`
- `LEGACY_REVIEWABILITY_FIELD`
- `LEGACY_REDUCTION_TRANSLATED`
- `LEGACY_REDUCTION_REJECTED`

See also: readiness spine canon (`docs/readiness_spine_v8.md`).
