# Real Slot Consistency v4

This document defines `SupportedRealSlotSetV1` and `SlotEvidenceSnapshotV1` as the shared source for:
- `ucf-ops models eligibility`
- `ucf-ops strict check`
- `ucf-ops operator report`
- gate preconditions.

## Supported real slot set

`SupportedRealSlotSetV1` is bounded to exactly two slots in this phase:
- `world_jepa`
- and exactly one second slot (`sae` or `ssm`).

Detection source remains `docs/series_state_snapshot.md` (`Second supported slot`).
Ordering is deterministic by `slot_id`. Ambiguity fails closed.

## Shared evidence snapshot

`resolve_slot_evidence(slot_id, target_hash, ctx)` returns `SlotEvidenceSnapshotV1` with:
- manifest and target hash prefixes
- latest probe/compare/shadow/active evidence digest prefixes
- drift status
- freshness ages
- hash consistency and missing flags.

This enables a single evidence interpretation path.

## Freshness + denial semantics

`EvidenceFreshnessPolicyV1` and `EvidenceDenialCodeV1` normalize reason families:
`NO_PROBE`, `STALE_PROBE`, `NO_COMPARE`, `STALE_COMPARE`, `HASH_MISMATCH`, `DRIFT_SEVERE`, `DRIFT_WARN`, `ACTIVE_NOT_ENABLED`, `UNSUPPORTED_SLOT_SET`.

## Consistency check command

Run:

```bash
cargo run -p ucf-ops -- models consistency-check --out ./out/models_consistency_check.json
```

The report returns `PASS`/`FAIL` and bounded mismatch categories (e.g. slot set, target hash, readiness booleans, denial reason alignment).
