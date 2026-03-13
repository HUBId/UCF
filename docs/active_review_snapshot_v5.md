# Active Review Snapshot v5

## Purpose

`active_review_snapshot_v1` is the canonical, exportable, read-only snapshot for "ready for active review" across the currently supported real-slot set.
It is bounded, deterministic, offline, and hardware-neutral.

It does **not** enable active mode and does **not** mutate runtime state.

## Command

```bash
cargo run -p ucf-ops -- models active-review-snapshot --out ./out/active_review_snapshot.json
```

## Scope

- Covers only the currently supported real slots from `SupportedRealSlotSetV1` / `SupportedRealSlotSetPolicyV2`.
- No new slot introduction.
- No promotion/rollback automation.

## Per-slot view (`ActiveReviewEvidenceV1`)

Each slot includes:

- `probe_ready`
- `shadow_ready`
- `active_eligible` (canonical eligibility signal)
- `strict_blocking`
- `drift_blocking`
- `alert_blocking`
- `primary_denial_code`
- bounded `remediation_codes` (max 4)
- contributing evidence digest prefixes
- `burn_resolution` (propagated from backend evidence snapshot)

## Aggregated view (`AggregatedActiveReviewSnapshotV1`)

Top-level fields:

- `schema_version`
- `supported_slot_set_digest`
- `policy_graph_digest_prefix`
- `manifest_digest_prefix`
- ordered `slots` (sorted by `slot_id`)
- `overall_review_status`
- `signoff_alignment`
- `snapshot_digest`

### `overall_review_status`

- `NONE_REVIEWABLE`: no supported slot is simultaneously active-eligible and unblocked.
- `PARTIAL_REVIEWABLE`: at least one slot is active-eligible and unblocked, but not all.
- `ALL_REVIEWABLE`: all supported slots are active-eligible and unblocked.

A slot is reviewable iff:

- `active_eligible == true`
- `strict_blocking == false`
- `drift_blocking == false`
- `alert_blocking == false`

## Difference to related surfaces

- **vs unified eligibility report**: eligibility report focuses on readiness progression (`probe/shadow/active`) and denials; active-review snapshot normalizes explicit active-review blockers into one canonical review surface.
- **vs operator signoff**: signoff is a decision artifact; active-review snapshot is normalized evidence state for operator review and downstream gates.

## Notes

- Snapshot generation reuses existing evidence and signoff surfaces.
- Output is suitable for later bugkit/repro/gate artifact attachment.


## Burn state visibility (v5)

Active-review snapshot now carries the chosen second-slot Burn resolution explicitly (supported-for-shadow-compare vs closed-unsupported), without granting any new active eligibility.

## Top-level operator entrypoint

`active_review_snapshot_v1` bleibt die kanonische aktive Review-Evidenz.
Die oberste Operator-Einstiegsfläche für korrelierte Prüfung ist jedoch:

```bash
cargo run -p ucf-ops -- operator review-packet --out ./out/operator_review_packet.json
```

Siehe `docs/operator_review_packet_v5.md`.


## v6 primary governance surface

In v6 this snapshot is one of the two primary governance surfaces and is validated together with `BackendEvidenceSnapshotV1` via `GovernancePrimarySurfacesV1` before downstream governance consumers derive conclusions.
