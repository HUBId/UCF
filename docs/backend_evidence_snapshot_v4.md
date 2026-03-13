# Backend Evidence Snapshot v4

`BackendEvidenceSnapshotV1` is the canonical, deterministic export for the currently supported real-slot set.

## Scope

Bounded scope only:
- `world_jepa`
- exactly one second slot declared in `docs/series_state_snapshot.md` (`sae` or `ssm`)

No new slot is introduced. No runtime behavior is changed.

## Export command

```bash
cargo run -p ucf-ops -- models evidence-snapshot --out ./out/backend_evidence_snapshot.json
```

Optional filters:

```bash
cargo run -p ucf-ops -- models evidence-snapshot --slot world --out ./out/backend_evidence_world.json
cargo run -p ucf-ops -- models evidence-snapshot --run <id> --out ./out/backend_evidence_snapshot.json
```

## What the snapshot contains

Top-level:
- `schema_version`
- `supported_slot_set_digest`
- `policy_graph_digest_prefix`
- `manifest_digest_prefix`
- `slots` (ordered by `slot_id`, max 2)
- `snapshot_digest`

Per slot:
- `slot_id`
- `target_hash_prefix`
- `backend_support` in fixed backend order: `stub`, `candle`, `burn`
- `evidence` digest prefixes + drift/freshness + hash consistency flag
- `readiness` booleans: `probe_ready`, `shadow_ready`, `active_eligible`
- `denials` (probe/shadow/active), normalized denial code enums
- `remediation_codes` (bounded to max 4)
- `burn_resolution` (`BurnSupportResolutionV1`) with explicit binary outcome for optional Burn state

## Backend support states vs readiness booleans

Backend support is capability/availability evidence for backend paths:
- `SUPPORTED`
- `UNSUPPORTED`
- `NOT_BUILT`
- `NOT_CONFIGURED`

Readiness booleans are slot readiness outcomes:
- `probe_ready`
- `shadow_ready`
- `active_eligible`

Support state does **not** automatically imply readiness, and readiness does **not** auto-activate any backend.

## Difference to existing reports

- `models evidence-snapshot`: canonical compact evidence contract for machines/tools.
- `models eligibility`: readiness-focused presentation derived from the same shared evidence layer.
- `operator report`: broad operator-facing health summary; may consume the snapshot when eligibility artifact is missing.

## Reuse guidance

Downstream tools (gate reports, bugkits, repro packs) should reference or embed this artifact instead of re-deriving overlapping evidence independently.


## Relationship to active-review snapshot

`backend_evidence_snapshot_v1` remains the backend/evidence substrate.
`active_review_snapshot_v1` derives a canonical active-review surface from this substrate plus strict/alert/signoff alignment context:

```bash
cargo run -p ucf-ops -- models active-review-snapshot --out ./out/active_review_snapshot.json
```


## Burn resolution integration (v5)

`BackendEvidenceSnapshotV1` now embeds a canonical `BurnSupportResolutionV1` per slot so optional Burn state is explicit and no longer ambiguous.

## Top-level operator entrypoint

`backend_evidence_snapshot_v1` bleibt das Evidenz-Substrat.
Für die deterministische Endkorrelation (Evidence + Active-Review + Signoff + Gates) nutze:

```bash
cargo run -p ucf-ops -- operator review-packet --out ./out/operator_review_packet.json
```

Siehe `docs/operator_review_packet_v5.md`.


## v6 primary governance surface

In v6 this snapshot is one of the two primary governance surfaces and is validated together with `AggregatedActiveReviewSnapshotV1` via `GovernancePrimarySurfacesV1` before downstream governance consumers derive conclusions.
