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
