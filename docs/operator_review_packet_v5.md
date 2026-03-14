# Operator Review Packet v5

## Was ist das?

`OperatorReviewPacketV1` ist die kanonische, deterministische, read-only Top-Level-Review-Oberfläche für Operatoren.

Es korreliert bestehende Artefakte in **ein** reproduzierbares Review-Paket:

- `BackendEvidenceSnapshotV1`
- `AggregatedActiveReviewSnapshotV1`
- `OperatorSignoffDecisionV1`
- `ConsolidatedOperatorReportV1`
- `v0/v1/v2/v3/v4` Gate-Reports
- optionales `backend_resolution_<slot>.json`

Es führt **keine** Runtime- oder Config-Mutation aus und aktiviert nichts.

## Kommando

```bash
cargo run -p ucf-ops -- operator review-packet --out ./out/operator_review_packet.json
```

Optional:

- `--run <id>`
- `--latest`
- `--text`

## Review-Stufen (`OperatorReviewStageV1`)

- `REVIEW_BLOCKED`
  - fehlende Pflichtartefakte
  - Gate-Fehler oder Gate fehlt
  - Signoff nicht bereit
  - Digest-/Kontext-Mismatch
  - ambiger unterstützter Slot-Scope
- `REVIEW_SHADOW_READY`
  - Shadow-Review hinreichend bereit
  - aber nicht ausreichend für Active-Review
- `REVIEW_ACTIVE_READY`
  - Active-Review-Snapshot ist reviewable und mit Signoff ausgerichtet
  - weiterhin nur Review-Bereitschaft, **keine** Aktivierung

## Unterschied zu Operator Report

- `operator report`: breiter Health-/Status-Überblick über mehrere Sektionen.
- `operator review-packet`: explizit harte, deterministische Review-Reduktion mit klarer Stage + Block-/Remediation-Codes in einem Artefakt.

## Unterschied zu Operator Signoff

- `operator signoff`: Entscheidung über Signoff-Zustand (`READY_FOR_SHADOW`, `READY_FOR_ACTIVE_REVIEW`, `NOT_READY`).
- `operator review-packet`: top-level Orchestrierung, die Signoff + Active-Review + Gates + Evidence + Report zusammenführt und als finale Review-Einstiegsfläche ausgibt.

## Interpretation kurz

- `REVIEW_BLOCKED`: erst Block-Codes beheben; Review nicht freigabefähig.
- `REVIEW_SHADOW_READY`: Shadow-Review möglich, Active-Review noch nicht vollständig bereit.
- `REVIEW_ACTIVE_READY`: Active-Review für menschliche Prüfung bereit; weiterhin kein Runtime-Control-Action.


## v6 consumption preference

`OperatorReviewPacketV1` now prefers validated governance primary surfaces (`BackendEvidenceSnapshotV1` + `AggregatedActiveReviewSnapshotV1`) via `GovernancePrimarySurfacesV1` and treats mismatches as fail-closed blockers.

## Applied scope authority (v6)
`OperatorReviewPacketV1` embeds applied-scope context digest reference and reports only applied slots. Extra-slot evidence is surfaced as blocking diagnostics and cannot elevate review stage.


## Export normalization v6

This surface participates in canonical export normalization (shared `CanonicalExportArtifactRefV1` and `CanonicalExportContextV1`) and is validated via `cargo run -p ucf-ops -- exports normalize-check --out ./out/export_normalize_check.json`. See `docs/export_normalization_v6.md` for semantics.


## Cross-surface interop proof (v6)

`OperatorReviewPacketV1` is part of the canonical interop matrix and must align with gate/strict/snapshot/signoff/export references:

```bash
cargo run -p ucf-ops -- interop consistency-matrix --out ./out/interop_consistency_matrix.json
```

Key failures include `SNAPSHOT_REFERENCE_MISMATCH`, `REMEDIATION_MISMATCH`, and `REQUIRED_SURFACE_MISSING`.


## Top-level operator workflow chain (v6)

Als primären Operator-Einstieg für die gesamte Review/Export-Kette verwende:

```bash
cargo run -p ucf-ops -- operator workflow --out ./out/operator_workflow_chain.json
```

Das Workflow-Chain-Artefakt korreliert Governance-Surfaces, Applied-Scope, Review-Packet, Signoff, Interop-Matrix und Export-Normalisierung in eine einzige deterministische Stage (`WORKFLOW_BLOCKED|WORKFLOW_REVIEW_READY|WORKFLOW_EXPORT_READY`). Siehe `docs/operator_workflow_chain_v6.md`.

## v7 applied scope authority

Canonical surfaces now require applied-scope authority from `AppliedSupportedSetContextV1`; legacy scope inference paths are blocked from canonical scope-authority checks.
