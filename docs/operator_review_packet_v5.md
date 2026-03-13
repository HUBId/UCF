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
