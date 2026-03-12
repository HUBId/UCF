# Operator Signoff Decision v1 (v4)

`ucf-ops operator signoff` erzeugt eine **einzige, deterministische, read-only** Operator-Signoff-Entscheidung aus bestehenden Artefakten:

- `BackendEvidenceSnapshotV1`
- `ConsolidatedOperatorReportV1`
- `v0/v1/v2/v3` Gate-Reports

## Kommando

```bash
cargo run -p ucf-ops -- operator signoff --out ./out/operator_signoff.json
```

Optional:

- `--run <id>`
- `--latest`
- `--text`

## Entscheidungszustände

- `READY_FOR_SHADOW`
  - Alle erforderlichen Gates PASS gemäß `SignoffPolicyV1`
  - Health nicht FAIL
  - Strict nicht FAIL
  - Für den unterstützten Slot-Satz: `probe_ready=true` und `shadow_ready=true`
  - `active_eligible` darf noch `false` sein
- `READY_FOR_ACTIVE_REVIEW`
  - Alle Bedingungen von `READY_FOR_SHADOW`
  - Mindestens ein unterstützter Real-Slot mit `active_eligible=true`
  - Keine schweren Alerts/Drift-Blocker
- `NOT_READY`
  - Sonst oder wenn erforderliche Evidenz fehlt (fail-closed)

## Was das bedeutet (und was nicht)

Diese Entscheidung ist eine deterministische Reduktion für Operatoren.

Sie **aktiviert nichts**:

- kein Promote
- kein Rollback
- keine Slot-Mode-Änderung
- keine Runtime-Mutation

## Scope

Signoff bleibt auf den aktuell unterstützten Real-Slot-Scope begrenzt:

- `world`
- plus genau ein zweiter Slot (`sae` oder `ssm`)

Wenn dieser Scope nicht konsistent nachweisbar ist, lautet die Entscheidung `NOT_READY` mit stabilen Block-Codes.

## Canonical remediation registry

Operator signoff now emits `canonical_remediation_codes` derived from the shared RemediationCodeRegistryV1 mapping layer.
