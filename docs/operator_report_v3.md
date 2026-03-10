# Consolidated Operator Report v1 (v3)

`ucf-ops operator report` erzeugt einen einzelnen, deterministischen Offline-Report für die aktuelle Operator-Lage.

## Inhalt

`ConsolidatedOperatorReportV1` bündelt:

- Health
- Unified Eligibility (probe/shadow/active)
- Drift
- Alerts
- Strict-Check-Status
- Gate-Status (v0/v1/v2, später v3 wenn vorhanden)

Fehlende Teilreports werden **explizit** als `MISSING` dargestellt (nicht stillschweigend ignoriert).

## Kommando

```bash
cargo run -p ucf-ops -- operator report --out ./out/operator_report.json
```

Optionen:

- `--run <id>`: bevorzugt run-spezifische Artefakte unter `./out/<id>/...`
- `--latest`: nimmt deterministisch den neuesten verfügbaren Report-Pfad
- `--text`: gibt zusätzlich eine kurze textuelle Operator-Zusammenfassung aus

## Overall-Status

Reduktion ist deterministisch:

- `FAIL` bei strict FAIL, health FAIL oder aktiven `SEVERE` Alerts
- `DEGRADED` bei Drift-Degradation, aktiven Alerts oder Eligibility-Degradation
- `WARN` bei fehlenden Pflicht-Teilreports oder nur partieller Readiness
- `OK` sonst

## Interpretation je Section

- `health_section`: Status, strict mode, last tick age, emergency flag
- `eligibility_section`: pro unterstütztem Real-Slot `probe_ready`, `shadow_ready`, `active_eligible`, primärer Denial-Grund
- `drift_section`: pro Slot Drift-Status + severe alarm count (bounded)
- `alerts_section`: aktive Alert-Anzahl + Top-Alerts (bounded)
- `strict_section`: letzter strict-Status + primärer Denial-Code
- `gates_section`: letzte bekannte v0/v1/v2 Statuswerte, sonst `MISSING`

## Unterschied zu Einzelreports

Der Operator-Report ersetzt **nicht** die Quellreports. Er ist ein read-only Orchestrierungs- und Normalisierungs-Layer:

- keine Aktivierung/Promotion/Rollback-Aktionen
- keine neue Entscheidungslogik
- nur konsolidierte Sicht + bounded Remediation-Codes
