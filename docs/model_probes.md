# Model Probe Dry-Run

`ucf-ops models probe` führt einen deterministischen Offline-Probe-Lauf über aktive Model-Slots aus.

## Command

```bash
ucf-ops models probe --manifest models/manifest.toml --out ./out/probe_report.json
```

## Eigenschaften

- **Offline**: keine Netzwerkzugriffe.
- **Deterministisch**: feste Probe-Inputs und fixer Seed.
- **Bounded**: kleine, feste Anzahl an Läufen pro Slot.
- **Sicher bei Timeout**: pro Slot Wall-Clock-Timeout mit Worker-Thread-Isolation.
- **Auditierbar**: Ergebnisse werden unter `.ucf/ess/model_probe_records.json` persistiert.

## Report-Interpretation

- `status=ok`: Probe war erfolgreich.
- `status=timeout`: Timeout erreicht, Slot gilt als degradiert.
- `status=error`: Probe-Ausführung ist fehlgeschlagen.
- `status=disabled`: Slot ist im Manifest deaktiviert.

Zusätzlich bewertet ein Tail-Guard die Latenz (`p50`/`p95`) gegen ein Budget. Bei Überschreitung wird die Qualität auf `degraded_fallback` gesetzt.
