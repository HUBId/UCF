# Change Impact Analyzer

`ucf-ops change-impact` berechnet offline aus einem lokalen Git-Diff einen konservativen, deterministischen Minimal-Testplan.

## Ziel

- Nur notwendige Gates für den konkreten Change ausführen.
- Niemals unterschätzen: bei Unsicherheit werden zusätzliche Gates aufgenommen.
- Kein PR-API Zugriff, nur lokales `git diff`.

## Regeln

Regeln stehen in `docs/change_impact_rules.toml`.

Wichtige Felder:

- `max_files`: begrenzt analysierte Dateien.
- `max_commands`: begrenzt ausgegebene Shell-Kommandos.
- `default_modules` / `default_gates`: konservativer Fallback für unbekannte Pfade.
- `command_catalog`: Gate-Name → Shell-Command.
- `[[rules]]`: `include` / `exclude` Globs plus `modules` und `gates`.

Glob-Syntax:

- `*` für ein Segment.
- `**` rekursiv über Pfadsegmente.

## Ausführen

```bash
cargo run -p ucf-ops -- change-impact \
  --base HEAD~1 \
  --head HEAD \
  --out ./out/change_impact_plan.md \
  --json-out ./out/change_impact_plan.json
```

Optionale Flags:

- `--rules <path>`: alternatives Regelset.

## Ausgaben

- Markdown-Plan (`--out`): lesbare Checkliste.
- JSON-Plan (`--json-out`): maschinenlesbar für CI/Artefakte.

Der Plan enthält:

- `changed_files` Metriken (total, analyzed, truncation)
- `affected_modules`
- `required_gates`
- `commands`

## CI-Integration (informational)

Empfohlen als nicht-blockierender Schritt:

1. `change-impact` ausführen.
2. Markdown + JSON als Build-Artefakt hochladen.
3. Später optional erzwingen, dass mindestens diese Gates gelaufen sind.
