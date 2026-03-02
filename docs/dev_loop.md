# Developer Loop (`ucf-ops dev loop`)

## Schnellstart

```bash
ucf-ops dev loop --profile dev --scenario golden_a --ticks 32 --out ./out/dev_loop/
```

Ablauf (deterministisch, bounded):
1. optionaler schneller Testlauf (`cargo test -p ucf-ops --lib --quiet`)
2. kurzer Bringup-Lauf
3. `docs lint` (strict)
4. Goldens-Verify für bis zu zwei Szenarien
5. kompakte PASS/FAIL-Zusammenfassung mit nächsten Befehlen

Artefakte werden unter dem angegebenen `--out`-Pfad geschrieben, inklusive `dev_loop_report.json`.

## Safe Hot Config Reload

Hot Reload ist nur für **nicht-sicherheitskritische** Schlüssel erlaubt:
- `compute_budget_profile`
- `sampling_enabled`
- `log_level`

Ein Reload wird abgelehnt (und protokolliert), wenn sicherheitskritische Werte sich ändern, z. B.:
- `policy_overlay`
- `strict_mode` / `determinism_lock_strict`
- `emergency_policy_pin`
- weitere nicht freigegebene Schlüssel

Records:
- Erfolgreich: `reports/config_reload_applied.jsonl`
- Abgelehnt: `reports/config_reload_denied.jsonl`

## Troubleshoot (`ucf-ops troubleshoot`)

```bash
ucf-ops troubleshoot --run <id> --out ./out/troubleshoot.json
```

Der Befehl sammelt deterministisch (stabile Reihenfolge, begrenzte Anzahl Issues):
- letzte Strict-Failure-Datei
- letzte Drift-Report-Datei
- letzte Readiness-Gate-Datei
- letzte Docs-Lint-Datei
- Anzahl Capability-Denials aus ESS

Ausgabe:
- Top-Issues
- konkrete nächste Kommandos
- JSON-Report unter `--out`
