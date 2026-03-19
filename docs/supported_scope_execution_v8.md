# Supported Scope Execution v8

## Hierarchie: Policy → Reevaluation → Execution

- `SupportedRealSlotSetPolicyV2` bleibt ein Review-/Policy-Artefakt.
- `SupportedScopeReevaluationV1` bewertet den Policy-Vorschlag gegen den aktuellen Zustand neu.
- `SupportedScopeExecutionV4` ist die **aktuelle autoritative Ausführungsentscheidung** für Freeze vs. Expansion.

Nur `SupportedScopeExecutionV4` darf den nächsten angewendeten Supported-Scope fortschreiben.

## Warum Canonical Governance Entry verpflichtend ist

Expansion wird nur zugelassen, wenn die aktuelle Autoritätskette weiterhin konsistent ist:

1. `AppliedSupportedSetContextV1`
2. `GovernancePrimarySurfacesV1`
3. `CanonicalGovernanceEntryV1` + `governance-entry-check` PASS

Wenn diese Kette nicht sauber trägt, wird **fail-closed** erneut `REAFFIRM_FREEZE` ausgegeben.

## Command

```bash
cargo run -p ucf-ops -- models supported-scope-reevaluate --out ./out/supported_scope_reeval.json
cargo run -p ucf-ops -- models supported-scope-execute-v4 --out ./out/supported_scope_execute_v4.json
cargo run -p ucf-ops -- models supported-set-apply --out ./out/supported_set_apply.json
```

## Freeze-Reaffirmation ist Erfolg

`REAFFIRM_FREEZE` ist ein gültiges, gewünschtes Ergebnis, wenn:

- keine exakt eine weiterhin tragfähige Kandidaten-Slot-Erweiterung existiert,
- Canonical Entry / Primary Surfaces nicht PASS sind,
- sekundäre Entry-Abhängigkeiten verbleiben,
- oder ein Candidate bereits im Scope liegt.

Das ist eine Schutzfunktion und keine partielle Aktivierung.
