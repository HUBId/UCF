# Supported Scope Execution v19 (Governance Closure Gated)

`SupportedScopeExecutionV14` ist die aktuelle autoritative Ausführungsentscheidung für Supported-Scope-Änderungen in v19.

## Hierarchie (v19)
- **Policy** (`SupportedRealSlotSetPolicyV2`): bewertet Kandidaten, aber ändert den Scope nicht.
- **Reevaluation** (`SupportedScopeReevaluationV1`): vorbereitende Validierung, ohne Apply-Autorität.
- **Prior execution artifacts** (v13 und älter): nur Kettenhistorie.
- **Current execution** (`SupportedScopeExecutionV14`): einzig zulässige Autorisierung für `models supported-set-apply` in v19.

## Warum Governance Closure jetzt Pflicht ist
Eine Expansion darf nur erfolgen, wenn die closure-komplette Governance-Kette PASS bleibt:

`AppliedSupportedSetContextV1` → `CanonicalGovernanceEntryV1` → `CanonicalGovernanceEntryAuthorityV2` → `FinalGovernanceConsumerAuthorityV1` → `FinalGovernanceResidualSweepV1` → `ResidualFreeGovernanceConsumerAuthorityV1` → `ResidualFreeGovernanceAbsoluteSweepV1` → `AbsoluteFinalGovernanceTerminalSweepV1` → `TerminalGovernanceUltimateSweepV1` → `GovernanceConvergenceSweepV1` → `GovernanceStabilizationSweepV1` → `GovernanceFinalConsolidationSweepV1` → `GovernanceClosureSweepV1`.

Damit wird eine Expansion fail-closed blockiert, falls Kandidaten auf Governance-Wrapper/Crosswalk/Secondary-Rendering-Pfade oder Sonderpfade in Export-/Bundle-/Continuity-Ketten angewiesen wären.

## Command

```bash
cargo run -p ucf-ops -- models supported-scope-execute-v14 --out ./out/supported_scope_execute_v14.json
```

## Freeze-Reaffirmation ist ein valider Erfolg
`REAFFIRM_FREEZE` ist in v19 weiterhin ein explizites, first-class Erfolgsresultat, wenn Governance-Closure oder die aktuelle Kandidatenlage keine saubere Ein-Slot-Expansion zulassen.

## Wie man erkennt, ob sich der Applied Scope geändert hat
1. `./out/supported_scope_execute_v14.json` prüfen:
   - `execution_decision`
   - `chosen_candidate_slot`
   - `resulting_supported_set_digest_prefix`
2. Danach anwenden:

```bash
cargo run -p ucf-ops -- models supported-set-apply --out ./out/supported_set_apply.json
```

3. In `supported_set_apply.json` `previous_slots` vs `resulting_slots` vergleichen.
