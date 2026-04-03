# Supported Scope Execution v20 (Governance Seal Gated)

`SupportedScopeExecutionV15` ist die aktuelle autoritative Ausführungsentscheidung für Supported-Scope-Änderungen in v20.

## Hierarchie (v20)

- **Policy** (`SupportedRealSlotSetPolicyV2`): bewertet Kandidaten, ändert den Scope aber nicht.
- **Reevaluation** (`SupportedScopeReevaluationV1`): prüft Kandidaten vorab, ohne Apply-Autorität.
- **Prior execution artifacts** (v14 und älter): nur Historie, keine direkte Apply-Autorität.
- **Current execution** (`SupportedScopeExecutionV15`): einzige zulässige Autorisierung für `models supported-set-apply` in v20.

## Warum Governance Seal jetzt Pflicht ist

Eine Expansion darf nur noch erfolgen, wenn die closure-komplette final-konsolidierte stabilisierte Governance-Kette **inklusive Seal** PASS bleibt:

`AppliedSupportedSetContextV1` → `CanonicalGovernanceEntryV1` → `CanonicalGovernanceEntryAuthorityV2` → `FinalGovernanceConsumerAuthorityV1` → `FinalGovernanceResidualSweepV1` → `ResidualFreeGovernanceConsumerAuthorityV1` → `ResidualFreeGovernanceAbsoluteSweepV1` → `AbsoluteFinalGovernanceTerminalSweepV1` → `TerminalGovernanceUltimateSweepV1` → `GovernanceConvergenceSweepV1` → `GovernanceStabilizationSweepV1` → `GovernanceFinalConsolidationSweepV1` → `GovernanceClosureSweepV1` → `GovernanceSealSweepV1`.

Damit bleibt Expansion fail-closed, wenn Governance-Shell-/Bridge-/Auxiliary-View-Abhängigkeiten oder Export-/Bundle-/Continuity-Sonderpfade nötig wären.

## Command

```bash
cargo run -p ucf-ops -- models supported-scope-execute-v15 --out ./out/supported_scope_execute_v15.json
```

## Warum Freeze-Reaffirmation ein Erfolg ist

`REAFFIRM_FREEZE` bleibt ein explizites, first-class Erfolgsresultat. Das ist korrekt, wenn unter dem aktuellen Seal-Status keine saubere Ein-Slot-Expansion möglich ist.

## Wie man erkennt, ob sich der Applied Scope geändert hat

1. `./out/supported_scope_execute_v15.json` prüfen:
   - `execution_decision`
   - `chosen_candidate_slot`
   - `resulting_supported_set_digest_prefix`
2. Danach anwenden:

```bash
cargo run -p ucf-ops -- models supported-set-apply --out ./out/supported_set_apply.json
```

3. In `supported_set_apply.json` `previous_slots` vs. `resulting_slots` vergleichen.
