# Canonical Governance Entry V8

`CanonicalGovernanceEntryV1` ist der verpflichtende autoritative Einstieg für kanonische Governance-/Review-/Export-Consumer.

## Autoritative Grundlagen

1. `AppliedSupportedSetContextV1`
2. `GovernancePrimarySurfacesV1`

Nur aus diesen beiden Grundlagen darf die kanonische Autorität abgeleitet werden.

## Felder (`CanonicalGovernanceEntryV1`)

- `applied_supported_set_digest_prefix`
- `applied_context_digest_prefix`
- `governance_primary_surfaces_digest_prefix`
- `authority_digest`
- `entry_status`

## Kanonische Consumer (v8)

- `AggregatedActiveReviewSnapshotV1`
- `OperatorSignoffDecisionV1`
- `OperatorReviewPacketV1`
- `OperatorWorkflowChainV1`
- `InteropConsistencyMatrixReportV1`

Diese Artefakte werden im `governance-entry-check` gegen denselben kanonischen Authority-Einstieg geprüft.

## Befehl

```bash
cargo run -p ucf-ops -- governance-entry-check --out ./out/governance_entry_check.json
```


## Auswirkungen auf Scope-Expansion

`models supported-scope-execute` darf Expansion nur dann als `EXECUTE_EXPAND_BY_ONE` ausgeben, wenn `CanonicalGovernanceEntryV1` aktuell PASS ist und keine sekundären Entry-Pfade benötigt werden. Andernfalls wird explizit `REAFFIRM_FREEZE` ausgegeben.

Zusatz: Der v8-Nachweis zur kanonischen Blocking/Remediation-Konsistenz über alle Spines läuft über `ucf-ops remediation-spine-check` (siehe `docs/remediation_spine_consistency_v8.md`).

## v9 update

v9 makes canonical governance entry universal across covered canonical surfaces via `governance-entry-sweep`.
