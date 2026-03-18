# Chip-2 Dokumentation

## Architektur
- [Chip-2 Überblick](architecture/chip2_overview.md)
- [Interfaces](architecture/interfaces.md)
- [Microcircuit-Ausbaupfad](architecture/microcircuit_path.md)
- [Teststrategie](architecture/testing_strategy.md)

## Module
- [DBM 13 Hypothalamus](modules/dbm_13_hypothalamus.md)
- [DBM 12 Insula](modules/dbm_12_insula.md)
- [DBM 0 Substantia Nigra](modules/dbm_0_sn.md)
- [DBM 7 LC](modules/dbm_7_lc.md)
- [DBM 8 Serotonin](modules/dbm_8_serotonin.md)
- [DBM 6 Dopamin/NAcc](modules/dbm_6_dopamin_nacc.md)
- [DBM 9 Amygdala](modules/dbm_9_amygdala.md)
- [DBM PAG](modules/dbm_pag.md)
- [DBM STN](modules/dbm_stn.md)
- [DBM PMRF](modules/dbm_pmrf.md)
- [DBM SC](modules/dbm_sc.md)
- [DBM PPRF](modules/dbm_pprf.md)
- [DBM 18 Cerebellum](modules/dbm_18_cerebellum.md)
- [DBM HPA](modules/dbm_hpa.md)

## Templates
- [Modul-Template](templates/module_template.md)
- [Golden-Stream-Testtemplate](templates/test_template.md)
- [Konfigurations-Template](templates/config_template.md)

## Roadmaps & Runbooks
- [v1.1 Plan: Real Models & Optional GPU Lane](v1_1_plan.md)
- [Weights Incident Response](runbooks/weights_incident_response.md)
- [Models Lifecycle Runbook v1](runbooks/models_v1.md)
- [Shadow Runbook v1](runbooks/shadow_v1.md)
- [Drift + Alerts Runbook v1](runbooks/drift_alerts_v1.md)

## Repository Hygiene
- [Branch Policy](branch_policy.md)
- [Contributing Workflow](contributing_workflow.md)
- [Codex Instructions (AGENTS.md)](../AGENTS.md)


## v8 portability/docs quick run

```bash
cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json
cargo run -p ucf-ops -- spec artifact-schemas-check --out ./out/artifact_schema_check.json
cargo run -p ucf-ops -- governance-entry-check --out ./out/governance_entry_check.json
cargo run -p ucf-ops -- models supported-scope-execute --out ./out/supported_scope_execute_v3.json --workdir .
cargo run -p ucf-ops -- readiness-spine-check --out ./out/readiness_spine_check.json
cargo run -p ucf-ops -- exports bundle-spine-check --in ./out/repro_portability.zip --out ./out/bundle_spine_check.json
cargo run -p ucf-ops -- remediation-spine-check --out ./out/remediation_spine_check.json
```
