# v10 Signoff

`ucf-ops v10 gate` is the final v10 governance/scope/readiness/bundle/primary-semantics/continuity consumer-authority gate.

## Command

```bash
cargo run -p ucf-ops -- v10 gate --out ./out/v10_gate_report.json
```

Exit codes:
- `0`: overall PASS
- `2`: overall FAIL

## PASS guarantees

PASS certifies all of the following under the **current applied supported scope**:
- final governance-consumer authority is enforced across canonical consumers
- current supported-scope execution v5 is explicit/current and coherent with applied scope + final governance consumer authority
- final readiness-consumer authority is enforced across canonical consumers
- final bundle-consumer authority is enforced across canonical export consumers
- final primary-semantics-consumer authority is enforced across canonical consumers
- exactly one top-level continuity proof (`final-continuity-sweep`) is authoritative for canonical operator/export flows
- artifact schema snapshot checks and portability/docs checks pass

## PASS does not guarantee

PASS does **not** certify:
- broader runtime capability
- additional slot/backend production readiness
- automatic slot activation
- GPU/remote compute/training readiness

## PASS / FAIL / SKIP interpretation

`V10GateReportV1` uses fixed-order checks with normalized statuses:
- `PASS`: required surface exists and is coherent
- `FAIL`: required surface missing/stale/inconsistent or authority mismatch
- `SKIP`: explicitly unsupported optional path only

Optional checks are:
- `optional_backend_path_consistent`
- `legacy_governance_input_translation_ok`
- `legacy_readiness_input_translation_ok`
- `legacy_bundle_input_translation_ok`
- `legacy_top_level_continuity_surface_demoted`

## Phase intent

v10 is a final **consumer-authority / continuity-unification hardening** phase. The current applied supported scope remains the only authoritative scope for this gate.

## Post-v10 continuation

After v10 gate PASS, continue at Prompt 280 via `docs/next_10_prompts.md`.
