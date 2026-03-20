# v9 Signoff

`ucf-ops v9 gate` is the v9 final governance/scope/readiness/bundle/continuity hardening gate.

## Command

```bash
cargo run -p ucf-ops -- v9 gate --out ./out/v9_gate_report.json
```

Exit codes:
- `0`: overall PASS
- `2`: overall FAIL

## PASS guarantees

PASS certifies all of the following under the **current applied supported scope**:
- final governance-entry authority is enforced across canonical surfaces
- supported-scope execution is explicit/current and consistent with applied scope + final governance authority
- final readiness authority is enforced across canonical surfaces
- final bundle authority is enforced across canonical export surfaces
- final primary blocking/remediation semantics are canonical
- full end-to-end continuity authority from governance entry to bundle is coherent
- artifact schema snapshot checks and portability/docs checks pass

## PASS does not guarantee

PASS does **not** certify:
- broader runtime capability
- additional slot/backend production readiness
- automatic slot activation
- GPU/remote compute/training readiness

## Check semantics

`V9GateReportV1` uses fixed-order checks with normalized statuses:
- `PASS`: required surface is present and coherent
- `FAIL`: required surface missing/stale/inconsistent or authority mismatch
- `SKIP`: explicitly unsupported optional path only

Optional checks are:
- `optional_backend_path_consistent`
- `legacy_bundle_translation_ok`
- `legacy_governance_entry_translation_ok`
- `legacy_readiness_translation_ok`

## Post-PASS continuation

After `ucf-ops v9 gate` PASS, continue at **Prompt 270** via `docs/next_10_prompts.md`.
