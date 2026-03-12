# Remediation Codes v1

Generated from the canonical remediation registry source.

| Code | Description | Suggestion Key |
|---|---|---|
| REMEDIATION_RUN_PROBE | Generate fresh probe evidence for the affected slot. | `run_probe` |
| REMEDIATION_RERUN_SHADOW_WINDOW | Regenerate compare/shadow-window evidence. | `rerun_shadow_window` |
| REMEDIATION_CHECK_DRIFT_REPORT | Inspect bounded drift report and clear severe drift. | `check_drift_report` |
| REMEDIATION_CHECK_STRICT_REPORT | Run strict checks and resolve failures. | `check_strict_report` |
| REMEDIATION_VERIFY_MANIFEST | Verify model manifest integrity and slot declarations. | `verify_manifest` |
| REMEDIATION_STAY_SHADOW | Keep slot in shadow mode until evidence converges. | `stay_shadow` |
| REMEDIATION_REVIEW_ACTIVE_EVIDENCE | Review active-evidence eligibility for the slot. | `review_active_evidence` |
| REMEDIATION_CHECK_PORTABILITY_REPORT | Check portability matrix and required backend support. | `check_portability_report` |
| REMEDIATION_REGENERATE_OPERATOR_REPORT | Regenerate consolidated operator report artifacts. | `regenerate_operator_report` |
| REMEDIATION_RESOLVE_HASH_MISMATCH | Resolve target/evidence hash mismatch before promotion. | `resolve_hash_mismatch` |
| REMEDIATION_REVIEW_REPORT_MANUALLY | Perform manual operator review of bounded report evidence. | `review_report_manually` |

## Remediation consistency enforcement

Canonical remediation consistency is enforced across strict check, eligibility, operator report, operator signoff, v4 gate surfaces, and enriched export manifests via `ucf-ops remediation-consistency-check`.
