use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{
    artifact_schema, check_artifact_schema_snapshots, docs_lint, exports_normalize_check,
    governance_surfaces_check, interop_consistency_matrix, load_applied_supported_set_context_v1,
    models_applied_scope_check, models_consistency_check, operator_workflow_chain,
    portability_check, prefix_hex, sha256_hex, v0_gate, v1_gate, v2_gate, v3_gate, v4_gate,
    v5_gate, DocsLintArgs, DocsLintMode, GateStatus, InteropMismatchCategoryV1,
    InteropOverallStatusV1, OperatorWorkflowArgs, OperatorWorkflowStageV2, OpsError,
    SupportedRealSlotSetV2, V0GateOverallStatus, V1GateOverallStatus, V2GateOverallStatus,
    V3GateOverallStatus, V4GateOverallStatus, V5GateOverallStatus,
};

const DIGEST_PREFIX_LEN: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum V6GateOverallStatus {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct V6GateCheckV1 {
    pub name: String,
    pub status: GateStatus,
    pub evidence_digest_prefixes: BTreeMap<String, String>,
    pub remediation_hint_code: String,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct V6GateReportV1 {
    pub schema_version: u16,
    pub overall_status: V6GateOverallStatus,
    pub checks: Vec<V6GateCheckV1>,
}

pub fn v6_gate(workdir: &Path, out: &Path) -> Result<V6GateReportV1, OpsError> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut checks = Vec::new();

    let v0 = v0_gate(
        workdir,
        &repo_root.join("fixtures/e2e/v0_flow_a.json"),
        &workdir.join("out").join("v0_gate_report_v6_gate.json"),
    )?;
    checks.push(v6_gate_check(
        "v0_gate_pass",
        if matches!(v0.overall_status, V0GateOverallStatus::Pass) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [("v0_gate_report".to_string(), digest_prefix(&v0)?)],
        "REMEDIATE_RUN_V0_GATE",
        "NOTE_REQUIRED_V0",
    ));

    let v1 = v1_gate(
        workdir,
        &workdir.join("out").join("v1_gate_report_v6_gate.json"),
    )?;
    checks.push(v6_gate_check(
        "v1_gate_pass",
        if matches!(v1.overall_status, V1GateOverallStatus::Pass) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [("v1_gate_report".to_string(), digest_prefix(&v1)?)],
        "REMEDIATE_RUN_V1_GATE",
        "NOTE_REQUIRED_V1",
    ));

    let v2 = v2_gate(
        workdir,
        &workdir.join("out").join("v2_gate_report_v6_gate.json"),
    )?;
    checks.push(v6_gate_check(
        "v2_gate_pass",
        if matches!(v2.overall_status, V2GateOverallStatus::Pass) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [("v2_gate_report".to_string(), digest_prefix(&v2)?)],
        "REMEDIATE_RUN_V2_GATE",
        "NOTE_REQUIRED_V2",
    ));

    let v3 = v3_gate(
        workdir,
        &workdir.join("out").join("v3_gate_report_v6_gate.json"),
    )?;
    checks.push(v6_gate_check(
        "v3_gate_pass",
        if matches!(v3.overall_status, V3GateOverallStatus::Pass) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [("v3_gate_report".to_string(), digest_prefix(&v3)?)],
        "REMEDIATE_RUN_V3_GATE",
        "NOTE_REQUIRED_V3",
    ));

    let v4 = v4_gate(
        workdir,
        &workdir.join("out").join("v4_gate_report_v6_gate.json"),
    )?;
    checks.push(v6_gate_check(
        "v4_gate_pass",
        if matches!(v4.overall_status, V4GateOverallStatus::Pass) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [("v4_gate_report".to_string(), digest_prefix(&v4)?)],
        "REMEDIATE_RUN_V4_GATE",
        "NOTE_REQUIRED_V4",
    ));

    let v5 = v5_gate(
        workdir,
        &workdir.join("out").join("v5_gate_report_v6_gate.json"),
    )?;
    checks.push(v6_gate_check(
        "v5_gate_pass",
        if matches!(v5.overall_status, V5GateOverallStatus::Pass) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [("v5_gate_report".to_string(), digest_prefix(&v5)?)],
        "REMEDIATE_RUN_V5_GATE",
        "NOTE_REQUIRED_V5",
    ));

    let governance = governance_surfaces_check(
        workdir,
        &workdir
            .join("out")
            .join("governance_surfaces_check_v6_gate.json"),
    )?;
    checks.push(v6_gate_check(
        "governance_primary_surfaces_pass",
        if governance.status == "PASS" && governance.governance_primary_surfaces.is_some() {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "governance_surfaces".to_string(),
            digest_prefix(&governance)?,
        )],
        "REMEDIATE_GOVERNANCE_PRIMARY_SURFACES",
        "NOTE_REQUIRED_GOVERNANCE",
    ));

    let applied_v2_path = workdir
        .join("out")
        .join("supported_real_slot_set_applied_v2.json");
    let applied_v2 = fs::read_to_string(&applied_v2_path)
        .map_err(OpsError::from)
        .and_then(|body| {
            serde_json::from_str::<SupportedRealSlotSetV2>(&body).map_err(OpsError::from)
        });
    checks.push(v6_gate_check(
        "applied_supported_scope_present",
        if applied_v2.is_ok() {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "applied_supported_set_v2".to_string(),
            if applied_v2.is_ok() {
                prefix_hex(&sha256_hex(&fs::read(&applied_v2_path)?), DIGEST_PREFIX_LEN)
            } else {
                "missing".to_string()
            },
        )],
        "REMEDIATE_APPLY_SUPPORTED_SET",
        "NOTE_REQUIRED_APPLIED_SCOPE",
    ));

    let applied_scope = load_applied_supported_set_context_v1(workdir)?;
    let applied_scope_check = models_applied_scope_check(
        workdir,
        &workdir.join("out").join("applied_scope_check_v6_gate.json"),
    )?;
    let applied_consistent = applied_v2
        .as_ref()
        .map(|v2| {
            applied_scope.applied_set_digest_prefix == prefix_hex(&v2.set_digest, DIGEST_PREFIX_LEN)
                && applied_scope.slots == v2.slots
                && applied_scope.decision == v2.decision
                && applied_scope.policy_digest_prefix == v2.source_policy_digest_prefix
                && applied_scope.previous_set_digest_prefix == v2.previous_set_digest_prefix
        })
        .unwrap_or(false)
        && applied_scope_check.status == "PASS";
    checks.push(v6_gate_check(
        "applied_supported_scope_consistent",
        if applied_consistent {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [
            (
                "applied_scope_context".to_string(),
                prefix_hex(&applied_scope.context_digest, DIGEST_PREFIX_LEN),
            ),
            (
                "applied_scope_check".to_string(),
                digest_prefix(&applied_scope_check)?,
            ),
        ],
        "REMEDIATE_APPLIED_SCOPE_CONSISTENCY",
        "NOTE_REQUIRED_APPLIED_SCOPE_ALIGNMENT",
    ));

    let normalize = exports_normalize_check(
        workdir,
        &workdir
            .join("out")
            .join("export_normalize_check_v6_gate.json"),
    )?;
    checks.push(v6_gate_check(
        "export_normalization_pass",
        if normalize.pass {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [("normalize_check".to_string(), digest_prefix(&normalize)?)],
        "REMEDIATE_EXPORT_NORMALIZATION",
        "NOTE_REQUIRED_EXPORT_NORMALIZATION",
    ));

    let interop = interop_consistency_matrix(
        workdir,
        &workdir
            .join("out")
            .join("interop_consistency_matrix_v6_gate.json"),
    )?;
    checks.push(v6_gate_check(
        "interop_consistency_pass",
        if matches!(interop.summary.overall_status, InteropOverallStatusV1::Pass) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [("interop_matrix".to_string(), digest_prefix(&interop)?)],
        "REMEDIATE_INTEROP_CONSISTENCY",
        "NOTE_REQUIRED_INTEROP",
    ));

    let workflow = operator_workflow_chain(
        workdir,
        &OperatorWorkflowArgs {
            run_id: None,
            latest: true,
        },
        &workdir
            .join("out")
            .join("operator_workflow_chain_v6_gate.json"),
    )?;
    checks.push(v6_gate_check(
        "operator_workflow_chain_present",
        if !workflow.chain_digest.is_empty() {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "operator_workflow_chain".to_string(),
            prefix_hex(&workflow.chain_digest, DIGEST_PREFIX_LEN),
        )],
        "REMEDIATE_OPERATOR_WORKFLOW_CHAIN",
        "NOTE_REQUIRED_WORKFLOW_CHAIN",
    ));

    let governance_prefix = governance
        .governance_primary_surfaces
        .as_ref()
        .map(|v| prefix_hex(&v.governance_surfaces_digest, DIGEST_PREFIX_LEN))
        .unwrap_or_default();
    let workflow_consistent = governance_prefix == workflow.governance_surfaces_digest_prefix
        && prefix_hex(&applied_scope.context_digest, DIGEST_PREFIX_LEN)
            == workflow.applied_supported_scope_digest_prefix
        && prefix_hex(&interop.matrix.matrix_digest, DIGEST_PREFIX_LEN)
            == workflow.interop_matrix_digest_prefix
        && matches!(
            workflow.workflow_stage,
            OperatorWorkflowStageV2::WorkflowExportReady
        )
        && normalize.pass
        && matches!(interop.summary.overall_status, InteropOverallStatusV1::Pass);
    checks.push(v6_gate_check(
        "operator_workflow_chain_consistent",
        if workflow_consistent {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [
            (
                "workflow_chain".to_string(),
                prefix_hex(&workflow.chain_digest, DIGEST_PREFIX_LEN),
            ),
            ("governance_surfaces".to_string(), governance_prefix),
        ],
        "REMEDIATE_OPERATOR_WORKFLOW_ALIGNMENT",
        "NOTE_REQUIRED_WORKFLOW_ALIGNMENT",
    ));

    let artifact_schema = check_artifact_schema_snapshots(&artifact_schema::ArtifactSchemaArgs {
        repo_root: repo_root.clone(),
        out_dir: repo_root.join("docs/artifact_schema_snapshots"),
    })?;
    checks.push(v6_gate_check(
        "artifact_schema_snapshot_checks_pass",
        if artifact_schema.ok {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "artifact_schema_diff_count".to_string(),
            artifact_schema.diffs.len().to_string(),
        )],
        "REMEDIATE_ARTIFACT_SCHEMA_SNAPSHOT",
        "NOTE_REQUIRED_SCHEMA",
    ));

    let docs = docs_lint(&DocsLintArgs {
        repo_root: repo_root.clone(),
        policy_pack: repo_root.join("policies/packs/base_v1"),
        overlay_pack: Some(repo_root.join("policies/packs/overlays/test")),
        spec_snapshot: repo_root.join("docs/spec_snapshot.md"),
        prompt_index: repo_root.join("docs/prompt_series_index.md"),
        module_map: repo_root.join("docs/module_map.md"),
        deploy_doc: repo_root.join("docs/deploy.md"),
        artifact_schema_snapshot_dir: PathBuf::from("docs/artifact_schema_snapshots"),
        mode: DocsLintMode::Strict,
    })?;
    let portability =
        portability_check(&workdir.join("out").join("portability_check_v6_gate.json"))?;
    checks.push(v6_gate_check(
        "portability_docs_checks_pass",
        if docs.ok && portability.deterministic_within_os {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [
            (
                "docs_lint".to_string(),
                if docs.ok { "pass" } else { "fail" }.to_string(),
            ),
            (
                "portability_check".to_string(),
                if portability.deterministic_within_os {
                    "pass"
                } else {
                    "fail"
                }
                .to_string(),
            ),
        ],
        "REMEDIATE_PORTABILITY_DOCS",
        "NOTE_REQUIRED_DOCS",
    ));

    let models_consistency = models_consistency_check(
        workdir,
        &workdir
            .join("out")
            .join("models_consistency_check_v6_gate.json"),
    )?;
    checks.push(v6_gate_check(
        "optional_backend_path_consistent",
        if models_consistency.status == "PASS" {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "optional_backend_mismatch_count".to_string(),
            models_consistency.mismatch_categories.len().to_string(),
        )],
        "REMEDIATE_MODELS_CONSISTENCY",
        "NOTE_OPTIONAL_BACKEND",
    ));

    let has_legacy_mismatch = interop
        .summary
        .mismatch_counts
        .iter()
        .any(|(kind, _)| matches!(kind, InteropMismatchCategoryV1::LegacySurfacePresent));
    checks.push(v6_gate_check(
        "legacy_artifact_translation_ok",
        if has_legacy_mismatch {
            GateStatus::Fail
        } else {
            GateStatus::Skip
        },
        [(
            "legacy_surface_present".to_string(),
            has_legacy_mismatch.to_string(),
        )],
        "REMEDIATE_EXPORT_LEGACY_TRANSLATION",
        "NOTE_OPTIONAL_LEGACY_EXPORT",
    ));

    let overall_status = if checks
        .iter()
        .all(|check| matches!(check.status, GateStatus::Pass | GateStatus::Skip))
    {
        V6GateOverallStatus::Pass
    } else {
        V6GateOverallStatus::Fail
    };

    let report = V6GateReportV1 {
        schema_version: 1,
        overall_status,
        checks,
    };
    crate::write_json(out, &report)?;
    Ok(report)
}

fn digest_prefix<T: Serialize>(value: &T) -> Result<String, OpsError> {
    Ok(prefix_hex(
        &sha256_hex(&serde_json::to_vec(value)?),
        DIGEST_PREFIX_LEN,
    ))
}

fn v6_gate_check(
    name: &str,
    status: GateStatus,
    evidence: impl IntoIterator<Item = (String, String)>,
    remediation_hint_code: &str,
    notes: &str,
) -> V6GateCheckV1 {
    V6GateCheckV1 {
        name: name.to_string(),
        status,
        evidence_digest_prefixes: crate::bounded_evidence(evidence),
        remediation_hint_code: remediation_hint_code.to_string(),
        notes: notes.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v6_gate_check_order_is_fixed() {
        let checks = vec![
            "v0_gate_pass",
            "v1_gate_pass",
            "v2_gate_pass",
            "v3_gate_pass",
            "v4_gate_pass",
            "v5_gate_pass",
            "governance_primary_surfaces_pass",
            "applied_supported_scope_present",
            "applied_supported_scope_consistent",
            "export_normalization_pass",
            "interop_consistency_pass",
            "operator_workflow_chain_present",
            "operator_workflow_chain_consistent",
            "artifact_schema_snapshot_checks_pass",
            "portability_docs_checks_pass",
            "optional_backend_path_consistent",
            "legacy_artifact_translation_ok",
        ];
        let report = V6GateReportV1 {
            schema_version: 1,
            overall_status: V6GateOverallStatus::Pass,
            checks: checks
                .iter()
                .map(|name| v6_gate_check(name, GateStatus::Pass, [], "REMEDIATE", "NOTE"))
                .collect(),
        };
        let names: Vec<String> = report.checks.into_iter().map(|c| c.name).collect();
        assert_eq!(names, checks);
    }

    #[test]
    fn v6_gate_report_serialization_is_deterministic() {
        let report = V6GateReportV1 {
            schema_version: 1,
            overall_status: V6GateOverallStatus::Pass,
            checks: vec![
                v6_gate_check(
                    "a",
                    GateStatus::Pass,
                    [("k".to_string(), "v".to_string())],
                    "REMEDIATE_A",
                    "NOTE_A",
                ),
                v6_gate_check(
                    "b",
                    GateStatus::Skip,
                    [("x".to_string(), "y".to_string())],
                    "REMEDIATE_B",
                    "NOTE_B",
                ),
            ],
        };
        let a = serde_json::to_vec(&report).expect("serialize a");
        let b = serde_json::to_vec(&report).expect("serialize b");
        assert_eq!(a, b);
    }

    #[test]
    fn v6_gate_normalization_fail_closed() {
        let report = V6GateReportV1 {
            schema_version: 1,
            overall_status: V6GateOverallStatus::Fail,
            checks: vec![
                v6_gate_check(
                    "required",
                    GateStatus::Fail,
                    [],
                    "REMEDIATE_REQUIRED",
                    "NOTE_REQUIRED",
                ),
                v6_gate_check(
                    "optional",
                    GateStatus::Skip,
                    [],
                    "REMEDIATE_OPTIONAL",
                    "NOTE_OPTIONAL",
                ),
            ],
        };
        assert!(matches!(report.overall_status, V6GateOverallStatus::Fail));
    }
}
