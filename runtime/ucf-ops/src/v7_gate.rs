use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{
    artifact_schema, check_artifact_schema_snapshots, docs_lint, exports_roundtrip_check,
    interop_consistency_matrix, load_applied_supported_set_context_v1, models_consistency_check,
    operator_export_chain_check, portability_check, remediation_interop_check, review_truth_check,
    scope_authority_check, v0_gate, v1_gate, v2_gate, v3_gate, v4_gate, v5_gate, v6_gate,
    BundleRoundTripOverallStatusV1, DocsLintArgs, DocsLintMode, GateStatus,
    InteropMismatchCategoryV1, InteropOverallStatusV1, OperatorExportAuthorityChainStatusV1,
    OpsError, ReviewTruthCheckStatusV1, ScopeAuthorityOverallStatusV1,
    SupportedScopeReevaluationV1, V0GateOverallStatus, V1GateOverallStatus, V2GateOverallStatus,
    V3GateOverallStatus, V4GateOverallStatus, V5GateOverallStatus, V6GateOverallStatus,
};

const DIGEST_PREFIX_LEN: usize = 16;
const ROUNDTRIP_BUNDLE_CANDIDATES: [&str; 3] = [
    "out/repro_portability.zip",
    "out/repro_pack.zip",
    "out/bugkit_bundle.zip",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum V7GateOverallStatus {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct V7GateCheckV1 {
    pub name: String,
    pub status: GateStatus,
    pub evidence_digest_prefixes: BTreeMap<String, String>,
    pub remediation_hint_code: String,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct V7GateReportV1 {
    pub schema_version: u16,
    pub overall_status: V7GateOverallStatus,
    pub checks: Vec<V7GateCheckV1>,
}

pub fn v7_gate(workdir: &Path, out: &Path) -> Result<V7GateReportV1, OpsError> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut checks = Vec::new();

    let v0 = v0_gate(
        workdir,
        &repo_root.join("fixtures/e2e/v0_flow_a.json"),
        &workdir.join("out").join("v0_gate_report_v7_gate.json"),
    )?;
    checks.push(v7_gate_check(
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
        &workdir.join("out").join("v1_gate_report_v7_gate.json"),
    )?;
    checks.push(v7_gate_check(
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
        &workdir.join("out").join("v2_gate_report_v7_gate.json"),
    )?;
    checks.push(v7_gate_check(
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
        &workdir.join("out").join("v3_gate_report_v7_gate.json"),
    )?;
    checks.push(v7_gate_check(
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
        &workdir.join("out").join("v4_gate_report_v7_gate.json"),
    )?;
    checks.push(v7_gate_check(
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
        &workdir.join("out").join("v5_gate_report_v7_gate.json"),
    )?;
    checks.push(v7_gate_check(
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

    let v6 = v6_gate(
        workdir,
        &workdir.join("out").join("v6_gate_report_v7_gate.json"),
    )?;
    checks.push(v7_gate_check(
        "v6_gate_pass",
        if matches!(v6.overall_status, V6GateOverallStatus::Pass) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [("v6_gate_report".to_string(), digest_prefix(&v6)?)],
        "REMEDIATE_RUN_V6_GATE",
        "NOTE_REQUIRED_V6",
    ));

    let applied_scope = load_applied_supported_set_context_v1(workdir)?;

    let scope_authority = scope_authority_check(
        workdir,
        &workdir
            .join("out")
            .join("scope_authority_check_v7_gate.json"),
    )?;
    checks.push(v7_gate_check(
        "applied_scope_authority_pass",
        if matches!(scope_authority.status, ScopeAuthorityOverallStatusV1::Pass) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "scope_authority".to_string(),
            digest_prefix(&scope_authority)?,
        )],
        "REMEDIATE_APPLIED_SCOPE_AUTHORITY",
        "NOTE_REQUIRED_APPLIED_SCOPE_AUTHORITY",
    ));

    let reeval_path = workdir.join("out").join("supported_scope_reeval.json");
    let reeval = fs::read_to_string(&reeval_path)
        .ok()
        .and_then(|body| serde_json::from_str::<SupportedScopeReevaluationV1>(&body).ok());
    checks.push(v7_gate_check(
        "supported_scope_reevaluation_present",
        if reeval.is_some() {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "supported_scope_reevaluation".to_string(),
            if reeval_path.exists() {
                crate::prefix_hex(
                    &crate::sha256_hex(&fs::read(&reeval_path)?),
                    DIGEST_PREFIX_LEN,
                )
            } else {
                "missing".to_string()
            },
        )],
        "REMEDIATE_SUPPORTED_SCOPE_REEVALUATION",
        "NOTE_REQUIRED_SUPPORTED_SCOPE_REEVALUATION",
    ));

    let reeval_consistent = reeval
        .as_ref()
        .map(|r| {
            r.previous_applied_set_digest_prefix == applied_scope.applied_set_digest_prefix
                && r.policy_digest_prefix == applied_scope.policy_digest_prefix
        })
        .unwrap_or(false);
    checks.push(v7_gate_check(
        "supported_scope_reevaluation_consistent",
        if reeval_consistent {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "applied_scope_context".to_string(),
            crate::prefix_hex(&applied_scope.context_digest, DIGEST_PREFIX_LEN),
        )],
        "REMEDIATE_SUPPORTED_SCOPE_REEVALUATION_STALE",
        "NOTE_REQUIRED_SCOPE_REEVALUATION_COHERENCE",
    ));

    let truth = review_truth_check(
        workdir,
        &workdir.join("out").join("review_truth_check_v7_gate.json"),
    )?;
    checks.push(v7_gate_check(
        "reviewability_truth_pass",
        if matches!(truth.status, ReviewTruthCheckStatusV1::Pass) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [("review_truth".to_string(), digest_prefix(&truth)?)],
        "REMEDIATE_REVIEWABILITY_TRUTH",
        "NOTE_REQUIRED_REVIEW_TRUTH",
    ));

    let roundtrip_bundle = discover_roundtrip_bundle(workdir);
    let roundtrip = roundtrip_bundle
        .as_ref()
        .map(|bundle| {
            exports_roundtrip_check(
                bundle,
                &workdir
                    .join("out")
                    .join("export_roundtrip_check_v7_gate.json"),
            )
        })
        .transpose()?;
    checks.push(v7_gate_check(
        "export_roundtrip_pass",
        if roundtrip
            .as_ref()
            .is_some_and(|r| matches!(r.overall_status, BundleRoundTripOverallStatusV1::Pass))
        {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "roundtrip_bundle".to_string(),
            roundtrip_bundle
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "missing".to_string()),
        )],
        "REMEDIATE_EXPORT_ROUNDTRIP",
        "NOTE_REQUIRED_EXPORT_ROUNDTRIP",
    ));

    let remediation =
        remediation_interop_check(&workdir.join("out/remediation_interop_check_v7_gate.json"))?;
    checks.push(v7_gate_check(
        "remediation_interop_pass",
        if remediation.mismatches_found == 0 {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "remediation_interop".to_string(),
            digest_prefix(&remediation)?,
        )],
        "REMEDIATE_REMEDIATION_INTEROP",
        "NOTE_REQUIRED_REMEDIATION_INTEROP",
    ));

    let export_chain = operator_export_chain_check(
        workdir,
        &workdir.join("out/operator_export_chain_v7_gate.json"),
    )?;
    checks.push(v7_gate_check(
        "operator_export_chain_pass",
        if matches!(
            export_chain.authority_chain_status,
            OperatorExportAuthorityChainStatusV1::Pass
        ) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "operator_export_chain".to_string(),
            digest_prefix(&export_chain)?,
        )],
        "REMEDIATE_OPERATOR_EXPORT_CHAIN",
        "NOTE_REQUIRED_OPERATOR_EXPORT_CHAIN",
    ));

    let artifact_schema = check_artifact_schema_snapshots(&artifact_schema::ArtifactSchemaArgs {
        repo_root: repo_root.clone(),
        out_dir: repo_root.join("docs/artifact_schema_snapshots"),
    })?;
    checks.push(v7_gate_check(
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
        portability_check(&workdir.join("out").join("portability_check_v7_gate.json"))?;
    checks.push(v7_gate_check(
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

    let model_consistency = models_consistency_check(
        workdir,
        &workdir
            .join("out")
            .join("models_consistency_check_v7_gate.json"),
    )?;
    let optional_status = if model_consistency.checked_slots.is_empty() {
        GateStatus::Skip
    } else if model_consistency.status == "PASS" {
        GateStatus::Pass
    } else {
        GateStatus::Fail
    };
    checks.push(v7_gate_check(
        "optional_backend_path_consistent",
        optional_status,
        [(
            "optional_backend_mismatch_count".to_string(),
            model_consistency.mismatch_categories.len().to_string(),
        )],
        "REMEDIATE_MODELS_CONSISTENCY",
        "NOTE_OPTIONAL_BACKEND",
    ));

    let interop = interop_consistency_matrix(
        workdir,
        &workdir
            .join("out")
            .join("interop_consistency_matrix_v7_gate.json"),
    )?;
    let has_legacy_mismatch = interop
        .summary
        .mismatch_counts
        .iter()
        .any(|(kind, _)| matches!(kind, InteropMismatchCategoryV1::LegacySurfacePresent));
    let legacy_status = if matches!(interop.summary.overall_status, InteropOverallStatusV1::Fail)
        && has_legacy_mismatch
    {
        GateStatus::Fail
    } else {
        GateStatus::Skip
    };
    checks.push(v7_gate_check(
        "legacy_bundle_translation_ok",
        legacy_status,
        [(
            "legacy_surface_present".to_string(),
            has_legacy_mismatch.to_string(),
        )],
        "REMEDIATE_EXPORT_LEGACY_TRANSLATION",
        "NOTE_OPTIONAL_LEGACY_EXPORT",
    ));

    let overall_status = overall_from_checks(&checks);
    let report = V7GateReportV1 {
        schema_version: 1,
        overall_status,
        checks,
    };
    crate::write_json(out, &report)?;
    Ok(report)
}

fn discover_roundtrip_bundle(workdir: &Path) -> Option<PathBuf> {
    ROUNDTRIP_BUNDLE_CANDIDATES
        .iter()
        .map(|relative| workdir.join(relative))
        .find(|path| path.exists())
}

fn overall_from_checks(checks: &[V7GateCheckV1]) -> V7GateOverallStatus {
    if checks
        .iter()
        .all(|check| matches!(check.status, GateStatus::Pass | GateStatus::Skip))
    {
        V7GateOverallStatus::Pass
    } else {
        V7GateOverallStatus::Fail
    }
}

fn digest_prefix<T: Serialize>(value: &T) -> Result<String, OpsError> {
    Ok(crate::prefix_hex(
        &crate::sha256_hex(&serde_json::to_vec(value)?),
        DIGEST_PREFIX_LEN,
    ))
}

fn v7_gate_check(
    name: &str,
    status: GateStatus,
    evidence: impl IntoIterator<Item = (String, String)>,
    remediation_hint_code: &str,
    notes: &str,
) -> V7GateCheckV1 {
    V7GateCheckV1 {
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
    fn v7_gate_check_order_is_fixed() {
        let checks = vec![
            "v0_gate_pass",
            "v1_gate_pass",
            "v2_gate_pass",
            "v3_gate_pass",
            "v4_gate_pass",
            "v5_gate_pass",
            "v6_gate_pass",
            "applied_scope_authority_pass",
            "supported_scope_reevaluation_present",
            "supported_scope_reevaluation_consistent",
            "reviewability_truth_pass",
            "export_roundtrip_pass",
            "remediation_interop_pass",
            "operator_export_chain_pass",
            "artifact_schema_snapshot_checks_pass",
            "portability_docs_checks_pass",
            "optional_backend_path_consistent",
            "legacy_bundle_translation_ok",
        ];
        let report = V7GateReportV1 {
            schema_version: 1,
            overall_status: V7GateOverallStatus::Pass,
            checks: checks
                .iter()
                .map(|name| v7_gate_check(name, GateStatus::Pass, [], "REMEDIATE", "NOTE"))
                .collect(),
        };
        let names: Vec<String> = report.checks.into_iter().map(|c| c.name).collect();
        assert_eq!(names, checks);
    }

    #[test]
    fn v7_gate_report_serialization_is_deterministic() {
        let report = V7GateReportV1 {
            schema_version: 1,
            overall_status: V7GateOverallStatus::Pass,
            checks: vec![
                v7_gate_check(
                    "a",
                    GateStatus::Pass,
                    [("k".to_string(), "v".to_string())],
                    "REMEDIATE_A",
                    "NOTE_A",
                ),
                v7_gate_check(
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
    fn v7_gate_normalization_is_fail_closed() {
        let checks = vec![
            v7_gate_check(
                "required",
                GateStatus::Fail,
                [],
                "REMEDIATE_REQUIRED",
                "NOTE_REQUIRED",
            ),
            v7_gate_check(
                "optional",
                GateStatus::Skip,
                [],
                "REMEDIATE_OPTIONAL",
                "NOTE_OPTIONAL",
            ),
        ];
        assert!(matches!(
            overall_from_checks(&checks),
            V7GateOverallStatus::Fail
        ));
    }
}
