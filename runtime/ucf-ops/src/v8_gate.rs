use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{
    artifact_schema, check_artifact_schema_snapshots, docs_lint, exports_bundle_spine_check,
    governance_entry_check, interop_consistency_matrix, load_applied_supported_set_context_v1,
    models_consistency_check, operator_roundtrip_chain_check, portability_check,
    readiness_spine_check, remediation_spine_check, v0_gate, v1_gate, v2_gate, v3_gate, v4_gate,
    v5_gate, v6_gate, v7_gate, BundleSpineStatusV1, CanonicalRoundTripChainStatusV1, DocsLintArgs,
    DocsLintMode, GateStatus, GovernanceEntryCheckStatusV1, InteropMismatchCategoryV1,
    InteropOverallStatusV1, OpsError, ReadinessSpineCheckStatusV1, V0GateOverallStatus,
    V1GateOverallStatus, V2GateOverallStatus, V3GateOverallStatus, V4GateOverallStatus,
    V5GateOverallStatus, V6GateOverallStatus, V7GateOverallStatus,
};

const DIGEST_PREFIX_LEN: usize = 16;
const ROUNDTRIP_BUNDLE_CANDIDATES: [&str; 3] = [
    "out/repro_portability.zip",
    "out/repro_pack.zip",
    "out/bugkit_bundle.zip",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum V8GateOverallStatus {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct V8GateCheckV1 {
    pub name: String,
    pub status: GateStatus,
    pub evidence_digest_prefixes: BTreeMap<String, String>,
    pub remediation_hint_code: String,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct V8GateReportV1 {
    pub schema_version: u16,
    pub overall_status: V8GateOverallStatus,
    pub checks: Vec<V8GateCheckV1>,
}

pub fn v8_gate(workdir: &Path, out: &Path) -> Result<V8GateReportV1, OpsError> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut checks = Vec::new();

    let v0 = v0_gate(
        workdir,
        &repo_root.join("fixtures/e2e/v0_flow_a.json"),
        &workdir.join("out/v0_gate_report_v8_gate.json"),
    )?;
    checks.push(v8_gate_check(
        "v0_gate_pass",
        gate_from_bool(matches!(v0.overall_status, V0GateOverallStatus::Pass)),
        [("v0_gate_report".to_string(), digest_prefix(&v0)?)],
        "REMEDIATE_RUN_V0_GATE",
        "NOTE_REQUIRED_V0",
    ));

    let v1 = v1_gate(workdir, &workdir.join("out/v1_gate_report_v8_gate.json"))?;
    checks.push(v8_gate_check(
        "v1_gate_pass",
        gate_from_bool(matches!(v1.overall_status, V1GateOverallStatus::Pass)),
        [("v1_gate_report".to_string(), digest_prefix(&v1)?)],
        "REMEDIATE_RUN_V1_GATE",
        "NOTE_REQUIRED_V1",
    ));

    let v2 = v2_gate(workdir, &workdir.join("out/v2_gate_report_v8_gate.json"))?;
    checks.push(v8_gate_check(
        "v2_gate_pass",
        gate_from_bool(matches!(v2.overall_status, V2GateOverallStatus::Pass)),
        [("v2_gate_report".to_string(), digest_prefix(&v2)?)],
        "REMEDIATE_RUN_V2_GATE",
        "NOTE_REQUIRED_V2",
    ));

    let v3 = v3_gate(workdir, &workdir.join("out/v3_gate_report_v8_gate.json"))?;
    checks.push(v8_gate_check(
        "v3_gate_pass",
        gate_from_bool(matches!(v3.overall_status, V3GateOverallStatus::Pass)),
        [("v3_gate_report".to_string(), digest_prefix(&v3)?)],
        "REMEDIATE_RUN_V3_GATE",
        "NOTE_REQUIRED_V3",
    ));

    let v4 = v4_gate(workdir, &workdir.join("out/v4_gate_report_v8_gate.json"))?;
    checks.push(v8_gate_check(
        "v4_gate_pass",
        gate_from_bool(matches!(v4.overall_status, V4GateOverallStatus::Pass)),
        [("v4_gate_report".to_string(), digest_prefix(&v4)?)],
        "REMEDIATE_RUN_V4_GATE",
        "NOTE_REQUIRED_V4",
    ));

    let v5 = v5_gate(workdir, &workdir.join("out/v5_gate_report_v8_gate.json"))?;
    checks.push(v8_gate_check(
        "v5_gate_pass",
        gate_from_bool(matches!(v5.overall_status, V5GateOverallStatus::Pass)),
        [("v5_gate_report".to_string(), digest_prefix(&v5)?)],
        "REMEDIATE_RUN_V5_GATE",
        "NOTE_REQUIRED_V5",
    ));

    let v6 = v6_gate(workdir, &workdir.join("out/v6_gate_report_v8_gate.json"))?;
    checks.push(v8_gate_check(
        "v6_gate_pass",
        gate_from_bool(matches!(v6.overall_status, V6GateOverallStatus::Pass)),
        [("v6_gate_report".to_string(), digest_prefix(&v6)?)],
        "REMEDIATE_RUN_V6_GATE",
        "NOTE_REQUIRED_V6",
    ));

    let v7 = v7_gate(workdir, &workdir.join("out/v7_gate_report_v8_gate.json"))?;
    checks.push(v8_gate_check(
        "v7_gate_pass",
        gate_from_bool(matches!(v7.overall_status, V7GateOverallStatus::Pass)),
        [("v7_gate_report".to_string(), digest_prefix(&v7)?)],
        "REMEDIATE_RUN_V7_GATE",
        "NOTE_REQUIRED_V7",
    ));

    let governance = governance_entry_check(
        workdir,
        &workdir.join("out/governance_entry_check_v8_gate.json"),
    )?;
    checks.push(v8_gate_check(
        "canonical_governance_entry_pass",
        gate_from_bool(matches!(
            governance.status,
            GovernanceEntryCheckStatusV1::Pass
        )),
        [(
            "governance_entry_check".to_string(),
            digest_prefix(&governance)?,
        )],
        "REMEDIATE_GOVERNANCE_ENTRY",
        "NOTE_REQUIRED_GOVERNANCE",
    ));

    let supported_scope_execution_path = workdir.join("out/supported_scope_execute_v3.json");
    let supported_scope_execution =
        load_supported_scope_execution(&supported_scope_execution_path)?;
    checks.push(v8_gate_check(
        "supported_scope_execution_present",
        gate_from_bool(supported_scope_execution.is_some()),
        [(
            "supported_scope_execution".to_string(),
            file_digest_or_missing(&supported_scope_execution_path)?,
        )],
        "REMEDIATE_RUN_SUPPORTED_SCOPE_EXECUTION",
        "NOTE_REQUIRED_SUPPORTED_SCOPE_EXECUTION",
    ));

    let applied_scope = load_applied_supported_set_context_v1(workdir)?;
    let supported_scope_consistent = supported_scope_execution
        .as_ref()
        .map(|exec| {
            exec.resulting_supported_set_digest_prefix == applied_scope.applied_set_digest_prefix
                && exec.previous_applied_set_digest_prefix
                    == applied_scope.previous_set_digest_prefix
                && exec.current_policy_digest_prefix == applied_scope.policy_digest_prefix
        })
        .unwrap_or(false);
    checks.push(v8_gate_check(
        "supported_scope_execution_consistent",
        gate_from_bool(supported_scope_consistent),
        [(
            "applied_scope_context".to_string(),
            crate::prefix_hex(&applied_scope.context_digest, DIGEST_PREFIX_LEN),
        )],
        "REMEDIATE_SUPPORTED_SCOPE_EXECUTION_STALE",
        "NOTE_REQUIRED_SUPPORTED_SCOPE_CURRENT",
    ));

    let readiness = readiness_spine_check(
        workdir,
        &workdir.join("out/readiness_spine_check_v8_gate.json"),
    )?;
    checks.push(v8_gate_check(
        "readiness_spine_pass",
        gate_from_bool(matches!(
            readiness.status,
            ReadinessSpineCheckStatusV1::Pass
        )),
        [("readiness_spine".to_string(), digest_prefix(&readiness)?)],
        "REMEDIATE_READINESS_SPINE",
        "NOTE_REQUIRED_READINESS_SPINE",
    ));

    let roundtrip_bundle = discover_roundtrip_bundle(workdir);
    let bundle_spine = roundtrip_bundle
        .as_ref()
        .map(|bundle| {
            exports_bundle_spine_check(bundle, &workdir.join("out/bundle_spine_check_v8_gate.json"))
        })
        .transpose()?;
    checks.push(v8_gate_check(
        "bundle_spine_pass",
        gate_from_bool(bundle_spine.as_ref().is_some_and(|report| {
            report.pass && matches!(report.spine.bundle_spine_status, BundleSpineStatusV1::Pass)
        })),
        [(
            "roundtrip_bundle".to_string(),
            roundtrip_bundle
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "missing".to_string()),
        )],
        "REMEDIATE_BUNDLE_SPINE",
        "NOTE_REQUIRED_BUNDLE_SPINE",
    ));

    let remediation =
        remediation_spine_check(&workdir.join("out/remediation_spine_check_v8_gate.json"))?;
    checks.push(v8_gate_check(
        "remediation_spine_pass",
        gate_from_bool(remediation.mismatches_found == 0),
        [(
            "remediation_spine".to_string(),
            digest_prefix(&remediation)?,
        )],
        "REMEDIATE_REMEDIATION_SPINE",
        "NOTE_REQUIRED_REMEDIATION_SPINE",
    ));

    let roundtrip_chain = roundtrip_bundle
        .as_ref()
        .map(|bundle| {
            operator_roundtrip_chain_check(
                workdir,
                bundle,
                &workdir.join("out/operator_roundtrip_chain_check_v8_gate.json"),
            )
        })
        .transpose()?;
    checks.push(v8_gate_check(
        "operator_roundtrip_chain_pass",
        gate_from_bool(roundtrip_chain.as_ref().is_some_and(|chain| {
            matches!(
                chain.roundtrip_status,
                CanonicalRoundTripChainStatusV1::Pass
            )
        })),
        [(
            "operator_roundtrip_chain".to_string(),
            roundtrip_chain
                .as_ref()
                .map(digest_prefix)
                .transpose()?
                .unwrap_or_else(|| "missing".to_string()),
        )],
        "REMEDIATE_OPERATOR_ROUNDTRIP_CHAIN",
        "NOTE_REQUIRED_ROUNDTRIP_CHAIN",
    ));

    let artifact_schema = check_artifact_schema_snapshots(&artifact_schema::ArtifactSchemaArgs {
        repo_root: repo_root.clone(),
        out_dir: repo_root.join("docs/artifact_schema_snapshots"),
    })?;
    checks.push(v8_gate_check(
        "artifact_schema_snapshot_checks_pass",
        gate_from_bool(artifact_schema.ok),
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
    let portability = portability_check(&workdir.join("out/portability_check_v8_gate.json"))?;
    checks.push(v8_gate_check(
        "portability_docs_checks_pass",
        gate_from_bool(docs.ok && portability.deterministic_within_os),
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
        &workdir.join("out/models_consistency_check_v8_gate.json"),
    )?;
    let optional_backend_status = if model_consistency.checked_slots.is_empty() {
        GateStatus::Skip
    } else if model_consistency.status == "PASS" {
        GateStatus::Pass
    } else {
        GateStatus::Fail
    };
    checks.push(v8_gate_check(
        "optional_backend_path_consistent",
        optional_backend_status,
        [(
            "optional_backend_mismatch_count".to_string(),
            model_consistency.mismatch_categories.len().to_string(),
        )],
        "REMEDIATE_MODELS_CONSISTENCY",
        "NOTE_OPTIONAL_BACKEND",
    ));

    let interop = interop_consistency_matrix(
        workdir,
        &workdir.join("out/interop_consistency_matrix_v8_gate.json"),
    )?;
    let has_legacy_bundle_path = interop.summary.mismatch_counts.iter().any(|(kind, count)| {
        *count > 0 && matches!(kind, InteropMismatchCategoryV1::LegacySurfacePresent)
    });
    checks.push(v8_gate_check(
        "legacy_bundle_translation_ok",
        if has_legacy_bundle_path {
            if matches!(interop.summary.overall_status, InteropOverallStatusV1::Pass) {
                GateStatus::Pass
            } else {
                GateStatus::Fail
            }
        } else {
            GateStatus::Skip
        },
        [(
            "legacy_surface_present".to_string(),
            has_legacy_bundle_path.to_string(),
        )],
        "REMEDIATE_EXPORT_LEGACY_TRANSLATION",
        "NOTE_OPTIONAL_LEGACY_EXPORT",
    ));

    let has_legacy_entry_path = governance.consumers.iter().any(|consumer| {
        consumer.mismatch_categories.iter().any(|category| {
            matches!(
                category,
                crate::GovernanceEntryMismatchCategoryV1::LegacyEntryPathPresent
                    | crate::GovernanceEntryMismatchCategoryV1::ConsumerUsedSecondaryEntry
            )
        })
    });
    checks.push(v8_gate_check(
        "legacy_entry_translation_ok",
        if has_legacy_entry_path {
            GateStatus::Fail
        } else {
            GateStatus::Skip
        },
        [(
            "legacy_entry_path_present".to_string(),
            has_legacy_entry_path.to_string(),
        )],
        "REMEDIATE_ENTRY_LEGACY_TRANSLATION",
        "NOTE_OPTIONAL_LEGACY_ENTRY",
    ));

    let report = V8GateReportV1 {
        schema_version: 1,
        overall_status: overall_from_checks(&checks),
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

fn load_supported_scope_execution(
    path: &Path,
) -> Result<Option<crate::SupportedScopeExecutionV3>, OpsError> {
    if !path.exists() {
        return Ok(None);
    }
    let body = fs::read(path)?;
    let report = serde_json::from_slice(&body)?;
    Ok(Some(report))
}

fn file_digest_or_missing(path: &Path) -> Result<String, OpsError> {
    if !path.exists() {
        return Ok("missing".to_string());
    }
    Ok(crate::prefix_hex(
        &crate::sha256_hex(&fs::read(path)?),
        DIGEST_PREFIX_LEN,
    ))
}

fn gate_from_bool(pass: bool) -> GateStatus {
    if pass {
        GateStatus::Pass
    } else {
        GateStatus::Fail
    }
}

fn overall_from_checks(checks: &[V8GateCheckV1]) -> V8GateOverallStatus {
    if checks
        .iter()
        .all(|check| matches!(check.status, GateStatus::Pass | GateStatus::Skip))
    {
        V8GateOverallStatus::Pass
    } else {
        V8GateOverallStatus::Fail
    }
}

fn digest_prefix<T: Serialize>(value: &T) -> Result<String, OpsError> {
    Ok(crate::prefix_hex(
        &crate::sha256_hex(&serde_json::to_vec(value)?),
        DIGEST_PREFIX_LEN,
    ))
}

fn v8_gate_check(
    name: &str,
    status: GateStatus,
    evidence: impl IntoIterator<Item = (String, String)>,
    remediation_hint_code: &str,
    notes: &str,
) -> V8GateCheckV1 {
    V8GateCheckV1 {
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
    fn v8_gate_check_order_is_fixed() {
        let checks = vec![
            "v0_gate_pass",
            "v1_gate_pass",
            "v2_gate_pass",
            "v3_gate_pass",
            "v4_gate_pass",
            "v5_gate_pass",
            "v6_gate_pass",
            "v7_gate_pass",
            "canonical_governance_entry_pass",
            "supported_scope_execution_present",
            "supported_scope_execution_consistent",
            "readiness_spine_pass",
            "bundle_spine_pass",
            "remediation_spine_pass",
            "operator_roundtrip_chain_pass",
            "artifact_schema_snapshot_checks_pass",
            "portability_docs_checks_pass",
            "optional_backend_path_consistent",
            "legacy_bundle_translation_ok",
            "legacy_entry_translation_ok",
        ];
        let report = V8GateReportV1 {
            schema_version: 1,
            overall_status: V8GateOverallStatus::Pass,
            checks: checks
                .iter()
                .map(|name| v8_gate_check(name, GateStatus::Pass, [], "REMEDIATE", "NOTE"))
                .collect(),
        };
        let names: Vec<String> = report.checks.into_iter().map(|c| c.name).collect();
        assert_eq!(names, checks);
    }

    #[test]
    fn v8_gate_report_serialization_is_deterministic() {
        let report = V8GateReportV1 {
            schema_version: 1,
            overall_status: V8GateOverallStatus::Pass,
            checks: vec![
                v8_gate_check(
                    "a",
                    GateStatus::Pass,
                    [("k".to_string(), "v".to_string())],
                    "REMEDIATE_A",
                    "NOTE_A",
                ),
                v8_gate_check(
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
    fn v8_gate_normalization_is_fail_closed() {
        let checks = vec![
            v8_gate_check(
                "required",
                GateStatus::Pass,
                [("e".to_string(), "1".to_string())],
                "REMEDIATE_REQUIRED",
                "NOTE_REQUIRED",
            ),
            v8_gate_check(
                "required_fail",
                GateStatus::Fail,
                [("e".to_string(), "2".to_string())],
                "REMEDIATE_REQUIRED_FAIL",
                "NOTE_REQUIRED_FAIL",
            ),
            v8_gate_check(
                "optional_skip",
                GateStatus::Skip,
                [("e".to_string(), "3".to_string())],
                "REMEDIATE_OPTIONAL",
                "NOTE_OPTIONAL",
            ),
        ];
        assert!(matches!(
            overall_from_checks(&checks),
            V8GateOverallStatus::Fail
        ));
    }

    #[test]
    fn v8_gate_normalization_supports_optional_skip_on_pass_path() {
        let checks = vec![
            v8_gate_check("required_a", GateStatus::Pass, [], "R", "N"),
            v8_gate_check("required_b", GateStatus::Pass, [], "R", "N"),
            v8_gate_check("optional", GateStatus::Skip, [], "R", "N"),
        ];
        assert!(matches!(
            overall_from_checks(&checks),
            V8GateOverallStatus::Pass
        ));
    }
}
