use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{
    artifact_schema, bundle_terminal_sweep, check_artifact_schema_snapshots, docs_lint,
    governance_terminal_sweep, load_applied_supported_set_context_v1, models_consistency_check,
    portability_check, primary_semantics_terminal_sweep, readiness_terminal_sweep,
    ultimate_terminal_absolute_final_input_continuity_sweep, v0_gate, v10_gate, v11_gate, v12_gate,
    v13_gate, v1_gate, v2_gate, v3_gate, v4_gate, v5_gate, v6_gate, v7_gate, v8_gate, v9_gate,
    AbsoluteFinalBundleTerminalSweepStatusV1, AbsoluteFinalGovernanceTerminalSweepStatusV1,
    AbsoluteFinalPrimarySemanticsTerminalSweepStatusV1,
    AbsoluteFinalReadinessTerminalSweepStatusV1, DocsLintArgs, DocsLintMode, GateStatus, OpsError,
    SupportedScopeExecutionV9, UltimateTerminalAbsoluteFinalInputContinuityStatusV1,
    V0GateOverallStatus, V10GateOverallStatus, V11GateOverallStatus, V12GateOverallStatus,
    V13GateOverallStatus, V1GateOverallStatus, V2GateOverallStatus, V3GateOverallStatus,
    V4GateOverallStatus, V5GateOverallStatus, V6GateOverallStatus, V7GateOverallStatus,
    V8GateOverallStatus, V9GateOverallStatus,
};

const DIGEST_PREFIX_LEN: usize = 16;
const CONTINUITY_BUNDLE_CANDIDATES: [&str; 4] = [
    "out/repro_portability.zip",
    "out/repro_pack.zip",
    "out/bugkit_bundle.zip",
    "out/repro_bundle.zip",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum V14GateOverallStatus {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct V14GateCheckV1 {
    pub name: String,
    pub status: GateStatus,
    pub evidence_digest_prefixes: BTreeMap<String, String>,
    pub remediation_hint_code: String,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct V14GateReportV1 {
    pub schema_version: u16,
    pub overall_status: V14GateOverallStatus,
    pub checks: Vec<V14GateCheckV1>,
}

pub fn v14_gate(workdir: &Path, out: &Path) -> Result<V14GateReportV1, OpsError> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut checks = Vec::new();

    let v0 = v0_gate(
        workdir,
        &repo_root.join("fixtures/e2e/v0_flow_a.json"),
        &workdir.join("out/v0_gate_report_v14_gate.json"),
    )?;
    checks.push(v14_gate_check(
        "v0_gate_pass",
        gate_from_bool(matches!(v0.overall_status, V0GateOverallStatus::Pass)),
        [("v0_gate_report".to_string(), digest_prefix(&v0)?)],
        "REMEDIATE_RUN_V0_GATE",
        "NOTE_REQUIRED_V0",
    ));

    let v1 = v1_gate(workdir, &workdir.join("out/v1_gate_report_v14_gate.json"))?;
    checks.push(v14_gate_check(
        "v1_gate_pass",
        gate_from_bool(matches!(v1.overall_status, V1GateOverallStatus::Pass)),
        [("v1_gate_report".to_string(), digest_prefix(&v1)?)],
        "REMEDIATE_RUN_V1_GATE",
        "NOTE_REQUIRED_V1",
    ));

    let v2 = v2_gate(workdir, &workdir.join("out/v2_gate_report_v14_gate.json"))?;
    checks.push(v14_gate_check(
        "v2_gate_pass",
        gate_from_bool(matches!(v2.overall_status, V2GateOverallStatus::Pass)),
        [("v2_gate_report".to_string(), digest_prefix(&v2)?)],
        "REMEDIATE_RUN_V2_GATE",
        "NOTE_REQUIRED_V2",
    ));

    let v3 = v3_gate(workdir, &workdir.join("out/v3_gate_report_v14_gate.json"))?;
    checks.push(v14_gate_check(
        "v3_gate_pass",
        gate_from_bool(matches!(v3.overall_status, V3GateOverallStatus::Pass)),
        [("v3_gate_report".to_string(), digest_prefix(&v3)?)],
        "REMEDIATE_RUN_V3_GATE",
        "NOTE_REQUIRED_V3",
    ));

    let v4 = v4_gate(workdir, &workdir.join("out/v4_gate_report_v14_gate.json"))?;
    checks.push(v14_gate_check(
        "v4_gate_pass",
        gate_from_bool(matches!(v4.overall_status, V4GateOverallStatus::Pass)),
        [("v4_gate_report".to_string(), digest_prefix(&v4)?)],
        "REMEDIATE_RUN_V4_GATE",
        "NOTE_REQUIRED_V4",
    ));

    let v5 = v5_gate(workdir, &workdir.join("out/v5_gate_report_v14_gate.json"))?;
    checks.push(v14_gate_check(
        "v5_gate_pass",
        gate_from_bool(matches!(v5.overall_status, V5GateOverallStatus::Pass)),
        [("v5_gate_report".to_string(), digest_prefix(&v5)?)],
        "REMEDIATE_RUN_V5_GATE",
        "NOTE_REQUIRED_V5",
    ));

    let v6 = v6_gate(workdir, &workdir.join("out/v6_gate_report_v14_gate.json"))?;
    checks.push(v14_gate_check(
        "v6_gate_pass",
        gate_from_bool(matches!(v6.overall_status, V6GateOverallStatus::Pass)),
        [("v6_gate_report".to_string(), digest_prefix(&v6)?)],
        "REMEDIATE_RUN_V6_GATE",
        "NOTE_REQUIRED_V6",
    ));

    let v7 = v7_gate(workdir, &workdir.join("out/v7_gate_report_v14_gate.json"))?;
    checks.push(v14_gate_check(
        "v7_gate_pass",
        gate_from_bool(matches!(v7.overall_status, V7GateOverallStatus::Pass)),
        [("v7_gate_report".to_string(), digest_prefix(&v7)?)],
        "REMEDIATE_RUN_V7_GATE",
        "NOTE_REQUIRED_V7",
    ));

    let v8 = v8_gate(workdir, &workdir.join("out/v8_gate_report_v14_gate.json"))?;
    checks.push(v14_gate_check(
        "v8_gate_pass",
        gate_from_bool(matches!(v8.overall_status, V8GateOverallStatus::Pass)),
        [("v8_gate_report".to_string(), digest_prefix(&v8)?)],
        "REMEDIATE_RUN_V8_GATE",
        "NOTE_REQUIRED_V8",
    ));

    let v9 = v9_gate(workdir, &workdir.join("out/v9_gate_report_v14_gate.json"))?;
    checks.push(v14_gate_check(
        "v9_gate_pass",
        gate_from_bool(matches!(v9.overall_status, V9GateOverallStatus::Pass)),
        [("v9_gate_report".to_string(), digest_prefix(&v9)?)],
        "REMEDIATE_RUN_V9_GATE",
        "NOTE_REQUIRED_V9",
    ));

    let v10 = v10_gate(workdir, &workdir.join("out/v10_gate_report_v14_gate.json"))?;
    checks.push(v14_gate_check(
        "v10_gate_pass",
        gate_from_bool(matches!(v10.overall_status, V10GateOverallStatus::Pass)),
        [("v10_gate_report".to_string(), digest_prefix(&v10)?)],
        "REMEDIATE_RUN_V10_GATE",
        "NOTE_REQUIRED_V10",
    ));

    let v11 = v11_gate(workdir, &workdir.join("out/v11_gate_report_v14_gate.json"))?;
    checks.push(v14_gate_check(
        "v11_gate_pass",
        gate_from_bool(matches!(v11.overall_status, V11GateOverallStatus::Pass)),
        [("v11_gate_report".to_string(), digest_prefix(&v11)?)],
        "REMEDIATE_RUN_V11_GATE",
        "NOTE_REQUIRED_V11",
    ));

    let v12 = v12_gate(workdir, &workdir.join("out/v12_gate_report_v14_gate.json"))?;
    checks.push(v14_gate_check(
        "v12_gate_pass",
        gate_from_bool(matches!(v12.overall_status, V12GateOverallStatus::Pass)),
        [("v12_gate_report".to_string(), digest_prefix(&v12)?)],
        "REMEDIATE_RUN_V12_GATE",
        "NOTE_REQUIRED_V12",
    ));

    let v13 = v13_gate(workdir, &workdir.join("out/v13_gate_report_v14_gate.json"))?;
    checks.push(v14_gate_check(
        "v13_gate_pass",
        gate_from_bool(matches!(v13.overall_status, V13GateOverallStatus::Pass)),
        [("v13_gate_report".to_string(), digest_prefix(&v13)?)],
        "REMEDIATE_RUN_V13_GATE",
        "NOTE_REQUIRED_V13",
    ));

    let governance = governance_terminal_sweep(
        workdir,
        &workdir.join("out/governance_terminal_sweep_v14_gate.json"),
    )?;
    checks.push(v14_gate_check(
        "terminal_absolute_residual_free_final_governance_inputs_pass",
        gate_from_bool(matches!(
            governance.sweep.sweep_status,
            AbsoluteFinalGovernanceTerminalSweepStatusV1::Pass
        )),
        [(
            "governance_terminal_sweep".to_string(),
            digest_prefix(&governance)?,
        )],
        "REMEDIATE_GOVERNANCE_TERMINAL_SWEEP",
        "NOTE_REQUIRED_TERMINAL_GOVERNANCE",
    ));

    let supported_scope_execution_path = workdir.join("out/supported_scope_execute_v9.json");
    let supported_scope_execution =
        load_supported_scope_execution_v9(&supported_scope_execution_path)?;
    checks.push(v14_gate_check(
        "supported_scope_execution_v9_present",
        gate_from_bool(supported_scope_execution.is_some()),
        [(
            "supported_scope_execution_v9".to_string(),
            file_digest_or_missing(&supported_scope_execution_path)?,
        )],
        "REMEDIATE_RUN_SUPPORTED_SCOPE_EXECUTION_V9",
        "NOTE_REQUIRED_SUPPORTED_SCOPE_EXECUTION_V9",
    ));

    let applied_scope = load_applied_supported_set_context_v1(workdir)?;
    let supported_scope_consistent = supported_scope_execution.as_ref().is_some_and(|exec| {
        exec.resulting_supported_set_digest_prefix == applied_scope.applied_set_digest_prefix
            && exec.previous_applied_set_digest_prefix == applied_scope.previous_set_digest_prefix
            && exec.current_policy_digest_prefix == applied_scope.policy_digest_prefix
            && exec.canonical_governance_entry_digest_prefix
                == governance.sweep.canonical_governance_entry_digest_prefix
            && exec.canonical_governance_authority_digest_prefix
                == governance
                    .sweep
                    .canonical_governance_authority_digest_prefix
            && exec.final_governance_consumer_authority_digest_prefix
                == governance
                    .sweep
                    .final_governance_consumer_authority_digest_prefix
            && exec.final_governance_residual_sweep_digest_prefix
                == governance
                    .sweep
                    .final_governance_residual_sweep_digest_prefix
            && exec.residual_free_governance_consumer_authority_digest_prefix
                == governance
                    .sweep
                    .residual_free_governance_consumer_authority_digest_prefix
            && exec.residual_free_governance_absolute_sweep_digest_prefix
                == governance
                    .sweep
                    .residual_free_governance_absolute_sweep_digest_prefix
            && exec.absolute_final_governance_terminal_sweep_digest_prefix
                == prefix_hex(&governance.sweep.sweep_digest)
    });
    checks.push(v14_gate_check(
        "supported_scope_execution_v9_consistent",
        gate_from_bool(supported_scope_consistent),
        [
            (
                "applied_scope_context".to_string(),
                prefix_hex(&applied_scope.context_digest),
            ),
            (
                "governance_terminal_sweep".to_string(),
                prefix_hex(&governance.sweep.sweep_digest),
            ),
        ],
        "REMEDIATE_SUPPORTED_SCOPE_EXECUTION_V9_STALE",
        "NOTE_REQUIRED_SUPPORTED_SCOPE_EXECUTION_V9_CURRENT",
    ));

    let readiness = readiness_terminal_sweep(
        workdir,
        &workdir.join("out/readiness_terminal_sweep_v14_gate.json"),
    )?;
    checks.push(v14_gate_check(
        "terminal_absolute_residual_free_final_readiness_inputs_pass",
        gate_from_bool(matches!(
            readiness.sweep.sweep_status,
            AbsoluteFinalReadinessTerminalSweepStatusV1::Pass
        )),
        [(
            "readiness_terminal_sweep".to_string(),
            digest_prefix(&readiness)?,
        )],
        "REMEDIATE_READINESS_TERMINAL_SWEEP",
        "NOTE_REQUIRED_TERMINAL_READINESS",
    ));

    let bundle = bundle_terminal_sweep(
        workdir,
        &workdir.join("out/bundle_terminal_sweep_v14_gate.json"),
    )?;
    checks.push(v14_gate_check(
        "terminal_absolute_residual_free_final_bundle_inputs_pass",
        gate_from_bool(matches!(
            bundle.sweep.sweep_status,
            AbsoluteFinalBundleTerminalSweepStatusV1::Pass
        )),
        [("bundle_terminal_sweep".to_string(), digest_prefix(&bundle)?)],
        "REMEDIATE_BUNDLE_TERMINAL_SWEEP",
        "NOTE_REQUIRED_TERMINAL_BUNDLE",
    ));

    let semantics = primary_semantics_terminal_sweep(
        workdir,
        &workdir.join("out/primary_semantics_terminal_sweep_v14_gate.json"),
    )?;
    checks.push(v14_gate_check(
        "terminal_absolute_residual_free_final_primary_semantics_inputs_pass",
        gate_from_bool(matches!(
            semantics.sweep.sweep_status,
            AbsoluteFinalPrimarySemanticsTerminalSweepStatusV1::Pass
        )),
        [(
            "primary_semantics_terminal_sweep".to_string(),
            digest_prefix(&semantics)?,
        )],
        "REMEDIATE_PRIMARY_SEMANTICS_TERMINAL_SWEEP",
        "NOTE_REQUIRED_TERMINAL_PRIMARY_SEMANTICS",
    ));

    let continuity_bundle = discover_continuity_bundle(workdir);
    let continuity = continuity_bundle
        .as_ref()
        .map(|bundle_path| {
            ultimate_terminal_absolute_final_input_continuity_sweep(
                workdir,
                bundle_path,
                &workdir.join(
                    "out/ultimate_terminal_absolute_final_input_continuity_sweep_v14_gate.json",
                ),
            )
        })
        .transpose()?;
    let continuity_status = if continuity_bundle.is_none() {
        GateStatus::Fail
    } else {
        gate_from_bool(continuity.as_ref().is_some_and(|report| {
            matches!(
                report.continuity_status,
                UltimateTerminalAbsoluteFinalInputContinuityStatusV1::Pass
            )
        }))
    };
    checks.push(v14_gate_check(
        "sole_terminal_absolute_residual_free_final_input_top_level_continuity_proof_pass",
        continuity_status,
        [(
            "continuity_bundle".to_string(),
            continuity_bundle
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "missing".to_string()),
        )],
        "REMEDIATE_ULTIMATE_TERMINAL_ABSOLUTE_FINAL_INPUT_CONTINUITY",
        "NOTE_REQUIRED_ULTIMATE_TERMINAL_ABSOLUTE_FINAL_INPUT_CONTINUITY",
    ));

    let artifact_schema = check_artifact_schema_snapshots(&artifact_schema::ArtifactSchemaArgs {
        repo_root: repo_root.clone(),
        out_dir: repo_root.join("docs/artifact_schema_snapshots"),
    })?;
    checks.push(v14_gate_check(
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
    let portability = portability_check(&workdir.join("out/portability_check_v14_gate.json"))?;
    checks.push(v14_gate_check(
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
        &workdir.join("out/models_consistency_check_v14_gate.json"),
    )?;
    let optional_backend_status = if model_consistency.checked_slots.is_empty() {
        GateStatus::Skip
    } else if model_consistency.status == "PASS" {
        GateStatus::Pass
    } else {
        GateStatus::Fail
    };
    checks.push(v14_gate_check(
        "optional_backend_path_consistent",
        optional_backend_status,
        [(
            "optional_backend_mismatch_count".to_string(),
            model_consistency.mismatch_categories.len().to_string(),
        )],
        "REMEDIATE_MODELS_CONSISTENCY",
        "NOTE_OPTIONAL_BACKEND",
    ));

    checks.push(v14_gate_check(
        "legacy_governance_path_translation_ok",
        legacy_status_from_governance(&governance),
        [(
            "legacy_governance_surface_present".to_string(),
            matches!(
                governance.sweep.sweep_status,
                AbsoluteFinalGovernanceTerminalSweepStatusV1::LegacyPresent
            )
            .to_string(),
        )],
        "REMEDIATE_LEGACY_GOVERNANCE_INPUTS",
        "NOTE_OPTIONAL_LEGACY_GOVERNANCE",
    ));

    checks.push(v14_gate_check(
        "legacy_readiness_path_translation_ok",
        legacy_status_from_readiness(&readiness),
        [(
            "legacy_readiness_surface_present".to_string(),
            matches!(
                readiness.sweep.sweep_status,
                AbsoluteFinalReadinessTerminalSweepStatusV1::LegacyPresent
            )
            .to_string(),
        )],
        "REMEDIATE_LEGACY_READINESS_INPUTS",
        "NOTE_OPTIONAL_LEGACY_READINESS",
    ));

    checks.push(v14_gate_check(
        "legacy_bundle_path_translation_ok",
        legacy_status_from_bundle(&bundle),
        [(
            "legacy_bundle_surface_present".to_string(),
            matches!(
                bundle.sweep.sweep_status,
                AbsoluteFinalBundleTerminalSweepStatusV1::LegacyPresent
            )
            .to_string(),
        )],
        "REMEDIATE_LEGACY_BUNDLE_INPUTS",
        "NOTE_OPTIONAL_LEGACY_BUNDLE",
    ));

    checks.push(v14_gate_check(
        "legacy_top_level_continuity_surface_demoted",
        continuity.as_ref().map_or(GateStatus::Skip, |report| {
            if matches!(
                report.continuity_status,
                UltimateTerminalAbsoluteFinalInputContinuityStatusV1::LegacyPresent
            ) {
                GateStatus::Fail
            } else {
                GateStatus::Pass
            }
        }),
        [(
            "legacy_top_level_continuity_present".to_string(),
            continuity
                .as_ref()
                .is_some_and(|report| {
                    matches!(
                        report.continuity_status,
                        UltimateTerminalAbsoluteFinalInputContinuityStatusV1::LegacyPresent
                    )
                })
                .to_string(),
        )],
        "REMEDIATE_LEGACY_TOP_LEVEL_CONTINUITY",
        "NOTE_OPTIONAL_LEGACY_CONTINUITY",
    ));

    let report = V14GateReportV1 {
        schema_version: 1,
        overall_status: overall_from_checks(&checks),
        checks,
    };
    crate::write_json(out, &report)?;
    Ok(report)
}

fn legacy_status_from_governance(report: &crate::GovernanceTerminalSweepReportV1) -> GateStatus {
    if matches!(
        report.sweep.sweep_status,
        AbsoluteFinalGovernanceTerminalSweepStatusV1::LegacyPresent
    ) {
        GateStatus::Fail
    } else {
        GateStatus::Pass
    }
}

fn legacy_status_from_readiness(report: &crate::ReadinessTerminalSweepReportV1) -> GateStatus {
    if matches!(
        report.sweep.sweep_status,
        AbsoluteFinalReadinessTerminalSweepStatusV1::LegacyPresent
    ) {
        GateStatus::Fail
    } else {
        GateStatus::Pass
    }
}

fn legacy_status_from_bundle(report: &crate::BundleTerminalSweepReportV1) -> GateStatus {
    if matches!(
        report.sweep.sweep_status,
        AbsoluteFinalBundleTerminalSweepStatusV1::LegacyPresent
    ) {
        GateStatus::Fail
    } else {
        GateStatus::Pass
    }
}

fn discover_continuity_bundle(workdir: &Path) -> Option<PathBuf> {
    CONTINUITY_BUNDLE_CANDIDATES
        .iter()
        .map(|relative| workdir.join(relative))
        .find(|path| path.exists())
}

fn load_supported_scope_execution_v9(
    path: &Path,
) -> Result<Option<SupportedScopeExecutionV9>, OpsError> {
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
    Ok(prefix_hex(&crate::sha256_hex(&fs::read(path)?)))
}

fn gate_from_bool(pass: bool) -> GateStatus {
    if pass {
        GateStatus::Pass
    } else {
        GateStatus::Fail
    }
}

fn overall_from_checks(checks: &[V14GateCheckV1]) -> V14GateOverallStatus {
    if checks
        .iter()
        .all(|check| matches!(check.status, GateStatus::Pass | GateStatus::Skip))
    {
        V14GateOverallStatus::Pass
    } else {
        V14GateOverallStatus::Fail
    }
}

fn digest_prefix<T: Serialize>(value: &T) -> Result<String, OpsError> {
    Ok(prefix_hex(&crate::sha256_hex(&serde_json::to_vec(value)?)))
}

fn prefix_hex(digest: &str) -> String {
    crate::prefix_hex(digest, DIGEST_PREFIX_LEN)
}

fn v14_gate_check(
    name: &str,
    status: GateStatus,
    evidence: impl IntoIterator<Item = (String, String)>,
    remediation_hint_code: &str,
    notes: &str,
) -> V14GateCheckV1 {
    V14GateCheckV1 {
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
    fn v14_gate_check_order_is_fixed() {
        let checks = vec![
            "v0_gate_pass",
            "v1_gate_pass",
            "v2_gate_pass",
            "v3_gate_pass",
            "v4_gate_pass",
            "v5_gate_pass",
            "v6_gate_pass",
            "v7_gate_pass",
            "v8_gate_pass",
            "v9_gate_pass",
            "v10_gate_pass",
            "v11_gate_pass",
            "v12_gate_pass",
            "v13_gate_pass",
            "terminal_absolute_residual_free_final_governance_inputs_pass",
            "supported_scope_execution_v9_present",
            "supported_scope_execution_v9_consistent",
            "terminal_absolute_residual_free_final_readiness_inputs_pass",
            "terminal_absolute_residual_free_final_bundle_inputs_pass",
            "terminal_absolute_residual_free_final_primary_semantics_inputs_pass",
            "sole_terminal_absolute_residual_free_final_input_top_level_continuity_proof_pass",
            "artifact_schema_snapshot_checks_pass",
            "portability_docs_checks_pass",
            "optional_backend_path_consistent",
            "legacy_governance_path_translation_ok",
            "legacy_readiness_path_translation_ok",
            "legacy_bundle_path_translation_ok",
            "legacy_top_level_continuity_surface_demoted",
        ];
        let report = V14GateReportV1 {
            schema_version: 1,
            overall_status: V14GateOverallStatus::Pass,
            checks: checks
                .iter()
                .map(|name| v14_gate_check(name, GateStatus::Pass, [], "REMEDIATE", "NOTE"))
                .collect(),
        };
        let names: Vec<String> = report.checks.into_iter().map(|c| c.name).collect();
        assert_eq!(names, checks);
    }

    #[test]
    fn v14_gate_report_serialization_is_deterministic() {
        let report = V14GateReportV1 {
            schema_version: 1,
            overall_status: V14GateOverallStatus::Pass,
            checks: vec![
                v14_gate_check(
                    "a",
                    GateStatus::Pass,
                    [("k".to_string(), "v".to_string())],
                    "REMEDIATE_A",
                    "NOTE_A",
                ),
                v14_gate_check(
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
    fn v14_gate_normalization_is_fail_closed() {
        let checks = vec![
            v14_gate_check("required", GateStatus::Pass, [], "R", "N"),
            v14_gate_check("required_fail", GateStatus::Fail, [], "R", "N"),
            v14_gate_check("optional_skip", GateStatus::Skip, [], "R", "N"),
        ];
        assert!(matches!(
            overall_from_checks(&checks),
            V14GateOverallStatus::Fail
        ));
    }

    #[test]
    fn v14_gate_normalization_supports_optional_skip_on_pass_path() {
        let checks = vec![
            v14_gate_check("required_a", GateStatus::Pass, [], "R", "N"),
            v14_gate_check("required_b", GateStatus::Pass, [], "R", "N"),
            v14_gate_check("optional", GateStatus::Skip, [], "R", "N"),
        ];
        assert!(matches!(
            overall_from_checks(&checks),
            V14GateOverallStatus::Pass
        ));
    }

    #[test]
    fn supported_scope_execution_v9_missing_is_none() {
        let temp = tempfile::tempdir().expect("tmp");
        let missing = temp.path().join("supported_scope_execute_v9.json");
        let loaded = load_supported_scope_execution_v9(&missing).expect("load");
        assert!(loaded.is_none());
    }

    #[test]
    fn required_terminal_checks_fail_normalization_cases() {
        for required in [
            "supported_scope_execution_v9_present",
            "terminal_absolute_residual_free_final_readiness_inputs_pass",
            "terminal_absolute_residual_free_final_bundle_inputs_pass",
            "terminal_absolute_residual_free_final_primary_semantics_inputs_pass",
            "sole_terminal_absolute_residual_free_final_input_top_level_continuity_proof_pass",
        ] {
            let checks = vec![
                v14_gate_check("v13_gate_pass", GateStatus::Pass, [], "R", "N"),
                v14_gate_check(required, GateStatus::Fail, [], "R", "N"),
                v14_gate_check(
                    "optional_backend_path_consistent",
                    GateStatus::Skip,
                    [],
                    "R",
                    "N",
                ),
            ];
            assert!(matches!(
                overall_from_checks(&checks),
                V14GateOverallStatus::Fail
            ));
        }
    }
}
