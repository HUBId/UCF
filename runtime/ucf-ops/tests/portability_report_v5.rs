use std::path::PathBuf;

struct CwdGuard {
    prev: PathBuf,
}

impl CwdGuard {
    fn enter(path: &std::path::Path) -> Self {
        let prev = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(path).expect("chdir");
        Self { prev }
    }
}

impl Drop for CwdGuard {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.prev);
    }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("runtime parent")
        .parent()
        .expect("repo root")
        .to_path_buf()
}

#[test]
fn portability_report_v11_contains_final_sweep_sections_in_stable_order() {
    let _guard = CwdGuard::enter(&repo_root());
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("out/portability_report.json");
    let report = ucf_ops::portability_report(dir.path(), &out).expect("portability report");
    assert_eq!(report.schema_version, 1);
    assert!(out.exists());

    let names = report
        .command_matrix
        .iter()
        .map(|entry| entry.command.as_str())
        .collect::<Vec<_>>();
    let active_idx = names
        .iter()
        .position(|c| c.contains("models active-review-snapshot"))
        .expect("active-review command present");
    let backend_idx = names
        .iter()
        .position(|c| c.contains("models backend-resolution"))
        .expect("backend-resolution command present");
    let repro_idx = names
        .iter()
        .position(|c| c.contains("repro pack"))
        .expect("repro command present");
    let bugkit_idx = names
        .iter()
        .position(|c| c.contains("bugkit build"))
        .expect("bugkit command present");
    let remediation_idx = names
        .iter()
        .position(|c| c.contains("remediation-consistency-check"))
        .expect("remediation consistency command present");
    let governance_idx = names
        .iter()
        .position(|c| c.contains("governance-surfaces-check"))
        .expect("governance surfaces command present");
    let governance_entry_idx = names
        .iter()
        .position(|c| c.contains("governance-entry-check"))
        .expect("governance entry command present");
    let governance_entry_sweep_idx = names
        .iter()
        .position(|c| c.contains("governance-entry-sweep"))
        .expect("governance entry sweep command present");
    let supported_reeval_idx = names
        .iter()
        .position(|c| c.contains("models supported-scope-reevaluate"))
        .expect("supported-scope-reevaluate command present");
    let supported_execute_idx = names
        .iter()
        .position(|c| c.contains("models supported-scope-execute"))
        .expect("supported-scope-execute command present");
    let readiness_spine_idx = names
        .iter()
        .position(|c| c.contains("readiness-spine-check"))
        .expect("readiness-spine-check command present");
    let readiness_spine_sweep_idx = names
        .iter()
        .position(|c| c.contains("readiness-spine-sweep"))
        .expect("readiness-spine-sweep command present");
    let supported_apply_idx = names
        .iter()
        .position(|c| c.contains("models supported-set-apply"))
        .expect("supported-set-apply command present");
    let applied_scope_idx = names
        .iter()
        .position(|c| c.contains("models applied-scope-check"))
        .expect("applied-scope-check command present");
    let export_normalize_idx = names
        .iter()
        .position(|c| c.contains("exports normalize-check"))
        .expect("normalize-check command present");
    let interop_idx = names
        .iter()
        .position(|c| c.contains("interop consistency-matrix"))
        .expect("interop matrix command present");
    let bundle_spine_idx = names
        .iter()
        .position(|c| c.contains("exports bundle-spine-check"))
        .expect("bundle spine command present");
    let bundle_spine_sweep_idx = names
        .iter()
        .position(|c| c.contains("exports bundle-spine-sweep"))
        .expect("bundle spine sweep command present");
    let primary_semantics_sweep_idx = names
        .iter()
        .position(|c| c.contains("primary-semantics-sweep"))
        .expect("primary semantics sweep command present");
    let remediation_spine_idx = names
        .iter()
        .position(|c| c.contains("remediation-spine-check"))
        .expect("remediation spine command present");
    let governance_residual_idx = names
        .iter()
        .position(|c| c.contains("governance-residual-sweep"))
        .expect("governance residual sweep command present");
    let residual_free_governance_idx = names
        .iter()
        .position(|c| c.contains("residual-free-governance-sweep"))
        .expect("residual-free-governance-sweep command present");
    let supported_execute_v8_idx = names
        .iter()
        .position(|c| c.contains("models supported-scope-execute-v8"))
        .expect("supported-scope-execute-v8 command present");
    let governance_terminal_idx = names
        .iter()
        .position(|c| c.contains("governance-terminal-sweep"))
        .expect("governance-terminal-sweep command present");
    let governance_ultimate_idx = names
        .iter()
        .position(|c| c.contains("governance-ultimate-sweep"))
        .expect("governance-ultimate-sweep command present");
    let supported_execute_v9_idx = names
        .iter()
        .position(|c| c.contains("models supported-scope-execute-v9"))
        .expect("supported-scope-execute-v9 command present");
    let supported_execute_v10_idx = names
        .iter()
        .position(|c| c.contains("models supported-scope-execute-v10"))
        .expect("supported-scope-execute-v10 command present");
    let readiness_residual_idx = names
        .iter()
        .position(|c| c.contains("readiness-residual-sweep"))
        .expect("readiness-residual-sweep command present");
    let residual_free_readiness_idx = names
        .iter()
        .position(|c| c.contains("residual-free-readiness-sweep"))
        .expect("residual-free-readiness-sweep command present");
    let readiness_terminal_idx = names
        .iter()
        .position(|c| c.contains("readiness-terminal-sweep"))
        .expect("readiness-terminal-sweep command present");
    let readiness_ultimate_idx = names
        .iter()
        .position(|c| c.contains("readiness-ultimate-sweep"))
        .expect("readiness-ultimate-sweep command present");
    let bundle_residual_idx = names
        .iter()
        .position(|c| c.contains("bundle-residual-sweep"))
        .expect("bundle-residual-sweep command present");
    let residual_free_bundle_idx = names
        .iter()
        .position(|c| c.contains("residual-free-bundle-sweep"))
        .expect("residual-free-bundle-sweep command present");
    let bundle_terminal_idx = names
        .iter()
        .position(|c| c.contains("bundle-terminal-sweep"))
        .expect("bundle-terminal-sweep command present");
    let bundle_ultimate_idx = names
        .iter()
        .position(|c| c.contains("bundle-ultimate-sweep"))
        .expect("bundle-ultimate-sweep command present");
    let primary_residual_idx = names
        .iter()
        .position(|c| c.contains("primary-semantics-residual-sweep"))
        .expect("primary-semantics-residual-sweep command present");
    let residual_free_primary_idx = names
        .iter()
        .position(|c| c.contains("residual-free-primary-semantics-sweep"))
        .expect("residual-free-primary-semantics-sweep command present");
    let primary_terminal_idx = names
        .iter()
        .position(|c| c.contains("primary-semantics-terminal-sweep"))
        .expect("primary-semantics-terminal-sweep command present");
    let primary_ultimate_idx = names
        .iter()
        .position(|c| c.contains("primary-semantics-ultimate-sweep"))
        .expect("primary-semantics-ultimate-sweep command present");
    let primary_convergence_idx = names
        .iter()
        .position(|c| c.contains("primary-semantics-convergence-sweep"))
        .expect("primary-semantics-convergence-sweep command present");

    assert!(governance_idx < governance_entry_idx);
    assert!(governance_entry_idx < governance_entry_sweep_idx);
    assert!(governance_entry_idx < supported_reeval_idx);
    assert!(governance_entry_sweep_idx < supported_reeval_idx);
    assert!(supported_reeval_idx < supported_apply_idx);
    assert!(supported_reeval_idx < supported_execute_idx);
    assert!(supported_execute_idx < readiness_spine_idx);
    assert!(readiness_spine_idx < readiness_spine_sweep_idx);
    assert!(readiness_spine_idx < supported_apply_idx);
    assert!(supported_apply_idx < applied_scope_idx);
    assert!(applied_scope_idx < export_normalize_idx);
    assert!(export_normalize_idx < interop_idx);
    assert!(bundle_spine_idx < remediation_spine_idx);
    assert!(bundle_spine_idx < bundle_spine_sweep_idx);
    assert!(bundle_spine_sweep_idx < primary_semantics_sweep_idx);
    assert!(governance_idx < governance_residual_idx);
    assert!(governance_residual_idx < residual_free_governance_idx);
    assert!(residual_free_governance_idx < governance_terminal_idx);
    assert!(governance_terminal_idx < governance_ultimate_idx);
    assert!(governance_terminal_idx < supported_execute_v8_idx);
    assert!(supported_execute_v8_idx < supported_execute_v9_idx);
    assert!(supported_execute_v9_idx < supported_execute_v10_idx);
    assert!(supported_execute_v8_idx < readiness_residual_idx);
    assert!(readiness_residual_idx < residual_free_readiness_idx);
    assert!(residual_free_readiness_idx < readiness_terminal_idx);
    assert!(readiness_terminal_idx < readiness_ultimate_idx);
    assert!(readiness_terminal_idx < bundle_residual_idx);
    assert!(bundle_residual_idx < residual_free_bundle_idx);
    assert!(residual_free_bundle_idx < bundle_terminal_idx);
    assert!(bundle_terminal_idx < bundle_ultimate_idx);
    assert!(bundle_terminal_idx < primary_residual_idx);
    assert!(residual_free_bundle_idx < primary_residual_idx);
    assert!(primary_residual_idx < residual_free_primary_idx);
    assert!(residual_free_primary_idx < primary_terminal_idx);
    assert!(primary_terminal_idx < primary_ultimate_idx);
    assert!(primary_ultimate_idx < primary_convergence_idx);
    assert!(interop_idx < active_idx);
    assert!(active_idx < backend_idx);
    assert!(backend_idx < repro_idx);
    assert!(repro_idx < bugkit_idx);
    assert!(bugkit_idx < remediation_idx);
}

#[test]
fn portability_report_v11_skips_optional_backend_resolution_cleanly() {
    let _guard = CwdGuard::enter(&repo_root());
    let dir = tempfile::tempdir().expect("tempdir");
    let out = PathBuf::from(dir.path()).join("out/portability_report.json");
    let report = ucf_ops::portability_report(dir.path(), &out).expect("portability report");
    assert!(
        matches!(
            report.backend_resolution_smoke.status,
            ucf_ops::PortabilityGateStatus::Pass | ucf_ops::PortabilityGateStatus::Skip
        ),
        "backend_resolution_smoke must PASS or SKIP"
    );
    if matches!(
        report.backend_resolution_smoke.status,
        ucf_ops::PortabilityGateStatus::Skip
    ) {
        assert!(report
            .backend_resolution_smoke
            .detail
            .contains("optional backend path unavailable"));
    }
    assert!(
        matches!(
            report.readiness_spine_check_smoke.status,
            ucf_ops::PortabilityGateStatus::Pass | ucf_ops::PortabilityGateStatus::Skip
        ),
        "readiness_spine_check_smoke must PASS or SKIP"
    );
    assert!(
        matches!(
            report.remediation_spine_check_smoke.status,
            ucf_ops::PortabilityGateStatus::Pass | ucf_ops::PortabilityGateStatus::Skip
        ),
        "remediation_spine_check_smoke must PASS or SKIP"
    );
    assert!(
        matches!(
            report.governance_entry_sweep_smoke.status,
            ucf_ops::PortabilityGateStatus::Pass | ucf_ops::PortabilityGateStatus::Skip
        ),
        "governance_entry_sweep_smoke must PASS or SKIP"
    );
    assert!(
        matches!(
            report.readiness_spine_sweep_smoke.status,
            ucf_ops::PortabilityGateStatus::Pass | ucf_ops::PortabilityGateStatus::Skip
        ),
        "readiness_spine_sweep_smoke must PASS or SKIP"
    );
    assert!(
        matches!(
            report.bundle_spine_sweep_smoke.status,
            ucf_ops::PortabilityGateStatus::Pass | ucf_ops::PortabilityGateStatus::Skip
        ),
        "bundle_spine_sweep_smoke must PASS or SKIP"
    );
    assert!(
        matches!(
            report.governance_residual_sweep_smoke.status,
            ucf_ops::PortabilityGateStatus::Pass | ucf_ops::PortabilityGateStatus::Skip
        ),
        "governance_residual_sweep_smoke must PASS or SKIP"
    );
    assert!(
        matches!(
            report.residual_free_governance_sweep_smoke.status,
            ucf_ops::PortabilityGateStatus::Pass | ucf_ops::PortabilityGateStatus::Skip
        ),
        "residual_free_governance_sweep_smoke must PASS or SKIP"
    );
    assert!(
        matches!(
            report.governance_ultimate_sweep_smoke.status,
            ucf_ops::PortabilityGateStatus::Pass | ucf_ops::PortabilityGateStatus::Skip
        ),
        "governance_ultimate_sweep_smoke must PASS or SKIP"
    );
    assert!(
        matches!(
            report.supported_scope_execute_v9_smoke.status,
            ucf_ops::PortabilityGateStatus::Pass | ucf_ops::PortabilityGateStatus::Skip
        ),
        "supported_scope_execute_v9_smoke must PASS or SKIP"
    );
    assert!(
        matches!(
            report.supported_scope_execute_v10_smoke.status,
            ucf_ops::PortabilityGateStatus::Pass | ucf_ops::PortabilityGateStatus::Skip
        ),
        "supported_scope_execute_v10_smoke must PASS or SKIP"
    );
    assert!(
        matches!(
            report.governance_terminal_sweep_smoke.status,
            ucf_ops::PortabilityGateStatus::Pass | ucf_ops::PortabilityGateStatus::Skip
        ),
        "governance_terminal_sweep_smoke must PASS or SKIP"
    );
    assert!(
        matches!(
            report.readiness_residual_sweep_smoke.status,
            ucf_ops::PortabilityGateStatus::Pass | ucf_ops::PortabilityGateStatus::Skip
        ),
        "readiness_residual_sweep_smoke must PASS or SKIP"
    );
    assert!(
        matches!(
            report.residual_free_readiness_sweep_smoke.status,
            ucf_ops::PortabilityGateStatus::Pass | ucf_ops::PortabilityGateStatus::Skip
        ),
        "residual_free_readiness_sweep_smoke must PASS or SKIP"
    );
    assert!(
        matches!(
            report.readiness_terminal_sweep_smoke.status,
            ucf_ops::PortabilityGateStatus::Pass | ucf_ops::PortabilityGateStatus::Skip
        ),
        "readiness_terminal_sweep_smoke must PASS or SKIP"
    );
    assert!(
        matches!(
            report.readiness_ultimate_sweep_smoke.status,
            ucf_ops::PortabilityGateStatus::Pass | ucf_ops::PortabilityGateStatus::Skip
        ),
        "readiness_ultimate_sweep_smoke must PASS or SKIP"
    );
    assert!(
        matches!(
            report.bundle_residual_sweep_smoke.status,
            ucf_ops::PortabilityGateStatus::Pass | ucf_ops::PortabilityGateStatus::Skip
        ),
        "bundle_residual_sweep_smoke must PASS or SKIP"
    );
    assert!(
        matches!(
            report.residual_free_bundle_sweep_smoke.status,
            ucf_ops::PortabilityGateStatus::Pass | ucf_ops::PortabilityGateStatus::Skip
        ),
        "residual_free_bundle_sweep_smoke must PASS or SKIP"
    );
    assert!(
        matches!(
            report.bundle_terminal_sweep_smoke.status,
            ucf_ops::PortabilityGateStatus::Pass | ucf_ops::PortabilityGateStatus::Skip
        ),
        "bundle_terminal_sweep_smoke must PASS or SKIP"
    );
    assert!(
        matches!(
            report.bundle_ultimate_sweep_smoke.status,
            ucf_ops::PortabilityGateStatus::Pass | ucf_ops::PortabilityGateStatus::Skip
        ),
        "bundle_ultimate_sweep_smoke must PASS or SKIP"
    );
    assert!(
        matches!(
            report.primary_semantics_residual_sweep_smoke.status,
            ucf_ops::PortabilityGateStatus::Pass | ucf_ops::PortabilityGateStatus::Skip
        ),
        "primary_semantics_residual_sweep_smoke must PASS or SKIP"
    );
    assert!(
        matches!(
            report.residual_free_primary_semantics_sweep_smoke.status,
            ucf_ops::PortabilityGateStatus::Pass | ucf_ops::PortabilityGateStatus::Skip
        ),
        "residual_free_primary_semantics_sweep_smoke must PASS or SKIP"
    );
    assert!(
        matches!(
            report.primary_semantics_ultimate_sweep_smoke.status,
            ucf_ops::PortabilityGateStatus::Pass | ucf_ops::PortabilityGateStatus::Skip
        ),
        "primary_semantics_ultimate_sweep_smoke must PASS or SKIP"
    );
    assert!(
        matches!(
            report.primary_semantics_terminal_sweep_smoke.status,
            ucf_ops::PortabilityGateStatus::Pass | ucf_ops::PortabilityGateStatus::Skip
        ),
        "primary_semantics_terminal_sweep_smoke must PASS or SKIP"
    );
}
