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
fn portability_report_v8_contains_spine_sections_in_stable_order() {
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
    let remediation_spine_idx = names
        .iter()
        .position(|c| c.contains("remediation-spine-check"))
        .expect("remediation spine command present");

    assert!(governance_idx < governance_entry_idx);
    assert!(governance_entry_idx < supported_reeval_idx);
    assert!(supported_reeval_idx < supported_apply_idx);
    assert!(supported_reeval_idx < supported_execute_idx);
    assert!(supported_execute_idx < readiness_spine_idx);
    assert!(readiness_spine_idx < supported_apply_idx);
    assert!(supported_apply_idx < applied_scope_idx);
    assert!(applied_scope_idx < export_normalize_idx);
    assert!(export_normalize_idx < interop_idx);
    assert!(bundle_spine_idx < remediation_spine_idx);
    assert!(interop_idx < active_idx);
    assert!(active_idx < backend_idx);
    assert!(backend_idx < repro_idx);
    assert!(repro_idx < bugkit_idx);
    assert!(bugkit_idx < remediation_idx);
}

#[test]
fn portability_report_v8_skips_optional_backend_resolution_cleanly() {
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
}
