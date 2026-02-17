use std::fs;

use tempfile::tempdir;
use ucf_ops::{
    bringup, diagnostics, export_bugreport, one_command_bringup, readiness_gate, replay_audit,
    replay_bugreport, verify_bugreport, ExportArgs, GateStatus,
};
use ucf_replay::{ReplayMode, ReplayStrictness};

#[test]
fn bringup_demo_is_deterministic() {
    let left = tempdir().expect("left");
    let right = tempdir().expect("right");

    let first = bringup(left.path(), true, 25).expect("bringup first");
    let second = bringup(right.path(), true, 25).expect("bringup second");

    assert_eq!(first.ess_digest, second.ess_digest);
}

#[test]
fn diagnostics_pass_after_bringup() {
    let dir = tempdir().expect("tempdir");
    bringup(dir.path(), true, 10).expect("bringup");

    let diag = diagnostics(dir.path()).expect("diag");
    assert!(diag.ok());
}

#[test]
fn replay_bugreport_generates_report() {
    let dir = tempdir().expect("tempdir");
    bringup(dir.path(), true, 12).expect("bringup");

    let report_dir = export_bugreport(
        dir.path(),
        &ExportArgs {
            last: Some(8),
            include_sandbox: false,
            include_audit: false,
        },
    )
    .expect("export");

    verify_bugreport(&report_dir).expect("verify");
    let replay_report = replay_bugreport(&report_dir, ReplayMode::ComputeOnly).expect("replay");
    let body = fs::read_to_string(replay_report).expect("read replay report");
    assert!(body.contains("\"total_items\""));
}

#[test]
fn replay_audit_writes_bounded_report() {
    let dir = tempdir().expect("tempdir");
    bringup(dir.path(), true, 16).expect("bringup");

    let report_path = dir.path().join("audit_replay.json");
    replay_audit(
        dir.path(),
        1,
        16,
        ReplayStrictness::VerifyOnly,
        false,
        &report_path,
    )
    .expect("replay audit");

    let body = fs::read_to_string(report_path).expect("read report");
    assert!(body.contains("\"overall_status\""));
    assert!(body.contains("\"details\""));
}

#[test]
fn one_command_bringup_writes_release_artifacts() {
    let dir = tempdir().expect("tempdir");
    let out = dir.path().join("out");
    let artifacts = one_command_bringup(
        dir.path(),
        std::path::Path::new("../../fixtures/e2e_scenario_a.json"),
        16,
        &out,
        true,
    )
    .expect("one-command bringup");

    assert!(!artifacts.run_metadata.code_version_tag.is_empty());
    assert!(out.join("metrics_summary.json").exists());
    assert!(out.join("run_metadata_record.json").exists());
    assert!(dir
        .path()
        .join("ess")
        .join("run_metadata_record.json")
        .exists());
}

#[test]
fn readiness_gate_writes_report() {
    std::env::set_var("UCF_SKIP_GATE_WORKSPACE_TESTS", "1");
    let dir = tempdir().expect("tempdir");
    let out = dir.path().join("gate_report.json");
    let report = readiness_gate(dir.path(), "test", &out).expect("readiness gate");
    std::env::remove_var("UCF_SKIP_GATE_WORKSPACE_TESTS");

    assert!(out.exists());
    assert!(!report.checks.is_empty());
    assert!(matches!(report.status, GateStatus::Pass | GateStatus::Fail));
}
