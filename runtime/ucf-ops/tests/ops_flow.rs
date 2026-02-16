use std::fs;

use tempfile::tempdir;
use ucf_ops::{
    bringup, diagnostics, export_bugreport, replay_audit, replay_bugreport, verify_bugreport,
    ExportArgs,
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
