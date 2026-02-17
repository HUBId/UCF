use std::fs;

use tempfile::tempdir;
use ucf_ops::{
    bringup, diagnostics, export_bugreport, load_signoff_checklist, one_command_bringup,
    out_manifest, readiness_gate, release_signoff_validate, replay_audit, replay_bugreport,
    verify_bugreport, ExportArgs, GateStatus,
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
    assert!(out.join("run_metadata.json").exists());
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

#[test]
fn checklist_toml_parses() {
    let checklist = load_signoff_checklist(std::path::Path::new(
        "../../release/v0_signoff_checklist.toml",
    ))
    .expect("checklist");
    assert_eq!(checklist.version, "v0");
    assert!(!checklist.items.is_empty());
}

#[test]
fn release_signoff_validate_fixture_out_dir() {
    let dir = tempdir().expect("tempdir");
    let out = dir.path().join("out/run-1");
    fs::create_dir_all(&out).expect("out dir");
    fs::write(out.join("run_metadata.json"), b"{}").expect("run metadata");
    fs::write(out.join("gate_report.json"), b"{}").expect("gate report");
    fs::write(out.join("adversarial_report.json"), b"{}").expect("adv report");
    fs::write(out.join("bench_report.json"), b"{}").expect("bench report");
    fs::write(out.join("replay_report.json"), b"{}").expect("replay report");
    fs::write(out.join("explain_tick_last.json"), b"{}").expect("explain report");
    fs::write(out.join("probe_report.json"), b"{}").expect("probe report");
    fs::write(out.join("snapshot.snap"), b"snap").expect("snapshot");

    let emit = dir.path().join("signoff_result.json");
    let report = release_signoff_validate(
        &out,
        std::path::Path::new("../../release/v0_signoff_checklist.toml"),
        &emit,
    )
    .expect("validate");

    assert!(report.pass);
    assert!(emit.exists());
    let manifest = out_manifest(&out).expect("manifest");
    assert!(manifest
        .entries
        .iter()
        .any(|e| e.file == "run_metadata.json"));
}
