use std::fs;
use std::sync::{Mutex, OnceLock};

use tempfile::tempdir;
use ucf_ops::{
    attest_bundle, attest_run, attest_verify, bringup, diagnostics, export_bugreport,
    load_signoff_checklist, models_promote, models_stage, one_command_bringup, out_manifest,
    parse_slot, readiness_gate, release_signoff_validate, replay_audit, replay_bugreport, v0_gate,
    verify_bugreport, world_shadow_report, ExportArgs, GateStatus, SpecSnapshotArgs,
};
use ucf_replay::{ReplayMode, ReplayStrictness};

fn gate_test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("runtime parent")
        .parent()
        .expect("repo root")
        .to_path_buf()
}

fn repo_path(rel: &str) -> std::path::PathBuf {
    repo_root().join(rel)
}

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
        repo_path("fixtures/e2e_scenario_a.json").as_path(),
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
    let _guard = gate_test_lock().lock().expect("lock");
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
    let checklist =
        load_signoff_checklist(repo_path("release/v0_signoff_checklist.toml").as_path())
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
        repo_path("release/v0_signoff_checklist.toml").as_path(),
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

#[test]
fn world_shadow_report_reads_bounded_windows() {
    let dir = tempdir().expect("tempdir");
    let run_id = "run-test";
    let report_dir = dir.path().join("reports/world_vljepa");
    fs::create_dir_all(&report_dir).expect("mkdir");
    let windows = [
        r#"{"window_id":0,"start_t":0,"end_t":511,"ticks":512,"latency_mean_ms":1.0,"latency_p95_ms":2.0,"error_mean_q":100,"error_p95_q":120,"error_delta_mean_q":10,"error_delta_p95_q":20,"invalid_rate":0.0,"saturation_rate":0.0}"#,
        r#"{"window_id":1,"start_t":512,"end_t":1023,"ticks":512,"latency_mean_ms":1.0,"latency_p95_ms":2.0,"error_mean_q":100,"error_p95_q":120,"error_delta_mean_q":10,"error_delta_p95_q":20,"invalid_rate":0.0,"saturation_rate":0.0}"#,
    ];
    fs::write(
        report_dir.join(format!("{}_windows.jsonl", run_id)),
        windows.join(
            "
",
        ),
    )
    .expect("windows");
    fs::write(report_dir.join(format!("{}_alarms.jsonl", run_id)), "").expect("alarms");
    fs::create_dir_all(dir.path().join("runs")).expect("runs");
    fs::write(
        dir.path().join("runs").join(format!("{run_id}.json")),
        r#"{"model_hashes_digest":"abc"}"#,
    )
    .expect("run meta");

    let out = dir.path().join("out/world_shadow_report.json");
    let report = world_shadow_report(dir.path(), run_id, 1, &out).expect("shadow report");
    assert_eq!(report.window_count, 1);
    assert_eq!(report.status, GateStatus::Pass);
}

#[test]
fn promotion_fails_when_staged_digest_mismatch() {
    let dir = tempdir().expect("tempdir");
    let cwd = std::env::current_dir().expect("cwd");
    std::env::set_current_dir(dir.path()).expect("chdir");

    let slot = parse_slot("llm").expect("slot");
    let src = dir.path().join("model_src");
    fs::create_dir_all(&src).expect("src");
    fs::write(src.join("model.safetensors"), b"stub").expect("model");
    let staged = models_stage(slot, &src).expect("stage");

    fs::write(
        dir.path()
            .join("models/staging/llm")
            .join(&staged.hash)
            .join("model.safetensors"),
        b"tampered",
    )
    .expect("tamper staged");

    let res = models_promote(slot, &staged.hash, None);
    assert!(res.is_err());
    std::env::set_current_dir(cwd).expect("restore cwd");
}

#[test]
fn spec_snapshot_writes_expected_sections() {
    let dir = tempdir().expect("tempdir");
    let out = dir.path().join("spec_snapshot.md");

    ucf_ops::generate_spec_snapshot(&SpecSnapshotArgs {
        policy: repo_path("policies/packs/base_v1"),
        overlay: Some(repo_path("policies/packs/overlays/test")),
        out: out.clone(),
    })
    .expect("snapshot");

    let body = fs::read_to_string(out).expect("read snapshot");
    assert!(body.contains("## A) Frames / Records"));
    assert!(body.contains("## B) Stage contracts"));
    assert!(body.contains("## D) Policy digests"));
    assert!(body.contains("## E) Model slots"));
}

#[test]
fn attest_run_and_verify_roundtrip() {
    let dir = tempdir().expect("tempdir");
    let out = dir.path().join("out");
    let artifacts = one_command_bringup(
        dir.path(),
        repo_path("fixtures/e2e_scenario_a.json").as_path(),
        12,
        &out,
        false,
    )
    .expect("bringup");

    let cert_path = out.join(format!("run_cert_{}.json", artifacts.run_metadata.run_id));
    let _cert =
        attest_run(dir.path(), &artifacts.run_metadata.run_id, &cert_path).expect("attest run");
    assert!(cert_path.exists());

    let verify = attest_verify(
        dir.path(),
        &cert_path,
        &dir.path().join("ess").join("ess_fixture.json"),
    )
    .expect("attest verify");
    assert!(verify.pass, "reasons={:?}", verify.reasons);
}

#[test]
fn attest_verify_fails_on_tamper() {
    let dir = tempdir().expect("tempdir");
    let out = dir.path().join("out");
    let artifacts = one_command_bringup(
        dir.path(),
        repo_path("fixtures/e2e_scenario_a.json").as_path(),
        12,
        &out,
        false,
    )
    .expect("bringup");

    let cert_path = out.join("run_cert_tamper.json");
    let _ = attest_run(dir.path(), &artifacts.run_metadata.run_id, &cert_path).expect("attest run");

    let mut cert_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&cert_path).expect("read cert")).expect("json");
    cert_json["manifest_digest"] = serde_json::Value::String("deadbeef".repeat(8));
    fs::write(
        &cert_path,
        serde_json::to_string_pretty(&cert_json).expect("serialize"),
    )
    .expect("write cert");

    let verify = attest_verify(
        dir.path(),
        &cert_path,
        &dir.path().join("ess").join("ess_fixture.json"),
    )
    .expect("attest verify");
    assert!(!verify.pass);
    assert!(!verify.reasons.is_empty());
}

#[test]
fn attest_bundle_exports_redaction_safe_artifacts() {
    let dir = tempdir().expect("tempdir");
    let out = dir.path().join("out");
    let artifacts = one_command_bringup(
        dir.path(),
        repo_path("fixtures/e2e_scenario_a.json").as_path(),
        10,
        &out,
        false,
    )
    .expect("bringup");

    let bundle_path = out.join("bundle.zip");
    let bundle =
        attest_bundle(dir.path(), &artifacts.run_metadata.run_id, &bundle_path).expect("bundle");
    assert!(bundle_path.exists());
    assert!(bundle.entries.iter().any(|e| e == "run_certificate.json"));
    assert!(bundle.entries.iter().any(|e| e == "segment_roots.json"));
}

#[test]
fn v0_gate_writes_deterministic_check_order() {
    let dir = tempdir().expect("tempdir");
    let out = dir.path().join("v0_gate_report.json");
    let scenario = repo_path("fixtures/e2e/v0_flow_a.json");
    let report = v0_gate(dir.path(), &scenario, &out).expect("v0 gate");

    assert!(out.exists());
    assert_eq!(report.schema_version, 1);
    assert_eq!(
        report
            .checks
            .iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>(),
        vec![
            "policy_graph_lock",
            "determinism_double_run",
            "e2e_flow_a",
            "record_boundedness",
            "schema_versions_known",
            "no_tool_execution",
        ]
    );
}
