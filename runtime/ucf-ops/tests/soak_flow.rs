use tempfile::tempdir;
use ucf_ops::{parse_inject, soak_run, SoakRunArgs, SoakStatus};

#[test]
fn short_soak_with_injected_timeout_fails_and_writes_postmortem() {
    let dir = tempdir().expect("tempdir");
    let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    std::env::set_current_dir(workspace_root).expect("workspace cwd");

    let out = dir.path().join("soak");
    let report = soak_run(
        dir.path(),
        &SoakRunArgs {
            duration_secs: 120,
            scenario: "golden_a".to_string(),
            out: out.clone(),
            health_poll_secs: 5,
            memory_sample_secs: 60,
            inject: vec![parse_inject("timeout:llm@t=20").expect("inject")],
            postmortem: false,
        },
    )
    .expect("soak run");

    assert!(matches!(report.status, SoakStatus::Fail));
    assert!(out.join("soak_report.json").exists());
    assert!(out.join("soak_timeseries.json").exists());
    let bundle = report.postmortem_bundle.expect("bundle path");
    assert!(std::path::PathBuf::from(bundle).exists());
}
