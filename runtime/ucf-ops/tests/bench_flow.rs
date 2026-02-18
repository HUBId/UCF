use tempfile::tempdir;
use ucf_ops::{bench_run, BenchArgs};

#[test]
fn bench_command_produces_report() {
    let dir = tempdir().expect("tempdir");
    let out = dir.path().join("bench_report.json");
    let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    std::env::set_current_dir(workspace_root).expect("set cwd to workspace root");
    let report = bench_run(&BenchArgs {
        scenario: std::path::PathBuf::from("fixtures/e2e_scenario_a.json"),
        ticks: 16,
        out: out.clone(),
        rss_sample_every: 4,
        rss_cap_mb: None,
    })
    .expect("bench run");

    assert!(out.exists());
    assert_eq!(report.schema_version, 1);
    assert_eq!(report.ticks, 16);
    assert!(report.stage_latency_ms.contains_key("world"));
    assert!(report.stage_latency_ms.contains_key("llm"));
    assert!(report.stage_latency_ms.contains_key("ebm"));
    assert!(report.counters.ebm_candidates_scored_total > 0);
}
