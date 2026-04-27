use std::process::Command;
use std::{fs, path::Path};

fn compute_elapsed_secs_from_report(path: &Path) -> f64 {
    let raw = fs::read_to_string(path).expect("read ucf-bench output report");
    let value: serde_json::Value = serde_json::from_str(&raw).expect("parse ucf-bench output json");
    let n = value["stats"]["n"]
        .as_u64()
        .expect("stats.n must be a positive integer");
    let throughput = value["stats"]["throughput_ops_sec"]
        .as_f64()
        .expect("stats.throughput_ops_sec must be a float");
    assert!(throughput > 0.0, "throughput_ops_sec must be > 0");
    n as f64 / throughput
}

#[test]
fn compute_fixture_smoke_under_four_seconds() {
    let out = std::env::temp_dir().join("ucf-bench-compute-smoke.json");
    let status = Command::new(env!("CARGO_BIN_EXE_ucf-bench"))
        .args([
            "compute",
            "--cases",
            "fixtures",
            "--backend",
            "stub",
            "--out",
            out.to_str().expect("tmp path"),
        ])
        .status()
        .expect("run ucf-bench compute");
    assert!(status.success());
    let elapsed_secs = compute_elapsed_secs_from_report(out.as_path());
    assert!(
        elapsed_secs < 4.0,
        "compute fixtures smoke exceeded 4s budget (elapsed={elapsed_secs:.3}s)"
    );
}

#[test]
fn sandbox_inproc_echo_smoke_runs() {
    let out = std::env::temp_dir().join("ucf-bench-sandbox-smoke.json");
    let status = Command::new(env!("CARGO_BIN_EXE_ucf-bench"))
        .args([
            "sandbox",
            "--runtime",
            "inproc",
            "--cases",
            "echo",
            "--n",
            "32",
            "--out",
            out.to_str().expect("tmp path"),
        ])
        .status()
        .expect("run ucf-bench sandbox");
    assert!(status.success());
}
