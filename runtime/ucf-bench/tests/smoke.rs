use std::process::Command;
use std::time::{Duration, Instant};

#[test]
fn compute_fixture_smoke_under_four_seconds() {
    let out = std::env::temp_dir().join("ucf-bench-compute-smoke.json");
    let start = Instant::now();
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
    assert!(
        start.elapsed() < Duration::from_secs(4),
        "compute fixtures smoke exceeded 4s budget"
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
