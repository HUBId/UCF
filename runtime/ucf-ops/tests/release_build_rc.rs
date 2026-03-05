use std::process::Command;

#[test]
#[ignore = "manual release flow smoke; runs cargo build --release"]
fn release_build_rc_fast_smoke() {
    let status = Command::new("cargo")
        .args([
            "run",
            "-p",
            "ucf-ops",
            "--",
            "release",
            "build-rc",
            "--version",
            "v0.0-rc0",
            "--profile",
            "prod",
            "--out",
            "./out/rc_ci_smoke",
            "--fast",
        ])
        .status()
        .expect("spawn release build-rc --fast");
    assert!(status.success());
}
