use std::fs;

#[test]
fn determinism_scan_finds_disallowed_rng_usage() {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = dir.path().join("src");
    fs::create_dir_all(&src).expect("mkdir");
    fs::write(
        src.join("lib.rs"),
        "fn x(){ let _ = rand::random::<u32>(); }\n",
    )
    .expect("write");
    let report = ucf_ops::determinism_scan(dir.path()).expect("scan");
    assert!(!report.violations.is_empty());
    assert!(report
        .violations
        .iter()
        .any(|v| v.pattern == "rand::random"));
}
