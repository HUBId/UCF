use std::path::PathBuf;

#[test]
fn portability_check_runs_and_writes_report() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("out/portability.json");
    let report = ucf_ops::portability_check(&out).expect("portability check");
    assert!(out.exists());
    assert!(!report.digest_prefixes.is_empty());
    assert!(report.fixed_point_summary.sample_count <= 1024);
    assert!(report.digest_prefixes.contains_key("ess_run_a"));
}

#[test]
fn path_scan_is_clean_for_repo() {
    let report = ucf_ops::path_scan(&PathBuf::from("../..")).expect("path scan");
    assert!(
        report.violations.is_empty(),
        "forbidden paths found: {:#?}",
        report.violations
    );
}
