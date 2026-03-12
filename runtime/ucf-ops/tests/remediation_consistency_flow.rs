use tempfile::tempdir;
use ucf_ops::{remediation_consistency_check, RemediationConsistencyStatusV1};

#[test]
fn remediation_consistency_report_is_deterministic_and_classified() {
    let dir = tempdir().expect("tempdir");
    let out = dir.path().join("remediation_consistency.json");
    let report = remediation_consistency_check(&out).expect("report should build");

    assert!(out.exists());
    assert_eq!(report.summary.total_conditions, report.checks.len());
    assert!(report
        .checks
        .iter()
        .any(|c| matches!(c.status, RemediationConsistencyStatusV1::Pass)));
    assert!(report
        .checks
        .iter()
        .any(|c| matches!(c.status, RemediationConsistencyStatusV1::Missing)));
    assert!(!report
        .checks
        .iter()
        .any(|c| matches!(c.status, RemediationConsistencyStatusV1::Fail)));
}
