use tempfile::tempdir;
use ucf_ops::gateway_threat_test;

#[test]
fn gateway_threat_test_writes_report() {
    let dir = tempdir().expect("tempdir");
    let out = dir.path().join("gateway_threat.json");
    let report = gateway_threat_test(&out).expect("threat report");

    assert!(out.exists());
    assert!(report.ok);
    assert_eq!(report.cases.len(), 4);
    assert!(report.abuse_log_total > 0);
}
