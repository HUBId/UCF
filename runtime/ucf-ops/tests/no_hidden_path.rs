use std::path::PathBuf;

#[test]
fn no_hidden_paths_in_strict_scan() {
    let report = ucf_ops::audit_scan(&PathBuf::from("../..")).expect("audit scan");
    assert!(
        report.violations.is_empty(),
        "forbidden paths found: {:#?}",
        report.violations
    );
}
