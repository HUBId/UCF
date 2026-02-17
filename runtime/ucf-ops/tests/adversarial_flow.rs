use tempfile::tempdir;
use ucf_ops::{adversarial_run, AdversarialRunArgs};

#[test]
fn adversarial_suite_v1_runs_and_reports_cases() {
    let dir = tempdir().expect("tempdir");
    let out = dir.path().join("adversarial_report.json");
    let report = adversarial_run(&AdversarialRunArgs {
        workdir: dir.path().to_path_buf(),
        suite: "v1".to_string(),
        out: out.clone(),
    })
    .expect("adversarial run");

    assert!(out.exists());
    assert!(report.cases.len() >= 5);
    assert!(report.cases.iter().all(|c| !c.name.is_empty()));
}
