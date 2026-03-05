use std::fs;

use tempfile::tempdir;
use ucf_console::{export_view, load_snapshot, ConsoleConfig, ViewTab};

#[test]
fn once_snapshot_works_without_gateway() {
    let dir = tempdir().expect("tmp");
    let cfg = ConsoleConfig {
        workdir: dir.path().to_path_buf(),
        alerts_path: dir.path().join("alerts.json"),
        drift_path: dir.path().join("drift.json"),
        ..ConsoleConfig::default()
    };

    let snap = load_snapshot(&cfg).expect("snapshot");
    assert!(!snap.overview.status.is_empty());

    let out = dir.path().join("console_once.json");
    export_view(&snap, ViewTab::Overview, &out).expect("export");
    let raw = fs::read_to_string(out).expect("read");
    assert!(raw.contains("status"));
}
