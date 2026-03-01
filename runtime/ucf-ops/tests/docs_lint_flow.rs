use std::fs;

use tempfile::tempdir;
use ucf_ops::{docs_lint, DocsLintArgs, DocsLintMode, DocsLintStatus};

#[test]
fn docs_lint_passes_on_repo_docs() {
    let report = docs_lint(&DocsLintArgs {
        repo_root: std::path::PathBuf::from("../.."),
        policy_pack: std::path::PathBuf::from("../../policies/packs/base_v1"),
        overlay_pack: Some(std::path::PathBuf::from(
            "../../policies/packs/overlays/test",
        )),
        spec_snapshot: std::path::PathBuf::from("../../docs/spec_snapshot.md"),
        prompt_index: std::path::PathBuf::from("../../docs/prompt_series_index.md"),
        module_map: std::path::PathBuf::from("../../docs/module_map.md"),
        deploy_doc: std::path::PathBuf::from("../../docs/deploy_portable.md"),
        mode: DocsLintMode::Strict,
    })
    .expect("docs lint should run");

    assert!(report.ok, "report should pass: {report:?}");
}

#[test]
fn docs_lint_fails_on_snapshot_mismatch() {
    let dir = tempdir().expect("tempdir");
    let bad_snapshot = dir.path().join("spec_snapshot.md");
    fs::write(&bad_snapshot, "# stale snapshot\n").expect("write snapshot");

    let report = docs_lint(&DocsLintArgs {
        repo_root: std::path::PathBuf::from("../.."),
        policy_pack: std::path::PathBuf::from("../../policies/packs/base_v1"),
        overlay_pack: Some(std::path::PathBuf::from(
            "../../policies/packs/overlays/test",
        )),
        spec_snapshot: bad_snapshot,
        prompt_index: std::path::PathBuf::from("../../docs/prompt_series_index.md"),
        module_map: std::path::PathBuf::from("../../docs/module_map.md"),
        deploy_doc: std::path::PathBuf::from("../../docs/deploy_portable.md"),
        mode: DocsLintMode::Strict,
    })
    .expect("docs lint should run");

    let snapshot = report
        .checks
        .iter()
        .find(|c| c.name == "spec_snapshot")
        .expect("spec snapshot check exists");
    assert_eq!(snapshot.status, DocsLintStatus::Fail);
    assert!(!report.ok);
}

#[test]
fn docs_lint_fails_on_hardware_terms_in_core_docs() {
    let dir = tempdir().expect("tempdir");
    let bad_prompt_index = dir.path().join("prompt_series_index.md");
    fs::write(
        &bad_prompt_index,
        "# Prompt Series Index\nCore NUC target\n",
    )
    .expect("write");

    let report = docs_lint(&DocsLintArgs {
        repo_root: std::path::PathBuf::from("../.."),
        policy_pack: std::path::PathBuf::from("../../policies/packs/base_v1"),
        overlay_pack: Some(std::path::PathBuf::from(
            "../../policies/packs/overlays/test",
        )),
        spec_snapshot: std::path::PathBuf::from("../../docs/spec_snapshot.md"),
        prompt_index: bad_prompt_index,
        module_map: std::path::PathBuf::from("../../docs/module_map.md"),
        deploy_doc: std::path::PathBuf::from("../../docs/deploy_portable.md"),
        mode: DocsLintMode::Strict,
    })
    .expect("docs lint should run");

    let hardware = report
        .checks
        .iter()
        .find(|c| c.name == "hardware_neutral_docs")
        .expect("hardware check exists");
    assert_eq!(hardware.status, DocsLintStatus::Fail);
}
