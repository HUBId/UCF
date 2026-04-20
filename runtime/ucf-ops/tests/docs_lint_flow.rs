use std::fs;
use std::path::{Path, PathBuf};

use tempfile::tempdir;
use ucf_ops::{
    docs_lint, generate_artifact_schema_snapshots, ArtifactSchemaArgs, DocsLintArgs, DocsLintMode,
    DocsLintStatus,
};

fn has_workspace_manifest(path: &Path) -> bool {
    let manifest = path.join("Cargo.toml");
    let Ok(contents) = fs::read_to_string(manifest) else {
        return false;
    };
    contents.contains("[workspace]")
}

fn repo_root() -> PathBuf {
    let start = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for candidate in start.ancestors() {
        if has_workspace_manifest(candidate) {
            return candidate.to_path_buf();
        }
    }

    panic!(
        "failed to locate workspace root from CARGO_MANIFEST_DIR={}",
        env!("CARGO_MANIFEST_DIR")
    );
}

fn repo_path(rel: &str) -> std::path::PathBuf {
    repo_root().join(rel)
}

#[test]
fn docs_lint_passes_on_repo_docs() {
    let report = docs_lint(&DocsLintArgs {
        repo_root: repo_root(),
        policy_pack: repo_path("policies/packs/base_v1"),
        overlay_pack: Some(repo_path("policies/packs/overlays/test")),
        spec_snapshot: repo_path("docs/spec_snapshot.md"),
        prompt_index: repo_path("docs/prompt_series_index.md"),
        module_map: repo_path("docs/module_map.md"),
        deploy_doc: repo_path("docs/deploy_portable.md"),
        artifact_schema_snapshot_dir: repo_path("docs/artifact_schema_snapshots"),
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
        repo_root: repo_root(),
        policy_pack: repo_path("policies/packs/base_v1"),
        overlay_pack: Some(repo_path("policies/packs/overlays/test")),
        spec_snapshot: bad_snapshot,
        prompt_index: repo_path("docs/prompt_series_index.md"),
        module_map: repo_path("docs/module_map.md"),
        deploy_doc: repo_path("docs/deploy_portable.md"),
        artifact_schema_snapshot_dir: repo_path("docs/artifact_schema_snapshots"),
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
        "# Prompt Series Index\nCore Xeon target\n",
    )
    .expect("write");

    let report = docs_lint(&DocsLintArgs {
        repo_root: repo_root(),
        policy_pack: repo_path("policies/packs/base_v1"),
        overlay_pack: Some(repo_path("policies/packs/overlays/test")),
        spec_snapshot: repo_path("docs/spec_snapshot.md"),
        prompt_index: bad_prompt_index,
        module_map: repo_path("docs/module_map.md"),
        deploy_doc: repo_path("docs/deploy_portable.md"),
        artifact_schema_snapshot_dir: repo_path("docs/artifact_schema_snapshots"),
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

#[test]
fn docs_lint_fails_on_artifact_schema_snapshot_mismatch() {
    let dir = tempdir().expect("tempdir");
    let snapshot_dir = dir.path().join("artifact_schema_snapshots");
    let repo = repo_root();
    generate_artifact_schema_snapshots(&ArtifactSchemaArgs {
        repo_root: repo.clone(),
        out_dir: snapshot_dir.clone(),
    })
    .expect("generate snapshots");
    fs::write(
        snapshot_dir.join("operator_report_v1.json"),
        r#"{
  "stale": true
}
"#,
    )
    .expect("write stale");

    let report = docs_lint(&DocsLintArgs {
        repo_root: repo_root(),
        policy_pack: repo_path("policies/packs/base_v1"),
        overlay_pack: Some(repo_path("policies/packs/overlays/test")),
        spec_snapshot: repo_path("docs/spec_snapshot.md"),
        prompt_index: repo_path("docs/prompt_series_index.md"),
        module_map: repo_path("docs/module_map.md"),
        deploy_doc: repo_path("docs/deploy_portable.md"),
        artifact_schema_snapshot_dir: snapshot_dir,
        mode: DocsLintMode::Strict,
    })
    .expect("docs lint should run");

    let check = report
        .checks
        .iter()
        .find(|c| c.name == "artifact_schema_snapshots")
        .expect("artifact schema check exists");
    assert_eq!(check.status, DocsLintStatus::Fail);
}
