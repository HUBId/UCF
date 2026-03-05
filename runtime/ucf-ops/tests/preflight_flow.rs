use std::fs;
use std::path::{Path, PathBuf};

use tempfile::tempdir;
use ucf_ops::{policy_validate, preflight, GateStatus};
use walkdir::WalkDir;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("runtime parent")
        .parent()
        .expect("repo root")
        .to_path_buf()
}

fn copy_tree(src: &Path, dst: &Path) {
    for entry in WalkDir::new(src).into_iter().flatten() {
        let rel = entry.path().strip_prefix(src).expect("relative");
        let target = dst.join(rel);
        if entry.file_type().is_dir() {
            fs::create_dir_all(&target).expect("mkdir");
        } else {
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent).expect("mkdir parent");
            }
            fs::copy(entry.path(), &target).expect("copy file");
        }
    }
}

fn make_bundle_fixture() -> (tempfile::TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let bundle = dir.path().join("bundle");
    fs::create_dir_all(bundle.join("bin")).expect("bin");
    fs::write(bundle.join("bin/ucf-ops"), b"stub").expect("stub binary");

    let repo = repo_root();
    copy_tree(&repo.join("configs"), &bundle.join("configs"));
    copy_tree(&repo.join("policies"), &bundle.join("policies"));
    copy_tree(&repo.join("models"), &bundle.join("models"));
    let mut manifest_body =
        fs::read_to_string(bundle.join("models/manifest.toml")).expect("read manifest");
    manifest_body.push_str("\n# promoted/\n");
    fs::write(bundle.join("models/manifest.toml"), manifest_body).expect("write fixture manifest");

    fs::create_dir_all(bundle.join("out")).expect("out");
    fs::write(
        bundle.join("out/gate_latest.json"),
        r#"{"status":"PASS","checks":[],"code_version_tag":"fixture"}"#,
    )
    .expect("gate");

    let manifest_digest = {
        use sha2::Digest;
        let body = fs::read(bundle.join("models/manifest.toml")).expect("manifest");
        let mut hasher = sha2::Sha256::new();
        hasher.update(body);
        hex::encode(hasher.finalize())
    };
    let policy_graph_digest = policy_validate(
        &bundle.join("policies/packs/base_v1"),
        Some(&bundle.join("policies/packs/overlays/test")),
    )
    .expect("policy validate")
    .policy_graph_digest;
    let version = format!(
        "code_version_tag=test\npolicy_graph_digest={policy_graph_digest}\nmanifest_digest={manifest_digest}\nprofile=test\n"
    );
    fs::write(bundle.join("VERSION.txt"), version).expect("version");
    (dir, bundle)
}

#[test]
fn preflight_passes_on_known_good_fixture() {
    let (_tmp, bundle) = make_bundle_fixture();
    let out = bundle.join("out/preflight.json");
    let report = preflight(&bundle, &out).expect("preflight report");

    assert_eq!(report.schema_version, 1);
    assert_eq!(report.overall, GateStatus::Pass);
    assert_eq!(report.exit_code, 0);
    let order: Vec<_> = report.checks.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(
        order,
        vec![
            "bundle_integrity",
            "strict_check",
            "docs_lint",
            "gate_status",
            "runtime_status",
            "rc_manifest"
        ]
    );
}

#[test]
fn preflight_fails_critical_on_tampered_version_manifest_digest() {
    let (_tmp, bundle) = make_bundle_fixture();
    fs::write(
        bundle.join("VERSION.txt"),
        "code_version_tag=test\npolicy_graph_digest=dummy\nmanifest_digest=deadbeef\nprofile=test\n",
    )
    .expect("tamper version");

    let out = bundle.join("out/preflight_tamper.json");
    let report = preflight(&bundle, &out).expect("preflight");
    assert_eq!(report.overall, GateStatus::Fail);
    assert_eq!(report.exit_code, 3);
}
