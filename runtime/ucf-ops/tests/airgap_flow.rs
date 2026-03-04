use std::fs;
use std::io::Write;
use std::path::PathBuf;

use tempfile::tempdir;
use ucf_ops::{
    airgap_export_policies, airgap_import, AirgapArtifactType, AirgapImportArgs, AirgapImportMode,
};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("runtime parent")
        .parent()
        .expect("repo root")
        .to_path_buf()
}

fn repo_path(rel: &str) -> PathBuf {
    repo_root().join(rel)
}

#[test]
fn airgap_export_zip_order_is_sorted() {
    let dir = tempdir().expect("tempdir");
    let out = dir.path().join("policies.zip");
    let _ = airgap_export_policies(
        dir.path(),
        repo_path("policies/packs/base_v1").as_path(),
        Some(repo_path("policies/packs/overlays/test").as_path()),
        &out,
    )
    .expect("export");

    let mut archive = zip::ZipArchive::new(fs::File::open(out).expect("open zip")).expect("zip");
    let mut names = Vec::new();
    for i in 0..archive.len() {
        names.push(archive.by_index(i).expect("entry").name().to_string());
    }
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(names, sorted);
}

#[test]
fn airgap_import_detects_tamper() {
    let dir = tempdir().expect("tempdir");
    let out = dir.path().join("policies.zip");
    let _ = airgap_export_policies(
        dir.path(),
        repo_path("policies/packs/base_v1").as_path(),
        Some(repo_path("policies/packs/overlays/test").as_path()),
        &out,
    )
    .expect("export");

    let tampered = dir.path().join("tampered.zip");
    let mut source = zip::ZipArchive::new(fs::File::open(out).expect("open zip")).expect("zip");
    let mut target = zip::ZipWriter::new(fs::File::create(&tampered).expect("create"));
    let opts = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    for i in 0..source.len() {
        let mut entry = source.by_index(i).expect("entry");
        let name = entry.name().to_string();
        let mut bytes = Vec::new();
        std::io::copy(&mut entry, &mut bytes).expect("copy");
        if name.ends_with("pack_manifest.toml") {
            bytes.extend_from_slice(b"\n#tamper\n");
        }
        target.start_file(name, opts).expect("start");
        target.write_all(&bytes).expect("write");
    }
    target.finish().expect("finish");

    let report = airgap_import(
        dir.path(),
        &AirgapImportArgs {
            artifact_type: AirgapArtifactType::Policies,
            input: tampered,
            out: dir.path().join("import.json"),
            mode: AirgapImportMode::Staging,
            policy_pack: repo_path("policies/packs/base_v1"),
            policy_overlay: Some(repo_path("policies/packs/overlays/test")),
            strict_signer: false,
        },
    )
    .expect("import report");
    assert!(!report.pass);
}

#[test]
fn airgap_import_rejects_untrusted_signer_in_strict_mode() {
    let dir = tempdir().expect("tempdir");
    let out = dir.path().join("policies.zip");
    let _ = airgap_export_policies(
        dir.path(),
        repo_path("policies/packs/base_v1").as_path(),
        Some(repo_path("policies/packs/overlays/test").as_path()),
        &out,
    )
    .expect("export");

    let report = airgap_import(
        dir.path(),
        &AirgapImportArgs {
            artifact_type: AirgapArtifactType::Policies,
            input: out,
            out: dir.path().join("import.json"),
            mode: AirgapImportMode::Staging,
            policy_pack: repo_path("policies/packs/base_v1"),
            policy_overlay: Some(repo_path("policies/packs/overlays/test")),
            strict_signer: true,
        },
    )
    .expect("import report");
    assert!(!report.pass);
    assert!(report
        .reasons
        .iter()
        .any(|r| r.contains("trusted allowlist")));
}
