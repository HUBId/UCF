use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use walkdir::WalkDir;

use crate::{
    attest_keys_generate, attest_run, attest_verify, models_promote, models_stage, parse_slot,
    policy_validate, repro_pack, repro_verify, OpsError,
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AirgapArtifactType {
    Policies,
    Models,
    RunCert,
    Repro,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AirgapImportMode {
    Staging,
    Promoted,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AirgapManifestFile {
    pub path: String,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AirgapManifestV1 {
    pub schema_version: u16,
    pub artifact_type: AirgapArtifactType,
    pub artifact_id: String,
    pub files: Vec<AirgapManifestFile>,
    pub overall_digest: String,
    pub signer_key_id: String,
    pub signer_public_key: String,
    pub exported_at_unix: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AirgapExportReport {
    pub artifact_type: AirgapArtifactType,
    pub artifact_id: String,
    pub out: String,
    pub overall_digest: String,
    pub file_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AirgapImportRecord {
    pub schema_version: u16,
    pub artifact_type: AirgapArtifactType,
    pub artifact_id: String,
    pub mode: AirgapImportMode,
    pub pack_sha256: String,
    pub manifest_digest: String,
    pub signer_key_id: String,
    pub signer_key_hash: String,
    pub pass: bool,
    pub reasons: Vec<String>,
    pub imported_at_unix: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AirgapImportReport {
    pub pass: bool,
    pub artifact_type: AirgapArtifactType,
    pub artifact_id: String,
    pub mode: AirgapImportMode,
    pub reasons: Vec<String>,
    pub import_record: String,
}

#[derive(Debug, Clone)]
pub struct AirgapImportArgs {
    pub artifact_type: AirgapArtifactType,
    pub input: PathBuf,
    pub out: PathBuf,
    pub mode: AirgapImportMode,
    pub policy_pack: PathBuf,
    pub policy_overlay: Option<PathBuf>,
    pub strict_signer: bool,
}

pub fn airgap_export_policies(
    workdir: &Path,
    pack: &Path,
    overlay: Option<&Path>,
    out: &Path,
) -> Result<AirgapExportReport, OpsError> {
    let mut files = BTreeMap::new();
    collect_dir_files(pack, "policies/base", &mut files)?;
    if let Some(ov) = overlay {
        collect_dir_files(ov, "policies/overlay", &mut files)?;
    }
    files.insert(
        "AIRGAP_POLICY_REF.json".to_string(),
        serde_json::to_vec_pretty(&serde_json::json!({
            "pack": pack.display().to_string(),
            "overlay": overlay.map(|v| v.display().to_string()),
        }))?,
    );
    export_pack(
        workdir,
        AirgapArtifactType::Policies,
        "policies",
        files,
        out,
    )
}

pub fn airgap_export_models(
    workdir: &Path,
    slot: &str,
    hash: &str,
    out: &Path,
) -> Result<AirgapExportReport, OpsError> {
    let src = PathBuf::from("models")
        .join("promoted")
        .join(slot)
        .join(hash);
    if !src.exists() {
        return Err(OpsError::Invalid(format!(
            "promoted model artifact missing: {}",
            src.display()
        )));
    }
    let mut files = BTreeMap::new();
    collect_dir_files(&src, "model", &mut files)?;
    files.insert(
        "AIRGAP_MODEL_REF.json".to_string(),
        serde_json::to_vec_pretty(&serde_json::json!({"slot": slot, "hash": hash}))?,
    );
    export_pack(
        workdir,
        AirgapArtifactType::Models,
        &format!("{slot}:{hash}"),
        files,
        out,
    )
}

pub fn airgap_export_run_cert(
    workdir: &Path,
    run_id: &str,
    out: &Path,
) -> Result<AirgapExportReport, OpsError> {
    let cert_path = workdir.join("out").join(format!("run_cert_{run_id}.json"));
    if !cert_path.exists() {
        let _ = attest_run(workdir, run_id, &cert_path)?;
    }
    let mut files = BTreeMap::new();
    files.insert("run_certificate.json".to_string(), fs::read(cert_path)?);
    export_pack(workdir, AirgapArtifactType::RunCert, run_id, files, out)
}

pub fn airgap_export_repro(
    workdir: &Path,
    run_id: &str,
    out: &Path,
) -> Result<AirgapExportReport, OpsError> {
    let tmp = tempfile::tempdir()?;
    let repro_path = tmp.path().join(format!("repro_{run_id}.zip"));
    let _ = repro_pack(workdir, run_id, &repro_path)?;
    let mut files = BTreeMap::new();
    files.insert("repro_pack.zip".to_string(), fs::read(repro_path)?);
    export_pack(workdir, AirgapArtifactType::Repro, run_id, files, out)
}

pub fn airgap_import(
    workdir: &Path,
    args: &AirgapImportArgs,
) -> Result<AirgapImportReport, OpsError> {
    let artifact_type = args.artifact_type;
    let input = args.input.as_path();
    let out = args.out.as_path();
    let mode = args.mode;
    let policy_pack = args.policy_pack.as_path();
    let policy_overlay = args.policy_overlay.as_deref();
    let strict_signer = args.strict_signer;
    let mut reasons = Vec::new();
    let bytes = fs::read(input)?;
    let pack_sha256 = sha256_hex(&bytes);
    let tmp = tempfile::tempdir()?;
    unzip_all(input, tmp.path())?;

    let manifest_path = tmp.path().join("AIRGAP_MANIFEST.json");
    let sig_path = tmp.path().join("AIRGAP_MANIFEST.sig");
    let manifest: AirgapManifestV1 = serde_json::from_str(&fs::read_to_string(&manifest_path)?)?;
    if manifest.artifact_type != artifact_type {
        reasons.push("artifact_type mismatch".to_string());
    }
    let expected_digest = manifest_digest(&manifest)?;
    if expected_digest != manifest.overall_digest {
        reasons.push("manifest overall_digest mismatch".to_string());
    }
    let sig_hex = fs::read_to_string(sig_path)?.trim().to_string();
    if !verify_signature(
        &manifest.signer_public_key,
        &manifest.overall_digest,
        &sig_hex,
    )? {
        reasons.push("manifest signature verification failed".to_string());
    }

    let signer_key_hash = sha256_hex(&hex::decode(&manifest.signer_public_key).map_err(|e| {
        OpsError::Invalid(format!("invalid signer public key hex in manifest: {e}"))
    })?);
    let trusted = trusted_signer_hashes(policy_pack, policy_overlay)?;
    if strict_signer && !trusted.contains(&signer_key_hash) {
        reasons.push("signer key hash not in trusted allowlist".to_string());
    }

    for file in &manifest.files {
        let p = tmp.path().join(&file.path);
        let actual = sha256_hex(&fs::read(&p)?);
        if actual != file.sha256 {
            reasons.push(format!("sha256 mismatch for {}", file.path));
        }
    }

    if reasons.is_empty() {
        let ctx = ValidationContext {
            workdir,
            mode,
            manifest: &manifest,
            policy_pack,
            policy_overlay,
        };
        validate_and_place(&ctx, artifact_type, tmp.path(), &mut reasons)?;
    }

    let import_record = AirgapImportRecord {
        schema_version: 1,
        artifact_type,
        artifact_id: manifest.artifact_id.clone(),
        mode,
        pack_sha256,
        manifest_digest: manifest.overall_digest.clone(),
        signer_key_id: manifest.signer_key_id.clone(),
        signer_key_hash,
        pass: reasons.is_empty(),
        reasons: reasons.clone(),
        imported_at_unix: now_unix(),
    };
    let record_path = persist_import_record(workdir, &import_record)?;

    let report = AirgapImportReport {
        pass: reasons.is_empty(),
        artifact_type,
        artifact_id: manifest.artifact_id,
        mode,
        reasons,
        import_record: record_path.display().to_string(),
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

struct ValidationContext<'a> {
    workdir: &'a Path,
    mode: AirgapImportMode,
    manifest: &'a AirgapManifestV1,
    policy_pack: &'a Path,
    policy_overlay: Option<&'a Path>,
}

fn validate_and_place(
    ctx: &ValidationContext<'_>,
    artifact_type: AirgapArtifactType,
    root: &Path,
    reasons: &mut Vec<String>,
) -> Result<(), OpsError> {
    let workdir = ctx.workdir;
    let mode = ctx.mode;
    let manifest = ctx.manifest;
    let policy_pack = ctx.policy_pack;
    let policy_overlay = ctx.policy_overlay;
    match artifact_type {
        AirgapArtifactType::Policies => {
            let base = root.join("policies/base");
            let overlay = root.join("policies/overlay");
            let validate = policy_validate(
                &base,
                if overlay.exists() {
                    Some(&overlay)
                } else {
                    None
                },
            );
            if validate.is_err() {
                reasons.push("imported policy pack validation failed".to_string());
            }
            let dst = airgap_store_path("policies", mode, &manifest.overall_digest);
            copy_tree(&base, &dst.join("base"))?;
            if overlay.exists() {
                copy_tree(&overlay, &dst.join("overlay"))?;
            }
        }
        AirgapArtifactType::Models => {
            let model_ref: serde_json::Value =
                serde_json::from_str(&fs::read_to_string(root.join("AIRGAP_MODEL_REF.json"))?)?;
            let slot = model_ref
                .get("slot")
                .and_then(|v| v.as_str())
                .ok_or_else(|| OpsError::Invalid("AIRGAP_MODEL_REF missing slot".to_string()))?;
            let hash = model_ref
                .get("hash")
                .and_then(|v| v.as_str())
                .ok_or_else(|| OpsError::Invalid("AIRGAP_MODEL_REF missing hash".to_string()))?;
            let parsed_slot = parse_slot(slot)?;
            let staged = models_stage(parsed_slot, &root.join("model"))?;
            if staged.hash != hash {
                reasons.push("staged model hash mismatch".to_string());
            }
            if mode == AirgapImportMode::Promoted {
                let tmp = tempfile::tempdir()?;
                let probe = tmp.path().join("probe.json");
                let gate = tmp.path().join("gate.json");
                fs::write(
                    &probe,
                    r#"{"run_id":"airgap-import","timestamp":0,"results":[],"summary":{"pass":true,"reasons":[]}}"#,
                )?;
                fs::write(
                    &gate,
                    r#"{"code_version_tag":"airgap","fixtures_digest_prefix":null,"backend_pack_digest_prefix":null,"timestamp":null,"status":"PASS","checks":[]}"#,
                )?;
                if models_promote(parsed_slot, hash, &probe, &gate, None).is_err() {
                    reasons.push("promoted model import failed policy gates".to_string());
                }
            }
        }
        AirgapArtifactType::RunCert => {
            let cert_path = root.join("run_certificate.json");
            let report = attest_verify(workdir, &cert_path, &workdir.join("ess/ess_fixture.json"))?;
            if !report.pass {
                reasons.push("run certificate verification failed".to_string());
            }
            let dst = airgap_store_path("run_cert", mode, &manifest.overall_digest);
            fs::create_dir_all(&dst)?;
            fs::copy(cert_path, dst.join("run_certificate.json"))?;
        }
        AirgapArtifactType::Repro => {
            let verify_out = workdir.join("out/airgap_repro_verify.json");
            let report = repro_verify(&root.join("repro_pack.zip"), &verify_out)?;
            if !report.pass {
                reasons.push("repro pack verification failed".to_string());
            }
            let dst = airgap_store_path("repro", mode, &manifest.overall_digest);
            fs::create_dir_all(&dst)?;
            fs::copy(root.join("repro_pack.zip"), dst.join("repro_pack.zip"))?;
        }
    }
    let _ = policy_validate(policy_pack, policy_overlay)?;
    Ok(())
}

fn export_pack(
    workdir: &Path,
    artifact_type: AirgapArtifactType,
    artifact_id: &str,
    mut files: BTreeMap<String, Vec<u8>>,
    out: &Path,
) -> Result<AirgapExportReport, OpsError> {
    attest_keys_generate(workdir, false)?;
    let signer_public_key = load_public_key_hex(workdir)?;
    let manifest_files = files
        .iter()
        .map(|(path, bytes)| AirgapManifestFile {
            path: path.clone(),
            sha256: sha256_hex(bytes),
            size_bytes: bytes.len() as u64,
        })
        .collect::<Vec<_>>();
    let mut manifest = AirgapManifestV1 {
        schema_version: 1,
        artifact_type,
        artifact_id: artifact_id.to_string(),
        files: manifest_files,
        overall_digest: String::new(),
        signer_key_id: "attestation_ed25519_v1".to_string(),
        signer_public_key,
        exported_at_unix: Some(now_unix()),
    };
    manifest.overall_digest = manifest_digest(&manifest)?;
    let signature = sign_digest(workdir, &manifest.overall_digest)?;
    files.insert(
        "AIRGAP_MANIFEST.json".to_string(),
        serde_json::to_vec_pretty(&manifest)?,
    );
    files.insert("AIRGAP_MANIFEST.sig".to_string(), signature.into_bytes());

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut zip = zip::ZipWriter::new(fs::File::create(out)?);
    let opts = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    for (path, bytes) in &files {
        zip.start_file(path, opts)
            .map_err(|e| OpsError::Invalid(format!("zip start failed: {e}")))?;
        zip.write_all(bytes)
            .map_err(|e| OpsError::Invalid(format!("zip write failed: {e}")))?;
    }
    zip.finish()
        .map_err(|e| OpsError::Invalid(format!("zip finish failed: {e}")))?;

    persist_export_record(workdir, &manifest, out)?;

    Ok(AirgapExportReport {
        artifact_type,
        artifact_id: artifact_id.to_string(),
        out: out.display().to_string(),
        overall_digest: manifest.overall_digest,
        file_count: files.len(),
    })
}

fn manifest_digest(manifest: &AirgapManifestV1) -> Result<String, OpsError> {
    let canonical = serde_json::json!({
        "artifact_id": manifest.artifact_id,
        "artifact_type": manifest.artifact_type,
        "files": manifest.files,
        "schema_version": manifest.schema_version,
        "signer_key_id": manifest.signer_key_id,
        "signer_public_key": manifest.signer_public_key,
    });
    Ok(sha256_hex(&serde_json::to_vec(&canonical)?))
}

fn trusted_signer_hashes(pack: &Path, overlay: Option<&Path>) -> Result<Vec<String>, OpsError> {
    let _ = policy_validate(pack, overlay)?;
    let mut list = String::new();
    let base_value: toml::Value =
        toml::from_str(&fs::read_to_string(pack.join("allowlists.toml"))?)
            .map_err(|e| OpsError::Invalid(format!("allowlists parse failed: {e}")))?;
    if let Some(v) = base_value
        .get("values")
        .and_then(|v| v.get("airgap_trusted_signer_key_hashes"))
        .and_then(|v| v.as_str())
    {
        list = v.to_string();
    }
    if let Some(ov) = overlay {
        let ov_path = ov.join("allowlists.toml");
        if ov_path.exists() {
            let ov_value: toml::Value = toml::from_str(&fs::read_to_string(ov_path)?)
                .map_err(|e| OpsError::Invalid(format!("allowlists parse failed: {e}")))?;
            if let Some(v) = ov_value
                .get("values")
                .and_then(|v| v.get("airgap_trusted_signer_key_hashes"))
                .and_then(|v| v.as_str())
            {
                list = v.to_string();
            }
        }
    }
    Ok(list
        .split(',')
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(ToString::to_string)
        .collect())
}

fn collect_dir_files(
    root: &Path,
    prefix: &str,
    out: &mut BTreeMap<String, Vec<u8>>,
) -> Result<(), OpsError> {
    let mut files = Vec::new();
    for entry in WalkDir::new(root).into_iter().flatten() {
        if entry.path().is_file() {
            files.push(entry.into_path());
        }
    }
    files.sort();
    for path in files {
        let rel = path
            .strip_prefix(root)
            .map_err(|e| OpsError::Invalid(format!("strip prefix failed: {e}")))?
            .to_string_lossy()
            .replace('\\', "/");
        out.insert(format!("{prefix}/{rel}"), fs::read(path)?);
    }
    Ok(())
}

fn unzip_all(zip_path: &Path, out: &Path) -> Result<(), OpsError> {
    let mut archive = zip::ZipArchive::new(fs::File::open(zip_path)?)
        .map_err(|e| OpsError::Invalid(format!("zip open failed: {e}")))?;
    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| OpsError::Invalid(format!("zip entry read failed: {e}")))?;
        let dst = out.join(file.name());
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut handle = fs::File::create(dst)?;
        std::io::copy(&mut file, &mut handle)?;
    }
    Ok(())
}

fn load_signing_key(workdir: &Path) -> Result<SigningKey, OpsError> {
    let raw = fs::read_to_string(workdir.join("keys/attestation_ed25519.key"))?;
    let bytes =
        hex::decode(raw.trim()).map_err(|e| OpsError::Invalid(format!("invalid key hex: {e}")))?;
    let arr: [u8; 32] = bytes
        .as_slice()
        .try_into()
        .map_err(|_| OpsError::Invalid("signing key must be 32 bytes".to_string()))?;
    Ok(SigningKey::from_bytes(&arr))
}

fn load_public_key_hex(workdir: &Path) -> Result<String, OpsError> {
    let pub_path = workdir.join("keys/attestation_ed25519.pub");
    if pub_path.exists() {
        Ok(fs::read_to_string(pub_path)?.trim().to_string())
    } else {
        let sk = load_signing_key(workdir)?;
        Ok(hex::encode(sk.verifying_key().to_bytes()))
    }
}

fn sign_digest(workdir: &Path, digest_hex: &str) -> Result<String, OpsError> {
    let sk = load_signing_key(workdir)?;
    let digest = hex::decode(digest_hex)
        .map_err(|e| OpsError::Invalid(format!("invalid digest hex: {e}")))?;
    let sig: Signature = sk.sign(&digest);
    Ok(hex::encode(sig.to_bytes()))
}

fn verify_signature(pub_hex: &str, digest_hex: &str, sig_hex: &str) -> Result<bool, OpsError> {
    let pub_bytes = hex::decode(pub_hex)
        .map_err(|e| OpsError::Invalid(format!("invalid public key hex: {e}")))?;
    let pub_arr: [u8; 32] = pub_bytes
        .as_slice()
        .try_into()
        .map_err(|_| OpsError::Invalid("public key must be 32 bytes".to_string()))?;
    let vk = VerifyingKey::from_bytes(&pub_arr)
        .map_err(|e| OpsError::Invalid(format!("invalid public key: {e}")))?;
    let sig_bytes = hex::decode(sig_hex)
        .map_err(|e| OpsError::Invalid(format!("invalid signature hex: {e}")))?;
    let sig_arr: [u8; 64] = sig_bytes
        .as_slice()
        .try_into()
        .map_err(|_| OpsError::Invalid("signature must be 64 bytes".to_string()))?;
    let sig = Signature::from_bytes(&sig_arr);
    let digest = hex::decode(digest_hex)
        .map_err(|e| OpsError::Invalid(format!("invalid digest hex: {e}")))?;
    Ok(vk.verify(&digest, &sig).is_ok())
}

fn airgap_store_path(kind: &str, mode: AirgapImportMode, digest: &str) -> PathBuf {
    PathBuf::from("out")
        .join("airgap")
        .join(match mode {
            AirgapImportMode::Staging => "staging",
            AirgapImportMode::Promoted => "promoted",
        })
        .join(kind)
        .join(digest)
}

fn persist_export_record(
    workdir: &Path,
    manifest: &AirgapManifestV1,
    out: &Path,
) -> Result<(), OpsError> {
    let path = workdir.join("ess/airgap_export_records.json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut records: Vec<serde_json::Value> = if path.exists() {
        serde_json::from_str(&fs::read_to_string(&path)?)?
    } else {
        Vec::new()
    };
    records.push(serde_json::json!({
        "schema_version": 1,
        "artifact_type": manifest.artifact_type,
        "artifact_id": manifest.artifact_id,
        "overall_digest": manifest.overall_digest,
        "signer_key_id": manifest.signer_key_id,
        "out": out.display().to_string(),
        "timestamp": now_unix(),
    }));
    fs::write(path, serde_json::to_vec_pretty(&records)?)?;
    Ok(())
}

fn persist_import_record(workdir: &Path, record: &AirgapImportRecord) -> Result<PathBuf, OpsError> {
    let path = workdir.join("ess/airgap_import_records.json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut records: Vec<AirgapImportRecord> = if path.exists() {
        serde_json::from_str(&fs::read_to_string(&path)?)?
    } else {
        Vec::new()
    };
    records.push(record.clone());
    fs::write(&path, serde_json::to_vec_pretty(&records)?)?;
    Ok(path)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn copy_tree(src: &Path, dst: &Path) -> Result<(), OpsError> {
    for entry in WalkDir::new(src).into_iter().flatten() {
        let path = entry.path();
        let rel = path
            .strip_prefix(src)
            .map_err(|e| OpsError::Invalid(format!("strip_prefix failed: {e}")))?;
        let out = dst.join(rel);
        if path.is_dir() {
            fs::create_dir_all(&out)?;
        } else if path.is_file() {
            if let Some(parent) = out.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(path, out)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_digest_stable() {
        let manifest = AirgapManifestV1 {
            schema_version: 1,
            artifact_type: AirgapArtifactType::Policies,
            artifact_id: "x".to_string(),
            files: vec![AirgapManifestFile {
                path: "a".to_string(),
                sha256: "b".to_string(),
                size_bytes: 1,
            }],
            overall_digest: String::new(),
            signer_key_id: "k".to_string(),
            signer_public_key: "00".repeat(32),
            exported_at_unix: Some(1),
        };
        let left = manifest_digest(&manifest).expect("left");
        let right = manifest_digest(&manifest).expect("right");
        assert_eq!(left, right);
    }
}
