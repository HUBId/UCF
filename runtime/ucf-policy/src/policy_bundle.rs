use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyBundleProvenance {
    pub version: String,
    pub bundle_sha256: String,
    pub enabled_features: Vec<String>,
    pub run_id: String,
}

#[derive(Debug, thiserror::Error)]
pub enum PolicyBundleError {
    #[error("policy manifest missing at {0}")]
    MissingManifest(String),
    #[error("invalid policy manifest: {0}")]
    InvalidManifest(String),
    #[error("policy file missing: {0}")]
    MissingFile(String),
    #[error("policy hash mismatch for {path}: expected {expected}, actual {actual}")]
    FileHashMismatch {
        path: String,
        expected: String,
        actual: String,
    },
    #[error("policy bundle hash mismatch: expected {expected}, actual {actual}")]
    BundleHashMismatch { expected: String, actual: String },
    #[error("UCF_POLICY_BUNDLE_SHA256 must be set")]
    MissingExpectedBundleHash,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    version: String,
    files: Vec<ManifestFile>,
    bundle_sha256: String,
}

#[derive(Debug, Deserialize)]
struct ManifestFile {
    path: String,
    sha256: String,
}

pub fn verify_policy_bundle(root: &Path) -> Result<PolicyBundleProvenance, PolicyBundleError> {
    let expected_bundle_hash = std::env::var("UCF_POLICY_BUNDLE_SHA256")
        .map_err(|_| PolicyBundleError::MissingExpectedBundleHash)?;
    let run_id = std::env::var("UCF_RUN_ID").unwrap_or_else(|_| "run-local".to_string());
    let manifest_path = root.join("manifest.toml");
    let manifest_str = fs::read_to_string(&manifest_path).map_err(|_| {
        PolicyBundleError::MissingManifest(manifest_path.to_string_lossy().to_string())
    })?;
    let manifest: Manifest = toml::from_str(&manifest_str)
        .map_err(|e| PolicyBundleError::InvalidManifest(e.to_string()))?;

    for file in &manifest.files {
        let abs = root.join(&file.path);
        let bytes = fs::read(&abs)
            .map_err(|_| PolicyBundleError::MissingFile(abs.to_string_lossy().to_string()))?;
        let actual = hex_lower(sha256(&bytes));
        if actual != file.sha256 {
            return Err(PolicyBundleError::FileHashMismatch {
                path: file.path.clone(),
                expected: file.sha256.clone(),
                actual,
            });
        }
    }

    let computed_bundle = compute_bundle_hash(root, &manifest.files)?;
    if computed_bundle != manifest.bundle_sha256 {
        return Err(PolicyBundleError::BundleHashMismatch {
            expected: manifest.bundle_sha256,
            actual: computed_bundle,
        });
    }
    if expected_bundle_hash != computed_bundle {
        return Err(PolicyBundleError::BundleHashMismatch {
            expected: expected_bundle_hash,
            actual: computed_bundle,
        });
    }

    Ok(PolicyBundleProvenance {
        version: manifest.version,
        bundle_sha256: manifest.bundle_sha256,
        enabled_features: vec![
            "sandbox_fs_v1".to_string(),
            "deny_by_construction".to_string(),
        ],
        run_id,
    })
}

fn compute_bundle_hash(root: &Path, files: &[ManifestFile]) -> Result<String, PolicyBundleError> {
    let mut canonical: Vec<(String, [u8; 32])> = Vec::with_capacity(files.len());
    for file in files {
        let abs: PathBuf = root.join(&file.path);
        let bytes = fs::read(&abs)
            .map_err(|_| PolicyBundleError::MissingFile(abs.to_string_lossy().to_string()))?;
        canonical.push((file.path.clone(), sha256(&bytes)));
    }
    canonical.sort_by(|a, b| a.0.cmp(&b.0));

    let mut hasher = Sha256::new();
    hasher.update(b"ucf.policy.bundle.v1");
    for (path, digest) in canonical {
        hasher.update(path.as_bytes());
        hasher.update([0]);
        hasher.update(digest);
    }
    Ok(hex_lower(hasher.finalize().into()))
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().into()
}

fn hex_lower(bytes: [u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_missing_env_hash() {
        let _ = std::env::remove_var("UCF_POLICY_BUNDLE_SHA256");
        let err = verify_policy_bundle(Path::new("policies")).expect_err("should fail");
        assert!(matches!(err, PolicyBundleError::MissingExpectedBundleHash));
    }
}
