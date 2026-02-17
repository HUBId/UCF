use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::ComputeError;

const DEFAULT_ALLOWLIST_ROOT: &str = "models";
const DEFAULT_MAX_BYTES: u64 = 64 * 1024 * 1024;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum ModelSlot {
    Llm,
    WorldJepa,
    Sae,
    Lfm,
    Ssm,
}

impl ModelSlot {
    pub const fn all() -> [Self; 5] {
        [Self::Llm, Self::WorldJepa, Self::Sae, Self::Lfm, Self::Ssm]
    }

    pub const fn env_key(self) -> &'static str {
        match self {
            Self::Llm => "LLM",
            Self::WorldJepa => "WORLD_JEPA",
            Self::Sae => "SAE",
            Self::Lfm => "LFM",
            Self::Ssm => "SSM",
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Llm => "llm",
            Self::WorldJepa => "world_jepa",
            Self::Sae => "sae",
            Self::Lfm => "lfm",
            Self::Ssm => "ssm",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFormat {
    CandleSafetensors,
    CandleBin,
    Burn,
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelDevice {
    CpuOnly,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSlotSpec {
    pub slot: ModelSlot,
    pub enabled: bool,
    pub path: Option<PathBuf>,
    pub expected_sha256: [u8; 32],
    pub max_bytes: u64,
    pub format: ModelFormat,
    pub device: ModelDevice,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelLoadError {
    Disabled,
    MissingPath,
    ManifestParse(String),
    PathOutsideAllowlist {
        path: PathBuf,
        allowlist_root: PathBuf,
    },
    PathTraversal {
        path: PathBuf,
    },
    OpenFailed {
        path: PathBuf,
        reason: String,
    },
    Oversized {
        path: PathBuf,
        max_bytes: u64,
        size_bytes: u64,
    },
    HashMismatch {
        expected: [u8; 32],
        found: [u8; 32],
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedModelSlot {
    pub slot: ModelSlot,
    pub path: PathBuf,
    pub sha256: [u8; 32],
    pub size_bytes: u64,
    pub format: ModelFormat,
    pub device: ModelDevice,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelProvenance {
    pub slot: ModelSlot,
    pub enabled: bool,
    pub resolved_path: Option<String>,
    pub sha256: Option<[u8; 32]>,
    pub size_bytes: Option<u64>,
    pub format: ModelFormat,
    pub backend_pack_digest: [u8; 32],
    pub run_id: String,
    pub schema_version: u16,
    pub disable_reason: Option<String>,
    pub found_hash_prefix: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ModelStore {
    pub allowlist_root: PathBuf,
    pub specs: BTreeMap<ModelSlot, ModelSlotSpec>,
}

impl ModelStore {
    pub fn from_manifest_and_env(manifest_path: &Path) -> Result<Self, ModelLoadError> {
        let mut doc = if manifest_path.exists() {
            let text = std::fs::read_to_string(manifest_path)
                .map_err(|e| ModelLoadError::ManifestParse(e.to_string()))?;
            toml::from_str::<ModelManifest>(&text)
                .map_err(|e| ModelLoadError::ManifestParse(e.to_string()))?
        } else {
            ModelManifest::default()
        };
        doc.apply_env_overrides();
        let allowlist_root = doc
            .allowlist_root
            .clone()
            .unwrap_or_else(|| PathBuf::from(DEFAULT_ALLOWLIST_ROOT));
        let specs = doc.to_specs();
        Ok(Self {
            allowlist_root,
            specs,
        })
    }

    pub fn from_env_default() -> Result<Self, ModelLoadError> {
        let path = std::env::var("UCF_MODEL_MANIFEST")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("models/manifest.toml"));
        Self::from_manifest_and_env(&path)
    }

    pub fn verify_slot(&self, slot: ModelSlot) -> Result<VerifiedModelSlot, ModelLoadError> {
        let Some(spec) = self.specs.get(&slot) else {
            return Err(ModelLoadError::Disabled);
        };
        if !spec.enabled {
            return Err(ModelLoadError::Disabled);
        }
        let rel_path = spec.path.clone().ok_or(ModelLoadError::MissingPath)?;
        let joined = self.allowlist_root.join(&rel_path);
        let allowlist_root =
            self.allowlist_root
                .canonicalize()
                .map_err(|_| ModelLoadError::PathTraversal {
                    path: self.allowlist_root.clone(),
                })?;
        let canonical = joined
            .canonicalize()
            .map_err(|_| ModelLoadError::PathTraversal {
                path: rel_path.clone(),
            })?;
        if !canonical.starts_with(&allowlist_root) {
            return Err(ModelLoadError::PathOutsideAllowlist {
                path: canonical,
                allowlist_root,
            });
        }

        let mut file = File::open(&canonical).map_err(|e| ModelLoadError::OpenFailed {
            path: canonical.clone(),
            reason: e.to_string(),
        })?;
        let size = file
            .metadata()
            .map_err(|e| ModelLoadError::OpenFailed {
                path: canonical.clone(),
                reason: e.to_string(),
            })?
            .len();
        if size > spec.max_bytes {
            return Err(ModelLoadError::Oversized {
                path: canonical,
                max_bytes: spec.max_bytes,
                size_bytes: size,
            });
        }
        let mut hasher = Sha256::new();
        let mut buf = [0_u8; 16 * 1024];
        loop {
            let read = file
                .read(&mut buf)
                .map_err(|e| ModelLoadError::OpenFailed {
                    path: canonical.clone(),
                    reason: e.to_string(),
                })?;
            if read == 0 {
                break;
            }
            hasher.update(&buf[..read]);
        }
        let found: [u8; 32] = hasher.finalize().into();
        if found != spec.expected_sha256 {
            return Err(ModelLoadError::HashMismatch {
                expected: spec.expected_sha256,
                found,
            });
        }

        Ok(VerifiedModelSlot {
            slot,
            path: canonical,
            sha256: found,
            size_bytes: size,
            format: spec.format,
            device: spec.device,
        })
    }

    pub fn read_verified_bytes(
        &self,
        verified: &VerifiedModelSlot,
    ) -> Result<Vec<u8>, ModelLoadError> {
        let mut file = File::open(&verified.path).map_err(|e| ModelLoadError::OpenFailed {
            path: verified.path.clone(),
            reason: e.to_string(),
        })?;
        let size = file
            .metadata()
            .map_err(|e| ModelLoadError::OpenFailed {
                path: verified.path.clone(),
                reason: e.to_string(),
            })?
            .len();
        let slot_spec = self
            .specs
            .get(&verified.slot)
            .ok_or(ModelLoadError::Disabled)?;
        if size > slot_spec.max_bytes {
            return Err(ModelLoadError::Oversized {
                path: verified.path.clone(),
                max_bytes: slot_spec.max_bytes,
                size_bytes: size,
            });
        }
        let mut bytes = Vec::with_capacity(size as usize);
        file.read_to_end(&mut bytes)
            .map_err(|e| ModelLoadError::OpenFailed {
                path: verified.path.clone(),
                reason: e.to_string(),
            })?;
        Ok(bytes)
    }

    pub fn verified_slots(&self) -> BTreeMap<ModelSlot, Result<VerifiedModelSlot, ModelLoadError>> {
        ModelSlot::all()
            .into_iter()
            .map(|slot| (slot, self.verify_slot(slot)))
            .collect()
    }

    pub fn model_hashes_digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        let verified = self.verified_slots();
        for slot in ModelSlot::all() {
            hasher.update(slot.as_str().as_bytes());
            if let Some(Ok(v)) = verified.get(&slot) {
                hasher.update([1]);
                hasher.update(v.sha256);
            } else {
                hasher.update([0]);
            }
        }
        hasher.finalize().into()
    }
}

#[derive(Debug, Deserialize, Default)]
struct ModelManifest {
    allowlist_root: Option<PathBuf>,
    slots: Option<ModelManifestSlots>,
}

#[derive(Debug, Deserialize, Default)]
struct ModelManifestSlots {
    llm: Option<ModelManifestSlotEntry>,
    world_jepa: Option<ModelManifestSlotEntry>,
    sae: Option<ModelManifestSlotEntry>,
    lfm: Option<ModelManifestSlotEntry>,
    ssm: Option<ModelManifestSlotEntry>,
}

#[derive(Debug, Deserialize, Clone)]
struct ModelManifestSlotEntry {
    enabled: Option<bool>,
    path: Option<PathBuf>,
    expected_sha256: Option<String>,
    max_bytes: Option<u64>,
    format: Option<ModelFormat>,
    device: Option<ModelDevice>,
}

impl ModelManifest {
    fn apply_env_overrides(&mut self) {
        for slot in ModelSlot::all() {
            let entry = self.entry_mut(slot);
            if let Ok(path) = std::env::var(format!("UCF_MODEL_{}_PATH", slot.env_key())) {
                entry.path = Some(PathBuf::from(path));
            }
            if let Ok(hash) = std::env::var(format!("UCF_MODEL_{}_SHA256", slot.env_key())) {
                entry.expected_sha256 = Some(hash);
            }
            if let Ok(max) = std::env::var(format!("UCF_MODEL_{}_MAX_BYTES", slot.env_key())) {
                entry.max_bytes = max.parse::<u64>().ok();
            }
            if let Ok(enabled) = std::env::var(format!("UCF_MODEL_{}_ENABLED", slot.env_key())) {
                entry.enabled = Some(matches!(enabled.as_str(), "1" | "true" | "TRUE"));
            }
        }
    }

    fn entry_mut(&mut self, slot: ModelSlot) -> &mut ModelManifestSlotEntry {
        let slots = self.slots.get_or_insert_with(ModelManifestSlots::default);
        match slot {
            ModelSlot::Llm => slots.llm.get_or_insert_with(default_entry),
            ModelSlot::WorldJepa => slots.world_jepa.get_or_insert_with(default_entry),
            ModelSlot::Sae => slots.sae.get_or_insert_with(default_entry),
            ModelSlot::Lfm => slots.lfm.get_or_insert_with(default_entry),
            ModelSlot::Ssm => slots.ssm.get_or_insert_with(default_entry),
        }
    }

    fn to_specs(&self) -> BTreeMap<ModelSlot, ModelSlotSpec> {
        let mut out = BTreeMap::new();
        for slot in ModelSlot::all() {
            let entry = self.entry(slot).cloned().unwrap_or_else(default_entry);
            let expected_sha256 = parse_hash(entry.expected_sha256.as_deref().unwrap_or(""));
            out.insert(
                slot,
                ModelSlotSpec {
                    slot,
                    enabled: entry.enabled.unwrap_or(false),
                    path: entry.path,
                    expected_sha256,
                    max_bytes: entry.max_bytes.unwrap_or(DEFAULT_MAX_BYTES),
                    format: entry.format.unwrap_or(ModelFormat::Custom),
                    device: entry.device.unwrap_or(ModelDevice::CpuOnly),
                },
            );
        }
        out
    }

    fn entry(&self, slot: ModelSlot) -> Option<&ModelManifestSlotEntry> {
        let slots = self.slots.as_ref()?;
        match slot {
            ModelSlot::Llm => slots.llm.as_ref(),
            ModelSlot::WorldJepa => slots.world_jepa.as_ref(),
            ModelSlot::Sae => slots.sae.as_ref(),
            ModelSlot::Lfm => slots.lfm.as_ref(),
            ModelSlot::Ssm => slots.ssm.as_ref(),
        }
    }
}

fn default_entry() -> ModelManifestSlotEntry {
    ModelManifestSlotEntry {
        enabled: Some(false),
        path: None,
        expected_sha256: None,
        max_bytes: Some(DEFAULT_MAX_BYTES),
        format: Some(ModelFormat::Custom),
        device: Some(ModelDevice::CpuOnly),
    }
}

fn parse_hash(value: &str) -> [u8; 32] {
    let trimmed = value.trim();
    let mut out = [0_u8; 32];
    if let Ok(decoded) = hex::decode(trimmed) {
        if decoded.len() == 32 {
            out.copy_from_slice(&decoded);
        }
    }
    out
}

impl From<ModelLoadError> for ComputeError {
    fn from(value: ModelLoadError) -> Self {
        ComputeError::InvalidInput {
            reason: format!("model load error: {value:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn hash_parse_bad_is_zero() {
        assert_eq!(parse_hash("abc"), [0; 32]);
    }

    #[test]
    fn rejects_path_outside_allowlist() {
        let temp = tempfile::tempdir().expect("tempdir");
        let models = temp.path().join("models");
        let outside = temp.path().join("outside.bin");
        fs::create_dir_all(&models).expect("models");
        fs::write(&outside, b"abc").expect("write");

        let mut specs = BTreeMap::new();
        specs.insert(
            ModelSlot::Llm,
            ModelSlotSpec {
                slot: ModelSlot::Llm,
                enabled: true,
                path: Some(PathBuf::from("../outside.bin")),
                expected_sha256: Sha256::digest(b"abc").into(),
                max_bytes: 1024,
                format: ModelFormat::Custom,
                device: ModelDevice::CpuOnly,
            },
        );
        let store = ModelStore {
            allowlist_root: models,
            specs,
        };
        let err = store.verify_slot(ModelSlot::Llm).expect_err("must fail");
        assert!(matches!(err, ModelLoadError::PathOutsideAllowlist { .. }));
    }

    #[test]
    fn detects_hash_mismatch() {
        let temp = tempfile::tempdir().expect("tempdir");
        let models = temp.path().join("models");
        fs::create_dir_all(&models).expect("models");
        fs::write(models.join("w.bin"), b"abc").expect("write");

        let mut specs = BTreeMap::new();
        specs.insert(
            ModelSlot::Llm,
            ModelSlotSpec {
                slot: ModelSlot::Llm,
                enabled: true,
                path: Some(PathBuf::from("w.bin")),
                expected_sha256: [9; 32],
                max_bytes: 1024,
                format: ModelFormat::Custom,
                device: ModelDevice::CpuOnly,
            },
        );
        let store = ModelStore {
            allowlist_root: models,
            specs,
        };
        let err = store.verify_slot(ModelSlot::Llm).expect_err("must fail");
        assert!(matches!(err, ModelLoadError::HashMismatch { .. }));
    }

    #[test]
    fn enforces_size_limit() {
        let temp = tempfile::tempdir().expect("tempdir");
        let models = temp.path().join("models");
        fs::create_dir_all(&models).expect("models");
        fs::write(models.join("w.bin"), b"abcdef").expect("write");

        let mut specs = BTreeMap::new();
        specs.insert(
            ModelSlot::Llm,
            ModelSlotSpec {
                slot: ModelSlot::Llm,
                enabled: true,
                path: Some(PathBuf::from("w.bin")),
                expected_sha256: Sha256::digest(b"abcdef").into(),
                max_bytes: 3,
                format: ModelFormat::Custom,
                device: ModelDevice::CpuOnly,
            },
        );
        let store = ModelStore {
            allowlist_root: models,
            specs,
        };
        let err = store.verify_slot(ModelSlot::Llm).expect_err("must fail");
        assert!(matches!(err, ModelLoadError::Oversized { .. }));
    }
}
