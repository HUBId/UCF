use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use ucf_compute::ModelSlot;

use crate::{sha256_hex, GateStatus, OpsError, ProbeReport, ReadinessGateReport};

const MANIFEST_HISTORY_KEEP: usize = 20;
const MODEL_FILE_NAME: &str = "model.safetensors";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelFileEntry {
    pub path: String,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LifecycleSlotManifest {
    pub active_hash: Option<String>,
    pub files: Vec<ModelFileEntry>,
    pub max_bytes: u64,
    pub contract_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LifecycleManifest {
    pub manifest_version: u16,
    pub slots: BTreeMap<String, LifecycleSlotManifest>,
    pub manifest_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StageResult {
    pub slot: String,
    pub hash: String,
    pub files: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LifecycleActionReport {
    pub action: String,
    pub slot: String,
    pub from_hash: Option<String>,
    pub to_hash: String,
    pub manifest_digest_prefix: String,
    pub probe_report_digest_prefix: Option<String>,
    pub readiness_gate_digest_prefix: Option<String>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelsListReport {
    pub slot: String,
    pub active_hash: Option<String>,
    pub staged_hashes: Vec<String>,
    pub promoted_hashes: Vec<String>,
}

pub fn models_stage(slot: ModelSlot, src_dir: &Path) -> Result<StageResult, OpsError> {
    if !src_dir.exists() {
        return Err(OpsError::Invalid(format!(
            "stage source missing: {}",
            src_dir.display()
        )));
    }
    let model_path = src_dir.join(MODEL_FILE_NAME);
    if !model_path.exists() {
        return Err(OpsError::Invalid(format!(
            "required {} missing",
            MODEL_FILE_NAME
        )));
    }
    let hash = file_sha256_hex(&model_path)?;
    let dst = PathBuf::from("models")
        .join("staging")
        .join(slot.as_str())
        .join(&hash);
    copy_tree(src_dir, &dst)?;
    Ok(StageResult {
        slot: slot.as_str().to_string(),
        hash,
        files: collect_files(&dst)?.len(),
    })
}

pub fn models_promote(
    slot: ModelSlot,
    hash: &str,
    probe_report_path: &Path,
    gate_report_path: &Path,
) -> Result<LifecycleActionReport, OpsError> {
    let staged = PathBuf::from("models")
        .join("staging")
        .join(slot.as_str())
        .join(hash);
    if !staged.exists() {
        return Err(OpsError::Invalid(format!(
            "staged artifact not found for {hash}"
        )));
    }
    let probe: ProbeReport = serde_json::from_str(&fs::read_to_string(probe_report_path)?)?;
    if !probe.summary.pass {
        return Err(OpsError::Invalid("probe report is not PASS".to_string()));
    }
    let gate: ReadinessGateReport = serde_json::from_str(&fs::read_to_string(gate_report_path)?)?;
    if gate.status != GateStatus::Pass {
        return Err(OpsError::Invalid("readiness gate is not PASS".to_string()));
    }

    let promoted = PathBuf::from("models")
        .join("promoted")
        .join(slot.as_str())
        .join(hash);
    copy_tree(&staged, &promoted)?;

    let mut manifest = load_or_init_manifest()?;
    let slot_key = slot.as_str().to_string();
    let from_hash = manifest
        .slots
        .get(&slot_key)
        .and_then(|s| s.active_hash.clone());
    let files = collect_file_entries(&promoted)?;
    manifest.slots.insert(
        slot_key.clone(),
        LifecycleSlotManifest {
            active_hash: Some(hash.to_string()),
            files,
            max_bytes: 64 * 1024 * 1024,
            contract_version: "v1".to_string(),
        },
    );
    persist_manifest_with_history(&mut manifest)?;
    persist_action_record(
        "promotion",
        &slot_key,
        &from_hash,
        hash,
        &manifest,
        &probe,
        &gate,
    )?;
    Ok(LifecycleActionReport {
        action: "promotion".to_string(),
        slot: slot_key,
        from_hash,
        to_hash: hash.to_string(),
        manifest_digest_prefix: manifest.manifest_digest.chars().take(12).collect(),
        probe_report_digest_prefix: Some(file_digest_prefix(probe_report_path)?),
        readiness_gate_digest_prefix: Some(file_digest_prefix(gate_report_path)?),
        timestamp: now_secs(),
    })
}

pub fn models_rollback(slot: ModelSlot, to_hash: &str) -> Result<LifecycleActionReport, OpsError> {
    let promoted = PathBuf::from("models")
        .join("promoted")
        .join(slot.as_str())
        .join(to_hash);
    if !promoted.exists() {
        return Err(OpsError::Invalid(
            "rollback hash is not present in promoted".to_string(),
        ));
    }
    let mut manifest = load_or_init_manifest()?;
    let slot_key = slot.as_str().to_string();
    let from_hash = manifest
        .slots
        .get(&slot_key)
        .and_then(|s| s.active_hash.clone());
    let files = collect_file_entries(&promoted)?;
    manifest.slots.insert(
        slot_key.clone(),
        LifecycleSlotManifest {
            active_hash: Some(to_hash.to_string()),
            files,
            max_bytes: 64 * 1024 * 1024,
            contract_version: "v1".to_string(),
        },
    );
    persist_manifest_with_history(&mut manifest)?;
    persist_rollback_record(&slot_key, &from_hash, to_hash, &manifest)?;
    Ok(LifecycleActionReport {
        action: "rollback".to_string(),
        slot: slot_key,
        from_hash,
        to_hash: to_hash.to_string(),
        manifest_digest_prefix: manifest.manifest_digest.chars().take(12).collect(),
        probe_report_digest_prefix: None,
        readiness_gate_digest_prefix: None,
        timestamp: now_secs(),
    })
}

pub fn models_list(slot: ModelSlot) -> Result<ModelsListReport, OpsError> {
    let manifest = load_or_init_manifest()?;
    let slot_key = slot.as_str().to_string();
    Ok(ModelsListReport {
        slot: slot_key.clone(),
        active_hash: manifest
            .slots
            .get(&slot_key)
            .and_then(|s| s.active_hash.clone()),
        staged_hashes: list_hash_dirs(
            &PathBuf::from("models").join("staging").join(slot.as_str()),
        )?,
        promoted_hashes: list_hash_dirs(
            &PathBuf::from("models").join("promoted").join(slot.as_str()),
        )?,
    })
}

pub fn parse_slot(value: &str) -> Result<ModelSlot, OpsError> {
    match value {
        "llm" => Ok(ModelSlot::Llm),
        "world_jepa" => Ok(ModelSlot::WorldJepa),
        "world_vljepa" => Ok(ModelSlot::WorldVljepa),
        "sae" => Ok(ModelSlot::Sae),
        "lfm" => Ok(ModelSlot::Lfm),
        "ssm" => Ok(ModelSlot::Ssm),
        "ebm_reasoner" | "ebm" => Ok(ModelSlot::EbmReasoner),
        _ => Err(OpsError::Invalid(format!("unknown slot: {value}"))),
    }
}

fn persist_manifest_with_history(manifest: &mut LifecycleManifest) -> Result<(), OpsError> {
    manifest.manifest_digest = compute_manifest_digest(manifest)?;
    fs::create_dir_all("models")?;
    let body = toml::to_string_pretty(manifest)
        .map_err(|e| OpsError::Invalid(format!("manifest serialize failed: {e}")))?;
    fs::write("models/MANIFEST.toml", body.as_bytes())?;
    let hist_dir = PathBuf::from("models/manifests/history");
    fs::create_dir_all(&hist_dir)?;
    let name = format!(
        "{}_{}.toml",
        now_secs(),
        manifest
            .manifest_digest
            .chars()
            .take(12)
            .collect::<String>()
    );
    fs::write(hist_dir.join(name), body.as_bytes())?;
    trim_history(&hist_dir, MANIFEST_HISTORY_KEEP)?;
    Ok(())
}

fn load_or_init_manifest() -> Result<LifecycleManifest, OpsError> {
    let path = PathBuf::from("models/MANIFEST.toml");
    if !path.exists() {
        let mut out = LifecycleManifest {
            manifest_version: 1,
            slots: BTreeMap::new(),
            manifest_digest: String::new(),
        };
        out.manifest_digest = compute_manifest_digest(&out)?;
        return Ok(out);
    }
    let raw = fs::read_to_string(path)?;
    toml::from_str(&raw).map_err(|e| OpsError::Invalid(format!("manifest parse failed: {e}")))
}

fn compute_manifest_digest(manifest: &LifecycleManifest) -> Result<String, OpsError> {
    let canonical = serde_json::to_vec(manifest)?;
    Ok(sha256_hex(&canonical))
}

fn collect_files(root: &Path) -> Result<Vec<PathBuf>, OpsError> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let p = entry.path();
            if entry.file_type()?.is_dir() {
                stack.push(p);
            } else {
                out.push(p);
            }
        }
    }
    out.sort();
    Ok(out)
}

fn collect_file_entries(root: &Path) -> Result<Vec<ModelFileEntry>, OpsError> {
    let files = collect_files(root)?;
    let mut out = Vec::new();
    for file in files {
        let rel = file
            .strip_prefix(root)
            .map_err(|e| OpsError::Invalid(e.to_string()))?;
        out.push(ModelFileEntry {
            path: rel.to_string_lossy().to_string(),
            sha256: file_sha256_hex(&file)?,
            size_bytes: fs::metadata(&file)?.len(),
        });
    }
    Ok(out)
}

fn copy_tree(src: &Path, dst: &Path) -> Result<(), OpsError> {
    fs::create_dir_all(dst)?;
    let files = collect_files(src)?;
    for file in files {
        let rel = file
            .strip_prefix(src)
            .map_err(|e| OpsError::Invalid(e.to_string()))?;
        let out = dst.join(rel);
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(file, out)?;
    }
    Ok(())
}

fn file_sha256_hex(path: &Path) -> Result<String, OpsError> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0_u8; 16 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn list_hash_dirs(base: &Path) -> Result<Vec<String>, OpsError> {
    if !base.exists() {
        return Ok(Vec::new());
    }
    let mut out = fs::read_dir(base)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect::<Vec<_>>();
    out.sort();
    Ok(out)
}

fn file_digest_prefix(path: &Path) -> Result<String, OpsError> {
    Ok(file_sha256_hex(path)?.chars().take(12).collect())
}

fn trim_history(dir: &Path, keep: usize) -> Result<(), OpsError> {
    let mut entries = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
        .collect::<Vec<_>>();
    entries.sort_by_key(|e| e.file_name());
    if entries.len() > keep {
        for entry in &entries[..entries.len() - keep] {
            fs::remove_file(entry.path())?;
        }
    }
    Ok(())
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn persist_action_record(
    action: &str,
    slot: &str,
    from_hash: &Option<String>,
    to_hash: &str,
    manifest: &LifecycleManifest,
    probe: &ProbeReport,
    gate: &ReadinessGateReport,
) -> Result<(), OpsError> {
    let report = LifecycleActionReport {
        action: action.to_string(),
        slot: slot.to_string(),
        from_hash: from_hash.clone(),
        to_hash: to_hash.to_string(),
        manifest_digest_prefix: manifest.manifest_digest.chars().take(12).collect(),
        probe_report_digest_prefix: Some(
            sha256_hex(serde_json::to_string(probe)?.as_bytes())
                .chars()
                .take(12)
                .collect(),
        ),
        readiness_gate_digest_prefix: Some(
            sha256_hex(serde_json::to_string(gate)?.as_bytes())
                .chars()
                .take(12)
                .collect(),
        ),
        timestamp: now_secs(),
    };
    let path = PathBuf::from("out").join("model_promotion_records.json");
    append_action(path, &report)
}

fn persist_rollback_record(
    slot: &str,
    from_hash: &Option<String>,
    to_hash: &str,
    manifest: &LifecycleManifest,
) -> Result<(), OpsError> {
    let report = LifecycleActionReport {
        action: "rollback".to_string(),
        slot: slot.to_string(),
        from_hash: from_hash.clone(),
        to_hash: to_hash.to_string(),
        manifest_digest_prefix: manifest.manifest_digest.chars().take(12).collect(),
        probe_report_digest_prefix: None,
        readiness_gate_digest_prefix: None,
        timestamp: now_secs(),
    };
    let path = PathBuf::from("out").join("model_rollback_records.json");
    append_action(path, &report)
}

fn append_action(path: PathBuf, report: &LifecycleActionReport) -> Result<(), OpsError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut all: Vec<LifecycleActionReport> = if path.exists() {
        serde_json::from_str(&fs::read_to_string(&path)?)?
    } else {
        Vec::new()
    };
    all.push(report.clone());
    fs::write(path, serde_json::to_string_pretty(&all)?)?;
    Ok(())
}
