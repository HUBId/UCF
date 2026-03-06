use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use ucf_compute::ModelSlot;
use ucf_ess::v1::ExperiencePayload;
use ucf_replay::load_fixture_records;

use crate::{
    sha256_hex, GateStatus, OpsError, ProbeReport, ReadinessGateReport, WorldShadowReport,
};

const MANIFEST_HISTORY_KEEP: usize = 20;
const MODEL_FILE_NAME: &str = "model.safetensors";
const MANIFEST_PATH: &str = "models/MANIFEST.toml";
const LEGACY_MANIFEST_PATH: &str = "models/lifecycle_manifest.toml";
const MAX_MANIFEST_BYTES: usize = 1024 * 1024;
const MAX_SLOT_FILES: usize = 512;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelFileEntry {
    pub path: String,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LifecycleSlotManifest {
    pub slot_id: String,
    pub active_hash: Option<String>,
    pub files: Vec<ModelFileEntry>,
    pub max_bytes: u64,
    pub contract_versions_supported: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LifecycleManifest {
    pub manifest_version: u16,
    pub created_at: Option<u64>,
    pub slots: Vec<LifecycleSlotManifest>,
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
    pub shadow_report_digest_prefix: Option<String>,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelsVerifyReport {
    pub manifest: String,
    pub manifest_present: bool,
    pub digest_match: bool,
    pub promoted_hashes_exist: bool,
    pub files_verified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RollbackRecommendation {
    pub slot: String,
    pub tick: u64,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelsRollbackRecommendationReport {
    pub slot: String,
    pub recommendations: Vec<RollbackRecommendation>,
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
    let source_entries = collect_file_entries(src_dir)?;
    if source_entries.is_empty() {
        return Err(OpsError::Invalid("stage source has no files".to_string()));
    }
    if source_entries.len() > MAX_SLOT_FILES {
        return Err(OpsError::Invalid(format!(
            "stage source has too many files: {} > {}",
            source_entries.len(),
            MAX_SLOT_FILES
        )));
    }
    let hash = digest_entries(&source_entries);
    let dst = PathBuf::from("models")
        .join("staging")
        .join(slot.as_str())
        .join(&hash);
    copy_tree(src_dir, &dst)?;
    ensure_manifest_slot(slot)?;
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
    shadow_report_path: Option<&Path>,
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
    let mut shadow_report_digest_prefix = None;
    let require_shadow = std::env::var("UCF_WORLD_VLJEPA_REQUIRE_SHADOW_EVIDENCE")
        .map(|v| v != "0")
        .unwrap_or(true);
    if slot == ModelSlot::WorldVljepa && require_shadow {
        let Some(path) = shadow_report_path else {
            return Err(OpsError::Invalid(
                "world_vljepa promotion requires --shadow-report".to_string(),
            ));
        };
        let shadow: WorldShadowReport = serde_json::from_str(&fs::read_to_string(path)?)?;
        if shadow.status != GateStatus::Pass {
            return Err(OpsError::Invalid(
                "world_vljepa shadow report is not PASS".to_string(),
            ));
        }
        let min_ticks = std::env::var("UCF_WORLD_VLJEPA_PROMOTION_MIN_TICKS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(10_000);
        if shadow.ticks_total < min_ticks {
            return Err(OpsError::Invalid(format!(
                "world_vljepa shadow soak too short: {} < {} ticks",
                shadow.ticks_total, min_ticks
            )));
        }
        shadow_report_digest_prefix = Some(file_digest_prefix(path)?);
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
        .iter()
        .find(|s| s.slot_id == slot_key)
        .and_then(|s| s.active_hash.clone());
    let files = collect_file_entries(&promoted)?;
    upsert_slot_manifest(
        &mut manifest,
        slot_key.clone(),
        Some(hash.to_string()),
        files,
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
        shadow_report_digest_prefix,
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
        .iter()
        .find(|s| s.slot_id == slot_key)
        .and_then(|s| s.active_hash.clone());
    let files = collect_file_entries(&promoted)?;
    upsert_slot_manifest(
        &mut manifest,
        slot_key.clone(),
        Some(to_hash.to_string()),
        files,
    );
    persist_manifest_with_history(&mut manifest)?;
    persist_rollback_record(&slot_key, &from_hash, to_hash, &manifest)?;
    Ok(LifecycleActionReport {
        action: "rollback".to_string(),
        shadow_report_digest_prefix: None,
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
    let active_hash = manifest
        .slots
        .iter()
        .find(|s| s.slot_id == slot_key)
        .and_then(|s| s.active_hash.clone());
    Ok(ModelsListReport {
        slot: slot_key.clone(),
        active_hash,
        staged_hashes: list_hash_dirs(
            &PathBuf::from("models").join("staging").join(slot.as_str()),
        )?,
        promoted_hashes: list_hash_dirs(
            &PathBuf::from("models").join("promoted").join(slot.as_str()),
        )?,
    })
}

pub fn models_verify(manifest: &Path) -> Result<ModelsVerifyReport, OpsError> {
    if !manifest.exists() {
        return Ok(ModelsVerifyReport {
            manifest: manifest.display().to_string(),
            manifest_present: false,
            digest_match: false,
            promoted_hashes_exist: false,
            files_verified: false,
        });
    }
    let raw = fs::read_to_string(manifest)?;
    if raw.len() > MAX_MANIFEST_BYTES {
        return Err(OpsError::Invalid(format!(
            "manifest too large: {} > {} bytes",
            raw.len(),
            MAX_MANIFEST_BYTES
        )));
    }
    let parsed: LifecycleManifest = toml::from_str(&raw)
        .map_err(|e| OpsError::Invalid(format!("manifest parse failed: {e}")))?;
    let digest_match = compute_manifest_digest(&parsed)? == parsed.manifest_digest;
    let mut promoted_hashes_exist = true;
    let mut files_verified = true;
    for slot in &parsed.slots {
        if slot.files.len() > MAX_SLOT_FILES {
            return Err(OpsError::Invalid(format!(
                "slot {} has too many files: {} > {}",
                slot.slot_id,
                slot.files.len(),
                MAX_SLOT_FILES
            )));
        }
        if let Some(active_hash) = slot.active_hash.as_ref() {
            let root = PathBuf::from("models")
                .join("promoted")
                .join(&slot.slot_id)
                .join(active_hash);
            if !root.exists() {
                promoted_hashes_exist = false;
                continue;
            }
            for file in &slot.files {
                let path = root.join(&file.path);
                let metadata = match fs::metadata(&path) {
                    Ok(v) => v,
                    Err(_) => {
                        files_verified = false;
                        continue;
                    }
                };
                if metadata.len() != file.size_bytes {
                    files_verified = false;
                    continue;
                }
                if file_sha256_hex(&path)? != file.sha256 {
                    files_verified = false;
                }
            }
        }
    }
    Ok(ModelsVerifyReport {
        manifest: manifest.display().to_string(),
        manifest_present: true,
        digest_match,
        promoted_hashes_exist,
        files_verified,
    })
}

pub fn models_recommend_rollback(
    slot: ModelSlot,
    workdir: &Path,
) -> Result<ModelsRollbackRecommendationReport, OpsError> {
    let ess_path = workdir.join("ess").join("ess_fixture.json");
    let slot_key = slot.as_str().to_string();
    if !ess_path.exists() {
        return Ok(ModelsRollbackRecommendationReport {
            slot: slot_key,
            recommendations: Vec::new(),
        });
    }
    let mut recommendations = Vec::new();
    for record in load_fixture_records(&ess_path)? {
        let ExperiencePayload::Text(note) = record.payload else {
            continue;
        };
        let note = note.to_string();
        if note.starts_with("model_rollback_recommendation")
            && note.contains(&format!("slot={}", slot.as_str()))
        {
            recommendations.push(RollbackRecommendation {
                slot: slot.as_str().to_string(),
                tick: record.time.tick.get(),
                note,
            });
        }
    }
    recommendations.sort_by_key(|r| r.tick);
    recommendations.reverse();
    recommendations.truncate(8);
    Ok(ModelsRollbackRecommendationReport {
        slot: slot_key,
        recommendations,
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
    manifest.created_at = Some(now_secs());
    fs::create_dir_all("models")?;
    let body = toml::to_string_pretty(manifest)
        .map_err(|e| OpsError::Invalid(format!("manifest serialize failed: {e}")))?;
    fs::write(MANIFEST_PATH, body.as_bytes())?;
    fs::write(
        LEGACY_MANIFEST_PATH,
        legacy_manifest_toml(manifest)?.as_bytes(),
    )?;
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
    let canonical_path = PathBuf::from(MANIFEST_PATH);
    let legacy_path = PathBuf::from(LEGACY_MANIFEST_PATH);
    if !canonical_path.exists() && !legacy_path.exists() {
        let mut out = LifecycleManifest {
            manifest_version: 1,
            created_at: None,
            slots: Vec::new(),
            manifest_digest: String::new(),
        };
        out.manifest_digest = compute_manifest_digest(&out)?;
        return Ok(out);
    }
    let path = if canonical_path.exists() {
        canonical_path
    } else {
        legacy_path
    };
    let raw = fs::read_to_string(path)?;
    if let Ok(parsed) = toml::from_str::<LifecycleManifest>(&raw) {
        return Ok(parsed);
    }
    #[derive(Deserialize)]
    struct LegacySlot {
        active_hash: Option<String>,
    }
    #[derive(Deserialize)]
    struct LegacyManifest {
        slots: std::collections::BTreeMap<String, LegacySlot>,
        manifest_digest: Option<String>,
    }
    let legacy: LegacyManifest = toml::from_str(&raw)
        .map_err(|e| OpsError::Invalid(format!("manifest parse failed: {e}")))?;
    Ok(LifecycleManifest {
        manifest_version: 1,
        created_at: None,
        slots: legacy
            .slots
            .into_iter()
            .map(|(slot_id, slot)| LifecycleSlotManifest {
                slot_id,
                active_hash: slot.active_hash,
                files: Vec::new(),
                max_bytes: 64 * 1024 * 1024,
                contract_versions_supported: vec!["v1".to_string()],
            })
            .collect(),
        manifest_digest: legacy.manifest_digest.unwrap_or_default(),
    })
}

fn legacy_manifest_toml(manifest: &LifecycleManifest) -> Result<String, OpsError> {
    #[derive(Serialize)]
    struct LegacySlotManifest {
        active_hash: Option<String>,
    }
    #[derive(Serialize)]
    struct LegacyManifest {
        slots: std::collections::BTreeMap<String, LegacySlotManifest>,
        manifest_digest: String,
    }
    let slots = manifest
        .slots
        .iter()
        .map(|s| {
            (
                s.slot_id.clone(),
                LegacySlotManifest {
                    active_hash: s.active_hash.clone(),
                },
            )
        })
        .collect();
    toml::to_string_pretty(&LegacyManifest {
        slots,
        manifest_digest: manifest.manifest_digest.clone(),
    })
    .map_err(|e| OpsError::Invalid(format!("legacy manifest serialize failed: {e}")))
}

fn compute_manifest_digest(manifest: &LifecycleManifest) -> Result<String, OpsError> {
    #[derive(Serialize)]
    struct Canonical<'a> {
        manifest_version: u16,
        slots: &'a [LifecycleSlotManifest],
    }
    let mut slots = manifest.slots.clone();
    slots.sort_by(|a, b| a.slot_id.cmp(&b.slot_id));
    for slot in &mut slots {
        slot.files.sort_by(|a, b| a.path.cmp(&b.path));
        slot.contract_versions_supported.sort();
    }
    let canonical = serde_json::to_vec(&Canonical {
        manifest_version: manifest.manifest_version,
        slots: &slots,
    })?;
    Ok(sha256_hex(&canonical))
}

fn ensure_manifest_slot(slot: ModelSlot) -> Result<(), OpsError> {
    let mut manifest = load_or_init_manifest()?;
    let slot_id = slot.as_str().to_string();
    if !manifest.slots.iter().any(|s| s.slot_id == slot_id) {
        manifest.slots.push(LifecycleSlotManifest {
            slot_id,
            active_hash: None,
            files: Vec::new(),
            max_bytes: 64 * 1024 * 1024,
            contract_versions_supported: vec!["v1".to_string()],
        });
        persist_manifest_with_history(&mut manifest)?;
    }
    Ok(())
}

fn upsert_slot_manifest(
    manifest: &mut LifecycleManifest,
    slot_id: String,
    active_hash: Option<String>,
    files: Vec<ModelFileEntry>,
) {
    if let Some(entry) = manifest.slots.iter_mut().find(|s| s.slot_id == slot_id) {
        entry.active_hash = active_hash;
        entry.files = files;
        entry.max_bytes = 64 * 1024 * 1024;
        entry.contract_versions_supported = vec!["v1".to_string()];
        return;
    }
    manifest.slots.push(LifecycleSlotManifest {
        slot_id,
        active_hash,
        files,
        max_bytes: 64 * 1024 * 1024,
        contract_versions_supported: vec!["v1".to_string()],
    });
}

fn digest_entries(entries: &[ModelFileEntry]) -> String {
    let mut hasher = Sha256::new();
    for entry in entries {
        hasher.update(entry.path.as_bytes());
        hasher.update([0]);
        hasher.update(entry.sha256.as_bytes());
        hasher.update([0]);
        hasher.update(entry.size_bytes.to_le_bytes());
    }
    hex::encode(hasher.finalize())
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
        shadow_report_digest_prefix: None,
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
        shadow_report_digest_prefix: None,
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

#[cfg(test)]
mod tests {
    use super::*;

    struct CwdGuard {
        prev: PathBuf,
    }

    impl CwdGuard {
        fn enter(path: &Path) -> Self {
            let prev = std::env::current_dir().expect("cwd");
            std::env::set_current_dir(path).expect("chdir");
            Self { prev }
        }
    }

    impl Drop for CwdGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.prev);
        }
    }

    #[test]
    fn manifest_digest_canonical_stable() {
        let mut a = LifecycleManifest {
            manifest_version: 1,
            created_at: Some(10),
            slots: vec![
                LifecycleSlotManifest {
                    slot_id: "sae".to_string(),
                    active_hash: Some("bb".to_string()),
                    files: vec![ModelFileEntry {
                        path: "b.txt".to_string(),
                        sha256: "2".repeat(64),
                        size_bytes: 2,
                    }],
                    max_bytes: 1024,
                    contract_versions_supported: vec!["v2".to_string(), "v1".to_string()],
                },
                LifecycleSlotManifest {
                    slot_id: "llm".to_string(),
                    active_hash: Some("aa".to_string()),
                    files: vec![ModelFileEntry {
                        path: "a.txt".to_string(),
                        sha256: "1".repeat(64),
                        size_bytes: 1,
                    }],
                    max_bytes: 1024,
                    contract_versions_supported: vec!["v1".to_string()],
                },
            ],
            manifest_digest: String::new(),
        };
        let mut b = LifecycleManifest {
            manifest_version: 1,
            created_at: Some(99),
            slots: vec![a.slots[1].clone(), a.slots[0].clone()],
            manifest_digest: String::new(),
        };
        a.manifest_digest = compute_manifest_digest(&a).expect("digest");
        b.manifest_digest = compute_manifest_digest(&b).expect("digest");
        assert_eq!(a.manifest_digest, b.manifest_digest);
    }

    #[test]
    fn stage_and_verify_detects_tamper() {
        let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
        let dir = tempfile::tempdir().expect("tempdir");
        let _cwd = CwdGuard::enter(dir.path());

        let src = dir.path().join("src");
        fs::create_dir_all(&src).expect("src");
        fs::write(src.join("model.safetensors"), b"abc").expect("write");
        let staged = models_stage(ModelSlot::Llm, &src).expect("stage");
        let staged_path = dir
            .path()
            .join("models/staging/llm")
            .join(&staged.hash)
            .join("model.safetensors");
        assert!(staged_path.exists());

        let promoted = dir.path().join("models/promoted/llm").join(&staged.hash);
        copy_tree(
            &dir.path().join("models/staging/llm").join(&staged.hash),
            &promoted,
        )
        .expect("promoted copy");
        let mut manifest = load_or_init_manifest().expect("manifest");
        upsert_slot_manifest(
            &mut manifest,
            "llm".to_string(),
            Some(staged.hash.clone()),
            collect_file_entries(&promoted).expect("entries"),
        );
        persist_manifest_with_history(&mut manifest).expect("persist");

        let ok = models_verify(Path::new("models/MANIFEST.toml")).expect("verify");
        assert!(ok.digest_match && ok.promoted_hashes_exist && ok.files_verified);

        fs::write(promoted.join("model.safetensors"), b"tampered").expect("tamper");
        let bad = models_verify(Path::new("models/MANIFEST.toml")).expect("verify bad");
        assert!(!bad.files_verified);
    }

    #[test]
    fn stage_rejects_too_many_files() {
        let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
        let dir = tempfile::tempdir().expect("tempdir");
        let _cwd = CwdGuard::enter(dir.path());
        let src = dir.path().join("src");
        fs::create_dir_all(&src).expect("src");
        fs::write(src.join("model.safetensors"), b"abc").expect("model");
        for i in 0..MAX_SLOT_FILES {
            fs::write(src.join(format!("f_{i}.bin")), b"x").expect("file");
        }
        let err = models_stage(ModelSlot::Llm, &src).expect_err("must reject");
        assert!(format!("{err}").contains("too many files"));
    }
}
