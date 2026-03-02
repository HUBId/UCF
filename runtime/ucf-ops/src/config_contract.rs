use crate::{sha256_hex, OpsConfig, OpsError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use ucf_compute::ComputeBackendKind;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ConfigV1 {
    pub profile_name: String,
    pub policy_overlay: String,
    pub device_profile: DeviceProfileName,
    pub slot_modes: SlotModesV1,
    pub paths: ConfigPathsV1,
    pub strictness: StrictnessV1,
    pub runtime: RuntimeConfigV1,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SlotModesV1 {
    pub ebm: SlotMode,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SlotMode {
    Shadow,
    Active,
    Off,
}

impl SlotMode {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Shadow => "shadow",
            Self::Active => "active",
            Self::Off => "off",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DeviceProfileName {
    Small,
    Medium,
    Large,
}

impl DeviceProfileName {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Small => "small",
            Self::Medium => "medium",
            Self::Large => "large",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ConfigPathsV1 {
    pub policy_pack: String,
    pub policy_overlay: String,
    pub models_manifest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StrictnessV1 {
    pub determinism_lock: bool,
    pub stage_isolation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RuntimeConfigV1 {
    pub backend_pack: String,
    pub offline: bool,
    pub compute_backend: ComputeBackendKind,
    pub compute_seed: u64,
    pub capabilities_default: String,
    pub sampling_enabled: bool,
    pub docs_lint_required: bool,
    pub isolation_runtime: String,
    pub emergency_policy_pin: Option<String>,
    pub log_level: String,
    pub llm_max_tokens: u16,
    pub probe_timeout_ms: u64,
}

impl ConfigV1 {
    pub fn from_toml_str(raw: &str) -> Result<Self, OpsError> {
        let cfg: Self =
            toml::from_str(raw).map_err(|e| OpsError::Invalid(format!("invalid ConfigV1: {e}")))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), OpsError> {
        if self.profile_name != "dev" && self.profile_name != "test" && self.profile_name != "prod"
        {
            return Err(OpsError::Invalid(format!(
                "profile_name must be dev|test|prod, got {}",
                self.profile_name
            )));
        }
        if !(1..=8192).contains(&self.runtime.llm_max_tokens) {
            return Err(OpsError::Invalid(format!(
                "runtime.llm_max_tokens out of range [1,8192]: {}",
                self.runtime.llm_max_tokens
            )));
        }
        if !(1..=60_000).contains(&self.runtime.probe_timeout_ms) {
            return Err(OpsError::Invalid(format!(
                "runtime.probe_timeout_ms out of range [1,60000]: {}",
                self.runtime.probe_timeout_ms
            )));
        }
        Ok(())
    }

    pub fn into_ops_config(self) -> OpsConfig {
        OpsConfig {
            profile: self.profile_name,
            policy_overlay: self.policy_overlay,
            backend_pack: self.runtime.backend_pack,
            slot_ebm_mode: self.slot_modes.ebm.as_str().to_string(),
            offline: self.runtime.offline,
            compute_backend: self.runtime.compute_backend,
            compute_seed: self.runtime.compute_seed,
            compute_budget_profile: self.device_profile.as_str().to_string(),
            device_profile: self.device_profile.as_str().to_string(),
            isolation_runtime: self.runtime.isolation_runtime,
            capabilities_default: self.runtime.capabilities_default,
            sampling_enabled: self.runtime.sampling_enabled,
            determinism_lock_strict: self.strictness.determinism_lock,
            docs_lint_required: self.runtime.docs_lint_required,
            stage_isolation_optional: self.strictness.stage_isolation,
            emergency_policy_pin: self.runtime.emergency_policy_pin,
            log_level: self.runtime.log_level,
            config_digest: String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MigrateReport {
    pub warnings: Vec<String>,
    pub actions: Vec<String>,
    pub old_digest: String,
    pub new_digest: String,
}

#[derive(Debug, Deserialize)]
struct LegacyConfig {
    profile: Option<String>,
    policy_overlay: Option<String>,
    backend_pack: Option<String>,
    slot_ebm_mode: Option<String>,
    offline: Option<bool>,
    compute_backend: Option<ComputeBackendKind>,
    compute_seed: Option<u64>,
    device_profile: Option<String>,
    capabilities_default: Option<String>,
    sampling_enabled: Option<bool>,
    determinism_lock_strict: Option<bool>,
    docs_lint_required: Option<bool>,
    stage_isolation_optional: Option<bool>,
    isolation_runtime: Option<String>,
    emergency_policy_pin: Option<String>,
    log_level: Option<String>,
    #[serde(flatten)]
    unknown: BTreeMap<String, toml::Value>,
}

pub fn migrate_config_v1(
    input: &Path,
    output: &Path,
    diff_out: &Path,
) -> Result<MigrateReport, OpsError> {
    let raw = fs::read_to_string(input)?;
    let old_digest = sha256_hex(raw.as_bytes());
    let legacy: LegacyConfig = toml::from_str(&raw).map_err(|e| {
        OpsError::Invalid(format!("invalid legacy config {}: {e}", input.display()))
    })?;

    let mut warnings = Vec::new();
    for key in legacy.unknown.keys() {
        warnings.push(format!("unknown legacy key ignored: {key}"));
    }

    let profile_name = legacy.profile.unwrap_or_else(|| "test".to_string());
    let device_profile = match legacy.device_profile.as_deref().unwrap_or("small") {
        "small" => DeviceProfileName::Small,
        "medium" => DeviceProfileName::Medium,
        "large" => DeviceProfileName::Large,
        other => {
            warnings.push(format!(
                "invalid legacy device_profile={other}; defaulting to small"
            ));
            DeviceProfileName::Small
        }
    };

    let slot_mode = match legacy.slot_ebm_mode.as_deref().unwrap_or("shadow") {
        "shadow" => SlotMode::Shadow,
        "active" => SlotMode::Active,
        "off" => SlotMode::Off,
        other => {
            warnings.push(format!(
                "invalid legacy slot_ebm_mode={other}; defaulting to shadow"
            ));
            SlotMode::Shadow
        }
    };

    let new_cfg = ConfigV1 {
        profile_name: profile_name.clone(),
        policy_overlay: legacy.policy_overlay.unwrap_or(profile_name.clone()),
        device_profile,
        slot_modes: SlotModesV1 { ebm: slot_mode },
        paths: ConfigPathsV1 {
            policy_pack: "policies/packs/base_v1".to_string(),
            policy_overlay: format!("policies/packs/overlays/{profile_name}"),
            models_manifest: "models/manifest.toml".to_string(),
        },
        strictness: StrictnessV1 {
            determinism_lock: legacy.determinism_lock_strict.unwrap_or(true),
            stage_isolation: legacy.stage_isolation_optional.unwrap_or(true),
        },
        runtime: RuntimeConfigV1 {
            backend_pack: legacy.backend_pack.unwrap_or_else(|| "toy_v1".to_string()),
            offline: legacy.offline.unwrap_or(true),
            compute_backend: legacy.compute_backend.unwrap_or(ComputeBackendKind::Stub),
            compute_seed: legacy.compute_seed.unwrap_or(0xDEC0DED),
            capabilities_default: legacy
                .capabilities_default
                .unwrap_or_else(|| "deny".to_string()),
            sampling_enabled: legacy.sampling_enabled.unwrap_or(false),
            docs_lint_required: legacy.docs_lint_required.unwrap_or(false),
            isolation_runtime: legacy
                .isolation_runtime
                .unwrap_or_else(|| "inproc".to_string()),
            emergency_policy_pin: legacy.emergency_policy_pin,
            log_level: legacy.log_level.unwrap_or_else(|| "info".to_string()),
            llm_max_tokens: 128,
            probe_timeout_ms: 200,
        },
    };
    new_cfg.validate()?;

    let new_raw = toml::to_string_pretty(&new_cfg)
        .map_err(|e| OpsError::Invalid(format!("unable to render ConfigV1: {e}")))?;
    let new_digest = sha256_hex(new_raw.as_bytes());
    fs::write(output, &new_raw)?;

    let mut diff = Vec::new();
    diff.push("Config migration summary".to_string());
    diff.push(format!("- old: {}", input.display()));
    diff.push(format!("- new: {}", output.display()));
    diff.push(format!("- old_digest: {old_digest}"));
    diff.push(format!("- new_digest: {new_digest}"));
    diff.push("- actions:".to_string());
    diff.push("  - verify runtime.paths.* are valid bundle-relative paths".to_string());
    diff.push("  - run `ucf-ops docs lint --strict` after migration".to_string());
    for warning in &warnings {
        diff.push(format!("- warning: {warning}"));
    }
    let mut bounded = diff.join("\n");
    if bounded.len() > 8192 {
        bounded.truncate(8192);
        bounded.push_str("\n...truncated...");
    }
    fs::write(diff_out, bounded)?;

    Ok(MigrateReport {
        warnings,
        actions: vec![
            "verify runtime.paths.* are valid bundle-relative paths".to_string(),
            "run `ucf-ops docs lint --strict` after migration".to_string(),
        ],
        old_digest,
        new_digest,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PolicyKeyEntryV1 {
    pub name: &'static str,
    pub key_type: &'static str,
    pub range: &'static str,
    pub default: &'static str,
    pub module: &'static str,
}

pub fn policy_key_registry_v1() -> Vec<PolicyKeyEntryV1> {
    vec![
        PolicyKeyEntryV1 {
            name: "governor_tier_1_q",
            key_type: "u64",
            range: "0..=10000",
            default: "2500",
            module: "governor",
        },
        PolicyKeyEntryV1 {
            name: "governor_tier_2_q",
            key_type: "u64",
            range: "0..=10000",
            default: "5000",
            module: "governor",
        },
        PolicyKeyEntryV1 {
            name: "governor_tier_3_q",
            key_type: "u64",
            range: "0..=10000",
            default: "7500",
            module: "governor",
        },
        PolicyKeyEntryV1 {
            name: "ebm_high_risk_q",
            key_type: "u64",
            range: "0..=10000",
            default: "7000",
            module: "ebm",
        },
        PolicyKeyEntryV1 {
            name: "ebm_low_risk_q",
            key_type: "u64",
            range: "0..=10000",
            default: "3000",
            module: "ebm",
        },
    ]
}

pub fn export_policy_key_registry_v1(out: &Path) -> Result<(), OpsError> {
    let mut lines = vec![
        "# Policy Key Registry v1".to_string(),
        "".to_string(),
        "| key | type | range | default | module |".to_string(),
        "|---|---|---|---|---|".to_string(),
    ];
    for entry in policy_key_registry_v1() {
        lines.push(format!(
            "| {} | {} | {} | {} | {} |",
            entry.name, entry.key_type, entry.range, entry.default, entry.module
        ));
    }
    fs::write(out, lines.join("\n"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_v1_rejects_unknown_keys() {
        let raw = r#"
profile_name = "test"
policy_overlay = "test"
device_profile = "small"

[slot_modes]
ebm = "shadow"

[paths]
policy_pack = "policies/packs/base_v1"
policy_overlay = "policies/packs/overlays/test"
models_manifest = "models/manifest.toml"

[strictness]
determinism_lock = true
stage_isolation = true

[runtime]
backend_pack = "toy_v1"
offline = true
compute_backend = "stub"
compute_seed = 1
capabilities_default = "deny"
sampling_enabled = false
docs_lint_required = false
isolation_runtime = "inproc"
log_level = "info"
llm_max_tokens = 128
probe_timeout_ms = 200
oops = 1
"#;
        assert!(ConfigV1::from_toml_str(raw).is_err());
    }

    #[test]
    fn migration_emits_warning_for_unknown_keys() {
        let dir = tempfile::tempdir().expect("tmp");
        let input = dir.path().join("old.toml");
        let output = dir.path().join("new.toml");
        let diff = dir.path().join("diff.txt");
        fs::write(
            &input,
            "profile='test'\npolicy_overlay='test'\nunknown_legacy=1\n",
        )
        .expect("write");
        let report = migrate_config_v1(&input, &output, &diff).expect("migrate");
        assert!(!report.warnings.is_empty());
        let new_raw = fs::read_to_string(output).expect("new");
        assert!(new_raw.contains("profile_name = \"test\""));
    }
}
