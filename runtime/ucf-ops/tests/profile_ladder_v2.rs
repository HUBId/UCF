use std::fs;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use tempfile::tempdir;
use ucf_ops::{load_or_init_config, migrate_config_v1, one_command_bringup, ConfigV1};

fn load_cfg(path: &Path) -> ConfigV1 {
    let raw = fs::read_to_string(path).expect("read config");
    ConfigV1::from_toml_str(&raw).expect("parse config")
}

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[test]
fn config_schema_rejects_unknown_keys() {
    let bad = r#"
profile = "test"
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
llm_max_tokens = 64
probe_timeout_ms = 150
extra_unsupported_knob = true
"#;
    let err = ConfigV1::from_toml_str(bad).expect_err("unknown key must fail");
    assert!(err.to_string().contains("unknown field"));
}

#[test]
fn ladder_never_reduces_safety() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dev = load_cfg(&repo.join("configs/dev.toml"));
    let test = load_cfg(&repo.join("configs/test.toml"));
    let prod = load_cfg(&repo.join("configs/prod.toml"));

    assert!(dev.runtime.offline && test.runtime.offline && prod.runtime.offline);
    assert!(!test.runtime.sampling_enabled && !prod.runtime.sampling_enabled);
    assert!(test.strictness.determinism_lock && prod.strictness.determinism_lock);
    assert_eq!(test.runtime.capabilities_default, "deny");
    assert_eq!(prod.runtime.capabilities_default, "deny");
    assert!(prod.runtime.docs_lint_required);
}

#[test]
fn migrated_config_validates_strictly() {
    let dir = tempdir().expect("tempdir");
    let old = dir.path().join("old.toml");
    let new = dir.path().join("new.toml");
    let diff = dir.path().join("diff.txt");
    fs::write(
        &old,
        "profile='test'\npolicy_overlay='test'\nbackend_pack='toy_v1'\nslot_ebm_mode='shadow'\n",
    )
    .expect("old write");
    migrate_config_v1(&old, &new, &diff).expect("migrate");
    let new_raw = fs::read_to_string(&new).expect("read new");
    let migrated = ConfigV1::from_toml_str(&new_raw).expect("validate strict");
    assert_eq!(migrated.profile_name, "test");
}

#[test]
fn bringup_records_profile_and_config_digest() {
    let _lock = env_lock().lock().expect("env lock");
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let scenario = repo.join("fixtures/e2e_scenario_a.json");
    for profile in ["dev", "test"] {
        std::env::set_var("UCF_PROFILE", profile);
        std::env::remove_var("UCF_POLICY_OVERLAY");
        std::env::remove_var("UCF_SLOT_EBM_MODE");
        std::env::remove_var("UCF_STAGE_ISOLATION");
        std::env::remove_var("UCF_EMERGENCY_POLICY_PIN");
        let dir = tempdir().expect("tempdir");
        let out = dir.path().join("out");

        let artifacts =
            one_command_bringup(dir.path(), &scenario, 6, &out, false).expect("bringup");
        let cfg = load_or_init_config(dir.path()).expect("cfg");

        assert_eq!(artifacts.run_metadata.profile, profile);
        assert_eq!(artifacts.run_metadata.config_digest, cfg.config_digest);
        assert_eq!(artifacts.run_metadata.policy_overlay, cfg.policy_overlay);
    }
}

#[test]
fn prod_bringup_requires_burn_backend_feature() {
    let _lock = env_lock().lock().expect("env lock");
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let scenario = repo.join("fixtures/e2e_scenario_a.json");
    std::env::set_var("UCF_PROFILE", "prod");
    std::env::remove_var("UCF_POLICY_OVERLAY");
    std::env::remove_var("UCF_SLOT_EBM_MODE");
    std::env::remove_var("UCF_STAGE_ISOLATION");
    std::env::remove_var("UCF_EMERGENCY_POLICY_PIN");
    let dir = tempdir().expect("tempdir");
    let out = dir.path().join("out");

    #[cfg(feature = "backend-burn")]
    {
        let artifacts =
            one_command_bringup(dir.path(), &scenario, 6, &out, false).expect("prod bringup");
        let cfg = load_or_init_config(dir.path()).expect("cfg");
        assert_eq!(artifacts.run_metadata.profile, "prod");
        assert_eq!(artifacts.run_metadata.config_digest, cfg.config_digest);
    }

    #[cfg(not(feature = "backend-burn"))]
    {
        let err =
            one_command_bringup(dir.path(), &scenario, 6, &out, false).expect_err("burn missing");
        assert!(err
            .to_string()
            .contains("pack burn_toy_v1 requires feature backend-burn"));
    }
}
