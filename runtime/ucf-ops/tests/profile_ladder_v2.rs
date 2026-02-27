use std::fs;
use std::path::Path;

use tempfile::tempdir;
use ucf_ops::{load_or_init_config, one_command_bringup, OpsConfig};

fn load_cfg(path: &Path) -> OpsConfig {
    let raw = fs::read_to_string(path).expect("read config");
    toml::from_str(&raw).expect("parse config")
}

#[test]
fn config_schema_rejects_unknown_keys() {
    let bad = r#"
profile = "test"
policy_overlay = "test"
backend_pack = "toy_v1"
slot_ebm_mode = "shadow"
offline = true
compute_backend = "stub"
compute_seed = 1
compute_budget_profile = "tight"
isolation_runtime = "inproc"
capabilities_default = "deny"
sampling_enabled = false
determinism_lock_strict = true
docs_lint_required = false
stage_isolation_optional = true
log_level = "info"
extra_unsupported_knob = true
"#;
    let err = toml::from_str::<OpsConfig>(bad).expect_err("unknown key must fail");
    assert!(err.to_string().contains("unknown field"));
}

#[test]
fn ladder_never_reduces_safety() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dev = load_cfg(&repo.join("configs/dev.toml"));
    let test = load_cfg(&repo.join("configs/test.toml"));
    let prod = load_cfg(&repo.join("configs/prod.toml"));

    assert!(dev.offline && test.offline && prod.offline);
    assert!(!test.sampling_enabled && !prod.sampling_enabled);
    assert!(test.determinism_lock_strict && prod.determinism_lock_strict);
    assert_eq!(test.capabilities_default, "deny");
    assert_eq!(prod.capabilities_default, "deny");
    assert!(prod.docs_lint_required);
}

#[test]
fn bringup_records_profile_and_config_digest() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let scenario = repo.join("fixtures/e2e_scenario_a.json");
    for profile in ["dev", "test", "prod"] {
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
