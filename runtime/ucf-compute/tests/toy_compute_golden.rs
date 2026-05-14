use std::sync::{Mutex, OnceLock};

use ucf_compute::{
    run_toy_compute_golden_fixture, toy_compute_golden_digest, AiComputeBackend, BackendClass,
    BackendComponentId, BackendPackConfig, BackendPackFactory, BackendPackKind, ModelSlot,
    ToyComputeBackend, TOY_COMPUTE_GOLDEN_VERSION,
};

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

struct EnvSnapshot(Vec<(String, Option<String>)>);

impl EnvSnapshot {
    fn clear_model_overrides() -> Self {
        let mut keys = vec![
            "UCF_MODEL_MANIFEST".to_string(),
            "UCF_BACKEND_PACK".to_string(),
            "UCF_BACKEND_SEED".to_string(),
            "UCF_REAL_ENABLEMENT_MODE".to_string(),
            "UCF_COMPUTE_BACKEND".to_string(),
            "UCF_NSR_MODE".to_string(),
        ];
        for slot in ModelSlot::all() {
            keys.push(format!("UCF_MODEL_{}_ENABLED", slot.env_key()));
            keys.push(format!("UCF_MODEL_PIN_{}", slot.env_key()));
            keys.push(format!("UCF_MODEL_CANDIDATE_{}", slot.env_key()));
            keys.push(format!("UCF_MODEL_COMPARE_{}", slot.env_key()));
            keys.push(format!("UCF_MODEL_SHADOW_{}", slot.env_key()));
        }
        let values = keys
            .iter()
            .map(|key| (key.clone(), std::env::var(key).ok()))
            .collect::<Vec<_>>();
        for key in keys {
            std::env::remove_var(key);
        }
        Self(values)
    }
}

impl Drop for EnvSnapshot {
    fn drop(&mut self) {
        for (key, value) in &self.0 {
            if let Some(value) = value {
                std::env::set_var(key, value);
            } else {
                std::env::remove_var(key);
            }
        }
    }
}

#[test]
fn toy_compute_golden_is_deterministic() {
    let _guard = env_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _env = EnvSnapshot::clear_model_overrides();

    let first = run_toy_compute_golden_fixture().expect("first toy golden");
    let second = run_toy_compute_golden_fixture().expect("second toy golden");

    println!("toy_golden_digest={}", hex::encode(first.digest));
    assert_eq!(first.provenance, second.provenance);
    assert_eq!(first.input, second.input);
    assert_eq!(first.signals, second.signals);
    assert_eq!(first.summary, second.summary);
    assert_eq!(first.digest, second.digest);
    assert_eq!(
        first.digest,
        toy_compute_golden_digest(
            &first.provenance,
            &first.input,
            &first.signals,
            &first.summary
        )
    );
    assert_ne!(first.digest, [0_u8; 32]);
}

#[test]
fn toy_compute_golden_reports_toy_provenance() {
    let _guard = env_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _env = EnvSnapshot::clear_model_overrides();

    let backend = ToyComputeBackend;
    let identity = backend.identity();
    let golden = run_toy_compute_golden_fixture().expect("toy golden");

    assert_eq!(identity.class, BackendClass::Toy);
    assert_eq!(golden.provenance.backend_class, BackendClass::Toy);
    assert_eq!(golden.provenance.backend_name, "toy_v1");
    assert_eq!(golden.provenance.fixture_id, TOY_COMPUTE_GOLDEN_VERSION);
    assert_eq!(golden.provenance.golden_version, TOY_COMPUTE_GOLDEN_VERSION);
    assert!(golden.provenance.toy_not_real);
    assert!(golden.provenance.no_real_inference);
    assert!(!golden.provenance.production_claim);
    assert_eq!(golden.summary.backend_profile, "toy:v1");
    assert_eq!(golden.summary.backend_pack_id, 1);
}

#[test]
fn toy_compute_golden_is_offline_no_external_artifacts() {
    let _guard = env_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _env = EnvSnapshot::clear_model_overrides();

    let identity = BackendPackKind::ToyV1.identity();
    let pack = BackendPackFactory::build(BackendPackConfig {
        pack: BackendPackKind::ToyV1,
        seed: 0x70,
    })
    .expect("toy pack");
    let golden = run_toy_compute_golden_fixture().expect("toy golden");

    assert!(identity.offline);
    assert!(identity.deterministic);
    assert!(!identity.external_service_required);
    assert!(!golden.provenance.external_service_required);
    assert!(pack.model_slot_provenance().iter().all(|slot| {
        !slot.required_for_pack && slot.resolved_path.is_none() && slot.hash_prefix.is_none()
    }));
    assert_eq!(golden.summary.llm_backend, BackendComponentId::ToyV1 as u8);
    assert_eq!(
        golden.summary.world_backend,
        BackendComponentId::ToyV1 as u8
    );
    assert_eq!(golden.summary.sae_backend, BackendComponentId::ToyV1 as u8);
    assert_eq!(golden.summary.ssm_backend, BackendComponentId::ToyV1 as u8);
    assert_eq!(golden.summary.lfm_backend, BackendComponentId::ToyV1 as u8);
}

#[test]
fn toy_compute_golden_does_not_claim_real_or_production() {
    let _guard = env_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _env = EnvSnapshot::clear_model_overrides();

    let identity = BackendPackKind::ToyV1.identity();
    let golden = run_toy_compute_golden_fixture().expect("toy golden");

    assert_ne!(identity.class, BackendClass::OptionalRealRuntime);
    assert_ne!(identity.class, BackendClass::Stub);
    assert!(!identity.runtime_inference_supported);
    assert!(!identity.claims_runtime_real_inference());
    assert!(!identity.production_claim);
    assert!(!golden.provenance.runtime_inference_supported);
    assert!(golden.provenance.no_real_inference);
    assert!(!golden.provenance.production_claim);
    assert!(!golden.provenance.minimal_spine_authority);
}

#[test]
fn no_current_toy_backend_claims_production() {
    for identity in [
        BackendPackKind::ToyV1.identity(),
        ToyComputeBackend.identity(),
    ] {
        assert_eq!(identity.class, BackendClass::Toy);
        assert!(identity.default_safe());
        assert!(!identity.production_claim);
        assert!(!identity.claims_runtime_real_inference());
    }
}

#[test]
fn toy_compute_golden_digest_is_pinned() {
    let _guard = env_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _env = EnvSnapshot::clear_model_overrides();

    let golden = run_toy_compute_golden_fixture().expect("toy golden");
    assert_eq!(
        hex::encode(golden.digest),
        "fe41668287a09278dc820b2d004df053755cc03d33c15a72fc323c4ec8425dad",
        "intentional toy golden output changes must review and update this digest"
    );
}
