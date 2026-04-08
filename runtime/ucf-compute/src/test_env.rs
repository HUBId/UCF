use std::sync::{Mutex, OnceLock};

use crate::ModelSlot;

pub fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

pub struct EnvSnapshot {
    values: Vec<(String, Option<String>)>,
}

impl Drop for EnvSnapshot {
    fn drop(&mut self) {
        for (key, value) in &self.values {
            if let Some(value) = value {
                std::env::set_var(key, value);
            } else {
                std::env::remove_var(key);
            }
        }
    }
}

pub fn clear_model_env_overrides() -> EnvSnapshot {
    let mut keys = vec![
        "UCF_MODEL_MANIFEST".to_string(),
        "UCF_REAL_ENABLEMENT_MODE".to_string(),
        "UCF_SHADOW_EVERY_N_TICKS".to_string(),
        "UCF_SLOT_SHADOW_RATE".to_string(),
        "UCF_SLOT_COMPARE_WINDOW".to_string(),
        "UCF_RUNTIME_MODE".to_string(),
        "UCF_DEPLOYMENT_PROFILE".to_string(),
        "UCF_COMPUTE_BACKEND".to_string(),
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
    EnvSnapshot { values }
}
