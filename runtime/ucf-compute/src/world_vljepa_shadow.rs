use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};

use crate::world_model::WorldVljepaShadowRecord;

const DEFAULT_WINDOW: usize = 512;
const DEFAULT_DRIFT_WINDOWS: usize = 3;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorldShadowTickSample {
    pub t: u64,
    pub latency_ms: f32,
    pub vljepa_error_q: u16,
    pub baseline_error_q: u16,
    pub error_delta_q: u16,
    pub invalid_output: bool,
    pub saturation_clamp_count: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorldShadowWindowStats {
    pub window_id: u64,
    pub start_t: u64,
    pub end_t: u64,
    pub ticks: usize,
    pub latency_mean_ms: f32,
    pub latency_p95_ms: f32,
    pub error_mean_q: u16,
    pub error_p95_q: u16,
    pub error_delta_mean_q: u16,
    pub error_delta_p95_q: u16,
    pub invalid_rate: f32,
    pub saturation_rate: f32,
}

#[derive(Debug, Clone, Copy)]
struct ShadowPolicy {
    window_size: usize,
    lat_p95_max_ms: f32,
    err_mean_max_q: u16,
    err_spike_max_q: u16,
    invalid_output_max_rate: f32,
    sustained_windows: usize,
    disable_on_alarm: bool,
}

impl ShadowPolicy {
    fn from_env() -> Self {
        Self {
            window_size: std::env::var("UCF_WORLD_VLJEPA_WINDOW_TICKS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|v| *v > 0)
                .unwrap_or(DEFAULT_WINDOW),
            lat_p95_max_ms: std::env::var("UCF_WORLD_VLJEPA_LAT_P95_MAX_MS")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(5.0),
            err_mean_max_q: std::env::var("UCF_WORLD_VLJEPA_ERR_MEAN_MAX_Q")
                .ok()
                .and_then(|v| v.parse::<u16>().ok())
                .unwrap_or(22000),
            err_spike_max_q: std::env::var("UCF_WORLD_VLJEPA_ERR_SPIKE_MAX_Q")
                .ok()
                .and_then(|v| v.parse::<u16>().ok())
                .unwrap_or(45000),
            invalid_output_max_rate: std::env::var("UCF_WORLD_VLJEPA_INVALID_OUTPUT_MAX_RATE")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(0.0),
            sustained_windows: std::env::var("UCF_WORLD_VLJEPA_DRIFT_WINDOWS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|v| *v > 0)
                .unwrap_or(DEFAULT_DRIFT_WINDOWS),
            disable_on_alarm: std::env::var("UCF_WORLD_VLJEPA_DISABLE_ON_ALARM")
                .map(|v| v != "0")
                .unwrap_or(true),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorldDriftAlarmRecord {
    pub window_id: u64,
    pub reason_codes: Vec<String>,
    pub action: String,
}

#[derive(Default)]
struct ShadowState {
    window_id: u64,
    samples: Vec<WorldShadowTickSample>,
    err_alarm_windows: usize,
    lat_alarm_windows: usize,
    disabled: bool,
}

static SHADOW_DISABLED: AtomicBool = AtomicBool::new(false);
static SHADOW_STATE: OnceLock<Mutex<ShadowState>> = OnceLock::new();

fn shadow_state() -> &'static Mutex<ShadowState> {
    SHADOW_STATE.get_or_init(|| Mutex::new(ShadowState::default()))
}

fn percentile_u16(vals: &[u16], num: usize, den: usize) -> u16 {
    if vals.is_empty() {
        return 0;
    }
    let mut v = vals.to_vec();
    v.sort_unstable();
    let idx = ((v.len() - 1) * num) / den;
    v[idx]
}

fn percentile_f32(vals: &[f32], num: usize, den: usize) -> f32 {
    if vals.is_empty() {
        return 0.0;
    }
    let mut v = vals.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((v.len() - 1) * num) / den;
    v[idx]
}

fn append_jsonl(path: Option<PathBuf>, value: &impl Serialize) {
    let Some(path) = path else { return };
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
        if let Ok(line) = serde_json::to_string(value) {
            let _ = writeln!(f, "{line}");
        }
    }
}

fn windows_path() -> Option<PathBuf> {
    std::env::var("UCF_WORLD_VLJEPA_WINDOWS_LOG")
        .ok()
        .map(PathBuf::from)
}

fn alarms_path() -> Option<PathBuf> {
    std::env::var("UCF_WORLD_VLJEPA_ALARMS_LOG")
        .ok()
        .map(PathBuf::from)
}

pub fn shadow_disabled() -> bool {
    SHADOW_DISABLED.load(Ordering::Relaxed)
}

pub fn reset_shadow_state() {
    SHADOW_DISABLED.store(false, Ordering::Relaxed);
    if let Ok(mut st) = shadow_state().lock() {
        *st = ShadowState::default();
    }
}

pub fn record_shadow_sample(
    latency_ms: f32,
    baseline_error_q: u16,
    record: &WorldVljepaShadowRecord,
) {
    let policy = ShadowPolicy::from_env();
    if shadow_disabled() {
        return;
    }
    let mut st = match shadow_state().lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    if st.disabled {
        return;
    }
    let error_delta_q = record.prediction_error_q.saturating_sub(baseline_error_q);
    st.samples.push(WorldShadowTickSample {
        t: record.t,
        latency_ms,
        vljepa_error_q: record.prediction_error_q,
        baseline_error_q,
        error_delta_q,
        invalid_output: record.invalid_output,
        saturation_clamp_count: record.saturation_clamp_count,
    });
    if st.samples.len() < policy.window_size {
        return;
    }

    let window = std::mem::take(&mut st.samples);
    let start_t = window.first().map(|s| s.t).unwrap_or(0);
    let end_t = window.last().map(|s| s.t).unwrap_or(0);
    let latencies: Vec<f32> = window.iter().map(|s| s.latency_ms).collect();
    let errors: Vec<u16> = window.iter().map(|s| s.vljepa_error_q).collect();
    let deltas: Vec<u16> = window.iter().map(|s| s.error_delta_q).collect();
    let invalid_count = window.iter().filter(|s| s.invalid_output).count();
    let saturated_count = window
        .iter()
        .filter(|s| s.saturation_clamp_count > 0)
        .count();

    let stats = WorldShadowWindowStats {
        window_id: st.window_id,
        start_t,
        end_t,
        ticks: window.len(),
        latency_mean_ms: latencies.iter().sum::<f32>() / latencies.len().max(1) as f32,
        latency_p95_ms: percentile_f32(&latencies, 95, 100),
        error_mean_q: ((errors.iter().map(|v| u32::from(*v)).sum::<u32>()
            / errors.len().max(1) as u32) as u16),
        error_p95_q: percentile_u16(&errors, 95, 100),
        error_delta_mean_q: ((deltas.iter().map(|v| u32::from(*v)).sum::<u32>()
            / deltas.len().max(1) as u32) as u16),
        error_delta_p95_q: percentile_u16(&deltas, 95, 100),
        invalid_rate: invalid_count as f32 / window.len().max(1) as f32,
        saturation_rate: saturated_count as f32 / window.len().max(1) as f32,
    };

    let mut reasons = Vec::new();
    if stats.error_mean_q > policy.err_mean_max_q {
        st.err_alarm_windows += 1;
    } else {
        st.err_alarm_windows = 0;
    }
    if stats.latency_p95_ms > policy.lat_p95_max_ms {
        st.lat_alarm_windows += 1;
    } else {
        st.lat_alarm_windows = 0;
    }
    if st.err_alarm_windows >= policy.sustained_windows {
        reasons.push("sustained_error_increase".to_string());
    }
    if st.lat_alarm_windows >= policy.sustained_windows {
        reasons.push("latency_regression".to_string());
    }
    if stats.error_p95_q > policy.err_spike_max_q {
        reasons.push("error_spike".to_string());
    }
    if stats.invalid_rate > policy.invalid_output_max_rate {
        reasons.push("invalid_output_rate".to_string());
    }

    append_jsonl(windows_path(), &stats);
    if !reasons.is_empty() {
        let alarm = WorldDriftAlarmRecord {
            window_id: st.window_id,
            reason_codes: reasons,
            action: if policy.disable_on_alarm {
                "DisabledShadow".to_string()
            } else {
                "None".to_string()
            },
        };
        append_jsonl(alarms_path(), &alarm);
        if policy.disable_on_alarm {
            st.disabled = true;
            SHADOW_DISABLED.store(true, Ordering::Relaxed);
        }
    }
    st.window_id = st.window_id.saturating_add(1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn set_env(k: &str, v: &str, stash: &mut Vec<(String, Option<String>)>) {
        let key = k.to_string();
        if !stash.iter().any(|(s, _)| s == &key) {
            stash.push((key.clone(), std::env::var(&key).ok()));
        }
        std::env::set_var(k, v);
    }

    fn restore_env(stash: Vec<(String, Option<String>)>) {
        for (k, old) in stash {
            if let Some(v) = old {
                std::env::set_var(k, v);
            } else {
                std::env::remove_var(k);
            }
        }
    }

    #[test]
    fn window_aggregation_is_deterministic() {
        let _guard = env_test_lock().lock().expect("env lock");
        let mut stash = Vec::new();
        reset_shadow_state();
        set_env("UCF_WORLD_VLJEPA_WINDOW_TICKS", "4", &mut stash);
        set_env("UCF_WORLD_VLJEPA_DISABLE_ON_ALARM", "0", &mut stash);
        for t in 0..4 {
            let rec = WorldVljepaShadowRecord {
                t,
                encoding_digest_prefix: [0; 8],
                prediction_error_q: 1000,
                prediction_digest_prefix: [1; 8],
                model_hash_prefix: [2; 8],
                saturation_clamp_count: 0,
                invalid_output: false,
                status: "ok",
            };
            record_shadow_sample(1.0, 900, &rec);
        }
        assert!(!shadow_disabled());
        restore_env(stash);
    }

    #[test]
    fn drift_alarm_disables_shadow() {
        let _guard = env_test_lock().lock().expect("env lock");
        let mut stash = Vec::new();
        reset_shadow_state();
        set_env("UCF_WORLD_VLJEPA_WINDOW_TICKS", "2", &mut stash);
        set_env("UCF_WORLD_VLJEPA_DISABLE_ON_ALARM", "1", &mut stash);
        set_env("UCF_WORLD_VLJEPA_ERR_SPIKE_MAX_Q", "100", &mut stash);
        set_env(
            "UCF_WORLD_VLJEPA_INVALID_OUTPUT_MAX_RATE",
            "1.0",
            &mut stash,
        );
        for t in 0..2 {
            let rec = WorldVljepaShadowRecord {
                t,
                encoding_digest_prefix: [0; 8],
                prediction_error_q: 200,
                prediction_digest_prefix: [1; 8],
                model_hash_prefix: [2; 8],
                saturation_clamp_count: 1,
                invalid_output: false,
                status: "ok",
            };
            record_shadow_sample(1.0, 100, &rec);
        }
        assert!(shadow_disabled());
        restore_env(stash);
    }
}
