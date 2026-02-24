use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{sha256_hex, GateStatus, OpsError};

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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorldDriftAlarmRecord {
    pub window_id: u64,
    pub reason_codes: Vec<String>,
    pub action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorldShadowReport {
    pub run_id: String,
    pub status: GateStatus,
    pub window_count: usize,
    pub ticks_total: u64,
    pub windows: Vec<WorldShadowWindowStats>,
    pub drift_alarms: Vec<WorldDriftAlarmRecord>,
    pub model_hashes_digest: Option<String>,
    pub manifest_digest: Option<String>,
    pub report_digest: String,
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>, OpsError> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let content = fs::read_to_string(path)?;
    content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<Vec<_>, _>>()
        .map_err(OpsError::from)
}

pub fn world_shadow_report(
    workdir: &Path,
    run_id: &str,
    windows: usize,
    out: &Path,
) -> Result<WorldShadowReport, OpsError> {
    let base = workdir.join("reports").join("world_vljepa");
    let window_path = base.join(format!("{}_windows.jsonl", run_id));
    let alarm_path = base.join(format!("{}_alarms.jsonl", run_id));
    let mut win = read_jsonl::<WorldShadowWindowStats>(&window_path)?;
    let alarms = read_jsonl::<WorldDriftAlarmRecord>(&alarm_path)?;
    win.sort_by_key(|w| w.window_id);
    if windows > 0 && win.len() > windows {
        win = win.split_off(win.len() - windows);
    }
    let ticks_total = win.iter().map(|w| w.ticks as u64).sum();

    let run_meta_path = workdir.join("runs").join(format!("{run_id}.json"));
    let run_meta = fs::read_to_string(run_meta_path)
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok());
    let model_hashes_digest = run_meta
        .as_ref()
        .and_then(|v| v.get("model_hashes_digest"))
        .and_then(|v| v.as_str())
        .map(ToString::to_string);

    let manifest_digest = fs::read_to_string("models/MANIFEST.toml")
        .ok()
        .and_then(|raw| {
            raw.lines()
                .find(|l| l.trim_start().starts_with("manifest_digest"))
                .and_then(|l| l.split('=').nth(1))
                .map(|v| v.trim().trim_matches('"').to_string())
        });

    let status = if alarms.is_empty() && !win.is_empty() {
        GateStatus::Pass
    } else {
        GateStatus::Fail
    };

    let mut report = WorldShadowReport {
        run_id: run_id.to_string(),
        status,
        window_count: win.len(),
        ticks_total,
        windows: win,
        drift_alarms: alarms,
        model_hashes_digest,
        manifest_digest,
        report_digest: String::new(),
    };
    report.report_digest = sha256_hex(&serde_json::to_vec(&report)?);
    fs::create_dir_all(
        out.parent()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(".")),
    )?;
    fs::write(out, serde_json::to_string_pretty(&report)?)?;
    Ok(report)
}
