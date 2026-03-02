use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use ucf_policy::policy_packs::{load_and_merge_policy_graph, DriftActionV1, DriftBudgetEntryV1};

use crate::world_shadow::{WorldDriftAlarmRecord, WorldShadowWindowStats};
use crate::{sha256_hex, GateStatus, OpsError};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DriftWindowStatsV1 {
    pub window_id: u64,
    pub latency_p95_ms_q: u32,
    pub invalid_rate_q: u16,
    pub timeout_rate_q: u16,
    pub delta_scalar_q: u16,
    pub digest_mismatch_rate_q: Option<u16>,
    pub evidence_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DriftAlarmRecordV1 {
    pub alarm_id: String,
    pub stage_id: String,
    pub window_id: u64,
    pub breached_fields: Vec<String>,
    pub observed: BTreeMap<String, u32>,
    pub action: String,
    pub reason_code: String,
    pub evidence_digests: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DriftStageReportV1 {
    pub stage_id: String,
    pub status: String,
    pub active_alarms: Vec<String>,
    pub recommended_action: String,
    pub windows: Vec<DriftWindowStatsV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DriftReportV1 {
    pub run_id: String,
    pub status: GateStatus,
    pub windows_limit: usize,
    pub stage_reports: Vec<DriftStageReportV1>,
    pub alarms: Vec<DriftAlarmRecordV1>,
    pub operator_summary: String,
    pub report_digest: String,
}

fn q01(v: f32) -> u16 {
    ((v.clamp(0.0, 1.0) * 10_000.0).round() as u32).min(u16::MAX as u32) as u16
}

fn action_str(action: DriftActionV1) -> &'static str {
    match action {
        DriftActionV1::DisableShadow => "disable_shadow",
        DriftActionV1::ForceToy => "force_toy",
        DriftActionV1::RecommendRollback => "recommend_rollback",
    }
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

pub fn drift_report(
    workdir: &Path,
    run_id: &str,
    windows: usize,
    out: &Path,
) -> Result<DriftReportV1, OpsError> {
    let overlay = std::env::var("UCF_POLICY_OVERLAY").ok();
    let overlay_path = overlay
        .as_deref()
        .map(|name| PathBuf::from("policies/packs/overlays").join(name));
    let (graph, _) =
        load_and_merge_policy_graph(Path::new("policies/packs/base_v1"), overlay_path.as_deref())?;

    let world_windows_path = workdir
        .join("reports")
        .join("world_vljepa")
        .join(format!("{}_windows.jsonl", run_id));
    let world_alarms_path = workdir
        .join("reports")
        .join("world_vljepa")
        .join(format!("{}_alarms.jsonl", run_id));
    let mut world_windows = read_jsonl::<WorldShadowWindowStats>(&world_windows_path)?;
    let _world_alarm_records = read_jsonl::<WorldDriftAlarmRecord>(&world_alarms_path)?;
    world_windows.sort_by_key(|w| w.window_id);

    let mut stage_windows: BTreeMap<String, Vec<DriftWindowStatsV1>> = BTreeMap::new();
    let w = world_windows
        .into_iter()
        .map(|it| DriftWindowStatsV1 {
            window_id: it.window_id,
            latency_p95_ms_q: it.latency_p95_ms.round().max(0.0) as u32,
            invalid_rate_q: q01(it.invalid_rate),
            timeout_rate_q: 0,
            delta_scalar_q: q01((it.error_delta_p95_q as f32 / 65_535.0).clamp(0.0, 1.0)),
            digest_mismatch_rate_q: None,
            evidence_digest: sha256_hex(
                format!(
                    "{}:{}:{}:{}",
                    it.window_id, it.end_t, it.error_delta_p95_q, it.latency_p95_ms
                )
                .as_bytes(),
            ),
        })
        .collect::<Vec<_>>();
    stage_windows.insert("world_vljepa".to_string(), w);

    let ssm_path = workdir.join("out/ssm_opt_parity.json");
    if let Ok(raw) = fs::read_to_string(&ssm_path) {
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&raw) {
            let drift = val
                .get("drift_alarm_rate")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0) as f32;
            let mismatch = val
                .get("digest_mismatch_rate")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0) as f32;
            stage_windows.insert(
                "ssm_opt".to_string(),
                vec![DriftWindowStatsV1 {
                    window_id: 0,
                    latency_p95_ms_q: 0,
                    invalid_rate_q: 0,
                    timeout_rate_q: 0,
                    delta_scalar_q: q01(drift),
                    digest_mismatch_rate_q: Some(q01(mismatch)),
                    evidence_digest: sha256_hex(raw.as_bytes()),
                }],
            );
        }
    }

    let gpu_path = workdir.join("out/gpu_parity_report.json");
    if let Ok(raw) = fs::read_to_string(&gpu_path) {
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&raw) {
            let drift = val
                .get("drift_alarm_rate")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0) as f32;
            let mismatch = val
                .get("digest_mismatch_rate")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0) as f32;
            stage_windows.insert(
                "gpu_lane".to_string(),
                vec![DriftWindowStatsV1 {
                    window_id: 0,
                    latency_p95_ms_q: 0,
                    invalid_rate_q: 0,
                    timeout_rate_q: 0,
                    delta_scalar_q: q01(drift),
                    digest_mismatch_rate_q: Some(q01(mismatch)),
                    evidence_digest: sha256_hex(raw.as_bytes()),
                }],
            );
        }
    }

    let mut alarms = Vec::new();
    let mut stage_reports = Vec::new();
    for entry in &graph.drift_budget.entries {
        let mut wins = stage_windows.remove(&entry.stage_id).unwrap_or_default();
        wins.sort_by_key(|x| x.window_id);
        if windows > 0 && wins.len() > windows {
            wins = wins.split_off(wins.len() - windows);
        }
        let evaluated = evaluate_stage(entry, &wins);
        alarms.extend(evaluated.0.clone());
        stage_reports.push(DriftStageReportV1 {
            stage_id: entry.stage_id.clone(),
            status: if evaluated.0.is_empty() {
                "OK".to_string()
            } else {
                "DEGRADED".to_string()
            },
            active_alarms: evaluated.0.iter().map(|a| a.alarm_id.clone()).collect(),
            recommended_action: evaluated.1,
            windows: wins,
        });
    }
    stage_reports.sort_by(|a, b| a.stage_id.cmp(&b.stage_id));
    alarms.sort_by(|a, b| a.alarm_id.cmp(&b.alarm_id));

    let status = if alarms.is_empty() {
        GateStatus::Pass
    } else {
        GateStatus::Fail
    };
    let operator_summary = if alarms.is_empty() {
        "All drift budgets within envelope. Keep current restrictions.".to_string()
    } else {
        "Drift alarms detected: apply tightening actions only (disable shadow / force toy / recommend rollback).".to_string()
    };
    let mut report = DriftReportV1 {
        run_id: run_id.to_string(),
        status,
        windows_limit: windows,
        stage_reports,
        alarms,
        operator_summary,
        report_digest: String::new(),
    };
    report.report_digest = sha256_hex(&serde_json::to_vec(&report)?);
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_string_pretty(&report)?)?;
    Ok(report)
}

fn evaluate_stage(
    entry: &DriftBudgetEntryV1,
    windows: &[DriftWindowStatsV1],
) -> (Vec<DriftAlarmRecordV1>, String) {
    let mut alarms = Vec::new();
    for w in windows {
        let mut breached = Vec::new();
        let mut observed = BTreeMap::new();
        if w.latency_p95_ms_q > entry.latency_p95_max_ms {
            breached.push("latency_p95_max_ms".to_string());
            observed.insert("latency_p95_ms_q".to_string(), w.latency_p95_ms_q);
        }
        if w.invalid_rate_q > entry.invalid_rate_max_q {
            breached.push("invalid_rate_max_q".to_string());
            observed.insert("invalid_rate_q".to_string(), u32::from(w.invalid_rate_q));
        }
        if w.timeout_rate_q > entry.timeout_rate_max_q {
            breached.push("timeout_rate_max_q".to_string());
            observed.insert("timeout_rate_q".to_string(), u32::from(w.timeout_rate_q));
        }
        if w.delta_scalar_q > entry.delta_scalar_max_q {
            breached.push("delta_scalar_max_q".to_string());
            observed.insert("delta_scalar_q".to_string(), u32::from(w.delta_scalar_q));
        }
        if let (Some(obs), Some(max)) = (w.digest_mismatch_rate_q, entry.digest_mismatch_rate_max_q)
        {
            if obs > max {
                breached.push("digest_mismatch_rate_max_q".to_string());
                observed.insert("digest_mismatch_rate_q".to_string(), u32::from(obs));
            }
        }
        if !breached.is_empty() {
            let reason_code = breached.join("+");
            alarms.push(DriftAlarmRecordV1 {
                alarm_id: format!("{}:{}", entry.stage_id, w.window_id),
                stage_id: entry.stage_id.clone(),
                window_id: w.window_id,
                breached_fields: breached,
                observed,
                action: action_str(entry.action_on_breach).to_string(),
                reason_code,
                evidence_digests: vec![w.evidence_digest.clone()],
            });
        }
    }
    let recommendation = if alarms.is_empty() {
        "none".to_string()
    } else {
        action_str(entry.action_on_breach).to_string()
    };
    (alarms, recommendation)
}

pub fn drift_status_map(report: &DriftReportV1) -> BTreeMap<String, String> {
    report
        .stage_reports
        .iter()
        .map(|stage| (stage.stage_id.clone(), stage.status.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_policy::policy_packs::DriftActionV1;

    #[test]
    fn evaluator_is_deterministic() {
        let entry = DriftBudgetEntryV1 {
            stage_id: "world_vljepa".to_string(),
            window_size: 2,
            latency_p95_max_ms: 5,
            invalid_rate_max_q: 100,
            timeout_rate_max_q: 100,
            delta_scalar_max_q: 100,
            digest_mismatch_rate_max_q: None,
            action_on_breach: DriftActionV1::DisableShadow,
        };
        let windows = vec![DriftWindowStatsV1 {
            window_id: 1,
            latency_p95_ms_q: 10,
            invalid_rate_q: 50,
            timeout_rate_q: 0,
            delta_scalar_q: 50,
            digest_mismatch_rate_q: None,
            evidence_digest: "abc".to_string(),
        }];
        let a = evaluate_stage(&entry, &windows);
        let b = evaluate_stage(&entry, &windows);
        assert_eq!(a, b);
        assert_eq!(a.0.len(), 1);
    }
}
