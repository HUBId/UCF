use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use ucf_ess::v1::{AuditPayload, ExperiencePayload, ExperienceRecord};
use ucf_policy::policy_packs::{load_and_merge_policy_graph, DriftActionV1, DriftBudgetEntryV1};
use ucf_replay::load_fixture_records;

use crate::world_shadow::{WorldDriftAlarmRecord, WorldShadowWindowStats};
use crate::{derive_drift_inputs_from_slot_compare, sha256_hex, GateStatus, OpsError};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DriftWindowStatsV1 {
    pub window_id: u64,
    pub latency_p95_ms_q: u32,
    pub invalid_rate_q: u16,
    pub scalar_deltas_q: BTreeMap<String, u16>,
    pub digest_mismatch_rate_q: Option<u16>,
    pub evidence_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DriftAlarmRecordV1 {
    pub alarm_id: String,
    pub slot_id: String,
    pub window_id: u64,
    pub breached_fields: Vec<String>,
    pub observed: BTreeMap<String, u32>,
    pub severity: String,
    pub action_taken: String,
    pub reason_code: String,
    pub evidence_digests: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DriftSlotReportV1 {
    pub slot_id: String,
    pub status: String,
    pub active_alarms: Vec<String>,
    pub recommended_actions: Vec<String>,
    pub windows: Vec<DriftWindowStatsV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DriftReportV1 {
    pub run_id: String,
    pub status: GateStatus,
    pub windows_limit: usize,
    pub slot_reports: Vec<DriftSlotReportV1>,
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
        DriftActionV1::None => "none",
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

    let mut slot_windows: BTreeMap<String, Vec<DriftWindowStatsV1>> = BTreeMap::new();

    let fixture_path = workdir.join("ess").join("ess_fixture.json");
    let mut compare_records: Vec<ExperienceRecord> =
        load_fixture_records(&fixture_path).unwrap_or_default();
    compare_records.sort_by(|a, b| a.id.0.cmp(&b.id.0));
    for record in compare_records {
        if let ExperiencePayload::Audit(AuditPayload::SlotCompareWindow(w)) = record.payload {
            if w.sample_count == 0 {
                continue;
            }
            let drift_input = derive_drift_inputs_from_slot_compare(&w.slot_id, run_id, &w, 0);
            slot_windows
                .entry(w.slot_id.clone())
                .or_default()
                .push(DriftWindowStatsV1 {
                    window_id: drift_input.window_id,
                    latency_p95_ms_q: drift_input.latency_p95_ms_q,
                    invalid_rate_q: drift_input.invalid_rate_q,
                    scalar_deltas_q: drift_input.scalar_deltas_q,
                    digest_mismatch_rate_q: Some(drift_input.digest_mismatch_rate_q),
                    evidence_digest: sha256_hex(
                        format!(
                            "{}:{}:{}:{}:{}",
                            w.slot_id, w.t0, w.t1, w.primary_p95_q, w.shadow_p95_q
                        )
                        .as_bytes(),
                    ),
                });
        }
    }

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

    let w = world_windows
        .into_iter()
        .map(|it| DriftWindowStatsV1 {
            window_id: it.window_id,
            latency_p95_ms_q: it.latency_p95_ms.round().max(0.0) as u32,
            invalid_rate_q: q01(it.invalid_rate),
            scalar_deltas_q: BTreeMap::from([(
                "error_delta_p95_q".to_string(),
                q01((it.error_delta_p95_q as f32 / 65_535.0).clamp(0.0, 1.0)),
            )]),
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
    slot_windows.insert("world_vljepa".to_string(), w);

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
            slot_windows.insert(
                "ssm_opt".to_string(),
                vec![DriftWindowStatsV1 {
                    window_id: 0,
                    latency_p95_ms_q: 0,
                    invalid_rate_q: 0,
                    scalar_deltas_q: BTreeMap::from([(
                        "drift_alarm_rate_q".to_string(),
                        q01(drift),
                    )]),
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
            slot_windows.insert(
                "gpu_lane".to_string(),
                vec![DriftWindowStatsV1 {
                    window_id: 0,
                    latency_p95_ms_q: 0,
                    invalid_rate_q: 0,
                    scalar_deltas_q: BTreeMap::from([(
                        "drift_alarm_rate_q".to_string(),
                        q01(drift),
                    )]),
                    digest_mismatch_rate_q: Some(q01(mismatch)),
                    evidence_digest: sha256_hex(raw.as_bytes()),
                }],
            );
        }
    }

    let mut alarms = Vec::new();
    let mut slot_reports = Vec::new();
    for entry in &graph.drift_budget.entries {
        let mut wins = slot_windows.remove(&entry.slot_id).unwrap_or_default();
        wins.sort_by_key(|x| x.window_id);
        let use_windows = if windows == 0 {
            entry.window_size_ticks as usize
        } else {
            windows.min(entry.window_size_ticks as usize)
        };
        if use_windows > 0 && wins.len() > use_windows {
            wins = wins.split_off(wins.len() - use_windows);
        }
        let evaluated = evaluate_slot(entry, &wins);
        alarms.extend(evaluated.0.clone());
        slot_reports.push(DriftSlotReportV1 {
            slot_id: entry.slot_id.clone(),
            status: evaluated.2,
            active_alarms: evaluated.0.iter().map(|a| a.alarm_id.clone()).collect(),
            recommended_actions: evaluated.1,
            windows: wins,
        });
    }
    slot_reports.sort_by(|a, b| a.slot_id.cmp(&b.slot_id));
    alarms.sort_by(|a, b| a.alarm_id.cmp(&b.alarm_id));
    if alarms.len() > 20 {
        alarms = alarms.split_off(alarms.len() - 20);
    }

    let status = if alarms.iter().any(|a| a.severity == "SEVERE") {
        GateStatus::Fail
    } else {
        GateStatus::Pass
    };
    let operator_summary = if alarms.is_empty() {
        "All drift budgets within envelope. Keep current restrictions.".to_string()
    } else {
        "Drift alarms detected: apply tightening actions only (disable shadow / recommend rollback)."
            .to_string()
    };
    let mut report = DriftReportV1 {
        run_id: run_id.to_string(),
        status,
        windows_limit: windows,
        slot_reports,
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

fn evaluate_slot(
    entry: &DriftBudgetEntryV1,
    windows: &[DriftWindowStatsV1],
) -> (Vec<DriftAlarmRecordV1>, Vec<String>, String) {
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
        for (name, limit) in &entry.scalar_delta_max_q {
            let observed_value = w.scalar_deltas_q.get(name).copied().unwrap_or(0);
            if observed_value > *limit {
                let field = format!("scalar_delta_max_q.{name}");
                breached.push(field.clone());
                observed.insert(field, u32::from(observed_value));
            }
        }
        if let (Some(obs), Some(max)) = (w.digest_mismatch_rate_q, entry.digest_mismatch_rate_max_q)
        {
            if obs > max {
                breached.push("digest_mismatch_rate_max_q".to_string());
                observed.insert("digest_mismatch_rate_q".to_string(), u32::from(obs));
            }
        }
        if !breached.is_empty() {
            let severe = breached
                .iter()
                .any(|field| entry.severity.severe_fields.iter().any(|s| s == field));
            let severity = if severe { "SEVERE" } else { "WARN" };
            let action_taken = if severe {
                action_str(entry.action_on_severe).to_string()
            } else {
                "none".to_string()
            };
            let reason_code = breached.join("+");
            alarms.push(DriftAlarmRecordV1 {
                alarm_id: format!("{}:{}", entry.slot_id, w.window_id),
                slot_id: entry.slot_id.clone(),
                window_id: w.window_id,
                breached_fields: breached,
                observed,
                severity: severity.to_string(),
                action_taken,
                reason_code,
                evidence_digests: vec![w.evidence_digest.clone()],
            });
        }
    }
    let status = if alarms.iter().any(|a| a.severity == "SEVERE") {
        "SEVERE"
    } else if alarms.is_empty() {
        "OK"
    } else {
        "WARN"
    };
    let mut recommended_actions = Vec::new();
    if alarms.iter().any(|a| a.severity == "SEVERE")
        && entry.action_on_severe == DriftActionV1::DisableShadow
    {
        recommended_actions.push("disable_shadow".to_string());
    }
    if !alarms.is_empty() {
        recommended_actions.push("recommend_rollback".to_string());
    }
    if recommended_actions.is_empty() {
        recommended_actions.push("none".to_string());
    }
    recommended_actions.sort();
    recommended_actions.dedup();
    (alarms, recommended_actions, status.to_string())
}

pub fn drift_status_map(report: &DriftReportV1) -> BTreeMap<String, String> {
    report
        .slot_reports
        .iter()
        .map(|stage| (stage.slot_id.clone(), stage.status.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_policy::policy_packs::{DriftActionV1, DriftSeverityMapV1};

    #[test]
    fn evaluator_is_deterministic() {
        let entry = DriftBudgetEntryV1 {
            slot_id: "compute".to_string(),
            window_size_ticks: 2,
            scalar_delta_max_q: BTreeMap::from([("risk_p95_q".to_string(), 100)]),
            latency_p95_max_ms: 5,
            invalid_rate_max_q: 100,
            digest_mismatch_rate_max_q: None,
            severity: DriftSeverityMapV1 {
                severe_fields: vec!["scalar_delta_max_q.risk_p95_q".to_string()],
            },
            action_on_severe: DriftActionV1::DisableShadow,
        };
        let windows = vec![DriftWindowStatsV1 {
            window_id: 1,
            latency_p95_ms_q: 10,
            invalid_rate_q: 50,
            scalar_deltas_q: BTreeMap::from([("risk_p95_q".to_string(), 120)]),
            digest_mismatch_rate_q: None,
            evidence_digest: "abc".to_string(),
        }];
        let a = evaluate_slot(&entry, &windows);
        let b = evaluate_slot(&entry, &windows);
        assert_eq!(a, b);
        assert_eq!(a.0.len(), 1);
        assert_eq!(a.0[0].severity, "SEVERE");
        assert_eq!(a.0[0].action_taken, "disable_shadow");
    }
}
