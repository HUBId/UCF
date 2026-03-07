#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use ucf_client::{Cli, ClientError, Command, Endpoint};
use ucf_ops::{runs_list, AlertRecordV1, AlertsReportV1, DriftReportV1, RunRegistryEntry};

const MAX_ALERTS: usize = 20;
const MAX_DRIFT_STAGES: usize = 20;
const MAX_RUNS: usize = 20;
const HEALTH_RETRIES: usize = 3;

#[derive(Debug, Clone)]
pub struct ConsoleConfig {
    pub workdir: PathBuf,
    pub endpoint: Endpoint,
    pub token: String,
    pub alerts_path: PathBuf,
    pub drift_path: PathBuf,
    pub export_path: PathBuf,
}

impl Default for ConsoleConfig {
    fn default() -> Self {
        Self {
            workdir: PathBuf::from("."),
            endpoint: Endpoint::default_local(),
            token: std::env::var("UCF_GATEWAY_TOKEN").unwrap_or_default(),
            alerts_path: PathBuf::from("./out/alerts_report.json"),
            drift_path: PathBuf::from("./out/drift_report.json"),
            export_path: PathBuf::from("./out/console_export.json"),
        }
    }
}

#[derive(Debug, Error)]
pub enum ConsoleError {
    #[error("client: {0}")]
    Client(#[from] ClientError),
    #[error("ops: {0}")]
    Ops(#[from] ucf_ops::OpsError),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverviewSnapshot {
    pub status: String,
    pub run_id: String,
    pub strict_mode: bool,
    pub policy_graph_digest_prefix: String,
    pub manifest_digest_prefix: String,
    pub drift_status: String,
    pub emergency_active: bool,
    pub last_tick_age_ms: u64,
    pub active_slots_summary: String,
    pub drift_alarms: u32,
    pub violations: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftStageSnapshot {
    pub stage_id: String,
    pub status: String,
    pub active_alarms: Vec<String>,
    pub recommended_action: String,
    pub windows_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsoleSnapshot {
    pub generated_at_ms: u128,
    pub overview: OverviewSnapshot,
    pub alerts_active: Vec<AlertRecordV1>,
    pub alerts_last_triggers: Vec<AlertRecordV1>,
    pub drift: Vec<DriftStageSnapshot>,
    pub runs: Vec<RunRegistryEntry>,
}

#[derive(Debug, Clone, Copy)]
pub enum ViewTab {
    Overview,
    Alerts,
    Drift,
    Runs,
}

#[derive(Debug, Deserialize)]
struct HealthJson {
    status: i32,
    run_id: String,
    strict_mode: bool,
    policy_graph_digest_prefix: String,
    manifest_digest_prefix: String,
    drift_status: i32,
    emergency_active: bool,
    last_tick_age_ms: u64,
    active_slots_summary: String,
    recent_alarm_counts: RecentAlarmCounts,
}

#[derive(Debug, Deserialize)]
struct RecentAlarmCounts {
    drift_alarms: u32,
    violations: u32,
}

pub fn load_snapshot(cfg: &ConsoleConfig) -> Result<ConsoleSnapshot, ConsoleError> {
    let overview = load_overview(cfg)?;
    let alerts = load_alerts(&cfg.alerts_path)?;
    let drift = load_drift(&cfg.drift_path)?;
    let mut runs = runs_list(&cfg.workdir, MAX_RUNS)?;
    runs.sort_by(|a, b| {
        a.started_at_tick
            .cmp(&b.started_at_tick)
            .then_with(|| a.run_id.cmp(&b.run_id))
    });

    Ok(ConsoleSnapshot {
        generated_at_ms: now_ms(),
        overview,
        alerts_active: alerts.active_alerts,
        alerts_last_triggers: alerts.last_triggers,
        drift,
        runs,
    })
}

fn load_overview(cfg: &ConsoleConfig) -> Result<OverviewSnapshot, ConsoleError> {
    let mut last_error = None;
    for attempt in 0..HEALTH_RETRIES {
        let cli = Cli {
            endpoint: cfg.endpoint.clone(),
            auth: cfg.token.clone(),
            command: Command::Health,
        };
        match ucf_client::run(cli) {
            Ok(raw) => {
                let h: HealthJson = serde_json::from_str(&raw)?;
                return Ok(OverviewSnapshot {
                    status: health_status_str(h.status).to_string(),
                    run_id: h.run_id,
                    strict_mode: h.strict_mode,
                    policy_graph_digest_prefix: h.policy_graph_digest_prefix,
                    manifest_digest_prefix: h.manifest_digest_prefix,
                    drift_status: drift_status_str(h.drift_status).to_string(),
                    emergency_active: h.emergency_active,
                    last_tick_age_ms: h.last_tick_age_ms,
                    active_slots_summary: h.active_slots_summary,
                    drift_alarms: h.recent_alarm_counts.drift_alarms,
                    violations: h.recent_alarm_counts.violations,
                });
            }
            Err(err) => {
                last_error = Some(err);
                if attempt + 1 < HEALTH_RETRIES {
                    let sleep_ms = 100_u64.saturating_mul(1_u64 << attempt);
                    thread::sleep(Duration::from_millis(sleep_ms));
                }
            }
        }
    }
    local_overview(cfg, last_error)
}

fn local_overview(
    cfg: &ConsoleConfig,
    last_error: Option<ClientError>,
) -> Result<OverviewSnapshot, ConsoleError> {
    let runs = runs_list(&cfg.workdir, 1)?;
    let run = runs.last();
    let (drift_status, drift_alarms) = if cfg.drift_path.exists() {
        let drift = serde_json::from_str::<DriftReportV1>(&fs::read_to_string(&cfg.drift_path)?)?;
        let status = if drift.alarms.is_empty() {
            "OK"
        } else {
            "DEGRADED"
        };
        (status.to_string(), drift.alarms.len() as u32)
    } else {
        ("UNKNOWN".to_string(), 0)
    };

    let mut status = if drift_alarms == 0 { "OK" } else { "DEGRADED" }.to_string();
    if last_error.is_some() {
        status = "DEGRADED".to_string();
    }
    Ok(OverviewSnapshot {
        status,
        run_id: run
            .map(|r| r.run_id.clone())
            .unwrap_or_else(|| "unknown".to_string()),
        strict_mode: false,
        policy_graph_digest_prefix: run
            .map(|r| r.policy_bundle_hash_prefix.clone())
            .unwrap_or_else(|| "n/a".to_string()),
        manifest_digest_prefix: run
            .map(|r| r.pack_digest_prefix.clone())
            .unwrap_or_else(|| "n/a".to_string()),
        drift_status,
        emergency_active: false,
        last_tick_age_ms: 0,
        active_slots_summary: "local-artifacts".to_string(),
        drift_alarms,
        violations: 0,
    })
}

fn load_alerts(path: &Path) -> Result<AlertsReportV1, ConsoleError> {
    if !path.exists() {
        return Ok(AlertsReportV1 {
            schema_version: 1,
            run_id: String::new(),
            active_alerts: Vec::new(),
            last_triggers: Vec::new(),
            suggested_commands: Vec::new(),
            summary_text: "alerts report missing".to_string(),
            report_digest: String::new(),
        });
    }
    let mut report = serde_json::from_str::<AlertsReportV1>(&fs::read_to_string(path)?)?;
    report
        .active_alerts
        .sort_by(|a, b| a.alert_id.cmp(&b.alert_id));
    report.active_alerts.truncate(MAX_ALERTS);
    report.last_triggers.sort_by(|a, b| {
        a.triggered_at_t
            .cmp(&b.triggered_at_t)
            .then_with(|| a.alert_id.cmp(&b.alert_id))
    });
    if report.last_triggers.len() > MAX_ALERTS {
        report.last_triggers = report
            .last_triggers
            .split_off(report.last_triggers.len() - MAX_ALERTS);
    }
    Ok(report)
}

fn load_drift(path: &Path) -> Result<Vec<DriftStageSnapshot>, ConsoleError> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut report = serde_json::from_str::<DriftReportV1>(&fs::read_to_string(path)?)?;
    report
        .slot_reports
        .sort_by(|a, b| a.slot_id.cmp(&b.slot_id));
    let mut items = report
        .slot_reports
        .into_iter()
        .map(|s| DriftStageSnapshot {
            stage_id: s.slot_id,
            status: s.status,
            active_alarms: s.active_alarms,
            recommended_action: s.recommended_actions.join(","),
            windows_count: s.windows.len(),
        })
        .collect::<Vec<_>>();
    items.truncate(MAX_DRIFT_STAGES);
    Ok(items)
}

pub fn export_view(
    snapshot: &ConsoleSnapshot,
    tab: ViewTab,
    out: &Path,
) -> Result<(), ConsoleError> {
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    let value = match tab {
        ViewTab::Overview => serde_json::to_value(&snapshot.overview)?,
        ViewTab::Alerts => serde_json::json!({
            "active": snapshot.alerts_active,
            "last_triggers": snapshot.alerts_last_triggers
        }),
        ViewTab::Drift => serde_json::to_value(&snapshot.drift)?,
        ViewTab::Runs => serde_json::to_value(&snapshot.runs)?,
    };
    fs::write(out, serde_json::to_string_pretty(&value)?)?;
    Ok(())
}

fn health_status_str(v: i32) -> &'static str {
    match v {
        1 => "OK",
        2 => "DEGRADED",
        3 => "FAIL",
        _ => "UNKNOWN",
    }
}

fn drift_status_str(v: i32) -> &'static str {
    match v {
        2 => "OK",
        3 => "DEGRADED",
        1 => "UNKNOWN",
        _ => "UNSPECIFIED",
    }
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn health_status_mapping_is_stable() {
        assert_eq!(health_status_str(1), "OK");
        assert_eq!(health_status_str(2), "DEGRADED");
        assert_eq!(health_status_str(3), "FAIL");
        assert_eq!(health_status_str(77), "UNKNOWN");
    }

    #[test]
    fn alerts_are_sorted_and_bounded() {
        let dir = tempdir().expect("tmp");
        let path = dir.path().join("alerts.json");
        let mut active = Vec::new();
        let mut triggers = Vec::new();
        for i in (0..30).rev() {
            active.push(AlertRecordV1 {
                schema_version: 1,
                alert_id: format!("a-{i:02}"),
                severity: "warn".to_string(),
                triggered_at_t: i,
                rule_id: "r".to_string(),
                observed_count: 1,
                window_start_t: i,
                window_end_t: i,
                evidence_digests: Vec::new(),
                remediation_codes: Vec::new(),
            });
            triggers.push(active.last().expect("item").clone());
        }
        let report = AlertsReportV1 {
            schema_version: 1,
            run_id: "r1".to_string(),
            active_alerts: active,
            last_triggers: triggers,
            suggested_commands: Vec::new(),
            summary_text: String::new(),
            report_digest: String::new(),
        };
        fs::write(&path, serde_json::to_string(&report).expect("json")).expect("write");
        let got = load_alerts(&path).expect("load");
        assert_eq!(got.active_alerts.len(), MAX_ALERTS);
        assert_eq!(got.active_alerts[0].alert_id, "a-00");
        assert_eq!(got.last_triggers.len(), MAX_ALERTS);
        assert_eq!(got.last_triggers.first().expect("first").triggered_at_t, 10);
    }

    #[test]
    fn drift_is_sorted_and_bounded() {
        let dir = tempdir().expect("tmp");
        let path = dir.path().join("drift.json");
        let mut stages = Vec::new();
        for i in (0..25).rev() {
            stages.push(serde_json::json!({
                "slot_id": format!("s-{i:02}"),
                "status": "OK",
                "active_alarms": [],
                "recommended_actions": ["none"],
                "windows": []
            }));
        }
        let report = serde_json::json!({
            "run_id": "r1",
            "status": "PASS",
            "windows_limit": 4,
            "slot_reports": stages,
            "alarms": [],
            "operator_summary": "",
            "report_digest": ""
        });
        fs::write(&path, serde_json::to_string(&report).expect("json")).expect("write");
        let got = load_drift(&path).expect("load");
        assert_eq!(got.len(), MAX_DRIFT_STAGES);
        assert_eq!(got[0].stage_id, "s-00");
    }
}
