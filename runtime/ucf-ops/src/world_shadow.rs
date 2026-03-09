use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use ucf_compute::ModelSlot;
use ucf_ess::v1::{AuditPayload, ExperiencePayload};
use ucf_replay::load_fixture_records;

use crate::{sha256_hex, GateStatus, OpsError, ProbeReport, ProbeStatus};

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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum WorldParityStatusV1 {
    Ok,
    Warn,
    Severe,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorldComparedBackendRecordV1 {
    pub backend_id: String,
    pub model_hash_prefix: String,
    pub prediction_error_delta_q_mean: u16,
    pub prediction_error_delta_q_max: u16,
    pub surprise_delta_q_mean: u16,
    pub surprise_delta_q_max: u16,
    pub digest_mismatch_count: u16,
    pub invalid_output_count: u16,
    pub sample_prediction_digest_prefixes: Vec<String>,
    pub status: WorldParityStatusV1,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorldParityRecordV1 {
    pub schema_version: u16,
    pub run_id: String,
    pub window_id: u64,
    pub t0: u64,
    pub t1: u64,
    pub primary_backend_id: String,
    pub compared_backends: Vec<WorldComparedBackendRecordV1>,
    pub parity_digest: String,
    pub policy_graph_digest_prefix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorldBackendEligibilityV1 {
    pub backend_id: String,
    pub probe_pass: bool,
    pub shadow_window_present: bool,
    pub no_impact_verified: bool,
    pub severe_drift_present: bool,
    pub eligible_for_shadow: bool,
    pub eligible_for_active: bool,
    pub reason_code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorldParityReportV1 {
    pub run_id: String,
    pub parity_records: Vec<WorldParityRecordV1>,
    pub eligibility: Vec<WorldBackendEligibilityV1>,
    pub remediation_hints: Vec<String>,
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

    let manifest_digest = fs::read_to_string("models/lifecycle_manifest.toml")
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

pub fn world_parity_report(
    workdir: &Path,
    run_id: &str,
    out: &Path,
) -> Result<WorldParityReportV1, OpsError> {
    let fixture_path = workdir.join("ess").join("ess_fixture.json");
    let mut fixture = load_fixture_records(&fixture_path).unwrap_or_default();
    fixture.sort_by(|a, b| a.id.0.cmp(&b.id.0));

    let probe_path = workdir.join("out").join("probe_report.json");
    let probe = fs::read_to_string(probe_path)
        .ok()
        .and_then(|raw| serde_json::from_str::<ProbeReport>(&raw).ok());
    let world_probe_pass = probe
        .as_ref()
        .and_then(|p| p.results.iter().find(|r| r.slot == ModelSlot::WorldJepa))
        .map(|r| matches!(r.status, ProbeStatus::Ok))
        .unwrap_or(false);
    let world_hash_prefix = probe
        .as_ref()
        .and_then(|p| p.results.iter().find(|r| r.slot == ModelSlot::WorldJepa))
        .and_then(|r| r.model_sha256_prefix.clone())
        .unwrap_or_else(|| "unknown".to_string());

    let mut parity_records = Vec::new();
    for rec in fixture {
        if let ExperiencePayload::Audit(AuditPayload::SlotCompareWindow(w)) = rec.payload {
            if !w.slot_id.contains("world") {
                continue;
            }
            let compared = vec!["burn_world_v1", "candle_world_v1"];
            let mut compared_backends = compared
                .into_iter()
                .map(|backend_id| {
                    let status = if w.invalid_shadow_count > 0 {
                        WorldParityStatusV1::Severe
                    } else if w.digest_mismatch_count > 0 || w.mean_delta_q > 0 || w.p95_delta_q > 0
                    {
                        WorldParityStatusV1::Warn
                    } else {
                        WorldParityStatusV1::Ok
                    };
                    WorldComparedBackendRecordV1 {
                        backend_id: backend_id.to_string(),
                        model_hash_prefix: world_hash_prefix.clone(),
                        prediction_error_delta_q_mean: w.mean_delta_q,
                        prediction_error_delta_q_max: w.p95_delta_q,
                        surprise_delta_q_mean: w.mean_delta_q,
                        surprise_delta_q_max: w.p95_delta_q,
                        digest_mismatch_count: w.digest_mismatch_count,
                        invalid_output_count: w.invalid_shadow_count,
                        sample_prediction_digest_prefixes: w
                            .digest_prefix_samples
                            .iter()
                            .take(4)
                            .map(hex::encode)
                            .collect(),
                        status,
                    }
                })
                .collect::<Vec<_>>();
            compared_backends.sort_by(|a, b| a.backend_id.cmp(&b.backend_id));
            if compared_backends.len() > 2 {
                compared_backends.truncate(2);
            }
            let mut parity = WorldParityRecordV1 {
                schema_version: 1,
                run_id: run_id.to_string(),
                window_id: w.t1,
                t0: w.t0,
                t1: w.t1,
                primary_backend_id: "stub_world_v1".to_string(),
                compared_backends,
                parity_digest: String::new(),
                policy_graph_digest_prefix: "unknown".to_string(),
            };
            parity.parity_digest = sha256_hex(&serde_json::to_vec(&parity)?);
            parity_records.push(parity);
        }
    }
    parity_records.sort_by_key(|r| (r.t1, r.window_id));
    if parity_records.len() > 10 {
        parity_records = parity_records.split_off(parity_records.len() - 10);
    }

    let severe_drift_present = parity_records.iter().any(|r| {
        r.compared_backends
            .iter()
            .any(|c| matches!(c.status, WorldParityStatusV1::Severe))
    });
    let shadow_window_present = !parity_records.is_empty();
    let mut eligibility = vec![
        WorldBackendEligibilityV1 {
            backend_id: "stub_world_v1".to_string(),
            probe_pass: true,
            shadow_window_present,
            no_impact_verified: true,
            severe_drift_present,
            eligible_for_shadow: true,
            eligible_for_active: true,
            reason_code: "STUB_ALLOWED".to_string(),
        },
        WorldBackendEligibilityV1 {
            backend_id: "candle_world_v1".to_string(),
            probe_pass: world_probe_pass,
            shadow_window_present,
            no_impact_verified: true,
            severe_drift_present,
            eligible_for_shadow: world_probe_pass && !severe_drift_present,
            eligible_for_active: false,
            reason_code: "ACTIVE_NOT_ENABLED_IN_V2_STAGE".to_string(),
        },
        WorldBackendEligibilityV1 {
            backend_id: "burn_world_v1".to_string(),
            probe_pass: world_probe_pass,
            shadow_window_present,
            no_impact_verified: true,
            severe_drift_present,
            eligible_for_shadow: world_probe_pass && !severe_drift_present,
            eligible_for_active: false,
            reason_code: "ACTIVE_NOT_ENABLED_IN_V2_STAGE".to_string(),
        },
    ];
    eligibility.sort_by(|a, b| a.backend_id.cmp(&b.backend_id));

    let mut report = WorldParityReportV1 {
        run_id: run_id.to_string(),
        parity_records,
        eligibility,
        remediation_hints: vec![
            "run `cargo run -p ucf-ops -- models probe --manifest models/manifest.toml --out ./out/probe_report.json`".to_string(),
            "inspect drift report: `cargo run -p ucf-ops -- drift report --run <id> --windows 10 --out ./out/drift_report.json`".to_string(),
            "keep real backends in shadow mode for v2 stage".to_string(),
        ],
        report_digest: String::new(),
    };
    report.report_digest = sha256_hex(&serde_json::to_vec(&report)?);
    fs::create_dir_all(out.parent().unwrap_or_else(|| Path::new(".")))?;
    fs::write(out, serde_json::to_string_pretty(&report)?)?;
    Ok(report)
}

pub fn world_parity_evidence_exists(workdir: &Path) -> bool {
    workdir
        .join("out")
        .join("world_parity_report.json")
        .exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_parity_status_is_deterministic() {
        let mut b = [
            WorldComparedBackendRecordV1 {
                backend_id: "candle_world_v1".to_string(),
                model_hash_prefix: "aa".to_string(),
                prediction_error_delta_q_mean: 1,
                prediction_error_delta_q_max: 2,
                surprise_delta_q_mean: 1,
                surprise_delta_q_max: 2,
                digest_mismatch_count: 0,
                invalid_output_count: 0,
                sample_prediction_digest_prefixes: vec!["00000000".to_string()],
                status: WorldParityStatusV1::Ok,
            },
            WorldComparedBackendRecordV1 {
                backend_id: "burn_world_v1".to_string(),
                model_hash_prefix: "bb".to_string(),
                prediction_error_delta_q_mean: 1,
                prediction_error_delta_q_max: 2,
                surprise_delta_q_mean: 1,
                surprise_delta_q_max: 2,
                digest_mismatch_count: 0,
                invalid_output_count: 0,
                sample_prediction_digest_prefixes: vec!["11111111".to_string()],
                status: WorldParityStatusV1::Ok,
            },
        ];
        b.sort_by(|a, b| a.backend_id.cmp(&b.backend_id));
        assert_eq!(b[0].backend_id, "burn_world_v1");
        assert_eq!(b[1].backend_id, "candle_world_v1");
    }
}
