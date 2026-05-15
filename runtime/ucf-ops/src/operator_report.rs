use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::drift::DriftSlotReportV1;
use crate::remediation::merge_canonical_remediations;
use crate::{
    operator_block_from_strict, resolve_strict_evidence, AggregatedEligibilityReportV1,
    AlertsReportV1, BackendEvidenceSnapshotV1, DriftReportV1, GateStatus, OpsError,
    StrictEvidenceContextV1, StrictEvidenceSnapshotV1, StrictEvidenceStatusV1, V0GateOverallStatus,
    V0GateReportV1, V1GateOverallStatus, V1GateReportV1, V2GateOverallStatus, V2GateReportV1,
};

const REMEDIATION_MAX: usize = 12;
const TOP_ALERTS_MAX: usize = 8;
const DRIFT_ALARM_COUNT_CAP: u16 = 255;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum OperatorStatus {
    Ok,
    Warn,
    Degraded,
    Fail,
    Missing,
    Skip,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConsolidatedOperatorReportV1 {
    pub schema_version: u16,
    pub generated_at: u64,
    pub overall_status: OperatorStatus,
    pub run_id: Option<String>,
    pub policy_graph_digest_prefix: Option<String>,
    pub manifest_digest_prefix: Option<String>,
    pub sections: OperatorSectionsV1,
    pub remediation_codes: Vec<String>,
    #[serde(default)]
    pub canonical_remediation_codes: Vec<String>,
    pub report_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OperatorSectionsV1 {
    pub health_section: NormalizedHealthSection,
    pub eligibility_section: NormalizedEligibilitySection,
    pub drift_section: NormalizedDriftSection,
    pub alerts_section: NormalizedAlertsSection,
    pub strict_section: NormalizedStrictSection,
    pub gates_section: NormalizedGatesSection,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NormalizedHealthSection {
    pub status: OperatorStatus,
    pub strict_mode_enabled: Option<bool>,
    pub last_tick_age_ms: Option<u64>,
    pub emergency_active: Option<bool>,
    pub evidence_digest_prefixes: Vec<String>,
    pub remediation_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EligibilitySlotSummary {
    pub slot_id: String,
    pub probe_ready: bool,
    pub shadow_ready: bool,
    pub active_eligible: bool,
    pub primary_denial_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NormalizedEligibilitySection {
    pub status: OperatorStatus,
    pub slots: Vec<EligibilitySlotSummary>,
    pub evidence_digest_prefixes: Vec<String>,
    pub remediation_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DriftSlotSummary {
    pub slot_id: String,
    pub drift_status: OperatorStatus,
    pub severe_alarm_count: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NormalizedDriftSection {
    pub status: OperatorStatus,
    pub slots: Vec<DriftSlotSummary>,
    pub evidence_digest_prefixes: Vec<String>,
    pub remediation_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AlertSummary {
    pub alert_id: String,
    pub severity: String,
    pub rule_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NormalizedAlertsSection {
    pub status: OperatorStatus,
    pub active_alert_count: usize,
    pub top_active_alerts: Vec<AlertSummary>,
    pub evidence_digest_prefixes: Vec<String>,
    pub remediation_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NormalizedStrictSection {
    pub status: OperatorStatus,
    pub strict_status: StrictEvidenceStatusV1,
    pub primary_denial_code: Option<String>,
    pub strict_report_digest_prefix: Option<String>,
    pub failing_check_ids: Vec<String>,
    pub evidence_digest_prefixes: Vec<String>,
    pub remediation_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GateStatusSummary {
    pub gate_id: String,
    pub status: OperatorStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NormalizedGatesSection {
    pub status: OperatorStatus,
    pub gates: Vec<GateStatusSummary>,
    pub evidence_digest_prefixes: Vec<String>,
    pub remediation_codes: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct OperatorReportArgs {
    pub run_id: Option<String>,
    pub latest: bool,
}

pub fn operator_report(
    workdir: &Path,
    args: &OperatorReportArgs,
    out: &Path,
) -> Result<ConsolidatedOperatorReportV1, OpsError> {
    let out_root = workdir.join("out");
    let health_value = maybe_read_json_value(&discover_report(&out_root, "health.json", args));
    let eligibility = maybe_read_json::<AggregatedEligibilityReportV1>(&discover_report(
        &out_root,
        "models_eligibility_report.json",
        args,
    ));
    let evidence_snapshot = maybe_read_json::<BackendEvidenceSnapshotV1>(&discover_report(
        &out_root,
        "backend_evidence_snapshot.json",
        args,
    ));
    let drift =
        maybe_read_json::<DriftReportV1>(&discover_report(&out_root, "drift_report.json", args));
    let alerts =
        maybe_read_json::<AlertsReportV1>(&discover_report(&out_root, "alerts_report.json", args));
    let v0 =
        maybe_read_json::<V0GateReportV1>(&discover_report(&out_root, "v0_gate_report.json", args));
    let v1 =
        maybe_read_json::<V1GateReportV1>(&discover_report(&out_root, "v1_gate_report.json", args));
    let v2 =
        maybe_read_json::<V2GateReportV1>(&discover_report(&out_root, "v2_gate_report.json", args));

    let strict_snapshot = resolve_strict_evidence(
        &out_root,
        &StrictEvidenceContextV1 {
            run_id: args.run_id.clone(),
            latest: args.latest,
            strict_required: true,
            expected_policy_graph_digest_prefix: eligibility
                .as_ref()
                .map(|r| r.policy_graph_digest_prefix.clone())
                .or_else(|| {
                    evidence_snapshot
                        .as_ref()
                        .map(|s| s.policy_graph_digest_prefix.clone())
                }),
            expected_manifest_digest_prefix: eligibility
                .as_ref()
                .and_then(|r| r.slots.first().map(|s| s.manifest_digest_prefix.clone()))
                .or_else(|| {
                    evidence_snapshot
                        .as_ref()
                        .map(|s| s.manifest_digest_prefix.clone())
                }),
            expected_supported_slot_set_digest_prefix: evidence_snapshot
                .as_ref()
                .map(|s| s.supported_slot_set_digest.clone()),
        },
    );

    let health_section = normalize_health(health_value.as_ref());
    let eligibility_section = if let Some(report) = eligibility.as_ref() {
        normalize_eligibility(Some(report))
    } else {
        normalize_eligibility_from_snapshot(evidence_snapshot.as_ref())
    };
    let drift_section = normalize_drift(drift.as_ref());
    let alerts_section = normalize_alerts(alerts.as_ref());
    let strict_section = normalize_strict(&strict_snapshot);
    let gates_section = normalize_gates(v0.as_ref(), v1.as_ref(), v2.as_ref());

    let overall_status = reduce_overall_status(
        &health_section,
        &eligibility_section,
        &drift_section,
        &alerts_section,
        &strict_section,
        &gates_section,
    );

    let mut remediation = BTreeSet::new();
    for code in health_section
        .remediation_codes
        .iter()
        .chain(eligibility_section.remediation_codes.iter())
        .chain(drift_section.remediation_codes.iter())
        .chain(alerts_section.remediation_codes.iter())
        .chain(strict_section.remediation_codes.iter())
        .chain(gates_section.remediation_codes.iter())
    {
        remediation.insert(code.clone());
    }

    let run_id = args
        .run_id
        .clone()
        .or_else(|| drift.as_ref().map(|r| r.run_id.clone()))
        .or_else(|| alerts.as_ref().map(|r| r.run_id.clone()));

    let policy_graph_digest_prefix = eligibility
        .as_ref()
        .map(|r| r.policy_graph_digest_prefix.clone())
        .or_else(|| {
            evidence_snapshot
                .as_ref()
                .map(|s| s.policy_graph_digest_prefix.clone())
        })
        .or_else(|| extract_health_prefix(health_value.as_ref(), "policy_graph_digest"));

    let manifest_digest_prefix = eligibility
        .as_ref()
        .and_then(|r| r.slots.first().map(|s| s.manifest_digest_prefix.clone()))
        .or_else(|| {
            evidence_snapshot
                .as_ref()
                .map(|s| s.manifest_digest_prefix.clone())
        })
        .or_else(|| extract_health_prefix(health_value.as_ref(), "manifest_digest"));

    let generated_at = unix_now_secs();
    let sections = OperatorSectionsV1 {
        health_section,
        eligibility_section,
        drift_section,
        alerts_section,
        strict_section,
        gates_section,
    };

    let mut report = ConsolidatedOperatorReportV1 {
        schema_version: 1,
        generated_at,
        overall_status,
        run_id,
        policy_graph_digest_prefix,
        manifest_digest_prefix,
        sections,
        remediation_codes: remediation.into_iter().take(REMEDIATION_MAX).collect(),
        canonical_remediation_codes: Vec::new(),
        report_digest: String::new(),
    };
    report.canonical_remediation_codes =
        merge_canonical_remediations(report.remediation_codes.iter(), REMEDIATION_MAX);
    report.report_digest = report_digest(&report)?;

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_string_pretty(&report)?)?;

    let _ = workdir;
    Ok(report)
}

pub fn operator_report_text(report: &ConsolidatedOperatorReportV1) -> String {
    let slots = report
        .sections
        .eligibility_section
        .slots
        .iter()
        .map(|slot| {
            format!(
                "{}: probe={} shadow={} active={}",
                slot.slot_id, slot.probe_ready, slot.shadow_ready, slot.active_eligible
            )
        })
        .collect::<Vec<_>>()
        .join("; ");

    format!(
        "overall={:?}\nslots={}\nactive_alerts={}\nstrict={:?}\nnext=ucf-ops strict check --strict --out ./out/strict_check.json\nnext=ucf-ops models eligibility --out ./out/models_eligibility_report.json\nnext=ucf-ops drift report --run <id> --windows 20 --out ./out/drift_report.json",
        report.overall_status,
        if slots.is_empty() { "none" } else { &slots },
        report.sections.alerts_section.active_alert_count,
        report.sections.strict_section.strict_status
    )
}

fn discover_report(out_root: &Path, file: &str, args: &OperatorReportArgs) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(run_id) = &args.run_id {
        candidates.push(out_root.join(run_id).join(file));
    }
    if args.latest {
        let mut dirs = fs::read_dir(out_root)
            .ok()?
            .filter_map(|entry| {
                let p = entry.ok()?.path();
                if p.is_dir() {
                    Some(p)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        dirs.sort();
        dirs.reverse();
        for dir in dirs {
            candidates.push(dir.join(file));
        }
    }
    candidates.push(out_root.join(file));
    candidates.into_iter().find(|p| p.exists())
}

fn maybe_read_json<T: for<'de> Deserialize<'de>>(path: &Option<PathBuf>) -> Option<T> {
    let path = path.as_ref()?;
    serde_json::from_slice(&fs::read(path).ok()?).ok()
}

fn maybe_read_json_value(path: &Option<PathBuf>) -> Option<serde_json::Value> {
    let path = path.as_ref()?;
    serde_json::from_slice(&fs::read(path).ok()?).ok()
}

pub fn normalize_health(value: Option<&serde_json::Value>) -> NormalizedHealthSection {
    let Some(value) = value else {
        return NormalizedHealthSection {
            status: OperatorStatus::Missing,
            strict_mode_enabled: None,
            last_tick_age_ms: None,
            emergency_active: None,
            evidence_digest_prefixes: Vec::new(),
            remediation_codes: vec!["run_health_check".to_string()],
        };
    };

    let status_raw = value.get("status").and_then(|v| v.as_i64()).unwrap_or(3);
    let status = match status_raw {
        1 => OperatorStatus::Ok,
        2 => OperatorStatus::Warn,
        _ => OperatorStatus::Fail,
    };

    NormalizedHealthSection {
        status,
        strict_mode_enabled: value.get("strict_mode_enabled").and_then(|v| v.as_bool()),
        last_tick_age_ms: value.get("last_tick_age_ms").and_then(|v| v.as_u64()),
        emergency_active: value.get("emergency_active").and_then(|v| v.as_bool()),
        evidence_digest_prefixes: collect_prefixes(value),
        remediation_codes: if status_raw == 1 {
            Vec::new()
        } else {
            vec!["run_health_check".to_string()]
        },
    }
}

pub fn normalize_eligibility(
    input: Option<&AggregatedEligibilityReportV1>,
) -> NormalizedEligibilitySection {
    let Some(input) = input else {
        return NormalizedEligibilitySection {
            status: OperatorStatus::Missing,
            slots: Vec::new(),
            evidence_digest_prefixes: Vec::new(),
            remediation_codes: vec!["run_models_eligibility".to_string()],
        };
    };

    let mut slots = input
        .slots
        .iter()
        .map(|slot| EligibilitySlotSummary {
            slot_id: slot.slot_id.clone(),
            probe_ready: slot.probe_ready,
            shadow_ready: slot.shadow_ready,
            active_eligible: slot.active_eligible,
            primary_denial_reason: slot
                .denial_reason_active
                .clone()
                .or_else(|| slot.denial_reason_shadow.clone())
                .or_else(|| slot.denial_reason_probe.clone()),
        })
        .collect::<Vec<_>>();
    slots.sort_by(|a, b| a.slot_id.cmp(&b.slot_id));

    let status = if slots.is_empty() {
        OperatorStatus::Warn
    } else if slots.iter().all(|s| s.active_eligible) {
        OperatorStatus::Ok
    } else if slots.iter().any(|s| s.shadow_ready || s.probe_ready) {
        OperatorStatus::Warn
    } else {
        OperatorStatus::Degraded
    };

    let mut remediation = input
        .slots
        .iter()
        .flat_map(|s| s.remediation_codes.clone())
        .collect::<BTreeSet<_>>();
    if !matches!(status, OperatorStatus::Ok) {
        remediation.insert("run_models_eligibility".to_string());
    }

    NormalizedEligibilitySection {
        status,
        slots,
        evidence_digest_prefixes: vec![prefix(&input.report_digest)],
        remediation_codes: remediation.into_iter().collect(),
    }
}

fn normalize_eligibility_from_snapshot(
    snapshot: Option<&BackendEvidenceSnapshotV1>,
) -> NormalizedEligibilitySection {
    let Some(snapshot) = snapshot else {
        return normalize_eligibility(None);
    };

    let mut slots = snapshot
        .slots
        .iter()
        .map(|slot| EligibilitySlotSummary {
            slot_id: slot.slot_id.clone(),
            probe_ready: slot.readiness.probe_ready,
            shadow_ready: slot.readiness.shadow_ready,
            active_eligible: slot.readiness.active_eligible,
            primary_denial_reason: slot
                .denials
                .active
                .as_ref()
                .or(slot.denials.shadow.as_ref())
                .or(slot.denials.probe.as_ref())
                .map(|v| format!("{:?}", v)),
        })
        .collect::<Vec<_>>();
    slots.sort_by(|a, b| a.slot_id.cmp(&b.slot_id));

    let status = if slots.is_empty() {
        OperatorStatus::Warn
    } else if slots.iter().all(|s| s.active_eligible) {
        OperatorStatus::Ok
    } else if slots.iter().any(|s| s.shadow_ready || s.probe_ready) {
        OperatorStatus::Warn
    } else {
        OperatorStatus::Degraded
    };

    let mut remediation = snapshot
        .slots
        .iter()
        .flat_map(|s| s.remediation_codes.clone())
        .collect::<BTreeSet<_>>();
    if !matches!(status, OperatorStatus::Ok) {
        remediation.insert("run_models_eligibility".to_string());
    }

    NormalizedEligibilitySection {
        status,
        slots,
        evidence_digest_prefixes: vec![prefix(&snapshot.snapshot_digest)],
        remediation_codes: remediation.into_iter().collect(),
    }
}

pub fn normalize_drift(report: Option<&DriftReportV1>) -> NormalizedDriftSection {
    let Some(report) = report else {
        return NormalizedDriftSection {
            status: OperatorStatus::Missing,
            slots: Vec::new(),
            evidence_digest_prefixes: Vec::new(),
            remediation_codes: vec!["run_drift_report".to_string()],
        };
    };

    let mut slots = report
        .slot_reports
        .iter()
        .map(|slot| map_drift_slot(slot, report))
        .collect::<Vec<_>>();
    slots.sort_by(|a, b| a.slot_id.cmp(&b.slot_id));

    let status = if slots
        .iter()
        .any(|s| matches!(s.drift_status, OperatorStatus::Fail))
    {
        OperatorStatus::Degraded
    } else {
        OperatorStatus::Ok
    };

    NormalizedDriftSection {
        status,
        slots,
        evidence_digest_prefixes: vec![prefix(&report.report_digest)],
        remediation_codes: if matches!(report.status, GateStatus::Pass) {
            Vec::new()
        } else {
            vec!["run_drift_report".to_string()]
        },
    }
}

pub fn normalize_alerts(report: Option<&AlertsReportV1>) -> NormalizedAlertsSection {
    let Some(report) = report else {
        return NormalizedAlertsSection {
            status: OperatorStatus::Missing,
            active_alert_count: 0,
            top_active_alerts: Vec::new(),
            evidence_digest_prefixes: Vec::new(),
            remediation_codes: vec!["run_alerts_report".to_string()],
        };
    };

    let mut alerts = report
        .active_alerts
        .iter()
        .map(|a| AlertSummary {
            alert_id: a.alert_id.clone(),
            severity: a.severity.clone(),
            rule_id: a.rule_id.clone(),
        })
        .collect::<Vec<_>>();
    alerts.sort_by(|a, b| a.alert_id.cmp(&b.alert_id));
    alerts.truncate(TOP_ALERTS_MAX);

    let mut remediation = report
        .active_alerts
        .iter()
        .flat_map(|a| a.remediation_codes.clone())
        .collect::<BTreeSet<_>>();
    if !report.active_alerts.is_empty() {
        remediation.insert("inspect_active_alerts".to_string());
    }

    let status = if report.active_alerts.is_empty() {
        OperatorStatus::Ok
    } else if report.active_alerts.iter().any(|a| a.severity == "SEVERE") {
        OperatorStatus::Fail
    } else {
        OperatorStatus::Degraded
    };

    NormalizedAlertsSection {
        status,
        active_alert_count: report.active_alerts.len(),
        top_active_alerts: alerts,
        evidence_digest_prefixes: vec![prefix(&report.report_digest)],
        remediation_codes: remediation.into_iter().collect(),
    }
}

pub fn normalize_strict(snapshot: &StrictEvidenceSnapshotV1) -> NormalizedStrictSection {
    let block = operator_block_from_strict(snapshot);
    let mut evidence_digest_prefixes = Vec::new();
    if let Some(d) = snapshot.strict_report_digest_prefix.as_ref() {
        evidence_digest_prefixes.push(d.clone());
    }
    if let Some(d) = snapshot.policy_graph_digest_prefix.as_ref() {
        evidence_digest_prefixes.push(d.clone());
    }
    if let Some(d) = snapshot.manifest_digest_prefix.as_ref() {
        evidence_digest_prefixes.push(d.clone());
    }
    if let Some(d) = snapshot.supported_slot_set_digest_prefix.as_ref() {
        evidence_digest_prefixes.push(d.clone());
    }

    NormalizedStrictSection {
        status: block.status_contribution,
        strict_status: snapshot.strict_status.clone(),
        primary_denial_code: block.primary_reason_code,
        strict_report_digest_prefix: snapshot.strict_report_digest_prefix.clone(),
        failing_check_ids: snapshot.failing_check_ids.clone(),
        evidence_digest_prefixes,
        remediation_codes: block.remediation_codes,
    }
}

pub fn normalize_gates(
    v0: Option<&V0GateReportV1>,
    v1: Option<&V1GateReportV1>,
    v2: Option<&V2GateReportV1>,
) -> NormalizedGatesSection {
    let mut gates = vec![
        GateStatusSummary {
            gate_id: "v0".to_string(),
            status: v0
                .map(|g| match g.overall_status {
                    V0GateOverallStatus::Pass => OperatorStatus::Ok,
                    V0GateOverallStatus::Fail => OperatorStatus::Fail,
                })
                .unwrap_or(OperatorStatus::Missing),
        },
        GateStatusSummary {
            gate_id: "v1".to_string(),
            status: v1
                .map(|g| match g.overall_status {
                    V1GateOverallStatus::Pass => OperatorStatus::Ok,
                    V1GateOverallStatus::Fail => OperatorStatus::Fail,
                })
                .unwrap_or(OperatorStatus::Missing),
        },
        GateStatusSummary {
            gate_id: "v2".to_string(),
            status: v2
                .map(|g| match g.overall_status {
                    V2GateOverallStatus::Pass => OperatorStatus::Ok,
                    V2GateOverallStatus::Fail => OperatorStatus::Fail,
                })
                .unwrap_or(OperatorStatus::Missing),
        },
    ];
    gates.sort_by(|a, b| a.gate_id.cmp(&b.gate_id));

    let status = if gates
        .iter()
        .any(|g| matches!(g.status, OperatorStatus::Fail))
    {
        OperatorStatus::Fail
    } else if gates
        .iter()
        .any(|g| matches!(g.status, OperatorStatus::Missing))
    {
        OperatorStatus::Warn
    } else {
        OperatorStatus::Ok
    };

    let mut remediation_codes = Vec::new();
    if gates
        .iter()
        .any(|g| matches!(g.status, OperatorStatus::Missing))
    {
        remediation_codes.push("run_missing_gates".to_string());
    }

    NormalizedGatesSection {
        status,
        gates,
        evidence_digest_prefixes: Vec::new(),
        remediation_codes,
    }
}

fn reduce_overall_status(
    health: &NormalizedHealthSection,
    eligibility: &NormalizedEligibilitySection,
    drift: &NormalizedDriftSection,
    alerts: &NormalizedAlertsSection,
    strict: &NormalizedStrictSection,
    gates: &NormalizedGatesSection,
) -> OperatorStatus {
    if matches!(strict.status, OperatorStatus::Fail)
        || matches!(health.status, OperatorStatus::Fail)
        || matches!(alerts.status, OperatorStatus::Fail)
    {
        return OperatorStatus::Fail;
    }

    if matches!(drift.status, OperatorStatus::Degraded)
        || matches!(alerts.status, OperatorStatus::Degraded)
        || matches!(eligibility.status, OperatorStatus::Degraded)
    {
        return OperatorStatus::Degraded;
    }

    if matches!(health.status, OperatorStatus::Missing)
        || matches!(eligibility.status, OperatorStatus::Missing)
        || matches!(drift.status, OperatorStatus::Missing)
        || matches!(alerts.status, OperatorStatus::Missing)
        || matches!(strict.status, OperatorStatus::Missing)
        || matches!(gates.status, OperatorStatus::Warn)
        || matches!(eligibility.status, OperatorStatus::Warn)
    {
        return OperatorStatus::Warn;
    }

    OperatorStatus::Ok
}

fn map_drift_slot(slot: &DriftSlotReportV1, report: &DriftReportV1) -> DriftSlotSummary {
    let severe = report
        .alarms
        .iter()
        .filter(|alarm| alarm.slot_id == slot.slot_id && alarm.severity == "SEVERE")
        .count()
        .min(DRIFT_ALARM_COUNT_CAP as usize) as u16;
    let drift_status = match slot.status.as_str() {
        "SEVERE" => OperatorStatus::Fail,
        "WARN" => OperatorStatus::Degraded,
        _ => OperatorStatus::Ok,
    };
    DriftSlotSummary {
        slot_id: slot.slot_id.clone(),
        drift_status,
        severe_alarm_count: severe,
    }
}

fn collect_prefixes(value: &serde_json::Value) -> Vec<String> {
    ["policy_graph_digest", "manifest_digest", "report_digest"]
        .into_iter()
        .filter_map(|k| value.get(k).and_then(|v| v.as_str()))
        .map(prefix)
        .collect()
}

fn prefix(input: &str) -> String {
    input.chars().take(16).collect()
}

fn extract_health_prefix(value: Option<&serde_json::Value>, key: &str) -> Option<String> {
    value
        .and_then(|v| v.get(key))
        .and_then(|v| v.as_str())
        .map(prefix)
}

fn report_digest(report: &ConsolidatedOperatorReportV1) -> Result<String, OpsError> {
    let mut cloned = report.clone();
    cloned.report_digest.clear();
    cloned.generated_at = 0;
    Ok(crate::sha256_hex(&serde_json::to_vec(&cloned)?))
}

fn unix_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drift::{DriftAlarmRecordV1, DriftSlotReportV1};
    use crate::models_lifecycle::{DriftStatusV1, EligibilityGeneratedFromV1};
    use crate::{AlertRecordV1, AlertsReportV1, DriftReportV1};

    #[test]
    fn missing_sections_are_deterministic() {
        let section = normalize_eligibility(None);
        assert_eq!(section.status, OperatorStatus::Missing);
        assert_eq!(section.remediation_codes, vec!["run_models_eligibility"]);
    }

    #[test]
    fn strict_fail_forces_overall_fail() {
        let strict = normalize_strict(&StrictEvidenceSnapshotV1 {
            schema_version: 1,
            strict_mode_enabled: true,
            strict_status: StrictEvidenceStatusV1::Fail,
            strict_report_digest_prefix: Some("abc".to_string()),
            policy_graph_digest_prefix: None,
            manifest_digest_prefix: None,
            supported_slot_set_digest_prefix: None,
            primary_denial_code: Some("strict.fail".to_string()),
            remediation_codes: vec!["run_strict_check".to_string()],
            failing_check_ids: vec!["strict_mode".to_string()],
            snapshot_digest: "d".to_string(),
        });
        let overall = reduce_overall_status(
            &normalize_health(None),
            &normalize_eligibility(None),
            &normalize_drift(None),
            &normalize_alerts(None),
            &strict,
            &normalize_gates(None, None, None),
        );
        assert_eq!(overall, OperatorStatus::Fail);
    }

    #[test]
    fn mixed_state_is_degraded() {
        let eligibility = normalize_eligibility(Some(&AggregatedEligibilityReportV1 {
            schema_version: 1,
            overall_status: crate::EligibilityOverallStatusV1::ShadowReadyPartial,
            slots: vec![
                crate::UnifiedEligibilityStatusV1 {
                    slot_id: "sae".to_string(),
                    target_hash_prefix: "h1".to_string(),
                    manifest_digest_prefix: "m1".to_string(),
                    probe_ready: true,
                    shadow_ready: true,
                    active_eligible: false,
                    latest_probe_digest_prefix: "p1".to_string(),
                    latest_shadow_evidence_digest_prefix: "s1".to_string(),
                    latest_active_evidence_digest_prefix: "a1".to_string(),
                    latest_drift_status: DriftStatusV1::Warn,
                    burn_support_state: crate::OptionalBackendSupportStateV1::NotConfigured,
                    burn_parity_present: false,
                    denial_reason_probe: None,
                    denial_reason_shadow: None,
                    denial_reason_active: Some("need_more_evidence".to_string()),
                    remediation_codes: vec![],
                    canonical_remediation_codes: vec![],
                    status_digest: "sd1".to_string(),
                },
                crate::UnifiedEligibilityStatusV1 {
                    slot_id: "world".to_string(),
                    target_hash_prefix: "h2".to_string(),
                    manifest_digest_prefix: "m2".to_string(),
                    probe_ready: true,
                    shadow_ready: true,
                    active_eligible: true,
                    latest_probe_digest_prefix: "p2".to_string(),
                    latest_shadow_evidence_digest_prefix: "s2".to_string(),
                    latest_active_evidence_digest_prefix: "a2".to_string(),
                    latest_drift_status: DriftStatusV1::Ok,
                    burn_support_state: crate::OptionalBackendSupportStateV1::Unsupported,
                    burn_parity_present: false,
                    denial_reason_probe: None,
                    denial_reason_shadow: None,
                    denial_reason_active: None,
                    remediation_codes: vec![],
                    canonical_remediation_codes: vec![],
                    status_digest: "sd2".to_string(),
                },
            ],
            report_digest: "abc123456789".to_string(),
            policy_graph_digest_prefix: "pg1".to_string(),
            generated_from: EligibilityGeneratedFromV1 {
                probe_report_digests: vec![],
                shadow_ready_report_digest: "s".to_string(),
                active_evidence_report_digest: "a".to_string(),
                second_slot_parity_report_digest: "p".to_string(),
            },
        }));
        let drift = normalize_drift(Some(&DriftReportV1 {
            run_id: "r1".to_string(),
            status: GateStatus::Pass,
            windows_limit: 20,
            slot_reports: vec![DriftSlotReportV1 {
                slot_id: "world".to_string(),
                status: "WARN".to_string(),
                active_alarms: vec![],
                recommended_actions: vec![],
                windows: vec![],
            }],
            alarms: vec![],
            operator_summary: String::new(),
            report_digest: "d1".to_string(),
        }));
        let alerts = normalize_alerts(Some(&AlertsReportV1 {
            schema_version: 1,
            run_id: "r1".to_string(),
            active_alerts: vec![AlertRecordV1 {
                schema_version: 1,
                alert_id: "a1".to_string(),
                severity: "WARN".to_string(),
                triggered_at_t: 1,
                rule_id: "r1".to_string(),
                observed_count: 1,
                window_start_t: 0,
                window_end_t: 1,
                evidence_digests: vec![],
                remediation_codes: vec![],
            }],
            last_triggers: vec![],
            suggested_commands: vec![],
            summary_text: String::new(),
            report_digest: "a1".to_string(),
        }));
        let strict = normalize_strict(&StrictEvidenceSnapshotV1 {
            schema_version: 1,
            strict_mode_enabled: true,
            strict_status: StrictEvidenceStatusV1::Pass,
            strict_report_digest_prefix: Some("abc".to_string()),
            policy_graph_digest_prefix: None,
            manifest_digest_prefix: None,
            supported_slot_set_digest_prefix: None,
            primary_denial_code: None,
            remediation_codes: vec![],
            failing_check_ids: vec![],
            snapshot_digest: "d".to_string(),
        });
        let overall = reduce_overall_status(
            &normalize_health(Some(&serde_json::json!({"status":1}))),
            &eligibility,
            &drift,
            &alerts,
            &strict,
            &normalize_gates(None, None, None),
        );
        assert_eq!(overall, OperatorStatus::Degraded);
    }

    #[test]
    fn drift_alarm_count_capped() {
        let report = DriftReportV1 {
            run_id: "r".to_string(),
            status: GateStatus::Fail,
            windows_limit: 1,
            slot_reports: vec![DriftSlotReportV1 {
                slot_id: "world".to_string(),
                status: "SEVERE".to_string(),
                active_alarms: vec![],
                recommended_actions: vec![],
                windows: vec![],
            }],
            alarms: (0..300)
                .map(|idx| DriftAlarmRecordV1 {
                    alarm_id: format!("a{idx}"),
                    slot_id: "world".to_string(),
                    window_id: 0,
                    breached_fields: vec![],
                    observed: Default::default(),
                    severity: "SEVERE".to_string(),
                    action_taken: String::new(),
                    reason_code: String::new(),
                    evidence_digests: vec![],
                })
                .collect(),
            operator_summary: String::new(),
            report_digest: "d".to_string(),
        };
        let normalized = normalize_drift(Some(&report));
        assert_eq!(normalized.slots[0].severe_alarm_count, 255);
    }

    #[test]
    fn operator_report_schema_serialization_is_stable() {
        let report = ConsolidatedOperatorReportV1 {
            schema_version: 1,
            generated_at: 1,
            overall_status: OperatorStatus::Warn,
            run_id: Some("run-1".to_string()),
            policy_graph_digest_prefix: Some("abc123".to_string()),
            manifest_digest_prefix: Some("def456".to_string()),
            sections: OperatorSectionsV1 {
                health_section: normalize_health(None),
                eligibility_section: normalize_eligibility(None),
                drift_section: normalize_drift(None),
                alerts_section: normalize_alerts(None),
                strict_section: normalize_strict(&StrictEvidenceSnapshotV1 {
                    schema_version: 1,
                    strict_mode_enabled: true,
                    strict_status: StrictEvidenceStatusV1::Missing,
                    strict_report_digest_prefix: None,
                    policy_graph_digest_prefix: None,
                    manifest_digest_prefix: None,
                    supported_slot_set_digest_prefix: None,
                    primary_denial_code: Some("STRICT_EVIDENCE_MISSING".to_string()),
                    remediation_codes: vec!["run_strict_check".to_string()],
                    failing_check_ids: vec![],
                    snapshot_digest: "d".to_string(),
                }),
                gates_section: normalize_gates(None, None, None),
            },
            remediation_codes: vec!["run_models_eligibility".to_string()],
            canonical_remediation_codes: vec![],
            report_digest: "digest".to_string(),
        };
        let a = serde_json::to_vec(&report).expect("serialize a");
        let b = serde_json::to_vec(&report).expect("serialize b");
        assert_eq!(a, b);
    }
}
