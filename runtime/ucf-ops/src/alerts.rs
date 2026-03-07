use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use ucf_ess::v1::{AuditPayload, ExperiencePayload, ExperienceRecord};
use ucf_policy::policy_packs::{
    load_and_merge_policy_graph, AlertActionV1, AlertRuleKindV1, AlertRulesV1, AlertSeverityV1,
};

use crate::drift::DriftAlarmRecordV1;
use crate::{load_fixture_records, persist_jsonl_record, sha256_hex, OpsError};

const ALERTS_MAX_ACTIVE: usize = 16;
const ALERTS_MAX_HISTORY: usize = 20;
const ALERTS_MAX_EVIDENCE: usize = 4;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AlertRecordV1 {
    pub schema_version: u16,
    pub alert_id: String,
    pub severity: String,
    pub triggered_at_t: u64,
    pub rule_id: String,
    pub observed_count: u32,
    pub window_start_t: u64,
    pub window_end_t: u64,
    pub evidence_digests: Vec<String>,
    pub remediation_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AlertClearRecordV1 {
    pub schema_version: u16,
    pub alert_id: String,
    pub rule_id: String,
    pub cleared_at_t: u64,
    pub window_start_t: u64,
    pub window_end_t: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum AlertEventV1 {
    Trigger(AlertRecordV1),
    Clear(AlertClearRecordV1),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AlertsReportV1 {
    pub schema_version: u16,
    pub run_id: String,
    pub active_alerts: Vec<AlertRecordV1>,
    pub last_triggers: Vec<AlertRecordV1>,
    pub suggested_commands: Vec<String>,
    pub summary_text: String,
    pub report_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AlertObservation {
    count: u32,
    window_start_t: u64,
    window_end_t: u64,
    evidence_digests: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct AlertEvaluatorStateV1 {
    schema_version: u16,
    below_threshold_windows: BTreeMap<String, u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GatewayAbuseRecordV1 {
    t_ms: u64,
    reason_code: String,
}

pub fn alerts_report(workdir: &Path, run_id: &str, out: &Path) -> Result<AlertsReportV1, OpsError> {
    let overlay = std::env::var("UCF_POLICY_OVERLAY").ok();
    let overlay_path = overlay
        .as_deref()
        .map(|name| PathBuf::from("policies/packs/overlays").join(name));
    let (graph, _) =
        load_and_merge_policy_graph(Path::new("policies/packs/base_v1"), overlay_path.as_deref())?;

    let records =
        load_fixture_records(&workdir.join("ess").join("ess_fixture.json")).unwrap_or_default();
    let evaluator = AlertEvaluator::new(graph.alerts);
    let mut events = evaluator.evaluate(workdir, run_id, &records)?;

    let events_path = workdir.join("out").join("alerts_records.jsonl");
    let history = load_jsonl::<AlertEventV1>(&events_path)?;
    let mut active: BTreeMap<String, AlertRecordV1> = BTreeMap::new();
    for event in history.iter().chain(events.iter()) {
        match event {
            AlertEventV1::Trigger(r) => {
                active.insert(r.alert_id.clone(), r.clone());
            }
            AlertEventV1::Clear(c) => {
                active.remove(&c.alert_id);
            }
        }
    }

    for event in events.drain(..) {
        persist_jsonl_record(&events_path, &event)?;
    }

    let mut active_alerts: Vec<_> = active.into_values().collect();
    active_alerts.sort_by(|a, b| a.alert_id.cmp(&b.alert_id));
    active_alerts.truncate(ALERTS_MAX_ACTIVE);

    let mut last_triggers = history
        .iter()
        .chain(load_jsonl::<AlertEventV1>(&events_path)?.iter())
        .filter_map(|event| match event {
            AlertEventV1::Trigger(r) => Some(r.clone()),
            AlertEventV1::Clear(_) => None,
        })
        .collect::<Vec<_>>();
    last_triggers.sort_by(|a, b| {
        a.triggered_at_t
            .cmp(&b.triggered_at_t)
            .then(a.alert_id.cmp(&b.alert_id))
    });
    if last_triggers.len() > ALERTS_MAX_HISTORY {
        last_triggers = last_triggers.split_off(last_triggers.len() - ALERTS_MAX_HISTORY);
    }

    let mut command_set = BTreeSet::new();
    for alert in &active_alerts {
        for code in &alert.remediation_codes {
            for cmd in remediation_commands(code) {
                command_set.insert(cmd.to_string());
            }
        }
    }
    let suggested_commands = command_set.into_iter().collect::<Vec<_>>();

    let summary_text = if active_alerts.is_empty() {
        "No active operational alerts.".to_string()
    } else {
        format!(
            "{} active alert(s), {} trigger event(s) in retained history.",
            active_alerts.len(),
            last_triggers.len()
        )
    };

    let mut report = AlertsReportV1 {
        schema_version: 1,
        run_id: run_id.to_string(),
        active_alerts,
        last_triggers,
        suggested_commands,
        summary_text,
        report_digest: String::new(),
    };
    report.report_digest = sha256_hex(&serde_json::to_vec(&report)?);

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_string_pretty(&report)?)?;
    Ok(report)
}

fn remediation_commands(code: &str) -> &'static [&'static str] {
    match code {
        "run_drift_report" => {
            &["ucf-ops drift report --run <id> --windows 20 --out ./out/drift_report.json"]
        }
        "run_gateway_threat_test" => {
            &["ucf-ops gateway threat-test --out ./out/gateway_threat.json"]
        }
        "run_strict_check" => &["ucf-ops strict check --strict --out ./out/strict_check.json"],
        "recommend_rollback" => &["ucf-ops models rollback --slot world --to <hash>"],
        "recommend_disable_shadow" => {
            &["ucf-ops drift report --run <id> --windows 20 --out ./out/drift_report.json"]
        }
        _ => &[],
    }
}

struct AlertEvaluator {
    rules: AlertRulesV1,
}

impl AlertEvaluator {
    fn new(rules: AlertRulesV1) -> Self {
        Self { rules }
    }

    fn evaluate(
        &self,
        workdir: &Path,
        run_id: &str,
        records: &[ExperienceRecord],
    ) -> Result<Vec<AlertEventV1>, OpsError> {
        let active_ids = self.current_active_ids(workdir)?;
        let state_path = workdir.join("out/alerts_state.json");
        let mut state = read_state(&state_path)?;
        let mut events = Vec::new();

        for rule in &self.rules.rules {
            let observation = observe_rule(rule.kind, rule.window_size, workdir, run_id, records)?;
            let alert_id = format!("{}:{}", kind_code(rule.kind), rule.id);
            let triggered = observation.count >= rule.threshold;
            let is_active = active_ids.contains(&alert_id);

            if triggered {
                state.below_threshold_windows.insert(alert_id.clone(), 0);
                if !is_active {
                    events.push(AlertEventV1::Trigger(AlertRecordV1 {
                        schema_version: 1,
                        alert_id: alert_id.clone(),
                        severity: severity_code(rule.severity).to_string(),
                        triggered_at_t: observation.window_end_t,
                        rule_id: rule.id.clone(),
                        observed_count: observation.count,
                        window_start_t: observation.window_start_t,
                        window_end_t: observation.window_end_t,
                        evidence_digests: observation.evidence_digests,
                        remediation_codes: remediation_codes_for_rule(rule.kind, rule.action),
                    }));
                }
                continue;
            }

            if is_active {
                let counter = state
                    .below_threshold_windows
                    .entry(alert_id.clone())
                    .or_insert(0);
                *counter = counter.saturating_add(1);
                if *counter >= rule.clear_after_windows.max(1) {
                    events.push(AlertEventV1::Clear(AlertClearRecordV1 {
                        schema_version: 1,
                        alert_id: alert_id.clone(),
                        rule_id: rule.id.clone(),
                        cleared_at_t: observation.window_end_t,
                        window_start_t: observation.window_start_t,
                        window_end_t: observation.window_end_t,
                    }));
                    state.below_threshold_windows.insert(alert_id, 0);
                }
            } else {
                state.below_threshold_windows.insert(alert_id, 0);
            }
        }

        write_state(&state_path, &state)?;
        Ok(events)
    }

    fn current_active_ids(&self, workdir: &Path) -> Result<BTreeSet<String>, OpsError> {
        let mut active = BTreeSet::new();
        for event in load_jsonl::<AlertEventV1>(&workdir.join("out").join("alerts_records.jsonl"))?
        {
            match event {
                AlertEventV1::Trigger(r) => {
                    active.insert(r.alert_id);
                }
                AlertEventV1::Clear(c) => {
                    active.remove(&c.alert_id);
                }
            }
        }
        Ok(active)
    }
}

fn observe_rule(
    kind: AlertRuleKindV1,
    window_size: u32,
    workdir: &Path,
    run_id: &str,
    records: &[ExperienceRecord],
) -> Result<AlertObservation, OpsError> {
    let end_t = records.iter().map(|r| r.time.tick.get()).max().unwrap_or(0);
    let start_t = end_t.saturating_sub(window_size as u64).saturating_add(1);
    match kind {
        AlertRuleKindV1::DriftSevereRate => {
            let path = workdir
                .join("reports")
                .join("world_vljepa")
                .join(format!("{}_alarms.jsonl", run_id));
            let mut alarms = load_jsonl::<DriftAlarmRecordV1>(&path)?;
            alarms.retain(|a| a.severity == "SEVERE");
            alarms.sort_by(|a, b| a.alarm_id.cmp(&b.alarm_id));
            if alarms.len() > window_size as usize {
                alarms = alarms.split_off(alarms.len() - window_size as usize);
            }
            let evidence = alarms
                .iter()
                .take(ALERTS_MAX_EVIDENCE)
                .map(|v| sha256_hex(&serde_json::to_vec(v).unwrap_or_default()))
                .collect::<Vec<_>>();
            Ok(AlertObservation {
                count: alarms.len() as u32,
                window_start_t: start_t,
                window_end_t: end_t,
                evidence_digests: evidence,
            })
        }
        AlertRuleKindV1::GatewayAuthFailRate => {
            let mut abuse =
                load_jsonl::<GatewayAbuseRecordV1>(&workdir.join("gateway_abuse_records.jsonl"))?;
            abuse.retain(|v| v.reason_code == "AuthFail");
            abuse.sort_by_key(|v| v.t_ms);
            if abuse.len() > window_size as usize {
                abuse = abuse.split_off(abuse.len() - window_size as usize);
            }
            let evidence = abuse
                .iter()
                .take(ALERTS_MAX_EVIDENCE)
                .map(|v| sha256_hex(&serde_json::to_vec(v).unwrap_or_default()))
                .collect::<Vec<_>>();
            Ok(AlertObservation {
                count: abuse.len() as u32,
                window_start_t: start_t,
                window_end_t: end_t,
                evidence_digests: evidence,
            })
        }
        AlertRuleKindV1::StrictFailurePresent => {
            let strict_path = workdir.join("out").join("strict_failure.json");
            let present = strict_path.exists();
            let mut evidence = Vec::new();
            if present {
                let body = fs::read(&strict_path)?;
                evidence.push(sha256_hex(&body));
            }
            Ok(AlertObservation {
                count: u32::from(present),
                window_start_t: start_t,
                window_end_t: end_t,
                evidence_digests: evidence,
            })
        }
        AlertRuleKindV1::DegradedFallbackRate => {
            let mut matched = records
                .iter()
                .filter(|r| r.time.tick.get() >= start_t && r.time.tick.get() <= end_t)
                .filter_map(|r| match &r.payload {
                    ExperiencePayload::Audit(AuditPayload::EbmEnvelopeViolation(_)) => {
                        Some((r.id.0, r.time.tick.get()))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            matched.sort();
            let evidence = matched
                .iter()
                .take(ALERTS_MAX_EVIDENCE)
                .map(|(id, t)| sha256_hex(format!("degraded:{id}:{t}").as_bytes()))
                .collect::<Vec<_>>();
            Ok(AlertObservation {
                count: matched.len() as u32,
                window_start_t: start_t,
                window_end_t: end_t,
                evidence_digests: evidence,
            })
        }
    }
}

fn kind_code(kind: AlertRuleKindV1) -> &'static str {
    match kind {
        AlertRuleKindV1::DriftSevereRate => "drift_severe_rate",
        AlertRuleKindV1::GatewayAuthFailRate => "gateway_auth_fail_rate",
        AlertRuleKindV1::StrictFailurePresent => "strict_failure_present",
        AlertRuleKindV1::DegradedFallbackRate => "degraded_fallback_rate",
    }
}

fn severity_code(severity: AlertSeverityV1) -> &'static str {
    match severity {
        AlertSeverityV1::Low => "low",
        AlertSeverityV1::Medium => "medium",
        AlertSeverityV1::High => "high",
        AlertSeverityV1::Critical => "critical",
    }
}

fn remediation_codes_for_rule(kind: AlertRuleKindV1, action: AlertActionV1) -> Vec<String> {
    let mut out = match kind {
        AlertRuleKindV1::DriftSevereRate => vec!["run_drift_report".to_string()],
        AlertRuleKindV1::GatewayAuthFailRate => vec!["run_gateway_threat_test".to_string()],
        AlertRuleKindV1::StrictFailurePresent => vec!["run_strict_check".to_string()],
        AlertRuleKindV1::DegradedFallbackRate => vec!["run_strict_check".to_string()],
    };
    match action {
        AlertActionV1::RecommendRollback => out.push("recommend_rollback".to_string()),
        AlertActionV1::RecommendDisableShadow => out.push("recommend_disable_shadow".to_string()),
        AlertActionV1::RequireOperator => {}
    }
    out.sort();
    out.dedup();
    out
}

fn load_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>, OpsError> {
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

fn read_state(path: &Path) -> Result<AlertEvaluatorStateV1, OpsError> {
    if !path.exists() {
        return Ok(AlertEvaluatorStateV1 {
            schema_version: 1,
            ..AlertEvaluatorStateV1::default()
        });
    }
    let state = serde_json::from_slice::<AlertEvaluatorStateV1>(&fs::read(path)?)?;
    Ok(state)
}

fn write_state(path: &Path, state: &AlertEvaluatorStateV1) -> Result<(), OpsError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(state)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn evaluator_triggers_and_clears_with_hysteresis() {
        let tmp = tempdir().expect("tmp");
        let alarms_dir = tmp.path().join("reports/world_vljepa");
        fs::create_dir_all(&alarms_dir).expect("mkdir");
        fs::write(
            alarms_dir.join("run-a_alarms.jsonl"),
            "{\"alarm_id\":\"a\",\"slot_id\":\"world_vljepa\",\"window_id\":1,\"breached_fields\":[],\"observed\":{},\"severity\":\"SEVERE\",\"action_taken\":\"disable_shadow\",\"reason_code\":\"x\",\"evidence_digests\":[]}\n",
        )
        .expect("write");
        let rules = AlertRulesV1 {
            schema_version: 1,
            rules: vec![ucf_policy::policy_packs::AlertRuleV1 {
                id: "drift_severe_rate".to_string(),
                kind: AlertRuleKindV1::DriftSevereRate,
                window_size: 32,
                threshold: 1,
                clear_after_windows: 2,
                severity: AlertSeverityV1::High,
                action: AlertActionV1::RecommendDisableShadow,
            }],
        };
        let evaluator = AlertEvaluator::new(rules);
        let events = evaluator
            .evaluate(tmp.path(), "run-a", &[])
            .expect("evaluate");
        assert!(matches!(events.first(), Some(AlertEventV1::Trigger(_))));
        for event in events {
            persist_jsonl_record(&tmp.path().join("out/alerts_records.jsonl"), &event)
                .expect("persist");
        }

        fs::write(alarms_dir.join("run-a_alarms.jsonl"), "").expect("clear alarms");
        let first_below = evaluator
            .evaluate(tmp.path(), "run-a", &[])
            .expect("evaluate first below");
        assert!(first_below.is_empty());
        let second_below = evaluator
            .evaluate(tmp.path(), "run-a", &[])
            .expect("evaluate second below");
        assert!(matches!(second_below.first(), Some(AlertEventV1::Clear(_))));
    }
}
