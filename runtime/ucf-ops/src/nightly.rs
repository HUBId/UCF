use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::{
    AdversarialReport, DocsLintReport, DriftReportV1, GateStatus, GoldenVerifyReport,
    ReadinessGateReport,
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum NightlyOverallStatus {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NightlyComponentReport {
    pub name: String,
    pub status: NightlyOverallStatus,
    pub detail: String,
    pub remediation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NightlySummaryReport {
    pub status: NightlyOverallStatus,
    pub golden_refresh_suggested: bool,
    pub failing_components: Vec<String>,
    pub components: Vec<NightlyComponentReport>,
    pub triage_hints: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct NightlySummarizeArgs {
    pub docs_lint_report: PathBuf,
    pub gate_report: PathBuf,
    pub adversarial_report: PathBuf,
    pub goldens_report: PathBuf,
    pub drift_report: Option<PathBuf>,
    pub out: PathBuf,
}

pub fn nightly_summarize(
    args: &NightlySummarizeArgs,
) -> Result<NightlySummaryReport, crate::OpsError> {
    let docs: DocsLintReport = serde_json::from_str(&fs::read_to_string(&args.docs_lint_report)?)?;
    let gate: ReadinessGateReport = serde_json::from_str(&fs::read_to_string(&args.gate_report)?)?;
    let adversarial: AdversarialReport =
        serde_json::from_str(&fs::read_to_string(&args.adversarial_report)?)?;
    let goldens: GoldenVerifyReport =
        serde_json::from_str(&fs::read_to_string(&args.goldens_report)?)?;
    let drift = if let Some(path) = &args.drift_report {
        Some(serde_json::from_str::<DriftReportV1>(&fs::read_to_string(
            path,
        )?)?)
    } else {
        None
    };

    let mut components = vec![
        component(
            "docs_lint",
            docs.ok,
            format!("checks={} strict={}", docs.checks.len(), matches!(docs.mode, crate::DocsLintMode::Strict)),
            "cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json",
        ),
        component(
            "readiness_gate",
            gate.status == GateStatus::Pass,
            format!("status={:?} checks={}", gate.status, gate.checks.len()),
            "cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json --workdir ./.ucf_gate",
        ),
        component(
            "adversarial",
            adversarial.pass,
            format!("suite={} cases={}", adversarial.suite_version, adversarial.cases.len()),
            "cargo run -p ucf-ops -- adversarial-run --suite v1 --out ./out/adversarial_report.json",
        ),
        component(
            "goldens_verify",
            goldens.status == GateStatus::Pass,
            format!("os={} scenarios={}", goldens.os, goldens.scenarios.len()),
            "cargo run -p ucf-ops -- goldens verify --all --os linux --report-out ./out/goldens_report.json",
        ),
    ];

    if let Some(drift_report) = drift {
        components.push(component(
            "drift_report",
            drift_report.status == GateStatus::Pass,
            format!("run_id={} stages={}", drift_report.run_id, drift_report.stage_reports.len()),
            "cargo run -p ucf-ops -- drift report --run <run_id> --windows 4 --out ./out/drift_report.json",
        ));
    }

    components.sort_by(|a, b| a.name.cmp(&b.name));
    let failing_components = components
        .iter()
        .filter(|it| it.status == NightlyOverallStatus::Fail)
        .map(|it| it.name.clone())
        .collect::<Vec<_>>();

    let mut triage: BTreeMap<String, String> = BTreeMap::new();
    for comp in &components {
        if comp.status == NightlyOverallStatus::Fail {
            triage.insert(comp.name.clone(), comp.remediation.clone());
        }
    }

    for scenario in &goldens.scenarios {
        if scenario.status == GateStatus::Fail {
            let key = format!("goldens:{}", scenario.scenario);
            triage.insert(key, scenario.remediation.clone());
        }
    }

    let golden_refresh_suggested = goldens.status == GateStatus::Fail
        && !goldens.scenarios.is_empty()
        && goldens
            .scenarios
            .iter()
            .all(|s| s.status == GateStatus::Fail && s.refresh_candidate);

    let report = NightlySummaryReport {
        status: if failing_components.is_empty() {
            NightlyOverallStatus::Pass
        } else {
            NightlyOverallStatus::Fail
        },
        golden_refresh_suggested,
        failing_components,
        components,
        triage_hints: triage.values().cloned().collect(),
    };
    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.out, serde_json::to_string_pretty(&report)?)?;
    Ok(report)
}

fn component(name: &str, ok: bool, detail: String, remediation: &str) -> NightlyComponentReport {
    NightlyComponentReport {
        name: name.to_string(),
        status: if ok {
            NightlyOverallStatus::Pass
        } else {
            NightlyOverallStatus::Fail
        },
        detail,
        remediation: remediation.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GoldenVerifyScenarioReport;

    #[test]
    fn deterministic_component_ordering() {
        let mut c = [
            component("z", true, "".to_string(), "a"),
            component("a", true, "".to_string(), "a"),
        ];
        c.sort_by(|a, b| a.name.cmp(&b.name));
        assert_eq!(c[0].name, "a");
    }

    #[test]
    fn refresh_suggested_only_for_candidate_failures() {
        let report = GoldenVerifyReport {
            os: "linux".to_string(),
            status: GateStatus::Fail,
            scenarios: vec![GoldenVerifyScenarioReport {
                scenario: "golden_a".to_string(),
                status: GateStatus::Fail,
                refresh_candidate: true,
                heuristic: crate::GoldenRefreshHeuristic::DigestPrefixOnly,
                detail: "x".to_string(),
                remediation: "cmd".to_string(),
            }],
        };
        let suggested = report.status == GateStatus::Fail
            && report
                .scenarios
                .iter()
                .all(|s| s.status == GateStatus::Fail && s.refresh_candidate);
        assert!(suggested);
    }
}
