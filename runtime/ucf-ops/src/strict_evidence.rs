use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::operator_report::OperatorStatus;
use crate::{sha256_hex, StrictCheckReport, StrictCheckStatus, StrictModeFailureReport};

const REMEDIATION_BOUND: usize = 4;
const FAILING_CHECK_BOUND: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum StrictEvidenceStatusV1 {
    Pass,
    Fail,
    Missing,
    Skip,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StrictEvidenceSnapshotV1 {
    pub schema_version: u16,
    pub strict_mode_enabled: bool,
    pub strict_status: StrictEvidenceStatusV1,
    pub strict_report_digest_prefix: Option<String>,
    pub policy_graph_digest_prefix: Option<String>,
    pub manifest_digest_prefix: Option<String>,
    pub supported_slot_set_digest_prefix: Option<String>,
    pub primary_denial_code: Option<String>,
    pub remediation_codes: Vec<String>,
    pub failing_check_ids: Vec<String>,
    pub snapshot_digest: String,
}

#[derive(Debug, Clone, Default)]
pub struct StrictEvidenceContextV1 {
    pub run_id: Option<String>,
    pub latest: bool,
    pub strict_required: bool,
    pub expected_policy_graph_digest_prefix: Option<String>,
    pub expected_manifest_digest_prefix: Option<String>,
    pub expected_supported_slot_set_digest_prefix: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OperatorBlockingViewV1 {
    pub status_contribution: OperatorStatus,
    pub primary_reason_code: Option<String>,
    pub remediation_codes: Vec<String>,
}

pub fn resolve_strict_evidence(
    out_root: &Path,
    ctx: &StrictEvidenceContextV1,
) -> StrictEvidenceSnapshotV1 {
    let report = discover_and_read_strict(out_root, ctx);
    let mut snapshot = match report {
        Some(strict_report) => snapshot_from_report(&strict_report),
        None => StrictEvidenceSnapshotV1 {
            schema_version: 1,
            strict_mode_enabled: false,
            strict_status: if ctx.strict_required {
                StrictEvidenceStatusV1::Missing
            } else {
                StrictEvidenceStatusV1::Skip
            },
            strict_report_digest_prefix: None,
            policy_graph_digest_prefix: None,
            manifest_digest_prefix: None,
            supported_slot_set_digest_prefix: None,
            primary_denial_code: if ctx.strict_required {
                Some("STRICT_EVIDENCE_MISSING".to_string())
            } else {
                None
            },
            remediation_codes: if ctx.strict_required {
                vec!["run_strict_check".to_string()]
            } else {
                Vec::new()
            },
            failing_check_ids: Vec::new(),
            snapshot_digest: String::new(),
        },
    };

    if has_context_mismatch(&snapshot, ctx) {
        snapshot.strict_status = StrictEvidenceStatusV1::Fail;
        snapshot.primary_denial_code = Some("STRICT_EVIDENCE_CONTEXT_MISMATCH".to_string());
        snapshot.remediation_codes = vec!["run_strict_check".to_string()];
        if snapshot.failing_check_ids.is_empty() {
            snapshot
                .failing_check_ids
                .push("strict.evidence.context_mismatch".to_string());
        }
    }

    snapshot.strict_mode_enabled = snapshot.strict_mode_enabled || ctx.strict_required;
    snapshot.snapshot_digest = strict_snapshot_digest(&snapshot);
    snapshot
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StrictExplainReportV1 {
    pub schema_version: u16,
    pub snapshot: StrictEvidenceSnapshotV1,
    pub operator_blocking_view: OperatorBlockingViewV1,
    pub explanation: String,
}

pub fn strict_explain(out_root: &Path, ctx: &StrictEvidenceContextV1) -> StrictExplainReportV1 {
    let snapshot = resolve_strict_evidence(out_root, ctx);
    let blocking = operator_block_from_strict(&snapshot);
    let explanation = match snapshot.strict_status {
        StrictEvidenceStatusV1::Pass => {
            "strict evidence explicitly passed; strict does not block operator surfaces".to_string()
        }
        StrictEvidenceStatusV1::Fail => format!(
            "strict evidence failed; operator report/signoff both block on {}",
            blocking
                .primary_reason_code
                .clone()
                .unwrap_or_else(|| "SIGNOFF_BLOCK_STRICT".to_string())
        ),
        StrictEvidenceStatusV1::Missing => {
            "strict evidence is missing while required; fail-closed blocking is applied".to_string()
        }
        StrictEvidenceStatusV1::Skip => {
            "strict evidence is not applicable in current bounded context".to_string()
        }
    };
    StrictExplainReportV1 {
        schema_version: 1,
        snapshot,
        operator_blocking_view: blocking,
        explanation,
    }
}

pub fn operator_block_from_strict(snapshot: &StrictEvidenceSnapshotV1) -> OperatorBlockingViewV1 {
    match snapshot.strict_status {
        StrictEvidenceStatusV1::Pass | StrictEvidenceStatusV1::Skip => OperatorBlockingViewV1 {
            status_contribution: OperatorStatus::Ok,
            primary_reason_code: None,
            remediation_codes: Vec::new(),
        },
        StrictEvidenceStatusV1::Fail => OperatorBlockingViewV1 {
            status_contribution: OperatorStatus::Fail,
            primary_reason_code: snapshot.primary_denial_code.clone(),
            remediation_codes: snapshot.remediation_codes.clone(),
        },
        StrictEvidenceStatusV1::Missing => OperatorBlockingViewV1 {
            status_contribution: OperatorStatus::Missing,
            primary_reason_code: snapshot.primary_denial_code.clone(),
            remediation_codes: snapshot.remediation_codes.clone(),
        },
    }
}

fn discover_and_read_strict(
    out_root: &Path,
    ctx: &StrictEvidenceContextV1,
) -> Option<StrictCheckReport> {
    let mut candidates = Vec::new();
    if let Some(run_id) = &ctx.run_id {
        candidates.push(out_root.join(run_id).join("strict_check.json"));
    }
    if ctx.latest {
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
            candidates.push(dir.join("strict_check.json"));
        }
    }
    candidates.push(out_root.join("strict_check.json"));

    for candidate in candidates {
        if !candidate.exists() {
            continue;
        }
        if let Ok(body) = fs::read(&candidate) {
            if let Ok(report) = serde_json::from_slice::<StrictCheckReport>(&body) {
                return Some(report);
            }
        }
    }
    None
}

fn snapshot_from_report(report: &StrictCheckReport) -> StrictEvidenceSnapshotV1 {
    let strict_status = if report.ok {
        StrictEvidenceStatusV1::Pass
    } else {
        StrictEvidenceStatusV1::Fail
    };

    let failed = gather_failed_checks(&report.report);
    let primary_denial_code = failed.first().and_then(|(_, codes)| codes.first().cloned());
    let failing_check_ids = failed
        .iter()
        .map(|(id, _)| id.clone())
        .take(FAILING_CHECK_BOUND)
        .collect::<Vec<_>>();

    let mut remediation = failed
        .iter()
        .flat_map(|(_, codes)| codes.iter().cloned())
        .collect::<BTreeSet<_>>();
    if !report.ok {
        remediation.insert("run_strict_check".to_string());
    }

    StrictEvidenceSnapshotV1 {
        schema_version: 1,
        strict_mode_enabled: report.strict_mode_enabled,
        strict_status,
        strict_report_digest_prefix: report.report.digest_hex().ok().map(|d| prefix16(&d)),
        policy_graph_digest_prefix: report
            .report
            .evidence_digest_prefixes
            .get("policy_graph_digest")
            .cloned(),
        manifest_digest_prefix: report
            .report
            .evidence_digest_prefixes
            .get("manifest_digest")
            .cloned(),
        supported_slot_set_digest_prefix: report
            .report
            .evidence_digest_prefixes
            .get("supported_slot_set_digest")
            .cloned(),
        primary_denial_code,
        remediation_codes: remediation.into_iter().take(REMEDIATION_BOUND).collect(),
        failing_check_ids,
        snapshot_digest: String::new(),
    }
}

fn gather_failed_checks(report: &StrictModeFailureReport) -> Vec<(String, Vec<String>)> {
    let mut failed = report
        .checks
        .iter()
        .chain(report.v1_checks.iter())
        .filter(|check| matches!(check.status, StrictCheckStatus::Fail))
        .map(|check| {
            let mut error_codes = check.error_codes.clone();
            error_codes.sort();
            (check.check_id.clone(), error_codes)
        })
        .collect::<Vec<_>>();
    if let Some(v3) = report.v3.as_ref() {
        for check in &v3.checks {
            if format!("{:?}", check.status).eq_ignore_ascii_case("fail") {
                failed.push((
                    check.check_id.clone(),
                    check
                        .denial_code
                        .clone()
                        .map(|code| vec![code])
                        .unwrap_or_default(),
                ));
            }
        }
    }
    failed.sort_by(|a, b| a.0.cmp(&b.0));
    failed
}

fn has_context_mismatch(
    snapshot: &StrictEvidenceSnapshotV1,
    ctx: &StrictEvidenceContextV1,
) -> bool {
    prefix_mismatch(
        snapshot.policy_graph_digest_prefix.as_deref(),
        ctx.expected_policy_graph_digest_prefix.as_deref(),
    ) || prefix_mismatch(
        snapshot.manifest_digest_prefix.as_deref(),
        ctx.expected_manifest_digest_prefix.as_deref(),
    ) || prefix_mismatch(
        snapshot.supported_slot_set_digest_prefix.as_deref(),
        ctx.expected_supported_slot_set_digest_prefix.as_deref(),
    )
}

fn prefix_mismatch(actual: Option<&str>, expected: Option<&str>) -> bool {
    match (actual, expected) {
        (Some(a), Some(e)) => !a.starts_with(e) && !e.starts_with(a),
        _ => false,
    }
}

fn strict_snapshot_digest(snapshot: &StrictEvidenceSnapshotV1) -> String {
    let mut cloned = snapshot.clone();
    cloned.snapshot_digest.clear();
    sha256_hex(&serde_json::to_vec(&cloned).unwrap_or_default())
}

fn prefix16(input: &str) -> String {
    input.chars().take(16).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{StrictCheckResult, StrictModeFailureReport};
    use std::path::PathBuf;

    #[test]
    fn strict_snapshot_digest_deterministic() {
        let mut snapshot = StrictEvidenceSnapshotV1 {
            schema_version: 1,
            strict_mode_enabled: true,
            strict_status: StrictEvidenceStatusV1::Fail,
            strict_report_digest_prefix: Some("abc".to_string()),
            policy_graph_digest_prefix: Some("pg".to_string()),
            manifest_digest_prefix: Some("mf".to_string()),
            supported_slot_set_digest_prefix: Some("ss".to_string()),
            primary_denial_code: Some("strict.fail".to_string()),
            remediation_codes: vec!["run_strict_check".to_string()],
            failing_check_ids: vec!["a".to_string()],
            snapshot_digest: String::new(),
        };
        snapshot.snapshot_digest = strict_snapshot_digest(&snapshot);
        let first = snapshot.snapshot_digest.clone();
        snapshot.snapshot_digest = strict_snapshot_digest(&snapshot);
        assert_eq!(first, snapshot.snapshot_digest);
    }

    #[test]
    fn missing_strict_evidence_is_explicit() {
        let ctx = StrictEvidenceContextV1 {
            strict_required: true,
            ..StrictEvidenceContextV1::default()
        };
        let snapshot = resolve_strict_evidence(&PathBuf::from("./out/does_not_exist"), &ctx);
        assert_eq!(snapshot.strict_status, StrictEvidenceStatusV1::Missing);
        assert_eq!(
            snapshot.primary_denial_code.as_deref(),
            Some("STRICT_EVIDENCE_MISSING")
        );
    }

    #[test]
    fn context_mismatch_maps_to_stable_fail_code() {
        let report = StrictCheckReport {
            strict_mode_enabled: true,
            ok: true,
            report: StrictModeFailureReport {
                schema_version: 1,
                strict_mode_enabled: true,
                profile: "test".to_string(),
                checks: vec![StrictCheckResult {
                    check_id: "strict_mode".to_string(),
                    status: StrictCheckStatus::Pass,
                    error_codes: vec![],
                    remediation: "none".to_string(),
                    canonical_remediation_codes: vec![],
                }],
                v1_checks: vec![],
                v3: None,
                evidence_digest_prefixes: [("policy_graph_digest".to_string(), "abc".to_string())]
                    .into_iter()
                    .collect(),
            },
        };
        let out = tempfile::tempdir().expect("tmp");
        fs::write(
            out.path().join("strict_check.json"),
            serde_json::to_vec(&report).expect("json"),
        )
        .expect("write");
        let snapshot = resolve_strict_evidence(
            out.path(),
            &StrictEvidenceContextV1 {
                strict_required: true,
                expected_policy_graph_digest_prefix: Some("zzz".to_string()),
                ..StrictEvidenceContextV1::default()
            },
        );
        assert_eq!(snapshot.strict_status, StrictEvidenceStatusV1::Fail);
        assert_eq!(
            snapshot.primary_denial_code.as_deref(),
            Some("STRICT_EVIDENCE_CONTEXT_MISMATCH")
        );
    }
}
