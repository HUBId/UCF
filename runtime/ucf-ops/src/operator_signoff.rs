use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::operator_report::{ConsolidatedOperatorReportV1, OperatorStatus};
use crate::remediation::merge_canonical_remediations;
use crate::{
    derive_canonical_governance_entry, derive_canonical_readiness_authority_v2,
    derive_canonical_readiness_spine, derive_slot_reviewability_truths,
    load_applied_supported_set_context_v1, operator_block_from_strict, reduce_reviewability,
    require_canonical_readiness_spine, resolve_strict_evidence,
    validate_governance_primary_surfaces_from_workdir,
    validate_governance_primary_surfaces_with_applied_scope, AggregatedActiveReviewSnapshotV1,
    AppliedSupportedSetContextV1, BackendEvidenceSnapshotV1, CanonicalReadinessAuthorityStatusV2,
    GateStatus, OpsError, ReviewabilityAggregateReadinessV1, StrictEvidenceContextV1,
    StrictEvidenceSnapshotV1, StrictEvidenceStatusV1, V0GateOverallStatus, V0GateReportV1,
    V1GateOverallStatus, V1GateReportV1, V2GateOverallStatus, V2GateReportV1, V3GateOverallStatus,
    V3GateReportV1,
};

const CODE_CAP: usize = 12;

#[derive(Debug, Clone)]
struct SignoffReductionInputs<'a> {
    snapshot: Option<&'a BackendEvidenceSnapshotV1>,
    operator: Option<&'a ConsolidatedOperatorReportV1>,
    gates: GateInputs,
    strict_snapshot: &'a StrictEvidenceSnapshotV1,
    active_review_snapshot: Option<&'a AggregatedActiveReviewSnapshotV1>,
    applied_scope: &'a AppliedSupportedSetContextV1,
    policy: &'a SignoffPolicyV1,
}

#[derive(Debug, Clone)]
struct GateInputs {
    v0: Option<V0GateReportV1>,
    v1: Option<V1GateReportV1>,
    v2: Option<V2GateReportV1>,
    v3: Option<V3GateReportV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SignoffDecisionStateV1 {
    ReadyForShadow,
    ReadyForActiveReview,
    NotReady,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GateReportDigestsV1 {
    pub v0: String,
    pub v1: String,
    pub v2: String,
    pub v3: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OperatorSignoffDecisionV1 {
    pub schema_version: u16,
    pub decision: SignoffDecisionStateV1,
    pub supported_slot_set_digest: String,
    pub policy_graph_digest_prefix: String,
    pub manifest_digest_prefix: String,
    pub evidence_snapshot_digest_prefix: String,
    pub active_review_snapshot_digest_prefix: Option<String>,
    pub operator_report_digest_prefix: String,
    pub applied_supported_set_digest_prefix: String,
    pub applied_context_digest_prefix: String,
    pub reviewability_reduction_digest_prefix: String,
    #[serde(default)]
    pub canonical_readiness_spine_digest_prefix: String,
    #[serde(default)]
    pub canonical_readiness_authority_digest_prefix: String,
    #[serde(default)]
    pub canonical_governance_entry_digest_prefix: String,
    #[serde(default)]
    pub final_governance_consumer_authority_digest_prefix: String,
    #[serde(default)]
    pub governance_residual_sweep_digest_prefix: String,
    pub gate_report_digests: GateReportDigestsV1,
    pub reasons: Vec<String>,
    pub remediation_codes: Vec<String>,
    #[serde(default)]
    pub canonical_remediation_codes: Vec<String>,
    pub decision_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SignoffPolicyV1 {
    pub require_v0_gate_pass: bool,
    pub require_v1_gate_pass: bool,
    pub require_v2_gate_pass: bool,
    pub require_v3_gate_pass: bool,
    pub block_on_strict_fail: bool,
    pub block_on_health_fail: bool,
    pub block_on_severe_alerts: bool,
    pub block_on_drift_severe: bool,
}

impl SignoffPolicyV1 {
    pub fn from_profile(profile: &str) -> Self {
        if profile.eq_ignore_ascii_case("dev") {
            return Self {
                require_v0_gate_pass: true,
                require_v1_gate_pass: true,
                require_v2_gate_pass: true,
                require_v3_gate_pass: true,
                block_on_strict_fail: true,
                block_on_health_fail: true,
                block_on_severe_alerts: true,
                block_on_drift_severe: true,
            };
        }
        Self {
            require_v0_gate_pass: true,
            require_v1_gate_pass: true,
            require_v2_gate_pass: true,
            require_v3_gate_pass: true,
            block_on_strict_fail: true,
            block_on_health_fail: true,
            block_on_severe_alerts: true,
            block_on_drift_severe: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OperatorSignoffArgs {
    pub run_id: Option<String>,
    pub latest: bool,
    pub profile: String,
}

pub fn operator_signoff(
    workdir: &Path,
    args: &OperatorSignoffArgs,
    out: &Path,
) -> Result<OperatorSignoffDecisionV1, OpsError> {
    let out_root = PathBuf::from("./out");
    let snapshot = maybe_read_json::<BackendEvidenceSnapshotV1>(&discover_report(
        &out_root,
        "backend_evidence_snapshot.json",
        args,
    ));
    let operator = maybe_read_json::<ConsolidatedOperatorReportV1>(&discover_report(
        &out_root,
        "operator_report.json",
        args,
    ));
    let active_review_snapshot = maybe_read_json::<AggregatedActiveReviewSnapshotV1>(
        &discover_report(&out_root, "active_review_snapshot.json", args),
    );
    let v0 =
        maybe_read_json::<V0GateReportV1>(&discover_report(&out_root, "v0_gate_report.json", args));
    let v1 =
        maybe_read_json::<V1GateReportV1>(&discover_report(&out_root, "v1_gate_report.json", args));
    let v2 =
        maybe_read_json::<V2GateReportV1>(&discover_report(&out_root, "v2_gate_report.json", args));
    let v3 =
        maybe_read_json::<V3GateReportV1>(&discover_report(&out_root, "v3_gate_report.json", args));

    let applied_scope = load_applied_supported_set_context_v1(workdir)?;
    let governance_surfaces_ok = match (snapshot.as_ref(), active_review_snapshot.as_ref()) {
        (Some(snapshot), Some(active_review)) => {
            validate_governance_primary_surfaces_from_workdir(workdir, snapshot, active_review)
                .is_ok()
        }
        _ => false,
    };

    let policy = SignoffPolicyV1::from_profile(&args.profile);
    let strict_snapshot = resolve_strict_evidence(
        &out_root,
        &StrictEvidenceContextV1 {
            run_id: args.run_id.clone(),
            latest: args.latest,
            strict_required: policy.block_on_strict_fail,
            expected_policy_graph_digest_prefix: snapshot
                .as_ref()
                .map(|s| s.policy_graph_digest_prefix.clone())
                .or_else(|| {
                    operator
                        .as_ref()
                        .and_then(|o| o.policy_graph_digest_prefix.clone())
                }),
            expected_manifest_digest_prefix: snapshot
                .as_ref()
                .map(|s| s.manifest_digest_prefix.clone())
                .or_else(|| {
                    operator
                        .as_ref()
                        .and_then(|o| o.manifest_digest_prefix.clone())
                }),
            expected_supported_slot_set_digest_prefix: snapshot
                .as_ref()
                .map(|s| s.supported_slot_set_digest.clone()),
        },
    );
    let mut decision = reduce_signoff(SignoffReductionInputs {
        snapshot: snapshot.as_ref(),
        operator: operator.as_ref(),
        gates: GateInputs { v0, v1, v2, v3 },
        strict_snapshot: &strict_snapshot,
        active_review_snapshot: active_review_snapshot.as_ref(),
        applied_scope: &applied_scope,
        policy: &policy,
    })?;

    if !governance_surfaces_ok {
        decision.decision = SignoffDecisionStateV1::NotReady;
        if !decision
            .reasons
            .iter()
            .any(|r| r == "SIGNOFF_BLOCK_GOVERNANCE_SURFACES_MISMATCH")
        {
            decision
                .reasons
                .push("SIGNOFF_BLOCK_GOVERNANCE_SURFACES_MISMATCH".to_string());
            decision.reasons.sort();
            decision.reasons.dedup();
        }
        if !decision
            .remediation_codes
            .iter()
            .any(|r| r == "run_governance_surfaces_check")
        {
            decision
                .remediation_codes
                .push("run_governance_surfaces_check".to_string());
            decision.remediation_codes.sort();
            decision.remediation_codes.dedup();
        }
        decision.canonical_remediation_codes =
            merge_canonical_remediations(decision.remediation_codes.iter(), CODE_CAP);
        decision.decision_digest = decision_digest(&decision)?;
    }
    if decision.canonical_governance_entry_digest_prefix.is_empty() {
        decision.canonical_governance_entry_digest_prefix = "MISSING".to_string();
    }
    decision.final_governance_consumer_authority_digest_prefix =
        read_final_governance_prefix(workdir, "out/final_governance_consumer_sweep.json");
    decision.governance_residual_sweep_digest_prefix = read_sweep_digest_prefix(
        workdir,
        "out/governance_residual_sweep.json",
        "sweep_digest",
    );
    decision.decision_digest = decision_digest(&decision)?;

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_string_pretty(&decision)?)?;

    let _ = workdir;
    Ok(decision)
}

pub fn operator_signoff_text(report: &OperatorSignoffDecisionV1) -> String {
    format!(
        "decision={:?}\nprimary_reasons={}\nremediation={}\nnext=cargo run -p ucf-ops -- operator signoff --out ./out/operator_signoff.json",
        report.decision,
        if report.reasons.is_empty() {
            "none".to_string()
        } else {
            report.reasons.join(",")
        },
        if report.remediation_codes.is_empty() {
            "none".to_string()
        } else {
            report.remediation_codes.join(",")
        }
    )
}

fn reduce_signoff(
    inputs: SignoffReductionInputs<'_>,
) -> Result<OperatorSignoffDecisionV1, OpsError> {
    let SignoffReductionInputs {
        snapshot,
        active_review_snapshot,
        operator,
        gates: GateInputs { v0, v1, v2, v3 },
        strict_snapshot,
        applied_scope,
        policy,
    } = inputs;

    let mut reasons = BTreeSet::new();
    let mut remediation = BTreeSet::new();

    let Some(snapshot) = snapshot else {
        reasons.insert("SIGNOFF_BLOCK_EVIDENCE_SNAPSHOT_MISSING".to_string());
        remediation.insert("run_backend_evidence_snapshot".to_string());
        return build_not_ready_minimal(reasons, remediation, v0, v1, v2, v3, applied_scope);
    };
    let Some(operator) = operator else {
        reasons.insert("SIGNOFF_BLOCK_OPERATOR_REPORT_MISSING".to_string());
        remediation.insert("run_operator_report".to_string());
        return build_not_ready_from_snapshot(
            snapshot,
            reasons,
            remediation,
            v0,
            v1,
            v2,
            v3,
            applied_scope,
        );
    };

    let snapshot_slots = snapshot
        .slots
        .iter()
        .map(|slot| slot.slot_id.clone())
        .collect::<Vec<_>>();
    if snapshot_slots != applied_scope.slots {
        reasons.insert("SIGNOFF_SCOPE_MISMATCH".to_string());
        remediation.insert("run_models_applied_scope_check".to_string());
    }

    check_gate_v0(
        v0.as_ref(),
        policy.require_v0_gate_pass,
        &mut reasons,
        &mut remediation,
    );
    check_gate_v1(
        v1.as_ref(),
        policy.require_v1_gate_pass,
        &mut reasons,
        &mut remediation,
    );
    check_gate_v2(
        v2.as_ref(),
        policy.require_v2_gate_pass,
        &mut reasons,
        &mut remediation,
    );
    check_gate_v3(
        v3.as_ref(),
        policy.require_v3_gate_pass,
        &mut reasons,
        &mut remediation,
    );

    if policy.block_on_health_fail
        && matches!(
            operator.sections.health_section.status,
            OperatorStatus::Fail | OperatorStatus::Missing
        )
    {
        reasons.insert("SIGNOFF_BLOCK_HEALTH".to_string());
        remediation.insert("run_health_check".to_string());
    }

    if policy.block_on_strict_fail {
        let strict_block = operator_block_from_strict(strict_snapshot);
        if matches!(
            strict_snapshot.strict_status,
            StrictEvidenceStatusV1::Fail | StrictEvidenceStatusV1::Missing
        ) {
            reasons.insert(
                strict_block
                    .primary_reason_code
                    .unwrap_or_else(|| "SIGNOFF_BLOCK_STRICT".to_string()),
            );
            remediation.extend(strict_block.remediation_codes);
        }
    }

    let mut reviewability_reduction_digest_prefix = "MISSING".to_string();
    let mut canonical_spine_prefix = "MISSING".to_string();
    let mut canonical_readiness_authority_digest_prefix = "MISSING".to_string();

    let mut shadow_ready = snapshot
        .slots
        .iter()
        .all(|slot| slot.readiness.probe_ready && slot.readiness.shadow_ready);
    let mut any_active = snapshot
        .slots
        .iter()
        .any(|slot| slot.readiness.active_eligible);

    if let Some(active) = active_review_snapshot
        .filter(|r| r.supported_slot_set_digest == snapshot.supported_slot_set_digest)
    {
        let truths =
            derive_slot_reviewability_truths(applied_scope, snapshot, active, strict_snapshot)?;
        let reduction = reduce_reviewability(applied_scope, &truths)?;
        reviewability_reduction_digest_prefix = prefix16(&reduction.reduction_digest);
        if let Ok(surfaces) =
            validate_governance_primary_surfaces_with_applied_scope(snapshot, active, applied_scope)
        {
            if let Ok(entry) = derive_canonical_governance_entry(applied_scope, &surfaces) {
                if let Ok(spine) = derive_canonical_readiness_spine(
                    applied_scope,
                    &entry,
                    &truths,
                    &reduction,
                    Some(&active.snapshot_digest),
                    None,
                    None,
                    None,
                ) {
                    if let Ok(spine) =
                        require_canonical_readiness_spine(applied_scope, &entry, Some(&spine))
                    {
                        canonical_spine_prefix = prefix16(&spine.spine_digest);
                        let authority = derive_canonical_readiness_authority_v2(
                            &applied_scope.applied_set_digest_prefix,
                            &prefix16(&entry.authority_digest),
                            &canonical_spine_prefix,
                            4,
                            CanonicalReadinessAuthorityStatusV2::Pass,
                        );
                        canonical_readiness_authority_digest_prefix =
                            prefix16(&authority.authority_digest);
                    } else {
                        reasons.insert("CANONICAL_READINESS_SPINE_REQUIRED".to_string());
                        remediation.insert("run_readiness_spine_sweep".to_string());
                    }
                }
            }
        }
        shadow_ready = truths
            .iter()
            .all(|slot| slot.probe_ready && slot.shadow_ready);
        any_active = !matches!(
            reduction.aggregate_readiness,
            ReviewabilityAggregateReadinessV1::NoneReviewable
        );
    }

    if !shadow_ready {
        reasons.insert("SIGNOFF_BLOCK_SHADOW_NOT_READY".to_string());
        remediation.insert("run_models_eligibility".to_string());
    }

    if active_review_snapshot.is_none() {
        reasons.insert("SIGNOFF_MISSING_APPLIED_SET".to_string());
        remediation.insert("run_models_active_review_snapshot".to_string());
    } else if let Some(active) = active_review_snapshot {
        if active
            .slots
            .iter()
            .any(|slot| !applied_scope.slots.contains(&slot.slot_id))
        {
            reasons.insert("SIGNOFF_EXTRA_SLOT_EVIDENCE_IGNORED_OR_BLOCKED".to_string());
            remediation.insert("run_models_applied_scope_check".to_string());
        }
    }

    let severe_alerts = matches!(
        operator.sections.alerts_section.status,
        OperatorStatus::Fail
    );
    if policy.block_on_severe_alerts && severe_alerts {
        reasons.insert("SIGNOFF_BLOCK_ALERT_SEVERE".to_string());
        remediation.insert("inspect_active_alerts".to_string());
    }

    let severe_drift = operator.sections.drift_section.slots.iter().any(|slot| {
        matches!(slot.drift_status, OperatorStatus::Fail) || slot.severe_alarm_count > 0
    }) || matches!(
        operator.sections.drift_section.status,
        OperatorStatus::Missing
    );
    if policy.block_on_drift_severe && severe_drift {
        reasons.insert("SIGNOFF_BLOCK_DRIFT_SEVERE".to_string());
        remediation.insert("run_drift_report".to_string());
    }

    if reasons.is_empty() && shadow_ready && any_active {
        reasons.insert("SIGNOFF_READY_ACTIVE_REVIEW".to_string());
        return build_decision(
            SignoffDecisionStateV1::ReadyForActiveReview,
            snapshot,
            active_review_snapshot,
            operator,
            v0,
            v1,
            v2,
            v3,
            reasons,
            remediation,
            applied_scope,
            &reviewability_reduction_digest_prefix,
            &canonical_spine_prefix,
            &canonical_readiness_authority_digest_prefix,
        );
    }

    if reasons.is_empty() && shadow_ready {
        reasons.insert("SIGNOFF_READY_SHADOW".to_string());
        return build_decision(
            SignoffDecisionStateV1::ReadyForShadow,
            snapshot,
            active_review_snapshot,
            operator,
            v0,
            v1,
            v2,
            v3,
            reasons,
            remediation,
            applied_scope,
            &reviewability_reduction_digest_prefix,
            &canonical_spine_prefix,
            &canonical_readiness_authority_digest_prefix,
        );
    }

    if shadow_ready && !any_active {
        reasons.insert("SIGNOFF_BLOCK_ACTIVE_NOT_ELIGIBLE".to_string());
        remediation.insert("run_models_active_check".to_string());
    }

    build_decision(
        SignoffDecisionStateV1::NotReady,
        snapshot,
        active_review_snapshot,
        operator,
        v0,
        v1,
        v2,
        v3,
        reasons,
        remediation,
        applied_scope,
        &reviewability_reduction_digest_prefix,
        &canonical_spine_prefix,
        &canonical_readiness_authority_digest_prefix,
    )
}

fn build_not_ready_minimal(
    reasons: BTreeSet<String>,
    remediation: BTreeSet<String>,
    v0: Option<V0GateReportV1>,
    v1: Option<V1GateReportV1>,
    v2: Option<V2GateReportV1>,
    v3: Option<V3GateReportV1>,
    applied_scope: &AppliedSupportedSetContextV1,
) -> Result<OperatorSignoffDecisionV1, OpsError> {
    let mut out = OperatorSignoffDecisionV1 {
        schema_version: 1,
        decision: SignoffDecisionStateV1::NotReady,
        supported_slot_set_digest: "MISSING".to_string(),
        policy_graph_digest_prefix: "MISSING".to_string(),
        manifest_digest_prefix: "MISSING".to_string(),
        evidence_snapshot_digest_prefix: "MISSING".to_string(),
        active_review_snapshot_digest_prefix: None,
        operator_report_digest_prefix: "MISSING".to_string(),
        applied_supported_set_digest_prefix: applied_scope.applied_set_digest_prefix.clone(),
        applied_context_digest_prefix: prefix16(&applied_scope.context_digest),
        reviewability_reduction_digest_prefix: "MISSING".to_string(),
        canonical_readiness_spine_digest_prefix: "MISSING".to_string(),
        canonical_readiness_authority_digest_prefix: "MISSING".to_string(),
        canonical_governance_entry_digest_prefix: "MISSING".to_string(),
        final_governance_consumer_authority_digest_prefix: "MISSING".to_string(),
        governance_residual_sweep_digest_prefix: "MISSING".to_string(),
        gate_report_digests: GateReportDigestsV1 {
            v0: digest_opt(v0.as_ref())?,
            v1: digest_opt(v1.as_ref())?,
            v2: digest_opt(v2.as_ref())?,
            v3: digest_opt(v3.as_ref())?,
        },
        reasons: bound_codes(reasons),
        remediation_codes: bound_codes(remediation),
        canonical_remediation_codes: Vec::new(),
        decision_digest: String::new(),
    };
    out.canonical_remediation_codes =
        merge_canonical_remediations(out.remediation_codes.iter(), CODE_CAP);
    out.decision_digest = decision_digest(&out)?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn build_not_ready_from_snapshot(
    snapshot: &BackendEvidenceSnapshotV1,
    reasons: BTreeSet<String>,
    remediation: BTreeSet<String>,
    v0: Option<V0GateReportV1>,
    v1: Option<V1GateReportV1>,
    v2: Option<V2GateReportV1>,
    v3: Option<V3GateReportV1>,
    applied_scope: &AppliedSupportedSetContextV1,
) -> Result<OperatorSignoffDecisionV1, OpsError> {
    let mut out = OperatorSignoffDecisionV1 {
        schema_version: 1,
        decision: SignoffDecisionStateV1::NotReady,
        supported_slot_set_digest: snapshot.supported_slot_set_digest.clone(),
        policy_graph_digest_prefix: snapshot.policy_graph_digest_prefix.clone(),
        manifest_digest_prefix: snapshot.manifest_digest_prefix.clone(),
        evidence_snapshot_digest_prefix: prefix16(&snapshot.snapshot_digest),
        active_review_snapshot_digest_prefix: None,
        operator_report_digest_prefix: "MISSING".to_string(),
        applied_supported_set_digest_prefix: applied_scope.applied_set_digest_prefix.clone(),
        applied_context_digest_prefix: prefix16(&applied_scope.context_digest),
        reviewability_reduction_digest_prefix: "MISSING".to_string(),
        canonical_readiness_spine_digest_prefix: "MISSING".to_string(),
        canonical_readiness_authority_digest_prefix: "MISSING".to_string(),
        canonical_governance_entry_digest_prefix: "MISSING".to_string(),
        final_governance_consumer_authority_digest_prefix: "MISSING".to_string(),
        governance_residual_sweep_digest_prefix: "MISSING".to_string(),
        gate_report_digests: GateReportDigestsV1 {
            v0: digest_opt(v0.as_ref())?,
            v1: digest_opt(v1.as_ref())?,
            v2: digest_opt(v2.as_ref())?,
            v3: digest_opt(v3.as_ref())?,
        },
        reasons: bound_codes(reasons),
        remediation_codes: bound_codes(remediation),
        canonical_remediation_codes: Vec::new(),
        decision_digest: String::new(),
    };
    out.canonical_remediation_codes =
        merge_canonical_remediations(out.remediation_codes.iter(), CODE_CAP);
    out.decision_digest = decision_digest(&out)?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn build_decision(
    decision: SignoffDecisionStateV1,
    snapshot: &BackendEvidenceSnapshotV1,
    active_review_snapshot: Option<&AggregatedActiveReviewSnapshotV1>,
    operator: &ConsolidatedOperatorReportV1,
    v0: Option<V0GateReportV1>,
    v1: Option<V1GateReportV1>,
    v2: Option<V2GateReportV1>,
    v3: Option<V3GateReportV1>,
    reasons: BTreeSet<String>,
    remediation: BTreeSet<String>,
    applied_scope: &AppliedSupportedSetContextV1,
    reviewability_reduction_digest_prefix: &str,
    canonical_readiness_spine_digest_prefix: &str,
    canonical_readiness_authority_digest_prefix: &str,
) -> Result<OperatorSignoffDecisionV1, OpsError> {
    let mut out = OperatorSignoffDecisionV1 {
        schema_version: 1,
        decision,
        supported_slot_set_digest: snapshot.supported_slot_set_digest.clone(),
        policy_graph_digest_prefix: snapshot.policy_graph_digest_prefix.clone(),
        manifest_digest_prefix: snapshot.manifest_digest_prefix.clone(),
        evidence_snapshot_digest_prefix: prefix16(&snapshot.snapshot_digest),
        active_review_snapshot_digest_prefix: active_review_snapshot
            .map(|v| prefix16(&v.snapshot_digest)),
        operator_report_digest_prefix: prefix16(&operator.report_digest),
        applied_supported_set_digest_prefix: applied_scope.applied_set_digest_prefix.clone(),
        applied_context_digest_prefix: prefix16(&applied_scope.context_digest),
        reviewability_reduction_digest_prefix: reviewability_reduction_digest_prefix.to_string(),
        canonical_readiness_spine_digest_prefix: canonical_readiness_spine_digest_prefix
            .to_string(),
        canonical_readiness_authority_digest_prefix: canonical_readiness_authority_digest_prefix
            .to_string(),
        canonical_governance_entry_digest_prefix: "MISSING".to_string(),
        final_governance_consumer_authority_digest_prefix: "MISSING".to_string(),
        governance_residual_sweep_digest_prefix: "MISSING".to_string(),
        gate_report_digests: GateReportDigestsV1 {
            v0: digest_opt(v0.as_ref())?,
            v1: digest_opt(v1.as_ref())?,
            v2: digest_opt(v2.as_ref())?,
            v3: digest_opt(v3.as_ref())?,
        },
        reasons: bound_codes(reasons),
        remediation_codes: bound_codes(remediation),
        canonical_remediation_codes: Vec::new(),
        decision_digest: String::new(),
    };
    out.canonical_remediation_codes =
        merge_canonical_remediations(out.remediation_codes.iter(), CODE_CAP);
    out.decision_digest = decision_digest(&out)?;
    Ok(out)
}

fn check_gate_v0(
    gate: Option<&V0GateReportV1>,
    required: bool,
    reasons: &mut BTreeSet<String>,
    remediation: &mut BTreeSet<String>,
) {
    if !required {
        return;
    }
    let Some(gate) = gate else {
        reasons.insert("SIGNOFF_BLOCK_GATE_V0_MISSING".to_string());
        remediation.insert("run_v0_gate".to_string());
        return;
    };
    if gate.overall_status != V0GateOverallStatus::Pass {
        reasons.insert("SIGNOFF_BLOCK_GATE_V0".to_string());
        remediation.insert("run_v0_gate".to_string());
    }
}

fn check_gate_v1(
    gate: Option<&V1GateReportV1>,
    required: bool,
    reasons: &mut BTreeSet<String>,
    remediation: &mut BTreeSet<String>,
) {
    if !required {
        return;
    }
    let Some(gate) = gate else {
        reasons.insert("SIGNOFF_BLOCK_GATE_V1_MISSING".to_string());
        remediation.insert("run_v1_gate".to_string());
        return;
    };
    if gate.overall_status != V1GateOverallStatus::Pass {
        reasons.insert("SIGNOFF_BLOCK_GATE_V1".to_string());
        remediation.insert("run_v1_gate".to_string());
    }
}

fn check_gate_v2(
    gate: Option<&V2GateReportV1>,
    required: bool,
    reasons: &mut BTreeSet<String>,
    remediation: &mut BTreeSet<String>,
) {
    if !required {
        return;
    }
    let Some(gate) = gate else {
        reasons.insert("SIGNOFF_BLOCK_GATE_V2_MISSING".to_string());
        remediation.insert("run_v2_gate".to_string());
        return;
    };
    if gate.overall_status != V2GateOverallStatus::Pass {
        reasons.insert("SIGNOFF_BLOCK_GATE_V2".to_string());
        remediation.insert("run_v2_gate".to_string());
    }
}

fn check_gate_v3(
    gate: Option<&V3GateReportV1>,
    required: bool,
    reasons: &mut BTreeSet<String>,
    remediation: &mut BTreeSet<String>,
) {
    if !required {
        return;
    }
    let Some(gate) = gate else {
        reasons.insert("SIGNOFF_BLOCK_GATE_V3_MISSING".to_string());
        remediation.insert("run_v3_gate".to_string());
        return;
    };
    if gate.overall_status != V3GateOverallStatus::Pass {
        reasons.insert("SIGNOFF_BLOCK_GATE_V3".to_string());
        remediation.insert("run_v3_gate".to_string());
    }
    if gate.checks.iter().any(|c| c.status == GateStatus::Fail) {
        reasons.insert("SIGNOFF_BLOCK_GATE_V3".to_string());
    }
}

fn digest_opt<T: Serialize>(value: Option<&T>) -> Result<String, OpsError> {
    match value {
        Some(v) => Ok(prefix16(&crate::sha256_hex(&serde_json::to_vec(v)?))),
        None => Ok("MISSING".to_string()),
    }
}

fn prefix16(value: &str) -> String {
    value.chars().take(16).collect()
}

fn bound_codes(codes: BTreeSet<String>) -> Vec<String> {
    codes.into_iter().take(CODE_CAP).collect()
}

fn decision_digest(report: &OperatorSignoffDecisionV1) -> Result<String, OpsError> {
    let mut cloned = report.clone();
    cloned.decision_digest.clear();
    Ok(crate::sha256_hex(&serde_json::to_vec(&cloned)?))
}

fn discover_report(out_root: &Path, file: &str, args: &OperatorSignoffArgs) -> Option<PathBuf> {
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

fn read_final_governance_prefix(workdir: &Path, rel_path: &str) -> String {
    let Ok(bytes) = fs::read(workdir.join(rel_path)) else {
        return "MISSING".to_string();
    };
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return "MISSING".to_string();
    };
    value
        .get("authority")
        .and_then(|authority| authority.get("authority_digest"))
        .and_then(serde_json::Value::as_str)
        .map(prefix16)
        .unwrap_or_else(|| "MISSING".to_string())
}

fn read_sweep_digest_prefix(workdir: &Path, rel_path: &str, field: &str) -> String {
    let Ok(bytes) = fs::read(workdir.join(rel_path)) else {
        return "MISSING".to_string();
    };
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return "MISSING".to_string();
    };
    value
        .get("sweep")
        .and_then(|sweep| sweep.get(field))
        .and_then(serde_json::Value::as_str)
        .map(prefix16)
        .unwrap_or_else(|| "MISSING".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models_lifecycle::{
        AppliedSupportedSetContextV1, BackendEvidenceSlotDenialsV1, BackendEvidenceSlotEvidenceV1,
        BackendEvidenceSlotReadinessV1, BackendEvidenceSlotSnapshotV1, BackendSupportMatrixV1,
        BurnResolutionStatusV1, BurnSupportResolutionV1, DriftStatusV1, EvidenceDenialCodeV1,
        SupportedRealSlotSetExecutionDecisionV2,
    };
    use crate::operator_report::{DriftSlotSummary, EligibilitySlotSummary, GateStatusSummary};
    use crate::operator_report::{
        NormalizedAlertsSection, NormalizedDriftSection, NormalizedEligibilitySection,
        NormalizedGatesSection, NormalizedHealthSection, NormalizedStrictSection,
        OperatorSectionsV1,
    };

    fn snapshot(active_eligible: bool) -> BackendEvidenceSnapshotV1 {
        BackendEvidenceSnapshotV1 {
            schema_version: 1,
            supported_slot_set_digest: "slotset123".to_string(),
            policy_graph_digest_prefix: "policy123".to_string(),
            manifest_digest_prefix: "manifest123".to_string(),
            slots: vec![
                BackendEvidenceSlotSnapshotV1 {
                    slot_id: "world".to_string(),
                    target_hash_prefix: "w".to_string(),
                    backend_support: BackendSupportMatrixV1 {
                        stub: crate::BackendSupportStateV1::Supported,
                        candle: crate::BackendSupportStateV1::Supported,
                        burn: crate::BackendSupportStateV1::Unsupported,
                    },
                    evidence: BackendEvidenceSlotEvidenceV1 {
                        latest_probe_report_digest_prefix: "p1".to_string(),
                        latest_compare_window_digest_prefix: "c1".to_string(),
                        latest_shadow_ready_digest_prefix: "s1".to_string(),
                        latest_active_evidence_digest_prefix: "a1".to_string(),
                        latest_drift_status: DriftStatusV1::Ok,
                        freshness_probe_age_ticks: Some(1),
                        freshness_compare_age_ticks: Some(1),
                        freshness_no_impact_age_ticks: Some(1),
                        freshness_drift_status_age_ticks: Some(1),
                        hash_consistency_ok: true,
                    },
                    readiness: BackendEvidenceSlotReadinessV1 {
                        probe_ready: true,
                        shadow_ready: true,
                        active_eligible,
                    },
                    denials: BackendEvidenceSlotDenialsV1 {
                        probe: None,
                        shadow: None,
                        active: None,
                    },
                    remediation_codes: vec![],
                    canonical_remediation_codes: vec![],
                    burn_resolution: BurnSupportResolutionV1 {
                        slot_id: "world_jepa".to_string(),
                        resolution: BurnResolutionStatusV1::BurnClosedUnsupported,
                        support_state: crate::OptionalBackendSupportStateV1::Unsupported,
                        rationale_codes: vec!["BURN_SLOT_FORMALLY_UNSUPPORTED".to_string()],
                        evidence_digest: "br1".to_string(),
                    },
                },
                BackendEvidenceSlotSnapshotV1 {
                    slot_id: "sae".to_string(),
                    target_hash_prefix: "s".to_string(),
                    backend_support: BackendSupportMatrixV1 {
                        stub: crate::BackendSupportStateV1::Supported,
                        candle: crate::BackendSupportStateV1::Supported,
                        burn: crate::BackendSupportStateV1::Unsupported,
                    },
                    evidence: BackendEvidenceSlotEvidenceV1 {
                        latest_probe_report_digest_prefix: "p2".to_string(),
                        latest_compare_window_digest_prefix: "c2".to_string(),
                        latest_shadow_ready_digest_prefix: "s2".to_string(),
                        latest_active_evidence_digest_prefix: "a2".to_string(),
                        latest_drift_status: DriftStatusV1::Ok,
                        freshness_probe_age_ticks: Some(1),
                        freshness_compare_age_ticks: Some(1),
                        freshness_no_impact_age_ticks: Some(1),
                        freshness_drift_status_age_ticks: Some(1),
                        hash_consistency_ok: true,
                    },
                    readiness: BackendEvidenceSlotReadinessV1 {
                        probe_ready: true,
                        shadow_ready: true,
                        active_eligible: false,
                    },
                    denials: BackendEvidenceSlotDenialsV1 {
                        probe: None,
                        shadow: None,
                        active: Some(EvidenceDenialCodeV1::ActiveNotEnabled),
                    },
                    remediation_codes: vec![],
                    canonical_remediation_codes: vec![],
                    burn_resolution: BurnSupportResolutionV1 {
                        slot_id: "sae".to_string(),
                        resolution: BurnResolutionStatusV1::BurnClosedUnsupported,
                        support_state: crate::OptionalBackendSupportStateV1::NotConfigured,
                        rationale_codes: vec!["BURN_SHADOW_NOT_CONFIGURED".to_string()],
                        evidence_digest: "br2".to_string(),
                    },
                },
            ],
            snapshot_digest: "snapshotdigest123456".to_string(),
        }
    }

    fn operator_report() -> ConsolidatedOperatorReportV1 {
        ConsolidatedOperatorReportV1 {
            schema_version: 1,
            generated_at: 1,
            overall_status: OperatorStatus::Ok,
            run_id: Some("r1".to_string()),
            policy_graph_digest_prefix: Some("pg".to_string()),
            manifest_digest_prefix: Some("mg".to_string()),
            sections: OperatorSectionsV1 {
                health_section: NormalizedHealthSection {
                    status: OperatorStatus::Ok,
                    strict_mode_enabled: Some(true),
                    last_tick_age_ms: Some(1),
                    emergency_active: Some(false),
                    evidence_digest_prefixes: vec![],
                    remediation_codes: vec![],
                },
                eligibility_section: NormalizedEligibilitySection {
                    status: OperatorStatus::Warn,
                    slots: vec![EligibilitySlotSummary {
                        slot_id: "world".to_string(),
                        probe_ready: true,
                        shadow_ready: true,
                        active_eligible: false,
                        primary_denial_reason: None,
                    }],
                    evidence_digest_prefixes: vec![],
                    remediation_codes: vec![],
                },
                drift_section: NormalizedDriftSection {
                    status: OperatorStatus::Ok,
                    slots: vec![DriftSlotSummary {
                        slot_id: "world".to_string(),
                        drift_status: OperatorStatus::Ok,
                        severe_alarm_count: 0,
                    }],
                    evidence_digest_prefixes: vec![],
                    remediation_codes: vec![],
                },
                alerts_section: NormalizedAlertsSection {
                    status: OperatorStatus::Ok,
                    active_alert_count: 0,
                    top_active_alerts: vec![],
                    evidence_digest_prefixes: vec![],
                    remediation_codes: vec![],
                },
                strict_section: NormalizedStrictSection {
                    status: OperatorStatus::Ok,
                    strict_status: StrictEvidenceStatusV1::Pass,
                    primary_denial_code: None,
                    strict_report_digest_prefix: Some("strictdigest".to_string()),
                    failing_check_ids: vec![],
                    evidence_digest_prefixes: vec![],
                    remediation_codes: vec![],
                },
                gates_section: NormalizedGatesSection {
                    status: OperatorStatus::Ok,
                    gates: vec![GateStatusSummary {
                        gate_id: "v3".to_string(),
                        status: OperatorStatus::Ok,
                    }],
                    evidence_digest_prefixes: vec![],
                    remediation_codes: vec![],
                },
            },
            remediation_codes: vec![],
            canonical_remediation_codes: vec![],
            report_digest: "operatordigest123456".to_string(),
        }
    }

    fn pass_v0() -> V0GateReportV1 {
        V0GateReportV1 {
            schema_version: 1,
            overall_status: V0GateOverallStatus::Pass,
            checks: vec![],
        }
    }

    fn pass_v1() -> V1GateReportV1 {
        V1GateReportV1 {
            schema_version: 1,
            overall_status: V1GateOverallStatus::Pass,
            checks: vec![],
        }
    }

    fn pass_v2() -> V2GateReportV1 {
        V2GateReportV1 {
            schema_version: 1,
            overall_status: V2GateOverallStatus::Pass,
            checks: vec![],
        }
    }

    fn pass_v3() -> V3GateReportV1 {
        V3GateReportV1 {
            schema_version: 1,
            overall_status: V3GateOverallStatus::Pass,
            checks: vec![],
        }
    }

    fn strict_snapshot(status: StrictEvidenceStatusV1) -> StrictEvidenceSnapshotV1 {
        StrictEvidenceSnapshotV1 {
            schema_version: 1,
            strict_mode_enabled: true,
            strict_status: status,
            strict_report_digest_prefix: Some("strictdigest".to_string()),
            policy_graph_digest_prefix: Some("pg".to_string()),
            manifest_digest_prefix: Some("mg".to_string()),
            supported_slot_set_digest_prefix: Some("slotset123".to_string()),
            primary_denial_code: Some("STRICT_FAIL".to_string()),
            remediation_codes: vec!["run_strict_check".to_string()],
            failing_check_ids: vec!["strict_mode".to_string()],
            snapshot_digest: "strictsnapshot".to_string(),
        }
    }

    fn applied_scope() -> AppliedSupportedSetContextV1 {
        AppliedSupportedSetContextV1 {
            schema_version: 1,
            applied_set_digest_prefix: "slotset123".to_string(),
            slots: vec!["world".to_string(), "sae".to_string()],
            decision: SupportedRealSlotSetExecutionDecisionV2::Frozen,
            previous_set_digest_prefix: "prev".to_string(),
            policy_digest_prefix: "policy".to_string(),
            context_digest: "ctx".repeat(16),
            compatibility_code: None,
        }
    }

    fn active_review_snapshot(active_eligible: bool) -> AggregatedActiveReviewSnapshotV1 {
        AggregatedActiveReviewSnapshotV1 {
            schema_version: 1,
            supported_slot_set_digest: "slotset123".to_string(),
            policy_graph_digest_prefix: "policy123".to_string(),
            manifest_digest_prefix: "manifest123".to_string(),
            slots: vec![
                crate::models_lifecycle::ActiveReviewEvidenceV1 {
                    slot_id: "sae".to_string(),
                    target_hash_prefix: "s".to_string(),
                    manifest_digest_prefix: "manifest123".to_string(),
                    probe_ready: true,
                    shadow_ready: true,
                    active_eligible,
                    strict_blocking: false,
                    drift_blocking: false,
                    alert_blocking: false,
                    primary_denial_code: None,
                    remediation_codes: Vec::new(),
                    contributing_evidence_digests:
                        crate::models_lifecycle::ActiveReviewContributingDigestsV1 {
                            probe_report_digest_prefix: "p".to_string(),
                            shadow_ready_digest_prefix: "s".to_string(),
                            active_evidence_digest_prefix: "a".to_string(),
                            strict_evidence_digest_prefix: "t".to_string(),
                        },
                    burn_resolution: BurnSupportResolutionV1 {
                        slot_id: "sae".to_string(),
                        resolution: BurnResolutionStatusV1::BurnClosedUnsupported,
                        support_state: crate::OptionalBackendSupportStateV1::Unsupported,
                        rationale_codes: vec!["X".to_string()],
                        evidence_digest: "bd1".to_string(),
                    },
                    evidence_digest: "ed1".to_string(),
                },
                crate::models_lifecycle::ActiveReviewEvidenceV1 {
                    slot_id: "world".to_string(),
                    target_hash_prefix: "w".to_string(),
                    manifest_digest_prefix: "manifest123".to_string(),
                    probe_ready: true,
                    shadow_ready: true,
                    active_eligible,
                    strict_blocking: false,
                    drift_blocking: false,
                    alert_blocking: false,
                    primary_denial_code: None,
                    remediation_codes: Vec::new(),
                    contributing_evidence_digests:
                        crate::models_lifecycle::ActiveReviewContributingDigestsV1 {
                            probe_report_digest_prefix: "p".to_string(),
                            shadow_ready_digest_prefix: "s".to_string(),
                            active_evidence_digest_prefix: "a".to_string(),
                            strict_evidence_digest_prefix: "t".to_string(),
                        },
                    burn_resolution: BurnSupportResolutionV1 {
                        slot_id: "world".to_string(),
                        resolution: BurnResolutionStatusV1::BurnClosedUnsupported,
                        support_state: crate::OptionalBackendSupportStateV1::Unsupported,
                        rationale_codes: vec!["X".to_string()],
                        evidence_digest: "bd2".to_string(),
                    },
                    evidence_digest: "ed2".to_string(),
                },
            ],
            overall_review_status:
                crate::models_lifecycle::ActiveReviewOverallStatusV1::AllReviewable,
            signoff_alignment: crate::models_lifecycle::ActiveReviewSignoffAlignmentV1 {
                aligned: true,
                status_code: "ALIGNED".to_string(),
            },
            canonical_governance_entry_digest_prefix: "MISSING".to_string(),
            final_governance_consumer_authority_digest_prefix: "MISSING".to_string(),
            governance_residual_sweep_digest_prefix: "MISSING".to_string(),
            snapshot_digest: "snapshot1111".to_string(),
        }
    }

    fn policy() -> SignoffPolicyV1 {
        SignoffPolicyV1::from_profile("test")
    }

    #[test]
    fn deterministic_reduction_equal_inputs_equal_outputs() {
        let snapshot = snapshot(false);
        let operator = operator_report();
        let a = reduce_signoff(SignoffReductionInputs {
            snapshot: Some(&snapshot),
            operator: Some(&operator),
            gates: GateInputs {
                v0: Some(pass_v0()),
                v1: Some(pass_v1()),
                v2: Some(pass_v2()),
                v3: Some(pass_v3()),
            },
            strict_snapshot: &strict_snapshot(StrictEvidenceStatusV1::Pass),
            active_review_snapshot: Some(&active_review_snapshot(false)),
            applied_scope: &applied_scope(),
            policy: &policy(),
        })
        .expect("decision a");
        let b = reduce_signoff(SignoffReductionInputs {
            snapshot: Some(&snapshot),
            operator: Some(&operator),
            gates: GateInputs {
                v0: Some(pass_v0()),
                v1: Some(pass_v1()),
                v2: Some(pass_v2()),
                v3: Some(pass_v3()),
            },
            strict_snapshot: &strict_snapshot(StrictEvidenceStatusV1::Pass),
            active_review_snapshot: Some(&active_review_snapshot(false)),
            applied_scope: &applied_scope(),
            policy: &policy(),
        })
        .expect("decision b");
        assert_eq!(a, b);
    }

    #[test]
    fn shadow_ready_only_is_ready_for_shadow() {
        let decision = reduce_signoff(SignoffReductionInputs {
            snapshot: Some(&snapshot(false)),
            operator: Some(&operator_report()),
            gates: GateInputs {
                v0: Some(pass_v0()),
                v1: Some(pass_v1()),
                v2: Some(pass_v2()),
                v3: Some(pass_v3()),
            },
            strict_snapshot: &strict_snapshot(StrictEvidenceStatusV1::Pass),
            active_review_snapshot: Some(&active_review_snapshot(false)),
            applied_scope: &applied_scope(),
            policy: &policy(),
        })
        .expect("decision");
        assert_eq!(decision.decision, SignoffDecisionStateV1::ReadyForShadow);
        assert_eq!(decision.reasons, vec!["SIGNOFF_READY_SHADOW"]);
    }

    #[test]
    fn active_eligible_slot_is_ready_for_active_review() {
        let decision = reduce_signoff(SignoffReductionInputs {
            snapshot: Some(&snapshot(true)),
            operator: Some(&operator_report()),
            gates: GateInputs {
                v0: Some(pass_v0()),
                v1: Some(pass_v1()),
                v2: Some(pass_v2()),
                v3: Some(pass_v3()),
            },
            strict_snapshot: &strict_snapshot(StrictEvidenceStatusV1::Pass),
            active_review_snapshot: Some(&active_review_snapshot(true)),
            applied_scope: &applied_scope(),
            policy: &policy(),
        })
        .expect("decision");
        assert_eq!(
            decision.decision,
            SignoffDecisionStateV1::ReadyForActiveReview
        );
        assert_eq!(decision.reasons, vec!["SIGNOFF_READY_ACTIVE_REVIEW"]);
    }

    #[test]
    fn strict_fail_or_v3_fail_is_not_ready() {
        let operator = operator_report();
        let decision = reduce_signoff(SignoffReductionInputs {
            snapshot: Some(&snapshot(true)),
            operator: Some(&operator),
            gates: GateInputs {
                v0: Some(pass_v0()),
                v1: Some(pass_v1()),
                v2: Some(pass_v2()),
                v3: Some(V3GateReportV1 {
                    schema_version: 1,
                    overall_status: V3GateOverallStatus::Fail,
                    checks: vec![],
                }),
            },
            strict_snapshot: &strict_snapshot(StrictEvidenceStatusV1::Fail),
            active_review_snapshot: None,
            applied_scope: &applied_scope(),
            policy: &policy(),
        })
        .expect("decision");
        assert_eq!(decision.decision, SignoffDecisionStateV1::NotReady);
        assert!(decision.reasons.contains(&"STRICT_FAIL".to_string()));
        assert!(decision
            .reasons
            .contains(&"SIGNOFF_BLOCK_GATE_V3".to_string()));
    }

    #[test]
    fn missing_snapshot_fails_closed() {
        let decision = reduce_signoff(SignoffReductionInputs {
            snapshot: None,
            operator: Some(&operator_report()),
            gates: GateInputs {
                v0: Some(pass_v0()),
                v1: Some(pass_v1()),
                v2: Some(pass_v2()),
                v3: Some(pass_v3()),
            },
            strict_snapshot: &strict_snapshot(StrictEvidenceStatusV1::Pass),
            active_review_snapshot: None,
            applied_scope: &applied_scope(),
            policy: &policy(),
        })
        .expect("decision");
        assert_eq!(decision.decision, SignoffDecisionStateV1::NotReady);
        assert_eq!(
            decision.reasons,
            vec!["SIGNOFF_BLOCK_EVIDENCE_SNAPSHOT_MISSING".to_string()]
        );
    }

    #[test]
    fn ambiguous_slot_set_fails_closed() {
        let mut snapshot = snapshot(true);
        snapshot.slots = vec![snapshot.slots[0].clone()];
        let decision = reduce_signoff(SignoffReductionInputs {
            snapshot: Some(&snapshot),
            operator: Some(&operator_report()),
            gates: GateInputs {
                v0: Some(pass_v0()),
                v1: Some(pass_v1()),
                v2: Some(pass_v2()),
                v3: Some(pass_v3()),
            },
            strict_snapshot: &strict_snapshot(StrictEvidenceStatusV1::Pass),
            active_review_snapshot: None,
            applied_scope: &applied_scope(),
            policy: &policy(),
        })
        .expect("decision");
        assert_eq!(decision.decision, SignoffDecisionStateV1::NotReady);
        assert!(decision
            .reasons
            .contains(&"SIGNOFF_SCOPE_MISMATCH".to_string()));
    }

    #[test]
    fn missing_required_gate_is_not_ready() {
        let decision = reduce_signoff(SignoffReductionInputs {
            snapshot: Some(&snapshot(true)),
            operator: Some(&operator_report()),
            gates: GateInputs {
                v0: Some(pass_v0()),
                v1: Some(pass_v1()),
                v2: Some(pass_v2()),
                v3: None,
            },
            strict_snapshot: &strict_snapshot(StrictEvidenceStatusV1::Pass),
            active_review_snapshot: None,
            applied_scope: &applied_scope(),
            policy: &policy(),
        })
        .expect("decision");
        assert_eq!(decision.decision, SignoffDecisionStateV1::NotReady);
        assert!(decision
            .reasons
            .contains(&"SIGNOFF_BLOCK_GATE_V3_MISSING".to_string()));
    }
}
