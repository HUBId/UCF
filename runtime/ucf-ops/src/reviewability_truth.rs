use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    load_applied_supported_set_context_v1, AggregatedActiveReviewSnapshotV1,
    AppliedSupportedSetContextV1, BackendEvidenceSnapshotV1, OperatorReviewPacketV1,
    OperatorReviewStageV1, OperatorSignoffDecisionV1, OpsError, SignoffDecisionStateV1,
    StrictEvidenceSnapshotV1,
};

const REMEDIATION_BOUND: usize = 4;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SlotReviewabilityEvidenceDigestsV1 {
    pub backend_evidence_snapshot_digest_prefix: String,
    pub active_evidence_digest_prefix: String,
    pub strict_evidence_digest_prefix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SlotReviewabilityTruthV1 {
    pub slot_id: String,
    pub target_hash_prefix: String,
    pub probe_ready: bool,
    pub shadow_ready: bool,
    pub active_eligible: bool,
    pub strict_blocking: bool,
    pub drift_blocking: bool,
    pub alert_blocking: bool,
    pub primary_denial_code: Option<String>,
    pub remediation_codes: Vec<String>,
    pub evidence_digests: SlotReviewabilityEvidenceDigestsV1,
    pub reviewability_truth_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReviewabilityReductionSlotSummaryV1 {
    pub slot_id: String,
    pub reviewable: bool,
    pub primary_denial_code: Option<String>,
    pub remediation_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ReviewabilityAggregateReadinessV1 {
    NoneReviewable,
    PartialReviewable,
    AllReviewable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReviewabilityReductionV1 {
    pub per_slot: Vec<ReviewabilityReductionSlotSummaryV1>,
    pub aggregate_readiness: ReviewabilityAggregateReadinessV1,
    pub overall_blocking_codes: Vec<String>,
    pub overall_remediation_codes: Vec<String>,
    pub reduction_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ReviewTruthCheckStatusV1 {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ReviewTruthMismatchCategoryV1 {
    PerSlotReviewabilityMismatch,
    AggregateReductionMismatch,
    SignoffReviewabilityDrift,
    ReviewPacketReviewabilityDrift,
    AppliedScopeSlotTruthMissing,
    LegacyReviewabilityField,
    LegacyReductionTranslated,
    LegacyReductionRejected,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReviewTruthCheckReportV1 {
    pub schema_version: u16,
    pub status: ReviewTruthCheckStatusV1,
    pub mismatch_categories: Vec<ReviewTruthMismatchCategoryV1>,
    pub remediation_codes: Vec<String>,
    pub reduction_digest_prefix: String,
}

pub fn slot_is_reviewable(truth: &SlotReviewabilityTruthV1) -> bool {
    truth.active_eligible
        && !truth.strict_blocking
        && !truth.drift_blocking
        && !truth.alert_blocking
}

pub fn derive_slot_reviewability_truths_from_active(
    applied_scope: &AppliedSupportedSetContextV1,
    backend: &BackendEvidenceSnapshotV1,
    active: &AggregatedActiveReviewSnapshotV1,
) -> Result<Vec<SlotReviewabilityTruthV1>, OpsError> {
    if backend.supported_slot_set_digest != applied_scope.applied_set_digest_prefix
        || active.supported_slot_set_digest != applied_scope.applied_set_digest_prefix
    {
        return Err(OpsError::Invalid(
            "APPLIED_SCOPE_SLOT_TRUTH_MISSING: digest mismatch".to_string(),
        ));
    }
    let mut truths = Vec::with_capacity(applied_scope.slots.len());
    for slot_id in &applied_scope.slots {
        let backend_slot = backend
            .slots
            .iter()
            .find(|s| &s.slot_id == slot_id)
            .ok_or_else(|| {
                OpsError::Invalid(format!(
                    "APPLIED_SCOPE_SLOT_TRUTH_MISSING: backend missing slot {}",
                    slot_id
                ))
            })?;
        let active_slot = active
            .slots
            .iter()
            .find(|s| &s.slot_id == slot_id)
            .ok_or_else(|| {
                OpsError::Invalid(format!(
                    "APPLIED_SCOPE_SLOT_TRUTH_MISSING: active missing slot {}",
                    slot_id
                ))
            })?;
        let mut remediation = BTreeSet::new();
        for code in &active_slot.remediation_codes {
            remediation.insert(code.clone());
        }
        let mut truth = SlotReviewabilityTruthV1 {
            slot_id: slot_id.clone(),
            target_hash_prefix: backend_slot.target_hash_prefix.clone(),
            probe_ready: active_slot.probe_ready,
            shadow_ready: active_slot.shadow_ready,
            active_eligible: active_slot.active_eligible,
            strict_blocking: active_slot.strict_blocking,
            drift_blocking: active_slot.drift_blocking,
            alert_blocking: active_slot.alert_blocking,
            primary_denial_code: active_slot.primary_denial_code.clone(),
            remediation_codes: remediation.into_iter().take(REMEDIATION_BOUND).collect(),
            evidence_digests: SlotReviewabilityEvidenceDigestsV1 {
                backend_evidence_snapshot_digest_prefix: crate::prefix_hex(
                    &backend.snapshot_digest,
                    16,
                ),
                active_evidence_digest_prefix: crate::prefix_hex(&active_slot.evidence_digest, 16),
                strict_evidence_digest_prefix: active_slot
                    .contributing_evidence_digests
                    .strict_evidence_digest_prefix
                    .clone(),
            },
            reviewability_truth_digest: String::new(),
        };
        truth.reviewability_truth_digest = truth_digest(&truth)?;
        truths.push(truth);
    }
    if active
        .slots
        .iter()
        .any(|slot| !applied_scope.slots.contains(&slot.slot_id))
    {
        return Err(OpsError::Invalid(
            "APPLIED_SCOPE_SLOT_TRUTH_MISSING: extra-slot truth injected".to_string(),
        ));
    }
    Ok(truths)
}

pub fn derive_slot_reviewability_truths(
    applied_scope: &AppliedSupportedSetContextV1,
    backend: &BackendEvidenceSnapshotV1,
    active: &AggregatedActiveReviewSnapshotV1,
    strict: &StrictEvidenceSnapshotV1,
) -> Result<Vec<SlotReviewabilityTruthV1>, OpsError> {
    if backend.supported_slot_set_digest != applied_scope.applied_set_digest_prefix
        || active.supported_slot_set_digest != applied_scope.applied_set_digest_prefix
    {
        return Err(OpsError::Invalid(
            "APPLIED_SCOPE_SLOT_TRUTH_MISSING: digest mismatch".to_string(),
        ));
    }

    let mut truths = Vec::with_capacity(applied_scope.slots.len());
    for slot_id in &applied_scope.slots {
        let backend_slot = backend
            .slots
            .iter()
            .find(|slot| &slot.slot_id == slot_id)
            .ok_or_else(|| {
                OpsError::Invalid(format!(
                    "APPLIED_SCOPE_SLOT_TRUTH_MISSING: backend missing slot {}",
                    slot_id
                ))
            })?;
        let active_slot = active
            .slots
            .iter()
            .find(|slot| &slot.slot_id == slot_id)
            .ok_or_else(|| {
                OpsError::Invalid(format!(
                    "APPLIED_SCOPE_SLOT_TRUTH_MISSING: active missing slot {}",
                    slot_id
                ))
            })?;

        let mut remediation = BTreeSet::new();
        for code in &active_slot.remediation_codes {
            remediation.insert(code.clone());
        }

        let mut truth = SlotReviewabilityTruthV1 {
            slot_id: slot_id.clone(),
            target_hash_prefix: backend_slot.target_hash_prefix.clone(),
            probe_ready: active_slot.probe_ready,
            shadow_ready: active_slot.shadow_ready,
            active_eligible: active_slot.active_eligible,
            strict_blocking: active_slot.strict_blocking,
            drift_blocking: active_slot.drift_blocking,
            alert_blocking: active_slot.alert_blocking,
            primary_denial_code: active_slot.primary_denial_code.clone(),
            remediation_codes: remediation.into_iter().take(REMEDIATION_BOUND).collect(),
            evidence_digests: SlotReviewabilityEvidenceDigestsV1 {
                backend_evidence_snapshot_digest_prefix: crate::prefix_hex(
                    &backend.snapshot_digest,
                    16,
                ),
                active_evidence_digest_prefix: crate::prefix_hex(&active_slot.evidence_digest, 16),
                strict_evidence_digest_prefix: crate::prefix_hex(&strict.snapshot_digest, 16),
            },
            reviewability_truth_digest: String::new(),
        };
        truth.reviewability_truth_digest = truth_digest(&truth)?;
        truths.push(truth);
    }

    if active
        .slots
        .iter()
        .any(|slot| !applied_scope.slots.contains(&slot.slot_id))
    {
        return Err(OpsError::Invalid(
            "APPLIED_SCOPE_SLOT_TRUTH_MISSING: extra-slot truth injected".to_string(),
        ));
    }

    Ok(truths)
}

pub fn reduce_reviewability(
    applied_scope: &AppliedSupportedSetContextV1,
    truths: &[SlotReviewabilityTruthV1],
) -> Result<ReviewabilityReductionV1, OpsError> {
    if truths.len() != applied_scope.slots.len() {
        return Err(OpsError::Invalid(
            "APPLIED_SCOPE_SLOT_TRUTH_MISSING: missing in-scope slot truth".to_string(),
        ));
    }
    for (idx, slot_id) in applied_scope.slots.iter().enumerate() {
        if truths.get(idx).map(|s| &s.slot_id) != Some(slot_id) {
            return Err(OpsError::Invalid(
                "APPLIED_SCOPE_SLOT_TRUTH_MISSING: slot truth order mismatch".to_string(),
            ));
        }
    }

    let mut blocking = BTreeSet::new();
    let mut remediation = BTreeSet::new();
    let per_slot = truths
        .iter()
        .map(|truth| {
            let reviewable = slot_is_reviewable(truth);
            if !reviewable {
                if let Some(code) = truth.primary_denial_code.as_ref() {
                    blocking.insert(code.clone());
                }
                for code in &truth.remediation_codes {
                    remediation.insert(code.clone());
                }
            }
            ReviewabilityReductionSlotSummaryV1 {
                slot_id: truth.slot_id.clone(),
                reviewable,
                primary_denial_code: truth.primary_denial_code.clone(),
                remediation_codes: truth.remediation_codes.clone(),
            }
        })
        .collect::<Vec<_>>();

    let reviewable_count = per_slot.iter().filter(|slot| slot.reviewable).count();
    let aggregate_readiness = if reviewable_count == 0 {
        ReviewabilityAggregateReadinessV1::NoneReviewable
    } else if reviewable_count == per_slot.len() {
        ReviewabilityAggregateReadinessV1::AllReviewable
    } else {
        ReviewabilityAggregateReadinessV1::PartialReviewable
    };

    let mut reduction = ReviewabilityReductionV1 {
        per_slot,
        aggregate_readiness,
        overall_blocking_codes: blocking.into_iter().take(12).collect(),
        overall_remediation_codes: remediation.into_iter().take(12).collect(),
        reduction_digest: String::new(),
    };
    reduction.reduction_digest = reduction_digest(applied_scope, &reduction, truths)?;
    Ok(reduction)
}

pub fn review_truth_check(
    workdir: &Path,
    out: &Path,
) -> Result<ReviewTruthCheckReportV1, OpsError> {
    let applied_scope = load_applied_supported_set_context_v1(workdir)?;
    let backend: BackendEvidenceSnapshotV1 =
        read_json(&workdir.join("out/backend_evidence_snapshot.json"))?;
    let active: AggregatedActiveReviewSnapshotV1 =
        read_json(&workdir.join("out/active_review_snapshot.json"))?;
    let signoff: OperatorSignoffDecisionV1 = read_json(&workdir.join("out/operator_signoff.json"))?;
    let packet: OperatorReviewPacketV1 =
        read_json(&workdir.join("out/operator_review_packet.json"))?;
    let strict: StrictEvidenceSnapshotV1 =
        read_json(&workdir.join("out/strict_evidence_snapshot.json"))?;

    let truths = derive_slot_reviewability_truths(&applied_scope, &backend, &active, &strict)?;
    let reduction = reduce_reviewability(&applied_scope, &truths)?;

    let mut mismatch = BTreeSet::new();
    let mut remediation = BTreeSet::new();

    for (active_slot, truth) in active.slots.iter().zip(truths.iter()) {
        if active_slot.slot_id != truth.slot_id
            || active_slot.probe_ready != truth.probe_ready
            || active_slot.shadow_ready != truth.shadow_ready
            || active_slot.active_eligible != truth.active_eligible
            || active_slot.strict_blocking != truth.strict_blocking
            || active_slot.drift_blocking != truth.drift_blocking
            || active_slot.alert_blocking != truth.alert_blocking
        {
            mismatch.insert(ReviewTruthMismatchCategoryV1::PerSlotReviewabilityMismatch);
            remediation.insert("rerun_models_active_review_snapshot".to_string());
        }
    }

    let expected_active = match reduction.aggregate_readiness {
        ReviewabilityAggregateReadinessV1::NoneReviewable => {
            crate::ActiveReviewOverallStatusV1::NoneReviewable
        }
        ReviewabilityAggregateReadinessV1::PartialReviewable => {
            crate::ActiveReviewOverallStatusV1::PartialReviewable
        }
        ReviewabilityAggregateReadinessV1::AllReviewable => {
            crate::ActiveReviewOverallStatusV1::AllReviewable
        }
    };
    if active.overall_review_status != expected_active {
        mismatch.insert(ReviewTruthMismatchCategoryV1::AggregateReductionMismatch);
        remediation.insert("rerun_models_active_review_snapshot".to_string());
    }

    if signoff.decision == SignoffDecisionStateV1::ReadyForActiveReview
        && matches!(
            reduction.aggregate_readiness,
            ReviewabilityAggregateReadinessV1::NoneReviewable
        )
    {
        mismatch.insert(ReviewTruthMismatchCategoryV1::SignoffReviewabilityDrift);
        remediation.insert("rerun_operator_signoff".to_string());
    }

    if packet.review_stage == OperatorReviewStageV1::ReviewActiveReady
        && signoff.decision != SignoffDecisionStateV1::ReadyForActiveReview
    {
        mismatch.insert(ReviewTruthMismatchCategoryV1::ReviewPacketReviewabilityDrift);
        remediation.insert("rerun_operator_review_packet".to_string());
    }

    if active.slots.len() != applied_scope.slots.len()
        || reduction.per_slot.len() != applied_scope.slots.len()
    {
        mismatch.insert(ReviewTruthMismatchCategoryV1::AppliedScopeSlotTruthMissing);
        remediation.insert("run_models_applied_scope_check".to_string());
    }

    let report = ReviewTruthCheckReportV1 {
        schema_version: 1,
        status: if mismatch.is_empty() {
            ReviewTruthCheckStatusV1::Pass
        } else {
            ReviewTruthCheckStatusV1::Fail
        },
        mismatch_categories: mismatch.into_iter().collect(),
        remediation_codes: remediation.into_iter().take(12).collect(),
        reduction_digest_prefix: crate::prefix_hex(&reduction.reduction_digest, 16),
    };

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, OpsError> {
    let bytes = fs::read(path)?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn truth_digest(truth: &SlotReviewabilityTruthV1) -> Result<String, OpsError> {
    let mut cloned = truth.clone();
    cloned.reviewability_truth_digest.clear();
    Ok(crate::sha256_hex(&serde_json::to_vec(&cloned)?))
}

fn reduction_digest(
    applied_scope: &AppliedSupportedSetContextV1,
    reduction: &ReviewabilityReductionV1,
    truths: &[SlotReviewabilityTruthV1],
) -> Result<String, OpsError> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(applied_scope.context_digest.as_bytes());
    let mut cloned = reduction.clone();
    cloned.reduction_digest.clear();
    bytes.extend_from_slice(&serde_json::to_vec(&cloned)?);
    for truth in truths {
        bytes.extend_from_slice(truth.reviewability_truth_digest.as_bytes());
    }
    Ok(crate::sha256_hex(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models_lifecycle::{
        ActiveReviewContributingDigestsV1, ActiveReviewEvidenceV1, ActiveReviewOverallStatusV1,
        ActiveReviewSignoffAlignmentV1, BackendEvidenceSlotDenialsV1,
        BackendEvidenceSlotEvidenceV1, BackendEvidenceSlotReadinessV1,
        BackendEvidenceSlotSnapshotV1, BackendSupportMatrixV1, BackendSupportStateV1,
        BurnResolutionStatusV1, BurnSupportResolutionV1, DriftStatusV1,
        SupportedRealSlotSetExecutionDecisionV2,
    };
    use crate::{OptionalBackendSupportStateV1, StrictEvidenceStatusV1};

    fn applied() -> AppliedSupportedSetContextV1 {
        AppliedSupportedSetContextV1 {
            schema_version: 1,
            applied_set_digest_prefix: "set123".to_string(),
            slots: vec!["slot_a".to_string(), "slot_b".to_string()],
            policy_digest_prefix: "pol".to_string(),
            previous_set_digest_prefix: "src".to_string(),
            decision: SupportedRealSlotSetExecutionDecisionV2::Frozen,
            compatibility_code: None,
            context_digest: "ctx1234567890abcdef".to_string(),
        }
    }

    fn backend() -> BackendEvidenceSnapshotV1 {
        let mk = |slot: &str| BackendEvidenceSlotSnapshotV1 {
            slot_id: slot.to_string(),
            target_hash_prefix: "th".to_string(),
            backend_support: BackendSupportMatrixV1 {
                stub: BackendSupportStateV1::Supported,
                candle: BackendSupportStateV1::Supported,
                burn: BackendSupportStateV1::Supported,
            },
            evidence: BackendEvidenceSlotEvidenceV1 {
                latest_probe_report_digest_prefix: "p".to_string(),
                latest_compare_window_digest_prefix: "c".to_string(),
                latest_shadow_ready_digest_prefix: "s".to_string(),
                latest_active_evidence_digest_prefix: "a".to_string(),
                latest_drift_status: DriftStatusV1::Ok,
                freshness_probe_age_ticks: None,
                freshness_compare_age_ticks: None,
                freshness_no_impact_age_ticks: None,
                freshness_drift_status_age_ticks: None,
                hash_consistency_ok: true,
            },
            readiness: BackendEvidenceSlotReadinessV1 {
                probe_ready: true,
                shadow_ready: true,
                active_eligible: true,
            },
            denials: BackendEvidenceSlotDenialsV1 {
                probe: None,
                shadow: None,
                active: None,
            },
            remediation_codes: vec![],
            canonical_remediation_codes: vec![],
            burn_resolution: BurnSupportResolutionV1 {
                slot_id: slot.to_string(),
                resolution: BurnResolutionStatusV1::BurnSupportedForShadowCompare,
                support_state: OptionalBackendSupportStateV1::Supported,
                rationale_codes: vec![],
                evidence_digest: "burn".to_string(),
            },
        };
        BackendEvidenceSnapshotV1 {
            schema_version: 1,
            supported_slot_set_digest: "set123".to_string(),
            policy_graph_digest_prefix: "pol".to_string(),
            manifest_digest_prefix: "man".to_string(),
            slots: vec![mk("slot_a"), mk("slot_b")],
            snapshot_digest: "backend_digest_1234567890".to_string(),
        }
    }

    fn active() -> AggregatedActiveReviewSnapshotV1 {
        let mk = |slot: &str, reviewable: bool| ActiveReviewEvidenceV1 {
            slot_id: slot.to_string(),
            target_hash_prefix: "th".to_string(),
            manifest_digest_prefix: "man".to_string(),
            probe_ready: true,
            shadow_ready: true,
            active_eligible: reviewable,
            strict_blocking: false,
            drift_blocking: false,
            alert_blocking: false,
            primary_denial_code: None,
            remediation_codes: vec![],
            contributing_evidence_digests: ActiveReviewContributingDigestsV1 {
                probe_report_digest_prefix: "p".to_string(),
                shadow_ready_digest_prefix: "s".to_string(),
                active_evidence_digest_prefix: "a".to_string(),
                strict_evidence_digest_prefix: "st".to_string(),
            },
            burn_resolution: BurnSupportResolutionV1 {
                slot_id: slot.to_string(),
                resolution: BurnResolutionStatusV1::BurnSupportedForShadowCompare,
                support_state: OptionalBackendSupportStateV1::Supported,
                rationale_codes: vec![],
                evidence_digest: "burn".to_string(),
            },
            evidence_digest: format!("ev_{slot}"),
        };
        AggregatedActiveReviewSnapshotV1 {
            schema_version: 1,
            supported_slot_set_digest: "set123".to_string(),
            policy_graph_digest_prefix: "pol".to_string(),
            manifest_digest_prefix: "man".to_string(),
            slots: vec![mk("slot_a", true), mk("slot_b", false)],
            overall_review_status: ActiveReviewOverallStatusV1::PartialReviewable,
            signoff_alignment: ActiveReviewSignoffAlignmentV1 {
                aligned: true,
                status_code: "ALIGNED".to_string(),
            },
            canonical_governance_entry_digest_prefix: "MISSING".to_string(),
            final_governance_consumer_authority_digest_prefix: "MISSING".to_string(),
            governance_residual_sweep_digest_prefix: "MISSING".to_string(),
            residual_free_governance_authority_digest_prefix: "MISSING".to_string(),
            final_readiness_consumer_authority_digest_prefix: "MISSING".to_string(),
            readiness_residual_sweep_digest_prefix: "MISSING".to_string(),
            residual_free_readiness_authority_digest_prefix: "MISSING".to_string(),
            snapshot_digest: "active_digest".to_string(),
        }
    }

    #[test]
    fn slot_truth_digest_is_stable() {
        let strict = StrictEvidenceSnapshotV1 {
            schema_version: 1,
            strict_mode_enabled: true,
            strict_status: StrictEvidenceStatusV1::Pass,
            strict_report_digest_prefix: None,
            policy_graph_digest_prefix: None,
            manifest_digest_prefix: None,
            supported_slot_set_digest_prefix: None,
            primary_denial_code: None,
            remediation_codes: vec![],
            failing_check_ids: vec![],
            snapshot_digest: "strict_digest".to_string(),
        };
        let truths = derive_slot_reviewability_truths(&applied(), &backend(), &active(), &strict)
            .expect("truths");
        let first = truths[0].reviewability_truth_digest.clone();
        let second = derive_slot_reviewability_truths(&applied(), &backend(), &active(), &strict)
            .expect("truths")[0]
            .reviewability_truth_digest
            .clone();
        assert_eq!(first, second);
    }

    #[test]
    fn reduction_deterministic() {
        let strict = StrictEvidenceSnapshotV1 {
            schema_version: 1,
            strict_mode_enabled: true,
            strict_status: StrictEvidenceStatusV1::Pass,
            strict_report_digest_prefix: None,
            policy_graph_digest_prefix: None,
            manifest_digest_prefix: None,
            supported_slot_set_digest_prefix: None,
            primary_denial_code: None,
            remediation_codes: vec![],
            failing_check_ids: vec![],
            snapshot_digest: "strict_digest".to_string(),
        };
        let truths = derive_slot_reviewability_truths(&applied(), &backend(), &active(), &strict)
            .expect("truths");
        let r1 = reduce_reviewability(&applied(), &truths).expect("r1");
        let r2 = reduce_reviewability(&applied(), &truths).expect("r2");
        assert_eq!(r1, r2);
        assert_eq!(
            r1.aggregate_readiness,
            ReviewabilityAggregateReadinessV1::PartialReviewable
        );
    }
}
