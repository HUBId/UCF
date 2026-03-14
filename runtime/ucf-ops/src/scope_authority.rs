use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    interop_consistency_matrix, load_applied_supported_set_context_v1,
    models_active_review_snapshot, models_evidence_snapshot, operator_review_packet,
    operator_signoff, v6_gate, AggregatedActiveReviewSnapshotV1, AppliedSupportedSetContextV1,
    BackendEvidenceSnapshotV1, InteropConsistencyMatrixReportV1, OperatorReviewPacketArgs,
    OperatorReviewPacketV1, OperatorSignoffArgs, OperatorSignoffDecisionV1, OpsError,
    V6GateReportV1,
};

pub const LEGACY_SCOPE_PATH_BLOCKED: &str = "LEGACY_SCOPE_PATH_BLOCKED";
pub const APPLIED_SCOPE_REQUIRED: &str = "APPLIED_SCOPE_REQUIRED";
pub const APPLIED_SCOPE_MISSING: &str = "APPLIED_SCOPE_MISSING";
pub const APPLIED_SCOPE_TRANSLATION_FAILED: &str = "APPLIED_SCOPE_TRANSLATION_FAILED";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ScopeAuthorityOverallStatusV1 {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ScopeAuthorityMismatchCategoryV1 {
    SurfaceDidNotUseAppliedScope,
    LegacyScopePathPresent,
    ExtraSlotFromLegacyInference,
    MissingInScopeSlot,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScopeAuthoritySurfaceResultV1 {
    pub surface: String,
    pub status: ScopeAuthorityOverallStatusV1,
    pub mismatch_categories: Vec<ScopeAuthorityMismatchCategoryV1>,
    pub blocking_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScopeAuthorityCheckReportV1 {
    pub schema_version: u16,
    pub status: ScopeAuthorityOverallStatusV1,
    pub applied_supported_set_digest_prefix: String,
    pub applied_context_digest_prefix: String,
    pub surfaces: Vec<ScopeAuthoritySurfaceResultV1>,
}

pub fn scope_authority_check(
    workdir: &Path,
    out: &Path,
) -> Result<ScopeAuthorityCheckReportV1, OpsError> {
    let applied_scope = load_applied_scope_strict(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active =
        models_active_review_snapshot(workdir, &workdir.join("out/active_review_snapshot.json"))?;
    let review = operator_review_packet(
        workdir,
        &OperatorReviewPacketArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_review_packet.json"),
    )?;
    let signoff = operator_signoff(
        workdir,
        &OperatorSignoffArgs {
            run_id: None,
            latest: true,
            profile: "test".to_string(),
        },
        &workdir.join("out/operator_signoff.json"),
    )?;
    let interop = interop_consistency_matrix(
        workdir,
        &workdir.join("out/interop_consistency_matrix.json"),
    )?;
    let v6 = v6_gate(workdir, &workdir.join("out/v6_gate_report.json"))?;

    let mut surfaces = vec![
        check_backend(&backend, &applied_scope),
        check_active(&active, &applied_scope),
        check_review(&review, &applied_scope),
        check_signoff(&signoff, &applied_scope),
        check_interop(&interop, &applied_scope),
        check_v6(&v6, &applied_scope),
    ];
    surfaces.sort_by(|a, b| a.surface.cmp(&b.surface));
    let status = if surfaces
        .iter()
        .all(|s| matches!(s.status, ScopeAuthorityOverallStatusV1::Pass))
    {
        ScopeAuthorityOverallStatusV1::Pass
    } else {
        ScopeAuthorityOverallStatusV1::Fail
    };
    let report = ScopeAuthorityCheckReportV1 {
        schema_version: 1,
        status,
        applied_supported_set_digest_prefix: applied_scope.applied_set_digest_prefix,
        applied_context_digest_prefix: crate::prefix_hex(&applied_scope.context_digest, 16),
        surfaces,
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

fn load_applied_scope_strict(workdir: &Path) -> Result<AppliedSupportedSetContextV1, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    if applied
        .compatibility_code
        .as_deref()
        .is_some_and(|code| code.contains("LEGACY"))
    {
        return Err(OpsError::Invalid(format!(
            "{LEGACY_SCOPE_PATH_BLOCKED}:{APPLIED_SCOPE_TRANSLATION_FAILED}"
        )));
    }
    Ok(applied)
}

fn check_backend(
    backend: &BackendEvidenceSnapshotV1,
    applied: &AppliedSupportedSetContextV1,
) -> ScopeAuthoritySurfaceResultV1 {
    let slots = backend
        .slots
        .iter()
        .map(|slot| slot.slot_id.clone())
        .collect::<Vec<_>>();
    check_slots(
        "backend_evidence_snapshot",
        crate::prefix_hex(&backend.supported_slot_set_digest, 16),
        slots,
        applied,
    )
}

fn check_active(
    active: &AggregatedActiveReviewSnapshotV1,
    applied: &AppliedSupportedSetContextV1,
) -> ScopeAuthoritySurfaceResultV1 {
    let slots = active
        .slots
        .iter()
        .map(|slot| slot.slot_id.clone())
        .collect::<Vec<_>>();
    check_slots(
        "active_review_snapshot",
        crate::prefix_hex(&active.supported_slot_set_digest, 16),
        slots,
        applied,
    )
}

fn check_review(
    review: &OperatorReviewPacketV1,
    applied: &AppliedSupportedSetContextV1,
) -> ScopeAuthoritySurfaceResultV1 {
    check_slots(
        "operator_review_packet",
        crate::prefix_hex(&review.supported_slot_set_digest, 16),
        review
            .supported_slots
            .iter()
            .map(|slot| slot.slot_id.clone())
            .collect(),
        applied,
    )
}

fn check_signoff(
    signoff: &OperatorSignoffDecisionV1,
    applied: &AppliedSupportedSetContextV1,
) -> ScopeAuthoritySurfaceResultV1 {
    check_slots(
        "operator_signoff",
        crate::prefix_hex(&signoff.supported_slot_set_digest, 16),
        Vec::new(),
        applied,
    )
}

fn check_interop(
    interop: &InteropConsistencyMatrixReportV1,
    applied: &AppliedSupportedSetContextV1,
) -> ScopeAuthoritySurfaceResultV1 {
    let status = if interop.matrix.applied_supported_set_digest_prefix
        == applied.applied_set_digest_prefix
    {
        ScopeAuthorityOverallStatusV1::Pass
    } else {
        ScopeAuthorityOverallStatusV1::Fail
    };
    ScopeAuthoritySurfaceResultV1 {
        surface: "interop_consistency_matrix".to_string(),
        status,
        mismatch_categories: if interop.matrix.applied_supported_set_digest_prefix
            == applied.applied_set_digest_prefix
        {
            Vec::new()
        } else {
            vec![ScopeAuthorityMismatchCategoryV1::SurfaceDidNotUseAppliedScope]
        },
        blocking_codes: if interop.matrix.applied_supported_set_digest_prefix
            == applied.applied_set_digest_prefix
        {
            Vec::new()
        } else {
            vec![APPLIED_SCOPE_REQUIRED.to_string()]
        },
    }
}

fn check_v6(
    v6: &V6GateReportV1,
    _applied: &AppliedSupportedSetContextV1,
) -> ScopeAuthoritySurfaceResultV1 {
    let has_scope_checks = v6
        .checks
        .iter()
        .any(|c| c.name == "applied_supported_scope_present")
        && v6
            .checks
            .iter()
            .any(|c| c.name == "applied_supported_scope_consistent");
    ScopeAuthoritySurfaceResultV1 {
        surface: "v6_gate".to_string(),
        status: if has_scope_checks {
            ScopeAuthorityOverallStatusV1::Pass
        } else {
            ScopeAuthorityOverallStatusV1::Fail
        },
        mismatch_categories: if has_scope_checks {
            Vec::new()
        } else {
            vec![ScopeAuthorityMismatchCategoryV1::SurfaceDidNotUseAppliedScope]
        },
        blocking_codes: if has_scope_checks {
            Vec::new()
        } else {
            vec![APPLIED_SCOPE_REQUIRED.to_string()]
        },
    }
}

fn check_slots(
    surface: &str,
    slot_digest_prefix: String,
    observed_slots: Vec<String>,
    applied: &AppliedSupportedSetContextV1,
) -> ScopeAuthoritySurfaceResultV1 {
    let mut mismatch_categories = Vec::new();
    let mut blocking_codes = Vec::new();
    if slot_digest_prefix != applied.applied_set_digest_prefix {
        mismatch_categories.push(ScopeAuthorityMismatchCategoryV1::SurfaceDidNotUseAppliedScope);
        blocking_codes.push(APPLIED_SCOPE_REQUIRED.to_string());
    }
    if !observed_slots.is_empty() {
        let observed = observed_slots.into_iter().collect::<BTreeSet<_>>();
        let expected = applied.slots.iter().cloned().collect::<BTreeSet<_>>();
        if observed.iter().any(|slot| !expected.contains(slot)) {
            mismatch_categories
                .push(ScopeAuthorityMismatchCategoryV1::ExtraSlotFromLegacyInference);
            blocking_codes.push(LEGACY_SCOPE_PATH_BLOCKED.to_string());
        }
        if expected.iter().any(|slot| !observed.contains(slot)) {
            mismatch_categories.push(ScopeAuthorityMismatchCategoryV1::MissingInScopeSlot);
            blocking_codes.push(APPLIED_SCOPE_REQUIRED.to_string());
        }
    }
    mismatch_categories.sort();
    mismatch_categories.dedup();
    blocking_codes.sort();
    blocking_codes.dedup();
    ScopeAuthoritySurfaceResultV1 {
        surface: surface.to_string(),
        status: if mismatch_categories.is_empty() {
            ScopeAuthorityOverallStatusV1::Pass
        } else {
            ScopeAuthorityOverallStatusV1::Fail
        },
        mismatch_categories,
        blocking_codes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn applied() -> AppliedSupportedSetContextV1 {
        AppliedSupportedSetContextV1 {
            schema_version: 1,
            applied_set_digest_prefix: "abcdabcdabcdabcd".to_string(),
            slots: vec!["sae".to_string(), "world".to_string()],
            decision: crate::SupportedRealSlotSetExecutionDecisionV2::Frozen,
            previous_set_digest_prefix: "prev".to_string(),
            policy_digest_prefix: "policy".to_string(),
            context_digest: "c".repeat(64),
            compatibility_code: None,
        }
    }

    #[test]
    fn slot_mismatch_categories_are_stable() {
        let r = check_slots(
            "surface",
            "wrong".to_string(),
            vec!["extra".to_string()],
            &applied(),
        );
        assert_eq!(r.status, ScopeAuthorityOverallStatusV1::Fail);
        assert_eq!(
            r.mismatch_categories,
            vec![
                ScopeAuthorityMismatchCategoryV1::SurfaceDidNotUseAppliedScope,
                ScopeAuthorityMismatchCategoryV1::ExtraSlotFromLegacyInference,
                ScopeAuthorityMismatchCategoryV1::MissingInScopeSlot,
            ]
        );
    }
}
