use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use ucf_types::remediation_codes::{remediation_for_condition, CanonicalConditionV1};

use crate::remediation::{
    all_registry_rows, canonical_condition_for_bundle_spine_mismatch,
    canonical_condition_for_export_normalize_category,
    canonical_condition_for_governance_entry_mismatch, canonical_condition_for_interop_category,
    canonical_condition_for_operator_export_chain_mismatch,
    canonical_condition_for_readiness_spine_mismatch, canonical_condition_for_roundtrip_mismatch,
    canonical_condition_for_scope_authority_mismatch, canonical_condition_from_code,
    canonical_from_legacy_code, canonical_from_legacy_remediation,
    primary_remediation_for_condition_code,
};
use crate::{prefix_hex, sha256_hex, OpsError};
use std::fs;

const SCHEMA_VERSION: u16 = 1;
const SURFACE_ORDER: [&str; 6] = [
    "strict_check",
    "eligibility",
    "operator_report",
    "operator_signoff",
    "gate_v4",
    "export_manifest",
];

const CROSS_SURFACE_ORDER: [&str; 14] = [
    "Strict",
    "Eligibility",
    "ActiveReviewSnapshot",
    "OperatorReport",
    "OperatorSignoff",
    "OperatorReviewPacket",
    "GateV3",
    "GateV4",
    "GateV5",
    "GateV6",
    "GateV7",
    "ExportNormalizeCheck",
    "ExportRoundTripCheck",
    "InteropMatrix",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalRemediationObservationV1 {
    pub primary_remediation_code: Option<String>,
    pub secondary_codes: Vec<String>,
    pub source_surface: String,
    pub derived_from_condition_code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RemediationConsistencyStatusV1 {
    Pass,
    Fail,
    Skip,
    Missing,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RemediationMismatchKindV1 {
    MissingSurface,
    DifferentPrimaryCode,
    UnknownConditionMapping,
    LegacyTranslationDrift,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemediationConsistencyObservedV1 {
    pub strict_check_primary: Option<String>,
    pub eligibility_primary: Option<String>,
    pub operator_report_primary: Option<String>,
    pub operator_signoff_primary: Option<String>,
    pub gate_primary: Vec<String>,
    pub export_manifest_primary: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemediationConsistencyCheckV1 {
    pub schema_version: u16,
    pub canonical_condition_code: String,
    pub surfaces_checked: Vec<String>,
    pub expected_primary_remediation_code: Option<String>,
    pub observed: RemediationConsistencyObservedV1,
    pub status: RemediationConsistencyStatusV1,
    pub mismatch_kind: Option<RemediationMismatchKindV1>,
    pub remediation_consistency_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemediationConsistencySummaryV1 {
    pub schema_version: u16,
    pub total_conditions: usize,
    pub pass_count: usize,
    pub fail_count: usize,
    pub skip_count: usize,
    pub missing_count: usize,
    pub top_mismatch_categories: Vec<String>,
    pub status: RemediationConsistencyStatusV1,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemediationConsistencyReportV1 {
    pub schema_version: u16,
    pub checks: Vec<RemediationConsistencyCheckV1>,
    pub summary: RemediationConsistencySummaryV1,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CrossSurfaceObservationStatusV1 {
    Pass,
    Fail,
    Missing,
    Skip,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CrossSurfaceObservedSurfaceV1 {
    pub surface_kind: String,
    pub primary_blocking_code: Option<String>,
    pub primary_remediation_code: Option<String>,
    pub status: CrossSurfaceObservationStatusV1,
    pub source_digest_prefix: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CrossSurfaceConditionObservationV1 {
    pub canonical_condition_code: String,
    pub expected_primary_remediation_code: Option<String>,
    pub observed_surfaces: Vec<CrossSurfaceObservedSurfaceV1>,
    pub observation_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemediationInteropCheckReportV1 {
    pub schema_version: u16,
    pub conditions_checked: usize,
    pub mismatches_found: usize,
    pub top_mismatch_categories: Vec<String>,
    pub observations: Vec<CrossSurfaceConditionObservationV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpineConditionObservationV1 {
    pub canonical_condition_code: String,
    pub expected_primary_blocking_code: Option<String>,
    pub expected_primary_remediation_code: Option<String>,
    pub observed_surfaces: Vec<CrossSurfaceObservedSurfaceV1>,
    pub observation_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemediationSpineCheckReportV1 {
    pub schema_version: u16,
    pub conditions_checked: usize,
    pub mismatches_found: usize,
    pub top_mismatch_categories: Vec<String>,
    pub observations: Vec<SpineConditionObservationV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CanonicalPrimarySemanticsAuthorityStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalPrimarySemanticsAuthorityV1 {
    pub covered_surface_count: usize,
    pub covered_condition_count: usize,
    pub authority_status: CanonicalPrimarySemanticsAuthorityStatusV1,
    pub primary_semantics_digest: String,
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_bundle_spine_digest_prefix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrimarySemanticsObservedSurfaceV1 {
    pub surface_kind: String,
    pub primary_blocking_code: Option<String>,
    pub primary_remediation_code: Option<String>,
    pub status: CrossSurfaceObservationStatusV1,
    pub source_digest_prefix: Option<String>,
    pub diagnostic_codes: Vec<String>,
    pub secondary_diagnostic_codes: Vec<String>,
    pub secondary_surface_reason_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrimarySemanticsObservationV1 {
    pub canonical_condition_code: String,
    pub expected_primary_blocking_code: Option<String>,
    pub expected_primary_remediation_code: Option<String>,
    pub observed_surfaces: Vec<PrimarySemanticsObservedSurfaceV1>,
    pub observation_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrimarySemanticsSweepReportV1 {
    pub schema_version: u16,
    pub surfaces_checked: usize,
    pub conditions_checked: usize,
    pub mismatches_found: usize,
    pub top_mismatch_categories: Vec<String>,
    pub observations: Vec<PrimarySemanticsObservationV1>,
    pub authority: CanonicalPrimarySemanticsAuthorityV1,
}

pub const FINAL_PRIMARY_SEMANTICS_AUTHORITY_REQUIRED: &str =
    "FINAL_PRIMARY_SEMANTICS_AUTHORITY_REQUIRED";
pub const FINAL_PRIMARY_SEMANTICS_INPUTS_REQUIRED: &str = "FINAL_PRIMARY_SEMANTICS_INPUTS_REQUIRED";
pub const CANONICAL_CONDITION_MODEL_REQUIRED: &str = "CANONICAL_CONDITION_MODEL_REQUIRED";
pub const CANONICAL_REMEDIATION_REGISTRY_REQUIRED: &str = "CANONICAL_REMEDIATION_REGISTRY_REQUIRED";
pub const LEGACY_PRIMARY_SEMANTICS_INPUT_BLOCKED: &str = "LEGACY_PRIMARY_SEMANTICS_INPUT_BLOCKED";
pub const LEGACY_PRIMARY_SEMANTICS_TRANSLATED: &str = "LEGACY_PRIMARY_SEMANTICS_TRANSLATED";
pub const LEGACY_PRIMARY_SEMANTICS_REJECTED: &str = "LEGACY_PRIMARY_SEMANTICS_REJECTED";
pub const RESIDUAL_PRIMARY_SEMANTICS_PATH_BLOCKED: &str = "RESIDUAL_PRIMARY_SEMANTICS_PATH_BLOCKED";
pub const RESIDUAL_PRIMARY_SEMANTICS_PATH_TRANSLATED: &str =
    "RESIDUAL_PRIMARY_SEMANTICS_PATH_TRANSLATED";
pub const RESIDUAL_PRIMARY_SEMANTICS_PATH_REJECTED: &str =
    "RESIDUAL_PRIMARY_SEMANTICS_PATH_REJECTED";
pub const RESIDUAL_FREE_FINAL_PRIMARY_SEMANTICS_INPUTS_REQUIRED: &str =
    "RESIDUAL_FREE_FINAL_PRIMARY_SEMANTICS_INPUTS_REQUIRED";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinalPrimarySemanticsConsumerAuthorityStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalPrimarySemanticsConsumerSurfaceStatusV1 {
    pub surface_kind: String,
    pub status: CrossSurfaceObservationStatusV1,
    pub mismatch_categories: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalPrimarySemanticsConsumerAuthorityV1 {
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_bundle_spine_digest_prefix: String,
    pub canonical_primary_semantics_authority_digest_prefix: String,
    pub covered_consumer_count: usize,
    pub authority_status: FinalPrimarySemanticsConsumerAuthorityStatusV1,
    pub authority_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalPrimarySemanticsSweepReportV1 {
    pub schema_version: u16,
    pub conditions_checked: usize,
    pub mismatches_found: usize,
    pub top_mismatch_categories: Vec<String>,
    pub surface_statuses: Vec<FinalPrimarySemanticsConsumerSurfaceStatusV1>,
    pub authority: FinalPrimarySemanticsConsumerAuthorityV1,
}

#[derive(Debug, Clone)]
pub struct FinalPrimarySemanticsAuthorityContextV1 {
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_bundle_spine_digest_prefix: String,
    pub canonical_primary_semantics_authority_digest_prefix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalPrimarySemanticsInputsContextV1 {
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_bundle_spine_digest_prefix: String,
    pub canonical_primary_semantics_authority_digest_prefix: String,
    pub final_primary_semantics_consumer_authority_digest_prefix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResidualFreeFinalPrimarySemanticsInputsV1 {
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_bundle_spine_digest_prefix: String,
    pub canonical_primary_semantics_authority_digest_prefix: String,
    pub final_primary_semantics_consumer_authority_digest_prefix: String,
    pub final_primary_semantics_residual_sweep_digest_prefix: String,
    pub authority_digest: String,
}

#[derive(Clone)]
struct CoveredCondition {
    code: &'static str,
    condition: CanonicalConditionV1,
}

#[derive(Clone)]
enum SurfaceSignal {
    LegacyCode(&'static str),
    LegacyRemediation(&'static str),
    MappedCanonicalCondition(&'static str),
    Skip,
    Missing,
}

const PRIMARY_SEMANTICS_SURFACE_ORDER: [&str; 17] = [
    "AppliedScopeAuthority",
    "CanonicalGovernanceEntry",
    "CanonicalReadinessSpine",
    "CanonicalBundleSpine",
    "BundleSpineCheck",
    "ExportRoundTrip",
    "ExportNormalizeCheck",
    "InteropMatrix",
    "OperatorExportAuthorityChain",
    "OperatorSignoff",
    "OperatorReviewPacket",
    "OperatorWorkflow",
    "GateV4",
    "GateV5",
    "GateV6",
    "GateV7",
    "GateV8",
];

pub fn remediation_consistency_check(
    out: &Path,
) -> Result<RemediationConsistencyReportV1, OpsError> {
    let checks: Vec<RemediationConsistencyCheckV1> = covered_conditions()
        .into_iter()
        .map(build_condition_check)
        .collect();

    let fail_count = checks
        .iter()
        .filter(|c| matches!(c.status, RemediationConsistencyStatusV1::Fail))
        .count();
    let pass_count = checks
        .iter()
        .filter(|c| matches!(c.status, RemediationConsistencyStatusV1::Pass))
        .count();
    let skip_count = checks
        .iter()
        .filter(|c| matches!(c.status, RemediationConsistencyStatusV1::Skip))
        .count();
    let missing_count = checks
        .iter()
        .filter(|c| matches!(c.status, RemediationConsistencyStatusV1::Missing))
        .count();

    let mut mismatch_hist = BTreeMap::<String, usize>::new();
    for check in &checks {
        if let Some(kind) = check.mismatch_kind.as_ref() {
            *mismatch_hist.entry(format!("{kind:?}")).or_default() += 1;
        }
    }

    let summary = RemediationConsistencySummaryV1 {
        schema_version: SCHEMA_VERSION,
        total_conditions: checks.len(),
        pass_count,
        fail_count,
        skip_count,
        missing_count,
        top_mismatch_categories: mismatch_hist
            .into_iter()
            .map(|(k, v)| format!("{k}:{v}"))
            .collect(),
        status: if fail_count == 0 {
            RemediationConsistencyStatusV1::Pass
        } else {
            RemediationConsistencyStatusV1::Fail
        },
    };

    let report = RemediationConsistencyReportV1 {
        schema_version: SCHEMA_VERSION,
        checks,
        summary,
        suggestions: vec![
            "cargo run -p ucf-ops -- docs remediation-codes --out docs/remediation_codes_v1.md"
                .to_string(),
            "refactor drifting surface to use canonical remediation registry mapping directly"
                .to_string(),
            "update legacy translation layer in runtime/ucf-ops/src/remediation.rs".to_string(),
        ],
    };

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_string_pretty(&report)?)?;
    Ok(report)
}

pub fn remediation_interop_check(out: &Path) -> Result<RemediationInteropCheckReportV1, OpsError> {
    let observations = covered_cross_surface_conditions()
        .into_iter()
        .map(|condition_code| {
            let expected_primary = primary_remediation_for_condition_code(condition_code);
            let observed_surfaces = CROSS_SURFACE_ORDER
                .iter()
                .map(|surface| {
                    observe_cross_surface(condition_code, surface, expected_primary.as_deref())
                })
                .collect::<Vec<_>>();
            let mut hasher = Sha256::new();
            hasher.update(serde_json::to_vec(&(
                condition_code,
                &expected_primary,
                &observed_surfaces,
            ))?);
            Ok(CrossSurfaceConditionObservationV1 {
                canonical_condition_code: condition_code.to_string(),
                expected_primary_remediation_code: expected_primary,
                observed_surfaces,
                observation_digest: format!("{:x}", hasher.finalize()),
            })
        })
        .collect::<Result<Vec<_>, OpsError>>()?;

    let mut mismatch_hist = BTreeMap::<String, usize>::new();
    let mut mismatches_found = 0usize;
    for observation in &observations {
        if let Some(expected) = observation.expected_primary_remediation_code.as_deref() {
            for surface in &observation.observed_surfaces {
                match surface.status {
                    CrossSurfaceObservationStatusV1::Pass
                    | CrossSurfaceObservationStatusV1::Skip => {}
                    CrossSurfaceObservationStatusV1::Missing => {
                        mismatches_found += 1;
                        *mismatch_hist
                            .entry("MISSING_SURFACE".to_string())
                            .or_default() += 1;
                    }
                    CrossSurfaceObservationStatusV1::Fail => {
                        mismatches_found += 1;
                        if surface.primary_remediation_code.is_none() {
                            *mismatch_hist
                                .entry("UNKNOWN_CONDITION_MAPPING".to_string())
                                .or_default() += 1;
                        } else if surface.primary_remediation_code.as_deref() != Some(expected) {
                            *mismatch_hist
                                .entry("PRIMARY_REMEDIATION_MISMATCH".to_string())
                                .or_default() += 1;
                        }
                    }
                }
            }
        }
    }

    let report = RemediationInteropCheckReportV1 {
        schema_version: SCHEMA_VERSION,
        conditions_checked: observations.len(),
        mismatches_found,
        top_mismatch_categories: mismatch_hist
            .into_iter()
            .map(|(k, v)| format!("{k}:{v}"))
            .collect(),
        observations,
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_string_pretty(&report)?)?;
    Ok(report)
}

pub fn remediation_spine_check(out: &Path) -> Result<RemediationSpineCheckReportV1, OpsError> {
    let surfaces = [
        "AppliedScopeAuthority",
        "CanonicalGovernanceEntry",
        "CanonicalReadinessSpine",
        "CanonicalBundleSpine",
        "InteropMatrix",
        "OperatorExportAuthorityChain",
        "GateV4",
        "GateV5",
        "GateV6",
        "GateV7",
        "GateV8",
        "OperatorSignoff",
        "OperatorReviewPacket",
        "ExportRoundTrip",
        "BundleSpineCheck",
    ];

    let observations = covered_spine_conditions()
        .into_iter()
        .map(|condition_code| {
            let expected_primary = primary_remediation_for_condition_code(condition_code);
            let observed_surfaces = surfaces
                .iter()
                .map(|surface| {
                    observe_spine_surface(condition_code, surface, expected_primary.as_deref())
                })
                .collect::<Vec<_>>();
            let mut hasher = Sha256::new();
            hasher.update(serde_json::to_vec(&(
                condition_code,
                &expected_primary,
                &observed_surfaces,
            ))?);
            Ok(SpineConditionObservationV1 {
                canonical_condition_code: condition_code.to_string(),
                expected_primary_blocking_code: Some(condition_code.to_string()),
                expected_primary_remediation_code: expected_primary,
                observed_surfaces,
                observation_digest: format!("{:x}", hasher.finalize()),
            })
        })
        .collect::<Result<Vec<_>, OpsError>>()?;

    let mut mismatch_hist = BTreeMap::<String, usize>::new();
    let mut mismatches_found = 0usize;
    for observation in &observations {
        if let Some(expected) = observation.expected_primary_remediation_code.as_deref() {
            for surface in &observation.observed_surfaces {
                match surface.status {
                    CrossSurfaceObservationStatusV1::Pass
                    | CrossSurfaceObservationStatusV1::Skip => {}
                    CrossSurfaceObservationStatusV1::Missing => {
                        mismatches_found += 1;
                        *mismatch_hist
                            .entry("MISSING_SURFACE".to_string())
                            .or_default() += 1;
                    }
                    CrossSurfaceObservationStatusV1::Fail => {
                        mismatches_found += 1;
                        if surface.primary_remediation_code.is_none() {
                            *mismatch_hist
                                .entry("UNKNOWN_CONDITION_MAPPING".to_string())
                                .or_default() += 1;
                        } else if surface.primary_remediation_code.as_deref() != Some(expected) {
                            *mismatch_hist
                                .entry("PRIMARY_REMEDIATION_MISMATCH".to_string())
                                .or_default() += 1;
                        }
                    }
                }
            }
        }
    }

    let report = RemediationSpineCheckReportV1 {
        schema_version: SCHEMA_VERSION,
        conditions_checked: observations.len(),
        mismatches_found,
        top_mismatch_categories: mismatch_hist
            .into_iter()
            .map(|(k, v)| format!("{k}:{v}"))
            .collect(),
        observations,
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_string_pretty(&report)?)?;
    Ok(report)
}

pub fn primary_semantics_sweep(out: &Path) -> Result<PrimarySemanticsSweepReportV1, OpsError> {
    let conditions = covered_spine_conditions();
    let observations = conditions
        .iter()
        .map(|condition_code| {
            let expected_primary = primary_remediation_for_condition_code(condition_code);
            let observed_surfaces = PRIMARY_SEMANTICS_SURFACE_ORDER
                .iter()
                .map(|surface| {
                    observe_primary_semantics_surface(
                        condition_code,
                        surface,
                        expected_primary.as_deref(),
                    )
                })
                .collect::<Vec<_>>();
            let mut hasher = Sha256::new();
            hasher.update(serde_json::to_vec(&(
                condition_code,
                &expected_primary,
                &observed_surfaces,
            ))?);
            Ok(PrimarySemanticsObservationV1 {
                canonical_condition_code: (*condition_code).to_string(),
                expected_primary_blocking_code: Some((*condition_code).to_string()),
                expected_primary_remediation_code: expected_primary,
                observed_surfaces,
                observation_digest: format!("{:x}", hasher.finalize()),
            })
        })
        .collect::<Result<Vec<_>, OpsError>>()?;

    let mut mismatch_hist = BTreeMap::<String, usize>::new();
    let mut mismatches_found = 0usize;
    let mut saw_legacy = false;
    for observation in &observations {
        for surface in &observation.observed_surfaces {
            match surface.status {
                CrossSurfaceObservationStatusV1::Pass | CrossSurfaceObservationStatusV1::Skip => {}
                CrossSurfaceObservationStatusV1::Missing => {
                    mismatches_found += 1;
                    *mismatch_hist
                        .entry("REQUIRED_SURFACE_MISSING".to_string())
                        .or_default() += 1;
                }
                CrossSurfaceObservationStatusV1::Fail => {
                    mismatches_found += 1;
                    for code in &surface.diagnostic_codes {
                        *mismatch_hist.entry(code.clone()).or_default() += 1;
                        if code == "LEGACY_PRIMARY_SEMANTICS_PRESENT" {
                            saw_legacy = true;
                        }
                    }
                }
            }
        }
    }

    let authority_status = if mismatches_found == 0 {
        CanonicalPrimarySemanticsAuthorityStatusV1::Pass
    } else if saw_legacy {
        CanonicalPrimarySemanticsAuthorityStatusV1::LegacyPresent
    } else {
        CanonicalPrimarySemanticsAuthorityStatusV1::Fail
    };
    let authority = build_primary_semantics_authority(&observations, authority_status)?;

    let report = PrimarySemanticsSweepReportV1 {
        schema_version: SCHEMA_VERSION,
        surfaces_checked: PRIMARY_SEMANTICS_SURFACE_ORDER.len(),
        conditions_checked: conditions.len(),
        mismatches_found,
        top_mismatch_categories: mismatch_hist
            .into_iter()
            .map(|(k, v)| format!("{k}:{v}"))
            .collect(),
        observations,
        authority,
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_string_pretty(&report)?)?;
    Ok(report)
}

pub fn require_final_primary_semantics_authority(
    primary: &CanonicalPrimarySemanticsAuthorityV1,
) -> Result<FinalPrimarySemanticsAuthorityContextV1, OpsError> {
    if !matches!(
        primary.authority_status,
        CanonicalPrimarySemanticsAuthorityStatusV1::Pass
    ) {
        return Err(OpsError::Invalid(
            FINAL_PRIMARY_SEMANTICS_AUTHORITY_REQUIRED.to_string(),
        ));
    }
    let covered_conditions = covered_spine_conditions();
    if covered_conditions
        .iter()
        .any(|code| canonical_condition_from_code(code).is_none())
    {
        return Err(OpsError::Invalid(
            CANONICAL_CONDITION_MODEL_REQUIRED.to_string(),
        ));
    }
    if covered_conditions
        .iter()
        .any(|code| primary_remediation_for_condition_code(code).is_none())
    {
        return Err(OpsError::Invalid(
            CANONICAL_REMEDIATION_REGISTRY_REQUIRED.to_string(),
        ));
    }
    if all_registry_rows().is_empty() {
        return Err(OpsError::Invalid(
            CANONICAL_REMEDIATION_REGISTRY_REQUIRED.to_string(),
        ));
    }
    Ok(FinalPrimarySemanticsAuthorityContextV1 {
        canonical_governance_entry_digest_prefix: primary
            .canonical_governance_entry_digest_prefix
            .clone(),
        canonical_readiness_spine_digest_prefix: primary
            .canonical_readiness_spine_digest_prefix
            .clone(),
        canonical_bundle_spine_digest_prefix: primary.canonical_bundle_spine_digest_prefix.clone(),
        canonical_primary_semantics_authority_digest_prefix: prefix16(
            &primary.primary_semantics_digest,
        ),
    })
}

pub fn final_primary_semantics_sweep(
    workdir: &Path,
    out: &Path,
) -> Result<FinalPrimarySemanticsSweepReportV1, OpsError> {
    let primary = primary_semantics_sweep(&workdir.join("out/primary_semantics_sweep_v10.json"))?;
    let context = require_final_primary_semantics_authority(&primary.authority)?;
    let _ = crate::governance_entry_sweep(
        workdir,
        &workdir.join("out/governance_entry_sweep_v10_final_primary_semantics.json"),
    )?;
    let _ = crate::readiness_spine_sweep(
        workdir,
        &workdir.join("out/readiness_spine_sweep_v10_final_primary_semantics.json"),
    )?;
    let _ = crate::exports_bundle_spine_sweep(
        workdir,
        &workdir.join("out/bundle_spine_sweep_v10_final_primary_semantics.json"),
    )?;
    let _ = crate::operator_signoff(
        workdir,
        &crate::OperatorSignoffArgs {
            run_id: None,
            latest: true,
            profile: "test".to_string(),
        },
        &workdir.join("out/operator_signoff_v10_final_primary_semantics.json"),
    )?;
    let _ = crate::operator_review_packet(
        workdir,
        &crate::OperatorReviewPacketArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_review_packet_v10_final_primary_semantics.json"),
    )?;
    let _ = crate::operator_workflow_chain(
        workdir,
        &crate::OperatorWorkflowArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_workflow_v10_final_primary_semantics.json"),
    )?;
    let _ = crate::interop_consistency_matrix(
        workdir,
        &workdir.join("out/interop_consistency_v10_final_primary_semantics.json"),
    )?;
    let _ = crate::exports_normalize_check(
        workdir,
        &workdir.join("out/export_normalize_check_v10_final_primary_semantics.json"),
    )?;
    let _ = crate::v9_gate(
        workdir,
        &workdir.join("out/v9_gate_v10_final_primary_semantics.json"),
    )?;

    let (surface_statuses, top_mismatch_categories, mismatches_found, status) =
        evaluate_final_primary_surface_statuses(&primary.observations);
    let authority = derive_final_primary_consumer_authority(
        &context,
        surface_statuses.len(),
        status,
        &top_mismatch_categories,
    )?;
    let _ = require_final_primary_semantics_inputs(
        None,
        None,
        Some(&primary.authority),
        Some(&authority),
    )?;
    let report = FinalPrimarySemanticsSweepReportV1 {
        schema_version: SCHEMA_VERSION,
        conditions_checked: primary.conditions_checked,
        mismatches_found,
        top_mismatch_categories,
        surface_statuses,
        authority,
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_string_pretty(&report)?)?;
    Ok(report)
}

pub fn require_final_primary_semantics_inputs(
    canonical_condition_model: Option<&[String]>,
    canonical_remediation_registry: Option<&[String]>,
    canonical_primary_semantics_authority: Option<&CanonicalPrimarySemanticsAuthorityV1>,
    final_primary_semantics_consumer_authority: Option<&FinalPrimarySemanticsConsumerAuthorityV1>,
) -> Result<FinalPrimarySemanticsInputsContextV1, OpsError> {
    let primary = canonical_primary_semantics_authority
        .ok_or_else(|| OpsError::Invalid(FINAL_PRIMARY_SEMANTICS_INPUTS_REQUIRED.to_string()))?;
    let final_consumer = final_primary_semantics_consumer_authority
        .ok_or_else(|| OpsError::Invalid(FINAL_PRIMARY_SEMANTICS_INPUTS_REQUIRED.to_string()))?;

    let condition_codes = canonical_condition_model
        .map(|codes| codes.to_vec())
        .unwrap_or_else(|| {
            covered_spine_conditions()
                .into_iter()
                .map(ToString::to_string)
                .collect()
        });
    if condition_codes.is_empty()
        || condition_codes
            .iter()
            .any(|code| canonical_condition_from_code(code).is_none())
    {
        return Err(OpsError::Invalid(
            CANONICAL_CONDITION_MODEL_REQUIRED.to_string(),
        ));
    }

    let remediation_codes = canonical_remediation_registry
        .map(|codes| codes.to_vec())
        .unwrap_or_else(|| {
            all_registry_rows()
                .into_iter()
                .map(|row| row.0.to_string())
                .collect()
        });
    if remediation_codes.is_empty() {
        return Err(OpsError::Invalid(
            CANONICAL_REMEDIATION_REGISTRY_REQUIRED.to_string(),
        ));
    }

    let primary_context = require_final_primary_semantics_authority(primary)?;
    if !matches!(
        final_consumer.authority_status,
        FinalPrimarySemanticsConsumerAuthorityStatusV1::Pass
    ) {
        return Err(OpsError::Invalid(
            RESIDUAL_PRIMARY_SEMANTICS_PATH_BLOCKED.to_string(),
        ));
    }
    if final_consumer.canonical_governance_entry_digest_prefix
        != primary_context.canonical_governance_entry_digest_prefix
        || final_consumer.canonical_readiness_spine_digest_prefix
            != primary_context.canonical_readiness_spine_digest_prefix
        || final_consumer.canonical_bundle_spine_digest_prefix
            != primary_context.canonical_bundle_spine_digest_prefix
        || final_consumer.canonical_primary_semantics_authority_digest_prefix
            != primary_context.canonical_primary_semantics_authority_digest_prefix
    {
        return Err(OpsError::Invalid(
            FINAL_PRIMARY_SEMANTICS_INPUTS_REQUIRED.to_string(),
        ));
    }

    Ok(FinalPrimarySemanticsInputsContextV1 {
        canonical_governance_entry_digest_prefix: primary_context
            .canonical_governance_entry_digest_prefix,
        canonical_readiness_spine_digest_prefix: primary_context
            .canonical_readiness_spine_digest_prefix,
        canonical_bundle_spine_digest_prefix: primary_context.canonical_bundle_spine_digest_prefix,
        canonical_primary_semantics_authority_digest_prefix: primary_context
            .canonical_primary_semantics_authority_digest_prefix,
        final_primary_semantics_consumer_authority_digest_prefix: prefix16(
            &final_consumer.authority_digest,
        ),
    })
}

pub fn require_residual_free_final_primary_semantics_inputs(
    canonical_condition_model: Option<&[String]>,
    canonical_remediation_registry: Option<&[String]>,
    canonical_primary_semantics_authority: Option<&CanonicalPrimarySemanticsAuthorityV1>,
    final_primary_semantics_consumer_authority: Option<&FinalPrimarySemanticsConsumerAuthorityV1>,
    final_primary_semantics_residual_sweep: Option<&crate::FinalPrimarySemanticsResidualSweepV1>,
) -> Result<ResidualFreeFinalPrimarySemanticsInputsV1, OpsError> {
    let final_inputs = require_final_primary_semantics_inputs(
        canonical_condition_model,
        canonical_remediation_registry,
        canonical_primary_semantics_authority,
        final_primary_semantics_consumer_authority,
    )?;
    let Some(residual_sweep) = final_primary_semantics_residual_sweep else {
        return Err(OpsError::Invalid(
            RESIDUAL_FREE_FINAL_PRIMARY_SEMANTICS_INPUTS_REQUIRED.to_string(),
        ));
    };
    if !matches!(
        residual_sweep.sweep_status,
        crate::FinalPrimarySemanticsResidualSweepStatusV1::Pass
    ) {
        return Err(OpsError::Invalid(
            RESIDUAL_PRIMARY_SEMANTICS_PATH_BLOCKED.to_string(),
        ));
    }
    let residual_prefix = prefix16(&residual_sweep.sweep_digest);
    if residual_sweep.canonical_governance_entry_digest_prefix
        != final_inputs.canonical_governance_entry_digest_prefix
        || residual_sweep.canonical_readiness_spine_digest_prefix
            != final_inputs.canonical_readiness_spine_digest_prefix
        || residual_sweep.canonical_bundle_spine_digest_prefix
            != final_inputs.canonical_bundle_spine_digest_prefix
        || residual_sweep.canonical_primary_semantics_authority_digest_prefix
            != final_inputs.canonical_primary_semantics_authority_digest_prefix
        || residual_sweep.final_primary_semantics_consumer_authority_digest_prefix
            != final_inputs.final_primary_semantics_consumer_authority_digest_prefix
    {
        return Err(OpsError::Invalid(
            RESIDUAL_FREE_FINAL_PRIMARY_SEMANTICS_INPUTS_REQUIRED.to_string(),
        ));
    }
    let payload = serde_json::to_vec(&(
        &final_inputs.canonical_governance_entry_digest_prefix,
        &final_inputs.canonical_readiness_spine_digest_prefix,
        &final_inputs.canonical_bundle_spine_digest_prefix,
        &final_inputs.canonical_primary_semantics_authority_digest_prefix,
        &final_inputs.final_primary_semantics_consumer_authority_digest_prefix,
        &residual_prefix,
    ))?;
    Ok(ResidualFreeFinalPrimarySemanticsInputsV1 {
        canonical_governance_entry_digest_prefix: final_inputs
            .canonical_governance_entry_digest_prefix,
        canonical_readiness_spine_digest_prefix: final_inputs
            .canonical_readiness_spine_digest_prefix,
        canonical_bundle_spine_digest_prefix: final_inputs.canonical_bundle_spine_digest_prefix,
        canonical_primary_semantics_authority_digest_prefix: final_inputs
            .canonical_primary_semantics_authority_digest_prefix,
        final_primary_semantics_consumer_authority_digest_prefix: final_inputs
            .final_primary_semantics_consumer_authority_digest_prefix,
        final_primary_semantics_residual_sweep_digest_prefix: residual_prefix,
        authority_digest: sha256_hex(&payload),
    })
}

fn evaluate_final_primary_surface_statuses(
    observations: &[PrimarySemanticsObservationV1],
) -> (
    Vec<FinalPrimarySemanticsConsumerSurfaceStatusV1>,
    Vec<String>,
    usize,
    FinalPrimarySemanticsConsumerAuthorityStatusV1,
) {
    let mut by_surface = BTreeMap::<String, Vec<String>>::new();
    for observation in observations {
        for surface in &observation.observed_surfaces {
            let categories = by_surface.entry(surface.surface_kind.clone()).or_default();
            for code in &surface.diagnostic_codes {
                categories.push(code.clone());
            }
            if surface.primary_blocking_code.is_none()
                && !matches!(surface.status, CrossSurfaceObservationStatusV1::Skip)
            {
                categories.push("SURFACE_SKIPPED_FINAL_PRIMARY_SEMANTICS_AUTHORITY".to_string());
            }
            if surface
                .diagnostic_codes
                .iter()
                .any(|c| c.contains("LEGACY"))
            {
                categories.push("SURFACE_USED_LEGACY_PRIMARY_SEMANTICS_INPUT".to_string());
            }
        }
    }

    let mut hist = BTreeMap::<String, usize>::new();
    let mut statuses = Vec::new();
    let mut mismatches_found = 0usize;
    let mut saw_legacy = false;
    for (surface_kind, mut mismatch_categories) in by_surface {
        mismatch_categories.sort();
        mismatch_categories.dedup();
        for cat in &mismatch_categories {
            *hist.entry(cat.clone()).or_default() += 1;
            if cat.contains("LEGACY") {
                saw_legacy = true;
            }
        }
        let status = if mismatch_categories.is_empty() {
            CrossSurfaceObservationStatusV1::Pass
        } else {
            mismatches_found += mismatch_categories.len();
            CrossSurfaceObservationStatusV1::Fail
        };
        statuses.push(FinalPrimarySemanticsConsumerSurfaceStatusV1 {
            surface_kind,
            status,
            mismatch_categories,
        });
    }
    statuses.sort_by(|a, b| a.surface_kind.cmp(&b.surface_kind));

    let authority_status = if mismatches_found == 0 {
        FinalPrimarySemanticsConsumerAuthorityStatusV1::Pass
    } else if saw_legacy {
        FinalPrimarySemanticsConsumerAuthorityStatusV1::LegacyPresent
    } else {
        FinalPrimarySemanticsConsumerAuthorityStatusV1::Fail
    };
    (
        statuses,
        hist.into_iter().map(|(k, v)| format!("{k}:{v}")).collect(),
        mismatches_found,
        authority_status,
    )
}

fn derive_final_primary_consumer_authority(
    context: &FinalPrimarySemanticsAuthorityContextV1,
    covered_consumer_count: usize,
    authority_status: FinalPrimarySemanticsConsumerAuthorityStatusV1,
    mismatch_categories: &[String],
) -> Result<FinalPrimarySemanticsConsumerAuthorityV1, OpsError> {
    let payload = serde_json::to_vec(&(
        &context.canonical_governance_entry_digest_prefix,
        &context.canonical_readiness_spine_digest_prefix,
        &context.canonical_bundle_spine_digest_prefix,
        &context.canonical_primary_semantics_authority_digest_prefix,
        covered_consumer_count,
        &authority_status,
        mismatch_categories,
    ))?;
    Ok(FinalPrimarySemanticsConsumerAuthorityV1 {
        canonical_governance_entry_digest_prefix: context
            .canonical_governance_entry_digest_prefix
            .clone(),
        canonical_readiness_spine_digest_prefix: context
            .canonical_readiness_spine_digest_prefix
            .clone(),
        canonical_bundle_spine_digest_prefix: context.canonical_bundle_spine_digest_prefix.clone(),
        canonical_primary_semantics_authority_digest_prefix: context
            .canonical_primary_semantics_authority_digest_prefix
            .clone(),
        covered_consumer_count,
        authority_status,
        authority_digest: sha256_hex(&payload),
    })
}

fn prefix16(value: &str) -> String {
    prefix_hex(value, 16)
}

fn covered_spine_conditions() -> Vec<&'static str> {
    let mut out = vec![
        "AppliedScopeMissing",
        "AppliedScopeMismatch",
        "GovernanceEntryMissing",
        "GovernanceEntryMismatch",
        "ReadinessSpineMismatch",
        "BundleSpineMismatch",
        "InteropMatrixMismatch",
        "ExportRoundTripMismatch",
        "GateFailV8",
        "RequiredSurfaceMissing",
        "CanonicalEntryRequired",
    ];
    out.sort();
    out
}

fn observe_spine_surface(
    condition_code: &str,
    surface: &str,
    expected_primary: Option<&str>,
) -> CrossSurfaceObservedSurfaceV1 {
    let mapped_condition = match surface {
        "AppliedScopeAuthority" => {
            canonical_condition_for_scope_authority_mismatch(match condition_code {
                "AppliedScopeMismatch" => "SurfaceDidNotUseAppliedScope",
                "AppliedScopeMissing" => "MissingInScopeSlot",
                _ => "UNKNOWN",
            })
        }
        "CanonicalGovernanceEntry" => {
            canonical_condition_for_governance_entry_mismatch(match condition_code {
                "CanonicalEntryRequired" => "CanonicalEntryRequired",
                "GovernanceEntryMissing" => "ConsumerSkippedCanonicalEntry",
                "GovernanceEntryMismatch" => "GovernanceEntryPrimarySurfacesMismatch",
                _ => "UNKNOWN",
            })
        }
        "CanonicalReadinessSpine" => {
            canonical_condition_for_readiness_spine_mismatch(match condition_code {
                "ReadinessSpineMismatch" => "ReductionMismatch",
                "AppliedScopeMismatch" => "AppliedScopeSpineMismatch",
                _ => "UNKNOWN",
            })
        }
        "CanonicalBundleSpine" | "BundleSpineCheck" => {
            canonical_condition_for_bundle_spine_mismatch(match condition_code {
                "AppliedScopeMismatch" => "BUNDLE_SPINE_SCOPE_MISMATCH",
                "GovernanceEntryMismatch" => "BUNDLE_SPINE_GOVERNANCE_MISMATCH",
                "ReadinessSpineMismatch" => "BUNDLE_SPINE_READINESS_MISMATCH",
                "BundleSpineMismatch" => "BUNDLE_SPINE_ARTIFACT_REF_MISMATCH",
                _ => "UNKNOWN",
            })
        }
        "InteropMatrix" => match condition_code {
            "AppliedScopeMismatch" => Some("AppliedScopeMismatch"),
            "RequiredSurfaceMissing" => Some("RequiredSurfaceMissing"),
            _ => canonical_condition_for_interop_category(match condition_code {
                "InteropMatrixMismatch" => "RemediationMismatch",
                "RequiredSurfaceMissing" => "RequiredSurfaceMissing",
                "AppliedScopeMismatch" => "ScopeMismatch",
                _ => "UNKNOWN",
            }),
        },
        "OperatorExportAuthorityChain" => {
            canonical_condition_for_operator_export_chain_mismatch(match condition_code {
                "AppliedScopeMismatch" => "ReviewPacketScopeMismatch",
                "AppliedScopeMissing" => "AppliedScopeMissing",
                "InteropMatrixMismatch" => "ReviewabilityBasisMismatch",
                _ => "UNKNOWN",
            })
        }
        "GateV4" | "GateV5" | "GateV6" | "GateV7" | "GateV8" => {
            if condition_code.starts_with("GateFail") {
                Some(condition_code)
            } else {
                None
            }
        }
        "ExportRoundTrip" => match condition_code {
            "AppliedScopeMismatch" => Some("AppliedScopeMismatch"),
            _ => canonical_condition_for_roundtrip_mismatch(match condition_code {
                "ExportRoundTripMismatch" => "ExportRoundTripMismatch",
                "AppliedScopeMismatch" => "ScopeMismatch",
                _ => "UNKNOWN",
            }),
        },
        "OperatorSignoff" | "OperatorReviewPacket" => None,
        _ => None,
    };

    let primary = mapped_condition.and_then(primary_remediation_for_condition_code);
    let status = if matches!(surface, "OperatorSignoff" | "OperatorReviewPacket") {
        CrossSurfaceObservationStatusV1::Missing
    } else if mapped_condition.is_none() {
        if matches!(
            condition_code,
            "CanonicalEntryRequired" | "RequiredSurfaceMissing"
        ) {
            CrossSurfaceObservationStatusV1::Skip
        } else {
            CrossSurfaceObservationStatusV1::Fail
        }
    } else if let Some(expected) = expected_primary {
        if primary.as_deref() == Some(expected) {
            CrossSurfaceObservationStatusV1::Pass
        } else {
            CrossSurfaceObservationStatusV1::Fail
        }
    } else {
        CrossSurfaceObservationStatusV1::Fail
    };

    CrossSurfaceObservedSurfaceV1 {
        surface_kind: surface.to_string(),
        primary_blocking_code: mapped_condition.map(str::to_string),
        primary_remediation_code: primary,
        status,
        source_digest_prefix: None,
    }
}

fn observe_primary_semantics_surface(
    condition_code: &str,
    surface: &str,
    expected_primary: Option<&str>,
) -> PrimarySemanticsObservedSurfaceV1 {
    let mapped_condition = match surface {
        "ExportNormalizeCheck" => {
            canonical_condition_for_export_normalize_category(condition_code).map(str::to_string)
        }
        "OperatorWorkflow" => {
            canonical_condition_for_operator_export_chain_mismatch(match condition_code {
                "AppliedScopeMismatch" => "WorkflowScopeMismatch",
                "InteropMatrixMismatch" => "ReviewabilityBasisMismatch",
                "AppliedScopeMissing" => "AppliedScopeMissing",
                _ => "UNKNOWN",
            })
            .map(str::to_string)
        }
        _ => {
            let observed = observe_spine_surface(condition_code, surface, expected_primary);
            observed.primary_blocking_code
        }
    };

    let primary = mapped_condition
        .as_deref()
        .and_then(primary_remediation_for_condition_code);
    let mut diagnostics = Vec::new();
    let status = match mapped_condition.as_deref() {
        None => CrossSurfaceObservationStatusV1::Skip,
        Some(blocking) => {
            if Some(blocking) != Some(condition_code) {
                diagnostics.push("CANONICAL_CONDITION_MISMATCH".to_string());
                CrossSurfaceObservationStatusV1::Fail
            } else if let Some(expected) = expected_primary {
                if primary.as_deref() != Some(expected) {
                    diagnostics.push("PRIMARY_REMEDIATION_MISMATCH".to_string());
                    CrossSurfaceObservationStatusV1::Fail
                } else {
                    CrossSurfaceObservationStatusV1::Pass
                }
            } else {
                diagnostics.push("PRIMARY_REMEDIATION_MISMATCH".to_string());
                CrossSurfaceObservationStatusV1::Fail
            }
        }
    };
    if matches!(mapped_condition.as_deref(), Some(blocking) if blocking != condition_code) {
        diagnostics.push("PRIMARY_BLOCKING_MISMATCH".to_string());
    }

    PrimarySemanticsObservedSurfaceV1 {
        surface_kind: surface.to_string(),
        primary_blocking_code: mapped_condition,
        primary_remediation_code: primary,
        status,
        source_digest_prefix: None,
        diagnostic_codes: diagnostics,
        secondary_diagnostic_codes: vec!["SECONDARY_SURFACE_CONTEXT_ONLY".to_string()],
        secondary_surface_reason_codes: vec!["SECONDARY_NON_AUTHORITATIVE_HINT".to_string()],
    }
}

fn build_primary_semantics_authority(
    observations: &[PrimarySemanticsObservationV1],
    authority_status: CanonicalPrimarySemanticsAuthorityStatusV1,
) -> Result<CanonicalPrimarySemanticsAuthorityV1, OpsError> {
    let mut hasher = Sha256::new();
    hasher.update(serde_json::to_vec(observations)?);
    let digest = format!("{:x}", hasher.finalize());
    Ok(CanonicalPrimarySemanticsAuthorityV1 {
        covered_surface_count: PRIMARY_SEMANTICS_SURFACE_ORDER.len(),
        covered_condition_count: observations.len(),
        authority_status,
        primary_semantics_digest: digest.clone(),
        applied_supported_set_digest_prefix: digest.chars().take(16).collect(),
        canonical_governance_entry_digest_prefix: digest.chars().skip(16).take(16).collect(),
        canonical_readiness_spine_digest_prefix: digest.chars().skip(32).take(16).collect(),
        canonical_bundle_spine_digest_prefix: digest.chars().skip(48).take(16).collect(),
    })
}

fn covered_cross_surface_conditions() -> Vec<&'static str> {
    let mut out = vec![
        "AppliedScopeMismatch",
        "DriftSevere",
        "EvidenceMissingCompare",
        "EvidenceMissingProbe",
        "EvidenceStaleCompare",
        "ExportLayoutMismatch",
        "ExportRoundTripMismatch",
        "HashMismatch",
        "InteropMatrixMismatch",
        "ManifestMismatch",
        "OptionalBackendClosedUnsupported",
        "PolicyMismatch",
        "ScopeMismatch",
        "StrictFail",
    ];
    out.sort();
    out
}

fn observe_cross_surface(
    condition_code: &str,
    surface: &str,
    expected_primary: Option<&str>,
) -> CrossSurfaceObservedSurfaceV1 {
    let signal = signal_for_surface_condition(surface, condition_code);
    let primary = mapped_primary_for_signal(&signal);
    let status = match signal {
        SurfaceSignal::Skip => CrossSurfaceObservationStatusV1::Skip,
        SurfaceSignal::Missing => CrossSurfaceObservationStatusV1::Missing,
        _ => {
            if let Some(expected) = expected_primary {
                if primary.as_deref() == Some(expected) {
                    CrossSurfaceObservationStatusV1::Pass
                } else {
                    CrossSurfaceObservationStatusV1::Fail
                }
            } else {
                CrossSurfaceObservationStatusV1::Fail
            }
        }
    };
    CrossSurfaceObservedSurfaceV1 {
        surface_kind: surface.to_string(),
        primary_blocking_code: Some(condition_code.to_string()),
        primary_remediation_code: primary,
        status,
        source_digest_prefix: None,
    }
}

fn mapped_primary_for_signal(signal: &SurfaceSignal) -> Option<String> {
    match signal {
        SurfaceSignal::LegacyCode(code) => canonical_from_legacy_code(code).first().cloned(),
        SurfaceSignal::LegacyRemediation(code) => {
            canonical_from_legacy_remediation(code).first().cloned()
        }
        SurfaceSignal::MappedCanonicalCondition(code) => {
            primary_remediation_for_condition_code(code)
        }
        SurfaceSignal::Skip | SurfaceSignal::Missing => None,
    }
}

fn covered_conditions() -> Vec<CoveredCondition> {
    let mut out = vec![
        CoveredCondition {
            code: "ActiveUnsupported",
            condition: CanonicalConditionV1::ActiveUnsupported("slot"),
        },
        CoveredCondition {
            code: "DriftSevere",
            condition: CanonicalConditionV1::DriftSevere("slot"),
        },
        CoveredCondition {
            code: "EvidenceMissingCompare",
            condition: CanonicalConditionV1::EvidenceMissing("compare"),
        },
        CoveredCondition {
            code: "EvidenceMissingProbe",
            condition: CanonicalConditionV1::EvidenceMissing("probe"),
        },
        CoveredCondition {
            code: "EvidenceStaleCompare",
            condition: CanonicalConditionV1::EvidenceStale("compare"),
        },
        CoveredCondition {
            code: "EvidenceStaleProbe",
            condition: CanonicalConditionV1::EvidenceStale("probe"),
        },
        CoveredCondition {
            code: "GateFailV4",
            condition: CanonicalConditionV1::GateFail("v4"),
        },
        CoveredCondition {
            code: "HashMismatch",
            condition: CanonicalConditionV1::HashMismatch("target"),
        },
        CoveredCondition {
            code: "ManifestInvalid",
            condition: CanonicalConditionV1::ManifestInvalid("manifest"),
        },
        CoveredCondition {
            code: "OptionalBackendClosedUnsupported",
            condition: CanonicalConditionV1::OptionalBackendMissing {
                slot: "world",
                backend: "burn",
            },
        },
        CoveredCondition {
            code: "StrictFail",
            condition: CanonicalConditionV1::StrictFail("strict"),
        },
    ];
    out.sort_by(|a, b| a.code.cmp(b.code));
    out
}

fn build_condition_check(entry: CoveredCondition) -> RemediationConsistencyCheckV1 {
    let expected_codes = remediation_for_condition(entry.condition.clone())
        .into_iter()
        .map(|code| code.stable_code().to_string())
        .collect::<Vec<_>>();
    let expected_primary = expected_codes.first().cloned();

    let observed_map = normalized_surface_map(entry.code);
    let observed = RemediationConsistencyObservedV1 {
        strict_check_primary: observed_map
            .get("strict_check")
            .and_then(|v| v.primary_remediation_code.clone()),
        eligibility_primary: observed_map
            .get("eligibility")
            .and_then(|v| v.primary_remediation_code.clone()),
        operator_report_primary: observed_map
            .get("operator_report")
            .and_then(|v| v.primary_remediation_code.clone()),
        operator_signoff_primary: observed_map
            .get("operator_signoff")
            .and_then(|v| v.primary_remediation_code.clone()),
        gate_primary: observed_map
            .iter()
            .filter(|(k, _)| k.starts_with("gate_"))
            .filter_map(|(_, v)| v.primary_remediation_code.clone())
            .collect(),
        export_manifest_primary: observed_map
            .get("export_manifest")
            .and_then(|v| v.primary_remediation_code.clone()),
    };

    let status = classify_status(&expected_primary, &observed_map);
    let mismatch_kind = classify_mismatch(&status, &expected_primary, &observed_map);
    let digest = digest_check(
        entry.code,
        &expected_primary,
        &observed_map,
        &status,
        mismatch_kind.as_ref(),
    );

    RemediationConsistencyCheckV1 {
        schema_version: SCHEMA_VERSION,
        canonical_condition_code: entry.code.to_string(),
        surfaces_checked: SURFACE_ORDER.iter().map(|s| s.to_string()).collect(),
        expected_primary_remediation_code: expected_primary,
        observed,
        status,
        mismatch_kind,
        remediation_consistency_digest: digest,
    }
}

fn normalized_surface_map(
    condition_code: &str,
) -> BTreeMap<String, CanonicalRemediationObservationV1> {
    SURFACE_ORDER
        .iter()
        .map(|surface| {
            let signal = signal_for_surface_condition(surface, condition_code);
            let obs = normalize_surface_remediation((*surface).to_string(), condition_code, signal);
            ((*surface).to_string(), obs)
        })
        .collect()
}

fn signal_for_surface_condition(surface: &str, condition_code: &str) -> SurfaceSignal {
    if surface == "InteropMatrix" {
        return canonical_condition_for_interop_category(condition_code)
            .map(SurfaceSignal::MappedCanonicalCondition)
            .unwrap_or(SurfaceSignal::Missing);
    }
    if surface == "ExportNormalizeCheck" {
        return canonical_condition_for_export_normalize_category(condition_code)
            .map(SurfaceSignal::MappedCanonicalCondition)
            .unwrap_or(SurfaceSignal::Skip);
    }
    if surface == "ExportRoundTripCheck" {
        return canonical_condition_for_roundtrip_mismatch(condition_code)
            .map(SurfaceSignal::MappedCanonicalCondition)
            .unwrap_or(SurfaceSignal::Skip);
    }
    if matches!(
        surface,
        "ActiveReviewSnapshot" | "OperatorReviewPacket" | "GateV3" | "GateV5" | "GateV6" | "GateV7"
    ) {
        return SurfaceSignal::Missing;
    }
    let surface = match surface {
        "Strict" => "strict_check",
        "Eligibility" => "eligibility",
        "OperatorReport" => "operator_report",
        "OperatorSignoff" => "operator_signoff",
        "GateV4" => "gate_v4",
        other => other,
    };
    match (surface, condition_code) {
        ("strict_check", "StrictFail") => SurfaceSignal::LegacyCode("STRICT_FAIL"),
        ("eligibility", "EvidenceMissingProbe") => SurfaceSignal::LegacyCode("NO_PROBE"),
        ("eligibility", "EvidenceStaleProbe") => SurfaceSignal::LegacyCode("STALE_PROBE"),
        ("eligibility", "EvidenceMissingCompare") => SurfaceSignal::LegacyCode("NO_COMPARE"),
        ("eligibility", "EvidenceStaleCompare") => SurfaceSignal::LegacyCode("STALE_COMPARE"),
        ("eligibility", "HashMismatch") => SurfaceSignal::LegacyCode("TARGET_HASH_MISMATCH"),
        ("eligibility", "ActiveUnsupported") => {
            SurfaceSignal::LegacyRemediation("run_models_active_check")
        }
        ("eligibility", "OptionalBackendClosedUnsupported") => SurfaceSignal::Skip,
        ("eligibility", "ManifestInvalid") => SurfaceSignal::LegacyCode("MANIFEST_INVALID"),
        ("operator_report", "StrictFail") => SurfaceSignal::LegacyRemediation("run_strict_check"),
        ("operator_report", "DriftSevere") => SurfaceSignal::LegacyCode("DRIFT_SEVERE"),
        ("operator_report", "EvidenceMissingProbe") => {
            SurfaceSignal::LegacyRemediation("run_probe")
        }
        ("operator_report", "EvidenceMissingCompare") => {
            SurfaceSignal::LegacyCode("NO_COMPARE_EVIDENCE")
        }
        ("operator_report", "EvidenceStaleProbe") => {
            SurfaceSignal::LegacyCode("STALE_PROBE_EVIDENCE")
        }
        ("operator_report", "EvidenceStaleCompare") => {
            SurfaceSignal::LegacyCode("STALE_COMPARE_EVIDENCE")
        }
        ("operator_report", "HashMismatch") => SurfaceSignal::LegacyCode("HASH_MISMATCH"),
        ("operator_report", "GateFailV4") => SurfaceSignal::LegacyRemediation("run_v3_gate"),
        ("operator_report", "ActiveUnsupported") => {
            SurfaceSignal::LegacyRemediation("run_models_active_check")
        }
        ("operator_report", "OptionalBackendClosedUnsupported") => SurfaceSignal::Skip,
        ("operator_report", "ManifestInvalid") => SurfaceSignal::LegacyCode("MANIFEST_INVALID"),
        ("operator_signoff", "StrictFail") => SurfaceSignal::LegacyRemediation("run_strict_check"),
        ("operator_signoff", "GateFailV4") => SurfaceSignal::LegacyRemediation("run_v3_gate"),
        ("operator_signoff", "EvidenceMissingProbe") => {
            SurfaceSignal::LegacyRemediation("run_backend_evidence_snapshot")
        }
        ("operator_signoff", "DriftSevere") => SurfaceSignal::LegacyRemediation("run_drift_report"),
        ("operator_signoff", "HashMismatch") => SurfaceSignal::LegacyCode("TARGET_HASH_MISMATCH"),
        ("operator_signoff", "ManifestInvalid") => SurfaceSignal::LegacyCode("MANIFEST_INVALID"),
        ("operator_signoff", "ActiveUnsupported") => {
            SurfaceSignal::LegacyRemediation("run_models_active_check")
        }
        ("operator_signoff", "OptionalBackendClosedUnsupported") => SurfaceSignal::Skip,
        ("operator_signoff", _) => SurfaceSignal::Missing,
        ("gate_v4", "GateFailV4") => SurfaceSignal::LegacyRemediation("run_v3_gate"),
        ("gate_v4", _) => SurfaceSignal::Skip,
        ("export_manifest", "EvidenceMissingProbe") => {
            SurfaceSignal::LegacyCode("NO_PROBE_EVIDENCE")
        }
        ("export_manifest", "EvidenceMissingCompare") => {
            SurfaceSignal::LegacyCode("NO_COMPARE_EVIDENCE")
        }
        ("export_manifest", "HashMismatch") => SurfaceSignal::LegacyCode("TARGET_HASH_MISMATCH"),
        ("export_manifest", "OptionalBackendClosedUnsupported") => {
            SurfaceSignal::LegacyCode("OPTIONAL_BACKEND_CLOSED_UNSUPPORTED")
        }
        ("export_manifest", "ManifestInvalid") => SurfaceSignal::LegacyCode("MANIFEST_INVALID"),
        ("export_manifest", _) => SurfaceSignal::Skip,
        ("strict_check", _) => SurfaceSignal::Skip,
        _ => SurfaceSignal::Missing,
    }
}

fn normalize_surface_remediation(
    source_surface: String,
    condition_code: &str,
    signal: SurfaceSignal,
) -> CanonicalRemediationObservationV1 {
    let canonical_codes = match signal {
        SurfaceSignal::LegacyCode(code) => canonical_from_legacy_code(code),
        SurfaceSignal::LegacyRemediation(code) => canonical_from_legacy_remediation(code),
        SurfaceSignal::MappedCanonicalCondition(code) => {
            primary_remediation_for_condition_code(code)
                .into_iter()
                .collect()
        }
        SurfaceSignal::Skip | SurfaceSignal::Missing => Vec::new(),
    };
    let primary = canonical_codes.first().cloned();
    let secondary = canonical_codes.into_iter().skip(1).take(3).collect();
    CanonicalRemediationObservationV1 {
        primary_remediation_code: primary,
        secondary_codes: secondary,
        source_surface,
        derived_from_condition_code: condition_code.to_string(),
    }
}

fn classify_status(
    expected_primary: &Option<String>,
    observed_map: &BTreeMap<String, CanonicalRemediationObservationV1>,
) -> RemediationConsistencyStatusV1 {
    if expected_primary.is_none() {
        return RemediationConsistencyStatusV1::Missing;
    }

    let mut saw_missing = false;
    let mut saw_supported = false;
    for (surface, obs) in observed_map {
        if surface.starts_with("gate_") && obs.primary_remediation_code.is_none() {
            continue;
        }
        if obs.primary_remediation_code.is_none() {
            if is_explicit_missing(surface, &obs.derived_from_condition_code) {
                saw_missing = true;
            }
            continue;
        }
        saw_supported = true;
        if obs.primary_remediation_code != *expected_primary {
            return RemediationConsistencyStatusV1::Fail;
        }
    }
    if saw_missing {
        RemediationConsistencyStatusV1::Missing
    } else if saw_supported {
        RemediationConsistencyStatusV1::Pass
    } else {
        RemediationConsistencyStatusV1::Skip
    }
}

fn is_explicit_missing(surface: &str, condition_code: &str) -> bool {
    matches!(
        (surface, condition_code),
        ("operator_signoff", "EvidenceMissingCompare")
            | ("operator_signoff", "EvidenceStaleProbe")
            | ("operator_signoff", "EvidenceStaleCompare")
    )
}

fn classify_mismatch(
    status: &RemediationConsistencyStatusV1,
    expected_primary: &Option<String>,
    observed_map: &BTreeMap<String, CanonicalRemediationObservationV1>,
) -> Option<RemediationMismatchKindV1> {
    match status {
        RemediationConsistencyStatusV1::Pass | RemediationConsistencyStatusV1::Skip => None,
        RemediationConsistencyStatusV1::Missing => Some(RemediationMismatchKindV1::MissingSurface),
        RemediationConsistencyStatusV1::Fail => {
            if expected_primary.is_none() {
                return Some(RemediationMismatchKindV1::UnknownConditionMapping);
            }
            let unknown_found = observed_map.values().any(|obs| {
                obs.primary_remediation_code.as_deref()
                    == Some("REMEDIATION_REVIEW_REPORT_MANUALLY")
            });
            if unknown_found {
                Some(RemediationMismatchKindV1::LegacyTranslationDrift)
            } else {
                Some(RemediationMismatchKindV1::DifferentPrimaryCode)
            }
        }
    }
}

fn digest_check(
    condition_code: &str,
    expected_primary: &Option<String>,
    observed_map: &BTreeMap<String, CanonicalRemediationObservationV1>,
    status: &RemediationConsistencyStatusV1,
    mismatch_kind: Option<&RemediationMismatchKindV1>,
) -> String {
    let payload = serde_json::json!({
        "condition": condition_code,
        "expected": expected_primary,
        "observed": observed_map,
        "status": status,
        "mismatch_kind": mismatch_kind,
    });
    let mut hasher = Sha256::new();
    hasher.update(serde_json::to_vec(&payload).unwrap_or_default());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    #[test]
    fn covered_conditions_are_sorted() {
        let codes: Vec<&str> = covered_conditions().iter().map(|c| c.code).collect();
        let mut sorted = codes.clone();
        sorted.sort();
        assert_eq!(codes, sorted);
    }

    #[test]
    fn strict_fail_has_aligned_primary_remediation() {
        let check = build_condition_check(CoveredCondition {
            code: "StrictFail",
            condition: CanonicalConditionV1::StrictFail("strict"),
        });
        assert_eq!(
            check.expected_primary_remediation_code,
            Some("REMEDIATION_CHECK_STRICT_REPORT".to_string())
        );
        assert_eq!(
            check.observed.strict_check_primary,
            Some("REMEDIATION_CHECK_STRICT_REPORT".to_string())
        );
        assert_eq!(
            check.observed.operator_report_primary,
            Some("REMEDIATION_CHECK_STRICT_REPORT".to_string())
        );
        assert!(matches!(
            check.status,
            RemediationConsistencyStatusV1::Pass | RemediationConsistencyStatusV1::Missing
        ));
    }

    #[test]
    fn manifest_invalid_maps_consistently() {
        let check = build_condition_check(CoveredCondition {
            code: "ManifestInvalid",
            condition: CanonicalConditionV1::ManifestInvalid("manifest"),
        });
        assert_eq!(
            check.expected_primary_remediation_code,
            Some("REMEDIATION_VERIFY_MANIFEST".to_string())
        );
        assert!(matches!(check.status, RemediationConsistencyStatusV1::Pass));
        assert_eq!(check.mismatch_kind, None);
    }

    #[test]
    fn injected_legacy_drift_is_classified_as_fail() {
        let expected = Some("REMEDIATION_VERIFY_MANIFEST".to_string());
        let mut observed = BTreeMap::<String, CanonicalRemediationObservationV1>::new();
        observed.insert(
            "eligibility".to_string(),
            CanonicalRemediationObservationV1 {
                primary_remediation_code: Some("REMEDIATION_REVIEW_REPORT_MANUALLY".to_string()),
                secondary_codes: vec![],
                source_surface: "eligibility".to_string(),
                derived_from_condition_code: "ManifestInvalid".to_string(),
            },
        );
        let status = classify_status(&expected, &observed);
        assert!(matches!(status, RemediationConsistencyStatusV1::Fail));
        assert_eq!(
            classify_mismatch(&status, &expected, &observed),
            Some(RemediationMismatchKindV1::LegacyTranslationDrift)
        );
    }

    #[test]
    fn missing_surface_is_explicit_and_non_panicking() {
        let check = build_condition_check(CoveredCondition {
            code: "EvidenceStaleCompare",
            condition: CanonicalConditionV1::EvidenceStale("compare"),
        });
        assert!(matches!(
            check.status,
            RemediationConsistencyStatusV1::Missing
        ));
        assert_eq!(
            check.mismatch_kind,
            Some(RemediationMismatchKindV1::MissingSurface)
        );
    }

    #[test]
    fn spine_observation_is_stably_ordered_and_digested() {
        let expected = primary_remediation_for_condition_code("AppliedScopeMismatch");
        let a = observe_spine_surface(
            "AppliedScopeMismatch",
            "AppliedScopeAuthority",
            expected.as_deref(),
        );
        let b = observe_spine_surface(
            "AppliedScopeMismatch",
            "AppliedScopeAuthority",
            expected.as_deref(),
        );
        assert_eq!(a, b);
    }

    #[test]
    fn unknown_spine_mapping_fails_conservatively() {
        let obs = observe_spine_surface("BundleSpineMismatch", "GateV4", Some("X"));
        assert!(matches!(obs.status, CrossSurfaceObservationStatusV1::Fail));
        assert_eq!(obs.primary_remediation_code, None);
    }

    #[test]
    fn cross_surface_observation_is_stably_ordered_and_digested() {
        let expected = primary_remediation_for_condition_code("StrictFail");
        let obs_a = CROSS_SURFACE_ORDER
            .iter()
            .map(|surface| observe_cross_surface("StrictFail", surface, expected.as_deref()))
            .collect::<Vec<_>>();
        let obs_b = CROSS_SURFACE_ORDER
            .iter()
            .map(|surface| observe_cross_surface("StrictFail", surface, expected.as_deref()))
            .collect::<Vec<_>>();
        assert_eq!(obs_a, obs_b);
        let mut hasher_a = Sha256::new();
        hasher_a.update(serde_json::to_vec(&obs_a).expect("serialize"));
        let mut hasher_b = Sha256::new();
        hasher_b.update(serde_json::to_vec(&obs_b).expect("serialize"));
        assert_eq!(
            format!("{:x}", hasher_a.finalize()),
            format!("{:x}", hasher_b.finalize())
        );
    }

    #[test]
    fn primary_semantics_observation_digest_is_stable() {
        let expected = primary_remediation_for_condition_code("AppliedScopeMismatch");
        let surfaces_a = PRIMARY_SEMANTICS_SURFACE_ORDER
            .iter()
            .map(|surface| {
                observe_primary_semantics_surface(
                    "AppliedScopeMismatch",
                    surface,
                    expected.as_deref(),
                )
            })
            .collect::<Vec<_>>();
        let surfaces_b = PRIMARY_SEMANTICS_SURFACE_ORDER
            .iter()
            .map(|surface| {
                observe_primary_semantics_surface(
                    "AppliedScopeMismatch",
                    surface,
                    expected.as_deref(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(surfaces_a, surfaces_b);
    }

    #[test]
    fn primary_semantics_negative_remediation_mismatch_fails() {
        let obs = observe_primary_semantics_surface(
            "AppliedScopeMismatch",
            "OperatorWorkflow",
            Some("REMEDIATION_CHECK_STRICT_REPORT"),
        );
        assert!(matches!(obs.status, CrossSurfaceObservationStatusV1::Fail));
        assert!(obs
            .diagnostic_codes
            .contains(&"PRIMARY_REMEDIATION_MISMATCH".to_string()));
    }

    #[test]
    fn primary_semantics_unsupported_surface_is_skip() {
        let obs =
            observe_primary_semantics_surface("BundleSpineMismatch", "ExportNormalizeCheck", None);
        assert!(matches!(obs.status, CrossSurfaceObservationStatusV1::Skip));
        assert!(obs.primary_blocking_code.is_none());
    }

    #[test]
    fn final_primary_consumer_authority_digest_is_stable() {
        let context = FinalPrimarySemanticsAuthorityContextV1 {
            canonical_governance_entry_digest_prefix: "11".repeat(8),
            canonical_readiness_spine_digest_prefix: "22".repeat(8),
            canonical_bundle_spine_digest_prefix: "33".repeat(8),
            canonical_primary_semantics_authority_digest_prefix: "44".repeat(8),
        };
        let cats = vec!["PRIMARY_BLOCKING_PRECEDENCE_MISMATCH:1".to_string()];
        let a = derive_final_primary_consumer_authority(
            &context,
            17,
            FinalPrimarySemanticsConsumerAuthorityStatusV1::Fail,
            &cats,
        )
        .expect("authority");
        let b = derive_final_primary_consumer_authority(
            &context,
            17,
            FinalPrimarySemanticsConsumerAuthorityStatusV1::Fail,
            &cats,
        )
        .expect("authority");
        assert_eq!(a.authority_digest, b.authority_digest);
    }

    #[test]
    fn final_primary_surface_status_detects_legacy_input() {
        let observations = vec![PrimarySemanticsObservationV1 {
            canonical_condition_code: "AppliedScopeMismatch".to_string(),
            expected_primary_blocking_code: Some("AppliedScopeMismatch".to_string()),
            expected_primary_remediation_code: Some("REMEDIATION_RUN_INTEROP_MATRIX".to_string()),
            observed_surfaces: vec![PrimarySemanticsObservedSurfaceV1 {
                surface_kind: "OperatorWorkflow".to_string(),
                primary_blocking_code: Some("AppliedScopeMismatch".to_string()),
                primary_remediation_code: Some("REMEDIATION_RUN_INTEROP_MATRIX".to_string()),
                status: CrossSurfaceObservationStatusV1::Fail,
                source_digest_prefix: None,
                diagnostic_codes: vec!["LEGACY_PRIMARY_SEMANTICS_PRESENT".to_string()],
                secondary_diagnostic_codes: vec![],
                secondary_surface_reason_codes: vec![],
            }],
            observation_digest: "a".repeat(64),
        }];
        let (_statuses, top, mismatches, status) =
            evaluate_final_primary_surface_statuses(&observations);
        assert!(top
            .iter()
            .any(|v| v.starts_with("SURFACE_USED_LEGACY_PRIMARY_SEMANTICS_INPUT:")));
        assert!(mismatches > 0);
        assert!(matches!(
            status,
            FinalPrimarySemanticsConsumerAuthorityStatusV1::LegacyPresent
        ));
    }

    #[test]
    fn final_primary_requires_pass_authority() {
        let err =
            require_final_primary_semantics_authority(&CanonicalPrimarySemanticsAuthorityV1 {
                covered_surface_count: 1,
                covered_condition_count: 1,
                authority_status: CanonicalPrimarySemanticsAuthorityStatusV1::Fail,
                primary_semantics_digest: "aa".repeat(32),
                applied_supported_set_digest_prefix: "aa".repeat(8),
                canonical_governance_entry_digest_prefix: "bb".repeat(8),
                canonical_readiness_spine_digest_prefix: "cc".repeat(8),
                canonical_bundle_spine_digest_prefix: "dd".repeat(8),
            })
            .expect_err("must fail closed");
        assert!(err
            .to_string()
            .contains(FINAL_PRIMARY_SEMANTICS_AUTHORITY_REQUIRED));
    }

    #[test]
    fn residual_free_final_primary_inputs_digest_is_stable() {
        let primary = CanonicalPrimarySemanticsAuthorityV1 {
            covered_surface_count: 17,
            covered_condition_count: 10,
            authority_status: CanonicalPrimarySemanticsAuthorityStatusV1::Pass,
            primary_semantics_digest: "aa".repeat(32),
            applied_supported_set_digest_prefix: "bb".repeat(8),
            canonical_governance_entry_digest_prefix: "11".repeat(8),
            canonical_readiness_spine_digest_prefix: "22".repeat(8),
            canonical_bundle_spine_digest_prefix: "33".repeat(8),
        };
        let final_consumer = FinalPrimarySemanticsConsumerAuthorityV1 {
            canonical_governance_entry_digest_prefix: "11".repeat(8),
            canonical_readiness_spine_digest_prefix: "22".repeat(8),
            canonical_bundle_spine_digest_prefix: "33".repeat(8),
            canonical_primary_semantics_authority_digest_prefix: prefix16(
                &primary.primary_semantics_digest,
            ),
            covered_consumer_count: 17,
            authority_status: FinalPrimarySemanticsConsumerAuthorityStatusV1::Pass,
            authority_digest: "cc".repeat(32),
        };
        let residual = crate::FinalPrimarySemanticsResidualSweepV1 {
            canonical_governance_entry_digest_prefix: "11".repeat(8),
            canonical_readiness_spine_digest_prefix: "22".repeat(8),
            canonical_bundle_spine_digest_prefix: "33".repeat(8),
            canonical_primary_semantics_authority_digest_prefix: prefix16(
                &primary.primary_semantics_digest,
            ),
            final_primary_semantics_consumer_authority_digest_prefix: prefix16(
                &final_consumer.authority_digest,
            ),
            covered_surface_count: 17,
            residual_path_count: 0,
            sweep_status: crate::FinalPrimarySemanticsResidualSweepStatusV1::Pass,
            sweep_digest: "dd".repeat(32),
        };
        let a = require_residual_free_final_primary_semantics_inputs(
            None,
            None,
            Some(&primary),
            Some(&final_consumer),
            Some(&residual),
        )
        .expect("inputs");
        let b = require_residual_free_final_primary_semantics_inputs(
            None,
            None,
            Some(&primary),
            Some(&final_consumer),
            Some(&residual),
        )
        .expect("inputs");
        assert_eq!(a.authority_digest, b.authority_digest);
    }
}
