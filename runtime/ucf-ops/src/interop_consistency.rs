use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::remediation::{
    canonical_condition_for_interop_category, primary_remediation_for_condition_code,
};
use crate::{
    load_applied_supported_set_context_v1, prefix_hex, resolve_strict_evidence, sha256_hex,
    AggregatedActiveReviewSnapshotV1, AppliedSupportedSetContextV1, BackendEvidenceSnapshotV1,
    BugKitManifestV1, CanonicalExportLayoutCompatibilityV1, ConsolidatedOperatorReportV1,
    OperatorReviewPacketV1, OperatorSignoffDecisionV1, OpsError, ReproPackManifestV1,
    StrictEvidenceContextV1, StrictEvidenceSnapshotV1, V5GateReportV1,
};

const DIGEST_PREFIX_LEN: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum CrossSurfaceKindV1 {
    V5Gate,
    StrictEvidence,
    BackendEvidenceSnapshot,
    ActiveReviewSnapshot,
    OperatorReport,
    OperatorSignoff,
    OperatorReviewPacket,
    ReproPackManifest,
    BugKitManifest,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CrossSurfaceContextMatchStatusV1 {
    Match,
    Missing,
    Mismatch,
    Legacy,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CrossSurfaceEntryV1 {
    pub surface_kind: CrossSurfaceKindV1,
    pub surface_digest_prefix: Option<String>,
    pub supported_set_digest_prefix: Option<String>,
    pub policy_graph_digest_prefix: Option<String>,
    pub manifest_digest_prefix: Option<String>,
    pub primary_blocking_code: Option<String>,
    pub primary_remediation_code: Option<String>,
    pub artifact_refs_digest_prefix: Option<String>,
    pub context_match_status: CrossSurfaceContextMatchStatusV1,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CrossSurfaceContextMatrixV1 {
    pub schema_version: u16,
    pub applied_supported_set_digest_prefix: String,
    #[serde(default)]
    pub canonical_governance_entry_digest_prefix: String,
    #[serde(default)]
    pub final_governance_consumer_authority_digest_prefix: String,
    #[serde(default)]
    pub governance_residual_sweep_digest_prefix: String,
    pub policy_graph_digest_prefix: String,
    pub manifest_digest_prefix: String,
    pub surfaces: Vec<CrossSurfaceEntryV1>,
    pub matrix_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum InteropMismatchCategoryV1 {
    ScopeMismatch,
    PolicyMismatch,
    ManifestMismatch,
    SnapshotReferenceMismatch,
    RemediationMismatch,
    ExportRefMismatch,
    LegacySurfacePresent,
    RequiredSurfaceMissing,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum InteropOverallStatusV1 {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CrossSurfaceMatchRulesV1 {
    pub schema_version: u16,
    pub mismatch_categories: Vec<InteropMismatchCategoryV1>,
    pub canonical_condition_codes: Vec<String>,
    pub primary_remediation_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InteropConsistencySummaryV1 {
    pub overall_status: InteropOverallStatusV1,
    pub mismatch_counts: Vec<(InteropMismatchCategoryV1, usize)>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InteropConsistencyRecordV1 {
    pub schema_version: u16,
    pub matrix_digest_prefix: String,
    pub overall_status: InteropOverallStatusV1,
    pub mismatch_counts: Vec<(InteropMismatchCategoryV1, usize)>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InteropConsistencyMatrixReportV1 {
    pub schema_version: u16,
    pub matrix: CrossSurfaceContextMatrixV1,
    pub match_rules: CrossSurfaceMatchRulesV1,
    pub summary: InteropConsistencySummaryV1,
    pub interop_record: InteropConsistencyRecordV1,
}

pub fn interop_consistency_matrix(
    workdir: &Path,
    out: &Path,
) -> Result<InteropConsistencyMatrixReportV1, OpsError> {
    let out_root = PathBuf::from("./out");
    let applied_scope = load_applied_supported_set_context_v1(workdir)?;

    let backend = maybe_read_json::<BackendEvidenceSnapshotV1>(
        &out_root.join("backend_evidence_snapshot.json"),
    );
    let active = maybe_read_json::<AggregatedActiveReviewSnapshotV1>(
        &out_root.join("active_review_snapshot.json"),
    );
    let operator_report =
        maybe_read_json::<ConsolidatedOperatorReportV1>(&out_root.join("operator_report.json"));
    let operator_signoff =
        maybe_read_json::<OperatorSignoffDecisionV1>(&out_root.join("operator_signoff.json"));
    let operator_review =
        maybe_read_json::<OperatorReviewPacketV1>(&out_root.join("operator_review_packet.json"));
    let v5_gate = maybe_read_json::<V5GateReportV1>(&out_root.join("v5_gate_report.json"));
    let strict = resolve_strict_evidence(
        &out_root,
        &StrictEvidenceContextV1 {
            strict_required: true,
            expected_policy_graph_digest_prefix: backend
                .as_ref()
                .map(|s| s.policy_graph_digest_prefix.clone()),
            expected_manifest_digest_prefix: backend
                .as_ref()
                .map(|s| s.manifest_digest_prefix.clone()),
            expected_supported_slot_set_digest_prefix: backend
                .as_ref()
                .map(|s| s.supported_slot_set_digest.clone()),
            ..StrictEvidenceContextV1::default()
        },
    );
    let repro = read_repro_manifest(&out_root)?;
    let bugkit = read_bugkit_manifest(&out_root)?;

    let expected_policy = backend
        .as_ref()
        .map(|v| v.policy_graph_digest_prefix.clone())
        .or_else(|| {
            active
                .as_ref()
                .map(|v| v.policy_graph_digest_prefix.clone())
        })
        .or_else(|| {
            operator_signoff
                .as_ref()
                .map(|v| v.policy_graph_digest_prefix.clone())
        })
        .or_else(|| {
            operator_report
                .as_ref()
                .and_then(|v| v.policy_graph_digest_prefix.clone())
        })
        .unwrap_or_default();
    let expected_manifest = backend
        .as_ref()
        .map(|v| v.manifest_digest_prefix.clone())
        .or_else(|| active.as_ref().map(|v| v.manifest_digest_prefix.clone()))
        .or_else(|| {
            operator_signoff
                .as_ref()
                .map(|v| v.manifest_digest_prefix.clone())
        })
        .or_else(|| {
            operator_report
                .as_ref()
                .and_then(|v| v.manifest_digest_prefix.clone())
        })
        .unwrap_or_default();

    let mut surfaces = vec![
        v5_entry(v5_gate.as_ref()),
        strict_entry(&strict),
        backend_entry(backend.as_ref()),
        active_entry(active.as_ref()),
        operator_report_entry(operator_report.as_ref()),
        operator_signoff_entry(operator_signoff.as_ref()),
        operator_review_entry(operator_review.as_ref()),
        repro_entry(repro.as_ref()),
        bugkit_entry(bugkit.as_ref()),
    ];

    for s in &mut surfaces {
        match s.context_match_status {
            CrossSurfaceContextMatchStatusV1::Missing
            | CrossSurfaceContextMatchStatusV1::Legacy => {}
            _ => {
                if s.supported_set_digest_prefix
                    .as_ref()
                    .is_some_and(|d| d != &applied_scope.applied_set_digest_prefix)
                    || s.policy_graph_digest_prefix
                        .as_ref()
                        .is_some_and(|d| d != &expected_policy)
                    || s.manifest_digest_prefix
                        .as_ref()
                        .is_some_and(|d| d != &expected_manifest)
                {
                    s.context_match_status = CrossSurfaceContextMatchStatusV1::Mismatch;
                } else {
                    s.context_match_status = CrossSurfaceContextMatchStatusV1::Match;
                }
            }
        }
    }

    let matrix_digest = matrix_digest(
        &applied_scope,
        &expected_policy,
        &expected_manifest,
        &surfaces,
    )?;
    let matrix = CrossSurfaceContextMatrixV1 {
        schema_version: 1,
        applied_supported_set_digest_prefix: applied_scope.applied_set_digest_prefix.clone(),
        canonical_governance_entry_digest_prefix: operator_review
            .as_ref()
            .map(|review| review.canonical_governance_entry_digest_prefix.clone())
            .unwrap_or_else(|| "MISSING".to_string()),
        final_governance_consumer_authority_digest_prefix: operator_review
            .as_ref()
            .map(|review| {
                review
                    .final_governance_consumer_authority_digest_prefix
                    .clone()
            })
            .unwrap_or_else(|| "MISSING".to_string()),
        governance_residual_sweep_digest_prefix: operator_review
            .as_ref()
            .map(|review| review.governance_residual_sweep_digest_prefix.clone())
            .unwrap_or_else(|| "MISSING".to_string()),
        policy_graph_digest_prefix: expected_policy,
        manifest_digest_prefix: expected_manifest,
        surfaces,
        matrix_digest,
    };

    let rules = enrich_rules_with_canonical_mapping(evaluate_rules(
        &matrix,
        RuleInputs {
            strict: &strict,
            backend: backend.as_ref(),
            active: active.as_ref(),
            signoff: operator_signoff.as_ref(),
            review: operator_review.as_ref(),
            repro: repro.as_ref(),
            bugkit: bugkit.as_ref(),
        },
    ));

    let mut counts: BTreeMap<InteropMismatchCategoryV1, usize> = BTreeMap::new();
    for c in &rules.mismatch_categories {
        *counts.entry(c.clone()).or_insert(0) += 1;
    }
    let mismatch_counts = counts.into_iter().collect::<Vec<_>>();
    let overall_status = if rules.mismatch_categories.is_empty() {
        InteropOverallStatusV1::Pass
    } else {
        InteropOverallStatusV1::Fail
    };

    let summary = InteropConsistencySummaryV1 {
        overall_status: overall_status.clone(),
        mismatch_counts: mismatch_counts.clone(),
    };
    let interop_record = InteropConsistencyRecordV1 {
        schema_version: 1,
        matrix_digest_prefix: prefix_hex(&matrix.matrix_digest, DIGEST_PREFIX_LEN),
        overall_status: overall_status.clone(),
        mismatch_counts,
    };
    let report = InteropConsistencyMatrixReportV1 {
        schema_version: 1,
        matrix,
        match_rules: rules,
        summary,
        interop_record,
    };

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_string_pretty(&report)?)?;
    Ok(report)
}

struct RuleInputs<'a> {
    strict: &'a StrictEvidenceSnapshotV1,
    backend: Option<&'a BackendEvidenceSnapshotV1>,
    active: Option<&'a AggregatedActiveReviewSnapshotV1>,
    signoff: Option<&'a OperatorSignoffDecisionV1>,
    review: Option<&'a OperatorReviewPacketV1>,
    repro: Option<&'a ReproPackManifestV1>,
    bugkit: Option<&'a BugKitManifestV1>,
}

fn evaluate_rules(
    matrix: &CrossSurfaceContextMatrixV1,
    inputs: RuleInputs<'_>,
) -> CrossSurfaceMatchRulesV1 {
    let RuleInputs {
        strict,
        backend,
        active,
        signoff,
        review,
        repro,
        bugkit,
    } = inputs;
    let mut out = BTreeSet::new();

    for surface in &matrix.surfaces {
        match surface.context_match_status {
            CrossSurfaceContextMatchStatusV1::Missing => {
                out.insert(InteropMismatchCategoryV1::RequiredSurfaceMissing);
            }
            CrossSurfaceContextMatchStatusV1::Legacy => {
                out.insert(InteropMismatchCategoryV1::LegacySurfacePresent);
            }
            CrossSurfaceContextMatchStatusV1::Mismatch => {
                if surface.supported_set_digest_prefix.is_some() {
                    out.insert(InteropMismatchCategoryV1::ScopeMismatch);
                }
                if surface.policy_graph_digest_prefix.is_some() {
                    out.insert(InteropMismatchCategoryV1::PolicyMismatch);
                }
                if surface.manifest_digest_prefix.is_some() {
                    out.insert(InteropMismatchCategoryV1::ManifestMismatch);
                }
            }
            CrossSurfaceContextMatchStatusV1::Match => {}
        }
    }

    if let (Some(backend), Some(signoff)) = (backend, signoff) {
        let backend_prefix = prefix_hex(&backend.snapshot_digest, DIGEST_PREFIX_LEN);
        if signoff.evidence_snapshot_digest_prefix != backend_prefix {
            out.insert(InteropMismatchCategoryV1::SnapshotReferenceMismatch);
        }
    }
    if let (Some(active), Some(signoff)) = (active, signoff) {
        let active_prefix = prefix_hex(&active.snapshot_digest, DIGEST_PREFIX_LEN);
        if signoff.active_review_snapshot_digest_prefix.as_deref() != Some(active_prefix.as_str()) {
            out.insert(InteropMismatchCategoryV1::SnapshotReferenceMismatch);
        }
    }
    if let (Some(signoff), Some(review)) = (signoff, review) {
        let signoff_prefix = prefix_hex(&signoff.decision_digest, DIGEST_PREFIX_LEN);
        if review.artifacts.operator_signoff_digest_prefix != signoff_prefix {
            out.insert(InteropMismatchCategoryV1::SnapshotReferenceMismatch);
        }
    }

    if let Some(review) = review {
        if let Some(strict_code) = strict.remediation_codes.first() {
            if !review.remediation_codes.is_empty()
                && review.remediation_codes.first() != Some(strict_code)
            {
                out.insert(InteropMismatchCategoryV1::RemediationMismatch);
            }
        }
    }

    if let (Some(repro), Some(bugkit)) = (repro, bugkit) {
        let repro_refs = repro
            .related_artifacts
            .iter()
            .map(|r| (r.artifact_kind.clone(), r.ref_digest.clone()))
            .collect::<BTreeMap<_, _>>();
        let bugkit_refs = bugkit
            .related_artifacts
            .iter()
            .map(|r| (r.artifact_kind.clone(), r.ref_digest.clone()))
            .collect::<BTreeMap<_, _>>();
        if repro_refs != bugkit_refs {
            out.insert(InteropMismatchCategoryV1::ExportRefMismatch);
        }
    }

    CrossSurfaceMatchRulesV1 {
        schema_version: 1,
        mismatch_categories: out.into_iter().collect(),
        canonical_condition_codes: Vec::new(),
        primary_remediation_codes: Vec::new(),
    }
}

fn enrich_rules_with_canonical_mapping(
    mut rules: CrossSurfaceMatchRulesV1,
) -> CrossSurfaceMatchRulesV1 {
    let mut conditions = rules
        .mismatch_categories
        .iter()
        .filter_map(|category| canonical_condition_for_interop_category(&format!("{category:?}")))
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    conditions.sort();
    conditions.dedup();
    let mut remediations = conditions
        .iter()
        .filter_map(|condition| primary_remediation_for_condition_code(condition))
        .collect::<Vec<_>>();
    remediations.sort();
    remediations.dedup();
    rules.canonical_condition_codes = conditions;
    rules.primary_remediation_codes = remediations;
    rules
}

fn matrix_digest(
    applied_scope: &AppliedSupportedSetContextV1,
    expected_policy: &str,
    expected_manifest: &str,
    surfaces: &[CrossSurfaceEntryV1],
) -> Result<String, OpsError> {
    let payload =
        serde_json::to_vec(&(applied_scope, expected_policy, expected_manifest, surfaces))?;
    Ok(sha256_hex(&payload))
}

fn v5_entry(v: Option<&V5GateReportV1>) -> CrossSurfaceEntryV1 {
    CrossSurfaceEntryV1 {
        surface_kind: CrossSurfaceKindV1::V5Gate,
        surface_digest_prefix: v.and_then(|r| {
            serde_json::to_vec(r)
                .ok()
                .map(|b| prefix_hex(&sha256_hex(&b), DIGEST_PREFIX_LEN))
        }),
        supported_set_digest_prefix: v.and_then(|r| {
            find_evidence_prefix(r, "supported_slot_set_digest")
                .or_else(|| find_evidence_prefix(r, "supported_slot_set_digest_prefix"))
        }),
        policy_graph_digest_prefix: v.and_then(|r| find_evidence_prefix(r, "policy_graph_digest")),
        manifest_digest_prefix: v.and_then(|r| find_evidence_prefix(r, "manifest_digest")),
        primary_blocking_code: v.and_then(|r| {
            r.checks
                .iter()
                .find(|c| !matches!(c.status, crate::GateStatus::Pass))
                .map(|c| c.name.clone())
        }),
        primary_remediation_code: v.and_then(|r| {
            r.checks
                .iter()
                .find(|c| !matches!(c.status, crate::GateStatus::Pass))
                .map(|c| c.remediation_hint_code.clone())
        }),
        artifact_refs_digest_prefix: None,
        context_match_status: if v.is_some() {
            CrossSurfaceContextMatchStatusV1::Match
        } else {
            CrossSurfaceContextMatchStatusV1::Missing
        },
    }
}

fn strict_entry(v: &StrictEvidenceSnapshotV1) -> CrossSurfaceEntryV1 {
    CrossSurfaceEntryV1 {
        surface_kind: CrossSurfaceKindV1::StrictEvidence,
        surface_digest_prefix: Some(prefix_hex(&v.snapshot_digest, DIGEST_PREFIX_LEN)),
        supported_set_digest_prefix: v.supported_slot_set_digest_prefix.clone(),
        policy_graph_digest_prefix: v.policy_graph_digest_prefix.clone(),
        manifest_digest_prefix: v.manifest_digest_prefix.clone(),
        primary_blocking_code: v.primary_denial_code.clone(),
        primary_remediation_code: v.remediation_codes.first().cloned(),
        artifact_refs_digest_prefix: None,
        context_match_status: CrossSurfaceContextMatchStatusV1::Match,
    }
}

fn backend_entry(v: Option<&BackendEvidenceSnapshotV1>) -> CrossSurfaceEntryV1 {
    CrossSurfaceEntryV1 {
        surface_kind: CrossSurfaceKindV1::BackendEvidenceSnapshot,
        surface_digest_prefix: v.map(|r| prefix_hex(&r.snapshot_digest, DIGEST_PREFIX_LEN)),
        supported_set_digest_prefix: v
            .map(|r| prefix_hex(&r.supported_slot_set_digest, DIGEST_PREFIX_LEN)),
        policy_graph_digest_prefix: v.map(|r| r.policy_graph_digest_prefix.clone()),
        manifest_digest_prefix: v.map(|r| r.manifest_digest_prefix.clone()),
        primary_blocking_code: v.and_then(|r| {
            r.slots
                .iter()
                .find_map(|s| s.denials.active.as_ref().map(|d| format!("{d:?}")))
        }),
        primary_remediation_code: v.and_then(|r| {
            r.slots
                .iter()
                .flat_map(|s| {
                    s.canonical_remediation_codes
                        .iter()
                        .chain(s.remediation_codes.iter())
                })
                .next()
                .cloned()
        }),
        artifact_refs_digest_prefix: None,
        context_match_status: if v.is_some() {
            CrossSurfaceContextMatchStatusV1::Match
        } else {
            CrossSurfaceContextMatchStatusV1::Missing
        },
    }
}

fn active_entry(v: Option<&AggregatedActiveReviewSnapshotV1>) -> CrossSurfaceEntryV1 {
    CrossSurfaceEntryV1 {
        surface_kind: CrossSurfaceKindV1::ActiveReviewSnapshot,
        surface_digest_prefix: v.map(|r| prefix_hex(&r.snapshot_digest, DIGEST_PREFIX_LEN)),
        supported_set_digest_prefix: v
            .map(|r| prefix_hex(&r.supported_slot_set_digest, DIGEST_PREFIX_LEN)),
        policy_graph_digest_prefix: v.map(|r| r.policy_graph_digest_prefix.clone()),
        manifest_digest_prefix: v.map(|r| r.manifest_digest_prefix.clone()),
        primary_blocking_code: v
            .and_then(|r| r.slots.iter().find_map(|s| s.primary_denial_code.clone())),
        primary_remediation_code: v.and_then(|r| {
            r.slots
                .iter()
                .flat_map(|s| s.remediation_codes.iter())
                .next()
                .cloned()
        }),
        artifact_refs_digest_prefix: None,
        context_match_status: if v.is_some() {
            CrossSurfaceContextMatchStatusV1::Match
        } else {
            CrossSurfaceContextMatchStatusV1::Missing
        },
    }
}

fn operator_report_entry(v: Option<&ConsolidatedOperatorReportV1>) -> CrossSurfaceEntryV1 {
    CrossSurfaceEntryV1 {
        surface_kind: CrossSurfaceKindV1::OperatorReport,
        surface_digest_prefix: v.map(|r| prefix_hex(&r.report_digest, DIGEST_PREFIX_LEN)),
        supported_set_digest_prefix: None,
        policy_graph_digest_prefix: v.and_then(|r| r.policy_graph_digest_prefix.clone()),
        manifest_digest_prefix: v.and_then(|r| r.manifest_digest_prefix.clone()),
        primary_blocking_code: v
            .and_then(|r| r.sections.strict_section.primary_denial_code.clone()),
        primary_remediation_code: v.and_then(|r| {
            r.canonical_remediation_codes
                .first()
                .cloned()
                .or_else(|| r.remediation_codes.first().cloned())
        }),
        artifact_refs_digest_prefix: None,
        context_match_status: if v.is_some() {
            CrossSurfaceContextMatchStatusV1::Match
        } else {
            CrossSurfaceContextMatchStatusV1::Missing
        },
    }
}

fn operator_signoff_entry(v: Option<&OperatorSignoffDecisionV1>) -> CrossSurfaceEntryV1 {
    CrossSurfaceEntryV1 {
        surface_kind: CrossSurfaceKindV1::OperatorSignoff,
        surface_digest_prefix: v.map(|r| prefix_hex(&r.decision_digest, DIGEST_PREFIX_LEN)),
        supported_set_digest_prefix: v
            .map(|r| prefix_hex(&r.supported_slot_set_digest, DIGEST_PREFIX_LEN)),
        policy_graph_digest_prefix: v.map(|r| r.policy_graph_digest_prefix.clone()),
        manifest_digest_prefix: v.map(|r| r.manifest_digest_prefix.clone()),
        primary_blocking_code: v.and_then(|r| r.reasons.first().cloned()),
        primary_remediation_code: v.and_then(|r| {
            r.canonical_remediation_codes
                .first()
                .cloned()
                .or_else(|| r.remediation_codes.first().cloned())
        }),
        artifact_refs_digest_prefix: None,
        context_match_status: if v.is_some() {
            CrossSurfaceContextMatchStatusV1::Match
        } else {
            CrossSurfaceContextMatchStatusV1::Missing
        },
    }
}

fn operator_review_entry(v: Option<&OperatorReviewPacketV1>) -> CrossSurfaceEntryV1 {
    CrossSurfaceEntryV1 {
        surface_kind: CrossSurfaceKindV1::OperatorReviewPacket,
        surface_digest_prefix: v.map(|r| prefix_hex(&r.packet_digest, DIGEST_PREFIX_LEN)),
        supported_set_digest_prefix: v
            .map(|r| prefix_hex(&r.supported_slot_set_digest, DIGEST_PREFIX_LEN)),
        policy_graph_digest_prefix: v.map(|r| r.policy_graph_digest_prefix.clone()),
        manifest_digest_prefix: v.map(|r| r.manifest_digest_prefix.clone()),
        primary_blocking_code: v.and_then(|r| r.blocking_codes.first().cloned()),
        primary_remediation_code: v.and_then(|r| r.remediation_codes.first().cloned()),
        artifact_refs_digest_prefix: v.and_then(|r| {
            serde_json::to_vec(&r.artifacts)
                .ok()
                .map(|b| prefix_hex(&sha256_hex(&b), DIGEST_PREFIX_LEN))
        }),
        context_match_status: if v.is_some() {
            CrossSurfaceContextMatchStatusV1::Match
        } else {
            CrossSurfaceContextMatchStatusV1::Missing
        },
    }
}

fn repro_entry(v: Option<&ReproPackManifestV1>) -> CrossSurfaceEntryV1 {
    let status = match v.map(|r| &r.export_layout_compatibility) {
        None => CrossSurfaceContextMatchStatusV1::Missing,
        Some(CanonicalExportLayoutCompatibilityV1::Canonical) => {
            CrossSurfaceContextMatchStatusV1::Match
        }
        Some(_) => CrossSurfaceContextMatchStatusV1::Legacy,
    };
    CrossSurfaceEntryV1 {
        surface_kind: CrossSurfaceKindV1::ReproPackManifest,
        surface_digest_prefix: v.map(|r| prefix_hex(&r.repro_pack_digest, DIGEST_PREFIX_LEN)),
        supported_set_digest_prefix: v
            .map(|r| r.export_context.supported_slot_set_digest_prefix.clone()),
        policy_graph_digest_prefix: v.map(|r| r.export_context.policy_graph_digest_prefix.clone()),
        manifest_digest_prefix: v.map(|r| r.export_context.manifest_digest_prefix.clone()),
        primary_blocking_code: v.and_then(|r| match r.export_layout_compatibility {
            CanonicalExportLayoutCompatibilityV1::LegacyExportTranslated => {
                Some("LEGACY_SURFACE_TRANSLATED".to_string())
            }
            CanonicalExportLayoutCompatibilityV1::LegacyExportUnsupported => {
                Some("LEGACY_SURFACE_UNSUPPORTED".to_string())
            }
            CanonicalExportLayoutCompatibilityV1::LegacyExportLayout => {
                Some("LEGACY_SURFACE_PRESENT".to_string())
            }
            CanonicalExportLayoutCompatibilityV1::Canonical => None,
        }),
        primary_remediation_code: None,
        artifact_refs_digest_prefix: v.and_then(artifact_refs_digest),
        context_match_status: status,
    }
}

fn bugkit_entry(v: Option<&BugKitManifestV1>) -> CrossSurfaceEntryV1 {
    let status = match v.map(|r| &r.export_layout_compatibility) {
        None => CrossSurfaceContextMatchStatusV1::Missing,
        Some(CanonicalExportLayoutCompatibilityV1::Canonical) => {
            CrossSurfaceContextMatchStatusV1::Match
        }
        Some(_) => CrossSurfaceContextMatchStatusV1::Legacy,
    };
    CrossSurfaceEntryV1 {
        surface_kind: CrossSurfaceKindV1::BugKitManifest,
        surface_digest_prefix: v.map(|r| prefix_hex(&r.bugkit_digest, DIGEST_PREFIX_LEN)),
        supported_set_digest_prefix: v
            .map(|r| r.export_context.supported_slot_set_digest_prefix.clone()),
        policy_graph_digest_prefix: v.map(|r| r.export_context.policy_graph_digest_prefix.clone()),
        manifest_digest_prefix: v.map(|r| r.export_context.manifest_digest_prefix.clone()),
        primary_blocking_code: v.and_then(|r| match r.export_layout_compatibility {
            CanonicalExportLayoutCompatibilityV1::LegacyExportTranslated => {
                Some("LEGACY_SURFACE_TRANSLATED".to_string())
            }
            CanonicalExportLayoutCompatibilityV1::LegacyExportUnsupported => {
                Some("LEGACY_SURFACE_UNSUPPORTED".to_string())
            }
            CanonicalExportLayoutCompatibilityV1::LegacyExportLayout => {
                Some("LEGACY_SURFACE_PRESENT".to_string())
            }
            CanonicalExportLayoutCompatibilityV1::Canonical => None,
        }),
        primary_remediation_code: None,
        artifact_refs_digest_prefix: v.and_then(artifact_refs_digest),
        context_match_status: status,
    }
}

fn artifact_refs_digest<T>(m: &T) -> Option<String>
where
    T: ExportManifestLike,
{
    let mut refs = m
        .related_artifacts()
        .iter()
        .map(|r| r.ref_digest.clone())
        .collect::<Vec<_>>();
    refs.sort();
    serde_json::to_vec(&refs)
        .ok()
        .map(|b| prefix_hex(&sha256_hex(&b), DIGEST_PREFIX_LEN))
}

trait ExportManifestLike {
    fn related_artifacts(&self) -> &[crate::CanonicalExportArtifactRefV1];
}

impl ExportManifestLike for ReproPackManifestV1 {
    fn related_artifacts(&self) -> &[crate::CanonicalExportArtifactRefV1] {
        &self.related_artifacts
    }
}

impl ExportManifestLike for BugKitManifestV1 {
    fn related_artifacts(&self) -> &[crate::CanonicalExportArtifactRefV1] {
        &self.related_artifacts
    }
}

fn find_evidence_prefix(report: &V5GateReportV1, key: &str) -> Option<String> {
    report
        .checks
        .iter()
        .find_map(|c| c.evidence_digest_prefixes.get(key).cloned())
}

fn maybe_read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Option<T> {
    let body = fs::read(path).ok()?;
    serde_json::from_slice(&body).ok()
}

fn read_repro_manifest(out_root: &Path) -> Result<Option<ReproPackManifestV1>, OpsError> {
    if let Some(v) =
        maybe_read_json::<ReproPackManifestV1>(&out_root.join("repro_pack_manifest.json"))
    {
        return Ok(Some(v));
    }
    let mut zips = fs::read_dir(out_root)
        .ok()
        .into_iter()
        .flat_map(|entries| entries.filter_map(|e| e.ok().map(|x| x.path())))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("zip"))
        .filter(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|n| n.contains("repro"))
        })
        .collect::<Vec<_>>();
    zips.sort();
    zips.reverse();
    for zip in zips {
        if let Ok(v) = read_zip_json::<ReproPackManifestV1>(&zip, "repro_pack_manifest.json") {
            return Ok(Some(v));
        }
    }
    Ok(None)
}

fn read_bugkit_manifest(out_root: &Path) -> Result<Option<BugKitManifestV1>, OpsError> {
    if let Some(v) = maybe_read_json::<BugKitManifestV1>(&out_root.join("BUGKIT_MANIFEST.json")) {
        return Ok(Some(v));
    }
    let mut zips = fs::read_dir(out_root)
        .ok()
        .into_iter()
        .flat_map(|entries| entries.filter_map(|e| e.ok().map(|x| x.path())))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("zip"))
        .filter(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|n| n.contains("bug"))
        })
        .collect::<Vec<_>>();
    zips.sort();
    zips.reverse();
    for zip in zips {
        if let Ok(v) = read_zip_json::<BugKitManifestV1>(&zip, "BUGKIT_MANIFEST.json") {
            return Ok(Some(v));
        }
    }
    Ok(None)
}

fn read_zip_json<T: for<'de> Deserialize<'de>>(
    zip_path: &Path,
    manifest_name: &str,
) -> Result<T, OpsError> {
    let f = fs::File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(f).map_err(|e| {
        OpsError::Invalid(format!("unable to open zip {}: {e}", zip_path.display()))
    })?;
    let mut file = archive.by_name(manifest_name).map_err(|e| {
        OpsError::Invalid(format!(
            "missing {manifest_name} in {}: {e}",
            zip_path.display()
        ))
    })?;
    let mut body = String::new();
    file.read_to_string(&mut body)?;
    Ok(serde_json::from_str(&body)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mismatch_categories_are_deterministic_and_sorted() {
        let matrix = CrossSurfaceContextMatrixV1 {
            schema_version: 1,
            applied_supported_set_digest_prefix: "a".to_string(),
            canonical_governance_entry_digest_prefix: "e".to_string(),
            final_governance_consumer_authority_digest_prefix: "g".to_string(),
            governance_residual_sweep_digest_prefix: "r".to_string(),
            policy_graph_digest_prefix: "p".to_string(),
            manifest_digest_prefix: "m".to_string(),
            surfaces: vec![
                CrossSurfaceEntryV1 {
                    surface_kind: CrossSurfaceKindV1::BackendEvidenceSnapshot,
                    surface_digest_prefix: None,
                    supported_set_digest_prefix: None,
                    policy_graph_digest_prefix: None,
                    manifest_digest_prefix: None,
                    primary_blocking_code: None,
                    primary_remediation_code: None,
                    artifact_refs_digest_prefix: None,
                    context_match_status: CrossSurfaceContextMatchStatusV1::Missing,
                },
                CrossSurfaceEntryV1 {
                    surface_kind: CrossSurfaceKindV1::ReproPackManifest,
                    surface_digest_prefix: None,
                    supported_set_digest_prefix: Some("x".to_string()),
                    policy_graph_digest_prefix: Some("y".to_string()),
                    manifest_digest_prefix: Some("z".to_string()),
                    primary_blocking_code: None,
                    primary_remediation_code: None,
                    artifact_refs_digest_prefix: None,
                    context_match_status: CrossSurfaceContextMatchStatusV1::Mismatch,
                },
            ],
            matrix_digest: "d".to_string(),
        };
        let strict = StrictEvidenceSnapshotV1 {
            schema_version: 1,
            strict_mode_enabled: true,
            strict_status: crate::StrictEvidenceStatusV1::Fail,
            strict_report_digest_prefix: None,
            policy_graph_digest_prefix: None,
            manifest_digest_prefix: None,
            supported_slot_set_digest_prefix: None,
            primary_denial_code: None,
            remediation_codes: Vec::new(),
            failing_check_ids: Vec::new(),
            snapshot_digest: "s".to_string(),
        };
        let rules = evaluate_rules(
            &matrix,
            RuleInputs {
                strict: &strict,
                backend: None,
                active: None,
                signoff: None,
                review: None,
                repro: None,
                bugkit: None,
            },
        );
        assert_eq!(
            rules.mismatch_categories,
            vec![
                InteropMismatchCategoryV1::ScopeMismatch,
                InteropMismatchCategoryV1::PolicyMismatch,
                InteropMismatchCategoryV1::ManifestMismatch,
                InteropMismatchCategoryV1::RequiredSurfaceMissing,
            ]
        );
    }

    #[test]
    fn matrix_digest_is_stable() {
        let applied = AppliedSupportedSetContextV1 {
            schema_version: 1,
            applied_set_digest_prefix: "abc".to_string(),
            slots: vec!["sae".to_string()],
            decision: crate::SupportedRealSlotSetExecutionDecisionV2::Frozen,
            previous_set_digest_prefix: "0".to_string(),
            policy_digest_prefix: "1".to_string(),
            context_digest: "2".to_string(),
            compatibility_code: None,
        };
        let surfaces = vec![CrossSurfaceEntryV1 {
            surface_kind: CrossSurfaceKindV1::StrictEvidence,
            surface_digest_prefix: Some("x".to_string()),
            supported_set_digest_prefix: Some("abc".to_string()),
            policy_graph_digest_prefix: Some("pg".to_string()),
            manifest_digest_prefix: Some("mg".to_string()),
            primary_blocking_code: None,
            primary_remediation_code: None,
            artifact_refs_digest_prefix: None,
            context_match_status: CrossSurfaceContextMatchStatusV1::Match,
        }];
        let d1 = matrix_digest(&applied, "pg", "mg", &surfaces).expect("digest");
        let d2 = matrix_digest(&applied, "pg", "mg", &surfaces).expect("digest");
        assert_eq!(d1, d2);
    }
}
