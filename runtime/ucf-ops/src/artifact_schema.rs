#![allow(clippy::result_large_err)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::OpsError;

const SNAPSHOT_INDEX_FILE: &str = "index.json";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DriftKind {
    Additive,
    Breaking,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct ArtifactSchemaArgs {
    pub repo_root: PathBuf,
    pub out_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactSchemaSnapshot {
    pub artifact_id: String,
    pub type_name: String,
    pub source_file: String,
    pub required_fields: Vec<String>,
    pub optional_fields: Vec<String>,
    pub field_types: BTreeMap<String, String>,
    pub enum_variants: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactSchemaSnapshotIndex {
    pub schema_version: u16,
    pub artifacts: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactSchemaDiffEntry {
    pub artifact: String,
    pub drift_kind: DriftKind,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactSchemaCheckReport {
    pub ok: bool,
    pub covered_artifacts: Vec<String>,
    pub diffs: Vec<ArtifactSchemaDiffEntry>,
    pub remediation: String,
}

#[derive(Debug, Clone, Copy)]
struct ArtifactSpec {
    artifact_id: &'static str,
    file_rel: &'static str,
    type_name: &'static str,
    enum_names: &'static [&'static str],
}

const ARTIFACT_SPECS: [ArtifactSpec; 44] = [
    ArtifactSpec {
        artifact_id: "active_review_snapshot_v1",
        file_rel: "runtime/ucf-ops/src/models_lifecycle.rs",
        type_name: "AggregatedActiveReviewSnapshotV1",
        enum_names: &["ActiveReviewOverallStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "backend_resolution_v1",
        file_rel: "runtime/ucf-ops/src/models_lifecycle.rs",
        type_name: "BurnSupportResolutionV1",
        enum_names: &["BurnResolutionStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "backend_evidence_snapshot_v1",
        file_rel: "runtime/ucf-ops/src/models_lifecycle.rs",
        type_name: "BackendEvidenceSnapshotV1",
        enum_names: &[],
    },
    ArtifactSpec {
        artifact_id: "governance_primary_surfaces_v1",
        file_rel: "runtime/ucf-ops/src/governance_surfaces.rs",
        type_name: "GovernancePrimarySurfacesV1",
        enum_names: &[],
    },
    ArtifactSpec {
        artifact_id: "supported_real_slot_set_v2",
        file_rel: "runtime/ucf-ops/src/models_lifecycle.rs",
        type_name: "SupportedRealSlotSetV2",
        enum_names: &["SupportedRealSlotSetExecutionDecisionV2"],
    },
    ArtifactSpec {
        artifact_id: "supported_scope_execution_v3",
        file_rel: "runtime/ucf-ops/src/models_lifecycle.rs",
        type_name: "SupportedScopeExecutionV3",
        enum_names: &["SupportedScopeExecutionDecisionV3"],
    },
    ArtifactSpec {
        artifact_id: "supported_scope_execution_v4",
        file_rel: "runtime/ucf-ops/src/models_lifecycle.rs",
        type_name: "SupportedScopeExecutionV4",
        enum_names: &["SupportedScopeExecutionDecisionV4"],
    },
    ArtifactSpec {
        artifact_id: "applied_scope_authority_v1",
        file_rel: "runtime/ucf-ops/src/scope_authority.rs",
        type_name: "ScopeAuthorityCheckReportV1",
        enum_names: &[
            "ScopeAuthorityMismatchCategoryV1",
            "ScopeAuthorityOverallStatusV1",
        ],
    },
    ArtifactSpec {
        artifact_id: "applied_supported_set_context_v1",
        file_rel: "runtime/ucf-ops/src/models_lifecycle.rs",
        type_name: "AppliedSupportedSetContextV1",
        enum_names: &["SupportedRealSlotSetExecutionDecisionV2"],
    },
    ArtifactSpec {
        artifact_id: "repro_pack_manifest_v1",
        file_rel: "runtime/ucf-ops/src/lib.rs",
        type_name: "ReproPackManifestV1",
        enum_names: &[],
    },
    ArtifactSpec {
        artifact_id: "bundle_roundtrip_consistency_v1",
        file_rel: "runtime/ucf-ops/src/lib.rs",
        type_name: "BundleRoundTripConsistencyV1",
        enum_names: &[
            "BundleRoundTripMatchStatusV1",
            "BundleRoundTripOverallStatusV1",
            "CanonicalBundleKindV1",
        ],
    },
    ArtifactSpec {
        artifact_id: "canonical_bundle_spine_v1",
        file_rel: "runtime/ucf-ops/src/lib.rs",
        type_name: "CanonicalBundleSpineV1",
        enum_names: &["BundleSpineStatusV1", "CanonicalBundleKindV1"],
    },
    ArtifactSpec {
        artifact_id: "canonical_bundle_authority_v2",
        file_rel: "runtime/ucf-ops/src/lib.rs",
        type_name: "CanonicalBundleAuthorityV2",
        enum_names: &["CanonicalBundleAuthorityStatusV2"],
    },
    ArtifactSpec {
        artifact_id: "bugkit_manifest_v1",
        file_rel: "runtime/ucf-ops/src/lib.rs",
        type_name: "BugKitManifestV1",
        enum_names: &[],
    },
    ArtifactSpec {
        artifact_id: "canonical_bundle_consumption_context_v1",
        file_rel: "runtime/ucf-ops/src/lib.rs",
        type_name: "CanonicalBundleConsumptionContextV1",
        enum_names: &["CanonicalBundleKindV1"],
    },
    ArtifactSpec {
        artifact_id: "canonical_export_artifact_ref_v1",
        file_rel: "runtime/ucf-ops/src/lib.rs",
        type_name: "CanonicalExportArtifactRefV1",
        enum_names: &["CanonicalArtifactIncludedStateV1"],
    },
    ArtifactSpec {
        artifact_id: "canonical_export_context_v1",
        file_rel: "runtime/ucf-ops/src/lib.rs",
        type_name: "CanonicalExportContextV1",
        enum_names: &[],
    },
    ArtifactSpec {
        artifact_id: "canonical_governance_entry_v1",
        file_rel: "runtime/ucf-ops/src/canonical_governance_entry.rs",
        type_name: "CanonicalGovernanceEntryV1",
        enum_names: &["CanonicalGovernanceEntryStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "canonical_governance_entry_authority_v2",
        file_rel: "runtime/ucf-ops/src/governance_entry_sweep.rs",
        type_name: "CanonicalGovernanceEntryAuthorityV2",
        enum_names: &["GovernanceEntryAuthorityStatusV2"],
    },
    ArtifactSpec {
        artifact_id: "canonical_readiness_spine_v1",
        file_rel: "runtime/ucf-ops/src/readiness_spine.rs",
        type_name: "CanonicalReadinessSpineV1",
        enum_names: &["CanonicalReadinessSpineStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "canonical_readiness_authority_v2",
        file_rel: "runtime/ucf-ops/src/readiness_spine.rs",
        type_name: "CanonicalReadinessAuthorityV2",
        enum_names: &["CanonicalReadinessAuthorityStatusV2"],
    },
    ArtifactSpec {
        artifact_id: "remediation_consistency_check_v1",
        file_rel: "runtime/ucf-ops/src/remediation_consistency.rs",
        type_name: "RemediationConsistencyReportV1",
        enum_names: &[],
    },
    ArtifactSpec {
        artifact_id: "cross_surface_context_matrix_v1",
        file_rel: "runtime/ucf-ops/src/interop_consistency.rs",
        type_name: "CrossSurfaceContextMatrixV1",
        enum_names: &[],
    },
    ArtifactSpec {
        artifact_id: "cross_surface_condition_observation_v1",
        file_rel: "runtime/ucf-ops/src/remediation_consistency.rs",
        type_name: "CrossSurfaceConditionObservationV1",
        enum_names: &["CrossSurfaceObservationStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "interop_consistency_matrix_report_v1",
        file_rel: "runtime/ucf-ops/src/interop_consistency.rs",
        type_name: "InteropConsistencyMatrixReportV1",
        enum_names: &[],
    },
    ArtifactSpec {
        artifact_id: "operator_report_v1",
        file_rel: "runtime/ucf-ops/src/operator_report.rs",
        type_name: "ConsolidatedOperatorReportV1",
        enum_names: &["OperatorStatus"],
    },
    ArtifactSpec {
        artifact_id: "operator_signoff_v1",
        file_rel: "runtime/ucf-ops/src/operator_signoff.rs",
        type_name: "OperatorSignoffDecisionV1",
        enum_names: &["SignoffDecisionStateV1"],
    },
    ArtifactSpec {
        artifact_id: "operator_review_packet_v1",
        file_rel: "runtime/ucf-ops/src/operator_review_packet.rs",
        type_name: "OperatorReviewPacketV1",
        enum_names: &["OperatorReviewStageV1"],
    },
    ArtifactSpec {
        artifact_id: "operator_workflow_chain_v1",
        file_rel: "runtime/ucf-ops/src/operator_workflow.rs",
        type_name: "OperatorWorkflowChainV1",
        enum_names: &["OperatorWorkflowStageV2"],
    },
    ArtifactSpec {
        artifact_id: "canonical_roundtrip_chain_v1",
        file_rel: "runtime/ucf-ops/src/roundtrip_chain.rs",
        type_name: "CanonicalRoundTripChainV1",
        enum_names: &["CanonicalRoundTripChainStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "canonical_continuity_authority_v1",
        file_rel: "runtime/ucf-ops/src/continuity_authority.rs",
        type_name: "CanonicalContinuityAuthorityV1",
        enum_names: &["ContinuityAuthorityStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "strict_failure_report_v3",
        file_rel: "runtime/ucf-ops/src/lib.rs",
        type_name: "StrictModeFailureReport",
        enum_names: &["StrictCheckStatus", "StrictCheckV3Status"],
    },
    ArtifactSpec {
        artifact_id: "v3_gate_report_v1",
        file_rel: "runtime/ucf-ops/src/lib.rs",
        type_name: "V3GateReportV1",
        enum_names: &["V3GateOverallStatus", "GateStatus"],
    },
    ArtifactSpec {
        artifact_id: "v4_gate_report_v1",
        file_rel: "runtime/ucf-ops/src/lib.rs",
        type_name: "V4GateReportV1",
        enum_names: &["V4GateOverallStatus", "GateStatus"],
    },
    ArtifactSpec {
        artifact_id: "v5_gate_report_v1",
        file_rel: "runtime/ucf-ops/src/lib.rs",
        type_name: "V5GateReportV1",
        enum_names: &["V5GateOverallStatus", "GateStatus"],
    },
    ArtifactSpec {
        artifact_id: "readiness_gate_report_v1",
        file_rel: "runtime/ucf-ops/src/lib.rs",
        type_name: "ReadinessGateReport",
        enum_names: &["GateStatus"],
    },
    ArtifactSpec {
        artifact_id: "reviewability_reduction_v1",
        file_rel: "runtime/ucf-ops/src/reviewability_truth.rs",
        type_name: "ReviewabilityReductionV1",
        enum_names: &["ReviewabilityAggregateReadinessV1"],
    },
    ArtifactSpec {
        artifact_id: "slot_reviewability_truth_v1",
        file_rel: "runtime/ucf-ops/src/reviewability_truth.rs",
        type_name: "SlotReviewabilityTruthV1",
        enum_names: &[],
    },
    ArtifactSpec {
        artifact_id: "spine_condition_observation_v1",
        file_rel: "runtime/ucf-ops/src/remediation_consistency.rs",
        type_name: "SpineConditionObservationV1",
        enum_names: &["CrossSurfaceObservationStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "primary_semantics_observation_v1",
        file_rel: "runtime/ucf-ops/src/remediation_consistency.rs",
        type_name: "PrimarySemanticsObservationV1",
        enum_names: &["CrossSurfaceObservationStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "canonical_primary_semantics_authority_v1",
        file_rel: "runtime/ucf-ops/src/remediation_consistency.rs",
        type_name: "CanonicalPrimarySemanticsAuthorityV1",
        enum_names: &["CanonicalPrimarySemanticsAuthorityStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "supported_scope_reevaluation_v1",
        file_rel: "runtime/ucf-ops/src/models_lifecycle.rs",
        type_name: "SupportedScopeReevaluationV1",
        enum_names: &["SupportedScopeReevaluationDecisionV1"],
    },
    ArtifactSpec {
        artifact_id: "v7_gate_report_v1",
        file_rel: "runtime/ucf-ops/src/v7_gate.rs",
        type_name: "V7GateReportV1",
        enum_names: &["V7GateOverallStatus"],
    },
    ArtifactSpec {
        artifact_id: "v8_gate_report_v1",
        file_rel: "runtime/ucf-ops/src/v8_gate.rs",
        type_name: "V8GateReportV1",
        enum_names: &["V8GateOverallStatus"],
    },
];

fn sorted_artifact_specs() -> Vec<ArtifactSpec> {
    let mut specs = ARTIFACT_SPECS.to_vec();
    specs.sort_by(|a, b| a.artifact_id.cmp(b.artifact_id));
    specs
}

pub fn generate_artifact_schema_snapshots(
    args: &ArtifactSchemaArgs,
) -> Result<Vec<String>, OpsError> {
    fs::create_dir_all(&args.out_dir)?;
    let mut covered = Vec::new();
    for spec in sorted_artifact_specs() {
        let snapshot = build_snapshot(&args.repo_root, spec)?;
        let out = args.out_dir.join(format!("{}.json", spec.artifact_id));
        fs::write(&out, serde_json::to_string_pretty(&snapshot)?)?;
        covered.push(spec.artifact_id.to_string());
    }
    let index = ArtifactSchemaSnapshotIndex {
        schema_version: 1,
        artifacts: covered.clone(),
    };
    fs::write(
        args.out_dir.join(SNAPSHOT_INDEX_FILE),
        serde_json::to_string_pretty(&index)?,
    )?;
    Ok(covered)
}

pub fn check_artifact_schema_snapshots(
    args: &ArtifactSchemaArgs,
) -> Result<ArtifactSchemaCheckReport, OpsError> {
    let tmp = tempfile::tempdir()?;
    let generated_dir = tmp.path().join("generated");
    let covered = generate_artifact_schema_snapshots(&ArtifactSchemaArgs {
        repo_root: args.repo_root.clone(),
        out_dir: generated_dir.clone(),
    })?;

    let mut diffs = Vec::new();
    for artifact in &covered {
        let file = format!("{artifact}.json");
        let committed = args.out_dir.join(&file);
        let generated = generated_dir.join(&file);
        if !committed.exists() {
            diffs.push(ArtifactSchemaDiffEntry {
                artifact: artifact.clone(),
                drift_kind: DriftKind::Breaking,
                summary: format!("missing committed snapshot: {}", committed.display()),
            });
            continue;
        }

        let old = match serde_json::from_str::<ArtifactSchemaSnapshot>(&fs::read_to_string(
            &committed,
        )?) {
            Ok(snapshot) => snapshot,
            Err(err) => {
                diffs.push(ArtifactSchemaDiffEntry {
                    artifact: artifact.clone(),
                    drift_kind: DriftKind::Unknown,
                    summary: format!(
                        "committed snapshot parse error in {}: {err}",
                        committed.display()
                    ),
                });
                continue;
            }
        };
        let new = match serde_json::from_str::<ArtifactSchemaSnapshot>(&fs::read_to_string(
            &generated,
        )?) {
            Ok(snapshot) => snapshot,
            Err(err) => {
                diffs.push(ArtifactSchemaDiffEntry {
                    artifact: artifact.clone(),
                    drift_kind: DriftKind::Unknown,
                    summary: format!(
                        "generated snapshot parse error in {}: {err}",
                        generated.display()
                    ),
                });
                continue;
            }
        };
        if old == new {
            continue;
        }
        let (kind, summary) = classify_drift(&old, &new);
        diffs.push(ArtifactSchemaDiffEntry {
            artifact: artifact.clone(),
            drift_kind: kind,
            summary,
        });
    }

    let mut unknown_files = Vec::new();
    for entry in fs::read_dir(&args.out_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|v| v.to_str()) != Some("json") {
            continue;
        }
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        if stem == "index" {
            continue;
        }
        if !covered.iter().any(|x| x == stem) {
            unknown_files.push(stem.to_string());
        }
    }
    unknown_files.sort();
    if !unknown_files.is_empty() {
        diffs.push(ArtifactSchemaDiffEntry {
            artifact: "__extra__".to_string(),
            drift_kind: DriftKind::Unknown,
            summary: format!("unexpected snapshot files: {}", unknown_files.join(",")),
        });
    }

    diffs.sort_by(|a, b| {
        a.artifact
            .cmp(&b.artifact)
            .then_with(|| format!("{:?}", a.drift_kind).cmp(&format!("{:?}", b.drift_kind)))
            .then_with(|| a.summary.cmp(&b.summary))
    });

    Ok(ArtifactSchemaCheckReport {
        ok: diffs.is_empty(),
        covered_artifacts: covered,
        diffs,
        remediation: "run: cargo run -p ucf-ops -- spec artifact-schemas --out docs/artifact_schema_snapshots && review git diff && git add docs/artifact_schema_snapshots".to_string(),
    })
}

pub fn classify_drift(
    old: &ArtifactSchemaSnapshot,
    new: &ArtifactSchemaSnapshot,
) -> (DriftKind, String) {
    let old_required: BTreeSet<_> = old.required_fields.iter().cloned().collect();
    let new_required: BTreeSet<_> = new.required_fields.iter().cloned().collect();
    let old_optional: BTreeSet<_> = old.optional_fields.iter().cloned().collect();
    let new_optional: BTreeSet<_> = new.optional_fields.iter().cloned().collect();

    for field in old_required.union(&old_optional) {
        if !new.field_types.contains_key(field) {
            return (DriftKind::Breaking, format!("field removed: {field}"));
        }
    }

    for field in &old_required {
        if !new_required.contains(field) {
            return (
                DriftKind::Breaking,
                format!("required field became optional/removed: {field}"),
            );
        }
    }

    for (field, old_ty) in &old.field_types {
        if let Some(new_ty) = new.field_types.get(field) {
            if new_ty != old_ty {
                return (
                    DriftKind::Breaking,
                    format!("field type changed for {field}: {old_ty} -> {new_ty}"),
                );
            }
        }
    }

    for (name, variants_old) in &old.enum_variants {
        let Some(variants_new) = new.enum_variants.get(name) else {
            return (
                DriftKind::Unknown,
                format!("enum snapshot missing in new shape: {name}"),
            );
        };
        let old_set: BTreeSet<_> = variants_old.iter().cloned().collect();
        let new_set: BTreeSet<_> = variants_new.iter().cloned().collect();
        if !old_set.is_subset(&new_set) {
            return (
                DriftKind::Breaking,
                format!("enum variants removed for {name}"),
            );
        }
    }

    let mut additive_notes = Vec::new();
    for field in new_required.difference(&old_required) {
        if !old_optional.contains(field) {
            return (
                DriftKind::Breaking,
                format!("new required field added: {field}"),
            );
        }
    }
    for field in new_optional.difference(&old_optional) {
        if !old_required.contains(field) {
            additive_notes.push(format!("optional field added: {field}"));
        }
    }

    for (name, variants_new) in &new.enum_variants {
        let old_set: BTreeSet<_> = old
            .enum_variants
            .get(name)
            .map(|v| v.iter().cloned().collect())
            .unwrap_or_default();
        for variant in variants_new {
            if !old_set.contains(variant) {
                additive_notes.push(format!("enum variant added: {name}.{variant}"));
            }
        }
    }

    if additive_notes.is_empty() {
        (
            DriftKind::Unknown,
            "shape changed but no bounded classification matched".to_string(),
        )
    } else {
        (DriftKind::Additive, additive_notes.join("; "))
    }
}

fn build_snapshot(
    repo_root: &Path,
    spec: ArtifactSpec,
) -> Result<ArtifactSchemaSnapshot, OpsError> {
    let source_path = repo_root.join(spec.file_rel);
    let source = fs::read_to_string(&source_path)?;
    let structure = parse_struct_shape(&source, spec.type_name)?;

    let mut enum_variants = BTreeMap::new();
    for enum_name in spec.enum_names {
        enum_variants.insert(
            (*enum_name).to_string(),
            parse_enum_variants(&source, enum_name)?,
        );
    }

    Ok(ArtifactSchemaSnapshot {
        artifact_id: spec.artifact_id.to_string(),
        type_name: spec.type_name.to_string(),
        source_file: spec.file_rel.to_string(),
        required_fields: structure.required_fields,
        optional_fields: structure.optional_fields,
        field_types: structure.field_types,
        enum_variants,
    })
}

struct StructShape {
    required_fields: Vec<String>,
    optional_fields: Vec<String>,
    field_types: BTreeMap<String, String>,
}

fn parse_struct_shape(source: &str, type_name: &str) -> Result<StructShape, OpsError> {
    let marker = format!("pub struct {type_name} {{");
    let start = source
        .find(&marker)
        .ok_or_else(|| OpsError::Invalid(format!("type {type_name} not found")))?;
    let body_start = start + marker.len();
    let rest = &source[body_start..];
    let end = rest.find('}').ok_or_else(|| {
        OpsError::Invalid(format!("closing brace not found for struct {type_name}"))
    })?;
    let body = &rest[..end];

    let mut required_fields = Vec::new();
    let mut optional_fields = Vec::new();
    let mut field_types = BTreeMap::new();
    let mut pending_default = false;

    for raw in body.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with("#[serde(default)") {
            pending_default = true;
            continue;
        }
        if !line.starts_with("pub ") {
            pending_default = false;
            continue;
        }
        let Some((name, ty_raw)) = line
            .strip_prefix("pub ")
            .and_then(|rest| rest.split_once(':'))
        else {
            continue;
        };
        let field_name = name.trim().to_string();
        let mut ty = ty_raw.trim().trim_end_matches(',').to_string();
        ty.retain(|c| !c.is_whitespace());
        field_types.insert(field_name.clone(), ty.clone());
        if pending_default || ty.starts_with("Option<") {
            optional_fields.push(field_name);
        } else {
            required_fields.push(field_name);
        }
        pending_default = false;
    }

    required_fields.sort();
    optional_fields.sort();

    Ok(StructShape {
        required_fields,
        optional_fields,
        field_types,
    })
}

fn parse_enum_variants(source: &str, enum_name: &str) -> Result<Vec<String>, OpsError> {
    let marker = format!("pub enum {enum_name} {{");
    let start = source
        .find(&marker)
        .ok_or_else(|| OpsError::Invalid(format!("enum {enum_name} not found")))?;
    let body_start = start + marker.len();
    let rest = &source[body_start..];
    let end = rest.find('}').ok_or_else(|| {
        OpsError::Invalid(format!("closing brace not found for enum {enum_name}"))
    })?;
    let body = &rest[..end];
    let mut out = Vec::new();
    for raw in body.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let candidate = line
            .trim_end_matches(',')
            .split_once('(')
            .map(|(head, _)| head)
            .unwrap_or(line)
            .split_once('{')
            .map(|(head, _)| head)
            .unwrap_or(line)
            .trim_end_matches(',')
            .trim();
        if candidate
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_uppercase())
        {
            out.push(candidate.to_string());
        }
    }
    out.sort();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_additive_optional_field() {
        let old = ArtifactSchemaSnapshot {
            artifact_id: "x".to_string(),
            type_name: "T".to_string(),
            source_file: "f".to_string(),
            required_fields: vec!["a".to_string()],
            optional_fields: vec![],
            field_types: BTreeMap::from([("a".to_string(), "u64".to_string())]),
            enum_variants: BTreeMap::new(),
        };
        let new = ArtifactSchemaSnapshot {
            optional_fields: vec!["b".to_string()],
            field_types: BTreeMap::from([
                ("a".to_string(), "u64".to_string()),
                ("b".to_string(), "Option<String>".to_string()),
            ]),
            ..old.clone()
        };
        let (kind, _) = classify_drift(&old, &new);
        assert_eq!(kind, DriftKind::Additive);
    }

    #[test]
    fn classify_breaking_removed_field() {
        let old = ArtifactSchemaSnapshot {
            artifact_id: "x".to_string(),
            type_name: "T".to_string(),
            source_file: "f".to_string(),
            required_fields: vec!["a".to_string()],
            optional_fields: vec![],
            field_types: BTreeMap::from([("a".to_string(), "u64".to_string())]),
            enum_variants: BTreeMap::new(),
        };
        let new = ArtifactSchemaSnapshot {
            required_fields: vec![],
            optional_fields: vec![],
            field_types: BTreeMap::new(),
            ..old.clone()
        };
        let (kind, _) = classify_drift(&old, &new);
        assert_eq!(kind, DriftKind::Breaking);
    }

    #[test]
    fn parse_struct_captures_optional_and_required() {
        let source = r#"
            pub struct Demo {
                pub required: String,
                #[serde(default)]
                pub optional_list: Vec<String>,
                pub optional_number: Option<u64>,
            }
        "#;
        let parsed = parse_struct_shape(source, "Demo").expect("parse");
        assert_eq!(parsed.required_fields, vec!["required".to_string()]);
        assert_eq!(
            parsed.optional_fields,
            vec!["optional_list".to_string(), "optional_number".to_string()]
        );
    }

    #[test]
    fn parse_struct_field_order_is_stable_sorted() {
        let source = r#"
            pub struct Demo {
                pub zeta: String,
                pub alpha: String,
                pub maybe: Option<u64>,
            }
        "#;
        let parsed = parse_struct_shape(source, "Demo").expect("parse");
        assert_eq!(
            parsed.required_fields,
            vec!["alpha".to_string(), "zeta".to_string()]
        );
        assert_eq!(parsed.optional_fields, vec!["maybe".to_string()]);
    }

    #[test]
    fn generated_artifact_order_is_deterministic() {
        let observed: Vec<_> = sorted_artifact_specs()
            .into_iter()
            .map(|spec| spec.artifact_id)
            .collect();
        assert_eq!(
            observed,
            vec![
                "active_review_snapshot_v1",
                "applied_scope_authority_v1",
                "applied_supported_set_context_v1",
                "backend_evidence_snapshot_v1",
                "backend_resolution_v1",
                "bugkit_manifest_v1",
                "bundle_roundtrip_consistency_v1",
                "canonical_bundle_authority_v2",
                "canonical_bundle_consumption_context_v1",
                "canonical_bundle_spine_v1",
                "canonical_continuity_authority_v1",
                "canonical_export_artifact_ref_v1",
                "canonical_export_context_v1",
                "canonical_governance_entry_authority_v2",
                "canonical_governance_entry_v1",
                "canonical_primary_semantics_authority_v1",
                "canonical_readiness_authority_v2",
                "canonical_readiness_spine_v1",
                "canonical_roundtrip_chain_v1",
                "cross_surface_condition_observation_v1",
                "cross_surface_context_matrix_v1",
                "governance_primary_surfaces_v1",
                "interop_consistency_matrix_report_v1",
                "operator_report_v1",
                "operator_review_packet_v1",
                "operator_signoff_v1",
                "operator_workflow_chain_v1",
                "primary_semantics_observation_v1",
                "readiness_gate_report_v1",
                "remediation_consistency_check_v1",
                "repro_pack_manifest_v1",
                "reviewability_reduction_v1",
                "slot_reviewability_truth_v1",
                "spine_condition_observation_v1",
                "strict_failure_report_v3",
                "supported_real_slot_set_v2",
                "supported_scope_execution_v3",
                "supported_scope_execution_v4",
                "supported_scope_reevaluation_v1",
                "v3_gate_report_v1",
                "v4_gate_report_v1",
                "v5_gate_report_v1",
                "v7_gate_report_v1",
                "v8_gate_report_v1",
            ]
        );
    }

    #[test]
    fn check_reports_missing_snapshot_as_breaking() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let report = check_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("runtime parent")
                .parent()
                .expect("repo root")
                .to_path_buf(),
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("check should complete");
        assert!(!report.ok);
        assert!(report.diffs.iter().any(|d| {
            d.artifact == "active_review_snapshot_v1" && d.drift_kind == DriftKind::Breaking
        }));
    }
}
