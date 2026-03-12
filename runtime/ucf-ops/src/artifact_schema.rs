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
}

#[derive(Debug, Clone, Copy)]
struct ArtifactSpec {
    artifact_id: &'static str,
    file_rel: &'static str,
    type_name: &'static str,
    enum_names: &'static [&'static str],
}

const ARTIFACT_SPECS: [ArtifactSpec; 8] = [
    ArtifactSpec {
        artifact_id: "active_review_snapshot_v1",
        file_rel: "runtime/ucf-ops/src/models_lifecycle.rs",
        type_name: "AggregatedActiveReviewSnapshotV1",
        enum_names: &["ActiveReviewOverallStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "backend_evidence_snapshot_v1",
        file_rel: "runtime/ucf-ops/src/models_lifecycle.rs",
        type_name: "BackendEvidenceSnapshotV1",
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
        artifact_id: "readiness_gate_report_v1",
        file_rel: "runtime/ucf-ops/src/lib.rs",
        type_name: "ReadinessGateReport",
        enum_names: &["GateStatus"],
    },
];

pub fn generate_artifact_schema_snapshots(
    args: &ArtifactSchemaArgs,
) -> Result<Vec<String>, OpsError> {
    fs::create_dir_all(&args.out_dir)?;
    let mut covered = Vec::new();
    for spec in ARTIFACT_SPECS {
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

    Ok(ArtifactSchemaCheckReport {
        ok: diffs.is_empty(),
        covered_artifacts: covered,
        diffs,
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
}
