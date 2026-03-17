#![allow(clippy::result_large_err)]
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::{
    check_artifact_schema_snapshots, generate_remediation_codes_doc, generate_spec_snapshot,
    policy_validate, ArtifactSchemaArgs, DriftKind, OpsError, SpecSnapshotArgs,
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DocsLintMode {
    Strict,
    Warn,
}

#[derive(Debug, Clone)]
pub struct DocsLintArgs {
    pub repo_root: PathBuf,
    pub policy_pack: PathBuf,
    pub overlay_pack: Option<PathBuf>,
    pub spec_snapshot: PathBuf,
    pub prompt_index: PathBuf,
    pub module_map: PathBuf,
    pub deploy_doc: PathBuf,
    pub artifact_schema_snapshot_dir: PathBuf,
    pub mode: DocsLintMode,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DocsLintStatus {
    Pass,
    Warn,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocsLintCheck {
    pub name: String,
    pub status: DocsLintStatus,
    pub detail: String,
    pub remediation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocsLintReport {
    pub ok: bool,
    pub mode: DocsLintMode,
    pub checks: Vec<DocsLintCheck>,
}

pub fn docs_lint(args: &DocsLintArgs) -> Result<DocsLintReport, OpsError> {
    let checks = vec![
        spec_snapshot_check(args)?,
        policy_pack_check(args)?,
        prompt_index_check(args)?,
        module_map_check(args)?,
        hardware_neutral_docs_check(args)?,
        v3_docs_consistency_check(args)?,
        v4_docs_consistency_check(args)?,
        v5_docs_consistency_check(args)?,
        v6_docs_consistency_check(args)?,
        v7_docs_consistency_check(args)?,
        v8_docs_consistency_check(args)?,
        remediation_registry_doc_check(args)?,
        artifact_schema_snapshot_check(args)?,
    ];
    let ok = checks.iter().all(|c| c.status != DocsLintStatus::Fail);
    Ok(DocsLintReport {
        ok,
        mode: args.mode,
        checks,
    })
}

fn spec_snapshot_check(args: &DocsLintArgs) -> Result<DocsLintCheck, OpsError> {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| OpsError::Invalid(format!("system clock error: {e}")))?
        .as_nanos();
    let temp_out = std::env::temp_dir().join(format!("ucf_spec_snapshot_{ts}.md"));
    generate_spec_snapshot(&SpecSnapshotArgs {
        policy: args.policy_pack.clone(),
        overlay: args.overlay_pack.clone(),
        out: temp_out.clone(),
    })?;

    let generated = fs::read_to_string(&temp_out)?;
    let committed = fs::read_to_string(&args.spec_snapshot)?;
    let _ = fs::remove_file(&temp_out);

    if generated == committed {
        return Ok(DocsLintCheck {
            name: "spec_snapshot".to_string(),
            status: DocsLintStatus::Pass,
            detail: "docs/spec_snapshot.md is up-to-date".to_string(),
            remediation: None,
        });
    }

    let diff = first_diff_line(&committed, &generated);
    Ok(DocsLintCheck {
        name: "spec_snapshot".to_string(),
        status: DocsLintStatus::Fail,
        detail: format!(
            "docs/spec_snapshot.md differs from regenerated snapshot at line {}",
            diff
        ),
        remediation: Some(
            "run: cargo run -p ucf-ops -- spec snapshot --policy policies/packs/base_v1 --overlay policies/packs/overlays/test --out docs/spec_snapshot.md && git add docs/spec_snapshot.md"
                .to_string(),
        ),
    })
}

fn artifact_schema_snapshot_check(args: &DocsLintArgs) -> Result<DocsLintCheck, OpsError> {
    let report = check_artifact_schema_snapshots(&ArtifactSchemaArgs {
        repo_root: args.repo_root.clone(),
        out_dir: args.artifact_schema_snapshot_dir.clone(),
    })?;

    if report.ok {
        return Ok(DocsLintCheck {
            name: "artifact_schema_snapshots".to_string(),
            status: DocsLintStatus::Pass,
            detail: format!(
                "artifact schema snapshots are up-to-date ({} artifacts)",
                report.covered_artifacts.len()
            ),
            remediation: None,
        });
    }

    let summary = report
        .diffs
        .iter()
        .map(|d| format!("{}:{:?}:{}", d.artifact, d.drift_kind, d.summary))
        .collect::<Vec<_>>()
        .join(" | ");

    let breaking = report
        .diffs
        .iter()
        .any(|d| matches!(d.drift_kind, DriftKind::Breaking | DriftKind::Unknown));

    Ok(DocsLintCheck {
        name: "artifact_schema_snapshots".to_string(),
        status: if breaking || matches!(args.mode, DocsLintMode::Strict) {
            DocsLintStatus::Fail
        } else {
            DocsLintStatus::Warn
        },
        detail: format!("schema snapshot drift detected: {summary}"),
        remediation: Some("run: cargo run -p ucf-ops -- spec artifact-schemas --out docs/artifact_schema_snapshots && review git diff && git add docs/artifact_schema_snapshots".to_string()),
    })
}

fn policy_pack_check(args: &DocsLintArgs) -> Result<DocsLintCheck, OpsError> {
    match policy_validate(&args.policy_pack, args.overlay_pack.as_deref()) {
        Ok(report) => Ok(DocsLintCheck {
            name: "policy_packs".to_string(),
            status: DocsLintStatus::Pass,
            detail: format!(
                "policy graph valid (schema={} digest={})",
                report.schema_version, report.policy_graph_digest
            ),
            remediation: None,
        }),
        Err(err) => Ok(DocsLintCheck {
            name: "policy_packs".to_string(),
            status: DocsLintStatus::Fail,
            detail: format!("policy validation failed: {err}"),
            remediation: Some(
                "run: cargo run -p ucf-ops -- policy validate --pack policies/packs/base_v1 --overlay policies/packs/overlays/test"
                    .to_string(),
            ),
        }),
    }
}

fn prompt_index_check(args: &DocsLintArgs) -> Result<DocsLintCheck, OpsError> {
    let body = fs::read_to_string(&args.prompt_index)?;
    let parsed = parse_prompt_ids(&body)?;
    if parsed.ids.is_empty() {
        return Ok(DocsLintCheck {
            name: "prompt_series_index".to_string(),
            status: DocsLintStatus::Fail,
            detail: "no prompt IDs found in docs/prompt_series_index.md".to_string(),
            remediation: Some(
                "ensure docs/prompt_series_index.md includes prompt IDs (table `| ID |` rows or `PROMPT <N> —` headings)"
                    .to_string(),
            ),
        });
    }

    let mut seen = BTreeSet::new();
    for id in &parsed.ids {
        if !seen.insert(*id) {
            return Ok(DocsLintCheck {
                name: "prompt_series_index".to_string(),
                status: DocsLintStatus::Fail,
                detail: format!("duplicate prompt ID detected: {id}"),
                remediation: Some(
                    "remove duplicate prompt ID entries in docs/prompt_series_index.md".to_string(),
                ),
            });
        }
    }

    for pair in parsed.ids.windows(2) {
        if pair[1] <= pair[0] {
            return Ok(DocsLintCheck {
                name: "prompt_series_index".to_string(),
                status: DocsLintStatus::Fail,
                detail: format!(
                    "prompt IDs are not strictly increasing: {} followed by {}",
                    pair[0], pair[1]
                ),
                remediation: Some("sort prompt entries so IDs are strictly increasing".to_string()),
            });
        }
    }

    if let Some(line) = parsed.invalid_header_line {
        return Ok(DocsLintCheck {
            name: "prompt_series_index".to_string(),
            status: DocsLintStatus::Fail,
            detail: format!("invalid prompt header format at line {line}"),
            remediation: Some(
                "use heading format `PROMPT <N> — <title>` for prompt headers".to_string(),
            ),
        });
    }

    Ok(DocsLintCheck {
        name: "prompt_series_index".to_string(),
        status: DocsLintStatus::Pass,
        detail: format!("{} prompt IDs validated", parsed.ids.len()),
        remediation: None,
    })
}

fn module_map_check(args: &DocsLintArgs) -> Result<DocsLintCheck, OpsError> {
    let module_map_body = fs::read_to_string(&args.module_map)?;
    let module_keys = parse_module_map_keys(&module_map_body);
    let metadata_names = cargo_metadata_package_names(&args.repo_root)?;

    let unknown: Vec<String> = module_keys
        .iter()
        .filter(|k| looks_like_crate_name(k) && !metadata_names.contains(k.as_str()))
        .cloned()
        .collect();

    if unknown.is_empty() {
        return Ok(DocsLintCheck {
            name: "module_map".to_string(),
            status: DocsLintStatus::Pass,
            detail: "module_map crate entries match cargo metadata".to_string(),
            remediation: None,
        });
    }

    let status = if matches!(args.mode, DocsLintMode::Strict) {
        DocsLintStatus::Fail
    } else {
        DocsLintStatus::Warn
    };

    Ok(DocsLintCheck {
        name: "module_map".to_string(),
        status,
        detail: format!(
            "{} module_map entries do not match cargo metadata: {}",
            unknown.len(),
            unknown.join(", ")
        ),
        remediation: Some(
            "update docs/module_map.md crate labels to match `cargo metadata --no-deps --format-version 1` package names"
                .to_string(),
        ),
    })
}

fn hardware_neutral_docs_check(args: &DocsLintArgs) -> Result<DocsLintCheck, OpsError> {
    let files = [
        ("prompt_series_index", &args.prompt_index, false),
        (
            "prompt_rulebook",
            &args.repo_root.join("docs").join("prompt_rulebook.md"),
            false,
        ),
        ("deploy_portable", &args.deploy_doc, true),
        (
            "models_eligibility_v3",
            &args.repo_root.join("docs").join("models_eligibility_v3.md"),
            false,
        ),
        (
            "strict_mode_v3",
            &args.repo_root.join("docs").join("strict_mode_v3.md"),
            false,
        ),
        (
            "operator_report_v3",
            &args.repo_root.join("docs").join("operator_report_v3.md"),
            false,
        ),
        (
            "backend_evidence_snapshot_v4",
            &args
                .repo_root
                .join("docs")
                .join("backend_evidence_snapshot_v4.md"),
            false,
        ),
        (
            "operator_signoff_v4",
            &args.repo_root.join("docs").join("operator_signoff_v4.md"),
            false,
        ),
        (
            "remediation_codes_v1",
            &args.repo_root.join("docs").join("remediation_codes_v1.md"),
            false,
        ),
        (
            "artifact_schema_snapshots",
            &args
                .repo_root
                .join("docs")
                .join("artifact_schema_snapshots.md"),
            false,
        ),
        (
            "active_review_snapshot_v5",
            &args
                .repo_root
                .join("docs")
                .join("active_review_snapshot_v5.md"),
            false,
        ),
        (
            "sae_burn_resolution_v5",
            &args
                .repo_root
                .join("docs")
                .join("sae_burn_resolution_v5.md"),
            false,
        ),
        (
            "repro_pack",
            &args.repo_root.join("docs").join("repro_pack.md"),
            false,
        ),
        (
            "bug_report_kit",
            &args.repo_root.join("docs").join("bug_report_kit.md"),
            false,
        ),
        (
            "remediation_consistency_v5",
            &args
                .repo_root
                .join("docs")
                .join("remediation_consistency_v5.md"),
            false,
        ),
        (
            "governance_primary_surfaces_v6",
            &args
                .repo_root
                .join("docs")
                .join("governance_primary_surfaces_v6.md"),
            false,
        ),
        (
            "supported_set_apply_v6",
            &args
                .repo_root
                .join("docs")
                .join("supported_set_apply_v6.md"),
            false,
        ),
        (
            "applied_supported_scope_v6",
            &args
                .repo_root
                .join("docs")
                .join("applied_supported_scope_v6.md"),
            false,
        ),
        (
            "export_normalization_v6",
            &args
                .repo_root
                .join("docs")
                .join("export_normalization_v6.md"),
            false,
        ),
        (
            "interop_consistency_v6",
            &args
                .repo_root
                .join("docs")
                .join("interop_consistency_v6.md"),
            false,
        ),
        (
            "applied_scope_authority_v7",
            &args
                .repo_root
                .join("docs")
                .join("applied_scope_authority_v7.md"),
            false,
        ),
        (
            "supported_scope_reevaluation_v7",
            &args
                .repo_root
                .join("docs")
                .join("supported_scope_reevaluation_v7.md"),
            false,
        ),
        (
            "reviewability_truth_v7",
            &args
                .repo_root
                .join("docs")
                .join("reviewability_truth_v7.md"),
            false,
        ),
        (
            "export_roundtrip_v7",
            &args.repo_root.join("docs").join("export_roundtrip_v7.md"),
            false,
        ),
        (
            "remediation_interop_consistency_v7",
            &args
                .repo_root
                .join("docs")
                .join("remediation_interop_consistency_v7.md"),
            false,
        ),
        (
            "canonical_governance_entry_v8",
            &args
                .repo_root
                .join("docs")
                .join("canonical_governance_entry_v8.md"),
            false,
        ),
        (
            "supported_scope_execution_v8",
            &args
                .repo_root
                .join("docs")
                .join("supported_scope_execution_v8.md"),
            false,
        ),
        (
            "readiness_spine_v8",
            &args.repo_root.join("docs").join("readiness_spine_v8.md"),
            false,
        ),
        (
            "bundle_spine_v8",
            &args.repo_root.join("docs").join("bundle_spine_v8.md"),
            false,
        ),
        (
            "remediation_spine_consistency_v8",
            &args
                .repo_root
                .join("docs")
                .join("remediation_spine_consistency_v8.md"),
            false,
        ),
    ];

    let banned = [
        "NUC",
        "Raspberry Pi",
        "RPi",
        "Intel Core",
        "Xeon",
        "AMD Ryzen",
        "Threadripper",
    ];

    let mut warnings = Vec::new();
    let mut failures = Vec::new();

    for (label, path, is_deploy_doc) in files {
        if !path.exists() {
            continue;
        }
        let body = fs::read_to_string(path)?;
        let mut in_history = false;
        for (idx, raw) in body.lines().enumerate() {
            let line = raw.trim();
            if line.starts_with('#') {
                in_history = line.to_ascii_lowercase().contains("history");
            }

            for term in &banned {
                if !line.contains(term) {
                    continue;
                }
                let hit = format!("{label}:{} contains `{term}`", idx + 1);
                if is_deploy_doc || in_history {
                    warnings.push(hit);
                } else {
                    failures.push(hit);
                }
            }
        }
    }

    if !failures.is_empty() {
        return Ok(DocsLintCheck {
            name: "hardware_neutral_docs".to_string(),
            status: DocsLintStatus::Fail,
            detail: format!(
                "hardware-specific terms found in core docs: {}",
                failures.join("; ")
            ),
            remediation: Some(
                "replace machine/vendor terms with DeviceProfile (small/medium/large) wording or move historical references into clearly marked History sections"
                    .to_string(),
            ),
        });
    }

    if !warnings.is_empty() {
        return Ok(DocsLintCheck {
            name: "hardware_neutral_docs".to_string(),
            status: DocsLintStatus::Warn,
            detail: format!(
                "hardware-specific terms allowed in deploy/history scope: {}",
                warnings.join("; ")
            ),
            remediation: None,
        });
    }

    Ok(DocsLintCheck {
        name: "hardware_neutral_docs".to_string(),
        status: DocsLintStatus::Pass,
        detail: "no hardware-specific terms detected in guarded docs".to_string(),
        remediation: None,
    })
}

fn remediation_registry_doc_check(args: &DocsLintArgs) -> Result<DocsLintCheck, OpsError> {
    let committed = args.repo_root.join("docs/remediation_codes_v1.md");
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| OpsError::Invalid(format!("system clock error: {e}")))?
        .as_nanos();
    let generated = std::env::temp_dir().join(format!("ucf_remediation_codes_{ts}.md"));
    generate_remediation_codes_doc(&generated)?;
    let committed_body = fs::read_to_string(&committed)?;
    let generated_body = fs::read_to_string(&generated)?;
    let _ = fs::remove_file(&generated);

    let committed_norm = normalize_newlines(&committed_body);
    let generated_norm = normalize_newlines(&generated_body);

    if committed_norm == generated_norm {
        return Ok(DocsLintCheck {
            name: "remediation_registry_doc".to_string(),
            status: DocsLintStatus::Pass,
            detail: "docs/remediation_codes_v1.md is up-to-date".to_string(),
            remediation: None,
        });
    }

    Ok(DocsLintCheck {
        name: "remediation_registry_doc".to_string(),
        status: DocsLintStatus::Fail,
        detail: format!(
            "docs/remediation_codes_v1.md differs from generated registry at line {}",
            first_diff_line(&committed_norm, &generated_norm)
        ),
        remediation: Some(
            "run: cargo run -p ucf-ops -- docs remediation-codes --out docs/remediation_codes_v1.md && git add docs/remediation_codes_v1.md"
                .to_string(),
        ),
    })
}

fn normalize_newlines(input: &str) -> String {
    input.replace("\r\n", "\n")
}

fn v4_docs_consistency_check(args: &DocsLintArgs) -> Result<DocsLintCheck, OpsError> {
    let required = [
        "docs/backend_evidence_snapshot_v4.md",
        "docs/operator_signoff_v4.md",
        "docs/remediation_codes_v1.md",
        "docs/artifact_schema_snapshots.md",
    ];
    for path in required {
        if !args.repo_root.join(path).exists() {
            return Ok(DocsLintCheck {
                name: "v4_docs_consistency".to_string(),
                status: DocsLintStatus::Fail,
                detail: format!("missing required v4 doc: {path}"),
                remediation: Some("restore missing v4 docs and re-run docs lint".to_string()),
            });
        }
    }

    let portability_gate = fs::read_to_string(args.repo_root.join("docs/portability_gate.md"))?;
    let docs_checks = fs::read_to_string(args.repo_root.join("docs/docs_checks.md"))?;
    let series_snapshot = fs::read_to_string(args.repo_root.join("docs/series_state_snapshot.md"))?;

    let missing = [
        (
            "docs/portability_gate.md",
            "backend_evidence_snapshot_v4.md",
            portability_gate.contains("backend_evidence_snapshot_v4.md"),
        ),
        (
            "docs/portability_gate.md",
            "operator_signoff_v4.md",
            portability_gate.contains("operator_signoff_v4.md"),
        ),
        (
            "docs/portability_gate.md",
            "remediation_codes_v1.md",
            portability_gate.contains("remediation_codes_v1.md"),
        ),
        (
            "docs/portability_gate.md",
            "artifact_schema_snapshots.md",
            portability_gate.contains("artifact_schema_snapshots.md"),
        ),
        (
            "docs/docs_checks.md",
            "docs/remediation_codes_v1.md",
            docs_checks.contains("docs/remediation_codes_v1.md"),
        ),
        (
            "docs/docs_checks.md",
            "docs/artifact_schema_snapshots.md",
            docs_checks.contains("docs/artifact_schema_snapshots.md"),
        ),
        (
            "docs/series_state_snapshot.md",
            "| 216 |",
            series_snapshot.contains("| 216 |"),
        ),
    ]
    .into_iter()
    .filter_map(|(file, needle, present)| {
        (!present).then_some(format!("{file} missing `{needle}`"))
    })
    .collect::<Vec<_>>();

    if !missing.is_empty() {
        return Ok(DocsLintCheck {
            name: "v4_docs_consistency".to_string(),
            status: DocsLintStatus::Fail,
            detail: format!("v4 docs linkage mismatch: {}", missing.join("; ")),
            remediation: Some(
                "link v4 docs from portability/docs checks and keep series snapshot in sync"
                    .to_string(),
            ),
        });
    }

    Ok(DocsLintCheck {
        name: "v4_docs_consistency".to_string(),
        status: DocsLintStatus::Pass,
        detail: "v4 docs are present and linked from portability/docs checks/index snapshots"
            .to_string(),
        remediation: None,
    })
}

fn v5_docs_consistency_check(args: &DocsLintArgs) -> Result<DocsLintCheck, OpsError> {
    let required = [
        "docs/active_review_snapshot_v5.md",
        "docs/sae_burn_resolution_v5.md",
        "docs/repro_pack.md",
        "docs/bug_report_kit.md",
        "docs/remediation_consistency_v5.md",
        "docs/artifact_schema_snapshots.md",
    ];
    for path in required {
        if !args.repo_root.join(path).exists() {
            return Ok(DocsLintCheck {
                name: "v5_docs_consistency".to_string(),
                status: DocsLintStatus::Fail,
                detail: format!("missing required v5 doc: {path}"),
                remediation: Some("restore missing v5 docs and re-run docs lint".to_string()),
            });
        }
    }

    let portability_gate = fs::read_to_string(args.repo_root.join("docs/portability_gate.md"))?;
    let docs_checks = fs::read_to_string(args.repo_root.join("docs/docs_checks.md"))?;

    let missing = [
        (
            "docs/portability_gate.md",
            "active_review_snapshot_v5.md",
            portability_gate.contains("active_review_snapshot_v5.md"),
        ),
        (
            "docs/portability_gate.md",
            "sae_burn_resolution_v5.md",
            portability_gate.contains("sae_burn_resolution_v5.md"),
        ),
        (
            "docs/portability_gate.md",
            "repro_pack.md",
            portability_gate.contains("repro_pack.md"),
        ),
        (
            "docs/portability_gate.md",
            "bug_report_kit.md",
            portability_gate.contains("bug_report_kit.md"),
        ),
        (
            "docs/portability_gate.md",
            "remediation_consistency_v5.md",
            portability_gate.contains("remediation_consistency_v5.md"),
        ),
        (
            "docs/docs_checks.md",
            "docs/active_review_snapshot_v5.md",
            docs_checks.contains("docs/active_review_snapshot_v5.md"),
        ),
        (
            "docs/docs_checks.md",
            "docs/sae_burn_resolution_v5.md",
            docs_checks.contains("docs/sae_burn_resolution_v5.md"),
        ),
        (
            "docs/docs_checks.md",
            "docs/repro_pack.md",
            docs_checks.contains("docs/repro_pack.md"),
        ),
        (
            "docs/docs_checks.md",
            "docs/bug_report_kit.md",
            docs_checks.contains("docs/bug_report_kit.md"),
        ),
        (
            "docs/docs_checks.md",
            "docs/remediation_consistency_v5.md",
            docs_checks.contains("docs/remediation_consistency_v5.md"),
        ),
    ]
    .into_iter()
    .filter_map(|(file, needle, present)| {
        (!present).then_some(format!("{file} missing `{needle}`"))
    })
    .collect::<Vec<_>>();

    if !missing.is_empty() {
        return Ok(DocsLintCheck {
            name: "v5_docs_consistency".to_string(),
            status: DocsLintStatus::Fail,
            detail: format!("v5 docs linkage mismatch: {}", missing.join("; ")),
            remediation: Some(
                "link v5 docs from portability/docs checks and keep docs references in sync"
                    .to_string(),
            ),
        });
    }

    Ok(DocsLintCheck {
        name: "v5_docs_consistency".to_string(),
        status: DocsLintStatus::Pass,
        detail: "v5 docs are present and linked from portability/docs checks".to_string(),
        remediation: None,
    })
}

fn v3_docs_consistency_check(args: &DocsLintArgs) -> Result<DocsLintCheck, OpsError> {
    let required = [
        "docs/models_eligibility_v3.md",
        "docs/strict_mode_v3.md",
        "docs/operator_report_v3.md",
    ];
    for path in required {
        if !args.repo_root.join(path).exists() {
            return Ok(DocsLintCheck {
                name: "v3_docs_consistency".to_string(),
                status: DocsLintStatus::Fail,
                detail: format!("missing required v3 doc: {path}"),
                remediation: Some("restore missing v3 docs and re-run docs lint".to_string()),
            });
        }
    }

    let portability_gate = fs::read_to_string(args.repo_root.join("docs/portability_gate.md"))?;
    let strict_mode = fs::read_to_string(args.repo_root.join("docs/strict_mode.md"))?;
    let series_snapshot = fs::read_to_string(args.repo_root.join("docs/series_state_snapshot.md"))?;

    let missing = [
        (
            "docs/portability_gate.md",
            "models_eligibility_v3.md",
            portability_gate.contains("models_eligibility_v3.md"),
        ),
        (
            "docs/portability_gate.md",
            "strict_mode_v3.md",
            portability_gate.contains("strict_mode_v3.md"),
        ),
        (
            "docs/portability_gate.md",
            "operator_report_v3.md",
            portability_gate.contains("operator_report_v3.md"),
        ),
        (
            "docs/strict_mode.md",
            "strict_mode_v3.md",
            strict_mode.contains("strict_mode_v3.md"),
        ),
        (
            "docs/series_state_snapshot.md",
            "| 207 |",
            series_snapshot.contains("| 207 |"),
        ),
    ]
    .into_iter()
    .filter_map(|(file, needle, present)| {
        (!present).then_some(format!("{file} missing `{needle}`"))
    })
    .collect::<Vec<_>>();

    if !missing.is_empty() {
        return Ok(DocsLintCheck {
            name: "v3_docs_consistency".to_string(),
            status: DocsLintStatus::Fail,
            detail: format!("v3 docs linkage mismatch: {}", missing.join("; ")),
            remediation: Some(
                "link v3 docs from portability/strict docs and keep prompt index + series snapshot in sync"
                    .to_string(),
            ),
        });
    }

    Ok(DocsLintCheck {
        name: "v3_docs_consistency".to_string(),
        status: DocsLintStatus::Pass,
        detail: "v3 docs are present and linked from portability/strict/index snapshots"
            .to_string(),
        remediation: None,
    })
}

fn v6_docs_consistency_check(args: &DocsLintArgs) -> Result<DocsLintCheck, OpsError> {
    let required = [
        "docs/governance_primary_surfaces_v6.md",
        "docs/supported_set_apply_v6.md",
        "docs/applied_supported_scope_v6.md",
        "docs/export_normalization_v6.md",
        "docs/interop_consistency_v6.md",
        "docs/artifact_schema_snapshots.md",
    ];
    for path in required {
        if !args.repo_root.join(path).exists() {
            return Ok(DocsLintCheck {
                name: "v6_docs_consistency".to_string(),
                status: DocsLintStatus::Fail,
                detail: format!("missing required v6 doc: {path}"),
                remediation: Some("restore missing v6 docs and re-run docs lint".to_string()),
            });
        }
    }

    let portability_gate = fs::read_to_string(args.repo_root.join("docs/portability_gate.md"))?;
    let docs_checks = fs::read_to_string(args.repo_root.join("docs/docs_checks.md"))?;

    let missing = [
        (
            "docs/portability_gate.md",
            "governance_primary_surfaces_v6.md",
            portability_gate.contains("governance_primary_surfaces_v6.md"),
        ),
        (
            "docs/portability_gate.md",
            "supported_set_apply_v6.md",
            portability_gate.contains("supported_set_apply_v6.md"),
        ),
        (
            "docs/portability_gate.md",
            "applied_supported_scope_v6.md",
            portability_gate.contains("applied_supported_scope_v6.md"),
        ),
        (
            "docs/portability_gate.md",
            "export_normalization_v6.md",
            portability_gate.contains("export_normalization_v6.md"),
        ),
        (
            "docs/portability_gate.md",
            "interop_consistency_v6.md",
            portability_gate.contains("interop_consistency_v6.md"),
        ),
        (
            "docs/portability_gate.md",
            "artifact_schema_snapshots.md",
            portability_gate.contains("artifact_schema_snapshots.md"),
        ),
        (
            "docs/docs_checks.md",
            "docs/governance_primary_surfaces_v6.md",
            docs_checks.contains("docs/governance_primary_surfaces_v6.md"),
        ),
        (
            "docs/docs_checks.md",
            "docs/supported_set_apply_v6.md",
            docs_checks.contains("docs/supported_set_apply_v6.md"),
        ),
        (
            "docs/docs_checks.md",
            "docs/applied_supported_scope_v6.md",
            docs_checks.contains("docs/applied_supported_scope_v6.md"),
        ),
        (
            "docs/docs_checks.md",
            "docs/export_normalization_v6.md",
            docs_checks.contains("docs/export_normalization_v6.md"),
        ),
        (
            "docs/docs_checks.md",
            "docs/interop_consistency_v6.md",
            docs_checks.contains("docs/interop_consistency_v6.md"),
        ),
        (
            "docs/docs_checks.md",
            "docs/artifact_schema_snapshots.md",
            docs_checks.contains("docs/artifact_schema_snapshots.md"),
        ),
    ]
    .into_iter()
    .filter_map(|(file, needle, present)| {
        (!present).then_some(format!("{file} missing `{needle}`"))
    })
    .collect::<Vec<_>>();

    if !missing.is_empty() {
        return Ok(DocsLintCheck {
            name: "v6_docs_consistency".to_string(),
            status: DocsLintStatus::Fail,
            detail: format!("v6 docs linkage mismatch: {}", missing.join("; ")),
            remediation: Some(
                "link v6 docs from portability/docs checks and keep docs references in sync"
                    .to_string(),
            ),
        });
    }

    Ok(DocsLintCheck {
        name: "v6_docs_consistency".to_string(),
        status: DocsLintStatus::Pass,
        detail: "v6 docs are present and linked from portability/docs checks".to_string(),
        remediation: None,
    })
}

fn v7_docs_consistency_check(args: &DocsLintArgs) -> Result<DocsLintCheck, OpsError> {
    let required = [
        "docs/applied_scope_authority_v7.md",
        "docs/supported_scope_reevaluation_v7.md",
        "docs/reviewability_truth_v7.md",
        "docs/export_roundtrip_v7.md",
        "docs/remediation_interop_consistency_v7.md",
        "docs/artifact_schema_snapshots.md",
    ];
    for path in required {
        if !args.repo_root.join(path).exists() {
            return Ok(DocsLintCheck {
                name: "v7_docs_consistency".to_string(),
                status: DocsLintStatus::Fail,
                detail: format!("missing required v7 doc: {path}"),
                remediation: Some("restore missing v7 docs and re-run docs lint".to_string()),
            });
        }
    }

    let portability_gate = fs::read_to_string(args.repo_root.join("docs/portability_gate.md"))?;
    let docs_checks = fs::read_to_string(args.repo_root.join("docs/docs_checks.md"))?;

    let missing = [
        (
            "docs/portability_gate.md",
            "applied_scope_authority_v7.md",
            portability_gate.contains("applied_scope_authority_v7.md"),
        ),
        (
            "docs/portability_gate.md",
            "supported_scope_reevaluation_v7.md",
            portability_gate.contains("supported_scope_reevaluation_v7.md"),
        ),
        (
            "docs/portability_gate.md",
            "reviewability_truth_v7.md",
            portability_gate.contains("reviewability_truth_v7.md"),
        ),
        (
            "docs/portability_gate.md",
            "export_roundtrip_v7.md",
            portability_gate.contains("export_roundtrip_v7.md"),
        ),
        (
            "docs/portability_gate.md",
            "remediation_interop_consistency_v7.md",
            portability_gate.contains("remediation_interop_consistency_v7.md"),
        ),
        (
            "docs/portability_gate.md",
            "artifact_schema_snapshots.md",
            portability_gate.contains("artifact_schema_snapshots.md"),
        ),
        (
            "docs/docs_checks.md",
            "docs/applied_scope_authority_v7.md",
            docs_checks.contains("docs/applied_scope_authority_v7.md"),
        ),
        (
            "docs/docs_checks.md",
            "docs/supported_scope_reevaluation_v7.md",
            docs_checks.contains("docs/supported_scope_reevaluation_v7.md"),
        ),
        (
            "docs/docs_checks.md",
            "docs/reviewability_truth_v7.md",
            docs_checks.contains("docs/reviewability_truth_v7.md"),
        ),
        (
            "docs/docs_checks.md",
            "docs/export_roundtrip_v7.md",
            docs_checks.contains("docs/export_roundtrip_v7.md"),
        ),
        (
            "docs/docs_checks.md",
            "docs/remediation_interop_consistency_v7.md",
            docs_checks.contains("docs/remediation_interop_consistency_v7.md"),
        ),
        (
            "docs/docs_checks.md",
            "docs/artifact_schema_snapshots.md",
            docs_checks.contains("docs/artifact_schema_snapshots.md"),
        ),
    ]
    .into_iter()
    .filter_map(|(file, needle, present)| {
        (!present).then_some(format!("{file} missing `{needle}`"))
    })
    .collect::<Vec<_>>();

    if !missing.is_empty() {
        return Ok(DocsLintCheck {
            name: "v7_docs_consistency".to_string(),
            status: DocsLintStatus::Fail,
            detail: format!("v7 docs linkage mismatch: {}", missing.join("; ")),
            remediation: Some(
                "link v7 docs from portability/docs checks and keep docs references in sync"
                    .to_string(),
            ),
        });
    }

    Ok(DocsLintCheck {
        name: "v7_docs_consistency".to_string(),
        status: DocsLintStatus::Pass,
        detail: "v7 docs are present and linked from portability/docs checks".to_string(),
        remediation: None,
    })
}

fn v8_docs_consistency_check(args: &DocsLintArgs) -> Result<DocsLintCheck, OpsError> {
    let required = [
        "docs/canonical_governance_entry_v8.md",
        "docs/supported_scope_execution_v8.md",
        "docs/readiness_spine_v8.md",
        "docs/bundle_spine_v8.md",
        "docs/remediation_spine_consistency_v8.md",
        "docs/artifact_schema_snapshots.md",
    ];
    for path in required {
        if !args.repo_root.join(path).exists() {
            return Ok(DocsLintCheck {
                name: "v8_docs_consistency".to_string(),
                status: DocsLintStatus::Fail,
                detail: format!("missing required v8 doc: {path}"),
                remediation: Some("restore missing v8 docs and re-run docs lint".to_string()),
            });
        }
    }

    let portability_gate = fs::read_to_string(args.repo_root.join("docs/portability_gate.md"))?;
    let docs_checks = fs::read_to_string(args.repo_root.join("docs/docs_checks.md"))?;

    let missing = [
        (
            "docs/portability_gate.md",
            "canonical_governance_entry_v8.md",
            portability_gate.contains("canonical_governance_entry_v8.md"),
        ),
        (
            "docs/portability_gate.md",
            "supported_scope_execution_v8.md",
            portability_gate.contains("supported_scope_execution_v8.md"),
        ),
        (
            "docs/portability_gate.md",
            "readiness_spine_v8.md",
            portability_gate.contains("readiness_spine_v8.md"),
        ),
        (
            "docs/portability_gate.md",
            "bundle_spine_v8.md",
            portability_gate.contains("bundle_spine_v8.md"),
        ),
        (
            "docs/portability_gate.md",
            "remediation_spine_consistency_v8.md",
            portability_gate.contains("remediation_spine_consistency_v8.md"),
        ),
        (
            "docs/portability_gate.md",
            "artifact_schema_snapshots.md",
            portability_gate.contains("artifact_schema_snapshots.md"),
        ),
        (
            "docs/docs_checks.md",
            "docs/canonical_governance_entry_v8.md",
            docs_checks.contains("docs/canonical_governance_entry_v8.md"),
        ),
        (
            "docs/docs_checks.md",
            "docs/supported_scope_execution_v8.md",
            docs_checks.contains("docs/supported_scope_execution_v8.md"),
        ),
        (
            "docs/docs_checks.md",
            "docs/readiness_spine_v8.md",
            docs_checks.contains("docs/readiness_spine_v8.md"),
        ),
        (
            "docs/docs_checks.md",
            "docs/bundle_spine_v8.md",
            docs_checks.contains("docs/bundle_spine_v8.md"),
        ),
        (
            "docs/docs_checks.md",
            "docs/remediation_spine_consistency_v8.md",
            docs_checks.contains("docs/remediation_spine_consistency_v8.md"),
        ),
        (
            "docs/docs_checks.md",
            "docs/artifact_schema_snapshots.md",
            docs_checks.contains("docs/artifact_schema_snapshots.md"),
        ),
    ]
    .into_iter()
    .filter_map(|(file, needle, present)| {
        (!present).then_some(format!("{file} missing `{needle}`"))
    })
    .collect::<Vec<_>>();

    if !missing.is_empty() {
        return Ok(DocsLintCheck {
            name: "v8_docs_consistency".to_string(),
            status: DocsLintStatus::Fail,
            detail: format!("v8 docs linkage mismatch: {}", missing.join("; ")),
            remediation: Some(
                "link v8 docs from portability/docs checks and keep docs references in sync"
                    .to_string(),
            ),
        });
    }

    Ok(DocsLintCheck {
        name: "v8_docs_consistency".to_string(),
        status: DocsLintStatus::Pass,
        detail: "v8 docs are present and linked from portability/docs checks".to_string(),
        remediation: None,
    })
}

fn cargo_metadata_package_names(repo_root: &Path) -> Result<BTreeSet<String>, OpsError> {
    let output = Command::new("cargo")
        .arg("metadata")
        .arg("--no-deps")
        .arg("--format-version")
        .arg("1")
        .current_dir(repo_root)
        .output()?;
    if !output.status.success() {
        return Err(OpsError::Invalid(format!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }

    #[derive(Deserialize)]
    struct Metadata {
        packages: Vec<Package>,
    }

    #[derive(Deserialize)]
    struct Package {
        name: String,
    }

    let metadata: Metadata = serde_json::from_slice(&output.stdout)?;
    let mut names = metadata
        .packages
        .into_iter()
        .map(|p| p.name)
        .collect::<BTreeSet<_>>();
    names.insert("ucf-ops".to_string());
    Ok(names)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PromptIdParseResult {
    ids: Vec<u32>,
    invalid_header_line: Option<usize>,
}

fn parse_prompt_ids(content: &str) -> Result<PromptIdParseResult, OpsError> {
    let mut ids = Vec::new();
    let mut invalid_header_line = None;

    for (idx, raw) in content.lines().enumerate() {
        let line = raw.trim();
        let line_no = idx + 1;

        if line.starts_with("#") && line.contains("PROMPT") {
            if let Some(id) = parse_prompt_id_from_heading(line) {
                ids.push(id);
            } else if invalid_header_line.is_none() {
                invalid_header_line = Some(line_no);
            }
            continue;
        }

        if line.starts_with('|') {
            if let Some(id) = parse_prompt_id_from_table_row(line) {
                ids.push(id);
            }
        }
    }

    if invalid_header_line.is_some() {
        return Ok(PromptIdParseResult {
            ids,
            invalid_header_line,
        });
    }

    Ok(PromptIdParseResult {
        ids,
        invalid_header_line: None,
    })
}

fn parse_prompt_id_from_heading(line: &str) -> Option<u32> {
    let prompt_pos = line.find("PROMPT ")?;
    let tail = &line[prompt_pos + "PROMPT ".len()..];
    let mut number = String::new();
    for ch in tail.chars() {
        if ch.is_ascii_digit() {
            number.push(ch);
        } else {
            break;
        }
    }
    if number.is_empty() {
        return None;
    }
    let id = number.parse::<u32>().ok()?;
    let after = &tail[number.len()..];
    if !after.trim_start().starts_with('—') {
        return None;
    }
    Some(id)
}

fn parse_prompt_id_from_table_row(line: &str) -> Option<u32> {
    let mut cells = line.split('|').map(str::trim);
    let _leading = cells.next()?;
    let first = cells.next()?;
    if first == "ID" || first.starts_with("---") {
        return None;
    }
    first.parse::<u32>().ok()
}

fn parse_module_map_keys(content: &str) -> Vec<String> {
    let mut keys = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("- **") {
            continue;
        }
        let Some(rest) = trimmed.strip_prefix("- **") else {
            continue;
        };
        let Some(end) = rest.find("**:") else {
            continue;
        };
        keys.push(rest[..end].to_string());
    }
    keys
}

fn looks_like_crate_name(key: &str) -> bool {
    key.contains('-')
        && key
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

fn first_diff_line(a: &str, b: &str) -> usize {
    let mut line = 1;
    let mut a_it = a.lines();
    let mut b_it = b.lines();
    loop {
        match (a_it.next(), b_it.next()) {
            (Some(al), Some(bl)) if al == bl => line += 1,
            (Some(_), Some(_)) | (None, Some(_)) | (Some(_), None) => return line,
            (None, None) => return line,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        first_diff_line, hardware_neutral_docs_check, parse_module_map_keys, parse_prompt_ids,
        remediation_registry_doc_check, v3_docs_consistency_check, v4_docs_consistency_check,
        v5_docs_consistency_check, v6_docs_consistency_check, v7_docs_consistency_check,
        v8_docs_consistency_check, DocsLintArgs, DocsLintMode, DocsLintStatus,
    };
    use std::path::PathBuf;

    #[test]
    fn prompt_parser_accepts_table_ids() {
        let content = "| ID | Title |\n|---:|---|\n| 1 | One |\n| 2 | Two |\n| 3 | Three |\n";
        let parsed = parse_prompt_ids(content).expect("parse");
        assert_eq!(parsed.ids, vec![1, 2, 3]);
        assert_eq!(parsed.invalid_header_line, None);
    }

    #[test]
    fn prompt_parser_flags_invalid_heading() {
        let content = "## PROMPT 12 - Missing em dash";
        let parsed = parse_prompt_ids(content).expect("parse");
        assert_eq!(parsed.invalid_header_line, Some(1));
    }

    #[test]
    fn module_map_parser_reads_keys() {
        let content = "- **ucf-ops**: 1\n- **docs**: 2\n";
        assert_eq!(parse_module_map_keys(content), vec!["ucf-ops", "docs"]);
    }

    #[test]
    fn diff_line_is_stable() {
        let old = "a\nb\nc\n";
        let new = "a\nb\nd\n";
        assert_eq!(first_diff_line(old, new), 3);
    }

    #[test]
    fn hardware_neutral_check_warns_in_history_section() {
        let dir = tempfile::tempdir().expect("tmp");
        let docs = dir.path().join("docs");
        std::fs::create_dir_all(&docs).expect("mkdir");
        std::fs::write(
            docs.join("prompt_series_index.md"),
            "# Prompt\n## History\nLegacy NUC mention\n",
        )
        .expect("write");
        std::fs::write(docs.join("prompt_rulebook.md"), "# Rules\n").expect("write");
        std::fs::write(docs.join("deploy_portable.md"), "# Deploy\nRPi adapter\n").expect("write");
        std::fs::write(docs.join("models_eligibility_v3.md"), "# v3\n").expect("write");
        std::fs::write(docs.join("strict_mode_v3.md"), "# v3\n").expect("write");
        std::fs::write(docs.join("operator_report_v3.md"), "# v3\n").expect("write");
        std::fs::write(docs.join("backend_evidence_snapshot_v4.md"), "# v4\n").expect("write");
        std::fs::write(docs.join("operator_signoff_v4.md"), "# v4\n").expect("write");
        std::fs::write(docs.join("remediation_codes_v1.md"), "# v4\n").expect("write");
        std::fs::write(docs.join("artifact_schema_snapshots.md"), "# v4\n").expect("write");
        std::fs::write(docs.join("active_review_snapshot_v5.md"), "# v5\n").expect("write");
        std::fs::write(docs.join("sae_burn_resolution_v5.md"), "# v5\n").expect("write");
        std::fs::write(docs.join("repro_pack.md"), "# v5\n").expect("write");
        std::fs::write(docs.join("bug_report_kit.md"), "# v5\n").expect("write");
        std::fs::write(docs.join("remediation_consistency_v5.md"), "# v5\n").expect("write");
        std::fs::write(docs.join("governance_primary_surfaces_v6.md"), "# v6\n").expect("write");
        std::fs::write(docs.join("supported_set_apply_v6.md"), "# v6\n").expect("write");
        std::fs::write(docs.join("applied_supported_scope_v6.md"), "# v6\n").expect("write");
        std::fs::write(docs.join("export_normalization_v6.md"), "# v6\n").expect("write");
        std::fs::write(docs.join("interop_consistency_v6.md"), "# v6\n").expect("write");
        std::fs::write(docs.join("module_map.md"), "- **ucf-ops**: x\n").expect("write");
        std::fs::write(docs.join("spec_snapshot.md"), "# x\n").expect("write");

        let check = hardware_neutral_docs_check(&DocsLintArgs {
            repo_root: dir.path().to_path_buf(),
            policy_pack: PathBuf::from("policies/packs/base_v1"),
            overlay_pack: None,
            spec_snapshot: docs.join("spec_snapshot.md"),
            prompt_index: docs.join("prompt_series_index.md"),
            module_map: docs.join("module_map.md"),
            deploy_doc: docs.join("deploy_portable.md"),
            artifact_schema_snapshot_dir: docs.join("artifact_schema_snapshots"),
            mode: DocsLintMode::Strict,
        })
        .expect("check");
        assert_eq!(check.status, DocsLintStatus::Warn);
    }

    #[test]
    fn hardware_neutral_check_fails_in_core_docs() {
        let dir = tempfile::tempdir().expect("tmp");
        let docs = dir.path().join("docs");
        std::fs::create_dir_all(&docs).expect("mkdir");
        std::fs::write(
            docs.join("prompt_series_index.md"),
            "# Prompt\nNUC target\n",
        )
        .expect("write");
        std::fs::write(docs.join("prompt_rulebook.md"), "# Rules\n").expect("write");
        std::fs::write(docs.join("deploy_portable.md"), "# Deploy\n").expect("write");
        std::fs::write(docs.join("models_eligibility_v3.md"), "# v3\n").expect("write");
        std::fs::write(docs.join("strict_mode_v3.md"), "# v3\n").expect("write");
        std::fs::write(docs.join("operator_report_v3.md"), "# v3\n").expect("write");
        std::fs::write(docs.join("backend_evidence_snapshot_v4.md"), "# v4\n").expect("write");
        std::fs::write(docs.join("operator_signoff_v4.md"), "# v4\n").expect("write");
        std::fs::write(docs.join("remediation_codes_v1.md"), "NUC\n").expect("write");
        std::fs::write(docs.join("artifact_schema_snapshots.md"), "# v4\n").expect("write");
        std::fs::write(docs.join("active_review_snapshot_v5.md"), "# v5\n").expect("write");
        std::fs::write(docs.join("sae_burn_resolution_v5.md"), "# v5\n").expect("write");
        std::fs::write(docs.join("repro_pack.md"), "# v5\n").expect("write");
        std::fs::write(docs.join("bug_report_kit.md"), "# v5\n").expect("write");
        std::fs::write(docs.join("remediation_consistency_v5.md"), "# v5\n").expect("write");
        std::fs::write(docs.join("governance_primary_surfaces_v6.md"), "# v6\n").expect("write");
        std::fs::write(docs.join("supported_set_apply_v6.md"), "# v6\n").expect("write");
        std::fs::write(docs.join("applied_supported_scope_v6.md"), "# v6\n").expect("write");
        std::fs::write(docs.join("export_normalization_v6.md"), "# v6\n").expect("write");
        std::fs::write(docs.join("interop_consistency_v6.md"), "# v6\n").expect("write");
        std::fs::write(docs.join("module_map.md"), "- **ucf-ops**: x\n").expect("write");
        std::fs::write(docs.join("spec_snapshot.md"), "# x\n").expect("write");

        let check = hardware_neutral_docs_check(&DocsLintArgs {
            repo_root: dir.path().to_path_buf(),
            policy_pack: PathBuf::from("policies/packs/base_v1"),
            overlay_pack: None,
            spec_snapshot: docs.join("spec_snapshot.md"),
            prompt_index: docs.join("prompt_series_index.md"),
            module_map: docs.join("module_map.md"),
            deploy_doc: docs.join("deploy_portable.md"),
            artifact_schema_snapshot_dir: docs.join("artifact_schema_snapshots"),
            mode: DocsLintMode::Strict,
        })
        .expect("check");
        assert_eq!(check.status, DocsLintStatus::Fail);
    }

    #[test]
    fn v3_docs_consistency_requires_links() {
        let dir = tempfile::tempdir().expect("tmp");
        let docs = dir.path().join("docs");
        std::fs::create_dir_all(&docs).expect("mkdir");
        std::fs::write(docs.join("prompt_series_index.md"), "| 207 | x |\n").expect("write");
        std::fs::write(docs.join("prompt_rulebook.md"), "# Rules\n").expect("write");
        std::fs::write(docs.join("deploy_portable.md"), "# Deploy\n").expect("write");
        std::fs::write(docs.join("module_map.md"), "- **ucf-ops**: x\n").expect("write");
        std::fs::write(docs.join("spec_snapshot.md"), "# x\n").expect("write");
        std::fs::write(
            docs.join("portability_gate.md"),
            "models_eligibility_v3.md strict_mode_v3.md operator_report_v3.md\n",
        )
        .expect("write");
        std::fs::write(docs.join("strict_mode.md"), "strict_mode_v3.md\n").expect("write");
        std::fs::write(docs.join("series_state_snapshot.md"), "| 207 | x |\n").expect("write");
        std::fs::write(docs.join("models_eligibility_v3.md"), "# x\n").expect("write");
        std::fs::write(docs.join("strict_mode_v3.md"), "# x\n").expect("write");
        std::fs::write(docs.join("operator_report_v3.md"), "# x\n").expect("write");

        let check = v3_docs_consistency_check(&DocsLintArgs {
            repo_root: dir.path().to_path_buf(),
            policy_pack: PathBuf::from("policies/packs/base_v1"),
            overlay_pack: None,
            spec_snapshot: docs.join("spec_snapshot.md"),
            prompt_index: docs.join("prompt_series_index.md"),
            module_map: docs.join("module_map.md"),
            deploy_doc: docs.join("deploy_portable.md"),
            artifact_schema_snapshot_dir: docs.join("artifact_schema_snapshots"),
            mode: DocsLintMode::Strict,
        })
        .expect("check");
        assert_eq!(check.status, DocsLintStatus::Pass);
    }

    #[test]
    fn v4_docs_consistency_requires_links() {
        let dir = tempfile::tempdir().expect("tmp");
        let docs = dir.path().join("docs");
        std::fs::create_dir_all(&docs).expect("mkdir");
        std::fs::write(docs.join("prompt_series_index.md"), "| 216 | x |\n").expect("write");
        std::fs::write(docs.join("prompt_rulebook.md"), "# Rules\n").expect("write");
        std::fs::write(docs.join("deploy_portable.md"), "# Deploy\n").expect("write");
        std::fs::write(docs.join("module_map.md"), "- **ucf-ops**: x\n").expect("write");
        std::fs::write(docs.join("spec_snapshot.md"), "# x\n").expect("write");
        std::fs::write(
            docs.join("portability_gate.md"),
            "backend_evidence_snapshot_v4.md operator_signoff_v4.md remediation_codes_v1.md artifact_schema_snapshots.md\n",
        )
        .expect("write");
        std::fs::write(
            docs.join("docs_checks.md"),
            "docs/remediation_codes_v1.md docs/artifact_schema_snapshots.md\n",
        )
        .expect("write");
        std::fs::write(docs.join("series_state_snapshot.md"), "| 216 | x |\n").expect("write");
        std::fs::write(docs.join("models_eligibility_v3.md"), "# x\n").expect("write");
        std::fs::write(docs.join("strict_mode_v3.md"), "# x\n").expect("write");
        std::fs::write(docs.join("operator_report_v3.md"), "# x\n").expect("write");
        std::fs::write(docs.join("backend_evidence_snapshot_v4.md"), "# x\n").expect("write");
        std::fs::write(docs.join("operator_signoff_v4.md"), "# x\n").expect("write");
        std::fs::write(docs.join("remediation_codes_v1.md"), "# x\n").expect("write");
        std::fs::write(docs.join("artifact_schema_snapshots.md"), "# x\n").expect("write");

        let check = v4_docs_consistency_check(&DocsLintArgs {
            repo_root: dir.path().to_path_buf(),
            policy_pack: PathBuf::from("policies/packs/base_v1"),
            overlay_pack: None,
            spec_snapshot: docs.join("spec_snapshot.md"),
            prompt_index: docs.join("prompt_series_index.md"),
            module_map: docs.join("module_map.md"),
            deploy_doc: docs.join("deploy_portable.md"),
            artifact_schema_snapshot_dir: docs.join("artifact_schema_snapshots"),
            mode: DocsLintMode::Strict,
        })
        .expect("check");
        assert_eq!(check.status, DocsLintStatus::Pass);
    }

    #[test]
    fn v5_docs_consistency_requires_links() {
        let dir = tempfile::tempdir().expect("tmp");
        let docs = dir.path().join("docs");
        std::fs::create_dir_all(&docs).expect("mkdir");
        std::fs::write(docs.join("prompt_series_index.md"), "| 226 | x |\n").expect("write");
        std::fs::write(docs.join("prompt_rulebook.md"), "# Rules\n").expect("write");
        std::fs::write(docs.join("deploy_portable.md"), "# Deploy\n").expect("write");
        std::fs::write(docs.join("module_map.md"), "- **ucf-ops**: x\n").expect("write");
        std::fs::write(docs.join("spec_snapshot.md"), "# x\n").expect("write");
        std::fs::write(
            docs.join("portability_gate.md"),
            "active_review_snapshot_v5.md sae_burn_resolution_v5.md repro_pack.md bug_report_kit.md remediation_consistency_v5.md\n",
        )
        .expect("write");
        std::fs::write(
            docs.join("docs_checks.md"),
            "docs/active_review_snapshot_v5.md docs/sae_burn_resolution_v5.md docs/repro_pack.md docs/bug_report_kit.md docs/remediation_consistency_v5.md\n",
        )
        .expect("write");
        std::fs::write(docs.join("series_state_snapshot.md"), "| 216 | x |\n").expect("write");
        std::fs::write(docs.join("models_eligibility_v3.md"), "# x\n").expect("write");
        std::fs::write(docs.join("strict_mode_v3.md"), "# x\n").expect("write");
        std::fs::write(docs.join("operator_report_v3.md"), "# x\n").expect("write");
        std::fs::write(docs.join("backend_evidence_snapshot_v4.md"), "# x\n").expect("write");
        std::fs::write(docs.join("operator_signoff_v4.md"), "# x\n").expect("write");
        std::fs::write(docs.join("remediation_codes_v1.md"), "# x\n").expect("write");
        std::fs::write(docs.join("artifact_schema_snapshots.md"), "# x\n").expect("write");
        std::fs::write(docs.join("active_review_snapshot_v5.md"), "# x\n").expect("write");
        std::fs::write(docs.join("sae_burn_resolution_v5.md"), "# x\n").expect("write");
        std::fs::write(docs.join("repro_pack.md"), "# x\n").expect("write");
        std::fs::write(docs.join("bug_report_kit.md"), "# x\n").expect("write");
        std::fs::write(docs.join("remediation_consistency_v5.md"), "# x\n").expect("write");

        let check = v5_docs_consistency_check(&DocsLintArgs {
            repo_root: dir.path().to_path_buf(),
            policy_pack: PathBuf::from("policies/packs/base_v1"),
            overlay_pack: None,
            spec_snapshot: docs.join("spec_snapshot.md"),
            prompt_index: docs.join("prompt_series_index.md"),
            module_map: docs.join("module_map.md"),
            deploy_doc: docs.join("deploy_portable.md"),
            artifact_schema_snapshot_dir: docs.join("artifact_schema_snapshots"),
            mode: DocsLintMode::Strict,
        })
        .expect("check");
        assert_eq!(check.status, DocsLintStatus::Pass);
    }

    #[test]
    fn v6_docs_consistency_requires_links() {
        let dir = tempfile::tempdir().expect("tmp");
        let docs = dir.path().join("docs");
        std::fs::create_dir_all(&docs).expect("mkdir");
        std::fs::write(docs.join("prompt_series_index.md"), "| 236 | x |\n").expect("write");
        std::fs::write(docs.join("prompt_rulebook.md"), "# Rules\n").expect("write");
        std::fs::write(docs.join("deploy_portable.md"), "# Deploy\n").expect("write");
        std::fs::write(docs.join("module_map.md"), "- **ucf-ops**: x\n").expect("write");
        std::fs::write(docs.join("spec_snapshot.md"), "# x\n").expect("write");
        std::fs::write(
            docs.join("portability_gate.md"),
            "governance_primary_surfaces_v6.md supported_set_apply_v6.md applied_supported_scope_v6.md export_normalization_v6.md interop_consistency_v6.md artifact_schema_snapshots.md\n",
        )
        .expect("write");
        std::fs::write(
            docs.join("docs_checks.md"),
            "docs/governance_primary_surfaces_v6.md docs/supported_set_apply_v6.md docs/applied_supported_scope_v6.md docs/export_normalization_v6.md docs/interop_consistency_v6.md docs/artifact_schema_snapshots.md\n",
        )
        .expect("write");
        std::fs::write(docs.join("series_state_snapshot.md"), "| 236 | x |\n").expect("write");
        std::fs::write(docs.join("models_eligibility_v3.md"), "# x\n").expect("write");
        std::fs::write(docs.join("strict_mode_v3.md"), "# x\n").expect("write");
        std::fs::write(docs.join("operator_report_v3.md"), "# x\n").expect("write");
        std::fs::write(docs.join("backend_evidence_snapshot_v4.md"), "# x\n").expect("write");
        std::fs::write(docs.join("operator_signoff_v4.md"), "# x\n").expect("write");
        std::fs::write(docs.join("remediation_codes_v1.md"), "# x\n").expect("write");
        std::fs::write(docs.join("artifact_schema_snapshots.md"), "# x\n").expect("write");
        std::fs::write(docs.join("active_review_snapshot_v5.md"), "# x\n").expect("write");
        std::fs::write(docs.join("sae_burn_resolution_v5.md"), "# x\n").expect("write");
        std::fs::write(docs.join("repro_pack.md"), "# x\n").expect("write");
        std::fs::write(docs.join("bug_report_kit.md"), "# x\n").expect("write");
        std::fs::write(docs.join("remediation_consistency_v5.md"), "# x\n").expect("write");
        std::fs::write(docs.join("governance_primary_surfaces_v6.md"), "# x\n").expect("write");
        std::fs::write(docs.join("supported_set_apply_v6.md"), "# x\n").expect("write");
        std::fs::write(docs.join("applied_supported_scope_v6.md"), "# x\n").expect("write");
        std::fs::write(docs.join("export_normalization_v6.md"), "# x\n").expect("write");
        std::fs::write(docs.join("interop_consistency_v6.md"), "# x\n").expect("write");

        let check = v6_docs_consistency_check(&DocsLintArgs {
            repo_root: dir.path().to_path_buf(),
            policy_pack: PathBuf::from("policies/packs/base_v1"),
            overlay_pack: None,
            spec_snapshot: docs.join("spec_snapshot.md"),
            prompt_index: docs.join("prompt_series_index.md"),
            module_map: docs.join("module_map.md"),
            deploy_doc: docs.join("deploy_portable.md"),
            artifact_schema_snapshot_dir: docs.join("artifact_schema_snapshots"),
            mode: DocsLintMode::Strict,
        })
        .expect("check");
        assert_eq!(check.status, DocsLintStatus::Pass);
    }

    #[test]
    fn v6_docs_consistency_missing_link_fails() {
        let dir = tempfile::tempdir().expect("tmp");
        let docs = dir.path().join("docs");
        std::fs::create_dir_all(&docs).expect("mkdir");
        std::fs::write(docs.join("prompt_series_index.md"), "| 236 | x |\n").expect("write");
        std::fs::write(docs.join("prompt_rulebook.md"), "# Rules\n").expect("write");
        std::fs::write(docs.join("deploy_portable.md"), "# Deploy\n").expect("write");
        std::fs::write(docs.join("module_map.md"), "- **ucf-ops**: x\n").expect("write");
        std::fs::write(docs.join("spec_snapshot.md"), "# x\n").expect("write");
        std::fs::write(
            docs.join("portability_gate.md"),
            "governance_primary_surfaces_v6.md\n",
        )
        .expect("write");
        std::fs::write(
            docs.join("docs_checks.md"),
            "docs/governance_primary_surfaces_v6.md\n",
        )
        .expect("write");
        std::fs::write(docs.join("series_state_snapshot.md"), "| 236 | x |\n").expect("write");
        for name in [
            "models_eligibility_v3.md",
            "strict_mode_v3.md",
            "operator_report_v3.md",
            "backend_evidence_snapshot_v4.md",
            "operator_signoff_v4.md",
            "remediation_codes_v1.md",
            "artifact_schema_snapshots.md",
            "active_review_snapshot_v5.md",
            "sae_burn_resolution_v5.md",
            "repro_pack.md",
            "bug_report_kit.md",
            "remediation_consistency_v5.md",
            "governance_primary_surfaces_v6.md",
            "supported_set_apply_v6.md",
            "applied_supported_scope_v6.md",
            "export_normalization_v6.md",
            "interop_consistency_v6.md",
        ] {
            std::fs::write(docs.join(name), "# x\n").expect("write");
        }

        let check = v6_docs_consistency_check(&DocsLintArgs {
            repo_root: dir.path().to_path_buf(),
            policy_pack: PathBuf::from("policies/packs/base_v1"),
            overlay_pack: None,
            spec_snapshot: docs.join("spec_snapshot.md"),
            prompt_index: docs.join("prompt_series_index.md"),
            module_map: docs.join("module_map.md"),
            deploy_doc: docs.join("deploy_portable.md"),
            artifact_schema_snapshot_dir: docs.join("artifact_schema_snapshots"),
            mode: DocsLintMode::Strict,
        })
        .expect("check");
        assert_eq!(check.status, DocsLintStatus::Fail);
    }

    #[test]
    fn remediation_registry_doc_check_detects_drift() {
        let dir = tempfile::tempdir().expect("tmp");
        let docs = dir.path().join("docs");
        std::fs::create_dir_all(&docs).expect("mkdir");
        std::fs::write(docs.join("remediation_codes_v1.md"), "# stale\n").expect("write");
        let check = remediation_registry_doc_check(&DocsLintArgs {
            repo_root: dir.path().to_path_buf(),
            policy_pack: PathBuf::from("policies/packs/base_v1"),
            overlay_pack: None,
            spec_snapshot: docs.join("spec_snapshot.md"),
            prompt_index: docs.join("prompt_series_index.md"),
            module_map: docs.join("module_map.md"),
            deploy_doc: docs.join("deploy_portable.md"),
            artifact_schema_snapshot_dir: docs.join("artifact_schema_snapshots"),
            mode: DocsLintMode::Strict,
        })
        .expect("check");
        assert_eq!(check.status, DocsLintStatus::Fail);
    }
    #[test]
    fn v7_docs_consistency_requires_links() {
        let dir = tempfile::tempdir().expect("tmp");
        let docs = dir.path().join("docs");
        std::fs::create_dir_all(&docs).expect("mkdir");
        std::fs::write(
            docs.join("prompt_series_index.md"),
            "| 245 | x |
",
        )
        .expect("write");
        std::fs::write(
            docs.join("prompt_rulebook.md"),
            "# Rules
",
        )
        .expect("write");
        std::fs::write(
            docs.join("deploy_portable.md"),
            "# Deploy
",
        )
        .expect("write");
        std::fs::write(
            docs.join("module_map.md"),
            "- **ucf-ops**: x
",
        )
        .expect("write");
        std::fs::write(
            docs.join("spec_snapshot.md"),
            "# x
",
        )
        .expect("write");
        std::fs::write(
            docs.join("portability_gate.md"),
            "applied_scope_authority_v7.md supported_scope_reevaluation_v7.md reviewability_truth_v7.md export_roundtrip_v7.md remediation_interop_consistency_v7.md artifact_schema_snapshots.md
",
        )
        .expect("write");
        std::fs::write(
            docs.join("docs_checks.md"),
            "docs/applied_scope_authority_v7.md docs/supported_scope_reevaluation_v7.md docs/reviewability_truth_v7.md docs/export_roundtrip_v7.md docs/remediation_interop_consistency_v7.md docs/artifact_schema_snapshots.md
",
        )
        .expect("write");
        for name in [
            "models_eligibility_v3.md",
            "strict_mode_v3.md",
            "operator_report_v3.md",
            "backend_evidence_snapshot_v4.md",
            "operator_signoff_v4.md",
            "remediation_codes_v1.md",
            "artifact_schema_snapshots.md",
            "active_review_snapshot_v5.md",
            "sae_burn_resolution_v5.md",
            "repro_pack.md",
            "bug_report_kit.md",
            "remediation_consistency_v5.md",
            "governance_primary_surfaces_v6.md",
            "supported_set_apply_v6.md",
            "applied_supported_scope_v6.md",
            "export_normalization_v6.md",
            "interop_consistency_v6.md",
            "applied_scope_authority_v7.md",
            "supported_scope_reevaluation_v7.md",
            "reviewability_truth_v7.md",
            "export_roundtrip_v7.md",
            "remediation_interop_consistency_v7.md",
        ] {
            std::fs::write(
                docs.join(name),
                "# x
",
            )
            .expect("write");
        }

        let check = v7_docs_consistency_check(&DocsLintArgs {
            repo_root: dir.path().to_path_buf(),
            policy_pack: PathBuf::from("policies/packs/base_v1"),
            overlay_pack: None,
            spec_snapshot: docs.join("spec_snapshot.md"),
            prompt_index: docs.join("prompt_series_index.md"),
            module_map: docs.join("module_map.md"),
            deploy_doc: docs.join("deploy_portable.md"),
            artifact_schema_snapshot_dir: docs.join("artifact_schema_snapshots"),
            mode: DocsLintMode::Strict,
        })
        .expect("check");
        assert_eq!(check.status, DocsLintStatus::Pass);
    }

    #[test]
    fn v7_docs_consistency_missing_link_fails() {
        let dir = tempfile::tempdir().expect("tmp");
        let docs = dir.path().join("docs");
        std::fs::create_dir_all(&docs).expect("mkdir");
        std::fs::write(
            docs.join("prompt_series_index.md"),
            "| 245 | x |
",
        )
        .expect("write");
        std::fs::write(
            docs.join("prompt_rulebook.md"),
            "# Rules
",
        )
        .expect("write");
        std::fs::write(
            docs.join("deploy_portable.md"),
            "# Deploy
",
        )
        .expect("write");
        std::fs::write(
            docs.join("module_map.md"),
            "- **ucf-ops**: x
",
        )
        .expect("write");
        std::fs::write(
            docs.join("spec_snapshot.md"),
            "# x
",
        )
        .expect("write");
        std::fs::write(
            docs.join("portability_gate.md"),
            "applied_scope_authority_v7.md
",
        )
        .expect("write");
        std::fs::write(
            docs.join("docs_checks.md"),
            "docs/applied_scope_authority_v7.md
",
        )
        .expect("write");
        for name in [
            "models_eligibility_v3.md",
            "strict_mode_v3.md",
            "operator_report_v3.md",
            "backend_evidence_snapshot_v4.md",
            "operator_signoff_v4.md",
            "remediation_codes_v1.md",
            "artifact_schema_snapshots.md",
            "active_review_snapshot_v5.md",
            "sae_burn_resolution_v5.md",
            "repro_pack.md",
            "bug_report_kit.md",
            "remediation_consistency_v5.md",
            "governance_primary_surfaces_v6.md",
            "supported_set_apply_v6.md",
            "applied_supported_scope_v6.md",
            "export_normalization_v6.md",
            "interop_consistency_v6.md",
            "applied_scope_authority_v7.md",
            "supported_scope_reevaluation_v7.md",
            "reviewability_truth_v7.md",
            "export_roundtrip_v7.md",
            "remediation_interop_consistency_v7.md",
        ] {
            std::fs::write(
                docs.join(name),
                "# x
",
            )
            .expect("write");
        }

        let check = v7_docs_consistency_check(&DocsLintArgs {
            repo_root: dir.path().to_path_buf(),
            policy_pack: PathBuf::from("policies/packs/base_v1"),
            overlay_pack: None,
            spec_snapshot: docs.join("spec_snapshot.md"),
            prompt_index: docs.join("prompt_series_index.md"),
            module_map: docs.join("module_map.md"),
            deploy_doc: docs.join("deploy_portable.md"),
            artifact_schema_snapshot_dir: docs.join("artifact_schema_snapshots"),
            mode: DocsLintMode::Strict,
        })
        .expect("check");
        assert_eq!(check.status, DocsLintStatus::Fail);
    }
    #[test]
    fn v8_docs_consistency_pass() {
        let dir = tempfile::tempdir().expect("tmp");
        let docs = dir.path().join("docs");
        std::fs::create_dir_all(&docs).expect("mkdir");
        std::fs::write(
            docs.join("prompt_series_index.md"),
            "| 256 | x |
",
        )
        .expect("write");
        std::fs::write(
            docs.join("prompt_rulebook.md"),
            "# Rules
",
        )
        .expect("write");
        std::fs::write(
            docs.join("deploy_portable.md"),
            "# Deploy
",
        )
        .expect("write");
        std::fs::write(
            docs.join("module_map.md"),
            "- **ucf-ops**: x
",
        )
        .expect("write");
        std::fs::write(
            docs.join("spec_snapshot.md"),
            "# x
",
        )
        .expect("write");
        std::fs::write(
            docs.join("portability_gate.md"),
            "canonical_governance_entry_v8.md supported_scope_execution_v8.md readiness_spine_v8.md bundle_spine_v8.md remediation_spine_consistency_v8.md artifact_schema_snapshots.md
",
        )
        .expect("write");
        std::fs::write(
            docs.join("docs_checks.md"),
            "docs/canonical_governance_entry_v8.md docs/supported_scope_execution_v8.md docs/readiness_spine_v8.md docs/bundle_spine_v8.md docs/remediation_spine_consistency_v8.md docs/artifact_schema_snapshots.md
",
        )
        .expect("write");
        for name in [
            "models_eligibility_v3.md",
            "strict_mode_v3.md",
            "operator_report_v3.md",
            "backend_evidence_snapshot_v4.md",
            "operator_signoff_v4.md",
            "remediation_codes_v1.md",
            "artifact_schema_snapshots.md",
            "active_review_snapshot_v5.md",
            "sae_burn_resolution_v5.md",
            "repro_pack.md",
            "bug_report_kit.md",
            "remediation_consistency_v5.md",
            "governance_primary_surfaces_v6.md",
            "supported_set_apply_v6.md",
            "applied_supported_scope_v6.md",
            "export_normalization_v6.md",
            "interop_consistency_v6.md",
            "applied_scope_authority_v7.md",
            "supported_scope_reevaluation_v7.md",
            "reviewability_truth_v7.md",
            "export_roundtrip_v7.md",
            "remediation_interop_consistency_v7.md",
            "canonical_governance_entry_v8.md",
            "supported_scope_execution_v8.md",
            "readiness_spine_v8.md",
            "bundle_spine_v8.md",
            "remediation_spine_consistency_v8.md",
        ] {
            std::fs::write(
                docs.join(name),
                "# x
",
            )
            .expect("write");
        }

        let check = v8_docs_consistency_check(&DocsLintArgs {
            repo_root: dir.path().to_path_buf(),
            policy_pack: PathBuf::from("policies/packs/base_v1"),
            overlay_pack: None,
            spec_snapshot: docs.join("spec_snapshot.md"),
            prompt_index: docs.join("prompt_series_index.md"),
            module_map: docs.join("module_map.md"),
            deploy_doc: docs.join("deploy_portable.md"),
            artifact_schema_snapshot_dir: docs.join("artifact_schema_snapshots"),
            mode: DocsLintMode::Strict,
        })
        .expect("check");
        assert_eq!(check.status, DocsLintStatus::Pass);
    }

    #[test]
    fn v8_docs_consistency_missing_link_fails() {
        let dir = tempfile::tempdir().expect("tmp");
        let docs = dir.path().join("docs");
        std::fs::create_dir_all(&docs).expect("mkdir");
        std::fs::write(
            docs.join("prompt_series_index.md"),
            "| 256 | x |
",
        )
        .expect("write");
        std::fs::write(
            docs.join("prompt_rulebook.md"),
            "# Rules
",
        )
        .expect("write");
        std::fs::write(
            docs.join("deploy_portable.md"),
            "# Deploy
",
        )
        .expect("write");
        std::fs::write(
            docs.join("module_map.md"),
            "- **ucf-ops**: x
",
        )
        .expect("write");
        std::fs::write(
            docs.join("spec_snapshot.md"),
            "# x
",
        )
        .expect("write");
        std::fs::write(
            docs.join("portability_gate.md"),
            "canonical_governance_entry_v8.md
",
        )
        .expect("write");
        std::fs::write(
            docs.join("docs_checks.md"),
            "docs/canonical_governance_entry_v8.md
",
        )
        .expect("write");
        for name in [
            "models_eligibility_v3.md",
            "strict_mode_v3.md",
            "operator_report_v3.md",
            "backend_evidence_snapshot_v4.md",
            "operator_signoff_v4.md",
            "remediation_codes_v1.md",
            "artifact_schema_snapshots.md",
            "active_review_snapshot_v5.md",
            "sae_burn_resolution_v5.md",
            "repro_pack.md",
            "bug_report_kit.md",
            "remediation_consistency_v5.md",
            "governance_primary_surfaces_v6.md",
            "supported_set_apply_v6.md",
            "applied_supported_scope_v6.md",
            "export_normalization_v6.md",
            "interop_consistency_v6.md",
            "applied_scope_authority_v7.md",
            "supported_scope_reevaluation_v7.md",
            "reviewability_truth_v7.md",
            "export_roundtrip_v7.md",
            "remediation_interop_consistency_v7.md",
            "canonical_governance_entry_v8.md",
            "supported_scope_execution_v8.md",
            "readiness_spine_v8.md",
            "bundle_spine_v8.md",
            "remediation_spine_consistency_v8.md",
        ] {
            std::fs::write(
                docs.join(name),
                "# x
",
            )
            .expect("write");
        }

        let check = v8_docs_consistency_check(&DocsLintArgs {
            repo_root: dir.path().to_path_buf(),
            policy_pack: PathBuf::from("policies/packs/base_v1"),
            overlay_pack: None,
            spec_snapshot: docs.join("spec_snapshot.md"),
            prompt_index: docs.join("prompt_series_index.md"),
            module_map: docs.join("module_map.md"),
            deploy_doc: docs.join("deploy_portable.md"),
            artifact_schema_snapshot_dir: docs.join("artifact_schema_snapshots"),
            mode: DocsLintMode::Strict,
        })
        .expect("check");
        assert_eq!(check.status, DocsLintStatus::Fail);
    }
}
