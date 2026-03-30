use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    exports_normalize_check, governance_surfaces_check, interop_consistency::CrossSurfaceKindV1,
    interop_consistency_matrix, models_applied_scope_check, operator_review_packet,
    operator_signoff, prefix_hex, sha256_hex, AppliedScopeCheckReportV1,
    ExportNormalizeCheckReportV1, GovernanceSurfacesCheckReportV1,
    InteropConsistencyMatrixReportV1, InteropOverallStatusV1, OperatorReviewPacketArgs,
    OperatorReviewPacketV1, OperatorSignoffArgs, OperatorSignoffDecisionV1, OpsError,
    SignoffDecisionStateV1,
};

const CODE_CAP: usize = 16;
const DIGEST_PREFIX_LEN: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum OperatorWorkflowStageV2 {
    WorkflowBlocked,
    WorkflowReviewReady,
    WorkflowExportReady,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OperatorWorkflowExportTargetsV1 {
    pub repro_ready: bool,
    pub bugkit_ready: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OperatorWorkflowChainV1 {
    pub schema_version: u16,
    pub workflow_stage: OperatorWorkflowStageV2,
    pub governance_surfaces_digest_prefix: String,
    pub applied_supported_scope_digest_prefix: String,
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
    #[serde(default)]
    pub residual_free_governance_authority_digest_prefix: String,
    #[serde(default)]
    pub governance_absolute_sweep_digest_prefix: String,
    #[serde(default)]
    pub governance_terminal_sweep_digest_prefix: String,
    #[serde(default)]
    pub absolute_final_governance_terminal_sweep_digest_prefix: String,
    #[serde(default)]
    pub governance_ultimate_sweep_digest_prefix: String,
    #[serde(default)]
    pub final_readiness_consumer_authority_digest_prefix: String,
    #[serde(default)]
    pub readiness_residual_sweep_digest_prefix: String,
    #[serde(default)]
    pub residual_free_readiness_authority_digest_prefix: String,
    #[serde(default)]
    pub readiness_absolute_sweep_digest_prefix: String,
    #[serde(default)]
    pub readiness_terminal_sweep_digest_prefix: String,
    #[serde(default)]
    pub readiness_ultimate_sweep_digest_prefix: String,
    #[serde(default)]
    pub final_bundle_residual_sweep_digest_prefix: String,
    #[serde(default)]
    pub final_primary_semantics_residual_sweep_digest_prefix: String,
    #[serde(default)]
    pub residual_free_primary_semantics_authority_digest_prefix: String,
    #[serde(default)]
    pub primary_semantics_absolute_sweep_digest_prefix: String,
    #[serde(default)]
    pub primary_semantics_terminal_sweep_digest_prefix: String,
    #[serde(default)]
    pub primary_semantics_ultimate_sweep_digest_prefix: String,
    #[serde(default)]
    pub primary_semantics_convergence_sweep_digest_prefix: String,
    #[serde(default)]
    pub governance_convergence_sweep_digest_prefix: String,
    #[serde(default)]
    pub readiness_convergence_sweep_digest_prefix: String,
    #[serde(default)]
    pub readiness_stabilization_sweep_digest_prefix: String,
    #[serde(default)]
    pub governance_stabilization_sweep_digest_prefix: String,
    #[serde(default)]
    pub bundle_stabilization_sweep_digest_prefix: String,
    #[serde(default)]
    pub primary_semantics_stabilization_sweep_digest_prefix: String,
    #[serde(default)]
    pub bundle_convergence_sweep_digest_prefix: String,
    #[serde(default)]
    pub canonical_convergence_continuity_authority_digest_prefix: String,
    #[serde(default)]
    pub canonical_stabilization_continuity_authority_digest_prefix: String,
    #[serde(default)]
    pub bundle_terminal_sweep_digest_prefix: String,
    #[serde(default)]
    pub residual_free_continuity_authority_digest_prefix: String,
    #[serde(default)]
    pub final_input_continuity_authority_digest_prefix: String,
    #[serde(default)]
    pub terminal_absolute_final_input_continuity_authority_digest_prefix: String,
    #[serde(default)]
    pub ultimate_terminal_absolute_final_input_continuity_authority_digest_prefix: String,
    pub operator_review_packet_digest_prefix: String,
    pub operator_signoff_digest_prefix: String,
    pub interop_matrix_digest_prefix: String,
    pub export_normalize_check_digest_prefix: String,
    pub export_targets: OperatorWorkflowExportTargetsV1,
    pub blocking_codes: Vec<String>,
    pub remediation_codes: Vec<String>,
    pub chain_digest: String,
}

#[derive(Debug, Clone)]
pub struct OperatorWorkflowArgs {
    pub run_id: Option<String>,
    pub latest: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OperatorWorkflowPolicyV1 {
    pub schema_version: u16,
}

impl Default for OperatorWorkflowPolicyV1 {
    fn default() -> Self {
        Self { schema_version: 1 }
    }
}

#[derive(Debug, Clone)]
pub struct OperatorWorkflowReductionInputs<'a> {
    pub governance: &'a GovernanceSurfacesCheckReportV1,
    pub applied_scope: &'a AppliedScopeCheckReportV1,
    pub review_packet: &'a OperatorReviewPacketV1,
    pub signoff: &'a OperatorSignoffDecisionV1,
    pub interop: &'a InteropConsistencyMatrixReportV1,
    pub normalize: &'a ExportNormalizeCheckReportV1,
    pub repro_verify: Option<bool>,
}

impl OperatorWorkflowPolicyV1 {
    pub fn reduce(
        &self,
        inputs: OperatorWorkflowReductionInputs<'_>,
    ) -> Result<OperatorWorkflowChainV1, OpsError> {
        let mut stage_blocking = BTreeSet::new();
        let mut export_blocking = BTreeSet::new();
        let mut remediation = BTreeSet::new();

        if inputs.governance.status != "PASS" {
            stage_blocking.insert("WORKFLOW_BLOCK_GOVERNANCE_SURFACES_INVALID".to_string());
            remediation.insert("run_governance_surfaces_check".to_string());
        }
        if inputs.governance.governance_primary_surfaces.is_none() {
            stage_blocking.insert("WORKFLOW_BLOCK_GOVERNANCE_SURFACES_MISSING".to_string());
            remediation.insert("run_governance_surfaces_check".to_string());
        }

        if inputs.applied_scope.status != "PASS" {
            stage_blocking.insert("WORKFLOW_BLOCK_APPLIED_SCOPE_MISMATCH".to_string());
            remediation.insert("run_models_applied_scope_check".to_string());
        }

        if inputs.review_packet.applied_supported_set_digest_prefix
            != inputs.applied_scope.applied_scope_digest
            || inputs.signoff.applied_supported_set_digest_prefix
                != inputs.applied_scope.applied_scope_digest
        {
            stage_blocking.insert("WORKFLOW_BLOCK_APPLIED_SCOPE_AUTHORITY_MISMATCH".to_string());
            remediation.insert("run_operator_export_chain_check".to_string());
        }

        if inputs.review_packet.applied_context_digest_prefix
            != inputs.signoff.applied_context_digest_prefix
            || inputs.review_packet.reviewability_reduction_digest_prefix
                != inputs.signoff.reviewability_reduction_digest_prefix
        {
            stage_blocking.insert("WORKFLOW_BLOCK_REVIEWABILITY_BASIS_MISMATCH".to_string());
            remediation.insert("run_review_truth_check".to_string());
        }
        if inputs.review_packet.canonical_readiness_spine_digest_prefix == "MISSING"
            || inputs.signoff.canonical_readiness_spine_digest_prefix == "MISSING"
            || inputs.review_packet.canonical_readiness_spine_digest_prefix
                != inputs.signoff.canonical_readiness_spine_digest_prefix
        {
            stage_blocking.insert("WORKFLOW_BLOCK_READINESS_SPINE_DRIFT".to_string());
            remediation.insert("run_readiness_spine_check".to_string());
        }
        if inputs
            .review_packet
            .canonical_readiness_authority_digest_prefix
            == "MISSING"
            || inputs.signoff.canonical_readiness_authority_digest_prefix == "MISSING"
        {
            stage_blocking.insert("CANONICAL_READINESS_SPINE_REQUIRED".to_string());
            remediation.insert("run_readiness_spine_sweep".to_string());
        }

        if !matches!(
            inputs.review_packet.review_stage,
            crate::OperatorReviewStageV1::ReviewActiveReady
                | crate::OperatorReviewStageV1::ReviewShadowReady
        ) {
            stage_blocking.insert("WORKFLOW_BLOCK_REVIEW_PACKET_BLOCKED".to_string());
            remediation.insert("run_operator_review_packet".to_string());
        }
        if matches!(inputs.signoff.decision, SignoffDecisionStateV1::NotReady) {
            stage_blocking.insert("WORKFLOW_BLOCK_OPERATOR_SIGNOFF_NOT_READY".to_string());
            remediation.insert("run_operator_signoff".to_string());
        }
        if !matches!(
            inputs.interop.summary.overall_status,
            InteropOverallStatusV1::Pass
        ) {
            stage_blocking.insert("WORKFLOW_BLOCK_INTEROP_CONSISTENCY_FAIL".to_string());
            remediation.insert("run_interop_consistency_matrix".to_string());
        }
        if !inputs.normalize.pass {
            stage_blocking.insert("WORKFLOW_BLOCK_EXPORT_NORMALIZE_FAIL".to_string());
            remediation.insert("run_exports_normalize_check".to_string());
        }

        let has_repro_surface = inputs
            .interop
            .matrix
            .surfaces
            .iter()
            .find(|entry| matches!(entry.surface_kind, CrossSurfaceKindV1::ReproPackManifest))
            .is_some_and(|entry| entry.surface_digest_prefix.is_some());
        let has_bugkit_surface = inputs
            .interop
            .matrix
            .surfaces
            .iter()
            .find(|entry| matches!(entry.surface_kind, CrossSurfaceKindV1::BugKitManifest))
            .is_some_and(|entry| entry.surface_digest_prefix.is_some());

        let repro_verify_expected_pass = inputs.repro_verify.unwrap_or(true);

        let core_ready = stage_blocking.is_empty();
        let repro_ready = core_ready && has_repro_surface && repro_verify_expected_pass;
        let bugkit_ready = core_ready && has_bugkit_surface;

        if core_ready {
            if !has_repro_surface {
                export_blocking.insert("WORKFLOW_BLOCK_EXPORT_REPRO_ARTIFACT_MISSING".to_string());
                remediation.insert("run_repro_pack".to_string());
                remediation.insert("run_repro_verify".to_string());
            }
            if !has_bugkit_surface {
                export_blocking.insert("WORKFLOW_BLOCK_EXPORT_BUGKIT_ARTIFACT_MISSING".to_string());
                remediation.insert("run_bugkit_build".to_string());
            }
            if !repro_verify_expected_pass {
                export_blocking.insert("WORKFLOW_BLOCK_EXPORT_REPRO_VERIFY_FAIL".to_string());
                remediation.insert("run_repro_verify".to_string());
            }
        }

        let workflow_stage = if !stage_blocking.is_empty() {
            OperatorWorkflowStageV2::WorkflowBlocked
        } else if export_blocking.is_empty() && repro_ready && bugkit_ready {
            OperatorWorkflowStageV2::WorkflowExportReady
        } else {
            OperatorWorkflowStageV2::WorkflowReviewReady
        };

        let mut blocking = stage_blocking;
        blocking.extend(export_blocking);

        let governance_surfaces_digest_prefix = inputs
            .governance
            .governance_primary_surfaces
            .as_ref()
            .map(|v| prefix_hex(&v.governance_surfaces_digest, DIGEST_PREFIX_LEN))
            .unwrap_or_default();
        let applied_supported_scope_digest_prefix =
            inputs.applied_scope.applied_scope_digest.clone();
        let applied_supported_set_digest_prefix = inputs
            .review_packet
            .applied_supported_set_digest_prefix
            .clone();
        let applied_context_digest_prefix =
            inputs.review_packet.applied_context_digest_prefix.clone();
        let reviewability_reduction_digest_prefix = inputs
            .review_packet
            .reviewability_reduction_digest_prefix
            .clone();
        let operator_review_packet_digest_prefix =
            prefix_hex(&inputs.review_packet.packet_digest, DIGEST_PREFIX_LEN);
        let operator_signoff_digest_prefix =
            prefix_hex(&inputs.signoff.decision_digest, DIGEST_PREFIX_LEN);
        let interop_matrix_digest_prefix =
            prefix_hex(&inputs.interop.matrix.matrix_digest, DIGEST_PREFIX_LEN);
        let export_normalize_check_digest_prefix = prefix_hex(
            &sha256_hex(&serde_json::to_vec(inputs.normalize)?),
            DIGEST_PREFIX_LEN,
        );

        let mut chain = OperatorWorkflowChainV1 {
            schema_version: 1,
            workflow_stage,
            governance_surfaces_digest_prefix,
            applied_supported_scope_digest_prefix,
            applied_supported_set_digest_prefix,
            applied_context_digest_prefix,
            reviewability_reduction_digest_prefix,
            canonical_readiness_spine_digest_prefix: inputs
                .review_packet
                .canonical_readiness_spine_digest_prefix
                .clone(),
            canonical_readiness_authority_digest_prefix: inputs
                .review_packet
                .canonical_readiness_authority_digest_prefix
                .clone(),
            canonical_governance_entry_digest_prefix: inputs
                .review_packet
                .canonical_governance_entry_digest_prefix
                .clone(),
            final_governance_consumer_authority_digest_prefix: inputs
                .review_packet
                .final_governance_consumer_authority_digest_prefix
                .clone(),
            governance_residual_sweep_digest_prefix: inputs
                .review_packet
                .governance_residual_sweep_digest_prefix
                .clone(),
            residual_free_governance_authority_digest_prefix: inputs
                .review_packet
                .residual_free_governance_authority_digest_prefix
                .clone(),
            governance_absolute_sweep_digest_prefix: inputs
                .review_packet
                .governance_absolute_sweep_digest_prefix
                .clone(),
            governance_terminal_sweep_digest_prefix: "MISSING".to_string(),
            absolute_final_governance_terminal_sweep_digest_prefix: inputs
                .review_packet
                .absolute_final_governance_terminal_sweep_digest_prefix
                .clone(),
            governance_ultimate_sweep_digest_prefix: inputs
                .review_packet
                .governance_ultimate_sweep_digest_prefix
                .clone(),
            final_readiness_consumer_authority_digest_prefix: inputs
                .review_packet
                .final_readiness_consumer_authority_digest_prefix
                .clone(),
            readiness_residual_sweep_digest_prefix: inputs
                .review_packet
                .readiness_residual_sweep_digest_prefix
                .clone(),
            residual_free_readiness_authority_digest_prefix: inputs
                .review_packet
                .residual_free_readiness_authority_digest_prefix
                .clone(),
            readiness_absolute_sweep_digest_prefix: inputs
                .review_packet
                .readiness_absolute_sweep_digest_prefix
                .clone(),
            readiness_terminal_sweep_digest_prefix: inputs
                .review_packet
                .readiness_terminal_sweep_digest_prefix
                .clone(),
            readiness_ultimate_sweep_digest_prefix: inputs
                .review_packet
                .readiness_ultimate_sweep_digest_prefix
                .clone(),
            final_bundle_residual_sweep_digest_prefix: "MISSING".to_string(),
            final_primary_semantics_residual_sweep_digest_prefix: "MISSING".to_string(),
            residual_free_primary_semantics_authority_digest_prefix: "MISSING".to_string(),
            primary_semantics_absolute_sweep_digest_prefix: "MISSING".to_string(),
            primary_semantics_terminal_sweep_digest_prefix: "MISSING".to_string(),
            primary_semantics_ultimate_sweep_digest_prefix: "MISSING".to_string(),
            primary_semantics_convergence_sweep_digest_prefix: "MISSING".to_string(),
            governance_convergence_sweep_digest_prefix: "MISSING".to_string(),
            readiness_convergence_sweep_digest_prefix: "MISSING".to_string(),
            readiness_stabilization_sweep_digest_prefix: "MISSING".to_string(),
            governance_stabilization_sweep_digest_prefix: "MISSING".to_string(),
            bundle_stabilization_sweep_digest_prefix: "MISSING".to_string(),
            primary_semantics_stabilization_sweep_digest_prefix: "MISSING".to_string(),
            bundle_convergence_sweep_digest_prefix: "MISSING".to_string(),
            canonical_convergence_continuity_authority_digest_prefix: "MISSING".to_string(),
            canonical_stabilization_continuity_authority_digest_prefix: "MISSING".to_string(),
            bundle_terminal_sweep_digest_prefix: "MISSING".to_string(),
            residual_free_continuity_authority_digest_prefix: "MISSING".to_string(),
            final_input_continuity_authority_digest_prefix: "MISSING".to_string(),
            terminal_absolute_final_input_continuity_authority_digest_prefix: "MISSING".to_string(),
            ultimate_terminal_absolute_final_input_continuity_authority_digest_prefix: "MISSING"
                .to_string(),
            operator_review_packet_digest_prefix,
            operator_signoff_digest_prefix,
            interop_matrix_digest_prefix,
            export_normalize_check_digest_prefix,
            export_targets: OperatorWorkflowExportTargetsV1 {
                repro_ready,
                bugkit_ready,
            },
            blocking_codes: blocking.into_iter().take(CODE_CAP).collect(),
            remediation_codes: remediation.into_iter().take(CODE_CAP).collect(),
            chain_digest: String::new(),
        };
        chain.chain_digest = chain_digest_hex(&chain)?;
        Ok(chain)
    }
}

pub fn operator_workflow_chain(
    workdir: &Path,
    args: &OperatorWorkflowArgs,
    out: &Path,
) -> Result<OperatorWorkflowChainV1, OpsError> {
    let out_root = workdir.join("out");
    fs::create_dir_all(&out_root)?;

    let governance = governance_surfaces_check(
        workdir,
        &out_root.join("governance_surfaces_check_operator_workflow.json"),
    )?;
    let applied_scope = models_applied_scope_check(
        workdir,
        &out_root.join("applied_scope_check_operator_workflow.json"),
    )?;
    let review_packet = operator_review_packet(
        workdir,
        &OperatorReviewPacketArgs {
            run_id: args.run_id.clone(),
            latest: args.latest,
        },
        &out_root.join("operator_review_packet_operator_workflow.json"),
    )?;
    let signoff = operator_signoff(
        workdir,
        &OperatorSignoffArgs {
            run_id: args.run_id.clone(),
            latest: args.latest,
            profile: std::env::var("UCF_PROFILE").unwrap_or_else(|_| "test".to_string()),
        },
        &out_root.join("operator_signoff_operator_workflow.json"),
    )?;
    let interop = interop_consistency_matrix(
        workdir,
        &out_root.join("interop_consistency_matrix_operator_workflow.json"),
    )?;
    let normalize = exports_normalize_check(
        workdir,
        &out_root.join("export_normalize_check_operator_workflow.json"),
    )?;

    let mut chain =
        OperatorWorkflowPolicyV1::default().reduce(OperatorWorkflowReductionInputs {
            governance: &governance,
            applied_scope: &applied_scope,
            review_packet: &review_packet,
            signoff: &signoff,
            interop: &interop,
            normalize: &normalize,
            repro_verify: discover_repro_verify_expectation(&out_root),
        })?;

    chain.final_bundle_residual_sweep_digest_prefix = discover_digest_prefix(
        &out_root,
        "bundle_residual_sweep.json",
        "sweep.sweep_digest",
    )
    .unwrap_or_else(|| "MISSING".to_string());
    chain.final_primary_semantics_residual_sweep_digest_prefix = discover_digest_prefix(
        &out_root,
        "primary_semantics_residual_sweep.json",
        "sweep.sweep_digest",
    )
    .unwrap_or_else(|| "MISSING".to_string());
    chain.residual_free_primary_semantics_authority_digest_prefix = discover_digest_prefix(
        &out_root,
        "residual_free_primary_semantics_sweep.json",
        "authority.authority_digest",
    )
    .unwrap_or_else(|| "MISSING".to_string());
    chain.primary_semantics_absolute_sweep_digest_prefix = discover_digest_prefix(
        &out_root,
        "primary_semantics_absolute_sweep.json",
        "sweep.sweep_digest",
    )
    .unwrap_or_else(|| "MISSING".to_string());
    chain.primary_semantics_terminal_sweep_digest_prefix = discover_digest_prefix(
        &out_root,
        "primary_semantics_terminal_sweep.json",
        "sweep.sweep_digest",
    )
    .unwrap_or_else(|| "MISSING".to_string());
    chain.primary_semantics_ultimate_sweep_digest_prefix = discover_digest_prefix(
        &out_root,
        "primary_semantics_ultimate_sweep.json",
        "sweep.sweep_digest",
    )
    .unwrap_or_else(|| "MISSING".to_string());
    chain.primary_semantics_convergence_sweep_digest_prefix = discover_digest_prefix(
        &out_root,
        "primary_semantics_convergence_sweep.json",
        "sweep.convergence_digest",
    )
    .unwrap_or_else(|| "MISSING".to_string());
    chain.governance_convergence_sweep_digest_prefix = discover_digest_prefix(
        &out_root,
        "governance_convergence_sweep.json",
        "sweep.convergence_digest",
    )
    .unwrap_or_else(|| "MISSING".to_string());
    chain.readiness_convergence_sweep_digest_prefix = discover_digest_prefix(
        &out_root,
        "readiness_convergence_sweep.json",
        "sweep.convergence_digest",
    )
    .unwrap_or_else(|| "MISSING".to_string());
    chain.readiness_stabilization_sweep_digest_prefix = discover_digest_prefix(
        &out_root,
        "readiness_stabilization_sweep.json",
        "sweep.stabilization_digest",
    )
    .unwrap_or_else(|| "MISSING".to_string());
    chain.governance_stabilization_sweep_digest_prefix = discover_digest_prefix(
        &out_root,
        "governance_stabilization_sweep.json",
        "sweep.stabilization_digest",
    )
    .unwrap_or_else(|| "MISSING".to_string());
    chain.bundle_stabilization_sweep_digest_prefix = discover_digest_prefix(
        &out_root,
        "bundle_stabilization_sweep.json",
        "sweep.stabilization_digest",
    )
    .unwrap_or_else(|| "MISSING".to_string());
    chain.primary_semantics_stabilization_sweep_digest_prefix = discover_digest_prefix(
        &out_root,
        "primary_semantics_stabilization_sweep.json",
        "sweep.stabilization_digest",
    )
    .unwrap_or_else(|| "MISSING".to_string());
    chain.bundle_convergence_sweep_digest_prefix = discover_digest_prefix(
        &out_root,
        "bundle_convergence_sweep.json",
        "sweep.convergence_digest",
    )
    .unwrap_or_else(|| "MISSING".to_string());
    chain.canonical_convergence_continuity_authority_digest_prefix = discover_digest_prefix(
        &out_root,
        "canonical_convergence_continuity_sweep.json",
        "authority_digest",
    )
    .unwrap_or_else(|| "MISSING".to_string());
    chain.canonical_stabilization_continuity_authority_digest_prefix = discover_digest_prefix(
        &out_root,
        "canonical_stabilization_continuity_sweep.json",
        "authority_digest",
    )
    .unwrap_or_else(|| "MISSING".to_string());
    chain.governance_terminal_sweep_digest_prefix = discover_digest_prefix(
        &out_root,
        "governance_terminal_sweep.json",
        "sweep.sweep_digest",
    )
    .unwrap_or_else(|| "MISSING".to_string());
    chain.absolute_final_governance_terminal_sweep_digest_prefix =
        chain.governance_terminal_sweep_digest_prefix.clone();
    chain.readiness_ultimate_sweep_digest_prefix = discover_digest_prefix(
        &out_root,
        "readiness_ultimate_sweep.json",
        "sweep.sweep_digest",
    )
    .unwrap_or_else(|| "MISSING".to_string());
    chain.governance_ultimate_sweep_digest_prefix = discover_digest_prefix(
        &out_root,
        "governance_ultimate_sweep.json",
        "sweep.sweep_digest",
    )
    .unwrap_or_else(|| "MISSING".to_string());
    chain.bundle_terminal_sweep_digest_prefix = discover_digest_prefix(
        &out_root,
        "bundle_terminal_sweep.json",
        "sweep.sweep_digest",
    )
    .unwrap_or_else(|| "MISSING".to_string());
    chain.residual_free_continuity_authority_digest_prefix = discover_digest_prefix(
        &out_root,
        "residual_free_continuity_sweep.json",
        "authority_digest",
    )
    .unwrap_or_else(|| "MISSING".to_string());
    chain.final_input_continuity_authority_digest_prefix = discover_digest_prefix(
        &out_root,
        "ultimate_terminal_absolute_final_input_continuity_sweep.json",
        "authority_digest",
    )
    .unwrap_or_else(|| "MISSING".to_string());
    chain.terminal_absolute_final_input_continuity_authority_digest_prefix =
        discover_digest_prefix(
            &out_root,
            "terminal_absolute_final_input_continuity_sweep.json",
            "authority_digest",
        )
        .unwrap_or_else(|| "MISSING".to_string());
    chain.ultimate_terminal_absolute_final_input_continuity_authority_digest_prefix =
        discover_digest_prefix(
            &out_root,
            "ultimate_terminal_absolute_final_input_continuity_sweep.json",
            "authority_digest",
        )
        .unwrap_or_else(|| "MISSING".to_string());
    chain.chain_digest = chain_digest_hex(&chain)?;

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&chain)?)?;
    Ok(chain)
}

pub fn operator_workflow_chain_text(chain: &OperatorWorkflowChainV1) -> String {
    format!(
        "workflow_stage={:?}\nblocking_codes={}\nrepro_ready={}\nbugkit_ready={}",
        chain.workflow_stage,
        if chain.blocking_codes.is_empty() {
            "none".to_string()
        } else {
            chain.blocking_codes.join(",")
        },
        chain.export_targets.repro_ready,
        chain.export_targets.bugkit_ready
    )
}

fn discover_repro_verify_expectation(out_root: &Path) -> Option<bool> {
    let direct = out_root.join("repro_verify.json");
    if let Ok(report) = read_json::<crate::ReproVerifyReport>(&direct) {
        return Some(report.pass);
    }

    let entries = fs::read_dir(out_root).ok()?;
    let mut candidates = entries
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            if name.starts_with("repro_verify") && name.ends_with(".json") {
                Some(path)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    candidates.sort();
    candidates
        .into_iter()
        .find_map(|path| read_json::<crate::ReproVerifyReport>(&path).ok())
        .map(|report| report.pass)
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, OpsError> {
    let body = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&body)?)
}

fn discover_digest_prefix(out_root: &Path, rel: &str, key_path: &str) -> Option<String> {
    let body = fs::read_to_string(out_root.join(rel)).ok()?;
    let value: serde_json::Value = serde_json::from_str(&body).ok()?;
    let mut current = &value;
    for key in key_path.split('.') {
        current = current.get(key)?;
    }
    current.as_str().map(|v| prefix_hex(v, DIGEST_PREFIX_LEN))
}

fn chain_digest_hex(chain: &OperatorWorkflowChainV1) -> Result<String, OpsError> {
    let mut digestible = chain.clone();
    digestible.chain_digest.clear();
    Ok(sha256_hex(&serde_json::to_vec(&digestible)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interop_consistency::{
        CrossSurfaceContextMatchStatusV1, CrossSurfaceKindV1, InteropConsistencyRecordV1,
        InteropConsistencySummaryV1,
    };
    use crate::operator_review_packet::{
        OperatorReviewPacketArtifactsV1, OperatorReviewPacketGateDigestsV1,
    };
    use crate::operator_signoff::GateReportDigestsV1;
    use crate::{
        CrossSurfaceContextMatrixV1, CrossSurfaceEntryV1, CrossSurfaceMatchRulesV1,
        InteropMismatchCategoryV1, OperatorReviewStageV1,
    };

    fn governance(pass: bool) -> GovernanceSurfacesCheckReportV1 {
        GovernanceSurfacesCheckReportV1 {
            schema_version: 1,
            status: if pass { "PASS" } else { "FAIL" }.to_string(),
            summary_code: if pass { "PASS" } else { "FAIL" }.to_string(),
            governance_primary_surfaces: pass.then(|| crate::GovernancePrimarySurfacesV1 {
                schema_version: 1,
                supported_slot_set_digest_prefix: "digest123456789012".to_string(),
                policy_graph_digest_prefix: "policy123456789012".to_string(),
                manifest_digest_prefix: "manifest1234567890".to_string(),
                backend_evidence_snapshot_digest_prefix: "back1234567890123".to_string(),
                active_review_snapshot_digest_prefix: "active12345678901".to_string(),
                consistency_ok: true,
                governance_surfaces_digest:
                    "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789".to_string(),
            }),
        }
    }

    fn applied_scope(pass: bool) -> AppliedScopeCheckReportV1 {
        AppliedScopeCheckReportV1 {
            schema_version: 1,
            status: if pass { "PASS" } else { "FAIL" }.to_string(),
            applied_scope_digest: "scope123456789012".to_string(),
            checked_artifacts: vec![],
            mismatch_categories: vec![],
            remediation_codes: vec![],
        }
    }

    fn review(stage: OperatorReviewStageV1) -> OperatorReviewPacketV1 {
        OperatorReviewPacketV1 {
            schema_version: 1,
            review_stage: stage,
            supported_slot_set_digest: "scope123456789012".to_string(),
            policy_graph_digest_prefix: "policy123456789012".to_string(),
            manifest_digest_prefix: "manifest1234567890".to_string(),
            applied_supported_set_digest_prefix: "scope123456789012".to_string(),
            applied_context_digest_prefix: "context1234567890".to_string(),
            reviewability_reduction_digest_prefix: "reviewred12345678".to_string(),
            canonical_readiness_spine_digest_prefix: "spine123456789012".to_string(),
            canonical_readiness_authority_digest_prefix: "spine123456789012".to_string(),
            canonical_governance_entry_digest_prefix: "entry123456789012".to_string(),
            final_governance_consumer_authority_digest_prefix: "gov1234567890123".to_string(),
            governance_residual_sweep_digest_prefix: "sweep12345678901".to_string(),
            residual_free_governance_authority_digest_prefix: "MISSING".to_string(),
            governance_absolute_sweep_digest_prefix: "MISSING".to_string(),
            absolute_final_governance_terminal_sweep_digest_prefix: "MISSING".to_string(),
            governance_ultimate_sweep_digest_prefix: "MISSING".to_string(),
            final_readiness_consumer_authority_digest_prefix: "ready123456789012".to_string(),
            readiness_residual_sweep_digest_prefix: "rrs1234567890123".to_string(),
            residual_free_readiness_authority_digest_prefix: "MISSING".to_string(),
            readiness_absolute_sweep_digest_prefix: "MISSING".to_string(),
            readiness_terminal_sweep_digest_prefix: "MISSING".to_string(),
            readiness_ultimate_sweep_digest_prefix: "MISSING".to_string(),
            readiness_stabilization_sweep_digest_prefix: "MISSING".to_string(),
            final_primary_semantics_residual_sweep_digest_prefix: "MISSING".to_string(),
            residual_free_primary_semantics_authority_digest_prefix: "MISSING".to_string(),
            primary_semantics_absolute_sweep_digest_prefix: "MISSING".to_string(),
            primary_semantics_terminal_sweep_digest_prefix: "MISSING".to_string(),
            primary_semantics_ultimate_sweep_digest_prefix: "MISSING".to_string(),
            primary_semantics_convergence_sweep_digest_prefix: "MISSING".to_string(),
            artifacts: OperatorReviewPacketArtifactsV1 {
                backend_evidence_snapshot_digest_prefix: "a".repeat(16),
                active_review_snapshot_digest_prefix: "b".repeat(16),
                operator_signoff_digest_prefix: "c".repeat(16),
                operator_report_digest_prefix: "d".repeat(16),
                gate_digests: OperatorReviewPacketGateDigestsV1 {
                    v0: "e".repeat(16),
                    v1: "f".repeat(16),
                    v2: "0".repeat(16),
                    v3: "1".repeat(16),
                    v4: "2".repeat(16),
                },
                backend_resolution_digest_prefix: Some("3".repeat(16)),
                applied_supported_set_context_digest_prefix: "scope123456789012".to_string(),
            },
            supported_slots: vec![],
            blocking_codes: vec![],
            remediation_codes: vec![],
            packet_digest: "d".repeat(64),
        }
    }

    fn signoff(decision: SignoffDecisionStateV1) -> OperatorSignoffDecisionV1 {
        OperatorSignoffDecisionV1 {
            schema_version: 1,
            decision,
            supported_slot_set_digest: "scope123456789012".to_string(),
            policy_graph_digest_prefix: "policy123456789012".to_string(),
            manifest_digest_prefix: "manifest1234567890".to_string(),
            evidence_snapshot_digest_prefix: "a".repeat(16),
            active_review_snapshot_digest_prefix: Some("b".repeat(16)),
            operator_report_digest_prefix: "c".repeat(16),
            applied_supported_set_digest_prefix: "scope123456789012".to_string(),
            applied_context_digest_prefix: "context1234567890".to_string(),
            reviewability_reduction_digest_prefix: "reviewred12345678".to_string(),
            canonical_readiness_spine_digest_prefix: "spine123456789012".to_string(),
            canonical_readiness_authority_digest_prefix: "spine123456789012".to_string(),
            canonical_governance_entry_digest_prefix: "entry123456789012".to_string(),
            final_governance_consumer_authority_digest_prefix: "gov1234567890123".to_string(),
            governance_residual_sweep_digest_prefix: "sweep12345678901".to_string(),
            residual_free_governance_authority_digest_prefix: "MISSING".to_string(),
            governance_absolute_sweep_digest_prefix: "MISSING".to_string(),
            absolute_final_governance_terminal_sweep_digest_prefix: "MISSING".to_string(),
            governance_ultimate_sweep_digest_prefix: "MISSING".to_string(),
            final_readiness_consumer_authority_digest_prefix: "ready123456789012".to_string(),
            readiness_residual_sweep_digest_prefix: "rrs1234567890123".to_string(),
            residual_free_readiness_authority_digest_prefix: "MISSING".to_string(),
            readiness_absolute_sweep_digest_prefix: "MISSING".to_string(),
            readiness_terminal_sweep_digest_prefix: "MISSING".to_string(),
            readiness_ultimate_sweep_digest_prefix: "MISSING".to_string(),
            readiness_stabilization_sweep_digest_prefix: "MISSING".to_string(),
            final_primary_semantics_residual_sweep_digest_prefix: "MISSING".to_string(),
            residual_free_primary_semantics_authority_digest_prefix: "MISSING".to_string(),
            primary_semantics_absolute_sweep_digest_prefix: "MISSING".to_string(),
            primary_semantics_terminal_sweep_digest_prefix: "MISSING".to_string(),
            primary_semantics_ultimate_sweep_digest_prefix: "MISSING".to_string(),
            primary_semantics_convergence_sweep_digest_prefix: "MISSING".to_string(),
            gate_report_digests: GateReportDigestsV1 {
                v0: "d".repeat(16),
                v1: "e".repeat(16),
                v2: "f".repeat(16),
                v3: "0".repeat(16),
            },
            reasons: vec![],
            remediation_codes: vec![],
            canonical_remediation_codes: vec![],
            decision_digest: "a".repeat(64),
        }
    }

    fn interop(pass: bool, with_exports: bool) -> InteropConsistencyMatrixReportV1 {
        let mut surfaces = vec![CrossSurfaceEntryV1 {
            surface_kind: CrossSurfaceKindV1::OperatorSignoff,
            surface_digest_prefix: Some("1".repeat(16)),
            supported_set_digest_prefix: Some("scope123456789012".to_string()),
            policy_graph_digest_prefix: Some("policy123456789012".to_string()),
            manifest_digest_prefix: Some("manifest1234567890".to_string()),
            primary_blocking_code: None,
            primary_remediation_code: None,
            artifact_refs_digest_prefix: None,
            context_match_status: CrossSurfaceContextMatchStatusV1::Match,
        }];
        if with_exports {
            surfaces.push(CrossSurfaceEntryV1 {
                surface_kind: CrossSurfaceKindV1::ReproPackManifest,
                surface_digest_prefix: Some("2".repeat(16)),
                supported_set_digest_prefix: Some("scope123456789012".to_string()),
                policy_graph_digest_prefix: Some("policy123456789012".to_string()),
                manifest_digest_prefix: Some("manifest1234567890".to_string()),
                primary_blocking_code: None,
                primary_remediation_code: None,
                artifact_refs_digest_prefix: None,
                context_match_status: CrossSurfaceContextMatchStatusV1::Match,
            });
            surfaces.push(CrossSurfaceEntryV1 {
                surface_kind: CrossSurfaceKindV1::BugKitManifest,
                surface_digest_prefix: Some("3".repeat(16)),
                supported_set_digest_prefix: Some("scope123456789012".to_string()),
                policy_graph_digest_prefix: Some("policy123456789012".to_string()),
                manifest_digest_prefix: Some("manifest1234567890".to_string()),
                primary_blocking_code: None,
                primary_remediation_code: None,
                artifact_refs_digest_prefix: None,
                context_match_status: CrossSurfaceContextMatchStatusV1::Match,
            });
        }
        InteropConsistencyMatrixReportV1 {
            schema_version: 1,
            matrix: CrossSurfaceContextMatrixV1 {
                schema_version: 1,
                applied_supported_set_digest_prefix: "scope123456789012".to_string(),
                canonical_governance_entry_digest_prefix: "entry123456789012".to_string(),
                final_governance_consumer_authority_digest_prefix: "gov1234567890123".to_string(),
                governance_residual_sweep_digest_prefix: "sweep12345678901".to_string(),
                residual_free_governance_authority_digest_prefix: "MISSING".to_string(),
                governance_absolute_sweep_digest_prefix: "MISSING".to_string(),
                absolute_final_governance_terminal_sweep_digest_prefix: "MISSING".to_string(),
                governance_ultimate_sweep_digest_prefix: "MISSING".to_string(),
                final_readiness_consumer_authority_digest_prefix: "ready123456789012".to_string(),
                readiness_residual_sweep_digest_prefix: "rrs1234567890123".to_string(),
                residual_free_readiness_authority_digest_prefix: "MISSING".to_string(),
                readiness_absolute_sweep_digest_prefix: "MISSING".to_string(),
                readiness_terminal_sweep_digest_prefix: "MISSING".to_string(),
                readiness_ultimate_sweep_digest_prefix: "MISSING".to_string(),
                readiness_stabilization_sweep_digest_prefix: "MISSING".to_string(),
                final_primary_semantics_residual_sweep_digest_prefix: "MISSING".to_string(),
                residual_free_primary_semantics_authority_digest_prefix: "MISSING".to_string(),
                primary_semantics_absolute_sweep_digest_prefix: "MISSING".to_string(),
                primary_semantics_terminal_sweep_digest_prefix: "MISSING".to_string(),
                primary_semantics_ultimate_sweep_digest_prefix: "MISSING".to_string(),
                primary_semantics_convergence_sweep_digest_prefix: "MISSING".to_string(),
                policy_graph_digest_prefix: "policy123456789012".to_string(),
                manifest_digest_prefix: "manifest1234567890".to_string(),
                surfaces,
                matrix_digest: "f".repeat(64),
            },
            match_rules: CrossSurfaceMatchRulesV1 {
                schema_version: 1,
                mismatch_categories: if pass {
                    vec![]
                } else {
                    vec![InteropMismatchCategoryV1::ScopeMismatch]
                },
                canonical_condition_codes: Vec::new(),
                primary_remediation_codes: Vec::new(),
            },
            summary: InteropConsistencySummaryV1 {
                overall_status: if pass {
                    InteropOverallStatusV1::Pass
                } else {
                    InteropOverallStatusV1::Fail
                },
                mismatch_counts: vec![],
            },
            interop_record: InteropConsistencyRecordV1 {
                schema_version: 1,
                matrix_digest_prefix: "f".repeat(16),
                overall_status: if pass {
                    InteropOverallStatusV1::Pass
                } else {
                    InteropOverallStatusV1::Fail
                },
                mismatch_counts: vec![],
            },
        }
    }

    fn normalize(pass: bool) -> ExportNormalizeCheckReportV1 {
        ExportNormalizeCheckReportV1 {
            schema_version: 1,
            pass,
            mismatch_count: usize::from(!pass),
            mismatches: vec![],
            allowed_states: vec![
                "INCLUDED".to_string(),
                "MISSING".to_string(),
                "EXCLUDED".to_string(),
            ],
        }
    }

    #[test]
    fn workflow_reduction_is_deterministic_and_digest_stable() {
        let policy = OperatorWorkflowPolicyV1::default();
        let governance = governance(true);
        let applied_scope = applied_scope(true);
        let review = review(OperatorReviewStageV1::ReviewActiveReady);
        let signoff = signoff(SignoffDecisionStateV1::ReadyForActiveReview);
        let interop = interop(true, true);
        let normalize = normalize(true);
        let inputs = OperatorWorkflowReductionInputs {
            governance: &governance,
            applied_scope: &applied_scope,
            review_packet: &review,
            signoff: &signoff,
            interop: &interop,
            normalize: &normalize,
            repro_verify: Some(true),
        };
        let a = policy.reduce(inputs.clone()).expect("chain");
        let b = policy.reduce(inputs).expect("chain");
        assert_eq!(a, b);
    }

    #[test]
    fn workflow_stage_export_ready() {
        let policy = OperatorWorkflowPolicyV1::default();
        let chain = policy
            .reduce(OperatorWorkflowReductionInputs {
                governance: &governance(true),
                applied_scope: &applied_scope(true),
                review_packet: &review(OperatorReviewStageV1::ReviewActiveReady),
                signoff: &signoff(SignoffDecisionStateV1::ReadyForActiveReview),
                interop: &interop(true, true),
                normalize: &normalize(true),
                repro_verify: Some(true),
            })
            .expect("chain");
        assert_eq!(
            chain.workflow_stage,
            OperatorWorkflowStageV2::WorkflowExportReady
        );
        assert!(chain.export_targets.repro_ready);
        assert!(chain.export_targets.bugkit_ready);
    }

    #[test]
    fn workflow_stage_review_ready_when_exports_missing() {
        let policy = OperatorWorkflowPolicyV1::default();
        let chain = policy
            .reduce(OperatorWorkflowReductionInputs {
                governance: &governance(true),
                applied_scope: &applied_scope(true),
                review_packet: &review(OperatorReviewStageV1::ReviewShadowReady),
                signoff: &signoff(SignoffDecisionStateV1::ReadyForShadow),
                interop: &interop(true, false),
                normalize: &normalize(true),
                repro_verify: Some(true),
            })
            .expect("chain");
        assert_eq!(
            chain.workflow_stage,
            OperatorWorkflowStageV2::WorkflowReviewReady
        );
        assert!(!chain.export_targets.repro_ready);
        assert!(!chain.export_targets.bugkit_ready);
    }

    #[test]
    fn workflow_stage_blocked_on_governance_or_interop_or_normalize() {
        let policy = OperatorWorkflowPolicyV1::default();
        let chain = policy
            .reduce(OperatorWorkflowReductionInputs {
                governance: &governance(false),
                applied_scope: &applied_scope(true),
                review_packet: &review(OperatorReviewStageV1::ReviewActiveReady),
                signoff: &signoff(SignoffDecisionStateV1::ReadyForActiveReview),
                interop: &interop(false, true),
                normalize: &normalize(false),
                repro_verify: Some(true),
            })
            .expect("chain");
        assert_eq!(
            chain.workflow_stage,
            OperatorWorkflowStageV2::WorkflowBlocked
        );
        assert!(chain
            .blocking_codes
            .iter()
            .any(|v| v == "WORKFLOW_BLOCK_GOVERNANCE_SURFACES_INVALID"));
        assert!(chain
            .blocking_codes
            .iter()
            .any(|v| v == "WORKFLOW_BLOCK_INTEROP_CONSISTENCY_FAIL"));
        assert!(chain
            .blocking_codes
            .iter()
            .any(|v| v == "WORKFLOW_BLOCK_EXPORT_NORMALIZE_FAIL"));
    }

    #[test]
    fn workflow_stage_blocked_when_applied_scope_missing() {
        let policy = OperatorWorkflowPolicyV1::default();
        let mut applied = applied_scope(false);
        applied
            .mismatch_categories
            .push("APPLIED_SCOPE_OPERATOR_SIGNOFF_MISSING".to_string());
        let chain = policy
            .reduce(OperatorWorkflowReductionInputs {
                governance: &governance(true),
                applied_scope: &applied,
                review_packet: &review(OperatorReviewStageV1::ReviewActiveReady),
                signoff: &signoff(SignoffDecisionStateV1::ReadyForActiveReview),
                interop: &interop(true, true),
                normalize: &normalize(true),
                repro_verify: Some(true),
            })
            .expect("chain");
        assert_eq!(
            chain.workflow_stage,
            OperatorWorkflowStageV2::WorkflowBlocked
        );
        assert!(chain
            .blocking_codes
            .iter()
            .any(|v| v == "WORKFLOW_BLOCK_APPLIED_SCOPE_MISMATCH"));
    }
}
