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

const ARTIFACT_SPECS: [ArtifactSpec; 108] = [
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
        artifact_id: "supported_scope_execution_v5",
        file_rel: "runtime/ucf-ops/src/models_lifecycle.rs",
        type_name: "SupportedScopeExecutionV5",
        enum_names: &["SupportedScopeExecutionDecisionV5"],
    },
    ArtifactSpec {
        artifact_id: "supported_scope_execution_v6",
        file_rel: "runtime/ucf-ops/src/models_lifecycle.rs",
        type_name: "SupportedScopeExecutionV6",
        enum_names: &["SupportedScopeExecutionDecisionV6"],
    },
    ArtifactSpec {
        artifact_id: "supported_scope_execution_v7",
        file_rel: "runtime/ucf-ops/src/models_lifecycle.rs",
        type_name: "SupportedScopeExecutionV7",
        enum_names: &["SupportedScopeExecutionDecisionV7"],
    },
    ArtifactSpec {
        artifact_id: "supported_scope_execution_v8",
        file_rel: "runtime/ucf-ops/src/models_lifecycle.rs",
        type_name: "SupportedScopeExecutionV8",
        enum_names: &["SupportedScopeExecutionDecisionV8"],
    },
    ArtifactSpec {
        artifact_id: "supported_scope_execution_v9",
        file_rel: "runtime/ucf-ops/src/models_lifecycle.rs",
        type_name: "SupportedScopeExecutionV9",
        enum_names: &["SupportedScopeExecutionDecisionV9"],
    },
    ArtifactSpec {
        artifact_id: "supported_scope_execution_v10",
        file_rel: "runtime/ucf-ops/src/models_lifecycle.rs",
        type_name: "SupportedScopeExecutionV10",
        enum_names: &["SupportedScopeExecutionDecisionV10"],
    },
    ArtifactSpec {
        artifact_id: "supported_scope_execution_v11",
        file_rel: "runtime/ucf-ops/src/models_lifecycle.rs",
        type_name: "SupportedScopeExecutionV11",
        enum_names: &["SupportedScopeExecutionDecisionV11"],
    },
    ArtifactSpec {
        artifact_id: "supported_scope_execution_v12",
        file_rel: "runtime/ucf-ops/src/models_lifecycle.rs",
        type_name: "SupportedScopeExecutionV12",
        enum_names: &["SupportedScopeExecutionDecisionV12"],
    },
    ArtifactSpec {
        artifact_id: "supported_scope_execution_v13",
        file_rel: "runtime/ucf-ops/src/models_lifecycle.rs",
        type_name: "SupportedScopeExecutionV13",
        enum_names: &["SupportedScopeExecutionDecisionV13"],
    },
    ArtifactSpec {
        artifact_id: "supported_scope_execution_v14",
        file_rel: "runtime/ucf-ops/src/models_lifecycle.rs",
        type_name: "SupportedScopeExecutionV14",
        enum_names: &["SupportedScopeExecutionDecisionV14"],
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
        artifact_id: "final_bundle_consumer_authority_v1",
        file_rel: "runtime/ucf-ops/src/final_bundle_consumer_sweep.rs",
        type_name: "FinalBundleConsumerAuthorityV1",
        enum_names: &["FinalBundleConsumerAuthorityStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "final_bundle_residual_sweep_v1",
        file_rel: "runtime/ucf-ops/src/bundle_residual_sweep.rs",
        type_name: "FinalBundleResidualSweepV1",
        enum_names: &[
            "FinalBundleResidualSweepStatusV1",
            "BundleResidualMismatchCategoryV1",
        ],
    },
    ArtifactSpec {
        artifact_id: "bundle_convergence_sweep_v1",
        file_rel: "runtime/ucf-ops/src/bundle_convergence_sweep.rs",
        type_name: "BundleConvergenceSweepV1",
        enum_names: &[
            "BundleConvergenceStatusV1",
            "BundleConvergenceMismatchCategoryV1",
        ],
    },
    ArtifactSpec {
        artifact_id: "bundle_stabilization_sweep_v1",
        file_rel: "runtime/ucf-ops/src/bundle_stabilization_sweep.rs",
        type_name: "BundleStabilizationSweepV1",
        enum_names: &[
            "BundleStabilizationStatusV1",
            "BundleStabilizationMismatchCategoryV1",
        ],
    },
    ArtifactSpec {
        artifact_id: "bundle_final_consolidation_sweep_v1",
        file_rel: "runtime/ucf-ops/src/bundle_final_consolidation_sweep.rs",
        type_name: "BundleFinalConsolidationSweepV1",
        enum_names: &[
            "BundleFinalConsolidationStatusV1",
            "BundleFinalConsolidationMismatchCategoryV1",
        ],
    },
    ArtifactSpec {
        artifact_id: "final_continuity_authority_v2",
        file_rel: "runtime/ucf-ops/src/final_continuity_sweep.rs",
        type_name: "FinalContinuityAuthorityV2",
        enum_names: &[
            "FinalContinuityStatusV2",
            "FinalContinuityMismatchCategoryV1",
        ],
    },
    ArtifactSpec {
        artifact_id: "residual_free_continuity_authority_v1",
        file_rel: "runtime/ucf-ops/src/residual_free_continuity_sweep.rs",
        type_name: "ResidualFreeContinuityAuthorityV1",
        enum_names: &[
            "ResidualFreeContinuityStatusV1",
            "ResidualFreeContinuityMismatchCategoryV1",
        ],
    },
    ArtifactSpec {
        artifact_id: "absolute_final_governance_terminal_sweep_v1",
        file_rel: "runtime/ucf-ops/src/governance_terminal_sweep.rs",
        type_name: "AbsoluteFinalGovernanceTerminalSweepV1",
        enum_names: &["AbsoluteFinalGovernanceTerminalSweepStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "terminal_governance_ultimate_sweep_v1",
        file_rel: "runtime/ucf-ops/src/governance_ultimate_sweep.rs",
        type_name: "TerminalGovernanceUltimateSweepV1",
        enum_names: &["TerminalGovernanceUltimateSweepStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "governance_convergence_sweep_v1",
        file_rel: "runtime/ucf-ops/src/governance_convergence_sweep.rs",
        type_name: "GovernanceConvergenceSweepV1",
        enum_names: &[
            "GovernanceConvergenceStatusV1",
            "GovernanceConvergenceMismatchCategoryV1",
        ],
    },
    ArtifactSpec {
        artifact_id: "governance_stabilization_sweep_v1",
        file_rel: "runtime/ucf-ops/src/governance_stabilization_sweep.rs",
        type_name: "GovernanceStabilizationSweepV1",
        enum_names: &[
            "GovernanceStabilizationStatusV1",
            "GovernanceStabilizationMismatchCategoryV1",
        ],
    },
    ArtifactSpec {
        artifact_id: "governance_final_consolidation_sweep_v1",
        file_rel: "runtime/ucf-ops/src/governance_final_consolidation_sweep.rs",
        type_name: "GovernanceFinalConsolidationSweepV1",
        enum_names: &[
            "GovernanceFinalConsolidationStatusV1",
            "GovernanceFinalConsolidationMismatchCategoryV1",
        ],
    },
    ArtifactSpec {
        artifact_id: "absolute_final_input_continuity_authority_v1",
        file_rel: "runtime/ucf-ops/src/absolute_final_input_continuity_sweep.rs",
        type_name: "AbsoluteFinalInputContinuityAuthorityV1",
        enum_names: &["AbsoluteFinalInputContinuityStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "terminal_absolute_final_input_continuity_authority_v1",
        file_rel: "runtime/ucf-ops/src/terminal_absolute_final_input_continuity_sweep.rs",
        type_name: "TerminalAbsoluteFinalInputContinuityAuthorityV1",
        enum_names: &["TerminalAbsoluteFinalInputContinuityStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "ultimate_terminal_absolute_final_input_continuity_authority_v1",
        file_rel: "runtime/ucf-ops/src/ultimate_terminal_absolute_final_input_continuity_sweep.rs",
        type_name: "UltimateTerminalAbsoluteFinalInputContinuityAuthorityV1",
        enum_names: &["UltimateTerminalAbsoluteFinalInputContinuityStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "canonical_convergence_continuity_authority_v1",
        file_rel: "runtime/ucf-ops/src/canonical_convergence_continuity_sweep.rs",
        type_name: "CanonicalConvergenceContinuityAuthorityV1",
        enum_names: &["CanonicalConvergenceContinuityStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "canonical_stabilization_continuity_authority_v1",
        file_rel: "runtime/ucf-ops/src/canonical_stabilization_continuity_sweep.rs",
        type_name: "CanonicalStabilizationContinuityAuthorityV1",
        enum_names: &["CanonicalStabilizationContinuityStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "canonical_final_consolidation_continuity_authority_v1",
        file_rel: "runtime/ucf-ops/src/canonical_final_consolidation_continuity_sweep.rs",
        type_name: "CanonicalFinalConsolidationContinuityAuthorityV1",
        enum_names: &["CanonicalFinalConsolidationContinuityStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "final_input_continuity_authority_v1",
        file_rel: "runtime/ucf-ops/src/final_input_continuity_sweep.rs",
        type_name: "FinalInputContinuityAuthorityV1",
        enum_names: &[
            "FinalInputContinuityStatusV1",
            "FinalInputContinuityMismatchCategoryV1",
        ],
    },
    ArtifactSpec {
        artifact_id: "final_governance_consumer_authority_v1",
        file_rel: "runtime/ucf-ops/src/final_governance_consumer_sweep.rs",
        type_name: "FinalGovernanceConsumerAuthorityV1",
        enum_names: &["FinalGovernanceConsumerAuthorityStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "final_governance_residual_sweep_v1",
        file_rel: "runtime/ucf-ops/src/governance_residual_sweep.rs",
        type_name: "FinalGovernanceResidualSweepV1",
        enum_names: &[
            "GovernanceResidualSweepStatusV1",
            "GovernanceResidualMismatchCategoryV1",
        ],
    },
    ArtifactSpec {
        artifact_id: "residual_free_governance_consumer_authority_v1",
        file_rel: "runtime/ucf-ops/src/residual_free_governance_sweep.rs",
        type_name: "ResidualFreeGovernanceConsumerAuthorityV1",
        enum_names: &[
            "ResidualFreeGovernanceConsumerAuthorityStatusV1",
            "ResidualFreeGovernanceMismatchCategoryV1",
        ],
    },
    ArtifactSpec {
        artifact_id: "residual_free_governance_absolute_sweep_v1",
        file_rel: "runtime/ucf-ops/src/governance_absolute_sweep.rs",
        type_name: "ResidualFreeGovernanceAbsoluteSweepV1",
        enum_names: &["ResidualFreeGovernanceAbsoluteSweepStatusV1"],
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
        artifact_id: "final_primary_semantics_consumer_authority_v1",
        file_rel: "runtime/ucf-ops/src/remediation_consistency.rs",
        type_name: "FinalPrimarySemanticsConsumerAuthorityV1",
        enum_names: &["FinalPrimarySemanticsConsumerAuthorityStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "final_primary_semantics_residual_sweep_v1",
        file_rel: "runtime/ucf-ops/src/primary_semantics_residual_sweep.rs",
        type_name: "FinalPrimarySemanticsResidualSweepV1",
        enum_names: &[
            "FinalPrimarySemanticsResidualSweepStatusV1",
            "FinalPrimarySemanticsResidualMismatchCategoryV1",
        ],
    },
    ArtifactSpec {
        artifact_id: "residual_free_primary_semantics_consumer_authority_v1",
        file_rel: "runtime/ucf-ops/src/residual_free_primary_semantics_sweep.rs",
        type_name: "ResidualFreePrimarySemanticsConsumerAuthorityV1",
        enum_names: &[
            "ResidualFreePrimarySemanticsAuthorityStatusV1",
            "ResidualFreePrimarySemanticsMismatchCategoryV1",
        ],
    },
    ArtifactSpec {
        artifact_id: "residual_free_primary_semantics_absolute_sweep_v1",
        file_rel: "runtime/ucf-ops/src/primary_semantics_absolute_sweep.rs",
        type_name: "ResidualFreePrimarySemanticsAbsoluteSweepV1",
        enum_names: &["ResidualFreePrimarySemanticsAbsoluteSweepStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "absolute_final_primary_semantics_terminal_sweep_v1",
        file_rel: "runtime/ucf-ops/src/primary_semantics_terminal_sweep.rs",
        type_name: "AbsoluteFinalPrimarySemanticsTerminalSweepV1",
        enum_names: &["AbsoluteFinalPrimarySemanticsTerminalSweepStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "terminal_primary_semantics_ultimate_sweep_v1",
        file_rel: "runtime/ucf-ops/src/primary_semantics_ultimate_sweep.rs",
        type_name: "TerminalPrimarySemanticsUltimateSweepV1",
        enum_names: &["TerminalPrimarySemanticsUltimateSweepStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "primary_semantics_convergence_sweep_v1",
        file_rel: "runtime/ucf-ops/src/primary_semantics_convergence_sweep.rs",
        type_name: "PrimarySemanticsConvergenceSweepV1",
        enum_names: &[
            "PrimarySemanticsConvergenceStatusV1",
            "PrimarySemanticsConvergenceMismatchCategoryV1",
        ],
    },
    ArtifactSpec {
        artifact_id: "primary_semantics_stabilization_sweep_v1",
        file_rel: "runtime/ucf-ops/src/primary_semantics_stabilization_sweep.rs",
        type_name: "PrimarySemanticsStabilizationSweepV1",
        enum_names: &[
            "PrimarySemanticsStabilizationStatusV1",
            "PrimarySemanticsStabilizationMismatchCategoryV1",
        ],
    },
    ArtifactSpec {
        artifact_id: "primary_semantics_final_consolidation_sweep_v1",
        file_rel: "runtime/ucf-ops/src/primary_semantics_final_consolidation_sweep.rs",
        type_name: "PrimarySemanticsFinalConsolidationSweepV1",
        enum_names: &[
            "PrimarySemanticsFinalConsolidationStatusV1",
            "PrimarySemanticsFinalConsolidationMismatchCategoryV1",
        ],
    },
    ArtifactSpec {
        artifact_id: "absolute_final_readiness_terminal_sweep_v1",
        file_rel: "runtime/ucf-ops/src/readiness_terminal_sweep.rs",
        type_name: "AbsoluteFinalReadinessTerminalSweepV1",
        enum_names: &["AbsoluteFinalReadinessTerminalSweepStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "terminal_readiness_ultimate_sweep_v1",
        file_rel: "runtime/ucf-ops/src/readiness_ultimate_sweep.rs",
        type_name: "TerminalReadinessUltimateSweepV1",
        enum_names: &["TerminalReadinessUltimateSweepStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "readiness_convergence_sweep_v1",
        file_rel: "runtime/ucf-ops/src/readiness_convergence_sweep.rs",
        type_name: "ReadinessConvergenceSweepV1",
        enum_names: &[
            "ReadinessConvergenceStatusV1",
            "ReadinessConvergenceMismatchCategoryV1",
        ],
    },
    ArtifactSpec {
        artifact_id: "readiness_stabilization_sweep_v1",
        file_rel: "runtime/ucf-ops/src/readiness_stabilization_sweep.rs",
        type_name: "ReadinessStabilizationSweepV1",
        enum_names: &[
            "ReadinessStabilizationStatusV1",
            "ReadinessStabilizationMismatchCategoryV1",
        ],
    },
    ArtifactSpec {
        artifact_id: "readiness_final_consolidation_sweep_v1",
        file_rel: "runtime/ucf-ops/src/readiness_final_consolidation_sweep.rs",
        type_name: "ReadinessFinalConsolidationSweepV1",
        enum_names: &[
            "ReadinessFinalConsolidationStatusV1",
            "ReadinessFinalConsolidationMismatchCategoryV1",
        ],
    },
    ArtifactSpec {
        artifact_id: "final_readiness_consumer_authority_v1",
        file_rel: "runtime/ucf-ops/src/final_readiness_consumer_sweep.rs",
        type_name: "FinalReadinessConsumerAuthorityV1",
        enum_names: &["FinalReadinessConsumerAuthorityStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "final_readiness_residual_sweep_v1",
        file_rel: "runtime/ucf-ops/src/readiness_residual_sweep.rs",
        type_name: "FinalReadinessResidualSweepV1",
        enum_names: &[
            "FinalReadinessResidualSweepStatusV1",
            "FinalReadinessResidualMismatchCategoryV1",
        ],
    },
    ArtifactSpec {
        artifact_id: "residual_free_readiness_consumer_authority_v1",
        file_rel: "runtime/ucf-ops/src/residual_free_readiness_sweep.rs",
        type_name: "ResidualFreeReadinessConsumerAuthorityV1",
        enum_names: &[
            "ResidualFreeReadinessConsumerAuthorityStatusV1",
            "ResidualFreeReadinessMismatchCategoryV1",
        ],
    },
    ArtifactSpec {
        artifact_id: "residual_free_readiness_absolute_sweep_v1",
        file_rel: "runtime/ucf-ops/src/readiness_absolute_sweep.rs",
        type_name: "ResidualFreeReadinessAbsoluteSweepV1",
        enum_names: &["ResidualFreeReadinessAbsoluteSweepStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "residual_free_bundle_consumer_authority_v1",
        file_rel: "runtime/ucf-ops/src/residual_free_bundle_sweep.rs",
        type_name: "ResidualFreeBundleConsumerAuthorityV1",
        enum_names: &[
            "ResidualFreeBundleConsumerAuthorityStatusV1",
            "ResidualFreeBundleMismatchCategoryV1",
        ],
    },
    ArtifactSpec {
        artifact_id: "residual_free_bundle_absolute_sweep_v1",
        file_rel: "runtime/ucf-ops/src/bundle_absolute_sweep.rs",
        type_name: "ResidualFreeBundleAbsoluteSweepV1",
        enum_names: &["ResidualFreeBundleAbsoluteSweepStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "absolute_final_bundle_terminal_sweep_v1",
        file_rel: "runtime/ucf-ops/src/bundle_terminal_sweep.rs",
        type_name: "AbsoluteFinalBundleTerminalSweepV1",
        enum_names: &["AbsoluteFinalBundleTerminalSweepStatusV1"],
    },
    ArtifactSpec {
        artifact_id: "terminal_bundle_ultimate_sweep_v1",
        file_rel: "runtime/ucf-ops/src/bundle_ultimate_sweep.rs",
        type_name: "TerminalBundleUltimateSweepV1",
        enum_names: &["TerminalBundleUltimateSweepStatusV1"],
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
    ArtifactSpec {
        artifact_id: "v9_gate_report_v1",
        file_rel: "runtime/ucf-ops/src/v9_gate.rs",
        type_name: "V9GateReportV1",
        enum_names: &["V9GateOverallStatus"],
    },
    ArtifactSpec {
        artifact_id: "v10_gate_report_v1",
        file_rel: "runtime/ucf-ops/src/v10_gate.rs",
        type_name: "V10GateReportV1",
        enum_names: &["V10GateOverallStatus"],
    },
    ArtifactSpec {
        artifact_id: "v11_gate_report_v1",
        file_rel: "runtime/ucf-ops/src/v11_gate.rs",
        type_name: "V11GateReportV1",
        enum_names: &["V11GateOverallStatus"],
    },
    ArtifactSpec {
        artifact_id: "v12_gate_report_v1",
        file_rel: "runtime/ucf-ops/src/v12_gate.rs",
        type_name: "V12GateReportV1",
        enum_names: &["V12GateOverallStatus"],
    },
    ArtifactSpec {
        artifact_id: "v13_gate_report_v1",
        file_rel: "runtime/ucf-ops/src/v13_gate.rs",
        type_name: "V13GateReportV1",
        enum_names: &["V13GateOverallStatus"],
    },
    ArtifactSpec {
        artifact_id: "v14_gate_report_v1",
        file_rel: "runtime/ucf-ops/src/v14_gate.rs",
        type_name: "V14GateReportV1",
        enum_names: &["V14GateOverallStatus"],
    },
    ArtifactSpec {
        artifact_id: "v15_gate_report_v1",
        file_rel: "runtime/ucf-ops/src/v15_gate.rs",
        type_name: "V15GateReportV1",
        enum_names: &["V15GateOverallStatus"],
    },
    ArtifactSpec {
        artifact_id: "v16_gate_report_v1",
        file_rel: "runtime/ucf-ops/src/v16_gate.rs",
        type_name: "V16GateReportV1",
        enum_names: &["V16GateOverallStatus"],
    },
    ArtifactSpec {
        artifact_id: "v17_gate_report_v1",
        file_rel: "runtime/ucf-ops/src/v17_gate.rs",
        type_name: "V17GateReportV1",
        enum_names: &["V17GateOverallStatus"],
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
                "absolute_final_bundle_terminal_sweep_v1",
                "absolute_final_governance_terminal_sweep_v1",
                "absolute_final_input_continuity_authority_v1",
                "absolute_final_primary_semantics_terminal_sweep_v1",
                "absolute_final_readiness_terminal_sweep_v1",
                "active_review_snapshot_v1",
                "applied_scope_authority_v1",
                "applied_supported_set_context_v1",
                "backend_evidence_snapshot_v1",
                "backend_resolution_v1",
                "bugkit_manifest_v1",
                "bundle_convergence_sweep_v1",
                "bundle_final_consolidation_sweep_v1",
                "bundle_roundtrip_consistency_v1",
                "bundle_stabilization_sweep_v1",
                "canonical_bundle_authority_v2",
                "canonical_bundle_consumption_context_v1",
                "canonical_bundle_spine_v1",
                "canonical_continuity_authority_v1",
                "canonical_convergence_continuity_authority_v1",
                "canonical_export_artifact_ref_v1",
                "canonical_export_context_v1",
                "canonical_final_consolidation_continuity_authority_v1",
                "canonical_governance_entry_authority_v2",
                "canonical_governance_entry_v1",
                "canonical_primary_semantics_authority_v1",
                "canonical_readiness_authority_v2",
                "canonical_readiness_spine_v1",
                "canonical_roundtrip_chain_v1",
                "canonical_stabilization_continuity_authority_v1",
                "cross_surface_condition_observation_v1",
                "cross_surface_context_matrix_v1",
                "final_bundle_consumer_authority_v1",
                "final_bundle_residual_sweep_v1",
                "final_continuity_authority_v2",
                "final_governance_consumer_authority_v1",
                "final_governance_residual_sweep_v1",
                "final_input_continuity_authority_v1",
                "final_primary_semantics_consumer_authority_v1",
                "final_primary_semantics_residual_sweep_v1",
                "final_readiness_consumer_authority_v1",
                "final_readiness_residual_sweep_v1",
                "governance_convergence_sweep_v1",
                "governance_final_consolidation_sweep_v1",
                "governance_primary_surfaces_v1",
                "governance_stabilization_sweep_v1",
                "interop_consistency_matrix_report_v1",
                "operator_report_v1",
                "operator_review_packet_v1",
                "operator_signoff_v1",
                "operator_workflow_chain_v1",
                "primary_semantics_convergence_sweep_v1",
                "primary_semantics_final_consolidation_sweep_v1",
                "primary_semantics_observation_v1",
                "primary_semantics_stabilization_sweep_v1",
                "readiness_convergence_sweep_v1",
                "readiness_final_consolidation_sweep_v1",
                "readiness_gate_report_v1",
                "readiness_stabilization_sweep_v1",
                "remediation_consistency_check_v1",
                "repro_pack_manifest_v1",
                "residual_free_bundle_absolute_sweep_v1",
                "residual_free_bundle_consumer_authority_v1",
                "residual_free_continuity_authority_v1",
                "residual_free_governance_absolute_sweep_v1",
                "residual_free_governance_consumer_authority_v1",
                "residual_free_primary_semantics_absolute_sweep_v1",
                "residual_free_primary_semantics_consumer_authority_v1",
                "residual_free_readiness_absolute_sweep_v1",
                "residual_free_readiness_consumer_authority_v1",
                "reviewability_reduction_v1",
                "slot_reviewability_truth_v1",
                "spine_condition_observation_v1",
                "strict_failure_report_v3",
                "supported_real_slot_set_v2",
                "supported_scope_execution_v10",
                "supported_scope_execution_v11",
                "supported_scope_execution_v12",
                "supported_scope_execution_v13",
                "supported_scope_execution_v14",
                "supported_scope_execution_v3",
                "supported_scope_execution_v4",
                "supported_scope_execution_v5",
                "supported_scope_execution_v6",
                "supported_scope_execution_v7",
                "supported_scope_execution_v8",
                "supported_scope_execution_v9",
                "supported_scope_reevaluation_v1",
                "terminal_absolute_final_input_continuity_authority_v1",
                "terminal_bundle_ultimate_sweep_v1",
                "terminal_governance_ultimate_sweep_v1",
                "terminal_primary_semantics_ultimate_sweep_v1",
                "terminal_readiness_ultimate_sweep_v1",
                "ultimate_terminal_absolute_final_input_continuity_authority_v1",
                "v10_gate_report_v1",
                "v11_gate_report_v1",
                "v12_gate_report_v1",
                "v13_gate_report_v1",
                "v14_gate_report_v1",
                "v15_gate_report_v1",
                "v16_gate_report_v1",
                "v17_gate_report_v1",
                "v3_gate_report_v1",
                "v4_gate_report_v1",
                "v5_gate_report_v1",
                "v7_gate_report_v1",
                "v8_gate_report_v1",
                "v9_gate_report_v1",
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

    #[test]
    fn check_reports_missing_v11_residual_snapshot_as_breaking() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("runtime parent")
            .parent()
            .expect("repo root")
            .to_path_buf();
        let tmp = tempfile::tempdir().expect("tempdir");
        generate_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root: repo_root.clone(),
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("generate snapshots");
        fs::remove_file(tmp.path().join("final_governance_residual_sweep_v1.json"))
            .expect("remove snapshot");

        let report = check_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root,
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("check should complete");
        assert!(!report.ok);
        assert!(report.diffs.iter().any(|d| {
            d.artifact == "final_governance_residual_sweep_v1"
                && d.drift_kind == DriftKind::Breaking
        }));
    }

    #[test]
    fn check_reports_missing_v12_residual_free_snapshot_as_breaking() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("runtime parent")
            .parent()
            .expect("repo root")
            .to_path_buf();
        let tmp = tempfile::tempdir().expect("tempdir");
        generate_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root: repo_root.clone(),
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("generate snapshots");
        fs::remove_file(
            tmp.path()
                .join("residual_free_governance_consumer_authority_v1.json"),
        )
        .expect("remove snapshot");

        let report = check_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root,
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("check should complete");
        assert!(!report.ok);
        assert!(report.diffs.iter().any(|d| {
            d.artifact == "residual_free_governance_consumer_authority_v1"
                && d.drift_kind == DriftKind::Breaking
        }));
    }

    #[test]
    fn check_reports_missing_v13_absolute_residual_free_snapshot_as_breaking() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("runtime parent")
            .parent()
            .expect("repo root")
            .to_path_buf();
        let tmp = tempfile::tempdir().expect("tempdir");
        generate_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root: repo_root.clone(),
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("generate snapshots");
        fs::remove_file(
            tmp.path()
                .join("residual_free_governance_absolute_sweep_v1.json"),
        )
        .expect("remove snapshot");

        let report = check_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root,
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("check should complete");
        assert!(!report.ok);
        assert!(report.diffs.iter().any(|d| {
            d.artifact == "residual_free_governance_absolute_sweep_v1"
                && d.drift_kind == DriftKind::Breaking
        }));
    }

    #[test]
    fn generator_then_check_passes_for_full_v18_coverage() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("runtime parent")
            .parent()
            .expect("repo root")
            .to_path_buf();
        let tmp = tempfile::tempdir().expect("tempdir");
        generate_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root: repo_root.clone(),
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("generate snapshots");

        let report = check_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root,
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("check should complete");
        assert!(report.ok, "expected no drift, got: {:?}", report.diffs);
    }

    #[test]
    fn check_reports_missing_v15_ultimate_snapshot_as_breaking() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("runtime parent")
            .parent()
            .expect("repo root")
            .to_path_buf();
        let tmp = tempfile::tempdir().expect("tempdir");
        generate_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root: repo_root.clone(),
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("generate snapshots");
        fs::remove_file(
            tmp.path()
                .join("terminal_governance_ultimate_sweep_v1.json"),
        )
        .expect("remove snapshot");

        let report = check_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root,
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("check should complete");
        assert!(!report.ok);
        assert!(report.diffs.iter().any(|d| {
            d.artifact == "terminal_governance_ultimate_sweep_v1"
                && d.drift_kind == DriftKind::Breaking
        }));
    }

    #[test]
    fn check_reports_v11_residual_shape_drift() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("runtime parent")
            .parent()
            .expect("repo root")
            .to_path_buf();
        let tmp = tempfile::tempdir().expect("tempdir");
        generate_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root: repo_root.clone(),
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("generate snapshots");

        let snapshot_path = tmp.path().join("final_readiness_residual_sweep_v1.json");
        let mut snapshot: ArtifactSchemaSnapshot =
            serde_json::from_str(&fs::read_to_string(&snapshot_path).expect("read snapshot"))
                .expect("parse snapshot");
        snapshot
            .field_types
            .insert("sweep_digest".to_string(), "u64".to_string());
        fs::write(
            &snapshot_path,
            serde_json::to_string_pretty(&snapshot).expect("serialize snapshot"),
        )
        .expect("write snapshot");

        let report = check_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root,
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("check should complete");
        assert!(!report.ok);
        assert!(report.diffs.iter().any(|d| {
            d.artifact == "final_readiness_residual_sweep_v1"
                && d.drift_kind == DriftKind::Breaking
                && d.summary.contains("field type changed")
        }));
    }

    #[test]
    fn check_reports_v13_supported_scope_execution_shape_drift() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("runtime parent")
            .parent()
            .expect("repo root")
            .to_path_buf();
        let tmp = tempfile::tempdir().expect("tempdir");
        generate_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root: repo_root.clone(),
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("generate snapshots");

        let snapshot_path = tmp.path().join("supported_scope_execution_v8.json");
        let mut snapshot: ArtifactSchemaSnapshot =
            serde_json::from_str(&fs::read_to_string(&snapshot_path).expect("read snapshot"))
                .expect("parse snapshot");
        snapshot.field_types.insert(
            "resulting_supported_set_digest_prefix".to_string(),
            "u32".to_string(),
        );
        fs::write(
            &snapshot_path,
            serde_json::to_string_pretty(&snapshot).expect("serialize snapshot"),
        )
        .expect("write snapshot");

        let report = check_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root,
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("check should complete");
        assert!(!report.ok);
        assert!(report.diffs.iter().any(|d| {
            d.artifact == "supported_scope_execution_v8"
                && d.drift_kind == DriftKind::Breaking
                && d.summary.contains("field type changed")
        }));
    }

    #[test]
    fn check_reports_v15_supported_scope_execution_shape_drift() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("runtime parent")
            .parent()
            .expect("repo root")
            .to_path_buf();
        let tmp = tempfile::tempdir().expect("tempdir");
        generate_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root: repo_root.clone(),
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("generate snapshots");

        let snapshot_path = tmp.path().join("supported_scope_execution_v10.json");
        let mut snapshot: ArtifactSchemaSnapshot =
            serde_json::from_str(&fs::read_to_string(&snapshot_path).expect("read snapshot"))
                .expect("parse snapshot");
        snapshot.field_types.insert(
            "terminal_governance_ultimate_sweep_digest_prefix".to_string(),
            "u32".to_string(),
        );
        fs::write(
            &snapshot_path,
            serde_json::to_string_pretty(&snapshot).expect("serialize snapshot"),
        )
        .expect("write snapshot");

        let report = check_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root,
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("check should complete");
        assert!(!report.ok);
        assert!(report.diffs.iter().any(|d| {
            d.artifact == "supported_scope_execution_v10"
                && d.drift_kind == DriftKind::Breaking
                && d.summary.contains("field type changed")
        }));
    }

    #[test]
    fn check_reports_missing_v16_governance_convergence_snapshot_as_breaking() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("runtime parent")
            .parent()
            .expect("repo root")
            .to_path_buf();
        let tmp = tempfile::tempdir().expect("tempdir");
        generate_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root: repo_root.clone(),
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("generate snapshots");
        fs::remove_file(tmp.path().join("governance_convergence_sweep_v1.json"))
            .expect("remove snapshot");

        let report = check_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root,
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("check should complete");
        assert!(!report.ok);
        assert!(report.diffs.iter().any(|d| {
            d.artifact == "governance_convergence_sweep_v1" && d.drift_kind == DriftKind::Breaking
        }));
    }

    #[test]
    fn check_reports_v16_governance_convergence_shape_drift() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("runtime parent")
            .parent()
            .expect("repo root")
            .to_path_buf();
        let tmp = tempfile::tempdir().expect("tempdir");
        generate_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root: repo_root.clone(),
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("generate snapshots");

        let snapshot_path = tmp.path().join("governance_convergence_sweep_v1.json");
        let mut snapshot: ArtifactSchemaSnapshot =
            serde_json::from_str(&fs::read_to_string(&snapshot_path).expect("read snapshot"))
                .expect("parse snapshot");
        snapshot
            .field_types
            .insert("convergence_digest".to_string(), "u64".to_string());
        fs::write(
            &snapshot_path,
            serde_json::to_string_pretty(&snapshot).expect("serialize snapshot"),
        )
        .expect("write snapshot");

        let report = check_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root,
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("check should complete");
        assert!(!report.ok);
        assert!(report.diffs.iter().any(|d| {
            d.artifact == "governance_convergence_sweep_v1"
                && d.drift_kind == DriftKind::Breaking
                && d.summary.contains("field type changed")
        }));
    }

    #[test]
    fn check_reports_missing_v17_primary_semantics_stabilization_snapshot_as_breaking() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("runtime parent")
            .parent()
            .expect("repo root")
            .to_path_buf();
        let tmp = tempfile::tempdir().expect("tempdir");
        generate_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root: repo_root.clone(),
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("generate snapshots");
        fs::remove_file(
            tmp.path()
                .join("primary_semantics_stabilization_sweep_v1.json"),
        )
        .expect("remove snapshot");

        let report = check_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root,
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("check should complete");
        assert!(!report.ok);
        assert!(report.diffs.iter().any(|d| {
            d.artifact == "primary_semantics_stabilization_sweep_v1"
                && d.drift_kind == DriftKind::Breaking
        }));
    }

    #[test]
    fn check_reports_v17_primary_semantics_stabilization_shape_drift() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("runtime parent")
            .parent()
            .expect("repo root")
            .to_path_buf();
        let tmp = tempfile::tempdir().expect("tempdir");
        generate_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root: repo_root.clone(),
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("generate snapshots");

        let snapshot_path = tmp
            .path()
            .join("primary_semantics_stabilization_sweep_v1.json");
        let mut snapshot: ArtifactSchemaSnapshot =
            serde_json::from_str(&fs::read_to_string(&snapshot_path).expect("read snapshot"))
                .expect("parse snapshot");
        snapshot
            .field_types
            .insert("stabilization_digest".to_string(), "u64".to_string());
        fs::write(
            &snapshot_path,
            serde_json::to_string_pretty(&snapshot).expect("serialize snapshot"),
        )
        .expect("write snapshot");

        let report = check_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root,
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("check should complete");
        assert!(!report.ok);
        assert!(report.diffs.iter().any(|d| {
            d.artifact == "primary_semantics_stabilization_sweep_v1"
                && d.drift_kind == DriftKind::Breaking
                && d.summary.contains("field type changed")
        }));
    }

    #[test]
    fn check_reports_missing_v18_final_consolidation_snapshot_as_breaking() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("runtime parent")
            .parent()
            .expect("repo root")
            .to_path_buf();
        let tmp = tempfile::tempdir().expect("tempdir");
        generate_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root: repo_root.clone(),
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("generate snapshots");
        fs::remove_file(
            tmp.path()
                .join("governance_final_consolidation_sweep_v1.json"),
        )
        .expect("remove snapshot");

        let report = check_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root,
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("check should complete");
        assert!(!report.ok);
        assert!(report.diffs.iter().any(|d| {
            d.artifact == "governance_final_consolidation_sweep_v1"
                && d.drift_kind == DriftKind::Breaking
        }));
    }

    #[test]
    fn check_reports_v18_supported_scope_execution_shape_drift() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("runtime parent")
            .parent()
            .expect("repo root")
            .to_path_buf();
        let tmp = tempfile::tempdir().expect("tempdir");
        generate_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root: repo_root.clone(),
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("generate snapshots");

        let snapshot_path = tmp.path().join("supported_scope_execution_v13.json");
        let mut snapshot: ArtifactSchemaSnapshot =
            serde_json::from_str(&fs::read_to_string(&snapshot_path).expect("read snapshot"))
                .expect("parse snapshot");
        snapshot.field_types.insert(
            "governance_final_consolidation_sweep_digest_prefix".to_string(),
            "u32".to_string(),
        );
        fs::write(
            &snapshot_path,
            serde_json::to_string_pretty(&snapshot).expect("serialize snapshot"),
        )
        .expect("write snapshot");

        let report = check_artifact_schema_snapshots(&ArtifactSchemaArgs {
            repo_root,
            out_dir: tmp.path().to_path_buf(),
        })
        .expect("check should complete");
        assert!(!report.ok);
        assert!(report.diffs.iter().any(|d| {
            d.artifact == "supported_scope_execution_v13"
                && d.drift_kind == DriftKind::Breaking
                && d.summary.contains("field type changed")
        }));
    }
}
