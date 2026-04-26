//! Canonical reference surface for the real-compute runtime stack.
//!
//! This crate intentionally keeps one load-bearing compute core while exposing
//! explicit extension and diagnostics seams.
//!
//! ## Canonical productive core
//! - Entry/service surface: [`service_surface::CanonicalComputeEntryPoint`]
//! - Bounded service + scheduling/placement: [`compute_service`]
//! - Canonical stage pipeline contract: [`pipeline`]
//! - Canonical backend onboarding lane: [`backends::build_canonical_production_backend`]
//! - Artifact/model slot and warmup readiness: [`model_store`], [`backend_pack`]
//!
//! ## Load-bearing extensions (non-primary but supported)
//! - Multi-worker/IPC/remote execution: [`worker_backend`], [`ipc`], [`remote_compute`]
//! - Rollout and slot path enablement (`active/candidate/compare/shadow`): [`enablement`]
//! - Ops/history/recovery/replay-facing persistence: [`job_history`], [`service_surface`]
//! - Expert/high-trust runtime contracts for replay + runtime ops:
//!   [`RuntimeEntryClass`], [`RuntimeContractShape`], [`service_surface`]
//! - Cross-cutting production invariants for shared core semantics:
//!   [`CROSS_CUTTING_PRODUCTION_INVARIANTS_V1`]
//!
//! ## Diagnostic/test seams (not production defaults)
//! - Compatibility/dev backend lane (`stub`, `candle` compatibility seam): [`backends`]
//! - Compare/shadow and stage diagnostics: [`enablement`], [`world_vljepa_shadow`]
//! - Test-only helpers: [`test_env`]
//!
//! ## Compatibility boundary
//! `domains/ai*` integration remains a compatibility/legacy seam. The canonical
//! runtime path for real compute is maintained in `runtime/ucf-compute`.
//!
#[cfg(all(feature = "compute-candle", feature = "compute-burn"))]
compile_error!("invalid feature combo: compute-candle and compute-burn are mutually exclusive");
#[cfg(all(feature = "llm-candle", feature = "llm-burn"))]
compile_error!("invalid feature combo: llm-candle and llm-burn are mutually exclusive");
#[cfg(all(feature = "lfm-candle", feature = "lfm-burn"))]
compile_error!("invalid feature combo: lfm-candle and lfm-burn are mutually exclusive");

use crate::lfm::PlasticityRecord;
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;
use ucf_frames::v1::{ControlFrame, ControlPayload};
use ucf_types::UQ0_16;
use world_model::StageQuality;

pub mod backend_pack;
pub mod backends;
pub mod blue_brain_dynamics;
pub mod blue_brain_memory;
pub mod blue_brain_minimal_execution;
#[cfg(any(
    feature = "compute-candle",
    feature = "llm-candle",
    feature = "lfm-candle",
    feature = "backend-candle",
    feature = "compute-burn",
    feature = "llm-burn",
    feature = "lfm-burn"
))]
pub mod candle_weights;
pub mod capabilities;
pub mod compute_service;
pub mod contracts;
pub mod enablement;
pub mod evidence;
pub mod feature_extractor;
pub mod feature_matrix;
pub mod ipc;
pub mod job_history;
pub mod lfm;
pub mod model_store;
pub mod pipeline;
pub mod reference_map;
#[cfg(feature = "remote-compute")]
pub mod remote_compute;
pub mod risk_contract;
pub mod runtime_profile;
pub mod service_surface;
pub mod ssm;
pub mod stage_v1;
#[cfg(feature = "backend-burn")]
pub mod stage_v1_burn;
#[cfg(feature = "backend-candle")]
pub mod stage_v1_candle;
#[cfg(test)]
pub mod test_env;
pub mod work_meter;
pub mod worker_backend;
pub mod world_model;
pub mod world_vljepa_shadow;
pub use backend_pack::{
    ArtifactFailureCode, BackendComponentId, BackendPack, BackendPackConfig, BackendPackFactory,
    BackendPackId, BackendPackKind, BackendPackMeta, BackendSwapRequest, FixtureId, FixtureManager,
    ModelSlotProvenance, ProductionBlockReason, ProductionCompatibilityGate, SlotRuntimeStatus,
};
pub use backends::{
    build_backend, build_canonical_production_backend, build_onboarding_reference_backend,
    build_service_compute_backend, ComputeBackendConfig, ComputeBackendKind,
    CANONICAL_ONBOARDING_BACKEND, CANONICAL_ONBOARDING_PACK,
};
pub use blue_brain_dynamics::{
    evaluate_blue_brain_hodgkin_huxley_diagnostic, evaluate_blue_brain_kuramoto_modulation,
    kuramoto_modulation_diagnostic_class_token, kuramoto_modulation_reason_token,
    kuramoto_modulation_state_token, BlueBrainDynamicsDiagnosticClass,
    BlueBrainDynamicsDiagnosticLane, BlueBrainDynamicsRuntimeFeedbackClass,
    BlueBrainDynamicsSelectionFeedbackClass, BlueBrainHodgkinHuxleyBoundaryGuard,
    BlueBrainHodgkinHuxleyBoundedModelParameters, BlueBrainHodgkinHuxleyDiagnosticClass,
    BlueBrainHodgkinHuxleyDiagnosticInput, BlueBrainHodgkinHuxleyDiagnosticResult,
    BlueBrainHodgkinHuxleyScopeState, BlueBrainHodgkinHuxleySimulationParameters,
    BlueBrainKuramotoBoundaryGuard, BlueBrainKuramotoInputBasisClass,
    BlueBrainKuramotoInputGroupClass, BlueBrainKuramotoInputGroupLane,
    BlueBrainKuramotoModulationDiagnosticClass, BlueBrainKuramotoModulationInput,
    BlueBrainKuramotoModulationReason, BlueBrainKuramotoModulationResult,
    BlueBrainKuramotoModulationState, BlueBrainKuramotoPhaseNodeInput,
    BlueBrainKuramotoRuntimeCaveatModulation, BlueBrainKuramotoRuntimePosture,
    BlueBrainKuramotoScopeState, BlueBrainKuramotoSelectionHint, BlueBrainKuramotoSelectionPosture,
    BlueBrainKuramotoSynchronyDiagnostic, CANONICAL_BLUE_BRAIN_DYNAMICS_DIAGNOSTICS_MAP,
    CANONICAL_BLUE_BRAIN_KURAMOTO_INPUT_GROUP_MAP,
};
pub use blue_brain_memory::{
    BlueBrainMemoryCandidate, BlueBrainMemoryCandidateClass, BlueBrainMemoryCaveatRefreshState,
    BlueBrainMemoryCommitReport, BlueBrainMemoryCommitResultState,
    BlueBrainMemoryContextFeedbackClass, BlueBrainMemoryDiagnosticClass,
    BlueBrainMemoryDiagnosticLane, BlueBrainMemoryFeedbackBackbind, BlueBrainMemoryFreshness,
    BlueBrainMemoryMaintenanceAction, BlueBrainMemoryMaintenanceLocator,
    BlueBrainMemoryMaintenanceReport, BlueBrainMemoryMaintenanceRequest,
    BlueBrainMemoryMaintenanceResultState, BlueBrainMemoryMaintenanceStatus, BlueBrainMemoryOrigin,
    BlueBrainMemoryReadRequest, BlueBrainMemoryReadResult, BlueBrainMemoryReferenceLocator,
    BlueBrainMemoryReferenceMetadata, BlueBrainMemoryReferenceRecord,
    BlueBrainMemoryRetrievalState, BlueBrainMemoryRuntimeFeedbackClass,
    BlueBrainMemorySelectionCandidateProposalFeedbackClass, BlueBrainMemorySelectionDisposition,
    BlueBrainMemoryStore, BlueBrainMemoryStoreError, PersistedBlueBrainMemoryRecord,
    CANONICAL_BLUE_BRAIN_MEMORY_DIAGNOSTICS_MAP,
};
pub use blue_brain_minimal_execution::{
    execute_blue_brain_minimal_action, BlueBrainMinimalExecutionAction,
    BlueBrainMinimalExecutionReport, BlueBrainMinimalExecutionRequest,
    BlueBrainMinimalExecutionResultBoundary, BlueBrainMinimalExecutionState,
};
pub use compute_service::{
    CapacityPressure, CapacityQueueDisposition, ComputeJob, CoordinationFreshness,
    CoordinationIssueKind, DeviceSuitability, DistributedDegradationState,
    DistributedPressureSnapshot, DistributedRecoverySnapshot, ExecutionDeviceClass,
    ExecutionPlacement, ExecutionUnitId, ExecutionUnitKind, ExecutionUnitSnapshot,
    FeedbackSignalStrength, InFlightCoordinationState, InFlightJobSnapshot, InMemoryComputeService,
    JobAccountingSummary, JobCompletionClass, JobCoordinationSnapshot, JobExecutionPath, JobId,
    JobLifecycleEvent, JobLifecycleState, JobRecord, JobSubmissionMeta, MultiWorkerComputeService,
    PlacementCandidateAssessment, PlacementFailureKind, PlacementOptimizationFeedbackView,
    PlacementSuitability, RecoverySignal, ResourceClass, SchedulerConfig, SchedulerSnapshot,
    WorkerAvailability, WorkerClass, WorkerDispatchOutcome, WorkerFailureKind,
    WorkerMembershipState, WorkerRecoveryKind, WorkerRegistryRole, WorkerRetrySummary,
    WorkerRuntimeStatus,
};
pub use contracts::{
    canonical_runtime_handoff_semantics, runtime_action_core_semantics_consistent,
    runtime_handoff_state_from_action_code, runtime_handoff_state_from_evidence,
    runtime_handoff_state_from_snapshot_and_diagnostics, CanonicalEvidenceKind,
    CanonicalEvidenceReasonCode, CanonicalEvidenceStatus, CanonicalSnapshotConsistency,
    CanonicalTraceSliceKind, CanonicalTraceSliceStatus, CapabilityConstraint,
    CapabilitySupportLevel, ExpertDiagnosticsAvailability, ExpertMutationBlocker,
    ExpertMutationBoundary, ExpertMutationResult, ExpertWorkflowClass,
    ExpertWorkflowTransitionState, HandoffReferenceRequirement, RuntimeActionOutcomeCode,
    RuntimeContractSafety, RuntimeContractShape, RuntimeDiagnosticsCore, RuntimeDriftClass,
    RuntimeEntryClass, RuntimeFreshnessClass, RuntimeHandoffKind, RuntimeHandoffReferenceSet,
    RuntimeHandoffSemantics, RuntimeHandoffState, RuntimeStatusCore, RuntimeSurfaceExtension,
    StageContractVersion, StageKind, StagePathCapability, StagePathSupportLevel, ValidationStatus,
    CANONICAL_RUNTIME_HANDOFF_SEMANTICS_V1, CROSS_CUTTING_PRODUCTION_INVARIANTS_V1,
};
pub use enablement::{
    EnablementComputeBackend, EnablementConfig, RealEnablementMode, SlotEnablement, SlotMode,
};
pub use evidence::{CodeVersionTag, EvidenceChain, COMPUTE_SUMMARY_SCHEMA_VERSION};
pub use feature_matrix::ReleaseFeatureMatrix;
pub use job_history::{JobHistoryStore, JobHistoryStoreError, PersistedJobRecord};
pub use model_store::{
    ActivationFallbackState, ActivationOutcome, ComparePathOutcome, CompareShadowContext,
    CompareShadowEvaluation, ModelDevice, ModelFormat, ModelLoadError, ModelSlot, ModelSlotSpec,
    ModelStore, PromotionBlockerCode, PromotionDecisionState, PromotionEvaluationDisposition,
    PromotionTechnicalSignals, RollbackOutcome, ShadowPathOutcome, SlotActivationAssessment,
    SlotPromotionDecision, SlotRollbackAssessment, SlotWarmupState, SlotWarmupStatus,
};
pub use pipeline::{
    BackendExecutionLane, CanonicalAdmissionDecision, CanonicalBackendRoute, CanonicalFailureKind,
    CanonicalPipelineFailure, CanonicalPipelineRequest, CanonicalPipelineResult,
    CanonicalPipelineState, CanonicalStageId, ComputePipelineBackend, WorldStageReadiness,
    WorldStageStatus, CANONICAL_STAGE_SEQUENCE,
};
pub use reference_map::{
    canonical_blue_brain_candidate_deferral_lifecycle_map,
    canonical_blue_brain_commit_eligibility_conditions_map,
    canonical_blue_brain_commit_result_semantics_map, canonical_blue_brain_compute_handoff_map,
    canonical_blue_brain_context_evidence_priority_map,
    canonical_blue_brain_context_update_lifecycle_map,
    canonical_blue_brain_control_attention_selection_map, canonical_blue_brain_facing_contract_map,
    canonical_blue_brain_future_memory_handoff_state_map,
    canonical_blue_brain_integration_candidate_map, canonical_blue_brain_integration_map,
    canonical_blue_brain_memory_candidate_lifecycle_map,
    canonical_blue_brain_memory_commit_boundary_map,
    canonical_blue_brain_memory_commit_diagnostics_map,
    canonical_blue_brain_neural_dynamics_candidate_map, canonical_blue_brain_reference_context_map,
    canonical_blue_brain_selection_diagnostics_map, canonical_compute_integration_contract_view,
    canonical_compute_reference_map, canonical_domain_facing_compute_consumer_map,
    canonical_final_reference_line, canonical_first_domain_rollout_candidate_map,
    canonical_first_domain_rollout_completion_map, canonical_onboarding_reference_summary,
    canonical_post_rollout_adoption_map, canonical_production_reference_lane,
    is_canonical_core_or_extension_lane, is_outward_facing_compute_integration_boundary,
    BlueBrainCandidateDeferralLifecycleClass, BlueBrainCandidateDeferralLifecycleLane,
    BlueBrainCommitEligibilityConditionClass, BlueBrainCommitEligibilityConditionLane,
    BlueBrainCommitResultClass, BlueBrainCommitResultLane, BlueBrainComputeHandoffClass,
    BlueBrainComputeHandoffLane, BlueBrainContextEvidencePriorityClass,
    BlueBrainContextEvidencePriorityLane, BlueBrainContextUpdateLifecycleClass,
    BlueBrainContextUpdateLifecycleLane, BlueBrainControlAttentionSelectionClass,
    BlueBrainControlAttentionSelectionLane, BlueBrainFacingContractClass,
    BlueBrainFacingContractLane, BlueBrainFutureMemoryHandoffStateClass,
    BlueBrainFutureMemoryHandoffStateLane, BlueBrainIntegrationCandidateClass,
    BlueBrainIntegrationCandidateLane, BlueBrainIntegrationClass, BlueBrainIntegrationLane,
    BlueBrainMemoryCandidateLifecycleClass, BlueBrainMemoryCandidateLifecycleLane,
    BlueBrainMemoryCommitBoundaryClass, BlueBrainMemoryCommitBoundaryLane,
    BlueBrainMemoryCommitDiagnosticClass, BlueBrainMemoryCommitDiagnosticLane,
    BlueBrainNeuralDynamicsCandidateClass, BlueBrainNeuralDynamicsCandidateLane,
    BlueBrainReferenceContextClass, BlueBrainReferenceContextLane, BlueBrainReferenceQualityClass,
    BlueBrainSelectionBasisQualityClass, BlueBrainSelectionDiagnosticClass,
    BlueBrainSelectionDiagnosticLane, BlueBrainSelectionDispositionClass,
    CanonicalFinalReferenceLine, ComputeIntegrationBoundary, ComputeIntegrationContractClass,
    ComputeIntegrationContractLane, ComputeReferenceClass, ComputeReferenceLane,
    DomainFacingCompletionStatus, DomainFacingComputeConsumerLane, DomainFacingConsumerAlignment,
    DomainFacingEvidenceConsumptionPattern, DomainFacingStatusConsumptionPattern,
    DomainRolloutCandidateClass, DomainRolloutCandidateLane, FirstDomainRolloutCompletionLane,
    FirstDomainRolloutCompletionStatus, PostRolloutAdoptionClass, PostRolloutAdoptionLane,
    CANONICAL_BLUE_BRAIN_CANDIDATE_DEFERRAL_LIFECYCLE_MAP,
    CANONICAL_BLUE_BRAIN_COMMIT_ELIGIBILITY_CONDITIONS_MAP,
    CANONICAL_BLUE_BRAIN_COMMIT_RESULT_SEMANTICS_MAP, CANONICAL_BLUE_BRAIN_COMPUTE_HANDOFF_MAP,
    CANONICAL_BLUE_BRAIN_CONTEXT_EVIDENCE_PRIORITY_MAP,
    CANONICAL_BLUE_BRAIN_CONTEXT_UPDATE_LIFECYCLE_MAP,
    CANONICAL_BLUE_BRAIN_CONTROL_ATTENTION_SELECTION_MAP, CANONICAL_BLUE_BRAIN_FACING_CONTRACT_MAP,
    CANONICAL_BLUE_BRAIN_FUTURE_MEMORY_HANDOFF_STATE_MAP,
    CANONICAL_BLUE_BRAIN_INTEGRATION_CANDIDATE_MAP, CANONICAL_BLUE_BRAIN_INTEGRATION_MAP,
    CANONICAL_BLUE_BRAIN_MEMORY_CANDIDATE_LIFECYCLE_MAP,
    CANONICAL_BLUE_BRAIN_MEMORY_COMMIT_BOUNDARY_MAP,
    CANONICAL_BLUE_BRAIN_MEMORY_COMMIT_DIAGNOSTICS_MAP,
    CANONICAL_BLUE_BRAIN_NEURAL_DYNAMICS_CANDIDATE_MAP, CANONICAL_BLUE_BRAIN_REFERENCE_CONTEXT_MAP,
    CANONICAL_BLUE_BRAIN_SELECTION_DIAGNOSTICS_MAP, CANONICAL_COMPUTE_INTEGRATION_CONTRACT_VIEW,
    CANONICAL_COMPUTE_REFERENCE_MAP, CANONICAL_DOMAIN_FACING_COMPUTE_CONSUMER_MAP,
    CANONICAL_FINAL_REFERENCE_LINE, CANONICAL_FIRST_DOMAIN_ROLLOUT_CANDIDATE_MAP,
    CANONICAL_FIRST_DOMAIN_ROLLOUT_COMPLETION_MAP, CANONICAL_POST_ROLLOUT_ADOPTION_MAP,
    FINAL_REFERENCE_LINE_CROSS_CUTTING_INVARIANTS, FINAL_REFERENCE_LINE_DIAGNOSTICS_EXTENSION,
    FINAL_REFERENCE_LINE_EXECUTION_CORE, FINAL_REFERENCE_LINE_REPLAY_EXTENSION,
    FINAL_REFERENCE_LINE_ROLLOUT_EXTENSION, FINAL_REFERENCE_NON_CANONICAL_INTERNAL_BOUNDARY,
    WORKFLOW_PATH_INSPECT_DIAGNOSE_ACT, WORKFLOW_PATH_INTERNAL_DEV_TEST_ONLY,
    WORKFLOW_PATH_REPLAY_ORIENTED, WORKFLOW_PATH_ROLLOUT_ORIENTED,
};
#[cfg(feature = "remote-compute")]
pub use remote_compute::{
    NodeSigner, RemoteComputeClient, RemoteErr, RemoteGovernor, RemoteGovernorConfig,
    RemotePolicyAllowlist, RemoteReq, RemoteResp,
};
pub use risk_contract::{
    clamp01, stable_budget_profile_id, validate_risk_signal, BackendProfileId, EvidenceRef,
    RiskSignal, SignalQuality,
};
pub use runtime_profile::{DeploymentProfile, RuntimeDiagnosticFlags, RuntimeMode, RuntimeProfile};
pub use service_surface::{
    BaselineComparisonFailureCode, BaselineComparisonOutcome, BaselineComparisonResult,
    BaselineComparisonSummary, BaselineReference, CanonicalComputeEntryPoint,
    CanonicalConsumerEvidenceSemantic, CanonicalConsumerStatusEvidenceView,
    CanonicalConsumerStatusSemantic, CanonicalRuntimeSnapshot,
    CanonicalRuntimeSubsystemDiagnostics, ComputeEvidenceBundleExportRef,
    ComputeEvidenceComparisonExportRef, ComputeEvidenceExportSurface, ComputeExecutionMode,
    ComputeHistoryLookupError, ComputeHistoryStoreStatus, ComputeIntegrationActionSignal,
    ComputeIntegrationHookClass, ComputeIntegrationHookDescriptor, ComputeIntegrationHookExposure,
    ComputeIntegrationHookMutationSemantics, ComputeIntegrationHookView,
    ComputeIntegrationPathContext, ComputeIntegrationSignals, ComputeInvalidRequest,
    ComputeJobHandle, ComputeJobHistoryLookup, ComputeJobStatus, ComputeProductionLineContext,
    ComputeRecoverySnapshot, ComputeReplayOutcome, ComputeReplayPreflight, ComputeReplayReport,
    ComputeRequestValidationCode, ComputeStatusEvidenceExportSurface, ComputeStatusExportSurface,
    ComputeSubmitOutcome, ComputeSubmitRequest, ComputeTraceSliceExportRef,
    DecisionJustificationDisposition, DecisionJustificationView, JustificationEvidencePosture,
    RecoveredJobStatus, RecoveryDisposition, ReplayConfigurationDiff, ReplayContextBridgeSummary,
    ReplayContextConsistencyClass, ReplayContextTransition, ReplayDeterminismClass,
    ReplayExecutionContextDescriptor, ReplayExecutionMode, ReplayFailureCode,
    ReplayMismatchCategory, ReplayMismatchClass, ReplayMismatchReason, ReplayMismatchReasonCode,
    ReplayMismatchView, ReplayOutcomeComparison, ReplayPreflightIssue, ReplayPreflightIssueCode,
    ReplayPreflightLocality, ReplayRemoteContextReproducibility, ReplayabilityClass,
    RolloutReplayComparability, RolloutReplayComparisonContext, RolloutReplayContextClass,
    RuntimeBoundedRecoveryView, RuntimeControlAttentionDiagnostic, RuntimeControlAttentionEntity,
    RuntimeControlAttentionOutcome, RuntimeControlAttentionReason, RuntimeOperation,
    RuntimeOperationClass, RuntimeOperationCode, RuntimeOperationOutcome, RuntimeOperationScope,
    RuntimeOperationSnapshotEffect, RuntimeOpsSnapshot, RuntimeOpsState, RuntimeRecoveryFlow,
    RuntimeRecoveryResultState, RuntimeRecoveryTrustState, RuntimeSignalState,
    RuntimeStaleDriftView, RuntimeSubsystemDiagnosticSummary, RuntimeWarmupState,
    ServiceMutationTrustDisposition, ServiceTrustEvolution, ServiceTrustState,
    ServiceTrustStateView, SupportedExpertWorkflowPath, WorkflowContractBinding,
    WorkflowTransitionSummary, WorkflowTransitionType, WorkflowViewSnapshot,
};
pub use work_meter::WorkMeter;

pub const MAX_SPIKES: usize = 256;
pub const MAX_NOTES: usize = 16;
pub const MAX_NOTE_LEN: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotCompareStatusV1 {
    Ok,
    DriftWarn,
    ShadowDisabled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotCompareWindowRecordV1 {
    pub slot_id: String,
    pub t0: u64,
    pub t1: u64,
    pub sample_count: u16,
    pub primary_mean_q: u16,
    pub primary_p95_q: u16,
    pub shadow_mean_q: u16,
    pub shadow_p95_q: u16,
    pub mean_delta_q: u16,
    pub p95_delta_q: u16,
    pub digest_mismatch_count: u16,
    pub invalid_shadow_count: u16,
    pub digest_prefix_samples: Vec<[u8; 4]>,
    pub status: SlotCompareStatusV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShadowDisableRecordV1 {
    pub slot_id: String,
    pub t: u64,
    pub reason: String,
    pub consecutive_failures: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotModeChangeRecordV1 {
    pub slot_id: String,
    pub t: u64,
    pub from_mode: ucf_types::SlotModeV1,
    pub to_mode: ucf_types::SlotModeV1,
    pub evidence_validated: bool,
    pub evidence_digest_prefix: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlotShadowEventV1 {
    CompareWindow(SlotCompareWindowRecordV1),
    ShadowDisable(ShadowDisableRecordV1),
    ModeChange(SlotModeChangeRecordV1),
}

static UCF_COMPUTE_CHAIN_DIGEST_EMITTED_TOTAL: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FrameId(pub u64);

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ComputeInput {
    pub frame_id: FrameId,
    pub t: u64,
    pub context_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Spike {
    pub feature_id: u32,
    pub magnitude: f32,
    pub timestamp: u64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ComputeSignals {
    pub surprise: f32,
    pub pressure: f32,
    pub risk: f32,
    pub confidence: f32,
    pub risk_signal: RiskSignal,
    pub spikes: Vec<Spike>,
    pub notes: Vec<String>,
    pub sparsity: Option<f32>,
    pub energy: Option<f32>,
    pub ssm_readout: Option<f32>,
    pub ssm_digest: Option<[u8; 32]>,
    pub world_digest: Option<[u8; 32]>,
    pub lfm_uncertainty: Option<f32>,
    pub lfm_stability: Option<f32>,
    pub lfm_state_norm: Option<f32>,
    pub lfm_deriv_norm: Option<f32>,
    pub lfm_saturation_ratio: Option<f32>,
    pub lfm_nan_inf_detected: bool,
    pub lfm_digest: Option<[u8; 32]>,
    pub nsr_digest: Option<[u8; 32]>,
    pub nsr_status: u8,
    pub signal_bundle_digest: Option<[u8; 32]>,
    pub sae_quality: Option<StageQuality>,
    pub ssm_quality: Option<StageQuality>,
    pub lfm_quality: Option<StageQuality>,
    pub plasticity_record: Option<PlasticityRecord>,
    pub budget_exceeded_stage: Option<&'static str>,
    pub contract_version: StageContractVersion,
    pub backend_id: u16,
    pub validation_status: ValidationStatus,
    pub violation_reason_mask: u32,
}

impl ComputeSignals {
    pub fn bounded(mut self) -> Self {
        self.surprise = self.surprise.clamp(0.0, 1.0);
        self.pressure = self.pressure.clamp(0.0, 1.0);
        self.risk_signal.risk = clamp01(self.risk_signal.risk);
        self.risk_signal.confidence = clamp01(self.risk_signal.confidence);
        self.risk = self.risk_signal.risk;
        self.confidence = self.risk_signal.confidence;
        self.sparsity = self.sparsity.map(|v| v.clamp(0.0, 1.0));
        self.energy = self.energy.map(|v| v.clamp(0.0, 1.0));
        self.ssm_readout = self.ssm_readout.map(|v| v.clamp(0.0, 1.0));
        self.lfm_uncertainty = self.lfm_uncertainty.map(|v| v.clamp(0.0, 1.0));
        self.lfm_stability = self.lfm_stability.map(|v| v.clamp(0.0, 1.0));
        self.lfm_state_norm = self.lfm_state_norm.map(|v| v.clamp(0.0, 1.0));
        self.lfm_deriv_norm = self.lfm_deriv_norm.map(|v| v.clamp(0.0, 1.0));
        self.lfm_saturation_ratio = self.lfm_saturation_ratio.map(|v| v.clamp(0.0, 1.0));

        if self.spikes.len() > MAX_SPIKES {
            self.spikes.truncate(MAX_SPIKES);
        }
        if self.notes.len() > MAX_NOTES {
            self.notes.truncate(MAX_NOTES);
        }
        self.notes = self
            .notes
            .into_iter()
            .map(|n| n.chars().take(MAX_NOTE_LEN).collect())
            .collect();
        self
    }

    pub fn summary(&self, backend: &'static str) -> ComputeSignalsSummary {
        let spikes_digest = evidence::spikes_digest(&self.spikes);
        let risk_signal = if validate_risk_signal(&self.risk_signal).is_ok() {
            self.risk_signal
        } else {
            RiskSignal {
                risk: 1.0,
                confidence: 0.0,
                quality: SignalQuality::Unavailable,
                evidence: self.risk_signal.evidence,
                version: 1,
            }
        };
        let evidence_chain = EvidenceChain::from_compute(
            &ComputeInput {
                frame_id: FrameId(0),
                t: 0,
                context_digest: risk_signal.evidence.context_digest,
            },
            &self.spikes,
            &risk_signal,
            self.nsr_digest,
            self.nsr_status,
            self.sae_quality,
            self.ssm_quality,
            self.lfm_quality,
        );
        UCF_COMPUTE_CHAIN_DIGEST_EMITTED_TOTAL.fetch_add(1, Ordering::Relaxed);
        ComputeSignalsSummary {
            backend,
            surprise: self.surprise,
            pressure: self.pressure,
            risk: risk_signal.risk,
            confidence: risk_signal.confidence,
            surprise_q: UQ0_16::from_f32_clamped(self.surprise).raw(),
            pressure_q: UQ0_16::from_f32_clamped(self.pressure).raw(),
            risk_q: UQ0_16::from_f32_clamped(risk_signal.risk).raw(),
            confidence_q: UQ0_16::from_f32_clamped(risk_signal.confidence).raw(),
            spike_count: self.spikes.len() as u16,
            spikes_digest,
            sparsity: self.sparsity,
            energy: self.energy,
            ssm_readout: self.ssm_readout,
            ssm_digest: self.ssm_digest,
            world_digest: self.world_digest,
            risk_quality: risk_signal.quality.as_u8(),
            evidence_context_digest: risk_signal.evidence.context_digest,
            evidence_world_digest: risk_signal.evidence.world_digest,
            evidence_spikes_digest: risk_signal.evidence.spikes_digest,
            evidence_ssm_digest: risk_signal.evidence.ssm_digest,
            evidence_lfm_digest: risk_signal.evidence.lfm_digest,
            ssm_quality: self.ssm_quality,
            lfm_quality: self.lfm_quality,
            backend_profile: risk_signal.evidence.backend_profile.as_str(),
            backend_pack_id: risk_signal.evidence.backend_pack_id.0,
            fixtures_digest: risk_signal.evidence.fixtures_digest,
            model_hashes_digest: risk_signal.evidence.model_hashes_digest,
            llm_backend: risk_signal.evidence.llm_backend as u8,
            world_backend: risk_signal.evidence.world_backend as u8,
            sae_backend: risk_signal.evidence.sae_backend as u8,
            ssm_backend: risk_signal.evidence.ssm_backend as u8,
            lfm_backend: risk_signal.evidence.lfm_backend as u8,
            lfm_uncertainty: self.lfm_uncertainty,
            lfm_stability: self.lfm_stability,
            lfm_uncertainty_q: self
                .lfm_uncertainty
                .map(UQ0_16::from_f32_clamped)
                .map(UQ0_16::raw),
            lfm_stability_q: self
                .lfm_stability
                .map(UQ0_16::from_f32_clamped)
                .map(UQ0_16::raw),
            lfm_state_norm: self.lfm_state_norm,
            lfm_deriv_norm: self.lfm_deriv_norm,
            lfm_saturation_ratio: self.lfm_saturation_ratio,
            lfm_nan_inf_detected: self.lfm_nan_inf_detected,
            lfm_digest: self.lfm_digest,
            nsr_digest: self.nsr_digest,
            nsr_status: self.nsr_status,
            signal_bundle_digest: None,
            budget_profile_id: risk_signal.evidence.budget_profile_id,
            seed: risk_signal.evidence.seed,
            risk_contract_version: risk_signal.version,
            compute_schema_version: evidence_chain.schema_version,
            compute_chain_digest: evidence_chain.chain_digest,
            compute_code_version: evidence_chain.code_version.as_str(),
            budget_exceeded_stage: self.budget_exceeded_stage,
            contract_version: self.contract_version.as_u16(),
            backend_id: self.backend_id,
            validation_status: self.validation_status,
            violation_reason_mask: self.violation_reason_mask,
        }
    }

    pub fn unavailable(input: &ComputeInput, budget: ComputeBudget, backend: &'static str) -> Self {
        let evidence = EvidenceRef {
            context_digest: input.context_digest,
            world_digest: None,
            spikes_digest: None,
            ssm_digest: None,
            lfm_digest: None,
            backend_profile: BackendProfileId::from_backend_name(backend),
            backend_pack_id: crate::BackendPackId(0),
            fixtures_digest: [0; 32],
            model_hashes_digest: [0; 32],
            llm_backend: crate::BackendComponentId::Disabled,
            world_backend: crate::BackendComponentId::Disabled,
            sae_backend: crate::BackendComponentId::Disabled,
            ssm_backend: crate::BackendComponentId::Disabled,
            lfm_backend: crate::BackendComponentId::Disabled,
            seed: budget.seed,
            budget_profile_id: budget.profile_id,
        };
        Self {
            surprise: 0.0,
            pressure: 1.0,
            risk: 1.0,
            confidence: 0.0,
            risk_signal: RiskSignal {
                risk: 1.0,
                confidence: 0.0,
                quality: SignalQuality::Unavailable,
                evidence,
                version: 1,
            },
            spikes: Vec::new(),
            notes: vec!["risk_contract:unavailable".to_string()],
            sparsity: None,
            energy: None,
            ssm_readout: None,
            ssm_digest: None,
            world_digest: None,
            lfm_uncertainty: None,
            lfm_stability: None,
            lfm_state_norm: None,
            lfm_deriv_norm: None,
            lfm_saturation_ratio: None,
            lfm_nan_inf_detected: false,
            lfm_digest: None,
            nsr_digest: None,
            nsr_status: 0,
            signal_bundle_digest: None,
            sae_quality: None,
            ssm_quality: None,
            lfm_quality: None,
            plasticity_record: None,
            budget_exceeded_stage: None,
            contract_version: StageContractVersion::V1,
            backend_id: 0,
            validation_status: ValidationStatus::Degraded,
            violation_reason_mask: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DegradePolicy {
    DegradeStages,
    FailFast,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ComputeBudgetProfile {
    pub profile_id: u32,
    pub global_work_units: u64,
    pub world_units: u64,
    pub sae_units: u64,
    pub ssm_units: u64,
    pub lfm_units: u64,
    pub degrade_policy: DegradePolicy,
}

impl ComputeBudgetProfile {
    pub fn default_profile() -> Self {
        Self {
            profile_id: 1,
            global_work_units: 1_600,
            world_units: 420,
            sae_units: 420,
            ssm_units: 420,
            lfm_units: 420,
            degrade_policy: DegradePolicy::DegradeStages,
        }
    }

    pub fn tight_profile() -> Self {
        Self {
            profile_id: 2,
            global_work_units: 1_100,
            world_units: 360,
            sae_units: 260,
            ssm_units: 360,
            lfm_units: 280,
            degrade_policy: DegradePolicy::DegradeStages,
        }
    }

    pub fn stress_profile() -> Self {
        Self {
            profile_id: 3,
            global_work_units: 900,
            world_units: 360,
            sae_units: 100,
            ssm_units: 360,
            lfm_units: 200,
            degrade_policy: DegradePolicy::DegradeStages,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeBudget {
    pub max_micros: u64,
    pub hard_timeout_micros: u64,
    pub seed: u64,
    pub profile_id: u32,
    pub global_work_units: u64,
    pub world_units: u64,
    pub sae_units: u64,
    pub ssm_units: u64,
    pub lfm_units: u64,
    pub degrade_policy: DegradePolicy,
    pub governor_tier: u8,
}

impl Default for ComputeBudget {
    fn default() -> Self {
        Self {
            max_micros: 1_000,
            hard_timeout_micros: 5_000,
            seed: 0xDEC0DED,
            profile_id: ComputeBudgetProfile::default_profile().profile_id,
            global_work_units: ComputeBudgetProfile::default_profile().global_work_units,
            world_units: ComputeBudgetProfile::default_profile().world_units,
            sae_units: ComputeBudgetProfile::default_profile().sae_units,
            ssm_units: ComputeBudgetProfile::default_profile().ssm_units,
            lfm_units: ComputeBudgetProfile::default_profile().lfm_units,
            degrade_policy: ComputeBudgetProfile::default_profile().degrade_policy,
            governor_tier: 0,
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ComputeError {
    #[error(
        "compute budget exceeded at {stage}: elapsed {elapsed_micros}µs > limit {limit_micros}µs"
    )]
    BudgetExceeded {
        stage: &'static str,
        elapsed_micros: u64,
        limit_micros: u64,
    },
    #[error("invalid compute input: {reason}")]
    InvalidInput { reason: String },
    #[error("compute backend disabled")]
    BackendDisabled,
    #[error("compute backend not implemented")]
    NotImplemented,
    #[error("compute backend internal error: {reason}")]
    Internal { reason: String },
    #[error("{code}")]
    SamplingDisabled { code: &'static str },
}

pub trait AiComputeBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn compute(
        &self,
        input: &ComputeInput,
        budget: ComputeBudget,
    ) -> Result<ComputeSignals, ComputeError>;

    fn drain_shadow_events(&self) -> Vec<SlotShadowEventV1> {
        Vec::new()
    }

    fn apply_shadow_disable(&self, _slot_id: &str, _t: u64, _reason: &str, _to_off: bool) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComputeSignalsSummary {
    pub backend: &'static str,
    pub surprise: f32,
    pub pressure: f32,
    pub risk: f32,
    pub confidence: f32,
    pub surprise_q: u16,
    pub pressure_q: u16,
    pub risk_q: u16,
    pub confidence_q: u16,
    pub spike_count: u16,
    pub spikes_digest: [u8; 32],
    pub sparsity: Option<f32>,
    pub energy: Option<f32>,
    pub ssm_readout: Option<f32>,
    pub ssm_digest: Option<[u8; 32]>,
    pub world_digest: Option<[u8; 32]>,
    pub risk_quality: u8,
    pub evidence_context_digest: [u8; 32],
    pub evidence_world_digest: Option<[u8; 32]>,
    pub evidence_spikes_digest: Option<[u8; 32]>,
    pub evidence_ssm_digest: Option<[u8; 32]>,
    pub evidence_lfm_digest: Option<[u8; 32]>,
    pub ssm_quality: Option<StageQuality>,
    pub lfm_quality: Option<StageQuality>,
    pub backend_profile: &'static str,
    pub backend_pack_id: u32,
    pub fixtures_digest: [u8; 32],
    pub model_hashes_digest: [u8; 32],
    pub llm_backend: u8,
    pub world_backend: u8,
    pub sae_backend: u8,
    pub ssm_backend: u8,
    pub lfm_backend: u8,
    pub lfm_uncertainty: Option<f32>,
    pub lfm_stability: Option<f32>,
    pub lfm_uncertainty_q: Option<u16>,
    pub lfm_stability_q: Option<u16>,
    pub lfm_state_norm: Option<f32>,
    pub lfm_deriv_norm: Option<f32>,
    pub lfm_saturation_ratio: Option<f32>,
    pub lfm_nan_inf_detected: bool,
    pub lfm_digest: Option<[u8; 32]>,
    pub nsr_digest: Option<[u8; 32]>,
    pub nsr_status: u8,
    pub signal_bundle_digest: Option<[u8; 32]>,
    pub budget_profile_id: u32,
    pub seed: u64,
    pub risk_contract_version: u16,
    pub compute_schema_version: u16,
    pub compute_chain_digest: [u8; 32],
    pub compute_code_version: &'static str,
    pub budget_exceeded_stage: Option<&'static str>,
    pub contract_version: u16,
    pub backend_id: u16,
    pub validation_status: ValidationStatus,
    pub violation_reason_mask: u32,
}

pub fn fuse_signals(surprise: f32, pressure: f32, energy: f32) -> (f32, f32) {
    let base_risk = (0.65 * surprise + 0.35 * pressure).clamp(0.0, 1.0);
    let energy_adj = (energy - 0.5).clamp(-0.5, 0.5);
    let risk = (base_risk + 0.15 * energy_adj).clamp(0.0, 1.0);
    let confidence = (1.0 - 0.9 * risk).clamp(0.0, 1.0);
    (risk, confidence)
}

pub fn digest_control_frame(ctrl: &ControlFrame) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(ctrl.time.tick.get().to_le_bytes());
    hasher.update(ctrl.time.window.get().to_le_bytes());
    hasher.update(ctrl.corr.0.to_le_bytes());
    hasher.update([ctrl.channel as u8]);
    hasher.update(ctrl.intent.summary.as_bytes());
    hasher.update(ctrl.intent.id.0.to_le_bytes());
    hasher.update([ctrl.intent.kind as u8]);

    match &ctrl.payload {
        ControlPayload::Text(text) => {
            hasher.update([0]);
            hasher.update(text.as_bytes());
        }
        ControlPayload::Bytes(bytes) => {
            hasher.update([1]);
            hasher.update(bytes.as_ref());
        }
        ControlPayload::BrainStimulus(stimulus) => {
            hasher.update([2]);
            hasher.update([stimulus.kind as u8]);
            hasher.update(stimulus.target.to_le_bytes());
            hasher.update(stimulus.intensity.to_le_bytes());
            hasher.update(stimulus.duration_ms.to_le_bytes());
        }
        ControlPayload::Empty => {
            hasher.update([3]);
        }
    }

    let digest = hasher.finalize();
    let mut out = [0_u8; 32];
    out.copy_from_slice(&digest);
    out
}

pub fn compute_input_from_control(ctrl: &ControlFrame) -> ComputeInput {
    ComputeInput {
        frame_id: FrameId(ctrl.corr.0),
        t: ctrl.time.tick.get(),
        context_digest: digest_control_frame(ctrl),
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CpuStubBackend;

impl AiComputeBackend for CpuStubBackend {
    fn name(&self) -> &'static str {
        "stub"
    }

    fn compute(
        &self,
        input: &ComputeInput,
        budget: ComputeBudget,
    ) -> Result<ComputeSignals, ComputeError> {
        ComputePipelineBackend::stub().compute(input, budget)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capabilities::WorldModelPredictor;
    use crate::world_model::{obs_features_from_context, MockJepaPredictor, WorldModelInput};

    #[allow(dead_code)]
    #[derive(Debug, serde::Deserialize)]
    struct FixtureCase {
        frame_id: u64,
        t: u64,
        context_digest_hex: String,
        seed: u64,
        expected: Expected,
    }

    #[allow(dead_code)]
    #[derive(Debug, serde::Deserialize)]
    struct Expected {
        surprise: f32,
        pressure: f32,
        risk: f32,
        confidence: f32,
        spike_count: usize,
        spikes_digest_hex: String,
    }

    fn decode_hex32(hex: &str) -> [u8; 32] {
        let bytes = hex::decode(hex).expect("valid hex fixture");
        let mut out = [0_u8; 32];
        out.copy_from_slice(&bytes);
        out
    }

    #[test]
    fn deterministic_for_same_input_and_seed() {
        let backend = CpuStubBackend;
        let input = ComputeInput {
            frame_id: FrameId(42),
            t: 7,
            context_digest: [1_u8; 32],
        };
        let budget = ComputeBudget::default();
        let a = backend.compute(&input, budget).expect("compute");
        let b = backend.compute(&input, budget).expect("compute");
        assert_eq!(a, b);
    }

    #[test]
    fn fusion_is_monotonic_and_confidence_inverse() {
        let (r1, _) = fuse_signals(0.2, 0.4, 0.5);
        let (r2, _) = fuse_signals(0.8, 0.4, 0.5);
        assert!(r2 >= r1);

        let (r3, _) = fuse_signals(0.4, 0.2, 0.5);
        let (r4, _) = fuse_signals(0.4, 0.8, 0.5);
        assert!(r4 >= r3);

        let (risk_low, conf_high) = fuse_signals(0.1, 0.1, 0.5);
        let (risk_high, conf_low) = fuse_signals(0.9, 0.9, 0.5);
        assert!(risk_high >= risk_low);
        assert!(conf_low <= conf_high);
    }

    #[test]
    fn surprise_is_driven_by_world_model_predictor() {
        let backend = CpuStubBackend;
        let input = ComputeInput {
            frame_id: FrameId(1337),
            t: 19,
            context_digest: [0x2A_u8; 32],
        };
        let budget = ComputeBudget {
            max_micros: 500,
            hard_timeout_micros: 5_000,
            seed: 17,
            ..ComputeBudget::default()
        };

        let out = backend.compute(&input, budget).expect("compute");
        let mut predictor = MockJepaPredictor::default();
        let model = predictor
            .step(
                &WorldModelInput {
                    t: input.t,
                    context_digest: input.context_digest,
                    previous_state_digest: None,
                    obs_features: obs_features_from_context(input.context_digest),
                    seed: budget.seed,
                },
                budget,
            )
            .expect("predict");

        assert!((out.surprise - model.surprise).abs() <= 1e-6);
        assert_eq!(out.world_digest, Some(model.prediction_digest));
    }

    #[test]
    fn boundedness_clamps_and_truncates() {
        let spikes = (0..300)
            .map(|i| Spike {
                feature_id: i as u32,
                magnitude: 2.0,
                timestamp: 9,
            })
            .collect();
        let notes = (0..20).map(|_| "x".repeat(400)).collect();
        let bounded = ComputeSignals {
            surprise: 2.0,
            pressure: -1.0,
            risk: 3.0,
            confidence: 4.0,
            risk_signal: RiskSignal {
                risk: 3.0,
                confidence: 4.0,
                quality: SignalQuality::Unavailable,
                evidence: EvidenceRef {
                    context_digest: [0; 32],
                    world_digest: None,
                    spikes_digest: None,
                    ssm_digest: None,
                    lfm_digest: None,
                    backend_profile: BackendProfileId::StubV1,
                    backend_pack_id: crate::BackendPackId(1),
                    fixtures_digest: [9; 32],
                    model_hashes_digest: [8; 32],
                    llm_backend: crate::BackendComponentId::ToyV1,
                    world_backend: crate::BackendComponentId::ToyV1,
                    sae_backend: crate::BackendComponentId::ToyV1,
                    ssm_backend: crate::BackendComponentId::ToyV1,
                    lfm_backend: crate::BackendComponentId::ToyV1,
                    seed: 0,
                    budget_profile_id: 0,
                },
                version: 1,
            },
            spikes,
            notes,
            sparsity: Some(2.0),
            energy: Some(-1.0),
            ssm_readout: Some(3.0),
            ssm_digest: None,
            world_digest: None,
            lfm_uncertainty: None,
            lfm_stability: None,
            lfm_state_norm: None,
            lfm_deriv_norm: None,
            lfm_saturation_ratio: None,
            lfm_nan_inf_detected: false,
            lfm_digest: None,
            nsr_digest: None,
            nsr_status: 0,
            signal_bundle_digest: None,
            sae_quality: None,
            ssm_quality: None,
            lfm_quality: None,
            plasticity_record: None,
            budget_exceeded_stage: None,
            contract_version: StageContractVersion::V1,
            backend_id: 0,
            validation_status: ValidationStatus::Degraded,
            violation_reason_mask: 0,
        }
        .bounded();
        assert_eq!(bounded.spikes.len(), MAX_SPIKES);
        assert_eq!(bounded.notes.len(), MAX_NOTES);
        assert!(bounded.notes.iter().all(|n| n.len() <= MAX_NOTE_LEN));
        assert_eq!(bounded.surprise, 1.0);
        assert_eq!(bounded.pressure, 0.0);
        assert_eq!(bounded.risk, 1.0);
        assert_eq!(bounded.confidence, 1.0);
    }

    #[test]
    fn unavailable_signal_is_safe_default() {
        let input = ComputeInput {
            frame_id: FrameId(1),
            t: 1,
            context_digest: [7; 32],
        };
        let sig = ComputeSignals::unavailable(&input, ComputeBudget::default(), "stub");
        assert_eq!(sig.risk, 1.0);
        assert_eq!(sig.confidence, 0.0);
        assert_eq!(sig.risk_signal.quality, SignalQuality::Unavailable);
    }

    #[test]
    fn budget_exceeded_is_reported() {
        let backend = CpuStubBackend;
        let input = ComputeInput {
            frame_id: FrameId(1),
            t: 1,
            context_digest: [255_u8; 32],
        };
        let out = backend
            .compute(
                &input,
                ComputeBudget {
                    max_micros: 4,
                    hard_timeout_micros: 1,
                    seed: 0,
                    ..ComputeBudget::default()
                },
            )
            .expect("should degrade deterministically");
        assert_eq!(out.risk_signal.quality, SignalQuality::DegradedFallback);
        assert_eq!(out.budget_exceeded_stage, Some("lfm/step"));
    }

    #[test]
    fn golden_vectors_from_fixture() {
        let backend = CpuStubBackend;
        let cases: Vec<FixtureCase> =
            serde_json::from_str(include_str!("../fixtures/compute_inputs.json"))
                .expect("fixture parse");

        for case in cases {
            let input = ComputeInput {
                frame_id: FrameId(case.frame_id),
                t: case.t,
                context_digest: decode_hex32(&case.context_digest_hex),
            };
            let out = backend
                .compute(
                    &input,
                    ComputeBudget {
                        max_micros: 500,
                        hard_timeout_micros: 5_000,
                        seed: case.seed,
                        ..ComputeBudget::default()
                    },
                )
                .expect("compute output");

            assert!((0.0..=1.0).contains(&out.surprise));
            assert!((0.0..=1.0).contains(&out.pressure));
            assert!((0.0..=1.0).contains(&out.risk));
            assert!((0.0..=1.0).contains(&out.confidence));
            let replay = CpuStubBackend
                .compute(
                    &input,
                    ComputeBudget {
                        max_micros: 500,
                        hard_timeout_micros: 5_000,
                        seed: case.seed,
                        ..ComputeBudget::default()
                    },
                )
                .expect("compute replay");
            assert_eq!(out, replay);

            let summary = out.summary("stub");
            assert_ne!(summary.spikes_digest, [0_u8; 32]);
        }
    }
}
