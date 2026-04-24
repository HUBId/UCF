use crate::backends::{CANONICAL_ONBOARDING_BACKEND, CANONICAL_ONBOARDING_PACK};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeReferenceClass {
    CanonicalProduction,
    CanonicalExpertRuntimeControl,
    CanonicalDiagnosticsEvidence,
    InternalOrLegacy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeReferenceLane {
    pub class: ComputeReferenceClass,
    pub lane: &'static str,
    pub canonical_path: &'static str,
    pub scope: &'static str,
    pub shared_core_invariants: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeIntegrationContractClass {
    Execution,
    DiagnosticsStatus,
    EvidenceReference,
    ExpertInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeIntegrationBoundary {
    OutwardFacing,
    ExpertInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeIntegrationContractLane {
    pub class: ComputeIntegrationContractClass,
    pub boundary: ComputeIntegrationBoundary,
    pub lane: &'static str,
    pub canonical_anchor: &'static str,
    pub semantic_scope: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainFacingConsumerAlignment {
    AlignedCanonicalOutward,
    LegacyCompatPath,
    NeedsFinalIntegrationAdjustment,
    InternalDevTestOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainFacingCompletionStatus {
    AlignedToFinalComputeLine,
    MostlyAlignedWithCaveats,
    MixedTransitional,
    InternalOnlyNotTrueOutwardConsumer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainFacingStatusConsumptionPattern {
    CanonicalStatusConsumer,
    MixedLegacyConsumption,
    InternalDevTestOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainFacingEvidenceConsumptionPattern {
    CanonicalEvidenceReferenceConsumer,
    MixedLegacyConsumption,
    InternalDevTestOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DomainFacingComputeConsumerLane {
    pub consumer: &'static str,
    pub repo_surface: &'static str,
    pub execution_contract_path: &'static str,
    pub status_diagnostics_path: &'static str,
    pub evidence_reference_path: &'static str,
    pub status_pattern: DomainFacingStatusConsumptionPattern,
    pub evidence_pattern: DomainFacingEvidenceConsumptionPattern,
    pub alignment: DomainFacingConsumerAlignment,
    pub completion_status: DomainFacingCompletionStatus,
    pub caveat: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainRolloutCandidateClass {
    RolloutReadyCandidate,
    RolloutPlausibleWithCaveats,
    MixedTransitionalCandidate,
    NotRealRolloutCandidateNow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FirstDomainRolloutCompletionStatus {
    Aligned,
    AlignedWithCaveats,
    MixedTransitional,
    NotYetTrueRolloutCompletion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FirstDomainRolloutCompletionLane {
    pub rollout_case: &'static str,
    pub completion_status: FirstDomainRolloutCompletionStatus,
    pub execution_contract_check: &'static str,
    pub outward_status_evidence_check: &'static str,
    pub integration_safe_hook_check: &'static str,
    pub hidden_legacy_dependency_check: &'static str,
    pub caveat: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DomainRolloutCandidateLane {
    pub candidate: &'static str,
    pub rollout_class: DomainRolloutCandidateClass,
    pub outward_execution_contract: &'static str,
    pub outward_status_evidence_surface: &'static str,
    pub integration_safe_hook_posture: &'static str,
    pub excluded_internal_or_legacy_paths: &'static str,
    pub caveat: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostRolloutAdoptionClass {
    AlreadyAligned,
    FirstRealRolloutEstablished,
    BroaderAdoptionReviewCandidate,
    NotPursuedNow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PostRolloutAdoptionLane {
    pub surface: &'static str,
    pub adoption_class: PostRolloutAdoptionClass,
    pub rollout_anchor_comparison: &'static str,
    pub outward_contract_fit: &'static str,
    pub legacy_internal_dependency_posture: &'static str,
    pub caveat: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainIntegrationClass {
    RealBlueBrainCoreCandidate,
    BlueBrainAdjacentComputeConsumer,
    IndirectOrCompatibilityTouchingSurface,
    InternalOnlyOrNotMeaningfulForBlueBrainIntegration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainIntegrationLane {
    pub surface: &'static str,
    pub class: BlueBrainIntegrationClass,
    pub repo_surface: &'static str,
    pub execution_contract_path: &'static str,
    pub status_diagnostics_contract_path: &'static str,
    pub evidence_reference_contract_path: &'static str,
    pub integration_safe_hook_posture: &'static str,
    pub coupling_posture: &'static str,
    pub caveat: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFacingContractClass {
    InferenceFacing,
    StateFacing,
    StatusHealthTrustFacing,
    EvidenceReferenceFacing,
    ExpertInternalOnlyNonBlueBrain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainFacingContractLane {
    pub class: BlueBrainFacingContractClass,
    pub lane: &'static str,
    pub canonical_anchor: &'static str,
    pub allowed_semantics: &'static str,
    pub excluded_semantics: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainComputeHandoffClass {
    InferenceHandoff,
    StatusDiagnosticsHandoff,
    EvidenceReferenceHandoff,
    StateAdjacentReferenceHandoff,
    ExpertInternalOnlyNonCanonicalHandoff,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainComputeHandoffLane {
    pub class: BlueBrainComputeHandoffClass,
    pub lane: &'static str,
    pub canonical_transition: &'static str,
    pub outbound_payload_shape: &'static str,
    pub return_payload_shape: &'static str,
    pub canonical_references: &'static str,
    pub non_canonical_boundary: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainIntegrationCandidateClass {
    IntegrationReadyCandidate,
    PlausibleWithCaveats,
    MixedTransitionalCandidate,
    NotRealBlueBrainIntegrationCandidateNow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainIntegrationCandidateLane {
    pub surface: &'static str,
    pub class: BlueBrainIntegrationCandidateClass,
    pub candidate_selection_posture: &'static str,
    pub inference_contract_binding: &'static str,
    pub status_handoff_binding: &'static str,
    pub evidence_handoff_binding: &'static str,
    pub state_adjacent_binding: &'static str,
    pub excluded_internal_or_legacy_paths: &'static str,
    pub caveat: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainRuntimeSurfaceClass {
    StateBearingSurface,
    InferenceBearingSurface,
    StatusHealthTrustFacingSurface,
    EvidenceReplayFacingSurface,
    InternalOnlyRuntimeControlSurface,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainRuntimeSurfaceLane {
    pub class: BlueBrainRuntimeSurfaceClass,
    pub lane: &'static str,
    pub canonical_anchor: &'static str,
    pub runtime_scope: &'static str,
    pub compute_line_binding: &'static str,
    pub boundary_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainRuntimePhaseClass {
    StateContextAvailable,
    ComputeInvocationRequested,
    ComputeResultIntegrated,
    StatusEvidenceObserved,
    CaveatedOrDegradedOrPartialRuntimeState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainRuntimePhaseLane {
    pub class: BlueBrainRuntimePhaseClass,
    pub lane: &'static str,
    pub phase_transition: &'static str,
    pub canonical_inputs: &'static str,
    pub canonical_outputs: &'static str,
    pub non_goal_boundary: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainTransitionTriggerClass {
    PureStateTransition,
    ComputeTriggeringTransition,
    EvidenceStatusUpdateTransition,
    InternalOnlyOrNonCanonicalTransition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainTransitionTriggerLane {
    pub class: BlueBrainTransitionTriggerClass,
    pub lane: &'static str,
    pub canonical_transition: &'static str,
    pub trigger_point: &'static str,
    pub canonical_contract_binding: &'static str,
    pub reference_continuity: &'static str,
    pub non_canonical_boundary: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainContextMemoryBoundaryClass {
    PureComputeConsumer,
    ContextBearingSurface,
    MemoryAdjacentSurface,
    EvidenceReferenceConsumer,
    InternalOnlyOrNonCanonicalContextPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainContextMemoryBoundaryLane {
    pub class: BlueBrainContextMemoryBoundaryClass,
    pub lane: &'static str,
    pub surface: &'static str,
    pub canonical_anchor: &'static str,
    pub compute_invocation_reference: &'static str,
    pub context_reference: &'static str,
    pub evidence_or_replay_reference: &'static str,
    pub memory_posture: &'static str,
    pub boundary_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainRuntimeFeedbackClass {
    ComputeResultFeedback,
    StatusTrustFeedback,
    EvidenceReferenceFeedback,
    DiagnosticCaveatFeedback,
    ContextUptakeFeedback,
    NonCanonicalInternalExpertFeedback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainRuntimeFeedbackLane {
    pub class: BlueBrainRuntimeFeedbackClass,
    pub lane: &'static str,
    pub canonical_source: &'static str,
    pub runtime_feedback_semantics: &'static str,
    pub transition_binding: &'static str,
    pub memory_boundary: &'static str,
    pub non_canonical_boundary: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainContextMemorySurfaceClass {
    TransientRuntimeContext,
    EvidenceBackedContext,
    ReplayReferenceBackedContext,
    MemoryAdjacentCandidate,
    PersistedMemory,
    NonCanonicalInternalOnlyMemoryLikePath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainContextMemorySurfaceLane {
    pub class: BlueBrainContextMemorySurfaceClass,
    pub lane: &'static str,
    pub source_surface: &'static str,
    pub context_shape: &'static str,
    pub evidence_or_reference_binding: &'static str,
    pub persistence_binding: &'static str,
    pub canonical_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainContextUpdateLifecycleClass {
    ContextInitialized,
    UpdatedFromComputeResult,
    UpdatedFromEvidenceReference,
    UpdatedFromReplayReference,
    ContextUnchanged,
    UpdateBlockedOrInsufficient,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainContextUpdateLifecycleLane {
    pub class: BlueBrainContextUpdateLifecycleClass,
    pub lane: &'static str,
    pub source_surface: &'static str,
    pub update_semantics: &'static str,
    pub candidate_effect: &'static str,
    pub persistence_semantics: &'static str,
    pub canonical_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainMemoryCandidateLifecycleClass {
    CandidateProposed,
    CandidateEvidenceBacked,
    CandidateContextDerived,
    CandidateComputeResultDerived,
    AcceptedForFutureMemoryHandling,
    CandidateRejected,
    CandidateStale,
    CandidateInsufficient,
    PersistenceUnavailableOrDeferred,
    PersistencePerformedViaRealPathOnly,
    NoPersistencePerformed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainMemoryCandidateLifecycleLane {
    pub class: BlueBrainMemoryCandidateLifecycleClass,
    pub lane: &'static str,
    pub source_surface: &'static str,
    pub candidate_semantics: &'static str,
    pub context_mutation_semantics: &'static str,
    pub persistence_semantics: &'static str,
    pub canonical_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainMemoryCommitBoundaryClass {
    NotMemoryCandidate,
    MemoryCandidateProposed,
    MemoryCandidateDeferred,
    MemoryCandidateRejected,
    MemoryCandidateStale,
    MemoryCandidateInsufficient,
    CommitEligibleCandidate,
    FutureMemoryReadyCandidate,
    CommittedMemoryIfRealPath,
    ReferenceOnlyNotMemory,
    NonCanonicalInternalOnlyPersistencePath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainMemoryCommitBoundaryLane {
    pub class: BlueBrainMemoryCommitBoundaryClass,
    pub lane: &'static str,
    pub source_binding: &'static str,
    pub eligibility_semantics: &'static str,
    pub persistence_path_semantics: &'static str,
    pub canonical_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainCommitEligibilityConditionClass {
    EvidenceReferenceBasis,
    SelectionAttentionGate,
    ContextFreshnessGate,
    BlockingCaveatGate,
    CanonicalDependencyGate,
    PersistencePathGate,
    FutureMemoryReadyHandoffGate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainCommitEligibilityConditionLane {
    pub class: BlueBrainCommitEligibilityConditionClass,
    pub lane: &'static str,
    pub requirement: &'static str,
    pub when_satisfied: &'static str,
    pub when_not_satisfied: &'static str,
    pub canonical_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainPersistenceBoundaryClass {
    TransientRuntimeContext,
    EvidenceReferenceBackedContext,
    MemoryAdjacentCandidate,
    FutureMemoryReadyCandidate,
    ActualPersistedMemory,
    HistorySnapshotReferenceButNotMemory,
    NonCanonicalInternalOnlyPersistenceLikePath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainPersistenceBoundaryLane {
    pub class: BlueBrainPersistenceBoundaryClass,
    pub lane: &'static str,
    pub source_surface: &'static str,
    pub boundary_semantics: &'static str,
    pub future_attachment_semantics: &'static str,
    pub canonical_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFutureMemoryAttachmentClass {
    CandidateHandoffProposalOnly,
    CandidateFutureReadyNoCommit,
    CandidateRejectedOrInsufficient,
    PersistenceDeferredOrUnavailable,
    PersistenceCommitOnlyIfRealPathExists,
    HistoryReferenceBasisOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainFutureMemoryAttachmentLane {
    pub class: BlueBrainFutureMemoryAttachmentClass,
    pub lane: &'static str,
    pub trigger_or_source: &'static str,
    pub required_fields: &'static str,
    pub caveats: &'static str,
    pub commit_boundary: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFutureMemoryHandoffStateClass {
    HandoffReady,
    HandoffDeferred,
    HandoffBlocked,
    HandoffRejected,
    HandoffCaveated,
    HandoffUnavailable,
    HandoffInternalOnlyNonCanonical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainFutureMemoryHandoffStateLane {
    pub class: BlueBrainFutureMemoryHandoffStateClass,
    pub lane: &'static str,
    pub trigger_or_source: &'static str,
    pub handoff_fields: &'static str,
    pub state_semantics: &'static str,
    pub canonical_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainCommitResultClass {
    CommitUnavailable,
    CommitDeferred,
    CommitCommitted,
    CommitCommittedWithCaveats,
    CommitRejected,
    CommitBlocked,
    CommitFailed,
    CommitNoOp,
    CommitReferenceRecordedOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainCommitResultLane {
    pub class: BlueBrainCommitResultClass,
    pub lane: &'static str,
    pub trigger_or_source: &'static str,
    pub result_semantics: &'static str,
    pub runtime_diagnostics_binding: &'static str,
    pub canonical_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainMemoryCommitDiagnosticClass {
    HandoffDiagnostic,
    CommitEligibilityDiagnostic,
    CommitRejectedDiagnostic,
    CommitBlockedDiagnostic,
    CommitDeferredDiagnostic,
    CommitCaveatedDiagnostic,
    CommitUnavailableDiagnostic,
    CommittedIfPresentDiagnostic,
    NoPersistenceDiagnostic,
    NonCanonicalInternalOnlyDiagnostic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainMemoryCommitDiagnosticLane {
    pub class: BlueBrainMemoryCommitDiagnosticClass,
    pub lane: &'static str,
    pub compact_reason: &'static str,
    pub handoff_or_commit_binding: &'static str,
    pub candidate_lifecycle_binding: &'static str,
    pub selection_deferral_binding: &'static str,
    pub runtime_context_binding: &'static str,
    pub canonical_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainReferenceContextClass {
    EvidenceBackedContext,
    ReplayBackedContext,
    SnapshotReferenceBackedContext,
    TraceBackedContext,
    CaveatedReferenceContext,
    InsufficientReferenceContext,
    NonCanonicalInternalOnlyReferencePath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainReferenceQualityClass {
    Sufficient,
    Partial,
    Stale,
    Caveated,
    Insufficient,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainReferenceContextLane {
    pub class: BlueBrainReferenceContextClass,
    pub quality: BlueBrainReferenceQualityClass,
    pub lane: &'static str,
    pub source_surface: &'static str,
    pub runtime_context_semantics: &'static str,
    pub context_update_semantics: &'static str,
    pub candidate_semantics: &'static str,
    pub persistence_boundary: &'static str,
    pub canonical_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainControlAttentionSelectionClass {
    AttentionTarget,
    ContextSelection,
    EvidenceReferenceSelection,
    MemoryCandidateSelection,
    ComputeTriggerSelection,
    NonCanonicalInternalOnlySelectionPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainComputeTriggerArbitrationClass {
    TriggerCandidate,
    SelectedTrigger,
    DeferredTrigger,
    SuppressedTrigger,
    BlockedTrigger,
    InsufficientTriggerBasis,
    CaveatedTrigger,
    NonCanonicalInternalOnlyTrigger,
    InvocationResultFeedback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainComputeTriggerSourceClass {
    ContextDerived,
    EvidenceReferenceDerived,
    RuntimeStateDerived,
    MemoryCandidateDerived,
    FeedbackDerived,
    ManualInternalOnlyNonCanonical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainSelectionGatedInvocationClass {
    InvocationRequested,
    NoInvocationDeferred,
    NoInvocationBlocked,
    CaveatedInvocationAllowed,
    InsufficientBasisRequiresMoreContextOrEvidence,
    InvocationCompleted,
    InvocationFailed,
    InvocationBlockedByComputeContract,
    InvocationCaveatedOrDegraded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainComputeTriggerArbitrationLane {
    pub class: BlueBrainComputeTriggerArbitrationClass,
    pub source: BlueBrainComputeTriggerSourceClass,
    pub invocation: BlueBrainSelectionGatedInvocationClass,
    pub basis_quality: BlueBrainSelectionBasisQualityClass,
    pub lane: &'static str,
    pub arbitration_semantics: &'static str,
    pub selection_binding: &'static str,
    pub outward_compute_contract_binding: &'static str,
    pub memory_commit_boundary: &'static str,
    pub canonical_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainSelectionDispositionClass {
    Selected,
    Deferred,
    IgnoredOrIrrelevant,
    Blocked,
    Insufficient,
    Caveated,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainSelectionBasisQualityClass {
    Sufficient,
    Partial,
    Stale,
    Caveated,
    Insufficient,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainControlAttentionSelectionLane {
    pub class: BlueBrainControlAttentionSelectionClass,
    pub disposition: BlueBrainSelectionDispositionClass,
    pub basis_quality: BlueBrainSelectionBasisQualityClass,
    pub lane: &'static str,
    pub selection_scope: &'static str,
    pub source_surface: &'static str,
    pub compute_trigger_binding: &'static str,
    pub memory_persistence_semantics: &'static str,
    pub canonical_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainContextEvidencePriorityClass {
    PrimaryContext,
    SupportingContext,
    DeferredContext,
    IgnoredContext,
    StaleContext,
    InsufficientContext,
    PrimaryEvidenceReference,
    SupportingEvidenceReference,
    CaveatedEvidenceReference,
    NonCanonicalInternalOnlyPriorityPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainContextEvidencePriorityLane {
    pub class: BlueBrainContextEvidencePriorityClass,
    pub quality: BlueBrainSelectionBasisQualityClass,
    pub lane: &'static str,
    pub priority_semantics: &'static str,
    pub source_binding: &'static str,
    pub trigger_arbitration_binding: &'static str,
    pub candidate_binding: &'static str,
    pub deferral_or_caveat_reason: &'static str,
    pub recheck_condition: &'static str,
    pub canonical_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainCandidateDeferralLifecycleClass {
    CandidateSelected,
    CandidateDeferred,
    CandidateDeferredPendingStrongerEvidence,
    CandidateDeferredPendingContextUpdate,
    CandidateRejected,
    CandidateStale,
    CandidateInsufficient,
    CandidateNotPersisted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainCandidateDeferralLifecycleLane {
    pub class: BlueBrainCandidateDeferralLifecycleClass,
    pub quality: BlueBrainSelectionBasisQualityClass,
    pub lane: &'static str,
    pub lifecycle_semantics: &'static str,
    pub source_binding: &'static str,
    pub deferral_reason: &'static str,
    pub recheck_condition: &'static str,
    pub trigger_binding: &'static str,
    pub memory_commit_boundary: &'static str,
    pub canonical_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainSelectionDiagnosticClass {
    SelectedItemDiagnostic,
    DeferredItemDiagnostic,
    IgnoredItemDiagnostic,
    RejectedItemDiagnostic,
    BlockedSelectionDiagnostic,
    InsufficientSelectionDiagnostic,
    CaveatedSelectionDiagnostic,
    NonCanonicalInternalOnlyDiagnosticDetail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainSelectionDiagnosticLane {
    pub class: BlueBrainSelectionDiagnosticClass,
    pub lane: &'static str,
    pub entity_scope: &'static str,
    pub outcome_binding: &'static str,
    pub compact_reason: &'static str,
    pub runtime_diagnostics_binding: &'static str,
    pub state_surface_binding: &'static str,
    pub canonical_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainPlanningReasoningCandidateClass {
    RuntimeDerivedPlanningCandidate,
    ContextDerivedReasoningCandidate,
    EvidenceReferenceDerivedReasoningCandidate,
    SelectionDerivedActionCandidate,
    MemoryCandidateDerivedReasoningCandidate,
    CommitFeedbackDerivedCandidate,
    InsufficientCandidateBasis,
    NonCanonicalInternalOnlyPlanningLikePath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainPlanningReasoningCandidateBasisState {
    BasisAvailable,
    BasisPartialOrCaveated,
    BasisStale,
    BasisInsufficient,
    CandidateDeferred,
    CandidateProposedUnresolved,
    EvidenceObservedNoCandidate,
    CandidateBlocked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainPlanningReasoningCandidateLane {
    pub class: BlueBrainPlanningReasoningCandidateClass,
    pub basis_state: BlueBrainPlanningReasoningCandidateBasisState,
    pub lane: &'static str,
    pub source_binding: &'static str,
    pub candidate_semantics: &'static str,
    pub quality_or_caveat: &'static str,
    pub resolution_boundary: &'static str,
    pub no_execution_implication: &'static str,
    pub memory_commit_boundary: &'static str,
    pub canonical_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainCandidateActionBoundaryClass {
    PlanningReasoningCandidate,
    ActionProposalNonExecuting,
    SelectedProposal,
    DeferredProposal,
    RejectedProposal,
    BlockedProposal,
    CaveatedProposal,
    InsufficientProposalBasis,
    ExecutedActionCanonicalIfPresent,
    NonCanonicalInternalOnlyActionLikePath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainCandidateActionBoundaryExecutionState {
    NoExecutionPerformed,
    SelectedForPossibleFutureAction,
    FutureActionReadyTriggerCandidateOnly,
    ExecutedViaCanonicalComputePathOnlyIfExplicitlyInvoked,
    NonCanonicalInternalOnlyNoAuthority,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainCandidateActionBoundaryLane {
    pub class: BlueBrainCandidateActionBoundaryClass,
    pub execution_state: BlueBrainCandidateActionBoundaryExecutionState,
    pub lane: &'static str,
    pub candidate_or_proposal_semantics: &'static str,
    pub basis_binding: &'static str,
    pub context_evidence_selection_binding: &'static str,
    pub trigger_origin_binding: &'static str,
    pub memory_commit_feedback_binding: &'static str,
    pub caveat_binding: &'static str,
    pub compute_invocation_boundary: &'static str,
    pub memory_commit_boundary: &'static str,
    pub tool_execution_boundary: &'static str,
    pub canonical_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainCandidateToProposalTransitionClass {
    CandidateRemainsCandidate,
    CandidateYieldsActionProposal,
    CandidateInsufficientForProposal,
    CandidateYieldsCaveatedProposal,
    CandidateRejectedBeforeProposal,
    CandidateDeferredBeforeProposal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainCandidateToProposalTransitionLane {
    pub class: BlueBrainCandidateToProposalTransitionClass,
    pub lane: &'static str,
    pub source_candidate_binding: &'static str,
    pub transition_semantics: &'static str,
    pub proposal_outcome: &'static str,
    pub execution_boundary: &'static str,
    pub compute_boundary: &'static str,
    pub memory_commit_boundary: &'static str,
    pub canonical_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainNonExecutingActionProposalStateClass {
    ProposalCreated,
    ProposalSelectedForPossibleFutureAction,
    ProposalDeferred,
    ProposalRejected,
    ProposalBlocked,
    ProposalCaveated,
    ProposalInsufficientBasis,
    NoExecutionPerformed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainNonExecutingActionProposalStateLane {
    pub class: BlueBrainNonExecutingActionProposalStateClass,
    pub lane: &'static str,
    pub proposal_state_semantics: &'static str,
    pub proposal_basis_binding: &'static str,
    pub execution_boundary: &'static str,
    pub compute_trigger_boundary: &'static str,
    pub memory_commit_boundary: &'static str,
    pub tool_execution_boundary: &'static str,
    pub canonical_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainReasoningCandidateDiagnosticClass {
    CandidateBasisDiagnostic,
    SufficientCandidateDiagnostic,
    PartialCandidateDiagnostic,
    CaveatedCandidateDiagnostic,
    StaleCandidateDiagnostic,
    InsufficientCandidateDiagnostic,
    DeferredCandidateDiagnostic,
    RejectedCandidateDiagnostic,
    ProposalReadyDiagnostic,
    NonCanonicalInternalOnlyDiagnostic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainReasoningCandidateDiagnosticLane {
    pub class: BlueBrainReasoningCandidateDiagnosticClass,
    pub lane: &'static str,
    pub basis_binding: &'static str,
    pub insufficiency_or_caveat_reason: &'static str,
    pub proposal_boundary_binding: &'static str,
    pub selection_deferral_binding: &'static str,
    pub memory_boundary_binding: &'static str,
    pub runtime_context_feedback_binding: &'static str,
    pub canonical_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainCandidateComparisonClass {
    ComparableCandidates,
    ComparisonBasisAvailable,
    ComparisonMeaningful,
    ComparisonCaveated,
    ComparisonInconclusive,
    ComparisonNotMeaningful,
    ComparisonBlocked,
    NonCanonicalInternalOnlyComparison,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainCandidateComparisonLane {
    pub class: BlueBrainCandidateComparisonClass,
    pub lane: &'static str,
    pub candidate_scope: &'static str,
    pub runtime_basis_binding: &'static str,
    pub context_basis_binding: &'static str,
    pub evidence_reference_basis_binding: &'static str,
    pub selection_basis_binding: &'static str,
    pub memory_basis_binding: &'static str,
    pub proposal_status_basis_binding: &'static str,
    pub comparison_quality_or_caveat: &'static str,
    pub selection_interaction_boundary: &'static str,
    pub proposal_interaction_boundary: &'static str,
    pub runtime_diagnostics_binding: &'static str,
    pub canonical_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainMinimalPlanningActionInterfaceClass {
    DiagnosticOnlyProposal,
    PlanReadyProposal,
    ActionReadyProposal,
    DeferredProposal,
    BlockedProposal,
    RejectedProposal,
    CaveatedProposal,
    InsufficientProposalBasis,
    ExecutedActionCanonicalIfPresent,
    NonCanonicalInternalOnlyActionPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainMinimalPlanningActionInterfaceLane {
    pub class: BlueBrainMinimalPlanningActionInterfaceClass,
    pub lane: &'static str,
    pub readiness_semantics: &'static str,
    pub proposal_basis_binding: &'static str,
    pub diagnostics_comparison_binding: &'static str,
    pub context_evidence_selection_binding: &'static str,
    pub memory_commit_feedback_binding: &'static str,
    pub execution_boundary: &'static str,
    pub plan_boundary: &'static str,
    pub compute_invocation_boundary: &'static str,
    pub tool_invocation_boundary: &'static str,
    pub memory_commit_boundary: &'static str,
    pub canonical_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainPlanActionReadinessDiagnosticClass {
    PlanReadyDiagnostic,
    ActionReadyDiagnostic,
    DiagnosticOnlyProposalDiagnostic,
    DeferredReadinessDiagnostic,
    BlockedReadinessDiagnostic,
    RejectedReadinessDiagnostic,
    CaveatedReadinessDiagnostic,
    InsufficientReadinessDiagnostic,
    NonCanonicalInternalOnlyReadinessDiagnostic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainPlanActionReadinessDiagnosticLane {
    pub class: BlueBrainPlanActionReadinessDiagnosticClass,
    pub lane: &'static str,
    pub readiness_reason: &'static str,
    pub proposal_boundary_feedback: &'static str,
    pub selection_deferral_feedback: &'static str,
    pub context_evidence_memory_feedback: &'static str,
    pub runtime_feedback: &'static str,
    pub blocked_action_feedback: &'static str,
    pub execution_tool_policy_boundary: &'static str,
    pub canonical_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFutureActionHandoffClass {
    FutureActionReady,
    FuturePlanReady,
    HandoffDeferred,
    HandoffBlocked,
    HandoffRejected,
    HandoffCaveated,
    HandoffUnavailable,
    DiagnosticOnlyNoHandoff,
    InternalOnlyNonCanonicalHandoff,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainFutureActionHandoffLane {
    pub class: BlueBrainFutureActionHandoffClass,
    pub lane: &'static str,
    pub handoff_semantics: &'static str,
    pub proposal_identity_binding: &'static str,
    pub proposal_origin_binding: &'static str,
    pub readiness_basis_binding: &'static str,
    pub evidence_reference_basis_binding: &'static str,
    pub selection_attention_binding: &'static str,
    pub caveat_or_blocker_binding: &'static str,
    pub execution_and_commit_boundary: &'static str,
    pub runtime_diagnostics_binding: &'static str,
    pub canonical_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainActionResultPlaceholderClass {
    ResultPlaceholderPrepared,
    ResultPlaceholderUnavailable,
    ResultPlaceholderBlocked,
    ResultPlaceholderCaveated,
    NoResultExpected,
    NoActionExecuted,
    NoToolResult,
    InternalOnlyNonCanonicalPlaceholder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainActionResultPlaceholderLane {
    pub class: BlueBrainActionResultPlaceholderClass,
    pub lane: &'static str,
    pub placeholder_semantics: &'static str,
    pub handoff_dependency_binding: &'static str,
    pub result_slot_shape: &'static str,
    pub boundary_semantics: &'static str,
    pub runtime_diagnostics_binding: &'static str,
    pub canonical_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainSafetyPrecheckClass {
    Passed,
    Failed,
    Blocked,
    Caveated,
    Insufficient,
    Unavailable,
    NotApplicable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainSafetyPrecheckLane {
    pub class: BlueBrainSafetyPrecheckClass,
    pub lane: &'static str,
    pub precheck_semantics: &'static str,
    pub basis_binding: &'static str,
    pub eligibility_effect: &'static str,
    pub execution_boundary: &'static str,
    pub canonical_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainActionExecutionEligibilityClass {
    FutureActionReadyHandoff,
    ExecutionEligibleHandoff,
    ExecutionIneligibleHandoff,
    ExecutionBlockedHandoff,
    ExecutionCaveatedHandoff,
    ExecutionInsufficientBasis,
    SafetyPrecheckPassed,
    SafetyPrecheckFailed,
    SafetyPrecheckBlocked,
    SafetyPrecheckCaveated,
    SafetyPrecheckUnavailable,
    ExecutedActionCanonicalIfPresent,
    NonCanonicalInternalOnlyExecutionPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainActionExecutionEligibilityLane {
    pub class: BlueBrainActionExecutionEligibilityClass,
    pub lane: &'static str,
    pub eligibility_semantics: &'static str,
    pub handoff_binding: &'static str,
    pub context_evidence_basis: &'static str,
    pub selection_candidate_basis: &'static str,
    pub memory_basis: &'static str,
    pub safety_precheck_binding: &'static str,
    pub execution_boundary: &'static str,
    pub canonical_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainExecutionEligibilityDiagnosticClass {
    ExecutionEligibleDiagnostic,
    ExecutionIneligibleDiagnostic,
    ExecutionBlockedDiagnostic,
    ExecutionCaveatedDiagnostic,
    ExecutionInsufficientDiagnostic,
    SafetyPrecheckPassedDiagnostic,
    SafetyPrecheckFailedDiagnostic,
    SafetyPrecheckBlockedDiagnostic,
    SafetyPrecheckCaveatedDiagnostic,
    SafetyPrecheckUnavailableDiagnostic,
    NonCanonicalInternalOnlyExecutionDiagnostic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainExecutionEligibilityReasonClass {
    EligibleSufficientProposalContextEvidenceMemoryBasis,
    IneligibleInsufficientProposalBasis,
    BlockedStaleOrInvalidatedMemory,
    BlockedMissingContextOrEvidence,
    BlockedSafetyPrecheckFailed,
    CaveatedPartialEvidenceOrMemory,
    UnavailableNoExecutionSubsystem,
    BlockedInternalOnlyDependency,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainExecutionEligibilityDiagnosticLane {
    pub class: BlueBrainExecutionEligibilityDiagnosticClass,
    pub lane: &'static str,
    pub reason_class: BlueBrainExecutionEligibilityReasonClass,
    pub reason_compact: &'static str,
    pub handoff_proposal_binding: &'static str,
    pub selection_deferral_binding: &'static str,
    pub context_evidence_memory_binding: &'static str,
    pub runtime_feedback_binding: &'static str,
    pub boundary_guard: &'static str,
}

pub const WORKFLOW_PATH_INSPECT_DIAGNOSE_ACT: &str =
    "operations_snapshot -> diagnostics assessment -> runtime operation";
pub const WORKFLOW_PATH_REPLAY_ORIENTED: &str =
    "operations_snapshot -> replay_preflight -> replay_with_entry";
pub const WORKFLOW_PATH_ROLLOUT_ORIENTED: &str =
    "operations_snapshot.rollout diagnostics -> activation/fallback/rollback action";
pub const WORKFLOW_PATH_INTERNAL_DEV_TEST_ONLY: &str =
    "run_operation_with_entry(..., InternalDevTest)";

pub const FINAL_REFERENCE_LINE_EXECUTION_CORE: &str =
    "submit -> compute_canonical -> result/fault/status -> execution_snapshot";
pub const FINAL_REFERENCE_LINE_ROLLOUT_EXTENSION: &str =
    "rollout diagnostics -> activation/fallback/rollback -> active production line";
pub const FINAL_REFERENCE_LINE_REPLAY_EXTENSION: &str =
    "replay_preflight -> replay_with_entry -> comparison/evidence on same result/fault/status core";
pub const FINAL_REFERENCE_LINE_DIAGNOSTICS_EXTENSION: &str =
    "runtime snapshot/diagnostics + expert workflow surface -> same canonical core state";
pub const FINAL_REFERENCE_LINE_CROSS_CUTTING_INVARIANTS: &str =
    "blocked!=failed!=no_op; partial/stale/caveated/degraded remain distinct; rollout/replay/expert extend shared core";
pub const FINAL_REFERENCE_NON_CANONICAL_INTERNAL_BOUNDARY: &str =
    "compatibility backends + internal/legacy worker/domain lanes are extension/internal only";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalFinalReferenceLine {
    pub execution_core: &'static str,
    pub rollout_extension: &'static str,
    pub replay_extension: &'static str,
    pub diagnostics_extension: &'static str,
    pub cross_cutting_invariants: &'static str,
    pub internal_boundary: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftPreventionCheckClass {
    ReferenceLineConsistency,
    OutwardFacingContractConsistency,
    SharedCoreSemantics,
    DocCodeAlignment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DriftPreventionCheckLane {
    pub class: DriftPreventionCheckClass,
    pub check_id: &'static str,
    pub guarded_line: &'static str,
    pub check_surface: &'static str,
    pub drift_risk: &'static str,
}

pub const CANONICAL_FINAL_REFERENCE_LINE: CanonicalFinalReferenceLine =
    CanonicalFinalReferenceLine {
        execution_core: FINAL_REFERENCE_LINE_EXECUTION_CORE,
        rollout_extension: FINAL_REFERENCE_LINE_ROLLOUT_EXTENSION,
        replay_extension: FINAL_REFERENCE_LINE_REPLAY_EXTENSION,
        diagnostics_extension: FINAL_REFERENCE_LINE_DIAGNOSTICS_EXTENSION,
        cross_cutting_invariants: FINAL_REFERENCE_LINE_CROSS_CUTTING_INVARIANTS,
        internal_boundary: FINAL_REFERENCE_NON_CANONICAL_INTERNAL_BOUNDARY,
    };

pub const CANONICAL_DRIFT_PREVENTION_CHECK_MAP: [DriftPreventionCheckLane; 4] = [
    DriftPreventionCheckLane {
        class: DriftPreventionCheckClass::ReferenceLineConsistency,
        check_id: "reference_line_consistency",
        guarded_line: FINAL_REFERENCE_LINE_EXECUTION_CORE,
        check_surface: "reference_map::final_reference_doc_and_code_constants_are_kept_in_sync",
        drift_risk: "final reference line text and canonical execution path silently diverge",
    },
    DriftPreventionCheckLane {
        class: DriftPreventionCheckClass::OutwardFacingContractConsistency,
        check_id: "outward_facing_contract_consistency",
        guarded_line: "status_evidence_export_surface + integration_hook_view remain outward-facing",
        check_surface:
            "service_surface::{integration_hook_view_keeps_outward_hooks_read_only_or_caveated,status_evidence_export_surface_keeps_internal_runtime_details_out_of_default_surface}",
        drift_risk: "outward hooks drift into internal/expert-only semantics",
    },
    DriftPreventionCheckLane {
        class: DriftPreventionCheckClass::SharedCoreSemantics,
        check_id: "shared_core_semantics_consistency",
        guarded_line:
            "blocked/failed/no_op and current/partial/stale/caveated/degraded stay non-interchangeable",
        check_surface:
            "contracts::{cross_cutting_invariants_and_outcome_classes_are_explicit,runtime_action_core_semantics_are_stable,evidence_and_trace_partial_caveat_semantics_are_aligned}",
        drift_risk: "load-bearing semantic classes collapse into path-local synonyms",
    },
    DriftPreventionCheckLane {
        class: DriftPreventionCheckClass::DocCodeAlignment,
        check_id: "doc_code_alignment",
        guarded_line: "Serie O maintenance-only boundary stays tied to final reference line",
        check_surface:
            "reference_map::{serie_o_maintenance_boundary_doc_keeps_minimal_change_classes_explicit,serie_o_drift_prevention_checks_doc_stays_tied_to_canonical_line}",
        drift_risk: "docs become a second truth detached from code-pinned invariants",
    },
];

pub const CANONICAL_COMPUTE_INTEGRATION_CONTRACT_VIEW: [ComputeIntegrationContractLane; 6] = [
    ComputeIntegrationContractLane {
        class: ComputeIntegrationContractClass::Execution,
        boundary: ComputeIntegrationBoundary::OutwardFacing,
        lane: "compute_execution_contract",
        canonical_anchor:
            "service_surface::CanonicalComputeEntryPoint::{submit,status,drain_scheduler}",
        semantic_scope: "request/job/run execution on canonical result/fault/status core",
    },
    ComputeIntegrationContractLane {
        class: ComputeIntegrationContractClass::DiagnosticsStatus,
        boundary: ComputeIntegrationBoundary::OutwardFacing,
        lane: "compute_status_diagnostics_contract",
        canonical_anchor:
            "service_surface::CanonicalComputeEntryPoint::status_evidence_export_surface (status)",
        semantic_scope: "runtime state/freshness/drift + top-level diagnostics signals",
    },
    ComputeIntegrationContractLane {
        class: ComputeIntegrationContractClass::EvidenceReference,
        boundary: ComputeIntegrationBoundary::OutwardFacing,
        lane: "compute_evidence_reference_contract",
        canonical_anchor:
            "service_surface::CanonicalComputeEntryPoint::status_evidence_export_surface (evidence)",
        semantic_scope: "snapshot/evidence/trace/history references without redefining run truth",
    },
    ComputeIntegrationContractLane {
        class: ComputeIntegrationContractClass::ExpertInternalOnly,
        boundary: ComputeIntegrationBoundary::ExpertInternalOnly,
        lane: "compute_expert_runtime_control_contract",
        canonical_anchor: "service_surface::{replay_with_entry,run_operation_with_entry}",
        semantic_scope: "expert high-trust replay/runtime operations on shared core invariants",
    },
    ComputeIntegrationContractLane {
        class: ComputeIntegrationContractClass::ExpertInternalOnly,
        boundary: ComputeIntegrationBoundary::ExpertInternalOnly,
        lane: "compatibility_backend_internal_lane",
        canonical_anchor: "backends::build_backend(kind=stub|candle)",
        semantic_scope: "compatibility/dev lane and not an outward-facing contract",
    },
    ComputeIntegrationContractLane {
        class: ComputeIntegrationContractClass::ExpertInternalOnly,
        boundary: ComputeIntegrationBoundary::ExpertInternalOnly,
        lane: "legacy_domains_internal_lane",
        canonical_anchor: "build_backend(kind=worker) + domains/ai*",
        semantic_scope: "legacy compatibility boundary and internal execution entry",
    },
];

pub const CANONICAL_COMPUTE_REFERENCE_MAP: [ComputeReferenceLane; 7] = [
    ComputeReferenceLane {
        class: ComputeReferenceClass::CanonicalProduction,
        lane: "service_entry",
        canonical_path: "service_surface::CanonicalComputeEntryPoint::submit",
        scope: "request/job/run canonical submission and execution",
        shared_core_invariants: "request->job admission; run result/fault/status stays canonical",
    },
    ComputeReferenceLane {
        class: ComputeReferenceClass::CanonicalProduction,
        lane: "pipeline_execution_core",
        canonical_path: "pipeline::ComputePipelineBackend::compute_canonical",
        scope: "result/fault/status core for canonical stage sequence",
        shared_core_invariants: "every run returns canonical pipeline result or failure contract",
    },
    ComputeReferenceLane {
        class: ComputeReferenceClass::CanonicalProduction,
        lane: "rollout_activation_core",
        canonical_path: "enablement::{active,candidate,compare,shadow} + model_store activation",
        scope: "rollout/activation/fallback/rollback core",
        shared_core_invariants:
            "active/candidate/guarded/fallback/rollback semantics stay explicit",
    },
    ComputeReferenceLane {
        class: ComputeReferenceClass::CanonicalExpertRuntimeControl,
        lane: "expert_workflow_surface",
        canonical_path:
            "service_surface::{workflow_view,replay_with_entry,run_operation_with_entry}",
        scope: "expert replay/runtime-control path on canonical contracts",
        shared_core_invariants:
            "expert/internal extend shared action/result invariants; blocked/failed/no-op stay distinct",
    },
    ComputeReferenceLane {
        class: ComputeReferenceClass::CanonicalDiagnosticsEvidence,
        lane: "diagnostics_evidence_history",
        canonical_path: "service_surface + evidence + job_history",
        scope: "snapshot/evidence/diagnostics/replay comparability core",
        shared_core_invariants:
            "current/partial/stale + evidence sufficient/partial/caveated/insufficient + degraded alignment",
    },
    ComputeReferenceLane {
        class: ComputeReferenceClass::InternalOrLegacy,
        lane: "compatibility_backend_lane",
        canonical_path: "backends::build_backend(kind=stub|candle)",
        scope: "compatibility/dev lane; never canonical production default",
        shared_core_invariants:
            "extension lane only; cannot redefine canonical request/job/run contracts",
    },
    ComputeReferenceLane {
        class: ComputeReferenceClass::InternalOrLegacy,
        lane: "internal_worker_legacy_domain_lane",
        canonical_path: "build_backend(kind=worker) + domains/ai* compatibility crates",
        scope: "internal execution lane and legacy compatibility boundary",
        shared_core_invariants:
            "internal/legacy boundary; shared-core contracts remain authoritative",
    },
];

pub const CANONICAL_DOMAIN_FACING_COMPUTE_CONSUMER_MAP: [DomainFacingComputeConsumerLane; 5] = [
    DomainFacingComputeConsumerLane {
        consumer: "runtime_orchestrator_env_bootstrap",
        repo_surface: "runtime/ucf-runtime/src/orchestrator.rs::RuntimeOrchestrator::try_new_from_env",
        execution_contract_path: "build_backend(cfg from env)",
        status_diagnostics_path: "compute summary -> runtime orchestration state",
        evidence_reference_path: "compute_summary.compute_chain_digest + runtime evidence chain",
        status_pattern: DomainFacingStatusConsumptionPattern::MixedLegacyConsumption,
        evidence_pattern: DomainFacingEvidenceConsumptionPattern::MixedLegacyConsumption,
        alignment: DomainFacingConsumerAlignment::NeedsFinalIntegrationAdjustment,
        completion_status: DomainFacingCompletionStatus::MostlyAlignedWithCaveats,
        caveat:
            "load-bearing runtime consumer; supports compat backend kinds and needs progressive canonical submit/status-evidence surface adoption",
    },
    DomainFacingComputeConsumerLane {
        consumer: "ops_compute_probe",
        repo_surface: "runtime/ucf-ops/src/lib.rs::run_compute_probe",
        execution_contract_path:
            "CanonicalComputeEntryPoint::submit(ComputeSubmitRequest{ExecuteInline})",
        status_diagnostics_path:
            "CanonicalComputeEntryPoint::status + status_evidence_export_surface (status)",
        evidence_reference_path:
            "CanonicalComputeEntryPoint::status_evidence_export_surface (evidence refs)",
        status_pattern: DomainFacingStatusConsumptionPattern::CanonicalStatusConsumer,
        evidence_pattern:
            DomainFacingEvidenceConsumptionPattern::CanonicalEvidenceReferenceConsumer,
        alignment: DomainFacingConsumerAlignment::AlignedCanonicalOutward,
        completion_status: DomainFacingCompletionStatus::AlignedToFinalComputeLine,
        caveat:
            "constrained probe: consumes top-level status/evidence signals only, not deep internals",
    },
    DomainFacingComputeConsumerLane {
        consumer: "replay_diff_backend_recompute",
        repo_surface: "runtime/ucf-replay/src/lib.rs::replay_records",
        execution_contract_path: "build_backend(cfg from replay spec) -> backend.compute(...)",
        status_diagnostics_path: "summary/diff policy comparison (no runtime snapshot contract)",
        evidence_reference_path:
            "persisted replay evidence refs + drift reasons (reference-level, not full runtime export)",
        status_pattern: DomainFacingStatusConsumptionPattern::MixedLegacyConsumption,
        evidence_pattern: DomainFacingEvidenceConsumptionPattern::MixedLegacyConsumption,
        alignment: DomainFacingConsumerAlignment::LegacyCompatPath,
        completion_status: DomainFacingCompletionStatus::MixedTransitional,
        caveat:
            "compatibility-oriented replay recompute lane; intentionally not treated as outward-facing runtime service contract",
    },
    DomainFacingComputeConsumerLane {
        consumer: "bench_compute_subcommand",
        repo_surface: "runtime/ucf-bench/src/main.rs::run_compute",
        execution_contract_path: "build_backend(cfg) -> backend.compute(...) loop",
        status_diagnostics_path: "latency/alloc benchmark aggregation only",
        evidence_reference_path: "none (performance harness)",
        status_pattern: DomainFacingStatusConsumptionPattern::InternalDevTestOnly,
        evidence_pattern: DomainFacingEvidenceConsumptionPattern::InternalDevTestOnly,
        alignment: DomainFacingConsumerAlignment::InternalDevTestOnly,
        completion_status: DomainFacingCompletionStatus::InternalOnlyNotTrueOutwardConsumer,
        caveat:
            "benchmark harness path; internal/dev-test only and never a canonical domain integration contract",
    },
    DomainFacingComputeConsumerLane {
        consumer: "domains_ai_compat_lane",
        repo_surface: "domains/ai* + domains/ai-backends compatibility crates",
        execution_contract_path: "legacy host ABI adapters",
        status_diagnostics_path: "legacy compatibility signals only",
        evidence_reference_path: "compat adapter outputs (non-canonical evidence surface)",
        status_pattern: DomainFacingStatusConsumptionPattern::MixedLegacyConsumption,
        evidence_pattern: DomainFacingEvidenceConsumptionPattern::MixedLegacyConsumption,
        alignment: DomainFacingConsumerAlignment::LegacyCompatPath,
        completion_status: DomainFacingCompletionStatus::InternalOnlyNotTrueOutwardConsumer,
        caveat:
            "retained compatibility seam explicitly outside outward-facing canonical compute contracts",
    },
];

pub const CANONICAL_FIRST_DOMAIN_ROLLOUT_CANDIDATE_MAP: [DomainRolloutCandidateLane; 5] = [
    DomainRolloutCandidateLane {
        candidate: "ops_compute_probe",
        rollout_class: DomainRolloutCandidateClass::RolloutReadyCandidate,
        outward_execution_contract:
            "CanonicalComputeEntryPoint::submit(ComputeSubmitRequest{ExecuteInline})",
        outward_status_evidence_surface:
            "CanonicalComputeEntryPoint::status + status_evidence_export_surface",
        integration_safe_hook_posture:
            "integration_hook_view is read_only_integration_safe or caveated_conditional only",
        excluded_internal_or_legacy_paths:
            "does not use build_backend(kind=stub|candle|worker) or domains/ai* compat lanes",
        caveat: "constrained by design: rollout anchor consumes canonical top-level contracts only",
    },
    DomainRolloutCandidateLane {
        candidate: "runtime_orchestrator_env_bootstrap",
        rollout_class: DomainRolloutCandidateClass::RolloutPlausibleWithCaveats,
        outward_execution_contract: "mixed intake today: build_backend(cfg from env)",
        outward_status_evidence_surface:
            "compute summary + runtime evidence chain, not fully canonical export surface yet",
        integration_safe_hook_posture:
            "must stay on integration_hook_view boundary; no expert/internal mutation path rollout",
        excluded_internal_or_legacy_paths:
            "compat backend kinds and legacy env path remain explicitly non-rollout authority",
        caveat:
            "load-bearing path with narrow residual canonicalization needed before rollout-ready",
    },
    DomainRolloutCandidateLane {
        candidate: "replay_diff_backend_recompute",
        rollout_class: DomainRolloutCandidateClass::MixedTransitionalCandidate,
        outward_execution_contract: "build_backend(cfg from replay spec) -> backend.compute(...)",
        outward_status_evidence_surface:
            "replay comparison/evidence refs without outward runtime service status contract",
        integration_safe_hook_posture:
            "replay diagnostics may observe hooks but are not a rollout-facing hook consumer",
        excluded_internal_or_legacy_paths:
            "replay/compat lane is intentionally not an outward service rollout baseline",
        caveat: "technical comparison lane only; keep boundary explicit and non-rollout",
    },
    DomainRolloutCandidateLane {
        candidate: "bench_compute_subcommand",
        rollout_class: DomainRolloutCandidateClass::NotRealRolloutCandidateNow,
        outward_execution_contract: "build_backend(cfg) -> backend.compute(...) loop (benchmark)",
        outward_status_evidence_surface: "benchmark metrics only",
        integration_safe_hook_posture: "internal harness; hook posture not rollout-bearing",
        excluded_internal_or_legacy_paths:
            "internal/dev-test harness intentionally excluded from outward rollout",
        caveat: "not a domain-facing rollout candidate",
    },
    DomainRolloutCandidateLane {
        candidate: "domains_ai_compat_lane",
        rollout_class: DomainRolloutCandidateClass::NotRealRolloutCandidateNow,
        outward_execution_contract: "legacy host ABI adapters",
        outward_status_evidence_surface: "legacy compatibility signals only",
        integration_safe_hook_posture:
            "compat adapters are outside canonical integration-safe hook rollout boundary",
        excluded_internal_or_legacy_paths:
            "domains/ai* compatibility lane remains explicitly legacy/internal-only",
        caveat: "legacy seam retained but not rollout basis on final compute line",
    },
];

pub const CANONICAL_FIRST_DOMAIN_ROLLOUT_COMPLETION_MAP: [FirstDomainRolloutCompletionLane; 1] = [
    FirstDomainRolloutCompletionLane {
        rollout_case: "ops_compute_probe",
        completion_status: FirstDomainRolloutCompletionStatus::Aligned,
        execution_contract_check:
            "uses CanonicalComputeEntryPoint::submit(ComputeSubmitRequest{ExecuteInline}) on submit -> compute_canonical -> result/fault/status -> execution_snapshot",
        outward_status_evidence_check:
            "reads CanonicalComputeEntryPoint::status + status_evidence_export_surface and uses canonical_consumer_view() semantics",
        integration_safe_hook_check:
            "integration_hook_view remains read_only_integration_safe or caveated_conditional and stays non-mutating",
        hidden_legacy_dependency_check:
            "no build_backend(kind=stub|candle|worker) path and no domains/ai* compatibility lane dependency in rollout authority",
        caveat:
            "constrained by design: rollout proof consumes outward-facing status/evidence semantics, not expert internals",
    },
];

pub const CANONICAL_POST_ROLLOUT_ADOPTION_MAP: [PostRolloutAdoptionLane; 6] = [
    PostRolloutAdoptionLane {
        surface: "final_compute_reference_line",
        adoption_class: PostRolloutAdoptionClass::AlreadyAligned,
        rollout_anchor_comparison:
            "final technical production line is already aligned and remains the completed baseline",
        outward_contract_fit:
            "canonical submit -> status/evidence semantics are already established on the final line",
        legacy_internal_dependency_posture:
            "no additional legacy/internal authority is required for baseline alignment",
        caveat:
            "not a broader adoption candidate; keep as fixed baseline without reopening core rollout work",
    },
    PostRolloutAdoptionLane {
        surface: "runtime_orchestrator_env_bootstrap",
        adoption_class: PostRolloutAdoptionClass::BroaderAdoptionReviewCandidate,
        rollout_anchor_comparison:
            "closest load-bearing consumer to first rollout anchor ops_compute_probe; same outward line is reachable with narrow intake canonicalization",
        outward_contract_fit:
            "execution/status-evidence path can be tightened to CanonicalComputeEntryPoint::submit + status_evidence_export_surface without compute-core redesign",
        legacy_internal_dependency_posture:
            "current env/compat intake still mixed; must not rely on build_backend(kind=stub|candle|worker) as outward authority",
        caveat:
            "reviewable only as later adoption candidate if narrowed to the outward-facing contract/evidence semantics already proven by first rollout",
    },
    PostRolloutAdoptionLane {
        surface: "replay_diff_backend_recompute",
        adoption_class: PostRolloutAdoptionClass::BroaderAdoptionReviewCandidate,
        rollout_anchor_comparison:
            "demystified by first rollout as technical comparison support but still not an outward runtime service contract",
        outward_contract_fit:
            "shares compute semantics and evidence references but lacks canonical outward status/service interface as primary consumer contract",
        legacy_internal_dependency_posture:
            "replay/compat pathway remains intentionally distinct from outward rollout authority",
        caveat:
            "keep as review-only candidate; do not treat as established rollout or outward service baseline",
    },
    PostRolloutAdoptionLane {
        surface: "domains_ai_compat_lane",
        adoption_class: PostRolloutAdoptionClass::NotPursuedNow,
        rollout_anchor_comparison:
            "appears adjacent due to historical coupling, but first rollout proof does not transfer to compat adapters",
        outward_contract_fit:
            "legacy host ABI adapters do not provide canonical submit + outward status/evidence semantics",
        legacy_internal_dependency_posture:
            "explicit legacy/internal boundary; compatibility seam retained without rollout authority",
        caveat:
            "explicitly not pursued now to avoid accidental legacy-led adoption expansion",
    },
    PostRolloutAdoptionLane {
        surface: "bench_compute_subcommand",
        adoption_class: PostRolloutAdoptionClass::NotPursuedNow,
        rollout_anchor_comparison:
            "internal benchmark harness and not a domain-facing continuation of first rollout line",
        outward_contract_fit:
            "no outward-facing execution/status/evidence contract responsibilities",
        legacy_internal_dependency_posture:
            "internal dev/test path; not a compatibility authority and not a rollout anchor",
        caveat: "explicitly not pursued now; internal harness remains outside adoption scope",
    },
    PostRolloutAdoptionLane {
        surface: "ops_compute_probe",
        adoption_class: PostRolloutAdoptionClass::FirstRealRolloutEstablished,
        rollout_anchor_comparison:
            "first real rollout anchor already established; serves as baseline reference, not next adoption target",
        outward_contract_fit:
            "already on CanonicalComputeEntryPoint::submit + status_evidence_export_surface + integration_safe hooks",
        legacy_internal_dependency_posture:
            "no hidden legacy/internal dependency in rollout authority path",
        caveat:
            "keep stable as established first rollout baseline; do not reopen as new rollout work",
    },
];

pub const CANONICAL_BLUE_BRAIN_INTEGRATION_MAP: [BlueBrainIntegrationLane; 6] = [
    BlueBrainIntegrationLane {
        surface: "runtime_orchestrator_stateful_loop",
        class: BlueBrainIntegrationClass::RealBlueBrainCoreCandidate,
        repo_surface:
            "runtime/ucf-runtime/src/orchestrator.rs::RuntimeOrchestrator::{try_new_from_env,step_once}",
        execution_contract_path: "target: CanonicalComputeEntryPoint::submit (today partly build_backend env intake)",
        status_diagnostics_contract_path:
            "target: CanonicalComputeEntryPoint::status_evidence_export_surface (status); today mixed compute summary intake",
        evidence_reference_contract_path:
            "target: status_evidence_export_surface (evidence refs) + runtime evidence chain linkage",
        integration_safe_hook_posture:
            "must remain bounded to integration_hook_view (read_only_integration_safe|caveated_conditional)",
        coupling_posture:
            "real stateful orchestration surface with technical compute dependence; currently caveated due to mixed intake path",
        caveat:
            "primary Blue-Brain integration candidate only if progressive canonicalization removes residual env/compat intake",
    },
    BlueBrainIntegrationLane {
        surface: "ops_compute_probe",
        class: BlueBrainIntegrationClass::BlueBrainAdjacentComputeConsumer,
        repo_surface: "runtime/ucf-ops/src/lib.rs::run_compute_probe",
        execution_contract_path:
            "CanonicalComputeEntryPoint::submit(ComputeSubmitRequest{ExecuteInline})",
        status_diagnostics_contract_path:
            "CanonicalComputeEntryPoint::status + status_evidence_export_surface (status)",
        evidence_reference_contract_path:
            "CanonicalComputeEntryPoint::status_evidence_export_surface (evidence refs)",
        integration_safe_hook_posture:
            "reads integration_hook_view classification only; no mutating or expert-only semantics",
        coupling_posture:
            "clean outward-facing compute consumer and reference anchor, but not itself a Blue-Brain core loop",
        caveat:
            "adjacent integration anchor for contract stability checks; not a stateful Blue-Brain orchestration kernel",
    },
    BlueBrainIntegrationLane {
        surface: "replay_diff_backend_recompute",
        class: BlueBrainIntegrationClass::IndirectOrCompatibilityTouchingSurface,
        repo_surface: "runtime/ucf-replay/src/lib.rs::replay_records",
        execution_contract_path: "build_backend(cfg from replay spec) -> backend.compute(...)",
        status_diagnostics_contract_path:
            "replay diff/status heuristics (no canonical outward status contract as primary surface)",
        evidence_reference_contract_path:
            "replay-local evidence refs; not canonical outward evidence export as primary consumer contract",
        integration_safe_hook_posture:
            "diagnostic observation only; not an integration-safe hook consumer contract",
        coupling_posture:
            "indirect comparability/recompute support with legacy/compat characteristics",
        caveat:
            "useful as diagnostics adjunct only; should not be promoted to primary Blue-Brain compute integration",
    },
    BlueBrainIntegrationLane {
        surface: "domains_ai_compat_lane",
        class: BlueBrainIntegrationClass::IndirectOrCompatibilityTouchingSurface,
        repo_surface: "domains/ai* + domains/ai-backends compatibility crates",
        execution_contract_path: "legacy host ABI adapters",
        status_diagnostics_contract_path: "legacy compatibility signals",
        evidence_reference_contract_path: "compat adapter outputs (non-canonical export semantics)",
        integration_safe_hook_posture:
            "outside canonical integration_hook_view semantics and not outward integration-safe authority",
        coupling_posture:
            "legacy compatibility seam adjacent to compute but not a canonical Blue-Brain integration lane",
        caveat:
            "retain only as compatibility boundary; no Blue-Brain core or rollout authority",
    },
    BlueBrainIntegrationLane {
        surface: "bench_compute_subcommand",
        class: BlueBrainIntegrationClass::InternalOnlyOrNotMeaningfulForBlueBrainIntegration,
        repo_surface: "runtime/ucf-bench/src/main.rs::run_compute",
        execution_contract_path: "build_backend(cfg) -> backend.compute(...) benchmark loop",
        status_diagnostics_contract_path: "benchmark-only latency/allocation metrics",
        evidence_reference_contract_path: "none",
        integration_safe_hook_posture: "internal/dev-test harness only",
        coupling_posture:
            "internal benchmark path can touch compute but has no Blue-Brain integration semantics",
        caveat: "explicitly excluded from Blue-Brain integration scope",
    },
    BlueBrainIntegrationLane {
        surface: "runtime_hooks_and_frame_helpers",
        class: BlueBrainIntegrationClass::InternalOnlyOrNotMeaningfulForBlueBrainIntegration,
        repo_surface: "runtime/ucf-runtime/src/hooks.rs + domains/ucf-frames/src/v1/*",
        execution_contract_path: "none (helper/summary adaptation path)",
        status_diagnostics_contract_path: "frame/helper reads of compute summary signals",
        evidence_reference_contract_path: "digest/reference field carrying only",
        integration_safe_hook_posture:
            "internal data/helper boundary; integration_hook_view remains canonical outward hook boundary",
        coupling_posture:
            "schema/helper proximity to compute signals but no standalone outward compute-consumer contract",
        caveat:
            "do not treat helper proximity as Blue-Brain core integration readiness",
    },
];

pub const CANONICAL_BLUE_BRAIN_FACING_CONTRACT_MAP: [BlueBrainFacingContractLane; 5] = [
    BlueBrainFacingContractLane {
        class: BlueBrainFacingContractClass::InferenceFacing,
        lane: "blue_brain_inference_facing_execution_contract",
        canonical_anchor:
            "service_surface::CanonicalComputeEntryPoint::submit(ComputeSubmitRequest{ExecuteInline})",
        allowed_semantics:
            "canonical execution via submit -> compute_canonical -> result/fault/status; no second execution world",
        excluded_semantics:
            "no direct build_backend(kind=stub|candle|worker) authority and no replay/expert operation semantics as default inference API",
    },
    BlueBrainFacingContractLane {
        class: BlueBrainFacingContractClass::StateFacing,
        lane: "blue_brain_state_facing_context_reference_contract",
        canonical_anchor:
            "compute request context_digest + runtime_handoff_state_from_evidence/runtime_handoff_state_from_action_code",
        allowed_semantics:
            "state-adjacent reference/context handoff only; outward context linkage without leaking runtime-internal structs",
        excluded_semantics:
            "no speculative cognitive-state architecture and no direct runtime scheduler or in-memory orchestration internals exposed",
    },
    BlueBrainFacingContractLane {
        class: BlueBrainFacingContractClass::StatusHealthTrustFacing,
        lane: "blue_brain_status_health_trust_contract",
        canonical_anchor:
            "service_surface::CanonicalComputeEntryPoint::status + status_evidence_export_surface (status)",
        allowed_semantics:
            "top-level current/partial/stale/caveated/degraded plus trust/service state signals on canonical surface",
        excluded_semantics:
            "no internal diagnostic graph ownership and no expert workflow control semantics in outward status contract",
    },
    BlueBrainFacingContractLane {
        class: BlueBrainFacingContractClass::EvidenceReferenceFacing,
        lane: "blue_brain_evidence_reference_contract",
        canonical_anchor:
            "service_surface::CanonicalComputeEntryPoint::status_evidence_export_surface (evidence refs)",
        allowed_semantics:
            "snapshot/evidence/trace/history references including partial/caveated evidence posture",
        excluded_semantics:
            "no raw internal diagnostics/trace object export as required Blue-Brain-facing contract payload",
    },
    BlueBrainFacingContractLane {
        class: BlueBrainFacingContractClass::ExpertInternalOnlyNonBlueBrain,
        lane: "blue_brain_expert_internal_only_non_contract",
        canonical_anchor:
            "service_surface::{replay_with_entry,run_operation_with_entry} + backends::build_backend(kind=stub|candle|worker) + domains/ai*",
        allowed_semantics:
            "expert/internal diagnostics-control and compatibility execution lanes remain explicitly non Blue-Brain-facing",
        excluded_semantics:
            "must not be presented as canonical Blue-Brain-facing integration contract",
    },
];

pub const CANONICAL_BLUE_BRAIN_COMPUTE_HANDOFF_MAP: [BlueBrainComputeHandoffLane; 5] = [
    BlueBrainComputeHandoffLane {
        class: BlueBrainComputeHandoffClass::InferenceHandoff,
        lane: "blue_brain_to_compute_inference_handoff",
        canonical_transition:
            "CanonicalComputeEntryPoint::submit(ComputeSubmitRequest{ExecuteInline}) -> compute_canonical -> result/fault/status",
        outbound_payload_shape:
            "submit request envelope only (canonical request + mode), no expert/internal operation payload",
        return_payload_shape:
            "canonical result/fault/status + execution snapshot semantics on same outward execution line",
        canonical_references:
            "request/run identity via ComputeJobHandle + outward status semantics + bounded evidence linkage",
        non_canonical_boundary:
            "exclude replay_with_entry/run_operation_with_entry/build_backend from default inference handoff",
    },
    BlueBrainComputeHandoffLane {
        class: BlueBrainComputeHandoffClass::StatusDiagnosticsHandoff,
        lane: "blue_brain_to_compute_status_diagnostics_handoff",
        canonical_transition:
            "CanonicalComputeEntryPoint::status + status_evidence_export_surface(status) -> top-level service/trust/status view",
        outbound_payload_shape:
            "status probe request only; no ownership transfer of internal diagnostic graphs",
        return_payload_shape:
            "current|partial|stale|caveated|degraded + trust/service state on canonical outward surface",
        canonical_references:
            "outward status references + runtime snapshot status semantics aligned to final compute line",
        non_canonical_boundary:
            "exclude internal-only diagnostic objects and expert workflow internals from canonical handoff payload",
    },
    BlueBrainComputeHandoffLane {
        class: BlueBrainComputeHandoffClass::EvidenceReferenceHandoff,
        lane: "blue_brain_to_compute_evidence_reference_handoff",
        canonical_transition:
            "CanonicalComputeEntryPoint::status_evidence_export_surface(evidence refs) -> bounded snapshot/evidence/trace/history references",
        outbound_payload_shape:
            "reference consumption request only; no raw internal trace object requirement",
        return_payload_shape:
            "evidence bundle references + trace/evidence references with partial/caveated posture where applicable",
        canonical_references:
            "snapshot/evidence references + trace slice references + history/replay-comparison refs where outward relevant",
        non_canonical_boundary:
            "exclude internal diagnostics blobs/audit platform payloads as mandatory Blue-Brain-facing handoff data",
    },
    BlueBrainComputeHandoffLane {
        class: BlueBrainComputeHandoffClass::StateAdjacentReferenceHandoff,
        lane: "blue_brain_to_compute_state_adjacent_reference_handoff",
        canonical_transition:
            "context_digest + runtime_handoff_state_from_evidence/runtime_handoff_state_from_action_code reference mapping",
        outbound_payload_shape:
            "context/reference linkage only; no direct runtime scheduler or in-memory orchestration struct leakage",
        return_payload_shape:
            "state-adjacent handoff state refs (complete|partial|caveated|blocked) derived from canonical evidence/action semantics",
        canonical_references:
            "request context_digest + runtime handoff state references + active production context where load-bearing",
        non_canonical_boundary:
            "exclude speculative cognitive-state platform semantics and compute-internal runtime structs",
    },
    BlueBrainComputeHandoffLane {
        class: BlueBrainComputeHandoffClass::ExpertInternalOnlyNonCanonicalHandoff,
        lane: "blue_brain_non_canonical_expert_internal_handoff",
        canonical_transition:
            "replay_with_entry/run_operation_with_entry + build_backend(kind=stub|candle|worker) remain expert/internal lanes",
        outbound_payload_shape:
            "expert/internal controls and compat adapters only; not default outward handoff authority",
        return_payload_shape:
            "internal diagnostics/operation outcomes can exist, but are never canonical Blue-Brain-facing standard payload",
        canonical_references:
            "must down-map to outward canonical status/evidence references before any Blue-Brain-facing use",
        non_canonical_boundary:
            "explicit non-canonical boundary: never advertise expert/internal lanes as default Blue-Brain-to-compute handoff",
    },
];

pub const CANONICAL_BLUE_BRAIN_INTEGRATION_CANDIDATE_MAP: [BlueBrainIntegrationCandidateLane; 4] = [
    BlueBrainIntegrationCandidateLane {
        surface: "runtime_orchestrator_stateful_loop",
        class: BlueBrainIntegrationCandidateClass::PlausibleWithCaveats,
        candidate_selection_posture:
            "selected_first_real_blue_brain_integration_candidate: closest real stateful Blue-Brain-facing surface on final compute line",
        inference_contract_binding:
            "blue_brain_to_compute_inference_handoff -> CanonicalComputeEntryPoint::submit(ComputeSubmitRequest{ExecuteInline})",
        status_handoff_binding:
            "blue_brain_to_compute_status_diagnostics_handoff -> status + status_evidence_export_surface(status)",
        evidence_handoff_binding:
            "blue_brain_to_compute_evidence_reference_handoff -> status_evidence_export_surface(evidence refs) + runtime evidence chain linkage",
        state_adjacent_binding:
            "blue_brain_to_compute_state_adjacent_reference_handoff -> context_digest + runtime_handoff_state_from_evidence/runtime_handoff_state_from_action_code",
        excluded_internal_or_legacy_paths:
            "exclude replay_with_entry/run_operation_with_entry/build_backend(kind=stub|candle|worker) + domains/ai* as candidate authority",
        caveat:
            "remains caveated until residual mixed env/compat intake in orchestrator setup is fully canonicalized",
    },
    BlueBrainIntegrationCandidateLane {
        surface: "ops_compute_probe",
        class: BlueBrainIntegrationCandidateClass::IntegrationReadyCandidate,
        candidate_selection_posture:
            "integration-ready adjacent anchor for canonical outward contract/handoff checks",
        inference_contract_binding:
            "CanonicalComputeEntryPoint::submit(ComputeSubmitRequest{ExecuteInline})",
        status_handoff_binding:
            "CanonicalComputeEntryPoint::status + status_evidence_export_surface(status)",
        evidence_handoff_binding:
            "CanonicalComputeEntryPoint::status_evidence_export_surface(evidence refs)",
        state_adjacent_binding:
            "state-adjacent semantics not load-bearing here; treated as compute context reference passthrough only",
        excluded_internal_or_legacy_paths:
            "no expert/internal-only lane or compat adapters in primary outward consumer path",
        caveat:
            "not a stateful Blue-Brain orchestration kernel; remains adjacent compute consumer",
    },
    BlueBrainIntegrationCandidateLane {
        surface: "replay_diff_backend_recompute",
        class: BlueBrainIntegrationCandidateClass::MixedTransitionalCandidate,
        candidate_selection_posture:
            "mixed/transitional diagnostics lane; useful comparison support but not canonical Blue-Brain baseline",
        inference_contract_binding:
            "indirect backend.compute(...) path; no canonical submit authority as primary lane",
        status_handoff_binding:
            "replay diff/status heuristics instead of canonical outward status handoff",
        evidence_handoff_binding:
            "replay-local references; not canonical outward evidence export surface as primary contract",
        state_adjacent_binding:
            "no canonical state-adjacent handoff contract ownership",
        excluded_internal_or_legacy_paths:
            "must not be promoted as first Blue-Brain integration basis",
        caveat:
            "acceptable only as diagnostics adjunct under canonical candidate, never as outward baseline",
    },
    BlueBrainIntegrationCandidateLane {
        surface: "domains_ai_compat_lane + bench_compute_subcommand + runtime_hooks_and_frame_helpers",
        class: BlueBrainIntegrationCandidateClass::NotRealBlueBrainIntegrationCandidateNow,
        candidate_selection_posture:
            "explicit exclusion bucket to prevent internal/compat/helper drift into Blue-Brain integration claims",
        inference_contract_binding:
            "legacy host ABI adapters/internal benchmark/helper paths; no canonical outward inference authority",
        status_handoff_binding:
            "compat/internal diagnostics only",
        evidence_handoff_binding:
            "non-canonical or absent outward evidence semantics",
        state_adjacent_binding:
            "helper proximity only; no Blue-Brain-facing state-adjacent contract ownership",
        excluded_internal_or_legacy_paths:
            "explicitly excluded from first real Blue-Brain integration candidate scope",
        caveat:
            "retain only as boundaries; do not market or interpret as integration candidate progress",
    },
];

pub const CANONICAL_BLUE_BRAIN_RUNTIME_SURFACE_MAP: [BlueBrainRuntimeSurfaceLane; 5] = [
    BlueBrainRuntimeSurfaceLane {
        class: BlueBrainRuntimeSurfaceClass::StateBearingSurface,
        lane: "blue_brain_state_bearing_surface",
        canonical_anchor:
            "runtime/ucf-runtime/src/orchestrator.rs::RuntimeOrchestrator::{try_new_from_env,step_once}",
        runtime_scope:
            "state/context bearing orchestration around compute request context_digest and handoff state references",
        compute_line_binding:
            "context linkage references CanonicalComputeEntryPoint submit/status-evidence semantics but does not redefine compute internals",
        boundary_guard:
            "no direct export of runtime scheduler internals or speculative cognitive-state matrix",
    },
    BlueBrainRuntimeSurfaceLane {
        class: BlueBrainRuntimeSurfaceClass::InferenceBearingSurface,
        lane: "blue_brain_inference_bearing_surface",
        canonical_anchor:
            "service_surface::CanonicalComputeEntryPoint::submit(ComputeSubmitRequest{ExecuteInline})",
        runtime_scope: "compute invocation handoff for Blue-Brain-facing inference-bearing runtime step",
        compute_line_binding:
            "submit -> compute_canonical -> result/fault/status on final compute reference line",
        boundary_guard:
            "exclude replay_with_entry/run_operation_with_entry/build_backend as default inference runtime surface",
    },
    BlueBrainRuntimeSurfaceLane {
        class: BlueBrainRuntimeSurfaceClass::StatusHealthTrustFacingSurface,
        lane: "blue_brain_status_health_trust_surface",
        canonical_anchor:
            "service_surface::CanonicalComputeEntryPoint::status + status_evidence_export_surface(status)",
        runtime_scope: "runtime-relevant current/partial/stale/caveated/degraded plus service trust posture",
        compute_line_binding:
            "outward status/evidence export surface remains read-only/caveated integration contract",
        boundary_guard:
            "exclude internal diagnostic graph ownership and expert control semantics from canonical status surface",
    },
    BlueBrainRuntimeSurfaceLane {
        class: BlueBrainRuntimeSurfaceClass::EvidenceReplayFacingSurface,
        lane: "blue_brain_evidence_replay_facing_surface",
        canonical_anchor:
            "service_surface::CanonicalComputeEntryPoint::status_evidence_export_surface (evidence refs + history/replay refs)",
        runtime_scope:
            "evidence/reference uptake for runtime replayability and diagnostics anchoring (sufficient|partial|caveated|insufficient)",
        compute_line_binding:
            "bounded evidence references tied to canonical run/action evidence bundle semantics",
        boundary_guard:
            "exclude raw internal diagnostics blobs as required Blue-Brain runtime payload",
    },
    BlueBrainRuntimeSurfaceLane {
        class: BlueBrainRuntimeSurfaceClass::InternalOnlyRuntimeControlSurface,
        lane: "blue_brain_internal_only_runtime_control_surface",
        canonical_anchor:
            "service_surface::{replay_with_entry,run_operation_with_entry} + backends::build_backend(kind=stub|candle|worker) + domains/ai*",
        runtime_scope:
            "expert/internal diagnostics-control and compatibility paths on shared compute semantics",
        compute_line_binding:
            "must down-map to outward status/evidence references before Blue-Brain-facing usage",
        boundary_guard:
            "explicitly non-canonical Blue-Brain runtime surface; never default outward authority",
    },
];

pub const CANONICAL_BLUE_BRAIN_RUNTIME_PHASE_MAP: [BlueBrainRuntimePhaseLane; 5] = [
    BlueBrainRuntimePhaseLane {
        class: BlueBrainRuntimePhaseClass::StateContextAvailable,
        lane: "blue_brain_phase_state_context_available",
        phase_transition:
            "state/context prepared -> request context_digest + handoff state reference becomes available",
        canonical_inputs:
            "runtime_orchestrator_stateful_loop state/context and reference-level handoff state",
        canonical_outputs:
            "state-adjacent context reference ready for canonical compute invocation",
        non_goal_boundary:
            "no compute-internal runtime struct modeling and no speculative cognitive pipeline expansion",
    },
    BlueBrainRuntimePhaseLane {
        class: BlueBrainRuntimePhaseClass::ComputeInvocationRequested,
        lane: "blue_brain_phase_compute_invocation_requested",
        phase_transition:
            "compute invocation requested via CanonicalComputeEntryPoint::submit(ComputeSubmitRequest{ExecuteInline})",
        canonical_inputs: "state/context reference + canonical submit request envelope",
        canonical_outputs:
            "ComputeJobHandle identity plus canonical run request on final compute execution line",
        non_goal_boundary:
            "no side-entry replay/runtime-operation lane as default runtime phase",
    },
    BlueBrainRuntimePhaseLane {
        class: BlueBrainRuntimePhaseClass::ComputeResultIntegrated,
        lane: "blue_brain_phase_compute_result_integrated",
        phase_transition:
            "canonical compute result/fault/status integrated back into Blue-Brain runtime state",
        canonical_inputs:
            "submit result tuple + status semantics from shared result/fault/status core",
        canonical_outputs:
            "runtime-facing result integration with explicit complete|partial|caveated|blocked handoff-state references",
        non_goal_boundary:
            "no second compute truth model and no compute-core semantic fork",
    },
    BlueBrainRuntimePhaseLane {
        class: BlueBrainRuntimePhaseClass::StatusEvidenceObserved,
        lane: "blue_brain_phase_status_evidence_observed",
        phase_transition:
            "status/evidence observed via status + status_evidence_export_surface(status/evidence refs)",
        canonical_inputs:
            "top-level status/trust signals + evidence bundle references + trace/history refs",
        canonical_outputs:
            "runtime-visible status/evidence uptake anchored in outward-facing compute contracts",
        non_goal_boundary:
            "no separate monitoring platform or mandatory raw diagnostics payload ingestion",
    },
    BlueBrainRuntimePhaseLane {
        class: BlueBrainRuntimePhaseClass::CaveatedOrDegradedOrPartialRuntimeState,
        lane: "blue_brain_phase_caveated_degraded_partial_runtime_state",
        phase_transition:
            "runtime posture enters caveated/degraded/partial state when outward status/evidence signals are stale or insufficient",
        canonical_inputs:
            "current|partial|stale|caveated|degraded status + sufficient|partial|caveated|insufficient evidence posture",
        canonical_outputs:
            "explicit runtime caveat/degraded marker without hidden expert/internal escalation",
        non_goal_boundary:
            "no implicit high-trust fallback authority through expert/internal runtime control surfaces",
    },
];

pub const CANONICAL_BLUE_BRAIN_TRANSITION_TRIGGER_MAP: [BlueBrainTransitionTriggerLane; 11] = [
    BlueBrainTransitionTriggerLane {
        class: BlueBrainTransitionTriggerClass::PureStateTransition,
        lane: "blue_brain_transition_context_available",
        canonical_transition:
            "state/context prepared and available -> context reference published without compute invocation",
        trigger_point: "context available transition only; no compute trigger implied",
        canonical_contract_binding:
            "state-facing reference continuity only; submit remains an explicit later transition",
        reference_continuity:
            "preserve context digest references, handoff-state references, and active context posture",
        non_canonical_boundary:
            "context availability must not be interpreted as persistent memory commit",
    },
    BlueBrainTransitionTriggerLane {
        class: BlueBrainTransitionTriggerClass::PureStateTransition,
        lane: "blue_brain_transition_state_context_refreshed",
        canonical_transition:
            "runtime state/context refresh -> context_digest and handoff state references updated",
        trigger_point: "pure transition only; no compute trigger",
        canonical_contract_binding:
            "state-facing reference continuity only; no direct submit call on this transition",
        reference_continuity:
            "request/run identity not yet minted; preserve active production context and state references",
        non_canonical_boundary:
            "must not escalate through helper/internal lanes to force compute from state refresh alone",
    },
    BlueBrainTransitionTriggerLane {
        class: BlueBrainTransitionTriggerClass::ComputeTriggeringTransition,
        lane: "blue_brain_transition_context_used_for_compute_trigger",
        canonical_transition:
            "available context references are consumed to satisfy canonical compute-trigger preconditions",
        trigger_point:
            "context reference is used for trigger qualification; compute trigger remains explicit",
        canonical_contract_binding:
            "CanonicalComputeEntryPoint::submit(ComputeSubmitRequest{ExecuteInline})",
        reference_continuity:
            "context/state references are carried as trigger inputs, not treated as memory writes",
        non_canonical_boundary:
            "no memory persistence implied by context usage during trigger admission",
    },
    BlueBrainTransitionTriggerLane {
        class: BlueBrainTransitionTriggerClass::ComputeTriggeringTransition,
        lane: "blue_brain_transition_compute_trigger_from_context_availability",
        canonical_transition:
            "state/context available -> runtime requests compute through canonical submit",
        trigger_point:
            "trigger from state/context availability when context_digest + handoff references are present",
        canonical_contract_binding:
            "CanonicalComputeEntryPoint::submit(ComputeSubmitRequest{ExecuteInline})",
        reference_continuity:
            "carry request/run identity, state handoff references, and active production context into submit",
        non_canonical_boundary:
            "no replay_with_entry/run_operation_with_entry/build_backend side-trigger as default path",
    },
    BlueBrainTransitionTriggerLane {
        class: BlueBrainTransitionTriggerClass::ComputeTriggeringTransition,
        lane: "blue_brain_transition_compute_trigger_from_inference_required",
        canonical_transition:
            "runtime enters inference-required transition -> canonical compute invocation is requested",
        trigger_point:
            "trigger from inference-required transition only on outward-facing execution contract",
        canonical_contract_binding:
            "CanonicalComputeEntryPoint::submit + result/fault/status core semantics",
        reference_continuity:
            "propagate request/run identity and prior status/evidence references into canonical run admission",
        non_canonical_boundary:
            "no implicit helper object or internal diagnostic graph may satisfy inference trigger requirements",
    },
    BlueBrainTransitionTriggerLane {
        class: BlueBrainTransitionTriggerClass::ComputeTriggeringTransition,
        lane: "blue_brain_transition_compute_trigger_blocked_insufficient_context",
        canonical_transition:
            "inference-required transition with missing context/state -> trigger remains blocked",
        trigger_point:
            "trigger blocked due to insufficient context/state; no canonical compute invocation is emitted",
        canonical_contract_binding:
            "status + status_evidence_export_surface report blocked/caveated posture without hidden submit",
        reference_continuity:
            "preserve request intent reference and insufficiency evidence reference for next eligible transition",
        non_canonical_boundary:
            "must not unblock through internal-only expert hook or compatibility adapter",
    },
    BlueBrainTransitionTriggerLane {
        class: BlueBrainTransitionTriggerClass::InternalOnlyOrNonCanonicalTransition,
        lane: "blue_brain_transition_compute_trigger_suppressed_internal_only_path",
        canonical_transition:
            "internal/expert path would satisfy trigger preconditions but remains suppressed for canonical runtime",
        trigger_point:
            "trigger suppressed because only internal/expert lane could satisfy missing prerequisites",
        canonical_contract_binding:
            "non-canonical lane must down-map to outward status/evidence references before any Blue-Brain-facing use",
        reference_continuity:
            "retain state/status/evidence references while canonical trigger stays unresolved",
        non_canonical_boundary:
            "explicit non-canonical boundary: no default Blue-Brain trigger authority for expert/internal lanes",
    },
    BlueBrainTransitionTriggerLane {
        class: BlueBrainTransitionTriggerClass::EvidenceStatusUpdateTransition,
        lane: "blue_brain_transition_compute_result_integrated",
        canonical_transition:
            "compute result/fault/status received -> runtime integrates result transition without changing trigger authority",
        trigger_point:
            "compute result integrated transition after canonical submit completion",
        canonical_contract_binding:
            "submit result/fault/status + status_evidence_export_surface(status/evidence refs)",
        reference_continuity:
            "join run identity with outward status references, evidence references, and active production context",
        non_canonical_boundary:
            "no internal diagnostics blob adoption as required Blue-Brain payload",
    },
    BlueBrainTransitionTriggerLane {
        class: BlueBrainTransitionTriggerClass::EvidenceStatusUpdateTransition,
        lane: "blue_brain_transition_evidence_observed_without_memory_commit",
        canonical_transition:
            "evidence bundle or replay basis observed -> evidence/reference uptake only",
        trigger_point: "evidence observed transition without memory commit",
        canonical_contract_binding:
            "CanonicalComputeEntryPoint::status_evidence_export_surface(evidence refs + replay refs)",
        reference_continuity:
            "retain evidence/replay references as outward references, not as persisted memory entries",
        non_canonical_boundary:
            "evidence/replay observation must not be represented as memory persistence",
    },
    BlueBrainTransitionTriggerLane {
        class: BlueBrainTransitionTriggerClass::EvidenceStatusUpdateTransition,
        lane: "blue_brain_transition_memory_adjacent_candidate_identified_not_committed",
        canonical_transition:
            "memory-adjacent candidate recognized from context/evidence linkage without storage action",
        trigger_point: "memory-adjacent candidate identification only; no memory commit",
        canonical_contract_binding:
            "status + evidence/reference contracts remain authoritative; no memory subsystem contract in this series",
        reference_continuity:
            "preserve candidate references for future BB3 work while keeping current runtime deterministic",
        non_canonical_boundary:
            "explicitly no long-term memory persistence, vector-db write, or cognitive-state storage claim",
    },
    BlueBrainTransitionTriggerLane {
        class: BlueBrainTransitionTriggerClass::EvidenceStatusUpdateTransition,
        lane: "blue_brain_transition_status_evidence_update_without_compute_trigger",
        canonical_transition:
            "status/evidence update observed (including caveated/degraded/partial) -> runtime state update only",
        trigger_point:
            "evidence/status update transition without new compute trigger",
        canonical_contract_binding:
            "CanonicalComputeEntryPoint::status + status_evidence_export_surface(status/evidence refs)",
        reference_continuity:
            "preserve request/run identity links when available; otherwise keep outward status/evidence references stable",
        non_canonical_boundary:
            "must not auto-trigger compute through legacy/compat/internal helper paths on status-only updates",
    },
];

pub const CANONICAL_BLUE_BRAIN_CONTEXT_MEMORY_BOUNDARY_MAP: [BlueBrainContextMemoryBoundaryLane; 7] =
    [
        BlueBrainContextMemoryBoundaryLane {
            class: BlueBrainContextMemoryBoundaryClass::PureComputeConsumer,
            lane: "blue_brain_pure_compute_consumer_ops_probe",
            surface: "ops_compute_probe",
            canonical_anchor: "runtime/ucf-ops/src/lib.rs::run_compute_probe",
            compute_invocation_reference:
                "CanonicalComputeEntryPoint::submit(ComputeSubmitRequest{ExecuteInline})",
            context_reference: "context digest passthrough only; no runtime-owned context state",
            evidence_or_replay_reference:
                "optional status/evidence references consumed for probe diagnostics",
            memory_posture:
                "no memory-adjacent semantics; compute invocation and diagnostics only",
            boundary_guard:
                "must not be promoted as Blue-Brain context/memory authority",
        },
        BlueBrainContextMemoryBoundaryLane {
            class: BlueBrainContextMemoryBoundaryClass::ContextBearingSurface,
            lane: "blue_brain_context_bearing_runtime_orchestrator",
            surface: "runtime_orchestrator_stateful_loop",
            canonical_anchor:
                "runtime/ucf-runtime/src/orchestrator.rs::RuntimeOrchestrator::{try_new_from_env,step_once}",
            compute_invocation_reference:
                "context-bearing runtime step may invoke CanonicalComputeEntryPoint::submit",
            context_reference:
                "state/context references (context_digest + runtime_handoff_state) are runtime-local and bounded",
            evidence_or_replay_reference:
                "status/evidence references are consumed via outward export surfaces",
            memory_posture:
                "context-bearing only; no persistent memory subsystem contract",
            boundary_guard:
                "state/context reference must not be relabeled as memory persistence",
        },
        BlueBrainContextMemoryBoundaryLane {
            class: BlueBrainContextMemoryBoundaryClass::MemoryAdjacentSurface,
            lane: "blue_brain_memory_adjacent_context_integration_candidate",
            surface: "runtime_handoff_state_from_evidence + transition trigger map",
            canonical_anchor:
                "service_surface::runtime_handoff_state_from_evidence + blue_brain_transition_memory_adjacent_candidate_identified_not_committed",
            compute_invocation_reference:
                "compute invocation remains explicit and independent from memory-adjacent candidate detection",
            context_reference:
                "context integration keeps current-runtime references only",
            evidence_or_replay_reference:
                "uses outward evidence/replay references as candidate basis",
            memory_posture:
                "memory-adjacent candidate only; explicitly not committed or persisted",
            boundary_guard:
                "prepares BB3 boundary without introducing storage/model-memory architecture",
        },
        BlueBrainContextMemoryBoundaryLane {
            class: BlueBrainContextMemoryBoundaryClass::EvidenceReferenceConsumer,
            lane: "blue_brain_evidence_reference_consumer_surface",
            surface: "status_evidence_export_surface evidence/ref uptake",
            canonical_anchor:
                "service_surface::CanonicalComputeEntryPoint::status_evidence_export_surface",
            compute_invocation_reference:
                "no implicit compute trigger on evidence-only uptake",
            context_reference:
                "evidence may update runtime context posture but is not context ownership by itself",
            evidence_or_replay_reference:
                "bundle_refs + trace_refs + history/replay references remain reference-grade",
            memory_posture:
                "no memory persistence implied by evidence/replay references",
            boundary_guard:
                "evidence references are not memory records and not memory commits",
        },
        BlueBrainContextMemoryBoundaryLane {
            class: BlueBrainContextMemoryBoundaryClass::EvidenceReferenceConsumer,
            lane: "blue_brain_replay_reference_basis_surface",
            surface: "replay/history reference basis",
            canonical_anchor:
                "service_surface::{replay_preflight,replay_with_entry} diagnostics bound to outward references",
            compute_invocation_reference:
                "replay/reference basis remains diagnostics context and does not auto-trigger canonical submit",
            context_reference:
                "provides replay/reference basis for runtime context comparisons",
            evidence_or_replay_reference:
                "replay comparison refs + context bridge refs are consumed as evidence basis",
            memory_posture:
                "replay/reference basis is not persistent memory and not a substitute memory store",
            boundary_guard:
                "must be down-mapped to outward references before any Blue-Brain-facing use",
        },
        BlueBrainContextMemoryBoundaryLane {
            class: BlueBrainContextMemoryBoundaryClass::InternalOnlyOrNonCanonicalContextPath,
            lane: "blue_brain_internal_or_expert_only_context_path",
            surface: "internal/expert runtime control paths",
            canonical_anchor:
                "service_surface::{run_operation_with_entry,replay_with_entry} + backends::build_backend(kind=stub|candle|worker) + domains/ai*",
            compute_invocation_reference:
                "non-canonical invocation path; not Blue-Brain default compute authority",
            context_reference:
                "expert/internal context details are non-canonical for Blue-Brain runtime contracts",
            evidence_or_replay_reference:
                "must be remapped to outward status/evidence references before external consumption",
            memory_posture:
                "not eligible as memory-adjacent Blue-Brain surface",
            boundary_guard:
                "explicit non-canonical boundary for context/memory integration scope",
        },
        BlueBrainContextMemoryBoundaryLane {
            class: BlueBrainContextMemoryBoundaryClass::ContextBearingSurface,
            lane: "blue_brain_context_uptake_without_memory_commit",
            surface: "compute result integrated into current runtime context",
            canonical_anchor:
                "blue_brain_transition_compute_result_integrated + blue_brain_transition_evidence_observed_without_memory_commit",
            compute_invocation_reference:
                "compute result integration consumes prior canonical submit output",
            context_reference:
                "updates current context posture and handoff-state references",
            evidence_or_replay_reference:
                "captures evidence/reference uptake continuity from outward export surfaces",
            memory_posture:
                "explicitly no memory persistence implied during context uptake",
            boundary_guard:
                "separates context integration from memory commit semantics",
        },
    ];

pub const CANONICAL_BLUE_BRAIN_RUNTIME_FEEDBACK_MAP: [BlueBrainRuntimeFeedbackLane; 10] = [
    BlueBrainRuntimeFeedbackLane {
        class: BlueBrainRuntimeFeedbackClass::ComputeResultFeedback,
        lane: "blue_brain_feedback_result_integrated_current_runtime_state",
        canonical_source:
            "CanonicalComputeEntryPoint::submit -> result/fault/status + blue_brain_transition_compute_result_integrated",
        runtime_feedback_semantics:
            "result integrated into current runtime state with explicit reference continuity",
        transition_binding:
            "blue_brain_transition_compute_result_integrated",
        memory_boundary:
            "no memory persistence implied by result integration",
        non_canonical_boundary:
            "no direct adoption of compute-internal execution diagnostics",
    },
    BlueBrainRuntimeFeedbackLane {
        class: BlueBrainRuntimeFeedbackClass::ComputeResultFeedback,
        lane: "blue_brain_feedback_result_rejected_or_blocked",
        canonical_source:
            "submit result/fault/status + blue_brain_transition_compute_trigger_blocked_insufficient_context",
        runtime_feedback_semantics:
            "result rejected/blocked due to outward fault semantics; runtime records blocked posture",
        transition_binding:
            "blue_brain_transition_compute_trigger_blocked_insufficient_context",
        memory_boundary:
            "blocked result posture does not imply context persistence or memory write",
        non_canonical_boundary:
            "must not auto-unblock via expert/internal trigger path",
    },
    BlueBrainRuntimeFeedbackLane {
        class: BlueBrainRuntimeFeedbackClass::ComputeResultFeedback,
        lane: "blue_brain_feedback_result_integrated_with_caveat",
        canonical_source:
            "submit result/fault/status + status_evidence_export_surface(status/evidence refs)",
        runtime_feedback_semantics:
            "result integrated with caveat when status/evidence remains partial/caveated/insufficient",
        transition_binding:
            "blue_brain_transition_status_evidence_update_without_compute_trigger",
        memory_boundary:
            "caveated result integration updates runtime posture only; no memory commit",
        non_canonical_boundary:
            "no raw diagnostic blob required for caveat visibility",
    },
    BlueBrainRuntimeFeedbackLane {
        class: BlueBrainRuntimeFeedbackClass::StatusTrustFeedback,
        lane: "blue_brain_feedback_status_trust_current_to_insufficient",
        canonical_source:
            "CanonicalComputeEntryPoint::status + status_evidence_export_surface(status)",
        runtime_feedback_semantics:
            "runtime consumes outward status/trust signals: current|trusted, partial, stale, caveated, degraded, insufficient/blocked",
        transition_binding:
            "blue_brain_phase_caveated_degraded_partial_runtime_state + blue_brain_transition_status_evidence_update_without_compute_trigger",
        memory_boundary:
            "status/trust update is runtime state input, not persistence action",
        non_canonical_boundary:
            "expert/internal status lanes have no default Blue-Brain authority",
    },
    BlueBrainRuntimeFeedbackLane {
        class: BlueBrainRuntimeFeedbackClass::EvidenceReferenceFeedback,
        lane: "blue_brain_feedback_evidence_observed_and_attached",
        canonical_source:
            "CanonicalComputeEntryPoint::status_evidence_export_surface(evidence refs + replay refs)",
        runtime_feedback_semantics:
            "evidence observed and attached to current runtime context as outward references",
        transition_binding:
            "blue_brain_transition_evidence_observed_without_memory_commit",
        memory_boundary:
            "evidence attachment is reference-grade only; no automatic memory commit",
        non_canonical_boundary:
            "no audit/reasoning platform payload required",
    },
    BlueBrainRuntimeFeedbackLane {
        class: BlueBrainRuntimeFeedbackClass::EvidenceReferenceFeedback,
        lane: "blue_brain_feedback_evidence_caveated_partial_or_insufficient",
        canonical_source:
            "status_evidence_export_surface evidence posture + runtime_handoff_state_from_evidence",
        runtime_feedback_semantics:
            "runtime marks evidence as caveated/partial and can classify it as insufficient for stronger transition",
        transition_binding:
            "blue_brain_transition_status_evidence_update_without_compute_trigger",
        memory_boundary:
            "insufficient evidence does not escalate to memory-adjacent commit",
        non_canonical_boundary:
            "no internal trace object requirement for canonical evidence feedback",
    },
    BlueBrainRuntimeFeedbackLane {
        class: BlueBrainRuntimeFeedbackClass::DiagnosticCaveatFeedback,
        lane: "blue_brain_feedback_diagnostic_only_caveat",
        canonical_source:
            "status_evidence_export_surface caveat markers on outward diagnostics/status line",
        runtime_feedback_semantics:
            "diagnostic-only caveat is visible but non-blocking for current runtime continuity",
        transition_binding:
            "blue_brain_transition_status_evidence_update_without_compute_trigger",
        memory_boundary:
            "diagnostic caveat does not imply context or memory persistence",
        non_canonical_boundary:
            "do not expose compute-internal expert diagnostics as canonical payload",
    },
    BlueBrainRuntimeFeedbackLane {
        class: BlueBrainRuntimeFeedbackClass::DiagnosticCaveatFeedback,
        lane: "blue_brain_feedback_trigger_blocking_or_context_uptake_caveat",
        canonical_source:
            "blocked/insufficient outward status + blue_brain_transition_compute_trigger_blocked_insufficient_context",
        runtime_feedback_semantics:
            "runtime-relevant caveat can block trigger or limit context uptake until outward evidence/status improves",
        transition_binding:
            "blue_brain_transition_compute_trigger_blocked_insufficient_context + blue_brain_transition_status_evidence_update_without_compute_trigger",
        memory_boundary:
            "blocked context uptake remains transient and non-persistent",
        non_canonical_boundary:
            "no implicit override by expert/internal hooks",
    },
    BlueBrainRuntimeFeedbackLane {
        class: BlueBrainRuntimeFeedbackClass::ContextUptakeFeedback,
        lane: "blue_brain_feedback_context_uptake_transient_memory_adjacent_candidate",
        canonical_source:
            "blue_brain_transition_compute_result_integrated + blue_brain_transition_memory_adjacent_candidate_identified_not_committed",
        runtime_feedback_semantics:
            "separates observed evidence, context uptake, transient runtime context, and memory-adjacent candidate",
        transition_binding:
            "blue_brain_transition_compute_result_integrated + blue_brain_transition_memory_adjacent_candidate_identified_not_committed",
        memory_boundary:
            "actual memory persistence not implemented in BB2; candidate remains non-committed",
        non_canonical_boundary:
            "must not present context uptake as BB3 memory subsystem completion",
    },
    BlueBrainRuntimeFeedbackLane {
        class: BlueBrainRuntimeFeedbackClass::NonCanonicalInternalExpertFeedback,
        lane: "blue_brain_feedback_non_canonical_internal_expert_only",
        canonical_source:
            "service_surface::{run_operation_with_entry,replay_with_entry} + backends::build_backend(kind=stub|candle|worker) + legacy/compat surfaces",
        runtime_feedback_semantics:
            "internal/expert diagnostics may exist but are not canonical Blue-Brain runtime feedback",
        transition_binding:
            "blue_brain_transition_compute_trigger_suppressed_internal_only_path",
        memory_boundary:
            "internal diagnostics are not memory-adjacent authority and not persistence input",
        non_canonical_boundary:
            "must be down-mapped to outward status/evidence references before any Blue-Brain-facing usage",
    },
];

pub const CANONICAL_BLUE_BRAIN_CONTEXT_MEMORY_SURFACE_MAP: [BlueBrainContextMemorySurfaceLane; 13] =
    [
        BlueBrainContextMemorySurfaceLane {
            class: BlueBrainContextMemorySurfaceClass::TransientRuntimeContext,
            lane: "blue_brain_transient_runtime_context_handoff_state",
            source_surface:
                "runtime_orchestrator_stateful_loop + runtime_handoff_state_from_evidence/action_code",
            context_shape:
                "runtime-local context and handoff-state references tied to current execution window",
            evidence_or_reference_binding:
                "may consume outward status/evidence references without changing their reference-grade meaning",
            persistence_binding:
                "no durable commit; bounded in-process runtime state only",
            canonical_guard:
                "transient runtime context must not be relabeled as persisted memory",
        },
        BlueBrainContextMemorySurfaceLane {
            class: BlueBrainContextMemorySurfaceClass::TransientRuntimeContext,
            lane: "blue_brain_transient_runtime_context_available_for_transition",
            source_surface:
                "runtime_orchestrator_stateful_loop + blue_brain_transition_context_available",
            context_shape:
                "context slice available for current transition window before trigger decision",
            evidence_or_reference_binding:
                "uses already-observed references as runtime hints without changing evidence grade",
            persistence_binding:
                "transition window context is temporary and discarded when window closes",
            canonical_guard:
                "available-for-transition context must not imply memory persistence",
        },
        BlueBrainContextMemorySurfaceLane {
            class: BlueBrainContextMemorySurfaceClass::TransientRuntimeContext,
            lane: "blue_brain_transient_runtime_context_used_for_compute_trigger",
            source_surface:
                "blue_brain_transition_context_used_for_compute_trigger + blue_brain_transition_compute_trigger_from_context_availability",
            context_shape:
                "trigger-facing subset of transient context for deciding compute invocation eligibility",
            evidence_or_reference_binding:
                "trigger uses context/evidence posture references but keeps them as references only",
            persistence_binding:
                "trigger-time context use does not create durable memory state",
            canonical_guard:
                "compute trigger decisions must remain independent from memory commit semantics",
        },
        BlueBrainContextMemorySurfaceLane {
            class: BlueBrainContextMemorySurfaceClass::EvidenceBackedContext,
            lane: "blue_brain_evidence_backed_context_status_export",
            source_surface:
                "CanonicalComputeEntryPoint::status_evidence_export_surface + runtime_handoff_state_from_evidence",
            context_shape:
                "context posture informed by status/evidence quality (current|partial|stale|caveated|degraded|insufficient)",
            evidence_or_reference_binding:
                "bundle/trace/history references remain evidence-grade and are attached as context support",
            persistence_binding:
                "evidence-backed context updates runtime posture only; no automatic memory write",
            canonical_guard:
                "compute outputs and evidence feedback are not memory commits by default",
        },
        BlueBrainContextMemorySurfaceLane {
            class: BlueBrainContextMemorySurfaceClass::EvidenceBackedContext,
            lane: "blue_brain_evidence_backed_context_attached_or_caveated",
            source_surface:
                "blue_brain_feedback_evidence_observed_and_attached + blue_brain_feedback_evidence_caveated_partial_or_insufficient",
            context_shape:
                "evidence observed and attached to current context, with caveated/partial posture captured explicitly",
            evidence_or_reference_binding:
                "trace/history bundles stay reference-backed and can be marked partial/insufficient",
            persistence_binding:
                "partial or insufficient evidence cannot escalate into memory persistence or candidate commit",
            canonical_guard:
                "insufficient evidence remains context caveat, not memory authority",
        },
        BlueBrainContextMemorySurfaceLane {
            class: BlueBrainContextMemorySurfaceClass::ReplayReferenceBackedContext,
            lane: "blue_brain_replay_reference_backed_context",
            source_surface: "service_surface::{replay_preflight,replay_with_entry}",
            context_shape:
                "diagnostic/replay comparability context anchored on replay/reference metadata",
            evidence_or_reference_binding:
                "replay refs and context-bridge refs are reference inputs for interpretation only",
            persistence_binding:
                "no durable memory commit through replay/reference observation path",
            canonical_guard:
                "replay/reference context must stay distinct from memory persistence semantics",
        },
        BlueBrainContextMemorySurfaceLane {
            class: BlueBrainContextMemorySurfaceClass::ReplayReferenceBackedContext,
            lane: "blue_brain_replay_reference_backed_context_caveated_or_partial",
            source_surface:
                "service_surface::{ReplayRemoteContextReproducibility,ReplayContextConsistencyClass}",
            context_shape:
                "replay/reference context with explicit partial/missing fidelity and comparability caveats",
            evidence_or_reference_binding:
                "context bridge + remote context reproducibility stay interpretive references only",
            persistence_binding:
                "caveated replay/reference fidelity is never a persistence write path",
            canonical_guard:
                "partial replay/reference context cannot be promoted to memory without explicit future subsystem",
        },
        BlueBrainContextMemorySurfaceLane {
            class: BlueBrainContextMemorySurfaceClass::MemoryAdjacentCandidate,
            lane: "blue_brain_memory_adjacent_candidate_not_committed",
            source_surface:
                "blue_brain_transition_memory_adjacent_candidate_identified_not_committed + runtime feedback context uptake",
            context_shape:
                "candidate extracted from context/evidence linkage for future memory integration decisions",
            evidence_or_reference_binding:
                "candidate derivation is evidence/reference-backed and remains auditable",
            persistence_binding:
                "candidate only; explicitly not persisted and not committed in BB2/BB3 prompt-1 surface",
            canonical_guard:
                "memory-adjacent candidate must not be exposed as actual memory persistence",
        },
        BlueBrainContextMemorySurfaceLane {
            class: BlueBrainContextMemorySurfaceClass::MemoryAdjacentCandidate,
            lane: "blue_brain_memory_adjacent_candidate_derived_sources_uncommitted",
            source_surface:
                "context/evidence/result linkage across transition + feedback maps (without commit lane)",
            context_shape:
                "candidate may be derived from context window, compute result uptake, or evidence/reference continuity",
            evidence_or_reference_binding:
                "derivation basis remains inspectable via transition/evidence/replay references",
            persistence_binding:
                "derived candidate remains non-committed and requires future explicit memory policy/subsystem",
            canonical_guard:
                "candidate derivation source richness must not be mistaken for persisted memory",
        },
        BlueBrainContextMemorySurfaceLane {
            class: BlueBrainContextMemorySurfaceClass::PersistedMemory,
            lane: "blue_brain_persisted_memory_none_in_current_baseline",
            source_surface: "none (no canonical Blue-Brain memory persistence lane in current repo baseline)",
            context_shape:
                "persisted-memory contract intentionally absent from canonical Blue-Brain runtime surfaces",
            evidence_or_reference_binding:
                "evidence/replay/history references can support future persistence decisions but are not persistence by themselves",
            persistence_binding:
                "actual persisted memory lane not implemented",
            canonical_guard:
                "explicit null lane prevents accidental reinterpretation of history/evidence as memory store",
        },
        BlueBrainContextMemorySurfaceLane {
            class: BlueBrainContextMemorySurfaceClass::NonCanonicalInternalOnlyMemoryLikePath,
            lane: "blue_brain_internal_expert_memory_like_path_non_canonical",
            source_surface:
                "service_surface::{run_operation_with_entry,replay_with_entry} + backends::build_backend(kind=stub|candle|worker) + domains/ai*",
            context_shape:
                "internal/expert diagnostics or compatibility context that can look memory-like but is non-canonical for Blue-Brain",
            evidence_or_reference_binding:
                "must be down-mapped to outward status/evidence references before Blue-Brain-facing use",
            persistence_binding:
                "not a canonical persistence authority",
            canonical_guard:
                "internal/expert/compat paths are excluded from canonical context-memory surface authority",
        },
        BlueBrainContextMemorySurfaceLane {
            class: BlueBrainContextMemorySurfaceClass::TransientRuntimeContext,
            lane: "blue_brain_compute_result_context_uptake_non_memory",
            source_surface:
                "blue_brain_transition_compute_result_integrated + blue_brain_feedback_result_integrated_current_runtime_state",
            context_shape:
                "compute-result uptake updates current runtime context for subsequent state transitions",
            evidence_or_reference_binding:
                "result/evidence continuity tracked by outward references and transition bindings",
            persistence_binding:
                "uptake is transient runtime mutation, not durable memory persistence",
            canonical_guard:
                "compute result integration must remain separate from memory persistence claims",
        },
        BlueBrainContextMemorySurfaceLane {
            class: BlueBrainContextMemorySurfaceClass::TransientRuntimeContext,
            lane: "blue_brain_transient_runtime_context_updated_then_discarded",
            source_surface:
                "blue_brain_transition_compute_result_integrated + blue_brain_transition_status_evidence_update_without_compute_trigger",
            context_shape:
                "runtime context can be updated by result/evidence feedback and later discarded from active window",
            evidence_or_reference_binding:
                "updates retain outward evidence linkage while discard keeps no durable memory side effect",
            persistence_binding:
                "discard path keeps no persisted memory and no implicit long-term state write",
            canonical_guard:
                "runtime context lifecycle (available/use/update/discard) must stay non-memory by default",
        },
    ];

pub const CANONICAL_BLUE_BRAIN_CONTEXT_UPDATE_LIFECYCLE_MAP:
    [BlueBrainContextUpdateLifecycleLane; 9] = [
    BlueBrainContextUpdateLifecycleLane {
        class: BlueBrainContextUpdateLifecycleClass::ContextInitialized,
        lane: "blue_brain_context_initialized_for_runtime_window",
        source_surface: "runtime_orchestrator_stateful_loop + runtime_handoff_state_from_evidence/action_code",
        update_semantics: "context initialized for active runtime transition window",
        candidate_effect: "update only; no candidate proposal implied",
        persistence_semantics: "initialization is transient runtime state; no persistence performed",
        canonical_guard: "initial runtime context must stay distinct from memory lifecycle",
    },
    BlueBrainContextUpdateLifecycleLane {
        class: BlueBrainContextUpdateLifecycleClass::UpdatedFromComputeResult,
        lane: "blue_brain_context_updated_from_compute_result",
        source_surface:
            "blue_brain_transition_compute_result_integrated + blue_brain_feedback_result_integrated_current_runtime_state",
        update_semantics: "context updated from compute result uptake on canonical result/fault/status line",
        candidate_effect: "result integrated but no candidate required by default",
        persistence_semantics: "compute-result context uptake is non-memory and non-persistent",
        canonical_guard: "result integration must not be interpreted as memory commit",
    },
    BlueBrainContextUpdateLifecycleLane {
        class: BlueBrainContextUpdateLifecycleClass::UpdatedFromComputeResult,
        lane: "blue_brain_context_updated_and_candidate_proposed",
        source_surface:
            "blue_brain_transition_compute_result_integrated + blue_brain_transition_memory_adjacent_candidate_identified_not_committed",
        update_semantics: "context update can be followed by explicit candidate proposal under bounded BB3 semantics",
        candidate_effect: "update plus candidate proposal (explicit and separate events)",
        persistence_semantics:
            "candidate proposal remains non-persistent and does not imply commit",
        canonical_guard: "context update and candidate lifecycle are linked but not collapsed",
    },
    BlueBrainContextUpdateLifecycleLane {
        class: BlueBrainContextUpdateLifecycleClass::UpdatedFromEvidenceReference,
        lane: "blue_brain_context_updated_from_evidence_reference",
        source_surface:
            "blue_brain_feedback_evidence_observed_and_attached + status_evidence_export_surface",
        update_semantics: "context updated from outward evidence references with posture retained",
        candidate_effect: "evidence attachment may support later candidate formation but does not require it",
        persistence_semantics: "evidence/reference update has no memory write path",
        canonical_guard: "evidence-backed context update must remain reference-grade",
    },
    BlueBrainContextUpdateLifecycleLane {
        class: BlueBrainContextUpdateLifecycleClass::UpdatedFromReplayReference,
        lane: "blue_brain_context_updated_from_replay_reference_basis",
        source_surface: "service_surface::{replay_preflight,replay_with_entry}",
        update_semantics: "context updated from replay/reference basis when comparability context is available",
        candidate_effect: "replay/reference context can support candidate basis without automatic proposal",
        persistence_semantics: "replay/reference update does not persist memory",
        canonical_guard: "replay context is interpretive support and not a memory store",
    },
    BlueBrainContextUpdateLifecycleLane {
        class: BlueBrainContextUpdateLifecycleClass::ContextUnchanged,
        lane: "blue_brain_context_unchanged_after_transition_check",
        source_surface:
            "blue_brain_transition_context_available + blue_brain_transition_status_evidence_update_without_compute_trigger",
        update_semantics: "context remains unchanged when transition checks yield no safe mutation",
        candidate_effect: "no candidate created by unchanged context path",
        persistence_semantics: "unchanged path performs no persistence",
        canonical_guard: "no-op/unchanged transition outcomes must stay explicit",
    },
    BlueBrainContextUpdateLifecycleLane {
        class: BlueBrainContextUpdateLifecycleClass::UpdateBlockedOrInsufficient,
        lane: "blue_brain_context_update_blocked_insufficient_evidence",
        source_surface:
            "blue_brain_transition_compute_trigger_blocked_insufficient_context + blue_brain_feedback_evidence_caveated_partial_or_insufficient",
        update_semantics: "context update blocked or caveated due to insufficient/partial/stale evidence posture",
        candidate_effect: "blocked update does not silently mint candidate",
        persistence_semantics: "blocked/insufficient state has no persistence side effect",
        canonical_guard: "blocked or insufficient context must not be reinterpreted as implicit memory action",
    },
    BlueBrainContextUpdateLifecycleLane {
        class: BlueBrainContextUpdateLifecycleClass::ContextUnchanged,
        lane: "blue_brain_candidate_rejected_context_preserved",
        source_surface:
            "blue_brain_transition_memory_adjacent_candidate_identified_not_committed + status_evidence_export_surface(status)",
        update_semantics: "candidate may be rejected while existing context is preserved",
        candidate_effect: "rejected candidate with context preserved and no forced mutation",
        persistence_semantics: "rejection has no persistence write",
        canonical_guard: "candidate rejection must not rewrite context history implicitly",
    },
    BlueBrainContextUpdateLifecycleLane {
        class: BlueBrainContextUpdateLifecycleClass::ContextUnchanged,
        lane: "blue_brain_candidate_only_without_context_mutation",
        source_surface:
            "blue_brain_memory_adjacent_candidate_derived_sources_uncommitted + replay/status references",
        update_semantics: "candidate can be proposed from references without mutating current runtime context",
        candidate_effect: "candidate without context mutation is explicitly representable",
        persistence_semantics: "proposal-only path remains uncommitted and non-persistent",
        canonical_guard: "candidate-only paths cannot imply hidden context writes",
    },
];

pub const CANONICAL_BLUE_BRAIN_MEMORY_CANDIDATE_LIFECYCLE_MAP:
    [BlueBrainMemoryCandidateLifecycleLane; 13] = [
    BlueBrainMemoryCandidateLifecycleLane {
        class: BlueBrainMemoryCandidateLifecycleClass::CandidateProposed,
        lane: "blue_brain_candidate_proposed",
        source_surface:
            "blue_brain_transition_memory_adjacent_candidate_identified_not_committed",
        candidate_semantics: "candidate proposed explicitly for future memory handling",
        context_mutation_semantics: "proposal may follow update or exist without context mutation",
        persistence_semantics: "proposal does not perform persistence",
        canonical_guard: "proposed candidate is not a committed memory object",
    },
    BlueBrainMemoryCandidateLifecycleLane {
        class: BlueBrainMemoryCandidateLifecycleClass::CandidateEvidenceBacked,
        lane: "blue_brain_candidate_evidence_backed_reference",
        source_surface:
            "blue_brain_feedback_evidence_observed_and_attached + status_evidence_export_surface(evidence refs)",
        candidate_semantics:
            "candidate backed by evidence reference with explicit reference-grade provenance",
        context_mutation_semantics: "evidence backing can occur with or without further context change",
        persistence_semantics: "evidence-backed candidate remains non-persistent",
        canonical_guard: "evidence support must not be relabeled as persistence",
    },
    BlueBrainMemoryCandidateLifecycleLane {
        class: BlueBrainMemoryCandidateLifecycleClass::CandidateContextDerived,
        lane: "blue_brain_candidate_context_derived",
        source_surface:
            "blue_brain_transient_runtime_context_updated_then_discarded + handoff state",
        candidate_semantics: "candidate derived from bounded runtime context transitions",
        context_mutation_semantics: "derived candidate remains separate from any future context mutation",
        persistence_semantics: "context-derived candidate is not persisted",
        canonical_guard: "runtime-context derivation does not equal memory commit",
    },
    BlueBrainMemoryCandidateLifecycleLane {
        class: BlueBrainMemoryCandidateLifecycleClass::CandidateComputeResultDerived,
        lane: "blue_brain_candidate_compute_result_derived_proposed",
        source_surface:
            "blue_brain_transition_compute_result_integrated + result/fault/status continuity",
        candidate_semantics:
            "result-derived candidate may be proposed only when bounded compute semantics support it",
        context_mutation_semantics: "compute result may update context without forcing candidate",
        persistence_semantics: "result-derived candidate is explicitly non-persistent",
        canonical_guard: "inference/compute result must not auto-persist into memory",
    },
    BlueBrainMemoryCandidateLifecycleLane {
        class: BlueBrainMemoryCandidateLifecycleClass::AcceptedForFutureMemoryHandling,
        lane: "blue_brain_candidate_accepted_for_future_memory_handling",
        source_surface:
            "memory_adjacent_candidate lane + status/evidence/replay references",
        candidate_semantics:
            "candidate accepted for future memory handling queueing without current commit",
        context_mutation_semantics:
            "acceptance is candidate-state change and does not require additional context mutation",
        persistence_semantics: "accepted-for-future-handling still performs no persistence",
        canonical_guard: "accepted state must stay distinct from persisted memory",
    },
    BlueBrainMemoryCandidateLifecycleLane {
        class: BlueBrainMemoryCandidateLifecycleClass::CandidateRejected,
        lane: "blue_brain_candidate_rejected_due_to_fault_or_caveat",
        source_surface:
            "blue_brain_feedback_result_fault_or_caveated + transition/status caveat bindings",
        candidate_semantics: "result-derived or reference-derived candidate rejected due to fault/caveat",
        context_mutation_semantics: "context may remain preserved when candidate rejected",
        persistence_semantics: "rejected candidate never persists",
        canonical_guard: "rejection outcome must remain explicit and deterministic",
    },
    BlueBrainMemoryCandidateLifecycleLane {
        class: BlueBrainMemoryCandidateLifecycleClass::CandidateStale,
        lane: "blue_brain_candidate_stale_reference_basis",
        source_surface:
            "replay/status references with stale posture (current|partial|stale classes)",
        candidate_semantics: "candidate marked stale when reference basis ages out",
        context_mutation_semantics: "stale candidate marking does not require context rewrite",
        persistence_semantics: "stale marker has no persistence effect",
        canonical_guard: "stale references must be visible as caveated candidate basis",
    },
    BlueBrainMemoryCandidateLifecycleLane {
        class: BlueBrainMemoryCandidateLifecycleClass::CandidateInsufficient,
        lane: "blue_brain_candidate_insufficient_reference_basis",
        source_surface:
            "blue_brain_feedback_evidence_caveated_partial_or_insufficient + replay partial basis",
        candidate_semantics: "candidate marked insufficient when evidence/reference basis is weak",
        context_mutation_semantics: "insufficient candidate can coexist with unchanged context",
        persistence_semantics: "insufficient state does not persist",
        canonical_guard: "insufficient candidate must not be promoted implicitly",
    },
    BlueBrainMemoryCandidateLifecycleLane {
        class: BlueBrainMemoryCandidateLifecycleClass::PersistenceUnavailableOrDeferred,
        lane: "blue_brain_candidate_persistence_unavailable_or_deferred",
        source_surface:
            "blue_brain_persisted_memory_none_in_current_baseline + candidate acceptance/rejection lanes",
        candidate_semantics:
            "candidate outcome explicitly records persistence unavailable/deferred in current baseline",
        context_mutation_semantics:
            "deferred persistence state does not require context mutation and can coexist with update-only flows",
        persistence_semantics:
            "persistence unavailable/deferred marker is explicit; no hidden commit path exists",
        canonical_guard:
            "deferred marker prevents implicit persistence assumptions for accepted candidates",
    },
    BlueBrainMemoryCandidateLifecycleLane {
        class: BlueBrainMemoryCandidateLifecycleClass::PersistencePerformedViaRealPathOnly,
        lane: "blue_brain_candidate_persistence_performed_only_if_real_path_exists",
        source_surface:
            "future memory subsystem attachment contract (not implemented in current baseline)",
        candidate_semantics:
            "candidate may transition to persisted-memory-performed only through a real explicit persistence path",
        context_mutation_semantics:
            "context and candidate states stay separately observable even if a future real path is added",
        persistence_semantics:
            "current baseline exposes perform-only-if-real-path rule and does not provide such a path",
        canonical_guard:
            "forbids auto-persist behavior and blocks synthetic commit claims via history/evidence/replay",
    },
    BlueBrainMemoryCandidateLifecycleLane {
        class: BlueBrainMemoryCandidateLifecycleClass::NoPersistencePerformed,
        lane: "blue_brain_candidate_no_persistence_performed",
        source_surface:
            "blue_brain_persisted_memory_none_in_current_baseline + candidate lifecycle lanes",
        candidate_semantics: "all candidate states end with explicit no-persistence marker in BB3 prompt-2",
        context_mutation_semantics: "context/candidate outcomes are observable without commit side effects",
        persistence_semantics: "no persistence performed; actual memory commit intentionally deferred",
        canonical_guard: "null persisted-memory lane remains authoritative",
    },
    BlueBrainMemoryCandidateLifecycleLane {
        class: BlueBrainMemoryCandidateLifecycleClass::CandidateEvidenceBacked,
        lane: "blue_brain_candidate_backed_by_replay_reference_context",
        source_surface:
            "blue_brain_replay_reference_backed_context + replay_preflight/replay_with_entry",
        candidate_semantics:
            "candidate backed by replay/reference context without claiming memory storage",
        context_mutation_semantics: "replay-backed proposal may be candidate-only",
        persistence_semantics: "replay-backed candidate is non-persistent",
        canonical_guard: "replay reference support remains interpretive and bounded",
    },
    BlueBrainMemoryCandidateLifecycleLane {
        class: BlueBrainMemoryCandidateLifecycleClass::CandidateComputeResultDerived,
        lane: "blue_brain_candidate_compute_result_derived_rejected_or_not_persisted",
        source_surface:
            "compute result/fault status + memory_adjacent candidate decision boundary",
        candidate_semantics:
            "result-derived candidate can be rejected on fault/caveat or kept as not-persisted proposal",
        context_mutation_semantics:
            "result path may update context even when candidate is rejected",
        persistence_semantics: "result-derived candidate path never auto-commits memory",
        canonical_guard: "compute-result candidate formation is gated and explicitly non-persistent",
    },
];

pub const CANONICAL_BLUE_BRAIN_MEMORY_COMMIT_BOUNDARY_MAP: [BlueBrainMemoryCommitBoundaryLane;
    11] = [
    BlueBrainMemoryCommitBoundaryLane {
        class: BlueBrainMemoryCommitBoundaryClass::NotMemoryCandidate,
        lane: "blue_brain_memory_commit_boundary_not_a_memory_candidate",
        source_binding:
            "selection ignored/irrelevant + non-memory runtime transitions + non-memory references",
        eligibility_semantics: "item is not a memory candidate and cannot become commit-eligible",
        persistence_path_semantics: "no memory persistence semantics apply",
        canonical_guard: "not-a-candidate must remain distinct from candidate lifecycle states",
    },
    BlueBrainMemoryCommitBoundaryLane {
        class: BlueBrainMemoryCommitBoundaryClass::MemoryCandidateProposed,
        lane: "blue_brain_memory_commit_boundary_candidate_proposed",
        source_binding:
            "blue_brain_candidate_proposed + candidate context/evidence/replay references",
        eligibility_semantics:
            "proposal state only; commit-eligibility requires additional selection/evidence/context checks",
        persistence_path_semantics: "proposal does not commit and does not imply persistence path",
        canonical_guard: "proposal state must not be relabeled as commit-ready automatically",
    },
    BlueBrainMemoryCommitBoundaryLane {
        class: BlueBrainMemoryCommitBoundaryClass::MemoryCandidateDeferred,
        lane: "blue_brain_memory_commit_boundary_candidate_deferred",
        source_binding:
            "candidate deferred + deferred pending stronger evidence/context update",
        eligibility_semantics:
            "deferred candidate is explicitly not commit-eligible until rechecked and selected",
        persistence_path_semantics: "deferred path remains non-commit and non-persistent",
        canonical_guard: "deferred must stay distinct from rejected and commit-eligible states",
    },
    BlueBrainMemoryCommitBoundaryLane {
        class: BlueBrainMemoryCommitBoundaryClass::MemoryCandidateRejected,
        lane: "blue_brain_memory_commit_boundary_candidate_rejected",
        source_binding:
            "candidate rejected due to fault/caveat or incompatible runtime outcome",
        eligibility_semantics: "rejected candidate is not commit-eligible",
        persistence_path_semantics: "rejected candidate never commits",
        canonical_guard: "rejected outcome is terminal for current candidate instance",
    },
    BlueBrainMemoryCommitBoundaryLane {
        class: BlueBrainMemoryCommitBoundaryClass::MemoryCandidateStale,
        lane: "blue_brain_memory_commit_boundary_candidate_stale",
        source_binding: "stale reference quality posture + candidate stale class",
        eligibility_semantics: "stale candidate is commit-blocked until refreshed context/reference basis exists",
        persistence_path_semantics: "stale state has no commit authority",
        canonical_guard: "stale evidence/reference basis must stay explicit and non-promotable",
    },
    BlueBrainMemoryCommitBoundaryLane {
        class: BlueBrainMemoryCommitBoundaryClass::MemoryCandidateInsufficient,
        lane: "blue_brain_memory_commit_boundary_candidate_insufficient",
        source_binding: "insufficient evidence/reference posture + insufficient candidate class",
        eligibility_semantics: "insufficient candidate is not commit-eligible",
        persistence_path_semantics: "insufficient state performs no persistence",
        canonical_guard: "insufficient quality cannot be escalated to eligible without new basis",
    },
    BlueBrainMemoryCommitBoundaryLane {
        class: BlueBrainMemoryCommitBoundaryClass::CommitEligibleCandidate,
        lane: "blue_brain_memory_commit_boundary_candidate_commit_eligible",
        source_binding:
            "candidate selected/accepted + sufficient evidence/reference + non-stale context + no blocking caveat",
        eligibility_semantics:
            "candidate is commit-eligible only when canonical minimal conditions are met",
        persistence_path_semantics:
            "eligible state may hand off to future-memory-ready and may commit only if real path exists",
        canonical_guard: "commit-eligible is a boundary class, not a guaranteed commit result",
    },
    BlueBrainMemoryCommitBoundaryLane {
        class: BlueBrainMemoryCommitBoundaryClass::FutureMemoryReadyCandidate,
        lane: "blue_brain_memory_commit_boundary_candidate_future_memory_ready",
        source_binding:
            "accepted candidate + explicit persistence unavailable/deferred marker + handoff envelope",
        eligibility_semantics:
            "future-memory-ready candidate preserves commit eligibility posture without asserting actual commit",
        persistence_path_semantics:
            "handoff-only in current baseline because no real persisted-memory path is implemented",
        canonical_guard: "future-ready must remain non-commit until explicit real path exists",
    },
    BlueBrainMemoryCommitBoundaryLane {
        class: BlueBrainMemoryCommitBoundaryClass::CommittedMemoryIfRealPath,
        lane: "blue_brain_memory_commit_boundary_committed_memory_only_if_real_path_exists",
        source_binding:
            "future memory subsystem contract id + commit result envelope (not implemented in current baseline)",
        eligibility_semantics:
            "commit result class is reachable only through explicit real persistence contract",
        persistence_path_semantics:
            "current repository baseline has no canonical Blue-Brain actual memory commit path",
        canonical_guard:
            "history/snapshot/evidence/replay/internal hooks cannot synthesize committed-memory claims",
    },
    BlueBrainMemoryCommitBoundaryLane {
        class: BlueBrainMemoryCommitBoundaryClass::ReferenceOnlyNotMemory,
        lane: "blue_brain_memory_commit_boundary_reference_only_not_memory",
        source_binding:
            "job history + snapshot + evidence refs + replay/trace references + status export",
        eligibility_semantics:
            "reference-only continuity may support candidate evaluation but is not itself a memory candidate",
        persistence_path_semantics: "reference continuity is not memory persistence",
        canonical_guard: "history/snapshot/evidence/replay/trace must stay not-memory by default",
    },
    BlueBrainMemoryCommitBoundaryLane {
        class: BlueBrainMemoryCommitBoundaryClass::NonCanonicalInternalOnlyPersistencePath,
        lane: "blue_brain_memory_commit_boundary_non_canonical_internal_persistence_path",
        source_binding:
            "run_operation_with_entry/replay_with_entry expert lanes + legacy/compat/internal diagnostics",
        eligibility_semantics:
            "internal/expert-only paths are excluded from canonical commit-eligibility decisions",
        persistence_path_semantics:
            "non-canonical paths cannot act as commit authority or canonical persistence path",
        canonical_guard: "must remap to outward candidate/evidence/selection references before any future use",
    },
];

pub const CANONICAL_BLUE_BRAIN_COMMIT_ELIGIBILITY_CONDITIONS_MAP:
    [BlueBrainCommitEligibilityConditionLane; 7] = [
    BlueBrainCommitEligibilityConditionLane {
        class: BlueBrainCommitEligibilityConditionClass::EvidenceReferenceBasis,
        lane: "blue_brain_commit_eligibility_condition_evidence_reference_quality",
        requirement: "sufficient evidence/reference basis required; partial/caveated allowed only as caveated eligibility",
        when_satisfied: "candidate can advance toward commit-eligible evaluation",
        when_not_satisfied: "candidate remains insufficient, stale, or deferred and not commit-eligible",
        canonical_guard: "reference-only basis must preserve explicit quality posture and caveats",
    },
    BlueBrainCommitEligibilityConditionLane {
        class: BlueBrainCommitEligibilityConditionClass::SelectionAttentionGate,
        lane: "blue_brain_commit_eligibility_condition_selection_attention_gate",
        requirement: "candidate must be selected or explicitly accepted; deferred/ignored/rejected are ineligible",
        when_satisfied: "selected candidate can become commit-eligible if other gates pass",
        when_not_satisfied: "candidate remains deferred/rejected/not-memory and cannot be commit-eligible",
        canonical_guard: "selection/attention informs eligibility but is not a planning or policy engine",
    },
    BlueBrainCommitEligibilityConditionLane {
        class: BlueBrainCommitEligibilityConditionClass::ContextFreshnessGate,
        lane: "blue_brain_commit_eligibility_condition_non_stale_context",
        requirement: "candidate context basis must be non-stale and non-expired",
        when_satisfied: "context gate permits commit-eligibility progression",
        when_not_satisfied: "stale context blocks commit-eligibility until refresh/recheck",
        canonical_guard: "stale context cannot be bypassed through history or replay traces",
    },
    BlueBrainCommitEligibilityConditionLane {
        class: BlueBrainCommitEligibilityConditionClass::BlockingCaveatGate,
        lane: "blue_brain_commit_eligibility_condition_no_blocking_caveat",
        requirement: "no blocking caveat/fault posture may be active on candidate basis",
        when_satisfied: "candidate remains eligible candidate or future-memory-ready",
        when_not_satisfied: "candidate is blocked or rejected and cannot commit",
        canonical_guard: "caveated partial basis must remain explicit and cannot silently upgrade to committed",
    },
    BlueBrainCommitEligibilityConditionLane {
        class: BlueBrainCommitEligibilityConditionClass::CanonicalDependencyGate,
        lane: "blue_brain_commit_eligibility_condition_no_internal_only_dependency",
        requirement: "eligibility must not depend on internal/expert-only/compat persistence-like hooks",
        when_satisfied: "candidate stays on canonical outward references",
        when_not_satisfied: "path is classified non-canonical and commit-ineligible",
        canonical_guard: "compute-core internal details are excluded from canonical memory authority",
    },
    BlueBrainCommitEligibilityConditionLane {
        class: BlueBrainCommitEligibilityConditionClass::PersistencePathGate,
        lane: "blue_brain_commit_eligibility_condition_explicit_persistence_path_or_none",
        requirement:
            "actual commit requires an explicit real persisted-memory path implemented in repository",
        when_satisfied: "commit may occur only through that explicit canonical path",
        when_not_satisfied: "no actual commit occurs and candidate remains future-memory-ready/deferred",
        canonical_guard: "absence of real path is authoritative and blocks synthetic commit claims",
    },
    BlueBrainCommitEligibilityConditionLane {
        class: BlueBrainCommitEligibilityConditionClass::FutureMemoryReadyHandoffGate,
        lane: "blue_brain_commit_eligibility_condition_future_memory_handoff",
        requirement:
            "if no real persistence path exists, preserve candidate in explicit future-memory-ready handoff envelope",
        when_satisfied: "handoff remains explicit with candidate/evidence/reference/caveat bindings",
        when_not_satisfied:
            "candidate must remain proposal/deferred/rejected/insufficient and not claim commit progression",
        canonical_guard: "future-memory-ready is a no-commit handoff class in current baseline",
    },
];

pub const CANONICAL_BLUE_BRAIN_PERSISTENCE_BOUNDARY_MAP: [BlueBrainPersistenceBoundaryLane; 9] = [
    BlueBrainPersistenceBoundaryLane {
        class: BlueBrainPersistenceBoundaryClass::TransientRuntimeContext,
        lane: "blue_brain_persistence_boundary_transient_runtime_context",
        source_surface:
            "runtime_orchestrator_stateful_loop + runtime_handoff_state_from_evidence/action_code",
        boundary_semantics:
            "runtime context is transient, bounded to active execution windows, and discarded without durable memory commit",
        future_attachment_semantics:
            "future memory subsystem may inspect candidate handoff fields but cannot treat transient context as persisted memory",
        canonical_guard:
            "transient context must not be reclassified as memory persistence or memory record history",
    },
    BlueBrainPersistenceBoundaryLane {
        class: BlueBrainPersistenceBoundaryClass::EvidenceReferenceBackedContext,
        lane: "blue_brain_persistence_boundary_evidence_reference_context",
        source_surface:
            "status_evidence_export_surface + blue_brain_reference_context_* lanes",
        boundary_semantics:
            "evidence/reference-backed context is reference-grade support for runtime posture, not persistence",
        future_attachment_semantics:
            "attachment requires explicit evidence/reference fields and quality caveats before candidate handoff",
        canonical_guard:
            "evidence/replay/snapshot/trace references cannot be promoted to memory commit by observation alone",
    },
    BlueBrainPersistenceBoundaryLane {
        class: BlueBrainPersistenceBoundaryClass::MemoryAdjacentCandidate,
        lane: "blue_brain_persistence_boundary_memory_adjacent_candidate",
        source_surface:
            "blue_brain_transition_memory_adjacent_candidate_identified_not_committed + candidate lifecycle lanes",
        boundary_semantics:
            "memory-adjacent candidate is proposal-grade only and remains non-persistent in current baseline",
        future_attachment_semantics:
            "future subsystem handoff is allowed as proposal with explicit non-commit boundary",
        canonical_guard:
            "candidate identification never performs implicit commit",
    },
    BlueBrainPersistenceBoundaryLane {
        class: BlueBrainPersistenceBoundaryClass::FutureMemoryReadyCandidate,
        lane: "blue_brain_persistence_boundary_future_memory_ready_candidate",
        source_surface:
            "blue_brain_candidate_accepted_for_future_memory_handling + blue_brain_candidate_persistence_unavailable_or_deferred",
        boundary_semantics:
            "future-memory-ready candidate remains explicitly non-committed until a real persistence path exists",
        future_attachment_semantics:
            "required fields include candidate id, context/evidence/replay references, and caveat posture",
        canonical_guard:
            "future-ready acceptance is not a persistence acknowledgment",
    },
    BlueBrainPersistenceBoundaryLane {
        class: BlueBrainPersistenceBoundaryClass::ActualPersistedMemory,
        lane: "blue_brain_persistence_boundary_actual_persisted_memory_deferred",
        source_surface: "none for Blue-Brain memory in current repository baseline",
        boundary_semantics:
            "actual persisted memory for Blue-Brain context/candidates is intentionally deferred and not implemented",
        future_attachment_semantics:
            "future subsystem must introduce explicit persisted-memory contract before any commit state can exist",
        canonical_guard:
            "absence of real path is canonical and blocks synthetic persisted-memory claims",
    },
    BlueBrainPersistenceBoundaryLane {
        class: BlueBrainPersistenceBoundaryClass::HistorySnapshotReferenceButNotMemory,
        lane: "blue_brain_persistence_boundary_history_snapshot_reference_not_memory",
        source_surface:
            "service_surface job_history + replay_preflight/replay_with_entry + status evidence refs",
        boundary_semantics:
            "history/snapshot/replay/reference persistence remains diagnostic/evidence continuity, not memory persistence",
        future_attachment_semantics:
            "these references may support candidate evaluation with caveats but cannot substitute memory storage",
        canonical_guard:
            "history or snapshot presence must never be labeled as persisted memory",
    },
    BlueBrainPersistenceBoundaryLane {
        class: BlueBrainPersistenceBoundaryClass::HistorySnapshotReferenceButNotMemory,
        lane: "blue_brain_persistence_boundary_compute_status_evidence_not_memory",
        source_surface:
            "CanonicalComputeEntryPoint::status + status_evidence_export_surface + execution_snapshot",
        boundary_semantics:
            "status/evidence/export surfaces can persist operational records while remaining non-memory for BB3 semantics",
        future_attachment_semantics:
            "future attachment can read outward references but must keep not-memory caveat",
        canonical_guard:
            "compute status/evidence persistence is not a Blue-Brain memory subsystem",
    },
    BlueBrainPersistenceBoundaryLane {
        class: BlueBrainPersistenceBoundaryClass::NonCanonicalInternalOnlyPersistenceLikePath,
        lane: "blue_brain_persistence_boundary_internal_expert_persistence_like_path",
        source_surface:
            "service_surface::{run_operation_with_entry,replay_with_entry} + backends::build_backend(kind=stub|candle|worker)",
        boundary_semantics:
            "internal/expert persistence-like diagnostics are non-canonical for Blue-Brain memory boundaries",
        future_attachment_semantics:
            "not eligible as direct future memory attachment until remapped through outward canonical references",
        canonical_guard:
            "internal-only/expert-only paths are excluded from canonical memory authority",
    },
    BlueBrainPersistenceBoundaryLane {
        class: BlueBrainPersistenceBoundaryClass::NonCanonicalInternalOnlyPersistenceLikePath,
        lane: "blue_brain_persistence_boundary_noncanonical_domains_ai_paths",
        source_surface: "domains/ai* compatibility and legacy persistence-adjacent traces",
        boundary_semantics:
            "legacy/compat domains can expose persistence-like traces but are not canonical Blue-Brain memory surfaces",
        future_attachment_semantics:
            "future subsystem must consume outward-facing evidence/reference contracts instead of direct internal traces",
        canonical_guard:
            "prevents compute-core internal hooks from becoming implicit memory connectors",
    },
];

pub const CANONICAL_BLUE_BRAIN_FUTURE_MEMORY_ATTACHMENT_MAP:
    [BlueBrainFutureMemoryAttachmentLane; 7] = [
    BlueBrainFutureMemoryAttachmentLane {
        class: BlueBrainFutureMemoryAttachmentClass::CandidateHandoffProposalOnly,
        lane: "blue_brain_future_memory_attachment_candidate_proposed_only",
        trigger_or_source:
            "blue_brain_candidate_proposed + blue_brain_candidate_evidence_backed_reference",
        required_fields:
            "candidate_id + candidate digest + context digest + evidence/reference basis + reference quality posture",
        caveats:
            "proposal may be partial/stale/caveated and must keep those caveats explicit",
        commit_boundary:
            "proposal lane is handoff-only and does not commit memory",
    },
    BlueBrainFutureMemoryAttachmentLane {
        class: BlueBrainFutureMemoryAttachmentClass::CandidateFutureReadyNoCommit,
        lane: "blue_brain_future_memory_attachment_candidate_future_ready_no_commit",
        trigger_or_source: "blue_brain_candidate_accepted_for_future_memory_handling",
        required_fields:
            "accepted candidate state + explicit future-memory-ready marker + evidence/replay references",
        caveats:
            "accepted-for-future state remains non-persistent until explicit memory subsystem exists",
        commit_boundary:
            "future-ready handoff cannot commit in current baseline",
    },
    BlueBrainFutureMemoryAttachmentLane {
        class: BlueBrainFutureMemoryAttachmentClass::CandidateRejectedOrInsufficient,
        lane: "blue_brain_future_memory_attachment_candidate_rejected_or_insufficient",
        trigger_or_source:
            "blue_brain_candidate_rejected_due_to_fault_or_caveat + blue_brain_candidate_insufficient_reference_basis + blue_brain_candidate_stale_reference_basis",
        required_fields:
            "candidate state + rejection/insufficient/stale reason + supporting reference posture",
        caveats:
            "rejected/insufficient/stale states are terminal-or-hold states for current baseline and must stay explicit",
        commit_boundary:
            "rejected or insufficient candidates are never committed",
    },
    BlueBrainFutureMemoryAttachmentLane {
        class: BlueBrainFutureMemoryAttachmentClass::PersistenceDeferredOrUnavailable,
        lane: "blue_brain_future_memory_attachment_persistence_unavailable_deferred",
        trigger_or_source:
            "blue_brain_candidate_persistence_unavailable_or_deferred + blue_brain_candidate_no_persistence_performed",
        required_fields:
            "candidate state + explicit deferred reason + canonical null persisted-memory lane reference",
        caveats:
            "deferred/unavailable must be visible to avoid implicit commit assumptions",
        commit_boundary:
            "explicitly no commit while real persistence path is absent",
    },
    BlueBrainFutureMemoryAttachmentLane {
        class: BlueBrainFutureMemoryAttachmentClass::PersistenceCommitOnlyIfRealPathExists,
        lane: "blue_brain_future_memory_attachment_commit_only_if_real_path_exists",
        trigger_or_source:
            "blue_brain_candidate_persistence_performed_only_if_real_path_exists",
        required_fields:
            "future real persistence contract id + candidate id + evidence/reference provenance + commit result envelope",
        caveats:
            "current baseline has no such real path; rule exists to constrain future implementation",
        commit_boundary:
            "commit allowed only when a real explicit persisted-memory contract is implemented",
    },
    BlueBrainFutureMemoryAttachmentLane {
        class: BlueBrainFutureMemoryAttachmentClass::HistoryReferenceBasisOnly,
        lane: "blue_brain_future_memory_attachment_history_snapshot_reference_basis_only",
        trigger_or_source:
            "job_history + replay_preflight/replay_with_entry + status_evidence_export_surface references",
        required_fields:
            "history/snapshot/replay identifiers + evidence digest references + caveat markers",
        caveats:
            "history/snapshot/evidence/replay remain reference basis only and cannot serve as memory commit proof",
        commit_boundary:
            "history/reference basis can support proposal quality only, never direct commit",
    },
    BlueBrainFutureMemoryAttachmentLane {
        class: BlueBrainFutureMemoryAttachmentClass::HistoryReferenceBasisOnly,
        lane: "blue_brain_future_memory_attachment_internal_paths_not_ready",
        trigger_or_source:
            "run_operation_with_entry/replay_with_entry expert lanes + backend worker/internal diagnostics",
        required_fields:
            "down-mapped outward status/evidence references if future integration is needed",
        caveats:
            "internal/expert-only persistence-like paths are non-canonical and not attachment-ready",
        commit_boundary:
            "internal paths cannot be used as commit authority",
    },
];

pub const CANONICAL_BLUE_BRAIN_FUTURE_MEMORY_HANDOFF_STATE_MAP:
    [BlueBrainFutureMemoryHandoffStateLane; 7] = [
    BlueBrainFutureMemoryHandoffStateLane {
        class: BlueBrainFutureMemoryHandoffStateClass::HandoffReady,
        lane: "blue_brain_future_memory_handoff_ready",
        trigger_or_source:
            "commit-eligible candidate + future-memory-ready marker + canonical evidence/reference basis",
        handoff_fields:
            "candidate identity + origin (context/evidence/replay/reference/compute-result/selection) + evidence/reference basis + selection status + caveats + freshness/staleness + commit-eligibility state",
        state_semantics:
            "handoff payload is ready for future subsystem intake and remains explicitly non-commit in current baseline",
        canonical_guard:
            "handoff-ready must never be labeled as committed memory by default",
    },
    BlueBrainFutureMemoryHandoffStateLane {
        class: BlueBrainFutureMemoryHandoffStateClass::HandoffDeferred,
        lane: "blue_brain_future_memory_handoff_deferred",
        trigger_or_source:
            "candidate deferred or persistence deferred while awaiting refreshed context/evidence",
        handoff_fields:
            "candidate identity + deferred reason + evidence/reference quality posture + selection/attention status + freshness",
        state_semantics:
            "handoff is postponed with explicit deferred posture; no commit progression is implied",
        canonical_guard:
            "deferred handoff must remain distinct from rejected/blocked/unavailable classes",
    },
    BlueBrainFutureMemoryHandoffStateLane {
        class: BlueBrainFutureMemoryHandoffStateClass::HandoffBlocked,
        lane: "blue_brain_future_memory_handoff_blocked",
        trigger_or_source:
            "blocking caveat/fault active or stale context gate prevents handoff advancement",
        handoff_fields:
            "candidate identity + blocking caveat + stale/insufficient markers + evidence/reference provenance",
        state_semantics:
            "handoff is blocked pending caveat clearance or refreshed basis and does not commit",
        canonical_guard:
            "blocked handoff cannot be promoted through history/snapshot/replay side records",
    },
    BlueBrainFutureMemoryHandoffStateLane {
        class: BlueBrainFutureMemoryHandoffStateClass::HandoffRejected,
        lane: "blue_brain_future_memory_handoff_rejected",
        trigger_or_source:
            "candidate rejected due to incompatibility/fault/caveat under canonical selection semantics",
        handoff_fields:
            "candidate identity + rejection reason + supporting evidence/reference posture + selection disposition",
        state_semantics:
            "handoff is rejected for current candidate instance and cannot transition to commit result",
        canonical_guard:
            "rejected handoff remains terminal-or-hold and non-commit",
    },
    BlueBrainFutureMemoryHandoffStateLane {
        class: BlueBrainFutureMemoryHandoffStateClass::HandoffCaveated,
        lane: "blue_brain_future_memory_handoff_caveated",
        trigger_or_source:
            "partial/caveated reference basis with explicit non-blocking caveats preserved",
        handoff_fields:
            "candidate identity + caveat set + partial reference/evidence basis + selection/attention caveats + freshness posture",
        state_semantics:
            "handoff is allowed with caveats and caveats must be preserved through diagnostics/result envelopes",
        canonical_guard:
            "caveated handoff does not imply clean commit eligibility or committed memory",
    },
    BlueBrainFutureMemoryHandoffStateLane {
        class: BlueBrainFutureMemoryHandoffStateClass::HandoffUnavailable,
        lane: "blue_brain_future_memory_handoff_unavailable",
        trigger_or_source:
            "no canonical future-memory intake path or required canonical handoff fields unavailable",
        handoff_fields:
            "candidate identity when available + unavailable reason + missing field markers + canonical null-path indicator",
        state_semantics:
            "handoff cannot be performed and candidate remains no-commit/future-memory-ready-or-deferred",
        canonical_guard:
            "unavailable handoff must stay explicit and cannot be inferred from stored references",
    },
    BlueBrainFutureMemoryHandoffStateLane {
        class: BlueBrainFutureMemoryHandoffStateClass::HandoffInternalOnlyNonCanonical,
        lane: "blue_brain_future_memory_handoff_internal_only_non_canonical",
        trigger_or_source:
            "run_operation_with_entry/replay_with_entry/internal hooks or legacy compat persistence-like paths",
        handoff_fields:
            "non-canonical source markers + required remap targets toward outward candidate/evidence/selection references",
        state_semantics:
            "internal-only path is marked non-canonical and excluded from BB5 future-memory handoff authority",
        canonical_guard:
            "internal/expert-only handoff path cannot serve as canonical handoff or commit authority",
    },
];

pub const CANONICAL_BLUE_BRAIN_COMMIT_RESULT_SEMANTICS_MAP: [BlueBrainCommitResultLane; 9] = [
    BlueBrainCommitResultLane {
        class: BlueBrainCommitResultClass::CommitUnavailable,
        lane: "blue_brain_commit_result_unavailable",
        trigger_or_source:
            "commit-eligible/future-memory-ready candidate where no real persisted-memory path exists",
        result_semantics:
            "commit unavailable is canonical baseline result and must preserve no persisted memory state",
        runtime_diagnostics_binding:
            "runtime diagnostics expose commit_unavailable with candidate id, evidence basis, and caveats",
        canonical_guard:
            "commit unavailable must stay separate from deferred and from reference/history recording",
    },
    BlueBrainCommitResultLane {
        class: BlueBrainCommitResultClass::CommitDeferred,
        lane: "blue_brain_commit_result_deferred",
        trigger_or_source:
            "candidate remains future-memory-ready/deferred pending future subsystem or refreshed eligibility gates",
        result_semantics:
            "commit deferred records postponed commit progression while preserving no-commit boundary",
        runtime_diagnostics_binding:
            "runtime diagnostics expose commit_deferred with explicit defer reason and selection/evidence posture",
        canonical_guard:
            "deferred is not a success and not equivalent to blocked/rejected",
    },
    BlueBrainCommitResultLane {
        class: BlueBrainCommitResultClass::CommitCommitted,
        lane: "blue_brain_commit_result_committed_only_if_real_path",
        trigger_or_source:
            "explicit canonical persisted-memory path commits candidate payload with required provenance",
        result_semantics:
            "committed result is valid only when a real repository persistence contract path exists and executes",
        runtime_diagnostics_binding:
            "runtime diagnostics expose committed result with persistence contract id and commit receipt reference",
        canonical_guard:
            "current baseline has no such path; this class is reserved and must not be synthesized",
    },
    BlueBrainCommitResultLane {
        class: BlueBrainCommitResultClass::CommitCommittedWithCaveats,
        lane: "blue_brain_commit_result_committed_with_caveats_only_if_real_path",
        trigger_or_source:
            "explicit canonical persisted-memory path commits while preserving non-blocking caveats",
        result_semantics:
            "committed-with-caveats is allowed only via real path and must preserve caveat envelope",
        runtime_diagnostics_binding:
            "runtime diagnostics expose committed_with_caveats and retain full caveat/evidence summary",
        canonical_guard:
            "cannot appear unless committed class is reachable through real persistence contract",
    },
    BlueBrainCommitResultLane {
        class: BlueBrainCommitResultClass::CommitRejected,
        lane: "blue_brain_commit_result_rejected",
        trigger_or_source:
            "candidate rejected by eligibility/selection/caveat gates before any commit attempt",
        result_semantics:
            "commit rejected indicates explicit refusal and no persisted-memory mutation",
        runtime_diagnostics_binding:
            "runtime diagnostics expose commit_rejected with rejection reason and supporting references",
        canonical_guard:
            "rejected commit result must remain distinct from blocked, failed, and unavailable",
    },
    BlueBrainCommitResultLane {
        class: BlueBrainCommitResultClass::CommitBlocked,
        lane: "blue_brain_commit_result_blocked",
        trigger_or_source:
            "blocking caveat/staleness/internal-dependency guard prevents commit progression",
        result_semantics:
            "commit blocked indicates gate denial without persistence side effects",
        runtime_diagnostics_binding:
            "runtime diagnostics expose commit_blocked with blocking gate and caveat/freshness markers",
        canonical_guard:
            "blocked cannot be auto-converted to deferred or committed by retries/history traces",
    },
    BlueBrainCommitResultLane {
        class: BlueBrainCommitResultClass::CommitFailed,
        lane: "blue_brain_commit_result_failed_only_if_real_path_attempted",
        trigger_or_source:
            "real commit path attempt fails after commit was attempted through canonical contract",
        result_semantics:
            "commit failed applies only to actual attempted persistence operations",
        runtime_diagnostics_binding:
            "runtime diagnostics expose commit_failed with failure code and attempt reference",
        canonical_guard:
            "without a real commit attempt, failed result must not be emitted",
    },
    BlueBrainCommitResultLane {
        class: BlueBrainCommitResultClass::CommitNoOp,
        lane: "blue_brain_commit_result_no_op",
        trigger_or_source:
            "commit request resolves to no-op under canonical idempotency or already-materialized semantics",
        result_semantics:
            "commit no-op records deterministic no-change outcome and is not equivalent to commit success",
        runtime_diagnostics_binding:
            "runtime diagnostics expose commit_no_op with idempotency reason and candidate reference",
        canonical_guard:
            "no-op cannot masquerade as committed result",
    },
    BlueBrainCommitResultLane {
        class: BlueBrainCommitResultClass::CommitReferenceRecordedOnly,
        lane: "blue_brain_commit_result_reference_recorded_only",
        trigger_or_source:
            "history/snapshot/evidence/replay entry recorded without any memory commit path",
        result_semantics:
            "reference recorded/evidence observed/handoff prepared remain non-commit outcomes",
        runtime_diagnostics_binding:
            "runtime diagnostics expose reference_recorded_only alongside commit_unavailable or deferred",
        canonical_guard:
            "history/snapshot/evidence/replay persistence must not be classified as memory commit result",
    },
];

pub const CANONICAL_BLUE_BRAIN_MEMORY_COMMIT_DIAGNOSTICS_MAP:
    [BlueBrainMemoryCommitDiagnosticLane; 10] = [
    BlueBrainMemoryCommitDiagnosticLane {
        class: BlueBrainMemoryCommitDiagnosticClass::HandoffDiagnostic,
        lane: "blue_brain_memory_commit_diagnostic_handoff",
        compact_reason: "handoff_prepared_with_candidate_and_reference_basis",
        handoff_or_commit_binding:
            "blue_brain_future_memory_handoff_ready|deferred|blocked|rejected|caveated|unavailable",
        candidate_lifecycle_binding:
            "blue_brain_candidate_proposed + blue_brain_candidate_accepted_for_future_memory_handling",
        selection_deferral_binding:
            "blue_brain_selection_diagnostic_selected|deferred|blocked + candidate_deferral_lifecycle",
        runtime_context_binding:
            "ComputeStatusEvidenceExportSurface::control_attention_diagnostics + context updated but not persisted",
        canonical_guard:
            "handoff diagnostics are candidate/runtime signals and not memory commit proof",
    },
    BlueBrainMemoryCommitDiagnosticLane {
        class: BlueBrainMemoryCommitDiagnosticClass::CommitEligibilityDiagnostic,
        lane: "blue_brain_memory_commit_diagnostic_commit_eligibility",
        compact_reason: "commit_eligible_with_sufficient_basis_and_non_stale_context",
        handoff_or_commit_binding:
            "blue_brain_memory_commit_boundary_candidate_commit_eligible + blue_brain_future_memory_handoff_ready",
        candidate_lifecycle_binding:
            "candidate remains future-memory-ready until explicit commit result exists",
        selection_deferral_binding:
            "selected candidate may become commit-eligible; insufficient candidate cannot become trigger/commit basis",
        runtime_context_binding:
            "runtime diagnostics show evidence attached but not committed",
        canonical_guard:
            "eligibility is a gate result and must not be interpreted as committed memory",
    },
    BlueBrainMemoryCommitDiagnosticLane {
        class: BlueBrainMemoryCommitDiagnosticClass::CommitRejectedDiagnostic,
        lane: "blue_brain_memory_commit_diagnostic_commit_rejected",
        compact_reason: "rejected_due_to_weak_or_insufficient_evidence_or_candidate_state",
        handoff_or_commit_binding:
            "blue_brain_future_memory_handoff_rejected + blue_brain_commit_result_rejected",
        candidate_lifecycle_binding:
            "candidate rejected after handoff and removed from active candidate set for current instance",
        selection_deferral_binding:
            "rejected candidate removed from future consideration unless a new candidate instance forms",
        runtime_context_binding:
            "runtime/context diagnostics expose rejection reason with no persistence performed",
        canonical_guard:
            "rejection reasons stay compact and canonical; no speculative reasoning prose",
    },
    BlueBrainMemoryCommitDiagnosticLane {
        class: BlueBrainMemoryCommitDiagnosticClass::CommitBlockedDiagnostic,
        lane: "blue_brain_memory_commit_diagnostic_commit_blocked",
        compact_reason:
            "blocked_due_to_stale_context_or_missing_persistence_path_or_internal_only_dependency",
        handoff_or_commit_binding:
            "blue_brain_future_memory_handoff_blocked + blue_brain_commit_result_blocked",
        candidate_lifecycle_binding:
            "candidate blocked from commit and remains non-persistent until blocking gate is cleared",
        selection_deferral_binding:
            "blocked candidate remains blocked/deferred and cannot trigger commit progression",
        runtime_context_binding:
            "runtime diagnostics expose blocked gate while context/evidence remain reference-only",
        canonical_guard:
            "blocked diagnostics must stay separate from rejected/deferred/unavailable",
    },
    BlueBrainMemoryCommitDiagnosticLane {
        class: BlueBrainMemoryCommitDiagnosticClass::CommitDeferredDiagnostic,
        lane: "blue_brain_memory_commit_diagnostic_commit_deferred",
        compact_reason: "candidate_deferred_after_handoff_pending_refresh_or_future_path",
        handoff_or_commit_binding:
            "blue_brain_future_memory_handoff_deferred + blue_brain_commit_result_deferred",
        candidate_lifecycle_binding:
            "candidate deferred after handoff and remains future-memory-ready",
        selection_deferral_binding:
            "deferred candidate remains deferred and recheckable under BB4 deferral semantics",
        runtime_context_binding:
            "runtime/context updated but not persisted and commit remains postponed",
        canonical_guard:
            "deferred is not equivalent to unavailable, blocked, or committed",
    },
    BlueBrainMemoryCommitDiagnosticLane {
        class: BlueBrainMemoryCommitDiagnosticClass::CommitCaveatedDiagnostic,
        lane: "blue_brain_memory_commit_diagnostic_commit_caveated",
        compact_reason: "caveated_due_to_partial_reference_basis",
        handoff_or_commit_binding:
            "blue_brain_future_memory_handoff_caveated + commit_result_committed_with_caveats_only_if_real_path",
        candidate_lifecycle_binding:
            "candidate remains caveated and recheckable without forced rejection",
        selection_deferral_binding:
            "caveated candidate remains recheckable and does not become clean trigger/commit basis",
        runtime_context_binding:
            "runtime diagnostics preserve caveat envelope and evidence quality posture",
        canonical_guard:
            "caveated diagnostics carry bounded caveats and are not an explainability platform",
    },
    BlueBrainMemoryCommitDiagnosticLane {
        class: BlueBrainMemoryCommitDiagnosticClass::CommitUnavailableDiagnostic,
        lane: "blue_brain_memory_commit_diagnostic_commit_unavailable",
        compact_reason: "unavailable_no_actual_memory_subsystem_exists_in_current_baseline",
        handoff_or_commit_binding:
            "blue_brain_future_memory_handoff_unavailable + blue_brain_commit_result_unavailable",
        candidate_lifecycle_binding:
            "candidate committed only if real path exists; otherwise no persistence performed",
        selection_deferral_binding:
            "selected candidate can still report commit unavailable while staying non-persistent",
        runtime_context_binding:
            "memory handoff prepared but memory commit unavailable",
        canonical_guard:
            "unavailable commit remains explicit baseline outcome and cannot be synthesized from history/snapshot",
    },
    BlueBrainMemoryCommitDiagnosticLane {
        class: BlueBrainMemoryCommitDiagnosticClass::CommittedIfPresentDiagnostic,
        lane: "blue_brain_memory_commit_diagnostic_committed_if_present",
        compact_reason: "committed_only_if_real_persistence_path_is_present",
        handoff_or_commit_binding:
            "blue_brain_memory_commit_boundary_committed_memory_only_if_real_path_exists + blue_brain_commit_result_committed_only_if_real_path",
        candidate_lifecycle_binding:
            "candidate may enter persisted class only through real explicit persistence contract",
        selection_deferral_binding:
            "selection cannot manufacture committed state without actual persistence execution",
        runtime_context_binding:
            "runtime diagnostics may report committed-if-present only with commit receipt reference",
        canonical_guard:
            "committed-if-present is reserved and absent in current baseline without real path",
    },
    BlueBrainMemoryCommitDiagnosticLane {
        class: BlueBrainMemoryCommitDiagnosticClass::NoPersistenceDiagnostic,
        lane: "blue_brain_memory_commit_diagnostic_no_persistence",
        compact_reason: "no_persistence_performed_context_updated_or_evidence_attached_only",
        handoff_or_commit_binding:
            "blue_brain_candidate_no_persistence_performed + blue_brain_commit_result_reference_recorded_only",
        candidate_lifecycle_binding:
            "candidate lifecycle and context changes stay visible while persistence remains none",
        selection_deferral_binding:
            "selection/deferral outcomes do not imply persistence side effects",
        runtime_context_binding:
            "context updated but not persisted; evidence attached but not committed",
        canonical_guard:
            "history/snapshot/evidence/replay/trace references are not memory commit outcomes",
    },
    BlueBrainMemoryCommitDiagnosticLane {
        class: BlueBrainMemoryCommitDiagnosticClass::NonCanonicalInternalOnlyDiagnostic,
        lane: "blue_brain_memory_commit_diagnostic_non_canonical_internal_only",
        compact_reason: "internal_only_non_canonical_commit_like_signal",
        handoff_or_commit_binding:
            "blue_brain_future_memory_handoff_internal_only_non_canonical + expert/internal hooks",
        candidate_lifecycle_binding:
            "internal diagnostics are excluded from canonical candidate lifecycle authority",
        selection_deferral_binding:
            "internal/expert-only diagnostics must not appear as canonical selection/deferral facts",
        runtime_context_binding:
            "runtime may surface marker with canonical=false only",
        canonical_guard:
            "non-canonical diagnostics are excluded unless down-mapped to outward candidate/evidence/selection/runtime references",
    },
];

pub const CANONICAL_BLUE_BRAIN_REFERENCE_CONTEXT_MAP: [BlueBrainReferenceContextLane; 12] = [
    BlueBrainReferenceContextLane {
        class: BlueBrainReferenceContextClass::EvidenceBackedContext,
        quality: BlueBrainReferenceQualityClass::Sufficient,
        lane: "blue_brain_reference_context_evidence_backed_sufficient",
        source_surface:
            "CanonicalComputeEntryPoint::status_evidence_export_surface(evidence refs + status posture)",
        runtime_context_semantics:
            "runtime context updated with evidence reference when outward evidence basis is sufficient",
        context_update_semantics:
            "context updated with evidence reference and evidence-grade provenance remains visible",
        candidate_semantics:
            "candidate may be evidence-backed without forcing candidate creation or commit",
        persistence_boundary: "no persistence implied by evidence-backed context update",
        canonical_guard:
            "evidence-backed context is canonical runtime/reference semantics, not memory persistence",
    },
    BlueBrainReferenceContextLane {
        class: BlueBrainReferenceContextClass::EvidenceBackedContext,
        quality: BlueBrainReferenceQualityClass::Partial,
        lane: "blue_brain_reference_context_evidence_backed_partial_caveated",
        source_surface:
            "status_evidence_export_surface partial posture + blue_brain_feedback_evidence_caveated_partial_or_insufficient",
        runtime_context_semantics:
            "context update caveated by partial evidence and runtime caveat posture remains explicit",
        context_update_semantics:
            "context update can proceed with caveat marker, without upgrading partial evidence to sufficient",
        candidate_semantics:
            "candidate basis may be weak-reference caveated and remains explicitly non-committed",
        persistence_boundary: "partial evidence never implies persistence or hidden memory write",
        canonical_guard: "partial evidence cannot be reinterpreted as durable memory authority",
    },
    BlueBrainReferenceContextLane {
        class: BlueBrainReferenceContextClass::EvidenceBackedContext,
        quality: BlueBrainReferenceQualityClass::Insufficient,
        lane: "blue_brain_reference_context_evidence_insufficient_blocked_update",
        source_surface:
            "blue_brain_transition_compute_trigger_blocked_insufficient_context + status_evidence_export_surface caveat posture",
        runtime_context_semantics:
            "evidence observed without context update when reference basis is insufficient",
        context_update_semantics: "context update blocked due to insufficient evidence",
        candidate_semantics:
            "candidate marked insufficient due to missing/weak evidence basis and remains non-persistent",
        persistence_boundary: "blocked/insufficient evidence path performs no persistence",
        canonical_guard:
            "insufficient evidence is explicit runtime posture and not promotable memory signal",
    },
    BlueBrainReferenceContextLane {
        class: BlueBrainReferenceContextClass::ReplayBackedContext,
        quality: BlueBrainReferenceQualityClass::Sufficient,
        lane: "blue_brain_reference_context_replay_backed_runtime_restored_or_informed",
        source_surface: "service_surface::{replay_preflight,replay_with_entry}",
        runtime_context_semantics:
            "runtime context restored or informed by replay/reference basis with explicit comparability scope",
        context_update_semantics:
            "context updated from replay/reference basis for runtime interpretation only",
        candidate_semantics:
            "candidate can be replay/reference-backed while remaining a non-commit proposal",
        persistence_boundary: "replay/reference used for context only, not memory commit",
        canonical_guard:
            "replay-backed context is reference-only semantic support and not a persistence channel",
    },
    BlueBrainReferenceContextLane {
        class: BlueBrainReferenceContextClass::ReplayBackedContext,
        quality: BlueBrainReferenceQualityClass::Caveated,
        lane: "blue_brain_reference_context_replay_backed_caveated",
        source_surface:
            "ReplayContextConsistencyClass + ReplayRemoteContextReproducibility caveat posture",
        runtime_context_semantics:
            "replay/reference context caveated when bridge fidelity or comparability is reduced",
        context_update_semantics:
            "context update remains caveated and may fall back to unchanged runtime context",
        candidate_semantics:
            "candidate replay/reference backing remains caveated and cannot imply acceptance quality",
        persistence_boundary: "caveated replay/reference posture keeps no persistence path",
        canonical_guard: "low-fidelity replay context cannot be promoted to memory state",
    },
    BlueBrainReferenceContextLane {
        class: BlueBrainReferenceContextClass::ReplayBackedContext,
        quality: BlueBrainReferenceQualityClass::Insufficient,
        lane: "blue_brain_reference_context_replay_reference_unavailable_or_insufficient",
        source_surface:
            "replay_preflight failure classes + replay context bridge insufficiency markers",
        runtime_context_semantics:
            "reference basis unavailable or insufficient and runtime context is not restored from replay",
        context_update_semantics: "context update blocked due to unavailable replay/reference basis",
        candidate_semantics:
            "candidate insufficient due to missing replay/reference basis; no commit side effect",
        persistence_boundary: "unavailable replay/reference basis performs no persistence",
        canonical_guard:
            "missing replay/reference basis must stay visible as insufficient reference quality",
    },
    BlueBrainReferenceContextLane {
        class: BlueBrainReferenceContextClass::SnapshotReferenceBackedContext,
        quality: BlueBrainReferenceQualityClass::Sufficient,
        lane: "blue_brain_reference_context_snapshot_reference_backed",
        source_surface:
            "status_evidence_export_surface(snapshot/history refs) + runtime_handoff_state_from_evidence",
        runtime_context_semantics:
            "snapshot/reference-backed context informs runtime posture with outward-facing snapshot refs",
        context_update_semantics:
            "context updated from snapshot/reference basis without turning snapshot into memory store",
        candidate_semantics:
            "candidate trace/snapshot-backed semantics remain explicit proposal-only states",
        persistence_boundary: "snapshot/reference context does not perform persistence",
        canonical_guard:
            "snapshot references are context evidence, not committed Blue-Brain memory",
    },
    BlueBrainReferenceContextLane {
        class: BlueBrainReferenceContextClass::TraceBackedContext,
        quality: BlueBrainReferenceQualityClass::Sufficient,
        lane: "blue_brain_reference_context_trace_backed",
        source_surface: "status_evidence_export_surface(trace refs) + trace slice exports",
        runtime_context_semantics:
            "trace-backed context attaches bounded trace references for runtime interpretation",
        context_update_semantics:
            "context can be updated with trace references while keeping trace and memory semantics split",
        candidate_semantics:
            "candidate trace/snapshot-backed references remain inspectable and non-persistent",
        persistence_boundary: "trace reference use performs no memory persistence",
        canonical_guard:
            "trace-backed context is evidence/reference semantics and not durable memory",
    },
    BlueBrainReferenceContextLane {
        class: BlueBrainReferenceContextClass::CaveatedReferenceContext,
        quality: BlueBrainReferenceQualityClass::Stale,
        lane: "blue_brain_reference_context_stale_or_age_limited",
        source_surface:
            "status/replay reference freshness posture (current|partial|stale) + candidate stale class",
        runtime_context_semantics:
            "stale reference basis remains visible as caveated runtime context quality",
        context_update_semantics:
            "context update caveated by stale reference basis and may be held at unchanged posture",
        candidate_semantics:
            "candidate caveated by weak/stale reference and can be marked stale or insufficient",
        persistence_boundary: "stale/caveated references remain non-persistent",
        canonical_guard: "stale basis cannot silently pass as sufficient reference quality",
    },
    BlueBrainReferenceContextLane {
        class: BlueBrainReferenceContextClass::CaveatedReferenceContext,
        quality: BlueBrainReferenceQualityClass::Caveated,
        lane: "blue_brain_reference_context_caveated_quality_explicit",
        source_surface:
            "blue_brain_feedback_evidence_caveated_partial_or_insufficient + diagnostic caveat lanes",
        runtime_context_semantics:
            "caveated reference basis is explicitly surfaced in runtime diagnostics and context posture",
        context_update_semantics:
            "context updates may proceed with explicit caveat tags and no hidden quality promotion",
        candidate_semantics:
            "candidate remains caveated by weak reference quality and no persistence performed",
        persistence_boundary: "caveat visibility has no persistence side effect",
        canonical_guard: "caveated reference posture must remain first-class and deterministic",
    },
    BlueBrainReferenceContextLane {
        class: BlueBrainReferenceContextClass::InsufficientReferenceContext,
        quality: BlueBrainReferenceQualityClass::Insufficient,
        lane: "blue_brain_reference_context_insufficient_basis_explicit",
        source_surface:
            "blocked/insufficient transition + candidate insufficient lifecycle lane",
        runtime_context_semantics:
            "insufficient reference basis explicitly blocks reference-backed context mutation",
        context_update_semantics:
            "context update blocked due to insufficient reference basis and remains observable",
        candidate_semantics:
            "candidate insufficient due to missing/stale reference with no persistence performed",
        persistence_boundary: "insufficient basis cannot produce automatic memory persistence",
        canonical_guard: "insufficient status remains explicit across context and candidate semantics",
    },
    BlueBrainReferenceContextLane {
        class: BlueBrainReferenceContextClass::NonCanonicalInternalOnlyReferencePath,
        quality: BlueBrainReferenceQualityClass::Caveated,
        lane: "blue_brain_reference_context_non_canonical_internal_only_path",
        source_surface:
            "run_operation_with_entry/replay_with_entry + build_backend(kind=stub|candle|worker) + domains/ai*",
        runtime_context_semantics:
            "internal/expert-only reference paths are non-canonical for Blue-Brain-facing context authority",
        context_update_semantics:
            "internal references require down-mapping to outward status/evidence refs before context update",
        candidate_semantics:
            "internal-only reference paths cannot appear as canonical candidate backing sources",
        persistence_boundary: "non-canonical/internal-only path has no memory persistence authority",
        canonical_guard:
            "mark non-canonical/internal-only reference paths explicit and exclude from canonical source set",
    },
];

pub const CANONICAL_BLUE_BRAIN_CONTROL_ATTENTION_SELECTION_MAP:
    [BlueBrainControlAttentionSelectionLane; 22] = [
    BlueBrainControlAttentionSelectionLane {
        class: BlueBrainControlAttentionSelectionClass::AttentionTarget,
        disposition: BlueBrainSelectionDispositionClass::Selected,
        basis_quality: BlueBrainSelectionBasisQualityClass::Sufficient,
        lane: "blue_brain_attention_target_current_transition",
        selection_scope: "attention target for current runtime transition",
        source_surface:
            "blue_brain_transition_context_available + blue_brain_phase_state_context_available",
        compute_trigger_binding: "no compute trigger implied",
        memory_persistence_semantics: "attention target selection is transient and non-persistent",
        canonical_guard:
            "attention targeting is control/selection posture, not policy/reasoning/memory commit",
    },
    BlueBrainControlAttentionSelectionLane {
        class: BlueBrainControlAttentionSelectionClass::ContextSelection,
        disposition: BlueBrainSelectionDispositionClass::Selected,
        basis_quality: BlueBrainSelectionBasisQualityClass::Sufficient,
        lane: "blue_brain_context_selected_for_current_transition",
        selection_scope: "context selected for current transition",
        source_surface:
            "blue_brain_context_initialized_for_runtime_window + blue_brain_transition_context_available",
        compute_trigger_binding: "transition may proceed without compute trigger",
        memory_persistence_semantics: "context selection does not imply memory persistence",
        canonical_guard:
            "context selection stays runtime-scoped and distinct from memory, policy, and reasoning",
    },
    BlueBrainControlAttentionSelectionLane {
        class: BlueBrainControlAttentionSelectionClass::ContextSelection,
        disposition: BlueBrainSelectionDispositionClass::Selected,
        basis_quality: BlueBrainSelectionBasisQualityClass::Partial,
        lane: "blue_brain_context_selected_for_compute_trigger_with_caveat",
        selection_scope: "context selected for compute trigger with caveat",
        source_surface:
            "blue_brain_transition_context_used_for_compute_trigger + blue_brain_feedback_evidence_caveated_partial_or_insufficient",
        compute_trigger_binding:
            "compute trigger selected on CanonicalComputeEntryPoint::submit with caveat posture",
        memory_persistence_semantics: "compute-trigger context selection performs no memory commit",
        canonical_guard:
            "caveated context selection remains explicit and cannot be upgraded to reasoning/persistence",
    },
    BlueBrainControlAttentionSelectionLane {
        class: BlueBrainControlAttentionSelectionClass::ContextSelection,
        disposition: BlueBrainSelectionDispositionClass::Deferred,
        basis_quality: BlueBrainSelectionBasisQualityClass::Partial,
        lane: "blue_brain_context_deferred_for_later_transition",
        selection_scope: "context deferred for later transition window",
        source_surface:
            "blue_brain_context_unchanged_after_transition_check + status_evidence_export_surface(status)",
        compute_trigger_binding: "compute trigger deferred",
        memory_persistence_semantics: "deferred context remains runtime-only and non-persistent",
        canonical_guard: "defer state is explicit control posture and not hidden no-op",
    },
    BlueBrainControlAttentionSelectionLane {
        class: BlueBrainControlAttentionSelectionClass::ContextSelection,
        disposition: BlueBrainSelectionDispositionClass::IgnoredOrIrrelevant,
        basis_quality: BlueBrainSelectionBasisQualityClass::Sufficient,
        lane: "blue_brain_context_ignored_as_not_relevant_to_transition",
        selection_scope: "context ignored/irrelevant to active transition",
        source_surface:
            "blue_brain_transition_status_evidence_update_without_compute_trigger + runtime context window",
        compute_trigger_binding: "no compute trigger implied",
        memory_persistence_semantics: "ignored context has no persistence side effect",
        canonical_guard: "ignored/irrelevant remains explicit and distinct from blocked/insufficient",
    },
    BlueBrainControlAttentionSelectionLane {
        class: BlueBrainControlAttentionSelectionClass::ContextSelection,
        disposition: BlueBrainSelectionDispositionClass::Blocked,
        basis_quality: BlueBrainSelectionBasisQualityClass::Stale,
        lane: "blue_brain_context_blocked_due_to_stale_basis",
        selection_scope: "context blocked due to stale reference basis",
        source_surface:
            "blue_brain_reference_context_stale_or_age_limited + blue_brain_candidate_stale_reference_basis",
        compute_trigger_binding: "compute trigger blocked due to stale selection basis",
        memory_persistence_semantics: "blocked context causes no memory commit",
        canonical_guard: "stale basis must not be treated as sufficient selection authority",
    },
    BlueBrainControlAttentionSelectionLane {
        class: BlueBrainControlAttentionSelectionClass::ContextSelection,
        disposition: BlueBrainSelectionDispositionClass::Caveated,
        basis_quality: BlueBrainSelectionBasisQualityClass::Caveated,
        lane: "blue_brain_context_selected_with_caveat",
        selection_scope: "context selected with explicit caveat marker",
        source_surface:
            "blue_brain_reference_context_caveated_quality_explicit + blue_brain_phase_caveated_degraded_partial_runtime_state",
        compute_trigger_binding: "compute trigger may proceed only with caveat posture propagated",
        memory_persistence_semantics: "caveated context selection remains non-persistent",
        canonical_guard: "caveated selection cannot imply policy override or reasoning finality",
    },
    BlueBrainControlAttentionSelectionLane {
        class: BlueBrainControlAttentionSelectionClass::EvidenceReferenceSelection,
        disposition: BlueBrainSelectionDispositionClass::Selected,
        basis_quality: BlueBrainSelectionBasisQualityClass::Sufficient,
        lane: "blue_brain_evidence_reference_selected",
        selection_scope: "evidence/reference selected for runtime context support",
        source_surface:
            "blue_brain_reference_context_evidence_backed_sufficient + status_evidence_export_surface(evidence refs)",
        compute_trigger_binding: "may inform compute trigger selection without forcing trigger",
        memory_persistence_semantics: "evidence selection has no memory commit implied",
        canonical_guard: "evidence/reference selection is not an audit or reasoning claim",
    },
    BlueBrainControlAttentionSelectionLane {
        class: BlueBrainControlAttentionSelectionClass::EvidenceReferenceSelection,
        disposition: BlueBrainSelectionDispositionClass::IgnoredOrIrrelevant,
        basis_quality: BlueBrainSelectionBasisQualityClass::Partial,
        lane: "blue_brain_evidence_reference_ignored",
        selection_scope: "evidence/reference ignored due to low relevance",
        source_surface:
            "blue_brain_transition_status_evidence_update_without_compute_trigger + status posture",
        compute_trigger_binding: "no compute trigger implied",
        memory_persistence_semantics: "ignored evidence has no persistence effect",
        canonical_guard: "ignored evidence stays explicit and does not mutate candidate state",
    },
    BlueBrainControlAttentionSelectionLane {
        class: BlueBrainControlAttentionSelectionClass::EvidenceReferenceSelection,
        disposition: BlueBrainSelectionDispositionClass::Deferred,
        basis_quality: BlueBrainSelectionBasisQualityClass::Partial,
        lane: "blue_brain_evidence_reference_deferred",
        selection_scope: "evidence/reference deferred pending stronger basis",
        source_surface:
            "blue_brain_reference_context_evidence_backed_partial_caveated + replay context bridge",
        compute_trigger_binding: "compute trigger deferred until basis improves",
        memory_persistence_semantics: "deferred evidence selection remains non-persistent",
        canonical_guard: "deferred evidence state must remain separate from blocked/ignored states",
    },
    BlueBrainControlAttentionSelectionLane {
        class: BlueBrainControlAttentionSelectionClass::EvidenceReferenceSelection,
        disposition: BlueBrainSelectionDispositionClass::Insufficient,
        basis_quality: BlueBrainSelectionBasisQualityClass::Insufficient,
        lane: "blue_brain_evidence_reference_insufficient",
        selection_scope: "evidence/reference insufficient for selection",
        source_surface:
            "blue_brain_reference_context_insufficient_basis_explicit + blocked transition lane",
        compute_trigger_binding: "compute trigger blocked due to insufficient selection basis",
        memory_persistence_semantics: "insufficient evidence selection performs no memory commit",
        canonical_guard: "insufficient evidence cannot be promoted to selected by internal heuristics",
    },
    BlueBrainControlAttentionSelectionLane {
        class: BlueBrainControlAttentionSelectionClass::EvidenceReferenceSelection,
        disposition: BlueBrainSelectionDispositionClass::Caveated,
        basis_quality: BlueBrainSelectionBasisQualityClass::Caveated,
        lane: "blue_brain_evidence_reference_caveated",
        selection_scope: "evidence/reference selected with caveat",
        source_surface:
            "blue_brain_reference_context_replay_backed_caveated + diagnostic caveat feedback lanes",
        compute_trigger_binding: "compute trigger may proceed in caveated mode only",
        memory_persistence_semantics: "caveated evidence selection has no memory commit implied",
        canonical_guard: "caveated evidence stays reference-grade and non-audit-authoritative",
    },
    BlueBrainControlAttentionSelectionLane {
        class: BlueBrainControlAttentionSelectionClass::MemoryCandidateSelection,
        disposition: BlueBrainSelectionDispositionClass::Selected,
        basis_quality: BlueBrainSelectionBasisQualityClass::Sufficient,
        lane: "blue_brain_memory_candidate_selected_for_future_handling",
        selection_scope: "candidate selected for future memory handling",
        source_surface: "blue_brain_candidate_accepted_for_future_memory_handling",
        compute_trigger_binding: "no direct compute trigger required",
        memory_persistence_semantics: "selected candidate remains not persisted in current baseline",
        canonical_guard: "candidate selection is lifecycle posture and not persistence commit",
    },
    BlueBrainControlAttentionSelectionLane {
        class: BlueBrainControlAttentionSelectionClass::MemoryCandidateSelection,
        disposition: BlueBrainSelectionDispositionClass::Deferred,
        basis_quality: BlueBrainSelectionBasisQualityClass::Partial,
        lane: "blue_brain_memory_candidate_deferred",
        selection_scope: "candidate deferred for later memory handling review",
        source_surface:
            "blue_brain_candidate_persistence_unavailable_or_deferred + blue_brain_candidate_proposed",
        compute_trigger_binding: "compute trigger deferred or unchanged",
        memory_persistence_semantics: "deferred candidate remains not persisted",
        canonical_guard: "deferred candidate state must remain explicit and non-committing",
    },
    BlueBrainControlAttentionSelectionLane {
        class: BlueBrainControlAttentionSelectionClass::MemoryCandidateSelection,
        disposition: BlueBrainSelectionDispositionClass::Rejected,
        basis_quality: BlueBrainSelectionBasisQualityClass::Caveated,
        lane: "blue_brain_memory_candidate_rejected",
        selection_scope: "candidate rejected due to caveat/fault posture",
        source_surface: "blue_brain_candidate_rejected_due_to_fault_or_caveat",
        compute_trigger_binding: "no compute trigger implied by rejection",
        memory_persistence_semantics: "rejected candidate never persists",
        canonical_guard: "rejected state must stay distinct from ignored/deferred",
    },
    BlueBrainControlAttentionSelectionLane {
        class: BlueBrainControlAttentionSelectionClass::MemoryCandidateSelection,
        disposition: BlueBrainSelectionDispositionClass::IgnoredOrIrrelevant,
        basis_quality: BlueBrainSelectionBasisQualityClass::Partial,
        lane: "blue_brain_memory_candidate_ignored",
        selection_scope: "candidate ignored as not relevant to current memory-adjacent workflow",
        source_surface:
            "blue_brain_candidate_only_without_context_mutation + runtime context unchanged lane",
        compute_trigger_binding: "no compute trigger implied",
        memory_persistence_semantics: "ignored candidate has no persistence side effect",
        canonical_guard: "ignored candidate is explicit and not equivalent to rejection",
    },
    BlueBrainControlAttentionSelectionLane {
        class: BlueBrainControlAttentionSelectionClass::MemoryCandidateSelection,
        disposition: BlueBrainSelectionDispositionClass::Blocked,
        basis_quality: BlueBrainSelectionBasisQualityClass::Insufficient,
        lane: "blue_brain_memory_candidate_blocked_weak_reference_context",
        selection_scope: "candidate blocked due to weak evidence/reference basis",
        source_surface:
            "blue_brain_candidate_insufficient_reference_basis + blue_brain_reference_context_insufficient_basis_explicit",
        compute_trigger_binding: "no compute trigger until basis improves",
        memory_persistence_semantics: "blocked candidate remains not persisted",
        canonical_guard: "blocked-by-weak-basis candidate must not be auto-promoted",
    },
    BlueBrainControlAttentionSelectionLane {
        class: BlueBrainControlAttentionSelectionClass::ComputeTriggerSelection,
        disposition: BlueBrainSelectionDispositionClass::Selected,
        basis_quality: BlueBrainSelectionBasisQualityClass::Sufficient,
        lane: "blue_brain_compute_trigger_selected_from_context",
        selection_scope: "compute trigger selected from context",
        source_surface: "blue_brain_transition_compute_trigger_from_context_availability",
        compute_trigger_binding: "CanonicalComputeEntryPoint::submit",
        memory_persistence_semantics: "compute-trigger selection has no memory commit implied",
        canonical_guard: "compute trigger selection stays on BB2 canonical handoff semantics",
    },
    BlueBrainControlAttentionSelectionLane {
        class: BlueBrainControlAttentionSelectionClass::ComputeTriggerSelection,
        disposition: BlueBrainSelectionDispositionClass::Selected,
        basis_quality: BlueBrainSelectionBasisQualityClass::Partial,
        lane: "blue_brain_compute_trigger_selected_from_evidence_reference_need",
        selection_scope: "compute trigger selected from evidence/reference need",
        source_surface: "blue_brain_transition_compute_trigger_from_inference_required",
        compute_trigger_binding:
            "CanonicalComputeEntryPoint::submit with evidence/reference caveat posture",
        memory_persistence_semantics: "evidence-driven trigger selection performs no memory commit",
        canonical_guard:
            "evidence/reference need can trigger compute without implying planning engine behavior",
    },
    BlueBrainControlAttentionSelectionLane {
        class: BlueBrainControlAttentionSelectionClass::ComputeTriggerSelection,
        disposition: BlueBrainSelectionDispositionClass::Blocked,
        basis_quality: BlueBrainSelectionBasisQualityClass::Insufficient,
        lane: "blue_brain_compute_trigger_blocked_insufficient_selection_basis",
        selection_scope: "compute trigger blocked due to insufficient selection basis",
        source_surface: "blue_brain_transition_compute_trigger_blocked_insufficient_context",
        compute_trigger_binding: "trigger remains blocked on canonical BB2 boundary",
        memory_persistence_semantics: "blocked compute trigger has no memory side effect",
        canonical_guard: "blocked trigger must remain distinct from deferred and ignored",
    },
    BlueBrainControlAttentionSelectionLane {
        class: BlueBrainControlAttentionSelectionClass::ComputeTriggerSelection,
        disposition: BlueBrainSelectionDispositionClass::Deferred,
        basis_quality: BlueBrainSelectionBasisQualityClass::Caveated,
        lane: "blue_brain_compute_trigger_deferred",
        selection_scope: "compute trigger deferred under caveated/partial basis",
        source_surface:
            "blue_brain_transition_status_evidence_update_without_compute_trigger + context deferred lanes",
        compute_trigger_binding: "trigger deferred while staying on BB2 transition semantics",
        memory_persistence_semantics: "deferred trigger performs no memory persistence",
        canonical_guard: "deferred trigger is explicit control state and not hidden failure/no-op",
    },
    BlueBrainControlAttentionSelectionLane {
        class: BlueBrainControlAttentionSelectionClass::NonCanonicalInternalOnlySelectionPath,
        disposition: BlueBrainSelectionDispositionClass::Blocked,
        basis_quality: BlueBrainSelectionBasisQualityClass::Insufficient,
        lane: "blue_brain_compute_trigger_internal_expert_only_non_canonical",
        selection_scope: "internal/expert-only trigger or selection path",
        source_surface:
            "blue_brain_transition_compute_trigger_suppressed_internal_only_path + run_operation_with_entry/replay_with_entry",
        compute_trigger_binding: "no internal/expert-only trigger used as canonical authority",
        memory_persistence_semantics: "internal-only path cannot imply memory persistence",
        canonical_guard:
            "non-canonical internal selection/control paths are explicitly excluded from BB4 canonical surface",
    },
];

pub const CANONICAL_BLUE_BRAIN_COMPUTE_TRIGGER_ARBITRATION_MAP:
    [BlueBrainComputeTriggerArbitrationLane; 16] = [
    BlueBrainComputeTriggerArbitrationLane {
        class: BlueBrainComputeTriggerArbitrationClass::TriggerCandidate,
        source: BlueBrainComputeTriggerSourceClass::ContextDerived,
        invocation: BlueBrainSelectionGatedInvocationClass::InvocationRequested,
        basis_quality: BlueBrainSelectionBasisQualityClass::Sufficient,
        lane: "blue_brain_trigger_candidate_context_derived",
        arbitration_semantics:
            "context-derived candidate enters arbitration as explicit trigger candidate instead of implicit auto-invocation",
        selection_binding:
            "blue_brain_compute_trigger_selected_from_context + attention/context selection lanes",
        outward_compute_contract_binding:
            "selected candidate invokes CanonicalComputeEntryPoint::submit only after selection-gated approval",
        memory_commit_boundary: "trigger candidate never implies memory commit",
        canonical_guard:
            "prevents context updates from being conflated with scheduling/planning/reasoning engines",
    },
    BlueBrainComputeTriggerArbitrationLane {
        class: BlueBrainComputeTriggerArbitrationClass::TriggerCandidate,
        source: BlueBrainComputeTriggerSourceClass::EvidenceReferenceDerived,
        invocation: BlueBrainSelectionGatedInvocationClass::CaveatedInvocationAllowed,
        basis_quality: BlueBrainSelectionBasisQualityClass::Caveated,
        lane: "blue_brain_trigger_candidate_evidence_reference_derived_caveated",
        arbitration_semantics:
            "evidence/reference-derived candidate stays explicit and may be allowed with caveat posture",
        selection_binding:
            "blue_brain_compute_trigger_selected_from_evidence_reference_need + evidence/reference caveated lanes",
        outward_compute_contract_binding:
            "if selected, invocation stays on CanonicalComputeEntryPoint::submit with caveat propagation",
        memory_commit_boundary: "evidence-derived candidate has no persistence side effect",
        canonical_guard:
            "caveated evidence can gate invocation but cannot silently upgrade quality or imply policy override",
    },
    BlueBrainComputeTriggerArbitrationLane {
        class: BlueBrainComputeTriggerArbitrationClass::TriggerCandidate,
        source: BlueBrainComputeTriggerSourceClass::RuntimeStateDerived,
        invocation: BlueBrainSelectionGatedInvocationClass::NoInvocationDeferred,
        basis_quality: BlueBrainSelectionBasisQualityClass::Partial,
        lane: "blue_brain_trigger_candidate_runtime_state_derived_deferred",
        arbitration_semantics:
            "runtime-state-derived candidate can be deferred while keeping deferred state explicit",
        selection_binding:
            "blue_brain_compute_trigger_deferred + blue_brain_context_deferred_for_later_transition",
        outward_compute_contract_binding:
            "deferred candidate does not call CanonicalComputeEntryPoint::submit",
        memory_commit_boundary: "runtime-state deferral performs no memory commit",
        canonical_guard:
            "deferred arbitration state is not hidden no-op and not a scheduler/planner construct",
    },
    BlueBrainComputeTriggerArbitrationLane {
        class: BlueBrainComputeTriggerArbitrationClass::TriggerCandidate,
        source: BlueBrainComputeTriggerSourceClass::MemoryCandidateDerived,
        invocation:
            BlueBrainSelectionGatedInvocationClass::InsufficientBasisRequiresMoreContextOrEvidence,
        basis_quality: BlueBrainSelectionBasisQualityClass::Insufficient,
        lane: "blue_brain_trigger_candidate_memory_candidate_insufficient",
        arbitration_semantics:
            "memory-candidate-derived trigger basis can be deemed insufficient and requires stronger context/evidence",
        selection_binding:
            "blue_brain_memory_candidate_blocked_weak_reference_context + candidate insufficient lifecycle lanes",
        outward_compute_contract_binding:
            "insufficient memory-candidate basis prevents invocation request on outward compute contract",
        memory_commit_boundary: "no compute and no persistence are implied by insufficient candidate basis",
        canonical_guard:
            "memory candidates can inform arbitration but cannot auto-trigger compute or commit memory",
    },
    BlueBrainComputeTriggerArbitrationLane {
        class: BlueBrainComputeTriggerArbitrationClass::SelectedTrigger,
        source: BlueBrainComputeTriggerSourceClass::ContextDerived,
        invocation: BlueBrainSelectionGatedInvocationClass::InvocationRequested,
        basis_quality: BlueBrainSelectionBasisQualityClass::Sufficient,
        lane: "blue_brain_trigger_selected_and_invocation_requested",
        arbitration_semantics:
            "selected trigger is canonical invocation request state after explicit arbitration",
        selection_binding:
            "blue_brain_compute_trigger_selected_from_context + blue_brain_transition_compute_trigger_from_context_availability",
        outward_compute_contract_binding:
            "invocation request path is CanonicalComputeEntryPoint::submit",
        memory_commit_boundary: "selected trigger has no automatic memory persistence semantics",
        canonical_guard:
            "keeps trigger prioritization as selection posture, not planning/policy/scheduling machinery",
    },
    BlueBrainComputeTriggerArbitrationLane {
        class: BlueBrainComputeTriggerArbitrationClass::DeferredTrigger,
        source: BlueBrainComputeTriggerSourceClass::RuntimeStateDerived,
        invocation: BlueBrainSelectionGatedInvocationClass::NoInvocationDeferred,
        basis_quality: BlueBrainSelectionBasisQualityClass::Partial,
        lane: "blue_brain_trigger_deferred_no_invocation",
        arbitration_semantics:
            "deferred trigger remains an explicit arbitration state and no invocation occurs",
        selection_binding:
            "blue_brain_compute_trigger_deferred + runtime status/evidence update without compute trigger",
        outward_compute_contract_binding:
            "deferred trigger keeps outward-facing invocation idle",
        memory_commit_boundary: "deferred trigger never implies memory persistence",
        canonical_guard:
            "deferred state remains semantically separate from blocked, suppressed, and failed invocation",
    },
    BlueBrainComputeTriggerArbitrationLane {
        class: BlueBrainComputeTriggerArbitrationClass::SuppressedTrigger,
        source: BlueBrainComputeTriggerSourceClass::ManualInternalOnlyNonCanonical,
        invocation: BlueBrainSelectionGatedInvocationClass::NoInvocationBlocked,
        basis_quality: BlueBrainSelectionBasisQualityClass::Insufficient,
        lane: "blue_brain_trigger_suppressed_internal_or_non_canonical_source",
        arbitration_semantics:
            "internal-only/manual trigger path is suppressed from canonical arbitration authority",
        selection_binding:
            "blue_brain_transition_compute_trigger_suppressed_internal_only_path + internal/expert-only selection lane",
        outward_compute_contract_binding:
            "suppressed trigger cannot reach CanonicalComputeEntryPoint::submit as canonical request",
        memory_commit_boundary: "suppressed internal trigger has no memory semantics",
        canonical_guard:
            "explicitly excludes expert/internal/compat trigger sources from canonical Blue-Brain invocation",
    },
    BlueBrainComputeTriggerArbitrationLane {
        class: BlueBrainComputeTriggerArbitrationClass::BlockedTrigger,
        source: BlueBrainComputeTriggerSourceClass::ContextDerived,
        invocation: BlueBrainSelectionGatedInvocationClass::NoInvocationBlocked,
        basis_quality: BlueBrainSelectionBasisQualityClass::Stale,
        lane: "blue_brain_trigger_blocked_stale_or_blocked_basis",
        arbitration_semantics:
            "blocked trigger state is explicit when context/reference basis is stale or blocked",
        selection_binding:
            "blue_brain_context_blocked_due_to_stale_basis + blue_brain_compute_trigger_blocked_insufficient_selection_basis",
        outward_compute_contract_binding:
            "blocked trigger state emits no outward invocation",
        memory_commit_boundary: "blocked trigger yields no compute and no persistence side effect",
        canonical_guard:
            "blocked semantics are explicit and must not be collapsed into deferred or failed execution",
    },
    BlueBrainComputeTriggerArbitrationLane {
        class: BlueBrainComputeTriggerArbitrationClass::InsufficientTriggerBasis,
        source: BlueBrainComputeTriggerSourceClass::EvidenceReferenceDerived,
        invocation:
            BlueBrainSelectionGatedInvocationClass::InsufficientBasisRequiresMoreContextOrEvidence,
        basis_quality: BlueBrainSelectionBasisQualityClass::Insufficient,
        lane: "blue_brain_trigger_insufficient_requires_context_or_evidence",
        arbitration_semantics:
            "insufficient trigger basis requires more context/evidence before invocation is eligible",
        selection_binding:
            "blue_brain_evidence_reference_insufficient + blue_brain_reference_context_insufficient_basis_explicit",
        outward_compute_contract_binding:
            "no invocation request until sufficient basis is available",
        memory_commit_boundary: "insufficient trigger basis cannot imply memory commit",
        canonical_guard:
            "keeps insufficient state visible and prevents heuristic auto-escalation to compute invocation",
    },
    BlueBrainComputeTriggerArbitrationLane {
        class: BlueBrainComputeTriggerArbitrationClass::CaveatedTrigger,
        source: BlueBrainComputeTriggerSourceClass::FeedbackDerived,
        invocation: BlueBrainSelectionGatedInvocationClass::CaveatedInvocationAllowed,
        basis_quality: BlueBrainSelectionBasisQualityClass::Caveated,
        lane: "blue_brain_trigger_caveated_but_allowed",
        arbitration_semantics:
            "feedback-derived caveated trigger may be allowed while caveat posture stays explicit",
        selection_binding:
            "blue_brain_feedback_result_integrated_with_caveat + blue_brain_evidence_reference_caveated",
        outward_compute_contract_binding:
            "if invoked, request remains on CanonicalComputeEntryPoint::submit with caveat/degraded visibility",
        memory_commit_boundary: "caveated invocation result updates runtime context only",
        canonical_guard:
            "caveated execution is allowed without introducing secondary compute-result semantics",
    },
    BlueBrainComputeTriggerArbitrationLane {
        class: BlueBrainComputeTriggerArbitrationClass::NonCanonicalInternalOnlyTrigger,
        source: BlueBrainComputeTriggerSourceClass::ManualInternalOnlyNonCanonical,
        invocation: BlueBrainSelectionGatedInvocationClass::NoInvocationBlocked,
        basis_quality: BlueBrainSelectionBasisQualityClass::Insufficient,
        lane: "blue_brain_trigger_non_canonical_internal_only",
        arbitration_semantics:
            "manual/internal-only trigger source is marked non-canonical for BB4 selection-gated invocation",
        selection_binding:
            "run_operation_with_entry/replay_with_entry + build_backend(kind=stub|candle|worker) + domains/ai*",
        outward_compute_contract_binding:
            "requires explicit down-mapping before any outward canonical invocation",
        memory_commit_boundary: "non-canonical trigger source has no commit authority",
        canonical_guard:
            "internal/expert/compat paths are excluded as canonical trigger authorities",
    },
    BlueBrainComputeTriggerArbitrationLane {
        class: BlueBrainComputeTriggerArbitrationClass::InvocationResultFeedback,
        source: BlueBrainComputeTriggerSourceClass::FeedbackDerived,
        invocation: BlueBrainSelectionGatedInvocationClass::InvocationCompleted,
        basis_quality: BlueBrainSelectionBasisQualityClass::Sufficient,
        lane: "blue_brain_trigger_invocation_completed_runtime_updated",
        arbitration_semantics:
            "successful invocation completes and result updates runtime context/evidence surfaces",
        selection_binding:
            "blue_brain_transition_compute_result_integrated + blue_brain_feedback_result_integrated_current_runtime_state",
        outward_compute_contract_binding:
            "completion result remains canonical output of CanonicalComputeEntryPoint::submit",
        memory_commit_boundary: "result integration updates runtime context, not memory commit",
        canonical_guard:
            "invocation completion feeds runtime/context surfaces without spawning parallel result contracts",
    },
    BlueBrainComputeTriggerArbitrationLane {
        class: BlueBrainComputeTriggerArbitrationClass::InvocationResultFeedback,
        source: BlueBrainComputeTriggerSourceClass::FeedbackDerived,
        invocation: BlueBrainSelectionGatedInvocationClass::InvocationFailed,
        basis_quality: BlueBrainSelectionBasisQualityClass::Insufficient,
        lane: "blue_brain_trigger_invocation_failed_runtime_caveated",
        arbitration_semantics:
            "failed invocation is explicit and feeds caveated/degraded runtime posture",
        selection_binding:
            "blue_brain_feedback_result_rejected_or_blocked + status/trust caveated/degraded transitions",
        outward_compute_contract_binding:
            "failure is represented on canonical outward status/fault surfaces",
        memory_commit_boundary: "failed invocation does not produce memory commit",
        canonical_guard:
            "keeps failed invocation distinct from blocked/no-invocation arbitration states",
    },
    BlueBrainComputeTriggerArbitrationLane {
        class: BlueBrainComputeTriggerArbitrationClass::InvocationResultFeedback,
        source: BlueBrainComputeTriggerSourceClass::FeedbackDerived,
        invocation: BlueBrainSelectionGatedInvocationClass::InvocationBlockedByComputeContract,
        basis_quality: BlueBrainSelectionBasisQualityClass::Insufficient,
        lane: "blue_brain_trigger_invocation_blocked_by_compute_contract",
        arbitration_semantics:
            "compute contract can block invocation even after request, and blocked result remains explicit",
        selection_binding:
            "CanonicalComputeEntryPoint::submit blocked posture + blue_brain_feedback_result_rejected_or_blocked",
        outward_compute_contract_binding:
            "blocked-by-contract status is surfaced on canonical status/evidence exports",
        memory_commit_boundary: "blocked invocation has no memory commit side effect",
        canonical_guard:
            "prevents blocked contract outcomes from being hidden behind retries or implicit orchestration helpers",
    },
    BlueBrainComputeTriggerArbitrationLane {
        class: BlueBrainComputeTriggerArbitrationClass::InvocationResultFeedback,
        source: BlueBrainComputeTriggerSourceClass::FeedbackDerived,
        invocation: BlueBrainSelectionGatedInvocationClass::InvocationCaveatedOrDegraded,
        basis_quality: BlueBrainSelectionBasisQualityClass::Caveated,
        lane: "blue_brain_trigger_invocation_caveated_or_degraded",
        arbitration_semantics:
            "caveated/degraded invocation result is explicit and remains within canonical feedback semantics",
        selection_binding:
            "blue_brain_feedback_result_integrated_with_caveat + blue_brain_phase_caveated_degraded_partial_runtime_state",
        outward_compute_contract_binding:
            "result remains outward-facing canonical status/evidence contract with caveated markers",
        memory_commit_boundary: "caveated/degraded result updates runtime context only",
        canonical_guard:
            "caveated/degraded outcomes do not create alternate compute result schemas or memory commits",
    },
    BlueBrainComputeTriggerArbitrationLane {
        class: BlueBrainComputeTriggerArbitrationClass::TriggerCandidate,
        source: BlueBrainComputeTriggerSourceClass::MemoryCandidateDerived,
        invocation: BlueBrainSelectionGatedInvocationClass::NoInvocationDeferred,
        basis_quality: BlueBrainSelectionBasisQualityClass::Partial,
        lane: "blue_brain_trigger_candidate_memory_candidate_deferred",
        arbitration_semantics:
            "memory candidate can be selected as trigger basis and still be deferred without invocation",
        selection_binding:
            "blue_brain_memory_candidate_deferred + blue_brain_candidate_persistence_unavailable_or_deferred",
        outward_compute_contract_binding: "no invocation while candidate trigger basis remains deferred",
        memory_commit_boundary: "deferred candidate trigger basis implies no commit",
        canonical_guard:
            "candidate-derived arbitration state must remain explicit and cannot auto-invoke compute",
    },
];

pub const CANONICAL_BLUE_BRAIN_CONTEXT_EVIDENCE_PRIORITY_MAP:
    [BlueBrainContextEvidencePriorityLane; 10] = [
    BlueBrainContextEvidencePriorityLane {
        class: BlueBrainContextEvidencePriorityClass::PrimaryContext,
        quality: BlueBrainSelectionBasisQualityClass::Sufficient,
        lane: "blue_brain_priority_primary_context_for_trigger",
        priority_semantics: "primary context selected as canonical trigger basis",
        source_binding:
            "blue_brain_context_selected_for_current_transition + blue_brain_compute_trigger_selected_from_context",
        trigger_arbitration_binding:
            "primary context can emit selected trigger on CanonicalComputeEntryPoint::submit",
        candidate_binding: "candidate selection may consume primary context without commit",
        deferral_or_caveat_reason: "none; sufficient context basis",
        recheck_condition: "not required while context remains current and sufficient",
        canonical_guard:
            "primary context priority is class-based and deterministic, not a numeric ranking engine",
    },
    BlueBrainContextEvidencePriorityLane {
        class: BlueBrainContextEvidencePriorityClass::SupportingContext,
        quality: BlueBrainSelectionBasisQualityClass::Partial,
        lane: "blue_brain_priority_supporting_context_caveated",
        priority_semantics: "supporting context is usable with explicit caveat posture",
        source_binding:
            "blue_brain_context_selected_for_compute_trigger_with_caveat + blue_brain_context_selected_with_caveat",
        trigger_arbitration_binding: "supporting context may allow caveated trigger",
        candidate_binding: "candidate may be proposed with caveated supporting context",
        deferral_or_caveat_reason: "partial context quality requires caveat visibility",
        recheck_condition:
            "recheck when stronger evidence/context update improves from partial to sufficient",
        canonical_guard:
            "supporting class cannot silently upgrade to primary or hide caveats",
    },
    BlueBrainContextEvidencePriorityLane {
        class: BlueBrainContextEvidencePriorityClass::DeferredContext,
        quality: BlueBrainSelectionBasisQualityClass::Partial,
        lane: "blue_brain_priority_deferred_context_pending_update",
        priority_semantics:
            "context is deferred: not selected now, still potentially relevant later",
        source_binding:
            "blue_brain_context_deferred_for_later_transition + blue_brain_compute_trigger_deferred",
        trigger_arbitration_binding: "deferred context emits no trigger invocation yet",
        candidate_binding: "deferred candidate/context remains non-persisted and reviewable",
        deferral_or_caveat_reason:
            "basis currently partial/caveated and not strong enough for selected trigger",
        recheck_condition:
            "recheck on context update or stronger evidence/reference basis becoming available",
        canonical_guard:
            "deferred context is not rejected, not ignored, and not a hidden scheduler lane",
    },
    BlueBrainContextEvidencePriorityLane {
        class: BlueBrainContextEvidencePriorityClass::IgnoredContext,
        quality: BlueBrainSelectionBasisQualityClass::Sufficient,
        lane: "blue_brain_priority_ignored_context_not_relevant",
        priority_semantics: "context explicitly ignored for current transition scope",
        source_binding: "blue_brain_context_ignored_as_not_relevant_to_transition",
        trigger_arbitration_binding: "ignored context produces no trigger candidate",
        candidate_binding: "ignored context does not mutate candidate acceptance/rejection",
        deferral_or_caveat_reason: "not relevant to active transition scope",
        recheck_condition: "recheck only when transition scope changes",
        canonical_guard:
            "ignored context remains semantically separate from deferred, stale, and rejected",
    },
    BlueBrainContextEvidencePriorityLane {
        class: BlueBrainContextEvidencePriorityClass::StaleContext,
        quality: BlueBrainSelectionBasisQualityClass::Stale,
        lane: "blue_brain_priority_stale_context_blocked",
        priority_semantics: "stale context stays visible and blocked from trigger selection",
        source_binding: "blue_brain_context_blocked_due_to_stale_basis",
        trigger_arbitration_binding: "stale context blocks trigger selection",
        candidate_binding: "candidates on stale basis may be marked stale",
        deferral_or_caveat_reason: "reference freshness is stale/aged",
        recheck_condition: "recheck when freshness returns to current/partial and is reviewable",
        canonical_guard: "stale state cannot be collapsed into deferred or insufficient",
    },
    BlueBrainContextEvidencePriorityLane {
        class: BlueBrainContextEvidencePriorityClass::InsufficientContext,
        quality: BlueBrainSelectionBasisQualityClass::Insufficient,
        lane: "blue_brain_priority_insufficient_context_blocks_trigger",
        priority_semantics: "insufficient context explicitly blocks trigger eligibility",
        source_binding:
            "blue_brain_context_update_blocked_insufficient_evidence + blue_brain_evidence_reference_insufficient",
        trigger_arbitration_binding:
            "insufficient basis requires more context/evidence and blocks invocation",
        candidate_binding: "candidate can be marked insufficient with no commit",
        deferral_or_caveat_reason: "insufficient context/evidence basis",
        recheck_condition: "recheck only when sufficient basis is observed",
        canonical_guard:
            "insufficient posture remains explicit and cannot auto-escalate by internal heuristics",
    },
    BlueBrainContextEvidencePriorityLane {
        class: BlueBrainContextEvidencePriorityClass::PrimaryEvidenceReference,
        quality: BlueBrainSelectionBasisQualityClass::Sufficient,
        lane: "blue_brain_priority_primary_evidence_reference",
        priority_semantics: "primary evidence/reference selected as high-confidence basis",
        source_binding:
            "blue_brain_evidence_reference_selected + blue_brain_reference_context_evidence_backed_sufficient",
        trigger_arbitration_binding: "primary evidence can support selected trigger candidate",
        candidate_binding: "candidate may be evidence-backed with sufficient quality",
        deferral_or_caveat_reason: "none; sufficient evidence quality",
        recheck_condition: "recheck on evidence freshness or caveat transitions",
        canonical_guard:
            "evidence priority remains reference-grade and not a planning/reasoning authority",
    },
    BlueBrainContextEvidencePriorityLane {
        class: BlueBrainContextEvidencePriorityClass::SupportingEvidenceReference,
        quality: BlueBrainSelectionBasisQualityClass::Partial,
        lane: "blue_brain_priority_supporting_evidence_reference",
        priority_semantics: "supporting evidence/reference informs but does not dominate selection",
        source_binding:
            "blue_brain_evidence_reference_deferred + blue_brain_reference_context_evidence_backed_partial_caveated",
        trigger_arbitration_binding: "supporting evidence tends to deferred trigger posture",
        candidate_binding: "candidate deferral can be justified by supporting-only evidence",
        deferral_or_caveat_reason: "partial evidence quality pending stronger basis",
        recheck_condition: "recheck when additional evidence improves quality",
        canonical_guard:
            "supporting evidence is distinct from primary and from insufficient evidence",
    },
    BlueBrainContextEvidencePriorityLane {
        class: BlueBrainContextEvidencePriorityClass::CaveatedEvidenceReference,
        quality: BlueBrainSelectionBasisQualityClass::Caveated,
        lane: "blue_brain_priority_caveated_evidence_reference",
        priority_semantics: "caveated evidence remains selectable only with caveat propagation",
        source_binding: "blue_brain_evidence_reference_caveated",
        trigger_arbitration_binding: "caveated evidence permits caveated trigger only",
        candidate_binding: "candidate remains caveated and non-commit",
        deferral_or_caveat_reason: "quality caveat must remain explicit",
        recheck_condition: "recheck after caveat resolution or stronger corroboration",
        canonical_guard:
            "caveated evidence cannot be represented as fully sufficient or ignored silently",
    },
    BlueBrainContextEvidencePriorityLane {
        class: BlueBrainContextEvidencePriorityClass::NonCanonicalInternalOnlyPriorityPath,
        quality: BlueBrainSelectionBasisQualityClass::Insufficient,
        lane: "blue_brain_priority_non_canonical_internal_only_path",
        priority_semantics:
            "internal/expert-only priority hints are non-canonical for BB4 authority",
        source_binding:
            "blue_brain_compute_trigger_internal_expert_only_non_canonical + run_operation_with_entry/replay_with_entry",
        trigger_arbitration_binding:
            "non-canonical path cannot directly trigger CanonicalComputeEntryPoint::submit",
        candidate_binding: "internal-only path cannot mark canonical candidate priority",
        deferral_or_caveat_reason: "requires down-mapping to canonical outward references",
        recheck_condition: "recheck only after canonical down-mapping is provided",
        canonical_guard:
            "prevents internal/legacy/compat paths from appearing as canonical priority authority",
    },
];

pub const CANONICAL_BLUE_BRAIN_CANDIDATE_DEFERRAL_LIFECYCLE_MAP:
    [BlueBrainCandidateDeferralLifecycleLane; 8] = [
    BlueBrainCandidateDeferralLifecycleLane {
        class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
        quality: BlueBrainSelectionBasisQualityClass::Sufficient,
        lane: "blue_brain_candidate_deferral_lifecycle_selected",
        lifecycle_semantics: "candidate selected for future handling and remains non-persisted",
        source_binding: "blue_brain_memory_candidate_selected_for_future_handling",
        deferral_reason: "none; selected as strongest available candidate basis",
        recheck_condition: "recheck as normal lifecycle review when new context/evidence arrives",
        trigger_binding: "candidate can inform trigger arbitration but does not auto-trigger compute",
        memory_commit_boundary: "selected candidate remains not persisted and no auto-commit exists",
        canonical_guard: "candidate selected state is lifecycle-only and not a memory commit",
    },
    BlueBrainCandidateDeferralLifecycleLane {
        class: BlueBrainCandidateDeferralLifecycleClass::CandidateDeferred,
        quality: BlueBrainSelectionBasisQualityClass::Partial,
        lane: "blue_brain_candidate_deferral_lifecycle_deferred",
        lifecycle_semantics:
            "candidate deferred means currently not selected but still potentially relevant",
        source_binding: "blue_brain_memory_candidate_deferred",
        deferral_reason: "candidate deferred due to partial/caveated basis",
        recheck_condition: "recheck when stronger evidence or context update is available",
        trigger_binding: "deferred candidate does not trigger compute",
        memory_commit_boundary: "deferred candidate is not persisted and not rejected",
        canonical_guard:
            "deferred is distinct from rejected/ignored/stale/insufficient and remains explicit",
    },
    BlueBrainCandidateDeferralLifecycleLane {
        class: BlueBrainCandidateDeferralLifecycleClass::CandidateDeferredPendingStrongerEvidence,
        quality: BlueBrainSelectionBasisQualityClass::Partial,
        lane: "blue_brain_candidate_deferral_pending_stronger_evidence",
        lifecycle_semantics:
            "candidate deferred pending stronger evidence/reference basis",
        source_binding:
            "blue_brain_candidate_evidence_backed_reference + blue_brain_evidence_reference_deferred",
        deferral_reason: "evidence quality partial/caveated and not yet sufficient",
        recheck_condition: "recheck when evidence quality reaches sufficient",
        trigger_binding: "no compute trigger until stronger evidence is observed",
        memory_commit_boundary: "pending-evidence deferral performs no memory commit",
        canonical_guard:
            "deferral pending stronger evidence is not a ranking score or planning output",
    },
    BlueBrainCandidateDeferralLifecycleLane {
        class: BlueBrainCandidateDeferralLifecycleClass::CandidateDeferredPendingContextUpdate,
        quality: BlueBrainSelectionBasisQualityClass::Partial,
        lane: "blue_brain_candidate_deferral_pending_context_update",
        lifecycle_semantics: "candidate deferred pending context refresh/transition update",
        source_binding:
            "blue_brain_context_deferred_for_later_transition + blue_brain_candidate_only_without_context_mutation",
        deferral_reason: "context transition window is not ready for selection",
        recheck_condition: "recheck when runtime context update completes",
        trigger_binding: "deferred pending context update keeps trigger in no-invocation posture",
        memory_commit_boundary: "no memory commit while pending context update",
        canonical_guard:
            "pending-context deferral is explicit lifecycle state, not hidden workflow scheduling",
    },
    BlueBrainCandidateDeferralLifecycleLane {
        class: BlueBrainCandidateDeferralLifecycleClass::CandidateRejected,
        quality: BlueBrainSelectionBasisQualityClass::Caveated,
        lane: "blue_brain_candidate_deferral_lifecycle_rejected",
        lifecycle_semantics: "candidate rejected due to fault/caveat posture",
        source_binding: "blue_brain_memory_candidate_rejected",
        deferral_reason: "rejected due to fault/caveat and removed from active candidate set",
        recheck_condition: "no recheck unless new candidate instance is formed",
        trigger_binding: "rejected candidate does not trigger compute",
        memory_commit_boundary: "rejected candidate is never persisted",
        canonical_guard: "rejected remains distinct from deferred and ignored",
    },
    BlueBrainCandidateDeferralLifecycleLane {
        class: BlueBrainCandidateDeferralLifecycleClass::CandidateStale,
        quality: BlueBrainSelectionBasisQualityClass::Stale,
        lane: "blue_brain_candidate_deferral_lifecycle_stale",
        lifecycle_semantics: "candidate stale due to aged reference basis",
        source_binding: "blue_brain_candidate_stale_reference_basis",
        deferral_reason: "reference freshness stale",
        recheck_condition: "recheck when refreshed reference basis exists",
        trigger_binding: "stale candidate cannot trigger compute",
        memory_commit_boundary: "stale candidate remains non-persistent",
        canonical_guard: "stale remains separate from insufficient and deferred",
    },
    BlueBrainCandidateDeferralLifecycleLane {
        class: BlueBrainCandidateDeferralLifecycleClass::CandidateInsufficient,
        quality: BlueBrainSelectionBasisQualityClass::Insufficient,
        lane: "blue_brain_candidate_deferral_lifecycle_insufficient",
        lifecycle_semantics: "candidate insufficient due to weak or missing basis",
        source_binding: "blue_brain_candidate_insufficient_reference_basis",
        deferral_reason: "insufficient context/evidence basis",
        recheck_condition: "recheck only on sufficient basis availability",
        trigger_binding: "insufficient candidate blocks trigger invocation",
        memory_commit_boundary: "insufficient candidate is not persisted",
        canonical_guard: "insufficient remains explicit and not merged with rejection",
    },
    BlueBrainCandidateDeferralLifecycleLane {
        class: BlueBrainCandidateDeferralLifecycleClass::CandidateNotPersisted,
        quality: BlueBrainSelectionBasisQualityClass::Insufficient,
        lane: "blue_brain_candidate_deferral_lifecycle_not_persisted",
        lifecycle_semantics:
            "all candidate outcomes explicitly carry not-persisted boundary in current baseline",
        source_binding:
            "blue_brain_candidate_persistence_unavailable_or_deferred + blue_brain_candidate_no_persistence_performed",
        deferral_reason: "actual memory persistence path intentionally absent/deferred",
        recheck_condition: "recheck only if future explicit persistence contract is added",
        trigger_binding: "not-persisted marker does not trigger compute",
        memory_commit_boundary: "no memory commit in current baseline",
        canonical_guard: "candidate deferral lifecycle is not a memory commit or consolidation engine",
    },
];

pub const CANONICAL_BLUE_BRAIN_SELECTION_DIAGNOSTICS_MAP: [BlueBrainSelectionDiagnosticLane; 8] = [
    BlueBrainSelectionDiagnosticLane {
        class: BlueBrainSelectionDiagnosticClass::SelectedItemDiagnostic,
        lane: "blue_brain_selection_diagnostic_selected",
        entity_scope: "context/evidence/trigger/candidate selected outcomes",
        outcome_binding:
            "selected due to sufficient context or selected due to primary evidence/reference",
        compact_reason: "selected_sufficient_context_or_primary_evidence",
        runtime_diagnostics_binding:
            "ComputeStatusEvidenceExportSurface::control_attention_diagnostics",
        state_surface_binding:
            "RuntimeOpsSnapshot selection-gated transition view + status_evidence_export_surface",
        canonical_guard:
            "selected diagnostics stay compact and avoid planning/reasoning/explainability claims",
    },
    BlueBrainSelectionDiagnosticLane {
        class: BlueBrainSelectionDiagnosticClass::DeferredItemDiagnostic,
        lane: "blue_brain_selection_diagnostic_deferred",
        entity_scope: "context/evidence/trigger/candidate deferred outcomes",
        outcome_binding:
            "deferred due to partial evidence or deferred pending context update",
        compact_reason: "deferred_partial_basis_or_pending_context_update",
        runtime_diagnostics_binding:
            "ComputeStatusEvidenceExportSurface::control_attention_diagnostics",
        state_surface_binding:
            "RuntimeOpsSnapshot transition remains selection-gated with no invocation/commit",
        canonical_guard:
            "deferred remains explicit and non-persistent; not a hidden scheduler/planner lane",
    },
    BlueBrainSelectionDiagnosticLane {
        class: BlueBrainSelectionDiagnosticClass::IgnoredItemDiagnostic,
        lane: "blue_brain_selection_diagnostic_ignored",
        entity_scope: "context/candidate outcomes ignored for current transition",
        outcome_binding: "ignored because not relevant to current transition",
        compact_reason: "ignored_not_relevant_to_current_transition",
        runtime_diagnostics_binding:
            "ComputeStatusEvidenceExportSurface::control_attention_diagnostics",
        state_surface_binding: "status/runtime context unchanged with explicit ignored marker",
        canonical_guard: "ignored remains distinct from rejected/deferred/blocked/insufficient",
    },
    BlueBrainSelectionDiagnosticLane {
        class: BlueBrainSelectionDiagnosticClass::RejectedItemDiagnostic,
        lane: "blue_brain_selection_diagnostic_rejected",
        entity_scope: "candidate/trigger paths rejected due to fault or caveat posture",
        outcome_binding: "rejected due to fault/caveat",
        compact_reason: "rejected_fault_or_caveat",
        runtime_diagnostics_binding:
            "ComputeStatusEvidenceExportSurface::control_attention_diagnostics",
        state_surface_binding:
            "RuntimeOpsSnapshot state remains explicit without memory persistence implication",
        canonical_guard: "rejected never implies compute invocation success or memory commit",
    },
    BlueBrainSelectionDiagnosticLane {
        class: BlueBrainSelectionDiagnosticClass::BlockedSelectionDiagnostic,
        lane: "blue_brain_selection_diagnostic_blocked",
        entity_scope: "context/evidence/trigger/candidate blocked outcomes",
        outcome_binding: "blocked due to stale/insufficient basis",
        compact_reason: "blocked_stale_or_insufficient_basis",
        runtime_diagnostics_binding:
            "ComputeStatusEvidenceExportSurface::control_attention_diagnostics",
        state_surface_binding:
            "selection caveat/blocked posture affects next trigger eligibility",
        canonical_guard: "blocked remains separate from failed/no-op/deferred",
    },
    BlueBrainSelectionDiagnosticLane {
        class: BlueBrainSelectionDiagnosticClass::InsufficientSelectionDiagnostic,
        lane: "blue_brain_selection_diagnostic_insufficient",
        entity_scope: "context/evidence/trigger/candidate insufficient outcomes",
        outcome_binding: "insufficient selection basis requires stronger context/evidence",
        compact_reason: "insufficient_requires_stronger_basis",
        runtime_diagnostics_binding:
            "ComputeStatusEvidenceExportSurface::control_attention_diagnostics",
        state_surface_binding:
            "runtime diagnostics observed but no memory persistence or auto-invocation implied",
        canonical_guard: "insufficient cannot be auto-promoted by internal heuristics",
    },
    BlueBrainSelectionDiagnosticLane {
        class: BlueBrainSelectionDiagnosticClass::CaveatedSelectionDiagnostic,
        lane: "blue_brain_selection_diagnostic_caveated",
        entity_scope: "selected/deferred trigger and evidence outcomes with caveats",
        outcome_binding: "caveated invocation allowed or caveated/degraded result observed",
        compact_reason: "caveated_invocation_or_result",
        runtime_diagnostics_binding:
            "ComputeStatusEvidenceExportSurface::control_attention_diagnostics",
        state_surface_binding:
            "selection caveat propagated to runtime state and next trigger eligibility",
        canonical_guard:
            "caveated diagnostics stay technical and compact; not policy/audit/reasoning claims",
    },
    BlueBrainSelectionDiagnosticLane {
        class: BlueBrainSelectionDiagnosticClass::NonCanonicalInternalOnlyDiagnosticDetail,
        lane: "blue_brain_selection_diagnostic_non_canonical_internal_only_detail",
        entity_scope: "internal/expert/legacy trigger-selection details",
        outcome_binding: "non-canonical detail marked and excluded from authority",
        compact_reason: "internal_only_non_canonical_detail",
        runtime_diagnostics_binding:
            "ComputeStatusEvidenceExportSurface::control_attention_diagnostics (canonical=false)",
        state_surface_binding:
            "RuntimeOpsSnapshot keeps non-canonical detail segregated from canonical authority",
        canonical_guard:
            "internal/expert-only details cannot appear as canonical BB4 control/attention diagnostics",
    },
];

pub const CANONICAL_BLUE_BRAIN_PLANNING_REASONING_CANDIDATE_MAP:
    [BlueBrainPlanningReasoningCandidateLane; 15] = [
    BlueBrainPlanningReasoningCandidateLane {
        class: BlueBrainPlanningReasoningCandidateClass::RuntimeDerivedPlanningCandidate,
        basis_state: BlueBrainPlanningReasoningCandidateBasisState::BasisAvailable,
        lane: "blue_brain_runtime_derived_planning_candidate_available",
        source_binding:
            "CANONICAL_BLUE_BRAIN_RUNTIME_SURFACE_MAP + CANONICAL_BLUE_BRAIN_TRANSITION_TRIGGER_MAP",
        candidate_semantics: "runtime state and trigger posture suggest a planning-near candidate basis",
        quality_or_caveat: "runtime-derived basis is sufficient for candidate framing only",
        resolution_boundary: "candidate basis available but no planner decision is resolved",
        no_execution_implication: "no plan selected and no action executed",
        memory_commit_boundary: "runtime-derived candidate does not imply memory commit",
        canonical_guard:
            "runtime/trigger semantics inform candidate basis without introducing a planning engine",
    },
    BlueBrainPlanningReasoningCandidateLane {
        class: BlueBrainPlanningReasoningCandidateClass::RuntimeDerivedPlanningCandidate,
        basis_state: BlueBrainPlanningReasoningCandidateBasisState::CandidateBlocked,
        lane: "blue_brain_trigger_suggested_candidate_blocked_basis",
        source_binding:
            "blue_brain_trigger_insufficient_requires_context_or_evidence + blue_brain_trigger_blocked_stale_or_blocked_basis",
        candidate_semantics: "trigger suggests action candidate but blocked/stale basis keeps it unresolved",
        quality_or_caveat: "blocked trigger yields blocked candidate basis",
        resolution_boundary: "candidate remains unresolved and blocked pending stronger canonical basis",
        no_execution_implication: "no planner decision implied and no invocation executed",
        memory_commit_boundary: "blocked trigger candidate has no persistence side effect",
        canonical_guard: "blocked trigger basis is explicit and cannot auto-promote to execution",
    },
    BlueBrainPlanningReasoningCandidateLane {
        class: BlueBrainPlanningReasoningCandidateClass::RuntimeDerivedPlanningCandidate,
        basis_state: BlueBrainPlanningReasoningCandidateBasisState::BasisPartialOrCaveated,
        lane: "blue_brain_trigger_caveated_candidate_basis",
        source_binding:
            "blue_brain_trigger_caveated_but_allowed + blue_brain_feedback_result_integrated_with_caveat",
        candidate_semantics:
            "trigger/evidence caveat posture can produce a caveated planning-near candidate basis",
        quality_or_caveat: "candidate basis is caveated and partial",
        resolution_boundary: "caveated candidate proposed but not resolved",
        no_execution_implication: "no action execution implied by caveated candidate availability",
        memory_commit_boundary: "caveated trigger candidate does not imply commit",
        canonical_guard:
            "caveated trigger lane stays selection/arbitration semantics and not planner authority",
    },
    BlueBrainPlanningReasoningCandidateLane {
        class: BlueBrainPlanningReasoningCandidateClass::ContextDerivedReasoningCandidate,
        basis_state: BlueBrainPlanningReasoningCandidateBasisState::BasisAvailable,
        lane: "blue_brain_context_derived_reasoning_candidate_sufficient",
        source_binding:
            "CANONICAL_BLUE_BRAIN_CONTEXT_MEMORY_SURFACE_MAP + CANONICAL_BLUE_BRAIN_CONTEXT_UPDATE_LIFECYCLE_MAP",
        candidate_semantics: "context-derived signal provides sufficient reasoning-candidate basis",
        quality_or_caveat: "sufficient context basis",
        resolution_boundary: "reasoning candidate proposed but not resolved",
        no_execution_implication: "candidate basis does not imply policy application or action execution",
        memory_commit_boundary: "context-derived candidate remains transient and non-committing",
        canonical_guard: "context basis is canonical candidate input and not a reasoning engine output",
    },
    BlueBrainPlanningReasoningCandidateLane {
        class: BlueBrainPlanningReasoningCandidateClass::EvidenceReferenceDerivedReasoningCandidate,
        basis_state: BlueBrainPlanningReasoningCandidateBasisState::BasisPartialOrCaveated,
        lane: "blue_brain_evidence_reference_reasoning_candidate_partial_caveated",
        source_binding:
            "CANONICAL_BLUE_BRAIN_REFERENCE_CONTEXT_MAP + CANONICAL_BLUE_BRAIN_CONTEXT_EVIDENCE_PRIORITY_MAP",
        candidate_semantics: "evidence/reference signal supports reasoning candidate with caveats",
        quality_or_caveat: "partial or caveated evidence/reference basis",
        resolution_boundary: "candidate remains unresolved until stronger evidence arrives",
        no_execution_implication: "no reasoning completion and no action execution implied",
        memory_commit_boundary: "evidence/reference basis has no direct commit authority",
        canonical_guard: "evidence/reference basis remains reference-grade and not audit/reasoning authority",
    },
    BlueBrainPlanningReasoningCandidateLane {
        class: BlueBrainPlanningReasoningCandidateClass::EvidenceReferenceDerivedReasoningCandidate,
        basis_state: BlueBrainPlanningReasoningCandidateBasisState::BasisStale,
        lane: "blue_brain_evidence_reference_reasoning_candidate_stale",
        source_binding:
            "blue_brain_context_blocked_due_to_stale_basis + blue_brain_reference_context_insufficient_basis_explicit",
        candidate_semantics: "stale evidence/reference basis is surfaced as stale reasoning candidate basis",
        quality_or_caveat: "stale basis requires recheck and cannot be silently upgraded",
        resolution_boundary: "candidate is stale and unresolved",
        no_execution_implication: "stale basis does not imply decision completion",
        memory_commit_boundary: "stale reasoning basis cannot imply memory persistence",
        canonical_guard: "stale basis must stay explicit and distinct from sufficient candidate basis",
    },
    BlueBrainPlanningReasoningCandidateLane {
        class: BlueBrainPlanningReasoningCandidateClass::SelectionDerivedActionCandidate,
        basis_state: BlueBrainPlanningReasoningCandidateBasisState::BasisAvailable,
        lane: "blue_brain_selection_derived_action_candidate_selected_context",
        source_binding:
            "CANONICAL_BLUE_BRAIN_CONTROL_ATTENTION_SELECTION_MAP + CANONICAL_BLUE_BRAIN_COMPUTE_TRIGGER_ARBITRATION_MAP",
        candidate_semantics: "selected context/trigger produces selection-derived action candidate basis",
        quality_or_caveat: "selected context yields candidate basis",
        resolution_boundary: "selection-derived candidate remains a basis state, not execution result",
        no_execution_implication: "selected candidate does not auto-execute action",
        memory_commit_boundary: "selection-derived action candidate has no memory commit semantics",
        canonical_guard: "selection is filtering/prioritization posture and not a planning/policy engine",
    },
    BlueBrainPlanningReasoningCandidateLane {
        class: BlueBrainPlanningReasoningCandidateClass::SelectionDerivedActionCandidate,
        basis_state: BlueBrainPlanningReasoningCandidateBasisState::CandidateDeferred,
        lane: "blue_brain_selection_derived_action_candidate_deferred",
        source_binding:
            "blue_brain_compute_trigger_deferred + blue_brain_candidate_deferred_pending_context_update",
        candidate_semantics: "deferred selection leaves candidate unresolved",
        quality_or_caveat: "candidate deferred pending stronger basis",
        resolution_boundary: "deferred candidate remains unresolved and recheck-gated",
        no_execution_implication: "deferred candidate implies no action execution",
        memory_commit_boundary: "deferred selection has no persistence side effect",
        canonical_guard: "deferred state remains explicit and separate from ignored/rejected",
    },
    BlueBrainPlanningReasoningCandidateLane {
        class: BlueBrainPlanningReasoningCandidateClass::SelectionDerivedActionCandidate,
        basis_state: BlueBrainPlanningReasoningCandidateBasisState::BasisInsufficient,
        lane: "blue_brain_selection_ignored_or_rejected_no_candidate",
        source_binding: "blue_brain_context_ignored_priority_lane + blue_brain_memory_candidate_rejected",
        candidate_semantics: "ignored or rejected selection item does not produce actionable candidate",
        quality_or_caveat: "ignored/rejected item is insufficient for candidate continuation",
        resolution_boundary: "no candidate resolved from ignored/rejected item",
        no_execution_implication: "no action execution implied",
        memory_commit_boundary: "rejected/ignored selection item has no commit authority",
        canonical_guard: "ignored and rejected states remain explicit non-candidate outcomes",
    },
    BlueBrainPlanningReasoningCandidateLane {
        class: BlueBrainPlanningReasoningCandidateClass::MemoryCandidateDerivedReasoningCandidate,
        basis_state: BlueBrainPlanningReasoningCandidateBasisState::CandidateProposedUnresolved,
        lane: "blue_brain_future_memory_ready_reasoning_candidate",
        source_binding:
            "CANONICAL_BLUE_BRAIN_MEMORY_CANDIDATE_LIFECYCLE_MAP + CANONICAL_BLUE_BRAIN_FUTURE_MEMORY_HANDOFF_STATE_MAP",
        candidate_semantics: "future-memory-ready candidate may support later reasoning work",
        quality_or_caveat: "candidate is preparatory and unresolved",
        resolution_boundary: "future-memory-ready is candidate basis only, not reasoning completion",
        no_execution_implication: "no plan/action execution implied",
        memory_commit_boundary: "future-memory-ready remains not committed unless real commit path exists",
        canonical_guard: "memory-candidate semantics remain separated from commit and reasoning engines",
    },
    BlueBrainPlanningReasoningCandidateLane {
        class: BlueBrainPlanningReasoningCandidateClass::MemoryCandidateDerivedReasoningCandidate,
        basis_state: BlueBrainPlanningReasoningCandidateBasisState::BasisInsufficient,
        lane: "blue_brain_memory_candidate_rejected_weakens_reasoning_basis",
        source_binding:
            "blue_brain_candidate_rejected_due_to_fault_or_caveat + blue_brain_memory_candidate_rejected",
        candidate_semantics: "rejected memory candidate weakens reasoning candidate basis",
        quality_or_caveat: "rejected candidate indicates insufficient/caveated basis",
        resolution_boundary: "reasoning candidate remains unresolved and degraded",
        no_execution_implication: "no action or policy execution implied",
        memory_commit_boundary: "rejected candidate cannot produce memory commit",
        canonical_guard: "rejected memory-candidate path cannot be promoted to canonical reasoning basis",
    },
    BlueBrainPlanningReasoningCandidateLane {
        class: BlueBrainPlanningReasoningCandidateClass::CommitFeedbackDerivedCandidate,
        basis_state: BlueBrainPlanningReasoningCandidateBasisState::BasisPartialOrCaveated,
        lane: "blue_brain_commit_feedback_candidate_unavailable_limits_basis",
        source_binding:
            "CANONICAL_BLUE_BRAIN_COMMIT_RESULT_SEMANTICS_MAP + CANONICAL_BLUE_BRAIN_MEMORY_COMMIT_DIAGNOSTICS_MAP",
        candidate_semantics: "commit unavailable/deferred feedback limits reasoning candidate basis",
        quality_or_caveat: "commit feedback signals partial or unavailable basis",
        resolution_boundary: "candidate basis remains caveated until real commit path exists",
        no_execution_implication: "commit-feedback candidate does not imply decision completion",
        memory_commit_boundary: "commit unavailable/deferred remains non-commit baseline",
        canonical_guard: "commit diagnostics inform candidate quality without creating a commit engine",
    },
    BlueBrainPlanningReasoningCandidateLane {
        class: BlueBrainPlanningReasoningCandidateClass::CommitFeedbackDerivedCandidate,
        basis_state: BlueBrainPlanningReasoningCandidateBasisState::BasisAvailable,
        lane: "blue_brain_committed_if_present_strengthens_basis_conditionally",
        source_binding:
            "blue_brain_commit_result_committed_if_present + blue_brain_committed_if_present_diagnostic",
        candidate_semantics:
            "committed-if-present can strengthen candidate basis only where real commit path exists",
        quality_or_caveat: "conditionally stronger basis",
        resolution_boundary: "candidate basis strengthened conditionally but still not reasoning completion",
        no_execution_implication: "no policy/action execution implied",
        memory_commit_boundary: "strengthening applies only if concrete commit path is real and canonical",
        canonical_guard:
            "committed-if-present is bounded diagnostic semantics and not a blanket commit guarantee",
    },
    BlueBrainPlanningReasoningCandidateLane {
        class: BlueBrainPlanningReasoningCandidateClass::InsufficientCandidateBasis,
        basis_state: BlueBrainPlanningReasoningCandidateBasisState::EvidenceObservedNoCandidate,
        lane: "blue_brain_evidence_observed_no_reasoning_candidate",
        source_binding:
            "blue_brain_feedback_evidence_observed_and_attached + blue_brain_evidence_reference_ignored",
        candidate_semantics: "evidence may be observed while no reasoning candidate is produced",
        quality_or_caveat: "observed evidence is not itself sufficient candidate basis",
        resolution_boundary: "no reasoning candidate proposed",
        no_execution_implication: "no reasoning/action/policy execution implied",
        memory_commit_boundary: "observed evidence remains non-commit reference",
        canonical_guard: "evidence observation is explicit and must not be conflated with reasoning outcome",
    },
    BlueBrainPlanningReasoningCandidateLane {
        class: BlueBrainPlanningReasoningCandidateClass::NonCanonicalInternalOnlyPlanningLikePath,
        basis_state: BlueBrainPlanningReasoningCandidateBasisState::BasisInsufficient,
        lane: "blue_brain_non_canonical_internal_planning_like_path",
        source_binding:
            "run_operation_with_entry/replay_with_entry/build_backend(kind=stub|candle|worker)/domains/ai*",
        candidate_semantics:
            "internal/expert/compat planning-like path remains non-canonical unless down-mapped",
        quality_or_caveat: "non-canonical internal-only basis",
        resolution_boundary: "path is excluded from canonical planning/reasoning candidate authority",
        no_execution_implication: "internal-only planning-like signal does not imply canonical action execution",
        memory_commit_boundary: "non-canonical internal-only path has no commit authority",
        canonical_guard:
            "compute-internal details and legacy hooks are excluded from canonical BB6 candidate basis",
    },
];

pub const CANONICAL_BLUE_BRAIN_CANDIDATE_ACTION_BOUNDARY_MAP:
    [BlueBrainCandidateActionBoundaryLane; 10] = [
    BlueBrainCandidateActionBoundaryLane {
        class: BlueBrainCandidateActionBoundaryClass::PlanningReasoningCandidate,
        execution_state: BlueBrainCandidateActionBoundaryExecutionState::NoExecutionPerformed,
        lane: "blue_brain_planning_reasoning_candidate_boundary",
        candidate_or_proposal_semantics:
            "planning/reasoning candidate is basis-only and not an action proposal by default",
        basis_binding: "CANONICAL_BLUE_BRAIN_PLANNING_REASONING_CANDIDATE_MAP",
        context_evidence_selection_binding:
            "CANONICAL_BLUE_BRAIN_CONTEXT_EVIDENCE_PRIORITY_MAP + CANONICAL_BLUE_BRAIN_CONTROL_ATTENTION_SELECTION_MAP",
        trigger_origin_binding: "CANONICAL_BLUE_BRAIN_COMPUTE_TRIGGER_ARBITRATION_MAP",
        memory_commit_feedback_binding:
            "CANONICAL_BLUE_BRAIN_COMMIT_RESULT_SEMANTICS_MAP + CANONICAL_BLUE_BRAIN_MEMORY_COMMIT_DIAGNOSTICS_MAP",
        caveat_binding: "candidate basis may be partial/caveated/stale/insufficient/deferred/blocked",
        compute_invocation_boundary: "candidate does not invoke CanonicalComputeEntryPoint::submit",
        memory_commit_boundary: "candidate does not imply memory commit",
        tool_execution_boundary: "candidate does not imply tool execution",
        canonical_guard:
            "candidate semantics stay separate from proposal/decision/execution semantics",
    },
    BlueBrainCandidateActionBoundaryLane {
        class: BlueBrainCandidateActionBoundaryClass::ActionProposalNonExecuting,
        execution_state: BlueBrainCandidateActionBoundaryExecutionState::NoExecutionPerformed,
        lane: "blue_brain_action_proposal_non_executing_created",
        candidate_or_proposal_semantics:
            "action proposal is a non-executing option derived from qualified candidate basis",
        basis_binding: "CANONICAL_BLUE_BRAIN_CANDIDATE_TO_PROPOSAL_TRANSITION_MAP",
        context_evidence_selection_binding:
            "proposal carries context/evidence/selection references as explicit basis",
        trigger_origin_binding: "proposal stores trigger/candidate origin reference",
        memory_commit_feedback_binding:
            "proposal may include memory-candidate/commit-feedback basis without persistence authority",
        caveat_binding: "proposal caveats remain explicit and unresolved until future action path",
        compute_invocation_boundary: "proposal creation performs no compute invocation",
        memory_commit_boundary: "proposal creation performs no memory commit",
        tool_execution_boundary: "proposal creation performs no tool execution",
        canonical_guard: "proposal is non-executing by contract",
    },
    BlueBrainCandidateActionBoundaryLane {
        class: BlueBrainCandidateActionBoundaryClass::SelectedProposal,
        execution_state: BlueBrainCandidateActionBoundaryExecutionState::SelectedForPossibleFutureAction,
        lane: "blue_brain_action_proposal_selected_future_action_ready_only",
        candidate_or_proposal_semantics:
            "selected proposal is selected for possible future action and remains non-executing",
        basis_binding: "CANONICAL_BLUE_BRAIN_NON_EXECUTING_ACTION_PROPOSAL_STATE_MAP",
        context_evidence_selection_binding: "selection references remain attached to proposal basis",
        trigger_origin_binding: "selected proposal may become future trigger-candidate only",
        memory_commit_feedback_binding: "commit feedback may adjust confidence, not execution",
        caveat_binding: "selected proposal can remain caveated or conditional",
        compute_invocation_boundary: "selection does not auto-invoke compute",
        memory_commit_boundary: "selection does not auto-commit memory",
        tool_execution_boundary: "selection does not auto-execute tools",
        canonical_guard: "selected proposal != executed action",
    },
    BlueBrainCandidateActionBoundaryLane {
        class: BlueBrainCandidateActionBoundaryClass::DeferredProposal,
        execution_state:
            BlueBrainCandidateActionBoundaryExecutionState::FutureActionReadyTriggerCandidateOnly,
        lane: "blue_brain_action_proposal_deferred",
        candidate_or_proposal_semantics:
            "deferred proposal remains pending stronger context/evidence or trigger posture",
        basis_binding: "CANONICAL_BLUE_BRAIN_NON_EXECUTING_ACTION_PROPOSAL_STATE_MAP",
        context_evidence_selection_binding: "deferred proposal keeps prior basis references explicit",
        trigger_origin_binding: "deferred proposal may re-enter as trigger-candidate later",
        memory_commit_feedback_binding: "defer state has no persistence authority",
        caveat_binding: "defer reasons/caveats remain attached",
        compute_invocation_boundary: "defer performs no compute invocation",
        memory_commit_boundary: "defer performs no memory commit",
        tool_execution_boundary: "defer performs no tool execution",
        canonical_guard: "deferred proposal stays unresolved and non-executing",
    },
    BlueBrainCandidateActionBoundaryLane {
        class: BlueBrainCandidateActionBoundaryClass::RejectedProposal,
        execution_state: BlueBrainCandidateActionBoundaryExecutionState::NoExecutionPerformed,
        lane: "blue_brain_action_proposal_rejected",
        candidate_or_proposal_semantics:
            "rejected proposal is explicitly closed and not promoted to action execution",
        basis_binding: "CANONICAL_BLUE_BRAIN_NON_EXECUTING_ACTION_PROPOSAL_STATE_MAP",
        context_evidence_selection_binding: "rejection references failed basis/caveat reasons",
        trigger_origin_binding: "rejected trigger/candidate origin remains diagnostic-only",
        memory_commit_feedback_binding: "rejection provides no commit authority",
        caveat_binding: "rejection rationale remains explicit for diagnostics",
        compute_invocation_boundary: "rejection performs no compute invocation",
        memory_commit_boundary: "rejection performs no memory commit",
        tool_execution_boundary: "rejection performs no tool execution",
        canonical_guard: "rejected proposal cannot be treated as selected/executed",
    },
    BlueBrainCandidateActionBoundaryLane {
        class: BlueBrainCandidateActionBoundaryClass::BlockedProposal,
        execution_state: BlueBrainCandidateActionBoundaryExecutionState::NoExecutionPerformed,
        lane: "blue_brain_action_proposal_blocked",
        candidate_or_proposal_semantics:
            "blocked proposal remains blocked on stale/insufficient basis",
        basis_binding: "CANONICAL_BLUE_BRAIN_NON_EXECUTING_ACTION_PROPOSAL_STATE_MAP",
        context_evidence_selection_binding: "blocked state references stale/insufficient basis explicitly",
        trigger_origin_binding: "blocked trigger source is retained as diagnostic context",
        memory_commit_feedback_binding: "blocked proposal has no commit authority",
        caveat_binding: "blocked basis caveat remains explicit",
        compute_invocation_boundary: "blocked proposal cannot invoke compute",
        memory_commit_boundary: "blocked proposal cannot commit memory",
        tool_execution_boundary: "blocked proposal cannot execute tools",
        canonical_guard: "blocked remains distinct from deferred/rejected",
    },
    BlueBrainCandidateActionBoundaryLane {
        class: BlueBrainCandidateActionBoundaryClass::CaveatedProposal,
        execution_state: BlueBrainCandidateActionBoundaryExecutionState::NoExecutionPerformed,
        lane: "blue_brain_action_proposal_caveated",
        candidate_or_proposal_semantics:
            "caveated proposal remains available with explicit caveats and no execution side effects",
        basis_binding: "CANONICAL_BLUE_BRAIN_NON_EXECUTING_ACTION_PROPOSAL_STATE_MAP",
        context_evidence_selection_binding: "partial/caveated context-evidence basis is preserved",
        trigger_origin_binding: "caveated trigger origin is carried forward explicitly",
        memory_commit_feedback_binding: "caveated commit-feedback basis remains non-committing",
        caveat_binding: "proposal caveat is canonical state, not implicit rejection",
        compute_invocation_boundary: "caveated proposal performs no compute invocation",
        memory_commit_boundary: "caveated proposal performs no memory commit",
        tool_execution_boundary: "caveated proposal performs no tool execution",
        canonical_guard: "caveat semantics are explicit and cannot be silently dropped",
    },
    BlueBrainCandidateActionBoundaryLane {
        class: BlueBrainCandidateActionBoundaryClass::InsufficientProposalBasis,
        execution_state: BlueBrainCandidateActionBoundaryExecutionState::NoExecutionPerformed,
        lane: "blue_brain_action_proposal_insufficient_basis",
        candidate_or_proposal_semantics:
            "insufficient proposal basis blocks proposal promotion and execution",
        basis_binding: "CANONICAL_BLUE_BRAIN_CANDIDATE_TO_PROPOSAL_TRANSITION_MAP",
        context_evidence_selection_binding:
            "insufficient context/evidence/selection basis remains explicit and unresolved",
        trigger_origin_binding: "insufficient trigger/candidate origin stays diagnostic",
        memory_commit_feedback_binding: "insufficient basis has no persistence authority",
        caveat_binding: "insufficient basis may carry caveats requiring re-check",
        compute_invocation_boundary: "insufficient basis cannot invoke compute",
        memory_commit_boundary: "insufficient basis cannot commit memory",
        tool_execution_boundary: "insufficient basis cannot execute tools",
        canonical_guard: "insufficient basis must remain visible and non-promoted",
    },
    BlueBrainCandidateActionBoundaryLane {
        class: BlueBrainCandidateActionBoundaryClass::ExecutedActionCanonicalIfPresent,
        execution_state:
            BlueBrainCandidateActionBoundaryExecutionState::ExecutedViaCanonicalComputePathOnlyIfExplicitlyInvoked,
        lane: "blue_brain_executed_action_only_via_explicit_canonical_path",
        candidate_or_proposal_semantics:
            "executed action exists only where explicit canonical invocation path is taken",
        basis_binding:
            "service_surface::CanonicalComputeEntryPoint::{submit,status,drain_scheduler}",
        context_evidence_selection_binding:
            "execution result/status/evidence remains separate from proposal-state semantics",
        trigger_origin_binding: "explicit invocation decision is required before execution",
        memory_commit_feedback_binding:
            "execution output/evidence is not a memory commit by default",
        caveat_binding: "caveated runtime result does not redefine proposal boundary",
        compute_invocation_boundary:
            "execution requires explicit call on canonical compute execution contract",
        memory_commit_boundary: "executed action does not auto-commit memory",
        tool_execution_boundary:
            "tool execution remains separately gated and not implied by proposal presence",
        canonical_guard: "proposal state never implies executed action",
    },
    BlueBrainCandidateActionBoundaryLane {
        class: BlueBrainCandidateActionBoundaryClass::NonCanonicalInternalOnlyActionLikePath,
        execution_state:
            BlueBrainCandidateActionBoundaryExecutionState::NonCanonicalInternalOnlyNoAuthority,
        lane: "blue_brain_non_canonical_internal_action_like_path",
        candidate_or_proposal_semantics:
            "internal/expert/compat action-like path is non-canonical for BB6 proposal authority",
        basis_binding:
            "run_operation_with_entry/replay_with_entry/build_backend(kind=stub|candle|worker)/domains/ai*",
        context_evidence_selection_binding:
            "non-canonical path must down-map to canonical context/evidence/selection references",
        trigger_origin_binding: "internal-only trigger source has no canonical outward authority",
        memory_commit_feedback_binding: "internal-only path has no memory commit authority",
        caveat_binding: "non-canonical marker is mandatory and load-bearing",
        compute_invocation_boundary:
            "internal-only path cannot be treated as canonical proposal-to-execution bridge",
        memory_commit_boundary: "internal-only path cannot promote proposal to commit",
        tool_execution_boundary: "internal-only helper path cannot imply tool execution authority",
        canonical_guard:
            "non-canonical action-like paths are excluded from BB6 canonical candidate/proposal boundary",
    },
];

pub const CANONICAL_BLUE_BRAIN_CANDIDATE_TO_PROPOSAL_TRANSITION_MAP:
    [BlueBrainCandidateToProposalTransitionLane; 6] = [
    BlueBrainCandidateToProposalTransitionLane {
        class: BlueBrainCandidateToProposalTransitionClass::CandidateRemainsCandidate,
        lane: "blue_brain_candidate_remains_candidate",
        source_candidate_binding: "CANONICAL_BLUE_BRAIN_PLANNING_REASONING_CANDIDATE_MAP",
        transition_semantics:
            "candidate remains candidate when basis is exploratory/deferred and not proposal-ready",
        proposal_outcome: "no proposal created",
        execution_boundary: "candidate remains non-executing",
        compute_boundary: "candidate remains non-invoking",
        memory_commit_boundary: "candidate remains non-committing",
        canonical_guard: "not every candidate yields a proposal",
    },
    BlueBrainCandidateToProposalTransitionLane {
        class: BlueBrainCandidateToProposalTransitionClass::CandidateYieldsActionProposal,
        lane: "blue_brain_candidate_yields_non_executing_action_proposal",
        source_candidate_binding:
            "runtime/context/evidence/selection basis with explicit trigger origin and caveat posture",
        transition_semantics: "qualified candidate yields action proposal as non-executing option",
        proposal_outcome: "proposal created",
        execution_boundary: "proposal creation does not execute action",
        compute_boundary: "proposal creation does not invoke compute",
        memory_commit_boundary: "proposal creation does not commit memory",
        canonical_guard: "proposal remains basis-bound and non-executing",
    },
    BlueBrainCandidateToProposalTransitionLane {
        class: BlueBrainCandidateToProposalTransitionClass::CandidateInsufficientForProposal,
        lane: "blue_brain_candidate_insufficient_for_proposal",
        source_candidate_binding: "insufficient/stale/blocked candidate basis",
        transition_semantics:
            "candidate basis is insufficient and cannot be promoted into action proposal",
        proposal_outcome: "proposal insufficient basis",
        execution_boundary: "insufficient candidate cannot execute action",
        compute_boundary: "insufficient candidate cannot invoke compute",
        memory_commit_boundary: "insufficient candidate cannot commit memory",
        canonical_guard: "insufficient stays explicit and non-promoted",
    },
    BlueBrainCandidateToProposalTransitionLane {
        class: BlueBrainCandidateToProposalTransitionClass::CandidateYieldsCaveatedProposal,
        lane: "blue_brain_candidate_yields_caveated_proposal",
        source_candidate_binding: "partial/caveated candidate basis",
        transition_semantics: "candidate yields caveated proposal with unresolved caveat posture",
        proposal_outcome: "proposal caveated",
        execution_boundary: "caveated proposal does not execute action",
        compute_boundary: "caveated proposal does not invoke compute",
        memory_commit_boundary: "caveated proposal does not commit memory",
        canonical_guard: "caveat remains explicit and load-bearing",
    },
    BlueBrainCandidateToProposalTransitionLane {
        class: BlueBrainCandidateToProposalTransitionClass::CandidateRejectedBeforeProposal,
        lane: "blue_brain_candidate_rejected_before_proposal",
        source_candidate_binding: "rejected candidate basis from selection/memory-candidate boundary",
        transition_semantics:
            "candidate is rejected prior to proposal creation and remains diagnostics-only",
        proposal_outcome: "proposal rejected",
        execution_boundary: "rejected candidate does not execute action",
        compute_boundary: "rejected candidate does not invoke compute",
        memory_commit_boundary: "rejected candidate does not commit memory",
        canonical_guard: "rejected-before-proposal is distinct from deferred/blocked",
    },
    BlueBrainCandidateToProposalTransitionLane {
        class: BlueBrainCandidateToProposalTransitionClass::CandidateDeferredBeforeProposal,
        lane: "blue_brain_candidate_deferred_before_proposal",
        source_candidate_binding: "deferred candidate basis pending context/evidence refresh",
        transition_semantics: "candidate is deferred prior to proposal creation",
        proposal_outcome: "proposal deferred",
        execution_boundary: "deferred candidate does not execute action",
        compute_boundary: "deferred candidate does not invoke compute",
        memory_commit_boundary: "deferred candidate does not commit memory",
        canonical_guard: "deferred-before-proposal remains explicit unresolved state",
    },
];

pub const CANONICAL_BLUE_BRAIN_NON_EXECUTING_ACTION_PROPOSAL_STATE_MAP:
    [BlueBrainNonExecutingActionProposalStateLane; 8] = [
    BlueBrainNonExecutingActionProposalStateLane {
        class: BlueBrainNonExecutingActionProposalStateClass::ProposalCreated,
        lane: "blue_brain_action_proposal_created",
        proposal_state_semantics: "proposal created from qualified candidate basis",
        proposal_basis_binding:
            "context basis + evidence/reference basis + selection state + trigger origin + caveats",
        execution_boundary: "proposal created: no execution performed",
        compute_trigger_boundary: "proposal created: no compute trigger performed",
        memory_commit_boundary: "proposal created: no memory commit performed",
        tool_execution_boundary: "proposal created: no tool execution performed",
        canonical_guard: "proposal creation is non-executing by default",
    },
    BlueBrainNonExecutingActionProposalStateLane {
        class: BlueBrainNonExecutingActionProposalStateClass::ProposalSelectedForPossibleFutureAction,
        lane: "blue_brain_action_proposal_selected",
        proposal_state_semantics: "proposal selected for possible future action",
        proposal_basis_binding: "selection outcome with preserved basis and caveats",
        execution_boundary: "proposal selected: no execution performed",
        compute_trigger_boundary: "proposal selected: no compute trigger performed",
        memory_commit_boundary: "proposal selected: no memory commit performed",
        tool_execution_boundary: "proposal selected: no tool execution performed",
        canonical_guard: "selected proposal is not executed action",
    },
    BlueBrainNonExecutingActionProposalStateLane {
        class: BlueBrainNonExecutingActionProposalStateClass::ProposalDeferred,
        lane: "blue_brain_action_proposal_deferred_state",
        proposal_state_semantics: "proposal deferred pending stronger basis",
        proposal_basis_binding: "defer reason ties to context/evidence/selection gap",
        execution_boundary: "proposal deferred: no execution performed",
        compute_trigger_boundary: "proposal deferred: no compute trigger performed",
        memory_commit_boundary: "proposal deferred: no memory commit performed",
        tool_execution_boundary: "proposal deferred: no tool execution performed",
        canonical_guard: "deferred remains separate from rejected/blocked",
    },
    BlueBrainNonExecutingActionProposalStateLane {
        class: BlueBrainNonExecutingActionProposalStateClass::ProposalRejected,
        lane: "blue_brain_action_proposal_rejected_state",
        proposal_state_semantics: "proposal rejected with explicit reason",
        proposal_basis_binding: "rejection references insufficient/invalid basis",
        execution_boundary: "proposal rejected: no execution performed",
        compute_trigger_boundary: "proposal rejected: no compute trigger performed",
        memory_commit_boundary: "proposal rejected: no memory commit performed",
        tool_execution_boundary: "proposal rejected: no tool execution performed",
        canonical_guard: "rejected proposal is terminal for current basis window",
    },
    BlueBrainNonExecutingActionProposalStateLane {
        class: BlueBrainNonExecutingActionProposalStateClass::ProposalBlocked,
        lane: "blue_brain_action_proposal_blocked_state",
        proposal_state_semantics: "proposal blocked due to stale/blocked basis",
        proposal_basis_binding: "blocked basis markers from trigger/context/evidence lanes",
        execution_boundary: "proposal blocked: no execution performed",
        compute_trigger_boundary: "proposal blocked: no compute trigger performed",
        memory_commit_boundary: "proposal blocked: no memory commit performed",
        tool_execution_boundary: "proposal blocked: no tool execution performed",
        canonical_guard: "blocked is explicit and non-interchangeable with deferred",
    },
    BlueBrainNonExecutingActionProposalStateLane {
        class: BlueBrainNonExecutingActionProposalStateClass::ProposalCaveated,
        lane: "blue_brain_action_proposal_caveated_state",
        proposal_state_semantics: "proposal caveated with partial/uncertain basis",
        proposal_basis_binding: "caveated context/evidence/selection basis with explicit caveat",
        execution_boundary: "proposal caveated: no execution performed",
        compute_trigger_boundary: "proposal caveated: no compute trigger performed",
        memory_commit_boundary: "proposal caveated: no memory commit performed",
        tool_execution_boundary: "proposal caveated: no tool execution performed",
        canonical_guard: "caveat state must be preserved in proposal lifecycle",
    },
    BlueBrainNonExecutingActionProposalStateLane {
        class: BlueBrainNonExecutingActionProposalStateClass::ProposalInsufficientBasis,
        lane: "blue_brain_action_proposal_insufficient_state",
        proposal_state_semantics: "proposal basis insufficient for progression",
        proposal_basis_binding: "insufficient candidate/context/evidence basis markers",
        execution_boundary: "proposal insufficient: no execution performed",
        compute_trigger_boundary: "proposal insufficient: no compute trigger performed",
        memory_commit_boundary: "proposal insufficient: no memory commit performed",
        tool_execution_boundary: "proposal insufficient: no tool execution performed",
        canonical_guard: "insufficient state blocks proposal promotion",
    },
    BlueBrainNonExecutingActionProposalStateLane {
        class: BlueBrainNonExecutingActionProposalStateClass::NoExecutionPerformed,
        lane: "blue_brain_action_proposal_no_execution_performed",
        proposal_state_semantics:
            "proposal lifecycle state confirms no execution/compute/tool/memory side effect occurred",
        proposal_basis_binding: "all proposal states keep non-executing boundary explicit",
        execution_boundary: "no execution performed",
        compute_trigger_boundary: "no compute trigger performed",
        memory_commit_boundary: "no memory commit performed",
        tool_execution_boundary: "no tool execution performed",
        canonical_guard: "BB6 proposal surface is non-executing by definition",
    },
];

pub const CANONICAL_BLUE_BRAIN_REASONING_CANDIDATE_DIAGNOSTICS_MAP:
    [BlueBrainReasoningCandidateDiagnosticLane; 10] = [
    BlueBrainReasoningCandidateDiagnosticLane {
        class: BlueBrainReasoningCandidateDiagnosticClass::CandidateBasisDiagnostic,
        lane: "blue_brain_reasoning_candidate_basis_diagnostic",
        basis_binding:
            "runtime-derived + context-derived + evidence/reference-derived + selection-derived + memory-candidate-derived + commit-feedback-derived + proposal-derived basis references",
        insufficiency_or_caveat_reason: "basis observed with compact canonical source references",
        proposal_boundary_binding: "candidate remains candidate until proposal-ready diagnostic is present",
        selection_deferral_binding:
            "selection may observe candidate basis while preserving selected/deferred/rejected distinction",
        memory_boundary_binding:
            "basis references include commit feedback and memory-candidate signals without commit authority",
        runtime_context_feedback_binding:
            "runtime/context feedback can report candidate basis observed without reasoning-completed claim",
        canonical_guard:
            "basis diagnostics are compact source-maps, not free-form explainability or reasoning output",
    },
    BlueBrainReasoningCandidateDiagnosticLane {
        class: BlueBrainReasoningCandidateDiagnosticClass::SufficientCandidateDiagnostic,
        lane: "blue_brain_reasoning_candidate_sufficient_diagnostic",
        basis_binding:
            "blue_brain_context_derived_reasoning_candidate_sufficient + blue_brain_selection_derived_action_candidate_selected_context",
        insufficiency_or_caveat_reason: "sufficient context/evidence/selection basis for candidate quality",
        proposal_boundary_binding:
            "candidate can become proposal-ready but remains non-executing and non-actioned",
        selection_deferral_binding: "sufficient candidate can be selected without collapsing deferred/rejected states",
        memory_boundary_binding:
            "committed-if-present can strengthen basis only if real commit path exists",
        runtime_context_feedback_binding:
            "runtime can report sufficient candidate basis without claiming completed reasoning",
        canonical_guard:
            "sufficient diagnostics do not imply action executed, memory committed, or policy applied",
    },
    BlueBrainReasoningCandidateDiagnosticLane {
        class: BlueBrainReasoningCandidateDiagnosticClass::PartialCandidateDiagnostic,
        lane: "blue_brain_reasoning_candidate_partial_diagnostic",
        basis_binding:
            "blue_brain_evidence_reference_reasoning_candidate_partial_caveated + blue_brain_feedback_evidence_caveated_partial_or_insufficient",
        insufficiency_or_caveat_reason:
            "candidate is partial due to weak or incomplete evidence/reference basis",
        proposal_boundary_binding:
            "candidate may yield caveated proposal or stay candidate pending stronger basis",
        selection_deferral_binding:
            "partial candidate may be deferred pending stronger evidence/context refresh",
        memory_boundary_binding:
            "partial basis has no memory commit authority and remains non-persistent",
        runtime_context_feedback_binding:
            "runtime/context diagnostics expose partial basis and unresolved caveat posture",
        canonical_guard: "partial diagnostics stay bounded; no planning/ranking/policy engine semantics",
    },
    BlueBrainReasoningCandidateDiagnosticLane {
        class: BlueBrainReasoningCandidateDiagnosticClass::CaveatedCandidateDiagnostic,
        lane: "blue_brain_reasoning_candidate_caveated_diagnostic",
        basis_binding:
            "blue_brain_trigger_caveated_candidate_basis + blue_brain_action_proposal_caveated_state",
        insufficiency_or_caveat_reason:
            "caveated due to partial evidence, selection/attention caveat, or unavailable memory commit",
        proposal_boundary_binding: "candidate yields caveated proposal or remains caveated candidate",
        selection_deferral_binding: "caveated candidate may be deferred and remains explicitly caveated",
        memory_boundary_binding:
            "commit unavailable/deferred keeps caveat unresolved and does not permit commit",
        runtime_context_feedback_binding:
            "runtime feedback keeps caveats explicit and does not relabel caveated as sufficient",
        canonical_guard:
            "caveat diagnostics are canonical candidate quality markers, not audit/explainability output",
    },
    BlueBrainReasoningCandidateDiagnosticLane {
        class: BlueBrainReasoningCandidateDiagnosticClass::StaleCandidateDiagnostic,
        lane: "blue_brain_reasoning_candidate_stale_diagnostic",
        basis_binding:
            "blue_brain_evidence_reference_reasoning_candidate_stale + blue_brain_context_blocked_due_to_stale_basis",
        insufficiency_or_caveat_reason: "insufficient due to stale reference basis requiring recheck",
        proposal_boundary_binding: "stale candidate cannot become proposal-ready without basis refresh",
        selection_deferral_binding: "stale candidate requires recheck and is deferral-gated",
        memory_boundary_binding: "stale basis cannot imply memory commit or commit-readiness",
        runtime_context_feedback_binding:
            "runtime reports stale candidate basis explicitly and keeps it separate from rejected",
        canonical_guard: "stale state remains non-interchangeable with sufficient/partial/deferred/rejected",
    },
    BlueBrainReasoningCandidateDiagnosticLane {
        class: BlueBrainReasoningCandidateDiagnosticClass::InsufficientCandidateDiagnostic,
        lane: "blue_brain_reasoning_candidate_insufficient_diagnostic",
        basis_binding:
            "blue_brain_evidence_observed_no_reasoning_candidate + blue_brain_action_proposal_insufficient_state",
        insufficiency_or_caveat_reason:
            "insufficient due to missing context, weak evidence, or rejected memory-candidate basis",
        proposal_boundary_binding:
            "insufficient candidate cannot become selected proposal or proposal-ready",
        selection_deferral_binding:
            "insufficient candidate cannot be selected and remains unresolved until basis improves",
        memory_boundary_binding:
            "insufficient basis blocks commit progression and keeps no-commit boundary explicit",
        runtime_context_feedback_binding:
            "runtime/context diagnostics mark insufficient candidate with no reasoning-completed claim",
        canonical_guard:
            "insufficient diagnostic does not auto-trigger compute/proposal/action/memory pathways",
    },
    BlueBrainReasoningCandidateDiagnosticLane {
        class: BlueBrainReasoningCandidateDiagnosticClass::DeferredCandidateDiagnostic,
        lane: "blue_brain_reasoning_candidate_deferred_diagnostic",
        basis_binding:
            "blue_brain_selection_derived_action_candidate_deferred + blue_brain_candidate_deferred_before_proposal",
        insufficiency_or_caveat_reason:
            "deferred due to partial/caveated basis or pending context/evidence recheck",
        proposal_boundary_binding: "candidate deferred before proposal and remains non-executing",
        selection_deferral_binding: "deferred candidate remains explicit in BB4 deferral lifecycle",
        memory_boundary_binding: "deferred candidate has no memory commit authority",
        runtime_context_feedback_binding:
            "runtime/context sees deferred candidate state with no action execution claim",
        canonical_guard:
            "deferred diagnostics remain compact operational state, not reasoning completion narrative",
    },
    BlueBrainReasoningCandidateDiagnosticLane {
        class: BlueBrainReasoningCandidateDiagnosticClass::RejectedCandidateDiagnostic,
        lane: "blue_brain_reasoning_candidate_rejected_diagnostic",
        basis_binding:
            "blue_brain_selection_ignored_or_rejected_no_candidate + blue_brain_candidate_rejected_before_proposal",
        insufficiency_or_caveat_reason:
            "rejected due to invalid/blocked candidate basis or rejected memory-candidate feedback",
        proposal_boundary_binding: "candidate rejected before proposal and excluded from current proposal path",
        selection_deferral_binding: "rejected candidate is excluded from current selection window",
        memory_boundary_binding: "rejected memory-candidate feedback weakens or blocks candidate basis",
        runtime_context_feedback_binding:
            "runtime/context diagnostics expose rejection without implying fault recovery or action",
        canonical_guard: "rejected state is explicit and not remapped to deferred/blocked/selected",
    },
    BlueBrainReasoningCandidateDiagnosticLane {
        class: BlueBrainReasoningCandidateDiagnosticClass::ProposalReadyDiagnostic,
        lane: "blue_brain_reasoning_candidate_proposal_ready_diagnostic",
        basis_binding:
            "blue_brain_candidate_yields_non_executing_action_proposal + blue_brain_action_proposal_created",
        insufficiency_or_caveat_reason:
            "candidate is proposal-ready only after sufficient bounded basis and explicit transition",
        proposal_boundary_binding: "proposal-ready means non-executing proposal-ready, not executed action",
        selection_deferral_binding:
            "proposal-ready can be selected/deferred/rejected in proposal lifecycle without auto-execution",
        memory_boundary_binding: "proposal-ready does not commit memory and does not imply commit-eligible",
        runtime_context_feedback_binding:
            "runtime/context can observe proposal-ready candidate while no memory commit occurs",
        canonical_guard:
            "proposal-ready is distinct from action-executed, memory-committed, and reasoning-completed",
    },
    BlueBrainReasoningCandidateDiagnosticLane {
        class: BlueBrainReasoningCandidateDiagnosticClass::NonCanonicalInternalOnlyDiagnostic,
        lane: "blue_brain_reasoning_candidate_non_canonical_internal_only_diagnostic",
        basis_binding:
            "blue_brain_non_canonical_internal_planning_like_path + blue_brain_non_canonical_internal_action_like_path",
        insufficiency_or_caveat_reason:
            "blocked due to non-canonical/internal dependency unless explicitly down-mapped",
        proposal_boundary_binding:
            "internal/expert-only diagnostics cannot act as canonical proposal-ready authority",
        selection_deferral_binding:
            "internal-only diagnostics are excluded from canonical BB4 selection/deferral authority",
        memory_boundary_binding:
            "internal-only diagnostics cannot claim memory commit path, eligibility, or commit result",
        runtime_context_feedback_binding:
            "runtime/context marks diagnostics as canonical=false and keeps them segregated",
        canonical_guard:
            "compute-internal, expert-only, legacy, and unstable dev/test hooks are non-canonical for BB6 diagnostics",
    },
];

pub const CANONICAL_BLUE_BRAIN_CANDIDATE_COMPARISON_MAP: [BlueBrainCandidateComparisonLane; 9] = [
    BlueBrainCandidateComparisonLane {
        class: BlueBrainCandidateComparisonClass::ComparableCandidates,
        lane: "blue_brain_candidate_comparison_comparable_candidates",
        candidate_scope:
            "multiple planning/reasoning candidates are present and explicitly identified for structured comparison",
        runtime_basis_binding:
            "CANONICAL_BLUE_BRAIN_PLANNING_REASONING_CANDIDATE_MAP + CANONICAL_BLUE_BRAIN_REASONING_CANDIDATE_DIAGNOSTICS_MAP",
        context_basis_binding:
            "CANONICAL_BLUE_BRAIN_CONTEXT_MEMORY_SURFACE_MAP + CANONICAL_BLUE_BRAIN_CONTEXT_UPDATE_LIFECYCLE_MAP",
        evidence_reference_basis_binding:
            "CANONICAL_BLUE_BRAIN_REFERENCE_CONTEXT_MAP + CANONICAL_BLUE_BRAIN_CONTEXT_EVIDENCE_PRIORITY_MAP",
        selection_basis_binding:
            "CANONICAL_BLUE_BRAIN_CONTROL_ATTENTION_SELECTION_MAP + CANONICAL_BLUE_BRAIN_CANDIDATE_DEFERRAL_LIFECYCLE_MAP",
        memory_basis_binding:
            "CANONICAL_BLUE_BRAIN_FUTURE_MEMORY_HANDOFF_STATE_MAP + CANONICAL_BLUE_BRAIN_MEMORY_COMMIT_DIAGNOSTICS_MAP",
        proposal_status_basis_binding:
            "CANONICAL_BLUE_BRAIN_CANDIDATE_ACTION_BOUNDARY_MAP + CANONICAL_BLUE_BRAIN_NON_EXECUTING_ACTION_PROPOSAL_STATE_MAP",
        comparison_quality_or_caveat:
            "comparison is candidate-comparison only and not a ranking/selection/planning decision",
        selection_interaction_boundary:
            "compared but not selected is explicit and remains a valid outcome",
        proposal_interaction_boundary: "comparison only; no proposal generated by default",
        runtime_diagnostics_binding:
            "runtime diagnostics list which candidates were compared and keep comparison class explicit",
        canonical_guard:
            "comparison semantics provide structured comparability and diagnosis, not planner/ranking/policy authority",
    },
    BlueBrainCandidateComparisonLane {
        class: BlueBrainCandidateComparisonClass::ComparisonBasisAvailable,
        lane: "blue_brain_candidate_comparison_basis_available",
        candidate_scope: "candidate pair/set has explicit runtime/context/evidence/selection/memory/proposal references",
        runtime_basis_binding:
            "runtime-derived + diagnostics-derived candidate basis references are present",
        context_basis_binding:
            "context freshness/update lifecycle references are present and canonical",
        evidence_reference_basis_binding:
            "evidence/reference anchors are attached and traceable by reference",
        selection_basis_binding:
            "selection/priority/deferral posture is visible but remains separate from comparison result",
        memory_basis_binding:
            "future-memory-ready/rejected/commit-feedback signals are visible as basis modifiers only",
        proposal_status_basis_binding:
            "proposal-ready/deferred/rejected/insufficient status can be compared without execution semantics",
        comparison_quality_or_caveat:
            "comparison basis available does not imply comparison meaningful yet",
        selection_interaction_boundary:
            "comparison basis can inform selection posture, but does not decide selection",
        proposal_interaction_boundary:
            "comparison basis can support proposal caveat, but does not auto-create proposal",
        runtime_diagnostics_binding:
            "runtime diagnostics expose comparison basis references and unresolved caveats",
        canonical_guard:
            "explicit basis map is required; no free speculative prose as canonical comparison authority",
    },
    BlueBrainCandidateComparisonLane {
        class: BlueBrainCandidateComparisonClass::ComparisonMeaningful,
        lane: "blue_brain_candidate_comparison_meaningful_shared_basis",
        candidate_scope: "candidates share sufficiently compatible runtime/context/evidence/selection/memory/proposal basis",
        runtime_basis_binding: "shared runtime window and transition posture are aligned",
        context_basis_binding: "context window and freshness posture are mutually compatible",
        evidence_reference_basis_binding: "evidence/reference basis overlaps enough for meaningful comparison",
        selection_basis_binding: "selection/deferral posture is comparable without collapsing selection outcomes",
        memory_basis_binding:
            "memory-candidate and commit-feedback basis can be compared without implying memory commit",
        proposal_status_basis_binding:
            "proposal status can be contrasted (proposal-ready/deferred/etc.) with no execution implication",
        comparison_quality_or_caveat:
            "meaningful because candidates share comparable basis and caveats are bounded",
        selection_interaction_boundary:
            "comparison informs selection, but does not decide selected/deferred/rejected automatically",
        proposal_interaction_boundary:
            "comparison supports proposal caveat framing while keeping proposal creation explicit",
        runtime_diagnostics_binding:
            "runtime diagnostics record meaningful comparison reason and residual caveats",
        canonical_guard:
            "meaningful comparison is still non-ranking and non-decision semantics",
    },
    BlueBrainCandidateComparisonLane {
        class: BlueBrainCandidateComparisonClass::ComparisonCaveated,
        lane: "blue_brain_candidate_comparison_caveated_basis_mismatch",
        candidate_scope: "candidates are comparable but evidence/reference or memory/proposal basis differs materially",
        runtime_basis_binding: "runtime basis is present with caveated transition/trigger posture",
        context_basis_binding: "context basis is partially aligned and caveated",
        evidence_reference_basis_binding:
            "caveated because evidence/reference differs across compared candidates",
        selection_basis_binding:
            "selection posture may differ (selected/deferred) and remains explicit rather than collapsed",
        memory_basis_binding:
            "commit unavailable or rejected memory basis weakens comparison confidence without commit side effect",
        proposal_status_basis_binding:
            "proposal caveat/deferred/insufficient states remain visible in comparison output",
        comparison_quality_or_caveat:
            "comparison caveated and usable for diagnostics but not for deterministic winner claims",
        selection_interaction_boundary:
            "compared and candidate remains deferred is valid and non-contradictory",
        proposal_interaction_boundary:
            "comparison supports proposal caveat and can be insufficient for proposal",
        runtime_diagnostics_binding:
            "runtime diagnostics preserve caveat reasons and do not relabel caveated as meaningful",
        canonical_guard:
            "caveated comparison must preserve caveats and stay separate from policy/decision outcomes",
    },
    BlueBrainCandidateComparisonLane {
        class: BlueBrainCandidateComparisonClass::ComparisonInconclusive,
        lane: "blue_brain_candidate_comparison_inconclusive_partial_or_stale_basis",
        candidate_scope: "comparison attempted but partial/stale basis prevents conclusive quality class",
        runtime_basis_binding: "runtime basis exists but partial/stale or transition-limited",
        context_basis_binding: "context basis partial/stale prevents stable comparison conclusion",
        evidence_reference_basis_binding:
            "inconclusive due to partial/stale evidence/reference basis",
        selection_basis_binding:
            "selection posture may be present, but basis quality prevents meaningful comparison conclusion",
        memory_basis_binding:
            "commit unavailable/deferred feedback limits comparison confidence",
        proposal_status_basis_binding:
            "proposal-ready status remains possible but comparison remains inconclusive",
        comparison_quality_or_caveat:
            "comparison inconclusive, not failed and not promoted to meaningful/not-meaningful automatically",
        selection_interaction_boundary:
            "no selection decision is implied by inconclusive comparison",
        proposal_interaction_boundary:
            "comparison insufficient for proposal can be emitted explicitly",
        runtime_diagnostics_binding:
            "runtime diagnostics mark inconclusive reasons and required basis refresh",
        canonical_guard:
            "inconclusive is a first-class canonical outcome and not an implicit ranking tie-breaker",
    },
    BlueBrainCandidateComparisonLane {
        class: BlueBrainCandidateComparisonClass::ComparisonNotMeaningful,
        lane: "blue_brain_candidate_comparison_not_meaningful_incompatible_context",
        candidate_scope: "candidate contexts are incompatible, so comparison is not meaningful",
        runtime_basis_binding: "runtime windows are incompatible for a meaningful shared comparison frame",
        context_basis_binding: "not meaningful due to incompatible context basis",
        evidence_reference_basis_binding:
            "evidence/reference basis cannot be aligned sufficiently for canonical comparison meaning",
        selection_basis_binding:
            "selection outcomes are preserved, but cannot be interpreted as comparable quality signal",
        memory_basis_binding:
            "memory basis differences remain informational only and do not rescue incompatibility",
        proposal_status_basis_binding:
            "proposal status remains explicit, but comparison remains not meaningful",
        comparison_quality_or_caveat:
            "comparison not meaningful due to incompatible basis rather than missing data",
        selection_interaction_boundary:
            "comparison not meaningful does not block future selection reassessment",
        proposal_interaction_boundary:
            "no proposal generated from not-meaningful comparison alone",
        runtime_diagnostics_binding:
            "runtime diagnostics explicitly mark not-meaningful and list incompatibility references",
        canonical_guard:
            "not-meaningful comparison must not be remapped to rejection, ranking, or execution authority",
    },
    BlueBrainCandidateComparisonLane {
        class: BlueBrainCandidateComparisonClass::ComparisonBlocked,
        lane: "blue_brain_candidate_comparison_blocked_missing_or_noncanonical_basis",
        candidate_scope:
            "comparison blocked due to missing candidate basis or blocked by non-canonical dependency",
        runtime_basis_binding: "missing runtime candidate basis blocks canonical comparison",
        context_basis_binding: "missing context basis blocks canonical comparison",
        evidence_reference_basis_binding:
            "missing evidence/reference anchors or stale blocked references prevent comparison",
        selection_basis_binding:
            "selection state alone cannot unblock comparison without canonical candidate basis",
        memory_basis_binding:
            "commit unavailable can limit comparison; no memory commit implied by unblocking attempt",
        proposal_status_basis_binding:
            "proposal status alone cannot substitute missing candidate-comparison basis",
        comparison_quality_or_caveat:
            "blocked due to missing basis or blocked due to non-canonical dependency",
        selection_interaction_boundary:
            "blocked comparison cannot produce automatic selected outcome",
        proposal_interaction_boundary:
            "blocked comparison yields no proposal generated and no action executed",
        runtime_diagnostics_binding:
            "runtime diagnostics expose blocked reason and preserve canonical=false when needed",
        canonical_guard:
            "blocked comparison cannot be bypassed through expert/internal hooks",
    },
    BlueBrainCandidateComparisonLane {
        class: BlueBrainCandidateComparisonClass::NonCanonicalInternalOnlyComparison,
        lane: "blue_brain_candidate_comparison_non_canonical_internal_only_path",
        candidate_scope: "internal/expert/legacy/dev-only comparison path without canonical down-map",
        runtime_basis_binding:
            "run_operation_with_entry/replay_with_entry/build_backend(kind=stub|candle|worker)/domains/ai*",
        context_basis_binding:
            "internal compatibility context views are non-canonical for BB6 comparison authority",
        evidence_reference_basis_binding:
            "expert/internal-only evidence views are non-canonical unless down-mapped",
        selection_basis_binding:
            "internal selection hooks cannot define canonical compared-but-not-selected semantics",
        memory_basis_binding:
            "internal commit-like diagnostics cannot define canonical comparison-memory quality",
        proposal_status_basis_binding:
            "internal action-like states cannot define canonical proposal-status comparison",
        comparison_quality_or_caveat:
            "non-canonical/internal-only comparison excluded from canonical candidate-comparison map",
        selection_interaction_boundary:
            "internal-only comparison cannot decide canonical selection or deferral status",
        proposal_interaction_boundary:
            "internal-only comparison cannot generate canonical proposal nor action execution",
        runtime_diagnostics_binding:
            "runtime/context mark path as canonical=false and keep it segregated",
        canonical_guard:
            "compute-internal details, expert hooks, legacy compat objects, and unstable dev/test surfaces are non-canonical",
    },
    BlueBrainCandidateComparisonLane {
        class: BlueBrainCandidateComparisonClass::ComparisonMeaningful,
        lane: "blue_brain_candidate_comparison_informs_selection_and_proposal_without_deciding",
        candidate_scope: "meaningful/caveated comparison can inform downstream selection/proposal interpretation",
        runtime_basis_binding: "comparison class feeds runtime diagnostics as informational context only",
        context_basis_binding: "context caveats remain attached when comparison informs later transitions",
        evidence_reference_basis_binding:
            "evidence/reference caveats are preserved when comparison informs proposal caveat language",
        selection_basis_binding:
            "comparison informs selection, but compared candidate may remain deferred or not selected",
        memory_basis_binding:
            "future-memory-ready support may strengthen basis; rejected/commit-unavailable remains limiting",
        proposal_status_basis_binding:
            "compared and candidate remains proposal-ready is explicit and still non-executing",
        comparison_quality_or_caveat:
            "comparison affects diagnostics and posture only; no decision/execution/reasoning-completed claim",
        selection_interaction_boundary:
            "compared and candidate remains deferred / compared but not selected are canonical outcomes",
        proposal_interaction_boundary:
            "comparison supports proposal caveat or insufficiency; no automatic proposal generation",
        runtime_diagnostics_binding:
            "runtime diagnostics show comparison impact on selection/proposal state without decision claim",
        canonical_guard:
            "comparison layer is bounded integration glue, not a planner, policy engine, or execution orchestrator",
    },
];

pub const CANONICAL_BLUE_BRAIN_MINIMAL_PLANNING_ACTION_INTERFACE_MAP:
    [BlueBrainMinimalPlanningActionInterfaceLane; 16] = [
    BlueBrainMinimalPlanningActionInterfaceLane {
        class: BlueBrainMinimalPlanningActionInterfaceClass::DiagnosticOnlyProposal,
        lane: "blue_brain_proposal_diagnostic_only",
        readiness_semantics:
            "diagnostic-only proposal remains candidate/proposal diagnostics without plan-ready or action-ready claim",
        proposal_basis_binding:
            "CANONICAL_BLUE_BRAIN_NON_EXECUTING_ACTION_PROPOSAL_STATE_MAP::ProposalCreated + ProposalCaveated",
        diagnostics_comparison_binding:
            "CANONICAL_BLUE_BRAIN_REASONING_CANDIDATE_DIAGNOSTICS_MAP::CandidateBasisDiagnostic|PartialCandidateDiagnostic|CaveatedCandidateDiagnostic + CANONICAL_BLUE_BRAIN_CANDIDATE_COMPARISON_MAP::ComparisonCaveated|ComparisonInconclusive",
        context_evidence_selection_binding:
            "context/evidence/selection basis attached but not sufficient for readiness promotion",
        memory_commit_feedback_binding:
            "memory-candidate and commit feedback are diagnostics-only and remain non-persistent",
        execution_boundary: "diagnostic-only: no action execution performed",
        plan_boundary: "diagnostic-only: no plan generated",
        compute_invocation_boundary: "diagnostic-only: no compute invocation performed",
        tool_invocation_boundary: "diagnostic-only: no tool invocation performed",
        memory_commit_boundary: "diagnostic-only: no memory commit performed",
        canonical_guard:
            "diagnostic-only remains canonical as bounded observability and is not an execution proxy",
    },
    BlueBrainMinimalPlanningActionInterfaceLane {
        class: BlueBrainMinimalPlanningActionInterfaceClass::PlanReadyProposal,
        lane: "blue_brain_proposal_plan_ready_no_plan_generated",
        readiness_semantics:
            "plan-ready proposal means prepared for future planning boundary only; no plan exists yet",
        proposal_basis_binding:
            "CANONICAL_BLUE_BRAIN_CANDIDATE_TO_PROPOSAL_TRANSITION_MAP::CandidateYieldsActionProposal + CANONICAL_BLUE_BRAIN_NON_EXECUTING_ACTION_PROPOSAL_STATE_MAP::ProposalCreated",
        diagnostics_comparison_binding:
            "CANONICAL_BLUE_BRAIN_REASONING_CANDIDATE_DIAGNOSTICS_MAP::ProposalReadyDiagnostic + CANONICAL_BLUE_BRAIN_CANDIDATE_COMPARISON_MAP::ComparisonBasisAvailable|ComparisonMeaningful",
        context_evidence_selection_binding:
            "context/evidence/selection basis is sufficiently attached for future plan processing",
        memory_commit_feedback_binding:
            "commit feedback can strengthen or caveat readiness but does not create plan execution semantics",
        execution_boundary: "plan-ready: no action execution performed",
        plan_boundary: "plan-ready but no plan generated or executed",
        compute_invocation_boundary: "plan-ready: no compute invocation performed",
        tool_invocation_boundary: "plan-ready: no tool invocation performed",
        memory_commit_boundary: "plan-ready: no memory commit performed",
        canonical_guard: "plan-ready != plan-generated != plan-executed",
    },
    BlueBrainMinimalPlanningActionInterfaceLane {
        class: BlueBrainMinimalPlanningActionInterfaceClass::PlanReadyProposal,
        lane: "blue_brain_proposal_plan_ready_with_caveat",
        readiness_semantics:
            "plan-ready can remain caveated when basis is usable but not fully resolved",
        proposal_basis_binding:
            "CANONICAL_BLUE_BRAIN_NON_EXECUTING_ACTION_PROPOSAL_STATE_MAP::ProposalCaveated + ProposalCreated",
        diagnostics_comparison_binding:
            "CANONICAL_BLUE_BRAIN_REASONING_CANDIDATE_DIAGNOSTICS_MAP::CaveatedCandidateDiagnostic|ProposalReadyDiagnostic + CANONICAL_BLUE_BRAIN_CANDIDATE_COMPARISON_MAP::ComparisonCaveated",
        context_evidence_selection_binding:
            "context/evidence/selection basis is attached with explicit caveat markers",
        memory_commit_feedback_binding:
            "commit unavailable/deferred caveat can be carried as plan-ready caveat",
        execution_boundary: "plan-ready caveated: no action execution performed",
        plan_boundary: "plan-ready caveated: no plan generated",
        compute_invocation_boundary: "plan-ready caveated: no compute invocation performed",
        tool_invocation_boundary: "plan-ready caveated: no tool invocation performed",
        memory_commit_boundary: "plan-ready caveated: no memory commit performed",
        canonical_guard: "caveated plan-ready state is explicit and remains non-executing",
    },
    BlueBrainMinimalPlanningActionInterfaceLane {
        class: BlueBrainMinimalPlanningActionInterfaceClass::PlanReadyProposal,
        lane: "blue_brain_proposal_plan_ready_deferred",
        readiness_semantics:
            "plan-ready proposal can be deferred while preserving readiness basis for later planning window",
        proposal_basis_binding:
            "CANONICAL_BLUE_BRAIN_NON_EXECUTING_ACTION_PROPOSAL_STATE_MAP::ProposalDeferred",
        diagnostics_comparison_binding:
            "CANONICAL_BLUE_BRAIN_REASONING_CANDIDATE_DIAGNOSTICS_MAP::DeferredCandidateDiagnostic + CANONICAL_BLUE_BRAIN_CANDIDATE_COMPARISON_MAP::ComparisonInconclusive",
        context_evidence_selection_binding:
            "selection/deferral posture is explicit and references deferred context/evidence basis",
        memory_commit_feedback_binding:
            "deferred readiness can carry memory-candidate caveats without commit authority",
        execution_boundary: "plan-ready deferred: no action execution performed",
        plan_boundary: "plan-ready deferred: no plan generated",
        compute_invocation_boundary: "plan-ready deferred: no compute invocation performed",
        tool_invocation_boundary: "plan-ready deferred: no tool invocation performed",
        memory_commit_boundary: "plan-ready deferred: no memory commit performed",
        canonical_guard: "deferred readiness remains explicit and distinct from rejected/blocked",
    },
    BlueBrainMinimalPlanningActionInterfaceLane {
        class: BlueBrainMinimalPlanningActionInterfaceClass::PlanReadyProposal,
        lane: "blue_brain_proposal_plan_ready_blocked_insufficient_basis",
        readiness_semantics:
            "plan-ready can be blocked when candidate comparison or basis quality is insufficient",
        proposal_basis_binding:
            "CANONICAL_BLUE_BRAIN_NON_EXECUTING_ACTION_PROPOSAL_STATE_MAP::ProposalBlocked|ProposalInsufficientBasis",
        diagnostics_comparison_binding:
            "CANONICAL_BLUE_BRAIN_REASONING_CANDIDATE_DIAGNOSTICS_MAP::InsufficientCandidateDiagnostic + CANONICAL_BLUE_BRAIN_CANDIDATE_COMPARISON_MAP::ComparisonInconclusive|ComparisonBlocked",
        context_evidence_selection_binding:
            "blocked/insufficient context/evidence/selection basis is preserved as explicit blocker",
        memory_commit_feedback_binding:
            "insufficient or rejected memory-candidate feedback can block plan-readiness progression",
        execution_boundary: "plan-ready blocked: no action execution performed",
        plan_boundary: "plan-ready blocked: no plan generated",
        compute_invocation_boundary: "plan-ready blocked: no compute invocation performed",
        tool_invocation_boundary: "plan-ready blocked: no tool invocation performed",
        memory_commit_boundary: "plan-ready blocked: no memory commit performed",
        canonical_guard: "blocked due to insufficient basis is canonical and not auto-promoted",
    },
    BlueBrainMinimalPlanningActionInterfaceLane {
        class: BlueBrainMinimalPlanningActionInterfaceClass::ActionReadyProposal,
        lane: "blue_brain_proposal_action_ready_not_executed",
        readiness_semantics:
            "action-ready proposal means boundary-ready for future action subsystem only and remains non-executed",
        proposal_basis_binding:
            "CANONICAL_BLUE_BRAIN_NON_EXECUTING_ACTION_PROPOSAL_STATE_MAP::ProposalSelectedForPossibleFutureAction + ProposalCreated",
        diagnostics_comparison_binding:
            "CANONICAL_BLUE_BRAIN_REASONING_CANDIDATE_DIAGNOSTICS_MAP::ProposalReadyDiagnostic + CANONICAL_BLUE_BRAIN_CANDIDATE_COMPARISON_MAP::ComparisonMeaningful",
        context_evidence_selection_binding:
            "context/evidence/selection basis is sufficient and selection posture marks possible future action",
        memory_commit_feedback_binding:
            "memory-candidate/commit feedback may inform confidence and caveats only",
        execution_boundary: "action-ready but not executed",
        plan_boundary: "action-ready does not imply plan generated",
        compute_invocation_boundary: "action-ready: no compute invocation performed",
        tool_invocation_boundary: "action-ready: no tool invocation performed",
        memory_commit_boundary: "action-ready: no memory commit performed",
        canonical_guard: "action-ready != executed action",
    },
    BlueBrainMinimalPlanningActionInterfaceLane {
        class: BlueBrainMinimalPlanningActionInterfaceClass::ActionReadyProposal,
        lane: "blue_brain_proposal_action_ready_with_caveat",
        readiness_semantics:
            "action-ready proposal can be emitted with caveats and remains non-executing",
        proposal_basis_binding:
            "CANONICAL_BLUE_BRAIN_NON_EXECUTING_ACTION_PROPOSAL_STATE_MAP::ProposalSelectedForPossibleFutureAction + ProposalCaveated",
        diagnostics_comparison_binding:
            "CANONICAL_BLUE_BRAIN_REASONING_CANDIDATE_DIAGNOSTICS_MAP::CaveatedCandidateDiagnostic|ProposalReadyDiagnostic + CANONICAL_BLUE_BRAIN_CANDIDATE_COMPARISON_MAP::ComparisonCaveated",
        context_evidence_selection_binding:
            "action basis includes explicit context/evidence/selection caveats",
        memory_commit_feedback_binding:
            "commit unavailable/deferred can remain a non-blocking caveat if bounded for future action",
        execution_boundary: "action-ready with caveat but not executed",
        plan_boundary: "action-ready with caveat does not imply plan generated",
        compute_invocation_boundary: "action-ready with caveat: no compute invocation performed",
        tool_invocation_boundary: "action-ready with caveat: no tool invocation performed",
        memory_commit_boundary: "action-ready with caveat: no memory commit performed",
        canonical_guard: "caveated action-ready remains explicit and non-executing",
    },
    BlueBrainMinimalPlanningActionInterfaceLane {
        class: BlueBrainMinimalPlanningActionInterfaceClass::ActionReadyProposal,
        lane: "blue_brain_proposal_action_ready_blocked_missing_boundary",
        readiness_semantics:
            "action-ready proposal may be blocked by missing explicit action boundary or subsystem handoff",
        proposal_basis_binding:
            "CANONICAL_BLUE_BRAIN_CANDIDATE_ACTION_BOUNDARY_MAP::SelectedProposal + BlockedProposal",
        diagnostics_comparison_binding:
            "CANONICAL_BLUE_BRAIN_REASONING_CANDIDATE_DIAGNOSTICS_MAP::ProposalReadyDiagnostic|InsufficientCandidateDiagnostic + CANONICAL_BLUE_BRAIN_CANDIDATE_COMPARISON_MAP::ComparisonBlocked",
        context_evidence_selection_binding:
            "selection may mark action-ready posture while boundary availability remains blocked",
        memory_commit_feedback_binding:
            "memory feedback can remain attached while action boundary is unavailable",
        execution_boundary: "action-ready blocked by missing boundary: no action execution performed",
        plan_boundary: "action-ready blocked: no plan generated",
        compute_invocation_boundary:
            "action-ready blocked by missing boundary: no compute invocation performed",
        tool_invocation_boundary:
            "action-ready blocked by missing boundary: no tool invocation performed",
        memory_commit_boundary: "action-ready blocked: no memory commit performed",
        canonical_guard: "missing action boundary blocks promotion to executed action",
    },
    BlueBrainMinimalPlanningActionInterfaceLane {
        class: BlueBrainMinimalPlanningActionInterfaceClass::ActionReadyProposal,
        lane: "blue_brain_proposal_action_ready_requires_future_subsystem",
        readiness_semantics:
            "action-ready proposal can explicitly require a future action subsystem and stays non-executing today",
        proposal_basis_binding:
            "CANONICAL_BLUE_BRAIN_CANDIDATE_ACTION_BOUNDARY_MAP::SelectedProposal + CANONICAL_BLUE_BRAIN_NON_EXECUTING_ACTION_PROPOSAL_STATE_MAP::NoExecutionPerformed",
        diagnostics_comparison_binding:
            "CANONICAL_BLUE_BRAIN_REASONING_CANDIDATE_DIAGNOSTICS_MAP::ProposalReadyDiagnostic + CANONICAL_BLUE_BRAIN_CANDIDATE_COMPARISON_MAP::ComparisonMeaningful|ComparisonCaveated",
        context_evidence_selection_binding:
            "basis remains attached and can be handed off to future action subsystem without mutation",
        memory_commit_feedback_binding:
            "future subsystem requirement does not alter memory commit boundaries",
        execution_boundary: "action-ready requires future subsystem: no action execution performed",
        plan_boundary: "action-ready requires future subsystem: no plan generated",
        compute_invocation_boundary:
            "action-ready requires future subsystem: no compute invocation performed",
        tool_invocation_boundary:
            "action-ready requires future subsystem: no tool invocation performed",
        memory_commit_boundary:
            "action-ready requires future subsystem: no memory commit performed",
        canonical_guard:
            "future-action-ready marker is preparatory only and does not create execution semantics",
    },
    BlueBrainMinimalPlanningActionInterfaceLane {
        class: BlueBrainMinimalPlanningActionInterfaceClass::DeferredProposal,
        lane: "blue_brain_proposal_readiness_deferred",
        readiness_semantics: "proposal readiness deferred pending basis refresh",
        proposal_basis_binding:
            "CANONICAL_BLUE_BRAIN_NON_EXECUTING_ACTION_PROPOSAL_STATE_MAP::ProposalDeferred",
        diagnostics_comparison_binding:
            "CANONICAL_BLUE_BRAIN_REASONING_CANDIDATE_DIAGNOSTICS_MAP::DeferredCandidateDiagnostic + CANONICAL_BLUE_BRAIN_CANDIDATE_COMPARISON_MAP::ComparisonInconclusive",
        context_evidence_selection_binding:
            "deferral references context/evidence freshness and selection posture",
        memory_commit_feedback_binding:
            "deferred proposal carries memory-candidate/commit caveats as references only",
        execution_boundary: "deferred proposal: no action execution performed",
        plan_boundary: "deferred proposal: no plan generated",
        compute_invocation_boundary: "deferred proposal: no compute invocation performed",
        tool_invocation_boundary: "deferred proposal: no tool invocation performed",
        memory_commit_boundary: "deferred proposal: no memory commit performed",
        canonical_guard: "deferred readiness is canonical and non-executing",
    },
    BlueBrainMinimalPlanningActionInterfaceLane {
        class: BlueBrainMinimalPlanningActionInterfaceClass::BlockedProposal,
        lane: "blue_brain_proposal_readiness_blocked",
        readiness_semantics: "proposal readiness blocked by insufficient or non-canonical basis",
        proposal_basis_binding:
            "CANONICAL_BLUE_BRAIN_NON_EXECUTING_ACTION_PROPOSAL_STATE_MAP::ProposalBlocked|ProposalInsufficientBasis",
        diagnostics_comparison_binding:
            "CANONICAL_BLUE_BRAIN_REASONING_CANDIDATE_DIAGNOSTICS_MAP::InsufficientCandidateDiagnostic|NonCanonicalInternalOnlyDiagnostic + CANONICAL_BLUE_BRAIN_CANDIDATE_COMPARISON_MAP::ComparisonBlocked|NonCanonicalInternalOnlyComparison",
        context_evidence_selection_binding:
            "blocked basis references explicit context/evidence/selection blockers",
        memory_commit_feedback_binding: "blocked basis has no memory-commit progression authority",
        execution_boundary: "blocked proposal: no action execution performed",
        plan_boundary: "blocked proposal: no plan generated",
        compute_invocation_boundary: "blocked proposal: no compute invocation performed",
        tool_invocation_boundary: "blocked proposal: no tool invocation performed",
        memory_commit_boundary: "blocked proposal: no memory commit performed",
        canonical_guard: "blocked remains separate from rejected and caveated",
    },
    BlueBrainMinimalPlanningActionInterfaceLane {
        class: BlueBrainMinimalPlanningActionInterfaceClass::RejectedProposal,
        lane: "blue_brain_proposal_readiness_rejected",
        readiness_semantics: "proposal rejected in current readiness window",
        proposal_basis_binding:
            "CANONICAL_BLUE_BRAIN_NON_EXECUTING_ACTION_PROPOSAL_STATE_MAP::ProposalRejected",
        diagnostics_comparison_binding:
            "CANONICAL_BLUE_BRAIN_REASONING_CANDIDATE_DIAGNOSTICS_MAP::RejectedCandidateDiagnostic",
        context_evidence_selection_binding:
            "rejected basis captures invalid/insufficient context/evidence/selection basis",
        memory_commit_feedback_binding:
            "rejected memory-candidate feedback can contribute to rejection reason only",
        execution_boundary: "rejected proposal: no action execution performed",
        plan_boundary: "rejected proposal: no plan generated",
        compute_invocation_boundary: "rejected proposal: no compute invocation performed",
        tool_invocation_boundary: "rejected proposal: no tool invocation performed",
        memory_commit_boundary: "rejected proposal: no memory commit performed",
        canonical_guard: "rejected readiness is explicit and non-interchangeable with blocked/deferred",
    },
    BlueBrainMinimalPlanningActionInterfaceLane {
        class: BlueBrainMinimalPlanningActionInterfaceClass::CaveatedProposal,
        lane: "blue_brain_proposal_readiness_caveated",
        readiness_semantics: "proposal readiness remains caveated with bounded caveat references",
        proposal_basis_binding:
            "CANONICAL_BLUE_BRAIN_NON_EXECUTING_ACTION_PROPOSAL_STATE_MAP::ProposalCaveated",
        diagnostics_comparison_binding:
            "CANONICAL_BLUE_BRAIN_REASONING_CANDIDATE_DIAGNOSTICS_MAP::CaveatedCandidateDiagnostic + CANONICAL_BLUE_BRAIN_CANDIDATE_COMPARISON_MAP::ComparisonCaveated",
        context_evidence_selection_binding:
            "caveated context/evidence/selection basis remains explicit",
        memory_commit_feedback_binding:
            "caveated commit-feedback can limit readiness while keeping non-commit baseline",
        execution_boundary: "caveated proposal: no action execution performed",
        plan_boundary: "caveated proposal: no plan generated",
        compute_invocation_boundary: "caveated proposal: no compute invocation performed",
        tool_invocation_boundary: "caveated proposal: no tool invocation performed",
        memory_commit_boundary: "caveated proposal: no memory commit performed",
        canonical_guard: "caveated readiness does not collapse into sufficient/blocked automatically",
    },
    BlueBrainMinimalPlanningActionInterfaceLane {
        class: BlueBrainMinimalPlanningActionInterfaceClass::InsufficientProposalBasis,
        lane: "blue_brain_proposal_readiness_insufficient_basis",
        readiness_semantics: "insufficient candidate basis blocks readiness promotion",
        proposal_basis_binding:
            "CANONICAL_BLUE_BRAIN_NON_EXECUTING_ACTION_PROPOSAL_STATE_MAP::ProposalInsufficientBasis",
        diagnostics_comparison_binding:
            "CANONICAL_BLUE_BRAIN_REASONING_CANDIDATE_DIAGNOSTICS_MAP::InsufficientCandidateDiagnostic + CANONICAL_BLUE_BRAIN_CANDIDATE_COMPARISON_MAP::ComparisonBlocked|ComparisonInconclusive",
        context_evidence_selection_binding:
            "insufficient context/evidence/selection basis remains canonical blocker",
        memory_commit_feedback_binding:
            "insufficient basis preserves no-commit boundary and cannot become commit-authoritative",
        execution_boundary: "insufficient basis: no action execution performed",
        plan_boundary: "insufficient basis: no plan generated",
        compute_invocation_boundary: "insufficient basis: no compute invocation performed",
        tool_invocation_boundary: "insufficient basis: no tool invocation performed",
        memory_commit_boundary: "insufficient basis: no memory commit performed",
        canonical_guard: "insufficient basis cannot be auto-promoted to plan-ready or action-ready",
    },
    BlueBrainMinimalPlanningActionInterfaceLane {
        class: BlueBrainMinimalPlanningActionInterfaceClass::ExecutedActionCanonicalIfPresent,
        lane: "blue_brain_executed_action_canonical_if_explicitly_invoked",
        readiness_semantics:
            "executed action is recognized only where canonical compute execution is explicitly invoked",
        proposal_basis_binding:
            "CANONICAL_BLUE_BRAIN_CANDIDATE_ACTION_BOUNDARY_MAP::ExecutedActionCanonicalIfPresent",
        diagnostics_comparison_binding:
            "readiness diagnostics can hand off context, but do not perform execution",
        context_evidence_selection_binding:
            "execution handoff must remain explicit and external to readiness classification",
        memory_commit_feedback_binding:
            "execution does not imply memory commit; commit boundary remains separate",
        execution_boundary:
            "executed action requires explicit CanonicalComputeEntryPoint::submit invocation",
        plan_boundary: "executed action recognition does not imply generated plan",
        compute_invocation_boundary: "compute invocation only via explicit canonical call path",
        tool_invocation_boundary: "tool invocation remains explicit and outside readiness map",
        memory_commit_boundary: "executed action does not auto-commit memory",
        canonical_guard:
            "execution semantics are out of minimal readiness scope except explicit canonical handoff",
    },
    BlueBrainMinimalPlanningActionInterfaceLane {
        class: BlueBrainMinimalPlanningActionInterfaceClass::NonCanonicalInternalOnlyActionPath,
        lane: "blue_brain_non_canonical_planning_action_path",
        readiness_semantics:
            "internal/expert/legacy planning-action-like path is non-canonical and blocked for readiness authority",
        proposal_basis_binding:
            "run_operation_with_entry/replay_with_entry/build_backend(kind=stub|candle|worker)/domains/ai*",
        diagnostics_comparison_binding:
            "non-canonical/internal diagnostics and comparison lanes stay excluded from canonical readiness authority",
        context_evidence_selection_binding:
            "missing canonical context/evidence/selection bindings block canonical readiness",
        memory_commit_feedback_binding:
            "internal-only path has no canonical memory-commit feedback authority",
        execution_boundary: "non-canonical path: no canonical action execution authority",
        plan_boundary: "non-canonical path: no canonical plan generation authority",
        compute_invocation_boundary: "non-canonical path: no canonical compute invocation authority",
        tool_invocation_boundary: "non-canonical path: no canonical tool invocation authority",
        memory_commit_boundary: "non-canonical path: no canonical memory commit authority",
        canonical_guard:
            "non-canonical planning/action path must be down-mapped before any readiness claim",
    },
];

pub const CANONICAL_BLUE_BRAIN_PLAN_ACTION_READINESS_DIAGNOSTICS_MAP:
    [BlueBrainPlanActionReadinessDiagnosticLane; 9] = [
    BlueBrainPlanActionReadinessDiagnosticLane {
        class: BlueBrainPlanActionReadinessDiagnosticClass::PlanReadyDiagnostic,
        lane: "blue_brain_plan_readiness_diagnostic_canonical",
        readiness_reason: "ready due to sufficient candidate basis",
        proposal_boundary_feedback:
            "candidate becomes plan-ready proposal while remaining non-executing and non-planner",
        selection_deferral_feedback:
            "plan-ready can remain selected-for-future-boundary or deferred by current window",
        context_evidence_memory_feedback:
            "ready due to sufficient context/evidence + commit feedback remains reference-only",
        runtime_feedback:
            "runtime marks plan-ready observed with no action execution/tool invocation/compute invocation/memory commit",
        blocked_action_feedback:
            "not blocked: proposal can progress to future planning boundary without execution claim",
        execution_tool_policy_boundary:
            "readiness diagnostic only, not planner output, not tool result, not policy decision",
        canonical_guard:
            "plan-ready diagnostics stay canonical only when candidate/context/evidence/selection references are present",
    },
    BlueBrainPlanActionReadinessDiagnosticLane {
        class: BlueBrainPlanActionReadinessDiagnosticClass::ActionReadyDiagnostic,
        lane: "blue_brain_action_readiness_diagnostic_canonical",
        readiness_reason: "ready due to selection/attention state",
        proposal_boundary_feedback:
            "candidate becomes action-ready proposal for future boundary only and remains non-executed",
        selection_deferral_feedback:
            "action-ready proposal can be selected for future boundary without auto execution",
        context_evidence_memory_feedback:
            "ready due to sufficient candidate/context/evidence with selection-attention support",
        runtime_feedback:
            "runtime marks action-ready observed and explicitly preserves no action/tool/compute/memory side effect",
        blocked_action_feedback:
            "not blocked unless boundary/readiness basis drops below canonical threshold",
        execution_tool_policy_boundary:
            "action-ready diagnostic is not action execution result and not tool invocation status",
        canonical_guard:
            "action-ready != executed; canonical only as bounded readiness state",
    },
    BlueBrainPlanActionReadinessDiagnosticLane {
        class: BlueBrainPlanActionReadinessDiagnosticClass::DiagnosticOnlyProposalDiagnostic,
        lane: "blue_brain_diagnostic_only_proposal_readiness_diagnostic",
        readiness_reason: "diagnostic-only due to basis still below readiness promotion threshold",
        proposal_boundary_feedback:
            "candidate remains candidate/proposal diagnostic and does not become plan-ready/action-ready",
        selection_deferral_feedback:
            "diagnostic-only stays visible to selection/deferral without creating selected-action posture",
        context_evidence_memory_feedback:
            "context/evidence/memory caveats are attached as diagnostics and remain non-committing",
        runtime_feedback:
            "runtime marks diagnostic-only observed with no execution/tool/compute/memory commit",
        blocked_action_feedback:
            "blocked-action feedback may appear if missing boundary or insufficient readiness basis is explicit",
        execution_tool_policy_boundary:
            "diagnostic-only feedback does not imply planner rejection, policy gate, or tool failure",
        canonical_guard:
            "diagnostic-only is canonical and not an implicit execution request",
    },
    BlueBrainPlanActionReadinessDiagnosticLane {
        class: BlueBrainPlanActionReadinessDiagnosticClass::DeferredReadinessDiagnostic,
        lane: "blue_brain_readiness_deferred_partial_evidence",
        readiness_reason: "deferred due to partial evidence",
        proposal_boundary_feedback:
            "proposal deferred while preserving candidate/proposal boundary and future readiness reevaluation",
        selection_deferral_feedback:
            "caveated proposal remains deferred and can re-enter selection when evidence improves",
        context_evidence_memory_feedback:
            "deferred due to partial context/evidence freshness; memory feedback remains non-commit",
        runtime_feedback:
            "runtime emits deferred readiness with explicit no execution/tool/compute/memory commit",
        blocked_action_feedback:
            "deferred is not blocked failure and not action execution failure",
        execution_tool_policy_boundary:
            "deferral diagnostic is readiness posture only, not policy or planner arbitration output",
        canonical_guard:
            "deferred readiness remains distinct from blocked/rejected/insufficient",
    },
    BlueBrainPlanActionReadinessDiagnosticLane {
        class: BlueBrainPlanActionReadinessDiagnosticClass::BlockedReadinessDiagnostic,
        lane: "blue_brain_readiness_blocked_stale_or_boundary_gap",
        readiness_reason:
            "blocked due to stale context | blocked due to insufficient candidate basis | blocked due to missing action boundary",
        proposal_boundary_feedback:
            "proposal stays blocked and cannot transition to action-ready or future-action-ready until basis is repaired",
        selection_deferral_feedback:
            "blocked proposal may require stronger context/evidence before selection can consider it again",
        context_evidence_memory_feedback:
            "blocked keeps stale context/candidate insufficiency/action-boundary gap explicit and no memory commit implied",
        runtime_feedback:
            "runtime emits blocked readiness diagnostics with explicit no action/tool/compute/memory side effects",
        blocked_action_feedback:
            "blocked-action feedback means readiness transition could not occur; it never means tool executed, action failed, policy denied, or planner denied",
        execution_tool_policy_boundary:
            "blocked readiness is not execution result and not governance decision channel",
        canonical_guard:
            "blocked-action feedback is canonical only as readiness/boundary signal",
    },
    BlueBrainPlanActionReadinessDiagnosticLane {
        class: BlueBrainPlanActionReadinessDiagnosticClass::RejectedReadinessDiagnostic,
        lane: "blue_brain_readiness_rejected_candidate_or_proposal",
        readiness_reason: "rejected due to candidate/proposal rejection",
        proposal_boundary_feedback:
            "proposal rejected in current window and excluded from current readiness promotion",
        selection_deferral_feedback:
            "rejected proposal excluded from current selection and does not produce action-ready posture",
        context_evidence_memory_feedback:
            "rejection references candidate/context/evidence diagnostics and optional commit-feedback caveats",
        runtime_feedback:
            "runtime emits rejected readiness with no action execution/tool invocation/compute invocation/memory commit",
        blocked_action_feedback:
            "rejected is distinct from blocked-action feedback and should not be remapped to execution failure",
        execution_tool_policy_boundary:
            "rejection diagnostic is proposal-boundary feedback only, not planner execution or policy verdict",
        canonical_guard:
            "rejected remains distinct from deferred, blocked, caveated, and insufficient",
    },
    BlueBrainPlanActionReadinessDiagnosticLane {
        class: BlueBrainPlanActionReadinessDiagnosticClass::CaveatedReadinessDiagnostic,
        lane: "blue_brain_readiness_caveated_commit_unavailable",
        readiness_reason: "caveated due to memory/commit unavailability",
        proposal_boundary_feedback:
            "proposal can remain plan-ready/action-ready with explicit caveat and no execution claim",
        selection_deferral_feedback:
            "caveated proposal remains deferred when caveat risk exceeds current selection confidence",
        context_evidence_memory_feedback:
            "caveat binds candidate comparison caveats plus memory candidate/commit-unavailable feedback",
        runtime_feedback:
            "runtime emits caveated readiness with explicit no tool/compute/action execution/memory commit side effects",
        blocked_action_feedback:
            "caveated does not mean blocked-action by default; it can still hand off to future readiness windows",
        execution_tool_policy_boundary:
            "caveat feedback is diagnostic metadata, not policy gate or execution outcome",
        canonical_guard:
            "caveated readiness must stay compact and canonical, not free-form speculative prose",
    },
    BlueBrainPlanActionReadinessDiagnosticLane {
        class: BlueBrainPlanActionReadinessDiagnosticClass::InsufficientReadinessDiagnostic,
        lane: "blue_brain_readiness_insufficient_basis",
        readiness_reason: "blocked due to insufficient candidate basis",
        proposal_boundary_feedback:
            "insufficient proposal cannot become selected/action-ready and remains below readiness boundary",
        selection_deferral_feedback:
            "insufficient proposal cannot become selected/action-ready until candidate/context/evidence basis improves",
        context_evidence_memory_feedback:
            "insufficient keeps context/evidence/reference and candidate-comparison caveats visible",
        runtime_feedback:
            "runtime emits insufficient readiness with strict no execution/tool/compute/memory side effects",
        blocked_action_feedback:
            "insufficient may surface as blocked-action feedback when action-ready transition is requested",
        execution_tool_policy_boundary:
            "insufficient diagnostic is not tool-result/error, not policy decision, and not planner denial",
        canonical_guard:
            "insufficient basis remains canonical blocker and cannot be auto-promoted",
    },
    BlueBrainPlanActionReadinessDiagnosticLane {
        class: BlueBrainPlanActionReadinessDiagnosticClass::NonCanonicalInternalOnlyReadinessDiagnostic,
        lane: "blue_brain_readiness_non_canonical_internal_only",
        readiness_reason:
            "non-canonical/internal-only readiness diagnostic (compute-internal/expert/legacy/dev helper path)",
        proposal_boundary_feedback:
            "internal-only path cannot author canonical candidate->proposal readiness transition without down-map",
        selection_deferral_feedback:
            "internal-only readiness cannot define canonical selected/deferred/rejected state",
        context_evidence_memory_feedback:
            "internal-only evidence/context/commit hooks remain non-canonical until mapped to canonical references",
        runtime_feedback:
            "runtime marks canonical=false and keeps internal-only readiness separate from canonical diagnostics",
        blocked_action_feedback:
            "internal-only blocked-like signals are not canonical blocked-action feedback",
        execution_tool_policy_boundary:
            "non-canonical diagnostics must not be exposed as canonical readiness/tool/execution/policy output",
        canonical_guard:
            "compute-internal details, expert hooks, legacy compat objects, unstable dev/test surfaces are excluded",
    },
];

pub const CANONICAL_BLUE_BRAIN_FUTURE_ACTION_HANDOFF_MAP: [BlueBrainFutureActionHandoffLane; 9] = [
    BlueBrainFutureActionHandoffLane {
        class: BlueBrainFutureActionHandoffClass::FutureActionReady,
        lane: "blue_brain_future_action_handoff_future_action_ready",
        handoff_semantics:
            "future-action-ready handoff prepared from action-ready proposal and not executed",
        proposal_identity_binding:
            "proposal identity references selected candidate/proposal digest and lane id",
        proposal_origin_binding:
            "candidate/context/evidence/selection/comparison/memory-boundary references are attached",
        readiness_basis_binding:
            "CANONICAL_BLUE_BRAIN_MINIMAL_PLANNING_ACTION_INTERFACE_MAP::ActionReadyProposal + CANONICAL_BLUE_BRAIN_PLAN_ACTION_READINESS_DIAGNOSTICS_MAP::ActionReadyDiagnostic",
        evidence_reference_basis_binding:
            "CANONICAL_BLUE_BRAIN_REFERENCE_CONTEXT_MAP + CANONICAL_BLUE_BRAIN_CONTEXT_EVIDENCE_PRIORITY_MAP",
        selection_attention_binding:
            "CANONICAL_BLUE_BRAIN_CONTROL_ATTENTION_SELECTION_MAP selected/attention-supported and still non-executing",
        caveat_or_blocker_binding:
            "no blocking reason required; caveat field can remain empty and explicit",
        execution_and_commit_boundary:
            "handoff only: no action execution, no tool invocation, no compute invocation, no memory commit",
        runtime_diagnostics_binding:
            "runtime marks future-action handoff prepared and keeps no-action/no-tool/no-compute/no-commit flags explicit",
        canonical_guard:
            "future-action-ready handoff is preparatory and never an execution or result claim",
    },
    BlueBrainFutureActionHandoffLane {
        class: BlueBrainFutureActionHandoffClass::FuturePlanReady,
        lane: "blue_brain_future_action_handoff_future_plan_ready",
        handoff_semantics: "future-plan-ready handoff prepared from plan-ready proposal only",
        proposal_identity_binding:
            "proposal identity references plan-ready proposal digest and readiness class",
        proposal_origin_binding:
            "proposal origin links candidate/context/evidence/selection/comparison/memory-boundary traces",
        readiness_basis_binding:
            "CANONICAL_BLUE_BRAIN_MINIMAL_PLANNING_ACTION_INTERFACE_MAP::PlanReadyProposal + CANONICAL_BLUE_BRAIN_PLAN_ACTION_READINESS_DIAGNOSTICS_MAP::PlanReadyDiagnostic",
        evidence_reference_basis_binding:
            "context/evidence references remain explicit and can include caveated basis",
        selection_attention_binding:
            "selection can remain deferred or selected-for-future-boundary without plan generation",
        caveat_or_blocker_binding:
            "caveat field optional; no blocked/rejected reason required for ready state",
        execution_and_commit_boundary:
            "handoff only: no action execution, no tool invocation, no compute invocation, no memory commit",
        runtime_diagnostics_binding:
            "runtime marks future-plan handoff prepared and preserves proposal-not-executed semantics",
        canonical_guard: "future-plan-ready handoff != plan generated != plan executed",
    },
    BlueBrainFutureActionHandoffLane {
        class: BlueBrainFutureActionHandoffClass::HandoffDeferred,
        lane: "blue_brain_future_action_handoff_deferred",
        handoff_semantics:
            "handoff deferred because readiness basis is incomplete or postponed for later window",
        proposal_identity_binding:
            "proposal identity retained with deferred reason and retry window references",
        proposal_origin_binding:
            "origin remains candidate/proposal/context/evidence/selection/comparison/memory-boundary",
        readiness_basis_binding:
            "CANONICAL_BLUE_BRAIN_PLAN_ACTION_READINESS_DIAGNOSTICS_MAP::DeferredReadinessDiagnostic",
        evidence_reference_basis_binding:
            "deferred due to partial or stale evidence/reference/context freshness signals",
        selection_attention_binding: "selection posture can remain deferred without rejection semantics",
        caveat_or_blocker_binding:
            "deferred reason must be explicit and not remapped to blocked/rejected",
        execution_and_commit_boundary:
            "deferred handoff remains non-executing: no action/tool/compute invocation and no memory commit",
        runtime_diagnostics_binding:
            "runtime emits handoff deferred with cause references and no side effects",
        canonical_guard: "deferred handoff is canonical and non-terminal",
    },
    BlueBrainFutureActionHandoffLane {
        class: BlueBrainFutureActionHandoffClass::HandoffBlocked,
        lane: "blue_brain_future_action_handoff_blocked",
        handoff_semantics:
            "handoff blocked due to insufficient basis or missing canonical action/plan boundary",
        proposal_identity_binding: "proposal identity retained with blocked reason taxonomy",
        proposal_origin_binding:
            "origin links blocked state to candidate/proposal/context/evidence/selection/comparison/memory-boundary references",
        readiness_basis_binding:
            "CANONICAL_BLUE_BRAIN_PLAN_ACTION_READINESS_DIAGNOSTICS_MAP::BlockedReadinessDiagnostic|InsufficientReadinessDiagnostic",
        evidence_reference_basis_binding:
            "blocked by stale/missing evidence-reference anchors or non-canonical dependencies",
        selection_attention_binding:
            "selection state cannot override blocked readiness without basis repair",
        caveat_or_blocker_binding:
            "blocked reason required and distinct from deferred/rejected/caveated",
        execution_and_commit_boundary:
            "blocked handoff performs no action execution, no tool invocation, no compute invocation, no memory commit",
        runtime_diagnostics_binding:
            "runtime emits blocked-action handoff feedback and keeps it separate from execution failure",
        canonical_guard:
            "blocked handoff cannot be bypassed by internal hooks or implicit tool/runtime orchestration",
    },
    BlueBrainFutureActionHandoffLane {
        class: BlueBrainFutureActionHandoffClass::HandoffRejected,
        lane: "blue_brain_future_action_handoff_rejected",
        handoff_semantics: "handoff rejected due to proposal rejection in current readiness window",
        proposal_identity_binding: "proposal identity retained for rejected traceability",
        proposal_origin_binding:
            "origin captures rejected candidate/proposal and linked context/evidence references",
        readiness_basis_binding:
            "CANONICAL_BLUE_BRAIN_PLAN_ACTION_READINESS_DIAGNOSTICS_MAP::RejectedReadinessDiagnostic",
        evidence_reference_basis_binding:
            "rejection reason references canonical evidence/context diagnostics only",
        selection_attention_binding: "rejected proposal excluded from current selection-to-handoff path",
        caveat_or_blocker_binding:
            "rejection reason required and cannot be rewritten as execution outcome",
        execution_and_commit_boundary:
            "rejected handoff remains non-executing: no action/tool/compute invocation and no memory commit",
        runtime_diagnostics_binding:
            "runtime emits handoff rejected and preserves rejected vs blocked separation",
        canonical_guard: "rejected handoff is proposal-boundary feedback and not action/runtime result",
    },
    BlueBrainFutureActionHandoffLane {
        class: BlueBrainFutureActionHandoffClass::HandoffCaveated,
        lane: "blue_brain_future_action_handoff_caveated",
        handoff_semantics:
            "handoff caveated where basis is usable but caveats from comparison/memory/context persist",
        proposal_identity_binding:
            "proposal identity retained with caveat vector and caveat provenance references",
        proposal_origin_binding:
            "origin includes candidate comparison and memory-boundary caveats",
        readiness_basis_binding:
            "CANONICAL_BLUE_BRAIN_PLAN_ACTION_READINESS_DIAGNOSTICS_MAP::CaveatedReadinessDiagnostic",
        evidence_reference_basis_binding: "evidence/reference basis attached with explicit caveat markers",
        selection_attention_binding:
            "selection can defer or allow caveated handoff but remains non-executing",
        caveat_or_blocker_binding:
            "caveat reason required and bounded; no free speculative prose as canonical authority",
        execution_and_commit_boundary:
            "caveated handoff performs no action execution, no tool invocation, no compute invocation, no memory commit",
        runtime_diagnostics_binding:
            "runtime emits handoff caveated with comparison and memory-boundary caveats retained",
        canonical_guard: "caveated handoff remains preparatory and does not claim result certainty",
    },
    BlueBrainFutureActionHandoffLane {
        class: BlueBrainFutureActionHandoffClass::HandoffUnavailable,
        lane: "blue_brain_future_action_handoff_unavailable",
        handoff_semantics: "handoff unavailable because no canonical future action/plan subsystem is present",
        proposal_identity_binding:
            "proposal identity can be present, but handoff endpoint is unavailable",
        proposal_origin_binding:
            "origin stays linked to candidate/proposal/context/evidence/selection baseline",
        readiness_basis_binding:
            "readiness may be action-ready/plan-ready but handoff endpoint remains unavailable",
        evidence_reference_basis_binding:
            "evidence references remain diagnostics-only while handoff endpoint is absent",
        selection_attention_binding:
            "selection/attention support does not create fallback execution path",
        caveat_or_blocker_binding:
            "unavailable reason required and distinct from blocked/rejected/deferred",
        execution_and_commit_boundary:
            "unavailable handoff performs no action execution, no tool invocation, no compute invocation, no memory commit",
        runtime_diagnostics_binding:
            "runtime emits handoff unavailable and no-action/no-tool/no-compute/no-commit explicitly",
        canonical_guard:
            "handoff unavailable confirms BB7 boundary: no automatic transition to execution subsystems",
    },
    BlueBrainFutureActionHandoffLane {
        class: BlueBrainFutureActionHandoffClass::DiagnosticOnlyNoHandoff,
        lane: "blue_brain_future_action_handoff_diagnostic_only_no_handoff",
        handoff_semantics:
            "diagnostic-only/no-handoff state where proposal remains below future handoff readiness",
        proposal_identity_binding:
            "proposal identity optional; if present remains diagnostics-only and non-handoff",
        proposal_origin_binding:
            "origin references candidate/proposal diagnostics without handoff claim",
        readiness_basis_binding:
            "CANONICAL_BLUE_BRAIN_PLAN_ACTION_READINESS_DIAGNOSTICS_MAP::DiagnosticOnlyProposalDiagnostic",
        evidence_reference_basis_binding:
            "diagnostic basis may be partial/caveated and insufficient for handoff transition",
        selection_attention_binding:
            "selection feedback may exist but cannot escalate diagnostic-only to handoff",
        caveat_or_blocker_binding:
            "diagnostic-only reason required when no handoff object is produced",
        execution_and_commit_boundary:
            "diagnostic-only/no-handoff: no action execution, no tool invocation, no compute invocation, no memory commit",
        runtime_diagnostics_binding:
            "runtime emits diagnostic-only/no-handoff status as first-class readiness output",
        canonical_guard:
            "diagnostic-only/no-handoff is canonical and must not be inferred as hidden execution request",
    },
    BlueBrainFutureActionHandoffLane {
        class: BlueBrainFutureActionHandoffClass::InternalOnlyNonCanonicalHandoff,
        lane: "blue_brain_future_action_handoff_internal_only_non_canonical",
        handoff_semantics:
            "internal/expert/legacy handoff-like path without canonical down-map to proposal/readiness basis",
        proposal_identity_binding:
            "identity from internal hooks is non-canonical unless mapped to canonical proposal identity",
        proposal_origin_binding:
            "compute-internal/expert/dev/test origins are non-canonical for BB7 future handoff authority",
        readiness_basis_binding:
            "internal-only readiness/tool/orchestration helpers cannot define canonical handoff state",
        evidence_reference_basis_binding:
            "internal evidence views are non-canonical unless mapped to canonical references",
        selection_attention_binding:
            "internal selection hooks cannot produce canonical future-action/future-plan handoff claims",
        caveat_or_blocker_binding:
            "internal-only/non-canonical reason required and exported with canonical=false posture",
        execution_and_commit_boundary:
            "non-canonical handoff carries no canonical action/tool/compute/memory authority",
        runtime_diagnostics_binding:
            "runtime marks internal-only handoff as canonical=false and segregates it from canonical map",
        canonical_guard:
            "compute-internal details, expert hooks, legacy compat objects, unstable dev/test surfaces are excluded",
    },
];

pub const CANONICAL_BLUE_BRAIN_ACTION_RESULT_PLACEHOLDER_MAP:
    [BlueBrainActionResultPlaceholderLane; 8] = [
    BlueBrainActionResultPlaceholderLane {
        class: BlueBrainActionResultPlaceholderClass::ResultPlaceholderPrepared,
        lane: "blue_brain_action_result_placeholder_prepared",
        placeholder_semantics:
            "result placeholder prepared for future subsystem output shape and no actual result populated",
        handoff_dependency_binding:
            "CANONICAL_BLUE_BRAIN_FUTURE_ACTION_HANDOFF_MAP::FutureActionReady|FuturePlanReady|HandoffCaveated",
        result_slot_shape:
            "slot supports future status/result/error references while current payload remains placeholder-only",
        boundary_semantics:
            "prepared placeholder means no action executed, no tool result, no compute invocation, no memory commit",
        runtime_diagnostics_binding:
            "runtime emits result placeholder prepared with explicit no-action/no-tool/no-compute/no-commit flags",
        canonical_guard:
            "placeholder prepared is not a result claim and not proof of execution",
    },
    BlueBrainActionResultPlaceholderLane {
        class: BlueBrainActionResultPlaceholderClass::ResultPlaceholderUnavailable,
        lane: "blue_brain_action_result_placeholder_unavailable",
        placeholder_semantics:
            "result placeholder unavailable because handoff endpoint or slot provisioning is unavailable",
        handoff_dependency_binding:
            "CANONICAL_BLUE_BRAIN_FUTURE_ACTION_HANDOFF_MAP::HandoffUnavailable",
        result_slot_shape: "no placeholder slot can be provisioned in current boundary window",
        boundary_semantics:
            "unavailable placeholder means no result, no action execution, no tool result, no compute invocation, no memory commit",
        runtime_diagnostics_binding:
            "runtime emits placeholder unavailable and keeps it separate from blocked/rejected handoff",
        canonical_guard: "placeholder unavailable cannot be rewritten as execution failure or result error",
    },
    BlueBrainActionResultPlaceholderLane {
        class: BlueBrainActionResultPlaceholderClass::ResultPlaceholderBlocked,
        lane: "blue_brain_action_result_placeholder_blocked",
        placeholder_semantics:
            "result placeholder blocked when canonical handoff basis is blocked/insufficient",
        handoff_dependency_binding:
            "CANONICAL_BLUE_BRAIN_FUTURE_ACTION_HANDOFF_MAP::HandoffBlocked",
        result_slot_shape:
            "placeholder slot withheld until blocked basis is repaired and canonical handoff resumes",
        boundary_semantics:
            "blocked placeholder means no action executed, no tool result, no compute invocation, no memory commit",
        runtime_diagnostics_binding:
            "runtime emits placeholder blocked with blocked reason references",
        canonical_guard: "placeholder blocked tracks handoff blocker and remains non-executing",
    },
    BlueBrainActionResultPlaceholderLane {
        class: BlueBrainActionResultPlaceholderClass::ResultPlaceholderCaveated,
        lane: "blue_brain_action_result_placeholder_caveated",
        placeholder_semantics:
            "result placeholder caveated when handoff is caveated and future result slot requires caveat carry-over",
        handoff_dependency_binding:
            "CANONICAL_BLUE_BRAIN_FUTURE_ACTION_HANDOFF_MAP::HandoffCaveated",
        result_slot_shape:
            "placeholder slot includes caveat metadata fields with no actual action/tool result payload",
        boundary_semantics:
            "caveated placeholder means no action executed, no tool result, no compute invocation, no memory commit",
        runtime_diagnostics_binding:
            "runtime emits placeholder caveated and preserves comparison/memory-boundary caveats",
        canonical_guard: "placeholder caveat is metadata only and not a result confidence claim",
    },
    BlueBrainActionResultPlaceholderLane {
        class: BlueBrainActionResultPlaceholderClass::NoResultExpected,
        lane: "blue_brain_action_result_placeholder_no_result_expected",
        placeholder_semantics:
            "no result expected for diagnostic-only/no-handoff/rejected paths in current window",
        handoff_dependency_binding:
            "CANONICAL_BLUE_BRAIN_FUTURE_ACTION_HANDOFF_MAP::DiagnosticOnlyNoHandoff|HandoffRejected",
        result_slot_shape: "result slot omitted or explicitly marked no_result_expected",
        boundary_semantics:
            "no-result-expected means no action executed, no tool result, no compute invocation, no memory commit",
        runtime_diagnostics_binding:
            "runtime emits no-result-expected and keeps proposal diagnostics available",
        canonical_guard:
            "no-result-expected is canonical and must not be interpreted as missing execution data",
    },
    BlueBrainActionResultPlaceholderLane {
        class: BlueBrainActionResultPlaceholderClass::NoActionExecuted,
        lane: "blue_brain_action_result_placeholder_no_action_executed",
        placeholder_semantics:
            "explicit no action executed marker carried with placeholder semantics",
        handoff_dependency_binding:
            "all canonical handoff classes retain no-action-executed baseline in BB7",
        result_slot_shape:
            "placeholder status includes no_action_executed=true and empty action outcome payload",
        boundary_semantics:
            "no-action-executed placeholder implies no tool result, no compute invocation, no memory commit",
        runtime_diagnostics_binding:
            "runtime emits no_action_executed marker for future handoff and placeholder lanes",
        canonical_guard: "no-action-executed is mandatory until explicit downstream execution path exists",
    },
    BlueBrainActionResultPlaceholderLane {
        class: BlueBrainActionResultPlaceholderClass::NoToolResult,
        lane: "blue_brain_action_result_placeholder_no_tool_result",
        placeholder_semantics:
            "explicit no tool result marker to avoid false tool execution inference",
        handoff_dependency_binding:
            "all canonical handoff classes preserve no_tool_result baseline",
        result_slot_shape:
            "placeholder status includes no_tool_result=true and empty tool output references",
        boundary_semantics:
            "no-tool-result placeholder implies no action executed, no compute invocation, no memory commit",
        runtime_diagnostics_binding:
            "runtime emits no_tool_result marker and keeps tool invocation boundary explicit",
        canonical_guard:
            "no_tool_result placeholder cannot be auto-upgraded to tool error or tool success without execution path",
    },
    BlueBrainActionResultPlaceholderLane {
        class: BlueBrainActionResultPlaceholderClass::InternalOnlyNonCanonicalPlaceholder,
        lane: "blue_brain_action_result_placeholder_internal_only_non_canonical",
        placeholder_semantics:
            "internal/expert/dev placeholder-like object without canonical handoff references",
        handoff_dependency_binding:
            "internal-only helper/orchestration/tooling path and not canonical BB7 placeholder",
        result_slot_shape:
            "shape may resemble result slot but lacks canonical proposal/handoff/readiness binding",
        boundary_semantics:
            "internal-only placeholder has no canonical action/tool/compute/memory authority",
        runtime_diagnostics_binding:
            "runtime exports internal-only placeholder as canonical=false and excludes it from canonical lane claims",
        canonical_guard:
            "internal/expert hooks, legacy compat objects, and unstable dev/test surfaces remain non-canonical",
    },
];

pub const CANONICAL_BLUE_BRAIN_SAFETY_PRECHECK_MAP: [BlueBrainSafetyPrecheckLane; 7] = [
    BlueBrainSafetyPrecheckLane {
        class: BlueBrainSafetyPrecheckClass::Passed,
        lane: "blue_brain_safety_precheck_passed",
        precheck_semantics: "precheck passed on canonical safety basis and remains non-executing",
        basis_binding:
            "references canonical context/evidence/selection/proposal/memory diagnostics without policy-governance expansion",
        eligibility_effect:
            "permits execution-eligible classification when all other eligibility conditions remain satisfied",
        execution_boundary:
            "precheck passed is not execution, not tool invocation, not compute invocation, and not memory commit",
        canonical_guard:
            "precheck is bounded safety semantics only; no governance policy verdict and no planner output",
    },
    BlueBrainSafetyPrecheckLane {
        class: BlueBrainSafetyPrecheckClass::Failed,
        lane: "blue_brain_safety_precheck_failed",
        precheck_semantics:
            "precheck failed due to canonical safety blocker and prevents eligibility promotion",
        basis_binding:
            "failed state references explicit canonical blocker diagnostics and not internal-only hooks",
        eligibility_effect: "forces execution-ineligible or execution-blocked classification",
        execution_boundary:
            "precheck failed emits diagnostics only and performs no action/tool/compute/memory side effects",
        canonical_guard: "failed precheck cannot be downplayed as caveated or advisory-only noise",
    },
    BlueBrainSafetyPrecheckLane {
        class: BlueBrainSafetyPrecheckClass::Blocked,
        lane: "blue_brain_safety_precheck_blocked",
        precheck_semantics:
            "precheck blocked because required canonical safety/context/evidence basis is blocked",
        basis_binding:
            "blocked state binds to missing or blocked canonical references and down-mapped non-canonical dependencies",
        eligibility_effect: "forces execution-blocked classification until basis repair",
        execution_boundary:
            "precheck blocked remains diagnostic and non-executing with no tool/compute/commit activity",
        canonical_guard:
            "blocked precheck is distinct from failed/insufficient/unavailable and must stay explicit",
    },
    BlueBrainSafetyPrecheckLane {
        class: BlueBrainSafetyPrecheckClass::Caveated,
        lane: "blue_brain_safety_precheck_caveated",
        precheck_semantics:
            "precheck caveated when safety basis is usable but bounded caveats remain explicit",
        basis_binding:
            "caveat references canonical memory, comparison, context, and evidence caveat diagnostics",
        eligibility_effect:
            "allows execution-caveated classification only when caveated-allowed conditions are explicit",
        execution_boundary:
            "precheck caveated never implies executed action and cannot invoke tools/compute/commit",
        canonical_guard:
            "caveated precheck is bounded and canonical, not free-form policy/governance prose",
    },
    BlueBrainSafetyPrecheckLane {
        class: BlueBrainSafetyPrecheckClass::Insufficient,
        lane: "blue_brain_safety_precheck_insufficient",
        precheck_semantics:
            "precheck insufficient because required canonical safety basis is below minimum threshold",
        basis_binding:
            "insufficient binds to canonical missing/weak context-evidence-selection-memory diagnostics",
        eligibility_effect: "forces execution-insufficient-basis classification",
        execution_boundary:
            "insufficient precheck is strictly non-executing and emits diagnostics only",
        canonical_guard:
            "insufficient precheck cannot be auto-promoted through internal/expert-only dependencies",
    },
    BlueBrainSafetyPrecheckLane {
        class: BlueBrainSafetyPrecheckClass::Unavailable,
        lane: "blue_brain_safety_precheck_unavailable",
        precheck_semantics:
            "precheck unavailable when canonical precheck surface is absent in current boundary window",
        basis_binding:
            "unavailable reason references endpoint or dependency unavailability with canonical basis retention",
        eligibility_effect: "prevents execution-eligible and yields execution-ineligible or blocked posture",
        execution_boundary:
            "unavailable precheck does not fallback to implicit execution path, tool call, compute call, or memory commit",
        canonical_guard:
            "unavailable is explicit and distinct from failed/blocked/insufficient to avoid semantic collapse",
    },
    BlueBrainSafetyPrecheckLane {
        class: BlueBrainSafetyPrecheckClass::NotApplicable,
        lane: "blue_brain_safety_precheck_not_applicable",
        precheck_semantics:
            "precheck not applicable for strictly diagnostic-only and non-handoff paths",
        basis_binding:
            "N/A state is valid only when no execution-eligibility transition is requested",
        eligibility_effect:
            "keeps handoff diagnostic-only or future-action-ready without execution-eligible promotion",
        execution_boundary:
            "not-applicable precheck has no execution implication and no action/tool/compute/commit side effects",
        canonical_guard:
            "not-applicable cannot be used to bypass required precheck for execution eligibility",
    },
];

pub const CANONICAL_BLUE_BRAIN_ACTION_EXECUTION_ELIGIBILITY_MAP:
    [BlueBrainActionExecutionEligibilityLane; 13] = [
    BlueBrainActionExecutionEligibilityLane {
        class: BlueBrainActionExecutionEligibilityClass::FutureActionReadyHandoff,
        lane: "blue_brain_execution_eligibility_future_action_ready_only",
        eligibility_semantics:
            "future-action-ready handoff is preparatory and not yet execution-eligible without safety precheck",
        handoff_binding:
            "CANONICAL_BLUE_BRAIN_FUTURE_ACTION_HANDOFF_MAP::FutureActionReady|FuturePlanReady",
        context_evidence_basis:
            "context/evidence basis can be sufficient or caveated while still remaining future-ready only",
        selection_candidate_basis:
            "selected or accepted proposal basis is required for future-ready handoff identity",
        memory_basis:
            "memory basis attached as current/stale/caveated/invalidated/missing diagnostics and not auto-committed",
        safety_precheck_binding:
            "CANONICAL_BLUE_BRAIN_SAFETY_PRECHECK_MAP::NotApplicable|Unavailable before eligibility promotion",
        execution_boundary:
            "future-action-ready only: no execution, no tool invocation, no compute invocation, no memory commit",
        canonical_guard:
            "future-action-ready must stay distinct from execution-eligible and executed action states",
    },
    BlueBrainActionExecutionEligibilityLane {
        class: BlueBrainActionExecutionEligibilityClass::ExecutionEligibleHandoff,
        lane: "blue_brain_execution_eligibility_execution_eligible",
        eligibility_semantics:
            "handoff is execution-eligible when basis conditions and safety precheck passed/caveated-allowed are explicit",
        handoff_binding: "requires canonical future-action-ready handoff identity",
        context_evidence_basis:
            "requires sufficient or caveated-allowed context and evidence/reference basis",
        selection_candidate_basis:
            "requires selected or accepted proposal state with no blocking candidate/proposal diagnostics",
        memory_basis:
            "requires current memory basis or caveated-acceptable memory status; invalidated memory blocks eligibility",
        safety_precheck_binding:
            "CANONICAL_BLUE_BRAIN_SAFETY_PRECHECK_MAP::Passed|Caveated",
        execution_boundary:
            "execution-eligible classification is non-executing: no tool/action/compute invocation and no memory commit",
        canonical_guard:
            "eligibility is a boundary status only and does not authorize autonomous execution loops",
    },
    BlueBrainActionExecutionEligibilityLane {
        class: BlueBrainActionExecutionEligibilityClass::ExecutionIneligibleHandoff,
        lane: "blue_brain_execution_eligibility_ineligible",
        eligibility_semantics: "handoff remains execution-ineligible when eligibility conditions are not met",
        handoff_binding:
            "future-action-ready or diagnostic-only handoff can remain ineligible without execution claims",
        context_evidence_basis:
            "missing or non-caveated-allowed context/evidence basis keeps ineligible posture",
        selection_candidate_basis:
            "unselected, rejected, or non-canonical proposal basis prevents eligibility",
        memory_basis: "missing memory basis can keep eligibility insufficient/ineligible",
        safety_precheck_binding:
            "precheck failed/unavailable/not-applicable for requested execution path keeps ineligible state",
        execution_boundary:
            "ineligible handoff never triggers action/tool/compute execution or memory commit",
        canonical_guard:
            "execution-ineligible is explicit and must not be collapsed into executed-action placeholders",
    },
    BlueBrainActionExecutionEligibilityLane {
        class: BlueBrainActionExecutionEligibilityClass::ExecutionBlockedHandoff,
        lane: "blue_brain_execution_eligibility_blocked",
        eligibility_semantics:
            "handoff execution blocked by canonical blocker in context/evidence/selection/memory/safety basis",
        handoff_binding: "blocked state retains canonical handoff identity and blocker reason",
        context_evidence_basis: "blocked context or evidence reference basis halts eligibility progression",
        selection_candidate_basis:
            "selection blocked or active deferral with blocker semantics blocks eligibility",
        memory_basis: "invalidated memory basis blocks eligibility until repaired",
        safety_precheck_binding: "CANONICAL_BLUE_BRAIN_SAFETY_PRECHECK_MAP::Blocked|Failed",
        execution_boundary:
            "blocked eligibility remains diagnostic-only with no action/tool/compute/commit side effects",
        canonical_guard:
            "blocked execution eligibility is distinct from failed execution result and policy governance outputs",
    },
    BlueBrainActionExecutionEligibilityLane {
        class: BlueBrainActionExecutionEligibilityClass::ExecutionCaveatedHandoff,
        lane: "blue_brain_execution_eligibility_caveated",
        eligibility_semantics:
            "handoff can be execution-caveated when basis is usable with bounded caveats",
        handoff_binding:
            "future-action-ready handoff carries explicit caveat vector and provenance references",
        context_evidence_basis: "context/evidence caveats are explicit and bounded for eligibility use",
        selection_candidate_basis:
            "proposal may be selected/accepted with caveated diagnostics and no blocking state",
        memory_basis: "stale or caveated memory can yield caveated eligibility if explicitly accepted",
        safety_precheck_binding: "CANONICAL_BLUE_BRAIN_SAFETY_PRECHECK_MAP::Caveated",
        execution_boundary:
            "execution-caveated remains non-executing and cannot auto-invoke tools/compute/commit",
        canonical_guard:
            "caveated eligibility is not equivalent to precheck failed/blocked or executed action",
    },
    BlueBrainActionExecutionEligibilityLane {
        class: BlueBrainActionExecutionEligibilityClass::ExecutionInsufficientBasis,
        lane: "blue_brain_execution_eligibility_insufficient_basis",
        eligibility_semantics:
            "handoff basis is insufficient for execution eligibility due to missing required canonical inputs",
        handoff_binding:
            "diagnostic-only, deferred, or future-ready handoff can remain insufficient without blocker semantics",
        context_evidence_basis:
            "insufficient context/evidence/reference basis keeps eligibility below minimum threshold",
        selection_candidate_basis:
            "candidate comparison inconclusive or proposal insufficient keeps insufficient state explicit",
        memory_basis:
            "missing or stale-not-allowed memory basis contributes to insufficient classification",
        safety_precheck_binding: "CANONICAL_BLUE_BRAIN_SAFETY_PRECHECK_MAP::Insufficient",
        execution_boundary:
            "insufficient eligibility performs no execution, tool call, compute call, or memory commit",
        canonical_guard:
            "insufficient is distinct from blocked/failed/unavailable and not a hidden policy decision",
    },
    BlueBrainActionExecutionEligibilityLane {
        class: BlueBrainActionExecutionEligibilityClass::SafetyPrecheckPassed,
        lane: "blue_brain_execution_eligibility_precheck_passed_binding",
        eligibility_semantics: "eligibility map tracks explicit safety-precheck passed binding",
        handoff_binding: "binds to canonical future-action-ready handoff identity",
        context_evidence_basis: "precheck passed requires canonical safety basis references",
        selection_candidate_basis: "selection/proposal basis remains required independent of precheck",
        memory_basis: "memory basis must still satisfy current or caveated-acceptable requirement",
        safety_precheck_binding: "CANONICAL_BLUE_BRAIN_SAFETY_PRECHECK_MAP::Passed",
        execution_boundary:
            "precheck passed binding alone does not execute action, tool, compute, or memory commit",
        canonical_guard:
            "precheck passed is necessary but not sufficient without full eligibility condition set",
    },
    BlueBrainActionExecutionEligibilityLane {
        class: BlueBrainActionExecutionEligibilityClass::SafetyPrecheckFailed,
        lane: "blue_brain_execution_eligibility_precheck_failed_binding",
        eligibility_semantics: "eligibility map tracks explicit safety-precheck failed binding",
        handoff_binding: "failed binding references same handoff identity for traceability",
        context_evidence_basis: "failed precheck is kept separate from context/evidence sufficiency diagnostics",
        selection_candidate_basis:
            "selection/proposal acceptance cannot override failed safety precheck",
        memory_basis: "memory basis cannot override failed safety precheck",
        safety_precheck_binding: "CANONICAL_BLUE_BRAIN_SAFETY_PRECHECK_MAP::Failed",
        execution_boundary:
            "precheck failed binding is non-executing and no fallback action/tool/compute/commit occurs",
        canonical_guard: "precheck failed remains canonical blocker for execution eligibility promotion",
    },
    BlueBrainActionExecutionEligibilityLane {
        class: BlueBrainActionExecutionEligibilityClass::SafetyPrecheckBlocked,
        lane: "blue_brain_execution_eligibility_precheck_blocked_binding",
        eligibility_semantics: "eligibility map tracks explicit safety-precheck blocked binding",
        handoff_binding: "blocked precheck preserves handoff and blocker reason traceability",
        context_evidence_basis: "blocked precheck can derive from blocked context/evidence prerequisites",
        selection_candidate_basis:
            "selection/proposal state cannot bypass blocked precheck requirements",
        memory_basis: "invalidated or unavailable memory basis can contribute to blocked precheck",
        safety_precheck_binding: "CANONICAL_BLUE_BRAIN_SAFETY_PRECHECK_MAP::Blocked",
        execution_boundary:
            "blocked precheck binding remains non-executing and cannot invoke tools/compute/commit",
        canonical_guard:
            "blocked precheck binding remains separate from failed/unavailable classifications",
    },
    BlueBrainActionExecutionEligibilityLane {
        class: BlueBrainActionExecutionEligibilityClass::SafetyPrecheckCaveated,
        lane: "blue_brain_execution_eligibility_precheck_caveated_binding",
        eligibility_semantics: "eligibility map tracks explicit safety-precheck caveated binding",
        handoff_binding: "caveated precheck preserves handoff identity and caveat references",
        context_evidence_basis:
            "partial context/evidence sufficiency can bind to caveated precheck without being treated as failed",
        selection_candidate_basis:
            "selection/proposal basis remains required and caveat metadata must stay explicit",
        memory_basis: "caveated or stale-accepted memory can bind to caveated precheck classification",
        safety_precheck_binding: "CANONICAL_BLUE_BRAIN_SAFETY_PRECHECK_MAP::Caveated",
        execution_boundary:
            "caveated precheck binding is non-executing and cannot auto-trigger tool/action/compute/commit",
        canonical_guard:
            "caveated precheck is distinct from passed/failed/blocked/unavailable and remains diagnostic",
    },
    BlueBrainActionExecutionEligibilityLane {
        class: BlueBrainActionExecutionEligibilityClass::SafetyPrecheckUnavailable,
        lane: "blue_brain_execution_eligibility_precheck_unavailable_binding",
        eligibility_semantics: "eligibility map tracks explicit safety-precheck unavailable binding",
        handoff_binding: "unavailable precheck still references canonical handoff identity",
        context_evidence_basis:
            "context/evidence may be sufficient but unavailable precheck still blocks eligibility promotion",
        selection_candidate_basis:
            "selected proposal state does not auto-upgrade unavailable precheck to passed",
        memory_basis: "memory readiness does not substitute for unavailable precheck endpoint",
        safety_precheck_binding: "CANONICAL_BLUE_BRAIN_SAFETY_PRECHECK_MAP::Unavailable",
        execution_boundary:
            "precheck unavailable binding is non-executing and retains no tool/action/compute/commit side effects",
        canonical_guard: "unavailable precheck is explicit boundary state and not implicit failure",
    },
    BlueBrainActionExecutionEligibilityLane {
        class: BlueBrainActionExecutionEligibilityClass::ExecutedActionCanonicalIfPresent,
        lane: "blue_brain_execution_eligibility_executed_action_only_if_real_path_exists",
        eligibility_semantics:
            "executed action classification is valid only if a separate real execution path exists",
        handoff_binding:
            "executed action, if present, must remain downstream of canonical eligibility and safety precheck surfaces",
        context_evidence_basis: "execution result references are external and not implied by eligibility map",
        selection_candidate_basis: "proposal readiness and selection do not themselves constitute execution",
        memory_basis: "eligibility classification does not auto-write memory commit after execution",
        safety_precheck_binding:
            "precheck lineage can be referenced for provenance without conflating eligibility with execution result",
        execution_boundary:
            "BB9 map defines eligibility/precheck boundary only and does not introduce execution engine behavior",
        canonical_guard:
            "if no real execution path exists, this class remains documentary only and never emitted as canonical runtime outcome",
    },
    BlueBrainActionExecutionEligibilityLane {
        class: BlueBrainActionExecutionEligibilityClass::NonCanonicalInternalOnlyExecutionPath,
        lane: "blue_brain_execution_eligibility_non_canonical_internal_only",
        eligibility_semantics:
            "internal/expert/legacy execution-like path is non-canonical for BB9 eligibility authority",
        handoff_binding:
            "internal hooks without canonical future-handoff binding cannot author canonical eligibility claims",
        context_evidence_basis:
            "compute-internal details or free-form prose are non-canonical unless down-mapped",
        selection_candidate_basis:
            "internal-only proposal/selection helpers are excluded from canonical eligibility authority",
        memory_basis:
            "legacy or unstable test/dev memory helpers are non-canonical unless mapped to BB8 canonical records",
        safety_precheck_binding:
            "internal-only safety checks are non-canonical unless mapped to CANONICAL_BLUE_BRAIN_SAFETY_PRECHECK_MAP",
        execution_boundary:
            "non-canonical path has no canonical execution/tool/compute/memory authority and must be marked canonical=false",
        canonical_guard:
            "compute-internal details, expert hooks, legacy compat objects, unstable test/dev surfaces are excluded",
    },
];

pub const CANONICAL_BLUE_BRAIN_EXECUTION_ELIGIBILITY_DIAGNOSTICS_MAP:
    [BlueBrainExecutionEligibilityDiagnosticLane; 11] = [
    BlueBrainExecutionEligibilityDiagnosticLane {
        class: BlueBrainExecutionEligibilityDiagnosticClass::ExecutionEligibleDiagnostic,
        lane: "blue_brain_execution_diagnostic_execution_eligible",
        reason_class:
            BlueBrainExecutionEligibilityReasonClass::EligibleSufficientProposalContextEvidenceMemoryBasis,
        reason_compact: "eligible/sufficient-proposal-context-evidence-memory",
        handoff_proposal_binding:
            "future-action-ready handoff can become execution-eligible when proposal basis is selected/accepted",
        selection_deferral_binding:
            "selection may mark this handoff selectable at future action boundary; no ranker/policy authority implied",
        context_evidence_memory_binding:
            "requires sufficient context/evidence references and current or caveated-acceptable memory basis",
        runtime_feedback_binding:
            "runtime records eligibility observed + no action execution/no tool invocation/no compute invocation/no memory commit",
        boundary_guard:
            "eligibility diagnostic is not a tool result, policy result, planner verdict, or executed action",
    },
    BlueBrainExecutionEligibilityDiagnosticLane {
        class: BlueBrainExecutionEligibilityDiagnosticClass::ExecutionIneligibleDiagnostic,
        lane: "blue_brain_execution_diagnostic_execution_ineligible",
        reason_class: BlueBrainExecutionEligibilityReasonClass::IneligibleInsufficientProposalBasis,
        reason_compact: "ineligible/insufficient-proposal-basis",
        handoff_proposal_binding:
            "future-action-ready or action-ready proposal may remain execution-ineligible without execution claim",
        selection_deferral_binding:
            "selection keeps deferred posture and cannot auto-promote insufficient proposal basis",
        context_evidence_memory_binding:
            "proposal basis is insufficient even when context/evidence/memory references are present",
        runtime_feedback_binding:
            "runtime surfaces explicit execution-ineligible diagnostic with zero execution/tool/compute/commit effects",
        boundary_guard:
            "ineligible diagnostic must not be collapsed into blocked failure or executed-action placeholders",
    },
    BlueBrainExecutionEligibilityDiagnosticLane {
        class: BlueBrainExecutionEligibilityDiagnosticClass::ExecutionBlockedDiagnostic,
        lane: "blue_brain_execution_diagnostic_execution_blocked",
        reason_class: BlueBrainExecutionEligibilityReasonClass::BlockedMissingContextOrEvidence,
        reason_compact: "blocked/missing-context-or-evidence",
        handoff_proposal_binding:
            "handoff remains blocked even when proposal exists if context/evidence basis is missing",
        selection_deferral_binding:
            "selection/deferral keeps blocked handoff excluded from current execution eligibility",
        context_evidence_memory_binding:
            "blocked due to missing context/evidence references; basis must be repaired before eligibility promotion",
        runtime_feedback_binding:
            "runtime reports blocked diagnostic only; no tool/action/compute path is invoked",
        boundary_guard:
            "blocked diagnostic indicates boundary denial and is not an execution failure result",
    },
    BlueBrainExecutionEligibilityDiagnosticLane {
        class: BlueBrainExecutionEligibilityDiagnosticClass::ExecutionCaveatedDiagnostic,
        lane: "blue_brain_execution_diagnostic_execution_caveated",
        reason_class: BlueBrainExecutionEligibilityReasonClass::CaveatedPartialEvidenceOrMemory,
        reason_compact: "caveated/partial-evidence-or-memory",
        handoff_proposal_binding:
            "future-action-ready handoff may stay caveated while proposal remains diagnostic-capable",
        selection_deferral_binding:
            "selection can keep caveated state explicit instead of silently upgrading to eligible",
        context_evidence_memory_binding:
            "partial evidence or caveated memory is preserved with bounded caveat diagnostics",
        runtime_feedback_binding:
            "runtime emits caveated eligibility diagnostics and explicitly records non-execution boundaries",
        boundary_guard:
            "caveated diagnostic is neither failure nor policy verdict and never implies tool invocation",
    },
    BlueBrainExecutionEligibilityDiagnosticLane {
        class: BlueBrainExecutionEligibilityDiagnosticClass::ExecutionInsufficientDiagnostic,
        lane: "blue_brain_execution_diagnostic_execution_insufficient",
        reason_class: BlueBrainExecutionEligibilityReasonClass::IneligibleInsufficientProposalBasis,
        reason_compact: "insufficient/required-basis-missing",
        handoff_proposal_binding:
            "proposal and handoff remain diagnostic-only when minimum basis is missing",
        selection_deferral_binding:
            "insufficient basis cannot become execution-eligible through selection-only changes",
        context_evidence_memory_binding:
            "insufficient context/evidence or missing memory reference keeps state below eligibility threshold",
        runtime_feedback_binding:
            "runtime records insufficient diagnostic with no action/tool/compute/commit side effects",
        boundary_guard:
            "insufficient diagnostic is distinct from blocked and from unavailable subsystem cases",
    },
    BlueBrainExecutionEligibilityDiagnosticLane {
        class: BlueBrainExecutionEligibilityDiagnosticClass::SafetyPrecheckPassedDiagnostic,
        lane: "blue_brain_execution_diagnostic_safety_precheck_passed",
        reason_class:
            BlueBrainExecutionEligibilityReasonClass::EligibleSufficientProposalContextEvidenceMemoryBasis,
        reason_compact: "safety/passed",
        handoff_proposal_binding:
            "precheck passed may support promotion from future-action-ready to execution-eligible",
        selection_deferral_binding:
            "selection still requires canonical proposal basis and can remain deferred",
        context_evidence_memory_binding:
            "passed precheck does not bypass context/evidence/memory basis requirements",
        runtime_feedback_binding:
            "runtime records precheck-passed diagnostic while preserving no-execution/no-tool/no-compute/no-commit",
        boundary_guard:
            "precheck passed is diagnostic support only, not execution authorization by itself",
    },
    BlueBrainExecutionEligibilityDiagnosticLane {
        class: BlueBrainExecutionEligibilityDiagnosticClass::SafetyPrecheckFailedDiagnostic,
        lane: "blue_brain_execution_diagnostic_safety_precheck_failed",
        reason_class: BlueBrainExecutionEligibilityReasonClass::BlockedSafetyPrecheckFailed,
        reason_compact: "safety/failed",
        handoff_proposal_binding:
            "future-action-ready handoff remains non-eligible when safety precheck fails",
        selection_deferral_binding:
            "failed safety precheck excludes current execution eligibility independent of selection score",
        context_evidence_memory_binding:
            "context/evidence/memory sufficiency cannot override failed precheck blocker",
        runtime_feedback_binding:
            "runtime reports failed precheck as boundary blocker and keeps execution/tool/compute/commit absent",
        boundary_guard:
            "failed precheck is not tool execution failure and not a governance policy decision",
    },
    BlueBrainExecutionEligibilityDiagnosticLane {
        class: BlueBrainExecutionEligibilityDiagnosticClass::SafetyPrecheckBlockedDiagnostic,
        lane: "blue_brain_execution_diagnostic_safety_precheck_blocked",
        reason_class: BlueBrainExecutionEligibilityReasonClass::BlockedStaleOrInvalidatedMemory,
        reason_compact: "safety/blocked-stale-or-invalidated-memory",
        handoff_proposal_binding:
            "handoff stays blocked when stale/invalidated memory blocks safety precheck completion",
        selection_deferral_binding:
            "selection remains deferred/blocked until memory blocker is resolved",
        context_evidence_memory_binding:
            "blocked precheck reflects stale or invalidated memory and must retain BB8 maintenance semantics",
        runtime_feedback_binding:
            "runtime emits blocked precheck diagnostic with explicit no-action/no-tool/no-compute/no-commit boundary",
        boundary_guard:
            "blocked safety feedback is boundary-state only and must not be interpreted as execution attempt",
    },
    BlueBrainExecutionEligibilityDiagnosticLane {
        class: BlueBrainExecutionEligibilityDiagnosticClass::SafetyPrecheckCaveatedDiagnostic,
        lane: "blue_brain_execution_diagnostic_safety_precheck_caveated",
        reason_class: BlueBrainExecutionEligibilityReasonClass::CaveatedPartialEvidenceOrMemory,
        reason_compact: "safety/caveated",
        handoff_proposal_binding:
            "handoff keeps caveated safety lineage for future-action diagnostics without auto-upgrade",
        selection_deferral_binding:
            "selection preserves caveat-aware deferral rather than treating caveated as fully passed",
        context_evidence_memory_binding:
            "partial evidence or caveated memory remains explicit in safety diagnostic payload",
        runtime_feedback_binding:
            "runtime records caveated precheck and still enforces no execution/tool/compute/commit effects",
        boundary_guard:
            "caveated precheck is diagnostic metadata and not execution readiness proof by itself",
    },
    BlueBrainExecutionEligibilityDiagnosticLane {
        class: BlueBrainExecutionEligibilityDiagnosticClass::SafetyPrecheckUnavailableDiagnostic,
        lane: "blue_brain_execution_diagnostic_safety_precheck_unavailable",
        reason_class: BlueBrainExecutionEligibilityReasonClass::UnavailableNoExecutionSubsystem,
        reason_compact: "safety/unavailable-no-execution-subsystem",
        handoff_proposal_binding:
            "future-action-ready handoff remains non-eligible when precheck/execution subsystem is unavailable",
        selection_deferral_binding:
            "selection keeps handoff deferred and cannot convert unavailable subsystem to eligible state",
        context_evidence_memory_binding:
            "sufficient context/evidence/memory cannot bypass unavailable precheck endpoint",
        runtime_feedback_binding:
            "runtime reports unavailable safety precheck while explicitly preserving no action/tool/compute/memory commit",
        boundary_guard:
            "unavailable means boundary subsystem absence, not action execution failure",
    },
    BlueBrainExecutionEligibilityDiagnosticLane {
        class:
            BlueBrainExecutionEligibilityDiagnosticClass::NonCanonicalInternalOnlyExecutionDiagnostic,
        lane: "blue_brain_execution_diagnostic_non_canonical_internal_only",
        reason_class: BlueBrainExecutionEligibilityReasonClass::BlockedInternalOnlyDependency,
        reason_compact: "non-canonical/internal-only-dependency",
        handoff_proposal_binding:
            "internal/expert hooks without canonical handoff/proposal mapping cannot produce canonical diagnostics",
        selection_deferral_binding:
            "internal-only helpers remain excluded from canonical selection/deferral eligibility authority",
        context_evidence_memory_binding:
            "compute-internal or unstable test/dev details are non-canonical unless down-mapped to BB3/BB8 references",
        runtime_feedback_binding:
            "runtime marks these diagnostics canonical=false and excludes them from canonical eligibility outcomes",
        boundary_guard:
            "non-canonical diagnostic must never be surfaced as canonical tool/policy/execution result",
    },
];

pub fn canonical_compute_reference_map() -> &'static [ComputeReferenceLane] {
    &CANONICAL_COMPUTE_REFERENCE_MAP
}

pub fn canonical_production_reference_lane() -> ComputeReferenceLane {
    CANONICAL_COMPUTE_REFERENCE_MAP[0]
}

pub fn canonical_blue_brain_selection_diagnostics_map(
) -> &'static [BlueBrainSelectionDiagnosticLane] {
    &CANONICAL_BLUE_BRAIN_SELECTION_DIAGNOSTICS_MAP
}

pub fn canonical_blue_brain_candidate_action_boundary_map(
) -> &'static [BlueBrainCandidateActionBoundaryLane] {
    &CANONICAL_BLUE_BRAIN_CANDIDATE_ACTION_BOUNDARY_MAP
}

pub fn canonical_blue_brain_candidate_to_proposal_transition_map(
) -> &'static [BlueBrainCandidateToProposalTransitionLane] {
    &CANONICAL_BLUE_BRAIN_CANDIDATE_TO_PROPOSAL_TRANSITION_MAP
}

pub fn canonical_blue_brain_non_executing_action_proposal_state_map(
) -> &'static [BlueBrainNonExecutingActionProposalStateLane] {
    &CANONICAL_BLUE_BRAIN_NON_EXECUTING_ACTION_PROPOSAL_STATE_MAP
}

pub fn canonical_blue_brain_reasoning_candidate_diagnostics_map(
) -> &'static [BlueBrainReasoningCandidateDiagnosticLane] {
    &CANONICAL_BLUE_BRAIN_REASONING_CANDIDATE_DIAGNOSTICS_MAP
}

pub fn canonical_blue_brain_candidate_comparison_map() -> &'static [BlueBrainCandidateComparisonLane]
{
    &CANONICAL_BLUE_BRAIN_CANDIDATE_COMPARISON_MAP
}

pub fn canonical_blue_brain_minimal_planning_action_interface_map(
) -> &'static [BlueBrainMinimalPlanningActionInterfaceLane] {
    &CANONICAL_BLUE_BRAIN_MINIMAL_PLANNING_ACTION_INTERFACE_MAP
}

pub fn canonical_blue_brain_plan_action_readiness_diagnostics_map(
) -> &'static [BlueBrainPlanActionReadinessDiagnosticLane] {
    &CANONICAL_BLUE_BRAIN_PLAN_ACTION_READINESS_DIAGNOSTICS_MAP
}

pub fn canonical_blue_brain_future_action_handoff_map(
) -> &'static [BlueBrainFutureActionHandoffLane] {
    &CANONICAL_BLUE_BRAIN_FUTURE_ACTION_HANDOFF_MAP
}

pub fn canonical_blue_brain_action_result_placeholder_map(
) -> &'static [BlueBrainActionResultPlaceholderLane] {
    &CANONICAL_BLUE_BRAIN_ACTION_RESULT_PLACEHOLDER_MAP
}

pub fn canonical_blue_brain_safety_precheck_map() -> &'static [BlueBrainSafetyPrecheckLane] {
    &CANONICAL_BLUE_BRAIN_SAFETY_PRECHECK_MAP
}

pub fn canonical_blue_brain_action_execution_eligibility_map(
) -> &'static [BlueBrainActionExecutionEligibilityLane] {
    &CANONICAL_BLUE_BRAIN_ACTION_EXECUTION_ELIGIBILITY_MAP
}

pub fn canonical_blue_brain_execution_eligibility_diagnostics_map(
) -> &'static [BlueBrainExecutionEligibilityDiagnosticLane] {
    &CANONICAL_BLUE_BRAIN_EXECUTION_ELIGIBILITY_DIAGNOSTICS_MAP
}

pub fn canonical_final_reference_line() -> CanonicalFinalReferenceLine {
    CANONICAL_FINAL_REFERENCE_LINE
}

pub fn is_canonical_core_or_extension_lane(class: ComputeReferenceClass) -> bool {
    !matches!(class, ComputeReferenceClass::InternalOrLegacy)
}

pub fn canonical_onboarding_reference_summary() -> (&'static str, &'static str) {
    (
        CANONICAL_ONBOARDING_BACKEND.as_env_str(),
        CANONICAL_ONBOARDING_PACK.as_str(),
    )
}

pub fn canonical_compute_integration_contract_view() -> &'static [ComputeIntegrationContractLane] {
    &CANONICAL_COMPUTE_INTEGRATION_CONTRACT_VIEW
}

pub fn is_outward_facing_compute_integration_boundary(
    boundary: ComputeIntegrationBoundary,
) -> bool {
    matches!(boundary, ComputeIntegrationBoundary::OutwardFacing)
}

pub fn canonical_domain_facing_compute_consumer_map() -> &'static [DomainFacingComputeConsumerLane]
{
    &CANONICAL_DOMAIN_FACING_COMPUTE_CONSUMER_MAP
}

pub fn canonical_first_domain_rollout_candidate_map() -> &'static [DomainRolloutCandidateLane] {
    &CANONICAL_FIRST_DOMAIN_ROLLOUT_CANDIDATE_MAP
}

pub fn canonical_first_domain_rollout_completion_map() -> &'static [FirstDomainRolloutCompletionLane]
{
    &CANONICAL_FIRST_DOMAIN_ROLLOUT_COMPLETION_MAP
}

pub fn canonical_post_rollout_adoption_map() -> &'static [PostRolloutAdoptionLane] {
    &CANONICAL_POST_ROLLOUT_ADOPTION_MAP
}

pub fn canonical_blue_brain_integration_map() -> &'static [BlueBrainIntegrationLane] {
    &CANONICAL_BLUE_BRAIN_INTEGRATION_MAP
}

pub fn canonical_blue_brain_facing_contract_map() -> &'static [BlueBrainFacingContractLane] {
    &CANONICAL_BLUE_BRAIN_FACING_CONTRACT_MAP
}

pub fn canonical_blue_brain_compute_handoff_map() -> &'static [BlueBrainComputeHandoffLane] {
    &CANONICAL_BLUE_BRAIN_COMPUTE_HANDOFF_MAP
}

pub fn canonical_blue_brain_integration_candidate_map(
) -> &'static [BlueBrainIntegrationCandidateLane] {
    &CANONICAL_BLUE_BRAIN_INTEGRATION_CANDIDATE_MAP
}

pub fn canonical_blue_brain_runtime_surface_map() -> &'static [BlueBrainRuntimeSurfaceLane] {
    &CANONICAL_BLUE_BRAIN_RUNTIME_SURFACE_MAP
}

pub fn canonical_blue_brain_runtime_phase_map() -> &'static [BlueBrainRuntimePhaseLane] {
    &CANONICAL_BLUE_BRAIN_RUNTIME_PHASE_MAP
}

pub fn canonical_blue_brain_transition_trigger_map() -> &'static [BlueBrainTransitionTriggerLane] {
    &CANONICAL_BLUE_BRAIN_TRANSITION_TRIGGER_MAP
}

pub fn canonical_blue_brain_context_memory_boundary_map(
) -> &'static [BlueBrainContextMemoryBoundaryLane] {
    &CANONICAL_BLUE_BRAIN_CONTEXT_MEMORY_BOUNDARY_MAP
}

pub fn canonical_blue_brain_runtime_feedback_map() -> &'static [BlueBrainRuntimeFeedbackLane] {
    &CANONICAL_BLUE_BRAIN_RUNTIME_FEEDBACK_MAP
}

pub fn canonical_blue_brain_context_memory_surface_map(
) -> &'static [BlueBrainContextMemorySurfaceLane] {
    &CANONICAL_BLUE_BRAIN_CONTEXT_MEMORY_SURFACE_MAP
}

pub fn canonical_blue_brain_context_update_lifecycle_map(
) -> &'static [BlueBrainContextUpdateLifecycleLane] {
    &CANONICAL_BLUE_BRAIN_CONTEXT_UPDATE_LIFECYCLE_MAP
}

pub fn canonical_blue_brain_memory_candidate_lifecycle_map(
) -> &'static [BlueBrainMemoryCandidateLifecycleLane] {
    &CANONICAL_BLUE_BRAIN_MEMORY_CANDIDATE_LIFECYCLE_MAP
}

pub fn canonical_blue_brain_memory_commit_boundary_map(
) -> &'static [BlueBrainMemoryCommitBoundaryLane] {
    &CANONICAL_BLUE_BRAIN_MEMORY_COMMIT_BOUNDARY_MAP
}

pub fn canonical_blue_brain_commit_eligibility_conditions_map(
) -> &'static [BlueBrainCommitEligibilityConditionLane] {
    &CANONICAL_BLUE_BRAIN_COMMIT_ELIGIBILITY_CONDITIONS_MAP
}

pub fn canonical_blue_brain_persistence_boundary_map() -> &'static [BlueBrainPersistenceBoundaryLane]
{
    &CANONICAL_BLUE_BRAIN_PERSISTENCE_BOUNDARY_MAP
}

pub fn canonical_blue_brain_future_memory_attachment_map(
) -> &'static [BlueBrainFutureMemoryAttachmentLane] {
    &CANONICAL_BLUE_BRAIN_FUTURE_MEMORY_ATTACHMENT_MAP
}

pub fn canonical_blue_brain_future_memory_handoff_state_map(
) -> &'static [BlueBrainFutureMemoryHandoffStateLane] {
    &CANONICAL_BLUE_BRAIN_FUTURE_MEMORY_HANDOFF_STATE_MAP
}

pub fn canonical_blue_brain_commit_result_semantics_map() -> &'static [BlueBrainCommitResultLane] {
    &CANONICAL_BLUE_BRAIN_COMMIT_RESULT_SEMANTICS_MAP
}

pub fn canonical_blue_brain_memory_commit_diagnostics_map(
) -> &'static [BlueBrainMemoryCommitDiagnosticLane] {
    &CANONICAL_BLUE_BRAIN_MEMORY_COMMIT_DIAGNOSTICS_MAP
}

pub fn canonical_blue_brain_reference_context_map() -> &'static [BlueBrainReferenceContextLane] {
    &CANONICAL_BLUE_BRAIN_REFERENCE_CONTEXT_MAP
}

pub fn canonical_blue_brain_control_attention_selection_map(
) -> &'static [BlueBrainControlAttentionSelectionLane] {
    &CANONICAL_BLUE_BRAIN_CONTROL_ATTENTION_SELECTION_MAP
}

pub fn canonical_blue_brain_context_evidence_priority_map(
) -> &'static [BlueBrainContextEvidencePriorityLane] {
    &CANONICAL_BLUE_BRAIN_CONTEXT_EVIDENCE_PRIORITY_MAP
}

pub fn canonical_blue_brain_candidate_deferral_lifecycle_map(
) -> &'static [BlueBrainCandidateDeferralLifecycleLane] {
    &CANONICAL_BLUE_BRAIN_CANDIDATE_DEFERRAL_LIFECYCLE_MAP
}

pub fn canonical_blue_brain_compute_trigger_arbitration_map(
) -> &'static [BlueBrainComputeTriggerArbitrationLane] {
    &CANONICAL_BLUE_BRAIN_COMPUTE_TRIGGER_ARBITRATION_MAP
}

pub fn canonical_blue_brain_planning_reasoning_candidate_map(
) -> &'static [BlueBrainPlanningReasoningCandidateLane] {
    &CANONICAL_BLUE_BRAIN_PLANNING_REASONING_CANDIDATE_MAP
}

pub fn canonical_drift_prevention_check_map() -> &'static [DriftPreventionCheckLane] {
    &CANONICAL_DRIFT_PREVENTION_CHECK_MAP
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_map_has_one_production_entry_lane() {
        let production_entries: Vec<_> = canonical_compute_reference_map()
            .iter()
            .filter(|lane| {
                lane.class == ComputeReferenceClass::CanonicalProduction
                    && lane.lane == "service_entry"
            })
            .collect();
        assert_eq!(production_entries.len(), 1);
        assert_eq!(
            production_entries[0].canonical_path,
            "service_surface::CanonicalComputeEntryPoint::submit"
        );
        assert!(production_entries[0]
            .shared_core_invariants
            .contains("request->job admission"));
    }

    #[test]
    fn canonical_map_keeps_compatibility_constructors_non_production() {
        assert!(canonical_compute_reference_map().iter().any(|lane| {
            lane.class == ComputeReferenceClass::InternalOrLegacy
                && lane
                    .canonical_path
                    .contains("backends::build_backend(kind=stub|candle)")
        }));
        assert!(!canonical_compute_reference_map().iter().any(|lane| {
            lane.class == ComputeReferenceClass::CanonicalProduction
                && lane
                    .canonical_path
                    .contains("build_backend(kind=stub|candle)")
        }));
    }

    #[test]
    fn canonical_map_lane_names_are_unique() {
        let mut lane_names: Vec<&str> = canonical_compute_reference_map()
            .iter()
            .map(|lane| lane.lane)
            .collect();
        lane_names.sort_unstable();
        lane_names.dedup();
        assert_eq!(lane_names.len(), canonical_compute_reference_map().len());
    }

    #[test]
    fn onboarding_summary_matches_pinned_canonical_constants() {
        let (backend, pack) = canonical_onboarding_reference_summary();
        assert_eq!(backend, "burn");
        assert_eq!(pack, "burn_toy_v1");
    }

    #[test]
    fn final_reference_line_covers_execution_rollout_replay_diagnostics_and_boundary() {
        let line = canonical_final_reference_line();
        assert!(line.execution_core.contains("submit -> compute_canonical"));
        assert!(line.execution_core.contains("result/fault/status"));
        assert!(line.execution_core.contains("execution_snapshot"));
        assert!(line
            .rollout_extension
            .contains("activation/fallback/rollback"));
        assert!(line.rollout_extension.contains("active production line"));
        assert!(line
            .replay_extension
            .contains("replay_preflight -> replay_with_entry"));
        assert!(line
            .replay_extension
            .contains("same result/fault/status core"));
        assert!(line
            .diagnostics_extension
            .contains("expert workflow surface -> same canonical core state"));
        assert!(line
            .cross_cutting_invariants
            .contains("blocked!=failed!=no_op"));
        assert!(line
            .cross_cutting_invariants
            .contains("partial/stale/caveated/degraded"));
        assert!(line.internal_boundary.contains("extension/internal only"));
    }

    #[test]
    fn internal_lanes_remain_non_canonical_in_reference_line() {
        assert!(canonical_compute_reference_map().iter().all(|lane| {
            let expected = lane.class != ComputeReferenceClass::InternalOrLegacy;
            is_canonical_core_or_extension_lane(lane.class) == expected
        }));
    }

    #[test]
    fn final_reference_doc_and_code_constants_are_kept_in_sync() {
        let doc = include_str!("../../../docs/final_reference_line_serie_j_v1.md");
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains(line.rollout_extension));
        assert!(doc.contains(line.replay_extension));
        assert!(doc.contains(line.diagnostics_extension));
        assert!(doc.contains(line.cross_cutting_invariants));
        assert!(doc.contains(line.internal_boundary));
        assert!(doc.contains("CANONICAL_RUNTIME_HANDOFF_SEMANTICS_V1"));
        assert!(doc.contains("complete"));
        assert!(doc.contains("partial"));
        assert!(doc.contains("caveated"));
        assert!(doc.contains("blocked"));
        assert!(doc.contains("compute_core_maintenance_boundary_serie_o_v1.md"));
    }

    #[test]
    fn production_readiness_evidence_pack_stays_aligned_with_canonical_core_contracts() {
        let doc =
            include_str!("../../../docs/final_production_readiness_evidence_pack_serie_j_v1.md");
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains(line.rollout_extension));
        assert!(doc.contains(line.replay_extension));
        assert!(doc.contains(line.diagnostics_extension));
        assert!(doc.contains(line.internal_boundary));
        assert!(doc.contains("CROSS_CUTTING_PRODUCTION_INVARIANTS_V1"));
        assert!(doc.contains("CANONICAL_RUNTIME_HANDOFF_SEMANTICS_V1"));
        assert!(doc.contains("stable production core"));
        assert!(doc.contains("production-usable but constrained"));
        assert!(doc.contains("partial / diagnostic"));
        assert!(doc.contains("intentionally deferred"));
    }
    #[test]
    fn serie_j_final_readiness_sweep_stays_aligned_with_canonical_production_line() {
        let doc = include_str!("../../../docs/real_compute_readiness_sweep_v27.md");
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("CROSS_CUTTING_PRODUCTION_INVARIANTS_V1"));
        assert!(doc.contains("CANONICAL_RUNTIME_HANDOFF_SEMANTICS_V1"));
        assert!(doc.contains("stable technical production line"));
        assert!(doc.contains("production-usable but constrained"));
        assert!(doc.contains("partial / diagnostic"));
        assert!(doc.contains("intentionally deferred"));
        assert!(doc.contains("Priorität jetzt: Serie K"));
    }

    #[test]
    fn integration_contract_view_keeps_minimal_classes_explicit() {
        let view = canonical_compute_integration_contract_view();
        assert!(view.iter().any(|lane| {
            lane.class == ComputeIntegrationContractClass::Execution
                && lane.boundary == ComputeIntegrationBoundary::OutwardFacing
        }));
        assert!(view.iter().any(|lane| {
            lane.class == ComputeIntegrationContractClass::DiagnosticsStatus
                && lane.boundary == ComputeIntegrationBoundary::OutwardFacing
        }));
        assert!(view.iter().any(|lane| {
            lane.class == ComputeIntegrationContractClass::EvidenceReference
                && lane.boundary == ComputeIntegrationBoundary::OutwardFacing
        }));
        assert!(view.iter().any(|lane| {
            lane.class == ComputeIntegrationContractClass::ExpertInternalOnly
                && lane.boundary == ComputeIntegrationBoundary::ExpertInternalOnly
        }));
    }

    #[test]
    fn outward_facing_integration_contracts_stay_pinned_to_final_execution_line() {
        let line = canonical_final_reference_line();
        let outward: Vec<_> = canonical_compute_integration_contract_view()
            .iter()
            .filter(|lane| is_outward_facing_compute_integration_boundary(lane.boundary))
            .collect();
        assert!(!outward.is_empty());
        assert!(outward
            .iter()
            .any(|lane| lane.class == ComputeIntegrationContractClass::Execution));
        assert!(line.execution_core.contains("submit -> compute_canonical"));
        assert!(line.execution_core.contains("result/fault/status"));
    }

    #[test]
    fn integration_contract_view_keeps_internal_paths_out_of_outward_boundary() {
        assert!(canonical_compute_integration_contract_view()
            .iter()
            .filter(|lane| lane.boundary == ComputeIntegrationBoundary::OutwardFacing)
            .all(|lane| {
                !lane
                    .canonical_anchor
                    .contains("build_backend(kind=stub|candle)")
                    && !lane.canonical_anchor.contains("domains/ai*")
            }));
    }

    #[test]
    fn serie_k_closure_doc_stays_aligned_with_outward_integration_boundaries() {
        let doc = include_str!("../../../docs/ops/serie_k_compute_facing_integration_closure.md");
        let line = canonical_final_reference_line();
        assert!(doc.contains("stable outward-facing integration surface"));
        assert!(doc.contains("integration-usable but constrained"));
        assert!(doc.contains("partial / internal-facing"));
        assert!(doc.contains("intentionally deferred"));
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("status_evidence_export_surface"));
        assert!(doc.contains("integration_hook_view"));
        assert!(doc.contains("run_operation_with_entry(..., ExpertHighTrust)"));
        assert!(doc.contains("build_backend kind=stub|candle"));
        assert!(doc.contains("Priorität: Serie L zuerst."));
    }

    #[test]
    fn serie_l_prompt2_boundary_doc_keeps_final_acceptance_line_explicit() {
        let doc = include_str!("../../../docs/real_compute_exit_boundary_serie_l_prompt2_v1.md");
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains(line.replay_extension));
        assert!(doc.contains(line.internal_boundary));
        assert!(doc.contains("stable"));
        assert!(doc.contains("constrained but accepted"));
        assert!(doc.contains("not accepted for final exit"));
        assert!(doc.contains("build_backend(kind=stub|candle)"));
    }

    #[test]
    fn domain_facing_consumer_map_keeps_alignment_classes_explicit() {
        let map = canonical_domain_facing_compute_consumer_map();
        assert!(map.iter().any(|c| {
            c.alignment == DomainFacingConsumerAlignment::AlignedCanonicalOutward
                && c.consumer == "ops_compute_probe"
        }));
        assert!(map
            .iter()
            .any(|c| c.alignment == DomainFacingConsumerAlignment::LegacyCompatPath));
        assert!(map.iter().any(|c| {
            c.alignment == DomainFacingConsumerAlignment::NeedsFinalIntegrationAdjustment
        }));
        assert!(map
            .iter()
            .any(|c| c.alignment == DomainFacingConsumerAlignment::InternalDevTestOnly));
    }

    #[test]
    fn outward_aligned_consumers_use_canonical_status_and_evidence_exports() {
        let aligned: Vec<_> = canonical_domain_facing_compute_consumer_map()
            .iter()
            .filter(|consumer| {
                consumer.alignment == DomainFacingConsumerAlignment::AlignedCanonicalOutward
            })
            .collect();
        assert!(!aligned.is_empty());
        assert!(aligned.iter().all(|consumer| {
            consumer
                .execution_contract_path
                .contains("CanonicalComputeEntryPoint::submit")
                && consumer.status_pattern
                    == DomainFacingStatusConsumptionPattern::CanonicalStatusConsumer
                && consumer.evidence_pattern
                    == DomainFacingEvidenceConsumptionPattern::CanonicalEvidenceReferenceConsumer
                && consumer
                    .status_diagnostics_path
                    .contains("status_evidence_export_surface")
                && consumer
                    .evidence_reference_path
                    .contains("status_evidence_export_surface")
        }));
    }

    #[test]
    fn completion_status_classifies_outward_vs_mixed_vs_internal_without_false_positive() {
        let map = canonical_domain_facing_compute_consumer_map();
        assert!(map.iter().any(|consumer| {
            consumer.completion_status == DomainFacingCompletionStatus::AlignedToFinalComputeLine
                && consumer.consumer == "ops_compute_probe"
        }));
        assert!(map.iter().any(|consumer| {
            consumer.completion_status == DomainFacingCompletionStatus::MostlyAlignedWithCaveats
                && consumer.consumer == "runtime_orchestrator_env_bootstrap"
        }));
        assert!(map.iter().any(|consumer| {
            consumer.completion_status == DomainFacingCompletionStatus::MixedTransitional
        }));
        assert!(map.iter().any(|consumer| {
            consumer.completion_status
                == DomainFacingCompletionStatus::InternalOnlyNotTrueOutwardConsumer
        }));
        assert!(map
            .iter()
            .filter(|consumer| {
                matches!(
                    consumer.completion_status,
                    DomainFacingCompletionStatus::MixedTransitional
                        | DomainFacingCompletionStatus::InternalOnlyNotTrueOutwardConsumer
                )
            })
            .all(|consumer| consumer.alignment
                != DomainFacingConsumerAlignment::AlignedCanonicalOutward));
    }

    #[test]
    fn only_ops_probe_is_marked_aligned_to_final_compute_line() {
        let aligned: Vec<_> = canonical_domain_facing_compute_consumer_map()
            .iter()
            .filter(|consumer| {
                consumer.completion_status
                    == DomainFacingCompletionStatus::AlignedToFinalComputeLine
            })
            .collect();
        assert_eq!(aligned.len(), 1);
        assert_eq!(aligned[0].consumer, "ops_compute_probe");
    }

    #[test]
    fn serie_m_consumer_map_doc_stays_in_sync_with_code() {
        let doc = include_str!("../../../docs/compute_consumer_integration_map_serie_m_v1.md");
        for consumer in canonical_domain_facing_compute_consumer_map() {
            assert!(doc.contains(consumer.consumer));
            assert!(doc.contains(consumer.repo_surface));
        }
        assert!(doc.contains("aligned_canonical_outward"));
        assert!(doc.contains("legacy_compat_path"));
        assert!(doc.contains("needs_final_integration_adjustment"));
        assert!(doc.contains("internal_dev_test_only"));
        assert!(doc.contains("aligned_to_final_compute_line"));
        assert!(doc.contains("mostly_aligned_with_caveats"));
        assert!(doc.contains("mixed_transitional"));
        assert!(doc.contains("internal_only_not_true_outward_consumer"));
        assert!(doc.contains("canonical_status_consumer"));
        assert!(doc.contains("canonical_evidence_reference_consumer"));
        assert!(doc.contains("mixed_legacy_consumption_pattern"));
    }

    #[test]
    fn serie_n_broader_system_map_stays_pinned_to_final_compute_line_and_priority_view() {
        let doc = include_str!("../../../docs/broader_system_integration_map_serie_n_v1.md");
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("status_evidence_export_surface"));
        assert!(doc.contains("integration_hook_view"));
        assert!(doc.contains(line.internal_boundary));

        assert!(doc.contains("high_leverage_aligned_candidate"));
        assert!(doc.contains("plausible_but_caveated_candidate"));
        assert!(doc.contains("low_value_or_legacy_driven_candidate"));
        assert!(doc.contains("not_worth_broader_integration_now"));

        assert!(doc.contains("ops_compute_probe"));
        assert!(doc.contains("runtime_orchestrator_env_bootstrap"));
        assert!(doc.contains("replay_diff_backend_recompute"));
        assert!(doc.contains("bench_compute_subcommand"));
        assert!(doc.contains("domains_ai_compat_lane"));
        assert!(doc.contains("Keine Wunschliste"));
        assert!(doc.contains("already_aligned"));
        assert!(doc.contains("first_post_core_aligned"));
        assert!(doc.contains("broader_review_candidate"));
        assert!(doc.contains("not_pursued_now"));
        assert!(doc.contains("nicht vorweg implementiert"));
        assert!(doc.contains("compute_core_maintenance_boundary_serie_o_v1.md"));
    }

    #[test]
    fn serie_n_priority_view_does_not_mark_legacy_or_internal_paths_as_aligned() {
        let doc = include_str!("../../../docs/broader_system_integration_map_serie_n_v1.md");
        let map = canonical_domain_facing_compute_consumer_map();
        let legacy_or_internal: Vec<_> = map
            .iter()
            .filter(|consumer| {
                matches!(
                    consumer.consumer,
                    "domains_ai_compat_lane"
                        | "bench_compute_subcommand"
                        | "replay_diff_backend_recompute"
                )
            })
            .collect();
        assert!(!legacy_or_internal.is_empty());
        assert!(legacy_or_internal.iter().all(|consumer| {
            consumer.alignment != DomainFacingConsumerAlignment::AlignedCanonicalOutward
        }));
        assert!(doc.contains("low_value_or_legacy_driven_candidate"));
        assert!(doc.contains("not_worth_broader_integration_now"));
    }

    #[test]
    fn blue_brain_integration_map_keeps_minimal_classes_and_outward_contract_basis_explicit() {
        let map = canonical_blue_brain_integration_map();
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainIntegrationClass::RealBlueBrainCoreCandidate
                && lane.surface == "runtime_orchestrator_stateful_loop"
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainIntegrationClass::BlueBrainAdjacentComputeConsumer
                && lane.surface == "ops_compute_probe"
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainIntegrationClass::IndirectOrCompatibilityTouchingSurface
        }));
        assert!(map.iter().any(|lane| {
            lane.class
                == BlueBrainIntegrationClass::InternalOnlyOrNotMeaningfulForBlueBrainIntegration
        }));
        assert!(map.iter().all(|lane| {
            lane.execution_contract_path
                .contains("CanonicalComputeEntryPoint::submit")
                || lane
                    .status_diagnostics_contract_path
                    .contains("status_evidence_export_surface")
                || lane
                    .integration_safe_hook_posture
                    .contains("integration_hook_view")
                || lane.class != BlueBrainIntegrationClass::RealBlueBrainCoreCandidate
        }));
    }

    #[test]
    fn serie_bb1_blue_brain_map_doc_stays_pinned_to_canonical_compute_contracts() {
        let doc = include_str!("../../../docs/blue_brain_integration_map_serie_bb1_prompt1_v1.md");
        let line = canonical_final_reference_line();

        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("compute_execution_contract"));
        assert!(doc.contains("compute_status_diagnostics_contract"));
        assert!(doc.contains("compute_evidence_reference_contract"));
        assert!(doc.contains("integration_hook_view"));

        assert!(doc.contains("real_blue_brain_core_candidate"));
        assert!(doc.contains("blue_brain_adjacent_compute_consumer"));
        assert!(doc.contains("indirect_or_compatibility_touching_surface"));
        assert!(doc.contains("internal_only_or_not_meaningful_for_blue_brain_integration"));

        assert!(doc.contains("runtime_orchestrator_stateful_loop"));
        assert!(doc.contains("ops_compute_probe"));
        assert!(doc.contains("replay_diff_backend_recompute"));
        assert!(doc.contains("domains_ai_compat_lane"));
        assert!(doc.contains("bench_compute_subcommand"));
        assert!(doc.contains("runtime_hooks_and_frame_helpers"));
        assert!(doc.contains("keine zweite Integrationssprache"));
        assert!(doc.contains("keine zweite Wahrheitsquelle"));
    }

    #[test]
    fn blue_brain_facing_contract_map_keeps_state_inference_status_evidence_split_explicit() {
        let map = canonical_blue_brain_facing_contract_map();
        assert_eq!(map.len(), 5);
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainFacingContractClass::InferenceFacing));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainFacingContractClass::StateFacing));
        assert!(map
            .iter()
            .any(|lane| { lane.class == BlueBrainFacingContractClass::StatusHealthTrustFacing }));
        assert!(map
            .iter()
            .any(|lane| { lane.class == BlueBrainFacingContractClass::EvidenceReferenceFacing }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainFacingContractClass::ExpertInternalOnlyNonBlueBrain
        }));
    }

    #[test]
    fn blue_brain_inference_contract_stays_pinned_to_canonical_submit_and_fault_status_core() {
        let lane = canonical_blue_brain_facing_contract_map()
            .iter()
            .find(|lane| lane.class == BlueBrainFacingContractClass::InferenceFacing)
            .expect("inference-facing lane");
        assert!(lane
            .canonical_anchor
            .contains("CanonicalComputeEntryPoint::submit"));
        assert!(lane
            .allowed_semantics
            .contains("submit -> compute_canonical -> result/fault/status"));
        assert!(lane.excluded_semantics.contains("no direct build_backend"));
    }

    #[test]
    fn blue_brain_status_and_evidence_contracts_reuse_canonical_export_surface() {
        let status_lane = canonical_blue_brain_facing_contract_map()
            .iter()
            .find(|lane| lane.class == BlueBrainFacingContractClass::StatusHealthTrustFacing)
            .expect("status-facing lane");
        assert!(status_lane
            .canonical_anchor
            .contains("status_evidence_export_surface"));
        assert!(status_lane
            .allowed_semantics
            .contains("current/partial/stale/caveated/degraded"));

        let evidence_lane = canonical_blue_brain_facing_contract_map()
            .iter()
            .find(|lane| lane.class == BlueBrainFacingContractClass::EvidenceReferenceFacing)
            .expect("evidence-facing lane");
        assert!(evidence_lane
            .canonical_anchor
            .contains("status_evidence_export_surface"));
        assert!(evidence_lane
            .allowed_semantics
            .contains("partial/caveated evidence"));
    }

    #[test]
    fn blue_brain_expert_internal_only_lane_is_explicitly_non_contract() {
        let lane = canonical_blue_brain_facing_contract_map()
            .iter()
            .find(|lane| lane.class == BlueBrainFacingContractClass::ExpertInternalOnlyNonBlueBrain)
            .expect("expert/internal lane");
        assert!(lane.canonical_anchor.contains("run_operation_with_entry"));
        assert!(lane
            .canonical_anchor
            .contains("build_backend(kind=stub|candle|worker)"));
        assert!(lane.excluded_semantics.contains("must not be presented"));
    }

    #[test]
    fn serie_bb1_prompt2_contract_doc_stays_pinned_to_single_compute_contract_language() {
        let doc = include_str!("../../../docs/blue_brain_facing_contracts_serie_bb1_prompt2_v1.md");
        let line = canonical_final_reference_line();

        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("blue_brain_inference_facing_execution_contract"));
        assert!(doc.contains("blue_brain_state_facing_context_reference_contract"));
        assert!(doc.contains("blue_brain_status_health_trust_contract"));
        assert!(doc.contains("blue_brain_evidence_reference_contract"));
        assert!(doc.contains("blue_brain_expert_internal_only_non_contract"));
        assert!(doc.contains("current / partial / stale / caveated / degraded"));
        assert!(doc.contains("status_evidence_export_surface"));
        assert!(doc.contains("integration_hook_view"));
        assert!(doc.contains("no second execution world"));
        assert!(doc.contains("keine zweite Wahrheitsquelle"));
    }

    #[test]
    fn blue_brain_handoff_map_keeps_minimal_canonical_split_and_non_canonical_boundary_explicit() {
        let map = canonical_blue_brain_compute_handoff_map();
        assert_eq!(map.len(), 5);
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainComputeHandoffClass::InferenceHandoff));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainComputeHandoffClass::StatusDiagnosticsHandoff));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainComputeHandoffClass::EvidenceReferenceHandoff));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainComputeHandoffClass::StateAdjacentReferenceHandoff));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainComputeHandoffClass::ExpertInternalOnlyNonCanonicalHandoff
        }));
    }

    #[test]
    fn blue_brain_handoff_inference_status_and_evidence_lanes_stay_on_canonical_compute_line() {
        let map = canonical_blue_brain_compute_handoff_map();
        let inference = map
            .iter()
            .find(|lane| lane.class == BlueBrainComputeHandoffClass::InferenceHandoff)
            .expect("inference handoff lane");
        assert!(inference
            .canonical_transition
            .contains("CanonicalComputeEntryPoint::submit"));
        assert!(inference
            .return_payload_shape
            .contains("result/fault/status"));

        let status = map
            .iter()
            .find(|lane| lane.class == BlueBrainComputeHandoffClass::StatusDiagnosticsHandoff)
            .expect("status handoff lane");
        assert!(status
            .canonical_transition
            .contains("status_evidence_export_surface(status)"));
        assert!(status
            .return_payload_shape
            .contains("current|partial|stale|caveated|degraded"));

        let evidence = map
            .iter()
            .find(|lane| lane.class == BlueBrainComputeHandoffClass::EvidenceReferenceHandoff)
            .expect("evidence handoff lane");
        assert!(evidence
            .canonical_transition
            .contains("status_evidence_export_surface(evidence refs)"));
        assert!(evidence.return_payload_shape.contains("partial/caveated"));
    }

    #[test]
    fn serie_bb1_prompt3_handoff_doc_stays_pinned_to_canonical_handoff_map() {
        let doc = include_str!("../../../docs/blue_brain_compute_handoffs_serie_bb1_prompt3_v1.md");
        let line = canonical_final_reference_line();

        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("blue_brain_to_compute_inference_handoff"));
        assert!(doc.contains("blue_brain_to_compute_status_diagnostics_handoff"));
        assert!(doc.contains("blue_brain_to_compute_evidence_reference_handoff"));
        assert!(doc.contains("blue_brain_to_compute_state_adjacent_reference_handoff"));
        assert!(doc.contains("blue_brain_non_canonical_expert_internal_handoff"));
        assert!(doc.contains("current / partial / stale / caveated / degraded"));
        assert!(doc.contains("status_evidence_export_surface"));
        assert!(doc.contains("runtime_handoff_state_from_evidence"));
        assert!(doc.contains("runtime_handoff_state_from_action_code"));
        assert!(doc.contains("keine Workflow-Engine"));
        assert!(doc.contains("keine zweite Wahrheitsquelle"));
    }

    #[test]
    fn blue_brain_candidate_map_keeps_minimal_candidate_classes_and_selects_one_real_candidate() {
        let map = canonical_blue_brain_integration_candidate_map();
        assert_eq!(map.len(), 4);
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainIntegrationCandidateClass::IntegrationReadyCandidate
                && lane.surface == "ops_compute_probe"
        }));
        let selected = map
            .iter()
            .find(|lane| lane.surface == "runtime_orchestrator_stateful_loop")
            .expect("runtime_orchestrator_stateful_loop candidate lane");
        assert_eq!(
            selected.class,
            BlueBrainIntegrationCandidateClass::PlausibleWithCaveats
        );
        assert!(selected
            .candidate_selection_posture
            .contains("selected_first_real_blue_brain_integration_candidate"));
        assert!(selected
            .inference_contract_binding
            .contains("CanonicalComputeEntryPoint::submit"));
        assert!(selected
            .status_handoff_binding
            .contains("status_evidence_export_surface(status)"));
        assert!(selected
            .evidence_handoff_binding
            .contains("status_evidence_export_surface(evidence refs)"));
        assert!(selected
            .state_adjacent_binding
            .contains("runtime_handoff_state_from_evidence"));
        assert!(selected
            .excluded_internal_or_legacy_paths
            .contains("build_backend(kind=stub|candle|worker)"));
    }

    #[test]
    fn serie_bb1_prompt4_candidate_doc_stays_pinned_to_canonical_contracts_and_handoffs() {
        let doc =
            include_str!("../../../docs/blue_brain_integration_candidate_serie_bb1_prompt4_v1.md");
        let line = canonical_final_reference_line();

        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("integration-ready candidate"));
        assert!(doc.contains("plausible with caveats"));
        assert!(doc.contains("mixed/transitional candidate"));
        assert!(doc.contains("not a real Blue-Brain integration candidate now"));
        assert!(doc.contains("runtime_orchestrator_stateful_loop"));
        assert!(doc.contains("selected_first_real_blue_brain_integration_candidate"));
        assert!(doc.contains("blue_brain_to_compute_inference_handoff"));
        assert!(doc.contains("blue_brain_to_compute_status_diagnostics_handoff"));
        assert!(doc.contains("blue_brain_to_compute_evidence_reference_handoff"));
        assert!(doc.contains("blue_brain_to_compute_state_adjacent_reference_handoff"));
        assert!(doc.contains("build_backend(kind=stub|candle|worker)"));
        assert!(doc.contains("keine zweite Wahrheitsquelle"));
    }

    #[test]
    fn serie_bb1_prompt5_readiness_doc_keeps_closure_matrix_and_baseline_pinned() {
        let doc = include_str!("../../../docs/blue_brain_readiness_sweep_serie_bb1_prompt5_v1.md");
        let line = canonical_final_reference_line();

        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("stable Blue-Brain integration foundation"));
        assert!(doc.contains("integration-usable with caveats"));
        assert!(doc.contains("preparatory / not yet a true integration path"));
        assert!(doc.contains("intentionally deferred"));

        assert!(doc.contains("runtime_orchestrator_stateful_loop"));
        assert!(doc.contains("selected_first_real_blue_brain_integration_candidate"));

        assert!(doc.contains("blue_brain_to_compute_inference_handoff"));
        assert!(doc.contains("blue_brain_to_compute_status_diagnostics_handoff"));
        assert!(doc.contains("blue_brain_to_compute_evidence_reference_handoff"));
        assert!(doc.contains("blue_brain_to_compute_state_adjacent_reference_handoff"));

        assert!(doc.contains("Serie BB2"));
        assert!(doc.contains("Priorität 1: Serie BB2"));
        assert!(
            doc.contains("kein Rückfall auf compute-interne, legacy- oder helper-dominierte Pfade")
        );
    }

    #[test]
    fn blue_brain_runtime_surface_map_keeps_five_minimal_runtime_classes_explicit() {
        let map = canonical_blue_brain_runtime_surface_map();
        assert_eq!(map.len(), 5);
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainRuntimeSurfaceClass::StateBearingSurface));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainRuntimeSurfaceClass::InferenceBearingSurface));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainRuntimeSurfaceClass::StatusHealthTrustFacingSurface
        }));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainRuntimeSurfaceClass::EvidenceReplayFacingSurface));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainRuntimeSurfaceClass::InternalOnlyRuntimeControlSurface
        }));
    }

    #[test]
    fn blue_brain_runtime_surface_map_stays_pinned_to_final_compute_line_without_internal_leak() {
        let map = canonical_blue_brain_runtime_surface_map();
        let inference_lane = map
            .iter()
            .find(|lane| lane.class == BlueBrainRuntimeSurfaceClass::InferenceBearingSurface)
            .expect("inference-bearing runtime lane");
        assert!(inference_lane
            .canonical_anchor
            .contains("CanonicalComputeEntryPoint::submit"));
        assert!(inference_lane
            .compute_line_binding
            .contains("submit -> compute_canonical -> result/fault/status"));

        let status_lane = map
            .iter()
            .find(|lane| lane.class == BlueBrainRuntimeSurfaceClass::StatusHealthTrustFacingSurface)
            .expect("status runtime lane");
        assert!(status_lane
            .canonical_anchor
            .contains("status_evidence_export_surface(status)"));

        let evidence_lane = map
            .iter()
            .find(|lane| lane.class == BlueBrainRuntimeSurfaceClass::EvidenceReplayFacingSurface)
            .expect("evidence runtime lane");
        assert!(evidence_lane
            .runtime_scope
            .contains("sufficient|partial|caveated|insufficient"));

        let internal_lane = map
            .iter()
            .find(|lane| {
                lane.class == BlueBrainRuntimeSurfaceClass::InternalOnlyRuntimeControlSurface
            })
            .expect("internal runtime control lane");
        assert!(internal_lane
            .boundary_guard
            .contains("explicitly non-canonical Blue-Brain runtime surface"));
    }

    #[test]
    fn blue_brain_runtime_phase_map_keeps_minimal_runtime_phases_and_caveat_state_explicit() {
        let phases = canonical_blue_brain_runtime_phase_map();
        assert_eq!(phases.len(), 5);
        assert!(phases
            .iter()
            .any(|lane| lane.class == BlueBrainRuntimePhaseClass::StateContextAvailable));
        assert!(phases
            .iter()
            .any(|lane| lane.class == BlueBrainRuntimePhaseClass::ComputeInvocationRequested));
        assert!(phases
            .iter()
            .any(|lane| lane.class == BlueBrainRuntimePhaseClass::ComputeResultIntegrated));
        assert!(phases
            .iter()
            .any(|lane| lane.class == BlueBrainRuntimePhaseClass::StatusEvidenceObserved));
        assert!(phases.iter().any(|lane| {
            lane.class == BlueBrainRuntimePhaseClass::CaveatedOrDegradedOrPartialRuntimeState
        }));
        assert!(phases.iter().any(|lane| {
            lane.lane == "blue_brain_phase_caveated_degraded_partial_runtime_state"
                && lane
                    .canonical_inputs
                    .contains("current|partial|stale|caveated|degraded")
        }));
    }

    #[test]
    fn serie_bb2_prompt1_runtime_surface_doc_stays_pinned_to_runtime_surface_and_phase_maps() {
        let doc =
            include_str!("../../../docs/blue_brain_state_runtime_surface_serie_bb2_prompt1_v1.md");
        let line = canonical_final_reference_line();

        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("blue_brain_state_bearing_surface"));
        assert!(doc.contains("blue_brain_inference_bearing_surface"));
        assert!(doc.contains("blue_brain_status_health_trust_surface"));
        assert!(doc.contains("blue_brain_evidence_replay_facing_surface"));
        assert!(doc.contains("blue_brain_internal_only_runtime_control_surface"));
        assert!(doc.contains("blue_brain_phase_state_context_available"));
        assert!(doc.contains("blue_brain_phase_compute_invocation_requested"));
        assert!(doc.contains("blue_brain_phase_compute_result_integrated"));
        assert!(doc.contains("blue_brain_phase_status_evidence_observed"));
        assert!(doc.contains("blue_brain_phase_caveated_degraded_partial_runtime_state"));
        assert!(doc.contains("keine zweite Compute-Semantik"));
        assert!(doc.contains("keine Workflow-Engine"));
        assert!(doc.contains("keine zweite Wahrheitsquelle"));
    }

    #[test]
    fn blue_brain_transition_trigger_map_keeps_minimal_transition_classes_explicit() {
        let map = canonical_blue_brain_transition_trigger_map();
        assert_eq!(map.len(), 11);
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainTransitionTriggerClass::PureStateTransition));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainTransitionTriggerClass::ComputeTriggeringTransition
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainTransitionTriggerClass::EvidenceStatusUpdateTransition
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainTransitionTriggerClass::InternalOnlyOrNonCanonicalTransition
        }));
    }

    #[test]
    fn blue_brain_transition_trigger_points_stay_on_outward_contracts_and_block_internal_defaults()
    {
        let map = canonical_blue_brain_transition_trigger_map();
        let context_available = map
            .iter()
            .find(|lane| lane.lane == "blue_brain_transition_context_available")
            .expect("context available transition lane");
        assert!(context_available
            .trigger_point
            .contains("no compute trigger implied"));
        assert!(context_available
            .non_canonical_boundary
            .contains("not be interpreted as persistent memory commit"));

        let context_used_trigger = map
            .iter()
            .find(|lane| lane.lane == "blue_brain_transition_context_used_for_compute_trigger")
            .expect("context used for compute-trigger lane");
        assert!(context_used_trigger
            .canonical_contract_binding
            .contains("CanonicalComputeEntryPoint::submit"));
        assert!(context_used_trigger
            .reference_continuity
            .contains("not treated as memory writes"));

        let context_trigger = map
            .iter()
            .find(|lane| {
                lane.lane == "blue_brain_transition_compute_trigger_from_context_availability"
            })
            .expect("context availability trigger lane");
        assert!(context_trigger
            .canonical_contract_binding
            .contains("CanonicalComputeEntryPoint::submit"));

        let blocked = map
            .iter()
            .find(|lane| {
                lane.lane == "blue_brain_transition_compute_trigger_blocked_insufficient_context"
            })
            .expect("blocked trigger lane");
        assert!(blocked
            .trigger_point
            .contains("blocked due to insufficient context/state"));
        assert!(blocked
            .canonical_contract_binding
            .contains("status_evidence_export_surface"));

        let suppressed = map
            .iter()
            .find(|lane| {
                lane.lane == "blue_brain_transition_compute_trigger_suppressed_internal_only_path"
            })
            .expect("suppressed trigger lane");
        assert!(suppressed.trigger_point.contains("trigger suppressed"));
        assert!(suppressed
            .non_canonical_boundary
            .contains("no default Blue-Brain trigger authority"));

        let status_only = map
            .iter()
            .find(|lane| {
                lane.lane == "blue_brain_transition_status_evidence_update_without_compute_trigger"
            })
            .expect("status-only transition lane");
        assert!(status_only
            .trigger_point
            .contains("without new compute trigger"));
        assert!(status_only
            .canonical_contract_binding
            .contains("CanonicalComputeEntryPoint::status"));

        let evidence_no_commit = map
            .iter()
            .find(|lane| {
                lane.lane == "blue_brain_transition_evidence_observed_without_memory_commit"
            })
            .expect("evidence observed without memory commit lane");
        assert!(evidence_no_commit
            .trigger_point
            .contains("without memory commit"));
        assert!(evidence_no_commit
            .non_canonical_boundary
            .contains("must not be represented as memory persistence"));

        let memory_adjacent = map
            .iter()
            .find(|lane| {
                lane.lane
                    == "blue_brain_transition_memory_adjacent_candidate_identified_not_committed"
            })
            .expect("memory-adjacent candidate lane");
        assert!(memory_adjacent.trigger_point.contains("no memory commit"));
        assert!(memory_adjacent
            .non_canonical_boundary
            .contains("no long-term memory persistence"));
    }

    #[test]
    fn serie_bb2_prompt2_transition_trigger_doc_stays_pinned_to_canonical_map_and_boundaries() {
        let doc =
            include_str!("../../../docs/blue_brain_transition_trigger_map_serie_bb2_prompt2_v1.md");
        let line = canonical_final_reference_line();

        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("blue_brain_transition_state_context_refreshed"));
        assert!(doc.contains("blue_brain_transition_context_available"));
        assert!(doc.contains("blue_brain_transition_context_used_for_compute_trigger"));
        assert!(doc.contains("blue_brain_transition_compute_trigger_from_context_availability"));
        assert!(doc.contains("blue_brain_transition_compute_trigger_from_inference_required"));
        assert!(doc.contains("blue_brain_transition_compute_trigger_blocked_insufficient_context"));
        assert!(doc.contains("blue_brain_transition_compute_trigger_suppressed_internal_only_path"));
        assert!(doc.contains("blue_brain_transition_compute_result_integrated"));
        assert!(doc.contains("blue_brain_transition_evidence_observed_without_memory_commit"));
        assert!(doc
            .contains("blue_brain_transition_memory_adjacent_candidate_identified_not_committed"));
        assert!(
            doc.contains("blue_brain_transition_status_evidence_update_without_compute_trigger")
        );
        assert!(doc.contains("keine Workflow- oder State-Machine-Plattform"));
        assert!(doc.contains("keine zweite Execution-Sprache"));
        assert!(doc.contains("keine zweite Wahrheitsquelle"));
    }

    #[test]
    fn blue_brain_context_memory_boundary_map_keeps_compute_context_memory_split_explicit() {
        let map = canonical_blue_brain_context_memory_boundary_map();
        assert_eq!(map.len(), 7);
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainContextMemoryBoundaryClass::PureComputeConsumer));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainContextMemoryBoundaryClass::ContextBearingSurface));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainContextMemoryBoundaryClass::MemoryAdjacentSurface));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainContextMemoryBoundaryClass::EvidenceReferenceConsumer
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainContextMemoryBoundaryClass::InternalOnlyOrNonCanonicalContextPath
        }));
    }

    #[test]
    fn blue_brain_context_memory_boundary_map_prevents_reference_and_memory_confusion() {
        let map = canonical_blue_brain_context_memory_boundary_map();
        let evidence_lane = map
            .iter()
            .find(|lane| lane.lane == "blue_brain_evidence_reference_consumer_surface")
            .expect("evidence reference consumer lane");
        assert!(evidence_lane
            .memory_posture
            .contains("no memory persistence implied"));
        assert!(evidence_lane
            .boundary_guard
            .contains("not memory records and not memory commits"));

        let replay_lane = map
            .iter()
            .find(|lane| lane.lane == "blue_brain_replay_reference_basis_surface")
            .expect("replay reference basis lane");
        assert!(replay_lane.memory_posture.contains("not persistent memory"));
        assert!(replay_lane
            .boundary_guard
            .contains("down-mapped to outward references"));

        let internal_lane = map
            .iter()
            .find(|lane| lane.lane == "blue_brain_internal_or_expert_only_context_path")
            .expect("internal path lane");
        assert!(internal_lane
            .boundary_guard
            .contains("non-canonical boundary"));
    }

    #[test]
    fn serie_bb2_prompt3_context_memory_boundary_doc_stays_pinned_to_code_map() {
        let doc = include_str!(
            "../../../docs/blue_brain_context_memory_boundary_serie_bb2_prompt3_v1.md"
        );
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_CONTEXT_MEMORY_BOUNDARY_MAP"));
        assert!(doc.contains("pure_compute_consumer"));
        assert!(doc.contains("context_bearing_blue_brain_surface"));
        assert!(doc.contains("memory_adjacent_blue_brain_surface"));
        assert!(doc.contains("evidence_reference_consumer"));
        assert!(doc.contains("internal_only_or_non_canonical_context_path"));
        assert!(doc.contains("blue_brain_transition_context_available"));
        assert!(doc.contains("blue_brain_transition_context_used_for_compute_trigger"));
        assert!(doc.contains("blue_brain_transition_compute_result_integrated"));
        assert!(doc.contains("blue_brain_transition_evidence_observed_without_memory_commit"));
        assert!(doc
            .contains("blue_brain_transition_memory_adjacent_candidate_identified_not_committed"));
        assert!(doc.contains("no memory persistence implied"));
        assert!(doc.contains("keine Memory-Architektur"));
    }

    #[test]
    fn blue_brain_context_memory_surface_map_keeps_bb3_classes_explicit() {
        let map = canonical_blue_brain_context_memory_surface_map();
        assert_eq!(map.len(), 13);
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainContextMemorySurfaceClass::TransientRuntimeContext));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainContextMemorySurfaceClass::EvidenceBackedContext));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainContextMemorySurfaceClass::ReplayReferenceBackedContext
        }));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainContextMemorySurfaceClass::MemoryAdjacentCandidate));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainContextMemorySurfaceClass::PersistedMemory));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainContextMemorySurfaceClass::NonCanonicalInternalOnlyMemoryLikePath
        }));
    }

    #[test]
    fn blue_brain_context_memory_surface_map_keeps_memory_semantics_non_ambiguous() {
        let map = canonical_blue_brain_context_memory_surface_map();
        let persisted_lane = map
            .iter()
            .find(|lane| lane.lane == "blue_brain_persisted_memory_none_in_current_baseline")
            .expect("persisted memory lane");
        assert!(persisted_lane
            .persistence_binding
            .contains("not implemented"));
        assert!(persisted_lane
            .canonical_guard
            .contains("accidental reinterpretation"));

        let evidence_backed = map
            .iter()
            .find(|lane| lane.lane == "blue_brain_evidence_backed_context_status_export")
            .expect("evidence-backed context lane");
        assert!(evidence_backed
            .persistence_binding
            .contains("no automatic memory write"));
        assert!(evidence_backed
            .canonical_guard
            .contains("not memory commits"));

        let candidate = map
            .iter()
            .find(|lane| lane.lane == "blue_brain_memory_adjacent_candidate_not_committed")
            .expect("memory-adjacent candidate lane");
        assert!(candidate
            .persistence_binding
            .contains("not persisted and not committed"));
        assert!(candidate
            .canonical_guard
            .contains("actual memory persistence"));

        let replay_caveated = map
            .iter()
            .find(|lane| {
                lane.lane == "blue_brain_replay_reference_backed_context_caveated_or_partial"
            })
            .expect("caveated replay/reference lane");
        assert!(replay_caveated
            .context_shape
            .contains("partial/missing fidelity"));
        assert!(replay_caveated
            .persistence_binding
            .contains("never a persistence write path"));

        let lifecycle_lane = map
            .iter()
            .find(|lane| lane.lane == "blue_brain_transient_runtime_context_updated_then_discarded")
            .expect("transient context lifecycle lane");
        assert!(lifecycle_lane
            .context_shape
            .contains("updated by result/evidence feedback"));
        assert!(lifecycle_lane
            .persistence_binding
            .contains("no persisted memory"));
    }

    #[test]
    fn serie_bb3_prompt1_context_memory_surface_doc_stays_pinned_to_code_map() {
        let doc =
            include_str!("../../../docs/blue_brain_context_memory_surface_serie_bb3_prompt1_v1.md");
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_CONTEXT_MEMORY_SURFACE_MAP"));
        assert!(doc.contains("transient_runtime_context"));
        assert!(doc.contains("evidence_backed_context"));
        assert!(doc.contains("replay_reference_backed_context"));
        assert!(doc.contains("memory_adjacent_candidate"));
        assert!(doc.contains("persisted_memory"));
        assert!(doc.contains("non_canonical_internal_only_memory_like_path"));
        assert!(doc.contains("blue_brain_persisted_memory_none_in_current_baseline"));
        assert!(doc.contains("blue_brain_transient_runtime_context_available_for_transition"));
        assert!(doc.contains("blue_brain_transient_runtime_context_used_for_compute_trigger"));
        assert!(doc.contains("blue_brain_transient_runtime_context_updated_then_discarded"));
        assert!(doc.contains("blue_brain_evidence_backed_context_attached_or_caveated"));
        assert!(doc.contains("blue_brain_replay_reference_backed_context_caveated_or_partial"));
        assert!(doc.contains("blue_brain_memory_adjacent_candidate_derived_sources_uncommitted"));
        assert!(doc.contains("insufficient"));
        assert!(doc.contains("compute outputs and evidence feedback"));
        assert!(doc.contains("kein Memory-Engine-Bau"));
    }

    #[test]
    fn blue_brain_context_update_lifecycle_map_keeps_states_structurally_distinct() {
        let map = canonical_blue_brain_context_update_lifecycle_map();
        assert_eq!(map.len(), 9);
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainContextUpdateLifecycleClass::ContextInitialized));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainContextUpdateLifecycleClass::UpdatedFromComputeResult
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainContextUpdateLifecycleClass::UpdatedFromEvidenceReference
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainContextUpdateLifecycleClass::UpdatedFromReplayReference
        }));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainContextUpdateLifecycleClass::ContextUnchanged));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainContextUpdateLifecycleClass::UpdateBlockedOrInsufficient
        }));

        let update_only = map
            .iter()
            .find(|lane| lane.lane == "blue_brain_context_updated_from_compute_result")
            .expect("update-only lane");
        assert!(update_only
            .candidate_effect
            .contains("no candidate required by default"));

        let update_plus_candidate = map
            .iter()
            .find(|lane| lane.lane == "blue_brain_context_updated_and_candidate_proposed")
            .expect("update-plus-candidate lane");
        assert!(update_plus_candidate
            .candidate_effect
            .contains("update plus candidate proposal"));

        let candidate_only = map
            .iter()
            .find(|lane| lane.lane == "blue_brain_candidate_only_without_context_mutation")
            .expect("candidate-only lane");
        assert!(candidate_only
            .update_semantics
            .contains("without mutating current runtime context"));
    }

    #[test]
    fn blue_brain_memory_candidate_lifecycle_map_keeps_no_persistence_boundary_explicit() {
        let map = canonical_blue_brain_memory_candidate_lifecycle_map();
        assert_eq!(map.len(), 13);
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainMemoryCandidateLifecycleClass::CandidateProposed));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainMemoryCandidateLifecycleClass::CandidateEvidenceBacked
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainMemoryCandidateLifecycleClass::CandidateContextDerived
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainMemoryCandidateLifecycleClass::CandidateComputeResultDerived
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainMemoryCandidateLifecycleClass::AcceptedForFutureMemoryHandling
        }));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainMemoryCandidateLifecycleClass::CandidateRejected));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainMemoryCandidateLifecycleClass::CandidateStale));
        assert!(map.iter().any(
            |lane| lane.class == BlueBrainMemoryCandidateLifecycleClass::CandidateInsufficient
        ));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainMemoryCandidateLifecycleClass::PersistenceUnavailableOrDeferred
        }));
        assert!(map.iter().any(|lane| {
            lane.class
                == BlueBrainMemoryCandidateLifecycleClass::PersistencePerformedViaRealPathOnly
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainMemoryCandidateLifecycleClass::NoPersistencePerformed
        }));

        let accepted = map
            .iter()
            .find(|lane| lane.lane == "blue_brain_candidate_accepted_for_future_memory_handling")
            .expect("accepted-for-future lane");
        assert!(accepted
            .persistence_semantics
            .contains("still performs no persistence"));

        let no_persist = map
            .iter()
            .find(|lane| lane.lane == "blue_brain_candidate_no_persistence_performed")
            .expect("no-persistence lane");
        assert!(no_persist
            .persistence_semantics
            .contains("intentionally deferred"));

        let unavailable = map
            .iter()
            .find(|lane| lane.lane == "blue_brain_candidate_persistence_unavailable_or_deferred")
            .expect("persistence unavailable/deferred lane");
        assert!(unavailable
            .persistence_semantics
            .contains("no hidden commit path"));

        let commit_only_if_real = map
            .iter()
            .find(|lane| {
                lane.lane == "blue_brain_candidate_persistence_performed_only_if_real_path_exists"
            })
            .expect("commit-only-if-real lane");
        assert!(commit_only_if_real
            .persistence_semantics
            .contains("does not provide such a path"));
    }

    #[test]
    fn blue_brain_memory_commit_boundary_map_distinguishes_candidate_and_not_memory_states() {
        let map = canonical_blue_brain_memory_commit_boundary_map();
        assert_eq!(map.len(), 11);
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainMemoryCommitBoundaryClass::NotMemoryCandidate));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainMemoryCommitBoundaryClass::MemoryCandidateProposed));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainMemoryCommitBoundaryClass::MemoryCandidateDeferred));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainMemoryCommitBoundaryClass::MemoryCandidateRejected));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainMemoryCommitBoundaryClass::MemoryCandidateStale));
        assert!(map
            .iter()
            .any(|lane| lane.class
                == BlueBrainMemoryCommitBoundaryClass::MemoryCandidateInsufficient));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainMemoryCommitBoundaryClass::CommitEligibleCandidate));
        assert!(map.iter().any(
            |lane| lane.class == BlueBrainMemoryCommitBoundaryClass::FutureMemoryReadyCandidate
        ));
        assert!(map.iter().any(
            |lane| lane.class == BlueBrainMemoryCommitBoundaryClass::CommittedMemoryIfRealPath
        ));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainMemoryCommitBoundaryClass::ReferenceOnlyNotMemory));
        assert!(map.iter().any(|lane| {
            lane.class
                == BlueBrainMemoryCommitBoundaryClass::NonCanonicalInternalOnlyPersistencePath
        }));

        let deferred = map
            .iter()
            .find(|lane| lane.lane == "blue_brain_memory_commit_boundary_candidate_deferred")
            .expect("deferred candidate lane");
        assert!(deferred
            .eligibility_semantics
            .contains("not commit-eligible"));

        let reference_only = map
            .iter()
            .find(|lane| lane.lane == "blue_brain_memory_commit_boundary_reference_only_not_memory")
            .expect("reference-only lane");
        assert!(reference_only
            .persistence_path_semantics
            .contains("not memory persistence"));

        let committed_if_real = map
            .iter()
            .find(|lane| {
                lane.lane
                    == "blue_brain_memory_commit_boundary_committed_memory_only_if_real_path_exists"
            })
            .expect("committed if real lane");
        assert!(committed_if_real
            .persistence_path_semantics
            .contains("has no canonical Blue-Brain actual memory commit path"));
    }

    #[test]
    fn blue_brain_commit_eligibility_conditions_map_blocks_ineligible_candidates_and_internal_paths(
    ) {
        let map = canonical_blue_brain_commit_eligibility_conditions_map();
        assert_eq!(map.len(), 7);
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainCommitEligibilityConditionClass::EvidenceReferenceBasis
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainCommitEligibilityConditionClass::SelectionAttentionGate
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainCommitEligibilityConditionClass::ContextFreshnessGate
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainCommitEligibilityConditionClass::BlockingCaveatGate
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainCommitEligibilityConditionClass::CanonicalDependencyGate
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainCommitEligibilityConditionClass::PersistencePathGate
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainCommitEligibilityConditionClass::FutureMemoryReadyHandoffGate
        }));

        let selection_gate = map
            .iter()
            .find(|lane| {
                lane.lane == "blue_brain_commit_eligibility_condition_selection_attention_gate"
            })
            .expect("selection gate lane");
        assert!(selection_gate
            .when_not_satisfied
            .contains("deferred/rejected/not-memory"));

        let persistence_gate = map
            .iter()
            .find(|lane| {
                lane.lane
                    == "blue_brain_commit_eligibility_condition_explicit_persistence_path_or_none"
            })
            .expect("persistence path gate");
        assert!(persistence_gate
            .when_not_satisfied
            .contains("no actual commit occurs"));

        let canonical_gate = map
            .iter()
            .find(|lane| {
                lane.lane == "blue_brain_commit_eligibility_condition_no_internal_only_dependency"
            })
            .expect("canonical dependency gate");
        assert!(canonical_gate
            .when_not_satisfied
            .contains("non-canonical and commit-ineligible"));
    }

    #[test]
    fn blue_brain_persistence_boundary_map_keeps_classes_and_compute_core_boundaries_explicit() {
        let map = canonical_blue_brain_persistence_boundary_map();
        assert_eq!(map.len(), 9);
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainPersistenceBoundaryClass::TransientRuntimeContext));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainPersistenceBoundaryClass::EvidenceReferenceBackedContext
        }));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainPersistenceBoundaryClass::MemoryAdjacentCandidate));
        assert!(map.iter().any(
            |lane| lane.class == BlueBrainPersistenceBoundaryClass::FutureMemoryReadyCandidate
        ));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainPersistenceBoundaryClass::ActualPersistedMemory));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainPersistenceBoundaryClass::HistorySnapshotReferenceButNotMemory
        }));
        assert!(map.iter().any(|lane| {
            lane.class
                == BlueBrainPersistenceBoundaryClass::NonCanonicalInternalOnlyPersistenceLikePath
        }));

        let actual = map
            .iter()
            .find(|lane| {
                lane.lane == "blue_brain_persistence_boundary_actual_persisted_memory_deferred"
            })
            .expect("actual persisted memory deferred lane");
        assert!(actual.boundary_semantics.contains("intentionally deferred"));

        let history = map
            .iter()
            .find(|lane| {
                lane.lane == "blue_brain_persistence_boundary_history_snapshot_reference_not_memory"
            })
            .expect("history/snapshot/reference-not-memory lane");
        assert!(history
            .boundary_semantics
            .contains("not memory persistence"));

        let non_canonical = map
            .iter()
            .find(|lane| {
                lane.lane == "blue_brain_persistence_boundary_internal_expert_persistence_like_path"
            })
            .expect("non-canonical internal persistence-like lane");
        assert!(non_canonical
            .canonical_guard
            .contains("excluded from canonical memory authority"));
    }

    #[test]
    fn blue_brain_future_memory_attachment_map_preserves_propose_not_commit_semantics() {
        let map = canonical_blue_brain_future_memory_attachment_map();
        assert_eq!(map.len(), 7);
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainFutureMemoryAttachmentClass::CandidateHandoffProposalOnly
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainFutureMemoryAttachmentClass::CandidateFutureReadyNoCommit
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainFutureMemoryAttachmentClass::CandidateRejectedOrInsufficient
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainFutureMemoryAttachmentClass::PersistenceDeferredOrUnavailable
        }));
        assert!(map.iter().any(|lane| {
            lane.class
                == BlueBrainFutureMemoryAttachmentClass::PersistenceCommitOnlyIfRealPathExists
        }));
        assert!(map
            .iter()
            .any(|lane| lane.class
                == BlueBrainFutureMemoryAttachmentClass::HistoryReferenceBasisOnly));

        let proposal = map
            .iter()
            .find(|lane| lane.lane == "blue_brain_future_memory_attachment_candidate_proposed_only")
            .expect("proposal-only lane");
        assert!(proposal.commit_boundary.contains("does not commit memory"));

        let deferred = map
            .iter()
            .find(|lane| {
                lane.lane == "blue_brain_future_memory_attachment_persistence_unavailable_deferred"
            })
            .expect("deferred/unavailable lane");
        assert!(deferred
            .commit_boundary
            .contains("no commit while real persistence path is absent"));

        let real_only = map
            .iter()
            .find(|lane| {
                lane.lane == "blue_brain_future_memory_attachment_commit_only_if_real_path_exists"
            })
            .expect("commit-only-if-real-path lane");
        assert!(real_only
            .caveats
            .contains("current baseline has no such real path"));
    }

    #[test]
    fn blue_brain_reference_context_map_keeps_evidence_replay_snapshot_trace_classes_distinct() {
        let map = canonical_blue_brain_reference_context_map();
        assert_eq!(map.len(), 12);
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainReferenceContextClass::EvidenceBackedContext));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainReferenceContextClass::ReplayBackedContext));
        assert!(map.iter().any(
            |lane| lane.class == BlueBrainReferenceContextClass::SnapshotReferenceBackedContext
        ));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainReferenceContextClass::TraceBackedContext));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainReferenceContextClass::CaveatedReferenceContext));
        assert!(
            map.iter()
                .any(|lane| lane.class
                    == BlueBrainReferenceContextClass::InsufficientReferenceContext)
        );
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainReferenceContextClass::NonCanonicalInternalOnlyReferencePath
        }));
        assert!(map
            .iter()
            .any(|lane| lane.quality == BlueBrainReferenceQualityClass::Sufficient));
        assert!(map
            .iter()
            .any(|lane| lane.quality == BlueBrainReferenceQualityClass::Partial));
        assert!(map
            .iter()
            .any(|lane| lane.quality == BlueBrainReferenceQualityClass::Stale));
        assert!(map
            .iter()
            .any(|lane| lane.quality == BlueBrainReferenceQualityClass::Caveated));
        assert!(map
            .iter()
            .any(|lane| lane.quality == BlueBrainReferenceQualityClass::Insufficient));
    }

    #[test]
    fn blue_brain_reference_context_map_preserves_no_persistence_and_non_canonical_boundaries() {
        let map = canonical_blue_brain_reference_context_map();

        let insufficient = map
            .iter()
            .find(|lane| lane.lane == "blue_brain_reference_context_insufficient_basis_explicit")
            .expect("insufficient reference lane");
        assert!(insufficient
            .context_update_semantics
            .contains("context update blocked"));
        assert!(insufficient
            .candidate_semantics
            .contains("no persistence performed"));

        let replay = map
            .iter()
            .find(|lane| {
                lane.lane
                    == "blue_brain_reference_context_replay_backed_runtime_restored_or_informed"
            })
            .expect("replay-backed reference lane");
        assert!(replay
            .runtime_context_semantics
            .contains("runtime context restored or informed"));
        assert!(replay.persistence_boundary.contains("not memory commit"));

        let internal = map
            .iter()
            .find(|lane| {
                lane.lane == "blue_brain_reference_context_non_canonical_internal_only_path"
            })
            .expect("non-canonical reference lane");
        assert!(internal
            .candidate_semantics
            .contains("cannot appear as canonical"));
        assert!(internal.canonical_guard.contains("exclude"));
    }

    #[test]
    fn serie_bb3_prompt2_lifecycle_doc_stays_pinned_to_code_map() {
        let doc = include_str!(
            "../../../docs/blue_brain_context_memory_lifecycle_serie_bb3_prompt2_v1.md"
        );
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_CONTEXT_UPDATE_LIFECYCLE_MAP"));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_MEMORY_CANDIDATE_LIFECYCLE_MAP"));
        assert!(doc.contains("context initialized"));
        assert!(doc.contains("context updated from compute result"));
        assert!(doc.contains("context updated from evidence/reference"));
        assert!(doc.contains("context updated from replay/reference basis"));
        assert!(doc.contains("context unchanged"));
        assert!(doc.contains("context update blocked or insufficient"));
        assert!(doc.contains("candidate proposed"));
        assert!(doc.contains("candidate evidence-backed"));
        assert!(doc.contains("candidate accepted for future memory handling"));
        assert!(doc.contains("candidate rejected"));
        assert!(doc.contains("candidate stale"));
        assert!(doc.contains("candidate insufficient"));
        assert!(doc.contains("persistence unavailable/deferred"));
        assert!(doc.contains("persistence performed only if real path exists"));
        assert!(doc.contains("no persistence performed"));
        assert!(doc.contains("actual memory commit remains intentionally deferred"));
    }

    #[test]
    fn serie_bb3_prompt4_persistence_boundary_doc_stays_pinned_to_code_map() {
        let doc = include_str!(
            "../../../docs/blue_brain_persistence_boundary_attachment_serie_bb3_prompt4_v1.md"
        );
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_PERSISTENCE_BOUNDARY_MAP"));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_FUTURE_MEMORY_ATTACHMENT_MAP"));
        assert!(doc.contains("transient runtime context"));
        assert!(doc.contains("evidence/reference-backed context"));
        assert!(doc.contains("memory-adjacent candidate"));
        assert!(doc.contains("future-memory-ready candidate"));
        assert!(doc.contains("actual persisted memory"));
        assert!(doc.contains("history/snapshot/reference but not memory"));
        assert!(doc.contains("non-canonical/internal-only persistence-like path"));
        assert!(doc.contains("candidate proposed"));
        assert!(doc.contains("candidate future-memory-ready"));
        assert!(doc.contains("candidate rejected"));
        assert!(doc.contains("candidate stale/insufficient"));
        assert!(doc.contains("persistence unavailable/deferred"));
        assert!(doc.contains("commit only if real explicit path exists"));
        assert!(doc.contains("BB3 implements no actual Blue-Brain memory persistence"));
        assert!(doc.contains("Compute-Core bleibt maintenance-only"));
    }

    #[test]
    fn serie_bb3_prompt5_readiness_doc_keeps_context_memory_baseline_and_deferred_persistence_explicit(
    ) {
        let doc = include_str!("../../../docs/blue_brain_readiness_sweep_serie_bb3_prompt5_v1.md");
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("stable context/memory foundation"));
        assert!(doc.contains("usable with caveats"));
        assert!(doc.contains("future-memory-ready / preparatory only"));
        assert!(doc.contains("reference-only / not memory"));
        assert!(doc.contains("non-canonical / internal-only"));
        assert!(doc.contains("intentionally deferred"));

        assert!(doc.contains("transient_runtime_context"));
        assert!(doc.contains("evidence_backed_context"));
        assert!(doc.contains("replay_reference_backed_context"));
        assert!(doc.contains("memory_adjacent_candidate"));
        assert!(doc.contains("context update blocked or insufficient"));
        assert!(doc.contains("candidate accepted for future memory handling"));
        assert!(doc.contains("persistence unavailable/deferred"));
        assert!(doc.contains("persistence performed only if real path exists"));
        assert!(doc.contains("no persistence performed"));
        assert!(doc.contains("Context Update ≠ Memory Commit"));
        assert!(doc.contains("Candidate ≠ Persisted Memory"));
        assert!(
            doc.contains("History/Snapshot/Replay/Trace/Evidence/Reference ≠ Memory Persistence")
        );
        assert!(doc.contains("keine neue Compute-Core-Arbeit"));
        assert!(doc.contains("finaler Compute-Linie"));
        assert!(doc.contains("maintenance-only Core"));
        assert!(doc.contains("Priorität 1: Serie BB5"));
        assert!(doc.contains("Hodgkin-Huxley/Kuramoto"));
    }

    #[test]
    fn serie_bb3_prompt3_evidence_reference_doc_stays_pinned_to_code_map() {
        let doc = include_str!(
            "../../../docs/blue_brain_evidence_reference_context_serie_bb3_prompt3_v1.md"
        );
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_REFERENCE_CONTEXT_MAP"));
        assert!(doc.contains("evidence-backed context"));
        assert!(doc.contains("replay-backed context"));
        assert!(doc.contains("snapshot/reference-backed context"));
        assert!(doc.contains("trace-backed context"));
        assert!(doc.contains("caveated reference context"));
        assert!(doc.contains("insufficient reference context"));
        assert!(doc.contains("non-canonical/internal-only reference path"));
        assert!(doc.contains("sufficient reference basis"));
        assert!(doc.contains("partial reference basis"));
        assert!(doc.contains("stale reference basis"));
        assert!(doc.contains("caveated reference basis"));
        assert!(doc.contains("insufficient reference basis"));
        assert!(doc.contains("context updated with evidence reference"));
        assert!(doc.contains("context update blocked due to insufficient evidence"));
        assert!(doc.contains("evidence observed without context update"));
        assert!(doc.contains("replay/reference used for context only, not memory commit"));
        assert!(doc.contains("candidate evidence-backed"));
        assert!(doc.contains("candidate replay/reference-backed"));
        assert!(doc.contains("candidate trace/snapshot-backed"));
        assert!(doc.contains("no persistence performed"));
    }

    #[test]
    fn blue_brain_control_attention_selection_map_keeps_canonical_classes_and_states_distinct() {
        let map = canonical_blue_brain_control_attention_selection_map();
        assert_eq!(map.len(), 22);
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainControlAttentionSelectionClass::AttentionTarget));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainControlAttentionSelectionClass::ContextSelection));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainControlAttentionSelectionClass::EvidenceReferenceSelection
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainControlAttentionSelectionClass::MemoryCandidateSelection
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainControlAttentionSelectionClass::ComputeTriggerSelection
        }));
        assert!(map.iter().any(|lane| {
            lane.class
                == BlueBrainControlAttentionSelectionClass::NonCanonicalInternalOnlySelectionPath
        }));

        assert!(map
            .iter()
            .any(|lane| lane.disposition == BlueBrainSelectionDispositionClass::Selected));
        assert!(map
            .iter()
            .any(|lane| lane.disposition == BlueBrainSelectionDispositionClass::Deferred));
        assert!(map.iter().any(|lane| {
            lane.disposition == BlueBrainSelectionDispositionClass::IgnoredOrIrrelevant
        }));
        assert!(map
            .iter()
            .any(|lane| lane.disposition == BlueBrainSelectionDispositionClass::Blocked));
        assert!(map
            .iter()
            .any(|lane| lane.disposition == BlueBrainSelectionDispositionClass::Insufficient));
        assert!(map
            .iter()
            .any(|lane| lane.disposition == BlueBrainSelectionDispositionClass::Caveated));
        assert!(map
            .iter()
            .any(|lane| lane.disposition == BlueBrainSelectionDispositionClass::Rejected));

        assert!(map
            .iter()
            .any(|lane| lane.basis_quality == BlueBrainSelectionBasisQualityClass::Sufficient));
        assert!(map
            .iter()
            .any(|lane| lane.basis_quality == BlueBrainSelectionBasisQualityClass::Partial));
        assert!(map
            .iter()
            .any(|lane| lane.basis_quality == BlueBrainSelectionBasisQualityClass::Stale));
        assert!(map
            .iter()
            .any(|lane| lane.basis_quality == BlueBrainSelectionBasisQualityClass::Caveated));
        assert!(map
            .iter()
            .any(|lane| lane.basis_quality == BlueBrainSelectionBasisQualityClass::Insufficient));
    }

    #[test]
    fn blue_brain_control_attention_selection_map_preserves_no_commit_and_canonical_trigger_handoff(
    ) {
        let map = canonical_blue_brain_control_attention_selection_map();
        for lane in map {
            assert!(
                lane.memory_persistence_semantics.contains("no")
                    || lane.memory_persistence_semantics.contains("not")
                    || lane.memory_persistence_semantics.contains("never")
                    || lane.memory_persistence_semantics.contains("non-")
                    || lane.memory_persistence_semantics.contains("cannot")
                    || lane.memory_persistence_semantics.contains("does not")
            );
        }

        let context_trigger = map
            .iter()
            .find(|lane| lane.lane == "blue_brain_compute_trigger_selected_from_context")
            .expect("context-trigger selection lane");
        assert_eq!(
            context_trigger.disposition,
            BlueBrainSelectionDispositionClass::Selected
        );
        assert!(context_trigger
            .compute_trigger_binding
            .contains("CanonicalComputeEntryPoint::submit"));

        let evidence_trigger = map
            .iter()
            .find(|lane| {
                lane.lane == "blue_brain_compute_trigger_selected_from_evidence_reference_need"
            })
            .expect("evidence-trigger selection lane");
        assert!(evidence_trigger
            .compute_trigger_binding
            .contains("CanonicalComputeEntryPoint::submit"));

        let blocked = map
            .iter()
            .find(|lane| {
                lane.lane == "blue_brain_compute_trigger_blocked_insufficient_selection_basis"
            })
            .expect("blocked trigger selection lane");
        assert_eq!(
            blocked.disposition,
            BlueBrainSelectionDispositionClass::Blocked
        );
        assert_eq!(
            blocked.basis_quality,
            BlueBrainSelectionBasisQualityClass::Insufficient
        );

        let internal = map
            .iter()
            .find(|lane| {
                lane.lane == "blue_brain_compute_trigger_internal_expert_only_non_canonical"
            })
            .expect("internal non-canonical selection lane");
        assert!(internal
            .compute_trigger_binding
            .contains("no internal/expert-only trigger used as canonical authority"));
        assert!(internal.canonical_guard.contains("excluded"));
    }

    #[test]
    fn serie_bb4_prompt1_control_attention_selection_doc_stays_pinned_to_code_map() {
        let doc = include_str!(
            "../../../docs/blue_brain_control_attention_selection_surface_serie_bb4_prompt1_v1.md"
        );
        let line = canonical_final_reference_line();

        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_CONTROL_ATTENTION_SELECTION_MAP"));
        assert!(doc.contains("attention target"));
        assert!(doc.contains("selected context"));
        assert!(doc.contains("selected evidence/reference"));
        assert!(doc.contains("selected memory candidate"));
        assert!(doc.contains("deferred"));
        assert!(doc.contains("ignored"));
        assert!(doc.contains("blocked"));
        assert!(doc.contains("insufficient selection basis"));
        assert!(doc.contains("caveated selection"));
        assert!(doc.contains("no memory commit implied"));
        assert!(doc.contains("no internal/expert-only trigger used"));
        assert!(doc.contains("keine Planning- oder Reasoning-Engine"));
        assert!(doc.contains("keine Policy-/Governance-Plattform"));
    }

    #[test]
    fn blue_brain_compute_trigger_arbitration_map_keeps_trigger_states_sources_and_invocation_gates_distinct(
    ) {
        let map = canonical_blue_brain_compute_trigger_arbitration_map();
        assert_eq!(map.len(), 16);
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainComputeTriggerArbitrationClass::TriggerCandidate
                && lane.source == BlueBrainComputeTriggerSourceClass::ContextDerived
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainComputeTriggerArbitrationClass::SelectedTrigger
                && lane.invocation == BlueBrainSelectionGatedInvocationClass::InvocationRequested
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainComputeTriggerArbitrationClass::DeferredTrigger
                && lane.invocation == BlueBrainSelectionGatedInvocationClass::NoInvocationDeferred
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainComputeTriggerArbitrationClass::SuppressedTrigger
                && lane.source == BlueBrainComputeTriggerSourceClass::ManualInternalOnlyNonCanonical
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainComputeTriggerArbitrationClass::BlockedTrigger
                && lane.invocation == BlueBrainSelectionGatedInvocationClass::NoInvocationBlocked
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainComputeTriggerArbitrationClass::InsufficientTriggerBasis
                && lane.invocation
                    == BlueBrainSelectionGatedInvocationClass::InsufficientBasisRequiresMoreContextOrEvidence
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainComputeTriggerArbitrationClass::CaveatedTrigger
                && lane.invocation
                    == BlueBrainSelectionGatedInvocationClass::CaveatedInvocationAllowed
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainComputeTriggerArbitrationClass::NonCanonicalInternalOnlyTrigger
        }));
        assert!(map.iter().any(|lane| {
            lane.source == BlueBrainComputeTriggerSourceClass::EvidenceReferenceDerived
        }));
        assert!(map.iter().any(|lane| {
            lane.source == BlueBrainComputeTriggerSourceClass::RuntimeStateDerived
        }));
        assert!(map
            .iter()
            .any(|lane| lane.source == BlueBrainComputeTriggerSourceClass::MemoryCandidateDerived));
        assert!(map
            .iter()
            .any(|lane| lane.source == BlueBrainComputeTriggerSourceClass::FeedbackDerived));
    }

    #[test]
    fn blue_brain_compute_trigger_arbitration_map_binds_invocation_to_outward_contract_and_no_commit_boundary(
    ) {
        let map = canonical_blue_brain_compute_trigger_arbitration_map();
        assert!(map.iter().any(|lane| {
            lane.invocation == BlueBrainSelectionGatedInvocationClass::InvocationCompleted
                && lane
                    .outward_compute_contract_binding
                    .contains("CanonicalComputeEntryPoint::submit")
        }));
        assert!(map.iter().any(|lane| {
            lane.invocation == BlueBrainSelectionGatedInvocationClass::InvocationFailed
                && lane
                    .outward_compute_contract_binding
                    .contains("canonical outward status/fault surfaces")
        }));
        assert!(map.iter().any(|lane| {
            lane.invocation
                == BlueBrainSelectionGatedInvocationClass::InvocationBlockedByComputeContract
                && lane
                    .outward_compute_contract_binding
                    .contains("status/evidence exports")
        }));
        assert!(map.iter().any(|lane| {
            lane.invocation == BlueBrainSelectionGatedInvocationClass::NoInvocationDeferred
                && lane
                    .outward_compute_contract_binding
                    .contains("does not call CanonicalComputeEntryPoint::submit")
        }));
        assert!(map.iter().any(|lane| {
            lane.invocation
                == BlueBrainSelectionGatedInvocationClass::InsufficientBasisRequiresMoreContextOrEvidence
                && lane
                    .outward_compute_contract_binding
                    .contains("no invocation request")
        }));
        assert!(map.iter().all(|lane| {
            lane.memory_commit_boundary.contains("no")
                || lane.memory_commit_boundary.contains("not")
                || lane.memory_commit_boundary.contains("never")
                || lane.memory_commit_boundary.contains("runtime context only")
        }));
    }

    #[test]
    fn serie_bb4_prompt2_compute_trigger_arbitration_doc_stays_pinned_to_code_map() {
        let doc = include_str!(
            "../../../docs/blue_brain_compute_trigger_arbitration_serie_bb4_prompt2_v1.md"
        );
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_COMPUTE_TRIGGER_ARBITRATION_MAP"));
        assert!(doc.contains("trigger candidate"));
        assert!(doc.contains("selected trigger"));
        assert!(doc.contains("deferred trigger"));
        assert!(doc.contains("suppressed trigger"));
        assert!(doc.contains("blocked trigger"));
        assert!(doc.contains("insufficient trigger basis"));
        assert!(doc.contains("caveated trigger"));
        assert!(doc.contains("non-canonical/internal-only trigger"));
        assert!(doc.contains("context-derived trigger"));
        assert!(doc.contains("evidence/reference-derived trigger"));
        assert!(doc.contains("runtime-state-derived trigger"));
        assert!(doc.contains("memory-candidate-derived trigger"));
        assert!(doc.contains("feedback-derived trigger"));
        assert!(doc.contains("trigger selected and invocation requested"));
        assert!(doc.contains("trigger deferred and no invocation"));
        assert!(doc.contains("trigger blocked and no invocation"));
        assert!(doc.contains("trigger caveated but allowed"));
        assert!(doc.contains("trigger insufficient and requires more context/evidence"));
        assert!(doc.contains("invocation completed"));
        assert!(doc.contains("invocation failed"));
        assert!(doc.contains("invocation blocked by Compute contract"));
        assert!(doc.contains("invocation caveated/degraded"));
        assert!(doc.contains("runtime context but not memory automatically"));
        assert!(doc.contains("Scheduler-/Planning-/Policy-/Reasoning-Plattform"));
    }

    #[test]
    fn blue_brain_context_evidence_priority_map_keeps_primary_supporting_deferred_and_quality_classes_distinct(
    ) {
        let map = canonical_blue_brain_context_evidence_priority_map();
        assert_eq!(map.len(), 10);
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainContextEvidencePriorityClass::PrimaryContext));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainContextEvidencePriorityClass::SupportingContext));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainContextEvidencePriorityClass::DeferredContext));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainContextEvidencePriorityClass::IgnoredContext));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainContextEvidencePriorityClass::StaleContext));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainContextEvidencePriorityClass::InsufficientContext));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainContextEvidencePriorityClass::PrimaryEvidenceReference
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainContextEvidencePriorityClass::SupportingEvidenceReference
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainContextEvidencePriorityClass::CaveatedEvidenceReference
        }));
        assert!(map.iter().any(|lane| {
            lane.class
                == BlueBrainContextEvidencePriorityClass::NonCanonicalInternalOnlyPriorityPath
        }));

        let deferred = map
            .iter()
            .find(|lane| lane.lane == "blue_brain_priority_deferred_context_pending_update")
            .expect("deferred context priority lane");
        assert!(deferred
            .trigger_arbitration_binding
            .contains("no trigger invocation yet"));
        assert!(deferred
            .deferral_or_caveat_reason
            .contains("partial/caveated"));
        assert!(deferred
            .recheck_condition
            .contains("context update or stronger evidence"));
        assert!(deferred
            .canonical_guard
            .contains("not rejected, not ignored"));

        let insufficient = map
            .iter()
            .find(|lane| lane.lane == "blue_brain_priority_insufficient_context_blocks_trigger")
            .expect("insufficient context priority lane");
        assert!(insufficient
            .trigger_arbitration_binding
            .contains("blocks invocation"));
    }

    #[test]
    fn blue_brain_candidate_deferral_lifecycle_map_distinguishes_deferred_rejected_stale_insufficient_and_not_persisted(
    ) {
        let map = canonical_blue_brain_candidate_deferral_lifecycle_map();
        assert_eq!(map.len(), 8);
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainCandidateDeferralLifecycleClass::CandidateSelected));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainCandidateDeferralLifecycleClass::CandidateDeferred));
        assert!(map.iter().any(|lane| {
            lane.class
                == BlueBrainCandidateDeferralLifecycleClass::CandidateDeferredPendingStrongerEvidence
        }));
        assert!(map.iter().any(|lane| {
            lane.class
                == BlueBrainCandidateDeferralLifecycleClass::CandidateDeferredPendingContextUpdate
        }));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainCandidateDeferralLifecycleClass::CandidateRejected));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainCandidateDeferralLifecycleClass::CandidateStale));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainCandidateDeferralLifecycleClass::CandidateInsufficient
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainCandidateDeferralLifecycleClass::CandidateNotPersisted
        }));

        let deferred = map
            .iter()
            .find(|lane| lane.class == BlueBrainCandidateDeferralLifecycleClass::CandidateDeferred)
            .expect("candidate deferred lane");
        assert!(deferred
            .trigger_binding
            .contains("does not trigger compute"));
        assert!(deferred
            .memory_commit_boundary
            .contains("not persisted and not rejected"));
        assert!(deferred
            .canonical_guard
            .contains("distinct from rejected/ignored/stale/insufficient"));

        let pending_evidence = map
            .iter()
            .find(|lane| {
                lane.class
                    == BlueBrainCandidateDeferralLifecycleClass::CandidateDeferredPendingStrongerEvidence
            })
            .expect("pending stronger evidence lane");
        assert!(pending_evidence
            .recheck_condition
            .contains("quality reaches sufficient"));

        let not_persisted = map
            .iter()
            .find(|lane| {
                lane.class == BlueBrainCandidateDeferralLifecycleClass::CandidateNotPersisted
            })
            .expect("not persisted lane");
        assert!(not_persisted
            .memory_commit_boundary
            .contains("no memory commit"));
    }

    #[test]
    fn serie_bb4_prompt3_priority_deferral_doc_stays_pinned_to_code_maps() {
        let doc = include_str!(
            "../../../docs/blue_brain_priority_deferral_semantics_serie_bb4_prompt3_v1.md"
        );
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_CONTEXT_EVIDENCE_PRIORITY_MAP"));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_CANDIDATE_DEFERRAL_LIFECYCLE_MAP"));
        assert!(doc.contains("primary context"));
        assert!(doc.contains("supporting context"));
        assert!(doc.contains("deferred context"));
        assert!(doc.contains("ignored context"));
        assert!(doc.contains("stale context"));
        assert!(doc.contains("insufficient context"));
        assert!(doc.contains("primary evidence/reference"));
        assert!(doc.contains("supporting evidence/reference"));
        assert!(doc.contains("caveated evidence/reference"));
        assert!(doc.contains("non-canonical/internal-only priority path"));
        assert!(doc.contains("candidate deferred pending stronger evidence"));
        assert!(doc.contains("candidate deferred pending context update"));
        assert!(doc.contains("deferred candidate does not trigger compute"));
        assert!(doc.contains("no memory commit"));
        assert!(doc.contains("not rejected"));
        assert!(doc.contains("keine numerische Ranking- oder Scoring-Engine"));
        assert!(doc.contains("keine Memory-Consolidation- oder Commit-Engine"));
    }

    #[test]
    fn blue_brain_selection_diagnostics_map_keeps_canonical_outcomes_reasons_and_non_canonical_boundary_explicit(
    ) {
        let map = canonical_blue_brain_selection_diagnostics_map();
        assert_eq!(map.len(), 8);
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainSelectionDiagnosticClass::SelectedItemDiagnostic));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainSelectionDiagnosticClass::DeferredItemDiagnostic));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainSelectionDiagnosticClass::IgnoredItemDiagnostic));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainSelectionDiagnosticClass::RejectedItemDiagnostic));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainSelectionDiagnosticClass::BlockedSelectionDiagnostic
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainSelectionDiagnosticClass::InsufficientSelectionDiagnostic
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainSelectionDiagnosticClass::CaveatedSelectionDiagnostic
        }));
        assert!(map.iter().any(|lane| {
            lane.class
                == BlueBrainSelectionDiagnosticClass::NonCanonicalInternalOnlyDiagnosticDetail
        }));
        assert!(map
            .iter()
            .all(|lane| lane.runtime_diagnostics_binding.contains("diagnostics")));
    }

    #[test]
    fn serie_bb4_prompt4_control_attention_diagnostics_doc_stays_pinned_to_code_map() {
        let doc = include_str!(
            "../../../docs/blue_brain_control_attention_diagnostics_serie_bb4_prompt4_v1.md"
        );
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_SELECTION_DIAGNOSTICS_MAP"));
        assert!(doc.contains("selected item diagnostic"));
        assert!(doc.contains("deferred item diagnostic"));
        assert!(doc.contains("ignored item diagnostic"));
        assert!(doc.contains("rejected item diagnostic"));
        assert!(doc.contains("blocked selection diagnostic"));
        assert!(doc.contains("insufficient selection diagnostic"));
        assert!(doc.contains("caveated selection diagnostic"));
        assert!(doc.contains("non-canonical/internal-only diagnostic detail"));
        assert!(doc.contains("selected due to sufficient context"));
        assert!(doc.contains("selected due to primary evidence/reference"));
        assert!(doc.contains("deferred due to partial evidence"));
        assert!(doc.contains("blocked due to stale/insufficient basis"));
        assert!(doc.contains("ignored because not relevant to current transition"));
        assert!(doc.contains("rejected due to fault/caveat"));
        assert!(doc.contains("selection-gated transition"));
        assert!(doc.contains("no memory persistence implied"));
        assert!(doc.contains("keine Explainability-, Planning-, Policy- oder Audit-Plattform"));
    }

    #[test]
    fn serie_bb4_prompt5_readiness_sweep_doc_stays_pinned_to_bb4_maps_and_boundaries() {
        let doc = include_str!("../../../docs/blue_brain_readiness_sweep_serie_bb4_prompt5_v1.md");
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_CONTROL_ATTENTION_SELECTION_MAP"));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_COMPUTE_TRIGGER_ARBITRATION_MAP"));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_CONTEXT_EVIDENCE_PRIORITY_MAP"));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_CANDIDATE_DEFERRAL_LIFECYCLE_MAP"));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_SELECTION_DIAGNOSTICS_MAP"));
        assert!(doc.contains("stable control/attention foundation"));
        assert!(doc.contains("usable with caveats"));
        assert!(doc.contains("preparatory only"));
        assert!(doc.contains("non-canonical / internal-only"));
        assert!(doc.contains("intentionally deferred"));
        assert!(doc.contains("Selection ist **keine** Planning Engine"));
        assert!(doc.contains("Candidate Selection impliziert **keinen** Memory Commit"));
        assert!(doc.contains("Compute-Kern bleibt maintenance-only"));
        assert!(doc.contains("Priorität: Serie BB5 zuerst"));
        assert!(doc.contains("Hodgkin-Huxley/Kuramoto"));
    }

    #[test]
    fn blue_brain_planning_reasoning_candidate_map_keeps_bb6_classes_and_basis_states_distinct() {
        let map = canonical_blue_brain_planning_reasoning_candidate_map();
        assert_eq!(map.len(), 15);
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainPlanningReasoningCandidateClass::RuntimeDerivedPlanningCandidate
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainPlanningReasoningCandidateClass::ContextDerivedReasoningCandidate
        }));
        assert!(map.iter().any(|lane| {
            lane.class
                == BlueBrainPlanningReasoningCandidateClass::EvidenceReferenceDerivedReasoningCandidate
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainPlanningReasoningCandidateClass::SelectionDerivedActionCandidate
        }));
        assert!(map.iter().any(|lane| {
            lane.class
                == BlueBrainPlanningReasoningCandidateClass::MemoryCandidateDerivedReasoningCandidate
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainPlanningReasoningCandidateClass::CommitFeedbackDerivedCandidate
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainPlanningReasoningCandidateClass::InsufficientCandidateBasis
        }));
        assert!(map.iter().any(|lane| {
            lane.class
                == BlueBrainPlanningReasoningCandidateClass::NonCanonicalInternalOnlyPlanningLikePath
        }));
        assert!(map.iter().any(|lane| {
            lane.basis_state == BlueBrainPlanningReasoningCandidateBasisState::BasisAvailable
        }));
        assert!(map.iter().any(|lane| {
            lane.basis_state
                == BlueBrainPlanningReasoningCandidateBasisState::BasisPartialOrCaveated
        }));
        assert!(map.iter().any(|lane| {
            lane.basis_state == BlueBrainPlanningReasoningCandidateBasisState::BasisStale
        }));
        assert!(map.iter().any(|lane| {
            lane.basis_state == BlueBrainPlanningReasoningCandidateBasisState::BasisInsufficient
        }));
        assert!(map.iter().any(|lane| {
            lane.basis_state == BlueBrainPlanningReasoningCandidateBasisState::CandidateDeferred
        }));
        assert!(map.iter().any(|lane| {
            lane.basis_state
                == BlueBrainPlanningReasoningCandidateBasisState::CandidateProposedUnresolved
        }));
        assert!(map.iter().any(|lane| {
            lane.basis_state
                == BlueBrainPlanningReasoningCandidateBasisState::EvidenceObservedNoCandidate
        }));
        assert!(map.iter().any(|lane| {
            lane.basis_state == BlueBrainPlanningReasoningCandidateBasisState::CandidateBlocked
        }));
    }

    #[test]
    fn blue_brain_planning_reasoning_candidate_map_makes_no_automatic_execution_or_commit_claims() {
        let map = canonical_blue_brain_planning_reasoning_candidate_map();
        assert!(map
            .iter()
            .all(|lane| lane.no_execution_implication.contains("no")));
        assert!(map
            .iter()
            .all(|lane| lane.memory_commit_boundary.contains("no")));
        assert!(map.iter().any(|lane| {
            lane.lane == "blue_brain_committed_if_present_strengthens_basis_conditionally"
                && lane.memory_commit_boundary.contains("only if")
        }));
        assert!(map.iter().any(|lane| {
            lane.lane == "blue_brain_non_canonical_internal_planning_like_path"
                && lane.resolution_boundary.contains("excluded from canonical")
        }));
    }

    #[test]
    fn serie_bb6_prompt1_planning_reasoning_candidate_doc_stays_pinned_to_code_map() {
        let doc = include_str!(
            "../../../docs/blue_brain_planning_reasoning_candidate_surface_serie_bb6_prompt1_v1.md"
        );
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_PLANNING_REASONING_CANDIDATE_MAP"));
        assert!(doc.contains("runtime-derived planning candidate"));
        assert!(doc.contains("context-derived reasoning candidate"));
        assert!(doc.contains("evidence/reference-derived reasoning candidate"));
        assert!(doc.contains("selection-derived action candidate"));
        assert!(doc.contains("memory-candidate-derived reasoning candidate"));
        assert!(doc.contains("commit-feedback-derived candidate"));
        assert!(doc.contains("insufficient candidate basis"));
        assert!(doc.contains("non-canonical/internal-only planning-like path"));
        assert!(doc.contains("candidate basis available"));
        assert!(doc.contains("partial/caveated"));
        assert!(doc.contains("stale"));
        assert!(doc.contains("insufficient"));
        assert!(doc.contains("deferred"));
        assert!(doc.contains("no action execution implied"));
        assert!(doc.contains("no memory commit implied"));
        assert!(doc.contains("keine Planning-Engine"));
        assert!(doc.contains("keine Reasoning-Engine"));
        assert!(doc.contains("keine Policy-/RL-/Agentenplattform"));
        assert!(doc.contains("Compute-Kern bleibt maintenance-only"));
        assert!(doc.contains("Hodgkin-Huxley/Kuramoto"));
    }

    #[test]
    fn blue_brain_candidate_action_boundary_map_keeps_candidate_proposal_and_execution_classes_distinct(
    ) {
        let map = canonical_blue_brain_candidate_action_boundary_map();
        assert_eq!(map.len(), 10);
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainCandidateActionBoundaryClass::PlanningReasoningCandidate
        }));
        assert!(map
            .iter()
            .any(|lane| lane.class
                == BlueBrainCandidateActionBoundaryClass::ActionProposalNonExecuting));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainCandidateActionBoundaryClass::SelectedProposal));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainCandidateActionBoundaryClass::DeferredProposal));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainCandidateActionBoundaryClass::RejectedProposal));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainCandidateActionBoundaryClass::BlockedProposal));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainCandidateActionBoundaryClass::CaveatedProposal));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainCandidateActionBoundaryClass::InsufficientProposalBasis
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainCandidateActionBoundaryClass::ExecutedActionCanonicalIfPresent
        }));
        assert!(map.iter().any(|lane| {
            lane.class
                == BlueBrainCandidateActionBoundaryClass::NonCanonicalInternalOnlyActionLikePath
        }));
    }

    #[test]
    fn blue_brain_non_executing_action_proposal_maps_block_auto_execution_compute_tool_and_commit()
    {
        let boundary_map = canonical_blue_brain_candidate_action_boundary_map();
        let proposal_states = canonical_blue_brain_non_executing_action_proposal_state_map();
        assert!(proposal_states
            .iter()
            .all(|lane| lane.execution_boundary.contains("no")));
        assert!(proposal_states
            .iter()
            .all(|lane| lane.compute_trigger_boundary.contains("no")));
        assert!(proposal_states
            .iter()
            .all(|lane| lane.memory_commit_boundary.contains("no")));
        assert!(proposal_states
            .iter()
            .all(|lane| lane.tool_execution_boundary.contains("no")));
        assert!(boundary_map.iter().any(|lane| {
            lane.class == BlueBrainCandidateActionBoundaryClass::ExecutedActionCanonicalIfPresent
                && lane
                    .compute_invocation_boundary
                    .contains("requires explicit call")
        }));
    }

    #[test]
    fn blue_brain_candidate_to_proposal_transition_map_keeps_non_auto_promotion_rules_explicit() {
        let transitions = canonical_blue_brain_candidate_to_proposal_transition_map();
        assert_eq!(transitions.len(), 6);
        assert!(transitions.iter().any(|lane| {
            lane.class == BlueBrainCandidateToProposalTransitionClass::CandidateRemainsCandidate
                && lane.proposal_outcome == "no proposal created"
        }));
        assert!(transitions.iter().any(|lane| {
            lane.class == BlueBrainCandidateToProposalTransitionClass::CandidateYieldsActionProposal
                && lane.proposal_outcome == "proposal created"
        }));
        assert!(transitions.iter().any(|lane| {
            lane.class
                == BlueBrainCandidateToProposalTransitionClass::CandidateInsufficientForProposal
                && lane.proposal_outcome.contains("insufficient")
        }));
        assert!(transitions.iter().any(|lane| {
            lane.class
                == BlueBrainCandidateToProposalTransitionClass::CandidateYieldsCaveatedProposal
                && lane.proposal_outcome.contains("caveated")
        }));
        assert!(transitions.iter().any(|lane| {
            lane.class
                == BlueBrainCandidateToProposalTransitionClass::CandidateRejectedBeforeProposal
                && lane.proposal_outcome.contains("rejected")
        }));
        assert!(transitions.iter().any(|lane| {
            lane.class
                == BlueBrainCandidateToProposalTransitionClass::CandidateDeferredBeforeProposal
                && lane.proposal_outcome.contains("deferred")
        }));
        assert!(transitions.iter().all(|lane| {
            lane.execution_boundary.contains("no")
                || lane.execution_boundary.contains("not")
                || lane.execution_boundary.contains("cannot")
                || lane.execution_boundary.contains("non-")
        }));
        assert!(transitions.iter().all(|lane| {
            lane.compute_boundary.contains("no")
                || lane.compute_boundary.contains("not")
                || lane.compute_boundary.contains("cannot")
                || lane.compute_boundary.contains("non-")
        }));
        assert!(transitions.iter().all(|lane| {
            lane.memory_commit_boundary.contains("no")
                || lane.memory_commit_boundary.contains("not")
                || lane.memory_commit_boundary.contains("cannot")
                || lane.memory_commit_boundary.contains("non-")
        }));
    }

    #[test]
    fn serie_bb6_prompt2_candidate_action_boundary_doc_stays_pinned_to_code_maps() {
        let doc = include_str!(
            "../../../docs/blue_brain_candidate_action_boundary_serie_bb6_prompt2_v1.md"
        );
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_CANDIDATE_ACTION_BOUNDARY_MAP"));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_CANDIDATE_TO_PROPOSAL_TRANSITION_MAP"));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_NON_EXECUTING_ACTION_PROPOSAL_STATE_MAP"));
        assert!(doc.contains("planning/reasoning candidate"));
        assert!(doc.contains("action proposal (non-executing)"));
        assert!(doc.contains("selected proposal"));
        assert!(doc.contains("deferred proposal"));
        assert!(doc.contains("rejected proposal"));
        assert!(doc.contains("blocked proposal"));
        assert!(doc.contains("caveated proposal"));
        assert!(doc.contains("insufficient proposal basis"));
        assert!(doc.contains("executed action (canonical path only if explicit invocation exists)"));
        assert!(doc.contains("non-canonical/internal-only action-like path"));
        assert!(doc.contains("proposal created"));
        assert!(doc.contains("proposal selected for possible future action"));
        assert!(doc.contains("proposal deferred"));
        assert!(doc.contains("proposal rejected"));
        assert!(doc.contains("proposal blocked"));
        assert!(doc.contains("proposal caveated"));
        assert!(doc.contains("proposal insufficient basis"));
        assert!(doc.contains("no execution performed"));
        assert!(doc.contains("candidate remains candidate"));
        assert!(doc.contains("candidate yields proposal"));
        assert!(doc.contains("candidate insufficient for proposal"));
        assert!(doc.contains("candidate yields caveated proposal"));
        assert!(doc.contains("candidate rejected before proposal"));
        assert!(doc.contains("candidate deferred before proposal"));
        assert!(doc.contains("no automatic action execution"));
        assert!(doc.contains("no automatic compute invocation"));
        assert!(doc.contains("no automatic memory commit"));
        assert!(doc.contains("no automatic tool execution"));
        assert!(doc.contains("Compute-Kern bleibt maintenance-only"));
        assert!(doc.contains("Hodgkin-Huxley/Kuramoto"));
    }

    #[test]
    fn blue_brain_reasoning_candidate_diagnostics_map_keeps_required_classes_distinct() {
        let map = canonical_blue_brain_reasoning_candidate_diagnostics_map();
        assert_eq!(map.len(), 10);
        assert!(map.iter().any(|lane| lane.class
            == BlueBrainReasoningCandidateDiagnosticClass::CandidateBasisDiagnostic));
        assert!(map.iter().any(|lane| lane.class
            == BlueBrainReasoningCandidateDiagnosticClass::SufficientCandidateDiagnostic));
        assert!(map.iter().any(|lane| lane.class
            == BlueBrainReasoningCandidateDiagnosticClass::PartialCandidateDiagnostic));
        assert!(map.iter().any(|lane| lane.class
            == BlueBrainReasoningCandidateDiagnosticClass::CaveatedCandidateDiagnostic));
        assert!(map.iter().any(|lane| lane.class
            == BlueBrainReasoningCandidateDiagnosticClass::StaleCandidateDiagnostic));
        assert!(map.iter().any(|lane| lane.class
            == BlueBrainReasoningCandidateDiagnosticClass::InsufficientCandidateDiagnostic));
        assert!(map.iter().any(|lane| lane.class
            == BlueBrainReasoningCandidateDiagnosticClass::DeferredCandidateDiagnostic));
        assert!(map.iter().any(|lane| lane.class
            == BlueBrainReasoningCandidateDiagnosticClass::RejectedCandidateDiagnostic));
        assert!(map.iter().any(|lane| lane.class
            == BlueBrainReasoningCandidateDiagnosticClass::ProposalReadyDiagnostic));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainReasoningCandidateDiagnosticClass::NonCanonicalInternalOnlyDiagnostic
        }));
    }

    #[test]
    fn blue_brain_reasoning_candidate_diagnostics_map_preserves_non_execution_non_commit_and_non_reasoning_completion_boundaries(
    ) {
        let map = canonical_blue_brain_reasoning_candidate_diagnostics_map();
        assert!(map.iter().all(|lane| !lane
            .runtime_context_feedback_binding
            .contains("reasoning completed")));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainReasoningCandidateDiagnosticClass::ProposalReadyDiagnostic
                && lane
                    .proposal_boundary_binding
                    .contains("not executed action")
        }));
        assert!(map.iter().any(|lane| {
            lane.class
                == BlueBrainReasoningCandidateDiagnosticClass::InsufficientCandidateDiagnostic
                && lane
                    .insufficiency_or_caveat_reason
                    .contains("missing context")
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainReasoningCandidateDiagnosticClass::StaleCandidateDiagnostic
                && lane
                    .insufficiency_or_caveat_reason
                    .contains("stale reference basis")
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainReasoningCandidateDiagnosticClass::CaveatedCandidateDiagnostic
                && lane
                    .insufficiency_or_caveat_reason
                    .contains("selection/attention caveat")
        }));
        assert!(map.iter().any(|lane| {
            lane.class
                == BlueBrainReasoningCandidateDiagnosticClass::NonCanonicalInternalOnlyDiagnostic
                && lane
                    .insufficiency_or_caveat_reason
                    .contains("non-canonical/internal dependency")
        }));
        assert!(map.iter().any(|lane| {
            lane.class
                == BlueBrainReasoningCandidateDiagnosticClass::InsufficientCandidateDiagnostic
                && lane
                    .memory_boundary_binding
                    .contains("blocks commit progression")
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainReasoningCandidateDiagnosticClass::ProposalReadyDiagnostic
                && lane
                    .memory_boundary_binding
                    .contains("does not commit memory")
        }));
    }

    #[test]
    fn serie_bb6_prompt3_reasoning_candidate_diagnostics_doc_stays_pinned_to_code_map() {
        let doc = include_str!(
            "../../../docs/blue_brain_reasoning_candidate_diagnostics_feedback_serie_bb6_prompt3_v1.md"
        );
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_REASONING_CANDIDATE_DIAGNOSTICS_MAP"));
        assert!(doc.contains("candidate-basis diagnostic"));
        assert!(doc.contains("sufficient candidate diagnostic"));
        assert!(doc.contains("partial candidate diagnostic"));
        assert!(doc.contains("caveated candidate diagnostic"));
        assert!(doc.contains("stale candidate diagnostic"));
        assert!(doc.contains("insufficient candidate diagnostic"));
        assert!(doc.contains("deferred candidate diagnostic"));
        assert!(doc.contains("rejected candidate diagnostic"));
        assert!(doc.contains("proposal-ready diagnostic"));
        assert!(doc.contains("non-canonical/internal-only diagnostic"));
        assert!(doc.contains("runtime-derived"));
        assert!(doc.contains("context-derived"));
        assert!(doc.contains("evidence/reference-derived"));
        assert!(doc.contains("selection-derived"));
        assert!(doc.contains("memory-candidate-derived"));
        assert!(doc.contains("commit-feedback-derived"));
        assert!(doc.contains("proposal-derived"));
        assert!(doc.contains("missing context"));
        assert!(doc.contains("weak or missing evidence"));
        assert!(doc.contains("stale reference basis"));
        assert!(doc.contains("partial evidence"));
        assert!(doc.contains("selection/attention caveat"));
        assert!(doc.contains("unavailable memory commit"));
        assert!(doc.contains("candidate remains candidate"));
        assert!(doc.contains("candidate becomes proposal-ready"));
        assert!(doc.contains("candidate yields caveated proposal"));
        assert!(doc.contains("candidate deferred before proposal"));
        assert!(doc.contains("candidate rejected before proposal"));
        assert!(doc.contains("no action execution implied"));
        assert!(doc.contains("no memory commit implied"));
        assert!(doc.contains("no reasoning completed claim"));
        assert!(doc.contains("keine Explainability-, Audit- oder Policy-Plattform"));
        assert!(doc.contains("Compute-Kern bleibt maintenance-only"));
        assert!(doc.contains("Hodgkin-Huxley/Kuramoto"));
    }

    #[test]
    fn blue_brain_candidate_comparison_map_keeps_classes_and_boundaries_distinct() {
        let map = canonical_blue_brain_candidate_comparison_map();
        assert_eq!(map.len(), 9);
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainCandidateComparisonClass::ComparableCandidates));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainCandidateComparisonClass::ComparisonBasisAvailable
        }));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainCandidateComparisonClass::ComparisonMeaningful));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainCandidateComparisonClass::ComparisonCaveated));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainCandidateComparisonClass::ComparisonInconclusive
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainCandidateComparisonClass::ComparisonNotMeaningful
        }));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainCandidateComparisonClass::ComparisonBlocked));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainCandidateComparisonClass::NonCanonicalInternalOnlyComparison
        }));
        assert!(map.iter().all(
            |lane| lane.selection_interaction_boundary.contains("does not")
                || lane.selection_interaction_boundary.contains("not selected")
                || lane
                    .selection_interaction_boundary
                    .contains("remains deferred")
                || lane.selection_interaction_boundary.contains("remains")
                || lane.selection_interaction_boundary.contains("informs")
                || lane.selection_interaction_boundary.contains("implied")
                || lane.selection_interaction_boundary.contains("automatic")
                || lane.selection_interaction_boundary.contains("cannot")
        ));
        assert!(map
            .iter()
            .all(|lane| lane.proposal_interaction_boundary.contains("no")
                || lane.proposal_interaction_boundary.contains("not")
                || lane.proposal_interaction_boundary.contains("insufficient")
                || lane.proposal_interaction_boundary.contains("supports")));
    }

    #[test]
    fn blue_brain_candidate_comparison_map_preserves_no_auto_selection_proposal_execution_or_commit(
    ) {
        let map = canonical_blue_brain_candidate_comparison_map();
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainCandidateComparisonClass::ComparisonMeaningful
                && lane
                    .selection_interaction_boundary
                    .contains("informs selection, but does not decide")
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainCandidateComparisonClass::ComparisonCaveated
                && lane
                    .proposal_interaction_boundary
                    .contains("insufficient for proposal")
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainCandidateComparisonClass::ComparisonNotMeaningful
                && lane
                    .proposal_interaction_boundary
                    .contains("no proposal generated")
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainCandidateComparisonClass::ComparisonBlocked
                && lane
                    .proposal_interaction_boundary
                    .contains("no action executed")
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainCandidateComparisonClass::NonCanonicalInternalOnlyComparison
                && lane.canonical_guard.contains("non-canonical")
        }));
    }

    #[test]
    fn serie_bb6_prompt4_candidate_comparison_doc_stays_pinned_to_code_map() {
        let doc = include_str!(
            "../../../docs/blue_brain_planning_reasoning_candidate_comparison_serie_bb6_prompt4_v1.md"
        );
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_CANDIDATE_COMPARISON_MAP"));
        assert!(doc.contains("comparable candidates"));
        assert!(doc.contains("comparison basis available"));
        assert!(doc.contains("comparison meaningful"));
        assert!(doc.contains("comparison caveated"));
        assert!(doc.contains("comparison inconclusive"));
        assert!(doc.contains("comparison not meaningful"));
        assert!(doc.contains("comparison blocked"));
        assert!(doc.contains("non-canonical/internal-only comparison"));
        assert!(doc.contains("runtime basis"));
        assert!(doc.contains("context basis"));
        assert!(doc.contains("evidence/reference basis"));
        assert!(doc.contains("selection/attention basis"));
        assert!(doc.contains("memory-candidate or commit-feedback basis"));
        assert!(doc.contains("proposal-status basis"));
        assert!(doc.contains("compared but not selected"));
        assert!(doc.contains("candidate remains deferred"));
        assert!(doc.contains("comparison informs selection, but does not decide"));
        assert!(doc.contains("comparison only"));
        assert!(doc.contains("comparison supports proposal caveat"));
        assert!(doc.contains("comparison insufficient for proposal"));
        assert!(doc.contains("no proposal generated"));
        assert!(doc.contains("no action executed"));
        assert!(doc.contains("no memory commit implied"));
        assert!(doc.contains("Compute-Kern bleibt maintenance-only"));
        assert!(doc.contains("keine Ranking-Engine"));
        assert!(doc.contains("keine Planning-Engine"));
        assert!(doc.contains("keine Reasoning-Engine"));
        assert!(doc.contains("Hodgkin-Huxley/Kuramoto"));
    }

    #[test]
    fn serie_bb6_prompt5_readiness_sweep_doc_keeps_final_candidate_closure_line_explicit() {
        let doc = include_str!("../../../docs/blue_brain_readiness_sweep_serie_bb6_prompt5_v1.md");
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_PLANNING_REASONING_CANDIDATE_MAP"));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_CANDIDATE_ACTION_BOUNDARY_MAP"));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_CANDIDATE_TO_PROPOSAL_TRANSITION_MAP"));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_NON_EXECUTING_ACTION_PROPOSAL_STATE_MAP"));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_REASONING_CANDIDATE_DIAGNOSTICS_MAP"));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_CANDIDATE_COMPARISON_MAP"));
        assert!(doc.contains("stable planning/reasoning candidate foundation"));
        assert!(doc.contains("usable with caveats"));
        assert!(doc.contains("preparatory only"));
        assert!(doc.contains("non-canonical / internal-only"));
        assert!(doc.contains("intentionally deferred"));
        assert!(doc.contains("Candidate ≠ Plan"));
        assert!(doc.contains("Candidate ≠ Reasoning Completed"));
        assert!(doc.contains("Proposal (auch `selected proposal`) ≠ Action Execution"));
        assert!(doc.contains("Candidate Diagnostics ≠ Explainability-/Reasoning-Plattform"));
        assert!(doc.contains("Candidate Comparison ≠ Ranking/Selection/Entscheidungszwang"));
        assert!(doc.contains("automatische Compute Invocation"));
        assert!(doc.contains("Memory Commit"));
        assert!(doc.contains("Compute-Kern bleibt maintenance-only"));
        assert!(doc.contains("Priorität: Serie BB7 zuerst"));
        assert!(doc.contains("Hodgkin-Huxley/Kuramoto"));
    }

    #[test]
    fn blue_brain_minimal_planning_action_interface_map_keeps_readiness_classes_distinct() {
        let map = canonical_blue_brain_minimal_planning_action_interface_map();
        assert_eq!(map.len(), 16);
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainMinimalPlanningActionInterfaceClass::DiagnosticOnlyProposal
        }));
        assert!(map
            .iter()
            .any(|lane| lane.class
                == BlueBrainMinimalPlanningActionInterfaceClass::PlanReadyProposal));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainMinimalPlanningActionInterfaceClass::ActionReadyProposal
        }));
        assert!(map.iter().any(
            |lane| lane.class == BlueBrainMinimalPlanningActionInterfaceClass::DeferredProposal
        ));
        assert!(map.iter().any(
            |lane| lane.class == BlueBrainMinimalPlanningActionInterfaceClass::BlockedProposal
        ));
        assert!(map.iter().any(
            |lane| lane.class == BlueBrainMinimalPlanningActionInterfaceClass::RejectedProposal
        ));
        assert!(map.iter().any(
            |lane| lane.class == BlueBrainMinimalPlanningActionInterfaceClass::CaveatedProposal
        ));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainMinimalPlanningActionInterfaceClass::InsufficientProposalBasis
        }));
        assert!(map.iter().any(|lane| {
            lane.class
                == BlueBrainMinimalPlanningActionInterfaceClass::ExecutedActionCanonicalIfPresent
        }));
        assert!(map.iter().any(|lane| {
            lane.class
                == BlueBrainMinimalPlanningActionInterfaceClass::NonCanonicalInternalOnlyActionPath
        }));
    }

    #[test]
    fn blue_brain_minimal_planning_action_interface_map_keeps_readiness_non_executing_boundaries() {
        let map = canonical_blue_brain_minimal_planning_action_interface_map();
        let readiness_lanes: Vec<_> = map
            .iter()
            .filter(|lane| {
                !matches!(
                    lane.class,
                    BlueBrainMinimalPlanningActionInterfaceClass::ExecutedActionCanonicalIfPresent
                        | BlueBrainMinimalPlanningActionInterfaceClass::NonCanonicalInternalOnlyActionPath
                )
            })
            .collect();
        assert!(readiness_lanes
            .iter()
            .all(|lane| lane.execution_boundary.contains("no")));
        assert!(readiness_lanes
            .iter()
            .all(|lane| lane.compute_invocation_boundary.contains("no")));
        assert!(readiness_lanes
            .iter()
            .all(|lane| lane.tool_invocation_boundary.contains("no")));
        assert!(readiness_lanes
            .iter()
            .all(|lane| lane.memory_commit_boundary.contains("no")));
        assert!(map.iter().any(|lane| {
            lane.lane == "blue_brain_proposal_action_ready_not_executed"
                && lane.execution_boundary.contains("not executed")
        }));
        assert!(map.iter().any(|lane| {
            lane.lane == "blue_brain_proposal_plan_ready_no_plan_generated"
                && lane.plan_boundary.contains("no plan generated")
        }));
        assert!(map.iter().any(|lane| {
            lane.lane == "blue_brain_proposal_action_ready_blocked_missing_boundary"
                && lane
                    .readiness_semantics
                    .contains("missing explicit action boundary")
        }));
        assert!(map.iter().any(|lane| {
            lane.lane == "blue_brain_proposal_action_ready_requires_future_subsystem"
                && lane.readiness_semantics.contains("future action subsystem")
        }));
    }

    #[test]
    fn blue_brain_minimal_planning_action_interface_map_references_diagnostics_and_comparison_basis(
    ) {
        let map = canonical_blue_brain_minimal_planning_action_interface_map();
        assert!(map
            .iter()
            .any(|lane| lane.context_evidence_selection_binding.contains("context")));
        assert!(map.iter().any(|lane| lane
            .context_evidence_selection_binding
            .contains("selection")));
        assert!(map.iter().all(|lane| lane
            .diagnostics_comparison_binding
            .contains("CANONICAL_BLUE_BRAIN_REASONING_CANDIDATE_DIAGNOSTICS_MAP")
            || lane
                .diagnostics_comparison_binding
                .contains("readiness diagnostics")
            || lane
                .diagnostics_comparison_binding
                .contains("non-canonical/internal diagnostics")));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainMinimalPlanningActionInterfaceClass::InsufficientProposalBasis
                && lane
                    .diagnostics_comparison_binding
                    .contains("InsufficientCandidateDiagnostic")
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainMinimalPlanningActionInterfaceClass::CaveatedProposal
                && lane
                    .diagnostics_comparison_binding
                    .contains("CaveatedCandidateDiagnostic")
        }));
        assert!(map.iter().any(|lane| {
            lane.class
                == BlueBrainMinimalPlanningActionInterfaceClass::NonCanonicalInternalOnlyActionPath
                && lane
                    .diagnostics_comparison_binding
                    .contains("non-canonical/internal diagnostics")
        }));
    }

    #[test]
    fn serie_bb7_prompt1_minimal_planning_action_interface_doc_stays_pinned_to_code_map() {
        let doc = include_str!(
            "../../../docs/blue_brain_minimal_planning_action_interface_serie_bb7_prompt1_v1.md"
        );
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_MINIMAL_PLANNING_ACTION_INTERFACE_MAP"));
        assert!(doc.contains("diagnostic-only proposal"));
        assert!(doc.contains("plan-ready proposal"));
        assert!(doc.contains("action-ready proposal"));
        assert!(doc.contains("deferred proposal"));
        assert!(doc.contains("blocked proposal"));
        assert!(doc.contains("rejected proposal"));
        assert!(doc.contains("caveated proposal"));
        assert!(doc.contains("insufficient proposal basis"));
        assert!(doc.contains("executed action (canonical path only if explicit invocation exists)"));
        assert!(doc.contains("non-canonical/internal-only action path"));
        assert!(doc.contains("action-ready but not executed"));
        assert!(doc.contains("action-ready with caveat"));
        assert!(doc.contains("action-ready blocked by missing boundary"));
        assert!(doc.contains("action-ready requires future action subsystem"));
        assert!(doc.contains("plan-ready but no plan generated"));
        assert!(doc.contains("plan-ready with caveat"));
        assert!(doc.contains("plan-ready deferred"));
        assert!(doc.contains("plan-ready blocked due to insufficient basis"));
        assert!(doc.contains("sufficient candidate basis permits readiness"));
        assert!(doc.contains("caveated candidate basis yields caveated readiness"));
        assert!(doc.contains("inconclusive comparison limits readiness"));
        assert!(doc.contains("insufficient candidate basis blocks readiness"));
        assert!(doc.contains("non-canonical basis blocks readiness"));
        assert!(doc.contains("no action execution"));
        assert!(doc.contains("no plan generation"));
        assert!(doc.contains("no tool invocation"));
        assert!(doc.contains("no compute invocation"));
        assert!(doc.contains("no memory commit"));
        assert!(doc.contains("Compute-Kern bleibt maintenance-only"));
        assert!(doc.contains("Hodgkin-Huxley/Kuramoto"));
    }

    #[test]
    fn blue_brain_plan_action_readiness_diagnostics_map_keeps_classes_distinct() {
        let map = canonical_blue_brain_plan_action_readiness_diagnostics_map();
        assert_eq!(map.len(), 9);
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainPlanActionReadinessDiagnosticClass::PlanReadyDiagnostic
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainPlanActionReadinessDiagnosticClass::ActionReadyDiagnostic
        }));
        assert!(map.iter().any(|lane| {
            lane.class
                == BlueBrainPlanActionReadinessDiagnosticClass::DiagnosticOnlyProposalDiagnostic
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainPlanActionReadinessDiagnosticClass::DeferredReadinessDiagnostic
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainPlanActionReadinessDiagnosticClass::BlockedReadinessDiagnostic
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainPlanActionReadinessDiagnosticClass::RejectedReadinessDiagnostic
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainPlanActionReadinessDiagnosticClass::CaveatedReadinessDiagnostic
        }));
        assert!(map.iter().any(|lane| {
            lane.class
                == BlueBrainPlanActionReadinessDiagnosticClass::InsufficientReadinessDiagnostic
        }));
        assert!(map.iter().any(|lane| {
            lane.class
                == BlueBrainPlanActionReadinessDiagnosticClass::NonCanonicalInternalOnlyReadinessDiagnostic
        }));
    }

    #[test]
    fn blue_brain_plan_action_readiness_diagnostics_map_keeps_blocked_action_feedback_non_executing(
    ) {
        let map = canonical_blue_brain_plan_action_readiness_diagnostics_map();
        let blocked = map
            .iter()
            .find(|lane| {
                lane.class
                    == BlueBrainPlanActionReadinessDiagnosticClass::BlockedReadinessDiagnostic
            })
            .expect("blocked readiness lane must exist");
        assert!(blocked
            .blocked_action_feedback
            .contains("never means tool executed"));
        assert!(blocked.blocked_action_feedback.contains("action failed"));
        assert!(blocked.blocked_action_feedback.contains("policy denied"));
        assert!(blocked.blocked_action_feedback.contains("planner denied"));
        assert!(map.iter().all(|lane| lane.runtime_feedback.contains("no")
            || lane.runtime_feedback.contains("canonical=false")));
        assert!(map
            .iter()
            .all(|lane| lane.execution_tool_policy_boundary.contains("not")
                || lane.execution_tool_policy_boundary.contains("must not")));
    }

    #[test]
    fn serie_bb7_prompt2_readiness_diagnostics_doc_stays_pinned_to_code_map() {
        let doc = include_str!(
            "../../../docs/blue_brain_plan_action_readiness_diagnostics_serie_bb7_prompt2_v1.md"
        );
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_PLAN_ACTION_READINESS_DIAGNOSTICS_MAP"));
        assert!(doc.contains("plan-ready diagnostic"));
        assert!(doc.contains("action-ready diagnostic"));
        assert!(doc.contains("diagnostic-only proposal diagnostic"));
        assert!(doc.contains("deferred readiness diagnostic"));
        assert!(doc.contains("blocked readiness diagnostic"));
        assert!(doc.contains("rejected readiness diagnostic"));
        assert!(doc.contains("caveated readiness diagnostic"));
        assert!(doc.contains("insufficient readiness diagnostic"));
        assert!(doc.contains("non-canonical/internal-only readiness diagnostic"));
        assert!(doc.contains("ready due to sufficient candidate basis"));
        assert!(doc.contains("ready due to sufficient context/evidence"));
        assert!(doc.contains("ready due to selection/attention state"));
        assert!(doc.contains("deferred due to partial evidence"));
        assert!(doc.contains("blocked due to stale context"));
        assert!(doc.contains("blocked due to missing action boundary"));
        assert!(doc.contains("caveated due to memory/commit unavailability"));
        assert!(doc.contains("rejected due to candidate/proposal rejection"));
        assert!(doc.contains("blocked-action feedback means readiness transition could not occur"));
        assert!(doc.contains("tool executed"));
        assert!(doc.contains("action failed"));
        assert!(doc.contains("policy denied"));
        assert!(doc.contains("planner denied"));
        assert!(doc.contains("no action execution"));
        assert!(doc.contains("no tool invocation"));
        assert!(doc.contains("no compute invocation"));
        assert!(doc.contains("no memory commit"));
        assert!(doc.contains("Compute-Kern bleibt maintenance-only"));
    }

    #[test]
    fn blue_brain_future_action_handoff_and_result_placeholder_maps_keep_classes_distinct() {
        let handoff_map = canonical_blue_brain_future_action_handoff_map();
        assert_eq!(handoff_map.len(), 9);
        assert!(handoff_map
            .iter()
            .any(|lane| lane.class == BlueBrainFutureActionHandoffClass::FutureActionReady));
        assert!(handoff_map
            .iter()
            .any(|lane| lane.class == BlueBrainFutureActionHandoffClass::FuturePlanReady));
        assert!(handoff_map
            .iter()
            .any(|lane| lane.class == BlueBrainFutureActionHandoffClass::HandoffDeferred));
        assert!(handoff_map
            .iter()
            .any(|lane| lane.class == BlueBrainFutureActionHandoffClass::HandoffBlocked));
        assert!(handoff_map
            .iter()
            .any(|lane| lane.class == BlueBrainFutureActionHandoffClass::HandoffRejected));
        assert!(handoff_map
            .iter()
            .any(|lane| lane.class == BlueBrainFutureActionHandoffClass::HandoffCaveated));
        assert!(handoff_map
            .iter()
            .any(|lane| lane.class == BlueBrainFutureActionHandoffClass::HandoffUnavailable));
        assert!(handoff_map
            .iter()
            .any(|lane| lane.class == BlueBrainFutureActionHandoffClass::DiagnosticOnlyNoHandoff));
        assert!(handoff_map.iter().any(|lane| {
            lane.class == BlueBrainFutureActionHandoffClass::InternalOnlyNonCanonicalHandoff
        }));
        assert!(handoff_map
            .iter()
            .all(|lane| lane.execution_and_commit_boundary.contains("no")));
        assert!(handoff_map
            .iter()
            .all(|lane| lane.runtime_diagnostics_binding.contains("runtime")));
        assert!(handoff_map.iter().any(|lane| {
            lane.class == BlueBrainFutureActionHandoffClass::InternalOnlyNonCanonicalHandoff
                && lane.runtime_diagnostics_binding.contains("canonical=false")
        }));

        let placeholder_map = canonical_blue_brain_action_result_placeholder_map();
        assert_eq!(placeholder_map.len(), 8);
        assert!(placeholder_map.iter().any(|lane| {
            lane.class == BlueBrainActionResultPlaceholderClass::ResultPlaceholderPrepared
        }));
        assert!(placeholder_map.iter().any(|lane| {
            lane.class == BlueBrainActionResultPlaceholderClass::ResultPlaceholderUnavailable
        }));
        assert!(placeholder_map.iter().any(|lane| {
            lane.class == BlueBrainActionResultPlaceholderClass::ResultPlaceholderBlocked
        }));
        assert!(placeholder_map.iter().any(|lane| {
            lane.class == BlueBrainActionResultPlaceholderClass::ResultPlaceholderCaveated
        }));
        assert!(placeholder_map
            .iter()
            .any(|lane| lane.class == BlueBrainActionResultPlaceholderClass::NoResultExpected));
        assert!(placeholder_map
            .iter()
            .any(|lane| lane.class == BlueBrainActionResultPlaceholderClass::NoActionExecuted));
        assert!(placeholder_map
            .iter()
            .any(|lane| lane.class == BlueBrainActionResultPlaceholderClass::NoToolResult));
        assert!(placeholder_map.iter().any(|lane| {
            lane.class == BlueBrainActionResultPlaceholderClass::InternalOnlyNonCanonicalPlaceholder
        }));
        assert!(placeholder_map
            .iter()
            .all(|lane| lane.boundary_semantics.contains("no")));
        assert!(placeholder_map.iter().any(|lane| {
            lane.class == BlueBrainActionResultPlaceholderClass::ResultPlaceholderPrepared
                && lane.placeholder_semantics.contains("no actual result")
        }));
    }

    #[test]
    fn serie_bb7_prompt3_future_action_handoff_doc_stays_pinned_to_code_maps() {
        let doc = include_str!(
            "../../../docs/blue_brain_future_action_handoff_result_placeholder_serie_bb7_prompt3_v1.md"
        );
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_FUTURE_ACTION_HANDOFF_MAP"));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_ACTION_RESULT_PLACEHOLDER_MAP"));
        assert!(doc.contains("future-action-ready"));
        assert!(doc.contains("future-plan-ready"));
        assert!(doc.contains("handoff deferred"));
        assert!(doc.contains("handoff blocked"));
        assert!(doc.contains("handoff rejected"));
        assert!(doc.contains("handoff caveated"));
        assert!(doc.contains("handoff unavailable"));
        assert!(doc.contains("diagnostic-only/no-handoff"));
        assert!(doc.contains("internal-only/non-canonical handoff"));
        assert!(doc.contains("result placeholder prepared"));
        assert!(doc.contains("result placeholder unavailable"));
        assert!(doc.contains("result placeholder blocked"));
        assert!(doc.contains("result placeholder caveated"));
        assert!(doc.contains("no result expected"));
        assert!(doc.contains("no action executed"));
        assert!(doc.contains("no tool result"));
        assert!(doc.contains("Placeholder ≠ Result"));
        assert!(doc.contains("Handoff ≠ Action Execution"));
        assert!(doc.contains("Compute Invocation"));
        assert!(doc.contains("Memory Commit"));
        assert!(doc.contains("canonical=false"));
        assert!(doc.contains("Compute-Kern bleibt maintenance-only"));
        assert!(doc.contains("Hodgkin-Huxley/Kuramoto"));
    }

    #[test]
    fn serie_bb7_prompt4_readiness_sweep_doc_keeps_final_planning_action_closure_line_explicit() {
        let doc = include_str!("../../../docs/blue_brain_readiness_sweep_serie_bb7_prompt4_v1.md");
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_MINIMAL_PLANNING_ACTION_INTERFACE_MAP"));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_PLAN_ACTION_READINESS_DIAGNOSTICS_MAP"));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_FUTURE_ACTION_HANDOFF_MAP"));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_ACTION_RESULT_PLACEHOLDER_MAP"));
        assert!(doc.contains("stable minimal planning/action interface"));
        assert!(doc.contains("usable with caveats"));
        assert!(doc.contains("preparatory / placeholder only"));
        assert!(doc.contains("non-canonical / internal-only"));
        assert!(doc.contains("intentionally deferred"));
        assert!(doc.contains("future-action-ready"));
        assert!(doc.contains("future-plan-ready"));
        assert!(doc.contains("result placeholder prepared"));
        assert!(doc.contains("plan-ready"));
        assert!(doc.contains("action-ready"));
        assert!(doc.contains("diagnostic-only"));
        assert!(doc.contains("Compute Invocation"));
        assert!(doc.contains("Memory Commit"));
        assert!(doc.contains("Compute-Kern bleibt maintenance-only"));
        assert!(doc.contains("Planning Engine"));
        assert!(doc.contains("Reasoning Engine"));
        assert!(doc.contains("Serie BB8"));
        assert!(doc.contains("Priorität: Serie BB8 zuerst"));
        assert!(doc.contains("Hodgkin-Huxley/Kuramoto"));
    }

    #[test]
    fn blue_brain_bb9_safety_precheck_and_execution_eligibility_maps_keep_classes_distinct() {
        let precheck_map = canonical_blue_brain_safety_precheck_map();
        assert_eq!(precheck_map.len(), 7);
        assert!(precheck_map
            .iter()
            .any(|lane| lane.class == BlueBrainSafetyPrecheckClass::Passed));
        assert!(precheck_map
            .iter()
            .any(|lane| lane.class == BlueBrainSafetyPrecheckClass::Failed));
        assert!(precheck_map
            .iter()
            .any(|lane| lane.class == BlueBrainSafetyPrecheckClass::Blocked));
        assert!(precheck_map
            .iter()
            .any(|lane| lane.class == BlueBrainSafetyPrecheckClass::Caveated));
        assert!(precheck_map
            .iter()
            .any(|lane| lane.class == BlueBrainSafetyPrecheckClass::Insufficient));
        assert!(precheck_map
            .iter()
            .any(|lane| lane.class == BlueBrainSafetyPrecheckClass::Unavailable));
        assert!(precheck_map
            .iter()
            .any(|lane| lane.class == BlueBrainSafetyPrecheckClass::NotApplicable));
        assert!(precheck_map
            .iter()
            .all(|lane| lane.execution_boundary.contains("not")
                || lane.execution_boundary.contains("no")));

        let eligibility_map = canonical_blue_brain_action_execution_eligibility_map();
        assert_eq!(eligibility_map.len(), 13);
        assert!(eligibility_map.iter().any(|lane| {
            lane.class == BlueBrainActionExecutionEligibilityClass::FutureActionReadyHandoff
        }));
        assert!(eligibility_map.iter().any(|lane| {
            lane.class == BlueBrainActionExecutionEligibilityClass::ExecutionEligibleHandoff
        }));
        assert!(eligibility_map.iter().any(|lane| {
            lane.class == BlueBrainActionExecutionEligibilityClass::ExecutionIneligibleHandoff
        }));
        assert!(eligibility_map.iter().any(|lane| {
            lane.class == BlueBrainActionExecutionEligibilityClass::ExecutionBlockedHandoff
        }));
        assert!(eligibility_map.iter().any(|lane| {
            lane.class == BlueBrainActionExecutionEligibilityClass::ExecutionCaveatedHandoff
        }));
        assert!(eligibility_map.iter().any(|lane| {
            lane.class == BlueBrainActionExecutionEligibilityClass::ExecutionInsufficientBasis
        }));
        assert!(eligibility_map.iter().any(|lane| {
            lane.class == BlueBrainActionExecutionEligibilityClass::SafetyPrecheckPassed
        }));
        assert!(eligibility_map.iter().any(|lane| {
            lane.class == BlueBrainActionExecutionEligibilityClass::SafetyPrecheckFailed
        }));
        assert!(eligibility_map.iter().any(|lane| {
            lane.class == BlueBrainActionExecutionEligibilityClass::SafetyPrecheckBlocked
        }));
        assert!(eligibility_map.iter().any(|lane| {
            lane.class == BlueBrainActionExecutionEligibilityClass::SafetyPrecheckCaveated
        }));
        assert!(eligibility_map.iter().any(|lane| {
            lane.class == BlueBrainActionExecutionEligibilityClass::SafetyPrecheckUnavailable
        }));
        assert!(eligibility_map.iter().any(|lane| {
            lane.class == BlueBrainActionExecutionEligibilityClass::ExecutedActionCanonicalIfPresent
        }));
        assert!(eligibility_map.iter().any(|lane| {
            lane.class
                == BlueBrainActionExecutionEligibilityClass::NonCanonicalInternalOnlyExecutionPath
        }));
        assert!(eligibility_map
            .iter()
            .all(|lane| lane.execution_boundary.contains("no")
                || lane.execution_boundary.contains("not")
                || lane.execution_boundary.contains("non-executing")
                || lane.execution_boundary.contains("never")));
        assert!(eligibility_map.iter().any(|lane| {
            lane.class == BlueBrainActionExecutionEligibilityClass::ExecutionEligibleHandoff
                && lane.execution_boundary.contains("non-executing")
        }));

        let diagnostics_map = canonical_blue_brain_execution_eligibility_diagnostics_map();
        assert_eq!(diagnostics_map.len(), 11);
        assert!(diagnostics_map.iter().any(|lane| {
            lane.class == BlueBrainExecutionEligibilityDiagnosticClass::ExecutionEligibleDiagnostic
        }));
        assert!(diagnostics_map.iter().any(|lane| {
            lane.class
                == BlueBrainExecutionEligibilityDiagnosticClass::ExecutionIneligibleDiagnostic
        }));
        assert!(diagnostics_map.iter().any(|lane| {
            lane.class == BlueBrainExecutionEligibilityDiagnosticClass::ExecutionBlockedDiagnostic
        }));
        assert!(diagnostics_map.iter().any(|lane| {
            lane.class == BlueBrainExecutionEligibilityDiagnosticClass::ExecutionCaveatedDiagnostic
        }));
        assert!(diagnostics_map.iter().any(|lane| {
            lane.class
                == BlueBrainExecutionEligibilityDiagnosticClass::ExecutionInsufficientDiagnostic
        }));
        assert!(diagnostics_map.iter().any(|lane| {
            lane.class
                == BlueBrainExecutionEligibilityDiagnosticClass::SafetyPrecheckPassedDiagnostic
        }));
        assert!(diagnostics_map.iter().any(|lane| {
            lane.class
                == BlueBrainExecutionEligibilityDiagnosticClass::SafetyPrecheckFailedDiagnostic
        }));
        assert!(diagnostics_map.iter().any(|lane| {
            lane.class
                == BlueBrainExecutionEligibilityDiagnosticClass::SafetyPrecheckBlockedDiagnostic
        }));
        assert!(diagnostics_map.iter().any(|lane| {
            lane.class
                == BlueBrainExecutionEligibilityDiagnosticClass::SafetyPrecheckCaveatedDiagnostic
        }));
        assert!(diagnostics_map.iter().any(|lane| {
            lane.class
                == BlueBrainExecutionEligibilityDiagnosticClass::SafetyPrecheckUnavailableDiagnostic
        }));
        assert!(diagnostics_map.iter().any(|lane| {
            lane.class
                == BlueBrainExecutionEligibilityDiagnosticClass::NonCanonicalInternalOnlyExecutionDiagnostic
        }));
        assert!(diagnostics_map.iter().any(|lane| {
            lane.class == BlueBrainExecutionEligibilityDiagnosticClass::ExecutionEligibleDiagnostic
                && lane.runtime_feedback_binding.contains("no action execution")
        }));
    }

    #[test]
    fn blue_brain_bb9_execution_eligibility_conditions_bind_to_memory_precheck_and_diagnostics() {
        let eligibility_map = canonical_blue_brain_action_execution_eligibility_map();
        let eligible = eligibility_map
            .iter()
            .find(|lane| {
                lane.class == BlueBrainActionExecutionEligibilityClass::ExecutionEligibleHandoff
            })
            .expect("eligible lane must exist");
        assert!(eligible
            .context_evidence_basis
            .contains("sufficient or caveated-allowed context"));
        assert!(eligible
            .selection_candidate_basis
            .contains("no blocking candidate/proposal diagnostics"));
        assert!(eligible
            .memory_basis
            .contains("invalidated memory blocks eligibility"));
        assert!(eligible.safety_precheck_binding.contains("Passed|Caveated"));
        assert!(eligible
            .execution_boundary
            .contains("no tool/action/compute invocation"));
        assert!(eligible.execution_boundary.contains("no memory commit"));

        let future_only = eligibility_map
            .iter()
            .find(|lane| {
                lane.class == BlueBrainActionExecutionEligibilityClass::FutureActionReadyHandoff
            })
            .expect("future-ready-only lane must exist");
        assert!(future_only
            .eligibility_semantics
            .contains("not yet execution-eligible"));

        let blocked = eligibility_map
            .iter()
            .find(|lane| {
                lane.class == BlueBrainActionExecutionEligibilityClass::ExecutionBlockedHandoff
            })
            .expect("blocked lane must exist");
        assert!(blocked
            .selection_candidate_basis
            .contains("selection blocked or active deferral"));
        assert!(blocked
            .memory_basis
            .contains("invalidated memory basis blocks eligibility"));

        let safety_failed = canonical_blue_brain_execution_eligibility_diagnostics_map()
            .iter()
            .find(|lane| {
                lane.class
                    == BlueBrainExecutionEligibilityDiagnosticClass::SafetyPrecheckFailedDiagnostic
            })
            .expect("safety-precheck-failed diagnostic must exist");
        assert_eq!(
            safety_failed.reason_class,
            BlueBrainExecutionEligibilityReasonClass::BlockedSafetyPrecheckFailed
        );
        assert!(safety_failed
            .boundary_guard
            .contains("not tool execution failure"));

        let non_canonical = canonical_blue_brain_execution_eligibility_diagnostics_map()
            .iter()
            .find(|lane| {
                lane.class
                    == BlueBrainExecutionEligibilityDiagnosticClass::NonCanonicalInternalOnlyExecutionDiagnostic
            })
            .expect("non-canonical execution diagnostic must exist");
        assert!(non_canonical
            .runtime_feedback_binding
            .contains("canonical=false"));
    }

    #[test]
    fn serie_bb9_prompt1_action_execution_boundary_doc_stays_pinned_to_code_maps() {
        let doc = include_str!(
            "../../../docs/blue_brain_action_execution_eligibility_boundary_serie_bb9_prompt1_v1.md"
        );
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_ACTION_EXECUTION_ELIGIBILITY_MAP"));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_SAFETY_PRECHECK_MAP"));
        assert!(doc.contains("future-action-ready handoff"));
        assert!(doc.contains("execution-eligible handoff"));
        assert!(doc.contains("execution-ineligible handoff"));
        assert!(doc.contains("execution-blocked handoff"));
        assert!(doc.contains("execution-caveated handoff"));
        assert!(doc.contains("execution-insufficient basis"));
        assert!(doc.contains("safety-precheck-passed"));
        assert!(doc.contains("safety-precheck-failed"));
        assert!(doc.contains("safety-precheck-blocked"));
        assert!(doc.contains("safety-precheck-caveated"));
        assert!(doc.contains("safety-precheck-unavailable"));
        assert!(doc.contains("precheck passed"));
        assert!(doc.contains("precheck failed"));
        assert!(doc.contains("precheck blocked"));
        assert!(doc.contains("precheck caveated"));
        assert!(doc.contains("precheck insufficient"));
        assert!(doc.contains("precheck unavailable"));
        assert!(doc.contains("precheck not applicable"));
        assert!(doc.contains("future-action-ready but not execution-eligible"));
        assert!(doc.contains("future-action-ready becomes execution-eligible after precheck"));
        assert!(doc.contains("execution-eligible != executed action"));
        assert!(doc.contains("keine Tool-Execution-Engine"));
        assert!(doc.contains("keine automatische Action Execution"));
        assert!(doc.contains("keine automatische Compute Invocation"));
        assert!(doc.contains("keine automatische Memory Persistence"));
        assert!(doc.contains("Compute-Kern bleibt maintenance-only"));
        assert!(doc.contains("Hodgkin-Huxley/Kuramoto"));
    }

    #[test]
    fn serie_bb9_prompt2_execution_eligibility_diagnostics_doc_stays_pinned_to_code_maps() {
        let doc = include_str!(
            "../../../docs/blue_brain_execution_eligibility_diagnostics_serie_bb9_prompt2_v1.md"
        );
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_ACTION_EXECUTION_ELIGIBILITY_MAP"));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_SAFETY_PRECHECK_MAP"));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_EXECUTION_ELIGIBILITY_DIAGNOSTICS_MAP"));
        assert!(doc.contains("execution-eligible diagnostic"));
        assert!(doc.contains("execution-ineligible diagnostic"));
        assert!(doc.contains("execution-blocked diagnostic"));
        assert!(doc.contains("execution-caveated diagnostic"));
        assert!(doc.contains("execution-insufficient diagnostic"));
        assert!(doc.contains("safety-precheck-passed diagnostic"));
        assert!(doc.contains("safety-precheck-failed diagnostic"));
        assert!(doc.contains("safety-precheck-blocked diagnostic"));
        assert!(doc.contains("safety-precheck-caveated diagnostic"));
        assert!(doc.contains("safety-precheck-unavailable diagnostic"));
        assert!(doc.contains("non-canonical/internal-only execution diagnostic"));
        assert!(doc.contains("no action execution"));
        assert!(doc.contains("no tool invocation"));
        assert!(doc.contains("no compute invocation"));
        assert!(doc.contains("no memory commit"));
    }

    #[test]
    fn serie_bb5_prompt1_memory_commit_boundary_doc_stays_pinned_to_code_maps() {
        let doc =
            include_str!("../../../docs/blue_brain_memory_commit_boundary_serie_bb5_prompt1_v1.md");
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_MEMORY_COMMIT_BOUNDARY_MAP"));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_COMMIT_ELIGIBILITY_CONDITIONS_MAP"));
        assert!(doc.contains("not a memory candidate"));
        assert!(doc.contains("memory candidate proposed"));
        assert!(doc.contains("memory candidate deferred"));
        assert!(doc.contains("memory candidate rejected"));
        assert!(doc.contains("memory candidate stale"));
        assert!(doc.contains("memory candidate insufficient"));
        assert!(doc.contains("commit-eligible candidate"));
        assert!(doc.contains("future-memory-ready candidate"));
        assert!(doc.contains("committed memory (only if real path exists)"));
        assert!(doc.contains("reference-only / not memory"));
        assert!(doc.contains("non-canonical/internal-only persistence path"));
        assert!(doc.contains("sufficient evidence/reference basis"));
        assert!(doc.contains("selected or accepted candidate status"));
        assert!(doc.contains("non-stale context basis"));
        assert!(doc.contains("no blocking caveat"));
        assert!(doc.contains("no internal/expert-only dependency"));
        assert!(doc.contains("explicit persistence path exists"));
        assert!(doc.contains("future-memory-ready handoff"));
        assert!(doc.contains("no actual memory commit is implemented"));
        assert!(doc.contains("History ≠ Memory"));
        assert!(doc.contains("Snapshot ≠ Memory"));
        assert!(doc.contains("Evidence ≠ Memory"));
        assert!(doc.contains("Replay/Trace ≠ Memory"));
        assert!(doc.contains("Compute-Kern bleibt maintenance-only"));
        assert!(doc.contains("keine Memory-Engine"));
        assert!(doc.contains("Hodgkin-Huxley/Kuramoto"));
    }

    #[test]
    fn blue_brain_future_memory_handoff_state_and_commit_result_maps_keep_classes_distinct() {
        let handoff_map = canonical_blue_brain_future_memory_handoff_state_map();
        assert_eq!(handoff_map.len(), 7);
        assert!(handoff_map
            .iter()
            .any(|lane| lane.class == BlueBrainFutureMemoryHandoffStateClass::HandoffReady));
        assert!(handoff_map
            .iter()
            .any(|lane| lane.class == BlueBrainFutureMemoryHandoffStateClass::HandoffDeferred));
        assert!(handoff_map
            .iter()
            .any(|lane| lane.class == BlueBrainFutureMemoryHandoffStateClass::HandoffBlocked));
        assert!(handoff_map
            .iter()
            .any(|lane| lane.class == BlueBrainFutureMemoryHandoffStateClass::HandoffRejected));
        assert!(handoff_map
            .iter()
            .any(|lane| lane.class == BlueBrainFutureMemoryHandoffStateClass::HandoffCaveated));
        assert!(handoff_map
            .iter()
            .any(|lane| lane.class == BlueBrainFutureMemoryHandoffStateClass::HandoffUnavailable));
        assert!(handoff_map.iter().any(|lane| {
            lane.class == BlueBrainFutureMemoryHandoffStateClass::HandoffInternalOnlyNonCanonical
        }));
        assert!(handoff_map
            .iter()
            .find(|lane| lane.class == BlueBrainFutureMemoryHandoffStateClass::HandoffReady)
            .expect("handoff ready lane")
            .state_semantics
            .contains("non-commit"));
        assert!(handoff_map
            .iter()
            .find(|lane| lane.class == BlueBrainFutureMemoryHandoffStateClass::HandoffUnavailable)
            .expect("handoff unavailable lane")
            .state_semantics
            .contains("cannot be performed"));

        let commit_map = canonical_blue_brain_commit_result_semantics_map();
        assert_eq!(commit_map.len(), 9);
        assert!(commit_map
            .iter()
            .any(|lane| lane.class == BlueBrainCommitResultClass::CommitUnavailable));
        assert!(commit_map
            .iter()
            .any(|lane| lane.class == BlueBrainCommitResultClass::CommitDeferred));
        assert!(commit_map
            .iter()
            .any(|lane| lane.class == BlueBrainCommitResultClass::CommitCommitted));
        assert!(commit_map
            .iter()
            .any(|lane| lane.class == BlueBrainCommitResultClass::CommitCommittedWithCaveats));
        assert!(commit_map
            .iter()
            .any(|lane| lane.class == BlueBrainCommitResultClass::CommitRejected));
        assert!(commit_map
            .iter()
            .any(|lane| lane.class == BlueBrainCommitResultClass::CommitBlocked));
        assert!(commit_map
            .iter()
            .any(|lane| lane.class == BlueBrainCommitResultClass::CommitFailed));
        assert!(commit_map
            .iter()
            .any(|lane| lane.class == BlueBrainCommitResultClass::CommitNoOp));
        assert!(commit_map
            .iter()
            .any(|lane| lane.class == BlueBrainCommitResultClass::CommitReferenceRecordedOnly));
        assert!(commit_map
            .iter()
            .find(|lane| lane.class == BlueBrainCommitResultClass::CommitUnavailable)
            .expect("commit unavailable lane")
            .result_semantics
            .contains("canonical baseline result"));
        assert!(commit_map
            .iter()
            .find(|lane| lane.class == BlueBrainCommitResultClass::CommitReferenceRecordedOnly)
            .expect("reference only lane")
            .canonical_guard
            .contains("must not be classified as memory commit result"));
    }

    #[test]
    fn serie_bb5_prompt2_future_memory_handoff_and_commit_result_doc_stays_pinned_to_code_maps() {
        let doc = include_str!(
            "../../../docs/blue_brain_future_memory_handoff_commit_result_serie_bb5_prompt2_v1.md"
        );
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_FUTURE_MEMORY_HANDOFF_STATE_MAP"));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_COMMIT_RESULT_SEMANTICS_MAP"));
        assert!(doc.contains("handoff ready"));
        assert!(doc.contains("handoff deferred"));
        assert!(doc.contains("handoff blocked"));
        assert!(doc.contains("handoff rejected"));
        assert!(doc.contains("handoff caveated"));
        assert!(doc.contains("handoff unavailable"));
        assert!(doc.contains("handoff internal-only/non-canonical"));
        assert!(doc.contains("commit unavailable"));
        assert!(doc.contains("commit deferred"));
        assert!(doc.contains("committed"));
        assert!(doc.contains("committed with caveats"));
        assert!(doc.contains("commit rejected"));
        assert!(doc.contains("commit blocked"));
        assert!(doc.contains("commit failed"));
        assert!(doc.contains("commit no-op"));
        assert!(doc.contains("reference recorded only"));
        assert!(doc.contains("no actual memory commit is implemented in the current baseline"));
        assert!(doc.contains("handoff-ready is not a memory commit"));
        assert!(doc.contains("History/Snapshot/Evidence/Replay are reference-only"));
        assert!(doc.contains("Compute-Kern bleibt maintenance-only"));
        assert!(doc.contains("keine Memory-Engine"));
        assert!(doc.contains("Hodgkin-Huxley/Kuramoto"));
    }

    #[test]
    fn blue_brain_memory_commit_diagnostics_map_keeps_required_classes_and_canonical_reasons() {
        let map = canonical_blue_brain_memory_commit_diagnostics_map();
        assert_eq!(map.len(), 10);
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainMemoryCommitDiagnosticClass::HandoffDiagnostic));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainMemoryCommitDiagnosticClass::CommitEligibilityDiagnostic
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainMemoryCommitDiagnosticClass::CommitRejectedDiagnostic
        }));
        assert!(map.iter().any(
            |lane| lane.class == BlueBrainMemoryCommitDiagnosticClass::CommitBlockedDiagnostic
        ));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainMemoryCommitDiagnosticClass::CommitDeferredDiagnostic
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainMemoryCommitDiagnosticClass::CommitCaveatedDiagnostic
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainMemoryCommitDiagnosticClass::CommitUnavailableDiagnostic
        }));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainMemoryCommitDiagnosticClass::CommittedIfPresentDiagnostic
        }));
        assert!(map.iter().any(
            |lane| lane.class == BlueBrainMemoryCommitDiagnosticClass::NoPersistenceDiagnostic
        ));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainMemoryCommitDiagnosticClass::NonCanonicalInternalOnlyDiagnostic
        }));

        let rejected = map
            .iter()
            .find(|lane| {
                lane.class == BlueBrainMemoryCommitDiagnosticClass::CommitRejectedDiagnostic
            })
            .expect("rejected diagnostic lane");
        assert!(rejected
            .compact_reason
            .contains("weak_or_insufficient_evidence_or_candidate_state"));

        let blocked = map
            .iter()
            .find(|lane| {
                lane.class == BlueBrainMemoryCommitDiagnosticClass::CommitBlockedDiagnostic
            })
            .expect("blocked diagnostic lane");
        assert!(blocked
            .compact_reason
            .contains("stale_context_or_missing_persistence_path_or_internal_only_dependency"));

        let unavailable = map
            .iter()
            .find(|lane| {
                lane.class == BlueBrainMemoryCommitDiagnosticClass::CommitUnavailableDiagnostic
            })
            .expect("unavailable diagnostic lane");
        assert!(unavailable
            .compact_reason
            .contains("no_actual_memory_subsystem_exists"));

        let caveated = map
            .iter()
            .find(|lane| {
                lane.class == BlueBrainMemoryCommitDiagnosticClass::CommitCaveatedDiagnostic
            })
            .expect("caveated diagnostic lane");
        assert!(caveated.compact_reason.contains("partial_reference_basis"));
    }

    #[test]
    fn blue_brain_memory_commit_diagnostics_bind_back_to_candidate_selection_runtime_and_boundaries(
    ) {
        let map = canonical_blue_brain_memory_commit_diagnostics_map();
        let handoff = map
            .iter()
            .find(|lane| lane.class == BlueBrainMemoryCommitDiagnosticClass::HandoffDiagnostic)
            .expect("handoff lane");
        assert!(handoff
            .candidate_lifecycle_binding
            .contains("accepted_for_future_memory_handling"));
        assert!(handoff
            .selection_deferral_binding
            .contains("selection_diagnostic"));
        assert!(handoff
            .runtime_context_binding
            .contains("context updated but not persisted"));

        let no_persistence = map
            .iter()
            .find(|lane| {
                lane.class == BlueBrainMemoryCommitDiagnosticClass::NoPersistenceDiagnostic
            })
            .expect("no persistence lane");
        assert!(no_persistence
            .runtime_context_binding
            .contains("evidence attached but not committed"));
        assert!(no_persistence
            .canonical_guard
            .contains("history/snapshot/evidence/replay/trace references are not memory commit"));

        let internal = map
            .iter()
            .find(|lane| {
                lane.class
                    == BlueBrainMemoryCommitDiagnosticClass::NonCanonicalInternalOnlyDiagnostic
            })
            .expect("internal-only lane");
        assert!(internal.runtime_context_binding.contains("canonical=false"));
        assert!(internal
            .canonical_guard
            .contains("excluded unless down-mapped"));
    }

    #[test]
    fn serie_bb5_prompt3_memory_commit_diagnostics_doc_stays_pinned_to_code_maps() {
        let doc = include_str!(
            "../../../docs/blue_brain_memory_commit_diagnostics_feedback_serie_bb5_prompt3_v1.md"
        );
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_MEMORY_COMMIT_DIAGNOSTICS_MAP"));
        assert!(doc.contains("handoff diagnostic"));
        assert!(doc.contains("commit eligibility diagnostic"));
        assert!(doc.contains("commit rejected diagnostic"));
        assert!(doc.contains("commit blocked diagnostic"));
        assert!(doc.contains("commit deferred diagnostic"));
        assert!(doc.contains("commit caveated diagnostic"));
        assert!(doc.contains("commit unavailable diagnostic"));
        assert!(doc.contains("committed-if-present diagnostic"));
        assert!(doc.contains("no-persistence diagnostic"));
        assert!(doc.contains("non-canonical/internal-only diagnostic"));
        assert!(doc.contains("weak or insufficient evidence"));
        assert!(doc.contains("candidate state"));
        assert!(doc.contains("stale context"));
        assert!(doc.contains("missing persistence path"));
        assert!(doc.contains("internal-only dependency"));
        assert!(doc.contains("no actual memory subsystem exists"));
        assert!(doc.contains("partial reference basis"));
        assert!(doc.contains("context updated but not persisted"));
        assert!(doc.contains("evidence attached but not committed"));
        assert!(doc.contains("History ≠ Memory Commit"));
        assert!(doc.contains("Snapshot ≠ Memory Commit"));
        assert!(doc.contains("Evidence Reference ≠ Memory Commit"));
        assert!(doc.contains("Replay/Trace Reference ≠ Memory Commit"));
        assert!(doc.contains("Compute-Kern bleibt maintenance-only"));
        assert!(doc.contains("keine Monitoring- oder Explainability-Plattform"));
        assert!(doc.contains("Hodgkin-Huxley/Kuramoto"));
    }

    #[test]
    fn serie_bb5_prompt4_readiness_sweep_doc_keeps_minimal_memory_commit_line_explicit() {
        let doc = include_str!("../../../docs/blue_brain_readiness_sweep_serie_bb5_prompt4_v1.md");
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_MEMORY_COMMIT_BOUNDARY_MAP"));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_COMMIT_ELIGIBILITY_CONDITIONS_MAP"));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_FUTURE_MEMORY_HANDOFF_STATE_MAP"));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_COMMIT_RESULT_SEMANTICS_MAP"));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_MEMORY_COMMIT_DIAGNOSTICS_MAP"));
        assert!(doc.contains("stable minimal memory-commit boundary"));
        assert!(doc.contains("future-memory-ready / preparatory only"));
        assert!(doc.contains("reference-only / not memory"));
        assert!(doc.contains("non-canonical / internal-only"));
        assert!(doc.contains("intentionally deferred"));
        assert!(doc.contains("commit-eligible"));
        assert!(doc.contains("future-memory-ready"));
        assert!(doc.contains("handoff-ready"));
        assert!(doc.contains("reference recorded only"));
        assert!(doc.contains("commit unavailable"));
        assert!(doc
            .contains("actual memory commit ist im aktuellen Repo weiterhin nicht implementiert"));
        assert!(doc.contains("Compute-Kern bleibt maintenance-only"));
        assert!(doc.contains("History/Snapshot/Evidence/Replay/Trace"));
        assert!(doc.contains("Priorität: Serie BB7 zuerst"));
        assert!(doc.contains("Hodgkin-Huxley/Kuramoto"));
    }

    #[test]
    fn blue_brain_runtime_feedback_map_keeps_canonical_feedback_classes_explicit() {
        let map = canonical_blue_brain_runtime_feedback_map();
        assert_eq!(map.len(), 10);
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainRuntimeFeedbackClass::ComputeResultFeedback));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainRuntimeFeedbackClass::StatusTrustFeedback));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainRuntimeFeedbackClass::EvidenceReferenceFeedback));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainRuntimeFeedbackClass::DiagnosticCaveatFeedback));
        assert!(map
            .iter()
            .any(|lane| lane.class == BlueBrainRuntimeFeedbackClass::ContextUptakeFeedback));
        assert!(map.iter().any(|lane| {
            lane.class == BlueBrainRuntimeFeedbackClass::NonCanonicalInternalExpertFeedback
        }));
    }

    #[test]
    fn blue_brain_runtime_feedback_map_preserves_result_status_evidence_context_boundaries() {
        let map = canonical_blue_brain_runtime_feedback_map();
        let result_lane = map
            .iter()
            .find(|lane| lane.lane == "blue_brain_feedback_result_integrated_current_runtime_state")
            .expect("result integrated lane");
        assert!(result_lane
            .runtime_feedback_semantics
            .contains("result integrated into current runtime state"));
        assert!(result_lane
            .memory_boundary
            .contains("no memory persistence implied"));

        let blocked_result_lane = map
            .iter()
            .find(|lane| lane.lane == "blue_brain_feedback_result_rejected_or_blocked")
            .expect("blocked result lane");
        assert!(blocked_result_lane
            .runtime_feedback_semantics
            .contains("rejected/blocked"));
        assert!(blocked_result_lane
            .transition_binding
            .contains("compute_trigger_blocked_insufficient_context"));

        let status_lane = map
            .iter()
            .find(|lane| lane.lane == "blue_brain_feedback_status_trust_current_to_insufficient")
            .expect("status/trust lane");
        assert!(status_lane
            .runtime_feedback_semantics
            .contains("current|trusted, partial, stale, caveated, degraded, insufficient/blocked"));

        let evidence_lane = map
            .iter()
            .find(|lane| lane.lane == "blue_brain_feedback_evidence_observed_and_attached")
            .expect("evidence observed lane");
        assert!(evidence_lane
            .runtime_feedback_semantics
            .contains("evidence observed and attached"));
        assert!(evidence_lane
            .memory_boundary
            .contains("no automatic memory commit"));

        let context_lane = map
            .iter()
            .find(|lane| {
                lane.lane
                    == "blue_brain_feedback_context_uptake_transient_memory_adjacent_candidate"
            })
            .expect("context uptake lane");
        assert!(context_lane
            .runtime_feedback_semantics
            .contains("transient runtime context"));
        assert!(context_lane
            .memory_boundary
            .contains("actual memory persistence not implemented in BB2"));

        let internal_lane = map
            .iter()
            .find(|lane| lane.lane == "blue_brain_feedback_non_canonical_internal_expert_only")
            .expect("non-canonical internal lane");
        assert!(internal_lane
            .non_canonical_boundary
            .contains("down-mapped to outward status/evidence references"));
    }

    #[test]
    fn serie_bb2_prompt4_runtime_feedback_doc_stays_pinned_to_feedback_map() {
        let doc = include_str!("../../../docs/blue_brain_runtime_feedback_serie_bb2_prompt4_v1.md");
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("CANONICAL_BLUE_BRAIN_RUNTIME_FEEDBACK_MAP"));
        assert!(doc.contains("compute_result_feedback"));
        assert!(doc.contains("status_trust_feedback"));
        assert!(doc.contains("evidence_reference_feedback"));
        assert!(doc.contains("diagnostic_caveat_feedback"));
        assert!(doc.contains("context_uptake_feedback"));
        assert!(doc.contains("non_canonical_internal_expert_feedback"));
        assert!(doc.contains("blue_brain_feedback_result_integrated_current_runtime_state"));
        assert!(doc.contains("blue_brain_feedback_result_rejected_or_blocked"));
        assert!(doc.contains("blue_brain_feedback_result_integrated_with_caveat"));
        assert!(doc.contains("blue_brain_feedback_status_trust_current_to_insufficient"));
        assert!(doc.contains("blue_brain_feedback_evidence_observed_and_attached"));
        assert!(doc.contains("blue_brain_feedback_evidence_caveated_partial_or_insufficient"));
        assert!(doc.contains("blue_brain_feedback_diagnostic_only_caveat"));
        assert!(doc.contains("blue_brain_feedback_trigger_blocking_or_context_uptake_caveat"));
        assert!(
            doc.contains("blue_brain_feedback_context_uptake_transient_memory_adjacent_candidate")
        );
        assert!(doc.contains("blue_brain_feedback_non_canonical_internal_expert_only"));
        assert!(doc.contains("keine Reasoning-Engine"));
        assert!(doc.contains("kein Memory-Commit"));
    }

    #[test]
    fn serie_bb2_prompt5_readiness_doc_keeps_runtime_baseline_and_compute_maintenance_boundary() {
        let doc = include_str!("../../../docs/blue_brain_readiness_sweep_serie_bb2_prompt5_v1.md");
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("stable Blue-Brain runtime foundation"));
        assert!(doc.contains("runtime-usable with caveats"));
        assert!(doc.contains("preparatory / memory-adjacent only"));
        assert!(doc.contains("internal-only / non-canonical"));
        assert!(doc.contains("intentionally deferred"));

        assert!(doc.contains("blue_brain_state_bearing_surface"));
        assert!(doc.contains("blue_brain_inference_bearing_surface"));
        assert!(doc.contains("blue_brain_status_health_trust_surface"));
        assert!(doc.contains("blue_brain_evidence_replay_facing_surface"));

        assert!(doc.contains("blue_brain_transition_context_used_for_compute_trigger"));
        assert!(doc.contains("blue_brain_transition_compute_trigger_from_context_availability"));
        assert!(doc.contains("blue_brain_transition_compute_trigger_from_inference_required"));
        assert!(doc.contains("blue_brain_transition_compute_trigger_suppressed_internal_only_path"));

        assert!(doc.contains("blue_brain_feedback_result_integrated_current_runtime_state"));
        assert!(doc.contains("blue_brain_feedback_status_trust_current_to_insufficient"));
        assert!(doc.contains("blue_brain_feedback_evidence_observed_and_attached"));
        assert!(
            doc.contains("blue_brain_feedback_context_uptake_transient_memory_adjacent_candidate")
        );
        assert!(doc
            .contains("blue_brain_transition_memory_adjacent_candidate_identified_not_committed"));
        assert!(doc.contains("kein Memory-Commit"));
        assert!(doc.contains("maintenance-only Core"));
        assert!(doc.contains("Priorität 1: Serie BB3"));
        assert!(doc.contains("Hodgkin-Huxley/Kuramoto"));
    }

    #[test]
    fn serie_o_maintenance_boundary_doc_keeps_minimal_change_classes_explicit() {
        let doc = include_str!("../../../docs/compute_core_maintenance_boundary_serie_o_v1.md");
        let line = canonical_final_reference_line();

        assert!(doc.contains(line.execution_core));

        assert!(doc.contains("maintenance_safe_change"));
        assert!(doc.contains("maintenance_safe_with_care"));
        assert!(doc.contains("not_maintenance_only_requires_new_integration_or_buildout"));

        assert!(doc.contains("bug fix"));
        assert!(doc.contains("small contract consistency fix"));
        assert!(doc.contains("narrow drift correction"));
        assert!(doc.contains("doc/readiness/reference alignment"));
        assert!(doc.contains("small guard/check hardening"));

        assert!(doc.contains("new runtime feature"));
        assert!(doc.contains("broader new integration"));
        assert!(doc.contains("new backend/device capability expansion"));
        assert!(doc.contains("new workflow/control surface"));
        assert!(doc.contains("architectural reshaping"));

        assert!(doc.contains("keine zweite Wahrheitsquelle"));
        assert!(doc.contains("compute_core_drift_prevention_checks_serie_o_v1.md"));
    }

    #[test]
    fn drift_prevention_check_map_keeps_four_minimal_load_bearing_classes() {
        let checks = canonical_drift_prevention_check_map();
        assert_eq!(checks.len(), 4);
        assert!(checks.iter().any(|check| {
            check.class == DriftPreventionCheckClass::ReferenceLineConsistency
                && check.check_id == "reference_line_consistency"
        }));
        assert!(checks.iter().any(|check| {
            check.class == DriftPreventionCheckClass::OutwardFacingContractConsistency
                && check.check_id == "outward_facing_contract_consistency"
        }));
        assert!(checks.iter().any(|check| {
            check.class == DriftPreventionCheckClass::SharedCoreSemantics
                && check.check_id == "shared_core_semantics_consistency"
        }));
        assert!(checks.iter().any(|check| {
            check.class == DriftPreventionCheckClass::DocCodeAlignment
                && check.check_id == "doc_code_alignment"
        }));
    }

    #[test]
    fn drift_prevention_checks_stay_pinned_to_canonical_outward_and_shared_semantics() {
        let line = canonical_final_reference_line();
        let checks = canonical_drift_prevention_check_map();

        let reference = checks
            .iter()
            .find(|check| check.class == DriftPreventionCheckClass::ReferenceLineConsistency)
            .expect("reference check");
        assert_eq!(reference.guarded_line, line.execution_core);

        let outward = checks
            .iter()
            .find(|check| {
                check.class == DriftPreventionCheckClass::OutwardFacingContractConsistency
            })
            .expect("outward check");
        assert!(outward
            .guarded_line
            .contains("status_evidence_export_surface"));
        assert!(outward.guarded_line.contains("integration_hook_view"));

        let shared = checks
            .iter()
            .find(|check| check.class == DriftPreventionCheckClass::SharedCoreSemantics)
            .expect("shared-core check");
        assert!(shared.guarded_line.contains("blocked/failed/no_op"));
        assert!(shared
            .guarded_line
            .contains("current/partial/stale/caveated/degraded"));
    }

    #[test]
    fn serie_o_drift_prevention_checks_doc_stays_tied_to_canonical_line() {
        let doc = include_str!("../../../docs/compute_core_drift_prevention_checks_serie_o_v1.md");
        let line = canonical_final_reference_line();

        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("reference_line_consistency"));
        assert!(doc.contains("outward_facing_contract_consistency"));
        assert!(doc.contains("shared_core_semantics_consistency"));
        assert!(doc.contains("doc_code_alignment"));
        assert!(doc.contains("blocked/failed/no_op"));
        assert!(doc.contains("current/partial/stale/caveated/degraded"));
        assert!(doc.contains("status_evidence_export_surface"));
        assert!(doc.contains("integration_hook_view"));
        assert!(doc.contains("keine CI-/Governance-/Policy-Plattform"));
    }

    #[test]
    fn serie_o_minimal_follow_up_canon_is_consistent_across_reference_and_exit_docs() {
        let final_reference_doc = include_str!("../../../docs/final_reference_line_serie_j_v1.md");
        let exit_doc = include_str!("../../../docs/real_compute_exit_dossier_serie_l_v1.md");

        for doc in [final_reference_doc, exit_doc] {
            assert!(doc.contains("allowed_maintenance_safe_changes"));
            assert!(doc.contains("discouraged_but_possible_with_care"));
            assert!(doc.contains("not_in_maintenance_lane"));
            assert!(doc.contains("Serie O"));
            assert!(doc.contains("geschlossen"));
            assert!(doc.contains("compute_core_maintenance_boundary_serie_o_v1.md"));
        }
    }

    #[test]
    fn serie_o_prompt4_readiness_sweep_keeps_matrix_follow_up_line_and_priority_explicit() {
        let doc = include_str!("../../../docs/serie_o_readiness_sweep_prompt4_v1.md");
        let line = canonical_final_reference_line();

        assert!(doc.contains(line.execution_core));

        assert!(doc.contains("maintenance-safe"));
        assert!(doc.contains("maintenance-safe with care"));
        assert!(doc.contains("outside maintenance lane"));

        assert!(doc.contains("maintenance_safe_change"));
        assert!(doc.contains("maintenance_safe_with_care"));
        assert!(doc.contains("not_maintenance_only_requires_new_integration_or_buildout"));

        assert!(doc.contains("Serie P"));
        assert!(doc.contains("Serie Q"));
        assert!(doc.contains("Serie R"));
        assert!(doc.contains("Priorität: Serie P"));
    }

    #[test]
    fn first_domain_rollout_candidate_map_keeps_minimal_classification_surface() {
        let map = canonical_first_domain_rollout_candidate_map();
        assert!(map.iter().any(|lane| {
            lane.rollout_class == DomainRolloutCandidateClass::RolloutReadyCandidate
        }));
        assert!(map.iter().any(|lane| {
            lane.rollout_class == DomainRolloutCandidateClass::RolloutPlausibleWithCaveats
        }));
        assert!(map.iter().any(|lane| {
            lane.rollout_class == DomainRolloutCandidateClass::MixedTransitionalCandidate
        }));
        assert!(map.iter().any(|lane| {
            lane.rollout_class == DomainRolloutCandidateClass::NotRealRolloutCandidateNow
        }));
    }

    #[test]
    fn rollout_ready_candidate_is_pinned_to_canonical_outward_contracts_only() {
        let ready: Vec<_> = canonical_first_domain_rollout_candidate_map()
            .iter()
            .filter(|lane| lane.rollout_class == DomainRolloutCandidateClass::RolloutReadyCandidate)
            .collect();
        assert_eq!(ready.len(), 1);
        let lane = ready[0];
        assert_eq!(lane.candidate, "ops_compute_probe");
        assert!(lane
            .outward_execution_contract
            .contains("CanonicalComputeEntryPoint::submit"));
        assert!(lane
            .outward_status_evidence_surface
            .contains("status_evidence_export_surface"));
        assert!(lane
            .integration_safe_hook_posture
            .contains("read_only_integration_safe"));
        assert!(lane
            .excluded_internal_or_legacy_paths
            .contains("build_backend"));
        assert!(lane
            .excluded_internal_or_legacy_paths
            .contains("domains/ai*"));
    }

    #[test]
    fn mixed_or_internal_candidates_never_appear_as_rollout_ready() {
        let map = canonical_first_domain_rollout_candidate_map();
        assert!(map
            .iter()
            .filter(|lane| {
                matches!(
                    lane.rollout_class,
                    DomainRolloutCandidateClass::MixedTransitionalCandidate
                        | DomainRolloutCandidateClass::NotRealRolloutCandidateNow
                )
            })
            .all(|lane| lane.candidate != "ops_compute_probe"));
        assert!(map.iter().any(|lane| {
            lane.candidate == "replay_diff_backend_recompute"
                && lane.rollout_class == DomainRolloutCandidateClass::MixedTransitionalCandidate
        }));
        assert!(map.iter().any(|lane| {
            lane.candidate == "domains_ai_compat_lane"
                && lane.rollout_class == DomainRolloutCandidateClass::NotRealRolloutCandidateNow
        }));
    }

    #[test]
    fn serie_p_first_domain_rollout_doc_stays_pinned_to_canonical_contracts_and_boundaries() {
        let doc = include_str!("../../../docs/first_domain_rollout_candidate_serie_p_v1.md");
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("rollout-ready candidate"));
        assert!(doc.contains("rollout-plausible with caveats"));
        assert!(doc.contains("mixed/transitional candidate"));
        assert!(doc.contains("not a real rollout candidate now"));
        assert!(doc.contains("ops_compute_probe"));
        assert!(doc.contains("runtime_orchestrator_env_bootstrap"));
        assert!(doc.contains("replay_diff_backend_recompute"));
        assert!(doc.contains("bench_compute_subcommand"));
        assert!(doc.contains("domains_ai_compat_lane"));
        assert!(doc.contains("status_evidence_export_surface"));
        assert!(doc.contains("integration_hook_view"));
        assert!(doc.contains("build_backend(kind=stub|candle|worker)"));
        assert!(doc.contains("no second integration language"));
    }

    #[test]
    fn first_domain_rollout_completion_map_marks_ops_probe_as_aligned() {
        let map = canonical_first_domain_rollout_completion_map();
        assert_eq!(map.len(), 1);
        let lane = map[0];
        assert_eq!(lane.rollout_case, "ops_compute_probe");
        assert_eq!(
            lane.completion_status,
            FirstDomainRolloutCompletionStatus::Aligned
        );
        assert!(lane
            .execution_contract_check
            .contains("CanonicalComputeEntryPoint::submit"));
        assert!(lane
            .outward_status_evidence_check
            .contains("status_evidence_export_surface"));
        assert!(lane
            .integration_safe_hook_check
            .contains("read_only_integration_safe"));
        assert!(lane
            .hidden_legacy_dependency_check
            .contains("build_backend(kind=stub|candle|worker)"));
        assert!(lane.hidden_legacy_dependency_check.contains("domains/ai*"));
    }

    #[test]
    fn first_domain_rollout_completion_statuses_are_narrow_and_non_ambiguous() {
        let all = [
            FirstDomainRolloutCompletionStatus::Aligned,
            FirstDomainRolloutCompletionStatus::AlignedWithCaveats,
            FirstDomainRolloutCompletionStatus::MixedTransitional,
            FirstDomainRolloutCompletionStatus::NotYetTrueRolloutCompletion,
        ];
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn serie_p_prompt3_completion_doc_stays_pinned_to_single_rollout_proof_case() {
        let doc =
            include_str!("../../../docs/first_domain_rollout_completion_serie_p_prompt3_v1.md");
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("aligned"));
        assert!(doc.contains("aligned with caveats"));
        assert!(doc.contains("mixed/transitional"));
        assert!(doc.contains("not yet true rollout completion"));
        assert!(doc.contains("ops_compute_probe"));
        assert!(doc.contains("status_evidence_export_surface"));
        assert!(doc.contains("integration_hook_view"));
        assert!(doc.contains("build_backend(kind=stub|candle|worker)"));
        assert!(doc.contains("domains/ai*"));
    }

    #[test]
    fn serie_p_prompt4_closure_doc_keeps_matrix_rollout_line_and_priority_explicit() {
        let doc = include_str!("../../../docs/serie_p_readiness_sweep_prompt4_v1.md");
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));

        assert!(doc.contains("real domain rollout line established"));
        assert!(doc.contains("rollout-usable with caveats"));
        assert!(doc.contains("transitional / not yet aligned"));
        assert!(doc.contains("intentionally deferred"));

        assert!(doc.contains("ops_compute_probe"));
        assert!(doc.contains("runtime_orchestrator_env_bootstrap"));
        assert!(doc.contains("replay_diff_backend_recompute"));
        assert!(doc.contains("bench_compute_subcommand"));
        assert!(doc.contains("domains_ai_compat_lane"));

        assert!(doc.contains("Priorität: Serie S"));
        assert!(doc.contains("follow-up integration work"));
        assert!(doc.contains("not compute-core completion work"));
    }

    #[test]
    fn post_rollout_adoption_map_keeps_minimal_narrow_classes_explicit() {
        let map = canonical_post_rollout_adoption_map();
        assert!(map
            .iter()
            .any(|lane| { lane.adoption_class == PostRolloutAdoptionClass::AlreadyAligned }));
        assert!(map.iter().any(|lane| {
            lane.adoption_class == PostRolloutAdoptionClass::FirstRealRolloutEstablished
        }));
        assert!(map.iter().any(|lane| {
            lane.adoption_class == PostRolloutAdoptionClass::BroaderAdoptionReviewCandidate
        }));
        assert!(map
            .iter()
            .any(|lane| { lane.adoption_class == PostRolloutAdoptionClass::NotPursuedNow }));
    }

    #[test]
    fn post_rollout_map_keeps_orchestrator_and_replay_as_review_candidates() {
        let next: Vec<_> = canonical_post_rollout_adoption_map()
            .iter()
            .filter(|lane| {
                lane.adoption_class == PostRolloutAdoptionClass::BroaderAdoptionReviewCandidate
            })
            .collect();
        assert_eq!(next.len(), 2);
        let lane = next
            .iter()
            .find(|lane| lane.surface == "runtime_orchestrator_env_bootstrap")
            .expect("runtime_orchestrator_env_bootstrap lane must be present");
        assert_eq!(lane.surface, "runtime_orchestrator_env_bootstrap");
        assert!(lane
            .outward_contract_fit
            .contains("CanonicalComputeEntryPoint::submit"));
        assert!(lane
            .outward_contract_fit
            .contains("status_evidence_export_surface"));
        assert!(lane
            .legacy_internal_dependency_posture
            .contains("build_backend(kind=stub|candle|worker)"));
    }

    #[test]
    fn post_rollout_map_keeps_replay_lane_explicitly_review_only() {
        let lane = canonical_post_rollout_adoption_map()
            .iter()
            .find(|lane| lane.surface == "replay_diff_backend_recompute")
            .expect("replay_diff_backend_recompute lane must be present");
        assert_eq!(
            lane.adoption_class,
            PostRolloutAdoptionClass::BroaderAdoptionReviewCandidate
        );
        assert!(lane
            .outward_contract_fit
            .contains("lacks canonical outward status/service interface"));
        assert!(lane.caveat.contains("review-only candidate"));
    }

    #[test]
    fn post_rollout_map_separates_baseline_and_not_pursued_surfaces_explicitly() {
        let map = canonical_post_rollout_adoption_map();
        assert!(map.iter().any(|lane| {
            lane.surface == "final_compute_reference_line"
                && lane.adoption_class == PostRolloutAdoptionClass::AlreadyAligned
        }));
        assert!(map.iter().any(|lane| {
            lane.surface == "ops_compute_probe"
                && lane.adoption_class == PostRolloutAdoptionClass::FirstRealRolloutEstablished
        }));
        assert!(map.iter().any(|lane| {
            lane.surface == "bench_compute_subcommand"
                && lane.adoption_class == PostRolloutAdoptionClass::NotPursuedNow
        }));
        assert!(map.iter().any(|lane| {
            lane.surface == "domains_ai_compat_lane"
                && lane.adoption_class == PostRolloutAdoptionClass::NotPursuedNow
        }));
    }

    #[test]
    fn serie_q_post_rollout_adoption_doc_stays_pinned_to_single_rollout_anchor_language() {
        let doc = include_str!("../../../docs/serie_q_post_rollout_adoption_map_v1.md");
        let line = canonical_final_reference_line();
        assert!(doc.contains(line.execution_core));
        assert!(doc.contains("genuine next adoption candidate"));
        assert!(doc.contains("plausible but deferred"));
        assert!(doc.contains("reviewed and not pursued now"));
        assert!(doc.contains("not meaningful as compute-backed adoption"));
        assert!(doc.contains("ops_compute_probe"));
        assert!(doc.contains("runtime_orchestrator_env_bootstrap"));
        assert!(doc.contains("replay_diff_backend_recompute"));
        assert!(doc.contains("bench_compute_subcommand"));
        assert!(doc.contains("domains_ai_compat_lane"));
        assert!(doc.contains("status_evidence_export_surface"));
        assert!(doc.contains("integration_hook_view"));
        assert!(doc.contains("Prioritized next direction: Serie S"));
        assert!(doc.contains("review + prioritization only"));
        assert!(doc.contains("no unplanned rollout"));
    }
}
