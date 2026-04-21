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

pub fn canonical_compute_reference_map() -> &'static [ComputeReferenceLane] {
    &CANONICAL_COMPUTE_REFERENCE_MAP
}

pub fn canonical_production_reference_lane() -> ComputeReferenceLane {
    CANONICAL_COMPUTE_REFERENCE_MAP[0]
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
