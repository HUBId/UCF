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
