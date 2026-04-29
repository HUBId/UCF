#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainDynamicsDiagnosticClass {
    DynamicsDiagnosticObserved,
    KuramotoModulationDiagnostic,
    HodgkinHuxleySimulationDiagnostic,
    DynamicsCaveated,
    DynamicsInsufficient,
    DynamicsFailed,
    DynamicsUnavailable,
    DynamicsIgnored,
    NonCanonicalInternalOnlyDynamicsDiagnostic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainDynamicsDiagnosticLane {
    pub diagnostic_class: BlueBrainDynamicsDiagnosticClass,
    pub lane: &'static str,
    pub canonical_guard: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainDynamicsExecutionFeedbackState {
    ExecutionInformedDynamicsInput,
    ReferenceInformedDynamicsInput,
    FailedExecutionFeedbackBasis,
    CancelledExecutionFeedbackBasis,
    InsufficientDynamicsFeedbackBasis,
    BlockedDynamicsFeedbackBasis,
    UnavailableDynamicsFeedbackBasis,
    DiagnosticOnlyDynamicsFeedback,
    NonCanonicalInternalOnlyFeedbackPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainDynamicsExecutionFeedbackLane {
    pub state: BlueBrainDynamicsExecutionFeedbackState,
    pub lane: &'static str,
    pub canonical_guard: &'static str,
}

pub const CANONICAL_BLUE_BRAIN_DYNAMICS_EXECUTION_FEEDBACK_MAP:
    [BlueBrainDynamicsExecutionFeedbackLane; 9] = [
    BlueBrainDynamicsExecutionFeedbackLane {
        state: BlueBrainDynamicsExecutionFeedbackState::ExecutionInformedDynamicsInput,
        lane: "blue_brain_dynamics_execution_informed_input",
        canonical_guard: "canonical execution result references may inform dynamics modulation in advisory-only mode",
    },
    BlueBrainDynamicsExecutionFeedbackLane {
        state: BlueBrainDynamicsExecutionFeedbackState::ReferenceInformedDynamicsInput,
        lane: "blue_brain_dynamics_reference_informed_input",
        canonical_guard: "bounded canonical references/context may inform dynamics diagnostics without direct execution authority",
    },
    BlueBrainDynamicsExecutionFeedbackLane {
        state: BlueBrainDynamicsExecutionFeedbackState::FailedExecutionFeedbackBasis,
        lane: "blue_brain_dynamics_feedback_basis_failed",
        canonical_guard: "failed execution basis may only inform caveated diagnostics and never promotes to successful/current modulation basis",
    },
    BlueBrainDynamicsExecutionFeedbackLane {
        state: BlueBrainDynamicsExecutionFeedbackState::CancelledExecutionFeedbackBasis,
        lane: "blue_brain_dynamics_feedback_basis_cancelled",
        canonical_guard: "cancelled execution basis remains weak/caveated and cannot be interpreted as failed/completed/current basis",
    },
    BlueBrainDynamicsExecutionFeedbackLane {
        state: BlueBrainDynamicsExecutionFeedbackState::InsufficientDynamicsFeedbackBasis,
        lane: "blue_brain_dynamics_feedback_basis_insufficient",
        canonical_guard: "insufficient feedback basis remains explicit and cannot trigger direct action/retry/re-execution",
    },
    BlueBrainDynamicsExecutionFeedbackLane {
        state: BlueBrainDynamicsExecutionFeedbackState::BlockedDynamicsFeedbackBasis,
        lane: "blue_brain_dynamics_feedback_basis_blocked",
        canonical_guard: "blocked basis stays bounded and cannot override safety or execution-integrity boundaries",
    },
    BlueBrainDynamicsExecutionFeedbackLane {
        state: BlueBrainDynamicsExecutionFeedbackState::UnavailableDynamicsFeedbackBasis,
        lane: "blue_brain_dynamics_feedback_basis_unavailable",
        canonical_guard: "unavailable basis stays explicit, does not claim operational readiness, and cannot trigger direct execution authority",
    },
    BlueBrainDynamicsExecutionFeedbackLane {
        state: BlueBrainDynamicsExecutionFeedbackState::DiagnosticOnlyDynamicsFeedback,
        lane: "blue_brain_dynamics_feedback_diagnostic_only",
        canonical_guard: "diagnostic-only feedback is observable but does not select actions, re-execute, retry, or mutate memory",
    },
    BlueBrainDynamicsExecutionFeedbackLane {
        state: BlueBrainDynamicsExecutionFeedbackState::NonCanonicalInternalOnlyFeedbackPath,
        lane: "blue_brain_dynamics_feedback_non_canonical_internal_only",
        canonical_guard: "non-canonical/internal-only feedback paths cannot become canonical modulation inputs",
    },
];

pub const CANONICAL_BLUE_BRAIN_DYNAMICS_DIAGNOSTICS_MAP: [BlueBrainDynamicsDiagnosticLane; 9] = [
    BlueBrainDynamicsDiagnosticLane {
        diagnostic_class: BlueBrainDynamicsDiagnosticClass::DynamicsDiagnosticObserved,
        lane: "blue_brain_dynamics_diagnostic_observed",
        canonical_guard: "dynamics diagnostics are observation-only and cannot directly mutate runtime/selection/memory/action/compute/policy",
    },
    BlueBrainDynamicsDiagnosticLane {
        diagnostic_class: BlueBrainDynamicsDiagnosticClass::KuramotoModulationDiagnostic,
        lane: "blue_brain_dynamics_kuramoto_modulation_diagnostic",
        canonical_guard: "kuramoto outputs are advisory modulation hints/caveats only and do not execute decisions",
    },
    BlueBrainDynamicsDiagnosticLane {
        diagnostic_class: BlueBrainDynamicsDiagnosticClass::HodgkinHuxleySimulationDiagnostic,
        lane: "blue_brain_dynamics_hh_simulation_diagnostic",
        canonical_guard: "hodgkin-huxley remains simulation/diagnostic-only with bounded parameters and no direct runtime authority",
    },
    BlueBrainDynamicsDiagnosticLane {
        diagnostic_class: BlueBrainDynamicsDiagnosticClass::DynamicsCaveated,
        lane: "blue_brain_dynamics_caveated",
        canonical_guard: "caveated dynamics must remain caveat feedback and never become direct action/memory/compute authority",
    },
    BlueBrainDynamicsDiagnosticLane {
        diagnostic_class: BlueBrainDynamicsDiagnosticClass::DynamicsInsufficient,
        lane: "blue_brain_dynamics_insufficient",
        canonical_guard: "insufficient dynamics signal cannot be escalated to direct modulation or execution",
    },
    BlueBrainDynamicsDiagnosticLane {
        diagnostic_class: BlueBrainDynamicsDiagnosticClass::DynamicsFailed,
        lane: "blue_brain_dynamics_failed",
        canonical_guard: "failed dynamics runs must not fabricate modulation outputs or execution eligibility",
    },
    BlueBrainDynamicsDiagnosticLane {
        diagnostic_class: BlueBrainDynamicsDiagnosticClass::DynamicsUnavailable,
        lane: "blue_brain_dynamics_unavailable",
        canonical_guard: "unavailable dynamics paths remain explicit diagnostics and cannot trigger fallback execution",
    },
    BlueBrainDynamicsDiagnosticLane {
        diagnostic_class: BlueBrainDynamicsDiagnosticClass::DynamicsIgnored,
        lane: "blue_brain_dynamics_ignored",
        canonical_guard: "ignored dynamics output is reported diagnostically and has no transition side-effect",
    },
    BlueBrainDynamicsDiagnosticLane {
        diagnostic_class: BlueBrainDynamicsDiagnosticClass::NonCanonicalInternalOnlyDynamicsDiagnostic,
        lane: "blue_brain_dynamics_non_canonical_internal_only_diagnostic",
        canonical_guard: "internal/expert-only dynamics diagnostics are explicitly non-canonical unless down-mapped to outward references",
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainKuramotoScopeState {
    SimulationOnly,
    DiagnosticOnly,
    SelectionModulating,
    RuntimeCaveatModulating,
    NotImplementedOrNotSuitableNow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainHodgkinHuxleyScopeState {
    SimulationOnly,
    DiagnosticOnly,
    ResearchDeferred,
    NotSuitableForCurrentBlueBrainRuntime,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainHodgkinHuxleySimulationParameters {
    /// Deterministic integration steps (`1..=4096`) for bounded diagnostics.
    pub integration_steps: u16,
    /// Integration step in microseconds (`1..=10_000`) for bounded diagnostics.
    pub dt_micros: u16,
    /// External current in nano-ampere permille (`0..=2000`) for bounded diagnostics.
    pub stimulus_nanoamp_permille: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainHodgkinHuxleyBoundedModelParameters {
    /// Conductance proxy in permille (`1..=2000`).
    pub sodium_conductance_permille: u16,
    /// Conductance proxy in permille (`1..=2000`).
    pub potassium_conductance_permille: u16,
    /// Leak proxy in permille (`1..=1000`).
    pub leak_conductance_permille: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainHodgkinHuxleyDiagnosticInput {
    pub scope: BlueBrainHodgkinHuxleyScopeState,
    pub diagnostic_run_id: String,
    pub context_refs: Vec<String>,
    pub evidence_refs: Vec<String>,
    pub simulation_parameters: BlueBrainHodgkinHuxleySimulationParameters,
    pub model_parameters: BlueBrainHodgkinHuxleyBoundedModelParameters,
}

impl BlueBrainHodgkinHuxleyDiagnosticInput {
    pub fn canonicalize(&mut self) {
        self.context_refs.sort_unstable();
        self.context_refs.dedup();
        self.evidence_refs.sort_unstable();
        self.evidence_refs.dedup();
        self.simulation_parameters.integration_steps =
            self.simulation_parameters.integration_steps.clamp(1, 4096);
        self.simulation_parameters.dt_micros =
            self.simulation_parameters.dt_micros.clamp(1, 10_000);
        self.simulation_parameters.stimulus_nanoamp_permille = self
            .simulation_parameters
            .stimulus_nanoamp_permille
            .min(2000);
        self.model_parameters.sodium_conductance_permille = self
            .model_parameters
            .sodium_conductance_permille
            .clamp(1, 2000);
        self.model_parameters.potassium_conductance_permille = self
            .model_parameters
            .potassium_conductance_permille
            .clamp(1, 2000);
        self.model_parameters.leak_conductance_permille = self
            .model_parameters
            .leak_conductance_permille
            .clamp(1, 1000);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainHodgkinHuxleyDiagnosticClass {
    SimulationDiagnosticSummary,
    SimulationDiagnosticCaveated,
    FailedOrInsufficientDiagnostic,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainHodgkinHuxleyBoundaryGuard {
    pub runtime_state_mutation_allowed: bool,
    pub selection_mutation_allowed: bool,
    pub memory_mutation_allowed: bool,
    pub action_execution_allowed: bool,
    pub tool_invocation_allowed: bool,
    pub compute_invocation_allowed: bool,
    pub safety_override_allowed: bool,
    pub policy_decision_allowed: bool,
    pub actual_action_result_allowed: bool,
    pub memory_commit_allowed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainHodgkinHuxleyDiagnosticResult {
    pub dynamics_diagnostic_class: BlueBrainDynamicsDiagnosticClass,
    pub diagnostic_class: BlueBrainHodgkinHuxleyDiagnosticClass,
    pub caveats: Vec<String>,
    pub trace_ref: Option<String>,
    pub bounded_metadata: Vec<(String, String)>,
    pub runtime_feedback: BlueBrainDynamicsRuntimeFeedbackClass,
    pub selection_feedback: BlueBrainDynamicsSelectionFeedbackClass,
    pub boundary_guard: BlueBrainHodgkinHuxleyBoundaryGuard,
}

pub fn evaluate_blue_brain_hodgkin_huxley_diagnostic(
    mut input: BlueBrainHodgkinHuxleyDiagnosticInput,
) -> BlueBrainHodgkinHuxleyDiagnosticResult {
    input.canonicalize();

    let mut caveats = Vec::new();
    match input.scope {
        BlueBrainHodgkinHuxleyScopeState::DiagnosticOnly
        | BlueBrainHodgkinHuxleyScopeState::SimulationOnly => {}
        BlueBrainHodgkinHuxleyScopeState::ResearchDeferred => {
            caveats.push("hh_scope_research_deferred".to_string());
        }
        BlueBrainHodgkinHuxleyScopeState::NotSuitableForCurrentBlueBrainRuntime => {
            caveats.push("hh_scope_not_suitable_for_runtime".to_string());
        }
        BlueBrainHodgkinHuxleyScopeState::NonCanonicalInternalOnly => {
            caveats.push("hh_scope_non_canonical_internal_only".to_string());
        }
    }

    if input.diagnostic_run_id.trim().is_empty() {
        caveats.push("missing_diagnostic_run_id".to_string());
        return BlueBrainHodgkinHuxleyDiagnosticResult {
            dynamics_diagnostic_class: BlueBrainDynamicsDiagnosticClass::DynamicsFailed,
            diagnostic_class: BlueBrainHodgkinHuxleyDiagnosticClass::FailedOrInsufficientDiagnostic,
            caveats,
            trace_ref: None,
            bounded_metadata: vec![("output_surface".to_string(), "diagnostic_only".to_string())],
            runtime_feedback:
                BlueBrainDynamicsRuntimeFeedbackClass::DynamicsIgnoredForCurrentTransition,
            selection_feedback:
                BlueBrainDynamicsSelectionFeedbackClass::DynamicsIgnoredForCurrentSelection,
            boundary_guard: hh_boundary_guard(),
        };
    }

    let effective_stability_permille = compute_effective_stability_permille(&input);
    let (diagnostic_class, dynamics_diagnostic_class) = if effective_stability_permille >= 700 {
        (
            BlueBrainHodgkinHuxleyDiagnosticClass::SimulationDiagnosticSummary,
            BlueBrainDynamicsDiagnosticClass::HodgkinHuxleySimulationDiagnostic,
        )
    } else if effective_stability_permille >= 450 {
        caveats.push("hh_signal_caveated".to_string());
        (
            BlueBrainHodgkinHuxleyDiagnosticClass::SimulationDiagnosticCaveated,
            BlueBrainDynamicsDiagnosticClass::DynamicsCaveated,
        )
    } else {
        caveats.push("hh_signal_insufficient".to_string());
        (
            BlueBrainHodgkinHuxleyDiagnosticClass::FailedOrInsufficientDiagnostic,
            BlueBrainDynamicsDiagnosticClass::DynamicsInsufficient,
        )
    };
    if matches!(
        input.scope,
        BlueBrainHodgkinHuxleyScopeState::ResearchDeferred
            | BlueBrainHodgkinHuxleyScopeState::NotSuitableForCurrentBlueBrainRuntime
    ) {
        caveats.push("hh_path_unavailable_for_runtime_modulation".to_string());
    }

    let trace_ref = Some(format!("diag:hh:{}", input.diagnostic_run_id.trim()));
    let bounded_metadata = vec![
        (
            "scope".to_string(),
            format!("{:?}", input.scope).to_lowercase(),
        ),
        (
            "effective_stability_permille".to_string(),
            effective_stability_permille.to_string(),
        ),
        (
            "integration_steps".to_string(),
            input.simulation_parameters.integration_steps.to_string(),
        ),
    ];

    BlueBrainHodgkinHuxleyDiagnosticResult {
        dynamics_diagnostic_class: match input.scope {
            BlueBrainHodgkinHuxleyScopeState::NonCanonicalInternalOnly => {
                BlueBrainDynamicsDiagnosticClass::NonCanonicalInternalOnlyDynamicsDiagnostic
            }
            BlueBrainHodgkinHuxleyScopeState::ResearchDeferred
            | BlueBrainHodgkinHuxleyScopeState::NotSuitableForCurrentBlueBrainRuntime => {
                BlueBrainDynamicsDiagnosticClass::DynamicsUnavailable
            }
            _ => dynamics_diagnostic_class,
        },
        diagnostic_class,
        caveats,
        trace_ref,
        bounded_metadata,
        runtime_feedback:
            BlueBrainDynamicsRuntimeFeedbackClass::DynamicsIgnoredForCurrentTransition,
        selection_feedback:
            BlueBrainDynamicsSelectionFeedbackClass::DynamicsIgnoredForCurrentSelection,
        boundary_guard: hh_boundary_guard(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainKuramotoSelectionPosture {
    Selected,
    Deferred,
    Blocked,
    Insufficient,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainKuramotoRuntimePosture {
    Stable,
    Caveated,
    Degraded,
    Blocked,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainKuramotoPhaseNodeInput {
    pub group_ref: String,
    /// Integer ring phase in permille (`0..=999`) to preserve deterministic arithmetic.
    pub phase_permille: u16,
    /// Coupling strength in permille (`0..=1000`).
    pub coupling_permille: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainKuramotoModulationInput {
    pub scope: BlueBrainKuramotoScopeState,
    pub selection_posture: BlueBrainKuramotoSelectionPosture,
    pub runtime_posture: BlueBrainKuramotoRuntimePosture,
    pub selected_context_refs: Vec<String>,
    pub selected_evidence_refs: Vec<String>,
    pub memory_caveats: Vec<String>,
    pub phase_nodes: Vec<BlueBrainKuramotoPhaseNodeInput>,
    pub unsupported_input_refs: Vec<String>,
    pub blocked_input_refs: Vec<String>,
    pub canonical_execution_result_refs: Vec<String>,
    pub failed_execution_result_refs: Vec<String>,
    pub cancelled_execution_result_refs: Vec<String>,
    pub blocked_execution_result_refs: Vec<String>,
    pub insufficient_execution_result_refs: Vec<String>,
    pub unavailable_execution_result_refs: Vec<String>,
    pub diagnostic_only_feedback_refs: Vec<String>,
    pub non_canonical_internal_only_path: bool,
}

impl BlueBrainKuramotoModulationInput {
    pub fn canonicalize(&mut self) {
        self.selected_context_refs.sort_unstable();
        self.selected_context_refs.dedup();
        self.selected_evidence_refs.sort_unstable();
        self.selected_evidence_refs.dedup();
        self.memory_caveats.sort_unstable();
        self.memory_caveats.dedup();
        self.unsupported_input_refs.sort_unstable();
        self.unsupported_input_refs.dedup();
        self.blocked_input_refs.sort_unstable();
        self.blocked_input_refs.dedup();
        self.canonical_execution_result_refs.sort_unstable();
        self.canonical_execution_result_refs.dedup();
        self.failed_execution_result_refs.sort_unstable();
        self.failed_execution_result_refs.dedup();
        self.cancelled_execution_result_refs.sort_unstable();
        self.cancelled_execution_result_refs.dedup();
        self.blocked_execution_result_refs.sort_unstable();
        self.blocked_execution_result_refs.dedup();
        self.insufficient_execution_result_refs.sort_unstable();
        self.insufficient_execution_result_refs.dedup();
        self.unavailable_execution_result_refs.sort_unstable();
        self.unavailable_execution_result_refs.dedup();
        self.diagnostic_only_feedback_refs.sort_unstable();
        self.diagnostic_only_feedback_refs.dedup();
        self.phase_nodes
            .sort_by(|left, right| left.group_ref.cmp(&right.group_ref));
        self.phase_nodes
            .dedup_by(|left, right| left.group_ref == right.group_ref);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainKuramotoSynchronyDiagnostic {
    Synchronized,
    PartiallySynchronized,
    Desynchronized,
    InsufficientInput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainKuramotoSelectionHint {
    KeepCurrentSelection,
    CaveateSelectionWeight,
    IncreaseDeferralConfidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainKuramotoRuntimeCaveatModulation {
    NoAdditionalCaveat,
    AttachDynamicsCaveat,
    EscalateRuntimeCaveat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainKuramotoModulationState {
    AppliedAdvisoryOnly,
    Caveated,
    Insufficient,
    Ignored,
    NoOp,
    Blocked,
    Unavailable,
    NonCanonicalInternalOnlyPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainKuramotoModulationDiagnosticClass {
    ModulationAppliedDiagnostic,
    ModulationCaveatedDiagnostic,
    ModulationInsufficientDiagnostic,
    ModulationIgnoredDiagnostic,
    ModulationNoOpDiagnostic,
    ModulationBlockedDiagnostic,
    ModulationUnavailableDiagnostic,
    NonCanonicalInternalOnlyDynamicsDiagnostic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainKuramotoModulationReason {
    InsufficientInputGroupBasis,
    CaveatedPartialOrWeakBasis,
    NoOpNeutralDeterministicResult,
    IgnoredByRuntimeSelectionContext,
    BlockedByGuardBoundaryCondition,
    UnavailableOperationalPreconditions,
    NonCanonicalInternalOnlyPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainKuramotoInputGroupClass {
    RuntimeStateGroup,
    SelectionAttentionGroup,
    ContextReferenceGroup,
    MemoryCaveatReferenceGroup,
    EvidenceDerivedAdvisoryGroup,
    UnsupportedNonCanonicalInputGroup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainKuramotoInputGroupLane {
    pub group_class: BlueBrainKuramotoInputGroupClass,
    pub group_ref: &'static str,
}

pub const CANONICAL_BLUE_BRAIN_KURAMOTO_INPUT_GROUP_MAP: [BlueBrainKuramotoInputGroupLane; 6] = [
    BlueBrainKuramotoInputGroupLane {
        group_class: BlueBrainKuramotoInputGroupClass::RuntimeStateGroup,
        group_ref: "runtime_state_group",
    },
    BlueBrainKuramotoInputGroupLane {
        group_class: BlueBrainKuramotoInputGroupClass::SelectionAttentionGroup,
        group_ref: "selection_attention_group",
    },
    BlueBrainKuramotoInputGroupLane {
        group_class: BlueBrainKuramotoInputGroupClass::ContextReferenceGroup,
        group_ref: "context_reference_group",
    },
    BlueBrainKuramotoInputGroupLane {
        group_class: BlueBrainKuramotoInputGroupClass::MemoryCaveatReferenceGroup,
        group_ref: "memory_caveat_reference_group",
    },
    BlueBrainKuramotoInputGroupLane {
        group_class: BlueBrainKuramotoInputGroupClass::EvidenceDerivedAdvisoryGroup,
        group_ref: "evidence_derived_advisory_group",
    },
    BlueBrainKuramotoInputGroupLane {
        group_class: BlueBrainKuramotoInputGroupClass::UnsupportedNonCanonicalInputGroup,
        group_ref: "unsupported_non_canonical_input_group",
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainKuramotoInputBasisClass {
    ValidInputBasis,
    CaveatedInputBasis,
    InsufficientInputBasis,
    UnsupportedInputBasis,
    BlockedInputBasis,
    NoOpNeutralInputBasis,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainKuramotoBoundaryGuard {
    pub runtime_state_mutation_allowed: bool,
    pub selection_mutation_allowed: bool,
    pub action_execution_allowed: bool,
    pub tool_invocation_allowed: bool,
    pub actual_action_result_allowed: bool,
    pub memory_persistence_allowed: bool,
    pub memory_commit_allowed: bool,
    pub compute_invocation_allowed: bool,
    pub safety_override_allowed: bool,
    pub policy_decision_allowed: bool,
    pub direct_reexecute_allowed: bool,
    pub direct_retry_orchestration_allowed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainDynamicsRuntimeFeedbackClass {
    RuntimeModulationObserved,
    DynamicsCaveatAttached,
    DynamicsInsufficientForModulation,
    DynamicsIgnoredForCurrentTransition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainDynamicsSelectionFeedbackClass {
    SelectionModulationObserved,
    DynamicsCaveatAttached,
    DynamicsInsufficientForModulation,
    DynamicsIgnoredForCurrentSelection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainDynamicsAdvisoryCouplingState {
    RuntimeAdvisoryCoupling,
    SelectionAdvisoryCoupling,
    CaveatedAdvisoryCoupling,
    InsufficientAdvisoryCoupling,
    BlockedAdvisoryCoupling,
    IgnoredAdvisoryCoupling,
    NonCanonicalInternalOnlyCouplingPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainDynamicsAdvisoryCouplingLane {
    pub coupling_state: BlueBrainDynamicsAdvisoryCouplingState,
    pub lane: &'static str,
    pub canonical_guard: &'static str,
}

pub const CANONICAL_BLUE_BRAIN_DYNAMICS_ADVISORY_COUPLING_MAP:
    [BlueBrainDynamicsAdvisoryCouplingLane; 7] = [
    BlueBrainDynamicsAdvisoryCouplingLane {
        coupling_state: BlueBrainDynamicsAdvisoryCouplingState::RuntimeAdvisoryCoupling,
        lane: "blue_brain_dynamics_runtime_advisory_coupling",
        canonical_guard: "runtime receives advisory dynamics caveat/modulation hints only and never direct execution authority",
    },
    BlueBrainDynamicsAdvisoryCouplingLane {
        coupling_state: BlueBrainDynamicsAdvisoryCouplingState::SelectionAdvisoryCoupling,
        lane: "blue_brain_dynamics_selection_advisory_coupling",
        canonical_guard: "selection observes advisory dynamics hints only and never directly selects actions from dynamics",
    },
    BlueBrainDynamicsAdvisoryCouplingLane {
        coupling_state: BlueBrainDynamicsAdvisoryCouplingState::CaveatedAdvisoryCoupling,
        lane: "blue_brain_dynamics_caveated_advisory_coupling",
        canonical_guard: "caveated coupling remains bounded advisory feedback and cannot escalate to retry/re-execution/memory mutation",
    },
    BlueBrainDynamicsAdvisoryCouplingLane {
        coupling_state: BlueBrainDynamicsAdvisoryCouplingState::InsufficientAdvisoryCoupling,
        lane: "blue_brain_dynamics_insufficient_advisory_coupling",
        canonical_guard: "insufficient coupling basis stays explicit and cannot create proposals, retries, or compute invocation",
    },
    BlueBrainDynamicsAdvisoryCouplingLane {
        coupling_state: BlueBrainDynamicsAdvisoryCouplingState::BlockedAdvisoryCoupling,
        lane: "blue_brain_dynamics_blocked_advisory_coupling",
        canonical_guard: "blocked coupling cannot override safety/eligibility boundaries or become execution requests",
    },
    BlueBrainDynamicsAdvisoryCouplingLane {
        coupling_state: BlueBrainDynamicsAdvisoryCouplingState::IgnoredAdvisoryCoupling,
        lane: "blue_brain_dynamics_ignored_advisory_coupling",
        canonical_guard: "ignored coupling is observable diagnostics-only and has no execution/retry/memory side-effect",
    },
    BlueBrainDynamicsAdvisoryCouplingLane {
        coupling_state: BlueBrainDynamicsAdvisoryCouplingState::NonCanonicalInternalOnlyCouplingPath,
        lane: "blue_brain_dynamics_non_canonical_internal_only_coupling_path",
        canonical_guard: "non-canonical/internal-only coupling paths remain excluded from canonical runtime/selection advisory lanes",
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainRuntimeSelectionContractSignal {
    RuntimeToSelectionAdvisorySignal,
    RuntimeToSelectionDeferredSignal,
    RuntimeToSelectionBlockedSignal,
    SelectionToRuntimeAdvisoryState,
    SelectionToRuntimeDeferredState,
    SelectionToRuntimeBlockedState,
    CaveatedContractSignal,
    InsufficientContractBasis,
    NonCanonicalInternalOnlyContractPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainRuntimeSelectionContractDiagnosticClass {
    RuntimeToSelectionDiagnostic,
    SelectionToRuntimeDiagnostic,
    DeferredContractDiagnostic,
    BlockedContractDiagnostic,
    CaveatedContractDiagnostic,
    InsufficientContractDiagnostic,
    AdvisoryOnlyContractDiagnostic,
    NonCanonicalInternalOnlyContractDiagnostic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainRuntimeSelectionContractReason {
    PriorityAdvisoryHintOnlyNoDirectSelectionAuthority,
    DeferredDueToBoundedPrioritySelectionState,
    BlockedDueToContractBoundaryOrReferenceWeakness,
    CaveatedDueToWeakOrPartialReferenceDynamicsExecutionBasis,
    InsufficientDueToMissingBoundedContractBasis,
    AdvisoryOnlyNoDirectActionAuthority,
    NonCanonicalInternalOnlyPathExcluded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainRuntimeSelectionContractDiagnosticLane {
    pub diagnostic_class: BlueBrainRuntimeSelectionContractDiagnosticClass,
    pub lane: &'static str,
    pub canonical_guard: &'static str,
}

pub const CANONICAL_BLUE_BRAIN_RUNTIME_SELECTION_CONTRACT_DIAGNOSTICS_MAP:
    [BlueBrainRuntimeSelectionContractDiagnosticLane; 8] = [
    BlueBrainRuntimeSelectionContractDiagnosticLane {
        diagnostic_class:
            BlueBrainRuntimeSelectionContractDiagnosticClass::RuntimeToSelectionDiagnostic,
        lane: "runtime_to_selection_contract_diagnostic",
        canonical_guard: "runtime-to-selection diagnostics stay directionally explicit and remain bounded contract feedback only",
    },
    BlueBrainRuntimeSelectionContractDiagnosticLane {
        diagnostic_class:
            BlueBrainRuntimeSelectionContractDiagnosticClass::SelectionToRuntimeDiagnostic,
        lane: "selection_to_runtime_contract_diagnostic",
        canonical_guard: "selection-to-runtime diagnostics stay directionally explicit and remain bounded contract feedback only",
    },
    BlueBrainRuntimeSelectionContractDiagnosticLane {
        diagnostic_class:
            BlueBrainRuntimeSelectionContractDiagnosticClass::DeferredContractDiagnostic,
        lane: "deferred_contract_diagnostic",
        canonical_guard: "deferred diagnostics remain distinct from blocked and do not imply retry orchestration or failed execution",
    },
    BlueBrainRuntimeSelectionContractDiagnosticLane {
        diagnostic_class:
            BlueBrainRuntimeSelectionContractDiagnosticClass::BlockedContractDiagnostic,
        lane: "blocked_contract_diagnostic",
        canonical_guard: "blocked diagnostics remain explicit contract-boundary feedback and are not execution-failure semantics",
    },
    BlueBrainRuntimeSelectionContractDiagnosticLane {
        diagnostic_class:
            BlueBrainRuntimeSelectionContractDiagnosticClass::CaveatedContractDiagnostic,
        lane: "caveated_contract_diagnostic",
        canonical_guard: "caveated diagnostics remain partial-basis transport and cannot become strong execution authority",
    },
    BlueBrainRuntimeSelectionContractDiagnosticLane {
        diagnostic_class:
            BlueBrainRuntimeSelectionContractDiagnosticClass::InsufficientContractDiagnostic,
        lane: "insufficient_contract_diagnostic",
        canonical_guard: "insufficient diagnostics remain missing-basis feedback and are never promoted to blocked control",
    },
    BlueBrainRuntimeSelectionContractDiagnosticLane {
        diagnostic_class:
            BlueBrainRuntimeSelectionContractDiagnosticClass::AdvisoryOnlyContractDiagnostic,
        lane: "advisory_only_contract_diagnostic",
        canonical_guard: "advisory-only diagnostics cannot create action authority, retry authority, compute invocation, or memory persistence",
    },
    BlueBrainRuntimeSelectionContractDiagnosticLane {
        diagnostic_class:
            BlueBrainRuntimeSelectionContractDiagnosticClass::NonCanonicalInternalOnlyContractDiagnostic,
        lane: "non_canonical_internal_only_contract_diagnostic",
        canonical_guard: "non-canonical/internal-only diagnostics are explicitly excluded from canonical runtime/selection contract exchange",
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainRuntimeSelectionContractLane {
    pub signal: BlueBrainRuntimeSelectionContractSignal,
    pub lane: &'static str,
    pub canonical_guard: &'static str,
}

pub const CANONICAL_BLUE_BRAIN_RUNTIME_SELECTION_CONTRACT_MAP:
    [BlueBrainRuntimeSelectionContractLane; 9] = [
    BlueBrainRuntimeSelectionContractLane {
        signal: BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionAdvisorySignal,
        lane: "runtime_to_selection_advisory_signal",
        canonical_guard: "runtime-to-selection contract signal is advisory-only and cannot directly trigger action/proposal execution",
    },
    BlueBrainRuntimeSelectionContractLane {
        signal: BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionDeferredSignal,
        lane: "runtime_to_selection_deferred_signal",
        canonical_guard: "runtime-to-selection deferred signal is bounded deferral feedback and never implies blocked/failed execution authority",
    },
    BlueBrainRuntimeSelectionContractLane {
        signal: BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionBlockedSignal,
        lane: "runtime_to_selection_blocked_signal",
        canonical_guard: "runtime-to-selection blocked signal is explicit contract boundary feedback and remains separate from failed execution semantics",
    },
    BlueBrainRuntimeSelectionContractLane {
        signal: BlueBrainRuntimeSelectionContractSignal::SelectionToRuntimeAdvisoryState,
        lane: "selection_to_runtime_advisory_state",
        canonical_guard: "selection-to-runtime advisory state cannot directly steer compute execution or planner authority",
    },
    BlueBrainRuntimeSelectionContractLane {
        signal: BlueBrainRuntimeSelectionContractSignal::SelectionToRuntimeDeferredState,
        lane: "selection_to_runtime_deferred_state",
        canonical_guard: "selection-to-runtime deferred state remains distinct from blocked and does not imply retry orchestration",
    },
    BlueBrainRuntimeSelectionContractLane {
        signal: BlueBrainRuntimeSelectionContractSignal::SelectionToRuntimeBlockedState,
        lane: "selection_to_runtime_blocked_state",
        canonical_guard: "selection-to-runtime blocked state remains explicit contract/safety/reference boundary feedback and is not low-priority deferral",
    },
    BlueBrainRuntimeSelectionContractLane {
        signal: BlueBrainRuntimeSelectionContractSignal::CaveatedContractSignal,
        lane: "caveated_contract_signal",
        canonical_guard: "caveated contract state remains caveat transport and cannot become strong execution authority",
    },
    BlueBrainRuntimeSelectionContractLane {
        signal: BlueBrainRuntimeSelectionContractSignal::InsufficientContractBasis,
        lane: "insufficient_contract_basis",
        canonical_guard: "insufficient contract basis remains explicit and is never promoted to blocked/execution control",
    },
    BlueBrainRuntimeSelectionContractLane {
        signal: BlueBrainRuntimeSelectionContractSignal::NonCanonicalInternalOnlyContractPath,
        lane: "non_canonical_internal_only_contract_path",
        canonical_guard: "non-canonical/internal-only contract paths remain excluded from canonical runtime/selection exchange",
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainPriorityDeferredBlockedBoundaryState {
    PriorityAdvisoryHint,
    DeferredContractState,
    BlockedContractState,
    CaveatedPriorityDeferredBlockedSignal,
    InsufficientContractBasis,
    NonCanonicalInternalOnlyCouplingPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainPriorityDeferredBlockedBoundaryLane {
    pub boundary_state: BlueBrainPriorityDeferredBlockedBoundaryState,
    pub lane: &'static str,
    pub canonical_guard: &'static str,
}

pub const CANONICAL_BLUE_BRAIN_PRIORITY_DEFERRED_BLOCKED_BOUNDARY_MAP:
    [BlueBrainPriorityDeferredBlockedBoundaryLane; 6] = [
    BlueBrainPriorityDeferredBlockedBoundaryLane {
        boundary_state: BlueBrainPriorityDeferredBlockedBoundaryState::PriorityAdvisoryHint,
        lane: "priority_advisory_hint",
        canonical_guard: "priority remains advisory-only and cannot directly select proposals/actions or override deferred/blocked boundaries",
    },
    BlueBrainPriorityDeferredBlockedBoundaryLane {
        boundary_state: BlueBrainPriorityDeferredBlockedBoundaryState::DeferredContractState,
        lane: "deferred_contract_state",
        canonical_guard: "deferred remains bounded postponement feedback and is not failed execution or blocked contract control",
    },
    BlueBrainPriorityDeferredBlockedBoundaryLane {
        boundary_state: BlueBrainPriorityDeferredBlockedBoundaryState::BlockedContractState,
        lane: "blocked_contract_state",
        canonical_guard: "blocked remains explicit contract/safety/reference boundary state and is not low priority",
    },
    BlueBrainPriorityDeferredBlockedBoundaryLane {
        boundary_state:
            BlueBrainPriorityDeferredBlockedBoundaryState::CaveatedPriorityDeferredBlockedSignal,
        lane: "caveated_priority_deferred_blocked_signal",
        canonical_guard: "caveated priority/deferred/blocked signal remains bounded partial-basis transport without execution authority",
    },
    BlueBrainPriorityDeferredBlockedBoundaryLane {
        boundary_state: BlueBrainPriorityDeferredBlockedBoundaryState::InsufficientContractBasis,
        lane: "insufficient_contract_basis_boundary_state",
        canonical_guard: "insufficient basis remains explicit missing-basis feedback and cannot be promoted to deferred/blocked control",
    },
    BlueBrainPriorityDeferredBlockedBoundaryLane {
        boundary_state:
            BlueBrainPriorityDeferredBlockedBoundaryState::NonCanonicalInternalOnlyCouplingPath,
        lane: "non_canonical_internal_only_coupling_path_boundary_state",
        canonical_guard: "non-canonical/internal-only coupling paths stay excluded from canonical runtime/selection boundary exchange",
    },
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainKuramotoModulationResult {
    pub dynamics_diagnostic_class: BlueBrainDynamicsDiagnosticClass,
    pub diagnostic_class: BlueBrainKuramotoModulationDiagnosticClass,
    pub modulation_state: BlueBrainKuramotoModulationState,
    pub modulation_reason: Option<BlueBrainKuramotoModulationReason>,
    pub diagnostic: BlueBrainKuramotoSynchronyDiagnostic,
    pub coherence_permille: u16,
    pub selection_hint: Option<BlueBrainKuramotoSelectionHint>,
    pub runtime_modulation: Option<BlueBrainKuramotoRuntimeCaveatModulation>,
    pub runtime_feedback: BlueBrainDynamicsRuntimeFeedbackClass,
    pub selection_feedback: BlueBrainDynamicsSelectionFeedbackClass,
    pub runtime_coupling_state: BlueBrainDynamicsAdvisoryCouplingState,
    pub selection_coupling_state: BlueBrainDynamicsAdvisoryCouplingState,
    pub runtime_to_selection_contract_signal: BlueBrainRuntimeSelectionContractSignal,
    pub selection_to_runtime_contract_signal: BlueBrainRuntimeSelectionContractSignal,
    pub runtime_to_selection_contract_diagnostic: BlueBrainRuntimeSelectionContractDiagnosticClass,
    pub selection_to_runtime_contract_diagnostic: BlueBrainRuntimeSelectionContractDiagnosticClass,
    pub runtime_to_selection_contract_reason: BlueBrainRuntimeSelectionContractReason,
    pub selection_to_runtime_contract_reason: BlueBrainRuntimeSelectionContractReason,
    pub runtime_boundary_state: BlueBrainPriorityDeferredBlockedBoundaryState,
    pub selection_boundary_state: BlueBrainPriorityDeferredBlockedBoundaryState,
    pub input_basis: BlueBrainKuramotoInputBasisClass,
    pub execution_feedback_state: BlueBrainDynamicsExecutionFeedbackState,
    pub caveats: Vec<String>,
    pub boundary_guard: BlueBrainKuramotoBoundaryGuard,
}

pub fn evaluate_blue_brain_kuramoto_modulation(
    mut input: BlueBrainKuramotoModulationInput,
) -> BlueBrainKuramotoModulationResult {
    input.canonicalize();

    let mut caveats = Vec::new();
    if matches!(
        input.scope,
        BlueBrainKuramotoScopeState::SimulationOnly
            | BlueBrainKuramotoScopeState::NotImplementedOrNotSuitableNow
    ) {
        caveats.push("scope_non_modulating".to_string());
    }
    if matches!(
        input.scope,
        BlueBrainKuramotoScopeState::NotImplementedOrNotSuitableNow
    ) {
        caveats.push("dynamics_path_unavailable".to_string());
    }
    if input.non_canonical_internal_only_path {
        caveats.push("non_canonical_internal_only_modulation_path".to_string());
    }
    if !input.blocked_input_refs.is_empty() {
        caveats.push("blocked_input_group_present".to_string());
    }
    if !input.unsupported_input_refs.is_empty() {
        caveats.push("unsupported_input_group_present".to_string());
    }
    let execution_feedback_state = classify_execution_feedback_state(&input);

    let mut canonical_phase_nodes = Vec::new();
    for node in &input.phase_nodes {
        match classify_kuramoto_input_group(node.group_ref.as_str()) {
            BlueBrainKuramotoInputGroupClass::RuntimeStateGroup
            | BlueBrainKuramotoInputGroupClass::SelectionAttentionGroup
            | BlueBrainKuramotoInputGroupClass::ContextReferenceGroup
            | BlueBrainKuramotoInputGroupClass::MemoryCaveatReferenceGroup
            | BlueBrainKuramotoInputGroupClass::EvidenceDerivedAdvisoryGroup => {
                canonical_phase_nodes.push(node.clone());
            }
            BlueBrainKuramotoInputGroupClass::UnsupportedNonCanonicalInputGroup => {
                caveats.push("unsupported_phase_node_group_ref".to_string());
            }
        }
    }
    canonical_phase_nodes.sort_by(|left, right| left.group_ref.cmp(&right.group_ref));
    canonical_phase_nodes.dedup_by(|left, right| left.group_ref == right.group_ref);

    if canonical_phase_nodes.len() < 2 {
        caveats.push("insufficient_dynamics_input".to_string());
        return BlueBrainKuramotoModulationResult {
            dynamics_diagnostic_class: BlueBrainDynamicsDiagnosticClass::DynamicsInsufficient,
            diagnostic_class:
                BlueBrainKuramotoModulationDiagnosticClass::ModulationInsufficientDiagnostic,
            modulation_state: BlueBrainKuramotoModulationState::Insufficient,
            modulation_reason: Some(BlueBrainKuramotoModulationReason::InsufficientInputGroupBasis),
            diagnostic: BlueBrainKuramotoSynchronyDiagnostic::InsufficientInput,
            coherence_permille: 0,
            selection_hint: None,
            runtime_modulation: None,
            runtime_feedback:
                BlueBrainDynamicsRuntimeFeedbackClass::DynamicsInsufficientForModulation,
            selection_feedback:
                BlueBrainDynamicsSelectionFeedbackClass::DynamicsInsufficientForModulation,
            runtime_coupling_state:
                BlueBrainDynamicsAdvisoryCouplingState::InsufficientAdvisoryCoupling,
            selection_coupling_state:
                BlueBrainDynamicsAdvisoryCouplingState::InsufficientAdvisoryCoupling,
            runtime_to_selection_contract_signal:
                BlueBrainRuntimeSelectionContractSignal::InsufficientContractBasis,
            selection_to_runtime_contract_signal:
                BlueBrainRuntimeSelectionContractSignal::InsufficientContractBasis,
            runtime_to_selection_contract_diagnostic:
                BlueBrainRuntimeSelectionContractDiagnosticClass::InsufficientContractDiagnostic,
            selection_to_runtime_contract_diagnostic:
                BlueBrainRuntimeSelectionContractDiagnosticClass::InsufficientContractDiagnostic,
            runtime_to_selection_contract_reason:
                BlueBrainRuntimeSelectionContractReason::InsufficientDueToMissingBoundedContractBasis,
            selection_to_runtime_contract_reason:
                BlueBrainRuntimeSelectionContractReason::InsufficientDueToMissingBoundedContractBasis,
            runtime_boundary_state:
                BlueBrainPriorityDeferredBlockedBoundaryState::InsufficientContractBasis,
            selection_boundary_state:
                BlueBrainPriorityDeferredBlockedBoundaryState::InsufficientContractBasis,
            input_basis: BlueBrainKuramotoInputBasisClass::InsufficientInputBasis,
            execution_feedback_state,
            caveats,
            boundary_guard: boundary_guard(),
        };
    }

    let coherence_permille = coherence_from_phase_nodes(&canonical_phase_nodes);
    let diagnostic = if coherence_permille >= 800 {
        BlueBrainKuramotoSynchronyDiagnostic::Synchronized
    } else if coherence_permille >= 500 {
        BlueBrainKuramotoSynchronyDiagnostic::PartiallySynchronized
    } else {
        BlueBrainKuramotoSynchronyDiagnostic::Desynchronized
    };

    let selection_hint = if matches!(
        input.scope,
        BlueBrainKuramotoScopeState::SelectionModulating
            | BlueBrainKuramotoScopeState::DiagnosticOnly
    ) {
        Some(selection_hint_from_signal(
            diagnostic,
            input.selection_posture,
            input.memory_caveats.is_empty(),
        ))
    } else {
        None
    };

    let runtime_modulation = if matches!(
        input.scope,
        BlueBrainKuramotoScopeState::RuntimeCaveatModulating
            | BlueBrainKuramotoScopeState::DiagnosticOnly
    ) {
        Some(runtime_modulation_from_signal(
            diagnostic,
            input.runtime_posture,
        ))
    } else {
        None
    };

    if matches!(
        diagnostic,
        BlueBrainKuramotoSynchronyDiagnostic::Desynchronized
    ) {
        caveats.push("dynamics_desynchrony".to_string());
    }

    if matches!(
        input.runtime_posture,
        BlueBrainKuramotoRuntimePosture::Caveated
            | BlueBrainKuramotoRuntimePosture::Degraded
            | BlueBrainKuramotoRuntimePosture::Blocked
    ) {
        caveats.push("runtime_caveat_posture_present".to_string());
    }

    let modulation_state = if input.non_canonical_internal_only_path {
        BlueBrainKuramotoModulationState::NonCanonicalInternalOnlyPath
    } else if !input.blocked_input_refs.is_empty() {
        BlueBrainKuramotoModulationState::Blocked
    } else if !input.unsupported_input_refs.is_empty() {
        BlueBrainKuramotoModulationState::Caveated
    } else if matches!(
        input.scope,
        BlueBrainKuramotoScopeState::NotImplementedOrNotSuitableNow
    ) {
        BlueBrainKuramotoModulationState::Unavailable
    } else if matches!(input.scope, BlueBrainKuramotoScopeState::SimulationOnly) {
        BlueBrainKuramotoModulationState::Ignored
    } else if matches!(
        (input.selection_posture, input.runtime_posture),
        (BlueBrainKuramotoSelectionPosture::Blocked, _)
            | (_, BlueBrainKuramotoRuntimePosture::Blocked)
    ) {
        BlueBrainKuramotoModulationState::Blocked
    } else if !caveats.is_empty() {
        BlueBrainKuramotoModulationState::Caveated
    } else if selection_hint == Some(BlueBrainKuramotoSelectionHint::KeepCurrentSelection)
        && runtime_modulation == Some(BlueBrainKuramotoRuntimeCaveatModulation::NoAdditionalCaveat)
    {
        BlueBrainKuramotoModulationState::NoOp
    } else {
        BlueBrainKuramotoModulationState::AppliedAdvisoryOnly
    };
    let input_basis = if matches!(
        modulation_state,
        BlueBrainKuramotoModulationState::NonCanonicalInternalOnlyPath
            | BlueBrainKuramotoModulationState::Blocked
    ) {
        BlueBrainKuramotoInputBasisClass::BlockedInputBasis
    } else if !input.unsupported_input_refs.is_empty() {
        BlueBrainKuramotoInputBasisClass::UnsupportedInputBasis
    } else if matches!(
        modulation_state,
        BlueBrainKuramotoModulationState::Insufficient
    ) {
        BlueBrainKuramotoInputBasisClass::InsufficientInputBasis
    } else if matches!(modulation_state, BlueBrainKuramotoModulationState::NoOp) {
        BlueBrainKuramotoInputBasisClass::NoOpNeutralInputBasis
    } else if !caveats.is_empty() {
        BlueBrainKuramotoInputBasisClass::CaveatedInputBasis
    } else {
        BlueBrainKuramotoInputBasisClass::ValidInputBasis
    };

    let (diagnostic_class, modulation_reason) = match modulation_state {
        BlueBrainKuramotoModulationState::AppliedAdvisoryOnly => (
            BlueBrainKuramotoModulationDiagnosticClass::ModulationAppliedDiagnostic,
            None,
        ),
        BlueBrainKuramotoModulationState::Caveated => (
            BlueBrainKuramotoModulationDiagnosticClass::ModulationCaveatedDiagnostic,
            Some(BlueBrainKuramotoModulationReason::CaveatedPartialOrWeakBasis),
        ),
        BlueBrainKuramotoModulationState::Insufficient => (
            BlueBrainKuramotoModulationDiagnosticClass::ModulationInsufficientDiagnostic,
            Some(BlueBrainKuramotoModulationReason::InsufficientInputGroupBasis),
        ),
        BlueBrainKuramotoModulationState::Ignored => (
            BlueBrainKuramotoModulationDiagnosticClass::ModulationIgnoredDiagnostic,
            Some(BlueBrainKuramotoModulationReason::IgnoredByRuntimeSelectionContext),
        ),
        BlueBrainKuramotoModulationState::NoOp => (
            BlueBrainKuramotoModulationDiagnosticClass::ModulationNoOpDiagnostic,
            Some(BlueBrainKuramotoModulationReason::NoOpNeutralDeterministicResult),
        ),
        BlueBrainKuramotoModulationState::Blocked => (
            BlueBrainKuramotoModulationDiagnosticClass::ModulationBlockedDiagnostic,
            Some(BlueBrainKuramotoModulationReason::BlockedByGuardBoundaryCondition),
        ),
        BlueBrainKuramotoModulationState::Unavailable => (
            BlueBrainKuramotoModulationDiagnosticClass::ModulationUnavailableDiagnostic,
            Some(BlueBrainKuramotoModulationReason::UnavailableOperationalPreconditions),
        ),
        BlueBrainKuramotoModulationState::NonCanonicalInternalOnlyPath => (
            BlueBrainKuramotoModulationDiagnosticClass::NonCanonicalInternalOnlyDynamicsDiagnostic,
            Some(BlueBrainKuramotoModulationReason::NonCanonicalInternalOnlyPath),
        ),
    };
    append_execution_feedback_caveats(&mut caveats, execution_feedback_state);
    append_kuramoto_guard_caveats(&mut caveats, modulation_state);
    let selection_feedback =
        selection_feedback_from_modulation_state(modulation_state, selection_hint);
    let runtime_feedback =
        runtime_feedback_from_modulation_state(modulation_state, runtime_modulation);
    let runtime_coupling_state =
        runtime_coupling_state_from_modulation_state(modulation_state, runtime_modulation);
    let selection_coupling_state =
        selection_coupling_state_from_modulation_state(modulation_state, selection_hint);
    let runtime_to_selection_contract_signal =
        runtime_to_selection_contract_signal_from_state(modulation_state, selection_hint);
    let selection_to_runtime_contract_signal =
        selection_to_runtime_contract_signal_from_state(modulation_state, runtime_modulation);
    let runtime_to_selection_contract_diagnostic =
        runtime_to_selection_contract_diagnostic_from_signal(runtime_to_selection_contract_signal);
    let selection_to_runtime_contract_diagnostic =
        selection_to_runtime_contract_diagnostic_from_signal(selection_to_runtime_contract_signal);
    let runtime_to_selection_contract_reason = runtime_to_selection_contract_reason_from_signal(
        runtime_to_selection_contract_signal,
        execution_feedback_state,
    );
    let selection_to_runtime_contract_reason = selection_to_runtime_contract_reason_from_signal(
        selection_to_runtime_contract_signal,
        execution_feedback_state,
    );
    let runtime_boundary_state = boundary_state_from_contract_signal_and_reason(
        runtime_to_selection_contract_signal,
        runtime_to_selection_contract_reason,
    );
    let selection_boundary_state = boundary_state_from_contract_signal_and_reason(
        selection_to_runtime_contract_signal,
        selection_to_runtime_contract_reason,
    );

    BlueBrainKuramotoModulationResult {
        dynamics_diagnostic_class: if input.non_canonical_internal_only_path {
            BlueBrainDynamicsDiagnosticClass::NonCanonicalInternalOnlyDynamicsDiagnostic
        } else {
            match input.scope {
                BlueBrainKuramotoScopeState::NotImplementedOrNotSuitableNow => {
                    BlueBrainDynamicsDiagnosticClass::DynamicsUnavailable
                }
                BlueBrainKuramotoScopeState::SimulationOnly => {
                    BlueBrainDynamicsDiagnosticClass::DynamicsIgnored
                }
                _ if matches!(
                    diagnostic,
                    BlueBrainKuramotoSynchronyDiagnostic::Desynchronized
                ) =>
                {
                    BlueBrainDynamicsDiagnosticClass::DynamicsCaveated
                }
                _ => BlueBrainDynamicsDiagnosticClass::KuramotoModulationDiagnostic,
            }
        },
        diagnostic_class,
        modulation_state,
        modulation_reason,
        diagnostic,
        coherence_permille,
        selection_hint,
        runtime_modulation,
        runtime_feedback,
        selection_feedback,
        runtime_coupling_state,
        selection_coupling_state,
        runtime_to_selection_contract_signal,
        selection_to_runtime_contract_signal,
        runtime_to_selection_contract_diagnostic,
        selection_to_runtime_contract_diagnostic,
        runtime_to_selection_contract_reason,
        selection_to_runtime_contract_reason,
        runtime_boundary_state,
        selection_boundary_state,
        input_basis,
        execution_feedback_state,
        caveats,
        boundary_guard: boundary_guard(),
    }
}

pub fn kuramoto_modulation_state_token(state: BlueBrainKuramotoModulationState) -> &'static str {
    match state {
        BlueBrainKuramotoModulationState::AppliedAdvisoryOnly => "applied_advisory_only",
        BlueBrainKuramotoModulationState::Caveated => "caveated",
        BlueBrainKuramotoModulationState::Insufficient => "insufficient",
        BlueBrainKuramotoModulationState::Ignored => "ignored",
        BlueBrainKuramotoModulationState::NoOp => "no_op",
        BlueBrainKuramotoModulationState::Blocked => "blocked",
        BlueBrainKuramotoModulationState::Unavailable => "unavailable",
        BlueBrainKuramotoModulationState::NonCanonicalInternalOnlyPath => {
            "non_canonical_internal_only_path"
        }
    }
}

pub fn kuramoto_modulation_diagnostic_class_token(
    class: BlueBrainKuramotoModulationDiagnosticClass,
) -> &'static str {
    match class {
        BlueBrainKuramotoModulationDiagnosticClass::ModulationAppliedDiagnostic => {
            "modulation_applied_diagnostic"
        }
        BlueBrainKuramotoModulationDiagnosticClass::ModulationCaveatedDiagnostic => {
            "modulation_caveated_diagnostic"
        }
        BlueBrainKuramotoModulationDiagnosticClass::ModulationInsufficientDiagnostic => {
            "modulation_insufficient_diagnostic"
        }
        BlueBrainKuramotoModulationDiagnosticClass::ModulationIgnoredDiagnostic => {
            "modulation_ignored_diagnostic"
        }
        BlueBrainKuramotoModulationDiagnosticClass::ModulationNoOpDiagnostic => {
            "modulation_no_op_diagnostic"
        }
        BlueBrainKuramotoModulationDiagnosticClass::ModulationBlockedDiagnostic => {
            "modulation_blocked_diagnostic"
        }
        BlueBrainKuramotoModulationDiagnosticClass::ModulationUnavailableDiagnostic => {
            "modulation_unavailable_diagnostic"
        }
        BlueBrainKuramotoModulationDiagnosticClass::NonCanonicalInternalOnlyDynamicsDiagnostic => {
            "non_canonical_internal_only_dynamics_diagnostic"
        }
    }
}

pub fn kuramoto_modulation_reason_token(
    reason: Option<BlueBrainKuramotoModulationReason>,
) -> &'static str {
    match reason {
        None => "none",
        Some(BlueBrainKuramotoModulationReason::InsufficientInputGroupBasis) => {
            "insufficient_input_group_basis"
        }
        Some(BlueBrainKuramotoModulationReason::CaveatedPartialOrWeakBasis) => {
            "caveated_partial_or_weak_basis"
        }
        Some(BlueBrainKuramotoModulationReason::NoOpNeutralDeterministicResult) => {
            "no_op_neutral_deterministic_result"
        }
        Some(BlueBrainKuramotoModulationReason::IgnoredByRuntimeSelectionContext) => {
            "ignored_by_runtime_selection_context"
        }
        Some(BlueBrainKuramotoModulationReason::BlockedByGuardBoundaryCondition) => {
            "blocked_by_guard_boundary_condition"
        }
        Some(BlueBrainKuramotoModulationReason::UnavailableOperationalPreconditions) => {
            "unavailable_operational_preconditions"
        }
        Some(BlueBrainKuramotoModulationReason::NonCanonicalInternalOnlyPath) => {
            "non_canonical_internal_only_path"
        }
    }
}

pub fn dynamics_execution_feedback_state_token(
    state: BlueBrainDynamicsExecutionFeedbackState,
) -> &'static str {
    match state {
        BlueBrainDynamicsExecutionFeedbackState::ExecutionInformedDynamicsInput => {
            "execution_informed_dynamics_input"
        }
        BlueBrainDynamicsExecutionFeedbackState::ReferenceInformedDynamicsInput => {
            "reference_informed_dynamics_input"
        }
        BlueBrainDynamicsExecutionFeedbackState::FailedExecutionFeedbackBasis
        | BlueBrainDynamicsExecutionFeedbackState::CancelledExecutionFeedbackBasis => {
            "caveated_execution_informed_dynamics_input"
        }
        BlueBrainDynamicsExecutionFeedbackState::InsufficientDynamicsFeedbackBasis => {
            "insufficient_dynamics_feedback_basis"
        }
        BlueBrainDynamicsExecutionFeedbackState::BlockedDynamicsFeedbackBasis => {
            "blocked_dynamics_feedback_basis"
        }
        BlueBrainDynamicsExecutionFeedbackState::UnavailableDynamicsFeedbackBasis => {
            "unavailable_dynamics_feedback_basis"
        }
        BlueBrainDynamicsExecutionFeedbackState::DiagnosticOnlyDynamicsFeedback => {
            "diagnostic_only_dynamics_feedback"
        }
        BlueBrainDynamicsExecutionFeedbackState::NonCanonicalInternalOnlyFeedbackPath => {
            "non_canonical_internal_only_feedback_path"
        }
    }
}

pub fn dynamics_advisory_coupling_state_token(
    state: BlueBrainDynamicsAdvisoryCouplingState,
) -> &'static str {
    match state {
        BlueBrainDynamicsAdvisoryCouplingState::RuntimeAdvisoryCoupling => {
            "runtime_advisory_coupling"
        }
        BlueBrainDynamicsAdvisoryCouplingState::SelectionAdvisoryCoupling => {
            "selection_advisory_coupling"
        }
        BlueBrainDynamicsAdvisoryCouplingState::CaveatedAdvisoryCoupling => {
            "caveated_advisory_coupling"
        }
        BlueBrainDynamicsAdvisoryCouplingState::InsufficientAdvisoryCoupling => {
            "insufficient_advisory_coupling"
        }
        BlueBrainDynamicsAdvisoryCouplingState::BlockedAdvisoryCoupling => {
            "blocked_advisory_coupling"
        }
        BlueBrainDynamicsAdvisoryCouplingState::IgnoredAdvisoryCoupling => {
            "ignored_advisory_coupling"
        }
        BlueBrainDynamicsAdvisoryCouplingState::NonCanonicalInternalOnlyCouplingPath => {
            "non_canonical_internal_only_coupling_path"
        }
    }
}

pub fn runtime_selection_contract_signal_token(
    signal: BlueBrainRuntimeSelectionContractSignal,
) -> &'static str {
    match signal {
        BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionAdvisorySignal => {
            "runtime_to_selection_advisory_signal"
        }
        BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionDeferredSignal => {
            "runtime_to_selection_deferred_signal"
        }
        BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionBlockedSignal => {
            "runtime_to_selection_blocked_signal"
        }
        BlueBrainRuntimeSelectionContractSignal::SelectionToRuntimeAdvisoryState => {
            "selection_to_runtime_advisory_state"
        }
        BlueBrainRuntimeSelectionContractSignal::SelectionToRuntimeDeferredState => {
            "selection_to_runtime_deferred_state"
        }
        BlueBrainRuntimeSelectionContractSignal::SelectionToRuntimeBlockedState => {
            "selection_to_runtime_blocked_state"
        }
        BlueBrainRuntimeSelectionContractSignal::CaveatedContractSignal => {
            "caveated_contract_signal"
        }
        BlueBrainRuntimeSelectionContractSignal::InsufficientContractBasis => {
            "insufficient_contract_basis"
        }
        BlueBrainRuntimeSelectionContractSignal::NonCanonicalInternalOnlyContractPath => {
            "non_canonical_internal_only_contract_path"
        }
    }
}

pub fn runtime_selection_contract_diagnostic_class_token(
    diagnostic_class: BlueBrainRuntimeSelectionContractDiagnosticClass,
) -> &'static str {
    match diagnostic_class {
        BlueBrainRuntimeSelectionContractDiagnosticClass::RuntimeToSelectionDiagnostic => {
            "runtime_to_selection_contract_diagnostic"
        }
        BlueBrainRuntimeSelectionContractDiagnosticClass::SelectionToRuntimeDiagnostic => {
            "selection_to_runtime_contract_diagnostic"
        }
        BlueBrainRuntimeSelectionContractDiagnosticClass::DeferredContractDiagnostic => {
            "deferred_contract_diagnostic"
        }
        BlueBrainRuntimeSelectionContractDiagnosticClass::BlockedContractDiagnostic => {
            "blocked_contract_diagnostic"
        }
        BlueBrainRuntimeSelectionContractDiagnosticClass::CaveatedContractDiagnostic => {
            "caveated_contract_diagnostic"
        }
        BlueBrainRuntimeSelectionContractDiagnosticClass::InsufficientContractDiagnostic => {
            "insufficient_contract_diagnostic"
        }
        BlueBrainRuntimeSelectionContractDiagnosticClass::AdvisoryOnlyContractDiagnostic => {
            "advisory_only_contract_diagnostic"
        }
        BlueBrainRuntimeSelectionContractDiagnosticClass::NonCanonicalInternalOnlyContractDiagnostic => {
            "non_canonical_internal_only_contract_diagnostic"
        }
    }
}

pub fn runtime_selection_contract_reason_token(
    reason: BlueBrainRuntimeSelectionContractReason,
) -> &'static str {
    match reason {
        BlueBrainRuntimeSelectionContractReason::DeferredDueToBoundedPrioritySelectionState => {
            "deferred_due_to_bounded_priority_selection_state"
        }
        BlueBrainRuntimeSelectionContractReason::PriorityAdvisoryHintOnlyNoDirectSelectionAuthority => {
            "priority_advisory_hint_only_no_direct_selection_authority"
        }
        BlueBrainRuntimeSelectionContractReason::BlockedDueToContractBoundaryOrReferenceWeakness => {
            "blocked_due_to_contract_boundary_or_reference_weakness"
        }
        BlueBrainRuntimeSelectionContractReason::CaveatedDueToWeakOrPartialReferenceDynamicsExecutionBasis => {
            "caveated_due_to_weak_or_partial_reference_dynamics_execution_basis"
        }
        BlueBrainRuntimeSelectionContractReason::InsufficientDueToMissingBoundedContractBasis => {
            "insufficient_due_to_missing_bounded_contract_basis"
        }
        BlueBrainRuntimeSelectionContractReason::AdvisoryOnlyNoDirectActionAuthority => {
            "advisory_only_no_direct_action_authority"
        }
        BlueBrainRuntimeSelectionContractReason::NonCanonicalInternalOnlyPathExcluded => {
            "non_canonical_internal_only_path_excluded"
        }
    }
}

pub fn priority_deferred_blocked_boundary_state_token(
    state: BlueBrainPriorityDeferredBlockedBoundaryState,
) -> &'static str {
    match state {
        BlueBrainPriorityDeferredBlockedBoundaryState::PriorityAdvisoryHint => {
            "priority_advisory_hint"
        }
        BlueBrainPriorityDeferredBlockedBoundaryState::DeferredContractState => {
            "deferred_contract_state"
        }
        BlueBrainPriorityDeferredBlockedBoundaryState::BlockedContractState => {
            "blocked_contract_state"
        }
        BlueBrainPriorityDeferredBlockedBoundaryState::CaveatedPriorityDeferredBlockedSignal => {
            "caveated_priority_deferred_blocked_signal"
        }
        BlueBrainPriorityDeferredBlockedBoundaryState::InsufficientContractBasis => {
            "insufficient_contract_basis_boundary_state"
        }
        BlueBrainPriorityDeferredBlockedBoundaryState::NonCanonicalInternalOnlyCouplingPath => {
            "non_canonical_internal_only_coupling_path_boundary_state"
        }
    }
}

fn coherence_from_phase_nodes(nodes: &[BlueBrainKuramotoPhaseNodeInput]) -> u16 {
    let mut distance_sum: u64 = 0;
    let mut pair_count: u64 = 0;

    for (idx, left) in nodes.iter().enumerate() {
        for right in &nodes[(idx + 1)..] {
            let left_phase = left.phase_permille % 1000;
            let right_phase = right.phase_permille % 1000;
            let diff = left_phase.abs_diff(right_phase);
            let circular_distance = diff.min(1000_u16.saturating_sub(diff));
            let coupling = (u32::from(left.coupling_permille.min(1000))
                + u32::from(right.coupling_permille.min(1000)))
                / 2;
            let weighted_distance = (u32::from(circular_distance) * coupling) / 1000;
            distance_sum = distance_sum.saturating_add(u64::from(weighted_distance));
            pair_count = pair_count.saturating_add(1);
        }
    }

    if pair_count == 0 {
        return 0;
    }

    let average_distance = (distance_sum / pair_count) as u16;
    let bounded_average = average_distance.min(500);
    let coherence = (500_u32.saturating_sub(u32::from(bounded_average)) * 2) as u16;
    coherence.min(1000)
}

fn selection_hint_from_signal(
    diagnostic: BlueBrainKuramotoSynchronyDiagnostic,
    selection_posture: BlueBrainKuramotoSelectionPosture,
    memory_caveats_empty: bool,
) -> BlueBrainKuramotoSelectionHint {
    match (diagnostic, selection_posture, memory_caveats_empty) {
        (
            BlueBrainKuramotoSynchronyDiagnostic::Synchronized,
            BlueBrainKuramotoSelectionPosture::Selected,
            true,
        ) => BlueBrainKuramotoSelectionHint::KeepCurrentSelection,
        (BlueBrainKuramotoSynchronyDiagnostic::Desynchronized, _, _) => {
            BlueBrainKuramotoSelectionHint::IncreaseDeferralConfidence
        }
        _ => BlueBrainKuramotoSelectionHint::CaveateSelectionWeight,
    }
}

fn runtime_modulation_from_signal(
    diagnostic: BlueBrainKuramotoSynchronyDiagnostic,
    runtime_posture: BlueBrainKuramotoRuntimePosture,
) -> BlueBrainKuramotoRuntimeCaveatModulation {
    match (diagnostic, runtime_posture) {
        (
            BlueBrainKuramotoSynchronyDiagnostic::Synchronized,
            BlueBrainKuramotoRuntimePosture::Stable,
        ) => BlueBrainKuramotoRuntimeCaveatModulation::NoAdditionalCaveat,
        (_, BlueBrainKuramotoRuntimePosture::Blocked) => {
            BlueBrainKuramotoRuntimeCaveatModulation::EscalateRuntimeCaveat
        }
        (BlueBrainKuramotoSynchronyDiagnostic::Desynchronized, _) => {
            BlueBrainKuramotoRuntimeCaveatModulation::EscalateRuntimeCaveat
        }
        _ => BlueBrainKuramotoRuntimeCaveatModulation::AttachDynamicsCaveat,
    }
}

fn classify_kuramoto_input_group(group_ref: &str) -> BlueBrainKuramotoInputGroupClass {
    match group_ref {
        "runtime_state_group" => BlueBrainKuramotoInputGroupClass::RuntimeStateGroup,
        "selection_attention_group" => BlueBrainKuramotoInputGroupClass::SelectionAttentionGroup,
        "context_reference_group" => BlueBrainKuramotoInputGroupClass::ContextReferenceGroup,
        "memory_caveat_reference_group" => {
            BlueBrainKuramotoInputGroupClass::MemoryCaveatReferenceGroup
        }
        "evidence_derived_advisory_group" => {
            BlueBrainKuramotoInputGroupClass::EvidenceDerivedAdvisoryGroup
        }
        _ => BlueBrainKuramotoInputGroupClass::UnsupportedNonCanonicalInputGroup,
    }
}

fn classify_execution_feedback_state(
    input: &BlueBrainKuramotoModulationInput,
) -> BlueBrainDynamicsExecutionFeedbackState {
    if input.non_canonical_internal_only_path {
        return BlueBrainDynamicsExecutionFeedbackState::NonCanonicalInternalOnlyFeedbackPath;
    }
    if !input.blocked_input_refs.is_empty() || !input.blocked_execution_result_refs.is_empty() {
        return BlueBrainDynamicsExecutionFeedbackState::BlockedDynamicsFeedbackBasis;
    }
    if !input.insufficient_execution_result_refs.is_empty() {
        return BlueBrainDynamicsExecutionFeedbackState::InsufficientDynamicsFeedbackBasis;
    }
    if !input.unavailable_execution_result_refs.is_empty() {
        return BlueBrainDynamicsExecutionFeedbackState::UnavailableDynamicsFeedbackBasis;
    }
    if !input.failed_execution_result_refs.is_empty() {
        return BlueBrainDynamicsExecutionFeedbackState::FailedExecutionFeedbackBasis;
    }
    if !input.cancelled_execution_result_refs.is_empty() {
        return BlueBrainDynamicsExecutionFeedbackState::CancelledExecutionFeedbackBasis;
    }
    if !input.canonical_execution_result_refs.is_empty() {
        return BlueBrainDynamicsExecutionFeedbackState::ExecutionInformedDynamicsInput;
    }
    if !input.selected_context_refs.is_empty() || !input.selected_evidence_refs.is_empty() {
        return BlueBrainDynamicsExecutionFeedbackState::ReferenceInformedDynamicsInput;
    }
    if !input.diagnostic_only_feedback_refs.is_empty() {
        return BlueBrainDynamicsExecutionFeedbackState::DiagnosticOnlyDynamicsFeedback;
    }
    BlueBrainDynamicsExecutionFeedbackState::InsufficientDynamicsFeedbackBasis
}

fn selection_feedback_from_modulation_state(
    modulation_state: BlueBrainKuramotoModulationState,
    selection_hint: Option<BlueBrainKuramotoSelectionHint>,
) -> BlueBrainDynamicsSelectionFeedbackClass {
    match modulation_state {
        BlueBrainKuramotoModulationState::Insufficient => {
            BlueBrainDynamicsSelectionFeedbackClass::DynamicsInsufficientForModulation
        }
        BlueBrainKuramotoModulationState::Ignored
        | BlueBrainKuramotoModulationState::Blocked
        | BlueBrainKuramotoModulationState::Unavailable
        | BlueBrainKuramotoModulationState::NonCanonicalInternalOnlyPath => {
            BlueBrainDynamicsSelectionFeedbackClass::DynamicsIgnoredForCurrentSelection
        }
        BlueBrainKuramotoModulationState::Caveated => {
            BlueBrainDynamicsSelectionFeedbackClass::DynamicsCaveatAttached
        }
        BlueBrainKuramotoModulationState::AppliedAdvisoryOnly
        | BlueBrainKuramotoModulationState::NoOp => match selection_hint {
            Some(BlueBrainKuramotoSelectionHint::KeepCurrentSelection) => {
                BlueBrainDynamicsSelectionFeedbackClass::SelectionModulationObserved
            }
            Some(_) => BlueBrainDynamicsSelectionFeedbackClass::DynamicsCaveatAttached,
            None => BlueBrainDynamicsSelectionFeedbackClass::DynamicsIgnoredForCurrentSelection,
        },
    }
}

fn runtime_feedback_from_modulation_state(
    modulation_state: BlueBrainKuramotoModulationState,
    runtime_modulation: Option<BlueBrainKuramotoRuntimeCaveatModulation>,
) -> BlueBrainDynamicsRuntimeFeedbackClass {
    match modulation_state {
        BlueBrainKuramotoModulationState::Insufficient => {
            BlueBrainDynamicsRuntimeFeedbackClass::DynamicsInsufficientForModulation
        }
        BlueBrainKuramotoModulationState::Ignored
        | BlueBrainKuramotoModulationState::Blocked
        | BlueBrainKuramotoModulationState::Unavailable
        | BlueBrainKuramotoModulationState::NonCanonicalInternalOnlyPath => {
            BlueBrainDynamicsRuntimeFeedbackClass::DynamicsIgnoredForCurrentTransition
        }
        BlueBrainKuramotoModulationState::Caveated => {
            BlueBrainDynamicsRuntimeFeedbackClass::DynamicsCaveatAttached
        }
        BlueBrainKuramotoModulationState::AppliedAdvisoryOnly
        | BlueBrainKuramotoModulationState::NoOp => match runtime_modulation {
            Some(BlueBrainKuramotoRuntimeCaveatModulation::NoAdditionalCaveat) => {
                BlueBrainDynamicsRuntimeFeedbackClass::RuntimeModulationObserved
            }
            Some(_) => BlueBrainDynamicsRuntimeFeedbackClass::DynamicsCaveatAttached,
            None => BlueBrainDynamicsRuntimeFeedbackClass::DynamicsIgnoredForCurrentTransition,
        },
    }
}

fn selection_coupling_state_from_modulation_state(
    modulation_state: BlueBrainKuramotoModulationState,
    selection_hint: Option<BlueBrainKuramotoSelectionHint>,
) -> BlueBrainDynamicsAdvisoryCouplingState {
    match modulation_state {
        BlueBrainKuramotoModulationState::Insufficient => {
            BlueBrainDynamicsAdvisoryCouplingState::InsufficientAdvisoryCoupling
        }
        BlueBrainKuramotoModulationState::Blocked => {
            BlueBrainDynamicsAdvisoryCouplingState::BlockedAdvisoryCoupling
        }
        BlueBrainKuramotoModulationState::Unavailable
        | BlueBrainKuramotoModulationState::Ignored
        | BlueBrainKuramotoModulationState::NoOp => {
            BlueBrainDynamicsAdvisoryCouplingState::IgnoredAdvisoryCoupling
        }
        BlueBrainKuramotoModulationState::NonCanonicalInternalOnlyPath => {
            BlueBrainDynamicsAdvisoryCouplingState::NonCanonicalInternalOnlyCouplingPath
        }
        BlueBrainKuramotoModulationState::Caveated => {
            BlueBrainDynamicsAdvisoryCouplingState::CaveatedAdvisoryCoupling
        }
        BlueBrainKuramotoModulationState::AppliedAdvisoryOnly => {
            if selection_hint.is_some() {
                BlueBrainDynamicsAdvisoryCouplingState::SelectionAdvisoryCoupling
            } else {
                BlueBrainDynamicsAdvisoryCouplingState::IgnoredAdvisoryCoupling
            }
        }
    }
}

fn runtime_to_selection_contract_signal_from_state(
    modulation_state: BlueBrainKuramotoModulationState,
    selection_hint: Option<BlueBrainKuramotoSelectionHint>,
) -> BlueBrainRuntimeSelectionContractSignal {
    match modulation_state {
        BlueBrainKuramotoModulationState::NonCanonicalInternalOnlyPath => {
            BlueBrainRuntimeSelectionContractSignal::NonCanonicalInternalOnlyContractPath
        }
        BlueBrainKuramotoModulationState::Insufficient => {
            BlueBrainRuntimeSelectionContractSignal::InsufficientContractBasis
        }
        BlueBrainKuramotoModulationState::Blocked => {
            BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionBlockedSignal
        }
        _ if matches!(
            selection_hint,
            Some(BlueBrainKuramotoSelectionHint::IncreaseDeferralConfidence)
        ) =>
        {
            BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionDeferredSignal
        }
        BlueBrainKuramotoModulationState::Caveated
        | BlueBrainKuramotoModulationState::Unavailable => {
            BlueBrainRuntimeSelectionContractSignal::CaveatedContractSignal
        }
        _ => BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionAdvisorySignal,
    }
}

fn runtime_to_selection_contract_diagnostic_from_signal(
    signal: BlueBrainRuntimeSelectionContractSignal,
) -> BlueBrainRuntimeSelectionContractDiagnosticClass {
    match signal {
        BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionDeferredSignal => {
            BlueBrainRuntimeSelectionContractDiagnosticClass::DeferredContractDiagnostic
        }
        BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionBlockedSignal => {
            BlueBrainRuntimeSelectionContractDiagnosticClass::BlockedContractDiagnostic
        }
        BlueBrainRuntimeSelectionContractSignal::CaveatedContractSignal => {
            BlueBrainRuntimeSelectionContractDiagnosticClass::CaveatedContractDiagnostic
        }
        BlueBrainRuntimeSelectionContractSignal::InsufficientContractBasis => {
            BlueBrainRuntimeSelectionContractDiagnosticClass::InsufficientContractDiagnostic
        }
        BlueBrainRuntimeSelectionContractSignal::NonCanonicalInternalOnlyContractPath => {
            BlueBrainRuntimeSelectionContractDiagnosticClass::NonCanonicalInternalOnlyContractDiagnostic
        }
        BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionAdvisorySignal
        | BlueBrainRuntimeSelectionContractSignal::SelectionToRuntimeAdvisoryState
        | BlueBrainRuntimeSelectionContractSignal::SelectionToRuntimeDeferredState
        | BlueBrainRuntimeSelectionContractSignal::SelectionToRuntimeBlockedState => {
            BlueBrainRuntimeSelectionContractDiagnosticClass::RuntimeToSelectionDiagnostic
        }
    }
}

fn selection_to_runtime_contract_diagnostic_from_signal(
    signal: BlueBrainRuntimeSelectionContractSignal,
) -> BlueBrainRuntimeSelectionContractDiagnosticClass {
    match signal {
        BlueBrainRuntimeSelectionContractSignal::SelectionToRuntimeDeferredState => {
            BlueBrainRuntimeSelectionContractDiagnosticClass::DeferredContractDiagnostic
        }
        BlueBrainRuntimeSelectionContractSignal::SelectionToRuntimeBlockedState => {
            BlueBrainRuntimeSelectionContractDiagnosticClass::BlockedContractDiagnostic
        }
        BlueBrainRuntimeSelectionContractSignal::CaveatedContractSignal => {
            BlueBrainRuntimeSelectionContractDiagnosticClass::CaveatedContractDiagnostic
        }
        BlueBrainRuntimeSelectionContractSignal::InsufficientContractBasis => {
            BlueBrainRuntimeSelectionContractDiagnosticClass::InsufficientContractDiagnostic
        }
        BlueBrainRuntimeSelectionContractSignal::NonCanonicalInternalOnlyContractPath => {
            BlueBrainRuntimeSelectionContractDiagnosticClass::NonCanonicalInternalOnlyContractDiagnostic
        }
        BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionAdvisorySignal
        | BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionDeferredSignal
        | BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionBlockedSignal
        | BlueBrainRuntimeSelectionContractSignal::SelectionToRuntimeAdvisoryState => {
            BlueBrainRuntimeSelectionContractDiagnosticClass::SelectionToRuntimeDiagnostic
        }
    }
}

fn runtime_to_selection_contract_reason_from_signal(
    signal: BlueBrainRuntimeSelectionContractSignal,
    execution_feedback_state: BlueBrainDynamicsExecutionFeedbackState,
) -> BlueBrainRuntimeSelectionContractReason {
    match signal {
        BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionDeferredSignal => {
            BlueBrainRuntimeSelectionContractReason::DeferredDueToBoundedPrioritySelectionState
        }
        BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionBlockedSignal => {
            BlueBrainRuntimeSelectionContractReason::BlockedDueToContractBoundaryOrReferenceWeakness
        }
        BlueBrainRuntimeSelectionContractSignal::CaveatedContractSignal => {
            BlueBrainRuntimeSelectionContractReason::CaveatedDueToWeakOrPartialReferenceDynamicsExecutionBasis
        }
        BlueBrainRuntimeSelectionContractSignal::InsufficientContractBasis => {
            BlueBrainRuntimeSelectionContractReason::InsufficientDueToMissingBoundedContractBasis
        }
        BlueBrainRuntimeSelectionContractSignal::NonCanonicalInternalOnlyContractPath => {
            BlueBrainRuntimeSelectionContractReason::NonCanonicalInternalOnlyPathExcluded
        }
        BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionAdvisorySignal
        | BlueBrainRuntimeSelectionContractSignal::SelectionToRuntimeAdvisoryState
        | BlueBrainRuntimeSelectionContractSignal::SelectionToRuntimeDeferredState
        | BlueBrainRuntimeSelectionContractSignal::SelectionToRuntimeBlockedState => {
            if matches!(
                execution_feedback_state,
                BlueBrainDynamicsExecutionFeedbackState::ReferenceInformedDynamicsInput
                    | BlueBrainDynamicsExecutionFeedbackState::ExecutionInformedDynamicsInput
            ) {
                BlueBrainRuntimeSelectionContractReason::PriorityAdvisoryHintOnlyNoDirectSelectionAuthority
            } else {
                BlueBrainRuntimeSelectionContractReason::BlockedDueToContractBoundaryOrReferenceWeakness
            }
        }
    }
}

fn selection_to_runtime_contract_reason_from_signal(
    signal: BlueBrainRuntimeSelectionContractSignal,
    execution_feedback_state: BlueBrainDynamicsExecutionFeedbackState,
) -> BlueBrainRuntimeSelectionContractReason {
    match signal {
        BlueBrainRuntimeSelectionContractSignal::SelectionToRuntimeDeferredState => {
            BlueBrainRuntimeSelectionContractReason::DeferredDueToBoundedPrioritySelectionState
        }
        BlueBrainRuntimeSelectionContractSignal::SelectionToRuntimeBlockedState => {
            BlueBrainRuntimeSelectionContractReason::BlockedDueToContractBoundaryOrReferenceWeakness
        }
        BlueBrainRuntimeSelectionContractSignal::CaveatedContractSignal => {
            BlueBrainRuntimeSelectionContractReason::CaveatedDueToWeakOrPartialReferenceDynamicsExecutionBasis
        }
        BlueBrainRuntimeSelectionContractSignal::InsufficientContractBasis => {
            BlueBrainRuntimeSelectionContractReason::InsufficientDueToMissingBoundedContractBasis
        }
        BlueBrainRuntimeSelectionContractSignal::NonCanonicalInternalOnlyContractPath => {
            BlueBrainRuntimeSelectionContractReason::NonCanonicalInternalOnlyPathExcluded
        }
        BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionAdvisorySignal
        | BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionDeferredSignal
        | BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionBlockedSignal
        | BlueBrainRuntimeSelectionContractSignal::SelectionToRuntimeAdvisoryState => {
            if matches!(
                execution_feedback_state,
                BlueBrainDynamicsExecutionFeedbackState::ReferenceInformedDynamicsInput
                    | BlueBrainDynamicsExecutionFeedbackState::ExecutionInformedDynamicsInput
            ) {
                BlueBrainRuntimeSelectionContractReason::AdvisoryOnlyNoDirectActionAuthority
            } else {
                BlueBrainRuntimeSelectionContractReason::CaveatedDueToWeakOrPartialReferenceDynamicsExecutionBasis
            }
        }
    }
}

fn selection_to_runtime_contract_signal_from_state(
    modulation_state: BlueBrainKuramotoModulationState,
    runtime_modulation: Option<BlueBrainKuramotoRuntimeCaveatModulation>,
) -> BlueBrainRuntimeSelectionContractSignal {
    match modulation_state {
        BlueBrainKuramotoModulationState::NonCanonicalInternalOnlyPath => {
            BlueBrainRuntimeSelectionContractSignal::NonCanonicalInternalOnlyContractPath
        }
        BlueBrainKuramotoModulationState::Insufficient => {
            BlueBrainRuntimeSelectionContractSignal::InsufficientContractBasis
        }
        BlueBrainKuramotoModulationState::Blocked
        | BlueBrainKuramotoModulationState::Unavailable => {
            BlueBrainRuntimeSelectionContractSignal::SelectionToRuntimeBlockedState
        }
        BlueBrainKuramotoModulationState::Caveated => {
            BlueBrainRuntimeSelectionContractSignal::CaveatedContractSignal
        }
        _ if runtime_modulation
            == Some(BlueBrainKuramotoRuntimeCaveatModulation::EscalateRuntimeCaveat) =>
        {
            BlueBrainRuntimeSelectionContractSignal::SelectionToRuntimeDeferredState
        }
        _ => BlueBrainRuntimeSelectionContractSignal::SelectionToRuntimeAdvisoryState,
    }
}

fn boundary_state_from_contract_signal_and_reason(
    signal: BlueBrainRuntimeSelectionContractSignal,
    reason: BlueBrainRuntimeSelectionContractReason,
) -> BlueBrainPriorityDeferredBlockedBoundaryState {
    match signal {
        BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionDeferredSignal
        | BlueBrainRuntimeSelectionContractSignal::SelectionToRuntimeDeferredState => {
            BlueBrainPriorityDeferredBlockedBoundaryState::DeferredContractState
        }
        BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionBlockedSignal
        | BlueBrainRuntimeSelectionContractSignal::SelectionToRuntimeBlockedState => {
            BlueBrainPriorityDeferredBlockedBoundaryState::BlockedContractState
        }
        BlueBrainRuntimeSelectionContractSignal::CaveatedContractSignal => {
            BlueBrainPriorityDeferredBlockedBoundaryState::CaveatedPriorityDeferredBlockedSignal
        }
        BlueBrainRuntimeSelectionContractSignal::InsufficientContractBasis => {
            BlueBrainPriorityDeferredBlockedBoundaryState::InsufficientContractBasis
        }
        BlueBrainRuntimeSelectionContractSignal::NonCanonicalInternalOnlyContractPath => {
            BlueBrainPriorityDeferredBlockedBoundaryState::NonCanonicalInternalOnlyCouplingPath
        }
        BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionAdvisorySignal
        | BlueBrainRuntimeSelectionContractSignal::SelectionToRuntimeAdvisoryState => {
            if matches!(
                reason,
                BlueBrainRuntimeSelectionContractReason::PriorityAdvisoryHintOnlyNoDirectSelectionAuthority
                    | BlueBrainRuntimeSelectionContractReason::AdvisoryOnlyNoDirectActionAuthority
            ) {
                BlueBrainPriorityDeferredBlockedBoundaryState::PriorityAdvisoryHint
            } else {
                BlueBrainPriorityDeferredBlockedBoundaryState::CaveatedPriorityDeferredBlockedSignal
            }
        }
    }
}

fn runtime_coupling_state_from_modulation_state(
    modulation_state: BlueBrainKuramotoModulationState,
    runtime_modulation: Option<BlueBrainKuramotoRuntimeCaveatModulation>,
) -> BlueBrainDynamicsAdvisoryCouplingState {
    match modulation_state {
        BlueBrainKuramotoModulationState::Insufficient => {
            BlueBrainDynamicsAdvisoryCouplingState::InsufficientAdvisoryCoupling
        }
        BlueBrainKuramotoModulationState::Blocked => {
            BlueBrainDynamicsAdvisoryCouplingState::BlockedAdvisoryCoupling
        }
        BlueBrainKuramotoModulationState::Unavailable
        | BlueBrainKuramotoModulationState::Ignored
        | BlueBrainKuramotoModulationState::NoOp => {
            BlueBrainDynamicsAdvisoryCouplingState::IgnoredAdvisoryCoupling
        }
        BlueBrainKuramotoModulationState::NonCanonicalInternalOnlyPath => {
            BlueBrainDynamicsAdvisoryCouplingState::NonCanonicalInternalOnlyCouplingPath
        }
        BlueBrainKuramotoModulationState::Caveated => {
            BlueBrainDynamicsAdvisoryCouplingState::CaveatedAdvisoryCoupling
        }
        BlueBrainKuramotoModulationState::AppliedAdvisoryOnly => {
            if runtime_modulation.is_some() {
                BlueBrainDynamicsAdvisoryCouplingState::RuntimeAdvisoryCoupling
            } else {
                BlueBrainDynamicsAdvisoryCouplingState::IgnoredAdvisoryCoupling
            }
        }
    }
}

fn boundary_guard() -> BlueBrainKuramotoBoundaryGuard {
    BlueBrainKuramotoBoundaryGuard {
        runtime_state_mutation_allowed: false,
        selection_mutation_allowed: false,
        action_execution_allowed: false,
        tool_invocation_allowed: false,
        actual_action_result_allowed: false,
        memory_persistence_allowed: false,
        memory_commit_allowed: false,
        compute_invocation_allowed: false,
        safety_override_allowed: false,
        policy_decision_allowed: false,
        direct_reexecute_allowed: false,
        direct_retry_orchestration_allowed: false,
    }
}

fn append_kuramoto_guard_caveats(
    caveats: &mut Vec<String>,
    modulation_state: BlueBrainKuramotoModulationState,
) {
    if !matches!(
        modulation_state,
        BlueBrainKuramotoModulationState::Blocked
            | BlueBrainKuramotoModulationState::Unavailable
            | BlueBrainKuramotoModulationState::NonCanonicalInternalOnlyPath
    ) {
        return;
    }
    caveats.push("no_direct_action_allowed".to_string());
    caveats.push("no_direct_memory_allowed".to_string());
    caveats.push("no_direct_compute_allowed".to_string());
    caveats.push("no_direct_reexecute_allowed".to_string());
    caveats.push("no_direct_retry_orchestration_allowed".to_string());
    caveats.push("no_safety_override_allowed".to_string());
}

fn append_execution_feedback_caveats(
    caveats: &mut Vec<String>,
    execution_feedback_state: BlueBrainDynamicsExecutionFeedbackState,
) {
    match execution_feedback_state {
        BlueBrainDynamicsExecutionFeedbackState::ExecutionInformedDynamicsInput => {
            caveats.push("execution_informed_modulation_observed".to_string());
        }
        BlueBrainDynamicsExecutionFeedbackState::ReferenceInformedDynamicsInput => {
            caveats.push("reference_informed_modulation_observed".to_string());
        }
        BlueBrainDynamicsExecutionFeedbackState::FailedExecutionFeedbackBasis
        | BlueBrainDynamicsExecutionFeedbackState::CancelledExecutionFeedbackBasis => {
            caveats.push("caveated_execution_feedback_basis".to_string());
        }
        BlueBrainDynamicsExecutionFeedbackState::InsufficientDynamicsFeedbackBasis => {
            caveats.push("insufficient_dynamics_feedback_basis".to_string());
        }
        BlueBrainDynamicsExecutionFeedbackState::BlockedDynamicsFeedbackBasis => {
            caveats.push("blocked_dynamics_feedback_basis".to_string());
        }
        BlueBrainDynamicsExecutionFeedbackState::UnavailableDynamicsFeedbackBasis => {
            caveats.push("unavailable_dynamics_feedback_basis".to_string());
        }
        BlueBrainDynamicsExecutionFeedbackState::DiagnosticOnlyDynamicsFeedback => {
            caveats.push("diagnostic_only_dynamics_feedback".to_string());
        }
        BlueBrainDynamicsExecutionFeedbackState::NonCanonicalInternalOnlyFeedbackPath => {
            caveats.push("non_canonical_internal_only_feedback_path".to_string());
        }
    }
}

fn hh_boundary_guard() -> BlueBrainHodgkinHuxleyBoundaryGuard {
    BlueBrainHodgkinHuxleyBoundaryGuard {
        runtime_state_mutation_allowed: false,
        selection_mutation_allowed: false,
        memory_mutation_allowed: false,
        action_execution_allowed: false,
        tool_invocation_allowed: false,
        compute_invocation_allowed: false,
        safety_override_allowed: false,
        policy_decision_allowed: false,
        actual_action_result_allowed: false,
        memory_commit_allowed: false,
    }
}

fn compute_effective_stability_permille(input: &BlueBrainHodgkinHuxleyDiagnosticInput) -> u16 {
    let sodium = i32::from(input.model_parameters.sodium_conductance_permille.min(2000));
    let potassium = i32::from(
        input
            .model_parameters
            .potassium_conductance_permille
            .min(2000),
    );
    let leak = i32::from(input.model_parameters.leak_conductance_permille.min(1000));
    let drive = i32::from(
        input
            .simulation_parameters
            .stimulus_nanoamp_permille
            .min(2000),
    );
    let steps = i32::from(input.simulation_parameters.integration_steps.min(4096));

    let ion_balance = 1000 - (sodium - potassium).abs().min(1000);
    let drive_penalty = drive.saturating_sub(1200).min(800);
    let leak_penalty = leak.saturating_sub(400).min(400);
    let short_horizon_penalty = (256 - steps).clamp(0, 256);

    let stability = ion_balance - drive_penalty - leak_penalty - short_horizon_penalty;
    stability.clamp(0, 1000) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_input(scope: BlueBrainKuramotoScopeState) -> BlueBrainKuramotoModulationInput {
        BlueBrainKuramotoModulationInput {
            scope,
            selection_posture: BlueBrainKuramotoSelectionPosture::Selected,
            runtime_posture: BlueBrainKuramotoRuntimePosture::Stable,
            selected_context_refs: vec!["ctx:b".to_string(), "ctx:a".to_string()],
            selected_evidence_refs: vec!["ev:2".to_string(), "ev:1".to_string()],
            memory_caveats: vec![],
            unsupported_input_refs: vec![],
            blocked_input_refs: vec![],
            canonical_execution_result_refs: vec![],
            failed_execution_result_refs: vec![],
            cancelled_execution_result_refs: vec![],
            blocked_execution_result_refs: vec![],
            insufficient_execution_result_refs: vec![],
            unavailable_execution_result_refs: vec![],
            diagnostic_only_feedback_refs: vec![],
            non_canonical_internal_only_path: false,
            phase_nodes: vec![
                BlueBrainKuramotoPhaseNodeInput {
                    group_ref: "selection_attention_group".to_string(),
                    phase_permille: 120,
                    coupling_permille: 700,
                },
                BlueBrainKuramotoPhaseNodeInput {
                    group_ref: "runtime_state_group".to_string(),
                    phase_permille: 130,
                    coupling_permille: 700,
                },
            ],
        }
    }

    #[test]
    fn scope_states_are_distinguishable_for_minimal_kuramoto_path() {
        let scopes = [
            BlueBrainKuramotoScopeState::SimulationOnly,
            BlueBrainKuramotoScopeState::DiagnosticOnly,
            BlueBrainKuramotoScopeState::SelectionModulating,
            BlueBrainKuramotoScopeState::RuntimeCaveatModulating,
            BlueBrainKuramotoScopeState::NotImplementedOrNotSuitableNow,
        ];
        assert_eq!(scopes.len(), 5);
        assert_ne!(scopes[0], scopes[1]);
        assert_ne!(scopes[1], scopes[2]);
        assert_ne!(scopes[2], scopes[3]);
        assert_ne!(scopes[3], scopes[4]);
    }

    #[test]
    fn canonical_dynamics_diagnostics_map_contains_required_classes() {
        let map = CANONICAL_BLUE_BRAIN_DYNAMICS_DIAGNOSTICS_MAP;
        assert!(map.iter().any(|lane| {
            lane.diagnostic_class == BlueBrainDynamicsDiagnosticClass::DynamicsDiagnosticObserved
        }));
        assert!(map.iter().any(|lane| {
            lane.diagnostic_class == BlueBrainDynamicsDiagnosticClass::KuramotoModulationDiagnostic
        }));
        assert!(map.iter().any(|lane| {
            lane.diagnostic_class
                == BlueBrainDynamicsDiagnosticClass::HodgkinHuxleySimulationDiagnostic
        }));
        assert!(map.iter().any(
            |lane| lane.diagnostic_class == BlueBrainDynamicsDiagnosticClass::DynamicsCaveated
        ));
        assert!(map.iter().any(|lane| {
            lane.diagnostic_class == BlueBrainDynamicsDiagnosticClass::DynamicsInsufficient
        }));
        assert!(map
            .iter()
            .any(|lane| lane.diagnostic_class == BlueBrainDynamicsDiagnosticClass::DynamicsFailed));
        assert!(map.iter().any(|lane| {
            lane.diagnostic_class == BlueBrainDynamicsDiagnosticClass::DynamicsUnavailable
        }));
        assert!(
            map.iter()
                .any(|lane| lane.diagnostic_class
                    == BlueBrainDynamicsDiagnosticClass::DynamicsIgnored)
        );
        assert!(map.iter().any(|lane| {
            lane.diagnostic_class
                == BlueBrainDynamicsDiagnosticClass::NonCanonicalInternalOnlyDynamicsDiagnostic
        }));
    }

    #[test]
    fn input_surface_stays_canonicalized_and_coherence_is_deterministic() {
        let result = evaluate_blue_brain_kuramoto_modulation(base_input(
            BlueBrainKuramotoScopeState::DiagnosticOnly,
        ));
        assert!(result.coherence_permille >= 900);
        assert_eq!(
            result.diagnostic,
            BlueBrainKuramotoSynchronyDiagnostic::Synchronized
        );
    }

    #[test]
    fn insufficient_inputs_are_explicitly_caveated() {
        let mut input = base_input(BlueBrainKuramotoScopeState::DiagnosticOnly);
        input.phase_nodes.truncate(1);
        let result = evaluate_blue_brain_kuramoto_modulation(input);
        assert_eq!(
            result.diagnostic,
            BlueBrainKuramotoSynchronyDiagnostic::InsufficientInput
        );
        assert_eq!(
            result.modulation_state,
            BlueBrainKuramotoModulationState::Insufficient
        );
        assert!(result
            .caveats
            .iter()
            .any(|item| item == "insufficient_dynamics_input"));
    }

    #[test]
    fn canonical_kuramoto_input_group_map_is_unique_and_complete() {
        let mut refs: Vec<&str> = CANONICAL_BLUE_BRAIN_KURAMOTO_INPUT_GROUP_MAP
            .iter()
            .map(|lane| lane.group_ref)
            .collect();
        refs.sort_unstable();
        refs.dedup();
        assert_eq!(
            refs.len(),
            CANONICAL_BLUE_BRAIN_KURAMOTO_INPUT_GROUP_MAP.len()
        );
        assert!(refs.contains(&"runtime_state_group"));
        assert!(refs.contains(&"selection_attention_group"));
        assert!(refs.contains(&"context_reference_group"));
        assert!(refs.contains(&"memory_caveat_reference_group"));
        assert!(refs.contains(&"evidence_derived_advisory_group"));
        assert!(refs.contains(&"unsupported_non_canonical_input_group"));
    }

    #[test]
    fn unsupported_and_blocked_input_groups_remain_explicit() {
        let mut unsupported = base_input(BlueBrainKuramotoScopeState::DiagnosticOnly);
        unsupported.unsupported_input_refs = vec!["tool:direct_action".to_string()];
        let unsupported_result = evaluate_blue_brain_kuramoto_modulation(unsupported);
        assert_eq!(
            unsupported_result.input_basis,
            BlueBrainKuramotoInputBasisClass::UnsupportedInputBasis
        );
        assert_eq!(
            unsupported_result.modulation_state,
            BlueBrainKuramotoModulationState::Caveated
        );
        assert!(unsupported_result
            .caveats
            .iter()
            .any(|item| item == "unsupported_input_group_present"));

        let mut blocked = base_input(BlueBrainKuramotoScopeState::RuntimeCaveatModulating);
        blocked.blocked_input_refs = vec!["policy:authority_hook".to_string()];
        let blocked_result = evaluate_blue_brain_kuramoto_modulation(blocked);
        assert_eq!(
            blocked_result.input_basis,
            BlueBrainKuramotoInputBasisClass::BlockedInputBasis
        );
        assert_eq!(
            blocked_result.modulation_state,
            BlueBrainKuramotoModulationState::Blocked
        );
        assert!(blocked_result
            .caveats
            .iter()
            .any(|item| item == "blocked_input_group_present"));
    }

    #[test]
    fn selection_and_runtime_modulation_are_advisory_only() {
        let selection_result = evaluate_blue_brain_kuramoto_modulation(base_input(
            BlueBrainKuramotoScopeState::SelectionModulating,
        ));
        assert!(selection_result.selection_hint.is_some());
        assert!(selection_result.runtime_modulation.is_none());
        assert_eq!(
            selection_result.selection_feedback,
            BlueBrainDynamicsSelectionFeedbackClass::SelectionModulationObserved
        );
        assert_eq!(
            selection_result.runtime_feedback,
            BlueBrainDynamicsRuntimeFeedbackClass::DynamicsIgnoredForCurrentTransition
        );

        let runtime_result = evaluate_blue_brain_kuramoto_modulation(base_input(
            BlueBrainKuramotoScopeState::RuntimeCaveatModulating,
        ));
        assert!(runtime_result.selection_hint.is_none());
        assert!(runtime_result.runtime_modulation.is_some());
        assert_eq!(
            runtime_result.selection_feedback,
            BlueBrainDynamicsSelectionFeedbackClass::DynamicsIgnoredForCurrentSelection
        );
    }

    #[test]
    fn modulation_path_cannot_trigger_execution_memory_compute_or_policy() {
        let result = evaluate_blue_brain_kuramoto_modulation(base_input(
            BlueBrainKuramotoScopeState::DiagnosticOnly,
        ));
        assert!(!result.boundary_guard.action_execution_allowed);
        assert!(!result.boundary_guard.tool_invocation_allowed);
        assert!(!result.boundary_guard.actual_action_result_allowed);
        assert!(!result.boundary_guard.memory_persistence_allowed);
        assert!(!result.boundary_guard.memory_commit_allowed);
        assert!(!result.boundary_guard.compute_invocation_allowed);
        assert!(!result.boundary_guard.safety_override_allowed);
        assert!(!result.boundary_guard.policy_decision_allowed);
        assert!(!result.boundary_guard.direct_reexecute_allowed);
        assert!(!result.boundary_guard.direct_retry_orchestration_allowed);
        assert!(!result.boundary_guard.runtime_state_mutation_allowed);
        assert!(!result.boundary_guard.selection_mutation_allowed);
    }

    #[test]
    fn unavailable_or_simulation_only_kuramoto_stays_non_executing_and_ignored() {
        let simulation_only = evaluate_blue_brain_kuramoto_modulation(base_input(
            BlueBrainKuramotoScopeState::SimulationOnly,
        ));
        assert_eq!(
            simulation_only.dynamics_diagnostic_class,
            BlueBrainDynamicsDiagnosticClass::DynamicsIgnored
        );
        assert_eq!(
            simulation_only.selection_feedback,
            BlueBrainDynamicsSelectionFeedbackClass::DynamicsIgnoredForCurrentSelection
        );

        let unavailable = evaluate_blue_brain_kuramoto_modulation(base_input(
            BlueBrainKuramotoScopeState::NotImplementedOrNotSuitableNow,
        ));
        assert_eq!(
            unavailable.dynamics_diagnostic_class,
            BlueBrainDynamicsDiagnosticClass::DynamicsUnavailable
        );
        assert_eq!(
            unavailable.modulation_state,
            BlueBrainKuramotoModulationState::Unavailable
        );
        assert!(unavailable
            .caveats
            .iter()
            .any(|item| item == "dynamics_path_unavailable"));
    }

    #[test]
    fn modulation_states_are_explicitly_distinguishable_for_operational_hardening() {
        let mut caveated = base_input(BlueBrainKuramotoScopeState::DiagnosticOnly);
        caveated.runtime_posture = BlueBrainKuramotoRuntimePosture::Caveated;
        assert_eq!(
            evaluate_blue_brain_kuramoto_modulation(caveated).modulation_state,
            BlueBrainKuramotoModulationState::Caveated
        );

        let ignored = evaluate_blue_brain_kuramoto_modulation(base_input(
            BlueBrainKuramotoScopeState::SimulationOnly,
        ));
        assert_eq!(
            ignored.modulation_state,
            BlueBrainKuramotoModulationState::Ignored
        );
        assert_eq!(
            ignored.modulation_reason,
            Some(BlueBrainKuramotoModulationReason::IgnoredByRuntimeSelectionContext)
        );

        let no_op = evaluate_blue_brain_kuramoto_modulation(base_input(
            BlueBrainKuramotoScopeState::DiagnosticOnly,
        ));
        assert_eq!(
            no_op.modulation_state,
            BlueBrainKuramotoModulationState::NoOp
        );
        assert_eq!(
            no_op.modulation_reason,
            Some(BlueBrainKuramotoModulationReason::NoOpNeutralDeterministicResult)
        );

        let mut blocked = base_input(BlueBrainKuramotoScopeState::RuntimeCaveatModulating);
        blocked.runtime_posture = BlueBrainKuramotoRuntimePosture::Blocked;
        assert_eq!(
            evaluate_blue_brain_kuramoto_modulation(blocked).modulation_state,
            BlueBrainKuramotoModulationState::Blocked
        );

        let mut applied = base_input(BlueBrainKuramotoScopeState::SelectionModulating);
        applied.selection_posture = BlueBrainKuramotoSelectionPosture::Deferred;
        applied.memory_caveats.clear();
        assert_eq!(
            evaluate_blue_brain_kuramoto_modulation(applied).modulation_state,
            BlueBrainKuramotoModulationState::AppliedAdvisoryOnly
        );

        let mut non_canonical = base_input(BlueBrainKuramotoScopeState::DiagnosticOnly);
        non_canonical.non_canonical_internal_only_path = true;
        let non_canonical_result = evaluate_blue_brain_kuramoto_modulation(non_canonical);
        assert_eq!(
            non_canonical_result.modulation_state,
            BlueBrainKuramotoModulationState::NonCanonicalInternalOnlyPath
        );
        assert_eq!(
            non_canonical_result.dynamics_diagnostic_class,
            BlueBrainDynamicsDiagnosticClass::NonCanonicalInternalOnlyDynamicsDiagnostic
        );
    }

    #[test]
    fn modulation_state_tokens_cover_canonical_operational_states() {
        let states = [
            BlueBrainKuramotoModulationState::AppliedAdvisoryOnly,
            BlueBrainKuramotoModulationState::Caveated,
            BlueBrainKuramotoModulationState::Insufficient,
            BlueBrainKuramotoModulationState::Ignored,
            BlueBrainKuramotoModulationState::NoOp,
            BlueBrainKuramotoModulationState::Blocked,
            BlueBrainKuramotoModulationState::Unavailable,
            BlueBrainKuramotoModulationState::NonCanonicalInternalOnlyPath,
        ];
        let mut tokens: Vec<&str> = states
            .into_iter()
            .map(kuramoto_modulation_state_token)
            .collect();
        tokens.sort_unstable();
        tokens.dedup();
        assert_eq!(tokens.len(), 8);
        assert!(tokens.contains(&"applied_advisory_only"));
        assert!(tokens.contains(&"caveated"));
        assert!(tokens.contains(&"insufficient"));
        assert!(tokens.contains(&"ignored"));
        assert!(tokens.contains(&"no_op"));
        assert!(tokens.contains(&"blocked"));
        assert!(tokens.contains(&"unavailable"));
        assert!(tokens.contains(&"non_canonical_internal_only_path"));
    }

    #[test]
    fn modulation_diagnostic_classes_cover_canonical_kuramoto_feedback_states() {
        let classes = [
            BlueBrainKuramotoModulationDiagnosticClass::ModulationAppliedDiagnostic,
            BlueBrainKuramotoModulationDiagnosticClass::ModulationCaveatedDiagnostic,
            BlueBrainKuramotoModulationDiagnosticClass::ModulationInsufficientDiagnostic,
            BlueBrainKuramotoModulationDiagnosticClass::ModulationIgnoredDiagnostic,
            BlueBrainKuramotoModulationDiagnosticClass::ModulationNoOpDiagnostic,
            BlueBrainKuramotoModulationDiagnosticClass::ModulationBlockedDiagnostic,
            BlueBrainKuramotoModulationDiagnosticClass::ModulationUnavailableDiagnostic,
            BlueBrainKuramotoModulationDiagnosticClass::NonCanonicalInternalOnlyDynamicsDiagnostic,
        ];
        let mut tokens: Vec<&str> = classes
            .into_iter()
            .map(kuramoto_modulation_diagnostic_class_token)
            .collect();
        tokens.sort_unstable();
        tokens.dedup();
        assert_eq!(tokens.len(), 8);
        assert!(tokens.contains(&"modulation_applied_diagnostic"));
        assert!(tokens.contains(&"modulation_caveated_diagnostic"));
        assert!(tokens.contains(&"modulation_insufficient_diagnostic"));
        assert!(tokens.contains(&"modulation_ignored_diagnostic"));
        assert!(tokens.contains(&"modulation_no_op_diagnostic"));
        assert!(tokens.contains(&"modulation_blocked_diagnostic"));
        assert!(tokens.contains(&"modulation_unavailable_diagnostic"));
        assert!(tokens.contains(&"non_canonical_internal_only_dynamics_diagnostic"));
    }

    #[test]
    fn modulation_reason_tokens_cover_canonical_caveat_noop_blocked_and_unavailable_reasons() {
        let reasons = [
            None,
            Some(BlueBrainKuramotoModulationReason::InsufficientInputGroupBasis),
            Some(BlueBrainKuramotoModulationReason::CaveatedPartialOrWeakBasis),
            Some(BlueBrainKuramotoModulationReason::NoOpNeutralDeterministicResult),
            Some(BlueBrainKuramotoModulationReason::IgnoredByRuntimeSelectionContext),
            Some(BlueBrainKuramotoModulationReason::BlockedByGuardBoundaryCondition),
            Some(BlueBrainKuramotoModulationReason::UnavailableOperationalPreconditions),
            Some(BlueBrainKuramotoModulationReason::NonCanonicalInternalOnlyPath),
        ];
        let mut tokens: Vec<&str> = reasons
            .into_iter()
            .map(kuramoto_modulation_reason_token)
            .collect();
        tokens.sort_unstable();
        tokens.dedup();
        assert_eq!(tokens.len(), 8);
        assert!(tokens.contains(&"none"));
        assert!(tokens.contains(&"insufficient_input_group_basis"));
        assert!(tokens.contains(&"caveated_partial_or_weak_basis"));
        assert!(tokens.contains(&"no_op_neutral_deterministic_result"));
        assert!(tokens.contains(&"ignored_by_runtime_selection_context"));
        assert!(tokens.contains(&"blocked_by_guard_boundary_condition"));
        assert!(tokens.contains(&"unavailable_operational_preconditions"));
        assert!(tokens.contains(&"non_canonical_internal_only_path"));
    }

    #[test]
    fn blocked_kuramoto_state_includes_explicit_no_direct_guard_caveats() {
        let mut blocked = base_input(BlueBrainKuramotoScopeState::RuntimeCaveatModulating);
        blocked.runtime_posture = BlueBrainKuramotoRuntimePosture::Blocked;
        let result = evaluate_blue_brain_kuramoto_modulation(blocked);
        assert_eq!(
            result.diagnostic_class,
            BlueBrainKuramotoModulationDiagnosticClass::ModulationBlockedDiagnostic
        );
        assert_eq!(
            result.modulation_reason,
            Some(BlueBrainKuramotoModulationReason::BlockedByGuardBoundaryCondition)
        );
        assert!(result
            .caveats
            .iter()
            .any(|item| item == "no_direct_action_allowed"));
        assert!(result
            .caveats
            .iter()
            .any(|item| item == "no_direct_memory_allowed"));
        assert!(result
            .caveats
            .iter()
            .any(|item| item == "no_direct_compute_allowed"));
        assert!(result
            .caveats
            .iter()
            .any(|item| item == "no_direct_reexecute_allowed"));
        assert!(result
            .caveats
            .iter()
            .any(|item| item == "no_direct_retry_orchestration_allowed"));
        assert!(result
            .caveats
            .iter()
            .any(|item| item == "no_safety_override_allowed"));
    }

    #[test]
    fn caveated_kuramoto_state_never_reports_strong_selection_or_runtime_feedback() {
        let mut input = base_input(BlueBrainKuramotoScopeState::DiagnosticOnly);
        input.runtime_posture = BlueBrainKuramotoRuntimePosture::Caveated;
        let result = evaluate_blue_brain_kuramoto_modulation(input);
        assert_eq!(
            result.modulation_state,
            BlueBrainKuramotoModulationState::Caveated
        );
        assert_eq!(
            result.selection_feedback,
            BlueBrainDynamicsSelectionFeedbackClass::DynamicsCaveatAttached
        );
        assert_eq!(
            result.runtime_feedback,
            BlueBrainDynamicsRuntimeFeedbackClass::DynamicsCaveatAttached
        );
    }

    #[test]
    fn insufficient_kuramoto_state_never_reports_supported_feedback() {
        let mut input = base_input(BlueBrainKuramotoScopeState::DiagnosticOnly);
        input.phase_nodes.truncate(1);
        let result = evaluate_blue_brain_kuramoto_modulation(input);
        assert_eq!(
            result.modulation_state,
            BlueBrainKuramotoModulationState::Insufficient
        );
        assert_eq!(
            result.selection_feedback,
            BlueBrainDynamicsSelectionFeedbackClass::DynamicsInsufficientForModulation
        );
        assert_eq!(
            result.runtime_feedback,
            BlueBrainDynamicsRuntimeFeedbackClass::DynamicsInsufficientForModulation
        );
    }

    #[test]
    fn dynamics_execution_feedback_map_contains_canonical_states() {
        let map = CANONICAL_BLUE_BRAIN_DYNAMICS_EXECUTION_FEEDBACK_MAP;
        assert!(map.iter().any(|lane| {
            lane.state == BlueBrainDynamicsExecutionFeedbackState::ExecutionInformedDynamicsInput
        }));
        assert!(map.iter().any(|lane| {
            lane.state == BlueBrainDynamicsExecutionFeedbackState::ReferenceInformedDynamicsInput
        }));
        assert!(map.iter().any(|lane| {
            lane.state == BlueBrainDynamicsExecutionFeedbackState::FailedExecutionFeedbackBasis
        }));
        assert!(map.iter().any(|lane| {
            lane.state == BlueBrainDynamicsExecutionFeedbackState::InsufficientDynamicsFeedbackBasis
        }));
        assert!(map.iter().any(|lane| {
            lane.state == BlueBrainDynamicsExecutionFeedbackState::BlockedDynamicsFeedbackBasis
        }));
        assert!(map.iter().any(|lane| {
            lane.state == BlueBrainDynamicsExecutionFeedbackState::UnavailableDynamicsFeedbackBasis
        }));
        assert!(map.iter().any(|lane| {
            lane.state == BlueBrainDynamicsExecutionFeedbackState::DiagnosticOnlyDynamicsFeedback
        }));
        assert!(map.iter().any(|lane| {
            lane.state
                == BlueBrainDynamicsExecutionFeedbackState::NonCanonicalInternalOnlyFeedbackPath
        }));
    }

    #[test]
    fn dynamics_advisory_coupling_map_contains_canonical_states() {
        let map = CANONICAL_BLUE_BRAIN_DYNAMICS_ADVISORY_COUPLING_MAP;
        assert!(map.iter().any(|lane| {
            lane.coupling_state == BlueBrainDynamicsAdvisoryCouplingState::RuntimeAdvisoryCoupling
        }));
        assert!(map.iter().any(|lane| {
            lane.coupling_state == BlueBrainDynamicsAdvisoryCouplingState::SelectionAdvisoryCoupling
        }));
        assert!(map.iter().any(|lane| {
            lane.coupling_state == BlueBrainDynamicsAdvisoryCouplingState::CaveatedAdvisoryCoupling
        }));
        assert!(map.iter().any(|lane| {
            lane.coupling_state
                == BlueBrainDynamicsAdvisoryCouplingState::InsufficientAdvisoryCoupling
        }));
        assert!(map.iter().any(|lane| {
            lane.coupling_state == BlueBrainDynamicsAdvisoryCouplingState::BlockedAdvisoryCoupling
        }));
        assert!(map.iter().any(|lane| {
            lane.coupling_state == BlueBrainDynamicsAdvisoryCouplingState::IgnoredAdvisoryCoupling
        }));
        assert!(map.iter().any(|lane| {
            lane.coupling_state
                == BlueBrainDynamicsAdvisoryCouplingState::NonCanonicalInternalOnlyCouplingPath
        }));
    }

    #[test]
    fn execution_and_reference_feedback_basis_stay_distinguishable() {
        let mut execution_informed = base_input(BlueBrainKuramotoScopeState::DiagnosticOnly);
        execution_informed
            .canonical_execution_result_refs
            .push("bb14:minimal_execution:h1:emit_canonical_signal:result:completed".to_string());
        let execution_informed_result = evaluate_blue_brain_kuramoto_modulation(execution_informed);
        assert_eq!(
            execution_informed_result.execution_feedback_state,
            BlueBrainDynamicsExecutionFeedbackState::ExecutionInformedDynamicsInput
        );
        assert!(execution_informed_result
            .caveats
            .iter()
            .any(|item| item == "execution_informed_modulation_observed"));

        let reference_informed = evaluate_blue_brain_kuramoto_modulation(base_input(
            BlueBrainKuramotoScopeState::DiagnosticOnly,
        ));
        assert_eq!(
            reference_informed.execution_feedback_state,
            BlueBrainDynamicsExecutionFeedbackState::ReferenceInformedDynamicsInput
        );
        assert!(reference_informed
            .caveats
            .iter()
            .any(|item| item == "reference_informed_modulation_observed"));
        assert_eq!(
            reference_informed.selection_coupling_state,
            BlueBrainDynamicsAdvisoryCouplingState::IgnoredAdvisoryCoupling
        );
        assert_eq!(
            reference_informed.runtime_coupling_state,
            BlueBrainDynamicsAdvisoryCouplingState::IgnoredAdvisoryCoupling
        );
    }

    #[test]
    fn runtime_and_selection_advisory_coupling_remain_distinguishable() {
        let selection = evaluate_blue_brain_kuramoto_modulation(base_input(
            BlueBrainKuramotoScopeState::SelectionModulating,
        ));
        assert_eq!(
            selection.selection_coupling_state,
            BlueBrainDynamicsAdvisoryCouplingState::SelectionAdvisoryCoupling
        );
        assert_eq!(
            selection.runtime_coupling_state,
            BlueBrainDynamicsAdvisoryCouplingState::IgnoredAdvisoryCoupling
        );

        let runtime = evaluate_blue_brain_kuramoto_modulation(base_input(
            BlueBrainKuramotoScopeState::RuntimeCaveatModulating,
        ));
        assert_eq!(
            runtime.runtime_coupling_state,
            BlueBrainDynamicsAdvisoryCouplingState::RuntimeAdvisoryCoupling
        );
        assert_eq!(
            runtime.selection_coupling_state,
            BlueBrainDynamicsAdvisoryCouplingState::IgnoredAdvisoryCoupling
        );
    }

    #[test]
    fn caveated_insufficient_blocked_and_noncanonical_coupling_states_stay_explicit() {
        let mut caveated = base_input(BlueBrainKuramotoScopeState::DiagnosticOnly);
        caveated.runtime_posture = BlueBrainKuramotoRuntimePosture::Caveated;
        let caveated_result = evaluate_blue_brain_kuramoto_modulation(caveated);
        assert_eq!(
            caveated_result.runtime_coupling_state,
            BlueBrainDynamicsAdvisoryCouplingState::CaveatedAdvisoryCoupling
        );

        let mut insufficient = base_input(BlueBrainKuramotoScopeState::DiagnosticOnly);
        insufficient.phase_nodes.truncate(1);
        let insufficient_result = evaluate_blue_brain_kuramoto_modulation(insufficient);
        assert_eq!(
            insufficient_result.selection_coupling_state,
            BlueBrainDynamicsAdvisoryCouplingState::InsufficientAdvisoryCoupling
        );

        let mut blocked = base_input(BlueBrainKuramotoScopeState::RuntimeCaveatModulating);
        blocked.runtime_posture = BlueBrainKuramotoRuntimePosture::Blocked;
        let blocked_result = evaluate_blue_brain_kuramoto_modulation(blocked);
        assert_eq!(
            blocked_result.runtime_coupling_state,
            BlueBrainDynamicsAdvisoryCouplingState::BlockedAdvisoryCoupling
        );

        let mut non_canonical = base_input(BlueBrainKuramotoScopeState::DiagnosticOnly);
        non_canonical.non_canonical_internal_only_path = true;
        let non_canonical_result = evaluate_blue_brain_kuramoto_modulation(non_canonical);
        assert_eq!(
            non_canonical_result.selection_coupling_state,
            BlueBrainDynamicsAdvisoryCouplingState::NonCanonicalInternalOnlyCouplingPath
        );
        assert_eq!(
            non_canonical_result.runtime_to_selection_contract_signal,
            BlueBrainRuntimeSelectionContractSignal::NonCanonicalInternalOnlyContractPath
        );
    }

    #[test]
    fn runtime_selection_contract_signals_stay_directional_and_distinct() {
        let selection_lane = evaluate_blue_brain_kuramoto_modulation(base_input(
            BlueBrainKuramotoScopeState::SelectionModulating,
        ));
        assert_eq!(
            selection_lane.runtime_to_selection_contract_signal,
            BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionAdvisorySignal
        );
        assert_eq!(
            selection_lane.selection_to_runtime_contract_signal,
            BlueBrainRuntimeSelectionContractSignal::SelectionToRuntimeAdvisoryState
        );
        assert_eq!(
            selection_lane.runtime_to_selection_contract_diagnostic,
            BlueBrainRuntimeSelectionContractDiagnosticClass::RuntimeToSelectionDiagnostic
        );
        assert_eq!(
            selection_lane.selection_to_runtime_contract_diagnostic,
            BlueBrainRuntimeSelectionContractDiagnosticClass::SelectionToRuntimeDiagnostic
        );
        assert_eq!(
            selection_lane.runtime_to_selection_contract_reason,
            BlueBrainRuntimeSelectionContractReason::PriorityAdvisoryHintOnlyNoDirectSelectionAuthority
        );
        assert_eq!(
            selection_lane.selection_to_runtime_contract_reason,
            BlueBrainRuntimeSelectionContractReason::AdvisoryOnlyNoDirectActionAuthority
        );
        assert_eq!(
            selection_lane.runtime_boundary_state,
            BlueBrainPriorityDeferredBlockedBoundaryState::PriorityAdvisoryHint
        );

        let mut blocked = base_input(BlueBrainKuramotoScopeState::RuntimeCaveatModulating);
        blocked.runtime_posture = BlueBrainKuramotoRuntimePosture::Blocked;
        let blocked_result = evaluate_blue_brain_kuramoto_modulation(blocked);
        assert_eq!(
            blocked_result.runtime_to_selection_contract_signal,
            BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionBlockedSignal
        );
        assert_eq!(
            blocked_result.selection_to_runtime_contract_signal,
            BlueBrainRuntimeSelectionContractSignal::SelectionToRuntimeBlockedState
        );
        assert_eq!(
            blocked_result.runtime_to_selection_contract_diagnostic,
            BlueBrainRuntimeSelectionContractDiagnosticClass::BlockedContractDiagnostic
        );
        assert_eq!(
            blocked_result.selection_to_runtime_contract_diagnostic,
            BlueBrainRuntimeSelectionContractDiagnosticClass::BlockedContractDiagnostic
        );
        assert_eq!(
            blocked_result.runtime_to_selection_contract_reason,
            BlueBrainRuntimeSelectionContractReason::BlockedDueToContractBoundaryOrReferenceWeakness
        );
        assert_eq!(
            blocked_result.selection_to_runtime_contract_reason,
            BlueBrainRuntimeSelectionContractReason::BlockedDueToContractBoundaryOrReferenceWeakness
        );
        assert_eq!(
            blocked_result.selection_boundary_state,
            BlueBrainPriorityDeferredBlockedBoundaryState::BlockedContractState
        );

        let mut insufficient = base_input(BlueBrainKuramotoScopeState::DiagnosticOnly);
        insufficient.phase_nodes.truncate(1);
        let insufficient_result = evaluate_blue_brain_kuramoto_modulation(insufficient);
        assert_eq!(
            insufficient_result.runtime_to_selection_contract_signal,
            BlueBrainRuntimeSelectionContractSignal::InsufficientContractBasis
        );
        assert_eq!(
            insufficient_result.selection_to_runtime_contract_signal,
            BlueBrainRuntimeSelectionContractSignal::InsufficientContractBasis
        );
        assert_eq!(
            insufficient_result.runtime_to_selection_contract_diagnostic,
            BlueBrainRuntimeSelectionContractDiagnosticClass::InsufficientContractDiagnostic
        );
        assert_eq!(
            insufficient_result.selection_to_runtime_contract_diagnostic,
            BlueBrainRuntimeSelectionContractDiagnosticClass::InsufficientContractDiagnostic
        );
        assert_eq!(
            insufficient_result.selection_boundary_state,
            BlueBrainPriorityDeferredBlockedBoundaryState::InsufficientContractBasis
        );

        let mut deferred = base_input(BlueBrainKuramotoScopeState::SelectionModulating);
        deferred.phase_nodes[0].phase_permille = 0;
        deferred.phase_nodes[1].phase_permille = 500;
        let deferred_result = evaluate_blue_brain_kuramoto_modulation(deferred);
        assert_eq!(
            deferred_result.runtime_to_selection_contract_signal,
            BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionDeferredSignal
        );
        assert_eq!(
            deferred_result.runtime_to_selection_contract_diagnostic,
            BlueBrainRuntimeSelectionContractDiagnosticClass::DeferredContractDiagnostic
        );
        assert_eq!(
            deferred_result.runtime_boundary_state,
            BlueBrainPriorityDeferredBlockedBoundaryState::DeferredContractState
        );
    }

    #[test]
    fn failed_cancelled_blocked_unavailable_basis_stays_strictly_separated() {
        let mut failed = base_input(BlueBrainKuramotoScopeState::DiagnosticOnly);
        failed
            .failed_execution_result_refs
            .push("bb14:minimal_execution:h1:emit_canonical_signal:result:failed".to_string());
        let failed_result = evaluate_blue_brain_kuramoto_modulation(failed);
        assert_eq!(
            failed_result.execution_feedback_state,
            BlueBrainDynamicsExecutionFeedbackState::FailedExecutionFeedbackBasis
        );
        assert!(failed_result
            .caveats
            .iter()
            .any(|item| item == "caveated_execution_feedback_basis"));

        let mut cancelled = base_input(BlueBrainKuramotoScopeState::DiagnosticOnly);
        cancelled
            .cancelled_execution_result_refs
            .push("bb14:minimal_execution:h1:emit_canonical_signal:result:cancelled".to_string());
        let cancelled_result = evaluate_blue_brain_kuramoto_modulation(cancelled);
        assert_eq!(
            cancelled_result.execution_feedback_state,
            BlueBrainDynamicsExecutionFeedbackState::CancelledExecutionFeedbackBasis
        );
        assert!(cancelled_result
            .caveats
            .iter()
            .any(|item| item == "caveated_execution_feedback_basis"));

        let mut blocked = base_input(BlueBrainKuramotoScopeState::DiagnosticOnly);
        blocked.blocked_execution_result_refs.push(
            "bb14:minimal_execution:h1:emit_canonical_signal:result:ExecutionBlocked".to_string(),
        );
        let blocked_result = evaluate_blue_brain_kuramoto_modulation(blocked);
        assert_eq!(
            blocked_result.execution_feedback_state,
            BlueBrainDynamicsExecutionFeedbackState::BlockedDynamicsFeedbackBasis
        );
        assert!(blocked_result
            .caveats
            .iter()
            .any(|item| item == "blocked_dynamics_feedback_basis"));

        let mut unavailable = base_input(BlueBrainKuramotoScopeState::DiagnosticOnly);
        unavailable.unavailable_execution_result_refs.push(
            "bb14:minimal_execution:h1:emit_canonical_signal:result:ExecutionUnavailable"
                .to_string(),
        );
        let unavailable_result = evaluate_blue_brain_kuramoto_modulation(unavailable);
        assert_eq!(
            unavailable_result.execution_feedback_state,
            BlueBrainDynamicsExecutionFeedbackState::UnavailableDynamicsFeedbackBasis
        );
        assert!(unavailable_result
            .caveats
            .iter()
            .any(|item| item == "unavailable_dynamics_feedback_basis"));
    }

    #[test]
    fn diagnostic_only_and_insufficient_feedback_basis_stay_explicit() {
        let mut diagnostic_only = base_input(BlueBrainKuramotoScopeState::DiagnosticOnly);
        diagnostic_only.selected_context_refs.clear();
        diagnostic_only.selected_evidence_refs.clear();
        diagnostic_only
            .diagnostic_only_feedback_refs
            .push("diag:execution_feedback".to_string());
        let diagnostic_result = evaluate_blue_brain_kuramoto_modulation(diagnostic_only);
        assert_eq!(
            diagnostic_result.execution_feedback_state,
            BlueBrainDynamicsExecutionFeedbackState::DiagnosticOnlyDynamicsFeedback
        );

        let mut insufficient = base_input(BlueBrainKuramotoScopeState::DiagnosticOnly);
        insufficient.selected_context_refs.clear();
        insufficient.selected_evidence_refs.clear();
        let insufficient_result = evaluate_blue_brain_kuramoto_modulation(insufficient);
        assert_eq!(
            insufficient_result.execution_feedback_state,
            BlueBrainDynamicsExecutionFeedbackState::InsufficientDynamicsFeedbackBasis
        );
    }

    fn base_hh_input(
        scope: BlueBrainHodgkinHuxleyScopeState,
    ) -> BlueBrainHodgkinHuxleyDiagnosticInput {
        BlueBrainHodgkinHuxleyDiagnosticInput {
            scope,
            diagnostic_run_id: "run-hh-1".to_string(),
            context_refs: vec!["ctx:bb2".to_string(), "ctx:bb2".to_string()],
            evidence_refs: vec!["ev:alpha".to_string(), "ev:alpha".to_string()],
            simulation_parameters: BlueBrainHodgkinHuxleySimulationParameters {
                integration_steps: 512,
                dt_micros: 50,
                stimulus_nanoamp_permille: 800,
            },
            model_parameters: BlueBrainHodgkinHuxleyBoundedModelParameters {
                sodium_conductance_permille: 1200,
                potassium_conductance_permille: 1200,
                leak_conductance_permille: 250,
            },
        }
    }

    #[test]
    fn hh_scope_states_are_distinguishable_for_diagnostic_path() {
        let scopes = [
            BlueBrainHodgkinHuxleyScopeState::SimulationOnly,
            BlueBrainHodgkinHuxleyScopeState::DiagnosticOnly,
            BlueBrainHodgkinHuxleyScopeState::ResearchDeferred,
            BlueBrainHodgkinHuxleyScopeState::NotSuitableForCurrentBlueBrainRuntime,
            BlueBrainHodgkinHuxleyScopeState::NonCanonicalInternalOnly,
        ];
        assert_eq!(scopes.len(), 5);
        assert_ne!(scopes[0], scopes[1]);
        assert_ne!(scopes[1], scopes[2]);
        assert_ne!(scopes[2], scopes[3]);
        assert_ne!(scopes[3], scopes[4]);
    }

    #[test]
    fn hh_diagnostic_surface_stays_bounded_and_returns_trace_reference() {
        let result = evaluate_blue_brain_hodgkin_huxley_diagnostic(base_hh_input(
            BlueBrainHodgkinHuxleyScopeState::DiagnosticOnly,
        ));
        assert_eq!(
            result.diagnostic_class,
            BlueBrainHodgkinHuxleyDiagnosticClass::SimulationDiagnosticSummary
        );
        assert_eq!(result.trace_ref.as_deref(), Some("diag:hh:run-hh-1"));
        assert!(result
            .bounded_metadata
            .iter()
            .any(|(k, _)| k == "effective_stability_permille"));
    }

    #[test]
    fn hh_diagnostics_fail_when_run_identity_is_missing() {
        let mut input = base_hh_input(BlueBrainHodgkinHuxleyScopeState::DiagnosticOnly);
        input.diagnostic_run_id.clear();
        let result = evaluate_blue_brain_hodgkin_huxley_diagnostic(input);
        assert_eq!(
            result.diagnostic_class,
            BlueBrainHodgkinHuxleyDiagnosticClass::FailedOrInsufficientDiagnostic
        );
        assert!(result
            .caveats
            .iter()
            .any(|item| item == "missing_diagnostic_run_id"));
        assert!(result.trace_ref.is_none());
    }

    #[test]
    fn hh_diagnostics_do_not_expose_runtime_selection_or_execution_authority() {
        let result = evaluate_blue_brain_hodgkin_huxley_diagnostic(base_hh_input(
            BlueBrainHodgkinHuxleyScopeState::SimulationOnly,
        ));
        assert!(!result.boundary_guard.runtime_state_mutation_allowed);
        assert!(!result.boundary_guard.selection_mutation_allowed);
        assert!(!result.boundary_guard.memory_mutation_allowed);
        assert!(!result.boundary_guard.memory_commit_allowed);
        assert!(!result.boundary_guard.action_execution_allowed);
        assert!(!result.boundary_guard.tool_invocation_allowed);
        assert!(!result.boundary_guard.actual_action_result_allowed);
        assert!(!result.boundary_guard.compute_invocation_allowed);
        assert!(!result.boundary_guard.safety_override_allowed);
        assert!(!result.boundary_guard.policy_decision_allowed);
    }

    #[test]
    fn diagnostic_only_kuramoto_provides_bounded_feedback_without_authority() {
        let result = evaluate_blue_brain_kuramoto_modulation(base_input(
            BlueBrainKuramotoScopeState::DiagnosticOnly,
        ));
        assert!(result.selection_hint.is_some());
        assert!(result.runtime_modulation.is_some());
        assert_eq!(
            result.dynamics_diagnostic_class,
            BlueBrainDynamicsDiagnosticClass::KuramotoModulationDiagnostic
        );
        assert!(!result.boundary_guard.action_execution_allowed);
        assert!(!result.boundary_guard.memory_commit_allowed);
        assert!(!result.boundary_guard.compute_invocation_allowed);
        assert!(!result.boundary_guard.safety_override_allowed);
        assert!(!result.boundary_guard.policy_decision_allowed);
    }

    #[test]
    fn canonical_dynamics_diagnostics_map_lanes_are_unique_and_non_empty() {
        let mut lanes: Vec<&str> = CANONICAL_BLUE_BRAIN_DYNAMICS_DIAGNOSTICS_MAP
            .iter()
            .map(|lane| lane.lane)
            .collect();
        lanes.sort_unstable();
        lanes.dedup();
        assert_eq!(
            lanes.len(),
            CANONICAL_BLUE_BRAIN_DYNAMICS_DIAGNOSTICS_MAP.len()
        );
        assert!(CANONICAL_BLUE_BRAIN_DYNAMICS_DIAGNOSTICS_MAP
            .iter()
            .all(|lane| !lane.canonical_guard.trim().is_empty()));
    }

    #[test]
    fn runtime_selection_contract_map_covers_canonical_signals_and_tokens() {
        let mut lanes: Vec<&str> = CANONICAL_BLUE_BRAIN_RUNTIME_SELECTION_CONTRACT_MAP
            .iter()
            .map(|lane| lane.lane)
            .collect();
        lanes.sort_unstable();
        lanes.dedup();
        assert_eq!(
            lanes.len(),
            CANONICAL_BLUE_BRAIN_RUNTIME_SELECTION_CONTRACT_MAP.len()
        );
        assert!(CANONICAL_BLUE_BRAIN_RUNTIME_SELECTION_CONTRACT_MAP
            .iter()
            .all(|lane| !lane.canonical_guard.trim().is_empty()));
        assert_eq!(
            runtime_selection_contract_signal_token(
                BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionAdvisorySignal
            ),
            "runtime_to_selection_advisory_signal"
        );
        assert_eq!(
            runtime_selection_contract_signal_token(
                BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionDeferredSignal
            ),
            "runtime_to_selection_deferred_signal"
        );
        assert_eq!(
            runtime_selection_contract_signal_token(
                BlueBrainRuntimeSelectionContractSignal::RuntimeToSelectionBlockedSignal
            ),
            "runtime_to_selection_blocked_signal"
        );
        assert_eq!(
            runtime_selection_contract_signal_token(
                BlueBrainRuntimeSelectionContractSignal::SelectionToRuntimeBlockedState
            ),
            "selection_to_runtime_blocked_state"
        );
        assert_eq!(
            runtime_selection_contract_signal_token(
                BlueBrainRuntimeSelectionContractSignal::CaveatedContractSignal
            ),
            "caveated_contract_signal"
        );
    }

    #[test]
    fn runtime_selection_contract_diagnostics_map_covers_canonical_diagnostics_and_reasons() {
        let mut lanes: Vec<&str> = CANONICAL_BLUE_BRAIN_RUNTIME_SELECTION_CONTRACT_DIAGNOSTICS_MAP
            .iter()
            .map(|lane| lane.lane)
            .collect();
        lanes.sort_unstable();
        lanes.dedup();
        assert_eq!(
            lanes.len(),
            CANONICAL_BLUE_BRAIN_RUNTIME_SELECTION_CONTRACT_DIAGNOSTICS_MAP.len()
        );
        assert!(
            CANONICAL_BLUE_BRAIN_RUNTIME_SELECTION_CONTRACT_DIAGNOSTICS_MAP
                .iter()
                .all(|lane| !lane.canonical_guard.trim().is_empty())
        );

        assert_eq!(
            runtime_selection_contract_diagnostic_class_token(
                BlueBrainRuntimeSelectionContractDiagnosticClass::DeferredContractDiagnostic
            ),
            "deferred_contract_diagnostic"
        );
        assert_eq!(
            runtime_selection_contract_diagnostic_class_token(
                BlueBrainRuntimeSelectionContractDiagnosticClass::AdvisoryOnlyContractDiagnostic
            ),
            "advisory_only_contract_diagnostic"
        );
        assert_eq!(
            runtime_selection_contract_reason_token(
                BlueBrainRuntimeSelectionContractReason::DeferredDueToBoundedPrioritySelectionState
            ),
            "deferred_due_to_bounded_priority_selection_state"
        );
        assert_eq!(
            runtime_selection_contract_reason_token(
                BlueBrainRuntimeSelectionContractReason::PriorityAdvisoryHintOnlyNoDirectSelectionAuthority
            ),
            "priority_advisory_hint_only_no_direct_selection_authority"
        );
    }

    #[test]
    fn priority_deferred_blocked_boundary_map_is_canonical_and_distinguishable() {
        let mut lanes: Vec<&str> = CANONICAL_BLUE_BRAIN_PRIORITY_DEFERRED_BLOCKED_BOUNDARY_MAP
            .iter()
            .map(|lane| lane.lane)
            .collect();
        lanes.sort_unstable();
        lanes.dedup();
        assert_eq!(
            lanes.len(),
            CANONICAL_BLUE_BRAIN_PRIORITY_DEFERRED_BLOCKED_BOUNDARY_MAP.len()
        );
        assert!(CANONICAL_BLUE_BRAIN_PRIORITY_DEFERRED_BLOCKED_BOUNDARY_MAP
            .iter()
            .all(|lane| !lane.canonical_guard.trim().is_empty()));
        assert_eq!(
            priority_deferred_blocked_boundary_state_token(
                BlueBrainPriorityDeferredBlockedBoundaryState::PriorityAdvisoryHint
            ),
            "priority_advisory_hint"
        );
        assert_eq!(
            priority_deferred_blocked_boundary_state_token(
                BlueBrainPriorityDeferredBlockedBoundaryState::DeferredContractState
            ),
            "deferred_contract_state"
        );
        assert_eq!(
            priority_deferred_blocked_boundary_state_token(
                BlueBrainPriorityDeferredBlockedBoundaryState::BlockedContractState
            ),
            "blocked_contract_state"
        );
    }
    #[test]
    fn hh_internal_only_scope_is_marked_non_canonical() {
        let result = evaluate_blue_brain_hodgkin_huxley_diagnostic(base_hh_input(
            BlueBrainHodgkinHuxleyScopeState::NonCanonicalInternalOnly,
        ));
        assert_eq!(
            result.dynamics_diagnostic_class,
            BlueBrainDynamicsDiagnosticClass::NonCanonicalInternalOnlyDynamicsDiagnostic
        );
        assert_eq!(
            result.runtime_feedback,
            BlueBrainDynamicsRuntimeFeedbackClass::DynamicsIgnoredForCurrentTransition
        );
        assert_eq!(
            result.selection_feedback,
            BlueBrainDynamicsSelectionFeedbackClass::DynamicsIgnoredForCurrentSelection
        );
    }

    #[test]
    fn serie_bb16_prompt1_doc_stays_pinned_to_feedback_states_and_no_direct_boundaries() {
        let doc = include_str!(
            "../../../docs/blue_brain_bounded_dynamics_execution_feedback_line_serie_bb16_prompt1_v1.md"
        );
        assert!(doc.contains("execution_informed_dynamics_input"));
        assert!(doc.contains("reference_informed_dynamics_input"));
        assert!(doc.contains("caveated_execution_informed_dynamics_input"));
        assert!(doc.contains("insufficient_dynamics_feedback_basis"));
        assert!(doc.contains("blocked_dynamics_feedback_basis"));
        assert!(doc.contains("unavailable_dynamics_feedback_basis"));
        assert!(doc.contains("diagnostic_only_dynamics_feedback"));
        assert!(doc.contains("non_canonical_internal_only_feedback_path"));
        assert!(doc.contains("kein direct re-execute"));
        assert!(doc.contains("kein direct retry orchestration"));
        assert!(doc.contains("kein direct re-execute trigger"));
        assert!(doc.contains("kein direct action selection"));
        assert!(doc.contains("kein direct memory commit"));
        assert!(doc.contains("kein direct compute invocation"));
        assert!(doc.contains("kein safety override"));
    }

    #[test]
    fn serie_bb16_prompt2_doc_stays_pinned_to_hardened_execution_informed_diagnostics_line() {
        let doc = include_str!(
            "../../../docs/blue_brain_execution_informed_dynamics_diagnostics_hardening_serie_bb16_prompt2_v1.md"
        );
        assert!(doc.contains("execution_informed_dynamics_input"));
        assert!(doc.contains("reference_informed_dynamics_input"));
        assert!(doc.contains("caveated_execution_informed_dynamics_input"));
        assert!(doc.contains("insufficient_dynamics_feedback_basis"));
        assert!(doc.contains("blocked_dynamics_feedback_basis"));
        assert!(doc.contains("unavailable_dynamics_feedback_basis"));
        assert!(doc.contains("diagnostic_only_dynamics_feedback"));
        assert!(doc.contains("non_canonical_internal_only_feedback_path"));
        assert!(doc.contains("no_direct_reexecute_allowed"));
        assert!(doc.contains("no_direct_retry_orchestration_allowed"));
    }

    #[test]
    fn serie_bb16_prompt3_doc_stays_pinned_to_selection_runtime_coupling_boundary() {
        let doc = include_str!(
            "../../../docs/blue_brain_selection_runtime_coupling_boundary_hardening_serie_bb16_prompt3_v1.md"
        );
        assert!(doc.contains("runtime_advisory_coupling"));
        assert!(doc.contains("selection_advisory_coupling"));
        assert!(doc.contains("caveated_advisory_coupling"));
        assert!(doc.contains("insufficient_advisory_coupling"));
        assert!(doc.contains("blocked_advisory_coupling"));
        assert!(doc.contains("ignored_advisory_coupling"));
        assert!(doc.contains("non_canonical_internal_only_coupling_path"));
        assert!(doc.contains("kein direct action selection"));
        assert!(doc.contains("kein direct retry trigger"));
        assert!(doc.contains("kein direct compute invocation"));
        assert!(doc.contains("kein direct memory commit"));
    }

    #[test]
    fn serie_bb16_prompt4_doc_stays_pinned_to_closure_matrix_and_boundaries() {
        let doc = include_str!(
            "../../../docs/blue_brain_bb16_readiness_sweep_bounded_dynamics_execution_line_serie_bb16_prompt4_v1.md"
        );
        assert!(doc.contains("stable bounded dynamics ↔ execution line"));
        assert!(doc.contains("usable with caveats"));
        assert!(doc.contains("advisory-only"));
        assert!(doc.contains("blocked/insufficient"));
        assert!(doc.contains("deferred/non-canonical"));

        assert!(doc.contains("execution_informed_dynamics_input"));
        assert!(doc.contains("reference_informed_dynamics_input"));
        assert!(doc.contains("caveated_execution_informed_dynamics_input"));
        assert!(doc.contains("insufficient_dynamics_feedback_basis"));
        assert!(doc.contains("blocked_dynamics_feedback_basis"));
        assert!(doc.contains("unavailable_dynamics_feedback_basis"));
        assert!(doc.contains("diagnostic_only_dynamics_feedback"));
        assert!(doc.contains("non_canonical_internal_only_feedback_path"));

        assert!(doc.contains("runtime_advisory_coupling"));
        assert!(doc.contains("selection_advisory_coupling"));
        assert!(doc.contains("caveated_advisory_coupling"));
        assert!(doc.contains("insufficient_advisory_coupling"));
        assert!(doc.contains("blocked_advisory_coupling"));
        assert!(doc.contains("ignored_advisory_coupling"));
        assert!(doc.contains("non_canonical_internal_only_coupling_path"));

        assert!(doc.contains("keine direkte Action-Execution"));
        assert!(doc.contains("keine direkte Retry-Orchestrierung"));
        assert!(doc.contains("keine Policy-/Governance-Entscheidungsautorität"));
        assert!(doc.contains("keine automatische Compute-Invocation"));
        assert!(doc.contains("keine automatische Memory-Persistenz"));
        assert!(doc.contains("keine Safety-Override-Semantik"));
        assert!(doc.contains("maintenance-only"));
        assert!(doc.contains("Priorität 1: BB17 context/memory/reference hardening follow-up"));
    }

    #[test]
    fn serie_bb19_prompt1_doc_stays_pinned_to_runtime_selection_contract_hardening() {
        let doc = include_str!(
            "../../../docs/blue_brain_runtime_selection_contract_hardening_serie_bb19_prompt1_v1.md"
        );
        assert!(doc.contains("runtime_to_selection_advisory_signal"));
        assert!(doc.contains("runtime_to_selection_blocked_or_deferral_signal"));
        assert!(doc.contains("selection_to_runtime_advisory_state"));
        assert!(doc.contains("selection_to_runtime_deferred_state"));
        assert!(doc.contains("caveated_contract_signal"));
        assert!(doc.contains("insufficient_contract_basis"));
        assert!(doc.contains("non_canonical_internal_only_contract_path"));
        assert!(doc.contains("deferred ist nicht blocked"));
        assert!(doc.contains("blocked ist nicht failed execution"));
        assert!(doc.contains("insufficient ist nicht blocked"));
        assert!(doc.contains("advisory-only bleibt advisory-only"));
        assert!(doc.contains("keine direkte Action-Execution"));
        assert!(doc.contains("keine direkte Retry-Orchestrierung"));
        assert!(doc.contains("keine automatische Compute-Invocation"));
        assert!(doc.contains("keine automatische Memory-Persistenz"));
    }

    #[test]
    fn serie_bb19_prompt2_doc_stays_pinned_to_runtime_selection_diagnostics_hardening() {
        let doc = include_str!(
            "../../../docs/blue_brain_runtime_selection_diagnostics_hardening_serie_bb19_prompt2_v1.md"
        );
        assert!(doc.contains("runtime_to_selection_contract_diagnostic"));
        assert!(doc.contains("selection_to_runtime_contract_diagnostic"));
        assert!(doc.contains("deferred_contract_diagnostic"));
        assert!(doc.contains("blocked_contract_diagnostic"));
        assert!(doc.contains("caveated_contract_diagnostic"));
        assert!(doc.contains("insufficient_contract_diagnostic"));
        assert!(doc.contains("advisory_only_contract_diagnostic"));
        assert!(doc.contains("non_canonical_internal_only_contract_diagnostic"));
        assert!(doc.contains("deferred_due_to_bounded_priority_selection_state"));
        assert!(doc.contains("blocked_due_to_contract_boundary_or_reference_weakness"));
        assert!(doc.contains("caveated_due_to_weak_or_partial_reference_dynamics_execution_basis"));
        assert!(doc.contains("insufficient_due_to_missing_bounded_contract_basis"));
        assert!(doc.contains("advisory_only_no_direct_action_authority"));
        assert!(doc.contains("no_direct_action_execution"));
        assert!(doc.contains("no_direct_retry_orchestration"));
        assert!(doc.contains("no_direct_compute_invocation"));
        assert!(doc.contains("no_implicit_memory_persistence"));
    }

    #[test]
    fn serie_bb19_prompt3_doc_stays_pinned_to_deferred_blocked_priority_boundary_cleanup() {
        let doc = include_str!(
            "../../../docs/blue_brain_runtime_selection_deferred_blocked_priority_boundary_cleanup_serie_bb19_prompt3_v1.md"
        );
        assert!(doc.contains("priority_advisory_hint"));
        assert!(doc.contains("deferred_contract_state"));
        assert!(doc.contains("blocked_contract_state"));
        assert!(doc.contains("caveated_priority_deferred_blocked_signal"));
        assert!(doc.contains("insufficient_contract_basis_boundary_state"));
        assert!(doc.contains("non_canonical_internal_only_coupling_path_boundary_state"));
        assert!(doc.contains("runtime_to_selection_deferred_signal"));
        assert!(doc.contains("runtime_to_selection_blocked_signal"));
        assert!(doc.contains("selection_to_runtime_blocked_state"));
        assert!(doc.contains("priority_advisory_hint_only_no_direct_selection_authority"));
        assert!(doc.contains("kein direct action execution"));
        assert!(doc.contains("kein direct retry orchestration"));
        assert!(doc.contains("kein direct compute invocation"));
        assert!(doc.contains("keine implizite memory persistenz"));
    }

    #[test]
    fn serie_bb19_prompt4_doc_stays_pinned_to_readiness_sweep_contract_line() {
        let doc = include_str!(
            "../../../docs/blue_brain_bb19_readiness_sweep_runtime_selection_contract_line_serie_bb19_prompt4_v1.md"
        );
        assert!(doc.contains("stable runtime/selection contract line"));
        assert!(doc.contains("usable with caveats"));
        assert!(doc.contains("advisory-only"));
        assert!(doc.contains("blocked/insufficient/deferred"));
        assert!(doc.contains("non-canonical/internal-only"));

        assert!(doc.contains("runtime_to_selection_advisory_signal"));
        assert!(doc.contains("runtime_to_selection_deferred_signal"));
        assert!(doc.contains("runtime_to_selection_blocked_signal"));
        assert!(doc.contains("selection_to_runtime_advisory_state"));
        assert!(doc.contains("selection_to_runtime_deferred_state"));
        assert!(doc.contains("selection_to_runtime_blocked_state"));
        assert!(doc.contains("runtime_to_selection_contract_diagnostic"));
        assert!(doc.contains("selection_to_runtime_contract_diagnostic"));
        assert!(doc.contains("deferred_contract_diagnostic"));
        assert!(doc.contains("blocked_contract_diagnostic"));
        assert!(doc.contains("caveated_contract_diagnostic"));
        assert!(doc.contains("insufficient_contract_diagnostic"));
        assert!(doc.contains("advisory_only_contract_diagnostic"));
        assert!(doc.contains("non_canonical_internal_only_contract_diagnostic"));
        assert!(doc.contains("priority_advisory_hint"));
        assert!(doc.contains("deferred_contract_state"));
        assert!(doc.contains("blocked_contract_state"));

        assert!(doc.contains("deferred` bleibt bounded Aufschub und ist **nicht** `blocked"));
        assert!(doc.contains(
            "priority_advisory_hint` bleibt Hinweis-Semantik und wird **nicht** zur direkten Selection-Entscheidungsautorität"
        ));
        assert!(doc.contains("keine direkte Action-Execution"));
        assert!(doc.contains("keine direkte Retry-Orchestrierung"));
        assert!(doc.contains("keine automatische Compute-Invocation"));
        assert!(doc.contains("keine automatische Memory-Persistenz"));
        assert!(doc.contains("keine neue allowed-actions-Erweiterung"));
        assert!(doc.contains("maintenance-only Core"));
        assert!(doc.contains("Priorität 1: BB20 execution/reference interaction hardening."));
    }
}
