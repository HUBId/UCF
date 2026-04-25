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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainKuramotoModulationResult {
    pub dynamics_diagnostic_class: BlueBrainDynamicsDiagnosticClass,
    pub modulation_state: BlueBrainKuramotoModulationState,
    pub diagnostic: BlueBrainKuramotoSynchronyDiagnostic,
    pub coherence_permille: u16,
    pub selection_hint: Option<BlueBrainKuramotoSelectionHint>,
    pub runtime_modulation: Option<BlueBrainKuramotoRuntimeCaveatModulation>,
    pub runtime_feedback: BlueBrainDynamicsRuntimeFeedbackClass,
    pub selection_feedback: BlueBrainDynamicsSelectionFeedbackClass,
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

    if input.phase_nodes.len() < 2 {
        caveats.push("insufficient_dynamics_input".to_string());
        return BlueBrainKuramotoModulationResult {
            dynamics_diagnostic_class: BlueBrainDynamicsDiagnosticClass::DynamicsInsufficient,
            modulation_state: BlueBrainKuramotoModulationState::Insufficient,
            diagnostic: BlueBrainKuramotoSynchronyDiagnostic::InsufficientInput,
            coherence_permille: 0,
            selection_hint: None,
            runtime_modulation: None,
            runtime_feedback:
                BlueBrainDynamicsRuntimeFeedbackClass::DynamicsInsufficientForModulation,
            selection_feedback:
                BlueBrainDynamicsSelectionFeedbackClass::DynamicsInsufficientForModulation,
            caveats,
            boundary_guard: boundary_guard(),
        };
    }

    let coherence_permille = coherence_from_phase_nodes(&input.phase_nodes);
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

    let selection_feedback = match selection_hint {
        Some(BlueBrainKuramotoSelectionHint::KeepCurrentSelection) => {
            BlueBrainDynamicsSelectionFeedbackClass::SelectionModulationObserved
        }
        Some(_) => BlueBrainDynamicsSelectionFeedbackClass::DynamicsCaveatAttached,
        None => BlueBrainDynamicsSelectionFeedbackClass::DynamicsIgnoredForCurrentSelection,
    };
    let runtime_feedback = match runtime_modulation {
        Some(BlueBrainKuramotoRuntimeCaveatModulation::NoAdditionalCaveat) => {
            BlueBrainDynamicsRuntimeFeedbackClass::RuntimeModulationObserved
        }
        Some(_) => BlueBrainDynamicsRuntimeFeedbackClass::DynamicsCaveatAttached,
        None => BlueBrainDynamicsRuntimeFeedbackClass::DynamicsIgnoredForCurrentTransition,
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
        modulation_state,
        diagnostic,
        coherence_permille,
        selection_hint,
        runtime_modulation,
        runtime_feedback,
        selection_feedback,
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
            non_canonical_internal_only_path: false,
            phase_nodes: vec![
                BlueBrainKuramotoPhaseNodeInput {
                    group_ref: "g2".to_string(),
                    phase_permille: 120,
                    coupling_permille: 700,
                },
                BlueBrainKuramotoPhaseNodeInput {
                    group_ref: "g1".to_string(),
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

        let no_op = evaluate_blue_brain_kuramoto_modulation(base_input(
            BlueBrainKuramotoScopeState::DiagnosticOnly,
        ));
        assert_eq!(
            no_op.modulation_state,
            BlueBrainKuramotoModulationState::NoOp
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
}
