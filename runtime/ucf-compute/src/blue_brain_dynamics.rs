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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainHodgkinHuxleyDiagnosticResult {
    pub diagnostic_class: BlueBrainHodgkinHuxleyDiagnosticClass,
    pub caveats: Vec<String>,
    pub trace_ref: Option<String>,
    pub bounded_metadata: Vec<(String, String)>,
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
            diagnostic_class: BlueBrainHodgkinHuxleyDiagnosticClass::FailedOrInsufficientDiagnostic,
            caveats,
            trace_ref: None,
            bounded_metadata: vec![("output_surface".to_string(), "diagnostic_only".to_string())],
            boundary_guard: hh_boundary_guard(),
        };
    }

    let effective_stability_permille = compute_effective_stability_permille(&input);
    let diagnostic_class = if effective_stability_permille >= 700 {
        BlueBrainHodgkinHuxleyDiagnosticClass::SimulationDiagnosticSummary
    } else if effective_stability_permille >= 450 {
        caveats.push("hh_signal_caveated".to_string());
        BlueBrainHodgkinHuxleyDiagnosticClass::SimulationDiagnosticCaveated
    } else {
        caveats.push("hh_signal_insufficient".to_string());
        BlueBrainHodgkinHuxleyDiagnosticClass::FailedOrInsufficientDiagnostic
    };

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
        diagnostic_class,
        caveats,
        trace_ref,
        bounded_metadata,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainKuramotoBoundaryGuard {
    pub action_execution_allowed: bool,
    pub tool_invocation_allowed: bool,
    pub memory_commit_allowed: bool,
    pub compute_invocation_allowed: bool,
    pub safety_override_allowed: bool,
    pub policy_decision_allowed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainKuramotoModulationResult {
    pub diagnostic: BlueBrainKuramotoSynchronyDiagnostic,
    pub coherence_permille: u16,
    pub selection_hint: Option<BlueBrainKuramotoSelectionHint>,
    pub runtime_modulation: Option<BlueBrainKuramotoRuntimeCaveatModulation>,
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

    if input.phase_nodes.len() < 2 {
        caveats.push("insufficient_dynamics_input".to_string());
        return BlueBrainKuramotoModulationResult {
            diagnostic: BlueBrainKuramotoSynchronyDiagnostic::InsufficientInput,
            coherence_permille: 0,
            selection_hint: None,
            runtime_modulation: None,
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

    BlueBrainKuramotoModulationResult {
        diagnostic,
        coherence_permille,
        selection_hint,
        runtime_modulation,
        caveats,
        boundary_guard: boundary_guard(),
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
        action_execution_allowed: false,
        tool_invocation_allowed: false,
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

        let runtime_result = evaluate_blue_brain_kuramoto_modulation(base_input(
            BlueBrainKuramotoScopeState::RuntimeCaveatModulating,
        ));
        assert!(runtime_result.selection_hint.is_none());
        assert!(runtime_result.runtime_modulation.is_some());
    }

    #[test]
    fn modulation_path_cannot_trigger_execution_memory_compute_or_policy() {
        let result = evaluate_blue_brain_kuramoto_modulation(base_input(
            BlueBrainKuramotoScopeState::DiagnosticOnly,
        ));
        assert!(!result.boundary_guard.action_execution_allowed);
        assert!(!result.boundary_guard.tool_invocation_allowed);
        assert!(!result.boundary_guard.memory_commit_allowed);
        assert!(!result.boundary_guard.compute_invocation_allowed);
        assert!(!result.boundary_guard.safety_override_allowed);
        assert!(!result.boundary_guard.policy_decision_allowed);
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
        assert!(!result.boundary_guard.action_execution_allowed);
        assert!(!result.boundary_guard.tool_invocation_allowed);
        assert!(!result.boundary_guard.compute_invocation_allowed);
        assert!(!result.boundary_guard.safety_override_allowed);
        assert!(!result.boundary_guard.policy_decision_allowed);
    }
}
