use crate::{
    BlueBrainCandidateDeferralLifecycleClass, BlueBrainContextEvidencePriorityClass,
    BlueBrainControlAttentionSelectionClass, BlueBrainReferenceValidity,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFirstRegionClass {
    AttentionSelectionRelated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFirstRegionPathClass {
    RegionToRuntimeAdvisorySignal,
    RuntimeToRegionBoundedInput,
    RegionToSelectionAdvisorySignal,
    SelectionToRegionBoundedStateInput,
    RegionReferenceSignal,
    CaveatedDeferredBlockedRegionContractSignal,
    ReferenceOnlyRegionContractSignal,
    RegionInputSurface,
    RegionStateSurface,
    RegionOutputAdvisorySurface,
    RegionReferenceSurface,
    BlockedDeferredRegionPath,
    NonCanonicalInternalOnlyRegionPath,
    TestOnlyHelperNonOperationalPath,
}

pub const CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP: [BlueBrainFirstRegionPathClass; 14] = [
    BlueBrainFirstRegionPathClass::RegionToRuntimeAdvisorySignal,
    BlueBrainFirstRegionPathClass::RuntimeToRegionBoundedInput,
    BlueBrainFirstRegionPathClass::RegionToSelectionAdvisorySignal,
    BlueBrainFirstRegionPathClass::SelectionToRegionBoundedStateInput,
    BlueBrainFirstRegionPathClass::RegionReferenceSignal,
    BlueBrainFirstRegionPathClass::CaveatedDeferredBlockedRegionContractSignal,
    BlueBrainFirstRegionPathClass::ReferenceOnlyRegionContractSignal,
    BlueBrainFirstRegionPathClass::RegionInputSurface,
    BlueBrainFirstRegionPathClass::RegionStateSurface,
    BlueBrainFirstRegionPathClass::RegionOutputAdvisorySurface,
    BlueBrainFirstRegionPathClass::RegionReferenceSurface,
    BlueBrainFirstRegionPathClass::BlockedDeferredRegionPath,
    BlueBrainFirstRegionPathClass::NonCanonicalInternalOnlyRegionPath,
    BlueBrainFirstRegionPathClass::TestOnlyHelperNonOperationalPath,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFirstRegionHardeningClass {
    GuardedCanonicalRegionSurface,
    GuardedDiagnosticsPath,
    BlockedForbiddenAuthorityPath,
    NonCanonicalInternalOnlyRegionPath,
    TestOnlyHelperNonOperationalPath,
}

pub const CANONICAL_BLUE_BRAIN_FIRST_REGION_HARDENING_MAP: [BlueBrainFirstRegionHardeningClass; 5] = [
    BlueBrainFirstRegionHardeningClass::GuardedCanonicalRegionSurface,
    BlueBrainFirstRegionHardeningClass::GuardedDiagnosticsPath,
    BlueBrainFirstRegionHardeningClass::BlockedForbiddenAuthorityPath,
    BlueBrainFirstRegionHardeningClass::NonCanonicalInternalOnlyRegionPath,
    BlueBrainFirstRegionHardeningClass::TestOnlyHelperNonOperationalPath,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFirstRegionFinalizationClass {
    StableFirstRegionBaseline,
    UsableWithCaveatsFirstRegionSurface,
    AdvisoryOnlyFrozenRegionSignal,
    DiagnosticOnlyDeferredRegionState,
    SecondRegionNotOpenedYet,
    NonCanonicalInternalOnlyRegionPath,
}

pub const CANONICAL_BLUE_BRAIN_FIRST_REGION_FINALIZATION_MAP:
    [BlueBrainFirstRegionFinalizationClass; 6] = [
    BlueBrainFirstRegionFinalizationClass::StableFirstRegionBaseline,
    BlueBrainFirstRegionFinalizationClass::UsableWithCaveatsFirstRegionSurface,
    BlueBrainFirstRegionFinalizationClass::AdvisoryOnlyFrozenRegionSignal,
    BlueBrainFirstRegionFinalizationClass::DiagnosticOnlyDeferredRegionState,
    BlueBrainFirstRegionFinalizationClass::SecondRegionNotOpenedYet,
    BlueBrainFirstRegionFinalizationClass::NonCanonicalInternalOnlyRegionPath,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFirstRegionStabilizationClass {
    StableFirstRegionBaseline,
    MaintenanceHardenedRegionSurface,
    MaintenanceHardenedDiagnosticsPath,
    MaintenanceHardenedContractPath,
    NonCanonicalInternalOnlyResidualPath,
}

pub const CANONICAL_BLUE_BRAIN_FIRST_REGION_STABILIZATION_MAP:
    [BlueBrainFirstRegionStabilizationClass; 5] = [
    BlueBrainFirstRegionStabilizationClass::StableFirstRegionBaseline,
    BlueBrainFirstRegionStabilizationClass::MaintenanceHardenedRegionSurface,
    BlueBrainFirstRegionStabilizationClass::MaintenanceHardenedDiagnosticsPath,
    BlueBrainFirstRegionStabilizationClass::MaintenanceHardenedContractPath,
    BlueBrainFirstRegionStabilizationClass::NonCanonicalInternalOnlyResidualPath,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainSecondRegionExpansionState {
    NotOpenedYetExplicitRescopeRequired,
}

pub const BLUE_BRAIN_SECOND_REGION_EXPANSION_STATE: BlueBrainSecondRegionExpansionState =
    BlueBrainSecondRegionExpansionState::NotOpenedYetExplicitRescopeRequired;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainSecondRegionClass {
    MemoryContextRelated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainSecondRegionSelectionClass {
    SecondExpansionCandidate,
    ViableButNotSecond,
    LaterPhaseCandidate,
    SimulationOnlyDeferredCandidate,
    NonCanonicalInternalOnlyPath,
}

pub const CANONICAL_BLUE_BRAIN_SECOND_REGION_SELECTION_MAP: [BlueBrainSecondRegionSelectionClass;
    5] = [
    BlueBrainSecondRegionSelectionClass::SecondExpansionCandidate,
    BlueBrainSecondRegionSelectionClass::ViableButNotSecond,
    BlueBrainSecondRegionSelectionClass::LaterPhaseCandidate,
    BlueBrainSecondRegionSelectionClass::SimulationOnlyDeferredCandidate,
    BlueBrainSecondRegionSelectionClass::NonCanonicalInternalOnlyPath,
];

pub const BLUE_BRAIN_SECOND_REGION_CLASS_SELECTION: BlueBrainSecondRegionClass =
    BlueBrainSecondRegionClass::MemoryContextRelated;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainSecondRegionPathClass {
    RegionToRuntimeAdvisorySignal,
    RuntimeToRegionBoundedInput,
    RegionToSelectionAdvisorySignal,
    SelectionToRegionBoundedStateInput,
    RegionReferenceSignal,
    CaveatedDeferredBlockedRegionContractSignal,
    ReferenceOnlyRegionContractSignal,
    RegionInputSurface,
    RegionStateSurface,
    RegionOutputAdvisorySurface,
    RegionReferenceSurface,
    BlockedDeferredRegionPath,
    NonCanonicalInternalOnlyRegionPath,
}

pub const CANONICAL_BLUE_BRAIN_SECOND_REGION_INTEGRATION_MAP: [BlueBrainSecondRegionPathClass; 13] = [
    BlueBrainSecondRegionPathClass::RegionToRuntimeAdvisorySignal,
    BlueBrainSecondRegionPathClass::RuntimeToRegionBoundedInput,
    BlueBrainSecondRegionPathClass::RegionToSelectionAdvisorySignal,
    BlueBrainSecondRegionPathClass::SelectionToRegionBoundedStateInput,
    BlueBrainSecondRegionPathClass::RegionReferenceSignal,
    BlueBrainSecondRegionPathClass::CaveatedDeferredBlockedRegionContractSignal,
    BlueBrainSecondRegionPathClass::ReferenceOnlyRegionContractSignal,
    BlueBrainSecondRegionPathClass::RegionInputSurface,
    BlueBrainSecondRegionPathClass::RegionStateSurface,
    BlueBrainSecondRegionPathClass::RegionOutputAdvisorySurface,
    BlueBrainSecondRegionPathClass::RegionReferenceSurface,
    BlueBrainSecondRegionPathClass::BlockedDeferredRegionPath,
    BlueBrainSecondRegionPathClass::NonCanonicalInternalOnlyRegionPath,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainInterRegionRelationClass {
    Region1ToRegion2Bounded,
    Region2ToRegion1Bounded,
    SharedReferenceMediated,
    CaveatedInterRegion,
    BlockedDeferredInterRegion,
    NonCanonicalInternalOnlyPath,
}

pub const CANONICAL_BLUE_BRAIN_INTER_REGION_RELATION_MAP: [BlueBrainInterRegionRelationClass; 6] = [
    BlueBrainInterRegionRelationClass::Region1ToRegion2Bounded,
    BlueBrainInterRegionRelationClass::Region2ToRegion1Bounded,
    BlueBrainInterRegionRelationClass::SharedReferenceMediated,
    BlueBrainInterRegionRelationClass::CaveatedInterRegion,
    BlueBrainInterRegionRelationClass::BlockedDeferredInterRegion,
    BlueBrainInterRegionRelationClass::NonCanonicalInternalOnlyPath,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainInterRegionRelationSurface {
    pub relation_class: BlueBrainInterRegionRelationClass,
    pub region1_to_region2_advisory_only: bool,
    pub region2_to_region1_advisory_only: bool,
    pub reference_mediated_only: bool,
    pub caveated: bool,
    pub deferred: bool,
    pub blocked: bool,
    pub direct_action_selection: bool,
    pub direct_execution_trigger: bool,
    pub direct_retry_trigger: bool,
    pub direct_memory_commit: bool,
    pub direct_compute_invocation: bool,
    pub safety_override: bool,
}

pub fn evaluate_blue_brain_inter_region_relation(
    region1: BlueBrainFirstRegionOutputSurface,
    region2: BlueBrainSecondRegionOutputSurface,
) -> BlueBrainInterRegionRelationSurface {
    let non_canonical = matches!(
        region1.contract_signal,
        BlueBrainFirstRegionContractSignal::NonCanonicalInternalOnly
    ) || matches!(
        region2.runtime_contract_signal,
        BlueBrainSecondRegionContractSignal::NonCanonicalInternalOnly
    ) || matches!(
        region2.selection_contract_signal,
        BlueBrainSecondRegionContractSignal::NonCanonicalInternalOnly
    ) || matches!(
        region2.reference_contract_signal,
        BlueBrainSecondRegionContractSignal::NonCanonicalInternalOnly
    );

    let blocked = matches!(
        region1.contract_signal,
        BlueBrainFirstRegionContractSignal::Blocked
    ) || matches!(
        region2.runtime_contract_signal,
        BlueBrainSecondRegionContractSignal::Blocked
    ) || matches!(
        region2.selection_contract_signal,
        BlueBrainSecondRegionContractSignal::Blocked
    );

    let deferred = matches!(
        region1.contract_signal,
        BlueBrainFirstRegionContractSignal::Deferred
    ) || matches!(
        region2.runtime_contract_signal,
        BlueBrainSecondRegionContractSignal::Deferred
    ) || matches!(
        region2.selection_contract_signal,
        BlueBrainSecondRegionContractSignal::Deferred
    );

    let caveated = matches!(
        region1.contract_signal,
        BlueBrainFirstRegionContractSignal::Caveated
            | BlueBrainFirstRegionContractSignal::Insufficient
    ) || matches!(
        region2.runtime_contract_signal,
        BlueBrainSecondRegionContractSignal::Caveated
            | BlueBrainSecondRegionContractSignal::Insufficient
    ) || matches!(
        region2.selection_contract_signal,
        BlueBrainSecondRegionContractSignal::Caveated
            | BlueBrainSecondRegionContractSignal::Insufficient
    ) || matches!(
        region2.reference_contract_signal,
        BlueBrainSecondRegionContractSignal::Caveated
            | BlueBrainSecondRegionContractSignal::Insufficient
    );

    let relation_class = if non_canonical {
        BlueBrainInterRegionRelationClass::NonCanonicalInternalOnlyPath
    } else if blocked || deferred {
        BlueBrainInterRegionRelationClass::BlockedDeferredInterRegion
    } else if caveated {
        BlueBrainInterRegionRelationClass::CaveatedInterRegion
    } else if matches!(
        region2.reference_contract_signal,
        BlueBrainSecondRegionContractSignal::ReferenceOnly
            | BlueBrainSecondRegionContractSignal::RegionReferenceSignal
    ) || region1.reference_only
    {
        BlueBrainInterRegionRelationClass::SharedReferenceMediated
    } else if matches!(
        region1.contract_signal,
        BlueBrainFirstRegionContractSignal::RegionToRuntimeAdvisory
            | BlueBrainFirstRegionContractSignal::RegionToSelectionAdvisory
    ) {
        BlueBrainInterRegionRelationClass::Region1ToRegion2Bounded
    } else {
        BlueBrainInterRegionRelationClass::Region2ToRegion1Bounded
    };

    BlueBrainInterRegionRelationSurface {
        relation_class,
        region1_to_region2_advisory_only: region1.runtime_advisory_only
            && region1.selection_advisory_only,
        region2_to_region1_advisory_only: region2.runtime_advisory_only
            && region2.selection_advisory_only,
        reference_mediated_only: region1.reference_bounded_only && region2.reference_bounded_only,
        caveated,
        deferred,
        blocked,
        direct_action_selection: false,
        direct_execution_trigger: false,
        direct_retry_trigger: false,
        direct_memory_commit: false,
        direct_compute_invocation: false,
        safety_override: false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainSecondRegionStateSurface {
    ActiveBoundedAdvisoryOnly,
    CaveatedReferenceState,
    DeferredState,
    BlockedState,
    ReferenceOnlyState,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainSecondRegionContractSignal {
    RegionToRuntimeAdvisory,
    RuntimeToRegionBoundedInput,
    RegionToSelectionAdvisory,
    SelectionToRegionBoundedStateInput,
    RegionReferenceSignal,
    Caveated,
    Insufficient,
    Deferred,
    Blocked,
    ReferenceOnly,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainSecondRegionDiagnosticState {
    Region2AdvisoryOnlyDiagnostic,
    Region2CaveatedDiagnostic,
    Region2DeferredDiagnostic,
    Region2BlockedDiagnostic,
    Region2InsufficientDiagnostic,
    Region2DiagnosticOnlyState,
    CaveatedInterRegionDiagnosticInfluence,
    NonCanonicalInternalOnlyRegion2DiagnosticPath,
}

pub const CANONICAL_BLUE_BRAIN_SECOND_REGION_DIAGNOSTIC_MAP:
    [BlueBrainSecondRegionDiagnosticState; 8] = [
    BlueBrainSecondRegionDiagnosticState::Region2AdvisoryOnlyDiagnostic,
    BlueBrainSecondRegionDiagnosticState::Region2CaveatedDiagnostic,
    BlueBrainSecondRegionDiagnosticState::Region2DeferredDiagnostic,
    BlueBrainSecondRegionDiagnosticState::Region2BlockedDiagnostic,
    BlueBrainSecondRegionDiagnosticState::Region2InsufficientDiagnostic,
    BlueBrainSecondRegionDiagnosticState::Region2DiagnosticOnlyState,
    BlueBrainSecondRegionDiagnosticState::CaveatedInterRegionDiagnosticInfluence,
    BlueBrainSecondRegionDiagnosticState::NonCanonicalInternalOnlyRegion2DiagnosticPath,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainSecondRegionInputSource {
    RuntimeDeferralSignal,
    ContextReferenceSignal,
    ContextEvidencePrioritySignal,
    ToolActionControlSignal,
    ComputeInternalStateSignal,
    SafetyOverrideSignal,
    ImplicitMemoryMutationSignal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainSecondRegionInputGuard {
    Canonical,
    RejectedToolActionControl,
    RejectedComputeInternalState,
    RejectedSafetyOverride,
    RejectedImplicitMemoryMutation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainSecondRegionInputSurface {
    pub deferral_class: BlueBrainCandidateDeferralLifecycleClass,
    pub reference_validity: BlueBrainReferenceValidity,
    pub context_priority: BlueBrainContextEvidencePriorityClass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainSecondRegionAdvisoryOutputClass {
    CaveatHint,
    DeferralHint,
    ReferenceBoundedSignal,
    BlockedDeferred,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainSecondRegionOutputSurface {
    pub advisory_class: BlueBrainSecondRegionAdvisoryOutputClass,
    pub runtime_contract_signal: BlueBrainSecondRegionContractSignal,
    pub selection_contract_signal: BlueBrainSecondRegionContractSignal,
    pub reference_contract_signal: BlueBrainSecondRegionContractSignal,
    pub runtime_diagnostic_state: BlueBrainSecondRegionDiagnosticState,
    pub selection_diagnostic_state: BlueBrainSecondRegionDiagnosticState,
    pub reference_diagnostic_state: BlueBrainSecondRegionDiagnosticState,
    pub runtime_advisory_only: bool,
    pub selection_advisory_only: bool,
    pub reference_bounded_only: bool,
    pub direct_action_selection: bool,
    pub direct_execution_trigger: bool,
    pub direct_retry_trigger: bool,
    pub direct_memory_commit: bool,
    pub direct_compute_invocation: bool,
    pub safety_override: bool,
}

pub fn blue_brain_second_region_runtime_contract_signal(
    output: BlueBrainSecondRegionOutputSurface,
) -> BlueBrainSecondRegionContractSignal {
    output.runtime_contract_signal
}

pub fn blue_brain_second_region_selection_contract_signal(
    output: BlueBrainSecondRegionOutputSurface,
) -> BlueBrainSecondRegionContractSignal {
    output.selection_contract_signal
}

pub fn blue_brain_second_region_reference_contract_signal(
    output: BlueBrainSecondRegionOutputSurface,
) -> BlueBrainSecondRegionContractSignal {
    output.reference_contract_signal
}

fn second_region_diagnostic_state_for_signal(
    signal: BlueBrainSecondRegionContractSignal,
) -> BlueBrainSecondRegionDiagnosticState {
    match signal {
        BlueBrainSecondRegionContractSignal::RegionToRuntimeAdvisory
        | BlueBrainSecondRegionContractSignal::RegionToSelectionAdvisory
        | BlueBrainSecondRegionContractSignal::RuntimeToRegionBoundedInput
        | BlueBrainSecondRegionContractSignal::SelectionToRegionBoundedStateInput => {
            BlueBrainSecondRegionDiagnosticState::Region2AdvisoryOnlyDiagnostic
        }
        BlueBrainSecondRegionContractSignal::Caveated => {
            BlueBrainSecondRegionDiagnosticState::Region2CaveatedDiagnostic
        }
        BlueBrainSecondRegionContractSignal::Deferred => {
            BlueBrainSecondRegionDiagnosticState::Region2DeferredDiagnostic
        }
        BlueBrainSecondRegionContractSignal::Blocked => {
            BlueBrainSecondRegionDiagnosticState::Region2BlockedDiagnostic
        }
        BlueBrainSecondRegionContractSignal::Insufficient => {
            BlueBrainSecondRegionDiagnosticState::Region2InsufficientDiagnostic
        }
        BlueBrainSecondRegionContractSignal::ReferenceOnly
        | BlueBrainSecondRegionContractSignal::RegionReferenceSignal => {
            BlueBrainSecondRegionDiagnosticState::Region2DiagnosticOnlyState
        }
        BlueBrainSecondRegionContractSignal::NonCanonicalInternalOnly => {
            BlueBrainSecondRegionDiagnosticState::NonCanonicalInternalOnlyRegion2DiagnosticPath
        }
    }
}

pub fn classify_blue_brain_second_region_input_guard(
    source: BlueBrainSecondRegionInputSource,
) -> BlueBrainSecondRegionInputGuard {
    match source {
        BlueBrainSecondRegionInputSource::RuntimeDeferralSignal
        | BlueBrainSecondRegionInputSource::ContextReferenceSignal
        | BlueBrainSecondRegionInputSource::ContextEvidencePrioritySignal => {
            BlueBrainSecondRegionInputGuard::Canonical
        }
        BlueBrainSecondRegionInputSource::ToolActionControlSignal => {
            BlueBrainSecondRegionInputGuard::RejectedToolActionControl
        }
        BlueBrainSecondRegionInputSource::ComputeInternalStateSignal => {
            BlueBrainSecondRegionInputGuard::RejectedComputeInternalState
        }
        BlueBrainSecondRegionInputSource::SafetyOverrideSignal => {
            BlueBrainSecondRegionInputGuard::RejectedSafetyOverride
        }
        BlueBrainSecondRegionInputSource::ImplicitMemoryMutationSignal => {
            BlueBrainSecondRegionInputGuard::RejectedImplicitMemoryMutation
        }
    }
}

pub fn evaluate_blue_brain_second_region_memory_context(
    input: BlueBrainSecondRegionInputSurface,
) -> (
    BlueBrainSecondRegionStateSurface,
    BlueBrainSecondRegionOutputSurface,
) {
    let (
        state,
        advisory_class,
        runtime_contract_signal,
        selection_contract_signal,
        reference_contract_signal,
    ) = if input.context_priority
        == BlueBrainContextEvidencePriorityClass::NonCanonicalInternalOnlyPriorityPath
        || input.reference_validity == BlueBrainReferenceValidity::NonCanonicalInternalOnlyPath
    {
        (
            BlueBrainSecondRegionStateSurface::NonCanonicalInternalOnly,
            BlueBrainSecondRegionAdvisoryOutputClass::NonCanonicalInternalOnly,
            BlueBrainSecondRegionContractSignal::NonCanonicalInternalOnly,
            BlueBrainSecondRegionContractSignal::NonCanonicalInternalOnly,
            BlueBrainSecondRegionContractSignal::NonCanonicalInternalOnly,
        )
    } else if input.deferral_class == BlueBrainCandidateDeferralLifecycleClass::CandidateRejected {
        (
            BlueBrainSecondRegionStateSurface::BlockedState,
            BlueBrainSecondRegionAdvisoryOutputClass::BlockedDeferred,
            BlueBrainSecondRegionContractSignal::Blocked,
            BlueBrainSecondRegionContractSignal::Blocked,
            BlueBrainSecondRegionContractSignal::Blocked,
        )
    } else if matches!(
        input.deferral_class,
        BlueBrainCandidateDeferralLifecycleClass::CandidateDeferred
    ) {
        (
            BlueBrainSecondRegionStateSurface::DeferredState,
            BlueBrainSecondRegionAdvisoryOutputClass::DeferralHint,
            BlueBrainSecondRegionContractSignal::Deferred,
            BlueBrainSecondRegionContractSignal::Deferred,
            BlueBrainSecondRegionContractSignal::Deferred,
        )
    } else if input.reference_validity == BlueBrainReferenceValidity::ReferenceOnly {
        (
            BlueBrainSecondRegionStateSurface::ReferenceOnlyState,
            BlueBrainSecondRegionAdvisoryOutputClass::ReferenceBoundedSignal,
            BlueBrainSecondRegionContractSignal::RegionToRuntimeAdvisory,
            BlueBrainSecondRegionContractSignal::RegionToSelectionAdvisory,
            BlueBrainSecondRegionContractSignal::ReferenceOnly,
        )
    } else if matches!(
        input.reference_validity,
        BlueBrainReferenceValidity::Insufficient
    ) {
        (
            BlueBrainSecondRegionStateSurface::CaveatedReferenceState,
            BlueBrainSecondRegionAdvisoryOutputClass::CaveatHint,
            BlueBrainSecondRegionContractSignal::Insufficient,
            BlueBrainSecondRegionContractSignal::Insufficient,
            BlueBrainSecondRegionContractSignal::Insufficient,
        )
    } else if matches!(
        input.reference_validity,
        BlueBrainReferenceValidity::Caveated
    ) {
        (
            BlueBrainSecondRegionStateSurface::CaveatedReferenceState,
            BlueBrainSecondRegionAdvisoryOutputClass::CaveatHint,
            BlueBrainSecondRegionContractSignal::Caveated,
            BlueBrainSecondRegionContractSignal::Caveated,
            BlueBrainSecondRegionContractSignal::Caveated,
        )
    } else {
        (
            BlueBrainSecondRegionStateSurface::ActiveBoundedAdvisoryOnly,
            BlueBrainSecondRegionAdvisoryOutputClass::ReferenceBoundedSignal,
            BlueBrainSecondRegionContractSignal::RegionToRuntimeAdvisory,
            BlueBrainSecondRegionContractSignal::RegionToSelectionAdvisory,
            BlueBrainSecondRegionContractSignal::RegionReferenceSignal,
        )
    };

    let runtime_diagnostic_state =
        second_region_diagnostic_state_for_signal(runtime_contract_signal);
    let selection_diagnostic_state =
        second_region_diagnostic_state_for_signal(selection_contract_signal);
    let reference_diagnostic_state =
        second_region_diagnostic_state_for_signal(reference_contract_signal);

    let output = BlueBrainSecondRegionOutputSurface {
        advisory_class,
        runtime_contract_signal,
        selection_contract_signal,
        reference_contract_signal,
        runtime_diagnostic_state,
        selection_diagnostic_state,
        reference_diagnostic_state,
        runtime_advisory_only: true,
        selection_advisory_only: true,
        reference_bounded_only: true,
        direct_action_selection: false,
        direct_execution_trigger: false,
        direct_retry_trigger: false,
        direct_memory_commit: false,
        direct_compute_invocation: false,
        safety_override: false,
    };
    (state, output)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFirstRegionStateSurface {
    ActiveBoundedAdvisoryOnly,
    BlockedDeferred,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFirstRegionInputSource {
    RuntimeSelectionSignal,
    RuntimeDeferralSignal,
    ContextReferenceSignal,
    ToolActionControlSignal,
    ComputeInternalStateSignal,
    SafetyOverrideSignal,
    ImplicitMemoryMutationSignal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFirstRegionInputGuard {
    Canonical,
    RejectedToolActionControl,
    RejectedComputeInternalState,
    RejectedSafetyOverride,
    RejectedImplicitMemoryMutation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainFirstRegionInputSurface {
    pub attention_class: BlueBrainControlAttentionSelectionClass,
    pub deferral_class: BlueBrainCandidateDeferralLifecycleClass,
    pub reference_validity: BlueBrainReferenceValidity,
    pub context_priority: BlueBrainContextEvidencePriorityClass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFirstRegionAdvisoryOutputClass {
    CaveatHint,
    PriorityHint,
    DeferralHint,
    ReferenceBoundedSignal,
    BlockedDeferred,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFirstRegionContractSignal {
    RegionToRuntimeAdvisory,
    RuntimeToRegionBoundedInput,
    RegionToSelectionAdvisory,
    SelectionToRegionBoundedStateInput,
    RegionReference,
    Caveated,
    Deferred,
    Blocked,
    Insufficient,
    DiagnosticOnly,
    ReferenceOnly,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFirstRegionDiagnosticState {
    RegionAdvisoryOnlyDiagnostic,
    RegionCaveatedDiagnostic,
    RegionDeferredDiagnostic,
    RegionBlockedDiagnostic,
    RegionInsufficientDiagnostic,
    RegionDiagnosticOnlyState,
    NonCanonicalInternalOnlyRegionDiagnosticPath,
}

pub const CANONICAL_BLUE_BRAIN_FIRST_REGION_DIAGNOSTIC_MAP: [BlueBrainFirstRegionDiagnosticState;
    7] = [
    BlueBrainFirstRegionDiagnosticState::RegionAdvisoryOnlyDiagnostic,
    BlueBrainFirstRegionDiagnosticState::RegionCaveatedDiagnostic,
    BlueBrainFirstRegionDiagnosticState::RegionDeferredDiagnostic,
    BlueBrainFirstRegionDiagnosticState::RegionBlockedDiagnostic,
    BlueBrainFirstRegionDiagnosticState::RegionInsufficientDiagnostic,
    BlueBrainFirstRegionDiagnosticState::RegionDiagnosticOnlyState,
    BlueBrainFirstRegionDiagnosticState::NonCanonicalInternalOnlyRegionDiagnosticPath,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainFirstRegionOutputSurface {
    pub advisory_class: BlueBrainFirstRegionAdvisoryOutputClass,
    pub runtime_advisory_only: bool,
    pub selection_advisory_only: bool,
    pub reference_bounded_only: bool,
    pub direct_action_selection: bool,
    pub direct_execution_trigger: bool,
    pub direct_retry_trigger: bool,
    pub direct_memory_commit: bool,
    pub direct_compute_invocation: bool,
    pub safety_override: bool,
    pub contract_signal: BlueBrainFirstRegionContractSignal,
    pub diagnostic_state: BlueBrainFirstRegionDiagnosticState,
    pub reference_only: bool,
}

pub fn classify_blue_brain_first_region_input_guard(
    source: BlueBrainFirstRegionInputSource,
) -> BlueBrainFirstRegionInputGuard {
    match source {
        BlueBrainFirstRegionInputSource::RuntimeSelectionSignal
        | BlueBrainFirstRegionInputSource::RuntimeDeferralSignal
        | BlueBrainFirstRegionInputSource::ContextReferenceSignal => {
            BlueBrainFirstRegionInputGuard::Canonical
        }
        BlueBrainFirstRegionInputSource::ToolActionControlSignal => {
            BlueBrainFirstRegionInputGuard::RejectedToolActionControl
        }
        BlueBrainFirstRegionInputSource::ComputeInternalStateSignal => {
            BlueBrainFirstRegionInputGuard::RejectedComputeInternalState
        }
        BlueBrainFirstRegionInputSource::SafetyOverrideSignal => {
            BlueBrainFirstRegionInputGuard::RejectedSafetyOverride
        }
        BlueBrainFirstRegionInputSource::ImplicitMemoryMutationSignal => {
            BlueBrainFirstRegionInputGuard::RejectedImplicitMemoryMutation
        }
    }
}

pub fn blue_brain_first_region_runtime_contract_signal(
    output: BlueBrainFirstRegionOutputSurface,
) -> BlueBrainFirstRegionContractSignal {
    output.contract_signal
}

pub fn blue_brain_first_region_selection_contract_signal(
    output: BlueBrainFirstRegionOutputSurface,
) -> BlueBrainFirstRegionContractSignal {
    output.contract_signal
}

pub fn blue_brain_first_region_reference_contract_signal(
    output: BlueBrainFirstRegionOutputSurface,
) -> BlueBrainFirstRegionContractSignal {
    output.contract_signal
}

pub fn blue_brain_first_region_is_canonical_contract_signal(
    signal: BlueBrainFirstRegionContractSignal,
) -> bool {
    !matches!(
        signal,
        BlueBrainFirstRegionContractSignal::NonCanonicalInternalOnly
    )
}

pub fn evaluate_blue_brain_first_region_attention_selection(
    input: BlueBrainFirstRegionInputSurface,
) -> (
    BlueBrainFirstRegionStateSurface,
    BlueBrainFirstRegionOutputSurface,
) {
    let blocked_or_deferred = matches!(
        input.deferral_class,
        BlueBrainCandidateDeferralLifecycleClass::CandidateDeferred
            | BlueBrainCandidateDeferralLifecycleClass::CandidateDeferredPendingStrongerEvidence
            | BlueBrainCandidateDeferralLifecycleClass::CandidateDeferredPendingContextUpdate
            | BlueBrainCandidateDeferralLifecycleClass::CandidateRejected
            | BlueBrainCandidateDeferralLifecycleClass::CandidateStale
            | BlueBrainCandidateDeferralLifecycleClass::CandidateInsufficient
    );

    let non_canonical_attention = input.attention_class
        == BlueBrainControlAttentionSelectionClass::NonCanonicalInternalOnlySelectionPath;

    let state = if non_canonical_attention {
        BlueBrainFirstRegionStateSurface::NonCanonicalInternalOnly
    } else if blocked_or_deferred {
        BlueBrainFirstRegionStateSurface::BlockedDeferred
    } else {
        BlueBrainFirstRegionStateSurface::ActiveBoundedAdvisoryOnly
    };

    let advisory_class = match state {
        BlueBrainFirstRegionStateSurface::NonCanonicalInternalOnly => {
            BlueBrainFirstRegionAdvisoryOutputClass::NonCanonicalInternalOnly
        }
        BlueBrainFirstRegionStateSurface::BlockedDeferred => {
            BlueBrainFirstRegionAdvisoryOutputClass::BlockedDeferred
        }
        BlueBrainFirstRegionStateSurface::ActiveBoundedAdvisoryOnly => {
            if input.reference_validity == BlueBrainReferenceValidity::Caveated {
                BlueBrainFirstRegionAdvisoryOutputClass::CaveatHint
            } else if matches!(
                input.context_priority,
                BlueBrainContextEvidencePriorityClass::PrimaryContext
                    | BlueBrainContextEvidencePriorityClass::PrimaryEvidenceReference
            ) {
                BlueBrainFirstRegionAdvisoryOutputClass::PriorityHint
            } else if matches!(
                input.context_priority,
                BlueBrainContextEvidencePriorityClass::DeferredContext
                    | BlueBrainContextEvidencePriorityClass::StaleContext
                    | BlueBrainContextEvidencePriorityClass::InsufficientContext
            ) {
                BlueBrainFirstRegionAdvisoryOutputClass::DeferralHint
            } else {
                BlueBrainFirstRegionAdvisoryOutputClass::ReferenceBoundedSignal
            }
        }
    };

    let contract_signal = if non_canonical_attention {
        BlueBrainFirstRegionContractSignal::NonCanonicalInternalOnly
    } else if blocked_or_deferred {
        if matches!(
            input.deferral_class,
            BlueBrainCandidateDeferralLifecycleClass::CandidateDeferred
                | BlueBrainCandidateDeferralLifecycleClass::CandidateDeferredPendingStrongerEvidence
                | BlueBrainCandidateDeferralLifecycleClass::CandidateDeferredPendingContextUpdate
        ) {
            BlueBrainFirstRegionContractSignal::Deferred
        } else {
            BlueBrainFirstRegionContractSignal::Blocked
        }
    } else if matches!(
        input.reference_validity,
        BlueBrainReferenceValidity::Caveated | BlueBrainReferenceValidity::Stale
    ) {
        BlueBrainFirstRegionContractSignal::Caveated
    } else if matches!(
        input.reference_validity,
        BlueBrainReferenceValidity::Insufficient
    ) {
        BlueBrainFirstRegionContractSignal::Insufficient
    } else if matches!(
        input.reference_validity,
        BlueBrainReferenceValidity::ReferenceOnly
    ) {
        BlueBrainFirstRegionContractSignal::DiagnosticOnly
    } else {
        BlueBrainFirstRegionContractSignal::RegionToRuntimeAdvisory
    };

    let diagnostic_state = match contract_signal {
        BlueBrainFirstRegionContractSignal::RegionToRuntimeAdvisory
        | BlueBrainFirstRegionContractSignal::RegionToSelectionAdvisory => {
            BlueBrainFirstRegionDiagnosticState::RegionAdvisoryOnlyDiagnostic
        }
        BlueBrainFirstRegionContractSignal::Caveated => {
            BlueBrainFirstRegionDiagnosticState::RegionCaveatedDiagnostic
        }
        BlueBrainFirstRegionContractSignal::Deferred => {
            BlueBrainFirstRegionDiagnosticState::RegionDeferredDiagnostic
        }
        BlueBrainFirstRegionContractSignal::Blocked => {
            BlueBrainFirstRegionDiagnosticState::RegionBlockedDiagnostic
        }
        BlueBrainFirstRegionContractSignal::Insufficient => {
            BlueBrainFirstRegionDiagnosticState::RegionInsufficientDiagnostic
        }
        BlueBrainFirstRegionContractSignal::DiagnosticOnly
        | BlueBrainFirstRegionContractSignal::ReferenceOnly
        | BlueBrainFirstRegionContractSignal::RegionReference => {
            BlueBrainFirstRegionDiagnosticState::RegionDiagnosticOnlyState
        }
        BlueBrainFirstRegionContractSignal::NonCanonicalInternalOnly => {
            BlueBrainFirstRegionDiagnosticState::NonCanonicalInternalOnlyRegionDiagnosticPath
        }
        BlueBrainFirstRegionContractSignal::RuntimeToRegionBoundedInput
        | BlueBrainFirstRegionContractSignal::SelectionToRegionBoundedStateInput => {
            BlueBrainFirstRegionDiagnosticState::RegionAdvisoryOnlyDiagnostic
        }
    };

    (
        state,
        BlueBrainFirstRegionOutputSurface {
            advisory_class,
            runtime_advisory_only: true,
            selection_advisory_only: true,
            reference_bounded_only: true,
            direct_action_selection: false,
            direct_execution_trigger: false,
            direct_retry_trigger: false,
            direct_memory_commit: false,
            direct_compute_invocation: false,
            safety_override: false,
            contract_signal,
            diagnostic_state,
            reference_only: matches!(
                contract_signal,
                BlueBrainFirstRegionContractSignal::DiagnosticOnly
                    | BlueBrainFirstRegionContractSignal::ReferenceOnly
            ),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_region_map_contains_all_required_paths() {
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstRegionPathClass::RegionToRuntimeAdvisorySignal));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstRegionPathClass::RuntimeToRegionBoundedInput));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstRegionPathClass::RegionToSelectionAdvisorySignal));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstRegionPathClass::SelectionToRegionBoundedStateInput));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstRegionPathClass::RegionReferenceSignal));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstRegionPathClass::CaveatedDeferredBlockedRegionContractSignal));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstRegionPathClass::ReferenceOnlyRegionContractSignal));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstRegionPathClass::RegionInputSurface));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstRegionPathClass::RegionStateSurface));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstRegionPathClass::RegionOutputAdvisorySurface));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstRegionPathClass::RegionReferenceSurface));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstRegionPathClass::BlockedDeferredRegionPath));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstRegionPathClass::NonCanonicalInternalOnlyRegionPath));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstRegionPathClass::TestOnlyHelperNonOperationalPath));
    }

    #[test]
    fn first_region_hardening_map_contains_required_classes() {
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_HARDENING_MAP
            .contains(&BlueBrainFirstRegionHardeningClass::GuardedCanonicalRegionSurface));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_HARDENING_MAP
            .contains(&BlueBrainFirstRegionHardeningClass::GuardedDiagnosticsPath));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_HARDENING_MAP
            .contains(&BlueBrainFirstRegionHardeningClass::BlockedForbiddenAuthorityPath));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_HARDENING_MAP
            .contains(&BlueBrainFirstRegionHardeningClass::NonCanonicalInternalOnlyRegionPath));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_HARDENING_MAP
            .contains(&BlueBrainFirstRegionHardeningClass::TestOnlyHelperNonOperationalPath));
    }

    #[test]
    fn first_region_diagnostic_map_contains_all_required_states() {
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_DIAGNOSTIC_MAP
            .contains(&BlueBrainFirstRegionDiagnosticState::RegionAdvisoryOnlyDiagnostic));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_DIAGNOSTIC_MAP
            .contains(&BlueBrainFirstRegionDiagnosticState::RegionCaveatedDiagnostic));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_DIAGNOSTIC_MAP
            .contains(&BlueBrainFirstRegionDiagnosticState::RegionDeferredDiagnostic));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_DIAGNOSTIC_MAP
            .contains(&BlueBrainFirstRegionDiagnosticState::RegionBlockedDiagnostic));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_DIAGNOSTIC_MAP
            .contains(&BlueBrainFirstRegionDiagnosticState::RegionInsufficientDiagnostic));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_DIAGNOSTIC_MAP
            .contains(&BlueBrainFirstRegionDiagnosticState::RegionDiagnosticOnlyState));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_DIAGNOSTIC_MAP.contains(
            &BlueBrainFirstRegionDiagnosticState::NonCanonicalInternalOnlyRegionDiagnosticPath
        ));
    }

    #[test]
    fn first_region_finalization_map_contains_required_classes() {
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_FINALIZATION_MAP
            .contains(&BlueBrainFirstRegionFinalizationClass::StableFirstRegionBaseline));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_FINALIZATION_MAP
            .contains(&BlueBrainFirstRegionFinalizationClass::UsableWithCaveatsFirstRegionSurface));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_FINALIZATION_MAP
            .contains(&BlueBrainFirstRegionFinalizationClass::AdvisoryOnlyFrozenRegionSignal));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_FINALIZATION_MAP
            .contains(&BlueBrainFirstRegionFinalizationClass::DiagnosticOnlyDeferredRegionState));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_FINALIZATION_MAP
            .contains(&BlueBrainFirstRegionFinalizationClass::SecondRegionNotOpenedYet));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_FINALIZATION_MAP
            .contains(&BlueBrainFirstRegionFinalizationClass::NonCanonicalInternalOnlyRegionPath));
    }

    #[test]
    fn first_region_stabilization_map_contains_required_classes() {
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_STABILIZATION_MAP
            .contains(&BlueBrainFirstRegionStabilizationClass::StableFirstRegionBaseline));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_STABILIZATION_MAP
            .contains(&BlueBrainFirstRegionStabilizationClass::MaintenanceHardenedRegionSurface));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_STABILIZATION_MAP
            .contains(&BlueBrainFirstRegionStabilizationClass::MaintenanceHardenedDiagnosticsPath));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_STABILIZATION_MAP
            .contains(&BlueBrainFirstRegionStabilizationClass::MaintenanceHardenedContractPath));
        assert!(
            CANONICAL_BLUE_BRAIN_FIRST_REGION_STABILIZATION_MAP.contains(
                &BlueBrainFirstRegionStabilizationClass::NonCanonicalInternalOnlyResidualPath
            )
        );
    }

    #[test]
    fn second_region_expansion_state_remains_not_opened_yet() {
        assert_eq!(
            BLUE_BRAIN_SECOND_REGION_EXPANSION_STATE,
            BlueBrainSecondRegionExpansionState::NotOpenedYetExplicitRescopeRequired
        );
    }

    #[test]
    fn second_region_selection_map_contains_required_classes() {
        assert!(CANONICAL_BLUE_BRAIN_SECOND_REGION_SELECTION_MAP
            .contains(&BlueBrainSecondRegionSelectionClass::SecondExpansionCandidate));
        assert!(CANONICAL_BLUE_BRAIN_SECOND_REGION_SELECTION_MAP
            .contains(&BlueBrainSecondRegionSelectionClass::ViableButNotSecond));
        assert!(CANONICAL_BLUE_BRAIN_SECOND_REGION_SELECTION_MAP
            .contains(&BlueBrainSecondRegionSelectionClass::LaterPhaseCandidate));
        assert!(CANONICAL_BLUE_BRAIN_SECOND_REGION_SELECTION_MAP
            .contains(&BlueBrainSecondRegionSelectionClass::SimulationOnlyDeferredCandidate));
        assert!(CANONICAL_BLUE_BRAIN_SECOND_REGION_SELECTION_MAP
            .contains(&BlueBrainSecondRegionSelectionClass::NonCanonicalInternalOnlyPath));
    }

    #[test]
    fn second_region_class_selection_is_memory_context_related() {
        assert_eq!(
            BLUE_BRAIN_SECOND_REGION_CLASS_SELECTION,
            BlueBrainSecondRegionClass::MemoryContextRelated
        );
    }

    #[test]
    fn second_region_integration_map_contains_minimal_surfaces() {
        assert!(CANONICAL_BLUE_BRAIN_SECOND_REGION_INTEGRATION_MAP
            .contains(&BlueBrainSecondRegionPathClass::RegionInputSurface));
        assert!(CANONICAL_BLUE_BRAIN_SECOND_REGION_INTEGRATION_MAP
            .contains(&BlueBrainSecondRegionPathClass::RegionStateSurface));
        assert!(CANONICAL_BLUE_BRAIN_SECOND_REGION_INTEGRATION_MAP
            .contains(&BlueBrainSecondRegionPathClass::RegionOutputAdvisorySurface));
        assert!(CANONICAL_BLUE_BRAIN_SECOND_REGION_INTEGRATION_MAP
            .contains(&BlueBrainSecondRegionPathClass::RegionReferenceSurface));
        assert!(CANONICAL_BLUE_BRAIN_SECOND_REGION_INTEGRATION_MAP
            .contains(&BlueBrainSecondRegionPathClass::BlockedDeferredRegionPath));
        assert!(CANONICAL_BLUE_BRAIN_SECOND_REGION_INTEGRATION_MAP
            .contains(&BlueBrainSecondRegionPathClass::NonCanonicalInternalOnlyRegionPath));
    }

    #[test]
    fn second_region_input_guard_accepts_only_canonical_inputs() {
        assert_eq!(
            classify_blue_brain_second_region_input_guard(
                BlueBrainSecondRegionInputSource::RuntimeDeferralSignal
            ),
            BlueBrainSecondRegionInputGuard::Canonical
        );
        assert_eq!(
            classify_blue_brain_second_region_input_guard(
                BlueBrainSecondRegionInputSource::ContextReferenceSignal
            ),
            BlueBrainSecondRegionInputGuard::Canonical
        );
        assert_eq!(
            classify_blue_brain_second_region_input_guard(
                BlueBrainSecondRegionInputSource::ContextEvidencePrioritySignal
            ),
            BlueBrainSecondRegionInputGuard::Canonical
        );
    }

    #[test]
    fn second_region_input_guard_rejects_forbidden_authority_inputs() {
        assert_eq!(
            classify_blue_brain_second_region_input_guard(
                BlueBrainSecondRegionInputSource::ToolActionControlSignal
            ),
            BlueBrainSecondRegionInputGuard::RejectedToolActionControl
        );
        assert_eq!(
            classify_blue_brain_second_region_input_guard(
                BlueBrainSecondRegionInputSource::ComputeInternalStateSignal
            ),
            BlueBrainSecondRegionInputGuard::RejectedComputeInternalState
        );
        assert_eq!(
            classify_blue_brain_second_region_input_guard(
                BlueBrainSecondRegionInputSource::SafetyOverrideSignal
            ),
            BlueBrainSecondRegionInputGuard::RejectedSafetyOverride
        );
        assert_eq!(
            classify_blue_brain_second_region_input_guard(
                BlueBrainSecondRegionInputSource::ImplicitMemoryMutationSignal
            ),
            BlueBrainSecondRegionInputGuard::RejectedImplicitMemoryMutation
        );
    }

    #[test]
    fn second_region_output_stays_bounded_advisory_only() {
        let (_, output) =
            evaluate_blue_brain_second_region_memory_context(BlueBrainSecondRegionInputSurface {
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority: BlueBrainContextEvidencePriorityClass::SupportingContext,
            });

        assert!(output.runtime_advisory_only);
        assert!(output.selection_advisory_only);
        assert!(output.reference_bounded_only);
        assert!(!output.direct_action_selection);
        assert!(!output.direct_execution_trigger);
        assert!(!output.direct_retry_trigger);
        assert!(!output.direct_memory_commit);
        assert!(!output.direct_compute_invocation);
        assert!(!output.safety_override);
    }

    #[test]
    fn second_region_distinguishes_caveated_deferred_and_non_canonical_states() {
        let (caveated_state, caveated_output) =
            evaluate_blue_brain_second_region_memory_context(BlueBrainSecondRegionInputSurface {
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::Caveated,
                context_priority: BlueBrainContextEvidencePriorityClass::SupportingContext,
            });
        assert_eq!(
            caveated_state,
            BlueBrainSecondRegionStateSurface::CaveatedReferenceState
        );
        assert_eq!(
            caveated_output.advisory_class,
            BlueBrainSecondRegionAdvisoryOutputClass::CaveatHint
        );
        assert_eq!(
            caveated_output.runtime_diagnostic_state,
            BlueBrainSecondRegionDiagnosticState::Region2CaveatedDiagnostic
        );

        let (deferred_state, deferred_output) =
            evaluate_blue_brain_second_region_memory_context(BlueBrainSecondRegionInputSurface {
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateDeferred,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority: BlueBrainContextEvidencePriorityClass::DeferredContext,
            });
        assert_eq!(
            deferred_state,
            BlueBrainSecondRegionStateSurface::DeferredState
        );
        assert_eq!(
            deferred_output.advisory_class,
            BlueBrainSecondRegionAdvisoryOutputClass::DeferralHint
        );
        assert_eq!(
            deferred_output.runtime_contract_signal,
            BlueBrainSecondRegionContractSignal::Deferred
        );
        assert_eq!(
            deferred_output.runtime_diagnostic_state,
            BlueBrainSecondRegionDiagnosticState::Region2DeferredDiagnostic
        );

        let (non_canonical_state, non_canonical_output) =
            evaluate_blue_brain_second_region_memory_context(BlueBrainSecondRegionInputSurface {
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority:
                    BlueBrainContextEvidencePriorityClass::NonCanonicalInternalOnlyPriorityPath,
            });
        assert_eq!(
            non_canonical_state,
            BlueBrainSecondRegionStateSurface::NonCanonicalInternalOnly
        );
        assert_eq!(
            non_canonical_output.advisory_class,
            BlueBrainSecondRegionAdvisoryOutputClass::NonCanonicalInternalOnly
        );

        let (blocked_state, blocked_output) =
            evaluate_blue_brain_second_region_memory_context(BlueBrainSecondRegionInputSurface {
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateRejected,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority: BlueBrainContextEvidencePriorityClass::SupportingContext,
            });
        assert_eq!(
            blocked_state,
            BlueBrainSecondRegionStateSurface::BlockedState
        );
        assert_eq!(
            blocked_output.runtime_contract_signal,
            BlueBrainSecondRegionContractSignal::Blocked
        );
        assert_eq!(
            blocked_output.runtime_diagnostic_state,
            BlueBrainSecondRegionDiagnosticState::Region2BlockedDiagnostic
        );
    }

    #[test]
    fn second_region_distinguishes_insufficient_from_caveated() {
        let (_, insufficient_output) =
            evaluate_blue_brain_second_region_memory_context(BlueBrainSecondRegionInputSurface {
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::Insufficient,
                context_priority: BlueBrainContextEvidencePriorityClass::InsufficientContext,
            });
        let (_, caveated_output) =
            evaluate_blue_brain_second_region_memory_context(BlueBrainSecondRegionInputSurface {
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::Caveated,
                context_priority: BlueBrainContextEvidencePriorityClass::SupportingContext,
            });
        assert_eq!(
            insufficient_output.runtime_contract_signal,
            BlueBrainSecondRegionContractSignal::Insufficient
        );
        assert_eq!(
            insufficient_output.runtime_diagnostic_state,
            BlueBrainSecondRegionDiagnosticState::Region2InsufficientDiagnostic
        );
        assert_eq!(
            caveated_output.runtime_diagnostic_state,
            BlueBrainSecondRegionDiagnosticState::Region2CaveatedDiagnostic
        );
    }

    #[test]
    fn second_region_distinguishes_reference_only_from_caveated_and_blocked() {
        let (reference_only_state, reference_only_output) =
            evaluate_blue_brain_second_region_memory_context(BlueBrainSecondRegionInputSurface {
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::ReferenceOnly,
                context_priority: BlueBrainContextEvidencePriorityClass::SupportingContext,
            });
        assert_eq!(
            reference_only_state,
            BlueBrainSecondRegionStateSurface::ReferenceOnlyState
        );
        assert_eq!(
            reference_only_output.reference_contract_signal,
            BlueBrainSecondRegionContractSignal::ReferenceOnly
        );
        assert_eq!(
            reference_only_output.runtime_contract_signal,
            BlueBrainSecondRegionContractSignal::RegionToRuntimeAdvisory
        );
        assert_eq!(
            reference_only_output.reference_diagnostic_state,
            BlueBrainSecondRegionDiagnosticState::Region2DiagnosticOnlyState
        );
    }

    #[test]
    fn second_region_runtime_selection_reference_use_same_diagnostics_surface() {
        let (_, output) =
            evaluate_blue_brain_second_region_memory_context(BlueBrainSecondRegionInputSurface {
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateDeferred,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority: BlueBrainContextEvidencePriorityClass::DeferredContext,
            });
        assert_eq!(
            blue_brain_second_region_runtime_contract_signal(output),
            BlueBrainSecondRegionContractSignal::Deferred
        );
        assert_eq!(
            blue_brain_second_region_selection_contract_signal(output),
            BlueBrainSecondRegionContractSignal::Deferred
        );
        assert_eq!(
            blue_brain_second_region_reference_contract_signal(output),
            BlueBrainSecondRegionContractSignal::Deferred
        );
        assert_eq!(
            output.runtime_diagnostic_state,
            output.selection_diagnostic_state
        );
        assert_eq!(
            output.selection_diagnostic_state,
            output.reference_diagnostic_state
        );
    }

    #[test]
    fn second_region_integration_doc_pins_boundaries_and_no_authority_escalation() {
        let doc = include_str!(
            "../../../docs/blue_brain_second_region_integration_serie_bb26_prompt2_v1.md"
        );
        assert!(doc.contains("Memory/Context-related"));
        assert!(doc.contains("region-2 input surface"));
        assert!(doc.contains("region-2 state surface"));
        assert!(doc.contains("region-2 output/advisory surface"));
        assert!(doc.contains("region-2 reference surface"));
        assert!(doc.contains("blocked/deferred region-2 path"));
        assert!(doc.contains("non-canonical/internal-only region-2 path"));
        assert!(doc.contains("keine direkte Action-/Retry-/Memory-/Compute-Autorität"));
        assert!(doc.contains("keine dritte Regionenklasse"));
        assert!(doc.contains("region-2-to-runtime advisory signal"));
        assert!(doc.contains("runtime-to-region-2 bounded input"));
        assert!(doc.contains("region-2-to-selection advisory signal"));
        assert!(doc.contains("selection-to-region-2 bounded state input"));
        assert!(doc.contains("region-2-reference signal"));
    }

    #[test]
    fn inter_region_relation_map_contains_required_classes() {
        assert!(CANONICAL_BLUE_BRAIN_INTER_REGION_RELATION_MAP
            .contains(&BlueBrainInterRegionRelationClass::Region1ToRegion2Bounded));
        assert!(CANONICAL_BLUE_BRAIN_INTER_REGION_RELATION_MAP
            .contains(&BlueBrainInterRegionRelationClass::Region2ToRegion1Bounded));
        assert!(CANONICAL_BLUE_BRAIN_INTER_REGION_RELATION_MAP
            .contains(&BlueBrainInterRegionRelationClass::SharedReferenceMediated));
        assert!(CANONICAL_BLUE_BRAIN_INTER_REGION_RELATION_MAP
            .contains(&BlueBrainInterRegionRelationClass::CaveatedInterRegion));
        assert!(CANONICAL_BLUE_BRAIN_INTER_REGION_RELATION_MAP
            .contains(&BlueBrainInterRegionRelationClass::BlockedDeferredInterRegion));
        assert!(CANONICAL_BLUE_BRAIN_INTER_REGION_RELATION_MAP
            .contains(&BlueBrainInterRegionRelationClass::NonCanonicalInternalOnlyPath));
    }

    #[test]
    fn inter_region_relation_stays_advisory_only_without_direct_authority() {
        let (_, first_output) = evaluate_blue_brain_first_region_attention_selection(
            BlueBrainFirstRegionInputSurface {
                attention_class: BlueBrainControlAttentionSelectionClass::AttentionTarget,
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
            },
        );
        let (_, second_output) =
            evaluate_blue_brain_second_region_memory_context(BlueBrainSecondRegionInputSurface {
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority: BlueBrainContextEvidencePriorityClass::SupportingContext,
            });

        let relation = evaluate_blue_brain_inter_region_relation(first_output, second_output);
        assert!(relation.region1_to_region2_advisory_only);
        assert!(relation.region2_to_region1_advisory_only);
        assert!(relation.reference_mediated_only);
        assert!(!relation.direct_action_selection);
        assert!(!relation.direct_execution_trigger);
        assert!(!relation.direct_retry_trigger);
        assert!(!relation.direct_memory_commit);
        assert!(!relation.direct_compute_invocation);
        assert!(!relation.safety_override);
    }

    #[test]
    fn inter_region_relation_distinguishes_caveated_deferred_and_shared_reference() {
        let (_, first_output) = evaluate_blue_brain_first_region_attention_selection(
            BlueBrainFirstRegionInputSurface {
                attention_class: BlueBrainControlAttentionSelectionClass::AttentionTarget,
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority: BlueBrainContextEvidencePriorityClass::SupportingContext,
            },
        );
        let (_, second_caveated) =
            evaluate_blue_brain_second_region_memory_context(BlueBrainSecondRegionInputSurface {
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::Caveated,
                context_priority: BlueBrainContextEvidencePriorityClass::SupportingContext,
            });
        let caveated = evaluate_blue_brain_inter_region_relation(first_output, second_caveated);
        assert_eq!(
            caveated.relation_class,
            BlueBrainInterRegionRelationClass::CaveatedInterRegion
        );

        let (_, second_deferred) =
            evaluate_blue_brain_second_region_memory_context(BlueBrainSecondRegionInputSurface {
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateDeferred,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority: BlueBrainContextEvidencePriorityClass::DeferredContext,
            });
        let deferred = evaluate_blue_brain_inter_region_relation(first_output, second_deferred);
        assert_eq!(
            deferred.relation_class,
            BlueBrainInterRegionRelationClass::BlockedDeferredInterRegion
        );
        assert!(deferred.deferred);

        let (_, second_reference_only) =
            evaluate_blue_brain_second_region_memory_context(BlueBrainSecondRegionInputSurface {
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::ReferenceOnly,
                context_priority: BlueBrainContextEvidencePriorityClass::SupportingContext,
            });
        let shared_ref =
            evaluate_blue_brain_inter_region_relation(first_output, second_reference_only);
        assert_eq!(
            shared_ref.relation_class,
            BlueBrainInterRegionRelationClass::SharedReferenceMediated
        );
    }

    #[test]
    fn inter_region_relation_doc_pins_scope_boundaries() {
        let doc = include_str!(
            "../../../docs/blue_brain_first_inter_region_relation_line_serie_bb26_prompt4_v1.md"
        );
        assert!(doc.contains("region-1-to-region-2 bounded relation"));
        assert!(doc.contains("region-2-to-region-1 bounded relation"));
        assert!(doc.contains("shared reference-mediated relation"));
        assert!(doc.contains("caveated inter-region relation"));
        assert!(doc.contains("blocked/deferred inter-region relation"));
        assert!(doc.contains("non-canonical/internal-only inter-region path"));
        assert!(doc.contains("no direct action selection"));
        assert!(doc.contains("no direct execution trigger"));
        assert!(doc.contains("no direct retry trigger"));
        assert!(doc.contains("no direct memory commit"));
        assert!(doc.contains("no direct compute invocation"));
        assert!(doc.contains("no safety override"));
    }

    #[test]
    fn second_region_diagnostics_doc_pins_canonical_state_line() {
        let doc = include_str!(
            "../../../docs/blue_brain_second_region_diagnostics_caveat_deferred_semantics_serie_bb26_prompt5_v1.md"
        );
        assert!(doc.contains("region-2 advisory-only diagnostic"));
        assert!(doc.contains("region-2 caveated diagnostic"));
        assert!(doc.contains("region-2 deferred diagnostic"));
        assert!(doc.contains("region-2 blocked diagnostic"));
        assert!(doc.contains("region-2 insufficient diagnostic"));
        assert!(doc.contains("region-2 diagnostic-only state"));
        assert!(doc.contains("caveated inter-region diagnostic influence"));
        assert!(doc.contains("non-canonical/internal-only region-2 diagnostic path"));
        assert!(doc.contains("no direct action trigger"));
        assert!(doc.contains("no direct execution trigger"));
        assert!(doc.contains("no direct retry trigger"));
        assert!(doc.contains("no direct memory commit"));
        assert!(doc.contains("no direct compute invocation"));
        assert!(doc.contains("no safety override"));
        assert!(doc.contains("no third-region expansion"));
        assert!(doc.contains("no broad inter-region platform"));
    }

    #[test]
    fn first_region_output_is_advisory_and_non_authoritative() {
        let (_, output) = evaluate_blue_brain_first_region_attention_selection(
            BlueBrainFirstRegionInputSurface {
                attention_class: BlueBrainControlAttentionSelectionClass::AttentionTarget,
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
            },
        );

        assert!(output.runtime_advisory_only);
        assert!(output.selection_advisory_only);
        assert!(output.reference_bounded_only);
        assert!(!output.direct_action_selection);
        assert!(!output.direct_execution_trigger);
        assert!(!output.direct_retry_trigger);
        assert!(!output.direct_memory_commit);
        assert!(!output.direct_compute_invocation);
        assert!(!output.safety_override);
        assert_eq!(
            output.contract_signal,
            BlueBrainFirstRegionContractSignal::RegionToRuntimeAdvisory
        );
        assert_eq!(
            output.diagnostic_state,
            BlueBrainFirstRegionDiagnosticState::RegionAdvisoryOnlyDiagnostic
        );
    }

    #[test]
    fn first_region_marks_blocked_and_non_canonical_paths() {
        let (blocked_state, _) = evaluate_blue_brain_first_region_attention_selection(
            BlueBrainFirstRegionInputSurface {
                attention_class: BlueBrainControlAttentionSelectionClass::AttentionTarget,
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateRejected,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority:
                    BlueBrainContextEvidencePriorityClass::SupportingEvidenceReference,
            },
        );
        assert_eq!(
            blocked_state,
            BlueBrainFirstRegionStateSurface::BlockedDeferred
        );

        let (non_canonical_state, non_canonical_output) =
            evaluate_blue_brain_first_region_attention_selection(BlueBrainFirstRegionInputSurface {
                attention_class:
                    BlueBrainControlAttentionSelectionClass::NonCanonicalInternalOnlySelectionPath,
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority: BlueBrainContextEvidencePriorityClass::SupportingEvidenceReference,
            });
        assert_eq!(
            non_canonical_state,
            BlueBrainFirstRegionStateSurface::NonCanonicalInternalOnly
        );
        assert_eq!(
            non_canonical_output.advisory_class,
            BlueBrainFirstRegionAdvisoryOutputClass::NonCanonicalInternalOnly
        );
        assert_eq!(
            non_canonical_output.contract_signal,
            BlueBrainFirstRegionContractSignal::NonCanonicalInternalOnly
        );
        assert_eq!(
            non_canonical_output.diagnostic_state,
            BlueBrainFirstRegionDiagnosticState::NonCanonicalInternalOnlyRegionDiagnosticPath
        );
    }

    #[test]
    fn first_region_distinguishes_deferred_blocked_caveated_and_reference_only() {
        let (_, deferred) = evaluate_blue_brain_first_region_attention_selection(
            BlueBrainFirstRegionInputSurface {
                attention_class: BlueBrainControlAttentionSelectionClass::AttentionTarget,
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateDeferred,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority: BlueBrainContextEvidencePriorityClass::DeferredContext,
            },
        );
        assert_eq!(
            deferred.contract_signal,
            BlueBrainFirstRegionContractSignal::Deferred
        );
        assert_eq!(
            deferred.diagnostic_state,
            BlueBrainFirstRegionDiagnosticState::RegionDeferredDiagnostic
        );

        let (_, blocked) = evaluate_blue_brain_first_region_attention_selection(
            BlueBrainFirstRegionInputSurface {
                attention_class: BlueBrainControlAttentionSelectionClass::AttentionTarget,
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateRejected,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
            },
        );
        assert_eq!(
            blocked.contract_signal,
            BlueBrainFirstRegionContractSignal::Blocked
        );
        assert_eq!(
            blocked.diagnostic_state,
            BlueBrainFirstRegionDiagnosticState::RegionBlockedDiagnostic
        );

        let (_, caveated) = evaluate_blue_brain_first_region_attention_selection(
            BlueBrainFirstRegionInputSurface {
                attention_class: BlueBrainControlAttentionSelectionClass::AttentionTarget,
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::Caveated,
                context_priority: BlueBrainContextEvidencePriorityClass::SupportingContext,
            },
        );
        assert_eq!(
            caveated.contract_signal,
            BlueBrainFirstRegionContractSignal::Caveated
        );
        assert_eq!(
            caveated.diagnostic_state,
            BlueBrainFirstRegionDiagnosticState::RegionCaveatedDiagnostic
        );

        let (_, reference_only) = evaluate_blue_brain_first_region_attention_selection(
            BlueBrainFirstRegionInputSurface {
                attention_class: BlueBrainControlAttentionSelectionClass::AttentionTarget,
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::ReferenceOnly,
                context_priority:
                    BlueBrainContextEvidencePriorityClass::SupportingEvidenceReference,
            },
        );
        assert_eq!(
            reference_only.contract_signal,
            BlueBrainFirstRegionContractSignal::DiagnosticOnly
        );
        assert!(reference_only.reference_only);
        assert_eq!(
            reference_only.diagnostic_state,
            BlueBrainFirstRegionDiagnosticState::RegionDiagnosticOnlyState
        );
    }

    #[test]
    fn first_region_marks_insufficient_without_promoting_to_caveated_or_blocked() {
        let (_, insufficient) = evaluate_blue_brain_first_region_attention_selection(
            BlueBrainFirstRegionInputSurface {
                attention_class: BlueBrainControlAttentionSelectionClass::AttentionTarget,
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::Insufficient,
                context_priority: BlueBrainContextEvidencePriorityClass::InsufficientContext,
            },
        );

        assert_eq!(
            insufficient.contract_signal,
            BlueBrainFirstRegionContractSignal::Insufficient
        );
        assert_eq!(
            insufficient.diagnostic_state,
            BlueBrainFirstRegionDiagnosticState::RegionInsufficientDiagnostic
        );
        assert!(!insufficient.direct_execution_trigger);
        assert!(!insufficient.direct_retry_trigger);
    }

    #[test]
    fn first_region_accepts_only_canonical_input_sources() {
        assert_eq!(
            classify_blue_brain_first_region_input_guard(
                BlueBrainFirstRegionInputSource::RuntimeSelectionSignal
            ),
            BlueBrainFirstRegionInputGuard::Canonical
        );
        assert_eq!(
            classify_blue_brain_first_region_input_guard(
                BlueBrainFirstRegionInputSource::RuntimeDeferralSignal
            ),
            BlueBrainFirstRegionInputGuard::Canonical
        );
        assert_eq!(
            classify_blue_brain_first_region_input_guard(
                BlueBrainFirstRegionInputSource::ContextReferenceSignal
            ),
            BlueBrainFirstRegionInputGuard::Canonical
        );
    }

    #[test]
    fn first_region_rejects_non_canonical_input_sources() {
        assert_eq!(
            classify_blue_brain_first_region_input_guard(
                BlueBrainFirstRegionInputSource::ToolActionControlSignal
            ),
            BlueBrainFirstRegionInputGuard::RejectedToolActionControl
        );
        assert_eq!(
            classify_blue_brain_first_region_input_guard(
                BlueBrainFirstRegionInputSource::ComputeInternalStateSignal
            ),
            BlueBrainFirstRegionInputGuard::RejectedComputeInternalState
        );
        assert_eq!(
            classify_blue_brain_first_region_input_guard(
                BlueBrainFirstRegionInputSource::SafetyOverrideSignal
            ),
            BlueBrainFirstRegionInputGuard::RejectedSafetyOverride
        );
        assert_eq!(
            classify_blue_brain_first_region_input_guard(
                BlueBrainFirstRegionInputSource::ImplicitMemoryMutationSignal
            ),
            BlueBrainFirstRegionInputGuard::RejectedImplicitMemoryMutation
        );
    }

    #[test]
    fn first_region_runtime_selection_reference_contract_reads_are_consistent() {
        let (_, output) = evaluate_blue_brain_first_region_attention_selection(
            BlueBrainFirstRegionInputSurface {
                attention_class: BlueBrainControlAttentionSelectionClass::AttentionTarget,
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::Caveated,
                context_priority: BlueBrainContextEvidencePriorityClass::SupportingContext,
            },
        );

        let runtime_signal = blue_brain_first_region_runtime_contract_signal(output);
        let selection_signal = blue_brain_first_region_selection_contract_signal(output);
        let reference_signal = blue_brain_first_region_reference_contract_signal(output);

        assert_eq!(runtime_signal, selection_signal);
        assert_eq!(runtime_signal, reference_signal);
        assert_eq!(runtime_signal, BlueBrainFirstRegionContractSignal::Caveated);
        assert!(blue_brain_first_region_is_canonical_contract_signal(
            runtime_signal
        ));
    }

    #[test]
    fn first_region_non_canonical_contract_signal_is_explicitly_non_canonical() {
        assert!(!blue_brain_first_region_is_canonical_contract_signal(
            BlueBrainFirstRegionContractSignal::NonCanonicalInternalOnly
        ));
    }

    #[test]
    fn region1_maintenance_reference_doc_pins_canonical_maps_and_boundaries() {
        let doc = include_str!(
            "../../../docs/blue_brain_region1_maintenance_reference_surface_serie_bb25_prompt2_v1.md"
        );
        assert!(doc.contains("Canonical Region-1 Maintenance Reference Map"));
        assert!(doc.contains("Canonical region-1 test surface"));
        assert!(doc.contains("Maintenance-facing index/reference path"));
        assert!(doc.contains("Non-canonical/internal-only or legacy region-1 path"));
        assert!(doc.contains("region-2-not-opened"));
        assert!(doc.contains("docs/roadmap/REPO_MAP.md"));
        assert!(doc.contains("first_region_stabilization_map_contains_required_classes"));
    }

    #[test]
    fn docs_readme_exposes_region1_maintenance_entrypoint() {
        let doc = include_str!("../../../docs/README.md");
        assert!(doc.contains("Region-1 maintenance reference surface (BB25)"));
        assert!(doc.contains(
            "docs/blue_brain_region1_maintenance_reference_surface_serie_bb25_prompt2_v1.md"
        ));
        assert!(doc.contains(
            "docs/blue_brain_region1_final_stabilization_sweep_serie_bb25_prompt3_v1.md"
        ));
    }

    #[test]
    fn region1_final_stabilization_sweep_doc_pins_status_classes_and_decision() {
        let doc = include_str!(
            "../../../docs/blue_brain_region1_final_stabilization_sweep_serie_bb25_prompt3_v1.md"
        );
        assert!(doc.contains("Stable maintenance-hardened region-1 baseline"));
        assert!(doc.contains("Usable with caveats"));
        assert!(doc.contains("Advisory-only"));
        assert!(doc.contains("Diagnostic-only / deferred"));
        assert!(doc.contains("Non-canonical / internal-only"));
        assert!(doc.contains("Maintenance genügt"));
        assert!(doc.contains("expliziter Region-2-Re-Scope"));
        assert!(doc.contains("NotOpenedYetExplicitRescopeRequired"));
    }

    #[test]
    fn roadmap_map_exposes_region1_maintenance_entrypoint_and_boundary() {
        let doc = include_str!("../../../docs/roadmap/REPO_MAP.md");
        assert!(doc.contains("Region-1 maintenance reference surface (BB25)"));
        assert!(doc.contains(
            "docs/blue_brain_region1_maintenance_reference_surface_serie_bb25_prompt2_v1.md"
        ));
        assert!(doc.contains(
            "docs/blue_brain_region1_final_stabilization_sweep_serie_bb25_prompt3_v1.md"
        ));
        assert!(doc.contains("Region 1 bleibt die einzige geöffnete Regionenklasse"));
    }

    #[test]
    fn second_region_selection_doc_pins_candidate_statuses_and_guards() {
        let doc = include_str!(
            "../../../docs/blue_brain_second_region_selection_serie_bb26_prompt1_v1.md"
        );
        assert!(doc.contains("Second-expansion candidate"));
        assert!(doc.contains("Memory/Context-related"));
        assert!(doc.contains("Viable but not second"));
        assert!(doc.contains("Later-phase candidate"));
        assert!(doc.contains("Simulation-only/deferred candidate"));
        assert!(doc.contains("Non-canonical/internal-only path"));
        assert!(doc.contains("keine direkte Action-/Retry-/Memory-/Compute-Autorität"));
        assert!(doc.contains("keine Öffnung einer dritten Regionenklasse"));
    }
}
