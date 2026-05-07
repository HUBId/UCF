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
    MaintenanceHardenedModelBoundary,
    NonCanonicalInternalOnlyResidualPath,
}

pub const CANONICAL_BLUE_BRAIN_FIRST_REGION_STABILIZATION_MAP:
    [BlueBrainFirstRegionStabilizationClass; 6] = [
    BlueBrainFirstRegionStabilizationClass::StableFirstRegionBaseline,
    BlueBrainFirstRegionStabilizationClass::MaintenanceHardenedRegionSurface,
    BlueBrainFirstRegionStabilizationClass::MaintenanceHardenedDiagnosticsPath,
    BlueBrainFirstRegionStabilizationClass::MaintenanceHardenedContractPath,
    BlueBrainFirstRegionStabilizationClass::MaintenanceHardenedModelBoundary,
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
pub enum BlueBrainThirdRegionClass {
    RuntimeFeedbackIntegrationRelated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainThirdRegionSelectionClass {
    ThirdExpansionCandidate,
    ViableButNotThird,
    LaterPhaseCandidate,
    SimulationOnlyDeferredCandidate,
    NonCanonicalInternalOnlyPath,
}

pub const CANONICAL_BLUE_BRAIN_THIRD_REGION_SELECTION_MAP: [BlueBrainThirdRegionSelectionClass; 5] = [
    BlueBrainThirdRegionSelectionClass::ThirdExpansionCandidate,
    BlueBrainThirdRegionSelectionClass::ViableButNotThird,
    BlueBrainThirdRegionSelectionClass::LaterPhaseCandidate,
    BlueBrainThirdRegionSelectionClass::SimulationOnlyDeferredCandidate,
    BlueBrainThirdRegionSelectionClass::NonCanonicalInternalOnlyPath,
];

pub const BLUE_BRAIN_THIRD_REGION_CLASS_SELECTION: BlueBrainThirdRegionClass =
    BlueBrainThirdRegionClass::RuntimeFeedbackIntegrationRelated;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFirstAnatomicalRegion {
    HippocampusLikeRegion,
}

pub const BLUE_BRAIN_FIRST_ANATOMICAL_REGION_SELECTION: BlueBrainFirstAnatomicalRegion =
    BlueBrainFirstAnatomicalRegion::HippocampusLikeRegion;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFirstAnatomicalRegionPathClass {
    AnatomicalRegionInputSurface,
    AnatomicalRegionStateSurface,
    AnatomicalRegionOutputAdvisorySurface,
    AnatomicalRegionReferenceSurface,
    AnatomicalToFunctionalRegionMapping,
    BlockedDeferredAnatomicalRegionPath,
    NonCanonicalInternalOnlyAnatomicalRegionPath,
}

pub const CANONICAL_BLUE_BRAIN_FIRST_ANATOMICAL_REGION_INTEGRATION_MAP:
    [BlueBrainFirstAnatomicalRegionPathClass; 7] = [
    BlueBrainFirstAnatomicalRegionPathClass::AnatomicalRegionInputSurface,
    BlueBrainFirstAnatomicalRegionPathClass::AnatomicalRegionStateSurface,
    BlueBrainFirstAnatomicalRegionPathClass::AnatomicalRegionOutputAdvisorySurface,
    BlueBrainFirstAnatomicalRegionPathClass::AnatomicalRegionReferenceSurface,
    BlueBrainFirstAnatomicalRegionPathClass::AnatomicalToFunctionalRegionMapping,
    BlueBrainFirstAnatomicalRegionPathClass::BlockedDeferredAnatomicalRegionPath,
    BlueBrainFirstAnatomicalRegionPathClass::NonCanonicalInternalOnlyAnatomicalRegionPath,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFirstAnatomicalRegionModelModeClass {
    AbstractFunctionalCurrentMode,
    BoundedKuramotoLikeCurrentMode,
    HodgkinHuxleySimulationOnlyDiagnosticOnlyCurrentMode,
    LaterSelectiveHodgkinHuxleyDeepening,
    DeferredNotSuitableNowModelPath,
    NonCanonicalInternalOnlyModelPath,
}

pub const CANONICAL_BLUE_BRAIN_FIRST_ANATOMICAL_REGION_MODEL_DECISION_MAP:
    [BlueBrainFirstAnatomicalRegionModelModeClass; 6] = [
    BlueBrainFirstAnatomicalRegionModelModeClass::AbstractFunctionalCurrentMode,
    BlueBrainFirstAnatomicalRegionModelModeClass::BoundedKuramotoLikeCurrentMode,
    BlueBrainFirstAnatomicalRegionModelModeClass::HodgkinHuxleySimulationOnlyDiagnosticOnlyCurrentMode,
    BlueBrainFirstAnatomicalRegionModelModeClass::LaterSelectiveHodgkinHuxleyDeepening,
    BlueBrainFirstAnatomicalRegionModelModeClass::DeferredNotSuitableNowModelPath,
    BlueBrainFirstAnatomicalRegionModelModeClass::NonCanonicalInternalOnlyModelPath,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainHippocampusIntegrationClass {
    HippocampusInputSurface,
    HippocampusStateSurface,
    HippocampusOutputAdvisorySurface,
    HippocampusReferenceSurface,
    BlockedDeferredHippocampusPath,
    NonCanonicalInternalOnlyHippocampusPath,
}

pub const CANONICAL_BLUE_BRAIN_HIPPOCAMPUS_INTEGRATION_MAP: [BlueBrainHippocampusIntegrationClass;
    6] = [
    BlueBrainHippocampusIntegrationClass::HippocampusInputSurface,
    BlueBrainHippocampusIntegrationClass::HippocampusStateSurface,
    BlueBrainHippocampusIntegrationClass::HippocampusOutputAdvisorySurface,
    BlueBrainHippocampusIntegrationClass::HippocampusReferenceSurface,
    BlueBrainHippocampusIntegrationClass::BlockedDeferredHippocampusPath,
    BlueBrainHippocampusIntegrationClass::NonCanonicalInternalOnlyHippocampusPath,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainThalamusIntegrationClass {
    ThalamusInputSurface,
    ThalamusStateSurface,
    ThalamusOutputAdvisorySurface,
    ThalamusReferenceSurface,
    BlockedDeferredThalamusPath,
    NonCanonicalInternalOnlyThalamusPath,
}

pub const CANONICAL_BLUE_BRAIN_THALAMUS_INTEGRATION_MAP: [BlueBrainThalamusIntegrationClass; 6] = [
    BlueBrainThalamusIntegrationClass::ThalamusInputSurface,
    BlueBrainThalamusIntegrationClass::ThalamusStateSurface,
    BlueBrainThalamusIntegrationClass::ThalamusOutputAdvisorySurface,
    BlueBrainThalamusIntegrationClass::ThalamusReferenceSurface,
    BlueBrainThalamusIntegrationClass::BlockedDeferredThalamusPath,
    BlueBrainThalamusIntegrationClass::NonCanonicalInternalOnlyThalamusPath,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainBasalGangliaRoleClass {
    ActionGatingRole,
    SuppressionInhibitionRole,
    BoundedSelectionChannelArbitrationRole,
    ExecutionReadinessModulationRole,
    NonRoleOutOfScopeBiologicalDetail,
}

pub const CANONICAL_BLUE_BRAIN_BASAL_GANGLIA_ROLE_MAP: [BlueBrainBasalGangliaRoleClass; 5] = [
    BlueBrainBasalGangliaRoleClass::ActionGatingRole,
    BlueBrainBasalGangliaRoleClass::SuppressionInhibitionRole,
    BlueBrainBasalGangliaRoleClass::BoundedSelectionChannelArbitrationRole,
    BlueBrainBasalGangliaRoleClass::ExecutionReadinessModulationRole,
    BlueBrainBasalGangliaRoleClass::NonRoleOutOfScopeBiologicalDetail,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainBasalGangliaIntegrationClass {
    BasalGangliaInputSurface,
    BasalGangliaStateSurface,
    BasalGangliaOutputAdvisorySurface,
    BasalGangliaReferenceBoundedSurface,
    BlockedDeferredBasalGangliaPath,
    NonCanonicalInternalOnlyBasalGangliaPath,
}

pub const CANONICAL_BLUE_BRAIN_BASAL_GANGLIA_INTEGRATION_MAP:
    [BlueBrainBasalGangliaIntegrationClass; 6] = [
    BlueBrainBasalGangliaIntegrationClass::BasalGangliaInputSurface,
    BlueBrainBasalGangliaIntegrationClass::BasalGangliaStateSurface,
    BlueBrainBasalGangliaIntegrationClass::BasalGangliaOutputAdvisorySurface,
    BlueBrainBasalGangliaIntegrationClass::BasalGangliaReferenceBoundedSurface,
    BlueBrainBasalGangliaIntegrationClass::BlockedDeferredBasalGangliaPath,
    BlueBrainBasalGangliaIntegrationClass::NonCanonicalInternalOnlyBasalGangliaPath,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainCerebellumRoleClass {
    PredictionRole,
    TimingCoordinationRole,
    ErrorCorrectionMismatchShapingRole,
    BoundedExecutionSupportRole,
    NonRoleOutOfScopeBiologicalDetail,
}

pub const CANONICAL_BLUE_BRAIN_CEREBELLUM_ROLE_MAP: [BlueBrainCerebellumRoleClass; 5] = [
    BlueBrainCerebellumRoleClass::PredictionRole,
    BlueBrainCerebellumRoleClass::TimingCoordinationRole,
    BlueBrainCerebellumRoleClass::ErrorCorrectionMismatchShapingRole,
    BlueBrainCerebellumRoleClass::BoundedExecutionSupportRole,
    BlueBrainCerebellumRoleClass::NonRoleOutOfScopeBiologicalDetail,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainCerebellumIntegrationClass {
    CerebellumInputSurface,
    CerebellumStateSurface,
    CerebellumOutputAdvisorySurface,
    CerebellumReferenceSurface,
    BlockedDeferredCerebellumPath,
    NonCanonicalInternalOnlyCerebellumPath,
}

pub const CANONICAL_BLUE_BRAIN_CEREBELLUM_INTEGRATION_MAP: [BlueBrainCerebellumIntegrationClass;
    6] = [
    BlueBrainCerebellumIntegrationClass::CerebellumInputSurface,
    BlueBrainCerebellumIntegrationClass::CerebellumStateSurface,
    BlueBrainCerebellumIntegrationClass::CerebellumOutputAdvisorySurface,
    BlueBrainCerebellumIntegrationClass::CerebellumReferenceSurface,
    BlueBrainCerebellumIntegrationClass::BlockedDeferredCerebellumPath,
    BlueBrainCerebellumIntegrationClass::NonCanonicalInternalOnlyCerebellumPath,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainCerebellumExpansionReadinessClass {
    StableCerebellumOperationalSurface,
    UsableWithCaveats,
    AdvisoryOnly,
    DeferredBlockedInsufficientDiagnosticOnlyReferenceOnly,
    StableCurrentModelMode,
    NonCanonicalInternalOnly,
}

pub const CANONICAL_BLUE_BRAIN_CEREBELLUM_EXPANSION_READINESS_MAP:
    [BlueBrainCerebellumExpansionReadinessClass; 6] = [
    BlueBrainCerebellumExpansionReadinessClass::StableCerebellumOperationalSurface,
    BlueBrainCerebellumExpansionReadinessClass::UsableWithCaveats,
    BlueBrainCerebellumExpansionReadinessClass::AdvisoryOnly,
    BlueBrainCerebellumExpansionReadinessClass::DeferredBlockedInsufficientDiagnosticOnlyReferenceOnly,
    BlueBrainCerebellumExpansionReadinessClass::StableCurrentModelMode,
    BlueBrainCerebellumExpansionReadinessClass::NonCanonicalInternalOnly,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainPostBr5NextDirection {
    Hypothalamus,
    InterRegionArchitectureStage,
}

pub const BLUE_BRAIN_POST_BR5_PRIORITIZED_NEXT_DIRECTION: BlueBrainPostBr5NextDirection =
    BlueBrainPostBr5NextDirection::InterRegionArchitectureStage;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainPostIr1NextDirection {
    Hypothalamus,
    SelectiveModelDeepening,
}

pub const BLUE_BRAIN_POST_IR1_PRIORITIZED_NEXT_DIRECTION: BlueBrainPostIr1NextDirection =
    BlueBrainPostIr1NextDirection::SelectiveModelDeepening;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainIr1ReadinessClass {
    StableImplementedRelation,
    UsableWithCaveats,
    AdvisoryOnly,
    DeferredNotYetActive,
    BlockedInsufficientDiagnosticOnly,
    NonCanonicalInternalOnly,
}

pub const CANONICAL_BLUE_BRAIN_IR1_READINESS_CLASS_MAP: [BlueBrainIr1ReadinessClass; 6] = [
    BlueBrainIr1ReadinessClass::StableImplementedRelation,
    BlueBrainIr1ReadinessClass::UsableWithCaveats,
    BlueBrainIr1ReadinessClass::AdvisoryOnly,
    BlueBrainIr1ReadinessClass::DeferredNotYetActive,
    BlueBrainIr1ReadinessClass::BlockedInsufficientDiagnosticOnly,
    BlueBrainIr1ReadinessClass::NonCanonicalInternalOnly,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainInterRegionArchitectureRegionRoleClass {
    ContextReferenceEpisodeIndexing,
    SaliencePriorityCaveat,
    RelayGatingRouting,
    ActionChannelSuppression,
    TimingPredictionCorrection,
    NonCanonicalDeferredRegionRole,
}

pub fn blue_brain_inter_region_architecture_region_role(
    region: BlueBrainAnatomicalRegionClass,
) -> BlueBrainInterRegionArchitectureRegionRoleClass {
    match region {
        BlueBrainAnatomicalRegionClass::Hippocampus => {
            BlueBrainInterRegionArchitectureRegionRoleClass::ContextReferenceEpisodeIndexing
        }
        BlueBrainAnatomicalRegionClass::Amygdala => {
            BlueBrainInterRegionArchitectureRegionRoleClass::SaliencePriorityCaveat
        }
        BlueBrainAnatomicalRegionClass::Thalamus => {
            BlueBrainInterRegionArchitectureRegionRoleClass::RelayGatingRouting
        }
        BlueBrainAnatomicalRegionClass::BasalGanglia => {
            BlueBrainInterRegionArchitectureRegionRoleClass::ActionChannelSuppression
        }
        BlueBrainAnatomicalRegionClass::Cerebellum => {
            BlueBrainInterRegionArchitectureRegionRoleClass::TimingPredictionCorrection
        }
        BlueBrainAnatomicalRegionClass::PrefrontalCortex
        | BlueBrainAnatomicalRegionClass::AnteriorCingulateCortex
        | BlueBrainAnatomicalRegionClass::Insula => {
            BlueBrainInterRegionArchitectureRegionRoleClass::NonCanonicalDeferredRegionRole
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainInterRegionArchitectureRelationClass {
    DirectBoundedAdvisoryRelation,
    ReferenceMediatedRelation,
    SelectionMediatedRelation,
    ExecutionInterfaceMediatedRelation,
    CaveatedInterRegionRelation,
    DeferredNotYetActiveRelation,
    BlockedRelation,
    NonCanonicalInternalOnlyRelationPath,
}

pub const CANONICAL_BLUE_BRAIN_INTER_REGION_ARCHITECTURE_RELATION_CLASS_MAP:
    [BlueBrainInterRegionArchitectureRelationClass; 8] = [
    BlueBrainInterRegionArchitectureRelationClass::DirectBoundedAdvisoryRelation,
    BlueBrainInterRegionArchitectureRelationClass::ReferenceMediatedRelation,
    BlueBrainInterRegionArchitectureRelationClass::SelectionMediatedRelation,
    BlueBrainInterRegionArchitectureRelationClass::ExecutionInterfaceMediatedRelation,
    BlueBrainInterRegionArchitectureRelationClass::CaveatedInterRegionRelation,
    BlueBrainInterRegionArchitectureRelationClass::DeferredNotYetActiveRelation,
    BlueBrainInterRegionArchitectureRelationClass::BlockedRelation,
    BlueBrainInterRegionArchitectureRelationClass::NonCanonicalInternalOnlyRelationPath,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainInterRegionArchitecturePair {
    HippocampusAmygdala,
    HippocampusThalamus,
    HippocampusBasalGanglia,
    HippocampusCerebellum,
    AmygdalaThalamus,
    AmygdalaBasalGanglia,
    AmygdalaCerebellum,
    ThalamusBasalGanglia,
    ThalamusCerebellum,
    BasalGangliaCerebellum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainInterRegionArchitectureRelation {
    pub pair: BlueBrainInterRegionArchitecturePair,
    pub relation_class: BlueBrainInterRegionArchitectureRelationClass,
    pub source_role: BlueBrainInterRegionArchitectureRegionRoleClass,
    pub target_role: BlueBrainInterRegionArchitectureRegionRoleClass,
    pub advisory_only: bool,
    pub reference_mediated_only: bool,
    pub selection_mediated_only: bool,
    pub execution_interface_mediated_only: bool,
    pub caveated: bool,
    pub deferred: bool,
    pub blocked: bool,
    pub direct_action_trigger: bool,
    pub direct_execution_trigger: bool,
    pub direct_retry_trigger: bool,
    pub direct_memory_commit: bool,
    pub direct_compute_invocation: bool,
    pub safety_override: bool,
    pub global_region_orchestration: bool,
}

const fn blue_brain_inter_region_architecture_relation(
    pair: BlueBrainInterRegionArchitecturePair,
    relation_class: BlueBrainInterRegionArchitectureRelationClass,
    source_role: BlueBrainInterRegionArchitectureRegionRoleClass,
    target_role: BlueBrainInterRegionArchitectureRegionRoleClass,
) -> BlueBrainInterRegionArchitectureRelation {
    BlueBrainInterRegionArchitectureRelation {
        pair,
        relation_class,
        source_role,
        target_role,
        advisory_only: true,
        reference_mediated_only: matches!(
            relation_class,
            BlueBrainInterRegionArchitectureRelationClass::ReferenceMediatedRelation
        ),
        selection_mediated_only: matches!(
            relation_class,
            BlueBrainInterRegionArchitectureRelationClass::SelectionMediatedRelation
        ),
        execution_interface_mediated_only: matches!(
            relation_class,
            BlueBrainInterRegionArchitectureRelationClass::ExecutionInterfaceMediatedRelation
        ),
        caveated: matches!(
            relation_class,
            BlueBrainInterRegionArchitectureRelationClass::CaveatedInterRegionRelation
        ),
        deferred: matches!(
            relation_class,
            BlueBrainInterRegionArchitectureRelationClass::DeferredNotYetActiveRelation
        ),
        blocked: matches!(
            relation_class,
            BlueBrainInterRegionArchitectureRelationClass::BlockedRelation
                | BlueBrainInterRegionArchitectureRelationClass::NonCanonicalInternalOnlyRelationPath
        ),
        direct_action_trigger: false,
        direct_execution_trigger: false,
        direct_retry_trigger: false,
        direct_memory_commit: false,
        direct_compute_invocation: false,
        safety_override: false,
        global_region_orchestration: false,
    }
}

pub const CANONICAL_BLUE_BRAIN_INTER_REGION_ARCHITECTURE_MAP:
    [BlueBrainInterRegionArchitectureRelation; 10] = [
    blue_brain_inter_region_architecture_relation(
        BlueBrainInterRegionArchitecturePair::HippocampusAmygdala,
        BlueBrainInterRegionArchitectureRelationClass::CaveatedInterRegionRelation,
        BlueBrainInterRegionArchitectureRegionRoleClass::ContextReferenceEpisodeIndexing,
        BlueBrainInterRegionArchitectureRegionRoleClass::SaliencePriorityCaveat,
    ),
    blue_brain_inter_region_architecture_relation(
        BlueBrainInterRegionArchitecturePair::HippocampusThalamus,
        BlueBrainInterRegionArchitectureRelationClass::ReferenceMediatedRelation,
        BlueBrainInterRegionArchitectureRegionRoleClass::ContextReferenceEpisodeIndexing,
        BlueBrainInterRegionArchitectureRegionRoleClass::RelayGatingRouting,
    ),
    blue_brain_inter_region_architecture_relation(
        BlueBrainInterRegionArchitecturePair::HippocampusBasalGanglia,
        BlueBrainInterRegionArchitectureRelationClass::BlockedRelation,
        BlueBrainInterRegionArchitectureRegionRoleClass::ContextReferenceEpisodeIndexing,
        BlueBrainInterRegionArchitectureRegionRoleClass::ActionChannelSuppression,
    ),
    blue_brain_inter_region_architecture_relation(
        BlueBrainInterRegionArchitecturePair::HippocampusCerebellum,
        BlueBrainInterRegionArchitectureRelationClass::ReferenceMediatedRelation,
        BlueBrainInterRegionArchitectureRegionRoleClass::ContextReferenceEpisodeIndexing,
        BlueBrainInterRegionArchitectureRegionRoleClass::TimingPredictionCorrection,
    ),
    blue_brain_inter_region_architecture_relation(
        BlueBrainInterRegionArchitecturePair::AmygdalaThalamus,
        BlueBrainInterRegionArchitectureRelationClass::DirectBoundedAdvisoryRelation,
        BlueBrainInterRegionArchitectureRegionRoleClass::SaliencePriorityCaveat,
        BlueBrainInterRegionArchitectureRegionRoleClass::RelayGatingRouting,
    ),
    blue_brain_inter_region_architecture_relation(
        BlueBrainInterRegionArchitecturePair::AmygdalaBasalGanglia,
        BlueBrainInterRegionArchitectureRelationClass::SelectionMediatedRelation,
        BlueBrainInterRegionArchitectureRegionRoleClass::SaliencePriorityCaveat,
        BlueBrainInterRegionArchitectureRegionRoleClass::ActionChannelSuppression,
    ),
    blue_brain_inter_region_architecture_relation(
        BlueBrainInterRegionArchitecturePair::AmygdalaCerebellum,
        BlueBrainInterRegionArchitectureRelationClass::DeferredNotYetActiveRelation,
        BlueBrainInterRegionArchitectureRegionRoleClass::SaliencePriorityCaveat,
        BlueBrainInterRegionArchitectureRegionRoleClass::TimingPredictionCorrection,
    ),
    blue_brain_inter_region_architecture_relation(
        BlueBrainInterRegionArchitecturePair::ThalamusBasalGanglia,
        BlueBrainInterRegionArchitectureRelationClass::SelectionMediatedRelation,
        BlueBrainInterRegionArchitectureRegionRoleClass::RelayGatingRouting,
        BlueBrainInterRegionArchitectureRegionRoleClass::ActionChannelSuppression,
    ),
    blue_brain_inter_region_architecture_relation(
        BlueBrainInterRegionArchitecturePair::ThalamusCerebellum,
        BlueBrainInterRegionArchitectureRelationClass::DirectBoundedAdvisoryRelation,
        BlueBrainInterRegionArchitectureRegionRoleClass::RelayGatingRouting,
        BlueBrainInterRegionArchitectureRegionRoleClass::TimingPredictionCorrection,
    ),
    blue_brain_inter_region_architecture_relation(
        BlueBrainInterRegionArchitecturePair::BasalGangliaCerebellum,
        BlueBrainInterRegionArchitectureRelationClass::ExecutionInterfaceMediatedRelation,
        BlueBrainInterRegionArchitectureRegionRoleClass::ActionChannelSuppression,
        BlueBrainInterRegionArchitectureRegionRoleClass::TimingPredictionCorrection,
    ),
];

pub fn blue_brain_inter_region_architecture_relation_for_pair(
    pair: BlueBrainInterRegionArchitecturePair,
) -> BlueBrainInterRegionArchitectureRelation {
    CANONICAL_BLUE_BRAIN_INTER_REGION_ARCHITECTURE_MAP
        .iter()
        .copied()
        .find(|relation| relation.pair == pair)
        .unwrap_or_else(|| {
            blue_brain_inter_region_architecture_relation(
                pair,
                BlueBrainInterRegionArchitectureRelationClass::NonCanonicalInternalOnlyRelationPath,
                BlueBrainInterRegionArchitectureRegionRoleClass::NonCanonicalDeferredRegionRole,
                BlueBrainInterRegionArchitectureRegionRoleClass::NonCanonicalDeferredRegionRole,
            )
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainInterRegionImplementationRelationClass {
    ImplementedDirectBoundedAdvisoryRelation,
    ImplementedReferenceMediatedRelation,
    ImplementedSelectionMediatedRelation,
    DeferredNotYetImplementedRelation,
    BlockedRelation,
    NonCanonicalInternalOnlyRelationPath,
}

pub const CANONICAL_BLUE_BRAIN_INTER_REGION_IMPLEMENTATION_RELATION_CLASS_MAP:
    [BlueBrainInterRegionImplementationRelationClass; 6] = [
    BlueBrainInterRegionImplementationRelationClass::ImplementedDirectBoundedAdvisoryRelation,
    BlueBrainInterRegionImplementationRelationClass::ImplementedReferenceMediatedRelation,
    BlueBrainInterRegionImplementationRelationClass::ImplementedSelectionMediatedRelation,
    BlueBrainInterRegionImplementationRelationClass::DeferredNotYetImplementedRelation,
    BlueBrainInterRegionImplementationRelationClass::BlockedRelation,
    BlueBrainInterRegionImplementationRelationClass::NonCanonicalInternalOnlyRelationPath,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainInterRegionImplementationSignal {
    SalienceCaveatAdvisory,
    RelayRoutingDiagnostic,
    ContextReferenceDiagnostic,
    SelectionReadinessDiagnostic,
    DeferredDiagnosticOnly,
    BlockedDiagnosticOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainInterRegionImplementationMediationPath {
    DirectBoundedAdvisoryOnly,
    ReferenceContextMediatedOnly,
    SelectionContractMediatedOnly,
    NotYetImplemented,
    BlockedUnavailable,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainInterRegionImplementationRelation {
    pub pair: BlueBrainInterRegionArchitecturePair,
    pub architecture_relation_class: BlueBrainInterRegionArchitectureRelationClass,
    pub implementation_relation_class: BlueBrainInterRegionImplementationRelationClass,
    pub source_role: BlueBrainInterRegionArchitectureRegionRoleClass,
    pub target_role: BlueBrainInterRegionArchitectureRegionRoleClass,
    pub bidirectional_pair_label: bool,
    pub source_to_target_signal: BlueBrainInterRegionImplementationSignal,
    pub target_to_source_signal: BlueBrainInterRegionImplementationSignal,
    pub mediation_path: BlueBrainInterRegionImplementationMediationPath,
    pub advisory_only: bool,
    pub direct_action_trigger: bool,
    pub direct_execution_trigger: bool,
    pub direct_retry_trigger: bool,
    pub direct_memory_commit: bool,
    pub direct_compute_invocation: bool,
    pub safety_override: bool,
    pub global_region_orchestration: bool,
}

const fn blue_brain_inter_region_implementation_relation(
    pair: BlueBrainInterRegionArchitecturePair,
    implementation_relation_class: BlueBrainInterRegionImplementationRelationClass,
    source_to_target_signal: BlueBrainInterRegionImplementationSignal,
    target_to_source_signal: BlueBrainInterRegionImplementationSignal,
    mediation_path: BlueBrainInterRegionImplementationMediationPath,
) -> BlueBrainInterRegionImplementationRelation {
    let architecture_relation = blue_brain_inter_region_architecture_relation_for_pair_const(pair);

    BlueBrainInterRegionImplementationRelation {
        pair,
        architecture_relation_class: architecture_relation.relation_class,
        implementation_relation_class,
        source_role: architecture_relation.source_role,
        target_role: architecture_relation.target_role,
        bidirectional_pair_label: true,
        source_to_target_signal,
        target_to_source_signal,
        mediation_path,
        advisory_only: true,
        direct_action_trigger: false,
        direct_execution_trigger: false,
        direct_retry_trigger: false,
        direct_memory_commit: false,
        direct_compute_invocation: false,
        safety_override: false,
        global_region_orchestration: false,
    }
}

const fn blue_brain_inter_region_architecture_relation_for_pair_const(
    pair: BlueBrainInterRegionArchitecturePair,
) -> BlueBrainInterRegionArchitectureRelation {
    match pair {
        BlueBrainInterRegionArchitecturePair::HippocampusAmygdala => {
            CANONICAL_BLUE_BRAIN_INTER_REGION_ARCHITECTURE_MAP[0]
        }
        BlueBrainInterRegionArchitecturePair::HippocampusThalamus => {
            CANONICAL_BLUE_BRAIN_INTER_REGION_ARCHITECTURE_MAP[1]
        }
        BlueBrainInterRegionArchitecturePair::HippocampusBasalGanglia => {
            CANONICAL_BLUE_BRAIN_INTER_REGION_ARCHITECTURE_MAP[2]
        }
        BlueBrainInterRegionArchitecturePair::HippocampusCerebellum => {
            CANONICAL_BLUE_BRAIN_INTER_REGION_ARCHITECTURE_MAP[3]
        }
        BlueBrainInterRegionArchitecturePair::AmygdalaThalamus => {
            CANONICAL_BLUE_BRAIN_INTER_REGION_ARCHITECTURE_MAP[4]
        }
        BlueBrainInterRegionArchitecturePair::AmygdalaBasalGanglia => {
            CANONICAL_BLUE_BRAIN_INTER_REGION_ARCHITECTURE_MAP[5]
        }
        BlueBrainInterRegionArchitecturePair::AmygdalaCerebellum => {
            CANONICAL_BLUE_BRAIN_INTER_REGION_ARCHITECTURE_MAP[6]
        }
        BlueBrainInterRegionArchitecturePair::ThalamusBasalGanglia => {
            CANONICAL_BLUE_BRAIN_INTER_REGION_ARCHITECTURE_MAP[7]
        }
        BlueBrainInterRegionArchitecturePair::ThalamusCerebellum => {
            CANONICAL_BLUE_BRAIN_INTER_REGION_ARCHITECTURE_MAP[8]
        }
        BlueBrainInterRegionArchitecturePair::BasalGangliaCerebellum => {
            CANONICAL_BLUE_BRAIN_INTER_REGION_ARCHITECTURE_MAP[9]
        }
    }
}

pub const CANONICAL_BLUE_BRAIN_FIRST_INTER_REGION_IMPLEMENTATION_MAP:
    [BlueBrainInterRegionImplementationRelation; 10] = [
    blue_brain_inter_region_implementation_relation(
        BlueBrainInterRegionArchitecturePair::HippocampusAmygdala,
        BlueBrainInterRegionImplementationRelationClass::DeferredNotYetImplementedRelation,
        BlueBrainInterRegionImplementationSignal::DeferredDiagnosticOnly,
        BlueBrainInterRegionImplementationSignal::DeferredDiagnosticOnly,
        BlueBrainInterRegionImplementationMediationPath::NotYetImplemented,
    ),
    blue_brain_inter_region_implementation_relation(
        BlueBrainInterRegionArchitecturePair::HippocampusThalamus,
        BlueBrainInterRegionImplementationRelationClass::ImplementedReferenceMediatedRelation,
        BlueBrainInterRegionImplementationSignal::ContextReferenceDiagnostic,
        BlueBrainInterRegionImplementationSignal::RelayRoutingDiagnostic,
        BlueBrainInterRegionImplementationMediationPath::ReferenceContextMediatedOnly,
    ),
    blue_brain_inter_region_implementation_relation(
        BlueBrainInterRegionArchitecturePair::HippocampusBasalGanglia,
        BlueBrainInterRegionImplementationRelationClass::BlockedRelation,
        BlueBrainInterRegionImplementationSignal::BlockedDiagnosticOnly,
        BlueBrainInterRegionImplementationSignal::BlockedDiagnosticOnly,
        BlueBrainInterRegionImplementationMediationPath::BlockedUnavailable,
    ),
    blue_brain_inter_region_implementation_relation(
        BlueBrainInterRegionArchitecturePair::HippocampusCerebellum,
        BlueBrainInterRegionImplementationRelationClass::DeferredNotYetImplementedRelation,
        BlueBrainInterRegionImplementationSignal::DeferredDiagnosticOnly,
        BlueBrainInterRegionImplementationSignal::DeferredDiagnosticOnly,
        BlueBrainInterRegionImplementationMediationPath::NotYetImplemented,
    ),
    blue_brain_inter_region_implementation_relation(
        BlueBrainInterRegionArchitecturePair::AmygdalaThalamus,
        BlueBrainInterRegionImplementationRelationClass::ImplementedDirectBoundedAdvisoryRelation,
        BlueBrainInterRegionImplementationSignal::SalienceCaveatAdvisory,
        BlueBrainInterRegionImplementationSignal::RelayRoutingDiagnostic,
        BlueBrainInterRegionImplementationMediationPath::DirectBoundedAdvisoryOnly,
    ),
    blue_brain_inter_region_implementation_relation(
        BlueBrainInterRegionArchitecturePair::AmygdalaBasalGanglia,
        BlueBrainInterRegionImplementationRelationClass::ImplementedSelectionMediatedRelation,
        BlueBrainInterRegionImplementationSignal::SalienceCaveatAdvisory,
        BlueBrainInterRegionImplementationSignal::SelectionReadinessDiagnostic,
        BlueBrainInterRegionImplementationMediationPath::SelectionContractMediatedOnly,
    ),
    blue_brain_inter_region_implementation_relation(
        BlueBrainInterRegionArchitecturePair::AmygdalaCerebellum,
        BlueBrainInterRegionImplementationRelationClass::DeferredNotYetImplementedRelation,
        BlueBrainInterRegionImplementationSignal::DeferredDiagnosticOnly,
        BlueBrainInterRegionImplementationSignal::DeferredDiagnosticOnly,
        BlueBrainInterRegionImplementationMediationPath::NotYetImplemented,
    ),
    blue_brain_inter_region_implementation_relation(
        BlueBrainInterRegionArchitecturePair::ThalamusBasalGanglia,
        BlueBrainInterRegionImplementationRelationClass::DeferredNotYetImplementedRelation,
        BlueBrainInterRegionImplementationSignal::DeferredDiagnosticOnly,
        BlueBrainInterRegionImplementationSignal::DeferredDiagnosticOnly,
        BlueBrainInterRegionImplementationMediationPath::NotYetImplemented,
    ),
    blue_brain_inter_region_implementation_relation(
        BlueBrainInterRegionArchitecturePair::ThalamusCerebellum,
        BlueBrainInterRegionImplementationRelationClass::DeferredNotYetImplementedRelation,
        BlueBrainInterRegionImplementationSignal::DeferredDiagnosticOnly,
        BlueBrainInterRegionImplementationSignal::DeferredDiagnosticOnly,
        BlueBrainInterRegionImplementationMediationPath::NotYetImplemented,
    ),
    blue_brain_inter_region_implementation_relation(
        BlueBrainInterRegionArchitecturePair::BasalGangliaCerebellum,
        BlueBrainInterRegionImplementationRelationClass::DeferredNotYetImplementedRelation,
        BlueBrainInterRegionImplementationSignal::DeferredDiagnosticOnly,
        BlueBrainInterRegionImplementationSignal::DeferredDiagnosticOnly,
        BlueBrainInterRegionImplementationMediationPath::NotYetImplemented,
    ),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainInterRegionDiagnosticsContractClass {
    AdvisoryOnlyRelationDiagnostic,
    CaveatedRelationDiagnostic,
    DeferredRelationDiagnostic,
    BlockedRelationDiagnostic,
    InsufficientRelationDiagnostic,
    DiagnosticOnlyRelationState,
    BoundedRelationContractSignal,
    NonCanonicalInternalOnlyRelationPath,
}

pub const CANONICAL_BLUE_BRAIN_INTER_REGION_DIAGNOSTICS_CONTRACT_CLASS_MAP:
    [BlueBrainInterRegionDiagnosticsContractClass; 8] = [
    BlueBrainInterRegionDiagnosticsContractClass::AdvisoryOnlyRelationDiagnostic,
    BlueBrainInterRegionDiagnosticsContractClass::CaveatedRelationDiagnostic,
    BlueBrainInterRegionDiagnosticsContractClass::DeferredRelationDiagnostic,
    BlueBrainInterRegionDiagnosticsContractClass::BlockedRelationDiagnostic,
    BlueBrainInterRegionDiagnosticsContractClass::InsufficientRelationDiagnostic,
    BlueBrainInterRegionDiagnosticsContractClass::DiagnosticOnlyRelationState,
    BlueBrainInterRegionDiagnosticsContractClass::BoundedRelationContractSignal,
    BlueBrainInterRegionDiagnosticsContractClass::NonCanonicalInternalOnlyRelationPath,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainInterRegionDiagnosticsRelationState {
    AdvisoryOnlyActive,
    CaveatedNoStrongPositiveSignal,
    DeferredNotYetUsable,
    BlockedByContractSafetyOrReference,
    InsufficientRelationalBasis,
    DiagnosticOnlyVisible,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainInterRegionContractSignalClass {
    BoundedRelationContractSignal,
    CaveatedRelationDiagnosticSignal,
    DeferredRelationDiagnosticSignal,
    BlockedRelationDiagnosticSignal,
    InsufficientRelationDiagnosticSignal,
    DiagnosticOnlyRelationSignal,
    NonCanonicalInternalOnlyRelationSignal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainInterRegionConsumerLayer {
    Runtime,
    Selection,
    Reference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainInterRegionDiagnosticsContractRead {
    pub pair: BlueBrainInterRegionArchitecturePair,
    pub consumer_layer: BlueBrainInterRegionConsumerLayer,
    pub implementation_relation_class: BlueBrainInterRegionImplementationRelationClass,
    pub mediation_path: BlueBrainInterRegionImplementationMediationPath,
    pub relation_state: BlueBrainInterRegionDiagnosticsRelationState,
    pub relation_diagnostic_class: BlueBrainInterRegionDiagnosticsContractClass,
    pub contract_signal_class: BlueBrainInterRegionContractSignalClass,
    pub bounded_contract_signal: bool,
    pub advisory_only: bool,
    pub caveated: bool,
    pub deferred: bool,
    pub blocked: bool,
    pub insufficient: bool,
    pub diagnostic_only: bool,
    pub non_canonical_internal_only: bool,
    pub direct_action_trigger: bool,
    pub direct_execution_trigger: bool,
    pub direct_retry_trigger: bool,
    pub direct_memory_commit: bool,
    pub direct_compute_invocation: bool,
    pub safety_override: bool,
    pub global_region_orchestration: bool,
}

const fn blue_brain_inter_region_relation_state_for_implementation_class(
    implementation_relation_class: BlueBrainInterRegionImplementationRelationClass,
) -> BlueBrainInterRegionDiagnosticsRelationState {
    match implementation_relation_class {
        BlueBrainInterRegionImplementationRelationClass::ImplementedDirectBoundedAdvisoryRelation
        | BlueBrainInterRegionImplementationRelationClass::ImplementedReferenceMediatedRelation
        | BlueBrainInterRegionImplementationRelationClass::ImplementedSelectionMediatedRelation => {
            BlueBrainInterRegionDiagnosticsRelationState::AdvisoryOnlyActive
        }
        BlueBrainInterRegionImplementationRelationClass::DeferredNotYetImplementedRelation => {
            BlueBrainInterRegionDiagnosticsRelationState::DeferredNotYetUsable
        }
        BlueBrainInterRegionImplementationRelationClass::BlockedRelation => {
            BlueBrainInterRegionDiagnosticsRelationState::BlockedByContractSafetyOrReference
        }
        BlueBrainInterRegionImplementationRelationClass::NonCanonicalInternalOnlyRelationPath => {
            BlueBrainInterRegionDiagnosticsRelationState::NonCanonicalInternalOnly
        }
    }
}

pub const fn blue_brain_inter_region_diagnostics_contract_class_for_state(
    relation_state: BlueBrainInterRegionDiagnosticsRelationState,
) -> BlueBrainInterRegionDiagnosticsContractClass {
    match relation_state {
        BlueBrainInterRegionDiagnosticsRelationState::AdvisoryOnlyActive => {
            BlueBrainInterRegionDiagnosticsContractClass::AdvisoryOnlyRelationDiagnostic
        }
        BlueBrainInterRegionDiagnosticsRelationState::CaveatedNoStrongPositiveSignal => {
            BlueBrainInterRegionDiagnosticsContractClass::CaveatedRelationDiagnostic
        }
        BlueBrainInterRegionDiagnosticsRelationState::DeferredNotYetUsable => {
            BlueBrainInterRegionDiagnosticsContractClass::DeferredRelationDiagnostic
        }
        BlueBrainInterRegionDiagnosticsRelationState::BlockedByContractSafetyOrReference => {
            BlueBrainInterRegionDiagnosticsContractClass::BlockedRelationDiagnostic
        }
        BlueBrainInterRegionDiagnosticsRelationState::InsufficientRelationalBasis => {
            BlueBrainInterRegionDiagnosticsContractClass::InsufficientRelationDiagnostic
        }
        BlueBrainInterRegionDiagnosticsRelationState::DiagnosticOnlyVisible => {
            BlueBrainInterRegionDiagnosticsContractClass::DiagnosticOnlyRelationState
        }
        BlueBrainInterRegionDiagnosticsRelationState::NonCanonicalInternalOnly => {
            BlueBrainInterRegionDiagnosticsContractClass::NonCanonicalInternalOnlyRelationPath
        }
    }
}

pub const fn blue_brain_inter_region_contract_signal_for_state(
    relation_state: BlueBrainInterRegionDiagnosticsRelationState,
) -> BlueBrainInterRegionContractSignalClass {
    match relation_state {
        BlueBrainInterRegionDiagnosticsRelationState::AdvisoryOnlyActive => {
            BlueBrainInterRegionContractSignalClass::BoundedRelationContractSignal
        }
        BlueBrainInterRegionDiagnosticsRelationState::CaveatedNoStrongPositiveSignal => {
            BlueBrainInterRegionContractSignalClass::CaveatedRelationDiagnosticSignal
        }
        BlueBrainInterRegionDiagnosticsRelationState::DeferredNotYetUsable => {
            BlueBrainInterRegionContractSignalClass::DeferredRelationDiagnosticSignal
        }
        BlueBrainInterRegionDiagnosticsRelationState::BlockedByContractSafetyOrReference => {
            BlueBrainInterRegionContractSignalClass::BlockedRelationDiagnosticSignal
        }
        BlueBrainInterRegionDiagnosticsRelationState::InsufficientRelationalBasis => {
            BlueBrainInterRegionContractSignalClass::InsufficientRelationDiagnosticSignal
        }
        BlueBrainInterRegionDiagnosticsRelationState::DiagnosticOnlyVisible => {
            BlueBrainInterRegionContractSignalClass::DiagnosticOnlyRelationSignal
        }
        BlueBrainInterRegionDiagnosticsRelationState::NonCanonicalInternalOnly => {
            BlueBrainInterRegionContractSignalClass::NonCanonicalInternalOnlyRelationSignal
        }
    }
}

pub const fn blue_brain_ir1_readiness_class_for_contract_read(
    read: BlueBrainInterRegionDiagnosticsContractRead,
) -> BlueBrainIr1ReadinessClass {
    match read.relation_state {
        BlueBrainInterRegionDiagnosticsRelationState::AdvisoryOnlyActive => {
            BlueBrainIr1ReadinessClass::StableImplementedRelation
        }
        BlueBrainInterRegionDiagnosticsRelationState::CaveatedNoStrongPositiveSignal => {
            BlueBrainIr1ReadinessClass::UsableWithCaveats
        }
        BlueBrainInterRegionDiagnosticsRelationState::DeferredNotYetUsable => {
            BlueBrainIr1ReadinessClass::DeferredNotYetActive
        }
        BlueBrainInterRegionDiagnosticsRelationState::BlockedByContractSafetyOrReference
        | BlueBrainInterRegionDiagnosticsRelationState::InsufficientRelationalBasis
        | BlueBrainInterRegionDiagnosticsRelationState::DiagnosticOnlyVisible => {
            BlueBrainIr1ReadinessClass::BlockedInsufficientDiagnosticOnly
        }
        BlueBrainInterRegionDiagnosticsRelationState::NonCanonicalInternalOnly => {
            BlueBrainIr1ReadinessClass::NonCanonicalInternalOnly
        }
    }
}

const fn blue_brain_inter_region_diagnostics_contract_read(
    relation: BlueBrainInterRegionImplementationRelation,
    consumer_layer: BlueBrainInterRegionConsumerLayer,
) -> BlueBrainInterRegionDiagnosticsContractRead {
    let relation_state = blue_brain_inter_region_relation_state_for_implementation_class(
        relation.implementation_relation_class,
    );
    let relation_diagnostic_class =
        blue_brain_inter_region_diagnostics_contract_class_for_state(relation_state);
    let contract_signal_class = blue_brain_inter_region_contract_signal_for_state(relation_state);

    BlueBrainInterRegionDiagnosticsContractRead {
        pair: relation.pair,
        consumer_layer,
        implementation_relation_class: relation.implementation_relation_class,
        mediation_path: relation.mediation_path,
        relation_state,
        relation_diagnostic_class,
        contract_signal_class,
        bounded_contract_signal: matches!(
            contract_signal_class,
            BlueBrainInterRegionContractSignalClass::BoundedRelationContractSignal
        ),
        advisory_only: matches!(
            relation_state,
            BlueBrainInterRegionDiagnosticsRelationState::AdvisoryOnlyActive
        ),
        caveated: matches!(
            relation_state,
            BlueBrainInterRegionDiagnosticsRelationState::CaveatedNoStrongPositiveSignal
        ),
        deferred: matches!(
            relation_state,
            BlueBrainInterRegionDiagnosticsRelationState::DeferredNotYetUsable
        ),
        blocked: matches!(
            relation_state,
            BlueBrainInterRegionDiagnosticsRelationState::BlockedByContractSafetyOrReference
        ),
        insufficient: matches!(
            relation_state,
            BlueBrainInterRegionDiagnosticsRelationState::InsufficientRelationalBasis
        ),
        diagnostic_only: matches!(
            relation_state,
            BlueBrainInterRegionDiagnosticsRelationState::DiagnosticOnlyVisible
                | BlueBrainInterRegionDiagnosticsRelationState::DeferredNotYetUsable
                | BlueBrainInterRegionDiagnosticsRelationState::BlockedByContractSafetyOrReference
                | BlueBrainInterRegionDiagnosticsRelationState::InsufficientRelationalBasis
                | BlueBrainInterRegionDiagnosticsRelationState::NonCanonicalInternalOnly
        ),
        non_canonical_internal_only: matches!(
            relation_state,
            BlueBrainInterRegionDiagnosticsRelationState::NonCanonicalInternalOnly
        ),
        direct_action_trigger: false,
        direct_execution_trigger: false,
        direct_retry_trigger: false,
        direct_memory_commit: false,
        direct_compute_invocation: false,
        safety_override: false,
        global_region_orchestration: false,
    }
}

pub fn blue_brain_inter_region_diagnostics_contract_read_for_pair(
    pair: BlueBrainInterRegionArchitecturePair,
    consumer_layer: BlueBrainInterRegionConsumerLayer,
) -> BlueBrainInterRegionDiagnosticsContractRead {
    blue_brain_inter_region_diagnostics_contract_read(
        blue_brain_first_inter_region_implementation_relation_for_pair(pair),
        consumer_layer,
    )
}

pub fn blue_brain_inter_region_consumer_contract_reads_are_aligned(
    pair: BlueBrainInterRegionArchitecturePair,
) -> bool {
    let runtime = blue_brain_inter_region_diagnostics_contract_read_for_pair(
        pair,
        BlueBrainInterRegionConsumerLayer::Runtime,
    );
    let selection = blue_brain_inter_region_diagnostics_contract_read_for_pair(
        pair,
        BlueBrainInterRegionConsumerLayer::Selection,
    );
    let reference = blue_brain_inter_region_diagnostics_contract_read_for_pair(
        pair,
        BlueBrainInterRegionConsumerLayer::Reference,
    );

    runtime.implementation_relation_class == selection.implementation_relation_class
        && runtime.implementation_relation_class == reference.implementation_relation_class
        && runtime.mediation_path == selection.mediation_path
        && runtime.mediation_path == reference.mediation_path
        && runtime.relation_state == selection.relation_state
        && runtime.relation_state == reference.relation_state
        && runtime.relation_diagnostic_class == selection.relation_diagnostic_class
        && runtime.relation_diagnostic_class == reference.relation_diagnostic_class
        && runtime.contract_signal_class == selection.contract_signal_class
        && runtime.contract_signal_class == reference.contract_signal_class
}

pub const CANONICAL_BLUE_BRAIN_INTER_REGION_DIAGNOSTICS_CONTRACT_MAP:
    [BlueBrainInterRegionDiagnosticsContractRead; 10] = [
    blue_brain_inter_region_diagnostics_contract_read(
        CANONICAL_BLUE_BRAIN_FIRST_INTER_REGION_IMPLEMENTATION_MAP[0],
        BlueBrainInterRegionConsumerLayer::Runtime,
    ),
    blue_brain_inter_region_diagnostics_contract_read(
        CANONICAL_BLUE_BRAIN_FIRST_INTER_REGION_IMPLEMENTATION_MAP[1],
        BlueBrainInterRegionConsumerLayer::Runtime,
    ),
    blue_brain_inter_region_diagnostics_contract_read(
        CANONICAL_BLUE_BRAIN_FIRST_INTER_REGION_IMPLEMENTATION_MAP[2],
        BlueBrainInterRegionConsumerLayer::Runtime,
    ),
    blue_brain_inter_region_diagnostics_contract_read(
        CANONICAL_BLUE_BRAIN_FIRST_INTER_REGION_IMPLEMENTATION_MAP[3],
        BlueBrainInterRegionConsumerLayer::Runtime,
    ),
    blue_brain_inter_region_diagnostics_contract_read(
        CANONICAL_BLUE_BRAIN_FIRST_INTER_REGION_IMPLEMENTATION_MAP[4],
        BlueBrainInterRegionConsumerLayer::Runtime,
    ),
    blue_brain_inter_region_diagnostics_contract_read(
        CANONICAL_BLUE_BRAIN_FIRST_INTER_REGION_IMPLEMENTATION_MAP[5],
        BlueBrainInterRegionConsumerLayer::Runtime,
    ),
    blue_brain_inter_region_diagnostics_contract_read(
        CANONICAL_BLUE_BRAIN_FIRST_INTER_REGION_IMPLEMENTATION_MAP[6],
        BlueBrainInterRegionConsumerLayer::Runtime,
    ),
    blue_brain_inter_region_diagnostics_contract_read(
        CANONICAL_BLUE_BRAIN_FIRST_INTER_REGION_IMPLEMENTATION_MAP[7],
        BlueBrainInterRegionConsumerLayer::Runtime,
    ),
    blue_brain_inter_region_diagnostics_contract_read(
        CANONICAL_BLUE_BRAIN_FIRST_INTER_REGION_IMPLEMENTATION_MAP[8],
        BlueBrainInterRegionConsumerLayer::Runtime,
    ),
    blue_brain_inter_region_diagnostics_contract_read(
        CANONICAL_BLUE_BRAIN_FIRST_INTER_REGION_IMPLEMENTATION_MAP[9],
        BlueBrainInterRegionConsumerLayer::Runtime,
    ),
];

pub fn blue_brain_first_inter_region_implementation_relation_for_pair(
    pair: BlueBrainInterRegionArchitecturePair,
) -> BlueBrainInterRegionImplementationRelation {
    CANONICAL_BLUE_BRAIN_FIRST_INTER_REGION_IMPLEMENTATION_MAP
        .iter()
        .copied()
        .find(|relation| relation.pair == pair)
        .unwrap_or_else(|| {
            blue_brain_inter_region_implementation_relation(
                pair,
                BlueBrainInterRegionImplementationRelationClass::NonCanonicalInternalOnlyRelationPath,
                BlueBrainInterRegionImplementationSignal::BlockedDiagnosticOnly,
                BlueBrainInterRegionImplementationSignal::BlockedDiagnosticOnly,
                BlueBrainInterRegionImplementationMediationPath::NonCanonicalInternalOnly,
            )
        })
}

pub fn is_blue_brain_first_inter_region_relation_implemented(
    pair: BlueBrainInterRegionArchitecturePair,
) -> bool {
    matches!(
        blue_brain_first_inter_region_implementation_relation_for_pair(pair)
            .implementation_relation_class,
        BlueBrainInterRegionImplementationRelationClass::ImplementedDirectBoundedAdvisoryRelation
            | BlueBrainInterRegionImplementationRelationClass::ImplementedReferenceMediatedRelation
            | BlueBrainInterRegionImplementationRelationClass::ImplementedSelectionMediatedRelation
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainMd1ModelDeepeningClass {
    AbstractSufficient,
    BoundedKuramotoLikeCandidate,
    HodgkinHuxleySimulationOnlyDiagnosticOnlyCandidate,
    LaterSelectiveHodgkinHuxleyDeepening,
    NoDeepeningNeededNow,
    NonCanonicalInternalOnlyModelPath,
}

pub const CANONICAL_BLUE_BRAIN_MD1_MODEL_DEEPENING_CLASS_MAP: [BlueBrainMd1ModelDeepeningClass; 6] = [
    BlueBrainMd1ModelDeepeningClass::AbstractSufficient,
    BlueBrainMd1ModelDeepeningClass::BoundedKuramotoLikeCandidate,
    BlueBrainMd1ModelDeepeningClass::HodgkinHuxleySimulationOnlyDiagnosticOnlyCandidate,
    BlueBrainMd1ModelDeepeningClass::LaterSelectiveHodgkinHuxleyDeepening,
    BlueBrainMd1ModelDeepeningClass::NoDeepeningNeededNow,
    BlueBrainMd1ModelDeepeningClass::NonCanonicalInternalOnlyModelPath,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainMd1DeepeningSurfaceKind {
    Region,
    Relation,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainMd1DeepeningPriorityClass {
    NextConcreteDeepeningCandidate,
    CandidateButWait,
    KeepAbstractOrDeferred,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainMd1RegionDeepeningDecision {
    pub region: BlueBrainAnatomicalRegionClass,
    pub system_role: BlueBrainAnatomicalRegionSystemRoleClass,
    pub current_model_mode: BlueBrainFirstAnatomicalRegionModelModeClass,
    pub deepening_class: BlueBrainMd1ModelDeepeningClass,
    pub priority_class: BlueBrainMd1DeepeningPriorityClass,
    pub coupling_synchrony_gating_timing_leverage: bool,
    pub excitability_spiking_membrane_leverage: bool,
    pub advisory_only: bool,
    pub direct_action_trigger: bool,
    pub direct_execution_trigger: bool,
    pub direct_retry_trigger: bool,
    pub direct_memory_commit: bool,
    pub direct_compute_invocation: bool,
    pub global_model_platform: bool,
}

const fn blue_brain_md1_region_deepening_decision_const(
    region: BlueBrainAnatomicalRegionClass,
    system_role: BlueBrainAnatomicalRegionSystemRoleClass,
    current_model_mode: BlueBrainFirstAnatomicalRegionModelModeClass,
    deepening_class: BlueBrainMd1ModelDeepeningClass,
    priority_class: BlueBrainMd1DeepeningPriorityClass,
    coupling_synchrony_gating_timing_leverage: bool,
    excitability_spiking_membrane_leverage: bool,
) -> BlueBrainMd1RegionDeepeningDecision {
    BlueBrainMd1RegionDeepeningDecision {
        region,
        system_role,
        current_model_mode,
        deepening_class,
        priority_class,
        coupling_synchrony_gating_timing_leverage,
        excitability_spiking_membrane_leverage,
        advisory_only: true,
        direct_action_trigger: false,
        direct_execution_trigger: false,
        direct_retry_trigger: false,
        direct_memory_commit: false,
        direct_compute_invocation: false,
        global_model_platform: false,
    }
}

pub const CANONICAL_BLUE_BRAIN_MD1_REGION_DEEPENING_DECISION_MAP:
    [BlueBrainMd1RegionDeepeningDecision; 5] = [
    blue_brain_md1_region_deepening_decision_const(
        BlueBrainAnatomicalRegionClass::Hippocampus,
        BlueBrainAnatomicalRegionSystemRoleClass::AttentionSelectionMediation,
        BlueBrainFirstAnatomicalRegionModelModeClass::AbstractFunctionalCurrentMode,
        BlueBrainMd1ModelDeepeningClass::AbstractSufficient,
        BlueBrainMd1DeepeningPriorityClass::KeepAbstractOrDeferred,
        false,
        false,
    ),
    blue_brain_md1_region_deepening_decision_const(
        BlueBrainAnatomicalRegionClass::Amygdala,
        BlueBrainAnatomicalRegionSystemRoleClass::ThreatSalienceCaveatMediation,
        BlueBrainFirstAnatomicalRegionModelModeClass::BoundedKuramotoLikeCurrentMode,
        BlueBrainMd1ModelDeepeningClass::BoundedKuramotoLikeCandidate,
        BlueBrainMd1DeepeningPriorityClass::CandidateButWait,
        true,
        false,
    ),
    blue_brain_md1_region_deepening_decision_const(
        BlueBrainAnatomicalRegionClass::Thalamus,
        BlueBrainAnatomicalRegionSystemRoleClass::RelayIntegrationMediation,
        BlueBrainFirstAnatomicalRegionModelModeClass::AbstractFunctionalCurrentMode,
        BlueBrainMd1ModelDeepeningClass::AbstractSufficient,
        BlueBrainMd1DeepeningPriorityClass::KeepAbstractOrDeferred,
        false,
        false,
    ),
    blue_brain_md1_region_deepening_decision_const(
        BlueBrainAnatomicalRegionClass::BasalGanglia,
        BlueBrainAnatomicalRegionSystemRoleClass::ActionGatingMediation,
        BlueBrainFirstAnatomicalRegionModelModeClass::AbstractFunctionalCurrentMode,
        BlueBrainMd1ModelDeepeningClass::NoDeepeningNeededNow,
        BlueBrainMd1DeepeningPriorityClass::KeepAbstractOrDeferred,
        false,
        false,
    ),
    blue_brain_md1_region_deepening_decision_const(
        BlueBrainAnatomicalRegionClass::Cerebellum,
        BlueBrainAnatomicalRegionSystemRoleClass::PredictionTimingCorrectionMediation,
        BlueBrainFirstAnatomicalRegionModelModeClass::AbstractFunctionalCurrentMode,
        BlueBrainMd1ModelDeepeningClass::HodgkinHuxleySimulationOnlyDiagnosticOnlyCandidate,
        BlueBrainMd1DeepeningPriorityClass::CandidateButWait,
        false,
        true,
    ),
];

pub fn blue_brain_md1_region_deepening_decision(
    region: BlueBrainAnatomicalRegionClass,
) -> BlueBrainMd1RegionDeepeningDecision {
    CANONICAL_BLUE_BRAIN_MD1_REGION_DEEPENING_DECISION_MAP
        .iter()
        .copied()
        .find(|decision| decision.region == region)
        .unwrap_or_else(|| {
            blue_brain_md1_region_deepening_decision_const(
                region,
                blue_brain_anatomical_region_system_role(region),
                blue_brain_anatomical_region_model_mode(region),
                BlueBrainMd1ModelDeepeningClass::NonCanonicalInternalOnlyModelPath,
                BlueBrainMd1DeepeningPriorityClass::NonCanonicalInternalOnly,
                false,
                false,
            )
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainMd1RelationDeepeningDecision {
    pub pair: BlueBrainInterRegionArchitecturePair,
    pub implementation_relation_class: BlueBrainInterRegionImplementationRelationClass,
    pub mediation_path: BlueBrainInterRegionImplementationMediationPath,
    pub deepening_class: BlueBrainMd1ModelDeepeningClass,
    pub priority_class: BlueBrainMd1DeepeningPriorityClass,
    pub coupling_synchrony_gating_timing_leverage: bool,
    pub excitability_spiking_membrane_leverage: bool,
    pub advisory_only: bool,
    pub direct_action_trigger: bool,
    pub direct_execution_trigger: bool,
    pub direct_retry_trigger: bool,
    pub direct_memory_commit: bool,
    pub direct_compute_invocation: bool,
    pub global_model_platform: bool,
}

const fn blue_brain_md1_relation_deepening_decision_const(
    pair: BlueBrainInterRegionArchitecturePair,
    implementation_relation_class: BlueBrainInterRegionImplementationRelationClass,
    mediation_path: BlueBrainInterRegionImplementationMediationPath,
    deepening_class: BlueBrainMd1ModelDeepeningClass,
    priority_class: BlueBrainMd1DeepeningPriorityClass,
    coupling_synchrony_gating_timing_leverage: bool,
    excitability_spiking_membrane_leverage: bool,
) -> BlueBrainMd1RelationDeepeningDecision {
    BlueBrainMd1RelationDeepeningDecision {
        pair,
        implementation_relation_class,
        mediation_path,
        deepening_class,
        priority_class,
        coupling_synchrony_gating_timing_leverage,
        excitability_spiking_membrane_leverage,
        advisory_only: true,
        direct_action_trigger: false,
        direct_execution_trigger: false,
        direct_retry_trigger: false,
        direct_memory_commit: false,
        direct_compute_invocation: false,
        global_model_platform: false,
    }
}

pub const CANONICAL_BLUE_BRAIN_MD1_RELATION_DEEPENING_DECISION_MAP:
    [BlueBrainMd1RelationDeepeningDecision; 10] = [
    blue_brain_md1_relation_deepening_decision_const(
        BlueBrainInterRegionArchitecturePair::HippocampusAmygdala,
        BlueBrainInterRegionImplementationRelationClass::DeferredNotYetImplementedRelation,
        BlueBrainInterRegionImplementationMediationPath::NotYetImplemented,
        BlueBrainMd1ModelDeepeningClass::NoDeepeningNeededNow,
        BlueBrainMd1DeepeningPriorityClass::KeepAbstractOrDeferred,
        false,
        false,
    ),
    blue_brain_md1_relation_deepening_decision_const(
        BlueBrainInterRegionArchitecturePair::HippocampusThalamus,
        BlueBrainInterRegionImplementationRelationClass::ImplementedReferenceMediatedRelation,
        BlueBrainInterRegionImplementationMediationPath::ReferenceContextMediatedOnly,
        BlueBrainMd1ModelDeepeningClass::AbstractSufficient,
        BlueBrainMd1DeepeningPriorityClass::KeepAbstractOrDeferred,
        false,
        false,
    ),
    blue_brain_md1_relation_deepening_decision_const(
        BlueBrainInterRegionArchitecturePair::HippocampusBasalGanglia,
        BlueBrainInterRegionImplementationRelationClass::BlockedRelation,
        BlueBrainInterRegionImplementationMediationPath::BlockedUnavailable,
        BlueBrainMd1ModelDeepeningClass::NoDeepeningNeededNow,
        BlueBrainMd1DeepeningPriorityClass::KeepAbstractOrDeferred,
        false,
        false,
    ),
    blue_brain_md1_relation_deepening_decision_const(
        BlueBrainInterRegionArchitecturePair::HippocampusCerebellum,
        BlueBrainInterRegionImplementationRelationClass::DeferredNotYetImplementedRelation,
        BlueBrainInterRegionImplementationMediationPath::NotYetImplemented,
        BlueBrainMd1ModelDeepeningClass::NoDeepeningNeededNow,
        BlueBrainMd1DeepeningPriorityClass::KeepAbstractOrDeferred,
        false,
        false,
    ),
    blue_brain_md1_relation_deepening_decision_const(
        BlueBrainInterRegionArchitecturePair::AmygdalaThalamus,
        BlueBrainInterRegionImplementationRelationClass::ImplementedDirectBoundedAdvisoryRelation,
        BlueBrainInterRegionImplementationMediationPath::DirectBoundedAdvisoryOnly,
        BlueBrainMd1ModelDeepeningClass::BoundedKuramotoLikeCandidate,
        BlueBrainMd1DeepeningPriorityClass::NextConcreteDeepeningCandidate,
        true,
        false,
    ),
    blue_brain_md1_relation_deepening_decision_const(
        BlueBrainInterRegionArchitecturePair::AmygdalaBasalGanglia,
        BlueBrainInterRegionImplementationRelationClass::ImplementedSelectionMediatedRelation,
        BlueBrainInterRegionImplementationMediationPath::SelectionContractMediatedOnly,
        BlueBrainMd1ModelDeepeningClass::BoundedKuramotoLikeCandidate,
        BlueBrainMd1DeepeningPriorityClass::NextConcreteDeepeningCandidate,
        true,
        false,
    ),
    blue_brain_md1_relation_deepening_decision_const(
        BlueBrainInterRegionArchitecturePair::AmygdalaCerebellum,
        BlueBrainInterRegionImplementationRelationClass::DeferredNotYetImplementedRelation,
        BlueBrainInterRegionImplementationMediationPath::NotYetImplemented,
        BlueBrainMd1ModelDeepeningClass::NoDeepeningNeededNow,
        BlueBrainMd1DeepeningPriorityClass::KeepAbstractOrDeferred,
        false,
        false,
    ),
    blue_brain_md1_relation_deepening_decision_const(
        BlueBrainInterRegionArchitecturePair::ThalamusBasalGanglia,
        BlueBrainInterRegionImplementationRelationClass::DeferredNotYetImplementedRelation,
        BlueBrainInterRegionImplementationMediationPath::NotYetImplemented,
        BlueBrainMd1ModelDeepeningClass::NoDeepeningNeededNow,
        BlueBrainMd1DeepeningPriorityClass::KeepAbstractOrDeferred,
        false,
        false,
    ),
    blue_brain_md1_relation_deepening_decision_const(
        BlueBrainInterRegionArchitecturePair::ThalamusCerebellum,
        BlueBrainInterRegionImplementationRelationClass::DeferredNotYetImplementedRelation,
        BlueBrainInterRegionImplementationMediationPath::NotYetImplemented,
        BlueBrainMd1ModelDeepeningClass::NoDeepeningNeededNow,
        BlueBrainMd1DeepeningPriorityClass::KeepAbstractOrDeferred,
        false,
        false,
    ),
    blue_brain_md1_relation_deepening_decision_const(
        BlueBrainInterRegionArchitecturePair::BasalGangliaCerebellum,
        BlueBrainInterRegionImplementationRelationClass::DeferredNotYetImplementedRelation,
        BlueBrainInterRegionImplementationMediationPath::NotYetImplemented,
        BlueBrainMd1ModelDeepeningClass::LaterSelectiveHodgkinHuxleyDeepening,
        BlueBrainMd1DeepeningPriorityClass::CandidateButWait,
        false,
        true,
    ),
];

pub fn blue_brain_md1_relation_deepening_decision(
    pair: BlueBrainInterRegionArchitecturePair,
) -> BlueBrainMd1RelationDeepeningDecision {
    CANONICAL_BLUE_BRAIN_MD1_RELATION_DEEPENING_DECISION_MAP
        .iter()
        .copied()
        .find(|decision| decision.pair == pair)
        .unwrap_or_else(|| {
            let relation = blue_brain_first_inter_region_implementation_relation_for_pair(pair);
            blue_brain_md1_relation_deepening_decision_const(
                pair,
                relation.implementation_relation_class,
                relation.mediation_path,
                BlueBrainMd1ModelDeepeningClass::NonCanonicalInternalOnlyModelPath,
                BlueBrainMd1DeepeningPriorityClass::NonCanonicalInternalOnly,
                false,
                false,
            )
        })
}

pub fn blue_brain_md1_relation_deepening_is_consistent_with_implementation(
    decision: BlueBrainMd1RelationDeepeningDecision,
) -> bool {
    let relation = blue_brain_first_inter_region_implementation_relation_for_pair(decision.pair);
    decision.implementation_relation_class == relation.implementation_relation_class
        && decision.mediation_path == relation.mediation_path
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainInterRegionArchitectureOutputClass {
    BoundedAdvisoryRead,
    ReferenceContextRead,
    SelectionContractRead,
    ExecutionInterfaceDiagnosticRead,
    CaveatDiagnosticRead,
    DeferredDiagnosticRead,
    BlockedDiagnosticRead,
    DirectActionTrigger,
    DirectExecutionTrigger,
    DirectRetryTrigger,
    DirectMemoryCommit,
    DirectComputeInvocation,
    SafetyOverride,
    GlobalRegionOrchestration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainInterRegionArchitectureOutputGuard {
    AllowedBoundedRead,
    BlockedForbiddenAuthorityPath,
}

pub fn classify_blue_brain_inter_region_architecture_output_guard(
    output: BlueBrainInterRegionArchitectureOutputClass,
) -> BlueBrainInterRegionArchitectureOutputGuard {
    match output {
        BlueBrainInterRegionArchitectureOutputClass::BoundedAdvisoryRead
        | BlueBrainInterRegionArchitectureOutputClass::ReferenceContextRead
        | BlueBrainInterRegionArchitectureOutputClass::SelectionContractRead
        | BlueBrainInterRegionArchitectureOutputClass::ExecutionInterfaceDiagnosticRead
        | BlueBrainInterRegionArchitectureOutputClass::CaveatDiagnosticRead
        | BlueBrainInterRegionArchitectureOutputClass::DeferredDiagnosticRead
        | BlueBrainInterRegionArchitectureOutputClass::BlockedDiagnosticRead => {
            BlueBrainInterRegionArchitectureOutputGuard::AllowedBoundedRead
        }
        BlueBrainInterRegionArchitectureOutputClass::DirectActionTrigger
        | BlueBrainInterRegionArchitectureOutputClass::DirectExecutionTrigger
        | BlueBrainInterRegionArchitectureOutputClass::DirectRetryTrigger
        | BlueBrainInterRegionArchitectureOutputClass::DirectMemoryCommit
        | BlueBrainInterRegionArchitectureOutputClass::DirectComputeInvocation
        | BlueBrainInterRegionArchitectureOutputClass::SafetyOverride
        | BlueBrainInterRegionArchitectureOutputClass::GlobalRegionOrchestration => {
            BlueBrainInterRegionArchitectureOutputGuard::BlockedForbiddenAuthorityPath
        }
    }
}

pub const BLUE_BRAIN_FIRST_ANATOMICAL_REGION_CURRENT_MODEL_MODE:
    BlueBrainFirstAnatomicalRegionModelModeClass =
    BlueBrainFirstAnatomicalRegionModelModeClass::AbstractFunctionalCurrentMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainAnatomicalRegionClass {
    Hippocampus,
    Amygdala,
    PrefrontalCortex,
    AnteriorCingulateCortex,
    BasalGanglia,
    Thalamus,
    Cerebellum,
    Insula,
}

pub const CANONICAL_BLUE_BRAIN_ANATOMICAL_REGION_MAP: [BlueBrainAnatomicalRegionClass; 8] = [
    BlueBrainAnatomicalRegionClass::Hippocampus,
    BlueBrainAnatomicalRegionClass::Amygdala,
    BlueBrainAnatomicalRegionClass::PrefrontalCortex,
    BlueBrainAnatomicalRegionClass::AnteriorCingulateCortex,
    BlueBrainAnatomicalRegionClass::BasalGanglia,
    BlueBrainAnatomicalRegionClass::Thalamus,
    BlueBrainAnatomicalRegionClass::Cerebellum,
    BlueBrainAnatomicalRegionClass::Insula,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainAnatomicalRegionSystemRoleClass {
    AttentionSelectionMediation,
    ThreatSalienceCaveatMediation,
    ControlPolicyConsistencyMediation,
    ConflictMonitoringMediation,
    ActionGatingMediation,
    RelayIntegrationMediation,
    PredictionTimingCorrectionMediation,
    InteroceptiveContextMediation,
}

pub fn blue_brain_anatomical_region_system_role(
    region: BlueBrainAnatomicalRegionClass,
) -> BlueBrainAnatomicalRegionSystemRoleClass {
    match region {
        BlueBrainAnatomicalRegionClass::Hippocampus => {
            BlueBrainAnatomicalRegionSystemRoleClass::AttentionSelectionMediation
        }
        BlueBrainAnatomicalRegionClass::Amygdala => {
            BlueBrainAnatomicalRegionSystemRoleClass::ThreatSalienceCaveatMediation
        }
        BlueBrainAnatomicalRegionClass::PrefrontalCortex => {
            BlueBrainAnatomicalRegionSystemRoleClass::ControlPolicyConsistencyMediation
        }
        BlueBrainAnatomicalRegionClass::AnteriorCingulateCortex => {
            BlueBrainAnatomicalRegionSystemRoleClass::ConflictMonitoringMediation
        }
        BlueBrainAnatomicalRegionClass::BasalGanglia => {
            BlueBrainAnatomicalRegionSystemRoleClass::ActionGatingMediation
        }
        BlueBrainAnatomicalRegionClass::Thalamus => {
            BlueBrainAnatomicalRegionSystemRoleClass::RelayIntegrationMediation
        }
        BlueBrainAnatomicalRegionClass::Cerebellum => {
            BlueBrainAnatomicalRegionSystemRoleClass::PredictionTimingCorrectionMediation
        }
        BlueBrainAnatomicalRegionClass::Insula => {
            BlueBrainAnatomicalRegionSystemRoleClass::InteroceptiveContextMediation
        }
    }
}

pub fn blue_brain_anatomical_region_model_mode(
    region: BlueBrainAnatomicalRegionClass,
) -> BlueBrainFirstAnatomicalRegionModelModeClass {
    match region {
        BlueBrainAnatomicalRegionClass::Hippocampus => {
            BlueBrainFirstAnatomicalRegionModelModeClass::AbstractFunctionalCurrentMode
        }
        BlueBrainAnatomicalRegionClass::Amygdala
        | BlueBrainAnatomicalRegionClass::AnteriorCingulateCortex => {
            BlueBrainFirstAnatomicalRegionModelModeClass::BoundedKuramotoLikeCurrentMode
        }
        BlueBrainAnatomicalRegionClass::Thalamus => {
            BlueBrainFirstAnatomicalRegionModelModeClass::AbstractFunctionalCurrentMode
        }
        BlueBrainAnatomicalRegionClass::PrefrontalCortex => {
            BlueBrainFirstAnatomicalRegionModelModeClass::LaterSelectiveHodgkinHuxleyDeepening
        }
        BlueBrainAnatomicalRegionClass::BasalGanglia
        | BlueBrainAnatomicalRegionClass::Cerebellum => {
            BlueBrainFirstAnatomicalRegionModelModeClass::AbstractFunctionalCurrentMode
        }
        BlueBrainAnatomicalRegionClass::Insula => {
            BlueBrainFirstAnatomicalRegionModelModeClass::DeferredNotSuitableNowModelPath
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFirstAnatomicalRegionContractSignal {
    AnatomicalToRuntimeAdvisory,
    RuntimeToAnatomicalBoundedInput,
    AnatomicalToSelectionAdvisory,
    SelectionToAnatomicalBoundedStateInput,
    AnatomicalReferenceSignal,
    Caveated,
    Deferred,
    Blocked,
    Insufficient,
    DiagnosticOnly,
    ReferenceOnly,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFirstAnatomicalRegionDiagnosticState {
    AnatomicalRegionAdvisoryOnlyDiagnostic,
    AnatomicalRegionCaveatedDiagnostic,
    AnatomicalRegionDeferredDiagnostic,
    AnatomicalRegionBlockedDiagnostic,
    AnatomicalRegionInsufficientDiagnostic,
    AnatomicalRegionDiagnosticOnlyState,
    NonCanonicalInternalOnlyAnatomicalRegionDiagnosticPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainHippocampusContractClass {
    HippocampusAdvisoryOnlyDiagnostic,
    HippocampusCaveatedDiagnostic,
    HippocampusDeferredDiagnostic,
    HippocampusBlockedDiagnostic,
    HippocampusInsufficientDiagnostic,
    HippocampusDiagnosticOnlyState,
    HippocampusBoundedContractSignal,
    NonCanonicalInternalOnlyHippocampusPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainAmygdalaContractClass {
    AmygdalaAdvisoryOnlyDiagnostic,
    AmygdalaCaveatedDiagnostic,
    AmygdalaDeferredDiagnostic,
    AmygdalaBlockedDiagnostic,
    AmygdalaInsufficientDiagnostic,
    AmygdalaDiagnosticOnlyState,
    AmygdalaBoundedContractSignal,
    NonCanonicalInternalOnlyAmygdalaPath,
}

pub const CANONICAL_BLUE_BRAIN_HIPPOCAMPUS_DIAGNOSTICS_CONTRACT_MAP:
    [BlueBrainHippocampusContractClass; 8] = [
    BlueBrainHippocampusContractClass::HippocampusAdvisoryOnlyDiagnostic,
    BlueBrainHippocampusContractClass::HippocampusCaveatedDiagnostic,
    BlueBrainHippocampusContractClass::HippocampusDeferredDiagnostic,
    BlueBrainHippocampusContractClass::HippocampusBlockedDiagnostic,
    BlueBrainHippocampusContractClass::HippocampusInsufficientDiagnostic,
    BlueBrainHippocampusContractClass::HippocampusDiagnosticOnlyState,
    BlueBrainHippocampusContractClass::HippocampusBoundedContractSignal,
    BlueBrainHippocampusContractClass::NonCanonicalInternalOnlyHippocampusPath,
];

pub const CANONICAL_BLUE_BRAIN_AMYGDALA_DIAGNOSTICS_CONTRACT_MAP: [BlueBrainAmygdalaContractClass;
    8] = [
    BlueBrainAmygdalaContractClass::AmygdalaAdvisoryOnlyDiagnostic,
    BlueBrainAmygdalaContractClass::AmygdalaCaveatedDiagnostic,
    BlueBrainAmygdalaContractClass::AmygdalaDeferredDiagnostic,
    BlueBrainAmygdalaContractClass::AmygdalaBlockedDiagnostic,
    BlueBrainAmygdalaContractClass::AmygdalaInsufficientDiagnostic,
    BlueBrainAmygdalaContractClass::AmygdalaDiagnosticOnlyState,
    BlueBrainAmygdalaContractClass::AmygdalaBoundedContractSignal,
    BlueBrainAmygdalaContractClass::NonCanonicalInternalOnlyAmygdalaPath,
];

pub const CANONICAL_BLUE_BRAIN_FIRST_ANATOMICAL_REGION_DIAGNOSTIC_MAP:
    [BlueBrainFirstAnatomicalRegionDiagnosticState; 7] = [
    BlueBrainFirstAnatomicalRegionDiagnosticState::AnatomicalRegionAdvisoryOnlyDiagnostic,
    BlueBrainFirstAnatomicalRegionDiagnosticState::AnatomicalRegionCaveatedDiagnostic,
    BlueBrainFirstAnatomicalRegionDiagnosticState::AnatomicalRegionDeferredDiagnostic,
    BlueBrainFirstAnatomicalRegionDiagnosticState::AnatomicalRegionBlockedDiagnostic,
    BlueBrainFirstAnatomicalRegionDiagnosticState::AnatomicalRegionInsufficientDiagnostic,
    BlueBrainFirstAnatomicalRegionDiagnosticState::AnatomicalRegionDiagnosticOnlyState,
    BlueBrainFirstAnatomicalRegionDiagnosticState::NonCanonicalInternalOnlyAnatomicalRegionDiagnosticPath,
];

pub fn blue_brain_first_anatomical_region_diagnostic_state_for_signal(
    signal: BlueBrainFirstAnatomicalRegionContractSignal,
) -> BlueBrainFirstAnatomicalRegionDiagnosticState {
    match signal {
        BlueBrainFirstAnatomicalRegionContractSignal::AnatomicalToRuntimeAdvisory
        | BlueBrainFirstAnatomicalRegionContractSignal::RuntimeToAnatomicalBoundedInput
        | BlueBrainFirstAnatomicalRegionContractSignal::AnatomicalToSelectionAdvisory
        | BlueBrainFirstAnatomicalRegionContractSignal::SelectionToAnatomicalBoundedStateInput => {
            BlueBrainFirstAnatomicalRegionDiagnosticState::AnatomicalRegionAdvisoryOnlyDiagnostic
        }
        BlueBrainFirstAnatomicalRegionContractSignal::Caveated => {
            BlueBrainFirstAnatomicalRegionDiagnosticState::AnatomicalRegionCaveatedDiagnostic
        }
        BlueBrainFirstAnatomicalRegionContractSignal::Deferred => {
            BlueBrainFirstAnatomicalRegionDiagnosticState::AnatomicalRegionDeferredDiagnostic
        }
        BlueBrainFirstAnatomicalRegionContractSignal::Blocked => {
            BlueBrainFirstAnatomicalRegionDiagnosticState::AnatomicalRegionBlockedDiagnostic
        }
        BlueBrainFirstAnatomicalRegionContractSignal::Insufficient => {
            BlueBrainFirstAnatomicalRegionDiagnosticState::AnatomicalRegionInsufficientDiagnostic
        }
        BlueBrainFirstAnatomicalRegionContractSignal::DiagnosticOnly
        | BlueBrainFirstAnatomicalRegionContractSignal::ReferenceOnly
        | BlueBrainFirstAnatomicalRegionContractSignal::AnatomicalReferenceSignal => {
            BlueBrainFirstAnatomicalRegionDiagnosticState::AnatomicalRegionDiagnosticOnlyState
        }
        BlueBrainFirstAnatomicalRegionContractSignal::NonCanonicalInternalOnly => {
            BlueBrainFirstAnatomicalRegionDiagnosticState::NonCanonicalInternalOnlyAnatomicalRegionDiagnosticPath
        }
    }
}

pub fn blue_brain_first_anatomical_region_runtime_diagnostic_read(
    signal: BlueBrainFirstAnatomicalRegionContractSignal,
) -> BlueBrainFirstAnatomicalRegionDiagnosticState {
    blue_brain_first_anatomical_region_diagnostic_state_for_signal(signal)
}

pub fn blue_brain_first_anatomical_region_selection_diagnostic_read(
    signal: BlueBrainFirstAnatomicalRegionContractSignal,
) -> BlueBrainFirstAnatomicalRegionDiagnosticState {
    blue_brain_first_anatomical_region_diagnostic_state_for_signal(signal)
}

pub fn blue_brain_first_anatomical_region_reference_diagnostic_read(
    signal: BlueBrainFirstAnatomicalRegionContractSignal,
) -> BlueBrainFirstAnatomicalRegionDiagnosticState {
    blue_brain_first_anatomical_region_diagnostic_state_for_signal(signal)
}

pub fn blue_brain_hippocampus_contract_class_for_signal(
    signal: BlueBrainFirstAnatomicalRegionContractSignal,
) -> BlueBrainHippocampusContractClass {
    match signal {
        BlueBrainFirstAnatomicalRegionContractSignal::AnatomicalToRuntimeAdvisory
        | BlueBrainFirstAnatomicalRegionContractSignal::RuntimeToAnatomicalBoundedInput
        | BlueBrainFirstAnatomicalRegionContractSignal::AnatomicalToSelectionAdvisory
        | BlueBrainFirstAnatomicalRegionContractSignal::SelectionToAnatomicalBoundedStateInput => {
            BlueBrainHippocampusContractClass::HippocampusAdvisoryOnlyDiagnostic
        }
        BlueBrainFirstAnatomicalRegionContractSignal::Caveated => {
            BlueBrainHippocampusContractClass::HippocampusCaveatedDiagnostic
        }
        BlueBrainFirstAnatomicalRegionContractSignal::Deferred => {
            BlueBrainHippocampusContractClass::HippocampusDeferredDiagnostic
        }
        BlueBrainFirstAnatomicalRegionContractSignal::Blocked => {
            BlueBrainHippocampusContractClass::HippocampusBlockedDiagnostic
        }
        BlueBrainFirstAnatomicalRegionContractSignal::Insufficient => {
            BlueBrainHippocampusContractClass::HippocampusInsufficientDiagnostic
        }
        BlueBrainFirstAnatomicalRegionContractSignal::DiagnosticOnly
        | BlueBrainFirstAnatomicalRegionContractSignal::ReferenceOnly
        | BlueBrainFirstAnatomicalRegionContractSignal::AnatomicalReferenceSignal => {
            BlueBrainHippocampusContractClass::HippocampusDiagnosticOnlyState
        }
        BlueBrainFirstAnatomicalRegionContractSignal::NonCanonicalInternalOnly => {
            BlueBrainHippocampusContractClass::NonCanonicalInternalOnlyHippocampusPath
        }
    }
}

pub fn blue_brain_amygdala_contract_class_for_signal(
    signal: BlueBrainFirstAnatomicalRegionContractSignal,
) -> BlueBrainAmygdalaContractClass {
    match signal {
        BlueBrainFirstAnatomicalRegionContractSignal::AnatomicalToRuntimeAdvisory
        | BlueBrainFirstAnatomicalRegionContractSignal::RuntimeToAnatomicalBoundedInput
        | BlueBrainFirstAnatomicalRegionContractSignal::AnatomicalToSelectionAdvisory
        | BlueBrainFirstAnatomicalRegionContractSignal::SelectionToAnatomicalBoundedStateInput => {
            BlueBrainAmygdalaContractClass::AmygdalaAdvisoryOnlyDiagnostic
        }
        BlueBrainFirstAnatomicalRegionContractSignal::Caveated => {
            BlueBrainAmygdalaContractClass::AmygdalaCaveatedDiagnostic
        }
        BlueBrainFirstAnatomicalRegionContractSignal::Deferred => {
            BlueBrainAmygdalaContractClass::AmygdalaDeferredDiagnostic
        }
        BlueBrainFirstAnatomicalRegionContractSignal::Blocked => {
            BlueBrainAmygdalaContractClass::AmygdalaBlockedDiagnostic
        }
        BlueBrainFirstAnatomicalRegionContractSignal::Insufficient => {
            BlueBrainAmygdalaContractClass::AmygdalaInsufficientDiagnostic
        }
        BlueBrainFirstAnatomicalRegionContractSignal::DiagnosticOnly
        | BlueBrainFirstAnatomicalRegionContractSignal::ReferenceOnly
        | BlueBrainFirstAnatomicalRegionContractSignal::AnatomicalReferenceSignal => {
            BlueBrainAmygdalaContractClass::AmygdalaDiagnosticOnlyState
        }
        BlueBrainFirstAnatomicalRegionContractSignal::NonCanonicalInternalOnly => {
            BlueBrainAmygdalaContractClass::NonCanonicalInternalOnlyAmygdalaPath
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFirstAnatomicalRegionInputClass {
    RuntimeSelectionContextSignal,
    AdvisoryReferenceSignal,
    ToolActionControlSignal,
    ComputeInternalRawState,
    SafetyOverrideSignal,
    MemoryMutationSignal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFirstAnatomicalRegionInputGuard {
    AllowedBoundedInput,
    BlockedForbiddenInput,
}

pub fn classify_blue_brain_first_anatomical_region_input_guard(
    input: BlueBrainFirstAnatomicalRegionInputClass,
) -> BlueBrainFirstAnatomicalRegionInputGuard {
    match input {
        BlueBrainFirstAnatomicalRegionInputClass::RuntimeSelectionContextSignal
        | BlueBrainFirstAnatomicalRegionInputClass::AdvisoryReferenceSignal => {
            BlueBrainFirstAnatomicalRegionInputGuard::AllowedBoundedInput
        }
        BlueBrainFirstAnatomicalRegionInputClass::ToolActionControlSignal
        | BlueBrainFirstAnatomicalRegionInputClass::ComputeInternalRawState
        | BlueBrainFirstAnatomicalRegionInputClass::SafetyOverrideSignal
        | BlueBrainFirstAnatomicalRegionInputClass::MemoryMutationSignal => {
            BlueBrainFirstAnatomicalRegionInputGuard::BlockedForbiddenInput
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFirstAnatomicalRegionOutputClass {
    AdvisorySalienceHint,
    AdvisoryGatingHint,
    AdvisoryMemoryContextHint,
    AdvisoryReferenceBoundedSignal,
    DirectActionSelection,
    DirectExecutionTrigger,
    DirectRetryTrigger,
    DirectMemoryCommit,
    DirectComputeInvocation,
    SafetyOverrideSignal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFirstAnatomicalRegionOutputGuard {
    AllowedAdvisoryOutput,
    BlockedForbiddenOutput,
}

pub fn classify_blue_brain_first_anatomical_region_output_guard(
    output: BlueBrainFirstAnatomicalRegionOutputClass,
) -> BlueBrainFirstAnatomicalRegionOutputGuard {
    match output {
        BlueBrainFirstAnatomicalRegionOutputClass::AdvisorySalienceHint
        | BlueBrainFirstAnatomicalRegionOutputClass::AdvisoryGatingHint
        | BlueBrainFirstAnatomicalRegionOutputClass::AdvisoryMemoryContextHint
        | BlueBrainFirstAnatomicalRegionOutputClass::AdvisoryReferenceBoundedSignal => {
            BlueBrainFirstAnatomicalRegionOutputGuard::AllowedAdvisoryOutput
        }
        BlueBrainFirstAnatomicalRegionOutputClass::DirectActionSelection
        | BlueBrainFirstAnatomicalRegionOutputClass::DirectExecutionTrigger
        | BlueBrainFirstAnatomicalRegionOutputClass::DirectRetryTrigger
        | BlueBrainFirstAnatomicalRegionOutputClass::DirectMemoryCommit
        | BlueBrainFirstAnatomicalRegionOutputClass::DirectComputeInvocation
        | BlueBrainFirstAnatomicalRegionOutputClass::SafetyOverrideSignal => {
            BlueBrainFirstAnatomicalRegionOutputGuard::BlockedForbiddenOutput
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainThirdRegionPathClass {
    Region3InputSurface,
    Region3StateSurface,
    Region3OutputAdvisorySurface,
    Region3ReferenceSurface,
    BlockedDeferredRegion3Path,
    NonCanonicalInternalOnlyRegion3Path,
}

pub const CANONICAL_BLUE_BRAIN_THIRD_REGION_INTEGRATION_MAP: [BlueBrainThirdRegionPathClass; 6] = [
    BlueBrainThirdRegionPathClass::Region3InputSurface,
    BlueBrainThirdRegionPathClass::Region3StateSurface,
    BlueBrainThirdRegionPathClass::Region3OutputAdvisorySurface,
    BlueBrainThirdRegionPathClass::Region3ReferenceSurface,
    BlueBrainThirdRegionPathClass::BlockedDeferredRegion3Path,
    BlueBrainThirdRegionPathClass::NonCanonicalInternalOnlyRegion3Path,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainThirdRegionContractClass {
    Region3ToRuntimeAdvisorySignal,
    RuntimeToRegion3BoundedInput,
    Region3ToSelectionAdvisorySignal,
    SelectionToRegion3BoundedStateInput,
    Region3ReferenceSignal,
    CaveatedDeferredBlockedRegion3ContractSignal,
    ReferenceOnlyRegion3ContractSignal,
    NonCanonicalInternalOnlyRegion3ContractPath,
}

pub const CANONICAL_BLUE_BRAIN_THIRD_REGION_CONTRACT_MAP: [BlueBrainThirdRegionContractClass; 8] = [
    BlueBrainThirdRegionContractClass::Region3ToRuntimeAdvisorySignal,
    BlueBrainThirdRegionContractClass::RuntimeToRegion3BoundedInput,
    BlueBrainThirdRegionContractClass::Region3ToSelectionAdvisorySignal,
    BlueBrainThirdRegionContractClass::SelectionToRegion3BoundedStateInput,
    BlueBrainThirdRegionContractClass::Region3ReferenceSignal,
    BlueBrainThirdRegionContractClass::CaveatedDeferredBlockedRegion3ContractSignal,
    BlueBrainThirdRegionContractClass::ReferenceOnlyRegion3ContractSignal,
    BlueBrainThirdRegionContractClass::NonCanonicalInternalOnlyRegion3ContractPath,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainThirdRegionContractSignal {
    Region3ToRuntimeAdvisory,
    RuntimeToRegion3BoundedInput,
    Region3ToSelectionAdvisory,
    SelectionToRegion3BoundedStateInput,
    Region3ReferenceSignal,
    Caveated,
    Deferred,
    Blocked,
    Insufficient,
    DiagnosticOnly,
    ReferenceOnly,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainThirdRegionDiagnosticState {
    Region3AdvisoryOnlyDiagnostic,
    Region3CaveatedDiagnostic,
    Region3DeferredDiagnostic,
    Region3BlockedDiagnostic,
    Region3InsufficientDiagnostic,
    Region3DiagnosticOnlyState,
    CaveatedInterRegionDiagnosticInfluence,
    NonCanonicalInternalOnlyRegion3DiagnosticPath,
}

pub const CANONICAL_BLUE_BRAIN_THIRD_REGION_DIAGNOSTIC_MAP: [BlueBrainThirdRegionDiagnosticState;
    8] = [
    BlueBrainThirdRegionDiagnosticState::Region3AdvisoryOnlyDiagnostic,
    BlueBrainThirdRegionDiagnosticState::Region3CaveatedDiagnostic,
    BlueBrainThirdRegionDiagnosticState::Region3DeferredDiagnostic,
    BlueBrainThirdRegionDiagnosticState::Region3BlockedDiagnostic,
    BlueBrainThirdRegionDiagnosticState::Region3InsufficientDiagnostic,
    BlueBrainThirdRegionDiagnosticState::Region3DiagnosticOnlyState,
    BlueBrainThirdRegionDiagnosticState::CaveatedInterRegionDiagnosticInfluence,
    BlueBrainThirdRegionDiagnosticState::NonCanonicalInternalOnlyRegion3DiagnosticPath,
];

pub fn blue_brain_third_region_diagnostic_state_for_signal(
    signal: BlueBrainThirdRegionContractSignal,
) -> BlueBrainThirdRegionDiagnosticState {
    match signal {
        BlueBrainThirdRegionContractSignal::Region3ToRuntimeAdvisory
        | BlueBrainThirdRegionContractSignal::RuntimeToRegion3BoundedInput
        | BlueBrainThirdRegionContractSignal::Region3ToSelectionAdvisory
        | BlueBrainThirdRegionContractSignal::SelectionToRegion3BoundedStateInput => {
            BlueBrainThirdRegionDiagnosticState::Region3AdvisoryOnlyDiagnostic
        }
        BlueBrainThirdRegionContractSignal::Caveated => {
            BlueBrainThirdRegionDiagnosticState::Region3CaveatedDiagnostic
        }
        BlueBrainThirdRegionContractSignal::Deferred => {
            BlueBrainThirdRegionDiagnosticState::Region3DeferredDiagnostic
        }
        BlueBrainThirdRegionContractSignal::Blocked => {
            BlueBrainThirdRegionDiagnosticState::Region3BlockedDiagnostic
        }
        BlueBrainThirdRegionContractSignal::Insufficient => {
            BlueBrainThirdRegionDiagnosticState::Region3InsufficientDiagnostic
        }
        BlueBrainThirdRegionContractSignal::DiagnosticOnly
        | BlueBrainThirdRegionContractSignal::ReferenceOnly
        | BlueBrainThirdRegionContractSignal::Region3ReferenceSignal => {
            BlueBrainThirdRegionDiagnosticState::Region3DiagnosticOnlyState
        }
        BlueBrainThirdRegionContractSignal::NonCanonicalInternalOnly => {
            BlueBrainThirdRegionDiagnosticState::NonCanonicalInternalOnlyRegion3DiagnosticPath
        }
    }
}

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
pub enum BlueBrainSecondRegionHardeningClass {
    GuardedCanonicalRegionSurface,
    GuardedRegionDiagnosticsPath,
    GuardedBoundedInterRegionRelationPath,
    BlockedForbiddenAuthorityPath,
    NonCanonicalInternalOnlyRegionPath,
    TestOnlyHelperNonOperationalPath,
}

pub const CANONICAL_BLUE_BRAIN_SECOND_REGION_HARDENING_MAP: [BlueBrainSecondRegionHardeningClass;
    6] = [
    BlueBrainSecondRegionHardeningClass::GuardedCanonicalRegionSurface,
    BlueBrainSecondRegionHardeningClass::GuardedRegionDiagnosticsPath,
    BlueBrainSecondRegionHardeningClass::GuardedBoundedInterRegionRelationPath,
    BlueBrainSecondRegionHardeningClass::BlockedForbiddenAuthorityPath,
    BlueBrainSecondRegionHardeningClass::NonCanonicalInternalOnlyRegionPath,
    BlueBrainSecondRegionHardeningClass::TestOnlyHelperNonOperationalPath,
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
pub enum BlueBrainThirdRegionRelationClass {
    Region3ToRegion1Bounded,
    Region1ToRegion3Bounded,
    Region3ToRegion2Bounded,
    Region2ToRegion3Bounded,
    SharedReferenceMediatedRelation,
    CaveatedInterRegionRelation,
    BlockedDeferredInterRegionRelation,
    NonCanonicalInternalOnlyInterRegionPath,
}

pub const CANONICAL_BLUE_BRAIN_THIRD_REGION_RELATION_MAP: [BlueBrainThirdRegionRelationClass; 8] = [
    BlueBrainThirdRegionRelationClass::Region3ToRegion1Bounded,
    BlueBrainThirdRegionRelationClass::Region1ToRegion3Bounded,
    BlueBrainThirdRegionRelationClass::Region3ToRegion2Bounded,
    BlueBrainThirdRegionRelationClass::Region2ToRegion3Bounded,
    BlueBrainThirdRegionRelationClass::SharedReferenceMediatedRelation,
    BlueBrainThirdRegionRelationClass::CaveatedInterRegionRelation,
    BlueBrainThirdRegionRelationClass::BlockedDeferredInterRegionRelation,
    BlueBrainThirdRegionRelationClass::NonCanonicalInternalOnlyInterRegionPath,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainThirdRegionHardeningClass {
    GuardedCanonicalRegion3Surface,
    GuardedRegion3DiagnosticsPath,
    GuardedBoundedInterRegionRelationPath,
    BlockedForbiddenAuthorityPath,
    NonCanonicalInternalOnlyRegion3Path,
    TestOnlyHelperNonOperationalPath,
}

pub const CANONICAL_BLUE_BRAIN_THIRD_REGION_HARDENING_MAP: [BlueBrainThirdRegionHardeningClass; 6] = [
    BlueBrainThirdRegionHardeningClass::GuardedCanonicalRegion3Surface,
    BlueBrainThirdRegionHardeningClass::GuardedRegion3DiagnosticsPath,
    BlueBrainThirdRegionHardeningClass::GuardedBoundedInterRegionRelationPath,
    BlueBrainThirdRegionHardeningClass::BlockedForbiddenAuthorityPath,
    BlueBrainThirdRegionHardeningClass::NonCanonicalInternalOnlyRegion3Path,
    BlueBrainThirdRegionHardeningClass::TestOnlyHelperNonOperationalPath,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainThirdRegionRelationSurface {
    pub relation_class: BlueBrainThirdRegionRelationClass,
    pub region3_to_region1_advisory_only: bool,
    pub region1_to_region3_advisory_only: bool,
    pub region3_to_region2_advisory_only: bool,
    pub region2_to_region3_advisory_only: bool,
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

pub fn evaluate_blue_brain_third_region_relation(
    region1: BlueBrainFirstRegionOutputSurface,
    region2: BlueBrainSecondRegionOutputSurface,
) -> BlueBrainThirdRegionRelationSurface {
    let relation12 = evaluate_blue_brain_inter_region_relation(region1, region2);
    let region3_is_feedback_advisory = matches!(
        region2.runtime_contract_signal,
        BlueBrainSecondRegionContractSignal::RegionToRuntimeAdvisory
    ) && matches!(
        region2.selection_contract_signal,
        BlueBrainSecondRegionContractSignal::RegionToSelectionAdvisory
    );

    let region3_receives_bounded_inputs = matches!(
        region2.runtime_contract_signal,
        BlueBrainSecondRegionContractSignal::Deferred
            | BlueBrainSecondRegionContractSignal::Blocked
            | BlueBrainSecondRegionContractSignal::Caveated
            | BlueBrainSecondRegionContractSignal::Insufficient
            | BlueBrainSecondRegionContractSignal::ReferenceOnly
            | BlueBrainSecondRegionContractSignal::RegionToRuntimeAdvisory
    ) && matches!(
        region2.selection_contract_signal,
        BlueBrainSecondRegionContractSignal::Deferred
            | BlueBrainSecondRegionContractSignal::Blocked
            | BlueBrainSecondRegionContractSignal::Caveated
            | BlueBrainSecondRegionContractSignal::Insufficient
            | BlueBrainSecondRegionContractSignal::RegionToSelectionAdvisory
    );

    let relation_class = if matches!(
        relation12.relation_class,
        BlueBrainInterRegionRelationClass::NonCanonicalInternalOnlyPath
    ) {
        BlueBrainThirdRegionRelationClass::NonCanonicalInternalOnlyInterRegionPath
    } else if relation12.blocked || relation12.deferred {
        BlueBrainThirdRegionRelationClass::BlockedDeferredInterRegionRelation
    } else if relation12.caveated {
        BlueBrainThirdRegionRelationClass::CaveatedInterRegionRelation
    } else if relation12.reference_mediated_only
        && (region1.reference_only
            || matches!(
                region2.reference_contract_signal,
                BlueBrainSecondRegionContractSignal::ReferenceOnly
                    | BlueBrainSecondRegionContractSignal::RegionReferenceSignal
            ))
    {
        BlueBrainThirdRegionRelationClass::SharedReferenceMediatedRelation
    } else if region3_is_feedback_advisory {
        BlueBrainThirdRegionRelationClass::Region3ToRegion1Bounded
    } else if region3_receives_bounded_inputs {
        BlueBrainThirdRegionRelationClass::Region1ToRegion3Bounded
    } else if relation12.region2_to_region1_advisory_only {
        BlueBrainThirdRegionRelationClass::Region3ToRegion2Bounded
    } else {
        BlueBrainThirdRegionRelationClass::Region2ToRegion3Bounded
    };

    BlueBrainThirdRegionRelationSurface {
        relation_class,
        region3_to_region1_advisory_only: region2.runtime_advisory_only
            && region2.selection_advisory_only,
        region1_to_region3_advisory_only: region1.runtime_advisory_only
            && region1.selection_advisory_only,
        region3_to_region2_advisory_only: region2.runtime_advisory_only,
        region2_to_region3_advisory_only: region2.selection_advisory_only,
        reference_mediated_only: relation12.reference_mediated_only,
        caveated: relation12.caveated,
        deferred: relation12.deferred,
        blocked: relation12.blocked,
        direct_action_selection: false,
        direct_execution_trigger: false,
        direct_retry_trigger: false,
        direct_memory_commit: false,
        direct_compute_invocation: false,
        safety_override: false,
    }
}

pub fn blue_brain_third_region_inter_region_diagnostic_influence(
    relation: BlueBrainThirdRegionRelationSurface,
) -> BlueBrainThirdRegionDiagnosticState {
    match relation.relation_class {
        BlueBrainThirdRegionRelationClass::CaveatedInterRegionRelation => {
            BlueBrainThirdRegionDiagnosticState::CaveatedInterRegionDiagnosticInfluence
        }
        BlueBrainThirdRegionRelationClass::BlockedDeferredInterRegionRelation => {
            if relation.blocked {
                BlueBrainThirdRegionDiagnosticState::Region3BlockedDiagnostic
            } else {
                BlueBrainThirdRegionDiagnosticState::Region3DeferredDiagnostic
            }
        }
        BlueBrainThirdRegionRelationClass::NonCanonicalInternalOnlyInterRegionPath => {
            BlueBrainThirdRegionDiagnosticState::NonCanonicalInternalOnlyRegion3DiagnosticPath
        }
        BlueBrainThirdRegionRelationClass::SharedReferenceMediatedRelation => {
            BlueBrainThirdRegionDiagnosticState::Region3DiagnosticOnlyState
        }
        BlueBrainThirdRegionRelationClass::Region3ToRegion1Bounded
        | BlueBrainThirdRegionRelationClass::Region1ToRegion3Bounded
        | BlueBrainThirdRegionRelationClass::Region3ToRegion2Bounded
        | BlueBrainThirdRegionRelationClass::Region2ToRegion3Bounded => {
            BlueBrainThirdRegionDiagnosticState::Region3AdvisoryOnlyDiagnostic
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainTwoRegionConsistencyClass {
    CanonicalRegion1Path,
    CanonicalRegion2Path,
    BoundedInterRegionRelationPath,
    CaveatedTwoRegionPath,
    BlockedInsufficientTwoRegionPath,
    NonCanonicalInternalOnlyTwoRegionPath,
}

pub const CANONICAL_BLUE_BRAIN_TWO_REGION_CONSISTENCY_MAP: [BlueBrainTwoRegionConsistencyClass; 6] = [
    BlueBrainTwoRegionConsistencyClass::CanonicalRegion1Path,
    BlueBrainTwoRegionConsistencyClass::CanonicalRegion2Path,
    BlueBrainTwoRegionConsistencyClass::BoundedInterRegionRelationPath,
    BlueBrainTwoRegionConsistencyClass::CaveatedTwoRegionPath,
    BlueBrainTwoRegionConsistencyClass::BlockedInsufficientTwoRegionPath,
    BlueBrainTwoRegionConsistencyClass::NonCanonicalInternalOnlyTwoRegionPath,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainThreeRegionConsistencyClass {
    CanonicalRegion1Path,
    CanonicalRegion2Path,
    CanonicalRegion3Path,
    BoundedInterRegionRelationPath,
    CaveatedThreeRegionPath,
    BlockedInsufficientThreeRegionPath,
    NonCanonicalInternalOnlyThreeRegionPath,
}

pub const CANONICAL_BLUE_BRAIN_THREE_REGION_CONSISTENCY_MAP:
    [BlueBrainThreeRegionConsistencyClass; 7] = [
    BlueBrainThreeRegionConsistencyClass::CanonicalRegion1Path,
    BlueBrainThreeRegionConsistencyClass::CanonicalRegion2Path,
    BlueBrainThreeRegionConsistencyClass::CanonicalRegion3Path,
    BlueBrainThreeRegionConsistencyClass::BoundedInterRegionRelationPath,
    BlueBrainThreeRegionConsistencyClass::CaveatedThreeRegionPath,
    BlueBrainThreeRegionConsistencyClass::BlockedInsufficientThreeRegionPath,
    BlueBrainThreeRegionConsistencyClass::NonCanonicalInternalOnlyThreeRegionPath,
];

pub fn classify_blue_brain_three_region_consistency(
    region1: BlueBrainFirstRegionOutputSurface,
    region2: BlueBrainSecondRegionOutputSurface,
    relation12: BlueBrainInterRegionRelationSurface,
    relation3: BlueBrainThirdRegionRelationSurface,
) -> BlueBrainThreeRegionConsistencyClass {
    if matches!(
        relation12.relation_class,
        BlueBrainInterRegionRelationClass::NonCanonicalInternalOnlyPath
    ) || matches!(
        relation3.relation_class,
        BlueBrainThirdRegionRelationClass::NonCanonicalInternalOnlyInterRegionPath
    ) {
        BlueBrainThreeRegionConsistencyClass::NonCanonicalInternalOnlyThreeRegionPath
    } else if relation12.blocked || relation12.deferred || relation3.blocked || relation3.deferred {
        BlueBrainThreeRegionConsistencyClass::BlockedInsufficientThreeRegionPath
    } else if relation12.caveated
        || relation3.caveated
        || region1.reference_only
        || matches!(
            region2.reference_contract_signal,
            BlueBrainSecondRegionContractSignal::ReferenceOnly
        )
    {
        BlueBrainThreeRegionConsistencyClass::CaveatedThreeRegionPath
    } else if matches!(
        relation3.relation_class,
        BlueBrainThirdRegionRelationClass::Region3ToRegion1Bounded
            | BlueBrainThirdRegionRelationClass::Region1ToRegion3Bounded
            | BlueBrainThirdRegionRelationClass::Region3ToRegion2Bounded
            | BlueBrainThirdRegionRelationClass::Region2ToRegion3Bounded
            | BlueBrainThirdRegionRelationClass::SharedReferenceMediatedRelation
    ) || matches!(
        relation12.relation_class,
        BlueBrainInterRegionRelationClass::Region1ToRegion2Bounded
            | BlueBrainInterRegionRelationClass::Region2ToRegion1Bounded
            | BlueBrainInterRegionRelationClass::SharedReferenceMediated
    ) {
        BlueBrainThreeRegionConsistencyClass::BoundedInterRegionRelationPath
    } else if matches!(
        relation3.relation_class,
        BlueBrainThirdRegionRelationClass::Region3ToRegion1Bounded
            | BlueBrainThirdRegionRelationClass::Region3ToRegion2Bounded
    ) {
        BlueBrainThreeRegionConsistencyClass::CanonicalRegion3Path
    } else if matches!(
        region2.runtime_contract_signal,
        BlueBrainSecondRegionContractSignal::RegionToRuntimeAdvisory
    ) {
        BlueBrainThreeRegionConsistencyClass::CanonicalRegion2Path
    } else {
        BlueBrainThreeRegionConsistencyClass::CanonicalRegion1Path
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainTwoRegionMaintenanceFindingClass {
    RealBug,
    SemanticInconsistency,
    GuardWeakness,
    DocTestDrift,
    NonCanonicalResidualPath,
    NoChangeNeededFinding,
}

pub const CANONICAL_BLUE_BRAIN_TWO_REGION_MAINTENANCE_FINDINGS_MAP:
    [BlueBrainTwoRegionMaintenanceFindingClass; 6] = [
    BlueBrainTwoRegionMaintenanceFindingClass::RealBug,
    BlueBrainTwoRegionMaintenanceFindingClass::SemanticInconsistency,
    BlueBrainTwoRegionMaintenanceFindingClass::GuardWeakness,
    BlueBrainTwoRegionMaintenanceFindingClass::DocTestDrift,
    BlueBrainTwoRegionMaintenanceFindingClass::NonCanonicalResidualPath,
    BlueBrainTwoRegionMaintenanceFindingClass::NoChangeNeededFinding,
];

pub fn classify_blue_brain_two_region_consistency(
    region1: BlueBrainFirstRegionOutputSurface,
    region2: BlueBrainSecondRegionOutputSurface,
    relation: BlueBrainInterRegionRelationSurface,
) -> BlueBrainTwoRegionConsistencyClass {
    if matches!(
        relation.relation_class,
        BlueBrainInterRegionRelationClass::NonCanonicalInternalOnlyPath
    ) {
        BlueBrainTwoRegionConsistencyClass::NonCanonicalInternalOnlyTwoRegionPath
    } else if relation.blocked || relation.deferred {
        BlueBrainTwoRegionConsistencyClass::BlockedInsufficientTwoRegionPath
    } else if relation.caveated
        || region1.reference_only
        || matches!(
            region2.reference_contract_signal,
            BlueBrainSecondRegionContractSignal::ReferenceOnly
        )
    {
        BlueBrainTwoRegionConsistencyClass::CaveatedTwoRegionPath
    } else if matches!(
        relation.relation_class,
        BlueBrainInterRegionRelationClass::Region1ToRegion2Bounded
            | BlueBrainInterRegionRelationClass::Region2ToRegion1Bounded
            | BlueBrainInterRegionRelationClass::SharedReferenceMediated
    ) {
        BlueBrainTwoRegionConsistencyClass::BoundedInterRegionRelationPath
    } else if matches!(
        region2.runtime_contract_signal,
        BlueBrainSecondRegionContractSignal::RegionToRuntimeAdvisory
    ) {
        BlueBrainTwoRegionConsistencyClass::CanonicalRegion2Path
    } else {
        BlueBrainTwoRegionConsistencyClass::CanonicalRegion1Path
    }
}

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
pub enum BlueBrainThalamusStateSurface {
    ActiveBoundedRelayAdvisoryOnly,
    CaveatedReferenceRoutingState,
    DeferredRoutingState,
    BlockedRoutingState,
    InsufficientRoutingState,
    ReferenceOnlyRoutingState,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainThalamusInputSource {
    RuntimeRelaySignal,
    SelectionGatingSignal,
    RoutingDeferralSignal,
    ContextReferenceSignal,
    ReferenceValiditySignal,
    ToolActionControlSignal,
    ComputeInternalRawStateSignal,
    SafetyOverrideSignal,
    ImplicitMemoryMutationSignal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainThalamusInputGuard {
    CanonicalBoundedInput,
    RejectedToolActionControl,
    RejectedComputeInternalRawState,
    RejectedSafetyOverride,
    RejectedImplicitMemoryMutation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainThalamusInputSurface {
    pub selection_signal: BlueBrainControlAttentionSelectionClass,
    pub deferral_class: BlueBrainCandidateDeferralLifecycleClass,
    pub reference_validity: BlueBrainReferenceValidity,
    pub context_priority: BlueBrainContextEvidencePriorityClass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainThalamusAdvisoryOutputClass {
    RelayHint,
    RoutingHint,
    GatingHint,
    CaveatHint,
    ReferenceBoundedSignal,
    BlockedDeferred,
    InsufficientDiagnosticOutput,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainThalamusContractSignal {
    ThalamusToRuntimeAdvisory,
    RuntimeToThalamusBoundedInput,
    ThalamusToSelectionAdvisory,
    SelectionToThalamusBoundedStateInput,
    ThalamusReferenceSignal,
    Caveated,
    Deferred,
    Blocked,
    Insufficient,
    ReferenceOnly,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainThalamusDiagnosticState {
    ThalamusAdvisoryOnlyDiagnostic,
    ThalamusCaveatedDiagnostic,
    ThalamusDeferredDiagnostic,
    ThalamusBlockedDiagnostic,
    ThalamusInsufficientDiagnostic,
    ThalamusDiagnosticOnlyState,
    NonCanonicalInternalOnlyThalamusDiagnosticPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainThalamusContractClass {
    ThalamusAdvisoryOnlyDiagnostic,
    ThalamusCaveatedDiagnostic,
    ThalamusDeferredDiagnostic,
    ThalamusBlockedDiagnostic,
    ThalamusInsufficientDiagnostic,
    ThalamusDiagnosticOnlyState,
    ThalamusBoundedContractSignal,
    NonCanonicalInternalOnlyThalamusPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainThalamusCanonicalRead {
    AdvisoryOnly,
    Caveated,
    Deferred,
    Blocked,
    Insufficient,
    DiagnosticOnly,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainThalamusConsumerLayer {
    Runtime,
    Selection,
    Routing,
    Reference,
}

pub const CANONICAL_BLUE_BRAIN_THALAMUS_DIAGNOSTICS_CONTRACT_MAP: [BlueBrainThalamusContractClass;
    8] = [
    BlueBrainThalamusContractClass::ThalamusAdvisoryOnlyDiagnostic,
    BlueBrainThalamusContractClass::ThalamusCaveatedDiagnostic,
    BlueBrainThalamusContractClass::ThalamusDeferredDiagnostic,
    BlueBrainThalamusContractClass::ThalamusBlockedDiagnostic,
    BlueBrainThalamusContractClass::ThalamusInsufficientDiagnostic,
    BlueBrainThalamusContractClass::ThalamusDiagnosticOnlyState,
    BlueBrainThalamusContractClass::ThalamusBoundedContractSignal,
    BlueBrainThalamusContractClass::NonCanonicalInternalOnlyThalamusPath,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainThalamusOutputSurface {
    pub advisory_class: BlueBrainThalamusAdvisoryOutputClass,
    pub runtime_contract_signal: BlueBrainThalamusContractSignal,
    pub selection_contract_signal: BlueBrainThalamusContractSignal,
    pub routing_contract_signal: BlueBrainThalamusContractSignal,
    pub reference_contract_signal: BlueBrainThalamusContractSignal,
    pub runtime_diagnostic_state: BlueBrainThalamusDiagnosticState,
    pub selection_diagnostic_state: BlueBrainThalamusDiagnosticState,
    pub routing_diagnostic_state: BlueBrainThalamusDiagnosticState,
    pub reference_diagnostic_state: BlueBrainThalamusDiagnosticState,
    pub canonical_contract_read: BlueBrainThalamusCanonicalRead,
    pub runtime_advisory_only: bool,
    pub selection_advisory_only: bool,
    pub routing_advisory_only: bool,
    pub reference_bounded_only: bool,
    pub direct_action_selection: bool,
    pub direct_execution_trigger: bool,
    pub direct_retry_trigger: bool,
    pub direct_memory_commit: bool,
    pub direct_compute_invocation: bool,
    pub safety_override: bool,
}

pub fn classify_blue_brain_thalamus_input_guard(
    source: BlueBrainThalamusInputSource,
) -> BlueBrainThalamusInputGuard {
    match source {
        BlueBrainThalamusInputSource::RuntimeRelaySignal
        | BlueBrainThalamusInputSource::SelectionGatingSignal
        | BlueBrainThalamusInputSource::RoutingDeferralSignal
        | BlueBrainThalamusInputSource::ContextReferenceSignal
        | BlueBrainThalamusInputSource::ReferenceValiditySignal => {
            BlueBrainThalamusInputGuard::CanonicalBoundedInput
        }
        BlueBrainThalamusInputSource::ToolActionControlSignal => {
            BlueBrainThalamusInputGuard::RejectedToolActionControl
        }
        BlueBrainThalamusInputSource::ComputeInternalRawStateSignal => {
            BlueBrainThalamusInputGuard::RejectedComputeInternalRawState
        }
        BlueBrainThalamusInputSource::SafetyOverrideSignal => {
            BlueBrainThalamusInputGuard::RejectedSafetyOverride
        }
        BlueBrainThalamusInputSource::ImplicitMemoryMutationSignal => {
            BlueBrainThalamusInputGuard::RejectedImplicitMemoryMutation
        }
    }
}

pub fn blue_brain_thalamus_diagnostic_state_for_signal(
    signal: BlueBrainThalamusContractSignal,
) -> BlueBrainThalamusDiagnosticState {
    match signal {
        BlueBrainThalamusContractSignal::ThalamusToRuntimeAdvisory
        | BlueBrainThalamusContractSignal::RuntimeToThalamusBoundedInput
        | BlueBrainThalamusContractSignal::ThalamusToSelectionAdvisory
        | BlueBrainThalamusContractSignal::SelectionToThalamusBoundedStateInput => {
            BlueBrainThalamusDiagnosticState::ThalamusAdvisoryOnlyDiagnostic
        }
        BlueBrainThalamusContractSignal::Caveated => {
            BlueBrainThalamusDiagnosticState::ThalamusCaveatedDiagnostic
        }
        BlueBrainThalamusContractSignal::Deferred => {
            BlueBrainThalamusDiagnosticState::ThalamusDeferredDiagnostic
        }
        BlueBrainThalamusContractSignal::Blocked => {
            BlueBrainThalamusDiagnosticState::ThalamusBlockedDiagnostic
        }
        BlueBrainThalamusContractSignal::Insufficient => {
            BlueBrainThalamusDiagnosticState::ThalamusInsufficientDiagnostic
        }
        BlueBrainThalamusContractSignal::ReferenceOnly
        | BlueBrainThalamusContractSignal::ThalamusReferenceSignal => {
            BlueBrainThalamusDiagnosticState::ThalamusDiagnosticOnlyState
        }
        BlueBrainThalamusContractSignal::NonCanonicalInternalOnly => {
            BlueBrainThalamusDiagnosticState::NonCanonicalInternalOnlyThalamusDiagnosticPath
        }
    }
}

pub fn blue_brain_thalamus_contract_class_for_signal(
    signal: BlueBrainThalamusContractSignal,
) -> BlueBrainThalamusContractClass {
    match signal {
        BlueBrainThalamusContractSignal::ThalamusToRuntimeAdvisory
        | BlueBrainThalamusContractSignal::ThalamusToSelectionAdvisory => {
            BlueBrainThalamusContractClass::ThalamusAdvisoryOnlyDiagnostic
        }
        BlueBrainThalamusContractSignal::RuntimeToThalamusBoundedInput
        | BlueBrainThalamusContractSignal::SelectionToThalamusBoundedStateInput
        | BlueBrainThalamusContractSignal::ThalamusReferenceSignal => {
            BlueBrainThalamusContractClass::ThalamusBoundedContractSignal
        }
        BlueBrainThalamusContractSignal::Caveated => {
            BlueBrainThalamusContractClass::ThalamusCaveatedDiagnostic
        }
        BlueBrainThalamusContractSignal::Deferred => {
            BlueBrainThalamusContractClass::ThalamusDeferredDiagnostic
        }
        BlueBrainThalamusContractSignal::Blocked => {
            BlueBrainThalamusContractClass::ThalamusBlockedDiagnostic
        }
        BlueBrainThalamusContractSignal::Insufficient => {
            BlueBrainThalamusContractClass::ThalamusInsufficientDiagnostic
        }
        BlueBrainThalamusContractSignal::ReferenceOnly => {
            BlueBrainThalamusContractClass::ThalamusDiagnosticOnlyState
        }
        BlueBrainThalamusContractSignal::NonCanonicalInternalOnly => {
            BlueBrainThalamusContractClass::NonCanonicalInternalOnlyThalamusPath
        }
    }
}

pub fn blue_brain_thalamus_canonical_read_for_state(
    state: BlueBrainThalamusStateSurface,
) -> BlueBrainThalamusCanonicalRead {
    match state {
        BlueBrainThalamusStateSurface::ActiveBoundedRelayAdvisoryOnly => {
            BlueBrainThalamusCanonicalRead::AdvisoryOnly
        }
        BlueBrainThalamusStateSurface::CaveatedReferenceRoutingState => {
            BlueBrainThalamusCanonicalRead::Caveated
        }
        BlueBrainThalamusStateSurface::DeferredRoutingState => {
            BlueBrainThalamusCanonicalRead::Deferred
        }
        BlueBrainThalamusStateSurface::BlockedRoutingState => {
            BlueBrainThalamusCanonicalRead::Blocked
        }
        BlueBrainThalamusStateSurface::InsufficientRoutingState => {
            BlueBrainThalamusCanonicalRead::Insufficient
        }
        BlueBrainThalamusStateSurface::ReferenceOnlyRoutingState => {
            BlueBrainThalamusCanonicalRead::DiagnosticOnly
        }
        BlueBrainThalamusStateSurface::NonCanonicalInternalOnly => {
            BlueBrainThalamusCanonicalRead::NonCanonicalInternalOnly
        }
    }
}

pub fn blue_brain_thalamus_consumer_contract_read(
    output: BlueBrainThalamusOutputSurface,
    _layer: BlueBrainThalamusConsumerLayer,
) -> BlueBrainThalamusCanonicalRead {
    output.canonical_contract_read
}
pub fn evaluate_blue_brain_thalamus_relay_routing(
    input: BlueBrainThalamusInputSurface,
) -> (
    BlueBrainThalamusStateSurface,
    BlueBrainThalamusOutputSurface,
) {
    let (state, advisory_class, runtime_signal, selection_signal, routing_signal, reference_signal) =
        if input.selection_signal
            == BlueBrainControlAttentionSelectionClass::NonCanonicalInternalOnlySelectionPath
            || input.context_priority
                == BlueBrainContextEvidencePriorityClass::NonCanonicalInternalOnlyPriorityPath
            || input.reference_validity == BlueBrainReferenceValidity::NonCanonicalInternalOnlyPath
        {
            (
                BlueBrainThalamusStateSurface::NonCanonicalInternalOnly,
                BlueBrainThalamusAdvisoryOutputClass::NonCanonicalInternalOnly,
                BlueBrainThalamusContractSignal::NonCanonicalInternalOnly,
                BlueBrainThalamusContractSignal::NonCanonicalInternalOnly,
                BlueBrainThalamusContractSignal::NonCanonicalInternalOnly,
                BlueBrainThalamusContractSignal::NonCanonicalInternalOnly,
            )
        } else if input.deferral_class
            == BlueBrainCandidateDeferralLifecycleClass::CandidateRejected
            || input.reference_validity == BlueBrainReferenceValidity::Blocked
        {
            (
                BlueBrainThalamusStateSurface::BlockedRoutingState,
                BlueBrainThalamusAdvisoryOutputClass::BlockedDeferred,
                BlueBrainThalamusContractSignal::Blocked,
                BlueBrainThalamusContractSignal::Blocked,
                BlueBrainThalamusContractSignal::Blocked,
                BlueBrainThalamusContractSignal::Blocked,
            )
        } else if input.deferral_class
            == BlueBrainCandidateDeferralLifecycleClass::CandidateInsufficient
            || input.reference_validity == BlueBrainReferenceValidity::Insufficient
            || input.context_priority == BlueBrainContextEvidencePriorityClass::InsufficientContext
        {
            (
                BlueBrainThalamusStateSurface::InsufficientRoutingState,
                BlueBrainThalamusAdvisoryOutputClass::InsufficientDiagnosticOutput,
                BlueBrainThalamusContractSignal::Insufficient,
                BlueBrainThalamusContractSignal::Insufficient,
                BlueBrainThalamusContractSignal::Insufficient,
                BlueBrainThalamusContractSignal::Insufficient,
            )
        } else if matches!(
            input.deferral_class,
            BlueBrainCandidateDeferralLifecycleClass::CandidateDeferred
                | BlueBrainCandidateDeferralLifecycleClass::CandidateDeferredPendingStrongerEvidence
                | BlueBrainCandidateDeferralLifecycleClass::CandidateDeferredPendingContextUpdate
                | BlueBrainCandidateDeferralLifecycleClass::CandidateStale
        ) || input.reference_validity == BlueBrainReferenceValidity::Stale
        {
            (
                BlueBrainThalamusStateSurface::DeferredRoutingState,
                BlueBrainThalamusAdvisoryOutputClass::RoutingHint,
                BlueBrainThalamusContractSignal::Deferred,
                BlueBrainThalamusContractSignal::Deferred,
                BlueBrainThalamusContractSignal::Deferred,
                BlueBrainThalamusContractSignal::Deferred,
            )
        } else if input.reference_validity == BlueBrainReferenceValidity::ReferenceOnly {
            (
                BlueBrainThalamusStateSurface::ReferenceOnlyRoutingState,
                BlueBrainThalamusAdvisoryOutputClass::ReferenceBoundedSignal,
                BlueBrainThalamusContractSignal::ThalamusToRuntimeAdvisory,
                BlueBrainThalamusContractSignal::ThalamusToSelectionAdvisory,
                BlueBrainThalamusContractSignal::ThalamusReferenceSignal,
                BlueBrainThalamusContractSignal::ReferenceOnly,
            )
        } else if input.reference_validity == BlueBrainReferenceValidity::Caveated
            || input.context_priority
                == BlueBrainContextEvidencePriorityClass::CaveatedEvidenceReference
        {
            (
                BlueBrainThalamusStateSurface::CaveatedReferenceRoutingState,
                BlueBrainThalamusAdvisoryOutputClass::CaveatHint,
                BlueBrainThalamusContractSignal::Caveated,
                BlueBrainThalamusContractSignal::Caveated,
                BlueBrainThalamusContractSignal::Caveated,
                BlueBrainThalamusContractSignal::Caveated,
            )
        } else if matches!(
            input.selection_signal,
            BlueBrainControlAttentionSelectionClass::AttentionTarget
                | BlueBrainControlAttentionSelectionClass::ContextSelection
        ) {
            (
                BlueBrainThalamusStateSurface::ActiveBoundedRelayAdvisoryOnly,
                BlueBrainThalamusAdvisoryOutputClass::GatingHint,
                BlueBrainThalamusContractSignal::ThalamusToRuntimeAdvisory,
                BlueBrainThalamusContractSignal::ThalamusToSelectionAdvisory,
                BlueBrainThalamusContractSignal::SelectionToThalamusBoundedStateInput,
                BlueBrainThalamusContractSignal::ThalamusReferenceSignal,
            )
        } else {
            (
                BlueBrainThalamusStateSurface::ActiveBoundedRelayAdvisoryOnly,
                BlueBrainThalamusAdvisoryOutputClass::RelayHint,
                BlueBrainThalamusContractSignal::ThalamusToRuntimeAdvisory,
                BlueBrainThalamusContractSignal::ThalamusToSelectionAdvisory,
                BlueBrainThalamusContractSignal::RuntimeToThalamusBoundedInput,
                BlueBrainThalamusContractSignal::ThalamusReferenceSignal,
            )
        };

    let output = BlueBrainThalamusOutputSurface {
        advisory_class,
        runtime_contract_signal: runtime_signal,
        selection_contract_signal: selection_signal,
        routing_contract_signal: routing_signal,
        reference_contract_signal: reference_signal,
        runtime_diagnostic_state: blue_brain_thalamus_diagnostic_state_for_signal(runtime_signal),
        selection_diagnostic_state: blue_brain_thalamus_diagnostic_state_for_signal(
            selection_signal,
        ),
        routing_diagnostic_state: blue_brain_thalamus_diagnostic_state_for_signal(routing_signal),
        reference_diagnostic_state: blue_brain_thalamus_diagnostic_state_for_signal(
            reference_signal,
        ),
        canonical_contract_read: blue_brain_thalamus_canonical_read_for_state(state),
        runtime_advisory_only: true,
        selection_advisory_only: true,
        routing_advisory_only: true,
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
pub enum BlueBrainBasalGangliaStateSurface {
    ActiveBoundedActionGatingAdvisoryOnly,
    SuppressionInhibitionAdvisoryState,
    ChannelSelectionArbitrationAdvisoryState,
    ExecutionReadinessCaveatState,
    ReferenceOnlyActionGatingState,
    DeferredActionGatingState,
    BlockedActionGatingState,
    InsufficientActionGatingState,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainBasalGangliaInputSource {
    RuntimeReadinessSignal,
    SelectionPrioritySignal,
    SelectionDeferralSignal,
    ActionGatingPostureSignal,
    ContextReferenceSignal,
    ReferenceValiditySignal,
    ToolActionControlSignal,
    ComputeInternalRawStateSignal,
    SafetyOverrideSignal,
    ImplicitMemoryMutationSignal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainBasalGangliaInputGuard {
    CanonicalBoundedInput,
    ReferenceOnlyBoundedInput,
    AdvisoryOnlyInput,
    RejectedToolActionControl,
    RejectedComputeInternalRawState,
    RejectedSafetyOverride,
    RejectedImplicitMemoryMutation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainBasalGangliaInputSurface {
    pub selection_signal: BlueBrainControlAttentionSelectionClass,
    pub deferral_class: BlueBrainCandidateDeferralLifecycleClass,
    pub reference_validity: BlueBrainReferenceValidity,
    pub context_priority: BlueBrainContextEvidencePriorityClass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainBasalGangliaAdvisoryOutputClass {
    GatingHint,
    SuppressionHint,
    ChannelSelectionHint,
    ExecutionReadinessCaveat,
    ReferenceBoundedSignal,
    BlockedDeferred,
    InsufficientDiagnosticOutput,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainBasalGangliaContractSignal {
    BasalGangliaToRuntimeAdvisory,
    RuntimeToBasalGangliaBoundedReadinessInput,
    BasalGangliaToSelectionAdvisory,
    SelectionToBasalGangliaBoundedActionGatingInput,
    BasalGangliaReferenceSignal,
    Caveated,
    Deferred,
    Blocked,
    Insufficient,
    ReferenceOnly,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainBasalGangliaDiagnosticState {
    BasalGangliaAdvisoryOnlyDiagnostic,
    BasalGangliaCaveatedDiagnostic,
    BasalGangliaDeferredDiagnostic,
    BasalGangliaBlockedDiagnostic,
    BasalGangliaInsufficientDiagnostic,
    BasalGangliaDiagnosticOnlyState,
    NonCanonicalInternalOnlyBasalGangliaDiagnosticPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainBasalGangliaContractClass {
    BasalGangliaAdvisoryOnlyDiagnostic,
    BasalGangliaCaveatedDiagnostic,
    BasalGangliaDeferredDiagnostic,
    BasalGangliaBlockedDiagnostic,
    BasalGangliaInsufficientDiagnostic,
    BasalGangliaDiagnosticOnlyState,
    BasalGangliaBoundedContractSignal,
    NonCanonicalInternalOnlyBasalGangliaPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainBasalGangliaCanonicalRead {
    AdvisoryOnly,
    Caveated,
    Deferred,
    Blocked,
    Insufficient,
    DiagnosticOnly,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainBasalGangliaConsumerLayer {
    Runtime,
    Selection,
    ExecutionInterface,
    Reference,
}

pub const CANONICAL_BLUE_BRAIN_BASAL_GANGLIA_DIAGNOSTICS_CONTRACT_MAP:
    [BlueBrainBasalGangliaContractClass; 8] = [
    BlueBrainBasalGangliaContractClass::BasalGangliaAdvisoryOnlyDiagnostic,
    BlueBrainBasalGangliaContractClass::BasalGangliaCaveatedDiagnostic,
    BlueBrainBasalGangliaContractClass::BasalGangliaDeferredDiagnostic,
    BlueBrainBasalGangliaContractClass::BasalGangliaBlockedDiagnostic,
    BlueBrainBasalGangliaContractClass::BasalGangliaInsufficientDiagnostic,
    BlueBrainBasalGangliaContractClass::BasalGangliaDiagnosticOnlyState,
    BlueBrainBasalGangliaContractClass::BasalGangliaBoundedContractSignal,
    BlueBrainBasalGangliaContractClass::NonCanonicalInternalOnlyBasalGangliaPath,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainBasalGangliaOutputSurface {
    pub advisory_class: BlueBrainBasalGangliaAdvisoryOutputClass,
    pub runtime_contract_signal: BlueBrainBasalGangliaContractSignal,
    pub selection_contract_signal: BlueBrainBasalGangliaContractSignal,
    pub execution_contract_signal: BlueBrainBasalGangliaContractSignal,
    pub reference_contract_signal: BlueBrainBasalGangliaContractSignal,
    pub runtime_diagnostic_state: BlueBrainBasalGangliaDiagnosticState,
    pub selection_diagnostic_state: BlueBrainBasalGangliaDiagnosticState,
    pub execution_diagnostic_state: BlueBrainBasalGangliaDiagnosticState,
    pub reference_diagnostic_state: BlueBrainBasalGangliaDiagnosticState,
    pub canonical_contract_read: BlueBrainBasalGangliaCanonicalRead,
    pub runtime_advisory_only: bool,
    pub selection_advisory_only: bool,
    pub execution_readiness_caveat_only: bool,
    pub reference_bounded_only: bool,
    pub direct_action_selection: bool,
    pub direct_action_trigger: bool,
    pub direct_execution_trigger: bool,
    pub direct_retry_trigger: bool,
    pub direct_memory_commit: bool,
    pub direct_compute_invocation: bool,
    pub safety_override: bool,
}

pub fn classify_blue_brain_basal_ganglia_input_guard(
    source: BlueBrainBasalGangliaInputSource,
) -> BlueBrainBasalGangliaInputGuard {
    match source {
        BlueBrainBasalGangliaInputSource::RuntimeReadinessSignal
        | BlueBrainBasalGangliaInputSource::SelectionPrioritySignal
        | BlueBrainBasalGangliaInputSource::SelectionDeferralSignal
        | BlueBrainBasalGangliaInputSource::ActionGatingPostureSignal => {
            BlueBrainBasalGangliaInputGuard::AdvisoryOnlyInput
        }
        BlueBrainBasalGangliaInputSource::ContextReferenceSignal
        | BlueBrainBasalGangliaInputSource::ReferenceValiditySignal => {
            BlueBrainBasalGangliaInputGuard::ReferenceOnlyBoundedInput
        }
        BlueBrainBasalGangliaInputSource::ToolActionControlSignal => {
            BlueBrainBasalGangliaInputGuard::RejectedToolActionControl
        }
        BlueBrainBasalGangliaInputSource::ComputeInternalRawStateSignal => {
            BlueBrainBasalGangliaInputGuard::RejectedComputeInternalRawState
        }
        BlueBrainBasalGangliaInputSource::SafetyOverrideSignal => {
            BlueBrainBasalGangliaInputGuard::RejectedSafetyOverride
        }
        BlueBrainBasalGangliaInputSource::ImplicitMemoryMutationSignal => {
            BlueBrainBasalGangliaInputGuard::RejectedImplicitMemoryMutation
        }
    }
}

pub fn blue_brain_basal_ganglia_diagnostic_state_for_signal(
    signal: BlueBrainBasalGangliaContractSignal,
) -> BlueBrainBasalGangliaDiagnosticState {
    match signal {
        BlueBrainBasalGangliaContractSignal::BasalGangliaToRuntimeAdvisory
        | BlueBrainBasalGangliaContractSignal::RuntimeToBasalGangliaBoundedReadinessInput
        | BlueBrainBasalGangliaContractSignal::BasalGangliaToSelectionAdvisory
        | BlueBrainBasalGangliaContractSignal::SelectionToBasalGangliaBoundedActionGatingInput => {
            BlueBrainBasalGangliaDiagnosticState::BasalGangliaAdvisoryOnlyDiagnostic
        }
        BlueBrainBasalGangliaContractSignal::Caveated => {
            BlueBrainBasalGangliaDiagnosticState::BasalGangliaCaveatedDiagnostic
        }
        BlueBrainBasalGangliaContractSignal::Deferred => {
            BlueBrainBasalGangliaDiagnosticState::BasalGangliaDeferredDiagnostic
        }
        BlueBrainBasalGangliaContractSignal::Blocked => {
            BlueBrainBasalGangliaDiagnosticState::BasalGangliaBlockedDiagnostic
        }
        BlueBrainBasalGangliaContractSignal::Insufficient => {
            BlueBrainBasalGangliaDiagnosticState::BasalGangliaInsufficientDiagnostic
        }
        BlueBrainBasalGangliaContractSignal::ReferenceOnly
        | BlueBrainBasalGangliaContractSignal::BasalGangliaReferenceSignal => {
            BlueBrainBasalGangliaDiagnosticState::BasalGangliaDiagnosticOnlyState
        }
        BlueBrainBasalGangliaContractSignal::NonCanonicalInternalOnly => {
            BlueBrainBasalGangliaDiagnosticState::NonCanonicalInternalOnlyBasalGangliaDiagnosticPath
        }
    }
}

pub fn blue_brain_basal_ganglia_contract_class_for_signal(
    signal: BlueBrainBasalGangliaContractSignal,
) -> BlueBrainBasalGangliaContractClass {
    match signal {
        BlueBrainBasalGangliaContractSignal::BasalGangliaToRuntimeAdvisory
        | BlueBrainBasalGangliaContractSignal::BasalGangliaToSelectionAdvisory => {
            BlueBrainBasalGangliaContractClass::BasalGangliaAdvisoryOnlyDiagnostic
        }
        BlueBrainBasalGangliaContractSignal::RuntimeToBasalGangliaBoundedReadinessInput
        | BlueBrainBasalGangliaContractSignal::SelectionToBasalGangliaBoundedActionGatingInput
        | BlueBrainBasalGangliaContractSignal::BasalGangliaReferenceSignal => {
            BlueBrainBasalGangliaContractClass::BasalGangliaBoundedContractSignal
        }
        BlueBrainBasalGangliaContractSignal::Caveated => {
            BlueBrainBasalGangliaContractClass::BasalGangliaCaveatedDiagnostic
        }
        BlueBrainBasalGangliaContractSignal::Deferred => {
            BlueBrainBasalGangliaContractClass::BasalGangliaDeferredDiagnostic
        }
        BlueBrainBasalGangliaContractSignal::Blocked => {
            BlueBrainBasalGangliaContractClass::BasalGangliaBlockedDiagnostic
        }
        BlueBrainBasalGangliaContractSignal::Insufficient => {
            BlueBrainBasalGangliaContractClass::BasalGangliaInsufficientDiagnostic
        }
        BlueBrainBasalGangliaContractSignal::ReferenceOnly => {
            BlueBrainBasalGangliaContractClass::BasalGangliaDiagnosticOnlyState
        }
        BlueBrainBasalGangliaContractSignal::NonCanonicalInternalOnly => {
            BlueBrainBasalGangliaContractClass::NonCanonicalInternalOnlyBasalGangliaPath
        }
    }
}

pub fn blue_brain_basal_ganglia_canonical_read_for_state(
    state: BlueBrainBasalGangliaStateSurface,
) -> BlueBrainBasalGangliaCanonicalRead {
    match state {
        BlueBrainBasalGangliaStateSurface::ActiveBoundedActionGatingAdvisoryOnly
        | BlueBrainBasalGangliaStateSurface::SuppressionInhibitionAdvisoryState
        | BlueBrainBasalGangliaStateSurface::ChannelSelectionArbitrationAdvisoryState => {
            BlueBrainBasalGangliaCanonicalRead::AdvisoryOnly
        }
        BlueBrainBasalGangliaStateSurface::ExecutionReadinessCaveatState => {
            BlueBrainBasalGangliaCanonicalRead::Caveated
        }
        BlueBrainBasalGangliaStateSurface::DeferredActionGatingState => {
            BlueBrainBasalGangliaCanonicalRead::Deferred
        }
        BlueBrainBasalGangliaStateSurface::BlockedActionGatingState => {
            BlueBrainBasalGangliaCanonicalRead::Blocked
        }
        BlueBrainBasalGangliaStateSurface::InsufficientActionGatingState => {
            BlueBrainBasalGangliaCanonicalRead::Insufficient
        }
        BlueBrainBasalGangliaStateSurface::ReferenceOnlyActionGatingState => {
            BlueBrainBasalGangliaCanonicalRead::DiagnosticOnly
        }
        BlueBrainBasalGangliaStateSurface::NonCanonicalInternalOnly => {
            BlueBrainBasalGangliaCanonicalRead::NonCanonicalInternalOnly
        }
    }
}

pub fn blue_brain_basal_ganglia_consumer_contract_read(
    output: BlueBrainBasalGangliaOutputSurface,
    _layer: BlueBrainBasalGangliaConsumerLayer,
) -> BlueBrainBasalGangliaCanonicalRead {
    output.canonical_contract_read
}

pub fn evaluate_blue_brain_basal_ganglia_action_gating(
    input: BlueBrainBasalGangliaInputSurface,
) -> (
    BlueBrainBasalGangliaStateSurface,
    BlueBrainBasalGangliaOutputSurface,
) {
    let (
        state,
        advisory_class,
        runtime_signal,
        selection_signal,
        execution_signal,
        reference_signal,
    ) = if input.selection_signal
        == BlueBrainControlAttentionSelectionClass::NonCanonicalInternalOnlySelectionPath
        || input.context_priority
            == BlueBrainContextEvidencePriorityClass::NonCanonicalInternalOnlyPriorityPath
        || input.reference_validity == BlueBrainReferenceValidity::NonCanonicalInternalOnlyPath
    {
        (
            BlueBrainBasalGangliaStateSurface::NonCanonicalInternalOnly,
            BlueBrainBasalGangliaAdvisoryOutputClass::NonCanonicalInternalOnly,
            BlueBrainBasalGangliaContractSignal::NonCanonicalInternalOnly,
            BlueBrainBasalGangliaContractSignal::NonCanonicalInternalOnly,
            BlueBrainBasalGangliaContractSignal::NonCanonicalInternalOnly,
            BlueBrainBasalGangliaContractSignal::NonCanonicalInternalOnly,
        )
    } else if input.deferral_class == BlueBrainCandidateDeferralLifecycleClass::CandidateRejected
        || input.reference_validity == BlueBrainReferenceValidity::Blocked
    {
        (
            BlueBrainBasalGangliaStateSurface::BlockedActionGatingState,
            BlueBrainBasalGangliaAdvisoryOutputClass::BlockedDeferred,
            BlueBrainBasalGangliaContractSignal::Blocked,
            BlueBrainBasalGangliaContractSignal::Blocked,
            BlueBrainBasalGangliaContractSignal::Blocked,
            BlueBrainBasalGangliaContractSignal::Blocked,
        )
    } else if input.deferral_class
        == BlueBrainCandidateDeferralLifecycleClass::CandidateInsufficient
        || input.reference_validity == BlueBrainReferenceValidity::Insufficient
        || input.context_priority == BlueBrainContextEvidencePriorityClass::InsufficientContext
    {
        (
            BlueBrainBasalGangliaStateSurface::InsufficientActionGatingState,
            BlueBrainBasalGangliaAdvisoryOutputClass::InsufficientDiagnosticOutput,
            BlueBrainBasalGangliaContractSignal::Insufficient,
            BlueBrainBasalGangliaContractSignal::Insufficient,
            BlueBrainBasalGangliaContractSignal::Insufficient,
            BlueBrainBasalGangliaContractSignal::Insufficient,
        )
    } else if matches!(
        input.deferral_class,
        BlueBrainCandidateDeferralLifecycleClass::CandidateDeferred
            | BlueBrainCandidateDeferralLifecycleClass::CandidateDeferredPendingStrongerEvidence
            | BlueBrainCandidateDeferralLifecycleClass::CandidateDeferredPendingContextUpdate
            | BlueBrainCandidateDeferralLifecycleClass::CandidateStale
    ) || input.reference_validity == BlueBrainReferenceValidity::Stale
    {
        (
            BlueBrainBasalGangliaStateSurface::DeferredActionGatingState,
            BlueBrainBasalGangliaAdvisoryOutputClass::SuppressionHint,
            BlueBrainBasalGangliaContractSignal::Deferred,
            BlueBrainBasalGangliaContractSignal::Deferred,
            BlueBrainBasalGangliaContractSignal::Deferred,
            BlueBrainBasalGangliaContractSignal::Deferred,
        )
    } else if input.reference_validity == BlueBrainReferenceValidity::ReferenceOnly {
        (
            BlueBrainBasalGangliaStateSurface::ReferenceOnlyActionGatingState,
            BlueBrainBasalGangliaAdvisoryOutputClass::ReferenceBoundedSignal,
            BlueBrainBasalGangliaContractSignal::BasalGangliaToRuntimeAdvisory,
            BlueBrainBasalGangliaContractSignal::BasalGangliaToSelectionAdvisory,
            BlueBrainBasalGangliaContractSignal::BasalGangliaReferenceSignal,
            BlueBrainBasalGangliaContractSignal::ReferenceOnly,
        )
    } else if input.reference_validity == BlueBrainReferenceValidity::Caveated
        || input.context_priority
            == BlueBrainContextEvidencePriorityClass::CaveatedEvidenceReference
    {
        (
            BlueBrainBasalGangliaStateSurface::ExecutionReadinessCaveatState,
            BlueBrainBasalGangliaAdvisoryOutputClass::ExecutionReadinessCaveat,
            BlueBrainBasalGangliaContractSignal::Caveated,
            BlueBrainBasalGangliaContractSignal::Caveated,
            BlueBrainBasalGangliaContractSignal::Caveated,
            BlueBrainBasalGangliaContractSignal::Caveated,
        )
    } else if matches!(
        input.selection_signal,
        BlueBrainControlAttentionSelectionClass::AttentionTarget
            | BlueBrainControlAttentionSelectionClass::ContextSelection
    ) {
        (
            BlueBrainBasalGangliaStateSurface::ActiveBoundedActionGatingAdvisoryOnly,
            BlueBrainBasalGangliaAdvisoryOutputClass::GatingHint,
            BlueBrainBasalGangliaContractSignal::BasalGangliaToRuntimeAdvisory,
            BlueBrainBasalGangliaContractSignal::BasalGangliaToSelectionAdvisory,
            BlueBrainBasalGangliaContractSignal::RuntimeToBasalGangliaBoundedReadinessInput,
            BlueBrainBasalGangliaContractSignal::BasalGangliaReferenceSignal,
        )
    } else if input.selection_signal
        == BlueBrainControlAttentionSelectionClass::MemoryCandidateSelection
    {
        (
            BlueBrainBasalGangliaStateSurface::SuppressionInhibitionAdvisoryState,
            BlueBrainBasalGangliaAdvisoryOutputClass::SuppressionHint,
            BlueBrainBasalGangliaContractSignal::BasalGangliaToRuntimeAdvisory,
            BlueBrainBasalGangliaContractSignal::BasalGangliaToSelectionAdvisory,
            BlueBrainBasalGangliaContractSignal::SelectionToBasalGangliaBoundedActionGatingInput,
            BlueBrainBasalGangliaContractSignal::BasalGangliaReferenceSignal,
        )
    } else {
        (
            BlueBrainBasalGangliaStateSurface::ChannelSelectionArbitrationAdvisoryState,
            BlueBrainBasalGangliaAdvisoryOutputClass::ChannelSelectionHint,
            BlueBrainBasalGangliaContractSignal::BasalGangliaToRuntimeAdvisory,
            BlueBrainBasalGangliaContractSignal::BasalGangliaToSelectionAdvisory,
            BlueBrainBasalGangliaContractSignal::SelectionToBasalGangliaBoundedActionGatingInput,
            BlueBrainBasalGangliaContractSignal::BasalGangliaReferenceSignal,
        )
    };

    let output = BlueBrainBasalGangliaOutputSurface {
        advisory_class,
        runtime_contract_signal: runtime_signal,
        selection_contract_signal: selection_signal,
        execution_contract_signal: execution_signal,
        reference_contract_signal: reference_signal,
        runtime_diagnostic_state: blue_brain_basal_ganglia_diagnostic_state_for_signal(
            runtime_signal,
        ),
        selection_diagnostic_state: blue_brain_basal_ganglia_diagnostic_state_for_signal(
            selection_signal,
        ),
        execution_diagnostic_state: blue_brain_basal_ganglia_diagnostic_state_for_signal(
            execution_signal,
        ),
        reference_diagnostic_state: blue_brain_basal_ganglia_diagnostic_state_for_signal(
            reference_signal,
        ),
        canonical_contract_read: blue_brain_basal_ganglia_canonical_read_for_state(state),
        runtime_advisory_only: true,
        selection_advisory_only: true,
        execution_readiness_caveat_only: true,
        reference_bounded_only: true,
        direct_action_selection: false,
        direct_action_trigger: false,
        direct_execution_trigger: false,
        direct_retry_trigger: false,
        direct_memory_commit: false,
        direct_compute_invocation: false,
        safety_override: false,
    };
    (state, output)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainCerebellumStateSurface {
    ActivePredictionTimingAdvisoryOnly,
    TimingCoordinationAdvisoryState,
    CorrectionMismatchAdvisoryState,
    ExecutionSupportCaveatState,
    ReferenceOnlyCorrectionState,
    DeferredCorrectionState,
    BlockedCorrectionState,
    InsufficientCorrectionState,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainCerebellumInputSource {
    RuntimePredictionSignal,
    RuntimeTimingSignal,
    SelectionCoordinationSignal,
    ExecutionFeedbackMismatchSignal,
    ContextReferenceSignal,
    ReferenceValiditySignal,
    ToolActionControlSignal,
    ComputeInternalRawStateSignal,
    SafetyOverrideSignal,
    ImplicitMemoryMutationSignal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainCerebellumInputGuard {
    AdvisoryOnlyInput,
    ReferenceOnlyBoundedInput,
    RejectedToolActionControl,
    RejectedComputeInternalRawState,
    RejectedSafetyOverride,
    RejectedImplicitMemoryMutation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainCerebellumInputSurface {
    pub selection_signal: BlueBrainControlAttentionSelectionClass,
    pub deferral_class: BlueBrainCandidateDeferralLifecycleClass,
    pub reference_validity: BlueBrainReferenceValidity,
    pub context_priority: BlueBrainContextEvidencePriorityClass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainCerebellumAdvisoryOutputClass {
    TimingHint,
    CorrectionHint,
    MismatchHint,
    ExecutionSupportCaveat,
    ReferenceBoundedSignal,
    BlockedDeferred,
    InsufficientDiagnosticOutput,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainCerebellumContractSignal {
    CerebellumToRuntimeAdvisory,
    RuntimeToCerebellumBoundedPredictionTimingInput,
    CerebellumToSelectionAdvisory,
    SelectionToCerebellumBoundedCoordinationInput,
    CerebellumExecutionSupportCaveatSignal,
    CerebellumReferenceSignal,
    Caveated,
    Deferred,
    Blocked,
    Insufficient,
    ReferenceOnly,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainCerebellumDiagnosticState {
    CerebellumAdvisoryOnlyDiagnostic,
    CerebellumCaveatedDiagnostic,
    CerebellumDeferredDiagnostic,
    CerebellumBlockedDiagnostic,
    CerebellumInsufficientDiagnostic,
    CerebellumDiagnosticOnlyState,
    NonCanonicalInternalOnlyCerebellumDiagnosticPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainCerebellumContractClass {
    CerebellumAdvisoryOnlyDiagnostic,
    CerebellumCaveatedDiagnostic,
    CerebellumDeferredDiagnostic,
    CerebellumBlockedDiagnostic,
    CerebellumInsufficientDiagnostic,
    CerebellumDiagnosticOnlyState,
    CerebellumBoundedContractSignal,
    NonCanonicalInternalOnlyCerebellumPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainCerebellumCanonicalRead {
    AdvisoryOnly,
    Caveated,
    Deferred,
    Blocked,
    Insufficient,
    DiagnosticOnly,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainCerebellumConsumerLayer {
    Runtime,
    Selection,
    ExecutionInterface,
    Reference,
}

pub const CANONICAL_BLUE_BRAIN_CEREBELLUM_DIAGNOSTICS_CONTRACT_MAP:
    [BlueBrainCerebellumContractClass; 8] = [
    BlueBrainCerebellumContractClass::CerebellumAdvisoryOnlyDiagnostic,
    BlueBrainCerebellumContractClass::CerebellumCaveatedDiagnostic,
    BlueBrainCerebellumContractClass::CerebellumDeferredDiagnostic,
    BlueBrainCerebellumContractClass::CerebellumBlockedDiagnostic,
    BlueBrainCerebellumContractClass::CerebellumInsufficientDiagnostic,
    BlueBrainCerebellumContractClass::CerebellumDiagnosticOnlyState,
    BlueBrainCerebellumContractClass::CerebellumBoundedContractSignal,
    BlueBrainCerebellumContractClass::NonCanonicalInternalOnlyCerebellumPath,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainCerebellumContractMapEntry {
    pub class: BlueBrainCerebellumContractClass,
    pub canonical_read: BlueBrainCerebellumCanonicalRead,
    pub direct_authority_allowed: bool,
}

pub const CANONICAL_BLUE_BRAIN_CEREBELLUM_DIAGNOSTICS_CONTRACT_ENTRIES:
    [BlueBrainCerebellumContractMapEntry; 8] = [
    BlueBrainCerebellumContractMapEntry {
        class: BlueBrainCerebellumContractClass::CerebellumAdvisoryOnlyDiagnostic,
        canonical_read: BlueBrainCerebellumCanonicalRead::AdvisoryOnly,
        direct_authority_allowed: false,
    },
    BlueBrainCerebellumContractMapEntry {
        class: BlueBrainCerebellumContractClass::CerebellumCaveatedDiagnostic,
        canonical_read: BlueBrainCerebellumCanonicalRead::Caveated,
        direct_authority_allowed: false,
    },
    BlueBrainCerebellumContractMapEntry {
        class: BlueBrainCerebellumContractClass::CerebellumDeferredDiagnostic,
        canonical_read: BlueBrainCerebellumCanonicalRead::Deferred,
        direct_authority_allowed: false,
    },
    BlueBrainCerebellumContractMapEntry {
        class: BlueBrainCerebellumContractClass::CerebellumBlockedDiagnostic,
        canonical_read: BlueBrainCerebellumCanonicalRead::Blocked,
        direct_authority_allowed: false,
    },
    BlueBrainCerebellumContractMapEntry {
        class: BlueBrainCerebellumContractClass::CerebellumInsufficientDiagnostic,
        canonical_read: BlueBrainCerebellumCanonicalRead::Insufficient,
        direct_authority_allowed: false,
    },
    BlueBrainCerebellumContractMapEntry {
        class: BlueBrainCerebellumContractClass::CerebellumDiagnosticOnlyState,
        canonical_read: BlueBrainCerebellumCanonicalRead::DiagnosticOnly,
        direct_authority_allowed: false,
    },
    BlueBrainCerebellumContractMapEntry {
        class: BlueBrainCerebellumContractClass::CerebellumBoundedContractSignal,
        canonical_read: BlueBrainCerebellumCanonicalRead::AdvisoryOnly,
        direct_authority_allowed: false,
    },
    BlueBrainCerebellumContractMapEntry {
        class: BlueBrainCerebellumContractClass::NonCanonicalInternalOnlyCerebellumPath,
        canonical_read: BlueBrainCerebellumCanonicalRead::NonCanonicalInternalOnly,
        direct_authority_allowed: false,
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainCerebellumOutputSurface {
    pub advisory_class: BlueBrainCerebellumAdvisoryOutputClass,
    pub runtime_contract_signal: BlueBrainCerebellumContractSignal,
    pub selection_contract_signal: BlueBrainCerebellumContractSignal,
    pub execution_contract_signal: BlueBrainCerebellumContractSignal,
    pub reference_contract_signal: BlueBrainCerebellumContractSignal,
    pub runtime_diagnostic_state: BlueBrainCerebellumDiagnosticState,
    pub selection_diagnostic_state: BlueBrainCerebellumDiagnosticState,
    pub execution_diagnostic_state: BlueBrainCerebellumDiagnosticState,
    pub reference_diagnostic_state: BlueBrainCerebellumDiagnosticState,
    pub canonical_contract_read: BlueBrainCerebellumCanonicalRead,
    pub runtime_advisory_only: bool,
    pub selection_advisory_only: bool,
    pub execution_support_caveat_only: bool,
    pub reference_bounded_only: bool,
    pub direct_action_selection: bool,
    pub direct_action_trigger: bool,
    pub direct_execution_trigger: bool,
    pub direct_retry_trigger: bool,
    pub direct_memory_commit: bool,
    pub direct_compute_invocation: bool,
    pub safety_override: bool,
}

pub fn classify_blue_brain_cerebellum_input_guard(
    source: BlueBrainCerebellumInputSource,
) -> BlueBrainCerebellumInputGuard {
    match source {
        BlueBrainCerebellumInputSource::RuntimePredictionSignal
        | BlueBrainCerebellumInputSource::RuntimeTimingSignal
        | BlueBrainCerebellumInputSource::SelectionCoordinationSignal
        | BlueBrainCerebellumInputSource::ExecutionFeedbackMismatchSignal => {
            BlueBrainCerebellumInputGuard::AdvisoryOnlyInput
        }
        BlueBrainCerebellumInputSource::ContextReferenceSignal
        | BlueBrainCerebellumInputSource::ReferenceValiditySignal => {
            BlueBrainCerebellumInputGuard::ReferenceOnlyBoundedInput
        }
        BlueBrainCerebellumInputSource::ToolActionControlSignal => {
            BlueBrainCerebellumInputGuard::RejectedToolActionControl
        }
        BlueBrainCerebellumInputSource::ComputeInternalRawStateSignal => {
            BlueBrainCerebellumInputGuard::RejectedComputeInternalRawState
        }
        BlueBrainCerebellumInputSource::SafetyOverrideSignal => {
            BlueBrainCerebellumInputGuard::RejectedSafetyOverride
        }
        BlueBrainCerebellumInputSource::ImplicitMemoryMutationSignal => {
            BlueBrainCerebellumInputGuard::RejectedImplicitMemoryMutation
        }
    }
}

pub fn blue_brain_cerebellum_diagnostic_state_for_signal(
    signal: BlueBrainCerebellumContractSignal,
) -> BlueBrainCerebellumDiagnosticState {
    match signal {
        BlueBrainCerebellumContractSignal::CerebellumToRuntimeAdvisory
        | BlueBrainCerebellumContractSignal::RuntimeToCerebellumBoundedPredictionTimingInput
        | BlueBrainCerebellumContractSignal::CerebellumToSelectionAdvisory
        | BlueBrainCerebellumContractSignal::SelectionToCerebellumBoundedCoordinationInput
        | BlueBrainCerebellumContractSignal::CerebellumReferenceSignal => {
            BlueBrainCerebellumDiagnosticState::CerebellumAdvisoryOnlyDiagnostic
        }
        BlueBrainCerebellumContractSignal::CerebellumExecutionSupportCaveatSignal
        | BlueBrainCerebellumContractSignal::Caveated => {
            BlueBrainCerebellumDiagnosticState::CerebellumCaveatedDiagnostic
        }
        BlueBrainCerebellumContractSignal::Deferred => {
            BlueBrainCerebellumDiagnosticState::CerebellumDeferredDiagnostic
        }
        BlueBrainCerebellumContractSignal::Blocked => {
            BlueBrainCerebellumDiagnosticState::CerebellumBlockedDiagnostic
        }
        BlueBrainCerebellumContractSignal::Insufficient => {
            BlueBrainCerebellumDiagnosticState::CerebellumInsufficientDiagnostic
        }
        BlueBrainCerebellumContractSignal::ReferenceOnly => {
            BlueBrainCerebellumDiagnosticState::CerebellumDiagnosticOnlyState
        }
        BlueBrainCerebellumContractSignal::NonCanonicalInternalOnly => {
            BlueBrainCerebellumDiagnosticState::NonCanonicalInternalOnlyCerebellumDiagnosticPath
        }
    }
}

pub fn blue_brain_cerebellum_contract_class_for_signal(
    signal: BlueBrainCerebellumContractSignal,
) -> BlueBrainCerebellumContractClass {
    match signal {
        BlueBrainCerebellumContractSignal::CerebellumToRuntimeAdvisory
        | BlueBrainCerebellumContractSignal::CerebellumToSelectionAdvisory => {
            BlueBrainCerebellumContractClass::CerebellumAdvisoryOnlyDiagnostic
        }
        BlueBrainCerebellumContractSignal::RuntimeToCerebellumBoundedPredictionTimingInput
        | BlueBrainCerebellumContractSignal::SelectionToCerebellumBoundedCoordinationInput
        | BlueBrainCerebellumContractSignal::CerebellumReferenceSignal => {
            BlueBrainCerebellumContractClass::CerebellumBoundedContractSignal
        }
        BlueBrainCerebellumContractSignal::CerebellumExecutionSupportCaveatSignal
        | BlueBrainCerebellumContractSignal::Caveated => {
            BlueBrainCerebellumContractClass::CerebellumCaveatedDiagnostic
        }
        BlueBrainCerebellumContractSignal::Deferred => {
            BlueBrainCerebellumContractClass::CerebellumDeferredDiagnostic
        }
        BlueBrainCerebellumContractSignal::Blocked => {
            BlueBrainCerebellumContractClass::CerebellumBlockedDiagnostic
        }
        BlueBrainCerebellumContractSignal::Insufficient => {
            BlueBrainCerebellumContractClass::CerebellumInsufficientDiagnostic
        }
        BlueBrainCerebellumContractSignal::ReferenceOnly => {
            BlueBrainCerebellumContractClass::CerebellumDiagnosticOnlyState
        }
        BlueBrainCerebellumContractSignal::NonCanonicalInternalOnly => {
            BlueBrainCerebellumContractClass::NonCanonicalInternalOnlyCerebellumPath
        }
    }
}

pub fn blue_brain_cerebellum_canonical_read_for_state(
    state: BlueBrainCerebellumStateSurface,
) -> BlueBrainCerebellumCanonicalRead {
    match state {
        BlueBrainCerebellumStateSurface::ActivePredictionTimingAdvisoryOnly
        | BlueBrainCerebellumStateSurface::TimingCoordinationAdvisoryState
        | BlueBrainCerebellumStateSurface::CorrectionMismatchAdvisoryState => {
            BlueBrainCerebellumCanonicalRead::AdvisoryOnly
        }
        BlueBrainCerebellumStateSurface::ExecutionSupportCaveatState => {
            BlueBrainCerebellumCanonicalRead::Caveated
        }
        BlueBrainCerebellumStateSurface::DeferredCorrectionState => {
            BlueBrainCerebellumCanonicalRead::Deferred
        }
        BlueBrainCerebellumStateSurface::BlockedCorrectionState => {
            BlueBrainCerebellumCanonicalRead::Blocked
        }
        BlueBrainCerebellumStateSurface::InsufficientCorrectionState => {
            BlueBrainCerebellumCanonicalRead::Insufficient
        }
        BlueBrainCerebellumStateSurface::ReferenceOnlyCorrectionState => {
            BlueBrainCerebellumCanonicalRead::DiagnosticOnly
        }
        BlueBrainCerebellumStateSurface::NonCanonicalInternalOnly => {
            BlueBrainCerebellumCanonicalRead::NonCanonicalInternalOnly
        }
    }
}

pub fn blue_brain_cerebellum_canonical_read_for_signal(
    signal: BlueBrainCerebellumContractSignal,
) -> BlueBrainCerebellumCanonicalRead {
    match signal {
        BlueBrainCerebellumContractSignal::CerebellumToRuntimeAdvisory
        | BlueBrainCerebellumContractSignal::RuntimeToCerebellumBoundedPredictionTimingInput
        | BlueBrainCerebellumContractSignal::CerebellumToSelectionAdvisory
        | BlueBrainCerebellumContractSignal::SelectionToCerebellumBoundedCoordinationInput
        | BlueBrainCerebellumContractSignal::CerebellumReferenceSignal => {
            BlueBrainCerebellumCanonicalRead::AdvisoryOnly
        }
        BlueBrainCerebellumContractSignal::CerebellumExecutionSupportCaveatSignal
        | BlueBrainCerebellumContractSignal::Caveated => BlueBrainCerebellumCanonicalRead::Caveated,
        BlueBrainCerebellumContractSignal::Deferred => BlueBrainCerebellumCanonicalRead::Deferred,
        BlueBrainCerebellumContractSignal::Blocked => BlueBrainCerebellumCanonicalRead::Blocked,
        BlueBrainCerebellumContractSignal::Insufficient => {
            BlueBrainCerebellumCanonicalRead::Insufficient
        }
        BlueBrainCerebellumContractSignal::ReferenceOnly => {
            BlueBrainCerebellumCanonicalRead::DiagnosticOnly
        }
        BlueBrainCerebellumContractSignal::NonCanonicalInternalOnly => {
            BlueBrainCerebellumCanonicalRead::NonCanonicalInternalOnly
        }
    }
}

pub fn blue_brain_cerebellum_output_has_no_direct_authority(
    output: BlueBrainCerebellumOutputSurface,
) -> bool {
    !output.direct_action_selection
        && !output.direct_action_trigger
        && !output.direct_execution_trigger
        && !output.direct_retry_trigger
        && !output.direct_memory_commit
        && !output.direct_compute_invocation
        && !output.safety_override
}

pub fn blue_brain_cerebellum_consumer_contract_reads_are_aligned(
    output: BlueBrainCerebellumOutputSurface,
) -> bool {
    let canonical = output.canonical_contract_read;
    [
        BlueBrainCerebellumConsumerLayer::Runtime,
        BlueBrainCerebellumConsumerLayer::Selection,
        BlueBrainCerebellumConsumerLayer::ExecutionInterface,
        BlueBrainCerebellumConsumerLayer::Reference,
    ]
    .into_iter()
    .all(|layer| blue_brain_cerebellum_consumer_contract_read(output, layer) == canonical)
}

pub fn blue_brain_cerebellum_consumer_contract_read(
    output: BlueBrainCerebellumOutputSurface,
    _layer: BlueBrainCerebellumConsumerLayer,
) -> BlueBrainCerebellumCanonicalRead {
    output.canonical_contract_read
}

pub fn evaluate_blue_brain_cerebellum_prediction_timing_correction(
    input: BlueBrainCerebellumInputSurface,
) -> (
    BlueBrainCerebellumStateSurface,
    BlueBrainCerebellumOutputSurface,
) {
    let (
        state,
        advisory_class,
        runtime_signal,
        selection_signal,
        execution_signal,
        reference_signal,
    ) = if input.selection_signal
        == BlueBrainControlAttentionSelectionClass::NonCanonicalInternalOnlySelectionPath
        || input.context_priority
            == BlueBrainContextEvidencePriorityClass::NonCanonicalInternalOnlyPriorityPath
        || input.reference_validity == BlueBrainReferenceValidity::NonCanonicalInternalOnlyPath
    {
        (
            BlueBrainCerebellumStateSurface::NonCanonicalInternalOnly,
            BlueBrainCerebellumAdvisoryOutputClass::NonCanonicalInternalOnly,
            BlueBrainCerebellumContractSignal::NonCanonicalInternalOnly,
            BlueBrainCerebellumContractSignal::NonCanonicalInternalOnly,
            BlueBrainCerebellumContractSignal::NonCanonicalInternalOnly,
            BlueBrainCerebellumContractSignal::NonCanonicalInternalOnly,
        )
    } else if input.deferral_class == BlueBrainCandidateDeferralLifecycleClass::CandidateRejected
        || input.reference_validity == BlueBrainReferenceValidity::Blocked
        || input.reference_validity == BlueBrainReferenceValidity::Invalidated
    {
        (
            BlueBrainCerebellumStateSurface::BlockedCorrectionState,
            BlueBrainCerebellumAdvisoryOutputClass::BlockedDeferred,
            BlueBrainCerebellumContractSignal::Blocked,
            BlueBrainCerebellumContractSignal::Blocked,
            BlueBrainCerebellumContractSignal::Blocked,
            BlueBrainCerebellumContractSignal::Blocked,
        )
    } else if input.deferral_class
        == BlueBrainCandidateDeferralLifecycleClass::CandidateInsufficient
        || input.reference_validity == BlueBrainReferenceValidity::Insufficient
        || input.context_priority == BlueBrainContextEvidencePriorityClass::InsufficientContext
    {
        (
            BlueBrainCerebellumStateSurface::InsufficientCorrectionState,
            BlueBrainCerebellumAdvisoryOutputClass::InsufficientDiagnosticOutput,
            BlueBrainCerebellumContractSignal::Insufficient,
            BlueBrainCerebellumContractSignal::Insufficient,
            BlueBrainCerebellumContractSignal::Insufficient,
            BlueBrainCerebellumContractSignal::Insufficient,
        )
    } else if matches!(
        input.deferral_class,
        BlueBrainCandidateDeferralLifecycleClass::CandidateDeferred
            | BlueBrainCandidateDeferralLifecycleClass::CandidateDeferredPendingStrongerEvidence
            | BlueBrainCandidateDeferralLifecycleClass::CandidateDeferredPendingContextUpdate
            | BlueBrainCandidateDeferralLifecycleClass::CandidateStale
            | BlueBrainCandidateDeferralLifecycleClass::CandidateNotPersisted
    ) || input.reference_validity == BlueBrainReferenceValidity::Stale
    {
        (
            BlueBrainCerebellumStateSurface::DeferredCorrectionState,
            BlueBrainCerebellumAdvisoryOutputClass::BlockedDeferred,
            BlueBrainCerebellumContractSignal::Deferred,
            BlueBrainCerebellumContractSignal::Deferred,
            BlueBrainCerebellumContractSignal::Deferred,
            BlueBrainCerebellumContractSignal::Deferred,
        )
    } else if input.reference_validity == BlueBrainReferenceValidity::ReferenceOnly {
        (
            BlueBrainCerebellumStateSurface::ReferenceOnlyCorrectionState,
            BlueBrainCerebellumAdvisoryOutputClass::ReferenceBoundedSignal,
            BlueBrainCerebellumContractSignal::ReferenceOnly,
            BlueBrainCerebellumContractSignal::ReferenceOnly,
            BlueBrainCerebellumContractSignal::ReferenceOnly,
            BlueBrainCerebellumContractSignal::ReferenceOnly,
        )
    } else if input.reference_validity == BlueBrainReferenceValidity::Caveated
        || input.context_priority
            == BlueBrainContextEvidencePriorityClass::CaveatedEvidenceReference
    {
        (
            BlueBrainCerebellumStateSurface::ExecutionSupportCaveatState,
            BlueBrainCerebellumAdvisoryOutputClass::ExecutionSupportCaveat,
            BlueBrainCerebellumContractSignal::Caveated,
            BlueBrainCerebellumContractSignal::Caveated,
            BlueBrainCerebellumContractSignal::Caveated,
            BlueBrainCerebellumContractSignal::Caveated,
        )
    } else if matches!(
        input.selection_signal,
        BlueBrainControlAttentionSelectionClass::AttentionTarget
            | BlueBrainControlAttentionSelectionClass::ContextSelection
    ) {
        (
            BlueBrainCerebellumStateSurface::ActivePredictionTimingAdvisoryOnly,
            BlueBrainCerebellumAdvisoryOutputClass::TimingHint,
            BlueBrainCerebellumContractSignal::CerebellumToRuntimeAdvisory,
            BlueBrainCerebellumContractSignal::CerebellumToSelectionAdvisory,
            BlueBrainCerebellumContractSignal::RuntimeToCerebellumBoundedPredictionTimingInput,
            BlueBrainCerebellumContractSignal::CerebellumReferenceSignal,
        )
    } else if input.selection_signal
        == BlueBrainControlAttentionSelectionClass::EvidenceReferenceSelection
    {
        (
            BlueBrainCerebellumStateSurface::CorrectionMismatchAdvisoryState,
            BlueBrainCerebellumAdvisoryOutputClass::MismatchHint,
            BlueBrainCerebellumContractSignal::CerebellumToRuntimeAdvisory,
            BlueBrainCerebellumContractSignal::CerebellumToSelectionAdvisory,
            BlueBrainCerebellumContractSignal::RuntimeToCerebellumBoundedPredictionTimingInput,
            BlueBrainCerebellumContractSignal::CerebellumReferenceSignal,
        )
    } else {
        (
            BlueBrainCerebellumStateSurface::TimingCoordinationAdvisoryState,
            BlueBrainCerebellumAdvisoryOutputClass::CorrectionHint,
            BlueBrainCerebellumContractSignal::CerebellumToRuntimeAdvisory,
            BlueBrainCerebellumContractSignal::CerebellumToSelectionAdvisory,
            BlueBrainCerebellumContractSignal::SelectionToCerebellumBoundedCoordinationInput,
            BlueBrainCerebellumContractSignal::CerebellumReferenceSignal,
        )
    };

    let output = BlueBrainCerebellumOutputSurface {
        advisory_class,
        runtime_contract_signal: runtime_signal,
        selection_contract_signal: selection_signal,
        execution_contract_signal: execution_signal,
        reference_contract_signal: reference_signal,
        runtime_diagnostic_state: blue_brain_cerebellum_diagnostic_state_for_signal(runtime_signal),
        selection_diagnostic_state: blue_brain_cerebellum_diagnostic_state_for_signal(
            selection_signal,
        ),
        execution_diagnostic_state: blue_brain_cerebellum_diagnostic_state_for_signal(
            execution_signal,
        ),
        reference_diagnostic_state: blue_brain_cerebellum_diagnostic_state_for_signal(
            reference_signal,
        ),
        canonical_contract_read: blue_brain_cerebellum_canonical_read_for_state(state),
        runtime_advisory_only: true,
        selection_advisory_only: true,
        execution_support_caveat_only: true,
        reference_bounded_only: true,
        direct_action_selection: false,
        direct_action_trigger: false,
        direct_execution_trigger: false,
        direct_retry_trigger: false,
        direct_memory_commit: false,
        direct_compute_invocation: false,
        safety_override: false,
    };
    (state, output)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainThirdRegionStateSurface {
    ActiveBoundedFeedbackAdvisoryOnly,
    CaveatedFeedbackState,
    DeferredFeedbackState,
    BlockedFeedbackState,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainThirdRegionInputSource {
    RuntimeFeedbackSignal,
    RuntimeDeferralSignal,
    SelectionCaveatSignal,
    ReferenceValiditySignal,
    ToolActionControlSignal,
    ComputeInternalStateSignal,
    SafetyOverrideSignal,
    ImplicitMemoryMutationSignal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainThirdRegionInputGuard {
    Canonical,
    RejectedToolActionControl,
    RejectedComputeInternalState,
    RejectedSafetyOverride,
    RejectedImplicitMemoryMutation,
}

pub fn classify_blue_brain_third_region_input_guard(
    source: BlueBrainThirdRegionInputSource,
) -> BlueBrainThirdRegionInputGuard {
    match source {
        BlueBrainThirdRegionInputSource::RuntimeFeedbackSignal
        | BlueBrainThirdRegionInputSource::RuntimeDeferralSignal
        | BlueBrainThirdRegionInputSource::SelectionCaveatSignal
        | BlueBrainThirdRegionInputSource::ReferenceValiditySignal => {
            BlueBrainThirdRegionInputGuard::Canonical
        }
        BlueBrainThirdRegionInputSource::ToolActionControlSignal => {
            BlueBrainThirdRegionInputGuard::RejectedToolActionControl
        }
        BlueBrainThirdRegionInputSource::ComputeInternalStateSignal => {
            BlueBrainThirdRegionInputGuard::RejectedComputeInternalState
        }
        BlueBrainThirdRegionInputSource::SafetyOverrideSignal => {
            BlueBrainThirdRegionInputGuard::RejectedSafetyOverride
        }
        BlueBrainThirdRegionInputSource::ImplicitMemoryMutationSignal => {
            BlueBrainThirdRegionInputGuard::RejectedImplicitMemoryMutation
        }
    }
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
    fn third_region_selection_map_contains_required_classes() {
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_SELECTION_MAP
            .contains(&BlueBrainThirdRegionSelectionClass::ThirdExpansionCandidate));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_SELECTION_MAP
            .contains(&BlueBrainThirdRegionSelectionClass::ViableButNotThird));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_SELECTION_MAP
            .contains(&BlueBrainThirdRegionSelectionClass::LaterPhaseCandidate));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_SELECTION_MAP
            .contains(&BlueBrainThirdRegionSelectionClass::SimulationOnlyDeferredCandidate));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_SELECTION_MAP
            .contains(&BlueBrainThirdRegionSelectionClass::NonCanonicalInternalOnlyPath));
    }

    #[test]
    fn third_region_class_selection_is_runtime_feedback_integration_related() {
        assert_eq!(
            BLUE_BRAIN_THIRD_REGION_CLASS_SELECTION,
            BlueBrainThirdRegionClass::RuntimeFeedbackIntegrationRelated
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
    fn second_region_hardening_map_contains_guarded_and_exclusion_paths() {
        assert!(CANONICAL_BLUE_BRAIN_SECOND_REGION_HARDENING_MAP
            .contains(&BlueBrainSecondRegionHardeningClass::GuardedCanonicalRegionSurface));
        assert!(CANONICAL_BLUE_BRAIN_SECOND_REGION_HARDENING_MAP
            .contains(&BlueBrainSecondRegionHardeningClass::GuardedRegionDiagnosticsPath));
        assert!(CANONICAL_BLUE_BRAIN_SECOND_REGION_HARDENING_MAP
            .contains(&BlueBrainSecondRegionHardeningClass::GuardedBoundedInterRegionRelationPath));
        assert!(CANONICAL_BLUE_BRAIN_SECOND_REGION_HARDENING_MAP
            .contains(&BlueBrainSecondRegionHardeningClass::BlockedForbiddenAuthorityPath));
        assert!(CANONICAL_BLUE_BRAIN_SECOND_REGION_HARDENING_MAP
            .contains(&BlueBrainSecondRegionHardeningClass::NonCanonicalInternalOnlyRegionPath));
        assert!(CANONICAL_BLUE_BRAIN_SECOND_REGION_HARDENING_MAP
            .contains(&BlueBrainSecondRegionHardeningClass::TestOnlyHelperNonOperationalPath));
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
        assert_eq!(
            output.runtime_contract_signal,
            output.selection_contract_signal
        );
    }

    #[test]
    fn second_region_no_direct_authority_flags_stay_false_for_all_diagnostics_paths() {
        let inputs = [
            BlueBrainSecondRegionInputSurface {
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority: BlueBrainContextEvidencePriorityClass::SupportingContext,
            },
            BlueBrainSecondRegionInputSurface {
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::Caveated,
                context_priority: BlueBrainContextEvidencePriorityClass::SupportingContext,
            },
            BlueBrainSecondRegionInputSurface {
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateDeferred,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority: BlueBrainContextEvidencePriorityClass::DeferredContext,
            },
            BlueBrainSecondRegionInputSurface {
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateRejected,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority: BlueBrainContextEvidencePriorityClass::SupportingContext,
            },
            BlueBrainSecondRegionInputSurface {
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::ReferenceOnly,
                context_priority: BlueBrainContextEvidencePriorityClass::SupportingContext,
            },
        ];

        for input in inputs {
            let (_, output) = evaluate_blue_brain_second_region_memory_context(input);
            assert!(!output.direct_action_selection);
            assert!(!output.direct_execution_trigger);
            assert!(!output.direct_retry_trigger);
            assert!(!output.direct_memory_commit);
            assert!(!output.direct_compute_invocation);
            assert!(!output.safety_override);
        }
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

    #[test]
    fn second_region_hardening_doc_pins_no_direct_and_non_canonical_cleanup() {
        let doc = include_str!(
            "../../../docs/blue_brain_second_region_tests_guards_cleanup_serie_bb26_prompt6_v1.md"
        );
        assert!(doc.contains("guarded canonical region-2 surface"));
        assert!(doc.contains("guarded region-2 diagnostics path"));
        assert!(doc.contains("guarded bounded inter-region relation path"));
        assert!(doc.contains("blocked forbidden authority path"));
        assert!(doc.contains("non-canonical/internal-only region-2 path"));
        assert!(doc.contains("test-only/helper path not operational"));
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
    fn third_region_selection_doc_pins_candidate_statuses_and_boundaries() {
        let doc = include_str!(
            "../../../docs/blue_brain_third_region_selection_serie_bb28_prompt1_v1.md"
        );
        assert!(doc.contains("Third-expansion candidate"));
        assert!(doc.contains("Runtime-feedback-integration-related"));
        assert!(doc.contains("Viable but not third"));
        assert!(doc.contains("Later-phase candidate"));
        assert!(doc.contains("Simulation-only/deferred candidate"));
        assert!(doc.contains("Non-canonical/internal-only path"));
        assert!(doc.contains("keine direkte Action-/Retry-/Memory-/Compute-Autorität"));
        assert!(doc.contains("keine Öffnung einer vierten Regionenklasse"));
        assert!(doc.contains("bounded advisory-only"));
    }

    #[test]
    fn third_region_integration_map_contains_required_classes() {
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_INTEGRATION_MAP
            .contains(&BlueBrainThirdRegionPathClass::Region3InputSurface));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_INTEGRATION_MAP
            .contains(&BlueBrainThirdRegionPathClass::Region3StateSurface));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_INTEGRATION_MAP
            .contains(&BlueBrainThirdRegionPathClass::Region3OutputAdvisorySurface));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_INTEGRATION_MAP
            .contains(&BlueBrainThirdRegionPathClass::Region3ReferenceSurface));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_INTEGRATION_MAP
            .contains(&BlueBrainThirdRegionPathClass::BlockedDeferredRegion3Path));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_INTEGRATION_MAP
            .contains(&BlueBrainThirdRegionPathClass::NonCanonicalInternalOnlyRegion3Path));
    }

    #[test]
    fn third_region_input_guards_preserve_no_direct_authority_inputs() {
        assert_eq!(
            classify_blue_brain_third_region_input_guard(
                BlueBrainThirdRegionInputSource::RuntimeFeedbackSignal
            ),
            BlueBrainThirdRegionInputGuard::Canonical
        );
        assert_eq!(
            classify_blue_brain_third_region_input_guard(
                BlueBrainThirdRegionInputSource::ToolActionControlSignal
            ),
            BlueBrainThirdRegionInputGuard::RejectedToolActionControl
        );
        assert_eq!(
            classify_blue_brain_third_region_input_guard(
                BlueBrainThirdRegionInputSource::ComputeInternalStateSignal
            ),
            BlueBrainThirdRegionInputGuard::RejectedComputeInternalState
        );
        assert_eq!(
            classify_blue_brain_third_region_input_guard(
                BlueBrainThirdRegionInputSource::SafetyOverrideSignal
            ),
            BlueBrainThirdRegionInputGuard::RejectedSafetyOverride
        );
        assert_eq!(
            classify_blue_brain_third_region_input_guard(
                BlueBrainThirdRegionInputSource::ImplicitMemoryMutationSignal
            ),
            BlueBrainThirdRegionInputGuard::RejectedImplicitMemoryMutation
        );
    }

    #[test]
    fn third_region_integration_doc_pins_surfaces_and_bounds() {
        let doc = include_str!(
            "../../../docs/blue_brain_third_region_integration_serie_bb28_prompt2_v1.md"
        );
        assert!(doc.contains("Canonical Third-Region Integration Map"));
        assert!(doc.contains("region-3 input surface"));
        assert!(doc.contains("region-3 state surface"));
        assert!(doc.contains("region-3 output/advisory surface"));
        assert!(doc.contains("region-3 reference surface"));
        assert!(doc.contains("blocked/deferred region-3 path"));
        assert!(doc.contains("non-canonical/internal-only region-3 path"));
        assert!(doc.contains("no direct action selection"));
        assert!(doc.contains("no direct execution trigger"));
        assert!(doc.contains("no direct retry trigger"));
        assert!(doc.contains("no direct memory commit"));
        assert!(doc.contains("no direct compute invocation"));
        assert!(doc.contains("no safety override"));
        assert!(doc.contains("keine Öffnung einer vierten Regionenklasse"));
    }

    #[test]
    fn third_region_contract_map_contains_required_classes() {
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_CONTRACT_MAP
            .contains(&BlueBrainThirdRegionContractClass::Region3ToRuntimeAdvisorySignal));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_CONTRACT_MAP
            .contains(&BlueBrainThirdRegionContractClass::RuntimeToRegion3BoundedInput));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_CONTRACT_MAP
            .contains(&BlueBrainThirdRegionContractClass::Region3ToSelectionAdvisorySignal));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_CONTRACT_MAP
            .contains(&BlueBrainThirdRegionContractClass::SelectionToRegion3BoundedStateInput));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_CONTRACT_MAP
            .contains(&BlueBrainThirdRegionContractClass::Region3ReferenceSignal));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_CONTRACT_MAP.contains(
            &BlueBrainThirdRegionContractClass::CaveatedDeferredBlockedRegion3ContractSignal
        ));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_CONTRACT_MAP
            .contains(&BlueBrainThirdRegionContractClass::ReferenceOnlyRegion3ContractSignal));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_CONTRACT_MAP.contains(
            &BlueBrainThirdRegionContractClass::NonCanonicalInternalOnlyRegion3ContractPath
        ));
    }

    #[test]
    fn third_region_diagnostic_map_contains_required_states() {
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_DIAGNOSTIC_MAP
            .contains(&BlueBrainThirdRegionDiagnosticState::Region3AdvisoryOnlyDiagnostic));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_DIAGNOSTIC_MAP
            .contains(&BlueBrainThirdRegionDiagnosticState::Region3CaveatedDiagnostic));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_DIAGNOSTIC_MAP
            .contains(&BlueBrainThirdRegionDiagnosticState::Region3DeferredDiagnostic));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_DIAGNOSTIC_MAP
            .contains(&BlueBrainThirdRegionDiagnosticState::Region3BlockedDiagnostic));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_DIAGNOSTIC_MAP
            .contains(&BlueBrainThirdRegionDiagnosticState::Region3InsufficientDiagnostic));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_DIAGNOSTIC_MAP
            .contains(&BlueBrainThirdRegionDiagnosticState::Region3DiagnosticOnlyState));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_DIAGNOSTIC_MAP.contains(
            &BlueBrainThirdRegionDiagnosticState::CaveatedInterRegionDiagnosticInfluence
        ));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_DIAGNOSTIC_MAP.contains(
            &BlueBrainThirdRegionDiagnosticState::NonCanonicalInternalOnlyRegion3DiagnosticPath
        ));
    }

    #[test]
    fn third_region_signal_to_diagnostic_state_distinguishes_core_states() {
        assert_eq!(
            blue_brain_third_region_diagnostic_state_for_signal(
                BlueBrainThirdRegionContractSignal::Region3ToRuntimeAdvisory
            ),
            BlueBrainThirdRegionDiagnosticState::Region3AdvisoryOnlyDiagnostic
        );
        assert_eq!(
            blue_brain_third_region_diagnostic_state_for_signal(
                BlueBrainThirdRegionContractSignal::Caveated
            ),
            BlueBrainThirdRegionDiagnosticState::Region3CaveatedDiagnostic
        );
        assert_eq!(
            blue_brain_third_region_diagnostic_state_for_signal(
                BlueBrainThirdRegionContractSignal::Deferred
            ),
            BlueBrainThirdRegionDiagnosticState::Region3DeferredDiagnostic
        );
        assert_eq!(
            blue_brain_third_region_diagnostic_state_for_signal(
                BlueBrainThirdRegionContractSignal::Blocked
            ),
            BlueBrainThirdRegionDiagnosticState::Region3BlockedDiagnostic
        );
        assert_eq!(
            blue_brain_third_region_diagnostic_state_for_signal(
                BlueBrainThirdRegionContractSignal::Insufficient
            ),
            BlueBrainThirdRegionDiagnosticState::Region3InsufficientDiagnostic
        );
        assert_eq!(
            blue_brain_third_region_diagnostic_state_for_signal(
                BlueBrainThirdRegionContractSignal::ReferenceOnly
            ),
            BlueBrainThirdRegionDiagnosticState::Region3DiagnosticOnlyState
        );
    }

    #[test]
    fn third_region_contract_doc_pins_runtime_selection_reference_semantics() {
        let doc = include_str!(
            "../../../docs/blue_brain_third_region_runtime_selection_reference_contract_serie_bb28_prompt3_v1.md"
        );
        assert!(doc.contains("third-region contract map"));
        assert!(doc.contains("region-3-to-runtime advisory signal"));
        assert!(doc.contains("runtime-to-region-3 bounded input"));
        assert!(doc.contains("region-3-to-selection advisory signal"));
        assert!(doc.contains("selection-to-region-3 bounded state input"));
        assert!(doc.contains("region-3-reference signal"));
        assert!(doc.contains("caveated/deferred/blocked region-3 contract signal"));
        assert!(doc.contains("reference-only region-3 contract signal"));
        assert!(doc.contains("non-canonical/internal-only region-3 contract path"));
        assert!(doc.contains("deferred != blocked"));
        assert!(doc.contains("blocked != failed execution"));
        assert!(doc.contains("caveated != strong region-3 signal"));
        assert!(doc.contains("reference-only != operative support basis"));
        assert!(doc.contains("no direct action trigger"));
        assert!(doc.contains("no direct execution trigger"));
        assert!(doc.contains("no direct retry trigger"));
        assert!(doc.contains("no direct memory commit"));
        assert!(doc.contains("no direct compute invocation"));
        assert!(doc.contains("no safety override"));
        assert!(doc.contains("no fourth-region opening"));
        assert!(doc.contains("no broad inter-region platform"));
    }

    #[test]
    fn third_region_relation_map_contains_required_classes() {
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_RELATION_MAP
            .contains(&BlueBrainThirdRegionRelationClass::Region3ToRegion1Bounded));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_RELATION_MAP
            .contains(&BlueBrainThirdRegionRelationClass::Region1ToRegion3Bounded));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_RELATION_MAP
            .contains(&BlueBrainThirdRegionRelationClass::Region3ToRegion2Bounded));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_RELATION_MAP
            .contains(&BlueBrainThirdRegionRelationClass::Region2ToRegion3Bounded));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_RELATION_MAP
            .contains(&BlueBrainThirdRegionRelationClass::SharedReferenceMediatedRelation));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_RELATION_MAP
            .contains(&BlueBrainThirdRegionRelationClass::CaveatedInterRegionRelation));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_RELATION_MAP
            .contains(&BlueBrainThirdRegionRelationClass::BlockedDeferredInterRegionRelation));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_RELATION_MAP
            .contains(&BlueBrainThirdRegionRelationClass::NonCanonicalInternalOnlyInterRegionPath));
    }

    #[test]
    fn third_region_relation_stays_bounded_advisory_only_without_direct_authority() {
        let (_, region1) = evaluate_blue_brain_first_region_attention_selection(
            BlueBrainFirstRegionInputSurface {
                attention_class: BlueBrainControlAttentionSelectionClass::AttentionTarget,
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
            },
        );
        let (_, region2) =
            evaluate_blue_brain_second_region_memory_context(BlueBrainSecondRegionInputSurface {
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority: BlueBrainContextEvidencePriorityClass::SupportingContext,
            });

        let relation = evaluate_blue_brain_third_region_relation(region1, region2);
        assert!(relation.region3_to_region1_advisory_only);
        assert!(relation.region1_to_region3_advisory_only);
        assert!(relation.region3_to_region2_advisory_only);
        assert!(relation.region2_to_region3_advisory_only);
        assert!(!relation.direct_action_selection);
        assert!(!relation.direct_execution_trigger);
        assert!(!relation.direct_retry_trigger);
        assert!(!relation.direct_memory_commit);
        assert!(!relation.direct_compute_invocation);
        assert!(!relation.safety_override);
    }

    #[test]
    fn third_region_relation_doc_pins_direction_semantics_and_boundaries() {
        let doc = include_str!(
            "../../../docs/blue_brain_third_region_relation_line_serie_bb28_prompt4_v1.md"
        );
        assert!(doc.contains("region-3-to-region-1 bounded relation"));
        assert!(doc.contains("region-1-to-region-3 bounded relation"));
        assert!(doc.contains("region-3-to-region-2 bounded relation"));
        assert!(doc.contains("region-2-to-region-3 bounded relation"));
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
        assert!(doc.contains("no broad inter-region platform"));
    }

    #[test]
    fn third_region_hardening_map_contains_required_classes() {
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_HARDENING_MAP
            .contains(&BlueBrainThirdRegionHardeningClass::GuardedCanonicalRegion3Surface));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_HARDENING_MAP
            .contains(&BlueBrainThirdRegionHardeningClass::GuardedRegion3DiagnosticsPath));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_HARDENING_MAP
            .contains(&BlueBrainThirdRegionHardeningClass::GuardedBoundedInterRegionRelationPath));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_HARDENING_MAP
            .contains(&BlueBrainThirdRegionHardeningClass::BlockedForbiddenAuthorityPath));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_HARDENING_MAP
            .contains(&BlueBrainThirdRegionHardeningClass::NonCanonicalInternalOnlyRegion3Path));
        assert!(CANONICAL_BLUE_BRAIN_THIRD_REGION_HARDENING_MAP
            .contains(&BlueBrainThirdRegionHardeningClass::TestOnlyHelperNonOperationalPath));
    }

    #[test]
    fn third_region_hardening_doc_pins_no_direct_guard_lines_and_cleanup() {
        let doc = include_str!(
            "../../../docs/blue_brain_third_region_tests_guards_cleanup_serie_bb28_prompt6_v1.md"
        );
        assert!(doc.contains("guarded canonical region-3 surface"));
        assert!(doc.contains("guarded region-3 diagnostics path"));
        assert!(doc.contains("guarded bounded inter-region relation path"));
        assert!(doc.contains("blocked forbidden authority path"));
        assert!(doc.contains("non-canonical/internal-only region-3 path"));
        assert!(doc.contains("test-only/helper path not operational"));
        assert!(doc.contains("no direct action trigger"));
        assert!(doc.contains("no direct execution trigger"));
        assert!(doc.contains("no direct retry trigger"));
        assert!(doc.contains("no direct memory commit"));
        assert!(doc.contains("no direct compute invocation"));
        assert!(doc.contains("no safety override"));
        assert!(doc.contains("no fourth-region opening"));
        assert!(doc.contains("no broad inter-region platform"));
    }

    #[test]
    fn two_region_maintenance_reference_cleanup_doc_pins_canonical_categories() {
        let doc = include_str!(
            "../../../docs/blue_brain_two_region_docs_tests_reference_cleanup_serie_bb27_prompt2_v1.md"
        );
        assert!(doc.contains("canonical two-region reference doc"));
        assert!(doc.contains("canonical region-1 test surface"));
        assert!(doc.contains("canonical region-2 test surface"));
        assert!(doc.contains("canonical bounded relation test surface"));
        assert!(doc.contains("maintenance-facing index/reference path"));
        assert!(doc.contains("non-canonical/internal-only or legacy two-region path"));
        assert!(doc.contains("Region 3 is **not open**"));
    }

    #[test]
    fn docs_indexes_expose_bb27_two_region_maintenance_reference_line() {
        let readme = include_str!("../../../docs/README.md");
        assert!(readme.contains("Two-region maintenance stabilization/reference line (BB27)"));
        assert!(readme.contains(
            "docs/blue_brain_two_region_docs_tests_reference_cleanup_serie_bb27_prompt2_v1.md"
        ));

        let repo_map = include_str!("../../../docs/roadmap/REPO_MAP.md");
        assert!(repo_map.contains("Two-region maintenance stabilization/reference line (BB27)"));
        assert!(repo_map.contains(
            "docs/blue_brain_two_region_docs_tests_reference_cleanup_serie_bb27_prompt2_v1.md"
        ));
    }

    #[test]
    fn three_region_docs_tests_cleanup_doc_pins_canonical_categories_and_region4_boundary() {
        let doc = include_str!(
            "../../../docs/blue_brain_three_region_docs_tests_index_cleanup_serie_bb29_prompt2_v1.md"
        );
        assert!(doc.contains("canonical three-region reference doc"));
        assert!(doc.contains("canonical region-1 test surface"));
        assert!(doc.contains("canonical region-2 test surface"));
        assert!(doc.contains("canonical region-3 test surface"));
        assert!(doc.contains("canonical bounded relation test surfaces"));
        assert!(doc.contains("maintenance-facing index/reference path"));
        assert!(doc.contains("non-canonical/internal-only or legacy three-region path"));
        assert!(doc.contains("no-direct-action"));
        assert!(doc.contains("no-direct-execution"));
        assert!(doc.contains("no-direct-retry"));
        assert!(doc.contains("no-direct-memory"));
        assert!(doc.contains("no-direct-compute"));
        assert!(doc.contains("no direct policy decision"));
        assert!(doc.contains("no safety override"));
        assert!(doc.contains("Region 4 ist **nicht offen**"));
    }

    #[test]
    fn docs_indexes_expose_bb29_three_region_maintenance_reference_line() {
        let readme = include_str!("../../../docs/README.md");
        assert!(readme.contains("Three-region maintenance stabilization/reference line (BB29)"));
        assert!(readme.contains(
            "docs/blue_brain_three_region_docs_tests_index_cleanup_serie_bb29_prompt2_v1.md"
        ));

        let repo_map = include_str!("../../../docs/roadmap/REPO_MAP.md");
        assert!(repo_map.contains("Three-region maintenance stabilization/reference line (BB29)"));
        assert!(repo_map.contains(
            "docs/blue_brain_three_region_docs_tests_index_cleanup_serie_bb29_prompt2_v1.md"
        ));
    }

    #[test]
    fn two_region_consistency_map_contains_required_classes() {
        assert!(CANONICAL_BLUE_BRAIN_TWO_REGION_CONSISTENCY_MAP
            .contains(&BlueBrainTwoRegionConsistencyClass::CanonicalRegion1Path));
        assert!(CANONICAL_BLUE_BRAIN_TWO_REGION_CONSISTENCY_MAP
            .contains(&BlueBrainTwoRegionConsistencyClass::CanonicalRegion2Path));
        assert!(CANONICAL_BLUE_BRAIN_TWO_REGION_CONSISTENCY_MAP
            .contains(&BlueBrainTwoRegionConsistencyClass::BoundedInterRegionRelationPath));
        assert!(CANONICAL_BLUE_BRAIN_TWO_REGION_CONSISTENCY_MAP
            .contains(&BlueBrainTwoRegionConsistencyClass::CaveatedTwoRegionPath));
        assert!(CANONICAL_BLUE_BRAIN_TWO_REGION_CONSISTENCY_MAP
            .contains(&BlueBrainTwoRegionConsistencyClass::BlockedInsufficientTwoRegionPath));
        assert!(CANONICAL_BLUE_BRAIN_TWO_REGION_CONSISTENCY_MAP
            .contains(&BlueBrainTwoRegionConsistencyClass::NonCanonicalInternalOnlyTwoRegionPath));
    }

    #[test]
    fn three_region_consistency_map_contains_required_classes() {
        assert!(CANONICAL_BLUE_BRAIN_THREE_REGION_CONSISTENCY_MAP
            .contains(&BlueBrainThreeRegionConsistencyClass::CanonicalRegion1Path));
        assert!(CANONICAL_BLUE_BRAIN_THREE_REGION_CONSISTENCY_MAP
            .contains(&BlueBrainThreeRegionConsistencyClass::CanonicalRegion2Path));
        assert!(CANONICAL_BLUE_BRAIN_THREE_REGION_CONSISTENCY_MAP
            .contains(&BlueBrainThreeRegionConsistencyClass::CanonicalRegion3Path));
        assert!(CANONICAL_BLUE_BRAIN_THREE_REGION_CONSISTENCY_MAP
            .contains(&BlueBrainThreeRegionConsistencyClass::BoundedInterRegionRelationPath));
        assert!(CANONICAL_BLUE_BRAIN_THREE_REGION_CONSISTENCY_MAP
            .contains(&BlueBrainThreeRegionConsistencyClass::CaveatedThreeRegionPath));
        assert!(CANONICAL_BLUE_BRAIN_THREE_REGION_CONSISTENCY_MAP
            .contains(&BlueBrainThreeRegionConsistencyClass::BlockedInsufficientThreeRegionPath));
        assert!(CANONICAL_BLUE_BRAIN_THREE_REGION_CONSISTENCY_MAP.contains(
            &BlueBrainThreeRegionConsistencyClass::NonCanonicalInternalOnlyThreeRegionPath
        ));
    }

    #[test]
    fn three_region_consistency_uses_shared_guard_contract_boundaries() {
        let (_, region1) = evaluate_blue_brain_first_region_attention_selection(
            BlueBrainFirstRegionInputSurface {
                attention_class: BlueBrainControlAttentionSelectionClass::AttentionTarget,
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
            },
        );
        let (_, region2) =
            evaluate_blue_brain_second_region_memory_context(BlueBrainSecondRegionInputSurface {
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
            });

        let relation12 = evaluate_blue_brain_inter_region_relation(region1, region2);
        let relation3 = evaluate_blue_brain_third_region_relation(region1, region2);
        assert_eq!(
            classify_blue_brain_three_region_consistency(region1, region2, relation12, relation3),
            BlueBrainThreeRegionConsistencyClass::BoundedInterRegionRelationPath
        );
        assert!(!relation3.direct_action_selection);
        assert!(!relation3.direct_execution_trigger);
        assert!(!relation3.direct_retry_trigger);
        assert!(!relation3.direct_memory_commit);
        assert!(!relation3.direct_compute_invocation);
        assert!(!relation3.safety_override);
    }

    #[test]
    fn three_region_consistency_doc_pins_canonical_paths_and_no_direct_guards() {
        let doc = include_str!(
            "../../../docs/blue_brain_three_region_guard_contract_consistency_serie_bb28_prompt7_v1.md"
        );
        assert!(doc.contains("consistent canonical region-1 path"));
        assert!(doc.contains("consistent canonical region-2 path"));
        assert!(doc.contains("consistent canonical region-3 path"));
        assert!(doc.contains("consistent bounded inter-region relation path"));
        assert!(doc.contains("caveated three-region path"));
        assert!(doc.contains("blocked/insufficient three-region path"));
        assert!(doc.contains("non-canonical/internal-only three-region path"));
        assert!(doc.contains("no direct action trigger"));
        assert!(doc.contains("no direct execution trigger"));
        assert!(doc.contains("no direct retry trigger"));
        assert!(doc.contains("no direct memory effect"));
        assert!(doc.contains("no direct compute effect"));
        assert!(doc.contains("no safety override"));
        assert!(doc.contains("no fourth-region opening"));
        assert!(doc.contains("no broad inter-region platform"));
    }

    #[test]
    fn two_region_maintenance_findings_map_contains_required_classes() {
        assert!(CANONICAL_BLUE_BRAIN_TWO_REGION_MAINTENANCE_FINDINGS_MAP
            .contains(&BlueBrainTwoRegionMaintenanceFindingClass::RealBug));
        assert!(CANONICAL_BLUE_BRAIN_TWO_REGION_MAINTENANCE_FINDINGS_MAP
            .contains(&BlueBrainTwoRegionMaintenanceFindingClass::SemanticInconsistency));
        assert!(CANONICAL_BLUE_BRAIN_TWO_REGION_MAINTENANCE_FINDINGS_MAP
            .contains(&BlueBrainTwoRegionMaintenanceFindingClass::GuardWeakness));
        assert!(CANONICAL_BLUE_BRAIN_TWO_REGION_MAINTENANCE_FINDINGS_MAP
            .contains(&BlueBrainTwoRegionMaintenanceFindingClass::DocTestDrift));
        assert!(CANONICAL_BLUE_BRAIN_TWO_REGION_MAINTENANCE_FINDINGS_MAP
            .contains(&BlueBrainTwoRegionMaintenanceFindingClass::NonCanonicalResidualPath));
        assert!(CANONICAL_BLUE_BRAIN_TWO_REGION_MAINTENANCE_FINDINGS_MAP
            .contains(&BlueBrainTwoRegionMaintenanceFindingClass::NoChangeNeededFinding));
    }

    #[test]
    fn two_region_relation_and_consistency_preserve_no_direct_authority() {
        let (_, region1) = evaluate_blue_brain_first_region_attention_selection(
            BlueBrainFirstRegionInputSurface {
                attention_class: BlueBrainControlAttentionSelectionClass::AttentionTarget,
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
            },
        );
        let (_, region2) =
            evaluate_blue_brain_second_region_memory_context(BlueBrainSecondRegionInputSurface {
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
            });

        let relation = evaluate_blue_brain_inter_region_relation(region1, region2);
        assert!(!relation.direct_action_selection);
        assert!(!relation.direct_execution_trigger);
        assert!(!relation.direct_retry_trigger);
        assert!(!relation.direct_memory_commit);
        assert!(!relation.direct_compute_invocation);
        assert!(!relation.safety_override);
        assert!(matches!(
            classify_blue_brain_two_region_consistency(region1, region2, relation),
            BlueBrainTwoRegionConsistencyClass::BoundedInterRegionRelationPath
        ));
    }

    #[test]
    fn two_region_consistency_keeps_caveated_distinct_from_blocked_insufficient() {
        let (_, region1) = evaluate_blue_brain_first_region_attention_selection(
            BlueBrainFirstRegionInputSurface {
                attention_class: BlueBrainControlAttentionSelectionClass::AttentionTarget,
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::Caveated,
                context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
            },
        );
        let (_, region2) =
            evaluate_blue_brain_second_region_memory_context(BlueBrainSecondRegionInputSurface {
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
            });
        let relation = evaluate_blue_brain_inter_region_relation(region1, region2);
        assert!(relation.caveated);
        assert!(!relation.blocked);
        assert!(!relation.deferred);
        assert!(matches!(
            classify_blue_brain_two_region_consistency(region1, region2, relation),
            BlueBrainTwoRegionConsistencyClass::CaveatedTwoRegionPath
        ));
    }

    #[test]
    fn first_anatomical_region_selection_is_hippocampus_like_region() {
        assert_eq!(
            BLUE_BRAIN_FIRST_ANATOMICAL_REGION_SELECTION,
            BlueBrainFirstAnatomicalRegion::HippocampusLikeRegion
        );
    }

    #[test]
    fn first_anatomical_region_integration_map_contains_required_classes() {
        assert!(CANONICAL_BLUE_BRAIN_FIRST_ANATOMICAL_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstAnatomicalRegionPathClass::AnatomicalRegionInputSurface));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_ANATOMICAL_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstAnatomicalRegionPathClass::AnatomicalRegionStateSurface));
        assert!(
            CANONICAL_BLUE_BRAIN_FIRST_ANATOMICAL_REGION_INTEGRATION_MAP.contains(
                &BlueBrainFirstAnatomicalRegionPathClass::AnatomicalRegionOutputAdvisorySurface
            )
        );
        assert!(CANONICAL_BLUE_BRAIN_FIRST_ANATOMICAL_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstAnatomicalRegionPathClass::AnatomicalRegionReferenceSurface));
        assert!(
            CANONICAL_BLUE_BRAIN_FIRST_ANATOMICAL_REGION_INTEGRATION_MAP.contains(
                &BlueBrainFirstAnatomicalRegionPathClass::AnatomicalToFunctionalRegionMapping
            )
        );
        assert!(
            CANONICAL_BLUE_BRAIN_FIRST_ANATOMICAL_REGION_INTEGRATION_MAP.contains(
                &BlueBrainFirstAnatomicalRegionPathClass::BlockedDeferredAnatomicalRegionPath
            )
        );
        assert!(CANONICAL_BLUE_BRAIN_FIRST_ANATOMICAL_REGION_INTEGRATION_MAP.contains(
            &BlueBrainFirstAnatomicalRegionPathClass::NonCanonicalInternalOnlyAnatomicalRegionPath
        ));
    }

    #[test]
    fn first_anatomical_region_input_output_guards_preserve_advisory_only_boundaries() {
        assert_eq!(
            classify_blue_brain_first_anatomical_region_input_guard(
                BlueBrainFirstAnatomicalRegionInputClass::RuntimeSelectionContextSignal
            ),
            BlueBrainFirstAnatomicalRegionInputGuard::AllowedBoundedInput
        );
        assert_eq!(
            classify_blue_brain_first_anatomical_region_input_guard(
                BlueBrainFirstAnatomicalRegionInputClass::ToolActionControlSignal
            ),
            BlueBrainFirstAnatomicalRegionInputGuard::BlockedForbiddenInput
        );
        assert_eq!(
            classify_blue_brain_first_anatomical_region_output_guard(
                BlueBrainFirstAnatomicalRegionOutputClass::AdvisoryReferenceBoundedSignal
            ),
            BlueBrainFirstAnatomicalRegionOutputGuard::AllowedAdvisoryOutput
        );
        assert_eq!(
            classify_blue_brain_first_anatomical_region_output_guard(
                BlueBrainFirstAnatomicalRegionOutputClass::DirectExecutionTrigger
            ),
            BlueBrainFirstAnatomicalRegionOutputGuard::BlockedForbiddenOutput
        );
    }

    #[test]
    fn first_anatomical_region_integration_doc_pins_minimal_surfaces_and_boundaries() {
        let doc = include_str!(
            "../../../docs/blue_brain_first_anatomical_region_integration_serie_bb30_prompt3_v1.md"
        );

        assert!(doc.contains("hippocampus_like_region"));
        assert!(doc.contains("anatomical region input surface"));
        assert!(doc.contains("anatomical region state surface"));
        assert!(doc.contains("anatomical region output/advisory surface"));
        assert!(doc.contains("anatomical region reference surface"));
        assert!(doc.contains("direct action selection"));
        assert!(doc.contains("direct execution trigger"));
        assert!(doc.contains("direct retry trigger"));
        assert!(doc.contains("direct memory commit"));
        assert!(doc.contains("direct compute invocation"));
        assert!(doc.contains("safety override"));
        assert!(doc.contains("no fourth-region opening"));
    }

    #[test]
    fn first_anatomical_region_model_decision_map_contains_required_classes() {
        assert!(
            CANONICAL_BLUE_BRAIN_FIRST_ANATOMICAL_REGION_MODEL_DECISION_MAP.contains(
                &BlueBrainFirstAnatomicalRegionModelModeClass::AbstractFunctionalCurrentMode
            )
        );
        assert!(
            CANONICAL_BLUE_BRAIN_FIRST_ANATOMICAL_REGION_MODEL_DECISION_MAP.contains(
                &BlueBrainFirstAnatomicalRegionModelModeClass::BoundedKuramotoLikeCurrentMode
            )
        );
        assert!(CANONICAL_BLUE_BRAIN_FIRST_ANATOMICAL_REGION_MODEL_DECISION_MAP.contains(
            &BlueBrainFirstAnatomicalRegionModelModeClass::HodgkinHuxleySimulationOnlyDiagnosticOnlyCurrentMode
        ));
        assert!(
            CANONICAL_BLUE_BRAIN_FIRST_ANATOMICAL_REGION_MODEL_DECISION_MAP.contains(
                &BlueBrainFirstAnatomicalRegionModelModeClass::LaterSelectiveHodgkinHuxleyDeepening
            )
        );
        assert!(
            CANONICAL_BLUE_BRAIN_FIRST_ANATOMICAL_REGION_MODEL_DECISION_MAP.contains(
                &BlueBrainFirstAnatomicalRegionModelModeClass::DeferredNotSuitableNowModelPath
            )
        );
        assert!(
            CANONICAL_BLUE_BRAIN_FIRST_ANATOMICAL_REGION_MODEL_DECISION_MAP.contains(
                &BlueBrainFirstAnatomicalRegionModelModeClass::NonCanonicalInternalOnlyModelPath
            )
        );
    }

    #[test]
    fn first_anatomical_region_current_model_mode_is_abstract_functional() {
        assert_eq!(
            BLUE_BRAIN_FIRST_ANATOMICAL_REGION_CURRENT_MODEL_MODE,
            BlueBrainFirstAnatomicalRegionModelModeClass::AbstractFunctionalCurrentMode
        );
    }

    #[test]
    fn first_anatomical_region_model_decision_doc_pins_current_mode_and_boundaries() {
        let doc = include_str!(
            "../../../docs/blue_brain_first_anatomical_region_model_decision_serie_bb30_prompt4_v1.md"
        );
        assert!(doc.contains("hippocampus_like_region"));
        assert!(doc.contains("abstract functional current mode"));
        assert!(doc.contains("bounded Kuramoto-like current mode"));
        assert!(doc.contains("Hodgkin-Huxley simulation-only/diagnostic-only current mode"));
        assert!(doc.contains("later selective HH deepening"));
        assert!(doc.contains("deferred/not-suitable-now model path"));
        assert!(doc.contains("non-canonical/internal-only model path"));
        assert!(doc.contains("no direct action execution"));
        assert!(doc.contains("no retry orchestration"));
        assert!(doc.contains("no automatic memory persistence"));
        assert!(doc.contains("no HH production integration"));
    }

    #[test]
    fn first_anatomical_region_diagnostic_map_contains_required_states() {
        assert!(CANONICAL_BLUE_BRAIN_FIRST_ANATOMICAL_REGION_DIAGNOSTIC_MAP.contains(
            &BlueBrainFirstAnatomicalRegionDiagnosticState::AnatomicalRegionAdvisoryOnlyDiagnostic
        ));
        assert!(
            CANONICAL_BLUE_BRAIN_FIRST_ANATOMICAL_REGION_DIAGNOSTIC_MAP.contains(
                &BlueBrainFirstAnatomicalRegionDiagnosticState::AnatomicalRegionCaveatedDiagnostic
            )
        );
        assert!(
            CANONICAL_BLUE_BRAIN_FIRST_ANATOMICAL_REGION_DIAGNOSTIC_MAP.contains(
                &BlueBrainFirstAnatomicalRegionDiagnosticState::AnatomicalRegionDeferredDiagnostic
            )
        );
        assert!(
            CANONICAL_BLUE_BRAIN_FIRST_ANATOMICAL_REGION_DIAGNOSTIC_MAP.contains(
                &BlueBrainFirstAnatomicalRegionDiagnosticState::AnatomicalRegionBlockedDiagnostic
            )
        );
        assert!(CANONICAL_BLUE_BRAIN_FIRST_ANATOMICAL_REGION_DIAGNOSTIC_MAP.contains(
            &BlueBrainFirstAnatomicalRegionDiagnosticState::AnatomicalRegionInsufficientDiagnostic
        ));
        assert!(
            CANONICAL_BLUE_BRAIN_FIRST_ANATOMICAL_REGION_DIAGNOSTIC_MAP.contains(
                &BlueBrainFirstAnatomicalRegionDiagnosticState::AnatomicalRegionDiagnosticOnlyState
            )
        );
        assert!(CANONICAL_BLUE_BRAIN_FIRST_ANATOMICAL_REGION_DIAGNOSTIC_MAP.contains(
            &BlueBrainFirstAnatomicalRegionDiagnosticState::NonCanonicalInternalOnlyAnatomicalRegionDiagnosticPath
        ));
    }

    #[test]
    fn first_anatomical_region_diagnostics_keep_contract_semantics_distinct() {
        assert_eq!(
            blue_brain_first_anatomical_region_diagnostic_state_for_signal(
                BlueBrainFirstAnatomicalRegionContractSignal::AnatomicalToRuntimeAdvisory
            ),
            BlueBrainFirstAnatomicalRegionDiagnosticState::AnatomicalRegionAdvisoryOnlyDiagnostic
        );
        assert_eq!(
            blue_brain_first_anatomical_region_diagnostic_state_for_signal(
                BlueBrainFirstAnatomicalRegionContractSignal::Caveated
            ),
            BlueBrainFirstAnatomicalRegionDiagnosticState::AnatomicalRegionCaveatedDiagnostic
        );
        assert_eq!(
            blue_brain_first_anatomical_region_diagnostic_state_for_signal(
                BlueBrainFirstAnatomicalRegionContractSignal::Deferred
            ),
            BlueBrainFirstAnatomicalRegionDiagnosticState::AnatomicalRegionDeferredDiagnostic
        );
        assert_eq!(
            blue_brain_first_anatomical_region_diagnostic_state_for_signal(
                BlueBrainFirstAnatomicalRegionContractSignal::Blocked
            ),
            BlueBrainFirstAnatomicalRegionDiagnosticState::AnatomicalRegionBlockedDiagnostic
        );
        assert_eq!(
            blue_brain_first_anatomical_region_diagnostic_state_for_signal(
                BlueBrainFirstAnatomicalRegionContractSignal::Insufficient
            ),
            BlueBrainFirstAnatomicalRegionDiagnosticState::AnatomicalRegionInsufficientDiagnostic
        );
        assert_eq!(
            blue_brain_first_anatomical_region_diagnostic_state_for_signal(
                BlueBrainFirstAnatomicalRegionContractSignal::DiagnosticOnly
            ),
            BlueBrainFirstAnatomicalRegionDiagnosticState::AnatomicalRegionDiagnosticOnlyState
        );
    }

    #[test]
    fn first_anatomical_region_runtime_selection_reference_reads_are_consistent() {
        let signal = BlueBrainFirstAnatomicalRegionContractSignal::ReferenceOnly;
        let runtime_read = blue_brain_first_anatomical_region_runtime_diagnostic_read(signal);
        let selection_read = blue_brain_first_anatomical_region_selection_diagnostic_read(signal);
        let reference_read = blue_brain_first_anatomical_region_reference_diagnostic_read(signal);
        assert_eq!(runtime_read, selection_read);
        assert_eq!(selection_read, reference_read);
        assert_eq!(
            runtime_read,
            BlueBrainFirstAnatomicalRegionDiagnosticState::AnatomicalRegionDiagnosticOnlyState
        );
    }

    #[test]
    fn first_anatomical_region_diagnostics_doc_pins_contract_and_no_direct_boundaries() {
        let doc = include_str!(
            "../../../docs/blue_brain_first_anatomical_region_diagnostics_contract_semantics_serie_bb30_prompt5_v1.md"
        );
        assert!(doc.contains("advisory-only diagnostic"));
        assert!(doc.contains("caveated diagnostic"));
        assert!(doc.contains("deferred diagnostic"));
        assert!(doc.contains("blocked diagnostic"));
        assert!(doc.contains("insufficient diagnostic"));
        assert!(doc.contains("diagnostic-only state"));
        assert!(doc.contains("non-canonical/internal-only anatomical region diagnostic path"));
        assert!(doc.contains("runtime/selection/reference"));
        assert!(doc.contains("no direct action trigger"));
        assert!(doc.contains("no direct execution trigger"));
        assert!(doc.contains("no direct retry trigger"));
        assert!(doc.contains("no direct memory commit"));
        assert!(doc.contains("no direct compute invocation"));
        assert!(doc.contains("no safety override"));
    }

    #[test]
    fn bb30_prompt6_readiness_doc_pins_first_anatomical_expansion_boundary() {
        let doc = include_str!(
            "../../../docs/blue_brain_bb30_readiness_sweep_first_anatomical_expansion_boundary_serie_bb30_prompt6_v1.md"
        );
        assert!(doc.contains("hippocampus_like_region"));
        assert!(doc.contains("stable first-anatomical operational surface"));
        assert!(doc.contains("usable with caveats"));
        assert!(doc.contains("advisory-only"));
        assert!(doc.contains("deferred/blocked/insufficient/diagnostic-only/reference-only"));
        assert!(doc.contains("stable current model mode"));
        assert!(doc.contains("non-canonical/internal-only"));
        assert!(doc.contains("abstract functional current mode"));
        assert!(doc.contains("no direct action trigger"));
        assert!(doc.contains("no direct execution trigger"));
        assert!(doc.contains("no direct retry trigger"));
        assert!(doc.contains("no direct memory commit"));
        assert!(doc.contains("no direct compute invocation"));
        assert!(doc.contains("no safety override"));
        assert!(doc.contains("zweite anatomische Region"));
        assert!(doc.contains("Stabilisierungspass der ersten anatomischen Region"));
        assert!(doc.contains("maintenance-only Core"));
    }

    #[test]
    fn cerebellum_br5_role_map_is_prediction_timing_correction_and_abstract_functional() {
        assert!(CANONICAL_BLUE_BRAIN_ANATOMICAL_REGION_MAP
            .contains(&BlueBrainAnatomicalRegionClass::Cerebellum));
        assert!(CANONICAL_BLUE_BRAIN_CEREBELLUM_ROLE_MAP
            .contains(&BlueBrainCerebellumRoleClass::PredictionRole));
        assert!(CANONICAL_BLUE_BRAIN_CEREBELLUM_ROLE_MAP
            .contains(&BlueBrainCerebellumRoleClass::TimingCoordinationRole));
        assert!(CANONICAL_BLUE_BRAIN_CEREBELLUM_ROLE_MAP
            .contains(&BlueBrainCerebellumRoleClass::ErrorCorrectionMismatchShapingRole));
        assert!(CANONICAL_BLUE_BRAIN_CEREBELLUM_ROLE_MAP
            .contains(&BlueBrainCerebellumRoleClass::BoundedExecutionSupportRole));
        assert!(CANONICAL_BLUE_BRAIN_CEREBELLUM_ROLE_MAP
            .contains(&BlueBrainCerebellumRoleClass::NonRoleOutOfScopeBiologicalDetail));

        assert!(CANONICAL_BLUE_BRAIN_CEREBELLUM_INTEGRATION_MAP
            .contains(&BlueBrainCerebellumIntegrationClass::CerebellumInputSurface));
        assert!(CANONICAL_BLUE_BRAIN_CEREBELLUM_INTEGRATION_MAP
            .contains(&BlueBrainCerebellumIntegrationClass::CerebellumStateSurface));
        assert!(CANONICAL_BLUE_BRAIN_CEREBELLUM_INTEGRATION_MAP
            .contains(&BlueBrainCerebellumIntegrationClass::CerebellumOutputAdvisorySurface));
        assert!(CANONICAL_BLUE_BRAIN_CEREBELLUM_INTEGRATION_MAP
            .contains(&BlueBrainCerebellumIntegrationClass::CerebellumReferenceSurface));

        assert_eq!(
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::Cerebellum),
            BlueBrainAnatomicalRegionSystemRoleClass::PredictionTimingCorrectionMediation
        );
        assert_eq!(
            blue_brain_anatomical_region_model_mode(BlueBrainAnatomicalRegionClass::Cerebellum),
            BlueBrainFirstAnatomicalRegionModelModeClass::AbstractFunctionalCurrentMode
        );
        assert_ne!(
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::Cerebellum),
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::Hippocampus)
        );
        assert_ne!(
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::Cerebellum),
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::Amygdala)
        );
        assert_ne!(
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::Cerebellum),
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::Thalamus)
        );
        assert_ne!(
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::Cerebellum),
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::BasalGanglia)
        );
    }

    #[test]
    fn cerebellum_br5_prompt1_doc_pins_role_mode_boundaries_and_handoff() {
        let doc = include_str!(
            "../../../docs/blue_brain_cerebellum_region_role_map_serie_br5_prompt1_v1.md"
        );

        assert!(doc.contains("cerebellum_like_region"));
        assert!(doc.contains("prediction role"));
        assert!(doc.contains("timing/coordination role"));
        assert!(doc.contains("error-correction or mismatch-shaping role"));
        assert!(doc.contains("bounded execution-support role"));
        assert!(doc.contains("abstract functional current mode"));
        assert!(doc.contains("no direct action trigger"));
        assert!(doc.contains("no direct execution trigger"));
        assert!(doc.contains("no retry orchestration"));
        assert!(doc.contains("no direct memory commit"));
        assert!(doc.contains("no direct compute invocation"));
        assert!(doc.contains("Cerebellum vor Hypothalamus"));
        assert!(doc.contains("keine semantische Dublette"));
    }

    #[test]
    fn cerebellum_br5_prompt4_readiness_map_and_next_direction_are_pinned() {
        assert_eq!(
            CANONICAL_BLUE_BRAIN_CEREBELLUM_EXPANSION_READINESS_MAP,
            [
                BlueBrainCerebellumExpansionReadinessClass::StableCerebellumOperationalSurface,
                BlueBrainCerebellumExpansionReadinessClass::UsableWithCaveats,
                BlueBrainCerebellumExpansionReadinessClass::AdvisoryOnly,
                BlueBrainCerebellumExpansionReadinessClass::DeferredBlockedInsufficientDiagnosticOnlyReferenceOnly,
                BlueBrainCerebellumExpansionReadinessClass::StableCurrentModelMode,
                BlueBrainCerebellumExpansionReadinessClass::NonCanonicalInternalOnly,
            ]
        );
        assert_ne!(
            BlueBrainCerebellumExpansionReadinessClass::AdvisoryOnly,
            BlueBrainCerebellumExpansionReadinessClass::UsableWithCaveats
        );
        assert_ne!(
            BlueBrainCerebellumExpansionReadinessClass::DeferredBlockedInsufficientDiagnosticOnlyReferenceOnly,
            BlueBrainCerebellumExpansionReadinessClass::NonCanonicalInternalOnly
        );
        assert_eq!(
            BLUE_BRAIN_POST_BR5_PRIORITIZED_NEXT_DIRECTION,
            BlueBrainPostBr5NextDirection::InterRegionArchitectureStage
        );
        assert_ne!(
            BLUE_BRAIN_POST_BR5_PRIORITIZED_NEXT_DIRECTION,
            BlueBrainPostBr5NextDirection::Hypothalamus
        );
    }

    #[test]
    fn cerebellum_br5_prompt2_integration_map_surfaces_are_canonical_and_distinct() {
        assert_eq!(CANONICAL_BLUE_BRAIN_CEREBELLUM_INTEGRATION_MAP.len(), 6);
        assert!(CANONICAL_BLUE_BRAIN_CEREBELLUM_INTEGRATION_MAP
            .contains(&BlueBrainCerebellumIntegrationClass::CerebellumInputSurface));
        assert!(CANONICAL_BLUE_BRAIN_CEREBELLUM_INTEGRATION_MAP
            .contains(&BlueBrainCerebellumIntegrationClass::CerebellumStateSurface));
        assert!(CANONICAL_BLUE_BRAIN_CEREBELLUM_INTEGRATION_MAP
            .contains(&BlueBrainCerebellumIntegrationClass::CerebellumOutputAdvisorySurface));
        assert!(CANONICAL_BLUE_BRAIN_CEREBELLUM_INTEGRATION_MAP
            .contains(&BlueBrainCerebellumIntegrationClass::CerebellumReferenceSurface));
        assert!(CANONICAL_BLUE_BRAIN_CEREBELLUM_INTEGRATION_MAP
            .contains(&BlueBrainCerebellumIntegrationClass::BlockedDeferredCerebellumPath));
        assert!(CANONICAL_BLUE_BRAIN_CEREBELLUM_INTEGRATION_MAP.contains(
            &BlueBrainCerebellumIntegrationClass::NonCanonicalInternalOnlyCerebellumPath
        ));
        assert_ne!(
            BlueBrainCerebellumIntegrationClass::CerebellumInputSurface,
            BlueBrainCerebellumIntegrationClass::CerebellumStateSurface
        );
        assert_ne!(
            BlueBrainCerebellumIntegrationClass::CerebellumOutputAdvisorySurface,
            BlueBrainCerebellumIntegrationClass::CerebellumReferenceSurface
        );
    }

    #[test]
    fn cerebellum_br5_prompt2_input_guards_reject_direct_authority_sources() {
        assert_eq!(
            classify_blue_brain_cerebellum_input_guard(
                BlueBrainCerebellumInputSource::RuntimePredictionSignal
            ),
            BlueBrainCerebellumInputGuard::AdvisoryOnlyInput
        );
        assert_eq!(
            classify_blue_brain_cerebellum_input_guard(
                BlueBrainCerebellumInputSource::ExecutionFeedbackMismatchSignal
            ),
            BlueBrainCerebellumInputGuard::AdvisoryOnlyInput
        );
        assert_eq!(
            classify_blue_brain_cerebellum_input_guard(
                BlueBrainCerebellumInputSource::ContextReferenceSignal
            ),
            BlueBrainCerebellumInputGuard::ReferenceOnlyBoundedInput
        );
        assert_eq!(
            classify_blue_brain_cerebellum_input_guard(
                BlueBrainCerebellumInputSource::ToolActionControlSignal
            ),
            BlueBrainCerebellumInputGuard::RejectedToolActionControl
        );
        assert_eq!(
            classify_blue_brain_cerebellum_input_guard(
                BlueBrainCerebellumInputSource::ComputeInternalRawStateSignal
            ),
            BlueBrainCerebellumInputGuard::RejectedComputeInternalRawState
        );
        assert_eq!(
            classify_blue_brain_cerebellum_input_guard(
                BlueBrainCerebellumInputSource::SafetyOverrideSignal
            ),
            BlueBrainCerebellumInputGuard::RejectedSafetyOverride
        );
        assert_eq!(
            classify_blue_brain_cerebellum_input_guard(
                BlueBrainCerebellumInputSource::ImplicitMemoryMutationSignal
            ),
            BlueBrainCerebellumInputGuard::RejectedImplicitMemoryMutation
        );
    }

    #[test]
    fn cerebellum_br5_prompt2_runtime_selection_execution_reference_reads_stay_bounded() {
        let cases = [
            (
                BlueBrainCerebellumInputSurface {
                    selection_signal: BlueBrainControlAttentionSelectionClass::AttentionTarget,
                    deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                    reference_validity: BlueBrainReferenceValidity::Current,
                    context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
                },
                BlueBrainCerebellumStateSurface::ActivePredictionTimingAdvisoryOnly,
                BlueBrainCerebellumCanonicalRead::AdvisoryOnly,
            ),
            (
                BlueBrainCerebellumInputSurface {
                    selection_signal: BlueBrainControlAttentionSelectionClass::EvidenceReferenceSelection,
                    deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                    reference_validity: BlueBrainReferenceValidity::ReferenceOnly,
                    context_priority: BlueBrainContextEvidencePriorityClass::SupportingEvidenceReference,
                },
                BlueBrainCerebellumStateSurface::ReferenceOnlyCorrectionState,
                BlueBrainCerebellumCanonicalRead::DiagnosticOnly,
            ),
            (
                BlueBrainCerebellumInputSurface {
                    selection_signal: BlueBrainControlAttentionSelectionClass::ContextSelection,
                    deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateStale,
                    reference_validity: BlueBrainReferenceValidity::Stale,
                    context_priority: BlueBrainContextEvidencePriorityClass::StaleContext,
                },
                BlueBrainCerebellumStateSurface::DeferredCorrectionState,
                BlueBrainCerebellumCanonicalRead::Deferred,
            ),
            (
                BlueBrainCerebellumInputSurface {
                    selection_signal: BlueBrainControlAttentionSelectionClass::ContextSelection,
                    deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                    reference_validity: BlueBrainReferenceValidity::Caveated,
                    context_priority: BlueBrainContextEvidencePriorityClass::CaveatedEvidenceReference,
                },
                BlueBrainCerebellumStateSurface::ExecutionSupportCaveatState,
                BlueBrainCerebellumCanonicalRead::Caveated,
            ),
            (
                BlueBrainCerebellumInputSurface {
                    selection_signal: BlueBrainControlAttentionSelectionClass::ContextSelection,
                    deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateRejected,
                    reference_validity: BlueBrainReferenceValidity::Blocked,
                    context_priority: BlueBrainContextEvidencePriorityClass::IgnoredContext,
                },
                BlueBrainCerebellumStateSurface::BlockedCorrectionState,
                BlueBrainCerebellumCanonicalRead::Blocked,
            ),
            (
                BlueBrainCerebellumInputSurface {
                    selection_signal:
                        BlueBrainControlAttentionSelectionClass::NonCanonicalInternalOnlySelectionPath,
                    deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                    reference_validity: BlueBrainReferenceValidity::Current,
                    context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
                },
                BlueBrainCerebellumStateSurface::NonCanonicalInternalOnly,
                BlueBrainCerebellumCanonicalRead::NonCanonicalInternalOnly,
            ),
        ];

        for (input, expected_state, expected_read) in cases {
            let (state, output) =
                evaluate_blue_brain_cerebellum_prediction_timing_correction(input);
            assert_eq!(state, expected_state);
            assert_eq!(output.canonical_contract_read, expected_read);
            assert_eq!(
                blue_brain_cerebellum_consumer_contract_read(
                    output,
                    BlueBrainCerebellumConsumerLayer::Runtime
                ),
                expected_read
            );
            assert_eq!(
                blue_brain_cerebellum_consumer_contract_read(
                    output,
                    BlueBrainCerebellumConsumerLayer::Selection
                ),
                expected_read
            );
            assert_eq!(
                blue_brain_cerebellum_consumer_contract_read(
                    output,
                    BlueBrainCerebellumConsumerLayer::ExecutionInterface
                ),
                expected_read
            );
            assert_eq!(
                blue_brain_cerebellum_consumer_contract_read(
                    output,
                    BlueBrainCerebellumConsumerLayer::Reference
                ),
                expected_read
            );
            assert!(blue_brain_cerebellum_consumer_contract_reads_are_aligned(
                output
            ));
            assert!(blue_brain_cerebellum_output_has_no_direct_authority(output));
            assert!(output.runtime_advisory_only);
            assert!(output.selection_advisory_only);
            assert!(output.execution_support_caveat_only);
            assert!(output.reference_bounded_only);
            assert!(!output.direct_action_selection);
            assert!(!output.direct_action_trigger);
            assert!(!output.direct_execution_trigger);
            assert!(!output.direct_retry_trigger);
            assert!(!output.direct_memory_commit);
            assert!(!output.direct_compute_invocation);
            assert!(!output.safety_override);
        }
    }

    #[test]
    fn cerebellum_br5_prompt2_diagnostics_contracts_mode_and_region_boundaries_are_pinned() {
        assert_eq!(
            CANONICAL_BLUE_BRAIN_CEREBELLUM_DIAGNOSTICS_CONTRACT_MAP,
            [
                BlueBrainCerebellumContractClass::CerebellumAdvisoryOnlyDiagnostic,
                BlueBrainCerebellumContractClass::CerebellumCaveatedDiagnostic,
                BlueBrainCerebellumContractClass::CerebellumDeferredDiagnostic,
                BlueBrainCerebellumContractClass::CerebellumBlockedDiagnostic,
                BlueBrainCerebellumContractClass::CerebellumInsufficientDiagnostic,
                BlueBrainCerebellumContractClass::CerebellumDiagnosticOnlyState,
                BlueBrainCerebellumContractClass::CerebellumBoundedContractSignal,
                BlueBrainCerebellumContractClass::NonCanonicalInternalOnlyCerebellumPath,
            ]
        );
        assert_eq!(
            blue_brain_cerebellum_contract_class_for_signal(
                BlueBrainCerebellumContractSignal::RuntimeToCerebellumBoundedPredictionTimingInput
            ),
            BlueBrainCerebellumContractClass::CerebellumBoundedContractSignal
        );
        assert_ne!(
            BlueBrainCerebellumDiagnosticState::CerebellumAdvisoryOnlyDiagnostic,
            BlueBrainCerebellumDiagnosticState::CerebellumCaveatedDiagnostic
        );
        assert_ne!(
            BlueBrainCerebellumDiagnosticState::CerebellumDeferredDiagnostic,
            BlueBrainCerebellumDiagnosticState::CerebellumBlockedDiagnostic
        );
        assert_eq!(
            blue_brain_anatomical_region_model_mode(BlueBrainAnatomicalRegionClass::Cerebellum),
            BlueBrainFirstAnatomicalRegionModelModeClass::AbstractFunctionalCurrentMode
        );
        assert_ne!(
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::Cerebellum),
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::Hippocampus)
        );
        assert_ne!(
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::Cerebellum),
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::Amygdala)
        );
        assert_ne!(
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::Cerebellum),
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::Thalamus)
        );
        assert_ne!(
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::Cerebellum),
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::BasalGanglia)
        );
    }

    #[test]
    fn cerebellum_br5_prompt3_contract_map_separates_all_canonical_reads() {
        assert_eq!(
            CANONICAL_BLUE_BRAIN_CEREBELLUM_DIAGNOSTICS_CONTRACT_ENTRIES.len(),
            CANONICAL_BLUE_BRAIN_CEREBELLUM_DIAGNOSTICS_CONTRACT_MAP.len()
        );
        for (entry, class) in CANONICAL_BLUE_BRAIN_CEREBELLUM_DIAGNOSTICS_CONTRACT_ENTRIES
            .iter()
            .zip(CANONICAL_BLUE_BRAIN_CEREBELLUM_DIAGNOSTICS_CONTRACT_MAP)
        {
            assert_eq!(entry.class, class);
            assert!(!entry.direct_authority_allowed);
        }
        assert_eq!(
            blue_brain_cerebellum_canonical_read_for_signal(
                BlueBrainCerebellumContractSignal::CerebellumToRuntimeAdvisory
            ),
            BlueBrainCerebellumCanonicalRead::AdvisoryOnly
        );
        assert_eq!(
            blue_brain_cerebellum_canonical_read_for_signal(
                BlueBrainCerebellumContractSignal::CerebellumExecutionSupportCaveatSignal
            ),
            BlueBrainCerebellumCanonicalRead::Caveated
        );
        assert_ne!(
            BlueBrainCerebellumCanonicalRead::AdvisoryOnly,
            BlueBrainCerebellumCanonicalRead::Caveated
        );
        assert_ne!(
            BlueBrainCerebellumCanonicalRead::Deferred,
            BlueBrainCerebellumCanonicalRead::Blocked
        );
        assert_ne!(
            BlueBrainCerebellumCanonicalRead::Blocked,
            BlueBrainCerebellumCanonicalRead::Insufficient
        );
        assert_ne!(
            BlueBrainCerebellumCanonicalRead::DiagnosticOnly,
            BlueBrainCerebellumCanonicalRead::AdvisoryOnly
        );
    }

    #[test]
    fn cerebellum_br5_prompt3_reference_only_and_caveat_do_not_promote_to_authority() {
        let reference_only = evaluate_blue_brain_cerebellum_prediction_timing_correction(
            BlueBrainCerebellumInputSurface {
                selection_signal:
                    BlueBrainControlAttentionSelectionClass::EvidenceReferenceSelection,
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::ReferenceOnly,
                context_priority:
                    BlueBrainContextEvidencePriorityClass::SupportingEvidenceReference,
            },
        )
        .1;
        assert_eq!(
            reference_only.canonical_contract_read,
            BlueBrainCerebellumCanonicalRead::DiagnosticOnly
        );
        assert_eq!(
            reference_only.runtime_diagnostic_state,
            BlueBrainCerebellumDiagnosticState::CerebellumDiagnosticOnlyState
        );
        assert_eq!(
            reference_only.selection_diagnostic_state,
            BlueBrainCerebellumDiagnosticState::CerebellumDiagnosticOnlyState
        );
        assert_eq!(
            reference_only.reference_diagnostic_state,
            BlueBrainCerebellumDiagnosticState::CerebellumDiagnosticOnlyState
        );
        assert!(blue_brain_cerebellum_consumer_contract_reads_are_aligned(
            reference_only
        ));
        assert!(blue_brain_cerebellum_output_has_no_direct_authority(
            reference_only
        ));

        let caveated = evaluate_blue_brain_cerebellum_prediction_timing_correction(
            BlueBrainCerebellumInputSurface {
                selection_signal: BlueBrainControlAttentionSelectionClass::ContextSelection,
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::Caveated,
                context_priority: BlueBrainContextEvidencePriorityClass::CaveatedEvidenceReference,
            },
        )
        .1;
        assert_eq!(
            caveated.canonical_contract_read,
            BlueBrainCerebellumCanonicalRead::Caveated
        );
        assert_eq!(
            caveated.runtime_diagnostic_state,
            BlueBrainCerebellumDiagnosticState::CerebellumCaveatedDiagnostic
        );
        assert_ne!(
            caveated.runtime_diagnostic_state,
            BlueBrainCerebellumDiagnosticState::CerebellumAdvisoryOnlyDiagnostic
        );
        assert!(blue_brain_cerebellum_consumer_contract_reads_are_aligned(
            caveated
        ));
        assert!(blue_brain_cerebellum_output_has_no_direct_authority(
            caveated
        ));
    }

    #[test]
    fn cerebellum_br5_prompt2_doc_pins_minimal_bounded_integration_line() {
        let doc = include_str!(
            "../../../docs/blue_brain_cerebellum_minimal_bounded_integration_serie_br5_prompt2_v1.md"
        );
        assert!(doc.contains("cerebellum input surface"));
        assert!(doc.contains("cerebellum state surface"));
        assert!(doc.contains("cerebellum output/advisory surface"));
        assert!(doc.contains("cerebellum reference surface"));
        assert!(doc.contains("blocked/deferred cerebellum path"));
        assert!(doc.contains("non-canonical/internal-only cerebellum path"));
        assert!(doc.contains("timing hint"));
        assert!(doc.contains("correction hint"));
        assert!(doc.contains("mismatch hint"));
        assert!(doc.contains("execution-support caveat"));
        assert!(doc.contains("reference-bounded signal"));
        assert!(doc.contains("abstract functional current mode"));
        assert!(doc.contains("Hodgkin-Huxley simulation-only/diagnostic-only"));
        assert!(doc.contains("hippocampus_like_region"));
        assert!(doc.contains("amygdala_like_region"));
        assert!(doc.contains("thalamus_like_region"));
        assert!(doc.contains("basal_ganglia_like_region"));
        assert!(doc.contains("cerebellum_like_region"));
        assert!(doc.contains("no direct action trigger"));
        assert!(doc.contains("no direct execution trigger"));
        assert!(doc.contains("no direct retry trigger"));
        assert!(doc.contains("no direct memory commit"));
        assert!(doc.contains("no automatic memory persistence"));
        assert!(doc.contains("no direct compute invocation"));
        assert!(doc.contains("no safety override"));
        assert!(doc.contains("öffnet keine sechste Region"));
    }

    #[test]
    fn cerebellum_br5_prompt4_doc_pins_readiness_expansion_boundary_and_next_stage() {
        let doc = include_str!(
            "../../../docs/blue_brain_br5_cerebellum_readiness_sweep_expansion_boundary_serie_br5_prompt4_v1.md"
        );
        assert!(doc.contains("BR5-expansion-readiness map"));
        assert!(doc.contains("stable cerebellum operational surface"));
        assert!(doc.contains("usable with caveats"));
        assert!(doc.contains("advisory-only"));
        assert!(doc.contains("deferred/blocked/insufficient/diagnostic-only/reference-only"));
        assert!(doc.contains("stable current model mode"));
        assert!(doc.contains("non-canonical/internal-only"));
        assert!(doc.contains("fünfte echte anatomische Hirnregion"));
        assert!(doc.contains("abstract functional current mode"));
        assert!(doc.contains("bounded Kuramoto-like candidate"));
        assert!(doc.contains("Hodgkin-Huxley simulation-only/diagnostic-only"));
        assert!(doc.contains("no direct action execution"));
        assert!(doc.contains("no retry orchestration"));
        assert!(doc.contains("no automatic memory persistence"));
        assert!(doc.contains("no direct compute invocation"));
        assert!(doc.contains("no safety override"));
        assert!(doc.contains("keine sechste Hirnregion"));
        assert!(doc.contains("Compute bleibt maintenance-only"));
        assert!(doc.contains("inter-region architecture stage"));
        assert!(doc.contains("Hypothalamus wartet"));
    }

    #[test]
    fn docs_indexes_expose_br5_cerebellum_readiness_sweep() {
        let readme = include_str!("../../../docs/README.md");
        assert!(readme.contains("Cerebellum-next role consolidation (BR5)"));
        assert!(readme.contains(
            "docs/blue_brain_br5_cerebellum_readiness_sweep_expansion_boundary_serie_br5_prompt4_v1.md"
        ));
    }

    #[test]
    fn first_anatomical_stabilization_map_is_maintenance_hardened_and_model_bounded() {
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_STABILIZATION_MAP
            .contains(&BlueBrainFirstRegionStabilizationClass::StableFirstRegionBaseline));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_STABILIZATION_MAP
            .contains(&BlueBrainFirstRegionStabilizationClass::MaintenanceHardenedRegionSurface));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_STABILIZATION_MAP
            .contains(&BlueBrainFirstRegionStabilizationClass::MaintenanceHardenedDiagnosticsPath));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_STABILIZATION_MAP
            .contains(&BlueBrainFirstRegionStabilizationClass::MaintenanceHardenedContractPath));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_STABILIZATION_MAP
            .contains(&BlueBrainFirstRegionStabilizationClass::MaintenanceHardenedModelBoundary));
        assert!(
            CANONICAL_BLUE_BRAIN_FIRST_REGION_STABILIZATION_MAP.contains(
                &BlueBrainFirstRegionStabilizationClass::NonCanonicalInternalOnlyResidualPath
            )
        );
    }

    #[test]
    fn first_anatomical_stabilization_doc_pins_surface_contract_model_and_guards() {
        let doc = include_str!(
            "../../../docs/blue_brain_first_anatomical_stabilization_line_serie_bb31_prompt1_v1.md"
        );
        assert!(doc.contains("stable first-anatomical baseline"));
        assert!(doc.contains("maintenance-hardened anatomical surface"));
        assert!(doc.contains("maintenance-hardened diagnostics path"));
        assert!(doc.contains("maintenance-hardened contract path"));
        assert!(doc.contains("maintenance-hardened model boundary"));
        assert!(doc.contains("non-canonical/internal-only residual path"));
        assert!(doc.contains("advisory-only remains advisory-only"));
        assert!(doc.contains("reference-only remains reference-only"));
        assert!(doc.contains("no direct action trigger"));
        assert!(doc.contains("no direct execution trigger"));
        assert!(doc.contains("no direct retry trigger"));
        assert!(doc.contains("no direct memory commit"));
        assert!(doc.contains("no direct compute invocation"));
        assert!(doc.contains("no safety override"));
        assert!(doc.contains("no second anatomical region"));
        assert!(doc.contains("abstract functional current mode"));
    }

    #[test]
    fn first_anatomical_docs_tests_index_cleanup_doc_pins_canonical_reference_map() {
        let doc = include_str!(
            "../../../docs/blue_brain_first_anatomical_docs_tests_index_cleanup_serie_bb31_prompt2_v1.md"
        );
        assert!(doc.contains("canonical anatomical-region reference doc"));
        assert!(doc.contains("canonical anatomical-region test surface"));
        assert!(doc.contains("maintenance-facing index/reference path"));
        assert!(doc.contains("non-canonical/internal-only or legacy anatomical-region path"));
        assert!(doc.contains("hippocampus_like_region"));
        assert!(doc.contains("no direct action trigger"));
        assert!(doc.contains("no direct execution trigger"));
        assert!(doc.contains("no direct retry trigger"));
        assert!(doc.contains("no direct memory commit"));
        assert!(doc.contains("no direct compute invocation"));
        assert!(doc.contains("no safety override"));
        assert!(doc.contains("keine zweite anatomische Region"));
    }

    #[test]
    fn docs_indexes_expose_bb31_first_anatomical_maintenance_reference_line() {
        let readme = include_str!("../../../docs/README.md");
        assert!(readme.contains("First anatomical maintenance stabilization/reference line (BB31)"));
        assert!(readme.contains(
            "docs/blue_brain_first_anatomical_stabilization_line_serie_bb31_prompt1_v1.md"
        ));
        assert!(readme.contains(
            "docs/blue_brain_first_anatomical_docs_tests_index_cleanup_serie_bb31_prompt2_v1.md"
        ));
        assert!(readme.contains(
            "docs/blue_brain_bb31_final_first_anatomical_stabilization_sweep_serie_bb31_prompt3_v1.md"
        ));

        let repo_map = include_str!("../../../docs/roadmap/REPO_MAP.md");
        assert!(repo_map.contains("First-anatomical maintenance reference surface (BB31)"));
        assert!(repo_map.contains(
            "docs/blue_brain_first_anatomical_docs_tests_index_cleanup_serie_bb31_prompt2_v1.md"
        ));
        assert!(repo_map.contains(
            "docs/blue_brain_bb31_final_first_anatomical_stabilization_sweep_serie_bb31_prompt3_v1.md"
        ));
    }

    #[test]
    fn bb31_prompt3_final_sweep_doc_pins_stability_map_and_maintenance_default() {
        let doc = include_str!(
            "../../../docs/blue_brain_bb31_final_first_anatomical_stabilization_sweep_serie_bb31_prompt3_v1.md"
        );
        assert!(doc.contains("hippocampus_like_region"));
        assert!(doc.contains("stable maintenance-hardened first-anatomical baseline"));
        assert!(doc.contains("usable-with-caveats first-anatomical contract lane"));
        assert!(doc.contains("advisory-only anatomical output lane"));
        assert!(doc.contains("diagnostic-only/deferred anatomical diagnostics lane"));
        assert!(doc.contains("non-canonical/internal-only anatomical residual lane"));
        assert!(doc.contains("no direct action/execution/retry/memory/compute"));
        assert!(doc.contains("no safety override"));
        assert!(doc.contains("Maintenance/Bugfix/Cleanup genügt"));
        assert!(doc.contains("expliziter anatomischer Region-2-Re-Scope"));
    }

    #[test]
    fn canonical_anatomical_region_map_pins_roles_and_model_modes() {
        assert!(CANONICAL_BLUE_BRAIN_ANATOMICAL_REGION_MAP
            .contains(&BlueBrainAnatomicalRegionClass::Hippocampus));
        assert!(CANONICAL_BLUE_BRAIN_ANATOMICAL_REGION_MAP
            .contains(&BlueBrainAnatomicalRegionClass::Amygdala));
        assert!(CANONICAL_BLUE_BRAIN_ANATOMICAL_REGION_MAP
            .contains(&BlueBrainAnatomicalRegionClass::PrefrontalCortex));
        assert!(CANONICAL_BLUE_BRAIN_ANATOMICAL_REGION_MAP
            .contains(&BlueBrainAnatomicalRegionClass::AnteriorCingulateCortex));
        assert!(CANONICAL_BLUE_BRAIN_ANATOMICAL_REGION_MAP
            .contains(&BlueBrainAnatomicalRegionClass::BasalGanglia));
        assert!(CANONICAL_BLUE_BRAIN_ANATOMICAL_REGION_MAP
            .contains(&BlueBrainAnatomicalRegionClass::Thalamus));
        assert!(CANONICAL_BLUE_BRAIN_ANATOMICAL_REGION_MAP
            .contains(&BlueBrainAnatomicalRegionClass::Insula));

        assert_eq!(
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::Hippocampus),
            BlueBrainAnatomicalRegionSystemRoleClass::AttentionSelectionMediation
        );
        assert_eq!(
            blue_brain_anatomical_region_model_mode(BlueBrainAnatomicalRegionClass::Hippocampus),
            BlueBrainFirstAnatomicalRegionModelModeClass::AbstractFunctionalCurrentMode
        );
        assert_eq!(
            blue_brain_anatomical_region_model_mode(BlueBrainAnatomicalRegionClass::BasalGanglia),
            BlueBrainFirstAnatomicalRegionModelModeClass::AbstractFunctionalCurrentMode
        );
    }

    #[test]
    fn anatomical_region_canonical_map_doc_pins_scope_and_deferred_boundary() {
        let doc = include_str!(
            "../../../docs/blue_brain_anatomical_region_canonical_map_serie_bb32_prompt1_v1.md"
        );
        assert!(doc.contains("canonical anatomical region map"));
        assert!(doc.contains("hippocampus"));
        assert!(doc.contains("amygdala"));
        assert!(doc.contains("prefrontal cortex"));
        assert!(doc.contains("bounded kuramoto-like"));
        assert!(doc.contains("HH simulation-only/diagnostic-only"));
        assert!(doc.contains("later selective HH deepening"));
        assert!(doc.contains("deferred"));
        assert!(doc.contains("no direct action trigger"));
        assert!(doc.contains("no direct execution trigger"));
        assert!(doc.contains("no direct retry trigger"));
        assert!(doc.contains("no direct memory commit"));
        assert!(doc.contains("no direct compute invocation"));
    }

    #[test]
    fn amygdala_br2_role_map_doc_pins_role_integration_mode_boundaries_and_separation() {
        let doc = include_str!(
            "../../../docs/blue_brain_amygdala_region_role_map_serie_br2_prompt1_v1.md"
        );
        assert!(doc.contains("salience weighting role"));
        assert!(doc.contains("threat/valence caveat role"));
        assert!(doc.contains("bounded priority modulation role"));
        assert!(doc.contains("reference-linked affective tagging role"));
        assert!(doc.contains("non-role / out-of-scope biological detail"));
        assert!(doc.contains("abstract functional current mode"));
        assert!(doc.contains("HH bleibt `simulation-only/diagnostic-only`"));
        assert!(doc.contains("later selective HH deepening"));
        assert!(doc.contains("hippocampus_like_region"));
        assert!(doc.contains("amygdala_like_region"));
        assert!(doc.contains("no direct action trigger"));
        assert!(doc.contains("no direct execution trigger"));
        assert!(doc.contains("no direct retry trigger"));
        assert!(doc.contains("no direct memory commit"));
        assert!(doc.contains("no direct compute invocation"));
        assert!(doc.contains("no safety override"));
        assert!(doc.contains("keine HH-Produktivintegration"));
        assert!(doc.contains("keine neue Compute-Core-Arbeit"));
    }

    #[test]
    fn thalamus_br3_role_map_doc_pins_role_integration_mode_boundaries_and_separation() {
        let doc = include_str!(
            "../../../docs/blue_brain_thalamus_region_role_map_serie_br3_prompt1_v1.md"
        );
        assert!(doc.contains("relay/gating role"));
        assert!(doc.contains("bounded routing role"));
        assert!(doc.contains("selection-support role"));
        assert!(doc.contains("reference-mediated signal routing role"));
        assert!(doc.contains("non-role / out-of-scope biological detail"));
        assert!(doc.contains("abstract functional current mode"));
        assert!(doc.contains("bounded Kuramoto-like candidate"));
        assert!(doc.contains("Hodgkin-Huxley simulation-only/diagnostic-only"));
        assert!(doc.contains("later selective HH deepening"));
        assert!(doc.contains("hippocampus_like_region"));
        assert!(doc.contains("amygdala_like_region"));
        assert!(doc.contains("thalamus_like_region"));
        assert!(doc.contains("no direct action trigger"));
        assert!(doc.contains("no direct execution trigger"));
        assert!(doc.contains("no direct retry trigger"));
        assert!(doc.contains("no direct memory commit"));
        assert!(doc.contains("no direct compute invocation"));
        assert!(doc.contains("no safety override"));
        assert!(doc.contains("kein vollständiger biologischer Thalamus-Nachbau"));
        assert!(doc.contains("keine HH-Produktivintegration"));
        assert!(doc.contains("keine neue Compute-Core-Arbeit"));
    }

    #[test]
    fn thalamus_model_mode_and_role_stay_distinct_from_hippocampus_and_amygdala() {
        assert_eq!(
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::Thalamus),
            BlueBrainAnatomicalRegionSystemRoleClass::RelayIntegrationMediation
        );
        assert_eq!(
            blue_brain_anatomical_region_model_mode(BlueBrainAnatomicalRegionClass::Thalamus),
            BlueBrainFirstAnatomicalRegionModelModeClass::AbstractFunctionalCurrentMode
        );
        assert_ne!(
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::Thalamus),
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::Amygdala)
        );
        assert_ne!(
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::Thalamus),
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::Hippocampus)
        );
    }

    #[test]
    fn basal_ganglia_br4_role_map_doc_pins_role_integration_mode_boundaries_and_separation() {
        let doc = include_str!(
            "../../../docs/blue_brain_basal_ganglia_region_role_map_serie_br4_prompt1_v1.md"
        );
        assert!(doc.contains("action gating role"));
        assert!(doc.contains("suppression/inhibition role"));
        assert!(doc.contains("bounded selection-channel arbitration role"));
        assert!(doc.contains("execution-readiness modulation role"));
        assert!(doc.contains("non-role / out-of-scope biological detail"));
        assert!(doc.contains("abstract functional current mode"));
        assert!(doc.contains("bounded Kuramoto-like candidate"));
        assert!(doc.contains("Hodgkin-Huxley simulation-only/diagnostic-only"));
        assert!(doc.contains("later selective HH deepening"));
        assert!(doc.contains("hippocampus_like_region"));
        assert!(doc.contains("amygdala_like_region"));
        assert!(doc.contains("thalamus_like_region"));
        assert!(doc.contains("basal_ganglia_like_region"));
        assert!(doc.contains("no direct action trigger"));
        assert!(doc.contains("no direct action selection"));
        assert!(doc.contains("no direct execution trigger"));
        assert!(doc.contains("no direct retry trigger"));
        assert!(doc.contains("no direct memory commit"));
        assert!(doc.contains("no direct compute invocation"));
        assert!(doc.contains("no safety override"));
        assert!(doc.contains("kein vollständiger biologischer Basal-Ganglia-Nachbau"));
        assert!(doc.contains("keine HH-Produktivintegration"));
        assert!(doc.contains("keine neue Compute-Core-Arbeit"));
    }

    #[test]
    fn basal_ganglia_br4_role_and_model_mode_stay_distinct_and_bounded() {
        assert!(CANONICAL_BLUE_BRAIN_BASAL_GANGLIA_ROLE_MAP
            .contains(&BlueBrainBasalGangliaRoleClass::ActionGatingRole));
        assert!(CANONICAL_BLUE_BRAIN_BASAL_GANGLIA_ROLE_MAP
            .contains(&BlueBrainBasalGangliaRoleClass::SuppressionInhibitionRole));
        assert!(CANONICAL_BLUE_BRAIN_BASAL_GANGLIA_ROLE_MAP
            .contains(&BlueBrainBasalGangliaRoleClass::BoundedSelectionChannelArbitrationRole));
        assert!(CANONICAL_BLUE_BRAIN_BASAL_GANGLIA_ROLE_MAP
            .contains(&BlueBrainBasalGangliaRoleClass::ExecutionReadinessModulationRole));
        assert!(CANONICAL_BLUE_BRAIN_BASAL_GANGLIA_ROLE_MAP
            .contains(&BlueBrainBasalGangliaRoleClass::NonRoleOutOfScopeBiologicalDetail));

        assert!(CANONICAL_BLUE_BRAIN_BASAL_GANGLIA_INTEGRATION_MAP
            .contains(&BlueBrainBasalGangliaIntegrationClass::BasalGangliaInputSurface));
        assert!(CANONICAL_BLUE_BRAIN_BASAL_GANGLIA_INTEGRATION_MAP
            .contains(&BlueBrainBasalGangliaIntegrationClass::BasalGangliaStateSurface));
        assert!(CANONICAL_BLUE_BRAIN_BASAL_GANGLIA_INTEGRATION_MAP
            .contains(&BlueBrainBasalGangliaIntegrationClass::BasalGangliaOutputAdvisorySurface));
        assert!(CANONICAL_BLUE_BRAIN_BASAL_GANGLIA_INTEGRATION_MAP
            .contains(&BlueBrainBasalGangliaIntegrationClass::BasalGangliaReferenceBoundedSurface));
        assert!(CANONICAL_BLUE_BRAIN_BASAL_GANGLIA_INTEGRATION_MAP
            .contains(&BlueBrainBasalGangliaIntegrationClass::BlockedDeferredBasalGangliaPath));
        assert!(CANONICAL_BLUE_BRAIN_BASAL_GANGLIA_INTEGRATION_MAP.contains(
            &BlueBrainBasalGangliaIntegrationClass::NonCanonicalInternalOnlyBasalGangliaPath
        ));

        assert_eq!(
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::BasalGanglia),
            BlueBrainAnatomicalRegionSystemRoleClass::ActionGatingMediation
        );
        assert_eq!(
            blue_brain_anatomical_region_model_mode(BlueBrainAnatomicalRegionClass::BasalGanglia),
            BlueBrainFirstAnatomicalRegionModelModeClass::AbstractFunctionalCurrentMode
        );
        assert_ne!(
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::BasalGanglia),
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::Hippocampus)
        );
        assert_ne!(
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::BasalGanglia),
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::Amygdala)
        );
        assert_ne!(
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::BasalGanglia),
            blue_brain_anatomical_region_system_role(BlueBrainAnatomicalRegionClass::Thalamus)
        );
    }

    #[test]
    fn hippocampus_br1_role_map_doc_pins_role_integration_mode_and_scope_boundaries() {
        let doc = include_str!(
            "../../../docs/blue_brain_hippocampus_region_role_map_serie_br1_prompt1_v1.md"
        );
        assert!(doc.contains("context indexing role"));
        assert!(doc.contains("memory association role"));
        assert!(doc.contains("episode/reference binding role"));
        assert!(doc.contains("bounded retrieval support role"));
        assert!(doc.contains("abstract functional (current/default)"));
        assert!(doc.contains("HH simulation-only/diagnostic-only"));
        assert!(doc.contains("later selective HH deepening"));
        assert!(doc.contains("No implicit HH requirement is introduced for Hippocampus in BR1."));
        assert!(doc.contains("no direct execution/safety/memory authority"));
        assert!(doc.contains("no new compute-core expansion"));
        assert!(doc.contains("no planner/agent/retry/orchestration platform"));
    }

    #[test]
    fn hippocampus_br1_integration_map_and_guards_remain_bounded() {
        assert!(CANONICAL_BLUE_BRAIN_HIPPOCAMPUS_INTEGRATION_MAP
            .contains(&BlueBrainHippocampusIntegrationClass::HippocampusInputSurface));
        assert!(CANONICAL_BLUE_BRAIN_HIPPOCAMPUS_INTEGRATION_MAP
            .contains(&BlueBrainHippocampusIntegrationClass::HippocampusStateSurface));
        assert!(CANONICAL_BLUE_BRAIN_HIPPOCAMPUS_INTEGRATION_MAP
            .contains(&BlueBrainHippocampusIntegrationClass::HippocampusOutputAdvisorySurface));
        assert!(CANONICAL_BLUE_BRAIN_HIPPOCAMPUS_INTEGRATION_MAP
            .contains(&BlueBrainHippocampusIntegrationClass::HippocampusReferenceSurface));
        assert!(CANONICAL_BLUE_BRAIN_HIPPOCAMPUS_INTEGRATION_MAP
            .contains(&BlueBrainHippocampusIntegrationClass::BlockedDeferredHippocampusPath));
        assert!(CANONICAL_BLUE_BRAIN_HIPPOCAMPUS_INTEGRATION_MAP.contains(
            &BlueBrainHippocampusIntegrationClass::NonCanonicalInternalOnlyHippocampusPath
        ));

        assert_eq!(
            classify_blue_brain_first_anatomical_region_input_guard(
                BlueBrainFirstAnatomicalRegionInputClass::ComputeInternalRawState
            ),
            BlueBrainFirstAnatomicalRegionInputGuard::BlockedForbiddenInput
        );
        assert_eq!(
            classify_blue_brain_first_anatomical_region_output_guard(
                BlueBrainFirstAnatomicalRegionOutputClass::DirectComputeInvocation
            ),
            BlueBrainFirstAnatomicalRegionOutputGuard::BlockedForbiddenOutput
        );
    }

    #[test]
    fn hippocampus_br1_integration_doc_pins_surfaces_contract_model_and_no_direct_boundaries() {
        let doc = include_str!(
            "../../../docs/blue_brain_hippocampus_minimal_bounded_integration_serie_br1_prompt2_v1.md"
        );
        assert!(doc.contains("hippocampus input surface"));
        assert!(doc.contains("hippocampus state surface"));
        assert!(doc.contains("hippocampus output/advisory surface"));
        assert!(doc.contains("hippocampus reference surface"));
        assert!(doc.contains("blocked/deferred hippocampus path"));
        assert!(doc.contains("non-canonical/internal-only hippocampus path"));
        assert!(doc.contains("runtime advisory read"));
        assert!(doc.contains("selection advisory read"));
        assert!(doc.contains("reference/context bounded read"));
        assert!(doc.contains("abstract functional current mode"));
        assert!(doc.contains("no direct action trigger"));
        assert!(doc.contains("no direct execution trigger"));
        assert!(doc.contains("no direct retry trigger"));
        assert!(doc.contains("no direct memory commit"));
        assert!(doc.contains("no direct compute invocation"));
        assert!(doc.contains("no safety override"));
        assert!(doc.contains("no parallel opening of additional anatomical regions"));
    }

    #[test]
    fn hippocampus_br1_prompt3_canonical_diagnostics_contract_map_is_complete_and_distinct() {
        assert!(CANONICAL_BLUE_BRAIN_HIPPOCAMPUS_DIAGNOSTICS_CONTRACT_MAP
            .contains(&BlueBrainHippocampusContractClass::HippocampusAdvisoryOnlyDiagnostic));
        assert!(CANONICAL_BLUE_BRAIN_HIPPOCAMPUS_DIAGNOSTICS_CONTRACT_MAP
            .contains(&BlueBrainHippocampusContractClass::HippocampusCaveatedDiagnostic));
        assert!(CANONICAL_BLUE_BRAIN_HIPPOCAMPUS_DIAGNOSTICS_CONTRACT_MAP
            .contains(&BlueBrainHippocampusContractClass::HippocampusDeferredDiagnostic));
        assert!(CANONICAL_BLUE_BRAIN_HIPPOCAMPUS_DIAGNOSTICS_CONTRACT_MAP
            .contains(&BlueBrainHippocampusContractClass::HippocampusBlockedDiagnostic));
        assert!(CANONICAL_BLUE_BRAIN_HIPPOCAMPUS_DIAGNOSTICS_CONTRACT_MAP
            .contains(&BlueBrainHippocampusContractClass::HippocampusInsufficientDiagnostic));
        assert!(CANONICAL_BLUE_BRAIN_HIPPOCAMPUS_DIAGNOSTICS_CONTRACT_MAP
            .contains(&BlueBrainHippocampusContractClass::HippocampusDiagnosticOnlyState));
        assert!(CANONICAL_BLUE_BRAIN_HIPPOCAMPUS_DIAGNOSTICS_CONTRACT_MAP
            .contains(&BlueBrainHippocampusContractClass::HippocampusBoundedContractSignal));
        assert!(CANONICAL_BLUE_BRAIN_HIPPOCAMPUS_DIAGNOSTICS_CONTRACT_MAP
            .contains(&BlueBrainHippocampusContractClass::NonCanonicalInternalOnlyHippocampusPath));
    }

    #[test]
    fn hippocampus_br1_prompt3_runtime_selection_reference_reads_stay_semantically_aligned() {
        let signal = BlueBrainFirstAnatomicalRegionContractSignal::Caveated;
        let runtime_read = blue_brain_first_anatomical_region_runtime_diagnostic_read(signal);
        let selection_read = blue_brain_first_anatomical_region_selection_diagnostic_read(signal);
        let reference_read = blue_brain_first_anatomical_region_reference_diagnostic_read(signal);
        assert_eq!(runtime_read, selection_read);
        assert_eq!(selection_read, reference_read);
        assert_eq!(
            blue_brain_hippocampus_contract_class_for_signal(signal),
            BlueBrainHippocampusContractClass::HippocampusCaveatedDiagnostic
        );
    }

    #[test]
    fn hippocampus_br1_prompt3_doc_pins_canonical_contract_map_and_guard_boundaries() {
        let doc = include_str!(
            "../../../docs/blue_brain_hippocampus_surface_diagnostics_contracts_hardening_serie_br1_prompt3_v1.md"
        );
        assert!(doc.contains("hippocampus advisory-only diagnostic"));
        assert!(doc.contains("hippocampus caveated diagnostic"));
        assert!(doc.contains("hippocampus deferred diagnostic"));
        assert!(doc.contains("hippocampus blocked diagnostic"));
        assert!(doc.contains("hippocampus insufficient diagnostic"));
        assert!(doc.contains("hippocampus diagnostic-only state"));
        assert!(doc.contains("hippocampus bounded contract signal"));
        assert!(doc.contains("non-canonical/internal-only hippocampus path"));
        assert!(doc.contains("deferred != blocked"));
        assert!(doc.contains("blocked != insufficient"));
        assert!(doc.contains("advisory-only != caveated"));
        assert!(doc.contains("no action request"));
        assert!(doc.contains("no execution trigger"));
        assert!(doc.contains("no retry trigger"));
        assert!(doc.contains("no memory commit"));
        assert!(doc.contains("no compute trigger"));
        assert!(doc.contains("no safety override"));
        assert!(doc.contains("abstract functional (current mode)"));
        assert!(doc.contains("HH simulation-only/diagnostic-only remains deferred"));
    }

    #[test]
    fn amygdala_br2_prompt3_canonical_diagnostics_contract_map_is_complete_and_distinct() {
        assert!(CANONICAL_BLUE_BRAIN_AMYGDALA_DIAGNOSTICS_CONTRACT_MAP
            .contains(&BlueBrainAmygdalaContractClass::AmygdalaAdvisoryOnlyDiagnostic));
        assert!(CANONICAL_BLUE_BRAIN_AMYGDALA_DIAGNOSTICS_CONTRACT_MAP
            .contains(&BlueBrainAmygdalaContractClass::AmygdalaCaveatedDiagnostic));
        assert!(CANONICAL_BLUE_BRAIN_AMYGDALA_DIAGNOSTICS_CONTRACT_MAP
            .contains(&BlueBrainAmygdalaContractClass::AmygdalaDeferredDiagnostic));
        assert!(CANONICAL_BLUE_BRAIN_AMYGDALA_DIAGNOSTICS_CONTRACT_MAP
            .contains(&BlueBrainAmygdalaContractClass::AmygdalaBlockedDiagnostic));
        assert!(CANONICAL_BLUE_BRAIN_AMYGDALA_DIAGNOSTICS_CONTRACT_MAP
            .contains(&BlueBrainAmygdalaContractClass::AmygdalaInsufficientDiagnostic));
        assert!(CANONICAL_BLUE_BRAIN_AMYGDALA_DIAGNOSTICS_CONTRACT_MAP
            .contains(&BlueBrainAmygdalaContractClass::AmygdalaDiagnosticOnlyState));
        assert!(CANONICAL_BLUE_BRAIN_AMYGDALA_DIAGNOSTICS_CONTRACT_MAP
            .contains(&BlueBrainAmygdalaContractClass::AmygdalaBoundedContractSignal));
        assert!(CANONICAL_BLUE_BRAIN_AMYGDALA_DIAGNOSTICS_CONTRACT_MAP
            .contains(&BlueBrainAmygdalaContractClass::NonCanonicalInternalOnlyAmygdalaPath));
    }

    #[test]
    fn amygdala_br2_prompt3_runtime_selection_reference_reads_stay_semantically_aligned() {
        let signal = BlueBrainFirstAnatomicalRegionContractSignal::Deferred;
        let runtime_read = blue_brain_first_anatomical_region_runtime_diagnostic_read(signal);
        let selection_read = blue_brain_first_anatomical_region_selection_diagnostic_read(signal);
        let reference_read = blue_brain_first_anatomical_region_reference_diagnostic_read(signal);
        assert_eq!(runtime_read, selection_read);
        assert_eq!(selection_read, reference_read);
        assert_eq!(
            blue_brain_amygdala_contract_class_for_signal(signal),
            BlueBrainAmygdalaContractClass::AmygdalaDeferredDiagnostic
        );
    }

    #[test]
    fn amygdala_br2_prompt3_doc_pins_surface_diagnostics_contracts_and_boundaries() {
        let doc = include_str!(
            "../../../docs/blue_brain_amygdala_surface_diagnostics_contracts_hardening_serie_br2_prompt3_v1.md"
        );
        assert!(doc.contains("amygdala input surface"));
        assert!(doc.contains("amygdala state surface"));
        assert!(doc.contains("amygdala output/advisory surface"));
        assert!(doc.contains("amygdala reference surface"));
        assert!(doc.contains("amygdala advisory-only diagnostic"));
        assert!(doc.contains("amygdala caveated diagnostic"));
        assert!(doc.contains("amygdala deferred diagnostic"));
        assert!(doc.contains("amygdala blocked diagnostic"));
        assert!(doc.contains("amygdala insufficient diagnostic"));
        assert!(doc.contains("amygdala diagnostic-only state"));
        assert!(doc.contains("amygdala bounded contract signal"));
        assert!(doc.contains("non-canonical/internal-only amygdala path"));
        assert!(doc.contains("advisory-only != caveated"));
        assert!(doc.contains("deferred != blocked"));
        assert!(doc.contains("blocked != insufficient"));
        assert!(doc.contains("no action request"));
        assert!(doc.contains("no execution trigger"));
        assert!(doc.contains("no retry trigger"));
        assert!(doc.contains("no memory commit"));
        assert!(doc.contains("no compute trigger"));
        assert!(doc.contains("no safety override"));
        assert!(doc.contains("current model mode remains unchanged"));
        assert!(doc.contains("hippocampus remains context/reference/episode/indexing"));
    }

    #[test]
    fn thalamus_br3_prompt3_canonical_diagnostics_contract_map_is_complete_and_distinct() {
        assert!(CANONICAL_BLUE_BRAIN_THALAMUS_DIAGNOSTICS_CONTRACT_MAP
            .contains(&BlueBrainThalamusContractClass::ThalamusAdvisoryOnlyDiagnostic));
        assert!(CANONICAL_BLUE_BRAIN_THALAMUS_DIAGNOSTICS_CONTRACT_MAP
            .contains(&BlueBrainThalamusContractClass::ThalamusCaveatedDiagnostic));
        assert!(CANONICAL_BLUE_BRAIN_THALAMUS_DIAGNOSTICS_CONTRACT_MAP
            .contains(&BlueBrainThalamusContractClass::ThalamusDeferredDiagnostic));
        assert!(CANONICAL_BLUE_BRAIN_THALAMUS_DIAGNOSTICS_CONTRACT_MAP
            .contains(&BlueBrainThalamusContractClass::ThalamusBlockedDiagnostic));
        assert!(CANONICAL_BLUE_BRAIN_THALAMUS_DIAGNOSTICS_CONTRACT_MAP
            .contains(&BlueBrainThalamusContractClass::ThalamusInsufficientDiagnostic));
        assert!(CANONICAL_BLUE_BRAIN_THALAMUS_DIAGNOSTICS_CONTRACT_MAP
            .contains(&BlueBrainThalamusContractClass::ThalamusDiagnosticOnlyState));
        assert!(CANONICAL_BLUE_BRAIN_THALAMUS_DIAGNOSTICS_CONTRACT_MAP
            .contains(&BlueBrainThalamusContractClass::ThalamusBoundedContractSignal));
        assert!(CANONICAL_BLUE_BRAIN_THALAMUS_DIAGNOSTICS_CONTRACT_MAP
            .contains(&BlueBrainThalamusContractClass::NonCanonicalInternalOnlyThalamusPath));
        assert_ne!(
            BlueBrainThalamusCanonicalRead::AdvisoryOnly,
            BlueBrainThalamusCanonicalRead::Caveated
        );
        assert_ne!(
            BlueBrainThalamusCanonicalRead::Deferred,
            BlueBrainThalamusCanonicalRead::Blocked
        );
        assert_ne!(
            BlueBrainThalamusCanonicalRead::Blocked,
            BlueBrainThalamusCanonicalRead::Insufficient
        );
    }

    #[test]
    fn thalamus_br3_prompt3_runtime_selection_routing_reference_reads_are_aligned() {
        let cases = [
            (
                BlueBrainThalamusInputSurface {
                    selection_signal: BlueBrainControlAttentionSelectionClass::AttentionTarget,
                    deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                    reference_validity: BlueBrainReferenceValidity::Current,
                    context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
                },
                BlueBrainThalamusCanonicalRead::AdvisoryOnly,
            ),
            (
                BlueBrainThalamusInputSurface {
                    selection_signal:
                        BlueBrainControlAttentionSelectionClass::EvidenceReferenceSelection,
                    deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                    reference_validity: BlueBrainReferenceValidity::Caveated,
                    context_priority:
                        BlueBrainContextEvidencePriorityClass::CaveatedEvidenceReference,
                },
                BlueBrainThalamusCanonicalRead::Caveated,
            ),
            (
                BlueBrainThalamusInputSurface {
                    selection_signal:
                        BlueBrainControlAttentionSelectionClass::EvidenceReferenceSelection,
                    deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateDeferred,
                    reference_validity: BlueBrainReferenceValidity::Current,
                    context_priority:
                        BlueBrainContextEvidencePriorityClass::SupportingEvidenceReference,
                },
                BlueBrainThalamusCanonicalRead::Deferred,
            ),
            (
                BlueBrainThalamusInputSurface {
                    selection_signal:
                        BlueBrainControlAttentionSelectionClass::EvidenceReferenceSelection,
                    deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateRejected,
                    reference_validity: BlueBrainReferenceValidity::Current,
                    context_priority:
                        BlueBrainContextEvidencePriorityClass::SupportingEvidenceReference,
                },
                BlueBrainThalamusCanonicalRead::Blocked,
            ),
            (
                BlueBrainThalamusInputSurface {
                    selection_signal:
                        BlueBrainControlAttentionSelectionClass::EvidenceReferenceSelection,
                    deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateInsufficient,
                    reference_validity: BlueBrainReferenceValidity::Current,
                    context_priority:
                        BlueBrainContextEvidencePriorityClass::SupportingEvidenceReference,
                },
                BlueBrainThalamusCanonicalRead::Insufficient,
            ),
            (
                BlueBrainThalamusInputSurface {
                    selection_signal:
                        BlueBrainControlAttentionSelectionClass::EvidenceReferenceSelection,
                    deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                    reference_validity: BlueBrainReferenceValidity::ReferenceOnly,
                    context_priority:
                        BlueBrainContextEvidencePriorityClass::SupportingEvidenceReference,
                },
                BlueBrainThalamusCanonicalRead::DiagnosticOnly,
            ),
        ];

        for (input, expected) in cases {
            let (state, output) = evaluate_blue_brain_thalamus_relay_routing(input);
            assert_eq!(
                blue_brain_thalamus_canonical_read_for_state(state),
                expected
            );
            assert_eq!(output.canonical_contract_read, expected);
            assert_eq!(
                blue_brain_thalamus_consumer_contract_read(
                    output,
                    BlueBrainThalamusConsumerLayer::Runtime
                ),
                expected
            );
            assert_eq!(
                blue_brain_thalamus_consumer_contract_read(
                    output,
                    BlueBrainThalamusConsumerLayer::Selection
                ),
                expected
            );
            assert_eq!(
                blue_brain_thalamus_consumer_contract_read(
                    output,
                    BlueBrainThalamusConsumerLayer::Routing
                ),
                expected
            );
            assert_eq!(
                blue_brain_thalamus_consumer_contract_read(
                    output,
                    BlueBrainThalamusConsumerLayer::Reference
                ),
                expected
            );
            assert!(!output.direct_action_selection);
            assert!(!output.direct_execution_trigger);
            assert!(!output.direct_retry_trigger);
            assert!(!output.direct_memory_commit);
            assert!(!output.direct_compute_invocation);
            assert!(!output.safety_override);
        }
    }

    #[test]
    fn thalamus_br3_prompt3_insufficient_is_not_caveated_deferred_or_blocked() {
        let input = BlueBrainThalamusInputSurface {
            selection_signal: BlueBrainControlAttentionSelectionClass::EvidenceReferenceSelection,
            deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateInsufficient,
            reference_validity: BlueBrainReferenceValidity::Current,
            context_priority: BlueBrainContextEvidencePriorityClass::SupportingEvidenceReference,
        };
        let (state, output) = evaluate_blue_brain_thalamus_relay_routing(input);
        assert_eq!(
            state,
            BlueBrainThalamusStateSurface::InsufficientRoutingState
        );
        assert_eq!(
            output.runtime_diagnostic_state,
            BlueBrainThalamusDiagnosticState::ThalamusInsufficientDiagnostic
        );
        assert_eq!(
            output.advisory_class,
            BlueBrainThalamusAdvisoryOutputClass::InsufficientDiagnosticOutput
        );
        assert_ne!(
            state,
            BlueBrainThalamusStateSurface::CaveatedReferenceRoutingState
        );
        assert_ne!(state, BlueBrainThalamusStateSurface::DeferredRoutingState);
        assert_ne!(state, BlueBrainThalamusStateSurface::BlockedRoutingState);
    }

    #[test]
    fn thalamus_br3_prompt2_integration_map_surfaces_are_complete_and_distinct() {
        assert!(CANONICAL_BLUE_BRAIN_THALAMUS_INTEGRATION_MAP
            .contains(&BlueBrainThalamusIntegrationClass::ThalamusInputSurface));
        assert!(CANONICAL_BLUE_BRAIN_THALAMUS_INTEGRATION_MAP
            .contains(&BlueBrainThalamusIntegrationClass::ThalamusStateSurface));
        assert!(CANONICAL_BLUE_BRAIN_THALAMUS_INTEGRATION_MAP
            .contains(&BlueBrainThalamusIntegrationClass::ThalamusOutputAdvisorySurface));
        assert!(CANONICAL_BLUE_BRAIN_THALAMUS_INTEGRATION_MAP
            .contains(&BlueBrainThalamusIntegrationClass::ThalamusReferenceSurface));
        assert!(CANONICAL_BLUE_BRAIN_THALAMUS_INTEGRATION_MAP
            .contains(&BlueBrainThalamusIntegrationClass::BlockedDeferredThalamusPath));
        assert!(CANONICAL_BLUE_BRAIN_THALAMUS_INTEGRATION_MAP
            .contains(&BlueBrainThalamusIntegrationClass::NonCanonicalInternalOnlyThalamusPath));
        assert_ne!(
            BlueBrainThalamusIntegrationClass::ThalamusInputSurface,
            BlueBrainThalamusIntegrationClass::ThalamusOutputAdvisorySurface
        );
    }

    #[test]
    fn thalamus_br3_prompt2_input_guard_blocks_forbidden_authority_inputs() {
        assert_eq!(
            classify_blue_brain_thalamus_input_guard(
                BlueBrainThalamusInputSource::RuntimeRelaySignal
            ),
            BlueBrainThalamusInputGuard::CanonicalBoundedInput
        );
        assert_eq!(
            classify_blue_brain_thalamus_input_guard(
                BlueBrainThalamusInputSource::SelectionGatingSignal
            ),
            BlueBrainThalamusInputGuard::CanonicalBoundedInput
        );
        assert_eq!(
            classify_blue_brain_thalamus_input_guard(
                BlueBrainThalamusInputSource::ContextReferenceSignal
            ),
            BlueBrainThalamusInputGuard::CanonicalBoundedInput
        );
        assert_eq!(
            classify_blue_brain_thalamus_input_guard(
                BlueBrainThalamusInputSource::ToolActionControlSignal
            ),
            BlueBrainThalamusInputGuard::RejectedToolActionControl
        );
        assert_eq!(
            classify_blue_brain_thalamus_input_guard(
                BlueBrainThalamusInputSource::ComputeInternalRawStateSignal
            ),
            BlueBrainThalamusInputGuard::RejectedComputeInternalRawState
        );
        assert_eq!(
            classify_blue_brain_thalamus_input_guard(
                BlueBrainThalamusInputSource::SafetyOverrideSignal
            ),
            BlueBrainThalamusInputGuard::RejectedSafetyOverride
        );
        assert_eq!(
            classify_blue_brain_thalamus_input_guard(
                BlueBrainThalamusInputSource::ImplicitMemoryMutationSignal
            ),
            BlueBrainThalamusInputGuard::RejectedImplicitMemoryMutation
        );
    }

    #[test]
    fn thalamus_br3_prompt2_runtime_selection_routing_reference_outputs_stay_bounded() {
        let input = BlueBrainThalamusInputSurface {
            selection_signal: BlueBrainControlAttentionSelectionClass::AttentionTarget,
            deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
            reference_validity: BlueBrainReferenceValidity::Current,
            context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
        };
        let (state, output) = evaluate_blue_brain_thalamus_relay_routing(input);

        assert_eq!(
            state,
            BlueBrainThalamusStateSurface::ActiveBoundedRelayAdvisoryOnly
        );
        assert_eq!(
            output.advisory_class,
            BlueBrainThalamusAdvisoryOutputClass::GatingHint
        );
        assert!(output.runtime_advisory_only);
        assert!(output.selection_advisory_only);
        assert!(output.routing_advisory_only);
        assert!(output.reference_bounded_only);
        assert!(!output.direct_action_selection);
        assert!(!output.direct_execution_trigger);
        assert!(!output.direct_retry_trigger);
        assert!(!output.direct_memory_commit);
        assert!(!output.direct_compute_invocation);
        assert!(!output.safety_override);
        assert_eq!(
            output.runtime_contract_signal,
            BlueBrainThalamusContractSignal::ThalamusToRuntimeAdvisory
        );
        assert_eq!(
            output.selection_contract_signal,
            BlueBrainThalamusContractSignal::ThalamusToSelectionAdvisory
        );
        assert_eq!(
            output.reference_contract_signal,
            BlueBrainThalamusContractSignal::ThalamusReferenceSignal
        );
    }

    #[test]
    fn thalamus_br3_prompt2_reference_deferred_blocked_and_noncanonical_paths_do_not_escalate() {
        let reference_only = BlueBrainThalamusInputSurface {
            selection_signal: BlueBrainControlAttentionSelectionClass::EvidenceReferenceSelection,
            deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
            reference_validity: BlueBrainReferenceValidity::ReferenceOnly,
            context_priority: BlueBrainContextEvidencePriorityClass::SupportingEvidenceReference,
        };
        let (state, output) = evaluate_blue_brain_thalamus_relay_routing(reference_only);
        assert_eq!(
            state,
            BlueBrainThalamusStateSurface::ReferenceOnlyRoutingState
        );
        assert_eq!(
            output.reference_diagnostic_state,
            BlueBrainThalamusDiagnosticState::ThalamusDiagnosticOnlyState
        );
        assert!(!output.direct_memory_commit);

        let deferred = BlueBrainThalamusInputSurface {
            deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateDeferred,
            reference_validity: BlueBrainReferenceValidity::Current,
            ..reference_only
        };
        let (state, output) = evaluate_blue_brain_thalamus_relay_routing(deferred);
        assert_eq!(state, BlueBrainThalamusStateSurface::DeferredRoutingState);
        assert_eq!(
            output.runtime_diagnostic_state,
            BlueBrainThalamusDiagnosticState::ThalamusDeferredDiagnostic
        );
        assert!(!output.direct_retry_trigger);

        let blocked = BlueBrainThalamusInputSurface {
            deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateRejected,
            reference_validity: BlueBrainReferenceValidity::Current,
            ..reference_only
        };
        let (state, output) = evaluate_blue_brain_thalamus_relay_routing(blocked);
        assert_eq!(state, BlueBrainThalamusStateSurface::BlockedRoutingState);
        assert_eq!(
            output.routing_diagnostic_state,
            BlueBrainThalamusDiagnosticState::ThalamusBlockedDiagnostic
        );
        assert!(!output.direct_execution_trigger);

        let noncanonical = BlueBrainThalamusInputSurface {
            selection_signal:
                BlueBrainControlAttentionSelectionClass::NonCanonicalInternalOnlySelectionPath,
            deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
            reference_validity: BlueBrainReferenceValidity::Current,
            context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
        };
        let (state, output) = evaluate_blue_brain_thalamus_relay_routing(noncanonical);
        assert_eq!(
            state,
            BlueBrainThalamusStateSurface::NonCanonicalInternalOnly
        );
        assert_eq!(
            blue_brain_thalamus_contract_class_for_signal(output.runtime_contract_signal),
            BlueBrainThalamusContractClass::NonCanonicalInternalOnlyThalamusPath
        );
    }

    #[test]
    fn thalamus_br3_prompt2_doc_pins_surfaces_boundaries_and_region_separation() {
        let doc = include_str!(
            "../../../docs/blue_brain_thalamus_minimal_bounded_integration_serie_br3_prompt2_v1.md"
        );
        assert!(doc.contains("thalamus input surface"));
        assert!(doc.contains("thalamus state surface"));
        assert!(doc.contains("thalamus output/advisory surface"));
        assert!(doc.contains("thalamus reference surface"));
        assert!(doc.contains("blocked/deferred thalamus path"));
        assert!(doc.contains("non-canonical/internal-only thalamus path"));
        assert!(doc.contains("relay-hint"));
        assert!(doc.contains("routing-hint"));
        assert!(doc.contains("gating-hint"));
        assert!(doc.contains("reference-bounded signal"));
        assert!(doc.contains("Runtime sieht den Thalamus ausschließlich als advisory"));
        assert!(doc.contains("Selection sieht ihn ausschließlich als gating-/selection-support"));
        assert!(doc.contains("keine zweite Referenzwirklichkeit"));
        assert!(doc.contains("keine implizite Memory-Persistenz"));
        assert!(doc.contains("abstract functional current mode"));
        assert!(doc.contains("keine Kuramoto-Produktivaufweitung"));
        assert!(doc.contains("keine Hodgkin-Huxley-Produktivintegration"));
        assert!(doc.contains("hippocampus_like_region"));
        assert!(doc.contains("amygdala_like_region"));
        assert!(doc.contains("thalamus_like_region"));
        assert!(doc.contains("no direct action trigger"));
        assert!(doc.contains("no direct execution trigger"));
        assert!(doc.contains("no direct retry trigger"));
        assert!(doc.contains("no direct memory commit"));
        assert!(doc.contains("no direct compute invocation"));
        assert!(doc.contains("no safety override"));
        assert!(doc.contains("keine implizite Öffnung weiterer anatomischer Regionen"));
    }

    #[test]
    fn thalamus_br3_prompt3_doc_pins_surface_diagnostics_contracts_and_boundaries() {
        let doc = include_str!(
            "../../../docs/blue_brain_thalamus_surface_diagnostics_contracts_hardening_serie_br3_prompt3_v1.md"
        );
        assert!(doc.contains("thalamus input surface"));
        assert!(doc.contains("thalamus state surface"));
        assert!(doc.contains("thalamus output/advisory surface"));
        assert!(doc.contains("thalamus reference surface"));
        assert!(doc.contains("thalamus advisory-only diagnostic"));
        assert!(doc.contains("thalamus caveated diagnostic"));
        assert!(doc.contains("thalamus deferred diagnostic"));
        assert!(doc.contains("thalamus blocked diagnostic"));
        assert!(doc.contains("thalamus insufficient diagnostic"));
        assert!(doc.contains("thalamus diagnostic-only state"));
        assert!(doc.contains("thalamus bounded contract signal"));
        assert!(doc.contains("non-canonical/internal-only thalamus path"));
        assert!(doc.contains("advisory-only != caveated"));
        assert!(doc.contains("deferred != blocked"));
        assert!(doc.contains("blocked != insufficient"));
        assert!(doc.contains("no action request"));
        assert!(doc.contains("no execution trigger"));
        assert!(doc.contains("no retry trigger"));
        assert!(doc.contains("no memory commit"));
        assert!(doc.contains("no compute trigger"));
        assert!(doc.contains("no safety override"));
        assert!(doc.contains("current model mode remains unchanged"));
        assert!(doc.contains("abstract functional current mode"));
        assert!(doc.contains("Hodgkin-Huxley simulation-only/diagnostic-only remains deferred"));
        assert!(doc.contains("hippocampus remains context/reference/episode/indexing"));
        assert!(doc.contains("amygdala remains salience/valence/caveat/priority"));
        assert!(doc.contains("thalamus remains relay/gating/routing"));
    }

    #[test]
    fn basal_ganglia_br4_prompt2_surfaces_roles_and_guards_are_canonical() {
        assert_eq!(CANONICAL_BLUE_BRAIN_BASAL_GANGLIA_INTEGRATION_MAP.len(), 6);
        assert!(CANONICAL_BLUE_BRAIN_BASAL_GANGLIA_INTEGRATION_MAP
            .contains(&BlueBrainBasalGangliaIntegrationClass::BasalGangliaInputSurface));
        assert!(CANONICAL_BLUE_BRAIN_BASAL_GANGLIA_INTEGRATION_MAP
            .contains(&BlueBrainBasalGangliaIntegrationClass::BasalGangliaStateSurface));
        assert!(CANONICAL_BLUE_BRAIN_BASAL_GANGLIA_INTEGRATION_MAP
            .contains(&BlueBrainBasalGangliaIntegrationClass::BasalGangliaOutputAdvisorySurface));
        assert!(CANONICAL_BLUE_BRAIN_BASAL_GANGLIA_INTEGRATION_MAP
            .contains(&BlueBrainBasalGangliaIntegrationClass::BasalGangliaReferenceBoundedSurface));
        assert!(CANONICAL_BLUE_BRAIN_BASAL_GANGLIA_INTEGRATION_MAP
            .contains(&BlueBrainBasalGangliaIntegrationClass::BlockedDeferredBasalGangliaPath));
        assert!(CANONICAL_BLUE_BRAIN_BASAL_GANGLIA_INTEGRATION_MAP.contains(
            &BlueBrainBasalGangliaIntegrationClass::NonCanonicalInternalOnlyBasalGangliaPath
        ));
        assert_eq!(CANONICAL_BLUE_BRAIN_BASAL_GANGLIA_ROLE_MAP.len(), 5);
        assert!(CANONICAL_BLUE_BRAIN_BASAL_GANGLIA_ROLE_MAP
            .contains(&BlueBrainBasalGangliaRoleClass::ActionGatingRole));
        assert!(CANONICAL_BLUE_BRAIN_BASAL_GANGLIA_ROLE_MAP
            .contains(&BlueBrainBasalGangliaRoleClass::SuppressionInhibitionRole));
        assert!(CANONICAL_BLUE_BRAIN_BASAL_GANGLIA_ROLE_MAP
            .contains(&BlueBrainBasalGangliaRoleClass::BoundedSelectionChannelArbitrationRole));
        assert!(CANONICAL_BLUE_BRAIN_BASAL_GANGLIA_ROLE_MAP
            .contains(&BlueBrainBasalGangliaRoleClass::ExecutionReadinessModulationRole));

        assert_eq!(
            classify_blue_brain_basal_ganglia_input_guard(
                BlueBrainBasalGangliaInputSource::RuntimeReadinessSignal
            ),
            BlueBrainBasalGangliaInputGuard::AdvisoryOnlyInput
        );
        assert_eq!(
            classify_blue_brain_basal_ganglia_input_guard(
                BlueBrainBasalGangliaInputSource::ReferenceValiditySignal
            ),
            BlueBrainBasalGangliaInputGuard::ReferenceOnlyBoundedInput
        );
        assert_eq!(
            classify_blue_brain_basal_ganglia_input_guard(
                BlueBrainBasalGangliaInputSource::ToolActionControlSignal
            ),
            BlueBrainBasalGangliaInputGuard::RejectedToolActionControl
        );
        assert_eq!(
            classify_blue_brain_basal_ganglia_input_guard(
                BlueBrainBasalGangliaInputSource::ComputeInternalRawStateSignal
            ),
            BlueBrainBasalGangliaInputGuard::RejectedComputeInternalRawState
        );
        assert_eq!(
            classify_blue_brain_basal_ganglia_input_guard(
                BlueBrainBasalGangliaInputSource::SafetyOverrideSignal
            ),
            BlueBrainBasalGangliaInputGuard::RejectedSafetyOverride
        );
        assert_eq!(
            classify_blue_brain_basal_ganglia_input_guard(
                BlueBrainBasalGangliaInputSource::ImplicitMemoryMutationSignal
            ),
            BlueBrainBasalGangliaInputGuard::RejectedImplicitMemoryMutation
        );
    }

    #[test]
    fn basal_ganglia_br4_prompt2_action_gating_outputs_remain_advisory_only() {
        let input = BlueBrainBasalGangliaInputSurface {
            selection_signal: BlueBrainControlAttentionSelectionClass::AttentionTarget,
            deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
            reference_validity: BlueBrainReferenceValidity::Current,
            context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
        };
        let (state, output) = evaluate_blue_brain_basal_ganglia_action_gating(input);

        assert_eq!(
            state,
            BlueBrainBasalGangliaStateSurface::ActiveBoundedActionGatingAdvisoryOnly
        );
        assert_eq!(
            output.advisory_class,
            BlueBrainBasalGangliaAdvisoryOutputClass::GatingHint
        );
        assert_eq!(
            blue_brain_basal_ganglia_consumer_contract_read(
                output,
                BlueBrainBasalGangliaConsumerLayer::Selection
            ),
            BlueBrainBasalGangliaCanonicalRead::AdvisoryOnly
        );
        assert!(output.runtime_advisory_only);
        assert!(output.selection_advisory_only);
        assert!(output.execution_readiness_caveat_only);
        assert!(output.reference_bounded_only);
        assert!(!output.direct_action_selection);
        assert!(!output.direct_action_trigger);
        assert!(!output.direct_execution_trigger);
        assert!(!output.direct_retry_trigger);
        assert!(!output.direct_memory_commit);
        assert!(!output.direct_compute_invocation);
        assert!(!output.safety_override);
        assert_eq!(
            output.runtime_contract_signal,
            BlueBrainBasalGangliaContractSignal::BasalGangliaToRuntimeAdvisory
        );
        assert_eq!(
            output.selection_contract_signal,
            BlueBrainBasalGangliaContractSignal::BasalGangliaToSelectionAdvisory
        );
    }

    #[test]
    fn basal_ganglia_br4_prompt2_reference_deferred_blocked_and_noncanonical_do_not_escalate() {
        let reference_only = BlueBrainBasalGangliaInputSurface {
            selection_signal: BlueBrainControlAttentionSelectionClass::EvidenceReferenceSelection,
            deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
            reference_validity: BlueBrainReferenceValidity::ReferenceOnly,
            context_priority: BlueBrainContextEvidencePriorityClass::SupportingEvidenceReference,
        };
        let (state, output) = evaluate_blue_brain_basal_ganglia_action_gating(reference_only);
        assert_eq!(
            state,
            BlueBrainBasalGangliaStateSurface::ReferenceOnlyActionGatingState
        );
        assert_eq!(
            output.reference_diagnostic_state,
            BlueBrainBasalGangliaDiagnosticState::BasalGangliaDiagnosticOnlyState
        );
        assert!(!output.direct_memory_commit);

        let deferred = BlueBrainBasalGangliaInputSurface {
            deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateDeferred,
            reference_validity: BlueBrainReferenceValidity::Current,
            ..reference_only
        };
        let (state, output) = evaluate_blue_brain_basal_ganglia_action_gating(deferred);
        assert_eq!(
            state,
            BlueBrainBasalGangliaStateSurface::DeferredActionGatingState
        );
        assert_eq!(
            output.runtime_diagnostic_state,
            BlueBrainBasalGangliaDiagnosticState::BasalGangliaDeferredDiagnostic
        );
        assert!(!output.direct_retry_trigger);

        let blocked = BlueBrainBasalGangliaInputSurface {
            deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateRejected,
            reference_validity: BlueBrainReferenceValidity::Current,
            ..reference_only
        };
        let (state, output) = evaluate_blue_brain_basal_ganglia_action_gating(blocked);
        assert_eq!(
            state,
            BlueBrainBasalGangliaStateSurface::BlockedActionGatingState
        );
        assert_eq!(
            output.execution_diagnostic_state,
            BlueBrainBasalGangliaDiagnosticState::BasalGangliaBlockedDiagnostic
        );
        assert!(!output.direct_execution_trigger);

        let noncanonical = BlueBrainBasalGangliaInputSurface {
            selection_signal:
                BlueBrainControlAttentionSelectionClass::NonCanonicalInternalOnlySelectionPath,
            deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
            reference_validity: BlueBrainReferenceValidity::Current,
            context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
        };
        let (state, output) = evaluate_blue_brain_basal_ganglia_action_gating(noncanonical);
        assert_eq!(
            state,
            BlueBrainBasalGangliaStateSurface::NonCanonicalInternalOnly
        );
        assert_eq!(
            blue_brain_basal_ganglia_contract_class_for_signal(output.runtime_contract_signal),
            BlueBrainBasalGangliaContractClass::NonCanonicalInternalOnlyBasalGangliaPath
        );
    }

    #[test]
    fn basal_ganglia_br4_prompt2_doc_pins_surfaces_boundaries_and_region_separation() {
        let doc = include_str!(
            "../../../docs/blue_brain_basal_ganglia_minimal_bounded_integration_serie_br4_prompt2_v1.md"
        );
        assert!(doc.contains("basal-ganglia input surface"));
        assert!(doc.contains("basal-ganglia state surface"));
        assert!(doc.contains("basal-ganglia output/advisory surface"));
        assert!(doc.contains("basal-ganglia reference surface"));
        assert!(doc.contains("blocked/deferred basal-ganglia path"));
        assert!(doc.contains("non-canonical/internal-only basal-ganglia path"));
        assert!(doc.contains("gating-hint"));
        assert!(doc.contains("suppression-hint"));
        assert!(doc.contains("channel-selection hint"));
        assert!(doc.contains("execution-readiness caveat"));
        assert!(doc.contains("reference-bounded signal"));
        assert!(doc.contains(
            "Runtime sieht Basal Ganglia ausschließlich als bounded diagnostic/advisory"
        ));
        assert!(doc.contains("Selection sieht Basal Ganglia ausschließlich als advisory"));
        assert!(doc.contains("keine zweite Referenzwirklichkeit"));
        assert!(doc.contains("keine implizite Memory-Persistenz"));
        assert!(doc.contains("abstract functional current mode"));
        assert!(doc.contains("keine Kuramoto-Produktivaufweitung"));
        assert!(doc.contains("keine Hodgkin-Huxley-Produktivintegration"));
        assert!(doc.contains("hippocampus_like_region"));
        assert!(doc.contains("amygdala_like_region"));
        assert!(doc.contains("thalamus_like_region"));
        assert!(doc.contains("basal_ganglia_like_region"));
        assert!(doc.contains("kein direct action trigger"));
        assert!(doc.contains("kein direct execution trigger"));
        assert!(doc.contains("kein direct retry trigger"));
        assert!(doc.contains("kein direct memory commit"));
        assert!(doc.contains("kein direct compute invocation"));
        assert!(doc.contains("kein safety override"));
        assert!(doc.contains("keine parallele Öffnung weiterer anatomischer Regionen"));
    }

    #[test]
    fn basal_ganglia_br4_prompt3_diagnostics_contract_map_is_canonical_and_distinct() {
        assert_eq!(
            CANONICAL_BLUE_BRAIN_BASAL_GANGLIA_DIAGNOSTICS_CONTRACT_MAP,
            [
                BlueBrainBasalGangliaContractClass::BasalGangliaAdvisoryOnlyDiagnostic,
                BlueBrainBasalGangliaContractClass::BasalGangliaCaveatedDiagnostic,
                BlueBrainBasalGangliaContractClass::BasalGangliaDeferredDiagnostic,
                BlueBrainBasalGangliaContractClass::BasalGangliaBlockedDiagnostic,
                BlueBrainBasalGangliaContractClass::BasalGangliaInsufficientDiagnostic,
                BlueBrainBasalGangliaContractClass::BasalGangliaDiagnosticOnlyState,
                BlueBrainBasalGangliaContractClass::BasalGangliaBoundedContractSignal,
                BlueBrainBasalGangliaContractClass::NonCanonicalInternalOnlyBasalGangliaPath,
            ]
        );
        assert_ne!(
            BlueBrainBasalGangliaDiagnosticState::BasalGangliaAdvisoryOnlyDiagnostic,
            BlueBrainBasalGangliaDiagnosticState::BasalGangliaCaveatedDiagnostic
        );
        assert_ne!(
            BlueBrainBasalGangliaDiagnosticState::BasalGangliaDeferredDiagnostic,
            BlueBrainBasalGangliaDiagnosticState::BasalGangliaBlockedDiagnostic
        );
        assert_ne!(
            BlueBrainBasalGangliaDiagnosticState::BasalGangliaBlockedDiagnostic,
            BlueBrainBasalGangliaDiagnosticState::BasalGangliaInsufficientDiagnostic
        );
        assert_eq!(
            blue_brain_basal_ganglia_contract_class_for_signal(
                BlueBrainBasalGangliaContractSignal::RuntimeToBasalGangliaBoundedReadinessInput
            ),
            BlueBrainBasalGangliaContractClass::BasalGangliaBoundedContractSignal
        );
        assert_eq!(
            blue_brain_basal_ganglia_contract_class_for_signal(
                BlueBrainBasalGangliaContractSignal::ReferenceOnly
            ),
            BlueBrainBasalGangliaContractClass::BasalGangliaDiagnosticOnlyState
        );
    }

    #[test]
    fn basal_ganglia_br4_prompt3_runtime_selection_reference_share_same_canonical_read() {
        let cases = [
            (
                BlueBrainBasalGangliaInputSurface {
                    selection_signal: BlueBrainControlAttentionSelectionClass::AttentionTarget,
                    deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                    reference_validity: BlueBrainReferenceValidity::Current,
                    context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
                },
                BlueBrainBasalGangliaCanonicalRead::AdvisoryOnly,
            ),
            (
                BlueBrainBasalGangliaInputSurface {
                    selection_signal: BlueBrainControlAttentionSelectionClass::ContextSelection,
                    deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                    reference_validity: BlueBrainReferenceValidity::Caveated,
                    context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
                },
                BlueBrainBasalGangliaCanonicalRead::Caveated,
            ),
            (
                BlueBrainBasalGangliaInputSurface {
                    selection_signal: BlueBrainControlAttentionSelectionClass::ContextSelection,
                    deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateDeferred,
                    reference_validity: BlueBrainReferenceValidity::Current,
                    context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
                },
                BlueBrainBasalGangliaCanonicalRead::Deferred,
            ),
            (
                BlueBrainBasalGangliaInputSurface {
                    selection_signal: BlueBrainControlAttentionSelectionClass::ContextSelection,
                    deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateRejected,
                    reference_validity: BlueBrainReferenceValidity::Current,
                    context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
                },
                BlueBrainBasalGangliaCanonicalRead::Blocked,
            ),
            (
                BlueBrainBasalGangliaInputSurface {
                    selection_signal: BlueBrainControlAttentionSelectionClass::ContextSelection,
                    deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateInsufficient,
                    reference_validity: BlueBrainReferenceValidity::Current,
                    context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
                },
                BlueBrainBasalGangliaCanonicalRead::Insufficient,
            ),
            (
                BlueBrainBasalGangliaInputSurface {
                    selection_signal: BlueBrainControlAttentionSelectionClass::ContextSelection,
                    deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                    reference_validity: BlueBrainReferenceValidity::ReferenceOnly,
                    context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
                },
                BlueBrainBasalGangliaCanonicalRead::DiagnosticOnly,
            ),
        ];

        for (input, expected_read) in cases {
            let (_, output) = evaluate_blue_brain_basal_ganglia_action_gating(input);
            assert_eq!(output.canonical_contract_read, expected_read);
            assert_eq!(
                blue_brain_basal_ganglia_consumer_contract_read(
                    output,
                    BlueBrainBasalGangliaConsumerLayer::Runtime
                ),
                expected_read
            );
            assert_eq!(
                blue_brain_basal_ganglia_consumer_contract_read(
                    output,
                    BlueBrainBasalGangliaConsumerLayer::Selection
                ),
                expected_read
            );
            assert_eq!(
                blue_brain_basal_ganglia_consumer_contract_read(
                    output,
                    BlueBrainBasalGangliaConsumerLayer::Reference
                ),
                expected_read
            );
            assert!(output.runtime_advisory_only);
            assert!(output.selection_advisory_only);
            assert!(output.reference_bounded_only);
            assert!(!output.direct_action_selection);
            assert!(!output.direct_action_trigger);
            assert!(!output.direct_execution_trigger);
            assert!(!output.direct_retry_trigger);
            assert!(!output.direct_memory_commit);
            assert!(!output.direct_compute_invocation);
            assert!(!output.safety_override);
        }
    }

    #[test]
    fn basal_ganglia_br4_prompt4_closeout_keeps_status_classes_and_guards_separate() {
        let cases = [
            (
                BlueBrainBasalGangliaInputSurface {
                    selection_signal: BlueBrainControlAttentionSelectionClass::ContextSelection,
                    deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                    reference_validity: BlueBrainReferenceValidity::Current,
                    context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
                },
                BlueBrainBasalGangliaStateSurface::ActiveBoundedActionGatingAdvisoryOnly,
                BlueBrainBasalGangliaCanonicalRead::AdvisoryOnly,
            ),
            (
                BlueBrainBasalGangliaInputSurface {
                    selection_signal: BlueBrainControlAttentionSelectionClass::ContextSelection,
                    deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                    reference_validity: BlueBrainReferenceValidity::Caveated,
                    context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
                },
                BlueBrainBasalGangliaStateSurface::ExecutionReadinessCaveatState,
                BlueBrainBasalGangliaCanonicalRead::Caveated,
            ),
            (
                BlueBrainBasalGangliaInputSurface {
                    selection_signal: BlueBrainControlAttentionSelectionClass::ContextSelection,
                    deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateDeferred,
                    reference_validity: BlueBrainReferenceValidity::Current,
                    context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
                },
                BlueBrainBasalGangliaStateSurface::DeferredActionGatingState,
                BlueBrainBasalGangliaCanonicalRead::Deferred,
            ),
            (
                BlueBrainBasalGangliaInputSurface {
                    selection_signal: BlueBrainControlAttentionSelectionClass::ContextSelection,
                    deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateRejected,
                    reference_validity: BlueBrainReferenceValidity::Current,
                    context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
                },
                BlueBrainBasalGangliaStateSurface::BlockedActionGatingState,
                BlueBrainBasalGangliaCanonicalRead::Blocked,
            ),
            (
                BlueBrainBasalGangliaInputSurface {
                    selection_signal: BlueBrainControlAttentionSelectionClass::ContextSelection,
                    deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateInsufficient,
                    reference_validity: BlueBrainReferenceValidity::Current,
                    context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
                },
                BlueBrainBasalGangliaStateSurface::InsufficientActionGatingState,
                BlueBrainBasalGangliaCanonicalRead::Insufficient,
            ),
            (
                BlueBrainBasalGangliaInputSurface {
                    selection_signal: BlueBrainControlAttentionSelectionClass::ContextSelection,
                    deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                    reference_validity: BlueBrainReferenceValidity::ReferenceOnly,
                    context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
                },
                BlueBrainBasalGangliaStateSurface::ReferenceOnlyActionGatingState,
                BlueBrainBasalGangliaCanonicalRead::DiagnosticOnly,
            ),
        ];

        for (input, expected_state, expected_read) in cases {
            let (state, output) = evaluate_blue_brain_basal_ganglia_action_gating(input);

            assert_eq!(state, expected_state);
            assert_eq!(
                blue_brain_basal_ganglia_canonical_read_for_state(state),
                expected_read
            );
            assert_eq!(output.canonical_contract_read, expected_read);
            assert_eq!(
                blue_brain_basal_ganglia_consumer_contract_read(
                    output,
                    BlueBrainBasalGangliaConsumerLayer::Runtime
                ),
                expected_read
            );
            assert_eq!(
                blue_brain_basal_ganglia_consumer_contract_read(
                    output,
                    BlueBrainBasalGangliaConsumerLayer::Selection
                ),
                expected_read
            );
            assert_eq!(
                blue_brain_basal_ganglia_consumer_contract_read(
                    output,
                    BlueBrainBasalGangliaConsumerLayer::ExecutionInterface
                ),
                expected_read
            );
            assert_eq!(
                blue_brain_basal_ganglia_consumer_contract_read(
                    output,
                    BlueBrainBasalGangliaConsumerLayer::Reference
                ),
                expected_read
            );
            assert!(output.runtime_advisory_only);
            assert!(output.selection_advisory_only);
            assert!(output.execution_readiness_caveat_only);
            assert!(output.reference_bounded_only);
            assert!(!output.direct_action_selection);
            assert!(!output.direct_action_trigger);
            assert!(!output.direct_execution_trigger);
            assert!(!output.direct_retry_trigger);
            assert!(!output.direct_memory_commit);
            assert!(!output.direct_compute_invocation);
            assert!(!output.safety_override);
        }

        assert_eq!(
            blue_brain_anatomical_region_model_mode(BlueBrainAnatomicalRegionClass::BasalGanglia),
            BlueBrainFirstAnatomicalRegionModelModeClass::AbstractFunctionalCurrentMode
        );
        assert_ne!(
            BlueBrainBasalGangliaDiagnosticState::BasalGangliaAdvisoryOnlyDiagnostic,
            BlueBrainBasalGangliaDiagnosticState::BasalGangliaCaveatedDiagnostic
        );
        assert_ne!(
            BlueBrainBasalGangliaDiagnosticState::BasalGangliaDeferredDiagnostic,
            BlueBrainBasalGangliaDiagnosticState::BasalGangliaBlockedDiagnostic
        );
        assert_ne!(
            BlueBrainBasalGangliaDiagnosticState::BasalGangliaBlockedDiagnostic,
            BlueBrainBasalGangliaDiagnosticState::BasalGangliaInsufficientDiagnostic
        );
    }

    #[test]
    fn basal_ganglia_br4_prompt4_doc_pins_readiness_sweep_and_next_region_boundary() {
        let doc = include_str!(
            "../../../docs/blue_brain_br4_basal_ganglia_readiness_sweep_expansion_boundary_serie_br4_prompt4_v1.md"
        );

        assert!(doc.contains("BR4-expansion-readiness map"));
        assert!(doc.contains("stable basal-ganglia operational surface"));
        assert!(doc.contains("stable current model mode"));
        assert!(doc.contains("basal-ganglia input surface` is not `basal-ganglia state surface"));
        assert!(doc
            .contains("basal-ganglia diagnostics states` are not `basal-ganglia contract signals"));
        assert!(doc.contains("no direct action execution"));
        assert!(doc.contains("no retry orchestration or retry trigger"));
        assert!(doc.contains("no automatic memory persistence, mutation, or commit"));
        assert!(doc.contains("no direct compute invocation"));
        assert!(doc.contains("no fifth anatomical region opened in this step"));
        assert!(doc.contains("final compute line"));
        assert!(doc.contains("maintenance-only core"));
        assert!(
            doc.contains("prioritize Cerebellum as the next single anatomical-region candidate")
        );
        assert!(doc.contains("keep Hypothalamus deferred"));
    }

    #[test]
    fn basal_ganglia_br4_prompt3_doc_pins_contract_hardening_line() {
        let doc = include_str!(
            "../../../docs/blue_brain_basal_ganglia_surface_diagnostics_contracts_hardening_serie_br4_prompt3_v1.md"
        );
        assert!(doc.contains("basal-ganglia advisory-only diagnostic"));
        assert!(doc.contains("basal-ganglia caveated diagnostic"));
        assert!(doc.contains("basal-ganglia deferred diagnostic"));
        assert!(doc.contains("basal-ganglia blocked diagnostic"));
        assert!(doc.contains("basal-ganglia insufficient diagnostic"));
        assert!(doc.contains("basal-ganglia diagnostic-only state"));
        assert!(doc.contains("basal-ganglia bounded contract signal"));
        assert!(doc.contains("non-canonical/internal-only basal-ganglia path"));
        assert!(doc.contains(
            "Runtime, Selection und Reference lesen Basal Ganglia nur über denselben kanonischen bounded contract read"
        ));
        assert!(doc.contains(
            "basal-ganglia advisory-only diagnostic != basal-ganglia caveated diagnostic"
        ));
        assert!(
            doc.contains("basal-ganglia deferred diagnostic != basal-ganglia blocked diagnostic")
        );
        assert!(doc
            .contains("basal-ganglia blocked diagnostic != basal-ganglia insufficient diagnostic"));
        assert!(doc.contains("current model mode remains unchanged"));
        assert!(doc.contains("abstract functional current mode"));
        assert!(doc.contains("bounded Kuramoto-like candidate"));
        assert!(doc.contains("Hodgkin-Huxley simulation-only/diagnostic-only"));
        assert!(doc.contains("hippocampus_like_region"));
        assert!(doc.contains("amygdala_like_region"));
        assert!(doc.contains("thalamus_like_region"));
        assert!(doc.contains("basal_ganglia_like_region"));
        assert!(doc.contains("no action request"));
        assert!(doc.contains("no execution trigger"));
        assert!(doc.contains("no retry trigger"));
        assert!(doc.contains("no memory commit"));
        assert!(doc.contains("no compute trigger"));
        assert!(doc.contains("no safety override"));
    }
    #[test]
    fn inter_region_architecture_map_covers_five_region_pair_classes() {
        assert_eq!(CANONICAL_BLUE_BRAIN_INTER_REGION_ARCHITECTURE_MAP.len(), 10);
        assert_eq!(
            CANONICAL_BLUE_BRAIN_INTER_REGION_ARCHITECTURE_RELATION_CLASS_MAP.len(),
            8
        );
        assert!(
            CANONICAL_BLUE_BRAIN_INTER_REGION_ARCHITECTURE_RELATION_CLASS_MAP.contains(
                &BlueBrainInterRegionArchitectureRelationClass::DirectBoundedAdvisoryRelation
            )
        );
        assert!(
            CANONICAL_BLUE_BRAIN_INTER_REGION_ARCHITECTURE_RELATION_CLASS_MAP.contains(
                &BlueBrainInterRegionArchitectureRelationClass::ReferenceMediatedRelation
            )
        );
        assert!(
            CANONICAL_BLUE_BRAIN_INTER_REGION_ARCHITECTURE_RELATION_CLASS_MAP.contains(
                &BlueBrainInterRegionArchitectureRelationClass::SelectionMediatedRelation
            )
        );
        assert!(
            CANONICAL_BLUE_BRAIN_INTER_REGION_ARCHITECTURE_RELATION_CLASS_MAP.contains(
                &BlueBrainInterRegionArchitectureRelationClass::ExecutionInterfaceMediatedRelation
            )
        );
        assert!(
            CANONICAL_BLUE_BRAIN_INTER_REGION_ARCHITECTURE_RELATION_CLASS_MAP.contains(
                &BlueBrainInterRegionArchitectureRelationClass::CaveatedInterRegionRelation
            )
        );
        assert!(
            CANONICAL_BLUE_BRAIN_INTER_REGION_ARCHITECTURE_RELATION_CLASS_MAP.contains(
                &BlueBrainInterRegionArchitectureRelationClass::DeferredNotYetActiveRelation
            )
        );
        assert!(
            CANONICAL_BLUE_BRAIN_INTER_REGION_ARCHITECTURE_RELATION_CLASS_MAP
                .contains(&BlueBrainInterRegionArchitectureRelationClass::BlockedRelation)
        );
        assert!(CANONICAL_BLUE_BRAIN_INTER_REGION_ARCHITECTURE_RELATION_CLASS_MAP.contains(
            &BlueBrainInterRegionArchitectureRelationClass::NonCanonicalInternalOnlyRelationPath
        ));

        assert_eq!(
            blue_brain_inter_region_architecture_relation_for_pair(
                BlueBrainInterRegionArchitecturePair::HippocampusAmygdala
            )
            .relation_class,
            BlueBrainInterRegionArchitectureRelationClass::CaveatedInterRegionRelation
        );
        assert_eq!(
            blue_brain_inter_region_architecture_relation_for_pair(
                BlueBrainInterRegionArchitecturePair::HippocampusThalamus
            )
            .relation_class,
            BlueBrainInterRegionArchitectureRelationClass::ReferenceMediatedRelation
        );
        assert_eq!(
            blue_brain_inter_region_architecture_relation_for_pair(
                BlueBrainInterRegionArchitecturePair::HippocampusBasalGanglia
            )
            .relation_class,
            BlueBrainInterRegionArchitectureRelationClass::BlockedRelation
        );
        assert_eq!(
            blue_brain_inter_region_architecture_relation_for_pair(
                BlueBrainInterRegionArchitecturePair::AmygdalaCerebellum
            )
            .relation_class,
            BlueBrainInterRegionArchitectureRelationClass::DeferredNotYetActiveRelation
        );
        assert_eq!(
            blue_brain_inter_region_architecture_relation_for_pair(
                BlueBrainInterRegionArchitecturePair::BasalGangliaCerebellum
            )
            .relation_class,
            BlueBrainInterRegionArchitectureRelationClass::ExecutionInterfaceMediatedRelation
        );
    }

    #[test]
    fn inter_region_architecture_roles_remain_functionally_separated() {
        assert_eq!(
            blue_brain_inter_region_architecture_region_role(
                BlueBrainAnatomicalRegionClass::Hippocampus
            ),
            BlueBrainInterRegionArchitectureRegionRoleClass::ContextReferenceEpisodeIndexing
        );
        assert_eq!(
            blue_brain_inter_region_architecture_region_role(
                BlueBrainAnatomicalRegionClass::Amygdala
            ),
            BlueBrainInterRegionArchitectureRegionRoleClass::SaliencePriorityCaveat
        );
        assert_eq!(
            blue_brain_inter_region_architecture_region_role(
                BlueBrainAnatomicalRegionClass::Thalamus
            ),
            BlueBrainInterRegionArchitectureRegionRoleClass::RelayGatingRouting
        );
        assert_eq!(
            blue_brain_inter_region_architecture_region_role(
                BlueBrainAnatomicalRegionClass::BasalGanglia
            ),
            BlueBrainInterRegionArchitectureRegionRoleClass::ActionChannelSuppression
        );
        assert_eq!(
            blue_brain_inter_region_architecture_region_role(
                BlueBrainAnatomicalRegionClass::Cerebellum
            ),
            BlueBrainInterRegionArchitectureRegionRoleClass::TimingPredictionCorrection
        );
    }

    #[test]
    fn inter_region_architecture_map_cannot_create_direct_authority() {
        for relation in CANONICAL_BLUE_BRAIN_INTER_REGION_ARCHITECTURE_MAP {
            assert!(relation.advisory_only);
            assert!(!relation.direct_action_trigger);
            assert!(!relation.direct_execution_trigger);
            assert!(!relation.direct_retry_trigger);
            assert!(!relation.direct_memory_commit);
            assert!(!relation.direct_compute_invocation);
            assert!(!relation.safety_override);
            assert!(!relation.global_region_orchestration);
        }

        for allowed in [
            BlueBrainInterRegionArchitectureOutputClass::BoundedAdvisoryRead,
            BlueBrainInterRegionArchitectureOutputClass::ReferenceContextRead,
            BlueBrainInterRegionArchitectureOutputClass::SelectionContractRead,
            BlueBrainInterRegionArchitectureOutputClass::ExecutionInterfaceDiagnosticRead,
            BlueBrainInterRegionArchitectureOutputClass::CaveatDiagnosticRead,
            BlueBrainInterRegionArchitectureOutputClass::DeferredDiagnosticRead,
            BlueBrainInterRegionArchitectureOutputClass::BlockedDiagnosticRead,
        ] {
            assert_eq!(
                classify_blue_brain_inter_region_architecture_output_guard(allowed),
                BlueBrainInterRegionArchitectureOutputGuard::AllowedBoundedRead
            );
        }

        for blocked in [
            BlueBrainInterRegionArchitectureOutputClass::DirectActionTrigger,
            BlueBrainInterRegionArchitectureOutputClass::DirectExecutionTrigger,
            BlueBrainInterRegionArchitectureOutputClass::DirectRetryTrigger,
            BlueBrainInterRegionArchitectureOutputClass::DirectMemoryCommit,
            BlueBrainInterRegionArchitectureOutputClass::DirectComputeInvocation,
            BlueBrainInterRegionArchitectureOutputClass::SafetyOverride,
            BlueBrainInterRegionArchitectureOutputClass::GlobalRegionOrchestration,
        ] {
            assert_eq!(
                classify_blue_brain_inter_region_architecture_output_guard(blocked),
                BlueBrainInterRegionArchitectureOutputGuard::BlockedForbiddenAuthorityPath
            );
        }
    }

    #[test]
    fn inter_region_architecture_doc_pins_scope_and_mediation_boundaries() {
        let doc = include_str!(
            "../../../docs/blue_brain_inter_region_architecture_serie_ir1_prompt1_v1.md"
        );
        assert!(doc.contains("direct bounded advisory relation"));
        assert!(doc.contains("reference-mediated relation"));
        assert!(doc.contains("selection-mediated relation"));
        assert!(doc.contains("execution-interface-mediated relation"));
        assert!(doc.contains("caveated inter-region relation"));
        assert!(doc.contains("deferred/not-yet-active relation"));
        assert!(doc.contains("blocked relation"));
        assert!(doc.contains("non-canonical/internal-only relation path"));
        assert!(doc.contains("Hippocampus ↔ Amygdala"));
        assert!(doc.contains("Hippocampus ↔ Basal Ganglia"));
        assert!(doc.contains("Basal Ganglia ↔ Cerebellum"));
        assert!(doc.contains("advisory-only relation is not strong authority"));
        assert!(doc.contains("caveated relation is not stable relation"));
        assert!(doc.contains("deferred relation is not blocked relation"));
        assert!(doc.contains("blocked relation is not failed execution"));
        assert!(doc.contains("reference-mediated relation is not direct inter-region authority"));
        assert!(doc.contains("no direct action trigger"));
        assert!(doc.contains("no direct execution trigger"));
        assert!(doc.contains("no direct retry trigger"));
        assert!(doc.contains("no retry orchestration"));
        assert!(doc.contains("no direct memory commit"));
        assert!(doc.contains("no automatic memory persistence"));
        assert!(doc.contains("no direct compute invocation"));
        assert!(doc.contains("no safety override"));
        assert!(doc.contains("no implicit global region orchestration"));
        assert!(doc.contains("no new inter-region platform formation"));
    }

    #[test]
    fn first_inter_region_implementation_map_anchors_exactly_three_relations() {
        assert_eq!(
            CANONICAL_BLUE_BRAIN_FIRST_INTER_REGION_IMPLEMENTATION_MAP.len(),
            10
        );
        assert_eq!(
            CANONICAL_BLUE_BRAIN_INTER_REGION_IMPLEMENTATION_RELATION_CLASS_MAP.len(),
            6
        );

        let implemented = CANONICAL_BLUE_BRAIN_FIRST_INTER_REGION_IMPLEMENTATION_MAP
            .iter()
            .filter(|relation| is_blue_brain_first_inter_region_relation_implemented(relation.pair))
            .count();
        assert_eq!(implemented, 3);

        assert_eq!(
            blue_brain_first_inter_region_implementation_relation_for_pair(
                BlueBrainInterRegionArchitecturePair::AmygdalaThalamus
            )
            .implementation_relation_class,
            BlueBrainInterRegionImplementationRelationClass::ImplementedDirectBoundedAdvisoryRelation
        );
        assert_eq!(
            blue_brain_first_inter_region_implementation_relation_for_pair(
                BlueBrainInterRegionArchitecturePair::HippocampusThalamus
            )
            .implementation_relation_class,
            BlueBrainInterRegionImplementationRelationClass::ImplementedReferenceMediatedRelation
        );
        assert_eq!(
            blue_brain_first_inter_region_implementation_relation_for_pair(
                BlueBrainInterRegionArchitecturePair::AmygdalaBasalGanglia
            )
            .implementation_relation_class,
            BlueBrainInterRegionImplementationRelationClass::ImplementedSelectionMediatedRelation
        );
    }

    #[test]
    fn first_inter_region_implementation_preserves_mediated_signal_boundaries() {
        let direct = blue_brain_first_inter_region_implementation_relation_for_pair(
            BlueBrainInterRegionArchitecturePair::AmygdalaThalamus,
        );
        assert_eq!(
            direct.mediation_path,
            BlueBrainInterRegionImplementationMediationPath::DirectBoundedAdvisoryOnly
        );
        assert_eq!(
            direct.source_to_target_signal,
            BlueBrainInterRegionImplementationSignal::SalienceCaveatAdvisory
        );
        assert_eq!(
            direct.target_to_source_signal,
            BlueBrainInterRegionImplementationSignal::RelayRoutingDiagnostic
        );

        let reference = blue_brain_first_inter_region_implementation_relation_for_pair(
            BlueBrainInterRegionArchitecturePair::HippocampusThalamus,
        );
        assert_eq!(
            reference.mediation_path,
            BlueBrainInterRegionImplementationMediationPath::ReferenceContextMediatedOnly
        );
        assert_eq!(
            reference.source_to_target_signal,
            BlueBrainInterRegionImplementationSignal::ContextReferenceDiagnostic
        );
        assert_eq!(
            reference.target_to_source_signal,
            BlueBrainInterRegionImplementationSignal::RelayRoutingDiagnostic
        );

        let selection = blue_brain_first_inter_region_implementation_relation_for_pair(
            BlueBrainInterRegionArchitecturePair::AmygdalaBasalGanglia,
        );
        assert_eq!(
            selection.mediation_path,
            BlueBrainInterRegionImplementationMediationPath::SelectionContractMediatedOnly
        );
        assert_eq!(
            selection.source_to_target_signal,
            BlueBrainInterRegionImplementationSignal::SalienceCaveatAdvisory
        );
        assert_eq!(
            selection.target_to_source_signal,
            BlueBrainInterRegionImplementationSignal::SelectionReadinessDiagnostic
        );
    }

    #[test]
    fn first_inter_region_implementation_defers_or_blocks_every_other_pair() {
        for deferred_pair in [
            BlueBrainInterRegionArchitecturePair::HippocampusAmygdala,
            BlueBrainInterRegionArchitecturePair::HippocampusCerebellum,
            BlueBrainInterRegionArchitecturePair::AmygdalaCerebellum,
            BlueBrainInterRegionArchitecturePair::ThalamusBasalGanglia,
            BlueBrainInterRegionArchitecturePair::ThalamusCerebellum,
            BlueBrainInterRegionArchitecturePair::BasalGangliaCerebellum,
        ] {
            let relation =
                blue_brain_first_inter_region_implementation_relation_for_pair(deferred_pair);
            assert_eq!(
                relation.implementation_relation_class,
                BlueBrainInterRegionImplementationRelationClass::DeferredNotYetImplementedRelation
            );
            assert_eq!(
                relation.mediation_path,
                BlueBrainInterRegionImplementationMediationPath::NotYetImplemented
            );
            assert!(!is_blue_brain_first_inter_region_relation_implemented(
                deferred_pair
            ));
        }

        let blocked = blue_brain_first_inter_region_implementation_relation_for_pair(
            BlueBrainInterRegionArchitecturePair::HippocampusBasalGanglia,
        );
        assert_eq!(
            blocked.implementation_relation_class,
            BlueBrainInterRegionImplementationRelationClass::BlockedRelation
        );
        assert_eq!(
            blocked.mediation_path,
            BlueBrainInterRegionImplementationMediationPath::BlockedUnavailable
        );
    }

    #[test]
    fn first_inter_region_implementation_cannot_create_direct_authority() {
        for relation in CANONICAL_BLUE_BRAIN_FIRST_INTER_REGION_IMPLEMENTATION_MAP {
            assert!(relation.advisory_only);
            assert!(relation.bidirectional_pair_label);
            assert!(!relation.direct_action_trigger);
            assert!(!relation.direct_execution_trigger);
            assert!(!relation.direct_retry_trigger);
            assert!(!relation.direct_memory_commit);
            assert!(!relation.direct_compute_invocation);
            assert!(!relation.safety_override);
            assert!(!relation.global_region_orchestration);
        }
    }

    #[test]
    fn first_inter_region_implementation_doc_pins_exact_relations_and_guards() {
        let doc = include_str!(
            "../../../docs/blue_brain_first_inter_region_implementation_serie_ir1_prompt2_v1.md"
        );
        assert!(doc.contains("implemented direct bounded advisory relation"));
        assert!(doc.contains("implemented reference-mediated relation"));
        assert!(doc.contains("implemented selection-mediated relation"));
        assert!(doc.contains("Hippocampus ↔ Thalamus"));
        assert!(doc.contains("Amygdala ↔ Thalamus"));
        assert!(doc.contains("Amygdala ↔ Basal Ganglia"));
        assert!(doc.contains("exactly three implemented relations"));
        assert!(doc.contains("SalienceCaveatAdvisory"));
        assert!(doc.contains("ContextReferenceDiagnostic"));
        assert!(doc.contains("SelectionReadinessDiagnostic"));
        assert!(doc.contains("deferred/not-yet-implemented relation"));
        assert!(doc.contains("no direct action trigger"));
        assert!(doc.contains("no direct execution trigger"));
        assert!(doc.contains("no direct retry trigger"));
        assert!(doc.contains("no automatic memory persistence"));
        assert!(doc.contains("no direct compute invocation"));
        assert!(doc.contains("no new inter-region platform formation"));
    }

    #[test]
    fn inter_region_diagnostics_contract_map_separates_canonical_states() {
        assert_eq!(
            CANONICAL_BLUE_BRAIN_INTER_REGION_DIAGNOSTICS_CONTRACT_MAP.len(),
            10
        );
        assert_eq!(
            CANONICAL_BLUE_BRAIN_INTER_REGION_DIAGNOSTICS_CONTRACT_CLASS_MAP.len(),
            8
        );
        for class in [
            BlueBrainInterRegionDiagnosticsContractClass::AdvisoryOnlyRelationDiagnostic,
            BlueBrainInterRegionDiagnosticsContractClass::CaveatedRelationDiagnostic,
            BlueBrainInterRegionDiagnosticsContractClass::DeferredRelationDiagnostic,
            BlueBrainInterRegionDiagnosticsContractClass::BlockedRelationDiagnostic,
            BlueBrainInterRegionDiagnosticsContractClass::InsufficientRelationDiagnostic,
            BlueBrainInterRegionDiagnosticsContractClass::DiagnosticOnlyRelationState,
            BlueBrainInterRegionDiagnosticsContractClass::BoundedRelationContractSignal,
            BlueBrainInterRegionDiagnosticsContractClass::NonCanonicalInternalOnlyRelationPath,
        ] {
            assert!(
                CANONICAL_BLUE_BRAIN_INTER_REGION_DIAGNOSTICS_CONTRACT_CLASS_MAP.contains(&class)
            );
        }

        assert_ne!(
            BlueBrainInterRegionDiagnosticsRelationState::AdvisoryOnlyActive,
            BlueBrainInterRegionDiagnosticsRelationState::CaveatedNoStrongPositiveSignal
        );
        assert_ne!(
            BlueBrainInterRegionDiagnosticsRelationState::DeferredNotYetUsable,
            BlueBrainInterRegionDiagnosticsRelationState::BlockedByContractSafetyOrReference
        );
        assert_ne!(
            BlueBrainInterRegionDiagnosticsRelationState::BlockedByContractSafetyOrReference,
            BlueBrainInterRegionDiagnosticsRelationState::InsufficientRelationalBasis
        );
        assert_ne!(
            BlueBrainInterRegionDiagnosticsRelationState::DiagnosticOnlyVisible,
            BlueBrainInterRegionDiagnosticsRelationState::AdvisoryOnlyActive
        );

        assert_eq!(
            blue_brain_inter_region_diagnostics_contract_class_for_state(
                BlueBrainInterRegionDiagnosticsRelationState::CaveatedNoStrongPositiveSignal
            ),
            BlueBrainInterRegionDiagnosticsContractClass::CaveatedRelationDiagnostic
        );
        assert_eq!(
            blue_brain_inter_region_contract_signal_for_state(
                BlueBrainInterRegionDiagnosticsRelationState::AdvisoryOnlyActive
            ),
            BlueBrainInterRegionContractSignalClass::BoundedRelationContractSignal
        );
        assert_eq!(
            blue_brain_inter_region_contract_signal_for_state(
                BlueBrainInterRegionDiagnosticsRelationState::InsufficientRelationalBasis
            ),
            BlueBrainInterRegionContractSignalClass::InsufficientRelationDiagnosticSignal
        );
    }

    #[test]
    fn implemented_inter_region_relations_share_one_runtime_selection_reference_read() {
        for pair in [
            BlueBrainInterRegionArchitecturePair::AmygdalaThalamus,
            BlueBrainInterRegionArchitecturePair::HippocampusThalamus,
            BlueBrainInterRegionArchitecturePair::AmygdalaBasalGanglia,
        ] {
            assert!(blue_brain_inter_region_consumer_contract_reads_are_aligned(
                pair
            ));
            let runtime = blue_brain_inter_region_diagnostics_contract_read_for_pair(
                pair,
                BlueBrainInterRegionConsumerLayer::Runtime,
            );
            let selection = blue_brain_inter_region_diagnostics_contract_read_for_pair(
                pair,
                BlueBrainInterRegionConsumerLayer::Selection,
            );
            let reference = blue_brain_inter_region_diagnostics_contract_read_for_pair(
                pair,
                BlueBrainInterRegionConsumerLayer::Reference,
            );

            assert_eq!(runtime.relation_state, selection.relation_state);
            assert_eq!(runtime.relation_state, reference.relation_state);
            assert_eq!(
                runtime.relation_state,
                BlueBrainInterRegionDiagnosticsRelationState::AdvisoryOnlyActive
            );
            assert_eq!(
                runtime.contract_signal_class,
                BlueBrainInterRegionContractSignalClass::BoundedRelationContractSignal
            );
            assert!(runtime.bounded_contract_signal);
            assert!(runtime.advisory_only);
            assert!(!runtime.caveated);
            assert!(!runtime.deferred);
            assert!(!runtime.blocked);
            assert!(!runtime.insufficient);
            assert!(!runtime.diagnostic_only);
        }
    }

    #[test]
    fn deferred_blocked_and_noncanonical_inter_region_reads_do_not_blur() {
        let deferred = blue_brain_inter_region_diagnostics_contract_read_for_pair(
            BlueBrainInterRegionArchitecturePair::ThalamusCerebellum,
            BlueBrainInterRegionConsumerLayer::Runtime,
        );
        assert_eq!(
            deferred.relation_state,
            BlueBrainInterRegionDiagnosticsRelationState::DeferredNotYetUsable
        );
        assert!(deferred.deferred);
        assert!(deferred.diagnostic_only);
        assert!(!deferred.blocked);
        assert!(!deferred.insufficient);
        assert!(!deferred.bounded_contract_signal);

        let blocked = blue_brain_inter_region_diagnostics_contract_read_for_pair(
            BlueBrainInterRegionArchitecturePair::HippocampusBasalGanglia,
            BlueBrainInterRegionConsumerLayer::Runtime,
        );
        assert_eq!(
            blocked.relation_state,
            BlueBrainInterRegionDiagnosticsRelationState::BlockedByContractSafetyOrReference
        );
        assert!(blocked.blocked);
        assert!(blocked.diagnostic_only);
        assert!(!blocked.deferred);
        assert!(!blocked.insufficient);
        assert!(!blocked.bounded_contract_signal);

        assert_eq!(
            blue_brain_inter_region_diagnostics_contract_class_for_state(
                BlueBrainInterRegionDiagnosticsRelationState::NonCanonicalInternalOnly
            ),
            BlueBrainInterRegionDiagnosticsContractClass::NonCanonicalInternalOnlyRelationPath
        );
    }

    #[test]
    fn inter_region_contract_reads_cannot_create_direct_authority_or_shortcuts() {
        for read in CANONICAL_BLUE_BRAIN_INTER_REGION_DIAGNOSTICS_CONTRACT_MAP {
            assert!(!read.direct_action_trigger);
            assert!(!read.direct_execution_trigger);
            assert!(!read.direct_retry_trigger);
            assert!(!read.direct_memory_commit);
            assert!(!read.direct_compute_invocation);
            assert!(!read.safety_override);
            assert!(!read.global_region_orchestration);

            match read.implementation_relation_class {
                BlueBrainInterRegionImplementationRelationClass::ImplementedDirectBoundedAdvisoryRelation => {
                    assert_eq!(
                        read.mediation_path,
                        BlueBrainInterRegionImplementationMediationPath::DirectBoundedAdvisoryOnly
                    );
                }
                BlueBrainInterRegionImplementationRelationClass::ImplementedReferenceMediatedRelation => {
                    assert_eq!(
                        read.mediation_path,
                        BlueBrainInterRegionImplementationMediationPath::ReferenceContextMediatedOnly
                    );
                }
                BlueBrainInterRegionImplementationRelationClass::ImplementedSelectionMediatedRelation => {
                    assert_eq!(
                        read.mediation_path,
                        BlueBrainInterRegionImplementationMediationPath::SelectionContractMediatedOnly
                    );
                }
                BlueBrainInterRegionImplementationRelationClass::DeferredNotYetImplementedRelation => {
                    assert_eq!(
                        read.mediation_path,
                        BlueBrainInterRegionImplementationMediationPath::NotYetImplemented
                    );
                }
                BlueBrainInterRegionImplementationRelationClass::BlockedRelation => {
                    assert_eq!(
                        read.mediation_path,
                        BlueBrainInterRegionImplementationMediationPath::BlockedUnavailable
                    );
                }
                BlueBrainInterRegionImplementationRelationClass::NonCanonicalInternalOnlyRelationPath => {
                    assert_eq!(
                        read.mediation_path,
                        BlueBrainInterRegionImplementationMediationPath::NonCanonicalInternalOnly
                    );
                }
            }
        }
    }

    #[test]
    fn ir1_readiness_sweep_keeps_operational_deferred_blocked_and_caveat_slots_explicit() {
        let mut stable_implemented = 0;
        let mut deferred = 0;
        let mut blocked_or_diagnostic = 0;
        let mut usable_with_caveats = 0;

        for read in CANONICAL_BLUE_BRAIN_INTER_REGION_DIAGNOSTICS_CONTRACT_MAP {
            match blue_brain_ir1_readiness_class_for_contract_read(read) {
                BlueBrainIr1ReadinessClass::StableImplementedRelation => {
                    stable_implemented += 1;
                    assert!(read.advisory_only);
                    assert!(read.bounded_contract_signal);
                    assert!(!read.diagnostic_only);
                }
                BlueBrainIr1ReadinessClass::DeferredNotYetActive => {
                    deferred += 1;
                    assert!(read.deferred);
                    assert!(read.diagnostic_only);
                    assert!(!read.blocked);
                    assert!(!read.bounded_contract_signal);
                }
                BlueBrainIr1ReadinessClass::BlockedInsufficientDiagnosticOnly => {
                    blocked_or_diagnostic += 1;
                    assert!(read.blocked || read.insufficient || read.diagnostic_only);
                    assert!(!read.bounded_contract_signal);
                }
                BlueBrainIr1ReadinessClass::UsableWithCaveats => {
                    usable_with_caveats += 1;
                }
                BlueBrainIr1ReadinessClass::AdvisoryOnly
                | BlueBrainIr1ReadinessClass::NonCanonicalInternalOnly => {}
            }
        }

        assert_eq!(stable_implemented, 3);
        assert_eq!(deferred, 6);
        assert_eq!(blocked_or_diagnostic, 1);
        assert_eq!(usable_with_caveats, 0);
    }

    #[test]
    fn ir1_readiness_doc_pins_closeout_next_direction_and_out_of_scope_boundary() {
        let doc = include_str!(
            "../../../docs/blue_brain_ir1_readiness_sweep_inter_region_closure_serie_ir1_prompt4_v1.md"
        );
        assert!(doc.contains("IR1-readiness map"));
        assert!(doc.contains("Stable implemented relation"));
        assert!(doc.contains("Usable with caveats"));
        assert!(doc.contains("No Prompt-4 pair is operationally usable-with-caveats"));
        assert!(doc.contains("Amygdala ↔ Thalamus"));
        assert!(doc.contains("Hippocampus ↔ Thalamus"));
        assert!(doc.contains("Amygdala ↔ Basal Ganglia"));
        assert!(doc.contains("DirectBoundedAdvisoryOnly"));
        assert!(doc.contains("ReferenceContextMediatedOnly"));
        assert!(doc.contains("SelectionContractMediatedOnly"));
        assert!(doc.contains("no direct Action Execution"));
        assert!(doc.contains("no Retry-Orchestrierung"));
        assert!(doc.contains("no automatische Memory-Persistenz"));
        assert!(doc.contains("keine implizite globale Inter-Region-Plattform"));
        assert!(doc.contains("Compute bleibt maintenance-only"));
        assert!(doc.contains("selektive Modellvertiefung"));
        assert_eq!(
            BLUE_BRAIN_POST_IR1_PRIORITIZED_NEXT_DIRECTION,
            BlueBrainPostIr1NextDirection::SelectiveModelDeepening
        );
    }

    #[test]
    fn inter_region_diagnostics_contract_doc_pins_prompt3_semantics() {
        let doc = include_str!(
            "../../../docs/blue_brain_inter_region_diagnostics_contracts_serie_ir1_prompt3_v1.md"
        );
        assert!(doc.contains("advisory-only relation diagnostic"));
        assert!(doc.contains("caveated relation diagnostic"));
        assert!(doc.contains("deferred relation diagnostic"));
        assert!(doc.contains("blocked relation diagnostic"));
        assert!(doc.contains("insufficient relation diagnostic"));
        assert!(doc.contains("diagnostic-only relation state"));
        assert!(doc.contains("bounded relation contract signal"));
        assert!(doc.contains("non-canonical/internal-only relation path"));
        assert!(doc.contains("Runtime, Selection, and Reference read the same relation_state"));
        assert!(doc.contains("relation contract signal is not an action request"));
        assert!(doc.contains("not an execution trigger"));
        assert!(doc.contains("not a retry trigger"));
        assert!(doc.contains("not a memory commit"));
        assert!(doc.contains("not a compute trigger"));
        assert!(doc.contains("not a safety override"));
        assert!(doc.contains("DirectBoundedAdvisoryOnly"));
        assert!(doc.contains("ReferenceContextMediatedOnly"));
        assert!(doc.contains("SelectionContractMediatedOnly"));
        assert!(doc.contains("exactly the three Prompt 2 implemented relations"));
    }

    #[test]
    fn md1_model_deepening_classes_regions_and_relations_are_distinct() {
        assert_eq!(CANONICAL_BLUE_BRAIN_MD1_MODEL_DEEPENING_CLASS_MAP.len(), 6);
        assert!(CANONICAL_BLUE_BRAIN_MD1_MODEL_DEEPENING_CLASS_MAP
            .contains(&BlueBrainMd1ModelDeepeningClass::AbstractSufficient));
        assert!(CANONICAL_BLUE_BRAIN_MD1_MODEL_DEEPENING_CLASS_MAP
            .contains(&BlueBrainMd1ModelDeepeningClass::BoundedKuramotoLikeCandidate));
        assert!(CANONICAL_BLUE_BRAIN_MD1_MODEL_DEEPENING_CLASS_MAP.contains(
            &BlueBrainMd1ModelDeepeningClass::HodgkinHuxleySimulationOnlyDiagnosticOnlyCandidate
        ));
        assert!(CANONICAL_BLUE_BRAIN_MD1_MODEL_DEEPENING_CLASS_MAP
            .contains(&BlueBrainMd1ModelDeepeningClass::LaterSelectiveHodgkinHuxleyDeepening));
        assert!(CANONICAL_BLUE_BRAIN_MD1_MODEL_DEEPENING_CLASS_MAP
            .contains(&BlueBrainMd1ModelDeepeningClass::NoDeepeningNeededNow));
        assert!(CANONICAL_BLUE_BRAIN_MD1_MODEL_DEEPENING_CLASS_MAP
            .contains(&BlueBrainMd1ModelDeepeningClass::NonCanonicalInternalOnlyModelPath));

        assert_eq!(
            CANONICAL_BLUE_BRAIN_MD1_REGION_DEEPENING_DECISION_MAP.len(),
            5
        );
        assert_eq!(
            blue_brain_md1_region_deepening_decision(BlueBrainAnatomicalRegionClass::Hippocampus)
                .deepening_class,
            BlueBrainMd1ModelDeepeningClass::AbstractSufficient
        );
        assert_eq!(
            blue_brain_md1_region_deepening_decision(BlueBrainAnatomicalRegionClass::Amygdala)
                .deepening_class,
            BlueBrainMd1ModelDeepeningClass::BoundedKuramotoLikeCandidate
        );
        assert_eq!(
            blue_brain_md1_region_deepening_decision(BlueBrainAnatomicalRegionClass::Thalamus)
                .deepening_class,
            BlueBrainMd1ModelDeepeningClass::AbstractSufficient
        );
        assert_eq!(
            blue_brain_md1_region_deepening_decision(BlueBrainAnatomicalRegionClass::BasalGanglia)
                .deepening_class,
            BlueBrainMd1ModelDeepeningClass::NoDeepeningNeededNow
        );
        assert_eq!(
            blue_brain_md1_region_deepening_decision(BlueBrainAnatomicalRegionClass::Cerebellum)
                .deepening_class,
            BlueBrainMd1ModelDeepeningClass::HodgkinHuxleySimulationOnlyDiagnosticOnlyCandidate
        );
        assert_eq!(
            blue_brain_md1_region_deepening_decision(
                BlueBrainAnatomicalRegionClass::PrefrontalCortex
            )
            .deepening_class,
            BlueBrainMd1ModelDeepeningClass::NonCanonicalInternalOnlyModelPath
        );

        assert_eq!(
            CANONICAL_BLUE_BRAIN_MD1_RELATION_DEEPENING_DECISION_MAP.len(),
            10
        );
        for decision in CANONICAL_BLUE_BRAIN_MD1_RELATION_DEEPENING_DECISION_MAP {
            assert!(blue_brain_md1_relation_deepening_is_consistent_with_implementation(decision));
        }
    }

    #[test]
    fn md1_prioritizes_only_two_bounded_relation_deepening_candidates() {
        let prioritized: Vec<_> = CANONICAL_BLUE_BRAIN_MD1_RELATION_DEEPENING_DECISION_MAP
            .iter()
            .copied()
            .filter(|decision| {
                decision.priority_class
                    == BlueBrainMd1DeepeningPriorityClass::NextConcreteDeepeningCandidate
            })
            .collect();

        assert_eq!(prioritized.len(), 2);
        assert!(prioritized.iter().any(|decision| {
            decision.pair == BlueBrainInterRegionArchitecturePair::AmygdalaThalamus
                && decision.deepening_class
                    == BlueBrainMd1ModelDeepeningClass::BoundedKuramotoLikeCandidate
                && decision.coupling_synchrony_gating_timing_leverage
                && !decision.excitability_spiking_membrane_leverage
        }));
        assert!(prioritized.iter().any(|decision| {
            decision.pair == BlueBrainInterRegionArchitecturePair::AmygdalaBasalGanglia
                && decision.deepening_class
                    == BlueBrainMd1ModelDeepeningClass::BoundedKuramotoLikeCandidate
                && decision.coupling_synchrony_gating_timing_leverage
                && !decision.excitability_spiking_membrane_leverage
        }));

        assert_eq!(
            blue_brain_md1_relation_deepening_decision(
                BlueBrainInterRegionArchitecturePair::HippocampusThalamus
            )
            .deepening_class,
            BlueBrainMd1ModelDeepeningClass::AbstractSufficient
        );
        assert_eq!(
            blue_brain_md1_relation_deepening_decision(
                BlueBrainInterRegionArchitecturePair::BasalGangliaCerebellum
            )
            .deepening_class,
            BlueBrainMd1ModelDeepeningClass::LaterSelectiveHodgkinHuxleyDeepening
        );
    }

    #[test]
    fn md1_model_deepening_cannot_expand_no_direct_scope() {
        for decision in CANONICAL_BLUE_BRAIN_MD1_REGION_DEEPENING_DECISION_MAP {
            assert!(decision.advisory_only);
            assert!(!decision.direct_action_trigger);
            assert!(!decision.direct_execution_trigger);
            assert!(!decision.direct_retry_trigger);
            assert!(!decision.direct_memory_commit);
            assert!(!decision.direct_compute_invocation);
            assert!(!decision.global_model_platform);
        }

        for decision in CANONICAL_BLUE_BRAIN_MD1_RELATION_DEEPENING_DECISION_MAP {
            assert!(decision.advisory_only);
            assert!(!decision.direct_action_trigger);
            assert!(!decision.direct_execution_trigger);
            assert!(!decision.direct_retry_trigger);
            assert!(!decision.direct_memory_commit);
            assert!(!decision.direct_compute_invocation);
            assert!(!decision.global_model_platform);
        }
    }

    #[test]
    fn md1_doc_pins_selective_model_deepening_decision_line() {
        let doc = include_str!(
            "../../../docs/blue_brain_md1_selective_model_deepening_decision_line_v1.md"
        );
        assert!(doc.contains("abstract sufficient"));
        assert!(doc.contains("bounded Kuramoto-like candidate"));
        assert!(doc.contains("HH simulation-only/diagnostic-only candidate"));
        assert!(doc.contains("later selective HH deepening"));
        assert!(doc.contains("no-deepening-needed-now"));
        assert!(doc.contains("non-canonical/internal-only model path"));
        assert!(doc.contains("Hippocampus"));
        assert!(doc.contains("Amygdala"));
        assert!(doc.contains("Thalamus"));
        assert!(doc.contains("Basal Ganglia"));
        assert!(doc.contains("Cerebellum"));
        assert!(doc.contains("Priority 1: `Amygdala ↔ Thalamus`"));
        assert!(doc.contains("Priority 2: `Amygdala ↔ Basal Ganglia`"));
        assert!(doc.contains("not direct Action control"));
        assert!(doc.contains("not direct Execution control"));
        assert!(doc.contains("not Retry orchestration"));
        assert!(doc.contains("not Memory mutation"));
        assert!(doc.contains("Compute-Core expansion"));
        assert!(doc.contains("global HH adoption"));

        let readme = include_str!("../../../docs/README.md");
        assert!(readme.contains("Selective model-deepening decision line (MD1)"));
        assert!(
            readme.contains("docs/blue_brain_md1_selective_model_deepening_decision_line_v1.md")
        );
    }
}
