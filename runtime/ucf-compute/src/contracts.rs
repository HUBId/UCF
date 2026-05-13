use sha2::{Digest, Sha256};

use crate::backend_pack::BackendComponentId;
use crate::capabilities::{LlmRequest, LlmResponse};
use crate::evidence::{digest_canonical, spikes_digest, EvidenceChain};
use crate::feature_extractor::{SaeInput, SaeOutput, SAE_MAX_SPIKES};
use crate::lfm::{LfmInput, LfmOutput};
use crate::ssm::{SsmInput, SsmOutput};
use crate::world_model::{WorldModelInput, WorldModelOutput};

pub const MAX_REASON_CODES: usize = 8;
pub const MAX_STAGE_ENCODED_BYTES: usize = 64 * 1024;
pub const NSR_CONTRACT_VERSION_V1: &str = "v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BackendClass {
    Stub,
    Toy,
    Mock,
    OptionalRealCompile,
    OptionalRealRuntime,
    RemoteExternal,
    Experimental,
    Deferred,
    ForbiddenForNow,
}

impl BackendClass {
    pub const fn runtime_real_claim(self) -> bool {
        matches!(self, Self::OptionalRealRuntime)
    }

    pub const fn remote_or_external(self) -> bool {
        matches!(self, Self::RemoteExternal)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BackendIdentity {
    pub name: &'static str,
    pub class: BackendClass,
    pub deterministic: bool,
    pub offline: bool,
    pub external_service_required: bool,
    pub runtime_inference_supported: bool,
    pub production_claim: bool,
}

impl BackendIdentity {
    pub const fn new(
        name: &'static str,
        class: BackendClass,
        deterministic: bool,
        offline: bool,
        external_service_required: bool,
        runtime_inference_supported: bool,
        production_claim: bool,
    ) -> Self {
        Self {
            name,
            class,
            deterministic,
            offline,
            external_service_required,
            runtime_inference_supported,
            production_claim,
        }
    }

    pub const fn stub(name: &'static str) -> Self {
        Self::new(name, BackendClass::Stub, true, true, false, false, false)
    }

    pub const fn toy(name: &'static str) -> Self {
        Self::new(name, BackendClass::Toy, true, true, false, false, false)
    }

    pub const fn mock(name: &'static str) -> Self {
        Self::new(name, BackendClass::Mock, true, true, false, false, false)
    }

    pub const fn optional_real_compile(name: &'static str) -> Self {
        Self::new(
            name,
            BackendClass::OptionalRealCompile,
            true,
            true,
            false,
            false,
            false,
        )
    }

    pub const fn optional_real_runtime(name: &'static str) -> Self {
        Self::new(
            name,
            BackendClass::OptionalRealRuntime,
            true,
            true,
            false,
            true,
            false,
        )
    }

    pub const fn remote_external(name: &'static str) -> Self {
        Self::new(
            name,
            BackendClass::RemoteExternal,
            false,
            false,
            true,
            false,
            false,
        )
    }

    pub const fn experimental(name: &'static str) -> Self {
        Self::new(
            name,
            BackendClass::Experimental,
            false,
            true,
            false,
            false,
            false,
        )
    }

    pub const fn deferred(name: &'static str) -> Self {
        Self::new(
            name,
            BackendClass::Deferred,
            false,
            true,
            false,
            false,
            false,
        )
    }

    pub const fn forbidden_for_now(name: &'static str) -> Self {
        Self::new(
            name,
            BackendClass::ForbiddenForNow,
            false,
            true,
            false,
            false,
            false,
        )
    }

    pub const fn claims_runtime_real_inference(self) -> bool {
        self.class.runtime_real_claim() && self.runtime_inference_supported
    }

    pub const fn default_safe(self) -> bool {
        self.offline && !self.external_service_required && !self.production_claim
    }
}

/// Shared-core action outcome code used by runtime operation/result invariants.
///
/// This is intentionally contract-only and reused across standard/expert/internal
/// surfaces to avoid path-local semantic drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeActionOutcomeCode {
    Accepted,
    Completed,
    NoOp,
    Blocked,
    Unsupported,
    Failed,
}

/// Canonical cross-cutting runtime invariants for load-bearing core paths.
///
/// These rules intentionally stay compact and are shared across execution,
/// rollout, replay, diagnostics, and expert runtime-op surfaces:
/// - `blocked`, `failed`, and `no_op` are distinct outcome classes.
/// - `partial`, `stale`, `caveated`, and `degraded` are not interchangeable.
/// - mutating actions require a trustable/current-enough state basis.
/// - snapshots/diagnostics/evidence extend canonical run truth and must not
///   become a competing source of truth.
/// - rollout/replay/expert paths extend the shared core; they do not replace it.
pub const CROSS_CUTTING_PRODUCTION_INVARIANTS_V1: [&str; 5] = [
    "blocked!=failed!=no_op",
    "partial/stale/caveated/degraded_separated",
    "mutating_actions_require_trustable_state_basis",
    "snapshot_diagnostics_evidence_extend_canonical_truth",
    "rollout_replay_expert_extend_shared_core",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeHandoffKind {
    Execution,
    Diagnostics,
    Replay,
    Rollout,
    ExpertAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeHandoffState {
    Complete,
    Partial,
    Caveated,
    Blocked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandoffReferenceRequirement {
    Required,
    RequiredIfAvailable,
    Optional,
    InternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RuntimeHandoffReferenceSet {
    pub job_run_identity: HandoffReferenceRequirement,
    pub snapshot_evidence_refs: HandoffReferenceRequirement,
    pub active_rollout_state_refs: HandoffReferenceRequirement,
    pub replay_context_refs: HandoffReferenceRequirement,
    pub trust_diagnostics_refs: HandoffReferenceRequirement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RuntimeHandoffSemantics {
    pub kind: RuntimeHandoffKind,
    pub canonical_transition: &'static str,
    pub references: RuntimeHandoffReferenceSet,
    pub side_path_policy: &'static str,
}

pub const CANONICAL_RUNTIME_HANDOFF_SEMANTICS_V1: [RuntimeHandoffSemantics; 5] = [
    RuntimeHandoffSemantics {
        kind: RuntimeHandoffKind::Execution,
        canonical_transition: "submit/compute -> execution_snapshot/diagnostics/evidence",
        references: RuntimeHandoffReferenceSet {
            job_run_identity: HandoffReferenceRequirement::Required,
            snapshot_evidence_refs: HandoffReferenceRequirement::Required,
            active_rollout_state_refs: HandoffReferenceRequirement::Optional,
            replay_context_refs: HandoffReferenceRequirement::Optional,
            trust_diagnostics_refs: HandoffReferenceRequirement::RequiredIfAvailable,
        },
        side_path_policy:
            "historical or helper paths are extension-only and must not redefine run truth",
    },
    RuntimeHandoffSemantics {
        kind: RuntimeHandoffKind::Diagnostics,
        canonical_transition: "runtime snapshot/diagnostics -> expert action preconditions",
        references: RuntimeHandoffReferenceSet {
            job_run_identity: HandoffReferenceRequirement::RequiredIfAvailable,
            snapshot_evidence_refs: HandoffReferenceRequirement::Required,
            active_rollout_state_refs: HandoffReferenceRequirement::RequiredIfAvailable,
            replay_context_refs: HandoffReferenceRequirement::Optional,
            trust_diagnostics_refs: HandoffReferenceRequirement::Required,
        },
        side_path_policy:
            "diagnostics adapters may enrich context but cannot bypass trustability gates",
    },
    RuntimeHandoffSemantics {
        kind: RuntimeHandoffKind::Replay,
        canonical_transition: "replay preflight -> replay execution -> replay diagnostics",
        references: RuntimeHandoffReferenceSet {
            job_run_identity: HandoffReferenceRequirement::Required,
            snapshot_evidence_refs: HandoffReferenceRequirement::RequiredIfAvailable,
            active_rollout_state_refs: HandoffReferenceRequirement::RequiredIfAvailable,
            replay_context_refs: HandoffReferenceRequirement::Required,
            trust_diagnostics_refs: HandoffReferenceRequirement::RequiredIfAvailable,
        },
        side_path_policy:
            "legacy replay shortcuts are non-canonical and must stay explicit/internal-only",
    },
    RuntimeHandoffSemantics {
        kind: RuntimeHandoffKind::Rollout,
        canonical_transition: "rollout decision -> activation/fallback/rollback outcome",
        references: RuntimeHandoffReferenceSet {
            job_run_identity: HandoffReferenceRequirement::RequiredIfAvailable,
            snapshot_evidence_refs: HandoffReferenceRequirement::RequiredIfAvailable,
            active_rollout_state_refs: HandoffReferenceRequirement::Required,
            replay_context_refs: HandoffReferenceRequirement::Optional,
            trust_diagnostics_refs: HandoffReferenceRequirement::Required,
        },
        side_path_policy:
            "guarded activation/fallback/rollback are canonical; hidden bypasses are not",
    },
    RuntimeHandoffSemantics {
        kind: RuntimeHandoffKind::ExpertAction,
        canonical_transition:
            "expert runtime op/replay op -> same core runtime state + diagnostics",
        references: RuntimeHandoffReferenceSet {
            job_run_identity: HandoffReferenceRequirement::RequiredIfAvailable,
            snapshot_evidence_refs: HandoffReferenceRequirement::RequiredIfAvailable,
            active_rollout_state_refs: HandoffReferenceRequirement::RequiredIfAvailable,
            replay_context_refs: HandoffReferenceRequirement::RequiredIfAvailable,
            trust_diagnostics_refs: HandoffReferenceRequirement::Required,
        },
        side_path_policy:
            "expert/high-trust path extends shared contracts and must not create second semantics",
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeEntryClass {
    StandardCanonical,
    ExpertHighTrust,
    InternalDevTest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeContractShape {
    CanonicalCompute,
    ExpertReplay,
    ExpertRuntimeOps,
    InternalControl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeContractSafety {
    StandardSafe,
    HighTrustOnly,
    InternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeSurfaceExtension {
    Standard,
    Expert,
    Internal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalSnapshotConsistency {
    Current,
    Partial,
    Stale,
    DriftAffected,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertDiagnosticsAvailability {
    Available,
    Partial,
    Unavailable,
    Blocked,
    InternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeStatusCore {
    Current,
    Partial,
    Stale,
    DriftSuspected,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeFreshnessClass {
    Current,
    Partial,
    Stale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeDriftClass {
    NoDriftDetected,
    DriftSuspected,
    InconsistentNeedsRefresh,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeDiagnosticsCore {
    Available,
    Partial,
    Unavailable,
    Blocked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalEvidenceKind {
    ExecutionRun,
    MutatingAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalEvidenceStatus {
    Sufficient,
    Partial,
    Caveated,
    Insufficient,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalTraceSliceKind {
    StagePath,
    PlacementDecision,
    RolloutActionDecision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalTraceSliceStatus {
    Sufficient,
    Partial,
    StaleOrCaveated,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalEvidenceReasonCode {
    PlacementPathChosen,
    PlacementConstrainedOrFallback,
    RolloutActionAllowed,
    RolloutActionBlocked,
    ReplayCaveated,
    ReplayBlocked,
    RecoveryTrustImproved,
    RecoveryTrustNotImproved,
    WarmupCapabilityCaveat,
    StaleDiagnosticBasis,
}

impl RuntimeEntryClass {
    pub const fn extension(self) -> RuntimeSurfaceExtension {
        match self {
            Self::StandardCanonical => RuntimeSurfaceExtension::Standard,
            Self::ExpertHighTrust => RuntimeSurfaceExtension::Expert,
            Self::InternalDevTest => RuntimeSurfaceExtension::Internal,
        }
    }

    pub const fn replay_contract_shape(self) -> RuntimeContractShape {
        match self {
            Self::StandardCanonical => RuntimeContractShape::CanonicalCompute,
            Self::ExpertHighTrust => RuntimeContractShape::ExpertReplay,
            Self::InternalDevTest => RuntimeContractShape::InternalControl,
        }
    }

    pub const fn runtime_ops_contract_shape(self) -> RuntimeContractShape {
        match self {
            Self::StandardCanonical => RuntimeContractShape::CanonicalCompute,
            Self::ExpertHighTrust => RuntimeContractShape::ExpertRuntimeOps,
            Self::InternalDevTest => RuntimeContractShape::InternalControl,
        }
    }

    pub const fn contract_safety(self) -> RuntimeContractSafety {
        match self {
            Self::StandardCanonical => RuntimeContractSafety::StandardSafe,
            Self::ExpertHighTrust => RuntimeContractSafety::HighTrustOnly,
            Self::InternalDevTest => RuntimeContractSafety::InternalOnly,
        }
    }
}

impl CanonicalSnapshotConsistency {
    pub const fn core(self) -> RuntimeStatusCore {
        match self {
            Self::Current => RuntimeStatusCore::Current,
            Self::Partial => RuntimeStatusCore::Partial,
            Self::Stale => RuntimeStatusCore::Stale,
            Self::DriftAffected => RuntimeStatusCore::DriftSuspected,
            Self::Unavailable => RuntimeStatusCore::Unavailable,
        }
    }
}

impl ExpertDiagnosticsAvailability {
    pub const fn core(self) -> Option<RuntimeDiagnosticsCore> {
        match self {
            Self::Available => Some(RuntimeDiagnosticsCore::Available),
            Self::Partial => Some(RuntimeDiagnosticsCore::Partial),
            Self::Unavailable => Some(RuntimeDiagnosticsCore::Unavailable),
            Self::Blocked => Some(RuntimeDiagnosticsCore::Blocked),
            Self::InternalOnly => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertWorkflowClass {
    InspectDiagnoseAct,
    ReplayOriented,
    RolloutOriented,
    InternalDevTestOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertWorkflowTransitionState {
    Supported,
    Partial,
    Blocked,
    InternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertMutationBoundary {
    ReadOnly,
    ControlledMutable,
    HighImpactMutable,
    InternalDevTestOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertMutationBlocker {
    StaleDiagnosticBasis,
    ConflictingRuntimeState,
    SubsystemConstrainedOrBusy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertMutationResult {
    NoMutationReadOnly,
    StateChanged,
    NoOp,
    GuardedMutation,
    PartialEffect,
    BlockedBySafetyRail,
    UnsupportedInRuntimeContext,
}

impl RuntimeFreshnessClass {
    pub const fn is_stale_or_partial(self) -> bool {
        matches!(self, Self::Partial | Self::Stale)
    }
}

impl RuntimeActionOutcomeCode {
    pub const fn is_blocked_or_failed(self) -> bool {
        matches!(self, Self::Blocked | Self::Failed)
    }

    pub const fn is_non_terminal_noop(self) -> bool {
        matches!(self, Self::NoOp)
    }
}

impl RuntimeDriftClass {
    pub const fn needs_refresh(self) -> bool {
        matches!(self, Self::InconsistentNeedsRefresh)
    }
}

impl CanonicalEvidenceStatus {
    pub const fn diagnostics_core(self) -> RuntimeDiagnosticsCore {
        match self {
            Self::Sufficient => RuntimeDiagnosticsCore::Available,
            Self::Partial | Self::Caveated => RuntimeDiagnosticsCore::Partial,
            Self::Insufficient => RuntimeDiagnosticsCore::Unavailable,
        }
    }
}

impl CanonicalTraceSliceStatus {
    pub const fn diagnostics_core(self) -> RuntimeDiagnosticsCore {
        match self {
            Self::Sufficient => RuntimeDiagnosticsCore::Available,
            Self::Partial | Self::StaleOrCaveated => RuntimeDiagnosticsCore::Partial,
            Self::Unavailable => RuntimeDiagnosticsCore::Unavailable,
        }
    }
}

pub const fn canonical_runtime_handoff_semantics() -> &'static [RuntimeHandoffSemantics] {
    &CANONICAL_RUNTIME_HANDOFF_SEMANTICS_V1
}

pub const fn runtime_handoff_state_from_snapshot_and_diagnostics(
    snapshot: CanonicalSnapshotConsistency,
    diagnostics: ExpertDiagnosticsAvailability,
) -> RuntimeHandoffState {
    match (snapshot, diagnostics) {
        (CanonicalSnapshotConsistency::Unavailable, _)
        | (
            _,
            ExpertDiagnosticsAvailability::Unavailable | ExpertDiagnosticsAvailability::Blocked,
        ) => RuntimeHandoffState::Blocked,
        (_, ExpertDiagnosticsAvailability::InternalOnly) => RuntimeHandoffState::Caveated,
        (CanonicalSnapshotConsistency::Current, ExpertDiagnosticsAvailability::Available) => {
            RuntimeHandoffState::Complete
        }
        (
            CanonicalSnapshotConsistency::Current | CanonicalSnapshotConsistency::Partial,
            ExpertDiagnosticsAvailability::Partial,
        ) => RuntimeHandoffState::Caveated,
        _ => RuntimeHandoffState::Partial,
    }
}

pub const fn runtime_handoff_state_from_evidence(
    evidence: CanonicalEvidenceStatus,
    trace: CanonicalTraceSliceStatus,
) -> RuntimeHandoffState {
    match (evidence, trace) {
        (CanonicalEvidenceStatus::Insufficient, _)
        | (_, CanonicalTraceSliceStatus::Unavailable) => RuntimeHandoffState::Blocked,
        (CanonicalEvidenceStatus::Sufficient, CanonicalTraceSliceStatus::Sufficient) => {
            RuntimeHandoffState::Complete
        }
        (CanonicalEvidenceStatus::Caveated, _)
        | (_, CanonicalTraceSliceStatus::StaleOrCaveated) => RuntimeHandoffState::Caveated,
        _ => RuntimeHandoffState::Partial,
    }
}

pub const fn runtime_handoff_state_from_action_code(
    code: RuntimeActionOutcomeCode,
) -> RuntimeHandoffState {
    match code {
        RuntimeActionOutcomeCode::Completed => RuntimeHandoffState::Complete,
        RuntimeActionOutcomeCode::Accepted | RuntimeActionOutcomeCode::NoOp => {
            RuntimeHandoffState::Partial
        }
        RuntimeActionOutcomeCode::Blocked
        | RuntimeActionOutcomeCode::Failed
        | RuntimeActionOutcomeCode::Unsupported => RuntimeHandoffState::Blocked,
    }
}

/// Shared-core compatibility rule for action-result semantics.
///
/// Invariants:
/// - `blocked` and `failed` map to `blocked_by_safety_rail`
/// - `no_op` maps to `no_op` or `guarded_mutation`
/// - `completed` maps to read-only completion, state change, or partial effect
/// - `unsupported` maps to `unsupported_in_runtime_context`
/// - `accepted` maps to `guarded_mutation`
pub fn runtime_action_core_semantics_consistent(
    code: RuntimeActionOutcomeCode,
    mutation_result: ExpertMutationResult,
) -> bool {
    match code {
        RuntimeActionOutcomeCode::Accepted => {
            mutation_result == ExpertMutationResult::GuardedMutation
        }
        RuntimeActionOutcomeCode::Completed => matches!(
            mutation_result,
            ExpertMutationResult::NoMutationReadOnly
                | ExpertMutationResult::StateChanged
                | ExpertMutationResult::PartialEffect
        ),
        RuntimeActionOutcomeCode::NoOp => matches!(
            mutation_result,
            ExpertMutationResult::NoOp | ExpertMutationResult::GuardedMutation
        ),
        RuntimeActionOutcomeCode::Blocked | RuntimeActionOutcomeCode::Failed => {
            mutation_result == ExpertMutationResult::BlockedBySafetyRail
        }
        RuntimeActionOutcomeCode::Unsupported => {
            mutation_result == ExpertMutationResult::UnsupportedInRuntimeContext
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[repr(u16)]
pub enum StageContractVersion {
    V1 = 1,
}

impl StageContractVersion {
    pub const fn as_u16(self) -> u16 {
        self as u16
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum ValidationStatus {
    Ok = 0,
    Warned = 1,
    Degraded = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum ViolationCode {
    ScalarOutOfRange = 1,
    SpikeCountExceeded = 2,
    SpikesDigestMismatch = 3,
    ReadoutDigestMismatch = 4,
    SizeCapExceeded = 5,
    ChainDigestMismatch = 6,
    BackendContractMismatch = 7,
    PressureJumpExceeded = 8,
    LlmDigestMismatch = 9,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidationReport {
    pub status: ValidationStatus,
    pub violation_mask: u32,
}

impl ValidationReport {
    pub const fn ok() -> Self {
        Self {
            status: ValidationStatus::Ok,
            violation_mask: 0,
        }
    }

    pub fn add_hard(&mut self, code: ViolationCode) {
        self.status = ValidationStatus::Degraded;
        self.set_code(code);
    }

    pub fn add_soft(&mut self, code: ViolationCode) {
        if self.status == ValidationStatus::Ok {
            self.status = ValidationStatus::Warned;
        }
        self.set_code(code);
    }

    fn set_code(&mut self, code: ViolationCode) {
        let shift = (code as u32).min(31);
        self.violation_mask |= 1_u32 << shift;
    }

    pub fn merge(mut self, rhs: Self) -> Self {
        self.status = match (self.status, rhs.status) {
            (ValidationStatus::Degraded, _) | (_, ValidationStatus::Degraded) => {
                ValidationStatus::Degraded
            }
            (ValidationStatus::Warned, _) | (_, ValidationStatus::Warned) => {
                ValidationStatus::Warned
            }
            _ => ValidationStatus::Ok,
        };
        self.violation_mask |= rhs.violation_mask;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageKind {
    World,
    Sae,
    Ssm,
    Lfm,
    Llm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilitySupportLevel {
    Supported,
    SupportedWithConstraints,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityConstraint {
    OnlyLocal,
    OnlyRemoteWorker,
    WarmReadyPreferred,
    GuardedDegradedUsage,
    CapacityOrColdStartCaveat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StagePathSupportLevel {
    Supported,
    SupportedWithConstraints,
    DegradedOnly,
    FallbackOnly,
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StagePathCapability {
    pub stage: StageKind,
    pub path_segment: &'static str,
    pub support: StagePathSupportLevel,
    pub constraints: Vec<CapabilityConstraint>,
    pub detail: Option<String>,
}

pub trait ContractRegistry {
    fn supports(
        &self,
        stage: StageKind,
        backend_id: BackendComponentId,
        version: StageContractVersion,
    ) -> bool;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct StageContractRegistry;

impl ContractRegistry for StageContractRegistry {
    fn supports(
        &self,
        stage: StageKind,
        backend_id: BackendComponentId,
        version: StageContractVersion,
    ) -> bool {
        if version != StageContractVersion::V1 {
            return false;
        }
        !matches!(
            (stage, backend_id),
            (_, BackendComponentId::Disabled) | (StageKind::World, BackendComponentId::StubV0)
        )
    }
}

pub struct WorldValidatorV1;
impl WorldValidatorV1 {
    pub fn validate(input: &WorldModelInput, output: &WorldModelOutput) -> ValidationReport {
        let mut report = ValidationReport::ok();
        if !(0.0..=1.0).contains(&output.surprise)
            || !(0.0..=1.0).contains(&output.prediction_error)
            || !(0.0..=1.0).contains(&output.state_norm)
        {
            report.add_hard(ViolationCode::ScalarOutOfRange);
        }
        if output.prediction_digest == [0; 32] && input.t > 0 {
            report.add_hard(ViolationCode::ScalarOutOfRange);
        }
        if serde_json::to_vec(output)
            .map(|buf| buf.len() > MAX_STAGE_ENCODED_BYTES)
            .unwrap_or(true)
        {
            report.add_hard(ViolationCode::SizeCapExceeded);
        }
        report
    }
}

pub struct SaeValidatorV1;
impl SaeValidatorV1 {
    pub fn validate(_input: &SaeInput, output: &SaeOutput) -> ValidationReport {
        let mut report = ValidationReport::ok();
        if usize::from(output.spike_count) > SAE_MAX_SPIKES || output.spikes.len() > SAE_MAX_SPIKES
        {
            report.add_hard(ViolationCode::SpikeCountExceeded);
        }
        if !(0.0..=1.0).contains(&output.sparsity) || !(0.0..=1.0).contains(&output.energy) {
            report.add_hard(ViolationCode::ScalarOutOfRange);
        }
        if output
            .spikes
            .iter()
            .any(|s| !(0.0..=1.0).contains(&s.magnitude))
        {
            report.add_hard(ViolationCode::ScalarOutOfRange);
        }
        if output
            .spikes
            .windows(2)
            .any(|pair| pair[0].feature_id >= pair[1].feature_id)
        {
            report.add_hard(ViolationCode::ScalarOutOfRange);
        }
        if spikes_digest(&output.spikes) != output.spikes_digest {
            report.add_hard(ViolationCode::SpikesDigestMismatch);
        }
        if serde_json::to_vec(output)
            .map(|buf| buf.len() > MAX_STAGE_ENCODED_BYTES)
            .unwrap_or(true)
        {
            report.add_hard(ViolationCode::SizeCapExceeded);
        }
        report
    }
}

pub struct SsmValidatorV1;
impl SsmValidatorV1 {
    pub fn validate(
        input: &SsmInput,
        output: &SsmOutput,
        previous_pressure: Option<f32>,
    ) -> ValidationReport {
        let mut report = ValidationReport::ok();
        if !(0.0..=1.0).contains(&output.pressure)
            || !(0.0..=1.0).contains(&output.readout)
            || !(0.0..=1.0).contains(&output.state_norm)
        {
            report.add_hard(ViolationCode::ScalarOutOfRange);
        }
        let expected_readout = {
            let mut hasher = Sha256::new();
            hasher.update(output.readout.to_bits().to_le_bytes());
            hasher.update(output.state_digest);
            hasher.finalize()
        };
        if expected_readout[..] != output.readout_digest {
            report.add_hard(ViolationCode::ReadoutDigestMismatch);
        }
        if input.spike_count as usize > SAE_MAX_SPIKES {
            report.add_hard(ViolationCode::SpikeCountExceeded);
        }
        if let Some(prev) = previous_pressure {
            if (output.pressure - prev).abs() > 0.65 {
                report.add_soft(ViolationCode::PressureJumpExceeded);
            }
        }
        if serde_json::to_vec(output)
            .map(|buf| buf.len() > MAX_STAGE_ENCODED_BYTES)
            .unwrap_or(true)
        {
            report.add_hard(ViolationCode::SizeCapExceeded);
        }
        report
    }
}

pub struct LfmValidatorV1;
impl LfmValidatorV1 {
    pub fn validate(input: &LfmInput, output: &LfmOutput) -> ValidationReport {
        let mut report = ValidationReport::ok();
        for scalar in [
            output.uncertainty,
            output.stability,
            output.state_norm,
            output.deriv_norm,
            output.saturation_ratio,
            output.homeostasis_error,
            input.surprise,
            input.sae_energy,
            input.pressure,
        ] {
            if !(0.0..=1.0).contains(&scalar) {
                report.add_hard(ViolationCode::ScalarOutOfRange);
            }
        }
        let mut readout_hasher = Sha256::new();
        readout_hasher.update(output.uncertainty_q.to_le_bytes());
        readout_hasher.update(output.stability_q.to_le_bytes());
        readout_hasher.update(output.homeostasis_error_q.to_le_bytes());
        readout_hasher.update(output.liquid_state_digest);
        let expected: [u8; 32] = readout_hasher.finalize().into();
        if expected != output.liquid_readout_digest {
            report.add_hard(ViolationCode::ReadoutDigestMismatch);
        }
        if output.nan_inf_detected {
            report.add_hard(ViolationCode::ScalarOutOfRange);
        }
        if serde_json::to_vec(output)
            .map(|buf| buf.len() > MAX_STAGE_ENCODED_BYTES)
            .unwrap_or(true)
        {
            report.add_hard(ViolationCode::SizeCapExceeded);
        }
        report
    }
}

pub struct LlmValidatorV1;
impl LlmValidatorV1 {
    pub fn validate(_req: &LlmRequest, response: &LlmResponse) -> ValidationReport {
        let mut report = ValidationReport::ok();
        if response.compute_digest() != response.digest {
            report.add_hard(ViolationCode::LlmDigestMismatch);
        }
        if serde_json::to_vec(&response.text)
            .map(|buf| buf.len() > MAX_STAGE_ENCODED_BYTES)
            .unwrap_or(true)
        {
            report.add_hard(ViolationCode::SizeCapExceeded);
        }
        report
    }
}

pub fn validate_evidence_chain_digest(chain: &EvidenceChain) -> ValidationReport {
    let mut cloned = *chain;
    cloned.chain_digest = [0; 32];
    let expected = digest_canonical(&cloned);
    let mut report = ValidationReport::ok();
    if expected != chain.chain_digest {
        report.add_hard(ViolationCode::ChainDigestMismatch);
    }
    report
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum NsrContractVersion {
    V1 = 1,
}

impl NsrContractVersion {
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::V1 => NSR_CONTRACT_VERSION_V1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NsrRequest {
    pub base_risk: f32,
    pub base_confidence: f32,
    pub pressure: f32,
    pub surprise: f32,
    pub compute_degraded: bool,
    pub contract_version: NsrContractVersion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NsrFailureKind {
    Disabled,
    Unavailable,
    ArtifactVerificationFailed,
    ContractMismatch,
    BackendUnavailable,
    ExecutionError,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NsrResult {
    pub risk: f32,
    pub confidence: f32,
    pub reason_codes: Vec<String>,
    pub digest: [u8; 32],
    pub engine_id: String,
    pub contract_version: NsrContractVersion,
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::capabilities::{FinishReason, LlmOutputClass, LlmRequest, LlmResponse, LlmStatus};

    proptest! {
        #[test]
        fn sae_validator_never_panics(spike_count in 0u16..300u16, mag in -2.0f32..2.0f32) {
            let spike = crate::Spike { feature_id: 1, magnitude: mag, timestamp: 1 };
            let output = SaeOutput {
                spikes: vec![spike],
                spike_count,
                sparsity: 0.5,
                energy: 0.5,
                spikes_digest: [0;32],
                quality: crate::world_model::StageQuality::Ok,
                notes: crate::feature_extractor::SmallNotes(vec![]),
            };
            let input = SaeInput { t:1, context_features:[0.0; crate::feature_extractor::SAE_INPUT_DIM], world_state_digest: None, seed: 1, evidence_chain_digest:[0;32]};
            let _ = SaeValidatorV1::validate(&input, &output);
        }
    }

    #[test]
    fn sae_digest_mutation_is_detected() {
        let mut out = SaeOutput {
            spikes: vec![crate::Spike {
                feature_id: 1,
                magnitude: 0.7,
                timestamp: 1,
            }],
            spike_count: 1,
            sparsity: 0.9,
            energy: 0.1,
            spikes_digest: [0; 32],
            quality: crate::world_model::StageQuality::Ok,
            notes: crate::feature_extractor::SmallNotes(vec![]),
        };
        out.spikes_digest = spikes_digest(&out.spikes);
        out.spikes[0].magnitude = 0.8;
        let input = SaeInput {
            t: 1,
            context_features: [0.0; crate::feature_extractor::SAE_INPUT_DIM],
            world_state_digest: None,
            seed: 1,
            evidence_chain_digest: [0; 32],
        };
        let report = SaeValidatorV1::validate(&input, &out);
        assert_eq!(report.status, ValidationStatus::Degraded);
    }

    #[test]
    fn sae_duplicate_feature_ids_are_rejected() {
        let mut out = SaeOutput {
            spikes: vec![
                crate::Spike {
                    feature_id: 2,
                    magnitude: 0.7,
                    timestamp: 1,
                },
                crate::Spike {
                    feature_id: 2,
                    magnitude: 0.6,
                    timestamp: 1,
                },
            ],
            spike_count: 2,
            sparsity: 0.8,
            energy: 0.2,
            spikes_digest: [0; 32],
            quality: crate::world_model::StageQuality::Ok,
            notes: crate::feature_extractor::SmallNotes(vec![]),
        };
        out.spikes_digest = spikes_digest(&out.spikes);
        let input = SaeInput {
            t: 1,
            context_features: [0.0; crate::feature_extractor::SAE_INPUT_DIM],
            world_state_digest: None,
            seed: 1,
            evidence_chain_digest: [0; 32],
        };
        let report = SaeValidatorV1::validate(&input, &out);
        assert_eq!(report.status, ValidationStatus::Degraded);
    }

    #[test]
    fn llm_digest_mutation_is_detected() {
        let req = LlmRequest {
            schema_version: 1,
            t: 1,
            decision_id: 1,
            candidate_id: 1,
            output_class: LlmOutputClass::SafeText,
            prompt: "ok".into(),
            context_digest: [0; 32],
            evidence_chain_digest: [1; 32],
            lfm_readout_digest: None,
            lfm_uncertainty: None,
            lfm_stability: None,
            coherence: None,
            instability: None,
            risk: None,
            confidence: None,
            seed: 1,
            max_tokens: 8,
            temperature: 0.1,
            top_p: 1.0,
            sampling_enabled: false,
        };
        let mut resp = LlmResponse::new(LlmStatus::Ok, "x".into(), 1, FinishReason::Stop);
        resp.digest = [0; 32];
        let report = LlmValidatorV1::validate(&req, &resp);
        assert_eq!(report.status, ValidationStatus::Degraded);
    }

    #[test]
    fn runtime_entry_class_contract_views_are_canonicalized() {
        assert_eq!(
            RuntimeEntryClass::StandardCanonical.extension(),
            RuntimeSurfaceExtension::Standard
        );
        assert_eq!(
            RuntimeEntryClass::StandardCanonical.replay_contract_shape(),
            RuntimeContractShape::CanonicalCompute
        );
        assert_eq!(
            RuntimeEntryClass::StandardCanonical.runtime_ops_contract_shape(),
            RuntimeContractShape::CanonicalCompute
        );
        assert_eq!(
            RuntimeEntryClass::StandardCanonical.contract_safety(),
            RuntimeContractSafety::StandardSafe
        );

        assert_eq!(
            RuntimeEntryClass::ExpertHighTrust.extension(),
            RuntimeSurfaceExtension::Expert
        );
        assert_eq!(
            RuntimeEntryClass::ExpertHighTrust.replay_contract_shape(),
            RuntimeContractShape::ExpertReplay
        );
        assert_eq!(
            RuntimeEntryClass::ExpertHighTrust.runtime_ops_contract_shape(),
            RuntimeContractShape::ExpertRuntimeOps
        );
        assert_eq!(
            RuntimeEntryClass::ExpertHighTrust.contract_safety(),
            RuntimeContractSafety::HighTrustOnly
        );

        assert_eq!(
            RuntimeEntryClass::InternalDevTest.extension(),
            RuntimeSurfaceExtension::Internal
        );
        assert_eq!(
            RuntimeEntryClass::InternalDevTest.replay_contract_shape(),
            RuntimeContractShape::InternalControl
        );
        assert_eq!(
            RuntimeEntryClass::InternalDevTest.runtime_ops_contract_shape(),
            RuntimeContractShape::InternalControl
        );
        assert_eq!(
            RuntimeEntryClass::InternalDevTest.contract_safety(),
            RuntimeContractSafety::InternalOnly
        );
    }

    #[test]
    fn snapshot_and_diagnostics_core_semantics_are_stable() {
        assert_eq!(
            CanonicalSnapshotConsistency::Current.core(),
            RuntimeStatusCore::Current
        );
        assert_eq!(
            CanonicalSnapshotConsistency::Partial.core(),
            RuntimeStatusCore::Partial
        );
        assert_eq!(
            CanonicalSnapshotConsistency::Stale.core(),
            RuntimeStatusCore::Stale
        );
        assert_eq!(
            CanonicalSnapshotConsistency::DriftAffected.core(),
            RuntimeStatusCore::DriftSuspected
        );
        assert_eq!(
            CanonicalSnapshotConsistency::Unavailable.core(),
            RuntimeStatusCore::Unavailable
        );

        assert_eq!(
            ExpertDiagnosticsAvailability::Available.core(),
            Some(RuntimeDiagnosticsCore::Available)
        );
        assert_eq!(
            ExpertDiagnosticsAvailability::Partial.core(),
            Some(RuntimeDiagnosticsCore::Partial)
        );
        assert_eq!(
            ExpertDiagnosticsAvailability::Unavailable.core(),
            Some(RuntimeDiagnosticsCore::Unavailable)
        );
        assert_eq!(
            ExpertDiagnosticsAvailability::Blocked.core(),
            Some(RuntimeDiagnosticsCore::Blocked)
        );
        assert_eq!(ExpertDiagnosticsAvailability::InternalOnly.core(), None);
    }

    #[test]
    fn runtime_action_core_semantics_are_stable() {
        assert!(runtime_action_core_semantics_consistent(
            RuntimeActionOutcomeCode::Accepted,
            ExpertMutationResult::GuardedMutation
        ));
        assert!(runtime_action_core_semantics_consistent(
            RuntimeActionOutcomeCode::Completed,
            ExpertMutationResult::StateChanged
        ));
        assert!(runtime_action_core_semantics_consistent(
            RuntimeActionOutcomeCode::Completed,
            ExpertMutationResult::NoMutationReadOnly
        ));
        assert!(runtime_action_core_semantics_consistent(
            RuntimeActionOutcomeCode::NoOp,
            ExpertMutationResult::NoOp
        ));
        assert!(runtime_action_core_semantics_consistent(
            RuntimeActionOutcomeCode::Blocked,
            ExpertMutationResult::BlockedBySafetyRail
        ));
        assert!(runtime_action_core_semantics_consistent(
            RuntimeActionOutcomeCode::Unsupported,
            ExpertMutationResult::UnsupportedInRuntimeContext
        ));
        assert!(!runtime_action_core_semantics_consistent(
            RuntimeActionOutcomeCode::Completed,
            ExpertMutationResult::BlockedBySafetyRail
        ));
        assert!(!runtime_action_core_semantics_consistent(
            RuntimeActionOutcomeCode::NoOp,
            ExpertMutationResult::StateChanged
        ));
    }

    #[test]
    fn evidence_and_trace_partial_caveat_semantics_are_aligned() {
        assert_eq!(
            CanonicalEvidenceStatus::Sufficient.diagnostics_core(),
            RuntimeDiagnosticsCore::Available
        );
        assert_eq!(
            CanonicalEvidenceStatus::Partial.diagnostics_core(),
            RuntimeDiagnosticsCore::Partial
        );
        assert_eq!(
            CanonicalEvidenceStatus::Caveated.diagnostics_core(),
            RuntimeDiagnosticsCore::Partial
        );
        assert_eq!(
            CanonicalEvidenceStatus::Insufficient.diagnostics_core(),
            RuntimeDiagnosticsCore::Unavailable
        );

        assert_eq!(
            CanonicalTraceSliceStatus::Sufficient.diagnostics_core(),
            RuntimeDiagnosticsCore::Available
        );
        assert_eq!(
            CanonicalTraceSliceStatus::Partial.diagnostics_core(),
            RuntimeDiagnosticsCore::Partial
        );
        assert_eq!(
            CanonicalTraceSliceStatus::StaleOrCaveated.diagnostics_core(),
            RuntimeDiagnosticsCore::Partial
        );
        assert_eq!(
            CanonicalTraceSliceStatus::Unavailable.diagnostics_core(),
            RuntimeDiagnosticsCore::Unavailable
        );
    }

    #[test]
    fn freshness_and_drift_refresh_invariants_are_explicit() {
        assert!(!RuntimeFreshnessClass::Current.is_stale_or_partial());
        assert!(RuntimeFreshnessClass::Partial.is_stale_or_partial());
        assert!(RuntimeFreshnessClass::Stale.is_stale_or_partial());

        assert!(!RuntimeDriftClass::NoDriftDetected.needs_refresh());
        assert!(!RuntimeDriftClass::DriftSuspected.needs_refresh());
        assert!(RuntimeDriftClass::InconsistentNeedsRefresh.needs_refresh());
    }

    #[test]
    fn cross_cutting_invariants_and_outcome_classes_are_explicit() {
        assert!(CROSS_CUTTING_PRODUCTION_INVARIANTS_V1.contains(&"blocked!=failed!=no_op"));
        assert!(CROSS_CUTTING_PRODUCTION_INVARIANTS_V1
            .contains(&"partial/stale/caveated/degraded_separated"));
        assert!(RuntimeActionOutcomeCode::Blocked.is_blocked_or_failed());
        assert!(RuntimeActionOutcomeCode::Failed.is_blocked_or_failed());
        assert!(!RuntimeActionOutcomeCode::NoOp.is_blocked_or_failed());
        assert!(RuntimeActionOutcomeCode::NoOp.is_non_terminal_noop());
        assert!(!RuntimeActionOutcomeCode::Completed.is_non_terminal_noop());
    }

    #[test]
    fn canonical_runtime_handoff_semantics_cover_required_transitions() {
        let handoffs = canonical_runtime_handoff_semantics();
        assert_eq!(handoffs.len(), 5);
        assert!(handoffs
            .iter()
            .any(|handoff| handoff.kind == RuntimeHandoffKind::Execution
                && handoff.canonical_transition.contains("execution_snapshot")));
        assert!(handoffs
            .iter()
            .any(|handoff| handoff.kind == RuntimeHandoffKind::Diagnostics
                && handoff
                    .canonical_transition
                    .contains("expert action preconditions")));
        assert!(handoffs
            .iter()
            .any(|handoff| handoff.kind == RuntimeHandoffKind::Replay
                && handoff.canonical_transition.contains("replay execution")));
        assert!(handoffs
            .iter()
            .any(|handoff| handoff.kind == RuntimeHandoffKind::Rollout
                && handoff
                    .canonical_transition
                    .contains("activation/fallback/rollback")));
        assert!(handoffs
            .iter()
            .any(|handoff| handoff.kind == RuntimeHandoffKind::ExpertAction
                && handoff
                    .canonical_transition
                    .contains("same core runtime state")));
    }

    #[test]
    fn handoff_states_are_explicit_for_complete_partial_caveated_and_blocked() {
        assert_eq!(
            runtime_handoff_state_from_snapshot_and_diagnostics(
                CanonicalSnapshotConsistency::Current,
                ExpertDiagnosticsAvailability::Available
            ),
            RuntimeHandoffState::Complete
        );
        assert_eq!(
            runtime_handoff_state_from_snapshot_and_diagnostics(
                CanonicalSnapshotConsistency::Current,
                ExpertDiagnosticsAvailability::Partial
            ),
            RuntimeHandoffState::Caveated
        );
        assert_eq!(
            runtime_handoff_state_from_snapshot_and_diagnostics(
                CanonicalSnapshotConsistency::Stale,
                ExpertDiagnosticsAvailability::Available
            ),
            RuntimeHandoffState::Partial
        );
        assert_eq!(
            runtime_handoff_state_from_snapshot_and_diagnostics(
                CanonicalSnapshotConsistency::Unavailable,
                ExpertDiagnosticsAvailability::Available
            ),
            RuntimeHandoffState::Blocked
        );
    }

    #[test]
    fn handoff_state_mapping_reuses_shared_evidence_and_action_semantics() {
        assert_eq!(
            runtime_handoff_state_from_evidence(
                CanonicalEvidenceStatus::Sufficient,
                CanonicalTraceSliceStatus::Sufficient
            ),
            RuntimeHandoffState::Complete
        );
        assert_eq!(
            runtime_handoff_state_from_evidence(
                CanonicalEvidenceStatus::Caveated,
                CanonicalTraceSliceStatus::Partial
            ),
            RuntimeHandoffState::Caveated
        );
        assert_eq!(
            runtime_handoff_state_from_evidence(
                CanonicalEvidenceStatus::Partial,
                CanonicalTraceSliceStatus::Partial
            ),
            RuntimeHandoffState::Partial
        );
        assert_eq!(
            runtime_handoff_state_from_evidence(
                CanonicalEvidenceStatus::Insufficient,
                CanonicalTraceSliceStatus::Sufficient
            ),
            RuntimeHandoffState::Blocked
        );

        assert_eq!(
            runtime_handoff_state_from_action_code(RuntimeActionOutcomeCode::Completed),
            RuntimeHandoffState::Complete
        );
        assert_eq!(
            runtime_handoff_state_from_action_code(RuntimeActionOutcomeCode::NoOp),
            RuntimeHandoffState::Partial
        );
        assert_eq!(
            runtime_handoff_state_from_action_code(RuntimeActionOutcomeCode::Blocked),
            RuntimeHandoffState::Blocked
        );
    }
}
