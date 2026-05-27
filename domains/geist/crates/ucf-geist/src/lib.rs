#![forbid(unsafe_code)]

use std::sync::Arc;

use blake3::Hasher;
use ucf_archive::ExperienceAppender;
use ucf_archive_store::{ArchiveAppender, ArchiveStore, RecordKind, RecordMeta};
use ucf_commit::commit_milestone_macro;
use ucf_evidence::{EvidenceEnvelope, EvidenceStore};
use ucf_policy_ecology::{ConsistencyReport, ConsistencyVerdict, DefaultPolicyEcology, GeistGate};
use ucf_sleep_coordinator::{
    MinimalSpineSleepAppliedBoundary, MinimalSpineSleepPlanAudit, MinimalSpineSleepPlanAuditStatus,
    SleepStateHandle, SleepStateUpdater,
};
use ucf_types::consolidation::ReplayApplied;
use ucf_types::v1::spec::{Digest as ProtoDigest, ExperienceRecord, MacroMilestone, ProofEnvelope};
use ucf_types::{Digest32, EvidenceId, LogicalTime, WallTime};

const SELFSTATE_DOMAIN: u16 = 0x4753; // "GS" for Geist SelfState
const DERIVED_DOMAIN: u16 = 0x4744; // "GD" for Geist Derived
const CANONICAL_SELFSTATE_DOMAIN: &[u8] = b"ucf.geist.self_state.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelfState {
    pub cycle_id: u64,
    pub ssm_commit: Digest32,
    pub workspace_commit: Digest32,
    pub risk_commit: Digest32,
    pub attn_commit: Digest32,
    pub ncde_commit: Digest32,
    pub consistency: u16,
    pub commit: Digest32,
}

impl SelfState {
    pub fn builder(cycle_id: u64) -> SelfStateBuilder {
        SelfStateBuilder::new(cycle_id)
    }

    pub fn stability_score(&self) -> u16 {
        self.consistency
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelfStateBuilder {
    cycle_id: u64,
    ssm_commit: Digest32,
    workspace_commit: Digest32,
    risk_commit: Digest32,
    attn_commit: Digest32,
    ncde_commit: Digest32,
    consistency: u16,
}

impl SelfStateBuilder {
    pub fn new(cycle_id: u64) -> Self {
        Self {
            cycle_id,
            ssm_commit: Digest32::new([0u8; 32]),
            workspace_commit: Digest32::new([0u8; 32]),
            risk_commit: Digest32::new([0u8; 32]),
            attn_commit: Digest32::new([0u8; 32]),
            ncde_commit: Digest32::new([0u8; 32]),
            consistency: 0,
        }
    }

    pub fn ssm_commit(mut self, commit: Digest32) -> Self {
        self.ssm_commit = commit;
        self
    }

    pub fn workspace_commit(mut self, commit: Digest32) -> Self {
        self.workspace_commit = commit;
        self
    }

    pub fn risk_commit(mut self, commit: Digest32) -> Self {
        self.risk_commit = commit;
        self
    }

    pub fn attn_commit(mut self, commit: Digest32) -> Self {
        self.attn_commit = commit;
        self
    }

    pub fn ncde_commit(mut self, commit: Digest32) -> Self {
        self.ncde_commit = commit;
        self
    }

    pub fn consistency(mut self, consistency: u16) -> Self {
        self.consistency = consistency.min(10_000);
        self
    }

    pub fn build(self) -> SelfState {
        let commit = commit_self_state(&self);
        SelfState {
            cycle_id: self.cycle_id,
            ssm_commit: self.ssm_commit,
            workspace_commit: self.workspace_commit,
            risk_commit: self.risk_commit,
            attn_commit: self.attn_commit,
            ncde_commit: self.ncde_commit,
            consistency: self.consistency,
            commit,
        }
    }
}

pub fn encode_self_state(state: &SelfState) -> Vec<u8> {
    let mut payload = Vec::with_capacity(8 + 6 * Digest32::LEN + 2);
    payload.extend_from_slice(&state.cycle_id.to_be_bytes());
    payload.extend_from_slice(state.ssm_commit.as_bytes());
    payload.extend_from_slice(state.workspace_commit.as_bytes());
    payload.extend_from_slice(state.risk_commit.as_bytes());
    payload.extend_from_slice(state.attn_commit.as_bytes());
    payload.extend_from_slice(state.ncde_commit.as_bytes());
    payload.extend_from_slice(&state.consistency.to_be_bytes());
    payload
}

fn commit_self_state(builder: &SelfStateBuilder) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(CANONICAL_SELFSTATE_DOMAIN);
    hasher.update(&builder.cycle_id.to_be_bytes());
    hasher.update(builder.ssm_commit.as_bytes());
    hasher.update(builder.workspace_commit.as_bytes());
    hasher.update(builder.risk_commit.as_bytes());
    hasher.update(builder.attn_commit.as_bytes());
    hasher.update(builder.ncde_commit.as_bytes());
    hasher.update(&builder.consistency.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

/// Version for Minimal Spine Geist projection candidates derived from bounded Sleep metadata.
pub const MINIMAL_SPINE_GEIST_PROJECTION_CANDIDATE_VERSION: u32 = 1;

/// Source marker for candidate-only Geist projection records derived from bounded Sleep metadata.
pub const MINIMAL_SPINE_GEIST_PROJECTION_CANDIDATE_SOURCE: &str =
    "minimal_spine_v1_geist_projection_candidate_from_sleep_boundary";

/// Version for verify-only Minimal Spine Geist projection audit values.
pub const MINIMAL_SPINE_GEIST_PROJECTION_AUDIT_VERSION: u32 = 1;

/// Source marker for local verify-only Geist projection audit reports.
pub const MINIMAL_SPINE_GEIST_PROJECTION_AUDIT_SOURCE: &str =
    "minimal_spine_v1_geist_projection_verify_only_audit";

/// Version for local Minimal Spine ISM candidate boundary values.
pub const MINIMAL_SPINE_ISM_CANDIDATE_BOUNDARY_VERSION: u32 = 1;

/// Source marker for local candidate/read-model ISM boundary values.
pub const MINIMAL_SPINE_ISM_CANDIDATE_BOUNDARY_SOURCE: &str =
    "minimal_spine_v1_ism_candidate_boundary_from_geist_projection_audit";

/// Digest/provenance input for a candidate-only Minimal Spine Geist projection.
///
/// This value is bounded Sleep metadata only. It carries no store, appender, Gateway, GeistKernel,
/// ISM, policy mutation, scheduler, queue, worker, or runtime handle and cannot apply Geist.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MinimalSpineGeistProjectionInput {
    pub sleep_plan_audit_digest: Digest32,
    pub sleep_plan_candidate_digest: Digest32,
    pub sleep_applied_boundary_digest: Option<Digest32>,
    pub replay_audit_digest: Digest32,
    pub replay_schedule_digest: Digest32,
    pub token_count: u32,
    pub source: &'static str,
}

/// Deterministic, candidate-only Geist projection value derived from bounded Sleep metadata.
///
/// This is a local projection candidate only. It is not Geist runtime apply, not ISM write/upsert,
/// not identity finalization or anchoring, not policy mutation, not Evidence/Archive append, not
/// Gateway visibility, and not memory stabilization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MinimalSpineGeistProjectionCandidate {
    pub version: u32,
    pub sleep_plan_audit_digest: Digest32,
    pub sleep_plan_candidate_digest: Digest32,
    pub sleep_applied_boundary_digest: Option<Digest32>,
    pub replay_audit_digest: Digest32,
    pub replay_schedule_digest: Digest32,
    pub token_count: u32,
    pub projection_digest: Digest32,
    pub source: &'static str,
    pub sleep_source: &'static str,
    pub candidate_only: bool,
    pub geist_applied: bool,
    pub ism_written: bool,
    pub identity_anchor: bool,
    pub identity_finalized: bool,
    pub policy_mutated: bool,
    pub evidence_archive_appended: bool,
    pub gateway_visible: bool,
}

impl MinimalSpineGeistProjectionCandidate {
    /// Deterministic bytes used for the projection candidate digest.
    pub fn deterministic_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_u32_be(&mut out, self.version);
        push_digest32(&mut out, self.sleep_plan_audit_digest);
        push_digest32(&mut out, self.sleep_plan_candidate_digest);
        match self.sleep_applied_boundary_digest {
            Some(digest) => {
                out.push(1);
                push_digest32(&mut out, digest);
            }
            None => out.push(0),
        }
        push_digest32(&mut out, self.replay_audit_digest);
        push_digest32(&mut out, self.replay_schedule_digest);
        push_u32_be(&mut out, self.token_count);
        push_str32(&mut out, self.source);
        push_str32(&mut out, self.sleep_source);
        out.push(u8::from(self.candidate_only));
        out.push(u8::from(self.geist_applied));
        out.push(u8::from(self.ism_written));
        out.push(u8::from(self.identity_anchor));
        out.push(u8::from(self.identity_finalized));
        out.push(u8::from(self.policy_mutated));
        out.push(u8::from(self.evidence_archive_appended));
        out.push(u8::from(self.gateway_visible));
        out
    }

    pub fn digest(&self) -> Digest32 {
        digest_blake3_domain(
            b"ucf.geist.minimal_spine.projection_candidate_from_sleep_boundary.v1",
            &self.deterministic_bytes(),
        )
    }
}

/// PASS/FAIL status for a verify-only Minimal Spine Geist projection audit.
///
/// `Pass` means the projection candidate is internally consistent and all forbidden side-effect
/// flags remain false. It does not mean Geist was applied, ISM was written, identity was anchored
/// or finalized, policy was mutated, Evidence/Archive was appended, or Gateway/runtime authority
/// was exposed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MinimalSpineGeistProjectionAuditStatus {
    Pass,
    Fail,
}

impl MinimalSpineGeistProjectionAuditStatus {
    fn code(self) -> u8 {
        match self {
            Self::Pass => 1,
            Self::Fail => 2,
        }
    }
}

/// Deterministic failure reasons emitted by the verify-only Geist projection audit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MinimalSpineGeistProjectionAuditFailure {
    VersionMismatch,
    ProjectionDigestMismatch,
    ZeroSleepPlanAuditDigest,
    ZeroSleepPlanCandidateDigest,
    ZeroSleepAppliedBoundaryDigest,
    ZeroReplayAuditDigest,
    ZeroReplayScheduleDigest,
    InvalidTokenCount,
    EmptySource,
    EmptySleepSource,
    NotCandidateOnly,
    GeistAppliedFlagSet,
    IsmWrittenFlagSet,
    IdentityAnchorFlagSet,
    IdentityFinalizedFlagSet,
    PolicyMutatedFlagSet,
    EvidenceArchiveAppendedFlagSet,
    GatewayVisibleFlagSet,
}

impl MinimalSpineGeistProjectionAuditFailure {
    fn code(self) -> u8 {
        match self {
            Self::VersionMismatch => 1,
            Self::ProjectionDigestMismatch => 2,
            Self::ZeroSleepPlanAuditDigest => 3,
            Self::ZeroSleepPlanCandidateDigest => 4,
            Self::ZeroSleepAppliedBoundaryDigest => 5,
            Self::ZeroReplayAuditDigest => 6,
            Self::ZeroReplayScheduleDigest => 7,
            Self::InvalidTokenCount => 8,
            Self::EmptySource => 9,
            Self::EmptySleepSource => 10,
            Self::NotCandidateOnly => 11,
            Self::GeistAppliedFlagSet => 12,
            Self::IsmWrittenFlagSet => 13,
            Self::IdentityAnchorFlagSet => 14,
            Self::IdentityFinalizedFlagSet => 15,
            Self::PolicyMutatedFlagSet => 16,
            Self::EvidenceArchiveAppendedFlagSet => 17,
            Self::GatewayVisibleFlagSet => 18,
        }
    }
}

/// Local verify-only audit report for a Minimal Spine Geist projection candidate.
///
/// The audit is a pure deterministic consistency check over a candidate value. It takes no
/// `GeistKernel`, ISM store, policy mutator, appender, Gateway, scheduler, worker, or runtime
/// handle; it does not mutate the candidate; and all side-effect boundary flags in the report are
/// hard-coded false to prevent overclaiming.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MinimalSpineGeistProjectionAudit {
    pub version: u32,
    pub status: MinimalSpineGeistProjectionAuditStatus,
    pub failure_reasons: Vec<MinimalSpineGeistProjectionAuditFailure>,
    pub projection_digest: Digest32,
    pub recomputed_projection_digest: Digest32,
    pub sleep_plan_audit_digest: Digest32,
    pub sleep_plan_candidate_digest: Digest32,
    pub sleep_applied_boundary_digest: Option<Digest32>,
    pub replay_audit_digest: Digest32,
    pub replay_schedule_digest: Digest32,
    pub token_count: u32,
    pub audit_digest: Digest32,
    pub source: &'static str,
    pub candidate_source: &'static str,
    pub sleep_source: &'static str,
    pub candidate_only: bool,
    pub geist_applied: bool,
    pub ism_written: bool,
    pub identity_anchor: bool,
    pub identity_finalized: bool,
    pub policy_mutated: bool,
    pub evidence_archive_appended: bool,
    pub gateway_visible: bool,
}

impl MinimalSpineGeistProjectionAudit {
    /// Deterministic bytes used for the audit digest.
    pub fn deterministic_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_u32_be(&mut out, self.version);
        out.push(self.status.code());
        push_u32_be(
            &mut out,
            u32::try_from(self.failure_reasons.len())
                .expect("minimal spine geist projection audit failure reason count fits u32"),
        );
        for reason in &self.failure_reasons {
            out.push(reason.code());
        }
        push_digest32(&mut out, self.projection_digest);
        push_digest32(&mut out, self.recomputed_projection_digest);
        push_digest32(&mut out, self.sleep_plan_audit_digest);
        push_digest32(&mut out, self.sleep_plan_candidate_digest);
        match self.sleep_applied_boundary_digest {
            Some(digest) => {
                out.push(1);
                push_digest32(&mut out, digest);
            }
            None => out.push(0),
        }
        push_digest32(&mut out, self.replay_audit_digest);
        push_digest32(&mut out, self.replay_schedule_digest);
        push_u32_be(&mut out, self.token_count);
        push_str32(&mut out, self.source);
        push_str32(&mut out, self.candidate_source);
        push_str32(&mut out, self.sleep_source);
        out.push(u8::from(self.candidate_only));
        out.push(u8::from(self.geist_applied));
        out.push(u8::from(self.ism_written));
        out.push(u8::from(self.identity_anchor));
        out.push(u8::from(self.identity_finalized));
        out.push(u8::from(self.policy_mutated));
        out.push(u8::from(self.evidence_archive_appended));
        out.push(u8::from(self.gateway_visible));
        out
    }

    pub fn digest(&self) -> Digest32 {
        digest_blake3_domain(
            b"ucf.geist.minimal_spine.projection_verify_only_audit.v1",
            &self.deterministic_bytes(),
        )
    }
}

/// Deterministic local ISM candidate/read-model boundary derived only from a PASS Geist projection
/// audit.
///
/// This is not a persistent ISM write, not `IsmStore::upsert_anchor`, not an IdentityAnchor, not
/// identity finalization, not memory stabilization, not policy mutation, not Evidence/Archive append,
/// and not Gateway/action authority. It carries no store, kernel, appender, policy mutator, or runtime
/// handle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MinimalSpineIsmCandidateBoundary {
    pub version: u32,
    pub geist_projection_audit_digest: Digest32,
    pub geist_projection_digest: Digest32,
    pub sleep_plan_audit_digest: Digest32,
    pub sleep_plan_candidate_digest: Digest32,
    pub sleep_applied_boundary_digest: Option<Digest32>,
    pub replay_audit_digest: Digest32,
    pub replay_schedule_digest: Digest32,
    pub token_count: u32,
    pub ism_candidate_digest: Digest32,
    pub source: &'static str,
    pub audit_source: &'static str,
    pub candidate_source: &'static str,
    pub sleep_source: &'static str,
    pub ism_candidate_only: bool,
    pub ism_written: bool,
    pub ism_upserted: bool,
    pub identity_anchor: bool,
    pub identity_finalized: bool,
    pub memory_stabilized: bool,
    pub policy_mutated: bool,
    pub evidence_archive_appended: bool,
    pub gateway_visible: bool,
}

impl MinimalSpineIsmCandidateBoundary {
    /// Deterministic bytes used for the local ISM candidate boundary digest.
    pub fn deterministic_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_u32_be(&mut out, self.version);
        push_digest32(&mut out, self.geist_projection_audit_digest);
        push_digest32(&mut out, self.geist_projection_digest);
        push_digest32(&mut out, self.sleep_plan_audit_digest);
        push_digest32(&mut out, self.sleep_plan_candidate_digest);
        match self.sleep_applied_boundary_digest {
            Some(digest) => {
                out.push(1);
                push_digest32(&mut out, digest);
            }
            None => out.push(0),
        }
        push_digest32(&mut out, self.replay_audit_digest);
        push_digest32(&mut out, self.replay_schedule_digest);
        push_u32_be(&mut out, self.token_count);
        push_str32(&mut out, self.source);
        push_str32(&mut out, self.audit_source);
        push_str32(&mut out, self.candidate_source);
        push_str32(&mut out, self.sleep_source);
        out.push(u8::from(self.ism_candidate_only));
        out.push(u8::from(self.ism_written));
        out.push(u8::from(self.ism_upserted));
        out.push(u8::from(self.identity_anchor));
        out.push(u8::from(self.identity_finalized));
        out.push(u8::from(self.memory_stabilized));
        out.push(u8::from(self.policy_mutated));
        out.push(u8::from(self.evidence_archive_appended));
        out.push(u8::from(self.gateway_visible));
        out
    }

    pub fn digest(&self) -> Digest32 {
        digest_blake3_domain(
            b"ucf.geist.minimal_spine.ism_candidate_boundary_from_geist_projection_audit.v1",
            &self.deterministic_bytes(),
        )
    }
}

/// Version for the explicit Minimal Spine Geist/ISM Evidence/Archive append payload.
pub const MINIMAL_SPINE_GEIST_ISM_APPEND_PAYLOAD_VERSION: u32 = 1;

/// Explicit contract marker carried in the Geist/ISM Evidence/Archive append payload.
pub const MINIMAL_SPINE_GEIST_ISM_APPEND_CONTRACT: &str = "minimal_spine_geist_ism_append_v1";

/// Extension kind used for bounded Geist/ISM append proof records in archive-store.
///
/// Existing archive kinds do not include a bounded Geist/ISM audit/provenance wrapper. `IsmAnchor`
/// would overclaim persistent ISM authority, and Replay/Sleep already allocate `Other(65)` and
/// `Other(66)`. Prompt 67 therefore allocates `Other(67)` for this explicit append/readback
/// contract without changing archive-store schema or Minimal Spine v1.x.
pub const MINIMAL_SPINE_GEIST_ISM_APPEND_ARCHIVE_KIND: RecordKind = RecordKind::Other(67);

/// Queryable bounded append/readback kinds for EAQ3 cross-layer read-model candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceArchiveQueryableKindV1 {
    ReplayAppendV1,
    SleepAppendV1,
    GeistIsmAppendV1,
}

impl EvidenceArchiveQueryableKindV1 {
    pub const fn record_kind(self) -> RecordKind {
        match self {
            Self::ReplayAppendV1 => RecordKind::Other(65),
            Self::SleepAppendV1 => RecordKind::Other(66),
            Self::GeistIsmAppendV1 => RecordKind::Other(67),
        }
    }

    fn canonical_tag(self) -> u8 {
        match self {
            Self::ReplayAppendV1 => 65,
            Self::SleepAppendV1 => 66,
            Self::GeistIsmAppendV1 => 67,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrossLayerReadbackQueryCandidateStatusV1 {
    Complete,
    MissingRecord,
    Mismatch,
}

impl CrossLayerReadbackQueryCandidateStatusV1 {
    fn code(self) -> u8 {
        match self {
            Self::Complete => 0,
            Self::MissingRecord => 1,
            Self::Mismatch => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrossLayerReadbackQueryAuditStatusV1 {
    Pass,
    Fail,
    CandidateMissingRecord,
    CandidateMismatch,
}

impl CrossLayerReadbackQueryAuditStatusV1 {
    fn code(self) -> u8 {
        match self {
            Self::Pass => 0,
            Self::Fail => 1,
            Self::CandidateMissingRecord => 2,
            Self::CandidateMismatch => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrossLayerReadbackQueryAuditFailureV1 {
    EmptyCandidate,
    CandidateMissingRecord,
    CandidateMismatch,
    UnboundedKind,
    AppendWriteAuthorityPresent,
    GatewayAuthorityPresent,
    IdentityAuthorityPresent,
    RuntimeAuthorityPresent,
}

impl CrossLayerReadbackQueryAuditFailureV1 {
    fn code(self) -> u8 {
        match self {
            Self::EmptyCandidate => 1,
            Self::CandidateMissingRecord => 2,
            Self::CandidateMismatch => 3,
            Self::UnboundedKind => 4,
            Self::AppendWriteAuthorityPresent => 5,
            Self::GatewayAuthorityPresent => 6,
            Self::IdentityAuthorityPresent => 7,
            Self::RuntimeAuthorityPresent => 8,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceArchiveQueryRecordRefV1 {
    pub kind: EvidenceArchiveQueryableKindV1,
    pub archive_key_digest: Digest32,
    pub evidence_id_digest: Digest32,
    pub payload_digest: Digest32,
    pub archive_record_digest: Digest32,
    pub readback_digest: Digest32,
    pub root_commit_digest: Option<Digest32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CrossLayerReadbackQueryCandidateV1 {
    pub records: Vec<EvidenceArchiveQueryRecordRefV1>,
    pub status: CrossLayerReadbackQueryCandidateStatusV1,
    pub failures: Vec<String>,
}

impl CrossLayerReadbackQueryCandidateV1 {
    pub fn new(
        records: Vec<EvidenceArchiveQueryRecordRefV1>,
        status: CrossLayerReadbackQueryCandidateStatusV1,
        failures: Vec<String>,
    ) -> Self {
        Self {
            records,
            status,
            failures,
        }
    }

    pub const fn read_model_only(&self) -> bool {
        true
    }
    pub const fn append_authority(&self) -> bool {
        false
    }
    pub const fn gateway_authority(&self) -> bool {
        false
    }
    pub const fn identity_authority(&self) -> bool {
        false
    }
    pub const fn evidence_archive_write_authority(&self) -> bool {
        false
    }
    pub const fn runtime_authority(&self) -> bool {
        false
    }

    pub fn deterministic_bytes(&self) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(b"ucf.geist.evidence_archive_query_candidate.v1");
        payload.push(self.status.code());
        payload.extend_from_slice(&(self.failures.len() as u32).to_be_bytes());
        for failure in &self.failures {
            let bytes = failure.as_bytes();
            payload.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
            payload.extend_from_slice(bytes);
        }
        payload.extend_from_slice(&(self.records.len() as u32).to_be_bytes());
        for record in &self.records {
            payload.push(record.kind.canonical_tag());
            payload.extend_from_slice(record.archive_key_digest.as_bytes());
            payload.extend_from_slice(record.evidence_id_digest.as_bytes());
            payload.extend_from_slice(record.payload_digest.as_bytes());
            payload.extend_from_slice(record.archive_record_digest.as_bytes());
            payload.extend_from_slice(record.readback_digest.as_bytes());
            match record.root_commit_digest {
                Some(digest) => {
                    payload.push(1);
                    payload.extend_from_slice(digest.as_bytes());
                }
                None => payload.push(0),
            }
        }
        payload
    }

    pub fn digest(&self) -> Digest32 {
        let mut hasher = Hasher::new();
        hasher.update(&self.deterministic_bytes());
        Digest32::new(*hasher.finalize().as_bytes())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CrossLayerReadbackQueryVerifyAuditV1 {
    pub status: CrossLayerReadbackQueryAuditStatusV1,
    pub failures: Vec<CrossLayerReadbackQueryAuditFailureV1>,
    pub candidate_digest: Digest32,
    pub record_count: u32,
    pub audit_digest: Digest32,
}

impl CrossLayerReadbackQueryVerifyAuditV1 {
    pub const fn verify_only(&self) -> bool {
        true
    }
    pub const fn read_model_only(&self) -> bool {
        true
    }
    pub const fn append_write_authority(&self) -> bool {
        false
    }
    pub const fn gateway_authority(&self) -> bool {
        false
    }
    pub const fn identity_authority(&self) -> bool {
        false
    }
    pub const fn runtime_authority(&self) -> bool {
        false
    }
    pub const fn is_pass(&self) -> bool {
        matches!(self.status, CrossLayerReadbackQueryAuditStatusV1::Pass)
    }

    pub fn deterministic_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"ucf.geist.cross_layer_readback_query_verify_audit.v1");
        out.push(self.status.code());
        out.extend_from_slice(&(self.failures.len() as u32).to_be_bytes());
        for failure in &self.failures {
            out.push(failure.code());
        }
        out.extend_from_slice(self.candidate_digest.as_bytes());
        out.extend_from_slice(&self.record_count.to_be_bytes());
        out
    }

    pub fn digest(&self) -> Digest32 {
        let mut hasher = Hasher::new();
        hasher.update(&self.deterministic_bytes());
        Digest32::new(*hasher.finalize().as_bytes())
    }
}

pub fn verify_cross_layer_readback_query_candidate_v1(
    candidate: &CrossLayerReadbackQueryCandidateV1,
) -> CrossLayerReadbackQueryVerifyAuditV1 {
    let mut failures = Vec::new();
    if candidate.records.is_empty() {
        failures.push(CrossLayerReadbackQueryAuditFailureV1::EmptyCandidate);
    }
    if matches!(
        candidate.status,
        CrossLayerReadbackQueryCandidateStatusV1::MissingRecord
    ) {
        failures.push(CrossLayerReadbackQueryAuditFailureV1::CandidateMissingRecord);
    }
    if matches!(
        candidate.status,
        CrossLayerReadbackQueryCandidateStatusV1::Mismatch
    ) {
        failures.push(CrossLayerReadbackQueryAuditFailureV1::CandidateMismatch);
    }
    if candidate.append_authority() || candidate.evidence_archive_write_authority() {
        failures.push(CrossLayerReadbackQueryAuditFailureV1::AppendWriteAuthorityPresent);
    }
    if candidate.gateway_authority() {
        failures.push(CrossLayerReadbackQueryAuditFailureV1::GatewayAuthorityPresent);
    }
    if candidate.identity_authority() {
        failures.push(CrossLayerReadbackQueryAuditFailureV1::IdentityAuthorityPresent);
    }
    if candidate.runtime_authority() {
        failures.push(CrossLayerReadbackQueryAuditFailureV1::RuntimeAuthorityPresent);
    }
    if candidate
        .records
        .iter()
        .any(|record| !matches!(record.kind.record_kind(), RecordKind::Other(65..=67)))
    {
        failures.push(CrossLayerReadbackQueryAuditFailureV1::UnboundedKind);
    }

    let status = if failures.is_empty() {
        CrossLayerReadbackQueryAuditStatusV1::Pass
    } else {
        match candidate.status {
            CrossLayerReadbackQueryCandidateStatusV1::MissingRecord => {
                CrossLayerReadbackQueryAuditStatusV1::CandidateMissingRecord
            }
            CrossLayerReadbackQueryCandidateStatusV1::Mismatch => {
                CrossLayerReadbackQueryAuditStatusV1::CandidateMismatch
            }
            CrossLayerReadbackQueryCandidateStatusV1::Complete => {
                CrossLayerReadbackQueryAuditStatusV1::Fail
            }
        }
    };

    let mut audit = CrossLayerReadbackQueryVerifyAuditV1 {
        status,
        failures,
        candidate_digest: candidate.digest(),
        record_count: candidate.records.len() as u32,
        audit_digest: Digest32::new([0; 32]),
    };
    audit.audit_digest = audit.digest();
    audit
}

/// Meaning marker for the explicit Geist/ISM append contract.
pub const MINIMAL_SPINE_GEIST_ISM_APPEND_MEANING: &str =
    "audit_provenance_persistence_only_no_geist_runtime_no_ism_write";

/// Deterministic bounded Geist/ISM append payload persisted through Evidence/Archive.
///
/// This payload is provenance only. It preserves digest links for the bounded Geist projection
/// candidate, verify-only audit, local ISM candidate boundary, and upstream Sleep/Replay artifacts.
/// All runtime, ISM write/upsert, identity, memory, policy, and Gateway boundary flags are explicit
/// false values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MinimalSpineGeistIsmAppendPayload {
    pub version: u32,
    pub append_contract: String,
    pub geist_projection_candidate_digest: Digest32,
    pub geist_projection_audit_digest: Digest32,
    pub ism_candidate_boundary_digest: Digest32,
    pub sleep_plan_audit_digest: Digest32,
    pub sleep_plan_candidate_digest: Digest32,
    pub sleep_applied_boundary_digest: Option<Digest32>,
    pub replay_audit_digest: Digest32,
    pub replay_schedule_digest: Digest32,
    pub token_count: u32,
    pub candidate_source: &'static str,
    pub audit_source: &'static str,
    pub boundary_source: &'static str,
    pub sleep_source: &'static str,
    pub source: &'static str,
    pub geist_runtime_applied: bool,
    pub ism_written: bool,
    pub ism_upserted: bool,
    pub identity_anchor: bool,
    pub identity_finalized: bool,
    pub memory_stabilized: bool,
    pub policy_mutated: bool,
    pub gateway_visible: bool,
    pub evidence_archive_appended_meaning: &'static str,
}

impl MinimalSpineGeistIsmAppendPayload {
    pub fn from_artifacts(
        candidate: &MinimalSpineGeistProjectionCandidate,
        audit: &MinimalSpineGeistProjectionAudit,
        boundary: &MinimalSpineIsmCandidateBoundary,
    ) -> Result<Self, GeistIsmAppendError> {
        validate_minimal_spine_geist_ism_append_inputs(candidate, audit, boundary)?;
        let payload = Self {
            version: MINIMAL_SPINE_GEIST_ISM_APPEND_PAYLOAD_VERSION,
            append_contract: MINIMAL_SPINE_GEIST_ISM_APPEND_CONTRACT.to_string(),
            geist_projection_candidate_digest: candidate.projection_digest,
            geist_projection_audit_digest: audit.audit_digest,
            ism_candidate_boundary_digest: boundary.ism_candidate_digest,
            sleep_plan_audit_digest: candidate.sleep_plan_audit_digest,
            sleep_plan_candidate_digest: candidate.sleep_plan_candidate_digest,
            sleep_applied_boundary_digest: candidate.sleep_applied_boundary_digest,
            replay_audit_digest: candidate.replay_audit_digest,
            replay_schedule_digest: candidate.replay_schedule_digest,
            token_count: candidate.token_count,
            candidate_source: candidate.source,
            audit_source: audit.source,
            boundary_source: boundary.source,
            sleep_source: candidate.sleep_source,
            source: MINIMAL_SPINE_GEIST_ISM_APPEND_CONTRACT,
            geist_runtime_applied: false,
            ism_written: false,
            ism_upserted: false,
            identity_anchor: false,
            identity_finalized: false,
            memory_stabilized: false,
            policy_mutated: false,
            gateway_visible: false,
            evidence_archive_appended_meaning: MINIMAL_SPINE_GEIST_ISM_APPEND_MEANING,
        };
        if !payload.validate_links_nonzero() {
            return Err(GeistIsmAppendError::InvalidInput(
                "append payload links must be non-empty/non-zero",
            ));
        }
        Ok(payload)
    }

    pub fn deterministic_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_u32_be(&mut out, self.version);
        push_str32(&mut out, &self.append_contract);
        push_digest32(&mut out, self.geist_projection_candidate_digest);
        push_digest32(&mut out, self.geist_projection_audit_digest);
        push_digest32(&mut out, self.ism_candidate_boundary_digest);
        push_digest32(&mut out, self.sleep_plan_audit_digest);
        push_digest32(&mut out, self.sleep_plan_candidate_digest);
        match self.sleep_applied_boundary_digest {
            Some(digest) => {
                out.push(1);
                push_digest32(&mut out, digest);
            }
            None => out.push(0),
        }
        push_digest32(&mut out, self.replay_audit_digest);
        push_digest32(&mut out, self.replay_schedule_digest);
        push_u32_be(&mut out, self.token_count);
        push_str32(&mut out, self.candidate_source);
        push_str32(&mut out, self.audit_source);
        push_str32(&mut out, self.boundary_source);
        push_str32(&mut out, self.sleep_source);
        push_str32(&mut out, self.source);
        out.push(u8::from(self.geist_runtime_applied));
        out.push(u8::from(self.ism_written));
        out.push(u8::from(self.ism_upserted));
        out.push(u8::from(self.identity_anchor));
        out.push(u8::from(self.identity_finalized));
        out.push(u8::from(self.memory_stabilized));
        out.push(u8::from(self.policy_mutated));
        out.push(u8::from(self.gateway_visible));
        push_str32(&mut out, self.evidence_archive_appended_meaning);
        out
    }

    pub fn digest(&self) -> Digest32 {
        digest_blake3_domain(
            b"ucf.geist.minimal_spine.geist_ism.append_payload.v1",
            &self.deterministic_bytes(),
        )
    }

    pub fn validate_links_nonzero(&self) -> bool {
        self.version == MINIMAL_SPINE_GEIST_ISM_APPEND_PAYLOAD_VERSION
            && self.append_contract == MINIMAL_SPINE_GEIST_ISM_APPEND_CONTRACT
            && !is_zero_digest(self.geist_projection_candidate_digest)
            && !is_zero_digest(self.geist_projection_audit_digest)
            && !is_zero_digest(self.ism_candidate_boundary_digest)
            && !is_zero_digest(self.sleep_plan_audit_digest)
            && !is_zero_digest(self.sleep_plan_candidate_digest)
            && !self
                .sleep_applied_boundary_digest
                .is_some_and(is_zero_digest)
            && !is_zero_digest(self.replay_audit_digest)
            && !is_zero_digest(self.replay_schedule_digest)
            && self.token_count > 0
            && !self.candidate_source.is_empty()
            && !self.audit_source.is_empty()
            && !self.boundary_source.is_empty()
            && !self.sleep_source.is_empty()
            && !self.source.is_empty()
            && !self.geist_runtime_applied
            && !self.ism_written
            && !self.ism_upserted
            && !self.identity_anchor
            && !self.identity_finalized
            && !self.memory_stabilized
            && !self.policy_mutated
            && !self.gateway_visible
            && self.evidence_archive_appended_meaning == MINIMAL_SPINE_GEIST_ISM_APPEND_MEANING
    }
}

/// Result of the explicit Minimal Spine Geist/ISM append/readback contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MinimalSpineGeistIsmAppendResult {
    pub payload_digest: Digest32,
    pub geist_projection_candidate_digest: Digest32,
    pub geist_projection_audit_digest: Digest32,
    pub ism_candidate_boundary_digest: Digest32,
    pub sleep_plan_audit_digest: Digest32,
    pub sleep_plan_candidate_digest: Digest32,
    pub sleep_applied_boundary_digest: Option<Digest32>,
    pub replay_audit_digest: Digest32,
    pub replay_schedule_digest: Digest32,
    pub token_count: u32,
    pub appended_evidence_id: EvidenceId,
    pub archive_key: Digest32,
    pub archive_record_digest: Digest32,
    pub readback_digest: Digest32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeistIsmAppendError {
    InvalidInput(&'static str),
    ReadbackMissing,
    ReadbackMismatch,
}

impl std::fmt::Display for GeistIsmAppendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(message) => {
                write!(f, "invalid minimal spine Geist/ISM append input: {message}")
            }
            Self::ReadbackMissing => f.write_str("minimal spine Geist/ISM append readback missing"),
            Self::ReadbackMismatch => {
                f.write_str("minimal spine Geist/ISM append readback mismatch")
            }
        }
    }
}

impl std::error::Error for GeistIsmAppendError {}

/// Explicitly append a bounded Geist/ISM audit/provenance payload through Evidence/Archive.
///
/// This helper is the only append surface for the bounded Geist/ISM contract. It persists a
/// deterministic payload and immediately reads back both stores. It does not apply Geist, call
/// `GeistKernel::ingest_macro`, write/upsert ISM, create identity anchors, finalize identity,
/// stabilize memory, mutate policy, expose Gateway/action authority, or create a second event log.
pub fn append_minimal_spine_geist_ism_record<E, S>(
    candidate: &MinimalSpineGeistProjectionCandidate,
    audit: &MinimalSpineGeistProjectionAudit,
    boundary: &MinimalSpineIsmCandidateBoundary,
    evidence_store: &E,
    archive_store: &S,
    archive_appender: &mut ArchiveAppender,
) -> Result<MinimalSpineGeistIsmAppendResult, GeistIsmAppendError>
where
    E: EvidenceStore,
    S: ArchiveStore,
{
    let payload = MinimalSpineGeistIsmAppendPayload::from_artifacts(candidate, audit, boundary)?;
    let payload_bytes = payload.deterministic_bytes();
    let payload_digest = payload.digest();
    let appended_evidence_id = EvidenceId::new(format!(
        "minimal-spine-geist-ism-append-{}",
        hex_encode(payload_digest.as_bytes())
    ));
    let proof = ProofEnvelope {
        envelope_id: format!(
            "{}:{}",
            MINIMAL_SPINE_GEIST_ISM_APPEND_CONTRACT,
            hex_encode(payload_digest.as_bytes())
        ),
        payload: payload_bytes,
        payload_digest: Some(ProtoDigest {
            algorithm: "blake3".to_string(),
            value: payload_digest.as_bytes().to_vec(),
            algo_id: None,
            domain: None,
            value_32: Some(payload_digest.as_bytes().to_vec()),
        }),
        vrf_tags: Vec::new(),
        signature_ids: Vec::new(),
    };
    let evidence_envelope = EvidenceEnvelope {
        evidence_id: appended_evidence_id.clone(),
        proof: Some(proof),
        fold_proof: None,
        logical_time: LogicalTime::new(0),
        wall_time: WallTime::new(0),
    };

    evidence_store.append(evidence_envelope);

    let archive_record = archive_appender.build_record_with_commit(
        MINIMAL_SPINE_GEIST_ISM_APPEND_ARCHIVE_KIND,
        payload_digest,
        RecordMeta {
            cycle_id: 0,
            tier: 3,
            flags: 0,
            boundary_commit: boundary.ism_candidate_digest,
        },
    );
    let archive_record_digest = archive_store.append(archive_record);
    let readback = evidence_store
        .get(appended_evidence_id.clone())
        .ok_or(GeistIsmAppendError::ReadbackMissing)?;
    let readback_digest = digest_geist_ism_evidence_envelope(&readback);
    let archive_readback = archive_store
        .get(archive_record.key)
        .ok_or(GeistIsmAppendError::ReadbackMissing)?;
    if archive_readback != archive_record {
        return Err(GeistIsmAppendError::ReadbackMismatch);
    }

    Ok(MinimalSpineGeistIsmAppendResult {
        payload_digest,
        geist_projection_candidate_digest: payload.geist_projection_candidate_digest,
        geist_projection_audit_digest: payload.geist_projection_audit_digest,
        ism_candidate_boundary_digest: payload.ism_candidate_boundary_digest,
        sleep_plan_audit_digest: payload.sleep_plan_audit_digest,
        sleep_plan_candidate_digest: payload.sleep_plan_candidate_digest,
        sleep_applied_boundary_digest: payload.sleep_applied_boundary_digest,
        replay_audit_digest: payload.replay_audit_digest,
        replay_schedule_digest: payload.replay_schedule_digest,
        token_count: payload.token_count,
        appended_evidence_id,
        archive_key: archive_record.key,
        archive_record_digest,
        readback_digest,
    })
}

fn validate_minimal_spine_geist_ism_append_inputs(
    candidate: &MinimalSpineGeistProjectionCandidate,
    audit: &MinimalSpineGeistProjectionAudit,
    boundary: &MinimalSpineIsmCandidateBoundary,
) -> Result<(), GeistIsmAppendError> {
    if audit.status != MinimalSpineGeistProjectionAuditStatus::Pass {
        return Err(GeistIsmAppendError::InvalidInput(
            "geist projection audit must pass",
        ));
    }
    if audit.audit_digest != audit.digest() {
        return Err(GeistIsmAppendError::InvalidInput(
            "geist projection audit digest mismatch",
        ));
    }
    if !audit.failure_reasons.is_empty() {
        return Err(GeistIsmAppendError::InvalidInput(
            "pass audit must not carry failure reasons",
        ));
    }
    if candidate.projection_digest != candidate.digest() {
        return Err(GeistIsmAppendError::InvalidInput(
            "candidate projection digest mismatch",
        ));
    }
    if audit.projection_digest != candidate.projection_digest
        || audit.recomputed_projection_digest != candidate.projection_digest
    {
        return Err(GeistIsmAppendError::InvalidInput(
            "audit/candidate projection mismatch",
        ));
    }
    if boundary.ism_candidate_digest != boundary.digest() {
        return Err(GeistIsmAppendError::InvalidInput(
            "ISM candidate boundary digest mismatch",
        ));
    }
    if boundary.geist_projection_audit_digest != audit.audit_digest
        || boundary.geist_projection_digest != candidate.projection_digest
    {
        return Err(GeistIsmAppendError::InvalidInput(
            "boundary/audit projection mismatch",
        ));
    }
    if audit.sleep_plan_audit_digest != candidate.sleep_plan_audit_digest
        || boundary.sleep_plan_audit_digest != candidate.sleep_plan_audit_digest
        || audit.sleep_plan_candidate_digest != candidate.sleep_plan_candidate_digest
        || boundary.sleep_plan_candidate_digest != candidate.sleep_plan_candidate_digest
        || audit.sleep_applied_boundary_digest != candidate.sleep_applied_boundary_digest
        || boundary.sleep_applied_boundary_digest != candidate.sleep_applied_boundary_digest
        || audit.replay_audit_digest != candidate.replay_audit_digest
        || boundary.replay_audit_digest != candidate.replay_audit_digest
        || audit.replay_schedule_digest != candidate.replay_schedule_digest
        || boundary.replay_schedule_digest != candidate.replay_schedule_digest
        || audit.token_count != candidate.token_count
        || boundary.token_count != candidate.token_count
    {
        return Err(GeistIsmAppendError::InvalidInput("provenance mismatch"));
    }
    if is_zero_digest(candidate.projection_digest)
        || is_zero_digest(audit.audit_digest)
        || is_zero_digest(boundary.ism_candidate_digest)
        || is_zero_digest(candidate.sleep_plan_audit_digest)
        || is_zero_digest(candidate.sleep_plan_candidate_digest)
        || candidate
            .sleep_applied_boundary_digest
            .is_some_and(is_zero_digest)
        || is_zero_digest(candidate.replay_audit_digest)
        || is_zero_digest(candidate.replay_schedule_digest)
    {
        return Err(GeistIsmAppendError::InvalidInput(
            "append inputs must use non-zero digests",
        ));
    }
    if candidate.token_count == 0 {
        return Err(GeistIsmAppendError::InvalidInput(
            "token count must be non-zero",
        ));
    }
    if candidate.source.is_empty()
        || audit.source.is_empty()
        || boundary.source.is_empty()
        || candidate.sleep_source.is_empty()
        || audit.candidate_source.is_empty()
        || boundary.audit_source.is_empty()
    {
        return Err(GeistIsmAppendError::InvalidInput(
            "append sources must be non-empty",
        ));
    }
    if !candidate.candidate_only
        || !audit.candidate_only
        || !boundary.ism_candidate_only
        || candidate.geist_applied
        || audit.geist_applied
        || candidate.ism_written
        || audit.ism_written
        || boundary.ism_written
        || boundary.ism_upserted
        || candidate.identity_anchor
        || audit.identity_anchor
        || boundary.identity_anchor
        || candidate.identity_finalized
        || audit.identity_finalized
        || boundary.identity_finalized
        || boundary.memory_stabilized
        || candidate.policy_mutated
        || audit.policy_mutated
        || boundary.policy_mutated
        || candidate.evidence_archive_appended
        || audit.evidence_archive_appended
        || boundary.evidence_archive_appended
        || candidate.gateway_visible
        || audit.gateway_visible
        || boundary.gateway_visible
    {
        return Err(GeistIsmAppendError::InvalidInput(
            "append inputs must not carry forbidden side-effect flags",
        ));
    }
    Ok(())
}

fn digest_geist_ism_evidence_envelope(envelope: &EvidenceEnvelope) -> Digest32 {
    let mut out = Vec::new();
    push_str32(&mut out, envelope.evidence_id.as_str());
    match &envelope.proof {
        Some(proof) => {
            out.push(1);
            push_str32(&mut out, &proof.envelope_id);
            push_u32_be(
                &mut out,
                u32::try_from(proof.payload.len())
                    .expect("minimal spine Geist/ISM append proof payload length fits u32"),
            );
            out.extend_from_slice(&proof.payload);
            match &proof.payload_digest {
                Some(digest) => {
                    out.push(1);
                    push_str32(&mut out, &digest.algorithm);
                    push_u32_be(
                        &mut out,
                        u32::try_from(digest.value.len())
                            .expect("minimal spine Geist/ISM append digest length fits u32"),
                    );
                    out.extend_from_slice(&digest.value);
                    push_optional_u32_be(&mut out, digest.algo_id);
                    push_optional_u32_be(&mut out, digest.domain);
                    push_optional_bytes32(&mut out, digest.value_32.as_deref());
                }
                None => out.push(0),
            }
            push_u32_be(
                &mut out,
                u32::try_from(proof.vrf_tags.len())
                    .expect("minimal spine Geist/ISM append vrf tag count fits u32"),
            );
            push_u32_be(
                &mut out,
                u32::try_from(proof.signature_ids.len())
                    .expect("minimal spine Geist/ISM append signature id count fits u32"),
            );
            for signature_id in &proof.signature_ids {
                push_str32(&mut out, signature_id);
            }
        }
        None => out.push(0),
    }
    out.push(u8::from(envelope.fold_proof.is_some()));
    out.extend_from_slice(&envelope.logical_time.tick.to_be_bytes());
    out.extend_from_slice(&envelope.wall_time.unix_ms.to_be_bytes());
    digest_blake3_domain(
        b"ucf.geist.minimal_spine.geist_ism.append_readback.v1",
        &out,
    )
}

fn push_optional_u32_be(out: &mut Vec<u8>, value: Option<u32>) {
    match value {
        Some(value) => {
            out.push(1);
            push_u32_be(out, value);
        }
        None => out.push(0),
    }
}

fn push_optional_bytes32(out: &mut Vec<u8>, value: Option<&[u8]>) {
    match value {
        Some(value) => {
            out.push(1);
            push_u32_be(
                out,
                u32::try_from(value.len())
                    .expect("minimal spine Geist/ISM append optional digest length fits u32"),
            );
            out.extend_from_slice(value);
        }
        None => out.push(0),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IsmCandidateBoundaryError {
    AuditStatusNotPass,
    AuditDigestMismatch,
    AuditHasFailureReasons,
    AuditProjectionDigestMismatch,
    AuditHasForbiddenBoundaryFlag,
    ZeroGeistProjectionAuditDigest,
    ZeroGeistProjectionDigest,
    ZeroSleepPlanAuditDigest,
    ZeroSleepPlanCandidateDigest,
    ZeroSleepAppliedBoundaryDigest,
    ZeroReplayAuditDigest,
    ZeroReplayScheduleDigest,
    ZeroTokenCount,
    EmptySource,
    EmptyCandidateSource,
    EmptySleepSource,
}

impl std::fmt::Display for IsmCandidateBoundaryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::AuditStatusNotPass => "geist projection audit status must be pass",
            Self::AuditDigestMismatch => "geist projection audit digest mismatch",
            Self::AuditHasFailureReasons => {
                "pass geist projection audit must not have failure reasons"
            }
            Self::AuditProjectionDigestMismatch => {
                "geist projection audit projection digest mismatch"
            }
            Self::AuditHasForbiddenBoundaryFlag => {
                "geist projection audit has a forbidden side-effect flag set"
            }
            Self::ZeroGeistProjectionAuditDigest => {
                "geist projection audit digest must be non-zero"
            }
            Self::ZeroGeistProjectionDigest => "geist projection digest must be non-zero",
            Self::ZeroSleepPlanAuditDigest => "sleep plan audit digest must be non-zero",
            Self::ZeroSleepPlanCandidateDigest => "sleep plan candidate digest must be non-zero",
            Self::ZeroSleepAppliedBoundaryDigest => {
                "sleep applied boundary digest must be non-zero"
            }
            Self::ZeroReplayAuditDigest => "replay audit digest must be non-zero",
            Self::ZeroReplayScheduleDigest => "replay schedule digest must be non-zero",
            Self::ZeroTokenCount => "token count must be non-zero",
            Self::EmptySource => "audit source must be non-empty",
            Self::EmptyCandidateSource => "candidate source must be non-empty",
            Self::EmptySleepSource => "sleep source must be non-empty",
        };
        f.write_str(message)
    }
}

impl std::error::Error for IsmCandidateBoundaryError {}

/// Build a deterministic local ISM candidate/read-model boundary from a PASS Geist projection audit.
///
/// This function is pure and local: it takes no `IsmStore`, does not call `upsert_anchor`, does not
/// create an identity anchor, does not finalize identity, does not stabilize memory, does not mutate
/// policy, does not append Evidence/Archive records, and does not expose Gateway or runtime authority.
pub fn build_ism_candidate_boundary_from_geist_audit(
    audit: &MinimalSpineGeistProjectionAudit,
) -> Result<MinimalSpineIsmCandidateBoundary, IsmCandidateBoundaryError> {
    validate_geist_projection_audit_for_ism_candidate_boundary(audit)?;

    let mut boundary = MinimalSpineIsmCandidateBoundary {
        version: MINIMAL_SPINE_ISM_CANDIDATE_BOUNDARY_VERSION,
        geist_projection_audit_digest: audit.audit_digest,
        geist_projection_digest: audit.projection_digest,
        sleep_plan_audit_digest: audit.sleep_plan_audit_digest,
        sleep_plan_candidate_digest: audit.sleep_plan_candidate_digest,
        sleep_applied_boundary_digest: audit.sleep_applied_boundary_digest,
        replay_audit_digest: audit.replay_audit_digest,
        replay_schedule_digest: audit.replay_schedule_digest,
        token_count: audit.token_count,
        ism_candidate_digest: Digest32::new([0u8; 32]),
        source: MINIMAL_SPINE_ISM_CANDIDATE_BOUNDARY_SOURCE,
        audit_source: audit.source,
        candidate_source: audit.candidate_source,
        sleep_source: audit.sleep_source,
        ism_candidate_only: true,
        ism_written: false,
        ism_upserted: false,
        identity_anchor: false,
        identity_finalized: false,
        memory_stabilized: false,
        policy_mutated: false,
        evidence_archive_appended: false,
        gateway_visible: false,
    };
    boundary.ism_candidate_digest = boundary.digest();
    Ok(boundary)
}

fn validate_geist_projection_audit_for_ism_candidate_boundary(
    audit: &MinimalSpineGeistProjectionAudit,
) -> Result<(), IsmCandidateBoundaryError> {
    if audit.status != MinimalSpineGeistProjectionAuditStatus::Pass {
        return Err(IsmCandidateBoundaryError::AuditStatusNotPass);
    }
    if audit.audit_digest != audit.digest() {
        return Err(IsmCandidateBoundaryError::AuditDigestMismatch);
    }
    if !audit.failure_reasons.is_empty() {
        return Err(IsmCandidateBoundaryError::AuditHasFailureReasons);
    }
    if audit.projection_digest != audit.recomputed_projection_digest {
        return Err(IsmCandidateBoundaryError::AuditProjectionDigestMismatch);
    }
    if !audit.candidate_only
        || audit.geist_applied
        || audit.ism_written
        || audit.identity_anchor
        || audit.identity_finalized
        || audit.policy_mutated
        || audit.evidence_archive_appended
        || audit.gateway_visible
    {
        return Err(IsmCandidateBoundaryError::AuditHasForbiddenBoundaryFlag);
    }
    if is_zero_digest(audit.audit_digest) {
        return Err(IsmCandidateBoundaryError::ZeroGeistProjectionAuditDigest);
    }
    if is_zero_digest(audit.projection_digest) {
        return Err(IsmCandidateBoundaryError::ZeroGeistProjectionDigest);
    }
    if is_zero_digest(audit.sleep_plan_audit_digest) {
        return Err(IsmCandidateBoundaryError::ZeroSleepPlanAuditDigest);
    }
    if is_zero_digest(audit.sleep_plan_candidate_digest) {
        return Err(IsmCandidateBoundaryError::ZeroSleepPlanCandidateDigest);
    }
    if let Some(digest) = audit.sleep_applied_boundary_digest {
        if is_zero_digest(digest) {
            return Err(IsmCandidateBoundaryError::ZeroSleepAppliedBoundaryDigest);
        }
    }
    if is_zero_digest(audit.replay_audit_digest) {
        return Err(IsmCandidateBoundaryError::ZeroReplayAuditDigest);
    }
    if is_zero_digest(audit.replay_schedule_digest) {
        return Err(IsmCandidateBoundaryError::ZeroReplayScheduleDigest);
    }
    if audit.token_count == 0 {
        return Err(IsmCandidateBoundaryError::ZeroTokenCount);
    }
    if audit.source.is_empty() {
        return Err(IsmCandidateBoundaryError::EmptySource);
    }
    if audit.candidate_source.is_empty() {
        return Err(IsmCandidateBoundaryError::EmptyCandidateSource);
    }
    if audit.sleep_source.is_empty() {
        return Err(IsmCandidateBoundaryError::EmptySleepSource);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeistProjectionError {
    AuditStatusNotPass,
    AuditDigestMismatch,
    AuditHasFailureReasons,
    AuditCandidateDigestMismatch,
    AuditHasForbiddenBoundaryFlag,
    BoundaryDigestMismatch,
    BoundaryAuditDigestMismatch,
    BoundaryCandidateDigestMismatch,
    BoundaryReplayAuditDigestMismatch,
    BoundaryReplayScheduleDigestMismatch,
    BoundaryTokenCountMismatch,
    BoundaryHasForbiddenSideEffectFlag,
    ZeroSleepPlanAuditDigest,
    ZeroSleepPlanCandidateDigest,
    ZeroSleepAppliedBoundaryDigest,
    ZeroReplayAuditDigest,
    ZeroReplayScheduleDigest,
    ZeroTokenCount,
    EmptySource,
    EmptySleepSource,
}

impl std::fmt::Display for GeistProjectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::AuditStatusNotPass => "sleep plan audit status must be pass",
            Self::AuditDigestMismatch => "sleep plan audit digest mismatch",
            Self::AuditHasFailureReasons => "pass sleep plan audit must not have failure reasons",
            Self::AuditCandidateDigestMismatch => "sleep plan audit candidate digest mismatch",
            Self::AuditHasForbiddenBoundaryFlag => {
                "sleep plan audit has a forbidden side-effect flag set"
            }
            Self::BoundaryDigestMismatch => "sleep applied boundary digest mismatch",
            Self::BoundaryAuditDigestMismatch => "sleep applied boundary audit digest mismatch",
            Self::BoundaryCandidateDigestMismatch => {
                "sleep applied boundary candidate digest mismatch"
            }
            Self::BoundaryReplayAuditDigestMismatch => {
                "sleep applied boundary replay audit digest mismatch"
            }
            Self::BoundaryReplayScheduleDigestMismatch => {
                "sleep applied boundary replay schedule digest mismatch"
            }
            Self::BoundaryTokenCountMismatch => "sleep applied boundary token count mismatch",
            Self::BoundaryHasForbiddenSideEffectFlag => {
                "sleep applied boundary has a forbidden side-effect flag set"
            }
            Self::ZeroSleepPlanAuditDigest => "sleep plan audit digest must be non-zero",
            Self::ZeroSleepPlanCandidateDigest => "sleep plan candidate digest must be non-zero",
            Self::ZeroSleepAppliedBoundaryDigest => {
                "sleep applied boundary digest must be non-zero"
            }
            Self::ZeroReplayAuditDigest => "replay audit digest must be non-zero",
            Self::ZeroReplayScheduleDigest => "replay schedule digest must be non-zero",
            Self::ZeroTokenCount => "token count must be non-zero",
            Self::EmptySource => "source must be non-empty",
            Self::EmptySleepSource => "sleep source must be non-empty",
        };
        f.write_str(message)
    }
}

impl std::error::Error for GeistProjectionError {}

/// Verify a Minimal Spine Geist projection candidate without applying Geist.
///
/// The audit is local and deterministic. It recomputes the projection digest, preserves Sleep and
/// Replay provenance, checks candidate-only and forbidden side-effect flags, takes no
/// `GeistKernel`/ISM/policy/Gateway/Evidence/Archive handles, and does not mutate the candidate.
pub fn verify_minimal_spine_geist_projection_candidate(
    candidate: &MinimalSpineGeistProjectionCandidate,
) -> MinimalSpineGeistProjectionAudit {
    let recomputed_projection_digest = candidate.digest();
    let mut reasons = Vec::new();

    if candidate.version != MINIMAL_SPINE_GEIST_PROJECTION_CANDIDATE_VERSION {
        reasons.push(MinimalSpineGeistProjectionAuditFailure::VersionMismatch);
    }
    if candidate.projection_digest != recomputed_projection_digest {
        reasons.push(MinimalSpineGeistProjectionAuditFailure::ProjectionDigestMismatch);
    }
    if is_zero_digest(candidate.sleep_plan_audit_digest) {
        reasons.push(MinimalSpineGeistProjectionAuditFailure::ZeroSleepPlanAuditDigest);
    }
    if is_zero_digest(candidate.sleep_plan_candidate_digest) {
        reasons.push(MinimalSpineGeistProjectionAuditFailure::ZeroSleepPlanCandidateDigest);
    }
    if candidate
        .sleep_applied_boundary_digest
        .is_some_and(is_zero_digest)
    {
        reasons.push(MinimalSpineGeistProjectionAuditFailure::ZeroSleepAppliedBoundaryDigest);
    }
    if is_zero_digest(candidate.replay_audit_digest) {
        reasons.push(MinimalSpineGeistProjectionAuditFailure::ZeroReplayAuditDigest);
    }
    if is_zero_digest(candidate.replay_schedule_digest) {
        reasons.push(MinimalSpineGeistProjectionAuditFailure::ZeroReplayScheduleDigest);
    }
    if candidate.token_count == 0 {
        reasons.push(MinimalSpineGeistProjectionAuditFailure::InvalidTokenCount);
    }
    if candidate.source.is_empty() {
        reasons.push(MinimalSpineGeistProjectionAuditFailure::EmptySource);
    }
    if candidate.sleep_source.is_empty() {
        reasons.push(MinimalSpineGeistProjectionAuditFailure::EmptySleepSource);
    }
    if !candidate.candidate_only {
        reasons.push(MinimalSpineGeistProjectionAuditFailure::NotCandidateOnly);
    }
    if candidate.geist_applied {
        reasons.push(MinimalSpineGeistProjectionAuditFailure::GeistAppliedFlagSet);
    }
    if candidate.ism_written {
        reasons.push(MinimalSpineGeistProjectionAuditFailure::IsmWrittenFlagSet);
    }
    if candidate.identity_anchor {
        reasons.push(MinimalSpineGeistProjectionAuditFailure::IdentityAnchorFlagSet);
    }
    if candidate.identity_finalized {
        reasons.push(MinimalSpineGeistProjectionAuditFailure::IdentityFinalizedFlagSet);
    }
    if candidate.policy_mutated {
        reasons.push(MinimalSpineGeistProjectionAuditFailure::PolicyMutatedFlagSet);
    }
    if candidate.evidence_archive_appended {
        reasons.push(MinimalSpineGeistProjectionAuditFailure::EvidenceArchiveAppendedFlagSet);
    }
    if candidate.gateway_visible {
        reasons.push(MinimalSpineGeistProjectionAuditFailure::GatewayVisibleFlagSet);
    }

    reasons.sort_unstable();
    reasons.dedup();
    let status = if reasons.is_empty() {
        MinimalSpineGeistProjectionAuditStatus::Pass
    } else {
        MinimalSpineGeistProjectionAuditStatus::Fail
    };

    let mut audit = MinimalSpineGeistProjectionAudit {
        version: MINIMAL_SPINE_GEIST_PROJECTION_AUDIT_VERSION,
        status,
        failure_reasons: reasons,
        projection_digest: candidate.projection_digest,
        recomputed_projection_digest,
        sleep_plan_audit_digest: candidate.sleep_plan_audit_digest,
        sleep_plan_candidate_digest: candidate.sleep_plan_candidate_digest,
        sleep_applied_boundary_digest: candidate.sleep_applied_boundary_digest,
        replay_audit_digest: candidate.replay_audit_digest,
        replay_schedule_digest: candidate.replay_schedule_digest,
        token_count: candidate.token_count,
        audit_digest: Digest32::new([0u8; Digest32::LEN]),
        source: MINIMAL_SPINE_GEIST_PROJECTION_AUDIT_SOURCE,
        candidate_source: candidate.source,
        sleep_source: candidate.sleep_source,
        candidate_only: candidate.candidate_only,
        geist_applied: false,
        ism_written: false,
        identity_anchor: false,
        identity_finalized: false,
        policy_mutated: false,
        evidence_archive_appended: false,
        gateway_visible: false,
    };
    audit.audit_digest = audit.digest();
    audit
}

/// Build a pure candidate-only Geist projection from bounded Sleep digests.
///
/// The function is deterministic and takes no store/appender/Gateway/GeistKernel/ISM/policy
/// mutation/scheduler handles. It does not apply Geist, write or upsert ISM, create identity
/// anchors, finalize identity, mutate policy, append Evidence/Archive records, expose a Gateway
/// value, or recurse.
pub fn build_geist_projection_candidate_from_sleep_input(
    input: &MinimalSpineGeistProjectionInput,
) -> Result<MinimalSpineGeistProjectionCandidate, GeistProjectionError> {
    validate_geist_projection_input(input)?;

    let mut candidate = MinimalSpineGeistProjectionCandidate {
        version: MINIMAL_SPINE_GEIST_PROJECTION_CANDIDATE_VERSION,
        sleep_plan_audit_digest: input.sleep_plan_audit_digest,
        sleep_plan_candidate_digest: input.sleep_plan_candidate_digest,
        sleep_applied_boundary_digest: input.sleep_applied_boundary_digest,
        replay_audit_digest: input.replay_audit_digest,
        replay_schedule_digest: input.replay_schedule_digest,
        token_count: input.token_count,
        projection_digest: Digest32::new([0u8; Digest32::LEN]),
        source: MINIMAL_SPINE_GEIST_PROJECTION_CANDIDATE_SOURCE,
        sleep_source: input.source,
        candidate_only: true,
        geist_applied: false,
        ism_written: false,
        identity_anchor: false,
        identity_finalized: false,
        policy_mutated: false,
        evidence_archive_appended: false,
        gateway_visible: false,
    };
    candidate.projection_digest = candidate.digest();
    Ok(candidate)
}

/// Build a pure candidate-only Geist projection from a PASS SleepPlan audit and optional local
/// SleepApplied boundary marker.
///
/// The optional boundary is provenance only. If supplied, it must match the audit digest,
/// candidate digest, replay digests, and token count. This does not activate the existing broad
/// `GeistKernel::ingest_macro` path and does not write ISM state.
pub fn build_geist_projection_candidate_from_sleep_audit(
    audit: &MinimalSpineSleepPlanAudit,
    sleep_boundary: Option<&MinimalSpineSleepAppliedBoundary>,
) -> Result<MinimalSpineGeistProjectionCandidate, GeistProjectionError> {
    validate_sleep_plan_audit_for_geist_projection(audit)?;
    if let Some(boundary) = sleep_boundary {
        validate_sleep_applied_boundary_for_geist_projection(audit, boundary)?;
    }

    let input = MinimalSpineGeistProjectionInput {
        sleep_plan_audit_digest: audit.audit_digest,
        sleep_plan_candidate_digest: audit.sleep_plan_candidate_digest,
        sleep_applied_boundary_digest: sleep_boundary
            .map(|boundary| boundary.applied_boundary_digest),
        replay_audit_digest: audit.replay_audit_digest,
        replay_schedule_digest: audit.replay_schedule_digest,
        token_count: audit.token_count,
        source: audit.source,
    };
    build_geist_projection_candidate_from_sleep_input(&input)
}

fn validate_sleep_plan_audit_for_geist_projection(
    audit: &MinimalSpineSleepPlanAudit,
) -> Result<(), GeistProjectionError> {
    if audit.status != MinimalSpineSleepPlanAuditStatus::Pass {
        return Err(GeistProjectionError::AuditStatusNotPass);
    }
    if audit.audit_digest != audit.digest() {
        return Err(GeistProjectionError::AuditDigestMismatch);
    }
    if !audit.failure_reasons.is_empty() {
        return Err(GeistProjectionError::AuditHasFailureReasons);
    }
    if audit.sleep_plan_candidate_digest != audit.recomputed_sleep_plan_candidate_digest {
        return Err(GeistProjectionError::AuditCandidateDigestMismatch);
    }
    if !audit.candidate_only
        || audit.sleep_applied
        || audit.sleep_completed
        || audit.geist_ingested
        || audit.ism_written
        || audit.identity_anchor
        || audit.evidence_archive_appended
        || audit.gateway_visible
    {
        return Err(GeistProjectionError::AuditHasForbiddenBoundaryFlag);
    }

    let input = MinimalSpineGeistProjectionInput {
        sleep_plan_audit_digest: audit.audit_digest,
        sleep_plan_candidate_digest: audit.sleep_plan_candidate_digest,
        sleep_applied_boundary_digest: None,
        replay_audit_digest: audit.replay_audit_digest,
        replay_schedule_digest: audit.replay_schedule_digest,
        token_count: audit.token_count,
        source: audit.source,
    };
    validate_geist_projection_input(&input)
}

fn validate_sleep_applied_boundary_for_geist_projection(
    audit: &MinimalSpineSleepPlanAudit,
    boundary: &MinimalSpineSleepAppliedBoundary,
) -> Result<(), GeistProjectionError> {
    if boundary.applied_boundary_digest != boundary.digest() {
        return Err(GeistProjectionError::BoundaryDigestMismatch);
    }
    if boundary.sleep_plan_audit_digest != audit.audit_digest {
        return Err(GeistProjectionError::BoundaryAuditDigestMismatch);
    }
    if boundary.sleep_plan_candidate_digest != audit.sleep_plan_candidate_digest {
        return Err(GeistProjectionError::BoundaryCandidateDigestMismatch);
    }
    if boundary.replay_audit_digest != audit.replay_audit_digest {
        return Err(GeistProjectionError::BoundaryReplayAuditDigestMismatch);
    }
    if boundary.replay_schedule_digest != audit.replay_schedule_digest {
        return Err(GeistProjectionError::BoundaryReplayScheduleDigestMismatch);
    }
    if boundary.token_count != audit.token_count {
        return Err(GeistProjectionError::BoundaryTokenCountMismatch);
    }
    if boundary.sleep_completed
        || boundary.coordinator_runtime_triggered
        || boundary.geist_ingested
        || boundary.ism_written
        || boundary.identity_anchor
        || boundary.memory_stabilized
        || boundary.evidence_archive_appended
        || boundary.gateway_visible
    {
        return Err(GeistProjectionError::BoundaryHasForbiddenSideEffectFlag);
    }
    if is_zero_digest(boundary.applied_boundary_digest) {
        return Err(GeistProjectionError::ZeroSleepAppliedBoundaryDigest);
    }
    Ok(())
}

fn validate_geist_projection_input(
    input: &MinimalSpineGeistProjectionInput,
) -> Result<(), GeistProjectionError> {
    if is_zero_digest(input.sleep_plan_audit_digest) {
        return Err(GeistProjectionError::ZeroSleepPlanAuditDigest);
    }
    if is_zero_digest(input.sleep_plan_candidate_digest) {
        return Err(GeistProjectionError::ZeroSleepPlanCandidateDigest);
    }
    if let Some(digest) = input.sleep_applied_boundary_digest {
        if is_zero_digest(digest) {
            return Err(GeistProjectionError::ZeroSleepAppliedBoundaryDigest);
        }
    }
    if is_zero_digest(input.replay_audit_digest) {
        return Err(GeistProjectionError::ZeroReplayAuditDigest);
    }
    if is_zero_digest(input.replay_schedule_digest) {
        return Err(GeistProjectionError::ZeroReplayScheduleDigest);
    }
    if input.token_count == 0 {
        return Err(GeistProjectionError::ZeroTokenCount);
    }
    if input.source.is_empty() {
        return Err(GeistProjectionError::EmptySource);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GeistConfig {
    pub recursion_depth: u8,
    pub per_cycle_steps: u16,
    pub consistency_threshold: u16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeistLoopState {
    pub level: u8,
    pub anchor: Digest32,
    pub context: Vec<Digest32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplayStabilization {
    pub drift_reduction: u16,
    pub commit: Digest32,
}

pub trait IsmStore {
    fn anchors(&self) -> Vec<Digest32>;
    fn upsert_anchor(&mut self, anchor: Digest32);
}

#[derive(Clone, Debug, Default)]
pub struct InMemoryIsm {
    anchors: Vec<Digest32>,
}

impl InMemoryIsm {
    pub fn new() -> Self {
        Self {
            anchors: Vec::new(),
        }
    }

    fn normalize(&mut self) {
        normalize_digests(&mut self.anchors);
    }
}

impl IsmStore for InMemoryIsm {
    fn anchors(&self) -> Vec<Digest32> {
        let mut anchors = self.anchors.clone();
        normalize_digests(&mut anchors);
        anchors
    }

    fn upsert_anchor(&mut self, anchor: Digest32) {
        self.anchors.push(anchor);
        self.normalize();
    }
}

pub struct GeistKernel<A: ExperienceAppender, I: IsmStore> {
    pub cfg: GeistConfig,
    pub archive: A,
    pub ism: I,
    gate: Arc<dyn GeistGate + Send + Sync>,
    sleep_state: Option<SleepStateHandle>,
}

impl<A: ExperienceAppender, I: IsmStore> GeistKernel<A, I> {
    pub fn new(cfg: GeistConfig, archive: A, ism: I) -> Self {
        Self::new_with_gate_and_sleep(
            cfg,
            archive,
            ism,
            Arc::new(DefaultPolicyEcology::default()),
            None,
        )
    }

    pub fn new_with_gate(
        cfg: GeistConfig,
        archive: A,
        ism: I,
        gate: Arc<dyn GeistGate + Send + Sync>,
    ) -> Self {
        Self::new_with_gate_and_sleep(cfg, archive, ism, gate, None)
    }

    pub fn new_with_gate_and_sleep(
        cfg: GeistConfig,
        archive: A,
        ism: I,
        gate: Arc<dyn GeistGate + Send + Sync>,
        sleep_state: Option<SleepStateHandle>,
    ) -> Self {
        Self {
            cfg,
            archive,
            ism,
            gate,
            sleep_state,
        }
    }

    pub fn ingest_macro(
        &mut self,
        macro_ms: MacroMilestone,
    ) -> (Vec<GeistLoopState>, ConsistencyReport, EvidenceId) {
        let macro_refs = derive_macro_refs(&macro_ms);
        let self_states = build_self_states(self.cfg.recursion_depth, &macro_refs);
        let base_state = self_states.first().expect("recursion_depth must be >= 1");
        let mut report = compute_consistency_report(&self.cfg, base_state, &self.ism);
        if report.verdict == ConsistencyVerdict::Accept {
            if self.gate.allow_ism_upsert(&report) {
                self.ism.upsert_anchor(base_state.anchor);
            } else {
                report.verdict = ConsistencyVerdict::Damp;
            }
        }

        let record = derived_record_for_macro(&macro_ms, &macro_refs, &self_states, &report);
        let evidence_id = self.archive.append(record);
        if let Some(state) = &self.sleep_state {
            if let Ok(mut guard) = state.lock() {
                guard.record_consistency_verdict(report.verdict);
                guard.record_derived_record(evidence_id.clone());
            }
        }
        (self_states, report, evidence_id)
    }

    pub fn apply_replay_effects(&mut self, effects: &[ReplayApplied]) -> ReplayStabilization {
        let drift_reduction = replay_drift_reduction(effects);
        let commit = commit_replay_stabilization(effects, drift_reduction);
        ReplayStabilization {
            drift_reduction,
            commit,
        }
    }
}

fn derive_macro_refs(macro_ms: &MacroMilestone) -> Vec<Digest32> {
    let commitment = commit_milestone_macro(macro_ms);
    vec![commitment.digest]
}

fn replay_drift_reduction(effects: &[ReplayApplied]) -> u16 {
    if effects.is_empty() {
        return 0;
    }
    let mut reduction = 0u16;
    for effect in effects {
        let bytes = effect.effect_digest.as_bytes();
        let sample = u16::from_be_bytes([bytes[0], bytes[1]]);
        reduction = reduction.saturating_add(sample % 1200);
    }
    reduction.min(6000)
}

fn commit_replay_stabilization(effects: &[ReplayApplied], reduction: u16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.geist.replay.stabilization.v1");
    for effect in effects {
        hasher.update(&[effect.tier as u8]);
        hasher.update(effect.target.as_bytes());
        hasher.update(effect.effect_digest.as_bytes());
    }
    hasher.update(&reduction.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn build_self_states(recursion_depth: u8, macro_refs: &[Digest32]) -> Vec<GeistLoopState> {
    let mut states = Vec::with_capacity(recursion_depth as usize);
    let mut previous_anchor = None;
    for level in 1..=recursion_depth {
        let state = build_self_state(level, macro_refs, previous_anchor);
        previous_anchor = Some(state.anchor);
        states.push(state);
    }
    states
}

fn build_self_state(
    level: u8,
    macro_refs: &[Digest32],
    previous_anchor: Option<Digest32>,
) -> GeistLoopState {
    let mut context = macro_refs.to_vec();
    if let Some(anchor) = previous_anchor {
        context.push(anchor);
    }
    normalize_digests(&mut context);
    let anchor = hash_self_state(level, macro_refs, previous_anchor);
    GeistLoopState {
        level,
        anchor,
        context,
    }
}

fn hash_self_state(
    level: u8,
    macro_refs: &[Digest32],
    previous_anchor: Option<Digest32>,
) -> Digest32 {
    let mut refs = macro_refs.to_vec();
    normalize_digests(&mut refs);
    let mut enc = Encoder::new();
    enc.write_u16(SELFSTATE_DOMAIN);
    enc.write_u8(level);
    enc.write_u32(refs.len() as u32);
    for digest in refs {
        enc.write_digest(&digest);
    }
    match previous_anchor {
        Some(anchor) => {
            enc.write_u8(1);
            enc.write_digest(&anchor);
        }
        None => enc.write_u8(0),
    }
    hash_bytes(enc.finish())
}

fn compute_consistency_report(
    cfg: &GeistConfig,
    self_state: &GeistLoopState,
    ism: &impl IsmStore,
) -> ConsistencyReport {
    let ism_anchors = ism.anchors();
    let matched_anchors = self_state
        .context
        .iter()
        .filter(|anchor| ism_anchors.contains(anchor))
        .count();
    let score = matched_anchors.min(u16::MAX as usize) as u16;
    let verdict = if score >= cfg.consistency_threshold {
        ConsistencyVerdict::Accept
    } else if score > 0 {
        ConsistencyVerdict::Damp
    } else {
        ConsistencyVerdict::Reject
    };
    ConsistencyReport {
        score,
        verdict,
        matched_anchors,
    }
}

fn derived_record_for_macro(
    macro_ms: &MacroMilestone,
    macro_refs: &[Digest32],
    self_states: &[GeistLoopState],
    report: &ConsistencyReport,
) -> ExperienceRecord {
    let commitment = commit_milestone_macro(macro_ms);
    let payload = encode_derived_payload(macro_refs, self_states, report);
    ExperienceRecord {
        record_id: format!("geist-derived-{}", hex_encode(commitment.digest.as_bytes())),
        observed_at_ms: macro_ms.achieved_at_ms,
        subject_id: "geist".to_string(),
        payload,
        digest: None,
        vrf_tag: None,
        proof_ref: None,
    }
}

fn encode_derived_payload(
    macro_refs: &[Digest32],
    self_states: &[GeistLoopState],
    report: &ConsistencyReport,
) -> Vec<u8> {
    let mut refs = macro_refs.to_vec();
    normalize_digests(&mut refs);
    let mut enc = Encoder::new();
    enc.write_u16(DERIVED_DOMAIN);
    enc.write_u32(refs.len() as u32);
    for digest in refs {
        enc.write_digest(&digest);
    }
    enc.write_u32(self_states.len() as u32);
    for state in self_states {
        enc.write_u8(state.level);
        enc.write_digest(&state.anchor);
    }
    enc.write_u16(report.score);
    enc.write_u8(report.verdict.as_u8());
    enc.write_u32(report.matched_anchors.min(u32::MAX as usize) as u32);
    enc.finish().to_vec()
}

fn is_zero_digest(digest: Digest32) -> bool {
    digest.as_bytes().iter().all(|byte| *byte == 0)
}

fn push_digest32(out: &mut Vec<u8>, digest: Digest32) {
    out.extend_from_slice(digest.as_bytes());
}

fn push_u32_be(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn push_str32(out: &mut Vec<u8>, value: &str) {
    let len =
        u32::try_from(value.len()).expect("minimal spine geist projection source length fits u32");
    push_u32_be(out, len);
    out.extend_from_slice(value.as_bytes());
}

fn digest_blake3_domain(domain: &[u8], bytes: &[u8]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(domain);
    hasher.update(&[0]);
    hasher.update(bytes);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn hash_bytes(bytes: &[u8]) -> Digest32 {
    let digest = blake3::hash(bytes);
    Digest32::new(*digest.as_bytes())
}

fn normalize_digests(digests: &mut Vec<Digest32>) {
    digests.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
    digests.dedup();
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

struct Encoder {
    bytes: Vec<u8>,
}

impl Encoder {
    fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    fn write_u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn write_u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn write_u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn write_digest(&mut self, digest: &Digest32) {
        self.bytes.extend_from_slice(digest.as_bytes());
    }

    fn finish(&self) -> &[u8] {
        &self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use ucf_archive::InMemoryArchive;
    use ucf_policy_ecology::{PolicyEcology, PolicyRule, PolicyWeights};

    fn sample_macro(id: &str) -> MacroMilestone {
        MacroMilestone {
            milestone_id: id.to_string(),
            achieved_at_ms: 42,
            label: "macro".to_string(),
            meso_milestone_ids: vec!["meso-1".to_string()],
        }
    }

    #[test]
    fn determinism_same_macro_same_anchors() {
        let macro_ms = sample_macro("macro-1");
        let macro_refs = derive_macro_refs(&macro_ms);
        let states_a = build_self_states(3, &macro_refs);
        let states_b = build_self_states(3, &macro_refs);
        let anchors_a: Vec<Digest32> = states_a.iter().map(|state| state.anchor).collect();
        let anchors_b: Vec<Digest32> = states_b.iter().map(|state| state.anchor).collect();
        assert_eq!(anchors_a, anchors_b);
    }

    #[test]
    fn consistency_reports_accept_and_reject() {
        let cfg = GeistConfig {
            recursion_depth: 1,
            per_cycle_steps: 1,
            consistency_threshold: 1,
        };
        let macro_ms = sample_macro("macro-1");
        let macro_refs = derive_macro_refs(&macro_ms);
        let state = build_self_state(1, &macro_refs, None);
        let ism = InMemoryIsm::new();
        let report = compute_consistency_report(&cfg, &state, &ism);
        assert_eq!(report.verdict, ConsistencyVerdict::Reject);

        let mut ism = InMemoryIsm::new();
        ism.upsert_anchor(macro_refs[0]);
        let report = compute_consistency_report(&cfg, &state, &ism);
        assert_eq!(report.verdict, ConsistencyVerdict::Accept);
    }

    #[test]
    fn ingest_macro_appends_record_and_updates_ism() {
        let cfg = GeistConfig {
            recursion_depth: 2,
            per_cycle_steps: 4,
            consistency_threshold: 1,
        };
        let macro_ms = sample_macro("macro-1");
        let macro_refs = derive_macro_refs(&macro_ms);
        let mut ism = InMemoryIsm::new();
        ism.upsert_anchor(macro_refs[0]);
        let archive = InMemoryArchive::new();
        let mut kernel = GeistKernel::new(cfg, archive, ism);

        let (states, report, _evidence_id) = kernel.ingest_macro(macro_ms);
        assert_eq!(report.verdict, ConsistencyVerdict::Accept);
        assert_eq!(kernel.archive.list().len(), 1);
        assert!(kernel.ism.anchors().contains(&states[0].anchor));
    }

    #[test]
    fn ingest_macro_dampens_when_gate_denies_upsert() {
        let cfg = GeistConfig {
            recursion_depth: 1,
            per_cycle_steps: 1,
            consistency_threshold: 0,
        };
        let macro_ms = sample_macro("macro-2");
        let archive = InMemoryArchive::new();
        let ism = InMemoryIsm::new();
        let policy = PolicyEcology::new(
            1,
            vec![PolicyRule::DenyIsmUpsertIfScoreBelow { min_score: 1 }],
            PolicyWeights,
        );
        let mut kernel = GeistKernel::new_with_gate(cfg, archive, ism, Arc::new(policy));

        let (_states, report, _evidence_id) = kernel.ingest_macro(macro_ms);

        assert_eq!(report.verdict, ConsistencyVerdict::Damp);
        assert_eq!(kernel.ism.anchors().len(), 0);
        assert_eq!(kernel.archive.list().len(), 1);
    }

    #[test]
    fn self_state_builder_is_deterministic() {
        let state_a = SelfState::builder(42)
            .ssm_commit(Digest32::new([1u8; 32]))
            .workspace_commit(Digest32::new([2u8; 32]))
            .risk_commit(Digest32::new([3u8; 32]))
            .attn_commit(Digest32::new([4u8; 32]))
            .ncde_commit(Digest32::new([5u8; 32]))
            .consistency(9000)
            .build();
        let state_b = SelfState::builder(42)
            .ssm_commit(Digest32::new([1u8; 32]))
            .workspace_commit(Digest32::new([2u8; 32]))
            .risk_commit(Digest32::new([3u8; 32]))
            .attn_commit(Digest32::new([4u8; 32]))
            .ncde_commit(Digest32::new([5u8; 32]))
            .consistency(9000)
            .build();
        assert_eq!(state_a, state_b);
        assert_eq!(state_a.commit, state_b.commit);
    }
}
