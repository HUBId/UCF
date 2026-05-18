#![forbid(unsafe_code)]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use blake3::Hasher;
use ucf_archive_store::{ArchiveAppender, ArchiveStore, RecordKind, RecordMeta};
use ucf_bus::BusPublisher;
use ucf_evidence::{EvidenceEnvelope, EvidenceStore};
use ucf_policy_ecology::{ConsistencyVerdict, SleepPhaseGate};
use ucf_predictive_coding::SurpriseBand;
use ucf_replay::{
    MinimalSpineReplayAppliedBoundary, MinimalSpineReplayAuditStatus,
    MinimalSpineReplayScheduleAudit,
};
use ucf_rsa::{RsaEngine, SleepCoordinator, SleepReportReady};
use ucf_structural_store::{StructuralCycleStats, StructuralDeltaProposal};
use ucf_types::v1::spec::{Digest as ProtoDigest, ProofEnvelope};
use ucf_types::{Digest32, EvidenceId, LogicalTime, WallTime};

/// Version for local Minimal Spine SleepApplied boundary values.
pub const MINIMAL_SPINE_SLEEP_APPLIED_BOUNDARY_VERSION: u32 = 1;

/// Source marker for local SleepApplied boundary bookkeeping records.
pub const MINIMAL_SPINE_SLEEP_APPLIED_BOUNDARY_SOURCE: &str =
    "minimal_spine_v1_sleep_applied_local_boundary";

/// Version for verify-only Minimal Spine SleepPlan audit values.
pub const MINIMAL_SPINE_SLEEP_PLAN_AUDIT_VERSION: u32 = 1;

/// Source marker for local verify-only SleepPlan audit reports.
pub const MINIMAL_SPINE_SLEEP_PLAN_AUDIT_SOURCE: &str =
    "minimal_spine_v1_sleep_plan_verify_only_audit";

/// Version for deterministic Minimal Spine SleepPlan candidate values.
pub const MINIMAL_SPINE_SLEEP_PLAN_CANDIDATE_VERSION: u32 = 1;

/// Source marker for candidate-only SleepPlan values derived from bounded replay metadata.
pub const MINIMAL_SPINE_SLEEP_PLAN_CANDIDATE_SOURCE: &str =
    "minimal_spine_v1_sleep_plan_candidate_from_replay_boundary";

/// Digest/provenance input for a candidate-only Minimal Spine SleepPlan.
///
/// This value is replay-boundary metadata only. It carries no store, appender, Gateway, Geist,
/// ISM, scheduler, queue, worker, or runtime handle and cannot apply Sleep.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MinimalSpineSleepPlanInput {
    pub replay_audit_digest: Digest32,
    pub replay_schedule_digest: Digest32,
    pub replay_applied_boundary_digest: Option<Digest32>,
    pub token_count: u32,
    pub source: &'static str,
}

/// Deterministic, candidate-only SleepPlan value derived from bounded replay metadata.
///
/// This is not Sleep execution, not SleepApplied, not Sleep completion, not Geist/ISM ingestion,
/// not identity finalization, not Evidence/Archive append, and not Gateway visibility.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MinimalSpineSleepPlanCandidate {
    pub version: u32,
    pub replay_audit_digest: Digest32,
    pub replay_schedule_digest: Digest32,
    pub replay_applied_boundary_digest: Option<Digest32>,
    pub token_count: u32,
    pub source: &'static str,
    pub replay_source: &'static str,
    pub candidate_only: bool,
    pub sleep_applied: bool,
    pub sleep_completed: bool,
    pub geist_ingested: bool,
    pub ism_written: bool,
    pub identity_anchor: bool,
    pub evidence_archive_appended: bool,
    pub gateway_visible: bool,
    pub sleep_plan_digest: Digest32,
}

impl MinimalSpineSleepPlanCandidate {
    /// Deterministic bytes used for the candidate digest.
    pub fn deterministic_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_u32_be(&mut out, self.version);
        push_digest32(&mut out, self.replay_audit_digest);
        push_digest32(&mut out, self.replay_schedule_digest);
        match self.replay_applied_boundary_digest {
            Some(digest) => {
                out.push(1);
                push_digest32(&mut out, digest);
            }
            None => out.push(0),
        }
        push_u32_be(&mut out, self.token_count);
        push_str32(&mut out, self.source);
        push_str32(&mut out, self.replay_source);
        out.push(u8::from(self.candidate_only));
        out.push(u8::from(self.sleep_applied));
        out.push(u8::from(self.sleep_completed));
        out.push(u8::from(self.geist_ingested));
        out.push(u8::from(self.ism_written));
        out.push(u8::from(self.identity_anchor));
        out.push(u8::from(self.evidence_archive_appended));
        out.push(u8::from(self.gateway_visible));
        out
    }

    pub fn digest(&self) -> Digest32 {
        digest_blake3_domain(
            b"ucf.sleep.minimal_spine.plan_candidate_from_replay_boundary.v1",
            &self.deterministic_bytes(),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SleepPlanCandidateError {
    AuditStatusNotPass,
    AuditDigestMismatch,
    AuditHasFailureReasons,
    AuditScheduleDigestMismatch,
    AuditHasForbiddenBoundaryFlag,
    BoundaryDigestMismatch,
    BoundaryAuditDigestMismatch,
    BoundaryScheduleDigestMismatch,
    BoundaryTokenCountMismatch,
    BoundaryHasForbiddenSideEffectFlag,
    ZeroReplayAuditDigest,
    ZeroReplayScheduleDigest,
    ZeroReplayAppliedBoundaryDigest,
    ZeroTokenCount,
    EmptySource,
}

impl std::fmt::Display for SleepPlanCandidateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::AuditStatusNotPass => "replay audit status must be pass",
            Self::AuditDigestMismatch => "replay audit digest mismatch",
            Self::AuditHasFailureReasons => "pass replay audit must not have failure reasons",
            Self::AuditScheduleDigestMismatch => "replay audit schedule digest mismatch",
            Self::AuditHasForbiddenBoundaryFlag => "replay audit has a forbidden boundary flag set",
            Self::BoundaryDigestMismatch => "replay applied boundary digest mismatch",
            Self::BoundaryAuditDigestMismatch => "replay applied boundary audit digest mismatch",
            Self::BoundaryScheduleDigestMismatch => {
                "replay applied boundary schedule digest mismatch"
            }
            Self::BoundaryTokenCountMismatch => "replay applied boundary token count mismatch",
            Self::BoundaryHasForbiddenSideEffectFlag => {
                "replay applied boundary has a forbidden side-effect flag set"
            }
            Self::ZeroReplayAuditDigest => "replay audit digest must be non-zero",
            Self::ZeroReplayScheduleDigest => "replay schedule digest must be non-zero",
            Self::ZeroReplayAppliedBoundaryDigest => {
                "replay applied boundary digest must be non-zero"
            }
            Self::ZeroTokenCount => "token count must be non-zero",
            Self::EmptySource => "source must be non-empty",
        };
        f.write_str(message)
    }
}

impl std::error::Error for SleepPlanCandidateError {}

/// PASS/FAIL status for a verify-only Minimal Spine SleepPlan audit.
///
/// `Pass` means the candidate is internally consistent and all forbidden side-effect flags remain
/// false. It does not mean Sleep was applied, completed, archived, exposed, or ingested anywhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MinimalSpineSleepPlanAuditStatus {
    Pass,
    Fail,
}

impl MinimalSpineSleepPlanAuditStatus {
    fn code(self) -> u8 {
        match self {
            Self::Pass => 1,
            Self::Fail => 2,
        }
    }
}

/// Deterministic failure reasons emitted by the verify-only SleepPlan audit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MinimalSpineSleepPlanAuditFailure {
    VersionMismatch,
    CandidateDigestMismatch,
    ZeroReplayAuditDigest,
    ZeroReplayScheduleDigest,
    ZeroReplayAppliedBoundaryDigest,
    InvalidTokenCount,
    EmptySource,
    EmptyReplaySource,
    NotCandidateOnly,
    SleepAppliedFlagSet,
    SleepCompletedFlagSet,
    GeistIngestedFlagSet,
    IsmWrittenFlagSet,
    IdentityAnchorFlagSet,
    EvidenceArchiveAppendedFlagSet,
    GatewayVisibleFlagSet,
}

impl MinimalSpineSleepPlanAuditFailure {
    fn code(self) -> u8 {
        match self {
            Self::VersionMismatch => 1,
            Self::CandidateDigestMismatch => 2,
            Self::ZeroReplayAuditDigest => 3,
            Self::ZeroReplayScheduleDigest => 4,
            Self::ZeroReplayAppliedBoundaryDigest => 5,
            Self::InvalidTokenCount => 6,
            Self::EmptySource => 7,
            Self::EmptyReplaySource => 8,
            Self::NotCandidateOnly => 9,
            Self::SleepAppliedFlagSet => 10,
            Self::SleepCompletedFlagSet => 11,
            Self::GeistIngestedFlagSet => 12,
            Self::IsmWrittenFlagSet => 13,
            Self::IdentityAnchorFlagSet => 14,
            Self::EvidenceArchiveAppendedFlagSet => 15,
            Self::GatewayVisibleFlagSet => 16,
        }
    }
}

/// Local verify-only audit report for a Minimal Spine SleepPlan candidate.
///
/// The audit is a pure deterministic consistency check over a candidate value. It takes no runtime,
/// coordinator, WAL, journal, store, appender, Gateway, Geist, ISM, scheduler, queue, or worker
/// handle; it does not mutate the candidate; and all side-effect boundary flags in the report are
/// hard-coded false to prevent overclaiming.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MinimalSpineSleepPlanAudit {
    pub version: u32,
    pub status: MinimalSpineSleepPlanAuditStatus,
    pub failure_reasons: Vec<MinimalSpineSleepPlanAuditFailure>,
    pub sleep_plan_candidate_digest: Digest32,
    pub recomputed_sleep_plan_candidate_digest: Digest32,
    pub replay_audit_digest: Digest32,
    pub replay_schedule_digest: Digest32,
    pub replay_applied_boundary_digest: Option<Digest32>,
    pub token_count: u32,
    pub audit_digest: Digest32,
    pub source: &'static str,
    pub candidate_source: &'static str,
    pub replay_source: &'static str,
    pub candidate_only: bool,
    pub sleep_applied: bool,
    pub sleep_completed: bool,
    pub geist_ingested: bool,
    pub ism_written: bool,
    pub identity_anchor: bool,
    pub evidence_archive_appended: bool,
    pub gateway_visible: bool,
}

/// Local sleep-subsystem boundary marker derived from a PASS SleepPlan audit.
///
/// This record acknowledges only local Sleep bookkeeping inside `ucf-sleep-coordinator`. It is not
/// Sleep Cycle completion, not coordinator runtime execution, not Geist ingestion, not an ISM
/// write/upsert, not identity finalization or anchoring, not memory stabilization, not
/// Evidence/Archive append, and not Gateway visibility.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MinimalSpineSleepAppliedBoundary {
    pub version: u32,
    pub sleep_plan_audit_digest: Digest32,
    pub sleep_plan_candidate_digest: Digest32,
    pub replay_audit_digest: Digest32,
    pub replay_schedule_digest: Digest32,
    pub replay_applied_boundary_digest: Option<Digest32>,
    pub token_count: u32,
    pub applied_boundary_digest: Digest32,
    pub source: &'static str,
    pub sleep_plan_audit_source: &'static str,
    pub candidate_source: &'static str,
    pub replay_source: &'static str,
    pub sleep_subsystem_applied: bool,
    pub sleep_completed: bool,
    pub coordinator_runtime_triggered: bool,
    pub geist_ingested: bool,
    pub ism_written: bool,
    pub identity_anchor: bool,
    pub memory_stabilized: bool,
    pub evidence_archive_appended: bool,
    pub gateway_visible: bool,
}

impl MinimalSpineSleepAppliedBoundary {
    /// Deterministic bytes used for the local boundary digest.
    pub fn deterministic_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_u32_be(&mut out, self.version);
        push_digest32(&mut out, self.sleep_plan_audit_digest);
        push_digest32(&mut out, self.sleep_plan_candidate_digest);
        push_digest32(&mut out, self.replay_audit_digest);
        push_digest32(&mut out, self.replay_schedule_digest);
        match self.replay_applied_boundary_digest {
            Some(digest) => {
                out.push(1);
                push_digest32(&mut out, digest);
            }
            None => out.push(0),
        }
        push_u32_be(&mut out, self.token_count);
        push_str32(&mut out, self.source);
        push_str32(&mut out, self.sleep_plan_audit_source);
        push_str32(&mut out, self.candidate_source);
        push_str32(&mut out, self.replay_source);
        out.push(u8::from(self.sleep_subsystem_applied));
        out.push(u8::from(self.sleep_completed));
        out.push(u8::from(self.coordinator_runtime_triggered));
        out.push(u8::from(self.geist_ingested));
        out.push(u8::from(self.ism_written));
        out.push(u8::from(self.identity_anchor));
        out.push(u8::from(self.memory_stabilized));
        out.push(u8::from(self.evidence_archive_appended));
        out.push(u8::from(self.gateway_visible));
        out
    }

    pub fn digest(&self) -> Digest32 {
        digest_blake3_domain(
            b"ucf.sleep.minimal_spine.applied_local_boundary.v1",
            &self.deterministic_bytes(),
        )
    }
}

/// Version for the explicit Minimal Spine Sleep append/readback contract payload.
pub const MINIMAL_SPINE_SLEEP_APPEND_PAYLOAD_VERSION: u32 = 1;

/// Explicit contract marker carried in the Sleep Evidence/Archive append payload.
pub const MINIMAL_SPINE_SLEEP_APPEND_CONTRACT: &str = "minimal_spine_sleep_append_v1";

/// Extension kind used for bounded Sleep append proof records in archive-store.
///
/// Prompt 66 allocates `RecordKind::Other(66)` for one bounded Sleep audit/provenance payload.
/// This avoids colliding with Replay's `Other(65)`, avoids reusing broad runtime-facing archive
/// kinds, and does not change archive-store schema or Minimal Spine v1.x authority.
pub const MINIMAL_SPINE_SLEEP_APPEND_ARCHIVE_KIND: RecordKind = RecordKind::Other(66);

/// Meaning marker for the explicit Sleep append contract.
pub const MINIMAL_SPINE_SLEEP_APPEND_MEANING: &str =
    "audit_provenance_persistence_only_no_sleep_runtime";

/// Deterministic bounded Sleep append payload persisted through Evidence/Archive.
///
/// This payload is provenance only. It preserves digest links for bounded Sleep artifacts plus
/// Replay provenance and carries explicit false boundary flags for runtime Sleep execution,
/// coordinator trigger/report/WAL/journal behavior, SleepCompleted, Geist/ISM, identity, memory
/// stabilization, and Gateway visibility.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MinimalSpineSleepAppendPayload {
    pub version: u32,
    pub append_contract: String,
    pub sleep_plan_candidate_digest: Digest32,
    pub sleep_plan_audit_digest: Digest32,
    pub sleep_applied_boundary_digest: Digest32,
    pub replay_audit_digest: Digest32,
    pub replay_schedule_digest: Digest32,
    pub replay_applied_boundary_digest: Option<Digest32>,
    pub token_count: u32,
    pub candidate_source: &'static str,
    pub audit_source: &'static str,
    pub applied_boundary_source: &'static str,
    pub replay_source: &'static str,
    pub source: &'static str,
    pub runtime_executed: bool,
    pub coordinator_triggered: bool,
    pub coordinator_report_written: bool,
    pub coordinator_wal_written: bool,
    pub coordinator_journal_written: bool,
    pub sleep_completed: bool,
    pub geist_ingested: bool,
    pub ism_written: bool,
    pub identity_anchor: bool,
    pub memory_stabilized: bool,
    pub gateway_visible: bool,
    pub evidence_archive_appended_meaning: &'static str,
}

impl MinimalSpineSleepAppendPayload {
    pub fn from_artifacts(
        candidate: &MinimalSpineSleepPlanCandidate,
        audit: &MinimalSpineSleepPlanAudit,
        boundary: &MinimalSpineSleepAppliedBoundary,
    ) -> Result<Self, SleepAppendError> {
        validate_minimal_spine_sleep_append_inputs(candidate, audit, boundary)?;
        let payload = Self {
            version: MINIMAL_SPINE_SLEEP_APPEND_PAYLOAD_VERSION,
            append_contract: MINIMAL_SPINE_SLEEP_APPEND_CONTRACT.to_string(),
            sleep_plan_candidate_digest: candidate.sleep_plan_digest,
            sleep_plan_audit_digest: audit.audit_digest,
            sleep_applied_boundary_digest: boundary.applied_boundary_digest,
            replay_audit_digest: candidate.replay_audit_digest,
            replay_schedule_digest: candidate.replay_schedule_digest,
            replay_applied_boundary_digest: candidate.replay_applied_boundary_digest,
            token_count: candidate.token_count,
            candidate_source: candidate.source,
            audit_source: audit.source,
            applied_boundary_source: boundary.source,
            replay_source: candidate.replay_source,
            source: MINIMAL_SPINE_SLEEP_APPEND_CONTRACT,
            runtime_executed: false,
            coordinator_triggered: false,
            coordinator_report_written: false,
            coordinator_wal_written: false,
            coordinator_journal_written: false,
            sleep_completed: false,
            geist_ingested: false,
            ism_written: false,
            identity_anchor: false,
            memory_stabilized: false,
            gateway_visible: false,
            evidence_archive_appended_meaning: MINIMAL_SPINE_SLEEP_APPEND_MEANING,
        };
        if !payload.validate_links_nonzero() {
            return Err(SleepAppendError::InvalidInput(
                "append payload links must be non-empty/non-zero",
            ));
        }
        Ok(payload)
    }

    pub fn deterministic_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_u32_be(&mut out, self.version);
        push_str32(&mut out, &self.append_contract);
        push_digest32(&mut out, self.sleep_plan_candidate_digest);
        push_digest32(&mut out, self.sleep_plan_audit_digest);
        push_digest32(&mut out, self.sleep_applied_boundary_digest);
        push_digest32(&mut out, self.replay_audit_digest);
        push_digest32(&mut out, self.replay_schedule_digest);
        match self.replay_applied_boundary_digest {
            Some(digest) => {
                out.push(1);
                push_digest32(&mut out, digest);
            }
            None => out.push(0),
        }
        push_u32_be(&mut out, self.token_count);
        push_str32(&mut out, self.candidate_source);
        push_str32(&mut out, self.audit_source);
        push_str32(&mut out, self.applied_boundary_source);
        push_str32(&mut out, self.replay_source);
        push_str32(&mut out, self.source);
        out.push(u8::from(self.runtime_executed));
        out.push(u8::from(self.coordinator_triggered));
        out.push(u8::from(self.coordinator_report_written));
        out.push(u8::from(self.coordinator_wal_written));
        out.push(u8::from(self.coordinator_journal_written));
        out.push(u8::from(self.sleep_completed));
        out.push(u8::from(self.geist_ingested));
        out.push(u8::from(self.ism_written));
        out.push(u8::from(self.identity_anchor));
        out.push(u8::from(self.memory_stabilized));
        out.push(u8::from(self.gateway_visible));
        push_str32(&mut out, self.evidence_archive_appended_meaning);
        out
    }

    pub fn digest(&self) -> Digest32 {
        digest_blake3_domain(
            b"ucf.sleep.minimal_spine.append_payload.v1",
            &self.deterministic_bytes(),
        )
    }

    pub fn validate_links_nonzero(&self) -> bool {
        self.version == MINIMAL_SPINE_SLEEP_APPEND_PAYLOAD_VERSION
            && self.append_contract == MINIMAL_SPINE_SLEEP_APPEND_CONTRACT
            && !is_zero_digest(self.sleep_plan_candidate_digest)
            && !is_zero_digest(self.sleep_plan_audit_digest)
            && !is_zero_digest(self.sleep_applied_boundary_digest)
            && !is_zero_digest(self.replay_audit_digest)
            && !is_zero_digest(self.replay_schedule_digest)
            && !self
                .replay_applied_boundary_digest
                .is_some_and(is_zero_digest)
            && self.token_count > 0
            && !self.candidate_source.is_empty()
            && !self.audit_source.is_empty()
            && !self.applied_boundary_source.is_empty()
            && !self.replay_source.is_empty()
            && !self.source.is_empty()
            && !self.runtime_executed
            && !self.coordinator_triggered
            && !self.coordinator_report_written
            && !self.coordinator_wal_written
            && !self.coordinator_journal_written
            && !self.sleep_completed
            && !self.geist_ingested
            && !self.ism_written
            && !self.identity_anchor
            && !self.memory_stabilized
            && !self.gateway_visible
            && self.evidence_archive_appended_meaning == MINIMAL_SPINE_SLEEP_APPEND_MEANING
    }
}

/// Result of the explicit Minimal Spine Sleep append/readback contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MinimalSpineSleepAppendResult {
    pub payload_digest: Digest32,
    pub sleep_plan_candidate_digest: Digest32,
    pub sleep_plan_audit_digest: Digest32,
    pub sleep_applied_boundary_digest: Digest32,
    pub replay_audit_digest: Digest32,
    pub replay_schedule_digest: Digest32,
    pub replay_applied_boundary_digest: Option<Digest32>,
    pub token_count: u32,
    pub appended_evidence_id: EvidenceId,
    pub archive_key: Digest32,
    pub archive_record_digest: Digest32,
    pub readback_digest: Digest32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SleepAppendError {
    InvalidInput(&'static str),
    ReadbackMissing,
    ReadbackMismatch,
}

impl std::fmt::Display for SleepAppendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(message) => {
                write!(f, "invalid minimal spine sleep append input: {message}")
            }
            Self::ReadbackMissing => f.write_str("minimal spine sleep append readback missing"),
            Self::ReadbackMismatch => f.write_str("minimal spine sleep append readback mismatch"),
        }
    }
}

impl std::error::Error for SleepAppendError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SleepAppliedBoundaryError {
    AuditStatusNotPass,
    AuditDigestMismatch,
    AuditHasFailureReasons,
    AuditCandidateDigestMismatch,
    AuditHasForbiddenSideEffectFlag,
    ZeroSleepPlanAuditDigest,
    ZeroSleepPlanCandidateDigest,
    ZeroReplayAuditDigest,
    ZeroReplayScheduleDigest,
    ZeroReplayAppliedBoundaryDigest,
    ZeroTokenCount,
    EmptySource,
    EmptyCandidateSource,
    EmptyReplaySource,
}

impl std::fmt::Display for SleepAppliedBoundaryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::AuditStatusNotPass => "sleep plan audit status must be pass",
            Self::AuditDigestMismatch => "sleep plan audit digest mismatch",
            Self::AuditHasFailureReasons => "pass sleep plan audit must not have failure reasons",
            Self::AuditCandidateDigestMismatch => "sleep plan audit candidate digest mismatch",
            Self::AuditHasForbiddenSideEffectFlag => {
                "sleep plan audit has a forbidden side-effect flag set"
            }
            Self::ZeroSleepPlanAuditDigest => "sleep plan audit digest must be non-zero",
            Self::ZeroSleepPlanCandidateDigest => "sleep plan candidate digest must be non-zero",
            Self::ZeroReplayAuditDigest => "replay audit digest must be non-zero",
            Self::ZeroReplayScheduleDigest => "replay schedule digest must be non-zero",
            Self::ZeroReplayAppliedBoundaryDigest => {
                "replay applied boundary digest must be non-zero"
            }
            Self::ZeroTokenCount => "token count must be non-zero",
            Self::EmptySource => "sleep plan audit source must be non-empty",
            Self::EmptyCandidateSource => "sleep plan candidate source must be non-empty",
            Self::EmptyReplaySource => "replay source must be non-empty",
        };
        f.write_str(message)
    }
}

impl std::error::Error for SleepAppliedBoundaryError {}

impl MinimalSpineSleepPlanAudit {
    /// Deterministic bytes used for the audit digest.
    pub fn deterministic_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_u32_be(&mut out, self.version);
        out.push(self.status.code());
        push_u32_be(
            &mut out,
            u32::try_from(self.failure_reasons.len())
                .expect("minimal spine sleep plan audit failure reason count fits u32"),
        );
        for reason in &self.failure_reasons {
            out.push(reason.code());
        }
        push_digest32(&mut out, self.sleep_plan_candidate_digest);
        push_digest32(&mut out, self.recomputed_sleep_plan_candidate_digest);
        push_digest32(&mut out, self.replay_audit_digest);
        push_digest32(&mut out, self.replay_schedule_digest);
        match self.replay_applied_boundary_digest {
            Some(digest) => {
                out.push(1);
                push_digest32(&mut out, digest);
            }
            None => out.push(0),
        }
        push_u32_be(&mut out, self.token_count);
        push_str32(&mut out, self.source);
        push_str32(&mut out, self.candidate_source);
        push_str32(&mut out, self.replay_source);
        out.push(u8::from(self.candidate_only));
        out.push(u8::from(self.sleep_applied));
        out.push(u8::from(self.sleep_completed));
        out.push(u8::from(self.geist_ingested));
        out.push(u8::from(self.ism_written));
        out.push(u8::from(self.identity_anchor));
        out.push(u8::from(self.evidence_archive_appended));
        out.push(u8::from(self.gateway_visible));
        out
    }

    pub fn digest(&self) -> Digest32 {
        digest_blake3_domain(
            b"ucf.sleep.minimal_spine.plan_verify_only_audit.v1",
            &self.deterministic_bytes(),
        )
    }
}

/// Explicitly append bounded Sleep audit/provenance evidence and read it back.
///
/// This helper is the only Sleep append surface. It persists a deterministic provenance payload
/// through the existing EvidenceStore and ArchiveStore APIs, then immediately verifies readback.
/// It does not execute Sleep runtime behavior, trigger/report/write coordinator WAL or journal
/// records, mark SleepCompleted, ingest into Geist/ISM, write identity anchors, expose Gateway
/// semantics, create a second event log, or change Minimal Spine v1.x.
pub fn append_minimal_spine_sleep_record<E, S>(
    candidate: &MinimalSpineSleepPlanCandidate,
    audit: &MinimalSpineSleepPlanAudit,
    boundary: &MinimalSpineSleepAppliedBoundary,
    evidence_store: &E,
    archive_store: &S,
    archive_appender: &mut ArchiveAppender,
) -> Result<MinimalSpineSleepAppendResult, SleepAppendError>
where
    E: EvidenceStore,
    S: ArchiveStore,
{
    let payload = MinimalSpineSleepAppendPayload::from_artifacts(candidate, audit, boundary)?;
    let payload_bytes = payload.deterministic_bytes();
    let payload_digest = payload.digest();
    let appended_evidence_id = EvidenceId::new(format!(
        "minimal-spine-sleep-append-{}",
        hex::encode(payload_digest.as_bytes())
    ));
    let proof = ProofEnvelope {
        envelope_id: format!(
            "{}:{}",
            MINIMAL_SPINE_SLEEP_APPEND_CONTRACT,
            hex::encode(payload_digest.as_bytes())
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
        MINIMAL_SPINE_SLEEP_APPEND_ARCHIVE_KIND,
        payload_digest,
        RecordMeta {
            cycle_id: 0,
            tier: 3,
            flags: 0,
            boundary_commit: boundary.applied_boundary_digest,
        },
    );
    let archive_record_digest = archive_store.append(archive_record);
    let readback = evidence_store
        .get(appended_evidence_id.clone())
        .ok_or(SleepAppendError::ReadbackMissing)?;
    let readback_digest = digest_sleep_evidence_envelope(&readback);
    let archive_readback = archive_store
        .get(archive_record.key)
        .ok_or(SleepAppendError::ReadbackMissing)?;
    if archive_readback != archive_record {
        return Err(SleepAppendError::ReadbackMismatch);
    }

    Ok(MinimalSpineSleepAppendResult {
        payload_digest,
        sleep_plan_candidate_digest: payload.sleep_plan_candidate_digest,
        sleep_plan_audit_digest: payload.sleep_plan_audit_digest,
        sleep_applied_boundary_digest: payload.sleep_applied_boundary_digest,
        replay_audit_digest: payload.replay_audit_digest,
        replay_schedule_digest: payload.replay_schedule_digest,
        replay_applied_boundary_digest: payload.replay_applied_boundary_digest,
        token_count: payload.token_count,
        appended_evidence_id,
        archive_key: archive_record.key,
        archive_record_digest,
        readback_digest,
    })
}

fn validate_minimal_spine_sleep_append_inputs(
    candidate: &MinimalSpineSleepPlanCandidate,
    audit: &MinimalSpineSleepPlanAudit,
    boundary: &MinimalSpineSleepAppliedBoundary,
) -> Result<(), SleepAppendError> {
    if candidate.sleep_plan_digest != candidate.digest() {
        return Err(SleepAppendError::InvalidInput("candidate digest mismatch"));
    }
    if audit.status != MinimalSpineSleepPlanAuditStatus::Pass {
        return Err(SleepAppendError::InvalidInput("audit status must be pass"));
    }
    if audit.audit_digest != audit.digest() {
        return Err(SleepAppendError::InvalidInput("audit digest mismatch"));
    }
    if !audit.failure_reasons.is_empty() {
        return Err(SleepAppendError::InvalidInput(
            "pass audit must not have failure reasons",
        ));
    }
    if boundary.applied_boundary_digest != boundary.digest() {
        return Err(SleepAppendError::InvalidInput(
            "applied boundary digest mismatch",
        ));
    }
    if audit.sleep_plan_candidate_digest != candidate.sleep_plan_digest
        || audit.recomputed_sleep_plan_candidate_digest != candidate.sleep_plan_digest
    {
        return Err(SleepAppendError::InvalidInput(
            "audit must match candidate digest",
        ));
    }
    if boundary.sleep_plan_audit_digest != audit.audit_digest
        || boundary.sleep_plan_candidate_digest != candidate.sleep_plan_digest
    {
        return Err(SleepAppendError::InvalidInput(
            "boundary must match audit and candidate digests",
        ));
    }
    if candidate.replay_audit_digest != audit.replay_audit_digest
        || candidate.replay_audit_digest != boundary.replay_audit_digest
        || candidate.replay_schedule_digest != audit.replay_schedule_digest
        || candidate.replay_schedule_digest != boundary.replay_schedule_digest
        || candidate.replay_applied_boundary_digest != audit.replay_applied_boundary_digest
        || candidate.replay_applied_boundary_digest != boundary.replay_applied_boundary_digest
    {
        return Err(SleepAppendError::InvalidInput(
            "sleep artifacts must preserve replay provenance",
        ));
    }
    if candidate.token_count != audit.token_count || candidate.token_count != boundary.token_count {
        return Err(SleepAppendError::InvalidInput(
            "token count must match candidate, audit, and boundary",
        ));
    }
    if is_zero_digest(candidate.sleep_plan_digest)
        || is_zero_digest(audit.audit_digest)
        || is_zero_digest(boundary.applied_boundary_digest)
        || is_zero_digest(candidate.replay_audit_digest)
        || is_zero_digest(candidate.replay_schedule_digest)
        || candidate
            .replay_applied_boundary_digest
            .is_some_and(is_zero_digest)
        || candidate.token_count == 0
    {
        return Err(SleepAppendError::InvalidInput(
            "append digests and token count must be non-zero",
        ));
    }
    if candidate.source.is_empty()
        || candidate.replay_source.is_empty()
        || audit.source.is_empty()
        || audit.candidate_source.is_empty()
        || audit.replay_source.is_empty()
        || boundary.source.is_empty()
        || boundary.sleep_plan_audit_source.is_empty()
        || boundary.candidate_source.is_empty()
        || boundary.replay_source.is_empty()
    {
        return Err(SleepAppendError::InvalidInput(
            "append source markers must be non-empty",
        ));
    }
    if !candidate.candidate_only
        || candidate.sleep_applied
        || candidate.sleep_completed
        || candidate.geist_ingested
        || candidate.ism_written
        || candidate.identity_anchor
        || candidate.evidence_archive_appended
        || candidate.gateway_visible
        || !audit.candidate_only
        || audit.sleep_applied
        || audit.sleep_completed
        || audit.geist_ingested
        || audit.ism_written
        || audit.identity_anchor
        || audit.evidence_archive_appended
        || audit.gateway_visible
        || !boundary.sleep_subsystem_applied
        || boundary.sleep_completed
        || boundary.coordinator_runtime_triggered
        || boundary.geist_ingested
        || boundary.ism_written
        || boundary.identity_anchor
        || boundary.memory_stabilized
        || boundary.evidence_archive_appended
        || boundary.gateway_visible
    {
        return Err(SleepAppendError::InvalidInput(
            "sleep append inputs must not carry forbidden side-effect flags",
        ));
    }
    Ok(())
}

fn digest_sleep_evidence_envelope(envelope: &EvidenceEnvelope) -> Digest32 {
    let mut out = Vec::new();
    push_str32(&mut out, envelope.evidence_id.as_str());
    match &envelope.proof {
        Some(proof) => {
            out.push(1);
            push_str32(&mut out, &proof.envelope_id);
            push_u32_be(
                &mut out,
                u32::try_from(proof.payload.len())
                    .expect("minimal spine sleep append proof payload length fits u32"),
            );
            out.extend_from_slice(&proof.payload);
            match &proof.payload_digest {
                Some(digest) => {
                    out.push(1);
                    push_str32(&mut out, &digest.algorithm);
                    push_u32_be(
                        &mut out,
                        u32::try_from(digest.value.len())
                            .expect("minimal spine sleep append digest length fits u32"),
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
                    .expect("minimal spine sleep append vrf tag count fits u32"),
            );
            push_u32_be(
                &mut out,
                u32::try_from(proof.signature_ids.len())
                    .expect("minimal spine sleep append signature id count fits u32"),
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
    digest_blake3_domain(b"ucf.sleep.minimal_spine.append_readback.v1", &out)
}

/// Build a local SleepApplied boundary marker from a PASS verify-only SleepPlan audit.
///
/// The output is deterministic Sleep bookkeeping only. It preserves SleepPlan and Replay
/// provenance, rejects FAIL audits and malformed PASS audits, takes no coordinator runtime, WAL,
/// journal, store, scheduler, Gateway, Geist, ISM, Evidence, or Archive handles, and does not
/// mutate the audit or candidate that produced it.
pub fn build_sleep_applied_boundary_from_audit(
    audit: &MinimalSpineSleepPlanAudit,
) -> Result<MinimalSpineSleepAppliedBoundary, SleepAppliedBoundaryError> {
    validate_sleep_plan_audit_for_applied_boundary(audit)?;

    let mut boundary = MinimalSpineSleepAppliedBoundary {
        version: MINIMAL_SPINE_SLEEP_APPLIED_BOUNDARY_VERSION,
        sleep_plan_audit_digest: audit.audit_digest,
        sleep_plan_candidate_digest: audit.sleep_plan_candidate_digest,
        replay_audit_digest: audit.replay_audit_digest,
        replay_schedule_digest: audit.replay_schedule_digest,
        replay_applied_boundary_digest: audit.replay_applied_boundary_digest,
        token_count: audit.token_count,
        applied_boundary_digest: Digest32::new([0u8; Digest32::LEN]),
        source: MINIMAL_SPINE_SLEEP_APPLIED_BOUNDARY_SOURCE,
        sleep_plan_audit_source: audit.source,
        candidate_source: audit.candidate_source,
        replay_source: audit.replay_source,
        sleep_subsystem_applied: true,
        sleep_completed: false,
        coordinator_runtime_triggered: false,
        geist_ingested: false,
        ism_written: false,
        identity_anchor: false,
        memory_stabilized: false,
        evidence_archive_appended: false,
        gateway_visible: false,
    };
    boundary.applied_boundary_digest = boundary.digest();
    Ok(boundary)
}

fn validate_sleep_plan_audit_for_applied_boundary(
    audit: &MinimalSpineSleepPlanAudit,
) -> Result<(), SleepAppliedBoundaryError> {
    if audit.status != MinimalSpineSleepPlanAuditStatus::Pass {
        return Err(SleepAppliedBoundaryError::AuditStatusNotPass);
    }
    if audit.audit_digest != audit.digest() {
        return Err(SleepAppliedBoundaryError::AuditDigestMismatch);
    }
    if !audit.failure_reasons.is_empty() {
        return Err(SleepAppliedBoundaryError::AuditHasFailureReasons);
    }
    if audit.sleep_plan_candidate_digest != audit.recomputed_sleep_plan_candidate_digest {
        return Err(SleepAppliedBoundaryError::AuditCandidateDigestMismatch);
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
        return Err(SleepAppliedBoundaryError::AuditHasForbiddenSideEffectFlag);
    }
    if is_zero_digest(audit.audit_digest) {
        return Err(SleepAppliedBoundaryError::ZeroSleepPlanAuditDigest);
    }
    if is_zero_digest(audit.sleep_plan_candidate_digest) {
        return Err(SleepAppliedBoundaryError::ZeroSleepPlanCandidateDigest);
    }
    if is_zero_digest(audit.replay_audit_digest) {
        return Err(SleepAppliedBoundaryError::ZeroReplayAuditDigest);
    }
    if is_zero_digest(audit.replay_schedule_digest) {
        return Err(SleepAppliedBoundaryError::ZeroReplayScheduleDigest);
    }
    if audit
        .replay_applied_boundary_digest
        .is_some_and(is_zero_digest)
    {
        return Err(SleepAppliedBoundaryError::ZeroReplayAppliedBoundaryDigest);
    }
    if audit.token_count == 0 {
        return Err(SleepAppliedBoundaryError::ZeroTokenCount);
    }
    if audit.source.is_empty() {
        return Err(SleepAppliedBoundaryError::EmptySource);
    }
    if audit.candidate_source.is_empty() {
        return Err(SleepAppliedBoundaryError::EmptyCandidateSource);
    }
    if audit.replay_source.is_empty() {
        return Err(SleepAppliedBoundaryError::EmptyReplaySource);
    }
    Ok(())
}

/// Verify a Minimal Spine SleepPlan candidate without applying Sleep or triggering a coordinator.
///
/// The returned audit can PASS or FAIL. PASS means only that the candidate metadata is internally
/// consistent and still candidate-only. It is not SleepApplied, not SleepCompleted, not
/// Geist/ISM/identity state, and not Evidence/Archive/Gateway publication.
pub fn verify_minimal_spine_sleep_plan_candidate(
    candidate: &MinimalSpineSleepPlanCandidate,
) -> MinimalSpineSleepPlanAudit {
    let recomputed_sleep_plan_candidate_digest = candidate.digest();
    let mut reasons = Vec::new();

    if candidate.version != MINIMAL_SPINE_SLEEP_PLAN_CANDIDATE_VERSION {
        reasons.push(MinimalSpineSleepPlanAuditFailure::VersionMismatch);
    }
    if candidate.sleep_plan_digest != recomputed_sleep_plan_candidate_digest {
        reasons.push(MinimalSpineSleepPlanAuditFailure::CandidateDigestMismatch);
    }
    if is_zero_digest(candidate.replay_audit_digest) {
        reasons.push(MinimalSpineSleepPlanAuditFailure::ZeroReplayAuditDigest);
    }
    if is_zero_digest(candidate.replay_schedule_digest) {
        reasons.push(MinimalSpineSleepPlanAuditFailure::ZeroReplayScheduleDigest);
    }
    if candidate
        .replay_applied_boundary_digest
        .is_some_and(is_zero_digest)
    {
        reasons.push(MinimalSpineSleepPlanAuditFailure::ZeroReplayAppliedBoundaryDigest);
    }
    if candidate.token_count == 0 {
        reasons.push(MinimalSpineSleepPlanAuditFailure::InvalidTokenCount);
    }
    if candidate.source.is_empty() {
        reasons.push(MinimalSpineSleepPlanAuditFailure::EmptySource);
    }
    if candidate.replay_source.is_empty() {
        reasons.push(MinimalSpineSleepPlanAuditFailure::EmptyReplaySource);
    }
    if !candidate.candidate_only {
        reasons.push(MinimalSpineSleepPlanAuditFailure::NotCandidateOnly);
    }
    if candidate.sleep_applied {
        reasons.push(MinimalSpineSleepPlanAuditFailure::SleepAppliedFlagSet);
    }
    if candidate.sleep_completed {
        reasons.push(MinimalSpineSleepPlanAuditFailure::SleepCompletedFlagSet);
    }
    if candidate.geist_ingested {
        reasons.push(MinimalSpineSleepPlanAuditFailure::GeistIngestedFlagSet);
    }
    if candidate.ism_written {
        reasons.push(MinimalSpineSleepPlanAuditFailure::IsmWrittenFlagSet);
    }
    if candidate.identity_anchor {
        reasons.push(MinimalSpineSleepPlanAuditFailure::IdentityAnchorFlagSet);
    }
    if candidate.evidence_archive_appended {
        reasons.push(MinimalSpineSleepPlanAuditFailure::EvidenceArchiveAppendedFlagSet);
    }
    if candidate.gateway_visible {
        reasons.push(MinimalSpineSleepPlanAuditFailure::GatewayVisibleFlagSet);
    }

    reasons.sort_unstable();
    reasons.dedup();
    let status = if reasons.is_empty() {
        MinimalSpineSleepPlanAuditStatus::Pass
    } else {
        MinimalSpineSleepPlanAuditStatus::Fail
    };

    let mut audit = MinimalSpineSleepPlanAudit {
        version: MINIMAL_SPINE_SLEEP_PLAN_AUDIT_VERSION,
        status,
        failure_reasons: reasons,
        sleep_plan_candidate_digest: candidate.sleep_plan_digest,
        recomputed_sleep_plan_candidate_digest,
        replay_audit_digest: candidate.replay_audit_digest,
        replay_schedule_digest: candidate.replay_schedule_digest,
        replay_applied_boundary_digest: candidate.replay_applied_boundary_digest,
        token_count: candidate.token_count,
        audit_digest: Digest32::new([0u8; Digest32::LEN]),
        source: MINIMAL_SPINE_SLEEP_PLAN_AUDIT_SOURCE,
        candidate_source: candidate.source,
        replay_source: candidate.replay_source,
        candidate_only: candidate.candidate_only,
        sleep_applied: false,
        sleep_completed: false,
        geist_ingested: false,
        ism_written: false,
        identity_anchor: false,
        evidence_archive_appended: false,
        gateway_visible: false,
    };
    audit.audit_digest = audit.digest();
    audit
}

/// Build a candidate-only SleepPlan from validated replay-boundary digest input.
///
/// The function is pure and deterministic. It takes no store/appender/Gateway/Geist/ISM/scheduler
/// handles, does not trigger the existing coordinator runtime, does not mutate replay metadata, and
/// does not append or expose anything.
pub fn build_sleep_plan_candidate_from_replay_boundary(
    input: &MinimalSpineSleepPlanInput,
) -> Result<MinimalSpineSleepPlanCandidate, SleepPlanCandidateError> {
    validate_sleep_plan_input(input)?;

    let mut candidate = MinimalSpineSleepPlanCandidate {
        version: MINIMAL_SPINE_SLEEP_PLAN_CANDIDATE_VERSION,
        replay_audit_digest: input.replay_audit_digest,
        replay_schedule_digest: input.replay_schedule_digest,
        replay_applied_boundary_digest: input.replay_applied_boundary_digest,
        token_count: input.token_count,
        source: MINIMAL_SPINE_SLEEP_PLAN_CANDIDATE_SOURCE,
        replay_source: input.source,
        candidate_only: true,
        sleep_applied: false,
        sleep_completed: false,
        geist_ingested: false,
        ism_written: false,
        identity_anchor: false,
        evidence_archive_appended: false,
        gateway_visible: false,
        sleep_plan_digest: Digest32::new([0u8; Digest32::LEN]),
    };
    candidate.sleep_plan_digest = candidate.digest();
    Ok(candidate)
}

/// Build a candidate-only SleepPlan directly from a PASS replay schedule audit and optional local
/// replay applied boundary marker.
///
/// The optional boundary is provenance only. It must match the audit digest, schedule digest, and
/// token count. This function does not execute Replay or Sleep and does not promote the existing
/// coordinator prototype to runtime authority.
pub fn build_sleep_plan_candidate_from_replay_audit(
    audit: &MinimalSpineReplayScheduleAudit,
    applied_boundary: Option<&MinimalSpineReplayAppliedBoundary>,
) -> Result<MinimalSpineSleepPlanCandidate, SleepPlanCandidateError> {
    validate_replay_audit_for_sleep_plan(audit)?;
    if let Some(boundary) = applied_boundary {
        validate_replay_applied_boundary_for_sleep_plan(audit, boundary)?;
    }

    let input = MinimalSpineSleepPlanInput {
        replay_audit_digest: audit.audit_digest,
        replay_schedule_digest: audit.schedule_digest,
        replay_applied_boundary_digest: applied_boundary
            .map(|boundary| boundary.applied_boundary_digest),
        token_count: audit.token_count,
        source: audit.source,
    };
    build_sleep_plan_candidate_from_replay_boundary(&input)
}

fn validate_replay_audit_for_sleep_plan(
    audit: &MinimalSpineReplayScheduleAudit,
) -> Result<(), SleepPlanCandidateError> {
    if audit.status != MinimalSpineReplayAuditStatus::Pass {
        return Err(SleepPlanCandidateError::AuditStatusNotPass);
    }
    if audit.audit_digest != audit.digest() {
        return Err(SleepPlanCandidateError::AuditDigestMismatch);
    }
    if !audit.failure_reasons.is_empty() {
        return Err(SleepPlanCandidateError::AuditHasFailureReasons);
    }
    if audit.schedule_digest != audit.recomputed_schedule_digest {
        return Err(SleepPlanCandidateError::AuditScheduleDigestMismatch);
    }
    if audit.applied
        || audit.replay_completed
        || audit.sleep_cycle
        || audit.geist_ingested
        || audit.identity_anchor
        || audit.evidence_archive_appended
    {
        return Err(SleepPlanCandidateError::AuditHasForbiddenBoundaryFlag);
    }
    let input = MinimalSpineSleepPlanInput {
        replay_audit_digest: audit.audit_digest,
        replay_schedule_digest: audit.schedule_digest,
        replay_applied_boundary_digest: None,
        token_count: audit.token_count,
        source: audit.source,
    };
    validate_sleep_plan_input(&input)
}

fn validate_replay_applied_boundary_for_sleep_plan(
    audit: &MinimalSpineReplayScheduleAudit,
    boundary: &MinimalSpineReplayAppliedBoundary,
) -> Result<(), SleepPlanCandidateError> {
    if boundary.applied_boundary_digest != boundary.digest() {
        return Err(SleepPlanCandidateError::BoundaryDigestMismatch);
    }
    if boundary.audit_digest != audit.audit_digest {
        return Err(SleepPlanCandidateError::BoundaryAuditDigestMismatch);
    }
    if boundary.schedule_digest != audit.schedule_digest {
        return Err(SleepPlanCandidateError::BoundaryScheduleDigestMismatch);
    }
    if boundary.token_count != audit.token_count {
        return Err(SleepPlanCandidateError::BoundaryTokenCountMismatch);
    }
    if boundary.geist_ingested
        || boundary.ism_written
        || boundary.identity_anchor
        || boundary.sleep_completed
        || boundary.evidence_archive_appended
        || boundary.gateway_visible
    {
        return Err(SleepPlanCandidateError::BoundaryHasForbiddenSideEffectFlag);
    }
    if is_zero_digest(boundary.applied_boundary_digest) {
        return Err(SleepPlanCandidateError::ZeroReplayAppliedBoundaryDigest);
    }
    Ok(())
}

fn validate_sleep_plan_input(
    input: &MinimalSpineSleepPlanInput,
) -> Result<(), SleepPlanCandidateError> {
    if is_zero_digest(input.replay_audit_digest) {
        return Err(SleepPlanCandidateError::ZeroReplayAuditDigest);
    }
    if is_zero_digest(input.replay_schedule_digest) {
        return Err(SleepPlanCandidateError::ZeroReplayScheduleDigest);
    }
    if let Some(digest) = input.replay_applied_boundary_digest {
        if is_zero_digest(digest) {
            return Err(SleepPlanCandidateError::ZeroReplayAppliedBoundaryDigest);
        }
    }
    if input.token_count == 0 {
        return Err(SleepPlanCandidateError::ZeroTokenCount);
    }
    if input.source.is_empty() {
        return Err(SleepPlanCandidateError::EmptySource);
    }
    Ok(())
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
    let len = u32::try_from(value.len()).expect("minimal spine sleep plan source length fits u32");
    push_u32_be(out, len);
    out.extend_from_slice(value.as_bytes());
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
                    .expect("minimal spine sleep append bytes length fits u32"),
            );
            out.extend_from_slice(value);
        }
        None => out.push(0),
    }
}

fn digest_blake3_domain(domain: &[u8], bytes: &[u8]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(domain);
    hasher.update(&[0]);
    hasher.update(bytes);
    Digest32::new(*hasher.finalize().as_bytes())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SleepHeuristics {
    pub min_records_since_last: u32,
    pub max_instability: u16,
    pub min_integration: u16,
    pub critical_surprise_threshold: u16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SleepState {
    pub last_cycle_id: u64,
    pub last_evidence: Option<EvidenceId>,
    pub records_since_last: u32,
    pub critical_surprise_count: u16,
    pub last_replay: Option<SleepReplaySummary>,
    pub structural_stats: Option<StructuralCycleStats>,
    pub structural_proposal: Option<StructuralDeltaProposal>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SleepTrigger {
    None,
    Instability,
    Density,
    LowIntegration,
    Manual,
    SurpriseCritical,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecentMetrics {
    pub consistency_verdicts: Vec<ConsistencyVerdict>,
    pub integration_scores: Vec<u16>,
    pub records_since_last: u32,
}

impl RecentMetrics {
    pub fn instability_score(&self) -> u16 {
        if self.consistency_verdicts.is_empty() {
            return 0;
        }
        let unstable = self
            .consistency_verdicts
            .iter()
            .filter(|verdict| **verdict != ConsistencyVerdict::Accept)
            .count();
        let total = self.consistency_verdicts.len();
        let score = (unstable as u32 * 10_000) / total.max(1) as u32;
        u16::try_from(score.min(u32::from(u16::MAX))).unwrap_or(u16::MAX)
    }

    pub fn average_integration_score(&self) -> u16 {
        if self.integration_scores.is_empty() {
            return 0;
        }
        let sum: u32 = self
            .integration_scores
            .iter()
            .map(|score| u32::from(*score))
            .sum();
        let avg = sum / self.integration_scores.len().max(1) as u32;
        u16::try_from(avg.min(u32::from(u16::MAX))).unwrap_or(u16::MAX)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SleepReplaySummary {
    pub micro: u16,
    pub meso: u16,
    pub macro_: u16,
}

pub trait SleepStateUpdater {
    fn record_derived_record(&mut self, evidence_id: EvidenceId);
    fn record_consistency_verdict(&mut self, verdict: ConsistencyVerdict);
    fn record_integration_score(&mut self, score: u16);
    fn record_surprise_band(&mut self, _band: SurpriseBand) {}
    fn record_replay_summary(&mut self, _summary: SleepReplaySummary) {}
    fn record_structural_stats(&mut self, _stats: StructuralCycleStats) {}
    fn record_structural_proposal(&mut self, _proposal: StructuralDeltaProposal) {}
}

pub type SleepStateHandle = Arc<Mutex<WalSleepCoordinator>>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SleepTriggered {
    pub cycle_id: u64,
    pub reason: SleepTrigger,
}

pub trait SleepPhaseRunner {
    fn run_sleep_phase(
        &self,
        cycle_id: u64,
        fixed_seed: [u8; 32],
        integration_score: u16,
        recent_evidence: &[EvidenceId],
        structural_stats: Option<StructuralCycleStats>,
        structural_proposal: Option<StructuralDeltaProposal>,
    ) -> Option<SleepReportReady>;
}

impl<P, R, O, B> SleepPhaseRunner for SleepCoordinator<P, R, O, B>
where
    P: SleepPhaseGate,
    R: RsaEngine,
    O: ucf_openevolve_port::OpenEvolvePort,
    B: BusPublisher<SleepReportReady>,
{
    fn run_sleep_phase(
        &self,
        cycle_id: u64,
        fixed_seed: [u8; 32],
        integration_score: u16,
        recent_evidence: &[EvidenceId],
        structural_stats: Option<StructuralCycleStats>,
        structural_proposal: Option<StructuralDeltaProposal>,
    ) -> Option<SleepReportReady> {
        SleepCoordinator::run_sleep_phase(
            self,
            cycle_id,
            fixed_seed,
            integration_score,
            recent_evidence,
            structural_stats,
            structural_proposal,
        )
    }
}

pub struct WalSleepCoordinator {
    heuristics: SleepHeuristics,
    state: SleepState,
    window: usize,
    consistency_verdicts: VecDeque<ConsistencyVerdict>,
    integration_scores: VecDeque<u16>,
    recent_evidence: VecDeque<EvidenceId>,
}

impl WalSleepCoordinator {
    pub fn new(heuristics: SleepHeuristics, window: usize) -> Self {
        Self {
            heuristics,
            state: SleepState {
                last_cycle_id: 0,
                last_evidence: None,
                records_since_last: 0,
                critical_surprise_count: 0,
                last_replay: None,
                structural_stats: None,
                structural_proposal: None,
            },
            window,
            consistency_verdicts: VecDeque::new(),
            integration_scores: VecDeque::new(),
            recent_evidence: VecDeque::new(),
        }
    }

    pub fn heuristics(&self) -> &SleepHeuristics {
        &self.heuristics
    }

    pub fn state(&self) -> &SleepState {
        &self.state
    }

    pub fn recent_metrics(&self) -> RecentMetrics {
        RecentMetrics {
            consistency_verdicts: self.consistency_verdicts.iter().copied().collect(),
            integration_scores: self.integration_scores.iter().copied().collect(),
            records_since_last: self.state.records_since_last,
        }
    }

    pub fn recent_evidence(&self) -> Vec<EvidenceId> {
        self.recent_evidence.iter().cloned().collect()
    }

    pub fn evaluate(&mut self, recent_metrics: &RecentMetrics) -> SleepTrigger {
        if self.heuristics.critical_surprise_threshold > 0
            && self.state.critical_surprise_count >= self.heuristics.critical_surprise_threshold
        {
            self.commit_trigger();
            return SleepTrigger::SurpriseCritical;
        }
        let instability = recent_metrics.instability_score();
        if instability > self.heuristics.max_instability {
            self.commit_trigger();
            return SleepTrigger::Instability;
        }

        let integration_score = recent_metrics.average_integration_score();
        if integration_score < self.heuristics.min_integration {
            self.commit_trigger();
            return SleepTrigger::LowIntegration;
        }

        if recent_metrics.records_since_last >= self.heuristics.min_records_since_last {
            self.commit_trigger();
            return SleepTrigger::Density;
        }

        SleepTrigger::None
    }

    pub fn maybe_trigger<R, B>(&mut self, runner: &R, bus: &B) -> Option<SleepReportReady>
    where
        R: SleepPhaseRunner + ?Sized,
        B: BusPublisher<SleepTriggered>,
    {
        let recent_metrics = self.recent_metrics();
        let recent_evidence = self.recent_evidence();
        let trigger = self.evaluate(&recent_metrics);
        if trigger == SleepTrigger::None {
            return None;
        }

        let cycle_id = self.state.last_cycle_id;
        let fixed_seed = derive_fixed_seed(cycle_id, self.state.last_evidence.as_ref());
        let integration_score = recent_metrics.average_integration_score();
        let structural_stats = self.state.structural_stats.clone();
        let structural_proposal = self.state.structural_proposal.clone();
        bus.publish(SleepTriggered {
            cycle_id,
            reason: trigger,
        });
        let result = runner.run_sleep_phase(
            cycle_id,
            fixed_seed,
            integration_score,
            &recent_evidence,
            structural_stats,
            structural_proposal,
        );
        self.reset_after_trigger();
        result
    }

    pub fn force_trigger(&mut self) -> SleepTrigger {
        self.commit_trigger();
        SleepTrigger::Manual
    }

    fn commit_trigger(&mut self) {
        self.state.last_cycle_id = self.state.last_cycle_id.saturating_add(1);
        self.state.records_since_last = 0;
        self.state.critical_surprise_count = 0;
    }

    fn reset_after_trigger(&mut self) {
        self.consistency_verdicts.clear();
        self.integration_scores.clear();
        self.recent_evidence.clear();
        self.state.critical_surprise_count = 0;
        self.state.structural_stats = None;
        self.state.structural_proposal = None;
    }

    fn push_bounded<T>(queue: &mut VecDeque<T>, window: usize, value: T) {
        queue.push_back(value);
        if window > 0 {
            while queue.len() > window {
                queue.pop_front();
            }
        }
    }
}

impl SleepStateUpdater for WalSleepCoordinator {
    fn record_derived_record(&mut self, evidence_id: EvidenceId) {
        self.state.records_since_last = self.state.records_since_last.saturating_add(1);
        self.state.last_evidence = Some(evidence_id.clone());
        Self::push_bounded(&mut self.recent_evidence, self.window, evidence_id);
    }

    fn record_consistency_verdict(&mut self, verdict: ConsistencyVerdict) {
        Self::push_bounded(&mut self.consistency_verdicts, self.window, verdict);
    }

    fn record_integration_score(&mut self, score: u16) {
        Self::push_bounded(&mut self.integration_scores, self.window, score);
    }

    fn record_surprise_band(&mut self, band: SurpriseBand) {
        if band == SurpriseBand::Critical {
            self.state.critical_surprise_count =
                self.state.critical_surprise_count.saturating_add(1);
        } else {
            self.state.critical_surprise_count = 0;
        }
    }

    fn record_replay_summary(&mut self, summary: SleepReplaySummary) {
        self.state.last_replay = Some(summary);
    }

    fn record_structural_stats(&mut self, stats: StructuralCycleStats) {
        self.state.structural_stats = Some(stats);
    }

    fn record_structural_proposal(&mut self, proposal: StructuralDeltaProposal) {
        self.state.structural_proposal = Some(proposal);
    }
}

fn derive_fixed_seed(cycle_id: u64, last_evidence: Option<&EvidenceId>) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.sleep.wal.v1");
    hasher.update(&cycle_id.to_le_bytes());
    if let Some(evidence) = last_evidence {
        hasher.update(evidence.as_str().as_bytes());
    }
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use ucf_archive::InMemoryArchive;
    use ucf_bus::{BusSubscriber, InMemoryBus};
    use ucf_openevolve_port::MockOpenEvolvePort;
    use ucf_policy_ecology::{DefaultPolicyEcology, PolicyEcology, PolicyRule, PolicyWeights};
    use ucf_rsa::MockRsaEngine;

    #[test]
    fn sleep_trigger_density_increments_cycle() {
        let heuristics = SleepHeuristics {
            min_records_since_last: 2,
            max_instability: 10_000,
            min_integration: 0,
            critical_surprise_threshold: 0,
        };
        let mut coordinator = WalSleepCoordinator::new(heuristics, 4);
        coordinator.record_derived_record(EvidenceId::new("rec-1"));
        coordinator.record_derived_record(EvidenceId::new("rec-2"));
        coordinator.record_consistency_verdict(ConsistencyVerdict::Accept);
        coordinator.record_integration_score(9000);

        let metrics = coordinator.recent_metrics();
        let trigger = coordinator.evaluate(&metrics);

        assert_eq!(trigger, SleepTrigger::Density);
        assert_eq!(coordinator.state.last_cycle_id, 1);
        assert_eq!(coordinator.state.records_since_last, 0);
    }

    #[test]
    fn sleep_trigger_instability_is_deterministic() {
        let heuristics = SleepHeuristics {
            min_records_since_last: 10,
            max_instability: 2000,
            min_integration: 0,
            critical_surprise_threshold: 0,
        };
        let mut coordinator = WalSleepCoordinator::new(heuristics, 4);
        coordinator.record_consistency_verdict(ConsistencyVerdict::Reject);
        coordinator.record_consistency_verdict(ConsistencyVerdict::Accept);
        coordinator.record_integration_score(9000);

        let metrics = coordinator.recent_metrics();
        let trigger = coordinator.evaluate(&metrics);

        assert_eq!(trigger, SleepTrigger::Instability);
        assert_eq!(coordinator.state.last_cycle_id, 1);
    }

    #[test]
    fn policy_denial_prevents_sleep_report_append() {
        let heuristics = SleepHeuristics {
            min_records_since_last: 1,
            max_instability: 10_000,
            min_integration: 0,
            critical_surprise_threshold: 0,
        };
        let mut coordinator = WalSleepCoordinator::new(heuristics, 4);
        coordinator.record_derived_record(EvidenceId::new("rec-1"));
        coordinator.record_consistency_verdict(ConsistencyVerdict::Accept);
        coordinator.record_integration_score(9000);

        let archive = Arc::new(InMemoryArchive::new());
        let sleep_bus = InMemoryBus::new();
        let triggered_bus = InMemoryBus::new();
        let triggered_rx = triggered_bus.subscribe();
        let policy = DefaultPolicyEcology::new();
        let rsa = MockRsaEngine::new();
        let openevolve = MockOpenEvolvePort::default();
        let runner = SleepCoordinator::new(policy, rsa, openevolve, archive.clone(), sleep_bus);

        let result = coordinator.maybe_trigger(&runner, &triggered_bus);

        assert!(result.is_none());
        assert!(archive.list().is_empty());
        let triggered = triggered_rx.recv().expect("sleep triggered event");
        assert_eq!(triggered.reason, SleepTrigger::Density);
    }

    #[test]
    fn policy_allow_appends_sleep_report() {
        let heuristics = SleepHeuristics {
            min_records_since_last: 1,
            max_instability: 10_000,
            min_integration: 0,
            critical_surprise_threshold: 0,
        };
        let mut coordinator = WalSleepCoordinator::new(heuristics, 4);
        coordinator.record_derived_record(EvidenceId::new("rec-1"));
        coordinator.record_consistency_verdict(ConsistencyVerdict::Accept);
        coordinator.record_integration_score(9000);

        let archive = Arc::new(InMemoryArchive::new());
        let sleep_bus = InMemoryBus::new();
        let triggered_bus = InMemoryBus::new();
        let policy = PolicyEcology::new(1, vec![PolicyRule::AllowSleepPhase], PolicyWeights);
        let rsa = MockRsaEngine::new();
        let openevolve = MockOpenEvolvePort::default();
        let runner = SleepCoordinator::new(policy, rsa, openevolve, archive.clone(), sleep_bus);

        let result = coordinator.maybe_trigger(&runner, &triggered_bus);

        assert!(result.is_some());
        assert_eq!(archive.list().len(), 1);
    }

    #[test]
    fn repeated_critical_surprise_triggers_sleep() {
        let heuristics = SleepHeuristics {
            min_records_since_last: 10,
            max_instability: 10_000,
            min_integration: 0,
            critical_surprise_threshold: 2,
        };
        let mut coordinator = WalSleepCoordinator::new(heuristics, 4);
        coordinator.record_surprise_band(SurpriseBand::Critical);
        coordinator.record_surprise_band(SurpriseBand::Critical);
        coordinator.record_integration_score(9000);

        let metrics = coordinator.recent_metrics();
        let trigger = coordinator.evaluate(&metrics);

        assert_eq!(trigger, SleepTrigger::SurpriseCritical);
        assert_eq!(coordinator.state.last_cycle_id, 1);
    }
}
