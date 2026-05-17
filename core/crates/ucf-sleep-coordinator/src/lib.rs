#![forbid(unsafe_code)]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use blake3::Hasher;
use ucf_bus::BusPublisher;
use ucf_policy_ecology::{ConsistencyVerdict, SleepPhaseGate};
use ucf_predictive_coding::SurpriseBand;
use ucf_replay::{
    MinimalSpineReplayAppliedBoundary, MinimalSpineReplayAuditStatus,
    MinimalSpineReplayScheduleAudit,
};
use ucf_rsa::{RsaEngine, SleepCoordinator, SleepReportReady};
use ucf_structural_store::{StructuralCycleStats, StructuralDeltaProposal};
use ucf_types::{Digest32, EvidenceId};

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
