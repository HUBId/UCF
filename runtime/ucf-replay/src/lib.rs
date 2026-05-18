#![forbid(unsafe_code)]

use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::{
    collections::{BTreeSet, HashMap},
    fmt::Write,
};

use hex::FromHex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use ucf_archive_store::{ArchiveAppender, ArchiveStore, RecordKind, RecordMeta};
use ucf_commit::commit_replay_token;
use ucf_compute::ComputeSignalsSummary as RecomputedComputeSummary;
use ucf_compute::{
    build_backend, compute_input_from_control, stable_budget_profile_id, ComputeBackendConfig,
    ComputeBackendKind, ComputeBudget,
};
use ucf_core::types::{SimTime, Tick, WindowId};
use ucf_ess::v1::{
    AuditPayload, CapabilityIssuanceRecord, ExperienceKind, ExperiencePayload, ExperienceRecord,
    LfmSummaryRecord,
};
use ucf_evidence::{EvidenceEnvelope, EvidenceStore};
use ucf_frames::v1::{
    ChannelCode, ComputeSignalsSummary, ControlFrame, CorrelationId, DecisionFrame, Intent,
    IntentId, IntentKind,
};
use ucf_types::consolidation::{MilestoneTier, ReplayScheduled, ReplayToken};
use ucf_types::v1::spec::{Digest as ProtoDigest, ProofEnvelope};
use ucf_types::{
    quantize_unit, Digest32, EvidenceId, LogicalTime, WallTime, CANONICAL_UNIT_QUANT_MAX,
};

const REPORT_CAP: usize = 1000;
const REPLAY_DIVERGENCE_CAP: usize = 64;
static UCF_COMPUTE_CHAIN_MISMATCH_TOTAL: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReplayStrictness {
    VerifyOnly,
    RecomputeStages,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayPlan {
    pub t0: u64,
    pub t1: u64,
    pub expected_backend_pack_digest: Option<[u8; 32]>,
    pub strictness: ReplayStrictness,
    pub stop_on_first_divergence: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReplayOverallStatus {
    Ok,
    DriftFound,
    MissingData,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReplayComponent {
    BackendPack,
    World,
    Sae,
    Ssm,
    Lfm,
    Risk,
    Nsr,
    Coherence,
    Governor,
    Issuance,
    Output,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Divergence {
    pub t: u64,
    pub component: ReplayComponent,
    pub expected_digest: String,
    pub observed_digest: String,
    pub hint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayCounters {
    pub missing_records: u64,
    pub mismatched_digests: u64,
    pub degraded_steps: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayReport {
    pub range: (u64, u64),
    pub overall_status: ReplayOverallStatus,
    pub first_divergence: Option<Divergence>,
    pub counters: ReplayCounters,
    pub details: Vec<Divergence>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReplayMode {
    ComputeOnly,
    DecisionScoring,
    FullNoAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplaySpec {
    pub from_tick: u64,
    pub to_tick: u64,
    pub backend_override: Option<ComputeBackendKind>,
    pub seed_override: Option<u64>,
    pub budget_override: Option<u32>,
    pub mode: ReplayMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffPolicy {
    pub eps: f32,
    pub digest_allowlist: Vec<String>,
}

impl DiffPolicy {
    pub fn for_backend(backend: &str) -> Self {
        let eps = if backend == "stub" { 1e-6 } else { 1e-5 };
        Self {
            eps,
            digest_allowlist: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayResult {
    pub total_items: usize,
    pub matched: usize,
    pub drifted: usize,
    pub unreplayable: usize,
    pub items: Vec<ReplayItem>,
    pub truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayItem {
    pub decision_id: u64,
    pub correlation_id: u64,
    pub persisted: PersistedSummary,
    pub recomputed: Option<RecomputedSummary>,
    pub diff: DiffSummary,
    pub status: ReplayStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedSummary {
    pub backend: String,
    pub risk: f32,
    pub confidence: f32,
    pub surprise: f32,
    pub pressure: f32,
    pub risk_quality: Option<u8>,
    pub spikes_digest_hex: String,
    pub context_digest_hex: Option<String>,
    pub chain_digest_hex: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecomputedSummary {
    pub backend: String,
    pub risk: f32,
    pub confidence: f32,
    pub surprise: f32,
    pub pressure: f32,
    pub risk_quality: Option<u8>,
    pub spikes_digest_hex: String,
    pub context_digest_hex: Option<String>,
    pub chain_digest_hex: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffSummary {
    pub risk_abs: Option<f32>,
    pub confidence_abs: Option<f32>,
    pub surprise_abs: Option<f32>,
    pub pressure_abs: Option<f32>,
    pub pass: bool,
    pub reasons: Vec<DriftReason>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReplayStatus {
    Match,
    Drift,
    Unreplayable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DriftReason {
    DigestMismatch {
        field: String,
        expected_prefix: String,
        got_prefix: String,
    },
    FloatMismatch {
        field: String,
        expected: f32,
        got: f32,
        abs_diff: f32,
    },
    MissingPersistedField {
        field: String,
    },
    BackendUnavailable {
        backend_profile: String,
    },
    DecisionScoringUnavailable,
}

#[derive(Debug, Error)]
pub enum ReplayError {
    #[error("invalid minimal spine replay token input: {0}")]
    InvalidMinimalSpineReplayTokenInput(&'static str),
    #[error("invalid minimal spine replay schedule input: {0}")]
    InvalidMinimalSpineReplayScheduleInput(&'static str),
    #[error("invalid minimal spine replay applied boundary input: {0}")]
    InvalidMinimalSpineReplayAppliedBoundaryInput(&'static str),
    #[error("invalid minimal spine replay append input: {0}")]
    InvalidMinimalSpineReplayAppendInput(&'static str),
    #[error("minimal spine replay append readback missing")]
    MinimalSpineReplayAppendReadbackMissing,
    #[error("minimal spine replay append readback mismatch")]
    MinimalSpineReplayAppendReadbackMismatch,
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Version for the Minimal Spine deterministic replay token builder output.
pub const MINIMAL_SPINE_REPLAY_TOKEN_BUILD_OUTPUT_VERSION: u32 = 1;

/// Source marker used by replay-token build outputs derived from bounded consolidation artifacts.
pub const MINIMAL_SPINE_REPLAY_TOKEN_SOURCE: &str =
    "minimal_spine_v1_macro_consolidation_replay_token";

/// Version for the Minimal Spine deterministic replay schedule builder output.
pub const MINIMAL_SPINE_REPLAY_SCHEDULE_BUILD_OUTPUT_VERSION: u32 = 1;

/// Source marker used by replay schedules derived from Minimal Spine replay token outputs.
pub const MINIMAL_SPINE_REPLAY_SCHEDULE_SOURCE: &str = "minimal_spine_v1_replay_schedule_builder";

/// Version for the Minimal Spine verify-only replay schedule audit output.
pub const MINIMAL_SPINE_REPLAY_SCHEDULE_AUDIT_VERSION: u32 = 1;

/// Source marker used by verify-only audits over Minimal Spine replay schedules.
pub const MINIMAL_SPINE_REPLAY_SCHEDULE_AUDIT_SOURCE: &str =
    "minimal_spine_v1_replay_schedule_verify_only_audit";

/// Version for the Minimal Spine local ReplayApplied boundary marker.
pub const MINIMAL_SPINE_REPLAY_APPLIED_BOUNDARY_VERSION: u32 = 1;

/// Source marker used by local replay-subsystem applied boundary records.
pub const MINIMAL_SPINE_REPLAY_APPLIED_BOUNDARY_SOURCE: &str =
    "minimal_spine_v1_replay_applied_local_boundary";

/// Version for the explicit Minimal Spine replay append/readback contract payload.
pub const MINIMAL_SPINE_REPLAY_APPEND_PAYLOAD_VERSION: u32 = 1;

/// Explicit contract marker carried in the replay Evidence/Archive append payload.
pub const MINIMAL_SPINE_REPLAY_APPEND_CONTRACT: &str = "minimal_spine_replay_append_v1";

/// Extension kind used for bounded Replay append proof records in archive-store.
///
/// `RecordKind::ReplayToken` and `RecordKind::ReplayApplied` are existing broad protocol-facing
/// kinds. Prompt 65 needs one bounded audit/provenance payload that wraps token, schedule, audit,
/// and local applied-boundary digests without claiming runtime replay application. Therefore the
/// contract allocates `Other(65)` instead of changing archive-store schema or reusing a broader
/// runtime-facing kind.
pub const MINIMAL_SPINE_REPLAY_APPEND_ARCHIVE_KIND: RecordKind = RecordKind::Other(65);

/// Meaning marker for the explicit replay append contract.
pub const MINIMAL_SPINE_REPLAY_APPEND_MEANING: &str =
    "audit_provenance_persistence_only_no_replay_execution";

/// Deterministic bounded Replay append payload persisted through Evidence/Archive.
///
/// This payload is provenance only. It preserves digest links for the bounded Replay artifacts and
/// carries explicit false boundary flags for runtime replay execution, scheduler activation,
/// Sleep, Geist/ISM, identity, and Gateway visibility.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MinimalSpineReplayAppendPayload {
    pub version: u32,
    pub append_contract: String,
    pub replay_token_digests: Vec<Digest32>,
    pub replay_schedule_digest: Digest32,
    pub replay_audit_digest: Digest32,
    pub replay_applied_boundary_digest: Digest32,
    pub token_count: u32,
    pub schedule_source: &'static str,
    pub audit_source: &'static str,
    pub applied_boundary_source: &'static str,
    pub source: &'static str,
    pub runtime_executed: bool,
    pub scheduler_activated: bool,
    pub sleep_triggered: bool,
    pub geist_ingested: bool,
    pub ism_written: bool,
    pub identity_anchor: bool,
    pub gateway_visible: bool,
    pub evidence_archive_appended_meaning: &'static str,
}

impl MinimalSpineReplayAppendPayload {
    pub fn from_artifacts(
        schedule: &MinimalSpineReplayScheduleBuildOutput,
        audit: &MinimalSpineReplayScheduleAudit,
        boundary: &MinimalSpineReplayAppliedBoundary,
    ) -> Result<Self, ReplayError> {
        validate_minimal_spine_replay_append_inputs(schedule, audit, boundary)?;
        let payload = Self {
            version: MINIMAL_SPINE_REPLAY_APPEND_PAYLOAD_VERSION,
            append_contract: MINIMAL_SPINE_REPLAY_APPEND_CONTRACT.to_string(),
            replay_token_digests: schedule.replay_token_digests.clone(),
            replay_schedule_digest: schedule.schedule_digest,
            replay_audit_digest: audit.audit_digest,
            replay_applied_boundary_digest: boundary.applied_boundary_digest,
            token_count: schedule.token_count,
            schedule_source: schedule.source,
            audit_source: audit.source,
            applied_boundary_source: boundary.source,
            source: MINIMAL_SPINE_REPLAY_APPEND_CONTRACT,
            runtime_executed: false,
            scheduler_activated: false,
            sleep_triggered: false,
            geist_ingested: false,
            ism_written: false,
            identity_anchor: false,
            gateway_visible: false,
            evidence_archive_appended_meaning: MINIMAL_SPINE_REPLAY_APPEND_MEANING,
        };
        if !payload.validate_links_nonzero() {
            return Err(ReplayError::InvalidMinimalSpineReplayAppendInput(
                "append payload links must be non-empty/non-zero",
            ));
        }
        Ok(payload)
    }

    pub fn deterministic_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_u32_be(&mut out, self.version);
        push_str32(&mut out, &self.append_contract);
        push_digest_vec(&mut out, &self.replay_token_digests);
        push_digest32(&mut out, self.replay_schedule_digest);
        push_digest32(&mut out, self.replay_audit_digest);
        push_digest32(&mut out, self.replay_applied_boundary_digest);
        push_u32_be(&mut out, self.token_count);
        push_str32(&mut out, self.schedule_source);
        push_str32(&mut out, self.audit_source);
        push_str32(&mut out, self.applied_boundary_source);
        push_str32(&mut out, self.source);
        out.push(u8::from(self.runtime_executed));
        out.push(u8::from(self.scheduler_activated));
        out.push(u8::from(self.sleep_triggered));
        out.push(u8::from(self.geist_ingested));
        out.push(u8::from(self.ism_written));
        out.push(u8::from(self.identity_anchor));
        out.push(u8::from(self.gateway_visible));
        push_str32(&mut out, self.evidence_archive_appended_meaning);
        out
    }

    pub fn digest(&self) -> Digest32 {
        digest_sha256_domain(
            b"ucf.replay.minimal_spine.append_payload.v1",
            &self.deterministic_bytes(),
        )
    }

    pub fn validate_links_nonzero(&self) -> bool {
        self.version == MINIMAL_SPINE_REPLAY_APPEND_PAYLOAD_VERSION
            && self.append_contract == MINIMAL_SPINE_REPLAY_APPEND_CONTRACT
            && !self.replay_token_digests.is_empty()
            && self.replay_token_digests.len() == self.token_count as usize
            && self
                .replay_token_digests
                .iter()
                .all(|digest| !is_zero_digest(*digest))
            && !is_zero_digest(self.replay_schedule_digest)
            && !is_zero_digest(self.replay_audit_digest)
            && !is_zero_digest(self.replay_applied_boundary_digest)
            && self.token_count > 0
            && !self.schedule_source.is_empty()
            && !self.audit_source.is_empty()
            && !self.applied_boundary_source.is_empty()
            && !self.source.is_empty()
            && !self.runtime_executed
            && !self.scheduler_activated
            && !self.sleep_triggered
            && !self.geist_ingested
            && !self.ism_written
            && !self.identity_anchor
            && !self.gateway_visible
            && self.evidence_archive_appended_meaning == MINIMAL_SPINE_REPLAY_APPEND_MEANING
    }
}

/// Result of the explicit Minimal Spine Replay append/readback contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MinimalSpineReplayAppendResult {
    pub payload_digest: Digest32,
    pub replay_token_digests: Vec<Digest32>,
    pub replay_schedule_digest: Digest32,
    pub replay_audit_digest: Digest32,
    pub replay_applied_boundary_digest: Digest32,
    pub token_count: u32,
    pub appended_evidence_id: EvidenceId,
    pub archive_key: Digest32,
    pub archive_record_digest: Digest32,
    pub readback_digest: Digest32,
}

/// Explicitly append bounded Replay audit/provenance evidence and read it back.
///
/// This helper is the only Replay append surface. It persists a deterministic provenance payload
/// through the existing EvidenceStore and ArchiveStore APIs, then immediately verifies readback.
/// It does not execute replay, activate a scheduler/queue/worker, trigger Sleep or Geist/ISM,
/// write identity anchors, expose Gateway semantics, or create a second event log.
pub fn append_minimal_spine_replay_record<E, S>(
    schedule: &MinimalSpineReplayScheduleBuildOutput,
    audit: &MinimalSpineReplayScheduleAudit,
    boundary: &MinimalSpineReplayAppliedBoundary,
    evidence_store: &E,
    archive_store: &S,
    archive_appender: &mut ArchiveAppender,
) -> Result<MinimalSpineReplayAppendResult, ReplayError>
where
    E: EvidenceStore,
    S: ArchiveStore,
{
    let payload = MinimalSpineReplayAppendPayload::from_artifacts(schedule, audit, boundary)?;
    let payload_bytes = payload.deterministic_bytes();
    let payload_digest = payload.digest();
    let appended_evidence_id = EvidenceId::new(format!(
        "minimal-spine-replay-append-{}",
        hex::encode(payload_digest.as_bytes())
    ));
    let proof = ProofEnvelope {
        envelope_id: format!(
            "{}:{}",
            MINIMAL_SPINE_REPLAY_APPEND_CONTRACT,
            hex::encode(payload_digest.as_bytes())
        ),
        payload: payload_bytes,
        payload_digest: Some(ProtoDigest {
            algorithm: "sha256".to_string(),
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
        MINIMAL_SPINE_REPLAY_APPEND_ARCHIVE_KIND,
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
        .ok_or(ReplayError::MinimalSpineReplayAppendReadbackMissing)?;
    let readback_digest = digest_replay_evidence_envelope(&readback);
    let archive_readback = archive_store
        .get(archive_record.key)
        .ok_or(ReplayError::MinimalSpineReplayAppendReadbackMissing)?;
    if archive_readback != archive_record {
        return Err(ReplayError::MinimalSpineReplayAppendReadbackMismatch);
    }

    Ok(MinimalSpineReplayAppendResult {
        payload_digest,
        replay_token_digests: payload.replay_token_digests,
        replay_schedule_digest: payload.replay_schedule_digest,
        replay_audit_digest: payload.replay_audit_digest,
        replay_applied_boundary_digest: payload.replay_applied_boundary_digest,
        token_count: payload.token_count,
        appended_evidence_id,
        archive_key: archive_record.key,
        archive_record_digest,
        readback_digest,
    })
}

fn validate_minimal_spine_replay_append_inputs(
    schedule: &MinimalSpineReplayScheduleBuildOutput,
    audit: &MinimalSpineReplayScheduleAudit,
    boundary: &MinimalSpineReplayAppliedBoundary,
) -> Result<(), ReplayError> {
    if schedule.replay_token_digests.is_empty() {
        return Err(ReplayError::InvalidMinimalSpineReplayAppendInput(
            "schedule token list must be non-empty",
        ));
    }
    if schedule
        .replay_token_digests
        .iter()
        .any(|digest| is_zero_digest(*digest))
    {
        return Err(ReplayError::InvalidMinimalSpineReplayAppendInput(
            "schedule token digests must be non-zero",
        ));
    }
    if schedule.schedule_digest != schedule.digest() {
        return Err(ReplayError::InvalidMinimalSpineReplayAppendInput(
            "schedule digest mismatch",
        ));
    }
    if audit.status != MinimalSpineReplayAuditStatus::Pass {
        return Err(ReplayError::InvalidMinimalSpineReplayAppendInput(
            "audit status must be pass",
        ));
    }
    if audit.audit_digest != audit.digest() {
        return Err(ReplayError::InvalidMinimalSpineReplayAppendInput(
            "audit digest mismatch",
        ));
    }
    if audit.schedule_digest != schedule.schedule_digest
        || audit.recomputed_schedule_digest != schedule.schedule_digest
    {
        return Err(ReplayError::InvalidMinimalSpineReplayAppendInput(
            "audit must match schedule digest",
        ));
    }
    if audit.token_count != schedule.token_count
        || audit.token_digests != schedule.replay_token_digests
    {
        return Err(ReplayError::InvalidMinimalSpineReplayAppendInput(
            "audit must preserve schedule token provenance",
        ));
    }
    if boundary.applied_boundary_digest != boundary.digest() {
        return Err(ReplayError::InvalidMinimalSpineReplayAppendInput(
            "applied boundary digest mismatch",
        ));
    }
    if boundary.audit_digest != audit.audit_digest
        || boundary.schedule_digest != schedule.schedule_digest
        || boundary.token_count != schedule.token_count
    {
        return Err(ReplayError::InvalidMinimalSpineReplayAppendInput(
            "applied boundary must match audit and schedule provenance",
        ));
    }
    if schedule.token_count == 0
        || usize::try_from(schedule.token_count).ok() != Some(schedule.replay_token_digests.len())
    {
        return Err(ReplayError::InvalidMinimalSpineReplayAppendInput(
            "token count must match non-empty token list",
        ));
    }
    if schedule.applied
        || schedule.sleep_cycle
        || schedule.geist_ingested
        || schedule.identity_anchor
        || schedule.evidence_archive_appended
        || audit.applied
        || audit.replay_completed
        || audit.sleep_cycle
        || audit.geist_ingested
        || audit.identity_anchor
        || audit.evidence_archive_appended
        || !boundary.replay_subsystem_applied
        || boundary.geist_ingested
        || boundary.ism_written
        || boundary.identity_anchor
        || boundary.sleep_completed
        || boundary.evidence_archive_appended
        || boundary.gateway_visible
    {
        return Err(ReplayError::InvalidMinimalSpineReplayAppendInput(
            "replay append inputs must not carry forbidden side-effect flags",
        ));
    }
    Ok(())
}

fn digest_replay_evidence_envelope(envelope: &EvidenceEnvelope) -> Digest32 {
    let mut out = Vec::new();
    push_str32(&mut out, envelope.evidence_id.as_str());
    match &envelope.proof {
        Some(proof) => {
            out.push(1);
            push_str32(&mut out, &proof.envelope_id);
            push_u32_be(
                &mut out,
                u32::try_from(proof.payload.len())
                    .expect("minimal spine replay append proof payload length fits u32"),
            );
            out.extend_from_slice(&proof.payload);
            match &proof.payload_digest {
                Some(digest) => {
                    out.push(1);
                    push_str32(&mut out, &digest.algorithm);
                    push_u32_be(
                        &mut out,
                        u32::try_from(digest.value.len())
                            .expect("minimal spine replay append digest length fits u32"),
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
                    .expect("minimal spine replay append vrf tag count fits u32"),
            );
            push_u32_be(
                &mut out,
                u32::try_from(proof.signature_ids.len())
                    .expect("minimal spine replay append signature id count fits u32"),
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
    digest_sha256_domain(b"ucf.replay.minimal_spine.append_readback.v1", &out)
}

/// PASS/FAIL status for the verify-only Minimal Spine replay schedule audit.
///
/// `Pass` means the schedule value is internally consistent. It does not mean replay was applied,
/// completed, archived, ingested by Geist/ISM, or attached to identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MinimalSpineReplayAuditStatus {
    Pass,
    Fail,
}

impl MinimalSpineReplayAuditStatus {
    fn code(self) -> u8 {
        match self {
            Self::Pass => 1,
            Self::Fail => 2,
        }
    }
}

/// Deterministic failure reasons emitted by the verify-only schedule audit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MinimalSpineReplayAuditFailureReason {
    VersionMismatch,
    TokenCountMismatch,
    EmptyTokenOrder,
    ScheduledTokenCountMismatch,
    ProvenanceCountMismatch,
    TokenBuildOutputDigestCountMismatch,
    DuplicateReplayTokenDigest,
    ScheduleDigestMismatch,
    SourceEmpty,
    AppliedFlagSet,
    SleepCycleFlagSet,
    GeistIngestedFlagSet,
    IdentityAnchorFlagSet,
    EvidenceArchiveAppendedFlagSet,
    ProvenanceOrderMismatch,
    ProvenanceDigestOrderMismatch,
    ScheduledTokenCommitMismatch,
}

impl MinimalSpineReplayAuditFailureReason {
    fn code(self) -> u8 {
        match self {
            Self::VersionMismatch => 1,
            Self::TokenCountMismatch => 2,
            Self::EmptyTokenOrder => 3,
            Self::ScheduledTokenCountMismatch => 4,
            Self::ProvenanceCountMismatch => 5,
            Self::TokenBuildOutputDigestCountMismatch => 6,
            Self::DuplicateReplayTokenDigest => 7,
            Self::ScheduleDigestMismatch => 8,
            Self::SourceEmpty => 9,
            Self::AppliedFlagSet => 10,
            Self::SleepCycleFlagSet => 11,
            Self::GeistIngestedFlagSet => 12,
            Self::IdentityAnchorFlagSet => 13,
            Self::EvidenceArchiveAppendedFlagSet => 14,
            Self::ProvenanceOrderMismatch => 15,
            Self::ProvenanceDigestOrderMismatch => 16,
            Self::ScheduledTokenCommitMismatch => 17,
        }
    }
}

/// Verify-only audit report for a Minimal Spine replay schedule value.
///
/// This report is an audit value only. It has no store/appender arguments, does not mutate the
/// schedule, does not apply replay, and does not create `ReplayApplied` runtime state. All boundary
/// flags are hard-coded false in the report to prevent overclaiming.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MinimalSpineReplayScheduleAudit {
    pub version: u32,
    pub status: MinimalSpineReplayAuditStatus,
    pub failure_reasons: Vec<MinimalSpineReplayAuditFailureReason>,
    pub schedule_digest: Digest32,
    pub recomputed_schedule_digest: Digest32,
    pub token_count: u32,
    pub token_digests: Vec<Digest32>,
    pub duplicate_free: bool,
    pub truncated: bool,
    pub audit_digest: Digest32,
    pub source: &'static str,
    pub applied: bool,
    pub replay_completed: bool,
    pub sleep_cycle: bool,
    pub geist_ingested: bool,
    pub identity_anchor: bool,
    pub evidence_archive_appended: bool,
}

impl MinimalSpineReplayScheduleAudit {
    pub fn deterministic_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_u32_be(&mut out, self.version);
        out.push(self.status.code());
        push_u32_be(
            &mut out,
            u32::try_from(self.failure_reasons.len())
                .expect("minimal spine replay audit failure reason count fits u32"),
        );
        for reason in &self.failure_reasons {
            out.push(reason.code());
        }
        push_digest32(&mut out, self.schedule_digest);
        push_digest32(&mut out, self.recomputed_schedule_digest);
        push_u32_be(&mut out, self.token_count);
        push_u32_be(
            &mut out,
            u32::try_from(self.token_digests.len())
                .expect("minimal spine replay audit token count fits u32"),
        );
        for digest in &self.token_digests {
            push_digest32(&mut out, *digest);
        }
        out.push(u8::from(self.duplicate_free));
        out.push(u8::from(self.truncated));
        push_str32(&mut out, self.source);
        out.push(u8::from(self.applied));
        out.push(u8::from(self.replay_completed));
        out.push(u8::from(self.sleep_cycle));
        out.push(u8::from(self.geist_ingested));
        out.push(u8::from(self.identity_anchor));
        out.push(u8::from(self.evidence_archive_appended));
        out
    }

    pub fn digest(&self) -> Digest32 {
        digest_sha256_domain(
            b"ucf.replay.minimal_spine.schedule_verify_only_audit.v1",
            &self.deterministic_bytes(),
        )
    }
}

/// Local replay-subsystem boundary marker derived from a PASS schedule audit.
///
/// This record acknowledges only replay bookkeeping inside `ucf-replay`. It is not the broad
/// protocol/type-level `ReplayApplied`, does not execute a replay runtime apply, and hard-codes all
/// Geist/ISM/identity/Sleep/Evidence/Archive/Gateway boundary flags to false.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MinimalSpineReplayAppliedBoundary {
    pub version: u32,
    pub audit_digest: Digest32,
    pub schedule_digest: Digest32,
    pub token_count: u32,
    pub applied_boundary_digest: Digest32,
    pub source: &'static str,
    pub replay_subsystem_applied: bool,
    pub geist_ingested: bool,
    pub ism_written: bool,
    pub identity_anchor: bool,
    pub sleep_completed: bool,
    pub evidence_archive_appended: bool,
    pub gateway_visible: bool,
}

impl MinimalSpineReplayAppliedBoundary {
    pub fn deterministic_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_u32_be(&mut out, self.version);
        push_digest32(&mut out, self.audit_digest);
        push_digest32(&mut out, self.schedule_digest);
        push_u32_be(&mut out, self.token_count);
        push_str32(&mut out, self.source);
        out.push(u8::from(self.replay_subsystem_applied));
        out.push(u8::from(self.geist_ingested));
        out.push(u8::from(self.ism_written));
        out.push(u8::from(self.identity_anchor));
        out.push(u8::from(self.sleep_completed));
        out.push(u8::from(self.evidence_archive_appended));
        out.push(u8::from(self.gateway_visible));
        out
    }

    pub fn digest(&self) -> Digest32 {
        digest_sha256_domain(
            b"ucf.replay.minimal_spine.applied_local_boundary.v1",
            &self.deterministic_bytes(),
        )
    }
}

/// Build a local ReplayApplied boundary marker from a PASS verify-only schedule audit.
///
/// The output is deterministic replay bookkeeping only. It preserves audit/schedule provenance,
/// rejects FAIL audits, takes no store/scheduler/runtime/Geist/ISM/Evidence/Archive/Gateway
/// handles, and does not mutate the audit, schedule, or tokens that produced it.
pub fn build_replay_applied_boundary_from_audit(
    audit: &MinimalSpineReplayScheduleAudit,
) -> Result<MinimalSpineReplayAppliedBoundary, ReplayError> {
    if audit.status != MinimalSpineReplayAuditStatus::Pass {
        return Err(ReplayError::InvalidMinimalSpineReplayAppliedBoundaryInput(
            "audit status must be pass",
        ));
    }
    if audit.audit_digest != audit.digest() {
        return Err(ReplayError::InvalidMinimalSpineReplayAppliedBoundaryInput(
            "audit digest mismatch",
        ));
    }

    let mut boundary = MinimalSpineReplayAppliedBoundary {
        version: MINIMAL_SPINE_REPLAY_APPLIED_BOUNDARY_VERSION,
        audit_digest: audit.audit_digest,
        schedule_digest: audit.schedule_digest,
        token_count: audit.token_count,
        applied_boundary_digest: Digest32::new([0u8; Digest32::LEN]),
        source: MINIMAL_SPINE_REPLAY_APPLIED_BOUNDARY_SOURCE,
        replay_subsystem_applied: true,
        geist_ingested: false,
        ism_written: false,
        identity_anchor: false,
        sleep_completed: false,
        evidence_archive_appended: false,
        gateway_visible: false,
    };
    boundary.applied_boundary_digest = boundary.digest();
    Ok(boundary)
}

/// Pure schedule-builder configuration.
///
/// `max_tokens` is optional. When present, it is applied after deterministic digest ordering and
/// records truncation metadata without executing or applying replay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MinimalSpineReplayScheduleConfig {
    pub max_tokens: Option<usize>,
    pub source: &'static str,
}

impl Default for MinimalSpineReplayScheduleConfig {
    fn default() -> Self {
        Self {
            max_tokens: None,
            source: MINIMAL_SPINE_REPLAY_SCHEDULE_SOURCE,
        }
    }
}

/// Provenance sidecar for a scheduled replay-token record.
///
/// `ReplayScheduled` mirrors the legacy scheduler-facing shape but cannot carry the Minimal Spine
/// token-builder provenance. This sidecar keeps the ordering and provenance explicit without
/// changing the shared `ReplayScheduled` schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MinimalSpineReplayScheduledTokenProvenance {
    pub order: u32,
    pub replay_token_digest: Digest32,
    pub token_build_output_digest: Digest32,
    pub macro_candidate_digest: Digest32,
    pub macro_milestone_digest: Digest32,
    pub meso_aggregation_digest: Digest32,
    pub macro_finalization_digest: Digest32,
    pub meso_count: u32,
    pub source: &'static str,
}

/// Deterministic schedule build output for Minimal Spine replay tokens.
///
/// This is planned ordering only. It is not applied replay, not Sleep Cycle coordination, not
/// Geist/ISM ingestion, not identity anchoring, and not Evidence/Archive append authority.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MinimalSpineReplayScheduleBuildOutput {
    pub version: u32,
    pub scheduled_tokens: Vec<ReplayScheduled>,
    pub scheduled_token_provenance: Vec<MinimalSpineReplayScheduledTokenProvenance>,
    pub replay_token_digests: Vec<Digest32>,
    pub token_build_output_digests: Vec<Digest32>,
    pub schedule_digest: Digest32,
    pub token_count: u32,
    pub truncated: bool,
    pub source: &'static str,
    pub applied: bool,
    pub sleep_cycle: bool,
    pub geist_ingested: bool,
    pub identity_anchor: bool,
    pub evidence_archive_appended: bool,
}

impl MinimalSpineReplayScheduleBuildOutput {
    pub fn deterministic_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_u32_be(&mut out, self.version);
        push_u32_be(&mut out, self.token_count);
        out.push(u8::from(self.truncated));
        push_str32(&mut out, self.source);
        out.push(u8::from(self.applied));
        out.push(u8::from(self.sleep_cycle));
        out.push(u8::from(self.geist_ingested));
        out.push(u8::from(self.identity_anchor));
        out.push(u8::from(self.evidence_archive_appended));
        push_u32_be(
            &mut out,
            u32::try_from(self.scheduled_tokens.len())
                .expect("minimal spine replay schedule length fits u32"),
        );
        for (scheduled, provenance) in self
            .scheduled_tokens
            .iter()
            .zip(self.scheduled_token_provenance.iter())
        {
            push_u32_be(&mut out, provenance.order);
            out.push(scheduled.tier as u8);
            push_digest32(&mut out, scheduled.target);
            out.extend_from_slice(&scheduled.budget.to_be_bytes());
            out.extend_from_slice(&scheduled.redaction.to_be_bytes());
            push_digest32(&mut out, scheduled.commit);
            push_digest32(&mut out, provenance.replay_token_digest);
            push_digest32(&mut out, provenance.token_build_output_digest);
            push_digest32(&mut out, provenance.macro_candidate_digest);
            push_digest32(&mut out, provenance.macro_milestone_digest);
            push_digest32(&mut out, provenance.meso_aggregation_digest);
            push_digest32(&mut out, provenance.macro_finalization_digest);
            push_u32_be(&mut out, provenance.meso_count);
            push_str32(&mut out, provenance.source);
        }
        out
    }

    pub fn digest(&self) -> Digest32 {
        digest_sha256_domain(
            b"ucf.replay.minimal_spine.schedule_build_output.v1",
            &self.deterministic_bytes(),
        )
    }
}

/// Verify a Minimal Spine replay schedule value without applying replay or appending evidence.
///
/// The audit is a pure deterministic consistency check over the schedule value. It reports PASS or
/// FAIL plus stable digests and metadata, but it never mutates the input, takes no store/appender
/// handles, does not create applied replay state, and keeps all runtime/authority boundary flags
/// false in the audit output.
pub fn verify_minimal_spine_replay_schedule(
    schedule: &MinimalSpineReplayScheduleBuildOutput,
) -> MinimalSpineReplayScheduleAudit {
    let recomputed_schedule_digest = schedule.digest();
    let mut reasons = Vec::new();

    if schedule.version != MINIMAL_SPINE_REPLAY_SCHEDULE_BUILD_OUTPUT_VERSION {
        reasons.push(MinimalSpineReplayAuditFailureReason::VersionMismatch);
    }
    if usize::try_from(schedule.token_count).ok() != Some(schedule.replay_token_digests.len()) {
        reasons.push(MinimalSpineReplayAuditFailureReason::TokenCountMismatch);
    }
    if schedule.replay_token_digests.is_empty() {
        reasons.push(MinimalSpineReplayAuditFailureReason::EmptyTokenOrder);
    }
    if schedule.scheduled_tokens.len() != schedule.replay_token_digests.len() {
        reasons.push(MinimalSpineReplayAuditFailureReason::ScheduledTokenCountMismatch);
    }
    if schedule.scheduled_token_provenance.len() != schedule.replay_token_digests.len() {
        reasons.push(MinimalSpineReplayAuditFailureReason::ProvenanceCountMismatch);
    }
    if schedule.token_build_output_digests.len() != schedule.replay_token_digests.len() {
        reasons.push(MinimalSpineReplayAuditFailureReason::TokenBuildOutputDigestCountMismatch);
    }

    let duplicate_free = replay_digest_order_is_duplicate_free(&schedule.replay_token_digests);
    if !duplicate_free {
        reasons.push(MinimalSpineReplayAuditFailureReason::DuplicateReplayTokenDigest);
    }
    if schedule.schedule_digest != recomputed_schedule_digest {
        reasons.push(MinimalSpineReplayAuditFailureReason::ScheduleDigestMismatch);
    }
    if schedule.source.is_empty() {
        reasons.push(MinimalSpineReplayAuditFailureReason::SourceEmpty);
    }
    if schedule.applied {
        reasons.push(MinimalSpineReplayAuditFailureReason::AppliedFlagSet);
    }
    if schedule.sleep_cycle {
        reasons.push(MinimalSpineReplayAuditFailureReason::SleepCycleFlagSet);
    }
    if schedule.geist_ingested {
        reasons.push(MinimalSpineReplayAuditFailureReason::GeistIngestedFlagSet);
    }
    if schedule.identity_anchor {
        reasons.push(MinimalSpineReplayAuditFailureReason::IdentityAnchorFlagSet);
    }
    if schedule.evidence_archive_appended {
        reasons.push(MinimalSpineReplayAuditFailureReason::EvidenceArchiveAppendedFlagSet);
    }

    for (index, provenance) in schedule.scheduled_token_provenance.iter().enumerate() {
        let expected_order =
            u32::try_from(index).expect("minimal spine replay audit index fits u32");
        if provenance.order != expected_order {
            reasons.push(MinimalSpineReplayAuditFailureReason::ProvenanceOrderMismatch);
            break;
        }
    }

    for (index, replay_token_digest) in schedule.replay_token_digests.iter().enumerate() {
        if schedule
            .scheduled_token_provenance
            .get(index)
            .is_some_and(|provenance| provenance.replay_token_digest != *replay_token_digest)
        {
            reasons.push(MinimalSpineReplayAuditFailureReason::ProvenanceDigestOrderMismatch);
            break;
        }
        if schedule
            .scheduled_tokens
            .get(index)
            .is_some_and(|scheduled| scheduled.commit != *replay_token_digest)
        {
            reasons.push(MinimalSpineReplayAuditFailureReason::ScheduledTokenCommitMismatch);
            break;
        }
    }

    reasons.sort_unstable();
    reasons.dedup();
    let status = if reasons.is_empty() {
        MinimalSpineReplayAuditStatus::Pass
    } else {
        MinimalSpineReplayAuditStatus::Fail
    };

    let mut audit = MinimalSpineReplayScheduleAudit {
        version: MINIMAL_SPINE_REPLAY_SCHEDULE_AUDIT_VERSION,
        status,
        failure_reasons: reasons,
        schedule_digest: schedule.schedule_digest,
        recomputed_schedule_digest,
        token_count: schedule.token_count,
        token_digests: schedule.replay_token_digests.clone(),
        duplicate_free,
        truncated: schedule.truncated,
        audit_digest: Digest32::new([0u8; Digest32::LEN]),
        source: MINIMAL_SPINE_REPLAY_SCHEDULE_AUDIT_SOURCE,
        applied: false,
        replay_completed: false,
        sleep_cycle: false,
        geist_ingested: false,
        identity_anchor: false,
        evidence_archive_appended: false,
    };
    audit.audit_digest = audit.digest();
    audit
}

fn replay_digest_order_is_duplicate_free(digests: &[Digest32]) -> bool {
    let mut seen = BTreeSet::new();
    digests.iter().all(|digest| seen.insert(*digest.as_bytes()))
}

/// Build a pure deterministic planned replay schedule from Minimal Spine replay-token outputs.
///
/// Inputs are normalized by ascending `replay_token_digest`. Duplicate token digests are rejected.
/// An optional positive cap truncates after sorting and records `truncated = true`. The builder has
/// no store/appender arguments and performs no replay application, queue activation, Sleep Cycle
/// work, Geist/ISM ingestion, identity anchoring, or Evidence/Archive append.
pub fn build_replay_schedule_from_minimal_spine_tokens(
    tokens: &[MinimalSpineReplayTokenBuildOutput],
    config: MinimalSpineReplayScheduleConfig,
) -> Result<MinimalSpineReplayScheduleBuildOutput, ReplayError> {
    validate_minimal_spine_replay_schedule_config(config)?;
    if tokens.is_empty() {
        return Err(ReplayError::InvalidMinimalSpineReplayScheduleInput(
            "token list must be non-empty",
        ));
    }

    let mut seen = BTreeSet::new();
    for token in tokens {
        validate_minimal_spine_replay_schedule_token(token)?;
        if !seen.insert(*token.replay_token_digest.as_bytes()) {
            return Err(ReplayError::InvalidMinimalSpineReplayScheduleInput(
                "duplicate replay token digest",
            ));
        }
    }

    let mut ordered: Vec<MinimalSpineReplayTokenBuildOutput> = tokens.to_vec();
    ordered.sort_by_key(|token| *token.replay_token_digest.as_bytes());

    let truncated = config
        .max_tokens
        .is_some_and(|max_tokens| ordered.len() > max_tokens);
    if let Some(max_tokens) = config.max_tokens {
        ordered.truncate(max_tokens);
    }

    let mut scheduled_tokens = Vec::with_capacity(ordered.len());
    let mut scheduled_token_provenance = Vec::with_capacity(ordered.len());
    let mut replay_token_digests = Vec::with_capacity(ordered.len());
    let mut token_build_output_digests = Vec::with_capacity(ordered.len());

    for (index, token) in ordered.iter().enumerate() {
        let scheduled = ReplayScheduled {
            tier: token.replay_token.tier,
            target: token.replay_token.target,
            budget: token.replay_token.budget,
            redaction: token.replay_token.redaction,
            commit: token.replay_token.commit,
        };
        let token_build_output_digest = token.digest();
        let order = u32::try_from(index).expect("minimal spine replay schedule order fits u32");
        scheduled_tokens.push(scheduled);
        scheduled_token_provenance.push(MinimalSpineReplayScheduledTokenProvenance {
            order,
            replay_token_digest: token.replay_token_digest,
            token_build_output_digest,
            macro_candidate_digest: token.macro_candidate_digest,
            macro_milestone_digest: token.macro_milestone_digest,
            meso_aggregation_digest: token.meso_aggregation_digest,
            macro_finalization_digest: token.macro_finalization_digest,
            meso_count: token.meso_count,
            source: token.source,
        });
        replay_token_digests.push(token.replay_token_digest);
        token_build_output_digests.push(token_build_output_digest);
    }

    let token_count = u32::try_from(scheduled_tokens.len())
        .expect("minimal spine replay schedule token count fits u32");
    let mut output = MinimalSpineReplayScheduleBuildOutput {
        version: MINIMAL_SPINE_REPLAY_SCHEDULE_BUILD_OUTPUT_VERSION,
        scheduled_tokens,
        scheduled_token_provenance,
        replay_token_digests,
        token_build_output_digests,
        schedule_digest: Digest32::new([0u8; Digest32::LEN]),
        token_count,
        truncated,
        source: config.source,
        applied: false,
        sleep_cycle: false,
        geist_ingested: false,
        identity_anchor: false,
        evidence_archive_appended: false,
    };
    output.schedule_digest = output.digest();
    Ok(output)
}

fn validate_minimal_spine_replay_schedule_config(
    config: MinimalSpineReplayScheduleConfig,
) -> Result<(), ReplayError> {
    if config.source.is_empty() {
        return Err(ReplayError::InvalidMinimalSpineReplayScheduleInput(
            "source must be non-empty",
        ));
    }
    if config.max_tokens == Some(0) {
        return Err(ReplayError::InvalidMinimalSpineReplayScheduleInput(
            "max tokens must be non-zero when configured",
        ));
    }
    Ok(())
}

fn validate_minimal_spine_replay_schedule_token(
    token: &MinimalSpineReplayTokenBuildOutput,
) -> Result<(), ReplayError> {
    if token.version != MINIMAL_SPINE_REPLAY_TOKEN_BUILD_OUTPUT_VERSION {
        return Err(ReplayError::InvalidMinimalSpineReplayScheduleInput(
            "token build output version mismatch",
        ));
    }
    if is_zero_digest(token.replay_token_digest) {
        return Err(ReplayError::InvalidMinimalSpineReplayScheduleInput(
            "replay token digest must be non-zero",
        ));
    }
    if token.replay_token_digest != token.replay_token.commit {
        return Err(ReplayError::InvalidMinimalSpineReplayScheduleInput(
            "replay token digest must match replay token commit",
        ));
    }
    if token.scheduled || token.applied {
        return Err(ReplayError::InvalidMinimalSpineReplayScheduleInput(
            "input token must be unscheduled and unapplied",
        ));
    }
    if token.sleep_cycle
        || token.geist_ingested
        || token.identity_anchor
        || token.evidence_archive_appended
    {
        return Err(ReplayError::InvalidMinimalSpineReplayScheduleInput(
            "input token must not carry side-effect boundary flags",
        ));
    }
    if token.source.is_empty() {
        return Err(ReplayError::InvalidMinimalSpineReplayScheduleInput(
            "token source must be non-empty",
        ));
    }
    Ok(())
}

/// Bounded digest-only input for building a deterministic replay intent/reference token.
///
/// This input is intentionally copied from consolidation artifacts instead of taking a direct
/// dependency on the consolidation crate. Callers should populate `macro_finalization_digest` from
/// the local `MinimalSpineMacroConsolidationFinalization::digest()` value and preserve the meso
/// aggregation/provenance digest from the macro candidate path. The builder is pure: it does not
/// schedule, apply, append evidence/archive data, trigger Sleep/Geist/ISM, or create an identity
/// anchor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MinimalSpineReplayTokenInput {
    pub macro_candidate_digest: Digest32,
    pub macro_milestone_digest: Digest32,
    pub meso_aggregation_digest: Digest32,
    pub macro_finalization_digest: Digest32,
    pub meso_count: u32,
    pub source: &'static str,
}

/// Deterministic wrapper around the existing `ReplayToken` plus missing consolidation provenance.
///
/// The existing `ReplayToken` can carry only tier/target/budget/redaction/commit, so this wrapper
/// preserves the macro candidate, macro milestone, meso aggregation, and finalization links without
/// reinterpreting the token as a schedule entry or applied replay result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MinimalSpineReplayTokenBuildOutput {
    pub version: u32,
    pub replay_token: ReplayToken,
    pub replay_token_digest: Digest32,
    pub macro_candidate_digest: Digest32,
    pub macro_milestone_digest: Digest32,
    pub meso_aggregation_digest: Digest32,
    pub macro_finalization_digest: Digest32,
    pub meso_count: u32,
    pub source: &'static str,
    pub scheduled: bool,
    pub applied: bool,
    pub sleep_cycle: bool,
    pub geist_ingested: bool,
    pub identity_anchor: bool,
    pub evidence_archive_appended: bool,
}

impl MinimalSpineReplayTokenBuildOutput {
    pub fn deterministic_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_u32_be(&mut out, self.version);
        out.push(self.replay_token.tier as u8);
        push_digest32(&mut out, self.replay_token.target);
        out.extend_from_slice(&self.replay_token.budget.to_be_bytes());
        out.extend_from_slice(&self.replay_token.redaction.to_be_bytes());
        push_digest32(&mut out, self.replay_token.commit);
        push_digest32(&mut out, self.replay_token_digest);
        push_digest32(&mut out, self.macro_candidate_digest);
        push_digest32(&mut out, self.macro_milestone_digest);
        push_digest32(&mut out, self.meso_aggregation_digest);
        push_digest32(&mut out, self.macro_finalization_digest);
        push_u32_be(&mut out, self.meso_count);
        push_str32(&mut out, self.source);
        out.push(u8::from(self.scheduled));
        out.push(u8::from(self.applied));
        out.push(u8::from(self.sleep_cycle));
        out.push(u8::from(self.geist_ingested));
        out.push(u8::from(self.identity_anchor));
        out.push(u8::from(self.evidence_archive_appended));
        out
    }

    pub fn digest(&self) -> Digest32 {
        digest_sha256_domain(
            b"ucf.replay.minimal_spine.token_build_output.v1",
            &self.deterministic_bytes(),
        )
    }
}

/// Build a pure deterministic replay intent/reference token from bounded consolidation links.
pub fn build_replay_token_from_minimal_spine_input(
    input: &MinimalSpineReplayTokenInput,
) -> Result<MinimalSpineReplayTokenBuildOutput, ReplayError> {
    validate_minimal_spine_replay_token_input(input)?;

    let token_target = digest_minimal_spine_replay_token_input(input);
    let mut token = ReplayToken {
        tier: MilestoneTier::Macro,
        target: token_target,
        budget: 0,
        redaction: 0,
        commit: Digest32::new([0u8; Digest32::LEN]),
    };
    token.commit = commit_replay_token(&token).digest;

    Ok(MinimalSpineReplayTokenBuildOutput {
        version: MINIMAL_SPINE_REPLAY_TOKEN_BUILD_OUTPUT_VERSION,
        replay_token: token,
        replay_token_digest: token.commit,
        macro_candidate_digest: input.macro_candidate_digest,
        macro_milestone_digest: input.macro_milestone_digest,
        meso_aggregation_digest: input.meso_aggregation_digest,
        macro_finalization_digest: input.macro_finalization_digest,
        meso_count: input.meso_count,
        source: input.source,
        scheduled: false,
        applied: false,
        sleep_cycle: false,
        geist_ingested: false,
        identity_anchor: false,
        evidence_archive_appended: false,
    })
}

fn digest_minimal_spine_replay_token_input(input: &MinimalSpineReplayTokenInput) -> Digest32 {
    let mut out = Vec::new();
    push_digest32(&mut out, input.macro_candidate_digest);
    push_digest32(&mut out, input.macro_milestone_digest);
    push_digest32(&mut out, input.meso_aggregation_digest);
    push_digest32(&mut out, input.macro_finalization_digest);
    push_u32_be(&mut out, input.meso_count);
    push_str32(&mut out, input.source);
    digest_sha256_domain(b"ucf.replay.minimal_spine.token_input.v1", &out)
}

fn validate_minimal_spine_replay_token_input(
    input: &MinimalSpineReplayTokenInput,
) -> Result<(), ReplayError> {
    if is_zero_digest(input.macro_candidate_digest) {
        return Err(ReplayError::InvalidMinimalSpineReplayTokenInput(
            "macro candidate digest must be non-zero",
        ));
    }
    if is_zero_digest(input.macro_milestone_digest) {
        return Err(ReplayError::InvalidMinimalSpineReplayTokenInput(
            "macro milestone digest must be non-zero",
        ));
    }
    if is_zero_digest(input.meso_aggregation_digest) {
        return Err(ReplayError::InvalidMinimalSpineReplayTokenInput(
            "meso aggregation digest must be non-zero",
        ));
    }
    if is_zero_digest(input.macro_finalization_digest) {
        return Err(ReplayError::InvalidMinimalSpineReplayTokenInput(
            "macro finalization digest must be non-zero",
        ));
    }
    if input.meso_count == 0 {
        return Err(ReplayError::InvalidMinimalSpineReplayTokenInput(
            "meso count must be non-zero",
        ));
    }
    if input.source.is_empty() {
        return Err(ReplayError::InvalidMinimalSpineReplayTokenInput(
            "source must be non-empty",
        ));
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

fn push_digest_vec(out: &mut Vec<u8>, digests: &[Digest32]) {
    push_u32_be(
        out,
        u32::try_from(digests.len()).expect("minimal spine replay digest vector length fits u32"),
    );
    for digest in digests {
        push_digest32(out, *digest);
    }
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
                    .expect("minimal spine replay optional bytes length fits u32"),
            );
            out.extend_from_slice(value);
        }
        None => out.push(0),
    }
}

fn push_str32(out: &mut Vec<u8>, value: &str) {
    let len = u32::try_from(value.len()).expect("minimal spine replay source length fits u32");
    push_u32_be(out, len);
    out.extend_from_slice(value.as_bytes());
}

fn digest_sha256_domain(domain: &[u8], bytes: &[u8]) -> Digest32 {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update([0]);
    hasher.update(bytes);
    Digest32::new(hasher.finalize().into())
}

pub fn replay_records(records: &[ExperienceRecord], spec: &ReplaySpec) -> ReplayResult {
    let mut items = Vec::new();
    let mut matched = 0usize;
    let mut drifted = 0usize;
    let mut unreplayable = 0usize;

    for rec in records {
        if rec.kind != ExperienceKind::DecisionOut {
            continue;
        }
        if rec.time.tick.get() < spec.from_tick || rec.time.tick.get() > spec.to_tick {
            continue;
        }

        let decision = match &rec.payload {
            ExperiencePayload::Decision(d) => d,
            _ => continue,
        };

        let persisted = match decision.compute_summary {
            Some(summary) => summary,
            None => {
                let item = ReplayItem {
                    decision_id: rec.id.0,
                    correlation_id: rec.corr.0,
                    persisted: empty_persisted("missing"),
                    recomputed: None,
                    diff: DiffSummary {
                        risk_abs: None,
                        confidence_abs: None,
                        surprise_abs: None,
                        pressure_abs: None,
                        pass: false,
                        reasons: vec![DriftReason::MissingPersistedField {
                            field: "compute_summary".to_string(),
                        }],
                    },
                    status: ReplayStatus::Unreplayable,
                };
                unreplayable += 1;
                items.push(item);
                continue;
            }
        };

        let control = records
            .iter()
            .rev()
            .find(|candidate| {
                candidate.kind == ExperienceKind::ControlIn
                    && candidate.corr == rec.corr
                    && candidate.time.tick.get() <= rec.time.tick.get()
            })
            .and_then(|candidate| match &candidate.payload {
                ExperiencePayload::Control(ctrl) => Some(ctrl.clone()),
                _ => None,
            });

        let Some(control) = control else {
            let item = ReplayItem {
                decision_id: rec.id.0,
                correlation_id: rec.corr.0,
                persisted: to_persisted(&persisted),
                recomputed: None,
                diff: DiffSummary {
                    risk_abs: None,
                    confidence_abs: None,
                    surprise_abs: None,
                    pressure_abs: None,
                    pass: false,
                    reasons: vec![DriftReason::MissingPersistedField {
                        field: "control_frame".to_string(),
                    }],
                },
                status: ReplayStatus::Unreplayable,
            };
            unreplayable += 1;
            items.push(item);
            continue;
        };

        let backend_kind = spec
            .backend_override
            .or_else(|| {
                ComputeBackendKind::parse(persisted.backend_profile.unwrap_or(persisted.backend))
            })
            .unwrap_or(ComputeBackendKind::Stub);
        let seed = spec
            .seed_override
            .or(persisted.seed)
            .unwrap_or(ComputeBudget::default().seed);
        let _budget_profile = spec
            .budget_override
            .or(persisted.budget_profile_id)
            .unwrap_or(stable_budget_profile_id(
                ComputeBudget::default().max_micros,
                ComputeBudget::default().hard_timeout_micros,
            ));

        let cfg = ComputeBackendConfig {
            kind: backend_kind,
            seed,
            ..ComputeBackendConfig::default()
        };
        let backend = match build_backend(&cfg) {
            Ok(backend) => backend,
            Err(_) => {
                let item = ReplayItem {
                    decision_id: rec.id.0,
                    correlation_id: rec.corr.0,
                    persisted: to_persisted(&persisted),
                    recomputed: None,
                    diff: DiffSummary {
                        risk_abs: None,
                        confidence_abs: None,
                        surprise_abs: None,
                        pressure_abs: None,
                        pass: false,
                        reasons: vec![DriftReason::BackendUnavailable {
                            backend_profile: format!("{:?}", backend_kind),
                        }],
                    },
                    status: ReplayStatus::Unreplayable,
                };
                unreplayable += 1;
                items.push(item);
                continue;
            }
        };

        let recomputed = match backend.compute(
            &compute_input_from_control(&control),
            ComputeBudget {
                seed,
                ..ComputeBudget::default()
            },
        ) {
            Ok(signals) => signals.summary(backend.name()),
            Err(_) => {
                unreplayable += 1;
                items.push(ReplayItem {
                    decision_id: rec.id.0,
                    correlation_id: rec.corr.0,
                    persisted: to_persisted(&persisted),
                    recomputed: None,
                    diff: DiffSummary {
                        risk_abs: None,
                        confidence_abs: None,
                        surprise_abs: None,
                        pressure_abs: None,
                        pass: false,
                        reasons: vec![DriftReason::BackendUnavailable {
                            backend_profile: backend.name().to_string(),
                        }],
                    },
                    status: ReplayStatus::Unreplayable,
                });
                continue;
            }
        };

        let policy = DiffPolicy::for_backend(recomputed.backend);
        let mut reasons = compare_summaries(&persisted, &recomputed, &policy);
        if matches!(
            spec.mode,
            ReplayMode::DecisionScoring | ReplayMode::FullNoAction
        ) {
            reasons.push(DriftReason::DecisionScoringUnavailable);
        }

        let diff = DiffSummary {
            risk_abs: Some((persisted.risk - recomputed.risk).abs()),
            confidence_abs: Some((persisted.confidence - recomputed.confidence).abs()),
            surprise_abs: Some((persisted.surprise - recomputed.surprise).abs()),
            pressure_abs: Some((persisted.pressure - recomputed.pressure).abs()),
            pass: reasons.is_empty(),
            reasons,
        };

        let status = if diff.pass {
            matched += 1;
            ReplayStatus::Match
        } else {
            drifted += 1;
            ReplayStatus::Drift
        };

        items.push(ReplayItem {
            decision_id: rec.id.0,
            correlation_id: rec.corr.0,
            persisted: to_persisted(&persisted),
            recomputed: Some(to_recomputed(&recomputed)),
            diff,
            status,
        });
    }

    let total_items = items.len();
    let truncated = items.len() > REPORT_CAP;
    items.truncate(REPORT_CAP);

    ReplayResult {
        total_items,
        matched,
        drifted,
        unreplayable,
        items,
        truncated,
    }
}

fn compare_summaries(
    persisted: &ComputeSignalsSummary,
    recomputed: &RecomputedComputeSummary,
    policy: &DiffPolicy,
) -> Vec<DriftReason> {
    if persisted.compute_chain_digest == Some(recomputed.compute_chain_digest) {
        return Vec::new();
    }

    let mut reasons = Vec::new();

    if let Some(expected) = persisted.compute_chain_digest {
        if expected != recomputed.compute_chain_digest {
            UCF_COMPUTE_CHAIN_MISMATCH_TOTAL.fetch_add(1, Ordering::Relaxed);
            reasons.push(DriftReason::DigestMismatch {
                field: "compute_chain_digest".to_string(),
                expected_prefix: opt_digest_prefix(Some(expected)),
                got_prefix: opt_digest_prefix(Some(recomputed.compute_chain_digest)),
            });
        }
    }

    compare_float(
        "risk",
        persisted.risk,
        recomputed.risk,
        policy.eps,
        &mut reasons,
    );
    compare_float(
        "confidence",
        persisted.confidence,
        recomputed.confidence,
        policy.eps,
        &mut reasons,
    );
    compare_float(
        "surprise",
        persisted.surprise,
        recomputed.surprise,
        policy.eps,
        &mut reasons,
    );
    compare_float(
        "pressure",
        persisted.pressure,
        recomputed.pressure,
        policy.eps,
        &mut reasons,
    );

    if persisted.spikes_digest != recomputed.spikes_digest {
        reasons.push(DriftReason::DigestMismatch {
            field: "spikes_digest".to_string(),
            expected_prefix: hex::encode(&persisted.spikes_digest[..6]),
            got_prefix: hex::encode(&recomputed.spikes_digest[..6]),
        });
    }
    if persisted.evidence_context_digest != Some(recomputed.evidence_context_digest) {
        reasons.push(DriftReason::DigestMismatch {
            field: "evidence_context_digest".to_string(),
            expected_prefix: opt_digest_prefix(persisted.evidence_context_digest),
            got_prefix: opt_digest_prefix(Some(recomputed.evidence_context_digest)),
        });
    }

    reasons
}

fn compare_float(field: &str, expected: f32, got: f32, eps: f32, reasons: &mut Vec<DriftReason>) {
    let abs_diff = (expected - got).abs();
    if abs_diff > eps {
        reasons.push(DriftReason::FloatMismatch {
            field: field.to_string(),
            expected,
            got,
            abs_diff,
        });
    }
}

fn to_persisted(summary: &ComputeSignalsSummary) -> PersistedSummary {
    PersistedSummary {
        backend: summary.backend.to_string(),
        risk: summary.risk,
        confidence: summary.confidence,
        surprise: summary.surprise,
        pressure: summary.pressure,
        risk_quality: summary.risk_quality,
        spikes_digest_hex: hex::encode(summary.spikes_digest),
        context_digest_hex: summary.evidence_context_digest.map(hex::encode),
        chain_digest_hex: summary.compute_chain_digest.map(hex::encode),
    }
}

fn to_recomputed(summary: &RecomputedComputeSummary) -> RecomputedSummary {
    RecomputedSummary {
        backend: summary.backend.to_string(),
        risk: summary.risk,
        confidence: summary.confidence,
        surprise: summary.surprise,
        pressure: summary.pressure,
        risk_quality: Some(summary.risk_quality),
        spikes_digest_hex: hex::encode(summary.spikes_digest),
        context_digest_hex: Some(hex::encode(summary.evidence_context_digest)),
        chain_digest_hex: Some(hex::encode(summary.compute_chain_digest)),
    }
}

fn empty_persisted(backend: &str) -> PersistedSummary {
    PersistedSummary {
        backend: backend.to_string(),
        risk: 0.0,
        confidence: 0.0,
        surprise: 0.0,
        pressure: 0.0,
        risk_quality: None,
        spikes_digest_hex: String::new(),
        context_digest_hex: None,
        chain_digest_hex: None,
    }
}

fn opt_digest_prefix(value: Option<[u8; 32]>) -> String {
    value
        .map(|digest| hex::encode(&digest[..6]))
        .unwrap_or_else(|| "none".to_string())
}

#[derive(Debug, Deserialize)]
pub struct Fixture {
    pub decisions: Vec<FixtureDecision>,
}

#[derive(Debug, Deserialize)]
pub struct FixtureDecision {
    pub decision_id: u64,
    pub corr: u64,
    pub tick: u64,
    pub window: u64,
    pub text: String,
    pub backend: String,
    pub risk: f32,
    pub confidence: f32,
    pub surprise: f32,
    pub pressure: f32,
    pub spike_count: u16,
    pub spikes_digest_hex: String,
    pub evidence_context_digest_hex: String,
    pub budget_profile_id: u32,
    pub seed: u64,
    pub risk_quality: u8,
}

pub fn load_fixture_records(path: &Path) -> Result<Vec<ExperienceRecord>, ReplayError> {
    let data = fs::read_to_string(path)?;
    let fixture: Fixture = serde_json::from_str(&data)?;
    let mut out = Vec::new();

    for entry in fixture.decisions {
        let time = SimTime {
            tick: Tick::new(entry.tick),
            window: WindowId::new(entry.window),
        };
        let ctrl = ControlFrame::new_text(
            time,
            CorrelationId(entry.corr),
            ChannelCode::ExternalOutput,
            Intent::new(IntentId(entry.corr), IntentKind::Speak, "fixture"),
            entry.text,
        );
        out.push(ExperienceRecord::from_control(
            ucf_ess::v1::ExperienceId(entry.decision_id * 10),
            ctrl,
        ));

        let spikes_digest = <[u8; 32]>::from_hex(entry.spikes_digest_hex).unwrap_or([0; 32]);
        let context_digest =
            <[u8; 32]>::from_hex(entry.evidence_context_digest_hex).unwrap_or([0; 32]);
        let backend_name = entry.backend.clone();
        let summary = ComputeSignalsSummary {
            backend: leak_str(backend_name.clone()),
            surprise: entry.surprise,
            pressure: entry.pressure,
            risk: entry.risk,
            confidence: entry.confidence,
            surprise_q: quantize_unit_u16(entry.surprise),
            pressure_q: quantize_unit_u16(entry.pressure),
            risk_q: quantize_unit_u16(entry.risk),
            confidence_q: quantize_unit_u16(entry.confidence),
            spike_count: entry.spike_count,
            spikes_digest,
            sparsity: None,
            energy: None,
            ssm_readout: None,
            ssm_digest: None,
            world_digest: None,
            risk_quality: Some(entry.risk_quality),
            evidence_context_digest: Some(context_digest),
            evidence_world_digest: None,
            evidence_spikes_digest: None,
            evidence_ssm_digest: None,
            evidence_lfm_digest: None,
            backend_profile: Some(leak_str(backend_name)),
            backend_pack_id: None,
            fixtures_digest: None,
            llm_backend: None,
            world_backend: None,
            sae_backend: None,
            ssm_backend: None,
            lfm_backend: None,
            lfm_uncertainty: None,
            lfm_stability: None,
            lfm_uncertainty_q: None,
            lfm_stability_q: None,
            lfm_state_norm: None,
            lfm_deriv_norm: None,
            lfm_saturation_ratio: None,
            lfm_nan_inf_detected: None,
            lfm_digest: None,
            signal_bundle_digest: None,
            budget_profile_id: Some(entry.budget_profile_id),
            seed: Some(entry.seed),
            risk_contract_version: Some(1),
            compute_schema_version: Some(1),
            compute_chain_digest: None,
            compute_code_version: None,
            budget_exceeded_stage: None,
            contract_version: Some(1),
            backend_id: Some(0),
            validation_status: Some(0),
            violation_reason_mask: Some(0),
            lfm_quality: None,
            coherence: None,
            instability: None,
            coherence_q: None,
            instability_q: None,
            phi_proxy: None,
            coherence_digest: None,
            iit_coherence_q: None,
            iit_incoherence_q: None,
            iit_reason_codes: None,
            stage_allow_mask: None,
            free_energy_proxy_q: None,
            ebm_energy_mean_topk_q: None,
            ebm_w_q: None,
            fep_coupling_version: None,
        };

        let decision = DecisionFrame::allow(time, CorrelationId(entry.corr), "fixture")
            .with_compute_summary(summary);
        out.push(ExperienceRecord::from_decision(
            ucf_ess::v1::ExperienceId(entry.decision_id),
            decision,
        ));
    }

    Ok(out)
}

fn leak_str(value: String) -> &'static str {
    Box::leak(value.into_boxed_str())
}

pub fn write_report(path: &Path, result: &ReplayResult) -> Result<(), ReplayError> {
    let body = serde_json::to_string_pretty(result)?;
    fs::write(path, body)?;
    Ok(())
}

pub fn ucf_compute_chain_mismatch_total() -> u64 {
    UCF_COMPUTE_CHAIN_MISMATCH_TOTAL.load(Ordering::Relaxed)
}

pub fn replay_audit(records: &[ExperienceRecord], plan: &ReplayPlan) -> ReplayReport {
    let mut report = ReplayReport {
        range: (plan.t0, plan.t1),
        overall_status: ReplayOverallStatus::Ok,
        first_divergence: None,
        counters: ReplayCounters {
            missing_records: 0,
            mismatched_digests: 0,
            degraded_steps: 0,
        },
        details: Vec::new(),
    };

    let in_range: Vec<&ExperienceRecord> = records
        .iter()
        .filter(|r| {
            let t = r.time.tick.get();
            t >= plan.t0 && t <= plan.t1
        })
        .collect();

    if in_range.is_empty() {
        report.overall_status = ReplayOverallStatus::MissingData;
        report.counters.missing_records += 1;
        return report;
    }

    let mut decision_chain_by_corr = HashMap::new();
    let mut decision_by_id = HashMap::new();

    for record in &in_range {
        if let ExperiencePayload::Decision(decision) = &record.payload {
            decision_by_id.insert(record.id.0, (**decision).clone());
            if let Some(summary) = decision.compute_summary {
                if let Some(chain) = summary.compute_chain_digest {
                    decision_chain_by_corr.insert(record.corr.0, chain);
                } else {
                    push_divergence(
                        &mut report,
                        Divergence {
                            t: record.time.tick.get(),
                            component: ReplayComponent::Risk,
                            expected_digest: "present".to_string(),
                            observed_digest: "missing".to_string(),
                            hint: "compute_chain_digest missing in DecisionFrame.compute_summary"
                                .to_string(),
                        },
                    );
                }
                verify_summary_links(
                    &mut report,
                    record.time.tick.get(),
                    summary,
                    plan.strictness,
                );
            } else {
                report.counters.missing_records += 1;
            }
        }
    }

    verify_backend_pack(records, plan, &mut report);

    for record in &in_range {
        let t = record.time.tick.get();
        match (&record.kind, &record.payload) {
            (ExperienceKind::Nsr, _) => {
                if let Some(nsr) = &record.nsr_record {
                    verify_chain_ref(
                        &mut report,
                        t,
                        ReplayComponent::Nsr,
                        nsr.evidence_chain_digest,
                        decision_chain_by_corr.get(&record.corr.0).copied(),
                    );
                }
            }
            (ExperienceKind::LfmSummary, _) => {
                if let Some(summary) = record.lfm_summary_record {
                    verify_lfm_summary_digest(&mut report, t, summary);
                    verify_chain_ref(
                        &mut report,
                        t,
                        ReplayComponent::Lfm,
                        summary.evidence_chain_digest,
                        decision_chain_by_corr.get(&record.corr.0).copied(),
                    );
                }
            }
            (
                ExperienceKind::CapabilityIssuance,
                ExperiencePayload::Audit(AuditPayload::CapabilityIssuance(issuance)),
            ) => {
                verify_chain_ref(
                    &mut report,
                    t,
                    ReplayComponent::Issuance,
                    issuance.evidence_chain_digest,
                    decision_chain_by_corr.get(&record.corr.0).copied(),
                );
                verify_issuance_decision(
                    &mut report,
                    t,
                    issuance,
                    &decision_by_id,
                    records,
                    plan.strictness,
                );
            }
            (ExperienceKind::Output, ExperiencePayload::Audit(AuditPayload::Output(out))) => {
                verify_chain_ref(
                    &mut report,
                    t,
                    ReplayComponent::Output,
                    out.evidence_chain_digest,
                    decision_chain_by_corr.get(&record.corr.0).copied(),
                );
                if out.llm_request_digest == [0; 32] || out.llm_response_digest == [0; 32] {
                    push_divergence(
                        &mut report,
                        Divergence {
                            t,
                            component: ReplayComponent::Output,
                            expected_digest: "non_zero".to_string(),
                            observed_digest: "zero".to_string(),
                            hint: "llm request/response digest missing or zero".to_string(),
                        },
                    );
                }
            }
            _ => {}
        }

        if plan.stop_on_first_divergence && report.first_divergence.is_some() {
            finalize_status(&mut report);
            return report;
        }
    }

    if matches!(plan.strictness, ReplayStrictness::RecomputeStages) {
        recompute_decision_chain(&mut report, &in_range);
    }

    finalize_status(&mut report);
    report
}

fn verify_summary_links(
    report: &mut ReplayReport,
    t: u64,
    summary: ComputeSignalsSummary,
    strictness: ReplayStrictness,
) {
    let checks = [
        (
            summary.world_digest,
            summary.evidence_world_digest,
            ReplayComponent::World,
        ),
        (
            Some(summary.spikes_digest),
            summary.evidence_spikes_digest,
            ReplayComponent::Sae,
        ),
        (
            summary.ssm_digest,
            summary.evidence_ssm_digest,
            ReplayComponent::Ssm,
        ),
        (
            summary.lfm_digest,
            summary.evidence_lfm_digest,
            ReplayComponent::Lfm,
        ),
    ];
    for (raw, evidence, component) in checks {
        if raw.is_some() && evidence.is_none() {
            push_divergence(
                report,
                Divergence {
                    t,
                    component,
                    expected_digest: "present".to_string(),
                    observed_digest: "missing".to_string(),
                    hint: "evidence-chain link missing".to_string(),
                },
            );
        }
    }

    if matches!(strictness, ReplayStrictness::RecomputeStages)
        && summary.coherence.is_some()
        && summary.coherence_digest.is_none()
    {
        push_divergence(
            report,
            Divergence {
                t,
                component: ReplayComponent::Coherence,
                expected_digest: "present".to_string(),
                observed_digest: "missing".to_string(),
                hint: "coherence active but coherence_digest missing".to_string(),
            },
        );
    }
}

fn verify_backend_pack(records: &[ExperienceRecord], plan: &ReplayPlan, report: &mut ReplayReport) {
    let mut stable: Option<[u8; 32]> = None;
    for record in records.iter().filter(|r| {
        r.kind == ExperienceKind::BackendPack
            && r.time.tick.get() >= plan.t0
            && r.time.tick.get() <= plan.t1
    }) {
        let Some(pack) = &record.backend_pack_record else {
            report.counters.missing_records += 1;
            continue;
        };
        if let Some(expected) = plan.expected_backend_pack_digest {
            if pack.meta_digest != expected {
                push_divergence(
                    report,
                    Divergence {
                        t: record.time.tick.get(),
                        component: ReplayComponent::BackendPack,
                        expected_digest: digest_prefix(expected),
                        observed_digest: digest_prefix(pack.meta_digest),
                        hint: "expected_backend_pack_digest mismatch".to_string(),
                    },
                );
            }
        }
        if let Some(first) = stable {
            if first != pack.meta_digest {
                push_divergence(
                    report,
                    Divergence {
                        t: record.time.tick.get(),
                        component: ReplayComponent::BackendPack,
                        expected_digest: digest_prefix(first),
                        observed_digest: digest_prefix(pack.meta_digest),
                        hint: "backend pack drift inside replay range".to_string(),
                    },
                );
            }
        } else {
            stable = Some(pack.meta_digest);
        }
    }
}

fn verify_chain_ref(
    report: &mut ReplayReport,
    t: u64,
    component: ReplayComponent,
    observed: [u8; 32],
    expected: Option<[u8; 32]>,
) {
    let Some(expected) = expected else {
        report.counters.missing_records += 1;
        push_divergence(
            report,
            Divergence {
                t,
                component,
                expected_digest: "decision_chain".to_string(),
                observed_digest: digest_prefix(observed),
                hint: "associated DecisionFrame/compute_chain_digest missing".to_string(),
            },
        );
        return;
    };
    if observed != expected {
        push_divergence(
            report,
            Divergence {
                t,
                component,
                expected_digest: digest_prefix(expected),
                observed_digest: digest_prefix(observed),
                hint: "evidence_chain_digest mismatch".to_string(),
            },
        );
    }
}

fn verify_lfm_summary_digest(report: &mut ReplayReport, t: u64, summary: LfmSummaryRecord) {
    let expected = summary.compute_digest();
    if summary.digest != expected {
        push_divergence(
            report,
            Divergence {
                t,
                component: ReplayComponent::Lfm,
                expected_digest: digest_prefix(expected),
                observed_digest: digest_prefix(summary.digest),
                hint: "LfmSummaryRecord.digest invalid".to_string(),
            },
        );
    }
}

fn verify_issuance_decision(
    report: &mut ReplayReport,
    t: u64,
    issuance: &CapabilityIssuanceRecord,
    decision_by_id: &HashMap<u64, DecisionFrame>,
    records: &[ExperienceRecord],
    strictness: ReplayStrictness,
) {
    let decision = decision_by_id.get(&issuance.decision_id);
    let nsr_risk = records
        .iter()
        .filter(|r| r.kind == ExperienceKind::Nsr && r.time.tick.get() <= t)
        .find_map(|r| r.nsr_record.as_ref())
        .map(|n| f32::from(n.nsr_risk_q) / 65535.0);
    let hormone_stress = records
        .iter()
        .filter(|r| r.kind == ExperienceKind::Hormone && r.time.tick.get() <= t)
        .find_map(|r| r.hormone_record)
        .map(|h| f32::from(h.stress_index_q) / 65535.0);

    let expected_signals_digest = governance_signals_digest(decision, t, nsr_risk, hormone_stress);
    if issuance.governance_signals_digest != expected_signals_digest {
        push_divergence(
            report,
            Divergence {
                t,
                component: ReplayComponent::Governor,
                expected_digest: digest_prefix(expected_signals_digest),
                observed_digest: digest_prefix(issuance.governance_signals_digest),
                hint: "governance_signals_digest mismatch".to_string(),
            },
        );
    }

    if matches!(strictness, ReplayStrictness::RecomputeStages) {
        let Some(summary) = decision.and_then(|d| d.compute_summary) else {
            report.counters.missing_records += 1;
            return;
        };
        let score = governor_score(
            nsr_risk.unwrap_or(summary.risk),
            summary.coherence,
            summary.instability,
            summary.lfm_uncertainty,
            hormone_stress,
        );
        let tier = issuance_tier(score);
        if issuance.tier != tier {
            push_divergence(
                report,
                Divergence {
                    t,
                    component: ReplayComponent::Issuance,
                    expected_digest: format!("tier:{tier}"),
                    observed_digest: format!("tier:{}", issuance.tier),
                    hint: "tier does not match recomputed governor score".to_string(),
                },
            );
        }

        let q = quantize_unit_u16(score);
        if issuance.governor_score_q != q {
            push_divergence(
                report,
                Divergence {
                    t,
                    component: ReplayComponent::Governor,
                    expected_digest: format!("score_q:{q}"),
                    observed_digest: format!("score_q:{}", issuance.governor_score_q),
                    hint: "governor_score_q mismatch".to_string(),
                },
            );
        }
    }
}

fn recompute_decision_chain(report: &mut ReplayReport, in_range: &[&ExperienceRecord]) {
    for rec in in_range {
        if rec.kind != ExperienceKind::DecisionOut {
            continue;
        }
        let decision = match &rec.payload {
            ExperiencePayload::Decision(d) => d,
            _ => continue,
        };
        let Some(persisted) = decision.compute_summary else {
            continue;
        };
        let control = in_range
            .iter()
            .rev()
            .find(|candidate| {
                candidate.kind == ExperienceKind::ControlIn
                    && candidate.corr == rec.corr
                    && candidate.time.tick.get() <= rec.time.tick.get()
            })
            .and_then(|candidate| match &candidate.payload {
                ExperiencePayload::Control(ctrl) => Some(ctrl.clone()),
                _ => None,
            });
        let Some(control) = control else {
            continue;
        };
        let backend_kind =
            ComputeBackendKind::parse(persisted.backend_profile.unwrap_or(persisted.backend))
                .unwrap_or(ComputeBackendKind::Stub);
        let seed = persisted.seed.unwrap_or(ComputeBudget::default().seed);
        let cfg = ComputeBackendConfig {
            kind: backend_kind,
            seed,
            ..ComputeBackendConfig::default()
        };
        let Ok(backend) = build_backend(&cfg) else {
            continue;
        };
        let Ok(recomputed) = backend.compute(
            &compute_input_from_control(&control),
            ComputeBudget {
                seed,
                ..ComputeBudget::default()
            },
        ) else {
            continue;
        };
        let recomputed = recomputed.summary(backend.name());
        if let Some(persisted_chain) = persisted.compute_chain_digest {
            if persisted_chain != recomputed.compute_chain_digest {
                push_divergence(
                    report,
                    Divergence {
                        t: rec.time.tick.get(),
                        component: ReplayComponent::Risk,
                        expected_digest: digest_prefix(recomputed.compute_chain_digest),
                        observed_digest: digest_prefix(persisted_chain),
                        hint: "recomputed compute_chain_digest mismatch".to_string(),
                    },
                );
            }
        }
    }
}

fn governance_signals_digest(
    decision: Option<&DecisionFrame>,
    t: u64,
    nsr_risk: Option<f32>,
    hormone_stress: Option<f32>,
) -> [u8; 32] {
    let summary = decision.and_then(|d| d.compute_summary);
    let risk = summary.map(|s| s.risk).unwrap_or(1.0).clamp(0.0, 1.0);
    let confidence = summary.map(|s| s.confidence).unwrap_or(0.0).clamp(0.0, 1.0);
    let coherence = summary.and_then(|s| s.coherence).map(|v| v.clamp(0.0, 1.0));
    let instability = summary
        .and_then(|s| s.instability)
        .map(|v| v.clamp(0.0, 1.0));
    let pressure = summary.map(|s| s.pressure).unwrap_or(1.0).clamp(0.0, 1.0);
    let surprise = summary.map(|s| s.surprise).unwrap_or(1.0).clamp(0.0, 1.0);
    let lfm_uncertainty = summary
        .and_then(|s| s.lfm_uncertainty)
        .map(|v| v.clamp(0.0, 1.0));
    let lfm_stability = summary
        .and_then(|s| s.lfm_stability)
        .map(|v| v.clamp(0.0, 1.0));

    let mut hasher = Sha256::new();
    hasher.update(t.to_le_bytes());
    hasher.update(quantize_unit_u16(risk).to_le_bytes());
    hasher.update(quantize_unit_u16(confidence).to_le_bytes());
    put_opt_f32(&mut hasher, nsr_risk.map(|v| v.clamp(0.0, 1.0)));
    put_opt_f32(&mut hasher, coherence);
    put_opt_f32(&mut hasher, instability);
    hasher.update(quantize_unit_u16(pressure).to_le_bytes());
    hasher.update(quantize_unit_u16(surprise).to_le_bytes());
    put_opt_f32(&mut hasher, lfm_uncertainty);
    put_opt_f32(&mut hasher, lfm_stability);
    put_opt_f32(&mut hasher, hormone_stress.map(|v| v.clamp(0.0, 1.0)));
    hasher.finalize().into()
}

fn put_opt_f32(hasher: &mut Sha256, value: Option<f32>) {
    if let Some(v) = value {
        hasher.update([1]);
        hasher.update(quantize_unit_u16(v).to_le_bytes());
    } else {
        hasher.update([0]);
    }
}

fn governor_score(
    risk: f32,
    coherence: Option<f32>,
    instability: Option<f32>,
    lfm_uncertainty: Option<f32>,
    hormone_stress: Option<f32>,
) -> f32 {
    (0.35 * risk
        + 0.20 * (1.0 - coherence.unwrap_or(1.0))
        + 0.20 * instability.unwrap_or(0.0)
        + 0.15 * lfm_uncertainty.unwrap_or(0.0)
        + 0.10 * hormone_stress.unwrap_or(0.0))
    .clamp(0.0, 1.0)
}

fn issuance_tier(score: f32) -> u8 {
    if score < 0.25 {
        0
    } else if score < 0.5 {
        1
    } else if score < 0.75 {
        2
    } else {
        3
    }
}

fn quantize_unit_u16(value: f32) -> u16 {
    quantize_unit(value, CANONICAL_UNIT_QUANT_MAX)
}

fn push_divergence(report: &mut ReplayReport, divergence: Divergence) {
    if report.first_divergence.is_none() {
        report.first_divergence = Some(divergence.clone());
    }
    if report.details.len() < REPLAY_DIVERGENCE_CAP {
        report.details.push(divergence);
    }
    report.counters.mismatched_digests = report.counters.mismatched_digests.saturating_add(1);
}

fn finalize_status(report: &mut ReplayReport) {
    if report.counters.missing_records > 0 {
        report.overall_status = ReplayOverallStatus::MissingData;
    }
    if report.first_divergence.is_some() {
        report.overall_status = ReplayOverallStatus::DriftFound;
    }
}

fn digest_prefix(digest: [u8; 32]) -> String {
    let mut out = String::new();
    for byte in &digest[..6] {
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}
