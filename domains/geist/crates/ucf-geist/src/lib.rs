#![forbid(unsafe_code)]

use std::sync::Arc;

use blake3::Hasher;
use ucf_archive::ExperienceAppender;
use ucf_commit::commit_milestone_macro;
use ucf_policy_ecology::{ConsistencyReport, ConsistencyVerdict, DefaultPolicyEcology, GeistGate};
use ucf_sleep_coordinator::{
    MinimalSpineSleepAppliedBoundary, MinimalSpineSleepPlanAudit, MinimalSpineSleepPlanAuditStatus,
    SleepStateHandle, SleepStateUpdater,
};
use ucf_types::consolidation::ReplayApplied;
use ucf_types::v1::spec::{ExperienceRecord, MacroMilestone};
use ucf_types::{Digest32, EvidenceId};

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
