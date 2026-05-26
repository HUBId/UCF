use blake3::Hasher;
use ucf_types::Digest32;

use crate::hormone_state_v1::{HormoneStateError, HormoneStateV1, NormalizedHormoneLevelV1};
use crate::hormone_update_v1::HormoneModulationOutputV1;
use crate::replay_sleep_candidate_v1::MetabolicReplaySleepCandidatesV1;

const METABOLIC_AUDIT_DOMAIN_V1: &[u8] = b"ucf.neuromod.metabolic.audit.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetabolicAuditStatusV1 {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MetabolicAuditFailureV1 {
    InvalidHormoneState,
    ModulationOutOfBounds,
    ReplayCandidateOutOfBounds,
    SleepCandidateOutOfBounds,
    RuntimeAuthorityPresent,
    SchedulerAuthorityPresent,
    GatewayAuthorityPresent,
    PolicyMutationPresent,
    EvidenceArchiveAuthorityPresent,
    IdentityAuthorityPresent,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetabolicVerifyAuditV1 {
    pub status: MetabolicAuditStatusV1,
    pub failures: Vec<MetabolicAuditFailureV1>,
    pub state_digest: Digest32,
    pub modulation_digest: Digest32,
    pub candidate_digest: Digest32,
    pub audit_digest: Digest32,
    pub advisory_only: bool,
    pub runtime_authority: bool,
    pub scheduler_authority: bool,
    pub gateway_visible: bool,
    pub policy_mutation: bool,
    pub evidence_archive_authority: bool,
    pub identity_authority: bool,
}

impl MetabolicVerifyAuditV1 {
    pub fn deterministic_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_u8(&mut out, status_code(self.status));
        push_u16(&mut out, self.failures.len() as u16);
        for failure in &self.failures {
            push_u8(&mut out, failure_code(*failure));
        }
        push_digest(&mut out, self.state_digest);
        push_digest(&mut out, self.modulation_digest);
        push_digest(&mut out, self.candidate_digest);
        push_bool(&mut out, self.advisory_only);
        push_bool(&mut out, self.runtime_authority);
        push_bool(&mut out, self.scheduler_authority);
        push_bool(&mut out, self.gateway_visible);
        push_bool(&mut out, self.policy_mutation);
        push_bool(&mut out, self.evidence_archive_authority);
        push_bool(&mut out, self.identity_authority);
        out
    }

    pub fn digest(&self) -> Digest32 {
        let mut hasher = Hasher::new();
        hasher.update(METABOLIC_AUDIT_DOMAIN_V1);
        hasher.update(&self.deterministic_bytes());
        Digest32::new(*hasher.finalize().as_bytes())
    }

    pub const fn metadata_only(&self) -> bool {
        true
    }

    pub const fn is_pass(&self) -> bool {
        matches!(self.status, MetabolicAuditStatusV1::Pass)
    }
}

pub fn verify_metabolic_candidates_v1(
    state: &HormoneStateV1,
    modulation: &HormoneModulationOutputV1,
    candidates: &MetabolicReplaySleepCandidatesV1,
) -> MetabolicVerifyAuditV1 {
    let mut failures = Vec::new();

    if matches!(state.validate(), Err(HormoneStateError::OutOfRange { .. })) {
        failures.push(MetabolicAuditFailureV1::InvalidHormoneState);
    }
    if !modulation_is_bounded(modulation) {
        failures.push(MetabolicAuditFailureV1::ModulationOutOfBounds);
    }
    if !replay_candidate_is_bounded(candidates) {
        failures.push(MetabolicAuditFailureV1::ReplayCandidateOutOfBounds);
    }
    if !sleep_candidate_is_bounded(candidates) {
        failures.push(MetabolicAuditFailureV1::SleepCandidateOutOfBounds);
    }

    let advisory_only = true;
    let runtime_authority = HormoneModulationOutputV1::runtime_authority();
    let scheduler_authority = false;
    let gateway_visible = HormoneStateV1::gateway_visible();
    let policy_mutation =
        HormoneStateV1::policy_mutating() || HormoneModulationOutputV1::policy_mutation();
    let evidence_archive_authority = HormoneStateV1::evidence_archive_authority();
    let identity_authority = HormoneStateV1::identity_authority();

    if runtime_authority {
        failures.push(MetabolicAuditFailureV1::RuntimeAuthorityPresent);
    }
    if scheduler_authority {
        failures.push(MetabolicAuditFailureV1::SchedulerAuthorityPresent);
    }
    if gateway_visible {
        failures.push(MetabolicAuditFailureV1::GatewayAuthorityPresent);
    }
    if policy_mutation {
        failures.push(MetabolicAuditFailureV1::PolicyMutationPresent);
    }
    if evidence_archive_authority {
        failures.push(MetabolicAuditFailureV1::EvidenceArchiveAuthorityPresent);
    }
    if identity_authority {
        failures.push(MetabolicAuditFailureV1::IdentityAuthorityPresent);
    }

    failures.sort_unstable();
    failures.dedup();

    let status = if failures.is_empty() {
        MetabolicAuditStatusV1::Pass
    } else {
        MetabolicAuditStatusV1::Fail
    };

    let state_digest = digest_state(state);
    let modulation_digest = digest_modulation(modulation);
    let candidate_digest = digest_candidates(candidates);

    let mut audit = MetabolicVerifyAuditV1 {
        status,
        failures,
        state_digest,
        modulation_digest,
        candidate_digest,
        audit_digest: Digest32::new([0; Digest32::LEN]),
        advisory_only,
        runtime_authority,
        scheduler_authority,
        gateway_visible,
        policy_mutation,
        evidence_archive_authority,
        identity_authority,
    };
    audit.audit_digest = audit.digest();
    audit
}

fn modulation_is_bounded(modulation: &HormoneModulationOutputV1) -> bool {
    let min = i64::from(NormalizedHormoneLevelV1::MIN);
    let max = i64::from(NormalizedHormoneLevelV1::MAX);
    let delta_limit = max - min;

    (min..=max).contains(&modulation.attention_gain)
        && (min..=max).contains(&modulation.learning_rate_multiplier)
        && (min..=max).contains(&modulation.replay_priority_multiplier)
        && (min..=max).contains(&modulation.noise_scale)
        && (min..=max).contains(&modulation.consolidation_gate)
        && (-delta_limit..=delta_limit).contains(&modulation.sleep_pressure_delta)
        && (min..=max).contains(&modulation.risk_damping)
}

fn replay_candidate_is_bounded(candidates: &MetabolicReplaySleepCandidatesV1) -> bool {
    let min = i64::from(NormalizedHormoneLevelV1::MIN);
    let max = i64::from(NormalizedHormoneLevelV1::MAX);
    let replay = &candidates.replay;
    (min..=max).contains(&replay.priority_hint)
        && (min..=max).contains(&replay.novelty_component)
        && (min..=max).contains(&replay.stability_component)
        && (min..=max).contains(&replay.arousal_component)
        && (min..=max).contains(&replay.risk_damping_component)
}

fn sleep_candidate_is_bounded(candidates: &MetabolicReplaySleepCandidatesV1) -> bool {
    let min = i64::from(NormalizedHormoneLevelV1::MIN);
    let max = i64::from(NormalizedHormoneLevelV1::MAX);
    let sleep = &candidates.sleep;
    (min..=max).contains(&sleep.pressure_hint)
        && (min..=max).contains(&sleep.sleep_delta_component)
        && (min..=max).contains(&sleep.risk_damping_component)
}

fn digest_state(state: &HormoneStateV1) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.neuromod.metabolic.audit.state.v1");
    hasher.update(&state_bytes(state));
    Digest32::new(*hasher.finalize().as_bytes())
}
fn digest_modulation(modulation: &HormoneModulationOutputV1) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.neuromod.metabolic.audit.modulation.v1");
    hasher.update(&modulation_bytes(modulation));
    Digest32::new(*hasher.finalize().as_bytes())
}
fn digest_candidates(candidates: &MetabolicReplaySleepCandidatesV1) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.neuromod.metabolic.audit.candidates.v1");
    hasher.update(&candidate_bytes(candidates));
    Digest32::new(*hasher.finalize().as_bytes())
}
fn push_bool(out: &mut Vec<u8>, v: bool) {
    push_u8(out, if v { 1 } else { 0 });
}
fn push_u8(out: &mut Vec<u8>, v: u8) {
    out.push(v);
}
fn push_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_be_bytes());
}
fn push_i64(out: &mut Vec<u8>, v: i64) {
    out.extend_from_slice(&v.to_be_bytes());
}
fn push_digest(out: &mut Vec<u8>, digest: Digest32) {
    out.extend_from_slice(digest.as_bytes());
}
fn status_code(status: MetabolicAuditStatusV1) -> u8 {
    match status {
        MetabolicAuditStatusV1::Pass => 1,
        MetabolicAuditStatusV1::Fail => 2,
    }
}
fn failure_code(failure: MetabolicAuditFailureV1) -> u8 {
    match failure {
        MetabolicAuditFailureV1::InvalidHormoneState => 1,
        MetabolicAuditFailureV1::ModulationOutOfBounds => 2,
        MetabolicAuditFailureV1::ReplayCandidateOutOfBounds => 3,
        MetabolicAuditFailureV1::SleepCandidateOutOfBounds => 4,
        MetabolicAuditFailureV1::RuntimeAuthorityPresent => 5,
        MetabolicAuditFailureV1::SchedulerAuthorityPresent => 6,
        MetabolicAuditFailureV1::GatewayAuthorityPresent => 7,
        MetabolicAuditFailureV1::PolicyMutationPresent => 8,
        MetabolicAuditFailureV1::EvidenceArchiveAuthorityPresent => 9,
        MetabolicAuditFailureV1::IdentityAuthorityPresent => 10,
    }
}
fn state_bytes(state: &HormoneStateV1) -> Vec<u8> {
    let mut out = Vec::new();
    push_i64(&mut out, i64::from(state.dopamine_like.as_units()));
    push_i64(&mut out, i64::from(state.serotonin_like.as_units()));
    push_i64(&mut out, i64::from(state.cortisol_like.as_units()));
    push_i64(&mut out, i64::from(state.arousal_like.as_units()));
    push_i64(&mut out, i64::from(state.sleep_pressure.as_units()));
    push_i64(&mut out, i64::from(state.novelty_pressure.as_units()));
    push_i64(&mut out, i64::from(state.stability_pressure.as_units()));
    out
}
fn modulation_bytes(modulation: &HormoneModulationOutputV1) -> Vec<u8> {
    let mut out = Vec::new();
    push_i64(&mut out, modulation.attention_gain);
    push_i64(&mut out, modulation.learning_rate_multiplier);
    push_i64(&mut out, modulation.replay_priority_multiplier);
    push_i64(&mut out, modulation.noise_scale);
    push_i64(&mut out, modulation.consolidation_gate);
    push_i64(&mut out, modulation.sleep_pressure_delta);
    push_i64(&mut out, modulation.risk_damping);
    out
}
fn candidate_bytes(candidates: &MetabolicReplaySleepCandidatesV1) -> Vec<u8> {
    let mut out = Vec::new();
    let replay = &candidates.replay;
    push_i64(&mut out, replay.priority_hint);
    push_i64(&mut out, replay.novelty_component);
    push_i64(&mut out, replay.stability_component);
    push_i64(&mut out, replay.arousal_component);
    push_i64(&mut out, replay.risk_damping_component);
    let sleep = &candidates.sleep;
    push_i64(&mut out, sleep.pressure_hint);
    push_i64(&mut out, sleep.sleep_delta_component);
    push_i64(&mut out, sleep.risk_damping_component);
    out
}
