#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_onn::{OscId, PhaseFrame};
use ucf_spike_encoder::encode_spike_with_window;
use ucf_spikebus::{SpikeBatch, SpikeKind};
use ucf_types::Digest32;

const FEATURE_SPIKE_CAP: usize = 32;
const SALT_DOMAIN: &[u8] = b"ucf.feature_spiker.salt.v1";
const MAX_SIGNAL: u16 = 10_000;
const FEATURE_THRESH_MIN: u16 = 200;
const FEATURE_THRESH_MAX: u16 = 9_000;
const THREAT_THRESH_MIN: u16 = 200;
const THREAT_THRESH_MAX: u16 = 9_000;
const PARAM_DOMAIN: &[u8] = b"ucf.feature_spiker.params.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeatureSpikeParams {
    pub feature_thresh: u16,
    pub threat_thresh: u16,
    pub commit: Digest32,
}

impl FeatureSpikeParams {
    pub fn new(feature_thresh: u16, threat_thresh: u16) -> Self {
        let feature_thresh = feature_thresh.clamp(FEATURE_THRESH_MIN, FEATURE_THRESH_MAX);
        let threat_thresh = threat_thresh.clamp(THREAT_THRESH_MIN, THREAT_THRESH_MAX);
        let commit = commit_params(feature_thresh, threat_thresh);
        Self {
            feature_thresh,
            threat_thresh,
            commit,
        }
    }
}

impl Default for FeatureSpikeParams {
    fn default() -> Self {
        Self::new(1200, 1500)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FeatureSpikerInputs {
    pub ssm_salience: u16,
    pub ssm_novelty: u16,
    pub wm_salience: u16,
    pub wm_novelty: u16,
    pub risk: u16,
    pub surprise: u16,
    pub cde_top_conf: Option<u16>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeatureSpikeSummary {
    pub produced: usize,
    pub cap_hit: bool,
}

pub fn build_feature_spike_batch(
    cycle_id: u64,
    phase: &PhaseFrame,
    phase_window: u16,
    params: FeatureSpikeParams,
    inputs: FeatureSpikerInputs,
) -> (SpikeBatch, FeatureSpikeSummary) {
    let mut events = Vec::new();
    let mut candidates = Vec::new();

    push_candidate(
        &mut candidates,
        SpikeKind::Feature,
        OscId::Ssm,
        inputs.ssm_salience,
        params.feature_thresh,
        b"ssm.salience",
    );
    push_candidate(
        &mut candidates,
        SpikeKind::Novelty,
        OscId::Ssm,
        inputs.ssm_novelty,
        params.feature_thresh,
        b"ssm.novelty",
    );
    push_candidate(
        &mut candidates,
        SpikeKind::Feature,
        OscId::BlueBrain,
        inputs.wm_salience,
        params.feature_thresh,
        b"wm.salience",
    );
    push_candidate(
        &mut candidates,
        SpikeKind::Novelty,
        OscId::BlueBrain,
        inputs.wm_novelty,
        params.feature_thresh,
        b"wm.novelty",
    );
    push_candidate(
        &mut candidates,
        SpikeKind::Threat,
        OscId::Nsr,
        inputs.risk,
        params.threat_thresh,
        b"risk.nsr",
    );
    push_candidate(
        &mut candidates,
        SpikeKind::Threat,
        OscId::Geist,
        inputs.risk,
        params.threat_thresh,
        b"risk.geist",
    );
    push_candidate(
        &mut candidates,
        SpikeKind::Novelty,
        OscId::Jepa,
        inputs.surprise,
        params.feature_thresh,
        b"surprise.replay",
    );
    push_candidate(
        &mut candidates,
        SpikeKind::Feature,
        OscId::Geist,
        inputs.ssm_salience,
        params.feature_thresh,
        b"ssm.salience.geist",
    );

    if let Some(conf) = inputs.cde_top_conf {
        push_candidate(
            &mut candidates,
            SpikeKind::Feature,
            OscId::Cde,
            conf,
            params.feature_thresh,
            b"cde.edge.conf",
        );
    }

    candidates.sort_by(|a, b| {
        b.strength
            .cmp(&a.strength)
            .then_with(|| a.label.cmp(b.label))
    });

    for candidate in candidates.into_iter().take(FEATURE_SPIKE_CAP) {
        if candidate.strength == 0 {
            continue;
        }
        let salt = feature_salt(candidate.label, candidate.strength, phase.commit);
        let event = encode_spike_with_window(
            cycle_id,
            candidate.kind,
            OscId::Jepa,
            candidate.dst,
            candidate.strength,
            phase.commit,
            salt,
            phase_window,
        );
        events.push(event);
    }

    let cap_hit = events.len() >= FEATURE_SPIKE_CAP;
    let batch = SpikeBatch::new(cycle_id, phase.commit, events);
    let produced = batch.events.len();
    (batch, FeatureSpikeSummary { produced, cap_hit })
}

#[derive(Clone)]
struct Candidate {
    kind: SpikeKind,
    dst: OscId,
    strength: u16,
    label: &'static [u8],
}

fn push_candidate(
    candidates: &mut Vec<Candidate>,
    kind: SpikeKind,
    dst: OscId,
    strength: u16,
    threshold: u16,
    label: &'static [u8],
) {
    if strength < threshold {
        return;
    }
    candidates.push(Candidate {
        kind,
        dst,
        strength: clamp_signal(strength),
        label,
    });
}

fn clamp_signal(value: u16) -> u16 {
    value.min(MAX_SIGNAL)
}

fn feature_salt(label: &[u8], strength: u16, phase_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(SALT_DOMAIN);
    hasher.update(label);
    hasher.update(&strength.to_be_bytes());
    hasher.update(phase_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

pub fn apply_feature_thresh_delta(params: &FeatureSpikeParams, delta: i16) -> FeatureSpikeParams {
    let feature_thresh = apply_i16_delta(
        params.feature_thresh,
        delta,
        FEATURE_THRESH_MIN,
        FEATURE_THRESH_MAX,
    );
    FeatureSpikeParams::new(feature_thresh, params.threat_thresh)
}

pub fn apply_threat_thresh_delta(params: &FeatureSpikeParams, delta: i16) -> FeatureSpikeParams {
    let threat_thresh = apply_i16_delta(
        params.threat_thresh,
        delta,
        THREAT_THRESH_MIN,
        THREAT_THRESH_MAX,
    );
    FeatureSpikeParams::new(params.feature_thresh, threat_thresh)
}

fn apply_i16_delta(value: u16, delta: i16, min: u16, max: u16) -> u16 {
    let value = i32::from(value);
    let delta = i32::from(delta);
    let updated = value
        .saturating_add(delta)
        .clamp(i32::from(min), i32::from(max));
    updated as u16
}

fn commit_params(feature_thresh: u16, threat_thresh: u16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(PARAM_DOMAIN);
    hasher.update(&feature_thresh.to_be_bytes());
    hasher.update(&threat_thresh.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn phase_frame() -> PhaseFrame {
        PhaseFrame {
            cycle_id: 1,
            global_phase: 1200,
            module_phase: Vec::new(),
            coherence_plv: 9000,
            pair_locks: Vec::new(),
            states_commit: Digest32::new([0u8; 32]),
            phase_frame_commit: Digest32::new([0u8; 32]),
            commit: Digest32::new([4u8; 32]),
        }
    }

    #[test]
    fn feature_spiker_is_deterministic() {
        let phase = phase_frame();
        let params = FeatureSpikeParams::default();
        let inputs = FeatureSpikerInputs {
            ssm_salience: 2000,
            ssm_novelty: 3000,
            wm_salience: 1000,
            wm_novelty: 500,
            risk: 4000,
            surprise: 2500,
            cde_top_conf: Some(6000),
        };
        let (batch_a, _) = build_feature_spike_batch(1, &phase, 1024, params, inputs);
        let (batch_b, _) = build_feature_spike_batch(1, &phase, 1024, params, inputs);
        assert_eq!(batch_a.root, batch_b.root);
        assert_eq!(batch_a.commit, batch_b.commit);
    }

    #[test]
    fn rsa_param_updates_are_clamped() {
        let params = FeatureSpikeParams::default();
        let updated = apply_feature_thresh_delta(&params, 20_000);
        assert_eq!(updated.feature_thresh, FEATURE_THRESH_MAX);

        let updated = apply_threat_thresh_delta(&params, -20_000);
        assert_eq!(updated.threat_thresh, THREAT_THRESH_MIN);
    }

    #[test]
    fn ssm_metrics_drive_spikes() {
        let phase = phase_frame();
        let params = FeatureSpikeParams::new(800, 1200);
        let inputs = FeatureSpikerInputs {
            ssm_salience: 9000,
            ssm_novelty: 8800,
            wm_salience: 0,
            wm_novelty: 0,
            risk: 0,
            surprise: 0,
            cde_top_conf: None,
        };

        let (batch, summary) = build_feature_spike_batch(2, &phase, 1024, params, inputs);
        assert!(summary.produced > 0);
        assert!(batch.events.iter().any(|event| event.dst == OscId::Ssm));
    }
}
