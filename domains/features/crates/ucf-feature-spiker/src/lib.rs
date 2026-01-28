#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_onn::{OscId, PhaseFrame};
use ucf_spike_encoder::encode_spike_with_window;
use ucf_spikebus::{SpikeBatch, SpikeKind};
use ucf_types::Digest32;

const FEATURE_SPIKE_CAP: usize = 32;
const SALT_DOMAIN: &[u8] = b"ucf.feature_spiker.salt.v1";
const MAX_SIGNAL: u16 = 10_000;

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

pub fn build_feature_spike_batch<F>(
    cycle_id: u64,
    phase: &PhaseFrame,
    phase_window_for_dst: F,
    inputs: FeatureSpikerInputs,
) -> (SpikeBatch, FeatureSpikeSummary)
where
    F: Fn(OscId) -> u16,
{
    let mut events = Vec::new();
    let mut candidates = Vec::new();

    push_candidate(
        &mut candidates,
        SpikeKind::Feature,
        OscId::Ssm,
        inputs.ssm_salience,
        b"ssm.salience",
    );
    push_candidate(
        &mut candidates,
        SpikeKind::Novelty,
        OscId::Ssm,
        inputs.ssm_novelty,
        b"ssm.novelty",
    );
    push_candidate(
        &mut candidates,
        SpikeKind::Feature,
        OscId::BlueBrain,
        inputs.wm_salience,
        b"wm.salience",
    );
    push_candidate(
        &mut candidates,
        SpikeKind::Novelty,
        OscId::BlueBrain,
        inputs.wm_novelty,
        b"wm.novelty",
    );
    push_candidate(
        &mut candidates,
        SpikeKind::Threat,
        OscId::Nsr,
        inputs.risk,
        b"risk.nsr",
    );
    push_candidate(
        &mut candidates,
        SpikeKind::Threat,
        OscId::Geist,
        inputs.risk,
        b"risk.geist",
    );
    push_candidate(
        &mut candidates,
        SpikeKind::Novelty,
        OscId::Jepa,
        inputs.surprise,
        b"surprise.replay",
    );
    push_candidate(
        &mut candidates,
        SpikeKind::Feature,
        OscId::Geist,
        inputs.ssm_salience,
        b"ssm.salience.geist",
    );

    if let Some(conf) = inputs.cde_top_conf {
        push_candidate(
            &mut candidates,
            SpikeKind::Feature,
            OscId::Cde,
            conf,
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
        let phase_window = phase_window_for_dst(candidate.dst);
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
    label: &'static [u8],
) {
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
        let inputs = FeatureSpikerInputs {
            ssm_salience: 2000,
            ssm_novelty: 3000,
            wm_salience: 1000,
            wm_novelty: 500,
            risk: 4000,
            surprise: 2500,
            cde_top_conf: Some(6000),
        };
        let (batch_a, _) = build_feature_spike_batch(1, &phase, |_| 1024, inputs);
        let (batch_b, _) = build_feature_spike_batch(1, &phase, |_| 1024, inputs);
        assert_eq!(batch_a.root, batch_b.root);
        assert_eq!(batch_a.commit, batch_b.commit);
    }
}
