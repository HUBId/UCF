#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_feature_translator::LensSelection;
use ucf_onn::{ModuleId, PhaseFrame};
use ucf_spikebus::{SpikeEvent, SpikeKind};
use ucf_structural_store::SnnKnobs;
use ucf_types::Digest32;

const TTFS_DOMAIN: &[u8] = b"ucf.spike_encoder.ttfs.v1";
const PAYLOAD_DOMAIN: &[u8] = b"ucf.spike_encoder.payload.v1";
const MAX_SIGNAL: u16 = 10_000;

#[allow(clippy::too_many_arguments)]
pub fn encode_from_features(
    cycle_id: u64,
    phase: &PhaseFrame,
    src: ModuleId,
    lens: Option<&LensSelection>,
    snn: &SnnKnobs,
    surprise: u16,
    drift: u16,
    risk: u16,
) -> Vec<SpikeEvent> {
    let attention_gain = lens
        .map(attention_gain_from_lens)
        .unwrap_or(0)
        .min(MAX_SIGNAL);
    let mut events = Vec::new();
    if let Some(selection) = lens {
        for feature in &selection.topk {
            let weight = feature.weight.unsigned_abs();
            let amplitude = clamp_signal(weight);
            let payload_commit = feature.commit;
            for dst in [ModuleId::Cde, ModuleId::Nsr] {
                if meets_threshold(snn, SpikeKind::CausalLink, amplitude) {
                    events.push(build_spike(
                        cycle_id,
                        phase,
                        src,
                        dst,
                        SpikeKind::CausalLink,
                        amplitude,
                        attention_gain,
                        payload_commit,
                    ));
                }
            }
        }
    }

    if surprise > 0 {
        let amplitude = clamp_signal(surprise);
        if meets_threshold(snn, SpikeKind::Novelty, amplitude) {
            let payload_commit = commit_signal_payload(phase.commit, SpikeKind::Novelty, surprise);
            for dst in [ModuleId::Ai, ModuleId::Ssm] {
                events.push(build_spike(
                    cycle_id,
                    phase,
                    src,
                    dst,
                    SpikeKind::Novelty,
                    amplitude,
                    attention_gain,
                    payload_commit,
                ));
            }
        }
    }

    if drift > 0 {
        let amplitude = clamp_signal(drift);
        if meets_threshold(snn, SpikeKind::ConsistencyAlert, amplitude) {
            let payload_commit =
                commit_signal_payload(phase.commit, SpikeKind::ConsistencyAlert, drift);
            for dst in [ModuleId::Geist, ModuleId::Replay] {
                events.push(build_spike(
                    cycle_id,
                    phase,
                    src,
                    dst,
                    SpikeKind::ConsistencyAlert,
                    amplitude,
                    attention_gain,
                    payload_commit,
                ));
            }
        }
    }

    if risk > 0 {
        let amplitude = clamp_signal(risk);
        if meets_threshold(snn, SpikeKind::Threat, amplitude) {
            let payload_commit = commit_signal_payload(phase.commit, SpikeKind::Threat, risk);
            for dst in [ModuleId::BlueBrain, ModuleId::Geist] {
                events.push(build_spike(
                    cycle_id,
                    phase,
                    src,
                    dst,
                    SpikeKind::Threat,
                    amplitude,
                    attention_gain,
                    payload_commit,
                ));
            }
        }
    }

    events
}

#[allow(clippy::too_many_arguments)]
pub fn encode_causal_link_spike(
    cycle_id: u64,
    phase: &PhaseFrame,
    src: ModuleId,
    dst: ModuleId,
    amplitude: u16,
    attention_gain: u16,
    payload_commit: Digest32,
) -> SpikeEvent {
    let amplitude = clamp_signal(amplitude);
    build_spike(
        cycle_id,
        phase,
        src,
        dst,
        SpikeKind::CausalLink,
        amplitude,
        attention_gain.min(MAX_SIGNAL),
        payload_commit,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_spike(
    cycle_id: u64,
    phase: &PhaseFrame,
    src: ModuleId,
    dst: ModuleId,
    kind: SpikeKind,
    amplitude: u16,
    attention_gain: u16,
    payload_commit: Digest32,
) -> SpikeEvent {
    let ttfs_code = ttfs_code(phase.global_phase, payload_commit, kind, dst, amplitude);
    let width = spike_width(kind, attention_gain);
    SpikeEvent::new(
        cycle_id,
        src,
        dst,
        kind,
        ttfs_code,
        amplitude,
        width,
        phase.commit,
        payload_commit,
    )
}

fn ttfs_code(
    base: u16,
    payload_commit: Digest32,
    kind: SpikeKind,
    dst: ModuleId,
    amplitude: u16,
) -> u16 {
    let offset_seed = hash16(payload_commit, kind, dst);
    let window = offset_window(amplitude);
    let offset = if window == 0 {
        offset_seed
    } else {
        offset_seed % window
    };
    base.wrapping_add(offset)
}

fn hash16(payload_commit: Digest32, kind: SpikeKind, dst: ModuleId) -> u16 {
    let mut hasher = Hasher::new();
    hasher.update(TTFS_DOMAIN);
    hasher.update(payload_commit.as_bytes());
    hasher.update(&kind.as_u16().to_be_bytes());
    hasher.update(&dst.as_u16().to_be_bytes());
    let bytes = hasher.finalize();
    let raw = bytes.as_bytes();
    u16::from_be_bytes([raw[0], raw[1]])
}

fn offset_window(amplitude: u16) -> u16 {
    let strength = u32::from(amplitude.min(MAX_SIGNAL));
    let reduction = strength.saturating_mul(6).min(60_000);
    let window = 65_535u32.saturating_sub(reduction);
    window.clamp(256, 65_535) as u16
}

fn spike_width(kind: SpikeKind, attention_gain: u16) -> u16 {
    let base: u16 = match kind {
        SpikeKind::Threat => 7_000,
        SpikeKind::ConsistencyAlert => 6_000,
        SpikeKind::CausalLink => 5_000,
        SpikeKind::Novelty => 4_000,
        SpikeKind::ReplayTrigger => 4_500,
        SpikeKind::AttentionShift => 3_500,
        SpikeKind::Unknown(_) => 3_000,
    };
    base.saturating_add(attention_gain / 4).min(MAX_SIGNAL)
}

fn attention_gain_from_lens(selection: &LensSelection) -> u16 {
    if selection.topk.is_empty() {
        return 0;
    }
    let sum: u32 = selection
        .topk
        .iter()
        .map(|feature| u32::from(feature.weight.unsigned_abs()))
        .sum();
    (sum / selection.topk.len() as u32).min(u32::from(MAX_SIGNAL)) as u16
}

fn clamp_signal(value: u16) -> u16 {
    value.min(MAX_SIGNAL)
}

fn meets_threshold(snn: &SnnKnobs, kind: SpikeKind, amplitude: u16) -> bool {
    amplitude >= snn.threshold_for(kind)
}

fn commit_signal_payload(phase_commit: Digest32, kind: SpikeKind, value: u16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(PAYLOAD_DOMAIN);
    hasher.update(phase_commit.as_bytes());
    hasher.update(&kind.as_u16().to_be_bytes());
    hasher.update(&value.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_feature_translator::{LensSelection, SparseFeature};

    fn make_phase() -> PhaseFrame {
        PhaseFrame {
            cycle_id: 1,
            global_phase: 32000,
            module_phase: Vec::new(),
            module_freq: Vec::new(),
            coherence_plv: 9000,
            commit: Digest32::new([5u8; 32]),
        }
    }

    #[test]
    fn encoding_is_deterministic_for_same_inputs() {
        let phase = make_phase();
        let lens = LensSelection::new(
            vec![SparseFeature::new(42, 1200, phase.commit)],
            phase.commit,
        );
        let snn = SnnKnobs::default();
        let first = encode_from_features(1, &phase, ModuleId::Ai, Some(&lens), &snn, 100, 200, 300);
        let second =
            encode_from_features(1, &phase, ModuleId::Ai, Some(&lens), &snn, 100, 200, 300);
        assert_eq!(
            first.iter().map(|ev| ev.commit).collect::<Vec<_>>(),
            second.iter().map(|ev| ev.commit).collect::<Vec<_>>()
        );
    }
}
