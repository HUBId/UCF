#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use blake3::Hasher;
use ucf_attn_controller::{AttentionWeights, FocusChannel};
use ucf_bluebrain_port::{BrainRegion, BrainStimulus, Spike};
use ucf_feature_translator::LensSelection;
use ucf_predictive_coding::SurpriseSignal;
use ucf_sandbox::ControlFrameNormalized;
use ucf_types::Digest32;
use ucf_workspace::{SignalKind, WorkspaceSnapshot};

const WIDTH_SHORT: u16 = 80;
const WIDTH_LONG: u16 = 120;
const REPLAY_BIAS_HIGH: u16 = 7000;

pub fn map_to_stimulus(
    cf: &ControlFrameNormalized,
    ws: &WorkspaceSnapshot,
    attn: &AttentionWeights,
    surprise: Option<&SurpriseSignal>,
    lens: Option<&LensSelection>,
) -> BrainStimulus {
    let mut spikes = BTreeMap::<u16, SpikeAccumulator>::new();
    if let Some(selection) = lens {
        for feature in &selection.topk {
            let region = feature_region(feature.id);
            let amplitude = feature_amplitude(feature.weight);
            let width = width_for_channel(attn.channel);
            add_spike(&mut spikes, region, amplitude, width);
        }
    } else {
        let surprise_score = surprise.map(|signal| signal.score).unwrap_or(0);
        let surprise_critical = surprise
            .map(|signal| matches!(signal.band, ucf_predictive_coding::SurpriseBand::Critical))
            .unwrap_or(false);

        if attn.channel == FocusChannel::Threat || surprise_critical {
            let hyp_amp = scaled_amp(6000, surprise_score / 2);
            let stem_amp = scaled_amp(5000, surprise_score / 3);
            add_spike(&mut spikes, BrainRegion::Hypothalamus, hyp_amp, WIDTH_LONG);
            add_spike(&mut spikes, BrainRegion::Brainstem, stem_amp, WIDTH_LONG);
        }

        if attn.channel == FocusChannel::Task {
            let task_amp = scaled_amp(3500, attn.gain / 3);
            add_spike(&mut spikes, BrainRegion::PFC, task_amp, WIDTH_SHORT);
            add_spike(&mut spikes, BrainRegion::Thalamus, task_amp, WIDTH_SHORT);
        }

        if attn.replay_bias >= REPLAY_BIAS_HIGH {
            let replay_amp = scaled_amp(3000, attn.replay_bias / 2);
            add_spike(
                &mut spikes,
                BrainRegion::Hippocampus,
                replay_amp,
                WIDTH_LONG,
            );
        }

        if attn.channel == FocusChannel::Social {
            let social_amp = scaled_amp(2500, attn.gain / 4);
            add_spike(&mut spikes, BrainRegion::Insula, social_amp, WIDTH_SHORT);
        }

        let suppression_count = output_suppression_count(ws);
        if suppression_count > 0 {
            let base = suppression_count.saturating_mul(1000);
            let nacc_amp = scaled_amp(1200, base);
            add_spike(&mut spikes, BrainRegion::NAcc, nacc_amp, WIDTH_SHORT);
        }
    }

    let spikes = spikes
        .into_values()
        .map(|acc| Spike::new(acc.region, acc.amplitude, acc.width))
        .collect();

    let seed = stimulus_seed(cf, ws, attn, surprise);
    BrainStimulus::with_seed(ws.cycle_id, spikes, seed)
}

struct SpikeAccumulator {
    region: BrainRegion,
    amplitude: u16,
    width: u16,
}

fn add_spike(
    spikes: &mut BTreeMap<u16, SpikeAccumulator>,
    region: BrainRegion,
    amplitude: u16,
    width: u16,
) {
    if amplitude == 0 {
        return;
    }
    let key = region.code();
    spikes
        .entry(key)
        .and_modify(|entry| {
            entry.amplitude = entry.amplitude.saturating_add(amplitude);
            entry.width = entry.width.max(width);
        })
        .or_insert(SpikeAccumulator {
            region,
            amplitude,
            width,
        });
}

fn scaled_amp(base: u16, boost: u16) -> u16 {
    base.saturating_add(boost).min(10_000)
}

fn feature_region(id: u32) -> BrainRegion {
    match id {
        0..=9_999 => BrainRegion::Thalamus,
        10_000..=19_999 => BrainRegion::PFC,
        20_000..=29_999 => BrainRegion::NAcc,
        30_000..=39_999 => BrainRegion::Insula,
        _ => BrainRegion::Hypothalamus,
    }
}

fn feature_amplitude(weight: i16) -> u16 {
    let magnitude = i32::from(weight).abs().min(10_000);
    magnitude as u16
}

fn width_for_channel(channel: FocusChannel) -> u16 {
    match channel {
        FocusChannel::Threat => WIDTH_LONG,
        FocusChannel::Task => WIDTH_SHORT,
        FocusChannel::Social => WIDTH_SHORT,
        FocusChannel::Memory => WIDTH_LONG,
        FocusChannel::Exploration => WIDTH_LONG,
        FocusChannel::Idle => WIDTH_SHORT,
    }
}

fn output_suppression_count(ws: &WorkspaceSnapshot) -> u16 {
    ws.broadcast
        .iter()
        .filter(|signal| {
            matches!(signal.kind, SignalKind::Output) && signal.summary.contains("OUTPUT=SUPPRESS")
        })
        .count()
        .min(u16::MAX as usize) as u16
}

fn stimulus_seed(
    cf: &ControlFrameNormalized,
    ws: &WorkspaceSnapshot,
    attn: &AttentionWeights,
    surprise: Option<&SurpriseSignal>,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.bluebrain.stimulus.seed.v1");
    hasher.update(cf.commitment().digest.as_bytes());
    hasher.update(ws.commit.as_bytes());
    hasher.update(attn.commit.as_bytes());
    if let Some(signal) = surprise {
        hasher.update(signal.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_attn_controller::FocusChannel;
    use ucf_feature_translator::{LensSelection, SparseFeature};
    use ucf_predictive_coding::{SurpriseBand, SurpriseSignal};
    use ucf_types::v1::spec::ControlFrame;
    use ucf_types::v1::spec::{ActionCode, DecisionKind, PolicyDecision};
    use ucf_workspace::WorkspaceSignal;

    fn control_frame() -> ControlFrameNormalized {
        let frame = ControlFrame {
            frame_id: "frame-1".to_string(),
            issued_at_ms: 1,
            decision: Some(PolicyDecision {
                kind: DecisionKind::DecisionKindAllow as i32,
                action: ActionCode::ActionCodeContinue as i32,
                rationale: "ok".to_string(),
                confidence_bp: 1000,
                constraint_ids: Vec::new(),
            }),
            evidence_ids: vec![],
            policy_id: "policy-1".to_string(),
        };
        ucf_sandbox::normalize(frame)
    }

    fn workspace_snapshot() -> WorkspaceSnapshot {
        WorkspaceSnapshot {
            cycle_id: 9,
            broadcast: vec![WorkspaceSignal {
                kind: SignalKind::Output,
                priority: 9000,
                digest: Digest32::new([1u8; 32]),
                summary: "OUTPUT=SUPPRESS".to_string(),
                slot: 0,
            }],
            recursion_used: 0,
            spike_root_commit: Digest32::new([0u8; 32]),
            ncde_commit: Digest32::new([0u8; 32]),
            commit: Digest32::new([2u8; 32]),
        }
    }

    fn workspace_snapshot_empty() -> WorkspaceSnapshot {
        WorkspaceSnapshot {
            cycle_id: 9,
            broadcast: Vec::new(),
            recursion_used: 0,
            spike_root_commit: Digest32::new([0u8; 32]),
            ncde_commit: Digest32::new([0u8; 32]),
            commit: Digest32::new([2u8; 32]),
        }
    }

    fn attention(channel: FocusChannel) -> AttentionWeights {
        AttentionWeights {
            channel,
            gain: 8000,
            noise_suppress: 2000,
            replay_bias: 9000,
            commit: Digest32::new([3u8; 32]),
        }
    }

    #[test]
    fn map_to_stimulus_is_deterministic() {
        let cf = control_frame();
        let ws = workspace_snapshot();
        let attn = attention(FocusChannel::Task);
        let surprise = SurpriseSignal {
            score: 9000,
            band: SurpriseBand::High,
            commit: Digest32::new([4u8; 32]),
        };

        let first = map_to_stimulus(&cf, &ws, &attn, Some(&surprise), None);
        let second = map_to_stimulus(&cf, &ws, &attn, Some(&surprise), None);

        assert_eq!(first, second);
    }

    #[test]
    fn threat_focus_produces_hypothalamus_spike() {
        let cf = control_frame();
        let ws = workspace_snapshot();
        let attn = attention(FocusChannel::Threat);
        let stim = map_to_stimulus(&cf, &ws, &attn, None, None);

        assert!(stim
            .spikes
            .iter()
            .any(|spike| spike.region == BrainRegion::Hypothalamus));
    }

    #[test]
    fn lens_selection_drives_spikes_when_present() {
        let cf = control_frame();
        let ws = workspace_snapshot_empty();
        let attn = AttentionWeights {
            channel: FocusChannel::Idle,
            gain: 1000,
            noise_suppress: 0,
            replay_bias: 0,
            commit: Digest32::new([5u8; 32]),
        };
        let seed = Digest32::new([9u8; 32]);
        let selection = LensSelection::new(vec![SparseFeature::new(15_000, 1200, seed)], seed);

        let without = map_to_stimulus(&cf, &ws, &attn, None, None);
        let with = map_to_stimulus(&cf, &ws, &attn, None, Some(&selection));

        assert!(without.spikes.is_empty());
        assert!(with
            .spikes
            .iter()
            .any(|spike| spike.region == BrainRegion::PFC));
    }
}
