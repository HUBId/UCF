use ucf_brainbus::v0::{OscPhase, Spike};
use ucf_core::types::Tick;
use ucf_frames::v1::{BrainStimulusPayload, ControlFrame};

pub struct BrainStimulusEncoder;

impl BrainStimulusEncoder {
    pub fn encode_to_spikes(ctrl: &ControlFrame, payload: &BrainStimulusPayload) -> Vec<Spike> {
        let n = (payload.duration_ms / 10).clamp(1, 8) as u64;

        (0..n)
            .map(|offset_ms| Spike {
                time: ucf_core::types::SimTime {
                    tick: Tick::new(ctrl.time.tick.get().saturating_add(offset_ms)),
                    window: ctrl.time.window,
                },
                corr: ctrl.corr,
                src: 1,
                dst: payload.target,
                code: payload.intensity,
                phase: None,
            })
            .collect()
    }

    pub fn attach_phase(spikes: &mut [Spike], cycle_hz: f32, theta_deg: f32) {
        let phase = OscPhase::new(cycle_hz, theta_deg);
        for spike in spikes {
            spike.phase = Some(phase);
        }
    }
}

#[cfg(test)]
mod tests {
    use ucf_core::types::{SimTime, Tick, WindowId};
    use ucf_frames::v1::{
        BrainStimulusKind, BrainStimulusPayload, ChannelCode, ControlFrame, ControlPayload,
        CorrelationId, Intent, IntentId, IntentKind,
    };

    use super::BrainStimulusEncoder;

    fn sim_time() -> SimTime {
        SimTime {
            tick: Tick::new(100),
            window: WindowId::new(0),
        }
    }

    fn intent() -> Intent {
        Intent::new(IntentId(99), IntentKind::System, "bridge-test")
    }

    #[test]
    fn encoder_is_deterministic() {
        let ctrl = ControlFrame {
            time: sim_time(),
            corr: CorrelationId(7),
            channel: ChannelCode::BrainStimulus,
            intent: intent(),
            payload: ControlPayload::Empty,
        };
        let payload = BrainStimulusPayload {
            kind: BrainStimulusKind::SpikeTrain,
            target: 42,
            intensity: 300,
            duration_ms: 37,
        };

        let first = BrainStimulusEncoder::encode_to_spikes(&ctrl, &payload);
        let second = BrainStimulusEncoder::encode_to_spikes(&ctrl, &payload);

        assert_eq!(first, second);
        assert_eq!(first.len(), 3);
        assert_eq!(first[0].time.tick.get(), 100);
        assert_eq!(first[1].time.tick.get(), 101);
        assert_eq!(first[2].time.tick.get(), 102);
    }

    #[test]
    fn attach_phase_normalizes_theta() {
        let ctrl = ControlFrame {
            time: sim_time(),
            corr: CorrelationId(7),
            channel: ChannelCode::BrainStimulus,
            intent: intent(),
            payload: ControlPayload::Empty,
        };
        let payload = BrainStimulusPayload {
            kind: BrainStimulusKind::SpikeTrain,
            target: 42,
            intensity: 300,
            duration_ms: 20,
        };

        let mut spikes = BrainStimulusEncoder::encode_to_spikes(&ctrl, &payload);
        BrainStimulusEncoder::attach_phase(&mut spikes, 40.0, -90.0);

        assert_eq!(spikes.len(), 2);
        for spike in spikes {
            let phase = spike.phase.expect("phase must be set");
            assert_eq!(phase.cycle_hz, 40.0);
            assert!((phase.theta_deg - 270.0).abs() < 1e-6);
        }
    }
}
