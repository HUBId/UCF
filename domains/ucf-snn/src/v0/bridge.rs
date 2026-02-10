use ucf_brainbus::v0::{OscPhase, Spike};
use ucf_core::types::{SimTime, Tick, WindowId};
use ucf_frames::v1::CorrelationId;

use crate::v0::SnnSpike;

pub fn to_brainbus(spikes: &[SnnSpike]) -> Vec<Spike> {
    spikes
        .iter()
        .map(|spike| {
            let time = SimTime {
                tick: Tick::new(spike.t.0),
                window: WindowId::new(0),
            };
            let mut out = Spike::new(time, CorrelationId(0), 0, spike.chan, spike.chan);
            if let Some(phase) = spike.phase {
                out = out.with_phase(OscPhase::new(0.0, phase.0));
            }
            out
        })
        .collect()
}
