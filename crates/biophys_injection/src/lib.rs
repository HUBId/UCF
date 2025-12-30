#![forbid(unsafe_code)]

use biophys_core::{CompartmentId, NeuronId};
use biophys_synapses_l4::SynKind;

pub const MAX_SPIKES_PER_TICK: usize = 256;
pub const MAX_TARGETS_PER_SPIKE: usize = 16;
pub const MAX_LABEL_LEN: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalSpike {
    pub region: String,
    pub population: String,
    pub neuron_group: u32,
    pub syn_kind: SynKind,
    pub amplitude_q: u16,
}

pub trait SpikeRouter {
    fn route(&self, spike: &ExternalSpike) -> Vec<(NeuronId, CompartmentId, SynKind, u16)>;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InjectionStats {
    pub dropped_spikes: u32,
    pub dropped_targets: u32,
}

fn normalize_spike(spike: &ExternalSpike) -> ExternalSpike {
    ExternalSpike {
        region: clamp_label(&spike.region),
        population: clamp_label(&spike.population),
        neuron_group: spike.neuron_group,
        syn_kind: spike.syn_kind,
        amplitude_q: spike.amplitude_q.min(1000),
    }
}

fn clamp_label(label: &str) -> String {
    label.chars().take(MAX_LABEL_LEN).collect()
}

pub fn sort_spikes(spikes: &mut [ExternalSpike]) {
    spikes.sort_by(|a, b| {
        (
            a.region.as_str(),
            a.population.as_str(),
            a.neuron_group,
            a.syn_kind as u8,
            a.amplitude_q,
        )
            .cmp(&(
                b.region.as_str(),
                b.population.as_str(),
                b.neuron_group,
                b.syn_kind as u8,
                b.amplitude_q,
            ))
    });
}

pub fn prepare_spikes(spikes: &[ExternalSpike]) -> (Vec<ExternalSpike>, u32) {
    let mut normalized = spikes.iter().map(normalize_spike).collect::<Vec<_>>();
    sort_spikes(&mut normalized);
    let dropped = if normalized.len() > MAX_SPIKES_PER_TICK {
        let dropped = normalized.len().saturating_sub(MAX_SPIKES_PER_TICK) as u32;
        normalized.truncate(MAX_SPIKES_PER_TICK);
        dropped
    } else {
        0
    };
    (normalized, dropped)
}
