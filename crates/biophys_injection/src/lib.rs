#![forbid(unsafe_code)]

use biophys_synapses_l4::{max_synapse_g_fixed, SynKind};

pub const MAX_SPIKES_PER_TICK: usize = 256;
pub const MAX_TARGETS_PER_SPIKE: usize = 16;
pub const MAX_LABEL_LEN: usize = 32;
pub const MAX_REASON_CODES: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalSpike {
    pub region: String,
    pub population: String,
    pub neuron_group: u32,
    pub syn_kind: SynKind,
    pub amplitude_q: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InjectedTarget {
    pub neuron_id: u32,
    pub compartment_id: u32,
    pub syn_kind: SynKind,
    pub g_add_q: u32,
}

pub trait SpikeRouter {
    fn route(&self, spike: &ExternalSpike) -> Vec<InjectedTarget>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InjectionReport {
    pub received_spikes: u32,
    pub dropped_spikes: u32,
    pub emitted_targets: u32,
    pub dropped_targets: u32,
    pub reason_codes: Vec<String>,
}

impl InjectionReport {
    pub fn new(
        received_spikes: u32,
        dropped_spikes: u32,
        emitted_targets: u32,
        dropped_targets: u32,
        mut reason_codes: Vec<String>,
    ) -> Self {
        reason_codes.sort();
        reason_codes.dedup();
        if reason_codes.len() > MAX_REASON_CODES {
            reason_codes.truncate(MAX_REASON_CODES);
        }
        Self {
            received_spikes,
            dropped_spikes,
            emitted_targets,
            dropped_targets,
            reason_codes,
        }
    }
}

fn clamp_label(label: &str) -> String {
    label.chars().take(MAX_LABEL_LEN).collect()
}

pub fn normalize_spikes(spikes: &mut Vec<ExternalSpike>) -> u32 {
    for spike in spikes.iter_mut() {
        spike.region = clamp_label(&spike.region);
        spike.population = clamp_label(&spike.population);
        spike.amplitude_q = spike.amplitude_q.min(1000);
    }
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
    if spikes.len() > MAX_SPIKES_PER_TICK {
        let dropped = spikes.len().saturating_sub(MAX_SPIKES_PER_TICK) as u32;
        spikes.truncate(MAX_SPIKES_PER_TICK);
        dropped
    } else {
        0
    }
}

#[derive(Debug, Clone)]
pub struct DefaultRouter {
    pub neuron_count: u32,
    pub soma_compartment_id: u32,
    pub proximal_dendrite_compartment_id: u32,
    pub distal_dendrite_compartment_id: u32,
}

impl DefaultRouter {
    fn hash_key(&self, spike: &ExternalSpike) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        let mut update = |byte: u8| {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        };
        for byte in spike.region.as_bytes() {
            update(*byte);
        }
        update(b'|');
        for byte in spike.population.as_bytes() {
            update(*byte);
        }
        update(b'|');
        for byte in spike.neuron_group.to_le_bytes() {
            update(byte);
        }
        hash
    }

    fn compartment_for(&self, kind: SynKind) -> u32 {
        match kind {
            SynKind::AMPA => self.proximal_dendrite_compartment_id,
            SynKind::NMDA => self.distal_dendrite_compartment_id,
            SynKind::GABA => self.soma_compartment_id,
        }
    }
}

impl SpikeRouter for DefaultRouter {
    fn route(&self, spike: &ExternalSpike) -> Vec<InjectedTarget> {
        if self.neuron_count == 0 {
            return Vec::new();
        }
        let hash = self.hash_key(spike);
        let neuron_id = (hash % self.neuron_count as u64) as u32;
        let g_add_q = (max_synapse_g_fixed() as u64 * spike.amplitude_q as u64 / 1000) as u32;
        let targets = vec![InjectedTarget {
            neuron_id,
            compartment_id: self.compartment_for(spike.syn_kind),
            syn_kind: spike.syn_kind,
            g_add_q,
        }];
        targets
    }
}
