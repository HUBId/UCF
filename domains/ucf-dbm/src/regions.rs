use crate::chemistry::NeuromodState;
use crate::hh::{hh_step, HhNeuron, NeuroTx, Synapse};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionKind {
    Amygdala,
    Pfc,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BrainRegion {
    pub kind: RegionKind,
    pub neurons: Vec<HhNeuron>,
    pub synapses: Vec<Synapse>,
    pub gaba_scale: f32,
    pub glu_scale: f32,
    pub ach_gate: f32,
}

impl BrainRegion {
    pub fn new(kind: RegionKind, n: usize) -> Self {
        let neurons = (0..n).map(|id| HhNeuron::new(id as u32)).collect();
        Self {
            kind,
            neurons,
            synapses: Vec::new(),
            gaba_scale: 1.0,
            glu_scale: 1.0,
            ach_gate: 0.5,
        }
    }
}

pub fn region_step(
    now_ms: u64,
    dt_ms: f32,
    r: &mut BrainRegion,
    mods: &NeuromodState,
    ext_drive: f32,
) -> (u32, f32) {
    let n = r.neurons.len();
    if n == 0 {
        return (0, 0.0);
    }

    let mut i_accum = vec![ext_drive; n];
    let mut spikes = 0_u32;
    let mut sum_v = 0.0;

    for idx in 0..n {
        let spiked = hh_step(now_ms, dt_ms, &mut r.neurons[idx], i_accum[idx], mods);
        if spiked {
            spikes = spikes.saturating_add(1);
            let pre_id = r.neurons[idx].id;
            for s in r.synapses.iter().filter(|s| s.pre == pre_id) {
                let post = s.post as usize;
                if post >= n {
                    continue;
                }
                match s.tx {
                    NeuroTx::Glu => {
                        i_accum[post] += s.w * r.glu_scale;
                    }
                    NeuroTx::Gaba => {
                        i_accum[post] -= s.w
                            * r.gaba_scale
                            * (1.0 + 0.3 * r.neurons[idx].sens_oxy * mods.oxytocin);
                    }
                    NeuroTx::Ach => {
                        r.ach_gate = (r.ach_gate + 0.01 * s.w.abs()).clamp(0.0, 1.0);
                    }
                }
            }
        }
        sum_v += r.neurons[idx].s.v;
    }

    (spikes, sum_v / n as f32)
}
