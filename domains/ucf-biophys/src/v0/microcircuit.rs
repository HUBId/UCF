use crate::v0::field::NeuromodulatorField;
use crate::v0::hh::{HHNeuron, HhStepIn};
use crate::v0::modulation::{modulate_hh, HhParams, ModulationCfg};
use crate::v0::synapse::{stp_step, NeuronId, SynKind, Synapse};

pub struct Microcircuit {
    pub neurons: Vec<HHNeuron>,
    pub outgoing: Vec<Vec<usize>>,
    pub synapses: Vec<Synapse>,
    pub t_ms: u64,
    prev_spikes: Vec<bool>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MicroStepOut {
    pub spikes: Vec<NeuronId>,
    pub avg_v: f32,
}

impl Microcircuit {
    pub fn new_ring(n: usize) -> Microcircuit {
        let mut neurons = vec![HHNeuron::default(); n];
        if let Some(first) = neurons.first_mut() {
            first.state.v_mv = -52.0;
        }
        let mut outgoing = vec![Vec::new(); n];
        let mut synapses = Vec::with_capacity(n * 2);

        for (i, edges) in outgoing.iter_mut().enumerate().take(n) {
            let post_exc = (i + 1) % n;
            let exc_idx = synapses.len();
            synapses.push(Synapse {
                pre: i as NeuronId,
                post: post_exc as NeuronId,
                kind: SynKind::Excitatory,
                weight: 0.5,
                delay_ms: 2,
                stp_u: 0.2,
                stp_x: 1.0,
            });
            edges.push(exc_idx);

            let post_inh = (i + 2) % n;
            let inh_idx = synapses.len();
            synapses.push(Synapse {
                pre: i as NeuronId,
                post: post_inh as NeuronId,
                kind: SynKind::Inhibitory,
                weight: 0.3,
                delay_ms: 3,
                stp_u: 0.2,
                stp_x: 1.0,
            });
            edges.push(inh_idx);
        }

        Microcircuit {
            neurons,
            outgoing,
            synapses,
            t_ms: 0,
            prev_spikes: vec![false; n],
        }
    }

    pub fn step(&mut self, field: NeuromodulatorField, dt_s: f32) -> MicroStepOut {
        let hh_params = modulate_hh(HhParams::default(), field, ModulationCfg::default());
        let mut incoming_currents = vec![0.0_f32; self.neurons.len()];

        for syn in &self.synapses {
            if self.prev_spikes[syn.pre as usize] {
                incoming_currents[syn.post as usize] +=
                    syn.effective_weight() * syn.stp_u * syn.stp_x;
            }
        }

        let ext_drive = (field.glutamate.get() - field.gaba.get()).clamp(-1.0, 1.0);
        let threshold_mv = -50.0 + hh_params.threshold_shift_mv;
        let mut spikes = Vec::new();
        let mut avg_v_sum = 0.0;
        let mut next_prev_spikes = vec![false; self.neurons.len()];

        for (idx, neuron) in self.neurons.iter_mut().enumerate() {
            neuron.params = hh_params;
            let out = neuron.step_stub(HhStepIn {
                i_ext: ext_drive,
                syn_i: incoming_currents[idx],
                dt_s,
                threshold_mv,
            });

            if out.spiked {
                spikes.push(idx as NeuronId);
                next_prev_spikes[idx] = true;
            }
            avg_v_sum += out.v_mv;
        }

        for syn in &mut self.synapses {
            stp_step(syn, dt_s);
        }

        self.prev_spikes = next_prev_spikes;
        self.t_ms = self.t_ms.saturating_add((dt_s * 1000.0) as u64);

        let avg_v = if self.neurons.is_empty() {
            0.0
        } else {
            avg_v_sum / self.neurons.len() as f32
        };

        MicroStepOut { spikes, avg_v }
    }
}
