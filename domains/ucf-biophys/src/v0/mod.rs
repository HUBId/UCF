mod cde;
mod field;
mod hh;
mod hormone;
mod hpa;
mod iit;
mod microcircuit;
mod modulation;
mod neuro;
mod nsr;
mod ode;
mod onn;
mod phase;
mod snn;
mod spike;
mod ssm;
mod synapse;

pub use cde::{CausalGraph, Counterfactual, Edge, Hypothesis, Intervention, VarId};
pub use field::{
    apply_coherence_feedback, FieldEvent, FieldEventKind, FieldUpdateCfg, NeuromodulatorField,
    Unit01,
};
pub use hh::{HHNeuron, HHState, HhStepIn, HhStepOut};
pub use hormone::{
    hormone_step, map_modulation, GatingModulation, HormoneCfg, HormoneInput, HormoneState,
    HormoneStateSummary, HormoneStepOutput,
};
pub use hpa::{cortisol_unit, hpa_step, HpaCfg, HpaState};
pub use iit::{classify, compute_integration, CoherenceState, IITCfg, IITInputs, IITState};
pub use microcircuit::{MicroStepOut, Microcircuit};
pub use modulation::{modulate_hh, summarize, HhParams, ModulationCfg};
pub use neuro::{
    neuro_step, BiophysModulation, NeuroCfg, NeuroInput, NeuroSpike, NeuroSpikeBatch,
    NeuroStateSummary, NeuroStepOutput, NeuronPopState, MAX_SPIKES_PER_TICK, NEURON_COUNT,
};
pub use nsr::{verify_graph, RuleCfg, VerifyVerdict};
pub use ode::{clamp01, prod_clear_step, step_euler, Integrator};
pub use onn::{
    ensure_osc, phase_bin, phase_diff_abs, step as onn_step, OnnCfg, OnnOut, OnnState, OscId,
};
pub use phase::{couple_pair, osc_step, phase_lock, wrap_phase, Osc, PhaseCfg, TAU};
pub use snn::{ttfs_from_strength, EventBus, SpikeEvent as SnnSpikeEvent, SpikeId, SpikeKind};
pub use spike::{spikes_from_ids, ttfs_phase, SpikeCodecCfg, SpikeEvent};
pub use ssm::{mix_inputs, step as ssm_step, SsmCfg, SsmInputs, SsmOut, SsmState, D as SSM_D};
pub use synapse::{stp_step, NeuronId, NoPlasticity, PlasticityRule, SynKind, Synapse};

#[cfg(test)]
mod tests;
