mod cde;
mod field;
mod hh;
mod hpa;
mod iit;
mod microcircuit;
mod modulation;
mod nsr;
mod ode;
mod phase;
mod spike;
mod synapse;

pub use cde::{CausalGraph, Counterfactual, Edge, Hypothesis, Intervention, VarId};
pub use field::{
    apply_coherence_feedback, FieldEvent, FieldEventKind, FieldUpdateCfg, NeuromodulatorField,
    Unit01,
};
pub use hh::{HHNeuron, HHState, HhStepIn, HhStepOut};
pub use hpa::{cortisol_unit, hpa_step, HpaCfg, HpaState};
pub use iit::{classify, compute_integration, CoherenceState, IITCfg, IITInputs, IITState};
pub use microcircuit::{MicroStepOut, Microcircuit};
pub use modulation::{modulate_hh, summarize, HhParams, ModulationCfg};
pub use nsr::{verify_graph, RuleCfg, VerifyVerdict};
pub use ode::{clamp01, prod_clear_step, step_euler, Integrator};
pub use phase::{couple_pair, osc_step, phase_lock, wrap_phase, Osc, PhaseCfg, TAU};
pub use spike::{spikes_from_ids, ttfs_phase, SpikeCodecCfg, SpikeEvent};
pub use synapse::{stp_step, NeuronId, NoPlasticity, PlasticityRule, SynKind, Synapse};

#[cfg(test)]
mod tests;
