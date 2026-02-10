mod field;
mod hh;
mod hpa;
mod microcircuit;
mod modulation;
mod ode;
mod synapse;

pub use field::{FieldEvent, FieldEventKind, FieldUpdateCfg, NeuromodulatorField, Unit01};
pub use hh::{HHNeuron, HHState, HhStepIn, HhStepOut};
pub use hpa::{cortisol_unit, hpa_step, HpaCfg, HpaState};
pub use microcircuit::{MicroStepOut, Microcircuit};
pub use modulation::{modulate_hh, summarize, HhParams, ModulationCfg};
pub use ode::{clamp01, prod_clear_step, step_euler, Integrator};
pub use synapse::{stp_step, NeuronId, NoPlasticity, PlasticityRule, SynKind, Synapse};

#[cfg(test)]
mod tests;
