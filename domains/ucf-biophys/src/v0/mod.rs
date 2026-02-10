mod field;
mod hpa;
mod modulation;
mod ode;

pub use field::{FieldEvent, FieldEventKind, FieldUpdateCfg, NeuromodulatorField, Unit01};
pub use hpa::{cortisol_unit, hpa_step, HpaCfg, HpaState};
pub use modulation::{modulate_hh, summarize, HhParams, ModulationCfg};
pub use ode::{clamp01, prod_clear_step, step_euler, Integrator};

#[cfg(test)]
mod tests;
