mod field;
mod modulation;

pub use field::{FieldEvent, FieldEventKind, FieldUpdateCfg, NeuromodulatorField, Unit01};
pub use modulation::{modulate_hh, summarize, HhParams, ModulationCfg};

#[cfg(test)]
mod tests;
