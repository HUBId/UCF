mod field;
mod rules;
mod scheduler;

pub use field::NeuromodulatorField;
pub use rules::compute_delta;
pub use scheduler::{NeuromodInputs, NeuromodScheduler};

#[cfg(test)]
mod tests;
