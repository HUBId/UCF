mod bridge;
mod encoder;
#[cfg(test)]
mod tests;
mod types;

pub use bridge::to_brainbus;
pub use encoder::{encode, FeatureEvent, SnnEncodeCfg};
pub use types::{SnnSpike, SpikeChan, SpikePayload, SpikeTimeMs, TtfsMs};
