mod bridge;
mod encoder;
#[cfg(test)]
mod tests;
mod types;

pub use bridge::to_brainbus;
pub use encoder::{encode, encode_ttfsp, snn_emit, FeatureEvent, SnnEncodeCfg};
pub use types::{
    SnnCfg, SnnOut, SnnSpike, SpikeChan, SpikeDst, SpikeEvent, SpikePayload, SpikeSrc, SpikeTimeMs,
    TtfsMs,
};
