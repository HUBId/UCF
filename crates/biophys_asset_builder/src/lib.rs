#![forbid(unsafe_code)]

use biophys_assets::{ChannelParamsSet, ConnectivityGraph, MorphologySet, SynapseParamsSet};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("asset bundle missing manifest")]
    MissingManifest,
    #[error("missing asset digest for {kind}")]
    MissingAssetDigest { kind: &'static str },
    #[error("asset digest mismatch for {kind}")]
    AssetDigestMismatch { kind: &'static str },
    #[error("asset count exceeds bounds: {label} {count} > {max}")]
    BoundsExceeded {
        label: &'static str,
        count: usize,
        max: usize,
    },
    #[error("invalid digest length for {label}: {len}")]
    InvalidDigestLength { label: &'static str, len: usize },
    #[error("unknown neuron id {neuron_id}")]
    UnknownNeuron { neuron_id: u32 },
    #[error("missing channel params for neuron {neuron_id} compartment {comp_id}")]
    MissingChannelParams { neuron_id: u32, comp_id: u32 },
    #[error("missing synapse params id {syn_param_id}")]
    MissingSynapseParams { syn_param_id: u32 },
    #[error("invalid asset data: {message}")]
    InvalidAssetData { message: String },
    #[error("rehydration error: {0}")]
    Rehydration(#[from] asset_rehydration::RehydrationError),
}

pub trait CircuitBuilderFromAssets: Sized {
    fn build_from_assets(
        morph: &MorphologySet,
        chan: &ChannelParamsSet,
        syn: &SynapseParamsSet,
        conn: &ConnectivityGraph,
    ) -> Result<Self, Error>;
}
