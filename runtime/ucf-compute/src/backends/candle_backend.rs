use crate::capabilities::SaeExtractor;
use crate::feature_extractor::{SaeInput, SaeOutput, SmallNotes};
use crate::world_model::StageQuality;
use crate::{ComputeBudget, ComputeError};

#[derive(Debug, Clone, Copy)]
pub struct CandleSaeExtractor {
    _seed: u64,
}

impl CandleSaeExtractor {
    pub fn new(seed: u64) -> Self {
        Self { _seed: seed }
    }
}

impl Default for CandleSaeExtractor {
    fn default() -> Self {
        Self::new(ComputeBudget::default().seed)
    }
}

impl SaeExtractor for CandleSaeExtractor {
    fn name(&self) -> &'static str {
        "candle_sae_v0"
    }

    fn extract(
        &self,
        _input: &SaeInput,
        _budget: ComputeBudget,
    ) -> Result<SaeOutput, ComputeError> {
        Ok(SaeOutput {
            spikes: Vec::new(),
            spike_count: 0,
            sparsity: 1.0,
            energy: 0.0,
            spikes_digest: [0; 32],
            quality: StageQuality::Unavailable,
            notes: SmallNotes(vec!["unavailable:candle_sae_not_implemented".to_string()]),
        })
    }
}
