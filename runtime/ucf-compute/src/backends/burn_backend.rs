use crate::capabilities::SaeExtractor;
use crate::feature_extractor::{SaeInput, SaeOutput};
use crate::{ComputeBudget, ComputeError};

#[derive(Debug, Clone, Copy)]
pub struct BurnSaeExtractor {
    _seed: u64,
}

impl BurnSaeExtractor {
    pub fn new(seed: u64) -> Self {
        Self { _seed: seed }
    }
}

impl SaeExtractor for BurnSaeExtractor {
    fn name(&self) -> &'static str {
        "burn_feature_extractor_v0"
    }

    fn extract(
        &self,
        _input: &SaeInput,
        _budget: ComputeBudget,
    ) -> Result<SaeOutput, ComputeError> {
        Err(ComputeError::NotImplemented)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_not_implemented_error() {
        let extractor = BurnSaeExtractor::new(7);
        let input = SaeInput {
            t: 3,
            context_features: [0.0; crate::feature_extractor::SAE_INPUT_DIM],
            world_state_digest: None,
            seed: 5,
            evidence_chain_digest: [0; 32],
        };

        let err = extractor
            .extract(&input, ComputeBudget::default())
            .expect_err("burn skeleton returns explicit error");
        assert!(matches!(err, ComputeError::NotImplemented));
    }
}
