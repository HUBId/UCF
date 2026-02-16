use crate::capabilities::FeatureExtractor;
use crate::feature_extractor::SaeOutput;
use crate::world_model::WorldModelOutput;
use crate::{ComputeBudget, ComputeError, ComputeInput};

#[derive(Debug, Clone, Copy)]
pub struct BurnFeatureExtractor {
    _seed: u64,
}

impl BurnFeatureExtractor {
    pub fn new(seed: u64) -> Self {
        Self { _seed: seed }
    }
}

impl FeatureExtractor for BurnFeatureExtractor {
    fn name(&self) -> &'static str {
        "burn_feature_extractor_v0"
    }

    fn extract(
        &self,
        _input: &ComputeInput,
        _world: &WorldModelOutput,
        _budget: ComputeBudget,
    ) -> Result<SaeOutput, ComputeError> {
        Err(ComputeError::NotImplemented)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capabilities::WorldModelPredictor;
    use crate::world_model::{obs_features_from_context, MockJepaPredictor, WorldModelInput};
    use crate::FrameId;

    #[test]
    fn explicit_not_implemented_error() {
        let extractor = BurnFeatureExtractor::new(7);
        let input = ComputeInput {
            frame_id: FrameId(1),
            t: 3,
            context_digest: [2_u8; 32],
        };
        let mut world_model = MockJepaPredictor::default();
        let world = world_model
            .step(
                &WorldModelInput {
                    t: input.t,
                    context_digest: input.context_digest,
                    obs_features: obs_features_from_context(input.context_digest),
                    seed: 5,
                },
                ComputeBudget::default(),
            )
            .expect("world model should work");

        let err = extractor
            .extract(&input, &world, ComputeBudget::default())
            .expect_err("burn skeleton returns explicit error");
        assert!(matches!(err, ComputeError::NotImplemented));
    }
}
