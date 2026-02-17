use std::sync::Arc;

use crate::capabilities::{SaeExtractor, WorldModelPredictor};
use crate::feature_extractor::{SaeInput, SaeOutput, ToySaeExtractor};
use crate::ssm::{SsmInput, SsmKernel, SsmOutput, ToySsmKernel};
use crate::world_model::{MockJepaPredictor, WorldModelInput, WorldModelOutput};
use crate::{ComputeBudget, ComputeError};

#[derive(Clone)]
pub struct BurnSaeExtractor {
    inner: Arc<dyn SaeExtractor + Send + Sync>,
}

impl BurnSaeExtractor {
    pub fn new(_model_hash: [u8; 32]) -> Self {
        Self {
            inner: Arc::new(ToySaeExtractor::default()),
        }
    }
}

impl SaeExtractor for BurnSaeExtractor {
    fn name(&self) -> &'static str {
        "burn_sae_v1"
    }

    fn extract(&self, input: &SaeInput, budget: ComputeBudget) -> Result<SaeOutput, ComputeError> {
        self.inner.extract(input, budget)
    }
}

#[derive(Debug, Default)]
pub struct BurnWorldPredictor {
    inner: MockJepaPredictor,
}

impl BurnWorldPredictor {
    pub fn new(_model_hash: [u8; 32]) -> Self {
        Self {
            inner: MockJepaPredictor::default(),
        }
    }
}

impl WorldModelPredictor for BurnWorldPredictor {
    fn name(&self) -> &'static str {
        "burn_world_jepa_v1"
    }

    fn step(
        &mut self,
        input: &WorldModelInput,
        budget: ComputeBudget,
    ) -> Result<WorldModelOutput, ComputeError> {
        self.inner.step(input, budget)
    }
}

#[derive(Debug, Default)]
pub struct BurnSsmKernel {
    inner: ToySsmKernel,
}

impl BurnSsmKernel {
    pub fn new(_model_hash: [u8; 32]) -> Self {
        Self {
            inner: ToySsmKernel::default(),
        }
    }
}

impl SsmKernel for BurnSsmKernel {
    fn name(&self) -> &'static str {
        "burn_ssm_selective_scan_v1"
    }

    fn step(&mut self, input: &SsmInput, budget: ComputeBudget) -> Result<SsmOutput, ComputeError> {
        self.inner.step(input, budget)
    }
}
