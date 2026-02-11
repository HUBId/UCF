use crate::feature_extractor::SaeOutput;
use crate::ssm::{SsmOutput, SsmState};
use crate::world_model::{WorldModelOutput, WorldState};
use crate::{ComputeBudget, ComputeError, ComputeInput, MAX_NOTE_LEN};

pub trait WorldModelPredictor: Send + Sync {
    fn name(&self) -> &'static str;
    fn init_state(&self, input: &ComputeInput, seed: u64) -> WorldState;
    fn predict(
        &self,
        state: &WorldState,
        input: &ComputeInput,
        budget: ComputeBudget,
    ) -> Result<WorldModelOutput, ComputeError>;
}

pub trait FeatureExtractor: Send + Sync {
    fn name(&self) -> &'static str;
    fn extract(
        &self,
        input: &ComputeInput,
        world: &WorldModelOutput,
        budget: ComputeBudget,
    ) -> Result<SaeOutput, ComputeError>;
}

pub trait WorkingMemoryModel: Send + Sync {
    fn name(&self) -> &'static str;
    fn init(&self, input: &ComputeInput, seed: u64) -> SsmState;
    fn step(
        &self,
        state: &SsmState,
        sae: &SaeOutput,
        world: &WorldModelOutput,
        budget: ComputeBudget,
    ) -> Result<SsmOutput, ComputeError>;
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LlmOutput {
    pub text: String,
    pub confidence: f32,
}

impl LlmOutput {
    pub fn bounded(mut self) -> Self {
        self.text = self.text.chars().take(MAX_NOTE_LEN * 16).collect();
        self.confidence = self.confidence.clamp(0.0, 1.0);
        self
    }
}

pub trait LlmInference: Send + Sync {
    fn name(&self) -> &'static str;
    fn infer(&self, prompt: &str, budget: ComputeBudget) -> Result<LlmOutput, ComputeError>;
}
