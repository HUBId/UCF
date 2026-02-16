use super::{LlmInference, LlmRequest, LlmResponse};
use crate::{ComputeBudget, ComputeError};

#[derive(Debug, Default)]
pub struct BurnLlmBackend;

impl LlmInference for BurnLlmBackend {
    fn name(&self) -> &'static str {
        "burn:toy_v1"
    }

    fn infer(
        &self,
        _req: &LlmRequest,
        _budget: ComputeBudget,
    ) -> Result<LlmResponse, ComputeError> {
        Err(ComputeError::NotImplemented)
    }
}
