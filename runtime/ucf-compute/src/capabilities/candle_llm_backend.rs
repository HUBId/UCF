use std::sync::Arc;

use candle_core::{Device, Tensor};
use sha2::Digest;

use super::llm_toy::{ToyWeights, TOY_EMBED_DIM};
use super::{FinishReason, LlmInference, LlmOutputClass, LlmRequest, LlmResponse, LlmStatus};
use crate::{ComputeBudget, ComputeError};

#[derive(Debug, Clone)]
pub struct CandleLlmBackend {
    weights: Arc<ToyWeights>,
}

impl CandleLlmBackend {
    pub fn from_fixture() -> Result<Self, ComputeError> {
        Ok(Self {
            weights: Arc::new(ToyWeights::load()?),
        })
    }

    #[cfg(test)]
    pub fn from_weights(weights: ToyWeights) -> Self {
        Self {
            weights: Arc::new(weights),
        }
    }

    fn vocab_len(&self) -> usize {
        self.weights.vocab.len()
    }

    fn token_from_prompt(req: &LlmRequest, vocab_len: usize) -> usize {
        let mut digest = [0_u8; 32];
        digest.copy_from_slice(&sha2::Sha256::digest(req.prompt.as_bytes()));
        let mixed = digest[(req.candidate_id as usize) % 32] as usize
            ^ req.context_digest[(req.seed as usize) % 32] as usize
            ^ (req.seed as usize);
        mixed % vocab_len
    }

    fn score_token(&self, token: usize) -> Result<Vec<f32>, ComputeError> {
        let row_start = token * TOY_EMBED_DIM;
        let x = &self.weights.embed[row_start..row_start + TOY_EMBED_DIM];

        let device = Device::Cpu;
        let x = Tensor::from_slice(x, TOY_EMBED_DIM, &device)
            .and_then(|t| t.reshape((1, TOY_EMBED_DIM)))
            .map_err(|e| ComputeError::Internal {
                reason: e.to_string(),
            })?;
        let w = Tensor::from_slice(
            &self.weights.linear_w,
            (self.vocab_len(), TOY_EMBED_DIM),
            &device,
        )
        .map_err(|e| ComputeError::Internal {
            reason: e.to_string(),
        })?;
        let b =
            Tensor::from_slice(&self.weights.linear_b, self.vocab_len(), &device).map_err(|e| {
                ComputeError::Internal {
                    reason: e.to_string(),
                }
            })?;

        w.broadcast_mul(&x)
            .and_then(|t| t.sum(1))
            .and_then(|t| t.broadcast_add(&b))
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| ComputeError::Internal {
                reason: e.to_string(),
            })
    }
}

impl LlmInference for CandleLlmBackend {
    fn name(&self) -> &'static str {
        "candle:toy_v1"
    }

    fn infer(&self, req: &LlmRequest, _budget: ComputeBudget) -> Result<LlmResponse, ComputeError> {
        if matches!(
            req.output_class,
            LlmOutputClass::ExternalIo | LlmOutputClass::ExecIntent
        ) {
            return Ok(LlmResponse::new(
                LlmStatus::Refused,
                "refused: output class requires tool-gated plan".to_string(),
                0,
                FinishReason::PolicyRefusal,
            ));
        }

        let req = req.clone().bounded();
        let mut text = String::new();
        if req.output_class == LlmOutputClass::SafeText {
            text.push_str("- Summary:\n- ");
        } else if req.output_class == LlmOutputClass::Code {
            text.push_str("```text\n");
        }

        let mut current = Self::token_from_prompt(&req, self.vocab_len());
        let mut emitted = 0_u32;
        let mut truncated = false;

        while emitted < req.max_tokens {
            let scores = self.score_token(current)?;
            let mut best_idx = 0usize;
            let mut best = f32::NEG_INFINITY;
            for (idx, val) in scores.into_iter().enumerate() {
                let q = (val * 1_000_000.0).round() / 1_000_000.0;
                if q > best {
                    best = q;
                    best_idx = idx;
                }
            }
            let token = &self.weights.vocab[best_idx];
            if text.len().saturating_add(token.len()).saturating_add(1) > super::MAX_LLM_TEXT_BYTES
            {
                truncated = true;
                break;
            }
            text.push_str(token);
            text.push(' ');
            emitted += 1;
            current = best_idx;
        }

        if req.output_class == LlmOutputClass::SafeText {
            text.push_str("\n- Next: verify policy and continue safely.");
        } else if req.output_class == LlmOutputClass::Code {
            text.push_str("\n```\n");
        }

        Ok(if truncated {
            LlmResponse::new(LlmStatus::Truncated, text, emitted, FinishReason::Length)
        } else {
            LlmResponse::new(LlmStatus::Ok, text, emitted, FinishReason::Stop)
        })
    }
}
