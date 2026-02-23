use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use candle_core::{Device, Tensor};
use sha2::Digest;

use super::llm_toy::{ToyWeights, TOY_EMBED_DIM};
use super::{FinishReason, LlmInference, LlmOutputClass, LlmRequest, LlmResponse, LlmStatus};
use crate::candle_weights::{load_safetensors, DType, DimExpr, TensorSpec, WeightErr, WeightSpec};
use crate::model_store::{ModelSlot, VerifiedModelSlot};
use crate::{ComputeBudget, ComputeError};

#[derive(Debug, Clone)]
pub struct CandleLlmBackend {
    model: Arc<CandleLlmModel>,
}

#[derive(Debug, Clone)]
struct CandleLlmModel {
    mode: CandleLlmMode,
    vocab: Vec<String>,
}

#[derive(Debug, Clone)]
enum CandleLlmMode {
    Toy(ToyWeights),
    V1 {
        tok_emb: Vec<f32>,
        lm_head: Vec<f32>,
        d_model: usize,
    },
}

const LLM_V1_REQ: [TensorSpec; 2] = [
    TensorSpec {
        name: "tok_emb",
        shape: &[DimExpr::Fixed(32), DimExpr::Fixed(64)],
        dtype: DType::F32,
    },
    TensorSpec {
        name: "lm_head",
        shape: &[DimExpr::Fixed(64), DimExpr::Fixed(32)],
        dtype: DType::F32,
    },
];

impl CandleLlmBackend {
    pub fn from_fixture() -> Result<Self, ComputeError> {
        let weights = ToyWeights::load()?;
        Ok(Self {
            model: Arc::new(CandleLlmModel {
                vocab: weights.vocab.clone(),
                mode: CandleLlmMode::Toy(weights),
            }),
        })
    }

    pub fn from_verified_slot(
        verified: &VerifiedModelSlot,
        tokenizer_path: &Path,
        tokenizer_sha256: [u8; 32],
    ) -> Result<Self, ComputeError> {
        let mut bytes = Vec::new();
        File::open(&verified.path)
            .and_then(|mut f| f.read_to_end(&mut bytes))
            .map_err(|e| ComputeError::InvalidInput {
                reason: format!("llm slot read failed: {e}"),
            })?;

        let loaded = load_safetensors(ModelSlot::Llm, &bytes, &llm_weight_spec())
            .map_err(|e| map_weight_err("llm weights invalid", e))?;
        let tok_emb = loaded
            .tensors
            .get("tok_emb")
            .ok_or_else(|| ComputeError::InvalidInput {
                reason: "missing tok_emb tensor".to_string(),
            })?
            .to_vec1::<f32>()
            .map_err(|e| ComputeError::InvalidInput {
                reason: format!("tok_emb decode failed: {e}"),
            })?;
        let lm_head = loaded
            .tensors
            .get("lm_head")
            .ok_or_else(|| ComputeError::InvalidInput {
                reason: "missing lm_head tensor".to_string(),
            })?
            .to_vec1::<f32>()
            .map_err(|e| ComputeError::InvalidInput {
                reason: format!("lm_head decode failed: {e}"),
            })?;

        let vocab = load_tokenizer_vocab(tokenizer_path, tokenizer_sha256)?;
        let d_model = 64;
        if tok_emb.len() != vocab.len() * d_model || lm_head.len() != d_model * vocab.len() {
            return Err(ComputeError::InvalidInput {
                reason: "llm tensor dims do not match tokenizer vocab".to_string(),
            });
        }

        Ok(Self {
            model: Arc::new(CandleLlmModel {
                mode: CandleLlmMode::V1 {
                    tok_emb,
                    lm_head,
                    d_model,
                },
                vocab,
            }),
        })
    }

    #[cfg(test)]
    pub fn from_weights(weights: ToyWeights) -> Self {
        Self {
            model: Arc::new(CandleLlmModel {
                vocab: weights.vocab.clone(),
                mode: CandleLlmMode::Toy(weights),
            }),
        }
    }

    fn vocab_len(&self) -> usize {
        self.model.vocab.len()
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
        match &self.model.mode {
            CandleLlmMode::Toy(weights) => {
                let row_start = token * TOY_EMBED_DIM;
                let x = &weights.embed[row_start..row_start + TOY_EMBED_DIM];

                let device = Device::Cpu;
                let x = Tensor::from_slice(x, TOY_EMBED_DIM, &device)
                    .and_then(|t| t.reshape((1, TOY_EMBED_DIM)))
                    .map_err(|e| ComputeError::Internal {
                        reason: e.to_string(),
                    })?;
                let w = Tensor::from_slice(
                    &weights.linear_w,
                    (self.vocab_len(), TOY_EMBED_DIM),
                    &device,
                )
                .map_err(|e| ComputeError::Internal {
                    reason: e.to_string(),
                })?;
                let b = Tensor::from_slice(&weights.linear_b, self.vocab_len(), &device).map_err(
                    |e| ComputeError::Internal {
                        reason: e.to_string(),
                    },
                )?;

                w.broadcast_mul(&x)
                    .and_then(|t| t.sum(1))
                    .and_then(|t| t.broadcast_add(&b))
                    .and_then(|t| t.to_vec1::<f32>())
                    .map_err(|e| ComputeError::Internal {
                        reason: e.to_string(),
                    })
            }
            CandleLlmMode::V1 {
                tok_emb,
                lm_head,
                d_model,
            } => {
                let x = &tok_emb[token * d_model..(token + 1) * d_model];
                let mut logits = vec![0.0_f32; self.vocab_len()];
                for v in 0..self.vocab_len() {
                    let mut sum = 0.0_f32;
                    for i in 0..*d_model {
                        sum += x[i] * lm_head[i * self.vocab_len() + v];
                    }
                    if !sum.is_finite() {
                        return Err(ComputeError::Internal {
                            reason: "llm logits contain NaN/Inf".to_string(),
                        });
                    }
                    logits[v] = (sum * 1_000_000.0).round() / 1_000_000.0;
                }
                Ok(logits)
            }
        }
    }
}

impl LlmInference for CandleLlmBackend {
    fn name(&self) -> &'static str {
        match &self.model.mode {
            CandleLlmMode::Toy(_) => "candle:toy_v1",
            CandleLlmMode::V1 { .. } => "candle:llm_v1",
        }
    }

    fn infer(&self, req: &LlmRequest, budget: ComputeBudget) -> Result<LlmResponse, ComputeError> {
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
        if req.sampling_enabled || req.temperature > 0.0 || req.top_p < 1.0 {
            return Err(ComputeError::SamplingDisabled {
                code: "SAMPLING_DISABLED",
            });
        }
        let mut text = String::new();
        if req.output_class == LlmOutputClass::SafeText {
            text.push_str("- Summary:\n- ");
        } else if req.output_class == LlmOutputClass::Code {
            text.push_str("```text\n");
        }

        let start = Instant::now();
        let mut current = Self::token_from_prompt(&req, self.vocab_len());
        let mut emitted = 0_u32;
        let mut truncated = false;

        while emitted < req.max_tokens {
            let elapsed = start.elapsed().as_micros() as u64;
            if elapsed > budget.hard_timeout_micros {
                return Err(ComputeError::BudgetExceeded {
                    stage: "llm/timeout",
                    elapsed_micros: elapsed,
                    limit_micros: budget.hard_timeout_micros,
                });
            }
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
            let token = &self.model.vocab[best_idx];
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

pub fn llm_weight_spec() -> WeightSpec {
    WeightSpec {
        slot: ModelSlot::Llm,
        tensors: &LLM_V1_REQ,
        optional: &[],
        max_bytes: 2 * 1024 * 1024,
        bindings: BTreeMap::new(),
    }
}

fn map_weight_err(prefix: &str, err: WeightErr) -> ComputeError {
    ComputeError::InvalidInput {
        reason: format!("{prefix}: {err:?}"),
    }
}

fn load_tokenizer_vocab(
    path: &Path,
    expected_sha256: [u8; 32],
) -> Result<Vec<String>, ComputeError> {
    let mut bytes = Vec::new();
    File::open(path)
        .and_then(|mut f| f.read_to_end(&mut bytes))
        .map_err(|e| ComputeError::InvalidInput {
            reason: format!("tokenizer read failed: {e}"),
        })?;
    let found: [u8; 32] = sha2::Sha256::digest(&bytes).into();
    if found != expected_sha256 {
        return Err(ComputeError::InvalidInput {
            reason: "tokenizer hash mismatch".to_string(),
        });
    }
    serde_json::from_slice::<Vec<String>>(&bytes).map_err(|e| ComputeError::InvalidInput {
        reason: format!("tokenizer vocab invalid: {e}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn tokenizer_hash_guard_rejects_invalid_hash() {
        let path = PathBuf::from("runtime/ucf-compute/fixtures/llm_v1_tiny_vocab.json");
        let bad = [0x11_u8; 32];
        let err = load_tokenizer_vocab(&path, bad).expect_err("must reject wrong hash");
        assert!(matches!(err, ComputeError::InvalidInput { .. }));
    }

    #[test]
    fn llm_weight_spec_has_expected_tensors() {
        let spec = llm_weight_spec();
        assert_eq!(spec.slot, ModelSlot::Llm);
        assert_eq!(spec.tensors.len(), 2);
        assert_eq!(spec.tensors[0].name, "tok_emb");
        assert_eq!(spec.tensors[1].name, "lm_head");
    }
}
