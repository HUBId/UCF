use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::feature_extractor::SaeOutput;
use crate::ssm::{SsmOutput, SsmState};
use crate::world_model::{WorldModelInput, WorldModelOutput};
use crate::{ComputeBudget, ComputeError, ComputeInput, MAX_NOTE_LEN};

#[cfg(any(feature = "compute-burn", feature = "llm-burn"))]
mod burn_llm_backend;
#[cfg(any(feature = "compute-candle", feature = "llm-candle"))]
mod candle_llm_backend;
#[cfg(any(test, feature = "compute-candle", feature = "llm-candle"))]
mod llm_toy;

#[cfg(any(feature = "compute-burn", feature = "llm-burn"))]
use burn_llm_backend::BurnLlmBackend;
#[cfg(any(feature = "compute-candle", feature = "llm-candle"))]
use candle_llm_backend::CandleLlmBackend;

const MAX_LLM_PROMPT_BYTES: usize = 8 * 1024;
pub(crate) const MAX_LLM_TEXT_BYTES: usize = 16 * 1024;
const MAX_LLM_TOKENS: u32 = 1024;
const VOCAB: [&str; 32] = [
    "safe",
    "stable",
    "bounded",
    "audit",
    "context",
    "digest",
    "signal",
    "coherence",
    "risk",
    "confidence",
    "summary",
    "plan",
    "step",
    "observe",
    "verify",
    "policy",
    "intent",
    "evidence",
    "deterministic",
    "offline",
    "local",
    "status",
    "fallback",
    "note",
    "review",
    "check",
    "state",
    "frame",
    "contract",
    "schema",
    "result",
    "trace",
];

pub trait WorldModelPredictor: Send + Sync {
    fn name(&self) -> &'static str;
    fn step(
        &mut self,
        input: &WorldModelInput,
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
    fn infer(&self, req: &LlmRequest, budget: ComputeBudget) -> Result<LlmResponse, ComputeError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum LlmBackendKind {
    #[default]
    Stub,
    Candle,
    Burn,
}

impl LlmBackendKind {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "stub" => Some(Self::Stub),
            "candle" => Some(Self::Candle),
            "burn" => Some(Self::Burn),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmOutputClass {
    SafeText,
    Code,
    ExternalIo,
    ExecIntent,
    Sensitive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmStatus {
    Ok,
    Truncated,
    Refused,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    PolicyRefusal,
    Error,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlmRequest {
    pub schema_version: u16,
    pub t: u64,
    pub decision_id: u64,
    pub candidate_id: u16,
    pub output_class: LlmOutputClass,
    pub prompt: String,
    pub context_digest: [u8; 32],
    pub evidence_chain_digest: [u8; 32],
    pub seed: u64,
    pub max_tokens: u32,
    pub temperature: f32,
}

impl LlmRequest {
    pub fn bounded(mut self) -> Self {
        self.schema_version = self.schema_version.max(1);
        self.prompt = self.prompt.chars().take(MAX_LLM_PROMPT_BYTES).collect();
        self.max_tokens = self.max_tokens.clamp(1, MAX_LLM_TOKENS);
        self.temperature = self.temperature.clamp(0.0, 2.0);
        self
    }

    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.schema_version.to_le_bytes());
        hasher.update(self.t.to_le_bytes());
        hasher.update(self.decision_id.to_le_bytes());
        hasher.update(self.candidate_id.to_le_bytes());
        hasher.update([self.output_class as u8]);
        hasher.update((self.prompt.len() as u32).to_le_bytes());
        hasher.update(self.prompt.as_bytes());
        hasher.update(self.context_digest);
        hasher.update(self.evidence_chain_digest);
        hasher.update(self.seed.to_le_bytes());
        hasher.update(self.max_tokens.to_le_bytes());
        hasher.update(self.temperature.to_bits().to_le_bytes());
        hasher.finalize().into()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmResponse {
    pub status: LlmStatus,
    pub text: String,
    pub token_count: u32,
    pub finish_reason: FinishReason,
    pub digest: [u8; 32],
}

impl LlmResponse {
    pub fn new(
        status: LlmStatus,
        text: String,
        token_count: u32,
        finish_reason: FinishReason,
    ) -> Self {
        let mut response = Self {
            status,
            text: text.chars().take(MAX_LLM_TEXT_BYTES).collect(),
            token_count: token_count.min(MAX_LLM_TOKENS),
            finish_reason,
            digest: [0; 32],
        };
        response.digest = response.compute_digest();
        response
    }

    fn compute_digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update([self.status as u8]);
        hasher.update((self.text.len() as u32).to_le_bytes());
        hasher.update(self.text.as_bytes());
        hasher.update(self.token_count.to_le_bytes());
        hasher.update([self.finish_reason as u8]);
        hasher.finalize().into()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlmBackendConfig {
    pub kind: LlmBackendKind,
    pub seed: u64,
    pub max_tokens: u32,
}

impl Default for LlmBackendConfig {
    fn default() -> Self {
        Self {
            kind: LlmBackendKind::Stub,
            seed: 0x5eed_u64,
            max_tokens: 128,
        }
    }
}

impl LlmBackendConfig {
    pub fn from_env() -> Result<Self, ComputeError> {
        let mut cfg = Self::default();
        if let Ok(value) = std::env::var("UCF_LLM_BACKEND") {
            cfg.kind = LlmBackendKind::parse(&value).ok_or_else(|| ComputeError::InvalidInput {
                reason: format!("unsupported UCF_LLM_BACKEND={value}"),
            })?;
        }
        if let Ok(value) = std::env::var("UCF_LLM_SEED") {
            cfg.seed = value
                .parse::<u64>()
                .map_err(|_| ComputeError::InvalidInput {
                    reason: format!("invalid UCF_LLM_SEED={value}"),
                })?;
        }
        if let Ok(value) = std::env::var("UCF_LLM_MAX_TOKENS") {
            cfg.max_tokens = value
                .parse::<u32>()
                .map_err(|_| ComputeError::InvalidInput {
                    reason: format!("invalid UCF_LLM_MAX_TOKENS={value}"),
                })?
                .clamp(1, MAX_LLM_TOKENS);
        }
        Ok(cfg)
    }
}

#[derive(Debug, Default)]
pub struct LlmStubBackend;

impl LlmInference for LlmStubBackend {
    fn name(&self) -> &'static str {
        "stub"
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
        let prompt_digest: [u8; 32] = Sha256::digest(req.prompt.as_bytes()).into();
        let mut text = String::new();
        match req.output_class {
            LlmOutputClass::SafeText => text.push_str("- Summary:\n- "),
            LlmOutputClass::Code => text.push_str("```text\n"),
            _ => {}
        }
        let mut truncated = false;
        for i in 0..req.max_tokens {
            let idx = ((u32::from(prompt_digest[(i as usize) % 32])
                ^ u32::from(req.context_digest[((i as usize) * 7) % 32])
                ^ (req.seed as u32)
                ^ i)
                % VOCAB.len() as u32) as usize;
            let token = VOCAB[idx];
            if text.len().saturating_add(token.len()).saturating_add(1) > MAX_LLM_TEXT_BYTES {
                truncated = true;
                break;
            }
            text.push_str(token);
            text.push(' ');
        }
        if req.output_class == LlmOutputClass::SafeText {
            text.push_str("\n- Next: verify policy and continue safely.");
        } else if req.output_class == LlmOutputClass::Code {
            text.push_str("\n```\n");
        }
        Ok(if truncated {
            LlmResponse::new(
                LlmStatus::Truncated,
                text,
                req.max_tokens,
                FinishReason::Length,
            )
        } else {
            LlmResponse::new(LlmStatus::Ok, text, req.max_tokens, FinishReason::Stop)
        })
    }
}

pub fn build_llm_backend(
    cfg: LlmBackendConfig,
) -> Result<Arc<dyn LlmInference + Send + Sync>, ComputeError> {
    match cfg.kind {
        LlmBackendKind::Stub => Ok(Arc::new(LlmStubBackend)),
        LlmBackendKind::Candle => {
            #[cfg(any(feature = "compute-candle", feature = "llm-candle"))]
            {
                Ok(Arc::new(CandleLlmBackend::from_fixture()?))
            }
            #[cfg(not(any(feature = "compute-candle", feature = "llm-candle")))]
            {
                Err(ComputeError::BackendDisabled)
            }
        }
        LlmBackendKind::Burn => {
            #[cfg(any(feature = "compute-burn", feature = "llm-burn"))]
            {
                Ok(Arc::new(BurnLlmBackend))
            }
            #[cfg(not(any(feature = "compute-burn", feature = "llm-burn")))]
            {
                Err(ComputeError::BackendDisabled)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(any(feature = "compute-candle", feature = "llm-candle"))]
    use crate::capabilities::llm_toy::ToyWeights;

    fn base_request() -> LlmRequest {
        LlmRequest {
            schema_version: 1,
            t: 7,
            decision_id: 42,
            candidate_id: 1,
            output_class: LlmOutputClass::SafeText,
            prompt: "hello deterministic world".to_string(),
            context_digest: [1; 32],
            evidence_chain_digest: [2; 32],
            seed: 9,
            max_tokens: 32,
            temperature: 0.0,
        }
    }

    #[test]
    fn stub_is_deterministic() {
        let backend = LlmStubBackend;
        let req = base_request();
        let a = backend
            .infer(&req, ComputeBudget::default())
            .expect("infer a");
        let b = backend
            .infer(&req, ComputeBudget::default())
            .expect("infer b");
        assert_eq!(a.digest, b.digest);
        assert_eq!(a.text, b.text);
    }

    #[test]
    fn stub_changes_when_seed_changes() {
        let backend = LlmStubBackend;
        let req = base_request();
        let a = backend
            .infer(&req, ComputeBudget::default())
            .expect("infer a");
        let mut req2 = req;
        req2.seed = 10;
        let b = backend
            .infer(&req2, ComputeBudget::default())
            .expect("infer b");
        assert_ne!(a.digest, b.digest);
    }

    #[test]
    fn stub_refuses_external_io_and_exec_intent() {
        let backend = LlmStubBackend;
        let mut req = base_request();
        req.output_class = LlmOutputClass::ExternalIo;
        let response = backend
            .infer(&req, ComputeBudget::default())
            .expect("infer");
        assert_eq!(response.status, LlmStatus::Refused);
        assert_eq!(response.finish_reason, FinishReason::PolicyRefusal);
    }

    #[cfg(any(feature = "compute-candle", feature = "llm-candle"))]
    #[test]
    fn candle_llm_is_deterministic_for_same_request() {
        let backend = build_llm_backend(LlmBackendConfig {
            kind: LlmBackendKind::Candle,
            ..LlmBackendConfig::default()
        })
        .expect("candle backend");
        let req = base_request();
        let a = backend
            .infer(&req, ComputeBudget::default())
            .expect("infer a");
        let b = backend
            .infer(&req, ComputeBudget::default())
            .expect("infer b");
        assert_eq!(a.digest, b.digest);
        assert_eq!(a.text, b.text);
    }

    #[cfg(any(feature = "compute-candle", feature = "llm-candle"))]
    #[test]
    fn candle_llm_weights_change_digest() {
        use crate::capabilities::candle_llm_backend::CandleLlmBackend;

        let req = base_request();
        let baseline = CandleLlmBackend::from_fixture()
            .expect("fixture")
            .infer(&req, ComputeBudget::default())
            .expect("infer baseline");

        let mut tweaked = ToyWeights::load().expect("load weights");
        for b in &mut tweaked.linear_b {
            *b = -10.0;
        }
        tweaked.linear_b[0] = 10.0;
        tweaked.vocab[0] = "forced_token".to_string();
        let alt = CandleLlmBackend::from_weights(tweaked)
            .infer(&req, ComputeBudget::default())
            .expect("infer alt");
        assert_ne!(baseline.digest, alt.digest);
    }

    #[cfg(any(feature = "compute-candle", feature = "llm-candle"))]
    #[test]
    fn candle_llm_honors_token_cap() {
        let backend = build_llm_backend(LlmBackendConfig {
            kind: LlmBackendKind::Candle,
            ..LlmBackendConfig::default()
        })
        .expect("candle backend");
        let mut req = base_request();
        req.max_tokens = 3;
        let out = backend
            .infer(&req, ComputeBudget::default())
            .expect("infer");
        assert_eq!(out.token_count, 3);
    }
}
