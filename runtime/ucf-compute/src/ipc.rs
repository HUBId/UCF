use std::io::{Read, Write};

use sha2::{Digest, Sha256};

use crate::capabilities::{FinishReason, LlmOutputClass, LlmRequest, LlmResponse, LlmStatus};
use crate::feature_extractor::{SaeInput, SaeOutput};
use crate::lfm::{LfmInput, LfmOutput};
use crate::ssm::{SsmInput, SsmOutput};
use crate::world_model::{WorldModelInput, WorldModelOutput};

pub const IPC_SCHEMA_VERSION: u16 = 1;
pub const MAX_IPC_PAYLOAD_BYTES: usize = 256 * 1024;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum WorkerStage {
    Llm,
    World,
    Sae,
    Ssm,
    Lfm,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LlmRequestIpc {
    pub schema_version: u16,
    pub t: u64,
    pub decision_id: u64,
    pub candidate_id: u16,
    pub output_class: u8,
    pub prompt: String,
    pub context_digest: [u8; 32],
    pub evidence_chain_digest: [u8; 32],
    pub lfm_readout_digest: Option<[u8; 32]>,
    pub lfm_uncertainty: Option<f32>,
    pub lfm_stability: Option<f32>,
    pub coherence: Option<f32>,
    pub instability: Option<f32>,
    pub risk: Option<f32>,
    pub confidence: Option<f32>,
    pub seed: u64,
    pub max_tokens: u32,
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub sampling_enabled: bool,
}

fn default_top_p() -> f32 {
    1.0
}

impl From<LlmRequest> for LlmRequestIpc {
    fn from(value: LlmRequest) -> Self {
        Self {
            schema_version: value.schema_version,
            t: value.t,
            decision_id: value.decision_id,
            candidate_id: value.candidate_id,
            output_class: value.output_class as u8,
            prompt: value.prompt,
            context_digest: value.context_digest,
            evidence_chain_digest: value.evidence_chain_digest,
            lfm_readout_digest: value.lfm_readout_digest,
            lfm_uncertainty: value.lfm_uncertainty,
            lfm_stability: value.lfm_stability,
            coherence: value.coherence,
            instability: value.instability,
            risk: value.risk,
            confidence: value.confidence,
            seed: value.seed,
            max_tokens: value.max_tokens,
            temperature: value.temperature,
            top_p: value.top_p,
            sampling_enabled: value.sampling_enabled,
        }
    }
}

impl From<LlmRequestIpc> for LlmRequest {
    fn from(value: LlmRequestIpc) -> Self {
        let output_class = match value.output_class {
            0 => LlmOutputClass::SafeText,
            1 => LlmOutputClass::Code,
            2 => LlmOutputClass::ExternalIo,
            3 => LlmOutputClass::ExecIntent,
            _ => LlmOutputClass::Sensitive,
        };
        LlmRequest {
            schema_version: value.schema_version,
            t: value.t,
            decision_id: value.decision_id,
            candidate_id: value.candidate_id,
            output_class,
            prompt: value.prompt,
            context_digest: value.context_digest,
            evidence_chain_digest: value.evidence_chain_digest,
            lfm_readout_digest: value.lfm_readout_digest,
            lfm_uncertainty: value.lfm_uncertainty,
            lfm_stability: value.lfm_stability,
            coherence: value.coherence,
            instability: value.instability,
            risk: value.risk,
            confidence: value.confidence,
            seed: value.seed,
            max_tokens: value.max_tokens,
            temperature: value.temperature,
            top_p: value.top_p,
            sampling_enabled: value.sampling_enabled,
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LlmResponseIpc {
    pub status: u8,
    pub text: String,
    pub token_count: u32,
    pub finish_reason: u8,
    pub digest: [u8; 32],
}

impl From<LlmResponse> for LlmResponseIpc {
    fn from(value: LlmResponse) -> Self {
        Self {
            status: value.status as u8,
            text: value.text,
            token_count: value.token_count,
            finish_reason: value.finish_reason as u8,
            digest: value.digest,
        }
    }
}

impl From<LlmResponseIpc> for LlmResponse {
    fn from(value: LlmResponseIpc) -> Self {
        let status = match value.status {
            0 => LlmStatus::Ok,
            1 => LlmStatus::Truncated,
            2 => LlmStatus::Refused,
            _ => LlmStatus::Failed,
        };
        let finish_reason = match value.finish_reason {
            0 => FinishReason::Stop,
            1 => FinishReason::Length,
            2 => FinishReason::PolicyRefusal,
            _ => FinishReason::Error,
        };
        let mut out = LlmResponse::new(status, value.text, value.token_count, finish_reason);
        out.digest = value.digest;
        out
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InitRequest {
    pub schema_version: u16,
    pub stage: WorkerStage,
    pub model_hashes_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ComputeRequest {
    pub schema_version: u16,
    pub request_id: u64,
    pub t: u64,
    pub stage: WorkerStage,
    pub seed: u64,
    pub timeout_ms: u32,
    pub input: StageInput,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum StageInput {
    Llm(LlmRequestIpc),
    World(WorldModelInput),
    Sae(SaeInput),
    Ssm(SsmInput),
    Lfm(LfmInput),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeStatus {
    Ok,
    Timeout,
    Error,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ComputeResponse {
    pub schema_version: u16,
    pub request_id: u64,
    pub stage: WorkerStage,
    pub status: ComputeStatus,
    pub elapsed_ms: u32,
    pub quality: u8,
    pub error_code: Option<String>,
    pub output: Option<StageOutput>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum StageOutput {
    Llm(LlmResponseIpc),
    World(WorldModelOutput),
    Sae(SaeOutput),
    Ssm(SsmOutput),
    Lfm(Box<LfmOutput>),
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum WorkerRequest {
    Init(InitRequest),
    Compute(Box<ComputeRequest>),
    Shutdown,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum WorkerResponse {
    InitAck {
        schema_version: u16,
    },
    Compute(Box<ComputeResponse>),
    Error {
        schema_version: u16,
        error_code: String,
    },
}

pub fn encode_frame<T: serde::Serialize>(msg: &T) -> Result<Vec<u8>, String> {
    let payload = rmp_serde::to_vec_named(msg).map_err(|e| format!("encode: {e}"))?;
    if payload.len() > MAX_IPC_PAYLOAD_BYTES {
        return Err("payload_too_large".to_string());
    }
    let mut frame = Vec::with_capacity(4 + payload.len() + 32);
    frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    frame.extend_from_slice(&payload);
    let checksum: [u8; 32] = Sha256::digest(&payload).into();
    frame.extend_from_slice(&checksum);
    Ok(frame)
}

pub fn decode_frame<T: serde::de::DeserializeOwned>(frame: &[u8]) -> Result<T, String> {
    if frame.len() < 4 + 32 {
        return Err("frame_too_short".to_string());
    }
    let mut len_bytes = [0_u8; 4];
    len_bytes.copy_from_slice(&frame[..4]);
    let payload_len = u32::from_le_bytes(len_bytes) as usize;
    if payload_len > MAX_IPC_PAYLOAD_BYTES {
        return Err("payload_too_large".to_string());
    }
    let expected_total = 4 + payload_len + 32;
    if frame.len() != expected_total {
        return Err("frame_length_mismatch".to_string());
    }
    let payload = &frame[4..(4 + payload_len)];
    let mut checksum = [0_u8; 32];
    checksum.copy_from_slice(&frame[(4 + payload_len)..]);
    let actual: [u8; 32] = Sha256::digest(payload).into();
    if checksum != actual {
        return Err("checksum_mismatch".to_string());
    }
    rmp_serde::from_slice(payload).map_err(|e| format!("decode: {e}"))
}

pub fn write_frame<W: Write, T: serde::Serialize>(writer: &mut W, msg: &T) -> Result<(), String> {
    let encoded = encode_frame(msg)?;
    writer
        .write_all(&encoded)
        .map_err(|e| format!("write: {e}"))?;
    writer.flush().map_err(|e| format!("flush: {e}"))
}

pub fn read_frame<R: Read, T: serde::de::DeserializeOwned>(reader: &mut R) -> Result<T, String> {
    let mut len_bytes = [0_u8; 4];
    reader
        .read_exact(&mut len_bytes)
        .map_err(|e| format!("read_len: {e}"))?;
    let payload_len = u32::from_le_bytes(len_bytes) as usize;
    if payload_len > MAX_IPC_PAYLOAD_BYTES {
        return Err("payload_too_large".to_string());
    }
    let mut payload = vec![0_u8; payload_len];
    reader
        .read_exact(&mut payload)
        .map_err(|e| format!("read_payload: {e}"))?;
    let mut checksum = [0_u8; 32];
    reader
        .read_exact(&mut checksum)
        .map_err(|e| format!("read_checksum: {e}"))?;

    let mut frame = Vec::with_capacity(4 + payload_len + 32);
    frame.extend_from_slice(&len_bytes);
    frame.extend_from_slice(&payload);
    frame.extend_from_slice(&checksum);
    decode_frame(&frame)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_roundtrip() {
        let req = WorkerRequest::Init(InitRequest {
            schema_version: IPC_SCHEMA_VERSION,
            stage: WorkerStage::World,
            model_hashes_digest: [7; 32],
        });
        let frame = encode_frame(&req).expect("encode");
        let decoded: WorkerRequest = decode_frame(&frame).expect("decode");
        assert_eq!(decoded, req);
    }

    #[test]
    fn payload_bound_enforced() {
        #[derive(serde::Serialize)]
        struct Big {
            data: Vec<u8>,
        }
        let msg = Big {
            data: vec![0_u8; MAX_IPC_PAYLOAD_BYTES + 1],
        };
        assert!(encode_frame(&msg).is_err());
    }

    #[test]
    fn checksum_validation() {
        let req = WorkerRequest::Shutdown;
        let mut frame = encode_frame(&req).expect("encode");
        let idx = frame.len() - 1;
        frame[idx] ^= 0xAA;
        assert!(decode_frame::<WorkerRequest>(&frame).is_err());
    }
}
