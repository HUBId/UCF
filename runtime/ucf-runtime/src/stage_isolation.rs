#[cfg(feature = "stage-isolation")]
use std::io::{Read, Write};
#[cfg(feature = "stage-isolation")]
use std::process::{Command, Stdio};

#[cfg(feature = "stage-isolation")]
use serde::{Deserialize, Serialize};
use ucf_compute::capabilities::LlmRequest;
use ucf_compute::capabilities::LlmResponse;
#[cfg(feature = "stage-isolation")]
use ucf_compute::capabilities::{FinishReason, LlmStatus};

#[cfg(feature = "stage-isolation")]
const MAX_IPC_BYTES: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageIsolationMode {
    Off,
    Llm,
    Ebm,
    All,
}

impl StageIsolationMode {
    pub fn from_env() -> Self {
        #[cfg(feature = "stage-isolation")]
        {
            return match std::env::var("UCF_STAGE_ISOLATION")
                .unwrap_or_else(|_| "off".to_string())
                .as_str()
            {
                "llm" => Self::Llm,
                "ebm" => Self::Ebm,
                "all" => Self::All,
                _ => Self::Off,
            };
        }
        #[cfg(not(feature = "stage-isolation"))]
        {
            Self::Off
        }
    }

    pub fn isolate_llm(self) -> bool {
        matches!(self, Self::Llm | Self::All)
    }
}

#[cfg(feature = "stage-isolation")]
#[derive(Serialize, Deserialize)]
struct LlmInferRequest {
    cmd: String,
    prompt: String,
    t: u64,
    max_tokens: u32,
    seed: u64,
    context_digest_hex: String,
}

#[cfg(feature = "stage-isolation")]
#[derive(Serialize, Deserialize)]
struct LlmInferResponse {
    status: u8,
    text: String,
    token_count: u32,
    finish_reason: u8,
    digest: [u8; 32],
}

#[cfg(feature = "stage-isolation")]
pub fn infer_llm_isolated(req: &LlmRequest) -> Result<LlmResponse, String> {
    let worker =
        std::env::var("UCF_STAGE_WORKER_BIN").unwrap_or_else(|_| "ucf-stage-worker".to_string());
    let request = LlmInferRequest {
        cmd: "llm.infer".to_string(),
        prompt: req.prompt.clone(),
        t: req.t,
        max_tokens: req.max_tokens,
        seed: req.seed,
        context_digest_hex: hex::encode(req.context_digest),
    };
    let payload = serde_json::to_vec(&request).map_err(|e| e.to_string())?;
    if payload.len() > MAX_IPC_BYTES {
        return Err("ipc_request_too_large".to_string());
    }
    let mut child = Command::new(worker)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .env_clear()
        .env("UCF_NO_NETWORK", "1")
        .spawn()
        .map_err(|e| format!("spawn_failed:{e}"))?;

    {
        let stdin = child.stdin.as_mut().ok_or("stdin_missing")?;
        stdin
            .write_all(&(payload.len() as u32).to_le_bytes())
            .and_then(|_| stdin.write_all(&payload))
            .map_err(|e| format!("ipc_write:{e}"))?;
    }

    let mut lb = [0u8; 4];
    let stdout = child.stdout.as_mut().ok_or("stdout_missing")?;
    stdout
        .read_exact(&mut lb)
        .map_err(|e| format!("ipc_read_len:{e}"))?;
    let n = u32::from_le_bytes(lb) as usize;
    if n > MAX_IPC_BYTES {
        return Err("ipc_reply_too_large".to_string());
    }
    let mut out = vec![0u8; n];
    stdout
        .read_exact(&mut out)
        .map_err(|e| format!("ipc_read_payload:{e}"))?;

    let status = child.wait().map_err(|e| format!("wait_failed:{e}"))?;
    if !status.success() {
        return Err(format!("worker_exit:{status}"));
    }

    let rep: LlmInferResponse = serde_json::from_slice(&out).map_err(|e| e.to_string())?;
    Ok(LlmResponse {
        status: match rep.status {
            1 => LlmStatus::Ok,
            2 => LlmStatus::Refused,
            3 => LlmStatus::Truncated,
            _ => LlmStatus::Failed,
        },
        text: rep.text,
        token_count: rep.token_count,
        finish_reason: match rep.finish_reason {
            1 => FinishReason::Stop,
            2 => FinishReason::Length,
            3 => FinishReason::PolicyRefusal,
            _ => FinishReason::Error,
        },
        digest: rep.digest,
    })
}

#[cfg(not(feature = "stage-isolation"))]
pub fn infer_llm_isolated(_req: &LlmRequest) -> Result<LlmResponse, String> {
    Err("stage_isolation_feature_disabled".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_mode() {
        std::env::set_var("UCF_STAGE_ISOLATION", "llm");
        #[cfg(feature = "stage-isolation")]
        assert!(StageIsolationMode::from_env().isolate_llm());
        #[cfg(not(feature = "stage-isolation"))]
        assert!(!StageIsolationMode::from_env().isolate_llm());
    }
}
