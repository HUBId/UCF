use std::io::{Read, Write};

use serde::{Deserialize, Serialize};
use ucf_compute::capabilities::{
    build_llm_backend, FinishReason, LlmBackendConfig, LlmRequest, LlmStatus,
};

const MAX_IPC_BYTES: usize = 64 * 1024;

#[derive(Serialize, Deserialize)]
struct LlmInferRequest {
    cmd: String,
    prompt: String,
    t: u64,
    max_tokens: u32,
    seed: u64,
    context_digest_hex: String,
}

#[derive(Serialize, Deserialize)]
struct LlmInferResponse {
    status: u8,
    text: String,
    token_count: u32,
    finish_reason: u8,
    digest: [u8; 32],
}

fn main() {
    #[cfg(target_os = "linux")]
    apply_linux_limits();

    let mut stdin = std::io::stdin().lock();
    let mut stdout = std::io::stdout().lock();
    let mut lb = [0u8; 4];
    if stdin.read_exact(&mut lb).is_err() {
        return;
    }
    let len = u32::from_le_bytes(lb) as usize;
    if len > MAX_IPC_BYTES {
        return;
    }
    let mut buf = vec![0u8; len];
    if stdin.read_exact(&mut buf).is_err() {
        return;
    }
    let Ok(req) = serde_json::from_slice::<LlmInferRequest>(&buf) else {
        return;
    };
    if req.cmd != "llm.infer" {
        return;
    }
    let backend = build_llm_backend(LlmBackendConfig::from_env().unwrap_or_default()).ok();
    let mut context_digest = [0u8; 32];
    let raw = hex::decode(req.context_digest_hex).unwrap_or_default();
    if raw.len() >= 32 {
        context_digest.copy_from_slice(&raw[..32]);
    }
    let request = LlmRequest {
        schema_version: 1,
        t: req.t,
        decision_id: 0,
        candidate_id: 0,
        output_class: ucf_compute::capabilities::LlmOutputClass::SafeText,
        prompt: req.prompt,
        context_digest,
        evidence_chain_digest: [0; 32],
        lfm_readout_digest: Some([0; 32]),
        lfm_uncertainty: None,
        lfm_stability: None,
        coherence: None,
        instability: None,
        risk: None,
        confidence: None,
        seed: req.seed,
        max_tokens: req.max_tokens,
        temperature: 0.0,
        top_p: 1.0,
        sampling_enabled: false,
    }
    .bounded();

    let response = if let Some(backend) = backend {
        backend
            .infer(&request, ucf_compute::ComputeBudget::default())
            .unwrap_or_else(|_| {
                ucf_compute::capabilities::LlmResponse::new(
                    LlmStatus::Failed,
                    "worker_failed".into(),
                    0,
                    FinishReason::Error,
                )
            })
    } else {
        ucf_compute::capabilities::LlmResponse::new(
            LlmStatus::Failed,
            "backend_unavailable".into(),
            0,
            FinishReason::Error,
        )
    };

    let rep = LlmInferResponse {
        status: match response.status {
            LlmStatus::Ok => 1,
            LlmStatus::Refused => 2,
            LlmStatus::Truncated => 3,
            _ => 4,
        },
        text: response.text,
        token_count: response.token_count,
        finish_reason: match response.finish_reason {
            FinishReason::Stop => 1,
            FinishReason::Length => 2,
            FinishReason::PolicyRefusal => 3,
            _ => 4,
        },
        digest: response.digest,
    };
    let out = serde_json::to_vec(&rep).unwrap_or_default();
    if out.len() > MAX_IPC_BYTES {
        return;
    }
    let _ = stdout.write_all(&(out.len() as u32).to_le_bytes());
    let _ = stdout.write_all(&out);
    let _ = stdout.flush();
}

#[cfg(target_os = "linux")]
fn apply_linux_limits() {
    unsafe {
        let no_core = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        let _ = libc::setrlimit(libc::RLIMIT_CORE, &no_core);
        let mem = libc::rlimit {
            rlim_cur: 512 * 1024 * 1024,
            rlim_max: 512 * 1024 * 1024,
        };
        let _ = libc::setrlimit(libc::RLIMIT_AS, &mem);
        let cpu = libc::rlimit {
            rlim_cur: 5,
            rlim_max: 5,
        };
        let _ = libc::setrlimit(libc::RLIMIT_CPU, &cpu);
        let files = libc::rlimit {
            rlim_cur: 32,
            rlim_max: 32,
        };
        let _ = libc::setrlimit(libc::RLIMIT_NOFILE, &files);
        let _ = libc::unshare(libc::CLONE_NEWNET);
    }
}
