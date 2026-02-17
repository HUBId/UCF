use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

use crate::ComputeError;

pub const REMOTE_SCHEMA_VERSION: u16 = 1;
pub const MAX_REMOTE_CANONICAL_BYTES: usize = 64 * 1024;

pub trait RemoteComputeClient: Send + Sync {
    fn call(&self, req: RemoteReq, timeout_ms: u64) -> Result<RemoteResp, RemoteErr>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteReq {
    pub request_id: u64,
    pub stage: String,
    pub canonical_input: Vec<u8>,
    pub pack_digest: [u8; 32],
    pub model_hash_digest: [u8; 32],
    pub nonce: [u8; 16],
    pub timestamp_ms: u64,
    pub signature: Vec<u8>,
}

impl RemoteReq {
    pub fn unsigned(
        request_id: u64,
        stage: impl Into<String>,
        canonical_input: Vec<u8>,
        pack_digest: [u8; 32],
        model_hash_digest: [u8; 32],
        nonce: [u8; 16],
        timestamp_ms: u64,
    ) -> Result<Self, RemoteErr> {
        if canonical_input.len() > MAX_REMOTE_CANONICAL_BYTES {
            return Err(RemoteErr::PayloadTooLarge {
                bytes: canonical_input.len(),
                max: MAX_REMOTE_CANONICAL_BYTES,
            });
        }
        Ok(Self {
            request_id,
            stage: stage.into(),
            canonical_input,
            pack_digest,
            model_hash_digest,
            nonce,
            timestamp_ms,
            signature: Vec::new(),
        })
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, RemoteErr> {
        canonical_encode_req(self)
    }

    pub fn sign(mut self, signer: &NodeSigner) -> Result<Self, RemoteErr> {
        let msg = self.canonical_bytes()?;
        self.signature = signer.sign(&msg);
        Ok(self)
    }

    pub fn verify_signature(&self, signer: &NodeSigner) -> Result<(), RemoteErr> {
        let msg = self.canonical_bytes()?;
        if signer.verify(&msg, &self.signature) {
            Ok(())
        } else {
            Err(RemoteErr::SignatureInvalid)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteResp {
    pub request_id: u64,
    pub status: u16,
    pub canonical_output: Vec<u8>,
    pub elapsed_ms: u64,
    pub server_signature: Vec<u8>,
    pub server_identity: String,
}

impl RemoteResp {
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, RemoteErr> {
        if self.canonical_output.len() > MAX_REMOTE_CANONICAL_BYTES {
            return Err(RemoteErr::PayloadTooLarge {
                bytes: self.canonical_output.len(),
                max: MAX_REMOTE_CANONICAL_BYTES,
            });
        }
        let mut out = Vec::new();
        out.extend_from_slice(&REMOTE_SCHEMA_VERSION.to_be_bytes());
        out.extend_from_slice(&self.request_id.to_be_bytes());
        out.extend_from_slice(&self.status.to_be_bytes());
        out.extend_from_slice(&(self.canonical_output.len() as u32).to_be_bytes());
        out.extend_from_slice(&self.canonical_output);
        out.extend_from_slice(&self.elapsed_ms.to_be_bytes());
        out.extend_from_slice(&(self.server_identity.len() as u16).to_be_bytes());
        out.extend_from_slice(self.server_identity.as_bytes());
        Ok(out)
    }

    pub fn verify(&self, request_id: u64, signer: &NodeSigner) -> Result<(), RemoteErr> {
        if self.request_id != request_id {
            return Err(RemoteErr::RequestIdMismatch {
                expected: request_id,
                actual: self.request_id,
            });
        }
        let msg = self.canonical_bytes()?;
        if !signer.verify(&msg, &self.server_signature) {
            return Err(RemoteErr::SignatureInvalid);
        }
        Ok(())
    }
}

fn canonical_encode_req(req: &RemoteReq) -> Result<Vec<u8>, RemoteErr> {
    if req.canonical_input.len() > MAX_REMOTE_CANONICAL_BYTES {
        return Err(RemoteErr::PayloadTooLarge {
            bytes: req.canonical_input.len(),
            max: MAX_REMOTE_CANONICAL_BYTES,
        });
    }
    let mut out = Vec::new();
    out.extend_from_slice(&REMOTE_SCHEMA_VERSION.to_be_bytes());
    out.extend_from_slice(&req.request_id.to_be_bytes());
    out.extend_from_slice(&(req.stage.len() as u16).to_be_bytes());
    out.extend_from_slice(req.stage.as_bytes());
    out.extend_from_slice(&(req.canonical_input.len() as u32).to_be_bytes());
    out.extend_from_slice(&req.canonical_input);
    out.extend_from_slice(&req.pack_digest);
    out.extend_from_slice(&req.model_hash_digest);
    out.extend_from_slice(&req.nonce);
    out.extend_from_slice(&req.timestamp_ms.to_be_bytes());
    Ok(out)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoteErr {
    SignatureInvalid,
    RequestIdMismatch { expected: u64, actual: u64 },
    PayloadTooLarge { bytes: usize, max: usize },
    PolicyDenied { reason: String },
    GovernorDenied { reason: String },
    Timeout,
    Internal { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeSigner {
    key: [u8; 32],
}

impl NodeSigner {
    pub fn from_key(key: [u8; 32]) -> Self {
        Self { key }
    }

    pub fn sign(&self, input: &[u8]) -> Vec<u8> {
        let mut h = Sha256::new();
        h.update(self.key);
        h.update(input);
        h.finalize().to_vec()
    }

    pub fn verify(&self, input: &[u8], signature: &[u8]) -> bool {
        self.sign(input).as_slice() == signature
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteGovernorConfig {
    pub max_concurrency: u16,
    pub max_retries: u8,
    pub timeout_ms: u64,
    pub per_stage_rate: BTreeMap<String, u16>,
}

impl Default for RemoteGovernorConfig {
    fn default() -> Self {
        Self {
            max_concurrency: 1,
            max_retries: 2,
            timeout_ms: 250,
            per_stage_rate: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TokenBucket {
    tokens: u16,
    last_refill_ms: u64,
    refill_per_sec: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteGovernor {
    pub cfg: RemoteGovernorConfig,
    active: u16,
    emergency_active: bool,
    buckets: BTreeMap<String, TokenBucket>,
}

impl RemoteGovernor {
    pub fn new(cfg: RemoteGovernorConfig) -> Self {
        let buckets = cfg
            .per_stage_rate
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    TokenBucket {
                        tokens: *v,
                        last_refill_ms: 0,
                        refill_per_sec: *v,
                    },
                )
            })
            .collect();
        Self {
            cfg,
            active: 0,
            emergency_active: false,
            buckets,
        }
    }

    pub fn set_emergency(&mut self, active: bool) {
        self.emergency_active = active;
    }

    pub fn acquire(&mut self, stage: &str, now_ms: u64) -> Result<(), RemoteErr> {
        if self.emergency_active {
            return Err(RemoteErr::GovernorDenied {
                reason: "emergency_active".to_string(),
            });
        }
        if self.active >= self.cfg.max_concurrency {
            return Err(RemoteErr::GovernorDenied {
                reason: "max_concurrency".to_string(),
            });
        }
        if let Some(bucket) = self.buckets.get_mut(stage) {
            refill_bucket(bucket, now_ms);
            if bucket.tokens == 0 {
                return Err(RemoteErr::GovernorDenied {
                    reason: "rate_limited".to_string(),
                });
            }
            bucket.tokens -= 1;
        }
        self.active += 1;
        Ok(())
    }

    pub fn release(&mut self) {
        self.active = self.active.saturating_sub(1);
    }
}

fn refill_bucket(bucket: &mut TokenBucket, now_ms: u64) {
    let elapsed_ms = now_ms.saturating_sub(bucket.last_refill_ms);
    if elapsed_ms < 1000 {
        return;
    }
    let refill_units = ((elapsed_ms / 1000) as u16).saturating_mul(bucket.refill_per_sec);
    bucket.tokens = bucket
        .tokens
        .saturating_add(refill_units)
        .min(bucket.refill_per_sec);
    bucket.last_refill_ms = now_ms;
}

#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Deserialize, serde::Serialize)]
pub struct RemotePolicyAllowlist {
    pub enabled: bool,
    pub allowed_endpoints: Vec<String>,
    pub allowed_stages: Vec<String>,
    pub allowed_policy_hashes: Vec<String>,
    pub max_input_bytes: usize,
    pub max_output_bytes: usize,
    pub max_rate_per_sec: u16,
    pub max_timeout_ms: u64,
}

#[derive(Debug, serde::Deserialize)]
struct RootAllowlists {
    #[serde(default)]
    remote_compute: RemotePolicyAllowlist,
}

impl RemotePolicyAllowlist {
    pub fn load(path: &Path) -> Result<Self, ComputeError> {
        let raw = std::fs::read_to_string(path).map_err(|e| ComputeError::InvalidInput {
            reason: format!("unable to read allowlist {}: {e}", path.display()),
        })?;
        let root: RootAllowlists =
            serde_json::from_str(&raw).map_err(|e| ComputeError::InvalidInput {
                reason: format!("unable to parse allowlist {}: {e}", path.display()),
            })?;
        Ok(root.remote_compute)
    }

    pub fn allows_policy_hash(&self, hash: &str) -> bool {
        self.allowed_policy_hashes.iter().any(|h| h == hash)
    }

    pub fn allows_stage(&self, stage: &str) -> bool {
        self.allowed_stages.iter().any(|s| s == stage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_request_sign_and_verify_roundtrip() {
        let signer = NodeSigner::from_key([7; 32]);
        let req = RemoteReq::unsigned(9, "world", vec![1, 2, 3], [1; 32], [2; 32], [3; 16], 100)
            .expect("unsigned req")
            .sign(&signer)
            .expect("signed");
        req.verify_signature(&signer).expect("verify");
    }

    #[test]
    fn remote_response_verification_checks_request_id() {
        let signer = NodeSigner::from_key([9; 32]);
        let mut resp = RemoteResp {
            request_id: 1,
            status: 200,
            canonical_output: vec![5, 6],
            elapsed_ms: 12,
            server_signature: Vec::new(),
            server_identity: "srv-a".to_string(),
        };
        let msg = resp.canonical_bytes().expect("canonical");
        resp.server_signature = signer.sign(&msg);
        let err = resp
            .verify(2, &signer)
            .expect_err("must reject wrong req id");
        assert!(matches!(err, RemoteErr::RequestIdMismatch { .. }));
    }

    #[test]
    fn bounded_payload_rejected() {
        let too_big = vec![0u8; MAX_REMOTE_CANONICAL_BYTES + 1];
        let err = RemoteReq::unsigned(1, "sae", too_big, [0; 32], [0; 32], [0; 16], 1)
            .expect_err("must reject big payload");
        assert!(matches!(err, RemoteErr::PayloadTooLarge { .. }));
    }

    #[test]
    fn governor_blocks_emergency_and_refills() {
        let mut cfg = RemoteGovernorConfig::default();
        cfg.per_stage_rate.insert("lfm/step".to_string(), 1);
        let mut gov = RemoteGovernor::new(cfg);
        gov.acquire("lfm/step", 0).expect("first token");
        gov.release();
        let denied = gov.acquire("lfm/step", 1).expect_err("rate limited");
        assert!(matches!(denied, RemoteErr::GovernorDenied { .. }));
        gov.acquire("lfm/step", 1000).expect("refilled");
        gov.release();
        gov.set_emergency(true);
        let denied = gov.acquire("lfm/step", 2000).expect_err("emergency");
        assert!(matches!(denied, RemoteErr::GovernorDenied { .. }));
    }
}
