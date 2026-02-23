use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DeterminismMode {
    #[default]
    DeterministicOnly,
    SeededAllowed,
    NondetAllowed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RngSiteId {
    LlmBeamSearch,
    LlmSampling,
    TestOnlyFuzz,
    SyntheticFixtureGen,
}

impl RngSiteId {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LlmBeamSearch => "llm_beam_search",
            Self::LlmSampling => "llm_sampling",
            Self::TestOnlyFuzz => "test_only_fuzz",
            Self::SyntheticFixtureGen => "synthetic_fixture_gen",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeterminismPolicyV1 {
    #[serde(default)]
    pub allowed_rng_sites: Vec<RngSiteId>,
    #[serde(default)]
    pub allowed_mode: DeterminismMode,
    #[serde(default = "default_seed_source")]
    pub global_seed_source: String,
}

fn default_seed_source() -> String {
    "run_id_policy_graph_digest".to_string()
}

impl Default for DeterminismPolicyV1 {
    fn default() -> Self {
        Self {
            allowed_rng_sites: Vec::new(),
            allowed_mode: DeterminismMode::DeterministicOnly,
            global_seed_source: default_seed_source(),
        }
    }
}

impl DeterminismPolicyV1 {
    pub fn digest_hex(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"ucf.policy.determinism.v1");
        h.update([self.allowed_mode as u8]);
        for site in &self.allowed_rng_sites {
            h.update(site.as_str().as_bytes());
            h.update([0]);
        }
        h.update(self.global_seed_source.as_bytes());
        hex::encode(h.finalize())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeterminismCtx {
    pub run_id: String,
    pub policy_graph_digest: [u8; 32],
    pub policy: DeterminismPolicyV1,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("RNG denied at site={site} code={code}")]
pub struct RngDenied {
    pub site: &'static str,
    pub code: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RngUsageRecord {
    pub site_id: RngSiteId,
    pub run_id: String,
    pub policy_graph_digest_prefix: [u8; 8],
    pub seed_digest_prefix: [u8; 8],
    pub reason_code: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RngDeniedRecord {
    pub site_id: RngSiteId,
    pub run_id: String,
    pub policy_graph_digest_prefix: [u8; 8],
    pub reason_code: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    pub fn next_u64(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
}

fn prefix8(d: [u8; 32]) -> [u8; 8] {
    let mut out = [0_u8; 8];
    out.copy_from_slice(&d[..8]);
    out
}

pub fn rng(
    site: RngSiteId,
    ctx: &DeterminismCtx,
) -> Result<(DeterministicRng, RngUsageRecord), RngDenied> {
    if !ctx.policy.allowed_rng_sites.contains(&site) {
        return Err(RngDenied {
            site: site.as_str(),
            code: "RNG_DENIED_BY_POLICY",
        });
    }

    let mut h = Sha256::new();
    h.update(b"ucf.rng.site.v1");
    h.update(ctx.run_id.as_bytes());
    h.update([0]);
    h.update(ctx.policy_graph_digest);
    h.update([0]);
    h.update(site.as_str().as_bytes());
    let digest: [u8; 32] = h.finalize().into();

    let mut seed_bytes = [0_u8; 8];
    seed_bytes.copy_from_slice(&digest[..8]);
    let state = u64::from_le_bytes(seed_bytes).max(1);
    Ok((
        DeterministicRng { state },
        RngUsageRecord {
            site_id: site,
            run_id: ctx.run_id.clone(),
            policy_graph_digest_prefix: prefix8(ctx.policy_graph_digest),
            seed_digest_prefix: prefix8(digest),
            reason_code: "RNG_GRANTED".to_string(),
        },
    ))
}

pub fn denied_record(site: RngSiteId, ctx: &DeterminismCtx, code: &str) -> RngDeniedRecord {
    RngDeniedRecord {
        site_id: site,
        run_id: ctx.run_id.clone(),
        policy_graph_digest_prefix: prefix8(ctx.policy_graph_digest),
        reason_code: code.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> DeterminismCtx {
        DeterminismCtx {
            run_id: "run-1".to_string(),
            policy_graph_digest: [7_u8; 32],
            policy: DeterminismPolicyV1 {
                allowed_rng_sites: vec![RngSiteId::TestOnlyFuzz],
                ..DeterminismPolicyV1::default()
            },
        }
    }

    #[test]
    fn deterministic_stream_is_stable() {
        let (mut a, _) = rng(RngSiteId::TestOnlyFuzz, &ctx()).expect("allowed");
        let (mut b, _) = rng(RngSiteId::TestOnlyFuzz, &ctx()).expect("allowed");
        assert_eq!(a.next_u64(), b.next_u64());
        assert_eq!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn denied_site_returns_stable_code() {
        let denied = rng(RngSiteId::LlmSampling, &ctx()).expect_err("must deny");
        assert_eq!(denied.code, "RNG_DENIED_BY_POLICY");
        let rec = denied_record(RngSiteId::LlmSampling, &ctx(), denied.code);
        assert_eq!(rec.reason_code, "RNG_DENIED_BY_POLICY");
    }
}
