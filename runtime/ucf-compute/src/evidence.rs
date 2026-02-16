use sha2::{Digest, Sha256};

use crate::backend_pack::{BackendComponentId, BackendPackId};
use crate::risk_contract::{BackendProfileId, EvidenceRef, RiskSignal};
use crate::world_model::StageQuality;
use crate::{ComputeInput, Spike};
use ucf_types::{
    canonicalize_f32, quantize_signed_unit_i16, quantize_unit, CANONICAL_UNIT_QUANT_MAX,
};

pub const COMPUTE_SUMMARY_SCHEMA_VERSION: u16 = 2;
const CODE_VERSION_MAX_LEN: usize = 16;

pub fn canonical_f32_bits(value: f32) -> u32 {
    canonicalize_f32(value)
}

pub fn quantize_unit_u16(value: f32) -> u16 {
    quantize_unit(value, CANONICAL_UNIT_QUANT_MAX)
}

pub fn quantize_signed_unit(value: f32) -> i16 {
    quantize_signed_unit_i16(value)
}

pub trait CanonicalEncode {
    fn encode_canonical(&self, out: &mut Vec<u8>);
}

pub fn digest32(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = [0_u8; 32];
    out.copy_from_slice(&digest);
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodeVersionTag {
    raw: &'static str,
}

impl CodeVersionTag {
    pub fn current() -> Self {
        if let Some(commit) = option_env!("UCF_GIT_COMMIT") {
            let commit = if commit.len() >= 12 {
                &commit[..12]
            } else {
                commit
            };
            return Self { raw: commit };
        }

        let version = env!("CARGO_PKG_VERSION");
        let version = if version.len() + 1 > CODE_VERSION_MAX_LEN {
            "v0.0.0"
        } else {
            option_env!("CARGO_PKG_VERSION").unwrap_or("0.0.0")
        };
        let prefixed = if version.starts_with('v') {
            version
        } else {
            match version {
                "0.0.0" => "v0.0.0",
                v => v,
            }
        };

        Self { raw: prefixed }
    }

    pub fn as_str(self) -> &'static str {
        self.raw
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvidenceChain {
    pub schema_version: u16,
    pub code_version: CodeVersionTag,
    pub backend_profile: BackendProfileId,
    pub backend_pack_id: BackendPackId,
    pub fixtures_digest: [u8; 32],
    pub llm_backend: BackendComponentId,
    pub world_backend: BackendComponentId,
    pub sae_backend: BackendComponentId,
    pub ssm_backend: BackendComponentId,
    pub lfm_backend: BackendComponentId,
    pub budget_profile_id: u32,
    pub seed: u64,
    pub context_digest: [u8; 32],
    pub world_digest: Option<[u8; 32]>,
    pub spikes_digest: Option<[u8; 32]>,
    pub ssm_digest: Option<[u8; 32]>,
    pub lfm_digest: Option<[u8; 32]>,
    pub risk_digest: [u8; 32],
    pub sae_quality: Option<StageQuality>,
    pub ssm_quality: Option<StageQuality>,
    pub lfm_quality: Option<StageQuality>,
    pub chain_digest: [u8; 32],
}

impl EvidenceChain {
    pub fn from_compute(
        input: &ComputeInput,
        spikes: &[Spike],
        risk_signal: &RiskSignal,
        sae_quality: Option<StageQuality>,
        ssm_quality: Option<StageQuality>,
        lfm_quality: Option<StageQuality>,
    ) -> Self {
        let risk_digest = digest_canonical(risk_signal);
        let mut chain = Self {
            schema_version: COMPUTE_SUMMARY_SCHEMA_VERSION,
            code_version: CodeVersionTag::current(),
            backend_profile: risk_signal.evidence.backend_profile,
            backend_pack_id: risk_signal.evidence.backend_pack_id,
            fixtures_digest: risk_signal.evidence.fixtures_digest,
            llm_backend: risk_signal.evidence.llm_backend,
            world_backend: risk_signal.evidence.world_backend,
            sae_backend: risk_signal.evidence.sae_backend,
            ssm_backend: risk_signal.evidence.ssm_backend,
            lfm_backend: risk_signal.evidence.lfm_backend,
            budget_profile_id: risk_signal.evidence.budget_profile_id,
            seed: risk_signal.evidence.seed,
            context_digest: input.context_digest,
            world_digest: risk_signal.evidence.world_digest,
            spikes_digest: risk_signal
                .evidence
                .spikes_digest
                .or_else(|| (!spikes.is_empty()).then(|| spikes_digest(spikes))),
            ssm_digest: risk_signal.evidence.ssm_digest,
            lfm_digest: risk_signal.evidence.lfm_digest,
            risk_digest,
            sae_quality,
            ssm_quality,
            lfm_quality,
            chain_digest: [0; 32],
        };
        chain.chain_digest = digest_canonical(&chain);
        chain
    }

    pub fn digest_prefix_hex(&self) -> String {
        hex::encode(&self.chain_digest[..6])
    }
}

pub fn digest_canonical<T: CanonicalEncode>(value: &T) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(256);
    value.encode_canonical(&mut bytes);
    digest32(&bytes)
}

pub fn spikes_digest(spikes: &[Spike]) -> [u8; 32] {
    let canonical = CanonicalSpikes::from(spikes);
    digest_canonical(&canonical)
}

#[derive(Debug, Clone)]
struct CanonicalSpikes(Vec<Spike>);

impl From<&[Spike]> for CanonicalSpikes {
    fn from(value: &[Spike]) -> Self {
        let mut spikes = value.to_vec();
        spikes.sort_by(|a, b| {
            a.timestamp
                .cmp(&b.timestamp)
                .then_with(|| a.feature_id.cmp(&b.feature_id))
                .then_with(|| canonical_f32_bits(a.magnitude).cmp(&canonical_f32_bits(b.magnitude)))
        });
        Self(spikes)
    }
}

impl CanonicalEncode for ComputeInput {
    fn encode_canonical(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.frame_id.0.to_le_bytes());
        out.extend_from_slice(&self.t.to_le_bytes());
        out.extend_from_slice(&self.context_digest);
    }
}

impl CanonicalEncode for EvidenceRef {
    fn encode_canonical(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.context_digest);
        encode_opt_digest(out, self.world_digest);
        encode_opt_digest(out, self.spikes_digest);
        encode_opt_digest(out, self.ssm_digest);
        encode_opt_digest(out, self.lfm_digest);
        out.push(self.backend_profile as u8);
        out.extend_from_slice(&self.backend_pack_id.0.to_le_bytes());
        out.extend_from_slice(&self.fixtures_digest);
        out.push(self.llm_backend as u8);
        out.push(self.world_backend as u8);
        out.push(self.sae_backend as u8);
        out.push(self.ssm_backend as u8);
        out.push(self.lfm_backend as u8);
        out.extend_from_slice(&self.budget_profile_id.to_le_bytes());
        out.extend_from_slice(&self.seed.to_le_bytes());
    }
}

impl CanonicalEncode for RiskSignal {
    fn encode_canonical(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&quantize_unit_u16(self.risk).to_le_bytes());
        out.extend_from_slice(&quantize_unit_u16(self.confidence).to_le_bytes());
        out.push(self.quality as u8);
        self.evidence.encode_canonical(out);
        out.extend_from_slice(&self.version.to_le_bytes());
    }
}

impl CanonicalEncode for CanonicalSpikes {
    fn encode_canonical(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.0.len() as u32).to_le_bytes());
        for spike in &self.0 {
            out.extend_from_slice(&spike.timestamp.to_le_bytes());
            out.extend_from_slice(&spike.feature_id.to_le_bytes());
            out.extend_from_slice(&quantize_unit_u16(spike.magnitude).to_le_bytes());
        }
    }
}

impl CanonicalEncode for EvidenceChain {
    fn encode_canonical(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.schema_version.to_le_bytes());
        encode_bounded_string(out, self.code_version.as_str());
        out.push(self.backend_profile as u8);
        out.extend_from_slice(&self.backend_pack_id.0.to_le_bytes());
        out.extend_from_slice(&self.fixtures_digest);
        out.push(self.llm_backend as u8);
        out.push(self.world_backend as u8);
        out.push(self.sae_backend as u8);
        out.push(self.ssm_backend as u8);
        out.push(self.lfm_backend as u8);
        out.extend_from_slice(&self.budget_profile_id.to_le_bytes());
        out.extend_from_slice(&self.seed.to_le_bytes());
        out.extend_from_slice(&self.context_digest);
        encode_opt_digest(out, self.world_digest);
        encode_opt_digest(out, self.spikes_digest);
        encode_opt_digest(out, self.ssm_digest);
        encode_opt_digest(out, self.lfm_digest);
        out.extend_from_slice(&self.risk_digest);
        match self.sae_quality {
            Some(q) => {
                out.push(1);
                out.push(q as u8);
            }
            None => out.push(0),
        }
        match self.ssm_quality {
            Some(q) => {
                out.push(1);
                out.push(q as u8);
            }
            None => out.push(0),
        }
        match self.lfm_quality {
            Some(q) => {
                out.push(1);
                out.push(q as u8);
            }
            None => out.push(0),
        }
    }
}

fn encode_opt_digest(out: &mut Vec<u8>, digest: Option<[u8; 32]>) {
    match digest {
        Some(digest) => {
            out.push(1);
            out.extend_from_slice(&digest);
        }
        None => out.push(0),
    }
}

fn encode_bounded_string(out: &mut Vec<u8>, value: &str) {
    let truncated = value.chars().take(CODE_VERSION_MAX_LEN).collect::<String>();
    out.extend_from_slice(&(truncated.len() as u16).to_le_bytes());
    out.extend_from_slice(truncated.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::risk_contract::SignalQuality;
    use crate::FrameId;
    use proptest::prelude::*;

    fn spike(feature_id: u32, magnitude: f32, timestamp: u64) -> Spike {
        Spike {
            feature_id,
            magnitude,
            timestamp,
        }
    }

    fn assert_invariants(chain: &EvidenceChain) {
        assert!(chain.schema_version >= 1);
        assert_ne!(chain.seed, 0);
        let mut canonical = *chain;
        canonical.chain_digest = [0; 32];
        assert_eq!(chain.chain_digest, digest_canonical(&canonical));
    }

    fn assert_evidence_chain(chain: &EvidenceChain) {
        assert_ne!(chain.risk_digest, [0; 32]);
        assert_invariants(chain);
    }

    #[test]
    fn canonical_spikes_digest_is_order_independent() {
        let a = vec![spike(2, 0.5, 9), spike(1, 0.25, 8)];
        let b = vec![spike(1, 0.25, 8), spike(2, 0.5, 9)];
        assert_eq!(spikes_digest(&a), spikes_digest(&b));
    }

    #[test]
    fn mini_fuzz_spikes_digest_no_panics() {
        let mut seed = 0xA5A5_5A5A_u64;
        for _ in 0..2048 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let len = ((seed >> 8) as usize) % 64;
            let mut spikes = Vec::with_capacity(len);
            for idx in 0..len {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let mag = ((seed >> 16) as u16) as f32 / u16::MAX as f32;
                spikes.push(Spike {
                    feature_id: (seed as u32) % 256,
                    magnitude: mag,
                    timestamp: idx as u64,
                });
            }
            let digest = spikes_digest(&spikes);
            assert_ne!(digest, [0; 32]);
        }
    }

    #[test]
    fn canonical_spikes_normalize_negative_zero() {
        let pos_zero = spikes_digest(&[spike(1, 0.0, 1)]);
        let neg_zero = spikes_digest(&[spike(1, -0.0, 1)]);
        assert_eq!(pos_zero, neg_zero);
    }

    #[test]
    fn risk_digest_stable_under_small_float_representation_drift() {
        let base = RiskSignal {
            risk: 0.3,
            confidence: 0.7,
            quality: SignalQuality::Unavailable,
            evidence: EvidenceRef {
                context_digest: [1; 32],
                world_digest: None,
                spikes_digest: None,
                ssm_digest: None,
                lfm_digest: None,
                backend_profile: BackendProfileId::StubV1,
                backend_pack_id: crate::BackendPackId(1),
                fixtures_digest: [2; 32],
                llm_backend: crate::BackendComponentId::ToyV1,
                world_backend: crate::BackendComponentId::ToyV1,
                sae_backend: crate::BackendComponentId::ToyV1,
                ssm_backend: crate::BackendComponentId::ToyV1,
                lfm_backend: crate::BackendComponentId::ToyV1,
                seed: 9,
                budget_profile_id: 1,
            },
            version: 1,
        };
        let mut drift = base;
        drift.risk = 0.1 + 0.2;
        assert_eq!(digest_canonical(&base), digest_canonical(&drift));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn canonical_digest_is_stable_for_same_input(seed in 1u64..10_000, risk in 0.0f32..1.0, confidence in 0.0f32..1.0) {
            let input = ComputeInput { frame_id: FrameId(42), t: 17, context_digest: [7; 32] };
            let risk_signal = RiskSignal {
                risk,
                confidence,
                quality: SignalQuality::Unavailable,
                evidence: EvidenceRef {
                    context_digest: input.context_digest,
                    world_digest: None,
                    spikes_digest: None,
                    ssm_digest: None,
                    lfm_digest: None,
                    backend_profile: BackendProfileId::StubV1,
                    backend_pack_id: crate::BackendPackId(1),
                    fixtures_digest: [9; 32],
                    llm_backend: crate::BackendComponentId::ToyV1,
                    world_backend: crate::BackendComponentId::ToyV1,
                    sae_backend: crate::BackendComponentId::ToyV1,
                    ssm_backend: crate::BackendComponentId::ToyV1,
                    lfm_backend: crate::BackendComponentId::ToyV1,
                    seed,
                    budget_profile_id: 1,
                },
                version: 1,
            };
            let a = EvidenceChain::from_compute(&input, &[], &risk_signal, None, None, None);
            let b = EvidenceChain::from_compute(&input, &[], &risk_signal, None, None, None);
            prop_assert_eq!(a.chain_digest, b.chain_digest);
            prop_assert_eq!(digest_canonical(&a), digest_canonical(&b));
            assert_evidence_chain(&a);
        }

        #[test]
        fn changing_seed_changes_chain_digest(seed in 1u64..10_000, bump in 1u64..1000) {
            let input = ComputeInput { frame_id: FrameId(1), t: 2, context_digest: [3; 32] };
            let mk = |s| RiskSignal {
                risk: 0.4,
                confidence: 0.6,
                quality: SignalQuality::Unavailable,
                evidence: EvidenceRef {
                    context_digest: input.context_digest,
                    world_digest: None,
                    spikes_digest: None,
                    ssm_digest: None,
                    lfm_digest: None,
                    backend_profile: BackendProfileId::StubV1,
                    backend_pack_id: crate::BackendPackId(1),
                    fixtures_digest: [9; 32],
                    llm_backend: crate::BackendComponentId::ToyV1,
                    world_backend: crate::BackendComponentId::ToyV1,
                    sae_backend: crate::BackendComponentId::ToyV1,
                    ssm_backend: crate::BackendComponentId::ToyV1,
                    lfm_backend: crate::BackendComponentId::ToyV1,
                    seed: s,
                    budget_profile_id: 9,
                },
                version: 1,
            };
            let a = EvidenceChain::from_compute(&input, &[], &mk(seed), None, None, None);
            let b = EvidenceChain::from_compute(&input, &[], &mk(seed.saturating_add(bump)), None, None, None);
            prop_assert_ne!(a.chain_digest, b.chain_digest);
            assert_evidence_chain(&a);
            assert_evidence_chain(&b);
        }
    }

    #[test]
    fn evidence_chain_changes_for_seed_or_backend() {
        let input = ComputeInput {
            frame_id: FrameId(7),
            t: 5,
            context_digest: [7; 32],
        };
        let mut risk = RiskSignal {
            risk: 0.2,
            confidence: 0.8,
            quality: SignalQuality::Unavailable,
            evidence: EvidenceRef {
                context_digest: input.context_digest,
                world_digest: None,
                spikes_digest: None,
                ssm_digest: None,
                lfm_digest: None,
                backend_profile: BackendProfileId::StubV1,
                backend_pack_id: crate::BackendPackId(1),
                fixtures_digest: [9; 32],
                llm_backend: crate::BackendComponentId::ToyV1,
                world_backend: crate::BackendComponentId::ToyV1,
                sae_backend: crate::BackendComponentId::ToyV1,
                ssm_backend: crate::BackendComponentId::ToyV1,
                lfm_backend: crate::BackendComponentId::ToyV1,
                seed: 11,
                budget_profile_id: 1,
            },
            version: 1,
        };

        let base = EvidenceChain::from_compute(&input, &[], &risk, None, None, None);
        let same = EvidenceChain::from_compute(&input, &[], &risk, None, None, None);
        assert_eq!(base.chain_digest, same.chain_digest);

        risk.evidence.seed = 12;
        let changed_seed = EvidenceChain::from_compute(&input, &[], &risk, None, None, None);
        assert_ne!(base.chain_digest, changed_seed.chain_digest);

        risk.evidence.seed = 11;
        risk.evidence.backend_profile = BackendProfileId::BurnV1;
        let changed_backend = EvidenceChain::from_compute(&input, &[], &risk, None, None, None);
        assert_ne!(base.chain_digest, changed_backend.chain_digest);
    }
}
