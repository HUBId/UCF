#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use sha2::Digest;
use ucf_types::{Digest32, DomainDigest, EvidenceId};

pub const MAX_PROOF_BYTES: usize = 512;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoldProof(#[serde(with = "BigArray")] pub [u8; MAX_PROOF_BYTES]);

impl FoldProof {
    pub fn as_bytes(&self) -> &[u8; MAX_PROOF_BYTES] {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoldState {
    pub epoch: u64,
    pub acc: Digest32,
    pub last_evidence: Option<EvidenceId>,
}

impl FoldState {
    pub fn genesis() -> Self {
        Self {
            epoch: 0,
            acc: Digest32::new([0u8; 32]),
            last_evidence: None,
        }
    }
}

pub trait FoldableProof {
    fn fold_step(
        state: &FoldState,
        evidence_digest: DomainDigest,
        prior: Option<&FoldProof>,
    ) -> (FoldState, FoldProof);

    fn verify_step(
        state_before: &FoldState,
        evidence_digest: DomainDigest,
        proof: &FoldProof,
        state_after: &FoldState,
    ) -> bool;
}

pub struct DummyFolder;

impl DummyFolder {
    fn hash_bytes(input: &[u8]) -> [u8; 32] {
        let digest = sha2::Sha256::digest(input);
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest);
        out
    }

    fn domain_digest_bytes(digest: &DomainDigest) -> Vec<u8> {
        let mut out = Vec::with_capacity(36);
        out.extend_from_slice(&digest.algo.id().to_le_bytes());
        out.extend_from_slice(&digest.domain.to_le_bytes());
        out.extend_from_slice(digest.digest.as_bytes());
        out
    }

    fn fold_acc(
        state: &FoldState,
        evidence_digest: &DomainDigest,
        prior_prefix: &[u8; 32],
    ) -> [u8; 32] {
        let mut input = Vec::with_capacity(2 + 36 + 32 + 32);
        input.extend_from_slice(&evidence_digest.domain.to_le_bytes());
        input.extend_from_slice(&Self::domain_digest_bytes(evidence_digest));
        input.extend_from_slice(prior_prefix);
        input.extend_from_slice(state.acc.as_bytes());
        Self::hash_bytes(&input)
    }

    fn expand_proof_bytes(seed: &[u8; 32], prior_prefix: &[u8; 32]) -> FoldProof {
        let mut out = [0u8; MAX_PROOF_BYTES];
        out[..32].copy_from_slice(prior_prefix);
        let mut offset = 32;
        let mut counter: u32 = 0;
        while offset < MAX_PROOF_BYTES {
            let mut input = Vec::with_capacity(32 + 4);
            input.extend_from_slice(seed);
            input.extend_from_slice(&counter.to_le_bytes());
            let chunk = Self::hash_bytes(&input);
            let remaining = MAX_PROOF_BYTES - offset;
            let to_copy = remaining.min(chunk.len());
            out[offset..offset + to_copy].copy_from_slice(&chunk[..to_copy]);
            offset += to_copy;
            counter += 1;
        }
        FoldProof(out)
    }
}

impl FoldableProof for DummyFolder {
    fn fold_step(
        state: &FoldState,
        evidence_digest: DomainDigest,
        prior: Option<&FoldProof>,
    ) -> (FoldState, FoldProof) {
        let mut prior_prefix = [0u8; 32];
        if let Some(prior) = prior {
            prior_prefix.copy_from_slice(&prior.0[..32]);
        }
        let new_acc = Self::fold_acc(state, &evidence_digest, &prior_prefix);
        let proof = Self::expand_proof_bytes(&new_acc, &prior_prefix);
        let new_state = FoldState {
            epoch: state.epoch + 1,
            acc: Digest32::new(new_acc),
            last_evidence: state.last_evidence.clone(),
        };
        (new_state, proof)
    }

    fn verify_step(
        state_before: &FoldState,
        evidence_digest: DomainDigest,
        proof: &FoldProof,
        state_after: &FoldState,
    ) -> bool {
        let mut prior_prefix = [0u8; 32];
        prior_prefix.copy_from_slice(&proof.0[..32]);
        let expected_acc = Self::fold_acc(state_before, &evidence_digest, &prior_prefix);
        let expected_proof = Self::expand_proof_bytes(&expected_acc, &prior_prefix);
        state_after.epoch == state_before.epoch + 1
            && state_after.acc == Digest32::new(expected_acc)
            && *proof == expected_proof
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_types::{AlgoId, DomainDigest};

    fn sample_digest(input: &[u8]) -> DomainDigest {
        let digest = DummyFolder::hash_bytes(input);
        let digest = Digest32::new(digest);
        DomainDigest::new(AlgoId::Sha256, 1, digest).expect("domain digest")
    }

    #[test]
    fn proof_size_constant() {
        let state = FoldState::genesis();
        let (next, proof) = DummyFolder::fold_step(&state, sample_digest(b"a"), None);
        assert_eq!(proof.as_bytes().len(), MAX_PROOF_BYTES);
        assert_eq!(next.epoch, 1);
    }

    #[test]
    fn verify_step_accepts_generated_proof() {
        let state = FoldState::genesis();
        let digest = sample_digest(b"b");
        let (next, proof) = DummyFolder::fold_step(&state, digest.clone(), None);
        assert!(DummyFolder::verify_step(&state, digest, &proof, &next));
    }

    #[test]
    fn successive_folds_change_acc() {
        let state = FoldState::genesis();
        let (next, proof) = DummyFolder::fold_step(&state, sample_digest(b"c"), None);
        let (next2, proof2) = DummyFolder::fold_step(&next, sample_digest(b"d"), Some(&proof));
        assert_ne!(next.acc, next2.acc);
        assert_eq!(proof2.as_bytes().len(), MAX_PROOF_BYTES);
    }
}
