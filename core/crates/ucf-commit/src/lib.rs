#![forbid(unsafe_code)]

use ucf_protocol::v1::spec::{
    ControlFrame, Digest as ProtoDigest, ExperienceRecord, MacroMilestone, MesoMilestone,
    MicroMilestone, PolicyDecision, ProofRef, VrfTag,
};
use ucf_types::{AlgoId, Digest32, DomainDigest};

/// Domain identifiers for canonical commitments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum CommitDomain {
    ExperienceRecord = 1,
    MicroMilestone = 2,
    MesoMilestone = 3,
    MacroMilestone = 4,
    PolicyDecision = 5,
    ProofEnvelope = 6,
    ControlFrame = 7,
}

impl CommitDomain {
    pub fn id(self) -> u16 {
        self as u16
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Commitment {
    pub domain: u16,
    pub algo: AlgoId,
    pub digest: Digest32,
}

impl Commitment {
    pub fn to_domain_digest(&self) -> DomainDigest {
        DomainDigest::new(self.algo, self.domain, self.digest).expect("valid domain digest")
    }
}

/// Canonical commitment for an ExperienceRecord.
///
/// Encoding rules:
/// - Fields are encoded in numeric order.
/// - Each field is encoded as: tag (u16 BE) + length (u32 BE) + value bytes.
/// - Scalars use fixed-width big-endian bytes.
/// - Optional fields are encoded as: presence (u8) + value (if present).
/// - Repeated fields are encoded as: count (u32 BE) + each value with length prefix.
/// - Repeated identifiers with no semantic order are sorted lexicographically.
pub fn commit_experience_record(rec: &ExperienceRecord) -> Commitment {
    let bytes = encode_experience_record(rec);
    Commitment {
        domain: CommitDomain::ExperienceRecord.id(),
        algo: AlgoId::Blake3_256,
        digest: hash_bytes(&bytes),
    }
}

pub fn commit_policy_decision(dec: &PolicyDecision) -> Commitment {
    let bytes = encode_policy_decision(dec);
    Commitment {
        domain: CommitDomain::PolicyDecision.id(),
        algo: AlgoId::Blake3_256,
        digest: hash_bytes(&bytes),
    }
}

pub fn commit_control_frame(cf: &ControlFrame) -> Commitment {
    let bytes = encode_control_frame(cf);
    Commitment {
        domain: CommitDomain::ControlFrame.id(),
        algo: AlgoId::Blake3_256,
        digest: hash_bytes(&bytes),
    }
}

pub fn canonical_control_frame_len(cf: &ControlFrame) -> usize {
    encode_control_frame(cf).len()
}

pub fn commit_milestone_micro(micro: &MicroMilestone) -> Commitment {
    let bytes = encode_milestone_micro(micro);
    Commitment {
        domain: CommitDomain::MicroMilestone.id(),
        algo: AlgoId::Blake3_256,
        digest: hash_bytes(&bytes),
    }
}

pub fn commit_milestone_meso(meso: &MesoMilestone) -> Commitment {
    let bytes = encode_milestone_meso(meso);
    Commitment {
        domain: CommitDomain::MesoMilestone.id(),
        algo: AlgoId::Blake3_256,
        digest: hash_bytes(&bytes),
    }
}

pub fn commit_milestone_macro(macro_ms: &MacroMilestone) -> Commitment {
    let bytes = encode_milestone_macro(macro_ms);
    Commitment {
        domain: CommitDomain::MacroMilestone.id(),
        algo: AlgoId::Blake3_256,
        digest: hash_bytes(&bytes),
    }
}

fn hash_bytes(input: &[u8]) -> Digest32 {
    let digest = blake3::hash(input);
    Digest32::new(*digest.as_bytes())
}

fn encode_experience_record(rec: &ExperienceRecord) -> Vec<u8> {
    let mut enc = Encoder::new();
    enc.write_u16(CommitDomain::ExperienceRecord.id());
    enc.write_field(1, encode_string(&rec.record_id));
    enc.write_field(2, encode_u64(rec.observed_at_ms));
    enc.write_field(3, encode_string(&rec.subject_id));
    enc.write_field(4, encode_bytes(&rec.payload));
    enc.write_field(5, encode_optional(rec.digest.as_ref(), encode_digest));
    enc.write_field(6, encode_optional(rec.vrf_tag.as_ref(), encode_vrf_tag));
    enc.write_field(7, encode_optional(rec.proof_ref.as_ref(), encode_proof_ref));
    enc.into_bytes()
}

fn encode_policy_decision(dec: &PolicyDecision) -> Vec<u8> {
    let mut enc = Encoder::new();
    enc.write_u16(CommitDomain::PolicyDecision.id());
    enc.write_field(1, encode_i32(dec.kind));
    enc.write_field(2, encode_i32(dec.action));
    enc.write_field(3, encode_string(&dec.rationale));
    enc.write_field(4, encode_u32(dec.confidence_bp));
    enc.write_field(5, encode_string_list_sorted(&dec.constraint_ids));
    enc.into_bytes()
}

fn encode_milestone_micro(micro: &MicroMilestone) -> Vec<u8> {
    let mut enc = Encoder::new();
    enc.write_u16(CommitDomain::MicroMilestone.id());
    enc.write_field(1, encode_string(&micro.milestone_id));
    enc.write_field(2, encode_u64(micro.achieved_at_ms));
    enc.write_field(3, encode_string(&micro.label));
    enc.into_bytes()
}

fn encode_milestone_meso(meso: &MesoMilestone) -> Vec<u8> {
    let mut enc = Encoder::new();
    enc.write_u16(CommitDomain::MesoMilestone.id());
    enc.write_field(1, encode_string(&meso.milestone_id));
    enc.write_field(2, encode_u64(meso.achieved_at_ms));
    enc.write_field(3, encode_string(&meso.label));
    enc.write_field(4, encode_string_list_sorted(&meso.micro_milestone_ids));
    enc.into_bytes()
}

fn encode_milestone_macro(macro_ms: &MacroMilestone) -> Vec<u8> {
    let mut enc = Encoder::new();
    enc.write_u16(CommitDomain::MacroMilestone.id());
    enc.write_field(1, encode_string(&macro_ms.milestone_id));
    enc.write_field(2, encode_u64(macro_ms.achieved_at_ms));
    enc.write_field(3, encode_string(&macro_ms.label));
    enc.write_field(4, encode_string_list_sorted(&macro_ms.meso_milestone_ids));
    enc.into_bytes()
}

fn encode_control_frame(cf: &ControlFrame) -> Vec<u8> {
    let mut enc = Encoder::new();
    enc.write_u16(CommitDomain::ControlFrame.id());
    enc.write_field(1, encode_string(&cf.frame_id));
    enc.write_field(2, encode_u64(cf.issued_at_ms));
    enc.write_field(
        3,
        encode_optional(cf.decision.as_ref(), encode_policy_decision),
    );
    enc.write_field(4, encode_string_list_sorted(&cf.evidence_ids));
    enc.write_field(5, encode_string(&cf.policy_id));
    enc.into_bytes()
}

fn encode_digest(digest: &ProtoDigest) -> Vec<u8> {
    let mut enc = Encoder::new();
    enc.write_field(1, encode_string(&digest.algorithm));
    enc.write_field(2, encode_bytes(&digest.value));
    enc.write_field(
        3,
        encode_optional(digest.algo_id.as_ref(), |value| encode_u32(*value)),
    );
    enc.write_field(
        4,
        encode_optional(digest.domain.as_ref(), |value| encode_u32(*value)),
    );
    enc.write_field(
        5,
        encode_optional(digest.value_32.as_ref(), |value| encode_bytes(value)),
    );
    enc.into_bytes()
}

fn encode_vrf_tag(tag: &VrfTag) -> Vec<u8> {
    let mut enc = Encoder::new();
    enc.write_field(1, encode_string(&tag.algorithm));
    enc.write_field(2, encode_bytes(&tag.proof));
    enc.write_field(3, encode_bytes(&tag.output));
    enc.write_field(
        4,
        encode_optional(tag.suite_id.as_ref(), |value| encode_u32(*value)),
    );
    enc.write_field(
        5,
        encode_optional(tag.domain.as_ref(), |value| encode_u32(*value)),
    );
    enc.write_field(
        6,
        encode_optional(tag.tag.as_ref(), |value| encode_bytes(value)),
    );
    enc.into_bytes()
}

fn encode_proof_ref(proof: &ProofRef) -> Vec<u8> {
    let mut enc = Encoder::new();
    enc.write_field(1, encode_string(&proof.proof_id));
    enc.write_field(
        2,
        encode_optional(proof.algo_id.as_ref(), |value| encode_u32(*value)),
    );
    enc.write_field(
        3,
        encode_optional(proof.suite_id.as_ref(), |value| encode_u32(*value)),
    );
    enc.write_field(
        4,
        encode_optional(proof.opaque.as_ref(), |value| encode_bytes(value)),
    );
    enc.into_bytes()
}

fn encode_optional<T>(value: Option<&T>, encode: impl FnOnce(&T) -> Vec<u8>) -> Vec<u8> {
    let mut out = Vec::new();
    match value {
        Some(value) => {
            out.push(1);
            let encoded = encode(value);
            out.extend_from_slice(&encoded.len().to_be_bytes());
            out.extend_from_slice(&encoded);
        }
        None => {
            out.push(0);
        }
    }
    out
}

fn encode_string(value: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + value.len());
    out.extend_from_slice(&(value.len() as u32).to_be_bytes());
    out.extend_from_slice(value.as_bytes());
    out
}

fn encode_bytes(value: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + value.len());
    out.extend_from_slice(&(value.len() as u32).to_be_bytes());
    out.extend_from_slice(value);
    out
}

fn encode_string_list_sorted(values: &[String]) -> Vec<u8> {
    let mut sorted = values.to_vec();
    sorted.sort();
    encode_string_list(&sorted)
}

fn encode_string_list(values: &[String]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(values.len() as u32).to_be_bytes());
    for value in values {
        let encoded = encode_string(value);
        out.extend_from_slice(&encoded);
    }
    out
}

fn encode_u32(value: u32) -> Vec<u8> {
    value.to_be_bytes().to_vec()
}

fn encode_u64(value: u64) -> Vec<u8> {
    value.to_be_bytes().to_vec()
}

fn encode_i32(value: i32) -> Vec<u8> {
    value.to_be_bytes().to_vec()
}

struct Encoder {
    bytes: Vec<u8>,
}

impl Encoder {
    fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    fn write_u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn write_field(&mut self, tag: u16, payload: Vec<u8>) {
        self.write_u16(tag);
        self.bytes
            .extend_from_slice(&(payload.len() as u32).to_be_bytes());
        self.bytes.extend_from_slice(&payload);
    }

    fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record() -> ExperienceRecord {
        ExperienceRecord {
            record_id: "rec-1".to_string(),
            observed_at_ms: 1_700_000_000_000,
            subject_id: "subject-1".to_string(),
            payload: vec![1, 2, 3],
            digest: None,
            vrf_tag: None,
            proof_ref: None,
        }
    }

    #[test]
    fn experience_record_commitment_is_deterministic() {
        let record = sample_record();
        let first = commit_experience_record(&record);
        let second = commit_experience_record(&record);
        assert_eq!(first, second);
    }

    #[test]
    fn experience_record_commitment_changes_on_payload() {
        let mut record = sample_record();
        let baseline = commit_experience_record(&record);
        record.payload.push(9);
        let updated = commit_experience_record(&record);
        assert_ne!(baseline, updated);
    }

    #[test]
    fn milestone_commitment_sorts_child_ids() {
        let meso_a = MesoMilestone {
            milestone_id: "m1".to_string(),
            achieved_at_ms: 42,
            label: "meso".to_string(),
            micro_milestone_ids: vec!["b".to_string(), "a".to_string()],
        };
        let meso_b = MesoMilestone {
            milestone_id: "m1".to_string(),
            achieved_at_ms: 42,
            label: "meso".to_string(),
            micro_milestone_ids: vec!["a".to_string(), "b".to_string()],
        };
        let commit_a = commit_milestone_meso(&meso_a);
        let commit_b = commit_milestone_meso(&meso_b);
        assert_eq!(commit_a, commit_b);
    }
}
