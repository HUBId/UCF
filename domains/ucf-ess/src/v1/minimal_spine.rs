use std::collections::BTreeMap;

use sha2::{Digest, Sha256};
use ucf_types::{Digest32, EvidenceId};

pub const MINIMAL_SPINE_ESS_PROJECTION_VERSION: u16 = 1;
pub const MINIMAL_SPINE_ESS_SOURCE: &str = "minimal_spine_v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinimalSpineEssProjection {
    pub version: u16,
    pub evidence_id: EvidenceId,
    pub input_digest: Digest32,
    pub candidate_set_record_digest: Digest32,
    pub output_record_digest: Digest32,
    pub archive_output_key: Digest32,
    pub policy_status: String,
    pub output_status: String,
    pub source: String,
}

impl MinimalSpineEssProjection {
    pub fn from_canonical_links(
        evidence_id: EvidenceId,
        input_digest: Digest32,
        candidate_set_record_digest: Digest32,
        output_record_digest: Digest32,
        archive_output_key: Digest32,
        policy_status: impl Into<String>,
        output_status: impl Into<String>,
    ) -> Self {
        Self {
            version: MINIMAL_SPINE_ESS_PROJECTION_VERSION,
            evidence_id,
            input_digest,
            candidate_set_record_digest,
            output_record_digest,
            archive_output_key,
            policy_status: policy_status.into(),
            output_status: output_status.into(),
            source: MINIMAL_SPINE_ESS_SOURCE.to_string(),
        }
    }

    pub fn projection_digest(&self) -> Digest32 {
        let mut hasher = Sha256::new();
        hasher.update(b"UCF:ESS:MINIMAL_SPINE:PROJECTION:v1");
        hasher.update(self.deterministic_bytes());
        Digest32::new(hasher.finalize().into())
    }

    fn deterministic_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_u16(&mut out, self.version);
        push_str(&mut out, self.evidence_id.as_str());
        push_digest(&mut out, self.input_digest);
        push_digest(&mut out, self.candidate_set_record_digest);
        push_digest(&mut out, self.output_record_digest);
        push_digest(&mut out, self.archive_output_key);
        push_str(&mut out, &self.policy_status);
        push_str(&mut out, &self.output_status);
        push_str(&mut out, &self.source);
        out
    }
}

#[derive(Debug, Default, Clone)]
pub struct MinimalSpineEssReadModel {
    projections: Vec<MinimalSpineEssProjection>,
    by_output_digest: BTreeMap<[u8; 32], usize>,
    by_evidence_id: BTreeMap<String, usize>,
    by_archive_output_key: BTreeMap<[u8; 32], usize>,
}

impl MinimalSpineEssReadModel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_projection(projection: MinimalSpineEssProjection) -> Self {
        let mut model = Self::new();
        model.project(projection);
        model
    }

    pub fn project(&mut self, projection: MinimalSpineEssProjection) {
        let idx = self.projections.len();
        self.by_output_digest
            .insert(*projection.output_record_digest.as_bytes(), idx);
        self.by_evidence_id
            .insert(projection.evidence_id.as_str().to_string(), idx);
        self.by_archive_output_key
            .insert(*projection.archive_output_key.as_bytes(), idx);
        self.projections.push(projection);
    }

    pub fn get_by_output_digest(
        &self,
        output_record_digest: Digest32,
    ) -> Option<&MinimalSpineEssProjection> {
        self.by_output_digest
            .get(output_record_digest.as_bytes())
            .and_then(|idx| self.projections.get(*idx))
    }

    pub fn get_by_evidence_id(
        &self,
        evidence_id: &EvidenceId,
    ) -> Option<&MinimalSpineEssProjection> {
        self.by_evidence_id
            .get(evidence_id.as_str())
            .and_then(|idx| self.projections.get(*idx))
    }

    pub fn get_by_archive_output_key(
        &self,
        archive_output_key: Digest32,
    ) -> Option<&MinimalSpineEssProjection> {
        self.by_archive_output_key
            .get(archive_output_key.as_bytes())
            .and_then(|idx| self.projections.get(*idx))
    }

    pub fn len(&self) -> usize {
        self.projections.len()
    }

    pub fn is_empty(&self) -> bool {
        self.projections.is_empty()
    }
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn push_str(out: &mut Vec<u8>, value: &str) {
    push_u32(out, value.len() as u32);
    out.extend_from_slice(value.as_bytes());
}

fn push_digest(out: &mut Vec<u8>, value: Digest32) {
    out.extend_from_slice(value.as_bytes());
}
