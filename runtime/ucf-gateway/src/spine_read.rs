use serde::{Deserialize, Serialize};
use thiserror::Error;
use ucf_archive_store::{ArchiveStore, RecordKind};
use ucf_evidence::EvidenceStore;
use ucf_protocol::decode_experience_record;
use ucf_types::{Digest32, EvidenceId};

pub const SPINE_READ_VERSION: &str = "v1.1";
pub const SPINE_READ_MODE: &str = "read_only";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpineReadHealth {
    pub status: String,
    pub mode: String,
    pub spine_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceReadSummary {
    pub evidence_id: String,
    pub proof_envelope_id: Option<String>,
    pub proof_payload_bytes_len: Option<usize>,
    pub proof_payload_digest_hex: Option<String>,
    pub proof_signature_count: usize,
    pub experience_record_id: Option<String>,
    pub experience_subject_id: Option<String>,
    pub experience_payload_bytes_len: Option<usize>,
    pub candidate_set_record_digest_hex: Option<String>,
    pub output_record_digest_hex: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputEventReadSummary {
    pub key_hex: String,
    pub record_kind: String,
    pub payload_commit_hex: String,
    pub boundary_commit_hex: String,
    pub output_record_digest_hex: String,
    pub root_commit_hex: Option<String>,
    pub cycle_id: u64,
    pub tier: u8,
    pub flags: u16,
    pub payload_bytes_len: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SpineReadError {
    #[error("evidence not found")]
    EvidenceNotFound,
    #[error("archive output event not found")]
    OutputEventNotFound,
    #[error("archive record kind is unsupported for minimal spine readback")]
    UnsupportedRecordKind,
}

pub struct SpineReadService<'a, E: EvidenceStore + ?Sized, A: ArchiveStore + ?Sized> {
    evidence_store: &'a E,
    archive_store: &'a A,
}

impl<'a, E: EvidenceStore + ?Sized, A: ArchiveStore + ?Sized> SpineReadService<'a, E, A> {
    pub fn new(evidence_store: &'a E, archive_store: &'a A) -> Self {
        Self {
            evidence_store,
            archive_store,
        }
    }

    pub fn spine_read_health(&self) -> SpineReadHealth {
        SpineReadHealth {
            status: "ok".to_string(),
            mode: SPINE_READ_MODE.to_string(),
            spine_version: SPINE_READ_VERSION.to_string(),
        }
    }

    pub fn read_evidence(
        &self,
        evidence_id: EvidenceId,
    ) -> Result<EvidenceReadSummary, SpineReadError> {
        let envelope = self
            .evidence_store
            .get(evidence_id)
            .ok_or(SpineReadError::EvidenceNotFound)?;
        let mut summary = EvidenceReadSummary {
            evidence_id: envelope.evidence_id.as_str().to_string(),
            proof_envelope_id: None,
            proof_payload_bytes_len: None,
            proof_payload_digest_hex: None,
            proof_signature_count: 0,
            experience_record_id: None,
            experience_subject_id: None,
            experience_payload_bytes_len: None,
            candidate_set_record_digest_hex: None,
            output_record_digest_hex: None,
        };

        if let Some(proof) = envelope.proof {
            summary.proof_envelope_id = Some(proof.envelope_id);
            summary.proof_payload_bytes_len = Some(proof.payload.len());
            summary.proof_payload_digest_hex = Some(blake3_hex(&proof.payload));
            summary.proof_signature_count = proof.signature_ids.len();

            if let Ok(record) = decode_experience_record(&proof.payload) {
                summary.experience_record_id = Some(record.record_id);
                summary.experience_subject_id = Some(record.subject_id);
                summary.experience_payload_bytes_len = Some(record.payload.len());
                let links = parse_minimal_spine_links(&record.payload);
                summary.candidate_set_record_digest_hex = links.candidate_set_record_digest_hex;
                summary.output_record_digest_hex = links.output_record_digest_hex;
            }
        }

        Ok(summary)
    }

    pub fn read_output_event(
        &self,
        key: Digest32,
    ) -> Result<OutputEventReadSummary, SpineReadError> {
        let record = self
            .archive_store
            .get(key)
            .ok_or(SpineReadError::OutputEventNotFound)?;
        if record.kind != RecordKind::OutputEvent {
            return Err(SpineReadError::UnsupportedRecordKind);
        }
        Ok(OutputEventReadSummary {
            key_hex: digest_hex(record.key),
            record_kind: record_kind_name(record.kind).to_string(),
            payload_commit_hex: digest_hex(record.payload_commit),
            boundary_commit_hex: digest_hex(record.meta.boundary_commit),
            output_record_digest_hex: digest_hex(record.meta.boundary_commit),
            root_commit_hex: self.archive_store.root_commit().map(digest_hex),
            cycle_id: record.meta.cycle_id,
            tier: record.meta.tier,
            flags: record.meta.flags,
            payload_bytes_len: None,
        })
    }
}

#[derive(Default)]
struct MinimalSpineLinks {
    candidate_set_record_digest_hex: Option<String>,
    output_record_digest_hex: Option<String>,
}

fn parse_minimal_spine_links(payload: &[u8]) -> MinimalSpineLinks {
    let Ok(payload) = std::str::from_utf8(payload) else {
        return MinimalSpineLinks::default();
    };
    let mut links = MinimalSpineLinks::default();
    for part in payload.split(';') {
        let Some((key, value)) = part.split_once('=') else {
            continue;
        };
        match key {
            "candidate_set_record_digest" => {
                links.candidate_set_record_digest_hex = Some(value.to_string());
            }
            "output_record_digest" => {
                links.output_record_digest_hex = Some(value.to_string());
            }
            _ => {}
        }
    }
    links
}

fn record_kind_name(kind: RecordKind) -> &'static str {
    match kind {
        RecordKind::OutputEvent => "OutputEvent",
        _ => "Unsupported",
    }
}

fn digest_hex(digest: Digest32) -> String {
    hex::encode(digest.as_bytes())
}

fn blake3_hex(bytes: &[u8]) -> String {
    hex::encode(blake3::hash(bytes).as_bytes())
}
