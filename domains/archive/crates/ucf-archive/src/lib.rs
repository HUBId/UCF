#![forbid(unsafe_code)]

use prost::Message;
use ucf_evidence::{EvidenceEnvelope, InMemoryEvidenceStore};
use ucf_types::v1::spec::{ExperienceRecord, ProofEnvelope};
use ucf_types::{EvidenceId, LogicalTime, WallTime};

pub trait ExperienceAppender {
    fn append(&self, rec: ExperienceRecord) -> EvidenceId;
}

#[derive(Default)]
pub struct InMemoryArchive {
    store: InMemoryEvidenceStore,
}

impl InMemoryArchive {
    pub fn new() -> Self {
        Self {
            store: InMemoryEvidenceStore::new(),
        }
    }

    pub fn list(&self) -> Vec<EvidenceEnvelope> {
        self.store.list()
    }
}

impl ExperienceAppender for InMemoryArchive {
    fn append(&self, rec: ExperienceRecord) -> EvidenceId {
        let evidence_id = EvidenceId::new(rec.record_id.clone());
        let envelope = EvidenceEnvelope {
            evidence_id: evidence_id.clone(),
            proof: ProofEnvelope {
                envelope_id: rec.record_id.clone(),
                payload: rec.encode_to_vec(),
                payload_digest: None,
                vrf_tags: Vec::new(),
                signature_ids: Vec::new(),
            },
            logical_time: LogicalTime::new(0),
            wall_time: WallTime::new(rec.observed_at_ms),
        };

        self.store.append(envelope)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_returns_id_and_stores_record() {
        let archive = InMemoryArchive::new();
        let record = ExperienceRecord {
            record_id: "rec-1".to_string(),
            observed_at_ms: 1_700_000_000_000,
            subject_id: "subject-1".to_string(),
            payload: vec![1, 2, 3],
            digest: None,
            vrf_tag: None,
        };

        let id = archive.append(record);
        let entries = archive.list();

        assert_eq!(id, EvidenceId::new("rec-1"));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].evidence_id, id);
    }
}
