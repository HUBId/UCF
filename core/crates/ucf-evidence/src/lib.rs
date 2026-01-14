#![forbid(unsafe_code)]

use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use ucf_protocol::v1::spec::ProofEnvelope;
use ucf_types::{EvidenceId, LogicalTime, WallTime};

pub mod file_store;

#[derive(Clone, Debug)]
pub enum StoreError {
    Unsupported(String),
    Corrupt {
        evidence_id: EvidenceId,
        offset: u64,
        expected_hash: AppendLogHash,
        actual_hash: AppendLogHash,
    },
    IOError(String),
}

pub type StoreResult<T> = Result<T, StoreError>;

pub type AppendLogHash = Vec<u8>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvidenceEnvelope {
    pub evidence_id: EvidenceId,
    pub proof: ProofEnvelope,
    pub logical_time: LogicalTime,
    pub wall_time: WallTime,
}

pub trait EvidenceStore {
    fn append(&self, evidence: EvidenceEnvelope) -> EvidenceId;

    fn get(&self, evidence_id: EvidenceId) -> Option<EvidenceEnvelope> {
        let _ = evidence_id;
        None
    }

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait AppendLog {
    fn append_bytes(&self, bytes: &[u8]) -> (u64, usize, AppendLogHash);
    fn read_at(&self, offset: u64, len: usize) -> Vec<u8>;
}

#[derive(Default)]
pub struct InMemoryEvidenceStore {
    entries: Mutex<Vec<EvidenceEnvelope>>,
}

impl InMemoryEvidenceStore {
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
        }
    }
}

impl EvidenceStore for InMemoryEvidenceStore {
    fn append(&self, evidence: EvidenceEnvelope) -> EvidenceId {
        let evidence_id = evidence.evidence_id.clone();
        let mut entries = self.entries.lock().expect("lock evidence store");
        entries.push(evidence);
        evidence_id
    }

    fn get(&self, evidence_id: EvidenceId) -> Option<EvidenceEnvelope> {
        let entries = self.entries.lock().expect("lock evidence store");
        entries
            .iter()
            .find(|entry| entry.evidence_id == evidence_id)
            .cloned()
    }

    fn len(&self) -> usize {
        let entries = self.entries.lock().expect("lock evidence store");
        entries.len()
    }
}

impl InMemoryEvidenceStore {
    pub fn list(&self) -> Vec<EvidenceEnvelope> {
        let entries = self.entries.lock().expect("lock evidence store");
        entries.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evidence_store_appends() {
        let store = InMemoryEvidenceStore::new();
        let envelope = EvidenceEnvelope {
            evidence_id: EvidenceId::new("evidence-1"),
            proof: ProofEnvelope {
                envelope_id: "proof-1".to_string(),
                payload: vec![1, 2, 3],
                payload_digest: None,
                vrf_tags: Vec::new(),
                signature_ids: Vec::new(),
            },
            logical_time: LogicalTime::new(5),
            wall_time: WallTime::new(1_700_000_000_000),
        };

        let stored_id = store.append(envelope.clone());

        assert_eq!(stored_id, envelope.evidence_id);
        assert_eq!(store.len(), 1);
        assert_eq!(store.get(stored_id), Some(envelope));
    }
}
