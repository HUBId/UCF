#![forbid(unsafe_code)]

use std::sync::Mutex;

use ucf_protocol::v1::spec::ProofEnvelope;
use ucf_types::{EvidenceId, LogicalTime, WallTime};

#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceEnvelope {
    pub evidence_id: EvidenceId,
    pub proof: ProofEnvelope,
    pub logical_time: LogicalTime,
    pub wall_time: WallTime,
}

pub trait EvidenceStore {
    fn append(&self, evidence: EvidenceEnvelope);
    fn list(&self) -> Vec<EvidenceEnvelope>;
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
    fn append(&self, evidence: EvidenceEnvelope) {
        let mut entries = self.entries.lock().expect("lock evidence store");
        entries.push(evidence);
    }

    fn list(&self) -> Vec<EvidenceEnvelope> {
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

        store.append(envelope.clone());
        let entries = store.list();

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], envelope);
    }
}
