#![forbid(unsafe_code)]
//! Dieses Archiv speichert Evidence in zwei Dateien: `evidence.log` enthält die
//! binären Payloads, während `evidence.manifest` Metadaten (Offset, Länge und
//! Hash) zu jedem Eintrag hält.
//!
//! Invarianten: Jeder Manifest-Eintrag muss auf einen gültigen Bereich im Log
//! zeigen, und Offset/Länge/Hash müssen exakt mit dem entsprechenden Log-Chunk
//! übereinstimmen. So wird sichergestellt, dass das Manifest die Log-Datei
//! korrekt beschreibt und Integrität geprüft werden kann.
//!
//! Die Abstraktion orientiert sich bereits an zukünftigen Proof-Envelopes: Die
//! gespeicherten Payloads werden als `ProofEnvelope` behandelt, sodass spätere
//! Erweiterungen der Envelope-Struktur ohne Änderung des Archivformats möglich
//! bleiben.

use std::path::Path;

use prost::Message;
use ucf_evidence::file_store::FileEvidenceStore;
use ucf_evidence::{EvidenceEnvelope, EvidenceStore, InMemoryEvidenceStore, StoreResult};
use ucf_types::v1::spec::{ExperienceRecord, ProofEnvelope};
use ucf_types::{EvidenceId, LogicalTime, WallTime};

pub trait ExperienceAppender {
    fn append_with_proof(&self, rec: ExperienceRecord, proof: Option<ProofEnvelope>) -> EvidenceId;

    fn append(&self, rec: ExperienceRecord) -> EvidenceId {
        self.append_with_proof(rec, None)
    }
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

pub struct FileArchive {
    store: FileEvidenceStore,
}

impl FileArchive {
    pub fn open(path: impl AsRef<Path>) -> StoreResult<Self> {
        let path = path.as_ref();
        let log_path = path.join("evidence.log");
        let manifest_path = path.join("evidence.manifest");
        let store = FileEvidenceStore::open(log_path, manifest_path)?;
        Ok(Self { store })
    }
}

fn append_record(
    store: &dyn EvidenceStore,
    rec: ExperienceRecord,
    proof: Option<ProofEnvelope>,
) -> EvidenceId {
    let evidence_id = EvidenceId::new(rec.record_id.clone());
    let proof = proof.or_else(|| {
        Some(ProofEnvelope {
            envelope_id: rec.record_id.clone(),
            payload: rec.encode_to_vec(),
            payload_digest: None,
            vrf_tags: Vec::new(),
            signature_ids: Vec::new(),
        })
    });
    let envelope = EvidenceEnvelope {
        evidence_id: evidence_id.clone(),
        proof,
        logical_time: LogicalTime::new(0),
        wall_time: WallTime::new(rec.observed_at_ms),
    };

    store.append(envelope)
}

impl ExperienceAppender for InMemoryArchive {
    fn append_with_proof(&self, rec: ExperienceRecord, proof: Option<ProofEnvelope>) -> EvidenceId {
        append_record(&self.store, rec, proof)
    }
}

impl ExperienceAppender for FileArchive {
    fn append_with_proof(&self, rec: ExperienceRecord, proof: Option<ProofEnvelope>) -> EvidenceId {
        append_record(&self.store, rec, proof)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

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
            proof_ref: None,
        };

        let id = archive.append(record);
        let entries = archive.list();

        assert_eq!(id, EvidenceId::new("rec-1"));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].evidence_id, id);
    }

    #[test]
    fn append_with_proof_overrides_envelope() {
        let archive = InMemoryArchive::new();
        let record = ExperienceRecord {
            record_id: "rec-2".to_string(),
            observed_at_ms: 1_700_000_000_123,
            subject_id: "subject-2".to_string(),
            payload: vec![9, 8, 7],
            digest: None,
            vrf_tag: None,
            proof_ref: None,
        };
        let proof = ProofEnvelope {
            envelope_id: "proof-override".to_string(),
            payload: vec![4, 5, 6],
            payload_digest: None,
            vrf_tags: Vec::new(),
            signature_ids: vec!["sig-1".to_string()],
        };

        let id = archive.append_with_proof(record, Some(proof.clone()));
        let entries = archive.list();

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].evidence_id, id);
        assert_eq!(entries[0].proof, Some(proof));
    }

    #[test]
    fn file_archive_persists_proof_envelope() {
        let temp_dir = TempDir::new().expect("temp dir");
        let archive = FileArchive::open(temp_dir.path()).expect("open archive");
        let record = ExperienceRecord {
            record_id: "rec-3".to_string(),
            observed_at_ms: 1_700_000_000_456,
            subject_id: "subject-3".to_string(),
            payload: vec![1, 1, 2, 3],
            digest: None,
            vrf_tag: None,
            proof_ref: None,
        };
        let proof = ProofEnvelope {
            envelope_id: "proof-rec-3".to_string(),
            payload: vec![9, 9, 9],
            payload_digest: None,
            vrf_tags: Vec::new(),
            signature_ids: vec!["sig-2".to_string()],
        };

        let evidence_id = archive.append_with_proof(record, Some(proof.clone()));
        drop(archive);

        let reopened = FileArchive::open(temp_dir.path()).expect("reopen archive");
        let stored = reopened
            .store
            .get(evidence_id.clone())
            .expect("load evidence");

        assert_eq!(stored.proof, Some(proof));
    }
}
