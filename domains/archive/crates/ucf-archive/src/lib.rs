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

fn append_record(store: &dyn EvidenceStore, rec: ExperienceRecord) -> EvidenceId {
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

    store.append(envelope)
}

impl ExperienceAppender for InMemoryArchive {
    fn append(&self, rec: ExperienceRecord) -> EvidenceId {
        append_record(&self.store, rec)
    }
}

impl ExperienceAppender for FileArchive {
    fn append(&self, rec: ExperienceRecord) -> EvidenceId {
        append_record(&self.store, rec)
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
