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

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use bincode::{deserialize, serialize};
use prost::Message;
use serde::{Deserialize, Serialize};
use ucf_commit::commit_experience_record;
use ucf_evidence::file_store::FileEvidenceStore;
use ucf_evidence::{EvidenceEnvelope, EvidenceStore, InMemoryEvidenceStore, StoreResult};
use ucf_fold::{DummyFolder, FoldProof, FoldState, FoldableProof, MAX_PROOF_BYTES};
use ucf_types::v1::spec::{ExperienceRecord, ProofEnvelope};
use ucf_types::{Digest32, DomainDigest, EvidenceId, LogicalTime, WallTime};

const FOLD_SNAPSHOT_FILE: &str = "fold_state.bin";

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FoldSnapshot {
    state: FoldState,
    last_proof: Option<FoldProof>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FoldSnapshotData {
    state: FoldState,
    last_proof: Option<Vec<u8>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FoldSnapshotFile {
    data: FoldSnapshotData,
    checksum: [u8; 32],
}

impl FoldSnapshot {
    fn new() -> Self {
        Self {
            state: FoldState::genesis(),
            last_proof: None,
        }
    }
}

pub trait ExperienceAppender {
    fn append_with_proof(&self, rec: ExperienceRecord, proof: Option<ProofEnvelope>) -> EvidenceId;

    fn append(&self, rec: ExperienceRecord) -> EvidenceId {
        self.append_with_proof(rec, None)
    }
}

pub struct InMemoryArchive {
    store: InMemoryEvidenceStore,
    fold_state: Mutex<FoldSnapshot>,
}

impl InMemoryArchive {
    pub fn new() -> Self {
        Self {
            store: InMemoryEvidenceStore::new(),
            fold_state: Mutex::new(FoldSnapshot::new()),
        }
    }

    pub fn list(&self) -> Vec<EvidenceEnvelope> {
        self.store.list()
    }

    pub fn append_and_fold(&self, rec: ExperienceRecord) -> (EvidenceId, FoldState) {
        let evidence_digest = compute_evidence_digest(&rec);
        let mut snapshot = self.fold_state.lock().expect("lock fold state");
        let (next_state, fold_proof) = DummyFolder::fold_step(
            &snapshot.state,
            evidence_digest,
            snapshot.last_proof.as_ref(),
        );
        let evidence_id = append_record(&self.store, rec, None, Some(fold_proof.clone()));
        let mut updated_state = next_state;
        updated_state.last_evidence = Some(evidence_id.clone());
        snapshot.state = updated_state.clone();
        snapshot.last_proof = Some(fold_proof);
        (evidence_id, updated_state)
    }
}

impl Default for InMemoryArchive {
    fn default() -> Self {
        Self::new()
    }
}

pub struct FileArchive {
    store: FileEvidenceStore,
    fold_path: PathBuf,
    fold_state: Mutex<FoldSnapshot>,
}

impl FileArchive {
    pub fn open(path: impl AsRef<Path>) -> StoreResult<Self> {
        let path = path.as_ref();
        let log_path = path.join("evidence.log");
        let manifest_path = path.join("evidence.manifest");
        let fold_path = path.join(FOLD_SNAPSHOT_FILE);
        let store = FileEvidenceStore::open(log_path, manifest_path)?;
        let fold_state = load_fold_snapshot(&fold_path)?;
        Ok(Self {
            store,
            fold_path,
            fold_state: Mutex::new(fold_state),
        })
    }

    pub fn append_and_fold(&self, rec: ExperienceRecord) -> StoreResult<(EvidenceId, FoldState)> {
        let evidence_digest = compute_evidence_digest(&rec);
        let mut snapshot = self.fold_state.lock().expect("lock fold state");
        let (next_state, fold_proof) = DummyFolder::fold_step(
            &snapshot.state,
            evidence_digest,
            snapshot.last_proof.as_ref(),
        );
        let evidence_id = append_record(&self.store, rec, None, Some(fold_proof.clone()));
        let mut updated_state = next_state;
        updated_state.last_evidence = Some(evidence_id.clone());
        snapshot.state = updated_state.clone();
        snapshot.last_proof = Some(fold_proof);
        persist_fold_snapshot(&self.fold_path, &snapshot)?;
        Ok((evidence_id, updated_state))
    }
}

fn append_record(
    store: &dyn EvidenceStore,
    rec: ExperienceRecord,
    proof: Option<ProofEnvelope>,
    fold_proof: Option<FoldProof>,
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
        fold_proof,
        logical_time: LogicalTime::new(0),
        wall_time: WallTime::new(rec.observed_at_ms),
    };

    store.append(envelope)
}

fn compute_evidence_digest(rec: &ExperienceRecord) -> DomainDigest {
    let commitment = commit_experience_record(rec);
    commitment.to_domain_digest()
}

fn load_fold_snapshot(path: &Path) -> StoreResult<FoldSnapshot> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(FoldSnapshot::new()),
        Err(err) => {
            return Err(ucf_evidence::StoreError::IOError(format!(
                "read fold snapshot: {err}"
            )))
        }
    };
    if bytes.is_empty() {
        return Err(ucf_evidence::StoreError::IOError(
            "empty fold snapshot".to_string(),
        ));
    }
    let snapshot_file: FoldSnapshotFile = deserialize(&bytes)
        .map_err(|err| ucf_evidence::StoreError::IOError(format!("decode fold snapshot: {err}")))?;
    let data_bytes = serialize(&snapshot_file.data)
        .map_err(|err| ucf_evidence::StoreError::IOError(format!("encode fold snapshot: {err}")))?;
    let expected = blake3::hash(&data_bytes);
    if *expected.as_bytes() != snapshot_file.checksum {
        return Err(ucf_evidence::StoreError::IOError(
            "fold snapshot checksum mismatch".to_string(),
        ));
    }
    if snapshot_file.data.state.acc.as_bytes().len() != Digest32::LEN {
        return Err(ucf_evidence::StoreError::IOError(
            "fold snapshot digest length invalid".to_string(),
        ));
    }
    let last_proof = if let Some(bytes) = snapshot_file.data.last_proof {
        if bytes.len() != MAX_PROOF_BYTES {
            return Err(ucf_evidence::StoreError::IOError(format!(
                "fold snapshot proof length invalid: {}",
                bytes.len()
            )));
        }
        let mut proof_bytes = [0u8; MAX_PROOF_BYTES];
        proof_bytes.copy_from_slice(&bytes);
        Some(FoldProof(proof_bytes))
    } else {
        None
    };
    Ok(FoldSnapshot {
        state: snapshot_file.data.state,
        last_proof,
    })
}

fn persist_fold_snapshot(path: &Path, snapshot: &FoldSnapshot) -> StoreResult<()> {
    let data = FoldSnapshotData {
        state: snapshot.state.clone(),
        last_proof: snapshot
            .last_proof
            .as_ref()
            .map(|proof| proof.as_bytes().to_vec()),
    };
    let data_bytes = serialize(&data)
        .map_err(|err| ucf_evidence::StoreError::IOError(format!("encode fold snapshot: {err}")))?;
    let checksum = blake3::hash(&data_bytes);
    let file = FoldSnapshotFile {
        data,
        checksum: *checksum.as_bytes(),
    };
    let bytes = serialize(&file)
        .map_err(|err| ucf_evidence::StoreError::IOError(format!("encode fold snapshot: {err}")))?;
    fs::write(path, &bytes)
        .map_err(|err| ucf_evidence::StoreError::IOError(format!("write fold snapshot: {err}")))?;
    Ok(())
}

impl ExperienceAppender for InMemoryArchive {
    fn append_with_proof(&self, rec: ExperienceRecord, proof: Option<ProofEnvelope>) -> EvidenceId {
        append_record(&self.store, rec, proof, None)
    }
}

impl ExperienceAppender for FileArchive {
    fn append_with_proof(&self, rec: ExperienceRecord, proof: Option<ProofEnvelope>) -> EvidenceId {
        append_record(&self.store, rec, proof, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use ucf_commit::commit_experience_record;
    use ucf_fold::{DummyFolder, FoldState, MAX_PROOF_BYTES};

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

    #[test]
    fn append_and_fold_updates_state_and_proofs() {
        let archive = InMemoryArchive::new();
        let record = ExperienceRecord {
            record_id: "fold-1".to_string(),
            observed_at_ms: 1_700_000_000_000,
            subject_id: "subject-fold-1".to_string(),
            payload: vec![1, 2, 3],
            digest: None,
            vrf_tag: None,
            proof_ref: None,
        };
        let record2 = ExperienceRecord {
            record_id: "fold-2".to_string(),
            observed_at_ms: 1_700_000_000_010,
            subject_id: "subject-fold-2".to_string(),
            payload: vec![4, 5, 6],
            digest: None,
            vrf_tag: None,
            proof_ref: None,
        };

        let (_, state1) = archive.append_and_fold(record);
        let (_, state2) = archive.append_and_fold(record2);

        let entries = archive.list();
        assert_eq!(state1.epoch, 1);
        assert_eq!(state2.epoch, 2);
        for entry in entries {
            let proof = entry.fold_proof.expect("fold proof stored");
            assert_eq!(proof.as_bytes().len(), MAX_PROOF_BYTES);
        }
    }

    #[test]
    fn append_and_fold_uses_commitment_digest() {
        let archive = InMemoryArchive::new();
        let record = ExperienceRecord {
            record_id: "fold-commit-1".to_string(),
            observed_at_ms: 1_700_000_000_111,
            subject_id: "subject-fold-commit-1".to_string(),
            payload: vec![9, 8, 7],
            digest: None,
            vrf_tag: None,
            proof_ref: None,
        };

        let commitment = commit_experience_record(&record);
        let expected_digest = commitment.to_domain_digest();
        let (expected_state, _) =
            DummyFolder::fold_step(&FoldState::genesis(), expected_digest, None);

        let (_, state) = archive.append_and_fold(record);
        assert_eq!(state.acc, expected_state.acc);
    }

    #[test]
    fn append_and_fold_is_deterministic_across_runs() {
        let record = ExperienceRecord {
            record_id: "fold-commit-2".to_string(),
            observed_at_ms: 1_700_000_000_222,
            subject_id: "subject-fold-commit-2".to_string(),
            payload: vec![5, 4, 3],
            digest: None,
            vrf_tag: None,
            proof_ref: None,
        };

        let archive_a = InMemoryArchive::new();
        let archive_b = InMemoryArchive::new();

        let (_, state_a) = archive_a.append_and_fold(record.clone());
        let (_, state_b) = archive_b.append_and_fold(record);

        assert_eq!(state_a.acc, state_b.acc);
        assert_eq!(state_a.epoch, state_b.epoch);
    }

    #[test]
    fn file_archive_persists_fold_snapshot() {
        let temp_dir = TempDir::new().expect("temp dir");
        let archive = FileArchive::open(temp_dir.path()).expect("open archive");
        let record = ExperienceRecord {
            record_id: "fold-3".to_string(),
            observed_at_ms: 1_700_000_000_020,
            subject_id: "subject-fold-3".to_string(),
            payload: vec![9, 9, 9],
            digest: None,
            vrf_tag: None,
            proof_ref: None,
        };
        let record2 = ExperienceRecord {
            record_id: "fold-4".to_string(),
            observed_at_ms: 1_700_000_000_030,
            subject_id: "subject-fold-4".to_string(),
            payload: vec![8, 8, 8],
            digest: None,
            vrf_tag: None,
            proof_ref: None,
        };

        let (_, state1) = archive.append_and_fold(record).expect("append fold");
        let (_, state2) = archive.append_and_fold(record2).expect("append fold");
        assert_eq!(state1.epoch, 1);
        assert_eq!(state2.epoch, 2);
        drop(archive);

        let reopened = FileArchive::open(temp_dir.path()).expect("reopen archive");
        let record3 = ExperienceRecord {
            record_id: "fold-5".to_string(),
            observed_at_ms: 1_700_000_000_040,
            subject_id: "subject-fold-5".to_string(),
            payload: vec![7, 7, 7],
            digest: None,
            vrf_tag: None,
            proof_ref: None,
        };
        let (_, state3) = reopened.append_and_fold(record3).expect("append fold");
        assert_eq!(state3.epoch, 3);
    }

    #[test]
    fn file_archive_detects_corrupt_fold_snapshot() {
        let temp_dir = TempDir::new().expect("temp dir");
        let archive = FileArchive::open(temp_dir.path()).expect("open archive");
        let record = ExperienceRecord {
            record_id: "fold-6".to_string(),
            observed_at_ms: 1_700_000_000_050,
            subject_id: "subject-fold-6".to_string(),
            payload: vec![6, 6, 6],
            digest: None,
            vrf_tag: None,
            proof_ref: None,
        };
        archive.append_and_fold(record).expect("append fold");
        drop(archive);

        let fold_path = temp_dir.path().join(FOLD_SNAPSHOT_FILE);
        let data = FoldSnapshotData {
            state: FoldState::genesis(),
            last_proof: None,
        };
        let corrupt = FoldSnapshotFile {
            data,
            checksum: [0u8; 32],
        };
        let bytes = serialize(&corrupt).expect("encode corrupt snapshot");
        fs::write(&fold_path, bytes).expect("write corrupt snapshot");

        match FileArchive::open(temp_dir.path()) {
            Err(ucf_evidence::StoreError::IOError(message)) => {
                assert!(message.contains("fold snapshot checksum mismatch"));
            }
            Err(other) => panic!("expected fold snapshot error, got {other:?}"),
            Ok(_) => panic!("expected fold snapshot error, got Ok"),
        }
    }

    #[test]
    fn file_archive_rejects_invalid_proof_length_in_snapshot() {
        let temp_dir = TempDir::new().expect("temp dir");
        let fold_path = temp_dir.path().join(FOLD_SNAPSHOT_FILE);
        let data = FoldSnapshotData {
            state: FoldState::genesis(),
            last_proof: Some(vec![0u8; MAX_PROOF_BYTES - 1]),
        };
        let data_bytes = serialize(&data).expect("encode snapshot data");
        let checksum = blake3::hash(&data_bytes);
        let file = FoldSnapshotFile {
            data,
            checksum: *checksum.as_bytes(),
        };
        let bytes = serialize(&file).expect("encode snapshot file");
        fs::write(&fold_path, bytes).expect("write snapshot");

        match FileArchive::open(temp_dir.path()) {
            Err(ucf_evidence::StoreError::IOError(message)) => {
                assert!(message.contains("fold snapshot proof length invalid"));
            }
            Err(other) => panic!("expected fold snapshot error, got {other:?}"),
            Ok(_) => panic!("expected fold snapshot error, got Ok"),
        }
    }
}
