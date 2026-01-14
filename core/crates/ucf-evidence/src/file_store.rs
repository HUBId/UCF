use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use bincode::{deserialize, serialize};
use serde::{Deserialize, Serialize};
use sha2::Digest;
use ucf_crypto::DigestAlgo;
use ucf_types::EvidenceId;

use crate::{AppendLog, AppendLogHash, EvidenceEnvelope, EvidenceStore, StoreError, StoreResult};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ManifestEntry {
    evidence_id: EvidenceId,
    offset: u64,
    len: u64,
    hash: AppendLogHash,
}

struct Sha256Digest;

impl DigestAlgo for Sha256Digest {
    fn algorithm(&self) -> &str {
        "sha-256"
    }

    fn digest(&self, input: &[u8]) -> Vec<u8> {
        sha2::Sha256::digest(input).to_vec()
    }
}

fn digest_to_hash(digest: &dyn DigestAlgo, input: &[u8]) -> AppendLogHash {
    let bytes = digest.digest(input);
    let mut hash = [0u8; 32];
    let target_len = hash.len();
    if bytes.len() >= target_len {
        hash.copy_from_slice(&bytes[..target_len]);
    } else {
        hash[..bytes.len()].copy_from_slice(&bytes);
    }
    hash
}

pub struct FileAppendLog {
    log: Mutex<File>,
    manifest: Mutex<File>,
    digest: Box<dyn DigestAlgo + Send + Sync>,
    entries: Mutex<HashMap<EvidenceId, ManifestEntry>>,
}

impl FileAppendLog {
    pub fn open(log_path: impl AsRef<Path>, manifest_path: impl AsRef<Path>) -> StoreResult<Self> {
        let log_path = log_path.as_ref().to_path_buf();
        let manifest_path = manifest_path.as_ref().to_path_buf();
        let log = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&log_path)
            .map_err(|err| StoreError::IOError(format!("open log: {err}")))?;
        let manifest = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&manifest_path)
            .map_err(|err| StoreError::IOError(format!("open manifest: {err}")))?;
        let digest: Box<dyn DigestAlgo + Send + Sync> = Box::new(Sha256Digest);
        let entries = load_manifest(&manifest_path)?;
        let store = Self {
            log: Mutex::new(log),
            manifest: Mutex::new(manifest),
            digest,
            entries: Mutex::new(entries),
        };
        store.verify_manifest()?;
        Ok(store)
    }

    fn verify_manifest(&self) -> StoreResult<()> {
        let entries = self.entries.lock().expect("lock manifest entries");
        for entry in entries.values() {
            let bytes = self.read_at_checked(entry.offset, entry.len as usize)?;
            let actual_hash = digest_to_hash(self.digest.as_ref(), &bytes);
            if actual_hash != entry.hash {
                return Err(StoreError::Corrupt {
                    evidence_id: entry.evidence_id.clone(),
                    offset: entry.offset,
                    expected_hash: entry.hash,
                    actual_hash,
                });
            }
        }
        Ok(())
    }

    fn read_at_checked(&self, offset: u64, len: usize) -> StoreResult<Vec<u8>> {
        let mut file = self.log.lock().expect("lock log file");
        file.seek(SeekFrom::Start(offset))
            .map_err(|err| StoreError::IOError(format!("seek log: {err}")))?;
        let mut buf = vec![0u8; len];
        let read = file
            .read(&mut buf)
            .map_err(|err| StoreError::IOError(format!("read log: {err}")))?;
        if read != len {
            buf.truncate(read);
        }
        Ok(buf)
    }

    fn persist_manifest(&self) -> StoreResult<()> {
        let mut entries: Vec<ManifestEntry> = self
            .entries
            .lock()
            .expect("lock manifest entries")
            .values()
            .cloned()
            .collect();
        entries.sort_by(|a, b| a.evidence_id.as_str().cmp(b.evidence_id.as_str()));
        let bytes = serialize(&entries)
            .map_err(|err| StoreError::IOError(format!("encode manifest: {err}")))?;
        let mut file = self.manifest.lock().expect("lock manifest file");
        file.seek(SeekFrom::Start(0))
            .map_err(|err| StoreError::IOError(format!("seek manifest: {err}")))?;
        file.set_len(0)
            .map_err(|err| StoreError::IOError(format!("truncate manifest: {err}")))?;
        file.write_all(&bytes)
            .map_err(|err| StoreError::IOError(format!("write manifest: {err}")))?;
        file.flush()
            .map_err(|err| StoreError::IOError(format!("flush manifest: {err}")))?;
        Ok(())
    }

    fn insert_entry(&self, entry: ManifestEntry) -> StoreResult<()> {
        let mut entries = self.entries.lock().expect("lock manifest entries");
        entries.insert(entry.evidence_id.clone(), entry);
        drop(entries);
        self.persist_manifest()
    }
}

impl AppendLog for FileAppendLog {
    fn append_bytes(&self, bytes: &[u8]) -> (u64, usize, AppendLogHash) {
        let mut file = self.log.lock().expect("lock log file");
        let offset = file.seek(SeekFrom::End(0)).expect("seek log end");
        file.write_all(bytes).expect("write log");
        file.flush().expect("flush log");
        let hash = digest_to_hash(self.digest.as_ref(), bytes);
        (offset, bytes.len(), hash)
    }

    fn read_at(&self, offset: u64, len: usize) -> Vec<u8> {
        self.read_at_checked(offset, len)
            .unwrap_or_else(|_| Vec::new())
    }
}

pub struct FileEvidenceStore {
    log: FileAppendLog,
}

impl FileEvidenceStore {
    pub fn open(log_path: impl AsRef<Path>, manifest_path: impl AsRef<Path>) -> StoreResult<Self> {
        let log = FileAppendLog::open(log_path, manifest_path)?;
        Ok(Self { log })
    }

    pub fn append_envelope(&self, evidence: EvidenceEnvelope) -> StoreResult<EvidenceId> {
        let evidence_id = evidence.evidence_id.clone();
        let bytes = serialize(&evidence)
            .map_err(|err| StoreError::IOError(format!("encode evidence: {err}")))?;
        let (offset, len, hash) = self.log.append_bytes(&bytes);
        let entry = ManifestEntry {
            evidence_id: evidence_id.clone(),
            offset,
            len: len as u64,
            hash,
        };
        self.log.insert_entry(entry)?;
        Ok(evidence_id)
    }

    pub fn get_envelope(&self, evidence_id: EvidenceId) -> StoreResult<Option<EvidenceEnvelope>> {
        let entries = self.log.entries.lock().expect("lock manifest entries");
        let entry = match entries.get(&evidence_id) {
            Some(entry) => entry.clone(),
            None => return Ok(None),
        };
        drop(entries);
        let bytes = self.log.read_at_checked(entry.offset, entry.len as usize)?;
        let evidence: EvidenceEnvelope = deserialize(&bytes)
            .map_err(|err| StoreError::IOError(format!("decode evidence: {err}")))?;
        Ok(Some(evidence))
    }
}

impl EvidenceStore for FileEvidenceStore {
    fn append(&self, evidence: EvidenceEnvelope) -> EvidenceId {
        self.append_envelope(evidence)
            .expect("append evidence to file store")
    }

    fn get(&self, evidence_id: EvidenceId) -> Option<EvidenceEnvelope> {
        self.get_envelope(evidence_id)
            .expect("read evidence from file store")
    }

    fn len(&self) -> usize {
        self.log
            .entries
            .lock()
            .expect("lock manifest entries")
            .len()
    }
}

fn load_manifest(path: &Path) -> StoreResult<HashMap<EvidenceId, ManifestEntry>> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(HashMap::new()),
        Err(err) => {
            return Err(StoreError::IOError(format!(
                "open manifest for load: {err}"
            )))
        }
    };
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|err| StoreError::IOError(format!("read manifest: {err}")))?;
    if bytes.is_empty() {
        return Ok(HashMap::new());
    }
    let entries: Vec<ManifestEntry> = deserialize(&bytes)
        .map_err(|err| StoreError::IOError(format!("decode manifest: {err}")))?;
    Ok(entries
        .into_iter()
        .map(|entry| (entry.evidence_id.clone(), entry))
        .collect())
}
