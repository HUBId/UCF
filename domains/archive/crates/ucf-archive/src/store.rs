use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ucf_evidence::{StoreError, StoreResult};
use ucf_types::Digest32;

pub trait RecordStore {
    fn put_record(&self, key: Digest32, value: &[u8]) -> StoreResult<()>;
    fn get_record(&self, key: Digest32) -> StoreResult<Option<Vec<u8>>>;
    fn scan_recent(&self, n: usize) -> StoreResult<Vec<Vec<u8>>>;
}

pub trait SnapshotStore {
    fn put_snapshot(&self, name: &str, bytes: &[u8]) -> StoreResult<()>;
    fn get_snapshot(&self, name: &str) -> StoreResult<Option<Vec<u8>>>;
}

#[derive(Clone, Copy, Debug)]
pub enum ArchiveBackend {
    InMemory,
    File,
    Firewood,
}

pub struct ArchiveStore {
    record_store: Box<dyn RecordStore + Send + Sync>,
    snapshot_store: Box<dyn SnapshotStore + Send + Sync>,
}

impl ArchiveStore {
    pub fn open(backend: ArchiveBackend, path: Option<&Path>) -> StoreResult<Self> {
        match backend {
            ArchiveBackend::InMemory => Ok(Self::in_memory()),
            ArchiveBackend::File => {
                let path = path.ok_or_else(|| {
                    StoreError::IOError("file backend requires a path".to_string())
                })?;
                let store = FileArchiveStore::open(path)?;
                Ok(Self::from_parts(store.clone(), store))
            }
            ArchiveBackend::Firewood => {
                #[cfg(feature = "firewood")]
                {
                    let path = path.ok_or_else(|| {
                        StoreError::IOError("firewood backend requires a path".to_string())
                    })?;
                    let store = FirewoodRecordStore::open(path)?;
                    Ok(Self::from_parts(store.clone(), store))
                }
                #[cfg(not(feature = "firewood"))]
                {
                    Err(StoreError::Unsupported(
                        "firewood feature not enabled".to_string(),
                    ))
                }
            }
        }
    }

    pub fn in_memory() -> Self {
        let store = InMemoryArchiveStore::new();
        Self::from_parts(store.clone(), store)
    }

    pub fn from_parts<R, S>(record_store: R, snapshot_store: S) -> Self
    where
        R: RecordStore + Send + Sync + 'static,
        S: SnapshotStore + Send + Sync + 'static,
    {
        Self {
            record_store: Box::new(record_store),
            snapshot_store: Box::new(snapshot_store),
        }
    }

    pub fn record_store(&self) -> &dyn RecordStore {
        &*self.record_store
    }

    pub fn snapshot_store(&self) -> &dyn SnapshotStore {
        &*self.snapshot_store
    }
}

#[derive(Clone)]
pub struct InMemoryArchiveStore {
    inner: std::sync::Arc<Mutex<InMemoryState>>,
}

#[derive(Default)]
struct InMemoryState {
    records: HashMap<Digest32, Vec<u8>>,
    order: Vec<Digest32>,
    snapshots: HashMap<String, Vec<u8>>,
}

impl InMemoryArchiveStore {
    pub fn new() -> Self {
        Self {
            inner: std::sync::Arc::new(Mutex::new(InMemoryState::default())),
        }
    }
}

impl Default for InMemoryArchiveStore {
    fn default() -> Self {
        Self::new()
    }
}

impl RecordStore for InMemoryArchiveStore {
    fn put_record(&self, key: Digest32, value: &[u8]) -> StoreResult<()> {
        let mut state = self.inner.lock().expect("lock in-memory archive");
        state.records.insert(key, value.to_vec());
        state.order.push(key);
        Ok(())
    }

    fn get_record(&self, key: Digest32) -> StoreResult<Option<Vec<u8>>> {
        let state = self.inner.lock().expect("lock in-memory archive");
        Ok(state.records.get(&key).cloned())
    }

    fn scan_recent(&self, n: usize) -> StoreResult<Vec<Vec<u8>>> {
        let state = self.inner.lock().expect("lock in-memory archive");
        let mut recent = Vec::with_capacity(n.min(state.order.len()));
        for key in state.order.iter().rev().take(n) {
            if let Some(value) = state.records.get(key) {
                recent.push(value.clone());
            }
        }
        Ok(recent)
    }
}

impl SnapshotStore for InMemoryArchiveStore {
    fn put_snapshot(&self, name: &str, bytes: &[u8]) -> StoreResult<()> {
        let mut state = self.inner.lock().expect("lock in-memory archive");
        state.snapshots.insert(name.to_string(), bytes.to_vec());
        Ok(())
    }

    fn get_snapshot(&self, name: &str) -> StoreResult<Option<Vec<u8>>> {
        let state = self.inner.lock().expect("lock in-memory archive");
        Ok(state.snapshots.get(name).cloned())
    }
}

#[derive(Clone)]
pub struct FileArchiveStore {
    root: std::sync::Arc<FilePaths>,
}

struct FilePaths {
    record_dir: PathBuf,
    snapshot_dir: PathBuf,
    index_path: PathBuf,
    lock: Mutex<()>,
}

impl FileArchiveStore {
    pub fn open(path: impl AsRef<Path>) -> StoreResult<Self> {
        let root = path.as_ref();
        let record_dir = root.join("records");
        let snapshot_dir = root.join("snapshots");
        let index_path = root.join("record_index.bin");
        fs::create_dir_all(&record_dir)
            .map_err(|err| StoreError::IOError(format!("create record dir: {err}")))?;
        fs::create_dir_all(&snapshot_dir)
            .map_err(|err| StoreError::IOError(format!("create snapshot dir: {err}")))?;
        Ok(Self {
            root: std::sync::Arc::new(FilePaths {
                record_dir,
                snapshot_dir,
                index_path,
                lock: Mutex::new(()),
            }),
        })
    }

    fn record_path(&self, key: Digest32) -> PathBuf {
        self.root.record_dir.join(hex_digest(key))
    }

    fn snapshot_path(&self, name: &str) -> PathBuf {
        self.root.snapshot_dir.join(name)
    }

    fn append_index(&self, key: Digest32) -> StoreResult<()> {
        let _guard = self.root.lock.lock().expect("lock file archive");
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.root.index_path)
            .map_err(|err| StoreError::IOError(format!("open record index: {err}")))?;
        use std::io::Write;
        file.write_all(key.as_bytes())
            .map_err(|err| StoreError::IOError(format!("append record index: {err}")))?;
        Ok(())
    }

    fn read_index(&self) -> StoreResult<Vec<Digest32>> {
        let bytes = match fs::read(&self.root.index_path) {
            Ok(bytes) => bytes,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => return Err(StoreError::IOError(format!("read record index: {err}"))),
        };
        if bytes.len() % Digest32::LEN != 0 {
            return Err(StoreError::IOError(
                "record index length invalid".to_string(),
            ));
        }
        let mut entries = Vec::with_capacity(bytes.len() / Digest32::LEN);
        for chunk in bytes.chunks_exact(Digest32::LEN) {
            let mut digest = [0u8; Digest32::LEN];
            digest.copy_from_slice(chunk);
            entries.push(Digest32::new(digest));
        }
        Ok(entries)
    }
}

impl RecordStore for FileArchiveStore {
    fn put_record(&self, key: Digest32, value: &[u8]) -> StoreResult<()> {
        let path = self.record_path(key);
        fs::write(&path, value)
            .map_err(|err| StoreError::IOError(format!("write record: {err}")))?;
        self.append_index(key)?;
        Ok(())
    }

    fn get_record(&self, key: Digest32) -> StoreResult<Option<Vec<u8>>> {
        let path = self.record_path(key);
        match fs::read(&path) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(StoreError::IOError(format!("read record: {err}"))),
        }
    }

    fn scan_recent(&self, n: usize) -> StoreResult<Vec<Vec<u8>>> {
        let entries = self.read_index()?;
        let mut recent = Vec::with_capacity(n.min(entries.len()));
        for key in entries.iter().rev().take(n) {
            match self.get_record(*key)? {
                Some(bytes) => recent.push(bytes),
                None => {
                    return Err(StoreError::IOError(format!(
                        "record missing for digest {}",
                        hex_digest(*key)
                    )))
                }
            }
        }
        Ok(recent)
    }
}

impl SnapshotStore for FileArchiveStore {
    fn put_snapshot(&self, name: &str, bytes: &[u8]) -> StoreResult<()> {
        let path = self.snapshot_path(name);
        fs::write(&path, bytes)
            .map_err(|err| StoreError::IOError(format!("write snapshot: {err}")))?;
        Ok(())
    }

    fn get_snapshot(&self, name: &str) -> StoreResult<Option<Vec<u8>>> {
        let path = self.snapshot_path(name);
        match fs::read(&path) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(StoreError::IOError(format!("read snapshot: {err}"))),
        }
    }
}

#[cfg(feature = "firewood")]
#[derive(Clone)]
pub struct FirewoodRecordStore {
    kv: std::sync::Arc<Mutex<storage_firewood::kv::FirewoodKv>>,
}

#[cfg(feature = "firewood")]
impl FirewoodRecordStore {
    pub fn open(path: impl AsRef<Path>) -> StoreResult<Self> {
        let kv = storage_firewood::kv::FirewoodKv::open(path.as_ref())
            .map_err(|err| StoreError::IOError(format!("open firewood: {err}")))?;
        Ok(Self {
            kv: std::sync::Arc::new(Mutex::new(kv)),
        })
    }

    fn record_key(key: Digest32) -> Vec<u8> {
        format!("rec/{}", hex_digest(key)).into_bytes()
    }

    fn snapshot_key(name: &str) -> Vec<u8> {
        format!("snap/{name}").into_bytes()
    }

    fn index_key(index: u64) -> Vec<u8> {
        format!("idx/{index:020}").into_bytes()
    }

    fn latest_key() -> &'static [u8] {
        b"idx/latest"
    }

    fn decode_u64(bytes: &[u8]) -> StoreResult<u64> {
        if bytes.len() != 8 {
            return Err(StoreError::IOError(
                "firewood latest index length invalid".to_string(),
            ));
        }
        let mut buf = [0u8; 8];
        buf.copy_from_slice(bytes);
        Ok(u64::from_be_bytes(buf))
    }
}

#[cfg(feature = "firewood")]
impl RecordStore for FirewoodRecordStore {
    fn put_record(&self, key: Digest32, value: &[u8]) -> StoreResult<()> {
        let mut kv = self.kv.lock().expect("lock firewood store");
        let latest_bytes = kv.get(Self::latest_key());
        let next_index = match latest_bytes {
            Some(bytes) => Self::decode_u64(&bytes)?.saturating_add(1),
            None => 0,
        };
        kv.put(Self::record_key(key), value.to_vec());
        kv.put(Self::index_key(next_index), key.as_bytes().to_vec());
        kv.put(
            Self::latest_key().to_vec(),
            next_index.to_be_bytes().to_vec(),
        );
        kv.commit()
            .map_err(|err| StoreError::IOError(format!("commit firewood: {err}")))?;
        Ok(())
    }

    fn get_record(&self, key: Digest32) -> StoreResult<Option<Vec<u8>>> {
        let kv = self.kv.lock().expect("lock firewood store");
        Ok(kv.get(&Self::record_key(key)))
    }

    fn scan_recent(&self, n: usize) -> StoreResult<Vec<Vec<u8>>> {
        let kv = self.kv.lock().expect("lock firewood store");
        let latest = match kv.get(Self::latest_key()) {
            Some(bytes) => Self::decode_u64(&bytes)?,
            None => return Ok(Vec::new()),
        };
        let mut recent = Vec::with_capacity(n);
        let mut current = latest;
        while recent.len() < n {
            let digest_bytes = kv
                .get(&Self::index_key(current))
                .ok_or_else(|| StoreError::IOError(format!("missing record index {current}")))?;
            let digest = Digest32::try_from(digest_bytes)
                .map_err(|err| StoreError::IOError(format!("invalid record digest: {err}")))?;
            let record = kv.get(&Self::record_key(digest)).ok_or_else(|| {
                StoreError::IOError(format!("record missing for digest {}", hex_digest(digest)))
            })?;
            recent.push(record);
            if current == 0 {
                break;
            }
            current -= 1;
        }
        Ok(recent)
    }
}

#[cfg(feature = "firewood")]
impl SnapshotStore for FirewoodRecordStore {
    fn put_snapshot(&self, name: &str, bytes: &[u8]) -> StoreResult<()> {
        let mut kv = self.kv.lock().expect("lock firewood store");
        kv.put(Self::snapshot_key(name), bytes.to_vec());
        kv.commit()
            .map_err(|err| StoreError::IOError(format!("commit firewood: {err}")))?;
        Ok(())
    }

    fn get_snapshot(&self, name: &str) -> StoreResult<Option<Vec<u8>>> {
        let kv = self.kv.lock().expect("lock firewood store");
        Ok(kv.get(&Self::snapshot_key(name)))
    }
}

fn hex_digest(digest: Digest32) -> String {
    let mut out = String::with_capacity(Digest32::LEN * 2);
    for byte in digest.as_bytes() {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    #[cfg(not(feature = "firewood"))]
    fn firewood_backend_requires_feature() {
        let result = ArchiveStore::open(ArchiveBackend::Firewood, None);
        match result {
            Err(StoreError::Unsupported(message)) => {
                assert!(message.contains("feature not enabled"));
            }
            Err(other) => panic!("expected unsupported error, got {other:?}"),
            Ok(_) => panic!("expected unsupported error, got Ok"),
        }
    }

    #[test]
    fn file_store_records_and_snapshots() {
        let temp_dir = TempDir::new().expect("temp dir");
        let store = FileArchiveStore::open(temp_dir.path()).expect("open file store");
        let digest = Digest32::new([7u8; Digest32::LEN]);
        store
            .put_record(digest, b"record-bytes")
            .expect("put record");
        store
            .put_snapshot("fold", b"snapshot")
            .expect("put snapshot");

        let record = store.get_record(digest).expect("get record");
        let snapshot = store.get_snapshot("fold").expect("get snapshot");

        assert_eq!(record, Some(b"record-bytes".to_vec()));
        assert_eq!(snapshot, Some(b"snapshot".to_vec()));
        let recent = store.scan_recent(1).expect("scan recent");
        assert_eq!(recent, vec![b"record-bytes".to_vec()]);
    }

    #[test]
    #[cfg(feature = "firewood")]
    fn firewood_store_records_snapshots_and_scan() {
        let temp_dir = TempDir::new().expect("temp dir");
        let store = FirewoodRecordStore::open(temp_dir.path()).expect("open firewood");
        let digest_a = Digest32::new([1u8; Digest32::LEN]);
        let digest_b = Digest32::new([2u8; Digest32::LEN]);

        store.put_record(digest_a, b"first").expect("put record");
        store.put_record(digest_b, b"second").expect("put record");
        store
            .put_snapshot("fold", b"snapshot")
            .expect("put snapshot");

        let record = store.get_record(digest_a).expect("get record");
        let snapshot = store.get_snapshot("fold").expect("get snapshot");
        let recent = store.scan_recent(2).expect("scan recent");

        assert_eq!(record, Some(b"first".to_vec()));
        assert_eq!(snapshot, Some(b"snapshot".to_vec()));
        assert_eq!(recent, vec![b"second".to_vec(), b"first".to_vec()]);
    }
}
