#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::sync::Mutex;

use blake3::Hasher;
use ucf_types::Digest32;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RecordKind {
    WorkspaceSnapshot,
    SelfState,
    IitReport,
    ConsistencyReport,
    ReplayToken,
    ReplayApplied,
    IsmAnchor,
    CyclePlan,
    OutputEvent,
    Other(u16),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RecordMeta {
    pub cycle_id: u64,
    pub tier: u8,
    pub flags: u16,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ArchiveRecord {
    pub kind: RecordKind,
    pub key: Digest32,
    pub payload_commit: Digest32,
    pub meta: RecordMeta,
}

pub trait ArchiveStore {
    fn append(&self, record: ArchiveRecord) -> Digest32;

    fn get(&self, key: Digest32) -> Option<ArchiveRecord>;

    fn iter_kind(
        &self,
        kind: RecordKind,
        limit: Option<usize>,
    ) -> Box<dyn Iterator<Item = ArchiveRecord> + '_>;

    fn root_commit(&self) -> Option<Digest32>;
}

const ROOT_COMMIT_DOMAIN: &[u8] = b"ucf.archive.store.root.v1";
const KIND_OTHER_BASE: u16 = 0x8000;

#[derive(Default)]
struct InMemoryState {
    records: Vec<ArchiveRecord>,
    index: HashMap<Digest32, usize>,
}

#[derive(Default)]
pub struct InMemoryArchiveStore {
    inner: Mutex<InMemoryState>,
}

impl InMemoryArchiveStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn record_digest(record: &ArchiveRecord) -> Digest32 {
        let mut hasher = Hasher::new();
        hasher.update(ROOT_COMMIT_DOMAIN);
        write_kind(&mut hasher, record.kind);
        hasher.update(record.key.as_bytes());
        hasher.update(record.payload_commit.as_bytes());
        write_meta(&mut hasher, &record.meta);
        Digest32::new(*hasher.finalize().as_bytes())
    }
}

impl ArchiveStore for InMemoryArchiveStore {
    fn append(&self, record: ArchiveRecord) -> Digest32 {
        let mut state = self.inner.lock().expect("lock in-memory archive store");
        let index = state.records.len();
        state.records.push(record);
        state.index.insert(record.key, index);
        Self::record_digest(&record)
    }

    fn get(&self, key: Digest32) -> Option<ArchiveRecord> {
        let state = self.inner.lock().expect("lock in-memory archive store");
        let index = state.index.get(&key).copied()?;
        state.records.get(index).copied()
    }

    fn iter_kind(
        &self,
        kind: RecordKind,
        limit: Option<usize>,
    ) -> Box<dyn Iterator<Item = ArchiveRecord> + '_> {
        let state = self.inner.lock().expect("lock in-memory archive store");
        let records = state
            .records
            .iter()
            .copied()
            .filter(|record| record.kind == kind);
        let records: Vec<ArchiveRecord> = match limit {
            Some(limit) => records.take(limit).collect(),
            None => records.collect(),
        };
        Box::new(records.into_iter())
    }

    fn root_commit(&self) -> Option<Digest32> {
        let state = self.inner.lock().expect("lock in-memory archive store");
        if state.records.is_empty() {
            return None;
        }
        let mut hasher = Hasher::new();
        hasher.update(ROOT_COMMIT_DOMAIN);
        for record in &state.records {
            write_kind(&mut hasher, record.kind);
            hasher.update(record.key.as_bytes());
            hasher.update(record.payload_commit.as_bytes());
            write_meta(&mut hasher, &record.meta);
        }
        Some(Digest32::new(*hasher.finalize().as_bytes()))
    }
}

fn write_kind(hasher: &mut Hasher, kind: RecordKind) {
    let tag = match kind {
        RecordKind::WorkspaceSnapshot => 1,
        RecordKind::SelfState => 2,
        RecordKind::IitReport => 3,
        RecordKind::ConsistencyReport => 4,
        RecordKind::ReplayToken => 5,
        RecordKind::ReplayApplied => 6,
        RecordKind::IsmAnchor => 7,
        RecordKind::CyclePlan => 8,
        RecordKind::OutputEvent => 9,
        RecordKind::Other(value) => KIND_OTHER_BASE | value,
    };
    hasher.update(&tag.to_be_bytes());
}

fn write_meta(hasher: &mut Hasher, meta: &RecordMeta) {
    hasher.update(&meta.cycle_id.to_be_bytes());
    hasher.update(&[meta.tier]);
    hasher.update(&meta.flags.to_be_bytes());
}
