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
    StructuralParams,
    StructuralProposal,
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
    pub boundary_commit: Digest32,
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

const ARCHIVE_KEY_DOMAIN: &[u8] = b"ucf.archive.record.key.v1";
const ROOT_COMMIT_DOMAIN: &[u8] = b"ucf.archive.store.root.v1";
const KIND_OTHER_BASE: u16 = 0x8000;

#[derive(Default)]
pub struct ArchiveAppender {
    seq_by_cycle: HashMap<u64, u64>,
}

impl ArchiveAppender {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn build_record_with_commit(
        &mut self,
        kind: RecordKind,
        payload_commit: Digest32,
        meta: RecordMeta,
    ) -> ArchiveRecord {
        let seq_no = self.next_seq_no(meta.cycle_id);
        let key = record_key(kind, &payload_commit, meta.cycle_id, seq_no);
        ArchiveRecord {
            kind,
            key,
            payload_commit,
            meta,
        }
    }

    pub fn build_record(
        &mut self,
        kind: RecordKind,
        payload: &[u8],
        meta: RecordMeta,
    ) -> ArchiveRecord {
        let payload_commit = digest_payload(payload);
        self.build_record_with_commit(kind, payload_commit, meta)
    }

    fn next_seq_no(&mut self, cycle_id: u64) -> u64 {
        let entry = self.seq_by_cycle.entry(cycle_id).or_insert(0);
        let seq_no = *entry;
        *entry = seq_no.saturating_add(1);
        seq_no
    }
}

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
        RecordKind::StructuralParams => 5,
        RecordKind::StructuralProposal => 6,
        RecordKind::ReplayToken => 7,
        RecordKind::ReplayApplied => 8,
        RecordKind::IsmAnchor => 9,
        RecordKind::CyclePlan => 10,
        RecordKind::OutputEvent => 11,
        RecordKind::Other(value) => KIND_OTHER_BASE | value,
    };
    hasher.update(&tag.to_be_bytes());
}

fn write_meta(hasher: &mut Hasher, meta: &RecordMeta) {
    hasher.update(&meta.cycle_id.to_be_bytes());
    hasher.update(&[meta.tier]);
    hasher.update(&meta.flags.to_be_bytes());
    hasher.update(meta.boundary_commit.as_bytes());
}

fn digest_payload(payload: &[u8]) -> Digest32 {
    let digest = blake3::hash(payload);
    Digest32::new(*digest.as_bytes())
}

fn record_key(kind: RecordKind, payload_commit: &Digest32, cycle_id: u64, seq_no: u64) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(ARCHIVE_KEY_DOMAIN);
    write_kind(&mut hasher, kind);
    hasher.update(payload_commit.as_bytes());
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(&seq_no.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expected_root_commit(records: &[ArchiveRecord]) -> Option<Digest32> {
        if records.is_empty() {
            return None;
        }
        let mut hasher = Hasher::new();
        hasher.update(ROOT_COMMIT_DOMAIN);
        for record in records {
            write_kind(&mut hasher, record.kind);
            hasher.update(record.key.as_bytes());
            hasher.update(record.payload_commit.as_bytes());
            write_meta(&mut hasher, &record.meta);
        }
        Some(Digest32::new(*hasher.finalize().as_bytes()))
    }

    #[test]
    fn append_get_roundtrip() {
        let store = InMemoryArchiveStore::new();
        let mut appender = ArchiveAppender::new();
        let meta = RecordMeta {
            cycle_id: 42,
            tier: 2,
            flags: 0x0a0b,
            boundary_commit: Digest32::new([0u8; 32]),
        };
        let record = appender.build_record(RecordKind::WorkspaceSnapshot, b"payload-1", meta);

        store.append(record);

        assert_eq!(store.get(record.key), Some(record));
    }

    #[test]
    fn iter_kind_is_deterministic() {
        let store = InMemoryArchiveStore::new();
        let mut appender = ArchiveAppender::new();
        let meta = RecordMeta {
            cycle_id: 7,
            tier: 1,
            flags: 0x0102,
            boundary_commit: Digest32::new([0u8; 32]),
        };
        let record_a = appender.build_record(RecordKind::WorkspaceSnapshot, b"a", meta);
        let record_b = appender.build_record(RecordKind::SelfState, b"b", meta);
        let record_c = appender.build_record(RecordKind::WorkspaceSnapshot, b"c", meta);
        let record_d = appender.build_record(RecordKind::WorkspaceSnapshot, b"d", meta);

        store.append(record_a);
        store.append(record_b);
        store.append(record_c);
        store.append(record_d);

        let collected: Vec<ArchiveRecord> = store
            .iter_kind(RecordKind::WorkspaceSnapshot, None)
            .collect();
        assert_eq!(collected, vec![record_a, record_c, record_d]);
    }

    #[test]
    fn root_commit_changes_and_matches_fold_rules() {
        let store = InMemoryArchiveStore::new();
        let mut appender = ArchiveAppender::new();
        let meta = RecordMeta {
            cycle_id: 99,
            tier: 0,
            flags: 0x0f0f,
            boundary_commit: Digest32::new([0u8; 32]),
        };
        let record_a = appender.build_record(RecordKind::OutputEvent, b"first", meta);
        let record_b = appender.build_record(RecordKind::OutputEvent, b"second", meta);

        store.append(record_a);
        store.append(record_b);

        let expected_initial = expected_root_commit(&[record_a, record_b]);
        let initial_commit = store.root_commit();
        assert_eq!(initial_commit, expected_initial);

        let record_c = appender.build_record(RecordKind::OutputEvent, b"third", meta);
        store.append(record_c);

        let expected_updated = expected_root_commit(&[record_a, record_b, record_c]);
        let updated_commit = store.root_commit();
        assert_ne!(updated_commit, initial_commit);
        assert_eq!(updated_commit, expected_updated);
    }
}
