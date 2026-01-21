#![forbid(unsafe_code)]

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

    fn iter_kind(&self, kind: RecordKind) -> Box<dyn Iterator<Item = ArchiveRecord> + '_>;

    fn root_commit(&self) -> Option<Digest32>;
}
