#![forbid(unsafe_code)]
//! Stub Firewood backend for the archive store.

use ucf_archive_store::{ArchiveRecord, ArchiveStore, RecordKind};
use ucf_types::Digest32;

#[derive(Clone, Copy, Debug, Default)]
pub struct FirewoodArchiveStore;

impl FirewoodArchiveStore {
    pub fn new() -> Self {
        Self
    }
}

impl ArchiveStore for FirewoodArchiveStore {
    fn append(&self, _record: ArchiveRecord) -> Digest32 {
        panic!("backend unavailable")
    }

    fn get(&self, _key: Digest32) -> Option<ArchiveRecord> {
        panic!("backend unavailable")
    }

    fn iter_kind(
        &self,
        _kind: RecordKind,
        _limit: Option<usize>,
    ) -> Box<dyn Iterator<Item = ArchiveRecord> + '_> {
        panic!("backend unavailable")
    }

    fn root_commit(&self) -> Option<Digest32> {
        panic!("backend unavailable")
    }
}

#[cfg(feature = "archive-firewood")]
#[cfg(test)]
mod tests {
    use super::FirewoodArchiveStore;

    #[test]
    fn constructs_firewood_store() {
        let _store = FirewoodArchiveStore::new();
    }

    #[test]
    fn constructs_firewood_store_default() {
        let _store = FirewoodArchiveStore::default();
    }
}
