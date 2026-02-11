use anyhow::Result;

#[cfg(feature = "firewood")]
pub mod firewood;
pub mod mem;

pub type SeqNo = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecordId(pub [u8; 32]);

#[derive(Clone, Debug, PartialEq)]
pub struct AppendResult {
    pub seq: SeqNo,
    pub id: RecordId,
    pub bytes: usize,
}

pub trait ArchiveStore: Send + Sync {
    fn append(&mut self, payload: &[u8]) -> Result<AppendResult>;
    fn get(&self, seq: SeqNo) -> Result<Option<Vec<u8>>>;
    fn last_seq(&self) -> Result<SeqNo>;
    fn flush(&mut self) -> Result<()>;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FlushPolicy {
    EveryAppend,
    IntervalMs(u64),
    Manual,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ArchiveCfg {
    pub flush: FlushPolicy,
}

impl Default for ArchiveCfg {
    fn default() -> Self {
        Self {
            flush: FlushPolicy::EveryAppend,
        }
    }
}

pub fn derive_record_id(seq: SeqNo, payload: &[u8]) -> RecordId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&seq.to_le_bytes());
    hasher.update(payload);
    RecordId(*hasher.finalize().as_bytes())
}

#[cfg(feature = "firewood")]
pub use firewood::FirewoodArchiveStore;
pub use mem::MemArchiveStore;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_id_for_same_seq_payload() {
        let payload = b"tick-summary";
        let id1 = derive_record_id(3, payload);
        let id2 = derive_record_id(3, payload);
        assert_eq!(id1, id2);
    }

    #[test]
    fn mem_archive_seq_starts_at_one_and_increments() {
        let mut store = MemArchiveStore::new();
        let first = store.append(b"a").expect("append first");
        let second = store.append(b"b").expect("append second");

        assert_eq!(first.seq, 1);
        assert_eq!(second.seq, 2);
    }
}
