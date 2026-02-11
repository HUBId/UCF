use anyhow::Result;

use super::{derive_record_id, AppendResult, ArchiveStore, SeqNo};

#[derive(Clone, Debug, Default)]
pub struct MemArchiveStore {
    records: Vec<Vec<u8>>,
}

impl MemArchiveStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ArchiveStore for MemArchiveStore {
    fn append(&mut self, payload: &[u8]) -> Result<AppendResult> {
        let seq = self.records.len() as SeqNo + 1;
        let bytes = payload.len();
        self.records.push(payload.to_vec());

        Ok(AppendResult {
            seq,
            id: derive_record_id(seq, payload),
            bytes,
        })
    }

    fn get(&self, seq: SeqNo) -> Result<Option<Vec<u8>>> {
        if seq == 0 {
            return Ok(None);
        }
        Ok(self.records.get((seq - 1) as usize).cloned())
    }

    fn last_seq(&self) -> Result<SeqNo> {
        Ok(self.records.len() as SeqNo)
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}
