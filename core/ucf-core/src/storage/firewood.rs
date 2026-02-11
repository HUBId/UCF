use anyhow::{anyhow, Result};
use std::path::Path;
use std::sync::{Arc, Mutex};

use super::{derive_record_id, AppendResult, ArchiveStore, SeqNo};

#[derive(Clone)]
pub struct FirewoodArchiveStore {
    kv: Arc<Mutex<storage_firewood::kv::FirewoodKv>>,
}

impl FirewoodArchiveStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let kv = storage_firewood::kv::FirewoodKv::open(path.as_ref())
            .map_err(|err| anyhow!("open firewood: {err}"))?;
        Ok(Self {
            kv: Arc::new(Mutex::new(kv)),
        })
    }

    fn seq_key(seq: SeqNo) -> [u8; 9] {
        let mut key = [0u8; 9];
        key[0] = b'a';
        key[1..].copy_from_slice(&seq.to_be_bytes());
        key
    }

    fn last_key() -> &'static [u8] {
        b"archive/last"
    }

    fn decode_seq(bytes: &[u8]) -> Result<SeqNo> {
        if bytes.len() != 8 {
            return Err(anyhow!("invalid seq length: {}", bytes.len()));
        }
        let mut buf = [0u8; 8];
        buf.copy_from_slice(bytes);
        Ok(SeqNo::from_be_bytes(buf))
    }
}

impl ArchiveStore for FirewoodArchiveStore {
    fn append(&mut self, payload: &[u8]) -> Result<AppendResult> {
        let mut kv = self.kv.lock().expect("lock firewood archive");
        let seq = match kv.get(Self::last_key()) {
            Some(bytes) => Self::decode_seq(&bytes)?.saturating_add(1),
            None => 1,
        };
        kv.put(Self::seq_key(seq).to_vec(), payload.to_vec());
        kv.put(Self::last_key().to_vec(), seq.to_be_bytes().to_vec());

        Ok(AppendResult {
            seq,
            id: derive_record_id(seq, payload),
            bytes: payload.len(),
        })
    }

    fn get(&self, seq: SeqNo) -> Result<Option<Vec<u8>>> {
        if seq == 0 {
            return Ok(None);
        }
        let kv = self.kv.lock().expect("lock firewood archive");
        Ok(kv.get(&Self::seq_key(seq)))
    }

    fn last_seq(&self) -> Result<SeqNo> {
        let kv = self.kv.lock().expect("lock firewood archive");
        match kv.get(Self::last_key()) {
            Some(bytes) => Self::decode_seq(&bytes),
            None => Ok(0),
        }
    }

    fn flush(&mut self) -> Result<()> {
        let mut kv = self.kv.lock().expect("lock firewood archive");
        kv.commit()
            .map_err(|err| anyhow!("commit firewood: {err}"))?;
        Ok(())
    }
}
