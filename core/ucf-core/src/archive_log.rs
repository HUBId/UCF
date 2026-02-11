use anyhow::Result;

use crate::storage::{AppendResult, ArchiveCfg, ArchiveStore, FlushPolicy, SeqNo};

pub struct ArchiveLog<S: ArchiveStore> {
    pub cfg: ArchiveCfg,
    pub store: S,
    pub last_flush_ms: u64,
}

impl<S: ArchiveStore> ArchiveLog<S> {
    pub fn new(store: S, cfg: ArchiveCfg) -> Self {
        Self {
            cfg,
            store,
            last_flush_ms: 0,
        }
    }

    pub fn append(&mut self, now_ms: u64, payload: &[u8]) -> Result<AppendResult> {
        let appended = self.store.append(payload)?;

        match self.cfg.flush {
            FlushPolicy::EveryAppend => {
                self.store.flush()?;
                self.last_flush_ms = now_ms;
            }
            FlushPolicy::IntervalMs(interval_ms) => {
                if now_ms.saturating_sub(self.last_flush_ms) >= interval_ms {
                    self.store.flush()?;
                    self.last_flush_ms = now_ms;
                }
            }
            FlushPolicy::Manual => {}
        }

        Ok(appended)
    }

    pub fn get(&self, seq: SeqNo) -> Result<Option<Vec<u8>>> {
        self.store.get(seq)
    }

    pub fn last_seq(&self) -> Result<SeqNo> {
        self.store.last_seq()
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::*;
    use crate::storage::{derive_record_id, ArchiveCfg, FlushPolicy, RecordId};

    #[derive(Default)]
    struct CountingStore {
        flushes: usize,
        records: Vec<Vec<u8>>,
    }

    impl ArchiveStore for CountingStore {
        fn append(&mut self, payload: &[u8]) -> Result<AppendResult> {
            let seq = self.records.len() as u64 + 1;
            self.records.push(payload.to_vec());
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
            Ok(self.records.get((seq - 1) as usize).cloned())
        }

        fn last_seq(&self) -> Result<SeqNo> {
            Ok(self.records.len() as u64)
        }

        fn flush(&mut self) -> Result<()> {
            self.flushes += 1;
            Ok(())
        }
    }

    #[test]
    fn every_append_flushes() {
        let store = CountingStore::default();
        let mut log = ArchiveLog::new(
            store,
            ArchiveCfg {
                flush: FlushPolicy::EveryAppend,
            },
        );

        log.append(10, b"a").expect("append");
        log.append(11, b"b").expect("append");

        assert_eq!(log.store.flushes, 2);
    }

    #[test]
    fn interval_flushes_only_after_elapsed_interval() {
        let store = CountingStore::default();
        let mut log = ArchiveLog::new(
            store,
            ArchiveCfg {
                flush: FlushPolicy::IntervalMs(10),
            },
        );

        log.append(5, b"a").expect("append");
        log.append(9, b"b").expect("append");
        assert_eq!(log.store.flushes, 0);

        log.append(10, b"c").expect("append");
        assert_eq!(log.store.flushes, 1);
    }

    #[test]
    fn manual_never_flushes_implicitly() {
        let store = CountingStore::default();
        let mut log = ArchiveLog::new(
            store,
            ArchiveCfg {
                flush: FlushPolicy::Manual,
            },
        );

        log.append(1, b"a").expect("append");
        log.append(100, b"b").expect("append");
        assert_eq!(log.store.flushes, 0);
    }

    #[test]
    fn append_result_contains_deterministic_id() {
        let store = CountingStore::default();
        let mut log = ArchiveLog::new(store, ArchiveCfg::default());
        let result = log.append(7, b"payload").expect("append");

        let expected = RecordId(derive_record_id(1, b"payload").0);
        assert_eq!(result.id, expected);
    }
}
