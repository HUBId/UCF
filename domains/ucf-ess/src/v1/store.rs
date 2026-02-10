use ucf_core::types::SimTime;

use crate::v1::{EssError, ExperienceId, ExperienceRecord};

#[allow(clippy::len_without_is_empty)]
pub trait ExperienceStore {
    fn append(&mut self, rec: ExperienceRecord) -> Result<(), EssError>;
    fn len(&self) -> usize;
    fn get(&self, idx: usize) -> Option<&ExperienceRecord>;
    fn tail_time(&self) -> Option<SimTime>;
}

#[derive(Debug, Default)]
pub struct InMemoryEss {
    records: Vec<ExperienceRecord>,
    last_time: Option<SimTime>,
}

impl InMemoryEss {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

impl ExperienceStore for InMemoryEss {
    fn append(&mut self, rec: ExperienceRecord) -> Result<(), EssError> {
        if let Some(last) = self.last_time {
            if rec.time.tick < last.tick
                || (rec.time.tick == last.tick && rec.time.window < last.window)
            {
                return Err(EssError::TimeWentBackwards);
            }
        }

        self.last_time = Some(rec.time);
        self.records.push(rec);
        Ok(())
    }

    fn len(&self) -> usize {
        self.records.len()
    }

    fn get(&self, idx: usize) -> Option<&ExperienceRecord> {
        self.records.get(idx)
    }

    fn tail_time(&self) -> Option<SimTime> {
        self.last_time
    }
}

#[derive(Debug, Clone)]
pub struct IdAllocator {
    next: u64,
}

impl IdAllocator {
    pub fn new(start: u64) -> Self {
        Self { next: start }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> ExperienceId {
        let id = ExperienceId(self.next);
        self.next = self.next.wrapping_add(1);
        id
    }
}
