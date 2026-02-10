use std::collections::BTreeMap;

use ucf_core::types::SimTime;
use ucf_frames::v1::{CorrelationId, DecisionFrame};

use crate::v1::{EssError, ExperienceId, ExperiencePayload, ExperienceRecord};

#[allow(clippy::len_without_is_empty)]
pub trait ExperienceStore {
    fn append(&mut self, rec: ExperienceRecord) -> Result<(), EssError>;
    fn len(&self) -> usize;
    fn get(&self, idx: usize) -> Option<&ExperienceRecord>;
    fn tail_time(&self) -> Option<SimTime>;
    fn indices_by_corr(&self, corr: CorrelationId) -> &[usize];
    fn trail_by_corr(&self, corr: CorrelationId) -> Vec<&ExperienceRecord>;
    fn last_decision_for_corr(&self, corr: CorrelationId) -> Option<&DecisionFrame>;
}

#[derive(Debug, Default)]
pub struct InMemoryEss {
    records: Vec<ExperienceRecord>,
    last_time: Option<SimTime>,
    by_corr: BTreeMap<CorrelationId, Vec<usize>>,
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

        let rec_idx = self.records.len();
        let rec_corr = rec.corr;
        self.last_time = Some(rec.time);
        self.records.push(rec);
        self.by_corr.entry(rec_corr).or_default().push(rec_idx);
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

    fn indices_by_corr(&self, corr: CorrelationId) -> &[usize] {
        const EMPTY: [usize; 0] = [];
        self.by_corr.get(&corr).map_or(&EMPTY, Vec::as_slice)
    }

    fn trail_by_corr(&self, corr: CorrelationId) -> Vec<&ExperienceRecord> {
        self.indices_by_corr(corr)
            .iter()
            .filter_map(|idx| self.records.get(*idx))
            .collect()
    }

    fn last_decision_for_corr(&self, corr: CorrelationId) -> Option<&DecisionFrame> {
        self.indices_by_corr(corr)
            .iter()
            .rev()
            .filter_map(|idx| self.records.get(*idx))
            .find_map(|record| match &record.payload {
                ExperiencePayload::Decision(decision) => Some(decision),
                _ => None,
            })
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
