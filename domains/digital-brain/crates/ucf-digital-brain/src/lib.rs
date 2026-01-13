#![forbid(unsafe_code)]

use std::sync::Mutex;

use ucf_types::v1::spec::ExperienceRecord;

pub trait DigitalBrainPort {
    fn ingest(&self, rec: ExperienceRecord);
}

#[derive(Default)]
pub struct InMemoryDigitalBrain {
    recent: Mutex<Vec<ExperienceRecord>>,
}

impl InMemoryDigitalBrain {
    pub fn new() -> Self {
        Self {
            recent: Mutex::new(Vec::new()),
        }
    }

    pub fn records(&self) -> Vec<ExperienceRecord> {
        let guard = self.recent.lock().expect("lock digital brain");
        guard.clone()
    }

    /// TODO: Add microcircuit routing once available in the core runtime.
    pub fn route_microcircuits(&self) {
        drop(self.recent.lock().expect("lock digital brain"));
    }
}

impl DigitalBrainPort for InMemoryDigitalBrain {
    fn ingest(&self, rec: ExperienceRecord) {
        let mut guard = self.recent.lock().expect("lock digital brain");
        guard.push(rec);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ingest_stores_record() {
        let brain = InMemoryDigitalBrain::new();
        let record = ExperienceRecord {
            record_id: "rec-1".to_string(),
            observed_at_ms: 1,
            subject_id: "subject-1".to_string(),
            payload: vec![9, 9, 9],
            digest: None,
            vrf_tag: None,
        };

        brain.ingest(record.clone());

        let records = brain.records();
        assert_eq!(records, vec![record]);
    }
}
