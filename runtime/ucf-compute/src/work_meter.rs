use crate::ComputeError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkMeter {
    remaining: u64,
}

impl WorkMeter {
    pub fn new(limit: u64) -> Self {
        Self { remaining: limit }
    }

    pub fn remaining(&self) -> u64 {
        self.remaining
    }

    pub fn spend(&mut self, units: u64, stage: &'static str) -> Result<(), ComputeError> {
        if units > self.remaining {
            return Err(ComputeError::BudgetExceeded {
                stage,
                elapsed_micros: 0,
                limit_micros: self.remaining,
            });
        }
        self.remaining -= units;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spend_tracks_remaining() {
        let mut meter = WorkMeter::new(5);
        meter.spend(2, "x").expect("within limit");
        assert_eq!(meter.remaining(), 3);
    }

    #[test]
    fn spend_reports_budget_exceeded() {
        let mut meter = WorkMeter::new(1);
        let err = meter.spend(2, "sae/extract").expect_err("should exceed");
        assert!(matches!(
            err,
            ComputeError::BudgetExceeded {
                stage: "sae/extract",
                ..
            }
        ));
    }
}
