#![forbid(unsafe_code)]

use dbm_core::DbmModule;
#[cfg(feature = "microcircuit-dopamin")]
use microcircuit_core::CircuitConfig;
use microcircuit_core::MicrocircuitBackend;
pub use microcircuit_dopamin_stub::{DopaInput, DopaOutput, DopaRules};
use std::fmt;

pub enum DopaBackend {
    Rules(DopaRules),
    Micro(Box<dyn MicrocircuitBackend<DopaInput, DopaOutput>>),
}

impl fmt::Debug for DopaBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DopaBackend::Rules(_) => f.write_str("DopaBackend::Rules"),
            DopaBackend::Micro(_) => f.write_str("DopaBackend::Micro"),
        }
    }
}

impl DopaBackend {
    fn tick(&mut self, input: &DopaInput) -> DopaOutput {
        match self {
            DopaBackend::Rules(rules) => rules.tick(input),
            DopaBackend::Micro(backend) => backend.step(input, 0),
        }
    }
}

#[derive(Debug)]
pub struct DopaminNacc {
    backend: DopaBackend,
}

impl DopaminNacc {
    pub fn new() -> Self {
        Self {
            backend: DopaBackend::Rules(DopaRules::new()),
        }
    }

    #[cfg(feature = "microcircuit-dopamin-attractor")]
    pub fn new_micro(config: CircuitConfig) -> Self {
        use microcircuit_dopamin_attractor::DopaAttractorMicrocircuit;

        Self {
            backend: DopaBackend::Micro(Box::new(DopaAttractorMicrocircuit::new(config))),
        }
    }

    #[cfg(all(
        feature = "microcircuit-dopamin",
        not(feature = "microcircuit-dopamin-attractor")
    ))]
    pub fn new_micro(config: CircuitConfig) -> Self {
        use microcircuit_dopamin_stub::DopaMicrocircuit;

        Self {
            backend: DopaBackend::Micro(Box::new(DopaMicrocircuit::new(config))),
        }
    }

    pub fn snapshot_digest(&self) -> Option<[u8; 32]> {
        match &self.backend {
            DopaBackend::Micro(backend) => Some(backend.snapshot_digest()),
            DopaBackend::Rules(_) => None,
        }
    }

    pub fn config_digest(&self) -> Option<[u8; 32]> {
        match &self.backend {
            DopaBackend::Micro(backend) => Some(backend.config_digest()),
            DopaBackend::Rules(_) => None,
        }
    }
}

impl Default for DopaminNacc {
    fn default() -> Self {
        Self::new()
    }
}

impl DbmModule for DopaminNacc {
    type Input = DopaInput;
    type Output = DopaOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        self.backend.tick(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dbm_core::{IntegrityState, LevelClass};

    fn base_input() -> DopaInput {
        DopaInput {
            ..Default::default()
        }
    }

    #[test]
    fn reward_block_caps_progress_and_incentive() {
        let mut module = DopaminNacc::new();
        let output = module.tick(&DopaInput {
            threat: LevelClass::High,
            exec_success_count_medium: 10,
            ..base_input()
        });

        assert_eq!(output.progress, LevelClass::Med);
        assert_eq!(output.incentive_focus, LevelClass::Low);
        assert!(output
            .reason_codes
            .codes
            .contains(&"RC.GV.PROGRESS.REWARD_BLOCKED".to_string()));
    }

    #[test]
    fn positive_deltas_raise_utility_score() {
        let mut module = DopaminNacc::new();
        let _ = module.tick(&DopaInput {
            exec_success_count_medium: 3,
            exec_failure_count_medium: 1,
            ..base_input()
        });
        let output = module.tick(&DopaInput {
            exec_success_count_medium: 4,
            exec_failure_count_medium: 1,
            ..base_input()
        });

        assert_eq!(output.progress, LevelClass::High);
    }

    #[test]
    fn replay_hint_after_three_nonpositive_deltas() {
        let mut module = DopaminNacc::new();
        for _ in 0..2 {
            let _ = module.tick(&DopaInput {
                exec_failure_count_medium: 1,
                ..base_input()
            });
        }

        let output = module.tick(&DopaInput {
            exec_failure_count_medium: 1,
            ..base_input()
        });

        assert!(output.replay_hint);
        assert!(output
            .reason_codes
            .codes
            .contains(&"RC.GV.REPLAY.DIMINISHING_RETURNS".to_string()));
    }

    #[test]
    fn reason_codes_ordering_is_deterministic() {
        let mut module = DopaminNacc::new();
        let output = module.tick(&DopaInput {
            threat: LevelClass::High,
            replay_mismatch_present: true,
            ..base_input()
        });

        let mut sorted = output.reason_codes.codes.clone();
        sorted.sort();
        assert_eq!(sorted, output.reason_codes.codes);
    }

    #[test]
    fn deterministic_outputs() {
        let mut module_a = DopaminNacc::new();
        let mut module_b = DopaminNacc::new();
        let input = DopaInput {
            exec_success_count_medium: 2,
            exec_failure_count_medium: 1,
            deny_count_medium: 5,
            ..base_input()
        };

        let out_a = module_a.tick(&input);
        let out_b = module_b.tick(&input);

        assert_eq!(out_a, out_b);
    }

    #[test]
    fn noncritical_inputs_allow_progress() {
        let mut module = DopaminNacc::new();
        let output = module.tick(&DopaInput {
            exec_success_count_medium: 3,
            exec_failure_count_medium: 1,
            integrity: IntegrityState::Ok,
            ..base_input()
        });

        assert!(output.progress != LevelClass::Low);
    }
}
