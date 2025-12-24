#![forbid(unsafe_code)]

use dbm_core::DbmModule;
use microcircuit_cerebellum_stub::CerebellumRules;
pub use microcircuit_cerebellum_stub::{CerInput, CerOutput, ToolFailureCounts};
#[cfg(feature = "microcircuit-cerebellum-pop")]
use microcircuit_core::CircuitConfig;
use microcircuit_core::MicrocircuitBackend;
use std::fmt;

pub enum CerebellumBackend {
    Rules(CerebellumRules),
    Micro(Box<dyn MicrocircuitBackend<CerInput, CerOutput>>),
}

impl fmt::Debug for CerebellumBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CerebellumBackend::Rules(_) => f.write_str("CerebellumBackend::Rules"),
            CerebellumBackend::Micro(_) => f.write_str("CerebellumBackend::Micro"),
        }
    }
}

impl CerebellumBackend {
    fn tick(&mut self, input: &CerInput) -> CerOutput {
        match self {
            CerebellumBackend::Rules(rules) => rules.tick(input),
            CerebellumBackend::Micro(backend) => backend.step(input, 0),
        }
    }
}

#[derive(Debug)]
pub struct Cerebellum {
    backend: CerebellumBackend,
}

impl Cerebellum {
    pub fn new() -> Self {
        Self {
            backend: CerebellumBackend::Rules(CerebellumRules::new()),
        }
    }

    #[cfg(feature = "microcircuit-cerebellum-pop")]
    pub fn new_micro(config: CircuitConfig) -> Self {
        use microcircuit_cerebellum_pop::CerebellumPopMicrocircuit;

        Self {
            backend: CerebellumBackend::Micro(Box::new(CerebellumPopMicrocircuit::new(config))),
        }
    }

    pub fn snapshot_digest(&self) -> Option<[u8; 32]> {
        match &self.backend {
            CerebellumBackend::Micro(backend) => Some(backend.snapshot_digest()),
            CerebellumBackend::Rules(_) => None,
        }
    }

    pub fn config_digest(&self) -> Option<[u8; 32]> {
        match &self.backend {
            CerebellumBackend::Micro(backend) => Some(backend.config_digest()),
            CerebellumBackend::Rules(_) => None,
        }
    }
}

impl Default for Cerebellum {
    fn default() -> Self {
        Self::new()
    }
}

impl DbmModule for Cerebellum {
    type Input = CerInput;
    type Output = CerOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        self.backend.tick(input)
    }
}
