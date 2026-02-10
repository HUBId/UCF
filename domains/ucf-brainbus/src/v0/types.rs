use ucf_core::types::SimTime;
use ucf_frames::v1::{BrainFrame, CorrelationId};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OscPhase {
    pub cycle_hz: f32,
    pub theta_deg: f32,
}

impl OscPhase {
    pub fn new(cycle_hz: f32, theta_deg: f32) -> Self {
        let normalized = theta_deg.rem_euclid(360.0);
        Self {
            cycle_hz,
            theta_deg: normalized,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Spike {
    pub time: SimTime,
    pub corr: CorrelationId,
    pub src: u16,
    pub dst: u16,
    pub code: u16,
    pub phase: Option<OscPhase>,
}

impl Spike {
    pub fn new(time: SimTime, corr: CorrelationId, src: u16, dst: u16, code: u16) -> Self {
        Self {
            time,
            corr,
            src,
            dst,
            code,
            phase: None,
        }
    }

    pub fn with_phase(mut self, phase: OscPhase) -> Self {
        self.phase = Some(phase);
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BrainEvent {
    Spike(Spike),
    Frame(BrainFrame),
}
