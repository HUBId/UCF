use std::sync::Arc;

use ucf_core::types::SimTime;
use ucf_frames::v1::{BrainFrame, ControlFrame, CorrelationId, DecisionFrame};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExperienceId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperienceKind {
    ControlIn,
    DecisionOut,
    BrainOut,
    Note,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExperienceRecord {
    pub id: ExperienceId,
    pub time: SimTime,
    pub corr: CorrelationId,
    pub kind: ExperienceKind,
    pub payload: ExperiencePayload,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExperiencePayload {
    Control(ControlFrame),
    Decision(DecisionFrame),
    Brain(BrainFrame),
    Text(Arc<str>),
    Empty,
}

impl ExperienceRecord {
    pub fn from_control(id: ExperienceId, ctrl: ControlFrame) -> Self {
        Self {
            id,
            time: ctrl.time,
            corr: ctrl.corr,
            kind: ExperienceKind::ControlIn,
            payload: ExperiencePayload::Control(ctrl),
        }
    }

    pub fn from_decision(id: ExperienceId, decision: DecisionFrame) -> Self {
        Self {
            id,
            time: decision.time,
            corr: decision.corr,
            kind: ExperienceKind::DecisionOut,
            payload: ExperiencePayload::Decision(decision),
        }
    }

    pub fn from_brain(id: ExperienceId, brain: BrainFrame) -> Self {
        Self {
            id,
            time: brain.time,
            corr: brain.corr,
            kind: ExperienceKind::BrainOut,
            payload: ExperiencePayload::Brain(brain),
        }
    }

    pub fn note(
        id: ExperienceId,
        time: SimTime,
        corr: CorrelationId,
        text: impl Into<Arc<str>>,
    ) -> Self {
        Self {
            id,
            time,
            corr,
            kind: ExperienceKind::Note,
            payload: ExperiencePayload::Text(text.into()),
        }
    }
}
