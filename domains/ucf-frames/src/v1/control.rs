use std::sync::Arc;

use ucf_core::types::SimTime;

use crate::v1::{BrainStimulusPayload, ChannelCode, Intent};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CorrelationId(pub u64);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlFrame {
    pub time: SimTime,
    pub corr: CorrelationId,
    pub channel: ChannelCode,
    pub intent: Intent,
    pub payload: ControlPayload,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlPayload {
    Text(Arc<str>),
    Bytes(Arc<[u8]>),
    BrainStimulus(BrainStimulusPayload),
    Empty,
}

impl ControlFrame {
    pub fn new_text(
        time: SimTime,
        corr: CorrelationId,
        channel: ChannelCode,
        intent: Intent,
        text: impl Into<Arc<str>>,
    ) -> Self {
        Self {
            time,
            corr,
            channel,
            intent,
            payload: ControlPayload::Text(text.into()),
        }
    }

    pub fn new_empty(
        time: SimTime,
        corr: CorrelationId,
        channel: ChannelCode,
        intent: Intent,
    ) -> Self {
        Self {
            time,
            corr,
            channel,
            intent,
            payload: ControlPayload::Empty,
        }
    }
}
