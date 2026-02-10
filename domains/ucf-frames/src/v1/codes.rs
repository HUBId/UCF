use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionCode {
    Allow,
    Deny,
    Defer,
}

impl Display for DecisionCode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Allow => f.write_str("allow"),
            Self::Deny => f.write_str("deny"),
            Self::Defer => f.write_str("defer"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenyReasonCode {
    MissingDecision,
    PolicyViolation,
    UnsafeContext,
    InvalidIntent,
    InternalError,
}

impl Display for DenyReasonCode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingDecision => f.write_str("missing_decision"),
            Self::PolicyViolation => f.write_str("policy_violation"),
            Self::UnsafeContext => f.write_str("unsafe_context"),
            Self::InvalidIntent => f.write_str("invalid_intent"),
            Self::InternalError => f.write_str("internal_error"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelCode {
    ExternalOutput,
    InternalThought,
    MemoryWrite,
    BrainStimulus,
}

impl Display for ChannelCode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExternalOutput => f.write_str("external_output"),
            Self::InternalThought => f.write_str("internal_thought"),
            Self::MemoryWrite => f.write_str("memory_write"),
            Self::BrainStimulus => f.write_str("brain_stimulus"),
        }
    }
}
