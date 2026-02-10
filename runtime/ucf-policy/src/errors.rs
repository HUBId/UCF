use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyError {
    MissingDecision,
    AdapterError(&'static str),
    InvalidFrame(&'static str),
}

impl Display for PolicyError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingDecision => f.write_str("missing decision"),
            Self::AdapterError(message) => write!(f, "adapter error: {message}"),
            Self::InvalidFrame(message) => write!(f, "invalid frame: {message}"),
        }
    }
}

impl std::error::Error for PolicyError {}
