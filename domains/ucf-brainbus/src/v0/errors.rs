use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrainBusError {
    InvalidPhase,
    QueueFull,
}

impl Display for BrainBusError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPhase => write!(f, "invalid phase value"),
            Self::QueueFull => write!(f, "brain bus queue is full"),
        }
    }
}

impl Error for BrainBusError {}
