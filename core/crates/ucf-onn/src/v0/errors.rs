use core::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnnError {
    InvalidCycleHz,
}

impl fmt::Display for OnnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCycleHz => write!(f, "cycle_hz must be finite and > 0"),
        }
    }
}

impl std::error::Error for OnnError {}
