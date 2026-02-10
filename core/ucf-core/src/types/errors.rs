use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum CoreError {
    InvalidId(&'static str),
    InvalidFixed(&'static str),
    InvalidTime(&'static str),
}

impl Display for CoreError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            CoreError::InvalidId(msg) => write!(f, "invalid id: {msg}"),
            CoreError::InvalidFixed(msg) => write!(f, "invalid fixed-point value: {msg}"),
            CoreError::InvalidTime(msg) => write!(f, "invalid time value: {msg}"),
        }
    }
}

impl Error for CoreError {}
