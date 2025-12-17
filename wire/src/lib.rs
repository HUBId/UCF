#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct Frame {
    pub channel: String,
    pub payload: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct SignedFrame {
    pub frame: Frame,
    pub signature: Option<Vec<u8>>,
}

#[derive(Debug, Error)]
pub enum WireError {
    #[error("frame validation is not implemented")]
    ValidationNotImplemented,
    #[error("io placeholder error: {0}")]
    IoPlaceholder(String),
}

pub trait FrameIo {
    fn send(&mut self, frame: SignedFrame) -> Result<(), WireError>;
    fn receive(&mut self) -> Result<Option<SignedFrame>, WireError>;

    fn verify(&self, _frame: &SignedFrame) -> Result<bool, WireError> {
        Err(WireError::ValidationNotImplemented)
    }

    fn sign(&self, _frame: Frame) -> Result<SignedFrame, WireError> {
        Err(WireError::ValidationNotImplemented)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_round_trip_placeholder() {
        let frame = Frame {
            channel: "rsv".to_string(),
            payload: vec![1, 2, 3],
        };
        let signed = SignedFrame {
            frame: frame.clone(),
            signature: None,
        };
        assert_eq!(signed.frame, frame);
    }
}
