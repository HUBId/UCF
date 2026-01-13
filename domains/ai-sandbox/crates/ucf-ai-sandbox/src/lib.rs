#![forbid(unsafe_code)]

use std::sync::Mutex;

use ucf_types::v1::spec::ControlFrame;

pub trait AiSandboxPort {
    fn submit_control_frame(&self, cf: ControlFrame) -> Result<(), SandboxError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SandboxError {
    StoreUnavailable,
}

#[derive(Default)]
pub struct InMemoryAiSandbox {
    last_frame: Mutex<Option<ControlFrame>>,
}

impl InMemoryAiSandbox {
    pub fn new() -> Self {
        Self {
            last_frame: Mutex::new(None),
        }
    }

    pub fn last_control_frame(&self) -> Option<ControlFrame> {
        let guard = self.last_frame.lock().expect("lock sandbox");
        guard.clone()
    }
}

impl AiSandboxPort for InMemoryAiSandbox {
    fn submit_control_frame(&self, cf: ControlFrame) -> Result<(), SandboxError> {
        let mut guard = self.last_frame.lock().expect("lock sandbox");
        *guard = Some(cf);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submit_stores_last_frame() {
        let sandbox = InMemoryAiSandbox::new();
        let frame = ControlFrame {
            frame_id: "frame-1".to_string(),
            issued_at_ms: 42,
            decision: None,
            evidence_ids: Vec::new(),
            policy_id: "policy-1".to_string(),
        };

        sandbox.submit_control_frame(frame.clone()).expect("submit");

        let stored = sandbox.last_control_frame().expect("stored");
        assert_eq!(stored, frame);
    }
}
