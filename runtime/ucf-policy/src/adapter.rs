use crate::errors::PolicyError;

pub trait ActionAdapter {
    fn emit_text(&mut self, text: &str) -> Result<(), PolicyError>;
    fn emit_brain(&mut self, _frame: ucf_frames::v1::BrainFrame) -> Result<(), PolicyError>;
    fn write_memory(&mut self, _bytes: &[u8]) -> Result<(), PolicyError>;
}

#[derive(Debug, Default)]
pub struct MockAdapter {
    pub emitted: Vec<String>,
    pub brain_events: usize,
    pub mem_writes: usize,
}

impl ActionAdapter for MockAdapter {
    fn emit_text(&mut self, text: &str) -> Result<(), PolicyError> {
        self.emitted.push(text.to_owned());
        Ok(())
    }

    fn emit_brain(&mut self, _frame: ucf_frames::v1::BrainFrame) -> Result<(), PolicyError> {
        self.brain_events += 1;
        Ok(())
    }

    fn write_memory(&mut self, _bytes: &[u8]) -> Result<(), PolicyError> {
        self.mem_writes += 1;
        Ok(())
    }
}
