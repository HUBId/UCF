use crate::errors::PolicyError;

pub trait ActionAdapter {
    fn emit_text(&mut self, text: &str) -> Result<(), PolicyError>;
    fn emit_brain(&mut self, _frame: ucf_frames::v1::BrainFrame) -> Result<(), PolicyError>;
    fn write_memory(&mut self, _bytes: &[u8]) -> Result<(), PolicyError>;
    fn emit_brain_spikes(
        &mut self,
        _spikes: Vec<ucf_brainbus::v0::Spike>,
    ) -> Result<(), PolicyError>;
    fn take_brain_spike_meta(&mut self) -> Option<(usize, u16)> {
        None
    }
}

#[derive(Debug, Default)]
pub struct MockAdapter {
    pub emitted: Vec<String>,
    pub brain_events: usize,
    pub mem_writes: usize,
    pub brain_spikes: Vec<ucf_brainbus::v0::Spike>,
    last_brain_spike_batch: Option<(usize, u16)>,
}

impl MockAdapter {
    pub fn brain_spikes(&self) -> &[ucf_brainbus::v0::Spike] {
        &self.brain_spikes
    }

    pub fn take_brain_spike_meta(&mut self) -> Option<(usize, u16)> {
        self.last_brain_spike_batch.take()
    }
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

    fn emit_brain_spikes(
        &mut self,
        spikes: Vec<ucf_brainbus::v0::Spike>,
    ) -> Result<(), PolicyError> {
        let meta = spikes.first().map(|s| (spikes.len(), s.dst));
        self.brain_spikes.extend(spikes);
        self.last_brain_spike_batch = meta;
        Ok(())
    }

    fn take_brain_spike_meta(&mut self) -> Option<(usize, u16)> {
        self.last_brain_spike_batch.take()
    }
}
