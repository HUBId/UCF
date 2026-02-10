#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrainStimulusKind {
    SpikeTrain,
    ParameterKick,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BrainStimulusPayload {
    pub kind: BrainStimulusKind,
    pub target: u16,
    pub intensity: u16,
    pub duration_ms: u16,
}
