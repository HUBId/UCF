use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IntentId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntentKind {
    Speak,
    Act,
    QueryMemory,
    WriteMemory,
    StimulateBrain,
    System,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntentType {
    InternalThought,
    ExternalCommunicate,
    WriteMemory,
    StimulateBrain,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Intent {
    pub id: IntentId,
    pub kind: IntentKind,
    pub summary: Arc<str>,
}

impl Intent {
    pub fn new(id: IntentId, kind: IntentKind, summary: impl Into<Arc<str>>) -> Self {
        Self {
            id,
            kind,
            summary: summary.into(),
        }
    }
}
