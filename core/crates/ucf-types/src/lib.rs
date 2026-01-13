#![forbid(unsafe_code)]

pub mod v1 {
    pub use ucf_protocol::v1::*;
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(String);

impl NodeId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for NodeId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<NodeId> for String {
    fn from(value: NodeId) -> Self {
        value.0
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StreamId(String);

impl StreamId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for StreamId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<StreamId> for String {
    fn from(value: StreamId) -> Self {
        value.0
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MilestoneId(String);

impl MilestoneId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for MilestoneId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<MilestoneId> for String {
    fn from(value: MilestoneId) -> Self {
        value.0
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EvidenceId(String);

impl EvidenceId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for EvidenceId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<EvidenceId> for String {
    fn from(value: EvidenceId) -> Self {
        value.0
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LogicalTime {
    pub tick: u64,
}

impl LogicalTime {
    pub fn new(tick: u64) -> Self {
        Self { tick }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WallTime {
    pub unix_ms: u64,
}

impl WallTime {
    pub fn new(unix_ms: u64) -> Self {
        Self { unix_ms }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_id_roundtrip() {
        let id = NodeId::new("node-1");
        let raw: String = id.clone().into();
        assert_eq!(raw, "node-1");
        let restored = NodeId::from(raw);
        assert_eq!(restored, id);
    }

    #[test]
    fn logical_time_value() {
        let time = LogicalTime::new(42);
        assert_eq!(time.tick, 42);
    }
}
