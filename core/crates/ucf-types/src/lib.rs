#![forbid(unsafe_code)]

use std::fmt;

pub mod v1 {
    pub use ucf_protocol::v1::*;
}

pub mod consolidation {
    use super::Digest32;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    #[repr(u8)]
    pub enum MilestoneTier {
        Micro = 1,
        Meso = 2,
        Macro = 3,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct MicroMilestone {
        pub items: Vec<Digest32>,
        pub horm_profile: Digest32,
        pub commit: Digest32,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct MesoMilestone {
        pub micros: Vec<Digest32>,
        pub topic_commit: Digest32,
        pub commit: Digest32,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct MacroMilestone {
        pub mesos: Vec<Digest32>,
        pub trait_updates: Digest32,
        pub commit: Digest32,
    }

    /// ReplayToken never carries raw user content; only digests and bounded metadata.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct ReplayToken {
        pub tier: MilestoneTier,
        pub target: Digest32,
        pub budget: u16,
        pub redaction: u16,
        pub commit: Digest32,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct ReplayScheduled {
        pub tier: MilestoneTier,
        pub target: Digest32,
        pub budget: u16,
        pub redaction: u16,
        pub commit: Digest32,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct ReplayApplied {
        pub tier: MilestoneTier,
        pub target: Digest32,
        pub effect_digest: Digest32,
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlgoId {
    Blake3_256,
    Sha256,
    Reserved(u16),
}

impl AlgoId {
    pub const BLAKE3_256_ID: u16 = 1;
    pub const SHA256_ID: u16 = 2;

    pub fn id(self) -> u16 {
        match self {
            Self::Blake3_256 => Self::BLAKE3_256_ID,
            Self::Sha256 => Self::SHA256_ID,
            Self::Reserved(id) => id,
        }
    }

    pub fn from_id(id: u16) -> Self {
        match id {
            Self::BLAKE3_256_ID => Self::Blake3_256,
            Self::SHA256_ID => Self::Sha256,
            other => Self::Reserved(other),
        }
    }
}

impl fmt::Display for AlgoId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Blake3_256 => write!(f, "blake3-256"),
            Self::Sha256 => write!(f, "sha256"),
            Self::Reserved(id) => write!(f, "reserved({id})"),
        }
    }
}

impl fmt::Debug for AlgoId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DigestInvariantError {
    InvalidLength { expected: usize, actual: usize },
    UnsetDomain,
    UnsetSuite,
}

impl fmt::Display for DigestInvariantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength { expected, actual } => {
                write!(f, "expected {expected} bytes, got {actual}")
            }
            Self::UnsetDomain => write!(f, "domain id must be non-zero"),
            Self::UnsetSuite => write!(f, "suite id must be non-zero"),
        }
    }
}

impl std::error::Error for DigestInvariantError {}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Digest32([u8; 32]);

impl Digest32 {
    pub const LEN: usize = 32;

    pub fn new(value: [u8; 32]) -> Self {
        Self(value)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl TryFrom<Vec<u8>> for Digest32 {
    type Error = DigestInvariantError;

    fn try_from(value: Vec<u8>) -> Result<Self, Self::Error> {
        if value.len() != Self::LEN {
            return Err(DigestInvariantError::InvalidLength {
                expected: Self::LEN,
                actual: value.len(),
            });
        }
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&value);
        Ok(Self(bytes))
    }
}

impl fmt::Display for Digest32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{}..", hex_prefix(&self.0))
    }
}

impl fmt::Debug for Digest32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Digest32({})", self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GainBudget {
    /// master multiplier 0..10000 (10000 = no reduction)
    pub master: u16,
    /// per-channel caps 0..10000
    pub coupling: u16,
    pub ssm_update: u16,
    pub ncde: u16,
    pub tcf_attention: u16,
    pub tcf_learning: u16,
    pub onn_coupling: u16,
    pub commit: Digest32,
}

impl GainBudget {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        master: u16,
        coupling: u16,
        ssm_update: u16,
        ncde: u16,
        tcf_attention: u16,
        tcf_learning: u16,
        onn_coupling: u16,
        commit: Digest32,
    ) -> Self {
        Self {
            master,
            coupling,
            ssm_update,
            ncde,
            tcf_attention,
            tcf_learning,
            onn_coupling,
            commit,
        }
    }

    pub fn apply(value: u16, budget: u16) -> u16 {
        let scaled = (u32::from(value) * u32::from(budget)) / 10_000;
        u16::try_from(scaled.min(u32::from(u16::MAX))).unwrap_or(u16::MAX)
    }

    pub fn apply_i16(value: i16, budget: u16) -> i16 {
        let scaled = (i32::from(value) * i32::from(budget)) / 10_000;
        scaled.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
    }
}

impl Default for GainBudget {
    fn default() -> Self {
        Self {
            master: 10_000,
            coupling: 10_000,
            ssm_update: 10_000,
            ncde: 10_000,
            tcf_attention: 10_000,
            tcf_learning: 10_000,
            onn_coupling: 10_000,
            commit: Digest32::new([0u8; 32]),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LearningSignal {
    pub cycle_id: u64,
    /// 0..10000
    pub learn_rate: u16,
    /// 0..10000
    pub update_mass: u16,
    /// derived severity: 0..2 (0=stable,1=adapt,2=consolidate)
    pub mode: u8,
    pub commit: Digest32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StructuralDelta {
    pub cycle_id: u64,
    /// deterministic structure change proxy
    pub delta_root: Digest32,
    /// 0..10000
    pub delta_mass: u16,
    /// top-k deterministic targets (0=none,1=onn,2=ssm,3=spike,4=tcf)
    pub targets: [u16; 4],
    pub commit: Digest32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputChannel {
    Thought,
    Speech,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AiOutput {
    pub channel: OutputChannel,
    pub content: String,
    pub confidence: u16,
    pub rationale_commit: Option<Digest32>,
    pub integration_score: Option<u16>,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct DomainDigest {
    pub algo: AlgoId,
    pub domain: u16,
    pub digest: Digest32,
}

impl DomainDigest {
    pub fn new(algo: AlgoId, domain: u16, digest: Digest32) -> Result<Self, DigestInvariantError> {
        if domain == 0 {
            return Err(DigestInvariantError::UnsetDomain);
        }
        Ok(Self {
            algo,
            domain,
            digest,
        })
    }
}

impl fmt::Display for DomainDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DomainDigest(algo={}, domain={}, digest={})",
            self.algo, self.domain, self.digest
        )
    }
}

impl fmt::Debug for DomainDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct VrfTag {
    pub suite: u16,
    pub domain: u16,
    pub tag: Digest32,
}

impl VrfTag {
    pub fn new(suite: u16, domain: u16, tag: Digest32) -> Result<Self, DigestInvariantError> {
        if suite == 0 {
            return Err(DigestInvariantError::UnsetSuite);
        }
        if domain == 0 {
            return Err(DigestInvariantError::UnsetDomain);
        }
        Ok(Self { suite, domain, tag })
    }
}

impl fmt::Display for VrfTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VrfTag(suite={}, domain={}, tag={})",
            self.suite, self.domain, self.tag
        )
    }
}

impl fmt::Debug for VrfTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

fn hex_prefix(bytes: &[u8]) -> String {
    bytes
        .iter()
        .take(4)
        .map(|byte| format!("{byte:02x}"))
        .collect()
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

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorldStateVec {
    pub bytes: Vec<u8>,
    pub dims: Vec<usize>,
}

impl WorldStateVec {
    pub fn new(bytes: Vec<u8>, dims: Vec<usize>) -> Self {
        Self { bytes, dims }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThoughtVec {
    pub bytes: Vec<u8>,
}

impl ThoughtVec {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Claim {
    pub predicate: String,
    pub args: Vec<String>,
}

impl Claim {
    pub fn new(predicate: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            predicate: predicate.into(),
            args,
        }
    }

    pub fn new_from_strs(predicate: impl Into<String>, args: Vec<&str>) -> Self {
        Self::new(predicate, args.into_iter().map(String::from).collect())
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymbolicClaims {
    pub claims: Vec<Claim>,
}

impl SymbolicClaims {
    pub fn new(claims: Vec<Claim>) -> Self {
        Self { claims }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CausalNode {
    pub id: String,
}

impl CausalNode {
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CausalEdge {
    pub from: String,
    pub to: String,
}

impl CausalEdge {
    pub fn new(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CausalGraphStub {
    pub nodes: Vec<CausalNode>,
    pub edges: Vec<CausalEdge>,
}

impl CausalGraphStub {
    pub fn new(nodes: Vec<CausalNode>, edges: Vec<CausalEdge>) -> Self {
        Self { nodes, edges }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CausalIntervention {
    pub node: u32,
    pub value: i32,
}

impl CausalIntervention {
    pub fn new(node: u32, value: i32) -> Self {
        Self { node, value }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CausalCounterfactual {
    pub interventions: Vec<CausalIntervention>,
    pub target: u32,
    pub predicted: i32,
    pub confidence: u16,
}

impl CausalCounterfactual {
    pub fn new(
        interventions: Vec<CausalIntervention>,
        target: u32,
        predicted: i32,
        confidence: u16,
    ) -> Self {
        Self {
            interventions,
            target,
            predicted,
            confidence,
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CausalReport {
    pub dag_commit: Digest32,
    pub counterfactual: Option<CausalCounterfactual>,
}

impl CausalReport {
    pub fn new(dag_commit: Digest32, counterfactual: Option<CausalCounterfactual>) -> Self {
        Self {
            dag_commit,
            counterfactual,
        }
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

    #[test]
    fn digest32_enforces_length() {
        let ok = Digest32::try_from(vec![0u8; 32]).expect("digest32");
        assert_eq!(ok.as_bytes().len(), 32);

        let err = Digest32::try_from(vec![0u8; 31]).expect_err("length error");
        assert_eq!(
            err,
            DigestInvariantError::InvalidLength {
                expected: 32,
                actual: 31
            }
        );
    }

    #[test]
    fn domain_digest_requires_domain() {
        let digest = Digest32::new([0u8; 32]);
        assert!(DomainDigest::new(AlgoId::Sha256, 0, digest).is_err());
        assert!(DomainDigest::new(AlgoId::Sha256, 7, digest).is_ok());
    }

    #[test]
    fn vrf_tag_requires_suite_and_domain() {
        let tag = Digest32::new([1u8; 32]);
        assert!(VrfTag::new(0, 1, tag).is_err());
        assert!(VrfTag::new(1, 0, tag).is_err());
        assert!(VrfTag::new(1, 1, tag).is_ok());
    }
}
