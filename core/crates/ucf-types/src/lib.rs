#![forbid(unsafe_code)]

use sha2::{Digest, Sha256};
use std::fmt;

pub mod error_codes;
pub mod record_schema;
pub mod remediation_codes;

pub const CANONICAL_QNAN_BITS_F32: u32 = 0x7FC0_0000;
pub const CANONICAL_UNIT_QUANT_MAX: u16 = u16::MAX;
pub const CANONICAL_SIGNED_UNIT_QUANT_MAX: i16 = i16::MAX;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UQ0_16(pub u16);

impl UQ0_16 {
    pub const SCALE: u32 = 1 << 16;
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(u16::MAX);

    pub const fn from_raw(raw: u16) -> Self {
        Self(raw)
    }

    pub const fn raw(self) -> u16 {
        self.0
    }

    pub fn from_f32_clamped(value: f32) -> Self {
        Self(quantize_unit(value, CANONICAL_UNIT_QUANT_MAX))
    }

    pub fn to_f32(self) -> f32 {
        f32::from(self.0) / f32::from(CANONICAL_UNIT_QUANT_MAX)
    }

    pub fn saturating_add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }

    pub fn saturating_mul(self, rhs: Self) -> Self {
        let lhs = u32::from(self.0);
        let rhs = u32::from(rhs.0);
        let product = lhs.saturating_mul(rhs);
        let scaled = (product.saturating_add(1 << 15)) >> 16;
        Self((scaled.min(u32::from(u16::MAX))) as u16)
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Q16_16(pub i32);

impl Q16_16 {
    pub const SCALE: i64 = 1 << 16;

    pub const fn from_raw(raw: i32) -> Self {
        Self(raw)
    }

    pub const fn raw(self) -> i32 {
        self.0
    }

    pub fn from_f32_clamped(value: f32) -> Self {
        let scaled = (f64::from(value) * Self::SCALE as f64).round();
        Self(scaled.clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32)
    }

    pub fn to_f32(self) -> f32 {
        self.0 as f32 / Self::SCALE as f32
    }

    pub fn saturating_add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }

    pub fn saturating_mul(self, rhs: Self) -> Self {
        let product = i64::from(self.0).saturating_mul(i64::from(rhs.0));
        let adjusted = if product >= 0 {
            product.saturating_add(1 << 15)
        } else {
            product.saturating_sub(1 << 15)
        };
        let scaled = adjusted >> 16;
        Self(scaled.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CanonicalF32(pub u32);

impl CanonicalF32 {
    pub fn from_f32(value: f32) -> Self {
        Self(canonicalize_f32(value))
    }

    pub fn bits(self) -> u32 {
        self.0
    }
}

pub fn canonicalize_f32(value: f32) -> u32 {
    if value == 0.0 {
        return 0.0f32.to_bits();
    }
    if value.is_nan() {
        return CANONICAL_QNAN_BITS_F32;
    }
    value.to_bits()
}

pub fn canonicalize_f32_clamped(value: f32, min: f32, max: f32) -> u32 {
    if value.is_nan() {
        return CANONICAL_QNAN_BITS_F32;
    }
    canonicalize_f32(value.clamp(min, max))
}

pub fn quantize_unit(value: f32, q: u16) -> u16 {
    let clamped = if value.is_nan() {
        0.0
    } else {
        value.clamp(0.0, 1.0)
    };
    let scaled = (clamped * f32::from(q)).round();
    if scaled <= 0.0 {
        0
    } else if scaled >= f32::from(q) {
        q
    } else {
        scaled as u16
    }
}

pub fn quantize_signed_unit_i16(value: f32) -> i16 {
    let clamped = if value.is_nan() {
        0.0
    } else {
        value.clamp(-1.0, 1.0)
    };
    let scaled = (clamped * f32::from(CANONICAL_SIGNED_UNIT_QUANT_MAX)).round();
    scaled.clamp(
        f32::from(i16::MIN) + 1.0,
        f32::from(CANONICAL_SIGNED_UNIT_QUANT_MAX),
    ) as i16
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SignalBundleV1 {
    pub t: u64,
    pub policy_graph_digest: [u8; 32],
    pub risk_q: UQ0_16,
    pub confidence_q: UQ0_16,
    pub surprise_q: UQ0_16,
    pub pressure_q: UQ0_16,
    pub uncertainty_q: UQ0_16,
    pub stability_q: UQ0_16,
    pub coherence_q: Option<UQ0_16>,
    pub world_prediction_digest: [u8; 32],
    pub sae_spikes_digest: [u8; 32],
    pub ssm_state_digest: [u8; 32],
    pub lfm_state_digest: [u8; 32],
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SlotModeV1 {
    Off,
    Shadow,
    Active,
}

impl SlotModeV1 {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Shadow => "shadow",
            Self::Active => "active",
        }
    }
}

impl SignalBundleV1 {
    pub fn signals_digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.t.to_be_bytes());
        hasher.update(self.policy_graph_digest);
        hasher.update(self.risk_q.raw().to_be_bytes());
        hasher.update(self.confidence_q.raw().to_be_bytes());
        hasher.update(self.surprise_q.raw().to_be_bytes());
        hasher.update(self.pressure_q.raw().to_be_bytes());
        hasher.update(self.uncertainty_q.raw().to_be_bytes());
        hasher.update(self.stability_q.raw().to_be_bytes());
        match self.coherence_q {
            Some(value) => {
                hasher.update([1]);
                hasher.update(value.raw().to_be_bytes());
            }
            None => hasher.update([0]),
        }
        hasher.update(self.world_prediction_digest);
        hasher.update(self.sae_spikes_digest);
        hasher.update(self.ssm_state_digest);
        hasher.update(self.lfm_state_digest);
        hasher.finalize().into()
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RiskConfidenceV1 {
    pub risk_q: UQ0_16,
    pub confidence_q: UQ0_16,
    pub update_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RiskConfidenceCoefficientsV1 {
    pub base_risk_q: UQ0_16,
    pub alpha_q: UQ0_16,
    pub beta_q: UQ0_16,
    pub gamma_q: UQ0_16,
}

impl RiskConfidenceV1 {
    pub fn update(
        previous: Option<Self>,
        signal_bundle: &SignalBundleV1,
        policy_graph_digest: [u8; 32],
        coeffs: RiskConfidenceCoefficientsV1,
    ) -> Self {
        let risk_adjust = coeffs
            .alpha_q
            .saturating_mul(signal_bundle.surprise_q)
            .saturating_add(coeffs.beta_q.saturating_mul(signal_bundle.pressure_q));
        let risk_q = coeffs.base_risk_q.saturating_add(risk_adjust);
        let confidence_penalty = coeffs.gamma_q.saturating_mul(signal_bundle.uncertainty_q);
        let confidence_q =
            UQ0_16::from_raw(UQ0_16::ONE.raw().saturating_sub(confidence_penalty.raw()));
        let mut hasher = Sha256::new();
        hasher.update(b"ucf.risk_confidence.v1");
        match previous {
            Some(prev) => {
                hasher.update([1]);
                hasher.update(prev.risk_q.raw().to_be_bytes());
                hasher.update(prev.confidence_q.raw().to_be_bytes());
                hasher.update(prev.update_digest);
            }
            None => hasher.update([0]),
        }
        hasher.update(signal_bundle.signals_digest());
        hasher.update(policy_graph_digest);
        hasher.update(coeffs.base_risk_q.raw().to_be_bytes());
        hasher.update(coeffs.alpha_q.raw().to_be_bytes());
        hasher.update(coeffs.beta_q.raw().to_be_bytes());
        hasher.update(coeffs.gamma_q.raw().to_be_bytes());
        let update_digest = hasher.finalize().into();
        Self {
            risk_q,
            confidence_q,
            update_digest,
        }
    }
}

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
    use proptest::prelude::*;

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
    fn canonicalize_f32_normalizes_negative_zero_and_nan() {
        assert_eq!(canonicalize_f32(-0.0), 0.0f32.to_bits());
        assert_eq!(canonicalize_f32(f32::NAN), CANONICAL_QNAN_BITS_F32);
    }

    #[test]
    fn quantize_unit_clamps_nan_and_bounds() {
        assert_eq!(quantize_unit(f32::NAN, CANONICAL_UNIT_QUANT_MAX), 0);
        assert_eq!(quantize_unit(-1.0, CANONICAL_UNIT_QUANT_MAX), 0);
        assert_eq!(
            quantize_unit(2.0, CANONICAL_UNIT_QUANT_MAX),
            CANONICAL_UNIT_QUANT_MAX
        );
    }

    #[test]
    fn vrf_tag_requires_suite_and_domain() {
        let tag = Digest32::new([1u8; 32]);
        assert!(VrfTag::new(0, 1, tag).is_err());
        assert!(VrfTag::new(1, 0, tag).is_err());
        assert!(VrfTag::new(1, 1, tag).is_ok());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn quantize_unit_is_monotonic_and_idempotent(a in -10.0f32..10.0, b in -10.0f32..10.0) {
            let qa = quantize_unit(a, CANONICAL_UNIT_QUANT_MAX);
            let qb = quantize_unit(b, CANONICAL_UNIT_QUANT_MAX);
            if a <= b {
                prop_assert!(qa <= qb);
            }
            let clamped = a.clamp(0.0, 1.0);
            prop_assert_eq!(qa, quantize_unit(clamped, CANONICAL_UNIT_QUANT_MAX));
        }
    }
}

#[cfg(test)]
mod fixed_point_tests {
    use super::{Q16_16, UQ0_16};

    #[test]
    fn uq0_16_roundtrip_and_clamp() {
        assert_eq!(UQ0_16::from_f32_clamped(-1.0).raw(), 0);
        assert_eq!(UQ0_16::from_f32_clamped(2.0).raw(), u16::MAX);
        let mid = UQ0_16::from_f32_clamped(0.5);
        assert!((mid.to_f32() - 0.5).abs() < 1e-4);
    }

    #[test]
    fn uq0_16_mul_is_deterministic() {
        let a = UQ0_16::from_f32_clamped(0.75);
        let b = UQ0_16::from_f32_clamped(0.5);
        let out = a.saturating_mul(b);
        assert!((out.to_f32() - 0.375).abs() < 1e-4);
    }

    #[test]
    fn q16_16_saturating_mul_and_add() {
        let a = Q16_16::from_f32_clamped(1.5);
        let b = Q16_16::from_f32_clamped(-0.25);
        let prod = a.saturating_mul(b);
        assert!((prod.to_f32() + 0.375).abs() < 1e-4);
        let sum = Q16_16::from_raw(i32::MAX).saturating_add(Q16_16::from_raw(1));
        assert_eq!(sum.raw(), i32::MAX);
    }
}

#[cfg(test)]
mod signal_bundle_tests {
    use super::{SignalBundleV1, UQ0_16};

    fn sample_bundle() -> SignalBundleV1 {
        SignalBundleV1 {
            t: 42,
            policy_graph_digest: [1; 32],
            risk_q: UQ0_16::from_raw(11),
            confidence_q: UQ0_16::from_raw(12),
            surprise_q: UQ0_16::from_raw(13),
            pressure_q: UQ0_16::from_raw(14),
            uncertainty_q: UQ0_16::from_raw(15),
            stability_q: UQ0_16::from_raw(16),
            coherence_q: None,
            world_prediction_digest: [2; 32],
            sae_spikes_digest: [3; 32],
            ssm_state_digest: [4; 32],
            lfm_state_digest: [5; 32],
        }
    }

    #[test]
    fn signal_bundle_digest_is_deterministic() {
        let a = sample_bundle().signals_digest();
        let b = sample_bundle().signals_digest();
        assert_eq!(a, b);
    }

    #[test]
    fn signal_bundle_digest_changes_with_field_mutation() {
        let a = sample_bundle().signals_digest();
        let mut changed = sample_bundle();
        changed.pressure_q = UQ0_16::from_raw(99);
        let b = changed.signals_digest();
        assert_ne!(a, b);
    }
}

#[cfg(test)]
mod risk_confidence_tests {
    use super::{RiskConfidenceCoefficientsV1, RiskConfidenceV1, SignalBundleV1, UQ0_16};

    fn sample_bundle() -> SignalBundleV1 {
        SignalBundleV1 {
            t: 9,
            policy_graph_digest: [2; 32],
            risk_q: UQ0_16::from_raw(1000),
            confidence_q: UQ0_16::from_raw(50000),
            surprise_q: UQ0_16::from_raw(20_000),
            pressure_q: UQ0_16::from_raw(30_000),
            uncertainty_q: UQ0_16::from_raw(40_000),
            stability_q: UQ0_16::from_raw(10_000),
            coherence_q: None,
            world_prediction_digest: [3; 32],
            sae_spikes_digest: [4; 32],
            ssm_state_digest: [5; 32],
            lfm_state_digest: [6; 32],
        }
    }

    #[test]
    fn risk_confidence_update_is_bounded() {
        let bundle = sample_bundle();
        let coeffs = RiskConfidenceCoefficientsV1 {
            base_risk_q: UQ0_16::from_raw(60_000),
            alpha_q: UQ0_16::from_raw(60_000),
            beta_q: UQ0_16::from_raw(60_000),
            gamma_q: UQ0_16::from_raw(65_535),
        };
        let out = RiskConfidenceV1::update(None, &bundle, [8; 32], coeffs);
        assert!((0.0..=1.0).contains(&out.risk_q.to_f32()));
        assert!((0.0..=1.0).contains(&out.confidence_q.to_f32()));
    }

    #[test]
    fn risk_confidence_update_is_deterministic() {
        let bundle = sample_bundle();
        let coeffs = RiskConfidenceCoefficientsV1 {
            base_risk_q: UQ0_16::from_raw(10_000),
            alpha_q: UQ0_16::from_raw(5_000),
            beta_q: UQ0_16::from_raw(3_000),
            gamma_q: UQ0_16::from_raw(8_000),
        };
        let first = RiskConfidenceV1::update(None, &bundle, [7; 32], coeffs);
        let second = RiskConfidenceV1::update(None, &bundle, [7; 32], coeffs);
        assert_eq!(first, second);
    }
}
