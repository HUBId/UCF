use std::cmp::Ordering;

use blake3::Hasher;
use ucf_frames::v1::{ControlFrame, DecisionCode, DecisionFrame};

use crate::capability::CapabilityKind;

pub const CANDIDATE_SCHEMA_VERSION_V1: u16 = 1;
pub const MAX_CANDIDATES: usize = 8;
pub const MAX_TOOL_INTENTS: usize = 8;
pub const MAX_RATIONALE_LINES: usize = 4;
pub const MAX_RATIONALE_CHARS: usize = 160;
pub const MAX_TARGET_PREVIEW_CHARS: usize = 48;

pub type CandidateId = u16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum OutputClass {
    SafeText,
    Code,
    ExternalIo,
    ExecIntent,
    Sensitive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum IntentKind {
    Respond,
    Plan,
    QueryEss,
    Consolidate,
    RequestTool,
    Defer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum EffectClass {
    ReadOnly,
    Write,
    Network,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PayloadHint {
    pub bytes_out: Option<u32>,
    pub bytes_in: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetRef {
    pub hash64: u64,
    pub preview: Option<String>,
}

impl TargetRef {
    pub fn new(hash64: u64, preview: Option<&str>) -> Self {
        Self {
            hash64,
            preview: preview.map(|text| bound_text(text, MAX_TARGET_PREVIEW_CHARS)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolIntent {
    pub kind: CapabilityKind,
    pub target: TargetRef,
    pub payload_hint: PayloadHint,
    pub expected_effect: EffectClass,
    pub requires_human: bool,
    pub intent_digest: [u8; 32],
}

impl ToolIntent {
    pub fn new(
        kind: CapabilityKind,
        target: TargetRef,
        payload_hint: PayloadHint,
        expected_effect: EffectClass,
        requires_human: bool,
    ) -> Self {
        let mut value = Self {
            kind,
            target,
            payload_hint,
            expected_effect,
            requires_human,
            intent_digest: [0; 32],
        };
        value.intent_digest = digest_tool_intent(&value);
        value
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CostHint {
    pub compute_units: u32,
    pub tool_calls: u8,
    pub bytes_out: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RationaleSummary {
    pub lines: Vec<String>,
}

impl RationaleSummary {
    pub fn bounded(lines: Vec<&str>) -> Self {
        Self {
            lines: lines
                .into_iter()
                .take(MAX_RATIONALE_LINES)
                .map(|line| bound_text(line, MAX_RATIONALE_CHARS))
                .collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecisionCandidate {
    pub schema_version: u16,
    pub candidate_id: CandidateId,
    pub t: u64,
    pub intent_kind: IntentKind,
    pub output_class: OutputClass,
    pub tool_intents: Vec<ToolIntent>,
    pub estimated_cost: CostHint,
    pub rationale: RationaleSummary,
    pub evidence_chain_digest: [u8; 32],
    pub digest: [u8; 32],
}

impl DecisionCandidate {
    pub fn new(mut candidate: Self) -> Self {
        candidate.schema_version = CANDIDATE_SCHEMA_VERSION_V1;
        candidate.tool_intents.sort_by(tool_intent_sort_key);
        candidate.tool_intents.truncate(MAX_TOOL_INTENTS);
        candidate.rationale.lines = candidate
            .rationale
            .lines
            .iter()
            .take(MAX_RATIONALE_LINES)
            .map(|line| bound_text(line, MAX_RATIONALE_CHARS))
            .collect();
        candidate.digest = digest_candidate(&candidate);
        candidate
    }

    pub fn is_noop(&self) -> bool {
        self.intent_kind == IntentKind::Defer
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidatePolicyHint {
    Block,
    SafeOnly,
    Normal,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CandidateAssessment {
    pub allowed: bool,
    pub policy_hint: CandidatePolicyHint,
    pub risk_adjusted_score: f32,
    pub reasons: Vec<&'static str>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecisionBudget {
    pub max_candidates: usize,
}

impl Default for DecisionBudget {
    fn default() -> Self {
        Self {
            max_candidates: MAX_CANDIDATES,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LiquidContextWindowSummary {
    pub sample_count: u16,
    pub mean_uncertainty: f32,
    pub max_uncertainty: f32,
    pub mean_stability: f32,
    pub rolling_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecisionContext {
    pub now_t: u64,
    pub risk: f32,
    pub confidence: f32,
    pub evidence_chain_digest: [u8; 32],
    pub planning_allowed: bool,
    pub liquid_context: Option<LiquidContextWindowSummary>,
}

pub trait CandidateGenerator {
    fn generate(
        &self,
        _frame: &ControlFrame,
        ctx: &DecisionContext,
        budget: DecisionBudget,
    ) -> Vec<DecisionCandidate>;
}

#[derive(Default)]
pub struct DefaultCandidateGeneratorV0;

impl CandidateGenerator for DefaultCandidateGeneratorV0 {
    fn generate(
        &self,
        frame: &ControlFrame,
        ctx: &DecisionContext,
        budget: DecisionBudget,
    ) -> Vec<DecisionCandidate> {
        let mut candidates = Vec::new();

        candidates.push(DecisionCandidate::new(DecisionCandidate {
            schema_version: CANDIDATE_SCHEMA_VERSION_V1,
            candidate_id: 1,
            t: ctx.now_t,
            intent_kind: IntentKind::Respond,
            output_class: OutputClass::SafeText,
            tool_intents: Vec::new(),
            estimated_cost: CostHint {
                compute_units: 1,
                tool_calls: 0,
                bytes_out: 256,
            },
            rationale: RationaleSummary::bounded(vec!["safe direct response"]),
            evidence_chain_digest: ctx.evidence_chain_digest,
            digest: [0; 32],
        }));

        if ctx.planning_allowed {
            let demo_toolread = match &frame.payload {
                ucf_frames::v1::ControlPayload::Text(text) => text.contains("tool_demo_file_read"),
                _ => false,
            };
            let liquid_uncertainty_high = ctx
                .liquid_context
                .map(|window| window.mean_uncertainty > 0.75 || window.max_uncertainty > 0.9)
                .unwrap_or(false);
            let tool_intents = if demo_toolread {
                let demo_intent = ToolIntent::new(
                    CapabilityKind::FileRead,
                    TargetRef::new(hash64("demo_root:hello.txt"), Some("demo_root:hello.txt")),
                    PayloadHint {
                        bytes_out: Some(128),
                        bytes_in: Some(64),
                    },
                    EffectClass::ReadOnly,
                    false,
                );
                vec![demo_intent.clone(), demo_intent]
            } else if ctx.risk > 0.7 || ctx.confidence < 0.35 || liquid_uncertainty_high {
                Vec::new()
            } else {
                vec![ToolIntent::new(
                    CapabilityKind::FileRead,
                    TargetRef::new(
                        hash64("/workspace/UCF/config"),
                        Some("/workspace/UCF/config"),
                    ),
                    PayloadHint {
                        bytes_out: None,
                        bytes_in: Some(1024),
                    },
                    EffectClass::ReadOnly,
                    false,
                )]
            };
            candidates.push(DecisionCandidate::new(DecisionCandidate {
                schema_version: CANDIDATE_SCHEMA_VERSION_V1,
                candidate_id: 2,
                t: ctx.now_t,
                intent_kind: IntentKind::Plan,
                output_class: if tool_intents.is_empty() {
                    OutputClass::SafeText
                } else {
                    OutputClass::ExternalIo
                },
                tool_intents,
                estimated_cost: if demo_toolread {
                    CostHint {
                        compute_units: 0,
                        tool_calls: 1,
                        bytes_out: 64,
                    }
                } else {
                    CostHint {
                        compute_units: 2,
                        tool_calls: 1,
                        bytes_out: 128,
                    }
                },
                rationale: RationaleSummary::bounded(vec!["deterministic plan candidate"]),
                evidence_chain_digest: ctx.evidence_chain_digest,
                digest: [0; 32],
            }));
        }

        candidates.push(DecisionCandidate::new(DecisionCandidate {
            schema_version: CANDIDATE_SCHEMA_VERSION_V1,
            candidate_id: 3,
            t: ctx.now_t,
            intent_kind: IntentKind::Defer,
            output_class: OutputClass::SafeText,
            tool_intents: Vec::new(),
            estimated_cost: CostHint::default(),
            rationale: RationaleSummary::bounded(vec!["defer/no-op fallback"]),
            evidence_chain_digest: ctx.evidence_chain_digest,
            digest: [0; 32],
        }));

        candidates.sort_by_key(|candidate| candidate.candidate_id);
        candidates.truncate(budget.max_candidates.min(MAX_CANDIDATES));
        candidates
    }
}

pub fn assess_candidate(
    candidate: &DecisionCandidate,
    decision: &DecisionFrame,
    nsr_hint_block: bool,
    nsr_hint_safe_only: bool,
) -> CandidateAssessment {
    let mut reasons = Vec::new();
    let mut allowed = !matches!(decision.decision, DecisionCode::Deny);
    let mut policy_hint = CandidatePolicyHint::Normal;

    if nsr_hint_block {
        reasons.push("nsr_block");
        policy_hint = CandidatePolicyHint::Block;
        allowed = candidate.is_noop();
    } else if nsr_hint_safe_only {
        reasons.push("nsr_safe_only");
        policy_hint = CandidatePolicyHint::SafeOnly;
        if candidate.output_class != OutputClass::SafeText {
            allowed = false;
        }
    }

    if matches!(decision.decision, DecisionCode::Defer) && !candidate.is_noop() {
        reasons.push("decision_defer");
        allowed = false;
    }

    let mut score = 1.0 - candidate.estimated_cost.compute_units as f32 * 0.05;
    if candidate.intent_kind == IntentKind::Plan {
        score += 0.1;
    }
    if candidate.is_noop() {
        score -= 0.2;
    }
    if !allowed {
        score -= 1.0;
    }

    CandidateAssessment {
        allowed,
        policy_hint,
        risk_adjusted_score: score,
        reasons,
    }
}

pub fn select_candidate(
    candidates: &[DecisionCandidate],
    assessments: &[CandidateAssessment],
) -> Option<(DecisionCandidate, CandidateAssessment)> {
    candidates
        .iter()
        .zip(assessments)
        .filter(|(_, assessment)| assessment.allowed)
        .max_by(
            |(left_candidate, left_assessment), (right_candidate, right_assessment)| {
                left_assessment
                    .risk_adjusted_score
                    .partial_cmp(&right_assessment.risk_adjusted_score)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| {
                        right_candidate
                            .candidate_id
                            .cmp(&left_candidate.candidate_id)
                    })
            },
        )
        .map(|(candidate, assessment)| (candidate.clone(), assessment.clone()))
}

fn tool_intent_sort_key(left: &ToolIntent, right: &ToolIntent) -> Ordering {
    left.kind
        .as_tag()
        .cmp(right.kind.as_tag())
        .then_with(|| left.target.hash64.cmp(&right.target.hash64))
}

pub fn digest_candidate(candidate: &DecisionCandidate) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(&candidate.schema_version.to_le_bytes());
    hasher.update(&candidate.candidate_id.to_le_bytes());
    hasher.update(&candidate.t.to_le_bytes());
    hasher.update(&(candidate.intent_kind as u8).to_le_bytes());
    hasher.update(&(candidate.output_class as u8).to_le_bytes());
    hasher.update(&candidate.evidence_chain_digest);
    hasher.update(&candidate.estimated_cost.compute_units.to_le_bytes());
    hasher.update(&candidate.estimated_cost.tool_calls.to_le_bytes());
    hasher.update(&candidate.estimated_cost.bytes_out.to_le_bytes());
    for line in &candidate.rationale.lines {
        hasher.update(line.as_bytes());
    }
    for tool_intent in &candidate.tool_intents {
        hasher.update(&tool_intent.intent_digest);
    }
    hasher.finalize().into()
}

pub fn digest_tool_intent(intent: &ToolIntent) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(intent.kind.as_tag().as_bytes());
    hasher.update(&intent.target.hash64.to_le_bytes());
    if let Some(preview) = &intent.target.preview {
        hasher.update(preview.as_bytes());
    }
    hasher.update(&intent.payload_hint.bytes_out.unwrap_or(0).to_le_bytes());
    hasher.update(&intent.payload_hint.bytes_in.unwrap_or(0).to_le_bytes());
    hasher.update(&(intent.expected_effect as u8).to_le_bytes());
    hasher.update(&[u8::from(intent.requires_human)]);
    hasher.finalize().into()
}

fn bound_text(input: &str, max_chars: usize) -> String {
    input.chars().take(max_chars).collect()
}

fn hash64(input: &str) -> u64 {
    let digest = blake3::hash(input.as_bytes());
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest.as_bytes()[..8]);
    u64::from_le_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_core::types::{SimTime, Tick};
    use ucf_frames::v1::{
        ChannelCode, CorrelationId, Intent, IntentId, IntentKind as FrameIntentKind,
    };

    fn frame() -> ControlFrame {
        ControlFrame::new_text(
            SimTime {
                tick: Tick::new(7),
                window: ucf_core::types::WindowId::new(0),
            },
            CorrelationId(9),
            ChannelCode::ExternalOutput,
            Intent::new(IntentId(1), FrameIntentKind::Speak, "test"),
            "ok",
        )
    }

    #[test]
    fn generator_is_deterministic() {
        let ctx = DecisionContext {
            now_t: 7,
            risk: 0.2,
            confidence: 0.8,
            evidence_chain_digest: [3; 32],
            planning_allowed: true,
            liquid_context: None,
        };
        let g = DefaultCandidateGeneratorV0;
        let left = g.generate(&frame(), &ctx, DecisionBudget::default());
        let right = g.generate(&frame(), &ctx, DecisionBudget::default());
        assert_eq!(left, right);
        assert!(left.len() <= MAX_CANDIDATES);
    }

    #[test]
    fn digest_stable_for_tool_ordering() {
        let mut candidate = DecisionCandidate::new(DecisionCandidate {
            schema_version: 1,
            candidate_id: 3,
            t: 1,
            intent_kind: IntentKind::Plan,
            output_class: OutputClass::ExternalIo,
            tool_intents: vec![
                ToolIntent::new(
                    CapabilityKind::NetHttp,
                    TargetRef::new(2, Some("b")),
                    PayloadHint::default(),
                    EffectClass::Network,
                    true,
                ),
                ToolIntent::new(
                    CapabilityKind::FileRead,
                    TargetRef::new(1, Some("a")),
                    PayloadHint::default(),
                    EffectClass::ReadOnly,
                    false,
                ),
            ],
            estimated_cost: CostHint::default(),
            rationale: RationaleSummary::bounded(vec!["x"]),
            evidence_chain_digest: [1; 32],
            digest: [0; 32],
        });
        let digest = candidate.digest;
        candidate.tool_intents.reverse();
        let reordered = DecisionCandidate::new(candidate);
        assert_eq!(digest, reordered.digest);
    }

    #[test]
    fn tie_breaker_prefers_lower_candidate_id() {
        let c1 = DecisionCandidate::new(DecisionCandidate {
            schema_version: 1,
            candidate_id: 1,
            t: 1,
            intent_kind: IntentKind::Respond,
            output_class: OutputClass::SafeText,
            tool_intents: Vec::new(),
            estimated_cost: CostHint::default(),
            rationale: RationaleSummary::bounded(vec!["a"]),
            evidence_chain_digest: [0; 32],
            digest: [0; 32],
        });
        let c2 = DecisionCandidate::new(DecisionCandidate {
            schema_version: 1,
            candidate_id: 2,
            t: 1,
            intent_kind: IntentKind::Respond,
            output_class: OutputClass::SafeText,
            tool_intents: Vec::new(),
            estimated_cost: CostHint::default(),
            rationale: RationaleSummary::bounded(vec!["b"]),
            evidence_chain_digest: [0; 32],
            digest: [0; 32],
        });
        let a1 = CandidateAssessment {
            allowed: true,
            policy_hint: CandidatePolicyHint::Normal,
            risk_adjusted_score: 1.0,
            reasons: Vec::new(),
        };
        let a2 = a1.clone();
        let selected = select_candidate(&[c1, c2], &[a1, a2]).expect("selection");
        assert_eq!(selected.0.candidate_id, 1);
    }
}
