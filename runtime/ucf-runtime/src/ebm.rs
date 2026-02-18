use sha2::{Digest, Sha256};
use ucf_compute::{BackendComponentId, StageContractVersion, WorkMeter};
use ucf_policy::candidate::{DecisionCandidate, OutputClass};
use ucf_types::UQ0_16;

pub const EBM_K_MAX: usize = 32;
pub const EBM_TOP_N_MAX: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EbmEnablementMode {
    Off,
    Shadow,
    Compare,
    Active,
}

impl EbmEnablementMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" => Some(Self::Off),
            "shadow" => Some(Self::Shadow),
            "compare" => Some(Self::Compare),
            "active" => Some(Self::Active),
            _ => None,
        }
    }

    pub const fn as_u8(self) -> u8 {
        match self {
            Self::Off => 0,
            Self::Shadow => 1,
            Self::Compare => 2,
            Self::Active => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EbmStatus {
    Ok,
    DegradedFallback,
    Disabled,
    BudgetExceeded,
    Error,
}

impl EbmStatus {
    pub const fn as_u8(self) -> u8 {
        match self {
            Self::Ok => 0,
            Self::DegradedFallback => 1,
            Self::Disabled => 2,
            Self::BudgetExceeded => 3,
            Self::Error => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateKind {
    SafeText,
    Json,
    ToolIntent,
    NoOp,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateFeature {
    pub candidate_id: u16,
    pub candidate_kind: CandidateKind,
    pub tool_class: Option<u8>,
    pub candidate_digest: [u8; 32],
    pub feature_vec_q: Vec<i16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EbmSignals {
    pub risk_q: UQ0_16,
    pub confidence_q: UQ0_16,
    pub pressure_q: UQ0_16,
    pub surprise_q: UQ0_16,
    pub uncertainty_q: UQ0_16,
    pub coherence_q: Option<UQ0_16>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EbmInput {
    pub t: u64,
    pub governor_tier: u8,
    pub emergency_active: bool,
    pub context_digest: [u8; 32],
    pub signals: EbmSignals,
    pub candidates: Vec<CandidateFeature>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EbmOutput {
    pub status: EbmStatus,
    pub energies_q: Vec<UQ0_16>,
    pub best_indices: Vec<u16>,
    pub aggregate_energy_q: UQ0_16,
    pub ebm_digest: [u8; 32],
}

pub trait EbmReasoner {
    fn contract_version(&self) -> StageContractVersion;
    fn backend_id(&self) -> BackendComponentId;
    fn score_candidates(&mut self, input: EbmInput, budget: &mut WorkMeter) -> EbmOutput;
}

#[derive(Debug, Default)]
pub struct CpuEbmStubV0;

const W_RISK_Q: UQ0_16 = UQ0_16::from_raw(20_000);
const W_UNCERT_Q: UQ0_16 = UQ0_16::from_raw(18_000);
const W_PRESS_Q: UQ0_16 = UQ0_16::from_raw(12_000);
const W_SUR_Q: UQ0_16 = UQ0_16::from_raw(10_000);
const TOOL_PENALTY_Q: UQ0_16 = UQ0_16::from_raw(18_000);
const JSON_PENALTY_Q: UQ0_16 = UQ0_16::from_raw(3_000);
const NOOP_BONUS_Q: UQ0_16 = UQ0_16::from_raw(2_500);
const EMERGENCY_TOOL_PENALTY_Q: UQ0_16 = UQ0_16::from_raw(u16::MAX);
const COST_PER_CANDIDATE: u64 = 2;

impl EbmReasoner for CpuEbmStubV0 {
    fn contract_version(&self) -> StageContractVersion {
        StageContractVersion::V1
    }

    fn backend_id(&self) -> BackendComponentId {
        BackendComponentId::StubV0
    }

    fn score_candidates(&mut self, mut input: EbmInput, budget: &mut WorkMeter) -> EbmOutput {
        input.candidates.truncate(EBM_K_MAX);
        if budget
            .spend(
                (input.candidates.len() as u64).saturating_mul(COST_PER_CANDIDATE),
                "ebm/score",
            )
            .is_err()
        {
            return degraded_fallback(
                &input,
                self.contract_version(),
                self.backend_id(),
                EbmStatus::BudgetExceeded,
            );
        }

        let mut scored: Vec<(usize, u16, UQ0_16)> = input
            .candidates
            .iter()
            .enumerate()
            .map(|(idx, c)| (idx, c.candidate_id, score_candidate(&input, c)))
            .collect();
        scored.sort_by(|a, b| a.2.raw().cmp(&b.2.raw()).then_with(|| a.1.cmp(&b.1)));

        let mut energies_q = vec![UQ0_16::ONE; input.candidates.len()];
        for (idx, _, energy) in &scored {
            energies_q[*idx] = *energy;
        }
        let best_indices = scored
            .iter()
            .take(EBM_TOP_N_MAX)
            .map(|(idx, _, _)| *idx as u16)
            .collect::<Vec<_>>();
        let aggregate = scored.first().map(|v| v.2).unwrap_or(UQ0_16::ONE);
        let digest = compute_ebm_digest(
            self.contract_version(),
            self.backend_id(),
            &input,
            &energies_q,
        );

        EbmOutput {
            status: EbmStatus::Ok,
            energies_q,
            best_indices,
            aggregate_energy_q: aggregate,
            ebm_digest: digest,
        }
    }
}

fn score_candidate(input: &EbmInput, candidate: &CandidateFeature) -> UQ0_16 {
    let mut acc = 0u32;
    acc = acc.saturating_add(mul_q(W_RISK_Q, input.signals.risk_q));
    acc = acc.saturating_add(mul_q(W_UNCERT_Q, input.signals.uncertainty_q));
    acc = acc.saturating_add(mul_q(W_PRESS_Q, input.signals.pressure_q));
    acc = acc.saturating_add(mul_q(W_SUR_Q, input.signals.surprise_q));

    match candidate.candidate_kind {
        CandidateKind::ToolIntent => {
            acc = acc.saturating_add(u32::from(TOOL_PENALTY_Q.raw()));
        }
        CandidateKind::Json => {
            acc = acc.saturating_add(u32::from(JSON_PENALTY_Q.raw()));
        }
        CandidateKind::NoOp => {
            acc = acc.saturating_sub(u32::from(NOOP_BONUS_Q.raw()));
        }
        CandidateKind::SafeText | CandidateKind::Other => {}
    }

    if input.emergency_active || input.governor_tier >= 3 {
        if matches!(candidate.candidate_kind, CandidateKind::ToolIntent) {
            acc = acc.saturating_add(u32::from(EMERGENCY_TOOL_PENALTY_Q.raw()));
        }
        if matches!(candidate.candidate_kind, CandidateKind::NoOp) {
            acc = acc.saturating_sub(u32::from(NOOP_BONUS_Q.raw()));
        }
    }

    UQ0_16::from_raw(acc.min(u32::from(u16::MAX)) as u16)
}

fn mul_q(a: UQ0_16, b: UQ0_16) -> u32 {
    ((u32::from(a.raw()) * u32::from(b.raw())) / u32::from(u16::MAX)).min(u32::from(u16::MAX))
}

fn compute_ebm_digest(
    contract_version: StageContractVersion,
    backend_id: BackendComponentId,
    input: &EbmInput,
    energies_q: &[UQ0_16],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(contract_version.as_u16().to_le_bytes());
    hasher.update([backend_id as u8]);
    hasher.update(input.t.to_le_bytes());
    hasher.update([input.governor_tier]);
    hasher.update([u8::from(input.emergency_active)]);
    hasher.update(input.context_digest);
    hasher.update(input.signals.risk_q.raw().to_le_bytes());
    hasher.update(input.signals.confidence_q.raw().to_le_bytes());
    hasher.update(input.signals.pressure_q.raw().to_le_bytes());
    hasher.update(input.signals.surprise_q.raw().to_le_bytes());
    hasher.update(input.signals.uncertainty_q.raw().to_le_bytes());
    hasher.update(
        input
            .signals
            .coherence_q
            .map(UQ0_16::raw)
            .unwrap_or(0)
            .to_le_bytes(),
    );
    for (candidate, energy) in input.candidates.iter().zip(energies_q.iter()) {
        hasher.update(candidate.candidate_id.to_le_bytes());
        hasher.update(energy.raw().to_le_bytes());
    }
    hasher.finalize().into()
}

fn degraded_fallback(
    input: &EbmInput,
    contract_version: StageContractVersion,
    backend_id: BackendComponentId,
    status: EbmStatus,
) -> EbmOutput {
    let mut energies_q = vec![UQ0_16::ONE; input.candidates.len()];
    if let Some((idx, _)) = input
        .candidates
        .iter()
        .enumerate()
        .find(|(_, c)| matches!(c.candidate_kind, CandidateKind::NoOp))
    {
        energies_q[idx] = UQ0_16::ZERO;
    } else if !energies_q.is_empty() {
        energies_q[0] = UQ0_16::ZERO;
    }
    let best_indices = if input.candidates.is_empty() {
        Vec::new()
    } else {
        let mut pairs: Vec<(usize, u16)> = input
            .candidates
            .iter()
            .enumerate()
            .map(|(idx, c)| (idx, c.candidate_id))
            .collect();
        pairs.sort_by(|a, b| {
            energies_q[a.0]
                .raw()
                .cmp(&energies_q[b.0].raw())
                .then_with(|| a.1.cmp(&b.1))
        });
        pairs
            .into_iter()
            .take(EBM_TOP_N_MAX)
            .map(|(idx, _)| idx as u16)
            .collect()
    };
    let digest = compute_ebm_digest(contract_version, backend_id, input, &energies_q);
    EbmOutput {
        status,
        aggregate_energy_q: best_indices
            .first()
            .map(|idx| energies_q[*idx as usize])
            .unwrap_or(UQ0_16::ONE),
        energies_q,
        best_indices,
        ebm_digest: digest,
    }
}

pub fn candidate_feature_from_decision(candidate: &DecisionCandidate) -> CandidateFeature {
    let candidate_kind = if candidate.is_noop() {
        CandidateKind::NoOp
    } else if !candidate.tool_intents.is_empty() {
        CandidateKind::ToolIntent
    } else {
        match candidate.output_class {
            OutputClass::SafeText => CandidateKind::SafeText,
            OutputClass::ExternalIo | OutputClass::Sensitive => CandidateKind::Json,
            OutputClass::Code | OutputClass::ExecIntent => CandidateKind::Other,
        }
    };
    let tool_class = candidate
        .tool_intents
        .first()
        .map(|tool| tool.kind.as_tag().bytes().next().unwrap_or(0));
    CandidateFeature {
        candidate_id: candidate.candidate_id,
        candidate_kind,
        tool_class,
        candidate_digest: candidate.digest,
        feature_vec_q: Vec::new(),
    }
}

pub fn fallback_best_index(output: &EbmOutput) -> Option<usize> {
    output.best_indices.first().map(|idx| usize::from(*idx))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_input() -> EbmInput {
        EbmInput {
            t: 5,
            governor_tier: 1,
            emergency_active: false,
            context_digest: [9; 32],
            signals: EbmSignals {
                risk_q: UQ0_16::from_raw(30_000),
                confidence_q: UQ0_16::from_raw(50_000),
                pressure_q: UQ0_16::from_raw(20_000),
                surprise_q: UQ0_16::from_raw(20_000),
                uncertainty_q: UQ0_16::from_raw(25_000),
                coherence_q: None,
            },
            candidates: vec![
                CandidateFeature {
                    candidate_id: 2,
                    candidate_kind: CandidateKind::ToolIntent,
                    tool_class: None,
                    candidate_digest: [2; 32],
                    feature_vec_q: Vec::new(),
                },
                CandidateFeature {
                    candidate_id: 1,
                    candidate_kind: CandidateKind::SafeText,
                    tool_class: None,
                    candidate_digest: [1; 32],
                    feature_vec_q: Vec::new(),
                },
            ],
        }
    }

    #[test]
    fn deterministic_for_same_input() {
        let mut ebm = CpuEbmStubV0;
        let mut budget_a = WorkMeter::new(100);
        let mut budget_b = WorkMeter::new(100);
        let a = ebm.score_candidates(mk_input(), &mut budget_a);
        let b = ebm.score_candidates(mk_input(), &mut budget_b);
        assert_eq!(a.energies_q, b.energies_q);
        assert_eq!(a.ebm_digest, b.ebm_digest);
    }

    #[test]
    fn tie_break_prefers_lower_candidate_id() {
        let mut input = mk_input();
        input.signals = EbmSignals {
            risk_q: UQ0_16::ZERO,
            confidence_q: UQ0_16::ZERO,
            pressure_q: UQ0_16::ZERO,
            surprise_q: UQ0_16::ZERO,
            uncertainty_q: UQ0_16::ZERO,
            coherence_q: None,
        };
        input.candidates = vec![
            CandidateFeature {
                candidate_id: 9,
                candidate_kind: CandidateKind::SafeText,
                tool_class: None,
                candidate_digest: [0; 32],
                feature_vec_q: Vec::new(),
            },
            CandidateFeature {
                candidate_id: 3,
                candidate_kind: CandidateKind::SafeText,
                tool_class: None,
                candidate_digest: [0; 32],
                feature_vec_q: Vec::new(),
            },
        ];
        let mut ebm = CpuEbmStubV0;
        let mut budget = WorkMeter::new(100);
        let out = ebm.score_candidates(input, &mut budget);
        assert_eq!(out.best_indices[0], 1);
    }

    #[test]
    fn emergency_penalizes_tool_intent() {
        let mut input = mk_input();
        input.emergency_active = true;
        let mut ebm = CpuEbmStubV0;
        let mut budget = WorkMeter::new(100);
        let out = ebm.score_candidates(input, &mut budget);
        let best = out.best_indices[0] as usize;
        assert!(!matches!(
            out.energies_q.get(best),
            Some(v) if *v == UQ0_16::ONE && best == 0
        ));
        assert_eq!(best, 1);
    }

    #[test]
    fn budget_exceeded_is_safe_and_deterministic() {
        let mut ebm = CpuEbmStubV0;
        let mut budget = WorkMeter::new(0);
        let out = ebm.score_candidates(mk_input(), &mut budget);
        assert_eq!(out.status, EbmStatus::BudgetExceeded);
        assert_eq!(out.best_indices.first().copied(), Some(0));
    }
}
