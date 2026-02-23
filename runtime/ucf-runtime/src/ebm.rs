use sha2::{Digest, Sha256};
use std::sync::OnceLock;
use ucf_compute::{BackendComponentId, StageContractVersion, WorkMeter};
use ucf_policy::candidate::{DecisionCandidate, OutputClass};
use ucf_types::UQ0_16;

pub const EBM_K_MAX: usize = 32;
pub const EBM_TOP_N_MAX: usize = 8;
pub const EBM_FEATURE_D_MAX: usize = 64;
pub const EBM_HIDDEN_MAX: usize = 32;
pub const EBM_SEARCH_STEPS_MAX: u8 = 16;
pub const EBM_CONSTRAINT_TOP_MAX: usize = 8;
pub const EBM_CONSTRAINT_TERM_MAX: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ConstraintTermId(pub u16);

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintTermKind {
    ToolIntentPenalty,
    CapabilityForbidden,
    CapabilityHighRisk,
    ContextRiskAmplifier,
    EmergencyDenyAllBias,
    OutputClassMismatch,
    BudgetExhaustedBias,
    NsrRiskAmplifier,
}

impl ConstraintTermKind {
    pub const fn as_label(self) -> &'static str {
        match self {
            Self::ToolIntentPenalty => "ToolIntentPenalty",
            Self::CapabilityForbidden => "CapabilityForbidden",
            Self::CapabilityHighRisk => "CapabilityHighRisk",
            Self::ContextRiskAmplifier => "ContextRiskAmplifier",
            Self::EmergencyDenyAllBias => "EmergencyDenyAllBias",
            Self::OutputClassMismatch => "OutputClassMismatch",
            Self::BudgetExhaustedBias => "BudgetExhaustedBias",
            Self::NsrRiskAmplifier => "NsrRiskAmplifier",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstraintParams {
    pub capability_class_id: Option<u8>,
    pub threshold_q: Option<UQ0_16>,
    pub candidate_kind: Option<CandidateKind>,
    pub governor_tier_min: Option<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstraintTermSpec {
    pub id: ConstraintTermId,
    pub kind: ConstraintTermKind,
    pub weight_q: UQ0_16,
    pub params: ConstraintParams,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TermContribution {
    pub id: u16,
    pub kind: ConstraintTermKind,
    pub contrib_q: UQ0_16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EbmConstraintLibrary {
    pub schema_version: u16,
    pub terms: Vec<ConstraintTermSpec>,
    pub fallback_used: bool,
    pub constraints_digest: [u8; 32],
}

impl EbmConstraintLibrary {
    pub fn fallback() -> Self {
        let terms = vec![ConstraintTermSpec {
            id: ConstraintTermId(1),
            kind: ConstraintTermKind::ToolIntentPenalty,
            weight_q: UQ0_16::from_raw(62_000),
            params: ConstraintParams {
                capability_class_id: None,
                threshold_q: None,
                candidate_kind: None,
                governor_tier_min: None,
            },
        }];
        Self {
            schema_version: 1,
            constraints_digest: digest_constraint_terms(1, &terms),
            terms,
            fallback_used: true,
        }
    }
}

static EBM_CONSTRAINT_LIBRARY: OnceLock<EbmConstraintLibrary> = OnceLock::new();

pub fn configure_ebm_constraints(library: EbmConstraintLibrary) {
    let _ = EBM_CONSTRAINT_LIBRARY.set(library);
}

pub fn active_ebm_constraints() -> &'static EbmConstraintLibrary {
    EBM_CONSTRAINT_LIBRARY.get_or_init(EbmConstraintLibrary::fallback)
}

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

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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
    pub nsr_risk_q: Option<UQ0_16>,
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
    pub base_energies_q: Vec<UQ0_16>,
    pub selected_term_contributions: Vec<TermContribution>,
    pub constraints_digest_prefix: [u8; 8],
    pub ebm_digest: [u8; 32],
    pub model_digest_prefix: [u8; 8],
    pub search_enabled: bool,
    pub search_steps_used: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EbmSignal {
    pub energy_min_q: UQ0_16,
    pub energy_mean_topk_q: UQ0_16,
    pub energy_dispersion_q: UQ0_16,
    pub ebm_digest_prefix: [u8; 8],
}

impl EbmSignal {
    pub fn from_output(output: &EbmOutput) -> Self {
        let mut top = output
            .best_indices
            .iter()
            .filter_map(|idx| output.energies_q.get(usize::from(*idx)).copied())
            .take(EBM_TOP_N_MAX)
            .collect::<Vec<_>>();
        if top.is_empty() {
            top.push(output.aggregate_energy_q);
        }
        top.sort_by_key(|v| v.raw());
        let min = top[0];
        let max = *top.last().unwrap_or(&min);
        let sum = top.iter().map(|v| u32::from(v.raw())).sum::<u32>();
        let mean_raw = (sum / top.len() as u32).min(u32::from(u16::MAX)) as u16;
        Self {
            energy_min_q: min,
            energy_mean_topk_q: UQ0_16::from_raw(mean_raw),
            energy_dispersion_q: UQ0_16::from_raw(max.raw().saturating_sub(min.raw())),
            ebm_digest_prefix: prefix8(output.ebm_digest),
        }
    }
}

fn prefix8(digest: [u8; 32]) -> [u8; 8] {
    let mut out = [0u8; 8];
    out.copy_from_slice(&digest[..8]);
    out
}

pub trait EbmReasoner: Send {
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
                [0; 8],
                false,
                0,
                EbmStatus::BudgetExceeded,
            );
        }

        let constraints = active_ebm_constraints();
        let mut scored: Vec<(usize, u16, UQ0_16, UQ0_16, Vec<TermContribution>)> = input
            .candidates
            .iter()
            .enumerate()
            .map(|(idx, c)| {
                let base = score_candidate(&input, c);
                let (total, contribs) = apply_constraint_terms(&input, c, base, constraints);
                (idx, c.candidate_id, total, base, contribs)
            })
            .collect();
        scored.sort_by(|a, b| a.2.raw().cmp(&b.2.raw()).then_with(|| a.1.cmp(&b.1)));

        let mut energies_q = vec![UQ0_16::ONE; input.candidates.len()];
        let mut base_energies_q = vec![UQ0_16::ONE; input.candidates.len()];
        for (idx, _, energy, base, _) in &scored {
            energies_q[*idx] = *energy;
            base_energies_q[*idx] = *base;
        }
        let best_indices = scored
            .iter()
            .take(EBM_TOP_N_MAX)
            .map(|(idx, _, _, _, _)| *idx as u16)
            .collect::<Vec<_>>();
        let aggregate = scored.first().map(|v| v.2).unwrap_or(UQ0_16::ONE);
        let digest = compute_ebm_digest(
            self.contract_version(),
            self.backend_id(),
            [0; 8],
            &input,
            &energies_q,
            false,
            0,
        );

        let selected_term_contributions = scored
            .first()
            .map(|(_, _, _, _, contribs)| contribs.clone())
            .unwrap_or_default();

        EbmOutput {
            status: EbmStatus::Ok,
            energies_q,
            best_indices,
            aggregate_energy_q: aggregate,
            base_energies_q,
            selected_term_contributions,
            constraints_digest_prefix: prefix8(constraints.constraints_digest),
            ebm_digest: digest,
            model_digest_prefix: [0; 8],
            search_enabled: false,
            search_steps_used: 0,
        }
    }
}

#[cfg(any(feature = "compute-candle", test))]
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct EbmMlModelV1 {
    input_dim: usize,
    hidden_dim: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: f32,
}

#[cfg(feature = "compute-candle")]
#[derive(Debug)]
pub struct CandleEbmReasonerV1 {
    model: Option<EbmMlModelV1>,
    model_digest_prefix: [u8; 8],
    search_enabled: bool,
}

#[cfg(feature = "compute-candle")]
impl CandleEbmReasonerV1 {
    pub fn from_model_store(search_enabled: bool) -> Self {
        use ucf_compute::candle_weights::{
            load_safetensors_raw, DType, DimExpr, TensorSpec, WeightSpec,
        };
        use ucf_compute::{ModelSlot, ModelStore};

        const EBM_REQ: &[TensorSpec] = &[
            TensorSpec {
                name: "ebm.w1",
                shape: &[DimExpr::Var("d"), DimExpr::Var("h")],
                dtype: DType::F32,
            },
            TensorSpec {
                name: "ebm.b1",
                shape: &[DimExpr::Var("h")],
                dtype: DType::F32,
            },
            TensorSpec {
                name: "ebm.w2",
                shape: &[DimExpr::Var("h"), DimExpr::Fixed(1)],
                dtype: DType::F32,
            },
            TensorSpec {
                name: "ebm.b2",
                shape: &[DimExpr::Fixed(1)],
                dtype: DType::F32,
            },
        ];
        let Ok(store) = ModelStore::from_env_default() else {
            return Self {
                model: None,
                model_digest_prefix: [0; 8],
                search_enabled,
            };
        };
        let Ok(verified) = store.verify_slot(ModelSlot::EbmReasoner) else {
            return Self {
                model: None,
                model_digest_prefix: [0; 8],
                search_enabled,
            };
        };
        let spec = WeightSpec {
            slot: ModelSlot::EbmReasoner,
            tensors: EBM_REQ,
            optional: &[],
            max_bytes: verified.size_bytes.max(1024),
            bindings: std::collections::BTreeMap::new(),
        };
        let Ok(bytes) = store.read_verified_bytes(&verified) else {
            return Self {
                model: None,
                model_digest_prefix: [0; 8],
                search_enabled,
            };
        };
        let Ok(loaded) = load_safetensors_raw(ModelSlot::EbmReasoner, &bytes, &spec) else {
            return Self {
                model: None,
                model_digest_prefix: [0; 8],
                search_enabled,
            };
        };

        let Some(w1) = loaded.tensors.get("ebm.w1") else {
            return Self {
                model: None,
                model_digest_prefix: [0; 8],
                search_enabled,
            };
        };
        let Some(b1) = loaded.tensors.get("ebm.b1") else {
            return Self {
                model: None,
                model_digest_prefix: [0; 8],
                search_enabled,
            };
        };
        let Some(w2) = loaded.tensors.get("ebm.w2") else {
            return Self {
                model: None,
                model_digest_prefix: [0; 8],
                search_enabled,
            };
        };
        let Some(b2) = loaded.tensors.get("ebm.b2") else {
            return Self {
                model: None,
                model_digest_prefix: [0; 8],
                search_enabled,
            };
        };
        if w1.shape.len() != 2 || b1.shape.len() != 1 || w2.shape.len() != 2 {
            return Self {
                model: None,
                model_digest_prefix: [0; 8],
                search_enabled,
            };
        }
        let d = w1.shape[0];
        let h = w1.shape[1];
        if d == 0 || h == 0 || d > EBM_FEATURE_D_MAX || h > EBM_HIDDEN_MAX {
            return Self {
                model: None,
                model_digest_prefix: [0; 8],
                search_enabled,
            };
        }
        if b1.shape[0] != h || w2.shape[0] != h || w2.shape[1] != 1 || b2.values_f32.len() != 1 {
            return Self {
                model: None,
                model_digest_prefix: [0; 8],
                search_enabled,
            };
        }
        let model = EbmMlModelV1 {
            input_dim: d,
            hidden_dim: h,
            w1: w1.values_f32.clone(),
            b1: b1.values_f32.clone(),
            w2: w2.values_f32.clone(),
            b2: b2.values_f32[0],
        };
        Self {
            model: Some(model),
            model_digest_prefix: prefix8(verified.sha256),
            search_enabled,
        }
    }

    #[cfg(test)]
    fn from_model_for_tests(model: EbmMlModelV1, search_enabled: bool) -> Self {
        Self {
            model: Some(model),
            model_digest_prefix: [0xAB; 8],
            search_enabled,
        }
    }
}

#[cfg(feature = "compute-candle")]
impl EbmReasoner for CandleEbmReasonerV1 {
    fn contract_version(&self) -> StageContractVersion {
        StageContractVersion::V1
    }

    fn backend_id(&self) -> BackendComponentId {
        BackendComponentId::CandleEbmV1
    }

    fn score_candidates(&mut self, mut input: EbmInput, budget: &mut WorkMeter) -> EbmOutput {
        input.candidates.truncate(EBM_K_MAX);
        let Some(model) = self.model.as_ref() else {
            return degraded_fallback(
                &input,
                self.contract_version(),
                self.backend_id(),
                self.model_digest_prefix,
                self.search_enabled,
                0,
                EbmStatus::DegradedFallback,
            );
        };
        if budget
            .spend(
                (input.candidates.len() as u64).saturating_mul(4),
                "ebm/candle_score",
            )
            .is_err()
        {
            return degraded_fallback(
                &input,
                self.contract_version(),
                self.backend_id(),
                self.model_digest_prefix,
                self.search_enabled,
                0,
                EbmStatus::BudgetExceeded,
            );
        }

        let mut energies_q = Vec::with_capacity(input.candidates.len());
        let mut steps_used = 0_u8;
        for candidate in &input.candidates {
            let (best_energy, used) = if self.search_enabled {
                score_with_bounded_search(model, &input.signals, candidate)
            } else {
                (score_mlp_candidate(model, &input.signals, candidate), 0)
            };
            steps_used = steps_used.saturating_add(used).min(EBM_SEARCH_STEPS_MAX);
            energies_q.push(best_energy);
        }

        if energies_q.iter().all(|e| *e == UQ0_16::ZERO)
            || energies_q.iter().all(|e| *e == UQ0_16::ONE)
        {
            return degraded_fallback(
                &input,
                self.contract_version(),
                self.backend_id(),
                self.model_digest_prefix,
                self.search_enabled,
                steps_used,
                EbmStatus::DegradedFallback,
            );
        }

        let mut scored: Vec<(usize, u16, UQ0_16)> = input
            .candidates
            .iter()
            .enumerate()
            .map(|(idx, c)| (idx, c.candidate_id, energies_q[idx]))
            .collect();
        scored.sort_by(|a, b| a.2.raw().cmp(&b.2.raw()).then_with(|| a.1.cmp(&b.1)));

        let best_indices = scored
            .iter()
            .take(EBM_TOP_N_MAX)
            .map(|(idx, _, _)| *idx as u16)
            .collect::<Vec<_>>();
        let aggregate = scored.first().map(|v| v.2).unwrap_or(UQ0_16::ONE);
        let digest = compute_ebm_digest(
            self.contract_version(),
            self.backend_id(),
            self.model_digest_prefix,
            &input,
            &energies_q,
            self.search_enabled,
            steps_used,
        );
        EbmOutput {
            status: EbmStatus::Ok,
            energies_q: energies_q.clone(),
            best_indices,
            aggregate_energy_q: aggregate,
            base_energies_q: energies_q,
            selected_term_contributions: Vec::new(),
            constraints_digest_prefix: [0; 8],
            ebm_digest: digest,
            model_digest_prefix: self.model_digest_prefix,
            search_enabled: self.search_enabled,
            search_steps_used: steps_used,
        }
    }
}

#[cfg(any(feature = "compute-candle", test))]
#[allow(dead_code)]
fn score_with_bounded_search(
    model: &EbmMlModelV1,
    signals: &EbmSignals,
    candidate: &CandidateFeature,
) -> (UQ0_16, u8) {
    let mut variants = bounded_variants(candidate);
    let mut best = UQ0_16::ONE;
    let mut used = 0_u8;
    for variant in variants.drain(..).take(4) {
        let e = score_mlp_candidate(model, signals, &variant);
        used = used.saturating_add(1).min(EBM_SEARCH_STEPS_MAX);
        if e.raw() < best.raw() {
            best = e;
        }
    }
    (best, used)
}

#[cfg(any(feature = "compute-candle", test))]
fn bounded_variants(candidate: &CandidateFeature) -> Vec<CandidateFeature> {
    let mut out = vec![candidate.clone()];
    if !matches!(candidate.candidate_kind, CandidateKind::NoOp) {
        let mut no_op = candidate.clone();
        no_op.candidate_kind = CandidateKind::NoOp;
        out.push(no_op);
    }
    if matches!(candidate.candidate_kind, CandidateKind::ToolIntent) {
        let mut lower = candidate.clone();
        lower.candidate_kind = CandidateKind::Json;
        out.push(lower);
    }
    let mut shorter = candidate.clone();
    if !shorter.feature_vec_q.is_empty() {
        shorter.feature_vec_q[0] = shorter.feature_vec_q[0].saturating_sub(2048);
    }
    out.push(shorter);
    out.sort_by(|a, b| {
        a.candidate_kind
            .cmp(&b.candidate_kind)
            .then_with(|| a.candidate_id.cmp(&b.candidate_id))
    });
    out.dedup_by(|a, b| {
        a.candidate_kind == b.candidate_kind
            && a.candidate_id == b.candidate_id
            && a.feature_vec_q == b.feature_vec_q
    });
    out
}

#[cfg(any(feature = "compute-candle", test))]
#[allow(dead_code)]
fn score_mlp_candidate(
    model: &EbmMlModelV1,
    signals: &EbmSignals,
    candidate: &CandidateFeature,
) -> UQ0_16 {
    let mut x = build_feature_vector(signals, candidate, model.input_dim);
    x.truncate(model.input_dim);
    while x.len() < model.input_dim {
        x.push(0.0);
    }

    let mut h = vec![0.0_f32; model.hidden_dim];
    for (j, hv) in h.iter_mut().enumerate() {
        let mut acc = model.b1[j];
        for (i, xv) in x.iter().enumerate() {
            acc += model.w1[i * model.hidden_dim + j] * *xv;
        }
        *hv = acc.tanh();
    }

    let mut e_raw = model.b2;
    for (j, hv) in h.iter().enumerate() {
        e_raw += model.w2[j] * *hv;
    }
    let e = sigmoid(e_raw);
    UQ0_16::from_f32_clamped(e)
}

#[cfg(any(feature = "compute-candle", test))]
#[allow(dead_code)]
fn build_feature_vector(
    signals: &EbmSignals,
    candidate: &CandidateFeature,
    dim: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(dim.min(EBM_FEATURE_D_MAX));
    out.push(q_to_f32(signals.risk_q));
    out.push(q_to_f32(signals.confidence_q));
    out.push(q_to_f32(signals.pressure_q));
    out.push(q_to_f32(signals.surprise_q));
    out.push(q_to_f32(signals.uncertainty_q));
    out.push(signals.coherence_q.map(q_to_f32).unwrap_or(0.0));
    out.push(match candidate.candidate_kind {
        CandidateKind::SafeText => 0.0,
        CandidateKind::Json => 0.25,
        CandidateKind::ToolIntent => 0.5,
        CandidateKind::NoOp => 0.75,
        CandidateKind::Other => 1.0,
    });
    out.push(f32::from(candidate.tool_class.unwrap_or(0)) / 255.0);
    for &q in candidate
        .feature_vec_q
        .iter()
        .take(EBM_FEATURE_D_MAX.saturating_sub(out.len()))
    {
        out.push((q as f32) / 32767.0);
    }
    out
}

#[cfg(any(feature = "compute-candle", test))]
#[allow(dead_code)]
fn sigmoid(v: f32) -> f32 {
    if v.is_nan() {
        1.0
    } else {
        1.0 / (1.0 + (-v).exp())
    }
}

#[cfg(any(feature = "compute-candle", test))]
#[allow(dead_code)]
fn q_to_f32(v: UQ0_16) -> f32 {
    f32::from(v.raw()) / f32::from(u16::MAX)
}

fn apply_constraint_terms(
    input: &EbmInput,
    candidate: &CandidateFeature,
    base: UQ0_16,
    constraints: &EbmConstraintLibrary,
) -> (UQ0_16, Vec<TermContribution>) {
    let mut total = u32::from(base.raw());
    let mut contributions = Vec::new();
    for term in &constraints.terms {
        let contrib = eval_constraint_term(term, input, candidate);
        if contrib.raw() > 0 {
            total = total.saturating_add(u32::from(contrib.raw()));
            contributions.push(TermContribution {
                id: term.id.0,
                kind: term.kind,
                contrib_q: contrib,
            });
            metrics::counter!("ucf_ebm_term_applied_total", "term_id" => term.id.0.to_string())
                .increment(1);
        }
    }
    contributions.sort_by(|a, b| {
        b.contrib_q
            .raw()
            .cmp(&a.contrib_q.raw())
            .then_with(|| a.id.cmp(&b.id))
    });
    contributions.truncate(EBM_CONSTRAINT_TOP_MAX);
    (
        UQ0_16::from_raw(total.min(u32::from(u16::MAX)) as u16),
        contributions,
    )
}

fn eval_constraint_term(
    term: &ConstraintTermSpec,
    input: &EbmInput,
    candidate: &CandidateFeature,
) -> UQ0_16 {
    match term.kind {
        ConstraintTermKind::ToolIntentPenalty => {
            if matches!(candidate.candidate_kind, CandidateKind::ToolIntent) {
                term.weight_q
            } else {
                UQ0_16::ZERO
            }
        }
        ConstraintTermKind::CapabilityForbidden => {
            if let (Some(expected), Some(actual)) =
                (term.params.capability_class_id, candidate.tool_class)
            {
                if expected == actual {
                    return UQ0_16::ONE;
                }
            }
            UQ0_16::ZERO
        }
        ConstraintTermKind::CapabilityHighRisk => {
            if let (Some(expected), Some(actual)) =
                (term.params.capability_class_id, candidate.tool_class)
            {
                if expected == actual && input.signals.risk_q.raw() >= 32768 {
                    return term.weight_q;
                }
            }
            UQ0_16::ZERO
        }
        ConstraintTermKind::ContextRiskAmplifier => {
            let threshold = term.params.threshold_q.unwrap_or(UQ0_16::from_raw(32_768));
            if input.signals.risk_q.raw() > threshold.raw() {
                let delta =
                    UQ0_16::from_raw(input.signals.risk_q.raw().saturating_sub(threshold.raw()));
                let raw = mul_q(term.weight_q, delta) as u16;
                UQ0_16::from_raw(raw)
            } else {
                UQ0_16::ZERO
            }
        }
        ConstraintTermKind::EmergencyDenyAllBias => {
            if input.emergency_active && !matches!(candidate.candidate_kind, CandidateKind::NoOp) {
                term.weight_q
            } else {
                UQ0_16::ZERO
            }
        }
        ConstraintTermKind::OutputClassMismatch => {
            if term
                .params
                .candidate_kind
                .is_some_and(|k| k != candidate.candidate_kind)
            {
                term.weight_q
            } else {
                UQ0_16::ZERO
            }
        }
        ConstraintTermKind::BudgetExhaustedBias => {
            if term
                .params
                .governor_tier_min
                .is_some_and(|min| input.governor_tier >= min)
                && !matches!(candidate.candidate_kind, CandidateKind::NoOp)
            {
                term.weight_q
            } else {
                UQ0_16::ZERO
            }
        }
        ConstraintTermKind::NsrRiskAmplifier => {
            let threshold = term.params.threshold_q.unwrap_or(UQ0_16::from_raw(32_768));
            let Some(nsr_q) = input.signals.nsr_risk_q else {
                return UQ0_16::ZERO;
            };
            if nsr_q.raw() > threshold.raw() {
                let delta = UQ0_16::from_raw(nsr_q.raw().saturating_sub(threshold.raw()));
                UQ0_16::from_raw(mul_q(term.weight_q, delta) as u16)
            } else {
                UQ0_16::ZERO
            }
        }
    }
}

fn digest_constraint_terms(schema_version: u16, terms: &[ConstraintTermSpec]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"ucf.ebm.constraints.v1");
    hasher.update(schema_version.to_le_bytes());
    for term in terms {
        hasher.update(term.id.0.to_le_bytes());
        hasher.update([term.kind as u8]);
        hasher.update(term.weight_q.raw().to_le_bytes());
        hasher.update([term.params.capability_class_id.unwrap_or(0)]);
        hasher.update(
            term.params
                .threshold_q
                .map(UQ0_16::raw)
                .unwrap_or(0)
                .to_le_bytes(),
        );
        hasher.update([term.params.candidate_kind.map(|v| v as u8).unwrap_or(0)]);
        hasher.update([term.params.governor_tier_min.unwrap_or(0)]);
    }
    hasher.finalize().into()
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
    model_digest_prefix: [u8; 8],
    input: &EbmInput,
    energies_q: &[UQ0_16],
    search_enabled: bool,
    search_steps_used: u8,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(contract_version.as_u16().to_le_bytes());
    hasher.update([backend_id as u8]);
    hasher.update(model_digest_prefix);
    hasher.update([u8::from(search_enabled)]);
    hasher.update([search_steps_used]);
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
    model_digest_prefix: [u8; 8],
    search_enabled: bool,
    search_steps_used: u8,
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
    let digest = compute_ebm_digest(
        contract_version,
        backend_id,
        model_digest_prefix,
        input,
        &energies_q,
        search_enabled,
        search_steps_used,
    );
    EbmOutput {
        status,
        aggregate_energy_q: best_indices
            .first()
            .map(|idx| energies_q[*idx as usize])
            .unwrap_or(UQ0_16::ONE),
        base_energies_q: energies_q.clone(),
        selected_term_contributions: Vec::new(),
        constraints_digest_prefix: [0; 8],
        energies_q,
        best_indices,
        ebm_digest: digest,
        model_digest_prefix,
        search_enabled,
        search_steps_used,
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
    let feature_vec_q = vec![
        (candidate.estimated_cost.compute_units.min(i16::MAX as u32)) as i16,
        (candidate.estimated_cost.bytes_out.min(i16::MAX as u32)) as i16,
        i16::from(candidate.estimated_cost.tool_calls),
    ];
    CandidateFeature {
        candidate_id: candidate.candidate_id,
        candidate_kind,
        tool_class,
        candidate_digest: candidate.digest,
        feature_vec_q,
    }
}

pub fn fallback_best_index(output: &EbmOutput) -> Option<usize> {
    output.best_indices.first().map(|idx| usize::from(*idx))
}

#[cfg(not(feature = "compute-candle"))]
pub type CandleEbmReasonerV1 = CpuEbmStubV0;

#[cfg(not(feature = "compute-candle"))]
impl CpuEbmStubV0 {
    pub fn from_model_store(_search_enabled: bool) -> Self {
        Self
    }
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
                nsr_risk_q: None,
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
            nsr_risk_q: None,
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
    fn budget_exceeded_is_safe_and_deterministic() {
        let mut ebm = CpuEbmStubV0;
        let mut budget = WorkMeter::new(0);
        let out = ebm.score_candidates(mk_input(), &mut budget);
        assert_eq!(out.status, EbmStatus::BudgetExceeded);
        assert_eq!(out.best_indices.first().copied(), Some(0));
    }

    #[test]
    fn variant_generation_is_deterministic() {
        let c = CandidateFeature {
            candidate_id: 3,
            candidate_kind: CandidateKind::ToolIntent,
            tool_class: Some(1),
            candidate_digest: [3; 32],
            feature_vec_q: vec![1024, 10],
        };
        assert_eq!(bounded_variants(&c), bounded_variants(&c));
    }

    #[cfg(feature = "compute-candle")]
    #[test]
    fn candle_forward_deterministic() {
        let model = EbmMlModelV1 {
            input_dim: 8,
            hidden_dim: 3,
            w1: vec![0.01; 24],
            b1: vec![0.0; 3],
            w2: vec![0.2, 0.1, 0.3],
            b2: 0.05,
        };
        let mut ebm = CandleEbmReasonerV1::from_model_for_tests(model, true);
        let mut a = WorkMeter::new(100);
        let mut b = WorkMeter::new(100);
        let out_a = ebm.score_candidates(mk_input(), &mut a);
        let out_b = ebm.score_candidates(mk_input(), &mut b);
        assert_eq!(out_a.energies_q, out_b.energies_q);
        assert_eq!(out_a.ebm_digest, out_b.ebm_digest);
    }
}
