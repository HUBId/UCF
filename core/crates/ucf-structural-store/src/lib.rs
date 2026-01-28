#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_ncde::NcdeParams;
use ucf_onn::OnnParams;
use ucf_spikebus::SpikeKind;
use ucf_ssm::SsmParams;
use ucf_types::Digest32;

const REPLAY_CAPS_DOMAIN: &[u8] = b"ucf.structural.replay_caps.v1";
const NSR_THRESHOLDS_DOMAIN: &[u8] = b"ucf.structural.nsr_thresholds.v1";
const ONN_KNOBS_DOMAIN: &[u8] = b"ucf.structural.onn_knobs.v1";
const SNN_KNOBS_DOMAIN: &[u8] = b"ucf.structural.snn_knobs.v1";
const STRUCTURAL_PARAMS_DOMAIN: &[u8] = b"ucf.structural.params.v1";
const STRUCTURAL_PROPOSAL_DOMAIN: &[u8] = b"ucf.structural.proposal.v1";
const STRUCTURAL_STORE_DOMAIN: &[u8] = b"ucf.structural.store.v1";
const STRUCTURAL_HISTORY_DOMAIN: &[u8] = b"ucf.structural.history.v1";
const STRUCTURAL_GATES_DOMAIN: &[u8] = b"ucf.structural.gates.v1";
const STRUCTURAL_EVAL_DOMAIN: &[u8] = b"ucf.structural.eval.v1";
const STRUCTURAL_STATS_DOMAIN: &[u8] = b"ucf.structural.cycle_stats.v1";
const PARAM_KEY_DOMAIN: &[u8] = b"ucf.structural.param_key.v1";
const RSA_LIMITS_DOMAIN: &[u8] = b"ucf.structural.rsa_limits.v1";

const REPLAY_MICRO_MIN: u16 = 1;
const REPLAY_MESO_MIN: u16 = 1;
const REPLAY_MACRO_MIN: u16 = 1;
const REPLAY_MICRO_MAX: u16 = 64;
const REPLAY_MESO_MAX: u16 = 32;
const REPLAY_MACRO_MAX: u16 = 16;

const NSR_WARN_MIN: u16 = 2000;
const NSR_DENY_MIN: u16 = 4000;
const NSR_WARN_STEP: u16 = 200;
const NSR_DENY_STEP: u16 = 200;

const ONN_BASE_STEP_MAX: u16 = 1024;
const ONN_COUPLING_MAX: u16 = 10_000;
const ONN_MAX_DELTA_MAX: u16 = 2048;
const ONN_LOCK_WINDOW_MIN: u16 = 4096;
const ONN_LOCK_WINDOW_STEP: u16 = 512;
const ONN_COUPLING_STEP: u16 = 200;

const SNN_VERIFY_MIN: u16 = 4;
const SNN_VERIFY_MAX: u16 = 128;
const SNN_VERIFY_STEP: u16 = 1;

const EVAL_TICKS: u8 = 4;

const RSA_PHI_MIN_DEFAULT: u16 = 2800;
const RSA_RISK_MAX_DEFAULT: u16 = 7000;

const PARAM_LABEL_NSR_WARN: &str = "nsr.warn";
const PARAM_LABEL_NSR_DENY: &str = "nsr.deny";
const PARAM_LABEL_ONN_BASE_STEP: &str = "onn.base_step";
const PARAM_LABEL_ONN_COUPLING: &str = "onn.coupling";
const PARAM_LABEL_ONN_MAX_DELTA: &str = "onn.max_delta";
const PARAM_LABEL_ONN_LOCK_WINDOW: &str = "onn.lock_window";
const PARAM_LABEL_SNN_VERIFY_LIMIT: &str = "snn.verify_limit";
const PARAM_LABEL_SNN_ATTENTION_SHIFT: &str = "snn.threshold.attention_shift";
const PARAM_LABEL_SNN_REPLAY_TRIGGER: &str = "snn.threshold.replay_trigger";
const PARAM_LABEL_REPLAY_MICRO: &str = "replay.micro_k";
const PARAM_LABEL_REPLAY_MESO: &str = "replay.meso_m";
const PARAM_LABEL_REPLAY_MACRO: &str = "replay.macro_n";
const PARAM_LABEL_SSM_SELECTIVITY: &str = "ssm.selectivity";
const PARAM_LABEL_NCDE_ATTRACTOR: &str = "ncde.attractor_strength";
const PARAM_LABEL_IIT_PHI_MIN_APPLY: &str = "iit.phi_min_apply";
const PARAM_LABEL_RSA_RISK_MAX_APPLY: &str = "rsa.risk_max_apply";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParamDeltaRef {
    pub key: u16,
    pub from: i32,
    pub to: i32,
}

pub fn param_key(label: &str) -> u16 {
    let mut hasher = Hasher::new();
    hasher.update(PARAM_KEY_DOMAIN);
    hasher.update(label.as_bytes());
    let digest = hasher.finalize();
    let bytes = digest.as_bytes();
    u16::from_be_bytes([bytes[0], bytes[1]])
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayCaps {
    pub micro_k: u16,
    pub meso_m: u16,
    pub macro_n: u16,
    pub commit: Digest32,
}

impl ReplayCaps {
    pub fn new(micro_k: u16, meso_m: u16, macro_n: u16) -> Self {
        let micro_k = micro_k.clamp(REPLAY_MICRO_MIN, REPLAY_MICRO_MAX);
        let meso_m = meso_m.clamp(REPLAY_MESO_MIN, REPLAY_MESO_MAX);
        let macro_n = macro_n.clamp(REPLAY_MACRO_MIN, REPLAY_MACRO_MAX);
        let commit = commit_replay_caps(micro_k, meso_m, macro_n);
        Self {
            micro_k,
            meso_m,
            macro_n,
            commit,
        }
    }
}

impl Default for ReplayCaps {
    fn default() -> Self {
        Self::new(8, 4, 2)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NsrThresholds {
    pub warn: u16,
    pub deny: u16,
    pub commit: Digest32,
}

impl NsrThresholds {
    pub fn new(warn: u16, deny: u16) -> Self {
        let warn = warn.max(NSR_WARN_MIN);
        let deny = deny.max(NSR_DENY_MIN).max(warn);
        let commit = commit_nsr_thresholds(warn, deny);
        Self { warn, deny, commit }
    }
}

impl Default for NsrThresholds {
    fn default() -> Self {
        Self::new(5000, 8500)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnnKnobs {
    pub base_step: u16,
    pub coupling: u16,
    pub max_delta: u16,
    pub lock_window: u16,
    pub commit: Digest32,
}

impl OnnKnobs {
    pub fn new(base_step: u16, coupling: u16, max_delta: u16, lock_window: u16) -> Self {
        let commit = commit_onn_knobs(base_step, coupling, max_delta, lock_window);
        Self {
            base_step,
            coupling,
            max_delta,
            lock_window,
            commit,
        }
    }
}

impl Default for OnnKnobs {
    fn default() -> Self {
        let params = OnnParams::default();
        Self::new(
            params.base_step,
            params.coupling,
            params.max_delta,
            params.lock_window,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnnKnobs {
    pub kind_thresholds: Vec<(SpikeKind, u16)>,
    pub verify_limit: u16,
    pub commit: Digest32,
}

impl SnnKnobs {
    pub fn new(kind_thresholds: Vec<(SpikeKind, u16)>, verify_limit: u16) -> Self {
        let verify_limit = verify_limit.clamp(SNN_VERIFY_MIN, SNN_VERIFY_MAX);
        let commit = commit_snn_knobs(&kind_thresholds, verify_limit);
        Self {
            kind_thresholds,
            verify_limit,
            commit,
        }
    }

    pub fn threshold_for(&self, kind: SpikeKind) -> u16 {
        self.kind_thresholds
            .iter()
            .find_map(|(k, value)| (*k == kind).then_some(*value))
            .unwrap_or(0)
    }
}

impl Default for SnnKnobs {
    fn default() -> Self {
        let thresholds = vec![
            (SpikeKind::Novelty, 0),
            (SpikeKind::Threat, 0),
            (SpikeKind::CausalLink, 0),
            (SpikeKind::ConsistencyAlert, 0),
            (SpikeKind::ReplayTrigger, 0),
            (SpikeKind::AttentionShift, 0),
            (SpikeKind::Thought, 0),
        ];
        Self::new(thresholds, 32)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RsaLimits {
    pub phi_min_apply: u16,
    pub risk_max_apply: u16,
    pub commit: Digest32,
}

impl RsaLimits {
    pub fn new(phi_min_apply: u16, risk_max_apply: u16) -> Self {
        let phi_min_apply = phi_min_apply.min(10_000);
        let risk_max_apply = risk_max_apply.min(10_000);
        let commit = commit_rsa_limits(phi_min_apply, risk_max_apply);
        Self {
            phi_min_apply,
            risk_max_apply,
            commit,
        }
    }
}

impl Default for RsaLimits {
    fn default() -> Self {
        Self::new(RSA_PHI_MIN_DEFAULT, RSA_RISK_MAX_DEFAULT)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuralParams {
    pub onn: OnnKnobs,
    pub snn: SnnKnobs,
    pub nsr: NsrThresholds,
    pub replay: ReplayCaps,
    pub ssm: SsmParams,
    pub ncde: NcdeParams,
    pub rsa: RsaLimits,
    pub commit: Digest32,
}

impl StructuralParams {
    pub fn new(
        onn: OnnKnobs,
        snn: SnnKnobs,
        nsr: NsrThresholds,
        replay: ReplayCaps,
        ssm: SsmParams,
        ncde: NcdeParams,
        rsa: RsaLimits,
    ) -> Self {
        let commit = commit_structural_params(&onn, &snn, &nsr, &replay, &ssm, &ncde, &rsa);
        Self {
            onn,
            snn,
            nsr,
            replay,
            ssm,
            ncde,
            rsa,
            commit,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasonCode {
    DriftHigh,
    SurpriseHigh,
    CoherenceLow,
    IntegrationLow,
    NsrWarn,
    ReplayInstability,
    Unknown(u16),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuralDeltaProposal {
    pub cycle_id: u64,
    pub base_commit: Digest32,
    pub proposed_commit: Digest32,
    pub reasons: Vec<ReasonCode>,
    pub evidence: Vec<Digest32>,
    pub commit: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuralStore {
    pub current: StructuralParams,
    pub history_root: Digest32,
    pub commit: Digest32,
}

pub fn default_params() -> StructuralParams {
    StructuralStore::default_params()
}

impl StructuralStore {
    pub fn default_params() -> StructuralParams {
        StructuralParams::new(
            OnnKnobs::default(),
            SnnKnobs::default(),
            NsrThresholds::default(),
            ReplayCaps::default(),
            SsmParams::default(),
            NcdeParams::default(),
            RsaLimits::default(),
        )
    }

    pub fn new(current: StructuralParams) -> Self {
        let history_root = commit_history_root(Digest32::new([0u8; 32]), current.commit);
        let commit = commit_structural_store(&current, history_root);
        Self {
            current,
            history_root,
            commit,
        }
    }

    pub fn current(&self) -> &StructuralParams {
        &self.current
    }

    pub fn apply_deltas(&self, deltas: &[ParamDeltaRef]) -> Option<StructuralParams> {
        apply_deltas_to_params(&self.current, deltas)
    }

    pub fn apply_params(&mut self, params: StructuralParams) {
        self.history_root = commit_history_root(self.history_root, params.commit);
        self.current = params;
        self.commit = commit_structural_store(&self.current, self.history_root);
    }

    pub fn propose(
        &self,
        cycle_id: u64,
        reasons: Vec<ReasonCode>,
        evidence: Vec<Digest32>,
        seed: Digest32,
    ) -> StructuralDeltaProposal {
        let mut ordered_reasons = reasons;
        ordered_reasons.sort_by_key(reason_rank);
        let proposed = apply_reasons(&self.current, &ordered_reasons);
        let commit = commit_structural_proposal(
            cycle_id,
            &self.current.commit,
            &proposed.commit,
            &ordered_reasons,
            &evidence,
            &seed,
        );
        StructuralDeltaProposal {
            cycle_id,
            base_commit: self.current.commit,
            proposed_commit: proposed.commit,
            reasons: ordered_reasons,
            evidence,
            commit,
        }
    }

    pub fn evaluate_and_commit(
        &mut self,
        proposal: &StructuralDeltaProposal,
        gates: &StructuralGates,
    ) -> StructuralCommitResult {
        if !gates.nsr_ok || !gates.policy_ok || !gates.consistency_ok {
            return StructuralCommitResult::Rejected {
                reason: "gate_blocked".to_string(),
                commit: proposal.commit,
            };
        }
        if proposal.base_commit != self.current.commit {
            return StructuralCommitResult::Rejected {
                reason: "base_commit_mismatch".to_string(),
                commit: proposal.commit,
            };
        }

        let proposed = apply_reasons(&self.current, &proposal.reasons);
        if proposed.commit != proposal.proposed_commit {
            return StructuralCommitResult::Rejected {
                reason: "proposal_commit_mismatch".to_string(),
                commit: proposal.commit,
            };
        }

        let eval_seed = eval_seed(proposal.commit, self.current.commit);
        if !evaluate_improvement(&self.current, &proposed, eval_seed) {
            return StructuralCommitResult::Rejected {
                reason: "evaluation_no_improvement".to_string(),
                commit: proposal.commit,
            };
        }

        self.history_root = commit_history_root(self.history_root, proposed.commit);
        self.current = proposed;
        self.commit = commit_structural_store(&self.current, self.history_root);

        StructuralCommitResult::Committed {
            new_commit: self.current.commit,
            commit: proposal.commit,
        }
    }
}

impl Default for StructuralStore {
    fn default() -> Self {
        Self::new(Self::default_params())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuralGates {
    pub nsr_ok: bool,
    pub policy_ok: bool,
    pub consistency_ok: bool,
    pub coherence_plv: u16,
    pub drift: u16,
    pub surprise: u16,
    pub commit: Digest32,
}

impl StructuralGates {
    pub fn new(
        nsr_ok: bool,
        policy_ok: bool,
        consistency_ok: bool,
        coherence_plv: u16,
        drift: u16,
        surprise: u16,
    ) -> Self {
        let commit = commit_structural_gates(
            nsr_ok,
            policy_ok,
            consistency_ok,
            coherence_plv,
            drift,
            surprise,
        );
        Self {
            nsr_ok,
            policy_ok,
            consistency_ok,
            coherence_plv,
            drift,
            surprise,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StructuralCommitResult {
    Rejected {
        reason: String,
        commit: Digest32,
    },
    Committed {
        new_commit: Digest32,
        commit: Digest32,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuralCycleStats {
    pub coherence_plv: u16,
    pub integration_phi: u16,
    pub drift: u16,
    pub surprise: u16,
    pub nsr_verdict: u8,
    pub policy_ok: bool,
    pub consistency_ok: bool,
    pub commit: Digest32,
}

impl StructuralCycleStats {
    pub fn new(
        coherence_plv: u16,
        integration_phi: u16,
        drift: u16,
        surprise: u16,
        nsr_verdict: u8,
        policy_ok: bool,
        consistency_ok: bool,
    ) -> Self {
        let commit = commit_cycle_stats(
            coherence_plv,
            integration_phi,
            drift,
            surprise,
            nsr_verdict,
            policy_ok,
            consistency_ok,
        );
        Self {
            coherence_plv,
            integration_phi,
            drift,
            surprise,
            nsr_verdict,
            policy_ok,
            consistency_ok,
            commit,
        }
    }
}

fn commit_replay_caps(micro_k: u16, meso_m: u16, macro_n: u16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(REPLAY_CAPS_DOMAIN);
    hasher.update(&micro_k.to_be_bytes());
    hasher.update(&meso_m.to_be_bytes());
    hasher.update(&macro_n.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_nsr_thresholds(warn: u16, deny: u16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(NSR_THRESHOLDS_DOMAIN);
    hasher.update(&warn.to_be_bytes());
    hasher.update(&deny.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_onn_knobs(base_step: u16, coupling: u16, max_delta: u16, lock_window: u16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(ONN_KNOBS_DOMAIN);
    hasher.update(&base_step.to_be_bytes());
    hasher.update(&coupling.to_be_bytes());
    hasher.update(&max_delta.to_be_bytes());
    hasher.update(&lock_window.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_snn_knobs(kind_thresholds: &[(SpikeKind, u16)], verify_limit: u16) -> Digest32 {
    let mut thresholds = kind_thresholds.to_vec();
    thresholds.sort_by_key(|(kind, value)| (kind.as_u16(), *value));
    let mut hasher = Hasher::new();
    hasher.update(SNN_KNOBS_DOMAIN);
    hasher.update(&verify_limit.to_be_bytes());
    hasher.update(
        &u32::try_from(thresholds.len())
            .unwrap_or(u32::MAX)
            .to_be_bytes(),
    );
    for (kind, value) in thresholds {
        hasher.update(&kind.as_u16().to_be_bytes());
        hasher.update(&value.to_be_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_structural_params(
    onn: &OnnKnobs,
    snn: &SnnKnobs,
    nsr: &NsrThresholds,
    replay: &ReplayCaps,
    ssm: &SsmParams,
    ncde: &NcdeParams,
    rsa: &RsaLimits,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(STRUCTURAL_PARAMS_DOMAIN);
    hasher.update(onn.commit.as_bytes());
    hasher.update(snn.commit.as_bytes());
    hasher.update(nsr.commit.as_bytes());
    hasher.update(replay.commit.as_bytes());
    hasher.update(ssm.commit.as_bytes());
    hasher.update(ncde.commit.as_bytes());
    hasher.update(rsa.commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_structural_proposal(
    cycle_id: u64,
    base_commit: &Digest32,
    proposed_commit: &Digest32,
    reasons: &[ReasonCode],
    evidence: &[Digest32],
    seed: &Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(STRUCTURAL_PROPOSAL_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(base_commit.as_bytes());
    hasher.update(proposed_commit.as_bytes());
    hasher.update(seed.as_bytes());
    hasher.update(
        &u32::try_from(reasons.len())
            .unwrap_or(u32::MAX)
            .to_be_bytes(),
    );
    for reason in reasons {
        hasher.update(&reason_to_u16(*reason).to_be_bytes());
    }
    hasher.update(
        &u32::try_from(evidence.len())
            .unwrap_or(u32::MAX)
            .to_be_bytes(),
    );
    for digest in evidence {
        hasher.update(digest.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_structural_store(current: &StructuralParams, history_root: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(STRUCTURAL_STORE_DOMAIN);
    hasher.update(current.commit.as_bytes());
    hasher.update(history_root.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_history_root(root: Digest32, next: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(STRUCTURAL_HISTORY_DOMAIN);
    hasher.update(root.as_bytes());
    hasher.update(next.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_structural_gates(
    nsr_ok: bool,
    policy_ok: bool,
    consistency_ok: bool,
    coherence_plv: u16,
    drift: u16,
    surprise: u16,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(STRUCTURAL_GATES_DOMAIN);
    hasher.update(&[nsr_ok as u8, policy_ok as u8, consistency_ok as u8]);
    hasher.update(&coherence_plv.to_be_bytes());
    hasher.update(&drift.to_be_bytes());
    hasher.update(&surprise.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_cycle_stats(
    coherence_plv: u16,
    integration_phi: u16,
    drift: u16,
    surprise: u16,
    nsr_verdict: u8,
    policy_ok: bool,
    consistency_ok: bool,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(STRUCTURAL_STATS_DOMAIN);
    hasher.update(&coherence_plv.to_be_bytes());
    hasher.update(&integration_phi.to_be_bytes());
    hasher.update(&drift.to_be_bytes());
    hasher.update(&surprise.to_be_bytes());
    hasher.update(&[nsr_verdict, policy_ok as u8, consistency_ok as u8]);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_rsa_limits(phi_min_apply: u16, risk_max_apply: u16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(RSA_LIMITS_DOMAIN);
    hasher.update(&phi_min_apply.to_be_bytes());
    hasher.update(&risk_max_apply.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn reason_rank(reason: &ReasonCode) -> u8 {
    match reason {
        ReasonCode::CoherenceLow => 0,
        ReasonCode::IntegrationLow => 1,
        ReasonCode::DriftHigh => 2,
        ReasonCode::SurpriseHigh => 3,
        ReasonCode::NsrWarn => 4,
        ReasonCode::ReplayInstability => 5,
        ReasonCode::Unknown(_) => 6,
    }
}

fn reason_to_u16(reason: ReasonCode) -> u16 {
    match reason {
        ReasonCode::DriftHigh => 1,
        ReasonCode::SurpriseHigh => 2,
        ReasonCode::CoherenceLow => 3,
        ReasonCode::IntegrationLow => 4,
        ReasonCode::NsrWarn => 5,
        ReasonCode::ReplayInstability => 6,
        ReasonCode::Unknown(value) => 0x8000 | value,
    }
}

fn apply_reasons(base: &StructuralParams, reasons: &[ReasonCode]) -> StructuralParams {
    let mut onn = base.onn.clone();
    let mut snn = base.snn.clone();
    let mut nsr = base.nsr.clone();
    let mut replay = base.replay.clone();
    let ssm = base.ssm;
    let ncde = base.ncde;
    let rsa = base.rsa.clone();

    for reason in reasons {
        match reason {
            ReasonCode::CoherenceLow => {
                if onn.coupling < ONN_COUPLING_MAX {
                    onn.coupling = (onn.coupling + ONN_COUPLING_STEP).min(ONN_COUPLING_MAX);
                } else {
                    onn.lock_window = onn
                        .lock_window
                        .saturating_sub(ONN_LOCK_WINDOW_STEP)
                        .max(ONN_LOCK_WINDOW_MIN);
                }
            }
            ReasonCode::IntegrationLow => {
                replay.meso_m = (replay.meso_m + 1).min(REPLAY_MESO_MAX);
                replay.macro_n = (replay.macro_n + 1).min(REPLAY_MACRO_MAX);
            }
            ReasonCode::DriftHigh | ReasonCode::ReplayInstability => {
                replay.macro_n = replay.macro_n.saturating_sub(1).max(REPLAY_MACRO_MIN);
                replay.meso_m = replay.meso_m.saturating_sub(1).max(REPLAY_MESO_MIN);
                snn.verify_limit = snn
                    .verify_limit
                    .saturating_sub(SNN_VERIFY_STEP)
                    .max(SNN_VERIFY_MIN);
            }
            ReasonCode::NsrWarn => {
                nsr.warn = nsr.warn.saturating_sub(NSR_WARN_STEP).max(NSR_WARN_MIN);
                nsr.deny = nsr
                    .deny
                    .saturating_sub(NSR_DENY_STEP)
                    .max(NSR_DENY_MIN)
                    .max(nsr.warn);
            }
            ReasonCode::SurpriseHigh => {
                onn.lock_window = onn
                    .lock_window
                    .saturating_sub(ONN_LOCK_WINDOW_STEP)
                    .max(ONN_LOCK_WINDOW_MIN);
            }
            ReasonCode::Unknown(_) => {}
        }
    }

    onn.commit = commit_onn_knobs(onn.base_step, onn.coupling, onn.max_delta, onn.lock_window);
    snn.commit = commit_snn_knobs(&snn.kind_thresholds, snn.verify_limit);
    nsr.commit = commit_nsr_thresholds(nsr.warn, nsr.deny);
    replay.commit = commit_replay_caps(replay.micro_k, replay.meso_m, replay.macro_n);

    StructuralParams::new(onn, snn, nsr, replay, ssm, ncde, rsa)
}

fn apply_deltas_to_params(
    params: &StructuralParams,
    deltas: &[ParamDeltaRef],
) -> Option<StructuralParams> {
    let mut onn = params.onn.clone();
    let mut snn = params.snn.clone();
    let mut nsr = params.nsr.clone();
    let mut replay = params.replay.clone();
    let mut ssm = params.ssm;
    let mut ncde = params.ncde;
    let mut rsa = params.rsa.clone();

    for delta in deltas {
        if delta.key == param_key(PARAM_LABEL_NSR_WARN) {
            let current = i32::from(nsr.warn);
            if delta.from != current {
                return None;
            }
            let next = clamp_i32(delta.to, 0, i32::from(u16::MAX));
            nsr.warn = next as u16;
            continue;
        }
        if delta.key == param_key(PARAM_LABEL_NSR_DENY) {
            let current = i32::from(nsr.deny);
            if delta.from != current {
                return None;
            }
            let next = clamp_i32(delta.to, 0, i32::from(u16::MAX));
            nsr.deny = next as u16;
            continue;
        }
        if delta.key == param_key(PARAM_LABEL_ONN_BASE_STEP) {
            let current = i32::from(onn.base_step);
            if delta.from != current {
                return None;
            }
            let next = clamp_i32(delta.to, 1, i32::from(ONN_BASE_STEP_MAX));
            onn.base_step = next as u16;
            continue;
        }
        if delta.key == param_key(PARAM_LABEL_ONN_COUPLING) {
            let current = i32::from(onn.coupling);
            if delta.from != current {
                return None;
            }
            let next = clamp_i32(delta.to, 0, i32::from(ONN_COUPLING_MAX));
            onn.coupling = next as u16;
            continue;
        }
        if delta.key == param_key(PARAM_LABEL_ONN_MAX_DELTA) {
            let current = i32::from(onn.max_delta);
            if delta.from != current {
                return None;
            }
            let next = clamp_i32(delta.to, 0, i32::from(ONN_MAX_DELTA_MAX));
            onn.max_delta = next as u16;
            continue;
        }
        if delta.key == param_key(PARAM_LABEL_ONN_LOCK_WINDOW) {
            let current = i32::from(onn.lock_window);
            if delta.from != current {
                return None;
            }
            let next = clamp_i32(
                delta.to,
                i32::from(ONN_LOCK_WINDOW_MIN),
                i32::from(u16::MAX),
            );
            onn.lock_window = next as u16;
            continue;
        }
        if delta.key == param_key(PARAM_LABEL_SNN_VERIFY_LIMIT) {
            let current = i32::from(snn.verify_limit);
            if delta.from != current {
                return None;
            }
            let next = clamp_i32(delta.to, 0, i32::from(u16::MAX));
            snn.verify_limit = next as u16;
            continue;
        }
        if delta.key == param_key(PARAM_LABEL_SNN_ATTENTION_SHIFT) {
            let current = i32::from(snn.threshold_for(SpikeKind::AttentionShift));
            if delta.from != current {
                return None;
            }
            let next = clamp_i32(delta.to, 0, 10_000);
            set_snn_threshold(&mut snn, SpikeKind::AttentionShift, next as u16);
            continue;
        }
        if delta.key == param_key(PARAM_LABEL_SNN_REPLAY_TRIGGER) {
            let current = i32::from(snn.threshold_for(SpikeKind::ReplayTrigger));
            if delta.from != current {
                return None;
            }
            let next = clamp_i32(delta.to, 0, 10_000);
            set_snn_threshold(&mut snn, SpikeKind::ReplayTrigger, next as u16);
            continue;
        }
        if delta.key == param_key(PARAM_LABEL_REPLAY_MICRO) {
            let current = i32::from(replay.micro_k);
            if delta.from != current {
                return None;
            }
            let next = clamp_i32(delta.to, 0, i32::from(u16::MAX));
            replay.micro_k = next as u16;
            continue;
        }
        if delta.key == param_key(PARAM_LABEL_REPLAY_MESO) {
            let current = i32::from(replay.meso_m);
            if delta.from != current {
                return None;
            }
            let next = clamp_i32(delta.to, 0, i32::from(u16::MAX));
            replay.meso_m = next as u16;
            continue;
        }
        if delta.key == param_key(PARAM_LABEL_REPLAY_MACRO) {
            let current = i32::from(replay.macro_n);
            if delta.from != current {
                return None;
            }
            let next = clamp_i32(delta.to, 0, i32::from(u16::MAX));
            replay.macro_n = next as u16;
            continue;
        }
        if delta.key == param_key(PARAM_LABEL_SSM_SELECTIVITY) {
            let current = i32::from(ssm.selectivity);
            if delta.from != current {
                return None;
            }
            let next = clamp_i32(delta.to, 0, 10_000);
            ssm.selectivity = next as u16;
            continue;
        }
        if delta.key == param_key(PARAM_LABEL_NCDE_ATTRACTOR) {
            let current = i32::from(ncde.attractor_strength);
            if delta.from != current {
                return None;
            }
            let next = clamp_i32(delta.to, 0, 10_000);
            ncde.attractor_strength = next as u16;
            continue;
        }
        if delta.key == param_key(PARAM_LABEL_IIT_PHI_MIN_APPLY) {
            let current = i32::from(rsa.phi_min_apply);
            if delta.from != current {
                return None;
            }
            let next = clamp_i32(delta.to, 0, 10_000);
            rsa.phi_min_apply = next as u16;
            continue;
        }
        if delta.key == param_key(PARAM_LABEL_RSA_RISK_MAX_APPLY) {
            let current = i32::from(rsa.risk_max_apply);
            if delta.from != current {
                return None;
            }
            let next = clamp_i32(delta.to, 0, 10_000);
            rsa.risk_max_apply = next as u16;
            continue;
        }
        return None;
    }

    let onn = OnnKnobs::new(onn.base_step, onn.coupling, onn.max_delta, onn.lock_window);
    let snn = SnnKnobs::new(snn.kind_thresholds, snn.verify_limit);
    let nsr = NsrThresholds::new(nsr.warn, nsr.deny);
    let replay = ReplayCaps::new(replay.micro_k, replay.meso_m, replay.macro_n);
    let ssm = SsmParams::new(
        ssm.dim_x,
        ssm.dim_u,
        ssm.scan_blocks,
        ssm.dt_q,
        ssm.leak,
        ssm.selectivity,
    );
    let ncde = NcdeParams::new(
        ncde.dim_h,
        ncde.dim_u,
        ncde.dt_q,
        ncde.steps,
        ncde.attractor_strength,
    );
    let rsa = RsaLimits::new(rsa.phi_min_apply, rsa.risk_max_apply);

    Some(StructuralParams::new(onn, snn, nsr, replay, ssm, ncde, rsa))
}

fn clamp_i32(value: i32, min: i32, max: i32) -> i32 {
    value.clamp(min, max)
}

fn set_snn_threshold(snn: &mut SnnKnobs, kind: SpikeKind, value: u16) {
    if let Some(entry) = snn
        .kind_thresholds
        .iter_mut()
        .find(|(entry_kind, _)| *entry_kind == kind)
    {
        entry.1 = value;
    } else {
        snn.kind_thresholds.push((kind, value));
    }
}

fn eval_seed(proposal_commit: Digest32, current_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(STRUCTURAL_EVAL_DOMAIN);
    hasher.update(proposal_commit.as_bytes());
    hasher.update(current_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn evaluate_improvement(
    current: &StructuralParams,
    proposed: &StructuralParams,
    seed: Digest32,
) -> bool {
    let before = simulate_metrics(current, seed, EVAL_TICKS);
    let after = simulate_metrics(proposed, seed, EVAL_TICKS);

    let coherence_improved = after.coherence > before.coherence;
    let drift_improved = after.drift < before.drift;

    (coherence_improved && after.drift <= before.drift)
        || (drift_improved && after.coherence >= before.coherence)
}

#[derive(Clone, Copy)]
struct SimulatedMetrics {
    coherence: u16,
    drift: u16,
}

fn simulate_metrics(params: &StructuralParams, seed: Digest32, ticks: u8) -> SimulatedMetrics {
    let mut coherence_total: i32 = 0;
    let mut drift_total: i32 = 0;
    let count = ticks.max(1) as i32;

    for idx in 0..ticks.max(1) {
        let jitter = jitter_value(seed, idx);
        let coherence = coherence_proxy(params).saturating_add(jitter / 2);
        let drift = drift_proxy(params).saturating_add(jitter / 3);
        coherence_total += i32::from(coherence);
        drift_total += i32::from(drift);
    }

    let coherence_avg = (coherence_total / count).clamp(0, i32::from(u16::MAX)) as u16;
    let drift_avg = (drift_total / count).clamp(0, i32::from(u16::MAX)) as u16;

    SimulatedMetrics {
        coherence: coherence_avg,
        drift: drift_avg,
    }
}

fn coherence_proxy(params: &StructuralParams) -> u16 {
    let coupling_boost = i32::from(params.onn.coupling) / 2;
    let base_step_boost = i32::from(params.onn.base_step) / 2;
    let max_delta_boost = i32::from(params.onn.max_delta) / 4;
    let window_penalty = i32::from(params.onn.lock_window) / 64;
    let verify_bonus = i32::from(params.snn.verify_limit) * 2;
    let score = coupling_boost + base_step_boost + max_delta_boost + verify_bonus - window_penalty;
    score.clamp(0, 10_000) as u16
}

fn drift_proxy(params: &StructuralParams) -> u16 {
    let replay_pressure = i32::from(params.replay.micro_k)
        + i32::from(params.replay.meso_m) * 2
        + i32::from(params.replay.macro_n) * 4;
    let nsr_tightness = 10_000i32.saturating_sub(i32::from(params.nsr.warn));
    let verify_pressure = i32::from(params.snn.verify_limit) / 2;
    let score = replay_pressure + verify_pressure - (nsr_tightness / 20);
    score.clamp(0, 10_000) as u16
}

fn jitter_value(seed: Digest32, idx: u8) -> u16 {
    let mut hasher = Hasher::new();
    hasher.update(STRUCTURAL_EVAL_DOMAIN);
    hasher.update(seed.as_bytes());
    hasher.update(&[idx]);
    let bytes = hasher.finalize();
    let raw = bytes.as_bytes();
    u16::from_be_bytes([raw[0], raw[1]]) % 200
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proposal_is_deterministic_for_same_seed() {
        let store = StructuralStore::default();
        let seed = Digest32::new([9u8; 32]);
        let reasons = vec![ReasonCode::CoherenceLow, ReasonCode::NsrWarn];
        let evidence = vec![Digest32::new([1u8; 32])];

        let first = store.propose(7, reasons.clone(), evidence.clone(), seed);
        let second = store.propose(7, reasons, evidence, seed);

        assert_eq!(first, second);
    }

    #[test]
    fn commit_blocked_when_gate_is_false() {
        let mut store = StructuralStore::default();
        let proposal = store.propose(
            1,
            vec![ReasonCode::CoherenceLow],
            Vec::new(),
            Digest32::new([3u8; 32]),
        );
        let gates = StructuralGates::new(false, true, true, 5000, 2000, 1500);

        let result = store.evaluate_and_commit(&proposal, &gates);

        match result {
            StructuralCommitResult::Rejected { reason, .. } => {
                assert_eq!(reason, "gate_blocked");
            }
            _ => panic!("expected rejection"),
        }
        assert_eq!(
            store.current.commit,
            StructuralStore::default().current.commit
        );
    }

    #[test]
    fn commit_occurs_when_evaluation_improves() {
        let mut store = StructuralStore::default();
        let proposal = store.propose(
            2,
            vec![ReasonCode::CoherenceLow],
            Vec::new(),
            Digest32::new([4u8; 32]),
        );
        let gates = StructuralGates::new(true, true, true, 6000, 2000, 1200);

        let result = store.evaluate_and_commit(&proposal, &gates);

        match result {
            StructuralCommitResult::Committed { new_commit, .. } => {
                assert_eq!(new_commit, store.current.commit);
            }
            _ => panic!("expected commit"),
        }
    }

    #[test]
    fn apply_deltas_clamps_and_recomputes_commit() {
        let store = StructuralStore::default();
        let delta = ParamDeltaRef {
            key: param_key("nsr.warn"),
            from: i32::from(store.current.nsr.warn),
            to: 10,
        };

        let updated = store.apply_deltas(&[delta]).expect("apply deltas");

        assert_ne!(updated.commit, store.current.commit);
        assert!(updated.nsr.warn >= 2000);
    }
}
