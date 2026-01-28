#![forbid(unsafe_code)]

use std::sync::Arc;

use blake3::Hasher;
use ucf_archive::ExperienceAppender;
use ucf_archive_store::{ArchiveAppender, ArchiveStore, RecordKind, RecordMeta};
use ucf_bus::BusPublisher;
use ucf_openevolve_port::{EvolutionProposal, OpenEvolvePort, SleepContext, SleepReport};
use ucf_policy_ecology::SleepPhaseGate;
use ucf_structural_store::{
    param_key, ReasonCode, StructuralCommitResult, StructuralCycleStats, StructuralDeltaProposal,
    StructuralGates, StructuralParams, StructuralStore,
};
use ucf_types::v1::spec::ExperienceRecord;
use ucf_types::{Digest32, EvidenceId};

const STRUCTURAL_SEED_DOMAIN: &[u8] = b"ucf.rsa.structural.seed.v1";
const NSR_DENY_VERDICT: u8 = 2;
const RSA_PARAM_DELTA_DOMAIN: &[u8] = b"ucf.rsa.param_delta.v1";
const RSA_PROPOSAL_DOMAIN: &[u8] = b"ucf.rsa.structural_proposal.v1";
const RSA_INPUT_DOMAIN: &[u8] = b"ucf.rsa.inputs.v1";
const RSA_OUTPUT_DOMAIN: &[u8] = b"ucf.rsa.outputs.v1";
const RSA_CORE_DOMAIN: &[u8] = b"ucf.rsa.core.v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SleepReportReady {
    pub evidence_id: EvidenceId,
    pub cycle_id: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProposalKind {
    TuneThresholds,
    TuneCoupling,
    TuneReplay,
    TuneAttention,
    Unknown(u16),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamDelta {
    pub key: u16,
    pub from: i32,
    pub to: i32,
    pub commit: Digest32,
}

impl ParamDelta {
    pub fn new(key: u16, from: i32, to: i32) -> Self {
        let commit = commit_param_delta(key, from, to);
        Self {
            key,
            from,
            to,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuralProposal {
    pub cycle_id: u64,
    pub kind: ProposalKind,
    pub basis_commit: Digest32,
    pub deltas: Vec<ParamDelta>,
    pub expected_gain: i16,
    pub risk_cost: i16,
    pub commit: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RsaInputs {
    pub cycle_id: u64,
    pub sleep_active: bool,
    pub phase_commit: Digest32,
    pub coherence_plv: u16,
    pub phi_proxy: u16,
    pub nsr_verdict: u8,
    pub nsr_trace_root: Digest32,
    pub policy_ok: bool,
    pub risk: u16,
    pub drift: u16,
    pub surprise: u16,
    pub replay_pressure: u16,
    pub ssm_salience: u16,
    pub ncde_energy: u16,
    pub current_params_commit: Digest32,
    pub commit: Digest32,
}

impl RsaInputs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cycle_id: u64,
        sleep_active: bool,
        phase_commit: Digest32,
        coherence_plv: u16,
        phi_proxy: u16,
        nsr_verdict: u8,
        nsr_trace_root: Digest32,
        policy_ok: bool,
        risk: u16,
        drift: u16,
        surprise: u16,
        replay_pressure: u16,
        ssm_salience: u16,
        ncde_energy: u16,
        current_params_commit: Digest32,
    ) -> Self {
        let commit = commit_rsa_inputs(
            cycle_id,
            sleep_active,
            phase_commit,
            coherence_plv,
            phi_proxy,
            nsr_verdict,
            nsr_trace_root,
            policy_ok,
            risk,
            drift,
            surprise,
            replay_pressure,
            ssm_salience,
            ncde_energy,
            current_params_commit,
        );
        Self {
            cycle_id,
            sleep_active,
            phase_commit,
            coherence_plv,
            phi_proxy,
            nsr_verdict,
            nsr_trace_root,
            policy_ok,
            risk,
            drift,
            surprise,
            replay_pressure,
            ssm_salience,
            ncde_energy,
            current_params_commit,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RsaOutputs {
    pub cycle_id: u64,
    pub proposals: Vec<StructuralProposal>,
    pub chosen: Option<Digest32>,
    pub applied: bool,
    pub new_params_commit: Option<Digest32>,
    pub commit: Digest32,
}

impl RsaOutputs {
    pub fn empty(cycle_id: u64) -> Self {
        let commit = commit_rsa_outputs(cycle_id, &[], None, false, None);
        Self {
            cycle_id,
            proposals: Vec::new(),
            chosen: None,
            applied: false,
            new_params_commit: None,
            commit,
        }
    }

    pub fn recompute_commit(&mut self) {
        self.commit = commit_rsa_outputs(
            self.cycle_id,
            &self.proposals,
            self.chosen,
            self.applied,
            self.new_params_commit,
        );
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RsaCore {
    pub last_apply_cycle: u64,
    pub min_apply_gap: u16,
    pub commit: Digest32,
}

impl Default for RsaCore {
    fn default() -> Self {
        let last_apply_cycle = 0;
        let min_apply_gap = 4;
        let commit = commit_rsa_core(last_apply_cycle, min_apply_gap);
        Self {
            last_apply_cycle,
            min_apply_gap,
            commit,
        }
    }
}

impl RsaCore {
    pub fn tick(&mut self, inp: &RsaInputs) -> RsaOutputs {
        let params = ucf_structural_store::default_params();
        self.tick_with_params(inp, &params)
    }

    pub fn tick_with_params(&mut self, inp: &RsaInputs, params: &StructuralParams) -> RsaOutputs {
        if !inp.sleep_active {
            return RsaOutputs::empty(inp.cycle_id);
        }
        if !rsa_gates_allow(inp, params) {
            return RsaOutputs::empty(inp.cycle_id);
        }

        let mut proposals = Vec::new();
        let nsr_warn_streak = inp.nsr_verdict >> 2;
        let verdict = nsr_verdict(inp.nsr_verdict);
        if verdict == 0 && nsr_warn_streak >= 2 {
            if let Some(proposal) = tune_thresholds(inp, params) {
                proposals.push(proposal);
            }
        }
        if inp.drift >= 6200 && inp.phi_proxy <= params.rsa.phi_min_apply.saturating_add(800) {
            if let Some(proposal) = tune_coupling(inp, params) {
                proposals.push(proposal);
            }
        }
        if inp.surprise >= 6800 && inp.replay_pressure >= 5000 {
            if let Some(proposal) = tune_replay(inp, params) {
                proposals.push(proposal);
            }
        }
        if inp.coherence_plv <= 4200 && inp.risk <= 2800 {
            if let Some(proposal) = tune_attention(inp, params) {
                proposals.push(proposal);
            }
        }

        proposals.truncate(4);
        let chosen_commit = choose_best(&proposals).map(|proposal| proposal.commit);
        let commit = commit_rsa_outputs(inp.cycle_id, &proposals, chosen_commit, false, None);
        RsaOutputs {
            cycle_id: inp.cycle_id,
            proposals,
            chosen: chosen_commit,
            applied: false,
            new_params_commit: None,
            commit,
        }
    }

    pub fn can_apply(&self, cycle_id: u64) -> bool {
        cycle_id
            >= self
                .last_apply_cycle
                .saturating_add(u64::from(self.min_apply_gap))
    }

    pub fn mark_applied(&mut self, cycle_id: u64) {
        self.last_apply_cycle = cycle_id;
        self.commit = commit_rsa_core(self.last_apply_cycle, self.min_apply_gap);
    }
}

pub fn rsa_gates_allow(inp: &RsaInputs, params: &StructuralParams) -> bool {
    nsr_verdict(inp.nsr_verdict) == 0
        && inp.policy_ok
        && inp.phi_proxy >= params.rsa.phi_min_apply
        && inp.risk <= params.rsa.risk_max_apply
}

pub trait RsaEngine {
    fn analyze(&self, ctx: &SleepContext, recent_evidence: &[EvidenceId]) -> SleepReport;
}

#[derive(Clone, Default)]
pub struct MockRsaEngine;

impl MockRsaEngine {
    pub fn new() -> Self {
        Self
    }

    fn proposal_from_evidence(
        &self,
        ctx: &SleepContext,
        evidence: &EvidenceId,
    ) -> EvolutionProposal {
        let mut hasher = Hasher::new();
        hasher.update(&ctx.fixed_seed);
        hasher.update(evidence.as_str().as_bytes());
        let digest = Digest32::new(*hasher.finalize().as_bytes());
        let expected_gain = (digest.as_bytes()[0] % 21) as i16 - 10;
        let risk = digest.as_bytes()[1] as u16;
        EvolutionProposal {
            id: digest,
            title: format!("RSA hint for {}", evidence.as_str()),
            rationale: format!("derived from evidence {}", evidence.as_str()),
            expected_gain,
            risk,
        }
    }
}

impl RsaEngine for MockRsaEngine {
    fn analyze(&self, ctx: &SleepContext, recent_evidence: &[EvidenceId]) -> SleepReport {
        let proposals = recent_evidence
            .iter()
            .map(|evidence| self.proposal_from_evidence(ctx, evidence))
            .collect::<Vec<_>>();
        let metrics = vec![
            ("rsa_evidence".to_string(), recent_evidence.len() as u32),
            (
                "integration_score".to_string(),
                ctx.integration_score as u32,
            ),
        ];
        SleepReport { proposals, metrics }
    }
}

pub struct SleepCoordinator<P, R, O, B>
where
    P: SleepPhaseGate,
    R: RsaEngine,
    O: OpenEvolvePort,
    B: BusPublisher<SleepReportReady>,
{
    policy: P,
    rsa: R,
    openevolve: O,
    archive: Arc<dyn ExperienceAppender + Send + Sync>,
    bus: B,
    structural_store: Option<Arc<std::sync::Mutex<StructuralStore>>>,
    structural_archive_store: Option<Arc<dyn ArchiveStore + Send + Sync>>,
    structural_archive_appender: std::sync::Mutex<ArchiveAppender>,
}

impl<P, R, O, B> SleepCoordinator<P, R, O, B>
where
    P: SleepPhaseGate,
    R: RsaEngine,
    O: OpenEvolvePort,
    B: BusPublisher<SleepReportReady>,
{
    pub fn new(
        policy: P,
        rsa: R,
        openevolve: O,
        archive: Arc<dyn ExperienceAppender + Send + Sync>,
        bus: B,
    ) -> Self {
        Self {
            policy,
            rsa,
            openevolve,
            archive,
            bus,
            structural_store: None,
            structural_archive_store: None,
            structural_archive_appender: std::sync::Mutex::new(ArchiveAppender::new()),
        }
    }

    pub fn with_structural_store(
        mut self,
        store: Arc<std::sync::Mutex<StructuralStore>>,
        archive_store: Arc<dyn ArchiveStore + Send + Sync>,
    ) -> Self {
        self.structural_store = Some(store);
        self.structural_archive_store = Some(archive_store);
        self
    }

    pub fn run_sleep_phase(
        &self,
        cycle_id: u64,
        fixed_seed: [u8; 32],
        integration_score: u16,
        recent_evidence: &[EvidenceId],
        structural_stats: Option<StructuralCycleStats>,
        structural_proposal: Option<StructuralDeltaProposal>,
    ) -> Option<SleepReportReady> {
        if !self.policy.allow_sleep() {
            return None;
        }

        let ctx = SleepContext {
            cycle_id,
            fixed_seed,
            integration_score,
        };
        let base_report = self.rsa.analyze(&ctx, recent_evidence);
        let proposals = self.openevolve.propose(&ctx, &base_report);
        let mut metrics = base_report.metrics.clone();
        metrics.push(("openevolve_selected".to_string(), proposals.len() as u32));
        let report = SleepReport { proposals, metrics };
        let record = build_sleep_record(&ctx, &report);
        let evidence_id = self.archive.append(record);
        self.maybe_commit_structural(cycle_id, structural_stats, structural_proposal);
        let event = SleepReportReady {
            evidence_id,
            cycle_id,
        };
        self.bus.publish(event.clone());
        Some(event)
    }

    fn maybe_commit_structural(
        &self,
        cycle_id: u64,
        structural_stats: Option<StructuralCycleStats>,
        structural_proposal: Option<StructuralDeltaProposal>,
    ) {
        let (Some(stats), Some(proposal)) = (structural_stats, structural_proposal) else {
            return;
        };
        let Some(store_handle) = self.structural_store.as_ref() else {
            return;
        };
        let mut store = match store_handle.lock() {
            Ok(store) => store,
            Err(err) => err.into_inner(),
        };
        let gates = StructuralGates::new(
            stats.nsr_verdict != NSR_DENY_VERDICT,
            stats.policy_ok,
            stats.consistency_ok,
            stats.coherence_plv,
            stats.drift,
            stats.surprise,
        );
        let result = store.evaluate_and_commit(&proposal, &gates);
        self.append_structural_records(cycle_id, &proposal, &result, &store);
    }

    fn append_structural_records(
        &self,
        cycle_id: u64,
        proposal: &StructuralDeltaProposal,
        result: &StructuralCommitResult,
        store: &StructuralStore,
    ) {
        let Some(archive_store) = self.structural_archive_store.as_ref() else {
            return;
        };
        let mut appender = self
            .structural_archive_appender
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let decision_flag = matches!(result, StructuralCommitResult::Committed { .. }) as u16;
        let meta = RecordMeta {
            cycle_id,
            tier: 0,
            flags: decision_flag,
            boundary_commit: Digest32::new([0u8; 32]),
        };
        let proposal_record = appender.build_record_with_commit(
            RecordKind::StructuralProposal,
            proposal.commit,
            meta,
        );
        archive_store.append(proposal_record);
        if let StructuralCommitResult::Committed { .. } = result {
            let params_record = appender.build_record_with_commit(
                RecordKind::StructuralParams,
                store.current.commit,
                meta,
            );
            archive_store.append(params_record);
        }
    }
}

#[derive(Clone, Debug)]
pub struct StructuralProposalEngine {
    coherence_floor: u16,
    integration_floor: u16,
    drift_ceiling: u16,
    nsr_warn_repeats: u16,
}

impl Default for StructuralProposalEngine {
    fn default() -> Self {
        Self {
            coherence_floor: 5200,
            integration_floor: 4200,
            drift_ceiling: 6800,
            nsr_warn_repeats: 2,
        }
    }
}

impl StructuralProposalEngine {
    pub fn maybe_propose(
        &self,
        store: &StructuralStore,
        cycle_id: u64,
        stats: &StructuralCycleStats,
        nsr_warn_streak: u16,
        evidence: Vec<Digest32>,
    ) -> Option<StructuralDeltaProposal> {
        let mut reasons = Vec::new();
        if stats.coherence_plv < self.coherence_floor {
            reasons.push(ReasonCode::CoherenceLow);
        }
        if stats.integration_phi < self.integration_floor {
            reasons.push(ReasonCode::IntegrationLow);
        }
        if stats.drift > self.drift_ceiling {
            reasons.push(ReasonCode::DriftHigh);
        }
        if nsr_warn_streak >= self.nsr_warn_repeats {
            reasons.push(ReasonCode::NsrWarn);
        }
        if reasons.is_empty() {
            return None;
        }

        let seed = structural_seed(store.current.commit, stats.commit);
        Some(store.propose(cycle_id, reasons, evidence, seed))
    }
}

fn tune_thresholds(inp: &RsaInputs, params: &StructuralParams) -> Option<StructuralProposal> {
    let warn = params.nsr.warn;
    let deny = params.nsr.deny;
    let warn_next = warn.saturating_sub(100);
    let deny_next = deny.saturating_sub(150).max(warn_next);
    let phi_next = params.rsa.phi_min_apply.saturating_add(100).min(10_000);
    let deltas = vec![
        ParamDelta::new(param_key("nsr.warn"), i32::from(warn), i32::from(warn_next)),
        ParamDelta::new(param_key("nsr.deny"), i32::from(deny), i32::from(deny_next)),
        ParamDelta::new(
            param_key("iit.phi_min_apply"),
            i32::from(params.rsa.phi_min_apply),
            i32::from(phi_next),
        ),
    ];
    build_proposal(inp, ProposalKind::TuneThresholds, deltas)
}

fn tune_coupling(inp: &RsaInputs, params: &StructuralParams) -> Option<StructuralProposal> {
    let coupling_next = params.onn.coupling.saturating_add(200).min(10_000);
    let ssm_next = params.ssm.selectivity.saturating_add(200).min(10_000);
    let deltas = vec![
        ParamDelta::new(
            param_key("onn.coupling"),
            i32::from(params.onn.coupling),
            i32::from(coupling_next),
        ),
        ParamDelta::new(
            param_key("ssm.selectivity"),
            i32::from(params.ssm.selectivity),
            i32::from(ssm_next),
        ),
    ];
    build_proposal(inp, ProposalKind::TuneCoupling, deltas)
}

fn tune_replay(inp: &RsaInputs, params: &StructuralParams) -> Option<StructuralProposal> {
    let micro_next = params.replay.micro_k.saturating_sub(1).max(1);
    let replay_trigger = params
        .snn
        .threshold_for(ucf_spikebus::SpikeKind::ReplayTrigger);
    let replay_trigger_next = replay_trigger.saturating_add(300).min(10_000);
    let deltas = vec![
        ParamDelta::new(
            param_key("replay.micro_k"),
            i32::from(params.replay.micro_k),
            i32::from(micro_next),
        ),
        ParamDelta::new(
            param_key("snn.threshold.replay_trigger"),
            i32::from(replay_trigger),
            i32::from(replay_trigger_next),
        ),
    ];
    build_proposal(inp, ProposalKind::TuneReplay, deltas)
}

fn tune_attention(inp: &RsaInputs, params: &StructuralParams) -> Option<StructuralProposal> {
    let attention = params
        .snn
        .threshold_for(ucf_spikebus::SpikeKind::AttentionShift);
    let attention_next = attention.saturating_add(250).min(10_000);
    let deltas = vec![ParamDelta::new(
        param_key("snn.threshold.attention_shift"),
        i32::from(attention),
        i32::from(attention_next),
    )];
    build_proposal(inp, ProposalKind::TuneAttention, deltas)
}

fn build_proposal(
    inp: &RsaInputs,
    kind: ProposalKind,
    mut deltas: Vec<ParamDelta>,
) -> Option<StructuralProposal> {
    if deltas.is_empty() {
        return None;
    }
    deltas.truncate(8);
    let (expected_gain, risk_cost) = proposal_scores(inp, &deltas, kind);
    let commit = commit_structural_proposal(
        inp.cycle_id,
        kind,
        inp.current_params_commit,
        &deltas,
        expected_gain,
        risk_cost,
    );
    Some(StructuralProposal {
        cycle_id: inp.cycle_id,
        kind,
        basis_commit: inp.current_params_commit,
        deltas,
        expected_gain,
        risk_cost,
        commit,
    })
}

fn proposal_scores(inp: &RsaInputs, deltas: &[ParamDelta], kind: ProposalKind) -> (i16, i16) {
    let coherence = i32::from(inp.coherence_plv) / 80;
    let phi = i32::from(inp.phi_proxy) / 70;
    let drift_penalty = i32::from(inp.drift) / 90;
    let surprise_penalty = i32::from(inp.surprise) / 120;
    let mut base_gain = coherence + phi - drift_penalty - surprise_penalty;
    base_gain += match kind {
        ProposalKind::TuneThresholds => 40,
        ProposalKind::TuneCoupling => 55,
        ProposalKind::TuneReplay => 30,
        ProposalKind::TuneAttention => 20,
        ProposalKind::Unknown(_) => 0,
    };
    let delta_mag: i32 = deltas
        .iter()
        .map(|delta| (delta.to - delta.from).abs())
        .sum();
    let base_risk = i32::from(inp.risk) / 50;
    let risk_cost = base_risk + delta_mag / 500;
    (bound_i16(base_gain), bound_i16(risk_cost))
}

fn bound_i16(value: i32) -> i16 {
    value.clamp(-10_000, 10_000) as i16
}

fn choose_best(proposals: &[StructuralProposal]) -> Option<&StructuralProposal> {
    proposals.iter().max_by(|left, right| {
        let left_score = i32::from(left.expected_gain) - i32::from(left.risk_cost);
        let right_score = i32::from(right.expected_gain) - i32::from(right.risk_cost);
        left_score
            .cmp(&right_score)
            .then_with(|| proposal_kind_rank(&right.kind).cmp(&proposal_kind_rank(&left.kind)))
            .then_with(|| right.commit.as_bytes().cmp(left.commit.as_bytes()))
    })
}

fn proposal_kind_rank(kind: &ProposalKind) -> u8 {
    match kind {
        ProposalKind::TuneThresholds => 0,
        ProposalKind::TuneCoupling => 1,
        ProposalKind::TuneReplay => 2,
        ProposalKind::TuneAttention => 3,
        ProposalKind::Unknown(_) => 4,
    }
}

fn proposal_kind_code(kind: ProposalKind) -> u16 {
    match kind {
        ProposalKind::TuneThresholds => 1,
        ProposalKind::TuneCoupling => 2,
        ProposalKind::TuneReplay => 3,
        ProposalKind::TuneAttention => 4,
        ProposalKind::Unknown(value) => 0x8000 | value,
    }
}

fn nsr_verdict(value: u8) -> u8 {
    value & 0b11
}

fn commit_param_delta(key: u16, from: i32, to: i32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(RSA_PARAM_DELTA_DOMAIN);
    hasher.update(&key.to_be_bytes());
    hasher.update(&from.to_be_bytes());
    hasher.update(&to.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_structural_proposal(
    cycle_id: u64,
    kind: ProposalKind,
    basis_commit: Digest32,
    deltas: &[ParamDelta],
    expected_gain: i16,
    risk_cost: i16,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(RSA_PROPOSAL_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(&proposal_kind_code(kind).to_be_bytes());
    hasher.update(basis_commit.as_bytes());
    hasher.update(&expected_gain.to_be_bytes());
    hasher.update(&risk_cost.to_be_bytes());
    hasher.update(
        &u16::try_from(deltas.len())
            .unwrap_or(u16::MAX)
            .to_be_bytes(),
    );
    for delta in deltas {
        hasher.update(delta.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

#[allow(clippy::too_many_arguments)]
fn commit_rsa_inputs(
    cycle_id: u64,
    sleep_active: bool,
    phase_commit: Digest32,
    coherence_plv: u16,
    phi_proxy: u16,
    nsr_verdict: u8,
    nsr_trace_root: Digest32,
    policy_ok: bool,
    risk: u16,
    drift: u16,
    surprise: u16,
    replay_pressure: u16,
    ssm_salience: u16,
    ncde_energy: u16,
    current_params_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(RSA_INPUT_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(&[sleep_active as u8, policy_ok as u8]);
    hasher.update(phase_commit.as_bytes());
    hasher.update(&coherence_plv.to_be_bytes());
    hasher.update(&phi_proxy.to_be_bytes());
    hasher.update(&[nsr_verdict]);
    hasher.update(nsr_trace_root.as_bytes());
    hasher.update(&risk.to_be_bytes());
    hasher.update(&drift.to_be_bytes());
    hasher.update(&surprise.to_be_bytes());
    hasher.update(&replay_pressure.to_be_bytes());
    hasher.update(&ssm_salience.to_be_bytes());
    hasher.update(&ncde_energy.to_be_bytes());
    hasher.update(current_params_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_rsa_outputs(
    cycle_id: u64,
    proposals: &[StructuralProposal],
    chosen: Option<Digest32>,
    applied: bool,
    new_params_commit: Option<Digest32>,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(RSA_OUTPUT_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(
        &u16::try_from(proposals.len())
            .unwrap_or(u16::MAX)
            .to_be_bytes(),
    );
    for proposal in proposals {
        hasher.update(proposal.commit.as_bytes());
    }
    match chosen {
        Some(commit) => {
            hasher.update(&[1]);
            hasher.update(commit.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    hasher.update(&[applied as u8]);
    match new_params_commit {
        Some(commit) => {
            hasher.update(&[1]);
            hasher.update(commit.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_rsa_core(last_apply_cycle: u64, min_apply_gap: u16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(RSA_CORE_DOMAIN);
    hasher.update(&last_apply_cycle.to_be_bytes());
    hasher.update(&min_apply_gap.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn structural_seed(current_commit: Digest32, stats_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(STRUCTURAL_SEED_DOMAIN);
    hasher.update(current_commit.as_bytes());
    hasher.update(stats_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn build_sleep_record(ctx: &SleepContext, report: &SleepReport) -> ExperienceRecord {
    let mut metrics = report.metrics.clone();
    metrics.sort_by(|left, right| left.0.cmp(&right.0));
    let metric_payload = metrics
        .iter()
        .map(|(key, value)| format!("{key}:{value}"))
        .collect::<Vec<_>>()
        .join(",");
    let proposals_payload = report
        .proposals
        .iter()
        .map(|proposal| {
            format!(
                "{}|{}|{}|{}|{}",
                proposal.id,
                proposal.title,
                proposal.expected_gain,
                proposal.risk,
                proposal.rationale
            )
        })
        .collect::<Vec<_>>()
        .join(";");
    let payload = format!(
        "derived=sleep_report;sleep_cycle={};integration_score={};proposal_count={};metrics={};proposals={}",
        ctx.cycle_id,
        ctx.integration_score,
        report.proposals.len(),
        metric_payload,
        proposals_payload
    )
    .into_bytes();
    ExperienceRecord {
        record_id: format!("sleep-report-{}", ctx.cycle_id),
        observed_at_ms: ctx.cycle_id,
        subject_id: "sleep-phase".to_string(),
        payload,
        digest: None,
        vrf_tag: None,
        proof_ref: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use prost::Message;
    use ucf_archive::InMemoryArchive;
    use ucf_bus::{BusSubscriber, InMemoryBus};
    use ucf_policy_ecology::{PolicyEcology, PolicyRule, PolicyWeights};
    use ucf_structural_store::StructuralStore;
    use ucf_types::v1::spec::ExperienceRecord as ProtoExperienceRecord;

    #[test]
    fn sleep_phase_denied_by_default() {
        let policy = ucf_policy_ecology::DefaultPolicyEcology::new();
        let rsa = MockRsaEngine::new();
        let openevolve = ucf_openevolve_port::MockOpenEvolvePort::default();
        let archive = Arc::new(InMemoryArchive::new());
        let bus = InMemoryBus::new();
        let receiver = bus.subscribe();
        let coordinator = SleepCoordinator::new(policy, rsa, openevolve, archive.clone(), bus);

        let result = coordinator.run_sleep_phase(1, [1u8; 32], 9, &[], None, None);

        assert!(result.is_none());
        assert!(archive.list().is_empty());
        assert!(receiver.try_recv().is_err());
    }

    #[test]
    fn sleep_phase_allowed_appends_report_and_emits_event() {
        let policy = PolicyEcology::new(1, vec![PolicyRule::AllowSleepPhase], PolicyWeights);
        let rsa = MockRsaEngine::new();
        let openevolve = ucf_openevolve_port::MockOpenEvolvePort::new(5);
        let archive = Arc::new(InMemoryArchive::new());
        let bus = InMemoryBus::new();
        let receiver = bus.subscribe();
        let coordinator = SleepCoordinator::new(policy, rsa, openevolve, archive.clone(), bus);
        let evidence = vec![EvidenceId::new("ev-1"), EvidenceId::new("ev-2")];

        let result = coordinator.run_sleep_phase(42, [7u8; 32], 12, &evidence, None, None);

        assert!(result.is_some());
        assert_eq!(archive.list().len(), 1);
        let envelope = &archive.list()[0];
        let proof = envelope.proof.as_ref().expect("proof envelope");
        let record =
            ProtoExperienceRecord::decode(proof.payload.as_slice()).expect("decode record");
        let payload = String::from_utf8(record.payload).expect("utf8 payload");
        assert!(payload.contains("derived=sleep_report"));
        assert!(payload.contains("proposal_count=2"));
        let event = receiver.recv().expect("sleep event");
        assert_eq!(event.cycle_id, 42);
    }

    #[test]
    fn sleep_report_proposals_are_deterministically_sorted() {
        let ctx = SleepContext {
            cycle_id: 5,
            fixed_seed: [2u8; 32],
            integration_score: 3,
        };
        let report = SleepReport {
            proposals: vec![
                EvolutionProposal {
                    id: Digest32::new([9u8; 32]),
                    title: "b".to_string(),
                    rationale: "b".to_string(),
                    expected_gain: 2,
                    risk: 4,
                },
                EvolutionProposal {
                    id: Digest32::new([1u8; 32]),
                    title: "a".to_string(),
                    rationale: "a".to_string(),
                    expected_gain: 4,
                    risk: 2,
                },
                EvolutionProposal {
                    id: Digest32::new([3u8; 32]),
                    title: "c".to_string(),
                    rationale: "c".to_string(),
                    expected_gain: 4,
                    risk: 8,
                },
            ],
            metrics: Vec::new(),
        };
        let openevolve = ucf_openevolve_port::MockOpenEvolvePort::new(3);

        let proposals = openevolve.propose(&ctx, &report);

        assert_eq!(proposals[0].id, Digest32::new([1u8; 32]));
        assert_eq!(proposals[1].id, Digest32::new([3u8; 32]));
        assert_eq!(proposals[2].id, Digest32::new([9u8; 32]));
    }

    #[test]
    fn rsa_tick_is_deterministic() {
        let params = StructuralStore::default_params();
        let inputs = RsaInputs::new(
            12,
            true,
            Digest32::new([1u8; 32]),
            4100,
            3600,
            0,
            Digest32::new([2u8; 32]),
            true,
            2000,
            6500,
            7200,
            6400,
            3000,
            2000,
            params.commit,
        );
        let mut core = RsaCore::default();

        let first = core.tick_with_params(&inputs, &params);
        let second = core.tick_with_params(&inputs, &params);

        assert_eq!(first.proposals, second.proposals);
        assert_eq!(first.chosen, second.chosen);
    }

    #[test]
    fn rsa_gating_blocks_when_nsr_denies_or_policy_fails() {
        let params = StructuralStore::default_params();
        let base = RsaInputs::new(
            13,
            true,
            Digest32::new([3u8; 32]),
            5000,
            4000,
            2,
            Digest32::new([4u8; 32]),
            true,
            2000,
            4000,
            3000,
            2000,
            1000,
            900,
            params.commit,
        );
        let mut core = RsaCore::default();

        let denied = core.tick_with_params(&base, &params);
        assert!(denied.proposals.is_empty());

        let policy_blocked = RsaInputs {
            policy_ok: false,
            ..base.clone()
        };
        let blocked = core.tick_with_params(&policy_blocked, &params);
        assert!(blocked.proposals.is_empty());
    }

    #[test]
    fn rsa_blocks_when_sleep_inactive() {
        let params = StructuralStore::default_params();
        let inputs = RsaInputs::new(
            14,
            false,
            Digest32::new([5u8; 32]),
            5000,
            4000,
            0,
            Digest32::new([6u8; 32]),
            true,
            2000,
            4000,
            3000,
            2000,
            1000,
            900,
            params.commit,
        );
        let mut core = RsaCore::default();

        let outputs = core.tick_with_params(&inputs, &params);

        assert!(outputs.proposals.is_empty());
        assert!(outputs.chosen.is_none());
    }

    #[test]
    fn rsa_apply_gap_blocks_repeated_commit() {
        let mut core = RsaCore {
            min_apply_gap: 3,
            ..Default::default()
        };
        core.mark_applied(10);

        assert!(!core.can_apply(11));
        assert!(core.can_apply(13));
    }
}
