#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::collections::BTreeMap;

use blake3::Hasher;
use ucf_params_registry::{ParamSnapshotDelta, ParamTargetId};
use ucf_types::Digest32;

const RSA_PARAM_DELTA_DOMAIN: &[u8] = b"ucf.rsa.v0.param_delta.v1";
const RSA_PROPOSAL_DOMAIN: &[u8] = b"ucf.rsa.v0.proposal.v1";
const RSA_DECISION_DOMAIN: &[u8] = b"ucf.rsa.v0.decision.v1";
const RSA_INPUT_DOMAIN: &[u8] = b"ucf.rsa.v0.inputs.v1";
const RSA_OUTPUT_DOMAIN: &[u8] = b"ucf.rsa.v0.outputs.v1";
const RSA_CORE_DOMAIN: &[u8] = b"ucf.rsa.v0.core.v1";
const RSA_RATIONALE_DOMAIN: &[u8] = b"ucf.rsa.v0.rationale.v1";

pub const PHI_APPLY_MIN: u16 = 3200;
pub const PLV_APPLY_MIN: u16 = 3200;
pub const RISK_APPLY_MAX: u16 = 7500;
const TREND_EPS: u16 = 200;
const DROP_THRESH: u16 = 350;
const DRIFT_RISE_THRESH: u16 = 400;

const NSR_ALLOW_VERDICT: u8 = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParamTarget {
    OnnCoupling,
    OnnLockWindow,
    TcfAttK,
    TcfReplayK,
    TcfEnergyK,
    NcdeGainPhase,
    NcdeGainSpike,
    NcdeLeak,
    CdeScoreStep,
    CdeEdgeThresh,
    FeatureSpikeThresh,
    ThreatSpikeThresh,
    Unknown(u16),
}

impl ParamTarget {
    pub fn as_u16(self) -> u16 {
        match self {
            Self::OnnCoupling => 1,
            Self::OnnLockWindow => 2,
            Self::TcfAttK => 3,
            Self::TcfReplayK => 4,
            Self::TcfEnergyK => 5,
            Self::NcdeGainPhase => 6,
            Self::NcdeGainSpike => 7,
            Self::NcdeLeak => 8,
            Self::CdeScoreStep => 9,
            Self::CdeEdgeThresh => 10,
            Self::FeatureSpikeThresh => 11,
            Self::ThreatSpikeThresh => 12,
            Self::Unknown(value) => value,
        }
    }
}

impl Ord for ParamTarget {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_u16().cmp(&other.as_u16())
    }
}

impl PartialOrd for ParamTarget {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamDelta {
    pub target: ParamTarget,
    pub delta: i16,
    pub commit: Digest32,
}

impl ParamDelta {
    pub fn new(target: ParamTarget, delta: i16) -> Self {
        let commit = commit_param_delta(target, delta);
        Self {
            target,
            delta,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RsaProposal {
    pub cycle_id: u64,
    pub base_params_commit: Digest32,
    pub deltas: Vec<ParamDelta>,
    pub rationale_commit: Digest32,
    pub commit: Digest32,
}

impl RsaProposal {
    pub fn new(
        cycle_id: u64,
        base_params_commit: Digest32,
        mut deltas: Vec<ParamDelta>,
        rationale_commit: Digest32,
    ) -> Self {
        deltas.sort_by_key(|delta| delta.target.as_u16());
        deltas.truncate(8);
        let commit = commit_proposal(cycle_id, base_params_commit, &deltas, rationale_commit);
        Self {
            cycle_id,
            base_params_commit,
            deltas,
            rationale_commit,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RsaDecision {
    pub apply: bool,
    pub reason_mask: u32,
    pub commit: Digest32,
}

impl RsaDecision {
    pub fn new(apply: bool, reason_mask: u32) -> Self {
        let commit = commit_decision(apply, reason_mask);
        Self {
            apply,
            reason_mask,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RsaInputs {
    pub cycle_id: u64,
    pub sleep_active: bool,
    pub replay_active: bool,
    pub nsr_verdict: u8,
    pub policy_ok: bool,
    pub phi_proxy: u16,
    pub global_plv: u16,
    pub drift: u16,
    pub surprise: u16,
    pub risk: u16,
    pub prev_phi: u16,
    pub prev_plv: u16,
    pub prev_drift: u16,
    pub proposal_strength_hint: i16,
    pub onn_params_commit: Digest32,
    pub tcf_params_commit: Digest32,
    pub ncde_params_commit: Digest32,
    pub cde_params_commit: Digest32,
    pub feature_params_commit: Digest32,
    pub commit: Digest32,
}

impl RsaInputs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cycle_id: u64,
        sleep_active: bool,
        replay_active: bool,
        nsr_verdict: u8,
        policy_ok: bool,
        phi_proxy: u16,
        global_plv: u16,
        drift: u16,
        surprise: u16,
        risk: u16,
        prev_phi: u16,
        prev_plv: u16,
        prev_drift: u16,
        proposal_strength_hint: i16,
        onn_params_commit: Digest32,
        tcf_params_commit: Digest32,
        ncde_params_commit: Digest32,
        cde_params_commit: Digest32,
        feature_params_commit: Digest32,
    ) -> Self {
        let commit = commit_inputs(
            cycle_id,
            sleep_active,
            replay_active,
            nsr_verdict,
            policy_ok,
            phi_proxy,
            global_plv,
            drift,
            surprise,
            risk,
            prev_phi,
            prev_plv,
            prev_drift,
            proposal_strength_hint,
            onn_params_commit,
            tcf_params_commit,
            ncde_params_commit,
            cde_params_commit,
            feature_params_commit,
        );
        Self {
            cycle_id,
            sleep_active,
            replay_active,
            nsr_verdict,
            policy_ok,
            phi_proxy,
            global_plv,
            drift,
            surprise,
            risk,
            prev_phi,
            prev_plv,
            prev_drift,
            proposal_strength_hint,
            onn_params_commit,
            tcf_params_commit,
            ncde_params_commit,
            cde_params_commit,
            feature_params_commit,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RsaOutputs {
    pub cycle_id: u64,
    pub proposal: Option<RsaProposal>,
    pub decision: RsaDecision,
    pub applied_params_root: Digest32,
    pub snapshot_chain_commit: Digest32,
    pub commit: Digest32,
}

impl RsaOutputs {
    pub fn recompute_commit(&mut self) {
        self.commit = commit_outputs(
            self.cycle_id,
            self.proposal.as_ref(),
            &self.decision,
            self.applied_params_root,
            self.snapshot_chain_commit,
        );
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RsaCore {
    pub last_snapshot_commit: Digest32,
    pub last_applied_root: Digest32,
    pub pending_revert: bool,
    pub commit: Digest32,
    last_applied_deltas: Vec<ParamDelta>,
    last_apply_cycle: Option<u64>,
}

impl Default for RsaCore {
    fn default() -> Self {
        let last_snapshot_commit = Digest32::new([0u8; 32]);
        let last_applied_root = Digest32::new([0u8; 32]);
        let pending_revert = false;
        let commit = commit_core(last_snapshot_commit, last_applied_root, pending_revert);
        Self {
            last_snapshot_commit,
            last_applied_root,
            pending_revert,
            commit,
            last_applied_deltas: Vec::new(),
            last_apply_cycle: None,
        }
    }
}

impl RsaCore {
    pub fn tick(&mut self, inp: &RsaInputs) -> RsaOutputs {
        let pending_before = self.pending_revert;
        let rollback_triggered = self.check_rollback_trigger(inp);
        let current_root = params_root(
            inp.onn_params_commit,
            inp.tcf_params_commit,
            inp.ncde_params_commit,
            inp.cde_params_commit,
            inp.feature_params_commit,
        );
        let proposal = if inp.sleep_active || inp.replay_active {
            if pending_before && !self.last_applied_deltas.is_empty() {
                let deltas = self
                    .last_applied_deltas
                    .iter()
                    .map(|delta| ParamDelta::new(delta.target, delta.delta.saturating_neg()))
                    .collect::<Vec<_>>();
                let rationale_commit = commit_rationale(inp, true);
                Some(RsaProposal::new(
                    inp.cycle_id,
                    current_root,
                    deltas,
                    rationale_commit,
                ))
            } else if rollback_triggered {
                None
            } else {
                self.generate_proposal(inp, current_root)
            }
        } else {
            None
        };
        let decision = decide_apply(inp, proposal.as_ref());
        let mut outputs = RsaOutputs {
            cycle_id: inp.cycle_id,
            proposal,
            decision,
            applied_params_root: current_root,
            snapshot_chain_commit: self.last_snapshot_commit,
            commit: Digest32::new([0u8; 32]),
        };
        outputs.recompute_commit();
        outputs
    }

    pub fn record_apply(&mut self, deltas: &[ParamDelta], applied_root: Digest32, cycle_id: u64) {
        self.last_applied_root = applied_root;
        self.last_applied_deltas = deltas.to_vec();
        self.last_apply_cycle = Some(cycle_id);
        self.pending_revert = false;
        self.commit = commit_core(self.last_snapshot_commit, self.last_applied_root, false);
    }

    pub fn record_snapshot_chain(&mut self, snapshot_chain_commit: Digest32) {
        self.last_snapshot_commit = snapshot_chain_commit;
        self.commit = commit_core(
            self.last_snapshot_commit,
            self.last_applied_root,
            self.pending_revert,
        );
    }

    fn check_rollback_trigger(&mut self, inp: &RsaInputs) -> bool {
        let Some(last_cycle) = self.last_apply_cycle else {
            return false;
        };
        if inp.cycle_id != last_cycle.saturating_add(1) {
            return false;
        }
        let phi_drop = inp.prev_phi.saturating_sub(inp.phi_proxy) >= DROP_THRESH;
        let drift_rise = inp.drift.saturating_sub(inp.prev_drift) >= DRIFT_RISE_THRESH;
        if phi_drop || drift_rise {
            self.pending_revert = true;
            self.commit = commit_core(self.last_snapshot_commit, self.last_applied_root, true);
            return true;
        }
        false
    }

    fn generate_proposal(&self, inp: &RsaInputs, current_root: Digest32) -> Option<RsaProposal> {
        let mut deltas: BTreeMap<ParamTarget, i16> = BTreeMap::new();
        let phi_trend = trend_dir(inp.phi_proxy, inp.prev_phi);
        let plv_trend = trend_dir(inp.global_plv, inp.prev_plv);
        let drift_trend = trend_dir(inp.drift, inp.prev_drift);
        let surprise_bucket = bucket_surprise(inp.surprise);
        let risk_bucket = bucket_risk(inp.risk);

        if drift_trend > 0 && phi_trend < 0 {
            bump_delta(&mut deltas, ParamTarget::TcfAttK, 120);
            bump_delta(&mut deltas, ParamTarget::TcfReplayK, 110);
            bump_delta(&mut deltas, ParamTarget::TcfEnergyK, 110);
            bump_delta(&mut deltas, ParamTarget::OnnLockWindow, 96);
        }
        if inp.global_plv < PLV_APPLY_MIN || plv_trend < 0 {
            bump_delta(&mut deltas, ParamTarget::OnnCoupling, 90);
        }
        if surprise_bucket > 1 && (2500..=6500).contains(&inp.phi_proxy) {
            bump_delta(&mut deltas, ParamTarget::FeatureSpikeThresh, 80);
        }
        if risk_bucket > 1 {
            bump_delta(&mut deltas, ParamTarget::OnnCoupling, -120);
            bump_delta(&mut deltas, ParamTarget::NcdeGainPhase, -140);
            bump_delta(&mut deltas, ParamTarget::NcdeGainSpike, -140);
        }
        if drift_trend > 0 && surprise_bucket > 1 {
            bump_delta(&mut deltas, ParamTarget::CdeScoreStep, 60);
            bump_delta(&mut deltas, ParamTarget::CdeEdgeThresh, 120);
        }
        if phi_trend > 0 && drift_trend < 0 {
            bump_delta(&mut deltas, ParamTarget::NcdeLeak, 60);
            bump_delta(&mut deltas, ParamTarget::ThreatSpikeThresh, 40);
        }

        if deltas.is_empty() {
            return None;
        }
        apply_hint_to_deltas(&mut deltas, inp.proposal_strength_hint);
        let delta_list = deltas
            .into_iter()
            .map(|(target, delta)| ParamDelta::new(target, delta))
            .collect::<Vec<_>>();
        let rationale_commit = commit_rationale(inp, false);
        Some(RsaProposal::new(
            inp.cycle_id,
            current_root,
            delta_list,
            rationale_commit,
        ))
    }
}

pub fn params_root(
    onn_params_commit: Digest32,
    tcf_params_commit: Digest32,
    ncde_params_commit: Digest32,
    cde_params_commit: Digest32,
    feature_params_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.rsa.v0.params.root.v1");
    hasher.update(onn_params_commit.as_bytes());
    hasher.update(tcf_params_commit.as_bytes());
    hasher.update(ncde_params_commit.as_bytes());
    hasher.update(cde_params_commit.as_bytes());
    hasher.update(feature_params_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

pub fn snapshot_deltas_from_proposal(proposal: &RsaProposal) -> Vec<ParamSnapshotDelta> {
    proposal
        .deltas
        .iter()
        .map(|delta| ParamSnapshotDelta::new(ParamTargetId(delta.target.as_u16()), delta.delta))
        .collect()
}

fn decide_apply(inp: &RsaInputs, proposal: Option<&RsaProposal>) -> RsaDecision {
    let mut reason_mask = 0u32;
    if !(inp.sleep_active || inp.replay_active) {
        reason_mask |= 1;
    }
    if inp.nsr_verdict != NSR_ALLOW_VERDICT {
        reason_mask |= 2;
    }
    if !inp.policy_ok {
        reason_mask |= 4;
    }
    if inp.phi_proxy < PHI_APPLY_MIN {
        reason_mask |= 8;
    }
    if inp.global_plv < PLV_APPLY_MIN {
        reason_mask |= 16;
    }
    if inp.risk > RISK_APPLY_MAX {
        reason_mask |= 32;
    }

    let apply = proposal.is_some() && reason_mask == 0;
    RsaDecision::new(apply, reason_mask)
}

fn trend_dir(current: u16, prev: u16) -> i16 {
    if current.saturating_sub(prev) >= TREND_EPS {
        return 1;
    }
    if prev.saturating_sub(current) >= TREND_EPS {
        return -1;
    }
    0
}

fn bucket_surprise(value: u16) -> u8 {
    match value {
        0..=2499 => 0,
        2500..=6999 => 1,
        _ => 2,
    }
}

fn bucket_risk(value: u16) -> u8 {
    match value {
        0..=2999 => 0,
        3000..=5999 => 1,
        _ => 2,
    }
}

fn bump_delta(deltas: &mut BTreeMap<ParamTarget, i16>, target: ParamTarget, delta: i16) {
    let entry = deltas.entry(target).or_insert(0);
    *entry = entry.saturating_add(delta);
}

fn apply_hint_to_deltas(deltas: &mut BTreeMap<ParamTarget, i16>, hint: i16) {
    if hint == 0 {
        return;
    }
    let hint = i32::from(hint.clamp(-5000, 5000));
    for value in deltas.values_mut() {
        let scaled = i32::from(*value) + (i32::from(*value) * hint / 10_000);
        *value = scaled.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    }
}

fn commit_param_delta(target: ParamTarget, delta: i16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(RSA_PARAM_DELTA_DOMAIN);
    hasher.update(&target.as_u16().to_be_bytes());
    hasher.update(&delta.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_proposal(
    cycle_id: u64,
    base_params_commit: Digest32,
    deltas: &[ParamDelta],
    rationale_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(RSA_PROPOSAL_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(base_params_commit.as_bytes());
    hasher.update(&(deltas.len() as u16).to_be_bytes());
    for delta in deltas {
        hasher.update(delta.commit.as_bytes());
    }
    hasher.update(rationale_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_decision(apply: bool, reason_mask: u32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(RSA_DECISION_DOMAIN);
    hasher.update(&[apply as u8]);
    hasher.update(&reason_mask.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[allow(clippy::too_many_arguments)]
fn commit_inputs(
    cycle_id: u64,
    sleep_active: bool,
    replay_active: bool,
    nsr_verdict: u8,
    policy_ok: bool,
    phi_proxy: u16,
    global_plv: u16,
    drift: u16,
    surprise: u16,
    risk: u16,
    prev_phi: u16,
    prev_plv: u16,
    prev_drift: u16,
    proposal_strength_hint: i16,
    onn_params_commit: Digest32,
    tcf_params_commit: Digest32,
    ncde_params_commit: Digest32,
    cde_params_commit: Digest32,
    feature_params_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(RSA_INPUT_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(&[sleep_active as u8]);
    hasher.update(&[replay_active as u8]);
    hasher.update(&[nsr_verdict]);
    hasher.update(&[policy_ok as u8]);
    hasher.update(&phi_proxy.to_be_bytes());
    hasher.update(&global_plv.to_be_bytes());
    hasher.update(&drift.to_be_bytes());
    hasher.update(&surprise.to_be_bytes());
    hasher.update(&risk.to_be_bytes());
    hasher.update(&prev_phi.to_be_bytes());
    hasher.update(&prev_plv.to_be_bytes());
    hasher.update(&prev_drift.to_be_bytes());
    hasher.update(&proposal_strength_hint.to_be_bytes());
    hasher.update(onn_params_commit.as_bytes());
    hasher.update(tcf_params_commit.as_bytes());
    hasher.update(ncde_params_commit.as_bytes());
    hasher.update(cde_params_commit.as_bytes());
    hasher.update(feature_params_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_outputs(
    cycle_id: u64,
    proposal: Option<&RsaProposal>,
    decision: &RsaDecision,
    applied_params_root: Digest32,
    snapshot_chain_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(RSA_OUTPUT_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    match proposal {
        Some(proposal) => {
            hasher.update(&[1]);
            hasher.update(proposal.commit.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    hasher.update(decision.commit.as_bytes());
    hasher.update(applied_params_root.as_bytes());
    hasher.update(snapshot_chain_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_core(
    last_snapshot_commit: Digest32,
    last_applied_root: Digest32,
    pending_revert: bool,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(RSA_CORE_DOMAIN);
    hasher.update(last_snapshot_commit.as_bytes());
    hasher.update(last_applied_root.as_bytes());
    hasher.update(&[pending_revert as u8]);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_rationale(inp: &RsaInputs, rollback: bool) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(RSA_RATIONALE_DOMAIN);
    hasher.update(&inp.phi_proxy.to_be_bytes());
    hasher.update(&inp.global_plv.to_be_bytes());
    hasher.update(&inp.drift.to_be_bytes());
    hasher.update(&inp.surprise.to_be_bytes());
    hasher.update(&inp.prev_phi.to_be_bytes());
    hasher.update(&inp.prev_plv.to_be_bytes());
    hasher.update(&inp.prev_drift.to_be_bytes());
    hasher.update(&inp.proposal_strength_hint.to_be_bytes());
    hasher.update(&[bucket_surprise(inp.surprise)]);
    hasher.update(&[bucket_risk(inp.risk)]);
    hasher.update(&[rollback as u8]);
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_inputs() -> RsaInputs {
        RsaInputs::new(
            10,
            true,
            false,
            NSR_ALLOW_VERDICT,
            true,
            4200,
            5200,
            2100,
            2800,
            1900,
            4300,
            5400,
            2200,
            0,
            Digest32::new([1u8; 32]),
            Digest32::new([2u8; 32]),
            Digest32::new([3u8; 32]),
            Digest32::new([4u8; 32]),
            Digest32::new([5u8; 32]),
        )
    }

    #[test]
    fn determinism_same_inputs() {
        let inputs = base_inputs();
        let mut core_a = RsaCore::default();
        let mut core_b = RsaCore::default();
        let out_a = core_a.tick(&inputs);
        let out_b = core_b.tick(&inputs);
        assert_eq!(out_a.commit, out_b.commit);
        assert_eq!(out_a.proposal, out_b.proposal);
    }

    #[test]
    fn gating_blocks_on_nsr_deny() {
        let mut inputs = base_inputs();
        inputs.nsr_verdict = 2;
        let mut core = RsaCore::default();
        let out = core.tick(&inputs);
        assert!(!out.decision.apply);
        assert_eq!(out.decision.reason_mask & 2, 2);
    }

    #[test]
    fn gating_blocks_on_nsr_restrict() {
        let mut inputs = base_inputs();
        inputs.nsr_verdict = 1;
        let mut core = RsaCore::default();
        let out = core.tick(&inputs);
        assert!(!out.decision.apply);
        assert_eq!(out.decision.reason_mask & 2, 2);
    }

    #[test]
    fn rollback_triggers_inverse_proposal_next_sleep() {
        let inputs = base_inputs();
        let mut core = RsaCore::default();
        let proposal = RsaProposal::new(
            inputs.cycle_id,
            params_root(
                inputs.onn_params_commit,
                inputs.tcf_params_commit,
                inputs.ncde_params_commit,
                inputs.cde_params_commit,
                inputs.feature_params_commit,
            ),
            vec![ParamDelta::new(ParamTarget::OnnCoupling, 100)],
            Digest32::new([9u8; 32]),
        );
        core.record_apply(&proposal.deltas, Digest32::new([8u8; 32]), inputs.cycle_id);

        let mut next_inputs = base_inputs();
        next_inputs.cycle_id = inputs.cycle_id + 1;
        next_inputs.prev_phi = 5000;
        next_inputs.phi_proxy = 4400;
        let out = core.tick(&next_inputs);
        assert!(core.pending_revert);
        assert!(out.proposal.is_none());

        let mut later_inputs = next_inputs.clone();
        later_inputs.cycle_id += 1;
        let out = core.tick(&later_inputs);
        let proposal = out.proposal.expect("inverse proposal");
        assert_eq!(proposal.deltas[0].delta, -100);
    }
}
