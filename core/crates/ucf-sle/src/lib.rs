#![forbid(unsafe_code)]

use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Mutex;

use blake3::Hasher;
use ucf_cde_port::CdeHypothesis;
use ucf_geist::{encode_self_state, SelfState};
use ucf_nsr_port::NsrReport;
use ucf_types::{AiOutput, Digest32, OutputChannel};
use ucf_workspace::{SignalKind, WorkspaceSnapshot};

const DOMAIN_INPUT: &[u8] = b"ucf.sle.input.v1";
const DOMAIN_OUTPUT: &[u8] = b"ucf.sle.output.v1";
const DOMAIN_REPORT: &[u8] = b"ucf.sle.report.v1";
const DOMAIN_SELF_SYMBOL: &[u8] = b"ucf.sle.self_symbol.v1";
const DOMAIN_SELF_REFLEX: &[u8] = b"ucf.sle.self_reflex.v1";
const DOMAIN_SLE_INPUTS: &[u8] = b"ucf.sle.inputs.v2";
const DOMAIN_SLE_STIMULUS: &[u8] = b"ucf.sle.stimulus.v2";
const DOMAIN_SLE_OUTPUTS: &[u8] = b"ucf.sle.outputs.v2";
const DOMAIN_SLE_SELF_SYMBOL: &[u8] = b"ucf.sle.self_symbol.v2";

const SLE_MAX_VALUE: i16 = 2000;
const SLE_MAX_STIMULI: usize = 8;
const SLE_HIGH_RISK: u16 = 9000;
const SLE_DECAY_MAX: u16 = 10_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelfReflex {
    pub loop_level: u8,
    pub self_symbol: Digest32,
    pub delta: i16,
    pub commit: Digest32,
}

#[derive(Debug)]
pub struct SleEngine {
    max_level: AtomicU8,
    history: Mutex<SleHistory>,
}

#[derive(Debug, Default)]
struct SleHistory {
    loop_level: u8,
    last_high_priority: Vec<Digest32>,
}

impl SleEngine {
    pub fn new(max_level: u8) -> Self {
        Self {
            max_level: AtomicU8::new(max_level),
            history: Mutex::new(SleHistory::default()),
        }
    }

    pub fn set_max_level(&self, max_level: u8) {
        self.max_level.store(max_level, Ordering::Relaxed);
    }

    pub fn evaluate(&self, snapshot: &WorkspaceSnapshot, last_state: &SelfState) -> SelfReflex {
        let self_symbol = hash_with_domain(DOMAIN_SELF_SYMBOL, |hasher| {
            hasher.update(&encode_self_state(last_state));
            hasher.update(ucf_workspace::encode_workspace_snapshot(snapshot).as_slice());
        });

        let (stable_high_priority, instability) = {
            let current = high_priority_digests(snapshot);
            let stable = self
                .history
                .lock()
                .ok()
                .map(|history| has_stable_overlap(&history.last_high_priority, &current))
                .unwrap_or(false);
            (stable, has_instability(snapshot))
        };

        let max_level = self.max_level.load(Ordering::Relaxed);
        let mut loop_level = self
            .history
            .lock()
            .map(|history| history.loop_level)
            .unwrap_or(0)
            .min(max_level);
        let delta: i16 = if instability {
            loop_level = loop_level.saturating_sub(1);
            -1
        } else if stable_high_priority {
            loop_level = loop_level.saturating_add(1).min(max_level);
            1
        } else {
            0
        };

        if let Ok(mut history) = self.history.lock() {
            history.loop_level = loop_level;
            history.last_high_priority = high_priority_digests(snapshot);
        }

        let commit = hash_with_domain(DOMAIN_SELF_REFLEX, |hasher| {
            hasher.update(&[loop_level]);
            hasher.update(&delta.to_be_bytes());
            hasher.update(self_symbol.as_bytes());
        });

        SelfReflex {
            loop_level,
            self_symbol,
            delta,
            commit,
        }
    }
}

fn high_priority_digests(snapshot: &WorkspaceSnapshot) -> Vec<Digest32> {
    let mut digests: Vec<Digest32> = snapshot
        .broadcast
        .iter()
        .filter(|signal| signal.priority >= 8000)
        .map(|signal| signal.digest)
        .collect();
    digests.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
    digests.dedup();
    digests
}

fn has_stable_overlap(previous: &[Digest32], current: &[Digest32]) -> bool {
    if previous.is_empty() || current.is_empty() {
        return false;
    }
    let prev: HashSet<Digest32> = previous.iter().copied().collect();
    let overlap = current
        .iter()
        .filter(|digest| prev.contains(digest))
        .count();
    overlap * 2 >= previous.len().max(1)
}

fn has_instability(snapshot: &WorkspaceSnapshot) -> bool {
    snapshot.broadcast.iter().any(|signal| {
        (matches!(signal.kind, SignalKind::Consistency)
            && (signal.summary.contains("NSR=DAMP")
                || signal.summary.contains("NSR=VIOL")
                || signal.summary.contains("DRIFT=HIGH")
                || signal.summary.contains("DRIFT=CRIT")))
            || (matches!(signal.kind, SignalKind::World)
                && (signal.summary.contains("SURPRISE=HIGH")
                    || signal.summary.contains("SURPRISE=CRIT")))
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoopFrame {
    pub input_commit: Digest32,
    pub output_commit: Digest32,
    pub report_commit: Digest32,
}

impl LoopFrame {
    pub fn seed(digest: Digest32) -> Self {
        Self {
            input_commit: digest,
            output_commit: digest,
            report_commit: digest,
        }
    }
}

#[derive(Debug)]
pub struct StrangeLoopEngine {
    pub depth: u8,
    frames: Mutex<VecDeque<LoopFrame>>,
}

impl StrangeLoopEngine {
    pub fn new(depth: u8) -> Self {
        Self {
            depth,
            frames: Mutex::new(VecDeque::new()),
        }
    }

    pub fn latest(&self) -> Option<LoopFrame> {
        self.frames.lock().ok()?.back().cloned()
    }

    pub fn reflect(
        &self,
        prev: &LoopFrame,
        current_output: &AiOutput,
        nsr_report: Option<&NsrReport>,
        cde_hyp: Option<&CdeHypothesis>,
    ) -> LoopFrame {
        let input_commit = hash_with_domain(DOMAIN_INPUT, |hasher| {
            hasher.update(prev.input_commit.as_bytes());
            hasher.update(prev.output_commit.as_bytes());
            hasher.update(prev.report_commit.as_bytes());
            encode_output(hasher, current_output);
        });

        let output_commit = hash_with_domain(DOMAIN_OUTPUT, |hasher| {
            hasher.update(input_commit.as_bytes());
            encode_output(hasher, current_output);
        });

        let report_commit = hash_with_domain(DOMAIN_REPORT, |hasher| {
            hasher.update(output_commit.as_bytes());
            encode_nsr_report(hasher, nsr_report);
            encode_cde_hypothesis(hasher, cde_hyp);
        });

        let frame = LoopFrame {
            input_commit,
            output_commit,
            report_commit,
        };

        if self.depth > 0 {
            if let Ok(mut frames) = self.frames.lock() {
                frames.push_back(frame.clone());
                let max_len = usize::from(self.depth);
                while frames.len() > max_len {
                    frames.pop_front();
                }
            }
        }

        frame
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SleLevel {
    L1,
    L2,
    L3,
}

impl SleLevel {
    fn as_u8(self) -> u8 {
        match self {
            SleLevel::L1 => 1,
            SleLevel::L2 => 2,
            SleLevel::L3 => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SleStimulusKind {
    WorldSummary,
    ReasoningSummary,
    SelfSummary,
    ThoughtOnlyPulse,
    Unknown(u16),
}

impl SleStimulusKind {
    fn as_u16(self) -> u16 {
        match self {
            SleStimulusKind::WorldSummary => 1,
            SleStimulusKind::ReasoningSummary => 2,
            SleStimulusKind::SelfSummary => 3,
            SleStimulusKind::ThoughtOnlyPulse => 4,
            SleStimulusKind::Unknown(code) => code,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SleStimulus {
    pub kind: SleStimulusKind,
    pub level: SleLevel,
    pub value: i16,
    pub src_commit: Digest32,
    pub commit: Digest32,
}

impl SleStimulus {
    fn new(kind: SleStimulusKind, level: SleLevel, value: i16, src_commit: Digest32) -> Self {
        let value = clamp_sle_value(value);
        let commit = hash_with_domain(DOMAIN_SLE_STIMULUS, |hasher| {
            hasher.update(&kind.as_u16().to_be_bytes());
            hasher.update(&[level.as_u8()]);
            hasher.update(&value.to_be_bytes());
            hasher.update(src_commit.as_bytes());
        });
        Self {
            kind,
            level,
            value,
            src_commit,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SleInputs {
    pub cycle_id: u64,
    pub phase_commit: Digest32,
    pub coherence_plv: u16,
    pub phi_proxy: u16,
    pub ssm_commit: Digest32,
    pub wm_salience: u16,
    pub wm_novelty: u16,
    pub ncde_commit: Digest32,
    pub ncde_energy: u16,
    pub geist_commit: Option<Digest32>,
    pub geist_consistency_ok: Option<bool>,
    pub ism_anchor_commit: Option<Digest32>,
    pub nsr_trace_root: Option<Digest32>,
    pub nsr_verdict: Option<u8>,
    pub policy_decision_commit: Option<Digest32>,
    pub risk: u16,
    pub drift: u16,
    pub surprise: u16,
    pub commit: Digest32,
}

impl SleInputs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cycle_id: u64,
        phase_commit: Digest32,
        coherence_plv: u16,
        phi_proxy: u16,
        ssm_commit: Digest32,
        wm_salience: u16,
        wm_novelty: u16,
        ncde_commit: Digest32,
        ncde_energy: u16,
        geist_commit: Option<Digest32>,
        geist_consistency_ok: Option<bool>,
        ism_anchor_commit: Option<Digest32>,
        nsr_trace_root: Option<Digest32>,
        nsr_verdict: Option<u8>,
        policy_decision_commit: Option<Digest32>,
        risk: u16,
        drift: u16,
        surprise: u16,
    ) -> Self {
        let commit = hash_with_domain(DOMAIN_SLE_INPUTS, |hasher| {
            hasher.update(&cycle_id.to_be_bytes());
            hasher.update(phase_commit.as_bytes());
            hasher.update(&coherence_plv.to_be_bytes());
            hasher.update(&phi_proxy.to_be_bytes());
            hasher.update(ssm_commit.as_bytes());
            hasher.update(&wm_salience.to_be_bytes());
            hasher.update(&wm_novelty.to_be_bytes());
            hasher.update(ncde_commit.as_bytes());
            hasher.update(&ncde_energy.to_be_bytes());
            encode_optional_digest(hasher, geist_commit);
            encode_optional_bool(hasher, geist_consistency_ok);
            encode_optional_digest(hasher, ism_anchor_commit);
            encode_optional_digest(hasher, nsr_trace_root);
            encode_optional_u8(hasher, nsr_verdict);
            encode_optional_digest(hasher, policy_decision_commit);
            hasher.update(&risk.to_be_bytes());
            hasher.update(&drift.to_be_bytes());
            hasher.update(&surprise.to_be_bytes());
        });
        Self {
            cycle_id,
            phase_commit,
            coherence_plv,
            phi_proxy,
            ssm_commit,
            wm_salience,
            wm_novelty,
            ncde_commit,
            ncde_energy,
            geist_commit,
            geist_consistency_ok,
            ism_anchor_commit,
            nsr_trace_root,
            nsr_verdict,
            policy_decision_commit,
            risk,
            drift,
            surprise,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SleOutputs {
    pub cycle_id: u64,
    pub stimuli: Vec<SleStimulus>,
    pub self_symbol_commit: Digest32,
    pub rate_limited: bool,
    pub commit: Digest32,
}

#[derive(Clone, Debug)]
pub struct SleCore {
    pub last_fire_cycle: u64,
    pub cooldown: u8,
    pub decay: u16,
    pub commit: Digest32,
    last_values: [i16; 3],
    last_pulse: i16,
}

impl SleCore {
    pub fn new(cooldown: u8, decay: u16) -> Self {
        let mut core = Self {
            last_fire_cycle: 0,
            cooldown: cooldown.max(1),
            decay: decay.min(SLE_DECAY_MAX),
            commit: Digest32::new([0u8; 32]),
            last_values: [0; 3],
            last_pulse: 0,
        };
        core.commit = core.commit_state();
        core
    }

    pub fn tick(&mut self, inp: &SleInputs) -> SleOutputs {
        let l1_src = resolve_src_commit(inp.ssm_commit, inp.phase_commit);
        let l2_src = inp
            .nsr_trace_root
            .unwrap_or_else(|| resolve_src_commit(inp.ncde_commit, inp.phase_commit));
        let l3_src = inp
            .geist_commit
            .unwrap_or_else(|| resolve_src_commit(inp.phase_commit, inp.ssm_commit));

        let l1_raw = compute_world_summary(inp.wm_salience, inp.wm_novelty, inp.coherence_plv);
        let l2_raw =
            compute_reasoning_summary(inp.nsr_verdict, inp.nsr_trace_root.is_some(), inp.phi_proxy);
        let l3_raw = compute_self_summary(
            inp.geist_consistency_ok,
            inp.drift,
            inp.risk,
            inp.ism_anchor_commit.is_some(),
        );

        let l1_value = apply_decay(l1_raw, self.last_values[0], self.decay);
        let l2_value = apply_decay(l2_raw, self.last_values[1], self.decay);
        let l3_value = apply_decay(l3_raw, self.last_values[2], self.decay);

        let l1 = SleStimulus::new(
            SleStimulusKind::WorldSummary,
            SleLevel::L1,
            l1_value,
            l1_src,
        );
        let l2 = SleStimulus::new(
            SleStimulusKind::ReasoningSummary,
            SleLevel::L2,
            l2_value,
            l2_src,
        );
        let l3 = SleStimulus::new(SleStimulusKind::SelfSummary, SleLevel::L3, l3_value, l3_src);

        self.last_values = [l1_value, l2_value, l3_value];

        let self_symbol_commit = hash_with_domain(DOMAIN_SLE_SELF_SYMBOL, |hasher| {
            hasher.update(l1.src_commit.as_bytes());
            hasher.update(l2.src_commit.as_bytes());
            hasher.update(l3.src_commit.as_bytes());
            hasher.update(inp.phase_commit.as_bytes());
        });

        let mut stimuli = vec![l1, l2, l3];
        let mut rate_limited = false;
        let mut pulse_value = thought_pulse_value(l1_value, l2_value, l3_value, self.last_pulse);
        let high_risk = inp.risk >= SLE_HIGH_RISK;
        let deny = matches!(inp.nsr_verdict, Some(2));
        if high_risk || deny {
            pulse_value = -pulse_value.abs().max(400);
            self.cooldown = self.cooldown.max(3);
        }
        if inp.cycle_id
            < self
                .last_fire_cycle
                .saturating_add(u64::from(self.cooldown))
        {
            rate_limited = true;
        } else {
            let pulse = SleStimulus::new(
                SleStimulusKind::ThoughtOnlyPulse,
                SleLevel::L3,
                pulse_value,
                self_symbol_commit,
            );
            stimuli.push(pulse);
            self.last_fire_cycle = inp.cycle_id;
            self.last_pulse = pulse_value;
        }

        stimuli.truncate(SLE_MAX_STIMULI);

        let commit = hash_with_domain(DOMAIN_SLE_OUTPUTS, |hasher| {
            hasher.update(&inp.cycle_id.to_be_bytes());
            hasher.update(self_symbol_commit.as_bytes());
            hasher.update(&[rate_limited as u8]);
            hasher.update(
                &u32::try_from(stimuli.len())
                    .unwrap_or(u32::MAX)
                    .to_be_bytes(),
            );
            for stimulus in &stimuli {
                hasher.update(stimulus.commit.as_bytes());
            }
        });

        self.commit = self.commit_state();

        SleOutputs {
            cycle_id: inp.cycle_id,
            stimuli,
            self_symbol_commit,
            rate_limited,
            commit,
        }
    }

    fn commit_state(&self) -> Digest32 {
        hash_with_domain(DOMAIN_SLE_OUTPUTS, |hasher| {
            hasher.update(&self.last_fire_cycle.to_be_bytes());
            hasher.update(&self.cooldown.to_be_bytes());
            hasher.update(&self.decay.to_be_bytes());
            for value in &self.last_values {
                hasher.update(&value.to_be_bytes());
            }
            hasher.update(&self.last_pulse.to_be_bytes());
        })
    }
}

impl Default for SleCore {
    fn default() -> Self {
        Self::new(2, 1200)
    }
}

fn resolve_src_commit(primary: Digest32, fallback: Digest32) -> Digest32 {
    if primary.as_bytes().iter().all(|byte| *byte == 0) {
        fallback
    } else {
        primary
    }
}

fn compute_world_summary(wm_salience: u16, wm_novelty: u16, coherence_plv: u16) -> i16 {
    let salience = i32::from(wm_salience);
    let novelty = i32::from(wm_novelty);
    let coherence = i32::from(coherence_plv);
    let raw = (salience + novelty) / 2 + (coherence - 5_000) / 2;
    clamp_sle_value(raw as i16)
}

fn compute_reasoning_summary(nsr_verdict: Option<u8>, has_trace: bool, phi_proxy: u16) -> i16 {
    let verdict_bias = match nsr_verdict.unwrap_or(0) {
        0 => 600,
        1 => -400,
        2 => -900,
        _ => 0,
    };
    let trace_bias = if has_trace { 200 } else { -200 };
    let phi_bias = (i32::from(phi_proxy) - 4_000) / 4;
    clamp_sle_value((verdict_bias + trace_bias + phi_bias) as i16)
}

fn compute_self_summary(
    geist_consistency_ok: Option<bool>,
    drift: u16,
    risk: u16,
    has_anchor: bool,
) -> i16 {
    let base = match geist_consistency_ok {
        Some(true) => 800,
        Some(false) => -800,
        None => 0,
    };
    let anchor = if has_anchor { 300 } else { -150 };
    let drift_penalty = i32::from(drift) / 5;
    let risk_penalty = i32::from(risk) / 6;
    let raw = base + anchor - drift_penalty - risk_penalty;
    clamp_sle_value(raw as i16)
}

fn thought_pulse_value(l1: i16, l2: i16, l3: i16, prev: i16) -> i16 {
    let weighted = i32::from(l1) + i32::from(l2) + i32::from(l3) * 2;
    let blended = weighted / 4 + i32::from(prev) / 5;
    clamp_sle_value(blended as i16)
}

fn apply_decay(raw: i16, prev: i16, decay: u16) -> i16 {
    let decay = decay.min(SLE_DECAY_MAX) as i32;
    let retain = 10_000i32.saturating_sub(decay);
    let retained = (i32::from(prev) * retain) / 10_000;
    clamp_sle_value((i32::from(raw) + retained) as i16)
}

fn clamp_sle_value(value: i16) -> i16 {
    value.clamp(-SLE_MAX_VALUE, SLE_MAX_VALUE)
}

fn encode_optional_digest(hasher: &mut Hasher, value: Option<Digest32>) {
    match value {
        Some(commit) => {
            hasher.update(&[1]);
            hasher.update(commit.as_bytes());
        }
        None => {
            hasher.update(&[0]);
            hasher.update(&[0u8; Digest32::LEN]);
        }
    }
}

fn encode_optional_bool(hasher: &mut Hasher, value: Option<bool>) {
    match value {
        Some(flag) => {
            hasher.update(&[1, flag as u8]);
        }
        None => {
            hasher.update(&[0, 0]);
        }
    }
}

fn encode_optional_u8(hasher: &mut Hasher, value: Option<u8>) {
    match value {
        Some(val) => {
            hasher.update(&[1, val]);
        }
        None => {
            hasher.update(&[0, 0]);
        }
    }
}

fn hash_with_domain(domain: &[u8], f: impl FnOnce(&mut Hasher)) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(domain);
    f(&mut hasher);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn encode_output(hasher: &mut Hasher, output: &AiOutput) {
    let channel_tag: u8 = match output.channel {
        OutputChannel::Thought => 0,
        OutputChannel::Speech => 1,
    };
    hasher.update(&[channel_tag]);
    hasher.update(&output.confidence.to_be_bytes());
    hasher.update(
        &u64::try_from(output.content.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    hasher.update(output.content.as_bytes());
    match output.rationale_commit {
        Some(commit) => {
            hasher.update(&[1]);
            hasher.update(commit.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    match output.integration_score {
        Some(score) => {
            hasher.update(&[1]);
            hasher.update(&score.to_be_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn encode_nsr_report(hasher: &mut Hasher, report: Option<&NsrReport>) {
    match report {
        Some(report) => {
            hasher.update(&[1]);
            hasher.update(&[report.verdict.as_u8()]);
            hasher.update(report.proof_digest.as_bytes());
            hasher.update(
                &u64::try_from(report.violations.len())
                    .unwrap_or(0)
                    .to_be_bytes(),
            );
            for violation in &report.violations {
                hasher.update(violation.code.as_bytes());
                hasher.update(violation.detail_digest.as_bytes());
                hasher.update(&violation.severity.to_be_bytes());
                hasher.update(violation.commit.as_bytes());
            }
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn encode_cde_hypothesis(hasher: &mut Hasher, hyp: Option<&CdeHypothesis>) {
    match hyp {
        Some(hyp) => {
            hasher.update(&[1]);
            hasher.update(hyp.digest.as_bytes());
            hasher.update(&u64::try_from(hyp.nodes).unwrap_or(0).to_be_bytes());
            hasher.update(&u64::try_from(hyp.edges).unwrap_or(0).to_be_bytes());
            hasher.update(&hyp.confidence.to_be_bytes());
            hasher.update(
                &u64::try_from(hyp.interventions.len())
                    .unwrap_or(0)
                    .to_be_bytes(),
            );
            for intervention in &hyp.interventions {
                hasher.update(
                    &u64::try_from(intervention.kind.len())
                        .unwrap_or(0)
                        .to_be_bytes(),
                );
                hasher.update(intervention.kind.as_bytes());
            }
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_recursion_controller::{RecursionController, RecursionInputs};
    use ucf_workspace::{SignalKind, WorkspaceSignal};

    #[test]
    fn reflect_is_deterministic() {
        let engine = StrangeLoopEngine::new(4);
        let prev = LoopFrame::seed(Digest32::new([1u8; 32]));
        let output = AiOutput {
            channel: OutputChannel::Thought,
            content: "ok".to_string(),
            confidence: 900,
            rationale_commit: Some(Digest32::new([2u8; 32])),
            integration_score: Some(1234),
        };
        let report = NsrReport {
            verdict: ucf_nsr_port::NsrVerdict::Allow,
            causal_report_commit: Digest32::new([0u8; 32]),
            violations: vec![ucf_nsr_port::NsrViolation {
                code: "rule-a".to_string(),
                detail_digest: Digest32::new([8u8; 32]),
                severity: 100,
                commit: Digest32::new([9u8; 32]),
            }],
            proof_digest: Digest32::new([7u8; 32]),
            commit: Digest32::new([6u8; 32]),
        };
        let hyp = CdeHypothesis {
            digest: Digest32::new([3u8; 32]),
            nodes: 2,
            edges: 1,
            confidence: 9000,
            interventions: vec![ucf_cde_port::InterventionStub::new("none")],
        };

        let first = engine.reflect(&prev, &output, Some(&report), Some(&hyp));
        let second = engine.reflect(&prev, &output, Some(&report), Some(&hyp));

        assert_eq!(first, second);
        let latest = engine.latest().expect("frame stored");
        assert_eq!(latest, second);
    }

    #[test]
    fn loop_level_increases_on_stable_high_priority() {
        let engine = SleEngine::new(4);
        let state = SelfState::builder(1).build();
        let signal = WorkspaceSignal {
            kind: SignalKind::Risk,
            priority: 9000,
            digest: Digest32::new([9u8; 32]),
            summary: "RISK=9000 DENY".to_string(),
            slot: 0,
        };
        let snapshot_a = WorkspaceSnapshot {
            cycle_id: 1,
            broadcast: vec![signal.clone()],
            recursion_used: 0,
            spike_seen_root: Digest32::new([0u8; 32]),
            spike_accepted_root: Digest32::new([0u8; 32]),
            spike_counts: Vec::new(),
            spike_causal_link_count: 0,
            spike_consistency_alert_count: 0,
            spike_thought_only_count: 0,
            spike_output_intent_count: 0,
            spike_cap_hit: false,
            ncde_commit: Digest32::new([0u8; 32]),
            ncde_state_commit: Digest32::new([0u8; 32]),
            ncde_energy: 0,
            cde_commit: Digest32::new([0u8; 32]),
            cde_graph_commit: Digest32::new([0u8; 32]),
            cde_top_edges: Vec::new(),
            ssm_commit: Digest32::new([0u8; 32]),
            ssm_state_commit: Digest32::new([0u8; 32]),
            influence_v2_commit: Digest32::new([0u8; 32]),
            influence_pulses_root: Digest32::new([0u8; 32]),
            influence_node_values: Vec::new(),
            onn_states_commit: Digest32::new([0u8; 32]),
            onn_global_plv: 0,
            onn_pair_locks_commit: Digest32::new([0u8; 32]),
            onn_phase_frame_commit: Digest32::new([0u8; 32]),
            iit_output: None,
            nsr_trace_root: None,
            nsr_prev_commit: None,
            nsr_verdict: None,
            rsa_commit: Digest32::new([0u8; 32]),
            rsa_chosen: None,
            rsa_applied: false,
            rsa_new_params_commit: None,
            sle_commit: Digest32::new([0u8; 32]),
            sle_self_symbol_commit: Digest32::new([0u8; 32]),
            sle_rate_limited: false,
            internal_utterances: Vec::new(),
            commit: Digest32::new([1u8; 32]),
        };
        let snapshot_b = WorkspaceSnapshot {
            cycle_id: 2,
            broadcast: vec![signal],
            recursion_used: 0,
            spike_seen_root: Digest32::new([0u8; 32]),
            spike_accepted_root: Digest32::new([0u8; 32]),
            spike_counts: Vec::new(),
            spike_causal_link_count: 0,
            spike_consistency_alert_count: 0,
            spike_thought_only_count: 0,
            spike_output_intent_count: 0,
            spike_cap_hit: false,
            ncde_commit: Digest32::new([0u8; 32]),
            ncde_state_commit: Digest32::new([0u8; 32]),
            ncde_energy: 0,
            cde_commit: Digest32::new([0u8; 32]),
            cde_graph_commit: Digest32::new([0u8; 32]),
            cde_top_edges: Vec::new(),
            ssm_commit: Digest32::new([0u8; 32]),
            ssm_state_commit: Digest32::new([0u8; 32]),
            influence_v2_commit: Digest32::new([0u8; 32]),
            influence_pulses_root: Digest32::new([0u8; 32]),
            influence_node_values: Vec::new(),
            onn_states_commit: Digest32::new([0u8; 32]),
            onn_global_plv: 0,
            onn_pair_locks_commit: Digest32::new([0u8; 32]),
            onn_phase_frame_commit: Digest32::new([0u8; 32]),
            iit_output: None,
            nsr_trace_root: None,
            nsr_prev_commit: None,
            nsr_verdict: None,
            rsa_commit: Digest32::new([0u8; 32]),
            rsa_chosen: None,
            rsa_applied: false,
            rsa_new_params_commit: None,
            sle_commit: Digest32::new([0u8; 32]),
            sle_self_symbol_commit: Digest32::new([0u8; 32]),
            sle_rate_limited: false,
            internal_utterances: Vec::new(),
            commit: Digest32::new([2u8; 32]),
        };

        let _ = engine.evaluate(&snapshot_a, &state);
        let reflex = engine.evaluate(&snapshot_b, &state);

        assert!(reflex.loop_level >= 1);
        assert_eq!(reflex.delta, 1);
    }

    #[test]
    fn loop_level_decreases_on_surprise_or_instability() {
        let engine = SleEngine::new(4);
        let state = SelfState::builder(1).build();
        let stable_signal = WorkspaceSignal {
            kind: SignalKind::Risk,
            priority: 9000,
            digest: Digest32::new([8u8; 32]),
            summary: "RISK=9000 DENY".to_string(),
            slot: 0,
        };
        let snapshot_a = WorkspaceSnapshot {
            cycle_id: 1,
            broadcast: vec![stable_signal.clone()],
            recursion_used: 0,
            spike_seen_root: Digest32::new([0u8; 32]),
            spike_accepted_root: Digest32::new([0u8; 32]),
            spike_counts: Vec::new(),
            spike_causal_link_count: 0,
            spike_consistency_alert_count: 0,
            spike_thought_only_count: 0,
            spike_output_intent_count: 0,
            spike_cap_hit: false,
            ncde_commit: Digest32::new([0u8; 32]),
            ncde_state_commit: Digest32::new([0u8; 32]),
            ncde_energy: 0,
            cde_commit: Digest32::new([0u8; 32]),
            cde_graph_commit: Digest32::new([0u8; 32]),
            cde_top_edges: Vec::new(),
            ssm_commit: Digest32::new([0u8; 32]),
            ssm_state_commit: Digest32::new([0u8; 32]),
            influence_v2_commit: Digest32::new([0u8; 32]),
            influence_pulses_root: Digest32::new([0u8; 32]),
            influence_node_values: Vec::new(),
            onn_states_commit: Digest32::new([0u8; 32]),
            onn_global_plv: 0,
            onn_pair_locks_commit: Digest32::new([0u8; 32]),
            onn_phase_frame_commit: Digest32::new([0u8; 32]),
            iit_output: None,
            nsr_trace_root: None,
            nsr_prev_commit: None,
            nsr_verdict: None,
            rsa_commit: Digest32::new([0u8; 32]),
            rsa_chosen: None,
            rsa_applied: false,
            rsa_new_params_commit: None,
            sle_commit: Digest32::new([0u8; 32]),
            sle_self_symbol_commit: Digest32::new([0u8; 32]),
            sle_rate_limited: false,
            internal_utterances: Vec::new(),
            commit: Digest32::new([1u8; 32]),
        };
        let snapshot_b = WorkspaceSnapshot {
            cycle_id: 2,
            broadcast: vec![stable_signal],
            recursion_used: 0,
            spike_seen_root: Digest32::new([0u8; 32]),
            spike_accepted_root: Digest32::new([0u8; 32]),
            spike_counts: Vec::new(),
            spike_causal_link_count: 0,
            spike_consistency_alert_count: 0,
            spike_thought_only_count: 0,
            spike_output_intent_count: 0,
            spike_cap_hit: false,
            ncde_commit: Digest32::new([0u8; 32]),
            ncde_state_commit: Digest32::new([0u8; 32]),
            ncde_energy: 0,
            cde_commit: Digest32::new([0u8; 32]),
            cde_graph_commit: Digest32::new([0u8; 32]),
            cde_top_edges: Vec::new(),
            ssm_commit: Digest32::new([0u8; 32]),
            ssm_state_commit: Digest32::new([0u8; 32]),
            influence_v2_commit: Digest32::new([0u8; 32]),
            influence_pulses_root: Digest32::new([0u8; 32]),
            influence_node_values: Vec::new(),
            onn_states_commit: Digest32::new([0u8; 32]),
            onn_global_plv: 0,
            onn_pair_locks_commit: Digest32::new([0u8; 32]),
            onn_phase_frame_commit: Digest32::new([0u8; 32]),
            iit_output: None,
            nsr_trace_root: None,
            nsr_prev_commit: None,
            nsr_verdict: None,
            rsa_commit: Digest32::new([0u8; 32]),
            rsa_chosen: None,
            rsa_applied: false,
            rsa_new_params_commit: None,
            sle_commit: Digest32::new([0u8; 32]),
            sle_self_symbol_commit: Digest32::new([0u8; 32]),
            sle_rate_limited: false,
            internal_utterances: Vec::new(),
            commit: Digest32::new([2u8; 32]),
        };
        let _ = engine.evaluate(&snapshot_a, &state);
        let _ = engine.evaluate(&snapshot_b, &state);

        let surprise_signal = WorkspaceSignal {
            kind: SignalKind::World,
            priority: 9000,
            digest: Digest32::new([7u8; 32]),
            summary: "SURPRISE=CRIT BAND=CRIT".to_string(),
            slot: 0,
        };
        let snapshot_c = WorkspaceSnapshot {
            cycle_id: 3,
            broadcast: vec![surprise_signal],
            recursion_used: 0,
            spike_seen_root: Digest32::new([0u8; 32]),
            spike_accepted_root: Digest32::new([0u8; 32]),
            spike_counts: Vec::new(),
            spike_causal_link_count: 0,
            spike_consistency_alert_count: 0,
            spike_thought_only_count: 0,
            spike_output_intent_count: 0,
            spike_cap_hit: false,
            ncde_commit: Digest32::new([0u8; 32]),
            ncde_state_commit: Digest32::new([0u8; 32]),
            ncde_energy: 0,
            cde_commit: Digest32::new([0u8; 32]),
            cde_graph_commit: Digest32::new([0u8; 32]),
            cde_top_edges: Vec::new(),
            ssm_commit: Digest32::new([0u8; 32]),
            ssm_state_commit: Digest32::new([0u8; 32]),
            influence_v2_commit: Digest32::new([0u8; 32]),
            influence_pulses_root: Digest32::new([0u8; 32]),
            influence_node_values: Vec::new(),
            onn_states_commit: Digest32::new([0u8; 32]),
            onn_global_plv: 0,
            onn_pair_locks_commit: Digest32::new([0u8; 32]),
            onn_phase_frame_commit: Digest32::new([0u8; 32]),
            iit_output: None,
            nsr_trace_root: None,
            nsr_prev_commit: None,
            nsr_verdict: None,
            rsa_commit: Digest32::new([0u8; 32]),
            rsa_chosen: None,
            rsa_applied: false,
            rsa_new_params_commit: None,
            sle_commit: Digest32::new([0u8; 32]),
            sle_self_symbol_commit: Digest32::new([0u8; 32]),
            sle_rate_limited: false,
            internal_utterances: Vec::new(),
            commit: Digest32::new([3u8; 32]),
        };

        let reflex = engine.evaluate(&snapshot_c, &state);
        assert_eq!(reflex.delta, -1);
    }

    #[test]
    fn loop_level_respects_recursion_budget_cap() {
        let controller = RecursionController::default();
        let budget = controller.compute(&RecursionInputs {
            phi: 2000,
            drift_score: 9000,
            surprise: 9000,
            risk: 9000,
            attn_gain: 1000,
            focus: 9000,
        });
        let engine = SleEngine::new(6);
        engine.set_max_level(budget.max_depth);
        let state = SelfState::builder(1).build();
        let signal = WorkspaceSignal {
            kind: SignalKind::Risk,
            priority: 9000,
            digest: Digest32::new([6u8; 32]),
            summary: "RISK=9000 DENY".to_string(),
            slot: 0,
        };
        let snapshot_a = WorkspaceSnapshot {
            cycle_id: 1,
            broadcast: vec![signal.clone()],
            recursion_used: 0,
            spike_seen_root: Digest32::new([0u8; 32]),
            spike_accepted_root: Digest32::new([0u8; 32]),
            spike_counts: Vec::new(),
            spike_causal_link_count: 0,
            spike_consistency_alert_count: 0,
            spike_thought_only_count: 0,
            spike_output_intent_count: 0,
            spike_cap_hit: false,
            ncde_commit: Digest32::new([0u8; 32]),
            ncde_state_commit: Digest32::new([0u8; 32]),
            ncde_energy: 0,
            cde_commit: Digest32::new([0u8; 32]),
            cde_graph_commit: Digest32::new([0u8; 32]),
            cde_top_edges: Vec::new(),
            ssm_commit: Digest32::new([0u8; 32]),
            ssm_state_commit: Digest32::new([0u8; 32]),
            influence_v2_commit: Digest32::new([0u8; 32]),
            influence_pulses_root: Digest32::new([0u8; 32]),
            influence_node_values: Vec::new(),
            onn_states_commit: Digest32::new([0u8; 32]),
            onn_global_plv: 0,
            onn_pair_locks_commit: Digest32::new([0u8; 32]),
            onn_phase_frame_commit: Digest32::new([0u8; 32]),
            iit_output: None,
            nsr_trace_root: None,
            nsr_prev_commit: None,
            nsr_verdict: None,
            rsa_commit: Digest32::new([0u8; 32]),
            rsa_chosen: None,
            rsa_applied: false,
            rsa_new_params_commit: None,
            sle_commit: Digest32::new([0u8; 32]),
            sle_self_symbol_commit: Digest32::new([0u8; 32]),
            sle_rate_limited: false,
            internal_utterances: Vec::new(),
            commit: Digest32::new([4u8; 32]),
        };
        let snapshot_b = WorkspaceSnapshot {
            cycle_id: 2,
            broadcast: vec![signal],
            recursion_used: 0,
            spike_seen_root: Digest32::new([0u8; 32]),
            spike_accepted_root: Digest32::new([0u8; 32]),
            spike_counts: Vec::new(),
            spike_causal_link_count: 0,
            spike_consistency_alert_count: 0,
            spike_thought_only_count: 0,
            spike_output_intent_count: 0,
            spike_cap_hit: false,
            ncde_commit: Digest32::new([0u8; 32]),
            ncde_state_commit: Digest32::new([0u8; 32]),
            ncde_energy: 0,
            cde_commit: Digest32::new([0u8; 32]),
            cde_graph_commit: Digest32::new([0u8; 32]),
            cde_top_edges: Vec::new(),
            ssm_commit: Digest32::new([0u8; 32]),
            ssm_state_commit: Digest32::new([0u8; 32]),
            influence_v2_commit: Digest32::new([0u8; 32]),
            influence_pulses_root: Digest32::new([0u8; 32]),
            influence_node_values: Vec::new(),
            onn_states_commit: Digest32::new([0u8; 32]),
            onn_global_plv: 0,
            onn_pair_locks_commit: Digest32::new([0u8; 32]),
            onn_phase_frame_commit: Digest32::new([0u8; 32]),
            iit_output: None,
            nsr_trace_root: None,
            nsr_prev_commit: None,
            nsr_verdict: None,
            rsa_commit: Digest32::new([0u8; 32]),
            rsa_chosen: None,
            rsa_applied: false,
            rsa_new_params_commit: None,
            sle_commit: Digest32::new([0u8; 32]),
            sle_self_symbol_commit: Digest32::new([0u8; 32]),
            sle_rate_limited: false,
            internal_utterances: Vec::new(),
            commit: Digest32::new([5u8; 32]),
        };

        let _ = engine.evaluate(&snapshot_a, &state);
        let reflex = engine.evaluate(&snapshot_b, &state);

        assert!(reflex.loop_level <= budget.max_depth);
    }

    fn base_inputs(cycle_id: u64) -> SleInputs {
        SleInputs::new(
            cycle_id,
            Digest32::new([1u8; 32]),
            5000,
            4200,
            Digest32::new([2u8; 32]),
            3000,
            2000,
            Digest32::new([3u8; 32]),
            1200,
            Some(Digest32::new([4u8; 32])),
            Some(true),
            Some(Digest32::new([5u8; 32])),
            Some(Digest32::new([6u8; 32])),
            Some(0),
            None,
            2000,
            1500,
            1200,
        )
    }

    #[test]
    fn sle_tick_is_deterministic() {
        let mut core = SleCore::new(2, 1200);
        let mut core_clone = SleCore::new(2, 1200);
        let inputs = base_inputs(10);
        let first = core.tick(&inputs);
        let second = core_clone.tick(&inputs);
        assert_eq!(first.commit, second.commit);
        assert_eq!(first.self_symbol_commit, second.self_symbol_commit);
        assert_eq!(first.stimuli, second.stimuli);
    }

    #[test]
    fn sle_cooldown_suppresses_thought_pulse() {
        let mut core = SleCore::new(3, 1200);
        let first = core.tick(&base_inputs(4));
        assert!(first
            .stimuli
            .iter()
            .any(|stim| matches!(stim.kind, SleStimulusKind::ThoughtOnlyPulse)));

        let second = core.tick(&base_inputs(5));
        assert!(second.rate_limited);
        assert!(!second
            .stimuli
            .iter()
            .any(|stim| matches!(stim.kind, SleStimulusKind::ThoughtOnlyPulse)));
    }

    #[test]
    fn sle_deny_forces_inhibitory_pulse() {
        let mut core = SleCore::new(1, 1200);
        let mut inputs = base_inputs(5);
        inputs.nsr_verdict = Some(2);
        let outputs = core.tick(&inputs);
        let pulse = outputs
            .stimuli
            .iter()
            .find(|stim| matches!(stim.kind, SleStimulusKind::ThoughtOnlyPulse))
            .expect("pulse emitted");
        assert!(pulse.value < 0);
    }
}
