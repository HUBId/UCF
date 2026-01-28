#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_spikebus::{SpikeEvent, SpikeKind, SpikeModuleId};
use ucf_types::Digest32;

const INPUT_DOMAIN: &[u8] = b"ucf.cde.v1.inputs";
const EDGE_DOMAIN: &[u8] = b"ucf.cde.v1.edge";
const DAG_DOMAIN: &[u8] = b"ucf.cde.v1.dag";
const INTERVENTION_DOMAIN: &[u8] = b"ucf.cde.v1.intervention";
const OUTPUT_DOMAIN: &[u8] = b"ucf.cde.v1.outputs";
const SUMMARY_DOMAIN: &[u8] = b"ucf.cde.v1.summary";
const CORE_DOMAIN: &[u8] = b"ucf.cde.v1.core";
const SPIKE_PAYLOAD_DOMAIN: &[u8] = b"ucf.cde.v1.spike.payload";
const DELTA_DOMAIN: &[u8] = b"ucf.cde.v1.delta";

const MAX_NODES: usize = 24;
const MAX_EDGES: usize = 64;
const MAX_TOP_EDGES: usize = 8;
const MAX_LAG: usize = 8;
const MAX_SCORE: i16 = 10_000;
const MIN_SCORE: i16 = -10_000;
const SCORE_SPIKE_THRESHOLD: i16 = 6_000;
const INTERVENTION_SCORE_MIN: i16 = 3_000;
const INTERVENTION_SCORE_BOOST: i16 = 1_200;
const INTERVENTION_SCORE_PENALTY: i16 = 2_000;
const PROXY_SCALE: i32 = 64;
const DECAY_DIV: i32 = 12;
const CENTER_VALUE: i32 = 5_000;
const TTFS_MAX: u16 = 180;
const TTFS_MIN: u16 = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VarId {
    PerceptionSalience,
    PerceptionNovelty,
    AttentionGain,
    LearningRate,
    ReplayPressure,
    SleepDrive,
    NcdeEnergy,
    CoherencePlv,
    PhiProxy,
    Risk,
    Drift,
    Surprise,
    OutputSuppression,
    Unknown(u16),
}

impl VarId {
    pub fn to_u16(self) -> u16 {
        match self {
            Self::PerceptionSalience => 1,
            Self::PerceptionNovelty => 2,
            Self::AttentionGain => 3,
            Self::LearningRate => 4,
            Self::ReplayPressure => 5,
            Self::SleepDrive => 6,
            Self::NcdeEnergy => 7,
            Self::CoherencePlv => 8,
            Self::PhiProxy => 9,
            Self::Risk => 10,
            Self::Drift => 11,
            Self::Surprise => 12,
            Self::OutputSuppression => 13,
            Self::Unknown(value) => value,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CausalEdge {
    pub from: VarId,
    pub to: VarId,
    pub score: i16,
    pub lag: u8,
    pub commit: Digest32,
}

impl CausalEdge {
    pub fn new(from: VarId, to: VarId, score: i16, lag: u8) -> Self {
        let score = score.clamp(MIN_SCORE, MAX_SCORE);
        let lag = lag.min(MAX_LAG as u8);
        let commit = digest_edge(from, to, score, lag);
        Self {
            from,
            to,
            score,
            lag,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CausalDag {
    pub nodes: Vec<VarId>,
    pub edges: Vec<CausalEdge>,
    pub commit: Digest32,
}

impl CausalDag {
    pub fn new(nodes: Vec<VarId>, edges: Vec<CausalEdge>) -> Self {
        let commit = digest_dag(&nodes, &edges);
        Self {
            nodes,
            edges,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Intervention {
    pub cycle_id: u64,
    pub var: VarId,
    pub delta: i16,
    pub basis_commit: Digest32,
    pub commit: Digest32,
}

impl Intervention {
    pub fn new(cycle_id: u64, var: VarId, delta: i16, basis_commit: Digest32) -> Self {
        let commit = digest_intervention(cycle_id, var, delta, basis_commit);
        Self {
            cycle_id,
            var,
            delta,
            basis_commit,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CdeInputs {
    pub cycle_id: u64,
    pub phase_commit: Digest32,
    pub ssm_salience: u16,
    pub ssm_novelty: u16,
    pub attention_gain: u16,
    pub learning_rate: u16,
    pub replay_pressure: u16,
    pub sleep_drive: u16,
    pub ncde_energy: u16,
    pub coherence_plv: u16,
    pub phi_proxy: u16,
    pub risk: u16,
    pub drift: u16,
    pub surprise: u16,
    pub sleep_active: bool,
    pub replay_active: bool,
    pub spike_accepted_root: Digest32,
    pub commit: Digest32,
}

impl CdeInputs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cycle_id: u64,
        phase_commit: Digest32,
        ssm_salience: u16,
        ssm_novelty: u16,
        attention_gain: u16,
        learning_rate: u16,
        replay_pressure: u16,
        sleep_drive: u16,
        ncde_energy: u16,
        coherence_plv: u16,
        phi_proxy: u16,
        risk: u16,
        drift: u16,
        surprise: u16,
        sleep_active: bool,
        replay_active: bool,
        spike_accepted_root: Digest32,
    ) -> Self {
        let mut inputs = Self {
            cycle_id,
            phase_commit,
            ssm_salience,
            ssm_novelty,
            attention_gain,
            learning_rate,
            replay_pressure,
            sleep_drive,
            ncde_energy,
            coherence_plv,
            phi_proxy,
            risk,
            drift,
            surprise,
            sleep_active,
            replay_active,
            spike_accepted_root,
            commit: Digest32::new([0u8; 32]),
        };
        inputs.commit = digest_inputs(&inputs);
        inputs
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CdeOutputs {
    pub cycle_id: u64,
    pub dag_commit: Digest32,
    pub top_edges: Vec<CausalEdge>,
    pub intervention: Option<Intervention>,
    pub summary_commit: Digest32,
    pub causal_link_spikes: Vec<SpikeEvent>,
    pub commit: Digest32,
}

impl CdeOutputs {
    pub fn new(
        cycle_id: u64,
        dag_commit: Digest32,
        top_edges: Vec<CausalEdge>,
        intervention: Option<Intervention>,
        causal_link_spikes: Vec<SpikeEvent>,
    ) -> Self {
        let summary_commit = digest_summary(dag_commit, &top_edges, intervention.as_ref());
        let commit = digest_outputs(
            cycle_id,
            dag_commit,
            summary_commit,
            &top_edges,
            intervention.as_ref(),
            &causal_link_spikes,
        );
        Self {
            cycle_id,
            dag_commit,
            top_edges,
            intervention,
            summary_commit,
            causal_link_spikes,
            commit,
        }
    }
}

pub struct CdeCore {
    pub dag: CausalDag,
    pub prev_values: [u16; 12],
    pub last_intervention_cycle: u64,
    pub min_intervention_gap: u16,
    pub commit: Digest32,
    edge_scores: Vec<CausalEdge>,
    delta_history: Vec<[i16; 12]>,
    pending_intervention: Option<PendingIntervention>,
}

impl Default for CdeCore {
    fn default() -> Self {
        Self::new()
    }
}

impl CdeCore {
    pub fn new() -> Self {
        let nodes = default_nodes();
        let edge_scores = default_edge_scores();
        let dag = CausalDag::new(nodes, Vec::new());
        let commit = digest_core(dag.commit, Digest32::new([0u8; 32]));
        Self {
            dag,
            prev_values: [0; 12],
            last_intervention_cycle: 0,
            min_intervention_gap: 3,
            commit,
            edge_scores,
            delta_history: Vec::new(),
            pending_intervention: None,
        }
    }

    pub fn tick(&mut self, inp: &CdeInputs) -> CdeOutputs {
        let current_values = observed_values(inp);
        let deltas = compute_deltas(current_values, self.prev_values);
        self.prev_values = current_values;
        self.push_deltas(deltas);
        let mut intervention_feedback = None;
        if let Some(pending) = self.pending_intervention.take() {
            let outcome = evaluate_intervention(&pending, &self.delta_history);
            intervention_feedback = Some((pending.edge_key, outcome));
        }
        let mut updated_edges = Vec::with_capacity(self.edge_scores.len());
        for edge in &self.edge_scores {
            let proxy = edge_proxy(edge, &self.delta_history);
            let updated = update_edge_score(edge, proxy, &self.dag);
            updated_edges.push(updated);
        }
        if let Some((edge_key, outcome)) = intervention_feedback {
            if let Some(edge) = updated_edges.iter_mut().find(|edge| edge.key() == edge_key) {
                let adjusted = if outcome {
                    edge.score.saturating_add(INTERVENTION_SCORE_BOOST)
                } else {
                    edge.score.saturating_sub(INTERVENTION_SCORE_PENALTY)
                };
                *edge = CausalEdge::new(edge.from, edge.to, adjusted, edge.lag);
            }
        }
        self.edge_scores = updated_edges;
        let edges = select_dag_edges(&self.edge_scores);
        self.dag = CausalDag::new(self.dag.nodes.clone(), edges);
        let top_edges = select_top_edges(&self.dag.edges);
        let intervention = self.select_intervention(inp, &top_edges);
        let summary_commit = digest_summary(self.dag.commit, &top_edges, intervention.as_ref());
        let spikes = build_spikes(inp, &top_edges, summary_commit);
        let outputs = CdeOutputs::new(
            inp.cycle_id,
            self.dag.commit,
            top_edges,
            intervention,
            spikes,
        );
        self.commit = digest_core(self.dag.commit, inp.commit);
        outputs
    }

    fn select_intervention(
        &mut self,
        inp: &CdeInputs,
        top_edges: &[CausalEdge],
    ) -> Option<Intervention> {
        if !(inp.sleep_active || inp.replay_active) {
            return None;
        }
        if inp.cycle_id
            < self
                .last_intervention_cycle
                .saturating_add(u64::from(self.min_intervention_gap))
        {
            return None;
        }
        let candidate = top_edges
            .iter()
            .find(|edge| {
                edge.score.abs() >= INTERVENTION_SCORE_MIN
                    && has_observed_delta(edge.from)
                    && has_observed_delta(edge.to)
            })?
            .clone();
        let delta = derive_intervention_delta(inp.commit, candidate.from);
        let intervention = Intervention::new(inp.cycle_id, candidate.from, delta, self.dag.commit);
        self.last_intervention_cycle = inp.cycle_id;
        self.pending_intervention = Some(PendingIntervention {
            edge_key: candidate.key(),
            expected_sign: delta.signum(),
            target: candidate.to,
        });
        Some(intervention)
    }

    fn push_deltas(&mut self, deltas: [i16; 12]) {
        self.delta_history.insert(0, deltas);
        if self.delta_history.len() > MAX_LAG {
            self.delta_history.truncate(MAX_LAG);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EdgeKey {
    from: VarId,
    to: VarId,
    lag: u8,
}

#[derive(Clone, Copy, Debug)]
struct PendingIntervention {
    edge_key: EdgeKey,
    expected_sign: i16,
    target: VarId,
}

impl CausalEdge {
    fn key(&self) -> EdgeKey {
        EdgeKey {
            from: self.from,
            to: self.to,
            lag: self.lag,
        }
    }
}

const OBSERVED_VARS: [VarId; 12] = [
    VarId::PerceptionSalience,
    VarId::PerceptionNovelty,
    VarId::AttentionGain,
    VarId::LearningRate,
    VarId::ReplayPressure,
    VarId::SleepDrive,
    VarId::NcdeEnergy,
    VarId::CoherencePlv,
    VarId::PhiProxy,
    VarId::Risk,
    VarId::Drift,
    VarId::Surprise,
];

const CANDIDATE_EDGES: &[(VarId, VarId, u8)] = &[
    (VarId::PerceptionSalience, VarId::AttentionGain, 0),
    (VarId::PerceptionNovelty, VarId::AttentionGain, 0),
    (VarId::AttentionGain, VarId::LearningRate, 0),
    (VarId::LearningRate, VarId::AttentionGain, 1),
    (VarId::AttentionGain, VarId::ReplayPressure, 1),
    (VarId::ReplayPressure, VarId::SleepDrive, 1),
    (VarId::SleepDrive, VarId::ReplayPressure, 2),
    (VarId::ReplayPressure, VarId::NcdeEnergy, 1),
    (VarId::NcdeEnergy, VarId::CoherencePlv, 1),
    (VarId::CoherencePlv, VarId::PhiProxy, 0),
    (VarId::PhiProxy, VarId::AttentionGain, 2),
    (VarId::Risk, VarId::OutputSuppression, 0),
    (VarId::Risk, VarId::ReplayPressure, 1),
    (VarId::Drift, VarId::ReplayPressure, 1),
    (VarId::Surprise, VarId::AttentionGain, 0),
    (VarId::Surprise, VarId::ReplayPressure, 1),
    (VarId::PerceptionNovelty, VarId::Surprise, 1),
    (VarId::PerceptionSalience, VarId::Surprise, 1),
    (VarId::ReplayPressure, VarId::Surprise, 2),
    (VarId::AttentionGain, VarId::Risk, 2),
    (VarId::Risk, VarId::AttentionGain, 1),
    (VarId::Drift, VarId::Risk, 1),
    (VarId::CoherencePlv, VarId::Risk, 2),
    (VarId::PhiProxy, VarId::Risk, 2),
];

fn default_nodes() -> Vec<VarId> {
    let mut nodes = Vec::new();
    for var in OBSERVED_VARS {
        nodes.push(var);
    }
    nodes.push(VarId::OutputSuppression);
    nodes.truncate(MAX_NODES);
    nodes
}

fn default_edge_scores() -> Vec<CausalEdge> {
    CANDIDATE_EDGES
        .iter()
        .map(|(from, to, lag)| CausalEdge::new(*from, *to, 0, *lag))
        .collect()
}

fn observed_values(inp: &CdeInputs) -> [u16; 12] {
    [
        inp.ssm_salience,
        inp.ssm_novelty,
        inp.attention_gain,
        inp.learning_rate,
        inp.replay_pressure,
        inp.sleep_drive,
        inp.ncde_energy,
        inp.coherence_plv,
        inp.phi_proxy,
        inp.risk,
        inp.drift,
        inp.surprise,
    ]
}

fn compute_deltas(current: [u16; 12], prev: [u16; 12]) -> [i16; 12] {
    let mut out = [0i16; 12];
    for (idx, value) in current.iter().enumerate() {
        let curr = center_shift(*value);
        let prev = center_shift(prev[idx]);
        out[idx] = curr.saturating_sub(prev);
    }
    out
}

fn center_shift(value: u16) -> i16 {
    let shifted = i32::from(value).saturating_sub(CENTER_VALUE);
    shifted.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

fn edge_proxy(edge: &CausalEdge, history: &[[i16; 12]]) -> i16 {
    let from_delta = delta_for_var(edge.from, history, edge.lag as usize);
    let to_delta = delta_for_var(edge.to, history, 0);
    if from_delta == 0 || to_delta == 0 {
        return 0;
    }
    let sign = from_delta.signum() * to_delta.signum();
    let magnitude = from_delta.abs().min(to_delta.abs());
    sign.saturating_mul(magnitude)
}

fn delta_for_var(var: VarId, history: &[[i16; 12]], lag: usize) -> i16 {
    let Some(idx) = observed_index(var) else {
        return 0;
    };
    history.get(lag).map(|frame| frame[idx]).unwrap_or(0)
}

fn observed_index(var: VarId) -> Option<usize> {
    OBSERVED_VARS.iter().position(|item| *item == var)
}

fn has_observed_delta(var: VarId) -> bool {
    observed_index(var).is_some()
}

fn update_edge_score(edge: &CausalEdge, proxy: i16, dag: &CausalDag) -> CausalEdge {
    let mut score = i32::from(edge.score);
    let decay = score / DECAY_DIV;
    let delta = i32::from(proxy) / PROXY_SCALE;
    if delta > 0 && would_create_cycle(dag, edge.from, edge.to) {
        score -= decay;
    } else {
        score = score - decay + delta;
    }
    let score = score.clamp(i32::from(MIN_SCORE), i32::from(MAX_SCORE)) as i16;
    CausalEdge::new(edge.from, edge.to, score, edge.lag)
}

fn would_create_cycle(dag: &CausalDag, from: VarId, to: VarId) -> bool {
    if from == to {
        return true;
    }
    let mut stack = Vec::new();
    let mut seen = Vec::new();
    stack.push(to);
    while let Some(node) = stack.pop() {
        if node == from {
            return true;
        }
        if seen.contains(&node) {
            continue;
        }
        seen.push(node);
        for edge in &dag.edges {
            if edge.from == node {
                stack.push(edge.to);
            }
        }
    }
    false
}

fn select_dag_edges(edges: &[CausalEdge]) -> Vec<CausalEdge> {
    let mut candidates = edges
        .iter()
        .filter(|edge| edge.score != 0)
        .cloned()
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| {
        b.score
            .abs()
            .cmp(&a.score.abs())
            .then_with(|| a.from.to_u16().cmp(&b.from.to_u16()))
            .then_with(|| a.to.to_u16().cmp(&b.to.to_u16()))
            .then_with(|| a.lag.cmp(&b.lag))
    });
    let mut selected = Vec::new();
    for edge in candidates {
        if selected.len() >= MAX_EDGES {
            break;
        }
        if would_create_cycle(
            &CausalDag::new(Vec::new(), selected.clone()),
            edge.from,
            edge.to,
        ) {
            continue;
        }
        selected.push(edge);
    }
    selected
}

fn select_top_edges(edges: &[CausalEdge]) -> Vec<CausalEdge> {
    let mut sorted = edges.to_vec();
    sorted.sort_by(|a, b| {
        b.score
            .abs()
            .cmp(&a.score.abs())
            .then_with(|| a.from.to_u16().cmp(&b.from.to_u16()))
            .then_with(|| a.to.to_u16().cmp(&b.to.to_u16()))
            .then_with(|| a.lag.cmp(&b.lag))
    });
    sorted.truncate(MAX_TOP_EDGES);
    sorted
}

fn evaluate_intervention(pending: &PendingIntervention, history: &[[i16; 12]]) -> bool {
    let delta = delta_for_var(pending.target, history, 0);
    if delta == 0 || pending.expected_sign == 0 {
        return false;
    }
    delta.signum() == pending.expected_sign
}

fn derive_intervention_delta(seed_commit: Digest32, var: VarId) -> i16 {
    let mut hasher = Hasher::new();
    hasher.update(DELTA_DOMAIN);
    hasher.update(seed_commit.as_bytes());
    hasher.update(&var.to_u16().to_be_bytes());
    let bytes = hasher.finalize();
    let raw = bytes.as_bytes();
    let magnitude = 200 + (u16::from(raw[0]) % 600);
    let sign = if raw[1].is_multiple_of(2) { 1 } else { -1 };
    (sign as i16).saturating_mul(magnitude as i16)
}

fn build_spikes(
    inp: &CdeInputs,
    edges: &[CausalEdge],
    summary_commit: Digest32,
) -> Vec<SpikeEvent> {
    let mut spikes = Vec::new();
    for edge in edges {
        if edge.score.abs() < SCORE_SPIKE_THRESHOLD {
            continue;
        }
        if spikes.len() >= MAX_TOP_EDGES {
            break;
        }
        let ttfs = score_to_ttfs(edge.score);
        let payload_commit = spike_payload_commit(edge.commit, summary_commit);
        spikes.push(SpikeEvent::new(
            inp.cycle_id,
            SpikeKind::CausalLink,
            SpikeModuleId::Cde,
            SpikeModuleId::Nsr,
            ttfs,
            inp.phase_commit,
            payload_commit,
        ));
    }
    spikes
}

fn score_to_ttfs(score: i16) -> u16 {
    let magnitude = score.abs().min(MAX_SCORE) as u16;
    let scaled = (magnitude / 60).min(TTFS_MAX - TTFS_MIN);
    TTFS_MAX.saturating_sub(scaled).max(TTFS_MIN)
}

fn digest_edge(from: VarId, to: VarId, score: i16, lag: u8) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(EDGE_DOMAIN);
    hasher.update(&from.to_u16().to_be_bytes());
    hasher.update(&to.to_u16().to_be_bytes());
    hasher.update(&score.to_be_bytes());
    hasher.update(&[lag]);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_dag(nodes: &[VarId], edges: &[CausalEdge]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(DAG_DOMAIN);
    hasher.update(&(nodes.len() as u16).to_be_bytes());
    let mut sorted_nodes = nodes.to_vec();
    sorted_nodes.sort_by_key(|node| node.to_u16());
    for node in sorted_nodes {
        hasher.update(&node.to_u16().to_be_bytes());
    }
    let mut sorted_edges = edges.to_vec();
    sorted_edges.sort_by(|a, b| {
        a.from
            .to_u16()
            .cmp(&b.from.to_u16())
            .then_with(|| a.to.to_u16().cmp(&b.to.to_u16()))
            .then_with(|| a.lag.cmp(&b.lag))
    });
    hasher.update(&(sorted_edges.len() as u16).to_be_bytes());
    for edge in sorted_edges {
        hasher.update(edge.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_intervention(cycle_id: u64, var: VarId, delta: i16, basis_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(INTERVENTION_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(&var.to_u16().to_be_bytes());
    hasher.update(&delta.to_be_bytes());
    hasher.update(basis_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_inputs(inputs: &CdeInputs) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(INPUT_DOMAIN);
    hasher.update(&inputs.cycle_id.to_be_bytes());
    hasher.update(inputs.phase_commit.as_bytes());
    hasher.update(&inputs.ssm_salience.to_be_bytes());
    hasher.update(&inputs.ssm_novelty.to_be_bytes());
    hasher.update(&inputs.attention_gain.to_be_bytes());
    hasher.update(&inputs.learning_rate.to_be_bytes());
    hasher.update(&inputs.replay_pressure.to_be_bytes());
    hasher.update(&inputs.sleep_drive.to_be_bytes());
    hasher.update(&inputs.ncde_energy.to_be_bytes());
    hasher.update(&inputs.coherence_plv.to_be_bytes());
    hasher.update(&inputs.phi_proxy.to_be_bytes());
    hasher.update(&inputs.risk.to_be_bytes());
    hasher.update(&inputs.drift.to_be_bytes());
    hasher.update(&inputs.surprise.to_be_bytes());
    hasher.update(&[inputs.sleep_active as u8]);
    hasher.update(&[inputs.replay_active as u8]);
    hasher.update(inputs.spike_accepted_root.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_summary(
    dag_commit: Digest32,
    edges: &[CausalEdge],
    intervention: Option<&Intervention>,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(SUMMARY_DOMAIN);
    hasher.update(dag_commit.as_bytes());
    hasher.update(&(edges.len() as u16).to_be_bytes());
    for edge in edges {
        hasher.update(edge.commit.as_bytes());
    }
    match intervention {
        Some(intervention) => {
            hasher.update(&[1]);
            hasher.update(intervention.commit.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_outputs(
    cycle_id: u64,
    dag_commit: Digest32,
    summary_commit: Digest32,
    edges: &[CausalEdge],
    intervention: Option<&Intervention>,
    spikes: &[SpikeEvent],
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(OUTPUT_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(dag_commit.as_bytes());
    hasher.update(summary_commit.as_bytes());
    hasher.update(&(edges.len() as u16).to_be_bytes());
    for edge in edges {
        hasher.update(edge.commit.as_bytes());
    }
    match intervention {
        Some(intervention) => {
            hasher.update(&[1]);
            hasher.update(intervention.commit.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    hasher.update(&(spikes.len() as u16).to_be_bytes());
    for spike in spikes {
        hasher.update(spike.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_core(dag_commit: Digest32, input_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(CORE_DOMAIN);
    hasher.update(dag_commit.as_bytes());
    hasher.update(input_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn spike_payload_commit(edge_commit: Digest32, summary_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(SPIKE_PAYLOAD_DOMAIN);
    hasher.update(edge_commit.as_bytes());
    hasher.update(summary_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_inputs(cycle_id: u64) -> CdeInputs {
        CdeInputs::new(
            cycle_id,
            Digest32::new([1u8; 32]),
            5_000,
            4_800,
            5_200,
            4_900,
            4_500,
            3_800,
            5_500,
            6_200,
            5_100,
            3_000,
            3_200,
            4_100,
            true,
            false,
            Digest32::new([9u8; 32]),
        )
    }

    #[test]
    fn cde_is_deterministic_for_same_inputs() {
        let mut core_a = CdeCore::new();
        let mut core_b = CdeCore::new();
        let inputs = base_inputs(1);

        let out_a = core_a.tick(&inputs);
        let out_b = core_b.tick(&inputs);

        assert_eq!(out_a.dag_commit, out_b.dag_commit);
        assert_eq!(out_a.commit, out_b.commit);
        assert_eq!(out_a.summary_commit, out_b.summary_commit);
    }

    #[test]
    fn cycle_edges_are_rejected() {
        let mut core = CdeCore::new();
        let edge_ab = CausalEdge::new(VarId::Risk, VarId::Drift, 5_000, 0);
        let edge_ba = CausalEdge::new(VarId::Drift, VarId::Risk, 5_000, 0);
        core.dag = CausalDag::new(core.dag.nodes.clone(), vec![edge_ab.clone()]);
        let mut edges = vec![edge_ab, edge_ba];
        let selected = select_dag_edges(&edges);
        assert!(selected.len() <= 1);
        edges[0].score = 0;
        let selected = select_dag_edges(&edges);
        assert!(selected.len() <= 1);
    }

    #[test]
    fn intervention_feedback_adjusts_score() {
        let mut core = CdeCore::new();
        core.edge_scores = vec![CausalEdge::new(VarId::Risk, VarId::Drift, 2_500, 0)];
        core.dag = CausalDag::new(core.dag.nodes.clone(), core.edge_scores.clone());
        core.pending_intervention = Some(PendingIntervention {
            edge_key: EdgeKey {
                from: VarId::Risk,
                to: VarId::Drift,
                lag: 0,
            },
            expected_sign: 1,
            target: VarId::Drift,
        });
        core.delta_history = vec![[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0]];
        let out = core.tick(&base_inputs(2));
        let boosted = out
            .top_edges
            .iter()
            .find(|edge| edge.from == VarId::Risk && edge.to == VarId::Drift)
            .map(|edge| edge.score)
            .unwrap_or(0);
        assert!(boosted >= 2_500);
    }

    #[test]
    fn high_score_emits_spike() {
        let mut core = CdeCore::new();
        core.edge_scores = vec![CausalEdge::new(
            VarId::Risk,
            VarId::Drift,
            SCORE_SPIKE_THRESHOLD + 3_000,
            0,
        )];
        core.dag = CausalDag::new(core.dag.nodes.clone(), core.edge_scores.clone());
        let outputs = core.tick(&base_inputs(3));
        assert!(!outputs.causal_link_spikes.is_empty());
    }
}
