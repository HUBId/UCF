#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_influence::NodeId;
use ucf_spikebus::SpikeKind;
use ucf_types::Digest32;

pub const CONF_MIN: u16 = 0;
pub const CONF_MAX: u16 = 10_000;
pub const CONF_SPIKE: u16 = 8_000;
pub const COH_MIN: u16 = 3_000;
pub const NSR_ATOM_MIN: u16 = 7_500;

const MAX_LAG: u8 = 8;
const EVIDENCE_MAX: i32 = 12;
const EDGE_DECAY_BASE: i32 = 2;
const SIM_CONF_MIN: u16 = 7_000;
const SIM_STEPS: usize = 3;
const MAX_TOP_EDGES: usize = 10;
const MAX_COUNTERFACTUAL_DELTAS: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CdeNodeId {
    Surprise,
    Drift,
    Risk,
    Coherence,
    Phi,
    AttentionGain,
    ReplayPressure,
    OutputSuppression,
    BlueBrainArousal,
    Unknown(u16),
}

impl CdeNodeId {
    pub fn to_u16(self) -> u16 {
        match self {
            Self::Surprise => 1,
            Self::Drift => 2,
            Self::Risk => 3,
            Self::Coherence => 4,
            Self::Phi => 5,
            Self::AttentionGain => 6,
            Self::ReplayPressure => 7,
            Self::OutputSuppression => 8,
            Self::BlueBrainArousal => 9,
            Self::Unknown(value) => value,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CausalEdge {
    pub from: CdeNodeId,
    pub to: CdeNodeId,
    pub conf: u16,
    pub lag: u8,
    pub last_update: u64,
    pub commit: Digest32,
}

impl CausalEdge {
    pub fn new(from: CdeNodeId, to: CdeNodeId, conf: u16, lag: u8, last_update: u64) -> Self {
        let conf = conf.clamp(CONF_MIN, CONF_MAX);
        let lag = lag.min(MAX_LAG);
        let commit = digest_edge(from, to, conf, lag, last_update);
        Self {
            from,
            to,
            conf,
            lag,
            last_update,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HypothesisGraph {
    pub nodes: Vec<CdeNodeId>,
    pub edges: Vec<CausalEdge>,
    pub root_commit: Digest32,
}

impl HypothesisGraph {
    pub fn new(nodes: Vec<CdeNodeId>, edges: Vec<CausalEdge>) -> Self {
        let root_commit = digest_graph(&nodes, &edges);
        Self {
            nodes,
            edges,
            root_commit,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InterventionKind {
    DoIncrease,
    DoDecrease,
    ClampLow,
    ClampHigh,
    Unknown(u16),
}

impl InterventionKind {
    pub fn to_u16(self) -> u16 {
        match self {
            Self::DoIncrease => 1,
            Self::DoDecrease => 2,
            Self::ClampLow => 3,
            Self::ClampHigh => 4,
            Self::Unknown(value) => value,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Intervention {
    pub target: CdeNodeId,
    pub kind: InterventionKind,
    pub strength: u16,
    pub commit: Digest32,
}

impl Intervention {
    pub fn new(target: CdeNodeId, kind: InterventionKind, strength: u16) -> Self {
        let strength = strength.min(CONF_MAX);
        let commit = digest_intervention(target, kind, strength);
        Self {
            target,
            kind,
            strength,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CdeInputs {
    pub cycle_id: u64,
    pub phase_commit: Digest32,
    pub global_phase: u16,
    pub coherence_plv: u16,
    pub phi_proxy: u16,
    pub spike_root: Digest32,
    pub spike_counts: Vec<(SpikeKind, u16)>,
    pub influence_commit: Digest32,
    pub influence_node_in: Vec<(NodeId, i16)>,
    pub ssm_salience: u16,
    pub ncde_energy: u16,
    pub drift: u16,
    pub surprise: u16,
    pub risk: u16,
    pub commit: Digest32,
}

impl CdeInputs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cycle_id: u64,
        phase_commit: Digest32,
        global_phase: u16,
        coherence_plv: u16,
        phi_proxy: u16,
        spike_root: Digest32,
        spike_counts: Vec<(SpikeKind, u16)>,
        influence_commit: Digest32,
        influence_node_in: Vec<(NodeId, i16)>,
        ssm_salience: u16,
        ncde_energy: u16,
        drift: u16,
        surprise: u16,
        risk: u16,
    ) -> Self {
        let mut inputs = Self {
            cycle_id,
            phase_commit,
            global_phase,
            coherence_plv,
            phi_proxy,
            spike_root,
            spike_counts,
            influence_commit,
            influence_node_in,
            ssm_salience,
            ncde_energy,
            drift,
            surprise,
            risk,
            commit: Digest32::new([0u8; 32]),
        };
        inputs.commit = digest_inputs(&inputs);
        inputs
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CdeOutputs {
    pub cycle_id: u64,
    pub graph_commit: Digest32,
    pub top_edges: Vec<(CdeNodeId, CdeNodeId, u16, u8)>,
    pub interventions: Vec<Intervention>,
    pub counterfactual_delta: Vec<(CdeNodeId, i16)>,
    pub emit_spikes: bool,
    pub commit: Digest32,
}

impl CdeOutputs {
    pub fn new(
        cycle_id: u64,
        graph_commit: Digest32,
        top_edges: Vec<(CdeNodeId, CdeNodeId, u16, u8)>,
        interventions: Vec<Intervention>,
        counterfactual_delta: Vec<(CdeNodeId, i16)>,
        emit_spikes: bool,
    ) -> Self {
        let commit = digest_outputs(
            cycle_id,
            graph_commit,
            &top_edges,
            &interventions,
            &counterfactual_delta,
            emit_spikes,
        );
        Self {
            cycle_id,
            graph_commit,
            top_edges,
            interventions,
            counterfactual_delta,
            emit_spikes,
            commit,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CdeEngine {
    pub graph: HypothesisGraph,
    pub commit: Digest32,
}

impl Default for CdeEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl CdeEngine {
    pub fn new() -> Self {
        let graph = default_graph();
        let commit = digest_engine(graph.root_commit, Digest32::new([0u8; 32]));
        Self { graph, commit }
    }

    pub fn tick(&mut self, inp: &CdeInputs) -> CdeOutputs {
        let mut edges = Vec::with_capacity(self.graph.edges.len());
        for edge in &self.graph.edges {
            edges.push(update_edge(edge, inp));
        }
        let graph = HypothesisGraph::new(self.graph.nodes.clone(), edges);
        let top_edges = select_top_edges(&graph.edges);
        let interventions = select_interventions(inp, &graph, &top_edges);
        let counterfactual_delta = simulate_counterfactuals(inp, &graph, &interventions);
        let emit_spikes = should_emit_spikes(inp, &top_edges);
        let outputs = CdeOutputs::new(
            inp.cycle_id,
            graph.root_commit,
            top_edges,
            interventions,
            counterfactual_delta,
            emit_spikes,
        );
        self.graph = graph;
        self.commit = digest_engine(self.graph.root_commit, inp.commit);
        outputs
    }
}

pub fn edge_key(from: CdeNodeId, to: CdeNodeId) -> u16 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.cde.edge.key.v1");
    hasher.update(&from.to_u16().to_be_bytes());
    hasher.update(&to.to_u16().to_be_bytes());
    let bytes = hasher.finalize();
    let raw = bytes.as_bytes();
    u16::from_be_bytes([raw[0], raw[1]])
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CounterfactualResult {
    pub predicted_delta: i16,
    pub confidence: u16,
    pub commit: Digest32,
}

impl CounterfactualResult {
    pub fn new(predicted_delta: i16, confidence: u16, seed: Digest32) -> Self {
        let commit = digest_counterfactual(predicted_delta, confidence, seed);
        Self {
            predicted_delta,
            confidence,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CausalReport {
    pub dag_commit: Digest32,
    pub counterfactuals: Vec<CounterfactualResult>,
    pub flags: u16,
    pub commit: Digest32,
}

impl CausalReport {
    pub fn new(
        dag_commit: Digest32,
        counterfactuals: Vec<CounterfactualResult>,
        flags: u16,
    ) -> Self {
        let commit = digest_causal_report(dag_commit, &counterfactuals, flags);
        Self {
            dag_commit,
            counterfactuals,
            flags,
            commit,
        }
    }
}

fn default_graph() -> HypothesisGraph {
    let nodes = vec![
        CdeNodeId::Surprise,
        CdeNodeId::Drift,
        CdeNodeId::Risk,
        CdeNodeId::Coherence,
        CdeNodeId::Phi,
        CdeNodeId::AttentionGain,
        CdeNodeId::ReplayPressure,
        CdeNodeId::OutputSuppression,
    ];
    let edges = vec![
        CausalEdge::new(CdeNodeId::Surprise, CdeNodeId::AttentionGain, 3_200, 0, 0),
        CausalEdge::new(CdeNodeId::Surprise, CdeNodeId::ReplayPressure, 2_800, 1, 0),
        CausalEdge::new(
            CdeNodeId::Surprise,
            CdeNodeId::OutputSuppression,
            2_400,
            1,
            0,
        ),
        CausalEdge::new(CdeNodeId::Drift, CdeNodeId::ReplayPressure, 3_400, 1, 0),
        CausalEdge::new(CdeNodeId::Drift, CdeNodeId::Risk, 2_600, 2, 0),
        CausalEdge::new(CdeNodeId::Risk, CdeNodeId::OutputSuppression, 3_500, 0, 0),
        CausalEdge::new(CdeNodeId::Risk, CdeNodeId::AttentionGain, 2_200, 1, 0),
        CausalEdge::new(CdeNodeId::Coherence, CdeNodeId::Phi, 3_300, 0, 0),
        CausalEdge::new(CdeNodeId::Phi, CdeNodeId::ReplayPressure, 3_100, 2, 0),
        CausalEdge::new(CdeNodeId::Phi, CdeNodeId::AttentionGain, 2_700, 1, 0),
        CausalEdge::new(CdeNodeId::AttentionGain, CdeNodeId::Drift, 3_000, 1, 0),
        CausalEdge::new(CdeNodeId::AttentionGain, CdeNodeId::Coherence, 2_500, 1, 0),
        CausalEdge::new(
            CdeNodeId::ReplayPressure,
            CdeNodeId::OutputSuppression,
            2_900,
            1,
            0,
        ),
        CausalEdge::new(CdeNodeId::ReplayPressure, CdeNodeId::Drift, 2_700, 2, 0),
        CausalEdge::new(
            CdeNodeId::OutputSuppression,
            CdeNodeId::AttentionGain,
            2_300,
            1,
            0,
        ),
        CausalEdge::new(CdeNodeId::Coherence, CdeNodeId::AttentionGain, 2_600, 0, 0),
        CausalEdge::new(CdeNodeId::Risk, CdeNodeId::ReplayPressure, 2_800, 1, 0),
        CausalEdge::new(CdeNodeId::Drift, CdeNodeId::AttentionGain, 2_500, 1, 0),
    ];
    HypothesisGraph::new(nodes, edges)
}

fn update_edge(edge: &CausalEdge, inp: &CdeInputs) -> CausalEdge {
    let evidence = pair_evidence(edge.from, edge.to, inp);
    let seed = hash_edge_seed(inp.commit, edge.commit);
    let alpha = 1 + (seed % 4) as i32;
    let decay = EDGE_DECAY_BASE + (seed % 2) as i32;
    let updated = i32::from(edge.conf)
        .saturating_add(alpha.saturating_mul(evidence))
        .saturating_sub(decay)
        .clamp(i32::from(CONF_MIN), i32::from(CONF_MAX)) as u16;
    CausalEdge::new(edge.from, edge.to, updated, edge.lag, inp.cycle_id)
}

fn pair_evidence(from: CdeNodeId, to: CdeNodeId, inp: &CdeInputs) -> i32 {
    let a = node_value(from, inp);
    let b = node_value(to, inp);
    let influence_a = influence_for_node(from, &inp.influence_node_in);
    let influence_b = influence_for_node(to, &inp.influence_node_in);
    let mut ev = 0i32;
    let a_high = a >= 6_500;
    let a_low = a <= 3_500;
    let b_high = b >= 6_000;
    let b_low = b <= 4_000;
    if a_high && b_high {
        ev += 5;
    }
    if a_high && b_low {
        ev -= 5;
    }
    if a_low && b_high {
        ev -= 3;
    }
    if a_low && b_low {
        ev += 1;
    }
    ev += i32::from(influence_a.signum());
    ev += i32::from(influence_b.signum());
    ev += spike_bias(inp);
    ev.clamp(-EVIDENCE_MAX, EVIDENCE_MAX)
}

fn spike_bias(inp: &CdeInputs) -> i32 {
    let mut bias = 0i32;
    let threat = spike_count(inp, SpikeKind::Threat);
    let novelty = spike_count(inp, SpikeKind::Novelty);
    let causal = spike_count(inp, SpikeKind::CausalLink);
    if threat > 0 {
        bias += 2;
    }
    if novelty > 0 {
        bias += 1;
    }
    if causal > 0 {
        bias += 2;
    }
    bias
}

fn spike_count(inp: &CdeInputs, kind: SpikeKind) -> u16 {
    inp.spike_counts
        .iter()
        .find_map(|(k, count)| (*k == kind).then_some(*count))
        .unwrap_or(0)
}

fn node_value(node: CdeNodeId, inp: &CdeInputs) -> i32 {
    match node {
        CdeNodeId::Surprise => i32::from(inp.surprise),
        CdeNodeId::Drift => {
            blend_metric(inp.drift, influence_for_node(node, &inp.influence_node_in))
        }
        CdeNodeId::Risk => blend_metric(inp.risk, influence_for_node(node, &inp.influence_node_in)),
        CdeNodeId::Coherence => i32::from(inp.coherence_plv),
        CdeNodeId::Phi => i32::from(inp.phi_proxy),
        CdeNodeId::AttentionGain => {
            influence_to_metric(influence_for_node(node, &inp.influence_node_in))
        }
        CdeNodeId::ReplayPressure => {
            influence_to_metric(influence_for_node(node, &inp.influence_node_in))
        }
        CdeNodeId::OutputSuppression => {
            influence_to_metric(influence_for_node(node, &inp.influence_node_in))
        }
        CdeNodeId::BlueBrainArousal => blend_metric(
            inp.ssm_salience,
            influence_for_node(node, &inp.influence_node_in),
        ),
        CdeNodeId::Unknown(_) => 0,
    }
    .min(i32::from(CONF_MAX))
}

fn influence_for_node(node: CdeNodeId, influence: &[(NodeId, i16)]) -> i16 {
    // Minimal mapping from InfluenceGraph nodes into CDE nodes.
    match node {
        CdeNodeId::AttentionGain => influence_value(influence, NodeId::Attention),
        CdeNodeId::ReplayPressure => influence_value(influence, NodeId::Replay),
        CdeNodeId::OutputSuppression => influence_value(influence, NodeId::Output),
        CdeNodeId::Drift => {
            let structure = influence_value(influence, NodeId::Structure);
            let geist = influence_value(influence, NodeId::Geist);
            structure.saturating_add(geist / 2)
        }
        CdeNodeId::Risk => influence_value(influence, NodeId::Geist),
        CdeNodeId::BlueBrainArousal => influence_value(influence, NodeId::BlueBrain),
        _ => 0,
    }
}

fn influence_value(influence: &[(NodeId, i16)], node: NodeId) -> i16 {
    influence
        .iter()
        .find_map(|(id, value)| (*id == node).then_some(*value))
        .unwrap_or(0)
}

fn influence_to_metric(value: i16) -> i32 {
    let clamped = value.clamp(-10_000, 10_000) as i32;
    (clamped + 10_000) / 2
}

fn blend_metric(base: u16, influence: i16) -> i32 {
    let influence_metric = influence_to_metric(influence);
    let base = i32::from(base);
    (base.saturating_add(influence_metric)) / 2
}

fn select_top_edges(edges: &[CausalEdge]) -> Vec<(CdeNodeId, CdeNodeId, u16, u8)> {
    let mut sorted = edges.to_vec();
    sorted.sort_by(|a, b| {
        b.conf
            .cmp(&a.conf)
            .then_with(|| a.from.to_u16().cmp(&b.from.to_u16()))
            .then_with(|| a.to.to_u16().cmp(&b.to.to_u16()))
    });
    sorted
        .into_iter()
        .take(MAX_TOP_EDGES)
        .map(|edge| (edge.from, edge.to, edge.conf, edge.lag))
        .collect()
}

fn select_interventions(
    inp: &CdeInputs,
    graph: &HypothesisGraph,
    top_edges: &[(CdeNodeId, CdeNodeId, u16, u8)],
) -> Vec<Intervention> {
    let mut interventions = Vec::new();
    if inp.surprise >= 7_000 {
        let strength = (inp.surprise / 2).clamp(1_000, CONF_MAX);
        interventions.push(Intervention::new(
            CdeNodeId::Surprise,
            InterventionKind::DoDecrease,
            strength,
        ));
    }
    if inp.drift >= 6_500 {
        let strength = (inp.drift / 2).clamp(1_000, CONF_MAX);
        interventions.push(Intervention::new(
            CdeNodeId::Drift,
            InterventionKind::DoDecrease,
            strength,
        ));
    }
    if inp.risk >= 6_500 {
        let strength = (inp.risk / 2).clamp(1_500, CONF_MAX);
        interventions.push(Intervention::new(
            CdeNodeId::OutputSuppression,
            InterventionKind::ClampHigh,
            strength,
        ));
    }

    if interventions.len() < 3 {
        if let Some((from, _, conf, _)) = top_edges.first() {
            if matches!(
                *from,
                CdeNodeId::Surprise | CdeNodeId::Drift | CdeNodeId::Risk
            ) && *conf >= 6_000
            {
                if let Some(edge) = graph.edges.iter().find(|edge| edge.from == *from) {
                    interventions.push(Intervention::new(
                        edge.to,
                        InterventionKind::DoDecrease,
                        (edge.conf / 2).clamp(800, CONF_MAX),
                    ));
                }
            }
        }
    }
    interventions.truncate(3);
    interventions
}

fn simulate_counterfactuals(
    _inp: &CdeInputs,
    graph: &HypothesisGraph,
    interventions: &[Intervention],
) -> Vec<(CdeNodeId, i16)> {
    if interventions.is_empty() {
        return Vec::new();
    }
    let mut deltas: Vec<i32> = graph.nodes.iter().map(|_| 0i32).collect();

    for intervention in interventions {
        if let Some(idx) = graph
            .nodes
            .iter()
            .position(|node| *node == intervention.target)
        {
            let strength = i32::from(intervention.strength) / 20;
            let delta = match intervention.kind {
                InterventionKind::DoIncrease => strength,
                InterventionKind::DoDecrease => -strength,
                InterventionKind::ClampLow => -strength / 2,
                InterventionKind::ClampHigh => strength / 2,
                InterventionKind::Unknown(_) => 0,
            };
            deltas[idx] = deltas[idx].saturating_add(delta);
        }
    }

    for _ in 0..SIM_STEPS {
        let mut next = deltas.clone();
        for edge in &graph.edges {
            if edge.conf < SIM_CONF_MIN {
                continue;
            }
            let from_idx = graph
                .nodes
                .iter()
                .position(|node| *node == edge.from)
                .unwrap_or(0);
            let to_idx = graph
                .nodes
                .iter()
                .position(|node| *node == edge.to)
                .unwrap_or(0);
            let lag = i32::from(edge.lag.max(1));
            let flow = (deltas[from_idx] * i32::from(edge.conf)) / 10_000;
            let attenuated = flow / lag;
            next[to_idx] = next[to_idx].saturating_add(attenuated / 2);
        }
        deltas = next;
    }

    let mut output = Vec::new();
    for (idx, node) in graph.nodes.iter().enumerate() {
        let delta = deltas[idx].clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
        if delta != 0 {
            output.push((*node, delta));
        }
    }
    output.truncate(MAX_COUNTERFACTUAL_DELTAS);
    if output.is_empty() {
        output.push((CdeNodeId::Surprise, 0));
    }
    output
}

fn should_emit_spikes(inp: &CdeInputs, edges: &[(CdeNodeId, CdeNodeId, u16, u8)]) -> bool {
    if inp.coherence_plv < COH_MIN {
        return false;
    }
    edges.iter().any(|(from, to, conf, _)| {
        *conf >= CONF_SPIKE
            && ((*from == CdeNodeId::Risk && *to == CdeNodeId::OutputSuppression)
                || (*from == CdeNodeId::Drift && *to == CdeNodeId::ReplayPressure))
    })
}

fn hash_edge_seed(input_commit: Digest32, edge_commit: Digest32) -> u64 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.cde.edge.seed.v1");
    hasher.update(input_commit.as_bytes());
    hasher.update(edge_commit.as_bytes());
    let bytes = hasher.finalize();
    let raw = bytes.as_bytes();
    u64::from_be_bytes(raw[0..8].try_into().expect("seed"))
}

fn digest_edge(from: CdeNodeId, to: CdeNodeId, conf: u16, lag: u8, last_update: u64) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.cde.edge.v2");
    hasher.update(&from.to_u16().to_be_bytes());
    hasher.update(&to.to_u16().to_be_bytes());
    hasher.update(&conf.to_be_bytes());
    hasher.update(&[lag]);
    hasher.update(&last_update.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_graph(nodes: &[CdeNodeId], edges: &[CausalEdge]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.cde.graph.v2");
    hasher.update(&u64::try_from(nodes.len()).unwrap_or(0).to_be_bytes());
    for node in nodes {
        hasher.update(&node.to_u16().to_be_bytes());
    }
    hasher.update(&u64::try_from(edges.len()).unwrap_or(0).to_be_bytes());
    for edge in edges {
        hasher.update(edge.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_intervention(target: CdeNodeId, kind: InterventionKind, strength: u16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.cde.intervention.v2");
    hasher.update(&target.to_u16().to_be_bytes());
    hasher.update(&kind.to_u16().to_be_bytes());
    hasher.update(&strength.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_counterfactual(predicted_delta: i16, confidence: u16, seed: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.cde.counterfactual.v1");
    hasher.update(&predicted_delta.to_be_bytes());
    hasher.update(&confidence.to_be_bytes());
    hasher.update(seed.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_causal_report(
    dag_commit: Digest32,
    counterfactuals: &[CounterfactualResult],
    flags: u16,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.cde.report.v2");
    hasher.update(dag_commit.as_bytes());
    hasher.update(&flags.to_be_bytes());
    hasher.update(
        &u64::try_from(counterfactuals.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    for counterfactual in counterfactuals {
        hasher.update(counterfactual.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_inputs(inputs: &CdeInputs) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.cde.inputs.v1");
    hasher.update(&inputs.cycle_id.to_be_bytes());
    hasher.update(inputs.phase_commit.as_bytes());
    hasher.update(&inputs.global_phase.to_be_bytes());
    hasher.update(&inputs.coherence_plv.to_be_bytes());
    hasher.update(&inputs.phi_proxy.to_be_bytes());
    hasher.update(inputs.spike_root.as_bytes());
    hasher.update(
        &u64::try_from(inputs.spike_counts.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    for (kind, count) in &inputs.spike_counts {
        hasher.update(&kind.as_u16().to_be_bytes());
        hasher.update(&count.to_be_bytes());
    }
    hasher.update(inputs.influence_commit.as_bytes());
    hasher.update(
        &u64::try_from(inputs.influence_node_in.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    for (node, value) in &inputs.influence_node_in {
        hasher.update(&node.to_u16().to_be_bytes());
        hasher.update(&value.to_be_bytes());
    }
    hasher.update(&inputs.ssm_salience.to_be_bytes());
    hasher.update(&inputs.ncde_energy.to_be_bytes());
    hasher.update(&inputs.drift.to_be_bytes());
    hasher.update(&inputs.surprise.to_be_bytes());
    hasher.update(&inputs.risk.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_outputs(
    cycle_id: u64,
    graph_commit: Digest32,
    top_edges: &[(CdeNodeId, CdeNodeId, u16, u8)],
    interventions: &[Intervention],
    counterfactual_delta: &[(CdeNodeId, i16)],
    emit_spikes: bool,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.cde.outputs.v1");
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(graph_commit.as_bytes());
    hasher.update(&u64::try_from(top_edges.len()).unwrap_or(0).to_be_bytes());
    for (from, to, conf, lag) in top_edges {
        hasher.update(&from.to_u16().to_be_bytes());
        hasher.update(&to.to_u16().to_be_bytes());
        hasher.update(&conf.to_be_bytes());
        hasher.update(&[*lag]);
    }
    hasher.update(
        &u64::try_from(interventions.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    for intervention in interventions {
        hasher.update(intervention.commit.as_bytes());
    }
    hasher.update(
        &u64::try_from(counterfactual_delta.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    for (node, delta) in counterfactual_delta {
        hasher.update(&node.to_u16().to_be_bytes());
        hasher.update(&delta.to_be_bytes());
    }
    hasher.update(&[emit_spikes as u8]);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_engine(graph_commit: Digest32, input_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.cde.engine.v1");
    hasher.update(graph_commit.as_bytes());
    hasher.update(input_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_inputs(seed: u8) -> CdeInputs {
        CdeInputs::new(
            1,
            Digest32::new([seed; 32]),
            200,
            4_500,
            4_000,
            Digest32::new([seed.wrapping_add(1); 32]),
            vec![(SpikeKind::Novelty, 1)],
            Digest32::new([seed.wrapping_add(2); 32]),
            vec![
                (NodeId::Attention, 6_000),
                (NodeId::Replay, 2_500),
                (NodeId::Output, 1_000),
            ],
            3_000,
            2_000,
            4_000,
            7_500,
            4_500,
        )
    }

    #[test]
    fn deterministic_outputs_for_same_inputs() {
        let mut engine_a = CdeEngine::new();
        let mut engine_b = CdeEngine::new();
        let inputs = base_inputs(7);

        let out_a = engine_a.tick(&inputs);
        let out_b = engine_b.tick(&inputs);

        assert_eq!(out_a.commit, out_b.commit);
        assert_eq!(engine_a.graph.root_commit, engine_b.graph.root_commit);
    }

    #[test]
    fn surprise_attention_alignment_increases_confidence() {
        let mut engine = CdeEngine::new();
        let mut inputs = base_inputs(9);
        inputs.surprise = 9_000;
        inputs.influence_node_in = vec![(NodeId::Attention, 9_000)];

        let before = engine
            .graph
            .edges
            .iter()
            .find(|edge| edge.from == CdeNodeId::Surprise && edge.to == CdeNodeId::AttentionGain)
            .map(|edge| edge.conf)
            .expect("edge");

        let _ = engine.tick(&inputs);
        let after = engine
            .graph
            .edges
            .iter()
            .find(|edge| edge.from == CdeNodeId::Surprise && edge.to == CdeNodeId::AttentionGain)
            .map(|edge| edge.conf)
            .expect("edge");

        assert!(after >= before);
    }

    #[test]
    fn interventions_are_bounded() {
        let mut engine = CdeEngine::new();
        let mut inputs = base_inputs(11);
        inputs.surprise = 9_000;
        inputs.drift = 8_500;
        inputs.risk = 9_000;

        let out = engine.tick(&inputs);

        assert!(out.interventions.len() <= 3);
        assert!(out
            .counterfactual_delta
            .iter()
            .all(|(_, delta)| delta.abs() <= i16::MAX));
    }

    #[test]
    fn emit_spikes_respects_coherence_threshold() {
        let mut engine = CdeEngine::new();
        if let Some(edge) =
            engine.graph.edges.iter_mut().find(|edge| {
                edge.from == CdeNodeId::Risk && edge.to == CdeNodeId::OutputSuppression
            })
        {
            edge.conf = CONF_SPIKE;
            edge.commit = digest_edge(edge.from, edge.to, edge.conf, edge.lag, edge.last_update);
        }
        let mut inputs = base_inputs(13);
        inputs.coherence_plv = COH_MIN + 100;
        let out_high = engine.tick(&inputs);
        assert!(out_high.emit_spikes);

        inputs.coherence_plv = COH_MIN.saturating_sub(100);
        let out_low = engine.tick(&inputs);
        assert!(!out_low.emit_spikes);
    }
}
