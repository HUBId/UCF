#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_spikebus::{ModuleId, Spike, SpikeKind};
use ucf_types::Digest32;

const INPUT_DOMAIN: &[u8] = b"ucf.cde.v1.inputs";
const EDGE_DOMAIN: &[u8] = b"ucf.cde.v1.edge";
const DAG_DOMAIN: &[u8] = b"ucf.cde.v1.dag";
const INTERVENTION_DOMAIN: &[u8] = b"ucf.cde.v1.intervention";
const GRAPH_EDGE_DOMAIN: &[u8] = b"ucf.cde.v1.graph.edge";
const GRAPH_DOMAIN: &[u8] = b"ucf.cde.v1.graph";
const QUERY_DOMAIN: &[u8] = b"ucf.cde.v1.query";
const QUERY_RESULT_DOMAIN: &[u8] = b"ucf.cde.v1.query.result";
const QUERY_PROOF_DOMAIN: &[u8] = b"ucf.cde.v1.query.proof";
const QUERY_EFFECT_DOMAIN: &[u8] = b"ucf.cde.v1.query.effect";
const QUERY_INTERVENTION_DOMAIN: &[u8] = b"ucf.cde.v1.query.intervention";
const OBSERVATION_KEY_DOMAIN: &[u8] = b"ucf.cde.v1.observation.key";
const OBSERVATION_RING_DOMAIN: &[u8] = b"ucf.cde.v1.observation.ring";
const OBSERVATION_INDEX_DOMAIN: &[u8] = b"ucf.cde.v1.observation.index";
const SURPRISE_BUCKET_DOMAIN: &[u8] = b"ucf.cde.v1.surprise.bucket";
const OUTPUT_DOMAIN: &[u8] = b"ucf.cde.v1.outputs";
const SUMMARY_DOMAIN: &[u8] = b"ucf.cde.v1.summary";
const CORE_DOMAIN: &[u8] = b"ucf.cde.v1.core";
const SPIKE_PAYLOAD_DOMAIN: &[u8] = b"ucf.cde.v1.spike.payload";
const DELTA_DOMAIN: &[u8] = b"ucf.cde.v1.delta";
const PARAM_DOMAIN: &[u8] = b"ucf.cde.v1.params";
const OBSERVATION_COMMIT_DOMAIN: &[u8] = b"ucf.cde.v1.observation.commit";

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
const SCORE_STEP_BASE: i32 = 1000;
const CENTER_VALUE: i32 = 5_000;
const SCORE_STEP_MIN: u16 = 200;
const SCORE_STEP_MAX: u16 = 2000;
const EDGE_THRESH_MIN: i16 = 2000;
const EDGE_THRESH_MAX: i16 = 9000;
const MAX_OBSERVATION_KEYS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CdeParams {
    pub score_step: u16,
    pub edge_threshold: i16,
    pub commit: Digest32,
}

impl CdeParams {
    pub fn new(score_step: u16, edge_threshold: i16) -> Self {
        let score_step = score_step.clamp(SCORE_STEP_MIN, SCORE_STEP_MAX);
        let edge_threshold = edge_threshold.clamp(EDGE_THRESH_MIN, EDGE_THRESH_MAX);
        let commit = digest_params(score_step, edge_threshold);
        Self {
            score_step,
            edge_threshold,
            commit,
        }
    }
}

impl Default for CdeParams {
    fn default() -> Self {
        Self::new(1000, SCORE_SPIKE_THRESHOLD)
    }
}

pub fn apply_score_step_delta(params: &CdeParams, delta: i16) -> CdeParams {
    let score_step = apply_i16_delta_u16(params.score_step, delta, SCORE_STEP_MIN, SCORE_STEP_MAX);
    CdeParams::new(score_step, params.edge_threshold)
}

pub fn apply_edge_thresh_delta(params: &CdeParams, delta: i16) -> CdeParams {
    let edge_threshold = apply_i16_delta_i16(
        params.edge_threshold,
        delta,
        EDGE_THRESH_MIN,
        EDGE_THRESH_MAX,
    );
    CdeParams::new(params.score_step, edge_threshold)
}

pub fn derive_observation_commit(
    world_state: Digest32,
    ssm_state_digest: Digest32,
    spike_accepted_root: Digest32,
    phase_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(OBSERVATION_COMMIT_DOMAIN);
    hasher.update(world_state.as_bytes());
    hasher.update(ssm_state_digest.as_bytes());
    hasher.update(spike_accepted_root.as_bytes());
    hasher.update(phase_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn apply_i16_delta_u16(value: u16, delta: i16, min: u16, max: u16) -> u16 {
    let value = i32::from(value);
    let delta = i32::from(delta);
    let updated = value
        .saturating_add(delta)
        .clamp(i32::from(min), i32::from(max));
    updated as u16
}

fn apply_i16_delta_i16(value: i16, delta: i16, min: i16, max: i16) -> i16 {
    let value = i32::from(value);
    let delta = i32::from(delta);
    let updated = value
        .saturating_add(delta)
        .clamp(i32::from(min), i32::from(max));
    updated as i16
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VarId(pub u16);

impl VarId {
    pub const WORLD_STATE: VarId = VarId(1);
    pub const SSM_STATE: VarId = VarId(2);
    pub const SPIKE_ROOT: VarId = VarId(3);
    pub const NSR_TRACE_ROOT: VarId = VarId(4);
    pub const PHASE_COMMIT: VarId = VarId(5);
    pub const SURPRISE_BUCKET: VarId = VarId(6);
    pub const TCF_ATTENTION_CAP: VarId = VarId(7);

    pub const PERCEPTION_SALIENCE: VarId = VarId(101);
    pub const PERCEPTION_NOVELTY: VarId = VarId(102);
    pub const ATTENTION_GAIN: VarId = VarId(103);
    pub const LEARNING_RATE: VarId = VarId(104);
    pub const REPLAY_PRESSURE: VarId = VarId(105);
    pub const SLEEP_DRIVE: VarId = VarId(106);
    pub const NCDE_ENERGY: VarId = VarId(107);
    pub const COHERENCE_PLV: VarId = VarId(108);
    pub const PHI_PROXY: VarId = VarId(109);
    pub const RISK: VarId = VarId(110);
    pub const DRIFT: VarId = VarId(111);
    pub const SURPRISE: VarId = VarId(112);
    pub const OUTPUT_SUPPRESSION: VarId = VarId(113);

    pub fn to_u16(self) -> u16 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Edge {
    pub from: VarId,
    pub to: VarId,
    pub weight_q15: i16,
    pub commit: Digest32,
}

impl Edge {
    pub fn new(from: VarId, to: VarId, weight_q15: i16) -> Self {
        let commit = digest_graph_edge(from, to, weight_q15);
        Self {
            from,
            to,
            weight_q15,
            commit,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObservationKey {
    pub cycle_id: u64,
    pub observation_commit: Digest32,
    pub world_state: Digest32,
    pub ssm_state: Digest32,
    pub spike_root: Digest32,
    pub phase_commit: Digest32,
    pub nsr_trace_root: Digest32,
    pub jepa_surprise: u16,
    pub commit: Digest32,
}

impl ObservationKey {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cycle_id: u64,
        observation_commit: Digest32,
        world_state: Digest32,
        ssm_state: Digest32,
        spike_root: Digest32,
        phase_commit: Digest32,
        nsr_trace_root: Digest32,
        jepa_surprise: u16,
    ) -> Self {
        let mut obs = Self {
            cycle_id,
            observation_commit,
            world_state,
            ssm_state,
            spike_root,
            phase_commit,
            nsr_trace_root,
            jepa_surprise,
            commit: Digest32::new([0u8; 32]),
        };
        obs.commit = digest_observation_key(&obs);
        obs
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CausalGraph {
    pub vars: Vec<VarId>,
    pub edges: Vec<Edge>,
    pub commit: Digest32,
}

impl CausalGraph {
    pub fn new(vars: Vec<VarId>, edges: Vec<Edge>) -> Self {
        let commit = digest_graph(&vars, &edges);
        Self {
            vars,
            edges,
            commit,
        }
    }

    fn refresh_commit(&mut self) {
        self.commit = digest_graph(&self.vars, &self.edges);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Intervention {
    pub var: VarId,
    pub value_commit: Digest32,
    pub strength_q15: i16,
    pub commit: Digest32,
}

impl Intervention {
    pub fn new(var: VarId, value_commit: Digest32, strength_q15: i16) -> Self {
        let commit = digest_query_intervention(var, value_commit, strength_q15);
        Self {
            var,
            value_commit,
            strength_q15,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Query {
    pub target: VarId,
    pub given: Vec<Intervention>,
    pub horizon: u8,
    pub commit: Digest32,
}

impl Query {
    pub fn new(target: VarId, given: Vec<Intervention>, horizon: u8) -> Self {
        let mut query = Self {
            target,
            given,
            horizon,
            commit: Digest32::new([0u8; 32]),
        };
        query.commit = digest_query(&query);
        query
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueryResult {
    pub target: VarId,
    pub effect: u16,
    pub proof_root: Digest32,
    pub commit: Digest32,
}

impl QueryResult {
    pub fn new(target: VarId, effect: u16, proof_root: Digest32, query_commit: Digest32) -> Self {
        let commit = digest_query_result(effect, proof_root, query_commit);
        Self {
            target,
            effect,
            proof_root,
            commit,
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
pub struct CdeIntervention {
    pub cycle_id: u64,
    pub var: VarId,
    pub delta: i16,
    pub basis_commit: Digest32,
    pub commit: Digest32,
}

impl CdeIntervention {
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
    pub phase_bucket: u8,
    pub ssm_salience: u16,
    pub ssm_novelty: u16,
    pub cde_bias: i16,
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
    pub observation_commit: Digest32,
    pub commit: Digest32,
}

impl CdeInputs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cycle_id: u64,
        phase_commit: Digest32,
        phase_bucket: u8,
        ssm_salience: u16,
        ssm_novelty: u16,
        cde_bias: i16,
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
        observation_commit: Digest32,
    ) -> Self {
        let mut inputs = Self {
            cycle_id,
            phase_commit,
            phase_bucket,
            ssm_salience,
            ssm_novelty,
            cde_bias,
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
            observation_commit,
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
    pub intervention: Option<CdeIntervention>,
    pub summary_commit: Digest32,
    pub causal_link_spikes: Vec<Spike>,
    pub commit: Digest32,
}

impl CdeOutputs {
    pub fn new(
        cycle_id: u64,
        dag_commit: Digest32,
        top_edges: Vec<CausalEdge>,
        intervention: Option<CdeIntervention>,
        causal_link_spikes: Vec<Spike>,
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

/// Causal engine boundary for DAG inference.
///
/// # Commit formula
/// - `CdeOutputs.commit = H(cycle_id, dag_commit, summary_commit, top_edges[..], intervention?, spikes[..])`
pub trait CausalEngine {
    fn tick(&mut self, inp: &CdeInputs) -> CdeOutputs;

    fn register_observation(&mut self, obs: ObservationKey);

    fn propose_edge(&mut self, e: Edge);

    fn query(&self, q: &Query) -> QueryResult;

    fn graph_commit(&self) -> Digest32;

    fn params(&self) -> CdeParams;

    fn apply_params(&mut self, params: CdeParams);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ObservationIndex {
    world_state: Digest32,
    ssm_state: Digest32,
    spike_root: Digest32,
    nsr_trace_root: Digest32,
    phase_commit: Digest32,
    surprise_bucket: Digest32,
    commit: Digest32,
}

impl ObservationIndex {
    fn new() -> Self {
        let zero = Digest32::new([0u8; 32]);
        let mut index = Self {
            world_state: zero,
            ssm_state: zero,
            spike_root: zero,
            nsr_trace_root: zero,
            phase_commit: zero,
            surprise_bucket: zero,
            commit: zero,
        };
        index.refresh_commit();
        index
    }

    fn update_from_observation(&mut self, obs: &ObservationKey) {
        self.world_state = obs.world_state;
        self.ssm_state = obs.ssm_state;
        self.spike_root = obs.spike_root;
        self.nsr_trace_root = obs.nsr_trace_root;
        self.phase_commit = obs.phase_commit;
        self.surprise_bucket = digest_surprise_bucket(obs.jepa_surprise);
        self.refresh_commit();
    }

    fn last_commit_for(&self, var: VarId) -> Option<Digest32> {
        match var {
            VarId::WORLD_STATE => Some(self.world_state),
            VarId::SSM_STATE => Some(self.ssm_state),
            VarId::SPIKE_ROOT => Some(self.spike_root),
            VarId::NSR_TRACE_ROOT => Some(self.nsr_trace_root),
            VarId::PHASE_COMMIT => Some(self.phase_commit),
            VarId::SURPRISE_BUCKET => Some(self.surprise_bucket),
            _ => None,
        }
    }

    fn refresh_commit(&mut self) {
        self.commit = digest_observation_index(
            self.world_state,
            self.ssm_state,
            self.spike_root,
            self.nsr_trace_root,
            self.phase_commit,
            self.surprise_bucket,
        );
    }
}

pub struct CdeCore {
    pub dag: CausalDag,
    pub graph: CausalGraph,
    pub prev_values: [u16; 12],
    pub last_intervention_cycle: u64,
    pub min_intervention_gap: u16,
    pub params: CdeParams,
    pub commit: Digest32,
    pub observation_keys: Vec<ObservationKey>,
    edge_scores: Vec<CausalEdge>,
    delta_history: Vec<[i16; 12]>,
    pending_intervention: Option<PendingIntervention>,
    observation_commit_root: Digest32,
    observation_index: ObservationIndex,
    last_input_commit: Digest32,
}

impl Default for CdeCore {
    fn default() -> Self {
        Self::new()
    }
}

impl CausalEngine for CdeCore {
    fn tick(&mut self, inp: &CdeInputs) -> CdeOutputs {
        Self::tick(self, inp)
    }

    fn register_observation(&mut self, obs: ObservationKey) {
        CdeCore::register_observation(self, obs);
    }

    fn propose_edge(&mut self, e: Edge) {
        CdeCore::propose_edge(self, e);
    }

    fn query(&self, q: &Query) -> QueryResult {
        CdeCore::query(self, q)
    }

    fn graph_commit(&self) -> Digest32 {
        self.graph.commit
    }

    fn params(&self) -> CdeParams {
        self.params
    }

    fn apply_params(&mut self, params: CdeParams) {
        CdeCore::apply_params(self, params);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MockCausalEngine {
    params: CdeParams,
    dag_commit: Digest32,
    graph: CausalGraph,
    observation_keys: Vec<ObservationKey>,
    observation_index: ObservationIndex,
    last_input_commit: Digest32,
    commit: Digest32,
}

impl MockCausalEngine {
    pub fn new(dag_commit: Digest32) -> Self {
        let graph = CausalGraph::new(default_graph_vars(), Vec::new());
        let observation_index = ObservationIndex::new();
        let last_input_commit = Digest32::new([0u8; 32]);
        let commit = digest_core(
            graph.commit,
            dag_commit,
            last_input_commit,
            CdeParams::default().commit,
            Digest32::new([0u8; 32]),
            observation_index.commit,
        );
        Self {
            params: CdeParams::default(),
            dag_commit,
            graph,
            observation_keys: Vec::new(),
            observation_index,
            last_input_commit,
            commit,
        }
    }
}

impl Default for MockCausalEngine {
    fn default() -> Self {
        Self::new(Digest32::new([5u8; 32]))
    }
}

impl CausalEngine for MockCausalEngine {
    fn tick(&mut self, inp: &CdeInputs) -> CdeOutputs {
        self.last_input_commit = inp.commit;
        CdeOutputs::new(inp.cycle_id, self.dag_commit, Vec::new(), None, Vec::new())
    }

    fn register_observation(&mut self, obs: ObservationKey) {
        self.observation_keys.insert(0, obs);
        if self.observation_keys.len() > MAX_OBSERVATION_KEYS {
            self.observation_keys.truncate(MAX_OBSERVATION_KEYS);
        }
        if let Some(latest) = self.observation_keys.first() {
            self.observation_index.update_from_observation(latest);
        }
        let observation_root = digest_observation_ring(&self.observation_keys);
        self.commit = digest_core(
            self.graph.commit,
            self.dag_commit,
            self.last_input_commit,
            self.params.commit,
            observation_root,
            self.observation_index.commit,
        );
    }

    fn propose_edge(&mut self, e: Edge) {
        update_graph_with_edge(&mut self.graph, e);
        self.commit = digest_core(
            self.graph.commit,
            self.dag_commit,
            self.last_input_commit,
            self.params.commit,
            digest_observation_ring(&self.observation_keys),
            self.observation_index.commit,
        );
    }

    fn query(&self, q: &Query) -> QueryResult {
        query_result_from(
            self.graph.commit,
            &self.observation_keys,
            &self.observation_index,
            q,
        )
    }

    fn graph_commit(&self) -> Digest32 {
        self.graph.commit
    }

    fn params(&self) -> CdeParams {
        self.params
    }

    fn apply_params(&mut self, params: CdeParams) {
        self.params = params;
        self.commit = digest_core(
            self.graph.commit,
            self.dag_commit,
            self.last_input_commit,
            self.params.commit,
            digest_observation_ring(&self.observation_keys),
            self.observation_index.commit,
        );
    }
}

impl CdeCore {
    pub fn new() -> Self {
        let nodes = default_nodes();
        let edge_scores = default_edge_scores();
        let dag = CausalDag::new(nodes, Vec::new());
        let graph = CausalGraph::new(default_graph_vars(), Vec::new());
        let params = CdeParams::default();
        let observation_commit_root = Digest32::new([0u8; 32]);
        let observation_index = ObservationIndex::new();
        let last_input_commit = Digest32::new([0u8; 32]);
        let commit = digest_core(
            graph.commit,
            dag.commit,
            last_input_commit,
            params.commit,
            observation_commit_root,
            observation_index.commit,
        );
        Self {
            dag,
            graph,
            prev_values: [0; 12],
            last_intervention_cycle: 0,
            min_intervention_gap: 3,
            params,
            commit,
            observation_keys: Vec::new(),
            edge_scores,
            delta_history: Vec::new(),
            pending_intervention: None,
            observation_commit_root,
            observation_index,
            last_input_commit,
        }
    }

    pub fn tick(&mut self, inp: &CdeInputs) -> CdeOutputs {
        self.last_input_commit = inp.commit;
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
            let updated = update_edge_score(edge, proxy, inp.cde_bias, &self.dag, &self.params);
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
        let spikes = build_spikes(inp, &top_edges, summary_commit, &self.params);
        let outputs = CdeOutputs::new(
            inp.cycle_id,
            self.dag.commit,
            top_edges,
            intervention,
            spikes,
        );
        self.commit = digest_core(
            self.graph.commit,
            self.dag.commit,
            self.last_input_commit,
            self.params.commit,
            self.observation_commit_root,
            self.observation_index.commit,
        );
        outputs
    }

    pub fn apply_params(&mut self, params: CdeParams) {
        self.params = params;
        self.commit = digest_core(
            self.graph.commit,
            self.dag.commit,
            self.last_input_commit,
            self.params.commit,
            self.observation_commit_root,
            self.observation_index.commit,
        );
    }

    pub fn register_observation(&mut self, obs: ObservationKey) {
        self.track_observation_key(obs);
        self.commit = digest_core(
            self.graph.commit,
            self.dag.commit,
            self.last_input_commit,
            self.params.commit,
            self.observation_commit_root,
            self.observation_index.commit,
        );
    }

    pub fn propose_edge(&mut self, e: Edge) {
        update_graph_with_edge(&mut self.graph, e);
        self.commit = digest_core(
            self.graph.commit,
            self.dag.commit,
            self.last_input_commit,
            self.params.commit,
            self.observation_commit_root,
            self.observation_index.commit,
        );
    }

    pub fn query(&self, q: &Query) -> QueryResult {
        query_result_from(
            self.graph.commit,
            &self.observation_keys,
            &self.observation_index,
            q,
        )
    }

    fn select_intervention(
        &mut self,
        inp: &CdeInputs,
        top_edges: &[CausalEdge],
    ) -> Option<CdeIntervention> {
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
        let intervention =
            CdeIntervention::new(inp.cycle_id, candidate.from, delta, self.dag.commit);
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

    fn track_observation_key(&mut self, obs: ObservationKey) {
        self.observation_keys.insert(0, obs);
        if self.observation_keys.len() > MAX_OBSERVATION_KEYS {
            self.observation_keys.truncate(MAX_OBSERVATION_KEYS);
        }
        if let Some(latest) = self.observation_keys.first() {
            self.observation_index.update_from_observation(latest);
        }
        self.observation_commit_root = digest_observation_ring(&self.observation_keys);
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
    VarId::PERCEPTION_SALIENCE,
    VarId::PERCEPTION_NOVELTY,
    VarId::ATTENTION_GAIN,
    VarId::LEARNING_RATE,
    VarId::REPLAY_PRESSURE,
    VarId::SLEEP_DRIVE,
    VarId::NCDE_ENERGY,
    VarId::COHERENCE_PLV,
    VarId::PHI_PROXY,
    VarId::RISK,
    VarId::DRIFT,
    VarId::SURPRISE,
];

const CANDIDATE_EDGES: &[(VarId, VarId, u8)] = &[
    (VarId::PERCEPTION_SALIENCE, VarId::ATTENTION_GAIN, 0),
    (VarId::PERCEPTION_NOVELTY, VarId::ATTENTION_GAIN, 0),
    (VarId::ATTENTION_GAIN, VarId::LEARNING_RATE, 0),
    (VarId::LEARNING_RATE, VarId::ATTENTION_GAIN, 1),
    (VarId::ATTENTION_GAIN, VarId::REPLAY_PRESSURE, 1),
    (VarId::REPLAY_PRESSURE, VarId::SLEEP_DRIVE, 1),
    (VarId::SLEEP_DRIVE, VarId::REPLAY_PRESSURE, 2),
    (VarId::REPLAY_PRESSURE, VarId::NCDE_ENERGY, 1),
    (VarId::NCDE_ENERGY, VarId::COHERENCE_PLV, 1),
    (VarId::COHERENCE_PLV, VarId::PHI_PROXY, 0),
    (VarId::PHI_PROXY, VarId::ATTENTION_GAIN, 2),
    (VarId::RISK, VarId::OUTPUT_SUPPRESSION, 0),
    (VarId::RISK, VarId::REPLAY_PRESSURE, 1),
    (VarId::DRIFT, VarId::REPLAY_PRESSURE, 1),
    (VarId::SURPRISE, VarId::ATTENTION_GAIN, 0),
    (VarId::SURPRISE, VarId::REPLAY_PRESSURE, 1),
    (VarId::PERCEPTION_NOVELTY, VarId::SURPRISE, 1),
    (VarId::PERCEPTION_SALIENCE, VarId::SURPRISE, 1),
    (VarId::REPLAY_PRESSURE, VarId::SURPRISE, 2),
    (VarId::ATTENTION_GAIN, VarId::RISK, 2),
    (VarId::RISK, VarId::ATTENTION_GAIN, 1),
    (VarId::DRIFT, VarId::RISK, 1),
    (VarId::COHERENCE_PLV, VarId::RISK, 2),
    (VarId::PHI_PROXY, VarId::RISK, 2),
];

fn default_nodes() -> Vec<VarId> {
    let mut nodes = Vec::new();
    for var in OBSERVED_VARS {
        nodes.push(var);
    }
    nodes.push(VarId::OUTPUT_SUPPRESSION);
    nodes.truncate(MAX_NODES);
    nodes
}

fn default_graph_vars() -> Vec<VarId> {
    vec![
        VarId::WORLD_STATE,
        VarId::SSM_STATE,
        VarId::SPIKE_ROOT,
        VarId::NSR_TRACE_ROOT,
        VarId::PHASE_COMMIT,
        VarId::SURPRISE_BUCKET,
        VarId::TCF_ATTENTION_CAP,
    ]
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

fn update_edge_score(
    edge: &CausalEdge,
    proxy: i16,
    cde_bias: i16,
    dag: &CausalDag,
    params: &CdeParams,
) -> CausalEdge {
    let mut score = i32::from(edge.score);
    let decay = score / DECAY_DIV;
    let scaled = i32::from(proxy) * i32::from(params.score_step);
    let delta = scaled / (PROXY_SCALE * SCORE_STEP_BASE);
    if delta > 0 && would_create_cycle(dag, edge.from, edge.to) {
        score -= decay;
    } else {
        score = score - decay + delta;
    }
    score = score.saturating_add(i32::from(cde_bias));
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

fn update_graph_with_edge(graph: &mut CausalGraph, edge: Edge) {
    let mut replaced = false;
    for existing in &mut graph.edges {
        if existing.from == edge.from && existing.to == edge.to {
            let edge_abs = i32::from(edge.weight_q15).abs();
            let existing_abs = i32::from(existing.weight_q15).abs();
            if edge_abs > existing_abs {
                *existing = edge;
            }
            replaced = true;
            break;
        }
    }
    if !replaced {
        graph.edges.push(edge);
    }
    if !graph.vars.contains(&edge.from) {
        graph.vars.push(edge.from);
    }
    if !graph.vars.contains(&edge.to) {
        graph.vars.push(edge.to);
    }
    graph.vars.sort_by_key(|var| var.to_u16());
    graph.vars.dedup();
    graph.edges.sort_by(|a, b| {
        a.from
            .to_u16()
            .cmp(&b.from.to_u16())
            .then_with(|| a.to.to_u16().cmp(&b.to.to_u16()))
    });
    graph.refresh_commit();
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
    params: &CdeParams,
) -> Vec<Spike> {
    let mut spikes = Vec::new();
    for edge in edges {
        if edge.score.abs() < params.edge_threshold {
            continue;
        }
        if spikes.len() >= MAX_TOP_EDGES {
            break;
        }
        let payload_commit = spike_payload_commit(edge.commit, summary_commit);
        let intensity = score_to_intensity(edge.score);
        spikes.push(Spike::new(
            inp.cycle_id,
            SpikeKind::CausalLink,
            intensity,
            inp.phase_bucket,
            ModuleId::Cde,
            payload_commit,
        ));
    }
    spikes
}

fn score_to_intensity(score: i16) -> u16 {
    let magnitude = score.abs().min(MAX_SCORE) as u16;
    (magnitude / 2).min(10_000)
}

fn query_result_from(
    graph_commit: Digest32,
    observation_keys: &[ObservationKey],
    observation_index: &ObservationIndex,
    q: &Query,
) -> QueryResult {
    let last_obs_commit = observation_index
        .last_commit_for(q.target)
        .unwrap_or_else(|| Digest32::new([0u8; 32]));
    let effect = digest_query_effect(graph_commit, q.commit, last_obs_commit);
    let proof_root = digest_query_proof(graph_commit, q, observation_keys);
    QueryResult::new(q.target, effect, proof_root, q.commit)
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

fn digest_graph_edge(from: VarId, to: VarId, weight_q15: i16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(GRAPH_EDGE_DOMAIN);
    hasher.update(&from.to_u16().to_be_bytes());
    hasher.update(&to.to_u16().to_be_bytes());
    hasher.update(&weight_q15.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_graph(vars: &[VarId], edges: &[Edge]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(GRAPH_DOMAIN);
    hasher.update(&(vars.len() as u16).to_be_bytes());
    let mut sorted_vars = vars.to_vec();
    sorted_vars.sort_by_key(|var| var.to_u16());
    for var in sorted_vars {
        hasher.update(&var.to_u16().to_be_bytes());
    }
    let mut sorted_edges = edges.to_vec();
    sorted_edges.sort_by(|a, b| {
        a.from
            .to_u16()
            .cmp(&b.from.to_u16())
            .then_with(|| a.to.to_u16().cmp(&b.to.to_u16()))
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

fn digest_query_intervention(var: VarId, value_commit: Digest32, strength_q15: i16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(QUERY_INTERVENTION_DOMAIN);
    hasher.update(&var.to_u16().to_be_bytes());
    hasher.update(value_commit.as_bytes());
    hasher.update(&strength_q15.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_query(query: &Query) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(QUERY_DOMAIN);
    hasher.update(&query.target.to_u16().to_be_bytes());
    hasher.update(&[query.horizon]);
    hasher.update(&(query.given.len() as u16).to_be_bytes());
    for intervention in &query.given {
        hasher.update(intervention.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_query_effect(graph_commit: Digest32, query_commit: Digest32, last_obs: Digest32) -> u16 {
    let mut hasher = Hasher::new();
    hasher.update(QUERY_EFFECT_DOMAIN);
    hasher.update(graph_commit.as_bytes());
    hasher.update(query_commit.as_bytes());
    hasher.update(last_obs.as_bytes());
    let bytes = hasher.finalize();
    let raw = bytes.as_bytes();
    let effect_seed = u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]);
    (effect_seed % 10_001) as u16
}

fn digest_query_proof(
    graph_commit: Digest32,
    query: &Query,
    observation_keys: &[ObservationKey],
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(QUERY_PROOF_DOMAIN);
    hasher.update(graph_commit.as_bytes());
    hasher.update(query.commit.as_bytes());
    for obs in observation_keys.iter().take(4) {
        hasher.update(obs.commit.as_bytes());
    }
    for intervention in &query.given {
        hasher.update(intervention.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_query_result(effect: u16, proof_root: Digest32, query_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(QUERY_RESULT_DOMAIN);
    hasher.update(&effect.to_be_bytes());
    hasher.update(proof_root.as_bytes());
    hasher.update(query_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_params(score_step: u16, edge_threshold: i16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(PARAM_DOMAIN);
    hasher.update(&score_step.to_be_bytes());
    hasher.update(&edge_threshold.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_observation_key(obs: &ObservationKey) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(OBSERVATION_KEY_DOMAIN);
    hasher.update(&obs.cycle_id.to_be_bytes());
    hasher.update(obs.observation_commit.as_bytes());
    hasher.update(obs.world_state.as_bytes());
    hasher.update(obs.ssm_state.as_bytes());
    hasher.update(obs.spike_root.as_bytes());
    hasher.update(obs.phase_commit.as_bytes());
    hasher.update(obs.nsr_trace_root.as_bytes());
    hasher.update(&obs.jepa_surprise.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_observation_ring(observations: &[ObservationKey]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(OBSERVATION_RING_DOMAIN);
    hasher.update(
        &u16::try_from(observations.len())
            .unwrap_or(u16::MAX)
            .to_be_bytes(),
    );
    for obs in observations {
        hasher.update(obs.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_observation_index(
    world_state: Digest32,
    ssm_state: Digest32,
    spike_root: Digest32,
    nsr_trace_root: Digest32,
    phase_commit: Digest32,
    surprise_bucket: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(OBSERVATION_INDEX_DOMAIN);
    hasher.update(world_state.as_bytes());
    hasher.update(ssm_state.as_bytes());
    hasher.update(spike_root.as_bytes());
    hasher.update(nsr_trace_root.as_bytes());
    hasher.update(phase_commit.as_bytes());
    hasher.update(surprise_bucket.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_surprise_bucket(surprise: u16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(SURPRISE_BUCKET_DOMAIN);
    hasher.update(&surprise.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_inputs(inputs: &CdeInputs) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(INPUT_DOMAIN);
    hasher.update(&inputs.cycle_id.to_be_bytes());
    hasher.update(inputs.phase_commit.as_bytes());
    hasher.update(&[inputs.phase_bucket]);
    hasher.update(&inputs.ssm_salience.to_be_bytes());
    hasher.update(&inputs.ssm_novelty.to_be_bytes());
    hasher.update(&inputs.cde_bias.to_be_bytes());
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
    hasher.update(inputs.observation_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_summary(
    dag_commit: Digest32,
    edges: &[CausalEdge],
    intervention: Option<&CdeIntervention>,
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
    intervention: Option<&CdeIntervention>,
    spikes: &[Spike],
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

fn digest_core(
    graph_commit: Digest32,
    dag_commit: Digest32,
    input_commit: Digest32,
    params_commit: Digest32,
    observation_commit_root: Digest32,
    observation_index_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(CORE_DOMAIN);
    hasher.update(graph_commit.as_bytes());
    hasher.update(dag_commit.as_bytes());
    hasher.update(input_commit.as_bytes());
    hasher.update(params_commit.as_bytes());
    hasher.update(observation_commit_root.as_bytes());
    hasher.update(observation_index_commit.as_bytes());
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
            2,
            5_000,
            4_800,
            0,
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
            Digest32::new([8u8; 32]),
        )
    }

    fn base_observation(cycle_id: u64) -> ObservationKey {
        ObservationKey::new(
            cycle_id,
            Digest32::new([1u8; 32]),
            Digest32::new([2u8; 32]),
            Digest32::new([3u8; 32]),
            Digest32::new([4u8; 32]),
            Digest32::new([5u8; 32]),
            Digest32::new([6u8; 32]),
            4_200,
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
        let edge_ab = CausalEdge::new(VarId::RISK, VarId::DRIFT, 5_000, 0);
        let edge_ba = CausalEdge::new(VarId::DRIFT, VarId::RISK, 5_000, 0);
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
        core.edge_scores = vec![CausalEdge::new(VarId::RISK, VarId::DRIFT, 2_500, 0)];
        core.dag = CausalDag::new(core.dag.nodes.clone(), core.edge_scores.clone());
        core.pending_intervention = Some(PendingIntervention {
            edge_key: EdgeKey {
                from: VarId::RISK,
                to: VarId::DRIFT,
                lag: 0,
            },
            expected_sign: 1,
            target: VarId::DRIFT,
        });
        core.delta_history = vec![[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0]];
        let out = core.tick(&base_inputs(2));
        let boosted = out
            .top_edges
            .iter()
            .find(|edge| edge.from == VarId::RISK && edge.to == VarId::DRIFT)
            .map(|edge| edge.score)
            .unwrap_or(0);
        assert!(boosted >= 2_500);
    }

    #[test]
    fn high_score_emits_spike() {
        let mut core = CdeCore::new();
        core.edge_scores = vec![CausalEdge::new(
            VarId::RISK,
            VarId::DRIFT,
            SCORE_SPIKE_THRESHOLD + 3_000,
            0,
        )];
        core.dag = CausalDag::new(core.dag.nodes.clone(), core.edge_scores.clone());
        let outputs = core.tick(&base_inputs(3));
        assert!(!outputs.causal_link_spikes.is_empty());
    }

    #[test]
    fn rsa_param_updates_are_clamped() {
        let params = CdeParams::default();
        let updated = apply_score_step_delta(&params, 5000);
        assert_eq!(updated.score_step, SCORE_STEP_MAX);

        let updated = apply_edge_thresh_delta(&params, -5000);
        assert_eq!(updated.edge_threshold, EDGE_THRESH_MIN);
    }

    #[test]
    fn observation_commit_is_deterministic() {
        let commit_a = derive_observation_commit(
            Digest32::new([1u8; 32]),
            Digest32::new([2u8; 32]),
            Digest32::new([3u8; 32]),
            Digest32::new([4u8; 32]),
        );
        let commit_b = derive_observation_commit(
            Digest32::new([1u8; 32]),
            Digest32::new([2u8; 32]),
            Digest32::new([3u8; 32]),
            Digest32::new([4u8; 32]),
        );
        assert_eq!(commit_a, commit_b);
    }

    #[test]
    fn propose_edge_dedup_and_sort() {
        let mut core = CdeCore::new();
        let edge_low = Edge::new(VarId::WORLD_STATE, VarId::SPIKE_ROOT, 1_000);
        let edge_high = Edge::new(VarId::WORLD_STATE, VarId::SPIKE_ROOT, 5_000);
        let edge_other = Edge::new(VarId::NSR_TRACE_ROOT, VarId::TCF_ATTENTION_CAP, -3_000);
        core.propose_edge(edge_low);
        core.propose_edge(edge_other);
        core.propose_edge(edge_high);
        assert_eq!(core.graph.edges.len(), 2);
        let edge = &core.graph.edges[0];
        assert_eq!(edge.from, VarId::WORLD_STATE);
        assert_eq!(edge.to, VarId::SPIKE_ROOT);
        assert_eq!(edge.weight_q15, 5_000);
    }

    #[test]
    fn query_is_deterministic() {
        let mut core = CdeCore::new();
        core.register_observation(base_observation(1));
        core.propose_edge(Edge::new(VarId::WORLD_STATE, VarId::SPIKE_ROOT, 2_000));
        let intervention = Intervention::new(VarId::WORLD_STATE, Digest32::new([9u8; 32]), 3_000);
        let query = Query::new(VarId::WORLD_STATE, vec![intervention], 1);
        let result_a = core.query(&query);
        let result_b = core.query(&query);
        assert_eq!(result_a, result_b);
    }

    #[test]
    fn register_observation_updates_commit() {
        let mut core = CdeCore::new();
        let initial_graph_commit = core.graph.commit;
        let initial_commit = core.commit;
        let obs = base_observation(9);
        core.register_observation(obs);
        assert_ne!(core.commit, initial_commit);
        assert_eq!(core.graph.commit, initial_graph_commit);
        let mut other = CdeCore::new();
        other.register_observation(base_observation(9));
        assert_eq!(core.commit, other.commit);
    }
}
