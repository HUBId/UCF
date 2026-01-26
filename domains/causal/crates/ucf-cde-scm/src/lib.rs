#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_types::Digest32;

pub type VarId = u16;

pub const VAR_RISK: VarId = 1;
pub const VAR_SURPRISE: VarId = 2;
pub const VAR_DRIFT: VarId = 3;
pub const VAR_AROUSAL: VarId = 4;
pub const VAR_OUTPUT_SUPPRESSED: VarId = 5;
pub const VAR_REPLAY_BIAS: VarId = 6;

const WEIGHT_MIN: i16 = -3000;
const WEIGHT_MAX: i16 = 3000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CausalVar {
    pub id: VarId,
    pub name: String,
    pub commit: Digest32,
}

impl CausalVar {
    pub fn new(id: VarId, name: impl Into<String>) -> Self {
        let name = name.into();
        let commit = digest_var(id, &name);
        Self { id, name, commit }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Edge {
    pub from: VarId,
    pub to: VarId,
    pub weight: i16,
    pub commit: Digest32,
}

impl Edge {
    pub fn new(from: VarId, to: VarId, weight: i16) -> Self {
        let commit = digest_edge(from, to, weight);
        Self {
            from,
            to,
            weight,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dag {
    pub vars: Vec<CausalVar>,
    pub edges: Vec<Edge>,
    pub commit: Digest32,
}

impl Dag {
    pub fn new(vars: Vec<CausalVar>, edges: Vec<Edge>) -> Self {
        let commit = digest_dag(&vars, &edges);
        Self {
            vars,
            edges,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservationPoint {
    pub cycle_id: u64,
    pub world_commit: Digest32,
    pub brain_commit: Option<Digest32>,
    pub risk: u16,
    pub surprise: u16,
    pub drift: u16,
    pub commit: Digest32,
}

impl ObservationPoint {
    pub fn new(
        cycle_id: u64,
        world_commit: Digest32,
        brain_commit: Option<Digest32>,
        risk: u16,
        surprise: u16,
        drift: u16,
    ) -> Self {
        let commit = digest_observation_point(
            cycle_id,
            world_commit,
            brain_commit.as_ref(),
            risk,
            surprise,
            drift,
        );
        Self {
            cycle_id,
            world_commit,
            brain_commit,
            risk,
            surprise,
            drift,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Intervention {
    pub var: VarId,
    pub set_to: i16,
    pub commit: Digest32,
}

impl Intervention {
    pub fn new(var: VarId, set_to: i16) -> Self {
        let commit = digest_intervention(var, set_to);
        Self {
            var,
            set_to,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CounterfactualQuery {
    pub base: ObservationPoint,
    pub do_ops: Vec<Intervention>,
    pub commit: Digest32,
}

impl CounterfactualQuery {
    pub fn new(base: ObservationPoint, do_ops: Vec<Intervention>) -> Self {
        let commit = digest_counterfactual_query(&base, &do_ops);
        Self {
            base,
            do_ops,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CounterfactualResult {
    pub predicted_delta: i16,
    pub confidence: u16,
    pub commit: Digest32,
}

impl CounterfactualResult {
    pub fn new(predicted_delta: i16, confidence: u16, query_commit: Digest32) -> Self {
        let commit = digest_counterfactual_result(predicted_delta, confidence, query_commit);
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
    pub interventions_checked: u16,
    pub counterfactuals: Vec<CounterfactualResult>,
    pub flags: u16,
    pub commit: Digest32,
}

impl CausalReport {
    pub fn new(
        dag_commit: Digest32,
        interventions_checked: u16,
        counterfactuals: Vec<CounterfactualResult>,
        flags: u16,
    ) -> Self {
        let commit =
            digest_causal_report(dag_commit, interventions_checked, &counterfactuals, flags);
        Self {
            dag_commit,
            interventions_checked,
            counterfactuals,
            flags,
            commit,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CdeScmEngine {
    dag: Dag,
}

impl Default for CdeScmEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl CdeScmEngine {
    pub fn new() -> Self {
        let vars = default_vars();
        let edges = build_edges(&vars, &Vec::new(), Digest32::new([0u8; 32]));
        let dag = Dag::new(vars, edges);
        Self { dag }
    }

    pub fn dag(&self) -> &Dag {
        &self.dag
    }

    pub fn update_dag(&mut self, obs: &ObservationPoint) -> Dag {
        let vars = if self.dag.vars.is_empty() {
            default_vars()
        } else {
            self.dag.vars.clone()
        };
        let edges = build_edges(&vars, &self.dag.edges, obs.commit);
        self.dag = Dag::new(vars, edges);
        self.dag.clone()
    }

    pub fn counterfactual(&self, q: &CounterfactualQuery) -> CounterfactualResult {
        let base = &q.base;
        let base_score = i32::from(base.risk)
            .saturating_add(i32::from(base.surprise))
            .saturating_sub(i32::from(base.drift));
        let intervention_sum = q
            .do_ops
            .iter()
            .fold(0i32, |acc, op| acc.saturating_add(i32::from(op.set_to)));
        let edge_sum = q.do_ops.iter().fold(0i32, |acc, op| {
            let sum = self
                .dag
                .edges
                .iter()
                .filter(|edge| edge.from == op.var)
                .fold(0i32, |inner, edge| {
                    inner.saturating_add(i32::from(edge.weight))
                });
            acc.saturating_add(sum)
        });
        let predicted = base_score
            .saturating_div(100)
            .saturating_add(intervention_sum)
            .saturating_add(edge_sum.saturating_div(10));
        let predicted_delta = predicted.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
        let abs_weight_sum = q.do_ops.iter().fold(0u32, |acc, op| {
            let sum = self
                .dag
                .edges
                .iter()
                .filter(|edge| edge.from == op.var)
                .fold(0u32, |inner, edge| {
                    let weight = i32::from(edge.weight).unsigned_abs();
                    inner.saturating_add(weight)
                });
            acc.saturating_add(sum)
        });
        let intervention_penalty =
            u32::from(q.do_ops.len().min(u16::MAX as usize) as u16).saturating_mul(200);
        let confidence = (1000u32
            .saturating_add(abs_weight_sum / 2)
            .saturating_sub(intervention_penalty))
        .min(u32::from(u16::MAX)) as u16;
        CounterfactualResult::new(predicted_delta, confidence, q.commit)
    }

    pub fn report(&self, obs: &ObservationPoint, queries: &[CounterfactualQuery]) -> CausalReport {
        let counterfactuals = queries
            .iter()
            .map(|query| self.counterfactual(query))
            .collect::<Vec<_>>();
        let interventions_checked = queries
            .iter()
            .map(|query| query.do_ops.len())
            .sum::<usize>()
            .min(u16::MAX as usize) as u16;
        let mut flags = 0u16;
        if obs.risk > 7000 {
            flags |= 0b0001;
        }
        if obs.drift > 6000 {
            flags |= 0b0010;
        }
        if obs.surprise > 8000 {
            flags |= 0b0100;
        }
        CausalReport::new(
            self.dag.commit,
            interventions_checked,
            counterfactuals,
            flags,
        )
    }
}

fn default_vars() -> Vec<CausalVar> {
    vec![
        CausalVar::new(VAR_RISK, "Risk"),
        CausalVar::new(VAR_SURPRISE, "Surprise"),
        CausalVar::new(VAR_DRIFT, "Drift"),
        CausalVar::new(VAR_AROUSAL, "Arousal"),
        CausalVar::new(VAR_OUTPUT_SUPPRESSED, "OutputSuppressed"),
        CausalVar::new(VAR_REPLAY_BIAS, "ReplayBias"),
    ]
}

fn build_edges(vars: &[CausalVar], existing: &[Edge], obs_commit: Digest32) -> Vec<Edge> {
    let mut edges = Vec::new();
    for (idx, from) in vars.iter().enumerate() {
        for to in vars.iter().skip(idx + 1) {
            let current = existing
                .iter()
                .find(|edge| edge.from == from.id && edge.to == to.id)
                .map(|edge| edge.weight)
                .unwrap_or(0);
            let updated = update_weight(obs_commit, from.id, to.id, current);
            edges.push(Edge::new(from.id, to.id, updated));
        }
    }
    edges
}

fn update_weight(obs_commit: Digest32, from: VarId, to: VarId, current: i16) -> i16 {
    let seed = hash_edge_seed(obs_commit, from, to);
    let step = (seed % 5).saturating_add(1) as i16;
    let sign = if seed & 1 == 0 { 1 } else { -1 };
    let next = current.saturating_add(step.saturating_mul(sign));
    next.clamp(WEIGHT_MIN, WEIGHT_MAX)
}

fn hash_edge_seed(obs_commit: Digest32, from: VarId, to: VarId) -> u64 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.cde.scm.edge.seed.v1");
    hasher.update(obs_commit.as_bytes());
    hasher.update(&from.to_be_bytes());
    hasher.update(&to.to_be_bytes());
    let digest = hasher.finalize();
    let bytes = digest.as_bytes();
    u64::from_be_bytes(bytes[0..8].try_into().expect("seed bytes"))
}

fn digest_var(id: VarId, name: &str) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.cde.var.v1");
    hasher.update(&id.to_be_bytes());
    hasher.update(name.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_edge(from: VarId, to: VarId, weight: i16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.cde.edge.v1");
    hasher.update(&from.to_be_bytes());
    hasher.update(&to.to_be_bytes());
    hasher.update(&weight.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_dag(vars: &[CausalVar], edges: &[Edge]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.cde.dag.v1");
    hasher.update(&u64::try_from(vars.len()).unwrap_or(0).to_be_bytes());
    for var in vars {
        hasher.update(&var.id.to_be_bytes());
        hasher.update(&u64::try_from(var.name.len()).unwrap_or(0).to_be_bytes());
        hasher.update(var.name.as_bytes());
        hasher.update(var.commit.as_bytes());
    }
    hasher.update(&u64::try_from(edges.len()).unwrap_or(0).to_be_bytes());
    for edge in edges {
        hasher.update(&edge.from.to_be_bytes());
        hasher.update(&edge.to.to_be_bytes());
        hasher.update(&edge.weight.to_be_bytes());
        hasher.update(edge.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_observation_point(
    cycle_id: u64,
    world_commit: Digest32,
    brain_commit: Option<&Digest32>,
    risk: u16,
    surprise: u16,
    drift: u16,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.cde.obs.v1");
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(world_commit.as_bytes());
    if let Some(commit) = brain_commit {
        hasher.update(commit.as_bytes());
    } else {
        hasher.update(&[0u8; 32]);
    }
    hasher.update(&risk.to_be_bytes());
    hasher.update(&surprise.to_be_bytes());
    hasher.update(&drift.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_intervention(var: VarId, set_to: i16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.cde.intervention.v1");
    hasher.update(&var.to_be_bytes());
    hasher.update(&set_to.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_counterfactual_query(base: &ObservationPoint, ops: &[Intervention]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.cde.counterfactual.query.v1");
    hasher.update(base.commit.as_bytes());
    hasher.update(&u64::try_from(ops.len()).unwrap_or(0).to_be_bytes());
    for op in ops {
        hasher.update(op.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_counterfactual_result(
    predicted_delta: i16,
    confidence: u16,
    query_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.cde.counterfactual.result.v1");
    hasher.update(&predicted_delta.to_be_bytes());
    hasher.update(&confidence.to_be_bytes());
    hasher.update(query_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_causal_report(
    dag_commit: Digest32,
    interventions_checked: u16,
    counterfactuals: &[CounterfactualResult],
    flags: u16,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.cde.report.v1");
    hasher.update(dag_commit.as_bytes());
    hasher.update(&interventions_checked.to_be_bytes());
    hasher.update(&flags.to_be_bytes());
    hasher.update(
        &u64::try_from(counterfactuals.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    for result in counterfactuals {
        hasher.update(result.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_obs(cycle_id: u64, seed: u8) -> ObservationPoint {
        ObservationPoint::new(
            cycle_id,
            Digest32::new([seed; 32]),
            None,
            5000 + u16::from(seed),
            2000 + u16::from(seed),
            1000 + u16::from(seed),
        )
    }

    #[test]
    fn dag_commit_is_deterministic_for_sequence() {
        let mut engine_a = CdeScmEngine::new();
        let mut engine_b = CdeScmEngine::new();
        let obs_a = sample_obs(1, 1);
        let obs_b = sample_obs(2, 2);

        engine_a.update_dag(&obs_a);
        engine_a.update_dag(&obs_b);
        engine_b.update_dag(&obs_a);
        engine_b.update_dag(&obs_b);

        assert_eq!(engine_a.dag().commit, engine_b.dag().commit);
    }

    #[test]
    fn dag_edges_remain_acyclic() {
        let mut engine = CdeScmEngine::new();
        let obs = sample_obs(1, 3);
        let dag = engine.update_dag(&obs);

        assert!(dag.edges.iter().all(|edge| edge.from < edge.to));
    }

    #[test]
    fn counterfactual_is_deterministic_and_confidence_scales() {
        let mut engine = CdeScmEngine::new();
        let obs = sample_obs(1, 4);
        engine.update_dag(&obs);
        let query =
            CounterfactualQuery::new(obs.clone(), vec![Intervention::new(VAR_REPLAY_BIAS, 1)]);
        let result_a = engine.counterfactual(&query);
        let result_b = engine.counterfactual(&query);
        assert_eq!(result_a, result_b);

        let confidence_a = result_a.confidence;
        for _ in 0..3 {
            engine.update_dag(&obs);
        }
        let result_c = engine.counterfactual(&query);
        assert!(result_c.confidence >= confidence_a);
    }
}
