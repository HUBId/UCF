#![forbid(unsafe_code)]

use std::collections::{BTreeMap, VecDeque};

use blake3::Hasher;
use ucf_cde_port::CdeHypothesis;
use ucf_types::{Digest32, WorldStateVec};

pub type NodeId = u32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CausalNode {
    pub id: NodeId,
    pub name: String,
}

impl CausalNode {
    pub fn new(id: NodeId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CausalEdge {
    pub from: NodeId,
    pub to: NodeId,
}

impl CausalEdge {
    pub fn new(from: NodeId, to: NodeId) -> Self {
        Self { from, to }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScmDag {
    pub nodes: Vec<CausalNode>,
    pub edges: Vec<CausalEdge>,
}

impl ScmDag {
    pub fn new(nodes: Vec<CausalNode>, edges: Vec<CausalEdge>) -> Self {
        Self { nodes, edges }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Intervention {
    pub node: NodeId,
    pub value: i32,
}

impl Intervention {
    pub fn new(node: NodeId, value: i32) -> Self {
        Self { node, value }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CounterfactualQuery {
    pub interventions: Vec<Intervention>,
    pub target: NodeId,
}

impl CounterfactualQuery {
    pub fn new(interventions: Vec<Intervention>, target: NodeId) -> Self {
        Self {
            interventions,
            target,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CounterfactualResult {
    pub predicted: i32,
    pub confidence: u16,
}

pub trait ScmPort {
    fn update(&mut self, obs: &WorldStateVec, hint: Option<&CdeHypothesis>) -> ScmDag;
    fn counterfactual(&self, q: &CounterfactualQuery) -> CounterfactualResult;
}

#[derive(Clone, Debug)]
pub struct MockScmPort {
    dag: ScmDag,
    dag_commit: Digest32,
    last_confidence: Option<u16>,
}

impl MockScmPort {
    pub fn new() -> Self {
        let dag = default_dag();
        let dag_commit = digest_dag(&dag);
        Self {
            dag,
            dag_commit,
            last_confidence: None,
        }
    }
}

impl Default for MockScmPort {
    fn default() -> Self {
        Self::new()
    }
}

impl ScmPort for MockScmPort {
    fn update(&mut self, obs: &WorldStateVec, hint: Option<&CdeHypothesis>) -> ScmDag {
        let candidate = build_dag_from_obs(obs);
        if is_acyclic(&candidate) {
            self.dag = candidate;
            self.dag_commit = digest_dag(&self.dag);
        }
        self.last_confidence = hint.map(|hyp| hyp.confidence);
        self.dag.clone()
    }

    fn counterfactual(&self, q: &CounterfactualQuery) -> CounterfactualResult {
        let mut hasher = Hasher::new();
        hasher.update(b"ucf.scm.counterfactual.v1");
        hasher.update(self.dag_commit.as_bytes());
        hasher.update(&q.target.to_be_bytes());
        hasher.update(
            &u64::try_from(q.interventions.len())
                .unwrap_or(0)
                .to_be_bytes(),
        );
        for intervention in &q.interventions {
            hasher.update(&intervention.node.to_be_bytes());
            hasher.update(&intervention.value.to_be_bytes());
        }
        let digest = hasher.finalize();
        let predicted = i32::from_be_bytes(digest.as_bytes()[0..4].try_into().expect("hash size"));
        let confidence = self.last_confidence.unwrap_or(5000);
        CounterfactualResult {
            predicted,
            confidence,
        }
    }
}

pub fn digest_dag(dag: &ScmDag) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.scm.dag.v1");
    hasher.update(&u64::try_from(dag.nodes.len()).unwrap_or(0).to_be_bytes());
    for node in &dag.nodes {
        hasher.update(&node.id.to_be_bytes());
        hasher.update(&u64::try_from(node.name.len()).unwrap_or(0).to_be_bytes());
        hasher.update(node.name.as_bytes());
    }
    hasher.update(&u64::try_from(dag.edges.len()).unwrap_or(0).to_be_bytes());
    for edge in &dag.edges {
        hasher.update(&edge.from.to_be_bytes());
        hasher.update(&edge.to.to_be_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn default_dag() -> ScmDag {
    let nodes = (0..4)
        .map(|id| CausalNode::new(id, format!("node-{id}")))
        .collect();
    let edges = vec![
        CausalEdge::new(0, 1),
        CausalEdge::new(1, 2),
        CausalEdge::new(2, 3),
    ];
    ScmDag::new(nodes, edges)
}

fn build_dag_from_obs(obs: &WorldStateVec) -> ScmDag {
    let digest = digest_observation(obs);
    let mut dag = default_dag();
    let flags = digest.as_bytes()[0];
    if flags & 0b0000_0001 != 0 {
        dag.edges.push(CausalEdge::new(0, 2));
    }
    if flags & 0b0000_0010 != 0 {
        dag.edges.push(CausalEdge::new(1, 3));
    }
    if flags & 0b0000_0100 != 0 {
        dag.edges.push(CausalEdge::new(0, 3));
    }
    dag
}

fn digest_observation(obs: &WorldStateVec) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.scm.obs.v1");
    hasher.update(&obs.bytes);
    for dim in &obs.dims {
        hasher.update(&dim.to_be_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn is_acyclic(dag: &ScmDag) -> bool {
    let mut indegree: BTreeMap<NodeId, usize> = dag.nodes.iter().map(|node| (node.id, 0)).collect();
    let mut adjacency: BTreeMap<NodeId, Vec<NodeId>> = BTreeMap::new();

    for edge in &dag.edges {
        adjacency.entry(edge.from).or_default().push(edge.to);
        let entry = indegree.entry(edge.to).or_insert(0);
        *entry = entry.saturating_add(1);
        indegree.entry(edge.from).or_insert(0);
    }

    let mut queue: VecDeque<NodeId> = indegree
        .iter()
        .filter_map(|(node, count)| if *count == 0 { Some(*node) } else { None })
        .collect();

    let mut visited = 0usize;
    while let Some(node) = queue.pop_front() {
        visited = visited.saturating_add(1);
        if let Some(neighbors) = adjacency.get(&node) {
            for neighbor in neighbors {
                if let Some(entry) = indegree.get_mut(neighbor) {
                    *entry = entry.saturating_sub(1);
                    if *entry == 0 {
                        queue.push_back(*neighbor);
                    }
                }
            }
        }
    }

    visited == indegree.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scm_update_is_deterministic() {
        let obs = WorldStateVec::new(vec![1, 2, 3], vec![3]);
        let mut port_a = MockScmPort::new();
        let mut port_b = MockScmPort::new();

        let dag_a = port_a.update(&obs, None);
        let dag_b = port_b.update(&obs, None);

        assert_eq!(dag_a, dag_b);
    }

    #[test]
    fn scm_dag_is_acyclic() {
        let obs = WorldStateVec::new(vec![4, 5], vec![2]);
        let mut port = MockScmPort::new();
        let dag = port.update(&obs, None);

        assert!(is_acyclic(&dag));
    }

    #[test]
    fn scm_counterfactual_is_deterministic() {
        let obs = WorldStateVec::new(vec![9, 8, 7], vec![3]);
        let mut port = MockScmPort::new();
        let dag = port.update(&obs, None);
        let query = CounterfactualQuery::new(vec![Intervention::new(dag.nodes[0].id, 1)], 1);

        let out_a = port.counterfactual(&query);
        let out_b = port.counterfactual(&query);

        assert_eq!(out_a, out_b);
    }
}
