#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_types::{CausalEdge, CausalGraphStub, CausalNode, Digest32, WorldStateVec};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CdeHypothesis {
    pub digest: Digest32,
    pub nodes: usize,
    pub edges: usize,
}

pub trait CdePort {
    fn infer(&self, graph: &mut CausalGraphStub, obs: &WorldStateVec) -> CdeHypothesis;
}

#[derive(Clone, Default)]
pub struct MockCdePort;

impl MockCdePort {
    pub fn new() -> Self {
        Self
    }
}

impl CdePort for MockCdePort {
    fn infer(&self, graph: &mut CausalGraphStub, obs: &WorldStateVec) -> CdeHypothesis {
        let digest = digest_graph_obs(graph, obs);
        let node_id = format!("obs-{}", hex_prefix(digest.as_bytes()));
        if !graph.nodes.iter().any(|node| node.id == node_id) {
            graph.nodes.push(CausalNode::new(node_id.clone()));
            if let Some(first) = graph.nodes.first() {
                if first.id != node_id {
                    graph.edges.push(CausalEdge::new(first.id.clone(), node_id));
                }
            }
        }

        CdeHypothesis {
            digest,
            nodes: graph.nodes.len(),
            edges: graph.edges.len(),
        }
    }
}

fn digest_graph_obs(graph: &CausalGraphStub, obs: &WorldStateVec) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(&u64::try_from(graph.nodes.len()).unwrap_or(0).to_be_bytes());
    for node in &graph.nodes {
        hasher.update(node.id.as_bytes());
    }
    hasher.update(&u64::try_from(graph.edges.len()).unwrap_or(0).to_be_bytes());
    for edge in &graph.edges {
        hasher.update(edge.from.as_bytes());
        hasher.update(edge.to.as_bytes());
    }
    hasher.update(&obs.bytes);
    for dim in &obs.dims {
        hasher.update(&dim.to_be_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn hex_prefix(bytes: &[u8; 32]) -> String {
    bytes
        .iter()
        .take(4)
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_cde_is_deterministic() {
        let mut graph_a = CausalGraphStub::new(Vec::new(), Vec::new());
        let mut graph_b = CausalGraphStub::new(Vec::new(), Vec::new());
        let obs = WorldStateVec::new(vec![1, 2], vec![2]);
        let port = MockCdePort::new();

        let out_a = port.infer(&mut graph_a, &obs);
        let out_b = port.infer(&mut graph_b, &obs);

        assert_eq!(out_a.digest, out_b.digest);
        assert_eq!(graph_a, graph_b);
    }
}
