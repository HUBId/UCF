use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

pub type VarId = u32;

pub const ACTIVE_CONFIDENCE_TH: f32 = 0.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Edge {
    pub from: VarId,
    pub to: VarId,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Hypothesis {
    pub edge: Edge,
    pub confidence: f32,
    pub evidence_count: u32,
    pub last_update_ms: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CausalGraph {
    pub vars: Vec<VarId>,
    pub hyps: Vec<Hypothesis>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Intervention {
    pub var: VarId,
    pub set_to: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Counterfactual {
    pub affected: Vec<(VarId, f32)>,
}

impl CausalGraph {
    pub fn upsert_var(&mut self, v: VarId) {
        if !self.vars.contains(&v) {
            self.vars.push(v);
        }
    }

    pub fn upsert_hypothesis(&mut self, edge: Edge, now_ms: u64, delta_conf: f32) {
        self.upsert_var(edge.from);
        self.upsert_var(edge.to);

        if let Some(h) = self.hyps.iter_mut().find(|h| h.edge == edge) {
            h.confidence = (h.confidence + delta_conf).clamp(0.0, 1.0);
            h.evidence_count = h.evidence_count.saturating_add(1);
            h.last_update_ms = now_ms;
            return;
        }

        self.hyps.push(Hypothesis {
            edge,
            confidence: (0.5 + delta_conf).clamp(0.0, 1.0),
            evidence_count: 1,
            last_update_ms: now_ms,
        });
    }

    pub fn is_acyclic(&self) -> bool {
        let mut adjacency: HashMap<VarId, Vec<VarId>> = HashMap::new();
        let mut nodes: HashSet<VarId> = self.vars.iter().copied().collect();

        for h in self
            .hyps
            .iter()
            .filter(|h| h.confidence >= ACTIVE_CONFIDENCE_TH)
        {
            nodes.insert(h.edge.from);
            nodes.insert(h.edge.to);
            adjacency.entry(h.edge.from).or_default().push(h.edge.to);
        }

        let mut visiting = HashSet::new();
        let mut visited = HashSet::new();

        for node in nodes {
            if Self::dfs_has_cycle(node, &adjacency, &mut visiting, &mut visited) {
                return false;
            }
        }
        true
    }

    fn dfs_has_cycle(
        node: VarId,
        adjacency: &HashMap<VarId, Vec<VarId>>,
        visiting: &mut HashSet<VarId>,
        visited: &mut HashSet<VarId>,
    ) -> bool {
        if visited.contains(&node) {
            return false;
        }
        if !visiting.insert(node) {
            return true;
        }

        if let Some(neighbors) = adjacency.get(&node) {
            for &next in neighbors {
                if Self::dfs_has_cycle(next, adjacency, visiting, visited) {
                    return true;
                }
            }
        }

        visiting.remove(&node);
        visited.insert(node);
        false
    }

    pub fn top_edges(&self, k: usize) -> Vec<Hypothesis> {
        let mut hyps = self.hyps.clone();
        hyps.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(Ordering::Equal)
                .then_with(|| b.evidence_count.cmp(&a.evidence_count))
        });
        hyps.truncate(k);
        hyps
    }

    pub fn simulate_intervention(&self, iv: Intervention) -> Counterfactual {
        let mut affected = self
            .hyps
            .iter()
            .filter(|h| h.edge.from == iv.var && h.confidence >= ACTIVE_CONFIDENCE_TH)
            .map(|h| (h.edge.to, (iv.set_to * h.confidence).clamp(-1.0, 1.0)))
            .collect::<Vec<_>>();

        affected.sort_by_key(|(var, _)| *var);
        Counterfactual { affected }
    }
}
