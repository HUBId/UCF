#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_types::Digest32;

const MAX_INFLUENCE: i32 = 10_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NodeId {
    Perception,
    Memory,
    Attention,
    Learning,
    Structure,
    Ai,
    Ssm,
    Cde,
    Nsr,
    Geist,
    Replay,
    Output,
    BlueBrain,
    Unknown(u16),
}

impl NodeId {
    pub const ORDERED: [NodeId; 13] = [
        NodeId::Perception,
        NodeId::Memory,
        NodeId::Attention,
        NodeId::Learning,
        NodeId::Structure,
        NodeId::Ai,
        NodeId::Ssm,
        NodeId::Cde,
        NodeId::Nsr,
        NodeId::Geist,
        NodeId::Replay,
        NodeId::Output,
        NodeId::BlueBrain,
    ];

    pub fn to_u16(self) -> u16 {
        match self {
            NodeId::Perception => 1,
            NodeId::Memory => 2,
            NodeId::Attention => 3,
            NodeId::Learning => 4,
            NodeId::Structure => 5,
            NodeId::Ai => 6,
            NodeId::Ssm => 7,
            NodeId::Cde => 8,
            NodeId::Nsr => 9,
            NodeId::Geist => 10,
            NodeId::Replay => 11,
            NodeId::Output => 12,
            NodeId::BlueBrain => 13,
            NodeId::Unknown(value) => value,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EdgeRuleId {
    Linear,
    Homeostatic,
    SurpriseGated,
    DriftGated,
    Unknown(u16),
}

impl EdgeRuleId {
    fn to_u16(self) -> u16 {
        match self {
            EdgeRuleId::Linear => 1,
            EdgeRuleId::Homeostatic => 2,
            EdgeRuleId::SurpriseGated => 3,
            EdgeRuleId::DriftGated => 4,
            EdgeRuleId::Unknown(value) => value,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InfluenceEdge {
    pub from: NodeId,
    pub to: NodeId,
    pub delay: u8,
    pub gain: i16,
    pub noise: u16,
    pub rule: EdgeRuleId,
    pub commit: Digest32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InfluencePulse {
    pub value: i16,
    pub commit: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelayLine {
    pub delay: u8,
    pub buf: Vec<InfluencePulse>,
    pub head: usize,
    pub commit: Digest32,
}

impl DelayLine {
    fn new(delay: u8, commit: Digest32) -> Self {
        let size = delay as usize + 1;
        let zero = InfluencePulse { value: 0, commit };
        Self {
            delay,
            buf: vec![zero; size],
            head: 0,
            commit,
        }
    }

    fn push(&mut self, pulse: InfluencePulse) -> InfluencePulse {
        if self.delay == 0 {
            self.buf[self.head] = pulse;
            return pulse;
        }
        let emit_index = (self.head + 1) % self.buf.len();
        let emit = self.buf[emit_index];
        self.buf[self.head] = pulse;
        self.head = emit_index;
        emit
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InfluenceState {
    pub edges: Vec<InfluenceEdge>,
    pub lines: Vec<DelayLine>,
    pub root_commit: Digest32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InfluenceInputs {
    pub cycle_id: u64,
    pub phase_commit: Digest32,
    pub spike_root: Digest32,
    pub drift: u16,
    pub surprise: u16,
    pub risk: u16,
    pub attn_gain: u16,
    pub commit: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InfluenceOutputs {
    pub node_in: Vec<(NodeId, i16)>,
    pub commit: Digest32,
}

impl InfluenceOutputs {
    pub fn node_value(&self, node: NodeId) -> i16 {
        self.node_in
            .iter()
            .find_map(|(id, value)| (*id == node).then_some(*value))
            .unwrap_or(0)
    }
}

impl InfluenceState {
    pub fn new_default() -> Self {
        let edges = default_edges();
        let lines: Vec<DelayLine> = edges
            .iter()
            .map(|edge| DelayLine::new(edge.delay, edge.commit))
            .collect();
        let root_commit = hash_edges(&edges);
        Self {
            edges,
            lines,
            root_commit,
        }
    }

    pub fn tick(&mut self, inp: &InfluenceInputs) -> InfluenceOutputs {
        let mut sums: Vec<(NodeId, i32)> =
            NodeId::ORDERED.iter().map(|node| (*node, 0i32)).collect();

        for (idx, edge) in self.edges.iter().enumerate() {
            let base = base_signal(edge.from, inp);
            let gated = apply_rule(edge.rule, base, inp);
            let gain_component = (i32::from(gated) * i32::from(edge.gain)) / 10_000;
            let noise = noise_sample(edge, inp);
            let value = clamp_influence(gain_component + i32::from(noise));
            let pulse = InfluencePulse {
                value,
                commit: inp.commit,
            };
            let emitted = self.lines[idx].push(pulse);

            if let Some((_, total)) = sums.iter_mut().find(|(node, _)| *node == edge.to) {
                *total = total.saturating_add(i32::from(emitted.value));
            } else {
                sums.push((edge.to, i32::from(emitted.value)));
            }
        }

        sums.sort_by_key(|(node, _)| node.to_u16());
        let node_in: Vec<(NodeId, i16)> = sums
            .into_iter()
            .map(|(node, total)| (node, clamp_influence(total)))
            .collect();
        let commit = outputs_commit(&node_in, self.root_commit, inp.commit);
        InfluenceOutputs { node_in, commit }
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

fn base_signal(node: NodeId, inp: &InfluenceInputs) -> i16 {
    match node {
        NodeId::Perception => inp.surprise.min(10_000) as i16,
        NodeId::Memory => (inp.attn_gain / 2) as i16,
        NodeId::Attention => inp.attn_gain.min(10_000) as i16,
        NodeId::Learning => inp.drift.min(10_000) as i16,
        NodeId::Structure => (inp.drift / 2) as i16,
        NodeId::Ai => (inp.risk / 2) as i16,
        NodeId::Ssm => (inp.risk / 3) as i16,
        NodeId::Cde => (inp.surprise / 2) as i16,
        NodeId::Nsr => inp.risk.min(10_000) as i16,
        NodeId::Geist => inp.drift.min(10_000) as i16,
        NodeId::Replay => (inp.attn_gain / 3) as i16,
        NodeId::Output => (inp.risk / 2) as i16,
        NodeId::BlueBrain => (inp.attn_gain / 4) as i16,
        NodeId::Unknown(_) => 0,
    }
}

fn apply_rule(rule: EdgeRuleId, base: i16, inp: &InfluenceInputs) -> i16 {
    match rule {
        EdgeRuleId::Linear => base,
        EdgeRuleId::Homeostatic => {
            let negate = inp.risk >= 6000 || inp.surprise >= 7000;
            if negate {
                base.saturating_neg()
            } else {
                base
            }
        }
        EdgeRuleId::SurpriseGated => {
            let factor = i32::from(inp.surprise.min(10_000));
            ((i32::from(base) * factor) / 10_000).clamp(i32::from(i16::MIN), i32::from(i16::MAX))
                as i16
        }
        EdgeRuleId::DriftGated => {
            let factor = i32::from(inp.drift.min(10_000));
            ((i32::from(base) * factor) / 10_000).clamp(i32::from(i16::MIN), i32::from(i16::MAX))
                as i16
        }
        EdgeRuleId::Unknown(_) => base,
    }
}

fn clamp_influence(value: i32) -> i16 {
    value
        .clamp(-MAX_INFLUENCE, MAX_INFLUENCE)
        .clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

fn noise_sample(edge: &InfluenceEdge, inp: &InfluenceInputs) -> i16 {
    if edge.noise == 0 {
        return 0;
    }
    let seed = hash_pair(inp.commit, edge.commit);
    let sample = prng16(&seed) as i32;
    let span = i32::from(edge.noise) * 2 + 1;
    let offset = sample % span;
    (offset - i32::from(edge.noise)) as i16
}

fn prng16(seed: &Digest32) -> u16 {
    let bytes = seed.as_bytes();
    let mut state = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    (state & 0xffff) as u16
}

fn hash_pair(a: Digest32, b: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.influence.seed.v1");
    hasher.update(a.as_bytes());
    hasher.update(b.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn edge_commit(edge: &InfluenceEdge) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.influence.edge.v1");
    hasher.update(&edge.from.to_u16().to_be_bytes());
    hasher.update(&edge.to.to_u16().to_be_bytes());
    hasher.update(&[edge.delay]);
    hasher.update(&edge.gain.to_be_bytes());
    hasher.update(&edge.noise.to_be_bytes());
    hasher.update(&edge.rule.to_u16().to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn hash_edges(edges: &[InfluenceEdge]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.influence.root.v1");
    for edge in edges {
        hasher.update(edge.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn outputs_commit(
    node_in: &[(NodeId, i16)],
    root_commit: Digest32,
    input_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.influence.outputs.v1");
    hasher.update(root_commit.as_bytes());
    hasher.update(input_commit.as_bytes());
    for (node, value) in node_in {
        hasher.update(&node.to_u16().to_be_bytes());
        hasher.update(&value.to_be_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn default_edges() -> Vec<InfluenceEdge> {
    let mut edges = vec![
        InfluenceEdge {
            from: NodeId::Perception,
            to: NodeId::Memory,
            delay: 1,
            gain: 1800,
            noise: 140,
            rule: EdgeRuleId::Linear,
            commit: Digest32::new([0u8; 32]),
        },
        InfluenceEdge {
            from: NodeId::Memory,
            to: NodeId::Attention,
            delay: 1,
            gain: 1600,
            noise: 160,
            rule: EdgeRuleId::Linear,
            commit: Digest32::new([0u8; 32]),
        },
        InfluenceEdge {
            from: NodeId::Attention,
            to: NodeId::Learning,
            delay: 0,
            gain: 1400,
            noise: 120,
            rule: EdgeRuleId::Linear,
            commit: Digest32::new([0u8; 32]),
        },
        InfluenceEdge {
            from: NodeId::Learning,
            to: NodeId::Structure,
            delay: 2,
            gain: 1200,
            noise: 180,
            rule: EdgeRuleId::Linear,
            commit: Digest32::new([0u8; 32]),
        },
        InfluenceEdge {
            from: NodeId::Perception,
            to: NodeId::Attention,
            delay: 0,
            gain: 1300,
            noise: 220,
            rule: EdgeRuleId::SurpriseGated,
            commit: Digest32::new([0u8; 32]),
        },
        InfluenceEdge {
            from: NodeId::Geist,
            to: NodeId::Replay,
            delay: 1,
            gain: 1500,
            noise: 200,
            rule: EdgeRuleId::DriftGated,
            commit: Digest32::new([0u8; 32]),
        },
        InfluenceEdge {
            from: NodeId::Nsr,
            to: NodeId::Output,
            delay: 0,
            gain: 1100,
            noise: 160,
            rule: EdgeRuleId::Homeostatic,
            commit: Digest32::new([0u8; 32]),
        },
        InfluenceEdge {
            from: NodeId::BlueBrain,
            to: NodeId::Attention,
            delay: 1,
            gain: 1000,
            noise: 140,
            rule: EdgeRuleId::Homeostatic,
            commit: Digest32::new([0u8; 32]),
        },
        InfluenceEdge {
            from: NodeId::Ssm,
            to: NodeId::Perception,
            delay: 1,
            gain: 900,
            noise: 120,
            rule: EdgeRuleId::Linear,
            commit: Digest32::new([0u8; 32]),
        },
        InfluenceEdge {
            from: NodeId::Replay,
            to: NodeId::Learning,
            delay: 1,
            gain: 800,
            noise: 100,
            rule: EdgeRuleId::Linear,
            commit: Digest32::new([0u8; 32]),
        },
        InfluenceEdge {
            from: NodeId::Structure,
            to: NodeId::Memory,
            delay: 2,
            gain: 700,
            noise: 100,
            rule: EdgeRuleId::Linear,
            commit: Digest32::new([0u8; 32]),
        },
        InfluenceEdge {
            from: NodeId::Ai,
            to: NodeId::Attention,
            delay: 0,
            gain: 600,
            noise: 80,
            rule: EdgeRuleId::Linear,
            commit: Digest32::new([0u8; 32]),
        },
    ];

    for edge in &mut edges {
        edge.commit = edge_commit(edge);
    }

    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_inputs(commit: Digest32) -> InfluenceInputs {
        InfluenceInputs {
            cycle_id: 1,
            phase_commit: Digest32::new([1u8; 32]),
            spike_root: Digest32::new([2u8; 32]),
            drift: 1000,
            surprise: 2000,
            risk: 3000,
            attn_gain: 4000,
            commit,
        }
    }

    #[test]
    fn tick_is_deterministic() {
        let mut state_a = InfluenceState::new_default();
        let mut state_b = InfluenceState::new_default();
        let commit = Digest32::new([3u8; 32]);
        let inputs = base_inputs(commit);
        let first = state_a.tick(&inputs);
        let second = state_b.tick(&inputs);
        assert_eq!(first, second);
    }

    #[test]
    fn delay_line_honors_delay() {
        let edge = InfluenceEdge {
            from: NodeId::Perception,
            to: NodeId::Memory,
            delay: 2,
            gain: 10_000,
            noise: 0,
            rule: EdgeRuleId::Linear,
            commit: Digest32::new([9u8; 32]),
        };
        let mut state = InfluenceState {
            edges: vec![edge.clone()],
            lines: vec![DelayLine::new(edge.delay, edge.commit)],
            root_commit: Digest32::new([7u8; 32]),
        };
        let commit = Digest32::new([5u8; 32]);
        let mut inputs = base_inputs(commit);
        inputs.surprise = 10_000;
        inputs.drift = 0;
        inputs.attn_gain = 0;

        let first = state.tick(&inputs).node_value(NodeId::Memory);
        let second = state.tick(&inputs).node_value(NodeId::Memory);
        let third = state.tick(&inputs).node_value(NodeId::Memory);

        assert_eq!(first, 0);
        assert_eq!(second, 0);
        assert!(third > 0);
    }

    #[test]
    fn noise_is_deterministic_for_same_seed() {
        let seed = hash_pair(Digest32::new([1u8; 32]), Digest32::new([2u8; 32]));
        let sample_a = prng16(&seed);
        let sample_b = prng16(&seed);
        assert_eq!(sample_a, sample_b);
    }
}
