#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_types::Digest32;

const MAX_EDGES: usize = 48;
const MAX_LAG: u8 = 8;
const MAX_PULSES: usize = 32;
const MAX_INFLUENCE: i32 = 10_000;
const MAX_WEIGHT: i16 = 10_000;
const NSR_NODE_CODE: u16 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InfluenceNodeId {
    Perception,
    WorldModel,
    WorkingMemory,
    ReplayPressure,
    SleepDrive,
    AttentionGain,
    LearningRate,
    StructuralPlasticity,
    Coherence,
    Integration,
    Risk,
    Drift,
    Surprise,
    OutputSuppression,
    BlueBrainArousal,
    Unknown(u16),
}

impl InfluenceNodeId {
    pub const ORDERED: [InfluenceNodeId; 15] = [
        InfluenceNodeId::Perception,
        InfluenceNodeId::WorldModel,
        InfluenceNodeId::WorkingMemory,
        InfluenceNodeId::ReplayPressure,
        InfluenceNodeId::SleepDrive,
        InfluenceNodeId::AttentionGain,
        InfluenceNodeId::LearningRate,
        InfluenceNodeId::StructuralPlasticity,
        InfluenceNodeId::Coherence,
        InfluenceNodeId::Integration,
        InfluenceNodeId::Risk,
        InfluenceNodeId::Drift,
        InfluenceNodeId::Surprise,
        InfluenceNodeId::OutputSuppression,
        InfluenceNodeId::BlueBrainArousal,
    ];

    pub const NSR_NODE: InfluenceNodeId = InfluenceNodeId::Unknown(NSR_NODE_CODE);

    pub fn to_u16(self) -> u16 {
        match self {
            InfluenceNodeId::Perception => 1,
            InfluenceNodeId::WorldModel => 2,
            InfluenceNodeId::WorkingMemory => 3,
            InfluenceNodeId::ReplayPressure => 4,
            InfluenceNodeId::SleepDrive => 5,
            InfluenceNodeId::AttentionGain => 6,
            InfluenceNodeId::LearningRate => 7,
            InfluenceNodeId::StructuralPlasticity => 8,
            InfluenceNodeId::Coherence => 9,
            InfluenceNodeId::Integration => 10,
            InfluenceNodeId::Risk => 11,
            InfluenceNodeId::Drift => 12,
            InfluenceNodeId::Surprise => 13,
            InfluenceNodeId::OutputSuppression => 14,
            InfluenceNodeId::BlueBrainArousal => 15,
            InfluenceNodeId::Unknown(value) => value,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InfluenceEdge {
    pub from: InfluenceNodeId,
    pub to: InfluenceNodeId,
    pub weight: i16,
    pub lag: u8,
    pub adapt: u16,
    pub gate: u16,
    pub commit: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InfluencePulse {
    pub cycle_id: u64,
    pub src: InfluenceNodeId,
    pub dst: InfluenceNodeId,
    pub value: i16,
    pub lag: u8,
    pub phase_commit: Digest32,
    pub commit: Digest32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InfluenceInputs {
    pub cycle_id: u64,
    pub phase_commit: Digest32,
    pub coherence_plv: u16,
    pub phi_proxy: u16,
    pub ssm_salience: u16,
    pub ssm_novelty: u16,
    pub ncde_energy: u16,
    pub nsr_verdict: u8,
    pub risk: u16,
    pub drift: u16,
    pub surprise: u16,
    pub cde_commit: Option<Digest32>,
    pub sle_self_symbol: Option<Digest32>,
    pub rsa_applied: bool,
    pub commit: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InfluenceOutputs {
    pub cycle_id: u64,
    pub pulses_root: Digest32,
    pub pulses: Vec<InfluencePulse>,
    pub node_values: Vec<(InfluenceNodeId, i16)>,
    pub commit: Digest32,
}

impl InfluenceOutputs {
    pub fn node_value(&self, node: InfluenceNodeId) -> i16 {
        self.node_values
            .iter()
            .find_map(|(id, value)| (*id == node).then_some(*value))
            .unwrap_or(0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InfluenceGraphV2 {
    pub edges: Vec<InfluenceEdge>,
    pub ringbuf: Vec<Vec<i16>>,
    pub commit: Digest32,
}

impl InfluenceGraphV2 {
    pub fn new_default() -> Self {
        let edges = default_edges();
        let ringbuf = vec![vec![0i16; MAX_LAG as usize]; all_nodes().len()];
        let commit = hash_edges(&edges);
        Self {
            edges,
            ringbuf,
            commit,
        }
    }

    pub fn tick(&mut self, inp: &InfluenceInputs) -> InfluenceOutputs {
        let nodes = all_nodes();
        let mut base_values = vec![0i32; nodes.len()];
        for (idx, node) in nodes.iter().enumerate() {
            base_values[idx] = i32::from(base_signal(*node, inp));
        }

        let mut sums = base_values.clone();
        let mut pulse_rank: Vec<(InfluencePulse, Digest32, i16)> = Vec::new();

        for edge in &self.edges {
            let source_value =
                source_value(edge.from, edge.lag, &nodes, &base_values, &self.ringbuf);
            let pulse_value = compute_pulse(source_value, edge, inp);
            let pulse = InfluencePulse {
                cycle_id: inp.cycle_id,
                src: edge.from,
                dst: edge.to,
                value: pulse_value,
                lag: edge.lag,
                phase_commit: inp.phase_commit,
                commit: pulse_commit(inp.cycle_id, inp.phase_commit, edge, pulse_value),
            };
            if let Some(dst_idx) = node_index(edge.to, &nodes) {
                sums[dst_idx] = sums[dst_idx].saturating_add(i32::from(pulse_value));
            }
            pulse_rank.push((pulse, edge.commit, pulse_value));
        }

        let mut node_values = Vec::with_capacity(nodes.len());
        for (idx, node) in nodes.iter().enumerate() {
            node_values.push((*node, clamp_influence(sums[idx])));
        }

        for (idx, buf) in self.ringbuf.iter_mut().enumerate() {
            buf.insert(0, node_values[idx].1);
            if buf.len() > MAX_LAG as usize {
                buf.truncate(MAX_LAG as usize);
            }
        }

        pulse_rank.sort_by(
            |(a_pulse, a_commit, a_value), (b_pulse, b_commit, b_value)| {
                let a_mag = i16::abs(*a_value);
                let b_mag = i16::abs(*b_value);
                b_mag
                    .cmp(&a_mag)
                    .then_with(|| a_commit.as_bytes().cmp(b_commit.as_bytes()))
                    .then_with(|| a_pulse.commit.as_bytes().cmp(b_pulse.commit.as_bytes()))
            },
        );

        pulse_rank.truncate(MAX_PULSES);
        let pulses: Vec<InfluencePulse> =
            pulse_rank.into_iter().map(|(pulse, _, _)| pulse).collect();
        let pulses_root = hash_pulses(&pulses);
        let commit = outputs_commit(inp, &pulses_root, &node_values, self.commit);

        self.adapt_edges(inp);

        InfluenceOutputs {
            cycle_id: inp.cycle_id,
            pulses_root,
            pulses,
            node_values,
            commit,
        }
    }

    fn adapt_edges(&mut self, inp: &InfluenceInputs) {
        let objective = i32::from(inp.coherence_plv)
            .saturating_add(i32::from(inp.phi_proxy))
            .saturating_sub(i32::from(inp.risk))
            .saturating_sub(i32::from(inp.drift));
        let direction = if objective >= 0 { 1i16 } else { -1i16 };

        for edge in &mut self.edges {
            if edge.adapt == 0 {
                continue;
            }
            let p = adapt_probability(inp.commit, edge.commit);
            if p < edge.adapt {
                let step = i16::try_from(edge.adapt / 2000 + 10).unwrap_or(10);
                let delta = step.saturating_mul(direction);
                let next = i32::from(edge.weight) + i32::from(delta);
                edge.weight = clamp_weight(next);
                edge.commit = edge_commit(edge);
            }
        }
        self.commit = hash_edges(&self.edges);
    }
}

fn all_nodes() -> Vec<InfluenceNodeId> {
    let mut nodes = InfluenceNodeId::ORDERED.to_vec();
    nodes.push(InfluenceNodeId::NSR_NODE);
    nodes
}

fn node_index(node: InfluenceNodeId, nodes: &[InfluenceNodeId]) -> Option<usize> {
    nodes.iter().position(|id| *id == node)
}

fn base_signal(node: InfluenceNodeId, inp: &InfluenceInputs) -> i16 {
    match node {
        InfluenceNodeId::Perception => centered_metric(inp.ssm_novelty),
        InfluenceNodeId::WorldModel => centered_metric(inp.ncde_energy),
        InfluenceNodeId::WorkingMemory => centered_metric(inp.ssm_salience),
        InfluenceNodeId::ReplayPressure => 0,
        InfluenceNodeId::SleepDrive => 0,
        InfluenceNodeId::AttentionGain => 0,
        InfluenceNodeId::LearningRate => 0,
        InfluenceNodeId::StructuralPlasticity => 0,
        InfluenceNodeId::Coherence => centered_metric(inp.coherence_plv),
        InfluenceNodeId::Integration => centered_metric(inp.phi_proxy),
        InfluenceNodeId::Risk => centered_metric(inp.risk),
        InfluenceNodeId::Drift => centered_metric(inp.drift),
        InfluenceNodeId::Surprise => centered_metric(inp.surprise),
        InfluenceNodeId::OutputSuppression => 0,
        InfluenceNodeId::BlueBrainArousal => 0,
        InfluenceNodeId::Unknown(code) if code == NSR_NODE_CODE => nsr_value(inp.nsr_verdict),
        InfluenceNodeId::Unknown(_) => 0,
    }
}

fn centered_metric(value: u16) -> i16 {
    let value = value.min(10_000) as i32;
    clamp_influence((value - 5_000).saturating_mul(2))
}

fn nsr_value(verdict: u8) -> i16 {
    match verdict {
        1 => 4_000,
        2 => 8_000,
        _ => 0,
    }
}

fn source_value(
    node: InfluenceNodeId,
    lag: u8,
    nodes: &[InfluenceNodeId],
    base_values: &[i32],
    ringbuf: &[Vec<i16>],
) -> i16 {
    let Some(idx) = node_index(node, nodes) else {
        return 0;
    };
    if lag == 0 {
        return clamp_influence(base_values[idx]);
    }
    let lag_index = (lag - 1) as usize;
    ringbuf
        .get(idx)
        .and_then(|buf| buf.get(lag_index))
        .copied()
        .unwrap_or(0)
}

fn compute_pulse(source: i16, edge: &InfluenceEdge, inp: &InfluenceInputs) -> i16 {
    if source == 0 || edge.weight == 0 {
        return 0;
    }
    let base = (i32::from(source) * i32::from(edge.weight)) / i32::from(MAX_WEIGHT);
    let cohesion = (i32::from(inp.coherence_plv) + i32::from(inp.phi_proxy)) / 2;
    let gate = (i32::from(edge.gate) * cohesion) / 10_000;
    let mut pulse = (base * gate) / 10_000;
    let risk = i32::from(inp.risk).min(10_000);
    let nsr_penalty = match inp.nsr_verdict {
        1 => 3_000,
        2 => 7_000,
        _ => 0,
    };
    let risk_factor = (risk + nsr_penalty).min(10_000);
    if pulse > 0 {
        if edge.to == InfluenceNodeId::OutputSuppression {
            pulse = (pulse * (10_000 + risk_factor / 2)) / 10_000;
        } else {
            pulse = (pulse * (10_000 - risk_factor)) / 10_000;
        }
    } else if pulse < 0 {
        pulse = (pulse * (10_000 + risk_factor / 2)) / 10_000;
    }
    clamp_influence(pulse)
}

fn clamp_influence(value: i32) -> i16 {
    value
        .clamp(-MAX_INFLUENCE, MAX_INFLUENCE)
        .clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

fn clamp_weight(value: i32) -> i16 {
    value
        .clamp(-i32::from(MAX_WEIGHT), i32::from(MAX_WEIGHT))
        .clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

fn adapt_probability(input_commit: Digest32, edge_commit: Digest32) -> u16 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.influence.adapt.v2");
    hasher.update(input_commit.as_bytes());
    hasher.update(edge_commit.as_bytes());
    let digest = Digest32::new(*hasher.finalize().as_bytes());
    let bytes = digest.as_bytes();
    let sample = u16::from_be_bytes([bytes[0], bytes[1]]);
    sample % 10_001
}

fn pulse_commit(
    cycle_id: u64,
    phase_commit: Digest32,
    edge: &InfluenceEdge,
    value: i16,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.influence.pulse.v2");
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(phase_commit.as_bytes());
    hasher.update(edge.commit.as_bytes());
    hasher.update(&edge.from.to_u16().to_be_bytes());
    hasher.update(&edge.to.to_u16().to_be_bytes());
    hasher.update(&edge.lag.to_be_bytes());
    hasher.update(&value.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn outputs_commit(
    inp: &InfluenceInputs,
    pulses_root: &Digest32,
    node_values: &[(InfluenceNodeId, i16)],
    graph_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.influence.outputs.v2");
    hasher.update(graph_commit.as_bytes());
    hasher.update(inp.commit.as_bytes());
    hasher.update(pulses_root.as_bytes());
    hasher.update(&inp.cycle_id.to_be_bytes());
    for (node, value) in node_values {
        hasher.update(&node.to_u16().to_be_bytes());
        hasher.update(&value.to_be_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn hash_pulses(pulses: &[InfluencePulse]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.influence.pulses_root.v2");
    for pulse in pulses {
        hasher.update(pulse.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn edge_commit(edge: &InfluenceEdge) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.influence.edge.v2");
    hasher.update(&edge.from.to_u16().to_be_bytes());
    hasher.update(&edge.to.to_u16().to_be_bytes());
    hasher.update(&edge.weight.to_be_bytes());
    hasher.update(&edge.lag.to_be_bytes());
    hasher.update(&edge.adapt.to_be_bytes());
    hasher.update(&edge.gate.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn hash_edges(edges: &[InfluenceEdge]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.influence.graph.v2");
    for edge in edges {
        hasher.update(edge.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn default_edges() -> Vec<InfluenceEdge> {
    let mut edges = vec![
        InfluenceEdge {
            from: InfluenceNodeId::Perception,
            to: InfluenceNodeId::WorkingMemory,
            weight: 3200,
            lag: 0,
            adapt: 1200,
            gate: 7500,
            commit: Digest32::new([0u8; 32]),
        },
        InfluenceEdge {
            from: InfluenceNodeId::WorkingMemory,
            to: InfluenceNodeId::AttentionGain,
            weight: 2800,
            lag: 1,
            adapt: 1400,
            gate: 7200,
            commit: Digest32::new([0u8; 32]),
        },
        InfluenceEdge {
            from: InfluenceNodeId::AttentionGain,
            to: InfluenceNodeId::LearningRate,
            weight: 2600,
            lag: 1,
            adapt: 1600,
            gate: 7000,
            commit: Digest32::new([0u8; 32]),
        },
        InfluenceEdge {
            from: InfluenceNodeId::LearningRate,
            to: InfluenceNodeId::StructuralPlasticity,
            weight: 2400,
            lag: 2,
            adapt: 1800,
            gate: 6800,
            commit: Digest32::new([0u8; 32]),
        },
        InfluenceEdge {
            from: InfluenceNodeId::StructuralPlasticity,
            to: InfluenceNodeId::Coherence,
            weight: 2200,
            lag: 2,
            adapt: 2000,
            gate: 7000,
            commit: Digest32::new([0u8; 32]),
        },
        InfluenceEdge {
            from: InfluenceNodeId::Coherence,
            to: InfluenceNodeId::Integration,
            weight: 2500,
            lag: 0,
            adapt: 1800,
            gate: 7600,
            commit: Digest32::new([0u8; 32]),
        },
        InfluenceEdge {
            from: InfluenceNodeId::Integration,
            to: InfluenceNodeId::SleepDrive,
            weight: -2100,
            lag: 1,
            adapt: 900,
            gate: 6400,
            commit: Digest32::new([0u8; 32]),
        },
        InfluenceEdge {
            from: InfluenceNodeId::Surprise,
            to: InfluenceNodeId::ReplayPressure,
            weight: 3000,
            lag: 0,
            adapt: 1400,
            gate: 7200,
            commit: Digest32::new([0u8; 32]),
        },
        InfluenceEdge {
            from: InfluenceNodeId::Drift,
            to: InfluenceNodeId::SleepDrive,
            weight: 2600,
            lag: 1,
            adapt: 1200,
            gate: 7000,
            commit: Digest32::new([0u8; 32]),
        },
        InfluenceEdge {
            from: InfluenceNodeId::Risk,
            to: InfluenceNodeId::OutputSuppression,
            weight: 3200,
            lag: 0,
            adapt: 1600,
            gate: 8500,
            commit: Digest32::new([0u8; 32]),
        },
        InfluenceEdge {
            from: InfluenceNodeId::NSR_NODE,
            to: InfluenceNodeId::OutputSuppression,
            weight: 3800,
            lag: 0,
            adapt: 800,
            gate: 9000,
            commit: Digest32::new([0u8; 32]),
        },
        InfluenceEdge {
            from: InfluenceNodeId::ReplayPressure,
            to: InfluenceNodeId::SleepDrive,
            weight: 2400,
            lag: 0,
            adapt: 1200,
            gate: 6800,
            commit: Digest32::new([0u8; 32]),
        },
        InfluenceEdge {
            from: InfluenceNodeId::SleepDrive,
            to: InfluenceNodeId::AttentionGain,
            weight: -2600,
            lag: 1,
            adapt: 1000,
            gate: 6500,
            commit: Digest32::new([0u8; 32]),
        },
    ];

    edges.truncate(MAX_EDGES);
    for edge in &mut edges {
        edge.lag = edge.lag.min(MAX_LAG);
        edge.weight = clamp_weight(i32::from(edge.weight));
        edge.adapt = edge.adapt.min(10_000);
        edge.gate = edge.gate.min(10_000);
        edge.commit = edge_commit(edge);
    }

    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_inputs(commit: Digest32) -> InfluenceInputs {
        InfluenceInputs {
            cycle_id: 7,
            phase_commit: Digest32::new([3u8; 32]),
            coherence_plv: 5200,
            phi_proxy: 4700,
            ssm_salience: 4200,
            ssm_novelty: 6400,
            ncde_energy: 5100,
            nsr_verdict: 0,
            risk: 2100,
            drift: 2300,
            surprise: 6500,
            cde_commit: Some(Digest32::new([9u8; 32])),
            sle_self_symbol: Some(Digest32::new([8u8; 32])),
            rsa_applied: false,
            commit,
        }
    }

    #[test]
    fn tick_is_deterministic() {
        let mut graph_a = InfluenceGraphV2::new_default();
        let mut graph_b = InfluenceGraphV2::new_default();
        let commit = Digest32::new([7u8; 32]);
        let inputs = base_inputs(commit);
        let first = graph_a.tick(&inputs);
        let second = graph_b.tick(&inputs);
        assert_eq!(first.commit, second.commit);
        assert_eq!(first.pulses_root, second.pulses_root);
        assert_eq!(first.node_values, second.node_values);
    }

    #[test]
    fn lag_respects_working_memory_to_attention() {
        let mut graph = InfluenceGraphV2::new_default();
        let mut inputs = base_inputs(Digest32::new([4u8; 32]));
        inputs.ssm_novelty = 10_000;
        inputs.ssm_salience = 5_000;
        inputs.surprise = 0;
        let first = graph.tick(&inputs);
        let second = graph.tick(&inputs);
        let attention_first = first.node_value(InfluenceNodeId::AttentionGain);
        let attention_second = second.node_value(InfluenceNodeId::AttentionGain);
        assert_eq!(attention_first, 0);
        assert!(attention_second > 0);
    }

    #[test]
    fn adaptive_update_is_deterministic() {
        let mut graph_a = InfluenceGraphV2::new_default();
        let mut graph_b = InfluenceGraphV2::new_default();
        let inputs = base_inputs(Digest32::new([11u8; 32]));
        graph_a.tick(&inputs);
        graph_b.tick(&inputs);
        graph_a.tick(&inputs);
        graph_b.tick(&inputs);
        assert_eq!(graph_a.edges, graph_b.edges);
    }

    #[test]
    fn high_risk_increases_output_suppression() {
        let mut graph = InfluenceGraphV2::new_default();
        let mut inputs = base_inputs(Digest32::new([13u8; 32]));
        inputs.risk = 9_500;
        inputs.nsr_verdict = 2;
        let outputs = graph.tick(&inputs);
        let suppression = outputs.node_value(InfluenceNodeId::OutputSuppression);
        assert!(suppression > 0);
    }
}
