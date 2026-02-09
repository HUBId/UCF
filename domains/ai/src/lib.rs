#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_spikebus::SpikeKind;
use ucf_types::Digest32;

const AI_INPUT_DOMAIN: &[u8] = b"ucf.ai.host.input.v1";
const AI_INTERNAL_DOMAIN: &[u8] = b"ucf.ai.host.internal.v1";
const AI_OUTPUT_DOMAIN: &[u8] = b"ucf.ai.host.output.candidate.v1";
const AI_FEATURE_DOMAIN: &[u8] = b"ucf.ai.host.feature.event.v1";
const AI_HOST_OUTPUTS_DOMAIN: &[u8] = b"ucf.ai.host.outputs.v1";
const AI_MOCK_THOUGHT_DOMAIN: &[u8] = b"ucf.ai.host.mock.thought.v1";
const AI_MOCK_FEATURE_DOMAIN: &[u8] = b"ucf.ai.host.mock.feature.v1";
const AI_MOCK_OUTPUT_DOMAIN: &[u8] = b"ucf.ai.host.mock.output.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelLabel {
    External,
    ThoughtOnly,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AiInputFrame {
    pub cycle_id: u64,
    pub external_commit: Digest32,
    pub phase_commit: Digest32,
    pub gamma_bucket: u8,
    pub global_plv: u16,
    pub ssm_state_digest: Digest32,
    pub world_state: Digest32,
    pub policy_snapshot_commit: Digest32,
    pub attention_cap: u16,
    pub learning_cap: u16,
    pub commit: Digest32,
}

impl AiInputFrame {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cycle_id: u64,
        external_commit: Digest32,
        phase_commit: Digest32,
        gamma_bucket: u8,
        global_plv: u16,
        ssm_state_digest: Digest32,
        world_state: Digest32,
        policy_snapshot_commit: Digest32,
        attention_cap: u16,
        learning_cap: u16,
    ) -> Self {
        let mut frame = Self {
            cycle_id,
            external_commit,
            phase_commit,
            gamma_bucket,
            global_plv,
            ssm_state_digest,
            world_state,
            policy_snapshot_commit,
            attention_cap,
            learning_cap,
            commit: Digest32::new([0u8; 32]),
        };
        frame.commit = commit_input_frame(&frame);
        frame
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AiInternalThought {
    pub cycle_id: u64,
    pub thought_root: Digest32,
    pub salience: u16,
    pub commit: Digest32,
}

impl AiInternalThought {
    pub fn new(cycle_id: u64, thought_root: Digest32, salience: u16) -> Self {
        let mut thought = Self {
            cycle_id,
            thought_root,
            salience,
            commit: Digest32::new([0u8; 32]),
        };
        thought.commit = commit_internal_thought(&thought);
        thought
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AiOutputCandidate {
    pub cycle_id: u64,
    pub label: ChannelLabel,
    pub payload_commit: Digest32,
    pub confidence: u16,
    pub commit: Digest32,
}

impl AiOutputCandidate {
    pub fn new(
        cycle_id: u64,
        label: ChannelLabel,
        payload_commit: Digest32,
        confidence: u16,
    ) -> Self {
        let mut output = Self {
            cycle_id,
            label,
            payload_commit,
            confidence,
            commit: Digest32::new([0u8; 32]),
        };
        output.commit = commit_output_candidate(&output);
        output
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AiFeatureEvent {
    pub cycle_id: u64,
    pub feature_id: u32,
    pub intensity: u16,
    pub kind: SpikeKind,
    pub bucket: u8,
    pub commit: Digest32,
}

impl AiFeatureEvent {
    pub fn new(
        cycle_id: u64,
        feature_id: u32,
        intensity: u16,
        kind: SpikeKind,
        bucket: u8,
    ) -> Self {
        let mut event = Self {
            cycle_id,
            feature_id,
            intensity,
            kind,
            bucket,
            commit: Digest32::new([0u8; 32]),
        };
        event.commit = commit_feature_event(&event);
        event
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AiHostOutputs {
    pub cycle_id: u64,
    pub internal_thoughts: Vec<AiInternalThought>,
    pub features: Vec<AiFeatureEvent>,
    pub outputs: Vec<AiOutputCandidate>,
    pub commit: Digest32,
}

impl AiHostOutputs {
    pub fn new(
        cycle_id: u64,
        internal_thoughts: Vec<AiInternalThought>,
        features: Vec<AiFeatureEvent>,
        outputs: Vec<AiOutputCandidate>,
    ) -> Self {
        let mut host_outputs = Self {
            cycle_id,
            internal_thoughts,
            features,
            outputs,
            commit: Digest32::new([0u8; 32]),
        };
        host_outputs.commit = commit_host_outputs(&host_outputs);
        host_outputs
    }
}

pub trait AiHost {
    fn tick(&mut self, inp: &AiInputFrame) -> AiHostOutputs;
}

pub struct MockAiHost {
    pub commit: Digest32,
}

impl MockAiHost {
    pub fn new(commit: Digest32) -> Self {
        Self { commit }
    }
}

impl Default for MockAiHost {
    fn default() -> Self {
        Self {
            commit: Digest32::new([0u8; 32]),
        }
    }
}

impl AiHost for MockAiHost {
    fn tick(&mut self, inp: &AiInputFrame) -> AiHostOutputs {
        let thought_root = digest_mock_thought_root(self.commit, inp);
        let salience = ((inp.cycle_id % 9_000) as u16).saturating_add(1000);
        let internal_thoughts = vec![AiInternalThought::new(inp.cycle_id, thought_root, salience)];

        let mut features = Vec::new();
        if matches!(inp.gamma_bucket, 0 | 4 | 8 | 12) {
            let feature_id = (inp.cycle_id % u64::from(u32::MAX)) as u32;
            let intensity = 1500u16.saturating_add((inp.gamma_bucket as u16) * 50);
            let feature_commit = digest_mock_feature_commit(self.commit, inp, feature_id);
            let event = AiFeatureEvent::new(
                inp.cycle_id,
                feature_id,
                intensity,
                SpikeKind::Feature,
                inp.gamma_bucket,
            );
            let event = AiFeatureEvent {
                commit: feature_commit,
                ..event
            };
            features.push(event);
        }

        let mut outputs = Vec::new();
        let policy_even = inp.policy_snapshot_commit.as_bytes()[31].is_multiple_of(2);
        if inp.global_plv >= 7000 && policy_even {
            let payload_commit = digest_mock_output_commit(self.commit, inp);
            let confidence = 7000u16.saturating_add((inp.global_plv % 2000).min(1000));
            outputs.push(AiOutputCandidate::new(
                inp.cycle_id,
                ChannelLabel::External,
                payload_commit,
                confidence,
            ));
        }

        AiHostOutputs::new(inp.cycle_id, internal_thoughts, features, outputs)
    }
}

fn commit_input_frame(frame: &AiInputFrame) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(AI_INPUT_DOMAIN);
    hasher.update(&frame.cycle_id.to_be_bytes());
    hasher.update(frame.external_commit.as_bytes());
    hasher.update(frame.phase_commit.as_bytes());
    hasher.update(&[frame.gamma_bucket]);
    hasher.update(&frame.global_plv.to_be_bytes());
    hasher.update(frame.ssm_state_digest.as_bytes());
    hasher.update(frame.world_state.as_bytes());
    hasher.update(frame.policy_snapshot_commit.as_bytes());
    hasher.update(&frame.attention_cap.to_be_bytes());
    hasher.update(&frame.learning_cap.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_internal_thought(thought: &AiInternalThought) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(AI_INTERNAL_DOMAIN);
    hasher.update(&thought.cycle_id.to_be_bytes());
    hasher.update(thought.thought_root.as_bytes());
    hasher.update(&thought.salience.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_output_candidate(output: &AiOutputCandidate) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(AI_OUTPUT_DOMAIN);
    hasher.update(&output.cycle_id.to_be_bytes());
    hasher.update(&[output.label as u8]);
    hasher.update(output.payload_commit.as_bytes());
    hasher.update(&output.confidence.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_feature_event(event: &AiFeatureEvent) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(AI_FEATURE_DOMAIN);
    hasher.update(&event.cycle_id.to_be_bytes());
    hasher.update(&event.feature_id.to_be_bytes());
    hasher.update(&event.intensity.to_be_bytes());
    hasher.update(&[event.kind.discriminant()]);
    hasher.update(&[event.bucket]);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_host_outputs(outputs: &AiHostOutputs) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(AI_HOST_OUTPUTS_DOMAIN);
    hasher.update(&outputs.cycle_id.to_be_bytes());
    for thought in &outputs.internal_thoughts {
        hasher.update(thought.commit.as_bytes());
    }
    for feature in &outputs.features {
        hasher.update(feature.commit.as_bytes());
    }
    for output in &outputs.outputs {
        hasher.update(output.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_mock_thought_root(seed: Digest32, inp: &AiInputFrame) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(AI_MOCK_THOUGHT_DOMAIN);
    hasher.update(seed.as_bytes());
    hasher.update(&inp.cycle_id.to_be_bytes());
    hasher.update(inp.phase_commit.as_bytes());
    hasher.update(inp.external_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_mock_feature_commit(seed: Digest32, inp: &AiInputFrame, feature_id: u32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(AI_MOCK_FEATURE_DOMAIN);
    hasher.update(seed.as_bytes());
    hasher.update(&inp.cycle_id.to_be_bytes());
    hasher.update(&feature_id.to_be_bytes());
    hasher.update(&[inp.gamma_bucket]);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_mock_output_commit(seed: Digest32, inp: &AiInputFrame) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(AI_MOCK_OUTPUT_DOMAIN);
    hasher.update(seed.as_bytes());
    hasher.update(&inp.cycle_id.to_be_bytes());
    hasher.update(&inp.global_plv.to_be_bytes());
    hasher.update(inp.policy_snapshot_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_input(
        cycle_id: u64,
        gamma_bucket: u8,
        global_plv: u16,
        policy_byte: u8,
    ) -> AiInputFrame {
        let mut policy = [0u8; 32];
        policy[31] = policy_byte;
        AiInputFrame::new(
            cycle_id,
            Digest32::new([1u8; 32]),
            Digest32::new([2u8; 32]),
            gamma_bucket,
            global_plv,
            Digest32::new([3u8; 32]),
            Digest32::new([4u8; 32]),
            Digest32::new(policy),
            9000,
            8000,
        )
    }

    #[test]
    fn mock_ai_host_does_not_emit_external_output_when_plv_low() {
        let mut host = MockAiHost::default();
        let input = base_input(42, 2, 1200, 2);
        let outputs = host.tick(&input);
        assert!(outputs.outputs.is_empty());
        assert_eq!(outputs.cycle_id, 42);
        assert_eq!(outputs.internal_thoughts.len(), 1);
    }

    #[test]
    fn mock_ai_host_emits_feature_events_for_expected_buckets() {
        let mut host = MockAiHost::default();
        for bucket in [0u8, 4, 8, 12] {
            let input = base_input(7, bucket, 1200, 2);
            let outputs = host.tick(&input);
            assert_eq!(outputs.features.len(), 1);
            assert_eq!(outputs.features[0].bucket, bucket);
        }
    }
}
