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
const AI_ABI_OUTPUT_DOMAIN: &[u8] = b"ucf.ai.host.abi.output.v1";

pub const MAX_INTERNAL_THOUGHTS: usize = 8;
pub const MAX_FEATURE_EVENTS: usize = 32;
pub const MAX_OUTPUT_CANDIDATES: usize = 16;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AiHostAbiInput<'a> {
    pub cycle_id: u64,
    pub gamma_bucket: u8,
    pub global_plv: u16,
    pub attention_cap: u16,
    pub learning_cap: u16,
    pub external_commit: &'a Digest32,
    pub phase_commit: &'a Digest32,
    pub ssm_state_digest: &'a Digest32,
    pub world_state: &'a Digest32,
    pub policy_snapshot_commit: &'a Digest32,
    pub frame_commit: &'a Digest32,
}

impl<'a> From<&'a AiInputFrame> for AiHostAbiInput<'a> {
    fn from(value: &'a AiInputFrame) -> Self {
        Self {
            cycle_id: value.cycle_id,
            gamma_bucket: value.gamma_bucket,
            global_plv: value.global_plv,
            attention_cap: value.attention_cap,
            learning_cap: value.learning_cap,
            external_commit: &value.external_commit,
            phase_commit: &value.phase_commit,
            ssm_state_digest: &value.ssm_state_digest,
            world_state: &value.world_state,
            policy_snapshot_commit: &value.policy_snapshot_commit,
            frame_commit: &value.commit,
        }
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AiHostAbiOutput {
    pub feature_events: Vec<AiFeatureEvent>,
    pub output_candidates: Vec<AiOutputCandidate>,
    pub internal_thoughts: Vec<AiInternalThought>,
    pub abi_commit: Digest32,
}

impl AiHostAbiOutput {
    pub fn bounded(
        mut feature_events: Vec<AiFeatureEvent>,
        mut output_candidates: Vec<AiOutputCandidate>,
        mut internal_thoughts: Vec<AiInternalThought>,
    ) -> Self {
        feature_events.truncate(MAX_FEATURE_EVENTS);
        output_candidates.truncate(MAX_OUTPUT_CANDIDATES);
        internal_thoughts.truncate(MAX_INTERNAL_THOUGHTS);
        let abi_commit = commit_abi_output(&feature_events, &output_candidates, &internal_thoughts);
        Self {
            feature_events,
            output_candidates,
            internal_thoughts,
            abi_commit,
        }
    }
}

pub trait AiBackend {
    fn infer(&mut self, inp: AiHostAbiInput<'_>) -> AiHostAbiOutput;
}

#[derive(Clone, Debug)]
pub struct MockBackend {
    pub commit: Digest32,
}

impl MockBackend {
    pub fn new(commit: Digest32) -> Self {
        Self { commit }
    }
}

impl Default for MockBackend {
    fn default() -> Self {
        Self::new(Digest32::new([0u8; 32]))
    }
}

impl AiBackend for MockBackend {
    fn infer(&mut self, inp: AiHostAbiInput<'_>) -> AiHostAbiOutput {
        let thought_root = digest_mock_thought_root(self.commit, inp);
        let salience = ((inp.cycle_id % 9_000) as u16).saturating_add(1000);
        let internal_thoughts = vec![AiInternalThought::new(inp.cycle_id, thought_root, salience)];

        let mut feature_events = Vec::new();
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
            feature_events.push(event);
        }

        let mut output_candidates = Vec::new();
        let policy_even = inp.policy_snapshot_commit.as_bytes()[31].is_multiple_of(2);
        if inp.global_plv >= 7000 && policy_even {
            let payload_commit = digest_mock_output_commit(self.commit, inp);
            let confidence = 7000u16.saturating_add((inp.global_plv % 2000).min(1000));
            output_candidates.push(AiOutputCandidate::new(
                inp.cycle_id,
                ChannelLabel::External,
                payload_commit,
                confidence,
            ));
        }

        AiHostAbiOutput::bounded(feature_events, output_candidates, internal_thoughts)
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

fn commit_abi_output(
    feature_events: &[AiFeatureEvent],
    output_candidates: &[AiOutputCandidate],
    internal_thoughts: &[AiInternalThought],
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(AI_ABI_OUTPUT_DOMAIN);
    for feature in feature_events {
        hasher.update(feature.commit.as_bytes());
    }
    for output in output_candidates {
        hasher.update(output.commit.as_bytes());
    }
    for thought in internal_thoughts {
        hasher.update(thought.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_mock_thought_root(seed: Digest32, inp: AiHostAbiInput<'_>) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(AI_MOCK_THOUGHT_DOMAIN);
    hasher.update(seed.as_bytes());
    hasher.update(&inp.cycle_id.to_be_bytes());
    hasher.update(inp.phase_commit.as_bytes());
    hasher.update(inp.external_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_mock_feature_commit(
    seed: Digest32,
    inp: AiHostAbiInput<'_>,
    feature_id: u32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(AI_MOCK_FEATURE_DOMAIN);
    hasher.update(seed.as_bytes());
    hasher.update(&inp.cycle_id.to_be_bytes());
    hasher.update(&feature_id.to_be_bytes());
    hasher.update(&[inp.gamma_bucket]);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_mock_output_commit(seed: Digest32, inp: AiHostAbiInput<'_>) -> Digest32 {
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

    const _: [(); 1] = [(); (std::mem::size_of::<AiHostAbiInput<'static>>() <= 128) as usize];

    fn assert_copy<T: Copy>() {}

    #[test]
    fn abi_input_is_small_and_copy_without_drop_glue() {
        assert_copy::<AiHostAbiInput<'_>>();
        assert!(std::mem::size_of::<AiHostAbiInput<'_>>() <= 128);
        assert!(!std::mem::needs_drop::<AiHostAbiInput<'_>>());
    }
}
