#![forbid(unsafe_code)]
//! Compatibility wrapper around the legacy host ABI contract.
//!
//! Canonical runtime compute execution lives in `runtime/ucf-compute`.
//! This crate stays intentionally narrow and is retained for adapter and
//! compatibility seams only.

use ucf_ai_host_abi::AiBackend;
pub use ucf_ai_host_abi::{
    AiFeatureEvent, AiHostAbiInput, AiHostAbiOutput, AiHostOutputs, AiInputFrame,
    AiInternalThought, AiOutputCandidate, ChannelLabel, MockBackend,
};

pub trait AiHost {
    fn tick(&mut self, inp: &AiInputFrame) -> AiHostOutputs;
}

pub struct AiHostRuntime<B> {
    backend: B,
}

impl<B> AiHostRuntime<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }
}

impl<B: Default> Default for AiHostRuntime<B> {
    fn default() -> Self {
        Self {
            backend: B::default(),
        }
    }
}

impl<B: AiBackend> AiHost for AiHostRuntime<B> {
    fn tick(&mut self, inp: &AiInputFrame) -> AiHostOutputs {
        let abi_output = self.backend.infer(AiHostAbiInput::from(inp));
        AiHostOutputs {
            cycle_id: inp.cycle_id,
            internal_thoughts: abi_output.internal_thoughts,
            features: abi_output.feature_events,
            outputs: abi_output.output_candidates,
            commit: abi_output.abi_commit,
        }
    }
}

pub type MockAiHost = AiHostRuntime<MockBackend>;

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_types::Digest32;

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
    fn default_runtime_uses_mock_backend_contract() {
        let mut host = MockAiHost::default();
        let input = base_input(7, 4, 7_500, 2);
        let outputs = host.tick(&input);

        assert_eq!(outputs.cycle_id, 7);
        assert_eq!(outputs.internal_thoughts.len(), 1);
        assert_eq!(outputs.features.len(), 1);
        assert_eq!(outputs.features[0].bucket, 4);
        assert_eq!(outputs.outputs.len(), 1);
    }

    #[test]
    fn default_runtime_keeps_coherence_loop_safe_when_plv_low() {
        let mut host = MockAiHost::default();
        let input = base_input(42, 2, 1_200, 2);
        let outputs = host.tick(&input);

        assert!(outputs.outputs.is_empty());
        assert_eq!(outputs.internal_thoughts.len(), 1);
    }
}
