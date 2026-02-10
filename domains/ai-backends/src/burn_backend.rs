use ucf_ai_host_abi::{AiBackend, AiHostAbiInput, AiHostAbiOutput};

#[derive(Default)]
pub struct BurnBackend;

impl AiBackend for BurnBackend {
    fn infer(&mut self, _inp: AiHostAbiInput<'_>) -> AiHostAbiOutput {
        // TODO(T102): wire Burn tensor I/O + LFM/RLM/SAE/Lens integration.
        AiHostAbiOutput::bounded(Vec::new(), Vec::new(), Vec::new())
    }
}
