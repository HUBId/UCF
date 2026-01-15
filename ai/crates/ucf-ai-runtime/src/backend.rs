use std::sync::Arc;

use ucf_sandbox::ControlFrameNormalized;
use ucf_types::AiOutput;

pub trait AiRuntimeBackend {
    fn infer_runtime(&self, cf: &ControlFrameNormalized) -> Vec<AiOutput>;
}

type RuntimeDelegate = Arc<dyn Fn(&ControlFrameNormalized) -> Vec<AiOutput> + Send + Sync>;

#[derive(Clone)]
pub struct NoopRuntimeBackend {
    delegate: RuntimeDelegate,
}

impl NoopRuntimeBackend {
    pub fn new(delegate: RuntimeDelegate) -> Self {
        Self { delegate }
    }
}

impl AiRuntimeBackend for NoopRuntimeBackend {
    fn infer_runtime(&self, cf: &ControlFrameNormalized) -> Vec<AiOutput> {
        (self.delegate)(cf)
    }
}
