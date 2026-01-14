#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_types::{Digest32, ThoughtVec};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NcdeContext {
    pub seed: Digest32,
}

impl NcdeContext {
    pub fn new(seed: Digest32) -> Self {
        Self { seed }
    }
}

pub trait NcdePort {
    fn integrate(&self, ctx: &NcdeContext, control: &ThoughtVec) -> ThoughtVec;
}

#[derive(Clone, Default)]
pub struct MockNcdePort;

impl MockNcdePort {
    pub fn new() -> Self {
        Self
    }
}

impl NcdePort for MockNcdePort {
    fn integrate(&self, ctx: &NcdeContext, control: &ThoughtVec) -> ThoughtVec {
        let mut hasher = Hasher::new();
        hasher.update(ctx.seed.as_bytes());
        hasher.update(&control.bytes);
        let digest = Digest32::new(*hasher.finalize().as_bytes());
        ThoughtVec::new(digest.as_bytes().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_ncde_is_deterministic() {
        let ctx = NcdeContext::new(Digest32::new([9u8; 32]));
        let control = ThoughtVec::new(vec![1, 2, 3]);
        let port = MockNcdePort::new();

        let out_a = port.integrate(&ctx, &control);
        let out_b = port.integrate(&ctx, &control);

        assert_eq!(out_a, out_b);
    }
}
