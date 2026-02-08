use blake3::Hasher;
use ucf_types::Digest32;

use crate::{ModuleId, Spike, SpikeKind};

const SAE_DOMAIN: &[u8] = b"ucf.spikebus.mock.sae.v1";
const LENS_DOMAIN: &[u8] = b"ucf.spikebus.mock.lens.v1";

pub trait SpikeProducer {
    fn produce(&self, cycle_id: u64, gamma_bucket: u8) -> Vec<Spike>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MockSaeProducer {
    pub commit: Digest32,
}

impl MockSaeProducer {
    pub fn new(commit: Digest32) -> Self {
        Self { commit }
    }

    fn payload_commit(&self, cycle_id: u64, gamma_bucket: u8) -> Digest32 {
        let mut hasher = Hasher::new();
        hasher.update(SAE_DOMAIN);
        hasher.update(&cycle_id.to_be_bytes());
        hasher.update(&[gamma_bucket]);
        hasher.update(self.commit.as_bytes());
        Digest32::new(*hasher.finalize().as_bytes())
    }
}

impl SpikeProducer for MockSaeProducer {
    fn produce(&self, cycle_id: u64, gamma_bucket: u8) -> Vec<Spike> {
        if !matches!(gamma_bucket, 0 | 4 | 8 | 12) {
            return Vec::new();
        }
        let payload_commit = self.payload_commit(cycle_id, gamma_bucket);
        vec![Spike::new(
            cycle_id,
            SpikeKind::Feature,
            6500,
            gamma_bucket,
            ModuleId::Sae,
            payload_commit,
        )]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MockLensProducer {
    pub commit: Digest32,
}

impl MockLensProducer {
    pub fn new(commit: Digest32) -> Self {
        Self { commit }
    }

    fn payload_commit(&self, cycle_id: u64, gamma_bucket: u8) -> Digest32 {
        let mut hasher = Hasher::new();
        hasher.update(LENS_DOMAIN);
        hasher.update(&cycle_id.to_be_bytes());
        hasher.update(&[gamma_bucket]);
        hasher.update(self.commit.as_bytes());
        Digest32::new(*hasher.finalize().as_bytes())
    }
}

impl SpikeProducer for MockLensProducer {
    fn produce(&self, cycle_id: u64, gamma_bucket: u8) -> Vec<Spike> {
        if !matches!(gamma_bucket, 2 | 6 | 10 | 14) {
            return Vec::new();
        }
        let payload_commit = self.payload_commit(cycle_id, gamma_bucket);
        vec![Spike::new(
            cycle_id,
            SpikeKind::Novelty,
            6200,
            gamma_bucket,
            ModuleId::Lens,
            payload_commit,
        )]
    }
}
