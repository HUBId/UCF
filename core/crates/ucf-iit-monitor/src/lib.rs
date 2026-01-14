#![forbid(unsafe_code)]

//! Deterministic proxy coupling metric between module digests.
//!
//! This is **not** a real IIT/Phi implementation. The score is a stable hash mapping
//! into `0..=10000` and only serves as a repeatable integration/coupling proxy.

use std::collections::VecDeque;

use blake3::Hasher;
use ucf_types::Digest32;

const DOMAIN_SCORE: &[u8] = b"ucf.iit.proxy.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CouplingSample {
    pub a: Digest32,
    pub b: Digest32,
    pub score: u16,
}

#[derive(Debug)]
pub struct IitMonitor {
    pub window: usize,
    samples: VecDeque<CouplingSample>,
}

impl IitMonitor {
    pub fn new(window: usize) -> Self {
        Self {
            window,
            samples: VecDeque::new(),
        }
    }

    pub fn sample(&mut self, module_a: Digest32, module_b: Digest32) -> CouplingSample {
        let score = coupling_score(&module_a, &module_b);
        let sample = CouplingSample {
            a: module_a,
            b: module_b,
            score,
        };
        self.samples.push_back(sample);
        if self.window > 0 {
            while self.samples.len() > self.window {
                self.samples.pop_front();
            }
        }
        sample
    }

    pub fn aggregate(&self) -> u16 {
        if self.samples.is_empty() {
            return 0;
        }
        let sum: u32 = self
            .samples
            .iter()
            .map(|sample| u32::from(sample.score))
            .sum();
        let avg = sum / self.samples.len().max(1) as u32;
        u16::try_from(avg.min(u32::from(u16::MAX))).unwrap_or(u16::MAX)
    }
}

fn coupling_score(a: &Digest32, b: &Digest32) -> u16 {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_SCORE);
    hasher.update(a.as_bytes());
    hasher.update(b.as_bytes());
    let bytes = hasher.finalize();
    let raw = u32::from_be_bytes([
        bytes.as_bytes()[0],
        bytes.as_bytes()[1],
        bytes.as_bytes()[2],
        bytes.as_bytes()[3],
    ]);
    let scaled = (u64::from(raw) * 10_000) / u64::from(u32::MAX);
    u16::try_from(scaled.min(10_000)).unwrap_or(10_000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_is_deterministic_and_bounded() {
        let mut monitor = IitMonitor::new(4);
        let a = Digest32::new([1u8; 32]);
        let b = Digest32::new([2u8; 32]);

        let first = monitor.sample(a, b);
        let second = monitor.sample(a, b);

        assert_eq!(first.score, second.score);
        assert!(first.score <= 10_000);
    }
}
