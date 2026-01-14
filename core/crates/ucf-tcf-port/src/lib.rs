#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_types::Digest32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TcfSignal {
    pub digest: Digest32,
    pub tick: u64,
}

impl TcfSignal {
    pub fn new(digest: Digest32, tick: u64) -> Self {
        Self { digest, tick }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TcfPulse {
    pub digest: Digest32,
    pub tick: u64,
    pub inputs: usize,
}

pub trait TcfPort {
    fn tick(&self, signal: &mut TcfSignal, inputs: &[Digest32]) -> TcfPulse;
}

#[derive(Clone, Default)]
pub struct MockTcfPort;

impl MockTcfPort {
    pub fn new() -> Self {
        Self
    }
}

impl TcfPort for MockTcfPort {
    fn tick(&self, signal: &mut TcfSignal, inputs: &[Digest32]) -> TcfPulse {
        let mut hasher = Hasher::new();
        hasher.update(signal.digest.as_bytes());
        hasher.update(&signal.tick.to_be_bytes());
        for input in inputs {
            hasher.update(input.as_bytes());
        }
        let digest = Digest32::new(*hasher.finalize().as_bytes());
        signal.digest = digest;
        signal.tick = signal.tick.saturating_add(1);
        TcfPulse {
            digest,
            tick: signal.tick,
            inputs: inputs.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_tcf_is_deterministic() {
        let mut signal_a = TcfSignal::new(Digest32::new([1u8; 32]), 0);
        let mut signal_b = TcfSignal::new(Digest32::new([1u8; 32]), 0);
        let inputs = vec![Digest32::new([2u8; 32]), Digest32::new([3u8; 32])];
        let port = MockTcfPort::new();

        let out_a = port.tick(&mut signal_a, &inputs);
        let out_b = port.tick(&mut signal_b, &inputs);

        assert_eq!(out_a, out_b);
        assert_eq!(signal_a.tick, 1);
    }
}
