#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_onn::{OnnParams, OscId};
use ucf_spikebus::{SpikeEvent, SpikeKind};
use ucf_types::Digest32;

const PAYLOAD_DOMAIN: &[u8] = b"ucf.spike_encoder.payload.v2";
const MAX_STRENGTH: u16 = 10_000;
fn default_phase_window() -> u16 {
    OnnParams::default().lock_window
}

pub fn ttfs_from_strength(strength: u16, phase_window: u16) -> u16 {
    if phase_window == 0 {
        return 0;
    }
    let clamped = strength.min(MAX_STRENGTH);
    let inverted = MAX_STRENGTH.saturating_sub(clamped) as u32;
    let window = u32::from(phase_window);
    let ttfs = (window.saturating_mul(inverted)) / u32::from(MAX_STRENGTH);
    ttfs.min(window) as u16
}

pub fn encode_spike(
    kind: SpikeKind,
    src: OscId,
    dst: OscId,
    strength: u16,
    phase_commit: Digest32,
    phase_bucket: u8,
    salt_commit: Digest32,
) -> SpikeEvent {
    encode_spike_with_window(
        0,
        kind,
        src,
        dst,
        strength,
        phase_commit,
        phase_bucket,
        salt_commit,
        default_phase_window(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn encode_spike_with_ttfs(
    cycle_id: u64,
    kind: SpikeKind,
    src: OscId,
    dst: OscId,
    phase_bucket: u8,
    ttfs: u16,
    phase_commit: Digest32,
    payload_commit: Digest32,
) -> SpikeEvent {
    SpikeEvent::new(
        cycle_id,
        kind,
        src,
        dst,
        phase_bucket,
        ttfs,
        phase_commit,
        payload_commit,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn encode_spike_with_payload_commit(
    cycle_id: u64,
    kind: SpikeKind,
    src: OscId,
    dst: OscId,
    strength: u16,
    phase_commit: Digest32,
    phase_bucket: u8,
    payload_commit: Digest32,
    phase_window: u16,
) -> SpikeEvent {
    let ttfs = ttfs_from_strength(strength, phase_window);
    encode_spike_with_ttfs(
        cycle_id,
        kind,
        src,
        dst,
        phase_bucket,
        ttfs,
        phase_commit,
        payload_commit,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn encode_spike_with_window(
    cycle_id: u64,
    kind: SpikeKind,
    src: OscId,
    dst: OscId,
    strength: u16,
    phase_commit: Digest32,
    phase_bucket: u8,
    salt_commit: Digest32,
    phase_window: u16,
) -> SpikeEvent {
    let ttfs = ttfs_from_strength(strength, phase_window);
    let payload_commit = commit_payload(kind, strength, phase_commit, salt_commit);
    encode_spike_with_ttfs(
        cycle_id,
        kind,
        src,
        dst,
        phase_bucket,
        ttfs,
        phase_commit,
        payload_commit,
    )
}

pub fn commit_payload(
    kind: SpikeKind,
    strength: u16,
    phase_commit: Digest32,
    salt_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(PAYLOAD_DOMAIN);
    hasher.update(&kind.as_u16().to_be_bytes());
    hasher.update(&strength.to_be_bytes());
    hasher.update(salt_commit.as_bytes());
    hasher.update(phase_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ttfs_monotonic_for_strength() {
        let window = 1_000;
        let low = ttfs_from_strength(100, window);
        let high = ttfs_from_strength(9000, window);
        assert!(high < low);
    }

    #[test]
    fn encode_spike_deterministic() {
        let phase_commit = Digest32::new([3u8; 32]);
        let salt_commit = Digest32::new([7u8; 32]);
        let first = encode_spike_with_window(
            1,
            SpikeKind::Feature,
            OscId::Jepa,
            OscId::Ssm,
            2400,
            phase_commit,
            4,
            salt_commit,
            1024,
        );
        let second = encode_spike_with_window(
            1,
            SpikeKind::Feature,
            OscId::Jepa,
            OscId::Ssm,
            2400,
            phase_commit,
            4,
            salt_commit,
            1024,
        );
        assert_eq!(first.commit, second.commit);
    }
}
