#![forbid(unsafe_code)]

pub const MAX_SESSION_ID_LEN: usize = 64;
pub const MAX_REASON_CODES: usize = 16;
pub const MAX_MICRO_CFG_DIGESTS: usize = 8;
pub const MAX_SPIKE_LIST_LEN: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BiophysFeedbackSnapshot {
    pub session_id: String,
    pub tick: u64,
    pub asset_manifest_digest: Option<[u8; 32]>,
    pub micro_cfg_digests: Vec<([u8; 32], &'static str)>,
    pub runtime_snapshot_digest: [u8; 32],
    pub spike_train_digest: [u8; 32],
    pub ca_spike_count: u32,
    pub event_queue_overflowed: bool,
    pub events_dropped: u32,
    pub injected_spikes_received: u32,
    pub injected_targets_applied: u32,
    pub cooldown_ticks_remaining: u32,
    pub last_update_kind: Option<biophys_governance::UpdateKind>,
    pub reason_codes: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BiophysFeedbackState {
    pub spike_neuron_ids: Vec<u32>,
    pub event_queue_overflowed: bool,
    pub events_dropped: u32,
    pub ca_spike_count: u32,
}

pub fn digest_seq(domain: &str, bytes: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain.as_bytes());
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

pub fn bound_session_id(session_id: &str) -> String {
    session_id.chars().take(MAX_SESSION_ID_LEN).collect()
}

pub fn normalize_reason_codes(mut codes: Vec<String>) -> Vec<String> {
    codes.sort();
    codes.dedup();
    if codes.len() > MAX_REASON_CODES {
        codes.truncate(MAX_REASON_CODES);
    }
    codes
}

pub fn spike_train_digest(neuron_ids: &[u32]) -> [u8; 32] {
    let mut ids = neuron_ids.to_vec();
    ids.sort_unstable();
    if ids.len() > MAX_SPIKE_LIST_LEN {
        ids.truncate(MAX_SPIKE_LIST_LEN);
    }
    let mut bytes = Vec::with_capacity(ids.len().saturating_mul(4));
    for id in ids {
        bytes.extend(id.to_le_bytes());
    }
    digest_seq("UCF:BIO:SPIKE_TRAIN", &bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spike_train_digest_is_deterministic_and_bounded() {
        let ids: Vec<u32> = (0..(MAX_SPIKE_LIST_LEN as u32 + 10)).collect();
        let digest_full = spike_train_digest(&ids);
        let digest_truncated = spike_train_digest(&ids[..MAX_SPIKE_LIST_LEN]);
        assert_eq!(digest_full, digest_truncated);
        let digest_again = spike_train_digest(&ids);
        assert_eq!(digest_full, digest_again);
    }

    #[test]
    fn reason_codes_are_sorted_and_bounded() {
        let mut codes = Vec::new();
        for idx in 0..(MAX_REASON_CODES + 3) {
            codes.push(format!("RC.TEST.{idx:02}"));
        }
        codes.reverse();
        let normalized = normalize_reason_codes(codes);
        assert_eq!(normalized.len(), MAX_REASON_CODES);
        let mut sorted = normalized.clone();
        sorted.sort();
        assert_eq!(normalized, sorted);
    }
}
