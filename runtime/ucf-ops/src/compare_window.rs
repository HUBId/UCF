use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use ucf_ess::v1::SlotCompareWindowRecordV1;

use crate::{prefix_hex, sha256_hex};

/// Unified compare-window semantics for v3.
///
/// Window boundaries are fixed tick windows with `[t0, t1]` evidence bounds.
/// `window_id` is deterministically derived from `(run_id, slot_id, t0, t1)`.
/// Compared backends are sorted by backend id.
/// Sample digest prefixes are bounded (`<=4`) and selected deterministically
/// from already-ordered source samples.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UnifiedCompareWindowSemanticsV1 {
    pub window_boundary: String,
    pub window_id_rule: String,
    pub compared_backends_ordering: String,
    pub freshness_rule: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CompareWindowBackendStatusV1 {
    Ok,
    Warn,
    Severe,
    Skip,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CompareWindowFreshnessV1 {
    Fresh,
    NoCompare,
    StaleCompare,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompareWindowMetaV1 {
    pub slot_id: String,
    pub run_id: String,
    pub window_id: u64,
    pub t0: u64,
    pub t1: u64,
    pub primary_backend_id: String,
    pub compared_backend_ids: Vec<String>,
    pub compare_window_digest: String,
    pub policy_graph_digest_prefix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DriftInputV1 {
    pub slot_id: String,
    pub window_id: u64,
    pub invalid_rate_q: u16,
    pub digest_mismatch_rate_q: u16,
    pub latency_p95_ms_q: u32,
    pub scalar_deltas_q: BTreeMap<String, u16>,
}

pub fn unified_compare_semantics_v1() -> UnifiedCompareWindowSemanticsV1 {
    UnifiedCompareWindowSemanticsV1 {
        window_boundary: "fixed_ticks_inclusive_t0_t1".to_string(),
        window_id_rule: "u64_prefix(sha256(run_id:slot_id:t0:t1))".to_string(),
        compared_backends_ordering: "lexicographic_backend_id".to_string(),
        freshness_rule: "current_tick - t1 <= max_age => FRESH else STALE_COMPARE".to_string(),
    }
}

pub fn derive_window_id(run_id: &str, slot_id: &str, t0: u64, t1: u64) -> u64 {
    let material = format!("{run_id}:{slot_id}:{t0}:{t1}");
    let digest = sha256_hex(material.as_bytes());
    u64::from_str_radix(&digest[..16], 16).unwrap_or(0)
}

pub fn sorted_compared_backend_ids<I: IntoIterator<Item = String>>(ids: I) -> Vec<String> {
    let mut sorted = ids.into_iter().collect::<Vec<_>>();
    sorted.sort();
    sorted.dedup();
    if sorted.len() > 2 {
        sorted.truncate(2);
    }
    sorted
}

pub fn compare_window_digest(
    slot_id: &str,
    run_id: &str,
    t0: u64,
    t1: u64,
    primary: &str,
) -> String {
    sha256_hex(format!("{slot_id}:{run_id}:{t0}:{t1}:{primary}").as_bytes())
}

pub fn build_compare_window_meta(
    slot_id: &str,
    run_id: &str,
    t0: u64,
    t1: u64,
    primary_backend_id: &str,
    compared_backend_ids: Vec<String>,
    policy_graph_digest_prefix: String,
) -> CompareWindowMetaV1 {
    let compared_backend_ids = sorted_compared_backend_ids(compared_backend_ids);
    let compare_window_digest = compare_window_digest(slot_id, run_id, t0, t1, primary_backend_id);
    CompareWindowMetaV1 {
        slot_id: slot_id.to_string(),
        run_id: run_id.to_string(),
        window_id: derive_window_id(run_id, slot_id, t0, t1),
        t0,
        t1,
        primary_backend_id: primary_backend_id.to_string(),
        compared_backend_ids,
        compare_window_digest: prefix_hex(&compare_window_digest, 16),
        policy_graph_digest_prefix,
    }
}

pub fn sample_digest_prefixes(samples: &[[u8; 4]]) -> Vec<String> {
    samples.iter().take(4).map(hex::encode).collect()
}

pub fn compare_freshness(
    latest_t1: Option<u64>,
    current_tick: u64,
    max_age_ticks: u64,
) -> CompareWindowFreshnessV1 {
    let Some(t1) = latest_t1 else {
        return CompareWindowFreshnessV1::NoCompare;
    };
    if current_tick.saturating_sub(t1) > max_age_ticks {
        CompareWindowFreshnessV1::StaleCompare
    } else {
        CompareWindowFreshnessV1::Fresh
    }
}

pub fn derive_drift_inputs_from_slot_compare(
    slot_id: &str,
    run_id: &str,
    compare: &SlotCompareWindowRecordV1,
    latency_p95_ms_q: u32,
) -> DriftInputV1 {
    let sample_count = u32::from(compare.sample_count).max(1);
    let invalid_rate_q = ((u32::from(compare.invalid_shadow_count) * 10_000) / sample_count)
        .min(u16::MAX as u32) as u16;
    let digest_mismatch_rate_q =
        ((u32::from(compare.digest_mismatch_count) * 10_000) / sample_count).min(10_000) as u16;
    let mut scalar_deltas_q = BTreeMap::new();
    scalar_deltas_q.insert(
        "risk_mean_q".to_string(),
        compare.primary_mean_q.abs_diff(compare.shadow_mean_q),
    );
    scalar_deltas_q.insert(
        "risk_p95_q".to_string(),
        compare.primary_p95_q.abs_diff(compare.shadow_p95_q),
    );
    DriftInputV1 {
        slot_id: slot_id.to_string(),
        window_id: derive_window_id(run_id, slot_id, compare.t0, compare.t1),
        invalid_rate_q,
        digest_mismatch_rate_q,
        latency_p95_ms_q,
        scalar_deltas_q,
    }
}

impl Default for CompareWindowMetaV1 {
    fn default() -> Self {
        Self {
            slot_id: "unknown".to_string(),
            run_id: "unknown".to_string(),
            window_id: 0,
            t0: 0,
            t1: 0,
            primary_backend_id: "unknown".to_string(),
            compared_backend_ids: Vec::new(),
            compare_window_digest: "unknown".to_string(),
            policy_graph_digest_prefix: "unknown".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_id_derivation_is_deterministic() {
        let a = derive_window_id("r1", "sae", 10, 20);
        let b = derive_window_id("r1", "sae", 10, 20);
        assert_eq!(a, b);
    }

    #[test]
    fn compared_backend_ids_are_sorted() {
        let ids = sorted_compared_backend_ids(vec![
            "candle_sae_v1".to_string(),
            "burn_sae_v1".to_string(),
        ]);
        assert_eq!(ids, vec!["burn_sae_v1", "candle_sae_v1"]);
    }

    #[test]
    fn freshness_is_unified() {
        let world = compare_freshness(Some(100), 120, 32);
        let second = compare_freshness(Some(100), 120, 32);
        assert_eq!(world, second);
    }
}
