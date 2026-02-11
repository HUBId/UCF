use serde::{Deserialize, Serialize};
use std::cmp;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchStats {
    pub n: usize,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub throughput_ops_sec: f64,
}

pub fn summarize(latencies_ns: &[u128], total_ns: u128) -> BenchStats {
    let mut sorted: Vec<u128> = latencies_ns.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let sum: u128 = sorted.iter().sum();

    let mean_ms = if n == 0 {
        0.0
    } else {
        ns_to_ms(sum) / n as f64
    };

    BenchStats {
        n,
        mean_ms,
        p50_ms: percentile_ms(&sorted, 0.50),
        p95_ms: percentile_ms(&sorted, 0.95),
        p99_ms: percentile_ms(&sorted, 0.99),
        min_ms: sorted.first().copied().map(ns_to_ms).unwrap_or(0.0),
        max_ms: sorted.last().copied().map(ns_to_ms).unwrap_or(0.0),
        throughput_ops_sec: if total_ns == 0 {
            0.0
        } else {
            n as f64 / (total_ns as f64 / 1_000_000_000.0)
        },
    }
}

fn percentile_ms(sorted_ns: &[u128], q: f64) -> f64 {
    if sorted_ns.is_empty() {
        return 0.0;
    }
    let idx = cmp::min(
        ((sorted_ns.len() as f64 - 1.0) * q).round() as usize,
        sorted_ns.len() - 1,
    );
    ns_to_ms(sorted_ns[idx])
}

fn ns_to_ms(value: u128) -> f64 {
    value as f64 / 1_000_000.0
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct AllocationSnapshot {
    pub alloc_count: u64,
    pub dealloc_count: u64,
    pub alloc_bytes_total: u64,
    pub dealloc_bytes_total: u64,
}

impl AllocationSnapshot {
    pub fn diff(self, before: Self) -> Self {
        Self {
            alloc_count: self.alloc_count.saturating_sub(before.alloc_count),
            dealloc_count: self.dealloc_count.saturating_sub(before.dealloc_count),
            alloc_bytes_total: self
                .alloc_bytes_total
                .saturating_sub(before.alloc_bytes_total),
            dealloc_bytes_total: self
                .dealloc_bytes_total
                .saturating_sub(before.dealloc_bytes_total),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegressionCheck {
    pub metric: String,
    pub baseline: f64,
    pub current: f64,
    pub ok: bool,
    pub detail: String,
}

pub fn parse_fixture_cases(raw: &str) -> Result<Vec<FixtureCase>, serde_json::Error> {
    serde_json::from_str(raw)
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct FixtureCase {
    pub frame_id: u64,
    pub t: u64,
    pub seed: u64,
    pub context_digest_hex: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarize_is_stable() {
        let lats = vec![1_000_000, 2_000_000, 5_000_000, 1_500_000, 1_200_000];
        let stats = summarize(&lats, lats.iter().sum());
        assert_eq!(stats.n, 5);
        assert!(stats.p95_ms >= stats.p50_ms);
        assert!(stats.max_ms >= stats.min_ms);
    }

    #[test]
    fn fixture_parse_works() {
        let payload = r#"[{"frame_id":1,"t":2,"seed":3,"context_digest_hex":"00"}]"#;
        let parsed = parse_fixture_cases(payload).expect("fixture parse");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].frame_id, 1);
    }
}
