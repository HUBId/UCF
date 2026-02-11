use std::collections::{HashMap, VecDeque};

use crate::capability::CapabilityKind;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RateKey {
    pub kind: CapabilityKind,
    pub target: String,
    pub token_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RateOutcome {
    pub allowed: bool,
    pub retry_after_ticks: u64,
}

#[derive(Clone, Debug)]
struct Entry {
    window_start: u64,
    calls: u32,
    inserted_at: u64,
}

#[derive(Clone, Debug)]
pub struct RateLimiter {
    entries: HashMap<RateKey, Entry>,
    order: VecDeque<RateKey>,
    max_entries: usize,
}

impl RateLimiter {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            max_entries: max_entries.max(1),
        }
    }

    pub fn check_and_record(
        &mut self,
        key: RateKey,
        now_t: u64,
        max_calls: u32,
        window_ticks: u64,
    ) -> RateOutcome {
        let max_calls = max_calls.max(1);
        let window_ticks = window_ticks.max(1);
        let mut retry_after = 0;

        if let Some(entry) = self.entries.get_mut(&key) {
            if now_t.saturating_sub(entry.window_start) >= window_ticks {
                entry.window_start = now_t;
                entry.calls = 0;
            }

            if entry.calls >= max_calls {
                retry_after = window_ticks.saturating_sub(now_t.saturating_sub(entry.window_start));
                return RateOutcome {
                    allowed: false,
                    retry_after_ticks: retry_after,
                };
            }
            entry.calls = entry.calls.saturating_add(1);
            return RateOutcome {
                allowed: true,
                retry_after_ticks: retry_after,
            };
        }

        self.order.push_back(key.clone());
        self.entries.insert(
            key,
            Entry {
                window_start: now_t,
                calls: 1,
                inserted_at: now_t,
            },
        );
        self.evict_if_needed();
        RateOutcome {
            allowed: true,
            retry_after_ticks: 0,
        }
    }

    fn evict_if_needed(&mut self) {
        while self.entries.len() > self.max_entries {
            let Some(oldest_key) = self
                .order
                .iter()
                .filter_map(|key| self.entries.get(key).map(|entry| (key, entry.inserted_at)))
                .min_by_key(|(_, inserted_at)| *inserted_at)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            self.entries.remove(&oldest_key);
            if let Some(idx) = self.order.iter().position(|k| *k == oldest_key) {
                self.order.remove(idx);
            }
        }
    }
}
