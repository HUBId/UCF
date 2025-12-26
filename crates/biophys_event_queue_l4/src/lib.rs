#![forbid(unsafe_code)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpikeEventL4 {
    pub deliver_step: u64,
    pub synapse_index: usize,
    pub release_gain_q: u16,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RuntimeHealth {
    pub overflowed: bool,
    pub dropped_events: u32,
    pub compacted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueLimits {
    pub max_events_total: usize,
    pub max_events_per_bucket: usize,
}

impl QueueLimits {
    pub fn new(max_events_total: usize, max_events_per_bucket: usize) -> Self {
        Self {
            max_events_total,
            max_events_per_bucket,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RuntimeCounters {
    pub steps_executed: u64,
    pub spikes_total: u64,
    pub events_pushed: u64,
    pub events_delivered: u64,
    pub events_dropped: u64,
    pub max_bucket_depth_seen: u32,
    pub compactions_run: u32,
    pub asset_bytes_decoded: u64,
}

#[derive(Debug, Clone)]
pub struct SpikeEventQueueL4 {
    buckets: Vec<Vec<SpikeEventL4>>,
    limits: QueueLimits,
    total_events: usize,
    pub dropped_event_count: u64,
    dropped_events_this_tick: u32,
    overflowed_this_tick: bool,
    needs_compaction: bool,
    counters: RuntimeCounters,
}

impl SpikeEventQueueL4 {
    pub fn new(max_delay_steps: u16, limits: QueueLimits) -> Self {
        let len = max_delay_steps as usize + 1;
        Self {
            buckets: vec![Vec::new(); len.max(1)],
            limits,
            total_events: 0,
            dropped_event_count: 0,
            dropped_events_this_tick: 0,
            overflowed_this_tick: false,
            needs_compaction: false,
            counters: RuntimeCounters::default(),
        }
    }

    pub fn schedule_spike<F, G>(
        &mut self,
        current_step: u64,
        synapse_indices: &[usize],
        delay_steps_for: F,
        mut release_gain_for: G,
    ) where
        F: Fn(usize) -> u16,
        G: FnMut(usize) -> u16,
    {
        if synapse_indices.is_empty() {
            return;
        }
        self.counters.spikes_total = self.counters.spikes_total.saturating_add(1);
        let mut sorted = synapse_indices.to_vec();
        sorted.sort_unstable();
        for synapse_index in sorted {
            if self.overflowed_this_tick || self.total_events >= self.limits.max_events_total {
                self.overflowed_this_tick = true;
                self.record_drop();
                continue;
            }
            let deliver_step = current_step.saturating_add(delay_steps_for(synapse_index) as u64);
            let bucket = (deliver_step as usize) % self.buckets.len();
            if self.buckets[bucket].len() >= self.limits.max_events_per_bucket {
                let (max_idx, max_value) = self.buckets[bucket]
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, event)| event.synapse_index)
                    .map(|(idx, event)| (idx, event.synapse_index))
                    .unwrap();
                if synapse_index < max_value {
                    self.buckets[bucket][max_idx] = SpikeEventL4 {
                        deliver_step,
                        synapse_index,
                        release_gain_q: release_gain_for(synapse_index),
                    };
                    self.counters.events_pushed = self.counters.events_pushed.saturating_add(1);
                    self.record_drop();
                    self.needs_compaction = true;
                } else {
                    self.record_drop();
                }
                continue;
            }
            self.buckets[bucket].push(SpikeEventL4 {
                deliver_step,
                synapse_index,
                release_gain_q: release_gain_for(synapse_index),
            });
            self.total_events = self.total_events.saturating_add(1);
            self.counters.events_pushed = self.counters.events_pushed.saturating_add(1);
            let depth = self.buckets[bucket].len() as u32;
            if depth > self.counters.max_bucket_depth_seen {
                self.counters.max_bucket_depth_seen = depth;
            }
        }
    }

    pub fn drain_current(&mut self, current_step: u64) -> Vec<SpikeEventL4> {
        let bucket = (current_step as usize) % self.buckets.len();
        let mut events = std::mem::take(&mut self.buckets[bucket]);
        let mut current = Vec::new();
        for event in events.drain(..) {
            if event.deliver_step == current_step {
                current.push(event);
            } else {
                self.buckets[bucket].push(event);
            }
        }
        let delivered = current.len() as u64;
        self.counters.steps_executed = self.counters.steps_executed.saturating_add(1);
        self.counters.events_delivered = self.counters.events_delivered.saturating_add(delivered);
        self.total_events = self.total_events.saturating_sub(delivered as usize);
        current
    }

    pub fn finish_tick(&mut self) -> RuntimeHealth {
        let compacted = if self.needs_compaction {
            self.compact()
        } else {
            false
        };
        let health = RuntimeHealth {
            overflowed: self.overflowed_this_tick,
            dropped_events: self.dropped_events_this_tick,
            compacted,
        };
        self.dropped_events_this_tick = 0;
        self.overflowed_this_tick = false;
        self.needs_compaction = false;
        health
    }

    pub fn counters_snapshot(&self) -> RuntimeCounters {
        self.counters
    }

    fn record_drop(&mut self) {
        self.dropped_event_count = self.dropped_event_count.saturating_add(1);
        self.dropped_events_this_tick = self.dropped_events_this_tick.saturating_add(1);
        self.counters.events_dropped = self.counters.events_dropped.saturating_add(1);
        self.needs_compaction = true;
    }

    fn compact(&mut self) -> bool {
        let mut events = Vec::with_capacity(self.total_events);
        for bucket in &self.buckets {
            events.extend(bucket.iter().copied());
        }
        events.sort_by_key(|event| (event.deliver_step, event.synapse_index));
        let mut dropped = 0usize;
        if events.len() > self.limits.max_events_total {
            dropped = events.len() - self.limits.max_events_total;
            events.truncate(self.limits.max_events_total);
        }
        self.buckets.iter_mut().for_each(Vec::clear);
        self.total_events = 0;
        for event in events {
            let bucket = (event.deliver_step as usize) % self.buckets.len();
            if self.buckets[bucket].len() >= self.limits.max_events_per_bucket {
                dropped = dropped.saturating_add(1);
                continue;
            }
            self.buckets[bucket].push(event);
            self.total_events = self.total_events.saturating_add(1);
        }
        if dropped > 0 {
            for _ in 0..dropped {
                self.record_drop();
            }
        }
        self.counters.compactions_run = self.counters.compactions_run.saturating_add(1);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_overflow_drops_highest_synapse_index() {
        let limits = QueueLimits::new(10, 2);
        let mut queue = SpikeEventQueueL4::new(0, limits);
        let synapse_indices = vec![3, 1, 2];
        queue.schedule_spike(0, &synapse_indices, |_| 0, |_| 1);
        let health = queue.finish_tick();
        assert_eq!(health.dropped_events, 1);
        let mut events = queue.drain_current(0);
        events.sort_by_key(|event| event.synapse_index);
        let indices = events
            .iter()
            .map(|event| event.synapse_index)
            .collect::<Vec<_>>();
        assert_eq!(indices, vec![1, 2]);
    }

    #[test]
    fn total_overflow_sets_flag_and_degrades_for_tick() {
        let limits = QueueLimits::new(2, 10);
        let mut queue = SpikeEventQueueL4::new(0, limits);
        let synapse_indices = vec![0, 1, 2];
        queue.schedule_spike(0, &synapse_indices, |_| 0, |_| 1);
        queue.drain_current(0);
        let health = queue.finish_tick();
        assert!(health.overflowed);
        assert_eq!(health.dropped_events, 1);

        queue.schedule_spike(1, &[0], |_| 0, |_| 1);
        let health = queue.finish_tick();
        assert!(!health.overflowed);
        assert_eq!(health.dropped_events, 0);
    }

    #[test]
    fn compaction_only_runs_when_needed() {
        let limits = QueueLimits::new(4, 2);
        let mut queue = SpikeEventQueueL4::new(0, limits);
        queue.schedule_spike(0, &[0], |_| 0, |_| 1);
        let health = queue.finish_tick();
        assert!(!health.compacted);
        assert_eq!(queue.counters_snapshot().compactions_run, 0);

        queue.schedule_spike(1, &[2, 3, 4], |_| 0, |_| 1);
        let health = queue.finish_tick();
        assert!(health.compacted);
        assert_eq!(queue.counters_snapshot().compactions_run, 1);
    }

    #[test]
    fn counters_are_deterministic() {
        let limits = QueueLimits::new(3, 2);
        let mut queue_a = SpikeEventQueueL4::new(0, limits);
        let mut queue_b = SpikeEventQueueL4::new(0, limits);

        let synapses = vec![2, 0, 1];
        queue_a.schedule_spike(0, &synapses, |_| 0, |_| 1);
        queue_b.schedule_spike(0, &synapses, |_| 0, |_| 1);
        queue_a.drain_current(0);
        queue_b.drain_current(0);
        queue_a.finish_tick();
        queue_b.finish_tick();

        assert_eq!(queue_a.counters_snapshot(), queue_b.counters_snapshot());
    }
}
