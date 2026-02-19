use std::collections::{HashMap, VecDeque};

use ucf_ess::v1::{ExperienceKind, ExperiencePayload, ExperienceRecord};

pub type FrameId = u64;
pub type DecisionId = u64;
pub type BackendProfileId = &'static str;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComputeSignalSummaryView {
    pub t: u64,
    pub frame_id: FrameId,
    pub risk: f32,
    pub confidence: f32,
    pub surprise: f32,
    pub pressure: f32,
    pub spike_count: u16,
    pub spikes_digest: Option<[u8; 32]>,
    pub quality: u8,
    pub backend_profile: BackendProfileId,
    pub evidence_context_digest: [u8; 32],
    pub decision_id: DecisionId,
}

pub fn view_from_ess_record(rec: &ExperienceRecord) -> Option<ComputeSignalSummaryView> {
    if rec.kind != ExperienceKind::DecisionOut {
        return None;
    }
    let summary = rec.compute_summary?;
    let profile = summary.backend_profile?;
    let evidence_context_digest = summary.evidence_context_digest?;
    let quality = summary.risk_quality?;
    match rec.payload {
        ExperiencePayload::Decision(_) => Some(ComputeSignalSummaryView {
            t: rec.time.tick.get(),
            frame_id: rec.corr.0,
            risk: summary.risk.clamp(0.0, 1.0),
            confidence: summary.confidence.clamp(0.0, 1.0),
            surprise: summary.surprise.clamp(0.0, 1.0),
            pressure: summary.pressure.clamp(0.0, 1.0),
            spike_count: summary.spike_count,
            spikes_digest: Some(summary.spikes_digest),
            quality: quality.min(2),
            backend_profile: profile,
            evidence_context_digest,
            decision_id: rec.id.0,
        }),
        _ => None,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ComputeMilestone {
    pub milestone_id: u64,
    pub window_secs: u64,
    pub window_start: u64,
    pub window_end: u64,
    pub sample_count: u16,
    pub mean_risk: f32,
    pub mean_surprise: f32,
    pub mean_pressure: f32,
    pub degraded_or_unavailable_count: u16,
    pub spike_count_sum: u32,
    pub top_spike_digests: Vec<([u8; 32], u16)>,
}

pub trait ConsolidationHook {
    fn on_append(
        &mut self,
        rec: &ExperienceRecord,
    ) -> Result<Vec<ComputeMilestone>, ConsolidationError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsolidationError {
    InvalidWindow,
}

#[derive(Clone, Debug)]
struct WindowState {
    start: u64,
    end: u64,
    count: u32,
    sum_risk: f32,
    sum_surprise: f32,
    sum_pressure: f32,
    degraded_or_unavailable_count: u32,
    spike_count_sum: u32,
    spike_digest_counts: HashMap<[u8; 32], u16>,
}

impl WindowState {
    fn new(start: u64, window_secs: u64) -> Self {
        Self {
            start,
            end: start.saturating_add(window_secs),
            count: 0,
            sum_risk: 0.0,
            sum_surprise: 0.0,
            sum_pressure: 0.0,
            degraded_or_unavailable_count: 0,
            spike_count_sum: 0,
            spike_digest_counts: HashMap::new(),
        }
    }

    fn absorb(&mut self, view: &ComputeSignalSummaryView) {
        self.count = self.count.saturating_add(1);
        self.sum_risk += view.risk.clamp(0.0, 1.0);
        self.sum_surprise += view.surprise.clamp(0.0, 1.0);
        self.sum_pressure += view.pressure.clamp(0.0, 1.0);
        if view.quality >= 1 {
            self.degraded_or_unavailable_count =
                self.degraded_or_unavailable_count.saturating_add(1);
        }
        self.spike_count_sum = self
            .spike_count_sum
            .saturating_add(u32::from(view.spike_count).min(10_000));
        if let Some(digest) = view.spikes_digest {
            let entry = self.spike_digest_counts.entry(digest).or_insert(0);
            *entry = entry.saturating_add(1);
        }
    }

    fn to_milestone(&self, milestone_id: u64, window_secs: u64, top_k: usize) -> ComputeMilestone {
        let denom = self.count.max(1) as f32;
        let mut digest_counts = self
            .spike_digest_counts
            .iter()
            .map(|(digest, count)| (*digest, *count))
            .collect::<Vec<_>>();
        digest_counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        digest_counts.truncate(top_k.min(8));

        ComputeMilestone {
            milestone_id,
            window_secs,
            window_start: self.start,
            window_end: self.end,
            sample_count: self.count.min(u32::from(u16::MAX)) as u16,
            mean_risk: (self.sum_risk / denom).clamp(0.0, 1.0),
            mean_surprise: (self.sum_surprise / denom).clamp(0.0, 1.0),
            mean_pressure: (self.sum_pressure / denom).clamp(0.0, 1.0),
            degraded_or_unavailable_count: self
                .degraded_or_unavailable_count
                .min(u32::from(u16::MAX)) as u16,
            spike_count_sum: self.spike_count_sum.min(50_000),
            top_spike_digests: digest_counts,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ComputeMilestoneAggregator {
    windows: Vec<u64>,
    top_k: usize,
    next_milestone_id: u64,
    states: Vec<WindowState>,
}

impl ComputeMilestoneAggregator {
    pub fn new(windows: Vec<u64>, top_k: usize) -> Result<Self, ConsolidationError> {
        if windows.is_empty() || windows.contains(&0) {
            return Err(ConsolidationError::InvalidWindow);
        }
        Ok(Self {
            windows,
            top_k,
            next_milestone_id: 1,
            states: Vec::new(),
        })
    }

    pub fn from_records(
        windows: Vec<u64>,
        top_k: usize,
        records: &[ExperienceRecord],
    ) -> Result<Vec<ComputeMilestone>, ConsolidationError> {
        let mut agg = Self::new(windows, top_k)?;
        let mut out = Vec::new();
        for rec in records {
            out.extend(agg.on_append(rec)?);
        }
        Ok(out)
    }
}

impl ConsolidationHook for ComputeMilestoneAggregator {
    fn on_append(
        &mut self,
        rec: &ExperienceRecord,
    ) -> Result<Vec<ComputeMilestone>, ConsolidationError> {
        let Some(view) = view_from_ess_record(rec) else {
            return Ok(Vec::new());
        };

        if self.states.is_empty() {
            self.states = self
                .windows
                .iter()
                .map(|window| {
                    let start = (view.t / *window) * *window;
                    WindowState::new(start, *window)
                })
                .collect();
        }

        let mut emitted = Vec::new();
        for (idx, state) in self.states.iter_mut().enumerate() {
            let window = self.windows[idx];
            while view.t >= state.end {
                if state.count > 0 {
                    emitted.push(state.to_milestone(self.next_milestone_id, window, self.top_k));
                    self.next_milestone_id = self.next_milestone_id.saturating_add(1);
                }
                *state = WindowState::new(state.end, window);
            }
            state.absorb(&view);
        }

        Ok(emitted)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeistUpdateCandidate {
    pub risk_baseline: f32,
    pub pressure_baseline: f32,
    pub stability: f32,
    pub source_milestone_id: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeistMacroState {
    pub risk_baseline: f32,
    pub pressure_baseline: f32,
    pub stability: f32,
    pub source_milestone_id: u64,
    pub provenance_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LiquidContextWindow {
    pub sample_count: u16,
    pub mean_uncertainty: f32,
    pub max_uncertainty: f32,
    pub mean_stability: f32,
    pub rolling_digest: [u8; 32],
}

#[derive(Clone, Debug, Default)]
pub struct LiquidTimelineIndex {
    summaries: Vec<ucf_ess::v1::LfmSummaryRecord>,
}

impl LiquidTimelineIndex {
    pub fn rebuild_from_ess(&mut self, ess: &impl ucf_ess::v1::ExperienceStore) {
        self.summaries.clear();
        for idx in 0..ess.len() {
            let Some(rec) = ess.get(idx) else {
                continue;
            };
            if rec.kind != ExperienceKind::LfmSummary {
                continue;
            }
            if let Some(summary) = rec.lfm_summary_record {
                self.summaries.push(summary);
            }
        }
        self.summaries
            .sort_by(|a, b| a.t.cmp(&b.t).then_with(|| a.digest.cmp(&b.digest)));
        metrics::counter!("ucf_lfm_index_rebuild_total").increment(1);
    }

    pub fn append(&mut self, summary: ucf_ess::v1::LfmSummaryRecord) {
        let insert_at = self
            .summaries
            .binary_search_by(|probe| {
                probe
                    .t
                    .cmp(&summary.t)
                    .then_with(|| probe.digest.cmp(&summary.digest))
            })
            .unwrap_or_else(|idx| idx);
        self.summaries.insert(insert_at, summary);
    }

    pub fn get_window(
        &self,
        t0: u64,
        t1: u64,
        max_results: usize,
    ) -> Vec<ucf_ess::v1::LfmSummaryRecord> {
        let capped = max_results.clamp(1, 256);
        let mut out = self
            .summaries
            .iter()
            .copied()
            .filter(|s| s.t >= t0 && s.t <= t1)
            .collect::<Vec<_>>();
        if out.len() > capped {
            out.drain(0..out.len() - capped);
        }
        metrics::counter!("ucf_lfm_query_window_total").increment(1);
        out
    }

    pub fn get_last(&self, n: usize) -> Vec<ucf_ess::v1::LfmSummaryRecord> {
        let capped = n.clamp(1, 128);
        if self.summaries.len() <= capped {
            return self.summaries.clone();
        }
        self.summaries[self.summaries.len() - capped..].to_vec()
    }

    pub fn context_window(&self, n: usize) -> Option<LiquidContextWindow> {
        let last = self.get_last(n);
        if last.is_empty() {
            return None;
        }
        let mut sum_u = 0.0f32;
        let mut sum_s = 0.0f32;
        let mut max_u = 0.0f32;
        let mut hasher = blake3::Hasher::new();
        for item in &last {
            sum_u += item.uncertainty.clamp(0.0, 1.0);
            sum_s += item.stability.clamp(0.0, 1.0);
            max_u = max_u.max(item.uncertainty.clamp(0.0, 1.0));
            hasher.update(&item.digest);
        }
        let n = last.len() as f32;
        Some(LiquidContextWindow {
            sample_count: last.len().min(usize::from(u16::MAX)) as u16,
            mean_uncertainty: (sum_u / n).clamp(0.0, 1.0),
            max_uncertainty: max_u,
            mean_stability: (sum_s / n).clamp(0.0, 1.0),
            rolling_digest: *hasher.finalize().as_bytes(),
        })
    }
}

pub trait GeistHook {
    fn on_milestone(
        &mut self,
        ms: &ComputeMilestone,
    ) -> Result<Option<GeistMacroState>, GeistError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeistRejectReason {
    NotEnoughSamples,
    Unstable,
    Degraded,
    Drift,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeistError {
    InvalidConfig,
}

#[derive(Clone, Debug)]
pub struct GeistStateUpdater {
    min_samples: u16,
    stability_threshold: f32,
    degraded_ratio_threshold: f32,
    drift_threshold: f32,
    history: VecDeque<ComputeMilestone>,
    history_cap: usize,
    last_state: Option<GeistMacroState>,
    last_reject: Option<GeistRejectReason>,
}

impl GeistStateUpdater {
    pub fn new(
        min_samples: u16,
        stability_threshold: f32,
        degraded_ratio_threshold: f32,
        drift_threshold: f32,
    ) -> Result<Self, GeistError> {
        if min_samples == 0 {
            return Err(GeistError::InvalidConfig);
        }
        Ok(Self {
            min_samples,
            stability_threshold: stability_threshold.clamp(0.0, 1.0),
            degraded_ratio_threshold: degraded_ratio_threshold.clamp(0.0, 1.0),
            drift_threshold: drift_threshold.clamp(0.0, 1.0),
            history: VecDeque::new(),
            history_cap: 8,
            last_state: None,
            last_reject: None,
        })
    }

    pub fn last_state(&self) -> Option<GeistMacroState> {
        self.last_state
    }

    pub fn last_reject_reason(&self) -> Option<GeistRejectReason> {
        self.last_reject
    }
}

impl GeistHook for GeistStateUpdater {
    fn on_milestone(
        &mut self,
        ms: &ComputeMilestone,
    ) -> Result<Option<GeistMacroState>, GeistError> {
        self.history.push_back(ms.clone());
        while self.history.len() > self.history_cap {
            self.history.pop_front();
        }

        let total_samples = self
            .history
            .iter()
            .map(|m| u32::from(m.sample_count))
            .sum::<u32>();
        if total_samples < u32::from(self.min_samples) {
            self.last_reject = Some(GeistRejectReason::NotEnoughSamples);
            return Ok(None);
        }

        let n = self.history.len() as f32;
        let mean_risk = self.history.iter().map(|m| m.mean_risk).sum::<f32>() / n;
        let variance = self
            .history
            .iter()
            .map(|m| {
                let d = m.mean_risk - mean_risk;
                d * d
            })
            .sum::<f32>()
            / n.max(1.0);
        let stddev = variance.sqrt().clamp(0.0, 1.0);
        let stability = (1.0 - stddev).clamp(0.0, 1.0);

        let degraded = self
            .history
            .iter()
            .map(|m| u32::from(m.degraded_or_unavailable_count))
            .sum::<u32>();
        let degraded_ratio = degraded as f32 / total_samples.max(1) as f32;
        if degraded_ratio > self.degraded_ratio_threshold {
            self.last_reject = Some(GeistRejectReason::Degraded);
            return Ok(None);
        }

        if stability < self.stability_threshold {
            self.last_reject = Some(GeistRejectReason::Unstable);
            return Ok(None);
        }

        let pressure_baseline =
            (self.history.iter().map(|m| m.mean_pressure).sum::<f32>() / n).clamp(0.0, 1.0);
        let risk_baseline = mean_risk.clamp(0.0, 1.0);

        if let Some(last) = self.last_state {
            let drift = (risk_baseline - last.risk_baseline).abs();
            if drift > self.drift_threshold {
                self.last_reject = Some(GeistRejectReason::Drift);
                return Ok(None);
            }
        }

        let candidate = GeistUpdateCandidate {
            risk_baseline,
            pressure_baseline,
            stability,
            source_milestone_id: ms.milestone_id,
        };

        let mut hasher = blake3::Hasher::new();
        hasher.update(b"ucf.runtime.geist.macro.v0");
        hasher.update(&candidate.risk_baseline.to_le_bytes());
        hasher.update(&candidate.pressure_baseline.to_le_bytes());
        hasher.update(&candidate.stability.to_le_bytes());
        hasher.update(&candidate.source_milestone_id.to_le_bytes());
        let provenance_digest = *hasher.finalize().as_bytes();

        let state = GeistMacroState {
            risk_baseline,
            pressure_baseline,
            stability,
            source_milestone_id: ms.milestone_id,
            provenance_digest,
        };
        self.last_state = Some(state);
        self.last_reject = None;
        Ok(Some(state))
    }
}

#[cfg(test)]
mod tests {
    use ucf_core::types::{SimTime, Tick, WindowId};
    use ucf_ess::v1::{ExperienceId, ExperienceRecord};
    use ucf_frames::v1::{CorrelationId, DecisionFrame, IntentType, ReasonCode};

    use super::*;

    fn sample_decision_rec(tick: u64, corr: u64, risk: f32, quality: u8) -> ExperienceRecord {
        let time = SimTime {
            tick: Tick::new(tick),
            window: WindowId::new(0),
        };
        let decision = DecisionFrame::allow_with_reason(
            time,
            CorrelationId(corr),
            IntentType::Unknown,
            ReasonCode("ok"),
            "ok",
        )
        .with_compute_summary(ucf_frames::v1::ComputeSignalsSummary {
            backend: "stub",
            surprise: 0.2,
            pressure: 0.4,
            risk,
            confidence: 0.8,
            surprise_q: 0,
            pressure_q: 0,
            risk_q: 0,
            confidence_q: 0,
            spike_count: 4,
            spikes_digest: [1; 32],
            sparsity: None,
            energy: None,
            ssm_readout: None,
            ssm_digest: None,
            world_digest: None,
            risk_quality: Some(quality),
            evidence_context_digest: Some([9; 32]),
            evidence_world_digest: None,
            evidence_spikes_digest: None,
            evidence_ssm_digest: None,
            evidence_lfm_digest: None,
            backend_profile: Some("cpu_stub_v1"),
            backend_pack_id: None,
            fixtures_digest: None,
            llm_backend: None,
            world_backend: None,
            sae_backend: None,
            ssm_backend: None,
            lfm_backend: None,
            lfm_uncertainty: None,
            lfm_stability: None,
            lfm_uncertainty_q: None,
            lfm_stability_q: None,
            lfm_state_norm: None,
            lfm_deriv_norm: None,
            lfm_saturation_ratio: None,
            lfm_nan_inf_detected: None,
            lfm_digest: None,
            budget_profile_id: Some(1),
            seed: Some(7),
            risk_contract_version: Some(1),
            compute_schema_version: Some(2),
            compute_chain_digest: Some([8; 32]),
            compute_code_version: Some("v0.0.0"),
            budget_exceeded_stage: None,
            contract_version: Some(1),
            backend_id: Some(0),
            validation_status: Some(0),
            violation_reason_mask: Some(0),
            lfm_quality: None,
            coherence: None,
            instability: None,
            coherence_q: None,
            instability_q: None,
            phi_proxy: None,
            coherence_digest: None,
            iit_coherence_q: None,
            iit_incoherence_q: None,
            iit_reason_codes: None,
            stage_allow_mask: None,
            free_energy_proxy_q: None,
            ebm_energy_mean_topk_q: None,
            ebm_w_q: None,
            fep_coupling_version: None,
        });
        ExperienceRecord::from_decision(ExperienceId(corr), decision)
    }

    #[test]
    fn view_extracts_expected_fields() {
        let rec = sample_decision_rec(10, 22, 0.6, 1);
        let view = view_from_ess_record(&rec).expect("view");
        assert_eq!(view.t, 10);
        assert_eq!(view.frame_id, 22);
        assert_eq!(view.decision_id, 22);
        assert_eq!(view.backend_profile, "cpu_stub_v1");
        assert_eq!(view.quality, 1);
        assert_eq!(view.evidence_context_digest, [9; 32]);
    }

    #[test]
    fn milestone_windows_are_deterministic_and_bounded() {
        let mut agg = ComputeMilestoneAggregator::new(vec![60], 2).expect("agg");
        let mut milestones = Vec::new();
        for tick in 0..121 {
            let rec = sample_decision_rec(tick, tick, 0.5, if tick % 2 == 0 { 0 } else { 1 });
            milestones.extend(agg.on_append(&rec).expect("append"));
        }
        assert_eq!(milestones.len(), 2);
        assert!(milestones.iter().all(|m| m.sample_count <= 60));
        assert!(milestones.iter().all(|m| m.top_spike_digests.len() <= 2));
        assert!(milestones
            .iter()
            .all(|m| (0.0..=1.0).contains(&m.mean_risk)));
    }

    #[test]
    fn rebuild_from_records_matches_streaming_output() {
        let records = (0..180)
            .map(|tick| sample_decision_rec(tick, tick, 0.2 + (tick as f32 % 10.0) * 0.01, 0))
            .collect::<Vec<_>>();
        let mut agg = ComputeMilestoneAggregator::new(vec![60], 2).expect("agg");
        let mut stream = Vec::new();
        for rec in &records {
            stream.extend(agg.on_append(rec).expect("append"));
        }

        let rebuilt =
            ComputeMilestoneAggregator::from_records(vec![60], 2, &records).expect("rebuild");
        assert_eq!(stream, rebuilt);
    }

    #[test]
    fn liquid_timeline_index_rebuild_and_query_are_deterministic() {
        use ucf_ess::v1::{ExperienceStore, IdAllocator, InMemoryEss, LfmSummaryRecord};

        let mut ess = InMemoryEss::new();
        let mut ids = IdAllocator::new(1);
        for tick in 0..20u64 {
            let time = SimTime {
                tick: Tick::new(tick),
                window: WindowId::new(0),
            };
            let summary = LfmSummaryRecord {
                t: tick,
                decision_id: Some(100 + tick),
                evidence_chain_digest: [1; 32],
                backend_pack_digest: [2; 32],
                liquid_state_digest: [tick as u8; 32],
                liquid_readout_digest: [3; 32],
                uncertainty: (tick as f32 / 20.0).clamp(0.0, 1.0),
                stability: (1.0 - tick as f32 / 20.0).clamp(0.0, 1.0),
                schema_version: 2,
                digest: [0; 32],
            }
            .with_digest();
            let rec =
                ExperienceRecord::from_lfm_summary(ids.next(), time, CorrelationId(tick), summary);
            ess.append(rec).expect("append");
        }

        let mut index = LiquidTimelineIndex::default();
        index.rebuild_from_ess(&ess);
        let last = index.get_last(8);
        assert_eq!(last.len(), 8);
        assert!(last.windows(2).all(|w| w[0].t <= w[1].t));

        let window = index.get_window(5, 9, 32);
        assert_eq!(window.len(), 5);
        assert_eq!(window.first().map(|s| s.t), Some(5));
        assert_eq!(window.last().map(|s| s.t), Some(9));

        let ctx = index.context_window(4).expect("ctx");
        assert_eq!(ctx.sample_count, 4);
        assert!((0.0..=1.0).contains(&ctx.mean_uncertainty));
        assert!((0.0..=1.0).contains(&ctx.mean_stability));
    }
    #[test]
    fn geist_rejects_unstable_or_degraded_milestones() {
        let mut geist = GeistStateUpdater::new(120, 0.95, 0.2, 0.4).expect("geist");
        let first = ComputeMilestone {
            milestone_id: 1,
            window_secs: 60,
            window_start: 0,
            window_end: 60,
            sample_count: 60,
            mean_risk: 0.0,
            mean_surprise: 0.2,
            mean_pressure: 0.2,
            degraded_or_unavailable_count: 0,
            spike_count_sum: 10,
            top_spike_digests: Vec::new(),
        };
        assert!(geist.on_milestone(&first).expect("ok").is_none());
        assert_eq!(
            geist.last_reject_reason(),
            Some(GeistRejectReason::NotEnoughSamples)
        );

        let unstable = ComputeMilestone {
            milestone_id: 2,
            mean_risk: 1.0,
            ..first.clone()
        };
        assert!(geist.on_milestone(&unstable).expect("ok").is_none());
        assert_eq!(
            geist.last_reject_reason(),
            Some(GeistRejectReason::Unstable)
        );

        let degraded = ComputeMilestone {
            milestone_id: 3,
            mean_risk: 0.5,
            degraded_or_unavailable_count: 40,
            ..first
        };
        assert!(geist.on_milestone(&degraded).expect("ok").is_none());
        assert_eq!(
            geist.last_reject_reason(),
            Some(GeistRejectReason::Degraded)
        );
        assert!(geist.last_state().is_none());
    }
}
