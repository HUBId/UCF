use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Instant;

use sha2::{Digest, Sha256};
use ucf_types::{quantize_unit, SlotModeV1, CANONICAL_UNIT_QUANT_MAX};

use crate::{
    AiComputeBackend, ComputeBudget, ComputeError, ComputeInput, ComputeSignals, ModelSlot,
    ShadowDisableRecordV1, SlotCompareStatusV1, SlotCompareWindowRecordV1, SlotModeChangeRecordV1,
    SlotShadowEventV1,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealEnablementMode {
    Off,
    Shadow,
    Compare,
    Active,
}

impl RealEnablementMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" => Some(Self::Off),
            "shadow" => Some(Self::Shadow),
            "compare" => Some(Self::Compare),
            "active" => Some(Self::Active),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotMode {
    Toy,
    Shadow,
    Active,
}

impl SlotMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "toy" => Some(Self::Toy),
            "shadow" => Some(Self::Shadow),
            "active" => Some(Self::Active),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnablementConfig {
    pub mode: RealEnablementMode,
    pub shadow_every_n_ticks: u64,
    pub shadow_rate: u64,
    pub compare_window: u64,
}

impl Default for EnablementConfig {
    fn default() -> Self {
        Self {
            mode: RealEnablementMode::Off,
            shadow_every_n_ticks: 4,
            shadow_rate: 4,
            compare_window: 256,
        }
    }
}

impl EnablementConfig {
    pub fn from_env() -> Result<Self, ComputeError> {
        let mut cfg = Self::default();
        if let Ok(mode) = std::env::var("UCF_REAL_ENABLEMENT_MODE") {
            cfg.mode =
                RealEnablementMode::parse(&mode).ok_or_else(|| ComputeError::InvalidInput {
                    reason: format!("invalid UCF_REAL_ENABLEMENT_MODE={mode}"),
                })?;
        }
        if let Ok(raw) = std::env::var("UCF_SHADOW_EVERY_N_TICKS") {
            cfg.shadow_every_n_ticks = raw
                .parse::<u64>()
                .map_err(|_| ComputeError::InvalidInput {
                    reason: format!("invalid UCF_SHADOW_EVERY_N_TICKS={raw}"),
                })?
                .max(1);
        }
        if let Ok(raw) = std::env::var("UCF_SLOT_SHADOW_RATE") {
            cfg.shadow_rate = raw
                .parse::<u64>()
                .map_err(|_| ComputeError::InvalidInput {
                    reason: format!("invalid UCF_SLOT_SHADOW_RATE={raw}"),
                })?
                .max(1);
        }
        if let Ok(raw) = std::env::var("UCF_SLOT_COMPARE_WINDOW") {
            cfg.compare_window = raw
                .parse::<u64>()
                .map_err(|_| ComputeError::InvalidInput {
                    reason: format!("invalid UCF_SLOT_COMPARE_WINDOW={raw}"),
                })?
                .max(1);
        }
        Ok(cfg)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotEnablement {
    pub llm: SlotMode,
    pub lfm: SlotMode,
    pub world_jepa: SlotMode,
    pub sae: SlotMode,
    pub ssm: SlotMode,
    pub ebm: SlotMode,
}

impl Default for SlotEnablement {
    fn default() -> Self {
        Self {
            llm: SlotMode::Toy,
            lfm: SlotMode::Toy,
            world_jepa: SlotMode::Toy,
            sae: SlotMode::Toy,
            ssm: SlotMode::Toy,
            ebm: SlotMode::Toy,
        }
    }
}

impl SlotEnablement {
    pub fn from_env() -> Result<Self, ComputeError> {
        let mut cfg = Self::default();
        for slot in ModelSlot::all() {
            let key = format!("UCF_SLOT_{}_MODE", slot.env_key());
            if let Ok(mode) = std::env::var(&key) {
                let parsed = SlotMode::parse(&mode).ok_or_else(|| ComputeError::InvalidInput {
                    reason: format!("invalid {key}={mode}"),
                })?;
                match slot {
                    ModelSlot::Llm => cfg.llm = parsed,
                    ModelSlot::Lfm => cfg.lfm = parsed,
                    ModelSlot::WorldJepa => cfg.world_jepa = parsed,
                    ModelSlot::WorldVljepa => cfg.world_jepa = parsed,
                    ModelSlot::Sae => cfg.sae = parsed,
                    ModelSlot::Ssm => cfg.ssm = parsed,
                    ModelSlot::EbmReasoner => cfg.ebm = parsed,
                }
            }
        }
        Ok(cfg)
    }

    pub fn for_slot(&self, slot: ModelSlot) -> SlotMode {
        match slot {
            ModelSlot::Llm => self.llm,
            ModelSlot::Lfm => self.lfm,
            ModelSlot::WorldJepa => self.world_jepa,
            ModelSlot::WorldVljepa => self.world_jepa,
            ModelSlot::Sae => self.sae,
            ModelSlot::Ssm => self.ssm,
            ModelSlot::EbmReasoner => self.ebm,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShadowComparisonRecord {
    pub t: u64,
    pub toy_digest_prefix: String,
    pub real_digest_prefix: String,
    pub elapsed_ms: u64,
    pub result: &'static str,
}

struct PrimaryOutput(ComputeSignals);
struct ShadowOutput(ComputeSignals);

#[derive(Debug, Clone)]
struct CompareWindowState {
    t0: u64,
    primary_q: Vec<u16>,
    shadow_q: Vec<u16>,
    digest_mismatch_count: u16,
    invalid_shadow_count: u16,
    digest_prefix_samples: Vec<[u8; 4]>,
}

impl CompareWindowState {
    fn new(t0: u64) -> Self {
        Self {
            t0,
            primary_q: Vec::new(),
            shadow_q: Vec::new(),
            digest_mismatch_count: 0,
            invalid_shadow_count: 0,
            digest_prefix_samples: Vec::new(),
        }
    }

    fn record(&mut self, primary: &ComputeSignals, shadow: &ComputeSignals) {
        self.primary_q
            .push(quantize_unit(primary.risk, CANONICAL_UNIT_QUANT_MAX));
        self.shadow_q
            .push(quantize_unit(shadow.risk, CANONICAL_UNIT_QUANT_MAX));
        let pfx_primary = digest_prefix4(primary);
        let pfx_shadow = digest_prefix4(shadow);
        if pfx_primary != pfx_shadow {
            self.digest_mismatch_count = self.digest_mismatch_count.saturating_add(1);
        }
        if !envelope_ok(shadow) {
            self.invalid_shadow_count = self.invalid_shadow_count.saturating_add(1);
        }
        if self.digest_prefix_samples.len() < 4 {
            self.digest_prefix_samples.push(pfx_shadow);
        }
    }

    fn flush(self, slot_id: &str, t1: u64, disabled: bool) -> SlotCompareWindowRecordV1 {
        let sample_count = self.primary_q.len().min(usize::from(u16::MAX)) as u16;
        let primary_mean_q = mean_q(&self.primary_q);
        let shadow_mean_q = mean_q(&self.shadow_q);
        let primary_p95_q = p95_q(self.primary_q);
        let shadow_p95_q = p95_q(self.shadow_q);
        let status = if disabled {
            SlotCompareStatusV1::ShadowDisabled
        } else if self.digest_mismatch_count > 0 || self.invalid_shadow_count > 0 {
            SlotCompareStatusV1::DriftWarn
        } else {
            SlotCompareStatusV1::Ok
        };
        SlotCompareWindowRecordV1 {
            slot_id: slot_id.to_string(),
            t0: self.t0,
            t1,
            sample_count,
            primary_mean_q,
            primary_p95_q,
            shadow_mean_q,
            shadow_p95_q,
            digest_mismatch_count: self.digest_mismatch_count,
            invalid_shadow_count: self.invalid_shadow_count,
            digest_prefix_samples: self.digest_prefix_samples,
            status,
        }
    }
}

#[derive(Debug, Clone)]
struct ShadowRuntime {
    mode: SlotModeV1,
    last_mode: SlotModeV1,
    phase_offset: u64,
    compare_window_state: CompareWindowState,
    consecutive_shadow_failures: u16,
    shadow_disabled: bool,
    outbox: Vec<SlotShadowEventV1>,
}

pub struct EnablementComputeBackend {
    primary: Box<dyn AiComputeBackend + Send + Sync>,
    shadow: Option<Box<dyn AiComputeBackend + Send + Sync>>,
    cfg: EnablementConfig,
    runtime: Mutex<ShadowRuntime>,
}

impl EnablementComputeBackend {
    pub fn new(
        primary: Box<dyn AiComputeBackend + Send + Sync>,
        shadow: Option<Box<dyn AiComputeBackend + Send + Sync>>,
        cfg: EnablementConfig,
    ) -> Self {
        let slot_id = "compute";
        let phase_offset = shadow_phase_offset(0, slot_id, cfg.shadow_rate);
        Self {
            primary,
            shadow,
            cfg,
            runtime: Mutex::new(ShadowRuntime {
                mode: slot_mode_from_enablement(cfg.mode),
                last_mode: slot_mode_from_enablement(cfg.mode),
                phase_offset,
                compare_window_state: CompareWindowState::new(0),
                consecutive_shadow_failures: 0,
                shadow_disabled: false,
                outbox: Vec::new(),
            }),
        }
    }
}

impl AiComputeBackend for EnablementComputeBackend {
    fn name(&self) -> &'static str {
        self.primary.name()
    }

    fn compute(
        &self,
        input: &ComputeInput,
        budget: ComputeBudget,
    ) -> Result<ComputeSignals, ComputeError> {
        let primary = PrimaryOutput(self.primary.compute(input, budget)?);
        metrics::gauge!("ucf_slot_active", "slot" => "compute").set(
            if self.cfg.mode == RealEnablementMode::Active {
                1.0
            } else {
                0.0
            },
        );

        let mut rt = self.runtime.lock().expect("shadow runtime lock poisoned");
        let current_mode = slot_mode_from_enablement(self.cfg.mode);
        if current_mode != rt.last_mode {
            let from_mode = rt.last_mode;
            rt.outbox
                .push(SlotShadowEventV1::ModeChange(SlotModeChangeRecordV1 {
                    slot_id: "compute".to_string(),
                    t: input.t,
                    from_mode,
                    to_mode: current_mode,
                }));
            rt.last_mode = current_mode;
            rt.mode = current_mode;
        }

        if rt.mode == SlotModeV1::Off {
            return Ok(primary.0);
        }

        if let Some(shadow) = &self.shadow {
            if should_run_shadow_at(input.t, rt.phase_offset, self.cfg.shadow_rate)
                && !rt.shadow_disabled
            {
                metrics::counter!("ucf_shadow_runs_total", "slot" => "compute").increment(1);
                let started = Instant::now();
                match shadow.compute(
                    input,
                    ComputeBudget {
                        max_micros: budget.max_micros / 2,
                        ..budget
                    },
                ) {
                    Ok(out) => {
                        let _elapsed_ms = started.elapsed().as_millis() as u64;
                        let shadow_out = ShadowOutput(out);
                        rt.consecutive_shadow_failures = 0;
                        rt.compare_window_state.record(&primary.0, &shadow_out.0);
                    }
                    Err(_) => {
                        rt.consecutive_shadow_failures =
                            rt.consecutive_shadow_failures.saturating_add(1);
                        metrics::counter!("ucf_shadow_timeouts_total", "slot" => "compute")
                            .increment(1);
                        if rt.consecutive_shadow_failures >= 3 {
                            rt.shadow_disabled = true;
                            let consecutive_failures = rt.consecutive_shadow_failures;
                            rt.outbox.push(SlotShadowEventV1::ShadowDisable(
                                ShadowDisableRecordV1 {
                                    slot_id: "compute".to_string(),
                                    t: input.t,
                                    reason: "shadow_compute_failed".to_string(),
                                    consecutive_failures,
                                },
                            ));
                        }
                    }
                }
            }
        }

        if (input.t + 1).is_multiple_of(self.cfg.compare_window) {
            let flushed = std::mem::replace(
                &mut rt.compare_window_state,
                CompareWindowState::new(input.t.saturating_add(1)),
            );
            let shadow_disabled = rt.shadow_disabled;
            rt.outbox
                .push(SlotShadowEventV1::CompareWindow(flushed.flush(
                    "compute",
                    input.t,
                    shadow_disabled,
                )));
        }

        Ok(primary.0)
    }

    fn drain_shadow_events(&self) -> Vec<SlotShadowEventV1> {
        let mut rt = self.runtime.lock().expect("shadow runtime lock poisoned");
        std::mem::take(&mut rt.outbox)
    }
}

fn envelope_ok(signals: &ComputeSignals) -> bool {
    let mut ok = signals.surprise.is_finite()
        && signals.pressure.is_finite()
        && signals.risk.is_finite()
        && (0.0..=1.0).contains(&signals.surprise)
        && (0.0..=1.0).contains(&signals.pressure)
        && (0.0..=1.0).contains(&signals.risk);
    if let Some(u) = signals.lfm_uncertainty {
        ok = ok && u.is_finite() && (0.0..=1.0).contains(&u);
    }
    ok
}

fn digest_prefix(signals: &ComputeSignals) -> String {
    let mut hasher = Sha256::new();
    hasher.update(signals.surprise.to_bits().to_le_bytes());
    hasher.update(signals.pressure.to_bits().to_le_bytes());
    hasher.update(signals.risk.to_bits().to_le_bytes());
    hasher.update(signals.confidence.to_bits().to_le_bytes());
    let digest = hasher.finalize();
    hex::encode(&digest[..4])
}

fn digest_prefix4(signals: &ComputeSignals) -> [u8; 4] {
    let mut out = [0_u8; 4];
    out.copy_from_slice(&Sha256::digest(digest_prefix(signals).as_bytes())[..4]);
    out
}

fn mean_q(values: &[u16]) -> u16 {
    if values.is_empty() {
        return 0;
    }
    let sum: u64 = values.iter().map(|v| u64::from(*v)).sum();
    (sum / values.len() as u64) as u16
}

fn p95_q(mut values: Vec<u16>) -> u16 {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    let idx = ((values.len() - 1) * 95) / 100;
    values[idx]
}

fn slot_mode_from_enablement(mode: RealEnablementMode) -> SlotModeV1 {
    match mode {
        RealEnablementMode::Off => SlotModeV1::Off,
        RealEnablementMode::Shadow | RealEnablementMode::Compare => SlotModeV1::Shadow,
        RealEnablementMode::Active => SlotModeV1::Active,
    }
}

fn shadow_phase_offset(run_id: u64, slot_id: &str, rate: u64) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(run_id.to_le_bytes());
    hasher.update(slot_id.as_bytes());
    let digest = hasher.finalize();
    let mut first8 = [0_u8; 8];
    first8.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(first8) % rate.max(1)
}

pub fn should_run_shadow(run_id: u64, slot_id: &str, t: u64, rate: u64) -> bool {
    let r = rate.max(1);
    let offset = shadow_phase_offset(run_id, slot_id, r);
    should_run_shadow_at(t, offset, r)
}

fn should_run_shadow_at(t: u64, offset: u64, rate: u64) -> bool {
    t % rate.max(1) == offset % rate.max(1)
}

pub fn parse_stage_ladder(raw: &str) -> BTreeMap<u64, SlotEnablement> {
    let mut out = BTreeMap::new();
    for entry in raw.split(';').map(str::trim).filter(|s| !s.is_empty()) {
        let parts: Vec<_> = entry.split('@').collect();
        if parts.len() != 2 {
            continue;
        }
        let tick = parts[1].trim_start_matches('t').parse::<u64>();
        let Ok(t) = tick else { continue };
        let mut state = SlotEnablement::default();
        match parts[0].trim().to_ascii_lowercase().as_str() {
            "phase1" => state.llm = SlotMode::Shadow,
            "phase2" => {
                state.llm = SlotMode::Active;
                state.lfm = SlotMode::Shadow;
            }
            "phase3" => {
                state.llm = SlotMode::Active;
                state.lfm = SlotMode::Active;
                state.world_jepa = SlotMode::Shadow;
            }
            _ => continue,
        }
        out.insert(t, state);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ladder_deterministic() {
        let parsed = parse_stage_ladder("phase1@t0;phase2@t64;phase3@t128");
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed.get(&0).expect("p1").llm, SlotMode::Shadow);
        assert_eq!(parsed.get(&64).expect("p2").lfm, SlotMode::Shadow);
        assert_eq!(parsed.get(&128).expect("p3").world_jepa, SlotMode::Shadow);
    }

    #[test]
    fn envelope_bounds() {
        let signals = ComputeSignals::unavailable(
            &ComputeInput {
                frame_id: crate::FrameId(1),
                t: 1,
                context_digest: [0; 32],
            },
            ComputeBudget::default(),
            "toy",
        );
        assert!(envelope_ok(&signals));
    }

    #[test]
    fn should_run_shadow_is_deterministic() {
        let a: Vec<u64> = (0..32)
            .filter(|t| should_run_shadow(7, "compute", *t, 4))
            .collect();
        let b: Vec<u64> = (0..32)
            .filter(|t| should_run_shadow(7, "compute", *t, 4))
            .collect();
        assert_eq!(a, b);
        assert_eq!(a.len(), 8);
    }

    #[test]
    fn compare_window_aggregate_is_deterministic() {
        let s = vec![10, 20, 30, 40, 50];
        assert_eq!(mean_q(&s), 30);
        assert_eq!(p95_q(s), 40);
    }

    #[derive(Clone)]
    struct MockBackend {
        risk: f32,
        fail: bool,
    }

    impl AiComputeBackend for MockBackend {
        fn name(&self) -> &'static str {
            "mock"
        }

        fn compute(
            &self,
            input: &ComputeInput,
            budget: ComputeBudget,
        ) -> Result<ComputeSignals, ComputeError> {
            if self.fail {
                return Err(ComputeError::InvalidInput {
                    reason: "forced".to_string(),
                });
            }
            let mut out = ComputeSignals::unavailable(input, budget, "mock");
            out.risk = self.risk;
            Ok(out)
        }
    }

    #[test]
    fn shadow_does_not_change_primary_output() {
        let input = ComputeInput {
            frame_id: crate::FrameId(1),
            t: 3,
            context_digest: [0; 32],
        };
        let budget = ComputeBudget::default();
        let backend = EnablementComputeBackend::new(
            Box::new(MockBackend {
                risk: 0.2,
                fail: false,
            }),
            Some(Box::new(MockBackend {
                risk: 0.8,
                fail: false,
            })),
            EnablementConfig {
                mode: RealEnablementMode::Shadow,
                shadow_every_n_ticks: 1,
                shadow_rate: 1,
                compare_window: 8,
            },
        );
        let out = backend.compute(&input, budget).expect("compute ok");
        assert_eq!(out.risk, 0.2);
    }

    #[test]
    fn shadow_failure_never_breaks_primary() {
        let input = ComputeInput {
            frame_id: crate::FrameId(1),
            t: 1,
            context_digest: [0; 32],
        };
        let budget = ComputeBudget::default();
        let backend = EnablementComputeBackend::new(
            Box::new(MockBackend {
                risk: 0.3,
                fail: false,
            }),
            Some(Box::new(MockBackend {
                risk: 0.9,
                fail: true,
            })),
            EnablementConfig {
                mode: RealEnablementMode::Shadow,
                shadow_every_n_ticks: 1,
                shadow_rate: 1,
                compare_window: 8,
            },
        );
        for t in 1..=3 {
            let mut in2 = input.clone();
            in2.t = t;
            let out = backend
                .compute(&in2, budget)
                .expect("primary must continue");
            assert_eq!(out.risk, 0.3);
        }
        let events = backend.drain_shadow_events();
        assert!(events
            .iter()
            .any(|e| matches!(e, SlotShadowEventV1::ShadowDisable(_))));
    }
}
