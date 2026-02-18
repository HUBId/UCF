use std::collections::BTreeMap;
use std::time::Instant;

use sha2::{Digest, Sha256};

use crate::{
    AiComputeBackend, ComputeBudget, ComputeError, ComputeInput, ComputeSignals, ModelSlot,
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
}

impl Default for EnablementConfig {
    fn default() -> Self {
        Self {
            mode: RealEnablementMode::Off,
            shadow_every_n_ticks: 4,
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

pub struct EnablementComputeBackend {
    primary: Box<dyn AiComputeBackend + Send + Sync>,
    shadow: Option<Box<dyn AiComputeBackend + Send + Sync>>,
    cfg: EnablementConfig,
}

impl EnablementComputeBackend {
    pub fn new(
        primary: Box<dyn AiComputeBackend + Send + Sync>,
        shadow: Option<Box<dyn AiComputeBackend + Send + Sync>>,
        cfg: EnablementConfig,
    ) -> Self {
        Self {
            primary,
            shadow,
            cfg,
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
        let primary = self.primary.compute(input, budget)?;
        metrics::gauge!("ucf_slot_active", "slot" => "compute").set(
            if self.cfg.mode == RealEnablementMode::Active {
                1.0
            } else {
                0.0
            },
        );
        if self.cfg.mode == RealEnablementMode::Off {
            return Ok(primary);
        }
        if let Some(shadow) = &self.shadow {
            if input.t % self.cfg.shadow_every_n_ticks == 0 {
                metrics::counter!("ucf_shadow_runs_total", "slot" => "compute").increment(1);
                let started = Instant::now();
                match shadow.compute(
                    input,
                    ComputeBudget {
                        max_micros: budget.max_micros / 2,
                        ..budget
                    },
                ) {
                    Ok(shadow_out) => {
                        let elapsed_ms = started.elapsed().as_millis() as u64;
                        let _rec = ShadowComparisonRecord {
                            t: input.t,
                            toy_digest_prefix: digest_prefix(&primary),
                            real_digest_prefix: digest_prefix(&shadow_out),
                            elapsed_ms,
                            result: "ok",
                        };
                        if !envelope_ok(&shadow_out) {
                            metrics::counter!("ucf_compare_envelope_violation_total", "slot" => "compute").increment(1);
                        }
                    }
                    Err(_) => {
                        metrics::counter!("ucf_shadow_timeouts_total", "slot" => "compute")
                            .increment(1);
                    }
                }
            }
        }
        Ok(primary)
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
}
