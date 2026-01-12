#![forbid(unsafe_code)]

use biophys_core::{ModLevel, ModulatorField};
use ucf::v1::{ReasonCode, SignalFrame};

/// Maximum governance reason codes retained per cooldown window.
pub const MAX_GOVERNANCE_REASON_CODES: usize = 8;
/// Plasticity scaling (0..1000) during governance cooldown.
pub const COOLDOWN_PLASTICITY_SCALE_Q: u16 = 100;
/// External spike amplitude cap (0..1000) during governance cooldown.
pub const COOLDOWN_INJECTION_AMPLITUDE_CAP_Q: u16 = 600;
/// Maximum fan-out per spike during governance cooldown.
pub const COOLDOWN_MAX_TARGETS_PER_SPIKE: usize = 8;
/// Maximum modulator level step change per tick during cooldown.
pub const COOLDOWN_MODULATOR_MAX_STEP: i8 = 1;

/// Reason code for SAE pack updates.
const REASON_SAE_PACK_UPDATED: &str = "RC.GV.SAE.PACK_UPDATED";
/// Reason code for mapping updates.
const REASON_MAP_UPDATED: &str = "RC.GV.MAP.UPDATED";
/// Reason code for liquid parameter updates.
const REASON_LIQUID_PARAMS_UPDATED: &str = "RC.GV.LIQUID.PARAMS_UPDATED";
/// Reason code for injection limit updates.
const REASON_INJECTION_LIMITS_UPDATED: &str = "RC.GV.INJECTION.LIMITS_UPDATED";

/// Cooldown duration for mapping updates.
const COOLDOWN_MAPPING_UPDATED_TICKS: u32 = 500;
/// Cooldown duration for SAE pack updates.
const COOLDOWN_SAE_PACK_UPDATED_TICKS: u32 = 800;
/// Cooldown duration for liquid parameter updates.
const COOLDOWN_LIQUID_PARAMS_UPDATED_TICKS: u32 = 1200;
/// Cooldown duration for injection limit updates.
const COOLDOWN_INJECTION_LIMITS_UPDATED_TICKS: u32 = 200;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateKind {
    MappingUpdated,
    SaePackUpdated,
    LiquidParamsUpdated,
    InjectionLimitsUpdated,
    Other,
}

impl UpdateKind {
    pub fn cooldown_ticks(self) -> u32 {
        match self {
            UpdateKind::MappingUpdated => COOLDOWN_MAPPING_UPDATED_TICKS,
            UpdateKind::SaePackUpdated => COOLDOWN_SAE_PACK_UPDATED_TICKS,
            UpdateKind::LiquidParamsUpdated => COOLDOWN_LIQUID_PARAMS_UPDATED_TICKS,
            UpdateKind::InjectionLimitsUpdated => COOLDOWN_INJECTION_LIMITS_UPDATED_TICKS,
            UpdateKind::Other => 0,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GovernanceUpdateState {
    pub cooldown_ticks_remaining: u32,
    pub last_update_digest: Option<[u8; 32]>,
    pub last_update_kind: Option<UpdateKind>,
    pub reason_codes: Vec<String>,
}

impl GovernanceUpdateState {
    pub fn ingest_signal_frame(&mut self, frame: &SignalFrame) {
        let mut strongest_kind = None;
        let mut strongest_ticks = 0u32;
        for code in frame
            .top_reason_codes
            .iter()
            .take(MAX_GOVERNANCE_REASON_CODES)
        {
            let Ok(reason) = ReasonCode::try_from(*code) else {
                continue;
            };
            if let Some(kind) = update_kind_for_reason(reason) {
                let ticks = kind.cooldown_ticks();
                if ticks > strongest_ticks {
                    strongest_ticks = ticks;
                    strongest_kind = Some(kind);
                }
                if let Some(label) = reason_label(reason) {
                    self.push_reason_code(label);
                }
            }
        }

        if let Some(kind) = strongest_kind {
            self.cooldown_ticks_remaining = self.cooldown_ticks_remaining.max(strongest_ticks);
            self.last_update_kind = Some(kind);
            self.last_update_digest = frame
                .signal_frame_digest
                .as_ref()
                .and_then(|bytes| <[u8; 32]>::try_from(bytes.as_slice()).ok());
        }
    }

    pub fn tick(&mut self) {
        if self.cooldown_ticks_remaining > 0 {
            self.cooldown_ticks_remaining = self.cooldown_ticks_remaining.saturating_sub(1);
        }
    }

    pub fn cooldown_active(&self) -> bool {
        self.cooldown_ticks_remaining > 0
    }

    pub fn plasticity_scale_q(&self) -> u16 {
        if self.cooldown_active() {
            COOLDOWN_PLASTICITY_SCALE_Q
        } else {
            1000
        }
    }

    pub fn injection_amplitude_cap_q(&self) -> u16 {
        if self.cooldown_active() {
            COOLDOWN_INJECTION_AMPLITUDE_CAP_Q
        } else {
            1000
        }
    }

    pub fn max_targets_per_spike(&self, baseline: usize) -> usize {
        if self.cooldown_active() {
            COOLDOWN_MAX_TARGETS_PER_SPIKE.min(baseline)
        } else {
            baseline
        }
    }

    pub fn stabilize_modulators(
        &self,
        previous: ModulatorField,
        target: ModulatorField,
    ) -> ModulatorField {
        if !self.cooldown_active() {
            return target;
        }
        ModulatorField {
            na: clamp_mod_level(previous.na, target.na, COOLDOWN_MODULATOR_MAX_STEP),
            da: clamp_mod_level(previous.da, target.da, COOLDOWN_MODULATOR_MAX_STEP),
            ht: clamp_mod_level(previous.ht, target.ht, COOLDOWN_MODULATOR_MAX_STEP),
        }
    }

    fn push_reason_code(&mut self, code: &str) {
        self.reason_codes.push(code.to_string());
        self.reason_codes.sort();
        self.reason_codes.dedup();
        if self.reason_codes.len() > MAX_GOVERNANCE_REASON_CODES {
            self.reason_codes.truncate(MAX_GOVERNANCE_REASON_CODES);
        }
    }
}

fn update_kind_for_reason(reason: ReasonCode) -> Option<UpdateKind> {
    match reason {
        ReasonCode::RcGvMapUpdated => Some(UpdateKind::MappingUpdated),
        ReasonCode::RcGvSaePackUpdated => Some(UpdateKind::SaePackUpdated),
        ReasonCode::RcGvLiquidParamsUpdated => Some(UpdateKind::LiquidParamsUpdated),
        ReasonCode::RcGvInjectionLimitsUpdated => Some(UpdateKind::InjectionLimitsUpdated),
        _ => None,
    }
}

fn reason_label(reason: ReasonCode) -> Option<&'static str> {
    match reason {
        ReasonCode::RcGvMapUpdated => Some(REASON_MAP_UPDATED),
        ReasonCode::RcGvSaePackUpdated => Some(REASON_SAE_PACK_UPDATED),
        ReasonCode::RcGvLiquidParamsUpdated => Some(REASON_LIQUID_PARAMS_UPDATED),
        ReasonCode::RcGvInjectionLimitsUpdated => Some(REASON_INJECTION_LIMITS_UPDATED),
        _ => None,
    }
}

fn clamp_mod_level(previous: ModLevel, target: ModLevel, max_step: i8) -> ModLevel {
    let prev_value = mod_level_to_i8(previous);
    let target_value = mod_level_to_i8(target);
    let delta = (target_value - prev_value).clamp(-max_step, max_step);
    mod_level_from_i8(prev_value + delta)
}

fn mod_level_to_i8(level: ModLevel) -> i8 {
    match level {
        ModLevel::Low => 0,
        ModLevel::Med => 1,
        ModLevel::High => 2,
    }
}

fn mod_level_from_i8(value: i8) -> ModLevel {
    match value.clamp(0, 2) {
        0 => ModLevel::Low,
        1 => ModLevel::Med,
        _ => ModLevel::High,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sae_pack_update_sets_cooldown() {
        let frame = SignalFrame {
            window_kind: 0,
            window_index: None,
            timestamp_ms: None,
            policy_stats: None,
            exec_stats: None,
            integrity_state: 0,
            top_reason_codes: vec![ReasonCode::RcGvSaePackUpdated as i32],
            signal_frame_digest: Some(vec![7u8; 32]),
            receipt_stats: None,
            reason_codes: Vec::new(),
        };
        let mut state = GovernanceUpdateState::default();
        state.ingest_signal_frame(&frame);

        assert_eq!(
            state.cooldown_ticks_remaining,
            COOLDOWN_SAE_PACK_UPDATED_TICKS
        );
        assert_eq!(state.last_update_kind, Some(UpdateKind::SaePackUpdated));
        assert_eq!(state.last_update_digest, Some([7u8; 32]));
    }

    #[test]
    fn cooldown_determinism_for_same_sequence() {
        let mut state_a = GovernanceUpdateState::default();
        let mut state_b = GovernanceUpdateState::default();
        let frames = [
            SignalFrame {
                window_kind: 0,
                window_index: None,
                timestamp_ms: None,
                policy_stats: None,
                exec_stats: None,
                integrity_state: 0,
                top_reason_codes: vec![ReasonCode::RcGvMapUpdated as i32],
                signal_frame_digest: None,
                receipt_stats: None,
                reason_codes: Vec::new(),
            },
            SignalFrame {
                window_kind: 0,
                window_index: None,
                timestamp_ms: None,
                policy_stats: None,
                exec_stats: None,
                integrity_state: 0,
                top_reason_codes: vec![ReasonCode::RcGvInjectionLimitsUpdated as i32],
                signal_frame_digest: None,
                receipt_stats: None,
                reason_codes: Vec::new(),
            },
        ];

        for frame in &frames {
            state_a.ingest_signal_frame(frame);
            state_b.ingest_signal_frame(frame);
            state_a.tick();
            state_b.tick();
        }

        assert_eq!(
            state_a.cooldown_ticks_remaining,
            state_b.cooldown_ticks_remaining
        );
        assert_eq!(state_a.last_update_kind, state_b.last_update_kind);
    }

    #[test]
    fn reason_codes_are_bounded() {
        let mut state = GovernanceUpdateState::default();
        for _ in 0..(MAX_GOVERNANCE_REASON_CODES + 4) {
            let frame = SignalFrame {
                window_kind: 0,
                window_index: None,
                timestamp_ms: None,
                policy_stats: None,
                exec_stats: None,
                integrity_state: 0,
                top_reason_codes: vec![
                    ReasonCode::RcGvMapUpdated as i32,
                    ReasonCode::RcGvSaePackUpdated as i32,
                    ReasonCode::RcGvLiquidParamsUpdated as i32,
                    ReasonCode::RcGvInjectionLimitsUpdated as i32,
                ],
                signal_frame_digest: None,
                receipt_stats: None,
                reason_codes: Vec::new(),
            };
            state.ingest_signal_frame(&frame);
        }

        assert!(state.reason_codes.len() <= MAX_GOVERNANCE_REASON_CODES);
    }
}
