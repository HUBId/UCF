#![forbid(unsafe_code)]

use dbm_core::{
    BaselineVector, CooldownClass, IntegrityState, IsvSnapshot, LevelClass, OverlaySet,
    ProfileState, ReasonSet, ThreatVector,
};
use microcircuit_core::{digest_config, digest_meta, CircuitConfig, MicrocircuitBackend};
use microcircuit_pag_stub::DefensePattern;
use microcircuit_pmrf_stub::SequenceMode;

const STRICTNESS_MIN: i32 = 0;
const STRICTNESS_MAX: i32 = 100;
const QUARANTINE_LATCH_MAX: u8 = 20;
const FORENSIC_STRICTNESS_FLOOR: i32 = 70;

const PROFILE_M0: u8 = 0;
const PROFILE_M1: u8 = 1;
const PROFILE_M2: u8 = 2;
const PROFILE_M3: u8 = 3;

#[derive(Debug, Clone)]
pub struct HypoInput {
    pub isv: IsvSnapshot,
    pub pag_pattern: Option<DefensePattern>,
    pub stn_hold_active: bool,
    pub pmrf_sequence_mode: SequenceMode,
    pub baseline: BaselineVector,
    pub unlock_present: bool,
    pub unlock_ready: bool,
    pub now_ms: u64,
}

impl Default for HypoInput {
    fn default() -> Self {
        Self {
            isv: IsvSnapshot::default(),
            pag_pattern: None,
            stn_hold_active: false,
            pmrf_sequence_mode: SequenceMode::Normal,
            baseline: BaselineVector::default(),
            unlock_present: false,
            unlock_ready: false,
            now_ms: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HypoOutput {
    pub profile_state: ProfileState,
    pub overlays: OverlaySet,
    pub deescalation_lock: bool,
    pub cooldown_class: LevelClass,
    pub reason_codes: ReasonSet,
}

impl Default for HypoOutput {
    fn default() -> Self {
        Self {
            profile_state: ProfileState::M0,
            overlays: OverlaySet::default(),
            deescalation_lock: false,
            cooldown_class: LevelClass::Low,
            reason_codes: ReasonSet::default(),
        }
    }
}

#[derive(Debug, Clone)]
struct HypoState {
    setpoint_strictness: i32,
    setpoint_profile: u8,
    drive_simulate: i32,
    drive_export_lock: i32,
    drive_novelty_lock: i32,
    forensic_latched: bool,
    quarantine_latched_steps: u8,
    step_count: u64,
}

impl Default for HypoState {
    fn default() -> Self {
        Self {
            setpoint_strictness: 0,
            setpoint_profile: PROFILE_M0,
            drive_simulate: 0,
            drive_export_lock: 0,
            drive_novelty_lock: 0,
            forensic_latched: false,
            quarantine_latched_steps: 0,
            step_count: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HypothalamusSetpointMicrocircuit {
    config: CircuitConfig,
    state: HypoState,
}

impl HypothalamusSetpointMicrocircuit {
    pub fn new(config: CircuitConfig) -> Self {
        Self {
            config,
            state: HypoState::default(),
        }
    }

    fn level_drive(level: LevelClass, med: i32, high: i32) -> i32 {
        match level {
            LevelClass::Med => med,
            LevelClass::High => high,
            LevelClass::Low => 0,
        }
    }

    fn is_dp2(pattern: Option<DefensePattern>) -> bool {
        matches!(pattern, Some(DefensePattern::DP2_QUARANTINE))
    }

    fn is_dp3(pattern: Option<DefensePattern>) -> bool {
        matches!(pattern, Some(DefensePattern::DP3_FORENSIC))
    }

    fn split_required(mode: SequenceMode) -> bool {
        matches!(mode, SequenceMode::SplitRequired)
    }

    fn receipt_failures_present(isv: &IsvSnapshot) -> bool {
        isv.dominant_reason_codes.codes.iter().any(|code| {
            let lower = code.to_ascii_lowercase();
            lower.contains("receipt") || lower.contains("dispatch_blocked")
        })
    }

    fn strictness_drive(&self, input: &HypoInput) -> i32 {
        let mut sd = 0i32;

        match input.isv.integrity {
            IntegrityState::Fail => sd += 80,
            IntegrityState::Degraded => sd += 40,
            IntegrityState::Ok => {}
        }

        match input.isv.threat {
            LevelClass::High => sd += 60,
            LevelClass::Med => sd += 30,
            LevelClass::Low => {}
        }

        if input.isv.policy_pressure == LevelClass::High {
            sd += 30;
        }

        if input.isv.arousal == LevelClass::High {
            sd += 20;
        }

        if input.isv.stability == LevelClass::High {
            sd += 20;
        }

        if Self::is_dp3(input.pag_pattern) {
            sd += 80;
        } else if Self::is_dp2(input.pag_pattern) {
            sd += 60;
        }

        if input.stn_hold_active {
            sd += 20;
        }

        if Self::split_required(input.pmrf_sequence_mode) {
            sd += 20;
        }

        sd += Self::level_drive(input.baseline.caution_floor, 10, 20);
        sd += Self::level_drive(input.baseline.export_strictness, 10, 20);
        sd += Self::level_drive(input.baseline.approval_strictness, 10, 20);
        sd += Self::level_drive(input.baseline.chain_conservatism, 10, 20);

        sd = sd.clamp(STRICTNESS_MIN, STRICTNESS_MAX);

        if input.unlock_present && input.unlock_ready && input.isv.integrity != IntegrityState::Fail
        {
            sd = (sd - 10).max(STRICTNESS_MIN);
        }

        sd
    }

    fn update_latches(&mut self, input: &HypoInput) {
        if input.isv.integrity == IntegrityState::Fail || Self::is_dp3(input.pag_pattern) {
            self.state.forensic_latched = true;
            self.state.setpoint_profile = PROFILE_M3;
        }

        if Self::is_dp2(input.pag_pattern) || input.isv.threat == LevelClass::High {
            self.state.quarantine_latched_steps = QUARANTINE_LATCH_MAX;
        } else if self.state.quarantine_latched_steps > 0 {
            self.state.quarantine_latched_steps -= 1;
        }
    }

    fn desired_profile(&self, input: &HypoInput) -> u8 {
        if self.state.forensic_latched && !input.unlock_ready {
            return PROFILE_M3;
        }

        if Self::is_dp3(input.pag_pattern) {
            return PROFILE_M3;
        }

        if Self::is_dp2(input.pag_pattern)
            || input.isv.threat == LevelClass::High
            || self.state.quarantine_latched_steps > 0
        {
            return PROFILE_M2;
        }

        if self.state.setpoint_strictness >= 50
            || input.isv.policy_pressure == LevelClass::High
            || Self::receipt_failures_present(&input.isv)
        {
            return PROFILE_M1;
        }

        PROFILE_M0
    }

    fn apply_profile_hysteresis(&mut self, desired: u8, unlock_ready: bool) {
        let current = self.state.setpoint_profile;
        let effective = if self.state.forensic_latched && unlock_ready {
            desired
        } else if current > desired {
            current
        } else {
            desired
        };

        self.state.setpoint_profile = effective;
    }

    fn profile_state(profile: u8) -> ProfileState {
        match profile {
            PROFILE_M3 => ProfileState::M3,
            PROFILE_M2 => ProfileState::M2,
            PROFILE_M1 => ProfileState::M1,
            _ => ProfileState::M0,
        }
    }

    fn drive_max(values: [i32; 5]) -> i32 {
        values.into_iter().max().unwrap_or(0).clamp(0, 100)
    }

    fn update_overlay_drives(&mut self, input: &HypoInput) {
        let simulate_drive = Self::drive_max([
            if input.stn_hold_active { 80 } else { 0 },
            if Self::split_required(input.pmrf_sequence_mode) {
                80
            } else {
                0
            },
            if input.isv.policy_pressure == LevelClass::High {
                60
            } else {
                0
            },
            if input.isv.threat == LevelClass::High {
                70
            } else {
                0
            },
            if input.baseline.chain_conservatism == LevelClass::High {
                60
            } else {
                0
            },
        ]);

        let exfil_present = input
            .isv
            .threat_vectors
            .as_ref()
            .map(|vectors| vectors.contains(&ThreatVector::Exfil))
            .unwrap_or(false);

        let export_drive = Self::drive_max([
            if exfil_present { 90 } else { 0 },
            if input.baseline.export_strictness == LevelClass::High {
                60
            } else {
                0
            },
            0,
            0,
            0,
        ]);

        let novelty_drive = Self::drive_max([
            if input.isv.arousal == LevelClass::High {
                70
            } else {
                0
            },
            if input.baseline.novelty_dampening == LevelClass::High {
                60
            } else {
                0
            },
            if input.isv.policy_pressure == LevelClass::High {
                50
            } else {
                0
            },
            0,
            0,
        ]);

        self.state.drive_simulate = simulate_drive;
        self.state.drive_export_lock = export_drive;
        self.state.drive_novelty_lock = novelty_drive;
    }

    fn overlay_set(&self, profile: ProfileState) -> OverlaySet {
        let mut overlays = OverlaySet {
            simulate_first: self.state.drive_simulate >= 60,
            export_lock: self.state.drive_export_lock >= 60,
            novelty_lock: self.state.drive_novelty_lock >= 60,
        };

        match profile {
            ProfileState::M3 | ProfileState::M2 => {
                overlays = OverlaySet::all_enabled();
            }
            ProfileState::M1 => {
                overlays.simulate_first = true;
            }
            ProfileState::M0 => {}
        }

        overlays
    }

    fn cooldown_level(&self, input: &HypoInput, profile: ProfileState) -> LevelClass {
        if input.isv.stability == LevelClass::High
            || input.baseline.cooldown_bias == CooldownClass::Longer
            || matches!(profile, ProfileState::M2 | ProfileState::M3)
        {
            LevelClass::High
        } else {
            LevelClass::Low
        }
    }

    fn reason_codes(&self, input: &HypoInput, profile: ProfileState) -> ReasonSet {
        let mut reason_codes = input.isv.dominant_reason_codes.clone();

        match profile {
            ProfileState::M3 => reason_codes.insert("RC.RX.ACTION.FORENSIC"),
            ProfileState::M2 => reason_codes.insert("RC.RX.ACTION.QUARANTINE"),
            ProfileState::M1 => reason_codes.insert("RC.RG.PROFILE.M1_RESTRICTED"),
            ProfileState::M0 => {}
        }

        if input.unlock_ready {
            reason_codes.insert("RC.GV.RECOVERY.UNLOCK_GRANTED");
        }

        reason_codes
    }
}

impl MicrocircuitBackend<HypoInput, HypoOutput> for HypothalamusSetpointMicrocircuit {
    fn step(&mut self, input: &HypoInput, _now_ms: u64) -> HypoOutput {
        self.state.step_count = self.state.step_count.saturating_add(1);

        let sd = self.strictness_drive(input);
        let delta = (sd - self.state.setpoint_strictness) / 4;
        self.state.setpoint_strictness =
            (self.state.setpoint_strictness + delta).clamp(STRICTNESS_MIN, STRICTNESS_MAX);

        self.update_latches(input);

        if self.state.forensic_latched && input.unlock_ready {
            self.state.setpoint_strictness = self
                .state
                .setpoint_strictness
                .max(FORENSIC_STRICTNESS_FLOOR);
        }

        let mut desired = self.desired_profile(input);
        if self.state.forensic_latched && input.unlock_ready {
            desired = desired.max(PROFILE_M1);
        }

        self.apply_profile_hysteresis(desired, input.unlock_ready);

        let profile_state = Self::profile_state(self.state.setpoint_profile);

        self.update_overlay_drives(input);
        let overlays = self.overlay_set(profile_state);

        let deescalation_lock = profile_state != ProfileState::M0
            || input.isv.stability == LevelClass::High
            || self.state.forensic_latched;

        let cooldown_class = self.cooldown_level(input, profile_state);
        let reason_codes = self.reason_codes(input, profile_state);

        HypoOutput {
            profile_state,
            overlays,
            deescalation_lock,
            cooldown_class,
            reason_codes,
        }
    }

    fn snapshot_digest(&self) -> [u8; 32] {
        let mut bytes = Vec::new();
        bytes.extend(self.state.setpoint_strictness.to_le_bytes());
        bytes.push(self.state.setpoint_profile);
        bytes.extend(self.state.drive_simulate.to_le_bytes());
        bytes.extend(self.state.drive_export_lock.to_le_bytes());
        bytes.extend(self.state.drive_novelty_lock.to_le_bytes());
        bytes.push(self.state.forensic_latched as u8);
        bytes.push(self.state.quarantine_latched_steps);
        bytes.extend(self.state.step_count.to_le_bytes());

        digest_meta("UCF:MC:HYPO", &bytes)
    }

    fn config_digest(&self) -> [u8; 32] {
        digest_config("UCF:MC:HYPO:CFG", &self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dbm_core::IntegrityState;
    use microcircuit_core::CircuitConfig;

    fn base_input() -> HypoInput {
        HypoInput::default()
    }

    #[test]
    fn deterministic_sequence_matches() {
        let mut circuit_a = HypothalamusSetpointMicrocircuit::new(CircuitConfig::default());
        let mut circuit_b = HypothalamusSetpointMicrocircuit::new(CircuitConfig::default());

        let inputs = vec![
            HypoInput::default(),
            HypoInput {
                isv: IsvSnapshot {
                    threat: LevelClass::High,
                    ..IsvSnapshot::default()
                },
                ..HypoInput::default()
            },
            HypoInput {
                isv: IsvSnapshot {
                    integrity: IntegrityState::Degraded,
                    ..IsvSnapshot::default()
                },
                stn_hold_active: true,
                ..HypoInput::default()
            },
        ];

        for input in inputs {
            let output_a = circuit_a.step(&input, input.now_ms);
            let output_b = circuit_b.step(&input, input.now_ms);
            assert_eq!(output_a, output_b);
        }
    }

    #[test]
    fn integrity_fail_forces_m3_and_overlays() {
        let mut circuit = HypothalamusSetpointMicrocircuit::new(CircuitConfig::default());
        let input = HypoInput {
            isv: IsvSnapshot {
                integrity: IntegrityState::Fail,
                ..IsvSnapshot::default()
            },
            ..base_input()
        };
        let output = circuit.step(&input, 0);

        assert_eq!(output.profile_state, ProfileState::M3);
        assert!(output.overlays.simulate_first);
        assert!(output.overlays.export_lock);
        assert!(output.overlays.novelty_lock);
    }

    #[test]
    fn threat_high_enforces_m2_or_higher() {
        let mut circuit = HypothalamusSetpointMicrocircuit::new(CircuitConfig::default());
        let input = HypoInput {
            isv: IsvSnapshot {
                threat: LevelClass::High,
                ..IsvSnapshot::default()
            },
            ..base_input()
        };
        let output = circuit.step(&input, 0);

        assert!(matches!(
            output.profile_state,
            ProfileState::M2 | ProfileState::M3
        ));
        assert!(output.overlays.simulate_first);
    }

    #[test]
    fn unlock_ready_allows_m3_to_m1_only() {
        let mut circuit = HypothalamusSetpointMicrocircuit::new(CircuitConfig::default());
        let fail_input = HypoInput {
            isv: IsvSnapshot {
                integrity: IntegrityState::Fail,
                ..IsvSnapshot::default()
            },
            ..base_input()
        };
        let output = circuit.step(&fail_input, 0);
        assert_eq!(output.profile_state, ProfileState::M3);

        let unlock_input = HypoInput {
            unlock_present: true,
            unlock_ready: true,
            isv: IsvSnapshot {
                integrity: IntegrityState::Ok,
                stability: LevelClass::High,
                ..IsvSnapshot::default()
            },
            ..base_input()
        };

        let output = circuit.step(&unlock_input, 0);
        assert_eq!(output.profile_state, ProfileState::M1);
        assert_ne!(output.profile_state, ProfileState::M0);
    }

    #[test]
    fn tighten_only_keeps_stricter_profile() {
        let mut circuit = HypothalamusSetpointMicrocircuit::new(CircuitConfig::default());
        let strict_input = HypoInput {
            isv: IsvSnapshot {
                threat: LevelClass::High,
                ..IsvSnapshot::default()
            },
            ..base_input()
        };
        let output = circuit.step(&strict_input, 0);
        assert_eq!(output.profile_state, ProfileState::M2);

        let relaxed_input = HypoInput::default();
        let output = circuit.step(&relaxed_input, 0);
        assert_eq!(output.profile_state, ProfileState::M2);
    }
}
