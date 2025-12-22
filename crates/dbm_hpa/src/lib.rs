#![forbid(unsafe_code)]

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use dbm_core::{DbmComponent, IntegrityState, LevelClass, ReasonSet};
use serde::{Deserialize, Serialize};

const MAX_CELL_VALUE: u16 = 100;
const STABLE_DECAY_INTERVAL: u16 = 5;
const OFFSET_MAX: u16 = 10;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct HpaState {
    pub allostatic_load: u16,
    pub exfil_sensitivity: u16,
    pub integrity_sensitivity: u16,
    pub probing_sensitivity: u16,
    pub reliability_sensitivity: u16,
    pub recovery_confidence: u16,
    pub stable_window_counter: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HpaInput {
    pub integrity_state: IntegrityState,
    pub replay_mismatch_present: bool,
    pub dlp_critical_present: bool,
    pub receipt_invalid_present: bool,
    pub deny_storm_present: bool,
    pub timeouts_burst_present: bool,
    pub unlock_present: bool,
    pub stable_medium_window: bool,
}

impl Default for HpaInput {
    fn default() -> Self {
        Self {
            integrity_state: IntegrityState::Ok,
            replay_mismatch_present: false,
            dlp_critical_present: false,
            receipt_invalid_present: false,
            deny_storm_present: false,
            timeouts_burst_present: false,
            unlock_present: false,
            stable_medium_window: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HpaOutput {
    pub allostatic_load_class: LevelClass,
    pub baseline_caution_offset: u16,
    pub baseline_novelty_dampening_offset: u16,
    pub baseline_approval_strictness_offset: u16,
    pub baseline_export_strictness_offset: u16,
    pub baseline_chain_conservatism_offset: u16,
    pub baseline_cooldown_multiplier_class: u16,
    pub reason_codes: ReasonSet,
}

impl Default for HpaOutput {
    fn default() -> Self {
        Self {
            allostatic_load_class: LevelClass::Low,
            baseline_caution_offset: 0,
            baseline_novelty_dampening_offset: 0,
            baseline_approval_strictness_offset: 0,
            baseline_export_strictness_offset: 0,
            baseline_chain_conservatism_offset: 0,
            baseline_cooldown_multiplier_class: 0,
            reason_codes: ReasonSet::default(),
        }
    }
}

#[derive(Debug)]
pub struct Hpa {
    store_path: PathBuf,
    state: HpaState,
}

impl Default for Hpa {
    fn default() -> Self {
        #[cfg(test)]
        {
            let path = std::env::temp_dir().join(format!("hpa_state_{}.json", std::process::id()));
            let _ = std::fs::remove_file(&path);
            Self::new(path)
        }

        #[cfg(not(test))]
        {
            Self::new(PathBuf::from("hpa_state.json"))
        }
    }
}

impl Hpa {
    pub fn new(store_path: impl Into<PathBuf>) -> Self {
        let store_path = store_path.into();
        let state = load_state(&store_path).unwrap_or_default();
        Self { store_path, state }
    }

    pub fn tick(&mut self, input: &HpaInput) -> HpaOutput {
        let mut reasons = ReasonSet::default();
        let mut stable_window = input.stable_medium_window;

        if matches!(input.integrity_state, IntegrityState::Fail) {
            self.bump_allostatic_load(5);
            self.state.integrity_sensitivity = self
                .state
                .integrity_sensitivity
                .saturating_add(5)
                .min(MAX_CELL_VALUE);
            self.state.recovery_confidence = 0;
            self.state.stable_window_counter = 0;
            stable_window = false;
            reasons.insert("hpa_integrity_fail");
        }

        if input.dlp_critical_present {
            self.bump_allostatic_load(2);
            self.state.exfil_sensitivity = self
                .state
                .exfil_sensitivity
                .saturating_add(3)
                .min(MAX_CELL_VALUE);
            self.state.stable_window_counter = 0;
            stable_window = false;
            reasons.insert("hpa_dlp_critical");
        }

        if input.receipt_invalid_present {
            self.bump_allostatic_load(2);
            self.state.integrity_sensitivity = self
                .state
                .integrity_sensitivity
                .saturating_add(3)
                .min(MAX_CELL_VALUE);
            self.state.stable_window_counter = 0;
            stable_window = false;
            reasons.insert("hpa_receipt_invalid");
        }

        if input.replay_mismatch_present {
            self.bump_allostatic_load(2);
            self.state.integrity_sensitivity = self
                .state
                .integrity_sensitivity
                .saturating_add(3)
                .min(MAX_CELL_VALUE);
            self.state.stable_window_counter = 0;
            stable_window = false;
            reasons.insert("hpa_replay_mismatch");
        }

        if input.deny_storm_present {
            self.bump_allostatic_load(1);
            self.state.probing_sensitivity = self
                .state
                .probing_sensitivity
                .saturating_add(2)
                .min(MAX_CELL_VALUE);
            self.state.stable_window_counter = 0;
            stable_window = false;
            reasons.insert("hpa_deny_storm");
        }

        if input.timeouts_burst_present {
            self.bump_allostatic_load(1);
            self.state.reliability_sensitivity = self
                .state
                .reliability_sensitivity
                .saturating_add(1)
                .min(MAX_CELL_VALUE);
            self.state.stable_window_counter = 0;
            stable_window = false;
            reasons.insert("hpa_timeout_burst");
        }

        if input.unlock_present && stable_window {
            self.state.recovery_confidence = self
                .state
                .recovery_confidence
                .saturating_add(3)
                .min(MAX_CELL_VALUE);
            self.state.allostatic_load = self.state.allostatic_load.saturating_sub(1);
            reasons.insert("hpa_recovery_unlock");
        }

        if stable_window {
            self.state.stable_window_counter = self.state.stable_window_counter.saturating_add(1);
            if self.state.stable_window_counter >= STABLE_DECAY_INTERVAL {
                self.state.allostatic_load = self.state.allostatic_load.saturating_sub(1);
                self.state.stable_window_counter = 0;
                reasons.insert("hpa_stable_decay");
            }
        } else {
            self.state.stable_window_counter = 0;
        }

        let output = self.derive_output(reasons.clone());
        let _ = save_state(&self.store_path, &self.state);
        output
    }

    pub fn current_output(&self) -> HpaOutput {
        self.derive_output(ReasonSet::default())
    }

    fn bump_allostatic_load(&mut self, amount: u16) {
        self.state.allostatic_load = self
            .state
            .allostatic_load
            .saturating_add(amount)
            .min(MAX_CELL_VALUE);
    }

    fn derive_output(&self, mut reasons: ReasonSet) -> HpaOutput {
        let allostatic_load_class = match self.state.allostatic_load {
            v if v >= 70 => LevelClass::High,
            v if v >= 30 => LevelClass::Med,
            _ => LevelClass::Low,
        };

        let caution_offset = clamp_offset(
            div_ceil(self.state.allostatic_load, 10)
                + div_ceil(self.state.integrity_sensitivity, 20)
                + div_ceil(self.state.exfil_sensitivity, 20),
        );
        let novelty_offset = clamp_offset(
            div_ceil(self.state.allostatic_load, 10) + div_ceil(self.state.probing_sensitivity, 20),
        );
        let approval_offset = clamp_offset(
            div_ceil(self.state.allostatic_load, 10)
                + div_ceil(self.state.integrity_sensitivity, 20)
                + div_ceil(self.state.probing_sensitivity, 20),
        );
        let export_offset = clamp_offset(
            div_ceil(self.state.exfil_sensitivity, 10) + div_ceil(self.state.allostatic_load, 20),
        );
        let chain_offset = clamp_offset(
            div_ceil(self.state.reliability_sensitivity, 10)
                + div_ceil(self.state.probing_sensitivity, 20)
                + div_ceil(self.state.allostatic_load, 20),
        );

        let cooldown_multiplier_class = if self.state.allostatic_load >= 80 {
            3
        } else if self.state.allostatic_load >= 50 {
            2
        } else if self.state.allostatic_load >= 20 {
            1
        } else {
            0
        };

        reasons.codes.sort();
        reasons.codes.dedup();

        HpaOutput {
            allostatic_load_class,
            baseline_caution_offset: caution_offset,
            baseline_novelty_dampening_offset: novelty_offset,
            baseline_approval_strictness_offset: approval_offset,
            baseline_export_strictness_offset: export_offset,
            baseline_chain_conservatism_offset: chain_offset,
            baseline_cooldown_multiplier_class: cooldown_multiplier_class,
            reason_codes: reasons,
        }
    }
}

impl DbmComponent for Hpa {}

fn div_ceil(value: u16, divisor: u16) -> u16 {
    if value == 0 {
        0
    } else {
        value.div_ceil(divisor)
    }
}

fn clamp_offset(value: u16) -> u16 {
    value.min(OFFSET_MAX)
}

fn load_state(path: &Path) -> Option<HpaState> {
    let mut file = File::open(path).ok()?;
    let mut buf = String::new();
    file.read_to_string(&mut buf).ok()?;
    serde_json::from_str(&buf).ok()
}

fn save_state(path: &Path, state: &HpaState) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let mut file = File::create(path)?;
    let serialized = serde_json::to_string_pretty(state)?;
    file.write_all(serialized.as_bytes())?;
    file.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(name)
    }

    fn clean_path(path: &Path) {
        let _ = fs::remove_file(path);
    }

    #[test]
    fn integrity_fail_increases_sensitivity() {
        let path = temp_path("hpa_integrity.json");
        clean_path(&path);
        let mut hpa = Hpa::new(&path);
        let output = hpa.tick(&HpaInput {
            integrity_state: IntegrityState::Fail,
            ..Default::default()
        });

        assert!(output.baseline_caution_offset >= 1);
        assert!(hpa.state.integrity_sensitivity >= 5);
        clean_path(&path);
    }

    #[test]
    fn stable_windows_reduce_allostatic_load() {
        let path = temp_path("hpa_stable.json");
        clean_path(&path);
        let mut hpa = Hpa::new(&path);
        hpa.state.allostatic_load = 10;
        hpa.state.stable_window_counter = 4;

        let output = hpa.tick(&HpaInput {
            stable_medium_window: true,
            ..Default::default()
        });

        assert!(output.baseline_caution_offset <= 1);
        assert_eq!(hpa.state.stable_window_counter, 0);
        clean_path(&path);
    }

    #[test]
    fn persistence_round_trip_retains_state() {
        let path = temp_path("hpa_persist.json");
        clean_path(&path);
        let mut hpa = Hpa::new(&path);
        hpa.tick(&HpaInput {
            integrity_state: IntegrityState::Fail,
            ..Default::default()
        });

        let hpa_reloaded = Hpa::new(&path);
        let output = hpa_reloaded.current_output();
        assert!(output.baseline_caution_offset > 0);
        clean_path(&path);
    }
}
