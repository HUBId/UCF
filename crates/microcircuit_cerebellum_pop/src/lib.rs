#![forbid(unsafe_code)]

use dbm_core::{IntegrityState, LevelClass, ReasonSet, SuspendRecommendation, ToolKey};
use microcircuit_cerebellum_stub::{CerInput, CerOutput, ToolFailureCounts};
use microcircuit_core::{digest_config, digest_meta, CircuitConfig, MicrocircuitBackend};
use std::collections::HashMap;

const MAX_TOOL_STATE: usize = 32;
const MAX_TOOL_ANOMALIES: usize = 16;
const MAX_SUSPEND_RECOMMENDATIONS: usize = 8;

#[derive(Debug, Clone, Default)]
struct ToolPopState {
    pop: i32,
    latch: u8,
}

#[derive(Debug, Clone, Default)]
struct CerPopState {
    pop_divergence: i32,
    pop_reliability: i32,
    step_count: u64,
    tools: HashMap<ToolKey, ToolPopState>,
}

#[derive(Debug, Clone)]
pub struct CerebellumPopMicrocircuit {
    config: CircuitConfig,
    state: CerPopState,
}

impl CerebellumPopMicrocircuit {
    pub fn new(config: CircuitConfig) -> Self {
        Self {
            config,
            state: CerPopState::default(),
        }
    }

    fn clamp_drive(value: i32) -> i32 {
        value.clamp(0, 100)
    }

    fn global_drive(input: &CerInput) -> i32 {
        let mut drive = 0;

        if input.integrity != IntegrityState::Ok {
            drive += 40;
        }
        if input.receipt_invalid_present {
            drive += 30;
        }
        if input.tool_unavailable_count_medium >= 3 {
            drive += 30;
        }
        if input.partial_failure_count_medium >= 5 {
            drive += 25;
        }
        if input.timeout_count_medium >= 10 {
            drive += 25;
        }
        if input.timeout_count_medium >= 2 {
            drive += 10;
        }

        Self::clamp_drive(drive)
    }

    fn tool_drive(counts: &ToolFailureCounts) -> i32 {
        let mut drive = 0;

        if counts.unavailable >= 1 {
            drive += 40;
        }
        if counts.partial_failures >= 3 {
            drive += 30;
        }
        if counts.timeouts >= 5 {
            drive += 30;
        }
        if counts.partial_failures >= 1 {
            drive += 15;
        }
        if counts.timeouts >= 2 {
            drive += 10;
        }

        Self::clamp_drive(drive)
    }

    fn update_global_pops(&mut self, drive: i32) {
        let mut divergence = self.state.pop_divergence;
        divergence += (drive - 20) / 4;
        divergence = divergence.clamp(0, 100);
        if drive < 20 {
            divergence = (divergence - 3).max(0);
        }
        self.state.pop_divergence = divergence;

        let mut reliability = self.state.pop_reliability;
        reliability += (drive - 30) / 5;
        reliability = reliability.clamp(0, 100);
        if drive < 20 {
            reliability = (reliability - 3).max(0);
        }
        self.state.pop_reliability = reliability;
    }

    fn get_or_insert_tool(&mut self, tool: &ToolKey) -> Option<&mut ToolPopState> {
        if self.state.tools.contains_key(tool) {
            return self.state.tools.get_mut(tool);
        }

        if self.state.tools.len() < MAX_TOOL_STATE {
            self.state
                .tools
                .insert(tool.clone(), ToolPopState::default());
            return self.state.tools.get_mut(tool);
        }

        let mut largest = tool.clone();
        for key in self.state.tools.keys() {
            if key > &largest {
                largest = key.clone();
            }
        }

        if &largest == tool {
            return None;
        }

        self.state.tools.remove(&largest);
        self.state
            .tools
            .insert(tool.clone(), ToolPopState::default());
        self.state.tools.get_mut(tool)
    }

    fn update_tool_state(&mut self, tool: &ToolKey, drive: i32) {
        let Some(state) = self.get_or_insert_tool(tool) else {
            return;
        };

        let mut pop = state.pop;
        pop += (drive - 15) / 3;
        pop = pop.clamp(0, 100);
        if drive < 15 {
            pop = (pop - 2).max(0);
        }
        state.pop = pop;

        if drive >= 40 {
            state.latch = (state.latch + 2).min(10);
        } else {
            state.latch = state.latch.saturating_sub(1);
        }
    }

    fn severity_rank(level: LevelClass) -> u8 {
        match level {
            LevelClass::Low => 0,
            LevelClass::Med => 1,
            LevelClass::High => 2,
        }
    }

    fn anomaly_severity(state: &ToolPopState) -> LevelClass {
        if state.pop >= 70 || state.latch >= 6 {
            LevelClass::High
        } else {
            LevelClass::Med
        }
    }

    fn recommendation_severity(state: &ToolPopState) -> LevelClass {
        if state.pop >= 70 || state.latch >= 6 {
            LevelClass::High
        } else {
            LevelClass::Med
        }
    }

    fn normalized_failures(input: &CerInput) -> Vec<(ToolKey, ToolFailureCounts)> {
        let mut failures: Vec<(ToolKey, ToolFailureCounts)> = input
            .tool_failures
            .iter()
            .map(|(key, counts)| (key.clone().normalized(), counts.clone()))
            .collect();
        failures.sort_by(|a, b| a.0.cmp(&b.0));
        failures
    }
}

impl MicrocircuitBackend<CerInput, CerOutput> for CerebellumPopMicrocircuit {
    fn step(&mut self, input: &CerInput, _now_ms: u64) -> CerOutput {
        let drive = Self::global_drive(input);
        self.update_global_pops(drive);

        let failures = Self::normalized_failures(input);
        let mut tool_drives: HashMap<ToolKey, i32> = HashMap::new();
        for (tool, counts) in &failures {
            let tool_drive = Self::tool_drive(counts);
            tool_drives.insert(tool.clone(), tool_drive);
            self.update_tool_state(tool, tool_drive);
        }

        self.state.step_count = self.state.step_count.saturating_add(1);

        let divergence = if self.state.pop_divergence >= 70
            || input.tool_unavailable_count_medium >= 3
            || input.integrity != IntegrityState::Ok
            || input.receipt_invalid_present
        {
            LevelClass::High
        } else if self.state.pop_divergence >= 40 {
            LevelClass::Med
        } else {
            LevelClass::Low
        };

        let mut tool_anomalies: Vec<(ToolKey, LevelClass)> = self
            .state
            .tools
            .iter()
            .filter_map(|(tool, state)| {
                if state.pop >= 50 || state.latch > 0 {
                    let drive = tool_drives.get(tool).copied().unwrap_or(0);
                    let severity = if drive >= 40 {
                        LevelClass::High
                    } else {
                        Self::anomaly_severity(state)
                    };
                    Some((tool.clone(), severity))
                } else {
                    None
                }
            })
            .collect();
        tool_anomalies.sort_by(|a, b| {
            let sev = Self::severity_rank(b.1).cmp(&Self::severity_rank(a.1));
            if sev == std::cmp::Ordering::Equal {
                a.0.cmp(&b.0)
            } else {
                sev
            }
        });
        tool_anomalies.truncate(MAX_TOOL_ANOMALIES);

        let mut counts_map: HashMap<ToolKey, ToolFailureCounts> = HashMap::new();
        for (tool, counts) in failures {
            counts_map.insert(tool, counts);
        }

        let mut suspend_recommendations: Vec<SuspendRecommendation> = self
            .state
            .tools
            .iter()
            .filter_map(|(tool, state)| {
                if state.pop >= 70 || state.latch >= 4 {
                    let mut reason_codes = ReasonSet::default();
                    reason_codes.insert("RC.GV.TOOL.SUSPEND_RECOMMENDED");
                    if let Some(counts) = counts_map.get(tool) {
                        if counts.timeouts > 0 {
                            reason_codes.insert("RC.GE.EXEC.TIMEOUT");
                        }
                        if counts.partial_failures > 0 {
                            reason_codes.insert("RC.GE.EXEC.PARTIAL_FAILURE");
                        }
                    }
                    Some(SuspendRecommendation {
                        tool: tool.clone(),
                        severity: Self::recommendation_severity(state),
                        reason_codes,
                    })
                } else {
                    None
                }
            })
            .collect();
        suspend_recommendations.sort_by(|a, b| {
            let sev = Self::severity_rank(b.severity).cmp(&Self::severity_rank(a.severity));
            if sev == std::cmp::Ordering::Equal {
                a.tool.cmp(&b.tool)
            } else {
                sev
            }
        });
        suspend_recommendations.truncate(MAX_SUSPEND_RECOMMENDATIONS);

        let mut reason_codes = ReasonSet::default();
        match divergence {
            LevelClass::High => {
                reason_codes.insert("RC.GV.DIVERGENCE.HIGH");
            }
            LevelClass::Med => {
                reason_codes.insert("RC.GV.DIVERGENCE.MED");
            }
            LevelClass::Low => {}
        }

        let suspend_recommended = !suspend_recommendations.is_empty();
        if suspend_recommended {
            reason_codes.insert("RC.GV.TOOL.SUSPEND_RECOMMENDED");
        }

        let side_effect_suspected = tool_anomalies
            .iter()
            .any(|(_, level)| level == &LevelClass::High);

        CerOutput {
            divergence,
            side_effect_suspected,
            suspend_recommended,
            tool_anomalies,
            suspend_recommendations,
            reason_codes,
        }
    }

    fn snapshot_digest(&self) -> [u8; 32] {
        let mut bytes = Vec::new();
        bytes.extend(self.state.pop_divergence.to_le_bytes());
        bytes.extend(self.state.pop_reliability.to_le_bytes());
        bytes.extend(self.state.step_count.to_le_bytes());

        digest_meta("UCF:MC:CER", &bytes)
    }

    fn config_digest(&self) -> [u8; 32] {
        digest_config("UCF:MC:CER:CFG", &self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dbm_core::DbmModule;
    use microcircuit_cerebellum_stub::CerebellumRules;

    fn base_input() -> CerInput {
        CerInput {
            integrity: IntegrityState::Ok,
            ..Default::default()
        }
    }

    fn tool_failure(
        tool: &str,
        timeouts: u32,
        partial_failures: u32,
        unavailable: u32,
    ) -> (ToolKey, ToolFailureCounts) {
        (
            ToolKey::new(tool.to_string(), "act".to_string()),
            ToolFailureCounts {
                timeouts,
                partial_failures,
                unavailable,
            },
        )
    }

    fn level_rank(level: LevelClass) -> u8 {
        match level {
            LevelClass::Low => 0,
            LevelClass::Med => 1,
            LevelClass::High => 2,
        }
    }

    #[test]
    fn deterministic_sequence_is_repeatable() {
        let mut circuit_a = CerebellumPopMicrocircuit::new(CircuitConfig::default());
        let mut circuit_b = CerebellumPopMicrocircuit::new(CircuitConfig::default());
        let sequence = vec![
            CerInput {
                timeout_count_medium: 10,
                ..base_input()
            },
            CerInput {
                partial_failure_count_medium: 5,
                ..base_input()
            },
            base_input(),
            CerInput {
                tool_unavailable_count_medium: 3,
                ..base_input()
            },
        ];

        for input in sequence {
            let out_a = circuit_a.step(&input, 0);
            let out_b = circuit_b.step(&input, 0);
            assert_eq!(out_a, out_b);
        }
    }

    #[test]
    fn unavailable_triggers_divergence_high() {
        let mut circuit = CerebellumPopMicrocircuit::new(CircuitConfig::default());
        let output = circuit.step(
            &CerInput {
                tool_unavailable_count_medium: 3,
                ..base_input()
            },
            0,
        );

        assert_eq!(output.divergence, LevelClass::High);
    }

    #[test]
    fn per_tool_failures_create_anomaly_and_recommendation() {
        let mut circuit = CerebellumPopMicrocircuit::new(CircuitConfig::default());
        let mut input = base_input();
        input.tool_failures = vec![tool_failure("tool-a", 5, 0, 0)];

        let _ = circuit.step(&input, 0);
        let output = circuit.step(&input, 0);

        assert!(output
            .tool_anomalies
            .iter()
            .any(|(tool, level)| tool.tool_id == "tool-a" && *level == LevelClass::High));
        assert!(output.suspend_recommended);
    }

    #[test]
    fn bounded_tool_state_eviction_is_deterministic() {
        let mut circuit = CerebellumPopMicrocircuit::new(CircuitConfig::default());
        let mut input = base_input();
        input.tool_failures = (0..40)
            .map(|idx| tool_failure(&format!("tool-{idx:02}"), 5, 0, 0))
            .collect();

        for _ in 0..2 {
            let _ = circuit.step(&input, 0);
        }

        assert!(circuit.state.tools.len() <= MAX_TOOL_STATE);
        assert!(!circuit
            .state
            .tools
            .keys()
            .any(|key| key.tool_id == "tool-39"));
    }

    #[test]
    fn invariants_hold_vs_rules_backend() {
        let mut micro = CerebellumPopMicrocircuit::new(CircuitConfig::default());
        let mut rules = CerebellumRules::new();
        let input = CerInput {
            tool_unavailable_count_medium: 3,
            receipt_invalid_present: true,
            integrity: IntegrityState::Fail,
            tool_failures: vec![tool_failure("tool-a", 5, 3, 1)],
            ..base_input()
        };

        let micro_out = micro.step(&input, 0);
        let rules_out = rules.tick(&input);

        assert!(matches!(micro_out.divergence, LevelClass::High));
        assert!(level_rank(micro_out.divergence) >= level_rank(rules_out.divergence));

        let micro_tool = micro_out
            .tool_anomalies
            .iter()
            .find(|(tool, _)| tool.tool_id == "tool-a")
            .map(|(_, level)| *level)
            .unwrap_or(LevelClass::Low);
        let rules_tool = rules_out
            .tool_anomalies
            .iter()
            .find(|(tool, _)| tool.tool_id == "tool-a")
            .map(|(_, level)| *level)
            .unwrap_or(LevelClass::Low);
        assert!(level_rank(micro_tool) >= level_rank(rules_tool));
    }
}
