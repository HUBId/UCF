#![forbid(unsafe_code)]

use blake3::Hasher;
use profiles::{
    apply_classification, classify_signal_frame, decide_with_fallback, ControlDecision,
    RegulationConfig,
};
use prost::Message;
use rsv::RsvState;
use std::path::PathBuf;
use ucf::v1::{ActiveProfile, ControlFrame, Overlays, ToolClassMask};

const CONTROL_FRAME_DOMAIN: &str = "UCF:HASH:CONTROL_FRAME";

pub struct RegulationEngine {
    pub rsv: RsvState,
    config: RegulationConfig,
}

impl Default for RegulationEngine {
    fn default() -> Self {
        let config_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("config");
        match RegulationConfig::load_from_dir(config_dir) {
            Ok(config) => RegulationEngine {
                rsv: RsvState::default(),
                config,
            },
            Err(_) => RegulationEngine {
                rsv: RsvState::default(),
                config: RegulationConfig::fallback(),
            },
        }
    }
}

impl RegulationEngine {
    pub fn new(config: RegulationConfig) -> Self {
        RegulationEngine {
            rsv: RsvState::default(),
            config,
        }
    }

    pub fn on_signal_frame(
        &mut self,
        mut frame: ucf::v1::SignalFrame,
        now_ms: u64,
    ) -> ControlFrame {
        if frame.timestamp_ms.is_none() {
            frame.timestamp_ms = Some(now_ms);
        }

        let timestamp_ms = frame.timestamp_ms.unwrap_or(now_ms);
        let classified = classify_signal_frame(&frame, &self.config.thresholds);
        apply_classification(&mut self.rsv, &classified, timestamp_ms);

        let decision = decide_with_fallback(&self.rsv, now_ms, &self.config);
        self.render_control_frame(decision)
    }

    pub fn on_tick(&mut self, now_ms: u64) -> ControlFrame {
        self.rsv.missing_frame_counter = self.rsv.missing_frame_counter.saturating_add(1);

        let decision = decide_with_fallback(&self.rsv, now_ms, &self.config);
        self.render_control_frame(decision)
    }

    fn render_control_frame(&self, decision: ControlDecision) -> ControlFrame {
        let overlays = Overlays {
            simulate_first: decision.overlays.simulate_first,
            export_lock: decision.overlays.export_lock,
            novelty_lock: decision.overlays.novelty_lock,
        };

        let profile = ActiveProfile {
            profile: decision.profile.as_str().to_string(),
        };

        let toolclass_mask_config = self.toolclass_mask_for(&decision);

        let mut profile_reason_codes: Vec<i32> = decision
            .profile_reason_codes
            .iter()
            .map(|code| *code as i32)
            .collect();
        profile_reason_codes.sort();

        let mut control_frame = ControlFrame {
            active_profile: Some(profile),
            overlays: Some(overlays),
            toolclass_mask: Some(toolclass_mask_config),
            profile_reason_codes,
            control_frame_digest: None,
        };

        let mut buf = Vec::new();
        control_frame.encode(&mut buf).unwrap();

        let mut hasher = Hasher::new_derive_key(CONTROL_FRAME_DOMAIN);
        hasher.update(&buf);
        let digest = hasher.finalize();

        control_frame.control_frame_digest = Some(digest.as_bytes().to_vec());

        control_frame
    }

    fn toolclass_mask_for(&self, decision: &ControlDecision) -> ToolClassMask {
        let mut mask_cfg = self
            .config
            .profiles
            .get(decision.profile)
            .toolclass_mask
            .clone();

        if decision.overlays.simulate_first {
            if let Some(overlay_mask) = &self.config.overlays.simulate_first.toolclass_mask {
                mask_cfg = mask_cfg.merge_overlay(overlay_mask);
            }
        }

        if decision.overlays.export_lock {
            if let Some(overlay_mask) = &self.config.overlays.export_lock.toolclass_mask {
                mask_cfg = mask_cfg.merge_overlay(overlay_mask);
            }
        }

        if decision.overlays.novelty_lock {
            if let Some(overlay_mask) = &self.config.overlays.novelty_lock.toolclass_mask {
                mask_cfg = mask_cfg.merge_overlay(overlay_mask);
            }
        }

        mask_cfg.to_tool_class_mask()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use profiles::{OverlaySet, ProfileState, RegulationConfig};
    use ucf::v1::{
        ExecStats, IntegrityStateClass, PolicyStats, ReasonCode, ReceiptStats, SignalFrame,
        WindowKind,
    };

    fn base_frame() -> SignalFrame {
        SignalFrame {
            window_kind: WindowKind::Short as i32,
            window_index: Some(1),
            timestamp_ms: Some(1),
            policy_stats: Some(PolicyStats {
                deny_count: 0,
                allow_count: 10,
            }),
            exec_stats: Some(ExecStats { timeout_count: 0 }),
            integrity_state: IntegrityStateClass::Ok as i32,
            top_reason_codes: Vec::new(),
            signal_frame_digest: None,
            receipt_stats: Some(ReceiptStats {
                receipt_missing_count: 0,
                receipt_invalid_count: 0,
            }),
        }
    }

    #[test]
    fn digest_is_deterministic() {
        let mut engine = RegulationEngine::default();
        let frame = base_frame();
        let control_a = engine.on_signal_frame(frame.clone(), 1);
        let control_b = engine.on_signal_frame(frame, 1);
        assert_eq!(
            control_a.control_frame_digest,
            control_b.control_frame_digest
        );
    }

    #[test]
    fn config_driven_profile_switches() {
        let mut engine = RegulationEngine::default();
        let mut frame = base_frame();
        frame.receipt_stats = Some(ReceiptStats {
            receipt_missing_count: 0,
            receipt_invalid_count: 2,
        });

        let control = engine.on_signal_frame(frame, 1);
        assert_eq!(control.active_profile.unwrap().profile, "M1_RESTRICTED");
        let overlays = control.overlays.unwrap();
        assert!(overlays.simulate_first && overlays.export_lock && overlays.novelty_lock);
        assert!(control
            .profile_reason_codes
            .contains(&(ReasonCode::RcGeExecDispatchBlocked as i32)));
    }

    #[test]
    fn integrity_fail_triggers_forensic_profile() {
        let mut engine = RegulationEngine::default();
        let mut frame = base_frame();
        frame.integrity_state = IntegrityStateClass::Fail as i32;
        let control = engine.on_signal_frame(frame, 1);

        let mask = control.toolclass_mask.unwrap();
        assert!(!mask.export && !mask.write && !mask.execute);
        assert_eq!(control.active_profile.unwrap().profile, "M3_FORENSIC");
    }

    #[test]
    fn missing_frame_triggers_restriction() {
        let mut engine = RegulationEngine::default();
        let frame = base_frame();
        let _ = engine.on_signal_frame(frame, 0);
        let control = engine.on_tick(60_000);
        assert_eq!(control.active_profile.unwrap().profile, "M1_RESTRICTED");
        assert!(control.overlays.unwrap().export_lock);
        assert!(control
            .profile_reason_codes
            .contains(&(ReasonCode::ReIntegrityDegraded as i32)));
    }

    #[test]
    fn overlay_masks_are_applied() {
        let engine = RegulationEngine::default();
        let decision = ControlDecision {
            profile: ProfileState::M0Research,
            overlays: OverlaySet {
                simulate_first: true,
                export_lock: false,
                novelty_lock: false,
            },
            deescalation_lock: false,
            missing_frame_override: false,
            profile_reason_codes: vec![ReasonCode::ReIntegrityDegraded],
        };

        let mask = engine.toolclass_mask_for(&decision);
        assert!(!mask.export);
        assert!(mask.read);
    }

    #[test]
    fn fallback_config_is_conservative() {
        let mut engine = RegulationEngine::new(RegulationConfig::fallback());
        let control = engine.on_signal_frame(base_frame(), 1);
        let mask = control.toolclass_mask.unwrap();
        assert_eq!(control.active_profile.unwrap().profile, "M1_RESTRICTED");
        assert!(!mask.export && !mask.write && !mask.execute);
    }

    #[test]
    fn reason_codes_are_sorted_before_digest() {
        let engine = RegulationEngine::default();
        let decision = ControlDecision {
            profile: ProfileState::M1Restricted,
            overlays: OverlaySet {
                simulate_first: true,
                export_lock: true,
                novelty_lock: true,
            },
            deescalation_lock: true,
            missing_frame_override: false,
            profile_reason_codes: vec![
                ReasonCode::RcThIntegrityCompromise,
                ReasonCode::RcGeExecDispatchBlocked,
            ],
        };

        let control_frame = engine.render_control_frame(decision);
        assert_eq!(
            control_frame.profile_reason_codes,
            vec![
                ReasonCode::RcGeExecDispatchBlocked as i32,
                ReasonCode::RcThIntegrityCompromise as i32,
            ]
        );
    }
}
