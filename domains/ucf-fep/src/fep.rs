#[derive(Clone, Debug, PartialEq)]
pub struct FepCfg {
    pub beta_surprise: f32,
    pub w_ebm_q: u16,
    pub beta_complexity: f32,
    pub beta_policy_risk: f32,
    pub beta_memory_pressure: f32,
    pub beta_coherence_lock: f32,
    pub lambda_nsr: f32,
    pub attention_min: f32,
    pub attention_max: f32,
    pub learn_gate_min: f32,
    pub learn_gate_max: f32,
    pub structure_delta_cap: f32,
}

impl FepCfg {
    pub fn default_v0() -> Self {
        Self {
            beta_surprise: 1.2,
            w_ebm_q: 1_966,
            beta_complexity: 0.8,
            beta_policy_risk: 1.4,
            beta_memory_pressure: 1.0,
            beta_coherence_lock: 1.1,
            lambda_nsr: 1.0,
            attention_min: 0.1,
            attention_max: 0.95,
            learn_gate_min: 0.05,
            learn_gate_max: 0.9,
            structure_delta_cap: 0.3,
        }
    }
}

use ucf_types::{RiskConfidenceV1, SignalBundleV1};

#[derive(Clone, Debug, PartialEq)]
pub struct FepInputs {
    pub now_ms: u64,
    pub ebm_energy_mean_topk_q: u16,
    pub dt_s: f32,
    pub signal_bundle: SignalBundleV1,
    pub risk_confidence: RiskConfidenceV1,
    pub complexity: f32,
    pub policy_risk: f32,
    pub onn_lock: f32,
    pub snn_event_rate: f32,
    pub ess_pressure: f32,
    pub ssm_pressure: f32,
    pub geist_drift: f32,
    pub hormone_risk_penalty_scale: f32,
    pub hormone_exploration_bias_delta: f32,
    pub hormone_attention_gain: f32,
    pub hormone_action_threshold_delta: f32,
    pub nsr_risk: Option<f32>,
    pub nsr_confidence: Option<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FepOutputs {
    pub attention_gain: f32,
    pub free_energy_proxy_q: u16,
    pub ebm_energy_mean_topk_q: u16,
    pub w_ebm_q: u16,
    pub coupling_version: u8,
    pub learn_gate: f32,
    pub memory_priority: f32,
    pub action_inhibit: f32,
    pub structural_delta: f32,
    pub confidence: f32,
}

pub fn fep_step(cfg: &FepCfg, inp: &FepInputs) -> FepOutputs {
    let surprise = inp.signal_bundle.surprise_q.to_f32().clamp(0.0, 1.0);
    let complexity = inp.complexity.clamp(0.0, 1.0);
    let policy_risk = inp.policy_risk.clamp(0.0, 1.0);
    let compute_confidence = inp.risk_confidence.confidence_q.to_f32().clamp(0.0, 1.0);
    let nsr_risk = inp.nsr_risk.unwrap_or(policy_risk).clamp(0.0, 1.0);
    let nsr_confidence = inp
        .nsr_confidence
        .unwrap_or(compute_confidence)
        .clamp(0.0, 1.0);
    let compute_risk = inp.risk_confidence.risk_q.to_f32().clamp(0.0, 1.0);
    let onn_lock = inp.onn_lock.clamp(0.0, 1.0);
    let snn_event_rate = inp.snn_event_rate.clamp(0.0, 1.0);
    let ess_pressure = inp.signal_bundle.pressure_q.to_f32().clamp(0.0, 1.0);
    let geist_drift = inp.geist_drift.clamp(0.0, 1.0);
    let hormone_risk_penalty_scale = inp.hormone_risk_penalty_scale.clamp(0.0, 4.0);
    let hormone_exploration_bias_delta = inp.hormone_exploration_bias_delta.clamp(-0.5, 0.5);
    let hormone_attention_gain = inp.hormone_attention_gain.clamp(0.0, 3.0);
    let hormone_action_threshold_delta = inp.hormone_action_threshold_delta.clamp(-0.5, 0.5);

    let evidence = (cfg.beta_coherence_lock * onn_lock + (1.0 - surprise)).clamp(0.0, 2.0);
    let base_risk_penalty = 0.7 * compute_risk + 0.3 * (1.0 - compute_confidence);
    let risk_penalty = (base_risk_penalty * hormone_risk_penalty_scale).clamp(0.0, 1.5);
    let free_energy = (cfg.beta_surprise * surprise
        + cfg.beta_complexity * complexity
        + cfg.beta_policy_risk * policy_risk
        + cfg.lambda_nsr * nsr_risk
        + cfg.beta_policy_risk * risk_penalty)
        .max(0.0);
    let base_free_energy_q = (free_energy.clamp(0.0, 1.0) * 65_535.0).round() as u16;
    let ebm_energy_mean_topk_q = inp.ebm_energy_mean_topk_q;
    let proxy_add = ((u32::from(cfg.w_ebm_q) * u32::from(ebm_energy_mean_topk_q)) >> 16) as u16;
    let free_energy_proxy_q = base_free_energy_q.saturating_add(proxy_add);

    let pressure_relief = (onn_lock * 0.7).clamp(0.0, 1.0);
    let pressure_penalty =
        (ess_pressure - pressure_relief).max(0.0) * cfg.beta_memory_pressure * 0.35;
    let attention_base = 0.45 + 0.55 * surprise
        - 0.65 * policy_risk
        - 0.2 * nsr_risk
        - 0.4 * risk_penalty
        - pressure_penalty
        + 0.1 * hormone_exploration_bias_delta;
    let attention_gain = (attention_base * hormone_attention_gain).clamp(
        cfg.attention_min.min(cfg.attention_max),
        cfg.attention_max.max(cfg.attention_min),
    );

    let stable_risk_window = (1.0 - policy_risk * 0.9).clamp(0.0, 1.0);
    let surprise_lock = surprise * onn_lock;
    let learn_raw = cfg.learn_gate_min
        + (cfg.learn_gate_max - cfg.learn_gate_min)
            * (0.6 * surprise_lock * stable_risk_window + 0.25 * (1.0 - complexity)
                - 0.5 * geist_drift
                - 0.4 * ess_pressure);
    let learn_gate = learn_raw.clamp(
        cfg.learn_gate_min.min(cfg.learn_gate_max),
        cfg.learn_gate_max.max(cfg.learn_gate_min),
    );

    let memory_priority = (0.45 * surprise + 0.35 * ess_pressure + 0.25 * snn_event_rate
        - 0.35 * complexity)
        .clamp(0.0, 1.0);

    let action_inhibit = (0.45 * policy_risk + 0.35 * risk_penalty + 0.45 * geist_drift
        - 0.3 * onn_lock * (1.0 - policy_risk)
        + hormone_action_threshold_delta)
        .clamp(0.0, 1.0);

    let structural_delta = ((surprise - complexity) * (1.0 - policy_risk)).clamp(
        -cfg.structure_delta_cap.abs(),
        cfg.structure_delta_cap.abs(),
    );

    let confidence =
        (0.3 * onn_lock + 0.35 * (evidence / 2.0) + 0.15 * (1.0 / (1.0 + free_energy))
            - 0.25 * policy_risk
            - 0.2 * (1.0 - nsr_confidence)
            - 0.25 * geist_drift)
            .clamp(0.0, 1.0);

    FepOutputs {
        attention_gain,
        free_energy_proxy_q,
        ebm_energy_mean_topk_q,
        w_ebm_q: cfg.w_ebm_q,
        coupling_version: 1,
        learn_gate,
        memory_priority,
        action_inhibit,
        structural_delta,
        confidence,
    }
}

#[cfg(test)]
mod tests {
    use super::{fep_step, FepCfg, FepInputs};
    use ucf_types::{RiskConfidenceV1, SignalBundleV1, UQ0_16};

    fn mk_input() -> FepInputs {
        FepInputs {
            now_ms: 1,
            ebm_energy_mean_topk_q: 0,
            dt_s: 0.01,
            signal_bundle: SignalBundleV1 {
                t: 1,
                policy_graph_digest: [0; 32],
                risk_q: ucf_types::UQ0_16::from_f32_clamped(0.2),
                confidence_q: ucf_types::UQ0_16::from_f32_clamped(0.8),
                surprise_q: ucf_types::UQ0_16::from_f32_clamped(0.3),
                pressure_q: ucf_types::UQ0_16::from_f32_clamped(0.3),
                uncertainty_q: ucf_types::UQ0_16::from_f32_clamped(0.2),
                stability_q: ucf_types::UQ0_16::from_f32_clamped(0.7),
                coherence_q: None,
                world_prediction_digest: [0; 32],
                sae_spikes_digest: [0; 32],
                ssm_state_digest: [0; 32],
                lfm_state_digest: [0; 32],
            },
            risk_confidence: RiskConfidenceV1 {
                risk_q: ucf_types::UQ0_16::from_f32_clamped(0.2),
                confidence_q: ucf_types::UQ0_16::from_f32_clamped(0.8),
                update_digest: [1; 32],
            },
            complexity: 0.2,
            policy_risk: 0.2,
            onn_lock: 0.5,
            snn_event_rate: 0.4,
            ess_pressure: 0.3,
            ssm_pressure: 0.2,
            geist_drift: 0.1,
            hormone_risk_penalty_scale: 1.0,
            hormone_exploration_bias_delta: 0.0,
            hormone_attention_gain: 1.0,
            hormone_action_threshold_delta: 0.0,
            nsr_risk: None,
            nsr_confidence: None,
        }
    }

    #[test]
    fn fep_step_clamps_outputs() {
        let cfg = FepCfg::default_v0();
        let mut inp = mk_input();
        inp.signal_bundle.surprise_q = UQ0_16::from_f32_clamped(10.0);
        inp.complexity = -2.0;
        inp.policy_risk = 5.0;
        inp.risk_confidence.risk_q = UQ0_16::from_f32_clamped(5.0);
        inp.risk_confidence.confidence_q = UQ0_16::from_f32_clamped(-3.0);
        inp.onn_lock = 4.0;
        inp.snn_event_rate = 2.0;
        inp.signal_bundle.pressure_q = UQ0_16::from_f32_clamped(2.0);
        inp.geist_drift = 2.0;

        let out = fep_step(&cfg, &inp);
        assert!((cfg.attention_min..=cfg.attention_max).contains(&out.attention_gain));
        assert!((cfg.learn_gate_min..=cfg.learn_gate_max).contains(&out.learn_gate));
        assert!((0.0..=1.0).contains(&out.memory_priority));
        assert!((0.0..=1.0).contains(&out.action_inhibit));
        assert!((0.0..=1.0).contains(&out.confidence));
        assert!(out.structural_delta.abs() <= cfg.structure_delta_cap);
    }

    #[test]
    fn high_policy_risk_increases_inhibit_and_lowers_confidence() {
        let cfg = FepCfg::default_v0();
        let low = mk_input();
        let mut high = mk_input();
        high.policy_risk = 0.95;

        let low_out = fep_step(&cfg, &low);
        let high_out = fep_step(&cfg, &high);
        assert!(high_out.action_inhibit >= low_out.action_inhibit);
        assert!(high_out.confidence <= low_out.confidence);
    }

    #[test]
    fn high_ess_pressure_reduces_learning_and_boosts_memory_priority() {
        let cfg = FepCfg::default_v0();
        let low = mk_input();
        let mut high = mk_input();
        high.signal_bundle.pressure_q = UQ0_16::from_f32_clamped(0.95);

        let low_out = fep_step(&cfg, &low);
        let high_out = fep_step(&cfg, &high);
        assert!(high_out.learn_gate <= low_out.learn_gate);
        assert!(high_out.memory_priority >= low_out.memory_priority);
    }

    #[test]
    fn high_onn_lock_increases_confidence() {
        let cfg = FepCfg::default_v0();
        let low = mk_input();
        let mut high = mk_input();
        high.onn_lock = 0.95;

        let low_out = fep_step(&cfg, &low);
        let high_out = fep_step(&cfg, &high);
        assert!(high_out.confidence >= low_out.confidence);
    }

    #[test]
    fn high_compute_risk_increases_inhibit() {
        let cfg = FepCfg::default_v0();
        let low = mk_input();
        let mut high = mk_input();
        high.risk_confidence.risk_q = UQ0_16::from_f32_clamped(0.95);
        high.risk_confidence.confidence_q = UQ0_16::from_f32_clamped(0.1);
        let low_out = fep_step(&cfg, &low);
        let high_out = fep_step(&cfg, &high);
        assert!(high_out.action_inhibit >= low_out.action_inhibit);
    }
}
