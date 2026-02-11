use sha2::{Digest, Sha256};

use crate::v0::ode::clamp01;

const DERIVATIVE_CAP: f32 = 0.2;
const ACTION_DELTA_CAP: f32 = 0.5;
const EXPLORATION_DELTA_CAP: f32 = 0.5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HormoneState {
    pub t: u64,
    pub crh: f32,
    pub acth: f32,
    pub cortisol: f32,
    pub drive: f32,
    pub digest: [u8; 32],
}

impl Default for HormoneState {
    fn default() -> Self {
        let mut state = Self {
            t: 0,
            crh: 0.1,
            acth: 0.1,
            cortisol: 0.1,
            drive: 0.5,
            digest: [0; 32],
        };
        state.digest = digest_hormone_state(&state);
        state
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HormoneInput {
    pub t: u64,
    pub pressure: f32,
    pub surprise: f32,
    pub risk: f32,
    pub confidence: f32,
    pub coherence: Option<f32>,
    pub instability: Option<f32>,
    pub evidence_chain_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HormoneStateSummary {
    pub t: u64,
    pub cortisol: f32,
    pub drive: f32,
    pub stress_index: f32,
    pub digest: [u8; 32],
    pub evidence_chain_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GatingModulation {
    pub risk_penalty_scale: f32,
    pub action_threshold_delta: f32,
    pub exploration_bias_delta: f32,
    pub attention_gain: f32,
    pub digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HormoneCfg {
    pub dt: f32,
    pub k1: f32,
    pub k2: f32,
    pub k3: f32,
    pub k4: f32,
    pub k5: f32,
    pub k6: f32,
    pub k_feedback: f32,
    pub drive_recovery: f32,
    pub drive_stress_coupling: f32,
    pub modulation_risk_scale: f32,
    pub modulation_action_scale: f32,
    pub modulation_exploration_scale: f32,
    pub modulation_attention_scale: f32,
}

impl Default for HormoneCfg {
    fn default() -> Self {
        Self {
            dt: 1.0,
            k1: 0.16,
            k2: 0.12,
            k3: 0.14,
            k4: 0.10,
            k5: 0.12,
            k6: 0.08,
            k_feedback: 0.10,
            drive_recovery: 0.06,
            drive_stress_coupling: 0.10,
            modulation_risk_scale: 1.5,
            modulation_action_scale: 0.3,
            modulation_exploration_scale: 0.2,
            modulation_attention_scale: 0.5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HormoneStepOutput {
    pub state: HormoneState,
    pub summary: HormoneStateSummary,
    pub modulation: GatingModulation,
    pub degraded: bool,
}

pub fn hormone_step(
    cfg: &HormoneCfg,
    prev: HormoneState,
    input: &HormoneInput,
) -> HormoneStepOutput {
    let dt = cfg.dt.max(0.000_1);
    let stress_drive = stress_drive(input);

    let k1 = derivatives(cfg, prev, stress_drive);
    let midpoint = clamp_state(HormoneState {
        t: input.t,
        crh: prev.crh + k1.0 * dt * 0.5,
        acth: prev.acth + k1.1 * dt * 0.5,
        cortisol: prev.cortisol + k1.2 * dt * 0.5,
        drive: prev.drive + k1.3 * dt * 0.5,
        digest: [0; 32],
    });
    let k2 = derivatives(cfg, midpoint, stress_drive);

    let mut state = HormoneState {
        t: input.t,
        crh: prev.crh + k2.0 * dt,
        acth: prev.acth + k2.1 * dt,
        cortisol: prev.cortisol + k2.2 * dt,
        drive: prev.drive + k2.3 * dt,
        digest: [0; 32],
    };

    let mut degraded = false;
    if !state.crh.is_finite()
        || !state.acth.is_finite()
        || !state.cortisol.is_finite()
        || !state.drive.is_finite()
    {
        degraded = true;
        state = HormoneState::default();
        state.t = input.t;
    }

    state = clamp_state(state);
    state.digest = digest_hormone_state(&state);

    let stress_index = clamp01(0.8 * state.cortisol + 0.2 * state.acth);
    let summary = HormoneStateSummary {
        t: input.t,
        cortisol: state.cortisol,
        drive: state.drive,
        stress_index,
        digest: digest_hormone_summary(
            input.t,
            state.cortisol,
            state.drive,
            stress_index,
            input.evidence_chain_digest,
        ),
        evidence_chain_digest: input.evidence_chain_digest,
    };
    let modulation = map_modulation(cfg, &summary);

    HormoneStepOutput {
        state,
        summary,
        modulation,
        degraded,
    }
}

pub fn stress_drive(input: &HormoneInput) -> f32 {
    let coherence = input.coherence.unwrap_or(0.5).clamp(0.0, 1.0);
    let instability = input.instability.unwrap_or(0.0).clamp(0.0, 1.0);
    clamp01(
        0.6 * input.pressure.clamp(0.0, 1.0)
            + 0.4 * input.surprise.clamp(0.0, 1.0)
            + 0.5 * input.risk.clamp(0.0, 1.0)
            - 0.3 * input.confidence.clamp(0.0, 1.0)
            - 0.1 * coherence
            + 0.2 * instability,
    )
}

fn derivatives(cfg: &HormoneCfg, state: HormoneState, stress: f32) -> (f32, f32, f32, f32) {
    let d_crh = clamp_derivative(
        cfg.k1 * stress - cfg.k2 * state.crh - cfg.k_feedback * state.cortisol * state.crh,
    );
    let d_acth = clamp_derivative(cfg.k3 * state.crh - cfg.k4 * state.acth);
    let d_cort = clamp_derivative(cfg.k5 * state.acth - cfg.k6 * state.cortisol);
    let target_drive = clamp01((1.0 - state.cortisol) * (1.0 - stress * 0.25));
    let d_drive = clamp_derivative(
        cfg.drive_recovery * (target_drive - state.drive)
            - cfg.drive_stress_coupling * stress * state.drive,
    );
    (d_crh, d_acth, d_cort, d_drive)
}

fn clamp_derivative(v: f32) -> f32 {
    v.clamp(-DERIVATIVE_CAP, DERIVATIVE_CAP)
}

fn clamp_state(mut state: HormoneState) -> HormoneState {
    state.crh = clamp01(state.crh);
    state.acth = clamp01(state.acth);
    state.cortisol = clamp01(state.cortisol);
    state.drive = clamp01(state.drive);
    state
}

pub fn map_modulation(cfg: &HormoneCfg, summary: &HormoneStateSummary) -> GatingModulation {
    let risk_penalty_scale = (1.0 + cfg.modulation_risk_scale * summary.stress_index).max(0.0);
    let action_threshold_delta = (cfg.modulation_action_scale * summary.stress_index)
        .clamp(-ACTION_DELTA_CAP, ACTION_DELTA_CAP);
    let exploration_bias_delta = (-cfg.modulation_exploration_scale * summary.stress_index)
        .clamp(-EXPLORATION_DELTA_CAP, EXPLORATION_DELTA_CAP);
    let attention_gain = (1.0 + cfg.modulation_attention_scale * summary.drive).max(0.0);
    let digest = digest_gating_modulation(
        risk_penalty_scale,
        action_threshold_delta,
        exploration_bias_delta,
        attention_gain,
        summary.digest,
    );
    GatingModulation {
        risk_penalty_scale,
        action_threshold_delta,
        exploration_bias_delta,
        attention_gain,
        digest,
    }
}

fn digest_hormone_state(state: &HormoneState) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(state.t.to_le_bytes());
    hasher.update(state.crh.to_le_bytes());
    hasher.update(state.acth.to_le_bytes());
    hasher.update(state.cortisol.to_le_bytes());
    hasher.update(state.drive.to_le_bytes());
    hasher.finalize().into()
}

fn digest_hormone_summary(
    t: u64,
    cortisol: f32,
    drive: f32,
    stress_index: f32,
    evidence_chain_digest: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(t.to_le_bytes());
    hasher.update(cortisol.to_le_bytes());
    hasher.update(drive.to_le_bytes());
    hasher.update(stress_index.to_le_bytes());
    hasher.update(evidence_chain_digest);
    hasher.finalize().into()
}

fn digest_gating_modulation(
    risk_penalty_scale: f32,
    action_threshold_delta: f32,
    exploration_bias_delta: f32,
    attention_gain: f32,
    hormone_digest: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(risk_penalty_scale.to_le_bytes());
    hasher.update(action_threshold_delta.to_le_bytes());
    hasher.update(exploration_bias_delta.to_le_bytes());
    hasher.update(attention_gain.to_le_bytes());
    hasher.update(hormone_digest);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::{hormone_step, HormoneCfg, HormoneInput, HormoneState};

    fn sample_input(
        t: u64,
        pressure: f32,
        surprise: f32,
        risk: f32,
        confidence: f32,
    ) -> HormoneInput {
        HormoneInput {
            t,
            pressure,
            surprise,
            risk,
            confidence,
            coherence: Some(0.6),
            instability: Some(0.2),
            evidence_chain_digest: [7; 32],
        }
    }

    #[test]
    fn deterministic_sequence_produces_identical_digests() {
        let cfg = HormoneCfg::default();
        let seq: Vec<_> = (0_u64..16)
            .map(|t| sample_input(t, 0.35, 0.3 + (t as f32) * 0.01, 0.4, 0.7))
            .collect();

        let mut a = HormoneState::default();
        let mut b = HormoneState::default();
        let mut digests_a = Vec::new();
        let mut digests_b = Vec::new();

        for inp in &seq {
            let out_a = hormone_step(&cfg, a, inp);
            let out_b = hormone_step(&cfg, b, inp);
            a = out_a.state;
            b = out_b.state;
            digests_a.push((out_a.summary.digest, out_a.modulation.digest));
            digests_b.push((out_b.summary.digest, out_b.modulation.digest));
        }

        assert_eq!(digests_a, digests_b);
        assert_eq!(a, b);
    }

    #[test]
    fn state_stays_bounded_under_sustained_stress() {
        let cfg = HormoneCfg::default();
        let mut state = HormoneState::default();

        for t in 0_u64..120 {
            let out = hormone_step(&cfg, state, &sample_input(t, 1.0, 1.0, 1.0, 0.0));
            state = out.state;
            assert!((0.0..=1.0).contains(&state.crh));
            assert!((0.0..=1.0).contains(&state.acth));
            assert!((0.0..=1.0).contains(&state.cortisol));
            assert!((0.0..=1.0).contains(&state.drive));
            assert!(!out.degraded);
        }
    }

    #[test]
    fn higher_sustained_risk_increases_cortisol_baseline() {
        let cfg = HormoneCfg::default();
        let mut low = HormoneState::default();
        let mut high = HormoneState::default();

        for t in 0_u64..80 {
            low = hormone_step(&cfg, low, &sample_input(t, 0.2, 0.2, 0.2, 0.8)).state;
            high = hormone_step(&cfg, high, &sample_input(t, 0.8, 0.8, 0.9, 0.1)).state;
        }

        assert!(high.cortisol > low.cortisol);
    }
}
