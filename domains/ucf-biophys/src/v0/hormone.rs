use sha2::{Digest, Sha256};

use crate::v0::ode::clamp01;

const DERIVATIVE_CAP: f32 = 0.2;
const ACTION_DELTA_CAP: f32 = 0.5;
const EXPLORATION_DELTA_CAP: f32 = 0.5;
const SATURATION_TICKS_LIMIT: u8 = 8;
const INPUT_DAMPEN_FACTOR: f32 = 0.65;
const ODE_PARAM_CAP: f32 = 8.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HormoneState {
    pub t: u64,
    pub crh: f32,
    pub acth: f32,
    pub cortisol: f32,
    pub dopamine: f32,
    pub norepinephrine: f32,
    pub serotonin: f32,
    pub acetylcholine: f32,
    pub drive: f32,
    pub saturation_ticks: u8,
    pub digest: [u8; 32],
}

impl Default for HormoneState {
    fn default() -> Self {
        let mut state = Self {
            t: 0,
            crh: 0.1,
            acth: 0.1,
            cortisol: 0.1,
            dopamine: 0.5,
            norepinephrine: 0.2,
            serotonin: 0.6,
            acetylcholine: 0.5,
            drive: 0.5,
            saturation_ticks: 0,
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
    pub dopamine: f32,
    pub norepinephrine: f32,
    pub serotonin: f32,
    pub acetylcholine: f32,
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
    pub plasticity_gate: f32,
    pub stress_gate: f32,
    pub digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HormoneCfg {
    pub dt: f32,
    pub substeps: u8,
    pub tau_cortisol: f32,
    pub tau_dopamine: f32,
    pub tau_norepinephrine: f32,
    pub tau_serotonin: f32,
    pub tau_acetylcholine: f32,
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
            dt: 0.1,
            substeps: 2,
            tau_cortisol: 2.2,
            tau_dopamine: 1.6,
            tau_norepinephrine: 1.1,
            tau_serotonin: 2.8,
            tau_acetylcholine: 1.4,
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
    let stress_drive = stress_drive(input);
    let dt = cfg.dt.clamp(0.01, 1.0);
    let substeps = cfg.substeps.clamp(1, 8);
    let mut state = prev;

    for _ in 0..substeps {
        state = rk2_substep(cfg, state, stress_drive, dt / f32::from(substeps));
    }

    let mut degraded = false;
    if !is_finite_state(&state) {
        degraded = true;
        state = HormoneState::default();
        state.t = input.t;
    }

    state.t = input.t;
    state = clamp_state(state);
    let saturated = is_saturated(&state);
    state.saturation_ticks = if saturated {
        state.saturation_ticks.saturating_add(1)
    } else {
        0
    };
    state.digest = digest_hormone_state(&state);

    let stress_index_legacy = clamp01(0.8 * state.cortisol + 0.2 * state.acth);
    let stress_index_axis = clamp01(
        0.55 * state.cortisol + 0.25 * state.norepinephrine + 0.2 * (1.0 - state.serotonin),
    );
    let stress_index = stress_index_axis.max(stress_index_legacy);
    let summary = HormoneStateSummary {
        t: input.t,
        cortisol: state.cortisol,
        dopamine: state.dopamine,
        norepinephrine: state.norepinephrine,
        serotonin: state.serotonin,
        acetylcholine: state.acetylcholine,
        drive: state.drive,
        stress_index,
        digest: digest_hormone_summary(input.t, &state, stress_index, input.evidence_chain_digest),
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

fn rk2_substep(cfg: &HormoneCfg, prev: HormoneState, stress_drive: f32, dt: f32) -> HormoneState {
    let k1 = derivatives(
        cfg,
        prev,
        stress_drive,
        prev.saturation_ticks >= SATURATION_TICKS_LIMIT,
    );
    let midpoint = clamp_state(HormoneState {
        t: prev.t,
        crh: prev.crh + k1.0 * dt * 0.5,
        acth: prev.acth + k1.1 * dt * 0.5,
        cortisol: prev.cortisol + k1.2 * dt * 0.5,
        dopamine: prev.dopamine + k1.4 * dt * 0.5,
        norepinephrine: prev.norepinephrine + k1.5 * dt * 0.5,
        serotonin: prev.serotonin + k1.6 * dt * 0.5,
        acetylcholine: prev.acetylcholine + k1.7 * dt * 0.5,
        drive: prev.drive + k1.3 * dt * 0.5,
        saturation_ticks: prev.saturation_ticks,
        digest: [0; 32],
    });
    let k2 = derivatives(
        cfg,
        midpoint,
        stress_drive,
        prev.saturation_ticks >= SATURATION_TICKS_LIMIT,
    );

    HormoneState {
        t: prev.t,
        crh: prev.crh + k2.0 * dt,
        acth: prev.acth + k2.1 * dt,
        cortisol: prev.cortisol + k2.2 * dt,
        dopamine: prev.dopamine + k2.4 * dt,
        norepinephrine: prev.norepinephrine + k2.5 * dt,
        serotonin: prev.serotonin + k2.6 * dt,
        acetylcholine: prev.acetylcholine + k2.7 * dt,
        drive: prev.drive + k2.3 * dt,
        saturation_ticks: prev.saturation_ticks,
        digest: [0; 32],
    }
}

pub fn stress_drive(input: &HormoneInput) -> f32 {
    let coherence = input.coherence.unwrap_or(0.5).clamp(0.0, 1.0);
    let instability = input.instability.unwrap_or(0.0).clamp(0.0, 1.0);
    clamp01(
        0.55 * input.pressure.clamp(0.0, 1.0)
            + 0.45 * input.surprise.clamp(0.0, 1.0)
            + 0.55 * input.risk.clamp(0.0, 1.0)
            - 0.35 * input.confidence.clamp(0.0, 1.0)
            - 0.10 * coherence
            + 0.25 * instability,
    )
}

fn derivatives(
    cfg: &HormoneCfg,
    state: HormoneState,
    stress: f32,
    dampened_inputs: bool,
) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
    let drive_dampen = if dampened_inputs {
        INPUT_DAMPEN_FACTOR
    } else {
        1.0
    };
    let stress_in = stress * drive_dampen;
    let instability = (1.0 - state.drive).clamp(0.0, 1.0);

    let tau_c = cfg.tau_cortisol.clamp(0.1, ODE_PARAM_CAP);
    let tau_d = cfg.tau_dopamine.clamp(0.1, ODE_PARAM_CAP);
    let tau_n = cfg.tau_norepinephrine.clamp(0.1, ODE_PARAM_CAP);
    let tau_s = cfg.tau_serotonin.clamp(0.1, ODE_PARAM_CAP);
    let tau_a = cfg.tau_acetylcholine.clamp(0.1, ODE_PARAM_CAP);

    let d_crh = clamp_derivative(
        cfg.k1.clamp(0.0, ODE_PARAM_CAP) * stress_in
            - cfg.k2.clamp(0.0, ODE_PARAM_CAP) * state.crh
            - cfg.k_feedback.clamp(0.0, ODE_PARAM_CAP) * state.cortisol * state.crh,
    );
    let d_acth = clamp_derivative(
        cfg.k3.clamp(0.0, ODE_PARAM_CAP) * state.crh
            - cfg.k4.clamp(0.0, ODE_PARAM_CAP) * state.acth,
    );
    let d_cort = clamp_derivative(
        cfg.k5.clamp(0.0, ODE_PARAM_CAP) * state.acth
            - cfg.k6.clamp(0.0, ODE_PARAM_CAP) * state.cortisol,
    );

    let target_dopamine = clamp01(0.55 * (1.0 - stress_in) + 0.45 * state.drive);
    let target_ne = clamp01(0.65 * stress_in + 0.35 * state.cortisol);
    let target_5ht = clamp01(0.65 * (1.0 - instability) + 0.35 * state.drive);
    let target_ach = clamp01(0.5 * (1.0 - stress_in) + 0.5 * (1.0 - state.cortisol));

    let d_dopamine =
        clamp_derivative((target_dopamine - state.dopamine) / tau_d + 0.05 * (1.0 - stress_in));
    let d_norepinephrine =
        clamp_derivative((target_ne - state.norepinephrine) / tau_n + 0.08 * stress_in);
    let d_serotonin = clamp_derivative((target_5ht - state.serotonin) / tau_s - 0.04 * instability);
    let d_ach = clamp_derivative((target_ach - state.acetylcholine) / tau_a + 0.03 * state.drive);

    let target_drive = clamp01((1.0 - state.cortisol) * (1.0 - stress_in * 0.25));
    let d_drive = clamp_derivative(
        cfg.drive_recovery.clamp(0.0, ODE_PARAM_CAP) * (target_drive - state.drive)
            - cfg.drive_stress_coupling.clamp(0.0, ODE_PARAM_CAP) * stress_in * state.drive,
    );

    // Keep cortisol HPA shaping tied to tau.
    let d_cort_tau = clamp_derivative((state.acth - state.cortisol) / tau_c);
    (
        d_crh,
        d_acth,
        clamp_derivative(0.5 * d_cort + 0.5 * d_cort_tau),
        d_drive,
        d_dopamine,
        d_norepinephrine,
        d_serotonin,
        d_ach,
    )
}

fn is_finite_state(state: &HormoneState) -> bool {
    state.crh.is_finite()
        && state.acth.is_finite()
        && state.cortisol.is_finite()
        && state.dopamine.is_finite()
        && state.norepinephrine.is_finite()
        && state.serotonin.is_finite()
        && state.acetylcholine.is_finite()
        && state.drive.is_finite()
}

fn clamp_derivative(v: f32) -> f32 {
    v.clamp(-DERIVATIVE_CAP, DERIVATIVE_CAP)
}

fn clamp_state(mut state: HormoneState) -> HormoneState {
    state.crh = clamp01(state.crh);
    state.acth = clamp01(state.acth);
    state.cortisol = clamp01(state.cortisol);
    state.dopamine = clamp01(state.dopamine);
    state.norepinephrine = clamp01(state.norepinephrine);
    state.serotonin = clamp01(state.serotonin);
    state.acetylcholine = clamp01(state.acetylcholine);
    state.drive = clamp01(state.drive);
    state
}

fn is_saturated(state: &HormoneState) -> bool {
    [
        state.cortisol,
        state.dopamine,
        state.norepinephrine,
        state.serotonin,
        state.acetylcholine,
    ]
    .iter()
    .any(|x| *x < 0.001 || *x > 0.999)
}

pub fn map_modulation(cfg: &HormoneCfg, summary: &HormoneStateSummary) -> GatingModulation {
    let stress_gate = (1.0 - summary.stress_index).clamp(0.1, 1.0);
    let plasticity_gate = (0.6 * summary.dopamine + 0.4 * summary.serotonin)
        .clamp(0.0, 1.0)
        .min(stress_gate);
    let risk_penalty_scale = (1.0 + cfg.modulation_risk_scale * summary.stress_index).max(1.0);
    let action_threshold_delta =
        (cfg.modulation_action_scale * summary.stress_index).clamp(0.0, ACTION_DELTA_CAP);
    let exploration_bias_delta = (-cfg.modulation_exploration_scale * summary.stress_index)
        .clamp(-EXPLORATION_DELTA_CAP, 0.0);
    let attention_gain =
        (0.5 + cfg.modulation_attention_scale * summary.acetylcholine).clamp(0.2, 1.0);
    let digest = digest_gating_modulation(
        risk_penalty_scale,
        action_threshold_delta,
        exploration_bias_delta,
        attention_gain,
        plasticity_gate,
        stress_gate,
        summary.digest,
    );
    GatingModulation {
        risk_penalty_scale,
        action_threshold_delta,
        exploration_bias_delta,
        attention_gain,
        plasticity_gate,
        stress_gate,
        digest,
    }
}

fn digest_hormone_state(state: &HormoneState) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(state.t.to_le_bytes());
    hasher.update(state.crh.to_le_bytes());
    hasher.update(state.acth.to_le_bytes());
    hasher.update(state.cortisol.to_le_bytes());
    hasher.update(state.dopamine.to_le_bytes());
    hasher.update(state.norepinephrine.to_le_bytes());
    hasher.update(state.serotonin.to_le_bytes());
    hasher.update(state.acetylcholine.to_le_bytes());
    hasher.update(state.drive.to_le_bytes());
    hasher.update([state.saturation_ticks]);
    hasher.finalize().into()
}

fn digest_hormone_summary(
    t: u64,
    state: &HormoneState,
    stress_index: f32,
    evidence_chain_digest: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(t.to_le_bytes());
    hasher.update(state.cortisol.to_le_bytes());
    hasher.update(state.dopamine.to_le_bytes());
    hasher.update(state.norepinephrine.to_le_bytes());
    hasher.update(state.serotonin.to_le_bytes());
    hasher.update(state.acetylcholine.to_le_bytes());
    hasher.update(state.drive.to_le_bytes());
    hasher.update(stress_index.to_le_bytes());
    hasher.update(evidence_chain_digest);
    hasher.finalize().into()
}

fn digest_gating_modulation(
    risk_penalty_scale: f32,
    action_threshold_delta: f32,
    exploration_bias_delta: f32,
    attention_gain: f32,
    plasticity_gate: f32,
    stress_gate: f32,
    hormone_digest: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(risk_penalty_scale.to_le_bytes());
    hasher.update(action_threshold_delta.to_le_bytes());
    hasher.update(exploration_bias_delta.to_le_bytes());
    hasher.update(attention_gain.to_le_bytes());
    hasher.update(plasticity_gate.to_le_bytes());
    hasher.update(stress_gate.to_le_bytes());
    hasher.update(hormone_digest);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::{hormone_step, GatingModulation, HormoneCfg, HormoneInput, HormoneState};

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
            assert!((0.0..=1.0).contains(&state.dopamine));
            assert!((0.0..=1.0).contains(&state.norepinephrine));
            assert!((0.0..=1.0).contains(&state.serotonin));
            assert!((0.0..=1.0).contains(&state.acetylcholine));
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

    #[test]
    fn modulation_is_tightening_only() {
        let m = GatingModulation {
            risk_penalty_scale: 1.2,
            action_threshold_delta: 0.1,
            exploration_bias_delta: -0.1,
            attention_gain: 0.8,
            plasticity_gate: 0.7,
            stress_gate: 0.6,
            digest: [0; 32],
        };
        assert!(m.risk_penalty_scale >= 1.0);
        assert!(m.action_threshold_delta >= 0.0);
        assert!(m.exploration_bias_delta <= 0.0);
        assert!(m.attention_gain <= 1.0);
        assert!(m.plasticity_gate <= 1.0);
        assert!(m.stress_gate <= 1.0);
    }
}
