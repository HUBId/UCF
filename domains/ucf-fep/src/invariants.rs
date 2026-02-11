#[derive(Clone, Debug, PartialEq)]
pub struct CoherenceCfg {
    pub min_closed_loop_gain: f32,
    pub max_unchecked_drift: f32,
    pub max_memory_pressure: f32,
    pub min_policy_inhibit_on_risk: f32,
}

impl CoherenceCfg {
    pub fn default_v0() -> Self {
        Self {
            min_closed_loop_gain: 0.35,
            max_unchecked_drift: 0.8,
            max_memory_pressure: 0.85,
            min_policy_inhibit_on_risk: 0.65,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CoherenceSnapshot {
    pub surprise: f32,
    pub ess_pressure: f32,
    pub ssm_pressure: f32,
    pub onn_lock: f32,
    pub policy_risk: f32,
    pub geist_drift: f32,
    pub attention_gain: f32,
    pub learn_gate: f32,
    pub memory_priority: f32,
    pub action_inhibit: f32,
    pub homeo_err: f32,
    pub chem_dopa: f32,
    pub chem_5ht: f32,
    pub chem_oxy: f32,
    pub chem_end: f32,
    pub brain_amyg_spikes: f32,
    pub brain_pfc_spikes: f32,
}

pub fn check_coherence_invariants(cfg: &CoherenceCfg, s: &CoherenceSnapshot) -> Result<(), String> {
    let policy_risk = s.policy_risk.clamp(0.0, 1.0);
    let action_inhibit = s.action_inhibit.clamp(0.0, 1.0);
    let ess_pressure = s.ess_pressure.clamp(0.0, 1.0);
    let learn_gate = s.learn_gate.clamp(0.0, 1.0);
    let memory_priority = s.memory_priority.clamp(0.0, 1.0);
    let geist_drift = s.geist_drift.clamp(0.0, 1.0);

    if policy_risk >= 0.8 && action_inhibit < cfg.min_policy_inhibit_on_risk {
        return Err(format!(
            "risk inhibit invariant violated: policy_risk={policy_risk:.3}, action_inhibit={action_inhibit:.3}, min_required={:.3}",
            cfg.min_policy_inhibit_on_risk
        ));
    }

    if ess_pressure >= cfg.max_memory_pressure && !(learn_gate <= 0.6 && memory_priority >= 0.6) {
        return Err(format!(
            "memory pressure invariant violated: ess_pressure={ess_pressure:.3}, learn_gate={learn_gate:.3}, memory_priority={memory_priority:.3}"
        ));
    }

    if geist_drift >= cfg.max_unchecked_drift && !(learn_gate <= 0.5 && action_inhibit >= 0.6) {
        return Err(format!(
            "drift invariant violated: geist_drift={geist_drift:.3}, learn_gate={learn_gate:.3}, action_inhibit={action_inhibit:.3}"
        ));
    }

    let coupling = 0.25 * s.attention_gain.clamp(0.0, 1.0)
        + 0.25 * memory_priority
        + 0.25 * action_inhibit
        + 0.25 * (1.0 - s.homeo_err.clamp(0.0, 1.0));
    if coupling < cfg.min_closed_loop_gain {
        return Err(format!(
            "closed-loop coupling invariant violated: coupling={coupling:.3}, min_required={:.3}, attention={:.3}, memprio={memory_priority:.3}, inhibit={action_inhibit:.3}, homeo_err={:.3}",
            cfg.min_closed_loop_gain,
            s.attention_gain,
            s.homeo_err
        ));
    }

    Ok(())
}
