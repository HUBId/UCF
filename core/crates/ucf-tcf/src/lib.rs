#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_types::Digest32;

const SCALE: i32 = 10_000;
const MAX_GAIN: u16 = 10_000;
const WINDOW_MIN: u8 = 1;
const WINDOW_MAX: u8 = 8;
const BUCKET_MIN: u8 = 1;
const BUCKET_MAX: u8 = 4;
const RISK_HIGH: u16 = 7_000;
const PHI_HIGH: u16 = 7_000;
const PLV_HIGH: u16 = 7_000;

const PARAMS_DOMAIN: &[u8] = b"ucf.tcf.params.v1";
const STATE_DOMAIN: &[u8] = b"ucf.tcf.state.v1";
const INPUT_DOMAIN: &[u8] = b"ucf.tcf.inputs.v1";
const PLAN_DOMAIN: &[u8] = b"ucf.tcf.plan.v1";
const CORE_DOMAIN: &[u8] = b"ucf.tcf.core.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TcfParams {
    /// LTI-ish smoothing (0..10000), higher = more smoothing
    pub smooth_focus: u16,
    pub smooth_learning: u16,
    pub smooth_replay: u16,
    /// windows in cycles (1..=8)
    pub win_focus: u8,
    pub win_output: u8,
    pub win_replay: u8,
    pub win_sleep: u8,
    /// thresholds
    pub phi_sleep_min: u16,
    pub drift_sleep_min: u16,
    pub surprise_replay_min: u16,
    /// clamp bounds
    pub max_gain: u16,
    pub commit: Digest32,
}

impl TcfParams {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        smooth_focus: u16,
        smooth_learning: u16,
        smooth_replay: u16,
        win_focus: u8,
        win_output: u8,
        win_replay: u8,
        win_sleep: u8,
        phi_sleep_min: u16,
        drift_sleep_min: u16,
        surprise_replay_min: u16,
        max_gain: u16,
    ) -> Self {
        let smooth_focus = clamp_gain(smooth_focus);
        let smooth_learning = clamp_gain(smooth_learning);
        let smooth_replay = clamp_gain(smooth_replay);
        let win_focus = clamp_window(win_focus);
        let win_output = clamp_window(win_output);
        let win_replay = clamp_window(win_replay);
        let win_sleep = clamp_window(win_sleep);
        let phi_sleep_min = clamp_gain(phi_sleep_min);
        let drift_sleep_min = clamp_gain(drift_sleep_min);
        let surprise_replay_min = clamp_gain(surprise_replay_min);
        let max_gain = clamp_gain(max_gain).max(1);
        let commit = commit_params(
            smooth_focus,
            smooth_learning,
            smooth_replay,
            win_focus,
            win_output,
            win_replay,
            win_sleep,
            phi_sleep_min,
            drift_sleep_min,
            surprise_replay_min,
            max_gain,
        );
        Self {
            smooth_focus,
            smooth_learning,
            smooth_replay,
            win_focus,
            win_output,
            win_replay,
            win_sleep,
            phi_sleep_min,
            drift_sleep_min,
            surprise_replay_min,
            max_gain,
            commit,
        }
    }
}

impl Default for TcfParams {
    fn default() -> Self {
        Self::new(
            6_500, 7_500, 5_500, 4, 4, 3, 3, 3_000, 6_000, 7_000, MAX_GAIN,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TcfState {
    /// filtered signals (i16 fixed-point)
    pub f_risk: i16,
    pub f_drift: i16,
    pub f_surprise: i16,
    pub f_phi: i16,
    pub f_plv: i16,
    /// window phase counters
    pub focus_ctr: u8,
    pub replay_ctr: u8,
    pub sleep_ctr: u8,
    /// last decisions
    pub sleep_active: bool,
    pub replay_active: bool,
    pub commit: Digest32,
}

impl TcfState {
    fn new(params: &TcfParams) -> Self {
        let state = Self {
            f_risk: 0,
            f_drift: 0,
            f_surprise: 0,
            f_phi: 0,
            f_plv: 0,
            focus_ctr: 0,
            replay_ctr: 0,
            sleep_ctr: 0,
            sleep_active: false,
            replay_active: false,
            commit: Digest32::new([0u8; 32]),
        };
        refresh_state_commit(state, params.commit)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TcfInputs {
    pub cycle_id: u64,
    pub phase_bus_commit: Digest32,
    pub gamma_bucket: u8,
    pub global_plv: u16,
    pub phi_proxy: u16,
    pub risk: u16,
    pub drift: u16,
    pub surprise: u16,
    /// upstream hints:
    pub iit_hints_commit: Digest32,
    pub tighten_sync: bool,
    pub damp_output: bool,
    pub damp_learning: bool,
    pub request_replay: bool,
    /// coupling:
    pub coupling_influences_root: Digest32,
    /// nsr/policy hard outcomes for extra restriction
    pub nsr_verdict: u8,
    pub policy_ok: bool,
    pub commit: Digest32,
}

impl TcfInputs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cycle_id: u64,
        phase_bus_commit: Digest32,
        gamma_bucket: u8,
        global_plv: u16,
        phi_proxy: u16,
        risk: u16,
        drift: u16,
        surprise: u16,
        iit_hints_commit: Digest32,
        tighten_sync: bool,
        damp_output: bool,
        damp_learning: bool,
        request_replay: bool,
        coupling_influences_root: Digest32,
        nsr_verdict: u8,
        policy_ok: bool,
    ) -> Self {
        let global_plv = clamp_gain(global_plv);
        let phi_proxy = clamp_gain(phi_proxy);
        let risk = clamp_gain(risk);
        let drift = clamp_gain(drift);
        let surprise = clamp_gain(surprise);
        let commit = commit_inputs(
            cycle_id,
            phase_bus_commit,
            gamma_bucket,
            global_plv,
            phi_proxy,
            risk,
            drift,
            surprise,
            iit_hints_commit,
            tighten_sync,
            damp_output,
            damp_learning,
            request_replay,
            coupling_influences_root,
            nsr_verdict,
            policy_ok,
        );
        Self {
            cycle_id,
            phase_bus_commit,
            gamma_bucket,
            global_plv,
            phi_proxy,
            risk,
            drift,
            surprise,
            iit_hints_commit,
            tighten_sync,
            damp_output,
            damp_learning,
            request_replay,
            coupling_influences_root,
            nsr_verdict,
            policy_ok,
            commit,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TcfPlan {
    pub cycle_id: u64,
    /// gating scalars 0..10000 applied downstream
    pub attention_gain_cap: u16,
    pub learning_gain_cap: u16,
    pub output_gain_cap: u16,
    /// window state:
    pub sleep_active: bool,
    pub replay_active: bool,
    /// coherence controls
    pub lock_window_buckets: u8,
    pub smoothing_override: u16,
    pub commit: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TcfCore {
    pub params: TcfParams,
    pub state: TcfState,
    pub commit: Digest32,
}

impl TcfCore {
    pub fn new(params: TcfParams) -> Self {
        let state = TcfState::new(&params);
        let commit = commit_core(params.commit, state.commit);
        Self {
            params,
            state,
            commit,
        }
    }

    pub fn tick(&mut self, inp: &TcfInputs) -> TcfPlan {
        let mut next_state = self.state;
        next_state.f_risk = ema(
            next_state.f_risk,
            u16_to_i16(inp.risk),
            self.params.smooth_focus,
        );
        next_state.f_drift = ema(
            next_state.f_drift,
            u16_to_i16(inp.drift),
            self.params.smooth_learning,
        );
        next_state.f_surprise = ema(
            next_state.f_surprise,
            u16_to_i16(inp.surprise),
            self.params.smooth_replay,
        );
        next_state.f_phi = ema(
            next_state.f_phi,
            u16_to_i16(inp.phi_proxy),
            self.params.smooth_focus,
        );
        next_state.f_plv = ema(
            next_state.f_plv,
            u16_to_i16(inp.global_plv),
            self.params.smooth_focus,
        );

        next_state.focus_ctr = advance_ctr(next_state.focus_ctr, self.params.win_focus);

        let phi_val = filtered_u16(next_state.f_phi);
        let drift_val = filtered_u16(next_state.f_drift);
        let surprise_val = filtered_u16(next_state.f_surprise);
        let risk_val = filtered_u16(next_state.f_risk);
        let plv_val = filtered_u16(next_state.f_plv);

        let phi_low = phi_val < self.params.phi_sleep_min;
        let drift_high = drift_val >= self.params.drift_sleep_min;
        let surprise_high = surprise_val >= self.params.surprise_replay_min;

        let sleep_trigger = (inp.tighten_sync || phi_low) && drift_high;
        if sleep_trigger {
            next_state.sleep_ctr = self.params.win_sleep;
        } else if next_state.sleep_ctr > 0 {
            next_state.sleep_ctr = next_state.sleep_ctr.saturating_sub(1);
        }
        next_state.sleep_active = next_state.sleep_ctr > 0;

        let replay_trigger = inp.request_replay || (surprise_high && phi_low);
        if replay_trigger {
            next_state.replay_ctr = self.params.win_replay;
        } else if next_state.replay_ctr > 0 {
            next_state.replay_ctr = next_state.replay_ctr.saturating_sub(1);
        }
        next_state.replay_active = next_state.replay_ctr > 0;

        let mut lock_window_buckets = map_focus_to_bucket(self.params.win_focus);
        if inp.tighten_sync || phi_low {
            lock_window_buckets = lock_window_buckets.saturating_sub(1).max(BUCKET_MIN);
        }
        if phi_val >= PHI_HIGH && plv_val >= PLV_HIGH {
            lock_window_buckets = lock_window_buckets.saturating_add(1).min(BUCKET_MAX);
        }

        let smoothing_base = self
            .params
            .smooth_focus
            .max(self.params.smooth_learning)
            .max(self.params.smooth_replay);
        let smoothing_override = if drift_high || phi_low {
            smoothing_base.saturating_add(1_500).min(MAX_GAIN)
        } else {
            smoothing_base
        };

        let mut attention_gain_cap = self.params.max_gain;
        if risk_val >= RISK_HIGH {
            attention_gain_cap = scale_gain(attention_gain_cap, 8_000);
        }
        if drift_high {
            attention_gain_cap = scale_gain(attention_gain_cap, 8_500);
        }
        if inp.tighten_sync {
            attention_gain_cap = scale_gain(attention_gain_cap, 9_000);
        }
        attention_gain_cap = attention_gain_cap.min(self.params.max_gain);

        let mut learning_gain_cap = self.params.max_gain;
        if inp.damp_learning {
            learning_gain_cap = scale_gain(learning_gain_cap, 7_000);
        }
        if risk_val >= RISK_HIGH {
            learning_gain_cap = scale_gain(learning_gain_cap, 8_500);
        }
        if next_state.sleep_active {
            learning_gain_cap = learning_gain_cap.min(self.params.max_gain / 2);
        }
        learning_gain_cap = learning_gain_cap.min(self.params.max_gain);

        let output_gain_cap =
            if !inp.policy_ok || inp.nsr_verdict != 0 || inp.damp_output || next_state.sleep_active
            {
                0
            } else {
                let coherence = phi_val.min(plv_val);
                scale_gain(self.params.max_gain, coherence)
            };

        let output_gain_cap = output_gain_cap.min(self.params.max_gain);

        next_state = refresh_state_commit(next_state, self.params.commit);
        self.state = next_state;
        self.commit = commit_core(self.params.commit, self.state.commit);

        let commit = commit_plan(
            inp.cycle_id,
            attention_gain_cap,
            learning_gain_cap,
            output_gain_cap,
            next_state.sleep_active,
            next_state.replay_active,
            lock_window_buckets,
            smoothing_override,
            inp.commit,
            self.params.commit,
            self.state.commit,
        );

        TcfPlan {
            cycle_id: inp.cycle_id,
            attention_gain_cap,
            learning_gain_cap,
            output_gain_cap,
            sleep_active: next_state.sleep_active,
            replay_active: next_state.replay_active,
            lock_window_buckets,
            smoothing_override,
            commit,
        }
    }
}

impl Default for TcfCore {
    fn default() -> Self {
        Self::new(TcfParams::default())
    }
}

fn clamp_gain(value: u16) -> u16 {
    value.min(MAX_GAIN)
}

fn clamp_window(value: u8) -> u8 {
    value.clamp(WINDOW_MIN, WINDOW_MAX)
}

fn map_focus_to_bucket(win_focus: u8) -> u8 {
    let mapped = (u16::from(win_focus).saturating_add(1) / 2) as u8;
    mapped.clamp(BUCKET_MIN, BUCKET_MAX)
}

fn u16_to_i16(value: u16) -> i16 {
    value.min(i16::MAX as u16) as i16
}

fn filtered_u16(value: i16) -> u16 {
    if value <= 0 {
        0
    } else {
        value as u16
    }
}

fn ema(prev: i16, curr: i16, smooth_k: u16) -> i16 {
    let k = i32::from(smooth_k.min(MAX_GAIN));
    let prev = i32::from(prev);
    let curr = i32::from(curr);
    let out = (k.saturating_mul(prev) + (SCALE - k).saturating_mul(curr)) / SCALE;
    out.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

fn scale_gain(value: u16, cap: u16) -> u16 {
    let scaled = u32::from(value).saturating_mul(u32::from(cap)) / u32::from(MAX_GAIN);
    scaled.min(u32::from(MAX_GAIN)) as u16
}

fn advance_ctr(current: u8, window: u8) -> u8 {
    if window <= 1 {
        return 0;
    }
    let next = current.saturating_add(1);
    if next >= window {
        0
    } else {
        next
    }
}

#[allow(clippy::too_many_arguments)]
fn commit_params(
    smooth_focus: u16,
    smooth_learning: u16,
    smooth_replay: u16,
    win_focus: u8,
    win_output: u8,
    win_replay: u8,
    win_sleep: u8,
    phi_sleep_min: u16,
    drift_sleep_min: u16,
    surprise_replay_min: u16,
    max_gain: u16,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(PARAMS_DOMAIN);
    hasher.update(&smooth_focus.to_be_bytes());
    hasher.update(&smooth_learning.to_be_bytes());
    hasher.update(&smooth_replay.to_be_bytes());
    hasher.update(&[win_focus]);
    hasher.update(&[win_output]);
    hasher.update(&[win_replay]);
    hasher.update(&[win_sleep]);
    hasher.update(&phi_sleep_min.to_be_bytes());
    hasher.update(&drift_sleep_min.to_be_bytes());
    hasher.update(&surprise_replay_min.to_be_bytes());
    hasher.update(&max_gain.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn refresh_state_commit(state: TcfState, params_commit: Digest32) -> TcfState {
    let mut hasher = Hasher::new();
    hasher.update(STATE_DOMAIN);
    hasher.update(params_commit.as_bytes());
    hasher.update(&state.f_risk.to_be_bytes());
    hasher.update(&state.f_drift.to_be_bytes());
    hasher.update(&state.f_surprise.to_be_bytes());
    hasher.update(&state.f_phi.to_be_bytes());
    hasher.update(&state.f_plv.to_be_bytes());
    hasher.update(&[state.focus_ctr]);
    hasher.update(&[state.replay_ctr]);
    hasher.update(&[state.sleep_ctr]);
    hasher.update(&[state.sleep_active as u8]);
    hasher.update(&[state.replay_active as u8]);
    let commit = Digest32::new(*hasher.finalize().as_bytes());
    TcfState { commit, ..state }
}

#[allow(clippy::too_many_arguments)]
fn commit_inputs(
    cycle_id: u64,
    phase_bus_commit: Digest32,
    gamma_bucket: u8,
    global_plv: u16,
    phi_proxy: u16,
    risk: u16,
    drift: u16,
    surprise: u16,
    iit_hints_commit: Digest32,
    tighten_sync: bool,
    damp_output: bool,
    damp_learning: bool,
    request_replay: bool,
    coupling_influences_root: Digest32,
    nsr_verdict: u8,
    policy_ok: bool,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(INPUT_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(phase_bus_commit.as_bytes());
    hasher.update(&[gamma_bucket]);
    hasher.update(&global_plv.to_be_bytes());
    hasher.update(&phi_proxy.to_be_bytes());
    hasher.update(&risk.to_be_bytes());
    hasher.update(&drift.to_be_bytes());
    hasher.update(&surprise.to_be_bytes());
    hasher.update(iit_hints_commit.as_bytes());
    hasher.update(&[tighten_sync as u8]);
    hasher.update(&[damp_output as u8]);
    hasher.update(&[damp_learning as u8]);
    hasher.update(&[request_replay as u8]);
    hasher.update(coupling_influences_root.as_bytes());
    hasher.update(&[nsr_verdict]);
    hasher.update(&[policy_ok as u8]);
    Digest32::new(*hasher.finalize().as_bytes())
}

#[allow(clippy::too_many_arguments)]
fn commit_plan(
    cycle_id: u64,
    attention_gain_cap: u16,
    learning_gain_cap: u16,
    output_gain_cap: u16,
    sleep_active: bool,
    replay_active: bool,
    lock_window_buckets: u8,
    smoothing_override: u16,
    input_commit: Digest32,
    params_commit: Digest32,
    state_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(PLAN_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(&attention_gain_cap.to_be_bytes());
    hasher.update(&learning_gain_cap.to_be_bytes());
    hasher.update(&output_gain_cap.to_be_bytes());
    hasher.update(&[sleep_active as u8]);
    hasher.update(&[replay_active as u8]);
    hasher.update(&[lock_window_buckets]);
    hasher.update(&smoothing_override.to_be_bytes());
    hasher.update(input_commit.as_bytes());
    hasher.update(params_commit.as_bytes());
    hasher.update(state_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_core(params_commit: Digest32, state_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(CORE_DOMAIN);
    hasher.update(params_commit.as_bytes());
    hasher.update(state_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_nsr::NsrVerdict;

    fn base_inputs() -> TcfInputs {
        TcfInputs::new(
            1,
            Digest32::new([1u8; 32]),
            2,
            6_000,
            6_000,
            2_000,
            2_000,
            2_000,
            Digest32::new([2u8; 32]),
            false,
            false,
            false,
            false,
            Digest32::new([3u8; 32]),
            NsrVerdict::Allow.as_u8(),
            true,
        )
    }

    #[test]
    fn plan_commit_is_deterministic_for_same_inputs() {
        let mut core_a = TcfCore::default();
        let mut core_b = TcfCore::default();
        let inputs = base_inputs();
        let plan_a = core_a.tick(&inputs);
        let plan_b = core_b.tick(&inputs);
        assert_eq!(plan_a.commit, plan_b.commit);
    }

    #[test]
    fn sleep_gating_holds_for_window() {
        let params = TcfParams::new(0, 0, 0, 4, 4, 2, 3, 3_000, 6_000, 7_000, 10_000);
        let mut core = TcfCore::new(params);
        let mut inputs = base_inputs();
        inputs.phi_proxy = 1_000;
        inputs.drift = 8_000;
        let plan = core.tick(&inputs);
        assert!(plan.sleep_active);

        inputs.phi_proxy = 9_000;
        inputs.drift = 1_000;
        let plan_next = core.tick(&inputs);
        assert!(plan_next.sleep_active);
        let plan_final = core.tick(&inputs);
        assert!(plan_final.sleep_active);
        let plan_done = core.tick(&inputs);
        assert!(!plan_done.sleep_active);
    }

    #[test]
    fn output_gain_is_suppressed_by_sleep_and_nsr() {
        let params = TcfParams::new(0, 0, 0, 4, 4, 2, 2, 3_000, 6_000, 7_000, 10_000);
        let mut core = TcfCore::new(params);
        let mut inputs = base_inputs();
        inputs.phi_proxy = 1_000;
        inputs.drift = 9_000;
        let plan = core.tick(&inputs);
        assert_eq!(plan.output_gain_cap, 0);

        let mut core = TcfCore::new(params);
        let mut inputs = base_inputs();
        inputs.phi_proxy = 9_000;
        inputs.drift = 1_000;
        inputs.nsr_verdict = NsrVerdict::Restrict.as_u8();
        let plan = core.tick(&inputs);
        assert_eq!(plan.output_gain_cap, 0);
    }

    #[test]
    fn tighten_sync_reduces_lock_window() {
        let params = TcfParams::new(0, 0, 0, 6, 4, 2, 2, 3_000, 6_000, 7_000, 10_000);
        let mut core = TcfCore::new(params);
        let mut inputs = base_inputs();
        inputs.phi_proxy = 5_000;
        inputs.global_plv = 5_000;
        let plan = core.tick(&inputs);
        let base_window = plan.lock_window_buckets;
        let mut core = TcfCore::new(params);
        let mut inputs = base_inputs();
        inputs.phi_proxy = 5_000;
        inputs.global_plv = 5_000;
        inputs.tighten_sync = true;
        let tightened = core.tick(&inputs);
        assert!(tightened.lock_window_buckets < base_window);
    }
}
