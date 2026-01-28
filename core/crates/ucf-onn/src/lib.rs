#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_influence::InfluenceNodeId;
use ucf_types::Digest32;

const PHASE_WRAP: i32 = 65_536;
const MAX_SIGNAL: u16 = 10_000;
const MAX_OSCILLATORS: usize = 16;
const SIN_LUT_SCALE: i16 = 10_000;
const SIN_LUT: [i16; 256] = [
    0, -245, -491, -736, -980, -1224, -1467, -1710, -1951, -2191, -2430, -2667, -2903, -3137,
    -3369, -3599, -3827, -4052, -4276, -4496, -4714, -4929, -5141, -5350, -5556, -5758, -5957,
    -6152, -6344, -6532, -6716, -6895, -7071, -7242, -7410, -7572, -7730, -7883, -8032, -8176,
    -8315, -8449, -8577, -8701, -8819, -8932, -9040, -9142, -9239, -9330, -9415, -9495, -9569,
    -9638, -9700, -9757, -9808, -9853, -9892, -9925, -9952, -9973, -9988, -9997, -10000, -9997,
    -9988, -9973, -9952, -9925, -9892, -9853, -9808, -9757, -9700, -9638, -9569, -9495, -9415,
    -9330, -9239, -9142, -9040, -8932, -8819, -8701, -8577, -8449, -8315, -8176, -8032, -7883,
    -7730, -7572, -7410, -7242, -7071, -6895, -6716, -6532, -6344, -6152, -5957, -5758, -5556,
    -5350, -5141, -4929, -4714, -4496, -4276, -4052, -3827, -3599, -3369, -3137, -2903, -2667,
    -2430, -2191, -1951, -1710, -1467, -1224, -980, -736, -491, -245, 0, 245, 491, 736, 980, 1224,
    1467, 1710, 1951, 2191, 2430, 2667, 2903, 3137, 3369, 3599, 3827, 4052, 4276, 4496, 4714, 4929,
    5141, 5350, 5556, 5758, 5957, 6152, 6344, 6532, 6716, 6895, 7071, 7242, 7410, 7572, 7730, 7883,
    8032, 8176, 8315, 8449, 8577, 8701, 8819, 8932, 9040, 9142, 9239, 9330, 9415, 9495, 9569, 9638,
    9700, 9757, 9808, 9853, 9892, 9925, 9952, 9973, 9988, 9997, 10000, 9997, 9988, 9973, 9952,
    9925, 9892, 9853, 9808, 9757, 9700, 9638, 9569, 9495, 9415, 9330, 9239, 9142, 9040, 8932, 8819,
    8701, 8577, 8449, 8315, 8176, 8032, 7883, 7730, 7572, 7410, 7242, 7071, 6895, 6716, 6532, 6344,
    6152, 5957, 5758, 5556, 5350, 5141, 4929, 4714, 4496, 4276, 4052, 3827, 3599, 3369, 3137, 2903,
    2667, 2430, 2191, 1951, 1710, 1467, 1224, 980, 736, 491, 245,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OscId {
    Jepa,
    Ssm,
    Ncde,
    Cde,
    Nsr,
    Geist,
    Iit,
    Output,
    BlueBrain,
    Unknown(u16),
}

impl OscId {
    pub fn as_u16(self) -> u16 {
        match self {
            Self::Jepa => 1,
            Self::Ssm => 2,
            Self::Ncde => 3,
            Self::Cde => 4,
            Self::Nsr => 5,
            Self::Geist => 6,
            Self::Iit => 7,
            Self::Output => 8,
            Self::BlueBrain => 9,
            Self::Unknown(value) => value,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OscState {
    pub id: OscId,
    pub phase: u16,
    pub omega: i16,
    pub amp: u16,
    pub commit: Digest32,
}

impl OscState {
    pub fn new(id: OscId, phase: u16, omega: i16, amp: u16) -> Self {
        let commit = commit_osc_state(id, phase, omega, amp);
        Self {
            id,
            phase,
            omega,
            amp,
            commit,
        }
    }

    pub fn update_commit(&mut self) {
        self.commit = commit_osc_state(self.id, self.phase, self.omega, self.amp);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnnParams {
    pub base_step: u16,
    pub coupling: u16,
    pub max_delta: u16,
    pub lock_window: u16,
    pub commit: Digest32,
}

impl OnnParams {
    pub fn new(base_step: u16, coupling: u16, max_delta: u16, lock_window: u16) -> Self {
        let commit = commit_params(base_step, coupling, max_delta, lock_window);
        Self {
            base_step,
            coupling,
            max_delta,
            lock_window,
            commit,
        }
    }
}

impl Default for OnnParams {
    fn default() -> Self {
        Self::new(128, 3200, 512, 16_384)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairLock {
    pub a: OscId,
    pub b: OscId,
    pub lock: u16,
    pub phase_diff: u16,
    pub commit: Digest32,
}

impl PairLock {
    fn new(a: OscId, b: OscId, lock: u16, phase_diff: u16) -> Self {
        let commit = commit_pair_lock(a, b, lock, phase_diff);
        Self {
            a,
            b,
            lock,
            phase_diff,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnnInputs {
    pub cycle_id: u64,
    pub influence_pulses_root: Digest32,
    pub influence_node_values: Vec<(InfluenceNodeId, i16)>,
    pub coherence_hint: u16,
    pub risk: u16,
    pub drift: u16,
    pub nsr_verdict: u8,
    pub commit: Digest32,
}

impl OnnInputs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cycle_id: u64,
        influence_pulses_root: Digest32,
        influence_node_values: Vec<(InfluenceNodeId, i16)>,
        coherence_hint: u16,
        risk: u16,
        drift: u16,
        nsr_verdict: u8,
    ) -> Self {
        let commit = commit_inputs(
            cycle_id,
            influence_pulses_root,
            &influence_node_values,
            coherence_hint,
            risk,
            drift,
            nsr_verdict,
        );
        Self {
            cycle_id,
            influence_pulses_root,
            influence_node_values,
            coherence_hint,
            risk,
            drift,
            nsr_verdict,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnnOutputs {
    pub cycle_id: u64,
    pub states_commit: Digest32,
    pub global_plv: u16,
    pub pair_locks: Vec<PairLock>,
    pub phase_frame_commit: Digest32,
    pub commit: Digest32,
}

impl OnnOutputs {
    fn new(
        cycle_id: u64,
        states_commit: Digest32,
        global_plv: u16,
        pair_locks: Vec<PairLock>,
    ) -> Self {
        let phase_frame_commit =
            commit_phase_frame(cycle_id, states_commit, global_plv, &pair_locks);
        let commit = commit_outputs(
            cycle_id,
            states_commit,
            global_plv,
            &pair_locks,
            phase_frame_commit,
        );
        Self {
            cycle_id,
            states_commit,
            global_plv,
            pair_locks,
            phase_frame_commit,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhaseFrame {
    pub cycle_id: u64,
    pub global_phase: u16,
    pub module_phase: Vec<(OscId, u16)>,
    pub coherence_plv: u16,
    pub pair_locks: Vec<PairLock>,
    pub states_commit: Digest32,
    pub phase_frame_commit: Digest32,
    pub commit: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnnCore {
    pub params: OnnParams,
    pub states: Vec<OscState>,
    pub commit: Digest32,
}

impl OnnCore {
    pub fn new(params: OnnParams) -> Self {
        let states = seed_states(&params);
        let commit = commit_core(params.commit, &states);
        Self {
            params,
            states,
            commit,
        }
    }

    pub fn phase_of(&self, id: OscId) -> Option<u16> {
        self.states
            .iter()
            .find_map(|state| (state.id == id).then_some(state.phase))
    }

    pub fn tick(&mut self, inp: &OnnInputs) -> OnnOutputs {
        let effective_coupling = effective_coupling(&self.params, inp);
        let drift_bias = i32::from(inp.drift) / 64;
        let base_step = i32::from(self.params.base_step).saturating_add(drift_bias);
        let max_delta = i32::from(self.params.max_delta);
        let states_snapshot = self.states.clone();

        for state in &mut self.states {
            let coupling_effect = coupling_effect(state, &states_snapshot, effective_coupling)
                .clamp(-max_delta, max_delta);
            let next_phase =
                i32::from(state.phase) + base_step + i32::from(state.omega) + coupling_effect;
            state.phase = wrap_phase(next_phase);
            state.amp = update_amp(state.amp, inp, effective_coupling);
            state.update_commit();
        }

        let states_commit = commit_states(&self.states);
        let pair_locks = pair_locks(&self.states, self.params.lock_window);
        let global_plv = average_pair_lock(&pair_locks);
        let outputs = OnnOutputs::new(inp.cycle_id, states_commit, global_plv, pair_locks);
        self.commit = commit_core(self.params.commit, &self.states);
        outputs
    }
}

impl Default for OnnCore {
    fn default() -> Self {
        Self::new(OnnParams::default())
    }
}

fn seed_states(params: &OnnParams) -> Vec<OscState> {
    let defaults = default_oscillators();
    defaults
        .into_iter()
        .take(MAX_OSCILLATORS)
        .map(|id| seed_state(id, params.commit))
        .collect()
}

fn default_oscillators() -> Vec<OscId> {
    vec![
        OscId::Jepa,
        OscId::Ssm,
        OscId::Ncde,
        OscId::Cde,
        OscId::Nsr,
        OscId::Geist,
        OscId::Iit,
        OscId::Output,
        OscId::BlueBrain,
    ]
}

fn seed_state(id: OscId, params_commit: Digest32) -> OscState {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.onn.seed.state.v2");
    hasher.update(&id.as_u16().to_be_bytes());
    hasher.update(params_commit.as_bytes());
    let hash = hasher.finalize();
    let bytes = hash.as_bytes();
    let phase = u16::from_be_bytes([bytes[0], bytes[1]]);
    let omega_seed = i16::from_be_bytes([bytes[2], bytes[3]]);
    let omega = (omega_seed % 64) - 32;
    let amp_seed = u16::from_be_bytes([bytes[4], bytes[5]]);
    let amp = 6000u16.saturating_add(amp_seed % 3000).min(MAX_SIGNAL);
    OscState::new(id, phase, omega, amp)
}

fn update_amp(current: u16, inp: &OnnInputs, effective_coupling: i32) -> u16 {
    let coherence = inp.coherence_hint.min(MAX_SIGNAL) as i32;
    let mut target = current as i32 + (coherence - current as i32) / 16;
    let risk = i32::from(inp.risk.min(MAX_SIGNAL));
    if risk > 0 {
        target = target.saturating_sub(risk / 20);
    }
    if inp.nsr_verdict >= 2 {
        target = target.saturating_sub(800);
    }
    if effective_coupling > 0 {
        target = target.saturating_add(effective_coupling / 12);
    }
    target.clamp(0, i32::from(MAX_SIGNAL)) as u16
}

fn effective_coupling(params: &OnnParams, inp: &OnnInputs) -> i32 {
    let mut coupling = i32::from(params.coupling);
    let coherence = influence_value(inp, InfluenceNodeId::Coherence).max(0) as i32;
    let integration = influence_value(inp, InfluenceNodeId::Integration).max(0) as i32;
    let coherence_boost = (coherence + integration) / 20;
    coupling = coupling.saturating_add(coherence_boost);

    let risk = i32::from(inp.risk.min(MAX_SIGNAL));
    let risk_influence = influence_value(inp, InfluenceNodeId::Risk).max(0) as i32;
    let risk_penalty = risk.max(risk_influence) / 18;
    coupling = coupling.saturating_sub(risk_penalty);
    if inp.nsr_verdict >= 2 {
        coupling = coupling.saturating_sub(1500);
    }
    coupling.clamp(0, i32::from(MAX_SIGNAL))
}

fn influence_value(inp: &OnnInputs, node: InfluenceNodeId) -> i16 {
    inp.influence_node_values
        .iter()
        .find_map(|(id, value)| (*id == node).then_some(*value))
        .unwrap_or(0)
}

fn coupling_effect(state: &OscState, states: &[OscState], effective_coupling: i32) -> i32 {
    if effective_coupling == 0 || states.is_empty() {
        return 0;
    }
    let mut sum: i64 = 0;
    for other in states {
        if other.id == state.id {
            continue;
        }
        let diff = signed_phase_delta(state.phase, other.phase);
        let sin_val = sin_lookup(diff) as i64;
        let amp = i64::from(other.amp);
        let contribution = (i64::from(effective_coupling) * sin_val * amp)
            / i64::from(SIN_LUT_SCALE)
            / i64::from(MAX_SIGNAL);
        sum = sum.saturating_add(contribution);
    }
    let count = states.len().max(1) as i64;
    (sum / count) as i32
}

fn signed_phase_delta(theta: u16, other: u16) -> i32 {
    let diff = other.wrapping_sub(theta) as i32;
    if diff > 32_768 {
        diff - PHASE_WRAP
    } else {
        diff
    }
}

fn sin_lookup(diff: i32) -> i32 {
    let shifted = diff + 32_768;
    let idx = ((shifted as i64 * SIN_LUT.len() as i64) / PHASE_WRAP as i64) as usize;
    SIN_LUT[idx.min(SIN_LUT.len() - 1)] as i32
}

fn wrap_phase(value: i32) -> u16 {
    value.rem_euclid(PHASE_WRAP) as u16
}

fn pair_locks(states: &[OscState], lock_window: u16) -> Vec<PairLock> {
    let pairs = [
        (OscId::Jepa, OscId::Nsr),
        (OscId::Cde, OscId::Nsr),
        (OscId::Geist, OscId::Iit),
        (OscId::Ssm, OscId::Ncde),
        (OscId::Output, OscId::Nsr),
    ];
    let mut locks = Vec::with_capacity(pairs.len());
    for (a, b) in pairs {
        let Some(phase_a) = lookup_phase(states, a) else {
            continue;
        };
        let Some(phase_b) = lookup_phase(states, b) else {
            continue;
        };
        let diff = abs_phase_delta(phase_a, phase_b);
        let lock = lock_from_diff(diff, lock_window);
        locks.push(PairLock::new(a, b, lock, diff));
    }
    locks
}

fn lookup_phase(states: &[OscState], id: OscId) -> Option<u16> {
    states
        .iter()
        .find_map(|state| (state.id == id).then_some(state.phase))
}

fn abs_phase_delta(a: u16, b: u16) -> u16 {
    let diff = u32::from(a.abs_diff(b));
    let wrap = PHASE_WRAP as u32 - diff;
    diff.min(wrap).min(32_768) as u16
}

fn lock_from_diff(diff: u16, lock_window: u16) -> u16 {
    if lock_window == 0 {
        return 0;
    }
    let quarter = lock_window / 4;
    if diff <= quarter {
        return MAX_SIGNAL;
    }
    if diff >= lock_window {
        return 0;
    }
    let range = lock_window.saturating_sub(quarter).max(1);
    let offset = diff.saturating_sub(quarter);
    let reduction = (u32::from(offset) * u32::from(MAX_SIGNAL)) / u32::from(range);
    MAX_SIGNAL.saturating_sub(reduction as u16)
}

fn average_pair_lock(pairs: &[PairLock]) -> u16 {
    if pairs.is_empty() {
        return 0;
    }
    let sum: u32 = pairs.iter().map(|pair| u32::from(pair.lock)).sum();
    let avg = sum / pairs.len() as u32;
    avg.min(u32::from(MAX_SIGNAL)) as u16
}

fn commit_params(base_step: u16, coupling: u16, max_delta: u16, lock_window: u16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.onn.params.v2");
    hasher.update(&base_step.to_be_bytes());
    hasher.update(&coupling.to_be_bytes());
    hasher.update(&max_delta.to_be_bytes());
    hasher.update(&lock_window.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_osc_state(id: OscId, phase: u16, omega: i16, amp: u16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.onn.state.v2");
    hasher.update(&id.as_u16().to_be_bytes());
    hasher.update(&phase.to_be_bytes());
    hasher.update(&omega.to_be_bytes());
    hasher.update(&amp.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_states(states: &[OscState]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.onn.states.v2");
    hasher.update(
        &u32::try_from(states.len())
            .unwrap_or(u32::MAX)
            .to_be_bytes(),
    );
    for state in states {
        hasher.update(state.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_pair_lock(a: OscId, b: OscId, lock: u16, diff: u16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.onn.lock.v2");
    hasher.update(&a.as_u16().to_be_bytes());
    hasher.update(&b.as_u16().to_be_bytes());
    hasher.update(&lock.to_be_bytes());
    hasher.update(&diff.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_phase_frame(
    cycle_id: u64,
    states_commit: Digest32,
    global_plv: u16,
    pair_locks: &[PairLock],
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.onn.phase_frame.v2");
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(states_commit.as_bytes());
    hasher.update(&global_plv.to_be_bytes());
    for pair in pair_locks {
        hasher.update(pair.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_outputs(
    cycle_id: u64,
    states_commit: Digest32,
    global_plv: u16,
    pair_locks: &[PairLock],
    phase_frame_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.onn.outputs.v2");
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(states_commit.as_bytes());
    hasher.update(&global_plv.to_be_bytes());
    for pair in pair_locks {
        hasher.update(pair.commit.as_bytes());
    }
    hasher.update(phase_frame_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_inputs(
    cycle_id: u64,
    influence_pulses_root: Digest32,
    influence_node_values: &[(InfluenceNodeId, i16)],
    coherence_hint: u16,
    risk: u16,
    drift: u16,
    nsr_verdict: u8,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.onn.inputs.v2");
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(influence_pulses_root.as_bytes());
    hasher.update(
        &u32::try_from(influence_node_values.len())
            .unwrap_or(u32::MAX)
            .to_be_bytes(),
    );
    for (node, value) in influence_node_values {
        hasher.update(&node.to_u16().to_be_bytes());
        hasher.update(&value.to_be_bytes());
    }
    hasher.update(&coherence_hint.to_be_bytes());
    hasher.update(&risk.to_be_bytes());
    hasher.update(&drift.to_be_bytes());
    hasher.update(&[nsr_verdict]);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_core(params_commit: Digest32, states: &[OscState]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.onn.core.v2");
    hasher.update(params_commit.as_bytes());
    hasher.update(
        &u32::try_from(states.len())
            .unwrap_or(u32::MAX)
            .to_be_bytes(),
    );
    for state in states {
        hasher.update(state.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_inputs(coherence: u16, risk: u16) -> OnnInputs {
        OnnInputs::new(
            1,
            Digest32::new([1u8; 32]),
            vec![
                (InfluenceNodeId::Coherence, coherence as i16),
                (InfluenceNodeId::Integration, coherence as i16),
                (InfluenceNodeId::Risk, risk as i16),
            ],
            coherence,
            risk,
            0,
            0,
        )
    }

    #[test]
    fn tick_is_deterministic_for_same_inputs() {
        let mut core_a = OnnCore::default();
        let mut core_b = OnnCore::default();
        let inputs = make_inputs(5000, 0);

        let out_a = core_a.tick(&inputs);
        let out_b = core_b.tick(&inputs);

        assert_eq!(out_a, out_b);
        assert_eq!(out_a.commit, out_b.commit);
    }

    #[test]
    fn lock_high_when_phases_match() {
        let params = OnnParams::new(0, 0, 0, 8000);
        let mut core = OnnCore::new(params);
        for state in &mut core.states {
            state.phase = 1000;
            state.omega = 0;
            state.amp = 8000;
            state.update_commit();
        }
        let inputs = make_inputs(8000, 0);
        let outputs = core.tick(&inputs);
        let pair = outputs
            .pair_locks
            .iter()
            .find(|pair| pair.a == OscId::Cde && pair.b == OscId::Nsr)
            .expect("pair lock present");
        assert!(pair.lock >= 9000);
    }

    #[test]
    fn lock_low_when_phases_diverge() {
        let params = OnnParams::new(0, 0, 0, 8000);
        let mut core = OnnCore::new(params);
        for state in &mut core.states {
            state.phase = 1000;
            state.omega = 0;
            state.amp = 8000;
            state.update_commit();
        }
        if let Some(state) = core.states.iter_mut().find(|s| s.id == OscId::Nsr) {
            state.phase = 40_000;
            state.update_commit();
        }
        let inputs = make_inputs(8000, 0);
        let outputs = core.tick(&inputs);
        let pair = outputs
            .pair_locks
            .iter()
            .find(|pair| pair.a == OscId::Cde && pair.b == OscId::Nsr)
            .expect("pair lock present");
        assert!(pair.lock <= 1000);
    }

    #[test]
    fn influence_modulates_global_plv() {
        let params = OnnParams::new(0, 3200, 512, 12_000);
        let mut core_high = OnnCore::new(params.clone());
        let mut core_low = OnnCore::new(params);
        for state in &mut core_high.states {
            state.phase = state.phase.wrapping_add(12_000);
            state.omega = 0;
            state.update_commit();
        }
        core_low.states = core_high.states.clone();
        let inputs_high = make_inputs(9000, 0);
        let inputs_low = make_inputs(0, 9000);
        let mut plv_high = 0;
        let mut plv_low = 0;
        for _ in 0..4 {
            plv_high = core_high.tick(&inputs_high).global_plv;
            plv_low = core_low.tick(&inputs_low).global_plv;
        }
        assert!(plv_high > plv_low);
    }
}
