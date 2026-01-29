#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_types::Digest32;

const MAX_OSCILLATORS: usize = 16;
const PHASE_Q12_WRAP: i32 = 4096;
const BUCKETS: u8 = 16;
const SIN_LUT: [i16; 64] = [
    0, 201, 400, 595, 784, 965, 1138, 1299, 1448, 1583, 1703, 1806, 1892, 1960, 2009, 2038, 2048,
    2038, 2009, 1960, 1892, 1806, 1703, 1583, 1448, 1299, 1138, 965, 784, 595, 400, 201, 0, -201,
    -400, -595, -784, -965, -1138, -1299, -1448, -1583, -1703, -1806, -1892, -1960, -2009, -2038,
    -2048, -2038, -2009, -1960, -1892, -1806, -1703, -1583, -1448, -1299, -1138, -965, -784, -595,
    -400, -201,
];

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OscId {
    Ssm = 0,
    Ncde = 1,
    Cde = 2,
    Nsr = 3,
    Tcf = 4,
    Iit = 5,
    Output = 6,
    Reserved7 = 7,
    Reserved8 = 8,
    Reserved9 = 9,
    Other(u8),
}

impl OscId {
    pub fn as_u8(self) -> u8 {
        match self {
            Self::Ssm => 0,
            Self::Ncde => 1,
            Self::Cde => 2,
            Self::Nsr => 3,
            Self::Tcf => 4,
            Self::Iit => 5,
            Self::Output => 6,
            Self::Reserved7 => 7,
            Self::Reserved8 => 8,
            Self::Reserved9 => 9,
            Self::Other(value) => value,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OnnParams {
    pub n: usize,
    pub omega_q12: [i16; 16],
    pub k_couple: u16,
    pub k_dither: u16,
    pub buckets: u8,
    pub couple_clamp_q12: i16,
    pub commit: Digest32,
}

impl OnnParams {
    pub fn new(
        n: usize,
        omega_q12: [i16; 16],
        k_couple: u16,
        k_dither: u16,
        buckets: u8,
        couple_clamp_q12: i16,
    ) -> Self {
        let n = n.clamp(1, MAX_OSCILLATORS);
        let buckets = if buckets == BUCKETS { buckets } else { BUCKETS };
        let k_couple = k_couple.min(10_000);
        let k_dither = k_dither.min(10_000);
        let commit = commit_params(n, &omega_q12, k_couple, k_dither, buckets, couple_clamp_q12);
        Self {
            n,
            omega_q12,
            k_couple,
            k_dither,
            buckets,
            couple_clamp_q12,
            commit,
        }
    }
}

impl Default for OnnParams {
    fn default() -> Self {
        let mut omega_q12 = [0i16; 16];
        let base = [34i16, 28, 30, 26, 22, 24, 20, 18, 16, 14, 0, 0, 0, 0, 0, 0];
        omega_q12.copy_from_slice(&base);
        Self::new(10, omega_q12, 3200, 600, BUCKETS, 512)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OnnState {
    pub theta_q12: [u16; 16],
    pub last_bucket: u8,
    pub last_plv: u16,
    pub commit: Digest32,
}

impl OnnState {
    pub fn new(params: &OnnParams) -> Self {
        let theta_q12 = seed_theta(params);
        let last_bucket = 0;
        let last_plv = 0;
        let commit = commit_state(&theta_q12, last_bucket, last_plv, params.commit);
        Self {
            theta_q12,
            last_bucket,
            last_plv,
            commit,
        }
    }

    pub fn update_commit(&mut self, params_commit: Digest32) {
        self.commit = commit_state(
            &self.theta_q12,
            self.last_bucket,
            self.last_plv,
            params_commit,
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OnnInputs {
    pub cycle_id: u64,
    pub ssm_state_commit: Digest32,
    pub ncde_state_digest: Digest32,
    pub cde_commit: Digest32,
    pub nsr_trace_root: Digest32,
    pub iit_hints_commit: Digest32,
    pub lock_window_buckets: u8,
    pub risk: u16,
    pub drift: u16,
    pub surprise: u16,
    pub commit: Digest32,
}

impl OnnInputs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cycle_id: u64,
        ssm_state_commit: Digest32,
        ncde_state_digest: Digest32,
        cde_commit: Digest32,
        nsr_trace_root: Digest32,
        iit_hints_commit: Digest32,
        lock_window_buckets: u8,
        risk: u16,
        drift: u16,
        surprise: u16,
    ) -> Self {
        let lock_window_buckets = lock_window_buckets.clamp(1, 4);
        let commit = commit_inputs(
            cycle_id,
            ssm_state_commit,
            ncde_state_digest,
            cde_commit,
            nsr_trace_root,
            iit_hints_commit,
            lock_window_buckets,
            risk,
            drift,
            surprise,
        );
        Self {
            cycle_id,
            ssm_state_commit,
            ncde_state_digest,
            cde_commit,
            nsr_trace_root,
            iit_hints_commit,
            lock_window_buckets,
            risk,
            drift,
            surprise,
            commit,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhaseBus {
    pub cycle_id: u64,
    pub gamma_bucket: u8,
    pub global_plv: u16,
    pub osc_buckets: [u8; 16],
    pub phase_commit: Digest32,
    pub commit: Digest32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhaseLockDecision {
    pub cycle_id: u64,
    pub lock_window_buckets: u8,
    pub accept_center: u8,
    pub commit: Digest32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OnnOutputs {
    pub phase_bus: PhaseBus,
    pub lock: PhaseLockDecision,
    pub commit: Digest32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OnnCore {
    pub params: OnnParams,
    pub state: OnnState,
    pub commit: Digest32,
}

impl OnnCore {
    pub fn new(params: OnnParams) -> Self {
        let state = OnnState::new(&params);
        let commit = commit_core(params.commit, state.commit);
        Self {
            params,
            state,
            commit,
        }
    }

    pub fn tick(&mut self, inp: &OnnInputs) -> OnnOutputs {
        let n = self.params.n.clamp(1, MAX_OSCILLATORS);
        let effective_k =
            adjusted_coupling(self.params.k_couple, inp.risk, inp.drift, inp.surprise);
        let mut next_theta = self.state.theta_q12;

        for (i, theta_next) in next_theta.iter_mut().enumerate().take(n) {
            let omega = i32::from(self.params.omega_q12[i]);
            let sum = coupling_sum_q12(i, n, &self.state.theta_q12);
            let mut couple_q12 = (i32::from(effective_k) * sum) / 10_000;
            couple_q12 = couple_q12.clamp(
                -i32::from(self.params.couple_clamp_q12),
                i32::from(self.params.couple_clamp_q12),
            );
            let dither = dither_for(i, inp, self.params.k_dither);
            let theta = i32::from(self.state.theta_q12[i]);
            let updated = theta + omega + couple_q12 + dither;
            *theta_next = wrap_q12(updated);
        }

        self.state.theta_q12 = next_theta;
        let osc_buckets = buckets_for(&self.state.theta_q12, n);
        let gamma_bucket = median_bucket(&osc_buckets, n);
        let global_plv = plv_from_buckets(&osc_buckets, n, gamma_bucket);
        self.state.last_bucket = gamma_bucket;
        self.state.last_plv = global_plv;
        self.state.update_commit(self.params.commit);

        let phase_commit = commit_phase_bus(
            inp.cycle_id,
            gamma_bucket,
            global_plv,
            &osc_buckets,
            n,
            inp.commit,
            self.params.commit,
        );
        let phase_bus_commit = commit_phase_bus_root(phase_commit);
        let phase_bus = PhaseBus {
            cycle_id: inp.cycle_id,
            gamma_bucket,
            global_plv,
            osc_buckets,
            phase_commit,
            commit: phase_bus_commit,
        };
        let lock = PhaseLockDecision {
            cycle_id: inp.cycle_id,
            lock_window_buckets: inp.lock_window_buckets.clamp(1, 4),
            accept_center: gamma_bucket,
            commit: commit_lock_decision(
                inp.cycle_id,
                inp.lock_window_buckets,
                gamma_bucket,
                phase_commit,
            ),
        };
        let commit = commit_outputs(phase_bus.commit, lock.commit);
        self.commit = commit_core(self.params.commit, self.state.commit);
        OnnOutputs {
            phase_bus,
            lock,
            commit,
        }
    }

    pub fn phase_bus(&self, cycle_id: u64, inputs_commit: Digest32) -> PhaseBus {
        let n = self.params.n.clamp(1, MAX_OSCILLATORS);
        let osc_buckets = buckets_for(&self.state.theta_q12, n);
        let gamma_bucket = self.state.last_bucket;
        let global_plv = self.state.last_plv;
        let phase_commit = commit_phase_bus(
            cycle_id,
            gamma_bucket,
            global_plv,
            &osc_buckets,
            n,
            inputs_commit,
            self.params.commit,
        );
        let commit = commit_phase_bus_root(phase_commit);
        PhaseBus {
            cycle_id,
            gamma_bucket,
            global_plv,
            osc_buckets,
            phase_commit,
            commit,
        }
    }
}

impl Default for OnnCore {
    fn default() -> Self {
        Self::new(OnnParams::default())
    }
}

pub fn accept_spike(lock: &PhaseLockDecision, spike_bucket: u8) -> bool {
    let radius = lock.lock_window_buckets.clamp(1, 4);
    let distance = circular_bucket_distance(lock.accept_center, spike_bucket);
    distance <= radius
}

pub fn apply_coupling_delta(params: &OnnParams, delta: i16) -> OnnParams {
    let k_couple = apply_i16_delta(params.k_couple, delta, 0, 10_000);
    OnnParams::new(
        params.n,
        params.omega_q12,
        k_couple,
        params.k_dither,
        params.buckets,
        params.couple_clamp_q12,
    )
}

pub fn apply_lock_window_delta(params: &OnnParams, delta: i16) -> OnnParams {
    let k_dither = apply_i16_delta(params.k_dither, delta, 0, 10_000);
    OnnParams::new(
        params.n,
        params.omega_q12,
        params.k_couple,
        k_dither,
        params.buckets,
        params.couple_clamp_q12,
    )
}

fn apply_i16_delta(value: u16, delta: i16, min: u16, max: u16) -> u16 {
    let value = i32::from(value);
    let delta = i32::from(delta);
    let updated = value
        .saturating_add(delta)
        .clamp(i32::from(min), i32::from(max));
    updated as u16
}

fn seed_theta(params: &OnnParams) -> [u16; 16] {
    let mut theta = [0u16; 16];
    for (i, theta_value) in theta
        .iter_mut()
        .enumerate()
        .take(params.n.min(MAX_OSCILLATORS))
    {
        let mut hasher = Hasher::new();
        hasher.update(b"ucf.onn.seed.theta.v1");
        hasher.update(&[i as u8]);
        hasher.update(params.commit.as_bytes());
        let hash = hasher.finalize();
        let bytes = hash.as_bytes();
        let raw = u16::from_be_bytes([bytes[0], bytes[1]]);
        *theta_value = raw & 0x0fff;
    }
    theta
}

fn adjusted_coupling(k_couple: u16, risk: u16, drift: u16, surprise: u16) -> u16 {
    let mut effective = i32::from(k_couple);
    let boost = (i32::from(risk) + i32::from(drift)) / 64;
    let reduction = i32::from(surprise) / 64;
    effective = effective.saturating_add(boost).saturating_sub(reduction);
    effective.clamp(0, 10_000) as u16
}

fn coupling_sum_q12(i: usize, n: usize, theta_q12: &[u16; 16]) -> i32 {
    let mut sum = 0i32;
    let theta_i = theta_q12[i];
    for theta_j in theta_q12.iter().take(n) {
        let delta = (theta_j.wrapping_sub(theta_i)) & 0x0fff;
        let idx = (delta >> 6) as usize;
        let sin_val = i32::from(SIN_LUT[idx]);
        sum = sum.saturating_add(sin_val);
    }
    sum / n.max(1) as i32
}

fn dither_for(index: usize, inp: &OnnInputs, k_dither: u16) -> i32 {
    if k_dither == 0 {
        return 0;
    }
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.onn.dither.v1");
    hasher.update(&[index as u8]);
    hasher.update(inp.ssm_state_commit.as_bytes());
    hasher.update(inp.ncde_state_digest.as_bytes());
    hasher.update(inp.cde_commit.as_bytes());
    hasher.update(inp.nsr_trace_root.as_bytes());
    hasher.update(inp.iit_hints_commit.as_bytes());
    let hash = hasher.finalize();
    let bytes = hash.as_bytes();
    let value = u16::from_be_bytes([bytes[0], bytes[1]]);
    let sign = if value & 1 == 0 { 1i32 } else { -1i32 };
    let small_mag = ((value >> 1) % 64) as i32;
    sign * (i32::from(k_dither) * small_mag) / 10_000
}

fn wrap_q12(value: i32) -> u16 {
    value.rem_euclid(PHASE_Q12_WRAP) as u16
}

fn buckets_for(theta_q12: &[u16; 16], n: usize) -> [u8; 16] {
    let mut buckets = [0u8; 16];
    for (i, bucket) in buckets.iter_mut().enumerate().take(n.min(MAX_OSCILLATORS)) {
        *bucket = ((u32::from(theta_q12[i]) * u32::from(BUCKETS)) >> 12) as u8;
    }
    buckets
}

fn median_bucket(osc_buckets: &[u8; 16], n: usize) -> u8 {
    if n == 0 {
        return 0;
    }
    let mut values = osc_buckets[..n.min(MAX_OSCILLATORS)].to_vec();
    values.sort_unstable();
    values[(values.len() - 1) / 2]
}

fn plv_from_buckets(osc_buckets: &[u8; 16], n: usize, gamma_bucket: u8) -> u16 {
    if n == 0 {
        return 0;
    }
    let mut sum: u32 = 0;
    for bucket in osc_buckets.iter().take(n.min(MAX_OSCILLATORS)) {
        sum = sum.saturating_add(u32::from(circular_bucket_distance(*bucket, gamma_bucket)));
    }
    let avg = (sum * 10_000) / (n as u32 * 8);
    10_000u16.saturating_sub(avg.min(10_000) as u16)
}

fn circular_bucket_distance(a: u8, b: u8) -> u8 {
    let diff = a.abs_diff(b);
    let wrap = BUCKETS.saturating_sub(diff);
    diff.min(wrap)
}

fn commit_params(
    n: usize,
    omega_q12: &[i16; 16],
    k_couple: u16,
    k_dither: u16,
    buckets: u8,
    couple_clamp_q12: i16,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.onn.params.v3");
    hasher.update(&u32::try_from(n).unwrap_or(u32::MAX).to_be_bytes());
    for omega in omega_q12 {
        hasher.update(&omega.to_be_bytes());
    }
    hasher.update(&k_couple.to_be_bytes());
    hasher.update(&k_dither.to_be_bytes());
    hasher.update(&[buckets]);
    hasher.update(&couple_clamp_q12.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_state(
    theta_q12: &[u16; 16],
    last_bucket: u8,
    last_plv: u16,
    params_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.onn.state.v1");
    hasher.update(params_commit.as_bytes());
    for theta in theta_q12 {
        hasher.update(&theta.to_be_bytes());
    }
    hasher.update(&[last_bucket]);
    hasher.update(&last_plv.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[allow(clippy::too_many_arguments)]
fn commit_inputs(
    cycle_id: u64,
    ssm_state_commit: Digest32,
    ncde_state_digest: Digest32,
    cde_commit: Digest32,
    nsr_trace_root: Digest32,
    iit_hints_commit: Digest32,
    lock_window_buckets: u8,
    risk: u16,
    drift: u16,
    surprise: u16,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.onn.inputs.v1");
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(ssm_state_commit.as_bytes());
    hasher.update(ncde_state_digest.as_bytes());
    hasher.update(cde_commit.as_bytes());
    hasher.update(nsr_trace_root.as_bytes());
    hasher.update(iit_hints_commit.as_bytes());
    hasher.update(&[lock_window_buckets]);
    hasher.update(&risk.to_be_bytes());
    hasher.update(&drift.to_be_bytes());
    hasher.update(&surprise.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_phase_bus(
    cycle_id: u64,
    gamma_bucket: u8,
    global_plv: u16,
    osc_buckets: &[u8; 16],
    n: usize,
    inputs_commit: Digest32,
    params_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.onn.phase_bus.v1");
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(&[gamma_bucket]);
    hasher.update(&global_plv.to_be_bytes());
    for bucket in osc_buckets.iter().take(n.min(MAX_OSCILLATORS)) {
        hasher.update(&[*bucket]);
    }
    hasher.update(inputs_commit.as_bytes());
    hasher.update(params_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_phase_bus_root(phase_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.onn.phase_bus.root.v1");
    hasher.update(phase_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_lock_decision(
    cycle_id: u64,
    lock_window_buckets: u8,
    accept_center: u8,
    phase_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.onn.lock.v1");
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(&[lock_window_buckets]);
    hasher.update(&[accept_center]);
    hasher.update(phase_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_outputs(phase_bus_commit: Digest32, lock_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.onn.outputs.v1");
    hasher.update(phase_bus_commit.as_bytes());
    hasher.update(lock_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_core(params_commit: Digest32, state_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.onn.core.v1");
    hasher.update(params_commit.as_bytes());
    hasher.update(state_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_inputs() -> OnnInputs {
        OnnInputs::new(
            1,
            Digest32::new([1u8; 32]),
            Digest32::new([2u8; 32]),
            Digest32::new([3u8; 32]),
            Digest32::new([4u8; 32]),
            Digest32::new([5u8; 32]),
            2,
            1200,
            900,
            500,
        )
    }

    #[test]
    fn tick_is_deterministic_for_same_inputs() {
        let mut core_a = OnnCore::default();
        let mut core_b = OnnCore::default();
        let inputs = make_inputs();

        let out_a = core_a.tick(&inputs);
        let out_b = core_b.tick(&inputs);

        assert_eq!(out_a.phase_bus.gamma_bucket, out_b.phase_bus.gamma_bucket);
        assert_eq!(out_a.phase_bus.global_plv, out_b.phase_bus.global_plv);
        assert_eq!(out_a.phase_bus.phase_commit, out_b.phase_bus.phase_commit);
    }

    #[test]
    fn coupling_increases_plv_over_ticks() {
        let mut params = OnnParams::default();
        params.k_couple = 9000;
        params.k_dither = 0;
        params.commit = commit_params(
            params.n,
            &params.omega_q12,
            params.k_couple,
            params.k_dither,
            params.buckets,
            params.couple_clamp_q12,
        );
        let mut core = OnnCore::new(params);
        let inputs = make_inputs();
        let mut plv = 0;
        for _ in 0..4 {
            plv = core.tick(&inputs).phase_bus.global_plv;
        }
        assert!(plv >= core.state.last_plv);
    }

    #[test]
    fn accept_spike_within_window() {
        let lock = PhaseLockDecision {
            cycle_id: 1,
            lock_window_buckets: 2,
            accept_center: 4,
            commit: Digest32::new([9u8; 32]),
        };
        assert!(accept_spike(&lock, 5));
        assert!(accept_spike(&lock, 6));
        assert!(!accept_spike(&lock, 8));
    }

    #[test]
    fn median_bucket_is_deterministic() {
        let mut buckets = [0u8; 16];
        buckets[0] = 1;
        buckets[1] = 5;
        buckets[2] = 4;
        buckets[3] = 2;
        let median = median_bucket(&buckets, 4);
        assert_eq!(median, 2);
    }
}
