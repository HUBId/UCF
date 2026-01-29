#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_types::Digest32;

const DOMAIN_INPUT: &[u8] = b"ucf.iit.inputs.v1";
const DOMAIN_HINTS: &[u8] = b"ucf.iit.hints.v1";
const DOMAIN_OUTPUT: &[u8] = b"ucf.iit.output.v1";
const DOMAIN_PARAMS: &[u8] = b"ucf.iit.params.v1";
const DOMAIN_STATE: &[u8] = b"ucf.iit.state.v1";
const DOMAIN_CORE: &[u8] = b"ucf.iit.core.v1";

const MAX_SCORE: u16 = 10_000;
const PLV_LOW: u16 = 3_500;
const RISK_HIGH: u16 = 7_000;

const DEP_BELL_BUCKET_SIZE: u8 = 4;
const DEP_BELL_TABLE: [u16; 33] = [
    0, 400, 900, 1500, 2300, 3200, 4400, 6000, 7600, 9000, 10000, 10000, 9800, 9200, 8400, 7200,
    6000, 4700, 3500, 2500, 1700, 1100, 700, 450, 280, 170, 110, 70, 45, 25, 12, 5, 0,
];

const STABILITY_TABLE: [u16; 33] = [
    10_000, 9_800, 9_500, 9_100, 8_600, 8_000, 7_200, 6_300, 5_400, 4_600, 3_900, 3_300, 2_800,
    2_300, 1_900, 1_600, 1_300, 1_100, 900, 750, 620, 520, 420, 340, 270, 210, 160, 120, 90, 60,
    40, 20, 0,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ModuleId {
    SsmState,
    NcdeState,
    Cde,
    NsrTrace,
    Coupling,
}

impl ModuleId {
    fn as_u8(self) -> u8 {
        match self {
            Self::SsmState => 1,
            Self::NcdeState => 2,
            Self::Cde => 3,
            Self::NsrTrace => 4,
            Self::Coupling => 5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IitParams {
    /// weights 0..10000
    pub w_plv: u16,
    pub w_dep: u16,
    pub w_stability: u16,
    /// thresholds
    pub phi_low: u16,
    pub phi_high: u16,
    pub drift_hi: u16,
    pub surprise_hi: u16,
    /// lag window sizes 1..8
    pub dep_lag: u8,
    pub stab_lag: u8,
    pub commit: Digest32,
}

impl Default for IitParams {
    fn default() -> Self {
        let mut params = Self {
            w_plv: 4_000,
            w_dep: 3_500,
            w_stability: 2_500,
            phi_low: 3_200,
            phi_high: 7_200,
            drift_hi: 7_000,
            surprise_hi: 7_000,
            dep_lag: 2,
            stab_lag: 3,
            commit: Digest32::new([0u8; 32]),
        };
        params.commit = commit_params(&params);
        params
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IitState {
    pub last_phi: u16,
    /// lag buffers for key commits (truncate to 16 bytes each)
    pub last_commits: Vec<(ModuleId, [u8; 16])>,
    pub commit: Digest32,
}

impl Default for IitState {
    fn default() -> Self {
        let mut state = Self {
            last_phi: 0,
            last_commits: Vec::new(),
            commit: Digest32::new([0u8; 32]),
        };
        state.commit = commit_state(&state);
        state
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IitInputs {
    pub cycle_id: u64,
    pub phase_bus_commit: Digest32,
    pub gamma_bucket: u8,
    pub global_plv: u16,
    pub ssm_state_commit: Digest32,
    pub ncde_state_digest: Digest32,
    pub cde_commit: Digest32,
    pub nsr_trace_root: Digest32,
    pub coupling_influences_root: Digest32,
    pub risk: u16,
    pub drift: u16,
    pub surprise: u16,
    pub commit: Digest32,
}

impl IitInputs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cycle_id: u64,
        phase_bus_commit: Digest32,
        gamma_bucket: u8,
        global_plv: u16,
        ssm_state_commit: Digest32,
        ncde_state_digest: Digest32,
        cde_commit: Digest32,
        nsr_trace_root: Digest32,
        coupling_influences_root: Digest32,
        risk: u16,
        drift: u16,
        surprise: u16,
    ) -> Self {
        let mut inputs = Self {
            cycle_id,
            phase_bus_commit,
            gamma_bucket,
            global_plv,
            ssm_state_commit,
            ncde_state_digest,
            cde_commit,
            nsr_trace_root,
            coupling_influences_root,
            risk,
            drift,
            surprise,
            commit: Digest32::new([0u8; 32]),
        };
        inputs.commit = commit_inputs(&inputs);
        inputs
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IitOutputs {
    pub cycle_id: u64,
    pub phi_proxy: u16,
    pub tighten_sync: bool,
    pub damp_output: bool,
    pub damp_learning: bool,
    pub request_replay: bool,
    pub hints_commit: Digest32,
    pub commit: Digest32,
}

pub type IitOutput = IitOutputs;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IitCore {
    pub params: IitParams,
    pub state: IitState,
    pub commit: Digest32,
}

impl IitCore {
    pub fn new(params: IitParams) -> Self {
        let state = IitState::default();
        let commit = commit_core(params.commit, state.commit);
        Self {
            params,
            state,
            commit,
        }
    }

    pub fn tick(&mut self, inp: &IitInputs) -> IitOutputs {
        let lock_strength = pair_lock_strength(&inp.phase_bus_commit);
        let plv_score = combine_plv(inp.global_plv, lock_strength);

        let module_commits = module_commits(inp);
        let dep_score = dependency_score(&self.state, &module_commits, self.params.dep_lag);
        let base_phi = weighted_sum(&[
            (plv_score, self.params.w_plv),
            (dep_score, self.params.w_dep),
        ]);
        let stability_score = stability_score(
            &self.state,
            &module_commits,
            self.params.stab_lag,
            base_phi,
            inp,
            &self.params,
        );
        let phi_proxy = weighted_sum(&[
            (plv_score, self.params.w_plv),
            (dep_score, self.params.w_dep),
            (stability_score, self.params.w_stability),
        ]);

        let tighten_sync = phi_proxy < self.params.phi_low
            || (inp.global_plv < PLV_LOW && inp.drift >= self.params.drift_hi);
        let damp_output = phi_proxy < self.params.phi_low
            || inp.risk >= RISK_HIGH
            || inp.drift >= self.params.drift_hi;
        let damp_learning = inp.drift >= self.params.drift_hi
            || (inp.surprise >= self.params.surprise_hi && phi_proxy < self.params.phi_low);
        let request_replay =
            inp.surprise >= self.params.surprise_hi && phi_proxy < self.params.phi_high;
        let hints_commit = commit_hints(
            inp.cycle_id,
            phi_proxy,
            tighten_sync,
            damp_output,
            damp_learning,
            request_replay,
        );
        let commit = commit_outputs(inp.cycle_id, phi_proxy, hints_commit, inp.commit);

        update_commit_history(
            &mut self.state,
            &module_commits,
            max_lag(self.params.dep_lag, self.params.stab_lag),
        );
        self.state.last_phi = phi_proxy;
        self.state.commit = commit_state(&self.state);
        self.commit = commit_core(self.params.commit, self.state.commit);

        IitOutputs {
            cycle_id: inp.cycle_id,
            phi_proxy,
            tighten_sync,
            damp_output,
            damp_learning,
            request_replay,
            hints_commit,
            commit,
        }
    }
}

impl Default for IitCore {
    fn default() -> Self {
        Self::new(IitParams::default())
    }
}

fn module_commits(inp: &IitInputs) -> Vec<(ModuleId, [u8; 16])> {
    vec![
        (ModuleId::SsmState, truncate_digest(&inp.ssm_state_commit)),
        (ModuleId::NcdeState, truncate_digest(&inp.ncde_state_digest)),
        (ModuleId::Cde, truncate_digest(&inp.cde_commit)),
        (ModuleId::NsrTrace, truncate_digest(&inp.nsr_trace_root)),
        (
            ModuleId::Coupling,
            truncate_digest(&inp.coupling_influences_root),
        ),
    ]
}

fn dependency_score(state: &IitState, commits: &[(ModuleId, [u8; 16])], lag: u8) -> u16 {
    let mut total: u32 = 0;
    let mut count: u32 = 0;
    for (module, current) in commits {
        if let Some(prev) = lagged_commit(&state.last_commits, *module, lag) {
            let dist = xor_popcount(*current, prev);
            total = total.saturating_add(u32::from(dep_bell_score(dist)));
            count = count.saturating_add(1);
        }
    }
    if count == 0 {
        return 0;
    }
    clamp_score(total / count)
}

fn stability_score(
    state: &IitState,
    commits: &[(ModuleId, [u8; 16])],
    lag: u8,
    base_phi: u16,
    inp: &IitInputs,
    params: &IitParams,
) -> u16 {
    let mut total_dist: u32 = 0;
    let mut count: u32 = 0;
    for (module, current) in commits {
        if let Some(prev) = lagged_commit(&state.last_commits, *module, lag) {
            let dist = xor_popcount(*current, prev);
            total_dist = total_dist.saturating_add(u32::from(dist));
            count = count.saturating_add(1);
        }
    }
    let avg_dist = if count == 0 {
        0
    } else {
        u16::try_from(total_dist / count).unwrap_or(0)
    };
    let mut stability = stability_table_score(avg_dist);
    if avg_dist > 0 {
        let delta_phi = base_phi.abs_diff(state.last_phi);
        if delta_phi > 1500 {
            stability = stability.saturating_sub(delta_phi.saturating_sub(1500));
        }
    }
    if inp.drift >= params.drift_hi {
        stability = (u32::from(stability) * 60 / 100) as u16;
    }
    if inp.surprise >= params.surprise_hi {
        stability = (u32::from(stability) * 70 / 100) as u16;
    }
    stability.min(MAX_SCORE)
}

fn dep_bell_score(dist: u16) -> u16 {
    let index = (dist / u16::from(DEP_BELL_BUCKET_SIZE)) as usize;
    DEP_BELL_TABLE
        .get(index.min(DEP_BELL_TABLE.len() - 1))
        .copied()
        .unwrap_or(0)
}

fn stability_table_score(dist: u16) -> u16 {
    let index = (dist / u16::from(DEP_BELL_BUCKET_SIZE)) as usize;
    STABILITY_TABLE
        .get(index.min(STABILITY_TABLE.len() - 1))
        .copied()
        .unwrap_or(0)
}

fn pair_lock_strength(commit: &Digest32) -> u16 {
    let popcount: u32 = commit
        .as_bytes()
        .iter()
        .map(|value| value.count_ones())
        .sum();
    let scaled = popcount * u32::from(MAX_SCORE) / (Digest32::LEN as u32 * 8);
    clamp_score(scaled)
}

fn combine_plv(global_plv: u16, lock_strength: u16) -> u16 {
    let sum = u32::from(global_plv.min(MAX_SCORE)) + u32::from(lock_strength);
    clamp_score(sum / 2)
}

fn lagged_commit(history: &[(ModuleId, [u8; 16])], module: ModuleId, lag: u8) -> Option<[u8; 16]> {
    if lag == 0 {
        return None;
    }
    let mut seen = 0u8;
    for (id, commit) in history.iter().rev() {
        if *id == module {
            seen = seen.saturating_add(1);
            if seen == lag {
                return Some(*commit);
            }
        }
    }
    None
}

fn update_commit_history(state: &mut IitState, commits: &[(ModuleId, [u8; 16])], max_lag: u8) {
    if max_lag == 0 {
        return;
    }
    let per_cycle = commits.len();
    let max_entries = per_cycle.saturating_mul(max_lag as usize);
    for entry in commits {
        state.last_commits.push(*entry);
    }
    if state.last_commits.len() > max_entries {
        let trim = state.last_commits.len().saturating_sub(max_entries);
        state.last_commits.drain(0..trim);
    }
}

fn max_lag(dep_lag: u8, stab_lag: u8) -> u8 {
    dep_lag.max(stab_lag).clamp(1, 8)
}

fn xor_popcount(a: [u8; 16], b: [u8; 16]) -> u16 {
    let mut count: u32 = 0;
    for (left, right) in a.iter().zip(b.iter()) {
        count = count.saturating_add((left ^ right).count_ones());
    }
    u16::try_from(count.min(u32::from(u16::MAX))).unwrap_or(u16::MAX)
}

fn truncate_digest(digest: &Digest32) -> [u8; 16] {
    let mut out = [0u8; 16];
    out.copy_from_slice(&digest.as_bytes()[..16]);
    out
}

fn weighted_sum(values: &[(u16, u16)]) -> u16 {
    let mut sum: u32 = 0;
    let mut weight: u32 = 0;
    for (value, w) in values {
        if *w == 0 {
            continue;
        }
        sum = sum.saturating_add(u32::from(*value) * u32::from(*w));
        weight = weight.saturating_add(u32::from(*w));
    }
    if weight == 0 {
        return 0;
    }
    clamp_score(sum / weight)
}

fn clamp_score(value: u32) -> u16 {
    u16::try_from(value.min(u32::from(MAX_SCORE))).unwrap_or(MAX_SCORE)
}

fn commit_inputs(inputs: &IitInputs) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_INPUT);
    hasher.update(&inputs.cycle_id.to_be_bytes());
    hasher.update(inputs.phase_bus_commit.as_bytes());
    hasher.update(&[inputs.gamma_bucket]);
    hasher.update(&inputs.global_plv.to_be_bytes());
    hasher.update(inputs.ssm_state_commit.as_bytes());
    hasher.update(inputs.ncde_state_digest.as_bytes());
    hasher.update(inputs.cde_commit.as_bytes());
    hasher.update(inputs.nsr_trace_root.as_bytes());
    hasher.update(inputs.coupling_influences_root.as_bytes());
    hasher.update(&inputs.risk.to_be_bytes());
    hasher.update(&inputs.drift.to_be_bytes());
    hasher.update(&inputs.surprise.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_hints(
    cycle_id: u64,
    phi: u16,
    tighten_sync: bool,
    damp_output: bool,
    damp_learning: bool,
    request_replay: bool,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_HINTS);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(&phi.to_be_bytes());
    hasher.update(&[
        tighten_sync as u8,
        damp_output as u8,
        damp_learning as u8,
        request_replay as u8,
    ]);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_outputs(
    cycle_id: u64,
    phi: u16,
    hints_commit: Digest32,
    inputs_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_OUTPUT);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(&phi.to_be_bytes());
    hasher.update(hints_commit.as_bytes());
    hasher.update(inputs_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_params(params: &IitParams) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_PARAMS);
    hasher.update(&params.w_plv.to_be_bytes());
    hasher.update(&params.w_dep.to_be_bytes());
    hasher.update(&params.w_stability.to_be_bytes());
    hasher.update(&params.phi_low.to_be_bytes());
    hasher.update(&params.phi_high.to_be_bytes());
    hasher.update(&params.drift_hi.to_be_bytes());
    hasher.update(&params.surprise_hi.to_be_bytes());
    hasher.update(&[params.dep_lag]);
    hasher.update(&[params.stab_lag]);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_state(state: &IitState) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_STATE);
    hasher.update(&state.last_phi.to_be_bytes());
    hasher.update(
        &u32::try_from(state.last_commits.len())
            .unwrap_or(u32::MAX)
            .to_be_bytes(),
    );
    for (module, commit) in &state.last_commits {
        hasher.update(&[module.as_u8()]);
        hasher.update(commit);
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_core(params_commit: Digest32, state_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_CORE);
    hasher.update(params_commit.as_bytes());
    hasher.update(state_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_inputs() -> IitInputs {
        IitInputs::new(
            7,
            Digest32::new([1u8; 32]),
            3,
            5000,
            Digest32::new([2u8; 32]),
            Digest32::new([3u8; 32]),
            Digest32::new([4u8; 32]),
            Digest32::new([5u8; 32]),
            Digest32::new([6u8; 32]),
            2000,
            1200,
            1500,
        )
    }

    #[test]
    fn deterministic_outputs_for_same_inputs() {
        let mut core = IitCore::default();
        let inputs = base_inputs();
        let out_a = core.tick(&inputs);
        let out_b = core.tick(&inputs);

        assert_eq!(out_a.phi_proxy, out_b.phi_proxy);
        assert_eq!(out_a.commit, out_b.commit);
        assert_eq!(out_a.hints_commit, out_b.hints_commit);
    }

    #[test]
    fn dep_bell_mapping_has_peak_and_tails() {
        let low = dep_bell_score(0);
        let mid = dep_bell_score(48);
        let high = dep_bell_score(120);

        assert!(mid > low);
        assert!(mid > high);
    }

    #[test]
    fn low_phi_sets_tighten_and_damp_output() {
        let mut core = IitCore::default();
        let mut inputs = base_inputs();
        inputs.global_plv = 1200;
        inputs.drift = 8000;
        let output = core.tick(&inputs);

        assert!(output.tighten_sync);
        assert!(output.damp_output);
    }
}
