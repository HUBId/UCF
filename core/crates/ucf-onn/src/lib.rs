#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_types::Digest32;

const COHERENCE_MAX: u16 = 10_000;
const DEFAULT_PHASE_WINDOW: u16 = 16_384;
const DEFAULT_K_GLOBAL: u16 = 42;
const DEFAULT_PAIR_GAIN: u16 = 24;
const MAX_COUPLING_DELTA: i32 = 256;
const PHASE_WRAP: i32 = 65_536;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModuleId {
    Ai,
    Ssm,
    Cde,
    Nsr,
    Geist,
    Replay,
    Output,
    BlueBrain,
    Unknown(u16),
}

impl ModuleId {
    pub fn as_u16(self) -> u16 {
        match self {
            Self::Ai => 1,
            Self::Ssm => 2,
            Self::Cde => 3,
            Self::Nsr => 4,
            Self::Geist => 5,
            Self::Replay => 6,
            Self::Output => 7,
            Self::BlueBrain => 8,
            Self::Unknown(value) => value,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhaseFrame {
    pub cycle_id: u64,
    pub global_phase: u16,
    pub module_phase: Vec<(ModuleId, u16)>,
    pub module_freq: Vec<(ModuleId, u16)>,
    pub coherence_plv: u16,
    pub commit: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnnParams {
    pub k_global: u16,
    pub k_pairs: Vec<(ModuleId, ModuleId, u16)>,
    pub phase_window: u16,
    pub commit: Digest32,
}

impl OnnParams {
    pub fn new(k_global: u16, k_pairs: Vec<(ModuleId, ModuleId, u16)>, phase_window: u16) -> Self {
        let commit = commit_params(k_global, phase_window, &k_pairs);
        Self {
            k_global,
            k_pairs,
            phase_window,
            commit,
        }
    }
}

impl Default for OnnParams {
    fn default() -> Self {
        let k_pairs = default_pair_gains();
        Self::new(DEFAULT_K_GLOBAL, k_pairs, DEFAULT_PHASE_WINDOW)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnnInputs {
    pub drift: u16,
    pub surprise: u16,
    pub attn_gain: u16,
    pub nsr_verdict: u8,
    pub brain_arousal: u16,
    pub commit: Digest32,
}

impl OnnInputs {
    pub fn new(
        drift: u16,
        surprise: u16,
        attn_gain: u16,
        nsr_verdict: u8,
        brain_arousal: u16,
    ) -> Self {
        let commit = commit_inputs(drift, surprise, attn_gain, nsr_verdict, brain_arousal);
        Self {
            drift,
            surprise,
            attn_gain,
            nsr_verdict,
            brain_arousal,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnnState {
    pub last: Option<PhaseFrame>,
    pub commit: Digest32,
}

impl OnnState {
    pub fn new(last: Option<PhaseFrame>) -> Self {
        let commit = commit_state(last.as_ref());
        Self { last, commit }
    }
}

impl Default for OnnState {
    fn default() -> Self {
        Self::new(None)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnnCore {
    pub params: OnnParams,
    pub state: OnnState,
}

impl OnnCore {
    pub fn new(params: OnnParams) -> Self {
        Self {
            params,
            state: OnnState::default(),
        }
    }

    pub fn tick(
        &mut self,
        cycle_id: u64,
        cycle_commit: Digest32,
        inputs: &OnnInputs,
    ) -> PhaseFrame {
        let default_modules = default_modules();
        let last_frame = self.state.last.as_ref();
        let module_list = merged_modules(last_frame, &default_modules);
        let (base_phase, base_freq) = match last_frame {
            Some(frame) => (frame.module_phase.clone(), frame.module_freq.clone()),
            None => (
                seed_module_phases(&module_list, self.params.commit, cycle_commit),
                seed_module_freqs(&module_list, self.params.commit),
            ),
        };
        let global_phase = next_global_phase(last_frame, inputs);
        let mut module_phase = Vec::with_capacity(module_list.len());
        let mut module_freq = Vec::with_capacity(module_list.len());

        for module in &module_list {
            let theta = lookup_module(*module, &base_phase).unwrap_or(0);
            let omega = lookup_module(*module, &base_freq).unwrap_or(0);
            let coupling = coupling_delta(*module, theta, global_phase, &base_phase, &self.params);
            let next_theta = wrap_phase(i32::from(theta) + i32::from(omega) + coupling);
            module_phase.push((*module, next_theta));
            module_freq.push((*module, omega));
        }

        let coherence_plv = coherence_plv(&module_phase, self.params.phase_window);
        let commit = commit_phase_frame(
            cycle_id,
            global_phase,
            &module_phase,
            &module_freq,
            coherence_plv,
            cycle_commit,
            self.params.commit,
            inputs.commit,
        );
        let frame = PhaseFrame {
            cycle_id,
            global_phase,
            module_phase,
            module_freq,
            coherence_plv,
            commit,
        };
        self.state = OnnState::new(Some(frame.clone()));
        frame
    }
}

impl Default for OnnCore {
    fn default() -> Self {
        Self::new(OnnParams::default())
    }
}

fn default_modules() -> Vec<ModuleId> {
    vec![
        ModuleId::Ai,
        ModuleId::Ssm,
        ModuleId::Cde,
        ModuleId::Nsr,
        ModuleId::Geist,
        ModuleId::Replay,
        ModuleId::Output,
        ModuleId::BlueBrain,
    ]
}

fn default_pair_gains() -> Vec<(ModuleId, ModuleId, u16)> {
    vec![
        (ModuleId::Ai, ModuleId::Ssm, DEFAULT_PAIR_GAIN),
        (ModuleId::Ssm, ModuleId::Cde, DEFAULT_PAIR_GAIN),
        (ModuleId::Cde, ModuleId::Nsr, DEFAULT_PAIR_GAIN),
        (ModuleId::Nsr, ModuleId::Output, DEFAULT_PAIR_GAIN),
        (ModuleId::Geist, ModuleId::Replay, DEFAULT_PAIR_GAIN),
    ]
}

fn merged_modules(last: Option<&PhaseFrame>, defaults: &[ModuleId]) -> Vec<ModuleId> {
    let mut modules = defaults.to_vec();
    if let Some(frame) = last {
        for (module, _) in &frame.module_phase {
            if !modules.contains(module) {
                modules.push(*module);
            }
        }
    }
    modules
}

fn next_global_phase(last: Option<&PhaseFrame>, inputs: &OnnInputs) -> u16 {
    let last_phase = last.map(|frame| frame.global_phase).unwrap_or(0);
    let base_step = 128u16.saturating_add(inputs.attn_gain / 64);
    let modulation = inputs.drift / 32 + inputs.surprise / 64;
    last_phase.wrapping_add(base_step).wrapping_add(modulation)
}

fn seed_module_phases(
    modules: &[ModuleId],
    params_commit: Digest32,
    cycle_commit: Digest32,
) -> Vec<(ModuleId, u16)> {
    modules
        .iter()
        .map(|module| {
            let mut hasher = Hasher::new();
            hasher.update(b"ucf.onn.seed.phase.v1");
            hasher.update(&module.as_u16().to_be_bytes());
            hasher.update(params_commit.as_bytes());
            hasher.update(cycle_commit.as_bytes());
            let hash = hasher.finalize();
            let bytes = hash.as_bytes();
            let value = u16::from_be_bytes([bytes[0], bytes[1]]);
            (*module, value)
        })
        .collect()
}

fn seed_module_freqs(modules: &[ModuleId], params_commit: Digest32) -> Vec<(ModuleId, u16)> {
    modules
        .iter()
        .map(|module| {
            let mut hasher = Hasher::new();
            hasher.update(b"ucf.onn.seed.freq.v1");
            hasher.update(&module.as_u16().to_be_bytes());
            hasher.update(params_commit.as_bytes());
            let hash = hasher.finalize();
            let bytes = hash.as_bytes();
            let raw = u16::from_be_bytes([bytes[0], bytes[1]]);
            let omega = 120u16.saturating_add(raw % 280);
            (*module, omega)
        })
        .collect()
}

fn lookup_module(module: ModuleId, list: &[(ModuleId, u16)]) -> Option<u16> {
    list.iter()
        .find_map(|(id, value)| (*id == module).then_some(*value))
}

fn coupling_delta(
    module: ModuleId,
    theta: u16,
    global_phase: u16,
    module_phase: &[(ModuleId, u16)],
    params: &OnnParams,
) -> i32 {
    let mut delta = 0i32;
    let window = i32::from(params.phase_window.max(1));
    let global_diff = signed_phase_delta(theta, global_phase);
    delta = delta.saturating_add(scale_diff(global_diff, params.k_global, window));

    for (a, b, gain) in &params.k_pairs {
        if *a == module {
            if let Some(other) = lookup_module(*b, module_phase) {
                let diff = signed_phase_delta(theta, other);
                delta = delta.saturating_add(scale_diff(diff, *gain, window));
            }
        } else if *b == module {
            if let Some(other) = lookup_module(*a, module_phase) {
                let diff = signed_phase_delta(theta, other);
                delta = delta.saturating_add(scale_diff(diff, *gain, window));
            }
        }
    }

    delta.clamp(-MAX_COUPLING_DELTA, MAX_COUPLING_DELTA)
}

fn scale_diff(diff: i32, gain: u16, window: i32) -> i32 {
    if gain == 0 || window <= 0 {
        return 0;
    }
    diff.saturating_mul(i32::from(gain)) / window
}

fn signed_phase_delta(theta: u16, other: u16) -> i32 {
    let diff = other.wrapping_sub(theta) as i32;
    if diff > 32_768 {
        diff - PHASE_WRAP
    } else {
        diff
    }
}

fn wrap_phase(value: i32) -> u16 {
    value.rem_euclid(PHASE_WRAP) as u16
}

fn coherence_plv(module_phase: &[(ModuleId, u16)], phase_window: u16) -> u16 {
    let window = phase_window.max(1);
    let pairs = [
        (ModuleId::Ai, ModuleId::Ssm),
        (ModuleId::Ssm, ModuleId::Cde),
        (ModuleId::Cde, ModuleId::Nsr),
        (ModuleId::Nsr, ModuleId::Output),
        (ModuleId::Geist, ModuleId::Replay),
    ];
    let mut sum: u32 = 0;
    let mut count: u32 = 0;
    for (a, b) in pairs {
        let Some(theta_a) = lookup_module(a, module_phase) else {
            continue;
        };
        let Some(theta_b) = lookup_module(b, module_phase) else {
            continue;
        };
        let dist = modular_distance(theta_a, theta_b);
        let similarity = window.saturating_sub(dist);
        let scaled = (u32::from(similarity) * u32::from(COHERENCE_MAX)) / u32::from(window);
        sum = sum.saturating_add(scaled);
        count = count.saturating_add(1);
    }
    if count == 0 {
        return 0;
    }
    let avg = sum / count;
    u16::try_from(avg.min(u32::from(COHERENCE_MAX))).unwrap_or(COHERENCE_MAX)
}

fn modular_distance(theta_a: u16, theta_b: u16) -> u16 {
    let diff = u32::from(theta_a.abs_diff(theta_b));
    let wrapped = PHASE_WRAP as u32 - diff;
    let min = diff.min(wrapped).min(32_768);
    min as u16
}

fn commit_params(
    k_global: u16,
    phase_window: u16,
    pairs: &[(ModuleId, ModuleId, u16)],
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.onn.params.v1");
    hasher.update(&k_global.to_be_bytes());
    hasher.update(&phase_window.to_be_bytes());
    for (a, b, gain) in pairs {
        hasher.update(&a.as_u16().to_be_bytes());
        hasher.update(&b.as_u16().to_be_bytes());
        hasher.update(&gain.to_be_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_inputs(
    drift: u16,
    surprise: u16,
    attn_gain: u16,
    nsr_verdict: u8,
    brain_arousal: u16,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.onn.inputs.v1");
    hasher.update(&drift.to_be_bytes());
    hasher.update(&surprise.to_be_bytes());
    hasher.update(&attn_gain.to_be_bytes());
    hasher.update(&[nsr_verdict]);
    hasher.update(&brain_arousal.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_state(last: Option<&PhaseFrame>) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.onn.state.v1");
    if let Some(frame) = last {
        hasher.update(frame.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

#[allow(clippy::too_many_arguments)]
fn commit_phase_frame(
    cycle_id: u64,
    global_phase: u16,
    module_phase: &[(ModuleId, u16)],
    module_freq: &[(ModuleId, u16)],
    coherence_plv: u16,
    cycle_commit: Digest32,
    params_commit: Digest32,
    inputs_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.onn.phase_frame.v1");
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(&global_phase.to_be_bytes());
    for (module, theta) in module_phase {
        hasher.update(&module.as_u16().to_be_bytes());
        hasher.update(&theta.to_be_bytes());
    }
    for (module, omega) in module_freq {
        hasher.update(&module.as_u16().to_be_bytes());
        hasher.update(&omega.to_be_bytes());
    }
    hasher.update(&coherence_plv.to_be_bytes());
    hasher.update(cycle_commit.as_bytes());
    hasher.update(params_commit.as_bytes());
    hasher.update(inputs_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_is_deterministic_for_same_inputs() {
        let mut core_a = OnnCore::default();
        let mut core_b = OnnCore::default();
        let inputs = OnnInputs::new(10, 20, 3000, 1, 500);
        let cycle_commit = Digest32::new([3u8; 32]);

        let frame_a = core_a.tick(1, cycle_commit, &inputs);
        let frame_b = core_b.tick(1, cycle_commit, &inputs);

        assert_eq!(frame_a, frame_b);
        assert_eq!(frame_a.commit, frame_b.commit);
    }

    #[test]
    fn coherence_drops_when_phases_diverge() {
        let params = OnnParams::new(0, Vec::new(), 4096);
        let aligned = PhaseFrame {
            cycle_id: 1,
            global_phase: 0,
            module_phase: vec![
                (ModuleId::Ai, 0),
                (ModuleId::Ssm, 0),
                (ModuleId::Cde, 0),
                (ModuleId::Nsr, 0),
                (ModuleId::Output, 0),
                (ModuleId::Geist, 0),
                (ModuleId::Replay, 0),
            ],
            module_freq: vec![
                (ModuleId::Ai, 0),
                (ModuleId::Ssm, 0),
                (ModuleId::Cde, 0),
                (ModuleId::Nsr, 0),
                (ModuleId::Output, 0),
                (ModuleId::Geist, 0),
                (ModuleId::Replay, 0),
            ],
            coherence_plv: 0,
            commit: Digest32::new([0u8; 32]),
        };
        let diverged = PhaseFrame {
            module_phase: vec![
                (ModuleId::Ai, 0),
                (ModuleId::Ssm, 20_000),
                (ModuleId::Cde, 30_000),
                (ModuleId::Nsr, 40_000),
                (ModuleId::Output, 50_000),
                (ModuleId::Geist, 10_000),
                (ModuleId::Replay, 55_000),
            ],
            ..aligned.clone()
        };
        let mut core_aligned = OnnCore {
            params: params.clone(),
            state: OnnState::new(Some(aligned)),
        };
        let mut core_diverged = OnnCore {
            params,
            state: OnnState::new(Some(diverged)),
        };
        let inputs = OnnInputs::new(0, 0, 0, 0, 0);
        let cycle_commit = Digest32::new([1u8; 32]);

        let aligned_frame = core_aligned.tick(2, cycle_commit, &inputs);
        let diverged_frame = core_diverged.tick(2, cycle_commit, &inputs);

        assert!(aligned_frame.coherence_plv > diverged_frame.coherence_plv);
    }
}
