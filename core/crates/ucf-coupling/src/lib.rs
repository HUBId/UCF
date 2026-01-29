#![forbid(unsafe_code)]

use blake3::Hasher;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use ucf_types::Digest32;

const MAX_LAG: u8 = 8;
const MIX_SCALE: i32 = 10_000;
const INFLUENCE_CAP: i16 = 10_000;
const ALPHA_MOD: u16 = 10_001;

const SAMPLE_DOMAIN: &[u8] = b"ucf.coupling.sample.v1";
const BUFFER_DOMAIN: &[u8] = b"ucf.coupling.buffer.v1";
const RULE_DOMAIN: &[u8] = b"ucf.coupling.rule.v1";
const INPUT_DOMAIN: &[u8] = b"ucf.coupling.inputs.v1";
const OUTPUT_DOMAIN: &[u8] = b"ucf.coupling.outputs.v1";
const OUTPUT_ROOT_DOMAIN: &[u8] = b"ucf.coupling.outputs.root.v1";
const CORE_DOMAIN: &[u8] = b"ucf.coupling.core.v1";
const ALPHA_DOMAIN: &[u8] = b"ucf.coupling.alpha.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SignalId {
    PerceptEnergy,
    SsmSalience,
    SsmNovelty,
    SsmAttentionGain,
    AttentionFinalGain,
    NcdeEnergy,
    PhiProxy,
    GlobalPlv,
    Risk,
    Drift,
    Surprise,
    ReplayPressure,
    SleepDrive,
    NsrVerdict,
    LearningHint,
    RsaProposalStrength,
    Unknown(u16),
}

impl SignalId {
    pub fn as_u16(self) -> u16 {
        match self {
            Self::PerceptEnergy => 1,
            Self::SsmSalience => 2,
            Self::SsmNovelty => 3,
            Self::SsmAttentionGain => 4,
            Self::AttentionFinalGain => 5,
            Self::NcdeEnergy => 6,
            Self::PhiProxy => 7,
            Self::GlobalPlv => 8,
            Self::Risk => 9,
            Self::Drift => 10,
            Self::Surprise => 11,
            Self::ReplayPressure => 12,
            Self::SleepDrive => 13,
            Self::NsrVerdict => 14,
            Self::LearningHint => 15,
            Self::RsaProposalStrength => 16,
            Self::Unknown(value) => value,
        }
    }

    fn is_whitelisted(self) -> bool {
        matches!(
            self,
            Self::PerceptEnergy
                | Self::SsmSalience
                | Self::SsmNovelty
                | Self::SsmAttentionGain
                | Self::AttentionFinalGain
                | Self::NcdeEnergy
                | Self::PhiProxy
                | Self::GlobalPlv
                | Self::Risk
                | Self::Drift
                | Self::Surprise
                | Self::ReplayPressure
                | Self::SleepDrive
                | Self::NsrVerdict
                | Self::LearningHint
                | Self::RsaProposalStrength
        )
    }
}

impl Ord for SignalId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_u16().cmp(&other.as_u16())
    }
}

impl PartialOrd for SignalId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignalSample {
    pub cycle_id: u64,
    pub id: SignalId,
    pub value: i16,
    pub commit: Digest32,
}

impl SignalSample {
    pub fn new(cycle_id: u64, id: SignalId, value: i16) -> Self {
        let commit = commit_sample(cycle_id, id, value);
        Self {
            cycle_id,
            id,
            value,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LagBuffer {
    pub id: SignalId,
    pub lag_max: u8,
    pub values: [i16; MAX_LAG as usize],
    pub head: u8,
    pub commit: Digest32,
}

impl LagBuffer {
    pub fn new(id: SignalId, lag_max: u8) -> Self {
        let lag_max = lag_max.clamp(1, MAX_LAG);
        let values = [0; MAX_LAG as usize];
        let head = 0;
        let commit = commit_buffer(id, lag_max, &values, head);
        Self {
            id,
            lag_max,
            values,
            head,
            commit,
        }
    }

    pub fn push(&mut self, value: i16) {
        let lag_max = usize::from(self.lag_max);
        if lag_max == 0 {
            return;
        }
        let head = usize::from(self.head);
        self.values[head] = value;
        self.head = ((head + 1) % lag_max) as u8;
        self.commit = commit_buffer(self.id, self.lag_max, &self.values, self.head);
    }

    pub fn read_lag(&self, lag: u8) -> i16 {
        if lag == 0 || lag > self.lag_max {
            return 0;
        }
        let lag_max = usize::from(self.lag_max);
        let head = usize::from(self.head);
        let lag = usize::from(lag);
        let idx = (head + lag_max - lag) % lag_max;
        self.values[idx]
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CouplingRule {
    pub src: SignalId,
    pub dst: SignalId,
    pub lag: u8,
    pub gain: i16,
    pub mix_k: u16,
    pub commit: Digest32,
}

impl CouplingRule {
    pub fn new(src: SignalId, dst: SignalId, lag: u8, gain: i16, mix_k: u16) -> Self {
        let lag = lag.clamp(1, MAX_LAG);
        let mix_k = mix_k.min(MIX_SCALE as u16);
        let commit = commit_rule(src, dst, lag, gain, mix_k);
        Self {
            src,
            dst,
            lag,
            gain,
            mix_k,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CouplingInputs {
    pub cycle_id: u64,
    pub phase_commit: Digest32,
    pub phase_bucket: u8,
    pub samples: Vec<SignalSample>,
    pub commit: Digest32,
}

impl CouplingInputs {
    pub fn new(
        cycle_id: u64,
        phase_commit: Digest32,
        phase_bucket: u8,
        mut samples: Vec<SignalSample>,
    ) -> Self {
        samples.sort_by(|a, b| a.id.cmp(&b.id).then_with(|| a.cycle_id.cmp(&b.cycle_id)));
        let commit = commit_inputs(cycle_id, phase_commit, phase_bucket, &samples);
        Self {
            cycle_id,
            phase_commit,
            phase_bucket,
            samples,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CouplingOutputs {
    pub cycle_id: u64,
    pub influences: Vec<(SignalId, i16)>,
    pub influences_root: Digest32,
    pub commit: Digest32,
}

impl CouplingOutputs {
    pub fn new(cycle_id: u64, mut influences: Vec<(SignalId, i16)>) -> Self {
        influences.sort_by(|a, b| a.0.cmp(&b.0));
        let influences_root = commit_influences_root(cycle_id, &influences);
        let commit = commit_outputs(cycle_id, influences_root, &influences);
        Self {
            cycle_id,
            influences,
            influences_root,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CouplingCore {
    pub bufs: Vec<LagBuffer>,
    pub rules: Vec<CouplingRule>,
    pub commit: Digest32,
}

impl CouplingCore {
    pub fn new(rules: Vec<CouplingRule>) -> Self {
        let commit = commit_core(&rules, &[]);
        Self {
            bufs: Vec::new(),
            rules,
            commit,
        }
    }

    pub fn new_default() -> Self {
        Self::new(default_rules())
    }

    pub fn tick(&mut self, inp: &CouplingInputs) -> CouplingOutputs {
        let mut current_map: BTreeMap<SignalId, i16> = BTreeMap::new();
        for sample in &inp.samples {
            current_map.insert(sample.id, sample.value);
            if sample.id.is_whitelisted() {
                self.push_sample(sample.id, sample.value);
            }
        }

        let mut influences: BTreeMap<SignalId, i16> = BTreeMap::new();
        for rule in &self.rules {
            let lagged = self.read_lag(rule.src, rule.lag);
            let current = if rule.lag == 1 {
                current_map.get(&rule.src).copied().unwrap_or(0)
            } else {
                0
            };
            let alpha = mixing_alpha(
                inp.phase_commit,
                rule.src,
                rule.dst,
                rule.lag,
                inp.cycle_id,
                rule.mix_k,
            );
            let mixed = mix_values(lagged, current, alpha);
            let influence = apply_gain(mixed, rule.gain);
            if influence != 0 {
                let entry = influences.entry(rule.dst).or_insert(0);
                let updated = i32::from(*entry).saturating_add(i32::from(influence));
                *entry = clamp_influence(updated);
            }
        }

        let outputs = CouplingOutputs::new(inp.cycle_id, influences.into_iter().collect());
        self.commit = commit_core(&self.rules, &self.bufs);
        outputs
    }

    pub fn buffer_commits(&self) -> Vec<(SignalId, Digest32)> {
        let mut commits = self
            .bufs
            .iter()
            .map(|buf| (buf.id, buf.commit))
            .collect::<Vec<_>>();
        commits.sort_by(|a, b| a.0.cmp(&b.0));
        commits
    }

    fn push_sample(&mut self, id: SignalId, value: i16) {
        let lag_max = max_lag_for(id, &self.rules);
        if let Some(buf) = self.bufs.iter_mut().find(|buf| buf.id == id) {
            if buf.lag_max != lag_max {
                *buf = LagBuffer::new(id, lag_max);
            }
            buf.push(value);
            return;
        }
        let mut buf = LagBuffer::new(id, lag_max);
        buf.push(value);
        self.bufs.push(buf);
    }

    fn read_lag(&self, id: SignalId, lag: u8) -> i16 {
        self.bufs
            .iter()
            .find(|buf| buf.id == id)
            .map(|buf| buf.read_lag(lag))
            .unwrap_or(0)
    }
}

impl Default for CouplingCore {
    fn default() -> Self {
        Self::new_default()
    }
}

fn max_lag_for(id: SignalId, rules: &[CouplingRule]) -> u8 {
    let mut max_lag = 1;
    for rule in rules {
        if rule.src == id {
            max_lag = max_lag.max(rule.lag);
        }
    }
    max_lag.clamp(1, MAX_LAG)
}

fn mixing_alpha(
    phase_commit: Digest32,
    src: SignalId,
    dst: SignalId,
    lag: u8,
    cycle_id: u64,
    mix_k: u16,
) -> u16 {
    let mut hasher = Hasher::new();
    hasher.update(ALPHA_DOMAIN);
    hasher.update(phase_commit.as_bytes());
    hasher.update(&src.as_u16().to_be_bytes());
    hasher.update(&dst.as_u16().to_be_bytes());
    hasher.update(&[lag]);
    hasher.update(&cycle_id.to_be_bytes());
    let hash = hasher.finalize();
    let bytes = hash.as_bytes();
    let base = u16::from_be_bytes([bytes[0], bytes[1]]);
    let mut alpha = base % ALPHA_MOD;
    alpha = ((u32::from(alpha) * u32::from(mix_k)) / MIX_SCALE as u32).min(MIX_SCALE as u32) as u16;
    alpha
}

fn mix_values(lagged: i16, current: i16, alpha: u16) -> i16 {
    let alpha = i32::from(alpha);
    let lagged = i32::from(lagged);
    let current = i32::from(current);
    let mixed = (alpha * lagged + (MIX_SCALE - alpha) * current) / MIX_SCALE;
    clamp_i16(mixed)
}

fn apply_gain(value: i16, gain: i16) -> i16 {
    let scaled = (i32::from(value) * i32::from(gain)) / MIX_SCALE;
    clamp_influence(scaled)
}

fn clamp_influence(value: i32) -> i16 {
    value.clamp(-i32::from(INFLUENCE_CAP), i32::from(INFLUENCE_CAP)) as i16
}

fn clamp_i16(value: i32) -> i16 {
    value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

fn commit_sample(cycle_id: u64, id: SignalId, value: i16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(SAMPLE_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(&id.as_u16().to_be_bytes());
    hasher.update(&value.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_buffer(
    id: SignalId,
    lag_max: u8,
    values: &[i16; MAX_LAG as usize],
    head: u8,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(BUFFER_DOMAIN);
    hasher.update(&id.as_u16().to_be_bytes());
    hasher.update(&[lag_max]);
    hasher.update(&[head]);
    for value in values {
        hasher.update(&value.to_be_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_rule(src: SignalId, dst: SignalId, lag: u8, gain: i16, mix_k: u16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(RULE_DOMAIN);
    hasher.update(&src.as_u16().to_be_bytes());
    hasher.update(&dst.as_u16().to_be_bytes());
    hasher.update(&[lag]);
    hasher.update(&gain.to_be_bytes());
    hasher.update(&mix_k.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_inputs(
    cycle_id: u64,
    phase_commit: Digest32,
    phase_bucket: u8,
    samples: &[SignalSample],
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(INPUT_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(phase_commit.as_bytes());
    hasher.update(&[phase_bucket]);
    hasher.update(
        &u32::try_from(samples.len())
            .unwrap_or(u32::MAX)
            .to_be_bytes(),
    );
    for sample in samples {
        hasher.update(sample.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_influences_root(cycle_id: u64, influences: &[(SignalId, i16)]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(OUTPUT_ROOT_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(
        &u32::try_from(influences.len())
            .unwrap_or(u32::MAX)
            .to_be_bytes(),
    );
    for (id, value) in influences {
        hasher.update(&id.as_u16().to_be_bytes());
        hasher.update(&value.to_be_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_outputs(
    cycle_id: u64,
    influences_root: Digest32,
    influences: &[(SignalId, i16)],
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(OUTPUT_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(influences_root.as_bytes());
    hasher.update(
        &u32::try_from(influences.len())
            .unwrap_or(u32::MAX)
            .to_be_bytes(),
    );
    for (id, value) in influences {
        hasher.update(&id.as_u16().to_be_bytes());
        hasher.update(&value.to_be_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_core(rules: &[CouplingRule], bufs: &[LagBuffer]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(CORE_DOMAIN);
    hasher.update(&u32::try_from(rules.len()).unwrap_or(u32::MAX).to_be_bytes());
    for rule in rules {
        hasher.update(rule.commit.as_bytes());
    }
    hasher.update(&u32::try_from(bufs.len()).unwrap_or(u32::MAX).to_be_bytes());
    for buf in bufs {
        hasher.update(buf.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn default_rules() -> Vec<CouplingRule> {
    vec![
        CouplingRule::new(
            SignalId::PerceptEnergy,
            SignalId::SsmSalience,
            1,
            2800,
            8000,
        ),
        CouplingRule::new(
            SignalId::SsmSalience,
            SignalId::AttentionFinalGain,
            1,
            2400,
            8200,
        ),
        CouplingRule::new(
            SignalId::SsmNovelty,
            SignalId::AttentionFinalGain,
            2,
            2100,
            7800,
        ),
        CouplingRule::new(
            SignalId::AttentionFinalGain,
            SignalId::NcdeEnergy,
            1,
            2200,
            7600,
        ),
        CouplingRule::new(
            SignalId::NcdeEnergy,
            SignalId::ReplayPressure,
            2,
            1800,
            7400,
        ),
        CouplingRule::new(SignalId::PhiProxy, SignalId::ReplayPressure, 1, -1600, 7400),
        CouplingRule::new(
            SignalId::AttentionFinalGain,
            SignalId::LearningHint,
            1,
            2000,
            8000,
        ),
        CouplingRule::new(
            SignalId::LearningHint,
            SignalId::RsaProposalStrength,
            3,
            2400,
            8200,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(cycle_id: u64, id: SignalId, value: i16) -> SignalSample {
        SignalSample::new(cycle_id, id, value)
    }

    #[test]
    fn determinism_holds_for_same_inputs() {
        let inputs = CouplingInputs::new(
            7,
            Digest32::new([1u8; 32]),
            2,
            vec![sample(7, SignalId::SsmSalience, 4200)],
        );
        let mut core_a = CouplingCore::new_default();
        let mut core_b = CouplingCore::new_default();

        let out_a = core_a.tick(&inputs);
        let out_b = core_b.tick(&inputs);

        assert_eq!(out_a, out_b);
        assert_eq!(out_a.commit, out_b.commit);
    }

    #[test]
    fn lagged_signal_only_influences_after_lag() {
        let rule = CouplingRule::new(
            SignalId::PerceptEnergy,
            SignalId::SsmSalience,
            2,
            5000,
            10_000,
        );
        let mut core = CouplingCore::new(vec![rule]);

        let inputs_1 = CouplingInputs::new(
            1,
            Digest32::new([2u8; 32]),
            2,
            vec![sample(1, SignalId::PerceptEnergy, 5000)],
        );
        let out_1 = core.tick(&inputs_1);
        assert!(out_1.influences.is_empty());

        let inputs_2 = CouplingInputs::new(
            2,
            Digest32::new([2u8; 32]),
            2,
            vec![sample(2, SignalId::PerceptEnergy, 5000)],
        );
        let out_2 = core.tick(&inputs_2);
        assert!(out_2
            .influences
            .iter()
            .any(|(id, value)| *id == SignalId::SsmSalience && *value != 0));
    }

    #[test]
    fn mixing_depends_on_phase_commit() {
        let rule = CouplingRule::new(
            SignalId::SsmSalience,
            SignalId::AttentionFinalGain,
            2,
            3000,
            10_000,
        );
        let mut core_a = CouplingCore::new(vec![rule.clone()]);
        let mut core_b = CouplingCore::new(vec![rule]);

        let seed = CouplingInputs::new(
            1,
            Digest32::new([2u8; 32]),
            3,
            vec![sample(1, SignalId::SsmSalience, 6000)],
        );
        core_a.tick(&seed);
        core_b.tick(&seed);

        let inputs_a = CouplingInputs::new(
            2,
            Digest32::new([3u8; 32]),
            3,
            vec![sample(2, SignalId::SsmSalience, 6000)],
        );
        let inputs_b = CouplingInputs::new(
            2,
            Digest32::new([4u8; 32]),
            3,
            vec![sample(2, SignalId::SsmSalience, 6000)],
        );

        let out_a = core_a.tick(&inputs_a);
        let out_b = core_b.tick(&inputs_b);

        assert_ne!(out_a.influences_root, out_b.influences_root);
    }

    #[test]
    fn influences_are_clamped_for_safety() {
        let rule = CouplingRule::new(
            SignalId::PerceptEnergy,
            SignalId::SsmSalience,
            1,
            10_000,
            10_000,
        );
        let mut core = CouplingCore::new(vec![rule]);
        let inputs = CouplingInputs::new(
            1,
            Digest32::new([5u8; 32]),
            1,
            vec![sample(1, SignalId::PerceptEnergy, i16::MAX)],
        );

        let outputs = core.tick(&inputs);
        let (_, value) = outputs.influences.first().expect("influence present");
        assert_eq!(*value, INFLUENCE_CAP);
    }

    #[test]
    fn attention_gain_influenced_by_ssm_with_lag() {
        let mut core = CouplingCore::new_default();
        let inputs_1 = CouplingInputs::new(
            1,
            Digest32::new([6u8; 32]),
            2,
            vec![sample(1, SignalId::SsmSalience, 4000)],
        );
        let out_1 = core.tick(&inputs_1);
        assert!(out_1
            .influences
            .iter()
            .any(|(id, value)| *id == SignalId::AttentionFinalGain && *value != 0));
    }
}
