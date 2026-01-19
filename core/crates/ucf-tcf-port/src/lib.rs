#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_attn_controller::{AttentionWeights, FocusChannel};
use ucf_predictive_coding::{SurpriseBand, SurpriseSignal};
use ucf_types::Digest32;

const PHASE_MAX: i32 = 1_000_000;
const ENERGY_MAX: u16 = 10_000;
const MAX_PULSES: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Phase {
    pub q: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TcfConfig {
    pub base_freq: u16,
    pub damping: u16,
    pub jitter_guard: u16,
}

impl Default for TcfConfig {
    fn default() -> Self {
        Self {
            base_freq: 1200,
            damping: 120,
            jitter_guard: 400,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TcfState {
    pub phase: Phase,
    pub energy: u16,
    pub commit: Digest32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PulseKind {
    Sense,
    Think,
    Verify,
    Consolidate,
    Broadcast,
    Sleep,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pulse {
    pub kind: PulseKind,
    pub weight: u16,
    pub slot: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CyclePlan {
    pub cycle_id: u64,
    pub pulses: Vec<Pulse>,
    pub commit: Digest32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CyclePlanned {
    pub cycle_id: u64,
    pub commit: Digest32,
    pub pulse_count: u8,
}

pub trait TcfPort {
    fn step(&mut self, attn: &AttentionWeights, surprise: Option<&SurpriseSignal>) -> CyclePlan;
    fn state(&self) -> &TcfState;
}

#[derive(Clone, Debug)]
pub struct DeterministicTcf {
    config: TcfConfig,
    state: TcfState,
    cycle_id: u64,
}

impl DeterministicTcf {
    pub fn new(config: TcfConfig) -> Self {
        let state = TcfState {
            phase: Phase { q: 0 },
            energy: ENERGY_MAX / 2,
            commit: Digest32::new([0u8; 32]),
        };
        Self {
            config,
            state: refresh_commit(state),
            cycle_id: 0,
        }
    }

    pub fn planned_event(plan: &CyclePlan) -> CyclePlanned {
        CyclePlanned {
            cycle_id: plan.cycle_id,
            commit: plan.commit,
            pulse_count: plan.pulses.len().min(u8::MAX as usize) as u8,
        }
    }
}

impl Default for DeterministicTcf {
    fn default() -> Self {
        Self::new(TcfConfig::default())
    }
}

impl TcfPort for DeterministicTcf {
    fn step(&mut self, attn: &AttentionWeights, surprise: Option<&SurpriseSignal>) -> CyclePlan {
        let (freq, energy) = apply_ltv(self.config, &self.state, attn, surprise);
        let phase = apply_lti(self.state.phase, freq);
        let state = refresh_commit(TcfState {
            phase,
            energy,
            commit: self.state.commit,
        });
        self.state = state;

        let pulses = build_pulse_plan(attn, surprise);
        let cycle_id = self.cycle_id;
        self.cycle_id = self.cycle_id.saturating_add(1);
        let commit = commit_cycle_plan(cycle_id, &self.state, &pulses);

        CyclePlan {
            cycle_id,
            pulses,
            commit,
        }
    }

    fn state(&self) -> &TcfState {
        &self.state
    }
}

pub fn idle_attention() -> AttentionWeights {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.tcf.attn.idle.v1");
    let digest = Digest32::new(*hasher.finalize().as_bytes());
    AttentionWeights {
        channel: FocusChannel::Idle,
        gain: 1000,
        noise_suppress: 1000,
        replay_bias: 1000,
        commit: digest,
    }
}

pub fn encode_cycle_plan(plan: &CyclePlan) -> Vec<u8> {
    let mut payload = Vec::with_capacity(2 + 8 + Digest32::LEN + 1 + plan.pulses.len());
    payload.extend_from_slice(b"TC");
    payload.extend_from_slice(&plan.cycle_id.to_be_bytes());
    payload.extend_from_slice(plan.commit.as_bytes());
    payload.push(plan.pulses.len().min(u8::MAX as usize) as u8);
    for pulse in &plan.pulses {
        payload.push(pulse.kind as u8);
    }
    payload
}

fn apply_lti(phase: Phase, freq: u16) -> Phase {
    let mut next = phase.q.saturating_add(i32::from(freq));
    next %= PHASE_MAX;
    if next < 0 {
        next += PHASE_MAX;
    }
    Phase { q: next }
}

fn apply_ltv(
    config: TcfConfig,
    state: &TcfState,
    attn: &AttentionWeights,
    surprise: Option<&SurpriseSignal>,
) -> (u16, u16) {
    let surprise_score = surprise.map(|signal| signal.score).unwrap_or(0);
    let attn_gain = attn.gain;
    let freq_delta = (i32::from(attn_gain) / 20) + (i32::from(surprise_score) / 40);
    let jitter_guard = i32::from(config.jitter_guard);
    let freq_delta = freq_delta.clamp(0, jitter_guard);
    let freq = clamp_u16(i32::from(config.base_freq) + freq_delta);

    let mut energy = state.energy.saturating_sub(config.damping);
    energy = energy.saturating_add(attn_gain / 4);
    energy = energy.saturating_sub(surprise_score / 5);
    if energy > ENERGY_MAX {
        energy = ENERGY_MAX;
    }
    (freq, energy)
}

fn build_pulse_plan(attn: &AttentionWeights, surprise: Option<&SurpriseSignal>) -> Vec<Pulse> {
    let mut sense = 5000u16;
    let mut think = 4500u16;
    let mut verify = 3000u16;
    let mut consolidate = 2600u16;
    let broadcast = 2000u16;

    match attn.channel {
        FocusChannel::Threat => {
            verify = verify.saturating_add(2000);
            sense = sense.saturating_add(500);
        }
        FocusChannel::Memory => {
            consolidate = consolidate.saturating_add(600);
        }
        FocusChannel::Exploration => {
            sense = sense.saturating_add(400);
            verify = verify.saturating_add(400);
        }
        _ => {}
    }

    if let Some(signal) = surprise {
        match signal.band {
            SurpriseBand::High => {
                sense = sense.saturating_add(900);
                verify = verify.saturating_add(700);
                consolidate = consolidate.saturating_add(700);
            }
            SurpriseBand::Critical => {
                sense = sense.saturating_add(1400);
                verify = verify.saturating_add(1100);
                consolidate = consolidate.saturating_add(1100);
            }
            _ => {}
        }
    }

    consolidate = consolidate.saturating_add(attn.replay_bias / 6);
    think = think.saturating_add(attn.gain / 30);

    let mut pulses = vec![
        Pulse {
            kind: PulseKind::Sense,
            weight: sense,
            slot: 0,
        },
        Pulse {
            kind: PulseKind::Think,
            weight: think,
            slot: 0,
        },
        Pulse {
            kind: PulseKind::Verify,
            weight: verify,
            slot: 0,
        },
        Pulse {
            kind: PulseKind::Consolidate,
            weight: consolidate,
            slot: 0,
        },
        Pulse {
            kind: PulseKind::Broadcast,
            weight: broadcast,
            slot: 0,
        },
    ];

    if should_add_sleep(attn, surprise) {
        pulses.push(Pulse {
            kind: PulseKind::Sleep,
            weight: 1500,
            slot: 0,
        });
    }

    pulses.sort_by(|a, b| b.weight.cmp(&a.weight).then_with(|| a.kind.cmp(&b.kind)));
    pulses.truncate(MAX_PULSES);
    for (slot, pulse) in pulses.iter_mut().enumerate() {
        pulse.slot = slot as u8;
    }
    pulses
}

fn should_add_sleep(attn: &AttentionWeights, surprise: Option<&SurpriseSignal>) -> bool {
    if attn.replay_bias < 8000 {
        return false;
    }
    !matches!(
        surprise.map(|signal| signal.band),
        Some(SurpriseBand::High | SurpriseBand::Critical)
    )
}

fn commit_cycle_plan(cycle_id: u64, state: &TcfState, pulses: &[Pulse]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.tcf.cycle.plan.v1");
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(&state.phase.q.to_be_bytes());
    hasher.update(&state.energy.to_be_bytes());
    hasher.update(state.commit.as_bytes());
    for pulse in pulses {
        hasher.update(&[pulse.kind as u8, pulse.slot]);
        hasher.update(&pulse.weight.to_be_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn refresh_commit(mut state: TcfState) -> TcfState {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.tcf.state.v1");
    hasher.update(&state.phase.q.to_be_bytes());
    hasher.update(&state.energy.to_be_bytes());
    state.commit = Digest32::new(*hasher.finalize().as_bytes());
    state
}

fn clamp_u16(value: i32) -> u16 {
    value.clamp(0, i32::from(u16::MAX)) as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_attn_controller::FocusChannel;
    use ucf_predictive_coding::SurpriseBand;

    fn make_attention(channel: FocusChannel) -> AttentionWeights {
        let mut attn = idle_attention();
        attn.channel = channel;
        attn
    }

    fn make_surprise(band: SurpriseBand, score: u16) -> SurpriseSignal {
        SurpriseSignal {
            score,
            band,
            commit: Digest32::new([7u8; 32]),
        }
    }

    #[test]
    fn cycle_plan_is_deterministic_for_same_inputs() {
        let mut tcf_a = DeterministicTcf::default();
        let mut tcf_b = DeterministicTcf::default();
        let attn = make_attention(FocusChannel::Idle);
        let surprise = make_surprise(SurpriseBand::Low, 100);

        let plan_a = tcf_a.step(&attn, Some(&surprise));
        let plan_b = tcf_b.step(&attn, Some(&surprise));

        assert_eq!(plan_a, plan_b);
        assert_eq!(plan_a.commit, plan_b.commit);
    }

    #[test]
    fn threat_focus_moves_verify_earlier_and_heavier() {
        let mut tcf = DeterministicTcf::default();
        let idle = make_attention(FocusChannel::Idle);
        let threat = make_attention(FocusChannel::Threat);

        let idle_plan = tcf.step(&idle, None);
        let threat_plan = tcf.step(&threat, None);

        let idle_verify = idle_plan
            .pulses
            .iter()
            .find(|pulse| pulse.kind == PulseKind::Verify)
            .expect("idle verify");
        let threat_verify = threat_plan
            .pulses
            .iter()
            .find(|pulse| pulse.kind == PulseKind::Verify)
            .expect("threat verify");

        assert!(threat_verify.weight > idle_verify.weight);
        assert!(threat_verify.slot < idle_verify.slot);
    }

    #[test]
    fn surprise_critical_boosts_sense_and_consolidate() {
        let mut tcf = DeterministicTcf::default();
        let attn = make_attention(FocusChannel::Idle);

        let base_plan = tcf.step(&attn, None);
        let critical = make_surprise(SurpriseBand::Critical, 9_000);
        let critical_plan = tcf.step(&attn, Some(&critical));

        let base_sense = base_plan
            .pulses
            .iter()
            .find(|pulse| pulse.kind == PulseKind::Sense)
            .expect("sense");
        let base_consolidate = base_plan
            .pulses
            .iter()
            .find(|pulse| pulse.kind == PulseKind::Consolidate)
            .expect("consolidate");
        let crit_sense = critical_plan
            .pulses
            .iter()
            .find(|pulse| pulse.kind == PulseKind::Sense)
            .expect("sense");
        let crit_consolidate = critical_plan
            .pulses
            .iter()
            .find(|pulse| pulse.kind == PulseKind::Consolidate)
            .expect("consolidate");

        assert!(crit_sense.weight > base_sense.weight);
        assert!(crit_consolidate.weight > base_consolidate.weight);
    }
}
