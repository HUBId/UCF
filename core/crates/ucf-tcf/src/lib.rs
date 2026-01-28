#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_types::Digest32;

const PHASE_WRAP: u32 = 65_536;
const SLOT_COUNT: usize = 10;
const MAX_SIGNAL: i32 = 10_000;
const SCALE_FACTOR: i32 = 10_000;
const JITTER_THRESHOLD: i32 = 2_000;

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TickSlot {
    Phase0_Ingest,
    Phase1_Onn,
    Phase2_Influence,
    Phase3_Core,
    Phase4_CdeNsr,
    Phase5_Geist,
    Phase6_SleRsa,
    Phase7_SpikeBatch,
    Phase8_SleepReplay,
    Phase9_Output,
}

impl TickSlot {
    pub const ORDER: [TickSlot; SLOT_COUNT] = [
        TickSlot::Phase0_Ingest,
        TickSlot::Phase1_Onn,
        TickSlot::Phase2_Influence,
        TickSlot::Phase3_Core,
        TickSlot::Phase4_CdeNsr,
        TickSlot::Phase5_Geist,
        TickSlot::Phase6_SleRsa,
        TickSlot::Phase7_SpikeBatch,
        TickSlot::Phase8_SleepReplay,
        TickSlot::Phase9_Output,
    ];

    pub fn as_u8(self) -> u8 {
        match self {
            TickSlot::Phase0_Ingest => 0,
            TickSlot::Phase1_Onn => 1,
            TickSlot::Phase2_Influence => 2,
            TickSlot::Phase3_Core => 3,
            TickSlot::Phase4_CdeNsr => 4,
            TickSlot::Phase5_Geist => 5,
            TickSlot::Phase6_SleRsa => 6,
            TickSlot::Phase7_SpikeBatch => 7,
            TickSlot::Phase8_SleepReplay => 8,
            TickSlot::Phase9_Output => 9,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TickPlanEntry {
    pub slot: TickSlot,
    pub phase_begin: u16,
    pub phase_end: u16,
    pub commit: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TickPlan {
    pub cycle_id: u64,
    pub phase_commit: Digest32,
    pub entries: Vec<TickPlanEntry>,
    pub commit: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LtiFilter {
    pub a: i16,
    pub b: i16,
    pub y: i32,
    pub commit: Digest32,
}

impl LtiFilter {
    pub fn new(a: i16, b: i16, y: i32) -> Self {
        let mut filter = Self {
            a,
            b,
            y: y.clamp(0, MAX_SIGNAL),
            commit: Digest32::new([0u8; 32]),
        };
        filter.commit = commit_lti_filter(&filter);
        filter
    }

    fn step(&mut self, x: u16) -> (u16, bool) {
        let y_prev = self.y;
        let diff = i32::from(x) - y_prev;
        let jittered = diff.abs() > JITTER_THRESHOLD;
        let denom = i32::from(self.b.max(1));
        let delta = (i64::from(self.a) * i64::from(diff)) / i64::from(denom);
        let next = y_prev.saturating_add(delta as i32).clamp(0, MAX_SIGNAL);
        self.y = next;
        self.commit = commit_lti_filter(self);
        (next as u16, jittered)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TcfInputs {
    pub cycle_id: u64,
    pub prev_phase_commit: Digest32,
    pub onn_phase_frame_commit: Digest32,
    pub global_plv: u16,
    pub risk: u16,
    pub drift: u16,
    pub surprise: u16,
    pub attention_raw: u16,
    pub replay_pressure_raw: u16,
    pub ncde_energy_raw: u16,
    pub commit: Digest32,
}

impl TcfInputs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cycle_id: u64,
        prev_phase_commit: Digest32,
        onn_phase_frame_commit: Digest32,
        global_plv: u16,
        risk: u16,
        drift: u16,
        surprise: u16,
        attention_raw: u16,
        replay_pressure_raw: u16,
        ncde_energy_raw: u16,
    ) -> Self {
        let mut inputs = Self {
            cycle_id,
            prev_phase_commit,
            onn_phase_frame_commit,
            global_plv,
            risk,
            drift,
            surprise,
            attention_raw,
            replay_pressure_raw,
            ncde_energy_raw,
            commit: Digest32::new([0u8; 32]),
        };
        inputs.commit = commit_tcf_inputs(&inputs);
        inputs
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TcfOutputs {
    pub cycle_id: u64,
    pub tick_plan: TickPlan,
    pub attention_smooth: u16,
    pub replay_pressure_smooth: u16,
    pub ncde_energy_smooth: u16,
    pub jitter_flag: bool,
    pub commit: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TcfCore {
    pub att_f: LtiFilter,
    pub rep_f: LtiFilter,
    pub ene_f: LtiFilter,
    pub params_commit: Digest32,
    pub commit: Digest32,
}

impl Default for TcfCore {
    fn default() -> Self {
        let att_f = LtiFilter::new(2_000, SCALE_FACTOR as i16, 0);
        let rep_f = LtiFilter::new(2_000, SCALE_FACTOR as i16, 0);
        let ene_f = LtiFilter::new(2_000, SCALE_FACTOR as i16, 0);
        let params_commit = commit_tcf_params(&att_f, &rep_f, &ene_f);
        let mut core = Self {
            att_f,
            rep_f,
            ene_f,
            params_commit,
            commit: Digest32::new([0u8; 32]),
        };
        core.commit = commit_tcf_core(&core);
        core
    }
}

impl TcfCore {
    pub fn tick(&mut self, inp: &TcfInputs) -> TcfOutputs {
        let (attention_smooth, attn_jitter) = self.att_f.step(inp.attention_raw);
        let (replay_pressure_smooth, rep_jitter) = self.rep_f.step(inp.replay_pressure_raw);
        let (ncde_energy_smooth, ene_jitter) = self.ene_f.step(inp.ncde_energy_raw);
        let jitter_flag = attn_jitter || rep_jitter || ene_jitter;

        self.params_commit = commit_tcf_params(&self.att_f, &self.rep_f, &self.ene_f);

        let tick_plan = build_tick_plan(inp.cycle_id, inp.onn_phase_frame_commit, inp.global_plv);
        let mut outputs = TcfOutputs {
            cycle_id: inp.cycle_id,
            tick_plan,
            attention_smooth,
            replay_pressure_smooth,
            ncde_energy_smooth,
            jitter_flag,
            commit: Digest32::new([0u8; 32]),
        };
        outputs.commit = commit_tcf_outputs(&outputs);
        self.commit = commit_tcf_core(self);
        outputs
    }
}

fn build_tick_plan(cycle_id: u64, phase_commit: Digest32, global_plv: u16) -> TickPlan {
    let widths = phase_windows(global_plv);
    let mut entries = Vec::with_capacity(SLOT_COUNT);
    let mut cursor: u32 = 0;
    for (idx, slot) in TickSlot::ORDER.iter().enumerate() {
        let width = widths[idx] as u32;
        let begin = cursor.min(PHASE_WRAP - 1) as u16;
        let end = cursor
            .saturating_add(width)
            .saturating_sub(1)
            .min(PHASE_WRAP - 1) as u16;
        let mut entry = TickPlanEntry {
            slot: *slot,
            phase_begin: begin,
            phase_end: end,
            commit: Digest32::new([0u8; 32]),
        };
        entry.commit = commit_tick_plan_entry(&entry, phase_commit);
        entries.push(entry);
        cursor = cursor.saturating_add(width);
    }
    let mut plan = TickPlan {
        cycle_id,
        phase_commit,
        entries,
        commit: Digest32::new([0u8; 32]),
    };
    plan.commit = commit_tick_plan(&plan);
    plan
}

fn phase_windows(global_plv: u16) -> [u16; SLOT_COUNT] {
    let base = PHASE_WRAP / SLOT_COUNT as u32;
    let remainder = PHASE_WRAP % SLOT_COUNT as u32;
    let scale = phase_scale(global_plv);
    let mut widths = [0u16; SLOT_COUNT];
    for (idx, width) in widths.iter_mut().enumerate() {
        let bump = if idx < remainder as usize { 1 } else { 0 };
        let nominal = base + bump;
        let scaled = (nominal as u64 * scale as u64) / SCALE_FACTOR as u64;
        *width = scaled.max(1).min(u16::MAX as u64) as u16;
    }
    adjust_phase_sum(&mut widths);
    widths
}

fn phase_scale(global_plv: u16) -> i32 {
    let plv = i32::from(global_plv.min(10_000));
    let tighten = (plv * 2_000) / 10_000;
    let scale = 11_000 - tighten;
    scale.clamp(9_000, 11_000)
}

fn adjust_phase_sum(widths: &mut [u16; SLOT_COUNT]) {
    let total: i32 = widths.iter().map(|value| i32::from(*value)).sum();
    let mut delta = PHASE_WRAP as i32 - total;
    if delta == 0 {
        return;
    }
    let mut idx = 0usize;
    while delta != 0 {
        let slot = idx % SLOT_COUNT;
        if delta > 0 {
            widths[slot] = widths[slot].saturating_add(1);
            delta -= 1;
        } else if widths[slot] > 1 {
            widths[slot] = widths[slot].saturating_sub(1);
            delta += 1;
        }
        idx = idx.wrapping_add(1);
    }
}

fn commit_tick_plan_entry(entry: &TickPlanEntry, phase_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.tcf.tick_plan.entry.v1");
    hasher.update(&[entry.slot.as_u8()]);
    hasher.update(&entry.phase_begin.to_be_bytes());
    hasher.update(&entry.phase_end.to_be_bytes());
    hasher.update(phase_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_tick_plan(plan: &TickPlan) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.tcf.tick_plan.v1");
    hasher.update(&plan.cycle_id.to_be_bytes());
    hasher.update(plan.phase_commit.as_bytes());
    for entry in &plan.entries {
        hasher.update(entry.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_lti_filter(filter: &LtiFilter) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.tcf.lti_filter.v1");
    hasher.update(&filter.a.to_be_bytes());
    hasher.update(&filter.b.to_be_bytes());
    hasher.update(&filter.y.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_tcf_inputs(inputs: &TcfInputs) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.tcf.inputs.v1");
    hasher.update(&inputs.cycle_id.to_be_bytes());
    hasher.update(inputs.prev_phase_commit.as_bytes());
    hasher.update(inputs.onn_phase_frame_commit.as_bytes());
    hasher.update(&inputs.global_plv.to_be_bytes());
    hasher.update(&inputs.risk.to_be_bytes());
    hasher.update(&inputs.drift.to_be_bytes());
    hasher.update(&inputs.surprise.to_be_bytes());
    hasher.update(&inputs.attention_raw.to_be_bytes());
    hasher.update(&inputs.replay_pressure_raw.to_be_bytes());
    hasher.update(&inputs.ncde_energy_raw.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_tcf_outputs(outputs: &TcfOutputs) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.tcf.outputs.v1");
    hasher.update(&outputs.cycle_id.to_be_bytes());
    hasher.update(outputs.tick_plan.commit.as_bytes());
    hasher.update(&outputs.attention_smooth.to_be_bytes());
    hasher.update(&outputs.replay_pressure_smooth.to_be_bytes());
    hasher.update(&outputs.ncde_energy_smooth.to_be_bytes());
    hasher.update(&[outputs.jitter_flag as u8]);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_tcf_params(att_f: &LtiFilter, rep_f: &LtiFilter, ene_f: &LtiFilter) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.tcf.params.v1");
    hasher.update(&att_f.a.to_be_bytes());
    hasher.update(&att_f.b.to_be_bytes());
    hasher.update(&rep_f.a.to_be_bytes());
    hasher.update(&rep_f.b.to_be_bytes());
    hasher.update(&ene_f.a.to_be_bytes());
    hasher.update(&ene_f.b.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_tcf_core(core: &TcfCore) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.tcf.core.v1");
    hasher.update(core.att_f.commit.as_bytes());
    hasher.update(core.rep_f.commit.as_bytes());
    hasher.update(core.ene_f.commit.as_bytes());
    hasher.update(core.params_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_deterministic_for_same_inputs() {
        let inputs = TcfInputs::new(
            7,
            Digest32::new([1u8; 32]),
            Digest32::new([2u8; 32]),
            7200,
            1200,
            800,
            900,
            4000,
            1500,
            2300,
        );
        let mut core_a = TcfCore::default();
        let mut core_b = TcfCore::default();
        let out_a = core_a.tick(&inputs);
        let out_b = core_b.tick(&inputs);
        assert_eq!(out_a.commit, out_b.commit);
        assert_eq!(out_a.tick_plan.commit, out_b.tick_plan.commit);
    }

    #[test]
    fn lti_step_converges_monotonic() {
        let mut filter = LtiFilter::new(2_000, SCALE_FACTOR as i16, 0);
        let mut prev = 0;
        for _ in 0..8 {
            let (next, _) = filter.step(10_000);
            assert!(next >= prev);
            prev = next;
        }
    }

    #[test]
    fn jitter_flag_triggers_on_jump() {
        let inputs = TcfInputs::new(
            1,
            Digest32::new([0u8; 32]),
            Digest32::new([0u8; 32]),
            5000,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let mut core = TcfCore::default();
        let _ = core.tick(&inputs);
        let jump = TcfInputs::new(
            2,
            Digest32::new([0u8; 32]),
            Digest32::new([0u8; 32]),
            5000,
            0,
            0,
            0,
            9000,
            0,
            0,
        );
        let out = core.tick(&jump);
        assert!(out.jitter_flag);
    }

    #[test]
    fn tick_plan_non_overlapping() {
        let plan = build_tick_plan(1, Digest32::new([3u8; 32]), 8000);
        assert_eq!(plan.entries.len(), SLOT_COUNT);
        let mut cursor: u32 = 0;
        for entry in &plan.entries {
            assert_eq!(entry.phase_begin as u32, cursor);
            let width = entry.phase_end as u32 - entry.phase_begin as u32 + 1;
            cursor += width;
        }
        assert_eq!(cursor, PHASE_WRAP);
    }
}
