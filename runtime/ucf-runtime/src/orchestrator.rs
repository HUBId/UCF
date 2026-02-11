use crate::errors::RuntimeError;
use ucf_biophys::v0::{
    apply_coherence_feedback, classify, compute_integration, couple_pair, ensure_osc, hpa_step,
    modulate_hh, onn_step, osc_step, phase_bin, phase_lock, ttfs_from_strength, ttfs_phase,
    verify_graph, CausalGraph, CoherenceState, Edge, EventBus, FieldEvent, FieldEventKind,
    FieldUpdateCfg, HhParams, HpaCfg, HpaState, IITCfg as BiophysIITCfg, IITInputs,
    IITState as BiophysIITState, Microcircuit, ModulationCfg, NeuromodulatorField as BiophysField,
    OnnCfg, OnnState, Osc, OscId, PhaseCfg, RuleCfg, SnnSpikeEvent, SpikeCodecCfg, SpikeKind,
    VerifyVerdict,
};
use ucf_cde::v0::{
    on_intervention, on_observation, tick_decay, CdeCfg, CdeState, CdeUpdateKind, Intervention,
    Observation, VarId,
};
use ucf_core::archive_log::ArchiveLog;
use ucf_core::storage::{ArchiveCfg, FlushPolicy, MemArchiveStore};
use ucf_ess::v1::{ExperienceRecord, ExperienceStore, IdAllocator, InMemoryEss};
use ucf_frames::v1::{
    ArchiveAppendFrame, BiophysFrame, BiophysHhParams, CdeFrame, ControlFrame, DecisionFrame,
    IitFrame, MicrocircuitFrame, NcdeFrame, NeuromodulatorSnapshot, NsrFrame, OnnFrame, PhaseFrame,
    SleFrame, SnnFrame, SsmFrame, TcfFrame,
};
use ucf_iit_proxy::v0::{iit_push_and_eval, IitCfg, IitSample, IitState};
use ucf_ncde::v0::{ncde_step, NcdeCfg, NcdeInput, NcdeState};
use ucf_neuromod::v0::{NeuromodInputs, NeuromodScheduler, NeuromodulatorField};
use ucf_nsr::v0::{Claim, NsrEngine, Verdict};
use ucf_onn::v0::{OnnCore, PhaseDeg};
use ucf_policy::{adapter::ActionAdapter, gem::Gem, pbm::Pbm};
use ucf_sle::v0::{sle_step, SleCfg, SleReason, SleSignals, SleState};
use ucf_snn::v0::{encode, to_brainbus, FeatureEvent, SnnEncodeCfg};
use ucf_spikes::{
    encode_ttfs_us, filter_phase_locked, PhaseLockCfg, Spike as BusSpike, SpikeBus,
    SpikeKind as BusSpikeKind,
};
use ucf_ssm::v0::{ssm_step, SsmCfg, SsmState};
use ucf_tcf::v0::{tcf_tick, TcfCfg, TcfState};

const OSC_SSM: OscId = 1;
const OSC_CDE: OscId = 2;
const OSC_NSR: OscId = 3;
const OSC_COHERENCE: OscId = 4;
const OSC_NSR_TCF_ENFORCE: u8 = 1;
const MOD_PBM: ucf_onn::v0::OscId = 13;

const ONN_COUPLINGS: &[(OscId, OscId, f32)] = &[
    (OSC_SSM, OSC_CDE, 1.0),
    (OSC_CDE, OSC_SSM, 1.0),
    (OSC_CDE, OSC_NSR, 1.0),
    (OSC_NSR, OSC_CDE, 1.0),
    (OSC_NSR, OSC_COHERENCE, 1.0),
    (OSC_COHERENCE, OSC_NSR, 1.0),
    (OSC_SSM, OSC_COHERENCE, 0.5),
    (OSC_COHERENCE, OSC_SSM, 0.5),
];

#[derive(Clone, Copy, Debug, Default)]
struct PhaseBus {
    osc_jepa: Osc,
    osc_nsr: Osc,
    osc_microcircuit: Osc,
}

#[derive(Clone, Debug, Default)]
struct WorkingContext {
    ssm_y: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyDecisionKind {
    InternalRecursion,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolicyDecision {
    pub kind: PolicyDecisionKind,
    pub allow: bool,
    pub max_depth: u8,
}

#[derive(Clone, Debug, PartialEq)]
struct MetaInput {
    tokens: Vec<u32>,
    weight: f32,
    depth: u8,
    reason: SleReason,
}

pub struct RuntimeOrchestrator {
    pub ess: InMemoryEss,
    pub ids: IdAllocator,
    pub pbm: Pbm,
    pub gem: Gem,
    neuromod_field: NeuromodulatorField,
    neuromod_scheduler: NeuromodScheduler,
    onn: OnnCore,
    snn_encode_cfg: SnnEncodeCfg,
    snn_tick_counter: u64,
    last_snn_spike_count: usize,
    biophys_field: BiophysField,
    biophys_cfg: FieldUpdateCfg,
    biophys_mod_cfg: ModulationCfg,
    hpa_state: HpaState,
    hpa_cfg: HpaCfg,
    last_biophys_tick_ms: Option<u64>,
    last_biophys_frame: Option<BiophysFrame>,
    microcircuit: Microcircuit,
    last_microcircuit_frame: Option<MicrocircuitFrame>,
    phase_bus: PhaseBus,
    phase_cfg: PhaseCfg,
    spike_codec_cfg: SpikeCodecCfg,
    last_phase_frame: Option<PhaseFrame>,
    onn_state: OnnState,
    onn_cfg: OnnCfg,
    event_bus: EventBus,
    spike_bus: SpikeBus,
    spike_seq: u32,
    last_onn_frame: Option<OnnFrame>,
    tcf_cfg: TcfCfg,
    tcf_state: TcfState,
    last_tcf_frame: Option<TcfFrame>,
    forced_mean_lock_for_test: Option<f32>,
    last_snn_frame: Option<SnnFrame>,
    last_spike_frames: Vec<ucf_frames::v1::SpikeFrame>,
    attention_event: bool,
    iit_cfg: BiophysIITCfg,
    iit_state: BiophysIITState,
    iit_proxy_cfg: IitCfg,
    iit_proxy_state: IitState,
    last_iit_frame: Option<IitFrame>,
    cde_graph: CausalGraph,
    cde_cfg: CdeCfg,
    cde_state: CdeState,
    nsr_cfg: RuleCfg,
    nsr_engine: NsrEngine,
    last_cde_frame: Option<CdeFrame>,
    last_nsr_frame: Option<NsrFrame>,
    ssm_state: SsmState,
    ssm_cfg: SsmCfg,
    working_context: WorkingContext,
    last_ssm_frame: Option<SsmFrame>,
    sle_cfg: SleCfg,
    sle_state: SleState,
    last_sle_frame: Option<SleFrame>,
    last_meta_input: Option<MetaInput>,
    policy_internal_recursion_allow: bool,
    policy_internal_recursion_max_depth: u8,
    ssm_gate: f32,
    ncde_cfg: NcdeCfg,
    ncde_state: NcdeState,
    last_ncde_frame: Option<NcdeFrame>,
    archive: ArchiveLog<MemArchiveStore>,
    last_archive_append_frame: Option<ArchiveAppendFrame>,
    last_archive_payload_len: usize,
}

impl RuntimeOrchestrator {
    pub fn new() -> Self {
        let mut onn = OnnCore::new(1.0, 0.0);
        onn.register(MOD_PBM, PhaseDeg(0.0));

        let mut onn_state = OnnState::default();
        ensure_osc(&mut onn_state, OSC_SSM);
        ensure_osc(&mut onn_state, OSC_CDE);
        ensure_osc(&mut onn_state, OSC_NSR);
        ensure_osc(&mut onn_state, OSC_COHERENCE);

        let ssm_cfg = SsmCfg::default_small();
        let ssm_state = SsmState::new(&ssm_cfg, 0);
        let ncde_cfg = NcdeCfg::default_v0();
        let ncde_state = NcdeState::new(&ncde_cfg);

        Self {
            ess: InMemoryEss::new(),
            ids: IdAllocator::new(1),
            pbm: Pbm,
            gem: Gem,
            neuromod_field: NeuromodulatorField::new_baseline(),
            neuromod_scheduler: NeuromodScheduler::new(1),
            onn,
            snn_encode_cfg: SnnEncodeCfg::default(),
            snn_tick_counter: 0,
            last_snn_spike_count: 0,
            biophys_field: BiophysField::default(),
            biophys_cfg: FieldUpdateCfg::default(),
            biophys_mod_cfg: ModulationCfg::default(),
            hpa_state: HpaState::default(),
            hpa_cfg: HpaCfg::default(),
            last_biophys_tick_ms: None,
            last_biophys_frame: None,
            microcircuit: Microcircuit::new_ring(32),
            last_microcircuit_frame: None,
            phase_bus: PhaseBus::default(),
            phase_cfg: PhaseCfg::default(),
            spike_codec_cfg: SpikeCodecCfg::default(),
            last_phase_frame: None,
            onn_state,
            onn_cfg: OnnCfg::default(),
            event_bus: EventBus::default(),
            spike_bus: SpikeBus::default(),
            spike_seq: 0,
            last_onn_frame: None,
            tcf_cfg: TcfCfg::default_gamma40(),
            tcf_state: TcfState::new(&TcfCfg::default_gamma40(), 0),
            last_tcf_frame: None,
            forced_mean_lock_for_test: None,
            last_snn_frame: None,
            last_spike_frames: Vec::new(),
            attention_event: false,
            iit_cfg: BiophysIITCfg::default(),
            iit_state: BiophysIITState::default(),
            iit_proxy_cfg: IitCfg::default_v0(),
            iit_proxy_state: IitState::new(&IitCfg::default_v0()),
            last_iit_frame: None,
            cde_graph: CausalGraph::default(),
            cde_cfg: CdeCfg::default(),
            cde_state: CdeState::default(),
            nsr_cfg: RuleCfg::default(),
            nsr_engine: NsrEngine::with_default_rules(),
            last_cde_frame: None,
            last_nsr_frame: None,
            ssm_state,
            ssm_cfg,
            working_context: WorkingContext::default(),
            last_ssm_frame: None,
            sle_cfg: SleCfg::default_v0(),
            sle_state: SleState::new(),
            last_sle_frame: None,
            last_meta_input: None,
            policy_internal_recursion_allow: true,
            policy_internal_recursion_max_depth: SleCfg::default_v0().max_depth,
            ssm_gate: 0.0,
            ncde_cfg,
            ncde_state,
            last_ncde_frame: None,
            archive: ArchiveLog::new(MemArchiveStore::new(), ArchiveCfg::default()),
            last_archive_append_frame: None,
            last_archive_payload_len: 0,
        }
    }

    pub fn last_snn_spike_count(&self) -> usize {
        self.last_snn_spike_count
    }

    pub fn last_biophys_frame(&self) -> Option<BiophysFrame> {
        self.last_biophys_frame
    }

    pub fn last_microcircuit_frame(&self) -> Option<MicrocircuitFrame> {
        self.last_microcircuit_frame
    }

    pub fn last_phase_frame(&self) -> Option<PhaseFrame> {
        self.last_phase_frame
    }

    pub fn last_iit_frame(&self) -> Option<IitFrame> {
        self.last_iit_frame
    }

    pub fn last_onn_frame(&self) -> Option<OnnFrame> {
        self.last_onn_frame
    }

    pub fn last_tcf_frame(&self) -> Option<TcfFrame> {
        self.last_tcf_frame
    }

    pub fn last_snn_frame(&self) -> Option<SnnFrame> {
        self.last_snn_frame
    }

    pub fn set_onn_coupling_for_test(&mut self, coupling: f32) {
        self.onn_cfg.coupling = coupling.clamp(0.0, 1.0);
    }

    pub fn force_mean_lock_for_test(&mut self, mean_lock: f32) {
        self.forced_mean_lock_for_test = Some(mean_lock.clamp(0.0, 1.0));
    }

    pub fn set_iit_proxy_cfg_for_test(&mut self, cfg: IitCfg) {
        self.iit_proxy_cfg = cfg.clone();
        self.iit_proxy_state = IitState::new(&cfg);
    }

    pub fn drain_event_bus_for_test(&mut self) -> Vec<SnnSpikeEvent> {
        self.event_bus.drain()
    }

    pub fn inject_spike_for_test(&mut self, spike: BusSpike) {
        self.spike_bus.push(spike);
    }

    pub fn last_cde_frame(&self) -> Option<CdeFrame> {
        self.last_cde_frame
    }

    pub fn last_nsr_frame(&self) -> Option<NsrFrame> {
        self.last_nsr_frame
    }

    pub fn last_ssm_frame(&self) -> Option<SsmFrame> {
        self.last_ssm_frame
    }

    pub fn last_sle_frame(&self) -> Option<SleFrame> {
        self.last_sle_frame
    }

    pub fn set_internal_recursion_policy_for_test(&mut self, allow: bool, max_depth: u8) {
        self.policy_internal_recursion_allow = allow;
        self.policy_internal_recursion_max_depth = max_depth.max(1);
    }

    pub fn set_sle_cfg_for_test(&mut self, cfg: SleCfg) {
        self.sle_cfg = cfg;
        self.policy_internal_recursion_max_depth = self
            .policy_internal_recursion_max_depth
            .min(self.sle_cfg.max_depth.max(1));
    }

    pub fn last_spike_frames(&self) -> &[ucf_frames::v1::SpikeFrame] {
        &self.last_spike_frames
    }

    pub fn attention_event(&self) -> bool {
        self.attention_event
    }

    pub fn last_archive_append_frame(&self) -> Option<ArchiveAppendFrame> {
        self.last_archive_append_frame
    }

    pub fn ssm_gate(&self) -> f32 {
        self.ssm_gate
    }

    pub fn working_context_ssm_y(&self) -> &[f32] {
        &self.working_context.ssm_y
    }

    pub fn last_ncde_frame(&self) -> Option<NcdeFrame> {
        self.last_ncde_frame
    }

    pub fn ncde_l2_norm_0_1(&self) -> f32 {
        self.last_ncde_frame
            .map(|frame| f32::from(frame.l2_q) / 255.0)
            .unwrap_or(0.0)
    }

    pub fn archive_last_seq(&self) -> u64 {
        self.archive.last_seq().unwrap_or(0)
    }

    pub fn last_archive_payload_len(&self) -> usize {
        self.last_archive_payload_len
    }

    pub fn force_causal_cycle_for_test(&mut self, now_ms: u64) {
        self.cde_state.hyps.clear();
        for (src, dst) in [(1_u16, 2_u16), (2, 3), (3, 1)] {
            self.cde_state.hyps.push(ucf_cde::v0::Hypothesis {
                edge: ucf_cde::v0::Edge { src, dst },
                score: 0.9,
                conf: 0.9,
                last_update_ms: now_ms,
                seen_obs: 0,
                seen_int: 1,
            });
        }
        sync_graph_from_cde_state(&mut self.cde_graph, &self.cde_state);
        self.cde_graph
            .upsert_hypothesis(Edge { from: 1, to: 2 }, now_ms, 0.2);
        self.cde_graph
            .upsert_hypothesis(Edge { from: 2, to: 3 }, now_ms, 0.2);
        self.cde_graph
            .upsert_hypothesis(Edge { from: 3, to: 1 }, now_ms, 0.2);
    }

    pub fn feed_cde_intervention_for_test(
        &mut self,
        now_ms: u64,
        do_set: Vec<(VarId, f32)>,
        measured: Vec<(VarId, f32)>,
    ) -> CdeUpdateKind {
        on_intervention(
            &mut self.cde_state,
            self.cde_cfg,
            Intervention {
                now_ms,
                do_set,
                measured,
            },
        )
    }

    pub fn ingest_and_process<A: ActionAdapter>(
        &mut self,
        adapter: &mut A,
        ctrl: ControlFrame,
    ) -> Result<DecisionFrame, RuntimeError> {
        self.update_biophys_tick(ctrl.time.tick.get());

        let inputs = NeuromodInputs::baseline();
        self.neuromod_scheduler
            .advance(ctrl.time.tick.get(), &mut self.neuromod_field, inputs);
        self.onn.step_ms(1);
        let snapshot = self.neuromod_field.snapshot();
        let phi = self.iit_phi_snapshot();

        let decision = Pbm::decide(&ctrl, Some(snapshot));
        self.ingest_with_decision_and_snapshot(adapter, ctrl, decision, snapshot, phi)
    }

    pub fn ingest_with_decision<A: ActionAdapter>(
        &mut self,
        adapter: &mut A,
        ctrl: ControlFrame,
        decision: DecisionFrame,
    ) -> Result<DecisionFrame, RuntimeError> {
        self.update_biophys_tick(ctrl.time.tick.get());

        let inputs = NeuromodInputs::baseline();
        self.neuromod_scheduler
            .advance(ctrl.time.tick.get(), &mut self.neuromod_field, inputs);
        self.onn.step_ms(1);
        let snapshot = self.neuromod_field.snapshot();
        let phi = self.iit_phi_snapshot();
        self.ingest_with_decision_and_snapshot(adapter, ctrl, decision, snapshot, phi)
    }

    fn decide_internal_recursion(&self) -> PolicyDecision {
        PolicyDecision {
            kind: PolicyDecisionKind::InternalRecursion,
            allow: self.policy_internal_recursion_allow,
            max_depth: self
                .policy_internal_recursion_max_depth
                .min(self.sle_cfg.max_depth.max(1)),
        }
    }

    fn update_biophys_tick(&mut self, now_ms: u64) {
        let dt_ms = self
            .last_biophys_tick_ms
            .map(|last| now_ms.saturating_sub(last) as u32)
            .unwrap_or(1)
            .max(1);
        self.last_biophys_tick_ms = Some(now_ms);
        let dt_s = (dt_ms as f32 / 1000.0).max(0.001);

        self.phase_bus.osc_jepa = osc_step(self.phase_bus.osc_jepa, dt_s);
        self.phase_bus.osc_nsr = osc_step(self.phase_bus.osc_nsr, dt_s);
        self.phase_bus.osc_microcircuit = osc_step(self.phase_bus.osc_microcircuit, dt_s);

        (self.phase_bus.osc_microcircuit, self.phase_bus.osc_nsr) = couple_pair(
            self.phase_bus.osc_microcircuit,
            self.phase_bus.osc_nsr,
            dt_s,
            self.phase_cfg,
        );
        (self.phase_bus.osc_nsr, self.phase_bus.osc_jepa) = couple_pair(
            self.phase_bus.osc_nsr,
            self.phase_bus.osc_jepa,
            dt_s,
            self.phase_cfg,
        );

        let onn_out = onn_step(&mut self.onn_state, dt_ms, self.onn_cfg, ONN_COUPLINGS);
        self.last_onn_frame = Some(OnnFrame {
            now_ms,
            global_phase: onn_out.global_phase,
            mean_lock: onn_out.mean_lock,
        });

        let coupling_targets = if self.iit_proxy_state.enforce {
            vec![(OSC_NSR_TCF_ENFORCE, self.tcf_state.global_phase)]
        } else {
            Vec::new()
        };
        let mut tcf_out = tcf_tick(
            &self.tcf_cfg,
            &mut self.tcf_state,
            now_ms,
            &coupling_targets,
        );
        if let Some(forced) = self.forced_mean_lock_for_test {
            tcf_out.mean_lock = forced.clamp(0.0, 1.0);
        }
        self.last_tcf_frame = Some(TcfFrame::from_metrics(
            tcf_out.now_ms,
            tcf_out.global_phase,
            tcf_out.mean_lock,
            tcf_out.jitter,
            tcf_out.phase_spread,
        ));

        let baseline = BiophysField::default();
        self.biophys_field = self
            .biophys_field
            .decay_towards(baseline, dt_s, self.biophys_cfg);

        if now_ms.is_multiple_of(10) {
            self.biophys_field = self.biophys_field.apply_event(
                FieldEvent {
                    kind: FieldEventKind::Reward,
                    magnitude: ucf_biophys::v0::Unit01::new(0.4),
                },
                self.biophys_cfg,
            );
        }

        if now_ms.is_multiple_of(15) {
            self.biophys_field = self.biophys_field.apply_event(
                FieldEvent {
                    kind: FieldEventKind::Stress,
                    magnitude: ucf_biophys::v0::Unit01::new(0.3),
                },
                self.biophys_cfg,
            );
        }

        let stress = if now_ms.is_multiple_of(15) { 0.7 } else { 0.05 };
        self.hpa_state = hpa_step(self.hpa_state, stress, dt_s, self.hpa_cfg);

        let field = self.biophys_field.with_hpa(self.hpa_state);
        let hh = modulate_hh(HhParams::default(), field, self.biophys_mod_cfg);
        let micro = self.microcircuit.step(field, dt_s);

        let _ttfs_zero_phase = ttfs_phase(0, self.spike_codec_cfg);

        let lock_nsr_jepa = phase_lock(self.phase_bus.osc_nsr, self.phase_bus.osc_jepa);
        let lock_micro_nsr = phase_lock(self.phase_bus.osc_microcircuit, self.phase_bus.osc_nsr);

        let spike_rate_hz = micro.spikes.len() as f32 / dt_s;
        let ssm_gate = self
            .last_ssm_frame
            .map(|frame| f32::from(frame.gate_q) / 255.0)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let nsr_verified_ratio = self
            .last_nsr_frame
            .map(|frame| f32::from(frame.verified_q) / 255.0)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let archive_append_bytes_q =
            ((self.last_archive_payload_len as f32) / 1024.0).clamp(0.0, 1.0);

        let obs_update = on_observation(
            &mut self.cde_state,
            self.cde_cfg,
            Observation {
                now_ms,
                vars: vec![
                    (1, tcf_out.mean_lock.clamp(0.0, 1.0)),
                    (2, ssm_gate),
                    (3, nsr_verified_ratio),
                    (4, archive_append_bytes_q),
                ],
            },
        );
        let decay_update = tick_decay(&mut self.cde_state, self.cde_cfg, now_ms);
        sync_graph_from_cde_state(&mut self.cde_graph, &self.cde_state);

        let (legacy_verdict, _legacy_verified_ratio) = verify_graph(&self.cde_graph, self.nsr_cfg);
        let top_conf = self
            .cde_state
            .hyps
            .iter()
            .map(|h| h.conf)
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(0.0);

        let planned_intent = self
            .last_ssm_frame
            .map(|frame| {
                if frame.gate_q > 128 {
                    "act".to_string()
                } else {
                    "noop".to_string()
                }
            })
            .unwrap_or_else(|| "noop".to_string());
        let claim = Claim {
            intent: planned_intent,
            tool: "mock".to_string(),
            channel: "terminal".to_string(),
            risk: if tcf_out.mean_lock < 0.25 {
                "high".to_string()
            } else {
                "low".to_string()
            },
            audience: "research".to_string(),
        };
        let nsr_result = self.nsr_engine.check(&claim);
        let verified_ratio = if nsr_result.total == 0 {
            0.0
        } else {
            f32::from(nsr_result.satisfied) / f32::from(nsr_result.total)
        };
        let nsr_risk = (1.0 - verified_ratio).clamp(0.0, 1.0);
        let verified_q = (verified_ratio * 255.0).round() as u8;

        let mut integration = compute_integration(
            IITInputs {
                lock_nsr_jepa,
                lock_micro_nsr,
                spike_rate_hz,
            },
            self.iit_cfg,
        );

        if legacy_verdict == VerifyVerdict::Verified && verified_ratio >= 0.85 {
            integration = (integration + 0.05).clamp(0.0, 1.0);
        }

        let alpha = self.iit_cfg.ema_alpha.clamp(0.0, 1.0);
        self.iit_state.integration_ema =
            (1.0 - alpha) * self.iit_state.integration_ema + alpha * integration;

        let mut coherence_state = classify(self.iit_state.integration_ema, self.iit_cfg);
        if nsr_result.verdict == Verdict::Block {
            coherence_state = CoherenceState::Fragmenting;
        }

        apply_coherence_feedback(
            &mut self.biophys_field,
            self.iit_state.integration_ema,
            coherence_state,
        );
        let top_conf_from_cde = top_conf.clamp(0.0, 1.0);
        let archive_append_bytes_q =
            ((self.last_archive_payload_len as f32) / 1024.0).clamp(0.0, 1.0);

        let spike_novelty = self
            .last_spike_frames
            .iter()
            .filter(|spike| spike.kind == 1)
            .map(|spike| f32::from(spike.strength_q) / 255.0)
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let sle_sig = SleSignals {
            now_ms,
            nsr_risk,
            cde_conf: top_conf_from_cde,
            phi_proxy: self.iit_proxy_state.phi_proxy.clamp(0.0, 1.0),
            spike_novelty,
            attention_event: self.attention_event,
        };
        let sle_out = sle_step(&self.sle_cfg, &mut self.sle_state, &sle_sig);
        let policy = self.decide_internal_recursion();
        let mut meta_weight = 0.0_f32;
        let mut meta_reason_norm = 0.0_f32;
        let mut sle_frame = SleFrame {
            now_ms,
            fired: 0,
            reason: 0,
            depth: 0,
            weight_q: 0,
            tok_n: 0,
        };
        self.last_meta_input = None;
        if let Some(meta) = sle_out {
            sle_frame.fired = 1;
            let reason_id = match meta.reason {
                SleReason::HighRisk => 1,
                SleReason::LowIntegration => 2,
                SleReason::NoveltyShock => 3,
                SleReason::Conflict => 4,
            };
            sle_frame.reason = reason_id;
            sle_frame.tok_n = meta.tokens.len().min(255) as u8;
            let mut depth = meta.depth.min(policy.max_depth.max(1));
            if depth == 0 {
                depth = 1;
            }
            sle_frame.depth = depth;
            if policy.allow {
                let weight = meta.weight.clamp(0.0, 1.0);
                meta_weight = weight;
                meta_reason_norm = (reason_id as f32 / 4.0).clamp(0.0, 1.0);
                sle_frame.weight_q = (weight * 255.0).round() as u8;
                self.last_meta_input = Some(MetaInput {
                    tokens: meta.tokens,
                    weight,
                    depth,
                    reason: meta.reason,
                });
            }
        }
        self.last_sle_frame = Some(sle_frame);

        let input = [
            tcf_out.mean_lock.clamp(0.0, 1.0),
            verified_ratio.clamp(0.0, 1.0),
            top_conf_from_cde,
            archive_append_bytes_q,
            if self.attention_event { 1.0 } else { 0.0 },
            meta_weight,
            meta_reason_norm,
            0.0,
        ];
        let gate = tcf_out.mean_lock.clamp(0.0, 1.0);
        let mut ssm_out = ssm_step(&self.ssm_cfg, &mut self.ssm_state, now_ms, &input, gate);
        self.working_context.ssm_y = ssm_out.y.clone();
        let ssm_energy = if ssm_out.y.is_empty() {
            0.0
        } else {
            ssm_out.y.iter().map(|v| v.abs()).sum::<f32>() / (ssm_out.y.len() as f32)
        };

        let tcf_phase_bin = self
            .last_tcf_frame
            .map(|frame| frame.phase_bin)
            .unwrap_or_else(|| phase_bin(tcf_out.global_phase, 255));

        if ssm_out.gate > 0.6 {
            let strength = ((ssm_out.gate - 0.6) / 0.4).clamp(0.0, 1.0);
            self.spike_bus.push(BusSpike {
                now_ms,
                kind: BusSpikeKind::Novelty,
                chan: OSC_SSM as u8,
                phase: tcf_phase_bin,
                strength,
                ttfs_us: encode_ttfs_us(strength),
            });
        }

        if matches!(obs_update, CdeUpdateKind::Updated { .. }) {
            let strength = top_conf.clamp(0.0, 1.0);
            self.spike_bus.push(BusSpike {
                now_ms,
                kind: BusSpikeKind::CausalHit,
                chan: OSC_CDE as u8,
                phase: tcf_phase_bin,
                strength,
                ttfs_us: encode_ttfs_us(strength),
            });
        }

        if matches!(nsr_result.verdict, Verdict::Allow | Verdict::Block) {
            let strength = verified_ratio.clamp(0.0, 1.0);
            self.spike_bus.push(BusSpike {
                now_ms,
                kind: BusSpikeKind::Verify,
                chan: OSC_NSR as u8,
                phase: tcf_phase_bin,
                strength,
                ttfs_us: encode_ttfs_us(strength),
            });
        }

        if tcf_out.mean_lock > 0.7 {
            let strength = tcf_out.mean_lock.clamp(0.0, 1.0);
            self.spike_bus.push(BusSpike {
                now_ms,
                kind: BusSpikeKind::AttentionShift,
                chan: OSC_COHERENCE as u8,
                phase: tcf_phase_bin,
                strength,
                ttfs_us: encode_ttfs_us(strength),
            });
        }

        let tick_spikes = filter_phase_locked(
            &PhaseLockCfg {
                max_dist: 24,
                attenuate: true,
            },
            tcf_phase_bin,
            &self.spike_bus.drain(),
        );

        self.attention_event = tick_spikes.iter().any(|spike| {
            spike.strength > 0.6
                && matches!(
                    spike.kind,
                    BusSpikeKind::Novelty | BusSpikeKind::AttentionShift
                )
        });

        self.last_spike_frames = tick_spikes
            .iter()
            .map(|spike| ucf_frames::v1::SpikeFrame {
                now_ms: spike.now_ms,
                kind: match spike.kind {
                    BusSpikeKind::Novelty => 1,
                    BusSpikeKind::Verify => 2,
                    BusSpikeKind::CausalHit => 3,
                    BusSpikeKind::MemoryMark => 4,
                    BusSpikeKind::AttentionShift => 5,
                },
                chan: spike.chan,
                phase: spike.phase,
                strength_q: (spike.strength.clamp(0.0, 1.0) * 255.0).round() as u8,
                ttfs_q: ((spike.ttfs_us as f32 / 5000.0).clamp(0.0, 1.0) * 255.0).round() as u8,
            })
            .collect();

        let mut feature = 0_u32;
        let mut causal = 0_u32;
        let mut verify = 0_u32;
        let mut attention_count = 0_u32;
        for spike in &tick_spikes {
            let mapped = match spike.kind {
                BusSpikeKind::Novelty => {
                    feature += 1;
                    SpikeKind::Feature
                }
                BusSpikeKind::Verify => {
                    verify += 1;
                    SpikeKind::Verify
                }
                BusSpikeKind::CausalHit => {
                    causal += 1;
                    SpikeKind::Causal
                }
                BusSpikeKind::MemoryMark => continue,
                BusSpikeKind::AttentionShift => {
                    attention_count += 1;
                    SpikeKind::Attention
                }
            };
            let delivery = self.make_spike(
                spike.now_ms,
                spike.chan.into(),
                mapped,
                spike.phase,
                spike.strength,
                Some(((spike.ttfs_us as f32 / 5000.0).clamp(0.0, 1.0) * 255.0).round() as u8),
            );
            self.event_bus.push(delivery);
        }
        self.last_snn_frame = Some(SnnFrame {
            now_ms,
            spikes: tick_spikes.len() as u32,
            feature,
            causal,
            verify,
            attention: attention_count,
        });

        self.last_biophys_frame = Some(BiophysFrame {
            now_ms,
            field: ucf_biophys::v0::summarize(field),
            hh_params: BiophysHhParams {
                g_na: hh.g_na,
                g_k: hh.g_k,
                g_l: hh.g_l,
                threshold_shift_mv: hh.threshold_shift_mv,
                max_firing_hz: hh.max_firing_hz,
            },
            hpa_cortisol: self.hpa_state.cortisol,
        });
        self.last_microcircuit_frame = Some(MicrocircuitFrame {
            now_ms,
            n: self.microcircuit.neurons.len() as u32,
            spike_count: micro.spikes.len() as u32,
            avg_v: micro.avg_v,
        });
        self.last_phase_frame = Some(PhaseFrame {
            now_ms,
            jepa_phase: self.phase_bus.osc_jepa.phase,
            nsr_phase: self.phase_bus.osc_nsr.phase,
            micro_phase: self.phase_bus.osc_microcircuit.phase,
            lock_nsr_jepa,
            lock_micro_nsr,
        });
        let spike_rate = (tick_spikes.len().min(32) as f32) / 32.0;
        let tcf_phase_norm = self.current_tcf_phase_0_1(now_ms);
        let ncde_inp = NcdeInput {
            u: vec![
                (1.0 - nsr_risk).clamp(0.0, 1.0),
                top_conf.clamp(0.0, 1.0),
                self.iit_proxy_state.phi_proxy.clamp(0.0, 1.0),
                ssm_out.gate.clamp(0.0, 1.0),
                spike_rate.clamp(0.0, 1.0),
                meta_weight.clamp(0.0, 1.0),
            ],
            phase: tcf_phase_norm,
        };
        let ncde_tick = ncde_step(&self.ncde_cfg, &mut self.ncde_state, now_ms, &ncde_inp);
        self.last_ncde_frame = Some(NcdeFrame {
            now_ms: ncde_tick.now_ms,
            l2_q: ncde_tick.l2_q,
            phase_q: ncde_tick.phase_q,
        });

        let iit_tick = iit_push_and_eval(
            &self.iit_proxy_cfg,
            &mut self.iit_proxy_state,
            IitSample {
                now_ms,
                tcf_lock: tcf_out.mean_lock.clamp(0.0, 1.0),
                ssm_gate: ssm_out.gate.clamp(0.0, 1.0),
                nsr_risk,
                cde_conf: top_conf.clamp(0.0, 1.0),
                spike_rate,
            },
        );
        if iit_tick.enforce {
            let _nsr_risk_floor = nsr_risk.max(0.8);
            ssm_out.gate = (ssm_out.gate * 0.5).clamp(0.0, 1.0);
        }

        self.ssm_gate = ssm_out.gate;
        self.last_ssm_frame = Some(SsmFrame {
            now_ms,
            gate_q: quantize_unit(ssm_out.gate),
            energy_q: quantize_unit(ssm_energy),
        });

        self.last_iit_frame = Some(IitFrame {
            now_ms,
            phi_q: quantize_unit(iit_tick.phi_proxy),
            coh_q: quantize_unit(iit_tick.coherence),
            flow_q: quantize_unit(iit_tick.flow),
            enforce: u8::from(iit_tick.enforce),
        });
        let changed = match obs_update {
            CdeUpdateKind::Updated { changed, .. } => changed as u16,
            _ => 0,
        };
        let pruned = match decay_update {
            CdeUpdateKind::Pruned { pruned } => pruned as u16,
            _ => 0,
        };
        self.last_cde_frame = Some(CdeFrame {
            now_ms,
            hyps: self.cde_state.hyps.len() as u16,
            changed,
            pruned,
            top_conf_q: (top_conf.clamp(0.0, 1.0) * 255.0).round() as u8,
        });
        self.last_nsr_frame = Some(NsrFrame {
            now_ms,
            verdict: match nsr_result.verdict {
                Verdict::Allow => 1,
                Verdict::Block => 2,
                Verdict::Unknown => 0,
            },
            satisfied: nsr_result.satisfied,
            total: nsr_result.total,
            verified_q,
        });

        let payload =
            tick_summary_payload(now_ms, coherence_state, tcf_out.mean_lock, ssm_out.gate);
        let flushed = match self.archive.cfg.flush {
            FlushPolicy::EveryAppend => true,
            FlushPolicy::IntervalMs(interval) => {
                now_ms.saturating_sub(self.archive.last_flush_ms) >= interval
            }
            FlushPolicy::Manual => false,
        };

        let append = self
            .archive
            .append(now_ms, &payload)
            .expect("append archive tick summary");
        self.last_archive_payload_len = payload.len();
        self.last_archive_append_frame = Some(ArchiveAppendFrame {
            now_ms,
            seq: append.seq,
            bytes: append.bytes as u32,
            flushed,
        });
    }

    fn current_tcf_phase_0_1(&self, now_ms: u64) -> f32 {
        self.last_tcf_frame
            .map(|frame| f32::from(frame.phase_bin) / 255.0)
            // Fallback if no TCF frame is available yet: deterministic sawtooth over 1s.
            .unwrap_or_else(|| ((now_ms % 1_000) as f32) / 1_000.0)
            .clamp(0.0, 1.0)
    }

    fn make_spike(
        &mut self,
        now_ms: u64,
        src: OscId,
        kind: SpikeKind,
        phase_bin: u8,
        magnitude: f32,
        ttfs_override: Option<u8>,
    ) -> SnnSpikeEvent {
        self.spike_seq = self.spike_seq.wrapping_add(1);
        SnnSpikeEvent {
            now_ms,
            src,
            kind,
            spike_id: self.spike_seq,
            phase_bin,
            ttfs_code: ttfs_override.unwrap_or_else(|| ttfs_from_strength(magnitude)),
            magnitude,
        }
    }

    fn deterministic_features(&mut self) -> [FeatureEvent; 3] {
        let k = self.snn_tick_counter as f32;
        self.snn_tick_counter = self.snn_tick_counter.wrapping_add(1);

        [
            FeatureEvent {
                chan: 10,
                intensity: 0.9,
                novelty: 0.85,
            },
            FeatureEvent {
                chan: 11,
                intensity: (k % 5.0) / 4.0,
                novelty: 0.15,
            },
            FeatureEvent {
                chan: 12,
                intensity: ((k + 1.0) % 4.0) / 3.0,
                novelty: 0.65,
            },
        ]
    }

    fn iit_phi_snapshot(&self) -> ucf_frames::v1::PhiProxySnapshot {
        ucf_frames::v1::PhiProxySnapshot {
            phi: self.iit_proxy_state.phi_proxy,
            coherence_mean: self.iit_proxy_state.coherence,
            coherence_min: self.iit_proxy_state.flow,
            n_pairs: self.iit_proxy_state.filled as u16,
        }
    }

    fn emit_snn_signals<A: ActionAdapter>(
        &mut self,
        adapter: &mut A,
        now_ms: u64,
    ) -> Result<(), RuntimeError> {
        let phase = self.onn.phase(MOD_PBM);
        let features = self.deterministic_features();
        let snn_spikes = encode(now_ms, phase, self.snn_encode_cfg, &features);
        self.last_snn_spike_count = snn_spikes.len();

        let brainbus_spikes = to_brainbus(&snn_spikes);
        adapter.emit_brain_spikes(brainbus_spikes)?;
        let _ = adapter.take_brain_spike_meta();
        Ok(())
    }

    fn ingest_with_decision_and_snapshot<A: ActionAdapter>(
        &mut self,
        adapter: &mut A,
        ctrl: ControlFrame,
        decision: DecisionFrame,
        snapshot: NeuromodulatorSnapshot,
        phi: ucf_frames::v1::PhiProxySnapshot,
    ) -> Result<DecisionFrame, RuntimeError> {
        self.emit_snn_signals(adapter, ctrl.time.tick.get())?;

        let decision = if let Some(nsr_frame) = self.last_nsr_frame {
            match nsr_frame.verdict {
                2 => DecisionFrame::deny_with_reason(
                    decision.time,
                    decision.corr,
                    decision.intent,
                    ucf_frames::v1::ReasonCode("deny_nsr_block"),
                    ucf_frames::v1::DenyReasonCode::PolicyViolation,
                    "deny_nsr_block",
                )
                .with_meta(decision.meta),
                0 | 1 => decision,
                _ => decision,
            }
        } else {
            decision
        };

        let eid1 = self.ids.next();
        self.ess.append(
            ExperienceRecord::from_control(eid1, ctrl.clone())
                .with_neuromod(snapshot)
                .with_iit_phi(phi),
        )?;

        let eid2 = self.ids.next();
        self.ess.append(
            ExperienceRecord::from_decision(eid2, decision.clone())
                .with_neuromod(snapshot)
                .with_iit_phi(phi),
        )?;

        if let Err(error) = Gem::execute(adapter, &ctrl, Some(&decision)) {
            let mut note = format!("gem_error:{error}");
            if note.chars().count() > 120 {
                note = note.chars().take(120).collect();
            }

            let eid3 = self.ids.next();
            self.ess.append(ExperienceRecord::note(
                eid3,
                decision.time,
                decision.corr,
                note,
            ))?;

            return Err(error.into());
        }

        if let Some((count, target)) = adapter.take_brain_spike_meta() {
            let mut note = format!("brain_spikes:n={count},dst={target}");
            if note.chars().count() > 120 {
                note = note.chars().take(120).collect();
            }

            let eid3 = self.ids.next();
            self.ess.append(ExperienceRecord::note(
                eid3,
                decision.time,
                decision.corr,
                note,
            ))?;
        }

        Ok(decision)
    }
}

fn tick_summary_payload(
    now_ms: u64,
    coherence_state: CoherenceState,
    mean_lock: f32,
    gate: f32,
) -> Vec<u8> {
    let mut payload = Vec::with_capacity(11);
    payload.extend_from_slice(&now_ms.to_le_bytes());
    payload.push(match coherence_state {
        CoherenceState::Stable => 0,
        CoherenceState::Drifting => 1,
        CoherenceState::Fragmenting => 2,
    });
    payload.push(quantize_unit(mean_lock));
    payload.push(quantize_unit(gate));
    payload
}

fn quantize_unit(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

impl Default for RuntimeOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

fn sync_graph_from_cde_state(graph: &mut CausalGraph, state: &CdeState) {
    graph.vars.clear();
    graph.hyps.clear();
    for hyp in &state.hyps {
        graph.upsert_var(u32::from(hyp.edge.src));
        graph.upsert_var(u32::from(hyp.edge.dst));
        graph.upsert_hypothesis(
            Edge {
                from: u32::from(hyp.edge.src),
                to: u32::from(hyp.edge.dst),
            },
            hyp.last_update_ms,
            hyp.conf.clamp(0.0, 1.0),
        );
    }
}
