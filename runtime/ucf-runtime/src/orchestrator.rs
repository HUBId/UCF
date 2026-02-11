use crate::errors::RuntimeError;
use ucf_biophys::v0::{
    apply_coherence_feedback, classify, compute_integration, couple_pair, ensure_osc, hpa_step,
    modulate_hh, onn_step, osc_step, phase_bin, phase_lock, ssm_step, ttfs_from_strength,
    ttfs_phase, verify_graph, CausalGraph, CoherenceState, Edge, EventBus, FieldEvent,
    FieldEventKind, FieldUpdateCfg, HhParams, HpaCfg, HpaState, IITCfg, IITInputs, IITState,
    Microcircuit, ModulationCfg, NeuromodulatorField as BiophysField, OnnCfg, OnnState, Osc, OscId,
    PhaseCfg, RuleCfg, SnnSpikeEvent, SpikeCodecCfg, SpikeKind, SsmCfg, SsmInputs, SsmState,
    VerifyVerdict, SSM_D,
};
use ucf_ess::v1::{ExperienceRecord, ExperienceStore, IdAllocator, InMemoryEss};
use ucf_frames::v1::{
    BiophysFrame, BiophysHhParams, CdeFrame, ControlFrame, DecisionFrame, IitFrame,
    MicrocircuitFrame, NeuromodulatorSnapshot, NsrFrame, OnnFrame, PhaseFrame, SnnFrame, SsmFrame,
};
use ucf_iit_proxy::v0::{
    IitConfig, IitMonitor, MOD_BLUE, MOD_GEIST, MOD_JEPA, MOD_NSR, MOD_PBM, MOD_SSM,
};
use ucf_neuromod::v0::{NeuromodInputs, NeuromodScheduler, NeuromodulatorField};
use ucf_onn::v0::{OnnCore, PhaseDeg};
use ucf_policy::{adapter::ActionAdapter, gem::Gem, pbm::Pbm};
use ucf_snn::v0::{encode, to_brainbus, FeatureEvent, SnnEncodeCfg};

const OSC_SSM: OscId = 1;
const OSC_CDE: OscId = 2;
const OSC_NSR: OscId = 3;
const OSC_COHERENCE: OscId = 4;

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

#[derive(Clone, Debug)]
struct WorkingContext {
    ctx: [f32; SSM_D],
}

impl Default for WorkingContext {
    fn default() -> Self {
        Self { ctx: [0.0; SSM_D] }
    }
}

pub struct RuntimeOrchestrator {
    pub ess: InMemoryEss,
    pub ids: IdAllocator,
    pub pbm: Pbm,
    pub gem: Gem,
    neuromod_field: NeuromodulatorField,
    neuromod_scheduler: NeuromodScheduler,
    onn: OnnCore,
    iit_monitor: IitMonitor,
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
    spike_seq: u32,
    last_onn_frame: Option<OnnFrame>,
    last_snn_frame: Option<SnnFrame>,
    iit_cfg: IITCfg,
    iit_state: IITState,
    last_iit_frame: Option<IitFrame>,
    cde_graph: CausalGraph,
    nsr_cfg: RuleCfg,
    last_cde_frame: Option<CdeFrame>,
    last_nsr_frame: Option<NsrFrame>,
    ssm_state: SsmState,
    ssm_cfg: SsmCfg,
    ssm_last_u: [f32; SSM_D],
    working_context: WorkingContext,
    last_ssm_frame: Option<SsmFrame>,
}

impl RuntimeOrchestrator {
    pub fn new() -> Self {
        let mut onn = OnnCore::new(1.0, 0.0);
        for module_id in [MOD_JEPA, MOD_SSM, MOD_NSR, MOD_PBM, MOD_GEIST, MOD_BLUE] {
            onn.register(module_id, PhaseDeg(0.0));
        }

        let mut onn_state = OnnState::default();
        ensure_osc(&mut onn_state, OSC_SSM);
        ensure_osc(&mut onn_state, OSC_CDE);
        ensure_osc(&mut onn_state, OSC_NSR);
        ensure_osc(&mut onn_state, OSC_COHERENCE);

        Self {
            ess: InMemoryEss::new(),
            ids: IdAllocator::new(1),
            pbm: Pbm,
            gem: Gem,
            neuromod_field: NeuromodulatorField::new_baseline(),
            neuromod_scheduler: NeuromodScheduler::new(1),
            onn,
            iit_monitor: IitMonitor::new(IitConfig::default()),
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
            spike_seq: 0,
            last_onn_frame: None,
            last_snn_frame: None,
            iit_cfg: IITCfg::default(),
            iit_state: IITState::default(),
            last_iit_frame: None,
            cde_graph: CausalGraph::default(),
            nsr_cfg: RuleCfg::default(),
            last_cde_frame: None,
            last_nsr_frame: None,
            ssm_state: SsmState::default(),
            ssm_cfg: SsmCfg::default(),
            ssm_last_u: [0.0; SSM_D],
            working_context: WorkingContext::default(),
            last_ssm_frame: None,
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

    pub fn last_snn_frame(&self) -> Option<SnnFrame> {
        self.last_snn_frame
    }

    pub fn set_onn_coupling_for_test(&mut self, coupling: f32) {
        self.onn_cfg.coupling = coupling.clamp(0.0, 1.0);
    }

    pub fn drain_event_bus_for_test(&mut self) -> Vec<SnnSpikeEvent> {
        self.event_bus.drain()
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

    pub fn force_causal_cycle_for_test(&mut self, now_ms: u64) {
        self.cde_graph
            .upsert_hypothesis(Edge { from: 1, to: 2 }, now_ms, 0.2);
        self.cde_graph
            .upsert_hypothesis(Edge { from: 2, to: 3 }, now_ms, 0.2);
        self.cde_graph
            .upsert_hypothesis(Edge { from: 3, to: 1 }, now_ms, 0.2);
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
        let phi = self.iit_monitor.compute(&self.onn);
        let snapshot = self.neuromod_field.snapshot();

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
        let phi = self.iit_monitor.compute(&self.onn);
        let snapshot = self.neuromod_field.snapshot();
        self.ingest_with_decision_and_snapshot(adapter, ctrl, decision, snapshot, phi)
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
        self.cde_graph.upsert_var(1);
        self.cde_graph.upsert_var(2);
        self.cde_graph.upsert_var(3);
        self.cde_graph.upsert_var(4);

        let lock_nsr_bucket = bucket_value(lock_nsr_jepa);
        let lock_micro_bucket = bucket_value(lock_micro_nsr);
        let spike_bucket =
            bucket_value((spike_rate_hz / self.iit_cfg.spike_rate_norm_hz).clamp(0.0, 1.0));
        let integration_bucket = bucket_value(self.iit_state.integration_ema);

        let edge_spike_micro = Edge { from: 3, to: 2 };
        if spike_bucket >= 1.0 && lock_micro_bucket <= 0.0 {
            self.cde_graph
                .upsert_hypothesis(edge_spike_micro, now_ms, 0.02);
        } else if self
            .cde_graph
            .hyps
            .iter()
            .any(|h| h.edge == edge_spike_micro)
        {
            self.cde_graph
                .upsert_hypothesis(edge_spike_micro, now_ms, -0.005);
        }

        let edge_lock_integration = Edge { from: 1, to: 4 };
        if integration_bucket <= 0.0 && lock_nsr_bucket <= 0.0 {
            self.cde_graph
                .upsert_hypothesis(edge_lock_integration, now_ms, 0.02);
        } else if self
            .cde_graph
            .hyps
            .iter()
            .any(|h| h.edge == edge_lock_integration)
        {
            self.cde_graph
                .upsert_hypothesis(edge_lock_integration, now_ms, -0.005);
        }

        let (verdict, verified_ratio) = verify_graph(&self.cde_graph, self.nsr_cfg);
        let top_conf = self
            .cde_graph
            .top_edges(1)
            .first()
            .map(|h| h.confidence)
            .unwrap_or(0.0);

        let mut integration = compute_integration(
            IITInputs {
                lock_nsr_jepa,
                lock_micro_nsr,
                spike_rate_hz,
            },
            self.iit_cfg,
        );

        if verdict == VerifyVerdict::Verified && verified_ratio >= 0.85 {
            integration = (integration + 0.05).clamp(0.0, 1.0);
        }

        let alpha = self.iit_cfg.ema_alpha.clamp(0.0, 1.0);
        self.iit_state.integration_ema =
            (1.0 - alpha) * self.iit_state.integration_ema + alpha * integration;

        let mut coherence_state = classify(self.iit_state.integration_ema, self.iit_cfg);
        if verdict == VerifyVerdict::Rejected {
            coherence_state = CoherenceState::Fragmenting;
        }

        apply_coherence_feedback(
            &mut self.biophys_field,
            self.iit_state.integration_ema,
            coherence_state,
        );
        let state = match coherence_state {
            CoherenceState::Stable => 0,
            CoherenceState::Drifting => 1,
            CoherenceState::Fragmenting => 2,
        };

        let attention = attention_from_coherence(coherence_state);

        let mut u = self.ssm_last_u;
        for value in &mut u[6..] {
            *value *= 0.95;
        }
        u[0] = self.iit_state.integration_ema.clamp(0.0, 1.0);
        u[1] = lock_nsr_jepa.clamp(0.0, 1.0);
        u[2] = lock_micro_nsr.clamp(0.0, 1.0);
        u[3] = (spike_rate_hz / self.iit_cfg.spike_rate_norm_hz).clamp(0.0, 1.0);
        u[4] = field.dopamine.get();
        u[5] = field.serotonin.get();

        let mut noise = (1.0 - lock_nsr_jepa).clamp(0.0, 1.0);
        if onn_out.mean_lock < 0.35 {
            noise = (noise + 0.05).clamp(0.0, 1.0);
        } else if onn_out.mean_lock > 0.75 {
            noise = (noise - 0.03).clamp(0.0, 1.0);
        }

        let ssm_out = ssm_step(
            &mut self.ssm_state,
            &u,
            SsmInputs {
                attention,
                integration: self.iit_state.integration_ema.clamp(0.0, 1.0),
                dopamine: field.dopamine.get(),
                noise,
            },
            self.ssm_cfg,
        );
        self.ssm_last_u = u;
        self.working_context.ctx = ssm_out.ctx;

        let phase_bin_ssm = phase_bin(phase_for_osc(&self.onn_state, OSC_SSM), 16);
        let phase_bin_cde = phase_bin(phase_for_osc(&self.onn_state, OSC_CDE), 16);
        let phase_bin_nsr = phase_bin(phase_for_osc(&self.onn_state, OSC_NSR), 16);
        let phase_bin_coh = phase_bin(phase_for_osc(&self.onn_state, OSC_COHERENCE), 16);

        let mut tick_spikes = Vec::new();
        if ssm_out.gate > 0.6 {
            let magnitude = ((ssm_out.gate - 0.6) / 0.4).clamp(0.0, 1.0);
            tick_spikes.push(self.make_spike(
                now_ms,
                OSC_SSM,
                SpikeKind::Feature,
                phase_bin_ssm,
                magnitude,
                None,
            ));
        }

        let cde_hyp_count = self.cde_graph.hyps.len() as u32;
        if cde_hyp_count > 0 {
            let magnitude = top_conf.min(1.0);
            tick_spikes.push(self.make_spike(
                now_ms,
                OSC_CDE,
                SpikeKind::Causal,
                phase_bin_cde,
                magnitude,
                None,
            ));
        }

        let verify_ttfs = if verdict == VerifyVerdict::Rejected {
            Some(0)
        } else {
            None
        };
        let verify_magnitude = if verdict == VerifyVerdict::Rejected {
            1.0
        } else {
            verified_ratio.clamp(0.0, 1.0)
        };
        tick_spikes.push(self.make_spike(
            now_ms,
            OSC_NSR,
            SpikeKind::Verify,
            phase_bin_nsr,
            verify_magnitude,
            verify_ttfs,
        ));

        if onn_out.mean_lock > 0.7 {
            tick_spikes.push(self.make_spike(
                now_ms,
                OSC_COHERENCE,
                SpikeKind::Attention,
                phase_bin_coh,
                onn_out.mean_lock.clamp(0.0, 1.0),
                None,
            ));
        }

        let mut feature = 0_u32;
        let mut causal = 0_u32;
        let mut verify = 0_u32;
        let mut attention_count = 0_u32;
        for ev in &tick_spikes {
            self.event_bus.push(*ev);
            match ev.kind {
                SpikeKind::Feature => feature += 1,
                SpikeKind::Causal => causal += 1,
                SpikeKind::Verify => verify += 1,
                SpikeKind::Attention => attention_count += 1,
            }
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
        self.last_ssm_frame = Some(SsmFrame {
            now_ms,
            gate: ssm_out.gate,
            norm_l2: ssm_out.norm_l2,
            sparsity: ssm_out.sparsity,
        });
        self.last_iit_frame = Some(IitFrame {
            now_ms,
            integration: self.iit_state.integration_ema,
            state,
        });
        self.last_cde_frame = Some(CdeFrame {
            now_ms,
            hyps: self.cde_graph.hyps.len() as u32,
            top_conf,
            acyclic: self.cde_graph.is_acyclic(),
        });
        self.last_nsr_frame = Some(NsrFrame {
            now_ms,
            verdict: match verdict {
                VerifyVerdict::Verified => 0,
                VerifyVerdict::Rejected => 1,
                VerifyVerdict::Unknown => 2,
            },
            verified_ratio,
        });
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

impl Default for RuntimeOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

fn bucket_value(value: f32) -> f32 {
    if value < 0.33 {
        0.0
    } else if value < 0.66 {
        0.5
    } else {
        1.0
    }
}

fn attention_from_coherence(state: CoherenceState) -> f32 {
    match state {
        CoherenceState::Stable => 0.55,
        CoherenceState::Drifting => 0.45,
        CoherenceState::Fragmenting => 0.35,
    }
}

fn phase_for_osc(state: &OnnState, id: OscId) -> f32 {
    state
        .osc
        .iter()
        .find(|(oid, _)| *oid == id)
        .map(|(_, phase)| *phase)
        .unwrap_or(state.global_phase)
}
