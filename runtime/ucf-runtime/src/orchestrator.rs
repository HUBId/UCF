use crate::errors::RuntimeError;
use ucf_biophys::v0::{
    hpa_step, modulate_hh, FieldEvent, FieldEventKind, FieldUpdateCfg, HhParams, HpaCfg, HpaState,
    ModulationCfg, NeuromodulatorField as BiophysField,
};
use ucf_ess::v1::{ExperienceRecord, ExperienceStore, IdAllocator, InMemoryEss};
use ucf_frames::v1::{
    BiophysFrame, BiophysHhParams, ControlFrame, DecisionFrame, NeuromodulatorSnapshot,
};
use ucf_iit_proxy::v0::{
    IitConfig, IitMonitor, MOD_BLUE, MOD_GEIST, MOD_JEPA, MOD_NSR, MOD_PBM, MOD_SSM,
};
use ucf_neuromod::v0::{NeuromodInputs, NeuromodScheduler, NeuromodulatorField};
use ucf_onn::v0::{OnnCore, PhaseDeg};
use ucf_policy::{adapter::ActionAdapter, gem::Gem, pbm::Pbm};
use ucf_snn::v0::{encode, to_brainbus, FeatureEvent, SnnEncodeCfg};

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
}

impl RuntimeOrchestrator {
    pub fn new() -> Self {
        let mut onn = OnnCore::new(1.0, 0.0);
        for module_id in [MOD_JEPA, MOD_SSM, MOD_NSR, MOD_PBM, MOD_GEIST, MOD_BLUE] {
            onn.register(module_id, PhaseDeg(0.0));
        }

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
        }
    }

    pub fn last_snn_spike_count(&self) -> usize {
        self.last_snn_spike_count
    }

    pub fn last_biophys_frame(&self) -> Option<BiophysFrame> {
        self.last_biophys_frame
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
        let dt_s = self
            .last_biophys_tick_ms
            .map(|last| now_ms.saturating_sub(last) as f32 / 1000.0)
            .unwrap_or(0.001);
        self.last_biophys_tick_ms = Some(now_ms);

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
                intensity: ((k % 5.0) / 4.0),
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
