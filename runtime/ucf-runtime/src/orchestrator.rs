use crate::coherence::{CoherenceRuntime, InterestProfile, Subscriber, TickInput};
use crate::errors::RuntimeError;
use crate::hooks::{
    ComputeMilestone, ComputeMilestoneAggregator, ConsolidationHook, GeistHook, GeistRejectReason,
    GeistStateUpdater,
};
use crate::sandbox::{call_spec_from_control, execute_tool_call, CapabilitySetSummary};
use ucf_biophys::v0::{
    apply_coherence_feedback, classify, compute_integration, couple_pair, hpa_step, modulate_hh,
    osc_step, phase_bin, phase_lock, ttfs_from_strength, ttfs_phase, verify_graph, CausalGraph,
    CoherenceState, Edge, EventBus, FieldEvent, FieldEventKind, FieldUpdateCfg, HhParams, HpaCfg,
    HpaState, IITCfg as BiophysIITCfg, IITInputs, IITState as BiophysIITState, Microcircuit,
    ModulationCfg, NeuromodulatorField as BiophysField, Osc, OscId, PhaseCfg, RuleCfg,
    SnnSpikeEvent, SpikeCodecCfg, SpikeKind, VerifyVerdict,
};
use ucf_cde::v0::{
    on_intervention, on_observation, tick_decay, CdeCfg, CdeState, CdeUpdateKind, Intervention,
    Observation, VarId,
};
use ucf_compute::{
    build_backend, compute_input_from_control, AiComputeBackend, ComputeBackendConfig,
    ComputeBudget, CpuStubBackend,
};
use ucf_core::archive_log::ArchiveLog;
use ucf_core::storage::{ArchiveCfg, FlushPolicy, MemArchiveStore};
use ucf_dbm::chemistry::{chemistry_step, ChemistryCfg, NeuromodState};
use ucf_dbm::regions::{region_step, BrainRegion, RegionKind};
use ucf_ess::v1::{
    AuditCheckpointRecord, AuditPayload, ExperienceKind, ExperienceRecord, ExperienceStore,
    IdAllocator, InMemoryEss, SandboxCallRecord, SandboxReplyRecord, ToolAuthRecord,
    ToolExecutionRecord, ToolRequestRecord,
};
use ucf_fep::{
    check_coherence_invariants, fep_step, homeostasis_step, CoherenceCfg, CoherenceSnapshot,
    FepCfg, FepInputs, FepOutputs, HomeoCfg, HomeoState,
};
use ucf_frames::v1::{
    quantize_avg_v_mv, quantize_hormone, ArchiveAppendFrame, BiophysFrame, BiophysHhParams,
    CdeFrame, ChemFrame, CoherenceFrame, ControlFrame, DecisionFrame, DigitalBrainFrame, FepFrame,
    IitFrame, MicrocircuitFrame, NcdeFrame, NeuromodulatorSnapshot, NsrFrame, OnnFrame, PhaseFrame,
    SleFrame, SnnFrame, SsmFrame, TcfFrame,
};
use ucf_iit_proxy::v0::{iit_push_and_eval, IitCfg, IitSample, IitState};
use ucf_ncde::v0::{ncde_step, NcdeCfg, NcdeInput, NcdeState};
use ucf_neuromod::v0::{NeuromodInputs, NeuromodScheduler, NeuromodulatorField};
use ucf_nsr::v0::{Claim, NsrEngine, Verdict};
use ucf_onn::v0::{onn_step, OnnCfg, OnnCore, OnnInput, OnnNode, OnnState, PhaseDeg};
use ucf_policy::{
    adapter::ActionAdapter,
    gem::{issue_capabilities, request_from, AuthorizationOutcome, Gem, ToolGate, ToolStatus},
    pbm::Pbm,
    rate_limiter::RateLimiter,
};
use ucf_sle::v0::{sle_step, SleCfg, SleReason, SleSignals, SleState};
use ucf_snn::v0::{encode, snn_emit, to_brainbus, FeatureEvent, SnnCfg, SnnEncodeCfg, SpikeSrc};
use ucf_spikes::{
    encode_ttfs_us, filter_phase_locked, PhaseLockCfg, Spike as BusSpike, SpikeBus,
    SpikeKind as BusSpikeKind,
};
use ucf_ssm::v0::{ssm_step, SsmCfg, SsmState};
use ucf_tcf::v0::{tcf_tick, TcfCfg, TcfState};

const OSC_SSM: u8 = 1;
const OSC_CDE: u8 = 2;
const OSC_NSR: u8 = 3;
const OSC_COHERENCE: u8 = 4;
const OSC_NSR_TCF_ENFORCE: u8 = 1;
const MOD_PBM: ucf_onn::v0::OscId = 13;

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

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

#[derive(Clone, Debug)]
struct DigitalBrainState {
    amygdala: BrainRegion,
    pfc: BrainRegion,
    chem: NeuromodState,
    chem_cfg: ChemistryCfg,
}

#[derive(Clone, Copy, Debug, Default)]
struct EmotionVector {
    valence: f32,
    arousal: f32,
}

#[derive(Clone, Debug)]
struct FepState {
    cfg: FepCfg,
    homeo_cfg: HomeoCfg,
    homeo_state: HomeoState,
    coh_cfg: CoherenceCfg,
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
    compute_backend: Box<dyn AiComputeBackend>,
    compute_budget: ComputeBudget,
    neuromod_field: NeuromodulatorField,
    neuromod_scheduler: NeuromodScheduler,
    onn: OnnCore,
    snn_encode_cfg: SnnEncodeCfg,
    snn_cfg: SnnCfg,
    snn_tick_counter: u64,
    last_snn_spike_count: usize,
    last_spike_rate_0_1: f32,
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
    forced_nsr_risk_for_test: Option<f32>,
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
    last_ncde_spike_u4: f32,
    digital_brain: DigitalBrainState,
    last_chem_frame: Option<ChemFrame>,
    last_digital_brain_frame: Option<DigitalBrainFrame>,
    emotion: EmotionVector,
    archive: ArchiveLog<MemArchiveStore>,
    last_archive_append_frame: Option<ArchiveAppendFrame>,
    last_archive_payload_len: usize,
    fep_state: FepState,
    forced_surprise_for_test: Option<f32>,
    forced_geist_drift_for_test: Option<f32>,
    forced_ess_pressure_for_test: Option<f32>,
    last_fep_frame: Option<FepFrame>,
    last_coherence_frame: Option<CoherenceFrame>,
    coherence_violation_count: u64,
    coherence_violation_flag: bool,
    last_fep_outputs: Option<FepOutputs>,
    risk_quality_counts: [u64; 3],
    backpressure: f32,
    compute_budget_exceeded_total: u64,
    backpressure_active_total: u64,
    fep_risk_penalty_applied_total: u64,
    coherence_gated_total: u64,
    coherence_runtime: CoherenceRuntime,
    consolidation_hook_enabled: bool,
    geist_hook_enabled: bool,
    consolidation_hook_errors_total: u64,
    geist_hook_errors_total: u64,
    consolidation_milestones_emitted_total: u64,
    geist_updates_accepted_total: u64,
    geist_updates_rejected_total: u64,
    geist_updates_rejected_not_enough_samples_total: u64,
    geist_updates_rejected_unstable_total: u64,
    geist_updates_rejected_degraded_total: u64,
    geist_updates_rejected_drift_total: u64,
    compute_milestone_aggregator: ComputeMilestoneAggregator,
    geist_state_updater: GeistStateUpdater,
    last_compute_milestone: Option<ComputeMilestone>,
    tool_gate: ToolGate,
    audit_head_digest: [u8; 32],
    audit_chain_checkpoint_total: u64,
}

impl RuntimeOrchestrator {
    pub fn try_new_from_env() -> Result<Self, RuntimeError> {
        let cfg = ComputeBackendConfig::from_env()?;
        let mut orchestrator = Self::new();
        orchestrator.compute_budget = cfg.to_budget();
        orchestrator.compute_backend = build_backend(&cfg)?;
        Ok(orchestrator)
    }

    pub fn new() -> Self {
        let mut onn = OnnCore::new(1.0, 0.0);
        onn.register(MOD_PBM, PhaseDeg(0.0));

        let onn_cfg = OnnCfg::default_v0();
        let onn_nodes = [
            OnnNode::Global,
            OnnNode::Tcf,
            OnnNode::Ssm,
            OnnNode::Nsr,
            OnnNode::Cde,
            OnnNode::Iit,
            OnnNode::Sle,
            OnnNode::Ncde,
            OnnNode::Spikes,
        ];
        let onn_state = OnnState::new(&onn_cfg, &onn_nodes);

        let ssm_cfg = SsmCfg::default_small();
        let ssm_state = SsmState::new(&ssm_cfg, 0);
        let ncde_cfg = NcdeCfg::default_v0();
        let ncde_state = NcdeState::new(&ncde_cfg);

        Self {
            ess: InMemoryEss::new(),
            ids: IdAllocator::new(1),
            pbm: Pbm,
            gem: Gem,
            compute_backend: Box::new(CpuStubBackend),
            compute_budget: ComputeBudget::default(),
            neuromod_field: NeuromodulatorField::new_baseline(),
            neuromod_scheduler: NeuromodScheduler::new(1),
            onn,
            snn_encode_cfg: SnnEncodeCfg::default(),
            snn_cfg: SnnCfg::default_v0(),
            snn_tick_counter: 0,
            last_snn_spike_count: 0,
            last_spike_rate_0_1: 0.0,
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
            onn_cfg,
            event_bus: EventBus::default(),
            spike_bus: SpikeBus::default(),
            spike_seq: 0,
            last_onn_frame: None,
            tcf_cfg: TcfCfg::default_gamma40(),
            tcf_state: TcfState::new(&TcfCfg::default_gamma40(), 0),
            last_tcf_frame: None,
            forced_mean_lock_for_test: None,
            forced_nsr_risk_for_test: None,
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
            last_ncde_spike_u4: 0.0,
            digital_brain: DigitalBrainState {
                amygdala: BrainRegion::new(RegionKind::Amygdala, 16),
                pfc: BrainRegion::new(RegionKind::Pfc, 16),
                chem: NeuromodState::baseline(),
                chem_cfg: ChemistryCfg::default_v0(),
            },
            last_chem_frame: None,
            last_digital_brain_frame: None,
            emotion: EmotionVector::default(),
            archive: ArchiveLog::new(MemArchiveStore::new(), ArchiveCfg::default()),
            last_archive_append_frame: None,
            last_archive_payload_len: 0,
            fep_state: FepState {
                cfg: FepCfg::default_v0(),
                homeo_cfg: HomeoCfg::default_v0(),
                homeo_state: HomeoState::new(),
                coh_cfg: CoherenceCfg::default_v0(),
            },
            forced_surprise_for_test: None,
            forced_geist_drift_for_test: None,
            forced_ess_pressure_for_test: None,
            last_fep_frame: None,
            last_coherence_frame: None,
            coherence_violation_count: 0,
            coherence_violation_flag: false,
            last_fep_outputs: None,
            risk_quality_counts: [0; 3],
            backpressure: 0.0,
            compute_budget_exceeded_total: 0,
            backpressure_active_total: 0,
            fep_risk_penalty_applied_total: 0,
            coherence_gated_total: 0,
            coherence_runtime: {
                let mut runtime = CoherenceRuntime::new();
                runtime.register_subscriber(Subscriber {
                    module_id: 1,
                    name: "cde",
                    interest: InterestProfile::HashBuckets(vec![0, 1, 2]),
                });
                runtime.register_subscriber(Subscriber {
                    module_id: 2,
                    name: "nsr",
                    interest: InterestProfile::HashBuckets(vec![3, 4, 5]),
                });
                runtime.register_subscriber(Subscriber {
                    module_id: 3,
                    name: "ssm",
                    interest: InterestProfile::TopKFeatures(vec![1, 2, 3, 5, 8, 13]),
                });
                runtime
            },
            consolidation_hook_enabled: env_flag("UCF_ENABLE_CONSOLIDATION_HOOK"),
            geist_hook_enabled: env_flag("UCF_ENABLE_GEIST_HOOK"),
            consolidation_hook_errors_total: 0,
            geist_hook_errors_total: 0,
            consolidation_milestones_emitted_total: 0,
            geist_updates_accepted_total: 0,
            geist_updates_rejected_total: 0,
            geist_updates_rejected_not_enough_samples_total: 0,
            geist_updates_rejected_unstable_total: 0,
            geist_updates_rejected_degraded_total: 0,
            geist_updates_rejected_drift_total: 0,
            compute_milestone_aggregator: ComputeMilestoneAggregator::new(vec![60, 600, 3600], 8)
                .expect("valid default consolidation windows"),
            geist_state_updater: GeistStateUpdater::new(60, 0.9, 0.2, 0.25)
                .expect("valid default geist hook config"),
            last_compute_milestone: None,
            tool_gate: ToolGate::new(
                ucf_policy::capability::CapabilitySet::empty(),
                RateLimiter::new(1024),
            ),
            audit_head_digest: [0; 32],
            audit_chain_checkpoint_total: 0,
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

    pub fn spike_rate_0_1(&self) -> f32 {
        self.last_spike_rate_0_1
    }

    pub fn set_onn_coupling_for_test(&mut self, coupling: f32) {
        self.onn_cfg.k_couple = coupling.clamp(0.0, 1.0);
    }

    pub fn force_mean_lock_for_test(&mut self, mean_lock: f32) {
        self.forced_mean_lock_for_test = Some(mean_lock.clamp(0.0, 1.0));
    }

    pub fn force_nsr_risk_for_test(&mut self, nsr_risk: f32) {
        self.forced_nsr_risk_for_test = Some(nsr_risk.clamp(0.0, 1.0));
    }

    pub fn force_surprise_for_test(&mut self, surprise: f32) {
        self.forced_surprise_for_test = Some(surprise.clamp(0.0, 1.0));
    }

    pub fn force_geist_drift_for_test(&mut self, drift: f32) {
        self.forced_geist_drift_for_test = Some(drift.clamp(0.0, 1.0));
    }

    pub fn force_ess_pressure_for_test(&mut self, pressure: f32) {
        self.forced_ess_pressure_for_test = Some(pressure.clamp(0.0, 1.0));
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

    pub fn last_chem_frame(&self) -> Option<ChemFrame> {
        self.last_chem_frame
    }

    pub fn last_digital_brain_frame(&self) -> Option<DigitalBrainFrame> {
        self.last_digital_brain_frame
    }

    pub fn emotion_vector(&self) -> (f32, f32) {
        (self.emotion.valence, self.emotion.arousal)
    }

    pub fn ncde_l2_norm_0_1(&self) -> f32 {
        self.last_ncde_frame
            .map(|frame| f32::from(frame.l2_q) / 255.0)
            .unwrap_or(0.0)
    }

    pub fn last_ncde_spike_u4(&self) -> f32 {
        self.last_ncde_spike_u4
    }

    pub fn archive_last_seq(&self) -> u64 {
        self.archive.last_seq().unwrap_or(0)
    }

    pub fn last_archive_payload_len(&self) -> usize {
        self.last_archive_payload_len
    }

    pub fn last_fep_frame(&self) -> Option<FepFrame> {
        self.last_fep_frame
    }

    pub fn last_coherence_frame(&self) -> Option<CoherenceFrame> {
        self.last_coherence_frame
    }

    pub fn coherence_violation_count(&self) -> u64 {
        self.coherence_violation_count
    }

    pub fn set_consolidation_hook_enabled_for_test(&mut self, enabled: bool) {
        self.consolidation_hook_enabled = enabled;
    }

    pub fn set_geist_hook_enabled_for_test(&mut self, enabled: bool) {
        self.geist_hook_enabled = enabled;
    }

    pub fn consolidation_milestones_emitted_total(&self) -> u64 {
        self.consolidation_milestones_emitted_total
    }

    pub fn geist_updates_accepted_total(&self) -> u64 {
        self.geist_updates_accepted_total
    }

    pub fn geist_updates_rejected_total(&self) -> u64 {
        self.geist_updates_rejected_total
    }

    pub fn hook_errors_total(&self) -> (u64, u64) {
        (
            self.consolidation_hook_errors_total,
            self.geist_hook_errors_total,
        )
    }

    pub fn orchestrator_backpressure(&self) -> f32 {
        self.backpressure
    }

    pub fn compute_budget_exceeded_total(&self) -> u64 {
        self.compute_budget_exceeded_total
    }

    pub fn orchestrator_backpressure_active_total(&self) -> u64 {
        self.backpressure_active_total
    }

    pub fn last_compute_milestone(&self) -> Option<&ComputeMilestone> {
        self.last_compute_milestone.as_ref()
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

        let attention_level = if self.attention_event {
            1.0
        } else {
            self.last_nsr_frame
                .map(|frame| 1.0 - (f32::from(frame.verified_q) / 255.0))
                .unwrap_or(1.0)
                .clamp(0.0, 1.0)
        };
        let anchors = vec![(OnnNode::Global, self.current_tcf_phase_0_1(now_ms))];
        let onn_out = onn_step(
            &self.onn_cfg,
            &mut self.onn_state,
            &OnnInput {
                now_ms,
                anchors,
                gate: attention_level,
            },
        );
        self.last_onn_frame = Some(OnnFrame {
            now_ms,
            global_phase_q: quantize_unit(onn_out.global_phase_0_1),
            lock_nsr_cde_q: quantize_unit(onn_out.lock_nsr_cde),
            lock_nsr_ssm_q: quantize_unit(onn_out.lock_nsr_ssm),
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
        let mut nsr_risk = (1.0 - verified_ratio).clamp(0.0, 1.0);
        if let Some(forced) = self.forced_nsr_risk_for_test {
            nsr_risk = forced.clamp(0.0, 1.0);
        }
        let verified_q = (verified_ratio * 255.0).round() as u8;

        let reward = (1.0 - nsr_risk).clamp(0.0, 1.0);
        let stress = nsr_risk.clamp(0.0, 1.0);
        let onn_lock = self
            .last_onn_frame
            .map(|frame| f32::from(frame.lock_nsr_cde_q) / 255.0)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let mut surprise = self
            .last_ncde_frame
            .map(|frame| f32::from(frame.l2_q) / 255.0)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        if let Some(forced) = self.forced_surprise_for_test {
            surprise = forced.clamp(0.0, 1.0);
        }
        let safety = onn_lock;
        let pain = ((surprise - 0.5).max(0.0) * 2.0).clamp(0.0, 1.0);
        let observed_brain_rate = self
            .last_digital_brain_frame
            .map(|f| f32::from(f.amyg_spikes.saturating_add(f.pfc_spikes)) / 32.0)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let mut ess_pressure = (self.ess.len() as f32 / 256.0).clamp(0.0, 1.0);
        if let Some(forced) = self.forced_ess_pressure_for_test {
            ess_pressure = forced.clamp(0.0, 1.0);
        }
        let mut geist_drift = (1.0 - onn_lock) * 0.5;
        if let Some(forced) = self.forced_geist_drift_for_test {
            geist_drift = forced.clamp(0.0, 1.0);
        }
        let snn_event_rate = self.last_spike_rate_0_1.clamp(0.0, 1.0);
        let complexity = (0.5 * snn_event_rate + 0.5 * ess_pressure).clamp(0.0, 1.0);
        let homeo_err = homeostasis_step(
            &self.fep_state.homeo_cfg,
            &mut self.fep_state.homeo_state,
            dt_s,
            snn_event_rate,
            observed_brain_rate,
            ess_pressure,
        );

        let last_compute_summary = self
            .ess
            .get(self.ess.len().saturating_sub(1))
            .and_then(|r| r.compute_summary);
        let compute_risk = last_compute_summary
            .map(|c| c.risk)
            .unwrap_or(nsr_risk)
            .clamp(0.0, 1.0);
        let compute_confidence = last_compute_summary
            .map(|c| c.confidence)
            .unwrap_or(1.0 - compute_risk)
            .clamp(0.0, 1.0);

        let fep_in = FepInputs {
            now_ms,
            dt_s,
            surprise,
            complexity,
            policy_risk: nsr_risk,
            compute_risk,
            compute_confidence,
            onn_lock,
            snn_event_rate,
            ess_pressure,
            ssm_pressure: ssm_gate,
            geist_drift,
        };
        self.fep_risk_penalty_applied_total = self.fep_risk_penalty_applied_total.saturating_add(1);
        let mut fep_out = fep_step(&self.fep_state.cfg, &fep_in);

        chemistry_step(
            dt_s,
            &mut self.digital_brain.chem,
            reward,
            stress,
            safety,
            pain,
            homeo_err,
            &self.digital_brain.chem_cfg,
        );

        let amyg_drive = stress * 1.2 + (1.0 - safety) * 0.6;
        let pfc_drive = safety * 0.8 + reward * 0.4 - stress * 0.3;
        let (amyg_spikes, amyg_v) = region_step(
            now_ms,
            dt_ms as f32,
            &mut self.digital_brain.amygdala,
            &self.digital_brain.chem,
            amyg_drive,
        );
        let (pfc_spikes, pfc_v) = region_step(
            now_ms,
            dt_ms as f32,
            &mut self.digital_brain.pfc,
            &self.digital_brain.chem,
            pfc_drive,
        );

        self.last_chem_frame = Some(ChemFrame {
            now_ms,
            dopa_q: quantize_hormone(self.digital_brain.chem.dopa),
            s5ht_q: quantize_hormone(self.digital_brain.chem.serotonin),
            oxy_q: quantize_hormone(self.digital_brain.chem.oxytocin),
            end_q: quantize_hormone(self.digital_brain.chem.endorphin),
        });
        self.last_digital_brain_frame = Some(DigitalBrainFrame {
            now_ms,
            amyg_spikes: amyg_spikes.min(u16::MAX as u32) as u16,
            pfc_spikes: pfc_spikes.min(u16::MAX as u32) as u16,
            amyg_avg_v_q: quantize_avg_v_mv(amyg_v),
            pfc_avg_v_q: quantize_avg_v_mv(pfc_v),
        });
        self.emotion.valence = (self.digital_brain.chem.dopa - stress).clamp(-1.0, 1.0);
        self.emotion.arousal = (stress - self.digital_brain.chem.serotonin * 0.5).clamp(-1.0, 1.0);

        let snapshot = CoherenceSnapshot {
            surprise,
            ess_pressure,
            ssm_pressure: ssm_gate,
            onn_lock,
            policy_risk: nsr_risk,
            geist_drift,
            attention_gain: fep_out.attention_gain,
            learn_gate: fep_out.learn_gate,
            memory_priority: fep_out.memory_priority,
            action_inhibit: fep_out.action_inhibit,
            homeo_err,
            chem_dopa: self.digital_brain.chem.dopa,
            chem_5ht: self.digital_brain.chem.serotonin,
            chem_oxy: self.digital_brain.chem.oxytocin,
            chem_end: self.digital_brain.chem.endorphin,
            brain_amyg_spikes: amyg_spikes as f32,
            brain_pfc_spikes: pfc_spikes as f32,
        };
        self.coherence_violation_flag = false;
        if check_coherence_invariants(&self.fep_state.coh_cfg, &snapshot).is_err() {
            self.coherence_violation_count = self.coherence_violation_count.saturating_add(1);
            self.coherence_violation_flag = true;
            fep_out.action_inhibit = (fep_out.action_inhibit + 0.1).clamp(0.0, 1.0);
        }
        self.last_fep_outputs = Some(fep_out.clone());
        self.last_fep_frame = Some(FepFrame {
            now_ms,
            attention_q: quantize_unit(fep_out.attention_gain),
            learn_gate_q: quantize_unit(fep_out.learn_gate),
            memprio_q: quantize_unit(fep_out.memory_priority),
            inhibit_q: quantize_unit(fep_out.action_inhibit),
            confidence_q: quantize_unit(fep_out.confidence),
            homeo_err_q: quantize_unit(homeo_err),
        });
        let coupling = (0.25 * fep_out.attention_gain
            + 0.25 * fep_out.memory_priority
            + 0.25 * fep_out.action_inhibit
            + 0.25 * (1.0 - homeo_err.clamp(0.0, 1.0)))
        .clamp(0.0, 1.0);
        self.last_coherence_frame = Some(CoherenceFrame {
            now_ms,
            coupling_q: quantize_unit(coupling),
            drift_q: quantize_unit(geist_drift),
            risk_q: quantize_unit(nsr_risk),
            lock_q: quantize_unit(onn_lock),
        });

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
                chan: OSC_SSM,
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
                chan: OSC_CDE,
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
                chan: OSC_NSR,
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
                chan: OSC_COHERENCE,
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

        for spike in &tick_spikes {
            let mapped = match spike.kind {
                BusSpikeKind::Novelty => SpikeKind::Feature,
                BusSpikeKind::Verify => SpikeKind::Verify,
                BusSpikeKind::CausalHit => SpikeKind::Causal,
                BusSpikeKind::MemoryMark => continue,
                BusSpikeKind::AttentionShift => SpikeKind::Attention,
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

        let onn_phase_0_1 = self
            .last_onn_frame
            .map(|frame| f32::from(frame.global_phase_q) / 255.0)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let cde_candidates: Vec<(u32, f32)> = self
            .cde_state
            .hyps
            .iter()
            .enumerate()
            .map(|(idx, hyp)| (1000 + (idx as u32), hyp.conf.clamp(0.0, 1.0)))
            .collect();
        let nsr_candidates = if nsr_risk > 0.0 {
            vec![(2000_u32, nsr_risk.clamp(0.0, 1.0))]
        } else {
            Vec::new()
        };
        let ssm_candidates = if ssm_out.gate > 0.0 {
            vec![(3000_u32, ssm_out.gate.clamp(0.0, 1.0))]
        } else {
            Vec::new()
        };

        let snn_from_cde = snn_emit(
            &self.snn_cfg,
            now_ms,
            onn_phase_0_1,
            SpikeSrc::Cde,
            &cde_candidates,
        );
        let snn_from_nsr = snn_emit(
            &self.snn_cfg,
            now_ms,
            onn_phase_0_1,
            SpikeSrc::Nsr,
            &nsr_candidates,
        );
        let snn_from_ssm = snn_emit(
            &self.snn_cfg,
            now_ms,
            onn_phase_0_1,
            SpikeSrc::Ssm,
            &ssm_candidates,
        );

        let fired = snn_from_cde
            .fired_count
            .saturating_add(snn_from_nsr.fired_count)
            .saturating_add(snn_from_ssm.fired_count);
        let suppressed = snn_from_cde
            .suppressed_count
            .saturating_add(snn_from_nsr.suppressed_count)
            .saturating_add(snn_from_ssm.suppressed_count);
        let max_amp_q = snn_from_cde
            .emitted
            .iter()
            .chain(snn_from_nsr.emitted.iter())
            .chain(snn_from_ssm.emitted.iter())
            .map(|event| event.amp_q)
            .max()
            .unwrap_or(0);

        self.last_snn_frame = Some(SnnFrame {
            now_ms,
            fired,
            suppressed,
            max_amp_q,
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
        let spike_rate = if self.snn_cfg.max_events_per_tick == 0 {
            0.0
        } else {
            f32::from(fired) / (self.snn_cfg.max_events_per_tick as f32)
        }
        .clamp(0.0, 1.0);
        self.last_spike_rate_0_1 = spike_rate;
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
        self.last_ncde_spike_u4 = ncde_inp.u[4];
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
        self.last_onn_frame
            .map(|frame| f32::from(frame.global_phase_q) / 255.0)
            .or_else(|| {
                self.last_tcf_frame
                    .map(|frame| f32::from(frame.phase_bin) / 255.0)
            })
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

        let mut decision = decision;

        let compute_input = compute_input_from_control(&ctrl);
        let mut compute_signals = match self
            .compute_backend
            .compute(&compute_input, self.compute_budget)
        {
            Ok(signals) => signals,
            Err(_) => ucf_compute::ComputeSignals::unavailable(
                &compute_input,
                self.compute_budget,
                self.compute_backend.name(),
            ),
        };
        if ucf_compute::validate_risk_signal(&compute_signals.risk_signal).is_err() {
            compute_signals = ucf_compute::ComputeSignals::unavailable(
                &compute_input,
                self.compute_budget,
                self.compute_backend.name(),
            );
        }
        let compute_summary = compute_signals.summary(self.compute_backend.name());
        let quality_idx = usize::from(compute_summary.risk_quality.min(2));
        self.risk_quality_counts[quality_idx] =
            self.risk_quality_counts[quality_idx].saturating_add(1);
        if compute_summary.budget_exceeded_stage.is_some() {
            self.compute_budget_exceeded_total =
                self.compute_budget_exceeded_total.saturating_add(1);
        }
        let quality_penalty = if compute_summary.risk_quality == 0 {
            0.0
        } else {
            0.2
        };
        let bp_input = (compute_summary.pressure + quality_penalty).clamp(0.0, 1.0);
        self.backpressure = (0.6 * self.backpressure + 0.4 * bp_input).clamp(0.0, 1.0);
        if compute_summary.budget_exceeded_stage.is_some() {
            self.backpressure = self.backpressure.max(0.85);
        }
        if self.backpressure > 0.8 {
            self.backpressure_active_total = self.backpressure_active_total.saturating_add(1);
        }
        decision = decision.with_compute_summary(ucf_frames::v1::ComputeSignalsSummary {
            backend: compute_summary.backend,
            surprise: compute_summary.surprise,
            pressure: compute_summary.pressure,
            risk: compute_summary.risk,
            confidence: compute_summary.confidence,
            spike_count: compute_summary.spike_count,
            spikes_digest: compute_summary.spikes_digest,
            sparsity: compute_summary.sparsity,
            energy: compute_summary.energy,
            ssm_readout: compute_summary.ssm_readout,
            ssm_digest: compute_summary.ssm_digest,
            world_digest: compute_summary.world_digest,
            risk_quality: Some(compute_summary.risk_quality),
            evidence_context_digest: Some(compute_summary.evidence_context_digest),
            evidence_world_digest: compute_summary.evidence_world_digest,
            evidence_spikes_digest: compute_summary.evidence_spikes_digest,
            evidence_ssm_digest: compute_summary.evidence_ssm_digest,
            backend_profile: Some(compute_summary.backend_profile),
            budget_profile_id: Some(compute_summary.budget_profile_id),
            seed: Some(compute_summary.seed),
            risk_contract_version: Some(compute_summary.risk_contract_version),
            compute_schema_version: Some(compute_summary.compute_schema_version),
            compute_chain_digest: Some(compute_summary.compute_chain_digest),
            compute_code_version: Some(compute_summary.compute_code_version),
            budget_exceeded_stage: compute_summary.budget_exceeded_stage,
            coherence: None,
            instability: None,
            phi_proxy: None,
            coherence_digest: None,
        });

        let (routing, windows, schedule, coherence_metrics, coherence_gate) =
            self.coherence_runtime.tick(
                &compute_signals.spikes,
                TickInput {
                    t: ctrl.time.tick.get(),
                    source_digest: compute_summary.spikes_digest,
                    pressure: compute_summary.pressure,
                    surprise: compute_summary.surprise,
                    risk: compute_summary.risk,
                    confidence: compute_summary.confidence,
                    budget_limit: 8,
                },
            );
        let _ = (routing, windows, schedule);

        if let Some(summary) = decision.compute_summary {
            decision = decision.with_compute_summary(ucf_frames::v1::ComputeSignalsSummary {
                coherence: Some(coherence_metrics.coherence),
                instability: Some(coherence_metrics.instability),
                phi_proxy: Some(coherence_metrics.phi_proxy),
                coherence_digest: Some(coherence_metrics.digest),
                ..summary
            });
        }
        decision = decision.with_gating_reason(coherence_gate);
        if coherence_gate.is_some() {
            self.coherence_gated_total = self.coherence_gated_total.saturating_add(1);
        }

        if self.backpressure > 0.8 && decision.decision == ucf_frames::v1::DecisionCode::Allow {
            let mut next = DecisionFrame::defer_with_reason(
                decision.time,
                decision.corr,
                decision.intent,
                ucf_frames::v1::ReasonCode("compute_backpressure"),
                "compute_backpressure",
            )
            .with_meta(decision.meta);
            if let Some(summary) = decision.compute_summary {
                next = next.with_compute_summary(summary);
            }
            decision = next;
        }

        if compute_summary.risk_quality == 2
            && decision.decision == ucf_frames::v1::DecisionCode::Allow
        {
            let mut next = DecisionFrame::defer_with_reason(
                decision.time,
                decision.corr,
                decision.intent,
                ucf_frames::v1::ReasonCode("compute_unavailable"),
                "compute_unavailable",
            )
            .with_meta(decision.meta);
            if let Some(summary) = decision.compute_summary {
                next = next.with_compute_summary(summary);
            }
            decision = next;
        }

        decision = if let Some(nsr_frame) = self.last_nsr_frame {
            match nsr_frame.verdict {
                2 => {
                    let mut next = DecisionFrame::deny_with_reason(
                        decision.time,
                        decision.corr,
                        decision.intent,
                        ucf_frames::v1::ReasonCode("deny_nsr_block"),
                        ucf_frames::v1::DenyReasonCode::PolicyViolation,
                        "deny_nsr_block",
                    )
                    .with_meta(decision.meta);
                    if let Some(summary) = decision.compute_summary {
                        next = next.with_compute_summary(summary);
                    }
                    next
                }
                0 | 1 => decision,
                _ => decision,
            }
        } else {
            decision
        };

        if let Some(fep) = &self.last_fep_outputs {
            if fep.action_inhibit >= 0.5 && decision.decision == ucf_frames::v1::DecisionCode::Allow
            {
                let mut next = DecisionFrame::defer_with_reason(
                    decision.time,
                    decision.corr,
                    decision.intent,
                    ucf_frames::v1::ReasonCode("fep_inhibit_high"),
                    "fep_inhibit_high",
                )
                .with_meta(decision.meta);
                if let Some(summary) = decision.compute_summary {
                    next = next.with_compute_summary(summary);
                }
                decision = next;
            }
        }

        if coherence_gate.is_some() && decision.decision == ucf_frames::v1::DecisionCode::Allow {
            let mut next = DecisionFrame::defer_with_reason(
                decision.time,
                decision.corr,
                decision.intent,
                ucf_frames::v1::ReasonCode("coherence_gate"),
                "coherence_gate",
            )
            .with_meta(decision.meta)
            .with_gating_reason(coherence_gate);
            if let Some(summary) = decision.compute_summary {
                next = next.with_compute_summary(summary);
            }
            decision = next;
        }

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

        if self.consolidation_hook_enabled {
            if let Some(decision_rec) = self.ess.get(self.ess.len().saturating_sub(1)) {
                match self.compute_milestone_aggregator.on_append(decision_rec) {
                    Ok(milestones) => {
                        for milestone in milestones {
                            self.consolidation_milestones_emitted_total = self
                                .consolidation_milestones_emitted_total
                                .saturating_add(1);
                            self.last_compute_milestone = Some(milestone.clone());
                            if self.geist_hook_enabled {
                                match self.geist_state_updater.on_milestone(&milestone) {
                                    Ok(Some(_)) => {
                                        self.geist_updates_accepted_total =
                                            self.geist_updates_accepted_total.saturating_add(1);
                                    }
                                    Ok(None) => {
                                        self.geist_updates_rejected_total =
                                            self.geist_updates_rejected_total.saturating_add(1);
                                        match self.geist_state_updater.last_reject_reason() {
                                            Some(GeistRejectReason::NotEnoughSamples) => {
                                                self.geist_updates_rejected_not_enough_samples_total = self
                                                    .geist_updates_rejected_not_enough_samples_total
                                                    .saturating_add(1);
                                            }
                                            Some(GeistRejectReason::Unstable) => {
                                                self.geist_updates_rejected_unstable_total = self
                                                    .geist_updates_rejected_unstable_total
                                                    .saturating_add(1);
                                            }
                                            Some(GeistRejectReason::Degraded) => {
                                                self.geist_updates_rejected_degraded_total = self
                                                    .geist_updates_rejected_degraded_total
                                                    .saturating_add(1);
                                            }
                                            Some(GeistRejectReason::Drift) => {
                                                self.geist_updates_rejected_drift_total = self
                                                    .geist_updates_rejected_drift_total
                                                    .saturating_add(1);
                                            }
                                            None => {}
                                        }
                                    }
                                    Err(_error) => {
                                        self.geist_hook_errors_total =
                                            self.geist_hook_errors_total.saturating_add(1);
                                    }
                                }
                            }
                        }
                    }
                    Err(_error) => {
                        self.consolidation_hook_errors_total =
                            self.consolidation_hook_errors_total.saturating_add(1);
                    }
                }
            }
        }

        if let Some(fep) = &self.last_fep_outputs {
            if fep.memory_priority >= 0.45 {
                let eid = self.ids.next();
                self.ess.append(ExperienceRecord::note(
                    eid,
                    decision.time,
                    decision.corr,
                    "consolidate:high_mem_priority",
                ))?;
            }
        }

        self.tool_gate.capabilities = issue_capabilities(Some(&decision), decision.time.tick.get());
        let request = request_from(&ctrl, &decision, eid2.0);
        let capability_summary = CapabilitySetSummary::from_set(&self.tool_gate.capabilities);
        let (module, op, input) = call_spec_from_control(&ctrl).map_err(|_| {
            ucf_policy::errors::PolicyError::AdapterError("sandbox_call_spec_failed")
        })?;
        let audit = execute_tool_call(
            adapter,
            &mut self.tool_gate,
            request.clone(),
            module.clone(),
            op.clone(),
            input,
            capability_summary,
        )
        .map_err(|_| ucf_policy::errors::PolicyError::AdapterError("sandbox_call_failed"))?;

        let req_payload = AuditPayload::ToolRequest(ToolRequestRecord {
            tool_request_id: request.id,
            capability_kind: request.kind.as_tag().to_string(),
            target: request.target.clone(),
            decision_id: request.decision_id,
            evidence_chain_digest: request.evidence_chain_digest,
        });
        let eid_req = self.ids.next();
        let req_record = ExperienceRecord::audit(
            eid_req,
            decision.time,
            decision.corr,
            ExperienceKind::ToolRequest,
            req_payload,
            self.audit_head_digest,
        );
        self.audit_head_digest = req_record.audit_digest.unwrap_or(self.audit_head_digest);
        self.ess.append(req_record)?;

        let call_payload = AuditPayload::SandboxCall(SandboxCallRecord {
            tool_request_id: request.id,
            call_digest: audit.call_digest,
            module,
            op,
            evidence_chain_digest: request.evidence_chain_digest,
            capability_count: audit.capability_summary.items.len() as u32,
            isolation_runtime: Some(
                std::env::var("UCF_ISOLATION_RUNTIME").unwrap_or_else(|_| "inproc".to_string()),
            ),
            wasm_module_digest: None,
            fuel_used: None,
        });
        let eid_call = self.ids.next();
        let call_record = ExperienceRecord::audit(
            eid_call,
            decision.time,
            decision.corr,
            ExperienceKind::SandboxCall,
            call_payload,
            self.audit_head_digest,
        );
        self.audit_head_digest = call_record.audit_digest.unwrap_or(self.audit_head_digest);
        self.ess.append(call_record)?;

        let (allowed, reason, token_digest) = match audit.auth {
            AuthorizationOutcome::Allowed { token_digest } => {
                (true, "allowed".to_string(), Some(token_digest))
            }
            AuthorizationOutcome::Denied { reason } => (false, format!("{reason:?}"), None),
            AuthorizationOutcome::RateLimited { retry_after_ticks } => {
                (false, format!("rate_limited:{retry_after_ticks}"), None)
            }
        };
        let auth_payload = AuditPayload::ToolAuth(ToolAuthRecord {
            tool_request_id: request.id,
            allowed,
            reason,
            token_digest,
        });
        let eid_auth = self.ids.next();
        let auth_record = ExperienceRecord::audit(
            eid_auth,
            decision.time,
            decision.corr,
            ExperienceKind::ToolAuth,
            auth_payload,
            self.audit_head_digest,
        );
        self.audit_head_digest = auth_record.audit_digest.unwrap_or(self.audit_head_digest);
        self.ess.append(auth_record)?;

        let exec_payload = AuditPayload::ToolExecution(ToolExecutionRecord {
            tool_request_id: request.id,
            status: format!("{:?}", audit.result.status),
            bytes_out: audit.result.bytes_out,
            bytes_in: audit.result.bytes_in,
            error_code: audit.result.error_code.clone(),
        });
        let eid_exec = self.ids.next();
        let exec_record = ExperienceRecord::audit(
            eid_exec,
            decision.time,
            decision.corr,
            ExperienceKind::ToolExecution,
            exec_payload,
            self.audit_head_digest,
        );
        self.audit_head_digest = exec_record.audit_digest.unwrap_or(self.audit_head_digest);
        self.ess.append(exec_record)?;

        let reply_payload = AuditPayload::SandboxReply(SandboxReplyRecord {
            tool_request_id: request.id,
            reply_digest: audit.reply_digest,
            status: format!("{:?}", audit.result.status),
            bytes_out: audit.result.bytes_out.unwrap_or(0),
            bytes_in: audit.result.bytes_in.unwrap_or(0),
            token_digest,
        });
        let eid_reply = self.ids.next();
        let reply_record = ExperienceRecord::audit(
            eid_reply,
            decision.time,
            decision.corr,
            ExperienceKind::SandboxReply,
            reply_payload,
            self.audit_head_digest,
        );
        self.audit_head_digest = reply_record.audit_digest.unwrap_or(self.audit_head_digest);
        self.ess.append(reply_record)?;

        let checkpoint_payload = AuditPayload::AuditCheckpoint(AuditCheckpointRecord {
            head_digest: self.audit_head_digest,
        });
        let eid_cp = self.ids.next();
        let checkpoint_record = ExperienceRecord::audit(
            eid_cp,
            decision.time,
            decision.corr,
            ExperienceKind::AuditCheckpoint,
            checkpoint_payload,
            self.audit_head_digest,
        );
        self.audit_head_digest = checkpoint_record
            .audit_digest
            .unwrap_or(self.audit_head_digest);
        self.audit_chain_checkpoint_total = self.audit_chain_checkpoint_total.saturating_add(1);
        self.ess.append(checkpoint_record)?;

        if matches!(audit.result.status, ToolStatus::Failed) {
            let mut note = format!(
                "gem_error:{}",
                audit
                    .result
                    .error_code
                    .unwrap_or_else(|| "unknown".to_string())
            );
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
            return Err(ucf_policy::errors::PolicyError::AdapterError("tool_failed").into());
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
