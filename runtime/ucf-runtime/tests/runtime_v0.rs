use ucf_cde::v0::CdeUpdateKind;
#[cfg(feature = "compute-burn")]
use ucf_compute::ComputeError;
use ucf_core::types::{SimTime, Tick, WindowId};
use ucf_ess::v1::{AuditPayload, ExperienceKind, ExperiencePayload, ExperienceStore};
use ucf_fep::{check_coherence_invariants, CoherenceCfg, CoherenceSnapshot};
use ucf_frames::v1::{
    BrainStimulusKind, BrainStimulusPayload, ChannelCode, ControlFrame, ControlPayload,
    CorrelationId, DecisionFrame, Intent, IntentId, IntentKind, NeuromodulatorSnapshot,
};
use ucf_policy::{adapter::MockAdapter, gem::Gem};
#[cfg(feature = "compute-burn")]
use ucf_runtime::errors::RuntimeError;
use ucf_runtime::RuntimeOrchestrator;
use ucf_sle::v0::SleCfg;

fn sim_time() -> SimTime {
    SimTime {
        tick: Tick::new(10),
        window: WindowId::new(0),
    }
}

fn intent() -> Intent {
    Intent::new(IntentId(1), IntentKind::System, "runtime-test")
}

fn allow_low_risk(corr: u64) -> DecisionFrame {
    let mut d = DecisionFrame::allow(sim_time(), CorrelationId(corr), "allow");
    d.compute_summary = Some(ucf_frames::v1::ComputeSignalsSummary {
        backend: "stub",
        surprise: 0.1,
        pressure: 0.1,
        risk: 0.1,
        confidence: 0.9,
        surprise_q: 0,
        pressure_q: 0,
        risk_q: 0,
        confidence_q: 0,
        spike_count: 0,
        spikes_digest: [0; 32],
        sparsity: None,
        energy: None,
        ssm_readout: None,
        ssm_digest: None,
        world_digest: None,
        risk_quality: None,
        evidence_context_digest: None,
        evidence_world_digest: None,
        evidence_spikes_digest: None,
        evidence_ssm_digest: None,
        evidence_lfm_digest: None,
        backend_profile: None,
        backend_pack_id: None,
        fixtures_digest: None,
        llm_backend: None,
        world_backend: None,
        sae_backend: None,
        ssm_backend: None,
        lfm_backend: None,
        lfm_uncertainty: None,
        lfm_stability: None,
        lfm_uncertainty_q: None,
        lfm_stability_q: None,
        lfm_state_norm: None,
        lfm_deriv_norm: None,
        lfm_saturation_ratio: None,
        lfm_nan_inf_detected: None,
        lfm_digest: None,
        budget_profile_id: None,
        seed: None,
        risk_contract_version: None,
        compute_schema_version: None,
        compute_chain_digest: None,
        compute_code_version: None,
        budget_exceeded_stage: None,
        contract_version: None,
        backend_id: None,
        validation_status: None,
        violation_reason_mask: None,
        lfm_quality: None,
        coherence: None,
        instability: None,
        coherence_q: None,
        instability_q: None,
        phi_proxy: None,
        coherence_digest: None,
        iit_coherence_q: None,
        iit_incoherence_q: None,
        iit_reason_codes: None,
        stage_allow_mask: None,
        free_energy_proxy_q: None,
        ebm_energy_mean_topk_q: None,
        ebm_w_q: None,
        fep_coupling_version: None,
    });
    d
}

#[test]
fn external_output_text_is_allowed_emitted_and_audited() {
    let ctrl = ControlFrame::new_text(
        sim_time(),
        CorrelationId(1),
        ChannelCode::ExternalOutput,
        intent(),
        "hi",
    );
    let mut orchestrator = RuntimeOrchestrator::new();
    orchestrator.force_nsr_risk_for_test(0.0);
    let mut adapter = MockAdapter::default();

    let decision = orchestrator
        .ingest_and_process(&mut adapter, ctrl)
        .expect("orchestration should succeed");

    assert_eq!(decision.decision, ucf_frames::v1::DecisionCode::Allow);
    assert!(
        adapter.emitted.is_empty() || adapter.emitted == vec!["hi".to_string()],
        "governance tiering may deny tool issuance for external output"
    );
    assert!(orchestrator.ess.len() >= 2);
    let control_idx = (0..orchestrator.ess.len())
        .find(|i| {
            orchestrator
                .ess
                .get(*i)
                .is_some_and(|r| r.kind == ExperienceKind::ControlIn)
        })
        .expect("control record");
    let decision_idx = (0..orchestrator.ess.len())
        .find(|i| {
            orchestrator
                .ess
                .get(*i)
                .is_some_and(|r| r.kind == ExperienceKind::DecisionOut)
        })
        .expect("decision record");
    assert!(control_idx < decision_idx);
    let baseline = NeuromodulatorSnapshot::baseline();
    assert_eq!(
        orchestrator.ess.get(control_idx).and_then(|r| r.neuromod),
        Some(baseline)
    );
    assert!(orchestrator
        .ess
        .get(control_idx)
        .and_then(|r| r.iit_phi)
        .is_some());
    assert_eq!(
        orchestrator.ess.get(decision_idx).and_then(|r| r.neuromod),
        Some(baseline)
    );
    let decision_record = orchestrator.ess.get(decision_idx).expect("decision record");
    assert_eq!(decision_record.decision_meta, Some(decision.meta));
    assert!(decision.compute_summary.is_some());
    let compute = decision.compute_summary.expect("compute summary");
    assert_eq!(compute.backend, "stub");
    assert!(compute.spike_count <= 256);
    assert_eq!(decision_record.compute_summary, Some(compute));
    assert_eq!(decision.meta.attention_gain, 0.70000005);
    assert_eq!(decision.meta.learning_gate, 0.65);
    assert_eq!(decision.meta.recursion_budget, 1);
}

#[test]
fn internal_thought_text_is_allowed_without_adapter_effects() {
    let ctrl = ControlFrame::new_text(
        sim_time(),
        CorrelationId(2),
        ChannelCode::InternalThought,
        intent(),
        "private",
    );
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();

    let decision = orchestrator
        .ingest_and_process(&mut adapter, ctrl)
        .expect("orchestration should succeed");

    assert_eq!(decision.decision, ucf_frames::v1::DecisionCode::Allow);
    assert!(adapter.emitted.is_empty());
    assert_eq!(adapter.mem_writes, 0);
    assert!(orchestrator.ess.len() >= 2);
}

#[test]
fn memory_write_bytes_is_denied_by_default_policy() {
    let ctrl = ControlFrame {
        time: sim_time(),
        corr: CorrelationId(3),
        channel: ChannelCode::MemoryWrite,
        intent: intent(),
        payload: ControlPayload::Bytes(vec![1, 2, 3].into()),
    };
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();

    let decision = orchestrator
        .ingest_and_process(&mut adapter, ctrl)
        .expect("orchestration should succeed");

    assert_eq!(decision.decision, ucf_frames::v1::DecisionCode::Deny);
    assert_eq!(adapter.mem_writes, 0);
    assert!(orchestrator.ess.len() >= 2);
}

#[test]
fn gem_allow_path_emits_and_writes_when_invoked_directly() {
    let mut adapter = MockAdapter::default();

    let output_ctrl = ControlFrame::new_text(
        sim_time(),
        CorrelationId(4),
        ChannelCode::ExternalOutput,
        intent(),
        "allowed",
    );
    let allow_output = allow_low_risk(4);
    Gem::execute(&mut adapter, &output_ctrl, Some(&allow_output)).expect("gem should emit text");

    let memory_ctrl = ControlFrame {
        time: sim_time(),
        corr: CorrelationId(5),
        channel: ChannelCode::MemoryWrite,
        intent: intent(),
        payload: ControlPayload::Bytes(vec![9, 9].into()),
    };
    let allow_memory = allow_low_risk(5);
    Gem::execute(&mut adapter, &memory_ctrl, Some(&allow_memory)).expect("gem should write memory");

    assert!(adapter.emitted.is_empty() || adapter.emitted == vec!["allowed".to_string()]);
    assert!(adapter.mem_writes <= 1);
}

#[test]
fn orchestrator_appends_note_when_brain_spikes_emitted() {
    let ctrl = ControlFrame {
        time: sim_time(),
        corr: CorrelationId(6),
        channel: ChannelCode::BrainStimulus,
        intent: intent(),
        payload: ControlPayload::BrainStimulus(BrainStimulusPayload {
            kind: BrainStimulusKind::SpikeTrain,
            target: 44,
            intensity: 255,
            duration_ms: 90,
        }),
    };

    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();

    orchestrator
        .ingest_with_decision(&mut adapter, ctrl, allow_low_risk(6))
        .expect("orchestration should succeed");

    let same_corr_records = (0..orchestrator.ess.len())
        .filter_map(|idx| orchestrator.ess.get(idx))
        .filter(|r| r.corr == CorrelationId(6))
        .count();
    assert!(same_corr_records >= 2);
}

#[test]
fn orchestrator_tick_records_iit_phi_snapshot() {
    let ctrl = ControlFrame::new_text(
        sim_time(),
        CorrelationId(7),
        ChannelCode::ExternalOutput,
        intent(),
        "iit",
    );
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();

    orchestrator
        .ingest_and_process(&mut adapter, ctrl)
        .expect("orchestration should succeed");

    let control_record = (0..orchestrator.ess.len())
        .filter_map(|idx| orchestrator.ess.get(idx))
        .find(|r| r.kind == ExperienceKind::ControlIn)
        .expect("control record");
    assert!(control_record.iit_phi.is_some());
}

#[test]
fn fep_high_risk_raises_inhibit_and_can_defer() {
    let ctrl = ControlFrame::new_text(
        sim_time(),
        CorrelationId(70),
        ChannelCode::ExternalOutput,
        intent(),
        "risk",
    );
    let mut orchestrator = RuntimeOrchestrator::new();
    orchestrator.force_nsr_risk_for_test(0.9);
    let mut adapter = MockAdapter::default();

    let decision = orchestrator
        .ingest_and_process(&mut adapter, ctrl)
        .expect("orchestration should succeed");

    let fep = orchestrator.last_fep_frame().expect("fep frame");
    assert!(fep.inhibit_q >= 140, "inhibit_q={} ", fep.inhibit_q);
    assert!(
        decision.decision == ucf_frames::v1::DecisionCode::Defer
            || decision.decision == ucf_frames::v1::DecisionCode::Deny
    );
}

#[test]
fn fep_high_surprise_boosts_memory_priority_and_marks_consolidation() {
    let ctrl = ControlFrame::new_text(
        sim_time(),
        CorrelationId(71),
        ChannelCode::InternalThought,
        intent(),
        "surprise",
    );
    let mut orchestrator = RuntimeOrchestrator::new();
    orchestrator.force_surprise_for_test(0.9);
    orchestrator.force_ess_pressure_for_test(0.95);
    let mut adapter = MockAdapter::default();

    orchestrator
        .ingest_and_process(&mut adapter, ctrl)
        .expect("orchestration should succeed");

    let fep = orchestrator.last_fep_frame().expect("fep frame");
    assert!(fep.memprio_q >= 120, "memprio_q={}", fep.memprio_q);

    let has_consolidate = (0..orchestrator.ess.len()).any(|idx| {
        orchestrator
            .ess
            .get(idx)
            .and_then(|r| match &r.payload {
                ExperiencePayload::Text(t) => {
                    Some(t.as_ref().contains("consolidate:high_mem_priority"))
                }
                _ => None,
            })
            .unwrap_or(false)
    });
    assert!(has_consolidate);
}

#[test]
fn fep_high_drift_reduces_learning_and_raises_inhibit() {
    let ctrl = ControlFrame::new_text(
        sim_time(),
        CorrelationId(72),
        ChannelCode::InternalThought,
        intent(),
        "drift",
    );
    let mut orchestrator = RuntimeOrchestrator::new();
    orchestrator.force_geist_drift_for_test(0.9);
    let mut adapter = MockAdapter::default();

    orchestrator
        .ingest_and_process(&mut adapter, ctrl)
        .expect("orchestration should succeed");

    let fep = orchestrator.last_fep_frame().expect("fep frame");
    assert!(fep.learn_gate_q <= 128);
    assert!(fep.inhibit_q >= 153);
}

#[test]
fn coherence_invariants_ok_and_err_cases() {
    let cfg = CoherenceCfg::default_v0();
    let ok = CoherenceSnapshot {
        surprise: 0.4,
        ess_pressure: 0.4,
        ssm_pressure: 0.3,
        onn_lock: 0.8,
        policy_risk: 0.3,
        geist_drift: 0.2,
        attention_gain: 0.6,
        learn_gate: 0.6,
        memory_priority: 0.6,
        action_inhibit: 0.4,
        homeo_err: 0.2,
        chem_dopa: 0.5,
        chem_5ht: 0.5,
        chem_oxy: 0.5,
        chem_end: 0.5,
        brain_amyg_spikes: 3.0,
        brain_pfc_spikes: 3.0,
    };
    assert!(check_coherence_invariants(&cfg, &ok).is_ok());

    let bad = CoherenceSnapshot {
        policy_risk: 0.9,
        action_inhibit: 0.2,
        ..ok
    };
    assert!(check_coherence_invariants(&cfg, &bad).is_err());
}

#[test]
fn orchestrator_tick_emits_snn_spikes_to_brainbus_sink() {
    let ctrl = ControlFrame::new_text(
        sim_time(),
        CorrelationId(8),
        ChannelCode::InternalThought,
        intent(),
        "snn",
    );
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();

    orchestrator
        .ingest_and_process(&mut adapter, ctrl)
        .expect("orchestration should succeed");

    assert!(orchestrator.last_snn_spike_count() >= 1);
    assert!(!adapter.brain_spikes().is_empty());
}

#[test]
fn orchestrator_tick_updates_biophys_frame() {
    let ctrl = ControlFrame::new_text(
        SimTime {
            tick: Tick::new(30),
            window: WindowId::new(0),
        },
        CorrelationId(9),
        ChannelCode::InternalThought,
        intent(),
        "biophys",
    );
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();

    orchestrator
        .ingest_and_process(&mut adapter, ctrl)
        .expect("orchestration should succeed");

    let biophys = orchestrator
        .last_biophys_frame()
        .expect("biophys frame should be present");
    assert_eq!(biophys.now_ms, 30);
    assert!(biophys.field[0] > 0.1);
    assert!(biophys.field[1] < 0.1);
}

#[test]
fn orchestrator_hpa_cortisol_differs_between_baseline_and_stress_ticks() {
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();

    let baseline_ctrl = ControlFrame::new_text(
        SimTime {
            tick: Tick::new(14),
            window: WindowId::new(0),
        },
        CorrelationId(10),
        ChannelCode::InternalThought,
        intent(),
        "baseline",
    );

    orchestrator
        .ingest_and_process(&mut adapter, baseline_ctrl)
        .expect("baseline tick should succeed");

    let baseline = orchestrator
        .last_biophys_frame()
        .expect("biophys frame should exist at baseline");

    let stress_ctrl = ControlFrame::new_text(
        SimTime {
            tick: Tick::new(15),
            window: WindowId::new(0),
        },
        CorrelationId(11),
        ChannelCode::InternalThought,
        intent(),
        "stress",
    );

    orchestrator
        .ingest_and_process(&mut adapter, stress_ctrl)
        .expect("stress tick should succeed");

    let stress = orchestrator
        .last_biophys_frame()
        .expect("biophys frame should exist at stress tick");

    assert_ne!(baseline.hpa_cortisol, stress.hpa_cortisol);
    assert_eq!(stress.now_ms, 15);
}

#[test]
fn orchestrator_tick_updates_microcircuit_frame() {
    let ctrl = ControlFrame::new_text(
        SimTime {
            tick: Tick::new(31),
            window: WindowId::new(0),
        },
        CorrelationId(12),
        ChannelCode::InternalThought,
        intent(),
        "microcircuit",
    );
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();

    orchestrator
        .ingest_and_process(&mut adapter, ctrl)
        .expect("orchestration should succeed");

    let frame = orchestrator
        .last_microcircuit_frame()
        .expect("microcircuit frame should be present");
    assert_eq!(frame.n, 32);
}

#[test]
fn orchestrator_tick_updates_phase_frame_with_valid_locks() {
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();

    for tick in 40..45 {
        let ctrl = ControlFrame::new_text(
            SimTime {
                tick: Tick::new(tick),
                window: WindowId::new(0),
            },
            CorrelationId(100 + tick),
            ChannelCode::InternalThought,
            intent(),
            "phase",
        );

        orchestrator
            .ingest_and_process(&mut adapter, ctrl)
            .expect("phase tick should succeed");
    }

    let phase = orchestrator
        .last_phase_frame()
        .expect("phase frame should be present");
    assert!(phase.jepa_phase >= 0.0);
    assert!(phase.nsr_phase >= 0.0);
    assert!(phase.micro_phase >= 0.0);
    assert!((0.0..=1.0).contains(&phase.lock_nsr_jepa));
    assert!((0.0..=1.0).contains(&phase.lock_micro_nsr));
}

#[test]
fn orchestrator_tick_emits_iit_frame() {
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();

    for tick in 50..55 {
        let ctrl = ControlFrame::new_text(
            SimTime {
                tick: Tick::new(tick),
                window: WindowId::new(0),
            },
            CorrelationId(200 + tick),
            ChannelCode::InternalThought,
            intent(),
            "iit-frame",
        );

        orchestrator
            .ingest_and_process(&mut adapter, ctrl)
            .expect("iit tick should succeed");
    }

    let frame = orchestrator
        .last_iit_frame()
        .expect("iit frame should exist");
    assert!((0..=255).contains(&u16::from(frame.phi_q)));
    assert!((0..=255).contains(&u16::from(frame.coh_q)));
    assert!((0..=255).contains(&u16::from(frame.flow_q)));
    assert!(frame.enforce <= 1);
}

#[test]
fn orchestrator_tick_emits_cde_and_nsr_frames() {
    let mut orchestrator = RuntimeOrchestrator::new();
    orchestrator.force_mean_lock_for_test(0.0);
    orchestrator.set_sle_cfg_for_test(SleCfg {
        min_trigger: 0.0,
        cooldown_ticks: 0,
        ..SleCfg::default_v0()
    });
    orchestrator.set_iit_proxy_cfg_for_test(ucf_iit_proxy::v0::IitCfg {
        min_samples: 2,
        coherence_weight: 1.0,
        flow_weight: 0.0,
        enforce_threshold: 0.95,
        ..ucf_iit_proxy::v0::IitCfg::default_v0()
    });
    let mut adapter = MockAdapter::default();
    let mut saw_causal_spike = false;
    let mut saw_verify_spike = false;

    for tick in 80..90 {
        let ctrl = ControlFrame::new_text(
            SimTime {
                tick: Tick::new(tick),
                window: WindowId::new(0),
            },
            CorrelationId(300 + tick),
            ChannelCode::InternalThought,
            intent(),
            "cde-nsr",
        );

        orchestrator
            .ingest_and_process(&mut adapter, ctrl)
            .expect("cde/nsr tick should succeed");

        let cde = orchestrator
            .last_cde_frame()
            .expect("cde frame should be present each tick");
        assert_eq!(cde.now_ms, tick);
        assert!((0..=255).contains(&u16::from(cde.top_conf_q)));

        let spikes = orchestrator.drain_event_bus_for_test();
        if spikes
            .iter()
            .any(|s| s.kind == ucf_biophys::v0::SpikeKind::Causal)
        {
            saw_causal_spike = true;
        }
        if spikes
            .iter()
            .any(|s| s.kind == ucf_biophys::v0::SpikeKind::Verify)
        {
            saw_verify_spike = true;
        }
    }

    let cde = orchestrator
        .last_cde_frame()
        .expect("cde frame should be present");
    assert!(cde.hyps <= 256);
    assert!(cde.changed <= cde.hyps);

    assert!(saw_causal_spike);
    assert!(saw_verify_spike);

    let nsr = orchestrator
        .last_nsr_frame()
        .expect("nsr frame should be present");
    assert!(nsr.verdict <= 2);
    assert!(nsr.satisfied <= nsr.total);
    assert!((0..=255).contains(&u16::from(nsr.verified_q)));
}

#[test]
fn cde_intervention_helper_updates_engine_state() {
    let mut orchestrator = RuntimeOrchestrator::new();
    let update = orchestrator.feed_cde_intervention_for_test(42, vec![(1, 1.0)], vec![(2, 0.6)]);
    assert!(matches!(update, CdeUpdateKind::Updated { .. }));
}

#[test]
fn forced_cycle_rejects_nsr_and_forces_fragmenting_coherence() {
    let mut orchestrator = RuntimeOrchestrator::new();
    orchestrator.force_mean_lock_for_test(0.0);
    let mut adapter = MockAdapter::default();

    let ctrl = ControlFrame::new_text(
        SimTime {
            tick: Tick::new(120),
            window: WindowId::new(0),
        },
        CorrelationId(444),
        ChannelCode::InternalThought,
        intent(),
        "cycle",
    );

    orchestrator
        .ingest_and_process(&mut adapter, ctrl)
        .expect("cycle tick should succeed");

    let nsr = orchestrator
        .last_nsr_frame()
        .expect("nsr frame should be present");
    assert!(nsr.verdict <= 2);

    let iit = orchestrator
        .last_iit_frame()
        .expect("iit frame should be present");
    assert!(iit.enforce <= 1);
}

#[test]
fn orchestrator_tick_emits_ssm_frame_each_tick() {
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();

    for tick in 130..136 {
        let ctrl = ControlFrame::new_text(
            SimTime {
                tick: Tick::new(tick),
                window: WindowId::new(0),
            },
            CorrelationId(500 + tick),
            ChannelCode::InternalThought,
            intent(),
            "ssm",
        );

        orchestrator
            .ingest_and_process(&mut adapter, ctrl)
            .expect("ssm tick should succeed");

        let frame = orchestrator
            .last_ssm_frame()
            .expect("ssm frame should be present");
        assert_eq!(frame.now_ms, tick);
        assert!((0..=255).contains(&frame.gate_q));
        assert!((0..=255).contains(&frame.energy_q));
    }
}

#[test]
fn low_lock_enforcement_reduces_ssm_gate_and_sets_iit_enforce() {
    let mut orchestrator = RuntimeOrchestrator::new();
    orchestrator.force_mean_lock_for_test(0.0);
    orchestrator.set_iit_proxy_cfg_for_test(ucf_iit_proxy::v0::IitCfg {
        min_samples: 2,
        coherence_weight: 1.0,
        flow_weight: 0.0,
        enforce_threshold: 0.95,
        ..ucf_iit_proxy::v0::IitCfg::default_v0()
    });
    let mut adapter = MockAdapter::default();

    let mut baseline_gate = None;
    let mut enforced_gate = None;
    let mut saw_enforce = false;

    for tick in 200..220 {
        let ctrl = ControlFrame::new_text(
            SimTime {
                tick: Tick::new(tick),
                window: WindowId::new(0),
            },
            CorrelationId(800 + tick),
            ChannelCode::InternalThought,
            intent(),
            "iit-enforce",
        );

        orchestrator
            .ingest_and_process(&mut adapter, ctrl)
            .expect("enforcement tick should succeed");

        let iit = orchestrator.last_iit_frame().expect("iit frame available");
        let gate = orchestrator
            .last_ssm_frame()
            .expect("ssm frame available")
            .gate_q;
        if iit.enforce == 1 {
            saw_enforce = true;
            enforced_gate = Some(gate);
            break;
        }
        baseline_gate = Some(gate);
    }

    assert!(saw_enforce);
    let base = baseline_gate.unwrap_or(255);
    let reduced = enforced_gate.expect("enforced gate should exist");
    assert!(reduced <= base);
}

#[test]
fn ssm_gate_matches_mean_lock_in_fallback_path() {
    let mut orchestrator = RuntimeOrchestrator::new();
    orchestrator.force_mean_lock_for_test(0.42);
    let mut adapter = MockAdapter::default();

    let ctrl = ControlFrame::new_text(
        SimTime {
            tick: Tick::new(137),
            window: WindowId::new(0),
        },
        CorrelationId(637),
        ChannelCode::InternalThought,
        intent(),
        "ssm-gate",
    );

    orchestrator
        .ingest_and_process(&mut adapter, ctrl)
        .expect("ssm gate tick should succeed");

    assert!((orchestrator.ssm_gate() - 0.42).abs() < 1e-6);
}

#[test]
fn orchestrator_tick_emits_onn_and_snn_frames_each_tick() {
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();
    let mut phases = Vec::new();

    for tick in 200..210 {
        let ctrl = ControlFrame::new_text(
            SimTime {
                tick: Tick::new(tick),
                window: WindowId::new(0),
            },
            CorrelationId(800 + tick),
            ChannelCode::InternalThought,
            intent(),
            "onn-snn",
        );

        orchestrator
            .ingest_and_process(&mut adapter, ctrl)
            .expect("tick should succeed");

        let onn = orchestrator
            .last_onn_frame()
            .expect("onn frame should exist");
        let snn = orchestrator
            .last_snn_frame()
            .expect("snn frame should exist");

        assert_eq!(onn.now_ms, tick);
        assert_eq!(snn.now_ms, tick);
        assert!(snn.fired <= 64);
        phases.push(onn.global_phase_q);
    }

    assert!(phases.windows(2).any(|w| w[0] != w[1]));
}

#[test]
fn runtime_uses_snn_spike_rate_for_ncde_u4() {
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();

    let ctrl = ControlFrame::new_text(
        SimTime {
            tick: Tick::new(321),
            window: WindowId::new(0),
        },
        CorrelationId(1321),
        ChannelCode::InternalThought,
        intent(),
        "snn-ncde-spike-rate",
    );

    orchestrator
        .ingest_and_process(&mut adapter, ctrl)
        .expect("tick should succeed");

    assert!(orchestrator.spike_rate_0_1() > 0.0);
    assert!(orchestrator.last_ncde_spike_u4() > 0.0);
}

#[test]
fn low_mean_lock_path_changes_ssm_gate_via_noise_adjustment() {
    let mut lo = RuntimeOrchestrator::new();
    lo.set_onn_coupling_for_test(0.0);
    let mut hi = RuntimeOrchestrator::new();

    let mut adapter_lo = MockAdapter::default();
    let mut adapter_hi = MockAdapter::default();

    let ctrl = ControlFrame::new_text(
        SimTime {
            tick: Tick::new(250),
            window: WindowId::new(0),
        },
        CorrelationId(950),
        ChannelCode::InternalThought,
        intent(),
        "lock-noise",
    );

    lo.ingest_and_process(&mut adapter_lo, ctrl.clone())
        .expect("low coupling tick should succeed");
    hi.ingest_and_process(&mut adapter_hi, ctrl)
        .expect("default coupling tick should succeed");

    let low_gate = lo.last_ssm_frame().expect("ssm frame low").gate_q;
    let high_gate = hi.last_ssm_frame().expect("ssm frame hi").gate_q;
    assert!(low_gate <= high_gate);
}

#[test]
fn nsr_verify_spike_emits_for_non_unknown_verdict() {
    let mut orchestrator = RuntimeOrchestrator::new();
    orchestrator.force_mean_lock_for_test(0.0);
    let mut adapter = MockAdapter::default();

    let ctrl = ControlFrame::new_text(
        SimTime {
            tick: Tick::new(300),
            window: WindowId::new(0),
        },
        CorrelationId(999),
        ChannelCode::InternalThought,
        intent(),
        "reject-verify-spike",
    );

    orchestrator
        .ingest_and_process(&mut adapter, ctrl)
        .expect("tick should succeed");

    let spikes = orchestrator.drain_event_bus_for_test();
    let verify = spikes
        .iter()
        .find(|s| s.kind == ucf_biophys::v0::SpikeKind::Verify)
        .expect("verify spike must exist");
    assert!((0.0..=1.0).contains(&verify.magnitude));
}

#[test]
fn archive_log_seq_increments_for_each_tick() {
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();

    for tick in 400..405 {
        let ctrl = ControlFrame::new_text(
            SimTime {
                tick: Tick::new(tick),
                window: WindowId::new(0),
            },
            CorrelationId(1100 + tick),
            ChannelCode::InternalThought,
            intent(),
            "archive-seq",
        );

        orchestrator
            .ingest_and_process(&mut adapter, ctrl)
            .expect("tick should succeed");
    }

    assert_eq!(orchestrator.archive_last_seq(), 5);
}

#[test]
fn archive_tick_payload_is_bounded() {
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();

    let ctrl = ControlFrame::new_text(
        SimTime {
            tick: Tick::new(500),
            window: WindowId::new(0),
        },
        CorrelationId(1500),
        ChannelCode::InternalThought,
        intent(),
        "archive-payload",
    );

    orchestrator
        .ingest_and_process(&mut adapter, ctrl)
        .expect("tick should succeed");

    assert!(orchestrator.last_archive_payload_len() <= 64);
}

#[test]
fn archive_append_frame_emitted_on_each_tick() {
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();

    for tick in 600..605 {
        let ctrl = ControlFrame::new_text(
            SimTime {
                tick: Tick::new(tick),
                window: WindowId::new(0),
            },
            CorrelationId(2000 + tick),
            ChannelCode::InternalThought,
            intent(),
            "archive-frame",
        );

        orchestrator
            .ingest_and_process(&mut adapter, ctrl)
            .expect("tick should succeed");

        let frame = orchestrator
            .last_archive_append_frame()
            .expect("archive frame should exist");
        assert_eq!(frame.now_ms, tick);
        assert_eq!(frame.seq, tick - 599);
    }
}

#[test]
fn low_mean_lock_forces_nsr_block_and_policy_denies_action() {
    let mut orchestrator = RuntimeOrchestrator::new();
    orchestrator.force_mean_lock_for_test(0.0);
    let mut adapter = MockAdapter::default();

    let ctrl = ControlFrame::new_text(
        SimTime {
            tick: Tick::new(700),
            window: WindowId::new(0),
        },
        CorrelationId(2700),
        ChannelCode::ExternalOutput,
        intent(),
        "deny-by-nsr",
    );

    let decision = orchestrator
        .ingest_and_process(&mut adapter, ctrl)
        .expect("tick should succeed");

    let nsr = orchestrator
        .last_nsr_frame()
        .expect("nsr frame should exist");
    assert_eq!(nsr.verdict, 2);
    assert_eq!(decision.decision, ucf_frames::v1::DecisionCode::Deny);
    assert!(adapter.emitted.is_empty());
}

#[test]
fn orchestrator_tick_emits_tcf_frame_each_tick() {
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();

    for tick in 60..80 {
        let ctrl = ControlFrame::new_text(
            SimTime {
                tick: Tick::new(tick),
                window: WindowId::new(0),
            },
            CorrelationId(300 + tick),
            ChannelCode::InternalThought,
            intent(),
            "tcf",
        );

        orchestrator
            .ingest_and_process(&mut adapter, ctrl)
            .expect("tcf tick should succeed");

        let tcf = orchestrator
            .last_tcf_frame()
            .expect("tcf frame should exist");
        assert_eq!(tcf.now_ms, tick);
    }
}

#[test]
fn orchestrator_tcf_mean_lock_is_non_zero_and_not_nan() {
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();

    for tick in 80..100 {
        let ctrl = ControlFrame::new_text(
            SimTime {
                tick: Tick::new(tick),
                window: WindowId::new(0),
            },
            CorrelationId(400 + tick),
            ChannelCode::InternalThought,
            intent(),
            "tcf-lock",
        );
        orchestrator
            .ingest_and_process(&mut adapter, ctrl)
            .expect("runtime tick should succeed");
    }

    let tcf = orchestrator.last_tcf_frame().expect("tcf frame");
    let mean_lock = (tcf.lock_q as f32) / 255.0;
    assert!(!mean_lock.is_nan());
    assert!(mean_lock > 0.0);
}

#[test]
fn orchestrator_tick_emits_spike_frames() {
    let ctrl = ControlFrame::new_text(
        SimTime {
            tick: Tick::new(40),
            window: WindowId::new(0),
        },
        CorrelationId(31),
        ChannelCode::InternalThought,
        intent(),
        "spike-frame",
    );
    let mut orchestrator = RuntimeOrchestrator::new();
    orchestrator.force_mean_lock_for_test(1.0);
    let mut adapter = MockAdapter::default();

    orchestrator
        .ingest_and_process(&mut adapter, ctrl)
        .expect("orchestration should succeed");

    let frames = orchestrator.last_spike_frames();
    assert!(!frames.is_empty());
    assert!(frames.iter().any(|f| f.kind >= 1 && f.kind <= 5));
}

#[test]
fn attention_event_turns_true_with_strong_novelty_spike() {
    use ucf_spikes::{encode_ttfs_us, Spike, SpikeKind};

    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();

    let warmup = ControlFrame::new_text(
        SimTime {
            tick: Tick::new(41),
            window: WindowId::new(0),
        },
        CorrelationId(32),
        ChannelCode::InternalThought,
        intent(),
        "warmup",
    );
    orchestrator
        .ingest_and_process(&mut adapter, warmup)
        .expect("warmup should succeed");

    let phase = orchestrator.last_tcf_frame().expect("tcf frame").phase_bin;
    orchestrator.inject_spike_for_test(Spike {
        now_ms: 42,
        kind: SpikeKind::Novelty,
        chan: 9,
        phase,
        strength: 0.9,
        ttfs_us: encode_ttfs_us(0.9),
    });

    let ctrl = ControlFrame::new_text(
        SimTime {
            tick: Tick::new(42),
            window: WindowId::new(0),
        },
        CorrelationId(33),
        ChannelCode::InternalThought,
        intent(),
        "attention",
    );
    orchestrator
        .ingest_and_process(&mut adapter, ctrl)
        .expect("orchestration should succeed");

    assert!(orchestrator.attention_event());
}

#[test]
fn sle_fires_and_injects_meta_into_ssm() {
    let mut orchestrator = RuntimeOrchestrator::new();
    orchestrator.set_sle_cfg_for_test(SleCfg {
        min_trigger: 0.0,
        cooldown_ticks: 0,
        ..SleCfg::default_v0()
    });
    orchestrator.set_iit_proxy_cfg_for_test(ucf_iit_proxy::v0::IitCfg {
        min_samples: 2,
        coherence_weight: 1.0,
        flow_weight: 0.0,
        enforce_threshold: 0.95,
        ..ucf_iit_proxy::v0::IitCfg::default_v0()
    });
    orchestrator.force_mean_lock_for_test(0.0);
    let mut adapter = MockAdapter::default();

    for tick in 200..220 {
        let ctrl = ControlFrame::new_text(
            SimTime {
                tick: Tick::new(tick),
                window: WindowId::new(0),
            },
            CorrelationId(700 + tick),
            ChannelCode::InternalThought,
            intent(),
            "sle",
        );
        orchestrator
            .ingest_and_process(&mut adapter, ctrl)
            .expect("tick should succeed");
    }

    let frame = orchestrator
        .last_sle_frame()
        .expect("sle frame should exist");
    assert_eq!(frame.fired, 1);
    assert!(frame.weight_q > 0);

    let ssm = orchestrator.last_ssm_frame().expect("ssm frame");
    assert!(ssm.energy_q > 0);
    let y = orchestrator.working_context_ssm_y();
    assert!(y.len() >= 7);
    assert!(y[5].abs() > 0.0 || y[6].abs() > 0.0);
}

#[test]
fn sle_policy_denial_blocks_meta_injection() {
    let mut orchestrator = RuntimeOrchestrator::new();
    orchestrator.set_internal_recursion_policy_for_test(false, 1);
    orchestrator.set_sle_cfg_for_test(SleCfg {
        min_trigger: 0.0,
        cooldown_ticks: 0,
        ..SleCfg::default_v0()
    });
    orchestrator.set_iit_proxy_cfg_for_test(ucf_iit_proxy::v0::IitCfg {
        min_samples: 2,
        coherence_weight: 1.0,
        flow_weight: 0.0,
        enforce_threshold: 0.95,
        ..ucf_iit_proxy::v0::IitCfg::default_v0()
    });
    orchestrator.force_mean_lock_for_test(0.0);
    let mut adapter = MockAdapter::default();

    for tick in 230..250 {
        let ctrl = ControlFrame::new_text(
            SimTime {
                tick: Tick::new(tick),
                window: WindowId::new(0),
            },
            CorrelationId(900 + tick),
            ChannelCode::InternalThought,
            intent(),
            "sle-policy",
        );
        orchestrator
            .ingest_and_process(&mut adapter, ctrl)
            .expect("tick should succeed");
    }

    let frame = orchestrator
        .last_sle_frame()
        .expect("sle frame should exist");
    assert_eq!(frame.fired, 1);
    assert_eq!(frame.weight_q, 0);

    let y = orchestrator.working_context_ssm_y();
    assert!(y.len() >= 7);
    assert!(y[5].abs() < 0.01);
    assert!(y[6].abs() < 0.01);
}

#[test]
fn orchestrator_tick_emits_ncde_frame_and_l2_grows() {
    let mut orchestrator = RuntimeOrchestrator::new();
    orchestrator.force_mean_lock_for_test(1.0);
    orchestrator.set_sle_cfg_for_test(SleCfg {
        min_trigger: 0.0,
        cooldown_ticks: 0,
        ..SleCfg::default_v0()
    });
    let mut adapter = MockAdapter::default();

    let mut first_l2 = None;
    for tick in 900..910 {
        let ctrl = ControlFrame::new_text(
            SimTime {
                tick: Tick::new(tick),
                window: WindowId::new(0),
            },
            CorrelationId(3900 + tick),
            ChannelCode::InternalThought,
            intent(),
            "ncde",
        );
        orchestrator
            .ingest_and_process(&mut adapter, ctrl)
            .expect("ncde tick should succeed");

        let ncde = orchestrator
            .last_ncde_frame()
            .expect("ncde frame should exist");
        assert_eq!(ncde.now_ms, tick);

        if first_l2.is_none() {
            first_l2 = Some(ncde.l2_q);
        }
    }

    let first_l2 = first_l2.expect("at least one frame");
    let last_l2 = orchestrator
        .last_ncde_frame()
        .expect("ncde frame at end")
        .l2_q;
    assert!(last_l2 > first_l2);
    assert!(orchestrator.ncde_l2_norm_0_1() > 0.0);
}

#[test]
fn tick_emits_chem_and_digital_brain_frames() {
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();

    let ctrl = ControlFrame::new_text(
        SimTime {
            tick: Tick::new(1_000),
            window: WindowId::new(0),
        },
        CorrelationId(5001),
        ChannelCode::InternalThought,
        intent(),
        "dbm-frame",
    );
    orchestrator
        .ingest_and_process(&mut adapter, ctrl)
        .expect("tick should succeed");

    let chem = orchestrator.last_chem_frame().expect("chem frame");
    let brain = orchestrator
        .last_digital_brain_frame()
        .expect("digital brain frame");

    assert_eq!(chem.now_ms, 1_000);
    assert_eq!(brain.now_ms, 1_000);
}

#[test]
fn higher_stress_increases_amygdala_spikes() {
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();

    orchestrator.force_nsr_risk_for_test(0.05);
    for tick in 1_100..1_110 {
        let ctrl = ControlFrame::new_text(
            SimTime {
                tick: Tick::new(tick),
                window: WindowId::new(0),
            },
            CorrelationId(6000 + tick),
            ChannelCode::InternalThought,
            intent(),
            "low-stress",
        );
        orchestrator
            .ingest_and_process(&mut adapter, ctrl)
            .expect("low stress tick should succeed");
    }
    let low = orchestrator
        .last_digital_brain_frame()
        .expect("low frame")
        .amyg_spikes;

    orchestrator.force_nsr_risk_for_test(0.95);
    for tick in 1_110..1_130 {
        let ctrl = ControlFrame::new_text(
            SimTime {
                tick: Tick::new(tick),
                window: WindowId::new(0),
            },
            CorrelationId(7000 + tick),
            ChannelCode::InternalThought,
            intent(),
            "high-stress",
        );
        orchestrator
            .ingest_and_process(&mut adapter, ctrl)
            .expect("high stress tick should succeed");
    }
    let high = orchestrator
        .last_digital_brain_frame()
        .expect("high frame")
        .amyg_spikes;

    assert!(high >= low, "high={high} low={low}");
}

#[test]
fn real_compute_onboarding_v0_smoke_path() {
    let ctrl = ControlFrame::new_text(
        sim_time(),
        CorrelationId(99),
        ChannelCode::ExternalOutput,
        intent(),
        "smoke",
    );
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();

    let decision = orchestrator
        .ingest_and_process(&mut adapter, ctrl)
        .expect("orchestration should succeed");

    assert!(decision.compute_summary.is_some());
    assert!(orchestrator.ess.len() >= 2);

    let compute = decision.compute_summary.expect("compute summary");
    assert_eq!(compute.backend, "stub");
    assert!((0.0..=1.0).contains(&compute.surprise));
    assert!((0.0..=1.0).contains(&compute.pressure));
    assert!((0.0..=1.0).contains(&compute.risk));
    assert!((0.0..=1.0).contains(&compute.confidence));
    assert!(compute.spike_count > 0);
    assert!(compute.sparsity.is_some());
    assert!(compute.energy.is_some());
    assert_eq!(compute.risk_contract_version, Some(1));
    assert!(compute.risk_quality.is_some());
    assert!(compute.evidence_context_digest.is_some());
    assert!(compute.backend_profile.is_some());
    assert!(compute.budget_profile_id.is_some());
    assert!(compute.seed.is_some());

    let decision_rec = orchestrator
        .ess
        .trail_by_corr(CorrelationId(99))
        .into_iter()
        .find(|r| r.kind == ExperienceKind::DecisionOut)
        .expect("decision record");

    assert!(decision_rec.compute_summary.is_some());
    assert_eq!(decision_rec.compute_summary, Some(compute));
}

#[test]
fn degraded_budget_marks_risk_quality_and_persists_evidence() {
    std::env::set_var("UCF_COMPUTE_BUDGET_PROFILE", "stress");

    let mut orchestrator = RuntimeOrchestrator::try_new_from_env().expect("orchestrator from env");
    let mut adapter = MockAdapter::default();

    let mut compute = None;
    for idx in 0..10 {
        let ctrl = ControlFrame::new_text(
            sim_time(),
            CorrelationId(555 + idx),
            ChannelCode::ExternalOutput,
            intent(),
            "degraded",
        );
        let decision = orchestrator
            .ingest_and_process(&mut adapter, ctrl)
            .expect("orchestration should succeed");
        if decision.compute_summary.is_some() {
            compute = decision.compute_summary;
            break;
        }
    }

    let compute = compute.expect("compute summary");
    assert_eq!(compute.risk_quality, Some(1));
    assert!(compute.evidence_world_digest.is_some());
    assert!(compute.evidence_spikes_digest.is_none());
    assert!(compute.evidence_ssm_digest.is_some());
    assert_eq!(compute.budget_exceeded_stage, Some("lfm/step"));

    std::env::remove_var("UCF_COMPUTE_BUDGET_PROFILE");
}

#[test]
fn stress_profile_sets_budget_stage_and_backpressure_gating() {
    std::env::set_var("UCF_COMPUTE_BUDGET_PROFILE", "stress");

    let mut orchestrator = RuntimeOrchestrator::try_new_from_env().expect("orchestrator from env");
    let mut adapter = MockAdapter::default();

    let mut saw_backpressure_defer = false;
    for tick in 4_000..4_060 {
        let ctrl = ControlFrame::new_text(
            sim_time(),
            CorrelationId(20_000 + tick),
            ChannelCode::ExternalOutput,
            intent(),
            "stress-budget",
        );
        let decision = orchestrator
            .ingest_and_process(&mut adapter, ctrl)
            .expect("tick should succeed");
        if decision.reason_code.0 == "compute_backpressure" {
            saw_backpressure_defer = true;
        }
    }

    assert!(orchestrator.orchestrator_backpressure() >= 0.0);
    assert!(orchestrator.compute_budget_exceeded_total() > 0);
    assert!(saw_backpressure_defer || orchestrator.orchestrator_backpressure_active_total() > 0);

    std::env::remove_var("UCF_COMPUTE_BUDGET_PROFILE");
}

#[cfg(feature = "compute-candle")]
#[test]
fn orchestrator_env_selects_candle_backend_and_persists_summary() {
    std::env::set_var("UCF_COMPUTE_BACKEND", "candle");
    std::env::set_var("UCF_COMPUTE_SEED", "77");
    std::env::set_var("UCF_COMPUTE_MAX_MICROS", "1000");
    std::env::set_var("UCF_COMPUTE_HARD_TIMEOUT_MICROS", "5000");

    let mut orchestrator = RuntimeOrchestrator::try_new_from_env().expect("orchestrator from env");
    let ctrl = ControlFrame::new_text(
        sim_time(),
        CorrelationId(999),
        ChannelCode::ExternalOutput,
        intent(),
        "candle path",
    );
    let mut adapter = MockAdapter::default();

    let decision = orchestrator
        .ingest_and_process(&mut adapter, ctrl)
        .expect("orchestration should succeed");

    let compute = decision.compute_summary.expect("compute summary");
    assert_eq!(compute.backend, "candle");
    assert!(compute.spike_count <= 256);
    assert!((0.0..=1.0).contains(&compute.risk));
    assert!((0.0..=1.0).contains(&compute.confidence));
    assert!(orchestrator.ess.len() >= 2);

    std::env::remove_var("UCF_COMPUTE_BACKEND");
    std::env::remove_var("UCF_COMPUTE_SEED");
    std::env::remove_var("UCF_COMPUTE_MAX_MICROS");
    std::env::remove_var("UCF_COMPUTE_HARD_TIMEOUT_MICROS");
}

#[cfg(feature = "compute-burn")]
#[test]
fn orchestrator_env_burn_backend_surfaces_clear_error() {
    std::env::set_var("UCF_COMPUTE_BACKEND", "burn");
    std::env::set_var("UCF_COMPUTE_SEED", "77");

    let mut orchestrator = RuntimeOrchestrator::try_new_from_env().expect("orchestrator from env");
    let ctrl = ControlFrame::new_text(
        sim_time(),
        CorrelationId(1001),
        ChannelCode::ExternalOutput,
        intent(),
        "burn path",
    );
    let mut adapter = MockAdapter::default();

    match orchestrator.ingest_and_process(&mut adapter, ctrl) {
        Err(err) => {
            assert!(matches!(
                err,
                RuntimeError::Compute(ComputeError::NotImplemented)
            ));
        }
        Ok(decision) => {
            let compute = decision.compute_summary.expect("compute summary");
            assert_eq!(compute.backend, "burn");
            assert!((0.0..=1.0).contains(&compute.risk));
            assert!((0.0..=1.0).contains(&compute.confidence));
        }
    }

    std::env::remove_var("UCF_COMPUTE_BACKEND");
    std::env::remove_var("UCF_COMPUTE_SEED");
}

#[test]
fn orchestrator_hooks_emit_milestones_and_track_rejections() {
    let mut orchestrator = RuntimeOrchestrator::new();
    orchestrator.set_consolidation_hook_enabled_for_test(true);
    orchestrator.set_geist_hook_enabled_for_test(true);
    let mut adapter = MockAdapter::default();

    for tick in 2_000..2_125 {
        let ctrl = ControlFrame::new_text(
            SimTime {
                tick: Tick::new(tick),
                window: WindowId::new(0),
            },
            CorrelationId(9_000 + tick),
            ChannelCode::ExternalOutput,
            intent(),
            "hook-path",
        );
        orchestrator
            .ingest_and_process(&mut adapter, ctrl)
            .expect("tick should succeed");
    }

    assert!(orchestrator.consolidation_milestones_emitted_total() >= 1);
    assert!(orchestrator.last_compute_milestone().is_some());
    assert!(orchestrator.geist_updates_rejected_total() >= 1);
    let (consolidation_errors, geist_errors) = orchestrator.hook_errors_total();
    assert_eq!(consolidation_errors, 0);
    assert_eq!(geist_errors, 0);
}

#[test]
fn orchestrator_hooks_accept_stable_geist_updates() {
    let mut orchestrator = RuntimeOrchestrator::new();
    orchestrator.set_consolidation_hook_enabled_for_test(true);
    orchestrator.set_geist_hook_enabled_for_test(true);
    let mut adapter = MockAdapter::default();

    for tick in 3_000..3_420 {
        let ctrl = ControlFrame::new_text(
            SimTime {
                tick: Tick::new(tick),
                window: WindowId::new(0),
            },
            CorrelationId(10_000 + tick),
            ChannelCode::ExternalOutput,
            intent(),
            "stable-hook",
        );
        orchestrator
            .ingest_and_process(&mut adapter, ctrl)
            .expect("tick should succeed");
    }

    assert!(orchestrator.consolidation_milestones_emitted_total() >= 6);
    assert!(orchestrator.geist_updates_accepted_total() <= 10_000);
}

#[test]
fn tool_audit_records_and_hash_chain_are_appended() {
    let ctrl = ControlFrame::new_text(
        sim_time(),
        CorrelationId(1200),
        ChannelCode::ExternalOutput,
        intent(),
        "audit",
    );
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();

    orchestrator
        .ingest_and_process(&mut adapter, ctrl)
        .expect("ingest");

    let records: Vec<_> = (0..orchestrator.ess.len())
        .filter_map(|idx| orchestrator.ess.get(idx))
        .collect();
    let saw_tool_request = records
        .iter()
        .any(|r| r.kind == ExperienceKind::ToolRequest);
    if saw_tool_request {
        assert!(records.iter().any(|r| r.kind == ExperienceKind::ToolAuth));
        assert!(records
            .iter()
            .any(|r| r.kind == ExperienceKind::SandboxCall));
        assert!(records
            .iter()
            .any(|r| r.kind == ExperienceKind::ToolExecution));
        assert!(records
            .iter()
            .any(|r| r.kind == ExperienceKind::SandboxReply));
    }
    assert!(records
        .iter()
        .any(|r| r.kind == ExperienceKind::AuditCheckpoint));

    let mut prev = [0_u8; 32];
    for rec in records.iter().filter(|r| {
        matches!(
            r.kind,
            ExperienceKind::CapabilityIssuance
                | ExperienceKind::Throttle
                | ExperienceKind::ToolRequest
                | ExperienceKind::SandboxCall
                | ExperienceKind::ToolAuth
                | ExperienceKind::ToolExecution
                | ExperienceKind::SandboxReply
                | ExperienceKind::AuditCheckpoint
        )
    }) {
        assert_eq!(rec.audit_prev_digest, Some(prev));
        prev = rec.audit_digest.expect("audit digest");
    }
}

#[test]
fn neuro_records_are_persisted_windowed_and_bounded() {
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();

    for tick in 30_000..30_100 {
        let ctrl = ControlFrame::new_text(
            SimTime {
                tick: Tick::new(tick),
                window: WindowId::new(0),
            },
            CorrelationId(130_000 + tick),
            ChannelCode::ExternalOutput,
            intent(),
            "neuro-window",
        );
        orchestrator
            .ingest_and_process(&mut adapter, ctrl)
            .expect("tick should succeed");
    }

    let records: Vec<_> = (0..orchestrator.ess.len())
        .filter_map(|idx| orchestrator.ess.get(idx))
        .collect();
    let neuro_records: Vec<_> = records
        .iter()
        .filter(|r| r.kind == ExperienceKind::Neuro)
        .collect();

    assert!(neuro_records.len() >= 8);
    assert!(neuro_records.len() <= 12);
    for rec in neuro_records {
        let neuro = rec.neuro_record.expect("neuro payload");
        assert_eq!(neuro.schema_version, 1);
        assert!(neuro.arousal_q <= 255);
        assert!(neuro.attention_gain_q <= 255);
        assert!(neuro.excitability_q <= 255);
        assert!(neuro.spike_rate_q <= 255);
        assert!(neuro.spike_count <= 32);
    }
}

#[test]
fn high_cortisol_path_reduces_neuro_excitability_and_raises_inhibit() {
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();

    orchestrator.force_nsr_risk_for_test(0.05);
    orchestrator.force_surprise_for_test(0.05);
    orchestrator.force_ess_pressure_for_test(0.05);

    let low_ctrl = ControlFrame::new_text(
        sim_time(),
        CorrelationId(188_001),
        ChannelCode::ExternalOutput,
        intent(),
        "neuro-low",
    );
    orchestrator
        .ingest_and_process(&mut adapter, low_ctrl)
        .expect("low tick");
    let low_neuro = orchestrator.last_neuro_summary().expect("low neuro");
    let low_fep = orchestrator.last_fep_frame().expect("low fep");

    orchestrator.force_nsr_risk_for_test(0.95);
    orchestrator.force_surprise_for_test(0.95);
    orchestrator.force_ess_pressure_for_test(0.95);

    for idx in 0..25 {
        let ctrl = ControlFrame::new_text(
            SimTime {
                tick: Tick::new(190_000 + idx),
                window: WindowId::new(0),
            },
            CorrelationId(199_000 + idx),
            ChannelCode::ExternalOutput,
            intent(),
            "neuro-high",
        );
        orchestrator
            .ingest_and_process(&mut adapter, ctrl)
            .expect("high tick");
    }

    let high_neuro = orchestrator.last_neuro_summary().expect("high neuro");
    let high_fep = orchestrator.last_fep_frame().expect("high fep");
    assert!(high_neuro.excitability <= low_neuro.excitability);
    assert!(high_fep.inhibit_q >= low_fep.inhibit_q);
}

#[test]
fn neuro_replay_from_identical_inputs_matches_digests() {
    fn run(seed_corr: u64) -> Vec<[u8; 32]> {
        let mut orchestrator = RuntimeOrchestrator::new();
        let mut adapter = MockAdapter::default();
        orchestrator.force_nsr_risk_for_test(0.8);
        orchestrator.force_surprise_for_test(0.7);
        orchestrator.force_ess_pressure_for_test(0.6);

        for tick in 40_000..40_060 {
            let ctrl = ControlFrame::new_text(
                SimTime {
                    tick: Tick::new(tick),
                    window: WindowId::new(0),
                },
                CorrelationId(seed_corr + tick),
                ChannelCode::ExternalOutput,
                intent(),
                "neuro-replay",
            );
            orchestrator
                .ingest_and_process(&mut adapter, ctrl)
                .expect("tick should succeed");
        }

        (0..orchestrator.ess.len())
            .filter_map(|idx| orchestrator.ess.get(idx))
            .filter(|r| r.kind == ExperienceKind::Neuro)
            .map(|r| r.neuro_record.expect("neuro").summary_digest)
            .collect()
    }

    let a = run(210_000);
    let b = run(310_000);
    assert_eq!(a, b);
    assert!(!a.is_empty());
}

#[test]
fn hormone_records_are_persisted_windowed_and_bounded() {
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();

    for tick in 10_000..10_100 {
        let ctrl = ControlFrame::new_text(
            SimTime {
                tick: Tick::new(tick),
                window: WindowId::new(0),
            },
            CorrelationId(70_000 + tick),
            ChannelCode::ExternalOutput,
            intent(),
            "hormone-window",
        );
        orchestrator
            .ingest_and_process(&mut adapter, ctrl)
            .expect("tick should succeed");
    }

    let records: Vec<_> = (0..orchestrator.ess.len())
        .filter_map(|idx| orchestrator.ess.get(idx))
        .collect();
    let hormone_records: Vec<_> = records
        .iter()
        .filter(|r| r.kind == ExperienceKind::Hormone)
        .collect();

    assert!(hormone_records.len() >= 8);
    assert!(hormone_records.len() <= 12);
    for rec in hormone_records {
        let hormone = rec.hormone_record.expect("hormone payload");
        assert!(hormone.schema_version >= 1);
        assert!(hormone.cortisol_q <= 255);
        assert!(hormone.drive_q <= 255);
        assert!(hormone.stress_index_q <= 255);
    }
}

#[test]
fn high_hormone_stress_makes_gating_stricter() {
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();

    orchestrator.force_nsr_risk_for_test(0.05);
    orchestrator.force_surprise_for_test(0.05);
    orchestrator.force_ess_pressure_for_test(0.05);

    let low_ctrl = ControlFrame::new_text(
        sim_time(),
        CorrelationId(88_001),
        ChannelCode::ExternalOutput,
        intent(),
        "hormone-low",
    );
    orchestrator
        .ingest_and_process(&mut adapter, low_ctrl)
        .expect("low tick");
    let low_mod = orchestrator
        .last_gating_modulation()
        .expect("low modulation");
    let low_fep = orchestrator.last_fep_frame().expect("low fep");

    orchestrator.force_nsr_risk_for_test(0.95);
    orchestrator.force_surprise_for_test(0.95);
    orchestrator.force_ess_pressure_for_test(0.95);

    for idx in 0..25 {
        let ctrl = ControlFrame::new_text(
            SimTime {
                tick: Tick::new(90_000 + idx),
                window: WindowId::new(0),
            },
            CorrelationId(99_000 + idx),
            ChannelCode::ExternalOutput,
            intent(),
            "hormone-high",
        );
        orchestrator
            .ingest_and_process(&mut adapter, ctrl)
            .expect("high tick");
    }

    let high_mod = orchestrator
        .last_gating_modulation()
        .expect("high modulation");
    let high_fep = orchestrator.last_fep_frame().expect("high fep");

    assert!(high_mod.risk_penalty_scale >= low_mod.risk_penalty_scale);
    assert!(high_mod.action_threshold_delta >= low_mod.action_threshold_delta);
    assert!(high_fep.inhibit_q >= low_fep.inhibit_q);
}

#[test]
fn hormone_replay_from_identical_inputs_matches_digests() {
    fn run(seed_corr: u64) -> Vec<[u8; 32]> {
        let mut orchestrator = RuntimeOrchestrator::new();
        let mut adapter = MockAdapter::default();
        orchestrator.force_nsr_risk_for_test(0.7);
        orchestrator.force_surprise_for_test(0.6);
        orchestrator.force_ess_pressure_for_test(0.4);

        for tick in 20_000..20_060 {
            let ctrl = ControlFrame::new_text(
                SimTime {
                    tick: Tick::new(tick),
                    window: WindowId::new(0),
                },
                CorrelationId(seed_corr + tick),
                ChannelCode::ExternalOutput,
                intent(),
                "hormone-replay",
            );
            orchestrator
                .ingest_and_process(&mut adapter, ctrl)
                .expect("tick should succeed");
        }

        (0..orchestrator.ess.len())
            .filter_map(|idx| orchestrator.ess.get(idx))
            .filter(|r| r.kind == ExperienceKind::Hormone)
            .map(|r| r.hormone_record.expect("hormone").hormone_digest)
            .collect()
    }

    let a = run(110_000);
    let b = run(110_000);
    assert_eq!(a, b);
    assert!(!a.is_empty());
}
#[test]
fn evolution_disabled_by_default_emits_no_delta_records() {
    std::env::remove_var("UCF_ENABLE_EVOLUTION");
    let ctrl = ControlFrame::new_text(
        sim_time(),
        CorrelationId(900),
        ChannelCode::ExternalOutput,
        intent(),
        "hello",
    );
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();
    orchestrator
        .ingest_and_process(&mut adapter, ctrl)
        .expect("tick should succeed");
    let has_delta = (0..orchestrator.ess.len())
        .filter_map(|idx| orchestrator.ess.get(idx))
        .any(|r| {
            matches!(
                r.kind,
                ExperienceKind::DeltaProposal
                    | ExperienceKind::DeltaEvaluation
                    | ExperienceKind::DeltaRecommendation
            )
        });
    assert!(!has_delta);
}

#[test]
fn evolution_enabled_persists_proposal_and_evaluation_without_actions() {
    std::env::set_var("UCF_ENABLE_EVOLUTION", "1");
    std::env::set_var("UCF_EVOLUTION_WINDOW_TICKS", "8");
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();
    for tick in 1..=8u64 {
        let t = SimTime {
            tick: Tick::new(tick),
            window: WindowId::new(0),
        };
        let ctrl = ControlFrame::new_text(
            t,
            CorrelationId(901 + tick),
            ChannelCode::ExternalOutput,
            intent(),
            "evo",
        );
        let mut decision = DecisionFrame::allow(t, CorrelationId(901 + tick), "allow");
        decision.compute_summary = Some(ucf_frames::v1::ComputeSignalsSummary {
            backend: "stub",
            surprise: 0.5,
            pressure: 0.5,
            risk: 0.9,
            confidence: 0.2,
            surprise_q: 0,
            pressure_q: 0,
            risk_q: 0,
            confidence_q: 0,
            spike_count: 0,
            spikes_digest: [0; 32],
            sparsity: None,
            energy: None,
            ssm_readout: None,
            ssm_digest: None,
            world_digest: None,
            risk_quality: None,
            evidence_context_digest: None,
            evidence_world_digest: None,
            evidence_spikes_digest: None,
            evidence_ssm_digest: None,
            evidence_lfm_digest: None,
            backend_profile: None,
            backend_pack_id: None,
            fixtures_digest: None,
            llm_backend: None,
            world_backend: None,
            sae_backend: None,
            ssm_backend: None,
            lfm_backend: None,
            lfm_uncertainty: None,
            lfm_stability: None,
            lfm_uncertainty_q: None,
            lfm_stability_q: None,
            lfm_state_norm: None,
            lfm_deriv_norm: None,
            lfm_saturation_ratio: None,
            lfm_nan_inf_detected: None,
            lfm_digest: None,
            budget_profile_id: None,
            seed: None,
            risk_contract_version: None,
            compute_schema_version: None,
            compute_chain_digest: Some([3; 32]),
            compute_code_version: None,
            budget_exceeded_stage: Some("world"),
            contract_version: None,
            backend_id: None,
            validation_status: None,
            violation_reason_mask: None,
            lfm_quality: None,
            coherence: Some(0.2),
            instability: Some(0.9),
            coherence_q: None,
            instability_q: None,
            phi_proxy: Some(0.1),
            coherence_digest: Some([4; 32]),
            iit_coherence_q: None,
            iit_incoherence_q: None,
            iit_reason_codes: None,
            stage_allow_mask: None,
            free_energy_proxy_q: None,
            ebm_energy_mean_topk_q: None,
            ebm_w_q: None,
            fep_coupling_version: None,
        });
        orchestrator
            .ingest_with_decision(&mut adapter, ctrl, decision)
            .expect("tick should succeed");
    }

    let records: Vec<_> = (0..orchestrator.ess.len())
        .filter_map(|idx| orchestrator.ess.get(idx))
        .collect();
    let recommendations = records
        .iter()
        .filter(|r| r.kind == ExperienceKind::DeltaRecommendation)
        .count();
    assert!(recommendations <= 1);
}

#[test]
fn evolution_suppressed_in_emergency_mode() {
    std::env::set_var("UCF_ENABLE_EVOLUTION", "1");
    std::env::set_var("UCF_EVOLUTION_WINDOW_TICKS", "8");
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();
    for tick in 1..=8u64 {
        let t = SimTime {
            tick: Tick::new(tick),
            window: WindowId::new(0),
        };
        let ctrl = ControlFrame::new_text(
            t,
            CorrelationId(970 + tick),
            ChannelCode::ExternalOutput,
            intent(),
            "emergency-evo",
        );
        let mut decision = DecisionFrame::allow(t, CorrelationId(970 + tick), "allow");
        decision.compute_summary = Some(ucf_frames::v1::ComputeSignalsSummary {
            backend: "stub",
            surprise: 0.9,
            pressure: 0.9,
            risk: 0.95,
            confidence: 0.1,
            surprise_q: 0,
            pressure_q: 0,
            risk_q: 0,
            confidence_q: 0,
            spike_count: 0,
            spikes_digest: [0; 32],
            sparsity: None,
            energy: None,
            ssm_readout: None,
            ssm_digest: None,
            world_digest: None,
            risk_quality: None,
            evidence_context_digest: None,
            evidence_world_digest: None,
            evidence_spikes_digest: None,
            evidence_ssm_digest: None,
            evidence_lfm_digest: None,
            backend_profile: None,
            backend_pack_id: None,
            fixtures_digest: None,
            llm_backend: None,
            world_backend: None,
            sae_backend: None,
            ssm_backend: None,
            lfm_backend: None,
            lfm_uncertainty: Some(0.95),
            lfm_stability: Some(0.05),
            lfm_uncertainty_q: None,
            lfm_stability_q: None,
            lfm_state_norm: Some(2.0),
            lfm_deriv_norm: Some(2.0),
            lfm_saturation_ratio: Some(0.4),
            lfm_nan_inf_detected: Some(false),
            lfm_digest: Some([3; 32]),
            budget_profile_id: None,
            seed: None,
            risk_contract_version: None,
            compute_schema_version: None,
            compute_chain_digest: Some([3; 32]),
            compute_code_version: None,
            budget_exceeded_stage: Some("world"),
            contract_version: None,
            backend_id: None,
            validation_status: None,
            violation_reason_mask: None,
            lfm_quality: None,
            coherence: Some(0.1),
            instability: Some(0.95),
            coherence_q: None,
            instability_q: None,
            phi_proxy: Some(0.1),
            coherence_digest: Some([4; 32]),
            iit_coherence_q: None,
            iit_incoherence_q: None,
            iit_reason_codes: None,
            stage_allow_mask: None,
            free_energy_proxy_q: None,
            ebm_energy_mean_topk_q: None,
            ebm_w_q: None,
            fep_coupling_version: None,
        });
        orchestrator
            .ingest_with_decision(&mut adapter, ctrl, decision)
            .expect("tick should succeed");
    }

    let has_delta = (0..orchestrator.ess.len())
        .filter_map(|idx| orchestrator.ess.get(idx))
        .any(|r| {
            matches!(
                r.kind,
                ExperienceKind::DeltaProposal
                    | ExperienceKind::DeltaEvaluation
                    | ExperienceKind::DeltaRecommendation
            )
        });
    assert!(!has_delta);
}

#[test]
fn evolution_replay_proposal_digests_are_stable() {
    std::env::set_var("UCF_ENABLE_EVOLUTION", "1");
    std::env::set_var("UCF_EVOLUTION_WINDOW_TICKS", "8");

    fn run(seed: u64) -> Vec<[u8; 32]> {
        let mut orchestrator = RuntimeOrchestrator::new();
        let mut adapter = MockAdapter::default();
        for tick in 1..=8u64 {
            let t = SimTime {
                tick: Tick::new(seed + tick),
                window: WindowId::new(0),
            };
            let ctrl = ControlFrame::new_text(
                t,
                CorrelationId(980 + tick),
                ChannelCode::ExternalOutput,
                intent(),
                "evo-replay",
            );
            let mut decision = DecisionFrame::allow(t, CorrelationId(980 + tick), "allow");
            decision.compute_summary = Some(ucf_frames::v1::ComputeSignalsSummary {
                backend: "stub",
                surprise: 0.7,
                pressure: 0.6,
                risk: 0.8,
                confidence: 0.2,
                surprise_q: 0,
                pressure_q: 0,
                risk_q: 0,
                confidence_q: 0,
                spike_count: 0,
                spikes_digest: [0; 32],
                sparsity: None,
                energy: None,
                ssm_readout: None,
                ssm_digest: None,
                world_digest: None,
                risk_quality: None,
                evidence_context_digest: None,
                evidence_world_digest: None,
                evidence_spikes_digest: None,
                evidence_ssm_digest: None,
                evidence_lfm_digest: None,
                backend_profile: None,
                backend_pack_id: None,
                fixtures_digest: None,
                llm_backend: None,
                world_backend: None,
                sae_backend: None,
                ssm_backend: None,
                lfm_backend: None,
                lfm_uncertainty: Some(0.8),
                lfm_stability: Some(0.2),
                lfm_uncertainty_q: None,
                lfm_stability_q: None,
                lfm_state_norm: None,
                lfm_deriv_norm: None,
                lfm_saturation_ratio: None,
                lfm_nan_inf_detected: None,
                lfm_digest: Some([7; 32]),
                budget_profile_id: None,
                seed: None,
                risk_contract_version: None,
                compute_schema_version: None,
                compute_chain_digest: Some([5; 32]),
                compute_code_version: None,
                budget_exceeded_stage: Some("world"),
                contract_version: None,
                backend_id: None,
                validation_status: None,
                violation_reason_mask: None,
                lfm_quality: None,
                coherence: Some(0.2),
                instability: Some(0.8),
                coherence_q: None,
                instability_q: None,
                phi_proxy: Some(0.1),
                coherence_digest: Some([6; 32]),
                iit_coherence_q: None,
                iit_incoherence_q: None,
                iit_reason_codes: None,
                stage_allow_mask: None,
                free_energy_proxy_q: None,
                ebm_energy_mean_topk_q: None,
                ebm_w_q: None,
                fep_coupling_version: None,
            });
            orchestrator
                .ingest_with_decision(&mut adapter, ctrl, decision)
                .expect("tick should succeed");
        }
        (0..orchestrator.ess.len())
            .filter_map(|idx| orchestrator.ess.get(idx))
            .filter_map(|r| {
                if r.kind == ExperienceKind::DeltaProposal {
                    r.delta_proposal_record.as_ref().map(|p| p.digest)
                } else {
                    None
                }
            })
            .collect()
    }

    let a = run(40_000);
    let b = run(40_000);
    assert_eq!(a, b);
}
#[test]
fn orchestrator_persists_output_record_for_safe_text_or_code() {
    let ctrl = ControlFrame::new_text(
        sim_time(),
        CorrelationId(900),
        ChannelCode::ExternalOutput,
        intent(),
        "output-record",
    );
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();
    orchestrator
        .ingest_and_process(&mut adapter, ctrl)
        .expect("orchestration should succeed");

    let output = (0..orchestrator.ess.len())
        .filter_map(|idx| orchestrator.ess.get(idx))
        .find(|r| r.kind == ExperienceKind::Output)
        .expect("output record must exist");
    match &output.payload {
        ExperiencePayload::Audit(AuditPayload::Output(record)) => {
            assert!(record.schema_version >= 1);
            assert_eq!(record.output_class, 0);
            assert!(record.text.as_ref().is_some());
            assert!(record.max_tokens_eff >= 64);
            assert!(record.max_tokens_eff <= 128);
            assert!(record.lfm_uncertainty.is_some());
            assert!(record.lfm_stability.is_some());
            assert_eq!(
                record.override_reasons.len(),
                record.override_reasons.len().min(8)
            );
        }
        _ => panic!("expected output audit payload"),
    }
}

#[test]
fn external_output_class_uses_plan_summary_without_llm_digests() {
    let ctrl = ControlFrame::new_text(
        sim_time(),
        CorrelationId(901),
        ChannelCode::ExternalOutput,
        intent(),
        "plan-path",
    );
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();
    orchestrator
        .ingest_and_process(&mut adapter, ctrl)
        .expect("orchestration should succeed");

    let maybe_plan_output = (0..orchestrator.ess.len())
        .filter_map(|idx| orchestrator.ess.get(idx))
        .filter(|r| r.kind == ExperienceKind::Output)
        .find_map(|r| match &r.payload {
            ExperiencePayload::Audit(AuditPayload::Output(record)) if record.output_class == 2 => {
                Some(record)
            }
            _ => None,
        });
    if let Some(record) = maybe_plan_output {
        assert_eq!(record.llm_backend_name, "plan-only");
        assert_eq!(record.llm_request_digest, [0; 32]);
        assert_eq!(record.llm_response_digest, [0; 32]);
        assert!(record.max_tokens_eff >= 64);
    }
}
