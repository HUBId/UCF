use ucf_cde::v0::CdeUpdateKind;
use ucf_core::types::{SimTime, Tick, WindowId};
use ucf_ess::v1::{ExperienceKind, ExperiencePayload, ExperienceStore};
use ucf_frames::v1::{
    BrainStimulusKind, BrainStimulusPayload, ChannelCode, ControlFrame, ControlPayload,
    CorrelationId, DecisionFrame, Intent, IntentId, IntentKind, NeuromodulatorSnapshot,
};
use ucf_policy::{adapter::MockAdapter, gem::Gem};
use ucf_runtime::RuntimeOrchestrator;

fn sim_time() -> SimTime {
    SimTime {
        tick: Tick::new(10),
        window: WindowId::new(0),
    }
}

fn intent() -> Intent {
    Intent::new(IntentId(1), IntentKind::System, "runtime-test")
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
    let mut adapter = MockAdapter::default();

    let decision = orchestrator
        .ingest_and_process(&mut adapter, ctrl)
        .expect("orchestration should succeed");

    assert_eq!(decision.decision, ucf_frames::v1::DecisionCode::Allow);
    assert_eq!(adapter.emitted, vec!["hi".to_string()]);
    assert_eq!(orchestrator.ess.len(), 2);
    assert_eq!(
        orchestrator.ess.get(0).map(|r| r.kind),
        Some(ExperienceKind::ControlIn)
    );
    assert_eq!(
        orchestrator.ess.get(1).map(|r| r.kind),
        Some(ExperienceKind::DecisionOut)
    );
    let baseline = NeuromodulatorSnapshot::baseline();
    assert_eq!(
        orchestrator.ess.get(0).and_then(|r| r.neuromod),
        Some(baseline)
    );
    assert!(orchestrator.ess.get(0).and_then(|r| r.iit_phi).is_some());
    assert_eq!(
        orchestrator.ess.get(1).and_then(|r| r.neuromod),
        Some(baseline)
    );
    let decision_record = orchestrator.ess.get(1).expect("decision record");
    assert_eq!(decision_record.decision_meta, Some(decision.meta));
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
    assert_eq!(orchestrator.ess.len(), 2);
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
    assert_eq!(orchestrator.ess.len(), 2);
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
    let allow_output = DecisionFrame::allow(sim_time(), CorrelationId(4), "allow_output");
    Gem::execute(&mut adapter, &output_ctrl, Some(&allow_output)).expect("gem should emit text");

    let memory_ctrl = ControlFrame {
        time: sim_time(),
        corr: CorrelationId(5),
        channel: ChannelCode::MemoryWrite,
        intent: intent(),
        payload: ControlPayload::Bytes(vec![9, 9].into()),
    };
    let allow_memory = DecisionFrame::allow(sim_time(), CorrelationId(5), "allow_memory");
    Gem::execute(&mut adapter, &memory_ctrl, Some(&allow_memory)).expect("gem should write memory");

    assert_eq!(adapter.emitted, vec!["allowed".to_string()]);
    assert_eq!(adapter.mem_writes, 1);
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
        .ingest_with_decision(
            &mut adapter,
            ctrl,
            DecisionFrame::allow(sim_time(), CorrelationId(6), "allow_brain"),
        )
        .expect("orchestration should succeed");

    let last = orchestrator
        .ess
        .get(orchestrator.ess.len() - 1)
        .expect("note record");
    assert_eq!(last.kind, ExperienceKind::Note);
    if let ExperiencePayload::Text(text) = &last.payload {
        assert_eq!(text.as_ref(), "brain_spikes:n=8,dst=44");
    } else {
        panic!("expected text note payload");
    }
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

    let control_record = orchestrator.ess.get(0).expect("control record");
    assert!(control_record.iit_phi.is_some());
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
    assert!((0.0..=1.0).contains(&frame.integration));
    assert!(frame.state <= 2);
}

#[test]
fn orchestrator_tick_emits_cde_and_nsr_frames() {
    let mut orchestrator = RuntimeOrchestrator::new();
    orchestrator.force_mean_lock_for_test(0.0);
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
    assert_eq!(iit.state, 2);
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
        assert!((0.0..=1.0).contains(&frame.gate));
    }
}

#[test]
fn ssm_gate_is_lower_for_fragmenting_than_stable_given_same_inputs() {
    let cfg = ucf_biophys::v0::SsmCfg::default();
    let mut stable_state = ucf_biophys::v0::SsmState::default();
    let mut fragmenting_state = ucf_biophys::v0::SsmState::default();
    let u = [0.5; ucf_biophys::v0::SSM_D];

    let stable = ucf_biophys::v0::ssm_step(
        &mut stable_state,
        &u,
        ucf_biophys::v0::SsmInputs {
            attention: 0.55,
            integration: 0.5,
            dopamine: 0.5,
            noise: 0.2,
        },
        cfg,
    );

    let fragmenting = ucf_biophys::v0::ssm_step(
        &mut fragmenting_state,
        &u,
        ucf_biophys::v0::SsmInputs {
            attention: 0.35,
            integration: 0.5,
            dopamine: 0.5,
            noise: 0.2,
        },
        cfg,
    );

    assert!(fragmenting.gate < stable.gate);
}

#[test]
fn orchestrator_tick_emits_onn_and_snn_frames_each_tick() {
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();

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
    }
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

    let low_gate = lo.last_ssm_frame().expect("ssm frame low").gate;
    let high_gate = hi.last_ssm_frame().expect("ssm frame hi").gate;
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
