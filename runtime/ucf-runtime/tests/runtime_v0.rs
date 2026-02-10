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
