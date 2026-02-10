use ucf_core::types::{SimTime, Tick, WindowId};
use ucf_frames::v1::{
    ChannelCode, ControlFrame, ControlPayload, CorrelationId, DecisionCode, DecisionFrame, Intent,
    IntentId, IntentKind,
};
use ucf_policy::{adapter::MockAdapter, errors::PolicyError, gem::Gem, pbm::Pbm};

fn sim_time() -> SimTime {
    SimTime {
        tick: Tick::new(1),
        window: WindowId::new(0),
    }
}

fn intent() -> Intent {
    Intent::new(IntentId(7), IntentKind::System, "test")
}

#[test]
fn no_decision_no_action() {
    let ctrl = ControlFrame::new_text(
        sim_time(),
        CorrelationId(1),
        ChannelCode::ExternalOutput,
        intent(),
        "blocked",
    );
    let mut adapter = MockAdapter::default();

    let result = Gem::execute(&mut adapter, &ctrl, None);

    assert_eq!(result, Err(PolicyError::MissingDecision));
    assert!(adapter.emitted.is_empty());
    assert_eq!(adapter.brain_events, 0);
    assert_eq!(adapter.mem_writes, 0);
}

#[test]
fn pbm_denies_external_output_and_gem_noops_on_deny() {
    let ctrl = ControlFrame::new_text(
        sim_time(),
        CorrelationId(2),
        ChannelCode::ExternalOutput,
        intent(),
        "deny-me",
    );
    let decision = Pbm::decide(&ctrl);
    let mut adapter = MockAdapter::default();

    let result = Gem::execute(&mut adapter, &ctrl, Some(&decision));

    assert_eq!(decision.decision, DecisionCode::Deny);
    assert!(result.is_ok());
    assert!(adapter.emitted.is_empty());
    assert_eq!(adapter.brain_events, 0);
    assert_eq!(adapter.mem_writes, 0);
}

#[test]
fn internal_thought_allowed_but_not_externalized() {
    let ctrl = ControlFrame::new_text(
        sim_time(),
        CorrelationId(3),
        ChannelCode::InternalThought,
        intent(),
        "private-thought",
    );
    let decision = Pbm::decide(&ctrl);
    let mut adapter = MockAdapter::default();

    let result = Gem::execute(&mut adapter, &ctrl, Some(&decision));

    assert_eq!(decision.decision, DecisionCode::Allow);
    assert!(result.is_ok());
    assert!(adapter.emitted.is_empty());
    assert_eq!(adapter.brain_events, 0);
    assert_eq!(adapter.mem_writes, 0);
}

#[test]
fn external_output_text_with_allow_emits_text() {
    let ctrl = ControlFrame::new_text(
        sim_time(),
        CorrelationId(4),
        ChannelCode::ExternalOutput,
        intent(),
        "hello-world",
    );
    let decision = DecisionFrame::allow(sim_time(), CorrelationId(4), "allow_test");
    let mut adapter = MockAdapter::default();

    let result = Gem::execute(&mut adapter, &ctrl, Some(&decision));

    assert!(result.is_ok());
    assert_eq!(adapter.emitted, vec!["hello-world".to_string()]);
    assert_eq!(adapter.mem_writes, 0);
}

#[test]
fn memory_write_bytes_with_allow_writes_memory() {
    let ctrl = ControlFrame {
        time: sim_time(),
        corr: CorrelationId(5),
        channel: ChannelCode::MemoryWrite,
        intent: intent(),
        payload: ControlPayload::Bytes(vec![1_u8, 2, 3].into()),
    };
    let decision = DecisionFrame::allow(sim_time(), CorrelationId(5), "allow_mem");
    let mut adapter = MockAdapter::default();

    let result = Gem::execute(&mut adapter, &ctrl, Some(&decision));

    assert!(result.is_ok());
    assert_eq!(adapter.mem_writes, 1);
    assert!(adapter.emitted.is_empty());
}
