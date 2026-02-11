use ucf_core::types::{SimTime, Tick, WindowId};
use ucf_frames::v1::{
    ChannelCode, ControlFrame, ControlPayload, CorrelationId, DecisionCode, DecisionFrame, Intent,
    IntentId, IntentKind,
};

fn sim_time(tick: u64) -> SimTime {
    SimTime {
        tick: Tick::new(tick),
        window: WindowId::new(0),
    }
}

fn intent() -> Intent {
    Intent::new(IntentId(11), IntentKind::System, "contract")
}

fn assert_invariants(ctrl: &ControlFrame, decision: &DecisionFrame) {
    assert!(ctrl.corr.0 > 0, "correlation id must be non-zero");
    assert!(decision.time.tick.get() >= ctrl.time.tick.get());
    assert!(!decision.reason_code.0.is_empty());
    assert!(decision.rationale.len() <= 256);
    assert!(matches!(
        decision.decision,
        DecisionCode::Allow | DecisionCode::Deny | DecisionCode::Defer
    ));
}

#[test]
fn control_decision_contract_invariants_hold() {
    let ctrl = ControlFrame::new_text(
        sim_time(10),
        CorrelationId(22),
        ChannelCode::InternalThought,
        intent(),
        "hello",
    );
    let decision = DecisionFrame::allow(sim_time(10), CorrelationId(22), "ok");
    assert_invariants(&ctrl, &decision);
}

#[test]
fn payload_channel_contract_is_consistent() {
    let ctrl = ControlFrame::new_text(
        sim_time(3),
        CorrelationId(7),
        ChannelCode::ExternalOutput,
        intent(),
        "out",
    );
    assert!(matches!(ctrl.payload, ControlPayload::Text(_)));
}
