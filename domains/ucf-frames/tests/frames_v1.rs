use ucf_core::types::{SimTime, Tick, WindowId};
use ucf_frames::v1::{
    ChannelCode, ControlFrame, ControlPayload, CorrelationId, DecisionCode, DecisionFrame,
    DenyReasonCode, Intent, IntentId, IntentKind, IntentType, ReasonCode,
};

#[test]
fn decision_deny_has_reason() {
    let frame = DecisionFrame::deny(
        SimTime {
            tick: Tick::new(42),
            window: WindowId::new(0),
        },
        CorrelationId(7),
        DenyReasonCode::PolicyViolation,
        "blocked by policy",
    );

    assert_eq!(frame.decision, DecisionCode::Deny);
    assert_eq!(frame.deny_reason, Some(DenyReasonCode::PolicyViolation));
}

#[test]
fn decision_allow_has_no_reason() {
    let frame = DecisionFrame::allow(
        SimTime {
            tick: Tick::new(42),
            window: WindowId::new(0),
        },
        CorrelationId(7),
        "allowed",
    );

    assert_eq!(frame.decision, DecisionCode::Allow);
    assert_eq!(frame.deny_reason, None);
}

#[test]
fn control_new_text_stores_payload() {
    let intent = Intent::new(IntentId(10), IntentKind::Speak, "brief summary");
    let frame = ControlFrame::new_text(
        SimTime {
            tick: Tick::new(9),
            window: WindowId::new(0),
        },
        CorrelationId(11),
        ChannelCode::ExternalOutput,
        intent,
        "hello",
    );

    match frame.payload {
        ControlPayload::Text(text) => assert_eq!(&*text, "hello"),
        _ => panic!("expected text payload"),
    }
}

#[test]
fn code_display_strings_are_stable() {
    assert_eq!(DecisionCode::Allow.to_string(), "allow");
    assert_eq!(DecisionCode::Deny.to_string(), "deny");
    assert_eq!(DecisionCode::Defer.to_string(), "defer");

    assert_eq!(
        DenyReasonCode::MissingDecision.to_string(),
        "missing_decision"
    );
    assert_eq!(
        DenyReasonCode::PolicyViolation.to_string(),
        "policy_violation"
    );
    assert_eq!(DenyReasonCode::UnsafeContext.to_string(), "unsafe_context");
    assert_eq!(DenyReasonCode::InvalidIntent.to_string(), "invalid_intent");
    assert_eq!(DenyReasonCode::InternalError.to_string(), "internal_error");

    assert_eq!(ChannelCode::ExternalOutput.to_string(), "external_output");
    assert_eq!(ChannelCode::InternalThought.to_string(), "internal_thought");
    assert_eq!(ChannelCode::MemoryWrite.to_string(), "memory_write");
    assert_eq!(ChannelCode::BrainStimulus.to_string(), "brain_stimulus");
}

#[test]
fn decision_defaults_include_intent_and_reason_code() {
    let frame = DecisionFrame::allow(
        SimTime {
            tick: Tick::new(1),
            window: WindowId::new(0),
        },
        CorrelationId(1),
        "ok",
    );

    assert_eq!(frame.intent, IntentType::Unknown);
    assert_eq!(frame.reason_code, ReasonCode("allow_default"));
}
