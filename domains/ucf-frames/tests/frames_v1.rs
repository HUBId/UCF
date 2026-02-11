use ucf_core::types::{SimTime, Tick, WindowId};
use ucf_frames::v1::{
    BiophysFrame, BiophysHhParams, CdeFrame, ChannelCode, ControlFrame, ControlPayload,
    CorrelationId, DecisionCode, DecisionFrame, DenyReasonCode, IitFrame, Intent, IntentId,
    IntentKind, IntentType, NsrFrame, OnnFrame, ReasonCode, SnnFrame,
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

#[test]
fn biophys_frame_can_be_constructed() {
    let frame = BiophysFrame {
        now_ms: 42,
        field: [0.1; 7],
        hh_params: BiophysHhParams {
            g_na: 120.0,
            g_k: 36.0,
            g_l: 0.3,
            threshold_shift_mv: 0.0,
            max_firing_hz: 200.0,
        },
        hpa_cortisol: 0.1,
    };

    assert_eq!(frame.now_ms, 42);
    assert_eq!(frame.field.len(), 7);
}

#[test]
fn iit_frame_can_be_constructed() {
    let frame = IitFrame {
        now_ms: 123,
        integration: 0.42,
        state: 1,
    };

    assert_eq!(frame.now_ms, 123);
    assert_eq!(frame.state, 1);
}

#[test]
fn onn_and_snn_frames_can_be_constructed() {
    let onn = OnnFrame {
        now_ms: 7,
        global_phase: 1.2,
        mean_lock: 0.8,
    };
    let snn = SnnFrame {
        now_ms: 7,
        spikes: 3,
        feature: 1,
        causal: 1,
        verify: 1,
        attention: 0,
    };

    assert_eq!(onn.now_ms, 7);
    assert_eq!(snn.spikes, 3);
}

#[test]
fn cde_frame_can_be_constructed() {
    let cde = CdeFrame {
        now_ms: 77,
        hyps: 3,
        changed: 1,
        pruned: 0,
        top_conf_q: 200,
    };

    assert_eq!(cde.now_ms, 77);
    assert_eq!(cde.hyps, 3);
    assert_eq!(cde.top_conf_q, 200);
}

#[test]
fn nsr_frame_can_be_constructed() {
    let nsr = NsrFrame {
        now_ms: 99,
        verdict: 1,
        satisfied: 2,
        total: 5,
        verified_q: 102,
    };

    assert_eq!(nsr.now_ms, 99);
    assert_eq!(nsr.verdict, 1);
    assert!(nsr.satisfied <= nsr.total);
}
