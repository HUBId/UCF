use ucf_sdk::{
    ControlFrameV1, DecisionEventV1, DecisionKindV1, Digest32, EssSummaryQueryV1,
    EssSummaryResponseV1, UQ0_16,
};

#[test]
fn deterministic_encoding_is_stable() {
    let control = ControlFrameV1::new("ctrl-7".to_string(), Digest32::new([1u8; 32]), 42, 99, 5);

    let decision = DecisionEventV1::new(
        "ctrl-7".to_string(),
        10,
        DecisionKindV1::Allow,
        77,
        UQ0_16::from_raw(1234),
        Digest32::new([2u8; 32]),
    );

    let query = EssSummaryQueryV1::new(10, 100, Some(DecisionKindV1::Allow));

    let response = EssSummaryResponseV1::new(
        10,
        100,
        91,
        70,
        20,
        1,
        UQ0_16::from_raw(4567),
        Digest32::new([3u8; 32]),
    );

    assert_eq!(
        control.encode_deterministic(),
        control.encode_deterministic()
    );
    assert_eq!(
        decision.encode_deterministic(),
        decision.encode_deterministic()
    );
    assert_eq!(query.encode_deterministic(), query.encode_deterministic());
    assert_eq!(
        response.encode_deterministic(),
        response.encode_deterministic()
    );
}

#[cfg(feature = "serde")]
#[test]
fn serde_round_trip() {
    let event = DecisionEventV1::new(
        "ctrl-9".to_string(),
        3,
        DecisionKindV1::Deny,
        2,
        UQ0_16::from_raw(1),
        Digest32::new([9u8; 32]),
    );

    let json = serde_json::to_string(&event).expect("serialize");
    let parsed: DecisionEventV1 = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed, event);
}
