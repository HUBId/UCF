use engine::RegulationEngine;
use ucf::v1::{
    ExecStats, IntegrityStateClass, PolicyStats, ReasonCode, ReceiptStats, SignalFrame, WindowKind,
};

fn base_medium_frame() -> SignalFrame {
    SignalFrame {
        window_kind: WindowKind::Medium as i32,
        window_index: Some(1),
        timestamp_ms: None,
        policy_stats: Some(PolicyStats {
            deny_count: 0,
            allow_count: 1,
            top_reason_codes: vec![],
        }),
        exec_stats: Some(ExecStats {
            timeout_count: 0,
            partial_failure_count: 0,
            tool_unavailable_count: 0,
            tool_id: None,
            dlp_block_count: 0,
            top_reason_codes: vec![],
        }),
        integrity_state: IntegrityStateClass::Ok as i32,
        top_reason_codes: vec![],
        signal_frame_digest: None,
        receipt_stats: Some(ReceiptStats {
            receipt_missing_count: 0,
            receipt_invalid_count: 0,
        }),
        reason_codes: vec![],
    }
}

fn drive_frame(engine: &mut RegulationEngine, frame: SignalFrame, now_ms: u64) -> ucf::v1::ControlFrame {
    engine.enqueue_signal_frame(frame).expect("frame enqueued");
    engine.tick(now_ms)
}

#[test]
fn reward_block_caps_progress_effects() {
    let mut engine = RegulationEngine::default();
    let mut frame = base_medium_frame();
    frame.policy_stats.as_mut().unwrap().allow_count = 10;
    frame
        .reason_codes
        .push(ReasonCode::RcReReplayMismatch as i32);

    let control = drive_frame(&mut engine, frame, 100);
    let _ = engine.tick(150);

    assert!(control
        .profile_reason_codes
        .contains(&(ReasonCode::RcGvProgressRewardBlocked as i32)));
}

#[test]
fn replay_hint_after_three_negative_windows() {
    let mut engine = RegulationEngine::default();

    for idx in 0u64..3 {
        let mut frame = base_medium_frame();
        frame.window_index = Some(idx + 1);
        if let Some(policy_stats) = frame.policy_stats.as_mut() {
            policy_stats.allow_count = 0;
            policy_stats.deny_count = 5;
        }
        if let Some(exec_stats) = frame.exec_stats.as_mut() {
            exec_stats.timeout_count = 1;
        }

        let control = drive_frame(&mut engine, frame, 200 + idx);

        if idx == 2 {
            let _ = engine.tick(400);
            assert!(control
                .profile_reason_codes
                .contains(&(ReasonCode::RcGvReplayDiminishingReturns as i32)));
        }
    }
}
