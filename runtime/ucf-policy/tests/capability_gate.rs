use ucf_core::types::{SimTime, Tick, WindowId};
use ucf_frames::v1::{
    ChannelCode, ControlFrame, CorrelationId, DecisionCode, DecisionFrame, Intent, IntentId,
    IntentKind,
};
use ucf_policy::{
    adapter::MockAdapter,
    capability::{CapabilityDenyReason, CapabilityKind, CapabilityScope},
    gem::{
        governor_score, issue_capabilities, issue_capabilities_governed, Gem, GovernanceSignals,
        IssuanceTier, ToolGate, ToolGovernor, ToolStatus,
    },
    rate_limiter::RateLimiter,
};

fn sim_time(tick: u64) -> SimTime {
    SimTime {
        tick: Tick::new(tick),
        window: WindowId::new(0),
    }
}

fn intent() -> Intent {
    Intent::new(IntentId(7), IntentKind::System, "test")
}

#[test]
fn scope_matching_domain_like_target() {
    let set = ucf_policy::capability::CapabilitySet {
        tokens: vec![ucf_policy::capability::CapabilityToken::issue(
            CapabilityKind::ExternalApi,
            CapabilityScope::ApiNames(vec!["external_output".to_string()]),
            ucf_policy::capability::CapabilityLimits {
                max_calls_per_window: 2,
                window_ticks: 5,
                max_bytes_out: Some(10),
                max_bytes_in: None,
                max_concurrent: 1,
            },
            "test",
            1,
            Some(4),
        )],
    };
    let req = ucf_policy::gem::ToolRequest {
        id: 1,
        kind: CapabilityKind::ExternalApi,
        target: "external_output".to_string(),
        payload_hint: ucf_policy::gem::PayloadHint {
            bytes_out: Some(8),
            bytes_in: None,
        },
        requested_at_t: 1,
        decision_id: 5,
        evidence_chain_digest: [0; 32],
        candidate_id: None,
        tool_intent_digest: None,
    };
    assert!(set.allows(&req, 2).is_ok());
    assert_eq!(set.allows(&req, 5), Err(CapabilityDenyReason::Expired));
}

#[test]
fn gate_denies_missing_decision_and_rate_limits_by_ticks() {
    let ctrl = ControlFrame::new_text(
        sim_time(1),
        CorrelationId(99),
        ChannelCode::ExternalOutput,
        intent(),
        "hello",
    );
    let mut decision = DecisionFrame::allow(sim_time(1), CorrelationId(99), "allow");
    decision.compute_summary = Some(ucf_frames::v1::ComputeSignalsSummary {
        backend: "stub",
        surprise: 0.1,
        pressure: 0.1,
        risk: 0.1,
        confidence: 0.9,
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
        lfm_quality: None,
        coherence: None,
        instability: None,
        phi_proxy: None,
        coherence_digest: None,
    });
    let mut gate = ToolGate::new(issue_capabilities(Some(&decision), 1), RateLimiter::new(2));
    let mut adapter = MockAdapter::default();

    let denied =
        Gem::execute_with_gate(&mut adapter, &ctrl, Some(&decision), 0, &mut gate).expect("audit");
    assert_eq!(denied.result.status, ToolStatus::Denied);

    let mut decision_ok = decision.clone();
    decision_ok.decision = DecisionCode::Allow;
    let first = Gem::execute_with_gate(&mut adapter, &ctrl, Some(&decision_ok), 7, &mut gate)
        .expect("first");
    let second = Gem::execute_with_gate(&mut adapter, &ctrl, Some(&decision_ok), 7, &mut gate)
        .expect("second");
    let third = Gem::execute_with_gate(&mut adapter, &ctrl, Some(&decision_ok), 7, &mut gate)
        .expect("third");
    let fourth = Gem::execute_with_gate(&mut adapter, &ctrl, Some(&decision_ok), 7, &mut gate)
        .expect("fourth");
    assert_eq!(first.result.status, ToolStatus::AllowedExecuted);
    assert_eq!(second.result.status, ToolStatus::AllowedExecuted);
    assert_eq!(third.result.status, ToolStatus::RateLimited);
    assert_eq!(fourth.result.status, ToolStatus::RateLimited);
}

#[test]
fn issuer_blocks_high_risk() {
    let mut decision = DecisionFrame::allow(sim_time(10), CorrelationId(3), "allow");
    decision.compute_summary = Some(ucf_frames::v1::ComputeSignalsSummary {
        backend: "stub",
        surprise: 0.2,
        pressure: 0.2,
        risk: 0.9,
        confidence: 0.2,
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
        lfm_quality: None,
        coherence: None,
        instability: None,
        phi_proxy: None,
        coherence_digest: None,
    });
    let caps = issue_capabilities(Some(&decision), 10);
    assert!(caps
        .tokens
        .iter()
        .all(|t| t.kind != CapabilityKind::ExternalApi));
}

#[test]
fn governor_score_and_tier_boundaries_are_deterministic() {
    let signals = GovernanceSignals::from_inputs(None, 42, Some(0.8), Some(0.4));
    let score1 = governor_score(signals);
    let score2 = governor_score(signals);
    assert_eq!(score1, score2);
    assert!(matches!(
        IssuanceTier::from_score(0.249),
        IssuanceTier::Tier0
    ));
    assert!(matches!(
        IssuanceTier::from_score(0.25),
        IssuanceTier::Tier1
    ));
    assert!(matches!(IssuanceTier::from_score(0.5), IssuanceTier::Tier2));
    assert!(matches!(
        IssuanceTier::from_score(0.75),
        IssuanceTier::Tier3
    ));
}

#[test]
fn tool_governor_escalates_cooldown_under_repeated_denies() {
    let mut governor = ToolGovernor::default();
    let mut decision = DecisionFrame::allow(sim_time(1), CorrelationId(8), "allow");
    decision.compute_summary = Some(ucf_frames::v1::ComputeSignalsSummary {
        backend: "stub",
        surprise: 0.9,
        pressure: 0.9,
        risk: 0.9,
        confidence: 0.1,
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
        lfm_uncertainty: Some(1.0),
        lfm_stability: Some(0.0),
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
        lfm_quality: None,
        coherence: Some(0.0),
        instability: Some(1.0),
        phi_proxy: None,
        coherence_digest: None,
    });
    let signals = GovernanceSignals::from_inputs(Some(&decision), 1, Some(0.95), Some(0.9));
    let (_, d1) = issue_capabilities_governed(Some(&decision), 1, signals, &mut governor);
    let (_, d2) = issue_capabilities_governed(Some(&decision), 2, signals, &mut governor);
    assert!(d1.tier.as_u8() >= 2);
    assert!(d2.tier.as_u8() >= 2);
    let slot = governor.slot(&CapabilityKind::ExternalApi);
    assert!(slot.cooldown_ticks > 0);
    assert!(slot.deny_count > 0);
}
