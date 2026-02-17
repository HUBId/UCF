use ucf_core::types::{SimTime, Tick, WindowId};
use ucf_frames::v1::{
    BrainStimulusKind, BrainStimulusPayload, ChannelCode, ControlFrame, ControlPayload,
    CorrelationId, DecisionCode, DecisionFrame, Intent, IntentId, IntentKind, IntentType,
    NeuromodulatorSnapshot,
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

fn with_low_risk(mut decision: DecisionFrame) -> DecisionFrame {
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
    decision
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
fn pbm_allows_external_output_text_and_gem_emits() {
    let ctrl = ControlFrame::new_text(
        sim_time(),
        CorrelationId(2),
        ChannelCode::ExternalOutput,
        intent(),
        "allow-me",
    );
    let decision = with_low_risk(Pbm::decide(&ctrl, None));
    let mut adapter = MockAdapter::default();

    let result = Gem::execute(&mut adapter, &ctrl, Some(&decision));

    assert_eq!(decision.decision, DecisionCode::Allow);
    assert_eq!(decision.intent, IntentType::ExternalCommunicate);
    assert_eq!(decision.reason_code.0, "allow_text_external");
    assert!(result.is_ok());
    assert_eq!(adapter.emitted, vec!["allow-me".to_string()]);
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
    let decision = Pbm::decide(&ctrl, None);
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
    let decision = with_low_risk(DecisionFrame::allow(
        sim_time(),
        CorrelationId(4),
        "allow_test",
    ));
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
    let decision = with_low_risk(DecisionFrame::allow(
        sim_time(),
        CorrelationId(5),
        "allow_mem",
    ));
    let mut adapter = MockAdapter::default();

    let result = Gem::execute(&mut adapter, &ctrl, Some(&decision));

    assert!(result.is_ok());
    assert_eq!(adapter.mem_writes, 1);
    assert!(adapter.emitted.is_empty());
}

#[test]
fn gem_allow_brain_stimulus_emits_spikes() {
    let ctrl = ControlFrame {
        time: sim_time(),
        corr: CorrelationId(6),
        channel: ChannelCode::BrainStimulus,
        intent: intent(),
        payload: ControlPayload::BrainStimulus(BrainStimulusPayload {
            kind: BrainStimulusKind::SpikeTrain,
            target: 23,
            intensity: 77,
            duration_ms: 25,
        }),
    };
    let decision = with_low_risk(DecisionFrame::allow(
        sim_time(),
        CorrelationId(6),
        "allow_brain",
    ));
    let mut adapter = MockAdapter::default();

    let result = Gem::execute(&mut adapter, &ctrl, Some(&decision));

    assert!(result.is_ok());
    assert_eq!(adapter.brain_spikes().len(), 2);
    assert_eq!(adapter.take_brain_spike_meta(), Some((2, 23)));
}

#[test]
fn pbm_denies_external_output_text_over_512_bytes() {
    let oversized = "x".repeat(513);
    let ctrl = ControlFrame::new_text(
        sim_time(),
        CorrelationId(22),
        ChannelCode::ExternalOutput,
        intent(),
        oversized,
    );

    let decision = Pbm::decide(&ctrl, None);

    assert_eq!(decision.decision, DecisionCode::Deny);
    assert_eq!(decision.intent, IntentType::ExternalCommunicate);
    assert_eq!(decision.reason_code.0, "deny_external_too_long");
}

#[test]
fn pbm_meta_baseline_when_neuromod_none() {
    let ctrl = ControlFrame::new_text(
        sim_time(),
        CorrelationId(30),
        ChannelCode::ExternalOutput,
        intent(),
        "meta-baseline",
    );

    let decision = Pbm::decide(&ctrl, None);

    assert_eq!(decision.meta.attention_gain, 0.5);
    assert_eq!(decision.meta.learning_gate, 0.5);
    assert_eq!(decision.meta.recursion_budget, 1);
}

#[test]
fn pbm_meta_high_stress_sets_recursion_budget_zero() {
    let ctrl = ControlFrame::new_text(
        sim_time(),
        CorrelationId(31),
        ChannelCode::ExternalOutput,
        intent(),
        "meta-stress",
    );
    let mut neuromod = NeuromodulatorSnapshot::baseline();
    neuromod.stress = 0.95;

    let decision = Pbm::decide(&ctrl, Some(neuromod));

    assert_eq!(decision.meta.recursion_budget, 0);
}
