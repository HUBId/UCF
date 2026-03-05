use std::{
    fs,
    path::PathBuf,
    sync::{Mutex, MutexGuard},
};

use sha2::{Digest, Sha256};
use ucf_core::types::{SimTime, Tick, WindowId};
use ucf_ess::v1::{AuditPayload, ExperienceKind, ExperiencePayload, ExperienceStore};
use ucf_frames::v1::{ChannelCode, ControlFrame, CorrelationId, Intent, IntentId, IntentKind};
use ucf_policy::{adapter::MockAdapter, candidate::OutputClass};
use ucf_replay::{replay_audit, ReplayOverallStatus, ReplayPlan, ReplayStrictness};
use ucf_runtime::RuntimeOrchestrator;

#[derive(Debug, Clone, serde::Deserialize)]
struct V0Fixture {
    scenario: String,
    ticks: usize,
    channel: String,
    intent_summary: String,
    policy_overlay: String,
    determinism: String,
    stub_backends: bool,
    signal_values: Vec<u32>,
}

#[derive(Debug)]
struct FlowRunSummary {
    records: Vec<ucf_ess::v1::ExperienceRecord>,
    selected_candidate_ids: Vec<u16>,
    last_signal_digest_prefix: [u8; 8],
    last_decision_digest: [u8; 32],
    last_experience_digest: [u8; 32],
    consolidation_milestones: u64,
    geist_total: u64,
}

static ENV_LOCK: Mutex<()> = Mutex::new(());

fn env_lock_guard() -> MutexGuard<'static, ()> {
    match ENV_LOCK.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/e2e/v0_flow_a.json")
}

fn load_fixture() -> V0Fixture {
    let raw = fs::read_to_string(fixture_path()).expect("fixture readable");
    let fixture: V0Fixture = serde_json::from_str(&raw).expect("fixture parses");
    assert_eq!(fixture.signal_values.len(), fixture.ticks);
    assert_eq!(fixture.scenario, "v0_flow_a");
    assert_eq!(fixture.determinism, "strict");
    assert!(fixture.stub_backends);
    fixture
}

fn ensure_policy_hash_env() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../policies/manifest.toml");
    let raw = fs::read_to_string(manifest).expect("policy manifest readable");
    let hash = raw
        .lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix("bundle_sha256 = ")
                .and_then(|rest| rest.strip_prefix('"'))
                .and_then(|rest| rest.strip_suffix('"'))
        })
        .expect("bundle_sha256 present");
    std::env::set_var("UCF_POLICY_BUNDLE_SHA256", hash);
}

fn channel_from_fixture(value: &str) -> ChannelCode {
    match value {
        "external_output" => ChannelCode::ExternalOutput,
        "internal_thought" => ChannelCode::InternalThought,
        other => panic!("unsupported fixture channel {other}"),
    }
}

fn decision_digest(record: &ucf_ess::v1::ExperienceRecord) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(format!("{:?}", record.payload));
    hasher.update(format!("{:?}", record.compute_summary));
    hasher.finalize().into()
}

fn run_v0_flow(enable_hooks: bool) -> FlowRunSummary {
    let _env_guard = env_lock_guard();
    std::env::set_var("UCF_COMPUTE_BACKEND", "stub");
    std::env::set_var("UCF_COMPUTE_SEED", "424242");
    std::env::set_var("UCF_LLM_BACKEND", "stub");
    std::env::set_var("UCF_LLM_SEED", "777");
    std::env::set_var("UCF_COMPUTE_BUDGET_PROFILE", "default");
    std::env::set_var("UCF_COMPUTE_MAX_MICROS", "20000");
    std::env::set_var("UCF_COMPUTE_HARD_TIMEOUT_MICROS", "50000");
    std::env::remove_var("UCF_ENABLE_EVOLUTION");
    ensure_policy_hash_env();

    let fixture = load_fixture();
    std::env::set_var("UCF_POLICY_OVERLAY", &fixture.policy_overlay);

    let mut orchestrator = RuntimeOrchestrator::try_new_from_env().expect("runtime from env");
    orchestrator.set_consolidation_hook_enabled_for_test(enable_hooks);
    orchestrator.set_geist_hook_enabled_for_test(enable_hooks);

    let mut adapter = MockAdapter::default();
    let mut selected_candidate_ids = Vec::with_capacity(fixture.ticks);
    let channel = channel_from_fixture(&fixture.channel);
    let mut signal_bundle_count = 0usize;
    let mut decision_inputs_count = 0usize;

    for (idx, value) in fixture.signal_values.iter().copied().enumerate() {
        let tick = idx as u64;
        let corr = CorrelationId(140_000 + tick);
        let ctrl = ControlFrame::new_text(
            SimTime {
                tick: Tick::new(tick),
                window: WindowId::new(0),
            },
            corr,
            channel,
            Intent::new(
                IntentId(175),
                IntentKind::System,
                fixture.intent_summary.as_str(),
            ),
            format!("v0-flow:{value:03}:tick:{tick:02}"),
        );

        let decision = orchestrator
            .ingest_and_process(&mut adapter, ctrl)
            .expect("ingest tick");
        let summary = decision.compute_summary.expect("compute summary");
        assert!(summary.signal_bundle_digest.is_some());

        let trail = orchestrator.ess.trail_by_corr(corr);
        let mut saw_signal_bundle = false;
        let mut saw_decision_inputs = false;
        let mut saw_decision_frame = false;
        let mut saw_experience = false;

        let mut prev_audit_digest = [0u8; 32];
        let mut first_audit = true;

        for record in &trail {
            assert!(format!("{:?}", record).len() <= 16_384);
            match record.kind {
                ExperienceKind::SignalBundle => {
                    saw_signal_bundle = true;
                    signal_bundle_count = signal_bundle_count.saturating_add(1);
                    let signal_bundle = record.signal_bundle_record.expect("signal bundle record");
                    assert_ne!(signal_bundle.signals_digest_prefix, [0; 8]);
                }
                ExperienceKind::DecisionInputs => {
                    saw_decision_inputs = true;
                    decision_inputs_count = decision_inputs_count.saturating_add(1);
                    let inputs = record.decision_inputs_record.expect("decision inputs");
                    assert_ne!(inputs.signals_digest_prefix, [0; 8]);
                    assert!(inputs.top_candidates_digest_prefix.is_some());
                }
                ExperienceKind::DecisionOut => {
                    saw_decision_frame = true;
                    assert!(record.compute_summary.is_some());
                }
                ExperienceKind::Output => {
                    saw_experience = true;
                }
                ExperienceKind::CandidateSet => {
                    if let ExperiencePayload::Audit(AuditPayload::CandidateSet(set)) =
                        &record.payload
                    {
                        let has_safe_text = set
                            .summaries
                            .iter()
                            .any(|s| s.output_class == OutputClass::SafeText as u8);
                        let has_noop = set.summaries.iter().any(|s| s.candidate_id == 3);
                        assert!(has_safe_text);
                        assert!(has_noop);
                        assert_ne!(set.selected_candidate_digest, [0; 32]);
                        selected_candidate_ids.push(set.selected_candidate_id);
                        for summary in &set.summaries {
                            assert_ne!(summary.digest, [0; 32]);
                        }
                    }
                }
                _ => {}
            }

            if let Some(audit_digest) = record.audit_digest {
                if first_audit {
                    first_audit = false;
                } else {
                    assert_eq!(record.audit_prev_digest, Some(prev_audit_digest));
                }
                prev_audit_digest = audit_digest;
            }
        }

        let _ = saw_signal_bundle;
        let _ = saw_decision_inputs;
        assert!(saw_decision_frame);
        assert!(saw_experience);
    }

    assert!(signal_bundle_count >= 1);
    assert!(decision_inputs_count >= 1);

    assert_eq!(adapter.mem_writes, 0, "tool execution must stay denied");

    let records: Vec<_> = (0..orchestrator.ess.len())
        .filter_map(|idx| orchestrator.ess.get(idx).cloned())
        .collect();

    let last_corr = CorrelationId(140_000 + (fixture.ticks as u64 - 1));
    let last_trail = orchestrator.ess.trail_by_corr(last_corr);

    let last_signal_digest_prefix = last_trail
        .iter()
        .find_map(|record| {
            if record.kind == ExperienceKind::SignalBundle {
                record
                    .signal_bundle_record
                    .map(|signal_bundle| signal_bundle.signals_digest_prefix)
            } else {
                record
                    .compute_summary
                    .and_then(|summary| summary.signal_bundle_digest)
                    .map(|digest| {
                        let mut prefix = [0u8; 8];
                        prefix.copy_from_slice(&digest[..8]);
                        prefix
                    })
            }
        })
        .expect("last signal digest exists");

    let last_decision_digest = last_trail
        .iter()
        .find(|record| record.kind == ExperienceKind::DecisionOut)
        .map(|record| decision_digest(record))
        .expect("decision frame record");

    let last_experience_digest = last_trail
        .iter()
        .rev()
        .find_map(|record| record.audit_digest)
        .expect("experience audit digest exists");

    let replay_report = replay_audit(
        &records,
        &ReplayPlan {
            t0: 0,
            t1: fixture.ticks as u64 - 1,
            expected_backend_pack_digest: None,
            strictness: ReplayStrictness::VerifyOnly,
            stop_on_first_divergence: true,
        },
    );
    assert_ne!(
        replay_report.overall_status,
        ReplayOverallStatus::MissingData
    );

    FlowRunSummary {
        records,
        selected_candidate_ids,
        last_signal_digest_prefix,
        last_decision_digest,
        last_experience_digest,
        consolidation_milestones: orchestrator.consolidation_milestones_emitted_total(),
        geist_total: orchestrator.geist_updates_accepted_total()
            + orchestrator.geist_updates_rejected_total(),
    }
}

#[test]
fn v0_flow_e2e_is_deterministic_and_chain_complete() {
    let first = run_v0_flow(true);
    let second = run_v0_flow(true);

    assert_eq!(
        first.last_signal_digest_prefix,
        second.last_signal_digest_prefix
    );
    assert_eq!(first.last_decision_digest, second.last_decision_digest);
    assert_eq!(first.last_experience_digest, second.last_experience_digest);

    assert_eq!(first.selected_candidate_ids, second.selected_candidate_ids);
    assert_eq!(first.records.len(), second.records.len());
    assert!(first.records.len() >= 8 * 8);
}

#[test]
fn v0_flow_optional_hooks_do_not_change_core_outputs() {
    let without_hooks = run_v0_flow(false);
    let with_hooks = run_v0_flow(true);

    assert_eq!(
        without_hooks.selected_candidate_ids,
        with_hooks.selected_candidate_ids
    );
    assert_eq!(
        without_hooks.last_signal_digest_prefix,
        with_hooks.last_signal_digest_prefix
    );
    assert_eq!(
        without_hooks.last_decision_digest,
        with_hooks.last_decision_digest
    );

    assert!(with_hooks.consolidation_milestones <= with_hooks.records.len() as u64);
    assert!(with_hooks.geist_total <= with_hooks.records.len() as u64);
}
