use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::{Mutex, MutexGuard},
};

use ucf_core::types::{SimTime, Tick, WindowId};
use ucf_ess::v1::{AuditPayload, ExperienceKind, ExperiencePayload, ExperienceStore};
use ucf_frames::v1::{ChannelCode, ControlFrame, CorrelationId, Intent, IntentId, IntentKind};
use ucf_policy::adapter::MockAdapter;
use ucf_policy::candidate::OutputClass;
use ucf_runtime::RuntimeOrchestrator;

#[derive(Debug, Clone, serde::Deserialize)]
struct ScenarioFixture {
    scenario: String,
    ticks: usize,
    channel: String,
    intent_summary: String,
    signal_values: Vec<u32>,
}

#[derive(Debug, Clone)]
struct TickSnapshot {
    tick: u64,
    corr: u64,
    pressure: f32,
    risk: f32,
    confidence: f32,
    surprise: f32,
    compute_chain_digest: [u8; 32],
    nsr_assessment_digest: [u8; 32],
    output_digest: [u8; 32],
    decision_id: u64,
}

static ENV_LOCK: Mutex<()> = Mutex::new(());

fn env_lock_guard() -> MutexGuard<'static, ()> {
    match ENV_LOCK.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[derive(Debug)]
struct ScenarioRun {
    fixture: ScenarioFixture,
    tick_snapshots: Vec<TickSnapshot>,
    budget_exceeded_ticks: usize,
    total_records: usize,
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(name)
}

fn load_fixture(name: &str) -> ScenarioFixture {
    let raw = fs::read_to_string(fixture_path(name)).expect("fixture to exist");
    let fixture: ScenarioFixture = serde_json::from_str(&raw).expect("fixture to parse");
    assert_eq!(fixture.signal_values.len(), fixture.ticks);
    fixture
}

fn channel_from_fixture(value: &str) -> ChannelCode {
    match value {
        "external_output" => ChannelCode::ExternalOutput,
        "internal_thought" => ChannelCode::InternalThought,
        other => panic!("unsupported channel fixture value: {other}"),
    }
}

fn digest_prefix_hex(digest: [u8; 32]) -> String {
    hex::encode(&digest[..10])
}

fn run_scenario(fixture_name: &str, budget_profile: &str) -> ScenarioRun {
    let _env_guard = env_lock_guard();
    std::env::set_var("UCF_COMPUTE_BACKEND", "stub");
    std::env::set_var("UCF_COMPUTE_SEED", "424242");
    std::env::set_var("UCF_LLM_BACKEND", "stub");
    std::env::set_var("UCF_LLM_SEED", "777");
    std::env::set_var("UCF_COMPUTE_BUDGET_PROFILE", budget_profile);
    std::env::remove_var("UCF_ENABLE_EVOLUTION");
    ensure_policy_hash_env();

    let fixture = load_fixture(fixture_name);
    match fixture.scenario.as_str() {
        "baseline" => {
            std::env::set_var("UCF_COMPUTE_MAX_MICROS", "20000");
            std::env::set_var("UCF_COMPUTE_HARD_TIMEOUT_MICROS", "50000");
        }
        "stress" => {
            std::env::set_var("UCF_COMPUTE_MAX_MICROS", "1");
            std::env::set_var("UCF_COMPUTE_HARD_TIMEOUT_MICROS", "2");
        }
        "tool_plan_demo" => {
            std::env::set_var("UCF_COMPUTE_MAX_MICROS", "20000");
            std::env::set_var("UCF_COMPUTE_HARD_TIMEOUT_MICROS", "50000");
            std::env::set_var("UCF_POLICY_OVERLAY", "demo_toolread");
        }
        other => panic!("unsupported scenario: {other}"),
    }
    let mut orchestrator = RuntimeOrchestrator::try_new_from_env().expect("runtime from env");
    orchestrator.set_consolidation_hook_enabled_for_test(true);
    orchestrator.set_geist_hook_enabled_for_test(true);

    let mut adapter = MockAdapter::default();
    let mut snapshots = Vec::with_capacity(fixture.ticks);
    let channel = channel_from_fixture(&fixture.channel);

    for (idx, value) in fixture.signal_values.iter().copied().enumerate() {
        let tick = idx as u64;
        let corr = 50_000 + tick;
        let ctrl = ControlFrame::new_text(
            SimTime {
                tick: Tick::new(tick),
                window: WindowId::new(0),
            },
            CorrelationId(corr),
            channel,
            Intent::new(
                IntentId(90),
                IntentKind::System,
                fixture.intent_summary.as_str(),
            ),
            if fixture.scenario == "tool_plan_demo" {
                format!(
                    "tool_demo_file_read sig:{value:03}:scenario:{}:tick:{tick:02}",
                    fixture.scenario
                )
            } else {
                format!(
                    "sig:{value:03}:scenario:{}:tick:{tick:02}",
                    fixture.scenario
                )
            },
        );

        let decision = orchestrator
            .ingest_and_process(&mut adapter, ctrl)
            .expect("ingest tick");

        let summary = decision.compute_summary.expect("compute summary");
        let decision_id = orchestrator
            .ess
            .trail_by_corr(CorrelationId(corr))
            .into_iter()
            .rev()
            .find_map(|record| {
                if record.kind == ExperienceKind::DecisionOut && record.corr.0 == corr {
                    Some(record.id.0)
                } else {
                    None
                }
            })
            .expect("decision id exists");

        let nsr_digest = orchestrator
            .ess
            .trail_by_corr(CorrelationId(corr))
            .into_iter()
            .rev()
            .find_map(|record| {
                if record.kind == ExperienceKind::Nsr && record.corr.0 == corr {
                    record
                        .nsr_record
                        .as_ref()
                        .map(|entry| entry.assessment_digest)
                } else {
                    None
                }
            })
            .expect("nsr digest exists");

        let output_digest = orchestrator
            .ess
            .trail_by_corr(CorrelationId(corr))
            .into_iter()
            .rev()
            .find_map(|record| {
                if record.kind == ExperienceKind::Output && record.corr.0 == corr {
                    match &record.payload {
                        ExperiencePayload::Audit(AuditPayload::Output(out)) => {
                            Some(out.llm_response_digest)
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            })
            .expect("output digest exists");

        snapshots.push(TickSnapshot {
            tick,
            corr,
            pressure: summary.pressure,
            risk: summary.risk,
            confidence: summary.confidence,
            surprise: summary.surprise,
            compute_chain_digest: summary.compute_chain_digest.expect("compute chain digest"),
            nsr_assessment_digest: nsr_digest,
            output_digest,
            decision_id,
        });
    }

    let budget_exceeded_ticks = snapshots
        .iter()
        .filter(|s| {
            orchestrator
                .ess
                .trail_by_corr(CorrelationId(s.corr))
                .iter()
                .any(|record| {
                    record.kind == ExperienceKind::DecisionOut
                        && record.corr.0 == s.corr
                        && record
                            .compute_summary
                            .and_then(|summary| summary.budget_exceeded_stage)
                            .is_some()
                })
        })
        .count();

    assert!(
        adapter.mem_writes == 0,
        "deny-by-default must avoid tool path"
    );

    if fixture.scenario == "tool_plan_demo" {
        std::env::remove_var("UCF_POLICY_OVERLAY");
    }

    ScenarioRun {
        fixture,
        tick_snapshots: snapshots,
        budget_exceeded_ticks,
        total_records: orchestrator.ess.len(),
    }
}

fn ensure_policy_hash_env() {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let _ = std::env::set_current_dir(&workspace_root);
    if std::env::var("UCF_POLICY_BUNDLE_SHA256").is_ok() {
        return;
    }
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

#[derive(Debug, Clone, Copy)]
enum EbmModeFixture {
    Off,
    Shadow,
    Active,
}

impl EbmModeFixture {
    fn as_env(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Shadow => "shadow",
            Self::Active => "active",
        }
    }
}

#[derive(Debug)]
struct EbmScenarioSummary {
    selected_candidate_ids: Vec<u16>,
    ebm_record_count: usize,
    ebm_digests: Vec<[u8; 8]>,
    ebm_statuses: Vec<u8>,
    tool_auth_denied: bool,
    has_constraint_provenance: bool,
    has_ebm_memory_tag: bool,
    saw_tool_intent_candidate: bool,
}

fn run_ebm_scenario(mode: EbmModeFixture) -> EbmScenarioSummary {
    let _env_guard = env_lock_guard();
    std::env::set_var("UCF_COMPUTE_BACKEND", "stub");
    std::env::set_var("UCF_COMPUTE_SEED", "424242");
    std::env::set_var("UCF_LLM_BACKEND", "stub");
    std::env::set_var("UCF_LLM_SEED", "777");
    std::env::set_var("UCF_COMPUTE_BUDGET_PROFILE", "default");
    std::env::set_var("UCF_SLOT_EBM_MODE", mode.as_env());
    std::env::remove_var("UCF_ENABLE_EVOLUTION");
    std::env::set_var("UCF_COMPUTE_MAX_MICROS", "20000");
    std::env::set_var("UCF_COMPUTE_HARD_TIMEOUT_MICROS", "50000");
    ensure_policy_hash_env();

    let fixture = load_fixture("e2e_scenario_ebm_v1.json");
    assert_eq!(fixture.scenario, "ebm_v1");

    let mut orchestrator = RuntimeOrchestrator::try_new_from_env().expect("runtime from env");
    orchestrator.set_consolidation_hook_enabled_for_test(true);
    orchestrator.set_geist_hook_enabled_for_test(true);

    let mut adapter = MockAdapter::default();
    let channel = channel_from_fixture(&fixture.channel);
    for (idx, value) in fixture.signal_values.iter().copied().enumerate() {
        let tick = idx as u64;
        let corr = 80_000 + tick;
        let ctrl = ControlFrame::new_text(
            SimTime {
                tick: Tick::new(tick),
                window: WindowId::new(0),
            },
            CorrelationId(corr),
            channel,
            Intent::new(
                IntentId(92),
                IntentKind::System,
                fixture.intent_summary.as_str(),
            ),
            format!("ebm:{value:03}:mode:{}:tick:{tick:02}", mode.as_env()),
        );
        orchestrator
            .ingest_and_process(&mut adapter, ctrl)
            .expect("ingest ebm tick");
    }

    let mut selected_candidate_ids = Vec::new();
    let mut ebm_record_count = 0usize;
    let mut ebm_digests = Vec::new();
    let mut ebm_statuses = Vec::new();
    let mut tool_auth_denied = false;
    let mut has_constraint_provenance = false;
    let mut has_ebm_memory_tag = false;
    let mut saw_tool_intent_candidate = false;

    for idx in 0..orchestrator.ess.len() {
        let rec = orchestrator.ess.get(idx).expect("record index");
        match &rec.payload {
            ExperiencePayload::Audit(AuditPayload::CandidateSet(set)) => {
                if rec.kind == ExperienceKind::CandidateSet {
                    let has_safe_text = set
                        .summaries
                        .iter()
                        .any(|s| s.output_class == OutputClass::SafeText as u8);
                    let has_noop = set.summaries.iter().any(|s| s.candidate_id == 3);
                    let has_tool_intent = set.summaries.iter().any(|s| {
                        s.candidate_id == 2 && s.output_class != OutputClass::SafeText as u8
                    });
                    assert!(has_safe_text, "candidate set must include SafeText");
                    assert!(has_noop, "candidate set must include NoOp fallback");
                    saw_tool_intent_candidate = saw_tool_intent_candidate || has_tool_intent;
                    selected_candidate_ids.push(set.selected_candidate_id);
                }
            }
            ExperiencePayload::Audit(AuditPayload::EbmReasoning(ebm)) => {
                ebm_record_count = ebm_record_count.saturating_add(1);
                ebm_digests.push(ebm.ebm_digest_prefix);
                ebm_statuses.push(ebm.status);
                assert!(ebm.top_energies_q.len() <= 8);
                assert!(ebm.top_term_contributions.len() <= 8);
            }
            ExperiencePayload::Audit(AuditPayload::ToolAuth(auth)) => {
                if rec.kind == ExperienceKind::ToolAuth {
                    tool_auth_denied = tool_auth_denied || !auth.allowed;
                }
            }
            ExperiencePayload::Audit(AuditPayload::EbmConstraintProvenance(prov)) => {
                if rec.kind == ExperienceKind::EbmConstraintProvenance {
                    has_constraint_provenance = has_constraint_provenance
                        || !prov.policy_hash_prefix.iter().all(|b| *b == 0);
                }
            }
            _ => {}
        }

        has_ebm_memory_tag = has_ebm_memory_tag
            || rec.ebm_tag.as_ref().is_some_and(|tag| {
                tag.ebm_top_terms.len() <= 4 && tag.ebm_reasoning_digest_prefix != [0; 8]
            });
    }

    EbmScenarioSummary {
        selected_candidate_ids,
        ebm_record_count,
        ebm_digests,
        ebm_statuses,
        tool_auth_denied,
        has_constraint_provenance,
        has_ebm_memory_tag,
        saw_tool_intent_candidate,
    }
}

#[test]
fn e2e_real_compute_onboarding_v0_chain_and_invariants() {
    let baseline = run_scenario("e2e_scenario_a.json", "default");
    let stress = run_scenario("e2e_scenario_b.json", "stress");

    assert_eq!(baseline.tick_snapshots.len(), 32);
    assert_eq!(stress.tick_snapshots.len(), 32);

    for run in [&baseline, &stress] {
        for snap in &run.tick_snapshots {
            assert!((0.0..=1.0).contains(&snap.surprise));
            assert!((0.0..=1.0).contains(&snap.pressure));
            assert!((0.0..=1.0).contains(&snap.risk));
            assert!((0.0..=1.0).contains(&snap.confidence));
            assert_ne!(snap.compute_chain_digest, [0; 32]);
            assert_ne!(snap.nsr_assessment_digest, [0; 32]);
            assert_ne!(snap.output_digest, [0; 32]);
        }
    }

    let avg_pressure = |snaps: &[TickSnapshot]| -> f32 {
        snaps.iter().map(|s| s.pressure).sum::<f32>() / snaps.len() as f32
    };
    let avg_risk = |snaps: &[TickSnapshot]| -> f32 {
        snaps.iter().map(|s| s.risk).sum::<f32>() / snaps.len() as f32
    };

    let baseline_avg_pressure = avg_pressure(&baseline.tick_snapshots);
    let stress_avg_pressure = avg_pressure(&stress.tick_snapshots);
    let baseline_avg_risk = avg_risk(&baseline.tick_snapshots);
    let stress_avg_risk = avg_risk(&stress.tick_snapshots);

    assert!(
        stress_avg_pressure >= baseline_avg_pressure,
        "stress pressure {} < baseline {}",
        stress_avg_pressure,
        baseline_avg_pressure
    );
    assert!(
        stress_avg_risk >= baseline_avg_risk,
        "stress risk {} < baseline {}",
        stress_avg_risk,
        baseline_avg_risk
    );
    assert!(
        stress.budget_exceeded_ticks > 0,
        "stress scenario must trigger at least one degraded stage"
    );

    // Deterministic checkpoints at representative ticks.
    let checkpoints = [0usize, 1, 2, 15, 31];
    let baseline_2 = run_scenario("e2e_scenario_a.json", "default");
    let stress_2 = run_scenario("e2e_scenario_b.json", "stress");

    for cp in checkpoints {
        let b = &baseline.tick_snapshots[cp];
        let b2 = &baseline_2.tick_snapshots[cp];
        let s = &stress.tick_snapshots[cp];
        let s2 = &stress_2.tick_snapshots[cp];

        let got_b = (
            digest_prefix_hex(b.compute_chain_digest),
            digest_prefix_hex(b.nsr_assessment_digest),
            digest_prefix_hex(b.output_digest),
        );
        let got_b2 = (
            digest_prefix_hex(b2.compute_chain_digest),
            digest_prefix_hex(b2.nsr_assessment_digest),
            digest_prefix_hex(b2.output_digest),
        );
        let got_s = (
            digest_prefix_hex(s.compute_chain_digest),
            digest_prefix_hex(s.nsr_assessment_digest),
            digest_prefix_hex(s.output_digest),
        );
        let got_s2 = (
            digest_prefix_hex(s2.compute_chain_digest),
            digest_prefix_hex(s2.nsr_assessment_digest),
            digest_prefix_hex(s2.output_digest),
        );

        assert_eq!(
            got_b, got_b2,
            "baseline deterministic mismatch at checkpoint {cp}"
        );
        assert_eq!(
            got_s, got_s2,
            "stress deterministic mismatch at checkpoint {cp}"
        );
    }

    for windows in [
        baseline.tick_snapshots.as_slice(),
        stress.tick_snapshots.as_slice(),
    ] {
        for pair in windows.windows(2) {
            assert!(pair[0].tick < pair[1].tick);
            assert!(pair[0].decision_id < pair[1].decision_id);
        }
    }
}
#[test]
fn e2e_real_compute_onboarding_v0_ess_linking_and_order() {
    let run = run_scenario("e2e_scenario_a.json", "default");

    // Inspect record chain by corr id and require load-bearing sequence per tick.
    let mut by_corr: BTreeMap<u64, Vec<ExperienceKind>> = BTreeMap::new();
    let mut orchestrator = RuntimeOrchestrator::new();
    let mut adapter = MockAdapter::default();
    let fixture = run.fixture;
    let channel = channel_from_fixture(&fixture.channel);

    for (idx, value) in fixture.signal_values.iter().copied().enumerate() {
        let tick = idx as u64;
        let corr = 70_000 + tick;
        let ctrl = ControlFrame::new_text(
            SimTime {
                tick: Tick::new(tick),
                window: WindowId::new(0),
            },
            CorrelationId(corr),
            channel,
            Intent::new(
                IntentId(91),
                IntentKind::System,
                fixture.intent_summary.as_str(),
            ),
            format!("link-check:{value:03}:{tick:02}"),
        );
        orchestrator
            .ingest_and_process(&mut adapter, ctrl)
            .expect("ingest link tick");
    }

    for idx in 0..orchestrator.ess.len() {
        let rec = orchestrator.ess.get(idx).expect("record index");
        by_corr.entry(rec.corr.0).or_default().push(rec.kind);
        if rec.kind == ExperienceKind::CandidateSet {
            if let ExperiencePayload::Audit(AuditPayload::CandidateSet(set)) = &rec.payload {
                assert!(set.decision_id > 0);
                assert!(set
                    .summaries
                    .iter()
                    .any(|summary| summary.candidate_id == set.selected_candidate_id));
            } else {
                panic!("candidate set payload mismatch");
            }
        }
        if rec.kind == ExperienceKind::Output {
            if let ExperiencePayload::Audit(AuditPayload::Output(output)) = &rec.payload {
                assert!(output.decision_id > 0);
                assert_ne!(output.evidence_chain_digest, [0; 32]);
                assert!(
                    output.output_class == OutputClass::SafeText as u8
                        || output.output_class == OutputClass::Code as u8,
                    "unexpected output class {}",
                    output.output_class
                );
            } else {
                panic!("output payload mismatch");
            }
        }

        if rec.kind == ExperienceKind::ToolAuth {
            if let ExperiencePayload::Audit(AuditPayload::ToolAuth(auth)) = &rec.payload {
                assert!(!auth.allowed, "tool gate must deny-by-default");
            }
        }
        if rec.kind == ExperienceKind::ToolExecution {
            if let ExperiencePayload::Audit(AuditPayload::ToolExecution(exec)) = &rec.payload {
                let status = exec.status.to_ascii_lowercase();
                assert!(
                    status.contains("denied")
                        || status.contains("blocked")
                        || status.contains("error"),
                    "unexpected tool execution status: {}",
                    exec.status
                );
            }
        }
    }

    for kinds in by_corr.values() {
        let pos = |kind: ExperienceKind| {
            kinds
                .iter()
                .position(|entry| *entry == kind)
                .unwrap_or(usize::MAX)
        };
        if pos(ExperienceKind::ControlIn) != usize::MAX
            && pos(ExperienceKind::DecisionOut) != usize::MAX
        {
            assert!(pos(ExperienceKind::ControlIn) < pos(ExperienceKind::DecisionOut));
        }
        if pos(ExperienceKind::DecisionOut) != usize::MAX
            && pos(ExperienceKind::CandidateSet) != usize::MAX
        {
            assert!(pos(ExperienceKind::DecisionOut) < pos(ExperienceKind::CandidateSet));
        }
        if pos(ExperienceKind::CandidateSet) != usize::MAX
            && pos(ExperienceKind::Output) != usize::MAX
        {
            assert!(pos(ExperienceKind::CandidateSet) < pos(ExperienceKind::Output));
        }
        if pos(ExperienceKind::Output) != usize::MAX && pos(ExperienceKind::Nsr) != usize::MAX {
            assert!(pos(ExperienceKind::Output) < pos(ExperienceKind::Nsr));
        }
    }

    assert!(run.total_records >= 32 * 5);
}

#[test]
fn e2e_scenario_ebm_v1_off_shadow_active() {
    let off = run_ebm_scenario(EbmModeFixture::Off);
    let shadow = run_ebm_scenario(EbmModeFixture::Shadow);
    let active = run_ebm_scenario(EbmModeFixture::Active);

    assert_eq!(
        off.ebm_record_count, 0,
        "off mode must not emit ebm records"
    );
    assert!(
        shadow.ebm_record_count > 0,
        "shadow mode must emit ebm records"
    );
    assert!(
        active.ebm_record_count > 0,
        "active mode must emit ebm records"
    );

    assert_eq!(
        off.selected_candidate_ids, shadow.selected_candidate_ids,
        "shadow mode must not alter candidate selection"
    );
    assert!(
        active.selected_candidate_ids.iter().all(|id| *id != 2),
        "active mode must rerank away from ToolIntent candidate"
    );

    let active_second = run_ebm_scenario(EbmModeFixture::Active);
    assert_eq!(
        active.ebm_digests, active_second.ebm_digests,
        "active ebm digests must be deterministic"
    );
    assert_eq!(active.ebm_statuses, active_second.ebm_statuses);

    if active.saw_tool_intent_candidate {
        assert!(active.tool_auth_denied, "tool auth must remain denied");
    }
    assert!(
        shadow.has_constraint_provenance && active.has_constraint_provenance,
        "constraints provenance record must be present"
    );
    let _ebm_memory_tag_observed = shadow.has_ebm_memory_tag || active.has_ebm_memory_tag;
    let _tool_intent_observed = off.saw_tool_intent_candidate
        || shadow.saw_tool_intent_candidate
        || active.saw_tool_intent_candidate;
}

#[test]
fn e2e_tool_plan_demo_chain_and_single_use_token_replay() {
    let _env_guard = env_lock_guard();
    let mut orchestrator = RuntimeOrchestrator::new();
    orchestrator.force_nsr_risk_for_test(0.05);
    orchestrator.force_surprise_for_test(0.05);
    orchestrator.force_ess_pressure_for_test(0.05);
    let mut adapter = MockAdapter::default();
    let ctrl = ControlFrame::new_text(
        SimTime {
            tick: Tick::new(10),
            window: WindowId::new(0),
        },
        CorrelationId(88_810),
        ChannelCode::ExternalOutput,
        Intent::new(IntentId(99), IntentKind::System, "tool-demo"),
        "tool_demo_file_read deterministic".to_string(),
    );

    orchestrator
        .ingest_and_process(&mut adapter, ctrl)
        .expect("demo ingest");

    let records: Vec<_> = (0..orchestrator.ess.len())
        .filter_map(|idx| orchestrator.ess.get(idx))
        .collect();

    assert!(records
        .iter()
        .any(|r| r.kind == ExperienceKind::CandidateSet));
    assert!(
        records.iter().any(|r| r.kind == ExperienceKind::ToolPlan),
        "kinds={:?}",
        records
            .iter()
            .map(|r| format!("{:?}", r.kind))
            .collect::<Vec<_>>()
    );
    assert!(records.iter().any(|r| r.kind == ExperienceKind::ToolIssue));

    let executions: Vec<_> = records
        .iter()
        .filter(|r| r.kind == ExperienceKind::ToolExecution)
        .collect();
    assert!(
        executions.iter().any(|r| {
            if let ExperiencePayload::Audit(AuditPayload::ToolExecution(exec)) = &r.payload {
                exec.status.contains("AllowedExecuted")
            } else {
                false
            }
        }),
        "expected one successful plugin execution, got {:?}",
        executions
            .iter()
            .filter_map(|r| match &r.payload {
                ExperiencePayload::Audit(AuditPayload::ToolExecution(exec)) => {
                    Some((exec.status.clone(), exec.error_code.clone()))
                }
                _ => None,
            })
            .collect::<Vec<_>>()
    );
    assert!(
        executions.iter().any(|r| {
            if let ExperiencePayload::Audit(AuditPayload::ToolExecution(exec)) = &r.payload {
                exec.error_code.as_deref() == Some("token_replay")
            } else {
                false
            }
        }),
        "expected replay denial for single-use token"
    );

    let exec_ok = executions
        .iter()
        .find_map(|r| match &r.payload {
            ExperiencePayload::Audit(AuditPayload::ToolExecution(exec))
                if exec.status.contains("AllowedExecuted") =>
            {
                Some(exec)
            }
            _ => None,
        })
        .expect("ok execution record");
    let note = exec_ok.error_code.as_deref().unwrap_or("");
    assert!(note.contains("result_digest="));
    assert!(note.contains("preview="));
    assert!(note.len() <= 120);
    assert!(!note.contains('\n'));
}
