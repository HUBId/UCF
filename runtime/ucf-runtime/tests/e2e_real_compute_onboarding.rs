use std::{collections::BTreeMap, fs, path::PathBuf};

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
    std::env::set_var("UCF_COMPUTE_BACKEND", "stub");
    std::env::set_var("UCF_COMPUTE_SEED", "424242");
    std::env::set_var("UCF_LLM_BACKEND", "stub");
    std::env::set_var("UCF_LLM_SEED", "777");
    std::env::set_var("UCF_COMPUTE_BUDGET_PROFILE", budget_profile);
    std::env::remove_var("UCF_ENABLE_EVOLUTION");

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
            format!(
                "sig:{value:03}:scenario:{}:tick:{tick:02}",
                fixture.scenario
            ),
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

    ScenarioRun {
        fixture,
        tick_snapshots: snapshots,
        budget_exceeded_ticks,
        total_records: orchestrator.ess.len(),
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
        stress_avg_pressure > baseline_avg_pressure,
        "stress pressure {} <= baseline {}",
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

    // Golden prefixes at representative ticks.
    let checkpoints = [0usize, 1, 2, 15, 31];
    let expected_baseline = [
        (
            "282940d8f0cfb33e21dc",
            "f641b7b3907d69fd7ca9",
            "d4713e3f78ab46a06003",
        ),
        (
            "5c0a877ba522e9940312",
            "188d3b8e85b58a5155b8",
            "a65398c6e65c5e4cbadf",
        ),
        (
            "1d2a763bb59b770f04cb",
            "32ba3f0a1b911e2c2a45",
            "a48912eb394dc476d338",
        ),
        (
            "6b9ba6acb2e376d6eee6",
            "fd2b983d1c8ccf680464",
            "7dd12211073d47952a78",
        ),
        (
            "25fa32e0a7c5862cff5e",
            "7c5109d3e87d216db74d",
            "3fab73d2433686b070c2",
        ),
    ];
    let expected_stress = [
        (
            "0f212e1660828c65ae8b",
            "f641b7b3907d69fd7ca9",
            "9816b16cca4b845f4e7e",
        ),
        (
            "4da2c653d9168022ce0a",
            "abfbf429c3647db0b9e4",
            "136a8b236134f006adb8",
        ),
        (
            "0948078ee19e6d6ea66c",
            "abfbf429c3647db0b9e4",
            "91bdf02d1d17ee3cb507",
        ),
        (
            "e97e66e3686ba8c453f3",
            "abfbf429c3647db0b9e4",
            "8b7c48030bc051638a49",
        ),
        (
            "9d1751cbfa5bb35180d5",
            "29c9a6f5a04c812b3c55",
            "274fb954dcc80d46b375",
        ),
    ];

    for (ix, cp) in checkpoints.into_iter().enumerate() {
        let b = &baseline.tick_snapshots[cp];
        let s = &stress.tick_snapshots[cp];

        let got_b = (
            digest_prefix_hex(b.compute_chain_digest),
            digest_prefix_hex(b.nsr_assessment_digest),
            digest_prefix_hex(b.output_digest),
        );
        let got_s = (
            digest_prefix_hex(s.compute_chain_digest),
            digest_prefix_hex(s.nsr_assessment_digest),
            digest_prefix_hex(s.output_digest),
        );

        assert_eq!(
            got_b,
            (
                expected_baseline[ix].0.to_string(),
                expected_baseline[ix].1.to_string(),
                expected_baseline[ix].2.to_string(),
            ),
            "baseline golden mismatch at checkpoint {cp}"
        );
        assert_eq!(
            got_s,
            (
                expected_stress[ix].0.to_string(),
                expected_stress[ix].1.to_string(),
                expected_stress[ix].2.to_string(),
            ),
            "stress golden mismatch at checkpoint {cp}"
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
        assert!(pos(ExperienceKind::ControlIn) < pos(ExperienceKind::DecisionOut));
        assert!(pos(ExperienceKind::DecisionOut) < pos(ExperienceKind::CandidateSet));
        assert!(pos(ExperienceKind::CandidateSet) < pos(ExperienceKind::Output));
        assert!(pos(ExperienceKind::Output) < pos(ExperienceKind::Nsr));
    }

    assert!(run.total_records >= 32 * 5);
}
