use biophys_core::ModulatorField;
use dbm_12_insula::InsulaInput;
use dbm_18_cerebellum::{CerInput, ToolFailureCounts};
use dbm_7_lc::LcInput;
use dbm_8_serotonin::SerInput;
use dbm_9_amygdala::AmyInput;
use dbm_bus::{BrainBus, BrainInput};
use dbm_core::{CooldownClass, IntegrityState, LevelClass, ProfileState, ToolKey};
use dbm_hpa::{Hpa, HpaInput, HpaOutput};
use dbm_pag::PagInput;
use dbm_pmrf::PmrfInput;
use dbm_stn::StnInput;
use std::sync::atomic::{AtomicUsize, Ordering};
use ucf::v1::WindowKind;

static HPA_STATE_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn make_bus() -> BrainBus {
    let id = HPA_STATE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "hpa_state_cerebellum_boundary_{}_{}.json",
        std::process::id(),
        id
    ));
    let _ = std::fs::remove_file(&path);
    BrainBus::with_hpa(Hpa::new(path), HpaOutput::default())
}

fn base_input(cerebellum: Option<CerInput>) -> BrainInput {
    BrainInput {
        now_ms: 10_000,
        window_kind: WindowKind::Medium,
        hpa: HpaInput::default(),
        cbv: None,
        pev: None,
        lc: LcInput::default(),
        serotonin: SerInput::default(),
        amygdala: AmyInput::default(),
        pag: PagInput {
            integrity: IntegrityState::Ok,
            threat: LevelClass::Low,
            vectors: Vec::new(),
            unlock_present: false,
            stability: LevelClass::Low,
            serotonin_cooldown: CooldownClass::Base,
            modulators: ModulatorField::default(),
        },
        cerebellum,
        stn: StnInput::default(),
        pmrf: PmrfInput::default(),
        dopamin: None,
        insula: InsulaInput::default(),
        sc_unlock_present: false,
        sc_replay_planned_present: false,
        pprf_cooldown_class: CooldownClass::Base,
        trace_fail_present: false,
        trace_pass_present: false,
        trace_fail_streak: 0,
    }
}

fn recommendation_input() -> CerInput {
    CerInput {
        integrity: IntegrityState::Ok,
        tool_failures: vec![(
            ToolKey::new("tool-a", "act"),
            ToolFailureCounts {
                timeouts: 5,
                partial_failures: 0,
                unavailable: 0,
            },
        )],
        ..Default::default()
    }
}

#[test]
fn cerebellum_suspend_recommendations_remain_advisory_output_not_retry_or_execution() {
    let mut bus = make_bus();
    let input = base_input(Some(recommendation_input()));

    let _ = bus.tick(input.clone());
    let output = bus.tick(input);

    let cerebellum = output.cerebellum.expect("cerebellum output");
    assert_eq!(cerebellum.divergence, LevelClass::Low);
    assert!(cerebellum.suspend_recommended);
    assert_eq!(
        output.suspend_recommendations,
        cerebellum.suspend_recommendations
    );
    assert_eq!(output.decision.profile_state, ProfileState::M1);
    assert!(output.decision.overlays.simulate_first);

    for reason in output.reason_codes {
        let reason = reason.to_ascii_lowercase();
        assert!(!reason.contains("retry"));
        assert!(!reason.contains("execute"));
        assert!(!reason.contains("memory_commit"));
        assert!(!reason.contains("compute"));
    }
}

#[test]
fn absent_cerebellum_input_does_not_create_cerebellum_state_or_recommendations() {
    let output = make_bus().tick(base_input(None));

    assert!(output.cerebellum.is_none());
    assert!(output.suspend_recommendations.is_empty());
}
