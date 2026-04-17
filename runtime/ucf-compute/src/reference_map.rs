use crate::backends::{CANONICAL_ONBOARDING_BACKEND, CANONICAL_ONBOARDING_PACK};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeReferenceClass {
    CanonicalProduction,
    CanonicalExpertRuntimeControl,
    CanonicalDiagnosticsEvidence,
    InternalOrLegacy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeReferenceLane {
    pub class: ComputeReferenceClass,
    pub lane: &'static str,
    pub canonical_path: &'static str,
    pub scope: &'static str,
}

pub const WORKFLOW_PATH_INSPECT_DIAGNOSE_ACT: &str =
    "operations_snapshot -> diagnostics assessment -> runtime operation";
pub const WORKFLOW_PATH_REPLAY_ORIENTED: &str =
    "operations_snapshot -> replay_preflight -> replay_with_entry";
pub const WORKFLOW_PATH_ROLLOUT_ORIENTED: &str =
    "operations_snapshot.rollout diagnostics -> activation/fallback/rollback action";
pub const WORKFLOW_PATH_INTERNAL_DEV_TEST_ONLY: &str =
    "run_operation_with_entry(..., InternalDevTest)";

pub const CANONICAL_COMPUTE_REFERENCE_MAP: [ComputeReferenceLane; 7] = [
    ComputeReferenceLane {
        class: ComputeReferenceClass::CanonicalProduction,
        lane: "service_entry",
        canonical_path: "service_surface::CanonicalComputeEntryPoint::submit",
        scope: "request/job/run canonical submission and execution",
    },
    ComputeReferenceLane {
        class: ComputeReferenceClass::CanonicalProduction,
        lane: "pipeline_execution_core",
        canonical_path: "pipeline::ComputePipelineBackend::compute_canonical",
        scope: "result/fault/status core for canonical stage sequence",
    },
    ComputeReferenceLane {
        class: ComputeReferenceClass::CanonicalProduction,
        lane: "rollout_activation_core",
        canonical_path: "enablement::{active,candidate,compare,shadow} + model_store activation",
        scope: "rollout/activation/fallback/rollback core",
    },
    ComputeReferenceLane {
        class: ComputeReferenceClass::CanonicalExpertRuntimeControl,
        lane: "expert_workflow_surface",
        canonical_path:
            "service_surface::{workflow_view,replay_with_entry,run_operation_with_entry}",
        scope: "expert replay/runtime-control path on canonical contracts",
    },
    ComputeReferenceLane {
        class: ComputeReferenceClass::CanonicalDiagnosticsEvidence,
        lane: "diagnostics_evidence_history",
        canonical_path: "service_surface + evidence + job_history",
        scope: "snapshot/evidence/diagnostics/replay comparability core",
    },
    ComputeReferenceLane {
        class: ComputeReferenceClass::InternalOrLegacy,
        lane: "compatibility_backend_lane",
        canonical_path: "backends::build_backend(kind=stub|candle)",
        scope: "compatibility/dev lane; never canonical production default",
    },
    ComputeReferenceLane {
        class: ComputeReferenceClass::InternalOrLegacy,
        lane: "internal_worker_legacy_domain_lane",
        canonical_path: "build_backend(kind=worker) + domains/ai* compatibility crates",
        scope: "internal execution lane and legacy compatibility boundary",
    },
];

pub fn canonical_compute_reference_map() -> &'static [ComputeReferenceLane] {
    &CANONICAL_COMPUTE_REFERENCE_MAP
}

pub fn canonical_production_reference_lane() -> ComputeReferenceLane {
    CANONICAL_COMPUTE_REFERENCE_MAP[0]
}

pub fn canonical_onboarding_reference_summary() -> (&'static str, &'static str) {
    (
        CANONICAL_ONBOARDING_BACKEND.as_env_str(),
        CANONICAL_ONBOARDING_PACK.as_str(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_map_has_one_production_entry_lane() {
        let production_entries: Vec<_> = canonical_compute_reference_map()
            .iter()
            .filter(|lane| {
                lane.class == ComputeReferenceClass::CanonicalProduction
                    && lane.lane == "service_entry"
            })
            .collect();
        assert_eq!(production_entries.len(), 1);
        assert_eq!(
            production_entries[0].canonical_path,
            "service_surface::CanonicalComputeEntryPoint::submit"
        );
    }

    #[test]
    fn canonical_map_keeps_compatibility_constructors_non_production() {
        assert!(canonical_compute_reference_map().iter().any(|lane| {
            lane.class == ComputeReferenceClass::InternalOrLegacy
                && lane
                    .canonical_path
                    .contains("backends::build_backend(kind=stub|candle)")
        }));
        assert!(!canonical_compute_reference_map().iter().any(|lane| {
            lane.class == ComputeReferenceClass::CanonicalProduction
                && lane
                    .canonical_path
                    .contains("build_backend(kind=stub|candle)")
        }));
    }

    #[test]
    fn onboarding_summary_matches_pinned_canonical_constants() {
        let (backend, pack) = canonical_onboarding_reference_summary();
        assert_eq!(backend, "burn");
        assert_eq!(pack, "burn_toy_v1");
    }
}
