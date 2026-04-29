#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainCanonicalReferenceKind {
    ContextReference,
    MemoryRecordReference,
    ExecutionResultReference,
    CombinedBoundedReference,
    DiagnosticReference,
    ReferenceOnlyNotMemoryOrResult,
    NonCanonicalInternalOnlyPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainReferenceConsumptionLayer {
    Runtime,
    Selection,
    Dynamics,
    Execution,
    Retrieval,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainReferenceConsumptionPath {
    RuntimeCanonicalReferenceConsumption,
    SelectionCanonicalReferenceConsumption,
    DynamicsCanonicalReferenceConsumption,
    ExecutionCanonicalReferenceConsumption,
    RetrievalCanonicalReferenceConsumption,
    NonCanonicalInternalOnlyReferenceConsumptionPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainCanonicalConsumptionStrength {
    StrongReferenceConsumption,
    WeakReferenceConsumption,
    ReferenceOnlyConsumption,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainReferenceConsumptionDecision {
    pub path: BlueBrainReferenceConsumptionPath,
    pub allowed: bool,
    pub advisory_only: bool,
    pub candidate_only: bool,
    pub strength: BlueBrainCanonicalConsumptionStrength,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainReferenceValidity {
    Current,
    Caveated,
    Stale,
    Invalidated,
    Blocked,
    Insufficient,
    ReferenceOnly,
    NonCanonicalInternalOnlyPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainExecutionReferenceOutcome {
    Successful,
    Failed,
    Cancelled,
    Blocked,
    Unavailable,
    Unsupported,
    PlaceholderOnly,
    NotExecutionResult,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainExecutionReferenceInteractionClass {
    ExecutionResultOnly,
    CanonicalResultReference,
    BoundedReferenceConsumption,
    FailedCancelledBlockedOrUnavailableReferenceBasis,
    CaveatedReferenceConsumption,
    NonCanonicalInternalOnlyTransitionPath,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainCanonicalReference {
    pub raw: String,
    pub kind: BlueBrainCanonicalReferenceKind,
    pub validity: BlueBrainReferenceValidity,
    pub execution_outcome: BlueBrainExecutionReferenceOutcome,
    pub canonical: bool,
}

pub fn execution_reference_interaction_class(
    classified: &BlueBrainCanonicalReference,
) -> BlueBrainExecutionReferenceInteractionClass {
    if matches!(
        classified.kind,
        BlueBrainCanonicalReferenceKind::NonCanonicalInternalOnlyPath
    ) {
        return BlueBrainExecutionReferenceInteractionClass::NonCanonicalInternalOnlyTransitionPath;
    }
    if !matches!(
        classified.kind,
        BlueBrainCanonicalReferenceKind::ExecutionResultReference
    ) {
        return BlueBrainExecutionReferenceInteractionClass::ExecutionResultOnly;
    }
    if matches!(
        classified.execution_outcome,
        BlueBrainExecutionReferenceOutcome::Successful
    ) && matches!(classified.validity, BlueBrainReferenceValidity::Current)
    {
        return BlueBrainExecutionReferenceInteractionClass::CanonicalResultReference;
    }
    if matches!(
        classified.execution_outcome,
        BlueBrainExecutionReferenceOutcome::Failed
            | BlueBrainExecutionReferenceOutcome::Cancelled
            | BlueBrainExecutionReferenceOutcome::Blocked
            | BlueBrainExecutionReferenceOutcome::Unavailable
            | BlueBrainExecutionReferenceOutcome::Unsupported
            | BlueBrainExecutionReferenceOutcome::PlaceholderOnly
            | BlueBrainExecutionReferenceOutcome::NotExecutionResult
    ) {
        return BlueBrainExecutionReferenceInteractionClass::FailedCancelledBlockedOrUnavailableReferenceBasis;
    }
    BlueBrainExecutionReferenceInteractionClass::CaveatedReferenceConsumption
}

pub fn canonical_reference_validity_state(
    classified: &BlueBrainCanonicalReference,
) -> BlueBrainReferenceValidity {
    if matches!(
        classified.kind,
        BlueBrainCanonicalReferenceKind::NonCanonicalInternalOnlyPath
    ) {
        return BlueBrainReferenceValidity::NonCanonicalInternalOnlyPath;
    }

    if matches!(
        classified.kind,
        BlueBrainCanonicalReferenceKind::ReferenceOnlyNotMemoryOrResult
            | BlueBrainCanonicalReferenceKind::DiagnosticReference
    ) {
        return BlueBrainReferenceValidity::ReferenceOnly;
    }

    if matches!(
        classified.execution_outcome,
        BlueBrainExecutionReferenceOutcome::Blocked
    ) || classified.raw.contains("maintenance_blocked")
        || classified.raw.contains(":blocked")
    {
        return BlueBrainReferenceValidity::Blocked;
    }

    if (matches!(
        classified.kind,
        BlueBrainCanonicalReferenceKind::ExecutionResultReference
    ) && matches!(
        classified.execution_outcome,
        BlueBrainExecutionReferenceOutcome::Unavailable
            | BlueBrainExecutionReferenceOutcome::Unsupported
            | BlueBrainExecutionReferenceOutcome::PlaceholderOnly
            | BlueBrainExecutionReferenceOutcome::NotExecutionResult
    )) || classified.raw.contains("missing")
        || classified.raw.contains("unavailable")
        || classified.raw.contains("insufficient")
    {
        return BlueBrainReferenceValidity::Insufficient;
    }

    if classified.raw.contains("invalidated") {
        return BlueBrainReferenceValidity::Invalidated;
    }
    if classified.raw.contains("stale") {
        return BlueBrainReferenceValidity::Stale;
    }
    if classified.raw.contains("caveat")
        || classified.raw.contains("caveated")
        || matches!(
            classified.execution_outcome,
            BlueBrainExecutionReferenceOutcome::Failed
                | BlueBrainExecutionReferenceOutcome::Cancelled
        )
    {
        return BlueBrainReferenceValidity::Caveated;
    }

    BlueBrainReferenceValidity::Current
}

pub fn classify_blue_brain_reference_path(path: &str) -> BlueBrainCanonicalReference {
    let lowered = path.to_ascii_lowercase();
    let non_canonical = lowered.contains("non_canonical")
        || lowered.contains("internal_only")
        || lowered.contains("legacy_internal");

    let kind = if non_canonical {
        BlueBrainCanonicalReferenceKind::NonCanonicalInternalOnlyPath
    } else if lowered.starts_with("diag:") {
        BlueBrainCanonicalReferenceKind::DiagnosticReference
    } else if lowered.starts_with("bb15:combined:") {
        BlueBrainCanonicalReferenceKind::CombinedBoundedReference
    } else if lowered.starts_with("bb8:memory_record:") {
        BlueBrainCanonicalReferenceKind::MemoryRecordReference
    } else if lowered.starts_with("bb14:execution:") || lowered.contains(":result:") {
        BlueBrainCanonicalReferenceKind::ExecutionResultReference
    } else if lowered.starts_with("bb3:context:")
        || lowered.starts_with("ctx:")
        || lowered.starts_with("lens_feature:")
        || lowered.starts_with("workspace_signal:")
    {
        BlueBrainCanonicalReferenceKind::ContextReference
    } else {
        BlueBrainCanonicalReferenceKind::ReferenceOnlyNotMemoryOrResult
    };

    let execution_outcome = if !matches!(
        kind,
        BlueBrainCanonicalReferenceKind::ExecutionResultReference
    ) {
        BlueBrainExecutionReferenceOutcome::NotExecutionResult
    } else if lowered.contains(":result:completed") {
        BlueBrainExecutionReferenceOutcome::Successful
    } else if lowered.contains(":result:failed") {
        BlueBrainExecutionReferenceOutcome::Failed
    } else if lowered.contains(":result:cancelled") {
        BlueBrainExecutionReferenceOutcome::Cancelled
    } else if lowered.contains(":result:executionblocked") {
        BlueBrainExecutionReferenceOutcome::Blocked
    } else if lowered.contains(":result:executionunavailable") {
        BlueBrainExecutionReferenceOutcome::Unavailable
    } else if lowered.contains(":result:executionunsupported") {
        BlueBrainExecutionReferenceOutcome::Unsupported
    } else if lowered.contains(":placeholder:") {
        BlueBrainExecutionReferenceOutcome::PlaceholderOnly
    } else {
        BlueBrainExecutionReferenceOutcome::NotExecutionResult
    };

    let mut classified = BlueBrainCanonicalReference {
        raw: path.to_string(),
        kind,
        validity: BlueBrainReferenceValidity::Current,
        execution_outcome,
        canonical: !matches!(
            kind,
            BlueBrainCanonicalReferenceKind::NonCanonicalInternalOnlyPath
        ),
    };
    classified.validity = canonical_reference_validity_state(&classified);
    classified
}

pub fn canonical_reference_consumption_decision(
    layer: BlueBrainReferenceConsumptionLayer,
    classified: &BlueBrainCanonicalReference,
) -> BlueBrainReferenceConsumptionDecision {
    if matches!(
        classified.kind,
        BlueBrainCanonicalReferenceKind::NonCanonicalInternalOnlyPath
    ) {
        return BlueBrainReferenceConsumptionDecision {
            path:
                BlueBrainReferenceConsumptionPath::NonCanonicalInternalOnlyReferenceConsumptionPath,
            allowed: false,
            advisory_only: true,
            candidate_only: true,
            strength: BlueBrainCanonicalConsumptionStrength::ReferenceOnlyConsumption,
        };
    }

    let interaction_class = execution_reference_interaction_class(classified);
    let weak_execution_basis = matches!(
        interaction_class,
        BlueBrainExecutionReferenceInteractionClass::FailedCancelledBlockedOrUnavailableReferenceBasis
            | BlueBrainExecutionReferenceInteractionClass::CaveatedReferenceConsumption
    );
    let canonical_cross_line_kind = matches!(
        classified.kind,
        BlueBrainCanonicalReferenceKind::ContextReference
            | BlueBrainCanonicalReferenceKind::MemoryRecordReference
            | BlueBrainCanonicalReferenceKind::ExecutionResultReference
            | BlueBrainCanonicalReferenceKind::CombinedBoundedReference
            | BlueBrainCanonicalReferenceKind::DiagnosticReference
            | BlueBrainCanonicalReferenceKind::ReferenceOnlyNotMemoryOrResult
    );

    let reference_only_kind = matches!(
        classified.kind,
        BlueBrainCanonicalReferenceKind::DiagnosticReference
            | BlueBrainCanonicalReferenceKind::ReferenceOnlyNotMemoryOrResult
    ) || matches!(
        classified.validity,
        BlueBrainReferenceValidity::ReferenceOnly
    );

    let (path, allowed, advisory_only, candidate_only) = match layer {
        BlueBrainReferenceConsumptionLayer::Runtime => (
            BlueBrainReferenceConsumptionPath::RuntimeCanonicalReferenceConsumption,
            canonical_cross_line_kind,
            true,
            false,
        ),
        BlueBrainReferenceConsumptionLayer::Selection => (
            BlueBrainReferenceConsumptionPath::SelectionCanonicalReferenceConsumption,
            canonical_cross_line_kind,
            true,
            true,
        ),
        BlueBrainReferenceConsumptionLayer::Dynamics => (
            BlueBrainReferenceConsumptionPath::DynamicsCanonicalReferenceConsumption,
            matches!(
                classified.kind,
                BlueBrainCanonicalReferenceKind::ExecutionResultReference
                    | BlueBrainCanonicalReferenceKind::DiagnosticReference
                    | BlueBrainCanonicalReferenceKind::ReferenceOnlyNotMemoryOrResult
            ),
            true,
            false,
        ),
        BlueBrainReferenceConsumptionLayer::Execution => (
            BlueBrainReferenceConsumptionPath::ExecutionCanonicalReferenceConsumption,
            matches!(
                interaction_class,
                BlueBrainExecutionReferenceInteractionClass::CanonicalResultReference
            ),
            false,
            false,
        ),
        BlueBrainReferenceConsumptionLayer::Retrieval => (
            BlueBrainReferenceConsumptionPath::RetrievalCanonicalReferenceConsumption,
            canonical_cross_line_kind,
            true,
            true,
        ),
    };

    let candidate_only = candidate_only || weak_execution_basis || reference_only_kind;
    let strength = if reference_only_kind {
        BlueBrainCanonicalConsumptionStrength::ReferenceOnlyConsumption
    } else if weak_execution_basis
        || matches!(
            classified.validity,
            BlueBrainReferenceValidity::Caveated
                | BlueBrainReferenceValidity::Stale
                | BlueBrainReferenceValidity::Invalidated
                | BlueBrainReferenceValidity::Blocked
                | BlueBrainReferenceValidity::Insufficient
        )
    {
        BlueBrainCanonicalConsumptionStrength::WeakReferenceConsumption
    } else {
        BlueBrainCanonicalConsumptionStrength::StrongReferenceConsumption
    };

    BlueBrainReferenceConsumptionDecision {
        path,
        allowed,
        advisory_only,
        candidate_only,
        strength,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_canonical_reference_kinds_without_overlap() {
        let context = classify_blue_brain_reference_path("bb3:context:turn:42");
        assert_eq!(
            context.kind,
            BlueBrainCanonicalReferenceKind::ContextReference
        );

        let memory = classify_blue_brain_reference_path("bb8:memory_record:mem-42");
        assert_eq!(
            memory.kind,
            BlueBrainCanonicalReferenceKind::MemoryRecordReference
        );

        let execution = classify_blue_brain_reference_path("bb14:execution:h1:result:completed");
        assert_eq!(
            execution.kind,
            BlueBrainCanonicalReferenceKind::ExecutionResultReference
        );
        assert_eq!(
            execution.execution_outcome,
            BlueBrainExecutionReferenceOutcome::Successful
        );

        let combined = classify_blue_brain_reference_path("bb15:combined:candidate:7");
        assert_eq!(
            combined.kind,
            BlueBrainCanonicalReferenceKind::CombinedBoundedReference
        );

        let diagnostic = classify_blue_brain_reference_path("diag:runtime:tick:7");
        assert_eq!(
            diagnostic.kind,
            BlueBrainCanonicalReferenceKind::DiagnosticReference
        );

        let internal = classify_blue_brain_reference_path(
            "bb14:execution:h1:result:completed:non_canonical_internal_only",
        );
        assert_eq!(
            internal.kind,
            BlueBrainCanonicalReferenceKind::NonCanonicalInternalOnlyPath
        );
        assert!(!internal.canonical);
    }

    #[test]
    fn classifies_lifecycle_and_execution_outcomes() {
        let stale = classify_blue_brain_reference_path("bb8:memory_record:mem-1:stale");
        assert_eq!(stale.validity, BlueBrainReferenceValidity::Stale);

        let invalidated = classify_blue_brain_reference_path("bb8:memory_record:mem-1:invalidated");
        assert_eq!(
            invalidated.validity,
            BlueBrainReferenceValidity::Invalidated
        );

        let blocked =
            classify_blue_brain_reference_path("bb14:execution:h2:result:ExecutionBlocked");
        assert_eq!(
            blocked.execution_outcome,
            BlueBrainExecutionReferenceOutcome::Blocked
        );

        let unavailable =
            classify_blue_brain_reference_path("bb14:execution:h2:result:ExecutionUnavailable");
        assert_eq!(
            unavailable.execution_outcome,
            BlueBrainExecutionReferenceOutcome::Unavailable
        );

        let placeholder = classify_blue_brain_reference_path("bb14:execution:h2:placeholder:slot");
        assert_eq!(
            placeholder.execution_outcome,
            BlueBrainExecutionReferenceOutcome::PlaceholderOnly
        );
        assert_eq!(
            placeholder.validity,
            BlueBrainReferenceValidity::Insufficient
        );

        let reference_only = classify_blue_brain_reference_path("diag:runtime:insufficient_basis");
        assert_eq!(
            reference_only.validity,
            BlueBrainReferenceValidity::ReferenceOnly
        );

        let blocked_memory =
            classify_blue_brain_reference_path("bb8:memory_record:mem-1:maintenance_blocked");
        assert_eq!(blocked_memory.validity, BlueBrainReferenceValidity::Blocked);

        let unavailable_execution = classify_blue_brain_reference_path(
            "bb14:execution:h2:result:ExecutionUnavailable:insufficient",
        );
        assert_eq!(
            unavailable_execution.validity,
            BlueBrainReferenceValidity::Insufficient
        );
    }

    #[test]
    fn reference_consumption_layers_reject_non_canonical_internal_only_paths() {
        let internal = classify_blue_brain_reference_path(
            "bb8:memory_record:mem-1:non_canonical_internal_only",
        );
        for layer in [
            BlueBrainReferenceConsumptionLayer::Runtime,
            BlueBrainReferenceConsumptionLayer::Selection,
            BlueBrainReferenceConsumptionLayer::Dynamics,
            BlueBrainReferenceConsumptionLayer::Execution,
            BlueBrainReferenceConsumptionLayer::Retrieval,
        ] {
            let decision = canonical_reference_consumption_decision(layer, &internal);
            assert_eq!(
                decision.path,
                BlueBrainReferenceConsumptionPath::NonCanonicalInternalOnlyReferenceConsumptionPath
            );
            assert!(!decision.allowed);
            assert_eq!(
                decision.strength,
                BlueBrainCanonicalConsumptionStrength::ReferenceOnlyConsumption
            );
        }
    }

    #[test]
    fn reference_consumption_layers_enforce_expected_canonical_reference_kinds() {
        let context = classify_blue_brain_reference_path("bb3:context:turn:42");
        let memory = classify_blue_brain_reference_path("bb8:memory_record:mem-42");
        let execution = classify_blue_brain_reference_path("bb14:execution:h7:result:completed");
        let combined = classify_blue_brain_reference_path("bb15:combined:candidate-1");
        let diagnostic = classify_blue_brain_reference_path("diag:runtime:tick:8");
        let reference_only = classify_blue_brain_reference_path("aux:reference:hint");

        let runtime_context = canonical_reference_consumption_decision(
            BlueBrainReferenceConsumptionLayer::Runtime,
            &context,
        );
        assert!(runtime_context.allowed);
        assert_eq!(
            runtime_context.strength,
            BlueBrainCanonicalConsumptionStrength::StrongReferenceConsumption
        );

        let runtime_execution = canonical_reference_consumption_decision(
            BlueBrainReferenceConsumptionLayer::Runtime,
            &execution,
        );
        assert!(runtime_execution.allowed);
        assert!(runtime_execution.advisory_only);
        assert_eq!(
            runtime_execution.strength,
            BlueBrainCanonicalConsumptionStrength::StrongReferenceConsumption
        );

        let selection_combined = canonical_reference_consumption_decision(
            BlueBrainReferenceConsumptionLayer::Selection,
            &combined,
        );
        assert!(selection_combined.allowed);
        assert!(selection_combined.candidate_only);

        let execution_context = canonical_reference_consumption_decision(
            BlueBrainReferenceConsumptionLayer::Execution,
            &context,
        );
        assert!(!execution_context.allowed);

        let runtime_memory = canonical_reference_consumption_decision(
            BlueBrainReferenceConsumptionLayer::Runtime,
            &memory,
        );
        assert!(runtime_memory.allowed);
        assert_eq!(
            runtime_memory.strength,
            BlueBrainCanonicalConsumptionStrength::StrongReferenceConsumption
        );

        let selection_diagnostic = canonical_reference_consumption_decision(
            BlueBrainReferenceConsumptionLayer::Selection,
            &diagnostic,
        );
        assert!(selection_diagnostic.allowed);
        assert_eq!(
            selection_diagnostic.strength,
            BlueBrainCanonicalConsumptionStrength::ReferenceOnlyConsumption
        );

        let selection_reference_only = canonical_reference_consumption_decision(
            BlueBrainReferenceConsumptionLayer::Selection,
            &reference_only,
        );
        assert!(selection_reference_only.allowed);
        assert!(selection_reference_only.candidate_only);
        assert_eq!(
            selection_reference_only.strength,
            BlueBrainCanonicalConsumptionStrength::ReferenceOnlyConsumption
        );
    }

    #[test]
    fn validity_states_remain_distinct_across_canonical_and_non_canonical_lanes() {
        let current = classify_blue_brain_reference_path("bb3:context:turn:99");
        assert_eq!(current.validity, BlueBrainReferenceValidity::Current);

        let caveated =
            classify_blue_brain_reference_path("bb14:execution:h9:result:failed:with_caveat");
        assert_eq!(caveated.validity, BlueBrainReferenceValidity::Caveated);

        let stale = classify_blue_brain_reference_path("bb8:memory_record:mem-3:stale");
        assert_eq!(stale.validity, BlueBrainReferenceValidity::Stale);

        let invalidated = classify_blue_brain_reference_path("bb8:memory_record:mem-4:invalidated");
        assert_eq!(
            invalidated.validity,
            BlueBrainReferenceValidity::Invalidated
        );

        let blocked =
            classify_blue_brain_reference_path("bb8:memory_record:mem-5:maintenance_blocked");
        assert_eq!(blocked.validity, BlueBrainReferenceValidity::Blocked);

        let insufficient = classify_blue_brain_reference_path(
            "bb14:execution:h10:result:ExecutionUnavailable:insufficient",
        );
        assert_eq!(
            insufficient.validity,
            BlueBrainReferenceValidity::Insufficient
        );

        let reference_only = classify_blue_brain_reference_path("diag:runtime:insufficient_basis");
        assert_eq!(
            reference_only.validity,
            BlueBrainReferenceValidity::ReferenceOnly
        );

        let non_canonical =
            classify_blue_brain_reference_path("bb3:context:turn:100:non_canonical_internal_only");
        assert_eq!(
            non_canonical.validity,
            BlueBrainReferenceValidity::NonCanonicalInternalOnlyPath
        );
    }

    #[test]
    fn execution_reference_classification_is_case_insensitive_for_status_tokens() {
        let blocked =
            classify_blue_brain_reference_path("bb14:execution:h2:result:executionblocked");
        assert_eq!(
            blocked.execution_outcome,
            BlueBrainExecutionReferenceOutcome::Blocked
        );

        let unavailable =
            classify_blue_brain_reference_path("BB14:EXECUTION:H2:RESULT:ExecutionUnavailable");
        assert_eq!(
            unavailable.execution_outcome,
            BlueBrainExecutionReferenceOutcome::Unavailable
        );

        let unsupported =
            classify_blue_brain_reference_path("bb14:execution:h2:result:EXECUTIONUNSUPPORTED");
        assert_eq!(
            unsupported.execution_outcome,
            BlueBrainExecutionReferenceOutcome::Unsupported
        );
    }

    #[test]
    fn interaction_classes_keep_result_reference_and_consumption_boundaries_explicit() {
        let completed = classify_blue_brain_reference_path("bb14:execution:h1:result:completed");
        assert_eq!(
            execution_reference_interaction_class(&completed),
            BlueBrainExecutionReferenceInteractionClass::CanonicalResultReference
        );
        let failed = classify_blue_brain_reference_path("bb14:execution:h2:result:failed");
        assert_eq!(
            execution_reference_interaction_class(&failed),
            BlueBrainExecutionReferenceInteractionClass::FailedCancelledBlockedOrUnavailableReferenceBasis
        );
        let placeholder = classify_blue_brain_reference_path("bb14:execution:h2:placeholder:slot");
        assert_eq!(
            execution_reference_interaction_class(&placeholder),
            BlueBrainExecutionReferenceInteractionClass::FailedCancelledBlockedOrUnavailableReferenceBasis
        );
    }

    #[test]
    fn weak_execution_basis_stays_candidate_only_and_cannot_drive_execution_layer() {
        let failed = classify_blue_brain_reference_path("bb14:execution:h2:result:failed");
        let runtime = canonical_reference_consumption_decision(
            BlueBrainReferenceConsumptionLayer::Runtime,
            &failed,
        );
        assert!(runtime.allowed);
        assert!(runtime.advisory_only);
        assert!(runtime.candidate_only);
        assert_eq!(
            runtime.strength,
            BlueBrainCanonicalConsumptionStrength::WeakReferenceConsumption
        );

        let execution = canonical_reference_consumption_decision(
            BlueBrainReferenceConsumptionLayer::Execution,
            &failed,
        );
        assert!(!execution.allowed);
    }

    #[test]
    fn bb21_prompt4_doc_stays_pinned_to_canonical_boundaries_and_no_direct_guards() {
        let doc = std::fs::read_to_string(
            "../../docs/blue_brain_bb21_readiness_sweep_execution_reference_interaction_closure_serie_bb21_prompt4_v1.md",
        )
        .expect("BB21 prompt4 closure doc must exist");
        assert!(doc.contains("stable execution/reference interaction line"));
        assert!(doc.contains("StrongReferenceConsumption"));
        assert!(doc.contains("WeakReferenceConsumption"));
        assert!(doc.contains("ReferenceOnlyConsumption"));
        assert!(doc.contains("NonCanonicalInternalOnlyPath"));
        assert!(doc.contains("keine direkte Folge-Execution"));
        assert!(doc.contains("keine Retry-Orchestrierung"));
        assert!(doc.contains("keine automatische Memory-Persistenz"));
        assert!(doc.contains("keine Compute-Core-Ausweitung"));
        assert!(doc.contains("BB14 execution-integrity line"));
        assert!(doc.contains("BB15 bounded retrieval/reference line"));
        assert!(doc.contains("BB17 context/memory/reference hardening line"));
        assert!(doc.contains("BB19 runtime/selection contract line"));
        assert!(doc.contains("Priorität 1: BB22 narrow cross-line stabilization pass."));
    }

    #[test]
    fn bb22_prompt1_doc_pins_cross_line_stabilization_classes_and_boundaries() {
        let doc = std::fs::read_to_string(
            "../../docs/blue_brain_bb22_narrow_cross_line_stabilization_pass_serie_bb22_prompt1_v1.md",
        )
        .expect("BB22 prompt1 stabilization doc must exist");
        assert!(doc.contains("stable cross-line path"));
        assert!(doc.contains("cross-line usable with caveats"));
        assert!(doc.contains("advisory-only bounded path"));
        assert!(doc.contains("weak/reference-only path"));
        assert!(doc.contains("blocked/insufficient path"));
        assert!(doc.contains("non-canonical/internal-only path"));
        assert!(doc.contains("Runtime ↔ Selection"));
        assert!(doc.contains("Execution → Reference → Consumption"));
        assert!(doc.contains("bounded advisory-only Dynamics → Runtime/Selection"));
        assert!(doc.contains("keine direkte Folge-Execution"));
        assert!(doc.contains("keine Retry-Orchestrierung"));
        assert!(doc.contains("keine Compute Invocation außerhalb kanonischer Pfade"));
        assert!(doc.contains("keine implizite Memory-Persistenz"));
        assert!(doc.contains("keine Planner-/Policy-/Agentenlogik-Erweiterung"));
        assert!(doc.contains("keine Neurodynamik-Autoritätserweiterung"));
        assert!(doc.contains("Priorität 1: finaler repo-weiter Abschluss-/Freeze-Pass"));
    }
}
