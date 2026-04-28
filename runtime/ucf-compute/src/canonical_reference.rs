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
pub struct BlueBrainReferenceConsumptionDecision {
    pub path: BlueBrainReferenceConsumptionPath,
    pub allowed: bool,
    pub advisory_only: bool,
    pub candidate_only: bool,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainCanonicalReference {
    pub raw: String,
    pub kind: BlueBrainCanonicalReferenceKind,
    pub validity: BlueBrainReferenceValidity,
    pub execution_outcome: BlueBrainExecutionReferenceOutcome,
    pub canonical: bool,
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
        };
    }

    let (path, allowed, advisory_only, candidate_only) = match layer {
        BlueBrainReferenceConsumptionLayer::Runtime => (
            BlueBrainReferenceConsumptionPath::RuntimeCanonicalReferenceConsumption,
            matches!(
                classified.kind,
                BlueBrainCanonicalReferenceKind::ExecutionResultReference
                    | BlueBrainCanonicalReferenceKind::DiagnosticReference
                    | BlueBrainCanonicalReferenceKind::ReferenceOnlyNotMemoryOrResult
            ),
            true,
            false,
        ),
        BlueBrainReferenceConsumptionLayer::Selection => (
            BlueBrainReferenceConsumptionPath::SelectionCanonicalReferenceConsumption,
            matches!(
                classified.kind,
                BlueBrainCanonicalReferenceKind::ContextReference
                    | BlueBrainCanonicalReferenceKind::MemoryRecordReference
                    | BlueBrainCanonicalReferenceKind::CombinedBoundedReference
                    | BlueBrainCanonicalReferenceKind::ReferenceOnlyNotMemoryOrResult
            ),
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
                classified.kind,
                BlueBrainCanonicalReferenceKind::ExecutionResultReference
            ),
            false,
            false,
        ),
        BlueBrainReferenceConsumptionLayer::Retrieval => (
            BlueBrainReferenceConsumptionPath::RetrievalCanonicalReferenceConsumption,
            matches!(
                classified.kind,
                BlueBrainCanonicalReferenceKind::ContextReference
                    | BlueBrainCanonicalReferenceKind::MemoryRecordReference
                    | BlueBrainCanonicalReferenceKind::ExecutionResultReference
                    | BlueBrainCanonicalReferenceKind::CombinedBoundedReference
                    | BlueBrainCanonicalReferenceKind::DiagnosticReference
                    | BlueBrainCanonicalReferenceKind::ReferenceOnlyNotMemoryOrResult
            ),
            true,
            true,
        ),
    };

    BlueBrainReferenceConsumptionDecision {
        path,
        allowed,
        advisory_only,
        candidate_only,
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
        assert!(!runtime_context.allowed);

        let runtime_execution = canonical_reference_consumption_decision(
            BlueBrainReferenceConsumptionLayer::Runtime,
            &execution,
        );
        assert!(runtime_execution.allowed);
        assert!(runtime_execution.advisory_only);

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
        assert!(!runtime_memory.allowed);

        let selection_diagnostic = canonical_reference_consumption_decision(
            BlueBrainReferenceConsumptionLayer::Selection,
            &diagnostic,
        );
        assert!(!selection_diagnostic.allowed);

        let selection_reference_only = canonical_reference_consumption_decision(
            BlueBrainReferenceConsumptionLayer::Selection,
            &reference_only,
        );
        assert!(selection_reference_only.allowed);
        assert!(selection_reference_only.candidate_only);
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
}
