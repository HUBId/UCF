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
    } else if path.starts_with("diag:") {
        BlueBrainCanonicalReferenceKind::DiagnosticReference
    } else if path.starts_with("bb15:combined:") {
        BlueBrainCanonicalReferenceKind::CombinedBoundedReference
    } else if path.starts_with("bb8:memory_record:") {
        BlueBrainCanonicalReferenceKind::MemoryRecordReference
    } else if path.starts_with("bb14:execution:") || path.contains(":result:") {
        BlueBrainCanonicalReferenceKind::ExecutionResultReference
    } else if path.starts_with("bb3:context:")
        || path.starts_with("ctx:")
        || path.starts_with("lens_feature:")
        || path.starts_with("workspace_signal:")
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
    } else if path.contains(":result:completed") {
        BlueBrainExecutionReferenceOutcome::Successful
    } else if path.contains(":result:failed") {
        BlueBrainExecutionReferenceOutcome::Failed
    } else if path.contains(":result:cancelled") {
        BlueBrainExecutionReferenceOutcome::Cancelled
    } else if path.contains(":result:ExecutionBlocked") {
        BlueBrainExecutionReferenceOutcome::Blocked
    } else if path.contains(":result:ExecutionUnavailable") {
        BlueBrainExecutionReferenceOutcome::Unavailable
    } else if path.contains(":result:ExecutionUnsupported") {
        BlueBrainExecutionReferenceOutcome::Unsupported
    } else if path.contains(":placeholder:") {
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
}
