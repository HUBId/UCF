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
    Unknown,
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

    let validity = if lowered.contains("invalidated") {
        BlueBrainReferenceValidity::Invalidated
    } else if lowered.contains("stale") {
        BlueBrainReferenceValidity::Stale
    } else if lowered.contains("caveat")
        || lowered.contains("caveated")
        || path.contains(":placeholder:")
    {
        BlueBrainReferenceValidity::Caveated
    } else {
        BlueBrainReferenceValidity::Current
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

    BlueBrainCanonicalReference {
        raw: path.to_string(),
        kind,
        validity,
        execution_outcome,
        canonical: !matches!(
            kind,
            BlueBrainCanonicalReferenceKind::NonCanonicalInternalOnlyPath
        ),
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
        assert_eq!(placeholder.validity, BlueBrainReferenceValidity::Caveated);
    }
}
