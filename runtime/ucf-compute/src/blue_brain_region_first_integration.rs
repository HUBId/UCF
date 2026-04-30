use crate::{
    BlueBrainCandidateDeferralLifecycleClass, BlueBrainContextEvidencePriorityClass,
    BlueBrainControlAttentionSelectionClass, BlueBrainReferenceValidity,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFirstRegionClass {
    AttentionSelectionRelated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFirstRegionPathClass {
    RegionToRuntimeAdvisorySignal,
    RuntimeToRegionBoundedInput,
    RegionToSelectionAdvisorySignal,
    SelectionToRegionBoundedStateInput,
    RegionReferenceSignal,
    CaveatedDeferredBlockedRegionContractSignal,
    ReferenceOnlyRegionContractSignal,
    RegionInputSurface,
    RegionStateSurface,
    RegionOutputAdvisorySurface,
    RegionReferenceSurface,
    BlockedDeferredRegionPath,
    NonCanonicalInternalOnlyRegionPath,
}

pub const CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP: [BlueBrainFirstRegionPathClass; 13] = [
    BlueBrainFirstRegionPathClass::RegionToRuntimeAdvisorySignal,
    BlueBrainFirstRegionPathClass::RuntimeToRegionBoundedInput,
    BlueBrainFirstRegionPathClass::RegionToSelectionAdvisorySignal,
    BlueBrainFirstRegionPathClass::SelectionToRegionBoundedStateInput,
    BlueBrainFirstRegionPathClass::RegionReferenceSignal,
    BlueBrainFirstRegionPathClass::CaveatedDeferredBlockedRegionContractSignal,
    BlueBrainFirstRegionPathClass::ReferenceOnlyRegionContractSignal,
    BlueBrainFirstRegionPathClass::RegionInputSurface,
    BlueBrainFirstRegionPathClass::RegionStateSurface,
    BlueBrainFirstRegionPathClass::RegionOutputAdvisorySurface,
    BlueBrainFirstRegionPathClass::RegionReferenceSurface,
    BlueBrainFirstRegionPathClass::BlockedDeferredRegionPath,
    BlueBrainFirstRegionPathClass::NonCanonicalInternalOnlyRegionPath,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFirstRegionStateSurface {
    ActiveBoundedAdvisoryOnly,
    BlockedDeferred,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFirstRegionInputSource {
    RuntimeSelectionSignal,
    RuntimeDeferralSignal,
    ContextReferenceSignal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFirstRegionInputGuard {
    Canonical,
    RejectedToolActionControl,
    RejectedComputeInternalState,
    RejectedSafetyOverride,
    RejectedImplicitMemoryMutation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainFirstRegionInputSurface {
    pub attention_class: BlueBrainControlAttentionSelectionClass,
    pub deferral_class: BlueBrainCandidateDeferralLifecycleClass,
    pub reference_validity: BlueBrainReferenceValidity,
    pub context_priority: BlueBrainContextEvidencePriorityClass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFirstRegionAdvisoryOutputClass {
    CaveatHint,
    PriorityHint,
    DeferralHint,
    ReferenceBoundedSignal,
    BlockedDeferred,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainFirstRegionContractSignal {
    RegionToRuntimeAdvisory,
    RuntimeToRegionBoundedInput,
    RegionToSelectionAdvisory,
    SelectionToRegionBoundedStateInput,
    RegionReference,
    Caveated,
    Deferred,
    Blocked,
    ReferenceOnly,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainFirstRegionOutputSurface {
    pub advisory_class: BlueBrainFirstRegionAdvisoryOutputClass,
    pub runtime_advisory_only: bool,
    pub selection_advisory_only: bool,
    pub reference_bounded_only: bool,
    pub direct_action_selection: bool,
    pub direct_execution_trigger: bool,
    pub direct_retry_trigger: bool,
    pub direct_memory_commit: bool,
    pub direct_compute_invocation: bool,
    pub safety_override: bool,
    pub contract_signal: BlueBrainFirstRegionContractSignal,
    pub reference_only: bool,
}

pub fn classify_blue_brain_first_region_input_guard(
    source: BlueBrainFirstRegionInputSource,
) -> BlueBrainFirstRegionInputGuard {
    match source {
        BlueBrainFirstRegionInputSource::RuntimeSelectionSignal
        | BlueBrainFirstRegionInputSource::RuntimeDeferralSignal
        | BlueBrainFirstRegionInputSource::ContextReferenceSignal => {
            BlueBrainFirstRegionInputGuard::Canonical
        }
    }
}

pub fn evaluate_blue_brain_first_region_attention_selection(
    input: BlueBrainFirstRegionInputSurface,
) -> (
    BlueBrainFirstRegionStateSurface,
    BlueBrainFirstRegionOutputSurface,
) {
    let blocked_or_deferred = matches!(
        input.deferral_class,
        BlueBrainCandidateDeferralLifecycleClass::CandidateDeferred
            | BlueBrainCandidateDeferralLifecycleClass::CandidateDeferredPendingStrongerEvidence
            | BlueBrainCandidateDeferralLifecycleClass::CandidateDeferredPendingContextUpdate
            | BlueBrainCandidateDeferralLifecycleClass::CandidateRejected
            | BlueBrainCandidateDeferralLifecycleClass::CandidateStale
            | BlueBrainCandidateDeferralLifecycleClass::CandidateInsufficient
    );

    let non_canonical_attention = input.attention_class
        == BlueBrainControlAttentionSelectionClass::NonCanonicalInternalOnlySelectionPath;

    let state = if non_canonical_attention {
        BlueBrainFirstRegionStateSurface::NonCanonicalInternalOnly
    } else if blocked_or_deferred {
        BlueBrainFirstRegionStateSurface::BlockedDeferred
    } else {
        BlueBrainFirstRegionStateSurface::ActiveBoundedAdvisoryOnly
    };

    let advisory_class = match state {
        BlueBrainFirstRegionStateSurface::NonCanonicalInternalOnly => {
            BlueBrainFirstRegionAdvisoryOutputClass::NonCanonicalInternalOnly
        }
        BlueBrainFirstRegionStateSurface::BlockedDeferred => {
            BlueBrainFirstRegionAdvisoryOutputClass::BlockedDeferred
        }
        BlueBrainFirstRegionStateSurface::ActiveBoundedAdvisoryOnly => {
            if input.reference_validity == BlueBrainReferenceValidity::Caveated {
                BlueBrainFirstRegionAdvisoryOutputClass::CaveatHint
            } else if matches!(
                input.context_priority,
                BlueBrainContextEvidencePriorityClass::PrimaryContext
                    | BlueBrainContextEvidencePriorityClass::PrimaryEvidenceReference
            ) {
                BlueBrainFirstRegionAdvisoryOutputClass::PriorityHint
            } else if matches!(
                input.context_priority,
                BlueBrainContextEvidencePriorityClass::DeferredContext
                    | BlueBrainContextEvidencePriorityClass::StaleContext
                    | BlueBrainContextEvidencePriorityClass::InsufficientContext
            ) {
                BlueBrainFirstRegionAdvisoryOutputClass::DeferralHint
            } else {
                BlueBrainFirstRegionAdvisoryOutputClass::ReferenceBoundedSignal
            }
        }
    };

    let contract_signal = if non_canonical_attention {
        BlueBrainFirstRegionContractSignal::NonCanonicalInternalOnly
    } else if blocked_or_deferred {
        if matches!(
            input.deferral_class,
            BlueBrainCandidateDeferralLifecycleClass::CandidateDeferred
                | BlueBrainCandidateDeferralLifecycleClass::CandidateDeferredPendingStrongerEvidence
                | BlueBrainCandidateDeferralLifecycleClass::CandidateDeferredPendingContextUpdate
        ) {
            BlueBrainFirstRegionContractSignal::Deferred
        } else {
            BlueBrainFirstRegionContractSignal::Blocked
        }
    } else if matches!(
        input.reference_validity,
        BlueBrainReferenceValidity::Caveated | BlueBrainReferenceValidity::Stale
    ) {
        BlueBrainFirstRegionContractSignal::Caveated
    } else if matches!(
        input.reference_validity,
        BlueBrainReferenceValidity::ReferenceOnly
    ) {
        BlueBrainFirstRegionContractSignal::ReferenceOnly
    } else {
        BlueBrainFirstRegionContractSignal::RegionToRuntimeAdvisory
    };

    (
        state,
        BlueBrainFirstRegionOutputSurface {
            advisory_class,
            runtime_advisory_only: true,
            selection_advisory_only: true,
            reference_bounded_only: true,
            direct_action_selection: false,
            direct_execution_trigger: false,
            direct_retry_trigger: false,
            direct_memory_commit: false,
            direct_compute_invocation: false,
            safety_override: false,
            contract_signal,
            reference_only: contract_signal == BlueBrainFirstRegionContractSignal::ReferenceOnly,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_region_map_contains_all_required_paths() {
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstRegionPathClass::RegionToRuntimeAdvisorySignal));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstRegionPathClass::RuntimeToRegionBoundedInput));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstRegionPathClass::RegionToSelectionAdvisorySignal));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstRegionPathClass::SelectionToRegionBoundedStateInput));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstRegionPathClass::RegionReferenceSignal));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstRegionPathClass::CaveatedDeferredBlockedRegionContractSignal));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstRegionPathClass::ReferenceOnlyRegionContractSignal));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstRegionPathClass::RegionInputSurface));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstRegionPathClass::RegionStateSurface));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstRegionPathClass::RegionOutputAdvisorySurface));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstRegionPathClass::RegionReferenceSurface));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstRegionPathClass::BlockedDeferredRegionPath));
        assert!(CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP
            .contains(&BlueBrainFirstRegionPathClass::NonCanonicalInternalOnlyRegionPath));
    }

    #[test]
    fn first_region_output_is_advisory_and_non_authoritative() {
        let (_, output) = evaluate_blue_brain_first_region_attention_selection(
            BlueBrainFirstRegionInputSurface {
                attention_class: BlueBrainControlAttentionSelectionClass::AttentionTarget,
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
            },
        );

        assert!(output.runtime_advisory_only);
        assert!(output.selection_advisory_only);
        assert!(output.reference_bounded_only);
        assert!(!output.direct_action_selection);
        assert!(!output.direct_execution_trigger);
        assert!(!output.direct_retry_trigger);
        assert!(!output.direct_memory_commit);
        assert!(!output.direct_compute_invocation);
        assert!(!output.safety_override);
        assert_eq!(
            output.contract_signal,
            BlueBrainFirstRegionContractSignal::RegionToRuntimeAdvisory
        );
    }

    #[test]
    fn first_region_marks_blocked_and_non_canonical_paths() {
        let (blocked_state, _) = evaluate_blue_brain_first_region_attention_selection(
            BlueBrainFirstRegionInputSurface {
                attention_class: BlueBrainControlAttentionSelectionClass::AttentionTarget,
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateRejected,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority:
                    BlueBrainContextEvidencePriorityClass::SupportingEvidenceReference,
            },
        );
        assert_eq!(
            blocked_state,
            BlueBrainFirstRegionStateSurface::BlockedDeferred
        );

        let (non_canonical_state, non_canonical_output) =
            evaluate_blue_brain_first_region_attention_selection(BlueBrainFirstRegionInputSurface {
                attention_class:
                    BlueBrainControlAttentionSelectionClass::NonCanonicalInternalOnlySelectionPath,
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority: BlueBrainContextEvidencePriorityClass::SupportingEvidenceReference,
            });
        assert_eq!(
            non_canonical_state,
            BlueBrainFirstRegionStateSurface::NonCanonicalInternalOnly
        );
        assert_eq!(
            non_canonical_output.advisory_class,
            BlueBrainFirstRegionAdvisoryOutputClass::NonCanonicalInternalOnly
        );
        assert_eq!(
            non_canonical_output.contract_signal,
            BlueBrainFirstRegionContractSignal::NonCanonicalInternalOnly
        );
    }

    #[test]
    fn first_region_distinguishes_deferred_blocked_caveated_and_reference_only() {
        let (_, deferred) = evaluate_blue_brain_first_region_attention_selection(
            BlueBrainFirstRegionInputSurface {
                attention_class: BlueBrainControlAttentionSelectionClass::AttentionTarget,
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateDeferred,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority: BlueBrainContextEvidencePriorityClass::DeferredContext,
            },
        );
        assert_eq!(
            deferred.contract_signal,
            BlueBrainFirstRegionContractSignal::Deferred
        );

        let (_, blocked) = evaluate_blue_brain_first_region_attention_selection(
            BlueBrainFirstRegionInputSurface {
                attention_class: BlueBrainControlAttentionSelectionClass::AttentionTarget,
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateRejected,
                reference_validity: BlueBrainReferenceValidity::Current,
                context_priority: BlueBrainContextEvidencePriorityClass::PrimaryContext,
            },
        );
        assert_eq!(
            blocked.contract_signal,
            BlueBrainFirstRegionContractSignal::Blocked
        );

        let (_, caveated) = evaluate_blue_brain_first_region_attention_selection(
            BlueBrainFirstRegionInputSurface {
                attention_class: BlueBrainControlAttentionSelectionClass::AttentionTarget,
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::Caveated,
                context_priority: BlueBrainContextEvidencePriorityClass::SupportingContext,
            },
        );
        assert_eq!(
            caveated.contract_signal,
            BlueBrainFirstRegionContractSignal::Caveated
        );

        let (_, reference_only) = evaluate_blue_brain_first_region_attention_selection(
            BlueBrainFirstRegionInputSurface {
                attention_class: BlueBrainControlAttentionSelectionClass::AttentionTarget,
                deferral_class: BlueBrainCandidateDeferralLifecycleClass::CandidateSelected,
                reference_validity: BlueBrainReferenceValidity::ReferenceOnly,
                context_priority:
                    BlueBrainContextEvidencePriorityClass::SupportingEvidenceReference,
            },
        );
        assert_eq!(
            reference_only.contract_signal,
            BlueBrainFirstRegionContractSignal::ReferenceOnly
        );
        assert!(reference_only.reference_only);
    }
}
