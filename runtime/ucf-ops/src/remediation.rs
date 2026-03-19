use std::collections::BTreeSet;

use ucf_types::remediation_codes::{remediation_for_condition, CanonicalConditionV1};

pub fn remediation_code_strings_for_condition(condition: CanonicalConditionV1) -> Vec<String> {
    remediation_for_condition(condition)
        .into_iter()
        .map(|c| c.stable_code().to_string())
        .collect()
}

pub fn canonical_from_legacy_code(code: &str) -> Vec<String> {
    let condition = match code {
        "NO_PROBE" | "NO_PROBE_EVIDENCE" => CanonicalConditionV1::EvidenceMissing("probe"),
        "STALE_PROBE" | "STALE_PROBE_EVIDENCE" => CanonicalConditionV1::EvidenceStale("probe"),
        "NO_COMPARE" | "NO_COMPARE_EVIDENCE" => CanonicalConditionV1::EvidenceMissing("compare"),
        "STALE_COMPARE" | "STALE_COMPARE_EVIDENCE" => {
            CanonicalConditionV1::EvidenceStale("compare")
        }
        "HASH_MISMATCH" | "TARGET_HASH_MISMATCH" => CanonicalConditionV1::HashMismatch("target"),
        "DRIFT_SEVERE" => CanonicalConditionV1::DriftSevere("slot"),
        "STRICT_FAIL" => CanonicalConditionV1::StrictFail("strict"),
        "MANIFEST_INVALID" => CanonicalConditionV1::ManifestInvalid("manifest"),
        "OPTIONAL_BACKEND_CLOSED_UNSUPPORTED" => CanonicalConditionV1::OptionalBackendMissing {
            slot: "world",
            backend: "burn",
        },
        _ => CanonicalConditionV1::Unknown,
    };
    remediation_code_strings_for_condition(condition)
}

pub fn canonical_from_legacy_remediation(remediation: &str) -> Vec<String> {
    let condition = match remediation {
        "run_models_eligibility" | "run_probe" => CanonicalConditionV1::EvidenceMissing("probe"),
        "run_strict_check" => CanonicalConditionV1::StrictFail("strict"),
        "run_drift_report" | "inspect_active_alerts" => CanonicalConditionV1::DriftSevere("slot"),
        "run_operator_report" => CanonicalConditionV1::GateFail("operator"),
        "run_backend_evidence_snapshot" => CanonicalConditionV1::EvidenceMissing("probe"),
        "run_v0_gate" | "run_v1_gate" | "run_v2_gate" | "run_v3_gate" | "run_missing_gates" => {
            CanonicalConditionV1::GateFail("v")
        }
        "run_models_active_check" => CanonicalConditionV1::ActiveUnsupported("slot"),
        "run_portability_report" => CanonicalConditionV1::OptionalBackendMissing {
            slot: "world",
            backend: "burn",
        },
        "run_verify_manifest" => CanonicalConditionV1::ManifestInvalid("manifest"),
        _ => CanonicalConditionV1::Unknown,
    };
    remediation_code_strings_for_condition(condition)
}

pub fn merge_canonical_remediations<I>(items: I, cap: usize) -> Vec<String>
where
    I: IntoIterator,
    I::Item: AsRef<str>,
{
    let mut out = BTreeSet::new();
    for item in items {
        for code in canonical_from_legacy_remediation(item.as_ref()) {
            out.insert(code);
        }
    }
    out.into_iter().take(cap).collect()
}

pub fn all_registry_rows() -> Vec<(&'static str, &'static str, &'static str)> {
    ucf_types::remediation_codes::REMEDIATION_REGISTRY_V1
        .iter()
        .map(|code| {
            (
                code.stable_code(),
                code.description(),
                code.suggestion_key(),
            )
        })
        .collect()
}

pub fn primary_remediation_for_condition_code(code: &str) -> Option<String> {
    canonical_condition_from_code(code)
        .and_then(|condition| remediation_for_condition(condition).first().copied())
        .map(|code| code.stable_code().to_string())
}

pub fn canonical_condition_from_code(code: &str) -> Option<CanonicalConditionV1> {
    match code {
        "ScopeMismatch" | "PolicyMismatch" | "InteropMatrixMismatch" | "AppliedScopeMismatch" => {
            Some(CanonicalConditionV1::GateFail("interop"))
        }
        "AppliedScopeMissing" => Some(CanonicalConditionV1::GateFail("scope")),
        "GovernanceEntryMissing" | "GovernanceEntryMismatch" | "CanonicalEntryRequired" => {
            Some(CanonicalConditionV1::GateFail("governance"))
        }
        "ReadinessSpineMismatch" => Some(CanonicalConditionV1::GateFail("readiness")),
        "BundleSpineMismatch" => Some(CanonicalConditionV1::GateFail("bundle")),
        "GateFailV8" => Some(CanonicalConditionV1::GateFail("v8")),
        "RequiredSurfaceMissing" => Some(CanonicalConditionV1::EvidenceMissing("compare")),
        "ManifestMismatch" | "ManifestInvalid" => {
            Some(CanonicalConditionV1::ManifestInvalid("manifest"))
        }
        "HashMismatch" => Some(CanonicalConditionV1::HashMismatch("target")),
        "EvidenceMissingProbe" => Some(CanonicalConditionV1::EvidenceMissing("probe")),
        "EvidenceMissingCompare" => Some(CanonicalConditionV1::EvidenceMissing("compare")),
        "EvidenceStaleCompare" => Some(CanonicalConditionV1::EvidenceStale("compare")),
        "DriftSevere" => Some(CanonicalConditionV1::DriftSevere("slot")),
        "StrictFail" => Some(CanonicalConditionV1::StrictFail("strict")),
        "OptionalBackendClosedUnsupported" => Some(CanonicalConditionV1::OptionalBackendMissing {
            slot: "world",
            backend: "burn",
        }),
        "ExportLayoutMismatch" | "ExportRoundTripMismatch" => {
            Some(CanonicalConditionV1::GateFail("export"))
        }
        _ => None,
    }
}

pub fn canonical_condition_for_interop_category(category: &str) -> Option<&'static str> {
    match category {
        "ScopeMismatch" => Some("ScopeMismatch"),
        "PolicyMismatch" => Some("PolicyMismatch"),
        "ManifestMismatch" => Some("ManifestMismatch"),
        "InteropMatrixMismatch" => Some("InteropMatrixMismatch"),
        "SnapshotReferenceMismatch" | "RemediationMismatch" | "ExportRefMismatch" => {
            Some("InteropMatrixMismatch")
        }
        "LegacySurfacePresent" => Some("ExportLayoutMismatch"),
        "RequiredSurfaceMissing" => Some("EvidenceMissingCompare"),
        _ => None,
    }
}

pub fn canonical_condition_for_export_normalize_category(category: &str) -> Option<&'static str> {
    match category {
        "ManifestMismatch" => Some("ManifestMismatch"),
        "ExportRoundTripMismatch" => Some("ExportRoundTripMismatch"),
        "ExportLayoutMismatch" => Some("ExportLayoutMismatch"),
        "PathNamingDrift" | "ContextFieldDrift" | "DigestFieldDrift" => Some("ManifestMismatch"),
        "IncludedStateDrift" => Some("ExportRoundTripMismatch"),
        "LegacyExportLayout" => Some("ExportLayoutMismatch"),
        _ => None,
    }
}

pub fn canonical_condition_for_roundtrip_mismatch(code: &str) -> Option<&'static str> {
    match code {
        "ScopeMismatch" => Some("ScopeMismatch"),
        "PolicyMismatch" => Some("PolicyMismatch"),
        "ManifestMismatch" => Some("ManifestMismatch"),
        "ExportRoundTripMismatch" => Some("ExportRoundTripMismatch"),
        "ExportLayoutMismatch" => Some("ExportLayoutMismatch"),
        "BUNDLE_SCOPE_MISMATCH" => Some("ScopeMismatch"),
        "BUNDLE_POLICY_MISMATCH" => Some("PolicyMismatch"),
        "BUNDLE_MANIFEST_MISMATCH" => Some("ManifestMismatch"),
        "BUNDLE_ARTIFACT_REF_MISMATCH" | "BUNDLE_INCLUDED_STATE_MISMATCH" => {
            Some("ExportRoundTripMismatch")
        }
        "LEGACY_BUNDLE_LAYOUT" | "LEGACY_BUNDLE_TRANSLATED" | "LEGACY_BUNDLE_UNSUPPORTED" => {
            Some("ExportLayoutMismatch")
        }
        _ => None,
    }
}

pub fn canonical_condition_for_scope_authority_mismatch(category: &str) -> Option<&'static str> {
    match category {
        "SurfaceDidNotUseAppliedScope" | "ExtraSlotFromLegacyInference" => {
            Some("AppliedScopeMismatch")
        }
        "MissingInScopeSlot" => Some("AppliedScopeMissing"),
        "LegacyScopePathPresent" => Some("AppliedScopeMismatch"),
        _ => None,
    }
}

pub fn canonical_condition_for_governance_entry_mismatch(category: &str) -> Option<&'static str> {
    match category {
        "CanonicalEntryRequired" => Some("CanonicalEntryRequired"),
        "ConsumerSkippedCanonicalEntry" | "ConsumerUsedSecondaryEntry" => {
            Some("GovernanceEntryMissing")
        }
        "GovernanceEntryScopeMismatch" | "GovernanceEntryPrimarySurfacesMismatch" => {
            Some("GovernanceEntryMismatch")
        }
        "LegacyEntryPathPresent" => Some("GovernanceEntryMismatch"),
        _ => None,
    }
}

pub fn canonical_condition_for_readiness_spine_mismatch(category: &str) -> Option<&'static str> {
    match category {
        "SlotTruthMismatch"
        | "ReductionMismatch"
        | "SignoffSpineDrift"
        | "ReviewPacketSpineDrift"
        | "WorkflowSpineDrift" => Some("ReadinessSpineMismatch"),
        "AppliedScopeSpineMismatch" => Some("AppliedScopeMismatch"),
        "LegacyReadinessField" | "LegacyReadinessTranslated" | "LegacyReadinessRejected" => {
            Some("ReadinessSpineMismatch")
        }
        _ => None,
    }
}

pub fn canonical_condition_for_bundle_spine_mismatch(code: &str) -> Option<&'static str> {
    match code {
        "BUNDLE_SPINE_SCOPE_MISMATCH" => Some("AppliedScopeMismatch"),
        "BUNDLE_SPINE_GOVERNANCE_MISMATCH" => Some("GovernanceEntryMismatch"),
        "BUNDLE_SPINE_READINESS_MISMATCH" => Some("ReadinessSpineMismatch"),
        "BUNDLE_SPINE_ARTIFACT_REF_MISMATCH" | "BUNDLE_SPINE_INCLUDED_STATE_MISMATCH" => {
            Some("BundleSpineMismatch")
        }
        "LEGACY_BUNDLE_SPINE_TRANSLATED" | "LEGACY_BUNDLE_SPINE_UNSUPPORTED" => {
            Some("BundleSpineMismatch")
        }
        _ => None,
    }
}

pub fn canonical_condition_for_operator_export_chain_mismatch(
    category: &str,
) -> Option<&'static str> {
    match category {
        "ReviewPacketScopeMismatch"
        | "SignoffScopeMismatch"
        | "WorkflowScopeMismatch"
        | "ExportContextScopeMismatch" => Some("AppliedScopeMismatch"),
        "ReviewabilityBasisMismatch" => Some("InteropMatrixMismatch"),
        "AppliedScopeMissing" => Some("AppliedScopeMissing"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_roundtrip_codes_map_to_canonical_conditions() {
        assert_eq!(
            canonical_condition_for_roundtrip_mismatch("BUNDLE_SCOPE_MISMATCH"),
            Some("ScopeMismatch")
        );
        assert_eq!(
            canonical_condition_for_roundtrip_mismatch("BUNDLE_ARTIFACT_REF_MISMATCH"),
            Some("ExportRoundTripMismatch")
        );
    }

    #[test]
    fn interop_category_maps_to_known_primary_remediation() {
        let condition = canonical_condition_for_interop_category("ScopeMismatch").unwrap();
        let remediation = primary_remediation_for_condition_code(condition);
        assert_eq!(
            remediation,
            Some("REMEDIATION_REGENERATE_OPERATOR_REPORT".to_string())
        );
    }
}
