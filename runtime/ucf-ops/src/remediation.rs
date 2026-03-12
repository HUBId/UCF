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
