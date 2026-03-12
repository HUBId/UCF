use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use ucf_types::remediation_codes::{remediation_for_condition, CanonicalConditionV1};

use crate::remediation::{canonical_from_legacy_code, canonical_from_legacy_remediation};
use crate::OpsError;
use std::fs;

const SCHEMA_VERSION: u16 = 1;
const SURFACE_ORDER: [&str; 6] = [
    "strict_check",
    "eligibility",
    "operator_report",
    "operator_signoff",
    "gate_v4",
    "export_manifest",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalRemediationObservationV1 {
    pub primary_remediation_code: Option<String>,
    pub secondary_codes: Vec<String>,
    pub source_surface: String,
    pub derived_from_condition_code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RemediationConsistencyStatusV1 {
    Pass,
    Fail,
    Skip,
    Missing,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RemediationMismatchKindV1 {
    MissingSurface,
    DifferentPrimaryCode,
    UnknownConditionMapping,
    LegacyTranslationDrift,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemediationConsistencyObservedV1 {
    pub strict_check_primary: Option<String>,
    pub eligibility_primary: Option<String>,
    pub operator_report_primary: Option<String>,
    pub operator_signoff_primary: Option<String>,
    pub gate_primary: Vec<String>,
    pub export_manifest_primary: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemediationConsistencyCheckV1 {
    pub schema_version: u16,
    pub canonical_condition_code: String,
    pub surfaces_checked: Vec<String>,
    pub expected_primary_remediation_code: Option<String>,
    pub observed: RemediationConsistencyObservedV1,
    pub status: RemediationConsistencyStatusV1,
    pub mismatch_kind: Option<RemediationMismatchKindV1>,
    pub remediation_consistency_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemediationConsistencySummaryV1 {
    pub schema_version: u16,
    pub total_conditions: usize,
    pub pass_count: usize,
    pub fail_count: usize,
    pub skip_count: usize,
    pub missing_count: usize,
    pub top_mismatch_categories: Vec<String>,
    pub status: RemediationConsistencyStatusV1,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemediationConsistencyReportV1 {
    pub schema_version: u16,
    pub checks: Vec<RemediationConsistencyCheckV1>,
    pub summary: RemediationConsistencySummaryV1,
    pub suggestions: Vec<String>,
}

#[derive(Clone)]
struct CoveredCondition {
    code: &'static str,
    condition: CanonicalConditionV1,
}

#[derive(Clone)]
enum SurfaceSignal {
    LegacyCode(&'static str),
    LegacyRemediation(&'static str),
    Skip,
    Missing,
}

pub fn remediation_consistency_check(
    out: &Path,
) -> Result<RemediationConsistencyReportV1, OpsError> {
    let checks: Vec<RemediationConsistencyCheckV1> = covered_conditions()
        .into_iter()
        .map(build_condition_check)
        .collect();

    let fail_count = checks
        .iter()
        .filter(|c| matches!(c.status, RemediationConsistencyStatusV1::Fail))
        .count();
    let pass_count = checks
        .iter()
        .filter(|c| matches!(c.status, RemediationConsistencyStatusV1::Pass))
        .count();
    let skip_count = checks
        .iter()
        .filter(|c| matches!(c.status, RemediationConsistencyStatusV1::Skip))
        .count();
    let missing_count = checks
        .iter()
        .filter(|c| matches!(c.status, RemediationConsistencyStatusV1::Missing))
        .count();

    let mut mismatch_hist = BTreeMap::<String, usize>::new();
    for check in &checks {
        if let Some(kind) = check.mismatch_kind.as_ref() {
            *mismatch_hist.entry(format!("{kind:?}")).or_default() += 1;
        }
    }

    let summary = RemediationConsistencySummaryV1 {
        schema_version: SCHEMA_VERSION,
        total_conditions: checks.len(),
        pass_count,
        fail_count,
        skip_count,
        missing_count,
        top_mismatch_categories: mismatch_hist
            .into_iter()
            .map(|(k, v)| format!("{k}:{v}"))
            .collect(),
        status: if fail_count == 0 {
            RemediationConsistencyStatusV1::Pass
        } else {
            RemediationConsistencyStatusV1::Fail
        },
    };

    let report = RemediationConsistencyReportV1 {
        schema_version: SCHEMA_VERSION,
        checks,
        summary,
        suggestions: vec![
            "cargo run -p ucf-ops -- docs remediation-codes --out docs/remediation_codes_v1.md"
                .to_string(),
            "refactor drifting surface to use canonical remediation registry mapping directly"
                .to_string(),
            "update legacy translation layer in runtime/ucf-ops/src/remediation.rs".to_string(),
        ],
    };

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_string_pretty(&report)?)?;
    Ok(report)
}

fn covered_conditions() -> Vec<CoveredCondition> {
    let mut out = vec![
        CoveredCondition {
            code: "ActiveUnsupported",
            condition: CanonicalConditionV1::ActiveUnsupported("slot"),
        },
        CoveredCondition {
            code: "DriftSevere",
            condition: CanonicalConditionV1::DriftSevere("slot"),
        },
        CoveredCondition {
            code: "EvidenceMissingCompare",
            condition: CanonicalConditionV1::EvidenceMissing("compare"),
        },
        CoveredCondition {
            code: "EvidenceMissingProbe",
            condition: CanonicalConditionV1::EvidenceMissing("probe"),
        },
        CoveredCondition {
            code: "EvidenceStaleCompare",
            condition: CanonicalConditionV1::EvidenceStale("compare"),
        },
        CoveredCondition {
            code: "EvidenceStaleProbe",
            condition: CanonicalConditionV1::EvidenceStale("probe"),
        },
        CoveredCondition {
            code: "GateFailV4",
            condition: CanonicalConditionV1::GateFail("v4"),
        },
        CoveredCondition {
            code: "HashMismatch",
            condition: CanonicalConditionV1::HashMismatch("target"),
        },
        CoveredCondition {
            code: "ManifestInvalid",
            condition: CanonicalConditionV1::ManifestInvalid("manifest"),
        },
        CoveredCondition {
            code: "OptionalBackendClosedUnsupported",
            condition: CanonicalConditionV1::OptionalBackendMissing {
                slot: "world",
                backend: "burn",
            },
        },
        CoveredCondition {
            code: "StrictFail",
            condition: CanonicalConditionV1::StrictFail("strict"),
        },
    ];
    out.sort_by(|a, b| a.code.cmp(b.code));
    out
}

fn build_condition_check(entry: CoveredCondition) -> RemediationConsistencyCheckV1 {
    let expected_codes = remediation_for_condition(entry.condition.clone())
        .into_iter()
        .map(|code| code.stable_code().to_string())
        .collect::<Vec<_>>();
    let expected_primary = expected_codes.first().cloned();

    let observed_map = normalized_surface_map(entry.code);
    let observed = RemediationConsistencyObservedV1 {
        strict_check_primary: observed_map
            .get("strict_check")
            .and_then(|v| v.primary_remediation_code.clone()),
        eligibility_primary: observed_map
            .get("eligibility")
            .and_then(|v| v.primary_remediation_code.clone()),
        operator_report_primary: observed_map
            .get("operator_report")
            .and_then(|v| v.primary_remediation_code.clone()),
        operator_signoff_primary: observed_map
            .get("operator_signoff")
            .and_then(|v| v.primary_remediation_code.clone()),
        gate_primary: observed_map
            .iter()
            .filter(|(k, _)| k.starts_with("gate_"))
            .filter_map(|(_, v)| v.primary_remediation_code.clone())
            .collect(),
        export_manifest_primary: observed_map
            .get("export_manifest")
            .and_then(|v| v.primary_remediation_code.clone()),
    };

    let status = classify_status(&expected_primary, &observed_map);
    let mismatch_kind = classify_mismatch(&status, &expected_primary, &observed_map);
    let digest = digest_check(
        entry.code,
        &expected_primary,
        &observed_map,
        &status,
        mismatch_kind.as_ref(),
    );

    RemediationConsistencyCheckV1 {
        schema_version: SCHEMA_VERSION,
        canonical_condition_code: entry.code.to_string(),
        surfaces_checked: SURFACE_ORDER.iter().map(|s| s.to_string()).collect(),
        expected_primary_remediation_code: expected_primary,
        observed,
        status,
        mismatch_kind,
        remediation_consistency_digest: digest,
    }
}

fn normalized_surface_map(
    condition_code: &str,
) -> BTreeMap<String, CanonicalRemediationObservationV1> {
    SURFACE_ORDER
        .iter()
        .map(|surface| {
            let signal = signal_for_surface_condition(surface, condition_code);
            let obs = normalize_surface_remediation((*surface).to_string(), condition_code, signal);
            ((*surface).to_string(), obs)
        })
        .collect()
}

fn signal_for_surface_condition(surface: &str, condition_code: &str) -> SurfaceSignal {
    match (surface, condition_code) {
        ("strict_check", "StrictFail") => SurfaceSignal::LegacyCode("STRICT_FAIL"),
        ("eligibility", "EvidenceMissingProbe") => SurfaceSignal::LegacyCode("NO_PROBE"),
        ("eligibility", "EvidenceStaleProbe") => SurfaceSignal::LegacyCode("STALE_PROBE"),
        ("eligibility", "EvidenceMissingCompare") => SurfaceSignal::LegacyCode("NO_COMPARE"),
        ("eligibility", "EvidenceStaleCompare") => SurfaceSignal::LegacyCode("STALE_COMPARE"),
        ("eligibility", "HashMismatch") => SurfaceSignal::LegacyCode("TARGET_HASH_MISMATCH"),
        ("eligibility", "ActiveUnsupported") => {
            SurfaceSignal::LegacyRemediation("run_models_active_check")
        }
        ("eligibility", "OptionalBackendClosedUnsupported") => SurfaceSignal::Skip,
        ("eligibility", "ManifestInvalid") => SurfaceSignal::LegacyCode("MANIFEST_INVALID"),
        ("operator_report", "StrictFail") => SurfaceSignal::LegacyRemediation("run_strict_check"),
        ("operator_report", "DriftSevere") => SurfaceSignal::LegacyCode("DRIFT_SEVERE"),
        ("operator_report", "EvidenceMissingProbe") => {
            SurfaceSignal::LegacyRemediation("run_probe")
        }
        ("operator_report", "EvidenceMissingCompare") => {
            SurfaceSignal::LegacyCode("NO_COMPARE_EVIDENCE")
        }
        ("operator_report", "EvidenceStaleProbe") => {
            SurfaceSignal::LegacyCode("STALE_PROBE_EVIDENCE")
        }
        ("operator_report", "EvidenceStaleCompare") => {
            SurfaceSignal::LegacyCode("STALE_COMPARE_EVIDENCE")
        }
        ("operator_report", "HashMismatch") => SurfaceSignal::LegacyCode("HASH_MISMATCH"),
        ("operator_report", "GateFailV4") => SurfaceSignal::LegacyRemediation("run_v3_gate"),
        ("operator_report", "ActiveUnsupported") => {
            SurfaceSignal::LegacyRemediation("run_models_active_check")
        }
        ("operator_report", "OptionalBackendClosedUnsupported") => SurfaceSignal::Skip,
        ("operator_report", "ManifestInvalid") => SurfaceSignal::LegacyCode("MANIFEST_INVALID"),
        ("operator_signoff", "StrictFail") => SurfaceSignal::LegacyRemediation("run_strict_check"),
        ("operator_signoff", "GateFailV4") => SurfaceSignal::LegacyRemediation("run_v3_gate"),
        ("operator_signoff", "EvidenceMissingProbe") => {
            SurfaceSignal::LegacyRemediation("run_backend_evidence_snapshot")
        }
        ("operator_signoff", "DriftSevere") => SurfaceSignal::LegacyRemediation("run_drift_report"),
        ("operator_signoff", "HashMismatch") => SurfaceSignal::LegacyCode("TARGET_HASH_MISMATCH"),
        ("operator_signoff", "ManifestInvalid") => SurfaceSignal::LegacyCode("MANIFEST_INVALID"),
        ("operator_signoff", "ActiveUnsupported") => {
            SurfaceSignal::LegacyRemediation("run_models_active_check")
        }
        ("operator_signoff", "OptionalBackendClosedUnsupported") => SurfaceSignal::Skip,
        ("operator_signoff", _) => SurfaceSignal::Missing,
        ("gate_v4", "GateFailV4") => SurfaceSignal::LegacyRemediation("run_v3_gate"),
        ("gate_v4", _) => SurfaceSignal::Skip,
        ("export_manifest", "EvidenceMissingProbe") => {
            SurfaceSignal::LegacyCode("NO_PROBE_EVIDENCE")
        }
        ("export_manifest", "EvidenceMissingCompare") => {
            SurfaceSignal::LegacyCode("NO_COMPARE_EVIDENCE")
        }
        ("export_manifest", "HashMismatch") => SurfaceSignal::LegacyCode("TARGET_HASH_MISMATCH"),
        ("export_manifest", "OptionalBackendClosedUnsupported") => {
            SurfaceSignal::LegacyCode("OPTIONAL_BACKEND_CLOSED_UNSUPPORTED")
        }
        ("export_manifest", "ManifestInvalid") => SurfaceSignal::LegacyCode("MANIFEST_INVALID"),
        ("export_manifest", _) => SurfaceSignal::Skip,
        ("strict_check", _) => SurfaceSignal::Skip,
        _ => SurfaceSignal::Missing,
    }
}

fn normalize_surface_remediation(
    source_surface: String,
    condition_code: &str,
    signal: SurfaceSignal,
) -> CanonicalRemediationObservationV1 {
    let canonical_codes = match signal {
        SurfaceSignal::LegacyCode(code) => canonical_from_legacy_code(code),
        SurfaceSignal::LegacyRemediation(code) => canonical_from_legacy_remediation(code),
        SurfaceSignal::Skip | SurfaceSignal::Missing => Vec::new(),
    };
    let primary = canonical_codes.first().cloned();
    let secondary = canonical_codes.into_iter().skip(1).take(3).collect();
    CanonicalRemediationObservationV1 {
        primary_remediation_code: primary,
        secondary_codes: secondary,
        source_surface,
        derived_from_condition_code: condition_code.to_string(),
    }
}

fn classify_status(
    expected_primary: &Option<String>,
    observed_map: &BTreeMap<String, CanonicalRemediationObservationV1>,
) -> RemediationConsistencyStatusV1 {
    if expected_primary.is_none() {
        return RemediationConsistencyStatusV1::Missing;
    }

    let mut saw_missing = false;
    let mut saw_supported = false;
    for (surface, obs) in observed_map {
        if surface.starts_with("gate_") && obs.primary_remediation_code.is_none() {
            continue;
        }
        if obs.primary_remediation_code.is_none() {
            if is_explicit_missing(surface, &obs.derived_from_condition_code) {
                saw_missing = true;
            }
            continue;
        }
        saw_supported = true;
        if obs.primary_remediation_code != *expected_primary {
            return RemediationConsistencyStatusV1::Fail;
        }
    }
    if saw_missing {
        RemediationConsistencyStatusV1::Missing
    } else if saw_supported {
        RemediationConsistencyStatusV1::Pass
    } else {
        RemediationConsistencyStatusV1::Skip
    }
}

fn is_explicit_missing(surface: &str, condition_code: &str) -> bool {
    matches!(
        (surface, condition_code),
        ("operator_signoff", "EvidenceMissingCompare")
            | ("operator_signoff", "EvidenceStaleProbe")
            | ("operator_signoff", "EvidenceStaleCompare")
    )
}

fn classify_mismatch(
    status: &RemediationConsistencyStatusV1,
    expected_primary: &Option<String>,
    observed_map: &BTreeMap<String, CanonicalRemediationObservationV1>,
) -> Option<RemediationMismatchKindV1> {
    match status {
        RemediationConsistencyStatusV1::Pass | RemediationConsistencyStatusV1::Skip => None,
        RemediationConsistencyStatusV1::Missing => Some(RemediationMismatchKindV1::MissingSurface),
        RemediationConsistencyStatusV1::Fail => {
            if expected_primary.is_none() {
                return Some(RemediationMismatchKindV1::UnknownConditionMapping);
            }
            let unknown_found = observed_map.values().any(|obs| {
                obs.primary_remediation_code.as_deref()
                    == Some("REMEDIATION_REVIEW_REPORT_MANUALLY")
            });
            if unknown_found {
                Some(RemediationMismatchKindV1::LegacyTranslationDrift)
            } else {
                Some(RemediationMismatchKindV1::DifferentPrimaryCode)
            }
        }
    }
}

fn digest_check(
    condition_code: &str,
    expected_primary: &Option<String>,
    observed_map: &BTreeMap<String, CanonicalRemediationObservationV1>,
    status: &RemediationConsistencyStatusV1,
    mismatch_kind: Option<&RemediationMismatchKindV1>,
) -> String {
    let payload = serde_json::json!({
        "condition": condition_code,
        "expected": expected_primary,
        "observed": observed_map,
        "status": status,
        "mismatch_kind": mismatch_kind,
    });
    let mut hasher = Sha256::new();
    hasher.update(serde_json::to_vec(&payload).unwrap_or_default());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    #[test]
    fn covered_conditions_are_sorted() {
        let codes: Vec<&str> = covered_conditions().iter().map(|c| c.code).collect();
        let mut sorted = codes.clone();
        sorted.sort();
        assert_eq!(codes, sorted);
    }

    #[test]
    fn strict_fail_has_aligned_primary_remediation() {
        let check = build_condition_check(CoveredCondition {
            code: "StrictFail",
            condition: CanonicalConditionV1::StrictFail("strict"),
        });
        assert_eq!(
            check.expected_primary_remediation_code,
            Some("REMEDIATION_CHECK_STRICT_REPORT".to_string())
        );
        assert_eq!(
            check.observed.strict_check_primary,
            Some("REMEDIATION_CHECK_STRICT_REPORT".to_string())
        );
        assert_eq!(
            check.observed.operator_report_primary,
            Some("REMEDIATION_CHECK_STRICT_REPORT".to_string())
        );
        assert!(matches!(
            check.status,
            RemediationConsistencyStatusV1::Pass | RemediationConsistencyStatusV1::Missing
        ));
    }

    #[test]
    fn manifest_invalid_maps_consistently() {
        let check = build_condition_check(CoveredCondition {
            code: "ManifestInvalid",
            condition: CanonicalConditionV1::ManifestInvalid("manifest"),
        });
        assert_eq!(
            check.expected_primary_remediation_code,
            Some("REMEDIATION_VERIFY_MANIFEST".to_string())
        );
        assert!(matches!(check.status, RemediationConsistencyStatusV1::Pass));
        assert_eq!(check.mismatch_kind, None);
    }

    #[test]
    fn injected_legacy_drift_is_classified_as_fail() {
        let expected = Some("REMEDIATION_VERIFY_MANIFEST".to_string());
        let mut observed = BTreeMap::<String, CanonicalRemediationObservationV1>::new();
        observed.insert(
            "eligibility".to_string(),
            CanonicalRemediationObservationV1 {
                primary_remediation_code: Some("REMEDIATION_REVIEW_REPORT_MANUALLY".to_string()),
                secondary_codes: vec![],
                source_surface: "eligibility".to_string(),
                derived_from_condition_code: "ManifestInvalid".to_string(),
            },
        );
        let status = classify_status(&expected, &observed);
        assert!(matches!(status, RemediationConsistencyStatusV1::Fail));
        assert_eq!(
            classify_mismatch(&status, &expected, &observed),
            Some(RemediationMismatchKindV1::LegacyTranslationDrift)
        );
    }

    #[test]
    fn missing_surface_is_explicit_and_non_panicking() {
        let check = build_condition_check(CoveredCondition {
            code: "EvidenceStaleCompare",
            condition: CanonicalConditionV1::EvidenceStale("compare"),
        });
        assert!(matches!(
            check.status,
            RemediationConsistencyStatusV1::Missing
        ));
        assert_eq!(
            check.mismatch_kind,
            Some(RemediationMismatchKindV1::MissingSurface)
        );
    }
}
