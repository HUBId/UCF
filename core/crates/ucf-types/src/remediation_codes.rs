#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RemediationCodeV1 {
    RunProbe,
    RerunShadowWindow,
    CheckDriftReport,
    CheckStrictReport,
    VerifyManifest,
    StayShadow,
    ReviewActiveEvidence,
    CheckPortabilityReport,
    RegenerateOperatorReport,
    ResolveHashMismatch,
    ReviewReportManually,
}

impl RemediationCodeV1 {
    pub const fn stable_code(self) -> &'static str {
        match self {
            Self::RunProbe => "REMEDIATION_RUN_PROBE",
            Self::RerunShadowWindow => "REMEDIATION_RERUN_SHADOW_WINDOW",
            Self::CheckDriftReport => "REMEDIATION_CHECK_DRIFT_REPORT",
            Self::CheckStrictReport => "REMEDIATION_CHECK_STRICT_REPORT",
            Self::VerifyManifest => "REMEDIATION_VERIFY_MANIFEST",
            Self::StayShadow => "REMEDIATION_STAY_SHADOW",
            Self::ReviewActiveEvidence => "REMEDIATION_REVIEW_ACTIVE_EVIDENCE",
            Self::CheckPortabilityReport => "REMEDIATION_CHECK_PORTABILITY_REPORT",
            Self::RegenerateOperatorReport => "REMEDIATION_REGENERATE_OPERATOR_REPORT",
            Self::ResolveHashMismatch => "REMEDIATION_RESOLVE_HASH_MISMATCH",
            Self::ReviewReportManually => "REMEDIATION_REVIEW_REPORT_MANUALLY",
        }
    }

    pub const fn description(self) -> &'static str {
        match self {
            Self::RunProbe => "Generate fresh probe evidence for the affected slot.",
            Self::RerunShadowWindow => "Regenerate compare/shadow-window evidence.",
            Self::CheckDriftReport => "Inspect bounded drift report and clear severe drift.",
            Self::CheckStrictReport => "Run strict checks and resolve failures.",
            Self::VerifyManifest => "Verify model manifest integrity and slot declarations.",
            Self::StayShadow => "Keep slot in shadow mode until evidence converges.",
            Self::ReviewActiveEvidence => "Review active-evidence eligibility for the slot.",
            Self::CheckPortabilityReport => {
                "Check portability matrix and required backend support."
            }
            Self::RegenerateOperatorReport => "Regenerate consolidated operator report artifacts.",
            Self::ResolveHashMismatch => "Resolve target/evidence hash mismatch before promotion.",
            Self::ReviewReportManually => {
                "Perform manual operator review of bounded report evidence."
            }
        }
    }

    pub const fn suggestion_key(self) -> &'static str {
        match self {
            Self::RunProbe => "run_probe",
            Self::RerunShadowWindow => "rerun_shadow_window",
            Self::CheckDriftReport => "check_drift_report",
            Self::CheckStrictReport => "check_strict_report",
            Self::VerifyManifest => "verify_manifest",
            Self::StayShadow => "stay_shadow",
            Self::ReviewActiveEvidence => "review_active_evidence",
            Self::CheckPortabilityReport => "check_portability_report",
            Self::RegenerateOperatorReport => "regenerate_operator_report",
            Self::ResolveHashMismatch => "resolve_hash_mismatch",
            Self::ReviewReportManually => "review_report_manually",
        }
    }
}

pub const REMEDIATION_REGISTRY_V1: &[RemediationCodeV1] = &[
    RemediationCodeV1::RunProbe,
    RemediationCodeV1::RerunShadowWindow,
    RemediationCodeV1::CheckDriftReport,
    RemediationCodeV1::CheckStrictReport,
    RemediationCodeV1::VerifyManifest,
    RemediationCodeV1::StayShadow,
    RemediationCodeV1::ReviewActiveEvidence,
    RemediationCodeV1::CheckPortabilityReport,
    RemediationCodeV1::RegenerateOperatorReport,
    RemediationCodeV1::ResolveHashMismatch,
    RemediationCodeV1::ReviewReportManually,
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanonicalConditionV1 {
    EvidenceMissing(&'static str),
    EvidenceStale(&'static str),
    HashMismatch(&'static str),
    StrictFail(&'static str),
    GateFail(&'static str),
    DriftSevere(&'static str),
    AlertSevere(&'static str),
    ActiveUnsupported(&'static str),
    OptionalBackendMissing {
        slot: &'static str,
        backend: &'static str,
    },
    ManifestInvalid(&'static str),
    Unknown,
}

pub fn remediation_for_condition(condition: CanonicalConditionV1) -> Vec<RemediationCodeV1> {
    let mut out = match condition {
        CanonicalConditionV1::EvidenceMissing(kind) => match kind {
            "probe" => vec![RemediationCodeV1::RunProbe],
            "compare" | "no_impact" => vec![RemediationCodeV1::RerunShadowWindow],
            "drift" => vec![RemediationCodeV1::CheckDriftReport],
            _ => vec![RemediationCodeV1::ReviewReportManually],
        },
        CanonicalConditionV1::EvidenceStale(kind) => match kind {
            "probe" => vec![RemediationCodeV1::RunProbe],
            "compare" | "no_impact" => vec![RemediationCodeV1::RerunShadowWindow],
            "drift" => vec![RemediationCodeV1::CheckDriftReport],
            _ => vec![RemediationCodeV1::ReviewReportManually],
        },
        CanonicalConditionV1::HashMismatch(_) => {
            vec![
                RemediationCodeV1::ResolveHashMismatch,
                RemediationCodeV1::VerifyManifest,
            ]
        }
        CanonicalConditionV1::StrictFail(_) => vec![RemediationCodeV1::CheckStrictReport],
        CanonicalConditionV1::GateFail(_) => vec![
            RemediationCodeV1::RegenerateOperatorReport,
            RemediationCodeV1::CheckStrictReport,
        ],
        CanonicalConditionV1::DriftSevere(_) => {
            vec![
                RemediationCodeV1::CheckDriftReport,
                RemediationCodeV1::StayShadow,
            ]
        }
        CanonicalConditionV1::AlertSevere(_) => vec![RemediationCodeV1::RegenerateOperatorReport],
        CanonicalConditionV1::ActiveUnsupported(_) => {
            vec![
                RemediationCodeV1::CheckPortabilityReport,
                RemediationCodeV1::StayShadow,
            ]
        }
        CanonicalConditionV1::OptionalBackendMissing { .. } => {
            vec![
                RemediationCodeV1::CheckPortabilityReport,
                RemediationCodeV1::StayShadow,
            ]
        }
        CanonicalConditionV1::ManifestInvalid(_) => {
            vec![RemediationCodeV1::VerifyManifest]
        }
        CanonicalConditionV1::Unknown => vec![RemediationCodeV1::ReviewReportManually],
    };
    out.truncate(3);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_is_stable_ordered() {
        let codes: Vec<&str> = REMEDIATION_REGISTRY_V1
            .iter()
            .map(|c| c.stable_code())
            .collect();
        assert_eq!(codes.first().copied(), Some("REMEDIATION_RUN_PROBE"));
        assert_eq!(
            codes.last().copied(),
            Some("REMEDIATION_REVIEW_REPORT_MANUALLY")
        );
    }

    #[test]
    fn unknown_maps_to_fallback() {
        let out = remediation_for_condition(CanonicalConditionV1::Unknown);
        assert_eq!(out, vec![RemediationCodeV1::ReviewReportManually]);
    }
}
