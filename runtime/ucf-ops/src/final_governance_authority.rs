use serde::{Deserialize, Serialize};

use crate::{
    prefix_hex, AppliedSupportedSetContextV1, CanonicalGovernanceEntryAuthorityV2,
    CanonicalGovernanceEntryV1, GovernanceEntryAuthorityStatusV2, OpsError,
};

pub const FINAL_GOVERNANCE_AUTHORITY_REQUIRED: &str = "FINAL_GOVERNANCE_AUTHORITY_REQUIRED";
pub const APPLIED_SCOPE_REQUIRED: &str = "APPLIED_SCOPE_REQUIRED";
pub const CANONICAL_GOVERNANCE_ENTRY_REQUIRED: &str = "CANONICAL_GOVERNANCE_ENTRY_REQUIRED";
pub const LEGACY_GOVERNANCE_INPUT_BLOCKED: &str = "LEGACY_GOVERNANCE_INPUT_BLOCKED";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalGovernanceAuthorityContextV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_governance_authority_digest_prefix: String,
}

pub fn require_final_governance_authority(
    applied: Option<&AppliedSupportedSetContextV1>,
    entry: Option<&CanonicalGovernanceEntryV1>,
    authority: Option<&CanonicalGovernanceEntryAuthorityV2>,
) -> Result<FinalGovernanceAuthorityContextV1, OpsError> {
    let Some(applied) = applied else {
        return Err(OpsError::Invalid(APPLIED_SCOPE_REQUIRED.to_string()));
    };
    let Some(entry) = entry else {
        return Err(OpsError::Invalid(
            CANONICAL_GOVERNANCE_ENTRY_REQUIRED.to_string(),
        ));
    };
    let Some(authority) = authority else {
        return Err(OpsError::Invalid(
            FINAL_GOVERNANCE_AUTHORITY_REQUIRED.to_string(),
        ));
    };

    let expected_scope = applied.applied_set_digest_prefix.clone();
    let expected_entry = prefix_hex(&entry.authority_digest, 16);
    let expected_authority = prefix_hex(&authority.authority_digest, 16);

    if authority.applied_supported_set_digest_prefix != expected_scope
        || authority.canonical_governance_entry_digest_prefix != expected_entry
        || !matches!(
            authority.authority_status,
            GovernanceEntryAuthorityStatusV2::Pass
        )
    {
        return Err(OpsError::Invalid(
            LEGACY_GOVERNANCE_INPUT_BLOCKED.to_string(),
        ));
    }

    Ok(FinalGovernanceAuthorityContextV1 {
        applied_supported_set_digest_prefix: expected_scope,
        canonical_governance_entry_digest_prefix: expected_entry,
        canonical_governance_authority_digest_prefix: expected_authority,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CanonicalGovernanceEntryAuthorityV2, CanonicalGovernanceEntryStatusV1,
        GovernanceEntryAuthorityStatusV2,
    };

    #[test]
    fn require_final_governance_authority_rejects_missing_inputs() {
        let err = require_final_governance_authority(None, None, None).expect_err("must fail");
        assert!(err.to_string().contains(APPLIED_SCOPE_REQUIRED));
    }

    #[test]
    fn require_final_governance_authority_accepts_matching_inputs() {
        let applied = crate::AppliedSupportedSetContextV1 {
            schema_version: 1,
            applied_set_digest_prefix: "scope123456789012".to_string(),
            slots: vec!["slot_a".to_string()],
            decision: crate::SupportedRealSlotSetExecutionDecisionV2::Frozen,
            previous_set_digest_prefix: "a".repeat(16),
            policy_digest_prefix: "b".repeat(16),
            context_digest: "b".repeat(64),
            compatibility_code: None,
        };
        let entry = crate::CanonicalGovernanceEntryV1 {
            schema_version: 1,
            applied_supported_set_digest_prefix: applied.applied_set_digest_prefix.clone(),
            applied_context_digest_prefix: prefix_hex(&applied.context_digest, 16),
            governance_primary_surfaces_digest_prefix: "c".repeat(16),
            authority_digest: "d".repeat(64),
            entry_status: CanonicalGovernanceEntryStatusV1::Pass,
        };
        let authority = CanonicalGovernanceEntryAuthorityV2 {
            schema_version: 2,
            applied_supported_set_digest_prefix: applied.applied_set_digest_prefix.clone(),
            applied_context_digest_prefix: prefix_hex(&applied.context_digest, 16),
            canonical_governance_entry_digest_prefix: prefix_hex(&entry.authority_digest, 16),
            covered_surface_count: 1,
            authority_status: GovernanceEntryAuthorityStatusV2::Pass,
            authority_digest: "f".repeat(64),
        };
        let context =
            require_final_governance_authority(Some(&applied), Some(&entry), Some(&authority))
                .expect("must pass");
        assert_eq!(
            context.canonical_governance_authority_digest_prefix,
            prefix_hex(&authority.authority_digest, 16)
        );
    }
}
