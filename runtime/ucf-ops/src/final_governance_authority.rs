use serde::{Deserialize, Serialize};

use crate::{
    prefix_hex, AppliedSupportedSetContextV1, CanonicalGovernanceEntryAuthorityV2,
    CanonicalGovernanceEntryV1, FinalGovernanceConsumerAuthorityStatusV1,
    FinalGovernanceConsumerAuthorityV1, FinalGovernanceResidualSweepV1,
    GovernanceEntryAuthorityStatusV2, GovernanceResidualSweepStatusV1, OpsError,
};

pub const FINAL_GOVERNANCE_AUTHORITY_REQUIRED: &str = "FINAL_GOVERNANCE_AUTHORITY_REQUIRED";
pub const FINAL_GOVERNANCE_INPUTS_REQUIRED: &str = "FINAL_GOVERNANCE_INPUTS_REQUIRED";
pub const APPLIED_SCOPE_REQUIRED: &str = "APPLIED_SCOPE_REQUIRED";
pub const CANONICAL_GOVERNANCE_ENTRY_REQUIRED: &str = "CANONICAL_GOVERNANCE_ENTRY_REQUIRED";
pub const RESIDUAL_GOVERNANCE_PATH_BLOCKED: &str = "RESIDUAL_GOVERNANCE_PATH_BLOCKED";
pub const LEGACY_GOVERNANCE_INPUT_BLOCKED: &str = "LEGACY_GOVERNANCE_INPUT_BLOCKED";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResidualFreeFinalGovernanceInputsV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_governance_authority_digest_prefix: String,
    pub final_governance_consumer_authority_digest_prefix: String,
    pub final_governance_residual_sweep_digest_prefix: String,

    pub authority_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalGovernanceAuthorityContextV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_governance_authority_digest_prefix: String,
    pub final_governance_consumer_authority_digest_prefix: String,
}

pub fn require_final_governance_authority(
    applied: Option<&AppliedSupportedSetContextV1>,
    entry: Option<&CanonicalGovernanceEntryV1>,
    authority: Option<&CanonicalGovernanceEntryAuthorityV2>,
) -> Result<FinalGovernanceAuthorityContextV1, OpsError> {
    require_final_governance_inputs(applied, entry, authority, None)
}

pub fn require_residual_free_final_governance_inputs(
    applied: Option<&AppliedSupportedSetContextV1>,
    entry: Option<&CanonicalGovernanceEntryV1>,
    authority: Option<&CanonicalGovernanceEntryAuthorityV2>,
    final_consumer: Option<&FinalGovernanceConsumerAuthorityV1>,
    residual_sweep: Option<&FinalGovernanceResidualSweepV1>,
) -> Result<ResidualFreeFinalGovernanceInputsV1, OpsError> {
    let base = require_final_governance_inputs(applied, entry, authority, final_consumer)?;
    let Some(residual_sweep) = residual_sweep else {
        return Err(OpsError::Invalid(
            FINAL_GOVERNANCE_INPUTS_REQUIRED.to_string(),
        ));
    };
    if !matches!(
        residual_sweep.sweep_status,
        GovernanceResidualSweepStatusV1::Pass
    ) || residual_sweep.applied_supported_set_digest_prefix
        != base.applied_supported_set_digest_prefix
        || residual_sweep.canonical_governance_entry_digest_prefix
            != base.canonical_governance_entry_digest_prefix
        || residual_sweep.canonical_governance_authority_digest_prefix
            != base.canonical_governance_authority_digest_prefix
        || residual_sweep.final_governance_consumer_authority_digest_prefix
            != base.final_governance_consumer_authority_digest_prefix
    {
        return Err(OpsError::Invalid(
            RESIDUAL_GOVERNANCE_PATH_BLOCKED.to_string(),
        ));
    }

    let final_governance_residual_sweep_digest_prefix =
        prefix_hex(&residual_sweep.sweep_digest, 16);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"residual_free_final_governance_inputs_v1");
    bytes.extend_from_slice(base.applied_supported_set_digest_prefix.as_bytes());
    bytes.extend_from_slice(base.canonical_governance_entry_digest_prefix.as_bytes());
    bytes.extend_from_slice(base.canonical_governance_authority_digest_prefix.as_bytes());
    bytes.extend_from_slice(
        base.final_governance_consumer_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(final_governance_residual_sweep_digest_prefix.as_bytes());

    Ok(ResidualFreeFinalGovernanceInputsV1 {
        applied_supported_set_digest_prefix: base.applied_supported_set_digest_prefix,
        canonical_governance_entry_digest_prefix: base.canonical_governance_entry_digest_prefix,
        canonical_governance_authority_digest_prefix: base
            .canonical_governance_authority_digest_prefix,
        final_governance_consumer_authority_digest_prefix: base
            .final_governance_consumer_authority_digest_prefix,
        final_governance_residual_sweep_digest_prefix,
        authority_digest: crate::sha256_hex(&bytes),
    })
}

pub fn require_final_governance_inputs(
    applied: Option<&AppliedSupportedSetContextV1>,
    entry: Option<&CanonicalGovernanceEntryV1>,
    authority: Option<&CanonicalGovernanceEntryAuthorityV2>,
    final_consumer: Option<&FinalGovernanceConsumerAuthorityV1>,
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
    let expected_final_consumer = if let Some(final_consumer) = final_consumer {
        let expected = prefix_hex(&final_consumer.authority_digest, 16);
        if final_consumer.applied_supported_set_digest_prefix != expected_scope
            || final_consumer.canonical_governance_entry_digest_prefix != expected_entry
            || final_consumer.canonical_governance_authority_digest_prefix != expected_authority
            || !matches!(
                final_consumer.authority_status,
                FinalGovernanceConsumerAuthorityStatusV1::Pass
            )
        {
            return Err(OpsError::Invalid(
                RESIDUAL_GOVERNANCE_PATH_BLOCKED.to_string(),
            ));
        }
        expected
    } else {
        String::new()
    };

    Ok(FinalGovernanceAuthorityContextV1 {
        applied_supported_set_digest_prefix: expected_scope,
        canonical_governance_entry_digest_prefix: expected_entry,
        canonical_governance_authority_digest_prefix: expected_authority,
        final_governance_consumer_authority_digest_prefix: expected_final_consumer,
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
            covered_surface_count: 0,
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
