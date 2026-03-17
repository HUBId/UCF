use serde::{Deserialize, Serialize};

use crate::{
    prefix_hex, sha256_hex, AppliedSupportedSetContextV1, GovernancePrimarySurfacesV1, OpsError,
};

pub const CANONICAL_ENTRY_REQUIRED: &str = "CANONICAL_ENTRY_REQUIRED";
pub const GOVERNANCE_PRIMARY_SURFACES_REQUIRED: &str = "GOVERNANCE_PRIMARY_SURFACES_REQUIRED";
pub const SECONDARY_ENTRY_PATH_BLOCKED: &str = "SECONDARY_ENTRY_PATH_BLOCKED";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CanonicalGovernanceEntryStatusV1 {
    Pass,
    Fail,
    Legacy,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalGovernanceEntryV1 {
    pub schema_version: u16,
    pub applied_supported_set_digest_prefix: String,
    pub applied_context_digest_prefix: String,
    pub governance_primary_surfaces_digest_prefix: String,
    pub authority_digest: String,
    pub entry_status: CanonicalGovernanceEntryStatusV1,
}

pub fn derive_canonical_governance_entry(
    applied: &AppliedSupportedSetContextV1,
    surfaces: &GovernancePrimarySurfacesV1,
) -> Result<CanonicalGovernanceEntryV1, OpsError> {
    let applied_context_digest_prefix = prefix_hex(&applied.context_digest, 16);
    let governance_primary_surfaces_digest_prefix =
        prefix_hex(&surfaces.governance_surfaces_digest, 16);
    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(b"canonical_governance_entry_v1");
    digest_source.extend_from_slice(applied.applied_set_digest_prefix.as_bytes());
    digest_source.extend_from_slice(applied_context_digest_prefix.as_bytes());
    digest_source.extend_from_slice(governance_primary_surfaces_digest_prefix.as_bytes());

    Ok(CanonicalGovernanceEntryV1 {
        schema_version: 1,
        applied_supported_set_digest_prefix: applied.applied_set_digest_prefix.clone(),
        applied_context_digest_prefix,
        governance_primary_surfaces_digest_prefix,
        authority_digest: sha256_hex(&digest_source),
        entry_status: CanonicalGovernanceEntryStatusV1::Pass,
    })
}

pub fn canonical_entry_from_optional(
    applied: Option<&AppliedSupportedSetContextV1>,
    surfaces: Option<&GovernancePrimarySurfacesV1>,
) -> Result<CanonicalGovernanceEntryV1, OpsError> {
    let Some(applied) = applied else {
        return Err(OpsError::Invalid("APPLIED_SCOPE_REQUIRED".to_string()));
    };
    let Some(surfaces) = surfaces else {
        return Err(OpsError::Invalid(
            GOVERNANCE_PRIMARY_SURFACES_REQUIRED.to_string(),
        ));
    };
    derive_canonical_governance_entry(applied, surfaces)
}
