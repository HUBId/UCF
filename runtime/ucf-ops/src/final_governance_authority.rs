use serde::{Deserialize, Serialize};

use crate::{
    prefix_hex, AbsoluteFinalGovernanceTerminalSweepStatusV1,
    AbsoluteFinalGovernanceTerminalSweepV1, AppliedSupportedSetContextV1,
    CanonicalGovernanceEntryAuthorityV2, CanonicalGovernanceEntryV1,
    FinalGovernanceConsumerAuthorityStatusV1, FinalGovernanceConsumerAuthorityV1,
    FinalGovernanceResidualSweepV1, GovernanceConvergenceStatusV1, GovernanceConvergenceSweepV1,
    GovernanceEntryAuthorityStatusV2, GovernanceResidualSweepStatusV1, OpsError,
    ResidualFreeGovernanceAbsoluteSweepStatusV1, ResidualFreeGovernanceAbsoluteSweepV1,
    ResidualFreeGovernanceConsumerAuthorityStatusV1, ResidualFreeGovernanceConsumerAuthorityV1,
    TerminalGovernanceUltimateSweepStatusV1, TerminalGovernanceUltimateSweepV1,
};

pub const FINAL_GOVERNANCE_AUTHORITY_REQUIRED: &str = "FINAL_GOVERNANCE_AUTHORITY_REQUIRED";
pub const FINAL_GOVERNANCE_INPUTS_REQUIRED: &str = "FINAL_GOVERNANCE_INPUTS_REQUIRED";
pub const APPLIED_SCOPE_REQUIRED: &str = "APPLIED_SCOPE_REQUIRED";
pub const CANONICAL_GOVERNANCE_ENTRY_REQUIRED: &str = "CANONICAL_GOVERNANCE_ENTRY_REQUIRED";
pub const RESIDUAL_GOVERNANCE_PATH_BLOCKED: &str = "RESIDUAL_GOVERNANCE_PATH_BLOCKED";
pub const LEGACY_GOVERNANCE_INPUT_BLOCKED: &str = "LEGACY_GOVERNANCE_INPUT_BLOCKED";
pub const RESIDUAL_FREE_FINAL_GOVERNANCE_INPUTS_REQUIRED: &str =
    "RESIDUAL_FREE_FINAL_GOVERNANCE_INPUTS_REQUIRED";
pub const HISTORICAL_GOVERNANCE_LINEAGE_BLOCKED: &str = "HISTORICAL_GOVERNANCE_LINEAGE_BLOCKED";
pub const ABSOLUTE_RESIDUAL_FREE_FINAL_GOVERNANCE_INPUTS_REQUIRED: &str =
    "ABSOLUTE_RESIDUAL_FREE_FINAL_GOVERNANCE_INPUTS_REQUIRED";
pub const GOVERNANCE_ECHO_PATH_BLOCKED: &str = "GOVERNANCE_ECHO_PATH_BLOCKED";
pub const TERMINAL_ABSOLUTE_RESIDUAL_FREE_FINAL_GOVERNANCE_INPUTS_REQUIRED: &str =
    "TERMINAL_ABSOLUTE_RESIDUAL_FREE_FINAL_GOVERNANCE_INPUTS_REQUIRED";
pub const GOVERNANCE_CACHE_PATH_BLOCKED: &str = "GOVERNANCE_CACHE_PATH_BLOCKED";
pub const GOVERNANCE_CACHE_PATH_TRANSLATED: &str = "GOVERNANCE_CACHE_PATH_TRANSLATED";
pub const GOVERNANCE_CACHE_PATH_REJECTED: &str = "GOVERNANCE_CACHE_PATH_REJECTED";
pub const ULTIMATE_TERMINAL_ABSOLUTE_GOVERNANCE_INPUTS_REQUIRED: &str =
    "ULTIMATE_TERMINAL_ABSOLUTE_GOVERNANCE_INPUTS_REQUIRED";
pub const GOVERNANCE_MEMO_PATH_BLOCKED: &str = "GOVERNANCE_MEMO_PATH_BLOCKED";
pub const GOVERNANCE_MEMO_PATH_TRANSLATED: &str = "GOVERNANCE_MEMO_PATH_TRANSLATED";
pub const GOVERNANCE_MEMO_PATH_REJECTED: &str = "GOVERNANCE_MEMO_PATH_REJECTED";
pub const CONVERGED_CANONICAL_GOVERNANCE_INPUTS_REQUIRED: &str =
    "CONVERGED_CANONICAL_GOVERNANCE_INPUTS_REQUIRED";
pub const GOVERNANCE_ADAPTER_PATH_BLOCKED: &str = "GOVERNANCE_ADAPTER_PATH_BLOCKED";
pub const GOVERNANCE_ADAPTER_PATH_TRANSLATED: &str = "GOVERNANCE_ADAPTER_PATH_TRANSLATED";
pub const GOVERNANCE_ADAPTER_PATH_REJECTED: &str = "GOVERNANCE_ADAPTER_PATH_REJECTED";

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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResidualFreeGovernanceAbsoluteInputsV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_governance_authority_digest_prefix: String,
    pub final_governance_consumer_authority_digest_prefix: String,
    pub final_governance_residual_sweep_digest_prefix: String,
    pub residual_free_governance_consumer_authority_digest_prefix: String,
    pub authority_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AbsoluteFinalGovernanceTerminalInputsV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_governance_authority_digest_prefix: String,
    pub final_governance_consumer_authority_digest_prefix: String,
    pub final_governance_residual_sweep_digest_prefix: String,
    pub residual_free_governance_consumer_authority_digest_prefix: String,
    pub residual_free_governance_absolute_sweep_digest_prefix: String,
    pub authority_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TerminalGovernanceUltimateInputsV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_governance_authority_digest_prefix: String,
    pub final_governance_consumer_authority_digest_prefix: String,
    pub final_governance_residual_sweep_digest_prefix: String,
    pub residual_free_governance_consumer_authority_digest_prefix: String,
    pub residual_free_governance_absolute_sweep_digest_prefix: String,
    pub absolute_final_governance_terminal_sweep_digest_prefix: String,
    pub authority_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GovernanceConvergenceInputsV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_governance_authority_digest_prefix: String,
    pub final_governance_consumer_authority_digest_prefix: String,
    pub final_governance_residual_sweep_digest_prefix: String,
    pub residual_free_governance_consumer_authority_digest_prefix: String,
    pub residual_free_governance_absolute_sweep_digest_prefix: String,
    pub absolute_final_governance_terminal_sweep_digest_prefix: String,
    pub terminal_governance_ultimate_sweep_digest_prefix: String,
    pub authority_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GovernanceStabilizationInputsV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_governance_authority_digest_prefix: String,
    pub final_governance_consumer_authority_digest_prefix: String,
    pub final_governance_residual_sweep_digest_prefix: String,
    pub residual_free_governance_consumer_authority_digest_prefix: String,
    pub residual_free_governance_absolute_sweep_digest_prefix: String,
    pub absolute_final_governance_terminal_sweep_digest_prefix: String,
    pub terminal_governance_ultimate_sweep_digest_prefix: String,
    pub governance_convergence_sweep_digest_prefix: String,
    pub governance_stabilization_sweep_digest_prefix: String,
    pub authority_digest: String,
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

pub fn require_residual_free_governance_absolute_inputs(
    applied: Option<&AppliedSupportedSetContextV1>,
    entry: Option<&CanonicalGovernanceEntryV1>,
    authority: Option<&CanonicalGovernanceEntryAuthorityV2>,
    final_consumer: Option<&FinalGovernanceConsumerAuthorityV1>,
    residual_sweep: Option<&FinalGovernanceResidualSweepV1>,
    residual_free_consumer: Option<&ResidualFreeGovernanceConsumerAuthorityV1>,
) -> Result<ResidualFreeGovernanceAbsoluteInputsV1, OpsError> {
    let base = require_residual_free_final_governance_inputs(
        applied,
        entry,
        authority,
        final_consumer,
        residual_sweep,
    )?;
    let Some(residual_free_consumer) = residual_free_consumer else {
        return Err(OpsError::Invalid(
            RESIDUAL_FREE_FINAL_GOVERNANCE_INPUTS_REQUIRED.to_string(),
        ));
    };
    let expected_residual_free_prefix = prefix_hex(&residual_free_consumer.authority_digest, 16);
    if !matches!(
        residual_free_consumer.authority_status,
        ResidualFreeGovernanceConsumerAuthorityStatusV1::Pass
    ) || residual_free_consumer.applied_supported_set_digest_prefix
        != base.applied_supported_set_digest_prefix
        || residual_free_consumer.canonical_governance_entry_digest_prefix
            != base.canonical_governance_entry_digest_prefix
        || residual_free_consumer.canonical_governance_authority_digest_prefix
            != base.canonical_governance_authority_digest_prefix
        || residual_free_consumer.final_governance_consumer_authority_digest_prefix
            != base.final_governance_consumer_authority_digest_prefix
        || residual_free_consumer.final_governance_residual_sweep_digest_prefix
            != base.final_governance_residual_sweep_digest_prefix
    {
        return Err(OpsError::Invalid(
            HISTORICAL_GOVERNANCE_LINEAGE_BLOCKED.to_string(),
        ));
    }

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"residual_free_governance_absolute_inputs_v1");
    bytes.extend_from_slice(base.applied_supported_set_digest_prefix.as_bytes());
    bytes.extend_from_slice(base.canonical_governance_entry_digest_prefix.as_bytes());
    bytes.extend_from_slice(base.canonical_governance_authority_digest_prefix.as_bytes());
    bytes.extend_from_slice(
        base.final_governance_consumer_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        base.final_governance_residual_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(expected_residual_free_prefix.as_bytes());

    Ok(ResidualFreeGovernanceAbsoluteInputsV1 {
        applied_supported_set_digest_prefix: base.applied_supported_set_digest_prefix,
        canonical_governance_entry_digest_prefix: base.canonical_governance_entry_digest_prefix,
        canonical_governance_authority_digest_prefix: base
            .canonical_governance_authority_digest_prefix,
        final_governance_consumer_authority_digest_prefix: base
            .final_governance_consumer_authority_digest_prefix,
        final_governance_residual_sweep_digest_prefix: base
            .final_governance_residual_sweep_digest_prefix,
        residual_free_governance_consumer_authority_digest_prefix: expected_residual_free_prefix,
        authority_digest: crate::sha256_hex(&bytes),
    })
}

pub fn require_absolute_final_governance_terminal_inputs(
    applied: Option<&AppliedSupportedSetContextV1>,
    entry: Option<&CanonicalGovernanceEntryV1>,
    authority: Option<&CanonicalGovernanceEntryAuthorityV2>,
    final_consumer: Option<&FinalGovernanceConsumerAuthorityV1>,
    residual_sweep: Option<&FinalGovernanceResidualSweepV1>,
    residual_free_consumer: Option<&ResidualFreeGovernanceConsumerAuthorityV1>,
    absolute_sweep: Option<&ResidualFreeGovernanceAbsoluteSweepV1>,
) -> Result<AbsoluteFinalGovernanceTerminalInputsV1, OpsError> {
    let base = require_residual_free_governance_absolute_inputs(
        applied,
        entry,
        authority,
        final_consumer,
        residual_sweep,
        residual_free_consumer,
    )?;
    let Some(absolute_sweep) = absolute_sweep else {
        return Err(OpsError::Invalid(
            ABSOLUTE_RESIDUAL_FREE_FINAL_GOVERNANCE_INPUTS_REQUIRED.to_string(),
        ));
    };
    let expected_absolute_prefix = prefix_hex(&absolute_sweep.sweep_digest, 16);
    if !matches!(
        absolute_sweep.sweep_status,
        ResidualFreeGovernanceAbsoluteSweepStatusV1::Pass
    ) || absolute_sweep.applied_supported_set_digest_prefix
        != base.applied_supported_set_digest_prefix
        || absolute_sweep.canonical_governance_entry_digest_prefix
            != base.canonical_governance_entry_digest_prefix
        || absolute_sweep.canonical_governance_authority_digest_prefix
            != base.canonical_governance_authority_digest_prefix
        || absolute_sweep.final_governance_consumer_authority_digest_prefix
            != base.final_governance_consumer_authority_digest_prefix
        || absolute_sweep.final_governance_residual_sweep_digest_prefix
            != base.final_governance_residual_sweep_digest_prefix
        || absolute_sweep.residual_free_governance_consumer_authority_digest_prefix
            != base.residual_free_governance_consumer_authority_digest_prefix
    {
        return Err(OpsError::Invalid(GOVERNANCE_ECHO_PATH_BLOCKED.to_string()));
    }
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"absolute_final_governance_terminal_inputs_v1");
    bytes.extend_from_slice(base.applied_supported_set_digest_prefix.as_bytes());
    bytes.extend_from_slice(base.canonical_governance_entry_digest_prefix.as_bytes());
    bytes.extend_from_slice(base.canonical_governance_authority_digest_prefix.as_bytes());
    bytes.extend_from_slice(
        base.final_governance_consumer_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        base.final_governance_residual_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        base.residual_free_governance_consumer_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(expected_absolute_prefix.as_bytes());

    Ok(AbsoluteFinalGovernanceTerminalInputsV1 {
        applied_supported_set_digest_prefix: base.applied_supported_set_digest_prefix,
        canonical_governance_entry_digest_prefix: base.canonical_governance_entry_digest_prefix,
        canonical_governance_authority_digest_prefix: base
            .canonical_governance_authority_digest_prefix,
        final_governance_consumer_authority_digest_prefix: base
            .final_governance_consumer_authority_digest_prefix,
        final_governance_residual_sweep_digest_prefix: base
            .final_governance_residual_sweep_digest_prefix,
        residual_free_governance_consumer_authority_digest_prefix: base
            .residual_free_governance_consumer_authority_digest_prefix,
        residual_free_governance_absolute_sweep_digest_prefix: expected_absolute_prefix,
        authority_digest: crate::sha256_hex(&bytes),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn require_terminal_governance_ultimate_inputs(
    applied: Option<&AppliedSupportedSetContextV1>,
    entry: Option<&CanonicalGovernanceEntryV1>,
    authority: Option<&CanonicalGovernanceEntryAuthorityV2>,
    final_consumer: Option<&FinalGovernanceConsumerAuthorityV1>,
    residual_sweep: Option<&FinalGovernanceResidualSweepV1>,
    residual_free_consumer: Option<&ResidualFreeGovernanceConsumerAuthorityV1>,
    absolute_sweep: Option<&ResidualFreeGovernanceAbsoluteSweepV1>,
    terminal_sweep: Option<&AbsoluteFinalGovernanceTerminalSweepV1>,
) -> Result<TerminalGovernanceUltimateInputsV1, OpsError> {
    let base = require_absolute_final_governance_terminal_inputs(
        applied,
        entry,
        authority,
        final_consumer,
        residual_sweep,
        residual_free_consumer,
        absolute_sweep,
    )?;
    let Some(terminal_sweep) = terminal_sweep else {
        return Err(OpsError::Invalid(
            TERMINAL_ABSOLUTE_RESIDUAL_FREE_FINAL_GOVERNANCE_INPUTS_REQUIRED.to_string(),
        ));
    };
    let terminal_prefix = prefix_hex(&terminal_sweep.sweep_digest, 16);
    if !matches!(
        terminal_sweep.sweep_status,
        AbsoluteFinalGovernanceTerminalSweepStatusV1::Pass
    ) || terminal_sweep.applied_supported_set_digest_prefix
        != base.applied_supported_set_digest_prefix
        || terminal_sweep.canonical_governance_entry_digest_prefix
            != base.canonical_governance_entry_digest_prefix
        || terminal_sweep.canonical_governance_authority_digest_prefix
            != base.canonical_governance_authority_digest_prefix
        || terminal_sweep.final_governance_consumer_authority_digest_prefix
            != base.final_governance_consumer_authority_digest_prefix
        || terminal_sweep.final_governance_residual_sweep_digest_prefix
            != base.final_governance_residual_sweep_digest_prefix
        || terminal_sweep.residual_free_governance_consumer_authority_digest_prefix
            != base.residual_free_governance_consumer_authority_digest_prefix
        || terminal_sweep.residual_free_governance_absolute_sweep_digest_prefix
            != base.residual_free_governance_absolute_sweep_digest_prefix
    {
        return Err(OpsError::Invalid(GOVERNANCE_CACHE_PATH_BLOCKED.to_string()));
    }

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"terminal_governance_ultimate_inputs_v1");
    bytes.extend_from_slice(base.applied_supported_set_digest_prefix.as_bytes());
    bytes.extend_from_slice(base.canonical_governance_entry_digest_prefix.as_bytes());
    bytes.extend_from_slice(base.canonical_governance_authority_digest_prefix.as_bytes());
    bytes.extend_from_slice(
        base.final_governance_consumer_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        base.final_governance_residual_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        base.residual_free_governance_consumer_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        base.residual_free_governance_absolute_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(terminal_prefix.as_bytes());

    Ok(TerminalGovernanceUltimateInputsV1 {
        applied_supported_set_digest_prefix: base.applied_supported_set_digest_prefix,
        canonical_governance_entry_digest_prefix: base.canonical_governance_entry_digest_prefix,
        canonical_governance_authority_digest_prefix: base
            .canonical_governance_authority_digest_prefix,
        final_governance_consumer_authority_digest_prefix: base
            .final_governance_consumer_authority_digest_prefix,
        final_governance_residual_sweep_digest_prefix: base
            .final_governance_residual_sweep_digest_prefix,
        residual_free_governance_consumer_authority_digest_prefix: base
            .residual_free_governance_consumer_authority_digest_prefix,
        residual_free_governance_absolute_sweep_digest_prefix: base
            .residual_free_governance_absolute_sweep_digest_prefix,
        absolute_final_governance_terminal_sweep_digest_prefix: terminal_prefix,
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

#[allow(clippy::too_many_arguments)]
pub fn require_governance_convergence_inputs(
    applied: Option<&AppliedSupportedSetContextV1>,
    entry: Option<&CanonicalGovernanceEntryV1>,
    authority: Option<&CanonicalGovernanceEntryAuthorityV2>,
    final_consumer: Option<&FinalGovernanceConsumerAuthorityV1>,
    residual_sweep: Option<&FinalGovernanceResidualSweepV1>,
    residual_free_consumer: Option<&ResidualFreeGovernanceConsumerAuthorityV1>,
    absolute_sweep: Option<&ResidualFreeGovernanceAbsoluteSweepV1>,
    terminal_sweep: Option<&AbsoluteFinalGovernanceTerminalSweepV1>,
    ultimate_sweep: Option<&TerminalGovernanceUltimateSweepV1>,
) -> Result<GovernanceConvergenceInputsV1, OpsError> {
    let terminal_inputs = require_terminal_governance_ultimate_inputs(
        applied,
        entry,
        authority,
        final_consumer,
        residual_sweep,
        residual_free_consumer,
        absolute_sweep,
        terminal_sweep,
    )?;
    let Some(ultimate_sweep) = ultimate_sweep else {
        return Err(OpsError::Invalid(
            ULTIMATE_TERMINAL_ABSOLUTE_GOVERNANCE_INPUTS_REQUIRED.to_string(),
        ));
    };
    let expected_ultimate_prefix = prefix_hex(&ultimate_sweep.sweep_digest, 16);
    if !matches!(
        ultimate_sweep.sweep_status,
        TerminalGovernanceUltimateSweepStatusV1::Pass
    ) || ultimate_sweep.applied_supported_set_digest_prefix
        != terminal_inputs.applied_supported_set_digest_prefix
        || ultimate_sweep.canonical_governance_entry_digest_prefix
            != terminal_inputs.canonical_governance_entry_digest_prefix
        || ultimate_sweep.canonical_governance_authority_digest_prefix
            != terminal_inputs.canonical_governance_authority_digest_prefix
        || ultimate_sweep.final_governance_consumer_authority_digest_prefix
            != terminal_inputs.final_governance_consumer_authority_digest_prefix
        || ultimate_sweep.final_governance_residual_sweep_digest_prefix
            != terminal_inputs.final_governance_residual_sweep_digest_prefix
        || ultimate_sweep.residual_free_governance_consumer_authority_digest_prefix
            != terminal_inputs.residual_free_governance_consumer_authority_digest_prefix
        || ultimate_sweep.residual_free_governance_absolute_sweep_digest_prefix
            != terminal_inputs.residual_free_governance_absolute_sweep_digest_prefix
        || ultimate_sweep.absolute_final_governance_terminal_sweep_digest_prefix
            != terminal_inputs.absolute_final_governance_terminal_sweep_digest_prefix
    {
        return Err(OpsError::Invalid(GOVERNANCE_MEMO_PATH_BLOCKED.to_string()));
    }
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"governance_convergence_inputs_v1");
    bytes.extend_from_slice(
        terminal_inputs
            .applied_supported_set_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        terminal_inputs
            .canonical_governance_entry_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        terminal_inputs
            .canonical_governance_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        terminal_inputs
            .final_governance_consumer_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        terminal_inputs
            .final_governance_residual_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        terminal_inputs
            .residual_free_governance_consumer_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        terminal_inputs
            .residual_free_governance_absolute_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        terminal_inputs
            .absolute_final_governance_terminal_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(expected_ultimate_prefix.as_bytes());

    Ok(GovernanceConvergenceInputsV1 {
        applied_supported_set_digest_prefix: terminal_inputs.applied_supported_set_digest_prefix,
        canonical_governance_entry_digest_prefix: terminal_inputs
            .canonical_governance_entry_digest_prefix,
        canonical_governance_authority_digest_prefix: terminal_inputs
            .canonical_governance_authority_digest_prefix,
        final_governance_consumer_authority_digest_prefix: terminal_inputs
            .final_governance_consumer_authority_digest_prefix,
        final_governance_residual_sweep_digest_prefix: terminal_inputs
            .final_governance_residual_sweep_digest_prefix,
        residual_free_governance_consumer_authority_digest_prefix: terminal_inputs
            .residual_free_governance_consumer_authority_digest_prefix,
        residual_free_governance_absolute_sweep_digest_prefix: terminal_inputs
            .residual_free_governance_absolute_sweep_digest_prefix,
        absolute_final_governance_terminal_sweep_digest_prefix: terminal_inputs
            .absolute_final_governance_terminal_sweep_digest_prefix,
        terminal_governance_ultimate_sweep_digest_prefix: expected_ultimate_prefix,
        authority_digest: crate::sha256_hex(&bytes),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn require_governance_stabilization_inputs(
    applied: Option<&AppliedSupportedSetContextV1>,
    entry: Option<&CanonicalGovernanceEntryV1>,
    authority: Option<&CanonicalGovernanceEntryAuthorityV2>,
    final_consumer: Option<&FinalGovernanceConsumerAuthorityV1>,
    residual_sweep: Option<&FinalGovernanceResidualSweepV1>,
    residual_free_consumer: Option<&ResidualFreeGovernanceConsumerAuthorityV1>,
    absolute_sweep: Option<&ResidualFreeGovernanceAbsoluteSweepV1>,
    terminal_sweep: Option<&AbsoluteFinalGovernanceTerminalSweepV1>,
    ultimate_sweep: Option<&TerminalGovernanceUltimateSweepV1>,
    convergence_sweep: Option<&GovernanceConvergenceSweepV1>,
) -> Result<GovernanceStabilizationInputsV1, OpsError> {
    let convergence_inputs = require_governance_convergence_inputs(
        applied,
        entry,
        authority,
        final_consumer,
        residual_sweep,
        residual_free_consumer,
        absolute_sweep,
        terminal_sweep,
        ultimate_sweep,
    )?;
    let Some(convergence_sweep) = convergence_sweep else {
        return Err(OpsError::Invalid(
            CONVERGED_CANONICAL_GOVERNANCE_INPUTS_REQUIRED.to_string(),
        ));
    };
    let expected_convergence_prefix = prefix_hex(&convergence_sweep.convergence_digest, 16);
    if !matches!(
        convergence_sweep.convergence_status,
        GovernanceConvergenceStatusV1::Pass
    ) || convergence_sweep.applied_supported_set_digest_prefix
        != convergence_inputs.applied_supported_set_digest_prefix
        || convergence_sweep.canonical_governance_entry_digest_prefix
            != convergence_inputs.canonical_governance_entry_digest_prefix
        || convergence_sweep.canonical_governance_authority_digest_prefix
            != convergence_inputs.canonical_governance_authority_digest_prefix
        || convergence_sweep.final_governance_consumer_authority_digest_prefix
            != convergence_inputs.final_governance_consumer_authority_digest_prefix
        || convergence_sweep.final_governance_residual_sweep_digest_prefix
            != convergence_inputs.final_governance_residual_sweep_digest_prefix
        || convergence_sweep.residual_free_governance_consumer_authority_digest_prefix
            != convergence_inputs.residual_free_governance_consumer_authority_digest_prefix
        || convergence_sweep.residual_free_governance_absolute_sweep_digest_prefix
            != convergence_inputs.residual_free_governance_absolute_sweep_digest_prefix
        || convergence_sweep.absolute_final_governance_terminal_sweep_digest_prefix
            != convergence_inputs.absolute_final_governance_terminal_sweep_digest_prefix
        || convergence_sweep.terminal_governance_ultimate_sweep_digest_prefix
            != convergence_inputs.terminal_governance_ultimate_sweep_digest_prefix
    {
        return Err(OpsError::Invalid(
            GOVERNANCE_ADAPTER_PATH_BLOCKED.to_string(),
        ));
    }

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"governance_stabilization_inputs_v1");
    bytes.extend_from_slice(
        convergence_inputs
            .applied_supported_set_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        convergence_inputs
            .canonical_governance_entry_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        convergence_inputs
            .canonical_governance_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        convergence_inputs
            .final_governance_consumer_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        convergence_inputs
            .final_governance_residual_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        convergence_inputs
            .residual_free_governance_consumer_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        convergence_inputs
            .residual_free_governance_absolute_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        convergence_inputs
            .absolute_final_governance_terminal_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        convergence_inputs
            .terminal_governance_ultimate_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(expected_convergence_prefix.as_bytes());

    Ok(GovernanceStabilizationInputsV1 {
        applied_supported_set_digest_prefix: convergence_inputs.applied_supported_set_digest_prefix,
        canonical_governance_entry_digest_prefix: convergence_inputs
            .canonical_governance_entry_digest_prefix,
        canonical_governance_authority_digest_prefix: convergence_inputs
            .canonical_governance_authority_digest_prefix,
        final_governance_consumer_authority_digest_prefix: convergence_inputs
            .final_governance_consumer_authority_digest_prefix,
        final_governance_residual_sweep_digest_prefix: convergence_inputs
            .final_governance_residual_sweep_digest_prefix,
        residual_free_governance_consumer_authority_digest_prefix: convergence_inputs
            .residual_free_governance_consumer_authority_digest_prefix,
        residual_free_governance_absolute_sweep_digest_prefix: convergence_inputs
            .residual_free_governance_absolute_sweep_digest_prefix,
        absolute_final_governance_terminal_sweep_digest_prefix: convergence_inputs
            .absolute_final_governance_terminal_sweep_digest_prefix,
        terminal_governance_ultimate_sweep_digest_prefix: convergence_inputs
            .terminal_governance_ultimate_sweep_digest_prefix,
        governance_convergence_sweep_digest_prefix: expected_convergence_prefix,
        governance_stabilization_sweep_digest_prefix: String::new(),
        authority_digest: crate::sha256_hex(&bytes),
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
