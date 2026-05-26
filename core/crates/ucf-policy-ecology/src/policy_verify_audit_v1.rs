use blake3::Hasher;

use crate::{
    PolicyEvaluationCandidateStatusV1, PolicyEvaluationCandidateV1, PolicyFieldV1,
    PolicyViolationV1,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PolicyVerifyAuditFailureV1 {
    InvalidPolicyField,
    FieldDigestMismatch,
    CandidateNotMetadataOnly,
    CandidateHasActionAuthority,
    CandidateHasGatewayAuthority,
    CandidateHasRuntimeEnforcement,
    CandidateHasIdentityAuthority,
    CandidateHasEvidenceArchiveAuthority,
}

impl PolicyVerifyAuditFailureV1 {
    fn as_u8(self) -> u8 {
        match self {
            Self::InvalidPolicyField => 1,
            Self::FieldDigestMismatch => 2,
            Self::CandidateNotMetadataOnly => 3,
            Self::CandidateHasActionAuthority => 4,
            Self::CandidateHasGatewayAuthority => 5,
            Self::CandidateHasRuntimeEnforcement => 6,
            Self::CandidateHasIdentityAuthority => 7,
            Self::CandidateHasEvidenceArchiveAuthority => 8,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyVerifyAuditStatusV1 {
    Pass,
    Constrained,
    RejectCandidate,
    Fail,
}

impl PolicyVerifyAuditStatusV1 {
    fn as_u8(self) -> u8 {
        match self {
            Self::Pass => 1,
            Self::Constrained => 2,
            Self::RejectCandidate => 3,
            Self::Fail => 4,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyVerifyAuditV1 {
    pub status: PolicyVerifyAuditStatusV1,
    pub failures: Vec<PolicyVerifyAuditFailureV1>,
    pub field_digest: [u8; 32],
    pub candidate_digest: [u8; 32],
    pub violation_count: u32,
    pub audit_digest: [u8; 32],
    pub verify_only: bool,
    pub action_authority: bool,
    pub gateway_authority: bool,
    pub runtime_enforcement: bool,
    pub policy_mutation: bool,
    pub identity_authority: bool,
    pub evidence_archive_authority: bool,
}

impl PolicyVerifyAuditV1 {
    pub fn metadata_only(&self) -> bool {
        self.verify_only
    }

    pub fn verify_only(&self) -> bool {
        self.verify_only
    }

    pub fn is_action_approval(&self) -> bool {
        false
    }

    pub fn action_authority(&self) -> bool {
        self.action_authority
    }

    pub fn gateway_authority(&self) -> bool {
        self.gateway_authority
    }

    pub fn runtime_enforcement(&self) -> bool {
        self.runtime_enforcement
    }

    pub fn policy_mutation(&self) -> bool {
        self.policy_mutation
    }

    pub fn identity_authority(&self) -> bool {
        self.identity_authority
    }

    pub fn evidence_archive_authority(&self) -> bool {
        self.evidence_archive_authority
    }

    pub fn deterministic_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"ucf.policy_ecology.policy_verify_audit_v1\0");
        out.push(self.status.as_u8());
        write_u32(&mut out, self.failures.len() as u32);
        for failure in &self.failures {
            out.push(failure.as_u8());
        }
        out.extend_from_slice(&self.field_digest);
        out.extend_from_slice(&self.candidate_digest);
        write_u32(&mut out, self.violation_count);
        write_bool(&mut out, self.verify_only);
        write_bool(&mut out, self.action_authority);
        write_bool(&mut out, self.gateway_authority);
        write_bool(&mut out, self.runtime_enforcement);
        write_bool(&mut out, self.policy_mutation);
        write_bool(&mut out, self.identity_authority);
        write_bool(&mut out, self.evidence_archive_authority);
        out
    }

    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(&self.deterministic_bytes());
        *hasher.finalize().as_bytes()
    }
}

pub fn verify_policy_evaluation_v1(
    field: &PolicyFieldV1,
    candidate: &PolicyEvaluationCandidateV1,
) -> PolicyVerifyAuditV1 {
    let mut failures = Vec::new();

    if field.validate().is_err() {
        failures.push(PolicyVerifyAuditFailureV1::InvalidPolicyField);
    }

    let field_digest = field.digest();
    if candidate.field_digest != field_digest {
        failures.push(PolicyVerifyAuditFailureV1::FieldDigestMismatch);
    }
    if !candidate.candidate_only() {
        failures.push(PolicyVerifyAuditFailureV1::CandidateNotMetadataOnly);
    }
    if candidate.action_authority() {
        failures.push(PolicyVerifyAuditFailureV1::CandidateHasActionAuthority);
    }
    if candidate.gateway_authority() {
        failures.push(PolicyVerifyAuditFailureV1::CandidateHasGatewayAuthority);
    }
    if candidate.runtime_enforcement() {
        failures.push(PolicyVerifyAuditFailureV1::CandidateHasRuntimeEnforcement);
    }
    if candidate.identity_authority() {
        failures.push(PolicyVerifyAuditFailureV1::CandidateHasIdentityAuthority);
    }
    if candidate.evidence_archive_authority() {
        failures.push(PolicyVerifyAuditFailureV1::CandidateHasEvidenceArchiveAuthority);
    }

    failures.sort();
    failures.dedup();

    let status = if failures.is_empty() {
        mirror_candidate_status(candidate.status)
    } else {
        PolicyVerifyAuditStatusV1::Fail
    };

    let mut audit = PolicyVerifyAuditV1 {
        status,
        failures,
        field_digest,
        candidate_digest: candidate.digest(),
        violation_count: candidate.violations.len() as u32,
        audit_digest: [0u8; 32],
        verify_only: true,
        action_authority: false,
        gateway_authority: false,
        runtime_enforcement: false,
        policy_mutation: false,
        identity_authority: false,
        evidence_archive_authority: false,
    };

    audit.audit_digest = audit.digest();
    audit
}

fn mirror_candidate_status(status: PolicyEvaluationCandidateStatusV1) -> PolicyVerifyAuditStatusV1 {
    match status {
        PolicyEvaluationCandidateStatusV1::Pass => PolicyVerifyAuditStatusV1::Pass,
        PolicyEvaluationCandidateStatusV1::Constrained => PolicyVerifyAuditStatusV1::Constrained,
        PolicyEvaluationCandidateStatusV1::RejectCandidate => {
            PolicyVerifyAuditStatusV1::RejectCandidate
        }
    }
}

#[allow(dead_code)]
fn _violation_summary(violations: &[PolicyViolationV1]) -> u32 {
    violations.len() as u32
}

fn write_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_bool(out: &mut Vec<u8>, value: bool) {
    out.push(u8::from(value));
}
