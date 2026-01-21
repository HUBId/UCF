#![forbid(unsafe_code)]

use std::sync::Arc;

use ucf_nsr::{compute_proof_digest, NsrEngine};
use ucf_types::Digest32;

pub use ucf_nsr::{ActionIntent, NsrInput, NsrReport, NsrVerdict, NsrViolation};

pub trait NsrBackend {
    fn evaluate(&self, input: &NsrInput) -> NsrReport;
}

#[derive(Clone)]
pub struct NsrPort {
    backend: Arc<dyn NsrBackend + Send + Sync>,
}

impl NsrPort {
    pub fn new(backend: Arc<dyn NsrBackend + Send + Sync>) -> Self {
        Self { backend }
    }

    pub fn evaluate(&self, input: &NsrInput) -> NsrReport {
        self.backend.evaluate(input)
    }

    pub fn check(&self, input: &NsrInput) -> NsrReport {
        self.evaluate(input)
    }
}

impl Default for NsrPort {
    fn default() -> Self {
        Self::new(Arc::new(NsrEngine::new()))
    }
}

#[derive(Clone, Debug)]
pub struct NsrStubBackend {
    violation_predicate: Option<String>,
}

impl NsrStubBackend {
    pub fn new() -> Self {
        Self {
            violation_predicate: None,
        }
    }

    pub fn with_violation_predicate(mut self, predicate: impl Into<String>) -> Self {
        self.violation_predicate = Some(predicate.into());
        self
    }
}

impl Default for NsrStubBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl NsrBackend for NsrStubBackend {
    fn evaluate(&self, input: &NsrInput) -> NsrReport {
        let predicate = self.violation_predicate.as_deref();
        let mut violations = Vec::new();
        let mut rules_fired = Vec::new();
        for action in &input.proposed_actions {
            if predicate == Some(action.tag.as_str()) {
                let detail = format!("action_tag={}", action.tag);
                violations.push(NsrViolation {
                    code: "NSR_STUB_MATCH".to_string(),
                    detail_digest: digest_violation_detail(&detail, input.commit),
                    severity: 9000,
                    commit: digest_violation_commit(&detail, input.commit),
                });
                rules_fired.push("NSR_STUB_MATCH".to_string());
            }
        }
        if violations.is_empty() && predicate == Some("deny") {
            let detail = "predicate=deny".to_string();
            violations.push(NsrViolation {
                code: "NSR_STUB_MATCH".to_string(),
                detail_digest: digest_violation_detail(&detail, input.commit),
                severity: 9000,
                commit: digest_violation_commit(&detail, input.commit),
            });
            rules_fired.push("NSR_STUB_MATCH".to_string());
        }
        let proof_digest = compute_proof_digest(input, &rules_fired, &[]);
        ucf_nsr::finalize_report(input, violations, proof_digest)
    }
}

impl NsrBackend for ucf_nsr::NsrEngine {
    fn evaluate(&self, input: &NsrInput) -> NsrReport {
        NsrEngine::evaluate(self, input)
    }
}

fn digest_violation_detail(detail: &str, input_commit: Digest32) -> Digest32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"ucf.nsr.stub.detail.v1");
    hasher.update(detail.as_bytes());
    hasher.update(input_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_violation_commit(detail: &str, input_commit: Digest32) -> Digest32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"ucf.nsr.stub.violation.v1");
    hasher.update(detail.as_bytes());
    hasher.update(input_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn mock_nsr_is_deterministic() {
        let claims = NsrInput::new(
            1,
            ucf_sandbox::IntentSummary::new(1, 1),
            1,
            vec![ActionIntent::new("ok")],
            Digest32::new([9u8; 32]),
        );
        let port = NsrPort::default();

        let out_a = port.check(&claims);
        let out_b = port.check(&claims);

        assert_eq!(out_a, out_b);
        assert_eq!(out_a.verdict, NsrVerdict::Ok);
    }

    #[test]
    fn mock_nsr_flags_configured_predicate() {
        let claims = NsrInput::new(
            2,
            ucf_sandbox::IntentSummary::new(1, 1),
            1,
            vec![ActionIntent::new("deny")],
            Digest32::new([2u8; 32]),
        );
        let backend = NsrStubBackend::new().with_violation_predicate("deny");
        let port = NsrPort::new(Arc::new(backend));

        let out = port.check(&claims);

        assert_eq!(out.verdict, NsrVerdict::Deny);
        assert_eq!(out.violations.len(), 1);
    }
}
