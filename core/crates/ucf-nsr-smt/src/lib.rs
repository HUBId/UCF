#![forbid(unsafe_code)]

use ucf_nsr::{compute_proof_digest, finalize_report, ConstraintChecker};
use ucf_nsr_port::{NsrBackend, NsrInput, NsrReport};

#[derive(Clone, Debug, Default)]
pub struct NsrSmtBackend;

impl NsrSmtBackend {
    pub fn new() -> Self {
        Self
    }
}

impl NsrBackend for NsrSmtBackend {
    fn evaluate(&self, input: &NsrInput) -> NsrReport {
        let checker = ConstraintChecker;
        let constraint_result = checker.evaluate(input);
        let proof_digest = compute_proof_digest(input, &[], &constraint_result.constraints_checked);
        finalize_report(input, constraint_result.violations, proof_digest)
    }
}
