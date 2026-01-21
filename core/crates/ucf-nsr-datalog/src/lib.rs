#![forbid(unsafe_code)]

use ucf_nsr::{compute_proof_digest, finalize_report, RuleChecker};
use ucf_nsr_port::{NsrBackend, NsrInput, NsrReport};

#[derive(Clone, Debug, Default)]
pub struct NsrDatalogBackend;

impl NsrDatalogBackend {
    pub fn new() -> Self {
        Self
    }
}

impl NsrBackend for NsrDatalogBackend {
    fn evaluate(&self, input: &NsrInput) -> NsrReport {
        let rule_checker = RuleChecker;
        let rule_result = rule_checker.evaluate(input);
        let proof_digest = compute_proof_digest(input, &rule_result.rules_fired, &[]);
        finalize_report(input, rule_result.violations, proof_digest)
    }
}
