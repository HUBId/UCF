#![forbid(unsafe_code)]

use ucf_nsr_port::{NsrBackend, NsrReport};
use ucf_types::SymbolicClaims;

#[derive(Clone, Debug, Default)]
pub struct NsrSmtBackend;

impl NsrSmtBackend {
    pub fn new() -> Self {
        Self
    }
}

impl NsrBackend for NsrSmtBackend {
    fn check(&self, _claims: &SymbolicClaims) -> NsrReport {
        NsrReport {
            ok: false,
            violations: vec!["backend_unavailable:nsr-smt".to_string()],
        }
    }
}
