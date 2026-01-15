#![forbid(unsafe_code)]

use ucf_nsr_port::{NsrBackend, NsrReport};
use ucf_types::SymbolicClaims;

#[derive(Clone, Debug, Default)]
pub struct NsrDatalogBackend;

impl NsrDatalogBackend {
    pub fn new() -> Self {
        Self
    }
}

impl NsrBackend for NsrDatalogBackend {
    fn check(&self, _claims: &SymbolicClaims) -> NsrReport {
        NsrReport {
            ok: false,
            violations: vec!["backend_unavailable:nsr-datalog".to_string()],
        }
    }
}
