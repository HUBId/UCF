#![forbid(unsafe_code)]

use std::sync::Arc;

use blake3::Hasher;
use ucf_types::{Digest32, SymbolicClaims};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NsrReport {
    pub ok: bool,
    pub violations: Vec<String>,
}

pub trait NsrBackend {
    fn check(&self, claims: &SymbolicClaims) -> NsrReport;
}

#[derive(Clone)]
pub struct NsrPort {
    backend: Arc<dyn NsrBackend + Send + Sync>,
}

impl NsrPort {
    pub fn new(backend: Arc<dyn NsrBackend + Send + Sync>) -> Self {
        Self { backend }
    }

    pub fn check(&self, claims: &SymbolicClaims) -> NsrReport {
        self.backend.check(claims)
    }
}

impl Default for NsrPort {
    fn default() -> Self {
        Self::new(Arc::new(NsrStubBackend::new()))
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
    fn check(&self, claims: &SymbolicClaims) -> NsrReport {
        let digest = digest_claims(claims);
        let violations = claims
            .claims
            .iter()
            .filter(|claim| self.matches_violation(claim.predicate.as_str()))
            .map(|claim| {
                format!(
                    "violation:{}:{}",
                    claim.predicate,
                    hex_prefix(digest.as_bytes())
                )
            })
            .collect::<Vec<_>>();

        NsrReport {
            ok: violations.is_empty(),
            violations,
        }
    }
}

fn digest_claims(claims: &SymbolicClaims) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(
        &u64::try_from(claims.claims.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    for claim in &claims.claims {
        hasher.update(claim.predicate.as_bytes());
        hasher.update(&u64::try_from(claim.args.len()).unwrap_or(0).to_be_bytes());
        for arg in &claim.args {
            hasher.update(arg.as_bytes());
        }
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn hex_prefix(bytes: &[u8; 32]) -> String {
    bytes
        .iter()
        .take(4)
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

impl NsrStubBackend {
    fn matches_violation(&self, predicate: &str) -> bool {
        self.violation_predicate.as_deref() == Some(predicate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use ucf_types::Claim;

    #[test]
    fn mock_nsr_is_deterministic() {
        let claims = SymbolicClaims::new(vec![Claim::new_from_strs("ok", vec!["a"])]);
        let port = NsrPort::default();

        let out_a = port.check(&claims);
        let out_b = port.check(&claims);

        assert_eq!(out_a, out_b);
        assert!(out_a.ok);
    }

    #[test]
    fn mock_nsr_flags_configured_predicate() {
        let claims = SymbolicClaims::new(vec![Claim::new_from_strs("deny", vec!["x"])]);
        let backend = NsrStubBackend::new().with_violation_predicate("deny");
        let port = NsrPort::new(Arc::new(backend));

        let out = port.check(&claims);

        assert!(!out.ok);
        assert_eq!(out.violations.len(), 1);
    }
}
