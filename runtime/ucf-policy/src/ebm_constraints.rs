use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EbmConstraintsBundle {
    pub schema_version: u16,
    pub terms: Vec<EbmConstraintTerm>,
    pub constraints_digest: [u8; 32],
    pub fallback_used: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EbmConstraintTerm {
    pub id: u16,
    pub kind: String,
    pub weight_q: u16,
    pub capability_class_id: Option<u8>,
    pub threshold_q: Option<u16>,
    pub candidate_kind: Option<String>,
    pub governor_tier_min: Option<u8>,
}

#[derive(Debug, thiserror::Error)]
pub enum EbmConstraintError {
    #[error("missing ebm constraint file: {0}")]
    Missing(String),
    #[error("invalid ebm constraints: {0}")]
    Invalid(String),
}

#[derive(Debug, Deserialize)]
struct ConstraintsFile {
    schema_version: u16,
    terms: Vec<RawTerm>,
}

#[derive(Debug, Deserialize)]
struct RawTerm {
    id: u16,
    kind: String,
    weight_q: u16,
    capability_class_id: Option<u8>,
    threshold_q: Option<u16>,
    candidate_kind: Option<String>,
    governor_tier_min: Option<u8>,
}

pub fn load_ebm_constraints(root: &Path) -> Result<EbmConstraintsBundle, EbmConstraintError> {
    let path = root.join("bundle_v1/ebm_constraints.toml");
    let text = fs::read_to_string(&path)
        .map_err(|_| EbmConstraintError::Missing(path.to_string_lossy().to_string()))?;
    let raw: ConstraintsFile =
        toml::from_str(&text).map_err(|e| EbmConstraintError::Invalid(e.to_string()))?;
    validate(&raw)?;
    let terms = raw
        .terms
        .into_iter()
        .map(|t| EbmConstraintTerm {
            id: t.id,
            kind: t.kind,
            weight_q: t.weight_q,
            capability_class_id: t.capability_class_id,
            threshold_q: t.threshold_q,
            candidate_kind: t.candidate_kind,
            governor_tier_min: t.governor_tier_min,
        })
        .collect::<Vec<_>>();
    let constraints_digest = digest(raw.schema_version, &terms);
    Ok(EbmConstraintsBundle {
        schema_version: raw.schema_version,
        terms,
        constraints_digest,
        fallback_used: false,
    })
}

fn validate(raw: &ConstraintsFile) -> Result<(), EbmConstraintError> {
    if raw.terms.len() > 32 {
        return Err(EbmConstraintError::Invalid("term count > 32".to_string()));
    }
    let mut ids = raw.terms.iter().map(|t| t.id).collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    if ids.len() != raw.terms.len() {
        return Err(EbmConstraintError::Invalid("duplicate term id".to_string()));
    }
    Ok(())
}

fn digest(schema_version: u16, terms: &[EbmConstraintTerm]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"ucf.policy.ebm_constraints.v1");
    hasher.update(schema_version.to_le_bytes());
    for t in terms {
        hasher.update(t.id.to_le_bytes());
        hasher.update(t.kind.as_bytes());
        hasher.update([0]);
        hasher.update(t.weight_q.to_le_bytes());
        hasher.update([t.capability_class_id.unwrap_or(0)]);
        hasher.update(t.threshold_q.unwrap_or(0).to_le_bytes());
        hasher.update(t.candidate_kind.as_deref().unwrap_or("").as_bytes());
        hasher.update([0]);
        hasher.update([t.governor_tier_min.unwrap_or(0)]);
    }
    hasher.finalize().into()
}
