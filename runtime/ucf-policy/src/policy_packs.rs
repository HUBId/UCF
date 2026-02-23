use crate::determinism::DeterminismPolicyV1;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const MAX_GRAPH_BYTES: usize = 64 * 1024;
const MAX_RULES: usize = 512;
const MAX_TERMS: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyGraphV1 {
    pub schema_version: u16,
    pub base_name: String,
    pub base_version: String,
    pub overlay_name: Option<String>,
    pub overlay_version: Option<String>,
    pub pbm_gem_rules: Vec<PolicyRule>,
    pub nsr_rules: Vec<String>,
    pub ebm_terms: Vec<EbmTerm>,
    pub thresholds: BTreeMap<String, i64>,
    pub budgets: BTreeMap<String, i64>,
    pub allowlists: BTreeMap<String, String>,
    pub determinism: DeterminismPolicyV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct PolicyRule {
    pub id: String,
    pub channel: String,
    pub decision: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct EbmTerm {
    pub id: u16,
    pub kind: String,
    pub weight_q: u16,
    pub threshold_q: Option<u16>,
    pub governor_tier_min: Option<u8>,
    pub capability_class_id: Option<u8>,
    pub candidate_kind: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyGraphProvenanceRecord {
    pub base_pack_digest: String,
    pub overlay_pack_digest: Option<String>,
    pub policy_graph_digest: String,
    pub schema_version: u16,
    pub base_version: String,
    pub overlay_version: Option<String>,
    pub validation_ok: bool,
    pub determinism_policy_digest: String,
}

#[derive(Debug, thiserror::Error)]
pub enum PolicyPackError {
    #[error("missing file: {0}")]
    Missing(String),
    #[error("invalid manifest: {0}")]
    InvalidManifest(String),
    #[error("invalid pack: {0}")]
    InvalidPack(String),
    #[error("hash mismatch for {path}: expected {expected}, actual {actual}")]
    HashMismatch {
        path: String,
        expected: String,
        actual: String,
    },
    #[error("digest mismatch: expected {expected}, actual {actual}")]
    DigestMismatch { expected: String, actual: String },
    #[error("merge conflict: {0}")]
    MergeConflict(String),
    #[error("graph too large")]
    GraphTooLarge,
}

#[derive(Debug, Deserialize)]
struct PackManifest {
    name: String,
    version: String,
    schema_version: u16,
    files: Vec<ManifestFile>,
    pack_digest: String,
}

#[derive(Debug, Deserialize)]
struct ManifestFile {
    path: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct RuleFile {
    #[serde(default)]
    rules: Vec<PolicyRule>,
}

#[derive(Debug, Deserialize)]
struct TermFile {
    schema_version: u16,
    #[serde(default)]
    terms: Vec<EbmTerm>,
}

#[derive(Debug, Deserialize)]
struct KvFile {
    #[serde(default)]
    values: BTreeMap<String, i64>,
}

#[derive(Debug, Deserialize)]
struct AllowlistFile {
    #[serde(default)]
    values: BTreeMap<String, String>,
}

#[derive(Debug)]
struct LoadedPack {
    manifest: PackManifest,
    pbm_gem_rules: Vec<PolicyRule>,
    nsr_rules: Vec<String>,
    ebm_terms: Vec<EbmTerm>,
    thresholds: BTreeMap<String, i64>,
    budgets: BTreeMap<String, i64>,
    allowlists: BTreeMap<String, String>,
    determinism: DeterminismPolicyV1,
}

pub fn load_and_merge_policy_graph(
    base_dir: &Path,
    overlay_dir: Option<&Path>,
) -> Result<(PolicyGraphV1, PolicyGraphProvenanceRecord), PolicyPackError> {
    let base = load_pack(base_dir)?;
    let overlay = if let Some(dir) = overlay_dir {
        Some(load_pack(dir)?)
    } else {
        None
    };

    let mut rules_by_id: BTreeMap<String, PolicyRule> = base
        .pbm_gem_rules
        .iter()
        .map(|r| (r.id.clone(), r.clone()))
        .collect();
    let mut ordered_rule_ids: Vec<String> = rules_by_id.keys().cloned().collect();

    if let Some(ov) = &overlay {
        for r in &ov.pbm_gem_rules {
            if rules_by_id.contains_key(&r.id) {
                return Err(PolicyPackError::MergeConflict(format!(
                    "duplicate pbm/gem rule id {}",
                    r.id
                )));
            }
            ordered_rule_ids.push(r.id.clone());
            rules_by_id.insert(r.id.clone(), r.clone());
        }
    }

    let mut nsr = base.nsr_rules.clone();
    if let Some(ov) = &overlay {
        for line in &ov.nsr_rules {
            if !nsr.contains(line) {
                nsr.push(line.clone());
            }
        }
    }

    let mut terms_by_id: BTreeMap<u16, EbmTerm> =
        base.ebm_terms.iter().map(|t| (t.id, t.clone())).collect();
    if let Some(ov) = &overlay {
        for t in &ov.ebm_terms {
            if terms_by_id.contains_key(&t.id) {
                return Err(PolicyPackError::MergeConflict(format!(
                    "duplicate ebm term id {}",
                    t.id
                )));
            }
            terms_by_id.insert(t.id, t.clone());
        }
    }

    let thresholds = merge_kv(&base.thresholds, overlay.as_ref().map(|x| &x.thresholds))?;
    let budgets = merge_kv(&base.budgets, overlay.as_ref().map(|x| &x.budgets))?;
    let allowlists = merge_allowlists(&base.allowlists, overlay.as_ref().map(|x| &x.allowlists))?;

    ordered_rule_ids.sort();
    let mut pbm_gem_rules: Vec<PolicyRule> = ordered_rule_ids
        .into_iter()
        .filter_map(|id| rules_by_id.get(&id).cloned())
        .collect();
    pbm_gem_rules.sort_by(|a, b| a.id.cmp(&b.id));

    let mut ebm_terms: Vec<EbmTerm> = terms_by_id.into_values().collect();
    ebm_terms.sort_by_key(|t| t.id);
    nsr.sort();

    let graph = PolicyGraphV1 {
        schema_version: base.manifest.schema_version,
        base_name: base.manifest.name.clone(),
        base_version: base.manifest.version.clone(),
        overlay_name: overlay.as_ref().map(|o| o.manifest.name.clone()),
        overlay_version: overlay.as_ref().map(|o| o.manifest.version.clone()),
        pbm_gem_rules,
        nsr_rules: nsr,
        ebm_terms,
        thresholds,
        budgets,
        allowlists,
        determinism: overlay
            .as_ref()
            .map(|o| o.determinism.clone())
            .unwrap_or_else(|| base.determinism.clone()),
    };
    let digest = policy_graph_digest(&graph)?;
    let provenance = PolicyGraphProvenanceRecord {
        base_pack_digest: base.manifest.pack_digest,
        overlay_pack_digest: overlay.as_ref().map(|o| o.manifest.pack_digest.clone()),
        policy_graph_digest: digest,
        schema_version: graph.schema_version,
        base_version: graph.base_version.clone(),
        overlay_version: graph.overlay_version.clone(),
        validation_ok: true,
        determinism_policy_digest: graph.determinism.digest_hex(),
    };
    Ok((graph, provenance))
}

fn merge_kv(
    base: &BTreeMap<String, i64>,
    overlay: Option<&BTreeMap<String, i64>>,
) -> Result<BTreeMap<String, i64>, PolicyPackError> {
    let mut out = base.clone();
    if let Some(ov) = overlay {
        for (k, v) in ov {
            if !base.contains_key(k) {
                return Err(PolicyPackError::MergeConflict(format!(
                    "overlay key not in base: {k}"
                )));
            }
            out.insert(k.clone(), *v);
        }
    }
    Ok(out)
}

fn merge_allowlists(
    base: &BTreeMap<String, String>,
    overlay: Option<&BTreeMap<String, String>>,
) -> Result<BTreeMap<String, String>, PolicyPackError> {
    let mut out = base.clone();
    if let Some(ov) = overlay {
        for (k, v) in ov {
            if !base.contains_key(k) {
                return Err(PolicyPackError::MergeConflict(format!(
                    "overlay key not in base: {k}"
                )));
            }
            out.insert(k.clone(), v.clone());
        }
    }
    Ok(out)
}

fn load_pack(root: &Path) -> Result<LoadedPack, PolicyPackError> {
    let manifest_path = root.join("pack_manifest.toml");
    let text = fs::read_to_string(&manifest_path)
        .map_err(|_| PolicyPackError::Missing(manifest_path.to_string_lossy().to_string()))?;
    let manifest: PackManifest =
        toml::from_str(&text).map_err(|e| PolicyPackError::InvalidManifest(e.to_string()))?;
    validate_semver(&manifest.version)?;

    for file in &manifest.files {
        let abs = root.join(&file.path);
        let bytes = fs::read(&abs)
            .map_err(|_| PolicyPackError::Missing(abs.to_string_lossy().to_string()))?;
        let actual = hex_lower(sha256(&bytes));
        if actual != file.sha256 {
            return Err(PolicyPackError::HashMismatch {
                path: file.path.clone(),
                expected: file.sha256.clone(),
                actual,
            });
        }
    }

    let computed = compute_pack_digest(root, &manifest.files)?;
    if computed != manifest.pack_digest {
        return Err(PolicyPackError::DigestMismatch {
            expected: manifest.pack_digest,
            actual: computed,
        });
    }

    let pbm_text = fs::read_to_string(root.join("pbm_gem_rules.toml"))
        .map_err(|_| PolicyPackError::Missing("pbm_gem_rules.toml".to_string()))?;
    let pbm_file: RuleFile =
        toml::from_str(&pbm_text).map_err(|e| PolicyPackError::InvalidPack(e.to_string()))?;

    let nsr_text = fs::read_to_string(root.join("nsr_rules_v1.dl"))
        .map_err(|_| PolicyPackError::Missing("nsr_rules_v1.dl".to_string()))?;
    let nsr_rules = nsr_text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(ToString::to_string)
        .collect::<Vec<_>>();

    let ebm_text = fs::read_to_string(root.join("ebm_constraints.toml"))
        .map_err(|_| PolicyPackError::Missing("ebm_constraints.toml".to_string()))?;
    let term_file: TermFile =
        toml::from_str(&ebm_text).map_err(|e| PolicyPackError::InvalidPack(e.to_string()))?;

    let thresholds: KvFile = toml::from_str(
        &fs::read_to_string(root.join("thresholds.toml"))
            .map_err(|_| PolicyPackError::Missing("thresholds.toml".to_string()))?,
    )
    .map_err(|e| PolicyPackError::InvalidPack(e.to_string()))?;
    let budgets: KvFile = toml::from_str(
        &fs::read_to_string(root.join("budgets.toml"))
            .map_err(|_| PolicyPackError::Missing("budgets.toml".to_string()))?,
    )
    .map_err(|e| PolicyPackError::InvalidPack(e.to_string()))?;
    let allowlists: AllowlistFile = toml::from_str(
        &fs::read_to_string(root.join("allowlists.toml"))
            .map_err(|_| PolicyPackError::Missing("allowlists.toml".to_string()))?,
    )
    .map_err(|e| PolicyPackError::InvalidPack(e.to_string()))?;

    let determinism: DeterminismPolicyV1 = toml::from_str(
        &fs::read_to_string(root.join("determinism.toml"))
            .map_err(|_| PolicyPackError::Missing("determinism.toml".to_string()))?,
    )
    .map_err(|e| PolicyPackError::InvalidPack(e.to_string()))?;

    if pbm_file.rules.len() > MAX_RULES || term_file.terms.len() > MAX_TERMS {
        return Err(PolicyPackError::GraphTooLarge);
    }
    let ids = pbm_file
        .rules
        .iter()
        .map(|r| r.id.clone())
        .collect::<BTreeSet<_>>();
    if ids.len() != pbm_file.rules.len() {
        return Err(PolicyPackError::InvalidPack(
            "duplicate rule id".to_string(),
        ));
    }
    if term_file.schema_version != manifest.schema_version {
        return Err(PolicyPackError::InvalidPack(
            "schema version mismatch".to_string(),
        ));
    }

    Ok(LoadedPack {
        manifest,
        pbm_gem_rules: pbm_file.rules,
        nsr_rules,
        ebm_terms: term_file.terms,
        thresholds: thresholds.values,
        budgets: budgets.values,
        allowlists: allowlists.values,
        determinism,
    })
}

pub fn policy_graph_digest(graph: &PolicyGraphV1) -> Result<String, PolicyPackError> {
    let mut hasher = Sha256::new();
    hasher.update(b"ucf.policy.graph.v1");
    hasher.update(graph.schema_version.to_le_bytes());
    hasher.update(graph.base_name.as_bytes());
    hasher.update([0]);
    hasher.update(graph.base_version.as_bytes());
    hasher.update([0]);
    hasher.update(graph.overlay_name.as_deref().unwrap_or("").as_bytes());
    hasher.update([0]);
    hasher.update(graph.overlay_version.as_deref().unwrap_or("").as_bytes());

    for rule in &graph.pbm_gem_rules {
        hasher.update(rule.id.as_bytes());
        hasher.update([0]);
        hasher.update(rule.channel.as_bytes());
        hasher.update([0]);
        hasher.update(rule.decision.as_bytes());
        hasher.update([0]);
    }
    for line in &graph.nsr_rules {
        hasher.update(line.as_bytes());
        hasher.update([0]);
    }
    for term in &graph.ebm_terms {
        hasher.update(term.id.to_le_bytes());
        hasher.update(term.kind.as_bytes());
        hasher.update([0]);
        hasher.update(term.weight_q.to_le_bytes());
        hasher.update(term.threshold_q.unwrap_or(0).to_le_bytes());
        hasher.update([term.governor_tier_min.unwrap_or(0)]);
        hasher.update([term.capability_class_id.unwrap_or(0)]);
        hasher.update(term.candidate_kind.as_deref().unwrap_or("").as_bytes());
        hasher.update([0]);
    }
    hash_map_i64(&mut hasher, &graph.thresholds);
    hash_map_i64(&mut hasher, &graph.budgets);
    hash_map_str(&mut hasher, &graph.allowlists);
    hasher.update(graph.determinism.digest_hex().as_bytes());
    let digest = hex_lower(hasher.finalize().into());

    let approx_size = graph.pbm_gem_rules.len() * 48
        + graph.nsr_rules.iter().map(|x| x.len()).sum::<usize>()
        + graph.ebm_terms.len() * 32
        + graph.thresholds.len() * 24
        + graph.budgets.len() * 24
        + graph
            .allowlists
            .iter()
            .map(|(k, v)| k.len() + v.len())
            .sum::<usize>();
    if approx_size > MAX_GRAPH_BYTES {
        return Err(PolicyPackError::GraphTooLarge);
    }
    Ok(digest)
}

fn hash_map_i64(hasher: &mut Sha256, map: &BTreeMap<String, i64>) {
    for (k, v) in map {
        hasher.update(k.as_bytes());
        hasher.update([0]);
        hasher.update(v.to_le_bytes());
    }
}

fn hash_map_str(hasher: &mut Sha256, map: &BTreeMap<String, String>) {
    for (k, v) in map {
        hasher.update(k.as_bytes());
        hasher.update([0]);
        hasher.update(v.as_bytes());
        hasher.update([0]);
    }
}

fn compute_pack_digest(root: &Path, files: &[ManifestFile]) -> Result<String, PolicyPackError> {
    let mut canonical: Vec<(String, [u8; 32])> = Vec::with_capacity(files.len());
    for file in files {
        let abs: PathBuf = root.join(&file.path);
        let bytes = fs::read(&abs)
            .map_err(|_| PolicyPackError::Missing(abs.to_string_lossy().to_string()))?;
        canonical.push((file.path.clone(), sha256(&bytes)));
    }
    canonical.sort_by(|a, b| a.0.cmp(&b.0));
    let mut hasher = Sha256::new();
    hasher.update(b"ucf.policy.pack.v1");
    for (path, digest) in canonical {
        hasher.update(path.as_bytes());
        hasher.update([0]);
        hasher.update(digest);
    }
    Ok(hex_lower(hasher.finalize().into()))
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().into()
}

fn hex_lower(bytes: [u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

fn validate_semver(v: &str) -> Result<(), PolicyPackError> {
    let parts: Vec<_> = v.split('.').collect();
    if parts.len() != 3 || parts.iter().any(|p| p.parse::<u64>().is_err()) {
        return Err(PolicyPackError::InvalidManifest(format!(
            "version {v} is not semver"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::Digest;

    #[test]
    fn rejects_bad_semver() {
        assert!(validate_semver("v1").is_err());
    }

    #[test]
    fn merge_is_deterministic_for_base_and_dev_overlay() {
        let base = Path::new("../../policies/packs/base_v1");
        let (_, a) = load_and_merge_policy_graph(base, None).expect("merge a");
        let (_, b) = load_and_merge_policy_graph(base, None).expect("merge b");
        assert_eq!(a.policy_graph_digest, b.policy_graph_digest);
    }

    #[test]
    fn unknown_overlay_key_fails() {
        let dir = tempfile::tempdir().expect("tmp");
        let base = Path::new("../../policies/packs/base_v1");
        let seed = Path::new("../../policies/packs/base_v1");
        for name in [
            "pbm_gem_rules.toml",
            "nsr_rules_v1.dl",
            "ebm_constraints.toml",
            "budgets.toml",
            "thresholds.toml",
            "allowlists.toml",
            "determinism.toml",
        ] {
            std::fs::copy(seed.join(name), dir.path().join(name)).expect("copy");
        }
        std::fs::write(
            dir.path().join("thresholds.toml"),
            "[values]
unknown = 1
",
        )
        .expect("write");

        let files = [
            "pbm_gem_rules.toml",
            "nsr_rules_v1.dl",
            "ebm_constraints.toml",
            "budgets.toml",
            "thresholds.toml",
            "allowlists.toml",
            "determinism.toml",
        ];
        let mut lines =
            String::from("name = \"overlay_tmp\"\nversion = \"1.0.0\"\nschema_version = 1\n\n");
        let mut dig = sha2::Sha256::new();
        dig.update(b"ucf.policy.pack.v1");
        let mut pairs = Vec::new();
        for f in files {
            let b = std::fs::read(dir.path().join(f)).expect("read");
            let d = sha256(&b);
            let h = hex_lower(d);
            lines.push_str(&format!(
                "[[files]]\npath = \"{}\"\nsha256 = \"{}\"\n\n",
                f, h
            ));
            pairs.push((f.to_string(), d));
        }
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        for (f, d) in pairs {
            dig.update(f.as_bytes());
            dig.update([0]);
            dig.update(d);
        }
        let pack_digest = hex_lower(dig.finalize().into());
        lines.push_str(&format!("pack_digest = \"{}\"\n", pack_digest));
        std::fs::write(dir.path().join("pack_manifest.toml"), lines).expect("manifest");

        assert!(load_and_merge_policy_graph(base, Some(dir.path())).is_err());
    }
}
