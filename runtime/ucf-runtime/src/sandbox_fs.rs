use metrics::counter;
use std::fs;
use std::path::{Component, Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FsCapabilityKind {
    FileRead,
    FileWrite,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FsCapabilityToken {
    pub kind: FsCapabilityKind,
    pub root_id: String,
}

#[derive(Clone, Debug)]
pub struct SandboxFs {
    roots: Vec<(String, PathBuf)>,
}

#[derive(Debug, thiserror::Error)]
pub enum SandboxFsError {
    #[error("root not found: {0}")]
    RootNotFound(String),
    #[error("path traversal denied")]
    TraversalDenied,
    #[error("path escape denied")]
    EscapeDenied,
    #[error("io: {0}")]
    Io(String),
    #[error("capability denied")]
    CapabilityDenied,
}

impl SandboxFs {
    pub fn new(roots: Vec<(String, PathBuf)>) -> Self {
        Self { roots }
    }

    pub fn read_to_string(
        &self,
        token: &FsCapabilityToken,
        rel: &Path,
    ) -> Result<String, SandboxFsError> {
        if token.kind != FsCapabilityKind::FileRead {
            return Err(SandboxFsError::CapabilityDenied);
        }
        let path = self.resolve(token, rel)?;
        fs::read_to_string(path).map_err(|e| SandboxFsError::Io(e.to_string()))
    }

    fn resolve(&self, token: &FsCapabilityToken, rel: &Path) -> Result<PathBuf, SandboxFsError> {
        let (_, root) = self
            .roots
            .iter()
            .find(|(id, _)| *id == token.root_id)
            .ok_or_else(|| SandboxFsError::RootNotFound(token.root_id.clone()))?;
        if rel.is_absolute() || rel.components().any(|c| matches!(c, Component::ParentDir)) {
            counter!("ucf_sandbox_fs_denied_total", "reason" => "traversal".to_string())
                .increment(1);
            return Err(SandboxFsError::TraversalDenied);
        }
        let joined = root.join(rel);
        let canon_root = root
            .canonicalize()
            .map_err(|e| SandboxFsError::Io(e.to_string()))?;
        let canon_joined = joined
            .canonicalize()
            .map_err(|e| SandboxFsError::Io(e.to_string()))?;
        if !canon_joined.starts_with(&canon_root) {
            counter!("ucf_sandbox_fs_denied_total", "reason" => "escape".to_string()).increment(1);
            return Err(SandboxFsError::EscapeDenied);
        }
        Ok(canon_joined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_parent_traversal() {
        let fs = SandboxFs::new(vec![("models_root".to_string(), PathBuf::from("./models"))]);
        let tok = FsCapabilityToken {
            kind: FsCapabilityKind::FileRead,
            root_id: "models_root".to_string(),
        };
        let err = fs
            .resolve(&tok, Path::new("../Cargo.toml"))
            .expect_err("must deny");
        assert!(matches!(err, SandboxFsError::TraversalDenied));
    }
}
