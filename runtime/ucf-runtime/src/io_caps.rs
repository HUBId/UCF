use blake3::Hasher;
use std::fs;
use std::path::{Component, Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IoMode {
    Read,
    Write,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IoViolationRecord {
    pub path_digest: [u8; 32],
    pub reason: &'static str,
    pub mode: IoMode,
}

#[derive(Clone, Debug)]
pub struct IoCaps {
    roots: Vec<PathBuf>,
    pub allow_read: bool,
    pub allow_write: bool,
    pub max_bytes_per_op: usize,
    denylist: Vec<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum IoCapsError {
    #[error("capability denied")]
    CapabilityDenied,
    #[error("path traversal denied")]
    TraversalDenied,
    #[error("root escape denied")]
    RootEscapeDenied,
    #[error("denylisted path")]
    Denylisted,
    #[error("size exceeds cap")]
    SizeCapExceeded,
    #[error("io: {0}")]
    Io(String),
}

impl IoCaps {
    pub fn runtime_default() -> Self {
        let workspace_root = std::env::var("UCF_BUNDLE_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."));
        Self {
            roots: vec![
                workspace_root.join("policies"),
                workspace_root.join("models"),
                workspace_root,
                PathBuf::from("policies"),
                PathBuf::from("models"),
                PathBuf::from("."),
            ],
            allow_read: true,
            allow_write: false,
            max_bytes_per_op: 128 * 1024,
            denylist: vec![".ssh".to_string(), "/proc".to_string(), "..".to_string()],
        }
    }

    pub fn read_to_string(&self, path: &Path) -> Result<String, IoCapsError> {
        if !self.allow_read {
            return Err(IoCapsError::CapabilityDenied);
        }
        let resolved = self.resolve(path, IoMode::Read)?;
        let bytes = fs::read(&resolved).map_err(|e| IoCapsError::Io(e.to_string()))?;
        if bytes.len() > self.max_bytes_per_op {
            return Err(IoCapsError::SizeCapExceeded);
        }
        Ok(String::from_utf8_lossy(&bytes).to_string())
    }

    pub fn read(&self, path: &Path) -> Result<Vec<u8>, IoCapsError> {
        if !self.allow_read {
            return Err(IoCapsError::CapabilityDenied);
        }
        let resolved = self.resolve(path, IoMode::Read)?;
        let bytes = fs::read(&resolved).map_err(|e| IoCapsError::Io(e.to_string()))?;
        if bytes.len() > self.max_bytes_per_op {
            return Err(IoCapsError::SizeCapExceeded);
        }
        Ok(bytes)
    }

    fn resolve(&self, path: &Path, _mode: IoMode) -> Result<PathBuf, IoCapsError> {
        if path.is_absolute() || path.components().any(|c| matches!(c, Component::ParentDir)) {
            return Err(IoCapsError::TraversalDenied);
        }
        let denied = path.to_string_lossy();
        if self.denylist.iter().any(|pat| denied.contains(pat)) {
            return Err(IoCapsError::Denylisted);
        }
        for root in &self.roots {
            let joined = root.join(path);
            let root_canon = root
                .canonicalize()
                .map_err(|e| IoCapsError::Io(e.to_string()))?;
            if let Ok(joined_canon) = joined.canonicalize() {
                if joined_canon.starts_with(&root_canon) {
                    return Ok(joined_canon);
                }
            }
        }
        Err(IoCapsError::RootEscapeDenied)
    }

    pub fn violation_for(path: &Path, mode: IoMode, reason: &'static str) -> IoViolationRecord {
        let mut hasher = Hasher::new();
        hasher.update(path.to_string_lossy().as_bytes());
        IoViolationRecord {
            path_digest: hasher.finalize().into(),
            reason,
            mode,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_traversal() {
        let caps = IoCaps::runtime_default();
        let err = caps
            .read_to_string(Path::new("../Cargo.toml"))
            .expect_err("blocked");
        assert!(matches!(err, IoCapsError::TraversalDenied));
    }
}
