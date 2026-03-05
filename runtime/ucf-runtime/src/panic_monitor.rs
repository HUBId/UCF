use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::{fs as fileio, fs::OpenOptions};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PanicAction {
    Degraded,
    Shutdown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PanicRecordV1 {
    pub schema_version: u16,
    pub t: u64,
    pub module_stage_id: String,
    pub panic_digest: String,
    pub action_taken: PanicAction,
}

pub fn panic_payload_digest(payload: &(dyn std::any::Any + Send)) -> String {
    let msg = payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non_string_panic".to_string());
    let digest = Sha256::digest(msg.as_bytes());
    hex::encode(digest)
}

pub fn panic_log_path() -> PathBuf {
    std::env::var("UCF_PANIC_LOG_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("out/panic_records.jsonl"))
}

pub fn append_panic_record(path: &Path, record: &PanicRecordV1) {
    if let Some(parent) = path.parent() {
        let _ = fileio::create_dir_all(parent);
    }
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        if let Ok(line) = serde_json::to_string(record) {
            let _ = writeln!(file, "{line}");
        }
    }
}

pub fn strict_panic_fail_fast_enabled() -> bool {
    let strict = std::env::var("UCF_STRICT_MODE").ok().as_deref() == Some("1");
    let fail_fast = std::env::var("UCF_STRICT_PANIC_FAIL_FAST")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false);
    strict && fail_fast
}
