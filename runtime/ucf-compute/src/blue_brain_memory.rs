use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

const BLUE_BRAIN_MEMORY_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlueBrainMemoryCandidateClass {
    CommitEligible,
    Deferred,
    Rejected,
    Blocked,
    Insufficient,
    ReferenceOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlueBrainMemoryOrigin {
    Context,
    Evidence,
    Replay,
    Reference,
    ComputeResult,
    Selection,
    CommitFeedback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlueBrainMemoryFreshness {
    Current,
    Stale,
    Partial,
    Caveated,
    Degraded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlueBrainMemoryCommitResultState {
    Committed,
    CommittedWithCaveat,
    Rejected,
    Blocked,
    Failed,
    NoOp,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainMemoryCandidate {
    pub candidate_id: String,
    pub class: BlueBrainMemoryCandidateClass,
    pub origins: Vec<BlueBrainMemoryOrigin>,
    pub evidence_refs: Vec<String>,
    pub reference_refs: Vec<String>,
    pub context_basis_refs: Vec<String>,
    pub selection_basis_refs: Vec<String>,
    pub freshness: BlueBrainMemoryFreshness,
    pub caveats: Vec<String>,
    pub allow_caveated_commit: bool,
    pub allow_stale_context_commit: bool,
    pub has_internal_only_dependency: bool,
    pub commit_path_available: bool,
}

impl BlueBrainMemoryCandidate {
    pub fn canonicalized(mut self) -> Self {
        self.origins.sort_unstable_by_key(|origin| *origin as u8);
        self.origins.dedup();
        self.evidence_refs.sort_unstable();
        self.evidence_refs.dedup();
        self.reference_refs.sort_unstable();
        self.reference_refs.dedup();
        self.context_basis_refs.sort_unstable();
        self.context_basis_refs.dedup();
        self.selection_basis_refs.sort_unstable();
        self.selection_basis_refs.dedup();
        self.caveats.sort_unstable();
        self.caveats.dedup();
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PersistedBlueBrainMemoryRecord {
    pub schema_version: u16,
    pub memory_record_id: String,
    pub source_candidate_id: String,
    pub origins: Vec<BlueBrainMemoryOrigin>,
    pub evidence_refs: Vec<String>,
    pub reference_refs: Vec<String>,
    pub context_basis_refs: Vec<String>,
    pub selection_basis_refs: Vec<String>,
    pub freshness: BlueBrainMemoryFreshness,
    pub caveats: Vec<String>,
    pub committed_at_unix_ms: u64,
    pub commit_result_state: BlueBrainMemoryCommitResultState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainMemoryCommitReport {
    pub candidate_id: String,
    pub result_state: BlueBrainMemoryCommitResultState,
    pub memory_record_id: Option<String>,
    pub created_record: bool,
    pub diagnostic: String,
    pub caveats: Vec<String>,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum BlueBrainMemoryStoreError {
    #[error("blue-brain memory store io error during {operation} at {path}: {reason}")]
    Io {
        operation: &'static str,
        path: String,
        reason: String,
    },
    #[error("blue-brain memory store corrupted at line {line}: {reason}")]
    Corrupt { line: usize, reason: String },
    #[error("blue-brain memory encode failure: {reason}")]
    Encode { reason: String },
}

#[derive(Debug, Clone)]
pub struct BlueBrainMemoryStore {
    path: PathBuf,
    records: BTreeMap<String, PersistedBlueBrainMemoryRecord>,
    candidate_index: BTreeMap<String, String>,
}

impl BlueBrainMemoryStore {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, BlueBrainMemoryStoreError> {
        let path = path.into();
        let mut records = BTreeMap::new();
        let mut candidate_index = BTreeMap::new();
        if path.exists() {
            let file = fs::File::open(&path).map_err(|err| BlueBrainMemoryStoreError::Io {
                operation: "open",
                path: path.display().to_string(),
                reason: err.to_string(),
            })?;
            for (line_idx, line) in BufReader::new(file).lines().enumerate() {
                let line = line.map_err(|err| BlueBrainMemoryStoreError::Io {
                    operation: "read",
                    path: path.display().to_string(),
                    reason: err.to_string(),
                })?;
                if line.trim().is_empty() {
                    continue;
                }
                let parsed: PersistedBlueBrainMemoryRecord =
                    serde_json::from_str(&line).map_err(|err| {
                        BlueBrainMemoryStoreError::Corrupt {
                            line: line_idx + 1,
                            reason: err.to_string(),
                        }
                    })?;
                candidate_index.insert(
                    parsed.source_candidate_id.clone(),
                    parsed.memory_record_id.clone(),
                );
                records.insert(parsed.memory_record_id.clone(), parsed);
            }
        }
        Ok(Self {
            path,
            records,
            candidate_index,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn get(&self, memory_record_id: &str) -> Option<&PersistedBlueBrainMemoryRecord> {
        self.records.get(memory_record_id)
    }

    pub fn get_by_candidate(&self, candidate_id: &str) -> Option<&PersistedBlueBrainMemoryRecord> {
        self.candidate_index
            .get(candidate_id)
            .and_then(|id| self.get(id))
    }

    pub fn commit_candidate(
        &mut self,
        candidate: BlueBrainMemoryCandidate,
        committed_at_unix_ms: u64,
    ) -> BlueBrainMemoryCommitReport {
        let candidate = candidate.canonicalized();
        let mut caveats = candidate.caveats.clone();

        if !candidate.commit_path_available {
            return BlueBrainMemoryCommitReport {
                candidate_id: candidate.candidate_id,
                result_state: BlueBrainMemoryCommitResultState::Unavailable,
                memory_record_id: None,
                created_record: false,
                diagnostic: "canonical memory store unavailable".to_string(),
                caveats,
            };
        }

        if let Some(existing) = self.get_by_candidate(&candidate.candidate_id) {
            return BlueBrainMemoryCommitReport {
                candidate_id: candidate.candidate_id,
                result_state: BlueBrainMemoryCommitResultState::NoOp,
                memory_record_id: Some(existing.memory_record_id.clone()),
                created_record: false,
                diagnostic: "candidate already committed in canonical memory store".to_string(),
                caveats,
            };
        }

        let eligibility_guard = match candidate.class {
            BlueBrainMemoryCandidateClass::CommitEligible => None,
            BlueBrainMemoryCandidateClass::Deferred => Some((
                BlueBrainMemoryCommitResultState::Blocked,
                "candidate deferred and not commit-eligible",
            )),
            BlueBrainMemoryCandidateClass::Rejected => Some((
                BlueBrainMemoryCommitResultState::Rejected,
                "candidate rejected and not commit-eligible",
            )),
            BlueBrainMemoryCandidateClass::Blocked => Some((
                BlueBrainMemoryCommitResultState::Blocked,
                "candidate blocked and not commit-eligible",
            )),
            BlueBrainMemoryCandidateClass::Insufficient => Some((
                BlueBrainMemoryCommitResultState::Blocked,
                "candidate insufficient and not commit-eligible",
            )),
            BlueBrainMemoryCandidateClass::ReferenceOnly => Some((
                BlueBrainMemoryCommitResultState::Rejected,
                "reference-only candidate cannot be committed as memory",
            )),
        };
        if let Some((state, diagnostic)) = eligibility_guard {
            return BlueBrainMemoryCommitReport {
                candidate_id: candidate.candidate_id,
                result_state: state,
                memory_record_id: None,
                created_record: false,
                diagnostic: diagnostic.to_string(),
                caveats,
            };
        }

        if candidate.has_internal_only_dependency {
            return BlueBrainMemoryCommitReport {
                candidate_id: candidate.candidate_id,
                result_state: BlueBrainMemoryCommitResultState::Blocked,
                memory_record_id: None,
                created_record: false,
                diagnostic: "candidate depends on internal/expert-only basis".to_string(),
                caveats,
            };
        }

        let has_basis = !candidate.evidence_refs.is_empty()
            || !candidate.reference_refs.is_empty()
            || !candidate.context_basis_refs.is_empty();
        if !has_basis && !candidate.allow_caveated_commit {
            return BlueBrainMemoryCommitReport {
                candidate_id: candidate.candidate_id,
                result_state: BlueBrainMemoryCommitResultState::Rejected,
                memory_record_id: None,
                created_record: false,
                diagnostic: "missing evidence/reference/context basis for commit".to_string(),
                caveats,
            };
        }

        let stale_context = candidate.freshness == BlueBrainMemoryFreshness::Stale;
        if stale_context && !candidate.allow_stale_context_commit {
            return BlueBrainMemoryCommitReport {
                candidate_id: candidate.candidate_id,
                result_state: BlueBrainMemoryCommitResultState::Blocked,
                memory_record_id: None,
                created_record: false,
                diagnostic: "stale context basis blocked commit".to_string(),
                caveats,
            };
        }

        let result_state = if candidate.allow_caveated_commit
            || stale_context
            || !candidate.caveats.is_empty()
            || candidate.freshness != BlueBrainMemoryFreshness::Current
        {
            if caveats.is_empty() {
                caveats.push("commit accepted with caveat posture".to_string());
            }
            BlueBrainMemoryCommitResultState::CommittedWithCaveat
        } else {
            BlueBrainMemoryCommitResultState::Committed
        };

        let memory_record_id = build_memory_record_id(&candidate, committed_at_unix_ms);
        let persisted = PersistedBlueBrainMemoryRecord {
            schema_version: BLUE_BRAIN_MEMORY_SCHEMA_VERSION,
            memory_record_id: memory_record_id.clone(),
            source_candidate_id: candidate.candidate_id.clone(),
            origins: candidate.origins,
            evidence_refs: candidate.evidence_refs,
            reference_refs: candidate.reference_refs,
            context_basis_refs: candidate.context_basis_refs,
            selection_basis_refs: candidate.selection_basis_refs,
            freshness: candidate.freshness,
            caveats: caveats.clone(),
            committed_at_unix_ms,
            commit_result_state: result_state,
        };

        match self.upsert(persisted) {
            Ok(()) => BlueBrainMemoryCommitReport {
                candidate_id: candidate.candidate_id,
                result_state,
                memory_record_id: Some(memory_record_id),
                created_record: true,
                diagnostic: "candidate committed into canonical persisted memory store".to_string(),
                caveats,
            },
            Err(err) => BlueBrainMemoryCommitReport {
                candidate_id: candidate.candidate_id,
                result_state: BlueBrainMemoryCommitResultState::Failed,
                memory_record_id: None,
                created_record: false,
                diagnostic: format!("memory store write failed: {err}"),
                caveats,
            },
        }
    }

    fn upsert(
        &mut self,
        persisted: PersistedBlueBrainMemoryRecord,
    ) -> Result<(), BlueBrainMemoryStoreError> {
        let encoded =
            serde_json::to_string(&persisted).map_err(|err| BlueBrainMemoryStoreError::Encode {
                reason: err.to_string(),
            })?;
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(|err| BlueBrainMemoryStoreError::Io {
                operation: "mkdir",
                path: parent.display().to_string(),
                reason: err.to_string(),
            })?;
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|err| BlueBrainMemoryStoreError::Io {
                operation: "append-open",
                path: self.path.display().to_string(),
                reason: err.to_string(),
            })?;

        file.write_all(encoded.as_bytes())
            .and_then(|_| file.write_all(b"\n"))
            .and_then(|_| file.flush())
            .map_err(|err| BlueBrainMemoryStoreError::Io {
                operation: "append-write",
                path: self.path.display().to_string(),
                reason: err.to_string(),
            })?;

        self.candidate_index.insert(
            persisted.source_candidate_id.clone(),
            persisted.memory_record_id.clone(),
        );
        self.records
            .insert(persisted.memory_record_id.clone(), persisted);
        Ok(())
    }
}

fn build_memory_record_id(
    candidate: &BlueBrainMemoryCandidate,
    committed_at_unix_ms: u64,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(candidate.candidate_id.as_bytes());
    hasher.update(b"|");
    hasher.update(committed_at_unix_ms.to_le_bytes());
    hasher.update(b"|");
    hasher.update(format!("{:?}", candidate.freshness).as_bytes());
    hasher.update(b"|");
    for item in &candidate.evidence_refs {
        hasher.update(item.as_bytes());
        hasher.update(b";");
    }
    for item in &candidate.reference_refs {
        hasher.update(item.as_bytes());
        hasher.update(b";");
    }
    for item in &candidate.context_basis_refs {
        hasher.update(item.as_bytes());
        hasher.update(b";");
    }
    for item in &candidate.selection_basis_refs {
        hasher.update(item.as_bytes());
        hasher.update(b";");
    }
    for item in &candidate.caveats {
        hasher.update(item.as_bytes());
        hasher.update(b";");
    }
    format!("bb_mem_{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEST_PATH_SEQ: AtomicU64 = AtomicU64::new(0);

    fn temp_store_path(name: &str) -> PathBuf {
        let sequence = TEST_PATH_SEQ.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!(
            "ucf_blue_brain_memory_store_{}_{}_{}_{}.jsonl",
            name,
            std::process::id(),
            nanos,
            sequence
        ))
    }

    fn base_candidate(id: &str) -> BlueBrainMemoryCandidate {
        BlueBrainMemoryCandidate {
            candidate_id: id.to_string(),
            class: BlueBrainMemoryCandidateClass::CommitEligible,
            origins: vec![
                BlueBrainMemoryOrigin::Context,
                BlueBrainMemoryOrigin::Evidence,
            ],
            evidence_refs: vec!["evidence:bundle:42".to_string()],
            reference_refs: vec!["ref:status:7".to_string()],
            context_basis_refs: vec!["context:digest:abc".to_string()],
            selection_basis_refs: vec!["selection:attention:primary".to_string()],
            freshness: BlueBrainMemoryFreshness::Current,
            caveats: Vec::new(),
            allow_caveated_commit: false,
            allow_stale_context_commit: false,
            has_internal_only_dependency: false,
            commit_path_available: true,
        }
    }

    #[test]
    fn commit_eligible_candidate_is_persisted_and_readable() {
        let path = temp_store_path("commit_ok");
        let _ = std::fs::remove_file(&path);
        let mut store = BlueBrainMemoryStore::open(&path).expect("open memory store");

        let report = store.commit_candidate(base_candidate("cand-1"), 100);
        assert_eq!(
            report.result_state,
            BlueBrainMemoryCommitResultState::Committed
        );
        let record_id = report.memory_record_id.expect("record id");
        assert!(report.created_record);

        let record = store.get(&record_id).expect("persisted record");
        assert_eq!(record.source_candidate_id, "cand-1");
        assert_eq!(
            record.commit_result_state,
            BlueBrainMemoryCommitResultState::Committed
        );
        assert_eq!(record.evidence_refs, vec!["evidence:bundle:42"]);
    }

    #[test]
    fn commit_with_caveat_is_persisted_with_caveats() {
        let path = temp_store_path("commit_caveat");
        let _ = std::fs::remove_file(&path);
        let mut store = BlueBrainMemoryStore::open(&path).expect("open memory store");

        let mut candidate = base_candidate("cand-2");
        candidate.allow_caveated_commit = true;
        candidate.caveats = vec!["partial evidence basis".to_string()];

        let report = store.commit_candidate(candidate, 101);
        assert_eq!(
            report.result_state,
            BlueBrainMemoryCommitResultState::CommittedWithCaveat
        );
        let record = store
            .get(report.memory_record_id.as_deref().expect("record id"))
            .expect("record");
        assert_eq!(record.caveats, vec!["partial evidence basis"]);
    }

    #[test]
    fn deferred_rejected_insufficient_and_reference_only_are_not_committed() {
        let path = temp_store_path("guard_classes");
        let _ = std::fs::remove_file(&path);
        let mut store = BlueBrainMemoryStore::open(&path).expect("open memory store");

        let mut deferred = base_candidate("cand-3");
        deferred.class = BlueBrainMemoryCandidateClass::Deferred;
        assert_eq!(
            store.commit_candidate(deferred, 102).result_state,
            BlueBrainMemoryCommitResultState::Blocked
        );

        let mut rejected = base_candidate("cand-4");
        rejected.class = BlueBrainMemoryCandidateClass::Rejected;
        assert_eq!(
            store.commit_candidate(rejected, 103).result_state,
            BlueBrainMemoryCommitResultState::Rejected
        );

        let mut insufficient = base_candidate("cand-5");
        insufficient.class = BlueBrainMemoryCandidateClass::Insufficient;
        assert_eq!(
            store.commit_candidate(insufficient, 104).result_state,
            BlueBrainMemoryCommitResultState::Blocked
        );

        let mut reference_only = base_candidate("cand-6");
        reference_only.class = BlueBrainMemoryCandidateClass::ReferenceOnly;
        assert_eq!(
            store.commit_candidate(reference_only, 105).result_state,
            BlueBrainMemoryCommitResultState::Rejected
        );

        assert_eq!(store.len(), 0);
    }

    #[test]
    fn stale_context_and_internal_only_dependency_are_blocked() {
        let path = temp_store_path("stale_internal_blocked");
        let _ = std::fs::remove_file(&path);
        let mut store = BlueBrainMemoryStore::open(&path).expect("open memory store");

        let mut stale = base_candidate("cand-7");
        stale.freshness = BlueBrainMemoryFreshness::Stale;
        let stale_report = store.commit_candidate(stale, 106);
        assert_eq!(
            stale_report.result_state,
            BlueBrainMemoryCommitResultState::Blocked
        );

        let mut internal = base_candidate("cand-8");
        internal.has_internal_only_dependency = true;
        let internal_report = store.commit_candidate(internal, 107);
        assert_eq!(
            internal_report.result_state,
            BlueBrainMemoryCommitResultState::Blocked
        );
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn no_op_and_unavailable_states_are_reported_without_creating_records() {
        let path = temp_store_path("noop_unavailable");
        let _ = std::fs::remove_file(&path);
        let mut store = BlueBrainMemoryStore::open(&path).expect("open memory store");

        let first = store.commit_candidate(base_candidate("cand-9"), 108);
        assert_eq!(
            first.result_state,
            BlueBrainMemoryCommitResultState::Committed
        );

        let second = store.commit_candidate(base_candidate("cand-9"), 109);
        assert_eq!(second.result_state, BlueBrainMemoryCommitResultState::NoOp);
        assert!(!second.created_record);

        let mut unavailable = base_candidate("cand-10");
        unavailable.commit_path_available = false;
        let unavailable = store.commit_candidate(unavailable, 110);
        assert_eq!(
            unavailable.result_state,
            BlueBrainMemoryCommitResultState::Unavailable
        );
        assert!(unavailable.memory_record_id.is_none());
    }

    #[test]
    fn record_id_is_stable_and_reopen_supports_read_by_id_or_candidate() {
        let path = temp_store_path("reopen_read");
        let _ = std::fs::remove_file(&path);
        let mut store = BlueBrainMemoryStore::open(&path).expect("open memory store");
        let candidate = base_candidate("cand-11");

        let expected_id = super::build_memory_record_id(&candidate, 111);
        let report = store.commit_candidate(candidate.clone(), 111);
        assert_eq!(
            report.memory_record_id.as_deref(),
            Some(expected_id.as_str())
        );

        drop(store);
        let reopened = BlueBrainMemoryStore::open(&path).expect("reopen memory store");
        let by_id = reopened.get(&expected_id).expect("by id");
        let by_candidate = reopened.get_by_candidate("cand-11").expect("by candidate");
        assert_eq!(by_id.memory_record_id, by_candidate.memory_record_id);
    }

    #[test]
    fn failed_result_is_reported_when_store_path_is_not_writable() {
        let path = temp_store_path("commit_failed");
        let _ = std::fs::remove_file(&path);
        let mut store = BlueBrainMemoryStore::open(&path).expect("open memory store");

        let dir_path = std::env::temp_dir().join("ucf_blue_brain_memory_store_fail_dir");
        let _ = std::fs::remove_file(&dir_path);
        std::fs::create_dir_all(&dir_path).expect("create dir");
        store.path = dir_path.clone();

        let report = store.commit_candidate(base_candidate("cand-12"), 112);
        assert_eq!(
            report.result_state,
            BlueBrainMemoryCommitResultState::Failed
        );
        assert!(report.memory_record_id.is_none());

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(&dir_path);
    }
}
