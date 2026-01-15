#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::sync::Mutex;

use serde::Serialize;
use thiserror::Error;
use ucf_bus::MessageEnvelope;
use ucf_commit::{commit_experience_record, commit_milestone_macro};
use ucf_types::v1::spec::{ExperienceRecord, MacroMilestone};
use ucf_types::{Digest32, LogicalTime, NodeId, StreamId, WallTime};

pub const EMBEDDING_DIM: usize = 64;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum IndexError {
    #[error("vector size mismatch: expected {expected}, got {actual}")]
    VectorSizeMismatch { expected: usize, actual: usize },
    #[error("feature not enabled: {0}")]
    FeatureDisabled(&'static str),
    #[error("index backend error: {0}")]
    Backend(String),
}

pub trait VectorIndex: Send + Sync {
    fn upsert(&self, id: Digest32, vector: &[f32], metadata: &[u8]) -> Result<(), IndexError>;
    fn query(&self, vector: &[f32], top_k: usize) -> Result<Vec<Digest32>, IndexError>;
}

#[derive(Clone, Debug)]
pub struct QdrantConfig {
    pub endpoint: String,
    pub api_key: Option<String>,
    pub collection: String,
}

impl QdrantConfig {
    pub fn new(endpoint: impl Into<String>, collection: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            api_key: None,
            collection: collection.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum VectorIndexBackend {
    InMemory,
    Qdrant(QdrantConfig),
}

pub fn select_index_backend(
    backend: VectorIndexBackend,
) -> Result<Box<dyn VectorIndex>, IndexError> {
    match backend {
        VectorIndexBackend::InMemory => Ok(Box::new(InMemoryVectorIndex::new())),
        VectorIndexBackend::Qdrant(_) => {
            #[cfg(feature = "qdrant")]
            {
                Err(IndexError::Backend(
                    "qdrant adapter must be constructed via ucf-vector-index-qdrant".to_string(),
                ))
            }
            #[cfg(not(feature = "qdrant"))]
            {
                Err(IndexError::FeatureDisabled("qdrant"))
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct InMemoryVectorIndex {
    inner: std::sync::Arc<Mutex<InMemoryState>>,
}

#[derive(Debug)]
struct InMemoryState {
    dimension: Option<usize>,
    entries: HashMap<Digest32, InMemoryEntry>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct InMemoryEntry {
    vector: Vec<f32>,
    metadata: Vec<u8>,
}

impl InMemoryVectorIndex {
    pub fn new() -> Self {
        Self {
            inner: std::sync::Arc::new(Mutex::new(InMemoryState {
                dimension: None,
                entries: HashMap::new(),
            })),
        }
    }
}

impl Default for InMemoryVectorIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorIndex for InMemoryVectorIndex {
    fn upsert(&self, id: Digest32, vector: &[f32], metadata: &[u8]) -> Result<(), IndexError> {
        let mut guard = self.inner.lock().expect("lock index");
        if let Some(dim) = guard.dimension {
            if dim != vector.len() {
                return Err(IndexError::VectorSizeMismatch {
                    expected: dim,
                    actual: vector.len(),
                });
            }
        } else {
            guard.dimension = Some(vector.len());
        }
        guard.entries.insert(
            id,
            InMemoryEntry {
                vector: vector.to_vec(),
                metadata: metadata.to_vec(),
            },
        );
        Ok(())
    }

    fn query(&self, vector: &[f32], top_k: usize) -> Result<Vec<Digest32>, IndexError> {
        let guard = self.inner.lock().expect("lock index");
        if let Some(dim) = guard.dimension {
            if dim != vector.len() {
                return Err(IndexError::VectorSizeMismatch {
                    expected: dim,
                    actual: vector.len(),
                });
            }
        }
        let mut scored = guard
            .entries
            .iter()
            .map(|(id, entry)| (id, dot_product(vector, &entry.vector)))
            .collect::<Vec<_>>();
        scored.sort_by(|(id_a, score_a), (id_b, score_b)| {
            score_b
                .total_cmp(score_a)
                .then_with(|| id_a.as_bytes().cmp(id_b.as_bytes()))
        });
        Ok(scored.into_iter().take(top_k).map(|(id, _)| *id).collect())
    }
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn embedding_stub(seed: &Digest32) -> Vec<f32> {
    let mut out = Vec::with_capacity(EMBEDDING_DIM);
    let bytes = seed.as_bytes();
    for i in 0..EMBEDDING_DIM {
        let byte = bytes[i % bytes.len()];
        let mixed = byte.wrapping_add((i as u8).wrapping_mul(17));
        let value = (mixed as f32 / 255.0) * 2.0 - 1.0;
        out.push(value);
    }
    out
}

#[derive(Clone, Debug)]
pub struct MacroMilestoneFinalized {
    pub record: ExperienceRecord,
    pub milestone: MacroMilestone,
    pub policy_class: Option<String>,
}

#[derive(Clone, Debug)]
pub struct RecordAppended {
    pub record: ExperienceRecord,
}

#[derive(Debug, Serialize)]
struct MacroMilestoneMetadata {
    milestone_type: String,
    milestone_id: String,
    refs: Vec<String>,
    policy_class: Option<String>,
}

pub struct IndexSync<I> {
    index: I,
    macro_rx: std::sync::mpsc::Receiver<MessageEnvelope<MacroMilestoneFinalized>>,
    record_rx: std::sync::mpsc::Receiver<MessageEnvelope<RecordAppended>>,
}

impl<I> IndexSync<I>
where
    I: VectorIndex,
{
    pub fn new<M, R>(index: I, macro_bus: &M, record_bus: &R) -> Self
    where
        M: ucf_bus::BusSubscriber<MessageEnvelope<MacroMilestoneFinalized>>,
        R: ucf_bus::BusSubscriber<MessageEnvelope<RecordAppended>>,
    {
        Self {
            index,
            macro_rx: macro_bus.subscribe(),
            record_rx: record_bus.subscribe(),
        }
    }

    pub fn sync_once(&mut self) -> Result<usize, IndexError> {
        let mut processed = 0;
        while let Ok(event) = self.macro_rx.try_recv() {
            self.handle_macro_finalized(&event.payload)?;
            processed += 1;
        }
        while let Ok(_event) = self.record_rx.try_recv() {}
        Ok(processed)
    }

    fn handle_macro_finalized(&self, event: &MacroMilestoneFinalized) -> Result<(), IndexError> {
        let record_commit = commit_experience_record(&event.record);
        let macro_commit = commit_milestone_macro(&event.milestone);
        let embedding_seed = xor_digest(&record_commit.digest, &macro_commit.digest);
        let embedding = embedding_stub(&embedding_seed);
        let metadata = build_macro_metadata(&event.milestone, event.policy_class.as_ref())?;
        self.index
            .upsert(macro_commit.digest, &embedding, &metadata)
    }
}

pub fn build_record_appended_envelope(
    record: ExperienceRecord,
    node_id: NodeId,
) -> MessageEnvelope<RecordAppended> {
    MessageEnvelope {
        node_id,
        stream_id: StreamId::new("record-appended"),
        logical_time: LogicalTime::new(0),
        wall_time: WallTime::new(record.observed_at_ms),
        payload: RecordAppended { record },
    }
}

pub fn build_macro_finalized_envelope(
    record: ExperienceRecord,
    milestone: MacroMilestone,
    node_id: NodeId,
    policy_class: Option<String>,
) -> MessageEnvelope<MacroMilestoneFinalized> {
    MessageEnvelope {
        node_id,
        stream_id: StreamId::new("macro-milestone-finalized"),
        logical_time: LogicalTime::new(0),
        wall_time: WallTime::new(milestone.achieved_at_ms),
        payload: MacroMilestoneFinalized {
            record,
            milestone,
            policy_class,
        },
    }
}

fn build_macro_metadata(
    milestone: &MacroMilestone,
    policy_class: Option<&String>,
) -> Result<Vec<u8>, IndexError> {
    let metadata = MacroMilestoneMetadata {
        milestone_type: "macro_milestone".to_string(),
        milestone_id: milestone.milestone_id.clone(),
        refs: milestone.meso_milestone_ids.clone(),
        policy_class: policy_class.cloned(),
    };
    serde_json::to_vec(&metadata)
        .map_err(|err| IndexError::Backend(format!("metadata encode: {err}")))
}

fn xor_digest(a: &Digest32, b: &Digest32) -> Digest32 {
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = a.as_bytes()[i] ^ b.as_bytes()[i];
    }
    Digest32::new(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use ucf_bus::{BusPublisher, InMemoryBus};

    fn digest_with_byte(byte: u8) -> Digest32 {
        Digest32::new([byte; 32])
    }

    #[test]
    fn in_memory_upsert_and_query() {
        let index = InMemoryVectorIndex::new();
        index
            .upsert(digest_with_byte(1), &[1.0, 0.0], b"alpha")
            .expect("upsert");
        index
            .upsert(digest_with_byte(2), &[0.0, 1.0], b"beta")
            .expect("upsert");

        let result = index.query(&[1.0, 0.0], 1).expect("query");
        assert_eq!(result, vec![digest_with_byte(1)]);
    }

    #[test]
    fn embedding_stub_is_deterministic() {
        let seed = digest_with_byte(7);
        let a = embedding_stub(&seed);
        let b = embedding_stub(&seed);
        assert_eq!(a, b);
        assert_eq!(a.len(), EMBEDDING_DIM);
    }

    #[test]
    fn index_sync_upserts_on_macro_finalized() {
        struct SpyIndex {
            count: Arc<AtomicUsize>,
        }

        impl VectorIndex for SpyIndex {
            fn upsert(
                &self,
                _id: Digest32,
                _vector: &[f32],
                _metadata: &[u8],
            ) -> Result<(), IndexError> {
                self.count.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }

            fn query(&self, _vector: &[f32], _top_k: usize) -> Result<Vec<Digest32>, IndexError> {
                Ok(Vec::new())
            }
        }

        let count = Arc::new(AtomicUsize::new(0));
        let index = SpyIndex {
            count: count.clone(),
        };
        let macro_bus = InMemoryBus::new();
        let record_bus = InMemoryBus::new();
        let mut sync = IndexSync::new(index, &macro_bus, &record_bus);

        let record = ExperienceRecord {
            record_id: "rec-1".to_string(),
            observed_at_ms: 42,
            subject_id: "subject".to_string(),
            payload: vec![1, 2, 3],
            digest: None,
            vrf_tag: None,
            proof_ref: None,
        };
        let milestone = MacroMilestone {
            milestone_id: "macro-1".to_string(),
            achieved_at_ms: 42,
            label: "macro".to_string(),
            meso_milestone_ids: vec!["meso-1".to_string()],
        };
        let envelope = build_macro_finalized_envelope(record, milestone, NodeId::new("node"), None);
        macro_bus.publish(envelope);
        sync.sync_once().expect("sync");

        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[test]
    #[cfg(not(feature = "qdrant"))]
    fn selecting_qdrant_without_feature_errors() {
        let config = QdrantConfig::new("http://localhost:6333", "ucf");
        match select_index_backend(VectorIndexBackend::Qdrant(config)) {
            Err(err) => assert_eq!(err, IndexError::FeatureDisabled("qdrant")),
            Ok(_) => panic!("expected feature disabled error"),
        }
    }
}
