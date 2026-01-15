#![forbid(unsafe_code)]

use ucf_types::Digest32;
use ucf_vector_index::{IndexError, QdrantConfig, VectorIndex};

#[derive(Clone, Debug)]
pub struct QdrantClient {
    pub endpoint: String,
    pub api_key: Option<String>,
}

impl QdrantClient {
    pub fn new(config: &QdrantConfig) -> Self {
        Self {
            endpoint: config.endpoint.clone(),
            api_key: config.api_key.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct QdrantVectorIndex {
    config: QdrantConfig,
    _client: QdrantClient,
}

impl QdrantVectorIndex {
    pub fn new(config: QdrantConfig) -> Result<Self, IndexError> {
        let client = QdrantClient::new(&config);
        Ok(Self {
            config,
            _client: client,
        })
    }

    pub fn config(&self) -> &QdrantConfig {
        &self.config
    }
}

impl VectorIndex for QdrantVectorIndex {
    fn upsert(&self, _id: Digest32, _vector: &[f32], _metadata: &[u8]) -> Result<(), IndexError> {
        Err(IndexError::Backend(
            "qdrant adapter not wired for network operations".to_string(),
        ))
    }

    fn query(&self, _vector: &[f32], _top_k: usize) -> Result<Vec<Digest32>, IndexError> {
        Err(IndexError::Backend(
            "qdrant adapter not wired for network operations".to_string(),
        ))
    }
}
