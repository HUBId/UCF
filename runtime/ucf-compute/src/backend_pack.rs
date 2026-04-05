use std::sync::{Arc, Mutex};

use sha2::{Digest, Sha256};

#[cfg(feature = "compute-burn")]
use crate::backends::{BurnSaeExtractor, BurnSsmKernel, BurnWorldPredictor};
#[cfg(feature = "compute-candle")]
use crate::backends::{CandleSaeExtractor, CandleSsmKernel, CandleWorldPredictor};
#[cfg(any(feature = "compute-candle", feature = "llm-candle"))]
use crate::capabilities::build_candle_llm_from_slot;
use crate::capabilities::{
    build_llm_backend, LlmBackendConfig, LlmInference, LlmStubBackend, SaeExtractor,
    WorldModelPredictor,
};
use crate::feature_extractor::ToySaeExtractor;
#[cfg(feature = "lfm-burn")]
use crate::lfm::BurnLfmKernel;
#[cfg(feature = "lfm-candle")]
use crate::lfm::CandleLfmKernel;
#[cfg(feature = "lfm-lnn")]
use crate::lfm::LnnOdeLfmKernel;
use crate::lfm::{LfmKernel, ToyLfmKernel};
use crate::ssm::{SsmKernel, ToySsmKernel};
use crate::worker_backend::WorkerBackendPack;
use crate::world_model::MockJepaPredictor;
use crate::{CodeVersionTag, ComputeError, ModelFormat, ModelLoadError, ModelSlot, ModelStore};

const FIXTURE_SCHEMA_V1: u16 = 1;
const MAX_FIXTURE_BYTES: usize = 1024 * 1024;
#[cfg(any(feature = "compute-candle", feature = "llm-candle"))]
const DEFAULT_LLM_TOKENIZER_PATH: &str = "runtime/ucf-compute/fixtures/llm_v1_tiny_vocab.json";
#[cfg(any(feature = "compute-candle", feature = "llm-candle"))]
const DEFAULT_LLM_TOKENIZER_SHA256: &str =
    "e867a121231210b47a6d8d482434a6c436911f21428664bc70f2ef3c0c5272d3";

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BackendPackId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum BackendComponentId {
    StubV0 = 0,
    ToyV1 = 1,
    CandleToyV1 = 2,
    BurnToyV1 = 3,
    LnnOdeV1 = 4,
    RemoteProxyV1 = 5,
    CandleJepaV1 = 10,
    VljepaAdapterV0 = 14,
    CandleVljepaV1 = 15,
    CandleSaeV1 = 11,
    CandleSsmV1 = 12,
    CandleEbmV1 = 13,
    BurnJepaV1 = 20,
    BurnSaeV1 = 21,
    BurnSsmV1 = 22,
    Disabled = 255,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixtureId {
    ToyLlmWeightsV1,
    JepaDynV1,
    SaeProjV1,
    SsmParamsV1,
    LfmParamsV1,
    LfmLnnParamsV1,
}

impl FixtureId {
    pub const fn canonical_order() -> [Self; 6] {
        [
            Self::ToyLlmWeightsV1,
            Self::JepaDynV1,
            Self::SaeProjV1,
            Self::SsmParamsV1,
            Self::LfmParamsV1,
            Self::LfmLnnParamsV1,
        ]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FixtureBlob {
    pub id: FixtureId,
    pub schema_version: u16,
    pub bytes: &'static [u8],
    pub digest: [u8; 32],
}

#[derive(Debug, Default, Clone, Copy)]
pub struct FixtureManager;

impl FixtureManager {
    pub fn get(&self, id: FixtureId) -> Result<FixtureBlob, ComputeError> {
        let bytes = match id {
            FixtureId::ToyLlmWeightsV1 => {
                include_bytes!("../fixtures/toy_weights_v1.json").as_slice()
            }
            FixtureId::JepaDynV1 => include_bytes!("../fixtures/jepa_dyn_v1.json").as_slice(),
            FixtureId::SaeProjV1 => include_bytes!("../fixtures/sae_proj_v1.json").as_slice(),
            FixtureId::SsmParamsV1 => include_bytes!("../fixtures/ssm_toy_v1.json").as_slice(),
            FixtureId::LfmParamsV1 => include_bytes!("../fixtures/lfm_params_v1.json").as_slice(),
            FixtureId::LfmLnnParamsV1 => {
                include_bytes!("../fixtures/lfm_lnn_params_v1.json").as_slice()
            }
        };
        if bytes.len() > MAX_FIXTURE_BYTES {
            return Err(ComputeError::InvalidInput {
                reason: format!("fixture too large: {:?}", id),
            });
        }
        let digest: [u8; 32] = Sha256::digest(bytes).into();
        Ok(FixtureBlob {
            id,
            schema_version: FIXTURE_SCHEMA_V1,
            bytes,
            digest,
        })
    }

    pub fn overall_digest(&self) -> Result<[u8; 32], ComputeError> {
        let mut hasher = Sha256::new();
        for id in FixtureId::canonical_order() {
            hasher.update(self.get(id)?.digest);
        }
        Ok(hasher.finalize().into())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendPackMeta {
    pub schema_version: u16,
    pub pack_name: &'static str,
    pub pack_id: BackendPackId,
    pub llm_backend: BackendComponentId,
    pub world_backend: BackendComponentId,
    pub sae_backend: BackendComponentId,
    pub ssm_backend: BackendComponentId,
    pub lfm_backend: BackendComponentId,
    pub fixtures_digest: [u8; 32],
    pub model_hashes_digest: [u8; 32],
    pub code_version: CodeVersionTag,
    pub digest: [u8; 32],
}

impl BackendPackMeta {
    pub fn canonical_digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.schema_version.to_le_bytes());
        hasher.update((self.pack_name.len() as u16).to_le_bytes());
        hasher.update(self.pack_name.as_bytes());
        hasher.update(self.pack_id.0.to_le_bytes());
        hasher.update([self.llm_backend as u8]);
        hasher.update([self.world_backend as u8]);
        hasher.update([self.sae_backend as u8]);
        hasher.update([self.ssm_backend as u8]);
        hasher.update([self.lfm_backend as u8]);
        hasher.update(self.fixtures_digest);
        hasher.update(self.model_hashes_digest);
        hasher.update((self.code_version.as_str().len() as u16).to_le_bytes());
        hasher.update(self.code_version.as_str().as_bytes());
        hasher.finalize().into()
    }
}

pub trait BackendPack: Send + Sync {
    fn meta(&self) -> &BackendPackMeta;
    fn model_slot_provenance(&self) -> &[ModelSlotProvenance];
    fn llm(&self) -> &dyn LlmInference;
    fn world(&self) -> &Mutex<Box<dyn WorldModelPredictor + Send + Sync>>;
    fn sae(&self) -> &dyn SaeExtractor;
    fn ssm(&self) -> &Mutex<Box<dyn SsmKernel + Send + Sync>>;
    fn lfm(&self) -> &Mutex<Box<dyn LfmKernel + Send + Sync>>;
    fn reset_session(&mut self, _seed: u64) {}
    fn supports_hot_swap(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactFailureCode {
    Disabled,
    MissingPath,
    MissingExpectedHash,
    HashMismatch,
    Oversized,
    PathViolation,
    ArtifactUnavailable,
    ArtifactVerificationFailed,
    ArtifactIncompatible,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotRuntimeStatus {
    Used,
    Disabled,
    Unavailable,
    VerificationFailed,
    Incompatible,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelSlotProvenance {
    pub slot: ModelSlot,
    pub stage: &'static str,
    pub required_for_pack: bool,
    pub status: SlotRuntimeStatus,
    pub code: Option<ArtifactFailureCode>,
    pub detail: Option<String>,
    pub resolved_path: Option<String>,
    pub hash_prefix: Option<String>,
    pub contract_version: Option<String>,
    pub format: Option<ModelFormat>,
}

pub struct UnifiedBackendPack {
    meta: BackendPackMeta,
    slot_provenance: Vec<ModelSlotProvenance>,
    llm: Arc<dyn LlmInference + Send + Sync>,
    world: Mutex<Box<dyn WorldModelPredictor + Send + Sync>>,
    sae: Arc<dyn SaeExtractor + Send + Sync>,
    ssm: Mutex<Box<dyn SsmKernel + Send + Sync>>,
    lfm: Mutex<Box<dyn LfmKernel + Send + Sync>>,
}

impl BackendPack for UnifiedBackendPack {
    fn meta(&self) -> &BackendPackMeta {
        &self.meta
    }

    fn model_slot_provenance(&self) -> &[ModelSlotProvenance] {
        &self.slot_provenance
    }

    fn llm(&self) -> &dyn LlmInference {
        self.llm.as_ref()
    }

    fn world(&self) -> &Mutex<Box<dyn WorldModelPredictor + Send + Sync>> {
        &self.world
    }

    fn sae(&self) -> &dyn SaeExtractor {
        self.sae.as_ref()
    }

    fn ssm(&self) -> &Mutex<Box<dyn SsmKernel + Send + Sync>> {
        &self.ssm
    }

    fn lfm(&self) -> &Mutex<Box<dyn LfmKernel + Send + Sync>> {
        &self.lfm
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendPackKind {
    StubV0,
    ToyV1,
    CandleToyV1,
    CandleLiquidV1,
    BurnToyV1,
    ToyLnnV1,
    WorkerV1,
    #[cfg(feature = "remote-compute")]
    RemoteV1,
}

impl BackendPackKind {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "stub_v0" => Some(Self::StubV0),
            "toy_v1" => Some(Self::ToyV1),
            "candle_toy_v1" => Some(Self::CandleToyV1),
            "candle_liquid_v1" => Some(Self::CandleLiquidV1),
            "burn_toy_v1" => Some(Self::BurnToyV1),
            "toy_lnn_v1" => Some(Self::ToyLnnV1),
            "worker_v1" => Some(Self::WorkerV1),
            #[cfg(feature = "remote-compute")]
            "remote_v1" => Some(Self::RemoteV1),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::StubV0 => "stub_v0",
            Self::ToyV1 => "toy_v1",
            Self::CandleToyV1 => "candle_toy_v1",
            Self::CandleLiquidV1 => "candle_liquid_v1",
            Self::BurnToyV1 => "burn_toy_v1",
            Self::ToyLnnV1 => "toy_lnn_v1",
            Self::WorkerV1 => "worker_v1",
            #[cfg(feature = "remote-compute")]
            Self::RemoteV1 => "remote_v1",
        }
    }

    pub fn id(self) -> BackendPackId {
        match self {
            Self::StubV0 => BackendPackId(0),
            Self::ToyV1 => BackendPackId(1),
            Self::CandleToyV1 => BackendPackId(2),
            Self::CandleLiquidV1 => BackendPackId(4),
            Self::BurnToyV1 => BackendPackId(3),
            Self::ToyLnnV1 => BackendPackId(5),
            Self::WorkerV1 => BackendPackId(6),
            #[cfg(feature = "remote-compute")]
            Self::RemoteV1 => BackendPackId(7),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendPackConfig {
    pub pack: BackendPackKind,
    pub seed: u64,
}

impl Default for BackendPackConfig {
    fn default() -> Self {
        Self {
            pack: BackendPackKind::ToyV1,
            seed: 0x5eed_u64,
        }
    }
}

impl BackendPackConfig {
    pub fn from_env() -> Result<Self, ComputeError> {
        let mut cfg = Self::default();
        if let Ok(value) = std::env::var("UCF_BACKEND_PACK") {
            cfg.pack =
                BackendPackKind::parse(&value).ok_or_else(|| ComputeError::InvalidInput {
                    reason: format!("unsupported UCF_BACKEND_PACK={value}"),
                })?;
        }
        if let Ok(value) = std::env::var("UCF_BACKEND_SEED") {
            cfg.seed = value
                .parse::<u64>()
                .map_err(|_| ComputeError::InvalidInput {
                    reason: format!("invalid UCF_BACKEND_SEED={value}"),
                })?;
        }
        Ok(cfg)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendSwapRequest {
    pub t_effective: u64,
    pub target_pack: BackendPackKind,
    pub seed: Option<u64>,
    pub reason: String,
}

pub struct BackendPackFactory;

impl BackendPackFactory {
    pub fn build(cfg: BackendPackConfig) -> Result<Arc<dyn BackendPack>, ComputeError> {
        crate::ReleaseFeatureMatrix::detect().validate_pack(cfg.pack)?;
        if cfg.pack == BackendPackKind::WorkerV1 {
            return WorkerBackendPack::build(cfg.seed);
        }

        let fixtures = FixtureManager;
        let fixtures_digest = fixtures.overall_digest()?;
        let model_store = match ModelStore::from_env_default() {
            Ok(store) => store,
            Err(err) => {
                if model_slots_enabled_from_env() {
                    return Err(ComputeError::InvalidInput {
                        reason: format!("model store startup validation failed: {err:?}"),
                    });
                }
                ModelStore {
                    allowlist_root: std::path::PathBuf::from("models"),
                    specs: std::collections::BTreeMap::new(),
                }
            }
        };
        enforce_promoted_only_for_enabled_slots(&model_store)?;
        let slot_provenance = resolve_slot_provenance(&model_store, cfg.pack);
        if let Some(blocking) = first_blocking_artifact_failure(cfg.pack, &slot_provenance) {
            return Err(ComputeError::InvalidInput {
                reason: format!(
                    "model slot {} rejected for stage {}: {:?} ({})",
                    blocking.slot.as_str(),
                    blocking.stage,
                    blocking
                        .code
                        .unwrap_or(ArtifactFailureCode::ArtifactUnavailable),
                    blocking
                        .detail
                        .as_deref()
                        .unwrap_or("no additional detail provided")
                ),
            });
        }
        let model_hashes_digest = model_store.model_hashes_digest();
        let (llm_component, world_component, sae_component, ssm_component) = match cfg.pack {
            BackendPackKind::StubV0 => (
                BackendComponentId::StubV0,
                BackendComponentId::StubV0,
                BackendComponentId::StubV0,
                BackendComponentId::StubV0,
            ),
            BackendPackKind::ToyV1 => (
                BackendComponentId::ToyV1,
                BackendComponentId::ToyV1,
                BackendComponentId::ToyV1,
                BackendComponentId::ToyV1,
            ),
            BackendPackKind::CandleToyV1 => (
                BackendComponentId::CandleToyV1,
                BackendComponentId::CandleJepaV1,
                BackendComponentId::CandleSaeV1,
                BackendComponentId::CandleSsmV1,
            ),
            BackendPackKind::CandleLiquidV1 => (
                BackendComponentId::ToyV1,
                BackendComponentId::ToyV1,
                BackendComponentId::ToyV1,
                BackendComponentId::ToyV1,
            ),
            BackendPackKind::BurnToyV1 => (
                BackendComponentId::BurnToyV1,
                BackendComponentId::BurnJepaV1,
                BackendComponentId::BurnSaeV1,
                BackendComponentId::BurnSsmV1,
            ),
            BackendPackKind::ToyLnnV1 | BackendPackKind::WorkerV1 => (
                BackendComponentId::ToyV1,
                BackendComponentId::ToyV1,
                BackendComponentId::ToyV1,
                BackendComponentId::ToyV1,
            ),
            #[cfg(feature = "remote-compute")]
            BackendPackKind::RemoteV1 => (
                BackendComponentId::RemoteProxyV1,
                BackendComponentId::RemoteProxyV1,
                BackendComponentId::RemoteProxyV1,
                BackendComponentId::RemoteProxyV1,
            ),
        };

        let _world_model_hash = model_store
            .verify_slot(ModelSlot::WorldJepa)
            .ok()
            .map(|slot| slot.sha256)
            .unwrap_or([0x21; 32]);
        let _sae_model_hash = model_store
            .verify_slot(ModelSlot::Sae)
            .ok()
            .map(|slot| slot.sha256)
            .unwrap_or([0x31; 32]);
        let _ssm_model_hash = model_store
            .verify_slot(ModelSlot::Ssm)
            .ok()
            .map(|slot| slot.sha256)
            .unwrap_or([0x41; 32]);

        let llm_cfg = match cfg.pack {
            BackendPackKind::StubV0 | BackendPackKind::ToyV1 => LlmBackendConfig {
                kind: crate::capabilities::LlmBackendKind::Stub,
                seed: cfg.seed,
                max_tokens: 128,
                tokenizer_path: None,
                tokenizer_sha256: None,
            },
            BackendPackKind::CandleToyV1 => LlmBackendConfig {
                kind: crate::capabilities::LlmBackendKind::Candle,
                seed: cfg.seed,
                max_tokens: 128,
                tokenizer_path: None,
                tokenizer_sha256: None,
            },
            BackendPackKind::CandleLiquidV1
            | BackendPackKind::ToyLnnV1
            | BackendPackKind::WorkerV1 => LlmBackendConfig {
                kind: crate::capabilities::LlmBackendKind::Stub,
                seed: cfg.seed,
                max_tokens: 128,
                tokenizer_path: None,
                tokenizer_sha256: None,
            },
            #[cfg(feature = "remote-compute")]
            BackendPackKind::RemoteV1 => LlmBackendConfig {
                kind: crate::capabilities::LlmBackendKind::Stub,
                seed: cfg.seed,
                max_tokens: 128,
                tokenizer_path: None,
                tokenizer_sha256: None,
            },
            BackendPackKind::BurnToyV1 => LlmBackendConfig {
                kind: crate::capabilities::LlmBackendKind::Burn,
                seed: cfg.seed,
                max_tokens: 128,
                tokenizer_path: None,
                tokenizer_sha256: None,
            },
        };

        #[cfg(any(feature = "compute-candle", feature = "llm-candle"))]
        let llm: Arc<dyn LlmInference + Send + Sync> =
            if matches!(cfg.pack, BackendPackKind::CandleToyV1) {
                if let Ok(verified_llm) = model_store.verify_slot(ModelSlot::Llm) {
                    let tokenizer_path = std::env::var("UCF_LLM_TOKENIZER_PATH")
                        .map(std::path::PathBuf::from)
                        .unwrap_or_else(|_| DEFAULT_LLM_TOKENIZER_PATH.into());
                    let tokenizer_sha = std::env::var("UCF_LLM_TOKENIZER_SHA256")
                        .ok()
                        .and_then(|hex| decode_hash(&hex).ok())
                        .or_else(|| decode_hash(DEFAULT_LLM_TOKENIZER_SHA256).ok())
                        .unwrap_or([0_u8; 32]);
                    build_candle_llm_from_slot(&verified_llm, &tokenizer_path, tokenizer_sha)
                        .or_else(|_| build_llm_backend(llm_cfg))
                        .unwrap_or_else(|_| Arc::new(LlmStubBackend))
                } else {
                    build_llm_backend(llm_cfg).unwrap_or_else(|_| Arc::new(LlmStubBackend))
                }
            } else {
                build_llm_backend(llm_cfg).unwrap_or_else(|_| Arc::new(LlmStubBackend))
            };
        #[cfg(not(any(feature = "compute-candle", feature = "llm-candle")))]
        let llm: Arc<dyn LlmInference + Send + Sync> =
            build_llm_backend(llm_cfg).unwrap_or_else(|_| Arc::new(LlmStubBackend));
        let (lfm_component, lfm_kernel): (BackendComponentId, Box<dyn LfmKernel + Send + Sync>) =
            match cfg.pack {
                BackendPackKind::StubV0 | BackendPackKind::ToyV1 => {
                    (BackendComponentId::ToyV1, Box::new(ToyLfmKernel::default()))
                }
                BackendPackKind::CandleToyV1 | BackendPackKind::CandleLiquidV1 => {
                    #[cfg(feature = "lfm-candle")]
                    {
                        (
                            BackendComponentId::CandleToyV1,
                            Box::new(CandleLfmKernel::default()),
                        )
                    }
                    #[cfg(not(feature = "lfm-candle"))]
                    {
                        return Err(ComputeError::BackendDisabled);
                    }
                }
                BackendPackKind::BurnToyV1 => (
                    BackendComponentId::Disabled,
                    Box::new(ToyLfmKernel::default()),
                ),
                BackendPackKind::ToyLnnV1 => {
                    #[cfg(feature = "lfm-lnn")]
                    {
                        (
                            BackendComponentId::LnnOdeV1,
                            Box::new(LnnOdeLfmKernel::default()),
                        )
                    }
                    #[cfg(not(feature = "lfm-lnn"))]
                    {
                        return Err(ComputeError::BackendDisabled);
                    }
                }
                BackendPackKind::WorkerV1 => {
                    return Err(ComputeError::Internal {
                        reason: "worker pack handled above".to_string(),
                    })
                }
                #[cfg(feature = "remote-compute")]
                BackendPackKind::RemoteV1 => {
                    return Self::build_remote_pack(cfg.seed, fixtures_digest, model_hashes_digest)
                }
            };

        let mut meta = BackendPackMeta {
            schema_version: 1,
            pack_name: cfg.pack.as_str(),
            pack_id: cfg.pack.id(),
            llm_backend: llm_component,
            world_backend: world_component,
            sae_backend: sae_component,
            ssm_backend: ssm_component,
            lfm_backend: lfm_component,
            fixtures_digest,
            model_hashes_digest,
            code_version: CodeVersionTag::current(),
            digest: [0; 32],
        };
        meta.digest = meta.canonical_digest();

        let world_backend: Box<dyn WorldModelPredictor + Send + Sync> = match cfg.pack {
            BackendPackKind::CandleToyV1 => {
                #[cfg(feature = "compute-candle")]
                {
                    Box::new(CandleWorldPredictor::new(_world_model_hash))
                }
                #[cfg(not(feature = "compute-candle"))]
                {
                    return Err(ComputeError::BackendDisabled);
                }
            }
            BackendPackKind::BurnToyV1 => {
                #[cfg(feature = "compute-burn")]
                {
                    Box::new(BurnWorldPredictor::new(_world_model_hash))
                }
                #[cfg(not(feature = "compute-burn"))]
                {
                    return Err(ComputeError::BackendDisabled);
                }
            }
            _ => Box::new(MockJepaPredictor::default()),
        };

        let sae_backend: Arc<dyn SaeExtractor + Send + Sync> = match cfg.pack {
            BackendPackKind::CandleToyV1 => {
                #[cfg(feature = "compute-candle")]
                {
                    Arc::new(CandleSaeExtractor::new(_sae_model_hash))
                }
                #[cfg(not(feature = "compute-candle"))]
                {
                    return Err(ComputeError::BackendDisabled);
                }
            }
            BackendPackKind::BurnToyV1 => {
                #[cfg(feature = "compute-burn")]
                {
                    Arc::new(BurnSaeExtractor::new(_sae_model_hash))
                }
                #[cfg(not(feature = "compute-burn"))]
                {
                    return Err(ComputeError::BackendDisabled);
                }
            }
            _ => Arc::new(ToySaeExtractor::default()),
        };

        let ssm_backend: Box<dyn SsmKernel + Send + Sync> = match cfg.pack {
            BackendPackKind::CandleToyV1 => {
                #[cfg(feature = "compute-candle")]
                {
                    Box::new(CandleSsmKernel::from_model_store_or_hash(
                        &model_store,
                        _ssm_model_hash,
                    )?)
                }
                #[cfg(not(feature = "compute-candle"))]
                {
                    return Err(ComputeError::BackendDisabled);
                }
            }
            BackendPackKind::BurnToyV1 => {
                #[cfg(feature = "compute-burn")]
                {
                    Box::new(BurnSsmKernel::new(_ssm_model_hash))
                }
                #[cfg(not(feature = "compute-burn"))]
                {
                    return Err(ComputeError::BackendDisabled);
                }
            }
            _ => Box::new(ToySsmKernel::default()),
        };

        Ok(Arc::new(UnifiedBackendPack {
            meta,
            slot_provenance,
            llm,
            world: Mutex::new(world_backend),
            sae: sae_backend,
            ssm: Mutex::new(ssm_backend),
            lfm: Mutex::new(lfm_kernel),
        }))
    }
}

fn first_blocking_artifact_failure(
    pack: BackendPackKind,
    provenance: &[ModelSlotProvenance],
) -> Option<ModelSlotProvenance> {
    required_slots_for_pack(pack).into_iter().find_map(|slot| {
        provenance
            .iter()
            .find(|p| p.slot == slot)
            .and_then(|entry| match entry.status {
                SlotRuntimeStatus::Unavailable
                | SlotRuntimeStatus::VerificationFailed
                | SlotRuntimeStatus::Incompatible => Some(entry.clone()),
                SlotRuntimeStatus::Used | SlotRuntimeStatus::Disabled => None,
            })
    })
}

fn required_slots_for_pack(pack: BackendPackKind) -> Vec<ModelSlot> {
    match pack {
        BackendPackKind::CandleToyV1 | BackendPackKind::BurnToyV1 => {
            vec![ModelSlot::WorldJepa, ModelSlot::Sae, ModelSlot::Ssm]
        }
        _ => Vec::new(),
    }
}

fn resolve_slot_provenance(store: &ModelStore, pack: BackendPackKind) -> Vec<ModelSlotProvenance> {
    ModelSlot::all()
        .into_iter()
        .map(|slot| {
            let Some(spec) = store.specs.get(&slot) else {
                return ModelSlotProvenance {
                    slot,
                    stage: expected_stage_for_slot(slot),
                    required_for_pack: required_slots_for_pack(pack).contains(&slot),
                    status: SlotRuntimeStatus::Disabled,
                    code: Some(ArtifactFailureCode::Disabled),
                    detail: Some("slot missing from manifest spec map".to_string()),
                    resolved_path: None,
                    hash_prefix: None,
                    contract_version: None,
                    format: None,
                };
            };
            if !spec.enabled {
                return ModelSlotProvenance {
                    slot,
                    stage: expected_stage_for_slot(slot),
                    required_for_pack: required_slots_for_pack(pack).contains(&slot),
                    status: SlotRuntimeStatus::Disabled,
                    code: Some(ArtifactFailureCode::Disabled),
                    detail: Some("slot disabled by manifest/env".to_string()),
                    resolved_path: None,
                    hash_prefix: None,
                    contract_version: spec.contract_version.clone(),
                    format: Some(spec.format),
                };
            }
            match store.verify_slot(slot) {
                Ok(verified) => {
                    let (status, code, detail) = check_slot_compatibility(
                        pack,
                        slot,
                        spec.format,
                        spec.contract_version.as_deref(),
                    );
                    ModelSlotProvenance {
                        slot,
                        stage: expected_stage_for_slot(slot),
                        required_for_pack: required_slots_for_pack(pack).contains(&slot),
                        status,
                        code,
                        detail,
                        resolved_path: Some(verified.path.display().to_string()),
                        hash_prefix: Some(hex::encode(&verified.sha256[..6])),
                        contract_version: verified.contract_version.clone(),
                        format: Some(verified.format),
                    }
                }
                Err(err) => {
                    let (status, code, detail) = classify_model_error(err);
                    ModelSlotProvenance {
                        slot,
                        stage: expected_stage_for_slot(slot),
                        required_for_pack: required_slots_for_pack(pack).contains(&slot),
                        status,
                        code: Some(code),
                        detail: Some(detail),
                        resolved_path: None,
                        hash_prefix: None,
                        contract_version: spec.contract_version.clone(),
                        format: Some(spec.format),
                    }
                }
            }
        })
        .collect()
}

fn check_slot_compatibility(
    pack: BackendPackKind,
    slot: ModelSlot,
    format: ModelFormat,
    contract_version: Option<&str>,
) -> (
    SlotRuntimeStatus,
    Option<ArtifactFailureCode>,
    Option<String>,
) {
    let expected_formats: &[ModelFormat] = match (pack, slot) {
        (BackendPackKind::CandleToyV1, ModelSlot::Llm)
        | (BackendPackKind::CandleToyV1, ModelSlot::WorldJepa)
        | (BackendPackKind::CandleToyV1, ModelSlot::Sae)
        | (BackendPackKind::CandleToyV1, ModelSlot::Ssm) => &[ModelFormat::CandleSafetensors],
        (BackendPackKind::BurnToyV1, ModelSlot::WorldJepa)
        | (BackendPackKind::BurnToyV1, ModelSlot::Sae)
        | (BackendPackKind::BurnToyV1, ModelSlot::Ssm) => &[ModelFormat::Burn],
        _ => &[],
    };
    if !expected_formats.is_empty() && !expected_formats.contains(&format) {
        return (
            SlotRuntimeStatus::Incompatible,
            Some(ArtifactFailureCode::ArtifactIncompatible),
            Some(format!(
                "slot format {:?} incompatible with backend pack {}",
                format,
                pack.as_str()
            )),
        );
    }
    if required_slots_for_pack(pack).contains(&slot) {
        let version = contract_version.unwrap_or("v1");
        if version != "v1" && version != "1" {
            return (
                SlotRuntimeStatus::Incompatible,
                Some(ArtifactFailureCode::ArtifactIncompatible),
                Some(format!(
                    "slot contract_version {version} incompatible with expected v1"
                )),
            );
        }
    }
    (SlotRuntimeStatus::Used, None, None)
}

fn classify_model_error(err: ModelLoadError) -> (SlotRuntimeStatus, ArtifactFailureCode, String) {
    match err {
        ModelLoadError::Disabled => (
            SlotRuntimeStatus::Disabled,
            ArtifactFailureCode::Disabled,
            "slot disabled".to_string(),
        ),
        ModelLoadError::MissingPath => (
            SlotRuntimeStatus::Unavailable,
            ArtifactFailureCode::MissingPath,
            "slot missing path".to_string(),
        ),
        ModelLoadError::MissingExpectedHash { slot } => (
            SlotRuntimeStatus::VerificationFailed,
            ArtifactFailureCode::MissingExpectedHash,
            format!("missing expected hash for {}", slot.as_str()),
        ),
        ModelLoadError::HashMismatch { .. } => (
            SlotRuntimeStatus::VerificationFailed,
            ArtifactFailureCode::HashMismatch,
            "model hash mismatch".to_string(),
        ),
        ModelLoadError::Oversized { .. } => (
            SlotRuntimeStatus::VerificationFailed,
            ArtifactFailureCode::Oversized,
            "model exceeds max_bytes".to_string(),
        ),
        ModelLoadError::PathOutsideAllowlist { .. } | ModelLoadError::PathTraversal { .. } => (
            SlotRuntimeStatus::VerificationFailed,
            ArtifactFailureCode::PathViolation,
            "model path violates allowlist root".to_string(),
        ),
        ModelLoadError::OpenFailed { reason, .. } => (
            SlotRuntimeStatus::Unavailable,
            ArtifactFailureCode::ArtifactUnavailable,
            format!("artifact unavailable: {reason}"),
        ),
        ModelLoadError::ManifestParse(reason) => (
            SlotRuntimeStatus::Unavailable,
            ArtifactFailureCode::ArtifactUnavailable,
            format!("manifest parse failed: {reason}"),
        ),
    }
}

fn expected_stage_for_slot(slot: ModelSlot) -> &'static str {
    match slot {
        ModelSlot::Llm => "llm",
        ModelSlot::WorldJepa | ModelSlot::WorldVljepa => "world",
        ModelSlot::Sae => "sae",
        ModelSlot::Lfm => "lfm",
        ModelSlot::Ssm => "ssm",
        ModelSlot::EbmReasoner => "ebm_reasoner",
    }
}

fn enforce_promoted_only_for_enabled_slots(store: &ModelStore) -> Result<(), ComputeError> {
    for slot in ModelSlot::all() {
        let Some(spec) = store.specs.get(&slot) else {
            continue;
        };
        if !spec.enabled {
            continue;
        }
        let pin_set = std::env::var(format!("UCF_MODEL_PIN_{}", slot.env_key()))
            .ok()
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false);
        if !pin_set
            && spec
                .active_hash
                .as_ref()
                .map(|v| v.is_empty())
                .unwrap_or(true)
        {
            return Err(ComputeError::InvalidInput {
                reason: format!(
                    "enabled slot {} requires active_hash or UCF_MODEL_PIN_{}",
                    slot.as_str(),
                    slot.env_key()
                ),
            });
        }
    }
    Ok(())
}
fn model_slots_enabled_from_env() -> bool {
    ModelSlot::all().into_iter().any(|slot| {
        std::env::var(format!("UCF_MODEL_{}_ENABLED", slot.env_key()))
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
            .unwrap_or(false)
    })
}

#[cfg(any(feature = "compute-candle", feature = "llm-candle"))]
fn decode_hash(hex_value: &str) -> Result<[u8; 32], ()> {
    let bytes = hex::decode(hex_value).map_err(|_| ())?;
    if bytes.len() != 32 {
        return Err(());
    }
    let mut out = [0_u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}

#[cfg(feature = "remote-compute")]
impl BackendPackFactory {
    fn build_remote_pack(
        seed: u64,
        fixtures_digest: [u8; 32],
        model_hashes_digest: [u8; 32],
    ) -> Result<Arc<dyn BackendPack>, ComputeError> {
        let remote_enable = std::env::var("UCF_REMOTE_ENABLE").unwrap_or_default();
        if remote_enable != "1" {
            return Err(ComputeError::BackendDisabled);
        }
        let allowlist_path = std::path::Path::new("policies/bundle_v1/allowlists.json");
        let policy_hash = std::env::var("UCF_POLICY_BUNDLE_SHA256").unwrap_or_default();
        let allowlist = crate::remote_compute::RemotePolicyAllowlist::load(allowlist_path)?;
        if !allowlist.enabled || !allowlist.allows_policy_hash(&policy_hash) {
            return Err(ComputeError::BackendDisabled);
        }

        let mut meta = BackendPackMeta {
            schema_version: 1,
            pack_name: "remote_v1",
            pack_id: BackendPackKind::RemoteV1.id(),
            llm_backend: BackendComponentId::RemoteProxyV1,
            world_backend: BackendComponentId::RemoteProxyV1,
            sae_backend: BackendComponentId::RemoteProxyV1,
            ssm_backend: BackendComponentId::RemoteProxyV1,
            lfm_backend: BackendComponentId::RemoteProxyV1,
            fixtures_digest,
            model_hashes_digest,
            code_version: CodeVersionTag::current(),
            digest: [0; 32],
        };
        meta.digest = meta.canonical_digest();
        Ok(Arc::new(RemoteBackendPack {
            inner: UnifiedBackendPack {
                meta,
                slot_provenance: Vec::new(),
                llm: Arc::new(LlmStubBackend),
                world: Mutex::new(Box::new(MockJepaPredictor::default())),
                sae: Arc::new(ToySaeExtractor::default()),
                ssm: Mutex::new(Box::new(ToySsmKernel::default())),
                lfm: Mutex::new(Box::new(ToyLfmKernel::default())),
            },
            _seed: seed,
        }))
    }
}

#[cfg(feature = "remote-compute")]
pub struct RemoteBackendPack {
    inner: UnifiedBackendPack,
    _seed: u64,
}

#[cfg(feature = "remote-compute")]
impl BackendPack for RemoteBackendPack {
    fn meta(&self) -> &BackendPackMeta {
        self.inner.meta()
    }
    fn model_slot_provenance(&self) -> &[ModelSlotProvenance] {
        self.inner.model_slot_provenance()
    }
    fn llm(&self) -> &dyn LlmInference {
        self.inner.llm()
    }
    fn world(&self) -> &Mutex<Box<dyn WorldModelPredictor + Send + Sync>> {
        self.inner.world()
    }
    fn sae(&self) -> &dyn SaeExtractor {
        self.inner.sae()
    }
    fn ssm(&self) -> &Mutex<Box<dyn SsmKernel + Send + Sync>> {
        self.inner.ssm()
    }
    fn lfm(&self) -> &Mutex<Box<dyn LfmKernel + Send + Sync>> {
        self.inner.lfm()
    }
}

pub fn slot_verified_or_reason(slot: ModelSlot) -> Result<(), String> {
    let store = ModelStore::from_env_default().map_err(|e| format!("{e:?}"))?;
    match store.verify_slot(slot) {
        Ok(_) => Ok(()),
        Err(ModelLoadError::Disabled) => Err("disabled".to_string()),
        Err(err) => Err(format!("{err:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::fs;

    #[test]
    fn fixture_digest_stable() {
        let manager = FixtureManager;
        assert_eq!(
            manager.overall_digest().expect("digest"),
            manager.overall_digest().expect("digest")
        );
    }

    #[test]
    fn factory_deterministic() {
        let _guard = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _env = crate::test_env::clear_model_env_overrides();
        let a = BackendPackFactory::build(BackendPackConfig::default()).expect("a");
        let b = BackendPackFactory::build(BackendPackConfig::default()).expect("b");
        assert_eq!(a.meta().digest, b.meta().digest);
    }

    #[test]
    fn parse_candle_liquid_pack() {
        assert_eq!(
            BackendPackKind::parse("candle_liquid_v1"),
            Some(BackendPackKind::CandleLiquidV1)
        );
    }

    #[test]
    fn parse_toy_lnn_pack() {
        assert_eq!(
            BackendPackKind::parse("toy_lnn_v1"),
            Some(BackendPackKind::ToyLnnV1)
        );
    }

    #[test]
    fn toy_lnn_feature_gate_behavior() {
        let _guard = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _env = crate::test_env::clear_model_env_overrides();
        let cfg = BackendPackConfig {
            pack: BackendPackKind::ToyLnnV1,
            seed: 5,
        };
        #[cfg(not(feature = "lfm-lnn"))]
        {
            let result = BackendPackFactory::build(cfg);
            assert!(matches!(
                result,
                Err(ComputeError::BackendDisabled) | Err(ComputeError::InvalidInput { .. })
            ));
        }
        #[cfg(feature = "lfm-lnn")]
        {
            let pack = BackendPackFactory::build(cfg).expect("pack");
            assert_eq!(pack.meta().lfm_backend, BackendComponentId::LnnOdeV1);
            assert_eq!(pack.meta().pack_name, "toy_lnn_v1");
        }
    }

    #[test]
    fn candle_liquid_feature_gate_behavior() {
        let _guard = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _env = crate::test_env::clear_model_env_overrides();
        let cfg = BackendPackConfig {
            pack: BackendPackKind::CandleLiquidV1,
            seed: 5,
        };
        #[cfg(not(feature = "lfm-candle"))]
        {
            let result = BackendPackFactory::build(cfg);
            assert!(matches!(
                result,
                Err(ComputeError::BackendDisabled) | Err(ComputeError::InvalidInput { .. })
            ));
        }
        #[cfg(feature = "lfm-candle")]
        {
            let pack = BackendPackFactory::build(cfg).expect("pack");
            assert_eq!(pack.meta().lfm_backend, BackendComponentId::CandleToyV1);
            assert_eq!(pack.meta().pack_name, "candle_liquid_v1");
        }
    }

    #[cfg(feature = "remote-compute")]
    #[test]
    fn remote_pack_requires_runtime_and_policy_opt_in() {
        std::env::remove_var("UCF_REMOTE_ENABLE");
        let cfg = BackendPackConfig {
            pack: BackendPackKind::RemoteV1,
            seed: 42,
        };
        let result = BackendPackFactory::build(cfg);
        assert!(matches!(result, Err(ComputeError::BackendDisabled)));
    }

    #[test]
    fn enabled_slot_requires_manifest_at_startup() {
        let _guard = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _env = crate::test_env::clear_model_env_overrides();
        let dir = tempfile::tempdir().expect("tempdir");
        let missing_manifest = dir.path().join("missing-manifest.toml");
        std::env::set_var("UCF_MODEL_LLM_ENABLED", "true");
        std::env::set_var("UCF_MODEL_MANIFEST", &missing_manifest);

        let res = BackendPackFactory::build(BackendPackConfig::default());
        assert!(matches!(res, Err(ComputeError::InvalidInput { .. })));
    }

    #[test]
    fn provenance_marks_valid_required_slot_as_used() {
        let temp = tempfile::tempdir().expect("tempdir");
        let models = temp.path().join("models");
        fs::create_dir_all(&models).expect("models dir");
        let bytes = b"world-jepa";
        let hash = hex::encode(Sha256::digest(bytes));
        let model_path = models
            .join("promoted")
            .join("world_jepa")
            .join(&hash)
            .join("model.safetensors");
        fs::create_dir_all(model_path.parent().expect("parent")).expect("mkdirs");
        fs::write(&model_path, bytes).expect("write");
        let manifest_path = temp.path().join("manifest.toml");
        fs::write(
            &manifest_path,
            format!(
                "allowlist_root = '{}'\n[slots.world_jepa]\nenabled = true\nexpected_sha256 = \"{}\"\nactive_hash = \"{}\"\nformat = \"burn\"\ncontract_version = \"v1\"\n",
                models.display(),
                hash,
                hash
            ),
        )
        .expect("manifest");
        let store = ModelStore::from_manifest_and_env(&manifest_path).expect("store");
        let provenance = resolve_slot_provenance(&store, BackendPackKind::BurnToyV1);
        let world = provenance
            .iter()
            .find(|entry| entry.slot == ModelSlot::WorldJepa)
            .expect("world slot");
        assert!(world.required_for_pack);
        assert_eq!(world.status, SlotRuntimeStatus::Used);
        assert!(world.hash_prefix.is_some());
    }

    #[test]
    fn provenance_marks_missing_expected_hash_as_verification_failed() {
        let temp = tempfile::tempdir().expect("tempdir");
        let models = temp.path().join("models");
        fs::create_dir_all(&models).expect("models dir");
        let bytes = b"world-jepa";
        let hash = hex::encode(Sha256::digest(bytes));
        let model_path = models
            .join("promoted")
            .join("world_jepa")
            .join(&hash)
            .join("model.safetensors");
        fs::create_dir_all(model_path.parent().expect("parent")).expect("mkdirs");
        fs::write(&model_path, bytes).expect("write");
        let manifest_path = temp.path().join("manifest.toml");
        fs::write(
            &manifest_path,
            format!(
                "allowlist_root = '{}'\n[slots.world_jepa]\nenabled = true\npath = \"world_jepa.bin\"\nformat = \"burn\"\ncontract_version = \"v1\"\n",
                models.display(),
            ),
        )
        .expect("manifest");
        fs::write(models.join("world_jepa.bin"), bytes).expect("artifact");
        let store = ModelStore::from_manifest_and_env(&manifest_path).expect("store");
        let provenance = resolve_slot_provenance(&store, BackendPackKind::BurnToyV1);
        let world = provenance
            .iter()
            .find(|entry| entry.slot == ModelSlot::WorldJepa)
            .expect("world slot");
        assert_eq!(world.status, SlotRuntimeStatus::VerificationFailed);
        assert_eq!(world.code, Some(ArtifactFailureCode::MissingExpectedHash));
    }

    #[test]
    fn provenance_distinguishes_disabled_from_incompatible() {
        let temp = tempfile::tempdir().expect("tempdir");
        let models = temp.path().join("models");
        fs::create_dir_all(&models).expect("models dir");
        let bytes = b"weights";
        let hash = hex::encode(Sha256::digest(bytes));
        let model_path = models
            .join("promoted")
            .join("world_jepa")
            .join(&hash)
            .join("model.safetensors");
        fs::create_dir_all(model_path.parent().expect("parent")).expect("mkdirs");
        fs::write(&model_path, bytes).expect("write");
        let manifest_path = temp.path().join("manifest.toml");
        fs::write(
            &manifest_path,
            format!(
                "allowlist_root = '{}'\n[slots.world_jepa]\nenabled = true\nexpected_sha256 = \"{}\"\nactive_hash = \"{}\"\nformat = \"burn\"\ncontract_version = \"v1\"\n[slots.sae]\nenabled = false\n",
                models.display(),
                hash,
                hash
            ),
        )
        .expect("manifest");
        let store = ModelStore::from_manifest_and_env(&manifest_path).expect("store");
        let provenance = resolve_slot_provenance(&store, BackendPackKind::CandleToyV1);
        let world = provenance
            .iter()
            .find(|entry| entry.slot == ModelSlot::WorldJepa)
            .expect("world slot");
        assert_eq!(world.status, SlotRuntimeStatus::Incompatible);
        let sae = provenance
            .iter()
            .find(|entry| entry.slot == ModelSlot::Sae)
            .expect("sae slot");
        assert_eq!(sae.status, SlotRuntimeStatus::Disabled);
    }
}
