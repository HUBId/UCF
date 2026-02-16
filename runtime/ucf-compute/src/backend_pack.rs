use std::sync::{Arc, Mutex};

use sha2::{Digest, Sha256};

use crate::capabilities::{
    build_llm_backend, LlmBackendConfig, LlmInference, LlmStubBackend, SaeExtractor,
    WorldModelPredictor,
};
use crate::feature_extractor::ToySaeExtractor;
#[cfg(feature = "lfm-burn")]
use crate::lfm::BurnLfmKernel;
#[cfg(feature = "lfm-candle")]
use crate::lfm::CandleLfmKernel;
use crate::lfm::{LfmKernel, ToyLfmKernel};
use crate::ssm::{SsmKernel, ToySsmKernel};
use crate::world_model::MockJepaPredictor;
use crate::{CodeVersionTag, ComputeError};

const FIXTURE_SCHEMA_V1: u16 = 1;
const MAX_FIXTURE_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BackendPackId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum BackendComponentId {
    StubV0 = 0,
    ToyV1 = 1,
    CandleToyV1 = 2,
    BurnToyV1 = 3,
    Disabled = 255,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixtureId {
    ToyLlmWeightsV1,
    JepaDynV1,
    SaeProjV1,
    SsmParamsV1,
    LfmParamsV1,
}

impl FixtureId {
    pub const fn canonical_order() -> [Self; 5] {
        [
            Self::ToyLlmWeightsV1,
            Self::JepaDynV1,
            Self::SaeProjV1,
            Self::SsmParamsV1,
            Self::LfmParamsV1,
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
        hasher.update((self.code_version.as_str().len() as u16).to_le_bytes());
        hasher.update(self.code_version.as_str().as_bytes());
        hasher.finalize().into()
    }
}

pub trait BackendPack: Send + Sync {
    fn meta(&self) -> &BackendPackMeta;
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

pub struct UnifiedBackendPack {
    meta: BackendPackMeta,
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
}

impl BackendPackKind {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "stub_v0" => Some(Self::StubV0),
            "toy_v1" => Some(Self::ToyV1),
            "candle_toy_v1" => Some(Self::CandleToyV1),
            "candle_liquid_v1" => Some(Self::CandleLiquidV1),
            "burn_toy_v1" => Some(Self::BurnToyV1),
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
        }
    }

    pub fn id(self) -> BackendPackId {
        match self {
            Self::StubV0 => BackendPackId(0),
            Self::ToyV1 => BackendPackId(1),
            Self::CandleToyV1 => BackendPackId(2),
            Self::CandleLiquidV1 => BackendPackId(4),
            Self::BurnToyV1 => BackendPackId(3),
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
        let fixtures = FixtureManager;
        let fixtures_digest = fixtures.overall_digest()?;
        let (llm_component, sae_component) = match cfg.pack {
            BackendPackKind::StubV0 => (BackendComponentId::StubV0, BackendComponentId::StubV0),
            BackendPackKind::ToyV1 => (BackendComponentId::ToyV1, BackendComponentId::ToyV1),
            BackendPackKind::CandleToyV1 => (
                BackendComponentId::CandleToyV1,
                BackendComponentId::CandleToyV1,
            ),
            BackendPackKind::CandleLiquidV1 => {
                (BackendComponentId::ToyV1, BackendComponentId::ToyV1)
            }
            BackendPackKind::BurnToyV1 => {
                (BackendComponentId::BurnToyV1, BackendComponentId::BurnToyV1)
            }
        };

        let llm_cfg = match cfg.pack {
            BackendPackKind::StubV0 | BackendPackKind::ToyV1 => LlmBackendConfig {
                kind: crate::capabilities::LlmBackendKind::Stub,
                seed: cfg.seed,
                max_tokens: 128,
            },
            BackendPackKind::CandleToyV1 => LlmBackendConfig {
                kind: crate::capabilities::LlmBackendKind::Candle,
                seed: cfg.seed,
                max_tokens: 128,
            },
            BackendPackKind::CandleLiquidV1 => LlmBackendConfig {
                kind: crate::capabilities::LlmBackendKind::Stub,
                seed: cfg.seed,
                max_tokens: 128,
            },
            BackendPackKind::BurnToyV1 => LlmBackendConfig {
                kind: crate::capabilities::LlmBackendKind::Burn,
                seed: cfg.seed,
                max_tokens: 128,
            },
        };

        let llm = build_llm_backend(llm_cfg).unwrap_or_else(|_| Arc::new(LlmStubBackend));
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
                BackendPackKind::BurnToyV1 => {
                    #[cfg(feature = "lfm-burn")]
                    {
                        (BackendComponentId::BurnToyV1, Box::new(BurnLfmKernel))
                    }
                    #[cfg(not(feature = "lfm-burn"))]
                    {
                        return Err(ComputeError::BackendDisabled);
                    }
                }
            };

        let mut meta = BackendPackMeta {
            schema_version: 1,
            pack_name: cfg.pack.as_str(),
            pack_id: cfg.pack.id(),
            llm_backend: llm_component,
            world_backend: BackendComponentId::ToyV1,
            sae_backend: sae_component,
            ssm_backend: BackendComponentId::ToyV1,
            lfm_backend: lfm_component,
            fixtures_digest,
            code_version: CodeVersionTag::current(),
            digest: [0; 32],
        };
        meta.digest = meta.canonical_digest();

        Ok(Arc::new(UnifiedBackendPack {
            meta,
            llm,
            world: Mutex::new(Box::new(MockJepaPredictor::default())),
            sae: Arc::new(ToySaeExtractor::default()),
            ssm: Mutex::new(Box::new(ToySsmKernel::default())),
            lfm: Mutex::new(lfm_kernel),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn candle_liquid_feature_gate_behavior() {
        let cfg = BackendPackConfig {
            pack: BackendPackKind::CandleLiquidV1,
            seed: 5,
        };
        #[cfg(not(feature = "lfm-candle"))]
        {
            let result = BackendPackFactory::build(cfg);
            assert!(matches!(result, Err(ComputeError::BackendDisabled)));
        }
        #[cfg(feature = "lfm-candle")]
        {
            let pack = BackendPackFactory::build(cfg).expect("pack");
            assert_eq!(pack.meta().lfm_backend, BackendComponentId::CandleToyV1);
            assert_eq!(pack.meta().pack_name, "candle_liquid_v1");
        }
    }
}
