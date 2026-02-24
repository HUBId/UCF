use std::collections::BTreeMap;

#[cfg(any(
    feature = "compute-candle",
    feature = "llm-candle",
    feature = "lfm-candle"
))]
use candle_core::{DType as CandleDType, Device, Tensor};
use safetensors::tensor::SafeTensors;

use crate::model_store::{ModelSlot, VerifiedModelSlot};
use crate::{ComputeError, ModelLoadError, ModelStore};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F16,
    BF16,
    F32,
    I32,
    U8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimExpr {
    Fixed(usize),
    Var(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: &'static str,
    pub shape: &'static [DimExpr],
    pub dtype: DType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightSpec {
    pub slot: ModelSlot,
    pub tensors: &'static [TensorSpec],
    pub optional: &'static [TensorSpec],
    pub max_bytes: u64,
    pub bindings: BTreeMap<&'static str, usize>,
}

impl WeightSpec {
    pub fn with_bindings(mut self, bindings: BTreeMap<&'static str, usize>) -> Self {
        self.bindings = bindings;
        self
    }
}

#[derive(Debug, Clone)]
pub struct LoadedWeightsRaw {
    pub slot: ModelSlot,
    pub tensors: BTreeMap<String, LoadedTensorRaw>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoadedTensorRaw {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub values_f32: Vec<f32>,
}

#[cfg(any(
    feature = "compute-candle",
    feature = "llm-candle",
    feature = "lfm-candle"
))]
#[derive(Debug, Clone)]
pub struct LoadedWeights {
    pub slot: ModelSlot,
    pub tensors: BTreeMap<String, Tensor>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightErr {
    MissingTensor {
        code: &'static str,
        name: String,
    },
    ShapeMismatch {
        code: &'static str,
        name: String,
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    DTypeMismatch {
        code: &'static str,
        name: String,
        expected: DType,
        found: String,
    },
    TooLarge {
        code: &'static str,
        max_bytes: u64,
        found_bytes: u64,
    },
    ParseError {
        code: &'static str,
        reason: String,
    },
    HashMismatch {
        code: &'static str,
    },
}

impl WeightErr {
    pub fn code(&self) -> &'static str {
        match self {
            Self::MissingTensor { code, .. }
            | Self::ShapeMismatch { code, .. }
            | Self::DTypeMismatch { code, .. }
            | Self::TooLarge { code, .. }
            | Self::ParseError { code, .. }
            | Self::HashMismatch { code } => code,
        }
    }
}

pub fn load_safetensors_raw(
    slot: ModelSlot,
    bytes: &[u8],
    spec: &WeightSpec,
) -> Result<LoadedWeightsRaw, WeightErr> {
    if bytes.len() as u64 > spec.max_bytes {
        return Err(WeightErr::TooLarge {
            code: "WEIGHT_TOO_LARGE",
            max_bytes: spec.max_bytes,
            found_bytes: bytes.len() as u64,
        });
    }
    let raw = SafeTensors::deserialize(bytes).map_err(|err| WeightErr::ParseError {
        code: "WEIGHT_PARSE_ERROR",
        reason: err.to_string(),
    })?;
    let mut bindings = spec.bindings.clone();
    let mut loaded: BTreeMap<String, LoadedTensorRaw> = BTreeMap::new();

    for required in spec.tensors {
        load_one(&raw, &mut bindings, required, &mut loaded)?;
    }
    for optional in spec.optional {
        if raw.tensor(optional.name).is_ok() {
            load_one(&raw, &mut bindings, optional, &mut loaded)?;
        }
    }

    Ok(LoadedWeightsRaw {
        slot,
        tensors: loaded,
    })
}

fn load_one(
    raw: &SafeTensors<'_>,
    bindings: &mut BTreeMap<&'static str, usize>,
    expected: &TensorSpec,
    loaded: &mut BTreeMap<String, LoadedTensorRaw>,
) -> Result<(), WeightErr> {
    let tensor_view = raw
        .tensor(expected.name)
        .map_err(|_| WeightErr::MissingTensor {
            code: "WEIGHT_MISSING_TENSOR",
            name: expected.name.to_string(),
        })?;

    let found_shape = tensor_view.shape().to_vec();
    let expected_shape = bind_shape(expected.shape, &found_shape, bindings);

    if expected_shape != found_shape {
        return Err(WeightErr::ShapeMismatch {
            code: "WEIGHT_SHAPE_MISMATCH",
            name: expected.name.to_string(),
            expected: expected_shape,
            found: found_shape,
        });
    }

    let expected_dtype = map_dtype(expected.dtype);
    if tensor_view.dtype() != expected_dtype {
        return Err(WeightErr::DTypeMismatch {
            code: "WEIGHT_DTYPE_MISMATCH",
            name: expected.name.to_string(),
            expected: expected.dtype,
            found: format!("{:?}", tensor_view.dtype()),
        });
    }

    let values_f32 = decode_tensor_to_f32(expected.dtype, tensor_view.data())?;

    loaded.insert(
        expected.name.to_string(),
        LoadedTensorRaw {
            name: expected.name.to_string(),
            dtype: expected.dtype,
            shape: found_shape,
            values_f32,
        },
    );
    Ok(())
}

fn decode_tensor_to_f32(dtype: DType, bytes: &[u8]) -> Result<Vec<f32>, WeightErr> {
    match dtype {
        DType::F32 => {
            if bytes.len() % 4 != 0 {
                return Err(WeightErr::ParseError {
                    code: "WEIGHT_PARSE_ERROR",
                    reason: "invalid f32 tensor byte length".to_string(),
                });
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect())
        }
        DType::F16 => {
            if bytes.len() % 2 != 0 {
                return Err(WeightErr::ParseError {
                    code: "WEIGHT_PARSE_ERROR",
                    reason: "invalid f16 tensor byte length".to_string(),
                });
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|chunk| {
                    half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32()
                })
                .collect())
        }
        DType::BF16 => {
            if bytes.len() % 2 != 0 {
                return Err(WeightErr::ParseError {
                    code: "WEIGHT_PARSE_ERROR",
                    reason: "invalid bf16 tensor byte length".to_string(),
                });
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|chunk| {
                    half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32()
                })
                .collect())
        }
        DType::I32 => {
            if bytes.len() % 4 != 0 {
                return Err(WeightErr::ParseError {
                    code: "WEIGHT_PARSE_ERROR",
                    reason: "invalid i32 tensor byte length".to_string(),
                });
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32)
                .collect())
        }
        DType::U8 => Ok(bytes.iter().map(|v| *v as f32).collect()),
    }
}

#[cfg(any(
    feature = "compute-candle",
    feature = "llm-candle",
    feature = "lfm-candle"
))]
pub fn raw_to_candle(raw: &LoadedWeightsRaw) -> Result<LoadedWeights, WeightErr> {
    let mut tensors = BTreeMap::new();
    for (name, tensor_raw) in &raw.tensors {
        let tensor = Tensor::from_vec(
            tensor_raw.values_f32.clone(),
            tensor_raw.shape.as_slice(),
            &Device::Cpu,
        )
        .map_err(|err| WeightErr::ParseError {
            code: "WEIGHT_PARSE_ERROR",
            reason: err.to_string(),
        })?;
        tensors.insert(name.clone(), tensor);
    }
    Ok(LoadedWeights {
        slot: raw.slot,
        tensors,
    })
}

#[cfg(any(
    feature = "compute-candle",
    feature = "llm-candle",
    feature = "lfm-candle"
))]
pub fn load_safetensors(
    slot: ModelSlot,
    bytes: &[u8],
    spec: &WeightSpec,
) -> Result<LoadedWeights, WeightErr> {
    let raw = load_safetensors_raw(slot, bytes, spec)?;
    raw_to_candle(&raw)
}
fn bind_shape(
    shape: &[DimExpr],
    found: &[usize],
    bindings: &mut BTreeMap<&'static str, usize>,
) -> Vec<usize> {
    let mut out = Vec::with_capacity(shape.len());
    for (idx, dim) in shape.iter().enumerate() {
        match dim {
            DimExpr::Fixed(value) => out.push(*value),
            DimExpr::Var(name) => {
                let found_dim = found[idx];
                match bindings.get(name) {
                    Some(bound) => out.push(*bound),
                    None => {
                        bindings.insert(name, found_dim);
                        out.push(found_dim);
                    }
                }
            }
        }
    }
    out
}

fn map_dtype(dtype: DType) -> safetensors::Dtype {
    match dtype {
        DType::F16 => safetensors::Dtype::F16,
        DType::BF16 => safetensors::Dtype::BF16,
        DType::F32 => safetensors::Dtype::F32,
        DType::I32 => safetensors::Dtype::I32,
        DType::U8 => safetensors::Dtype::U8,
    }
}

#[cfg(any(
    feature = "compute-candle",
    feature = "llm-candle",
    feature = "lfm-candle"
))]
#[allow(dead_code)]
fn map_candle_dtype(dtype: DType) -> Result<CandleDType, WeightErr> {
    match dtype {
        DType::F16 => Ok(CandleDType::F16),
        DType::BF16 => Ok(CandleDType::BF16),
        DType::F32 => Ok(CandleDType::F32),
        DType::U8 => Ok(CandleDType::U8),
        DType::I32 => Err(WeightErr::ParseError {
            code: "WEIGHT_PARSE_ERROR",
            reason: "I32 is not supported by this candle-core version".to_string(),
        }),
    }
}
pub fn map_model_load_error(err: &ModelLoadError) -> Option<WeightErr> {
    match err {
        ModelLoadError::HashMismatch { .. } => Some(WeightErr::HashMismatch {
            code: "WEIGHT_HASH_MISMATCH",
        }),
        ModelLoadError::Oversized {
            max_bytes,
            size_bytes,
            ..
        } => Some(WeightErr::TooLarge {
            code: "WEIGHT_TOO_LARGE",
            max_bytes: *max_bytes,
            found_bytes: *size_bytes,
        }),
        _ => None,
    }
}

pub fn load_verified_slot_raw(
    store: &ModelStore,
    verified: &VerifiedModelSlot,
    spec: &WeightSpec,
) -> Result<LoadedWeightsRaw, WeightErr> {
    let bytes = store.read_verified_bytes(verified).map_err(|err| {
        map_model_load_error(&err).unwrap_or(WeightErr::ParseError {
            code: "WEIGHT_PARSE_ERROR",
            reason: format!("{err:?}"),
        })
    })?;
    load_safetensors_raw(verified.slot, &bytes, spec)
}

#[cfg(any(
    feature = "compute-candle",
    feature = "llm-candle",
    feature = "lfm-candle"
))]
pub fn load_verified_slot(
    store: &ModelStore,
    verified: &VerifiedModelSlot,
    spec: &WeightSpec,
) -> Result<LoadedWeights, WeightErr> {
    let raw = load_verified_slot_raw(store, verified, spec)?;
    raw_to_candle(&raw)
}

pub fn backend_disable_for_weight_error(err: &WeightErr) -> ComputeError {
    tracing::error!("weight validation failed code={} err={:?}", err.code(), err);
    ComputeError::BackendDisabled
}

const D: DimExpr = DimExpr::Var("D");
const H: DimExpr = DimExpr::Var("H");
const F: DimExpr = DimExpr::Var("F");
const N: DimExpr = DimExpr::Var("N");

const JEPA_REQ: &[TensorSpec] = &[
    TensorSpec {
        name: "W1",
        shape: &[D, H],
        dtype: DType::F32,
    },
    TensorSpec {
        name: "b1",
        shape: &[H],
        dtype: DType::F32,
    },
    TensorSpec {
        name: "W2",
        shape: &[H, D],
        dtype: DType::F32,
    },
    TensorSpec {
        name: "b2",
        shape: &[D],
        dtype: DType::F32,
    },
];

const VLJEPA_REQ: &[TensorSpec] = &[
    TensorSpec {
        name: "vljepa.w1",
        shape: &[D, H],
        dtype: DType::F32,
    },
    TensorSpec {
        name: "vljepa.b1",
        shape: &[H],
        dtype: DType::F32,
    },
    TensorSpec {
        name: "vljepa.w2",
        shape: &[H, D],
        dtype: DType::F32,
    },
    TensorSpec {
        name: "vljepa.b2",
        shape: &[D],
        dtype: DType::F32,
    },
];

const SAE_REQ: &[TensorSpec] = &[
    TensorSpec {
        name: "sae.w_enc",
        shape: &[F, D],
        dtype: DType::F32,
    },
    TensorSpec {
        name: "sae.b_enc",
        shape: &[F],
        dtype: DType::F32,
    },
];

const SAE_OPT: &[TensorSpec] = &[
    TensorSpec {
        name: "sae.w_dec",
        shape: &[D, F],
        dtype: DType::F32,
    },
    TensorSpec {
        name: "sae.b_dec",
        shape: &[D],
        dtype: DType::F32,
    },
];

const SSM_REQ: &[TensorSpec] = &[
    TensorSpec {
        name: "A",
        shape: &[N, N],
        dtype: DType::F32,
    },
    TensorSpec {
        name: "B",
        shape: &[N],
        dtype: DType::F32,
    },
    TensorSpec {
        name: "C",
        shape: &[N],
        dtype: DType::F32,
    },
];

const LFM_REQ: &[TensorSpec] = &[
    TensorSpec {
        name: "alpha",
        shape: &[N],
        dtype: DType::F32,
    },
    TensorSpec {
        name: "Wx",
        shape: &[N, N],
        dtype: DType::F32,
    },
    TensorSpec {
        name: "Wu",
        shape: &[N],
        dtype: DType::F32,
    },
    TensorSpec {
        name: "b",
        shape: &[N],
        dtype: DType::F32,
    },
];

const LLM_REQ: &[TensorSpec] = &[
    TensorSpec {
        name: "tok_emb",
        shape: &[DimExpr::Fixed(32), DimExpr::Fixed(64)],
        dtype: DType::F32,
    },
    TensorSpec {
        name: "lm_head",
        shape: &[DimExpr::Fixed(64), DimExpr::Fixed(32)],
        dtype: DType::F32,
    },
];

const EBM_REQ: &[TensorSpec] = &[
    TensorSpec {
        name: "ebm.w1",
        shape: &[DimExpr::Var("d"), DimExpr::Var("h")],
        dtype: DType::F32,
    },
    TensorSpec {
        name: "ebm.b1",
        shape: &[DimExpr::Var("h")],
        dtype: DType::F32,
    },
    TensorSpec {
        name: "ebm.w2",
        shape: &[DimExpr::Var("h"), DimExpr::Fixed(1)],
        dtype: DType::F32,
    },
    TensorSpec {
        name: "ebm.b2",
        shape: &[DimExpr::Fixed(1)],
        dtype: DType::F32,
    },
];
pub fn spec_for_slot(slot: ModelSlot, max_bytes: u64) -> WeightSpec {
    let tensors = match slot {
        ModelSlot::WorldJepa => JEPA_REQ,
        ModelSlot::WorldVljepa => VLJEPA_REQ,
        ModelSlot::Sae => SAE_REQ,
        ModelSlot::Ssm => SSM_REQ,
        ModelSlot::Lfm => LFM_REQ,
        ModelSlot::Llm => LLM_REQ,
        ModelSlot::EbmReasoner => EBM_REQ,
    };
    let optional = match slot {
        ModelSlot::Sae => SAE_OPT,
        _ => &[],
    };
    WeightSpec {
        slot,
        tensors,
        optional,
        max_bytes,
        bindings: BTreeMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::{serialize, tensor::TensorView, Dtype};

    fn u8_tensor(shape: &[usize], values: &[u8]) -> TensorView<'static> {
        let leaked = Box::leak(values.to_vec().into_boxed_slice());
        TensorView::new(Dtype::U8, shape.to_vec(), leaked).expect("u8 tensor")
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>()
    }

    fn i32_bytes(values: &[i32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>()
    }

    fn f32_tensor(shape: &[usize], values: &[f32]) -> TensorView<'static> {
        let leaked = Box::leak(f32_bytes(values).into_boxed_slice());
        TensorView::new(Dtype::F32, shape.to_vec(), leaked).expect("f32 tensor")
    }

    fn i32_tensor(shape: &[usize], values: &[i32]) -> TensorView<'static> {
        let leaked = Box::leak(i32_bytes(values).into_boxed_slice());
        TensorView::new(Dtype::I32, shape.to_vec(), leaked).expect("i32 tensor")
    }

    fn fixture_jepa_small() -> Vec<u8> {
        serialize(
            [
                ("W1", f32_tensor(&[3, 4], &[0.0; 12])),
                ("W2", f32_tensor(&[4, 3], &[0.0; 12])),
                ("b1", f32_tensor(&[4], &[0.0; 4])),
                ("b2", f32_tensor(&[3], &[0.0; 3])),
            ],
            &None,
        )
        .expect("serialize")
    }

    fn fixture_jepa_missing() -> Vec<u8> {
        serialize(
            [
                ("W1", f32_tensor(&[3, 4], &[0.0; 12])),
                ("W2", f32_tensor(&[4, 3], &[0.0; 12])),
                ("b1", f32_tensor(&[4], &[0.0; 4])),
            ],
            &None,
        )
        .expect("serialize")
    }

    fn fixture_jepa_wrong_shape() -> Vec<u8> {
        serialize(
            [
                ("W1", f32_tensor(&[3, 5], &[0.0; 15])),
                ("W2", f32_tensor(&[4, 3], &[0.0; 12])),
                ("b1", f32_tensor(&[4], &[0.0; 4])),
                ("b2", f32_tensor(&[3], &[0.0; 3])),
            ],
            &None,
        )
        .expect("serialize")
    }

    fn fixture_jepa_wrong_dtype() -> Vec<u8> {
        serialize(
            [
                ("W1", i32_tensor(&[3, 4], &[0; 12])),
                ("W2", f32_tensor(&[4, 3], &[0.0; 12])),
                ("b1", f32_tensor(&[4], &[0.0; 4])),
                ("b2", f32_tensor(&[3], &[0.0; 3])),
            ],
            &None,
        )
        .expect("serialize")
    }

    fn fixture_ssm_small() -> Vec<u8> {
        serialize(
            [
                ("A", f32_tensor(&[3, 3], &[0.0; 9])),
                ("B", f32_tensor(&[3], &[0.0; 3])),
                ("C", f32_tensor(&[3], &[0.0; 3])),
            ],
            &None,
        )
        .expect("serialize")
    }

    #[test]
    fn loads_golden_jepa() {
        let spec = spec_for_slot(ModelSlot::WorldJepa, 1024 * 1024);
        let bytes = fixture_jepa_small();
        let loaded = load_safetensors_raw(ModelSlot::WorldJepa, &bytes, &spec).expect("valid");
        let keys: Vec<_> = loaded.tensors.keys().cloned().collect();
        assert_eq!(keys, vec!["W1", "W2", "b1", "b2"]);
    }

    #[test]
    fn rejects_missing_tensor() {
        let spec = spec_for_slot(ModelSlot::WorldJepa, 1024 * 1024);
        let bytes = fixture_jepa_missing();
        let err = load_safetensors_raw(ModelSlot::WorldJepa, &bytes, &spec).expect_err("must fail");
        assert!(matches!(err, WeightErr::MissingTensor { .. }));
        assert_eq!(err.code(), "WEIGHT_MISSING_TENSOR");
    }

    #[test]
    fn rejects_shape_mismatch() {
        let spec = spec_for_slot(ModelSlot::WorldJepa, 1024 * 1024);
        let bytes = fixture_jepa_wrong_shape();
        let err = load_safetensors_raw(ModelSlot::WorldJepa, &bytes, &spec).expect_err("must fail");
        assert!(matches!(err, WeightErr::ShapeMismatch { .. }));
        assert_eq!(err.code(), "WEIGHT_SHAPE_MISMATCH");
    }

    #[test]
    fn rejects_dtype_mismatch() {
        let spec = spec_for_slot(ModelSlot::WorldJepa, 1024 * 1024);
        let bytes = fixture_jepa_wrong_dtype();
        let err = load_safetensors_raw(ModelSlot::WorldJepa, &bytes, &spec).expect_err("must fail");
        assert!(matches!(err, WeightErr::DTypeMismatch { .. }));
        assert_eq!(err.code(), "WEIGHT_DTYPE_MISMATCH");
    }

    #[test]
    fn deterministic_ordering() {
        let spec = spec_for_slot(ModelSlot::Ssm, 1024 * 1024);
        let bytes = fixture_ssm_small();
        let loaded = load_safetensors_raw(ModelSlot::Ssm, &bytes, &spec).expect("valid");
        let keys: Vec<_> = loaded.tensors.keys().cloned().collect();
        assert_eq!(keys, vec!["A", "B", "C"]);
    }

    #[test]
    fn too_large_rejected_before_parse() {
        let spec = spec_for_slot(ModelSlot::WorldJepa, 3);
        let bytes = vec![1_u8, 2, 3, 4];
        let err = load_safetensors_raw(ModelSlot::WorldJepa, &bytes, &spec).expect_err("too large");
        assert!(matches!(err, WeightErr::TooLarge { .. }));
        assert_eq!(err.code(), "WEIGHT_TOO_LARGE");
    }

    #[test]
    fn model_store_roundtrip_and_backend_disabled() {
        use sha2::Digest;
        use std::collections::BTreeMap;

        let temp = tempfile::tempdir().expect("temp");
        let models = temp.path().join("models");
        std::fs::create_dir_all(&models).expect("dir");
        let src = fixture_jepa_wrong_shape();
        let rel = "jepa.safetensors";
        let full = models.join(rel);
        std::fs::write(&full, &src).expect("write");

        let mut specs = BTreeMap::new();
        specs.insert(
            ModelSlot::WorldJepa,
            crate::model_store::ModelSlotSpec {
                slot: ModelSlot::WorldJepa,
                enabled: true,
                path: Some(rel.into()),
                expected_sha256: sha2::Sha256::digest(&src).into(),
                max_bytes: 1024 * 1024,
                format: crate::model_store::ModelFormat::CandleSafetensors,
                device: crate::model_store::ModelDevice::CpuOnly,
                active_hash: None,
                contract_version: None,
            },
        );
        let store = ModelStore {
            allowlist_root: models,
            specs,
        };
        let verified = store.verify_slot(ModelSlot::WorldJepa).expect("verified");
        let spec = spec_for_slot(ModelSlot::WorldJepa, verified.size_bytes);
        let err = load_verified_slot_raw(&store, &verified, &spec).expect_err("shape mismatch");
        assert!(matches!(
            backend_disable_for_weight_error(&err),
            ComputeError::BackendDisabled
        ));
    }

    #[test]
    fn supports_u8_dtype_variant() {
        let bytes = [7_u8, 8, 9, 10];
        let blob = serialize([("x", u8_tensor(&[4], &bytes))], &None).expect("serialize");
        let spec = WeightSpec {
            slot: ModelSlot::Llm,
            tensors: &[TensorSpec {
                name: "x",
                shape: &[DimExpr::Fixed(4)],
                dtype: DType::U8,
            }],
            optional: &[],
            max_bytes: 1024,
            bindings: BTreeMap::new(),
        };
        let loaded = load_safetensors_raw(ModelSlot::Llm, &blob, &spec).expect("u8 ok");
        assert!(loaded.tensors.contains_key("x"));
    }

    #[test]
    fn sae_spec_uses_v1_1_tensor_names_and_optional_decoder() {
        let spec = spec_for_slot(ModelSlot::Sae, 1024);
        let required: Vec<_> = spec.tensors.iter().map(|t| t.name).collect();
        let optional: Vec<_> = spec.optional.iter().map(|t| t.name).collect();
        assert_eq!(required, vec!["sae.w_enc", "sae.b_enc"]);
        assert_eq!(optional, vec!["sae.w_dec", "sae.b_dec"]);
    }
}
