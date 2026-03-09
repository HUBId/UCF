use std::collections::BTreeMap;

use crate::candle_weights::{load_verified_slot_raw, DType, DimExpr, TensorSpec, WeightSpec};
use crate::model_store::{ModelSlot, ModelStore};
use crate::stage_v1::{
    digest_prediction, mix_q, novelty_from_context, SaeExtractorV1, SaeInputV1, SaeOutputV1,
    SaeSpikeV1, StageError, StageErrorCode, WorldInputV1, WorldOutputV1, WorldPredictorV1,
    MAX_SAE_SPIKES, MOCK_WORLD_DIM, STAGE_CONTRACT_VERSION_V1,
};
use candle_core::{Device, Tensor};
use sha2::{Digest, Sha256};

const D: DimExpr = DimExpr::Var("D");
const H: DimExpr = DimExpr::Var("H");
const F: DimExpr = DimExpr::Var("F");
const CANDLE_WORLD_BACKEND_ID: u16 = 302;
const CANDLE_SAE_BACKEND_ID: u16 = 303;
const SAE_INPUT_DIM: usize = 16;

const WORLD_MLP_REQ: &[TensorSpec] = &[
    TensorSpec {
        name: "w1",
        shape: &[D, H],
        dtype: DType::F32,
    },
    TensorSpec {
        name: "b1",
        shape: &[H],
        dtype: DType::F32,
    },
    TensorSpec {
        name: "w2",
        shape: &[H, D],
        dtype: DType::F32,
    },
    TensorSpec {
        name: "b2",
        shape: &[D],
        dtype: DType::F32,
    },
];

#[derive(Debug, Clone)]
struct WorldMlp {
    d: usize,
    h: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct CandleWorldAdapterV0 {
    mlp: Option<WorldMlp>,
}

#[derive(Debug, Clone)]
struct CandleSaeModel {
    model_hash: [u8; 32],
    feature_dim: usize,
    input_dim: usize,
    w_enc: Tensor,
    b_enc: Tensor,
}

#[derive(Debug, Clone)]
pub struct CandleSaeAdapterV0 {
    encoder: Option<CandleSaeModel>,
}

impl CandleWorldAdapterV0 {
    pub fn disabled() -> Self {
        Self { mlp: None }
    }

    pub fn from_model_store(store: &ModelStore) -> Self {
        let Ok(verified) = store.verify_slot(ModelSlot::WorldJepa) else {
            return Self::disabled();
        };
        let spec = world_weight_spec(verified.size_bytes);
        let Ok(loaded) = load_verified_slot_raw(store, &verified, &spec) else {
            return Self::disabled();
        };

        let w1 = loaded.tensors.get("w1").expect("validated");
        let b1 = loaded.tensors.get("b1").expect("validated");
        let w2 = loaded.tensors.get("w2").expect("validated");
        let b2 = loaded.tensors.get("b2").expect("validated");

        Self {
            mlp: Some(WorldMlp {
                d: w1.shape[0],
                h: w1.shape[1],
                w1: w1.values_f32.clone(),
                b1: b1.values_f32.clone(),
                w2: w2.values_f32.clone(),
                b2: b2.values_f32.clone(),
            }),
        }
    }

    fn features_from_input(input: &WorldInputV1) -> [f32; MOCK_WORLD_DIM] {
        let mut out = [0.0; MOCK_WORLD_DIM];
        let prev = input.previous_world_state_digest.unwrap_or([0; 32]);
        for (i, item) in out.iter_mut().enumerate() {
            let ctx = (f32::from(input.context_digest[i]) - 128.0) / 128.0;
            let prv = (f32::from(prev[MOCK_WORLD_DIM + i]) - 128.0) / 128.0;
            let sig = f32::from((input.signal_q & 0x00ff) as u8) / 255.0;
            *item = (ctx * 0.75 + prv * 0.2 + sig * 0.05).clamp(-1.0, 1.0);
        }
        out
    }

    fn run_mlp(mlp: &WorldMlp, x: &[f32; MOCK_WORLD_DIM]) -> [i16; MOCK_WORLD_DIM] {
        let mut h = vec![0.0_f32; mlp.h];
        for (j, hj) in h.iter_mut().enumerate() {
            let mut acc = mlp.b1[j];
            for (i, xi) in x.iter().enumerate().take(mlp.d) {
                acc += *xi * mlp.w1[i * mlp.h + j];
            }
            *hj = acc.tanh();
        }

        let mut out = [0i16; MOCK_WORLD_DIM];
        for (j, item) in out.iter_mut().enumerate().take(mlp.d) {
            let mut acc = mlp.b2[j];
            for (i, hi) in h.iter().enumerate() {
                acc += *hi * mlp.w2[i * mlp.d + j];
            }
            let clamped = acc.clamp(-1.0, 1.0);
            *item = (clamped * 32767.0).round() as i16;
        }
        out
    }
}

impl CandleSaeAdapterV0 {
    pub fn disabled() -> Self {
        Self { encoder: None }
    }

    pub fn from_model_store(store: &ModelStore) -> Self {
        let Ok(verified) = store.verify_slot(ModelSlot::Sae) else {
            return Self::disabled();
        };
        let spec = sae_weight_spec(verified.size_bytes);
        let Ok(loaded) = load_verified_slot_raw(store, &verified, &spec) else {
            return Self::disabled();
        };
        let w_enc = loaded.tensors.get("sae.w_enc").expect("validated");
        let b_enc = loaded.tensors.get("sae.b_enc").expect("validated");
        let device = Device::Cpu;
        let Ok(w_enc_t) = Tensor::from_vec(
            w_enc.values_f32.clone(),
            (w_enc.shape[0], w_enc.shape[1]),
            &device,
        ) else {
            return Self::disabled();
        };
        let Ok(b_enc_t) = Tensor::from_vec(b_enc.values_f32.clone(), b_enc.shape[0], &device)
        else {
            return Self::disabled();
        };

        Self {
            encoder: Some(CandleSaeModel {
                model_hash: verified.sha256,
                feature_dim: w_enc.shape[0],
                input_dim: w_enc.shape[1],
                w_enc: w_enc_t,
                b_enc: b_enc_t,
            }),
        }
    }

    fn input_vector(input: &SaeInputV1) -> [f32; SAE_INPUT_DIM] {
        let mut out = [0.0_f32; SAE_INPUT_DIM];
        for (idx, item) in out.iter_mut().enumerate() {
            let a = (f32::from(input.context_digest[idx]) - 127.5) / 127.5;
            let b = (f32::from(input.prediction_digest[idx]) - 127.5) / 127.5;
            *item = (0.7 * a + 0.3 * b).clamp(-1.0, 1.0);
        }
        out
    }

    fn deterministic_top_k(values: &[f32], top_k: usize) -> Vec<(u16, f32)> {
        let mut ranked: Vec<(u16, f32)> = values
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, v)| (idx as u16, v.max(0.0)))
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked
            .into_iter()
            .filter(|(_, v)| *v > 0.0)
            .take(top_k)
            .collect()
    }

    fn spike_digest(model_hash: [u8; 32], input: &SaeInputV1, spikes: &[SaeSpikeV1]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(model_hash);
        hasher.update(input.context_digest);
        hasher.update(input.prediction_digest);
        hasher.update([input.top_k]);
        for spike in spikes {
            hasher.update(spike.feature_id.to_le_bytes());
            hasher.update(spike.magnitude_q.to_le_bytes());
        }
        hasher.finalize().into()
    }
}

impl WorldPredictorV1 for CandleWorldAdapterV0 {
    fn contract_version(&self) -> u16 {
        STAGE_CONTRACT_VERSION_V1
    }

    fn backend_id(&self) -> u16 {
        CANDLE_WORLD_BACKEND_ID
    }

    fn step(&self, input: &WorldInputV1) -> Result<WorldOutputV1, StageError> {
        let Some(mlp) = self.mlp.as_ref() else {
            return Err(StageError::backend_disabled(
                "world candle adapter disabled",
            ));
        };
        let prev = input.previous_world_state_digest.unwrap_or([0; 32]);
        let features = Self::features_from_input(input);
        let prediction_q = Self::run_mlp(mlp, &features);
        let prediction_digest = digest_prediction(&prediction_q);

        let mut error_sum = 0u32;
        for idx in 0..MOCK_WORLD_DIM {
            let cur = i32::from(prediction_q[idx]);
            let prev_val = i32::from(i16::from_le_bytes([prev[idx * 2], prev[idx * 2 + 1]]));
            error_sum = error_sum.saturating_add(cur.abs_diff(prev_val));
        }
        let mean_abs = error_sum / MOCK_WORLD_DIM as u32;
        let prediction_error_q = mean_abs.min(u32::from(u16::MAX)) as u16;
        let novelty_q = novelty_from_context(input.context_digest);
        let surprise_q = mix_q(prediction_error_q, novelty_q, 45875, 19661);

        Ok(WorldOutputV1 {
            prediction_q,
            prediction_error_q,
            surprise_q,
            prediction_digest,
        })
    }
}

pub fn world_weight_spec(max_bytes: u64) -> WeightSpec {
    WeightSpec {
        slot: ModelSlot::WorldJepa,
        tensors: WORLD_MLP_REQ,
        optional: &[],
        max_bytes,
        bindings: BTreeMap::new(),
    }
}

pub fn sae_weight_spec(max_bytes: u64) -> WeightSpec {
    WeightSpec {
        slot: ModelSlot::Sae,
        tensors: &[
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
        ],
        optional: &[],
        max_bytes,
        bindings: BTreeMap::new(),
    }
}

impl SaeExtractorV1 for CandleSaeAdapterV0 {
    fn contract_version(&self) -> u16 {
        STAGE_CONTRACT_VERSION_V1
    }

    fn backend_id(&self) -> u16 {
        CANDLE_SAE_BACKEND_ID
    }

    fn infer(&self, input: &SaeInputV1) -> Result<SaeOutputV1, StageError> {
        let Some(encoder) = self.encoder.as_ref() else {
            return Err(StageError::backend_disabled("sae candle adapter disabled"));
        };
        if encoder.input_dim != SAE_INPUT_DIM {
            return Err(StageError {
                code: StageErrorCode::ValidationFailed,
                reason: "sae candle input dim mismatch",
            });
        }
        let x = Self::input_vector(input);
        let x = Tensor::from_slice(&x, SAE_INPUT_DIM, &Device::Cpu).map_err(|_| StageError {
            code: StageErrorCode::Internal,
            reason: "sae candle tensor input",
        })?;
        let z = encoder
            .w_enc
            .matmul(&x.reshape((SAE_INPUT_DIM, 1)).map_err(|_| StageError {
                code: StageErrorCode::Internal,
                reason: "sae candle reshape input",
            })?)
            .and_then(|t| t.reshape(encoder.feature_dim))
            .and_then(|t| t.broadcast_add(&encoder.b_enc))
            .map_err(|_| StageError {
                code: StageErrorCode::Internal,
                reason: "sae candle matmul",
            })?;
        let z = z.to_vec1::<f32>().map_err(|_| StageError {
            code: StageErrorCode::Internal,
            reason: "sae candle extract",
        })?;

        let k = usize::from(input.top_k)
            .min(MAX_SAE_SPIKES)
            .min(encoder.feature_dim);
        let ranked = Self::deterministic_top_k(&z, k);
        let max_mag = ranked
            .iter()
            .fold(0.0_f32, |acc, (_, v)| acc.max(*v))
            .max(1e-9);
        let mut spikes = Vec::with_capacity(ranked.len());
        for (feature_id, magnitude) in ranked {
            let normalized = (magnitude / max_mag).clamp(0.0, 1.0);
            spikes.push(SaeSpikeV1 {
                feature_id,
                magnitude_q: (normalized * 65535.0).round() as u16,
            });
        }
        spikes.sort_by_key(|s| s.feature_id);
        let spikes_digest = Self::spike_digest(encoder.model_hash, input, &spikes);
        Ok(SaeOutputV1 {
            spikes,
            spikes_digest,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::{serialize, tensor::TensorView, Dtype};

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn f32_tensor(shape: &[usize], values: &[f32]) -> TensorView<'static> {
        let leaked = Box::leak(f32_bytes(values).into_boxed_slice());
        TensorView::new(Dtype::F32, shape.to_vec(), leaked).expect("f32 tensor")
    }

    #[test]
    fn weight_spec_accepts_small_fixture() {
        let bytes = serialize(
            [
                (
                    "w1",
                    f32_tensor(&[MOCK_WORLD_DIM, 4], &[0.1; MOCK_WORLD_DIM * 4]),
                ),
                ("b1", f32_tensor(&[4], &[0.0; 4])),
                (
                    "w2",
                    f32_tensor(&[4, MOCK_WORLD_DIM], &[0.1; 4 * MOCK_WORLD_DIM]),
                ),
                ("b2", f32_tensor(&[MOCK_WORLD_DIM], &[0.0; MOCK_WORLD_DIM])),
            ],
            &None,
        )
        .expect("serialize");
        let spec = world_weight_spec(bytes.len() as u64 + 8);
        let loaded =
            crate::candle_weights::load_safetensors_raw(ModelSlot::WorldJepa, &bytes, &spec)
                .expect("valid fixture");
        assert_eq!(loaded.tensors.len(), 4);
    }

    #[test]
    fn deterministic_forward_and_disabled_path() {
        let mut mlp = WorldMlp {
            d: MOCK_WORLD_DIM,
            h: 4,
            w1: vec![0.0; MOCK_WORLD_DIM * 4],
            b1: vec![0.0; 4],
            w2: vec![0.0; 4 * MOCK_WORLD_DIM],
            b2: vec![0.0; MOCK_WORLD_DIM],
        };
        mlp.w1[0] = 0.5;
        mlp.w2[0] = 0.5;
        let model = CandleWorldAdapterV0 { mlp: Some(mlp) };
        let input = WorldInputV1 {
            context_digest: [9; 32],
            previous_world_state_digest: Some([3; 32]),
            signal_q: 77,
        };
        let a = model.step(&input).expect("ok");
        let b = model.step(&input).expect("ok");
        assert_eq!(a, b);

        let disabled = CandleWorldAdapterV0::disabled();
        let err = disabled.step(&input).expect_err("disabled");
        assert_eq!(err.code, crate::stage_v1::StageErrorCode::BackendDisabled);
    }

    #[test]
    fn sae_weight_spec_accepts_small_fixture() {
        let bytes = serialize(
            [
                (
                    "sae.w_enc",
                    f32_tensor(&[64, SAE_INPUT_DIM], &[0.1; 64 * SAE_INPUT_DIM]),
                ),
                ("sae.b_enc", f32_tensor(&[64], &[0.0; 64])),
            ],
            &None,
        )
        .expect("serialize");
        let spec = sae_weight_spec(bytes.len() as u64 + 8);
        let loaded = crate::candle_weights::load_safetensors_raw(ModelSlot::Sae, &bytes, &spec)
            .expect("valid fixture");
        assert_eq!(loaded.tensors.len(), 2);
    }

    #[test]
    fn sae_topk_is_deterministic_and_tie_breaks() {
        let device = Device::Cpu;
        let model = CandleSaeAdapterV0 {
            encoder: Some(CandleSaeModel {
                model_hash: [7; 32],
                feature_dim: 64,
                input_dim: SAE_INPUT_DIM,
                w_enc: Tensor::from_vec(
                    vec![0.0_f32; 64 * SAE_INPUT_DIM],
                    (64, SAE_INPUT_DIM),
                    &device,
                )
                .expect("w"),
                b_enc: {
                    let mut b = vec![0.0_f32; 64];
                    b[1] = 1.0;
                    b[2] = 1.0;
                    Tensor::from_vec(b, 64, &device).expect("b")
                },
            }),
        };
        let input = SaeInputV1 {
            context_digest: [3; 32],
            prediction_digest: [9; 32],
            top_k: 2,
        };
        let a = model.infer(&input).expect("ok");
        let b = model.infer(&input).expect("ok");
        assert_eq!(a, b);
        assert_eq!(a.spikes.len(), 2);
        assert_eq!(a.spikes[0].feature_id, 1);
        assert_eq!(a.spikes[1].feature_id, 2);

        let disabled = CandleSaeAdapterV0::disabled();
        let err = disabled.infer(&input).expect_err("disabled");
        assert_eq!(err.code, StageErrorCode::BackendDisabled);
    }
}
