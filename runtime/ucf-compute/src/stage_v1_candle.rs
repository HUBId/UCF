use std::collections::BTreeMap;

use crate::candle_weights::{load_verified_slot_raw, DType, DimExpr, TensorSpec, WeightSpec};
use crate::model_store::{ModelSlot, ModelStore};
use crate::stage_v1::{
    digest_prediction, mix_q, novelty_from_context, StageError, WorldInputV1, WorldOutputV1,
    WorldPredictorV1, MOCK_WORLD_DIM, STAGE_CONTRACT_VERSION_V1,
};

const D: DimExpr = DimExpr::Var("D");
const H: DimExpr = DimExpr::Var("H");
const CANDLE_WORLD_BACKEND_ID: u16 = 302;

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
}
