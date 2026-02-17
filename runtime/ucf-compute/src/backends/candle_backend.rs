use candle_core::{Device, Tensor};
use sha2::{Digest, Sha256};

use crate::capabilities::{SaeExtractor, WorldModelPredictor};
use crate::evidence::{quantize_signed_unit, quantize_unit_u16};
use crate::feature_extractor::{
    SaeInput, SaeOutput, SmallNotes, SAE_FEATURE_DIM, SAE_INPUT_DIM, SAE_TOP_K,
};
use crate::ssm::{SsmInput, SsmKernel, SsmOutput, SSM_STATE_DIM};
use crate::world_model::{
    state_norm_01, StageQuality, WorldModelInput, WorldModelOutput, WORLD_MODEL_FEATURE_DIM,
};
use crate::{ComputeBudget, ComputeError, Spike};

#[derive(Debug, Clone, Copy)]
pub struct CandleWorldPredictor {
    state: [f32; WORLD_MODEL_FEATURE_DIM],
    model_hash: [u8; 32],
}

impl CandleWorldPredictor {
    pub fn new(model_hash: [u8; 32]) -> Self {
        Self {
            state: [0.0; WORLD_MODEL_FEATURE_DIM],
            model_hash,
        }
    }

    fn weight(&self, i: usize, j: usize, tag: u8) -> f32 {
        let idx = (i * WORLD_MODEL_FEATURE_DIM + j + usize::from(tag)) % self.model_hash.len();
        let v = self.model_hash[idx] as f32 / 255.0;
        (v * 2.0 - 1.0) * 0.2
    }

    fn digest(values: &[f32], model_hash: [u8; 32], t: u64, seed: u64) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(model_hash);
        hasher.update(t.to_le_bytes());
        hasher.update(seed.to_le_bytes());
        for v in values {
            hasher.update(quantize_signed_unit(*v).to_le_bytes());
        }
        hasher.finalize().into()
    }
}

impl WorldModelPredictor for CandleWorldPredictor {
    fn name(&self) -> &'static str {
        "candle_world_jepa_v1"
    }

    fn step(
        &mut self,
        input: &WorldModelInput,
        _budget: ComputeBudget,
    ) -> Result<WorldModelOutput, ComputeError> {
        let mut pred = [0.0; WORLD_MODEL_FEATURE_DIM];
        for (i, pred_i) in pred.iter_mut().enumerate().take(WORLD_MODEL_FEATURE_DIM) {
            let mut v = self.weight(i, i, 1);
            for j in 0..WORLD_MODEL_FEATURE_DIM {
                v += self.weight(i, j, 3) * self.state[j]
                    + self.weight(i, j, 7) * input.obs_features[j];
            }
            *pred_i = v.clamp(-1.0, 1.0);
        }
        let err = pred
            .iter()
            .zip(input.obs_features)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / WORLD_MODEL_FEATURE_DIM as f32;
        for (s, p) in self.state.iter_mut().zip(pred.iter()) {
            *s = (0.8 * *s + 0.2 * *p).clamp(-1.0, 1.0);
        }
        Ok(WorldModelOutput {
            prediction_digest: Self::digest(&pred, self.model_hash, input.t, input.seed),
            state_digest: Self::digest(
                &self.state,
                self.model_hash,
                input.t.saturating_add(1),
                input.seed,
            ),
            prediction_error: err.clamp(0.0, 1.0),
            surprise: err.clamp(0.0, 1.0),
            state_norm: state_norm_01(&self.state),
            quality: StageQuality::Ok,
            notes: vec![format!("model={}", hex::encode(&self.model_hash[..6]))],
        }
        .bounded())
    }
}

#[derive(Debug, Clone)]
pub struct CandleSaeExtractor {
    weights: Vec<f32>,
    bias: Vec<f32>,
    model_hash: [u8; 32],
}

impl CandleSaeExtractor {
    pub fn new(model_hash: [u8; 32]) -> Self {
        let mut weights = vec![0.0; SAE_FEATURE_DIM * SAE_INPUT_DIM];
        let mut bias = vec![0.0; SAE_FEATURE_DIM];
        for i in 0..SAE_FEATURE_DIM {
            bias[i] = ((model_hash[i % 32] as f32 / 255.0) - 0.5) * 0.2;
            for j in 0..SAE_INPUT_DIM {
                let idx = (i * SAE_INPUT_DIM + j) % 32;
                weights[i * SAE_INPUT_DIM + j] = ((model_hash[idx] as f32 / 255.0) - 0.5) * 0.25;
            }
        }
        Self {
            weights,
            bias,
            model_hash,
        }
    }
}

impl Default for CandleSaeExtractor {
    fn default() -> Self {
        Self::new([0x11; 32])
    }
}

impl SaeExtractor for CandleSaeExtractor {
    fn name(&self) -> &'static str {
        "candle_sae_v1"
    }

    fn extract(&self, input: &SaeInput, _budget: ComputeBudget) -> Result<SaeOutput, ComputeError> {
        let device = Device::Cpu;
        let x =
            Tensor::from_slice(&input.context_features, SAE_INPUT_DIM, &device).map_err(|e| {
                ComputeError::Internal {
                    reason: e.to_string(),
                }
            })?;
        let w = Tensor::from_slice(&self.weights, (SAE_FEATURE_DIM, SAE_INPUT_DIM), &device)
            .map_err(|e| ComputeError::Internal {
                reason: e.to_string(),
            })?;
        let b = Tensor::from_slice(&self.bias, SAE_FEATURE_DIM, &device).map_err(|e| {
            ComputeError::Internal {
                reason: e.to_string(),
            }
        })?;
        let x_col = x
            .reshape((SAE_INPUT_DIM, 1))
            .map_err(|e| ComputeError::Internal {
                reason: e.to_string(),
            })?;
        let y = w
            .matmul(&x_col)
            .and_then(|t| t.reshape(SAE_FEATURE_DIM))
            .and_then(|t| t.broadcast_add(&b))
            .map_err(|e| ComputeError::Internal {
                reason: e.to_string(),
            })?;
        let mut activations = y.to_vec1::<f32>().map_err(|e| ComputeError::Internal {
            reason: e.to_string(),
        })?;
        activations.iter_mut().for_each(|v| *v = v.max(0.0));
        let max_a = activations.iter().copied().fold(0.0, f32::max).max(1e-6);
        let mut ranked: Vec<(usize, f32)> = activations.into_iter().enumerate().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let mut spikes = Vec::new();
        for (fid, a) in ranked.into_iter().take(SAE_TOP_K) {
            if a <= 0.0 {
                break;
            }
            spikes.push(Spike {
                feature_id: fid as u32,
                magnitude: (a / max_a).clamp(0.0, 1.0),
                timestamp: input.t,
            });
        }
        spikes.sort_by_key(|s| s.feature_id);
        let energy =
            (spikes.iter().map(|s| s.magnitude).sum::<f32>() / SAE_TOP_K as f32).clamp(0.0, 1.0);
        let sparsity = (1.0 - spikes.len() as f32 / SAE_FEATURE_DIM as f32).clamp(0.0, 1.0);
        let mut hasher = Sha256::new();
        hasher.update(self.model_hash);
        for s in &spikes {
            hasher.update(s.feature_id.to_le_bytes());
            hasher.update(quantize_unit_u16(s.magnitude).to_le_bytes());
        }
        hasher.update(input.t.to_le_bytes());
        let spikes_digest = hasher.finalize().into();
        Ok(SaeOutput {
            spikes: spikes.clone(),
            spike_count: spikes.len() as u16,
            sparsity,
            energy,
            spikes_digest,
            quality: StageQuality::Ok,
            notes: SmallNotes(vec![format!(
                "model={}",
                hex::encode(&self.model_hash[..6])
            )]),
        }
        .bounded())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CandleSsmKernel {
    x: [f32; SSM_STATE_DIM],
    model_hash: [u8; 32],
}

impl CandleSsmKernel {
    pub fn new(model_hash: [u8; 32]) -> Self {
        Self {
            x: [0.0; SSM_STATE_DIM],
            model_hash,
        }
    }
}

impl SsmKernel for CandleSsmKernel {
    fn name(&self) -> &'static str {
        "candle_ssm_selective_scan_v1"
    }

    fn step(
        &mut self,
        input: &SsmInput,
        _budget: ComputeBudget,
    ) -> Result<SsmOutput, ComputeError> {
        let u = (0.4 * (f32::from(input.spike_count) / 32.0)
            + 0.3 * input.sae_energy
            + 0.3 * input.world_surprise)
            .clamp(0.0, 1.0);
        for i in 0..SSM_STATE_DIM {
            let decay = 0.94 + (self.model_hash[i % 32] as f32 / 255.0) * 0.04;
            let gain = ((self.model_hash[(i + 7) % 32] as f32 / 255.0) - 0.5) * 0.5;
            self.x[i] = (decay * self.x[i] + gain * u).clamp(-1.0, 1.0);
        }
        let state_norm =
            (self.x.iter().map(|v| v.abs()).sum::<f32>() / SSM_STATE_DIM as f32).clamp(0.0, 1.0);
        let readout =
            ((self.x.iter().sum::<f32>() / SSM_STATE_DIM as f32) * 0.5 + 0.5).clamp(0.0, 1.0);
        let pressure = (0.5 * u + 0.5 * state_norm).clamp(0.0, 1.0);
        let mut s = Sha256::new();
        s.update(self.model_hash);
        s.update(input.t.to_le_bytes());
        for v in &self.x {
            s.update(quantize_signed_unit(*v).to_le_bytes());
        }
        let state_digest = s.finalize().into();
        let mut r = Sha256::new();
        r.update(state_digest);
        r.update(quantize_unit_u16(readout).to_le_bytes());
        let readout_digest = r.finalize().into();
        Ok(SsmOutput {
            pressure,
            state_digest,
            readout_digest,
            state_norm,
            readout,
            quality: StageQuality::Ok,
            notes: SmallNotes(vec![format!(
                "model={}",
                hex::encode(&self.model_hash[..6])
            )]),
        })
    }
}
