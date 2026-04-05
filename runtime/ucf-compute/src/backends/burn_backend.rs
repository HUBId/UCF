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

const DEGRADED_MARKER: &[u8] = b"degraded_v1";

#[derive(Debug, Clone)]
pub struct BurnWorldPredictor {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    d: usize,
    h: usize,
    model_hash: [u8; 32],
    last_state_digest: Option<[u8; 32]>,
}

impl BurnWorldPredictor {
    pub fn new(model_hash: [u8; 32]) -> Self {
        let d = WORLD_MODEL_FEATURE_DIM;
        let h = WORLD_MODEL_FEATURE_DIM;
        let mut w1 = vec![0.0; d * h];
        let mut w2 = vec![0.0; h * d];
        let mut b1 = vec![0.0; h];
        let mut b2 = vec![0.0; d];
        for i in 0..d {
            for j in 0..h {
                let idx = (i * h + j) % 32;
                w1[i * h + j] = ((model_hash[idx] as f32 / 255.0) * 2.0 - 1.0) * 0.25;
            }
        }
        for i in 0..h {
            for j in 0..d {
                let idx = (11 + i * d + j) % 32;
                w2[i * d + j] = ((model_hash[idx] as f32 / 255.0) * 2.0 - 1.0) * 0.25;
            }
            b1[i] = ((model_hash[(i + 3) % 32] as f32 / 255.0) - 0.5) * 0.1;
        }
        for (i, v) in b2.iter_mut().enumerate() {
            *v = ((model_hash[(i + 17) % 32] as f32 / 255.0) - 0.5) * 0.1;
        }
        Self {
            w1,
            b1,
            w2,
            b2,
            d,
            h,
            model_hash,
            last_state_digest: None,
        }
    }

    fn degraded(input: &WorldModelInput, reason: &'static str) -> WorldModelOutput {
        let mut hasher = Sha256::new();
        hasher.update(DEGRADED_MARKER);
        hasher.update(b"world");
        hasher.update(reason.as_bytes());
        hasher.update(input.t.to_le_bytes());
        hasher.update(input.context_digest);
        let digest: [u8; 32] = hasher.finalize().into();
        WorldModelOutput {
            prediction_digest: digest,
            state_digest: digest,
            prediction_error: 1.0,
            surprise: 1.0,
            state_norm: 1.0,
            quality: StageQuality::DegradedFallback,
            notes: vec![format!("violation:{reason}")],
        }
    }

    fn contains_non_finite(values: &[f32]) -> bool {
        values.iter().any(|v| !v.is_finite())
    }

    fn prediction_digest(&self, y: &[f32], input: &WorldModelInput) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for v in y {
            hasher.update(quantize_signed_unit(v.clamp(-1.0, 1.0)).to_le_bytes());
        }
        hasher.update(self.model_hash);
        hasher.update(input.t.to_le_bytes());
        hasher.update(input.context_digest);
        if let Some(previous) = input.previous_state_digest {
            hasher.update([1]);
            hasher.update(previous);
        } else {
            hasher.update([0]);
        }
        hasher.finalize().into()
    }
}

impl WorldModelPredictor for BurnWorldPredictor {
    fn name(&self) -> &'static str {
        "burn_jepa_v1"
    }

    fn canonical_slot(&self) -> Option<crate::ModelSlot> {
        Some(crate::ModelSlot::WorldJepa)
    }

    fn current_state_digest(&self) -> Option<[u8; 32]> {
        self.last_state_digest
    }

    fn step(
        &mut self,
        input: &WorldModelInput,
        _budget: ComputeBudget,
    ) -> Result<WorldModelOutput, ComputeError> {
        let x = &input.obs_features[..self.d.min(input.obs_features.len())];
        let mut h = vec![0.0_f32; self.h];
        for (j, hj) in h.iter_mut().enumerate() {
            let mut acc = self.b1[j];
            for (i, xi) in x.iter().enumerate() {
                acc += *xi * self.w1[i * self.h + j];
            }
            *hj = acc.tanh();
        }

        let mut y = vec![0.0_f32; self.d];
        for (j, yj) in y.iter_mut().enumerate() {
            let mut acc = self.b2[j];
            for (i, hi) in h.iter().enumerate() {
                acc += *hi * self.w2[i * self.d + j];
            }
            *yj = acc.clamp(-1.0, 1.0);
        }

        if Self::contains_non_finite(&y) {
            return Ok(Self::degraded(input, "nan_inf").bounded());
        }

        let err = (y.iter().map(|v| v.abs()).sum::<f32>() / self.d as f32).clamp(0.0, 1.0);
        let prediction_digest = self.prediction_digest(&y, input);
        self.last_state_digest = Some(prediction_digest);

        Ok(WorldModelOutput {
            prediction_digest,
            state_digest: prediction_digest,
            prediction_error: err,
            surprise: err,
            state_norm: state_norm_01(&input.obs_features),
            quality: StageQuality::Ok,
            notes: vec![format!("model={}", hex::encode(&self.model_hash[..6]))],
        }
        .bounded())
    }
}

#[derive(Debug, Clone)]
pub struct BurnSaeExtractor {
    w: Vec<f32>,
    b: Vec<f32>,
    f: usize,
    d: usize,
    model_hash: [u8; 32],
}

impl BurnSaeExtractor {
    pub fn new(model_hash: [u8; 32]) -> Self {
        let mut w = vec![0.0; SAE_FEATURE_DIM * SAE_INPUT_DIM];
        let mut b = vec![0.0; SAE_FEATURE_DIM];
        for i in 0..SAE_FEATURE_DIM {
            b[i] = ((model_hash[(i + 5) % 32] as f32 / 255.0) - 0.5) * 0.15;
            for j in 0..SAE_INPUT_DIM {
                w[i * SAE_INPUT_DIM + j] =
                    ((model_hash[(i * SAE_INPUT_DIM + j + 13) % 32] as f32 / 255.0) - 0.5) * 0.3;
            }
        }
        Self {
            w,
            b,
            f: SAE_FEATURE_DIM,
            d: SAE_INPUT_DIM,
            model_hash,
        }
    }

    fn degraded(input: &SaeInput, reason: &'static str) -> SaeOutput {
        let mut hasher = Sha256::new();
        hasher.update(DEGRADED_MARKER);
        hasher.update(b"sae");
        hasher.update(reason.as_bytes());
        hasher.update(input.t.to_le_bytes());
        hasher.update(input.evidence_chain_digest);
        SaeOutput {
            spikes: Vec::new(),
            spike_count: 0,
            sparsity: 1.0,
            energy: 0.0,
            spikes_digest: hasher.finalize().into(),
            quality: StageQuality::DegradedFallback,
            notes: SmallNotes(vec![format!("violation:{reason}")]),
        }
    }

    fn spikes_digest(&self, spikes: &[Spike], input: &SaeInput) -> [u8; 32] {
        let mut sorted = spikes.to_vec();
        sorted.sort_by_key(|s| s.feature_id);
        let mut hasher = Sha256::new();
        for spike in sorted {
            hasher.update(spike.feature_id.to_le_bytes());
            let sign: i16 = if spike.magnitude >= 0.0 { 1 } else { -1 };
            hasher.update(sign.to_le_bytes());
            hasher.update(quantize_unit_u16(spike.magnitude.abs().clamp(0.0, 1.0)).to_le_bytes());
        }
        hasher.update(self.model_hash);
        hasher.update(input.t.to_le_bytes());
        hasher.update(input.evidence_chain_digest);
        hasher.finalize().into()
    }
}

impl Default for BurnSaeExtractor {
    fn default() -> Self {
        Self::new([0x11; 32])
    }
}

impl SaeExtractor for BurnSaeExtractor {
    fn name(&self) -> &'static str {
        "burn_sae_v1"
    }

    fn extract(&self, input: &SaeInput, _budget: ComputeBudget) -> Result<SaeOutput, ComputeError> {
        let mut y = vec![0.0_f32; self.f];
        for (i, yi) in y.iter_mut().enumerate() {
            let mut acc = self.b[i];
            for j in 0..self.d {
                acc += self.w[i * self.d + j] * input.context_features[j];
            }
            *yi = acc;
        }

        if y.iter().any(|v| !v.is_finite()) {
            return Ok(Self::degraded(input, "nan_inf").bounded());
        }

        let mut rank: Vec<(usize, f32)> = y
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.max(0.0).clamp(0.0, 1.0)))
            .collect();
        rank.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let mut spikes = Vec::new();
        for (fid, mag) in rank.into_iter().take(SAE_TOP_K) {
            if mag == 0.0 {
                continue;
            }
            spikes.push(Spike {
                feature_id: fid as u32,
                magnitude: mag,
                timestamp: input.t,
            });
        }
        spikes.sort_by_key(|s| s.feature_id);

        let energy = (y.iter().map(|v| v.abs()).sum::<f32>() / self.f as f32).clamp(0.0, 1.0);
        let spikes_digest = self.spikes_digest(&spikes, input);

        Ok(SaeOutput {
            spike_count: spikes.len() as u16,
            sparsity: (1.0 - spikes.len() as f32 / self.f as f32).clamp(0.0, 1.0),
            energy,
            spikes,
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

#[derive(Debug, Clone)]
pub struct BurnSsmKernel {
    state: [f32; SSM_STATE_DIM],
    a: [f32; SSM_STATE_DIM],
    b: [f32; SSM_STATE_DIM],
    c: [f32; SSM_STATE_DIM],
    model_hash: [u8; 32],
}

impl BurnSsmKernel {
    pub fn new(model_hash: [u8; 32]) -> Self {
        let mut a = [0.0; SSM_STATE_DIM];
        let mut b = [0.0; SSM_STATE_DIM];
        let mut c = [0.0; SSM_STATE_DIM];
        for i in 0..SSM_STATE_DIM {
            a[i] = 0.9 + (model_hash[i % 32] as f32 / 255.0) * 0.08;
            b[i] = ((model_hash[(i + 7) % 32] as f32 / 255.0) * 2.0 - 1.0) * 0.2;
            c[i] = ((model_hash[(i + 13) % 32] as f32 / 255.0) * 2.0 - 1.0) * 0.5;
        }
        Self {
            state: [0.0; SSM_STATE_DIM],
            a,
            b,
            c,
            model_hash,
        }
    }

    fn degraded(input: &SsmInput, reason: &'static str) -> SsmOutput {
        let mut hasher = Sha256::new();
        hasher.update(DEGRADED_MARKER);
        hasher.update(b"ssm");
        hasher.update(reason.as_bytes());
        hasher.update(input.t.to_le_bytes());
        hasher.update(input.context_digest);
        let digest: [u8; 32] = hasher.finalize().into();
        SsmOutput {
            pressure: 1.0,
            state_digest: digest,
            readout_digest: digest,
            state_norm: 1.0,
            readout: 1.0,
            quality: StageQuality::DegradedFallback,
            notes: SmallNotes(vec![format!("violation:{reason}")]),
        }
    }
}

impl SsmKernel for BurnSsmKernel {
    fn name(&self) -> &'static str {
        "burn_ssm_v1"
    }

    fn step(
        &mut self,
        input: &SsmInput,
        _budget: ComputeBudget,
    ) -> Result<SsmOutput, ComputeError> {
        let inp = (0.5 * (f32::from(input.spike_count) / SAE_TOP_K as f32)
            + 0.3 * input.sae_energy
            + 0.2 * input.world_surprise)
            .clamp(0.0, 1.0);

        for i in 0..SSM_STATE_DIM {
            self.state[i] = (self.a[i] * self.state[i] + self.b[i] * inp).clamp(-1.0, 1.0);
        }

        if self.state.iter().any(|v| !v.is_finite()) {
            return Ok(Self::degraded(input, "nan_inf"));
        }

        let mean_abs = self.state.iter().map(|v| v.abs()).sum::<f32>() / SSM_STATE_DIM as f32;
        let pressure = mean_abs.clamp(0.0, 1.0);
        let readout = ((0..SSM_STATE_DIM)
            .map(|i| self.state[i] * self.c[i])
            .sum::<f32>()
            / SSM_STATE_DIM as f32
            * 0.5
            + 0.5)
            .clamp(0.0, 1.0);

        let mut state_h = Sha256::new();
        for v in self.state {
            state_h.update(quantize_signed_unit(v).to_le_bytes());
        }
        state_h.update(self.model_hash);
        state_h.update(input.t.to_le_bytes());
        state_h.update(input.spikes_digest);
        let state_digest: [u8; 32] = state_h.finalize().into();

        let mut read_h = Sha256::new();
        read_h.update(state_digest);
        read_h.update(quantize_unit_u16(readout).to_le_bytes());
        let readout_digest: [u8; 32] = read_h.finalize().into();

        Ok(SsmOutput {
            pressure,
            state_digest,
            readout_digest,
            state_norm: pressure,
            readout,
            quality: StageQuality::Ok,
            notes: SmallNotes(vec![format!(
                "model={}",
                hex::encode(&self.model_hash[..6])
            )]),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn burn_sae_tie_break_is_stable() {
        let mut w = vec![0.0; SAE_FEATURE_DIM * SAE_INPUT_DIM];
        let mut b = vec![0.0; SAE_FEATURE_DIM];
        w[0] = 1.0;
        w[SAE_INPUT_DIM] = 1.0;
        b[0] = 0.0;
        b[1] = 0.0;
        let sae = BurnSaeExtractor {
            w,
            b,
            f: SAE_FEATURE_DIM,
            d: SAE_INPUT_DIM,
            model_hash: [7; 32],
        };
        let mut input = SaeInput {
            t: 1,
            context_features: [0.0; SAE_INPUT_DIM],
            world_state_digest: None,
            seed: 0,
            evidence_chain_digest: [1; 32],
        };
        input.context_features[0] = 1.0;
        let out = sae
            .extract(&input, ComputeBudget::default())
            .expect("extract");
        assert_eq!(out.spikes[0].feature_id, 0);
        assert_eq!(out.spikes[1].feature_id, 1);
    }
}
