use sha2::{Digest, Sha256};

use crate::candle_weights::{
    backend_disable_for_weight_error, load_verified_slot_raw, spec_for_slot,
};
use crate::capabilities::{SaeExtractor, WorldModelPredictor};
use crate::evidence::{quantize_signed_unit, quantize_unit_u16};
use crate::feature_extractor::{
    SaeInput, SaeOutput, SmallNotes, SAE_FEATURE_DIM, SAE_INPUT_DIM, SAE_TOP_K,
};
use crate::model_store::ModelStore;
use crate::ssm::{SsmInput, SsmKernel, SsmOutput, SSM_STATE_DIM};
use crate::world_model::{
    state_norm_01, StageQuality, WorldModelInput, WorldModelOutput, WORLD_MODEL_FEATURE_DIM,
};
use crate::{ComputeBudget, ComputeError, Spike};

const DEGRADED_MARKER: &[u8] = b"degraded_v1";

#[derive(Debug, Clone)]
pub struct CandleWorldPredictor {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    d: usize,
    h: usize,
    model_hash: [u8; 32],
}

impl CandleWorldPredictor {
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
        hasher.finalize().into()
    }
}

impl WorldModelPredictor for CandleWorldPredictor {
    fn name(&self) -> &'static str {
        "candle_jepa_v1"
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

        let mut err = 0.0_f32;
        for (yi, xi) in y.iter().zip(x.iter()) {
            err += (yi - xi).abs();
        }
        err = (err / self.d as f32).clamp(0.0, 1.0);
        let prediction_digest = self.prediction_digest(&y, input);

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
pub struct CandleSaeExtractor {
    w: Vec<f32>,
    b: Vec<f32>,
    f: usize,
    d: usize,
    model_hash: [u8; 32],
}

impl CandleSaeExtractor {
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
pub struct CandleSsmKernel {
    state: [f32; SSM_STATE_DIM],
    a: [f32; SSM_STATE_DIM],
    b: [f32; SSM_STATE_DIM],
    c: Option<[f32; SSM_STATE_DIM]>,
    model_hash: [u8; 32],
}

impl CandleSsmKernel {
    pub fn from_model_store(store: &ModelStore) -> Result<Self, ComputeError> {
        let verified = store
            .verify_slot(crate::model_store::ModelSlot::Ssm)
            .map_err(|_| ComputeError::BackendDisabled)?;
        let spec = spec_for_slot(crate::model_store::ModelSlot::Ssm, verified.size_bytes);
        let loaded = load_verified_slot_raw(store, &verified, &spec)
            .map_err(|err| backend_disable_for_weight_error(&err))?;

        let Some(a_t) = loaded.tensors.get("ssm.a") else {
            return Err(ComputeError::BackendDisabled);
        };
        let Some(b_t) = loaded.tensors.get("ssm.b") else {
            return Err(ComputeError::BackendDisabled);
        };
        if a_t.shape.first().copied() != Some(SSM_STATE_DIM)
            || b_t.shape.first().copied() != Some(SSM_STATE_DIM)
        {
            return Err(ComputeError::BackendDisabled);
        }

        let mut a = [0.0; SSM_STATE_DIM];
        a.copy_from_slice(&a_t.values_f32[..SSM_STATE_DIM]);
        let mut b = [0.0; SSM_STATE_DIM];
        b.copy_from_slice(&b_t.values_f32[..SSM_STATE_DIM]);

        let c = loaded.tensors.get("ssm.c").map(|tensor| {
            let mut out = [0.0; SSM_STATE_DIM];
            out.copy_from_slice(&tensor.values_f32[..SSM_STATE_DIM]);
            out
        });
        let state = loaded
            .tensors
            .get("ssm.init")
            .map(|tensor| {
                let mut out = [0.0; SSM_STATE_DIM];
                out.copy_from_slice(&tensor.values_f32[..SSM_STATE_DIM]);
                out
            })
            .unwrap_or([0.0; SSM_STATE_DIM]);

        Ok(Self {
            state,
            a,
            b,
            c,
            model_hash: verified.sha256,
        })
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

#[cfg(test)]
impl CandleSsmKernel {
    fn for_test(a: f32, b: f32, model_hash: [u8; 32]) -> Self {
        Self {
            state: [0.0; SSM_STATE_DIM],
            a: [a; SSM_STATE_DIM],
            b: [b; SSM_STATE_DIM],
            c: None,
            model_hash,
        }
    }
}

impl SsmKernel for CandleSsmKernel {
    fn name(&self) -> &'static str {
        "candle_ssm_v2_adapter_v0"
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
        let inp_i = ((quantize_unit_u16(inp) as f32) / (u16::MAX as f32)).clamp(0.0, 1.0);

        for i in 0..SSM_STATE_DIM {
            let mut next = self.a[i] * self.state[i] + self.b[i] * inp_i;
            if let Some(c) = self.c {
                next += c[i];
            }
            self.state[i] = next.clamp(-1.0, 1.0);
        }

        if self.state.iter().any(|v| !v.is_finite()) {
            return Ok(Self::degraded(input, "nan_inf"));
        }

        let mut abs_sum_q = 0_u32;
        for value in self.state {
            abs_sum_q = abs_sum_q.saturating_add(u32::from(quantize_unit_u16(value.abs())));
        }
        let mean_abs_q = (abs_sum_q / (SSM_STATE_DIM as u32)) as u16;
        let pressure = f32::from(mean_abs_q) / f32::from(u16::MAX);
        let readout = if let Some(c) = self.c {
            let mut acc = 0.0_f32;
            for (i, coeff) in c.iter().enumerate() {
                acc += self.state[i] * *coeff;
            }
            (acc / SSM_STATE_DIM as f32 * 0.5 + 0.5).clamp(0.0, 1.0)
        } else {
            pressure
        };

        let mut state_h = Sha256::new();
        for v in self.state {
            state_h.update(quantize_signed_unit(v).to_le_bytes());
        }
        state_h.update(self.model_hash);
        state_h.update(input.t.to_le_bytes());
        state_h.update(input.spikes_digest);
        state_h.update(input.context_digest);
        state_h.update(quantize_unit_u16(input.world_surprise).to_le_bytes());
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
                "model={} mode=shadow_only",
                hex::encode(&self.model_hash[..6])
            )]),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sae_tie_break_is_stable() {
        let mut w = vec![0.0; SAE_FEATURE_DIM * SAE_INPUT_DIM];
        let mut b = vec![0.0; SAE_FEATURE_DIM];
        w[0] = 1.0;
        w[SAE_INPUT_DIM] = 1.0;
        b[0] = 0.0;
        b[1] = 0.0;
        let sae = CandleSaeExtractor {
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

    #[test]
    fn ssm_pressure_is_bounded() {
        let mut ssm = CandleSsmKernel::for_test(0.9, 0.1, [9; 32]);
        let inp = SsmInput {
            t: 4,
            spikes_digest: [1; 32],
            spike_count: 1000,
            sae_energy: 1.0,
            world_surprise: 1.0,
            risk: 0.0,
            seed: 0,
            context_digest: [2; 32],
        };
        let out = ssm.step(&inp, ComputeBudget::default()).expect("step");
        assert!((0.0..=1.0).contains(&out.pressure));
    }

    #[test]
    fn ssm_digest_is_stable_for_same_inputs() {
        let mut a = CandleSsmKernel::for_test(0.93, 0.04, [4; 32]);
        let mut b = CandleSsmKernel::for_test(0.93, 0.04, [4; 32]);
        let inp = SsmInput {
            t: 7,
            spikes_digest: [1; 32],
            spike_count: 12,
            sae_energy: 0.2,
            world_surprise: 0.3,
            risk: 0.1,
            seed: 0,
            context_digest: [2; 32],
        };
        let out_a = a.step(&inp, ComputeBudget::default()).expect("a");
        let out_b = b.step(&inp, ComputeBudget::default()).expect("b");
        assert_eq!(out_a.state_digest, out_b.state_digest);
        assert_eq!(out_a.pressure, out_b.pressure);
    }
}
