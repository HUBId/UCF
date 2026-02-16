use sha2::{Digest, Sha256};

use crate::evidence::{quantize_signed_unit, quantize_unit_u16};
use crate::feature_extractor::SmallNotes;
use crate::world_model::StageQuality;
use crate::{ComputeBudget, ComputeError};

pub const LFM_STATE_DIM: usize = 32;
const LFM_SCHEMA_V1: u16 = 1;
const LFM_WORK_SCALE: u64 = 8;
const LFM_FIXTURE_JSON: &str = include_str!("../fixtures/lfm_params_v1.json");
const LFM_DEGRADED_MARKER: [u8; 32] = [0xEE; 32];

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LfmInput {
    pub t: u64,
    pub context_digest: [u8; 32],
    pub world_digest: [u8; 32],
    pub surprise: f32,
    pub spikes_digest: [u8; 32],
    pub spike_count: u16,
    pub sae_energy: f32,
    pub pressure: f32,
    pub coherence: Option<f32>,
    pub instability: Option<f32>,
    pub hormone_stress: Option<f32>,
    pub neuro_arousal: Option<f32>,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LfmOutput {
    pub liquid_state_digest: [u8; 32],
    pub liquid_readout_digest: [u8; 32],
    pub uncertainty: f32,
    pub stability: f32,
    pub state_norm: f32,
    pub quality: StageQuality,
    pub notes: SmallNotes,
}

impl LfmOutput {
    pub fn degraded(reason: &'static str) -> Self {
        Self {
            liquid_state_digest: LFM_DEGRADED_MARKER,
            liquid_readout_digest: LFM_DEGRADED_MARKER,
            uncertainty: 1.0,
            stability: 0.0,
            state_norm: 1.0,
            quality: StageQuality::DegradedFallback,
            notes: SmallNotes(vec![format!("degraded:{reason}")]),
        }
    }
}

pub trait LfmKernel: Send + Sync {
    fn name(&self) -> &'static str;
    fn reset_session(&mut self, seed: u64);
    fn step(&mut self, input: &LfmInput, budget: ComputeBudget) -> Result<LfmOutput, ComputeError>;
}

#[derive(Debug, Clone, Copy)]
struct LfmFixture {
    kmax: f32,
    state_scale: f32,
    w: [f32; 8],
    a: [f32; LFM_STATE_DIM],
    b: [f32; LFM_STATE_DIM],
    c: [f32; LFM_STATE_DIM],
    digest: [u8; 32],
}

impl LfmFixture {
    fn parse_json(raw: &str) -> Result<Self, ComputeError> {
        #[derive(serde::Deserialize)]
        struct Formula {
            modulus: u32,
            mul_i: u32,
            add: u32,
            scale: f32,
            bias: f32,
        }

        #[derive(serde::Deserialize)]
        struct LfmFixtureJson {
            schema_version: u16,
            n: usize,
            kmax: u16,
            state_scale: f32,
            w1: f32,
            w2: f32,
            w3: f32,
            w4: f32,
            w5: f32,
            w6: f32,
            w7: f32,
            w8: f32,
            decay_formula: Formula,
            gain_formula: Formula,
            readout_formula: Formula,
            fixture_digest_hex: String,
        }

        let parsed: LfmFixtureJson =
            serde_json::from_str(raw).map_err(|err| ComputeError::InvalidInput {
                reason: format!("invalid LFM fixture json: {err}"),
            })?;

        if parsed.schema_version != LFM_SCHEMA_V1 || parsed.n != LFM_STATE_DIM {
            return Err(ComputeError::InvalidInput {
                reason: format!(
                    "unsupported LFM fixture schema={} n={}",
                    parsed.schema_version, parsed.n
                ),
            });
        }

        fn gen(formula: &Formula) -> [f32; LFM_STATE_DIM] {
            let mut out = [0.0_f32; LFM_STATE_DIM];
            let modulus = formula.modulus.max(1);
            for (i, value) in out.iter_mut().enumerate() {
                let numerator = ((i as u32)
                    .saturating_mul(formula.mul_i)
                    .saturating_add(formula.add))
                    % modulus;
                let normalized = (numerator as f64) / (modulus as f64);
                *value = (normalized * f64::from(formula.scale) + f64::from(formula.bias)) as f32;
            }
            out
        }

        let w = [
            parsed.w1, parsed.w2, parsed.w3, parsed.w4, parsed.w5, parsed.w6, parsed.w7, parsed.w8,
        ];
        let a = gen(&parsed.decay_formula);
        let b = gen(&parsed.gain_formula);
        let c = gen(&parsed.readout_formula);

        let mut canonical = Vec::with_capacity(4 * 16 + 4 * LFM_STATE_DIM * 3);
        canonical.extend_from_slice(&parsed.schema_version.to_le_bytes());
        canonical.extend_from_slice(&(parsed.n as u16).to_le_bytes());
        canonical.extend_from_slice(&parsed.kmax.to_le_bytes());
        canonical.extend_from_slice(&parsed.state_scale.to_bits().to_le_bytes());
        for value in w {
            canonical.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        for value in a.into_iter().chain(b).chain(c) {
            canonical.extend_from_slice(&value.to_bits().to_le_bytes());
        }

        let expected: [u8; 32] = Sha256::digest(&canonical).into();
        let decoded =
            hex::decode(parsed.fixture_digest_hex).map_err(|err| ComputeError::InvalidInput {
                reason: format!("invalid LFM fixture digest hex: {err}"),
            })?;
        if decoded.len() != 32 {
            return Err(ComputeError::InvalidInput {
                reason: "invalid LFM fixture digest length".to_string(),
            });
        }
        let mut digest = [0_u8; 32];
        digest.copy_from_slice(&decoded);
        if digest != expected {
            tracing::warn!(
                "lfm fixture digest mismatch embedded={} computed={}",
                hex::encode(digest),
                hex::encode(expected)
            );
            digest = expected;
        }

        Ok(Self {
            kmax: f32::from(parsed.kmax.max(1)),
            state_scale: parsed.state_scale.max(0.1),
            w,
            a,
            b,
            c,
            digest,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ToyLfmKernel {
    x: [f32; LFM_STATE_DIM],
    fixture: LfmFixture,
}

impl Default for ToyLfmKernel {
    fn default() -> Self {
        let fixture =
            LfmFixture::parse_json(LFM_FIXTURE_JSON).expect("embedded LFM fixture must be valid");
        let mut this = Self {
            x: [0.0; LFM_STATE_DIM],
            fixture,
        };
        this.reset_session(0);
        this
    }
}

#[cfg(feature = "lfm-candle")]
#[derive(Debug, Clone)]
pub struct CandleLfmKernel {
    fixture: LfmFixture,
    x_shadow: Vec<f32>,
}

#[cfg(feature = "lfm-candle")]
impl Default for CandleLfmKernel {
    fn default() -> Self {
        let fixture =
            LfmFixture::parse_json(LFM_FIXTURE_JSON).expect("embedded LFM fixture must be valid");
        let mut this = Self {
            fixture,
            x_shadow: vec![0.0; LFM_STATE_DIM],
        };
        this.reset_session(0);
        this
    }
}

#[cfg(feature = "lfm-candle")]
impl LfmKernel for CandleLfmKernel {
    fn name(&self) -> &'static str {
        "candle_lfm_liquid_dynamics_v1"
    }

    fn reset_session(&mut self, seed: u64) {
        let mut hasher = Sha256::new();
        hasher.update(self.fixture.digest);
        hasher.update(seed.to_le_bytes());
        let bytes: [u8; 32] = hasher.finalize().into();
        for (idx, value) in self.x_shadow.iter_mut().enumerate() {
            let b = bytes[idx % bytes.len()];
            let centered = ((f32::from(b) / 255.0) * 2.0) - 1.0;
            *value = centered.clamp(-1.0, 1.0);
        }
    }

    fn step(&mut self, input: &LfmInput, budget: ComputeBudget) -> Result<LfmOutput, ComputeError> {
        use candle_core::{Device, Tensor};

        let mut work_units = 16_u64;
        ToyLfmKernel::check_budget(work_units, budget)?;
        let u = ToyLfmKernel {
            x: [0.0; LFM_STATE_DIM],
            fixture: self.fixture,
        }
        .drive(input);
        let mask = ToyLfmKernel::select_mask(input.spikes_digest);

        let mut gated = vec![0.0_f32; LFM_STATE_DIM];
        for idx in 0..LFM_STATE_DIM {
            work_units = work_units.saturating_add(4);
            ToyLfmKernel::check_budget(work_units, budget)?;
            gated[idx] = if mask[idx] {
                self.fixture.b[idx] * u
            } else {
                0.0
            };
        }

        let device = Device::Cpu;
        let x = Tensor::from_slice(&self.x_shadow, LFM_STATE_DIM, &device).map_err(|err| {
            ComputeError::Internal {
                reason: format!("lfm candle x tensor: {err}"),
            }
        })?;
        let a = Tensor::from_slice(&self.fixture.a, LFM_STATE_DIM, &device).map_err(|err| {
            ComputeError::Internal {
                reason: format!("lfm candle a tensor: {err}"),
            }
        })?;
        let decay = Tensor::from_slice(&[0.985_f32; LFM_STATE_DIM], LFM_STATE_DIM, &device)
            .map_err(|err| ComputeError::Internal {
                reason: format!("lfm candle decay tensor: {err}"),
            })?;
        let mask_f32: Vec<f32> = mask
            .iter()
            .map(|enabled| if *enabled { 1.0 } else { 0.0 })
            .collect();
        let mask_t = Tensor::from_slice(&mask_f32, LFM_STATE_DIM, &device).map_err(|err| {
            ComputeError::Internal {
                reason: format!("lfm candle mask tensor: {err}"),
            }
        })?;
        let gated_t = Tensor::from_slice(&gated, LFM_STATE_DIM, &device).map_err(|err| {
            ComputeError::Internal {
                reason: format!("lfm candle gated tensor: {err}"),
            }
        })?;

        let active = a
            .broadcast_mul(&x)
            .and_then(|t| t.broadcast_add(&gated_t))
            .map_err(|err| ComputeError::Internal {
                reason: format!("lfm candle active update: {err}"),
            })?;
        let inactive = decay
            .broadcast_mul(&x)
            .map_err(|err| ComputeError::Internal {
                reason: format!("lfm candle inactive update: {err}"),
            })?;
        let one_minus_mask = Tensor::from_slice(
            &mask_f32
                .iter()
                .map(|v| if *v > 0.5 { 0.0 } else { 1.0 })
                .collect::<Vec<f32>>(),
            LFM_STATE_DIM,
            &device,
        )
        .map_err(|err| ComputeError::Internal {
            reason: format!("lfm candle inverse mask tensor: {err}"),
        })?;
        let merged = mask_t
            .broadcast_mul(&active)
            .and_then(|t| {
                one_minus_mask
                    .broadcast_mul(&inactive)
                    .and_then(|i| t.broadcast_add(&i))
            })
            .map_err(|err| ComputeError::Internal {
                reason: format!("lfm candle merge update: {err}"),
            })?;

        let mut x_next = merged
            .to_vec1::<f32>()
            .map_err(|err| ComputeError::Internal {
                reason: format!("lfm candle readback: {err}"),
            })?;
        for value in &mut x_next {
            *value = value.clamp(-1.0, 1.0);
        }
        self.x_shadow.copy_from_slice(&x_next);

        let mut readout = 0.0_f32;
        for idx in 0..LFM_STATE_DIM {
            work_units = work_units.saturating_add(2);
            ToyLfmKernel::check_budget(work_units, budget)?;
            readout += self.fixture.c[idx] * self.x_shadow[idx];
        }

        let state_norm = (self.x_shadow.iter().map(|v| v.abs()).sum::<f32>()
            / LFM_STATE_DIM as f32
            / self.fixture.state_scale)
            .clamp(0.0, 1.0);
        let uncertainty = (0.6 * u + 0.4 * state_norm).clamp(0.0, 1.0);
        let stability = (1.0 - uncertainty).clamp(0.0, 1.0);

        let mut hasher = Sha256::new();
        hasher.update(self.fixture.digest);
        hasher.update(input.t.to_le_bytes());
        hasher.update(input.context_digest);
        hasher.update(input.world_digest);
        for value in &self.x_shadow {
            hasher.update(quantize_signed_unit(value).to_le_bytes());
        }
        let liquid_state_digest: [u8; 32] = hasher.finalize().into();

        let mut readout_hasher = Sha256::new();
        readout_hasher.update(quantize_unit_u16(readout).to_le_bytes());
        readout_hasher.update(liquid_state_digest);
        let liquid_readout_digest: [u8; 32] = readout_hasher.finalize().into();

        Ok(LfmOutput {
            liquid_state_digest,
            liquid_readout_digest,
            uncertainty,
            stability,
            state_norm,
            quality: StageQuality::Ok,
            notes: SmallNotes(vec![format!(
                "fixture={}",
                hex::encode(&self.fixture.digest[..6])
            )]),
        })
    }
}

#[cfg(feature = "lfm-burn")]
#[derive(Debug, Clone, Default)]
pub struct BurnLfmKernel;

#[cfg(feature = "lfm-burn")]
impl LfmKernel for BurnLfmKernel {
    fn name(&self) -> &'static str {
        "burn_lfm_liquid_dynamics_v0"
    }

    fn reset_session(&mut self, _seed: u64) {}

    fn step(
        &mut self,
        _input: &LfmInput,
        _budget: ComputeBudget,
    ) -> Result<LfmOutput, ComputeError> {
        Err(ComputeError::BackendDisabled)
    }
}

impl ToyLfmKernel {
    fn check_budget(work_units: u64, budget: ComputeBudget) -> Result<(), ComputeError> {
        let elapsed_micros = work_units / LFM_WORK_SCALE;
        if work_units > budget.max_micros.saturating_mul(LFM_WORK_SCALE) {
            return Err(ComputeError::BudgetExceeded {
                stage: "lfm/step",
                elapsed_micros,
                limit_micros: budget.max_micros,
            });
        }
        Ok(())
    }

    fn drive(&self, input: &LfmInput) -> f32 {
        let spikes = (f32::from(input.spike_count) / self.fixture.kmax).clamp(0.0, 1.0);
        let coherence_penalty = 1.0 - input.coherence.unwrap_or(1.0).clamp(0.0, 1.0);
        (self.fixture.w[0] * input.pressure.clamp(0.0, 1.0)
            + self.fixture.w[1] * input.surprise.clamp(0.0, 1.0)
            + self.fixture.w[2] * spikes
            + self.fixture.w[3] * input.sae_energy.clamp(0.0, 1.0)
            + self.fixture.w[4] * coherence_penalty
            + self.fixture.w[5] * input.instability.unwrap_or(0.0).clamp(0.0, 1.0)
            + self.fixture.w[6] * input.hormone_stress.unwrap_or(0.0).clamp(0.0, 1.0)
            + self.fixture.w[7] * input.neuro_arousal.unwrap_or(0.0).clamp(0.0, 1.0))
        .clamp(0.0, 1.0)
    }

    fn select_mask(spikes_digest: [u8; 32]) -> [bool; LFM_STATE_DIM] {
        let mut mask = [false; LFM_STATE_DIM];
        let target = 1 + usize::from(spikes_digest[0]) % (LFM_STATE_DIM / 3);
        let mut selected = 0usize;
        for offset in 0..(spikes_digest.len() * 4) {
            let byte = spikes_digest[offset % spikes_digest.len()];
            let idx = (usize::from(byte) + offset) % LFM_STATE_DIM;
            if !mask[idx] {
                mask[idx] = true;
                selected += 1;
                if selected >= target {
                    break;
                }
            }
        }
        mask
    }

    fn state_digest(&self, input: &LfmInput) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.fixture.digest);
        hasher.update(input.t.to_le_bytes());
        hasher.update(input.context_digest);
        hasher.update(input.world_digest);
        for value in self.x {
            hasher.update(quantize_signed_unit(value).to_le_bytes());
        }
        hasher.finalize().into()
    }
}

impl LfmKernel for ToyLfmKernel {
    fn name(&self) -> &'static str {
        "toy_lfm_liquid_dynamics_v0"
    }

    fn reset_session(&mut self, seed: u64) {
        let mut hasher = Sha256::new();
        hasher.update(self.fixture.digest);
        hasher.update(seed.to_le_bytes());
        let bytes: [u8; 32] = hasher.finalize().into();
        for (idx, value) in self.x.iter_mut().enumerate() {
            let b = bytes[idx % bytes.len()];
            let centered = ((f32::from(b) / 255.0) * 2.0) - 1.0;
            *value = centered.clamp(-1.0, 1.0);
        }
    }

    fn step(&mut self, input: &LfmInput, budget: ComputeBudget) -> Result<LfmOutput, ComputeError> {
        let mut work_units = 16_u64;
        Self::check_budget(work_units, budget)?;
        let u = self.drive(input);
        let mask = Self::select_mask(input.spikes_digest);

        for (idx, x) in self.x.iter_mut().enumerate() {
            work_units = work_units.saturating_add(4);
            Self::check_budget(work_units, budget)?;
            if mask[idx] {
                *x = (self.fixture.a[idx] * *x + self.fixture.b[idx] * u).clamp(-1.0, 1.0);
            } else {
                *x = (0.985 * *x).clamp(-1.0, 1.0);
            }
        }

        let mut readout = 0.0_f32;
        for idx in 0..LFM_STATE_DIM {
            work_units = work_units.saturating_add(2);
            Self::check_budget(work_units, budget)?;
            readout += self.fixture.c[idx] * self.x[idx];
        }

        let state_norm = (self.x.iter().map(|v| v.abs()).sum::<f32>()
            / LFM_STATE_DIM as f32
            / self.fixture.state_scale)
            .clamp(0.0, 1.0);
        let uncertainty = (0.6 * u + 0.4 * state_norm).clamp(0.0, 1.0);
        let stability = (1.0 - uncertainty).clamp(0.0, 1.0);

        let liquid_state_digest = self.state_digest(input);
        let mut readout_hasher = Sha256::new();
        readout_hasher.update(quantize_unit_u16(readout).to_le_bytes());
        readout_hasher.update(liquid_state_digest);
        let liquid_readout_digest: [u8; 32] = readout_hasher.finalize().into();

        Ok(LfmOutput {
            liquid_state_digest,
            liquid_readout_digest,
            uncertainty,
            stability,
            state_norm,
            quality: StageQuality::Ok,
            notes: SmallNotes(vec![format!(
                "fixture={}",
                hex::encode(&self.fixture.digest[..6])
            )]),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> LfmInput {
        LfmInput {
            t: 3,
            context_digest: [0x42; 32],
            world_digest: [0x7A; 32],
            surprise: 0.3,
            spikes_digest: [0x11; 32],
            spike_count: 12,
            sae_energy: 0.4,
            pressure: 0.5,
            coherence: Some(0.8),
            instability: Some(0.2),
            hormone_stress: Some(0.1),
            neuro_arousal: Some(0.2),
            seed: 7,
        }
    }

    #[test]
    fn deterministic_for_same_sequence() {
        let mut a = ToyLfmKernel::default();
        let mut b = ToyLfmKernel::default();
        a.reset_session(9);
        b.reset_session(9);

        let first_a = a.step(&input(), ComputeBudget::default()).expect("step");
        let first_b = b.step(&input(), ComputeBudget::default()).expect("step");
        assert_eq!(first_a, first_b);

        let mut next = input();
        next.t = 4;
        next.pressure = 0.8;
        let second_a = a.step(&next, ComputeBudget::default()).expect("step");
        let second_b = b.step(&next, ComputeBudget::default()).expect("step");
        assert_eq!(second_a, second_b);
    }

    #[test]
    fn bounded_outputs() {
        let mut kernel = ToyLfmKernel::default();
        let out = kernel
            .step(&input(), ComputeBudget::default())
            .expect("step");
        assert!((0.0..=1.0).contains(&out.uncertainty));
        assert!((0.0..=1.0).contains(&out.stability));
        assert!((0.0..=1.0).contains(&out.state_norm));
    }

    #[test]
    fn pressure_increases_uncertainty() {
        let mut low_kernel = ToyLfmKernel::default();
        let mut high_kernel = ToyLfmKernel::default();

        let mut low = input();
        low.pressure = 0.1;
        let mut high = low;
        high.pressure = 0.9;

        let low_out = low_kernel
            .step(&low, ComputeBudget::default())
            .expect("low");
        let high_out = high_kernel
            .step(&high, ComputeBudget::default())
            .expect("high");
        assert!(high_out.uncertainty >= low_out.uncertainty);
    }

    #[cfg(feature = "lfm-candle")]
    #[test]
    fn candle_deterministic_for_same_sequence() {
        let mut a = CandleLfmKernel::default();
        let mut b = CandleLfmKernel::default();
        a.reset_session(9);
        b.reset_session(9);

        let first_a = a.step(&input(), ComputeBudget::default()).expect("step");
        let first_b = b.step(&input(), ComputeBudget::default()).expect("step");
        assert_eq!(first_a, first_b);

        let mut next = input();
        next.t = 4;
        next.pressure = 0.8;
        let second_a = a.step(&next, ComputeBudget::default()).expect("step");
        let second_b = b.step(&next, ComputeBudget::default()).expect("step");
        assert_eq!(second_a, second_b);
    }

    #[cfg(feature = "lfm-candle")]
    #[test]
    fn candle_matches_toy_contract_outputs() {
        let mut toy = ToyLfmKernel::default();
        let mut candle = CandleLfmKernel::default();
        toy.reset_session(13);
        candle.reset_session(13);
        let out_toy = toy.step(&input(), ComputeBudget::default()).expect("toy");
        let out_candle = candle
            .step(&input(), ComputeBudget::default())
            .expect("candle");
        assert_eq!(out_toy, out_candle);
    }
    #[test]
    fn budget_exceeded_is_deterministic() {
        let mut kernel = ToyLfmKernel::default();
        let err = kernel
            .step(
                &input(),
                ComputeBudget {
                    max_micros: 1,
                    ..ComputeBudget::default()
                },
            )
            .expect_err("must exceed");
        match err {
            ComputeError::BudgetExceeded { stage, .. } => assert_eq!(stage, "lfm/step"),
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
