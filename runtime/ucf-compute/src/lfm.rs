use sha2::{Digest, Sha256};

use crate::evidence::{quantize_signed_unit, quantize_unit_u16};
use crate::feature_extractor::SmallNotes;
use crate::world_model::StageQuality;
use crate::{ComputeBudget, ComputeError};

pub const LFM_STATE_DIM: usize = 32;
const LFM_SCHEMA_V1: u16 = 1;
const LFM_WORK_SCALE: u64 = 8;
const LFM_FIXTURE_JSON: &str = include_str!("../fixtures/lfm_params_v1.json");
#[cfg(feature = "lfm-lnn")]
const LFM_LNN_PARAMS_V1_JSON: &str = include_str!("../fixtures/lfm_lnn_params_v1.json");
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
    pub governor_tier: Option<u8>,
    pub prediction_error: Option<f32>,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LfmOutput {
    pub liquid_state_digest: [u8; 32],
    pub liquid_readout_digest: [u8; 32],
    pub uncertainty: f32,
    pub stability: f32,
    pub state_norm: f32,
    pub deriv_norm: f32,
    pub saturation_ratio: f32,
    pub nan_inf_detected: bool,
    pub quality: StageQuality,
    pub notes: SmallNotes,
    pub plasticity: Option<PlasticityRecord>,
}

impl LfmOutput {
    pub fn degraded(reason: &'static str) -> Self {
        Self {
            liquid_state_digest: LFM_DEGRADED_MARKER,
            liquid_readout_digest: LFM_DEGRADED_MARKER,
            uncertainty: 1.0,
            stability: 0.0,
            state_norm: 1.0,
            deriv_norm: 1.0,
            saturation_ratio: 1.0,
            nan_inf_detected: false,
            quality: StageQuality::DegradedFallback,
            notes: SmallNotes(vec![format!("degraded:{reason}")]),
            plasticity: None,
        }
    }
}

#[cfg(feature = "lfm-lnn")]
const PLASTICITY_MAX_PARAMS: usize = 8;
#[cfg(feature = "lfm-lnn")]
const PLASTICITY_MAX_UPDATES_PER_TICK: usize = 4;
#[cfg(feature = "lfm-lnn")]
const PLASTICITY_DELTA_RESOLUTION: f32 = 1e-3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ParamKey {
    AlphaI(u16),
    WuI(u16),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ParamDelta {
    pub key: ParamKey,
    pub delta_q: i16,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PlasticityInput {
    pub t: u64,
    pub governor_tier: u8,
    pub uncertainty: f32,
    pub coherence: Option<f32>,
    pub pressure: f32,
    pub surprise: f32,
    pub prediction_error: Option<f32>,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PlasticityUpdate {
    pub enabled: bool,
    pub updated_params: Vec<ParamDelta>,
    pub delta_digest: [u8; 32],
    pub params_digest_after: [u8; 32],
    pub quality: StageQuality,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PlasticityRecord {
    pub schema_version: u16,
    pub t: u64,
    pub backend_pack_digest: [u8; 32],
    pub lfm_fixture_digest: [u8; 32],
    pub enabled: bool,
    pub governor_tier: u8,
    pub uncertainty_q: u16,
    pub coherence_q: Option<u16>,
    pub pressure_q: u16,
    pub surprise_q: u16,
    pub prediction_error_q: Option<i16>,
    pub param_deltas: Vec<ParamDelta>,
    pub delta_digest: [u8; 32],
    pub params_digest_after: [u8; 32],
    pub evidence_chain_digest: [u8; 32],
    pub emergency_disabled: bool,
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
        let deriv_norm = 0.0_f32;
        let saturation_ratio =
            self.x_shadow.iter().filter(|v| v.abs() >= 0.999).count() as f32 / LFM_STATE_DIM as f32;
        let nan_inf_detected = self.x_shadow.iter().any(|v| !v.is_finite());
        let uncertainty = (0.6 * u + 0.4 * state_norm).clamp(0.0, 1.0);
        let stability = (1.0 - uncertainty).clamp(0.0, 1.0);

        let mut hasher = Sha256::new();
        hasher.update(self.fixture.digest);
        hasher.update(input.t.to_le_bytes());
        hasher.update(input.context_digest);
        hasher.update(input.world_digest);
        for value in &self.x_shadow {
            hasher.update(quantize_signed_unit(*value).to_le_bytes());
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
            deriv_norm,
            saturation_ratio,
            nan_inf_detected,
            quality: StageQuality::Ok,
            notes: SmallNotes(vec![format!(
                "fixture={}",
                hex::encode(&self.fixture.digest[..6])
            )]),
            plasticity: None,
        })
    }
}

#[cfg(feature = "lfm-lnn")]
#[derive(Debug, Clone)]
struct LnnParamsV1 {
    n: usize,
    dt: f32,
    clamp_state: f32,
    clamp_deriv: f32,
    kmax: f32,
    state_scale: f32,
    drive_w: [f32; 8],
    alpha: Vec<f32>,
    wx: Vec<f32>,
    wu: Vec<f32>,
    b: Vec<f32>,
    digest: [u8; 32],
}

#[cfg(feature = "lfm-lnn")]
impl LnnParamsV1 {
    fn parse_json(raw: &str) -> Result<Self, ComputeError> {
        #[derive(serde::Deserialize)]
        struct LnnFixtureJson {
            schema_version: u16,
            n: usize,
            dt: f32,
            clamp_state: f32,
            clamp_deriv: f32,
            kmax: u16,
            state_scale: f32,
            drive_w: Vec<f32>,
            alpha: Vec<f32>,
            wx: Vec<f32>,
            wu: Vec<f32>,
            b: Vec<f32>,
            fixture_digest_hex: String,
        }

        let parsed: LnnFixtureJson =
            serde_json::from_str(raw).map_err(|err| ComputeError::InvalidInput {
                reason: format!("invalid LNN fixture json: {err}"),
            })?;

        let schema_version = parsed.schema_version;
        if schema_version != 1 {
            tracing::warn!("lfm-lnn invalid schema_version={schema_version}");
            return Err(ComputeError::BackendDisabled);
        }
        let n = parsed.n;
        if n == 0 || n > 64 {
            tracing::warn!("lfm-lnn invalid state size n={n}");
            return Err(ComputeError::BackendDisabled);
        }
        let dt = parsed.dt;
        let clamp_state = parsed.clamp_state;
        let clamp_deriv = parsed.clamp_deriv;
        let kmax = f32::from(parsed.kmax);
        let state_scale = parsed.state_scale;
        if parsed.drive_w.len() != 8
            || parsed.alpha.len() != n
            || parsed.wx.len() != n * n
            || parsed.wu.len() != n
            || parsed.b.len() != n
        {
            tracing::warn!("lfm-lnn invalid fixture array lengths");
            return Err(ComputeError::BackendDisabled);
        }
        let mut drive_w = [0.0_f32; 8];
        drive_w.copy_from_slice(&parsed.drive_w);
        let alpha = parsed.alpha;
        let wx = parsed.wx;
        let wu = parsed.wu;
        let b = parsed.b;
        if !(dt.is_finite() && dt > 0.0 && dt <= 1.0) {
            tracing::warn!("lfm-lnn invalid dt={dt}");
            return Err(ComputeError::BackendDisabled);
        }
        if !(clamp_state.is_finite() && clamp_state > 0.0 && clamp_state <= 1.0) {
            tracing::warn!("lfm-lnn invalid clamp_state={clamp_state}");
            return Err(ComputeError::BackendDisabled);
        }
        if !(clamp_deriv.is_finite() && clamp_deriv > 0.0 && clamp_deriv <= 1.0) {
            tracing::warn!("lfm-lnn invalid clamp_deriv={clamp_deriv}");
            return Err(ComputeError::BackendDisabled);
        }
        if !(state_scale.is_finite() && state_scale > 0.0) {
            tracing::warn!("lfm-lnn invalid state_scale={state_scale}");
            return Err(ComputeError::BackendDisabled);
        }
        for value in alpha.iter().copied() {
            if !value.is_finite() || !(0.1..=2.0).contains(&value) {
                tracing::warn!("lfm-lnn invalid alpha");
                return Err(ComputeError::BackendDisabled);
            }
        }
        for value in wx.iter().chain(&wu).chain(&b).copied() {
            if !value.is_finite() || value.abs() > 1.0 {
                tracing::warn!("lfm-lnn invalid weight magnitude");
                return Err(ComputeError::BackendDisabled);
            }
        }

        Ok(Self {
            n,
            dt,
            clamp_state,
            clamp_deriv,
            kmax: kmax.max(1.0),
            state_scale,
            drive_w,
            alpha,
            wx,
            wu,
            b,
            digest: {
                let computed: [u8; 32] = Sha256::digest(raw.as_bytes()).into();
                match hex::decode(parsed.fixture_digest_hex) {
                    Ok(decoded) if decoded.len() == 32 => {
                        let mut embedded = [0_u8; 32];
                        embedded.copy_from_slice(&decoded);
                        if embedded != computed {
                            tracing::warn!(
                                "lfm-lnn fixture digest mismatch embedded={} computed={}",
                                hex::encode(embedded),
                                hex::encode(computed)
                            );
                        }
                        computed
                    }
                    _ => computed,
                }
            },
        })
    }
}

#[cfg(feature = "lfm-lnn")]
#[derive(Debug, Clone)]
struct LnnRuntimeOverlay {
    alpha: Vec<f32>,
    wu: Vec<f32>,
}

#[cfg(feature = "lfm-lnn")]
#[derive(Debug, Clone)]
pub struct LnnOdeLfmKernel {
    base_params: LnnParamsV1,
    overlay: LnnRuntimeOverlay,
    x: Vec<f32>,
}

#[cfg(feature = "lfm-lnn")]
impl Default for LnnOdeLfmKernel {
    fn default() -> Self {
        let params = LnnParamsV1::parse_json(LFM_LNN_PARAMS_V1_JSON)
            .expect("embedded LNN fixture must be valid");
        let mut this = Self {
            x: vec![0.0_f32; params.n],
            overlay: LnnRuntimeOverlay {
                alpha: params.alpha.clone(),
                wu: params.wu.clone(),
            },
            base_params: params,
        };
        this.reset_session(0);
        this
    }
}

#[cfg(feature = "lfm-lnn")]
impl LnnOdeLfmKernel {
    fn check_budget(&self, budget: ComputeBudget) -> Result<(), ComputeError> {
        let n = self.base_params.n as u64;
        let work_units = 32_u64.saturating_add(2 * n * n).saturating_add(24 * n);
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

    fn quantize_delta(delta: f32) -> i16 {
        (delta / PLASTICITY_DELTA_RESOLUTION)
            .round()
            .clamp(i16::MIN as f32, i16::MAX as f32) as i16
    }

    fn dequantize_delta(delta_q: i16) -> f32 {
        f32::from(delta_q) * PLASTICITY_DELTA_RESOLUTION
    }

    fn params_digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.base_params.digest);
        for v in &self.overlay.alpha {
            hasher.update(quantize_unit_u16((v / 2.0).clamp(0.0, 1.0)).to_le_bytes());
        }
        for v in &self.overlay.wu {
            hasher.update(quantize_signed_unit(v.clamp(-1.0, 1.0)).to_le_bytes());
        }
        hasher.finalize().into()
    }

    fn delta_digest(updated_params: &[ParamDelta]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for delta in updated_params {
            match delta.key {
                ParamKey::AlphaI(i) => {
                    hasher.update([0]);
                    hasher.update(i.to_le_bytes());
                }
                ParamKey::WuI(i) => {
                    hasher.update([1]);
                    hasher.update(i.to_le_bytes());
                }
            }
            hasher.update(delta.delta_q.to_le_bytes());
        }
        hasher.finalize().into()
    }

    fn apply_plasticity(&mut self, input: &PlasticityInput) -> PlasticityUpdate {
        let coherence = input.coherence.unwrap_or(1.0).clamp(0.0, 1.0);
        let enabled = input.governor_tier <= 1 && input.uncertainty <= 0.45 && coherence >= 0.6;
        if !enabled {
            let params_digest_after = self.params_digest();
            return PlasticityUpdate {
                enabled,
                updated_params: Vec::new(),
                delta_digest: Self::delta_digest(&[]),
                params_digest_after,
                quality: StageQuality::Ok,
            };
        }

        let err = input
            .prediction_error
            .unwrap_or(input.surprise)
            .clamp(0.0, 1.0);
        let d =
            (0.8 * (err - 0.4) - 0.7 * (input.pressure.clamp(0.0, 1.0) - 0.65)).clamp(-1.0, 1.0);
        let lr = 0.01_f32;
        let raw_delta = (lr * d).clamp(-0.02, 0.02);
        let delta_q = Self::quantize_delta(raw_delta);
        if delta_q == 0 {
            let params_digest_after = self.params_digest();
            return PlasticityUpdate {
                enabled,
                updated_params: Vec::new(),
                delta_digest: Self::delta_digest(&[]),
                params_digest_after,
                quality: StageQuality::Ok,
            };
        }
        let delta = Self::dequantize_delta(delta_q);
        let n = self.overlay.alpha.len();
        let update_count = PLASTICITY_MAX_UPDATES_PER_TICK
            .min(n)
            .min(PLASTICITY_MAX_PARAMS);
        let mut updated_params = Vec::with_capacity(update_count);

        for k in 0..update_count {
            let idx_seed = input.seed ^ input.t.rotate_left((k as u32) & 31);
            let idx = ((idx_seed as usize)
                .wrapping_add((usize::from(input.governor_tier) + 1) * (k + 3))
                .wrapping_add(usize::from(input.t.to_le_bytes()[k % 8])))
                % n;
            let base = self.base_params.alpha[idx];
            let low = (base - 0.5).max(0.1);
            let high = (base + 0.5).min(2.0);
            let before = self.overlay.alpha[idx];
            let after = (before + delta).clamp(low, high);
            let applied_q = Self::quantize_delta(after - before);
            if applied_q != 0 {
                self.overlay.alpha[idx] =
                    (before + Self::dequantize_delta(applied_q)).clamp(low, high);
                updated_params.push(ParamDelta {
                    key: ParamKey::AlphaI(idx as u16),
                    delta_q: applied_q,
                });
            }
        }
        updated_params.truncate(PLASTICITY_MAX_PARAMS);
        let delta_digest = Self::delta_digest(&updated_params);
        let params_digest_after = self.params_digest();
        PlasticityUpdate {
            enabled,
            updated_params,
            delta_digest,
            params_digest_after,
            quality: StageQuality::Ok,
        }
    }

    fn drive(&self, input: &LfmInput) -> f32 {
        let spikes = (f32::from(input.spike_count) / self.base_params.kmax).clamp(0.0, 1.0);
        let coherence_penalty = 1.0 - input.coherence.unwrap_or(1.0).clamp(0.0, 1.0);
        (self.base_params.drive_w[0] * input.pressure.clamp(0.0, 1.0)
            + self.base_params.drive_w[1] * input.surprise.clamp(0.0, 1.0)
            + self.base_params.drive_w[2] * spikes
            + self.base_params.drive_w[3] * input.sae_energy.clamp(0.0, 1.0)
            + self.base_params.drive_w[4] * coherence_penalty
            + self.base_params.drive_w[5] * input.instability.unwrap_or(0.0).clamp(0.0, 1.0)
            + self.base_params.drive_w[6] * input.hormone_stress.unwrap_or(0.0).clamp(0.0, 1.0)
            + self.base_params.drive_w[7] * input.neuro_arousal.unwrap_or(0.0).clamp(0.0, 1.0))
        .clamp(0.0, 1.0)
    }

    fn deriv(&self, x: &[f32], u: f32, out: &mut [f32]) {
        let n = self.base_params.n;
        for i in 0..n {
            let mut acc = self.base_params.b[i] + self.overlay.wu[i] * u;
            let row = i * n;
            for (j, x_j) in x.iter().enumerate().take(n) {
                acc += self.base_params.wx[row + j] * *x_j;
            }
            let nonlin = acc.tanh();
            out[i] = (-self.overlay.alpha[i] * x[i] + nonlin)
                .clamp(-self.base_params.clamp_deriv, self.base_params.clamp_deriv);
        }
    }
}

#[cfg(feature = "lfm-lnn")]
impl LfmKernel for LnnOdeLfmKernel {
    fn name(&self) -> &'static str {
        "lnn_ode_lfm_rk2_v1"
    }

    fn reset_session(&mut self, seed: u64) {
        let mut hasher = Sha256::new();
        hasher.update(self.base_params.digest);
        hasher.update(seed.to_le_bytes());
        let bytes: [u8; 32] = hasher.finalize().into();
        for (idx, value) in self.x.iter_mut().enumerate() {
            let b0 = bytes[idx % bytes.len()];
            let b1 = bytes[(idx + 7) % bytes.len()];
            let raw = i16::from_le_bytes([b0, b1]);
            let centered = f32::from(raw % 2048) / 2048.0;
            *value = centered.clamp(-self.base_params.clamp_state, self.base_params.clamp_state);
        }
    }

    fn step(&mut self, input: &LfmInput, budget: ComputeBudget) -> Result<LfmOutput, ComputeError> {
        self.check_budget(budget)?;

        let u = self.drive(input);
        let n = self.base_params.n;
        let mut k1 = vec![0.0_f32; n];
        let mut x_mid = vec![0.0_f32; n];
        let mut k2 = vec![0.0_f32; n];

        self.deriv(&self.x, u, &mut k1);
        for i in 0..n {
            x_mid[i] = (self.x[i] + 0.5 * self.base_params.dt * k1[i])
                .clamp(-self.base_params.clamp_state, self.base_params.clamp_state);
        }
        self.deriv(&x_mid, u, &mut k2);
        for (i, k2_i) in k2.iter().enumerate().take(n) {
            self.x[i] = (self.x[i] + self.base_params.dt * *k2_i)
                .clamp(-self.base_params.clamp_state, self.base_params.clamp_state);
        }

        let readout = self.x.iter().sum::<f32>() / n as f32;
        let state_norm =
            (self.x.iter().map(|v| v.abs()).sum::<f32>() / n as f32 / self.base_params.state_scale)
                .clamp(0.0, 1.0);
        let deriv_norm =
            (k2.iter().map(|v| v.abs()).sum::<f32>() / n as f32 / self.base_params.clamp_deriv)
                .clamp(0.0, 1.0);
        let saturation_ratio = self
            .x
            .iter()
            .filter(|v| v.abs() >= self.base_params.clamp_state - 1.0e-6)
            .count() as f32
            / n as f32;
        let nan_inf_detected = self.x.iter().chain(k2.iter()).any(|v| !v.is_finite());
        let uncertainty = (0.5 * u + 0.3 * state_norm + 0.2 * deriv_norm).clamp(0.0, 1.0);
        let stability = (1.0 - uncertainty).clamp(0.0, 1.0);

        let plasticity_input = PlasticityInput {
            t: input.t,
            governor_tier: input.governor_tier.unwrap_or(3),
            uncertainty,
            coherence: input.coherence,
            pressure: input.pressure,
            surprise: input.surprise,
            prediction_error: input.prediction_error,
            seed: input.seed,
        };
        let plasticity_update = self.apply_plasticity(&plasticity_input);

        if plasticity_update.enabled {
            metrics::counter!("ucf_plasticity_enabled_total").increment(1);
        } else {
            let reason = if plasticity_input.governor_tier > 1 {
                "governance"
            } else if plasticity_input.uncertainty > 0.45 {
                "uncertainty"
            } else {
                "coherence"
            };
            metrics::counter!("ucf_plasticity_disabled_total", "reason" => reason).increment(1);
        }
        metrics::counter!("ucf_plasticity_param_updates_total")
            .increment(plasticity_update.updated_params.len() as u64);
        let alpha_mean = self.overlay.alpha.iter().sum::<f32>() / self.overlay.alpha.len() as f32;
        metrics::gauge!("ucf_lfm_alpha_mean").set(f64::from(alpha_mean));

        metrics::counter!("ucf_lfm_lnn_step_total").increment(1);
        metrics::histogram!("ucf_lfm_lnn_deriv_norm").record(f64::from(deriv_norm));

        let mut hasher = Sha256::new();
        hasher.update(self.base_params.digest);
        hasher.update(plasticity_update.params_digest_after);
        hasher.update(input.t.to_le_bytes());
        hasher.update(input.context_digest);
        for value in &self.x {
            let normalized = (*value / self.base_params.clamp_state).clamp(-1.0, 1.0);
            hasher.update(quantize_signed_unit(normalized).to_le_bytes());
        }
        hasher.update(quantize_unit_u16(u).to_le_bytes());
        hasher.update(quantize_unit_u16(uncertainty).to_le_bytes());
        hasher.update(quantize_unit_u16(stability).to_le_bytes());
        let liquid_state_digest: [u8; 32] = hasher.finalize().into();

        let mut readout_hasher = Sha256::new();
        readout_hasher.update(quantize_signed_unit(readout).to_le_bytes());
        readout_hasher.update(liquid_state_digest);
        let liquid_readout_digest: [u8; 32] = readout_hasher.finalize().into();

        Ok(LfmOutput {
            liquid_state_digest,
            liquid_readout_digest,
            uncertainty,
            stability,
            state_norm,
            deriv_norm,
            saturation_ratio,
            nan_inf_detected,
            quality: StageQuality::Ok,
            notes: SmallNotes(vec![
                format!("fixture={}", hex::encode(&self.base_params.digest[..6])),
                format!("deriv_q={}", quantize_unit_u16(deriv_norm)),
            ]),
            plasticity: Some(PlasticityRecord {
                schema_version: 1,
                t: input.t,
                backend_pack_digest: [0; 32],
                lfm_fixture_digest: self.base_params.digest,
                enabled: plasticity_update.enabled,
                governor_tier: plasticity_input.governor_tier,
                uncertainty_q: quantize_unit_u16(plasticity_input.uncertainty),
                coherence_q: plasticity_input.coherence.map(quantize_unit_u16),
                pressure_q: quantize_unit_u16(plasticity_input.pressure),
                surprise_q: quantize_unit_u16(plasticity_input.surprise),
                prediction_error_q: plasticity_input.prediction_error.map(quantize_signed_unit),
                param_deltas: plasticity_update.updated_params,
                delta_digest: plasticity_update.delta_digest,
                params_digest_after: plasticity_update.params_digest_after,
                evidence_chain_digest: [0; 32],
                emergency_disabled: false,
            }),
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
        let x_prev = self.x;

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
        let deriv_norm = (self
            .x
            .iter()
            .zip(x_prev.iter())
            .map(|(x, prev)| (x - prev).abs())
            .sum::<f32>()
            / LFM_STATE_DIM as f32)
            .clamp(0.0, 1.0);
        let saturation_ratio =
            self.x.iter().filter(|v| v.abs() >= 0.999).count() as f32 / LFM_STATE_DIM as f32;
        let nan_inf_detected = self.x.iter().any(|v| !v.is_finite());
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
            deriv_norm,
            saturation_ratio,
            nan_inf_detected,
            quality: StageQuality::Ok,
            notes: SmallNotes(vec![format!(
                "fixture={}",
                hex::encode(&self.fixture.digest[..6])
            )]),
            plasticity: None,
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
            governor_tier: Some(0),
            prediction_error: Some(0.25),
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

    #[cfg(feature = "lfm-lnn")]
    #[test]
    fn lnn_deterministic_for_same_sequence() {
        let mut a = LnnOdeLfmKernel::default();
        let mut b = LnnOdeLfmKernel::default();
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

    #[cfg(feature = "lfm-lnn")]
    #[test]
    fn lnn_rk2_known_case() {
        let mut kernel = LnnOdeLfmKernel::default();
        kernel.reset_session(0);
        for value in &mut kernel.x {
            *value = 0.0;
        }

        let mut inp = input();
        inp.pressure = 0.0;
        inp.surprise = 0.0;
        inp.spike_count = 0;
        inp.sae_energy = 0.0;
        inp.coherence = Some(1.0);
        inp.instability = Some(0.0);
        inp.hormone_stress = Some(0.0);
        inp.neuro_arousal = Some(0.0);

        let out = kernel.step(&inp, ComputeBudget::default()).expect("step");
        assert!((0.0..=1.0).contains(&out.uncertainty));
        assert!(kernel
            .x
            .iter()
            .all(|v| v.abs() <= kernel.base_params.clamp_state + f32::EPSILON));
    }

    #[cfg(feature = "lfm-lnn")]
    #[test]
    fn lnn_pressure_increases_uncertainty() {
        let mut low_kernel = LnnOdeLfmKernel::default();
        let mut high_kernel = LnnOdeLfmKernel::default();

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

    #[cfg(feature = "lfm-lnn")]
    #[test]
    fn lnn_plasticity_disabled_on_high_tier_or_uncertainty() {
        let mut kernel = LnnOdeLfmKernel::default();
        let mut inp = input();
        inp.governor_tier = Some(3);
        let out = kernel.step(&inp, ComputeBudget::default()).expect("step");
        let rec = out.plasticity.expect("plasticity record");
        assert!(!rec.enabled);

        let mut kernel = LnnOdeLfmKernel::default();
        let mut inp2 = input();
        inp2.governor_tier = Some(0);
        inp2.pressure = 1.0;
        inp2.surprise = 1.0;
        let out2 = kernel.step(&inp2, ComputeBudget::default()).expect("step");
        let rec2 = out2.plasticity.expect("plasticity record");
        assert!(!rec2.enabled || rec2.uncertainty_q <= quantize_unit_u16(0.45));
    }

    #[cfg(feature = "lfm-lnn")]
    #[test]
    fn lnn_plasticity_is_bounded_and_deterministic() {
        let mut a = LnnOdeLfmKernel::default();
        let mut b = LnnOdeLfmKernel::default();
        a.reset_session(17);
        b.reset_session(17);
        let mut inp = input();
        inp.governor_tier = Some(0);
        inp.coherence = Some(0.9);
        inp.prediction_error = Some(0.95);
        inp.pressure = 0.2;

        for tick in 1..=32 {
            inp.t = tick;
            let out_a = a.step(&inp, ComputeBudget::default()).expect("a step");
            let out_b = b.step(&inp, ComputeBudget::default()).expect("b step");
            let rec_a = out_a.plasticity.as_ref().expect("plasticity");
            let rec_b = out_b.plasticity.as_ref().expect("plasticity");
            assert_eq!(rec_a, rec_b);
            assert!(rec_a.param_deltas.len() <= PLASTICITY_MAX_UPDATES_PER_TICK);
            for delta in &rec_a.param_deltas {
                assert!(delta.delta_q.abs() <= 20);
            }
            assert!(a.overlay.alpha.iter().all(|v| (0.1..=2.0).contains(v)));
        }
    }

    #[cfg(feature = "lfm-lnn")]
    #[test]
    fn lnn_budget_exceeded_is_deterministic() {
        let mut kernel = LnnOdeLfmKernel::default();
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
