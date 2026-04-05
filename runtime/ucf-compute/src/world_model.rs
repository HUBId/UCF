use sha2::{Digest, Sha256};

use crate::evidence::quantize_signed_unit;
use crate::risk_contract::SignalQuality;
use crate::{ComputeBudget, ComputeError};

use crate::capabilities::WorldModelPredictor;

impl WorldModelPredictor for MockJepaPredictor {
    fn name(&self) -> &'static str {
        "mock_jepa_v0"
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
        budget: ComputeBudget,
    ) -> Result<WorldModelOutput, ComputeError> {
        if !self.initialized {
            self.init_state(input);
        }

        let mut work_units = 16_u64;
        Self::check_budget(work_units, budget)?;

        let mut pred = [0.0_f32; WORLD_MODEL_FEATURE_DIM];
        for (i, pred_i) in pred.iter_mut().enumerate().take(WORLD_MODEL_FEATURE_DIM) {
            let mut value = self.fixture.c[i];
            for (j, state_j) in self.state.iter().enumerate().take(WORLD_MODEL_FEATURE_DIM) {
                work_units = work_units.saturating_add(2);
                Self::check_budget(work_units, budget)?;
                value +=
                    self.fixture.a[i][j] * *state_j + self.fixture.b[i][j] * input.obs_features[j];
            }
            *pred_i = value.clamp(-1.0, 1.0);
        }

        let err = pred
            .iter()
            .zip(input.obs_features.iter())
            .map(|(p, o)| (p - o).abs())
            .sum::<f32>()
            / WORLD_MODEL_FEATURE_DIM as f32;

        let alpha = 0.22_f32;
        for (state, p) in self.state.iter_mut().zip(pred.iter()) {
            *state = (*state + alpha * (*p - *state)).clamp(-1.0, 1.0);
        }

        let prediction_digest =
            Self::digest_vector(&pred, input.seed, input.t, self.fixture.weights_digest);
        let state_digest = Self::digest_vector(
            &self.state,
            input.seed,
            input.t.saturating_add(1),
            self.fixture.weights_digest,
        );
        self.last_state_digest = Some(state_digest);
        let output = WorldModelOutput {
            prediction_digest,
            state_digest,
            prediction_error: err.clamp(0.0, 1.0),
            surprise: err.clamp(0.0, 1.0),
            state_norm: state_norm_01(&self.state),
            quality: StageQuality::Ok,
            notes: vec![
                format!("fixture={}", hex::encode(&self.fixture.weights_digest[..6])),
                format!("t={}", input.t),
            ],
        };
        Ok(output.bounded())
    }
}

pub const WORLD_MODEL_FEATURE_DIM: usize = 16;
pub const WORLD_MODEL_MAX_NOTES: usize = 4;
const JEPA_FIXTURE_SCHEMA_V1: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageQuality {
    Ok,
    DegradedFallback,
    Unavailable,
}

impl StageQuality {
    pub fn as_signal_quality(self) -> SignalQuality {
        match self {
            Self::Ok => SignalQuality::VerifiedPipeline,
            Self::DegradedFallback => SignalQuality::DegradedFallback,
            Self::Unavailable => SignalQuality::Unavailable,
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WorldModelInput {
    pub t: u64,
    pub context_digest: [u8; 32],
    pub previous_state_digest: Option<[u8; 32]>,
    pub obs_features: [f32; WORLD_MODEL_FEATURE_DIM],
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WorldModelOutput {
    pub prediction_digest: [u8; 32],
    pub state_digest: [u8; 32],
    pub prediction_error: f32,
    pub surprise: f32,
    pub state_norm: f32,
    pub quality: StageQuality,
    pub notes: Vec<String>,
}

impl WorldModelOutput {
    pub fn bounded(mut self) -> Self {
        self.prediction_error = self.prediction_error.clamp(0.0, 1.0);
        self.surprise = self.surprise.clamp(0.0, 1.0);
        self.state_norm = self.state_norm.clamp(0.0, 1.0);
        if self.notes.len() > WORLD_MODEL_MAX_NOTES {
            self.notes.truncate(WORLD_MODEL_MAX_NOTES);
        }
        self
    }

    pub fn degraded_budget(stage: &'static str) -> Self {
        Self {
            prediction_digest: [0; 32],
            state_digest: [0; 32],
            prediction_error: 1.0,
            surprise: 1.0,
            state_norm: 0.0,
            quality: StageQuality::DegradedFallback,
            notes: vec![format!("budget_exceeded_stage={stage}")],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WorldFixtureDigest {
    pub schema_version: u16,
    pub k: u16,
    pub weights_digest: [u8; 32],
}

#[derive(Debug, Clone)]
pub struct MockJepaPredictor {
    state: [f32; WORLD_MODEL_FEATURE_DIM],
    last_state_digest: Option<[u8; 32]>,
    initialized: bool,
    fixture: DynFixture,
}

#[derive(Debug, Clone)]
struct DynFixture {
    schema_version: u16,
    k: u16,
    a: [[f32; WORLD_MODEL_FEATURE_DIM]; WORLD_MODEL_FEATURE_DIM],
    b: [[f32; WORLD_MODEL_FEATURE_DIM]; WORLD_MODEL_FEATURE_DIM],
    c: [f32; WORLD_MODEL_FEATURE_DIM],
    weights_digest: [u8; 32],
}

impl DynFixture {
    fn parse_json(raw: &str) -> Result<Self, ComputeError> {
        #[derive(serde::Deserialize)]
        struct DynFixtureJson {
            schema_version: u16,
            k: usize,
            a: Vec<f32>,
            b: Vec<f32>,
            c: Vec<f32>,
            weights_digest_hex: String,
        }

        let parsed: DynFixtureJson =
            serde_json::from_str(raw).map_err(|err| ComputeError::InvalidInput {
                reason: format!("invalid JEPA fixture json: {err}"),
            })?;

        if parsed.schema_version != JEPA_FIXTURE_SCHEMA_V1 || parsed.k != WORLD_MODEL_FEATURE_DIM {
            return Err(ComputeError::InvalidInput {
                reason: format!(
                    "unsupported JEPA fixture schema={} k={}",
                    parsed.schema_version, parsed.k
                ),
            });
        }
        if parsed.a.len() != WORLD_MODEL_FEATURE_DIM * WORLD_MODEL_FEATURE_DIM
            || parsed.b.len() != WORLD_MODEL_FEATURE_DIM * WORLD_MODEL_FEATURE_DIM
            || parsed.c.len() != WORLD_MODEL_FEATURE_DIM
        {
            return Err(ComputeError::InvalidInput {
                reason: "invalid JEPA fixture dimensions".to_string(),
            });
        }

        let mut a = [[0.0_f32; WORLD_MODEL_FEATURE_DIM]; WORLD_MODEL_FEATURE_DIM];
        let mut b = [[0.0_f32; WORLD_MODEL_FEATURE_DIM]; WORLD_MODEL_FEATURE_DIM];
        let mut c = [0.0_f32; WORLD_MODEL_FEATURE_DIM];

        for i in 0..WORLD_MODEL_FEATURE_DIM {
            for j in 0..WORLD_MODEL_FEATURE_DIM {
                a[i][j] = parsed.a[i * WORLD_MODEL_FEATURE_DIM + j];
                b[i][j] = parsed.b[i * WORLD_MODEL_FEATURE_DIM + j];
            }
            c[i] = parsed.c[i];
        }

        let mut canonical =
            Vec::with_capacity((parsed.a.len() + parsed.b.len() + parsed.c.len()) * 4 + 8);
        canonical.extend_from_slice(&parsed.schema_version.to_le_bytes());
        canonical.extend_from_slice(&(parsed.k as u16).to_le_bytes());
        for value in parsed
            .a
            .iter()
            .chain(parsed.b.iter())
            .chain(parsed.c.iter())
        {
            canonical.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        let expected: [u8; 32] = Sha256::digest(&canonical).into();

        let digest_bytes =
            hex::decode(parsed.weights_digest_hex).map_err(|err| ComputeError::InvalidInput {
                reason: format!("invalid JEPA fixture digest hex: {err}"),
            })?;
        if digest_bytes.len() != 32 {
            return Err(ComputeError::InvalidInput {
                reason: "invalid JEPA fixture digest length".to_string(),
            });
        }
        let mut weights_digest = [0_u8; 32];
        weights_digest.copy_from_slice(&digest_bytes);

        if expected != weights_digest {
            return Err(ComputeError::InvalidInput {
                reason: "invalid JEPA fixture digest".to_string(),
            });
        }

        Ok(Self {
            schema_version: parsed.schema_version,
            k: parsed.k as u16,
            a,
            b,
            c,
            weights_digest,
        })
    }

    fn digest(&self) -> WorldFixtureDigest {
        WorldFixtureDigest {
            schema_version: self.schema_version,
            k: self.k,
            weights_digest: self.weights_digest,
        }
    }
}

impl Default for MockJepaPredictor {
    fn default() -> Self {
        let fixture = DynFixture::parse_json(include_str!("../fixtures/jepa_dyn_v1.json"))
            .expect("embedded JEPA fixture must be valid");
        Self {
            state: [0.0; WORLD_MODEL_FEATURE_DIM],
            last_state_digest: None,
            initialized: false,
            fixture,
        }
    }
}

impl MockJepaPredictor {
    fn check_budget(work_units: u64, budget: ComputeBudget) -> Result<(), ComputeError> {
        const SCALE: u64 = 8;
        if work_units > budget.max_micros.saturating_mul(SCALE) {
            return Err(ComputeError::BudgetExceeded {
                stage: "world_model/step",
                elapsed_micros: work_units / SCALE,
                limit_micros: budget.max_micros,
            });
        }
        Ok(())
    }

    fn init_state(&mut self, input: &WorldModelInput) {
        for (idx, slot) in self.state.iter_mut().enumerate() {
            let b = input.context_digest[idx] as f32 / 255.0;
            let mix = ((input.seed.rotate_left((idx % 31) as u32) ^ idx as u64) & 0xFFFF) as f32;
            *slot = (2.0 * b - 1.0 + (mix / 65535.0 - 0.5) * 0.02).clamp(-1.0, 1.0);
        }
        self.initialized = true;
    }

    fn digest_vector(values: &[f32], seed: u64, t: u64, fixture_digest: [u8; 32]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(seed.to_le_bytes());
        hasher.update(t.to_le_bytes());
        hasher.update(fixture_digest);
        for v in values {
            hasher.update(quantize_signed_unit(*v).to_le_bytes());
        }
        hasher.finalize().into()
    }

    pub fn fixture_digest(&self) -> WorldFixtureDigest {
        self.fixture.digest()
    }
}

#[cfg(test)]
impl MockJepaPredictor {
    pub fn reset_for_tests(&mut self) {
        self.state = [0.0; WORLD_MODEL_FEATURE_DIM];
        self.last_state_digest = None;
        self.initialized = false;
    }
}

pub(crate) fn obs_features_from_context(
    context_digest: [u8; 32],
) -> [f32; WORLD_MODEL_FEATURE_DIM] {
    let mut obs = [0.0_f32; WORLD_MODEL_FEATURE_DIM];
    for (i, slot) in obs.iter_mut().enumerate() {
        let a = context_digest[i] as f32 / 255.0;
        let b = context_digest[i + WORLD_MODEL_FEATURE_DIM] as f32 / 255.0;
        *slot = ((0.65 * a + 0.35 * b) * 2.0 - 1.0).clamp(-1.0, 1.0);
    }
    obs
}

pub fn state_norm_01(state: &[f32; WORLD_MODEL_FEATURE_DIM]) -> f32 {
    (state.iter().map(|v| v.abs()).sum::<f32>() / WORLD_MODEL_FEATURE_DIM as f32).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use crate::capabilities::WorldModelPredictor;

    use super::*;

    fn input() -> WorldModelInput {
        WorldModelInput {
            t: 12,
            context_digest: [0xAB; 32],
            previous_state_digest: None,
            obs_features: obs_features_from_context([0xAB; 32]),
            seed: 123,
        }
    }

    #[test]
    fn determinism_holds_for_same_sequence() {
        let mut a = MockJepaPredictor::default();
        let mut b = MockJepaPredictor::default();
        let inp = input();

        let out_a = a.step(&inp, ComputeBudget::default()).expect("step");
        let out_b = b.step(&inp, ComputeBudget::default()).expect("step");
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn bounded_values_and_quality() {
        let mut predictor = MockJepaPredictor::default();
        let out = predictor
            .step(&input(), ComputeBudget::default())
            .expect("step")
            .bounded();

        assert!((0.0..=1.0).contains(&out.prediction_error));
        assert!((0.0..=1.0).contains(&out.surprise));
        assert!((0.0..=1.0).contains(&out.state_norm));
        assert_eq!(out.quality, StageQuality::Ok);
        assert!(out.notes.len() <= WORLD_MODEL_MAX_NOTES);
    }

    #[test]
    fn budget_exceeded_is_returned() {
        let mut predictor = MockJepaPredictor::default();
        let err = predictor
            .step(
                &input(),
                ComputeBudget {
                    max_micros: 1,
                    hard_timeout_micros: 1,
                    ..ComputeBudget::default()
                },
            )
            .expect_err("budget should exceed");

        match err {
            ComputeError::BudgetExceeded { stage, .. } => assert_eq!(stage, "world_model/step"),
            other => panic!("unexpected error: {other:?}"),
        }
    }
}

pub const WORLD_VLJEPA_ENCODING_DIM: usize = 64;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WorldInputEncodingV1 {
    pub context_digest: [u8; 32],
    pub risk: f32,
    pub pressure: f32,
    pub surprise: f32,
    pub uncertainty: f32,
    pub confidence: f32,
    pub coherence: f32,
    pub sae_spikes_digest_prefix: Option<[u8; 8]>,
    pub ssm_state_digest_prefix: Option<[u8; 8]>,
    pub lfm_state_digest_prefix: Option<[u8; 8]>,
    pub token_summary_digest_prefix: Option<[u8; 8]>,
}

impl WorldInputEncodingV1 {
    pub fn encode_vector(&self) -> [f32; WORLD_VLJEPA_ENCODING_DIM] {
        let mut out = [0.0_f32; WORLD_VLJEPA_ENCODING_DIM];
        out[0] = (self.risk * 2.0 - 1.0).clamp(-1.0, 1.0);
        out[1] = (self.pressure * 2.0 - 1.0).clamp(-1.0, 1.0);
        out[2] = (self.surprise * 2.0 - 1.0).clamp(-1.0, 1.0);
        out[3] = (self.uncertainty * 2.0 - 1.0).clamp(-1.0, 1.0);
        out[4] = (self.confidence * 2.0 - 1.0).clamp(-1.0, 1.0);
        out[5] = (self.coherence * 2.0 - 1.0).clamp(-1.0, 1.0);

        let mut bytes = Vec::with_capacity(64);
        bytes.extend_from_slice(&self.context_digest);
        for p in [
            self.sae_spikes_digest_prefix,
            self.ssm_state_digest_prefix,
            self.lfm_state_digest_prefix,
            self.token_summary_digest_prefix,
        ]
        .into_iter()
        .flatten()
        {
            bytes.extend_from_slice(&p);
        }
        for i in 0..(WORLD_VLJEPA_ENCODING_DIM - 6) {
            let b = bytes.get(i % bytes.len()).copied().unwrap_or(0) as f32;
            out[6 + i] = (b / 127.5 - 1.0).clamp(-1.0, 1.0);
        }
        out
    }

    pub fn encoding_digest(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(self.context_digest);
        for v in self.encode_vector() {
            h.update(crate::evidence::quantize_signed_unit(v).to_le_bytes());
        }
        h.finalize().into()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WorldVljepaShadowRecord {
    pub t: u64,
    pub encoding_digest_prefix: [u8; 8],
    pub prediction_error_q: u16,
    pub prediction_digest_prefix: [u8; 8],
    pub model_hash_prefix: [u8; 8],
    pub saturation_clamp_count: u16,
    pub invalid_output: bool,
    pub status: &'static str,
}

pub fn world_vljepa_shadow_step(
    t: u64,
    encoding: &WorldInputEncodingV1,
    model_hash: [u8; 32],
) -> WorldVljepaShadowRecord {
    let x = encoding.encode_vector();
    let mut y = [0.0_f32; WORLD_VLJEPA_ENCODING_DIM];
    let mut saturation_clamp_count = 0_u16;
    for (idx, yi) in y.iter_mut().enumerate() {
        let a = model_hash[idx % 32] as f32 / 255.0;
        let raw = x[idx] * (0.6 + a * 0.4);
        let clamped = raw.clamp(-1.0, 1.0);
        if (raw - clamped).abs() > f32::EPSILON {
            saturation_clamp_count = saturation_clamp_count.saturating_add(1);
        }
        *yi = clamped;
    }
    let err = y
        .iter()
        .zip(x.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / WORLD_VLJEPA_ENCODING_DIM as f32;
    let invalid_output = !err.is_finite() || y.iter().any(|v| !v.is_finite());
    let err_q = ucf_types::UQ0_16::from_f32_clamped(if invalid_output { 1.0 } else { err }).raw();

    let mut hasher = Sha256::new();
    for v in y {
        hasher.update(crate::evidence::quantize_signed_unit(v).to_le_bytes());
    }
    hasher.update(model_hash);
    hasher.update(t.to_le_bytes());
    hasher.update(encoding.encoding_digest());
    let prediction_digest: [u8; 32] = hasher.finalize().into();

    let mut enc_pref = [0_u8; 8];
    enc_pref.copy_from_slice(&encoding.encoding_digest()[..8]);
    let mut pred_pref = [0_u8; 8];
    pred_pref.copy_from_slice(&prediction_digest[..8]);
    let mut model_pref = [0_u8; 8];
    model_pref.copy_from_slice(&model_hash[..8]);

    WorldVljepaShadowRecord {
        t,
        encoding_digest_prefix: enc_pref,
        prediction_error_q: err_q,
        prediction_digest_prefix: pred_pref,
        model_hash_prefix: model_pref,
        saturation_clamp_count,
        invalid_output,
        status: if invalid_output { "invalid" } else { "ok" },
    }
}

#[cfg(test)]
mod vljepa_tests {
    use super::*;

    #[test]
    fn vljepa_encoding_is_deterministic_and_bounded() {
        let enc = WorldInputEncodingV1 {
            context_digest: [0xAA; 32],
            risk: 0.2,
            pressure: 0.4,
            surprise: 0.1,
            uncertainty: 0.3,
            confidence: 0.9,
            coherence: 0.8,
            sae_spikes_digest_prefix: Some([1; 8]),
            ssm_state_digest_prefix: Some([2; 8]),
            lfm_state_digest_prefix: None,
            token_summary_digest_prefix: Some([3; 8]),
        };
        let a = enc.encode_vector();
        let b = enc.encode_vector();
        assert_eq!(a, b);
        assert!(a.iter().all(|v| (-1.0..=1.0).contains(v)));
        assert_eq!(enc.encoding_digest(), enc.encoding_digest());
    }

    #[test]
    fn vljepa_shadow_step_is_deterministic() {
        let enc = WorldInputEncodingV1 {
            context_digest: [0xBB; 32],
            risk: 0.6,
            pressure: 0.7,
            surprise: 0.2,
            uncertainty: 0.2,
            confidence: 0.8,
            coherence: 0.6,
            sae_spikes_digest_prefix: None,
            ssm_state_digest_prefix: None,
            lfm_state_digest_prefix: None,
            token_summary_digest_prefix: None,
        };
        let a = world_vljepa_shadow_step(99, &enc, [0x11; 32]);
        let b = world_vljepa_shadow_step(99, &enc, [0x11; 32]);
        assert_eq!(a, b);
    }
}
