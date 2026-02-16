use sha2::{Digest, Sha256};

use crate::evidence::quantize_signed_unit;
use crate::risk_contract::SignalQuality;
use crate::{ComputeBudget, ComputeError};

use crate::capabilities::WorldModelPredictor;

impl WorldModelPredictor for MockJepaPredictor {
    fn name(&self) -> &'static str {
        "mock_jepa_v0"
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
