use sha2::{Digest, Sha256};

use crate::capabilities::SaeExtractor;
use crate::evidence::quantize_unit_u16;
use crate::world_model::{StageQuality, WorldModelOutput};
use crate::{ComputeBudget, ComputeError, ComputeInput, Spike};

pub const SAE_INPUT_DIM: usize = 32;
pub const SAE_FEATURE_DIM: usize = 128;
pub const SAE_TOP_K: usize = 32;
pub const SAE_MAX_SPIKES: usize = SAE_TOP_K;
const SAE_NOTES_MAX: usize = 4;
const SAE_WORK_SCALE: u64 = 8;

const SAE_FIXTURE_JSON: &str = include_str!("../fixtures/sae_proj_v1.json");
const SAE_FIXTURE_DIGEST: [u8; 32] = [
    0xac, 0xb4, 0xd2, 0x20, 0x96, 0x73, 0xa6, 0x67, 0x2f, 0x6c, 0x49, 0xac, 0xf7, 0x6f, 0xc6, 0x5f,
    0x00, 0x1a, 0x3b, 0x12, 0x6c, 0xe5, 0xc8, 0x68, 0xd4, 0xce, 0x94, 0xcd, 0x20, 0xcd, 0x78, 0x48,
];

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SmallNotes(pub Vec<String>);

impl SmallNotes {
    pub fn bounded(mut self) -> Self {
        if self.0.len() > SAE_NOTES_MAX {
            self.0.truncate(SAE_NOTES_MAX);
        }
        self
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SaeInput {
    pub t: u64,
    pub context_features: [f32; SAE_INPUT_DIM],
    pub world_state_digest: Option<[u8; 32]>,
    pub seed: u64,
    pub evidence_chain_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SaeOutput {
    pub spikes: Vec<Spike>,
    pub spike_count: u16,
    pub sparsity: f32,
    pub energy: f32,
    pub spikes_digest: [u8; 32],
    pub quality: StageQuality,
    pub notes: SmallNotes,
}

impl SaeOutput {
    pub fn bounded(mut self) -> Self {
        self.spike_count = self.spike_count.min(SAE_MAX_SPIKES as u16);
        if self.spikes.len() > SAE_MAX_SPIKES {
            self.spikes.truncate(SAE_MAX_SPIKES);
        }
        for spike in &mut self.spikes {
            spike.magnitude = spike.magnitude.clamp(0.0, 1.0);
        }
        self.sparsity = self.sparsity.clamp(0.0, 1.0);
        self.energy = self.energy.clamp(0.0, 1.0);
        self.notes = self.notes.bounded();
        self
    }
}

#[derive(Debug, Clone)]
struct SaeFixture {
    energy_scale: f32,
    weights: Vec<f32>,
    bias: Vec<f32>,
    digest: [u8; 32],
}

impl SaeFixture {
    fn parse_json(raw: &str) -> Result<Self, ComputeError> {
        #[derive(serde::Deserialize)]
        struct Formula {
            modulus: u32,
            mul_i: u32,
            #[serde(default)]
            mul_j: u32,
            add: u32,
            scale: f32,
            bias: f32,
        }

        #[derive(serde::Deserialize)]
        struct SaeFixtureJson {
            schema_version: u16,
            f: usize,
            d: usize,
            energy_scale: f32,
            weight_formula: Formula,
            bias_formula: Formula,
            weights_digest_hex: String,
        }

        let parsed: SaeFixtureJson =
            serde_json::from_str(raw).map_err(|err| ComputeError::InvalidInput {
                reason: format!("invalid SAE fixture json: {err}"),
            })?;

        if parsed.schema_version != 1 || parsed.f != SAE_FEATURE_DIM || parsed.d != SAE_INPUT_DIM {
            return Err(ComputeError::InvalidInput {
                reason: format!(
                    "unsupported SAE fixture schema={} f={} d={}",
                    parsed.schema_version, parsed.f, parsed.d
                ),
            });
        }

        let mut weights = Vec::with_capacity(SAE_FEATURE_DIM * SAE_INPUT_DIM);
        for i in 0..SAE_FEATURE_DIM {
            for j in 0..SAE_INPUT_DIM {
                let formula = &parsed.weight_formula;
                let numerator = ((i as u32)
                    .saturating_mul(formula.mul_i)
                    .saturating_add((j as u32).saturating_mul(formula.mul_j))
                    .saturating_add(formula.add))
                    % formula.modulus.max(1);
                let normalized = (numerator as f64) / (formula.modulus.max(1) as f64);
                let value =
                    (normalized * f64::from(formula.scale) + f64::from(formula.bias)) as f32;
                weights.push(value);
            }
        }

        let mut bias = Vec::with_capacity(SAE_FEATURE_DIM);
        for i in 0..SAE_FEATURE_DIM {
            let formula = &parsed.bias_formula;
            let numerator = ((i as u32)
                .saturating_mul(formula.mul_i)
                .saturating_add(formula.add))
                % formula.modulus.max(1);
            let normalized = (numerator as f64) / (formula.modulus.max(1) as f64);
            let value = (normalized * f64::from(formula.scale) + f64::from(formula.bias)) as f32;
            bias.push(value);
        }

        let mut canonical = Vec::with_capacity((weights.len() + bias.len()) * 4 + 10);
        canonical.extend_from_slice(&parsed.schema_version.to_le_bytes());
        canonical.extend_from_slice(&(parsed.f as u16).to_le_bytes());
        canonical.extend_from_slice(&(parsed.d as u16).to_le_bytes());
        canonical.extend_from_slice(&parsed.energy_scale.to_bits().to_le_bytes());
        for value in weights.iter().chain(bias.iter()) {
            canonical.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        let expected: [u8; 32] = Sha256::digest(&canonical).into();

        let digest_bytes =
            hex::decode(parsed.weights_digest_hex).map_err(|err| ComputeError::InvalidInput {
                reason: format!("invalid SAE fixture digest hex: {err}"),
            })?;
        if digest_bytes.len() != 32 {
            return Err(ComputeError::InvalidInput {
                reason: "invalid SAE fixture digest length".to_string(),
            });
        }
        let mut digest = [0_u8; 32];
        digest.copy_from_slice(&digest_bytes);
        if digest != expected || digest != SAE_FIXTURE_DIGEST {
            return Err(ComputeError::InvalidInput {
                reason: "invalid SAE fixture digest".to_string(),
            });
        }

        Ok(Self {
            energy_scale: parsed.energy_scale,
            weights,
            bias,
            digest,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ToySaeExtractor {
    fixture: SaeFixture,
}

impl Default for ToySaeExtractor {
    fn default() -> Self {
        let fixture =
            SaeFixture::parse_json(SAE_FIXTURE_JSON).expect("embedded SAE fixture must be valid");
        Self { fixture }
    }
}

impl ToySaeExtractor {
    fn check_budget(work_units: u64, budget: ComputeBudget) -> Result<(), ComputeError> {
        let elapsed_micros = work_units / SAE_WORK_SCALE;
        if work_units > budget.max_micros.saturating_mul(SAE_WORK_SCALE) {
            return Err(ComputeError::BudgetExceeded {
                stage: "sae/extract",
                elapsed_micros,
                limit_micros: budget.max_micros,
            });
        }
        Ok(())
    }

    pub fn fixture_digest(&self) -> [u8; 32] {
        self.fixture.digest
    }

    fn context_digest_prefix(input: &SaeInput) -> [u8; 8] {
        let mut prefix = [0_u8; 8];
        prefix.copy_from_slice(&input.evidence_chain_digest[..8]);
        prefix
    }

    fn empty_degraded(input: &SaeInput, reason: &'static str) -> SaeOutput {
        let mut hasher = Sha256::new();
        hasher.update(b"sae:empty");
        hasher.update(reason.as_bytes());
        hasher.update(input.t.to_le_bytes());
        hasher.update(input.seed.to_le_bytes());
        hasher.update(input.evidence_chain_digest);
        let spikes_digest: [u8; 32] = hasher.finalize().into();
        SaeOutput {
            spikes: Vec::new(),
            spike_count: 0,
            sparsity: 1.0,
            energy: 0.0,
            spikes_digest,
            quality: StageQuality::DegradedFallback,
            notes: SmallNotes(vec![format!("degraded:{reason}")]),
        }
    }

    fn normalize_context_features(
        input: &ComputeInput,
        world: &WorldModelOutput,
    ) -> [f32; SAE_INPUT_DIM] {
        let mut out = [0.0_f32; SAE_INPUT_DIM];
        for (idx, value) in out.iter_mut().enumerate() {
            let c = input.context_digest[idx % 32] as f32 / 255.0;
            let w = world.prediction_digest[idx % 32] as f32 / 255.0;
            *value = (0.7 * c + 0.3 * w).clamp(0.0, 1.0);
        }
        out
    }

    pub fn make_input(
        input: &ComputeInput,
        world: &WorldModelOutput,
        seed: u64,
        evidence_chain_digest: [u8; 32],
    ) -> SaeInput {
        SaeInput {
            t: input.t,
            context_features: Self::normalize_context_features(input, world),
            world_state_digest: Some(world.state_digest),
            seed,
            evidence_chain_digest,
        }
    }
}

impl SaeExtractor for ToySaeExtractor {
    fn name(&self) -> &'static str {
        "toy_sae_v0"
    }

    fn extract(&self, input: &SaeInput, budget: ComputeBudget) -> Result<SaeOutput, ComputeError> {
        let mut work_units = 16_u64;
        if let Err(err) = Self::check_budget(work_units, budget) {
            return match budget.degrade_policy {
                crate::DegradePolicy::DegradeStages => {
                    Ok(Self::empty_degraded(input, "budget_exceeded"))
                }
                crate::DegradePolicy::FailFast => Err(err),
            };
        }

        let mut activations = vec![0.0_f32; SAE_FEATURE_DIM];
        for (feature_idx, slot) in activations.iter_mut().enumerate() {
            let row_offset = feature_idx * SAE_INPUT_DIM;
            let mut acc = self.fixture.bias[feature_idx];
            for j in 0..SAE_INPUT_DIM {
                work_units = work_units.saturating_add(1);
                if let Err(err) = Self::check_budget(work_units, budget) {
                    return match budget.degrade_policy {
                        crate::DegradePolicy::DegradeStages => {
                            Ok(Self::empty_degraded(input, "budget_exceeded"))
                        }
                        crate::DegradePolicy::FailFast => Err(err),
                    };
                }
                acc += self.fixture.weights[row_offset + j] * input.context_features[j];
            }
            *slot = acc.max(0.0);
        }

        let max_activation = activations
            .iter()
            .copied()
            .fold(0.0_f32, f32::max)
            .max(1e-9);
        let mut rank: Vec<(usize, f32)> = activations.iter().copied().enumerate().collect();
        rank.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let top_k = SAE_TOP_K.min(rank.len());
        let mut spikes = Vec::with_capacity(top_k);
        for (feature_idx, activation) in rank.into_iter().take(top_k) {
            if activation <= 0.0 {
                break;
            }
            spikes.push(Spike {
                feature_id: feature_idx as u32,
                magnitude: (activation / max_activation).clamp(0.0, 1.0),
                timestamp: input.t,
            });
        }
        spikes.sort_by(|a, b| a.feature_id.cmp(&b.feature_id));

        let spike_count = spikes.len() as u16;
        let sparsity = (1.0 - (spike_count as f32 / SAE_FEATURE_DIM as f32)).clamp(0.0, 1.0);
        let mean_activation = activations.iter().sum::<f32>() / SAE_FEATURE_DIM as f32;
        let energy = (mean_activation / self.fixture.energy_scale.max(1e-6)).clamp(0.0, 1.0);

        let mut digest_hasher = Sha256::new();
        for spike in &spikes {
            digest_hasher.update(spike.feature_id.to_le_bytes());
            digest_hasher.update(quantize_unit_u16(spike.magnitude).to_le_bytes());
            digest_hasher.update(spike.timestamp.to_le_bytes());
        }
        digest_hasher.update(self.fixture.digest);
        digest_hasher.update(input.t.to_le_bytes());
        digest_hasher.update(input.seed.to_le_bytes());
        digest_hasher.update(Self::context_digest_prefix(input));
        let spikes_digest: [u8; 32] = digest_hasher.finalize().into();

        Ok(SaeOutput {
            spikes,
            spike_count,
            sparsity,
            energy,
            spikes_digest,
            quality: StageQuality::Ok,
            notes: SmallNotes(vec![format!(
                "fixture={}",
                hex::encode(&self.fixture.digest[..6])
            )]),
        }
        .bounded())
    }
}

#[cfg(test)]
mod tests {
    use crate::FrameId;

    use super::*;

    fn sample_input() -> SaeInput {
        SaeInput {
            t: 23,
            context_features: [0.3; SAE_INPUT_DIM],
            world_state_digest: Some([2; 32]),
            seed: 77,
            evidence_chain_digest: [5; 32],
        }
    }

    #[test]
    fn deterministic_top_k_selection_with_ties() {
        let sae = ToySaeExtractor::default();
        let input = sample_input();
        let out_a = sae.extract(&input, ComputeBudget::default()).expect("sae");
        let out_b = sae.extract(&input, ComputeBudget::default()).expect("sae");
        assert_eq!(out_a, out_b);
        assert!(out_a
            .spikes
            .windows(2)
            .all(|w| w[0].feature_id <= w[1].feature_id));
    }

    #[test]
    fn digest_stability_for_same_input() {
        let sae = ToySaeExtractor::default();
        let input = sample_input();
        let a = sae.extract(&input, ComputeBudget::default()).expect("sae");
        let b = sae.extract(&input, ComputeBudget::default()).expect("sae");
        assert_eq!(a.spikes_digest, b.spikes_digest);
    }

    #[test]
    fn output_bounds_hold() {
        let sae = ToySaeExtractor::default();
        let out = sae
            .extract(&sample_input(), ComputeBudget::default())
            .expect("sae");
        assert!(usize::from(out.spike_count) <= SAE_TOP_K);
        assert!(out.spikes.len() <= SAE_MAX_SPIKES);
        assert!(out
            .spikes
            .iter()
            .all(|s| (0.0..=1.0).contains(&s.magnitude)));
        assert!((0.0..=1.0).contains(&out.sparsity));
        assert!((0.0..=1.0).contains(&out.energy));
    }

    #[test]
    fn budget_exceeded_degrades_or_failfast() {
        let sae = ToySaeExtractor::default();
        let input = sample_input();
        let degrade = sae
            .extract(
                &input,
                ComputeBudget {
                    max_micros: 1,
                    ..ComputeBudget::default()
                },
            )
            .expect("degraded output");
        assert_eq!(degrade.quality, StageQuality::DegradedFallback);
        assert_eq!(degrade.spike_count, 0);

        let err = sae.extract(
            &input,
            ComputeBudget {
                max_micros: 1,
                degrade_policy: crate::DegradePolicy::FailFast,
                ..ComputeBudget::default()
            },
        );
        assert!(matches!(err, Err(ComputeError::BudgetExceeded { .. })));
    }

    #[test]
    fn make_input_is_stable() {
        let compute_input = ComputeInput {
            frame_id: FrameId(4),
            t: 9,
            context_digest: [7; 32],
        };
        let world = WorldModelOutput {
            prediction_digest: [1; 32],
            state_digest: [2; 32],
            prediction_error: 0.1,
            surprise: 0.1,
            state_norm: 0.1,
            quality: StageQuality::Ok,
            notes: vec![],
        };
        let a = ToySaeExtractor::make_input(&compute_input, &world, 11, [9; 32]);
        let b = ToySaeExtractor::make_input(&compute_input, &world, 11, [9; 32]);
        assert_eq!(a, b);
    }
}
