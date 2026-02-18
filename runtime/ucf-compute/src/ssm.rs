use sha2::{Digest, Sha256};

use crate::contracts::StageContractVersion;
use crate::evidence::{quantize_signed_unit, quantize_unit_u16};
use crate::feature_extractor::SmallNotes;
use crate::world_model::StageQuality;
use crate::{ComputeBudget, ComputeError};

pub const SSM_STATE_DIM: usize = 32;
const SSM_SCHEMA_V1: u16 = 1;
const SSM_WORK_SCALE: u64 = 8;
const SSM_FIXTURE_JSON: &str = include_str!("../fixtures/ssm_toy_v1.json");
const SSM_FIXTURE_DIGEST: [u8; 32] = [
    0x9b, 0xf9, 0x11, 0xca, 0x6b, 0x80, 0x8d, 0xee, 0x56, 0x9e, 0x84, 0xa0, 0x84, 0xab, 0x6f, 0x8d,
    0xcb, 0x15, 0x90, 0xd7, 0x6a, 0x82, 0x13, 0xca, 0xe0, 0x66, 0x44, 0xe0, 0x81, 0x39, 0xb0, 0x97,
];
const DEGRADED_MARKER: [u8; 32] = [0xDD; 32];

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SsmInput {
    pub t: u64,
    pub spikes_digest: [u8; 32],
    pub spike_count: u16,
    pub sae_energy: f32,
    pub world_surprise: f32,
    pub risk: f32,
    pub seed: u64,
    pub context_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SsmOutput {
    pub pressure: f32,
    pub state_digest: [u8; 32],
    pub readout_digest: [u8; 32],
    pub state_norm: f32,
    pub readout: f32,
    pub quality: StageQuality,
    pub notes: SmallNotes,
}

impl SsmOutput {
    pub fn degraded(reason: &'static str) -> Self {
        Self {
            pressure: 1.0,
            state_digest: DEGRADED_MARKER,
            readout_digest: DEGRADED_MARKER,
            state_norm: 1.0,
            readout: 1.0,
            quality: StageQuality::DegradedFallback,
            notes: SmallNotes(vec![format!("degraded:{reason}")]),
        }
    }
}

pub trait SsmKernel: Send + Sync {
    fn name(&self) -> &'static str;
    fn contract_version(&self) -> StageContractVersion {
        StageContractVersion::V1
    }
    fn step(&mut self, input: &SsmInput, budget: ComputeBudget) -> Result<SsmOutput, ComputeError>;
}

#[derive(Debug, Clone, Copy)]
struct SsmFixture {
    kmax: f32,
    w1: f32,
    w2: f32,
    w3: f32,
    state_scale: f32,
    a: [f32; SSM_STATE_DIM],
    b: [f32; SSM_STATE_DIM],
    c: [f32; SSM_STATE_DIM],
    digest: [u8; 32],
}

impl SsmFixture {
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
        struct SsmFixtureJson {
            schema_version: u16,
            n: usize,
            kmax: u16,
            w1: f32,
            w2: f32,
            w3: f32,
            state_scale: f32,
            decay_formula: Formula,
            gain_formula: Formula,
            readout_formula: Formula,
            fixture_digest_hex: String,
        }

        let parsed: SsmFixtureJson =
            serde_json::from_str(raw).map_err(|err| ComputeError::InvalidInput {
                reason: format!("invalid SSM fixture json: {err}"),
            })?;

        if parsed.schema_version != SSM_SCHEMA_V1 || parsed.n != SSM_STATE_DIM {
            return Err(ComputeError::InvalidInput {
                reason: format!(
                    "unsupported SSM fixture schema={} n={}",
                    parsed.schema_version, parsed.n
                ),
            });
        }

        fn gen(formula: &Formula) -> [f32; SSM_STATE_DIM] {
            let mut out = [0.0_f32; SSM_STATE_DIM];
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

        let a = gen(&parsed.decay_formula);
        let b = gen(&parsed.gain_formula);
        let c = gen(&parsed.readout_formula);

        let mut canonical = Vec::with_capacity(2 * 3 + 4 * 4 + 4 * SSM_STATE_DIM * 3);
        canonical.extend_from_slice(&parsed.schema_version.to_le_bytes());
        canonical.extend_from_slice(&(parsed.n as u16).to_le_bytes());
        canonical.extend_from_slice(&parsed.kmax.to_le_bytes());
        for value in [parsed.w1, parsed.w2, parsed.w3, parsed.state_scale] {
            canonical.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        for value in a.into_iter().chain(b).chain(c) {
            canonical.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        let expected: [u8; 32] = Sha256::digest(&canonical).into();

        let decoded =
            hex::decode(parsed.fixture_digest_hex).map_err(|err| ComputeError::InvalidInput {
                reason: format!("invalid SSM fixture digest hex: {err}"),
            })?;
        if decoded.len() != 32 {
            return Err(ComputeError::InvalidInput {
                reason: "invalid SSM fixture digest length".to_string(),
            });
        }

        let mut digest = [0_u8; 32];
        digest.copy_from_slice(&decoded);
        if digest != expected || digest != SSM_FIXTURE_DIGEST {
            return Err(ComputeError::InvalidInput {
                reason: "invalid SSM fixture digest".to_string(),
            });
        }

        Ok(Self {
            kmax: f32::from(parsed.kmax.max(1)),
            w1: parsed.w1,
            w2: parsed.w2,
            w3: parsed.w3,
            state_scale: parsed.state_scale.max(0.1),
            a,
            b,
            c,
            digest,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ToySsmKernel {
    x: [f32; SSM_STATE_DIM],
    fixture: SsmFixture,
}

impl Default for ToySsmKernel {
    fn default() -> Self {
        Self {
            x: [0.0; SSM_STATE_DIM],
            fixture: SsmFixture::parse_json(SSM_FIXTURE_JSON)
                .expect("embedded SSM fixture must be valid"),
        }
    }
}

impl ToySsmKernel {
    fn check_budget(work_units: u64, budget: ComputeBudget) -> Result<(), ComputeError> {
        let elapsed_micros = work_units / SSM_WORK_SCALE;
        if work_units > budget.max_micros.saturating_mul(SSM_WORK_SCALE) {
            return Err(ComputeError::BudgetExceeded {
                stage: "ssm/step",
                elapsed_micros,
                limit_micros: budget.max_micros,
            });
        }
        Ok(())
    }

    pub(crate) fn select_indices(spikes_digest: [u8; 32]) -> Vec<usize> {
        let max_m = SSM_STATE_DIM / 4;
        let target = 1 + (usize::from(spikes_digest[0]) % max_m.max(1));
        let mut mask = [false; SSM_STATE_DIM];
        let mut selected = 0usize;
        for offset in 0..(spikes_digest.len() * 8) {
            let byte = spikes_digest[offset % spikes_digest.len()];
            let idx = (usize::from(byte) + offset) % SSM_STATE_DIM;
            if !mask[idx] {
                mask[idx] = true;
                selected += 1;
                if selected >= target {
                    break;
                }
            }
        }

        let mut indices = Vec::with_capacity(selected);
        for (idx, is_selected) in mask.into_iter().enumerate() {
            if is_selected {
                indices.push(idx);
            }
        }
        indices
    }

    fn state_digest(&self, t: u64, context_digest: [u8; 32], seed: u64) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.fixture.digest);
        hasher.update(t.to_le_bytes());
        hasher.update(seed.to_le_bytes());
        hasher.update(context_digest);
        for value in self.x {
            hasher.update(quantize_signed_unit(value).to_le_bytes());
        }
        hasher.finalize().into()
    }

    fn readout_digest(&self, readout: f32, state_digest: [u8; 32]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(quantize_unit_u16(readout).to_le_bytes());
        hasher.update(state_digest);
        hasher.update(self.fixture.digest);
        hasher.finalize().into()
    }
}

impl SsmKernel for ToySsmKernel {
    fn name(&self) -> &'static str {
        "toy_ssm_selective_scan_v0"
    }

    fn step(&mut self, input: &SsmInput, budget: ComputeBudget) -> Result<SsmOutput, ComputeError> {
        let mut work_units = 16_u64;
        Self::check_budget(work_units, budget)?;

        let u_spikes = (f32::from(input.spike_count) / self.fixture.kmax).clamp(0.0, 1.0);
        let u = (self.fixture.w1 * u_spikes
            + self.fixture.w2 * input.sae_energy.clamp(0.0, 1.0)
            + self.fixture.w3 * input.world_surprise.clamp(0.0, 1.0))
        .clamp(0.0, 1.0);

        let selected = Self::select_indices(input.spikes_digest);
        let mut selected_mask = [false; SSM_STATE_DIM];
        for idx in selected {
            selected_mask[idx] = true;
        }

        for (idx, value) in self.x.iter_mut().enumerate() {
            work_units = work_units.saturating_add(4);
            Self::check_budget(work_units, budget)?;
            if selected_mask[idx] {
                *value = (self.fixture.a[idx] * *value + self.fixture.b[idx] * u).clamp(-1.0, 1.0);
            } else {
                *value = (0.98 * *value).clamp(-1.0, 1.0);
            }
        }

        let mut readout = 0.0_f32;
        for idx in 0..SSM_STATE_DIM {
            work_units = work_units.saturating_add(2);
            Self::check_budget(work_units, budget)?;
            readout += self.fixture.c[idx] * self.x[idx];
        }
        let readout = ((readout + 1.0) * 0.5).clamp(0.0, 1.0);

        let mean_abs = self.x.iter().map(|v| v.abs()).sum::<f32>() / SSM_STATE_DIM as f32;
        let state_norm = (mean_abs / self.fixture.state_scale).clamp(0.0, 1.0);
        let pressure = (0.5 * u + 0.5 * state_norm).clamp(0.0, 1.0);

        let state_digest = self.state_digest(input.t, input.context_digest, input.seed);
        let readout_digest = self.readout_digest(readout, state_digest);

        Ok(SsmOutput {
            pressure,
            state_digest,
            readout_digest,
            state_norm,
            readout,
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

    fn input() -> SsmInput {
        SsmInput {
            t: 4,
            spikes_digest: [0x23; 32],
            spike_count: 6,
            sae_energy: 0.25,
            world_surprise: 0.15,
            risk: 0.2,
            seed: 7,
            context_digest: [0x55; 32],
        }
    }

    #[test]
    fn deterministic_for_same_sequence() {
        let mut a = ToySsmKernel::default();
        let mut b = ToySsmKernel::default();
        let first_a = a.step(&input(), ComputeBudget::default()).expect("step");
        let first_b = b.step(&input(), ComputeBudget::default()).expect("step");
        assert_eq!(first_a, first_b);

        let mut next = input();
        next.t = 5;
        next.spike_count = 16;
        let second_a = a.step(&next, ComputeBudget::default()).expect("step");
        let second_b = b.step(&next, ComputeBudget::default()).expect("step");
        assert_eq!(second_a, second_b);
    }

    #[test]
    fn higher_spike_count_increases_pressure() {
        let mut low_kernel = ToySsmKernel::default();
        let mut high_kernel = ToySsmKernel::default();

        let mut low = input();
        low.spike_count = 2;
        let mut high = low;
        high.spike_count = 32;

        let low_out = low_kernel
            .step(&low, ComputeBudget::default())
            .expect("low");
        let high_out = high_kernel
            .step(&high, ComputeBudget::default())
            .expect("high");
        assert!(high_out.pressure >= low_out.pressure);
    }

    #[test]
    fn selective_indices_are_deterministic() {
        let idx_a = ToySsmKernel::select_indices([0xAB; 32]);
        let idx_b = ToySsmKernel::select_indices([0xAB; 32]);
        assert_eq!(idx_a, idx_b);
        assert!(idx_a.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn budget_exceeded_reports_stage() {
        let mut kernel = ToySsmKernel::default();
        let err = kernel
            .step(
                &input(),
                ComputeBudget {
                    max_micros: 1,
                    ..ComputeBudget::default()
                },
            )
            .expect_err("budget");
        match err {
            ComputeError::BudgetExceeded { stage, .. } => assert_eq!(stage, "ssm/step"),
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
