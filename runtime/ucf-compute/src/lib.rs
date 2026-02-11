use sha2::{Digest, Sha256};
use thiserror::Error;
use ucf_core::types::SimTime;
use ucf_frames::v1::{ControlFrame, ControlPayload};

pub const MAX_SPIKES: usize = 256;
pub const MAX_NOTES: usize = 16;
pub const MAX_NOTE_LEN: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FrameId(pub u64);

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ComputeInput {
    pub frame_id: FrameId,
    pub t: u64,
    pub context_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Spike {
    pub feature_id: u32,
    pub magnitude: f32,
    pub timestamp: u64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ComputeSignals {
    pub surprise: f32,
    pub pressure: f32,
    pub risk: f32,
    pub confidence: f32,
    pub spikes: Vec<Spike>,
    pub notes: Vec<String>,
}

impl ComputeSignals {
    pub fn bounded(mut self) -> Self {
        self.surprise = self.surprise.clamp(0.0, 1.0);
        self.pressure = self.pressure.clamp(0.0, 1.0);
        self.risk = self.risk.clamp(0.0, 1.0);
        self.confidence = self.confidence.clamp(0.0, 1.0);

        if self.spikes.len() > MAX_SPIKES {
            self.spikes.truncate(MAX_SPIKES);
        }
        if self.notes.len() > MAX_NOTES {
            self.notes.truncate(MAX_NOTES);
        }
        self.notes = self
            .notes
            .into_iter()
            .map(|n| n.chars().take(MAX_NOTE_LEN).collect())
            .collect();
        self
    }

    pub fn summary(&self, backend: &'static str) -> ComputeSignalsSummary {
        let mut hasher = Sha256::new();
        for spike in &self.spikes {
            hasher.update(spike.feature_id.to_le_bytes());
            hasher.update(spike.magnitude.to_bits().to_le_bytes());
            hasher.update(spike.timestamp.to_le_bytes());
        }
        let digest = hasher.finalize();
        let mut spikes_digest = [0_u8; 32];
        spikes_digest.copy_from_slice(&digest);
        ComputeSignalsSummary {
            backend,
            surprise: self.surprise,
            pressure: self.pressure,
            risk: self.risk,
            confidence: self.confidence,
            spike_count: self.spikes.len() as u16,
            spikes_digest,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeBudget {
    pub max_micros: u64,
    pub hard_timeout_micros: u64,
    pub seed: u64,
}

impl Default for ComputeBudget {
    fn default() -> Self {
        Self {
            max_micros: 1_000,
            hard_timeout_micros: 5_000,
            seed: 0xDEC0DED,
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ComputeError {
    #[error(
        "compute budget exceeded at {stage}: elapsed {elapsed_micros}µs > limit {limit_micros}µs"
    )]
    BudgetExceeded {
        stage: &'static str,
        elapsed_micros: u64,
        limit_micros: u64,
    },
    #[error("invalid compute input: {reason}")]
    InvalidInput { reason: String },
    #[error("compute backend disabled")]
    BackendDisabled,
    #[error("compute backend internal error: {reason}")]
    Internal { reason: String },
}

pub trait AiComputeBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn compute(
        &self,
        input: &ComputeInput,
        budget: ComputeBudget,
    ) -> Result<ComputeSignals, ComputeError>;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComputeSignalsSummary {
    pub backend: &'static str,
    pub surprise: f32,
    pub pressure: f32,
    pub risk: f32,
    pub confidence: f32,
    pub spike_count: u16,
    pub spikes_digest: [u8; 32],
}

pub fn digest_control_frame(ctrl: &ControlFrame) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(ctrl.time.tick.get().to_le_bytes());
    hasher.update(ctrl.time.window.get().to_le_bytes());
    hasher.update(ctrl.corr.0.to_le_bytes());
    hasher.update([ctrl.channel as u8]);
    hasher.update(ctrl.intent.summary.as_bytes());
    hasher.update(ctrl.intent.id.0.to_le_bytes());
    hasher.update([ctrl.intent.kind as u8]);

    match &ctrl.payload {
        ControlPayload::Text(text) => {
            hasher.update([0]);
            hasher.update(text.as_bytes());
        }
        ControlPayload::Bytes(bytes) => {
            hasher.update([1]);
            hasher.update(bytes.as_ref());
        }
        ControlPayload::BrainStimulus(stimulus) => {
            hasher.update([2]);
            hasher.update([stimulus.kind as u8]);
            hasher.update(stimulus.target.to_le_bytes());
            hasher.update(stimulus.intensity.to_le_bytes());
            hasher.update(stimulus.duration_ms.to_le_bytes());
        }
        ControlPayload::Empty => {
            hasher.update([3]);
        }
    }

    let digest = hasher.finalize();
    let mut out = [0_u8; 32];
    out.copy_from_slice(&digest);
    out
}

pub fn compute_input_from_control(ctrl: &ControlFrame) -> ComputeInput {
    ComputeInput {
        frame_id: FrameId(ctrl.corr.0),
        t: ctrl.time.tick.get(),
        context_digest: digest_control_frame(ctrl),
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CpuStubBackend;

impl CpuStubBackend {
    fn normalized_unit(bits: u64) -> f32 {
        let v = (bits >> 40) as u32;
        (v as f32) / (u32::MAX as f32)
    }

    fn check_budget(
        work_units: u64,
        stage: &'static str,
        budget: ComputeBudget,
    ) -> Result<(), ComputeError> {
        const SCALE: u64 = 8;
        let elapsed_micros = work_units / SCALE;
        if work_units > budget.max_micros.saturating_mul(SCALE) {
            return Err(ComputeError::BudgetExceeded {
                stage,
                elapsed_micros,
                limit_micros: budget.max_micros,
            });
        }
        Ok(())
    }
}

impl AiComputeBackend for CpuStubBackend {
    fn name(&self) -> &'static str {
        "cpu_stub"
    }

    fn compute(
        &self,
        input: &ComputeInput,
        budget: ComputeBudget,
    ) -> Result<ComputeSignals, ComputeError> {
        if input.t == 0 {
            return Err(ComputeError::InvalidInput {
                reason: "t must be non-zero".to_string(),
            });
        }

        let mut seed_bytes = [0_u8; 8];
        seed_bytes.copy_from_slice(&input.context_digest[0..8]);
        let context_seed = u64::from_le_bytes(seed_bytes);
        let mut prng = SplitMix64::new(context_seed ^ budget.seed ^ input.t.rotate_left(7));

        let surprise = Self::normalized_unit(prng.next_u64());
        let pressure_base = Self::normalized_unit(prng.next_u64());
        let periodic = ((input.t % 17) as f32) / 16.0;
        let pressure = (pressure_base * 0.75 + periodic * 0.25).clamp(0.0, 1.0);
        let risk = (0.6 * surprise + 0.4 * pressure).clamp(0.0, 1.0);
        let confidence = (1.0 - 0.8 * risk).clamp(0.0, 1.0);

        let mut work_units = 32_u64;
        Self::check_budget(work_units, "base", budget)?;

        let mut spikes = Vec::new();
        let max_spikes = ((input.context_digest[0] % 64) as usize).min(64);
        for idx in 0..max_spikes {
            work_units = work_units.saturating_add(7);
            Self::check_budget(work_units, "spikes", budget)?;

            let raw_mag = Self::normalized_unit(prng.next_u64());
            let magnitude = if raw_mag >= 0.8 { raw_mag } else { 0.0 };
            if magnitude > 0.0 {
                spikes.push(Spike {
                    feature_id: (prng.next_u64() as u32) ^ (idx as u32),
                    magnitude,
                    timestamp: input.t,
                });
            }
        }

        let mut notes = vec![
            format!("backend={}", self.name()),
            format!("frame={}", input.frame_id.0),
            format!(
                "digest_prefix={:02x}{:02x}",
                input.context_digest[0], input.context_digest[1]
            ),
        ];
        notes.sort();

        Ok(ComputeSignals {
            surprise,
            pressure,
            risk,
            confidence,
            spikes,
            notes,
        }
        .bounded())
    }
}

#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[cfg(feature = "compute-candle")]
pub mod candle {
    use crate::{
        AiComputeBackend, ComputeBudget, ComputeError, ComputeInput, ComputeSignals, CpuStubBackend,
    };

    #[derive(Debug, Default, Clone, Copy)]
    pub struct CandleBackend;

    impl AiComputeBackend for CandleBackend {
        fn name(&self) -> &'static str {
            "candle_dummy"
        }

        fn compute(
            &self,
            input: &ComputeInput,
            budget: ComputeBudget,
        ) -> Result<ComputeSignals, ComputeError> {
            CpuStubBackend.compute(input, budget)
        }
    }
}

#[cfg(feature = "compute-burn")]
pub mod burn {
    use crate::{
        AiComputeBackend, ComputeBudget, ComputeError, ComputeInput, ComputeSignals, CpuStubBackend,
    };

    #[derive(Debug, Default, Clone, Copy)]
    pub struct BurnBackend;

    impl AiComputeBackend for BurnBackend {
        fn name(&self) -> &'static str {
            "burn_dummy"
        }

        fn compute(
            &self,
            input: &ComputeInput,
            budget: ComputeBudget,
        ) -> Result<ComputeSignals, ComputeError> {
            CpuStubBackend.compute(input, budget)
        }
    }
}

pub fn frame_time_tick(time: SimTime) -> u64 {
    time.tick.get()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, serde::Deserialize)]
    struct FixtureCase {
        frame_id: u64,
        t: u64,
        context_digest_hex: String,
        seed: u64,
        expected: Expected,
    }

    #[derive(Debug, serde::Deserialize)]
    struct Expected {
        surprise: f32,
        pressure: f32,
        risk: f32,
        confidence: f32,
        spike_count: usize,
        spikes_digest_hex: String,
    }

    fn decode_hex32(hex: &str) -> [u8; 32] {
        let bytes = hex::decode(hex).expect("valid hex fixture");
        let mut out = [0_u8; 32];
        out.copy_from_slice(&bytes);
        out
    }

    #[test]
    fn deterministic_for_same_input_and_seed() {
        let backend = CpuStubBackend;
        let input = ComputeInput {
            frame_id: FrameId(42),
            t: 7,
            context_digest: [1_u8; 32],
        };
        let budget = ComputeBudget::default();
        let a = backend.compute(&input, budget).expect("compute");
        let b = backend.compute(&input, budget).expect("compute");
        assert_eq!(a, b);
    }

    #[test]
    fn boundedness_clamps_and_truncates() {
        let spikes = (0..300)
            .map(|i| Spike {
                feature_id: i as u32,
                magnitude: 2.0,
                timestamp: 9,
            })
            .collect();
        let notes = (0..20).map(|_| "x".repeat(400)).collect();
        let bounded = ComputeSignals {
            surprise: 2.0,
            pressure: -1.0,
            risk: 3.0,
            confidence: 4.0,
            spikes,
            notes,
        }
        .bounded();
        assert_eq!(bounded.spikes.len(), MAX_SPIKES);
        assert_eq!(bounded.notes.len(), MAX_NOTES);
        assert!(bounded.notes.iter().all(|n| n.len() <= MAX_NOTE_LEN));
        assert_eq!(bounded.surprise, 1.0);
        assert_eq!(bounded.pressure, 0.0);
        assert_eq!(bounded.risk, 1.0);
        assert_eq!(bounded.confidence, 1.0);
    }

    #[test]
    fn budget_exceeded_is_reported() {
        let backend = CpuStubBackend;
        let input = ComputeInput {
            frame_id: FrameId(1),
            t: 1,
            context_digest: [255_u8; 32],
        };
        let err = backend
            .compute(
                &input,
                ComputeBudget {
                    max_micros: 4,
                    hard_timeout_micros: 1,
                    seed: 0,
                },
            )
            .expect_err("should exceed");
        match err {
            ComputeError::BudgetExceeded { stage, .. } => assert_eq!(stage, "spikes"),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn golden_vectors_from_fixture() {
        let backend = CpuStubBackend;
        let cases: Vec<FixtureCase> =
            serde_json::from_str(include_str!("../fixtures/compute_inputs.json"))
                .expect("fixture parse");

        for case in cases {
            let input = ComputeInput {
                frame_id: FrameId(case.frame_id),
                t: case.t,
                context_digest: decode_hex32(&case.context_digest_hex),
            };
            let out = backend
                .compute(
                    &input,
                    ComputeBudget {
                        max_micros: 500,
                        hard_timeout_micros: 5_000,
                        seed: case.seed,
                    },
                )
                .expect("compute output");

            assert!((out.surprise - case.expected.surprise).abs() <= 1e-6);
            assert!((out.pressure - case.expected.pressure).abs() <= 1e-6);
            assert!((out.risk - case.expected.risk).abs() <= 1e-6);
            assert!((out.confidence - case.expected.confidence).abs() <= 1e-6);
            assert_eq!(out.spikes.len(), case.expected.spike_count);

            let summary = out.summary("cpu_stub");
            assert_eq!(
                hex::encode(summary.spikes_digest),
                case.expected.spikes_digest_hex
            );
        }
    }
}
