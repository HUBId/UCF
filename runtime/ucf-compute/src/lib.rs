use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;
use ucf_frames::v1::{ControlFrame, ControlPayload};
use world_model::StageQuality;

pub mod backends;
pub mod capabilities;
pub mod evidence;
pub mod feature_extractor;
pub mod pipeline;
pub mod risk_contract;
pub mod ssm;
pub mod work_meter;
pub mod world_model;
pub use backends::{build_backend, ComputeBackendConfig, ComputeBackendKind};
pub use evidence::{CodeVersionTag, EvidenceChain, COMPUTE_SUMMARY_SCHEMA_VERSION};
pub use pipeline::ComputePipelineBackend;
pub use risk_contract::{
    clamp01, stable_budget_profile_id, validate_risk_signal, BackendProfileId, EvidenceRef,
    RiskSignal, SignalQuality,
};

pub const MAX_SPIKES: usize = 256;
pub const MAX_NOTES: usize = 16;
pub const MAX_NOTE_LEN: usize = 256;

static UCF_COMPUTE_CHAIN_DIGEST_EMITTED_TOTAL: AtomicU64 = AtomicU64::new(0);

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
    pub risk_signal: RiskSignal,
    pub spikes: Vec<Spike>,
    pub notes: Vec<String>,
    pub sparsity: Option<f32>,
    pub energy: Option<f32>,
    pub ssm_readout: Option<f32>,
    pub ssm_digest: Option<[u8; 32]>,
    pub world_digest: Option<[u8; 32]>,
    pub sae_quality: Option<StageQuality>,
    pub ssm_quality: Option<StageQuality>,
    pub budget_exceeded_stage: Option<&'static str>,
}

impl ComputeSignals {
    pub fn bounded(mut self) -> Self {
        self.surprise = self.surprise.clamp(0.0, 1.0);
        self.pressure = self.pressure.clamp(0.0, 1.0);
        self.risk_signal.risk = clamp01(self.risk_signal.risk);
        self.risk_signal.confidence = clamp01(self.risk_signal.confidence);
        self.risk = self.risk_signal.risk;
        self.confidence = self.risk_signal.confidence;
        self.sparsity = self.sparsity.map(|v| v.clamp(0.0, 1.0));
        self.energy = self.energy.map(|v| v.clamp(0.0, 1.0));
        self.ssm_readout = self.ssm_readout.map(|v| v.clamp(0.0, 1.0));

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
        let spikes_digest = evidence::spikes_digest(&self.spikes);
        let risk_signal = if validate_risk_signal(&self.risk_signal).is_ok() {
            self.risk_signal
        } else {
            RiskSignal {
                risk: 1.0,
                confidence: 0.0,
                quality: SignalQuality::Unavailable,
                evidence: self.risk_signal.evidence,
                version: 1,
            }
        };
        let evidence_chain = EvidenceChain::from_compute(
            &ComputeInput {
                frame_id: FrameId(0),
                t: 0,
                context_digest: risk_signal.evidence.context_digest,
            },
            &self.spikes,
            &risk_signal,
            self.sae_quality,
            self.ssm_quality,
        );
        UCF_COMPUTE_CHAIN_DIGEST_EMITTED_TOTAL.fetch_add(1, Ordering::Relaxed);
        ComputeSignalsSummary {
            backend,
            surprise: self.surprise,
            pressure: self.pressure,
            risk: risk_signal.risk,
            confidence: risk_signal.confidence,
            spike_count: self.spikes.len() as u16,
            spikes_digest,
            sparsity: self.sparsity,
            energy: self.energy,
            ssm_readout: self.ssm_readout,
            ssm_digest: self.ssm_digest,
            world_digest: self.world_digest,
            risk_quality: risk_signal.quality.as_u8(),
            evidence_context_digest: risk_signal.evidence.context_digest,
            evidence_world_digest: risk_signal.evidence.world_digest,
            evidence_spikes_digest: risk_signal.evidence.spikes_digest,
            evidence_ssm_digest: risk_signal.evidence.ssm_digest,
            ssm_quality: self.ssm_quality,
            backend_profile: risk_signal.evidence.backend_profile.as_str(),
            budget_profile_id: risk_signal.evidence.budget_profile_id,
            seed: risk_signal.evidence.seed,
            risk_contract_version: risk_signal.version,
            compute_schema_version: evidence_chain.schema_version,
            compute_chain_digest: evidence_chain.chain_digest,
            compute_code_version: evidence_chain.code_version.as_str(),
            budget_exceeded_stage: self.budget_exceeded_stage,
        }
    }

    pub fn unavailable(input: &ComputeInput, budget: ComputeBudget, backend: &'static str) -> Self {
        let evidence = EvidenceRef {
            context_digest: input.context_digest,
            world_digest: None,
            spikes_digest: None,
            ssm_digest: None,
            backend_profile: BackendProfileId::from_backend_name(backend),
            seed: budget.seed,
            budget_profile_id: budget.profile_id,
        };
        Self {
            surprise: 0.0,
            pressure: 1.0,
            risk: 1.0,
            confidence: 0.0,
            risk_signal: RiskSignal {
                risk: 1.0,
                confidence: 0.0,
                quality: SignalQuality::Unavailable,
                evidence,
                version: 1,
            },
            spikes: Vec::new(),
            notes: vec!["risk_contract:unavailable".to_string()],
            sparsity: None,
            energy: None,
            ssm_readout: None,
            ssm_digest: None,
            world_digest: None,
            sae_quality: None,
            ssm_quality: None,
            budget_exceeded_stage: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DegradePolicy {
    DegradeStages,
    FailFast,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ComputeBudgetProfile {
    pub profile_id: u32,
    pub global_work_units: u64,
    pub world_units: u64,
    pub sae_units: u64,
    pub ssm_units: u64,
    pub degrade_policy: DegradePolicy,
}

impl ComputeBudgetProfile {
    pub fn default_profile() -> Self {
        Self {
            profile_id: 1,
            global_work_units: 1_600,
            world_units: 420,
            sae_units: 420,
            ssm_units: 420,
            degrade_policy: DegradePolicy::DegradeStages,
        }
    }

    pub fn tight_profile() -> Self {
        Self {
            profile_id: 2,
            global_work_units: 1_100,
            world_units: 360,
            sae_units: 260,
            ssm_units: 360,
            degrade_policy: DegradePolicy::DegradeStages,
        }
    }

    pub fn stress_profile() -> Self {
        Self {
            profile_id: 3,
            global_work_units: 900,
            world_units: 360,
            sae_units: 100,
            ssm_units: 360,
            degrade_policy: DegradePolicy::DegradeStages,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeBudget {
    pub max_micros: u64,
    pub hard_timeout_micros: u64,
    pub seed: u64,
    pub profile_id: u32,
    pub global_work_units: u64,
    pub world_units: u64,
    pub sae_units: u64,
    pub ssm_units: u64,
    pub degrade_policy: DegradePolicy,
}

impl Default for ComputeBudget {
    fn default() -> Self {
        Self {
            max_micros: 1_000,
            hard_timeout_micros: 5_000,
            seed: 0xDEC0DED,
            profile_id: ComputeBudgetProfile::default_profile().profile_id,
            global_work_units: ComputeBudgetProfile::default_profile().global_work_units,
            world_units: ComputeBudgetProfile::default_profile().world_units,
            sae_units: ComputeBudgetProfile::default_profile().sae_units,
            ssm_units: ComputeBudgetProfile::default_profile().ssm_units,
            degrade_policy: ComputeBudgetProfile::default_profile().degrade_policy,
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
    #[error("compute backend not implemented")]
    NotImplemented,
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
    pub sparsity: Option<f32>,
    pub energy: Option<f32>,
    pub ssm_readout: Option<f32>,
    pub ssm_digest: Option<[u8; 32]>,
    pub world_digest: Option<[u8; 32]>,
    pub risk_quality: u8,
    pub evidence_context_digest: [u8; 32],
    pub evidence_world_digest: Option<[u8; 32]>,
    pub evidence_spikes_digest: Option<[u8; 32]>,
    pub evidence_ssm_digest: Option<[u8; 32]>,
    pub ssm_quality: Option<StageQuality>,
    pub backend_profile: &'static str,
    pub budget_profile_id: u32,
    pub seed: u64,
    pub risk_contract_version: u16,
    pub compute_schema_version: u16,
    pub compute_chain_digest: [u8; 32],
    pub compute_code_version: &'static str,
    pub budget_exceeded_stage: Option<&'static str>,
}

pub fn fuse_signals(surprise: f32, pressure: f32, energy: f32) -> (f32, f32) {
    let base_risk = (0.65 * surprise + 0.35 * pressure).clamp(0.0, 1.0);
    let energy_adj = (energy - 0.5).clamp(-0.5, 0.5);
    let risk = (base_risk + 0.15 * energy_adj).clamp(0.0, 1.0);
    let confidence = (1.0 - 0.9 * risk).clamp(0.0, 1.0);
    (risk, confidence)
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

impl AiComputeBackend for CpuStubBackend {
    fn name(&self) -> &'static str {
        "stub"
    }

    fn compute(
        &self,
        input: &ComputeInput,
        budget: ComputeBudget,
    ) -> Result<ComputeSignals, ComputeError> {
        ComputePipelineBackend::stub().compute(input, budget)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capabilities::WorldModelPredictor;
    use crate::world_model::{obs_features_from_context, MockJepaPredictor, WorldModelInput};

    #[allow(dead_code)]
    #[derive(Debug, serde::Deserialize)]
    struct FixtureCase {
        frame_id: u64,
        t: u64,
        context_digest_hex: String,
        seed: u64,
        expected: Expected,
    }

    #[allow(dead_code)]
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
    fn fusion_is_monotonic_and_confidence_inverse() {
        let (r1, _) = fuse_signals(0.2, 0.4, 0.5);
        let (r2, _) = fuse_signals(0.8, 0.4, 0.5);
        assert!(r2 >= r1);

        let (r3, _) = fuse_signals(0.4, 0.2, 0.5);
        let (r4, _) = fuse_signals(0.4, 0.8, 0.5);
        assert!(r4 >= r3);

        let (risk_low, conf_high) = fuse_signals(0.1, 0.1, 0.5);
        let (risk_high, conf_low) = fuse_signals(0.9, 0.9, 0.5);
        assert!(risk_high >= risk_low);
        assert!(conf_low <= conf_high);
    }

    #[test]
    fn surprise_is_driven_by_world_model_predictor() {
        let backend = CpuStubBackend;
        let input = ComputeInput {
            frame_id: FrameId(1337),
            t: 19,
            context_digest: [0x2A_u8; 32],
        };
        let budget = ComputeBudget {
            max_micros: 500,
            hard_timeout_micros: 5_000,
            seed: 17,
            ..ComputeBudget::default()
        };

        let out = backend.compute(&input, budget).expect("compute");
        let mut predictor = MockJepaPredictor::default();
        let model = predictor
            .step(
                &WorldModelInput {
                    t: input.t,
                    context_digest: input.context_digest,
                    obs_features: obs_features_from_context(input.context_digest),
                    seed: budget.seed,
                },
                budget,
            )
            .expect("predict");

        assert!((out.surprise - model.surprise).abs() <= 1e-6);
        assert!(out.notes.iter().any(|n| n == "world_model=mock_jepa_v0"));
        assert!(out.notes.iter().any(|n| n.starts_with("pred_digest=")));
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
            risk_signal: RiskSignal {
                risk: 3.0,
                confidence: 4.0,
                quality: SignalQuality::Unavailable,
                evidence: EvidenceRef {
                    context_digest: [0; 32],
                    world_digest: None,
                    spikes_digest: None,
                    ssm_digest: None,
                    backend_profile: BackendProfileId::StubV1,
                    seed: 0,
                    budget_profile_id: 0,
                },
                version: 1,
            },
            spikes,
            notes,
            sparsity: Some(2.0),
            energy: Some(-1.0),
            ssm_readout: Some(3.0),
            ssm_digest: None,
            world_digest: None,
            sae_quality: None,
            ssm_quality: None,
            budget_exceeded_stage: None,
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
    fn unavailable_signal_is_safe_default() {
        let input = ComputeInput {
            frame_id: FrameId(1),
            t: 1,
            context_digest: [7; 32],
        };
        let sig = ComputeSignals::unavailable(&input, ComputeBudget::default(), "stub");
        assert_eq!(sig.risk, 1.0);
        assert_eq!(sig.confidence, 0.0);
        assert_eq!(sig.risk_signal.quality, SignalQuality::Unavailable);
    }

    #[test]
    fn budget_exceeded_is_reported() {
        let backend = CpuStubBackend;
        let input = ComputeInput {
            frame_id: FrameId(1),
            t: 1,
            context_digest: [255_u8; 32],
        };
        let out = backend
            .compute(
                &input,
                ComputeBudget {
                    max_micros: 4,
                    hard_timeout_micros: 1,
                    seed: 0,
                    ..ComputeBudget::default()
                },
            )
            .expect("should degrade deterministically");
        assert_eq!(out.risk_signal.quality, SignalQuality::DegradedFallback);
        assert_eq!(out.budget_exceeded_stage, Some("ssm/step"));
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
                        ..ComputeBudget::default()
                    },
                )
                .expect("compute output");

            assert!((0.0..=1.0).contains(&out.surprise));
            assert!((0.0..=1.0).contains(&out.pressure));
            assert!((0.0..=1.0).contains(&out.risk));
            assert!((0.0..=1.0).contains(&out.confidence));
            let replay = CpuStubBackend
                .compute(
                    &input,
                    ComputeBudget {
                        max_micros: 500,
                        hard_timeout_micros: 5_000,
                        seed: case.seed,
                        ..ComputeBudget::default()
                    },
                )
                .expect("compute replay");
            assert_eq!(out, replay);

            let summary = out.summary("stub");
            assert_ne!(summary.spikes_digest, [0_u8; 32]);
        }
    }
}
