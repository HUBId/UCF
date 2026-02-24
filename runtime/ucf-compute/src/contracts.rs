use sha2::{Digest, Sha256};

use crate::backend_pack::BackendComponentId;
use crate::capabilities::{LlmRequest, LlmResponse};
use crate::evidence::{digest_canonical, spikes_digest, EvidenceChain};
use crate::feature_extractor::{SaeInput, SaeOutput, SAE_MAX_SPIKES};
use crate::lfm::{LfmInput, LfmOutput};
use crate::ssm::{SsmInput, SsmOutput};
use crate::world_model::{WorldModelInput, WorldModelOutput};

pub const MAX_REASON_CODES: usize = 8;
pub const MAX_STAGE_ENCODED_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[repr(u16)]
pub enum StageContractVersion {
    V1 = 1,
}

impl StageContractVersion {
    pub const fn as_u16(self) -> u16 {
        self as u16
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum ValidationStatus {
    Ok = 0,
    Warned = 1,
    Degraded = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum ViolationCode {
    ScalarOutOfRange = 1,
    SpikeCountExceeded = 2,
    SpikesDigestMismatch = 3,
    ReadoutDigestMismatch = 4,
    SizeCapExceeded = 5,
    ChainDigestMismatch = 6,
    BackendContractMismatch = 7,
    PressureJumpExceeded = 8,
    LlmDigestMismatch = 9,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidationReport {
    pub status: ValidationStatus,
    pub violation_mask: u32,
}

impl ValidationReport {
    pub const fn ok() -> Self {
        Self {
            status: ValidationStatus::Ok,
            violation_mask: 0,
        }
    }

    pub fn add_hard(&mut self, code: ViolationCode) {
        self.status = ValidationStatus::Degraded;
        self.set_code(code);
    }

    pub fn add_soft(&mut self, code: ViolationCode) {
        if self.status == ValidationStatus::Ok {
            self.status = ValidationStatus::Warned;
        }
        self.set_code(code);
    }

    fn set_code(&mut self, code: ViolationCode) {
        let shift = (code as u32).min(31);
        self.violation_mask |= 1_u32 << shift;
    }

    pub fn merge(mut self, rhs: Self) -> Self {
        self.status = match (self.status, rhs.status) {
            (ValidationStatus::Degraded, _) | (_, ValidationStatus::Degraded) => {
                ValidationStatus::Degraded
            }
            (ValidationStatus::Warned, _) | (_, ValidationStatus::Warned) => {
                ValidationStatus::Warned
            }
            _ => ValidationStatus::Ok,
        };
        self.violation_mask |= rhs.violation_mask;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageKind {
    World,
    Sae,
    Ssm,
    Lfm,
    Llm,
}

pub trait ContractRegistry {
    fn supports(
        &self,
        stage: StageKind,
        backend_id: BackendComponentId,
        version: StageContractVersion,
    ) -> bool;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct StageContractRegistry;

impl ContractRegistry for StageContractRegistry {
    fn supports(
        &self,
        stage: StageKind,
        backend_id: BackendComponentId,
        version: StageContractVersion,
    ) -> bool {
        if version != StageContractVersion::V1 {
            return false;
        }
        !matches!(
            (stage, backend_id),
            (_, BackendComponentId::Disabled) | (StageKind::World, BackendComponentId::StubV0)
        )
    }
}

pub struct WorldValidatorV1;
impl WorldValidatorV1 {
    pub fn validate(input: &WorldModelInput, output: &WorldModelOutput) -> ValidationReport {
        let mut report = ValidationReport::ok();
        if !(0.0..=1.0).contains(&output.surprise)
            || !(0.0..=1.0).contains(&output.prediction_error)
            || !(0.0..=1.0).contains(&output.state_norm)
        {
            report.add_hard(ViolationCode::ScalarOutOfRange);
        }
        if output.prediction_digest == [0; 32] && input.t > 0 {
            report.add_hard(ViolationCode::ScalarOutOfRange);
        }
        if serde_json::to_vec(output)
            .map(|buf| buf.len() > MAX_STAGE_ENCODED_BYTES)
            .unwrap_or(true)
        {
            report.add_hard(ViolationCode::SizeCapExceeded);
        }
        report
    }
}

pub struct SaeValidatorV1;
impl SaeValidatorV1 {
    pub fn validate(_input: &SaeInput, output: &SaeOutput) -> ValidationReport {
        let mut report = ValidationReport::ok();
        if usize::from(output.spike_count) > SAE_MAX_SPIKES || output.spikes.len() > SAE_MAX_SPIKES
        {
            report.add_hard(ViolationCode::SpikeCountExceeded);
        }
        if !(0.0..=1.0).contains(&output.sparsity) || !(0.0..=1.0).contains(&output.energy) {
            report.add_hard(ViolationCode::ScalarOutOfRange);
        }
        if output
            .spikes
            .iter()
            .any(|s| !(0.0..=1.0).contains(&s.magnitude))
        {
            report.add_hard(ViolationCode::ScalarOutOfRange);
        }
        if output
            .spikes
            .windows(2)
            .any(|pair| pair[0].feature_id >= pair[1].feature_id)
        {
            report.add_hard(ViolationCode::ScalarOutOfRange);
        }
        if spikes_digest(&output.spikes) != output.spikes_digest {
            report.add_hard(ViolationCode::SpikesDigestMismatch);
        }
        if serde_json::to_vec(output)
            .map(|buf| buf.len() > MAX_STAGE_ENCODED_BYTES)
            .unwrap_or(true)
        {
            report.add_hard(ViolationCode::SizeCapExceeded);
        }
        report
    }
}

pub struct SsmValidatorV1;
impl SsmValidatorV1 {
    pub fn validate(
        input: &SsmInput,
        output: &SsmOutput,
        previous_pressure: Option<f32>,
    ) -> ValidationReport {
        let mut report = ValidationReport::ok();
        if !(0.0..=1.0).contains(&output.pressure)
            || !(0.0..=1.0).contains(&output.readout)
            || !(0.0..=1.0).contains(&output.state_norm)
        {
            report.add_hard(ViolationCode::ScalarOutOfRange);
        }
        let expected_readout = {
            let mut hasher = Sha256::new();
            hasher.update(output.readout.to_bits().to_le_bytes());
            hasher.update(output.state_digest);
            hasher.finalize()
        };
        if expected_readout[..] != output.readout_digest {
            report.add_hard(ViolationCode::ReadoutDigestMismatch);
        }
        if input.spike_count as usize > SAE_MAX_SPIKES {
            report.add_hard(ViolationCode::SpikeCountExceeded);
        }
        if let Some(prev) = previous_pressure {
            if (output.pressure - prev).abs() > 0.65 {
                report.add_soft(ViolationCode::PressureJumpExceeded);
            }
        }
        if serde_json::to_vec(output)
            .map(|buf| buf.len() > MAX_STAGE_ENCODED_BYTES)
            .unwrap_or(true)
        {
            report.add_hard(ViolationCode::SizeCapExceeded);
        }
        report
    }
}

pub struct LfmValidatorV1;
impl LfmValidatorV1 {
    pub fn validate(input: &LfmInput, output: &LfmOutput) -> ValidationReport {
        let mut report = ValidationReport::ok();
        for scalar in [
            output.uncertainty,
            output.stability,
            output.state_norm,
            output.deriv_norm,
            output.saturation_ratio,
            output.homeostasis_error,
            input.surprise,
            input.sae_energy,
            input.pressure,
        ] {
            if !(0.0..=1.0).contains(&scalar) {
                report.add_hard(ViolationCode::ScalarOutOfRange);
            }
        }
        let mut readout_hasher = Sha256::new();
        readout_hasher.update(output.uncertainty_q.to_le_bytes());
        readout_hasher.update(output.stability_q.to_le_bytes());
        readout_hasher.update(output.homeostasis_error_q.to_le_bytes());
        readout_hasher.update(output.liquid_state_digest);
        let expected: [u8; 32] = readout_hasher.finalize().into();
        if expected != output.liquid_readout_digest {
            report.add_hard(ViolationCode::ReadoutDigestMismatch);
        }
        if output.nan_inf_detected {
            report.add_hard(ViolationCode::ScalarOutOfRange);
        }
        if serde_json::to_vec(output)
            .map(|buf| buf.len() > MAX_STAGE_ENCODED_BYTES)
            .unwrap_or(true)
        {
            report.add_hard(ViolationCode::SizeCapExceeded);
        }
        report
    }
}

pub struct LlmValidatorV1;
impl LlmValidatorV1 {
    pub fn validate(_req: &LlmRequest, response: &LlmResponse) -> ValidationReport {
        let mut report = ValidationReport::ok();
        if response.compute_digest() != response.digest {
            report.add_hard(ViolationCode::LlmDigestMismatch);
        }
        if serde_json::to_vec(&response.text)
            .map(|buf| buf.len() > MAX_STAGE_ENCODED_BYTES)
            .unwrap_or(true)
        {
            report.add_hard(ViolationCode::SizeCapExceeded);
        }
        report
    }
}

pub fn validate_evidence_chain_digest(chain: &EvidenceChain) -> ValidationReport {
    let mut cloned = *chain;
    cloned.chain_digest = [0; 32];
    let expected = digest_canonical(&cloned);
    let mut report = ValidationReport::ok();
    if expected != chain.chain_digest {
        report.add_hard(ViolationCode::ChainDigestMismatch);
    }
    report
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::capabilities::{FinishReason, LlmOutputClass, LlmRequest, LlmResponse, LlmStatus};

    proptest! {
        #[test]
        fn sae_validator_never_panics(spike_count in 0u16..300u16, mag in -2.0f32..2.0f32) {
            let spike = crate::Spike { feature_id: 1, magnitude: mag, timestamp: 1 };
            let output = SaeOutput {
                spikes: vec![spike],
                spike_count,
                sparsity: 0.5,
                energy: 0.5,
                spikes_digest: [0;32],
                quality: crate::world_model::StageQuality::Ok,
                notes: crate::feature_extractor::SmallNotes(vec![]),
            };
            let input = SaeInput { t:1, context_features:[0.0; crate::feature_extractor::SAE_INPUT_DIM], world_state_digest: None, seed: 1, evidence_chain_digest:[0;32]};
            let _ = SaeValidatorV1::validate(&input, &output);
        }
    }

    #[test]
    fn sae_digest_mutation_is_detected() {
        let mut out = SaeOutput {
            spikes: vec![crate::Spike {
                feature_id: 1,
                magnitude: 0.7,
                timestamp: 1,
            }],
            spike_count: 1,
            sparsity: 0.9,
            energy: 0.1,
            spikes_digest: [0; 32],
            quality: crate::world_model::StageQuality::Ok,
            notes: crate::feature_extractor::SmallNotes(vec![]),
        };
        out.spikes_digest = spikes_digest(&out.spikes);
        out.spikes[0].magnitude = 0.8;
        let input = SaeInput {
            t: 1,
            context_features: [0.0; crate::feature_extractor::SAE_INPUT_DIM],
            world_state_digest: None,
            seed: 1,
            evidence_chain_digest: [0; 32],
        };
        let report = SaeValidatorV1::validate(&input, &out);
        assert_eq!(report.status, ValidationStatus::Degraded);
    }

    #[test]
    fn sae_duplicate_feature_ids_are_rejected() {
        let mut out = SaeOutput {
            spikes: vec![
                crate::Spike {
                    feature_id: 2,
                    magnitude: 0.7,
                    timestamp: 1,
                },
                crate::Spike {
                    feature_id: 2,
                    magnitude: 0.6,
                    timestamp: 1,
                },
            ],
            spike_count: 2,
            sparsity: 0.8,
            energy: 0.2,
            spikes_digest: [0; 32],
            quality: crate::world_model::StageQuality::Ok,
            notes: crate::feature_extractor::SmallNotes(vec![]),
        };
        out.spikes_digest = spikes_digest(&out.spikes);
        let input = SaeInput {
            t: 1,
            context_features: [0.0; crate::feature_extractor::SAE_INPUT_DIM],
            world_state_digest: None,
            seed: 1,
            evidence_chain_digest: [0; 32],
        };
        let report = SaeValidatorV1::validate(&input, &out);
        assert_eq!(report.status, ValidationStatus::Degraded);
    }

    #[test]
    fn llm_digest_mutation_is_detected() {
        let req = LlmRequest {
            schema_version: 1,
            t: 1,
            decision_id: 1,
            candidate_id: 1,
            output_class: LlmOutputClass::SafeText,
            prompt: "ok".into(),
            context_digest: [0; 32],
            evidence_chain_digest: [1; 32],
            lfm_readout_digest: None,
            lfm_uncertainty: None,
            lfm_stability: None,
            coherence: None,
            instability: None,
            risk: None,
            confidence: None,
            seed: 1,
            max_tokens: 8,
            temperature: 0.1,
            top_p: 1.0,
            sampling_enabled: false,
        };
        let mut resp = LlmResponse::new(LlmStatus::Ok, "x".into(), 1, FinishReason::Stop);
        resp.digest = [0; 32];
        let report = LlmValidatorV1::validate(&req, &resp);
        assert_eq!(report.status, ValidationStatus::Degraded);
    }
}
