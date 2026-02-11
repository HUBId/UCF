use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum BackendProfileId {
    StubV1 = 1,
    CandleV1 = 2,
    BurnV1 = 3,
    UnknownV1 = 255,
}

impl BackendProfileId {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::StubV1 => "stub:v1",
            Self::CandleV1 => "candle:v1",
            Self::BurnV1 => "burn:v1",
            Self::UnknownV1 => "unknown:v1",
        }
    }

    pub fn from_backend_name(name: &str) -> Self {
        match name {
            "stub" => Self::StubV1,
            "candle" => Self::CandleV1,
            "burn" => Self::BurnV1,
            _ => Self::UnknownV1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum SignalQuality {
    VerifiedPipeline = 0,
    DegradedFallback = 1,
    Unavailable = 2,
}

impl SignalQuality {
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EvidenceRef {
    pub context_digest: [u8; 32],
    pub world_digest: Option<[u8; 32]>,
    pub spikes_digest: Option<[u8; 32]>,
    pub ssm_digest: Option<[u8; 32]>,
    pub backend_profile: BackendProfileId,
    pub seed: u64,
    pub budget_profile_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[repr(C)]
pub struct RiskSignal {
    pub risk: f32,
    pub confidence: f32,
    pub quality: SignalQuality,
    pub evidence: EvidenceRef,
    pub version: u16,
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum RiskContractError {
    #[error("risk or confidence is non-finite")]
    NonFinite,
    #[error("risk out of range [0,1]")]
    RiskOutOfRange,
    #[error("confidence out of range [0,1]")]
    ConfidenceOutOfRange,
    #[error("verified pipeline signal missing required evidence")]
    MissingVerifiedEvidence,
}

pub fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

pub fn validate_risk_signal(rs: &RiskSignal) -> Result<(), RiskContractError> {
    if !rs.risk.is_finite() || !rs.confidence.is_finite() {
        return Err(RiskContractError::NonFinite);
    }
    if !(0.0..=1.0).contains(&rs.risk) {
        return Err(RiskContractError::RiskOutOfRange);
    }
    if !(0.0..=1.0).contains(&rs.confidence) {
        return Err(RiskContractError::ConfidenceOutOfRange);
    }
    if rs.quality == SignalQuality::VerifiedPipeline
        && (rs.evidence.world_digest.is_none()
            || rs.evidence.spikes_digest.is_none()
            || rs.evidence.ssm_digest.is_none())
    {
        return Err(RiskContractError::MissingVerifiedEvidence);
    }
    Ok(())
}

pub fn stable_budget_profile_id(max_micros: u64, hard_timeout_micros: u64) -> u32 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    h ^= max_micros;
    h = h.wrapping_mul(0x100_0000_01b3);
    h ^= hard_timeout_micros.rotate_left(13);
    h = h.wrapping_mul(0x100_0000_01b3);
    (h as u32) ^ ((h >> 32) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_signal() -> RiskSignal {
        RiskSignal {
            risk: 0.3,
            confidence: 0.7,
            quality: SignalQuality::VerifiedPipeline,
            evidence: EvidenceRef {
                context_digest: [1; 32],
                world_digest: Some([2; 32]),
                spikes_digest: Some([3; 32]),
                ssm_digest: Some([4; 32]),
                backend_profile: BackendProfileId::StubV1,
                seed: 1,
                budget_profile_id: 7,
            },
            version: 1,
        }
    }

    #[test]
    fn validate_rejects_non_finite() {
        let mut rs = base_signal();
        rs.risk = f32::NAN;
        assert_eq!(validate_risk_signal(&rs), Err(RiskContractError::NonFinite));
    }

    #[test]
    fn validate_rejects_out_of_range() {
        let mut rs = base_signal();
        rs.risk = 1.2;
        assert_eq!(
            validate_risk_signal(&rs),
            Err(RiskContractError::RiskOutOfRange)
        );

        rs = base_signal();
        rs.confidence = -0.1;
        assert_eq!(
            validate_risk_signal(&rs),
            Err(RiskContractError::ConfidenceOutOfRange)
        );
    }

    #[test]
    fn verified_pipeline_requires_evidence() {
        let mut rs = base_signal();
        rs.evidence.world_digest = None;
        assert_eq!(
            validate_risk_signal(&rs),
            Err(RiskContractError::MissingVerifiedEvidence)
        );
    }
}
