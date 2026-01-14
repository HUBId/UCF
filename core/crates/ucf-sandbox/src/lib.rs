#![forbid(unsafe_code)]

use std::collections::HashSet;

use ucf_commit::{canonical_control_frame_len, commit_control_frame, Commitment};
use ucf_types::v1::spec::{ControlFrame, DecisionKind, PolicyDecision};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum SandboxErrorCode {
    INVALID_SCHEMA,
    INVALID_RANGE,
    INVALID_CONTEXT_BINDING,
    UNSUPPORTED_VERSION,
    SIZE_LIMIT,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SandboxError {
    pub code: SandboxErrorCode,
    pub message: String,
    pub field: Option<&'static str>,
}

impl SandboxError {
    fn new(
        code: SandboxErrorCode,
        field: Option<&'static str>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code,
            message: message.into(),
            field,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatorLimits {
    pub max_bytes: usize,
    pub max_context_items: usize,
    pub max_depth: u8,
    pub supported_versions: Vec<u16>,
}

impl Default for ValidatorLimits {
    fn default() -> Self {
        Self {
            max_bytes: 64 * 1024,
            max_context_items: 32,
            max_depth: 8,
            supported_versions: vec![1],
        }
    }
}

#[derive(Clone, Debug)]
pub struct ControlFrameValidator {
    limits: ValidatorLimits,
}

impl ControlFrameValidator {
    pub fn new(limits: ValidatorLimits) -> Self {
        Self { limits }
    }

    pub fn limits(&self) -> &ValidatorLimits {
        &self.limits
    }

    pub fn validate(&self, cf: &ControlFrame) -> Result<(), SandboxError> {
        self.validate_schema(cf)?;
        self.validate_context_bindings(cf)?;
        self.validate_ranges(cf)?;
        Ok(())
    }

    pub fn validate_and_normalize(
        &self,
        cf: ControlFrame,
    ) -> Result<ControlFrameNormalized, SandboxError> {
        self.validate(&cf)?;
        let normalized = normalize(cf);
        self.validate_size(&normalized)?;
        Ok(normalized)
    }

    fn validate_schema(&self, cf: &ControlFrame) -> Result<(), SandboxError> {
        let version = control_frame_version(cf);
        if !self.limits.supported_versions.contains(&version) {
            return Err(SandboxError::new(
                SandboxErrorCode::UNSUPPORTED_VERSION,
                Some("version"),
                format!("unsupported control frame version {version}"),
            ));
        }

        if cf.frame_id.trim().is_empty() {
            return Err(SandboxError::new(
                SandboxErrorCode::INVALID_SCHEMA,
                Some("frame_id"),
                "frame_id is required",
            ));
        }

        if cf.policy_id.trim().is_empty() {
            return Err(SandboxError::new(
                SandboxErrorCode::INVALID_SCHEMA,
                Some("policy_id"),
                "policy_id is required",
            ));
        }

        let decision = cf.decision.as_ref().ok_or_else(|| {
            SandboxError::new(
                SandboxErrorCode::INVALID_SCHEMA,
                Some("decision"),
                "decision is required",
            )
        })?;

        if DecisionKind::try_from(decision.kind).is_err() {
            return Err(SandboxError::new(
                SandboxErrorCode::INVALID_SCHEMA,
                Some("decision.kind"),
                "decision kind is invalid",
            ));
        }

        Ok(())
    }

    fn validate_context_bindings(&self, cf: &ControlFrame) -> Result<(), SandboxError> {
        let mut unique_refs = HashSet::new();
        for evidence_id in &cf.evidence_ids {
            if !unique_refs.insert(evidence_id.as_str()) {
                return Err(SandboxError::new(
                    SandboxErrorCode::INVALID_CONTEXT_BINDING,
                    Some("evidence_ids"),
                    "duplicate evidence reference",
                ));
            }
        }

        if unique_refs.len() > self.limits.max_context_items {
            return Err(SandboxError::new(
                SandboxErrorCode::INVALID_CONTEXT_BINDING,
                Some("evidence_ids"),
                format!(
                    "evidence reference count {} exceeds limit {}",
                    unique_refs.len(),
                    self.limits.max_context_items
                ),
            ));
        }

        Ok(())
    }

    fn validate_ranges(&self, cf: &ControlFrame) -> Result<(), SandboxError> {
        if let Some(decision) = cf.decision.as_ref() {
            if decision.confidence_bp > 10_000 {
                return Err(SandboxError::new(
                    SandboxErrorCode::INVALID_RANGE,
                    Some("decision.confidence_bp"),
                    "confidence_bp must be between 0 and 10000",
                ));
            }
        }

        if let Some(depth) = recursion_depth_hint(cf) {
            if depth > self.limits.max_depth {
                return Err(SandboxError::new(
                    SandboxErrorCode::INVALID_RANGE,
                    Some("recursion_depth"),
                    format!(
                        "recursion depth {depth} exceeds limit {}",
                        self.limits.max_depth
                    ),
                ));
            }
        }

        Ok(())
    }

    fn validate_size(&self, normalized: &ControlFrameNormalized) -> Result<(), SandboxError> {
        let size = canonical_control_frame_len(normalized.as_ref());
        if size > self.limits.max_bytes {
            return Err(SandboxError::new(
                SandboxErrorCode::SIZE_LIMIT,
                Some("control_frame"),
                format!(
                    "canonical size {size} exceeds limit {}",
                    self.limits.max_bytes
                ),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ControlFrameNormalized {
    inner: ControlFrame,
}

impl ControlFrameNormalized {
    pub fn as_ref(&self) -> &ControlFrame {
        &self.inner
    }

    pub fn into_inner(self) -> ControlFrame {
        self.inner
    }

    pub fn commitment(&self) -> Commitment {
        commit_control_frame(&self.inner)
    }
}

impl From<ControlFrameNormalized> for ControlFrame {
    fn from(value: ControlFrameNormalized) -> Self {
        value.inner
    }
}

pub fn normalize(mut cf: ControlFrame) -> ControlFrameNormalized {
    cf.evidence_ids.sort();
    cf.evidence_ids.dedup();

    if let Some(decision) = cf.decision.as_mut() {
        normalize_decision(decision);
    }

    ControlFrameNormalized { inner: cf }
}

fn normalize_decision(decision: &mut PolicyDecision) {
    decision.constraint_ids.sort();
    decision.constraint_ids.dedup();
}

fn control_frame_version(_cf: &ControlFrame) -> u16 {
    1
}

fn recursion_depth_hint(_cf: &ControlFrame) -> Option<u8> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_types::v1::spec::{ActionCode, DecisionKind};

    fn base_frame() -> ControlFrame {
        ControlFrame {
            frame_id: "frame-1".to_string(),
            issued_at_ms: 1_700_000,
            decision: Some(PolicyDecision {
                kind: DecisionKind::DecisionKindAllow as i32,
                action: ActionCode::ActionCodeContinue as i32,
                rationale: "ok".to_string(),
                confidence_bp: 1000,
                constraint_ids: vec!["c2".to_string(), "c1".to_string()],
            }),
            evidence_ids: vec!["e2".to_string(), "e1".to_string()],
            policy_id: "policy-1".to_string(),
        }
    }

    #[test]
    fn validator_accepts_minimal_valid_control_frame() {
        let mut limits = ValidatorLimits::default();
        limits.max_context_items = 4;
        let validator = ControlFrameValidator::new(limits);
        let frame = ControlFrame {
            evidence_ids: vec!["e1".to_string()],
            ..base_frame()
        };

        let normalized = validator
            .validate_and_normalize(frame)
            .expect("valid control frame");
        assert_eq!(normalized.as_ref().evidence_ids, vec!["e1"]);
    }

    #[test]
    fn validator_rejects_unsupported_version() {
        let limits = ValidatorLimits {
            supported_versions: vec![2],
            ..ValidatorLimits::default()
        };
        let validator = ControlFrameValidator::new(limits);

        let err = validator
            .validate(&base_frame())
            .expect_err("unsupported version");

        assert_eq!(err.code, SandboxErrorCode::UNSUPPORTED_VERSION);
    }

    #[test]
    fn validator_rejects_too_many_context_refs() {
        let limits = ValidatorLimits {
            max_context_items: 1,
            ..ValidatorLimits::default()
        };
        let validator = ControlFrameValidator::new(limits);

        let mut frame = base_frame();
        frame.evidence_ids = vec!["e1".to_string(), "e2".to_string()];

        let err = validator.validate(&frame).expect_err("too many refs");

        assert_eq!(err.code, SandboxErrorCode::INVALID_CONTEXT_BINDING);
    }

    #[test]
    fn validator_rejects_invalid_ranges() {
        let validator = ControlFrameValidator::new(ValidatorLimits::default());
        let mut frame = base_frame();
        frame.decision.as_mut().unwrap().confidence_bp = 10_001;

        let err = validator.validate(&frame).expect_err("invalid range");

        assert_eq!(err.code, SandboxErrorCode::INVALID_RANGE);
    }

    #[test]
    fn normalization_is_deterministic() {
        let mut frame_a = base_frame();
        frame_a.evidence_ids = vec!["e3".to_string(), "e1".to_string(), "e2".to_string()];

        let mut frame_b = base_frame();
        frame_b.evidence_ids = vec!["e2".to_string(), "e3".to_string(), "e1".to_string()];

        let normalized_a = normalize(frame_a).commitment();
        let normalized_b = normalize(frame_b).commitment();

        assert_eq!(normalized_a, normalized_b);
    }
}
