#![forbid(unsafe_code)]

use std::collections::HashSet;

use blake3::Hasher;
use ucf_commit::{canonical_control_frame_len, commit_control_frame, Commitment};
use ucf_types::v1::spec::{ControlFrame, DecisionKind, PolicyDecision};
use ucf_types::{AiOutput, Digest32};

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
    pub fn into_inner(self) -> ControlFrame {
        self.inner
    }

    pub fn commitment(&self) -> Commitment {
        commit_control_frame(&self.inner)
    }
}

impl AsRef<ControlFrame> for ControlFrameNormalized {
    fn as_ref(&self) -> &ControlFrame {
        &self.inner
    }
}

impl From<ControlFrameNormalized> for ControlFrame {
    fn from(value: ControlFrameNormalized) -> Self {
        value.inner
    }
}

pub const AI_MODE_THOUGHT: u16 = 1;
pub const AI_MODE_SPEECH: u16 = 2;
pub const AI_MODE_INTERNAL: u16 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SandboxBudget {
    pub ops: u64,
    pub max_output_chars: usize,
    pub max_frames: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SandboxCaps {
    pub max_heap_bytes: usize,
    pub max_stack_bytes: usize,
}

impl Default for SandboxCaps {
    fn default() -> Self {
        Self {
            max_heap_bytes: 16 * 1024 * 1024,
            max_stack_bytes: 512 * 1024,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SandboxVerdict {
    Allow,
    Deny { reason: String },
}

impl SandboxVerdict {
    pub fn is_allow(&self) -> bool {
        matches!(self, SandboxVerdict::Allow)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SandboxReport {
    pub verdict: SandboxVerdict,
    pub ops_used: u64,
    pub commit: Digest32,
}

impl SandboxReport {
    fn new(verdict: SandboxVerdict, ops_used: u64, request_commit: Digest32) -> Self {
        let commit = sandbox_report_commit(&verdict, ops_used, request_commit);
        Self {
            verdict,
            ops_used,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntentSummary {
    pub intent: u16,
    pub risk: u16,
    pub commit: Digest32,
}

impl IntentSummary {
    pub fn new(intent: u16, risk: u16) -> Self {
        let commit = intent_summary_commit(intent, risk);
        Self {
            intent,
            risk,
            commit,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AiCallRequest {
    pub cycle_id: u64,
    pub input_commit: Digest32,
    pub mode: u16,
    pub budget: SandboxBudget,
    pub commit: Digest32,
}

impl AiCallRequest {
    pub fn new(cycle_id: u64, input_commit: Digest32, mode: u16, budget: SandboxBudget) -> Self {
        let commit = sandbox_request_commit(cycle_id, input_commit, mode, budget);
        Self {
            cycle_id,
            input_commit,
            mode,
            budget,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AiCallResult {
    pub outputs: Vec<AiOutput>,
    pub ops_used: u64,
    pub commit: Digest32,
}

impl AiCallResult {
    pub fn new(outputs: Vec<AiOutput>, ops_used: u64, request_commit: Digest32) -> Self {
        let commit = sandbox_result_commit(&outputs, ops_used, request_commit);
        Self {
            outputs,
            ops_used,
            commit,
        }
    }
}

pub trait SandboxPort {
    fn evaluate_call(
        &mut self,
        cf: &ControlFrameNormalized,
        intent: &IntentSummary,
        req: &AiCallRequest,
    ) -> SandboxReport;
    fn run_ai(&mut self, req: &AiCallRequest) -> Result<AiCallResult, SandboxReport>;
}

pub trait AiWorker: Send + Sync {
    fn run(
        &mut self,
        cf: &ControlFrameNormalized,
        intent: &IntentSummary,
        req: &AiCallRequest,
    ) -> Vec<AiOutput>;
}

struct PendingCall {
    cf: ControlFrameNormalized,
    intent: IntentSummary,
    req: AiCallRequest,
}

pub struct MockWasmSandbox {
    worker: Box<dyn AiWorker + Send + Sync>,
    caps: SandboxCaps,
    pending: Option<PendingCall>,
}

impl MockWasmSandbox {
    pub fn new(worker: Box<dyn AiWorker + Send + Sync>, caps: SandboxCaps) -> Self {
        Self {
            worker,
            caps,
            pending: None,
        }
    }

    pub fn caps(&self) -> SandboxCaps {
        self.caps
    }

    fn estimate_ops(input_commit: Digest32, mode: u16) -> u64 {
        const MAX_DETERMINISTIC_OPS: u64 = 5_000;
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&input_commit.as_bytes()[0..8]);
        let seed = u64::from_be_bytes(bytes) ^ (u64::from(mode) << 32);
        seed % MAX_DETERMINISTIC_OPS + 1
    }

    fn enforce_output_caps(
        &self,
        outputs: &[AiOutput],
        budget: SandboxBudget,
        ops_used: u64,
        request_commit: Digest32,
    ) -> Result<(), SandboxReport> {
        if outputs.len() > budget.max_frames as usize {
            return Err(SandboxReport::new(
                SandboxVerdict::Deny {
                    reason: "FRAME_CAP".to_string(),
                },
                ops_used,
                request_commit,
            ));
        }
        if outputs
            .iter()
            .any(|output| output.content.len() > budget.max_output_chars)
        {
            return Err(SandboxReport::new(
                SandboxVerdict::Deny {
                    reason: "OUTPUT_CAP".to_string(),
                },
                ops_used,
                request_commit,
            ));
        }
        Ok(())
    }
}

impl SandboxPort for MockWasmSandbox {
    fn evaluate_call(
        &mut self,
        cf: &ControlFrameNormalized,
        intent: &IntentSummary,
        req: &AiCallRequest,
    ) -> SandboxReport {
        let ops_used = Self::estimate_ops(req.input_commit, req.mode);
        if ops_used > req.budget.ops {
            self.pending = None;
            return SandboxReport::new(
                SandboxVerdict::Deny {
                    reason: "BUDGET_EXCEEDED".to_string(),
                },
                ops_used,
                req.commit,
            );
        }
        self.pending = Some(PendingCall {
            cf: cf.clone(),
            intent: intent.clone(),
            req: *req,
        });
        SandboxReport::new(SandboxVerdict::Allow, ops_used, req.commit)
    }

    fn run_ai(&mut self, req: &AiCallRequest) -> Result<AiCallResult, SandboxReport> {
        let pending = self.pending.take().ok_or_else(|| {
            SandboxReport::new(
                SandboxVerdict::Deny {
                    reason: "NO_PENDING_CALL".to_string(),
                },
                0,
                req.commit,
            )
        })?;
        if pending.req.commit != req.commit {
            return Err(SandboxReport::new(
                SandboxVerdict::Deny {
                    reason: "CALL_MISMATCH".to_string(),
                },
                0,
                req.commit,
            ));
        }
        let ops_used = Self::estimate_ops(req.input_commit, req.mode);
        if ops_used > req.budget.ops {
            return Err(SandboxReport::new(
                SandboxVerdict::Deny {
                    reason: "BUDGET_EXCEEDED".to_string(),
                },
                ops_used,
                req.commit,
            ));
        }
        let outputs = self.worker.run(&pending.cf, &pending.intent, req);
        self.enforce_output_caps(&outputs, req.budget, ops_used, req.commit)?;
        Ok(AiCallResult::new(outputs, ops_used, req.commit))
    }
}

fn intent_summary_commit(intent: u16, risk: u16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.sandbox.intent.v1");
    hasher.update(&intent.to_be_bytes());
    hasher.update(&risk.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn sandbox_request_commit(
    cycle_id: u64,
    input_commit: Digest32,
    mode: u16,
    budget: SandboxBudget,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.sandbox.req.v1");
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(input_commit.as_bytes());
    hasher.update(&mode.to_be_bytes());
    hasher.update(&budget.ops.to_be_bytes());
    hasher.update(&usize_to_u64(budget.max_output_chars).to_be_bytes());
    hasher.update(&budget.max_frames.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn sandbox_result_commit(
    outputs: &[AiOutput],
    ops_used: u64,
    request_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.sandbox.result.v1");
    hasher.update(request_commit.as_bytes());
    hasher.update(&ops_used.to_be_bytes());
    hasher.update(&u64::try_from(outputs.len()).unwrap_or(0).to_be_bytes());
    for output in outputs {
        hasher.update(&[output.channel as u8]);
        hasher.update(
            &u64::try_from(output.content.len())
                .unwrap_or(0)
                .to_be_bytes(),
        );
        hasher.update(output.content.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn sandbox_report_commit(
    verdict: &SandboxVerdict,
    ops_used: u64,
    request_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.sandbox.report.v1");
    hasher.update(request_commit.as_bytes());
    hasher.update(&ops_used.to_be_bytes());
    match verdict {
        SandboxVerdict::Allow => {
            hasher.update(&[1]);
        }
        SandboxVerdict::Deny { reason } => {
            hasher.update(&[0]);
            hasher.update(reason.as_bytes());
        }
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
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
    use ucf_types::OutputChannel;

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
        let limits = ValidatorLimits {
            max_context_items: 4,
            ..ValidatorLimits::default()
        };
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

    struct EmptyWorker;

    impl AiWorker for EmptyWorker {
        fn run(
            &mut self,
            _cf: &ControlFrameNormalized,
            _intent: &IntentSummary,
            _req: &AiCallRequest,
        ) -> Vec<AiOutput> {
            Vec::new()
        }
    }

    struct LoudWorker;

    impl AiWorker for LoudWorker {
        fn run(
            &mut self,
            _cf: &ControlFrameNormalized,
            _intent: &IntentSummary,
            _req: &AiCallRequest,
        ) -> Vec<AiOutput> {
            vec![AiOutput {
                channel: OutputChannel::Speech,
                content: "too-long".repeat(4),
                confidence: 5000,
                rationale_commit: None,
                integration_score: None,
            }]
        }
    }

    fn test_request(budget: SandboxBudget) -> AiCallRequest {
        AiCallRequest::new(1, Digest32::new([7u8; 32]), AI_MODE_THOUGHT, budget)
    }

    #[test]
    fn deny_when_ops_budget_exceeded() {
        let cf = normalize(base_frame());
        let intent = IntentSummary::new(1000, 0);
        let budget = SandboxBudget {
            ops: 0,
            max_output_chars: 256,
            max_frames: 4,
        };
        let req = test_request(budget);
        let mut sandbox = MockWasmSandbox::new(Box::new(EmptyWorker), SandboxCaps::default());

        let report = sandbox.evaluate_call(&cf, &intent, &req);

        assert!(matches!(report.verdict, SandboxVerdict::Deny { .. }));
    }

    #[test]
    fn deny_when_output_char_cap_exceeded() {
        let cf = normalize(base_frame());
        let intent = IntentSummary::new(1000, 0);
        let budget = SandboxBudget {
            ops: 20_000,
            max_output_chars: 4,
            max_frames: 4,
        };
        let req = test_request(budget);
        let mut sandbox = MockWasmSandbox::new(Box::new(LoudWorker), SandboxCaps::default());

        let report = sandbox.evaluate_call(&cf, &intent, &req);
        assert!(report.verdict.is_allow());

        let result = sandbox.run_ai(&req);
        let err = result.expect_err("expected output cap rejection");
        assert!(matches!(err.verdict, SandboxVerdict::Deny { .. }));
    }
}
