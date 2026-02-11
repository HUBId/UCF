use std::collections::BTreeMap;

use blake3::Hasher;
use ucf_frames::v1::{ChannelCode, ControlFrame, ControlPayload};
use ucf_policy::{
    adapter::ActionAdapter,
    capability::{CapabilityKind, CapabilityScope, CapabilitySet},
    gem::{
        AuthorizationOutcome, PayloadHint, ToolGate, ToolRequest, ToolResultSummary, ToolStatus,
    },
};

const MAX_OP_BYTES: usize = 64;
const MAX_MODULE_BYTES: usize = 64;
const MAX_CALL_INPUT_BYTES: usize = 64 * 1024;
const MAX_REPLY_OUTPUT_BYTES: usize = 64 * 1024;
const MAX_CAPABILITY_ITEMS: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CallId(pub u64);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapabilitySummaryItem {
    pub token_digest: [u8; 32],
    pub kind: String,
    pub scope: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct CapabilitySetSummary {
    pub items: Vec<CapabilitySummaryItem>,
}

impl CapabilitySetSummary {
    pub fn from_set(set: &CapabilitySet) -> Self {
        let mut items: Vec<_> = set
            .tokens
            .iter()
            .map(|token| CapabilitySummaryItem {
                token_digest: token.token_digest,
                kind: capability_kind_tag(&token.kind),
                scope: scope_summary(&token.scope),
            })
            .collect();
        items.sort_by(|a, b| {
            a.token_digest
                .cmp(&b.token_digest)
                .then_with(|| a.kind.cmp(&b.kind))
                .then_with(|| a.scope.cmp(&b.scope))
        });
        items.truncate(MAX_CAPABILITY_ITEMS);
        Self { items }
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        put_u32(&mut out, self.items.len() as u32);
        for item in &self.items {
            out.extend_from_slice(&item.token_digest);
            put_string(&mut out, &item.kind);
            put_string(&mut out, &item.scope);
        }
        out
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SandboxCall {
    pub call_id: CallId,
    pub t: u64,
    pub module: String,
    pub op: String,
    pub input: Vec<u8>,
    pub capabilities: CapabilitySetSummary,
    pub evidence_chain_digest: [u8; 32],
}

impl SandboxCall {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        put_u64(&mut out, self.call_id.0);
        put_u64(&mut out, self.t);
        put_string(&mut out, &self.module);
        put_string(&mut out, &self.op);
        put_bytes(&mut out, &self.input);
        out.extend_from_slice(&self.capabilities.canonical_bytes());
        out.extend_from_slice(&self.evidence_chain_digest);
        out
    }

    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(&self.canonical_bytes());
        hasher.finalize().into()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SandboxStatus {
    Ok,
    Denied,
    RateLimited,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SandboxAuditSummary {
    pub call_digest: [u8; 32],
    pub token_digest: Option<[u8; 32]>,
    pub bytes_out: u32,
    pub bytes_in: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SandboxReply {
    pub status: SandboxStatus,
    pub output: Vec<u8>,
    pub audit: SandboxAuditSummary,
    pub finished_at_t: u64,
}

impl SandboxReply {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(match self.status {
            SandboxStatus::Ok => 1,
            SandboxStatus::Denied => 2,
            SandboxStatus::RateLimited => 3,
            SandboxStatus::Failed => 4,
        });
        put_bytes(&mut out, &self.output);
        out.extend_from_slice(&self.audit.call_digest);
        put_optional_digest(&mut out, self.audit.token_digest);
        put_u32(&mut out, self.audit.bytes_out);
        put_u32(&mut out, self.audit.bytes_in);
        put_u64(&mut out, self.finished_at_t);
        out
    }

    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(&self.canonical_bytes());
        hasher.finalize().into()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SandboxBudget {
    pub work_units: u64,
    pub max_bytes_out: u32,
    pub max_bytes_in: u32,
    pub hard_timeout_ticks: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SandboxError {
    InvalidRequest(&'static str),
    BudgetExceeded(&'static str),
    BackendDisabled(&'static str),
    NotImplemented(&'static str),
}

pub trait IsolationRuntime {
    fn name(&self) -> &'static str;
    fn call(
        &mut self,
        req: SandboxCall,
        budget: SandboxBudget,
    ) -> Result<SandboxReply, SandboxError>;
}

pub struct InProcIsolationRuntime<'a, A: ActionAdapter> {
    gate: &'a mut ToolGate,
    adapter: &'a mut A,
    work_meter: u64,
    dispatch: BTreeMap<(&'static str, &'static str), DispatchHandler>,
}

type DispatchHandler = fn(&[u8], &mut dyn ActionAdapter) -> Result<(u32, u32), String>;

impl<'a, A: ActionAdapter> InProcIsolationRuntime<'a, A> {
    pub fn new(gate: &'a mut ToolGate, adapter: &'a mut A) -> Self {
        let mut dispatch = BTreeMap::new();
        dispatch.insert(
            ("tools.external", "emit_text"),
            handle_emit_text as DispatchHandler,
        );
        dispatch.insert(
            ("tools.memory", "write_bytes"),
            handle_write_bytes as DispatchHandler,
        );
        dispatch.insert(
            ("tools.brain", "emit_spike_count"),
            handle_emit_spike_count as DispatchHandler,
        );
        Self {
            gate,
            adapter,
            work_meter: 0,
            dispatch,
        }
    }

    fn enforce_request_bounds(
        req: &SandboxCall,
        budget: SandboxBudget,
    ) -> Result<(), SandboxError> {
        if req.module.len() > MAX_MODULE_BYTES {
            return Err(SandboxError::InvalidRequest("module_too_long"));
        }
        if req.op.len() > MAX_OP_BYTES {
            return Err(SandboxError::InvalidRequest("op_too_long"));
        }
        if req.input.len() > MAX_CALL_INPUT_BYTES {
            return Err(SandboxError::InvalidRequest("input_too_large"));
        }
        if req.input.len() as u32 > budget.max_bytes_in {
            return Err(SandboxError::BudgetExceeded("bytes_in"));
        }
        Ok(())
    }

    fn consume_work(&mut self, units: u64, budget: SandboxBudget) -> Result<(), SandboxError> {
        self.work_meter = self.work_meter.saturating_add(units);
        if self.work_meter > budget.work_units {
            return Err(SandboxError::BudgetExceeded("work_units"));
        }
        Ok(())
    }
}

impl<'a, A: ActionAdapter> IsolationRuntime for InProcIsolationRuntime<'a, A> {
    fn name(&self) -> &'static str {
        "in-proc"
    }

    fn call(
        &mut self,
        req: SandboxCall,
        budget: SandboxBudget,
    ) -> Result<SandboxReply, SandboxError> {
        Self::enforce_request_bounds(&req, budget)?;
        self.consume_work(1, budget)?;
        if budget.hard_timeout_ticks == 0 {
            return Err(SandboxError::BudgetExceeded("hard_timeout_ticks"));
        }

        let tool_request = decode_tool_request(&req)?;
        let auth = self.gate.authorize(&tool_request, req.t);
        let call_digest = req.digest();

        let (status, token_digest, output, error_code) = match auth {
            AuthorizationOutcome::Allowed { token_digest } => {
                let handler = self
                    .dispatch
                    .get(&(req.module.as_str(), req.op.as_str()))
                    .ok_or(SandboxError::InvalidRequest("unknown_dispatch"))?;
                match handler(&req.input, self.adapter as &mut dyn ActionAdapter) {
                    Ok(_) => (SandboxStatus::Ok, Some(token_digest), Vec::new(), None),
                    Err(code) => (
                        SandboxStatus::Failed,
                        Some(token_digest),
                        code.as_bytes().to_vec(),
                        Some(code),
                    ),
                }
            }
            AuthorizationOutcome::Denied { reason } => (
                SandboxStatus::Denied,
                None,
                format!("{reason:?}").into_bytes(),
                None,
            ),
            AuthorizationOutcome::RateLimited { retry_after_ticks } => (
                SandboxStatus::RateLimited,
                None,
                format!("retry_after:{retry_after_ticks}").into_bytes(),
                None,
            ),
        };

        self.consume_work(1, budget)?;
        if output.len() > MAX_REPLY_OUTPUT_BYTES || output.len() as u32 > budget.max_bytes_out {
            return Err(SandboxError::BudgetExceeded("bytes_out"));
        }

        let output = if let Some(code) = error_code {
            code.into_bytes()
        } else {
            output
        };
        let reply = SandboxReply {
            status,
            output,
            audit: SandboxAuditSummary {
                call_digest,
                token_digest,
                bytes_out: tool_request.payload_hint.bytes_out.unwrap_or(0),
                bytes_in: tool_request.payload_hint.bytes_in.unwrap_or(0),
            },
            finished_at_t: req.t,
        };
        Ok(reply)
    }
}

#[cfg(feature = "sandbox-wasm")]
pub struct WasmIsolationRuntime;

#[cfg(feature = "sandbox-wasm")]
impl IsolationRuntime for WasmIsolationRuntime {
    fn name(&self) -> &'static str {
        "wasm"
    }

    fn call(
        &mut self,
        _req: SandboxCall,
        _budget: SandboxBudget,
    ) -> Result<SandboxReply, SandboxError> {
        Err(SandboxError::BackendDisabled(
            "sandbox-wasm disabled at runtime",
        ))
    }
}

#[cfg(feature = "sandbox-proc")]
pub struct ProcessIsolationRuntime;

#[cfg(feature = "sandbox-proc")]
impl IsolationRuntime for ProcessIsolationRuntime {
    fn name(&self) -> &'static str {
        "process"
    }

    fn call(
        &mut self,
        _req: SandboxCall,
        _budget: SandboxBudget,
    ) -> Result<SandboxReply, SandboxError> {
        Err(SandboxError::NotImplemented("process runtime is a stub"))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SandboxToolExecution {
    pub request: ToolRequest,
    pub auth: AuthorizationOutcome,
    pub result: ToolResultSummary,
    pub call_digest: [u8; 32],
    pub reply_digest: [u8; 32],
    pub capability_summary: CapabilitySetSummary,
}

pub fn execute_tool_call<A: ActionAdapter>(
    adapter: &mut A,
    gate: &mut ToolGate,
    request: ToolRequest,
    module: String,
    op: String,
    input: Vec<u8>,
    capability_summary: CapabilitySetSummary,
) -> Result<SandboxToolExecution, SandboxError> {
    let call = SandboxCall {
        call_id: CallId(request.id),
        t: request.requested_at_t,
        module,
        op,
        input,
        capabilities: capability_summary.clone(),
        evidence_chain_digest: request.evidence_chain_digest,
    };
    let mut runtime = InProcIsolationRuntime::new(gate, adapter);
    let reply = runtime.call(
        call,
        SandboxBudget {
            work_units: 4,
            max_bytes_out: 4096,
            max_bytes_in: 65536,
            hard_timeout_ticks: 1,
        },
    )?;
    let auth = match reply.audit.token_digest {
        Some(token_digest) => AuthorizationOutcome::Allowed { token_digest },
        None => {
            if matches!(reply.status, SandboxStatus::RateLimited) {
                AuthorizationOutcome::RateLimited {
                    retry_after_ticks: 1,
                }
            } else {
                AuthorizationOutcome::Denied {
                    reason: ucf_policy::capability::CapabilityDenyReason::MissingToken,
                }
            }
        }
    };
    let result = ToolResultSummary {
        status: match reply.status {
            SandboxStatus::Ok => ToolStatus::AllowedExecuted,
            SandboxStatus::Denied => ToolStatus::Denied,
            SandboxStatus::RateLimited => ToolStatus::RateLimited,
            SandboxStatus::Failed => ToolStatus::Failed,
        },
        bytes_out: Some(reply.audit.bytes_out),
        bytes_in: Some(reply.audit.bytes_in),
        error_code: if reply.output.is_empty() {
            None
        } else {
            Some(String::from_utf8_lossy(&reply.output).to_string())
        },
        finished_at_t: reply.finished_at_t,
    };
    Ok(SandboxToolExecution {
        request,
        auth,
        result,
        call_digest: reply.audit.call_digest,
        reply_digest: reply.digest(),
        capability_summary,
    })
}

pub fn call_spec_from_control(
    ctrl: &ControlFrame,
) -> Result<(String, String, Vec<u8>), SandboxError> {
    match (&ctrl.channel, &ctrl.payload) {
        (ChannelCode::ExternalOutput, ControlPayload::Text(text)) => Ok((
            "tools.external".to_string(),
            "emit_text".to_string(),
            text.as_bytes().to_vec(),
        )),
        (ChannelCode::MemoryWrite, ControlPayload::Bytes(bytes)) => Ok((
            "tools.memory".to_string(),
            "write_bytes".to_string(),
            bytes.to_vec(),
        )),
        (ChannelCode::BrainStimulus, ControlPayload::BrainStimulus(_)) => Ok((
            "tools.brain".to_string(),
            "emit_spike_count".to_string(),
            vec![4],
        )),
        _ => Err(SandboxError::InvalidRequest("unsupported_control_payload")),
    }
}

fn decode_tool_request(call: &SandboxCall) -> Result<ToolRequest, SandboxError> {
    let (kind, target, payload_hint) = match (call.module.as_str(), call.op.as_str()) {
        ("tools.external", "emit_text") => (
            CapabilityKind::ExternalApi,
            "external_output".to_string(),
            PayloadHint {
                bytes_out: Some(call.input.len() as u32),
                bytes_in: None,
            },
        ),
        ("tools.memory", "write_bytes") => (
            CapabilityKind::FileWrite,
            "memory_write".to_string(),
            PayloadHint {
                bytes_out: Some(call.input.len() as u32),
                bytes_in: None,
            },
        ),
        ("tools.brain", "emit_spike_count") => (
            CapabilityKind::UiAutomation,
            "brain_target".to_string(),
            PayloadHint {
                bytes_out: call.input.first().map(|v| *v as u32),
                bytes_in: None,
            },
        ),
        _ => return Err(SandboxError::InvalidRequest("module_op")),
    };

    Ok(ToolRequest {
        id: call.call_id.0,
        kind,
        target,
        payload_hint,
        requested_at_t: call.t,
        decision_id: call.call_id.0,
        evidence_chain_digest: call.evidence_chain_digest,
    })
}

fn handle_emit_text(input: &[u8], adapter: &mut dyn ActionAdapter) -> Result<(u32, u32), String> {
    let text = String::from_utf8(input.to_vec()).map_err(|_| "invalid_utf8".to_string())?;
    adapter.emit_text(&text).map_err(|e| e.to_string())?;
    Ok((text.len() as u32, 0))
}

fn handle_write_bytes(input: &[u8], adapter: &mut dyn ActionAdapter) -> Result<(u32, u32), String> {
    adapter.write_memory(input).map_err(|e| e.to_string())?;
    Ok((input.len() as u32, 0))
}

fn handle_emit_spike_count(
    input: &[u8],
    adapter: &mut dyn ActionAdapter,
) -> Result<(u32, u32), String> {
    let count = input.first().copied().unwrap_or_default() as usize;
    let _ = count;
    adapter
        .emit_brain_spikes(Vec::new())
        .map_err(|e| e.to_string())?;
    Ok((count as u32, 0))
}

fn capability_kind_tag(kind: &CapabilityKind) -> String {
    match kind {
        CapabilityKind::NetHttp => "net_http".to_string(),
        CapabilityKind::FileRead => "file_read".to_string(),
        CapabilityKind::FileWrite => "file_write".to_string(),
        CapabilityKind::ProcessExec => "process_exec".to_string(),
        CapabilityKind::ClipboardRead => "clipboard_read".to_string(),
        CapabilityKind::ClipboardWrite => "clipboard_write".to_string(),
        CapabilityKind::ExternalApi => "external_api".to_string(),
        CapabilityKind::UiAutomation => "ui_automation".to_string(),
        CapabilityKind::Custom(tag) => format!("custom:{tag}"),
    }
}

fn scope_summary(scope: &CapabilityScope) -> String {
    match scope {
        CapabilityScope::All => "all".to_string(),
        CapabilityScope::Paths(paths) => format!("paths:{}", paths.join(",")),
        CapabilityScope::Domains(domains) => format!("domains:{}", domains.join(",")),
        CapabilityScope::ApiNames(api_names) => format!("api_names:{}", api_names.join(",")),
    }
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn put_string(out: &mut Vec<u8>, value: &str) {
    put_u32(out, value.len() as u32);
    out.extend_from_slice(value.as_bytes());
}

fn put_bytes(out: &mut Vec<u8>, value: &[u8]) {
    put_u32(out, value.len() as u32);
    out.extend_from_slice(value);
}

fn put_optional_digest(out: &mut Vec<u8>, value: Option<[u8; 32]>) {
    match value {
        Some(digest) => {
            out.push(1);
            out.extend_from_slice(&digest);
        }
        None => out.push(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_policy::{
        capability::{CapabilityLimits, CapabilityScope, CapabilityToken},
        rate_limiter::RateLimiter,
    };

    #[test]
    fn canonical_encoding_is_stable() {
        let call = SandboxCall {
            call_id: CallId(7),
            t: 9,
            module: "tools.external".to_string(),
            op: "emit_text".to_string(),
            input: b"hi".to_vec(),
            capabilities: CapabilitySetSummary::default(),
            evidence_chain_digest: [5; 32],
        };
        assert_eq!(call.digest(), call.digest());
    }

    #[test]
    fn denied_call_returns_denied_status() {
        let mut gate = ToolGate::new(CapabilitySet::empty(), RateLimiter::new(10));
        let mut adapter = ucf_policy::adapter::MockAdapter::default();
        let mut runtime = InProcIsolationRuntime::new(&mut gate, &mut adapter);
        let req = SandboxCall {
            call_id: CallId(4),
            t: 2,
            module: "tools.external".into(),
            op: "emit_text".into(),
            input: b"abc".to_vec(),
            capabilities: CapabilitySetSummary::default(),
            evidence_chain_digest: [0; 32],
        };

        let reply = runtime
            .call(
                req,
                SandboxBudget {
                    work_units: 4,
                    max_bytes_out: 128,
                    max_bytes_in: 128,
                    hard_timeout_ticks: 1,
                },
            )
            .expect("sandbox reply");
        assert_eq!(reply.status, SandboxStatus::Denied);
        assert!(reply.audit.token_digest.is_none());
    }

    #[test]
    fn deterministic_budget_enforced() {
        let token = CapabilityToken::issue(
            CapabilityKind::ExternalApi,
            CapabilityScope::ApiNames(vec!["external_output".to_string()]),
            CapabilityLimits {
                max_calls_per_window: 8,
                window_ticks: 20,
                max_bytes_out: Some(1024),
                max_bytes_in: None,
                max_concurrent: 1,
            },
            "issuer",
            1,
            Some(10),
        );
        let mut gate = ToolGate::new(
            CapabilitySet {
                tokens: vec![token],
            },
            RateLimiter::new(10),
        );
        let mut adapter = ucf_policy::adapter::MockAdapter::default();
        let mut runtime = InProcIsolationRuntime::new(&mut gate, &mut adapter);
        let req = SandboxCall {
            call_id: CallId(1),
            t: 1,
            module: "tools.external".into(),
            op: "emit_text".into(),
            input: b"abc".to_vec(),
            capabilities: CapabilitySetSummary::default(),
            evidence_chain_digest: [0; 32],
        };
        let err = runtime
            .call(
                req,
                SandboxBudget {
                    work_units: 1,
                    max_bytes_out: 10,
                    max_bytes_in: 10,
                    hard_timeout_ticks: 1,
                },
            )
            .expect_err("budget should fail");
        assert!(matches!(err, SandboxError::BudgetExceeded("work_units")));
    }
}
