use std::collections::BTreeMap;

use blake3::Hasher;
use ucf_bluebrain_bridge::BrainStimulusEncoder;
use ucf_brainbus::v0::Spike;
use ucf_core::types::{SimTime, Tick, WindowId};
use ucf_frames::v1::{ChannelCode, ControlFrame, ControlPayload, CorrelationId, DecisionCode};

#[cfg(feature = "sandbox-wasm")]
use wasmtime::{Caller, Config, Engine, Extern, Instance, Linker, Memory, Module, Store};

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
#[cfg(feature = "sandbox-wasm")]
const WASM_SCHEMA_VERSION: u16 = 1;
#[cfg(feature = "sandbox-wasm")]
const MAX_HOSTCALLS: u32 = 32;
#[cfg(feature = "sandbox-wasm")]
const MAX_WASM_MEMORY_PAGES: u64 = 2;
#[cfg(feature = "sandbox-wasm")]
const MAX_WASM_REPLY_BYTES: usize = 64 * 1024;

#[cfg(feature = "sandbox-wasm")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmCallEnvelope {
    pub schema_version: u16,
    pub module: String,
    pub op: String,
    pub payload: Vec<u8>,
    pub capability_set_summary: CapabilitySetSummary,
    pub evidence_chain_digest: [u8; 32],
    pub t: u64,
}

#[cfg(feature = "sandbox-wasm")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmReplyEnvelope {
    pub schema_version: u16,
    pub status: SandboxStatus,
    pub payload: Vec<u8>,
    pub audit: Option<SandboxAuditSummary>,
    pub finished_at_t: u64,
}

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
    ExecutionFailed(&'static str),
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

type DispatchHandler = fn(&SandboxCall, &mut dyn ActionAdapter) -> Result<(), String>;

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
            ("tools.brain", "emit_spikes"),
            handle_emit_spikes as DispatchHandler,
        );
        dispatch.insert(("tools.none", "noop"), handle_noop as DispatchHandler);
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

        let (status, token_digest, output) = match auth {
            AuthorizationOutcome::Allowed { token_digest } => {
                let handler = self
                    .dispatch
                    .get(&(req.module.as_str(), req.op.as_str()))
                    .ok_or(SandboxError::InvalidRequest("unknown_dispatch"))?;
                match handler(&req, self.adapter as &mut dyn ActionAdapter) {
                    Ok(()) => (SandboxStatus::Ok, Some(token_digest), Vec::new()),
                    Err(code) => (SandboxStatus::Failed, Some(token_digest), code.into_bytes()),
                }
            }
            AuthorizationOutcome::Denied { reason } => (
                SandboxStatus::Denied,
                None,
                format!("{reason:?}").into_bytes(),
            ),
            AuthorizationOutcome::RateLimited { retry_after_ticks } => (
                SandboxStatus::RateLimited,
                None,
                format!("retry_after:{retry_after_ticks}").into_bytes(),
            ),
        };

        self.consume_work(1, budget)?;
        if output.len() > MAX_REPLY_OUTPUT_BYTES || output.len() as u32 > budget.max_bytes_out {
            return Err(SandboxError::BudgetExceeded("bytes_out"));
        }

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
struct WasmHostState {
    call: SandboxCall,
    hostcalls: u32,
    tool_request_count: u32,
    tool_result: Vec<u8>,
}

#[cfg(feature = "sandbox-wasm")]
pub struct WasmIsolationRuntime {
    _gate: ToolGate,
    engine: Engine,
    modules: BTreeMap<String, Module>,
}

#[cfg(feature = "sandbox-wasm")]
impl WasmIsolationRuntime {
    pub fn new(gate: ToolGate) -> Result<Self, SandboxError> {
        let mut cfg = Config::new();
        cfg.consume_fuel(true);
        cfg.epoch_interruption(false);
        let engine = Engine::new(&cfg).map_err(|_| SandboxError::BackendDisabled("engine_init"))?;
        let mut modules = BTreeMap::new();
        let echo = Module::new(&engine, wasm_echo_bytes())
            .map_err(|_| SandboxError::BackendDisabled("echo_module"))?;
        let probe = Module::new(&engine, wasm_tool_probe_bytes())
            .map_err(|_| SandboxError::BackendDisabled("probe_module"))?;
        modules.insert("wasm.echo".to_string(), echo);
        modules.insert("wasm.tool_probe".to_string(), probe);
        Ok(Self {
            _gate: gate,
            engine,
            modules,
        })
    }

    fn map_tool_kind(kind: u32) -> Option<CapabilityKind> {
        match kind {
            1 => Some(CapabilityKind::FileRead),
            2 => Some(CapabilityKind::NetHttp),
            _ => None,
        }
    }

    fn decode_ptr_len(ret: i64) -> (u32, u32) {
        let ptr = (ret & 0xffff_ffff) as u32;
        let len = ((ret >> 32) & 0xffff_ffff) as u32;
        (ptr, len)
    }
}

#[cfg(feature = "sandbox-wasm")]
impl IsolationRuntime for WasmIsolationRuntime {
    fn name(&self) -> &'static str {
        "wasm"
    }

    fn call(
        &mut self,
        req: SandboxCall,
        budget: SandboxBudget,
    ) -> Result<SandboxReply, SandboxError> {
        InProcIsolationRuntime::<ucf_policy::adapter::MockAdapter>::enforce_request_bounds(
            &req, budget,
        )?;
        if budget.hard_timeout_ticks == 0 {
            return Err(SandboxError::BudgetExceeded("hard_timeout_ticks"));
        }
        let module = self
            .modules
            .get(&req.module)
            .ok_or(SandboxError::InvalidRequest("unknown_wasm_module"))?
            .clone();

        let env = WasmCallEnvelope {
            schema_version: WASM_SCHEMA_VERSION,
            module: req.module.clone(),
            op: req.op.clone(),
            payload: req.input.clone(),
            capability_set_summary: req.capabilities.clone(),
            evidence_chain_digest: req.evidence_chain_digest,
            t: req.t,
        };
        let input_bytes = encode_wasm_call_envelope(&env)?;

        let mut linker: Linker<WasmHostState> = Linker::new(&self.engine);
        linker
            .func_wrap(
                "host",
                "host_log",
                |_caller: Caller<'_, WasmHostState>, _level: i32, _ptr: i32, _len: i32| {},
            )
            .map_err(|_| SandboxError::BackendDisabled("host_log"))?;

        linker
            .func_wrap(
                "host",
                "host_tool_request",
                |mut caller: Caller<'_, WasmHostState>,
                 kind: i32,
                 target_ptr: i32,
                 target_len: i32,
                 _payload_hint: i32|
                 -> i32 {
                    {
                        let state = caller.data_mut();
                        state.hostcalls = state.hostcalls.saturating_add(1);
                        if state.hostcalls > MAX_HOSTCALLS {
                            state.tool_result = b"hostcall_limit".to_vec();
                            return 2;
                        }
                        state.tool_request_count = state.tool_request_count.saturating_add(1);
                        if state.tool_request_count > 1 {
                            state.tool_result = b"single_tool_request_only".to_vec();
                            return 2;
                        }
                    }
                    let Some(kind) = Self::map_tool_kind(kind as u32) else {
                        caller.data_mut().tool_result = b"unknown_kind".to_vec();
                        return 2;
                    };
                    let Some(Extern::Memory(mem)) = caller.get_export("memory") else {
                        caller.data_mut().tool_result = b"missing_memory".to_vec();
                        return 2;
                    };
                    let mut target = vec![0u8; target_len.max(0) as usize];
                    if mem
                        .read(&caller, target_ptr.max(0) as usize, &mut target)
                        .is_err()
                    {
                        caller.data_mut().tool_result = b"invalid_target".to_vec();
                        return 2;
                    }
                    let target = String::from_utf8_lossy(&target).to_string();
                    let _request = ToolRequest {
                        id: caller.data().call.call_id.0,
                        kind,
                        target,
                        payload_hint: PayloadHint::default(),
                        requested_at_t: caller.data().call.t,
                        decision_id: caller.data().call.call_id.0,
                        evidence_chain_digest: caller.data().call.evidence_chain_digest,
                    };
                    caller.data_mut().tool_result = b"denied_by_default".to_vec();
                    2
                },
            )
            .map_err(|_| SandboxError::BackendDisabled("host_tool_request"))?;

        let mut store = Store::new(
            &self.engine,
            WasmHostState {
                call: req.clone(),
                hostcalls: 0,
                tool_request_count: 0,
                tool_result: Vec::new(),
            },
        );
        store
            .set_fuel(budget.work_units)
            .map_err(|_| SandboxError::BudgetExceeded("fuel"))?;

        let instance = linker
            .instantiate(&mut store, &module)
            .map_err(|_| SandboxError::ExecutionFailed("instantiate"))?;
        let memory = get_memory(&mut store, &instance)?;
        let alloc = instance
            .get_typed_func::<i32, i32>(&mut store, "alloc")
            .map_err(|_| SandboxError::InvalidRequest("missing_alloc"))?;
        let entry = instance
            .get_typed_func::<(i32, i32), i64>(&mut store, "sandbox_call")
            .map_err(|_| SandboxError::InvalidRequest("missing_entry"))?;

        let in_ptr = alloc
            .call(&mut store, input_bytes.len() as i32)
            .map_err(|_| SandboxError::ExecutionFailed("alloc_input"))?;
        memory
            .write(&mut store, in_ptr as usize, &input_bytes)
            .map_err(|_| SandboxError::ExecutionFailed("write_input"))?;

        let ret = entry
            .call(&mut store, (in_ptr, input_bytes.len() as i32))
            .map_err(|_| SandboxError::BudgetExceeded("fuel"))?;
        let (out_ptr, out_len) = Self::decode_ptr_len(ret);
        if out_len as usize > MAX_WASM_REPLY_BYTES || out_len > budget.max_bytes_out {
            return Err(SandboxError::BudgetExceeded("bytes_out"));
        }
        let mut out = vec![0u8; out_len as usize];
        memory
            .read(&store, out_ptr as usize, &mut out)
            .map_err(|_| SandboxError::ExecutionFailed("read_output"))?;

        let (status, output) = if req.module == "wasm.tool_probe" {
            let code = out.first().copied().unwrap_or(2);
            let status = if code == 1 {
                SandboxStatus::Ok
            } else {
                SandboxStatus::Denied
            };
            (status, out)
        } else {
            let reply = decode_wasm_reply_envelope(&out).unwrap_or(WasmReplyEnvelope {
                schema_version: WASM_SCHEMA_VERSION,
                status: SandboxStatus::Ok,
                payload: req.input.clone(),
                audit: None,
                finished_at_t: req.t,
            });
            (reply.status, reply.payload)
        };
        Ok(SandboxReply {
            status,
            output,
            audit: SandboxAuditSummary {
                call_digest: req.digest(),
                token_digest: None,
                bytes_out: req.input.len() as u32,
                bytes_in: 0,
            },
            finished_at_t: req.t,
        })
    }
}

#[cfg(feature = "sandbox-wasm")]
fn get_memory(
    store: &mut Store<WasmHostState>,
    instance: &Instance,
) -> Result<Memory, SandboxError> {
    match instance.get_export(&mut *store, "memory") {
        Some(Extern::Memory(mem)) => {
            if mem.size(store) > MAX_WASM_MEMORY_PAGES {
                return Err(SandboxError::BudgetExceeded("memory_pages"));
            }
            Ok(mem)
        }
        _ => Err(SandboxError::InvalidRequest("missing_memory")),
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
    pub module: String,
    pub op: String,
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
        module: module.clone(),
        op: op.clone(),
        input,
        capabilities: capability_summary.clone(),
        evidence_chain_digest: request.evidence_chain_digest,
    };
    let budget = SandboxBudget {
        work_units: 4,
        max_bytes_out: 4096,
        max_bytes_in: 65536,
        hard_timeout_ticks: 1,
    };
    let runtime_choice =
        std::env::var("UCF_ISOLATION_RUNTIME").unwrap_or_else(|_| "inproc".to_string());
    let reply = if runtime_choice == "wasm" {
        #[cfg(feature = "sandbox-wasm")]
        {
            let wasm_gate = ToolGate::new(
                gate.capabilities.clone(),
                ucf_policy::rate_limiter::RateLimiter::new(32),
            );
            let mut runtime = WasmIsolationRuntime::new(wasm_gate)?;
            runtime.call(call, budget)?
        }
        #[cfg(not(feature = "sandbox-wasm"))]
        {
            return Err(SandboxError::BackendDisabled(
                "sandbox-wasm feature not enabled",
            ));
        }
    } else {
        let mut runtime = InProcIsolationRuntime::new(gate, adapter);
        runtime.call(call, budget)?
    };
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
        module,
        op,
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
        (ChannelCode::BrainStimulus, ControlPayload::BrainStimulus(payload)) => Ok((
            "tools.brain".to_string(),
            "emit_spikes".to_string(),
            encode_spike_meta(&BrainStimulusEncoder::encode_to_spikes(ctrl, payload)),
        )),
        (ChannelCode::InternalThought, _) => {
            Ok(("tools.none".to_string(), "noop".to_string(), Vec::new()))
        }
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
        ("tools.brain", "emit_spikes") => (
            CapabilityKind::UiAutomation,
            "brain_target".to_string(),
            PayloadHint {
                bytes_out: Some(decode_spike_meta(&call.input).map(|(n, _)| n).unwrap_or(0)),
                bytes_in: None,
            },
        ),
        ("tools.none", "noop") => (
            CapabilityKind::Custom("internal_thought".to_string()),
            "internal".to_string(),
            PayloadHint::default(),
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

fn handle_emit_text(call: &SandboxCall, adapter: &mut dyn ActionAdapter) -> Result<(), String> {
    let text = String::from_utf8(call.input.clone()).map_err(|_| "invalid_utf8".to_string())?;
    adapter.emit_text(&text).map_err(|e| e.to_string())
}

fn handle_write_bytes(call: &SandboxCall, adapter: &mut dyn ActionAdapter) -> Result<(), String> {
    adapter.write_memory(&call.input).map_err(|e| e.to_string())
}

fn handle_emit_spikes(call: &SandboxCall, adapter: &mut dyn ActionAdapter) -> Result<(), String> {
    let (count, dst) = decode_spike_meta(&call.input)?;
    let mut spikes = Vec::with_capacity(count as usize);
    for idx in 0..count {
        spikes.push(Spike::new(
            SimTime {
                tick: Tick::new(call.t),
                window: WindowId::new(0),
            },
            CorrelationId(call.call_id.0),
            idx as u16,
            dst,
            0,
        ));
    }
    adapter.emit_brain_spikes(spikes).map_err(|e| e.to_string())
}

fn handle_noop(_call: &SandboxCall, _adapter: &mut dyn ActionAdapter) -> Result<(), String> {
    Ok(())
}

fn encode_spike_meta(spikes: &[Spike]) -> Vec<u8> {
    let count = spikes.len().min(u16::MAX as usize) as u16;
    let dst = spikes.first().map(|s| s.dst).unwrap_or(0);
    let mut out = Vec::with_capacity(4);
    out.extend_from_slice(&count.to_be_bytes());
    out.extend_from_slice(&dst.to_be_bytes());
    out
}

fn decode_spike_meta(input: &[u8]) -> Result<(u32, u16), String> {
    if input.len() != 4 {
        return Err("invalid_spike_meta".to_string());
    }
    let count = u16::from_be_bytes([input[0], input[1]]) as u32;
    let dst = u16::from_be_bytes([input[2], input[3]]);
    Ok((count, dst))
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

pub fn decision_allows_tool(decision: DecisionCode) -> bool {
    matches!(decision, DecisionCode::Allow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_core::types::{SimTime, Tick, WindowId};
    use ucf_frames::v1::{
        BrainStimulusKind, BrainStimulusPayload, CorrelationId, Intent, IntentId, IntentKind,
    };
    use ucf_policy::capability::{CapabilityLimits, CapabilityScope, CapabilityToken};

    fn sim_time() -> SimTime {
        SimTime {
            tick: Tick::new(1),
            window: WindowId::new(0),
        }
    }
    fn intent() -> Intent {
        Intent::new(IntentId(1), IntentKind::System, "sandbox-test")
    }

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
    fn brain_call_spec_keeps_spike_count() {
        let ctrl = ControlFrame {
            time: sim_time(),
            corr: CorrelationId(1),
            channel: ChannelCode::BrainStimulus,
            intent: intent(),
            payload: ControlPayload::BrainStimulus(BrainStimulusPayload {
                kind: BrainStimulusKind::SpikeTrain,
                target: 44,
                intensity: 255,
                duration_ms: 90,
            }),
        };
        let (_, _, input) = call_spec_from_control(&ctrl).expect("spec");
        assert_eq!(input.len(), 4);
        assert_eq!(u16::from_be_bytes([input[0], input[1]]), 8);
    }

    #[test]
    fn denied_call_returns_denied_status() {
        let mut gate = ToolGate::new(
            CapabilitySet::empty(),
            ucf_policy::rate_limiter::RateLimiter::new(10),
        );
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
            ucf_policy::rate_limiter::RateLimiter::new(10),
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

#[cfg(feature = "sandbox-wasm")]
fn encode_wasm_call_envelope(env: &WasmCallEnvelope) -> Result<Vec<u8>, SandboxError> {
    if env.op.len() > MAX_OP_BYTES || env.module.len() > MAX_MODULE_BYTES {
        return Err(SandboxError::InvalidRequest("wasm_envelope_bounds"));
    }
    let mut out = Vec::new();
    out.extend_from_slice(&env.schema_version.to_be_bytes());
    put_string(&mut out, &env.module);
    put_string(&mut out, &env.op);
    put_bytes(&mut out, &env.payload);
    out.extend_from_slice(&env.capability_set_summary.canonical_bytes());
    out.extend_from_slice(&env.evidence_chain_digest);
    put_u64(&mut out, env.t);
    Ok(out)
}

#[cfg(feature = "sandbox-wasm")]
fn decode_wasm_reply_envelope(bytes: &[u8]) -> Result<WasmReplyEnvelope, SandboxError> {
    if bytes.len() < 3 {
        return Err(SandboxError::InvalidRequest("wasm_reply_short"));
    }
    let schema_version = u16::from_be_bytes([bytes[0], bytes[1]]);
    let status = match bytes[2] {
        1 => SandboxStatus::Ok,
        2 => SandboxStatus::Denied,
        3 => SandboxStatus::RateLimited,
        _ => SandboxStatus::Failed,
    };
    Ok(WasmReplyEnvelope {
        schema_version,
        status,
        payload: bytes[3..].to_vec(),
        audit: None,
        finished_at_t: 0,
    })
}

#[cfg(feature = "sandbox-wasm")]
fn wasm_echo_bytes() -> Vec<u8> {
    wat::parse_str(
        r#"(module
            (memory (export "memory") 1 2)
            (func (export "alloc") (param $len i32) (result i32)
                (i32.const 0))
            (func (export "sandbox_call") (param $ptr i32) (param $len i32) (result i64)
                local.get $len
                i64.extend_i32_u
                i64.const 32
                i64.shl
                local.get $ptr
                i64.extend_i32_u
                i64.or))"#,
    )
    .expect("valid wat")
}

#[cfg(feature = "sandbox-wasm")]
fn wasm_tool_probe_bytes() -> Vec<u8> {
    wat::parse_str(
        r#"(module
            (import "host" "host_tool_request" (func $host_tool_request (param i32 i32 i32 i32) (result i32)))
            (memory (export "memory") 1 2)
            (data (i32.const 16) "/tmp/probe")
            (func (export "alloc") (param $len i32) (result i32)
                (i32.const 0))
            (func (export "sandbox_call") (param $ptr i32) (param $len i32) (result i64)
                (local $status i32)
                i32.const 1
                i32.const 16
                i32.const 10
                i32.const 0
                call $host_tool_request
                local.set $status
                i32.const 0
                local.get $status
                i32.store8
                i32.const 4
                i64.extend_i32_u
                i64.const 32
                i64.shl
                i32.const 0
                i64.extend_i32_u
                i64.or))"#,
    )
    .expect("valid wat")
}

#[cfg(feature = "sandbox-wasm")]
#[test]
fn wasm_echo_is_deterministic() {
    let gate = ToolGate::new(
        CapabilitySet::empty(),
        ucf_policy::rate_limiter::RateLimiter::new(10),
    );
    let mut rt = WasmIsolationRuntime::new(gate).expect("wasm runtime");
    let req = SandboxCall {
        call_id: CallId(9),
        t: 1,
        module: "wasm.echo".into(),
        op: "echo".into(),
        input: b"ping".to_vec(),
        capabilities: CapabilitySetSummary::default(),
        evidence_chain_digest: [2; 32],
    };
    let budget = SandboxBudget {
        work_units: 100_000,
        max_bytes_out: 4096,
        max_bytes_in: 4096,
        hard_timeout_ticks: 1,
    };
    let a = rt.call(req.clone(), budget).expect("first");
    let b = rt.call(req, budget).expect("second");
    assert_eq!(a.digest(), b.digest());
}

#[cfg(feature = "sandbox-wasm")]
#[test]
fn wasm_tool_probe_denied_by_default() {
    let gate = ToolGate::new(
        CapabilitySet::empty(),
        ucf_policy::rate_limiter::RateLimiter::new(10),
    );
    let mut rt = WasmIsolationRuntime::new(gate).expect("wasm runtime");
    let req = SandboxCall {
        call_id: CallId(10),
        t: 1,
        module: "wasm.tool_probe".into(),
        op: "probe".into(),
        input: vec![],
        capabilities: CapabilitySetSummary::default(),
        evidence_chain_digest: [3; 32],
    };
    let reply = rt
        .call(
            req,
            SandboxBudget {
                work_units: 100_000,
                max_bytes_out: 4096,
                max_bytes_in: 4096,
                hard_timeout_ticks: 1,
            },
        )
        .expect("reply");
    assert_eq!(reply.status, SandboxStatus::Denied);
}
