use blake3::Hasher;
use std::io::{Read, Write};
use ucf_policy::gem::{PayloadHint, ToolRequest};
use ucf_runtime::sandbox::{
    CallId, CapabilitySetSummary, SandboxAuditSummary, SandboxCall, SandboxReply, SandboxStatus,
};

const SCHEMA_VERSION: u16 = 1;
const MAX_MSG_BYTES: usize = 128 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IpcKind {
    Call = 1,
    Reply = 2,
    ToolRequest = 3,
    ToolReply = 4,
    Heartbeat = 5,
    Shutdown = 6,
}

#[derive(Clone, Debug)]
struct IpcEnvelope {
    schema_version: u16,
    msg_id: u64,
    kind: IpcKind,
    payload: Vec<u8>,
    payload_digest: [u8; 32],
}

fn main() {
    let mut stdin = std::io::stdin().lock();
    let mut stdout = std::io::stdout().lock();
    let hb = encode_heartbeat("ucf-sandbox-worker-v0");
    let env = IpcEnvelope::new(0, IpcKind::Heartbeat, hb);
    if write_frame(&mut stdout, &encode_envelope(&env)).is_err() {
        return;
    }
    while let Ok(bytes) = read_frame(&mut stdin) {
        let Ok(env) = decode_envelope(&bytes) else {
            break;
        };
        match env.kind {
            IpcKind::Call => {
                let Ok(call) = decode_sandbox_call(&env.payload) else {
                    break;
                };
                let reply = execute_call(call, env.msg_id, &mut stdout, &mut stdin);
                let rep_env =
                    IpcEnvelope::new(env.msg_id, IpcKind::Reply, encode_sandbox_reply(&reply));
                if write_frame(&mut stdout, &encode_envelope(&rep_env)).is_err() {
                    break;
                }
            }
            IpcKind::Shutdown => break,
            _ => {}
        }
    }
}

fn execute_call(
    call: SandboxCall,
    msg_id: u64,
    stdout: &mut dyn Write,
    stdin: &mut dyn Read,
) -> SandboxReply {
    if call.module == "tools.none" && call.op == "noop" {
        return SandboxReply {
            status: SandboxStatus::Ok,
            output: Vec::new(),
            audit: SandboxAuditSummary {
                call_digest: call.digest(),
                token_digest: None,
                bytes_out: 0,
                bytes_in: call.input.len() as u32,
            },
            finished_at_t: call.t,
        };
    }

    let request = ToolRequest {
        id: call.call_id.0,
        kind: map_kind(&call.module, &call.op),
        target: map_target(&call.module, &call.op),
        payload_hint: PayloadHint {
            bytes_out: Some(call.input.len() as u32),
            bytes_in: None,
        },
        requested_at_t: call.t,
        decision_id: call.call_id.0,
        evidence_chain_digest: call.evidence_chain_digest,
        candidate_id: None,
        tool_intent_digest: None,
    };
    let treq = encode_tool_request(&request);
    let req_env = IpcEnvelope::new(msg_id, IpcKind::ToolRequest, treq);
    if write_frame(stdout, &encode_envelope(&req_env)).is_err() {
        return failed(&call, "IPC_WRITE");
    }

    let Ok(reply_frame) = read_frame(stdin) else {
        return failed(&call, "IPC_READ");
    };
    let Ok(reply_env) = decode_envelope(&reply_frame) else {
        return failed(&call, "IPC_DECODE");
    };
    if reply_env.kind != IpcKind::ToolReply {
        return failed(&call, "IPC_KIND");
    }
    let Ok(tool_reply) = decode_tool_reply(&reply_env.payload) else {
        return failed(&call, "TOOL_REPLY");
    };
    SandboxReply {
        status: match tool_reply.status {
            1 => SandboxStatus::Ok,
            2 => SandboxStatus::Denied,
            3 => SandboxStatus::RateLimited,
            _ => SandboxStatus::Failed,
        },
        output: tool_reply.error_code.unwrap_or_default().into_bytes(),
        audit: SandboxAuditSummary {
            call_digest: call.digest(),
            token_digest: None,
            bytes_out: tool_reply.bytes_out,
            bytes_in: tool_reply.bytes_in,
        },
        finished_at_t: call.t,
    }
}

fn failed(call: &SandboxCall, code: &str) -> SandboxReply {
    SandboxReply {
        status: SandboxStatus::Failed,
        output: code.as_bytes().to_vec(),
        audit: SandboxAuditSummary {
            call_digest: call.digest(),
            token_digest: None,
            bytes_out: 0,
            bytes_in: call.input.len() as u32,
        },
        finished_at_t: call.t,
    }
}

fn map_kind(module: &str, op: &str) -> ucf_policy::capability::CapabilityKind {
    match (module, op) {
        ("tools.external", "emit_text") => ucf_policy::capability::CapabilityKind::ExternalApi,
        ("tools.memory", "write_bytes") => ucf_policy::capability::CapabilityKind::FileWrite,
        ("tools.fs", "read") => ucf_policy::capability::CapabilityKind::FileRead,
        ("tools.http", "get") => ucf_policy::capability::CapabilityKind::NetHttp,
        _ => ucf_policy::capability::CapabilityKind::Custom("unknown".to_string()),
    }
}

fn map_target(module: &str, op: &str) -> String {
    match (module, op) {
        ("tools.external", "emit_text") => "external_output".to_string(),
        ("tools.memory", "write_bytes") => "memory_write".to_string(),
        ("tools.fs", "read") => "fs_read".to_string(),
        ("tools.http", "get") => "http_get".to_string(),
        _ => "unknown".to_string(),
    }
}

#[derive(Clone, Debug)]
struct ToolReplyWire {
    status: u8,
    bytes_out: u32,
    bytes_in: u32,
    error_code: Option<String>,
}

impl IpcEnvelope {
    fn new(msg_id: u64, kind: IpcKind, payload: Vec<u8>) -> Self {
        let mut h = Hasher::new();
        h.update(&payload);
        Self {
            schema_version: SCHEMA_VERSION,
            msg_id,
            kind,
            payload,
            payload_digest: h.finalize().into(),
        }
    }
}

fn write_frame(w: &mut dyn Write, payload: &[u8]) -> std::io::Result<()> {
    if payload.len() > MAX_MSG_BYTES {
        return Err(std::io::Error::other("msg_too_large"));
    }
    w.write_all(&(payload.len() as u32).to_le_bytes())?;
    w.write_all(payload)?;
    w.flush()?;
    Ok(())
}

fn read_frame(r: &mut dyn Read) -> std::io::Result<Vec<u8>> {
    let mut lb = [0u8; 4];
    r.read_exact(&mut lb)?;
    let len = u32::from_le_bytes(lb) as usize;
    if len > MAX_MSG_BYTES {
        return Err(std::io::Error::other("msg_too_large"));
    }
    let mut out = vec![0u8; len];
    r.read_exact(&mut out)?;
    Ok(out)
}

fn encode_envelope(env: &IpcEnvelope) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&env.schema_version.to_be_bytes());
    out.extend_from_slice(&env.msg_id.to_be_bytes());
    out.push(env.kind as u8);
    put_bytes(&mut out, &env.payload);
    out.extend_from_slice(&env.payload_digest);
    out
}

fn decode_envelope(input: &[u8]) -> Result<IpcEnvelope, ()> {
    let mut i = 0usize;
    let schema_version = read_u16(input, &mut i).map_err(|_| ())?;
    let msg_id = read_u64(input, &mut i).map_err(|_| ())?;
    let kind = match *input.get(i).ok_or(())? {
        1 => IpcKind::Call,
        2 => IpcKind::Reply,
        3 => IpcKind::ToolRequest,
        4 => IpcKind::ToolReply,
        5 => IpcKind::Heartbeat,
        6 => IpcKind::Shutdown,
        _ => return Err(()),
    };
    i += 1;
    let payload = read_bytes(input, &mut i).map_err(|_| ())?;
    let payload_digest = read_digest(input, &mut i).map_err(|_| ())?;
    let mut h = Hasher::new();
    h.update(&payload);
    let digest: [u8; 32] = h.finalize().into();
    if digest != payload_digest {
        return Err(());
    }
    Ok(IpcEnvelope {
        schema_version,
        msg_id,
        kind,
        payload,
        payload_digest,
    })
}

fn encode_heartbeat(tag: &str) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&SCHEMA_VERSION.to_be_bytes());
    put_string(&mut out, tag);
    out
}

fn encode_tool_request(req: &ToolRequest) -> Vec<u8> {
    let mut out = Vec::new();
    put_u64(&mut out, req.id);
    put_string(&mut out, req.kind.as_tag());
    put_string(&mut out, &req.target);
    put_u32(&mut out, req.payload_hint.bytes_out.unwrap_or(u32::MAX));
    put_u32(&mut out, req.payload_hint.bytes_in.unwrap_or(u32::MAX));
    put_u64(&mut out, req.requested_at_t);
    put_u64(&mut out, req.decision_id);
    out.extend_from_slice(&req.evidence_chain_digest);
    out
}

fn decode_tool_reply(input: &[u8]) -> Result<ToolReplyWire, ()> {
    let mut i = 0usize;
    let status = *input.get(i).ok_or(())?;
    i += 1;
    let bytes_out = read_u32(input, &mut i).map_err(|_| ())?;
    let bytes_in = read_u32(input, &mut i).map_err(|_| ())?;
    let has_error = *input.get(i).ok_or(())?;
    i += 1;
    let error_code = if has_error == 1 {
        Some(read_string(input, &mut i).map_err(|_| ())?)
    } else {
        None
    };
    Ok(ToolReplyWire {
        status,
        bytes_out: if bytes_out == u32::MAX { 0 } else { bytes_out },
        bytes_in: if bytes_in == u32::MAX { 0 } else { bytes_in },
        error_code,
    })
}

fn encode_sandbox_reply(reply: &SandboxReply) -> Vec<u8> {
    reply.canonical_bytes()
}

fn decode_sandbox_call(input: &[u8]) -> Result<SandboxCall, ()> {
    let mut i = 0usize;
    let call_id = read_u64(input, &mut i).map_err(|_| ())?;
    let t = read_u64(input, &mut i).map_err(|_| ())?;
    let module = read_string(input, &mut i).map_err(|_| ())?;
    let op = read_string(input, &mut i).map_err(|_| ())?;
    let data = read_bytes(input, &mut i).map_err(|_| ())?;
    let caps = read_u32(input, &mut i).map_err(|_| ())? as usize;
    for _ in 0..caps {
        let _ = read_digest(input, &mut i).map_err(|_| ())?;
        let _ = read_string(input, &mut i).map_err(|_| ())?;
        let _ = read_string(input, &mut i).map_err(|_| ())?;
    }
    let evidence = read_digest(input, &mut i).map_err(|_| ())?;
    Ok(SandboxCall {
        call_id: CallId(call_id),
        t,
        module,
        op,
        input: data,
        capabilities: CapabilitySetSummary::default(),
        evidence_chain_digest: evidence,
    })
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}
fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_be_bytes());
}
fn put_string(out: &mut Vec<u8>, s: &str) {
    put_u32(out, s.len() as u32);
    out.extend_from_slice(s.as_bytes());
}
fn put_bytes(out: &mut Vec<u8>, b: &[u8]) {
    put_u32(out, b.len() as u32);
    out.extend_from_slice(b);
}
fn read_u16(input: &[u8], i: &mut usize) -> Result<u16, ()> {
    if input.len().saturating_sub(*i) < 2 {
        return Err(());
    }
    let v = u16::from_be_bytes([input[*i], input[*i + 1]]);
    *i += 2;
    Ok(v)
}
fn read_u32(input: &[u8], i: &mut usize) -> Result<u32, ()> {
    if input.len().saturating_sub(*i) < 4 {
        return Err(());
    }
    let v = u32::from_be_bytes([input[*i], input[*i + 1], input[*i + 2], input[*i + 3]]);
    *i += 4;
    Ok(v)
}
fn read_u64(input: &[u8], i: &mut usize) -> Result<u64, ()> {
    if input.len().saturating_sub(*i) < 8 {
        return Err(());
    }
    let v = u64::from_be_bytes([
        input[*i],
        input[*i + 1],
        input[*i + 2],
        input[*i + 3],
        input[*i + 4],
        input[*i + 5],
        input[*i + 6],
        input[*i + 7],
    ]);
    *i += 8;
    Ok(v)
}
fn read_bytes(input: &[u8], i: &mut usize) -> Result<Vec<u8>, ()> {
    let l = read_u32(input, i)? as usize;
    if input.len().saturating_sub(*i) < l {
        return Err(());
    }
    let v = input[*i..*i + l].to_vec();
    *i += l;
    Ok(v)
}
fn read_string(input: &[u8], i: &mut usize) -> Result<String, ()> {
    String::from_utf8(read_bytes(input, i)?).map_err(|_| ())
}
fn read_digest(input: &[u8], i: &mut usize) -> Result<[u8; 32], ()> {
    if input.len().saturating_sub(*i) < 32 {
        return Err(());
    }
    let mut d = [0; 32];
    d.copy_from_slice(&input[*i..*i + 32]);
    *i += 32;
    Ok(d)
}
