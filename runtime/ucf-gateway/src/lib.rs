#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream};
#[cfg(unix)]
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};

use blake3::Hasher;
use prost::Message;
use serde::Serialize;
use thiserror::Error;
use ucf_core::types::{SimTime, Tick, WindowId};
use ucf_ess::v1::ExperienceStore;
use ucf_frames::v1::{
    ChannelCode, ControlFrame, ControlPayload, CorrelationId, DecisionCode, DecisionFrame, Intent,
    IntentId, IntentKind,
};
use ucf_ops::{explain_tick, readiness_gate, ExplainTickRequest};
use ucf_policy::adapter::MockAdapter;
use ucf_runtime::RuntimeOrchestrator;

pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/ucf.gateway.v1.rs"));
}

const SCHEMA_VERSION: u32 = 1;
const MAX_MESSAGE_BYTES: usize = 128 * 1024;
const MAX_LIST: usize = 64;

#[derive(Debug, Clone)]
pub enum GatewayTransport {
    Unix(PathBuf),
    TcpLocal(u16),
}

impl GatewayTransport {
    pub fn default_v1() -> Self {
        #[cfg(unix)]
        {
            return Self::Unix(PathBuf::from("/tmp/ucf_gateway_v1.sock"));
        }
        #[allow(unreachable_code)]
        Self::TcpLocal(44991)
    }
}

#[derive(Debug, Clone)]
pub struct GatewayConfig {
    pub run_id: String,
    pub policy_graph_digest_prefix: [u8; 8],
    pub auth_tokens: BTreeMap<String, BTreeSet<String>>,
    pub access_log_path: PathBuf,
    pub workdir: PathBuf,
    pub max_message_bytes: usize,
}

impl GatewayConfig {
    pub fn for_tests(tmp: &Path) -> Self {
        let mut auth_tokens = BTreeMap::new();
        auth_tokens.insert(
            "test-token".to_string(),
            ["submit", "subscribe", "ess:read", "report:read"]
                .into_iter()
                .map(ToString::to_string)
                .collect(),
        );
        Self {
            run_id: "run-test".to_string(),
            policy_graph_digest_prefix: [1, 2, 3, 4, 5, 6, 7, 8],
            auth_tokens,
            access_log_path: tmp.join("gateway_access_records.jsonl"),
            workdir: tmp.to_path_buf(),
            max_message_bytes: MAX_MESSAGE_BYTES,
        }
    }
}

#[derive(Debug, Error)]
pub enum GatewayError {
    #[error("unauthorized")]
    Unauthorized,
    #[error("unsupported version")]
    UnsupportedVersion,
    #[error("message too large")]
    MessageTooLarge,
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("encode/decode: {0}")]
    Proto(String),
}

#[derive(Debug, Serialize)]
pub struct GatewayAccessRecord {
    pub schema_version: u16,
    pub endpoint: String,
    pub t_ms: u64,
    pub status: String,
    pub client_id_digest: String,
}

pub struct GatewayService {
    config: GatewayConfig,
    orchestrator: RuntimeOrchestrator,
    adapter: MockAdapter,
    decisions: VecDeque<proto::DecisionEvent>,
}

impl GatewayService {
    pub fn new(config: GatewayConfig) -> Self {
        Self {
            config,
            orchestrator: RuntimeOrchestrator::new(),
            adapter: MockAdapter::default(),
            decisions: VecDeque::new(),
        }
    }

    pub fn negotiate(
        &self,
        req: &proto::HandshakeRequest,
    ) -> Result<proto::HandshakeResponse, GatewayError> {
        if req.schema_version != SCHEMA_VERSION {
            return Err(GatewayError::UnsupportedVersion);
        }
        let selected = req
            .supported_versions
            .iter()
            .copied()
            .filter(|v| *v == 1)
            .max()
            .ok_or(GatewayError::UnsupportedVersion)?;
        let granted = self.authorize(&req.auth_token, "submit")?;
        Ok(proto::HandshakeResponse {
            schema_version: SCHEMA_VERSION,
            selected_version: selected,
            error: None,
            granted_capabilities: granted.into_iter().collect(),
        })
    }

    fn authorize(&self, token: &str, capability: &str) -> Result<BTreeSet<String>, GatewayError> {
        let caps = self
            .config
            .auth_tokens
            .get(token)
            .cloned()
            .ok_or(GatewayError::Unauthorized)?;
        if !caps.contains(capability) {
            return Err(GatewayError::Unauthorized);
        }
        Ok(caps)
    }

    fn validate_common(
        &self,
        schema_version: u32,
        policy_digest: &[u8],
    ) -> Result<(), GatewayError> {
        if schema_version != SCHEMA_VERSION {
            return Err(GatewayError::UnsupportedVersion);
        }
        if policy_digest != self.config.policy_graph_digest_prefix {
            return Err(GatewayError::InvalidRequest(
                "policy_graph_digest_prefix mismatch".to_string(),
            ));
        }
        Ok(())
    }

    pub fn submit_control_frame(
        &mut self,
        token: &str,
        req: proto::ControlFrameSubmitRequest,
    ) -> Result<proto::ControlFrameSubmitResponse, GatewayError> {
        self.authorize(token, "submit")?;
        self.validate_common(req.schema_version, &req.policy_graph_digest_prefix)?;
        if req.payload_text_utf8.len() > 4096 || req.intent_summary.len() > 256 {
            return Err(GatewayError::MessageTooLarge);
        }
        let intent_kind = map_intent_kind(req.intent_kind)?;
        let channel = map_channel(req.channel)?;
        let text = std::str::from_utf8(&req.payload_text_utf8).map_err(|_| {
            GatewayError::InvalidRequest("payload_text_utf8 must be valid utf8".to_string())
        })?;
        let ctrl = ControlFrame::new_text(
            SimTime {
                tick: Tick::new(req.tick),
                window: WindowId::new(req.window),
            },
            CorrelationId(req.corr_id),
            channel,
            Intent::new(
                IntentId(req.intent_id),
                intent_kind,
                req.intent_summary.clone(),
            ),
            text,
        );
        let frame_digest = digest_control_frame(&ctrl);
        let decision = self
            .orchestrator
            .ingest_and_process(&mut self.adapter, ctrl)
            .map_err(|e| GatewayError::InvalidRequest(format!("runtime ingest failed: {e}")))?;
        let evt = decision_to_event(
            &self.config.run_id,
            self.config.policy_graph_digest_prefix,
            &decision,
        );
        self.decisions.push_back(evt.clone());
        while self.decisions.len() > MAX_LIST {
            self.decisions.pop_front();
        }

        Ok(proto::ControlFrameSubmitResponse {
            schema_version: SCHEMA_VERSION,
            run_id: self.config.run_id.clone(),
            frame_digest: frame_digest.to_vec(),
            decision: Some(evt),
            error: None,
        })
    }

    pub fn subscribe_decisions(
        &self,
        token: &str,
        req: proto::DecisionStreamSubscribeRequest,
    ) -> Result<proto::DecisionStreamSubscribeResponse, GatewayError> {
        self.authorize(token, "subscribe")?;
        self.validate_common(req.schema_version, &req.policy_graph_digest_prefix)?;
        let max = usize::try_from(req.max_events)
            .unwrap_or(MAX_LIST)
            .min(MAX_LIST);
        let len = self.decisions.len();
        let start = len.saturating_sub(max);
        Ok(proto::DecisionStreamSubscribeResponse {
            schema_version: SCHEMA_VERSION,
            events: self.decisions.iter().skip(start).cloned().collect(),
            error: None,
        })
    }

    pub fn query_ess(
        &self,
        token: &str,
        req: proto::EssQueryRequest,
    ) -> Result<proto::EssQueryResponse, GatewayError> {
        self.authorize(token, "ess:read")?;
        self.validate_common(req.schema_version, &req.policy_graph_digest_prefix)?;
        let max = usize::try_from(req.max_records)
            .unwrap_or(MAX_LIST)
            .min(MAX_LIST);
        let len = self.orchestrator.ess.len();
        let start = len.saturating_sub(max);
        let summaries = (start..len)
            .filter_map(|idx| self.orchestrator.ess.get(idx))
            .map(|rec| proto::EssSummary {
                experience_id: rec.id.0,
                tick: rec.time.tick.get(),
                kind: format!("{:?}", rec.kind),
                corr_id: rec.corr.0,
            })
            .collect();
        Ok(proto::EssQueryResponse {
            schema_version: SCHEMA_VERSION,
            summaries,
            error: None,
        })
    }

    pub fn get_report(
        &self,
        token: &str,
        req: proto::ReportRequest,
    ) -> Result<proto::ReportResponse, GatewayError> {
        self.authorize(token, "report:read")?;
        self.validate_common(req.schema_version, &req.policy_graph_digest_prefix)?;
        let report_json = match req.report_type.as_str() {
            "explain-tick" => {
                let r = explain_tick(
                    &self.config.workdir,
                    ExplainTickRequest {
                        t: Some(req.tick),
                        decision_id: None,
                        detail_level: 1,
                        digest_prefix_len: 8,
                    },
                )
                .map_err(|e| GatewayError::InvalidRequest(format!("explain-tick failed: {e}")))?;
                serde_json::to_vec(&r).map_err(|e| GatewayError::Proto(e.to_string()))?
            }
            "readiness-gate" => {
                let r = readiness_gate(
                    &self.config.workdir,
                    "test",
                    &self.config.workdir.join("out/gate_report.json"),
                )
                .map_err(|e| GatewayError::InvalidRequest(format!("readiness-gate failed: {e}")))?;
                serde_json::to_vec(&r).map_err(|e| GatewayError::Proto(e.to_string()))?
            }
            _ => {
                return Err(GatewayError::InvalidRequest(
                    "unsupported report_type".to_string(),
                ));
            }
        };
        Ok(proto::ReportResponse {
            schema_version: SCHEMA_VERSION,
            report_type: req.report_type,
            report_json_utf8: report_json,
            error: None,
        })
    }

    pub fn handle_request(
        &mut self,
        req: proto::GatewayRequest,
        token: &str,
        client_id: &str,
    ) -> proto::GatewayResponse {
        let (endpoint, result) = match req.payload {
            Some(proto::gateway_request::Payload::Handshake(r)) => (
                "handshake",
                self.negotiate(&r)
                    .map(proto::gateway_response::Payload::Handshake),
            ),
            Some(proto::gateway_request::Payload::Submit(r)) => (
                "submit",
                self.submit_control_frame(token, r)
                    .map(proto::gateway_response::Payload::Submit),
            ),
            Some(proto::gateway_request::Payload::Subscribe(r)) => (
                "subscribe",
                self.subscribe_decisions(token, r)
                    .map(proto::gateway_response::Payload::Subscribe),
            ),
            Some(proto::gateway_request::Payload::EssQuery(r)) => (
                "ess_query",
                self.query_ess(token, r)
                    .map(proto::gateway_response::Payload::EssQuery),
            ),
            Some(proto::gateway_request::Payload::Report(r)) => (
                "report",
                self.get_report(token, r)
                    .map(proto::gateway_response::Payload::Report),
            ),
            None => (
                "none",
                Err(GatewayError::InvalidRequest("missing payload".to_string())),
            ),
        };
        let status = if result.is_ok() { "ok" } else { "deny" };
        let _ = self.log_access(endpoint, status, client_id);
        match result {
            Ok(payload) => proto::GatewayResponse {
                negotiated_version: req.negotiated_version,
                payload: Some(payload),
            },
            Err(err) => proto::GatewayResponse {
                negotiated_version: req.negotiated_version,
                payload: Some(proto::gateway_response::Payload::Error(proto::Error {
                    code: map_err_code(&err),
                    message: err.to_string(),
                })),
            },
        }
    }

    fn log_access(
        &self,
        endpoint: &str,
        status: &str,
        client_id: &str,
    ) -> Result<(), GatewayError> {
        let rec = GatewayAccessRecord {
            schema_version: 1,
            endpoint: endpoint.to_string(),
            t_ms: now_ms(),
            status: status.to_string(),
            client_id_digest: hex::encode(digest_bytes(client_id.as_bytes())),
        };
        if let Some(parent) = self.config.access_log_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.config.access_log_path)?;
        let line = serde_json::to_string(&rec).map_err(|e| GatewayError::Proto(e.to_string()))?;
        writeln!(f, "{line}")?;
        Ok(())
    }
}

pub fn read_frame<R: Read>(
    reader: &mut R,
    max_bytes: usize,
) -> Result<proto::GatewayRequest, GatewayError> {
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > max_bytes {
        return Err(GatewayError::MessageTooLarge);
    }
    let mut body = vec![0u8; len];
    reader.read_exact(&mut body)?;
    proto::GatewayRequest::decode(body.as_slice()).map_err(|e| GatewayError::Proto(e.to_string()))
}

pub fn write_frame<W: Write>(
    writer: &mut W,
    response: &proto::GatewayResponse,
) -> Result<(), GatewayError> {
    let body = response.encode_to_vec();
    let len = u32::try_from(body.len()).map_err(|_| GatewayError::MessageTooLarge)?;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&body)?;
    writer.flush()?;
    Ok(())
}

pub fn run_tcp_once(service: &mut GatewayService, port: u16) -> Result<(), GatewayError> {
    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
    let listener = TcpListener::bind(addr)?;
    let (mut stream, _) = listener.accept()?;
    process_connection(service, &mut stream)
}

fn process_connection(
    service: &mut GatewayService,
    stream: &mut TcpStream,
) -> Result<(), GatewayError> {
    let req = read_frame(stream, service.config.max_message_bytes)?;
    let token = extract_token(&req).to_string();
    let client_id = extract_client_id(&req).to_string();
    let resp = service.handle_request(req, &token, &client_id);
    write_frame(stream, &resp)
}

#[cfg(unix)]
pub fn run_unix_once(service: &mut GatewayService, path: &Path) -> Result<(), GatewayError> {
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    let listener = UnixListener::bind(path)?;
    let (mut stream, _) = listener.accept()?;
    let req = read_frame_unix(&mut stream, service.config.max_message_bytes)?;
    let token = extract_token(&req).to_string();
    let client_id = extract_client_id(&req).to_string();
    let resp = service.handle_request(req, &token, &client_id);
    write_frame_unix(&mut stream, &resp)
}

#[cfg(unix)]
fn read_frame_unix(
    stream: &mut UnixStream,
    max_bytes: usize,
) -> Result<proto::GatewayRequest, GatewayError> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > max_bytes {
        return Err(GatewayError::MessageTooLarge);
    }
    let mut body = vec![0u8; len];
    stream.read_exact(&mut body)?;
    proto::GatewayRequest::decode(body.as_slice()).map_err(|e| GatewayError::Proto(e.to_string()))
}

#[cfg(unix)]
fn write_frame_unix(
    stream: &mut UnixStream,
    response: &proto::GatewayResponse,
) -> Result<(), GatewayError> {
    let body = response.encode_to_vec();
    let len = u32::try_from(body.len()).map_err(|_| GatewayError::MessageTooLarge)?;
    stream.write_all(&len.to_le_bytes())?;
    stream.write_all(&body)?;
    stream.flush()?;
    Ok(())
}

fn decision_to_event(
    run_id: &str,
    policy_prefix: [u8; 8],
    decision: &DecisionFrame,
) -> proto::DecisionEvent {
    proto::DecisionEvent {
        schema_version: SCHEMA_VERSION,
        run_id: run_id.to_string(),
        tick: decision.time.tick.get(),
        corr_id: decision.corr.0,
        decision_code: match decision.decision {
            DecisionCode::Allow => "allow",
            DecisionCode::Deny => "deny",
            DecisionCode::Defer => "defer",
        }
        .to_string(),
        reason_code: decision.reason_code.0.to_string(),
        rationale_redacted: redact(&decision.rationale),
        policy_graph_digest_prefix: policy_prefix.to_vec(),
    }
}

fn redact(input: &str) -> String {
    let mut out = input.chars().take(80).collect::<String>();
    if input.chars().count() > 80 {
        out.push('…');
    }
    out
}

fn map_intent_kind(kind: u32) -> Result<IntentKind, GatewayError> {
    match kind {
        1 => Ok(IntentKind::Speak),
        2 => Ok(IntentKind::Act),
        3 => Ok(IntentKind::QueryMemory),
        4 => Ok(IntentKind::WriteMemory),
        5 => Ok(IntentKind::StimulateBrain),
        6 => Ok(IntentKind::System),
        _ => Err(GatewayError::InvalidRequest(
            "unknown intent_kind".to_string(),
        )),
    }
}

fn map_channel(channel: u32) -> Result<ChannelCode, GatewayError> {
    match channel {
        1 => Ok(ChannelCode::ExternalOutput),
        2 => Ok(ChannelCode::InternalThought),
        3 => Ok(ChannelCode::MemoryWrite),
        4 => Ok(ChannelCode::BrainStimulus),
        _ => Err(GatewayError::InvalidRequest("unknown channel".to_string())),
    }
}

fn digest_control_frame(frame: &ControlFrame) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.gateway.control_frame.v1");
    hasher.update(&frame.time.tick.get().to_le_bytes());
    hasher.update(&frame.time.window.get().to_le_bytes());
    hasher.update(&frame.corr.0.to_le_bytes());
    hasher.update(frame.intent.summary.as_bytes());
    if let ControlPayload::Text(text) = &frame.payload {
        hasher.update(text.as_bytes());
    }
    *hasher.finalize().as_bytes()
}

fn digest_bytes(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.gateway.client.v1");
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

fn map_err_code(err: &GatewayError) -> u32 {
    match err {
        GatewayError::Unauthorized => 401,
        GatewayError::UnsupportedVersion => 426,
        GatewayError::MessageTooLarge => 413,
        GatewayError::InvalidRequest(_) => 400,
        GatewayError::Io(_) | GatewayError::Proto(_) => 500,
    }
}

fn extract_token(req: &proto::GatewayRequest) -> &str {
    match &req.payload {
        Some(proto::gateway_request::Payload::Handshake(h)) => h.auth_token.as_str(),
        Some(proto::gateway_request::Payload::Submit(r)) => r.auth_token.as_str(),
        Some(proto::gateway_request::Payload::Subscribe(r)) => r.auth_token.as_str(),
        Some(proto::gateway_request::Payload::EssQuery(r)) => r.auth_token.as_str(),
        Some(proto::gateway_request::Payload::Report(r)) => r.auth_token.as_str(),
        _ => "",
    }
}

fn extract_client_id(req: &proto::GatewayRequest) -> &str {
    match &req.payload {
        Some(proto::gateway_request::Payload::Handshake(h)) => h.client_id.as_str(),
        _ => "anonymous",
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub fn local_bind_for_platform(preferred_port: u16) -> GatewayTransport {
    #[cfg(unix)]
    {
        let _ = preferred_port;
        GatewayTransport::Unix(PathBuf::from("/tmp/ucf_gateway_v1.sock"))
    }
    #[cfg(not(unix))]
    {
        GatewayTransport::TcpLocal(preferred_port)
    }
}
