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
use serde::{Deserialize, Serialize};
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
const MAX_MESSAGE_BYTES: usize = 256 * 1024;
const MAX_LIST: usize = 64;
const SUBMIT_MAX_BYTES: usize = 4096;
const SUMMARY_MAX_CHARS: usize = 256;
const REQUEST_ID_PREFIX: usize = 12;

const ERR_AUTH_DENIED: u32 = 1001;
const ERR_RATE_LIMITED: u32 = 1002;
const ERR_SCHEMA_INVALID: u32 = 1003;
const ERR_POLICY_DENIED: u32 = 1004;
const ERR_TOO_LARGE: u32 = 1005;
const ERR_VERSION_MISMATCH: u32 = 1006;
const ERR_INTERNAL: u32 = 1500;
const ERR_UNAVAILABLE: u32 = 1501;

#[derive(Debug, Clone)]
pub enum GatewayTransport {
    Unix(PathBuf),
    Pipe(String),
    TcpLocal(u16),
}

impl GatewayTransport {
    pub fn default_v1(bundle_root: &Path) -> Self {
        #[cfg(unix)]
        {
            return Self::Unix(default_unix_socket_path(bundle_root));
        }
        #[cfg(windows)]
        {
            let _ = bundle_root;
            return Self::Pipe(default_windows_pipe_name());
        }
        #[allow(unreachable_code)]
        Self::TcpLocal(44991)
    }
}

pub fn default_unix_socket_path(bundle_root: &Path) -> PathBuf {
    bundle_root.join("data").join("ipc").join("gateway.sock")
}

pub fn default_windows_pipe_name() -> String {
    r"\\.\pipe\ucf_gateway".to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportKind {
    Unix,
    Pipe,
    Tcp,
}

impl TransportKind {
    pub fn parse(input: &str) -> Result<Self, GatewayError> {
        match input {
            "unix" => Ok(Self::Unix),
            "pipe" => Ok(Self::Pipe),
            "tcp" => Ok(Self::Tcp),
            _ => Err(GatewayError::SchemaInvalid),
        }
    }
}

pub fn transport_from_env(bundle_root: &Path) -> Result<GatewayTransport, GatewayError> {
    let kind = match std::env::var("UCF_GATEWAY_TRANSPORT") {
        Ok(raw) => TransportKind::parse(raw.trim())?,
        Err(_) => {
            #[cfg(unix)]
            {
                TransportKind::Unix
            }
            #[cfg(windows)]
            {
                TransportKind::Pipe
            }
            #[cfg(not(any(unix, windows)))]
            {
                TransportKind::Tcp
            }
        }
    };
    let transport = match kind {
        TransportKind::Unix => GatewayTransport::Unix(default_unix_socket_path(bundle_root)),
        TransportKind::Pipe => GatewayTransport::Pipe(default_windows_pipe_name()),
        TransportKind::Tcp => parse_explicit_tcp_bind()?,
    };
    enforce_transport_policy(bundle_root, &transport)?;
    Ok(transport)
}

fn enforce_transport_policy(
    bundle_root: &Path,
    transport: &GatewayTransport,
) -> Result<(), GatewayError> {
    match transport {
        GatewayTransport::Unix(_) | GatewayTransport::Pipe(_) => Ok(()),
        GatewayTransport::TcpLocal(port) => {
            let bind =
                std::env::var("UCF_GATEWAY_BIND").unwrap_or_else(|_| format!("127.0.0.1:{port}"));
            match bind.parse::<SocketAddr>() {
                Ok(addr) if addr.ip().is_loopback() => Ok(()),
                Ok(addr) => {
                    let action = if strict_mode_enabled() {
                        "shutdown"
                    } else {
                        "safe_only"
                    };
                    record_gateway_violation(
                        bundle_root,
                        "gateway_non_loopback_bind",
                        "non_loopback",
                        action,
                        format!("requested gateway bind {addr}"),
                    );
                    Err(GatewayError::Unauthorized)
                }
                Err(_) => Err(GatewayError::SchemaInvalid),
            }
        }
    }
}

fn runtime_socket_audit_tick(workdir: &Path) {
    if let Some(detail) = detect_non_loopback_tcp_established() {
        let action = if strict_mode_enabled() {
            "safe_only"
        } else {
            "warn"
        };
        record_gateway_violation(
            workdir,
            "runtime_non_loopback_tcp_detected",
            "non_loopback",
            action,
            detail,
        );
    }
}

fn parse_explicit_tcp_bind() -> Result<GatewayTransport, GatewayError> {
    let bind = std::env::var("UCF_GATEWAY_BIND").map_err(|_| GatewayError::Unauthorized)?;
    let parsed: SocketAddr = bind.parse().map_err(|_| GatewayError::SchemaInvalid)?;
    Ok(GatewayTransport::TcpLocal(parsed.port()))
}

#[derive(Debug, Clone, Copy)]
pub struct RateLimitConfig {
    pub capacity: u32,
    pub refill_per_sec: u32,
}

#[derive(Debug, Clone)]
pub struct GatewayConfig {
    pub run_id: String,
    pub policy_graph_digest_prefix: [u8; 8],
    pub token_capabilities_by_hash: BTreeMap<String, BTreeSet<String>>,
    pub access_log_path: PathBuf,
    pub abuse_log_path: PathBuf,
    pub workdir: PathBuf,
    pub max_message_bytes: usize,
    pub rate_limits: BTreeMap<String, RateLimitConfig>,
    pub strict_mode: bool,
    pub manifest_digest_prefix: String,
    pub env_mode: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NetworkViolationRecord {
    pub schema_version: u16,
    pub reason_code: String,
    pub endpoint_class: String,
    pub action: String,
    pub detail: String,
    pub strict_mode: bool,
    pub t_ms: u64,
}

fn strict_mode_enabled() -> bool {
    std::env::var("UCF_STRICT_MODE").ok().as_deref() == Some("1")
}

fn network_violation_log_path(workdir: &Path) -> PathBuf {
    workdir.join("network_violations.jsonl")
}

fn append_network_violation(
    workdir: &Path,
    record: &NetworkViolationRecord,
) -> Result<(), GatewayError> {
    if let Some(parent) = network_violation_log_path(workdir).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(network_violation_log_path(workdir))?;
    let line = serde_json::to_string(record).map_err(|_| GatewayError::Internal)?;
    file.write_all(line.as_bytes())?;
    file.write_all(
        b"
",
    )?;
    Ok(())
}

fn record_gateway_violation(
    workdir: &Path,
    reason_code: &str,
    endpoint_class: &str,
    action: &str,
    detail: String,
) {
    let record = NetworkViolationRecord {
        schema_version: 1,
        reason_code: reason_code.to_string(),
        endpoint_class: endpoint_class.to_string(),
        action: action.to_string(),
        detail,
        strict_mode: strict_mode_enabled(),
        t_ms: now_ms(),
    };
    let _ = append_network_violation(workdir, &record);
}

#[cfg(target_os = "linux")]
fn detect_non_loopback_tcp_established() -> Option<String> {
    let body = std::fs::read_to_string("/proc/net/tcp").ok()?;
    for line in body.lines().skip(1) {
        let cols = line.split_whitespace().collect::<Vec<_>>();
        if cols.len() < 4 {
            continue;
        }
        let remote = cols[2];
        let state = cols[3];
        if state != "01" {
            continue;
        }
        let Some((remote_ip_hex, remote_port_hex)) = remote.split_once(':') else {
            continue;
        };
        if remote_ip_hex.eq_ignore_ascii_case("00000000")
            || remote_ip_hex.eq_ignore_ascii_case("0100007F")
        {
            continue;
        }
        return Some(format!(
            "remote={remote_ip_hex}:{remote_port_hex} state={state}"
        ));
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn detect_non_loopback_tcp_established() -> Option<String> {
    None
}

impl GatewayConfig {
    pub fn for_tests(tmp: &Path) -> Self {
        let mut token_capabilities_by_hash = BTreeMap::new();
        token_capabilities_by_hash.insert(
            hash_token("test-token"),
            [
                "submit",
                "subscribe",
                "ess:read",
                "report:read",
                "health:read",
            ]
            .into_iter()
            .map(ToString::to_string)
            .collect(),
        );

        let mut rate_limits = BTreeMap::new();
        rate_limits.insert(
            "submit".to_string(),
            RateLimitConfig {
                capacity: 5,
                refill_per_sec: 5,
            },
        );
        rate_limits.insert(
            "ess_query".to_string(),
            RateLimitConfig {
                capacity: 10,
                refill_per_sec: 10,
            },
        );
        rate_limits.insert(
            "report".to_string(),
            RateLimitConfig {
                capacity: 2,
                refill_per_sec: 2,
            },
        );
        rate_limits.insert(
            "health".to_string(),
            RateLimitConfig {
                capacity: 2,
                refill_per_sec: 2,
            },
        );

        Self {
            run_id: "run-test".to_string(),
            policy_graph_digest_prefix: [1, 2, 3, 4, 5, 6, 7, 8],
            token_capabilities_by_hash,
            access_log_path: tmp.join("gateway_access_records.jsonl"),
            abuse_log_path: tmp.join("gateway_abuse_records.jsonl"),
            workdir: tmp.to_path_buf(),
            max_message_bytes: MAX_MESSAGE_BYTES,
            rate_limits,
            strict_mode: strict_mode_enabled(),
            manifest_digest_prefix: "unknown".to_string(),
            env_mode: "test".to_string(),
        }
    }

    pub fn from_env(tmp: &Path) -> Result<Self, GatewayError> {
        let mode = std::env::var("UCF_ENV").unwrap_or_else(|_| "dev".to_string());
        let mut cfg = Self::for_tests(tmp);
        cfg.env_mode = mode.clone();
        cfg.strict_mode = strict_mode_enabled();
        cfg.manifest_digest_prefix = load_manifest_digest_prefix();
        cfg.token_capabilities_by_hash.clear();
        if let Ok(token) = std::env::var("UCF_GATEWAY_TOKEN") {
            cfg.token_capabilities_by_hash.insert(
                hash_token(token.as_str()),
                [
                    "submit",
                    "subscribe",
                    "ess:read",
                    "report:read",
                    "health:read",
                ]
                .into_iter()
                .map(ToString::to_string)
                .collect(),
            );
            return Ok(cfg);
        }
        if mode == "dev" {
            eprintln!("warning: UCF_GATEWAY_TOKEN not set, enabling dev fallback token");
            cfg.token_capabilities_by_hash.insert(
                hash_token("dev-token"),
                [
                    "submit",
                    "subscribe",
                    "ess:read",
                    "report:read",
                    "health:read",
                ]
                .into_iter()
                .map(ToString::to_string)
                .collect(),
            );
            return Ok(cfg);
        }
        Err(GatewayError::Unauthorized)
    }
}

fn load_manifest_digest_prefix() -> String {
    std::fs::read_to_string("models/lifecycle_manifest.toml")
        .ok()
        .and_then(|raw| {
            raw.lines()
                .find(|l| l.trim_start().starts_with("manifest_digest"))
                .and_then(|line| line.split('=').nth(1))
                .map(|v| v.trim().trim_matches('"').chars().take(12).collect())
        })
        .filter(|s: &String| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

#[derive(Debug, Error)]
pub enum GatewayError {
    #[error("auth denied")]
    Unauthorized,
    #[error("rate limited")]
    RateLimited,
    #[error("unsupported version")]
    UnsupportedVersion,
    #[error("message too large")]
    MessageTooLarge,
    #[error("schema invalid")]
    SchemaInvalid,
    #[error("policy denied")]
    PolicyDenied,
    #[error("internal")]
    Internal,
    #[error("unavailable")]
    Unavailable,
    #[error("io")]
    Io(#[from] std::io::Error),
    #[error("encode/decode")]
    Proto,
}

#[derive(Debug, Serialize)]
pub struct GatewayAccessRecord {
    pub schema_version: u16,
    pub endpoint: String,
    pub t_ms: u64,
    pub status: String,
    pub client_id_digest_prefix: String,
    pub request_id: String,
}

#[derive(Debug, Serialize)]
pub struct GatewayAbuseRecord {
    pub schema_version: u16,
    pub t_ms: u64,
    pub endpoint: String,
    pub client_id_digest_prefix: String,
    pub reason_code: String,
    pub request_digest_prefix: String,
    pub request_id: String,
}

#[derive(Debug, Clone, Copy)]
struct TokenBucket {
    capacity: u32,
    refill_per_sec: u32,
    tokens_milli: u64,
    last_refill_ms: u64,
}

impl TokenBucket {
    fn new(cfg: RateLimitConfig, now_ms: u64) -> Self {
        Self {
            capacity: cfg.capacity,
            refill_per_sec: cfg.refill_per_sec,
            tokens_milli: u64::from(cfg.capacity) * 1000,
            last_refill_ms: now_ms,
        }
    }

    fn allow(&mut self, now_ms: u64) -> bool {
        let elapsed_ms = now_ms.saturating_sub(self.last_refill_ms);
        let refill_milli = elapsed_ms.saturating_mul(u64::from(self.refill_per_sec));
        let capacity_milli = u64::from(self.capacity) * 1000;
        self.tokens_milli = (self.tokens_milli + refill_milli).min(capacity_milli);
        self.last_refill_ms = now_ms;
        if self.tokens_milli >= 1000 {
            self.tokens_milli -= 1000;
            true
        } else {
            false
        }
    }
}

pub struct GatewayService {
    config: GatewayConfig,
    orchestrator: RuntimeOrchestrator,
    adapter: MockAdapter,
    decisions: VecDeque<proto::DecisionEvent>,
    token_buckets: BTreeMap<String, TokenBucket>,
    last_tick_wallclock_ms: Option<u64>,
}

impl GatewayService {
    pub fn new(config: GatewayConfig) -> Self {
        Self {
            config,
            orchestrator: RuntimeOrchestrator::new(),
            adapter: MockAdapter::default(),
            decisions: VecDeque::new(),
            token_buckets: BTreeMap::new(),
            last_tick_wallclock_ms: None,
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
            .token_capabilities_by_hash
            .get(&hash_token(token))
            .cloned()
            .ok_or(GatewayError::Unauthorized)?;
        if !caps.contains(capability) {
            return Err(GatewayError::Unauthorized);
        }
        Ok(caps)
    }

    fn enforce_rate_limit(
        &mut self,
        endpoint: &str,
        client_digest_prefix: &str,
    ) -> Result<(), GatewayError> {
        let Some(cfg) = self.config.rate_limits.get(endpoint).copied() else {
            return Ok(());
        };
        let key = format!("{endpoint}:{client_digest_prefix}");
        let now = now_ms();
        let bucket = self
            .token_buckets
            .entry(key)
            .or_insert_with(|| TokenBucket::new(cfg, now));
        if bucket.allow(now) {
            Ok(())
        } else {
            Err(GatewayError::RateLimited)
        }
    }

    fn validate_common(
        &self,
        schema_version: u32,
        run_id: &str,
        policy_digest: &[u8],
    ) -> Result<(), GatewayError> {
        if schema_version != SCHEMA_VERSION {
            return Err(GatewayError::UnsupportedVersion);
        }
        if run_id != self.config.run_id {
            return Err(GatewayError::SchemaInvalid);
        }
        if policy_digest != self.config.policy_graph_digest_prefix {
            return Err(GatewayError::PolicyDenied);
        }
        Ok(())
    }

    pub fn submit_control_frame(
        &mut self,
        token: &str,
        req: proto::ControlFrameSubmitRequest,
    ) -> Result<proto::ControlFrameSubmitResponse, GatewayError> {
        self.authorize(token, "submit")?;
        self.validate_common(
            req.schema_version,
            &req.run_id,
            &req.policy_graph_digest_prefix,
        )?;
        if req.payload_text_utf8.len() > SUBMIT_MAX_BYTES
            || req.intent_summary.chars().count() > SUMMARY_MAX_CHARS
        {
            return Err(GatewayError::MessageTooLarge);
        }
        if req.intent_summary.trim().is_empty() {
            return Err(GatewayError::SchemaInvalid);
        }
        let intent_kind = map_intent_kind(req.intent_kind)?;
        let channel = map_channel(req.channel)?;
        let text =
            std::str::from_utf8(&req.payload_text_utf8).map_err(|_| GatewayError::SchemaInvalid)?;
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
            .map_err(|_| GatewayError::Internal)?;
        self.last_tick_wallclock_ms = Some(now_ms());
        let evt = decision_to_event(
            &self.config.run_id,
            self.config.policy_graph_digest_prefix,
            &decision,
        );
        self.decisions.push_back(evt.clone());
        while self.decisions.len() > MAX_LIST {
            let _ = self.decisions.pop_front();
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
        self.validate_common(
            req.schema_version,
            &req.run_id,
            &req.policy_graph_digest_prefix,
        )?;
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
        self.validate_common(
            req.schema_version,
            &req.run_id,
            &req.policy_graph_digest_prefix,
        )?;
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
        self.validate_common(
            req.schema_version,
            &req.run_id,
            &req.policy_graph_digest_prefix,
        )?;
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
                .map_err(|_| GatewayError::Unavailable)?;
                serde_json::to_vec(&r).map_err(|_| GatewayError::Internal)?
            }
            "readiness-gate" => {
                let r = readiness_gate(
                    &self.config.workdir,
                    "test",
                    &self.config.workdir.join("out/gate_report.json"),
                )
                .map_err(|_| GatewayError::Unavailable)?;
                serde_json::to_vec(&r).map_err(|_| GatewayError::Internal)?
            }
            _ => {
                return Err(GatewayError::SchemaInvalid);
            }
        };
        Ok(proto::ReportResponse {
            schema_version: SCHEMA_VERSION,
            report_type: req.report_type,
            report_json_utf8: report_json,
            error: None,
        })
    }

    pub fn get_health(
        &self,
        token: &str,
        req: proto::HealthRequest,
    ) -> Result<proto::HealthResponseV1, GatewayError> {
        if req.schema_version != SCHEMA_VERSION {
            return Err(GatewayError::UnsupportedVersion);
        }
        if self.config.env_mode == "dev" && token.is_empty() {
            eprintln!("warning: unauthenticated health probe in dev mode");
        } else {
            self.authorize(token, "health:read")?;
        }

        let drift_alarms = read_recent_alarm_count(&self.config.workdir, &self.config.run_id, 128);
        let violations = read_recent_violations_count(&self.config.workdir, 128);
        let drift_status = if drift_alarms == 0 {
            proto::DriftStatus::Ok
        } else {
            proto::DriftStatus::Degraded
        };
        let emergency_active = self.orchestrator.is_emergency_active();
        let status = if emergency_active {
            proto::HealthStatus::Fail
        } else if drift_alarms > 0 || violations > 0 {
            proto::HealthStatus::Degraded
        } else {
            proto::HealthStatus::Ok
        };

        let last_tick_age_ms = self
            .last_tick_wallclock_ms
            .map(|ts| now_ms().saturating_sub(ts))
            .unwrap_or(u64::MAX);

        Ok(proto::HealthResponseV1 {
            schema_version: SCHEMA_VERSION,
            status: status as i32,
            run_id: self.config.run_id.clone(),
            strict_mode: self.config.strict_mode,
            policy_graph_digest_prefix: hex::encode(self.config.policy_graph_digest_prefix),
            manifest_digest_prefix: self.config.manifest_digest_prefix.clone(),
            drift_status: drift_status as i32,
            emergency_active,
            last_tick_age_ms,
            active_slots_summary: self.orchestrator.active_slots_summary(),
            recent_alarm_counts: Some(proto::AlarmCounts {
                drift_alarms,
                violations,
            }),
            error: None,
        })
    }

    pub fn handle_request(
        &mut self,
        req: proto::GatewayRequest,
        token: &str,
        client_id: &str,
    ) -> proto::GatewayResponse {
        let request_id = new_request_id(req.negotiated_version, token, client_id);
        let request_digest_prefix = digest_prefix(request_body_digest(&req).as_slice(), 16);
        let client_id_digest_prefix = if token.is_empty() {
            "unknown".to_string()
        } else {
            digest_prefix(hash_token(token).as_bytes(), 16)
        };

        let (endpoint, result) =
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if client_id == "__panic_test__" {
                    panic!("gateway panic test hook");
                }
                match req.payload {
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
                    Some(proto::gateway_request::Payload::Health(r)) => (
                        "health",
                        self.get_health(token, r)
                            .map(proto::gateway_response::Payload::Health),
                    ),
                    None => ("none", Err(GatewayError::SchemaInvalid)),
                }
            })) {
                Ok(outcome) => outcome,
                Err(_) => ("internal", Err(GatewayError::Internal)),
            };

        if let Ok(payload) = result {
            if let Err(err) = self.enforce_rate_limit(endpoint, &client_id_digest_prefix) {
                let _ = self.log_abuse(
                    endpoint,
                    &client_id_digest_prefix,
                    reason_code(&err),
                    &request_digest_prefix,
                    &request_id,
                );
                let _ = self.log_access(endpoint, "deny", &client_id_digest_prefix, &request_id);
                return safe_error_response(req.negotiated_version, err, request_id);
            }
            let _ = self.log_access(endpoint, "ok", &client_id_digest_prefix, &request_id);
            return proto::GatewayResponse {
                negotiated_version: req.negotiated_version,
                payload: Some(payload),
            };
        }

        let err = result.err().unwrap_or(GatewayError::Internal);
        let _ = self.log_abuse(
            endpoint,
            &client_id_digest_prefix,
            reason_code(&err),
            &request_digest_prefix,
            &request_id,
        );
        let _ = self.log_access(endpoint, "deny", &client_id_digest_prefix, &request_id);
        safe_error_response(req.negotiated_version, err, request_id)
    }

    fn log_access(
        &self,
        endpoint: &str,
        status: &str,
        client_id_digest_prefix: &str,
        request_id: &str,
    ) -> Result<(), GatewayError> {
        let rec = GatewayAccessRecord {
            schema_version: 1,
            endpoint: endpoint.to_string(),
            t_ms: now_ms(),
            status: status.to_string(),
            client_id_digest_prefix: client_id_digest_prefix.to_string(),
            request_id: request_id.to_string(),
        };
        append_jsonl(&self.config.access_log_path, &rec)
    }

    fn log_abuse(
        &self,
        endpoint: &str,
        client_id_digest_prefix: &str,
        reason: &str,
        request_digest_prefix: &str,
        request_id: &str,
    ) -> Result<(), GatewayError> {
        let rec = GatewayAbuseRecord {
            schema_version: 1,
            t_ms: now_ms(),
            endpoint: endpoint.to_string(),
            client_id_digest_prefix: client_id_digest_prefix.to_string(),
            reason_code: reason.to_string(),
            request_digest_prefix: request_digest_prefix.to_string(),
            request_id: request_id.to_string(),
        };
        append_jsonl(&self.config.abuse_log_path, &rec)
    }
}

fn append_jsonl<T: Serialize>(path: &Path, value: &T) -> Result<(), GatewayError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    let line = serde_json::to_string(value).map_err(|_| GatewayError::Internal)?;
    writeln!(f, "{line}")?;
    Ok(())
}

fn safe_error_response(
    negotiated_version: u32,
    err: GatewayError,
    request_id: String,
) -> proto::GatewayResponse {
    let (code, message) = map_err_code(&err);
    proto::GatewayResponse {
        negotiated_version,
        payload: Some(proto::gateway_response::Payload::Error(proto::Error {
            code,
            message: message.to_string(),
            request_id,
        })),
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
    proto::GatewayRequest::decode(body.as_slice()).map_err(|_| GatewayError::Proto)
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
    runtime_socket_audit_tick(&service.config.workdir);
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
    runtime_socket_audit_tick(&service.config.workdir);
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
    proto::GatewayRequest::decode(body.as_slice()).map_err(|_| GatewayError::Proto)
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
        _ => Err(GatewayError::SchemaInvalid),
    }
}

fn map_channel(channel: u32) -> Result<ChannelCode, GatewayError> {
    match channel {
        1 => Ok(ChannelCode::ExternalOutput),
        2 => Ok(ChannelCode::InternalThought),
        3 => Ok(ChannelCode::MemoryWrite),
        4 => Ok(ChannelCode::BrainStimulus),
        _ => Err(GatewayError::SchemaInvalid),
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

fn request_body_digest(req: &proto::GatewayRequest) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.gateway.request.v1");
    hasher.update(&req.encode_to_vec());
    *hasher.finalize().as_bytes()
}

fn digest_prefix(bytes: &[u8], chars: usize) -> String {
    hex::encode(bytes).chars().take(chars).collect()
}

pub fn hash_token(token: &str) -> String {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.gateway.auth.v1");
    hasher.update(token.as_bytes());
    hex::encode(hasher.finalize().as_bytes())
}

fn new_request_id(negotiated_version: u32, token: &str, client_id: &str) -> String {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.gateway.request_id.v1");
    hasher.update(&negotiated_version.to_le_bytes());
    hasher.update(token.as_bytes());
    hasher.update(client_id.as_bytes());
    hasher.update(&now_ms().to_le_bytes());
    digest_prefix(hasher.finalize().as_bytes(), REQUEST_ID_PREFIX)
}

fn map_err_code(err: &GatewayError) -> (u32, &'static str) {
    match err {
        GatewayError::Unauthorized => (ERR_AUTH_DENIED, "auth denied"),
        GatewayError::RateLimited => (ERR_RATE_LIMITED, "rate limit exceeded"),
        GatewayError::UnsupportedVersion => (ERR_VERSION_MISMATCH, "unsupported version"),
        GatewayError::MessageTooLarge => (ERR_TOO_LARGE, "request too large"),
        GatewayError::SchemaInvalid => (ERR_SCHEMA_INVALID, "schema invalid"),
        GatewayError::PolicyDenied => (ERR_POLICY_DENIED, "policy denied"),
        GatewayError::Unavailable => (ERR_UNAVAILABLE, "temporarily unavailable"),
        GatewayError::Internal | GatewayError::Io(_) | GatewayError::Proto => {
            (ERR_INTERNAL, "internal error")
        }
    }
}

fn reason_code(err: &GatewayError) -> &'static str {
    match err {
        GatewayError::Unauthorized => "AuthFail",
        GatewayError::RateLimited => "RateLimit",
        GatewayError::SchemaInvalid => "Malformed",
        GatewayError::MessageTooLarge => "TooLarge",
        GatewayError::UnsupportedVersion => "VersionMismatch",
        GatewayError::PolicyDenied => "PolicyDenied",
        GatewayError::Unavailable => "Unavailable",
        GatewayError::Internal | GatewayError::Io(_) | GatewayError::Proto => "Internal",
    }
}

fn extract_token(req: &proto::GatewayRequest) -> &str {
    match &req.payload {
        Some(proto::gateway_request::Payload::Handshake(h)) => h.auth_token.as_str(),
        Some(proto::gateway_request::Payload::Submit(r)) => r.auth_token.as_str(),
        Some(proto::gateway_request::Payload::Subscribe(r)) => r.auth_token.as_str(),
        Some(proto::gateway_request::Payload::EssQuery(r)) => r.auth_token.as_str(),
        Some(proto::gateway_request::Payload::Report(r)) => r.auth_token.as_str(),
        Some(proto::gateway_request::Payload::Health(r)) => r.auth_token.as_str(),
        _ => "",
    }
}

fn extract_client_id(req: &proto::GatewayRequest) -> &str {
    match &req.payload {
        Some(proto::gateway_request::Payload::Handshake(h)) => h.client_id.as_str(),
        Some(proto::gateway_request::Payload::Health(_)) => "health-client",
        _ => "anonymous",
    }
}

fn read_recent_alarm_count(workdir: &Path, run_id: &str, cap: usize) -> u32 {
    let path = workdir
        .join("world_shadow")
        .join(format!("{}_alarms.jsonl", run_id));
    let Ok(body) = std::fs::read_to_string(path) else {
        return 0;
    };
    let count = body.lines().take(cap).count();
    u32::try_from(count).unwrap_or(u32::MAX)
}

fn read_recent_violations_count(workdir: &Path, cap: usize) -> u32 {
    let path = workdir.join("network_violations.jsonl");
    let Ok(body) = std::fs::read_to_string(path) else {
        return 0;
    };
    let count = body.lines().take(cap).count();
    u32::try_from(count).unwrap_or(u32::MAX)
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
        GatewayTransport::Unix(default_unix_socket_path(Path::new(".ucf")))
    }
    #[cfg(windows)]
    {
        let _ = preferred_port;
        GatewayTransport::Pipe(default_windows_pipe_name())
    }
    #[cfg(not(any(unix, windows)))]
    {
        GatewayTransport::TcpLocal(preferred_port)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_hash_stable() {
        assert_eq!(hash_token("abc"), hash_token("abc"));
        assert_ne!(hash_token("abc"), hash_token("abcd"));
    }

    #[test]
    fn token_bucket_deterministic() {
        let cfg = RateLimitConfig {
            capacity: 2,
            refill_per_sec: 2,
        };
        let mut bucket = TokenBucket::new(cfg, 0);
        assert!(bucket.allow(0));
        assert!(bucket.allow(0));
        assert!(!bucket.allow(0));
        assert!(!bucket.allow(499));
        assert!(bucket.allow(500));
    }

    #[test]
    fn transport_kind_parse() {
        assert_eq!(
            TransportKind::parse("unix").expect("unix"),
            TransportKind::Unix
        );
        assert_eq!(
            TransportKind::parse("pipe").expect("pipe"),
            TransportKind::Pipe
        );
        assert_eq!(
            TransportKind::parse("tcp").expect("tcp"),
            TransportKind::Tcp
        );
        assert!(TransportKind::parse("bad").is_err());
    }
}
