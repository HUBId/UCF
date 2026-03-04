#![forbid(unsafe_code)]

use std::fs;
use std::io::{Read, Write};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpStream};
#[cfg(unix)]
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};

use prost::Message;
use serde::Deserialize;
use thiserror::Error;
use ucf_gateway::proto;

const SCHEMA_VERSION: u32 = 1;
const BOUNDED_MAX: u32 = 64;

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("usage: {0}")]
    Usage(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("proto decode: {0}")]
    Proto(String),
    #[error("gateway error {code}: {message}")]
    Gateway { code: u32, message: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Endpoint {
    Tcp(SocketAddr),
    Unix(PathBuf),
    Pipe(String),
}

impl Endpoint {
    pub fn default_local() -> Self {
        #[cfg(unix)]
        {
            return Self::Unix(PathBuf::from(".ucf/data/ipc/gateway.sock"));
        }
        #[cfg(windows)]
        {
            return Self::Pipe(r"\\.\pipe\ucf_gateway".to_string());
        }
        #[allow(unreachable_code)]
        Self::Tcp(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 44991))
    }

    pub fn parse(raw: &str) -> Result<Self, ClientError> {
        if let Some(addr) = raw.strip_prefix("tcp://") {
            let parsed: SocketAddr = addr
                .parse()
                .map_err(|_| ClientError::Usage("invalid tcp endpoint".to_string()))?;
            if !parsed.ip().is_loopback() {
                return Err(ClientError::Usage(
                    "endpoint must be loopback/local-only".to_string(),
                ));
            }
            return Ok(Self::Tcp(parsed));
        }
        if let Some(path) = raw.strip_prefix("unix://") {
            return Ok(Self::Unix(PathBuf::from(path)));
        }
        if let Some(name) = raw.strip_prefix("pipe://") {
            return Ok(Self::Pipe(name.to_string()));
        }
        if raw.contains(':') {
            let parsed: SocketAddr = raw
                .parse()
                .map_err(|_| ClientError::Usage("invalid endpoint".to_string()))?;
            if !parsed.ip().is_loopback() {
                return Err(ClientError::Usage(
                    "endpoint must be loopback/local-only".to_string(),
                ));
            }
            return Ok(Self::Tcp(parsed));
        }
        Ok(Self::Unix(PathBuf::from(raw)))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Submit { fixture: PathBuf },
    Stream { max: u32 },
    EssQuery { last: u32 },
    ReportExplainTick { tick: u64 },
    ReportReadinessGate,
    Health,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cli {
    pub endpoint: Endpoint,
    pub auth: String,
    pub command: Command,
}

#[derive(Debug, Deserialize)]
struct ControlFrameFixture {
    run_id: String,
    #[serde(default = "default_policy_digest")]
    policy_graph_digest_prefix_hex: String,
    tick: u64,
    #[serde(default)]
    window: u64,
    corr_id: u64,
    intent_id: u64,
    intent_kind: u32,
    channel: u32,
    intent_summary: String,
    payload_text_utf8: String,
}

fn default_policy_digest() -> String {
    "0102030405060708".to_string()
}

pub fn parse_cli(args: &[String]) -> Result<Cli, ClientError> {
    if args.len() < 2 {
        return Err(ClientError::Usage(usage()));
    }
    let endpoint = arg_value(args, "--endpoint")
        .map(|v| Endpoint::parse(&v))
        .transpose()?
        .unwrap_or_else(Endpoint::default_local);
    let auth = arg_value(args, "--auth").unwrap_or_default();

    match args[1].as_str() {
        "submit" => {
            let fixture = arg_value(args, "--fixture")
                .map(PathBuf::from)
                .ok_or_else(|| {
                    ClientError::Usage("submit requires --fixture <path>".to_string())
                })?;
            Ok(Cli {
                endpoint,
                auth,
                command: Command::Submit { fixture },
            })
        }
        "stream" => {
            let max = arg_u32(args, "--max").unwrap_or(10).min(BOUNDED_MAX);
            Ok(Cli {
                endpoint,
                auth,
                command: Command::Stream { max },
            })
        }
        "ess" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("");
            if sub != "query" {
                return Err(ClientError::Usage(
                    "usage: ucf-client ess query --last <n>".to_string(),
                ));
            }
            let last = arg_u32(args, "--last").unwrap_or(32).min(BOUNDED_MAX);
            Ok(Cli {
                endpoint,
                auth,
                command: Command::EssQuery { last },
            })
        }
        "report" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("");
            match sub {
                "explain-tick" => {
                    let tick = arg_u64(args, "--t").ok_or_else(|| {
                        ClientError::Usage("report explain-tick requires --t <tick>".to_string())
                    })?;
                    Ok(Cli {
                        endpoint,
                        auth,
                        command: Command::ReportExplainTick { tick },
                    })
                }
                "readiness-gate" => {
                    if !has_flag(args, "--latest") {
                        return Err(ClientError::Usage(
                            "report readiness-gate requires --latest".to_string(),
                        ));
                    }
                    Ok(Cli {
                        endpoint,
                        auth,
                        command: Command::ReportReadinessGate,
                    })
                }
                _ => Err(ClientError::Usage(usage())),
            }
        }
        "health" => Ok(Cli {
            endpoint,
            auth,
            command: Command::Health,
        }),
        _ => Err(ClientError::Usage(usage())),
    }
}

pub fn run(cli: Cli) -> Result<String, ClientError> {
    match cli.command {
        Command::Submit { fixture } => submit_fixture(&cli.endpoint, &cli.auth, &fixture),
        Command::Stream { max } => stream_decisions(&cli.endpoint, &cli.auth, max),
        Command::EssQuery { last } => ess_query(&cli.endpoint, &cli.auth, last),
        Command::ReportExplainTick { tick } => {
            report(&cli.endpoint, &cli.auth, "explain-tick", tick)
        }
        Command::ReportReadinessGate => report(&cli.endpoint, &cli.auth, "readiness-gate", 0),
        Command::Health => health(&cli.endpoint, &cli.auth),
    }
}

fn health(endpoint: &Endpoint, auth: &str) -> Result<String, ClientError> {
    let req = proto::GatewayRequest {
        negotiated_version: 1,
        payload: Some(proto::gateway_request::Payload::Health(
            proto::HealthRequest {
                schema_version: SCHEMA_VERSION,
                auth_token: auth.to_string(),
            },
        )),
    };
    let resp = send(endpoint, &req)?;
    match resp.payload {
        Some(proto::gateway_response::Payload::Health(r)) => {
            let value = serde_json::json!({
                "schema_version": r.schema_version,
                "status": r.status,
                "run_id": r.run_id,
                "strict_mode": r.strict_mode,
                "policy_graph_digest_prefix": r.policy_graph_digest_prefix,
                "manifest_digest_prefix": r.manifest_digest_prefix,
                "drift_status": r.drift_status,
                "emergency_active": r.emergency_active,
                "last_tick_age_ms": r.last_tick_age_ms,
                "active_slots_summary": r.active_slots_summary,
                "recent_alarm_counts": {
                    "drift_alarms": r
                        .recent_alarm_counts
                        .as_ref()
                        .map(|a| a.drift_alarms)
                        .unwrap_or_default(),
                    "violations": r
                        .recent_alarm_counts
                        .as_ref()
                        .map(|a| a.violations)
                        .unwrap_or_default(),
                }
            });
            serde_json::to_string_pretty(&value).map_err(ClientError::from)
        }
        Some(proto::gateway_response::Payload::Error(e)) => Err(ClientError::Gateway {
            code: e.code,
            message: e.message,
        }),
        _ => Err(ClientError::Proto(
            "unexpected response payload for health".to_string(),
        )),
    }
}

fn submit_fixture(endpoint: &Endpoint, auth: &str, fixture: &Path) -> Result<String, ClientError> {
    let fixture: ControlFrameFixture = serde_json::from_str(&fs::read_to_string(fixture)?)?;
    let req = proto::GatewayRequest {
        negotiated_version: 1,
        payload: Some(proto::gateway_request::Payload::Submit(
            proto::ControlFrameSubmitRequest {
                schema_version: SCHEMA_VERSION,
                run_id: fixture.run_id,
                policy_graph_digest_prefix: hex_to_bytes(&fixture.policy_graph_digest_prefix_hex)?,
                tick: fixture.tick,
                window: fixture.window,
                corr_id: fixture.corr_id,
                intent_id: fixture.intent_id,
                intent_kind: fixture.intent_kind,
                channel: fixture.channel,
                intent_summary: fixture.intent_summary,
                payload_text_utf8: fixture.payload_text_utf8.into_bytes(),
                auth_token: auth.to_string(),
            },
        )),
    };
    let resp = send(endpoint, &req)?;
    match resp.payload {
        Some(proto::gateway_response::Payload::Submit(r)) => {
            let decision = r
                .decision
                .map(|d| {
                    format!(
                        "decision={} reason={} tick={}",
                        d.decision_code, d.reason_code, d.tick
                    )
                })
                .unwrap_or_else(|| "decision=none".to_string());
            Ok(format!(
                "submit_ok run_id={} frame_digest={} {}",
                r.run_id,
                bytes_to_hex(&r.frame_digest),
                decision
            ))
        }
        Some(proto::gateway_response::Payload::Error(e)) => Err(ClientError::Gateway {
            code: e.code,
            message: e.message,
        }),
        _ => Err(ClientError::Proto(
            "unexpected response payload for submit".to_string(),
        )),
    }
}

fn stream_decisions(endpoint: &Endpoint, auth: &str, max: u32) -> Result<String, ClientError> {
    let req = proto::GatewayRequest {
        negotiated_version: 1,
        payload: Some(proto::gateway_request::Payload::Subscribe(
            proto::DecisionStreamSubscribeRequest {
                schema_version: SCHEMA_VERSION,
                run_id: "run-test".to_string(),
                policy_graph_digest_prefix: hex_to_bytes(&default_policy_digest())?,
                max_events: max.min(BOUNDED_MAX),
                auth_token: auth.to_string(),
            },
        )),
    };
    let resp = send(endpoint, &req)?;
    match resp.payload {
        Some(proto::gateway_response::Payload::Subscribe(r)) => {
            let rows = r
                .events
                .iter()
                .map(|e| {
                    format!(
                        "tick={} corr={} decision={} reason={}",
                        e.tick, e.corr_id, e.decision_code, e.reason_code
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            Ok(format!("events={}\n{}", r.events.len(), rows))
        }
        Some(proto::gateway_response::Payload::Error(e)) => Err(ClientError::Gateway {
            code: e.code,
            message: e.message,
        }),
        _ => Err(ClientError::Proto(
            "unexpected response payload for stream".to_string(),
        )),
    }
}

fn ess_query(endpoint: &Endpoint, auth: &str, last: u32) -> Result<String, ClientError> {
    let req = proto::GatewayRequest {
        negotiated_version: 1,
        payload: Some(proto::gateway_request::Payload::EssQuery(
            proto::EssQueryRequest {
                schema_version: SCHEMA_VERSION,
                run_id: "run-test".to_string(),
                policy_graph_digest_prefix: hex_to_bytes(&default_policy_digest())?,
                max_records: last.min(BOUNDED_MAX),
                auth_token: auth.to_string(),
            },
        )),
    };
    let resp = send(endpoint, &req)?;
    match resp.payload {
        Some(proto::gateway_response::Payload::EssQuery(r)) => {
            let rows = r
                .summaries
                .iter()
                .map(|s| {
                    format!(
                        "id={} tick={} kind={} corr={}",
                        s.experience_id, s.tick, s.kind, s.corr_id
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            Ok(format!("summaries={}\n{}", r.summaries.len(), rows))
        }
        Some(proto::gateway_response::Payload::Error(e)) => Err(ClientError::Gateway {
            code: e.code,
            message: e.message,
        }),
        _ => Err(ClientError::Proto(
            "unexpected response payload for ess query".to_string(),
        )),
    }
}

fn report(
    endpoint: &Endpoint,
    auth: &str,
    report_type: &str,
    tick: u64,
) -> Result<String, ClientError> {
    let req = proto::GatewayRequest {
        negotiated_version: 1,
        payload: Some(proto::gateway_request::Payload::Report(
            proto::ReportRequest {
                schema_version: SCHEMA_VERSION,
                run_id: "run-test".to_string(),
                policy_graph_digest_prefix: hex_to_bytes(&default_policy_digest())?,
                report_type: report_type.to_string(),
                tick,
                auth_token: auth.to_string(),
            },
        )),
    };
    let resp = send(endpoint, &req)?;
    match resp.payload {
        Some(proto::gateway_response::Payload::Report(r)) => {
            let utf8 = std::str::from_utf8(&r.report_json_utf8)
                .map_err(|e| ClientError::Proto(e.to_string()))?
                .to_string();
            let bounded = utf8.chars().take(4000).collect::<String>();
            Ok(format!("report_type={}\n{}", r.report_type, bounded))
        }
        Some(proto::gateway_response::Payload::Error(e)) => Err(ClientError::Gateway {
            code: e.code,
            message: e.message,
        }),
        _ => Err(ClientError::Proto(
            "unexpected response payload for report".to_string(),
        )),
    }
}

fn send(
    endpoint: &Endpoint,
    req: &proto::GatewayRequest,
) -> Result<proto::GatewayResponse, ClientError> {
    match endpoint {
        Endpoint::Tcp(addr) => {
            let mut stream = TcpStream::connect(addr)?;
            send_over_stream(&mut stream, req)
        }
        Endpoint::Unix(path) => {
            #[cfg(unix)]
            {
                let mut stream = UnixStream::connect(path)?;
                send_over_stream(&mut stream, req)
            }
            #[cfg(not(unix))]
            {
                let _ = path;
                Err(ClientError::Usage(
                    "unix sockets are not supported on this platform".to_string(),
                ))
            }
        }
        Endpoint::Pipe(name) => Err(ClientError::Usage(format!(
            "named pipe endpoint is not supported in this build: {name}"
        ))),
    }
}

fn send_over_stream<S: Read + Write>(
    stream: &mut S,
    req: &proto::GatewayRequest,
) -> Result<proto::GatewayResponse, ClientError> {
    let body = req.encode_to_vec();
    let len = u32::try_from(body.len())
        .map_err(|_| ClientError::Proto("message too large".to_string()))?;
    stream.write_all(&len.to_le_bytes())?;
    stream.write_all(&body)?;
    stream.flush()?;

    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let n = u32::from_le_bytes(len_buf) as usize;
    let mut body = vec![0u8; n];
    stream.read_exact(&mut body)?;
    proto::GatewayResponse::decode(body.as_slice()).map_err(|e| ClientError::Proto(e.to_string()))
}

fn usage() -> String {
    "ucf-client [--endpoint tcp://127.0.0.1:44991|unix://.ucf/data/ipc/gateway.sock|pipe://./pipe/ucf_gateway] [--auth token] <submit|stream|ess|report|health> ...".to_string()
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find_map(|w| (w[0] == flag).then(|| w[1].clone()))
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|arg| arg == flag)
}

fn arg_u32(args: &[String], flag: &str) -> Option<u32> {
    arg_value(args, flag).and_then(|v| v.parse::<u32>().ok())
}

fn arg_u64(args: &[String], flag: &str) -> Option<u64> {
    arg_value(args, flag).and_then(|v| v.parse::<u64>().ok())
}

fn hex_to_bytes(input: &str) -> Result<Vec<u8>, ClientError> {
    if !input.len().is_multiple_of(2) {
        return Err(ClientError::Usage("hex length must be even".to_string()));
    }
    (0..input.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&input[i..i + 2], 16)
                .map_err(|_| ClientError::Usage("invalid hex".to_string()))
        })
        .collect()
}

fn bytes_to_hex(data: &[u8]) -> String {
    data.iter().map(|b| format!("{b:02x}")).collect::<String>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_submit_cli() {
        let args = vec![
            "ucf-client".to_string(),
            "submit".to_string(),
            "--fixture".to_string(),
            "fixtures/client/controlframe_min.json".to_string(),
            "--endpoint".to_string(),
            "tcp://127.0.0.1:44991".to_string(),
        ];
        let cli = parse_cli(&args).expect("parse");
        assert!(matches!(cli.endpoint, Endpoint::Tcp(_)));
        assert_eq!(
            cli.command,
            Command::Submit {
                fixture: PathBuf::from("fixtures/client/controlframe_min.json")
            }
        );
    }

    #[test]
    fn parse_bounded_stream_max() {
        let args = vec![
            "ucf-client".to_string(),
            "stream".to_string(),
            "--max".to_string(),
            "1000".to_string(),
        ];
        let cli = parse_cli(&args).expect("parse");
        assert_eq!(cli.command, Command::Stream { max: BOUNDED_MAX });
    }

    #[test]
    fn decode_fixture() {
        let fixture = serde_json::from_str::<ControlFrameFixture>(
            r#"{
            "run_id":"run-test",
            "policy_graph_digest_prefix_hex":"0102030405060708",
            "tick":1,
            "corr_id":11,
            "intent_id":12,
            "intent_kind":1,
            "channel":1,
            "intent_summary":"hello",
            "payload_text_utf8":"safe"
        }"#,
        )
        .expect("fixture");
        assert_eq!(fixture.tick, 1);
        assert_eq!(fixture.window, 0);
    }
}
