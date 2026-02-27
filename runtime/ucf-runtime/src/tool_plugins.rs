use std::collections::BTreeMap;
use std::path::Path;

use sha2::{Digest, Sha256};
use ucf_policy::capability::{CapabilityScope, CapabilityToken};
use ucf_policy::gem::{AuthorizationOutcome, ToolRequest, ToolResultSummary, ToolStatus};

use crate::sandbox_fs::{FsCapabilityKind, FsCapabilityToken, SandboxFs};

pub const MAX_TOOL_ARGS_BYTES: usize = 4 * 1024;
pub const MAX_TOOL_PREVIEW_BYTES: usize = 128;

pub type ToolId = String;
pub type ToolClassId = String;

pub trait ToolPlugin: Send + Sync {
    fn tool_id(&self) -> ToolId;
    fn tool_class(&self) -> ToolClassId;
    fn execute(
        &self,
        req: ToolExecRequest,
        caps: &CapabilityToken,
        sandbox: &SandboxEnv<'_>,
    ) -> ToolExecResponse;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolExecRequest {
    pub request_id: u64,
    pub plan_digest: [u8; 32],
    pub args: Vec<u8>,
    pub allowed_roots: Vec<String>,
    pub max_bytes_out: u32,
    pub timeout_ms: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolExecStatus {
    Ok,
    Denied,
    Timeout,
    Error,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolExecResponse {
    pub status: ToolExecStatus,
    pub result_digest: [u8; 32],
    pub preview: String,
    pub bytes_out: u32,
    pub error_code: Option<String>,
}

impl ToolExecResponse {
    fn denied(code: &str) -> Self {
        let result_digest = digest_bytes(code.as_bytes());
        Self {
            status: ToolExecStatus::Denied,
            result_digest,
            preview: code.to_string(),
            bytes_out: 0,
            error_code: Some(code.to_string()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapabilityTokenBinding {
    pub token: CapabilityToken,
    pub plan_digest: [u8; 32],
}

pub struct SandboxEnv<'a> {
    pub fs: Option<&'a SandboxFs>,
}

#[derive(Default)]
pub struct ToolPluginRegistry {
    plugins: BTreeMap<(ToolId, ToolClassId), Box<dyn ToolPlugin>>,
}

impl ToolPluginRegistry {
    pub fn with_builtin_stubs() -> Self {
        let mut this = Self::default();
        this.register(Box::new(EchoTool));
        this.register(Box::new(FileReadTool));
        this.register(Box::new(MathTool));
        this
    }

    pub fn register(&mut self, plugin: Box<dyn ToolPlugin>) {
        let key = (plugin.tool_id(), plugin.tool_class());
        self.plugins.insert(key, plugin);
    }

    pub fn execute(
        &self,
        tool_id: &str,
        tool_class_id: &str,
        req: ToolExecRequest,
        binding: &CapabilityTokenBinding,
        sandbox: &SandboxEnv<'_>,
    ) -> ToolExecResponse {
        if binding.plan_digest != req.plan_digest {
            return ToolExecResponse::denied("tool_plan_digest_mismatch");
        }
        if req.timeout_ms == 0 {
            return ToolExecResponse {
                status: ToolExecStatus::Timeout,
                result_digest: digest_bytes(b"tool_timeout"),
                preview: "timeout".to_string(),
                bytes_out: 0,
                error_code: Some("tool_timeout".to_string()),
            };
        }
        let key = (tool_id.to_string(), tool_class_id.to_string());
        let Some(plugin) = self.plugins.get(&key) else {
            return ToolExecResponse::denied("tool_plugin_missing");
        };

        let bounded_req = ToolExecRequest {
            args: bound_vec(req.args, MAX_TOOL_ARGS_BYTES),
            allowed_roots: req.allowed_roots,
            ..req
        };
        let resp = plugin.execute(bounded_req, &binding.token, sandbox);
        sanitize_response(resp)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolPluginExecutionAudit {
    pub auth: AuthorizationOutcome,
    pub result: ToolResultSummary,
    pub result_digest: [u8; 32],
    pub preview: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PluginDispatchSpec {
    pub tool_id: String,
    pub tool_class_id: String,
    pub plan_digest: [u8; 32],
    pub args: Vec<u8>,
    pub binding: CapabilityTokenBinding,
}

pub fn run_plugin_tool(
    registry: &ToolPluginRegistry,
    request: &ToolRequest,
    dispatch: PluginDispatchSpec,
    sandbox: &SandboxEnv<'_>,
) -> ToolPluginExecutionAudit {
    let allowed_roots = extract_allowed_roots(&dispatch.binding.token.scope);
    let req = ToolExecRequest {
        request_id: request.id,
        plan_digest: dispatch.plan_digest,
        args: dispatch.args,
        allowed_roots,
        max_bytes_out: request.payload_hint.bytes_out.unwrap_or(512),
        timeout_ms: 100,
    };
    let resp = registry.execute(
        &dispatch.tool_id,
        &dispatch.tool_class_id,
        req,
        &dispatch.binding,
        sandbox,
    );
    let status = match resp.status {
        ToolExecStatus::Ok => ToolStatus::AllowedExecuted,
        ToolExecStatus::Denied => ToolStatus::Denied,
        ToolExecStatus::Timeout => ToolStatus::RateLimited,
        ToolExecStatus::Error => ToolStatus::Failed,
    };
    let error_code = resp.error_code.clone();
    ToolPluginExecutionAudit {
        auth: AuthorizationOutcome::Allowed {
            token_digest: dispatch.binding.token.token_digest,
        },
        result: ToolResultSummary {
            status,
            bytes_out: Some(resp.bytes_out),
            bytes_in: request.payload_hint.bytes_in,
            error_code,
            finished_at_t: request.requested_at_t,
        },
        result_digest: resp.result_digest,
        preview: resp.preview,
    }
}

pub struct EchoTool;

impl ToolPlugin for EchoTool {
    fn tool_id(&self) -> ToolId {
        "external_api".to_string()
    }

    fn tool_class(&self) -> ToolClassId {
        "external_output".to_string()
    }

    fn execute(
        &self,
        req: ToolExecRequest,
        _caps: &CapabilityToken,
        _sandbox: &SandboxEnv<'_>,
    ) -> ToolExecResponse {
        let digest = digest_bytes(&req.args);
        let preview = format!("echo:{}", hex::encode(&digest[..8]));
        ToolExecResponse {
            status: ToolExecStatus::Ok,
            result_digest: digest,
            preview,
            bytes_out: 16,
            error_code: None,
        }
    }
}

pub struct FileReadTool;

impl ToolPlugin for FileReadTool {
    fn tool_id(&self) -> ToolId {
        "file_read".to_string()
    }

    fn tool_class(&self) -> ToolClassId {
        "memory_write".to_string()
    }

    fn execute(
        &self,
        req: ToolExecRequest,
        _caps: &CapabilityToken,
        sandbox: &SandboxEnv<'_>,
    ) -> ToolExecResponse {
        let Some(fs) = sandbox.fs else {
            return ToolExecResponse::denied("sandbox_fs_missing");
        };
        let arg_text = String::from_utf8_lossy(&req.args);
        let rel_path = parse_arg(&arg_text, "path").unwrap_or("".to_string());
        let root_id = parse_arg(&arg_text, "root_id").unwrap_or_default();
        if !req.allowed_roots.iter().any(|r| r == &root_id) {
            return ToolExecResponse::denied("root_not_allowed");
        }
        let token = FsCapabilityToken {
            kind: FsCapabilityKind::FileRead,
            root_id,
        };
        match fs.read_to_string(&token, Path::new(&rel_path)) {
            Ok(text) => {
                let digest = digest_bytes(text.as_bytes());
                let preview = bound_preview(&text, MAX_TOOL_PREVIEW_BYTES);
                ToolExecResponse {
                    status: ToolExecStatus::Ok,
                    result_digest: digest,
                    preview,
                    bytes_out: text.len() as u32,
                    error_code: None,
                }
            }
            Err(_) => ToolExecResponse::denied("file_read_denied"),
        }
    }
}

pub struct MathTool;

impl ToolPlugin for MathTool {
    fn tool_id(&self) -> ToolId {
        "internal_thought".to_string()
    }

    fn tool_class(&self) -> ToolClassId {
        "internal".to_string()
    }

    fn execute(
        &self,
        req: ToolExecRequest,
        _caps: &CapabilityToken,
        _sandbox: &SandboxEnv<'_>,
    ) -> ToolExecResponse {
        let text = String::from_utf8_lossy(&req.args);
        let a = parse_arg(&text, "a")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(0);
        let b = parse_arg(&text, "b")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(0);
        let sum = a.saturating_add(b);
        let out = sum.to_string();
        ToolExecResponse {
            status: ToolExecStatus::Ok,
            result_digest: digest_bytes(out.as_bytes()),
            preview: out,
            bytes_out: 8,
            error_code: None,
        }
    }
}

fn sanitize_response(resp: ToolExecResponse) -> ToolExecResponse {
    let mut preview = bound_preview(&resp.preview, MAX_TOOL_PREVIEW_BYTES);
    preview = preview.replace(['\n', '\r'], " ");
    ToolExecResponse {
        preview,
        bytes_out: resp.bytes_out.min(MAX_TOOL_ARGS_BYTES as u32),
        ..resp
    }
}

fn extract_allowed_roots(scope: &CapabilityScope) -> Vec<String> {
    match scope {
        CapabilityScope::Paths(items) => items
            .iter()
            .map(|item| item.split(':').next().unwrap_or(item.as_str()).to_string())
            .collect(),
        _ => Vec::new(),
    }
}

fn parse_arg(input: &str, key: &str) -> Option<String> {
    input.lines().find_map(|line| {
        let (k, v) = line.split_once('=')?;
        if k.trim() == key {
            Some(v.trim().to_string())
        } else {
            None
        }
    })
}

fn bound_vec(mut bytes: Vec<u8>, max: usize) -> Vec<u8> {
    if bytes.len() > max {
        bytes.truncate(max);
    }
    bytes
}

fn bound_preview(text: &str, max_bytes: usize) -> String {
    let mut out = String::new();
    for ch in text.chars() {
        let len = ch.len_utf8();
        if out.len().saturating_add(len) > max_bytes {
            out.push('…');
            break;
        }
        if ch.is_control() && ch != ' ' {
            out.push('�');
        } else {
            out.push(ch);
        }
    }
    out
}

fn digest_bytes(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

pub fn compact_tool_result_note(result_digest: [u8; 32], preview: &str) -> String {
    let mut note = format!(
        "result_digest={} preview={} ",
        hex::encode(result_digest),
        bound_preview(preview, 48)
    );
    note.truncate(120);
    note
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use ucf_policy::capability::{CapabilityKind, CapabilityLimits};
    use ucf_policy::gem::PayloadHint;

    fn mk_token(scope: CapabilityScope) -> CapabilityToken {
        CapabilityToken::issue(
            CapabilityKind::FileRead,
            scope,
            CapabilityLimits {
                max_calls_per_window: 2,
                window_ticks: 64,
                max_bytes_out: Some(4096),
                max_bytes_in: Some(4096),
                max_concurrent: 1,
            },
            "governor",
            1,
            Some(10),
        )
    }

    #[test]
    fn token_binding_prevents_misuse() {
        let registry = ToolPluginRegistry::with_builtin_stubs();
        let token = mk_token(CapabilityScope::Paths(vec!["test_root".to_string()]));
        let binding = CapabilityTokenBinding {
            token,
            plan_digest: [7; 32],
        };
        let req = ToolExecRequest {
            request_id: 9,
            plan_digest: [8; 32],
            args: b"path=x".to_vec(),
            allowed_roots: vec!["test_root".to_string()],
            max_bytes_out: 10,
            timeout_ms: 100,
        };
        let resp = registry.execute(
            "file_read",
            "memory_write",
            req,
            &binding,
            &SandboxEnv { fs: None },
        );
        assert_eq!(resp.status, ToolExecStatus::Denied);
        assert_eq!(
            resp.error_code.as_deref(),
            Some("tool_plan_digest_mismatch")
        );
    }

    #[test]
    fn file_tool_blocked_outside_root() {
        let tmp = tempfile::tempdir().expect("tmp");
        let fs_env = SandboxFs::new(vec![("allowed".to_string(), PathBuf::from(tmp.path()))]);
        let registry = ToolPluginRegistry::with_builtin_stubs();
        let binding = CapabilityTokenBinding {
            token: mk_token(CapabilityScope::Paths(vec!["allowed".to_string()])),
            plan_digest: [3; 32],
        };
        let req = ToolExecRequest {
            request_id: 2,
            plan_digest: [3; 32],
            args: b"root_id=forbidden\npath=inside.txt".to_vec(),
            allowed_roots: vec!["allowed".to_string()],
            max_bytes_out: 100,
            timeout_ms: 50,
        };
        let resp = registry.execute(
            "file_read",
            "memory_write",
            req,
            &binding,
            &SandboxEnv { fs: Some(&fs_env) },
        );
        assert_eq!(resp.status, ToolExecStatus::Denied);
        assert_eq!(resp.error_code.as_deref(), Some("root_not_allowed"));
    }

    #[test]
    fn echo_results_are_bounded_and_deterministic() {
        let registry = ToolPluginRegistry::with_builtin_stubs();
        let token = CapabilityToken::issue(
            CapabilityKind::ExternalApi,
            CapabilityScope::ApiNames(vec!["external_output".to_string()]),
            CapabilityLimits {
                max_calls_per_window: 2,
                window_ticks: 64,
                max_bytes_out: Some(4096),
                max_bytes_in: Some(4096),
                max_concurrent: 1,
            },
            "governor",
            1,
            Some(10),
        );
        let binding = CapabilityTokenBinding {
            token,
            plan_digest: [5; 32],
        };
        let args = vec![b'a'; MAX_TOOL_ARGS_BYTES + 20];
        let req = ToolExecRequest {
            request_id: 3,
            plan_digest: [5; 32],
            args,
            allowed_roots: vec![],
            max_bytes_out: 16,
            timeout_ms: 10,
        };
        let a = registry.execute(
            "external_api",
            "external_output",
            req.clone(),
            &binding,
            &SandboxEnv { fs: None },
        );
        let b = registry.execute(
            "external_api",
            "external_output",
            req,
            &binding,
            &SandboxEnv { fs: None },
        );
        assert_eq!(a, b);
        assert!(a.preview.len() <= MAX_TOOL_PREVIEW_BYTES + 5);
    }

    #[test]
    fn run_plugin_tool_maps_status_and_keeps_digest_preview() {
        let registry = ToolPluginRegistry::with_builtin_stubs();
        let token = CapabilityToken::issue(
            CapabilityKind::ExternalApi,
            CapabilityScope::ApiNames(vec!["external_output".to_string()]),
            CapabilityLimits {
                max_calls_per_window: 2,
                window_ticks: 64,
                max_bytes_out: Some(4096),
                max_bytes_in: Some(4096),
                max_concurrent: 1,
            },
            "governor",
            1,
            Some(10),
        );
        let request = ToolRequest {
            id: 1,
            kind: CapabilityKind::ExternalApi,
            target: "external_output".to_string(),
            payload_hint: PayloadHint {
                bytes_out: Some(12),
                bytes_in: Some(1),
            },
            requested_at_t: 2,
            decision_id: 3,
            evidence_chain_digest: [0; 32],
            candidate_id: None,
            tool_intent_digest: None,
        };
        let audit = run_plugin_tool(
            &registry,
            &request,
            PluginDispatchSpec {
                tool_id: "external_api".to_string(),
                tool_class_id: "external_output".to_string(),
                plan_digest: [9; 32],
                args: b"a=1\nb=2".to_vec(),
                binding: CapabilityTokenBinding {
                    token,
                    plan_digest: [9; 32],
                },
            },
            &SandboxEnv { fs: None },
        );
        assert!(matches!(audit.auth, AuthorizationOutcome::Allowed { .. }));
        assert_eq!(audit.result.status, ToolStatus::AllowedExecuted);
        assert!(!audit.preview.is_empty());
        assert_ne!(audit.result_digest, [0; 32]);
    }
}
