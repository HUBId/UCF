use sha2::{Digest, Sha256};

use crate::gem::{PayloadHint, ToolRequest};

pub type PolicyModuleId = &'static str;

pub const MAX_SCOPE_ITEMS: usize = 16;
pub const MAX_SCOPE_ITEM_LEN: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CapabilityKind {
    NetHttp,
    FileRead,
    FileWrite,
    ProcessExec,
    ClipboardRead,
    ClipboardWrite,
    UiAutomation,
    ExternalApi,
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CapabilityScope {
    Domains(Vec<String>),
    Paths(Vec<String>),
    ApiNames(Vec<String>),
    All,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapabilityLimits {
    pub max_calls_per_window: u32,
    pub window_ticks: u64,
    pub max_bytes_out: Option<u32>,
    pub max_bytes_in: Option<u32>,
    pub max_concurrent: u16,
}

impl CapabilityLimits {
    pub fn clamped(mut self) -> Self {
        self.max_calls_per_window = self.max_calls_per_window.clamp(1, 10_000);
        self.window_ticks = self.window_ticks.clamp(1, 1_000_000);
        self.max_bytes_out = self.max_bytes_out.map(|v| v.clamp(1, 8_388_608));
        self.max_bytes_in = self.max_bytes_in.map(|v| v.clamp(1, 8_388_608));
        self.max_concurrent = self.max_concurrent.clamp(1, 256);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapabilityToken {
    pub kind: CapabilityKind,
    pub scope: CapabilityScope,
    pub limits: CapabilityLimits,
    pub issued_by: PolicyModuleId,
    pub issued_at_t: u64,
    pub expires_at_t: Option<u64>,
    pub token_digest: [u8; 32],
}

impl CapabilityToken {
    pub fn issue(
        kind: CapabilityKind,
        scope: CapabilityScope,
        limits: CapabilityLimits,
        issued_by: PolicyModuleId,
        issued_at_t: u64,
        expires_at_t: Option<u64>,
    ) -> Self {
        let scope = scope.clamped();
        let limits = limits.clamped();
        let token_digest =
            digest_token(&kind, &scope, &limits, issued_by, issued_at_t, expires_at_t);
        Self {
            kind,
            scope,
            limits,
            issued_by,
            issued_at_t,
            expires_at_t,
            token_digest,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapabilitySet {
    pub tokens: Vec<CapabilityToken>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapabilityDenyReason {
    MissingToken,
    KindMismatch,
    ScopeMismatch,
    Expired,
    DecisionMissing,
    ScopeAllDisallowed,
    ByteLimitExceeded,
}

impl CapabilitySet {
    pub fn empty() -> Self {
        Self { tokens: Vec::new() }
    }

    pub fn allows<'a>(
        &'a self,
        req: &ToolRequest,
        now_t: u64,
    ) -> Result<&'a CapabilityToken, CapabilityDenyReason> {
        if req.decision_id == 0 {
            return Err(CapabilityDenyReason::DecisionMissing);
        }

        let mut saw_kind = false;
        for token in &self.tokens {
            if token.kind != req.kind {
                continue;
            }
            saw_kind = true;
            if token.expires_at_t.is_some_and(|exp| now_t > exp) {
                continue;
            }
            if matches!(token.scope, CapabilityScope::All) {
                return Err(CapabilityDenyReason::ScopeAllDisallowed);
            }
            if !token.scope.matches(req.target.as_str()) {
                continue;
            }
            if exceeds_byte_limit(req.payload_hint, token.limits) {
                return Err(CapabilityDenyReason::ByteLimitExceeded);
            }
            return Ok(token);
        }

        if saw_kind {
            if self.tokens.iter().any(|token| {
                token.kind == req.kind && token.expires_at_t.is_some_and(|exp| now_t > exp)
            }) {
                return Err(CapabilityDenyReason::Expired);
            }
            return Err(CapabilityDenyReason::ScopeMismatch);
        }
        if self.tokens.iter().any(|token| token.kind != req.kind) {
            return Err(CapabilityDenyReason::KindMismatch);
        }
        Err(CapabilityDenyReason::MissingToken)
    }
}

impl CapabilityScope {
    pub fn clamped(self) -> Self {
        fn clamp_items(items: Vec<String>) -> Vec<String> {
            items
                .into_iter()
                .take(MAX_SCOPE_ITEMS)
                .map(|s| s.chars().take(MAX_SCOPE_ITEM_LEN).collect())
                .collect()
        }
        match self {
            Self::Domains(items) => Self::Domains(clamp_items(items)),
            Self::Paths(items) => Self::Paths(clamp_items(items)),
            Self::ApiNames(items) => Self::ApiNames(clamp_items(items)),
            Self::All => Self::All,
        }
    }

    fn matches(&self, target: &str) -> bool {
        match self {
            Self::Domains(domains) | Self::Paths(domains) | Self::ApiNames(domains) => {
                domains.iter().any(|domain| domain == target)
            }
            Self::All => true,
        }
    }
}

fn exceeds_byte_limit(hint: PayloadHint, limits: CapabilityLimits) -> bool {
    limits
        .max_bytes_out
        .is_some_and(|max| hint.bytes_out.unwrap_or(0) > max)
        || limits
            .max_bytes_in
            .is_some_and(|max| hint.bytes_in.unwrap_or(0) > max)
}

fn digest_token(
    kind: &CapabilityKind,
    scope: &CapabilityScope,
    limits: &CapabilityLimits,
    issued_by: &str,
    issued_at_t: u64,
    expires_at_t: Option<u64>,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(kind.as_tag().as_bytes());
    hasher.update([0u8]);
    hasher.update(scope.canonical_bytes());
    hasher.update(limits.max_calls_per_window.to_le_bytes());
    hasher.update(limits.window_ticks.to_le_bytes());
    hasher.update(limits.max_bytes_out.unwrap_or(0).to_le_bytes());
    hasher.update(limits.max_bytes_in.unwrap_or(0).to_le_bytes());
    hasher.update(limits.max_concurrent.to_le_bytes());
    hasher.update(issued_by.as_bytes());
    hasher.update(issued_at_t.to_le_bytes());
    hasher.update(expires_at_t.unwrap_or(0).to_le_bytes());
    hasher.finalize().into()
}

impl CapabilityKind {
    pub fn as_tag(&self) -> &str {
        match self {
            Self::NetHttp => "net_http",
            Self::FileRead => "file_read",
            Self::FileWrite => "file_write",
            Self::ProcessExec => "process_exec",
            Self::ClipboardRead => "clipboard_read",
            Self::ClipboardWrite => "clipboard_write",
            Self::UiAutomation => "ui_automation",
            Self::ExternalApi => "external_api",
            Self::Custom(custom) => custom.as_str(),
        }
    }
}

impl CapabilityScope {
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            Self::Domains(items) => {
                out.extend_from_slice(b"domains");
                append_items(&mut out, items);
            }
            Self::Paths(items) => {
                out.extend_from_slice(b"paths");
                append_items(&mut out, items);
            }
            Self::ApiNames(items) => {
                out.extend_from_slice(b"api_names");
                append_items(&mut out, items);
            }
            Self::All => out.extend_from_slice(b"all"),
        }
        out
    }
}

fn append_items(out: &mut Vec<u8>, items: &[String]) {
    for item in items {
        out.extend_from_slice(item.as_bytes());
        out.push(0xff);
    }
}
