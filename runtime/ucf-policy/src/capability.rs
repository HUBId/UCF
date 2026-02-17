use sha2::{Digest, Sha256};

use crate::gem::{PayloadHint, ToolRequest};

pub type PolicyModuleId = &'static str;

pub const MAX_SCOPE_ITEMS: usize = 16;
pub const MAX_SCOPE_ITEM_LEN: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeCaps {
    pub max_bytes: usize,
    pub max_vec_len: usize,
    pub max_str_len: usize,
    pub max_map_len: usize,
}

impl DecodeCaps {
    pub const fn caps_default() -> Self {
        Self {
            max_bytes: 32 * 1024,
            max_vec_len: 256,
            max_str_len: MAX_SCOPE_ITEM_LEN,
            max_map_len: MAX_SCOPE_ITEMS,
        }
    }

    pub const fn caps_ipc() -> Self {
        Self {
            max_bytes: 128 * 1024,
            max_vec_len: 1024,
            max_str_len: 256,
            max_map_len: 256,
        }
    }

    pub const fn caps_ess() -> Self {
        Self {
            max_bytes: 16 * 1024,
            max_vec_len: 128,
            max_str_len: MAX_SCOPE_ITEM_LEN,
            max_map_len: MAX_SCOPE_ITEMS,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeError {
    pub kind: &'static str,
    pub at: &'static str,
    pub context: String,
}

impl DecodeError {
    fn new(kind: &'static str, at: &'static str, context: impl Into<String>) -> Self {
        let mut context = context.into();
        context.truncate(96);
        Self { kind, at, context }
    }
}

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
    PolicyBundleUnverified,
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

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_be_bytes());
}

fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_be_bytes());
}

fn put_string(out: &mut Vec<u8>, value: &str) {
    put_u32(out, value.len() as u32);
    out.extend_from_slice(value.as_bytes());
}

fn read_u32(input: &[u8], idx: &mut usize) -> Result<u32, DecodeError> {
    if input.len().saturating_sub(*idx) < 4 {
        return Err(DecodeError::new("truncated", "decode_u32", "need_4_bytes"));
    }
    let v = u32::from_be_bytes([
        input[*idx],
        input[*idx + 1],
        input[*idx + 2],
        input[*idx + 3],
    ]);
    *idx += 4;
    Ok(v)
}

fn read_u64(input: &[u8], idx: &mut usize) -> Result<u64, DecodeError> {
    if input.len().saturating_sub(*idx) < 8 {
        return Err(DecodeError::new("truncated", "decode_u64", "need_8_bytes"));
    }
    let v = u64::from_be_bytes([
        input[*idx],
        input[*idx + 1],
        input[*idx + 2],
        input[*idx + 3],
        input[*idx + 4],
        input[*idx + 5],
        input[*idx + 6],
        input[*idx + 7],
    ]);
    *idx += 8;
    Ok(v)
}

fn read_string(input: &[u8], idx: &mut usize, caps: DecodeCaps) -> Result<String, DecodeError> {
    let len = read_u32(input, idx)? as usize;
    if len > caps.max_str_len {
        return Err(DecodeError::new(
            "bounds",
            "decode_string",
            format!("len>{}", caps.max_str_len),
        ));
    }
    if input.len().saturating_sub(*idx) < len {
        return Err(DecodeError::new("truncated", "decode_string", "payload"));
    }
    let out = String::from_utf8(input[*idx..*idx + len].to_vec())
        .map_err(|_| DecodeError::new("utf8", "decode_string", "invalid_utf8"))?;
    *idx += len;
    Ok(out)
}

impl CapabilityToken {
    pub fn encode_canonical(&self) -> Vec<u8> {
        let mut out = Vec::new();
        put_string(&mut out, self.kind.as_tag());
        match &self.scope {
            CapabilityScope::Domains(items) => {
                out.push(1);
                put_u32(&mut out, items.len() as u32);
                for item in items {
                    put_string(&mut out, item);
                }
            }
            CapabilityScope::Paths(items) => {
                out.push(2);
                put_u32(&mut out, items.len() as u32);
                for item in items {
                    put_string(&mut out, item);
                }
            }
            CapabilityScope::ApiNames(items) => {
                out.push(3);
                put_u32(&mut out, items.len() as u32);
                for item in items {
                    put_string(&mut out, item);
                }
            }
            CapabilityScope::All => out.push(4),
        }
        put_u32(&mut out, self.limits.max_calls_per_window);
        put_u64(&mut out, self.limits.window_ticks);
        put_u32(&mut out, self.limits.max_bytes_out.unwrap_or(u32::MAX));
        put_u32(&mut out, self.limits.max_bytes_in.unwrap_or(u32::MAX));
        out.extend_from_slice(&self.limits.max_concurrent.to_be_bytes());
        put_string(&mut out, self.issued_by);
        put_u64(&mut out, self.issued_at_t);
        match self.expires_at_t {
            Some(v) => {
                out.push(1);
                put_u64(&mut out, v);
            }
            None => out.push(0),
        }
        out.extend_from_slice(&self.token_digest);
        out
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedCapabilityToken {
    pub kind: CapabilityKind,
    pub scope: CapabilityScope,
    pub limits: CapabilityLimits,
    pub issued_by: String,
    pub issued_at_t: u64,
    pub expires_at_t: Option<u64>,
    pub token_digest: [u8; 32],
}

pub fn decode_capability_token(
    bytes: &[u8],
    caps: DecodeCaps,
) -> Result<DecodedCapabilityToken, DecodeError> {
    if bytes.len() > caps.max_bytes {
        return Err(DecodeError::new(
            "bounds",
            "decode_capability_token",
            format!("input>{}", caps.max_bytes),
        ));
    }
    let mut idx = 0usize;
    let kind_tag = read_string(bytes, &mut idx, caps)?;
    let scope_kind = *bytes
        .get(idx)
        .ok_or_else(|| DecodeError::new("truncated", "decode_scope", "scope_kind"))?;
    idx += 1;
    let scope = match scope_kind {
        1..=3 => {
            let len = read_u32(bytes, &mut idx)? as usize;
            if len > caps.max_map_len {
                return Err(DecodeError::new(
                    "bounds",
                    "decode_scope",
                    format!("items>{}", caps.max_map_len),
                ));
            }
            let mut items = Vec::with_capacity(len);
            for _ in 0..len {
                items.push(read_string(bytes, &mut idx, caps)?);
            }
            match scope_kind {
                1 => CapabilityScope::Domains(items),
                2 => CapabilityScope::Paths(items),
                _ => CapabilityScope::ApiNames(items),
            }
        }
        4 => CapabilityScope::All,
        _ => return Err(DecodeError::new("enum", "decode_scope", "scope_kind")),
    };
    let max_calls_per_window = read_u32(bytes, &mut idx)?;
    let window_ticks = read_u64(bytes, &mut idx)?;
    let max_bytes_out = read_u32(bytes, &mut idx)?;
    let max_bytes_in = read_u32(bytes, &mut idx)?;
    if bytes.len().saturating_sub(idx) < 2 {
        return Err(DecodeError::new(
            "truncated",
            "decode_limits",
            "max_concurrent",
        ));
    }
    let max_concurrent = u16::from_be_bytes([bytes[idx], bytes[idx + 1]]);
    idx += 2;
    let issued_by = read_string(bytes, &mut idx, caps)?;
    let issued_at_t = read_u64(bytes, &mut idx)?;
    let has_exp = *bytes
        .get(idx)
        .ok_or_else(|| DecodeError::new("truncated", "decode_exp", "tag"))?;
    idx += 1;
    let expires_at_t = if has_exp == 1 {
        Some(read_u64(bytes, &mut idx)?)
    } else {
        None
    };
    if bytes.len().saturating_sub(idx) < 32 {
        return Err(DecodeError::new(
            "truncated",
            "decode_digest",
            "token_digest",
        ));
    }
    let mut token_digest = [0u8; 32];
    token_digest.copy_from_slice(&bytes[idx..idx + 32]);
    let kind = match kind_tag.as_str() {
        "net_http" => CapabilityKind::NetHttp,
        "file_read" => CapabilityKind::FileRead,
        "file_write" => CapabilityKind::FileWrite,
        "process_exec" => CapabilityKind::ProcessExec,
        "clipboard_read" => CapabilityKind::ClipboardRead,
        "clipboard_write" => CapabilityKind::ClipboardWrite,
        "ui_automation" => CapabilityKind::UiAutomation,
        "external_api" => CapabilityKind::ExternalApi,
        custom => CapabilityKind::Custom(custom.to_string()),
    };
    let limits = CapabilityLimits {
        max_calls_per_window,
        window_ticks,
        max_bytes_out: (max_bytes_out != u32::MAX).then_some(max_bytes_out),
        max_bytes_in: (max_bytes_in != u32::MAX).then_some(max_bytes_in),
        max_concurrent,
    }
    .clamped();
    let scope = scope.clamped();
    if digest_token(
        &kind,
        &scope,
        &limits,
        issued_by.as_str(),
        issued_at_t,
        expires_at_t,
    ) != token_digest
    {
        return Err(DecodeError::new(
            "digest",
            "decode_capability_token",
            "token_digest_mismatch",
        ));
    }
    Ok(DecodedCapabilityToken {
        kind,
        scope,
        limits,
        issued_by,
        issued_at_t,
        expires_at_t,
        token_digest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_token_roundtrip() {
        let tok = CapabilityToken::issue(
            CapabilityKind::ExternalApi,
            CapabilityScope::ApiNames(vec!["external_output".to_string()]),
            CapabilityLimits {
                max_calls_per_window: 8,
                window_ticks: 10,
                max_bytes_out: Some(1024),
                max_bytes_in: Some(64),
                max_concurrent: 1,
            },
            "issuer",
            5,
            Some(9),
        );
        let bytes = tok.encode_canonical();
        let decoded = decode_capability_token(&bytes, DecodeCaps::caps_default()).expect("decode");
        assert_eq!(decoded.kind, tok.kind);
        assert_eq!(decoded.scope, tok.scope);
        assert_eq!(decoded.limits, tok.limits);
        assert_eq!(decoded.issued_by, tok.issued_by);
        assert_eq!(decoded.token_digest, tok.token_digest);
    }

    #[test]
    fn decode_capability_token_rejects_oversized() {
        let bytes = vec![0u8; DecodeCaps::caps_default().max_bytes + 1];
        assert!(decode_capability_token(&bytes, DecodeCaps::caps_default()).is_err());
    }
}
