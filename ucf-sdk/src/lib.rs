#![forbid(unsafe_code)]

//! Stable Core API for UCF library consumers.
//!
//! This crate intentionally exposes a minimal, deterministic boundary surface:
//! - [`ControlFrameV1`] for submit operations
//! - [`DecisionEventV1`] for decision stream/read operations
//! - [`EssSummaryQueryV1`] and [`EssSummaryResponseV1`] for ESS summaries

pub use ucf_types::{Digest32, UQ0_16};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum DecisionKindV1 {
    Allow = 1,
    Deny = 2,
    Defer = 3,
}

impl DecisionKindV1 {
    pub const fn code(self) -> u8 {
        self as u8
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ControlFrameV1 {
    pub control_id: String,
    pub input_digest: Digest32,
    pub policy_class: u16,
    pub cycle_hint: u64,
    pub nonce: u64,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct DecisionEventV1 {
    pub control_id: String,
    pub event_seq: u64,
    pub decision: DecisionKindV1,
    pub reason_code: u16,
    pub confidence: UQ0_16,
    pub decision_digest: Digest32,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct EssSummaryQueryV1 {
    pub from_seq_inclusive: u64,
    pub limit: u16,
    pub decision_filter: Option<DecisionKindV1>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct EssSummaryResponseV1 {
    pub from_seq_inclusive: u64,
    pub to_seq_inclusive: u64,
    pub total: u64,
    pub allow_count: u64,
    pub deny_count: u64,
    pub defer_count: u64,
    pub mean_confidence: UQ0_16,
    pub summary_digest: Digest32,
}

fn put_u16(buf: &mut Vec<u8>, value: u16) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(buf: &mut Vec<u8>, value: u64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn put_digest(buf: &mut Vec<u8>, value: &Digest32) {
    buf.extend_from_slice(value.as_bytes());
}

fn put_string(buf: &mut Vec<u8>, value: &str) {
    put_u64(buf, value.len() as u64);
    buf.extend_from_slice(value.as_bytes());
}

impl ControlFrameV1 {
    pub fn new(
        control_id: String,
        input_digest: Digest32,
        policy_class: u16,
        cycle_hint: u64,
        nonce: u64,
    ) -> Self {
        Self {
            control_id,
            input_digest,
            policy_class,
            cycle_hint,
            nonce,
        }
    }

    pub fn encode_deterministic(&self) -> Vec<u8> {
        let mut out = Vec::new();
        put_string(&mut out, &self.control_id);
        put_digest(&mut out, &self.input_digest);
        put_u16(&mut out, self.policy_class);
        put_u64(&mut out, self.cycle_hint);
        put_u64(&mut out, self.nonce);
        out
    }
}

impl DecisionEventV1 {
    pub fn new(
        control_id: String,
        event_seq: u64,
        decision: DecisionKindV1,
        reason_code: u16,
        confidence: UQ0_16,
        decision_digest: Digest32,
    ) -> Self {
        Self {
            control_id,
            event_seq,
            decision,
            reason_code,
            confidence,
            decision_digest,
        }
    }

    pub fn encode_deterministic(&self) -> Vec<u8> {
        let mut out = Vec::new();
        put_string(&mut out, &self.control_id);
        put_u64(&mut out, self.event_seq);
        out.push(self.decision.code());
        put_u16(&mut out, self.reason_code);
        put_u16(&mut out, self.confidence.raw());
        put_digest(&mut out, &self.decision_digest);
        out
    }
}

impl EssSummaryQueryV1 {
    pub fn new(
        from_seq_inclusive: u64,
        limit: u16,
        decision_filter: Option<DecisionKindV1>,
    ) -> Self {
        Self {
            from_seq_inclusive,
            limit,
            decision_filter,
        }
    }

    pub fn encode_deterministic(&self) -> Vec<u8> {
        let mut out = Vec::new();
        put_u64(&mut out, self.from_seq_inclusive);
        put_u16(&mut out, self.limit);
        match self.decision_filter {
            Some(kind) => {
                out.push(1);
                out.push(kind.code());
            }
            None => out.push(0),
        }
        out
    }
}

impl EssSummaryResponseV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        from_seq_inclusive: u64,
        to_seq_inclusive: u64,
        total: u64,
        allow_count: u64,
        deny_count: u64,
        defer_count: u64,
        mean_confidence: UQ0_16,
        summary_digest: Digest32,
    ) -> Self {
        Self {
            from_seq_inclusive,
            to_seq_inclusive,
            total,
            allow_count,
            deny_count,
            defer_count,
            mean_confidence,
            summary_digest,
        }
    }

    pub fn encode_deterministic(&self) -> Vec<u8> {
        let mut out = Vec::new();
        put_u64(&mut out, self.from_seq_inclusive);
        put_u64(&mut out, self.to_seq_inclusive);
        put_u64(&mut out, self.total);
        put_u64(&mut out, self.allow_count);
        put_u64(&mut out, self.deny_count);
        put_u64(&mut out, self.defer_count);
        put_u16(&mut out, self.mean_confidence.raw());
        put_digest(&mut out, &self.summary_digest);
        out
    }
}
