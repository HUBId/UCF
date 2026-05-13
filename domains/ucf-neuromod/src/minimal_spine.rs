use blake3::Hasher;
use ucf_types::{Digest32, EvidenceId};

pub const MINIMAL_SPINE_NEUROMOD_ENVELOPE_VERSION: u16 = 1;
pub const MINIMAL_SPINE_NEUROMOD_SOURCE: &str = "minimal_spine_v1";
pub const MINIMAL_SPINE_NEUROMOD_HINT_MAX: u16 = 1000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NeuromodHint(u16);

impl NeuromodHint {
    pub const MIN: Self = Self(0);
    pub const MAX: Self = Self(MINIMAL_SPINE_NEUROMOD_HINT_MAX);

    pub const fn new(value: u16) -> Result<Self, NeuromodEnvelopeError> {
        if value <= MINIMAL_SPINE_NEUROMOD_HINT_MAX {
            Ok(Self(value))
        } else {
            Err(NeuromodEnvelopeError::HintOutOfRange { value })
        }
    }

    pub const fn raw(self) -> u16 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MinimalSpineNeuromodHints {
    pub salience_hint: NeuromodHint,
    pub stability_hint: NeuromodHint,
    pub risk_hint: NeuromodHint,
    pub noise_hint: NeuromodHint,
    pub learning_hint: NeuromodHint,
}

impl MinimalSpineNeuromodHints {
    pub const fn new(
        salience_hint: NeuromodHint,
        stability_hint: NeuromodHint,
        risk_hint: NeuromodHint,
        noise_hint: NeuromodHint,
        learning_hint: NeuromodHint,
    ) -> Self {
        Self {
            salience_hint,
            stability_hint,
            risk_hint,
            noise_hint,
            learning_hint,
        }
    }

    pub fn conservative_from_status(policy_status: &str, output_status: &str) -> Self {
        let denied = matches!(policy_status, "deny" | "denied" | "suppress" | "suppressed");
        let materialized = matches!(
            output_status,
            "materialized" | "materialized-test-output" | "allowed" | "allow"
        );

        if denied {
            Self::new(
                NeuromodHint(250),
                NeuromodHint(250),
                NeuromodHint(800),
                NeuromodHint(700),
                NeuromodHint(100),
            )
        } else if materialized {
            Self::new(
                NeuromodHint(500),
                NeuromodHint(700),
                NeuromodHint(200),
                NeuromodHint(200),
                NeuromodHint(400),
            )
        } else {
            Self::default()
        }
    }

    pub const fn validate_bounds(&self) -> bool {
        self.salience_hint.raw() <= MINIMAL_SPINE_NEUROMOD_HINT_MAX
            && self.stability_hint.raw() <= MINIMAL_SPINE_NEUROMOD_HINT_MAX
            && self.risk_hint.raw() <= MINIMAL_SPINE_NEUROMOD_HINT_MAX
            && self.noise_hint.raw() <= MINIMAL_SPINE_NEUROMOD_HINT_MAX
            && self.learning_hint.raw() <= MINIMAL_SPINE_NEUROMOD_HINT_MAX
    }
}

impl Default for MinimalSpineNeuromodHints {
    fn default() -> Self {
        Self::new(
            NeuromodHint(300),
            NeuromodHint(500),
            NeuromodHint(300),
            NeuromodHint(300),
            NeuromodHint(300),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MinimalSpineNeuromodLinks {
    pub sequence: u64,
    pub evidence_id: EvidenceId,
    pub input_digest: Digest32,
    pub candidate_set_record_digest: Digest32,
    pub output_record_digest: Digest32,
    pub archive_output_key: Digest32,
    pub archive_output_event_digest: Digest32,
    pub policy_status: String,
    pub output_status: String,
}

impl MinimalSpineNeuromodLinks {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sequence: u64,
        evidence_id: EvidenceId,
        input_digest: Digest32,
        candidate_set_record_digest: Digest32,
        output_record_digest: Digest32,
        archive_output_key: Digest32,
        archive_output_event_digest: Digest32,
        policy_status: impl Into<String>,
        output_status: impl Into<String>,
    ) -> Self {
        Self {
            sequence,
            evidence_id,
            input_digest,
            candidate_set_record_digest,
            output_record_digest,
            archive_output_key,
            archive_output_event_digest,
            policy_status: policy_status.into(),
            output_status: output_status.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MinimalSpineNeuromodEnvelope {
    pub version: u16,
    pub sequence: u64,
    pub evidence_id: EvidenceId,
    pub input_digest: Digest32,
    pub candidate_set_record_digest: Digest32,
    pub output_record_digest: Digest32,
    pub archive_output_key: Digest32,
    pub archive_output_event_digest: Digest32,
    pub policy_status: String,
    pub output_status: String,
    pub salience_hint: NeuromodHint,
    pub stability_hint: NeuromodHint,
    pub risk_hint: NeuromodHint,
    pub noise_hint: NeuromodHint,
    pub learning_hint: NeuromodHint,
    pub source: String,
}

impl MinimalSpineNeuromodEnvelope {
    pub fn from_minimal_spine_links(
        links: MinimalSpineNeuromodLinks,
        hints: MinimalSpineNeuromodHints,
    ) -> Self {
        Self {
            version: MINIMAL_SPINE_NEUROMOD_ENVELOPE_VERSION,
            sequence: links.sequence,
            evidence_id: links.evidence_id,
            input_digest: links.input_digest,
            candidate_set_record_digest: links.candidate_set_record_digest,
            output_record_digest: links.output_record_digest,
            archive_output_key: links.archive_output_key,
            archive_output_event_digest: links.archive_output_event_digest,
            policy_status: links.policy_status,
            output_status: links.output_status,
            salience_hint: hints.salience_hint,
            stability_hint: hints.stability_hint,
            risk_hint: hints.risk_hint,
            noise_hint: hints.noise_hint,
            learning_hint: hints.learning_hint,
            source: MINIMAL_SPINE_NEUROMOD_SOURCE.to_string(),
        }
    }

    pub fn from_minimal_spine_links_with_conservative_hints(
        links: MinimalSpineNeuromodLinks,
    ) -> Self {
        let hints = MinimalSpineNeuromodHints::conservative_from_status(
            &links.policy_status,
            &links.output_status,
        );
        Self::from_minimal_spine_links(links, hints)
    }

    pub fn deterministic_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_u16(&mut out, self.version);
        push_u64(&mut out, self.sequence);
        push_str(&mut out, self.evidence_id.as_str());
        push_digest(&mut out, self.input_digest);
        push_digest(&mut out, self.candidate_set_record_digest);
        push_digest(&mut out, self.output_record_digest);
        push_digest(&mut out, self.archive_output_key);
        push_digest(&mut out, self.archive_output_event_digest);
        push_str(&mut out, &self.policy_status);
        push_str(&mut out, &self.output_status);
        push_u16(&mut out, self.salience_hint.raw());
        push_u16(&mut out, self.stability_hint.raw());
        push_u16(&mut out, self.risk_hint.raw());
        push_u16(&mut out, self.noise_hint.raw());
        push_u16(&mut out, self.learning_hint.raw());
        push_str(&mut out, &self.source);
        out
    }

    pub fn digest(&self) -> Digest32 {
        let mut hasher = Hasher::new();
        hasher.update(b"ucf.neuromod.minimal_spine.envelope.v1");
        hasher.update(&self.deterministic_bytes());
        Digest32::new(*hasher.finalize().as_bytes())
    }

    pub fn validate_links_nonzero(&self) -> bool {
        !self.evidence_id.as_str().is_empty()
            && !is_zero_digest(self.input_digest)
            && !is_zero_digest(self.candidate_set_record_digest)
            && !is_zero_digest(self.output_record_digest)
            && !is_zero_digest(self.archive_output_key)
            && !is_zero_digest(self.archive_output_event_digest)
    }

    pub const fn validate_bounds(&self) -> bool {
        self.salience_hint.raw() <= MINIMAL_SPINE_NEUROMOD_HINT_MAX
            && self.stability_hint.raw() <= MINIMAL_SPINE_NEUROMOD_HINT_MAX
            && self.risk_hint.raw() <= MINIMAL_SPINE_NEUROMOD_HINT_MAX
            && self.noise_hint.raw() <= MINIMAL_SPINE_NEUROMOD_HINT_MAX
            && self.learning_hint.raw() <= MINIMAL_SPINE_NEUROMOD_HINT_MAX
    }

    pub const fn allows_decision_override(&self) -> bool {
        false
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NeuromodEnvelopeError {
    HintOutOfRange { value: u16 },
}

fn is_zero_digest(digest: Digest32) -> bool {
    digest.as_bytes().iter().all(|byte| *byte == 0)
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn push_str(out: &mut Vec<u8>, value: &str) {
    let len = u32::try_from(value.len()).expect("minimal spine envelope field length fits u32");
    push_u32(out, len);
    out.extend_from_slice(value.as_bytes());
}

fn push_digest(out: &mut Vec<u8>, value: Digest32) {
    out.extend_from_slice(value.as_bytes());
}
