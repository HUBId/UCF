#![forbid(unsafe_code)]

use ucf_archive::ExperienceAppender;
use ucf_commit::commit_milestone_macro;
use ucf_types::v1::spec::{ExperienceRecord, MacroMilestone};
use ucf_types::{Digest32, EvidenceId};

const SELFSTATE_DOMAIN: u16 = 0x4753; // "GS" for Geist SelfState
const DERIVED_DOMAIN: u16 = 0x4744; // "GD" for Geist Derived

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GeistConfig {
    pub recursion_depth: u8,
    pub consistency_threshold: u16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelfState {
    pub level: u8,
    pub anchor: Digest32,
    pub context: Vec<Digest32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsistencyVerdict {
    Accept,
    Damp,
    Reject,
}

impl ConsistencyVerdict {
    fn as_u8(self) -> u8 {
        match self {
            Self::Accept => 1,
            Self::Damp => 2,
            Self::Reject => 3,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsistencyReport {
    pub score: u16,
    pub verdict: ConsistencyVerdict,
    pub matched_anchors: usize,
}

pub trait IsmStore {
    fn anchors(&self) -> Vec<Digest32>;
    fn upsert_anchor(&mut self, anchor: Digest32);
}

#[derive(Clone, Debug, Default)]
pub struct InMemoryIsm {
    anchors: Vec<Digest32>,
}

impl InMemoryIsm {
    pub fn new() -> Self {
        Self {
            anchors: Vec::new(),
        }
    }

    fn normalize(&mut self) {
        normalize_digests(&mut self.anchors);
    }
}

impl IsmStore for InMemoryIsm {
    fn anchors(&self) -> Vec<Digest32> {
        let mut anchors = self.anchors.clone();
        normalize_digests(&mut anchors);
        anchors
    }

    fn upsert_anchor(&mut self, anchor: Digest32) {
        self.anchors.push(anchor);
        self.normalize();
    }
}

pub struct GeistKernel<A: ExperienceAppender, I: IsmStore> {
    pub cfg: GeistConfig,
    pub archive: A,
    pub ism: I,
}

impl<A: ExperienceAppender, I: IsmStore> GeistKernel<A, I> {
    pub fn new(cfg: GeistConfig, archive: A, ism: I) -> Self {
        Self { cfg, archive, ism }
    }

    pub fn ingest_macro(
        &mut self,
        macro_ms: MacroMilestone,
    ) -> (Vec<SelfState>, ConsistencyReport, EvidenceId) {
        let macro_refs = derive_macro_refs(&macro_ms);
        let self_states = build_self_states(self.cfg.recursion_depth, &macro_refs);
        let base_state = self_states.first().expect("recursion_depth must be >= 1");
        let report = compute_consistency_report(&self.cfg, base_state, &self.ism);
        if report.verdict == ConsistencyVerdict::Accept {
            self.ism.upsert_anchor(base_state.anchor);
        }

        let record = derived_record_for_macro(&macro_ms, &macro_refs, &self_states, &report);
        let evidence_id = self.archive.append(record);
        (self_states, report, evidence_id)
    }
}

fn derive_macro_refs(macro_ms: &MacroMilestone) -> Vec<Digest32> {
    let commitment = commit_milestone_macro(macro_ms);
    vec![commitment.digest]
}

fn build_self_states(recursion_depth: u8, macro_refs: &[Digest32]) -> Vec<SelfState> {
    let mut states = Vec::with_capacity(recursion_depth as usize);
    let mut previous_anchor = None;
    for level in 1..=recursion_depth {
        let state = build_self_state(level, macro_refs, previous_anchor);
        previous_anchor = Some(state.anchor);
        states.push(state);
    }
    states
}

fn build_self_state(
    level: u8,
    macro_refs: &[Digest32],
    previous_anchor: Option<Digest32>,
) -> SelfState {
    let mut context = macro_refs.to_vec();
    if let Some(anchor) = previous_anchor {
        context.push(anchor);
    }
    normalize_digests(&mut context);
    let anchor = hash_self_state(level, macro_refs, previous_anchor);
    SelfState {
        level,
        anchor,
        context,
    }
}

fn hash_self_state(
    level: u8,
    macro_refs: &[Digest32],
    previous_anchor: Option<Digest32>,
) -> Digest32 {
    let mut refs = macro_refs.to_vec();
    normalize_digests(&mut refs);
    let mut enc = Encoder::new();
    enc.write_u16(SELFSTATE_DOMAIN);
    enc.write_u8(level);
    enc.write_u32(refs.len() as u32);
    for digest in refs {
        enc.write_digest(&digest);
    }
    match previous_anchor {
        Some(anchor) => {
            enc.write_u8(1);
            enc.write_digest(&anchor);
        }
        None => enc.write_u8(0),
    }
    hash_bytes(enc.finish())
}

fn compute_consistency_report(
    cfg: &GeistConfig,
    self_state: &SelfState,
    ism: &impl IsmStore,
) -> ConsistencyReport {
    let ism_anchors = ism.anchors();
    let matched_anchors = self_state
        .context
        .iter()
        .filter(|anchor| ism_anchors.contains(anchor))
        .count();
    let score = matched_anchors.min(u16::MAX as usize) as u16;
    let verdict = if score >= cfg.consistency_threshold {
        ConsistencyVerdict::Accept
    } else if score > 0 {
        ConsistencyVerdict::Damp
    } else {
        ConsistencyVerdict::Reject
    };
    ConsistencyReport {
        score,
        verdict,
        matched_anchors,
    }
}

fn derived_record_for_macro(
    macro_ms: &MacroMilestone,
    macro_refs: &[Digest32],
    self_states: &[SelfState],
    report: &ConsistencyReport,
) -> ExperienceRecord {
    let commitment = commit_milestone_macro(macro_ms);
    let payload = encode_derived_payload(macro_refs, self_states, report);
    ExperienceRecord {
        record_id: format!("geist-derived-{}", hex_encode(commitment.digest.as_bytes())),
        observed_at_ms: macro_ms.achieved_at_ms,
        subject_id: "geist".to_string(),
        payload,
        digest: None,
        vrf_tag: None,
        proof_ref: None,
    }
}

fn encode_derived_payload(
    macro_refs: &[Digest32],
    self_states: &[SelfState],
    report: &ConsistencyReport,
) -> Vec<u8> {
    let mut refs = macro_refs.to_vec();
    normalize_digests(&mut refs);
    let mut enc = Encoder::new();
    enc.write_u16(DERIVED_DOMAIN);
    enc.write_u32(refs.len() as u32);
    for digest in refs {
        enc.write_digest(&digest);
    }
    enc.write_u32(self_states.len() as u32);
    for state in self_states {
        enc.write_u8(state.level);
        enc.write_digest(&state.anchor);
    }
    enc.write_u16(report.score);
    enc.write_u8(report.verdict.as_u8());
    enc.write_u32(report.matched_anchors.min(u32::MAX as usize) as u32);
    enc.finish().to_vec()
}

fn hash_bytes(bytes: &[u8]) -> Digest32 {
    let digest = blake3::hash(bytes);
    Digest32::new(*digest.as_bytes())
}

fn normalize_digests(digests: &mut Vec<Digest32>) {
    digests.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
    digests.dedup();
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

struct Encoder {
    bytes: Vec<u8>,
}

impl Encoder {
    fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    fn write_u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn write_u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn write_u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn write_digest(&mut self, digest: &Digest32) {
        self.bytes.extend_from_slice(digest.as_bytes());
    }

    fn finish(&self) -> &[u8] {
        &self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_archive::InMemoryArchive;

    fn sample_macro(id: &str) -> MacroMilestone {
        MacroMilestone {
            milestone_id: id.to_string(),
            achieved_at_ms: 42,
            label: "macro".to_string(),
            meso_milestone_ids: vec!["meso-1".to_string()],
        }
    }

    #[test]
    fn determinism_same_macro_same_anchors() {
        let macro_ms = sample_macro("macro-1");
        let macro_refs = derive_macro_refs(&macro_ms);
        let states_a = build_self_states(3, &macro_refs);
        let states_b = build_self_states(3, &macro_refs);
        let anchors_a: Vec<Digest32> = states_a.iter().map(|state| state.anchor).collect();
        let anchors_b: Vec<Digest32> = states_b.iter().map(|state| state.anchor).collect();
        assert_eq!(anchors_a, anchors_b);
    }

    #[test]
    fn consistency_reports_accept_and_reject() {
        let cfg = GeistConfig {
            recursion_depth: 1,
            consistency_threshold: 1,
        };
        let macro_ms = sample_macro("macro-1");
        let macro_refs = derive_macro_refs(&macro_ms);
        let state = build_self_state(1, &macro_refs, None);
        let ism = InMemoryIsm::new();
        let report = compute_consistency_report(&cfg, &state, &ism);
        assert_eq!(report.verdict, ConsistencyVerdict::Reject);

        let mut ism = InMemoryIsm::new();
        ism.upsert_anchor(macro_refs[0]);
        let report = compute_consistency_report(&cfg, &state, &ism);
        assert_eq!(report.verdict, ConsistencyVerdict::Accept);
    }

    #[test]
    fn ingest_macro_appends_record_and_updates_ism() {
        let cfg = GeistConfig {
            recursion_depth: 2,
            consistency_threshold: 1,
        };
        let macro_ms = sample_macro("macro-1");
        let macro_refs = derive_macro_refs(&macro_ms);
        let mut ism = InMemoryIsm::new();
        ism.upsert_anchor(macro_refs[0]);
        let archive = InMemoryArchive::new();
        let mut kernel = GeistKernel::new(cfg, archive, ism);

        let (states, report, _evidence_id) = kernel.ingest_macro(macro_ms);
        assert_eq!(report.verdict, ConsistencyVerdict::Accept);
        assert_eq!(kernel.archive.list().len(), 1);
        assert!(kernel.ism.anchors().contains(&states[0].anchor));
    }
}
