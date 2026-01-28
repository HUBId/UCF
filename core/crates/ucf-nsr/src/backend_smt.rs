#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_structural_store::NsrThresholds;
use ucf_types::Digest32;

use crate::{NsrReasonCode, NsrTraceInputs, ReasoningAtom, SymbolicBackend, SymbolicResult};

const SMT_DOMAIN: &[u8] = b"ucf.nsr.backend.smt.v1";
const COHERENCE_MIN: u16 = 3_000;
const FORMAL_OK_KEY: u16 = 9_000;
const CONTRADICTION_RULE_ID: u16 = 9_001;
const CONTRADICTION_SEVERITY: u16 = 9_000;

pub struct SmtBackend {
    thresholds: NsrThresholds,
}

impl SmtBackend {
    // Placeholder for future SMT solver integration.
    pub fn new(thresholds: NsrThresholds) -> Self {
        Self { thresholds }
    }
}

impl SymbolicBackend for SmtBackend {
    fn check(&mut self, facts: &[ReasoningAtom], inp: &NsrTraceInputs) -> SymbolicResult {
        let warn_threshold = self.thresholds.warn.min(10_000);
        let deny_threshold = self.thresholds.deny.min(10_000).max(warn_threshold);
        let ok = inp.risk < deny_threshold && inp.coherence_plv >= COHERENCE_MIN;

        let mut contradictions = Vec::new();
        if !ok {
            let reason = if inp.risk >= deny_threshold {
                NsrReasonCode::HighRisk
            } else {
                NsrReasonCode::LowCoherence
            };
            contradictions.push((CONTRADICTION_RULE_ID, reason, CONTRADICTION_SEVERITY));
        }

        let mut derived_atoms = Vec::new();
        if ok {
            derived_atoms.push(ReasoningAtom::new(FORMAL_OK_KEY, 1, inp.commit));
        }

        let commit = digest_smt_commit(inp.commit, deny_threshold, COHERENCE_MIN, facts);
        SymbolicResult {
            ok,
            contradictions,
            derived_atoms,
            commit,
        }
    }
}

fn digest_smt_commit(
    inputs_commit: Digest32,
    deny_threshold: u16,
    coherence_min: u16,
    facts: &[ReasoningAtom],
) -> Digest32 {
    let mut ordered_facts = facts.to_vec();
    ordered_facts.sort_by(|a, b| {
        a.key
            .cmp(&b.key)
            .then_with(|| a.commit.as_bytes().cmp(b.commit.as_bytes()))
    });

    let mut hasher = Hasher::new();
    hasher.update(SMT_DOMAIN);
    hasher.update(inputs_commit.as_bytes());
    hasher.update(&deny_threshold.to_be_bytes());
    hasher.update(&coherence_min.to_be_bytes());
    hasher.update(
        &u64::try_from(ordered_facts.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    for atom in &ordered_facts {
        hasher.update(atom.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}
