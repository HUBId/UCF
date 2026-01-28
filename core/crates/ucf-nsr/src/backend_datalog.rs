#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_cde_scm::{edge_key, CdeNodeId, NSR_ATOM_MIN};
use ucf_structural_store::NsrThresholds;
use ucf_types::Digest32;

use crate::{NsrReasonCode, NsrTraceInputs, ReasoningAtom, SymbolicBackend, SymbolicResult};

const DATALOG_DOMAIN: &[u8] = b"ucf.nsr.backend.datalog.v1";
const RISK_CAUSED_KEY: u16 = 9_101;
const INSTABILITY_KEY: u16 = 9_102;
const CONTRADICTION_RULE_ID: u16 = 9_200;
const CONTRADICTION_SEVERITY: u16 = 9_000;
const PHI_LOW: u16 = 3_000;

pub struct DatalogBackend {
    thresholds: NsrThresholds,
}

impl DatalogBackend {
    // Placeholder for future Datalog engine integration.
    pub fn new(thresholds: NsrThresholds) -> Self {
        Self { thresholds }
    }
}

impl SymbolicBackend for DatalogBackend {
    fn check(&mut self, facts: &[ReasoningAtom], inp: &NsrTraceInputs) -> SymbolicResult {
        let warn_threshold = self.thresholds.warn.min(10_000);
        let deny_threshold = self.thresholds.deny.min(10_000).max(warn_threshold);

        let causal_key = edge_key(CdeNodeId::Risk, CdeNodeId::OutputSuppression);
        let causal_strong = facts
            .iter()
            .any(|atom| atom.key == causal_key && atom.value >= NSR_ATOM_MIN as i16);
        let risk_high = inp.risk >= deny_threshold;

        let mut derived_atoms = Vec::new();
        if causal_strong && risk_high {
            derived_atoms.push(ReasoningAtom::new(RISK_CAUSED_KEY, 1, inp.commit));
        }
        if inp.phi_proxy < PHI_LOW && inp.drift >= deny_threshold {
            derived_atoms.push(ReasoningAtom::new(INSTABILITY_KEY, 1, inp.commit));
        }

        let mut contradictions = Vec::new();
        let ok = if inp.policy_decision_commit.is_some() {
            contradictions.push((
                CONTRADICTION_RULE_ID,
                NsrReasonCode::PolicyConflict,
                CONTRADICTION_SEVERITY,
            ));
            false
        } else {
            true
        };

        let commit = digest_datalog_commit(inp.commit, deny_threshold, facts);
        SymbolicResult {
            ok,
            contradictions,
            derived_atoms,
            commit,
        }
    }
}

fn digest_datalog_commit(
    inputs_commit: Digest32,
    deny_threshold: u16,
    facts: &[ReasoningAtom],
) -> Digest32 {
    let mut ordered_facts = facts.to_vec();
    ordered_facts.sort_by(|a, b| {
        a.key
            .cmp(&b.key)
            .then_with(|| a.commit.as_bytes().cmp(b.commit.as_bytes()))
    });

    let mut hasher = Hasher::new();
    hasher.update(DATALOG_DOMAIN);
    hasher.update(inputs_commit.as_bytes());
    hasher.update(&deny_threshold.to_be_bytes());
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
