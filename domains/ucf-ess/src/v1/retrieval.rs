use sha2::{Digest, Sha256};

use crate::v1::{
    AuditPayload, EbmReasoningRecord, ExperienceEbmTagRecord, ExperienceId, ExperienceRecord,
    RetrievalDecisionRecord, RetrievalReasonCode, RetrievalSelectionRecord,
    RetrievedExperienceRole,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetrievalPolicy {
    pub low_energy_threshold_q: u16,
    pub high_energy_threshold_q: u16,
    pub max_selected: usize,
    pub max_avoid: usize,
}

impl Default for RetrievalPolicy {
    fn default() -> Self {
        Self {
            low_energy_threshold_q: 3_000,
            high_energy_threshold_q: 7_000,
            max_selected: 8,
            max_avoid: 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetrievalContext {
    pub high_risk: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetrievalCandidate {
    pub experience_id: ExperienceId,
    pub experience_digest_prefix: [u8; 8],
    pub base_score: u16,
    pub energy_q: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetrievalResult {
    pub selected: Vec<RetrievalSelectionRecord>,
    pub reasons: Vec<RetrievalReasonCode>,
}

pub fn make_ebm_tag_from_reasoning(
    reasoning: &EbmReasoningRecord,
    evidence_chain_digest: [u8; 32],
) -> ExperienceEbmTagRecord {
    let mut min_energy = reasoning.aggregate_energy_q;
    let mut total = 0u32;
    let mut count = 0u32;
    for energy in reasoning.top_energies_q.iter().copied().take(4) {
        min_energy = min_energy.min(energy);
        total = total.saturating_add(u32::from(energy));
        count = count.saturating_add(1);
    }
    let mean_topk = total
        .checked_div(count)
        .map(|mean| mean as u16)
        .unwrap_or(reasoning.aggregate_energy_q);

    ExperienceEbmTagRecord {
        decision_id: reasoning.decision_id,
        evidence_chain_digest,
        ebm_energy_min_q: min_energy,
        ebm_energy_mean_topk_q: mean_topk,
        ebm_constraints_digest_prefix: reasoning.constraints_digest_prefix,
        ebm_top_terms: reasoning.top_term_contributions.clone(),
        ebm_reasoning_digest_prefix: reasoning.ebm_digest_prefix,
    }
    .clamp_bounds()
}

pub fn compute_query_digest_prefix(input: &[u8]) -> [u8; 8] {
    let mut hasher = Sha256::new();
    hasher.update(b"ucf.ess.retrieval.query.v1");
    hasher.update(input);
    let digest: [u8; 32] = hasher.finalize().into();
    let mut out = [0u8; 8];
    out.copy_from_slice(&digest[..8]);
    out
}

pub fn apply_ebm_bias(
    candidates: &[RetrievalCandidate],
    policy: RetrievalPolicy,
    context: RetrievalContext,
) -> RetrievalResult {
    let mut selected = Vec::new();
    let mut reasons = vec![RetrievalReasonCode::EbmBiasApplied];
    if context.high_risk {
        reasons.push(RetrievalReasonCode::HighRiskContext);
    }

    let mut scored: Vec<(i64, RetrievalSelectionRecord)> = candidates
        .iter()
        .filter_map(|candidate| {
            let role = classify_role(candidate.energy_q, policy, context);
            if matches!(role, RetrievedExperienceRole::AvoidExample) {
                return None;
            }
            let safety_bias: i64 = if context.high_risk {
                i64::from(
                    policy
                        .low_energy_threshold_q
                        .saturating_sub(candidate.energy_q),
                )
            } else {
                i64::from(
                    policy
                        .high_energy_threshold_q
                        .saturating_sub(candidate.energy_q / 2),
                )
            };
            Some((
                i64::from(candidate.base_score) + safety_bias,
                RetrievalSelectionRecord {
                    experience_id: candidate.experience_id,
                    experience_digest_prefix: candidate.experience_digest_prefix,
                    role,
                },
            ))
        })
        .collect();

    scored.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| a.1.experience_id.0.cmp(&b.1.experience_id.0))
    });

    for (_, record) in scored.into_iter().take(policy.max_selected) {
        selected.push(record);
    }

    let mut avoid: Vec<RetrievalSelectionRecord> = candidates
        .iter()
        .filter(|candidate| {
            candidate.energy_q >= policy.high_energy_threshold_q
                && candidate
                    .energy_q
                    .saturating_sub(policy.high_energy_threshold_q)
                    .saturating_add(candidate.base_score)
                    > 0
        })
        .map(|candidate| RetrievalSelectionRecord {
            experience_id: candidate.experience_id,
            experience_digest_prefix: candidate.experience_digest_prefix,
            role: RetrievedExperienceRole::AvoidExample,
        })
        .collect();
    avoid.sort_by_key(|a| a.experience_id.0);
    avoid.truncate(policy.max_avoid);
    if !avoid.is_empty() {
        reasons.push(RetrievalReasonCode::AvoidExamplesIncluded);
    }
    selected.extend(avoid);

    selected.truncate(policy.max_selected);

    RetrievalResult { selected, reasons }
}

fn classify_role(
    energy_q: u16,
    policy: RetrievalPolicy,
    context: RetrievalContext,
) -> RetrievedExperienceRole {
    if energy_q >= policy.high_energy_threshold_q {
        RetrievedExperienceRole::AvoidExample
    } else if energy_q <= policy.low_energy_threshold_q || context.high_risk {
        RetrievedExperienceRole::PrecedentSafe
    } else {
        RetrievedExperienceRole::Template
    }
}

pub fn build_retrieval_decision_record(
    t: u64,
    query_digest_prefix: [u8; 8],
    policy_hash_prefix: [u8; 8],
    evidence_chain_digest_prefix: [u8; 8],
    policy: RetrievalPolicy,
    result: RetrievalResult,
) -> RetrievalDecisionRecord {
    RetrievalDecisionRecord {
        schema_version: 1,
        t,
        query_digest_prefix,
        selected: result.selected,
        low_energy_threshold_q: policy.low_energy_threshold_q,
        high_energy_threshold_q: policy.high_energy_threshold_q,
        policy_hash_prefix,
        evidence_chain_digest_prefix,
        reason_codes: result.reasons,
    }
}

pub fn find_ebm_energy(record: &ExperienceRecord) -> Option<u16> {
    record
        .ebm_tag
        .as_ref()
        .map(|tag| tag.ebm_energy_mean_topk_q)
}

pub fn extract_retrieval_decision(record: &ExperienceRecord) -> Option<&RetrievalDecisionRecord> {
    match &record.payload {
        crate::v1::ExperiencePayload::Audit(AuditPayload::RetrievalDecision(decision)) => {
            Some(decision)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_energy_is_never_template() {
        let policy = RetrievalPolicy::default();
        let role = classify_role(
            policy.high_energy_threshold_q,
            policy,
            RetrievalContext { high_risk: false },
        );
        assert_eq!(role, RetrievedExperienceRole::AvoidExample);
    }

    #[test]
    fn retrieval_bias_is_deterministic() {
        let policy = RetrievalPolicy::default();
        let context = RetrievalContext { high_risk: true };
        let candidates = vec![
            RetrievalCandidate {
                experience_id: ExperienceId(3),
                experience_digest_prefix: [3u8; 8],
                base_score: 900,
                energy_q: 2_000,
            },
            RetrievalCandidate {
                experience_id: ExperienceId(1),
                experience_digest_prefix: [1u8; 8],
                base_score: 900,
                energy_q: 8_000,
            },
            RetrievalCandidate {
                experience_id: ExperienceId(2),
                experience_digest_prefix: [2u8; 8],
                base_score: 900,
                energy_q: 2_000,
            },
        ];
        let a = apply_ebm_bias(&candidates, policy, context);
        let b = apply_ebm_bias(&candidates, policy, context);
        assert_eq!(a, b);
        assert!(a.selected.iter().all(
            |item| item.role != RetrievedExperienceRole::Template || item.experience_id.0 != 1
        ));
    }

    #[test]
    fn tag_generation_is_bounded_and_joined() {
        let reasoning = EbmReasoningRecord {
            suppressed_by_emergency: false,
            schema_version: 1,
            t: 42,
            run_id: 7,
            decision_id: 99,
            backend_pack_digest_prefix: [0; 8],
            ebm_backend_id: 1,
            ebm_model_digest_prefix: [1; 8],
            contract_version: 1,
            enablement_mode: 1,
            risk_q: 500,
            pressure_q: 600,
            surprise_q: 700,
            uncertainty_q: 800,
            aggregate_energy_q: 6000,
            base_energy_q: 5000,
            top_energies_q: vec![1000, 2000, 3000, 4000, 5000],
            top_candidate_ids: vec![1, 2],
            ebm_digest_prefix: [2; 8],
            constraints_digest_prefix: [3; 8],
            top_term_contributions: vec![(9, 2), (5, 8), (6, 7), (4, 5), (1, 1)],
            search_enabled: true,
            search_steps_used: 1,
            evidence_chain_digest_prefix: [4; 8],
            status: 0,
            reason_code: 0,
        };
        let tag = make_ebm_tag_from_reasoning(&reasoning, [7; 32]);
        assert_eq!(tag.decision_id, reasoning.decision_id);
        assert_eq!(tag.evidence_chain_digest, [7; 32]);
        assert_eq!(tag.ebm_energy_min_q, 1000);
        assert_eq!(tag.ebm_energy_mean_topk_q, 2500);
        assert!(tag.ebm_top_terms.len() <= 4);
    }

    #[test]
    fn retrieval_record_reproducible() {
        let policy = RetrievalPolicy::default();
        let context = RetrievalContext { high_risk: false };
        let candidates = vec![
            RetrievalCandidate {
                experience_id: ExperienceId(10),
                experience_digest_prefix: [10; 8],
                base_score: 400,
                energy_q: 2500,
            },
            RetrievalCandidate {
                experience_id: ExperienceId(11),
                experience_digest_prefix: [11; 8],
                base_score: 350,
                energy_q: 8500,
            },
        ];
        let result = apply_ebm_bias(&candidates, policy, context);
        let q = compute_query_digest_prefix(b"same-query");
        let a = build_retrieval_decision_record(12, q, [1; 8], [2; 8], policy, result.clone());
        let b = build_retrieval_decision_record(12, q, [1; 8], [2; 8], policy, result);
        assert_eq!(a, b);
    }
}
