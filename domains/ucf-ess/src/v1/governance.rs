use crate::v1::{AuditPayload, ExperiencePayload, ExperienceRecord, PayloadClassification};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataClass {
    DigestOnly,
    ScalarSummary,
    TextPayload,
    BinaryPayload,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetentionPolicyV1 {
    pub schema_version: u16,
    pub keep_full_for_ticks: u64,
    pub keep_full_for_days: u64,
    pub keep_digests_forever: bool,
    pub max_ess_bytes: u64,
    pub policy_marker: String,
}

impl Default for RetentionPolicyV1 {
    fn default() -> Self {
        Self {
            schema_version: 1,
            keep_full_for_ticks: 2_048,
            keep_full_for_days: 7,
            keep_digests_forever: true,
            max_ess_bytes: 256 * 1024 * 1024,
            policy_marker: "retention_v1_default".to_string(),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetentionStats {
    pub redactions_total: u64,
    pub payload_bytes_pruned_total: u64,
}

pub fn data_class_for_payload(payload: &AuditPayload) -> DataClass {
    match payload {
        AuditPayload::CandidateSet(_) => DataClass::ScalarSummary,
        AuditPayload::EbmReasoning(_) | AuditPayload::EbmEnvelopeViolation(_) => {
            DataClass::ScalarSummary
        }
        AuditPayload::CapabilityIssuance(_) => DataClass::ScalarSummary,
        AuditPayload::Output(_) => DataClass::TextPayload,
        AuditPayload::PolicyProvenance(_)
        | AuditPayload::EbmConstraintProvenance(_)
        | AuditPayload::AuditCheckpoint(_) => DataClass::DigestOnly,
        AuditPayload::ToolRequest(_)
        | AuditPayload::ToolAuth(_)
        | AuditPayload::ToolExecution(_)
        | AuditPayload::SandboxCall(_)
        | AuditPayload::SandboxReply(_)
        | AuditPayload::Throttle(_)
        | AuditPayload::Emergency(_)
        | AuditPayload::RemoteCall(_)
        | AuditPayload::RemoteCallDenied(_)
        | AuditPayload::ComputeBudgetWindow(_)
        | AuditPayload::ComputeBudgetViolation(_)
        | AuditPayload::RetrievalDecision(_) => DataClass::ScalarSummary,
    }
}

pub fn apply_retention(
    records: &mut [ExperienceRecord],
    policy: &RetentionPolicyV1,
    now_tick: u64,
) -> RetentionStats {
    let mut stats = RetentionStats::default();
    for record in records {
        let t = record.time.tick.get();
        let age_ticks = now_tick.saturating_sub(t);
        if age_ticks <= policy.keep_full_for_ticks {
            continue;
        }
        if let ExperiencePayload::Audit(AuditPayload::Output(output)) = &mut record.payload {
            if output.redacted {
                continue;
            }
            if let Some(text) = output.text.take() {
                output.payload_len = Some(text.len() as u32);
                output.content_digest = crate::v1::compute_content_digest(&text);
                output.redacted = true;
                output.payload_classification = Some(PayloadClassification::Private);
                output.redaction_policy_marker = Some(policy.policy_marker.clone());
                stats.redactions_total = stats.redactions_total.saturating_add(1);
                stats.payload_bytes_pruned_total = stats
                    .payload_bytes_pruned_total
                    .saturating_add(text.len() as u64);
            }
        }
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v1::{AuditPayload, ExperienceId, ExperienceKind, ExperiencePayload, OutputRecord};
    use ucf_core::types::{SimTime, Tick, WindowId};
    use ucf_frames::v1::CorrelationId;

    #[test]
    fn retention_redacts_old_output_records() {
        let output = OutputRecord {
            schema_version: 2,
            decision_id: 1,
            candidate_id: 1,
            t: 1,
            output_class: 0,
            llm_backend_name: "toy".to_string(),
            llm_request_digest: [1; 32],
            llm_response_digest: [2; 32],
            token_count: 4,
            status: 0,
            finish_reason: 0,
            content_digest: [0; 32],
            text: Some("hello".to_string()),
            redacted: false,
            payload_len: None,
            payload_classification: None,
            redaction_policy_marker: None,
            evidence_chain_digest: [3; 32],
            lfm_readout_digest: None,
            lfm_uncertainty: None,
            lfm_stability: None,
            max_tokens_eff: 32,
            output_override: None,
            override_reasons: vec![],
        };
        let mut records = vec![ExperienceRecord {
            id: ExperienceId(1),
            time: SimTime {
                tick: Tick::new(1),
                window: WindowId::new(0),
            },
            corr: CorrelationId(1),
            kind: ExperienceKind::Output,
            payload: ExperiencePayload::Audit(AuditPayload::Output(output)),
            neuromod: None,
            iit_phi: None,
            decision_meta: None,
            compute_summary: None,
            hormone_record: None,
            neuro_record: None,
            delta_proposal_record: None,
            delta_evaluation_record: None,
            delta_recommendation_record: None,
            nsr_record: None,
            backend_pack_record: None,
            lfm_summary_record: None,
            lfm_window_record: None,
            ebm_tag: None,
            audit_prev_digest: None,
            audit_digest: None,
        }];

        let policy = RetentionPolicyV1 {
            keep_full_for_ticks: 0,
            ..RetentionPolicyV1::default()
        };
        let stats = apply_retention(&mut records, &policy, 10);
        assert_eq!(stats.redactions_total, 1);
        let ExperiencePayload::Audit(AuditPayload::Output(redacted)) = &records[0].payload else {
            panic!("expected output");
        };
        assert!(redacted.redacted);
        assert_eq!(redacted.text, None);
        assert_eq!(redacted.payload_len, Some(5));
        assert_ne!(redacted.content_digest, [0; 32]);
    }
}
