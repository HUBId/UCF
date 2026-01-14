#![forbid(unsafe_code)]

use ucf_crypto as _;
use ucf_types::v1::spec::ExperienceRecord;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsistencyVerdict {
    Accept,
    Damp,
    Reject,
}

impl ConsistencyVerdict {
    pub fn as_u8(self) -> u8 {
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PolicyWeights;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PolicyRule {
    DenyReplayIfDecisionClass { class: u16 },
    DenyReplayIfIntensityBelow { min: u16 },
    DenyIsmUpsertIfScoreBelow { min_score: u16 },
    AllowAll,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyEcology {
    version: u32,
    rules: Vec<PolicyRule>,
    weights: PolicyWeights,
}

impl PolicyEcology {
    pub fn new(version: u32, rules: Vec<PolicyRule>, weights: PolicyWeights) -> Self {
        Self {
            version,
            rules,
            weights,
        }
    }

    pub fn allow_all() -> Self {
        Self::new(1, vec![PolicyRule::AllowAll], PolicyWeights::default())
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn rules(&self) -> &[PolicyRule] {
        &self.rules
    }

    pub fn weights(&self) -> &PolicyWeights {
        &self.weights
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DefaultPolicyEcology {
    inner: PolicyEcology,
}

impl DefaultPolicyEcology {
    pub fn new() -> Self {
        Self {
            inner: PolicyEcology::allow_all(),
        }
    }

    pub fn inner(&self) -> &PolicyEcology {
        &self.inner
    }
}

impl Default for DefaultPolicyEcology {
    fn default() -> Self {
        Self::new()
    }
}

pub trait ReplayGate {
    fn allow_replay(&self, rec: &ExperienceRecord) -> bool;
}

pub trait GeistGate {
    fn allow_ism_upsert(&self, report: &ConsistencyReport) -> bool;
}

impl ReplayGate for PolicyEcology {
    fn allow_replay(&self, rec: &ExperienceRecord) -> bool {
        let decision_class = record_decision_class(rec);
        let intensity = record_intensity(rec);
        for rule in &self.rules {
            match *rule {
                PolicyRule::DenyReplayIfDecisionClass { class } => {
                    if decision_class == Some(class) {
                        return false;
                    }
                }
                PolicyRule::DenyReplayIfIntensityBelow { min } => {
                    if intensity < min {
                        return false;
                    }
                }
                PolicyRule::DenyIsmUpsertIfScoreBelow { .. } => {}
                PolicyRule::AllowAll => {}
            }
        }
        true
    }
}

impl ReplayGate for DefaultPolicyEcology {
    fn allow_replay(&self, rec: &ExperienceRecord) -> bool {
        self.inner.allow_replay(rec)
    }
}

impl GeistGate for PolicyEcology {
    fn allow_ism_upsert(&self, report: &ConsistencyReport) -> bool {
        for rule in &self.rules {
            match *rule {
                PolicyRule::DenyIsmUpsertIfScoreBelow { min_score } => {
                    if report.score < min_score {
                        return false;
                    }
                }
                PolicyRule::DenyReplayIfDecisionClass { .. }
                | PolicyRule::DenyReplayIfIntensityBelow { .. }
                | PolicyRule::AllowAll => {}
            }
        }
        true
    }
}

impl GeistGate for DefaultPolicyEcology {
    fn allow_ism_upsert(&self, report: &ConsistencyReport) -> bool {
        self.inner.allow_ism_upsert(report)
    }
}

fn record_intensity(rec: &ExperienceRecord) -> u16 {
    rec.digest
        .as_ref()
        .and_then(|digest| digest.algo_id)
        .unwrap_or(0) as u16
}

fn record_decision_class(rec: &ExperienceRecord) -> Option<u16> {
    parse_decision_class_from_payload(&rec.payload)
}

fn parse_decision_class_from_payload(payload: &[u8]) -> Option<u16> {
    let text = std::str::from_utf8(payload).ok()?;
    for field in text.split(';') {
        if let Some(value) = field.strip_prefix("decision_kind=") {
            if let Ok(parsed) = value.parse::<u16>() {
                return Some(parsed);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_types::v1::spec::Digest;

    fn record_with_intensity(intensity: u32) -> ExperienceRecord {
        ExperienceRecord {
            record_id: "rec-1".to_string(),
            observed_at_ms: 1,
            subject_id: "subject".to_string(),
            payload: Vec::new(),
            digest: Some(Digest {
                algorithm: "intensity".to_string(),
                value: Vec::new(),
                algo_id: Some(intensity),
                domain: None,
                value_32: None,
            }),
            vrf_tag: None,
            proof_ref: None,
        }
    }

    #[test]
    fn deny_replay_when_intensity_below_minimum() {
        let ecology = PolicyEcology::new(
            1,
            vec![PolicyRule::DenyReplayIfIntensityBelow { min: 5 }],
            PolicyWeights::default(),
        );
        let record = record_with_intensity(3);

        assert!(!ecology.allow_replay(&record));
    }

    #[test]
    fn deny_ism_upsert_when_score_below_minimum() {
        let ecology = PolicyEcology::new(
            1,
            vec![PolicyRule::DenyIsmUpsertIfScoreBelow { min_score: 4 }],
            PolicyWeights::default(),
        );
        let report = ConsistencyReport {
            score: 2,
            verdict: ConsistencyVerdict::Accept,
            matched_anchors: 0,
        };

        assert!(!ecology.allow_ism_upsert(&report));
    }
}
