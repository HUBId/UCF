#![forbid(unsafe_code)]

use std::sync::Arc;

use prost::Message;
use ucf_archive::{ExperienceAppender, FileArchive, InMemoryArchive};
use ucf_commit::{
    commit_experience_record, commit_milestone_macro, commit_milestone_meso,
    commit_milestone_micro, Commitment,
};
use ucf_policy_ecology::{DefaultPolicyEcology, ReplayGate};
use ucf_types::v1::spec::{ExperienceRecord, MacroMilestone, MesoMilestone, MicroMilestone};

#[derive(Clone, Debug)]
pub struct ConsolidationConfig {
    pub micro_window: usize,
    pub meso_window: usize,
    pub macro_window: usize,
    pub replay_budget: usize,
}

impl ConsolidationConfig {
    pub fn total_window(&self) -> usize {
        self.micro_window
            .saturating_mul(self.meso_window)
            .saturating_mul(self.macro_window)
            .max(self.micro_window)
    }
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            micro_window: 8,
            meso_window: 8,
            macro_window: 8,
            replay_budget: 64,
        }
    }
}

pub trait RecordSource {
    fn recent_records(&self, n: usize) -> Vec<ExperienceRecord>;
}

pub trait MilestoneSink {
    fn emit_micro(&self, mm: MicroMilestone);
    fn emit_meso(&self, mm: MesoMilestone);
    fn emit_macro(&self, mm: MacroMilestone);
}

#[derive(Clone, Copy)]
pub struct InMemoryArchiveHandle<'a> {
    archive: &'a InMemoryArchive,
}

impl<'a> InMemoryArchiveHandle<'a> {
    pub fn new(archive: &'a InMemoryArchive) -> Self {
        Self { archive }
    }
}

impl RecordSource for InMemoryArchiveHandle<'_> {
    fn recent_records(&self, n: usize) -> Vec<ExperienceRecord> {
        let mut records = self
            .archive
            .list()
            .into_iter()
            .filter_map(|entry| entry.proof)
            .filter_map(|proof| ExperienceRecord::decode(proof.payload.as_slice()).ok())
            .collect::<Vec<_>>();
        if records.len() > n {
            records.drain(0..records.len() - n);
        }
        records
    }
}

impl ExperienceAppender for InMemoryArchiveHandle<'_> {
    fn append_with_proof(
        &self,
        rec: ExperienceRecord,
        proof: Option<ucf_types::v1::spec::ProofEnvelope>,
    ) -> ucf_types::EvidenceId {
        self.archive.append_with_proof(rec, proof)
    }
}

impl RecordSource for FileArchive {
    fn recent_records(&self, _n: usize) -> Vec<ExperienceRecord> {
        // TODO: integrate with file-backed evidence store.
        Vec::new()
    }
}

pub struct ArchiveMilestoneSink<'a, A: ExperienceAppender> {
    appender: &'a A,
}

impl<'a, A: ExperienceAppender> ArchiveMilestoneSink<'a, A> {
    pub fn new(appender: &'a A) -> Self {
        Self { appender }
    }
}

impl<A: ExperienceAppender> MilestoneSink for ArchiveMilestoneSink<'_, A> {
    fn emit_micro(&self, mm: MicroMilestone) {
        let record = derived_record_for_micro(&mm);
        self.appender.append(record);
    }

    fn emit_meso(&self, mm: MesoMilestone) {
        let record = derived_record_for_meso(&mm);
        self.appender.append(record);
    }

    fn emit_macro(&self, mm: MacroMilestone) {
        let record = derived_record_for_macro(&mm);
        self.appender.append(record);
    }
}

pub struct ConsolidationKernel<S: RecordSource, A: ExperienceAppender> {
    config: ConsolidationConfig,
    source: S,
    appender: A,
}

pub struct ConsolidationCycleSummary {
    pub micro_count: usize,
    pub meso_count: usize,
    pub macro_count: usize,
}

impl<S: RecordSource, A: ExperienceAppender> ConsolidationKernel<S, A> {
    pub fn new(config: ConsolidationConfig, source: S, appender: A) -> Self {
        Self {
            config,
            source,
            appender,
        }
    }

    pub fn run_one_cycle(&self) -> ConsolidationCycleSummary {
        if self.config.micro_window == 0
            || self.config.meso_window == 0
            || self.config.macro_window == 0
        {
            return ConsolidationCycleSummary {
                micro_count: 0,
                meso_count: 0,
                macro_count: 0,
            };
        }

        let records = self.source.recent_records(self.config.total_window());
        if records.is_empty() {
            return ConsolidationCycleSummary {
                micro_count: 0,
                meso_count: 0,
                macro_count: 0,
            };
        }

        let micros: Vec<MicroMilestone> = records
            .chunks(self.config.micro_window)
            .map(build_micro)
            .collect();
        let mesos: Vec<MesoMilestone> = micros
            .chunks(self.config.meso_window)
            .map(build_meso)
            .collect();
        let macros: Vec<MacroMilestone> = mesos
            .chunks(self.config.macro_window)
            .map(build_macro)
            .collect();

        let sink = ArchiveMilestoneSink::new(&self.appender);
        for micro in &micros {
            sink.emit_micro(micro.clone());
        }
        for meso in &mesos {
            sink.emit_meso(meso.clone());
        }
        for macro_ms in &macros {
            sink.emit_macro(macro_ms.clone());
        }

        ConsolidationCycleSummary {
            micro_count: micros.len(),
            meso_count: mesos.len(),
            macro_count: macros.len(),
        }
    }
}

pub fn build_micro(records: &[ExperienceRecord]) -> MicroMilestone {
    let commitments: Vec<Commitment> = records.iter().map(commit_experience_record).collect();
    let commitment_hexes: Vec<String> = commitments.iter().map(commitment_hex).collect();
    let milestone_id = commitments
        .first()
        .map(|commitment| format!("micro-{}", commitment_hex(commitment)))
        .unwrap_or_else(|| "micro-empty".to_string());
    let achieved_at_ms = records
        .iter()
        .map(|record| record.observed_at_ms)
        .max()
        .unwrap_or(0);
    let label = format!("micro:[{}]", commitment_hexes.join(","));

    MicroMilestone {
        milestone_id,
        achieved_at_ms,
        label,
    }
}

pub fn build_meso(micros: &[MicroMilestone]) -> MesoMilestone {
    let micro_ids: Vec<String> = micros
        .iter()
        .map(|micro| micro.milestone_id.clone())
        .collect();
    let micro_commitments: Vec<Commitment> = micros.iter().map(commit_milestone_micro).collect();
    let commitment_hexes: Vec<String> = micro_commitments.iter().map(commitment_hex).collect();
    let milestone_id = micro_commitments
        .first()
        .map(|commitment| format!("meso-{}", commitment_hex(commitment)))
        .unwrap_or_else(|| "meso-empty".to_string());
    let achieved_at_ms = micros
        .iter()
        .map(|micro| micro.achieved_at_ms)
        .max()
        .unwrap_or(0);
    let label = format!("meso:[{}]", commitment_hexes.join(","));

    MesoMilestone {
        milestone_id,
        achieved_at_ms,
        label,
        micro_milestone_ids: micro_ids,
    }
}

pub fn build_macro(mesos: &[MesoMilestone]) -> MacroMilestone {
    let meso_ids: Vec<String> = mesos.iter().map(|meso| meso.milestone_id.clone()).collect();
    let meso_commitments: Vec<Commitment> = mesos.iter().map(commit_milestone_meso).collect();
    let commitment_hexes: Vec<String> = meso_commitments.iter().map(commitment_hex).collect();
    let milestone_id = meso_commitments
        .first()
        .map(|commitment| format!("macro-{}", commitment_hex(commitment)))
        .unwrap_or_else(|| "macro-empty".to_string());
    let achieved_at_ms = mesos
        .iter()
        .map(|meso| meso.achieved_at_ms)
        .max()
        .unwrap_or(0);
    let label = format!("macro:[{}]", commitment_hexes.join(","));

    MacroMilestone {
        milestone_id,
        achieved_at_ms,
        label,
        meso_milestone_ids: meso_ids,
    }
}

pub struct ReplayScheduler {
    pub budget: usize,
    gate: Arc<dyn ReplayGate + Send + Sync>,
}

impl ReplayScheduler {
    pub fn new(budget: usize) -> Self {
        Self::new_with_gate(budget, Arc::new(DefaultPolicyEcology::default()))
    }

    pub fn new_with_gate(budget: usize, gate: Arc<dyn ReplayGate + Send + Sync>) -> Self {
        Self { budget, gate }
    }

    pub fn select_for_replay(&self, records: &[ExperienceRecord]) -> Vec<ExperienceRecord> {
        self.select_for_replay_with_gate(records, self.gate.as_ref())
    }

    pub fn select_for_replay_with_gate(
        &self,
        records: &[ExperienceRecord],
        gate: &dyn ReplayGate,
    ) -> Vec<ExperienceRecord> {
        let mut scored: Vec<(u64, [u8; 32], ExperienceRecord)> = records
            .iter()
            .filter(|record| gate.allow_replay(record))
            .map(|record| {
                let priority = record_priority(record);
                let commitment = commit_experience_record(record);
                (priority, *commitment.digest.as_bytes(), record.clone())
            })
            .collect();

        scored.sort_by(|(priority_a, digest_a, _), (priority_b, digest_b, _)| {
            priority_b
                .cmp(priority_a)
                .then_with(|| digest_a.cmp(digest_b))
        });

        scored
            .into_iter()
            .take(self.budget)
            .map(|(_, _, record)| record)
            .collect()
    }
}

fn record_priority(record: &ExperienceRecord) -> u64 {
    record
        .digest
        .as_ref()
        .and_then(|digest| digest.algo_id)
        .unwrap_or(0) as u64
}

fn commitment_hex(commitment: &Commitment) -> String {
    hex_encode(commitment.digest.as_bytes())
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn derived_record_for_micro(micro: &MicroMilestone) -> ExperienceRecord {
    let commitment = commit_milestone_micro(micro);
    ExperienceRecord {
        record_id: format!("derived-micro-{}", commitment_hex(&commitment)),
        observed_at_ms: micro.achieved_at_ms,
        subject_id: "consolidation".to_string(),
        payload: micro.encode_to_vec(),
        digest: None,
        vrf_tag: None,
        proof_ref: None,
    }
}

fn derived_record_for_meso(meso: &MesoMilestone) -> ExperienceRecord {
    let commitment = commit_milestone_meso(meso);
    ExperienceRecord {
        record_id: format!("derived-meso-{}", commitment_hex(&commitment)),
        observed_at_ms: meso.achieved_at_ms,
        subject_id: "consolidation".to_string(),
        payload: meso.encode_to_vec(),
        digest: None,
        vrf_tag: None,
        proof_ref: None,
    }
}

fn derived_record_for_macro(macro_ms: &MacroMilestone) -> ExperienceRecord {
    let commitment = commit_milestone_macro(macro_ms);
    ExperienceRecord {
        record_id: format!("derived-macro-{}", commitment_hex(&commitment)),
        observed_at_ms: macro_ms.achieved_at_ms,
        subject_id: "consolidation".to_string(),
        payload: macro_ms.encode_to_vec(),
        digest: None,
        vrf_tag: None,
        proof_ref: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use ucf_archive::{ExperienceAppender, InMemoryArchive};
    use ucf_policy_ecology::{PolicyEcology, PolicyRule, PolicyWeights};
    use ucf_types::v1::spec::Digest;

    fn sample_record(id: &str, observed_at_ms: u64, priority: Option<u32>) -> ExperienceRecord {
        ExperienceRecord {
            record_id: id.to_string(),
            observed_at_ms,
            subject_id: "subject".to_string(),
            payload: vec![1, 2, 3],
            digest: priority.map(|priority| Digest {
                algorithm: "priority".to_string(),
                value: vec![],
                algo_id: Some(priority),
                domain: None,
                value_32: None,
            }),
            vrf_tag: None,
            proof_ref: None,
        }
    }

    #[test]
    fn builds_milestones_deterministically() {
        let records = vec![
            sample_record("rec-1", 10, None),
            sample_record("rec-2", 11, None),
        ];

        let micro_a = build_micro(&records);
        let micro_b = build_micro(&records);
        assert_eq!(micro_a, micro_b);
        assert_eq!(
            commit_milestone_micro(&micro_a),
            commit_milestone_micro(&micro_b)
        );

        let meso_a = build_meso(std::slice::from_ref(&micro_a));
        let meso_b = build_meso(std::slice::from_ref(&micro_b));
        assert_eq!(
            commit_milestone_meso(&meso_a),
            commit_milestone_meso(&meso_b)
        );

        let macro_a = build_macro(std::slice::from_ref(&meso_a));
        let macro_b = build_macro(std::slice::from_ref(&meso_b));
        assert_eq!(
            commit_milestone_macro(&macro_a),
            commit_milestone_macro(&macro_b)
        );
    }

    #[test]
    fn replay_selection_is_deterministic() {
        let records = vec![
            sample_record("rec-1", 10, Some(1)),
            sample_record("rec-2", 11, Some(3)),
            sample_record("rec-3", 12, Some(3)),
        ];
        let scheduler = ReplayScheduler::new(2);

        let selected = scheduler.select_for_replay(&records);
        let selected_again = scheduler.select_for_replay(&records);

        assert_eq!(selected, selected_again);
        assert_eq!(selected.len(), 2);
        assert!(selected[0].record_id != selected[1].record_id);
    }

    #[test]
    fn replay_selection_respects_intensity_gate() {
        let records = vec![
            sample_record("rec-low", 10, Some(1)),
            sample_record("rec-high", 11, Some(5)),
        ];
        let policy = PolicyEcology::new(
            1,
            vec![PolicyRule::DenyReplayIfIntensityBelow { min: 3 }],
            PolicyWeights::default(),
        );
        let scheduler = ReplayScheduler::new_with_gate(2, Arc::new(policy));

        let selected = scheduler.select_for_replay(&records);

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].record_id, "rec-high");
    }

    #[test]
    fn kernel_appends_derived_milestones() {
        let archive = InMemoryArchive::new();
        for idx in 0..8 {
            archive.append(sample_record(&format!("rec-{idx}"), 100 + idx, None));
        }

        let handle = InMemoryArchiveHandle::new(&archive);
        let config = ConsolidationConfig {
            micro_window: 2,
            meso_window: 2,
            macro_window: 2,
            replay_budget: 8,
        };
        let kernel = ConsolidationKernel::new(config, handle, handle);

        let before_len = archive.list().len();
        let summary = kernel.run_one_cycle();
        let after_len = archive.list().len();

        assert_eq!(summary.micro_count, 4);
        assert_eq!(summary.meso_count, 2);
        assert_eq!(summary.macro_count, 1);
        assert_eq!(after_len, before_len + 7);

        let macro_records: Vec<ExperienceRecord> = archive
            .list()
            .into_iter()
            .filter_map(|entry| entry.proof)
            .filter_map(|proof| ExperienceRecord::decode(proof.payload.as_slice()).ok())
            .filter(|record| record.record_id.starts_with("derived-macro"))
            .collect();

        assert_eq!(macro_records.len(), 1);
        let macro_ms = MacroMilestone::decode(macro_records[0].payload.as_slice())
            .expect("decode macro milestone");
        assert_eq!(macro_ms.meso_milestone_ids.len(), 2);
    }
}
