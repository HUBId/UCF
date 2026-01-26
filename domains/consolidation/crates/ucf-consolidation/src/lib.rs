#![forbid(unsafe_code)]

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use blake3::Hasher;
use prost::Message;
use ucf_archive::{ExperienceAppender, FileArchive, InMemoryArchive};
use ucf_archive_store::{ArchiveAppender, ArchiveStore, RecordKind, RecordMeta};
use ucf_attn_controller::AttentionWeights;
use ucf_bus::{BusPublisher, MessageEnvelope};
use ucf_commit::{
    commit_experience_record, commit_memory_macro, commit_memory_meso, commit_memory_micro,
    commit_milestone_macro, commit_milestone_meso, commit_milestone_micro, commit_replay_token,
    Commitment,
};
use ucf_consistency_engine::{ConsistencyReport, DriftBand};
use ucf_iit_monitor::IitReport;
use ucf_influence::{InfluenceOutputs, NodeId as InfluenceNodeId};
use ucf_predictive_coding::{SurpriseBand, SurpriseSignal};
use ucf_sleep_coordinator::{SleepStateHandle, SleepStateUpdater};
use ucf_structural_store::ReplayCaps;
use ucf_tcf_port::{PulseKind, RecursionBudget};
use ucf_types::consolidation::{
    MacroMilestone as MemoryMacroMilestone, MesoMilestone as MemoryMesoMilestone,
    MicroMilestone as MemoryMicroMilestone, MilestoneTier, ReplayApplied, ReplayScheduled,
    ReplayToken,
};
use ucf_types::v1::spec::{
    ExperienceRecord, MacroMilestone as ProtoMacroMilestone, MesoMilestone as ProtoMesoMilestone,
    MicroMilestone as ProtoMicroMilestone,
};
use ucf_types::{Digest32, NodeId};
use ucf_vector_index::{
    build_macro_finalized_envelope, build_record_appended_envelope, MacroMilestoneFinalized,
    RecordAppended,
};
use ucf_workspace::{Workspace, WorkspaceSignal};

#[derive(Clone, Debug)]
pub struct ConsolidationConfig {
    pub micro_window: usize,
    pub meso_window: usize,
    pub macro_window: usize,
    pub replay_budget: usize,
    pub replay_micro_cap: usize,
    pub replay_meso_cap: usize,
    pub replay_macro_cap: usize,
}

impl ConsolidationConfig {
    pub fn total_window(&self) -> usize {
        self.micro_window
            .saturating_mul(self.meso_window)
            .saturating_mul(self.macro_window)
            .max(self.micro_window)
    }

    pub fn apply_replay_caps(&mut self, caps: &ReplayCaps) {
        self.replay_micro_cap = usize::from(caps.micro_k);
        self.replay_meso_cap = usize::from(caps.meso_m);
        self.replay_macro_cap = usize::from(caps.macro_n);
    }
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            micro_window: 8,
            meso_window: 8,
            macro_window: 8,
            replay_budget: 64,
            replay_micro_cap: 8,
            replay_meso_cap: 4,
            replay_macro_cap: 2,
        }
    }
}

pub trait RecordSource {
    fn recent_records(&self, n: usize) -> Vec<ExperienceRecord>;
}

pub trait MilestoneSink {
    fn emit_micro(&self, mm: ProtoMicroMilestone);
    fn emit_meso(&self, mm: ProtoMesoMilestone);
    fn emit_macro(&self, mm: ProtoMacroMilestone);
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
    sleep_state: Option<SleepStateHandle>,
    index_publishers: Option<IndexEventPublishers>,
}

impl<'a, A: ExperienceAppender> ArchiveMilestoneSink<'a, A> {
    pub fn new(
        appender: &'a A,
        sleep_state: Option<SleepStateHandle>,
        index_publishers: Option<IndexEventPublishers>,
    ) -> Self {
        Self {
            appender,
            sleep_state,
            index_publishers,
        }
    }
}

impl<A: ExperienceAppender> MilestoneSink for ArchiveMilestoneSink<'_, A> {
    fn emit_micro(&self, mm: ProtoMicroMilestone) {
        let record = derived_record_for_micro(&mm);
        let evidence_id = self.appender.append(record.clone());
        self.publish_record_appended(&record);
        if let Some(state) = &self.sleep_state {
            if let Ok(mut guard) = state.lock() {
                guard.record_derived_record(evidence_id);
            }
        }
    }

    fn emit_meso(&self, mm: ProtoMesoMilestone) {
        let record = derived_record_for_meso(&mm);
        let evidence_id = self.appender.append(record.clone());
        self.publish_record_appended(&record);
        if let Some(state) = &self.sleep_state {
            if let Ok(mut guard) = state.lock() {
                guard.record_derived_record(evidence_id);
            }
        }
    }

    fn emit_macro(&self, mm: ProtoMacroMilestone) {
        let record = derived_record_for_macro(&mm);
        let evidence_id = self.appender.append(record.clone());
        self.publish_record_appended(&record);
        self.publish_macro_finalized(&record, &mm);
        if let Some(state) = &self.sleep_state {
            if let Ok(mut guard) = state.lock() {
                guard.record_derived_record(evidence_id);
            }
        }
    }
}

impl<A: ExperienceAppender> ArchiveMilestoneSink<'_, A> {
    fn publish_record_appended(&self, record: &ExperienceRecord) {
        let Some(publishers) = &self.index_publishers else {
            return;
        };
        let Some(bus) = &publishers.record_appended else {
            return;
        };
        let envelope = build_record_appended_envelope(record.clone(), publishers.node_id.clone());
        bus.publish(envelope);
    }

    fn publish_macro_finalized(&self, record: &ExperienceRecord, milestone: &ProtoMacroMilestone) {
        let Some(publishers) = &self.index_publishers else {
            return;
        };
        let Some(bus) = &publishers.macro_finalized else {
            return;
        };
        let envelope = build_macro_finalized_envelope(
            record.clone(),
            milestone.clone(),
            publishers.node_id.clone(),
            None,
        );
        bus.publish(envelope);
    }
}

#[derive(Clone)]
pub struct IndexEventPublishers {
    pub node_id: NodeId,
    pub macro_finalized: Option<
        std::sync::Arc<dyn BusPublisher<MessageEnvelope<MacroMilestoneFinalized>> + Send + Sync>,
    >,
    pub record_appended:
        Option<std::sync::Arc<dyn BusPublisher<MessageEnvelope<RecordAppended>> + Send + Sync>>,
}

impl IndexEventPublishers {
    pub fn new(node_id: NodeId) -> Self {
        Self {
            node_id,
            macro_finalized: None,
            record_appended: None,
        }
    }
}

pub struct ConsolidationKernel<S: RecordSource, A: ExperienceAppender> {
    config: ConsolidationConfig,
    source: S,
    appender: A,
    archive_store: Arc<dyn ArchiveStore + Send + Sync>,
    archive_appender: Mutex<ArchiveAppender>,
    sleep_state: Option<SleepStateHandle>,
    index_publishers: Option<IndexEventPublishers>,
    workspace: Option<Arc<Mutex<Workspace>>>,
}

pub struct ConsolidationCycleSummary {
    pub micro_count: usize,
    pub meso_count: usize,
    pub macro_count: usize,
}

impl<S: RecordSource, A: ExperienceAppender> ConsolidationKernel<S, A> {
    pub fn new(
        config: ConsolidationConfig,
        source: S,
        appender: A,
        archive_store: Arc<dyn ArchiveStore + Send + Sync>,
        sleep_state: Option<SleepStateHandle>,
        index_publishers: Option<IndexEventPublishers>,
        workspace: Option<Arc<Mutex<Workspace>>>,
    ) -> Self {
        Self {
            config,
            source,
            appender,
            archive_store,
            archive_appender: Mutex::new(ArchiveAppender::new()),
            sleep_state,
            index_publishers,
            workspace,
        }
    }

    pub fn update_replay_caps(&mut self, caps: &ReplayCaps) {
        self.config.apply_replay_caps(caps);
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

        let micros: Vec<ProtoMicroMilestone> = records
            .chunks(self.config.micro_window)
            .map(build_micro)
            .collect();
        let mesos: Vec<ProtoMesoMilestone> = micros
            .chunks(self.config.meso_window)
            .map(build_meso)
            .collect();
        let macros: Vec<ProtoMacroMilestone> = mesos
            .chunks(self.config.macro_window)
            .map(build_macro)
            .collect();

        let sink = ArchiveMilestoneSink::new(
            &self.appender,
            self.sleep_state.clone(),
            self.index_publishers.clone(),
        );
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

    pub fn run_sleep_replay(
        &self,
        pulse: PulseKind,
        context: SleepReplayContext,
        bias: &ReplayBias,
        budget: RecursionBudget,
    ) -> Option<ReplayOutcome> {
        if pulse != PulseKind::Sleep {
            return None;
        }
        let records = self.source.recent_records(self.config.total_window());
        let graph = build_memory_graph(&records, &self.config);
        let cascade = ReplayCascade::new(ReplayCascadeConfig::new(
            self.config.replay_micro_cap,
            self.config.replay_meso_cap,
            self.config.replay_macro_cap,
        ));
        let outcome = cascade.schedule(&graph, bias, budget, &context);

        {
            let mut appender = self.archive_appender.lock().expect("archive appender lock");
            for scheduled in &outcome.scheduled {
                let meta = RecordMeta {
                    cycle_id: context.cycle_id,
                    tier: scheduled.tier as u8,
                    flags: scheduled.budget,
                    boundary_commit: Digest32::new([0u8; 32]),
                };
                let record = appender.build_record_with_commit(
                    RecordKind::ReplayToken,
                    scheduled.commit,
                    meta,
                );
                self.archive_store.append(record);
            }
            for applied in &outcome.applied {
                let meta = RecordMeta {
                    cycle_id: context.cycle_id,
                    tier: applied.tier as u8,
                    flags: 0,
                    boundary_commit: Digest32::new([0u8; 32]),
                };
                let record = appender.build_record_with_commit(
                    RecordKind::ReplayApplied,
                    applied.effect_digest,
                    meta,
                );
                self.archive_store.append(record);
            }
        }

        let (micro_count, meso_count, macro_count) = outcome.counts();
        if let Some(state) = &self.sleep_state {
            if let Ok(mut guard) = state.lock() {
                guard.record_replay_summary(ucf_sleep_coordinator::SleepReplaySummary {
                    micro: u16::try_from(micro_count).unwrap_or(u16::MAX),
                    meso: u16::try_from(meso_count).unwrap_or(u16::MAX),
                    macro_: u16::try_from(macro_count).unwrap_or(u16::MAX),
                });
            }
        }
        if let Some(workspace) = &self.workspace {
            if let Ok(mut guard) = workspace.lock() {
                guard.publish(WorkspaceSignal::from_replay_summary(
                    micro_count,
                    meso_count,
                    macro_count,
                    Some(bias.attention.gain),
                    None,
                ));
            }
        }

        Some(outcome)
    }
}

pub fn build_micro(records: &[ExperienceRecord]) -> ProtoMicroMilestone {
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

    ProtoMicroMilestone {
        milestone_id,
        achieved_at_ms,
        label,
    }
}

pub fn build_meso(micros: &[ProtoMicroMilestone]) -> ProtoMesoMilestone {
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

    ProtoMesoMilestone {
        milestone_id,
        achieved_at_ms,
        label,
        micro_milestone_ids: micro_ids,
    }
}

pub fn build_macro(mesos: &[ProtoMesoMilestone]) -> ProtoMacroMilestone {
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

    ProtoMacroMilestone {
        milestone_id,
        achieved_at_ms,
        label,
        meso_milestone_ids: meso_ids,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplayWeights {
    pub surprise: u16,
    pub drift: u16,
    pub attention: u16,
    pub age: u16,
}

impl Default for ReplayWeights {
    fn default() -> Self {
        Self {
            surprise: 4,
            drift: 3,
            attention: 2,
            age: 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayBias {
    pub attention: AttentionWeights,
    pub iit: IitReport,
    pub consistency: ConsistencyReport,
    pub surprise: SurpriseSignal,
    pub rdc_bias: u16,
    pub wm_novelty: u16,
    pub influence: Option<InfluenceOutputs>,
}

impl ReplayBias {
    pub fn total_bias(&self) -> u16 {
        let iit_bias = self.iit_bias();
        let consistency_bias = self.consistency_bias();
        let surprise_bias = self.surprise_bias();
        self.attention
            .replay_bias
            .saturating_add(iit_bias)
            .saturating_add(consistency_bias)
            .saturating_add(surprise_bias)
            .saturating_add(self.rdc_bias)
            .min(10_000)
    }

    pub fn priority_gain(&self) -> u16 {
        self.attention
            .gain
            .saturating_add(self.iit_bias())
            .saturating_add(self.rdc_bias)
            .min(10_000)
    }

    pub fn influence_priority_boost(&self) -> i64 {
        let (replay_in, learning_in) = self.influence_values();
        let combined = i32::from(replay_in) + i32::from(learning_in);
        let scaled = combined / 4;
        i64::from(scaled.clamp(-2000, 2000))
    }

    pub fn influence_window_shift(&self) -> (i16, i16) {
        let (replay_in, learning_in) = self.influence_values();
        let replay_shift = influence_shift(replay_in);
        let learning_shift = influence_shift(learning_in);
        (replay_shift, learning_shift)
    }

    pub fn influence_macro_boost(&self) -> i64 {
        let (_, learning_in) = self.influence_values();
        let scaled = i32::from(learning_in) / 4;
        i64::from(scaled.clamp(-2000, 2000))
    }

    fn iit_bias(&self) -> u16 {
        match self.iit.band {
            ucf_iit_monitor::IitBand::Low => 1600,
            ucf_iit_monitor::IitBand::Medium => 800,
            ucf_iit_monitor::IitBand::High => 200,
        }
    }

    fn consistency_bias(&self) -> u16 {
        match self.consistency.band {
            DriftBand::Low => 200,
            DriftBand::Medium => 900,
            DriftBand::High => 1800,
            DriftBand::Critical => 2500,
        }
    }

    fn surprise_bias(&self) -> u16 {
        let band_bias: u16 = match self.surprise.band {
            SurpriseBand::Low => 300,
            SurpriseBand::Medium => 1200,
            SurpriseBand::High => 2400,
            SurpriseBand::Critical => 3600,
        };
        band_bias.saturating_add(self.surprise.score / 4)
    }

    fn influence_values(&self) -> (i16, i16) {
        let Some(outputs) = self.influence.as_ref() else {
            return (0, 0);
        };
        (
            outputs.node_value(InfluenceNodeId::Replay),
            outputs.node_value(InfluenceNodeId::Learning),
        )
    }
}

#[derive(Clone, Debug)]
pub struct ReplayCascadeConfig {
    pub micro_cap: usize,
    pub meso_cap: usize,
    pub macro_cap: usize,
    pub weights: ReplayWeights,
}

impl ReplayCascadeConfig {
    pub fn new(micro_cap: usize, meso_cap: usize, macro_cap: usize) -> Self {
        Self {
            micro_cap,
            meso_cap,
            macro_cap,
            weights: ReplayWeights::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ReplayCascade {
    config: ReplayCascadeConfig,
}

impl ReplayCascade {
    pub fn new(config: ReplayCascadeConfig) -> Self {
        Self { config }
    }

    pub fn schedule(
        &self,
        graph: &MemoryMilestoneGraph,
        bias: &ReplayBias,
        budget: RecursionBudget,
        context: &SleepReplayContext,
    ) -> ReplayOutcome {
        let (micro_cap, meso_cap, macro_cap) = self.adjust_caps(bias);
        let micro_candidates = self.select_micro(graph, bias, micro_cap);
        let meso_candidates = self.select_meso(graph, &micro_candidates, meso_cap);
        let macro_candidates = self.select_macro(graph, bias, &meso_candidates, macro_cap);

        let redaction = base_redaction(bias.total_bias());
        let tokens = build_tokens(
            &micro_candidates,
            &meso_candidates,
            &macro_candidates,
            budget,
            redaction,
        );
        let scheduled: Vec<ReplayScheduled> = tokens
            .iter()
            .map(|token| ReplayScheduled {
                tier: token.tier,
                target: token.target,
                budget: token.budget,
                redaction: token.redaction,
                commit: token.commit,
            })
            .collect();
        let applied: Vec<ReplayApplied> = tokens
            .iter()
            .map(|token| ReplayApplied {
                tier: token.tier,
                target: token.target,
                effect_digest: replay_effect_digest(
                    token,
                    &context.ssm_state,
                    &context.workspace_commit,
                ),
            })
            .collect();

        ReplayOutcome {
            tokens,
            scheduled,
            applied,
            selected_micros: micro_candidates
                .iter()
                .map(|candidate| candidate.commit)
                .collect(),
            selected_mesos: meso_candidates
                .iter()
                .map(|candidate| candidate.commit)
                .collect(),
            selected_macros: macro_candidates
                .iter()
                .map(|candidate| candidate.commit)
                .collect(),
        }
    }

    fn select_micro(
        &self,
        graph: &MemoryMilestoneGraph,
        bias: &ReplayBias,
        cap: usize,
    ) -> Vec<RankedDigest> {
        let mut scored: Vec<RankedDigest> = graph
            .micros
            .iter()
            .enumerate()
            .map(|(idx, micro)| {
                let score = micro_score(
                    bias,
                    &self.config.weights,
                    u16::try_from(idx).unwrap_or(u16::MAX),
                );
                RankedDigest {
                    score,
                    commit: micro.commit,
                }
            })
            .collect();

        sort_ranked(&mut scored);
        scored.truncate(cap);
        scored
    }

    fn select_meso(
        &self,
        graph: &MemoryMilestoneGraph,
        micros: &[RankedDigest],
        cap: usize,
    ) -> Vec<RankedDigest> {
        let mut lookup = HashMap::new();
        for micro in micros {
            lookup.insert(micro.commit, micro.score);
        }
        let mut scored = Vec::new();
        for meso in &graph.mesos {
            let mut score = 0i64;
            for micro in &meso.micros {
                if let Some(value) = lookup.get(micro) {
                    score = score.saturating_add(*value);
                }
            }
            if score > 0 {
                scored.push(RankedDigest {
                    score,
                    commit: meso.commit,
                });
            }
        }

        sort_ranked(&mut scored);
        scored.truncate(cap);
        scored
    }

    fn select_macro(
        &self,
        graph: &MemoryMilestoneGraph,
        bias: &ReplayBias,
        mesos: &[RankedDigest],
        cap: usize,
    ) -> Vec<RankedDigest> {
        let drift_high = matches!(bias.consistency.band, DriftBand::High | DriftBand::Critical);
        let selected_mesos: HashSet<Digest32> = mesos.iter().map(|meso| meso.commit).collect();
        let boost = bias.influence_macro_boost();
        let mut scored = Vec::new();
        for macro_ms in &graph.macros {
            let contains = macro_ms
                .mesos
                .iter()
                .any(|meso| selected_mesos.contains(meso));
            if drift_high && !contains {
                continue;
            }
            let score: i64 = if contains { 10_000 } else { 5_000 };
            scored.push(RankedDigest {
                score: score.saturating_add(boost),
                commit: macro_ms.commit,
            });
        }

        sort_ranked(&mut scored);
        if drift_high {
            scored.truncate(cap);
            return scored;
        }

        let mut diversified = Vec::new();
        let mut seen_traits = HashSet::new();
        for macro_ms in &graph.macros {
            if diversified.len() >= cap {
                break;
            }
            if seen_traits.insert(macro_ms.trait_updates) {
                diversified.push(RankedDigest {
                    score: 8_000i64.saturating_add(boost),
                    commit: macro_ms.commit,
                });
            }
        }
        if diversified.is_empty() {
            scored.truncate(cap);
            scored
        } else {
            diversified
        }
    }

    fn adjust_caps(&self, bias: &ReplayBias) -> (usize, usize, usize) {
        let (replay_shift, learning_shift) = bias.influence_window_shift();
        let micro_cap = adjust_cap(self.config.micro_cap, replay_shift);
        let macro_cap = adjust_cap(self.config.macro_cap, learning_shift);
        let meso_shift = (replay_shift + learning_shift) / 2;
        let meso_cap = adjust_cap(self.config.meso_cap, meso_shift);
        (micro_cap, meso_cap, macro_cap)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RankedDigest {
    score: i64,
    commit: Digest32,
}

fn sort_ranked(values: &mut [RankedDigest]) {
    values.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| a.commit.as_bytes().cmp(b.commit.as_bytes()))
    });
}

fn adjust_cap(base: usize, delta: i16) -> usize {
    let base_i = base as i32;
    let adjusted = (base_i + i32::from(delta)).clamp(0, base_i + 2);
    adjusted as usize
}

fn influence_shift(value: i16) -> i16 {
    let scaled = value / 3000;
    scaled.clamp(-1, 1)
}

fn micro_score(bias: &ReplayBias, weights: &ReplayWeights, age_rank: u16) -> i64 {
    let surprise = i64::from(bias.surprise.score);
    let drift = i64::from(bias.consistency.drift_score);
    let attention = i64::from(bias.priority_gain());
    let influence = bias.influence_priority_boost();
    let wm_bias = i64::from(bias.wm_novelty) / 10;
    let age = i64::from(age_rank);
    let w1 = i64::from(weights.surprise);
    let w2 = i64::from(weights.drift);
    let w3 = i64::from(weights.attention);
    let w4 = i64::from(weights.age);
    w1 * surprise + w2 * drift + w3 * attention + influence + wm_bias - w4 * age
}

fn base_redaction(bias_total: u16) -> u16 {
    10_000u16.saturating_sub(bias_total.min(10_000))
}

fn build_tokens(
    micros: &[RankedDigest],
    mesos: &[RankedDigest],
    macros: &[RankedDigest],
    budget: RecursionBudget,
    redaction: u16,
) -> Vec<ReplayToken> {
    let mut tokens = Vec::new();
    for micro in micros {
        tokens.push(make_token(
            MilestoneTier::Micro,
            micro.commit,
            budget.micro,
            redaction.saturating_sub(800),
        ));
    }
    for meso in mesos {
        tokens.push(make_token(
            MilestoneTier::Meso,
            meso.commit,
            budget.meso,
            redaction,
        ));
    }
    for macro_ms in macros {
        tokens.push(make_token(
            MilestoneTier::Macro,
            macro_ms.commit,
            budget.macro_,
            redaction.saturating_add(600),
        ));
    }
    tokens
}

fn make_token(tier: MilestoneTier, target: Digest32, budget: u16, redaction: u16) -> ReplayToken {
    let redaction = redaction.min(10_000);
    let mut token = ReplayToken {
        tier,
        target,
        budget,
        redaction,
        commit: Digest32::new([0u8; 32]),
    };
    let commit = commit_replay_token(&token);
    token.commit = commit.digest;
    token
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayOutcome {
    pub tokens: Vec<ReplayToken>,
    pub scheduled: Vec<ReplayScheduled>,
    pub applied: Vec<ReplayApplied>,
    pub selected_micros: Vec<Digest32>,
    pub selected_mesos: Vec<Digest32>,
    pub selected_macros: Vec<Digest32>,
}

impl ReplayOutcome {
    pub fn counts(&self) -> (usize, usize, usize) {
        (
            self.selected_micros.len(),
            self.selected_mesos.len(),
            self.selected_macros.len(),
        )
    }
}

#[derive(Clone, Debug)]
pub struct MemoryMilestoneGraph {
    pub micros: Vec<MemoryMicroMilestone>,
    pub mesos: Vec<MemoryMesoMilestone>,
    pub macros: Vec<MemoryMacroMilestone>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SleepReplayContext {
    pub cycle_id: u64,
    pub ssm_state: Digest32,
    pub workspace_commit: Digest32,
}

fn build_memory_graph(
    records: &[ExperienceRecord],
    config: &ConsolidationConfig,
) -> MemoryMilestoneGraph {
    if records.is_empty()
        || config.micro_window == 0
        || config.meso_window == 0
        || config.macro_window == 0
    {
        return MemoryMilestoneGraph {
            micros: Vec::new(),
            mesos: Vec::new(),
            macros: Vec::new(),
        };
    }

    let micros: Vec<MemoryMicroMilestone> = records
        .chunks(config.micro_window)
        .map(build_memory_micro)
        .collect();
    let mesos: Vec<MemoryMesoMilestone> = micros
        .chunks(config.meso_window)
        .map(build_memory_meso)
        .collect();
    let macros: Vec<MemoryMacroMilestone> = mesos
        .chunks(config.macro_window)
        .map(build_memory_macro)
        .collect();

    MemoryMilestoneGraph {
        micros,
        mesos,
        macros,
    }
}

fn build_memory_micro(records: &[ExperienceRecord]) -> MemoryMicroMilestone {
    let mut items: Vec<Digest32> = records
        .iter()
        .map(commit_experience_record)
        .map(|commitment| commitment.digest)
        .collect();
    normalize_digests(&mut items);
    let horm_profile = digest_list(b"ucf.consolidation.horm_profile.v1", &items);
    let mut micro = MemoryMicroMilestone {
        items,
        horm_profile,
        commit: Digest32::new([0u8; 32]),
    };
    micro.commit = commit_memory_micro(&micro).digest;
    micro
}

fn build_memory_meso(micros: &[MemoryMicroMilestone]) -> MemoryMesoMilestone {
    let mut micro_commits: Vec<Digest32> = micros.iter().map(|micro| micro.commit).collect();
    normalize_digests(&mut micro_commits);
    let topic_commit = digest_list(b"ucf.consolidation.topic_commit.v1", &micro_commits);
    let mut meso = MemoryMesoMilestone {
        micros: micro_commits,
        topic_commit,
        commit: Digest32::new([0u8; 32]),
    };
    meso.commit = commit_memory_meso(&meso).digest;
    meso
}

fn build_memory_macro(mesos: &[MemoryMesoMilestone]) -> MemoryMacroMilestone {
    let mut meso_commits: Vec<Digest32> = mesos.iter().map(|meso| meso.commit).collect();
    normalize_digests(&mut meso_commits);
    let trait_updates = digest_list(b"ucf.consolidation.trait_updates.v1", &meso_commits);
    let mut macro_ms = MemoryMacroMilestone {
        mesos: meso_commits,
        trait_updates,
        commit: Digest32::new([0u8; 32]),
    };
    macro_ms.commit = commit_memory_macro(&macro_ms).digest;
    macro_ms
}

fn digest_list(domain: &[u8], digests: &[Digest32]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(domain);
    for digest in digests {
        hasher.update(digest.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn normalize_digests(digests: &mut [Digest32]) {
    digests.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
}

fn replay_effect_digest(
    token: &ReplayToken,
    ssm_state: &Digest32,
    workspace: &Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.replay.effect.v1");
    hasher.update(token.commit.as_bytes());
    hasher.update(ssm_state.as_bytes());
    hasher.update(workspace.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[allow(dead_code)]
fn encode_replay_scheduled(event: &ReplayScheduled) -> Vec<u8> {
    let mut payload = Vec::with_capacity(2 + 1 + 2 + 2 + Digest32::LEN * 2);
    payload.extend_from_slice(b"RS");
    payload.push(event.tier as u8);
    payload.extend_from_slice(&event.budget.to_be_bytes());
    payload.extend_from_slice(&event.redaction.to_be_bytes());
    payload.extend_from_slice(event.target.as_bytes());
    payload.extend_from_slice(event.commit.as_bytes());
    payload
}

#[allow(dead_code)]
fn encode_replay_applied(event: &ReplayApplied) -> Vec<u8> {
    let mut payload = Vec::with_capacity(2 + 1 + Digest32::LEN * 2);
    payload.extend_from_slice(b"RA");
    payload.push(event.tier as u8);
    payload.extend_from_slice(event.target.as_bytes());
    payload.extend_from_slice(event.effect_digest.as_bytes());
    payload
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

fn derived_record_for_micro(micro: &ProtoMicroMilestone) -> ExperienceRecord {
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

fn derived_record_for_meso(meso: &ProtoMesoMilestone) -> ExperienceRecord {
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

fn derived_record_for_macro(macro_ms: &ProtoMacroMilestone) -> ExperienceRecord {
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
    use ucf_archive_store::InMemoryArchiveStore;
    use ucf_attn_controller::FocusChannel;
    use ucf_consistency_engine::DriftBand;
    use ucf_iit_monitor::IitBand;
    use ucf_predictive_coding::{SurpriseBand, SurpriseSignal};
    use ucf_types::v1::spec::{Digest, MacroMilestone};
    use ucf_workspace::{Workspace, WorkspaceConfig};

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

    fn sample_bias(drift_band: DriftBand, drift_score: u16) -> ReplayBias {
        ReplayBias {
            attention: AttentionWeights {
                channel: FocusChannel::Memory,
                gain: 3200,
                noise_suppress: 1000,
                replay_bias: 4000,
                commit: Digest32::new([1u8; 32]),
            },
            iit: IitReport {
                phi: 2400,
                band: IitBand::Low,
                commit: Digest32::new([2u8; 32]),
            },
            consistency: ConsistencyReport {
                drift_score,
                band: drift_band,
                commit: Digest32::new([3u8; 32]),
            },
            surprise: SurpriseSignal {
                score: 4200,
                band: SurpriseBand::Medium,
                commit: Digest32::new([4u8; 32]),
            },
            rdc_bias: 600,
            wm_novelty: 0,
            influence: None,
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
    fn kernel_appends_derived_milestones() {
        let archive = InMemoryArchive::new();
        for idx in 0..8 {
            archive.append(sample_record(&format!("rec-{idx}"), 100 + idx, None));
        }

        let handle = InMemoryArchiveHandle::new(&archive);
        let archive_store = Arc::new(InMemoryArchiveStore::new());
        let config = ConsolidationConfig {
            micro_window: 2,
            meso_window: 2,
            macro_window: 2,
            replay_budget: 8,
            replay_micro_cap: 4,
            replay_meso_cap: 2,
            replay_macro_cap: 1,
        };
        let kernel =
            ConsolidationKernel::new(config, handle, handle, archive_store, None, None, None);

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

    #[test]
    fn replay_selection_is_deterministic_for_same_inputs() {
        let records: Vec<ExperienceRecord> = (0..12)
            .map(|idx| sample_record(&format!("rec-{idx}"), 100 + idx, None))
            .collect();
        let config = ConsolidationConfig {
            micro_window: 2,
            meso_window: 2,
            macro_window: 2,
            replay_budget: 8,
            replay_micro_cap: 3,
            replay_meso_cap: 2,
            replay_macro_cap: 1,
        };
        let graph = build_memory_graph(&records, &config);
        let bias = sample_bias(DriftBand::Medium, 4200);
        let cascade = ReplayCascade::new(ReplayCascadeConfig::new(3, 2, 1));
        let context = SleepReplayContext {
            cycle_id: 1,
            ssm_state: Digest32::new([5u8; 32]),
            workspace_commit: Digest32::new([6u8; 32]),
        };
        let budget = RecursionBudget::new(2, 2, 1);

        let first = cascade.schedule(&graph, &bias, budget, &context);
        let second = cascade.schedule(&graph, &bias, budget, &context);

        assert_eq!(first.tokens, second.tokens);
        assert_eq!(first.scheduled, second.scheduled);
    }

    #[test]
    fn replay_influence_biases_selection() {
        let records: Vec<ExperienceRecord> = (0..12)
            .map(|idx| sample_record(&format!("rec-{idx}"), 100 + idx, None))
            .collect();
        let config = ConsolidationConfig {
            micro_window: 2,
            meso_window: 2,
            macro_window: 2,
            replay_budget: 8,
            replay_micro_cap: 2,
            replay_meso_cap: 1,
            replay_macro_cap: 1,
        };
        let graph = build_memory_graph(&records, &config);
        let cascade = ReplayCascade::new(ReplayCascadeConfig::new(2, 1, 1));
        let context = SleepReplayContext {
            cycle_id: 3,
            ssm_state: Digest32::new([9u8; 32]),
            workspace_commit: Digest32::new([10u8; 32]),
        };
        let budget = RecursionBudget::new(2, 1, 1);

        let base_bias = sample_bias(DriftBand::Low, 1200);
        let base_outcome = cascade.schedule(&graph, &base_bias, budget, &context);

        let influence = InfluenceOutputs {
            node_in: vec![
                (InfluenceNodeId::Replay, 6000),
                (InfluenceNodeId::Learning, -500),
            ],
            commit: Digest32::new([11u8; 32]),
        };
        let biased = ReplayBias {
            influence: Some(influence),
            ..base_bias
        };
        let influenced_outcome = cascade.schedule(&graph, &biased, budget, &context);

        assert!(influenced_outcome.selected_micros.len() >= base_outcome.selected_micros.len());
    }

    #[test]
    fn cascade_respects_caps() {
        let records: Vec<ExperienceRecord> = (0..16)
            .map(|idx| sample_record(&format!("rec-{idx}"), 100 + idx, None))
            .collect();
        let config = ConsolidationConfig {
            micro_window: 2,
            meso_window: 2,
            macro_window: 2,
            replay_budget: 8,
            replay_micro_cap: 2,
            replay_meso_cap: 1,
            replay_macro_cap: 1,
        };
        let graph = build_memory_graph(&records, &config);
        let bias = sample_bias(DriftBand::Medium, 3200);
        let cascade = ReplayCascade::new(ReplayCascadeConfig::new(2, 1, 1));
        let context = SleepReplayContext {
            cycle_id: 2,
            ssm_state: Digest32::new([7u8; 32]),
            workspace_commit: Digest32::new([8u8; 32]),
        };
        let budget = RecursionBudget::new(2, 1, 1);
        let outcome = cascade.schedule(&graph, &bias, budget, &context);

        assert!(outcome.selected_micros.len() <= 2);
        assert!(outcome.selected_mesos.len() <= 1);
        assert!(outcome.selected_macros.len() <= 1);
    }

    #[test]
    fn replay_token_contains_no_raw_text() {
        let token = make_token(MilestoneTier::Micro, Digest32::new([9u8; 32]), 2, 5000);
        let scheduled = ReplayScheduled {
            tier: token.tier,
            target: token.target,
            budget: token.budget,
            redaction: token.redaction,
            commit: token.commit,
        };
        let payload = encode_replay_scheduled(&scheduled);
        let secret = b"VERY_SECRET_RAW_TEXT_DO_NOT_INCLUDE";
        assert!(!payload.windows(secret.len()).any(|window| window == secret));
    }

    #[test]
    fn drift_high_prioritizes_macro_selection() {
        let records: Vec<ExperienceRecord> = (0..12)
            .map(|idx| sample_record(&format!("rec-{idx}"), 100 + idx, None))
            .collect();
        let config = ConsolidationConfig {
            micro_window: 2,
            meso_window: 2,
            macro_window: 2,
            replay_budget: 8,
            replay_micro_cap: 3,
            replay_meso_cap: 2,
            replay_macro_cap: 1,
        };
        let graph = build_memory_graph(&records, &config);
        let bias = sample_bias(DriftBand::High, 8000);
        let cascade = ReplayCascade::new(ReplayCascadeConfig::new(3, 2, 1));
        let context = SleepReplayContext {
            cycle_id: 3,
            ssm_state: Digest32::new([1u8; 32]),
            workspace_commit: Digest32::new([2u8; 32]),
        };
        let budget = RecursionBudget::new(2, 2, 1);
        let outcome = cascade.schedule(&graph, &bias, budget, &context);

        assert!(!outcome.selected_macros.is_empty());
    }

    #[test]
    fn sleep_pulse_produces_replay_scheduled_events() {
        let archive = InMemoryArchive::new();
        for idx in 0..8 {
            archive.append(sample_record(&format!("rec-{idx}"), 100 + idx, None));
        }
        let handle = InMemoryArchiveHandle::new(&archive);
        let archive_store = Arc::new(InMemoryArchiveStore::new());
        let workspace = Arc::new(Mutex::new(Workspace::new(WorkspaceConfig {
            cap: 4,
            broadcast_cap: 4,
        })));
        let config = ConsolidationConfig {
            micro_window: 2,
            meso_window: 2,
            macro_window: 2,
            replay_budget: 8,
            replay_micro_cap: 2,
            replay_meso_cap: 1,
            replay_macro_cap: 1,
        };
        let kernel = ConsolidationKernel::new(
            config,
            handle,
            handle,
            archive_store,
            None,
            None,
            Some(workspace),
        );
        let context = SleepReplayContext {
            cycle_id: 42,
            ssm_state: Digest32::new([3u8; 32]),
            workspace_commit: Digest32::new([4u8; 32]),
        };
        let bias = sample_bias(DriftBand::Medium, 4200);
        let budget = RecursionBudget::new(2, 1, 1);

        let outcome = kernel
            .run_sleep_replay(PulseKind::Sleep, context, &bias, budget)
            .expect("sleep replay");

        assert!(!outcome.scheduled.is_empty());
    }
}
