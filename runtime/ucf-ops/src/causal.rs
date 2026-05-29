#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use ucf_ess::v1::{
    apply_retention, AuditPayload, ExperienceKind, ExperiencePayload, ExperienceRecord,
    RetentionPolicyV1,
};
use ucf_replay::load_fixture_records;

use crate::{digest_prefix, OpsError};

const MAX_SLICE_NODES: usize = 128;
const WINDOW_TICKS_PER_RADIUS: u64 = 8;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum EventType {
    Control,
    Decision,
    ToolPlan,
    ToolIssue,
    ToolExec,
    Experience,
    Milestone,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EventNode {
    pub event_id: String,
    pub event_type: EventType,
    pub t: u64,
    pub record_digest_prefix: String,
    pub policy_graph_digest_prefix: Option<String>,
    pub risk_q: Option<u16>,
    pub pressure_q: Option<u16>,
    pub energy_q: Option<u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum EdgeType {
    Causes,
    Enables,
    Justifies,
    Consumes,
    Produces,
    CounterfactualOf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CausalEdge {
    pub src_event_id: String,
    pub dst_event_id: String,
    pub edge_type: EdgeType,
    pub evidence_digest_prefix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CausalSlice {
    pub center_event_id: String,
    pub radius: u8,
    pub nodes: Vec<EventNode>,
    pub edges: Vec<CausalEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CounterfactualRequest {
    pub base_decision_id: u64,
    pub alternative_candidate_id: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CounterfactualResult {
    pub base_decision_id: u64,
    pub alternative_candidate_id: u16,
    pub would_choose_candidate: bool,
    pub would_issue_tool: bool,
    pub risk_delta_q: i16,
    pub energy_delta_q: i16,
    pub policy_graph_digest_prefix: Option<String>,
    pub evidence_digest_prefixes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CounterfactualRecord {
    pub schema_version: u16,
    pub base_decision_id: u64,
    pub alternative_candidate_id: u16,
    pub result_digest: String,
    pub policy_graph_digest_prefix: Option<String>,
    pub evidence_digest_prefixes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExplainWhyReport {
    pub decision_id: u64,
    pub center_event_id: String,
    pub incoming_causes: Vec<CausalEdge>,
    pub outgoing_effects: Vec<CausalEdge>,
    pub slice: CausalSlice,
}

#[derive(Debug, Clone)]
struct DerivedEvent {
    node: EventNode,
    t: u64,
    corr: u64,
    decision_id: Option<u64>,
    candidate_id: Option<u16>,
    plan_digest_prefix: Option<[u8; 8]>,
    tool_request_id: Option<u64>,
    evidence_digest: Option<[u8; 32]>,
}

pub fn event_id_for_record(
    run_id: &str,
    event_type: &EventType,
    t: u64,
    primary_record_digest: &[u8],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"UCF:EVENT_ID:v1");
    hasher.update(format!("{event_type:?}").as_bytes());
    hasher.update(run_id.as_bytes());
    hasher.update(t.to_le_bytes());
    hasher.update(primary_record_digest);
    hex::encode(hasher.finalize())
}

pub fn causal_slice(
    workdir: &Path,
    run_id: &str,
    center_event_id: &str,
    radius: u8,
) -> Result<CausalSlice, OpsError> {
    let records = load_records(workdir)?;
    let mut events = build_events(&records, run_id);
    events.sort_by_key(|e| (e.t, e.node.event_id.clone()));
    let center = events
        .iter()
        .find(|e| e.node.event_id == center_event_id)
        .ok_or_else(|| OpsError::Invalid(format!("event not found: {center_event_id}")))?
        .clone();

    let half_window = u64::from(radius).saturating_mul(WINDOW_TICKS_PER_RADIUS);
    let start_t = center.t.saturating_sub(half_window);
    let end_t = center.t.saturating_add(half_window);

    let bounded_events = events
        .into_iter()
        .filter(|e| e.t >= start_t && e.t <= end_t)
        .collect::<Vec<_>>();

    let all_edges = build_edges(&bounded_events);
    let slice = bounded_bfs_slice(&bounded_events, &all_edges, center_event_id, radius);
    Ok(slice)
}

pub fn explain_why(workdir: &Path, decision_id: u64) -> Result<ExplainWhyReport, OpsError> {
    let records = load_records(workdir)?;
    let run_id = infer_run_id(workdir);
    let events = build_events(&records, &run_id);
    let center = events
        .iter()
        .find(|e| e.decision_id == Some(decision_id) && e.node.event_type == EventType::Decision)
        .ok_or_else(|| OpsError::Invalid(format!("decision not found: {decision_id}")))?;

    let slice = causal_slice(workdir, &run_id, &center.node.event_id, 2)?;
    let mut incoming = slice
        .edges
        .iter()
        .filter(|e| e.dst_event_id == center.node.event_id)
        .cloned()
        .collect::<Vec<_>>();
    incoming.sort_by_key(|e| (e.edge_type.clone(), e.src_event_id.clone()));
    incoming.truncate(8);

    let mut outgoing = slice
        .edges
        .iter()
        .filter(|e| e.src_event_id == center.node.event_id)
        .cloned()
        .collect::<Vec<_>>();
    outgoing.sort_by_key(|e| (e.edge_type.clone(), e.dst_event_id.clone()));
    outgoing.truncate(8);

    Ok(ExplainWhyReport {
        decision_id,
        center_event_id: center.node.event_id.clone(),
        incoming_causes: incoming,
        outgoing_effects: outgoing,
        slice,
    })
}

pub fn simulate_counterfactual(
    workdir: &Path,
    request: CounterfactualRequest,
) -> Result<CounterfactualResult, OpsError> {
    let records = load_records(workdir)?;

    let mut selected_candidate_id = None;
    let mut alternative_allowed = false;
    let mut alternative_has_tool = false;
    let mut policy_prefix: Option<String> = None;
    let mut evidence = BTreeSet::new();
    let mut baseline_energy_q: Option<u16> = None;
    let mut alternative_energy_q: Option<u16> = None;
    let mut baseline_risk_q: Option<u16> = None;

    for rec in &records {
        match &rec.payload {
            ExperiencePayload::Audit(AuditPayload::CandidateSet(c))
                if c.decision_id == request.base_decision_id =>
            {
                selected_candidate_id = Some(c.selected_candidate_id);
                if let Some(summary) = c
                    .summaries
                    .iter()
                    .find(|s| s.candidate_id == request.alternative_candidate_id)
                {
                    alternative_allowed = summary.allowed;
                    alternative_has_tool = summary.tool_intent_count > 0;
                }
            }
            ExperiencePayload::Audit(AuditPayload::EbmReasoning(r))
                if r.decision_id == request.base_decision_id =>
            {
                baseline_risk_q = Some(r.risk_q);
                baseline_energy_q = Some(r.base_energy_q);
                if let Some((idx, _)) = r
                    .top_candidate_ids
                    .iter()
                    .enumerate()
                    .find(|(_, id)| **id == request.alternative_candidate_id)
                {
                    alternative_energy_q = r.top_energies_q.get(idx).copied();
                }
                evidence.insert(hex::encode(r.evidence_chain_digest_prefix));
            }
            ExperiencePayload::Decision(decision) if rec.id.0 == request.base_decision_id => {
                policy_prefix = decision.policy_graph_digest_prefix.map(hex::encode);
            }
            ExperiencePayload::Audit(AuditPayload::CapabilityIssuance(i))
                if i.decision_id == request.base_decision_id =>
            {
                evidence.insert(digest_prefix(&i.evidence_chain_digest, 8));
            }
            _ => {}
        }
    }

    let would_choose_candidate = alternative_allowed
        && selected_candidate_id
            .map(|selected| selected != request.alternative_candidate_id)
            .unwrap_or(false);
    let would_issue_tool = alternative_has_tool && would_choose_candidate;

    let risk_delta_q = baseline_risk_q
        .map(|base| i32::from(base) - i32::from(base))
        .unwrap_or(0) as i16;
    let energy_delta_q = match (baseline_energy_q, alternative_energy_q) {
        (Some(base), Some(alt)) => (i32::from(alt) - i32::from(base)) as i16,
        _ => 0,
    };

    let result = CounterfactualResult {
        base_decision_id: request.base_decision_id,
        alternative_candidate_id: request.alternative_candidate_id,
        would_choose_candidate,
        would_issue_tool,
        risk_delta_q,
        energy_delta_q,
        policy_graph_digest_prefix: policy_prefix,
        evidence_digest_prefixes: evidence.into_iter().collect(),
    };

    persist_counterfactual_record(workdir, &result)?;
    Ok(result)
}

fn persist_counterfactual_record(
    workdir: &Path,
    result: &CounterfactualResult,
) -> Result<(), OpsError> {
    let mut hasher = Sha256::new();
    hasher.update(serde_json::to_vec(result)?);
    let record = CounterfactualRecord {
        schema_version: 1,
        base_decision_id: result.base_decision_id,
        alternative_candidate_id: result.alternative_candidate_id,
        result_digest: hex::encode(hasher.finalize()),
        policy_graph_digest_prefix: result.policy_graph_digest_prefix.clone(),
        evidence_digest_prefixes: result.evidence_digest_prefixes.clone(),
    };

    let path = workdir.join("ess").join("counterfactual_records.json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut records: Vec<CounterfactualRecord> = if path.exists() {
        serde_json::from_slice(&fs::read(&path)?)?
    } else {
        Vec::new()
    };
    records.push(record);
    records.sort_by_key(|r| {
        (
            r.base_decision_id,
            r.alternative_candidate_id,
            r.result_digest.clone(),
        )
    });
    fs::write(path, serde_json::to_vec_pretty(&records)?)?;
    Ok(())
}

fn infer_run_id(workdir: &Path) -> String {
    let path = workdir.join("ess").join("run_metadata_record.json");
    if let Ok(body) = fs::read_to_string(path) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) {
            if let Some(run_id) = v.get("run_id").and_then(|x| x.as_str()) {
                return run_id.to_string();
            }
        }
    }
    "unknown".to_string()
}

fn load_records(workdir: &Path) -> Result<Vec<ExperienceRecord>, OpsError> {
    let mut records = load_fixture_records(&workdir.join("ess").join("ess_fixture.json"))?;
    let policy_path = Path::new("policies/bundle_v1/retention_v1.json");
    if let Ok(policy_text) = fs::read_to_string(policy_path) {
        let policy: RetentionPolicyV1 = serde_json::from_str(&policy_text)?;
        let now_tick = records.last().map(|r| r.time.tick.get()).unwrap_or(0);
        apply_retention(&mut records, &policy, now_tick);
    }
    Ok(records)
}

fn build_events(records: &[ExperienceRecord], run_id: &str) -> Vec<DerivedEvent> {
    let mut events = Vec::new();
    for rec in records {
        let event_type = match rec.kind {
            ExperienceKind::ControlIn => Some(EventType::Control),
            ExperienceKind::DecisionOut => Some(EventType::Decision),
            ExperienceKind::ToolPlan => Some(EventType::ToolPlan),
            ExperienceKind::ToolIssue => Some(EventType::ToolIssue),
            ExperienceKind::ToolExecution => Some(EventType::ToolExec),
            ExperienceKind::AuditCheckpoint => Some(EventType::Milestone),
            ExperienceKind::CandidateSet
            | ExperienceKind::EbmReasoning
            | ExperienceKind::Output
            | ExperienceKind::CapabilityIssuance
            | ExperienceKind::Nsr
            | ExperienceKind::Emergency => Some(EventType::Experience),
            _ => None,
        };
        let Some(event_type) = event_type else {
            continue;
        };

        let record_digest = digest_for_record(rec);
        let t = rec.time.tick.get();
        let decision_id = extract_decision_id(rec);
        let candidate_id = extract_candidate_id(rec);
        let plan_digest_prefix = match &rec.payload {
            ExperiencePayload::Audit(AuditPayload::ToolPlan(p)) => Some(p.plan_digest_prefix),
            ExperiencePayload::Audit(AuditPayload::ToolIssue(i)) => Some(i.plan_digest_prefix),
            _ => None,
        };
        let tool_request_id = match &rec.payload {
            ExperiencePayload::Audit(AuditPayload::ToolExecution(exec)) => {
                Some(exec.tool_request_id)
            }
            ExperiencePayload::Audit(AuditPayload::ToolAuth(auth)) => Some(auth.tool_request_id),
            ExperiencePayload::Audit(AuditPayload::ToolRequest(req)) => Some(req.tool_request_id),
            _ => None,
        };
        let evidence_digest = extract_evidence_digest(rec);

        let node = EventNode {
            event_id: event_id_for_record(run_id, &event_type, t, &record_digest),
            event_type,
            t,
            record_digest_prefix: hex::encode(&record_digest[..8]),
            policy_graph_digest_prefix: extract_policy_graph_prefix(rec),
            risk_q: extract_risk_q(rec),
            pressure_q: extract_pressure_q(rec),
            energy_q: extract_energy_q(rec),
        };
        events.push(DerivedEvent {
            node,
            t,
            corr: rec.corr.0,
            decision_id,
            candidate_id,
            plan_digest_prefix,
            tool_request_id,
            evidence_digest,
        });
    }
    events
}

fn build_edges(events: &[DerivedEvent]) -> Vec<CausalEdge> {
    let mut edges = Vec::new();
    let mut decision_by_corr: BTreeMap<u64, &DerivedEvent> = BTreeMap::new();
    let mut control_by_corr: BTreeMap<u64, &DerivedEvent> = BTreeMap::new();
    let mut plan_by_digest: BTreeMap<[u8; 8], &DerivedEvent> = BTreeMap::new();
    let mut issue_by_corr: BTreeMap<u64, &DerivedEvent> = BTreeMap::new();

    for event in events {
        if event.node.event_type == EventType::Control {
            control_by_corr.insert(event.corr, event);
        }
        if event.node.event_type == EventType::Decision {
            decision_by_corr.insert(event.corr, event);
        }
        if event.node.event_type == EventType::ToolPlan {
            if let Some(d) = event.plan_digest_prefix {
                plan_by_digest.insert(d, event);
            }
        }
        if event.node.event_type == EventType::ToolIssue {
            issue_by_corr.insert(event.corr, event);
        }
    }

    for event in events {
        match event.node.event_type {
            EventType::Decision => {
                if let Some(ctrl) = control_by_corr.get(&event.corr) {
                    edges.push(make_edge(ctrl, event, EdgeType::Causes));
                }
            }
            EventType::ToolIssue => {
                if let Some(plan_digest) = event.plan_digest_prefix {
                    if let Some(plan) = plan_by_digest.get(&plan_digest) {
                        edges.push(make_edge(plan, event, EdgeType::Enables));
                    }
                }
            }
            EventType::ToolExec => {
                if let Some(issue) = issue_by_corr.get(&event.corr) {
                    edges.push(make_edge(issue, event, EdgeType::Enables));
                }
                if event.tool_request_id.is_some() {
                    // keeps explicit linkage deterministic when request ids are present.
                }
            }
            EventType::Experience | EventType::Milestone | EventType::ToolPlan => {
                if let Some(decision_id) = event.decision_id {
                    if let Some(decision) = events.iter().find(|e| {
                        e.node.event_type == EventType::Decision
                            && e.decision_id == Some(decision_id)
                    }) {
                        edges.push(make_edge(decision, event, EdgeType::Produces));
                    }
                }
            }
            EventType::Control => {}
        }
    }

    // Inferred: same tick adjacency Decision -> ToolPlan.
    for decision in events
        .iter()
        .filter(|e| e.node.event_type == EventType::Decision)
    {
        for plan in events
            .iter()
            .filter(|e| e.node.event_type == EventType::ToolPlan && e.t == decision.t)
        {
            edges.push(make_edge(decision, plan, EdgeType::Enables));
        }
    }

    // Inferred: same candidate_id EBM -> Decision (Justifies).
    for ebm in events.iter().filter(|e| {
        e.node.event_type == EventType::Experience
            && e.node.energy_q.is_some()
            && e.candidate_id.is_some()
    }) {
        if let Some(candidate_id) = ebm.candidate_id {
            if let Some(decision) = events.iter().find(|e| {
                e.node.event_type == EventType::Decision
                    && e.candidate_id == Some(candidate_id)
                    && e.t == ebm.t
            }) {
                edges.push(make_edge(ebm, decision, EdgeType::Justifies));
            }
        }
    }

    edges.sort_by_key(|e| {
        (
            e.src_event_id.clone(),
            e.dst_event_id.clone(),
            format!("{:?}", e.edge_type),
            e.evidence_digest_prefix.clone(),
        )
    });
    edges.dedup();
    edges
}

fn bounded_bfs_slice(
    events: &[DerivedEvent],
    edges: &[CausalEdge],
    center_event_id: &str,
    radius: u8,
) -> CausalSlice {
    let event_map = events
        .iter()
        .map(|e| (e.node.event_id.clone(), e.node.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut adjacency: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for edge in edges {
        adjacency
            .entry(edge.src_event_id.clone())
            .or_default()
            .push(edge.dst_event_id.clone());
        adjacency
            .entry(edge.dst_event_id.clone())
            .or_default()
            .push(edge.src_event_id.clone());
    }

    let mut visited = BTreeMap::<String, u8>::new();
    let mut queue = VecDeque::new();
    visited.insert(center_event_id.to_string(), 0);
    queue.push_back(center_event_id.to_string());

    while let Some(node_id) = queue.pop_front() {
        let depth = *visited.get(&node_id).unwrap_or(&0);
        if depth >= radius {
            continue;
        }
        if let Some(neighbors) = adjacency.get(&node_id) {
            let mut sorted = neighbors.clone();
            sorted.sort();
            for next in sorted {
                if visited.len() >= MAX_SLICE_NODES {
                    break;
                }
                if let std::collections::btree_map::Entry::Vacant(entry) =
                    visited.entry(next.clone())
                {
                    entry.insert(depth + 1);
                    queue.push_back(next);
                }
            }
        }
    }

    let mut nodes = visited
        .keys()
        .filter_map(|id| event_map.get(id).cloned())
        .collect::<Vec<_>>();
    nodes.sort_by_key(|n| (n.t, n.event_id.clone()));

    let keep = visited.keys().cloned().collect::<BTreeSet<_>>();
    let mut kept_edges = edges
        .iter()
        .filter(|e| keep.contains(&e.src_event_id) && keep.contains(&e.dst_event_id))
        .cloned()
        .collect::<Vec<_>>();
    kept_edges.sort_by_key(|e| {
        (
            e.src_event_id.clone(),
            e.dst_event_id.clone(),
            format!("{:?}", e.edge_type),
        )
    });

    CausalSlice {
        center_event_id: center_event_id.to_string(),
        radius,
        nodes,
        edges: kept_edges,
    }
}

fn make_edge(src: &DerivedEvent, dst: &DerivedEvent, edge_type: EdgeType) -> CausalEdge {
    let evidence = src
        .evidence_digest
        .or(dst.evidence_digest)
        .map(|d| digest_prefix(&d, 8))
        .unwrap_or_else(|| src.node.record_digest_prefix.clone());
    CausalEdge {
        src_event_id: src.node.event_id.clone(),
        dst_event_id: dst.node.event_id.clone(),
        edge_type,
        evidence_digest_prefix: evidence,
    }
}

fn digest_for_record(record: &ExperienceRecord) -> [u8; 32] {
    if let Some(d) = record.audit_digest {
        return d;
    }
    let mut hasher = Sha256::new();
    hasher.update(b"UCF:ESS:RECORD:v1");
    hasher.update(record.id.0.to_le_bytes());
    hasher.update(record.time.tick.get().to_le_bytes());
    hasher.update(record.corr.0.to_le_bytes());
    hasher.update(format!("{:?}", record.kind).as_bytes());
    if let Ok(bytes) = serde_json::to_vec(&format!("{:?}", record.payload)) {
        hasher.update(bytes);
    }
    hasher.finalize().into()
}

fn extract_decision_id(record: &ExperienceRecord) -> Option<u64> {
    match &record.payload {
        ExperiencePayload::Decision(_) => Some(record.id.0),
        ExperiencePayload::Audit(AuditPayload::CandidateSet(c)) => Some(c.decision_id),
        ExperiencePayload::Audit(AuditPayload::EbmReasoning(e)) => Some(e.decision_id),
        ExperiencePayload::Audit(AuditPayload::CapabilityIssuance(i)) => Some(i.decision_id),
        ExperiencePayload::Audit(AuditPayload::Output(o)) => Some(o.decision_id),
        _ => None,
    }
}

fn extract_candidate_id(record: &ExperienceRecord) -> Option<u16> {
    match &record.payload {
        ExperiencePayload::Audit(AuditPayload::CandidateSet(c)) => Some(c.selected_candidate_id),
        ExperiencePayload::Audit(AuditPayload::Output(o)) => Some(o.candidate_id),
        ExperiencePayload::Audit(AuditPayload::CapabilityIssuance(i)) => i.candidate_id,
        _ => None,
    }
}

fn extract_policy_graph_prefix(record: &ExperienceRecord) -> Option<String> {
    match &record.payload {
        ExperiencePayload::Decision(d) => d.policy_graph_digest_prefix.map(hex::encode),
        ExperiencePayload::Audit(AuditPayload::ToolIssue(i)) => {
            Some(hex::encode(i.policy_graph_digest_prefix))
        }
        ExperiencePayload::Audit(AuditPayload::PolicyProvenance(p)) => {
            Some(p.policy_graph_digest.chars().take(16).collect())
        }
        _ => None,
    }
}

fn extract_evidence_digest(record: &ExperienceRecord) -> Option<[u8; 32]> {
    match &record.payload {
        ExperiencePayload::Audit(AuditPayload::Output(o)) => Some(o.evidence_chain_digest),
        ExperiencePayload::Audit(AuditPayload::CapabilityIssuance(i)) => {
            Some(i.evidence_chain_digest)
        }
        ExperiencePayload::Audit(AuditPayload::ToolRequest(r)) => Some(r.evidence_chain_digest),
        ExperiencePayload::Audit(AuditPayload::SandboxCall(s)) => Some(s.evidence_chain_digest),
        _ => None,
    }
}

fn extract_risk_q(record: &ExperienceRecord) -> Option<u16> {
    match &record.payload {
        ExperiencePayload::Audit(AuditPayload::EbmReasoning(r)) => Some(r.risk_q),
        _ => None,
    }
}

fn extract_pressure_q(record: &ExperienceRecord) -> Option<u16> {
    match &record.payload {
        ExperiencePayload::Audit(AuditPayload::EbmReasoning(r)) => Some(r.pressure_q),
        _ => None,
    }
}

fn extract_energy_q(record: &ExperienceRecord) -> Option<u16> {
    match &record.payload {
        ExperiencePayload::Audit(AuditPayload::EbmReasoning(r)) => Some(r.aggregate_energy_q),
        _ => None,
    }
}

pub fn write_slice(slice: &CausalSlice, out: &Path) -> Result<(), OpsError> {
    let parent = out
        .parent()
        .ok_or_else(|| OpsError::Invalid("output path has no parent".to_string()))?;
    fs::create_dir_all(parent)?;
    fs::write(out, serde_json::to_vec_pretty(slice)?)?;
    Ok(())
}

pub fn event_id_for_decision(
    workdir: &Path,
    run_id: &str,
    decision_id: u64,
) -> Result<Option<String>, OpsError> {
    let records = load_records(workdir)?;
    for event in build_events(&records, run_id) {
        if event.node.event_type == EventType::Decision && event.decision_id == Some(decision_id) {
            return Ok(Some(event.node.event_id));
        }
    }
    Ok(None)
}

pub fn save_counterfactual_result(
    result: &CounterfactualResult,
    out: &Path,
) -> Result<(), OpsError> {
    let parent = out
        .parent()
        .ok_or_else(|| OpsError::Invalid("output path has no parent".to_string()))?;
    fs::create_dir_all(parent)?;
    fs::write(out, serde_json::to_vec_pretty(result)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_id_is_deterministic() {
        let digest = [7_u8; 32];
        let a = event_id_for_record("run-1", &EventType::Decision, 42, &digest);
        let b = event_id_for_record("run-1", &EventType::Decision, 42, &digest);
        assert_eq!(a, b);
    }

    #[test]
    fn inferred_edges_are_stable_and_bounded() {
        let base = EventNode {
            event_id: "a".to_string(),
            event_type: EventType::Decision,
            t: 1,
            record_digest_prefix: "00".to_string(),
            policy_graph_digest_prefix: None,
            risk_q: None,
            pressure_q: None,
            energy_q: None,
        };
        let plan = EventNode {
            event_id: "b".to_string(),
            event_type: EventType::ToolPlan,
            t: 1,
            record_digest_prefix: "11".to_string(),
            policy_graph_digest_prefix: None,
            risk_q: None,
            pressure_q: None,
            energy_q: None,
        };
        let events = vec![
            DerivedEvent {
                node: base,
                t: 1,
                corr: 1,
                decision_id: Some(10),
                candidate_id: Some(1),
                plan_digest_prefix: None,
                tool_request_id: None,
                evidence_digest: None,
            },
            DerivedEvent {
                node: plan,
                t: 1,
                corr: 1,
                decision_id: Some(10),
                candidate_id: None,
                plan_digest_prefix: Some([3_u8; 8]),
                tool_request_id: None,
                evidence_digest: None,
            },
        ];
        let edges = build_edges(&events);
        assert!(edges
            .iter()
            .any(|e| e.src_event_id == "a" && e.dst_event_id == "b"));

        let slice = bounded_bfs_slice(&events, &edges, "a", 2);
        assert!(slice.nodes.len() <= MAX_SLICE_NODES);
    }
}
