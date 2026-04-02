use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::models_lifecycle::{
    AggregatedActiveReviewSnapshotV1, BackendEvidenceSnapshotV1, BurnSupportResolutionV1,
};
use crate::operator_report::ConsolidatedOperatorReportV1;
use crate::operator_signoff::{OperatorSignoffDecisionV1, SignoffDecisionStateV1};
use crate::{
    derive_slot_reviewability_truths_from_active, load_applied_supported_set_context_v1,
    reduce_reviewability, validate_governance_primary_surfaces_from_workdir,
    AppliedSupportedSetContextV1, GovernancePrimarySurfacesV1, OpsError,
    ReviewabilityAggregateReadinessV1, V0GateOverallStatus, V0GateReportV1, V1GateOverallStatus,
    V1GateReportV1, V2GateOverallStatus, V2GateReportV1, V3GateOverallStatus, V3GateReportV1,
    V4GateOverallStatus, V4GateReportV1,
};

const CODE_CAP: usize = 12;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum OperatorReviewStageV1 {
    ReviewBlocked,
    ReviewShadowReady,
    ReviewActiveReady,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OperatorReviewPacketArtifactsV1 {
    pub backend_evidence_snapshot_digest_prefix: String,
    pub active_review_snapshot_digest_prefix: String,
    pub operator_signoff_digest_prefix: String,
    pub operator_report_digest_prefix: String,
    pub gate_digests: OperatorReviewPacketGateDigestsV1,
    pub backend_resolution_digest_prefix: Option<String>,
    pub applied_supported_set_context_digest_prefix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OperatorReviewPacketGateDigestsV1 {
    pub v0: String,
    pub v1: String,
    pub v2: String,
    pub v3: String,
    pub v4: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OperatorReviewPacketSlotV1 {
    pub slot_id: String,
    pub target_hash_prefix: String,
    pub probe_ready: bool,
    pub shadow_ready: bool,
    pub active_eligible: bool,
    pub primary_denial_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OperatorReviewPacketV1 {
    pub schema_version: u16,
    pub review_stage: OperatorReviewStageV1,
    pub supported_slot_set_digest: String,
    pub policy_graph_digest_prefix: String,
    pub manifest_digest_prefix: String,
    pub applied_supported_set_digest_prefix: String,
    pub applied_context_digest_prefix: String,
    pub reviewability_reduction_digest_prefix: String,
    #[serde(default)]
    pub canonical_readiness_spine_digest_prefix: String,
    #[serde(default)]
    pub canonical_readiness_authority_digest_prefix: String,
    #[serde(default)]
    pub canonical_governance_entry_digest_prefix: String,
    #[serde(default)]
    pub final_governance_consumer_authority_digest_prefix: String,
    #[serde(default)]
    pub governance_residual_sweep_digest_prefix: String,
    #[serde(default)]
    pub residual_free_governance_authority_digest_prefix: String,
    #[serde(default)]
    pub governance_absolute_sweep_digest_prefix: String,
    #[serde(default)]
    pub absolute_final_governance_terminal_sweep_digest_prefix: String,
    #[serde(default)]
    pub governance_ultimate_sweep_digest_prefix: String,
    #[serde(default)]
    pub final_readiness_consumer_authority_digest_prefix: String,
    #[serde(default)]
    pub readiness_residual_sweep_digest_prefix: String,
    #[serde(default)]
    pub residual_free_readiness_authority_digest_prefix: String,
    #[serde(default)]
    pub readiness_absolute_sweep_digest_prefix: String,
    #[serde(default)]
    pub readiness_terminal_sweep_digest_prefix: String,
    #[serde(default)]
    pub readiness_ultimate_sweep_digest_prefix: String,
    #[serde(default)]
    pub readiness_stabilization_sweep_digest_prefix: String,
    #[serde(default)]
    pub readiness_final_consolidation_sweep_digest_prefix: String,
    #[serde(default)]
    pub governance_final_consolidation_sweep_digest_prefix: String,
    #[serde(default)]
    pub governance_closure_sweep_digest_prefix: String,
    #[serde(default)]
    pub final_primary_semantics_residual_sweep_digest_prefix: String,
    #[serde(default)]
    pub residual_free_primary_semantics_authority_digest_prefix: String,
    #[serde(default)]
    pub primary_semantics_absolute_sweep_digest_prefix: String,
    #[serde(default)]
    pub primary_semantics_terminal_sweep_digest_prefix: String,
    #[serde(default)]
    pub primary_semantics_ultimate_sweep_digest_prefix: String,
    #[serde(default)]
    pub primary_semantics_convergence_sweep_digest_prefix: String,
    pub artifacts: OperatorReviewPacketArtifactsV1,
    pub supported_slots: Vec<OperatorReviewPacketSlotV1>,
    pub blocking_codes: Vec<String>,
    pub remediation_codes: Vec<String>,
    pub packet_digest: String,
}

#[derive(Debug, Clone)]
pub struct OperatorReviewPacketArgs {
    pub run_id: Option<String>,
    pub latest: bool,
}

pub fn operator_review_packet(
    workdir: &Path,
    args: &OperatorReviewPacketArgs,
    out: &Path,
) -> Result<OperatorReviewPacketV1, OpsError> {
    let out_root = PathBuf::from("./out");

    let backend_snapshot = maybe_read_json::<BackendEvidenceSnapshotV1>(&discover_report(
        &out_root,
        "backend_evidence_snapshot.json",
        args,
    ));
    let active_review = maybe_read_json::<AggregatedActiveReviewSnapshotV1>(&discover_report(
        &out_root,
        "active_review_snapshot.json",
        args,
    ));
    let signoff = maybe_read_json::<OperatorSignoffDecisionV1>(&discover_report(
        &out_root,
        "operator_signoff.json",
        args,
    ));
    let operator_report = maybe_read_json::<ConsolidatedOperatorReportV1>(&discover_report(
        &out_root,
        "operator_report.json",
        args,
    ));
    let gate_v0 =
        maybe_read_json::<V0GateReportV1>(&discover_report(&out_root, "v0_gate_report.json", args));
    let gate_v1 =
        maybe_read_json::<V1GateReportV1>(&discover_report(&out_root, "v1_gate_report.json", args));
    let gate_v2 =
        maybe_read_json::<V2GateReportV1>(&discover_report(&out_root, "v2_gate_report.json", args));
    let gate_v3 =
        maybe_read_json::<V3GateReportV1>(&discover_report(&out_root, "v3_gate_report.json", args));
    let gate_v4 =
        maybe_read_json::<V4GateReportV1>(&discover_report(&out_root, "v4_gate_report.json", args));

    let backend_resolution = args.run_id.as_ref().and_then(|run_id| {
        let run_dir = out_root.join(run_id);
        fs::read_dir(&run_dir).ok().and_then(|entries| {
            let mut files = entries
                .filter_map(|entry| {
                    let path = entry.ok()?.path();
                    let name = path.file_name()?.to_str()?;
                    if name.starts_with("backend_resolution_") && name.ends_with(".json") {
                        Some(path)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            files.sort();
            files
                .into_iter()
                .find_map(|path| read_json::<BurnSupportResolutionV1>(&path).ok())
        })
    });

    let applied_scope = load_applied_supported_set_context_v1(workdir)?;
    let governance_surfaces = match (backend_snapshot.as_ref(), active_review.as_ref()) {
        (Some(backend), Some(active)) => {
            validate_governance_primary_surfaces_from_workdir(workdir, backend, active).ok()
        }
        _ => None,
    };

    let packet = reduce_review_packet(
        backend_snapshot,
        active_review,
        governance_surfaces,
        signoff,
        operator_report,
        gate_v0,
        gate_v1,
        gate_v2,
        gate_v3,
        gate_v4,
        backend_resolution,
        applied_scope,
    )?;

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_string_pretty(&packet)?)?;
    let _ = workdir;
    Ok(packet)
}

pub fn operator_review_packet_text(packet: &OperatorReviewPacketV1) -> String {
    format!(
        "review_stage={:?}\nsupported_slots={}\nblocking_codes={}\nremediation_codes={}",
        packet.review_stage,
        packet
            .supported_slots
            .iter()
            .map(|slot| format!(
                "{}:probe={},shadow={},active={}",
                slot.slot_id, slot.probe_ready, slot.shadow_ready, slot.active_eligible
            ))
            .collect::<Vec<_>>()
            .join(","),
        if packet.blocking_codes.is_empty() {
            "none".to_string()
        } else {
            packet.blocking_codes.join(",")
        },
        if packet.remediation_codes.is_empty() {
            "none".to_string()
        } else {
            packet.remediation_codes.join(",")
        }
    )
}

#[allow(clippy::too_many_arguments)]
fn reduce_review_packet(
    backend_snapshot: Option<BackendEvidenceSnapshotV1>,
    active_review: Option<AggregatedActiveReviewSnapshotV1>,
    governance_surfaces: Option<GovernancePrimarySurfacesV1>,
    signoff: Option<OperatorSignoffDecisionV1>,
    operator_report: Option<ConsolidatedOperatorReportV1>,
    gate_v0: Option<V0GateReportV1>,
    gate_v1: Option<V1GateReportV1>,
    gate_v2: Option<V2GateReportV1>,
    gate_v3: Option<V3GateReportV1>,
    gate_v4: Option<V4GateReportV1>,
    backend_resolution: Option<BurnSupportResolutionV1>,
    applied_scope: AppliedSupportedSetContextV1,
) -> Result<OperatorReviewPacketV1, OpsError> {
    let mut blocking = BTreeSet::new();
    let mut remediation = BTreeSet::new();

    let snapshot = match backend_snapshot {
        Some(snapshot) => snapshot,
        None => {
            blocking.insert("REVIEW_BLOCK_BACKEND_EVIDENCE_SNAPSHOT_MISSING".to_string());
            remediation.insert("run_backend_evidence_snapshot".to_string());
            return build_blocked_minimal(
                blocking,
                remediation,
                gate_v0,
                gate_v1,
                gate_v2,
                gate_v3,
                gate_v4,
                &applied_scope,
            );
        }
    };

    let active = match active_review {
        Some(active) => active,
        None => {
            blocking.insert("REVIEW_BLOCK_ACTIVE_REVIEW_SNAPSHOT_MISSING".to_string());
            remediation.insert("run_models_active_review_snapshot".to_string());
            return build_from_snapshot(
                snapshot,
                None,
                signoff,
                operator_report,
                gate_v0,
                gate_v1,
                gate_v2,
                gate_v3,
                gate_v4,
                backend_resolution,
                blocking,
                remediation,
                &applied_scope,
            );
        }
    };

    if governance_surfaces.is_none() {
        blocking.insert("REVIEW_BLOCK_GOVERNANCE_SURFACES_MISMATCH".to_string());
        remediation.insert("run_governance_surfaces_check".to_string());
    }

    let signoff = match signoff {
        Some(signoff) => signoff,
        None => {
            blocking.insert("REVIEW_BLOCK_OPERATOR_SIGNOFF_MISSING".to_string());
            remediation.insert("run_operator_signoff".to_string());
            return build_from_snapshot(
                snapshot,
                Some(active),
                None,
                operator_report,
                gate_v0,
                gate_v1,
                gate_v2,
                gate_v3,
                gate_v4,
                backend_resolution,
                blocking,
                remediation,
                &applied_scope,
            );
        }
    };

    let operator_report = match operator_report {
        Some(report) => report,
        None => {
            blocking.insert("REVIEW_BLOCK_OPERATOR_REPORT_MISSING".to_string());
            remediation.insert("run_operator_report".to_string());
            return build_from_snapshot(
                snapshot,
                Some(active),
                Some(signoff),
                None,
                gate_v0,
                gate_v1,
                gate_v2,
                gate_v3,
                gate_v4,
                backend_resolution,
                blocking,
                remediation,
                &applied_scope,
            );
        }
    };

    let snapshot_slots = snapshot
        .slots
        .iter()
        .map(|slot| slot.slot_id.clone())
        .collect::<Vec<_>>();
    if snapshot_slots != applied_scope.slots {
        blocking.insert("REVIEW_BLOCK_SCOPE_MISMATCH".to_string());
        remediation.insert("run_models_applied_scope_check".to_string());
    }

    check_gates(
        &gate_v0,
        &gate_v1,
        &gate_v2,
        &gate_v3,
        &gate_v4,
        &mut blocking,
        &mut remediation,
    );

    if signoff.supported_slot_set_digest != snapshot.supported_slot_set_digest {
        blocking.insert("REVIEW_BLOCK_DIGEST_SLOT_SET_MISMATCH".to_string());
        remediation.insert("rerun_operator_artifacts".to_string());
    }

    if signoff.policy_graph_digest_prefix != snapshot.policy_graph_digest_prefix
        || signoff.manifest_digest_prefix != snapshot.manifest_digest_prefix
    {
        blocking.insert("REVIEW_BLOCK_DIGEST_CONTEXT_MISMATCH".to_string());
        remediation.insert("rerun_operator_artifacts".to_string());
    }

    if signoff.evidence_snapshot_digest_prefix != prefix16(&snapshot.snapshot_digest)
        || signoff.operator_report_digest_prefix != prefix16(&operator_report.report_digest)
    {
        blocking.insert("REVIEW_BLOCK_DIGEST_ARTIFACT_MISMATCH".to_string());
        remediation.insert("rerun_operator_artifacts".to_string());
    }
    if signoff.canonical_readiness_spine_digest_prefix == "MISSING" {
        blocking.insert("CANONICAL_READINESS_SPINE_REQUIRED".to_string());
        remediation.insert("run_readiness_spine_sweep".to_string());
    }

    let (aggregate_readiness, shadow_ready, reviewability_reduction_digest_prefix) =
        match derive_slot_reviewability_truths_from_active(&applied_scope, &snapshot, &active)
            .and_then(|truths| {
                let shadow_ready = truths
                    .iter()
                    .all(|slot| slot.probe_ready && slot.shadow_ready);
                let reduction = reduce_reviewability(&applied_scope, &truths)?;
                Ok((
                    reduction.aggregate_readiness,
                    shadow_ready,
                    prefix16(&reduction.reduction_digest),
                ))
            }) {
            Ok(result) => result,
            Err(_) => {
                blocking.insert("LEGACY_REDUCTION_REJECTED".to_string());
                remediation.insert("run_models_applied_scope_check".to_string());
                (
                    ReviewabilityAggregateReadinessV1::NoneReviewable,
                    false,
                    "MISSING".to_string(),
                )
            }
        };
    let stage = reduce_stage(&signoff, &blocking, &aggregate_readiness, shadow_ready);

    let mut packet = build_packet(
        &snapshot,
        &active,
        &signoff,
        &operator_report,
        gate_v0.as_ref(),
        gate_v1.as_ref(),
        gate_v2.as_ref(),
        gate_v3.as_ref(),
        gate_v4.as_ref(),
        backend_resolution.as_ref(),
        stage,
        blocking,
        remediation,
        &applied_scope,
        &reviewability_reduction_digest_prefix,
    )?;
    packet.final_governance_consumer_authority_digest_prefix =
        read_final_governance_prefix(Path::new("."), "out/final_governance_consumer_sweep.json");
    packet.governance_residual_sweep_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/governance_residual_sweep.json",
        "sweep_digest",
    );
    packet.residual_free_governance_authority_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/residual_free_governance_sweep.json",
        "authority_digest",
    );
    packet.governance_absolute_sweep_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/governance_absolute_sweep.json",
        "sweep.sweep_digest",
    );
    packet.absolute_final_governance_terminal_sweep_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/governance_terminal_sweep.json",
        "sweep.sweep_digest",
    );
    packet.governance_ultimate_sweep_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/governance_ultimate_sweep.json",
        "sweep.sweep_digest",
    );
    packet.final_readiness_consumer_authority_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/final_readiness_consumer_sweep.json",
        "authority_digest",
    );
    packet.readiness_residual_sweep_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/readiness_residual_sweep.json",
        "sweep_digest",
    );
    packet.residual_free_readiness_authority_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/residual_free_readiness_sweep.json",
        "authority_digest",
    );
    packet.readiness_absolute_sweep_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/readiness_absolute_sweep.json",
        "sweep.sweep_digest",
    );
    packet.readiness_terminal_sweep_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/readiness_terminal_sweep.json",
        "sweep.sweep_digest",
    );
    packet.readiness_ultimate_sweep_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/readiness_ultimate_sweep.json",
        "sweep.sweep_digest",
    );
    packet.readiness_final_consolidation_sweep_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/readiness_final_consolidation_sweep.json",
        "sweep.consolidation_digest",
    );
    packet.final_primary_semantics_residual_sweep_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/primary_semantics_residual_sweep.json",
        "sweep_digest",
    );
    packet.residual_free_primary_semantics_authority_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/residual_free_primary_semantics_sweep.json",
        "authority.authority_digest",
    );
    packet.primary_semantics_absolute_sweep_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/primary_semantics_absolute_sweep.json",
        "sweep.sweep_digest",
    );
    packet.primary_semantics_terminal_sweep_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/primary_semantics_terminal_sweep.json",
        "sweep.sweep_digest",
    );
    packet.primary_semantics_ultimate_sweep_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/primary_semantics_ultimate_sweep.json",
        "sweep.sweep_digest",
    );
    packet.primary_semantics_convergence_sweep_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/primary_semantics_convergence_sweep.json",
        "sweep.convergence_digest",
    );
    packet.packet_digest = packet_digest(&packet)?;
    Ok(packet)
}

fn reduce_stage(
    signoff: &OperatorSignoffDecisionV1,
    blocking: &BTreeSet<String>,
    aggregate_readiness: &ReviewabilityAggregateReadinessV1,
    any_shadow_ready: bool,
) -> OperatorReviewStageV1 {
    if !blocking.is_empty() {
        return OperatorReviewStageV1::ReviewBlocked;
    }

    if signoff.decision == SignoffDecisionStateV1::ReadyForActiveReview
        && !matches!(
            aggregate_readiness,
            ReviewabilityAggregateReadinessV1::NoneReviewable
        )
    {
        return OperatorReviewStageV1::ReviewActiveReady;
    }

    if signoff.decision == SignoffDecisionStateV1::ReadyForShadow || any_shadow_ready {
        return OperatorReviewStageV1::ReviewShadowReady;
    }

    OperatorReviewStageV1::ReviewBlocked
}

#[allow(clippy::too_many_arguments)]
fn build_from_snapshot(
    snapshot: BackendEvidenceSnapshotV1,
    active: Option<AggregatedActiveReviewSnapshotV1>,
    signoff: Option<OperatorSignoffDecisionV1>,
    operator_report: Option<ConsolidatedOperatorReportV1>,
    gate_v0: Option<V0GateReportV1>,
    gate_v1: Option<V1GateReportV1>,
    gate_v2: Option<V2GateReportV1>,
    gate_v3: Option<V3GateReportV1>,
    gate_v4: Option<V4GateReportV1>,
    backend_resolution: Option<BurnSupportResolutionV1>,
    blocking: BTreeSet<String>,
    remediation: BTreeSet<String>,
    applied_scope: &AppliedSupportedSetContextV1,
) -> Result<OperatorReviewPacketV1, OpsError> {
    let active = active.unwrap_or_else(|| AggregatedActiveReviewSnapshotV1 {
        schema_version: 1,
        supported_slot_set_digest: snapshot.supported_slot_set_digest.clone(),
        policy_graph_digest_prefix: snapshot.policy_graph_digest_prefix.clone(),
        manifest_digest_prefix: snapshot.manifest_digest_prefix.clone(),
        slots: Vec::new(),
        overall_review_status: crate::ActiveReviewOverallStatusV1::NoneReviewable,
        signoff_alignment: crate::models_lifecycle::ActiveReviewSignoffAlignmentV1 {
            aligned: false,
            status_code: "MISSING".to_string(),
        },
        canonical_governance_entry_digest_prefix: "MISSING".to_string(),
        final_governance_consumer_authority_digest_prefix: "MISSING".to_string(),
        governance_residual_sweep_digest_prefix: "MISSING".to_string(),
        residual_free_governance_authority_digest_prefix: "MISSING".to_string(),
        governance_absolute_sweep_digest_prefix: "MISSING".to_string(),
        absolute_final_governance_terminal_sweep_digest_prefix: "MISSING".to_string(),
        governance_ultimate_sweep_digest_prefix: "MISSING".to_string(),
        final_readiness_consumer_authority_digest_prefix: "MISSING".to_string(),
        readiness_residual_sweep_digest_prefix: "MISSING".to_string(),
        residual_free_readiness_authority_digest_prefix: "MISSING".to_string(),
        readiness_absolute_sweep_digest_prefix: "MISSING".to_string(),
        readiness_terminal_sweep_digest_prefix: "MISSING".to_string(),
        readiness_ultimate_sweep_digest_prefix: "MISSING".to_string(),
        readiness_stabilization_sweep_digest_prefix: "MISSING".to_string(),
        snapshot_digest: "MISSING".to_string(),
    });
    let signoff = signoff.unwrap_or_else(|| OperatorSignoffDecisionV1 {
        schema_version: 1,
        decision: SignoffDecisionStateV1::NotReady,
        supported_slot_set_digest: snapshot.supported_slot_set_digest.clone(),
        policy_graph_digest_prefix: snapshot.policy_graph_digest_prefix.clone(),
        manifest_digest_prefix: snapshot.manifest_digest_prefix.clone(),
        evidence_snapshot_digest_prefix: "MISSING".to_string(),
        active_review_snapshot_digest_prefix: None,
        operator_report_digest_prefix: "MISSING".to_string(),
        applied_supported_set_digest_prefix: applied_scope.applied_set_digest_prefix.clone(),
        applied_context_digest_prefix: crate::prefix_hex(&applied_scope.context_digest, 16),
        reviewability_reduction_digest_prefix: "MISSING".to_string(),
        canonical_readiness_spine_digest_prefix: "MISSING".to_string(),
        canonical_readiness_authority_digest_prefix: "MISSING".to_string(),
        canonical_governance_entry_digest_prefix: "MISSING".to_string(),
        final_governance_consumer_authority_digest_prefix: "MISSING".to_string(),
        governance_residual_sweep_digest_prefix: "MISSING".to_string(),
        residual_free_governance_authority_digest_prefix: "MISSING".to_string(),
        governance_absolute_sweep_digest_prefix: "MISSING".to_string(),
        absolute_final_governance_terminal_sweep_digest_prefix: "MISSING".to_string(),
        governance_ultimate_sweep_digest_prefix: "MISSING".to_string(),
        final_readiness_consumer_authority_digest_prefix: "MISSING".to_string(),
        readiness_residual_sweep_digest_prefix: "MISSING".to_string(),
        residual_free_readiness_authority_digest_prefix: "MISSING".to_string(),
        readiness_absolute_sweep_digest_prefix: "MISSING".to_string(),
        readiness_terminal_sweep_digest_prefix: "MISSING".to_string(),
        readiness_ultimate_sweep_digest_prefix: "MISSING".to_string(),
        readiness_stabilization_sweep_digest_prefix: "MISSING".to_string(),
        readiness_final_consolidation_sweep_digest_prefix: "MISSING".to_string(),
        governance_final_consolidation_sweep_digest_prefix: "MISSING".to_string(),
        governance_closure_sweep_digest_prefix: "MISSING".to_string(),
        final_primary_semantics_residual_sweep_digest_prefix: "MISSING".to_string(),
        residual_free_primary_semantics_authority_digest_prefix: "MISSING".to_string(),
        primary_semantics_absolute_sweep_digest_prefix: "MISSING".to_string(),
        primary_semantics_terminal_sweep_digest_prefix: "MISSING".to_string(),
        primary_semantics_ultimate_sweep_digest_prefix: "MISSING".to_string(),
        primary_semantics_convergence_sweep_digest_prefix: "MISSING".to_string(),
        gate_report_digests: crate::operator_signoff::GateReportDigestsV1 {
            v0: "MISSING".to_string(),
            v1: "MISSING".to_string(),
            v2: "MISSING".to_string(),
            v3: "MISSING".to_string(),
        },
        reasons: Vec::new(),
        remediation_codes: Vec::new(),
        canonical_remediation_codes: Vec::new(),
        decision_digest: "MISSING".to_string(),
    });
    let operator_report = operator_report.unwrap_or_else(|| ConsolidatedOperatorReportV1 {
        schema_version: 1,
        generated_at: 0,
        overall_status: crate::operator_report::OperatorStatus::Missing,
        run_id: None,
        policy_graph_digest_prefix: None,
        manifest_digest_prefix: None,
        sections: crate::operator_report::OperatorSectionsV1 {
            health_section: crate::operator_report::NormalizedHealthSection {
                status: crate::operator_report::OperatorStatus::Missing,
                strict_mode_enabled: None,
                last_tick_age_ms: None,
                emergency_active: None,
                evidence_digest_prefixes: Vec::new(),
                remediation_codes: Vec::new(),
            },
            eligibility_section: crate::operator_report::NormalizedEligibilitySection {
                status: crate::operator_report::OperatorStatus::Missing,
                slots: Vec::new(),
                evidence_digest_prefixes: Vec::new(),
                remediation_codes: Vec::new(),
            },
            drift_section: crate::operator_report::NormalizedDriftSection {
                status: crate::operator_report::OperatorStatus::Missing,
                slots: Vec::new(),
                evidence_digest_prefixes: Vec::new(),
                remediation_codes: Vec::new(),
            },
            alerts_section: crate::operator_report::NormalizedAlertsSection {
                status: crate::operator_report::OperatorStatus::Missing,
                active_alert_count: 0,
                top_active_alerts: Vec::new(),
                evidence_digest_prefixes: Vec::new(),
                remediation_codes: Vec::new(),
            },
            strict_section: crate::operator_report::NormalizedStrictSection {
                status: crate::operator_report::OperatorStatus::Missing,
                strict_status: crate::StrictEvidenceStatusV1::Missing,
                primary_denial_code: None,
                strict_report_digest_prefix: None,
                failing_check_ids: Vec::new(),
                evidence_digest_prefixes: Vec::new(),
                remediation_codes: Vec::new(),
            },
            gates_section: crate::operator_report::NormalizedGatesSection {
                status: crate::operator_report::OperatorStatus::Missing,
                gates: Vec::new(),
                evidence_digest_prefixes: Vec::new(),
                remediation_codes: Vec::new(),
            },
        },
        remediation_codes: Vec::new(),
        canonical_remediation_codes: Vec::new(),
        report_digest: "MISSING".to_string(),
    });

    let mut packet = build_packet(
        &snapshot,
        &active,
        &signoff,
        &operator_report,
        gate_v0.as_ref(),
        gate_v1.as_ref(),
        gate_v2.as_ref(),
        gate_v3.as_ref(),
        gate_v4.as_ref(),
        backend_resolution.as_ref(),
        OperatorReviewStageV1::ReviewBlocked,
        blocking,
        remediation,
        applied_scope,
        "MISSING",
    )?;
    packet.final_governance_consumer_authority_digest_prefix =
        read_final_governance_prefix(Path::new("."), "out/final_governance_consumer_sweep.json");
    packet.governance_residual_sweep_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/governance_residual_sweep.json",
        "sweep_digest",
    );
    packet.residual_free_governance_authority_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/residual_free_governance_sweep.json",
        "authority_digest",
    );
    packet.governance_absolute_sweep_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/governance_absolute_sweep.json",
        "sweep.sweep_digest",
    );
    packet.absolute_final_governance_terminal_sweep_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/governance_terminal_sweep.json",
        "sweep.sweep_digest",
    );
    packet.governance_ultimate_sweep_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/governance_ultimate_sweep.json",
        "sweep.sweep_digest",
    );
    packet.final_readiness_consumer_authority_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/final_readiness_consumer_sweep.json",
        "authority_digest",
    );
    packet.readiness_residual_sweep_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/readiness_residual_sweep.json",
        "sweep_digest",
    );
    packet.residual_free_readiness_authority_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/residual_free_readiness_sweep.json",
        "authority_digest",
    );
    packet.readiness_absolute_sweep_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/readiness_absolute_sweep.json",
        "sweep.sweep_digest",
    );
    packet.readiness_terminal_sweep_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/readiness_terminal_sweep.json",
        "sweep.sweep_digest",
    );
    packet.readiness_ultimate_sweep_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/readiness_ultimate_sweep.json",
        "sweep.sweep_digest",
    );
    packet.readiness_final_consolidation_sweep_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/readiness_final_consolidation_sweep.json",
        "sweep.consolidation_digest",
    );
    packet.final_primary_semantics_residual_sweep_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/primary_semantics_residual_sweep.json",
        "sweep_digest",
    );
    packet.residual_free_primary_semantics_authority_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/residual_free_primary_semantics_sweep.json",
        "authority.authority_digest",
    );
    packet.primary_semantics_absolute_sweep_digest_prefix = read_sweep_digest_prefix(
        Path::new("."),
        "out/primary_semantics_absolute_sweep.json",
        "sweep.sweep_digest",
    );
    packet.packet_digest = packet_digest(&packet)?;
    Ok(packet)
}

#[allow(clippy::too_many_arguments)]
fn build_blocked_minimal(
    blocking: BTreeSet<String>,
    remediation: BTreeSet<String>,
    gate_v0: Option<V0GateReportV1>,
    gate_v1: Option<V1GateReportV1>,
    gate_v2: Option<V2GateReportV1>,
    gate_v3: Option<V3GateReportV1>,
    gate_v4: Option<V4GateReportV1>,
    applied_scope: &AppliedSupportedSetContextV1,
) -> Result<OperatorReviewPacketV1, OpsError> {
    let mut packet = OperatorReviewPacketV1 {
        schema_version: 1,
        review_stage: OperatorReviewStageV1::ReviewBlocked,
        supported_slot_set_digest: "MISSING".to_string(),
        policy_graph_digest_prefix: "MISSING".to_string(),
        manifest_digest_prefix: "MISSING".to_string(),
        applied_supported_set_digest_prefix: applied_scope.applied_set_digest_prefix.clone(),
        applied_context_digest_prefix: crate::prefix_hex(&applied_scope.context_digest, 16),
        reviewability_reduction_digest_prefix: "MISSING".to_string(),
        canonical_readiness_spine_digest_prefix: "MISSING".to_string(),
        canonical_readiness_authority_digest_prefix: "MISSING".to_string(),
        canonical_governance_entry_digest_prefix: "MISSING".to_string(),
        final_governance_consumer_authority_digest_prefix: "MISSING".to_string(),
        governance_residual_sweep_digest_prefix: "MISSING".to_string(),
        residual_free_governance_authority_digest_prefix: "MISSING".to_string(),
        governance_absolute_sweep_digest_prefix: "MISSING".to_string(),
        absolute_final_governance_terminal_sweep_digest_prefix: "MISSING".to_string(),
        governance_ultimate_sweep_digest_prefix: "MISSING".to_string(),
        final_readiness_consumer_authority_digest_prefix: "MISSING".to_string(),
        readiness_residual_sweep_digest_prefix: "MISSING".to_string(),
        residual_free_readiness_authority_digest_prefix: "MISSING".to_string(),
        readiness_absolute_sweep_digest_prefix: "MISSING".to_string(),
        readiness_terminal_sweep_digest_prefix: "MISSING".to_string(),
        readiness_ultimate_sweep_digest_prefix: "MISSING".to_string(),
        readiness_stabilization_sweep_digest_prefix: "MISSING".to_string(),
        readiness_final_consolidation_sweep_digest_prefix: "MISSING".to_string(),
        governance_final_consolidation_sweep_digest_prefix: "MISSING".to_string(),
        governance_closure_sweep_digest_prefix: "MISSING".to_string(),
        final_primary_semantics_residual_sweep_digest_prefix: "MISSING".to_string(),
        residual_free_primary_semantics_authority_digest_prefix: "MISSING".to_string(),
        primary_semantics_absolute_sweep_digest_prefix: "MISSING".to_string(),
        primary_semantics_terminal_sweep_digest_prefix: "MISSING".to_string(),
        primary_semantics_ultimate_sweep_digest_prefix: "MISSING".to_string(),
        primary_semantics_convergence_sweep_digest_prefix: "MISSING".to_string(),
        artifacts: OperatorReviewPacketArtifactsV1 {
            backend_evidence_snapshot_digest_prefix: "MISSING".to_string(),
            active_review_snapshot_digest_prefix: "MISSING".to_string(),
            operator_signoff_digest_prefix: "MISSING".to_string(),
            operator_report_digest_prefix: "MISSING".to_string(),
            gate_digests: OperatorReviewPacketGateDigestsV1 {
                v0: digest_opt(gate_v0.as_ref())?,
                v1: digest_opt(gate_v1.as_ref())?,
                v2: digest_opt(gate_v2.as_ref())?,
                v3: digest_opt(gate_v3.as_ref())?,
                v4: digest_opt(gate_v4.as_ref())?,
            },
            backend_resolution_digest_prefix: None,
            applied_supported_set_context_digest_prefix: crate::prefix_hex(
                &applied_scope.context_digest,
                16,
            ),
        },
        supported_slots: Vec::new(),
        blocking_codes: bound_codes(blocking),
        remediation_codes: bound_codes(remediation),
        packet_digest: String::new(),
    };
    packet.packet_digest = packet_digest(&packet)?;
    Ok(packet)
}

#[allow(clippy::too_many_arguments)]
fn build_packet(
    snapshot: &BackendEvidenceSnapshotV1,
    active: &AggregatedActiveReviewSnapshotV1,
    signoff: &OperatorSignoffDecisionV1,
    operator_report: &ConsolidatedOperatorReportV1,
    gate_v0: Option<&V0GateReportV1>,
    gate_v1: Option<&V1GateReportV1>,
    gate_v2: Option<&V2GateReportV1>,
    gate_v3: Option<&V3GateReportV1>,
    gate_v4: Option<&V4GateReportV1>,
    backend_resolution: Option<&BurnSupportResolutionV1>,
    review_stage: OperatorReviewStageV1,
    blocking: BTreeSet<String>,
    remediation: BTreeSet<String>,
    applied_scope: &AppliedSupportedSetContextV1,
    reviewability_reduction_digest_prefix: &str,
) -> Result<OperatorReviewPacketV1, OpsError> {
    let mut supported_slots = snapshot
        .slots
        .iter()
        .map(|slot| {
            let active_slot = active
                .slots
                .iter()
                .find(|active_slot| active_slot.slot_id == slot.slot_id);
            OperatorReviewPacketSlotV1 {
                slot_id: slot.slot_id.clone(),
                target_hash_prefix: slot.target_hash_prefix.clone(),
                probe_ready: slot.readiness.probe_ready,
                shadow_ready: slot.readiness.shadow_ready,
                active_eligible: active_slot
                    .map(|entry| entry.active_eligible)
                    .unwrap_or(slot.readiness.active_eligible),
                primary_denial_code: active_slot
                    .and_then(|entry| entry.primary_denial_code.clone()),
            }
        })
        .collect::<Vec<_>>();
    supported_slots.sort_by(|a, b| a.slot_id.cmp(&b.slot_id));

    Ok(OperatorReviewPacketV1 {
        schema_version: 1,
        review_stage,
        supported_slot_set_digest: snapshot.supported_slot_set_digest.clone(),
        policy_graph_digest_prefix: snapshot.policy_graph_digest_prefix.clone(),
        manifest_digest_prefix: snapshot.manifest_digest_prefix.clone(),
        applied_supported_set_digest_prefix: applied_scope.applied_set_digest_prefix.clone(),
        applied_context_digest_prefix: crate::prefix_hex(&applied_scope.context_digest, 16),
        reviewability_reduction_digest_prefix: reviewability_reduction_digest_prefix.to_string(),
        canonical_readiness_spine_digest_prefix: signoff
            .canonical_readiness_spine_digest_prefix
            .clone(),
        canonical_readiness_authority_digest_prefix: signoff
            .canonical_readiness_authority_digest_prefix
            .clone(),
        canonical_governance_entry_digest_prefix: signoff
            .canonical_governance_entry_digest_prefix
            .clone(),
        final_governance_consumer_authority_digest_prefix: signoff
            .final_governance_consumer_authority_digest_prefix
            .clone(),
        governance_residual_sweep_digest_prefix: signoff
            .governance_residual_sweep_digest_prefix
            .clone(),
        residual_free_governance_authority_digest_prefix: signoff
            .residual_free_governance_authority_digest_prefix
            .clone(),
        governance_absolute_sweep_digest_prefix: signoff
            .governance_absolute_sweep_digest_prefix
            .clone(),
        absolute_final_governance_terminal_sweep_digest_prefix: signoff
            .absolute_final_governance_terminal_sweep_digest_prefix
            .clone(),
        governance_ultimate_sweep_digest_prefix: signoff
            .governance_ultimate_sweep_digest_prefix
            .clone(),
        final_readiness_consumer_authority_digest_prefix: signoff
            .final_readiness_consumer_authority_digest_prefix
            .clone(),
        readiness_residual_sweep_digest_prefix: signoff
            .readiness_residual_sweep_digest_prefix
            .clone(),
        residual_free_readiness_authority_digest_prefix: signoff
            .residual_free_readiness_authority_digest_prefix
            .clone(),
        readiness_absolute_sweep_digest_prefix: signoff
            .readiness_absolute_sweep_digest_prefix
            .clone(),
        readiness_terminal_sweep_digest_prefix: signoff
            .readiness_terminal_sweep_digest_prefix
            .clone(),
        readiness_ultimate_sweep_digest_prefix: signoff
            .readiness_ultimate_sweep_digest_prefix
            .clone(),
        readiness_stabilization_sweep_digest_prefix: signoff
            .readiness_stabilization_sweep_digest_prefix
            .clone(),
        readiness_final_consolidation_sweep_digest_prefix: signoff
            .readiness_final_consolidation_sweep_digest_prefix
            .clone(),
        governance_final_consolidation_sweep_digest_prefix: signoff
            .governance_final_consolidation_sweep_digest_prefix
            .clone(),
        governance_closure_sweep_digest_prefix: signoff
            .governance_closure_sweep_digest_prefix
            .clone(),
        final_primary_semantics_residual_sweep_digest_prefix: signoff
            .final_primary_semantics_residual_sweep_digest_prefix
            .clone(),
        residual_free_primary_semantics_authority_digest_prefix: signoff
            .residual_free_primary_semantics_authority_digest_prefix
            .clone(),
        primary_semantics_absolute_sweep_digest_prefix: signoff
            .primary_semantics_absolute_sweep_digest_prefix
            .clone(),
        primary_semantics_terminal_sweep_digest_prefix: signoff
            .primary_semantics_terminal_sweep_digest_prefix
            .clone(),
        primary_semantics_ultimate_sweep_digest_prefix: signoff
            .primary_semantics_ultimate_sweep_digest_prefix
            .clone(),
        primary_semantics_convergence_sweep_digest_prefix: signoff
            .primary_semantics_convergence_sweep_digest_prefix
            .clone(),
        artifacts: OperatorReviewPacketArtifactsV1 {
            backend_evidence_snapshot_digest_prefix: prefix16(&snapshot.snapshot_digest),
            active_review_snapshot_digest_prefix: prefix16(&active.snapshot_digest),
            operator_signoff_digest_prefix: prefix16(&signoff.decision_digest),
            operator_report_digest_prefix: prefix16(&operator_report.report_digest),
            gate_digests: OperatorReviewPacketGateDigestsV1 {
                v0: digest_opt(gate_v0)?,
                v1: digest_opt(gate_v1)?,
                v2: digest_opt(gate_v2)?,
                v3: digest_opt(gate_v3)?,
                v4: digest_opt(gate_v4)?,
            },
            backend_resolution_digest_prefix: backend_resolution
                .map(|resolution| prefix16(&resolution.evidence_digest)),
            applied_supported_set_context_digest_prefix: prefix16(&applied_scope.context_digest),
        },
        supported_slots,
        blocking_codes: bound_codes(blocking),
        remediation_codes: bound_codes(remediation),
        packet_digest: String::new(),
    })
}

fn check_gates(
    gate_v0: &Option<V0GateReportV1>,
    gate_v1: &Option<V1GateReportV1>,
    gate_v2: &Option<V2GateReportV1>,
    gate_v3: &Option<V3GateReportV1>,
    gate_v4: &Option<V4GateReportV1>,
    blocking: &mut BTreeSet<String>,
    remediation: &mut BTreeSet<String>,
) {
    if !gate_v0
        .as_ref()
        .is_some_and(|report| report.overall_status == V0GateOverallStatus::Pass)
    {
        blocking.insert("REVIEW_BLOCK_GATE_V0".to_string());
        remediation.insert("run_v0_gate".to_string());
    }
    if !gate_v1
        .as_ref()
        .is_some_and(|report| report.overall_status == V1GateOverallStatus::Pass)
    {
        blocking.insert("REVIEW_BLOCK_GATE_V1".to_string());
        remediation.insert("run_v1_gate".to_string());
    }
    if !gate_v2
        .as_ref()
        .is_some_and(|report| report.overall_status == V2GateOverallStatus::Pass)
    {
        blocking.insert("REVIEW_BLOCK_GATE_V2".to_string());
        remediation.insert("run_v2_gate".to_string());
    }
    if !gate_v3
        .as_ref()
        .is_some_and(|report| report.overall_status == V3GateOverallStatus::Pass)
    {
        blocking.insert("REVIEW_BLOCK_GATE_V3".to_string());
        remediation.insert("run_v3_gate".to_string());
    }
    if !gate_v4
        .as_ref()
        .is_some_and(|report| report.overall_status == V4GateOverallStatus::Pass)
    {
        blocking.insert("REVIEW_BLOCK_GATE_V4".to_string());
        remediation.insert("run_v4_gate".to_string());
    }
}

fn maybe_read_json<T: for<'de> Deserialize<'de>>(path: &Option<PathBuf>) -> Option<T> {
    path.as_ref().and_then(|p| read_json::<T>(p).ok())
}

fn read_final_governance_prefix(workdir: &Path, rel_path: &str) -> String {
    let Ok(bytes) = fs::read(workdir.join(rel_path)) else {
        return "MISSING".to_string();
    };
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return "MISSING".to_string();
    };
    value
        .get("authority")
        .and_then(|authority| authority.get("authority_digest"))
        .and_then(serde_json::Value::as_str)
        .map(prefix16)
        .unwrap_or_else(|| "MISSING".to_string())
}

fn read_sweep_digest_prefix(workdir: &Path, rel_path: &str, field: &str) -> String {
    let Ok(bytes) = fs::read(workdir.join(rel_path)) else {
        return "MISSING".to_string();
    };
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return "MISSING".to_string();
    };
    value
        .get("sweep")
        .and_then(|sweep| sweep.get(field))
        .and_then(serde_json::Value::as_str)
        .map(prefix16)
        .unwrap_or_else(|| "MISSING".to_string())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, OpsError> {
    let bytes = fs::read(path)?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn discover_report(
    out_root: &Path,
    file: &str,
    args: &OperatorReviewPacketArgs,
) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(run_id) = &args.run_id {
        candidates.push(out_root.join(run_id).join(file));
    }
    if args.latest {
        let mut dirs = fs::read_dir(out_root)
            .ok()?
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                if path.is_dir() {
                    Some(path)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        dirs.sort();
        dirs.reverse();
        for dir in dirs {
            candidates.push(dir.join(file));
        }
    }
    candidates.push(out_root.join(file));
    candidates.into_iter().find(|path| path.is_file())
}

fn digest_opt<T: Serialize>(value: Option<&T>) -> Result<String, OpsError> {
    match value {
        Some(v) => Ok(prefix16(&crate::sha256_hex(&serde_json::to_vec(v)?))),
        None => Ok("MISSING".to_string()),
    }
}

fn prefix16(value: &str) -> String {
    value.chars().take(16).collect()
}

fn bound_codes(codes: BTreeSet<String>) -> Vec<String> {
    codes.into_iter().take(CODE_CAP).collect()
}

fn packet_digest(packet: &OperatorReviewPacketV1) -> Result<String, OpsError> {
    let mut cloned = packet.clone();
    cloned.packet_digest.clear();
    Ok(crate::sha256_hex(&serde_json::to_vec(&cloned)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models_lifecycle::{
        ActiveReviewContributingDigestsV1, ActiveReviewEvidenceV1, ActiveReviewOverallStatusV1,
        ActiveReviewSignoffAlignmentV1, AppliedSupportedSetContextV1, BackendEvidenceSlotDenialsV1,
        BackendEvidenceSlotEvidenceV1, BackendEvidenceSlotReadinessV1,
        BackendEvidenceSlotSnapshotV1, BackendSupportMatrixV1, BackendSupportStateV1,
        BurnResolutionStatusV1, DriftStatusV1, SupportedRealSlotSetExecutionDecisionV2,
    };
    use crate::validate_governance_primary_surfaces;

    fn pass_v0() -> V0GateReportV1 {
        V0GateReportV1 {
            schema_version: 1,
            overall_status: V0GateOverallStatus::Pass,
            checks: Vec::new(),
        }
    }
    fn pass_v1() -> V1GateReportV1 {
        V1GateReportV1 {
            schema_version: 1,
            overall_status: V1GateOverallStatus::Pass,
            checks: Vec::new(),
        }
    }
    fn pass_v2() -> V2GateReportV1 {
        V2GateReportV1 {
            schema_version: 1,
            overall_status: V2GateOverallStatus::Pass,
            checks: Vec::new(),
        }
    }
    fn pass_v3() -> V3GateReportV1 {
        V3GateReportV1 {
            schema_version: 1,
            overall_status: V3GateOverallStatus::Pass,
            checks: Vec::new(),
        }
    }
    fn pass_v4() -> V4GateReportV1 {
        V4GateReportV1 {
            schema_version: 1,
            overall_status: V4GateOverallStatus::Pass,
            checks: Vec::new(),
        }
    }

    fn snapshot() -> BackendEvidenceSnapshotV1 {
        let slot = |slot_id: &str, active_eligible: bool| BackendEvidenceSlotSnapshotV1 {
            slot_id: slot_id.to_string(),
            target_hash_prefix: "abc123".to_string(),
            backend_support: BackendSupportMatrixV1 {
                stub: BackendSupportStateV1::Supported,
                candle: BackendSupportStateV1::Supported,
                burn: BackendSupportStateV1::Supported,
            },
            evidence: BackendEvidenceSlotEvidenceV1 {
                latest_probe_report_digest_prefix: "p".to_string(),
                latest_compare_window_digest_prefix: "c".to_string(),
                latest_shadow_ready_digest_prefix: "s".to_string(),
                latest_active_evidence_digest_prefix: "a".to_string(),
                latest_drift_status: DriftStatusV1::Ok,
                freshness_probe_age_ticks: Some(1),
                freshness_compare_age_ticks: Some(1),
                freshness_no_impact_age_ticks: Some(1),
                freshness_drift_status_age_ticks: Some(1),
                hash_consistency_ok: true,
            },
            readiness: BackendEvidenceSlotReadinessV1 {
                probe_ready: true,
                shadow_ready: true,
                active_eligible,
            },
            denials: BackendEvidenceSlotDenialsV1 {
                probe: None,
                shadow: None,
                active: None,
            },
            remediation_codes: Vec::new(),
            canonical_remediation_codes: Vec::new(),
            burn_resolution: BurnSupportResolutionV1 {
                slot_id: slot_id.to_string(),
                resolution: BurnResolutionStatusV1::BurnSupportedForShadowCompare,
                support_state: crate::OptionalBackendSupportStateV1::Supported,
                rationale_codes: vec!["OK".to_string()],
                evidence_digest: "burn".to_string(),
            },
        };
        BackendEvidenceSnapshotV1 {
            schema_version: 1,
            supported_slot_set_digest: "slotset1".to_string(),
            policy_graph_digest_prefix: "policy1".to_string(),
            manifest_digest_prefix: "manifest1".to_string(),
            slots: vec![slot("world", true), slot("sae", true)],
            snapshot_digest: "snapshotdigest111111".to_string(),
        }
    }

    fn active_snapshot() -> AggregatedActiveReviewSnapshotV1 {
        let slot = |slot_id: &str, active_eligible: bool| ActiveReviewEvidenceV1 {
            slot_id: slot_id.to_string(),
            target_hash_prefix: "abc123".to_string(),
            manifest_digest_prefix: "manifest1".to_string(),
            probe_ready: true,
            shadow_ready: true,
            active_eligible,
            strict_blocking: false,
            drift_blocking: false,
            alert_blocking: false,
            primary_denial_code: None,
            remediation_codes: Vec::new(),
            contributing_evidence_digests: ActiveReviewContributingDigestsV1 {
                probe_report_digest_prefix: "p".to_string(),
                shadow_ready_digest_prefix: "s".to_string(),
                active_evidence_digest_prefix: "a".to_string(),
                strict_evidence_digest_prefix: "x".to_string(),
            },
            burn_resolution: BurnSupportResolutionV1 {
                slot_id: slot_id.to_string(),
                resolution: BurnResolutionStatusV1::BurnSupportedForShadowCompare,
                support_state: crate::OptionalBackendSupportStateV1::Supported,
                rationale_codes: vec!["OK".to_string()],
                evidence_digest: "burn".to_string(),
            },
            evidence_digest: "evidence".to_string(),
        };
        AggregatedActiveReviewSnapshotV1 {
            schema_version: 1,
            supported_slot_set_digest: "slotset1".to_string(),
            policy_graph_digest_prefix: "policy1".to_string(),
            manifest_digest_prefix: "manifest1".to_string(),
            slots: vec![slot("world", true), slot("sae", true)],
            overall_review_status: ActiveReviewOverallStatusV1::AllReviewable,
            signoff_alignment: ActiveReviewSignoffAlignmentV1 {
                aligned: true,
                status_code: "ALIGNED".to_string(),
            },
            canonical_governance_entry_digest_prefix: "MISSING".to_string(),
            final_governance_consumer_authority_digest_prefix: "MISSING".to_string(),
            governance_residual_sweep_digest_prefix: "MISSING".to_string(),
            residual_free_governance_authority_digest_prefix: "MISSING".to_string(),
            governance_absolute_sweep_digest_prefix: "MISSING".to_string(),
            absolute_final_governance_terminal_sweep_digest_prefix: "MISSING".to_string(),
            governance_ultimate_sweep_digest_prefix: "MISSING".to_string(),
            final_readiness_consumer_authority_digest_prefix: "MISSING".to_string(),
            readiness_residual_sweep_digest_prefix: "MISSING".to_string(),
            residual_free_readiness_authority_digest_prefix: "MISSING".to_string(),
            readiness_absolute_sweep_digest_prefix: "MISSING".to_string(),
            readiness_terminal_sweep_digest_prefix: "MISSING".to_string(),
            readiness_ultimate_sweep_digest_prefix: "MISSING".to_string(),
            readiness_stabilization_sweep_digest_prefix: "MISSING".to_string(),
            snapshot_digest: "activedigest111111".to_string(),
        }
    }

    fn signoff(decision: SignoffDecisionStateV1) -> OperatorSignoffDecisionV1 {
        OperatorSignoffDecisionV1 {
            schema_version: 1,
            decision,
            supported_slot_set_digest: "slotset1".to_string(),
            policy_graph_digest_prefix: "policy1".to_string(),
            manifest_digest_prefix: "manifest1".to_string(),
            evidence_snapshot_digest_prefix: prefix16("snapshotdigest111111"),
            active_review_snapshot_digest_prefix: Some(prefix16("activedigest111111")),
            operator_report_digest_prefix: prefix16("reportdigest111111"),
            applied_supported_set_digest_prefix: "slotset1".to_string(),
            applied_context_digest_prefix: "context1".to_string(),
            reviewability_reduction_digest_prefix: "reduction1".to_string(),
            canonical_readiness_spine_digest_prefix: "spine1".to_string(),
            canonical_readiness_authority_digest_prefix: "spine1".to_string(),
            canonical_governance_entry_digest_prefix: "entry1".to_string(),
            final_governance_consumer_authority_digest_prefix: "gov1".to_string(),
            governance_residual_sweep_digest_prefix: "sweep1".to_string(),
            residual_free_governance_authority_digest_prefix: "MISSING".to_string(),
            governance_absolute_sweep_digest_prefix: "MISSING".to_string(),
            absolute_final_governance_terminal_sweep_digest_prefix: "MISSING".to_string(),
            governance_ultimate_sweep_digest_prefix: "MISSING".to_string(),
            final_readiness_consumer_authority_digest_prefix: "ready1".to_string(),
            readiness_residual_sweep_digest_prefix: "rrs1".to_string(),
            residual_free_readiness_authority_digest_prefix: "MISSING".to_string(),
            readiness_absolute_sweep_digest_prefix: "MISSING".to_string(),
            readiness_terminal_sweep_digest_prefix: "MISSING".to_string(),
            readiness_ultimate_sweep_digest_prefix: "MISSING".to_string(),
            readiness_stabilization_sweep_digest_prefix: "MISSING".to_string(),
            readiness_final_consolidation_sweep_digest_prefix: "MISSING".to_string(),
            governance_final_consolidation_sweep_digest_prefix: "MISSING".to_string(),
            governance_closure_sweep_digest_prefix: "MISSING".to_string(),
            final_primary_semantics_residual_sweep_digest_prefix: "MISSING".to_string(),
            residual_free_primary_semantics_authority_digest_prefix: "MISSING".to_string(),
            primary_semantics_absolute_sweep_digest_prefix: "MISSING".to_string(),
            primary_semantics_terminal_sweep_digest_prefix: "MISSING".to_string(),
            primary_semantics_ultimate_sweep_digest_prefix: "MISSING".to_string(),
            primary_semantics_convergence_sweep_digest_prefix: "MISSING".to_string(),
            gate_report_digests: crate::operator_signoff::GateReportDigestsV1 {
                v0: "g0".to_string(),
                v1: "g1".to_string(),
                v2: "g2".to_string(),
                v3: "g3".to_string(),
            },
            reasons: Vec::new(),
            remediation_codes: Vec::new(),
            canonical_remediation_codes: Vec::new(),
            decision_digest: "decisiondigest111111".to_string(),
        }
    }

    fn operator_report() -> ConsolidatedOperatorReportV1 {
        ConsolidatedOperatorReportV1 {
            schema_version: 1,
            generated_at: 1,
            overall_status: crate::operator_report::OperatorStatus::Ok,
            run_id: Some("run".to_string()),
            policy_graph_digest_prefix: Some("policy1".to_string()),
            manifest_digest_prefix: Some("manifest1".to_string()),
            sections: crate::operator_report::OperatorSectionsV1 {
                health_section: crate::operator_report::NormalizedHealthSection {
                    status: crate::operator_report::OperatorStatus::Ok,
                    strict_mode_enabled: Some(true),
                    last_tick_age_ms: Some(1),
                    emergency_active: Some(false),
                    evidence_digest_prefixes: Vec::new(),
                    remediation_codes: Vec::new(),
                },
                eligibility_section: crate::operator_report::NormalizedEligibilitySection {
                    status: crate::operator_report::OperatorStatus::Ok,
                    slots: Vec::new(),
                    evidence_digest_prefixes: Vec::new(),
                    remediation_codes: Vec::new(),
                },
                drift_section: crate::operator_report::NormalizedDriftSection {
                    status: crate::operator_report::OperatorStatus::Ok,
                    slots: Vec::new(),
                    evidence_digest_prefixes: Vec::new(),
                    remediation_codes: Vec::new(),
                },
                alerts_section: crate::operator_report::NormalizedAlertsSection {
                    status: crate::operator_report::OperatorStatus::Ok,
                    active_alert_count: 0,
                    top_active_alerts: Vec::new(),
                    evidence_digest_prefixes: Vec::new(),
                    remediation_codes: Vec::new(),
                },
                strict_section: crate::operator_report::NormalizedStrictSection {
                    status: crate::operator_report::OperatorStatus::Ok,
                    strict_status: crate::StrictEvidenceStatusV1::Pass,
                    primary_denial_code: None,
                    strict_report_digest_prefix: None,
                    failing_check_ids: Vec::new(),
                    evidence_digest_prefixes: Vec::new(),
                    remediation_codes: Vec::new(),
                },
                gates_section: crate::operator_report::NormalizedGatesSection {
                    status: crate::operator_report::OperatorStatus::Ok,
                    gates: Vec::new(),
                    evidence_digest_prefixes: Vec::new(),
                    remediation_codes: Vec::new(),
                },
            },
            remediation_codes: Vec::new(),
            canonical_remediation_codes: Vec::new(),
            report_digest: "reportdigest111111".to_string(),
        }
    }

    fn governance(
        snapshot: &BackendEvidenceSnapshotV1,
        active: &AggregatedActiveReviewSnapshotV1,
    ) -> GovernancePrimarySurfacesV1 {
        validate_governance_primary_surfaces(snapshot, active).expect("governance")
    }

    fn applied_scope() -> AppliedSupportedSetContextV1 {
        AppliedSupportedSetContextV1 {
            schema_version: 1,
            applied_set_digest_prefix: "slotset1".to_string(),
            slots: vec!["world".to_string(), "sae".to_string()],
            decision: SupportedRealSlotSetExecutionDecisionV2::Frozen,
            previous_set_digest_prefix: "prev".to_string(),
            policy_digest_prefix: "policy".to_string(),
            context_digest: "abcd".repeat(16),
            compatibility_code: None,
        }
    }

    #[test]
    fn packet_is_deterministic() {
        let p1 = reduce_review_packet(
            Some(snapshot()),
            Some(active_snapshot()),
            Some(governance(&snapshot(), &active_snapshot())),
            Some(signoff(SignoffDecisionStateV1::ReadyForActiveReview)),
            Some(operator_report()),
            Some(pass_v0()),
            Some(pass_v1()),
            Some(pass_v2()),
            Some(pass_v3()),
            Some(pass_v4()),
            None,
            applied_scope(),
        )
        .expect("packet");
        let p2 = reduce_review_packet(
            Some(snapshot()),
            Some(active_snapshot()),
            Some(governance(&snapshot(), &active_snapshot())),
            Some(signoff(SignoffDecisionStateV1::ReadyForActiveReview)),
            Some(operator_report()),
            Some(pass_v0()),
            Some(pass_v1()),
            Some(pass_v2()),
            Some(pass_v3()),
            Some(pass_v4()),
            None,
            applied_scope(),
        )
        .expect("packet");
        assert_eq!(p1.review_stage, OperatorReviewStageV1::ReviewActiveReady);
        assert_eq!(p1.packet_digest, p2.packet_digest);
        let slot_ids = p1
            .supported_slots
            .iter()
            .map(|slot| slot.slot_id.clone())
            .collect::<Vec<_>>();
        assert_eq!(slot_ids, vec!["sae".to_string(), "world".to_string()]);
    }

    #[test]
    fn shadow_ready_stage_when_not_active() {
        let mut active = active_snapshot();
        active
            .slots
            .iter_mut()
            .for_each(|slot| slot.active_eligible = false);
        active.overall_review_status = ActiveReviewOverallStatusV1::NoneReviewable;
        let packet = reduce_review_packet(
            Some(snapshot()),
            Some(active.clone()),
            Some(governance(&snapshot(), &active)),
            Some(signoff(SignoffDecisionStateV1::ReadyForShadow)),
            Some(operator_report()),
            Some(pass_v0()),
            Some(pass_v1()),
            Some(pass_v2()),
            Some(pass_v3()),
            Some(pass_v4()),
            None,
            applied_scope(),
        )
        .expect("packet");
        assert_eq!(
            packet.review_stage,
            OperatorReviewStageV1::ReviewShadowReady
        );
    }

    #[test]
    fn blocked_on_missing_artifact() {
        let packet = reduce_review_packet(
            None,
            Some(active_snapshot()),
            Some(governance(&snapshot(), &active_snapshot())),
            Some(signoff(SignoffDecisionStateV1::ReadyForActiveReview)),
            Some(operator_report()),
            Some(pass_v0()),
            Some(pass_v1()),
            Some(pass_v2()),
            Some(pass_v3()),
            Some(pass_v4()),
            None,
            applied_scope(),
        )
        .expect("packet");
        assert_eq!(packet.review_stage, OperatorReviewStageV1::ReviewBlocked);
        assert!(packet
            .blocking_codes
            .contains(&"REVIEW_BLOCK_BACKEND_EVIDENCE_SNAPSHOT_MISSING".to_string()));
    }

    #[test]
    fn blocked_on_digest_mismatch() {
        let mut bad_signoff = signoff(SignoffDecisionStateV1::ReadyForActiveReview);
        bad_signoff.supported_slot_set_digest = "other".to_string();
        let packet = reduce_review_packet(
            Some(snapshot()),
            Some(active_snapshot()),
            Some(governance(&snapshot(), &active_snapshot())),
            Some(bad_signoff),
            Some(operator_report()),
            Some(pass_v0()),
            Some(pass_v1()),
            Some(pass_v2()),
            Some(pass_v3()),
            Some(pass_v4()),
            None,
            applied_scope(),
        )
        .expect("packet");
        assert_eq!(packet.review_stage, OperatorReviewStageV1::ReviewBlocked);
        assert!(packet
            .blocking_codes
            .contains(&"REVIEW_BLOCK_DIGEST_SLOT_SET_MISMATCH".to_string()));
    }

    #[test]
    fn blocked_on_ambiguous_slot_context() {
        let mut snap = snapshot();
        snap.slots = vec![snap.slots[0].clone()];
        let packet = reduce_review_packet(
            Some(snap),
            Some(active_snapshot()),
            Some(governance(&snapshot(), &active_snapshot())),
            Some(signoff(SignoffDecisionStateV1::ReadyForActiveReview)),
            Some(operator_report()),
            Some(pass_v0()),
            Some(pass_v1()),
            Some(pass_v2()),
            Some(pass_v3()),
            Some(pass_v4()),
            None,
            applied_scope(),
        )
        .expect("packet");
        assert_eq!(packet.review_stage, OperatorReviewStageV1::ReviewBlocked);
        assert!(packet
            .blocking_codes
            .contains(&"REVIEW_BLOCK_SCOPE_MISMATCH".to_string()));
    }

    #[test]
    fn blocked_on_gate_failure() {
        let packet = reduce_review_packet(
            Some(snapshot()),
            Some(active_snapshot()),
            Some(governance(&snapshot(), &active_snapshot())),
            Some(signoff(SignoffDecisionStateV1::ReadyForActiveReview)),
            Some(operator_report()),
            Some(V0GateReportV1 {
                schema_version: 1,
                overall_status: V0GateOverallStatus::Fail,
                checks: Vec::new(),
            }),
            Some(pass_v1()),
            Some(pass_v2()),
            Some(pass_v3()),
            Some(pass_v4()),
            None,
            applied_scope(),
        )
        .expect("packet");
        assert_eq!(packet.review_stage, OperatorReviewStageV1::ReviewBlocked);
        assert!(packet
            .blocking_codes
            .contains(&"REVIEW_BLOCK_GATE_V0".to_string()));
    }
}
