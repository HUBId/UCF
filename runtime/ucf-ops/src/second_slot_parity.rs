use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use ucf_compute::ModelSlot;
use ucf_ess::v1::{AuditPayload, ExperiencePayload};
use ucf_replay::load_fixture_records;

use crate::{
    build_compare_window_meta, prefix_hex, sample_digest_prefixes, sha256_hex,
    unified_compare_semantics_v1, CompareWindowBackendStatusV1, CompareWindowMetaV1, OpsError,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SaeComparedBackendRecordV1 {
    pub backend_id: String,
    pub model_hash_prefix: String,
    pub spike_count_delta_mean_q: u16,
    pub spike_count_delta_max_q: u16,
    pub magnitude_delta_mean_q: u16,
    pub digest_mismatch_count: u16,
    pub invalid_output_count: u16,
    pub sample_output_digest_prefixes: Vec<String>,
    pub status: CompareWindowBackendStatusV1,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SaeParityRecordV1 {
    pub schema_version: u16,
    #[serde(default)]
    pub meta: CompareWindowMetaV1,
    pub compared_backends: Vec<SaeComparedBackendRecordV1>,
    pub parity_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SsmComparedBackendRecordV1 {
    pub backend_id: String,
    pub model_hash_prefix: String,
    pub pressure_delta_mean_q: u16,
    pub pressure_delta_max_q: u16,
    pub digest_mismatch_count: u16,
    pub invalid_output_count: u16,
    pub sample_output_digest_prefixes: Vec<String>,
    pub status: CompareWindowBackendStatusV1,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SsmParityRecordV1 {
    pub schema_version: u16,
    #[serde(default)]
    pub meta: CompareWindowMetaV1,
    pub compared_backends: Vec<SsmComparedBackendRecordV1>,
    pub parity_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "record_kind", content = "payload")]
pub enum SecondSlotParityRecordV1 {
    Sae(SaeParityRecordV1),
    Ssm(SsmParityRecordV1),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SecondSlotParityReportV1 {
    pub run_id: String,
    pub semantics: String,
    pub slot_id: String,
    pub primary_backend_id: String,
    pub compared_backends: Vec<String>,
    pub burn_status: String,
    pub parity_records: Vec<SecondSlotParityRecordV1>,
    pub severe_windows: u16,
    pub warn_windows: u16,
    pub shadow_ready_hint: bool,
    pub parity_ready_hint: bool,
    pub report_digest: String,
}

pub fn detect_second_slot(repo_root: &Path) -> Result<ModelSlot, OpsError> {
    let body = fs::read_to_string(repo_root.join("docs/series_state_snapshot.md"))?;
    for line in body.lines() {
        if line.contains("Second supported slot") {
            let lower = line.to_ascii_lowercase();
            if lower.contains("sae") {
                return Ok(ModelSlot::Sae);
            }
            if lower.contains("ssm") {
                return Ok(ModelSlot::Ssm);
            }
        }
    }
    Err(OpsError::Invalid(
        "V2_SECOND_SLOT_UNKNOWN: expected sae or ssm in docs/series_state_snapshot.md".to_string(),
    ))
}

fn status_from_window(
    digest_mismatch_count: u16,
    invalid_output_count: u16,
    delta_mean: u16,
    delta_max: u16,
) -> CompareWindowBackendStatusV1 {
    if invalid_output_count > 0 {
        CompareWindowBackendStatusV1::Severe
    } else if digest_mismatch_count > 0 || delta_mean > 0 || delta_max > 0 {
        CompareWindowBackendStatusV1::Warn
    } else {
        CompareWindowBackendStatusV1::Ok
    }
}

fn compared_backend_ids(slot: ModelSlot) -> (Vec<&'static str>, &'static str) {
    let candle = match slot {
        ModelSlot::Sae => "candle_sae_v1",
        ModelSlot::Ssm => "candle_ssm_v1",
        _ => "candle_unknown_v1",
    };
    let burn = match slot {
        ModelSlot::Sae => "burn_sae_v1",
        ModelSlot::Ssm => "burn_ssm_v1",
        _ => "burn_unknown_v1",
    };
    if cfg!(feature = "backend-burn") {
        (vec![candle, burn], "PRESENT")
    } else {
        (vec![candle], "SKIP")
    }
}

fn model_hash_prefix(slot: ModelSlot) -> String {
    let probe_path = PathBuf::from("out").join(format!("probe_{}.json", slot.as_str()));
    fs::read_to_string(&probe_path)
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .and_then(|v| {
            v.get("model_sha256")
                .and_then(|v| v.as_str())
                .map(ToString::to_string)
        })
        .map(|v| prefix_hex(&v, 16))
        .unwrap_or_else(|| "unknown".to_string())
}

pub fn second_slot_parity_report(
    workdir: &Path,
    run_id: &str,
    out: &Path,
    requested_slot: Option<ModelSlot>,
) -> Result<SecondSlotParityReportV1, OpsError> {
    let detected = detect_second_slot(Path::new("."))?;
    let slot = requested_slot.unwrap_or(detected);
    if slot != detected {
        return Err(OpsError::Invalid(format!(
            "SECOND_SLOT_SCOPE_VIOLATION: configured second slot is {}",
            detected.as_str()
        )));
    }

    let fixture_path = workdir.join("ess").join("ess_fixture.json");
    let mut fixture = load_fixture_records(&fixture_path).unwrap_or_default();
    fixture.sort_by(|a, b| a.id.0.cmp(&b.id.0));

    let primary_backend_id = format!("stub_{}_v1", slot.as_str());
    let hash_prefix = model_hash_prefix(slot);
    let (backend_ids, burn_status) = compared_backend_ids(slot);

    let mut parity_records = Vec::new();
    for rec in fixture {
        if let ExperiencePayload::Audit(AuditPayload::SlotCompareWindow(w)) = rec.payload {
            if w.slot_id != slot.as_str() {
                continue;
            }
            match slot {
                ModelSlot::Sae => {
                    let mut compared_backends = backend_ids
                        .iter()
                        .map(|backend_id| {
                            let status = status_from_window(
                                w.digest_mismatch_count,
                                w.invalid_shadow_count,
                                w.mean_delta_q,
                                w.p95_delta_q,
                            );
                            SaeComparedBackendRecordV1 {
                                backend_id: (*backend_id).to_string(),
                                model_hash_prefix: hash_prefix.clone(),
                                spike_count_delta_mean_q: w.mean_delta_q,
                                spike_count_delta_max_q: w.p95_delta_q,
                                magnitude_delta_mean_q: w.mean_delta_q,
                                digest_mismatch_count: w.digest_mismatch_count,
                                invalid_output_count: w.invalid_shadow_count,
                                sample_output_digest_prefixes: sample_digest_prefixes(
                                    &w.digest_prefix_samples,
                                ),
                                status,
                            }
                        })
                        .collect::<Vec<_>>();
                    compared_backends.sort_by(|a, b| a.backend_id.cmp(&b.backend_id));
                    if compared_backends.len() > 2 {
                        compared_backends.truncate(2);
                    }
                    let meta = build_compare_window_meta(
                        &w.slot_id,
                        run_id,
                        w.t0,
                        w.t1,
                        &primary_backend_id,
                        compared_backends
                            .iter()
                            .map(|v| v.backend_id.clone())
                            .collect(),
                        "unknown".to_string(),
                    );
                    let mut parity = SaeParityRecordV1 {
                        schema_version: 1,
                        meta,
                        compared_backends,
                        parity_digest: String::new(),
                    };
                    parity.parity_digest = sha256_hex(&serde_json::to_vec(&parity)?);
                    parity_records.push(SecondSlotParityRecordV1::Sae(parity));
                }
                ModelSlot::Ssm => {
                    let mut compared_backends = backend_ids
                        .iter()
                        .map(|backend_id| {
                            let status = status_from_window(
                                w.digest_mismatch_count,
                                w.invalid_shadow_count,
                                w.mean_delta_q,
                                w.p95_delta_q,
                            );
                            SsmComparedBackendRecordV1 {
                                backend_id: (*backend_id).to_string(),
                                model_hash_prefix: hash_prefix.clone(),
                                pressure_delta_mean_q: w.mean_delta_q,
                                pressure_delta_max_q: w.p95_delta_q,
                                digest_mismatch_count: w.digest_mismatch_count,
                                invalid_output_count: w.invalid_shadow_count,
                                sample_output_digest_prefixes: sample_digest_prefixes(
                                    &w.digest_prefix_samples,
                                ),
                                status,
                            }
                        })
                        .collect::<Vec<_>>();
                    compared_backends.sort_by(|a, b| a.backend_id.cmp(&b.backend_id));
                    if compared_backends.len() > 2 {
                        compared_backends.truncate(2);
                    }
                    let meta = build_compare_window_meta(
                        &w.slot_id,
                        run_id,
                        w.t0,
                        w.t1,
                        &primary_backend_id,
                        compared_backends
                            .iter()
                            .map(|v| v.backend_id.clone())
                            .collect(),
                        "unknown".to_string(),
                    );
                    let mut parity = SsmParityRecordV1 {
                        schema_version: 1,
                        meta,
                        compared_backends,
                        parity_digest: String::new(),
                    };
                    parity.parity_digest = sha256_hex(&serde_json::to_vec(&parity)?);
                    parity_records.push(SecondSlotParityRecordV1::Ssm(parity));
                }
                _ => unreachable!(),
            }
        }
    }
    parity_records.sort_by_key(|r| match r {
        SecondSlotParityRecordV1::Sae(v) => (v.meta.t1, v.meta.window_id),
        SecondSlotParityRecordV1::Ssm(v) => (v.meta.t1, v.meta.window_id),
    });
    if parity_records.len() > 10 {
        parity_records = parity_records.split_off(parity_records.len() - 10);
    }

    let (warn_windows, severe_windows) =
        parity_records
            .iter()
            .fold((0u16, 0u16), |(warn, severe), r| {
                let statuses = match r {
                    SecondSlotParityRecordV1::Sae(v) => v
                        .compared_backends
                        .iter()
                        .map(|b| &b.status)
                        .collect::<Vec<_>>(),
                    SecondSlotParityRecordV1::Ssm(v) => v
                        .compared_backends
                        .iter()
                        .map(|b| &b.status)
                        .collect::<Vec<_>>(),
                };
                let severe_inc = if statuses
                    .iter()
                    .any(|s| matches!(s, CompareWindowBackendStatusV1::Severe))
                {
                    1
                } else {
                    0
                };
                let warn_inc = if severe_inc == 0
                    && statuses
                        .iter()
                        .any(|s| matches!(s, CompareWindowBackendStatusV1::Warn))
                {
                    1
                } else {
                    0
                };
                (
                    warn.saturating_add(warn_inc),
                    severe.saturating_add(severe_inc),
                )
            });

    let shadow_ready_hint = severe_windows == 0 && !parity_records.is_empty();
    let parity_ready_hint = severe_windows == 0;
    let compared_backends = backend_ids
        .iter()
        .map(|v| (*v).to_string())
        .collect::<Vec<_>>();

    let mut report = SecondSlotParityReportV1 {
        run_id: run_id.to_string(),
        semantics: unified_compare_semantics_v1().window_id_rule,
        slot_id: slot.as_str().to_string(),
        primary_backend_id,
        compared_backends,
        burn_status: burn_status.to_string(),
        parity_records,
        severe_windows,
        warn_windows,
        shadow_ready_hint,
        parity_ready_hint,
        report_digest: String::new(),
    };

    report.report_digest = sha256_hex(&serde_json::to_vec(&report)?);
    fs::create_dir_all(out.parent().unwrap_or_else(|| Path::new(".")))?;
    fs::write(out, serde_json::to_string_pretty(&report)?)?;
    Ok(report)
}

pub fn second_slot_parity_evidence_exists(workdir: &Path, slot: ModelSlot) -> bool {
    workdir
        .join("out")
        .join(format!("{}_parity_report.json", slot.as_str()))
        .exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_order_is_stable() {
        let mut b = ["candle_sae_v1".to_string(), "burn_sae_v1".to_string()];
        b.sort();
        assert_eq!(b[0], "burn_sae_v1");
        assert_eq!(b[1], "candle_sae_v1");
    }

    #[test]
    fn parity_digest_is_stable() {
        let record = SaeParityRecordV1 {
            schema_version: 1,
            meta: build_compare_window_meta(
                "sae",
                "r1",
                0,
                1,
                "stub_sae_v1",
                vec!["candle_sae_v1".to_string()],
                "unknown".to_string(),
            ),
            compared_backends: vec![SaeComparedBackendRecordV1 {
                backend_id: "candle_sae_v1".to_string(),
                model_hash_prefix: "abc".to_string(),
                spike_count_delta_mean_q: 0,
                spike_count_delta_max_q: 0,
                magnitude_delta_mean_q: 0,
                digest_mismatch_count: 0,
                invalid_output_count: 0,
                sample_output_digest_prefixes: vec!["0011aabb".to_string()],
                status: CompareWindowBackendStatusV1::Ok,
            }],
            parity_digest: String::new(),
        };
        let digest_a = sha256_hex(&serde_json::to_vec(&record).expect("serialize"));
        let digest_b = sha256_hex(&serde_json::to_vec(&record).expect("serialize"));
        assert_eq!(digest_a, digest_b);
    }
}
