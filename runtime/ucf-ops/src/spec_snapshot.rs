use std::fs;
use std::path::PathBuf;

use ucf_compute::{
    BackendComponentId, ModelSlot, ModelStore, StageContractVersion, COMPUTE_SUMMARY_SCHEMA_VERSION,
};
use ucf_policy::policy_packs::{load_and_merge_policy_graph, PolicyGraphProvenanceRecord};

use crate::OpsError;

#[derive(Debug, Clone)]
pub struct SpecSnapshotArgs {
    pub policy: PathBuf,
    pub overlay: Option<PathBuf>,
    pub out: PathBuf,
}

#[derive(Debug, Clone)]
struct RecordSchemaSpec {
    name: &'static str,
    schema_version: u16,
    key_fields: &'static [&'static str],
}

#[derive(Debug, Clone)]
struct StageContractSpec {
    stage: &'static str,
    versions: &'static [StageContractVersion],
    output_fields: &'static [&'static str],
    quantization: &'static str,
}

#[derive(Debug, Clone)]
struct TensorReqSpec {
    name: &'static str,
    shape: &'static str,
    dtype: &'static str,
}

const RECORD_SCHEMA_SPECS: &[RecordSchemaSpec] = &[
    RecordSchemaSpec {
        name: "compute.evidence_chain",
        schema_version: COMPUTE_SUMMARY_SCHEMA_VERSION,
        key_fields: &["schema_version", "spikes_digest", "chain_digest"],
    },
    RecordSchemaSpec {
        name: "compute.ipc",
        schema_version: ucf_compute::ipc::IPC_SCHEMA_VERSION,
        key_fields: &["schema_version", "request_id", "stage"],
    },
    RecordSchemaSpec {
        name: "compute.remote",
        schema_version: 1,
        key_fields: &["schema_version", "nonce", "payload_digest"],
    },
    RecordSchemaSpec {
        name: "policy.graph",
        schema_version: 1,
        key_fields: &["schema_version", "base_version", "overlay_version"],
    },
    RecordSchemaSpec {
        name: "ess.governance_state",
        schema_version: 2,
        key_fields: &["schema_version", "cooldown_until_tick", "flags"],
    },
    RecordSchemaSpec {
        name: "models.supported_real_slot_set_v1",
        schema_version: 1,
        key_fields: &["schema_version", "slots", "set_digest"],
    },
    RecordSchemaSpec {
        name: "models.backend_evidence_snapshot_v1",
        schema_version: 1,
        key_fields: &[
            "schema_version",
            "supported_slot_set_digest",
            "slots",
            "snapshot_digest",
        ],
    },
];

const STAGE_CONTRACT_SPECS: &[StageContractSpec] = &[
    StageContractSpec {
        stage: "world",
        versions: &[StageContractVersion::V1],
        output_fields: &[
            "surprise",
            "prediction_error",
            "state_norm",
            "prediction_digest",
        ],
        quantization: "f32 scalars in [0,1], digest bytes as fixed [u8;32]",
    },
    StageContractSpec {
        stage: "sae",
        versions: &[StageContractVersion::V1],
        output_fields: &[
            "spike_count",
            "spikes",
            "sparsity",
            "energy",
            "spikes_digest",
        ],
        quantization: "spike_count u16, magnitudes f32 in [0,1], digest bytes as fixed [u8;32]",
    },
    StageContractSpec {
        stage: "ssm",
        versions: &[StageContractVersion::V1],
        output_fields: &[
            "pressure",
            "readout",
            "state_norm",
            "state_digest",
            "readout_digest",
        ],
        quantization: "f32 scalars in [0,1], digest bytes as fixed [u8;32]",
    },
    StageContractSpec {
        stage: "lfm",
        versions: &[StageContractVersion::V1],
        output_fields: &["uncertainty", "stability", "energy", "state_digest"],
        quantization: "f32 scalars in [0,1], digest bytes as fixed [u8;32]",
    },
    StageContractSpec {
        stage: "llm",
        versions: &[StageContractVersion::V1],
        output_fields: &["risk", "confidence", "reason_codes", "reason_digest"],
        quantization: "risk/confidence are f32 in [0,1], reason codes bounded (MAX_REASON_CODES)",
    },
];

const BACKEND_META_FIELDS: &[&str] = &[
    "schema_version",
    "pack_name",
    "pack_id",
    "llm_backend",
    "world_backend",
    "sae_backend",
    "ssm_backend",
    "lfm_backend",
    "fixtures_digest",
    "model_hashes_digest",
    "code_version",
    "digest",
];

const BACKEND_COMPONENTS: &[BackendComponentId] = &[
    BackendComponentId::StubV0,
    BackendComponentId::ToyV1,
    BackendComponentId::CandleToyV1,
    BackendComponentId::BurnToyV1,
    BackendComponentId::LnnOdeV1,
    BackendComponentId::RemoteProxyV1,
    BackendComponentId::CandleJepaV1,
    BackendComponentId::CandleSaeV1,
    BackendComponentId::CandleSsmV1,
    BackendComponentId::CandleEbmV1,
    BackendComponentId::VljepaAdapterV0,
    BackendComponentId::CandleVljepaV1,
    BackendComponentId::BurnJepaV1,
    BackendComponentId::BurnSaeV1,
    BackendComponentId::BurnSsmV1,
    BackendComponentId::Disabled,
];

const JEPA_REQ: &[TensorReqSpec] = &[
    TensorReqSpec {
        name: "W1",
        shape: "[D,H]",
        dtype: "f32",
    },
    TensorReqSpec {
        name: "b1",
        shape: "[H]",
        dtype: "f32",
    },
    TensorReqSpec {
        name: "W2",
        shape: "[H,D]",
        dtype: "f32",
    },
    TensorReqSpec {
        name: "b2",
        shape: "[D]",
        dtype: "f32",
    },
];

const VLJEPA_REQ: &[TensorReqSpec] = &[
    TensorReqSpec {
        name: "vljepa.w1",
        shape: "[D,H]",
        dtype: "f32",
    },
    TensorReqSpec {
        name: "vljepa.b1",
        shape: "[H]",
        dtype: "f32",
    },
    TensorReqSpec {
        name: "vljepa.w2",
        shape: "[H,D]",
        dtype: "f32",
    },
    TensorReqSpec {
        name: "vljepa.b2",
        shape: "[D]",
        dtype: "f32",
    },
];

const SAE_REQ: &[TensorReqSpec] = &[
    TensorReqSpec {
        name: "sae.w_enc",
        shape: "[F,D]",
        dtype: "f32",
    },
    TensorReqSpec {
        name: "sae.b_enc",
        shape: "[F]",
        dtype: "f32",
    },
];

const SSM_REQ: &[TensorReqSpec] = &[
    TensorReqSpec {
        name: "A",
        shape: "[N,N]",
        dtype: "f32",
    },
    TensorReqSpec {
        name: "B",
        shape: "[N]",
        dtype: "f32",
    },
    TensorReqSpec {
        name: "C",
        shape: "[N]",
        dtype: "f32",
    },
];

const LFM_REQ: &[TensorReqSpec] = &[
    TensorReqSpec {
        name: "alpha",
        shape: "[N]",
        dtype: "f32",
    },
    TensorReqSpec {
        name: "Wx",
        shape: "[N,N]",
        dtype: "f32",
    },
    TensorReqSpec {
        name: "Wu",
        shape: "[N]",
        dtype: "f32",
    },
    TensorReqSpec {
        name: "b",
        shape: "[N]",
        dtype: "f32",
    },
];

const LLM_REQ: &[TensorReqSpec] = &[
    TensorReqSpec {
        name: "tok_emb",
        shape: "[32,64]",
        dtype: "f32",
    },
    TensorReqSpec {
        name: "lm_head",
        shape: "[64,32]",
        dtype: "f32",
    },
];

const EBM_REQ: &[TensorReqSpec] = &[
    TensorReqSpec {
        name: "ebm.w1",
        shape: "[d,h]",
        dtype: "f32",
    },
    TensorReqSpec {
        name: "ebm.b1",
        shape: "[h]",
        dtype: "f32",
    },
    TensorReqSpec {
        name: "ebm.w2",
        shape: "[h,1]",
        dtype: "f32",
    },
    TensorReqSpec {
        name: "ebm.b2",
        shape: "[1]",
        dtype: "f32",
    },
];

pub fn generate_spec_snapshot(args: &SpecSnapshotArgs) -> Result<(), OpsError> {
    let (_, provenance) = load_and_merge_policy_graph(&args.policy, args.overlay.as_deref())?;

    let model_store = ModelStore::from_env_default().ok();
    let markdown = render_snapshot(&provenance, model_store.as_ref());

    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.out, markdown)?;
    Ok(())
}

fn render_snapshot(
    provenance: &PolicyGraphProvenanceRecord,
    model_store: Option<&ModelStore>,
) -> String {
    let mut out = String::new();
    out.push_str("# Spec Snapshot\n\n");
    out.push_str(
        "Deterministic snapshot generated from code registries + policy pack manifests.\n\n",
    );

    out.push_str("## A) Frames / Records\n\n");
    for item in RECORD_SCHEMA_SPECS {
        out.push_str(&format!(
            "- `{}`: schema_version=`{}`; key fields: {}\n",
            item.name,
            item.schema_version,
            item.key_fields.join(", ")
        ));
    }

    out.push_str("\n## B) Stage contracts\n\n");
    for stage in STAGE_CONTRACT_SPECS {
        let versions = stage
            .versions
            .iter()
            .map(|v| v.as_u16().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&format!(
            "- `{}`: version(s) `{}`; output fields: {}; quantization: {}\n",
            stage.stage,
            versions,
            stage.output_fields.join(", "),
            stage.quantization
        ));
    }

    out.push_str("\n## C) Backends\n\n");
    out.push_str("### BackendComponentId\n\n");
    for component in BACKEND_COMPONENTS {
        out.push_str(&format!("- `{:?}` (`{}`)\n", component, *component as u8));
    }
    out.push_str("\n### BackendPackMeta schema\n\n");
    for field in BACKEND_META_FIELDS {
        out.push_str(&format!("- `{field}`\n"));
    }

    out.push_str("\n## D) Policy digests\n\n");
    out.push_str(&format!(
        "- base_pack_digest: `{}`\n",
        prefix_digest(&provenance.base_pack_digest)
    ));
    out.push_str(&format!(
        "- overlay_pack_digest: `{}`\n",
        provenance
            .overlay_pack_digest
            .as_ref()
            .map(|d| prefix_digest(d))
            .unwrap_or_else(|| "n/a".to_string())
    ));
    out.push_str(&format!(
        "- policy_graph_digest: `{}`\n",
        prefix_digest(&provenance.policy_graph_digest)
    ));
    out.push_str(&format!(
        "- determinism_policy_digest: `{}`\n",
        prefix_digest(&provenance.determinism_policy_digest)
    ));

    out.push_str("\n## E) Model slots\n\n");
    let mut slots = ModelSlot::all();
    slots.sort_by_key(|slot| slot.as_str());
    for slot in slots {
        let max_bytes = model_store
            .and_then(|store| store.specs.get(&slot).map(|s| s.max_bytes))
            .unwrap_or(64 * 1024 * 1024);
        let active_hash = model_store
            .and_then(|store| store.specs.get(&slot).and_then(|s| s.active_hash.as_ref()))
            .map(|h| prefix_digest(h))
            .unwrap_or_else(|| "n/a".to_string());
        let tensors = slot_tensor_specs(slot);

        out.push_str(&format!(
            "### `{}`\n- active_hash: `{}`\n- max_bytes: `{}`\n- required_tensors:\n",
            slot.as_str(),
            active_hash,
            max_bytes
        ));
        for t in tensors {
            out.push_str(&format!(
                "  - `{}` shape=`{}` dtype=`{}`\n",
                t.name, t.shape, t.dtype
            ));
        }
    }

    out.push_str("\n## F) Real-slot evidence contract\n\n");
    out.push_str(
        "- Supported real-slot set is bounded to `world_jepa` + exactly one of `sae`/`ssm`.\n",
    );
    out.push_str(
        "- `BackendEvidenceSnapshotV1` backend support states (`stub|candle|burn`): `SUPPORTED|UNSUPPORTED|NOT_BUILT|NOT_CONFIGURED`.\n",
    );

    out.push_str("\n## G) Safety invariants\n\n");
    out.push_str(
        "- no inferable invariant flags exported by registries; section intentionally bounded.\n",
    );

    out
}

fn prefix_digest(value: &str) -> String {
    if value.len() <= 16 {
        value.to_string()
    } else {
        format!("{}…", &value[..16])
    }
}

fn slot_tensor_specs(slot: ModelSlot) -> &'static [TensorReqSpec] {
    match slot {
        ModelSlot::Llm => LLM_REQ,
        ModelSlot::WorldJepa => JEPA_REQ,
        ModelSlot::WorldVljepa => VLJEPA_REQ,
        ModelSlot::Sae => SAE_REQ,
        ModelSlot::Lfm => LFM_REQ,
        ModelSlot::Ssm => SSM_REQ,
        ModelSlot::EbmReasoner => EBM_REQ,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn rendering_is_deterministic() {
        let provenance = PolicyGraphProvenanceRecord {
            base_pack_digest: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            overlay_pack_digest: Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string()),
            policy_graph_digest: "cccccccccccccccccccccccccccccccc".to_string(),
            schema_version: 1,
            base_version: "1.0.0".to_string(),
            overlay_version: Some("1.0.1".to_string()),
            validation_ok: true,
            determinism_policy_digest: "dddddddddddddddddddddddddddddddd".to_string(),
        };

        let a = render_snapshot(&provenance, None);
        let b = render_snapshot(&provenance, None);
        assert_eq!(a, b);
    }

    #[test]
    fn render_contains_required_sections() {
        let provenance = PolicyGraphProvenanceRecord {
            base_pack_digest: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            overlay_pack_digest: None,
            policy_graph_digest: "cccccccccccccccccccccccccccccccc".to_string(),
            schema_version: 1,
            base_version: "1.0.0".to_string(),
            overlay_version: None,
            validation_ok: true,
            determinism_policy_digest: "dddddddddddddddddddddddddddddddd".to_string(),
        };
        let rendered = render_snapshot(&provenance, None);
        for needle in [
            "## A) Frames / Records",
            "## B) Stage contracts",
            "## C) Backends",
            "## D) Policy digests",
            "## E) Model slots",
        ] {
            assert!(rendered.contains(needle));
        }
    }

    #[test]
    fn stage_registry_is_not_missing_known_stages() {
        let known = BTreeMap::from([
            ("world", true),
            ("sae", true),
            ("ssm", true),
            ("lfm", true),
            ("llm", true),
        ]);
        let have = STAGE_CONTRACT_SPECS
            .iter()
            .map(|s| (s.stage, true))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(known, have);
    }
}
