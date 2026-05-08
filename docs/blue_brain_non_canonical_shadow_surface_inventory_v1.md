# Blue-Brain Non-Canonical Shadow Surface Inventory v1

Status: **compact maintenance-facing inventory** for DBM, microcircuit, biophys/neuro and adjacent crates that are present in the workspace but are not part of the current canonical Blue-Brain operational authority. This document is classification-only and creates no new runtime, region, relation, model, platform, planner, agent, policy, retry or compute-core behavior.

## 1) Authority boundary

Current Blue-Brain operational authority remains limited to the six bounded anatomical regions and relation/model lines named in `docs/blue_brain_authority_chain_status_map.md`:

- Hippocampus
- Amygdala
- Thalamus
- Basal Ganglia
- Cerebellum
- Hypothalamus

Everything inventoried below is a **non-canonical/internal-only shadow surface** unless a current-authority document explicitly promotes it. Presence in `Cargo.toml`, tests, helpers, diagnostics, or historical docs is not promotion.

## 2) Shadow-surface classes

| Class | Meaning | Maintenance rule |
| --- | --- | --- |
| `dbm-shadow-crate` | DBM-style crate outside the canonical authority chain or not named as an active region authority. | May remain for traceability/internal work; no implicit region. |
| `microcircuit-shadow-crate` | Stub, attractor, spike, rhythm, fusion, population, setpoint, L4 or biophys microcircuit implementation detail. | Internal-only/deferred unless explicitly tied to a canonical authority doc. |
| `biophys-neuro-shadow-crate` | Biophysics, memristor, asset, morphology, channel, solver, trace, feedback, injection or governance support crate. | Support/helper surface only; not a global neurodynamics platform. |
| `adjacent-domain-shadow-surface` | Digital-brain, brain mapper/port, neuromod, SNN, FEP, ESS or bridge domain surface not named as current Blue-Brain authority. | Adjacent integration surface only; no Blue-Brain authority by existence. |

## 3) Compact crate inventory

### DBM shadow crates outside the six canonical region authority line

- `crates/dbm_pag`
- `crates/dbm_stn`
- `crates/dbm_pmrf`
- `crates/dbm_sc`
- `crates/dbm_pprf`
- `crates/dbm_hpa`
- any future `crates/dbm_*` crate not named by the authority map as one of the six current regions

These crates are **non-canonical/internal-only/deferred** for Blue-Brain maintenance. They are not a seventh region and do not extend IR1.

### Microcircuit shadow crates

- `crates/microcircuit_core`
- `crates/microcircuit_hpa_memristor`
- `crates/microcircuit_hypothalamus_setpoint`
- `crates/microcircuit_hypothalamus_l4`
- `crates/microcircuit_sn_stub`, `crates/microcircuit_sn_attractor`, `crates/microcircuit_sn_biophys`, `crates/microcircuit_sn_l4`
- `crates/microcircuit_lc_stub`, `crates/microcircuit_lc_spike`, `crates/microcircuit_lc_biophys`, `crates/microcircuit_lc_l4`
- `crates/microcircuit_dopamin_stub`, `crates/microcircuit_dopamin_attractor`
- `crates/microcircuit_serotonin_stub`, `crates/microcircuit_serotonin_attractor`
- `crates/microcircuit_amygdala_stub`, `crates/microcircuit_amygdala_pop`, `crates/microcircuit_amygdala_biophys`, `crates/microcircuit_amygdala_l4`
- `crates/microcircuit_cerebellum_stub`, `crates/microcircuit_cerebellum_pop`
- `crates/microcircuit_pag_stub`, `crates/microcircuit_pag_attractor`, `crates/microcircuit_pag_biophys`, `crates/microcircuit_pag_l4`
- `crates/microcircuit_stn_stub`, `crates/microcircuit_stn_hold`, `crates/microcircuit_stn_biophys`
- `crates/microcircuit_pmrf_stub`, `crates/microcircuit_pmrf_rhythm`, `crates/microcircuit_pmrf_biophys`
- `crates/microcircuit_sc_stub`, `crates/microcircuit_sc_attractor`
- `crates/microcircuit_insula_stub`, `crates/microcircuit_insula_fusion`, `crates/microcircuit_insula_l4`

Microcircuit crates are implementation/support shadows. Even when their names resemble anatomical structures, they do not create canonical Blue-Brain regions, relation authority, action authority, execution authority, retry authority, memory authority, safety authority or model-platform authority.

### Biophys/neuro support shadows

- `crates/biophys_core`
- `crates/biophys_assets`
- `crates/biophys_asset_builder`
- `crates/biophys_solver`
- `crates/biophys_runtime`
- `crates/biophys_morphology`
- `crates/biophys_channels`
- `crates/biophys_compartmental_solver`
- `crates/biophys_event_queue_l4`
- `crates/biophys_synapses_l4`
- `crates/biophys_plasticity_l4`
- `crates/biophys_homeostasis_l4`
- `crates/biophys_targeting_l4`
- `crates/biophys_trace`
- `crates/biophys_injection`
- `crates/biophys_governance`
- `crates/biophys_feedback`
- `crates/memristor_backend`
- `crates/emotion_field`

These crates are support surfaces only. They do not turn MD2/MD3 into a global model platform and do not reopen Hodgkin-Huxley, Kuramoto, memristor or biophysical production authority beyond the bounded current-authority wording.

### Adjacent domain shadows

- `domains/ucf-dbm`
- `domains/ucf-biophys`
- `domains/ucf-brainbus`
- `domains/ucf-snn`
- `domains/ucf-bluebrain-bridge`
- `domains/ucf-neuromod`
- `domains/ucf-fep`
- `domains/brain/crates/ucf-bluebrain-port`
- `domains/brain/crates/ucf-brain-mapper`
- `domains/digitalbrain/crates/ucf-digitalbrain-port`
- `domains/digital-brain/crates/ucf-digital-brain`

Adjacent domains may be useful for integration or experimentation, but they are not the Blue-Brain authority chain.

## 4) Maintenance guard

When touching any shadow surface:

1. Keep the change local, deterministic and explicitly scoped.
2. Do not describe the change as canonical region expansion unless the authority map is explicitly updated by policy/spec intent.
3. Do not infer new IR1 relation classes or model-deepening candidates from crate presence.
4. Do not add planner/agent/policy/retry/platform semantics.
5. If documentation mentions a shadow surface, label it `non-canonical/internal-only shadow surface` unless current authority says otherwise.
