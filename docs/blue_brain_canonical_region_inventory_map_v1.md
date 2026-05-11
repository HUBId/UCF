# Blue-Brain Canonical Region Inventory Map v1

Status: **canonical inventory consolidation** for the current UCF-relevant Blue-Brain region basis. This map is classification and role-closure only. It creates no seventh region, no new model deepening, no global neuroarchitecture platform, no planner/agent/policy/retry logic, and no compute-core work.

Authority note: `docs/blue_brain_authority_chain_status_map.md` remains the single authority-classification entrypoint. This file consolidates the region inventory used by that current-authority line and must be read as a supporting current reference, not as a second truth source.

## 1) Inventory classes

| Class | Meaning | Promotion rule |
| --- | --- | --- |
| `canonical active anatomical region` | A bounded UCF-relevant anatomical region named by the current Blue-Brain authority chain. | Active only when named by current authority and present in `CANONICAL_BLUE_BRAIN_REGION_INVENTORY_MAP`. |
| `supporting functional surface` | Runtime, Selection, Reference/Context, Memory, Execution, Dynamics, Relation or Model-Deepening surface that carries bounded reads/diagnostics for a region. | Supports a region role but is not itself a region. |
| `historical functional precursor` | Earlier functional lane that helped select or describe a later anatomical mapping. | Historical only; cannot reopen a region or override current roles. |
| `non-canonical/internal-only shadow surface` | DBM, microcircuit, biophys/neuro, adjacent-domain or deferred anatomical surface outside current authority. | No implicit promotion; explicit current-authority promotion is required. |

## 2) Canonical active anatomical regions

Exactly these six anatomical regions are canonical active UCF-relevant Blue-Brain regions now:

| Region | Canonical role | Bounded effect on Runtime / Selection / Reference / Execution | Non-overlap boundary |
| --- | --- | --- | --- |
| Hippocampus | context / reference / episode / indexing | Bounded context/reference reads can inform runtime diagnostics, selection context, reference lookup and execution-reference diagnostics. | Not salience, not relay, not action-gating, not prediction/timing, not drive/homeostasis, and never memory-commit or execution authority. |
| Amygdala | salience / valence / caveat / priority | Bounded salience/caveat reads can annotate runtime, selection, reference and execution-facing diagnostics. | Not context indexing, not relay routing, not action execution, not drive authority, not policy/safety override. |
| Thalamus | relay / gating / routing | Bounded relay/gating/routing reads can shape diagnostic routing visibility across existing contracts. | Not a global router, not action-channel selection authority, not memory/compute/safety authority. |
| Basal Ganglia | action-gating / suppression / channel-selection | Bounded selection-readiness reads can suppress or caveat action-channel candidates through existing Selection/Contract surfaces. | Not action execution, not allowed-actions expansion, not retry orchestration, not policy authority. |
| Cerebellum | prediction / timing / correction / mismatch | Bounded prediction/timing/mismatch reads can annotate execution-interface and reference diagnostics. | Not execution trigger, not action selection, not relay ownership, not compute invocation. |
| Hypothalamus | bounded drive / homeostasis / urgency / state-pressure | Bounded urgency/state-pressure/regulation reads can inform Runtime, Selection, Context/Reference and Execution/Reference diagnostics. | Not planner/agent logic, not policy/governance, not retry orchestration, not memory mutation, not action or safety authority. |

## 3) Supporting functional surfaces

The following surfaces support the six region roles without becoming regions themselves:

- Runtime transition/feedback read surface.
- Selection/priority/deferral contract read surface.
- Reference/Context read surface.
- Memory retrieval/reference diagnostic surface.
- Execution eligibility/reference interaction diagnostic surface.
- Bounded dynamics advisory diagnostics.
- IR1 bounded inter-region relation diagnostics.
- MD1/MD3 relation-local model-deepening diagnostics.

These surfaces are bounded consumers or diagnostic carriers. They do not create a new region role and cannot transfer one region's role to another.

## 4) Historical functional precursors

Historical functional lanes remain useful only as lineage labels:

- attention/selection-related functional path → Hippocampus lineage;
- caveat/threat salience lane → Amygdala lineage;
- relay integration lane → Thalamus lineage;
- action-gating mediation lane → Basal Ganglia lineage;
- prediction/timing/correction/mismatch calibration lane → Cerebellum lineage;
- bounded drive/homeostasis/urgency modulation lane → Hypothalamus lineage.

These precursors do not reopen Prefrontal Cortex, Anterior Cingulate Cortex, Insula or any other anatomical option as current active regions.

## 5) Non-canonical/internal-only shadow surfaces

The following remain non-canonical/internal-only unless a later current-authority document explicitly promotes them:

- Prefrontal Cortex, Anterior Cingulate Cortex and Insula historical/deferred anatomical options.
- `crates/dbm_*` DBM-style surfaces.
- `crates/microcircuit_*` microcircuit, L4, spike, rhythm, population, setpoint or related implementation surfaces.
- `crates/biophys_*` and adjacent neuro/asset/morphology/channel/solver support surfaces.
- Brain, DigitalBrain, Neuromod, SNN, FEP or adjacent-domain experimental surfaces.

Presence in workspace code, tests, fixtures, diagnostics or historical docs is not promotion into the canonical region inventory.

## 6) Out of scope for this closure pack prompt

This inventory closeout explicitly does not open:

- any additional anatomical region;
- any new model deepening or Hodgkin-Huxley productive integration;
- any global neuroarchitecture, neurodynamics or model platform;
- any planner, agent, policy-governance or retry-orchestration logic;
- any new Real Compute / Compute-Core behavior.

## 7) Guard checks

The code-side `CANONICAL_BLUE_BRAIN_REGION_INVENTORY_MAP` pins the six active regions, their non-overlapping roles, the shadow-surface classification and the no-scope-expansion boundary. The targeted tests assert that:

1. the active inventory equals the current bounded anatomical region map;
2. each active region has exactly one role;
3. historical/deferred anatomical options classify as non-canonical/internal-only shadow surfaces;
4. the authority map and this inventory do not conflict;
5. this closure pack prompt opens no seventh region, no global platform and no compute-core work.

## 8) Structural Closure Pack next steps

1. Use this inventory as the fixed input for a relation-level closure pass that reviews all bounded inter-region reads against the six final roles.
2. Re-check model-deepening references so MD1/MD3 stay relation-local and cannot become region authority.
3. Refresh maintenance/readiness evidence after the inventory is consumed by the next closure pass, without adding regions or compute behavior.
