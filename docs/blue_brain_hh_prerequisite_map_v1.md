# Blue-Brain HH-prerequisite map v1

Status: canonical HH-readiness prerequisite, detail and boundary map. This document mirrors `CANONICAL_BLUE_BRAIN_HH_PREREQUISITE_MAP` in `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`; it is not a second truth source, not HH implementation, not a productive HH use, not a third model deepening, not a global HH platform, and not new Runtime/Selection/Reference/Execution authority.

## 1) Scope decision

Decision: the only candidate covered here is **`Basal Ganglia ↔ Cerebellum` as a relation**.

This document tightens the prerequisites and guard/contract/runtime boundaries for a possible later HH re-scope. It does not activate the relation, create an HH input/output contract, invoke compute, add fixtures, add planner/orchestration logic, or open Cerebellum/Basal-Ganglia region-level HH.

## 2) Region-/relation-surface check

The candidate is checked against existing region and relation surfaces before any HH prerequisite is discussed:

| Surface | Existing canonical status | HH consequence |
| --- | --- | --- |
| `Basal Ganglia` region surface | action-gating / suppression / channel-selection only | May provide later fixture vocabulary about bounded selection-channel suppression; it does not provide action execution, policy or retry authority. |
| `Cerebellum` region surface | prediction / timing / correction / mismatch only | May provide later fixture vocabulary about timing/correction/mismatch; it does not provide execution trigger, action selection or compute invocation. |
| `Basal Ganglia ↔ Cerebellum` relation surface | execution-interface-mediated relation, architecture-lane-only, deferred/not-yet-implemented | The only later HH candidate is a relation-level diagnostic review lane. It is not an implemented relation and not a productive runtime path. |

This surface check is already present enough to bound the candidate, but it is not enough to implement HH.

## 3) HH-prerequisite map classes

| Class | Meaning | Boundary |
| --- | --- | --- |
| `prerequisite already satisfied` | A necessary precondition already exists in canonical maps. | Evidence only; no implementation authority. |
| `prerequisite missing` | A required technical condition is absent. | Blocks any later HH path until a separate re-scope supplies it. |
| `hard contract boundary` | Contract semantics cannot be crossed by HH state or output. | HH state is not Contract state and HH output is not operative authority. |
| `hard guard boundary` | No-direct authority guards remain non-negotiable. | No direct action selection, execution, retry, memory, compute or safety authority. |
| `hard runtime/selection/reference boundary` | Runtime/Selection/Reference consumption remains bounded. | Runtime may read only bounded diagnostic/advisory-only output after a separate contract; Selection cannot read HH as action authority; Reference/Context remains non-mutative. |
| `simulation-only HH allowance` | HH is conceivable only as a later deterministic simulation/diagnostic fixture surface. | Diagnostic-only evidence, not productive runtime behavior. |
| `forbidden productive HH path` | HH cannot become productive mode through this line. | No implicit production switch and no direct authority. |
| `non-canonical/internal-only HH path` | Adjacent biophysical or network-simulation paths are outside canonical UCF authority. | No consumer read, no contract support and no implementation lane. |

## 4) HH prerequisite detail map

| Detail entry | State | Current evidence | Missing or binding requirement | Later re-scope gate |
| --- | --- | --- | --- | --- |
| `candidate_relation_selected_and_bounded` | already present | Exactly one later-HH relation candidate exists: `Basal Ganglia ↔ Cerebellum`. | No missing candidate-selection decision. | Preserve one-candidate scope; do not open Cerebellum, Basal Ganglia or additional HH candidates. |
| `region_relation_surfaces_checked` | already present | Basal Ganglia and Cerebellum region roles are bounded; the relation is execution-interface-mediated architecture-lane-only. | No productive region-level HH surface exists. | Derive every later fixture input/output only from these bounded surfaces. |
| `ir1_relation_surface_not_implemented` | missing | The architecture names the lane, but the implementation relation remains deferred/not-yet-implemented. | Implemented or explicitly re-scoped `Basal Ganglia ↔ Cerebellum` relation surface is absent. | A separate architecture/fixture re-scope must exist before HH diagnostic wiring. |
| `candidate_input_contract_missing` | missing | No canonical HH input vocabulary exists. | Missing input contract for selection-channel suppression, timing/correction, excitability, threshold and refractory-shape probes. | Define deterministic simulation/diagnostic fixture inputs derived from approved relation evidence only. |
| `candidate_output_contract_missing` | missing | No canonical HH output vocabulary exists. | Missing output contract for diagnostic summaries, thresholds, caveats and failure states. | Define simulation-only diagnostic outputs; advisory use needs a separate bounded advisory contract and still no authority. |
| `deterministic_fixtures_goldens_missing` | missing | No fixture corpus or golden comparison exists. | Missing deterministic fixtures/goldens for representative allowed inputs, disallowed inputs and fail-closed outputs. | Supply offline deterministic fixtures and golden outputs before any diagnostic can run. |
| `fixed_encoding_missing` | missing | No canonical HH byte/order/fixed-point encoding exists. | Missing fixed encoding for fixture inputs, diagnostic outputs, comparison keys and golden artifacts. | Define stable field order, fixed-point scalars and canonical serialization before fixtures or diagnostics can be wired. |
| `performance_budget_missing` | missing | No bounded offline run envelope exists. | Missing performance budget for step count, fixture count, runtime, memory and artifact size. | Define bounded budgets and fail-closed budget-overflow behavior before diagnostics can run. |
| `diagnostic_consumer_mapping_missing` | missing | No consumer map exists for HH diagnostic summaries. | Missing diagnostic consumer mapping for Runtime, Selection, Reference, Execution, memory, retry, compute and safety layers. | Map each diagnostic summary to diagnostic-only consumers and prove no consumer can promote it to authority. |
| `hard_contract_boundary_hh_not_contract_state` | boundary present | Model-boundary rules already keep model state outside Contract state. | HH state is not Contract state; HH output is not operative authority. | Any later HH re-scope must down-map to diagnostics only unless separately bounded as advisory support without authority. |
| `hard_guard_boundary_no_direct_authority` | boundary present | Existing Blue-Brain guards forbid direct authority. | No direct action selection, execution trigger, retry trigger, memory commit, compute invocation or safety override. | Later HH must inherit this guard unchanged and fail closed on authority promotion attempts. |
| `hard_runtime_selection_reference_boundary` | boundary present | Runtime has no productive HH read path; Selection and Reference are non-authoritative for HH. | Runtime/Selection/Reference may not treat HH as authority; Execution authority stays external. | A later Runtime read may be bounded diagnostic/advisory-only only after separate contract approval. |
| `simulation_only_hh_allowance` | boundary present | HH is conceivable only as deferred diagnostic vocabulary. | No productive HH allowance exists. | Later HH evidence must remain deterministic, fixture-bounded, diagnostic-only and separated from runtime authority. |
| `forbidden_productive_hh_path` | boundary present | HH is nowhere productive mode and the candidate relation is deferred. | Productive HH remains forbidden. | No re-scope may silently convert HH diagnostics into productive action, execution, retry, memory, compute or safety behavior. |
| `non_canonical_internal_only_hh_paths` | boundary present | Microcircuit, network simulation, CoreNEURON, Neurodamus and adjacent biophysical paths have no canonical consumer. | No internal-only path is canonical HH authority. | Keep these paths internal-only unless an explicit future repo decision creates a new scope outside this line. |
| `kuramoto_deepening_interaction_boundary` | boundary present | Existing Kuramoto-like deepenings remain separate. | HH must not substitute, extend or backdoor-promote Kuramoto-like relation deepenings. | Preserve separation from `Amygdala ↔ Thalamus` and `Amygdala ↔ Basal Ganglia`. |
| `no_scope_expansion_boundary` | boundary present | This line only tightens prerequisites and boundaries. | No direct HH implementation, production switch, network simulation, global platform, planner/orchestration, policy, retry, memory or compute-core work. | Future work must be a separate HH-readiness re-scope and cannot infer implementation permission from this map. |

## 5) Prerequisites: already present versus missing

Already satisfied:

- exactly one later-HH candidate is selected;
- the candidate is relation-only;
- candidate region/relation surfaces are bounded and checked;
- abstract/Kuramoto-like/HH state classes are separated;
- no-direct guards are already canonical;
- model state is not Contract state and model output is not operative authority;
- non-canonical/internal-only HH paths have no consumer authority.

Prerequisites still missing:

- implemented or explicitly re-scoped `Basal Ganglia ↔ Cerebellum` relation surface;
- HH input contract;
- HH output contract;
- deterministic fixture/golden corpus;
- fixed encoding;
- performance budget;
- diagnostic consumer mapping proving no Runtime/Selection/Reference/Execution/memory/retry/compute/safety authority promotion.

These missing prerequisites are hard gates. Any later implementation proposal must close them first and must carry tests or fixtures that make the closure deterministic.

## 6) Allowed and disallowed later inputs

Allowed only after a separate HH re-scope:

- deterministic simulation/diagnostic fixtures for the `Basal Ganglia ↔ Cerebellum` relation;
- bounded relation evidence about selection-channel suppression versus timing/correction support;
- explicit diagnostic probes for excitability, spike-threshold and refractory-shape vocabulary;
- canonical, reproducible fixture inputs with fixed encoding and an approved performance envelope.

Disallowed inputs:

- live direct action selection inputs;
- live direct execution trigger inputs;
- direct retry trigger inputs;
- direct memory commit inputs;
- direct compute invocation inputs;
- safety override inputs;
- global network simulation, microcircuit, CoreNEURON or Neurodamus inputs in this scope;
- inputs for any second HH candidate.

## 7) Allowed and disallowed later outputs

Allowed only after a separate HH re-scope:

- simulation-only diagnostic summaries;
- diagnostic-only evidence about relation-level excitability vocabulary;
- fail-closed diagnostic statuses for missing contracts, invalid encodings, fixture mismatch or budget overflow;
- strictly bounded advisory-only output only if a later contract separately approves it and keeps it non-authoritative.

Disallowed outputs:

- direct action selection;
- direct execution trigger;
- direct retry trigger;
- direct memory commit;
- direct compute invocation;
- safety override;
- Contract state;
- operative authority;
- automatic advisory support without a separate bounded advisory contract.

## 8) Model and diagnostics boundaries

- HH state is not Contract state.
- HH output is not operative authority.
- HH diagnostic output is not automatically advisory support.
- HH remains only simulation-only/diagnostic-only conceivable until a separate re-scope exists.
- Bounded Kuramoto-like advisory diagnostics remain separate and do not grant HH equal modulation authority.
- Abstract functional current mode remains sufficient for current productive region and relation semantics.

## 9) Runtime, selection, reference and execution boundaries

- Runtime reads HH at most as bounded diagnostic/advisory-only evidence after a separate contract re-scope.
- Selection does not read HH as action authority.
- Reference/Context remains read/reference and is not mutated by HH.
- Execution-interface behavior is not autonomized by HH.
- HH does not create planner, agent, queue, retry or orchestration logic.

## 10) Non-goals

This line explicitly keeps these out of scope:

- no direct HH implementation;
- no productive HH use;
- no network simulation;
- no general CoreNEURON, Neurodamus or compute-core reopening;
- no global model platform and no global HH platform;
- no new Planner/Agent/Policy/Retry/Queue/Orchestration logic;
- no implicit memory persistence;
- no multiple HH candidates.

## 11) Why HH remains closed now

HH remains closed because the candidate relation is not implemented, the input and output contracts are absent, the fixture/golden corpus is absent, the fixed encoding is absent, the performance envelope is absent, the diagnostic consumer mapping is absent, and all Runtime/Selection/Reference/Execution authority paths remain hard-bounded. The only accepted value of this line is prerequisite clarity for a possible later, separate, simulation-only/diagnostic-only re-scope.

## 12) Next HH-readiness steps

1. If continued, define a fixture-free review checklist for the missing `Basal Ganglia ↔ Cerebellum` input/output contract without implementing HH.
2. Cross-check the checklist against IR1, BB10, BB12, BB16, BB17, BB19 and BB21 for wording drift.
3. Decide separately whether a later simulation-only diagnostic fixture proposal is warranted; do not infer implementation approval from this prerequisite map.

## 13) Prompt-4 closure link

Prompt 4 consumes this prerequisite map in `docs/blue_brain_hh_readiness_closure_map_v1.md` and `CANONICAL_BLUE_BRAIN_HH_READINESS_CLOSURE_MAP`. The closure keeps HH closed/deferred now and preserves only a later explicit simulation-only/diagnostic-only re-scope option for the single `Basal Ganglia ↔ Cerebellum` relation candidate. No prerequisite entry in this map is implementation permission.

## 14) Abschlussnotiz

The HH prerequisite-detail line is now explicit: existing surfaces only bound the candidate; missing input/output contracts, deterministic fixtures/goldens, fixed encoding, performance budget and diagnostic consumer mapping remain hard prerequisites; allowed inputs/outputs are simulation-only/diagnostic-only; and any implementation remains blocked until a separate re-scope closes those gates.
