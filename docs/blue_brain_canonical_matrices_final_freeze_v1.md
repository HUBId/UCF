# Blue-Brain canonical matrices final freeze v1

Status: **final canonical matrix freeze** for the current Blue-Brain completion line. This document freezes the repo-current structure in three matrices only: regions, relations, and model modes. It mirrors `CANONICAL_BLUE_BRAIN_FINAL_REGION_MATRIX`, `CANONICAL_BLUE_BRAIN_FINAL_RELATION_MATRIX`, and `CANONICAL_BLUE_BRAIN_FINAL_MODEL_MATRIX` in `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`.

Scope: no new region, no new relation implementation, no new model deepening, no productive HH implementation, no global region/model platform, no planner/agent/policy/retry expansion, and no compute-core work.

Authority note: `docs/blue_brain_authority_chain_status_map.md` remains the single authority-classification entrypoint. This file is a supporting current reference and a final matrix mirror, not a competing truth source.

## 1) Freeze invariants

- Exactly six canonical active regions are frozen.
- Exactly fifteen inter-region pairs are frozen.
- Relation architecture and implementation status remain separate fields.
- Exactly three relations are implemented direct bounded advisory reads.
- Exactly four relations are mediated reads.
- Exactly two relations are deferred.
- Exactly one relation is blocked.
- Exactly five relations are architecture-lane-only; architecture-lane-only is not implementation.
- Exactly two bounded Kuramoto-like relation-local model deepenings are frozen.
- HH remains simulation-only/diagnostic-only or later-HH/deferred; there is no productive HH mode.
- Abstract region/relation semantics remain leading unless a listed relation-local model deepening says otherwise.
- Non-canonical/internal-only code or docs do not promote any region, relation, or model mode.

## 2) Finale kanonische Regionenmatrix

Exactly six canonical active regions are frozen:

| Region | Final role | Model mode | Bounded consumer effect | Hard boundary |
| --- | --- | --- | --- | --- |
| Hippocampus | context / reference / episode / indexing | abstract functional/current mode | Runtime, Selection, Reference/Context and Execution-diagnostic reads may consume bounded context/reference state. | Not salience, relay, action-gating, prediction/timing, drive/homeostasis, memory commit, execution, retry, compute, policy, planner, agent or safety authority. |
| Amygdala | salience / valence / caveat / priority | abstract functional/current mode | Runtime, Selection, Reference/Context and Execution-diagnostic reads may consume bounded salience/caveat state. | Not context indexing, relay routing, action execution, timing ownership, drive authority, policy/safety override, planner or agent authority. |
| Thalamus | relay / gating / routing | abstract functional/current mode | Runtime, Selection, Reference/Context and Execution-diagnostic reads may consume bounded relay/routing state. | Not global routing, action-channel authority, memory commit, salience ownership, prediction ownership, drive authority, compute invocation or safety override. |
| Basal Ganglia | action-gating / suppression / channel-selection | abstract functional/current mode | Runtime, Selection, Reference/Context and Execution-diagnostic reads may consume bounded selection-readiness state. | Not action execution, allowed-actions expansion, retry orchestration, relay ownership, context indexing, salience authority, timing authority or policy authority. |
| Cerebellum | prediction / timing / correction / mismatch | abstract functional/current mode | Runtime, Selection, Reference/Context and Execution-diagnostic reads may consume bounded prediction/timing/mismatch diagnostics. | Not execution trigger, action selection, relay routing ownership, salience ownership, context indexing, drive authority, compute invocation or safety override. |
| Hypothalamus | bounded drive / homeostasis / urgency / state-pressure | abstract functional/current mode | Runtime, Selection, Reference/Context and Execution-diagnostic reads may consume bounded urgency/state-pressure diagnostics. | Not planner/agent logic, policy/governance, retry orchestration, memory mutation, action execution, compute invocation, salience override or safety override. |

Result: every active region remains an abstract functional/current-mode region surface. Advisory-only and diagnostic-only are read/diagnostic properties, not replacements for the six active region identities.

## 3) Finale kanonische Relationsmatrix

| Pair | Architecture lane | Final relation class | Current implementation status | Read path | Model mode |
| --- | --- | --- | --- | --- | --- |
| Hippocampus ↔ Amygdala | caveated inter-region relation | architecture-lane-only | deferred/not-yet-implemented | NotYetImplemented | abstract functional/current mode |
| Hippocampus ↔ Thalamus | reference-mediated relation | mediated | implemented reference-mediated relation | ReferenceContextMediatedOnly | abstract functional/current mode |
| Hippocampus ↔ Basal Ganglia | blocked relation | blocked | blocked relation | BlockedUnavailable | abstract functional/current mode |
| Hippocampus ↔ Cerebellum | reference-mediated relation | architecture-lane-only | deferred/not-yet-implemented | NotYetImplemented | abstract functional/current mode |
| Amygdala ↔ Thalamus | direct bounded advisory relation | implemented | implemented direct bounded advisory relation | DirectBoundedAdvisoryOnly | bounded Kuramoto-like |
| Amygdala ↔ Basal Ganglia | selection-mediated relation | mediated | implemented selection-mediated relation | SelectionContractMediatedOnly | bounded Kuramoto-like |
| Amygdala ↔ Cerebellum | deferred/not-yet-active relation | deferred | deferred/not-yet-implemented | NotYetImplemented | abstract functional/current mode |
| Thalamus ↔ Basal Ganglia | selection-mediated relation | architecture-lane-only | deferred/not-yet-implemented | NotYetImplemented | abstract functional/current mode |
| Thalamus ↔ Cerebellum | direct bounded advisory relation | architecture-lane-only | deferred/not-yet-implemented | NotYetImplemented | abstract functional/current mode |
| Basal Ganglia ↔ Cerebellum | execution-interface-mediated relation | architecture-lane-only | deferred/not-yet-implemented | NotYetImplemented | later-HH/deferred |
| Hippocampus ↔ Hypothalamus | reference-mediated relation | mediated | implemented reference-mediated relation | ReferenceContextMediatedOnly | abstract functional/current mode |
| Amygdala ↔ Hypothalamus | caveated inter-region relation | implemented | implemented direct bounded advisory relation carrying caveated architecture context | DirectBoundedAdvisoryOnly | abstract functional/current mode |
| Thalamus ↔ Hypothalamus | direct bounded advisory relation | implemented | implemented direct bounded advisory relation | DirectBoundedAdvisoryOnly | abstract functional/current mode |
| Basal Ganglia ↔ Hypothalamus | selection-mediated relation | mediated | implemented selection-mediated relation | SelectionContractMediatedOnly | abstract functional/current mode |
| Cerebellum ↔ Hypothalamus | deferred/not-yet-active relation | deferred | deferred/not-yet-implemented | NotYetImplemented | abstract functional/current mode |

Counts: exactly three implemented, exactly four mediated, exactly two deferred, exactly one blocked, and exactly five architecture-lane-only relation rows.

Relation semantics that remain frozen:

- implemented means advisory/diagnostic read-only, not strong operative coupling;
- mediated means the mediation path is part of the relation and cannot be bypassed;
- deferred is not blocked;
- blocked is not failed execution and not retry authority;
- architecture-lane-only is named architecture without consumer-readable implementation;
- no relation grants direct action, execution, retry, memory, compute or safety authority.

## 4) Finale kanonische Modellmatrix

| Surface group | Rows / surfaces | Final model classification | Boundary |
| --- | --- | --- | --- |
| Six active region surfaces | Hippocampus, Amygdala, Thalamus, Basal Ganglia, Cerebellum, Hypothalamus | abstract functional/current mode | Region semantics remain leading; no region is deepened into HH or a global dynamics platform. |
| Non-deepened active relation surfaces | Hippocampus ↔ Thalamus, Hippocampus ↔ Hypothalamus, Amygdala ↔ Hypothalamus, Thalamus ↔ Hypothalamus, Basal Ganglia ↔ Hypothalamus | abstract functional/current mode | Relation contracts remain leading; advisory/mediated reads do not become model authority. |
| Inactive abstract relation surfaces | Hippocampus ↔ Amygdala, Hippocampus ↔ Basal Ganglia, Hippocampus ↔ Cerebellum, Amygdala ↔ Cerebellum, Thalamus ↔ Basal Ganglia, Thalamus ↔ Cerebellum, Cerebellum ↔ Hypothalamus | abstract functional/current mode plus deferred/blocked/architecture-only closure where applicable | Inactive rows create no fallback authority or implementation claim. |
| Selective model deepening 1 | Amygdala ↔ Thalamus | bounded Kuramoto-like | First relation-local advisory/diagnostic model deepening only. |
| Selective model deepening 2 | Amygdala ↔ Basal Ganglia | bounded Kuramoto-like | Second and final current relation-local advisory/diagnostic model deepening only. |
| BB12 dynamics reference | bounded advisory dynamics surface | bounded Kuramoto-like | Shared reference vocabulary only; no global Kuramoto platform. |
| BB10/Cerebellum/Hypothalamus HH diagnostic paths | HH diagnostic surfaces | HH simulation-only/diagnostic-only | Diagnostic/simulation-only, no productive HH integration. |
| Basal Ganglia ↔ Cerebellum and later selective HH path | later-HH candidate/deferred lane | later-HH/deferred | Future explicit narrow re-scope only; not implementation. |
| DBM/microcircuit/biophys/neuro/adjacent paths | shadow/internal paths | non-canonical/internal-only | Presence in code/docs is not promotion into current Blue-Brain authority. |

Result: exactly two bounded Kuramoto-like model deepenings remain current. All other region/relation semantics are abstract, HH simulation-only/diagnostic-only, later-HH/deferred, or non-canonical/internal-only as listed. There is no new model deepening and no productive HH mode.

## 5) Stale or competing classification cleanup

The final freeze resolves stale or competing residues as follows:

- Historical BB25/BB27/BB29 region-count and expansion-lock language is trace evidence only.
- Earlier two-region/three-region relation language is trace evidence only when narrower than the fifteen-pair matrix above.
- MD1/MD2 language that sounds like only one model deepening is historical for the first deepening; MD3 adds and closes exactly the second deepening, not a third.
- HH-readiness and HH-candidate docs remain readiness/scope/guard references only and do not implement HH.
- DBM, microcircuit, biophys/neuro, Brain/DigitalBrain/Neuromod/SNN/FEP and similar paths remain non-canonical/internal-only shadows.

## 6) Gezielte Konsistenzchecks

The code-level consistency checks now pin:

1. final region, relation and model matrices alias the existing canonical maps exactly;
2. matrix row counts remain 6 / 15 / 27;
3. relation class counts remain 3 implemented / 4 mediated / 2 deferred / 1 blocked / 5 architecture-lane-only;
4. the model matrix contains exactly six abstract region rows;
5. the model matrix contains exactly two bounded Kuramoto-like selective model deepening rows;
6. no final model-matrix row is productive HH;
7. this document is indexed from `docs/README.md` and `docs/blue_brain_authority_chain_status_map.md` as supporting current reference only.

## 7) Abschlussnotiz

Geänderte Dateien for this freeze:

- `runtime/ucf-compute/src/blue_brain_region_first_integration.rs` adds final region/relation/model matrix aliases and consistency tests.
- `runtime/ucf-compute/src/lib.rs` re-exports the final matrix aliases.
- `docs/blue_brain_canonical_matrices_final_freeze_v1.md` records this final freeze.
- `docs/README.md` and `docs/blue_brain_authority_chain_status_map.md` index this document as a supporting current reference.

Finale Regionenmatrix: the six rows in section 2 are frozen as active canonical regions, all in abstract functional/current mode.

Finale Relationsmatrix: the fifteen rows in section 3 are frozen as implemented / mediated / deferred / blocked / architecture-lane-only with architecture and implementation kept separate.

Finale Modellmatrix: the grouped rows in section 4 are frozen as abstract / bounded Kuramoto-like / HH simulation-only / later-HH / deferred or non-canonical/internal-only.

Remaining caveats:

- The matrices are closure/status maps, not behavior changes.
- Architecture lanes are preserved but not implemented.
- Mediated relations stay mediated and cannot be treated as direct reads.
- Bounded Kuramoto-like output stays advisory/diagnostic and relation-local.
- HH remains non-productive and deferred or diagnostic-only.
- Any future region, relation implementation, HH implementation or model-platform work requires a separate explicit re-scope.
