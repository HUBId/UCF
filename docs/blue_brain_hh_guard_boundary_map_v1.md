# Blue-Brain HH guard boundary map v1

Status: canonical HH-preparation Prompt 3 guard-boundary map. This document mirrors `CANONICAL_BLUE_BRAIN_HH_GUARD_BOUNDARY_MAP` in `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`; it is not a second truth source, not HH implementation, not a productive HH mode, not contract-authority expansion, not Runtime/Selection/Reference/Execution authority, and not compute-core reopening.

## 1) Candidate checked against existing guards

The only HH candidate remains `Basal Ganglia ↔ Cerebellum` as an execution-interface-mediated relation. The candidate was checked against the existing HH-readiness, first-candidate, prerequisite, closure and scope maps and keeps all existing guard outcomes:

- relation-level only;
- not-yet-implemented;
- simulation-only/diagnostic-only only;
- no productive HH mode;
- no additional HH candidate;
- no global HH or neurodynamics platform;
- no compute-core reopening;
- no planner, agent, policy, queue, orchestration or retry platform.

## 2) HH guard boundary map

| Boundary id | Class | Boundary statement | Re-scope requirement |
| --- | --- | --- | --- |
| `hh_level_no_direct_trigger_barriers` | no-direct trigger barrier | HH-level no-direct barriers are pinned: no direct action trigger, no direct execution trigger, no direct retry trigger, no direct memory commit, no direct compute invocation and no safety override. | A later re-scope must restate every no-direct barrier explicitly and fail closed if any barrier is absent. |
| `hh_contract_authority_separation` | contract-authority separation | HH diagnostics are not Contract state and do not grant contract-authority changes, contract-state writes or contract-version promotion. | A later re-scope must define a separate bounded diagnostic contract and prove it cannot mutate or satisfy existing Contract state. |
| `hh_runtime_selection_execution_authority_separation` | runtime/selection/execution separation | HH diagnostics do not create Runtime authority, Selection authority, Reference mutation authority or Execution authority. | A later re-scope must keep Runtime, Selection, Reference and Execution consumers read-only or absent unless a separate authority change is approved. |
| `hh_state_vs_contract_state_separation` | state separation | HH-state is diagnostic/simulation state only; HH-state is not contract-state, memory state, execution state or selection state. | A later re-scope must encode HH-state separately from contract-state and must not reuse Contract-state fields as HH-state carriers. |
| `hh_diagnostic_output_vs_operative_authority_separation` | diagnostic-authority separation | HH diagnostic output is evidence only and is not operative authority, not automatic advisory support and not an action, execution, retry, memory, compute or safety channel. | A later re-scope must label every output as diagnostic-only and prove no consumer treats diagnostic output as operative authority. |
| `hh_rescope_scope_drift_barrier` | scope-drift barrier | The HH candidate remains the single `Basal Ganglia ↔ Cerebellum` relation and cannot drift into regions, platforms, extra candidates, productive HH mode or compute-core reopening. | A later re-scope must preserve the single relation candidate or explicitly close this preparation line before any different HH proposal is opened. |

## 3) HH-level no-direct barrier pin

The HH barrier line pins these no-direct guards at HH level and not only as generic Blue-Brain caveats:

- no direct action trigger;
- no direct execution trigger;
- no direct retry trigger;
- no direct memory commit;
- no direct compute invocation;
- no safety override.

Every later HH artifact must carry these exact no-direct barriers. Omission of any item is a fail-closed condition and cannot be treated as permission to infer authority.

## 4) HH-state vs contract-state separation

HH-state is diagnostic/simulation state only. HH-state is not Contract state, not contract-state, not a contract write, not a contract version bump, not memory state, not execution state and not selection state.

A later re-scope may define fixture-local HH-state encodings only after separate approval. Those encodings must remain distinct from Contract-state carriers, must not satisfy Contract preconditions, and must not be used to mutate existing Runtime/Selection/Reference/Execution contracts.

## 5) Diagnostic output vs operative authority separation

HH diagnostic output is evidence only. It is not operative authority, not automatic advisory support, not a selection signal, not a runtime decision, not an execution trigger, not retry orchestration, not a memory commit, not direct compute invocation and not a safety override.

Any later consumer mapping must prove that diagnostic output remains read-only or absent for Runtime, Selection, Reference and Execution consumers unless a separate authority change is explicitly approved outside this preparation block.

## 6) Contract and authority barriers

HH is separated from Contract authority:

- no HH-based contract-authority change;
- no HH contract-state write;
- no HH contract-version promotion;
- no HH output satisfying an existing Contract condition;
- no HH state reused as Contract state.

HH is separated from Runtime/Selection/Execution authority:

- no HH-based Runtime authority;
- no HH-based Selection authority;
- no HH-based Reference mutation authority;
- no HH-based Execution authority;
- no HH-based planner, agent, policy, queue, orchestration or retry authority.

## 7) Scope-drift prevention

A later HH re-scope must remain bounded to the single `Basal Ganglia ↔ Cerebellum` relation candidate or explicitly close this preparation line first. It must not infer permission for:

- a region-level HH path;
- a global HH platform;
- a global neurodynamics platform;
- network simulation;
- compute-core work;
- additional HH candidates;
- productive HH mode;
- replacement of abstract current mode;
- replacement of bounded Kuramoto-like current mode.

## 8) Documentation and test hook

The canonical code map, this document, `docs/blue_brain_hh_candidate_scope_map_v1.md`, `docs/blue_brain_hh_prerequisite_map_v1.md`, `docs/README.md` and `docs/blue_brain_authority_chain_status_map.md` form the HH guard-barrier line. The targeted checks assert that every map entry keeps all no-direct barriers true and all authority-opening flags false.

## 9) Abschlussnotiz

The HH guard-barrier line is hardened: no-direct barriers are pinned explicitly at HH level, HH-state remains separate from Contract state, diagnostic output remains separate from operative authority, Contract/Runtime/Selection/Reference/Execution authority stays closed, and later scope drift must fail closed unless a separate explicit re-scope preserves or closes this single-relation preparation line.
