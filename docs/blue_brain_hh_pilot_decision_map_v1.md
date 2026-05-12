# Blue-Brain HH-pilot decision map v1

Status: canonical HH-pilot decision map for Prompt 4. This document mirrors `CANONICAL_BLUE_BRAIN_HH_PILOT_DECISION_MAP` in `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`; it is not a second truth source, not an HH implementation, not productive HH use, not a current mode, not a global HH platform, not compute-core work and not Runtime/Selection/Reference/Execution authority.

## 1) Decision

Pilot opened: no.

The only checked candidate is exactly **`Basal Ganglia ↔ Cerebellum`** as a relation. The candidate remains **consciously deferred** because the existing prerequisite map does not yet carry even a minimal simulation-only/diagnostic-only pilot.

This decision keeps the candidate relation-level only and does not create a productive HH path, a current model mode, a new region feature, a Planner/Agent/Policy/Retry path, a compute-core reopening, a global HH platform or any additional HH candidate.

## 2) Prerequisite review against the existing map

| Prerequisite group | Current state | Pilot consequence |
| --- | --- | --- |
| Single candidate and bounded surfaces | Satisfied: exactly `Basal Ganglia ↔ Cerebellum` is isolated, and the Basal-Ganglia/Cerebellum surfaces are bounded. | Enough to review one relation candidate only; not enough to run a pilot. |
| Relation implementation | Missing: the relation is still architecture-lane/deferred/not-yet-implemented. | Blocks pilot opening. |
| HH input/output contracts | Missing: no deterministic HH input vocabulary or output vocabulary exists. | Blocks pilot opening. |
| Deterministic fixtures and goldens | Missing. | Blocks pilot opening. |
| Fixed encoding | Missing: no canonical byte/order/fixed-point encoding for HH fixture inputs, outputs or comparisons exists. | Blocks pilot opening. |
| Performance budget | Missing: no bounded offline step-count, fixture-count, runtime, memory or artifact-size budget exists. | Blocks pilot opening. |
| Diagnostic consumer mapping and authority proof | Missing: no consumer mapping proves diagnostic-only use without authority promotion. | Blocks pilot opening. |

Conclusion: the candidate remains a deferred backlog item, not an opened simulation-only/diagnostic-only HH pilot.

## 3) Guard and contract boundaries pinned

All HH-level guards stay pinned while the candidate remains deferred:

- no direct action trigger;
- no direct execution trigger;
- no direct retry trigger;
- no direct memory commit;
- no direct compute invocation;
- no safety override;
- no HH-based Runtime authority;
- no HH-based Selection authority;
- no HH-based Reference mutation authority;
- no HH-based Execution authority.

State separation is explicit: **HH model state is not Contract state**. Diagnostic separation is explicit: **HH diagnostic output is not operative authority**, not automatic advisory support, not a productive output and not a substitute for existing Contract state.

## 4) Simulation-only versus productive use

The only conceivable future shape remains simulation-only/diagnostic-only, but it is not opened here. A future proposal would need a separate re-scope with deterministic contracts, fixtures/goldens, fixed encodings, budgets and fail-closed authority checks before any diagnostic-only pilot could run.

This file explicitly keeps:

- no productive HH use;
- no current HH mode;
- no global HH platform;
- no compute-core reopening;
- no new Basal-Ganglia or Cerebellum region functionality;
- no Planner/Agent/Policy/Retry work;
- no extra HH candidates.

## 5) Targeted check intent

The matching code checks assert that every pilot-decision entry:

1. names only `Basal Ganglia ↔ Cerebellum`;
2. keeps `pilot_opened = false` and `deferred = true`;
3. remains relation-level only and simulation-only/diagnostic-only scoped;
4. forbids productive/current-mode HH;
5. keeps HH model state separate from Contract state;
6. keeps diagnostic output separate from operative authority;
7. keeps every no-direct-* guard pinned;
8. forbids global HH platform, compute-core reopening, new region functionality, Planner/Agent/Policy/Retry work and additional HH candidates.

## 6) Abschlussnotiz

Changed decision: the HH candidate was actively checked for a narrow pilot, but the pilot was **not** opened. It remains deferred because the prerequisite map still lacks the relation implementation, HH input/output contracts, deterministic fixtures/goldens, fixed encoding, performance budget, diagnostic consumer mapping and authority proofs required even for a simulation-only/diagnostic-only path.

This is still not an HH production implementation because it creates no runnable HH simulation, no productive output, no current model mode, no Runtime/Selection/Execution authority, no Contract state, no direct action/execution/retry/memory/compute/safety path, no global HH platform and no compute-core work.
