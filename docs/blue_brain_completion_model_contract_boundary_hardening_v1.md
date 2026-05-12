# Blue-Brain Completion model/contract boundary hardening v1

Status: Completion-Series Prompt 5 boundary pass for all active model-deepening and HH lines. This document mirrors `CANONICAL_BLUE_BRAIN_COMPLETION_MODEL_CONTRACT_BOUNDARY_MAP` in `runtime/ucf-compute/src/blue_brain_region_first_integration.rs` and adds no region functionality, model platform, HH opening, planner/agent/policy/retry work, memory mutation, execution capability, or compute-core behavior.

## 1) Final deepening-and-HH boundary map

| Line | Pair | Model status | Region/relation contract read | Boundary result |
| --- | --- | --- | --- | --- |
| First bounded Kuramoto-like deepening | `Amygdala ↔ Thalamus` | active, relation-local, bounded advisory/diagnostic | implemented direct bounded advisory relation; region semantics and relation contract remain leading | stays active but never becomes Contract state or authority |
| Second bounded Kuramoto-like deepening | `Amygdala ↔ Basal Ganglia` | active, relation-local, bounded advisory/diagnostic | implemented selection-mediated relation; selection contract remains leading | stays active but never becomes action, execution, retry, memory, compute, or safety authority |
| Third-deepening closure | `Thalamus ↔ Cerebellum` | not opened | architecture-lane-only / not-yet-implemented relation | no third model state, input, output, consumer read, or contract state |
| Single HH candidate line | `Basal Ganglia ↔ Cerebellum` | deferred simulation-only/diagnostic-only candidate; no productive HH pilot | execution-interface-mediated architecture lane remains bounded/read-only and not an HH authority path | HH stays deferred; no global HH platform and no additional HH opening |

## 2) Contract-hardening invariants

The hardening pass pins these invariants across all model lines:

- **model state ≠ contract state**: bounded Kuramoto-like phase/state and any HH diagnostic/simulation state remain internal model evidence, not Runtime/Selection/Reference/Execution contract state.
- **diagnostic output ≠ authority**: diagnostic summaries, caveats, insufficient/blocked/deferred tags, and HH notes are evidence only; they cannot authorize action, execution, retry, memory mutation, compute invocation, policy decisions, or safety override.
- **Regions and relations lead**: region surfaces define bounded region semantics; IR1 relation contracts define whether a consumer may read a bounded/caveated/deferred/blocked/reference signal.
- **Runtime/Selection/Reference/Execution-interface only bounded reads**: consumers may only read already-bounded contract signals. They cannot read raw model state, promote diagnostics, or create new model-controlled contracts.
- **no global model logic**: the two bounded Kuramoto-like lines remain separate; the closed third candidate and deferred HH line cannot be aggregated into a global model platform.

## 3) no-direct-* model takeover check

All entries in `CANONICAL_BLUE_BRAIN_COMPLETION_MODEL_CONTRACT_BOUNDARY_MAP` keep the guard line explicit:

- `no-direct-action`
- `no-direct-execution`
- `no-direct-retry`
- `no-direct-memory`
- `no-direct-compute`
- no direct policy decision
- no safety override

The guards are checked against model takeover, not only against region/relation takeover. A model output may support a bounded advisory/caveat interpretation only through the existing relation contract. It never bypasses the contract, writes memory, invokes compute, retries execution, starts execution, selects an action, or overrides safety.

## 4) Separation of the first, second, third, and HH lines

- First deepening: `Amygdala ↔ Thalamus` remains the first bounded Kuramoto-like advisory/diagnostic relation-local path.
- Second deepening: `Amygdala ↔ Basal Ganglia` remains the second bounded Kuramoto-like advisory/diagnostic relation-local path and is selection-mediated.
- Third deepening: `Thalamus ↔ Cerebellum` remains reviewed but closed; no model surface is opened.
- HH line: `Basal Ganglia ↔ Cerebellum` remains the single deferred HH candidate line, simulation-only/diagnostic-only in wording, with no productive pilot and no additional HH opening.

## 5) Final model/contract boundaries

Final boundaries for this pass:

1. Model state is evidence, never contract state.
2. Diagnostics are evidence, never authority.
3. Region surfaces and relation contracts remain the only leading semantics for bounded consumer reads.
4. Runtime, Selection, Reference, and Execution-interface consumers only read bounded/caveated/deferred/blocked/reference contract signals.
5. The two bounded Kuramoto-like deepenings remain relation-local and separate.
6. No third bounded model deepening is open.
7. The single HH candidate remains deferred and non-productive.
8. There is no global model logic.

## 6) Existing caveats

- Architecture-lane-only relation entries remain visible but are not implementation.
- Selection-mediated and execution-interface-mediated readings stay bounded and mediated; they are not action/execution authority.
- HH prerequisites remain insufficient for a productive pilot.
- Historical documents with broader candidate wording remain audit trail only and are read through the current authority chain.

## 7) Abschlussnotiz

Geänderte Dateien in diesem Prompt-5-Pass:

- `runtime/ucf-compute/src/blue_brain_region_first_integration.rs` adds `CANONICAL_BLUE_BRAIN_COMPLETION_MODEL_CONTRACT_BOUNDARY_MAP` and tests for active deepenings, closed third-deepening state, HH-deferred state, bounded consumer reads, and no-direct-* model-takeover guards.
- `docs/blue_brain_completion_model_contract_boundary_hardening_v1.md` records the final deepening-and-HH boundary map.
- `docs/README.md` and `docs/blue_brain_authority_chain_status_map.md` add discoverability for this supporting current reference.

Readiness: **readiness for the Completion-Sweep**. The model layer is hardened against the regions/relations contracts without opening new region functionality, a new model platform, additional HH scope, direct authority, or global model logic.
