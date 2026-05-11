# Blue-Brain HH-readiness decision map v1

Status: canonical HH-readiness prerequisite and boundary map for a possible later Hodgkin-Huxley path. This document mirrors `CANONICAL_BLUE_BRAIN_HH_READINESS_DECISION_MAP` in `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`; it is not a second truth source, not HH implementation, not a third model deepening, not a global neurodynamics platform, and not new Runtime/Selection/Reference/Execution authority.

## 1) Readiness classes

| HH-readiness class | Meaning | Current authority boundary |
| --- | --- | --- |
| HH not justified | HH has no repo-backed functional advantage for the current surface. | Keep the surface on its current abstract or bounded contract layer. |
| HH theoretically plausible later | HH may become a later selective re-scope topic only after explicit prerequisite evidence. | Deferred; no current implementation lane and no current productive mode. |
| HH simulation-only/diagnostic-only candidate | HH can be discussed only as simulation, audit, or diagnostic evidence. | Diagnostic output cannot become region semantics, contract state, execution state, selection authority, reference authority, retry authority, or memory authority. |
| HH blocked by current architecture | The current bounded architecture, maintenance-only stack, relation status, or authority chain blocks HH promotion. | Requires a separate re-scope before any implementation discussion. |
| abstract sufficient | The current abstract functional surface is adequate for productive semantics. | Abstract functional/current mode remains a legitimate default, not an HH gap. |
| Kuramoto-like still preferable | Existing bounded Kuramoto-like advisory/diagnostic coupling, synchrony, gating, or timing is the correct level. | Relation-local and advisory/diagnostic only; no global Kuramoto platform and no HH substitution. |
| non-canonical/internal-only HH path | HH-like, microcircuit, biophysical, or adjacent research paths are outside canonical Blue-Brain authority. | No consumer read, no contract support, no direct authority. |

## 2) Region HH-readiness decision table

| Surface | Surface kind | HH-readiness class | Current preferred layer | Decision note |
| --- | --- | --- | --- | --- |
| Hippocampus | region | abstract sufficient | abstract functional/current mode | Context, reference and episode-index semantics remain surface-level; HH is not justified for current productive semantics. |
| Amygdala | region | abstract sufficient | abstract functional/current mode | Salience, valence and caveat semantics remain abstract; relation-local Kuramoto-like deepening does not make Amygdala an HH candidate. |
| Thalamus | region | abstract sufficient | abstract functional/current mode | Relay, gating and routing semantics are already bounded by contracts; HH is not current value-add. |
| Basal Ganglia | region | abstract sufficient | abstract functional/current mode | Action-gating and selection-channel semantics remain advisory; HH cannot become action or selection authority. |
| Cerebellum | region | HH simulation-only/diagnostic-only candidate | abstract functional/current mode with diagnostic-only HH caveat | Microcircuit or membrane-near questions are plausible only as simulation/audit diagnostics; no productive HH integration. |
| Hypothalamus | region | HH simulation-only/diagnostic-only candidate | abstract functional/current mode with diagnostic-only HH caveat | Drive/homeostasis/urgency pressure remains abstract; HH stays diagnostic-only unless a later explicit re-scope is approved. |

## 3) Relation and model-deepening HH-readiness decision table

| Surface | Surface kind | HH-readiness class | Current preferred layer | Decision note |
| --- | --- | --- | --- | --- |
| Hippocampus ↔ Amygdala | relation | HH not justified | bounded relation lane only / abstract current mode | The current relation does not need membrane-level dynamics and is blocked from HH promotion by architecture status. |
| Hippocampus ↔ Thalamus | relation | abstract sufficient | abstract functional/current mode | Reference-mediated relay/routing remains contract-level. |
| Hippocampus ↔ Basal Ganglia | relation | HH blocked by current architecture | deferred/architecture lane only | No implemented HH lane and no current evidence need. |
| Hippocampus ↔ Cerebellum | relation | HH blocked by current architecture | deferred/architecture lane only | Timing/prediction questions do not open HH without explicit re-scope and fixtures. |
| Amygdala ↔ Thalamus | selective model deepening | Kuramoto-like still preferable | bounded Kuramoto-like advisory/diagnostic mode | This remains the first bounded relation-local deepening; HH would be too low-level for the current coupling/gating/timing contract. |
| Amygdala ↔ Basal Ganglia | selective model deepening | Kuramoto-like still preferable | bounded Kuramoto-like advisory/diagnostic mode | This remains the second and final current bounded relation-local deepening; no implicit third candidate follows. |
| Amygdala ↔ Cerebellum | relation | HH blocked by current architecture | deferred/architecture lane only | No active implementation or diagnostic need justifies HH. |
| Thalamus ↔ Basal Ganglia | relation | HH blocked by current architecture | deferred/architecture lane only | Selection/gating coordination remains contract-level or deferred, not HH. |
| Thalamus ↔ Cerebellum | relation | HH blocked by current architecture | deferred/architecture lane only | Timing/gating questions remain bounded and deferred. |
| Basal Ganglia ↔ Cerebellum | relation | HH theoretically plausible later | later-HH/deferred only | This is the only relation-level later-HH placeholder; it remains inactive and needs explicit re-scope. |
| Hippocampus ↔ Hypothalamus | relation | abstract sufficient | abstract functional/current mode | Context-drive interaction remains abstract. |
| Amygdala ↔ Hypothalamus | relation | abstract sufficient | abstract functional/current mode | Salience-drive interaction remains abstract. |
| Thalamus ↔ Hypothalamus | relation | abstract sufficient | abstract functional/current mode | Relay-drive interaction remains abstract. |
| Basal Ganglia ↔ Hypothalamus | relation | abstract sufficient | abstract functional/current mode | Selection-drive pressure remains abstract and advisory. |
| Cerebellum ↔ Hypothalamus | relation | HH blocked by current architecture | deferred/architecture lane only | No HH lane is active for drive/timing coupling. |
| BB12 bounded advisory dynamics | dynamics reference | Kuramoto-like still preferable | bounded Kuramoto-like advisory/diagnostic mode | Coupling, synchrony, gating and timing stay at the bounded Kuramoto-like layer. |
| BB10 HH diagnostic surface | dynamics reference | HH simulation-only/diagnostic-only candidate | diagnostic/simulation only | HH stays diagnostic and cannot become productive authority. |
| Cerebellum microcircuit HH diagnostic path | dynamics reference | HH simulation-only/diagnostic-only candidate | diagnostic/simulation only | Plausible only for bounded microcircuit diagnostics, not for productive cerebellum semantics. |
| Hypothalamus HH diagnostic path | dynamics reference | HH simulation-only/diagnostic-only candidate | diagnostic/simulation only | Plausible only for bounded diagnostic exploration, not for productive homeostasis semantics. |
| Later selective HH re-scope prerequisite gate | prerequisite boundary | HH theoretically plausible later | deferred gate | Any later HH work must first define candidates, inputs/outputs, fixtures, contracts, performance limits and authority exclusions. |
| Runtime/Selection/Reference/Execution HH authority boundary | prerequisite boundary | HH blocked by current architecture | authority boundary | HH may not become Runtime, Selection, Reference or Execution authority. |
| Compute-core HH authority boundary | prerequisite boundary | HH blocked by current architecture | maintenance-only compute boundary | No direct compute invocation and no compute-core reopening are allowed in this block. |
| Non-canonical internal-only HH path | non-canonical/internal-only surface | non-canonical/internal-only HH path | outside canonical authority | DBM, microcircuit, biophys, neuro or adjacent-domain HH paths remain internal-only unless separately re-scoped. |

## 4) System-wide model-mode separation

Current separation remains mandatory:

1. `abstract functional/current mode` remains the productive default for BR1-BR6 region semantics and non-deepened relation contracts.
2. `bounded Kuramoto-like current mode` remains relation-local, bounded, advisory/diagnostic and preferable for the two current selective model deepenings plus BB12 coupling/synchrony/gating/timing diagnostics.
3. `HH simulation-only/diagnostic-only candidate` is not the hidden foundation of all regions and does not redefine region, relation, runtime, selection, reference or execution contracts.
4. `later-HH/deferred` is a possible future audit/re-scope topic only; it is not an implementation lane.
5. `non-canonical/internal-only HH path` has no consumer authority.

## 5) Minimum prerequisites before any later HH re-scope

A later HH re-scope must be opened explicitly and must at minimum provide:

- the exact region or relation candidate list, with a reason why abstract current mode or bounded Kuramoto-like mode is insufficient;
- allowed HH inputs and outputs, constrained to diagnostic or simulation artifacts unless a separate authority change is approved;
- unchanged no-direct boundaries for action, execution, retry, memory commit, compute invocation and safety override;
- evidence fixtures, deterministic golden references, and tests proving that HH diagnostics cannot affect productive Runtime/Selection/Reference/Execution decisions;
- performance and execution limits that avoid new compute-core work and keep the Real Compute Stack maintenance-only;
- compatibility notes showing that current region role maps, IR1 relation status, BB12 bounded advisory dynamics, BB16 dynamics/execution feedback, BB17 reference hardening, BB19 runtime/selection contracts and BB21 execution/reference interactions still agree;
- a docs/readiness-gate path showing no implicit third model deepening and no global model platform.

## 6) Explicit HH non-goals for this block

This HH-readiness block explicitly does not allow:

- no HH-Produktivintegration;
- no global HH platform or global neurodynamics platform;
- no HH-based Runtime authority;
- no HH-based Selection authority;
- no HH-based Reference authority;
- no HH-based Execution authority;
- no HH-driven planner, agent or policy logic;
- no retry, queue or orchestration platform;
- no compute-core reopening;
- no implicit memory persistence or memory commit;
- no implicit third model deepening;
- no replacement of abstract current mode;
- no replacement of bounded Kuramoto-like current mode where it is already preferable.

## 7) Guard and contract closure

The HH-readiness line preserves these hard guards:

- no direct action trigger;
- no direct execution trigger;
- no direct retry trigger;
- no direct memory commit;
- no direct compute invocation;
- no safety override;
- no implicit third model deepening;
- no global model platform;
- no planner/agent logic;
- no policy-governance expansion;
- no retry orchestration;
- no hidden HH productive mode.

## 8) Prompt-2 first candidate narrowing

Prompt 2 narrows this map without changing its boundaries: exactly one first later-HH candidate is isolated in `docs/blue_brain_first_hh_candidate_map_v1.md` and `CANONICAL_BLUE_BRAIN_FIRST_HH_CANDIDATE_MAP`. The selected candidate is `Basal Ganglia ↔ Cerebellum` as a relation, not a region. Cerebellum and Hypothalamus remain plausible but not first diagnostic-only region paths; `Amygdala ↔ Thalamus`, `Amygdala ↔ Basal Ganglia` and BB12 remain Kuramoto-like-preferable; abstract surfaces remain abstract sufficient; architecture-blocked and non-canonical/internal-only paths remain blocked or outside canonical authority. This narrowing does not implement HH, does not open a third model deepening, and does not change any no-direct guard.

## 9) Decision

Decision: **HH-Readiness is admissible only as prerequisite analysis and guard clarification.**

The plausible later HH space is narrow: Cerebellum and Hypothalamus are simulation-only/diagnostic-only candidates, and Basal Ganglia ↔ Cerebellum is a later-HH/deferred relation placeholder. Amygdala ↔ Thalamus, Amygdala ↔ Basal Ganglia and BB12 remain better served by bounded Kuramoto-like advisory/diagnostic dynamics. Hippocampus, Amygdala, Thalamus, Basal Ganglia and most implemented or mediated relations remain abstract sufficient or HH not justified. Deferred architecture lanes are blocked by current architecture until an explicit later re-scope supplies evidence, fixtures, contracts and performance limits.


## 10) Prompt-4 closure

Prompt 4 closes the HH-readiness sweep in `docs/blue_brain_hh_readiness_closure_map_v1.md` and `CANONICAL_BLUE_BRAIN_HH_READINESS_CLOSURE_MAP`: HH stays closed/deferred now, while exactly one later path remains technically plausible only under an explicit, narrow, simulation-only/diagnostic-only re-scope for `Basal Ganglia ↔ Cerebellum`. This closure adds no HH implementation, no productive HH mode, no compute-core reopening, no global HH platform and no additional HH candidate.
