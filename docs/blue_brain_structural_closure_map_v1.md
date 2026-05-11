# Blue-Brain Structural Closure Map v1

Status: **structural closure closeout** for the current UCF Blue-Brain region/relation/model block. This map is a compact, repo-based roll-up over the canonical region inventory, IR1 relation map, model-boundary map, MD1/MD2 first deepening, MD3 second deepening, SC1 maintenance decision, and non-canonical shadow-surface inventory. It creates no new region, no new relation implementation, no new model deepening, no global neurodynamics platform, no planner/agent/policy/retry logic, and no compute-core work.

Authority note: `docs/blue_brain_authority_chain_status_map.md` remains the single authority-classification entrypoint. This file consolidates the structural closure state used by that authority line and mirrors `CANONICAL_BLUE_BRAIN_STRUCTURAL_CLOSURE_MAP` in `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`; it is not a competing truth source.

## 1) Structural closure classes

| Class | Meaning | Current status |
| --- | --- | --- |
| `canonical active region` | A currently active UCF-relevant anatomical region with bounded input/state/output/reference/diagnostic/contract surfaces. | Exactly six regions are active. |
| `canonical implemented relation` | A currently implemented direct bounded advisory relation. | Exactly three direct advisory relation reads are active; all are advisory/diagnostic only. |
| `canonical mediated relation` | A currently implemented relation whose read path is reference-mediated or selection-mediated. | Exactly four mediated relation reads are active. |
| `canonical model boundary` | System-wide model mode boundary: abstract current, bounded Kuramoto-like, HH simulation-only/diagnostic-only, later-HH/deferred, or internal-only. | Boundaries are status/guard facts, not model-platform authority. |
| `deferred/blocked/non-active` | Deferred, blocked, architecture-lane-only, or later-HH surfaces that remain inactive for consumer authority. | Preserved as explicit non-active states. |
| `non-canonical/internal-only` | Historical, DBM/microcircuit/biophys/neuro/adjacent-domain, test-only, or shadow surfaces outside current authority. | No consumer-readable operational authority. |

## 2) Canonical active regions

Exactly these six regions are canonical active now:

| Region | Region status | Surface status | Diagnostics/contract status | Bounded Runtime / Selection / Reference / Execution effect | Closure note |
| --- | --- | --- | --- | --- | --- |
| Hippocampus | stable maintenance-hardened | input/state/output/reference surfaces remain bounded and distinct | context/reference/episode/indexing diagnostics and contract reads | Bounded context/reference reads can inform runtime diagnostics, selection context, reference lookup and execution-reference diagnostics | Not salience, not relay, not action-gating, not prediction/timing, not drive/homeostasis, and never memory-commit or execution authority. |
| Amygdala | stable maintenance-hardened | salience/valence/caveat surfaces remain bounded | caveat/priority diagnostics and contract reads | Bounded salience/caveat reads can annotate runtime, selection, reference and execution-facing diagnostics | Not context indexing, not relay routing, not action execution, not drive authority, not policy/safety override. |
| Thalamus | stable maintenance-hardened | relay/gating/routing surfaces remain bounded | relay/routing diagnostics and contract reads | Bounded relay/gating/routing reads can shape diagnostic routing visibility across existing contracts | Not a global router, not action-channel selection authority, not memory/compute/safety authority. |
| Basal Ganglia | stable maintenance-hardened with selection caveat | action-gating/suppression/channel-selection surfaces remain bounded | selection-readiness diagnostics and contract reads | Bounded selection-readiness reads can suppress or caveat action-channel candidates through existing Selection/Contract surfaces | Not action execution, not allowed-actions expansion, not retry orchestration, not policy authority. |
| Cerebellum | stable maintenance-hardened with diagnostic-only HH caveat | prediction/timing/correction/mismatch surfaces remain bounded | execution-interface/reference diagnostics and contract reads | Bounded prediction/timing/mismatch reads can annotate execution-interface and reference diagnostics | Not execution trigger, not action selection, not relay ownership, not compute invocation. |
| Hypothalamus | stable maintenance-hardened with bounded drive caveat | urgency/state-pressure/regulation surfaces remain bounded | drive/homeostasis diagnostics and contract reads | Bounded urgency/state-pressure/regulation reads can inform Runtime, Selection, Context/Reference and Execution/Reference diagnostics | Not planner/agent logic, not policy/governance, not retry orchestration, not memory mutation, not action or safety authority. |

Classification result: no canonical active region is merely advisory-only, diagnostic-only/deferred, or non-canonical/internal-only. Advisory-only and diagnostic-only are properties of bounded reads and model paths, not replacements for the six active region identities.

## 3) Canonical relation closure

| Relation class | Relation pairs | Operational status | Boundary |
| --- | --- | --- | --- |
| `canonical implemented relation` | `Amygdala ↔ Thalamus`, `Amygdala ↔ Hypothalamus`, `Thalamus ↔ Hypothalamus` | Really operational as direct bounded advisory/diagnostic reads | No direct action, execution, retry, memory commit, compute invocation, safety override, or strong coupling. |
| `canonical mediated relation` | `Hippocampus ↔ Thalamus`, `Amygdala ↔ Basal Ganglia`, `Hippocampus ↔ Hypothalamus`, `Basal Ganglia ↔ Hypothalamus` | Really operational only through reference-mediated or selection-mediated read paths | Mediation remains part of the relation; it is not direct relation authority. |
| `canonical deferred relation` | `Amygdala ↔ Cerebellum`, `Cerebellum ↔ Hypothalamus` | Non-active deferred relation state | Deferred is not blocked, not failed execution, and not nearly active. |
| `canonical blocked relation` | `Hippocampus ↔ Basal Ganglia` | Non-active fail-closed relation path | Blocked is not failed execution and not retry authority. |
| `architecture lane only` | `Hippocampus ↔ Amygdala`, `Hippocampus ↔ Cerebellum`, `Thalamus ↔ Basal Ganglia`, `Thalamus ↔ Cerebellum`, `Basal Ganglia ↔ Cerebellum` | Architecture names a bounded lane, but implementation remains deferred/not-yet-implemented | Architecture lane is not implementation and not consumer-readable operational authority. |

Relation semantics closure:

- advisory-only relation is not an action signal;
- caveated relation is not strong positive support;
- deferred relation is not blocked relation;
- blocked relation is not failed execution;
- diagnostic-only relation is not operative relation;
- architecture-lane-only is not active implementation;
- implemented relation is not strong operative coupling.

## 4) Canonical model boundaries

| Model boundary | Canonical surface | Current status | Closure note |
| --- | --- | --- | --- |
| `abstract current mode` | BR1-BR6 region surfaces and non-deepened relation contracts | Current productive bounded status for region/contract semantics | Region and contract language remains leading; model state does not become contract state. |
| `bounded Kuramoto-like` | BB12 advisory dynamics plus exactly two selective model deepenings: `Amygdala ↔ Thalamus` and `Amygdala ↔ Basal Ganglia` | Current bounded advisory/diagnostic status only | No global Kuramoto platform, no third candidate, no direct output authority. |
| `HH simulation-only/diagnostic-only` | BB10 diagnostics, Cerebellum diagnostic path, Hypothalamus diagnostic path | Non-productive diagnostic/simulation status | No HH productive integration and no runtime/selection/execution authority. |
| `later-HH/deferred` | later selective HH deepening and Basal Ganglia ↔ Cerebellum later-HH relation language | Deferred/non-active | Requires a separate later explicit readiness decision before any implementation discussion. |
| `non-canonical/internal-only model path` | DBM/microcircuit/biophys/neuro/adjacent-domain paths | Internal-only/non-canonical | Presence in code or docs is not promotion into the current Blue-Brain model boundary. |

The two deliberately bounded model deepenings are closed as follows:

1. `Amygdala ↔ Thalamus` remains the first bounded Kuramoto-like advisory/diagnostic model-deepening surface.
2. `Amygdala ↔ Basal Ganglia` remains the second and final current bounded Kuramoto-like advisory/diagnostic model-deepening surface.

No third deepening candidate is open in the current structure phase.

## 5) Cross-surface and cross-line closure guards

The current structure keeps these distinctions explicit:

- region role does not become relation role;
- relation state does not become model state;
- model state does not become contract state;
- architecture lane does not become implementation status;
- implemented relation does not become direct execution authority;
- advisory-only, caveated, deferred, blocked, insufficient, diagnostic-only and reference-only remain distinct states;
- reference-mediated remains reference-mediated and does not become memory commit;
- selection-mediated remains selection-mediated and does not become action execution;
- HH diagnostic language remains non-productive and does not become HH implementation readiness.

## 6) No-direct and out-of-scope closure guards

The structural closure map keeps all no-direct and scope boundaries closed:

- no direct action trigger;
- no direct execution trigger;
- no direct retry trigger;
- no direct memory commit;
- no direct compute invocation;
- no safety override;
- no implicit new region;
- no implicit new model-deepening candidate;
- no global model platform;
- no global region orchestration;
- no planner/agent logic;
- no policy/governance expansion;
- no retry orchestration;
- no compute-core work.

## 7) Authority and baseline reconciliation

Current authority chain:

1. `docs/blue_brain_authority_chain_status_map.md` remains the single authority-classification entrypoint.
2. `docs/blue_brain_canonical_region_inventory_map_v1.md`, `docs/blue_brain_canonical_inter_region_relation_map_v1.md`, and `docs/blue_brain_canonical_model_boundary_map_v1.md` remain supporting current references.
3. This structural closure map is the closeout roll-up over those references and `CANONICAL_BLUE_BRAIN_STRUCTURAL_CLOSURE_MAP`.
4. Historical BB25/BB27/BB29 and implementation-stage IR1 docs remain trace evidence only when they describe narrower or older states.
5. The current default remains Maintenance/Bugfix/Cleanup/Report-Refresh unless a later explicit re-scope changes it.

## 8) Closure decision

Decision: **the Blue-Brain structure phase is closed enough for maintenance status and for a separate HH-Readiness block.**

Reason now: the repo has a single six-region inventory, a classified fifteen-pair relation map, explicit model boundaries, two bounded and closed model deepenings, and hard no-direct/out-of-scope guards. The remaining caveats are named guard states rather than open structure gaps.

Why HH-Readiness next, not HH implementation: HH language is currently simulation-only/diagnostic-only or later-HH/deferred. A separate HH-Readiness block may audit prerequisites, diagnostics, fixtures, risks and non-goals, but must not itself implement productive HH behavior, compute-core invocation, execution authority, or a global model platform.

## 9) Files changed by this closeout

- `runtime/ucf-compute/src/blue_brain_region_first_integration.rs` adds the canonical structural closure class map and structural closure map plus consistency checks.
- `runtime/ucf-compute/src/lib.rs` re-exports the structural closure classes and map.
- `docs/blue_brain_structural_closure_map_v1.md` records this structural closure decision.
- `docs/README.md` indexes this structural closure map in the current post-BR6 authority/reference line.
- `docs/blue_brain_authority_chain_status_map.md` includes this map as a supporting current reference.
