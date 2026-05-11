# Blue-Brain HH candidate scope map v1

Status: canonical HH preparation scope map for the single later Hodgkin-Huxley candidate. This document mirrors `CANONICAL_BLUE_BRAIN_HH_CANDIDATE_SCOPE_MAP` in `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`; it is not a second truth source, not HH implementation, not a productive HH mode, not a global HH/neurodynamics platform, and not new Runtime/Selection/Reference/Execution authority.

## 1) Final candidate line

The final HH candidate is exactly:

| Candidate | Kind | Architecture relation class | Current status | Later allowance |
| --- | --- | --- | --- | --- |
| `Basal Ganglia ↔ Cerebellum` | relation | `execution-interface-mediated relation` | not-yet-implemented; no productive mode | only a possible later simulation-only/diagnostic-only re-scope |

This line means the candidate is the relation where action-channel/readiness diagnostics and timing/correction diagnostics intersect near execution eligibility. It does not make Basal Ganglia an HH region, does not make Cerebellum an HH region, and does not add another candidate.

## 2) Canonical HH candidate scope map

| Scope entry | Class | Scope statement | Boundary statement |
| --- | --- | --- | --- |
| `basal_ganglia_cerebellum_final_relation_candidate` | final single relation candidate | The final bounded HH candidate is the `Basal Ganglia ↔ Cerebellum` relation, not either region by itself. | The IR1 execution-interface-mediated diagnostic relation is the only architecture edge under consideration. |
| `relation_level_not_yet_implemented_scope_invariant` | scope invariant | Scope is relation-level only, not-yet-implemented and not a live runtime path. | A later re-scope must start from missing input/output contracts, fixtures and budgets before any diagnostic implementation discussion. |
| `simulation_diagnostic_only_no_productive_mode_invariant` | scope invariant | The only conceivable later HH mode is simulation-only/diagnostic-only. | HH diagnostics cannot become Contract state, action selection, execution trigger, retry trigger, memory commit, compute invocation or safety override. |
| `kuramoto_and_abstract_current_mode_separation` | model-boundary separation | HH does not substitute for the two existing bounded Kuramoto-like relation deepenings and does not replace abstract current modes. | `Amygdala ↔ Thalamus` and `Amygdala ↔ Basal Ganglia` remain Kuramoto-like; current productive region semantics remain abstract functional/current mode. |
| `explicit_non_goals_no_platform_or_authority` | non-goal boundary | Non-goals are part of the scope line, not optional caveats. | No network simulation, no global HH platform, no runtime/selection/execution authority, no compute-core reopening, no new region functionality, and no planner/agent/policy/retry work. |
| `next_preparation_step_scope_basis` | next preparation step | The next preparation step can only define a fixture-free relation contract review checklist. | The checklist must stay offline, deterministic, diagnostic-only and bounded to the single relation without opening HH implementation. |

## 3) Scope invariants

Every entry in the scope map keeps these invariants closed:

- relation-level only;
- not-yet-implemented;
- simulation-only/diagnostic-only only;
- no productive HH mode;
- no additional HH candidate;
- no direct action trigger;
- no direct execution trigger;
- no direct retry trigger;
- no direct memory commit;
- no direct compute invocation;
- no safety override.

## 4) Separation from Kuramoto-like and abstract current modes

HH is separate from the two bounded Kuramoto-like deepenings:

- `Amygdala ↔ Thalamus` remains the first bounded Kuramoto-like advisory/diagnostic deepening.
- `Amygdala ↔ Basal Ganglia` remains the second bounded Kuramoto-like advisory/diagnostic deepening.
- `Basal Ganglia ↔ Cerebellum` is not a third current model deepening and is not a Kuramoto replacement.

HH is also separate from abstract functional/current modes:

- current productive region semantics remain abstract functional/current mode;
- abstract current mode is not an HH gap;
- HH diagnostics, even if later separately approved, do not rewrite region contracts.

## 5) Non-goals

This scope map explicitly does not open:

- network simulation;
- a global HH platform;
- a global neurodynamics platform;
- Runtime/Selection/Reference/Execution authority;
- action-selection authority;
- execution-trigger authority;
- retry, queue or orchestration authority;
- memory commit or automatic persistence;
- direct compute invocation;
- compute-core work or compute-core reopening;
- new region functionality;
- planner, agent, policy or retry work;
- a second HH candidate;
- productive HH use.

## 6) Next preparation step

The next preparation step, if any, is a fixture-free relation contract review checklist for `Basal Ganglia ↔ Cerebellum`. That checklist may identify required future inputs, outputs, fixtures, deterministic encodings and budgets, but it must not implement HH, must not create a runtime path, and must not introduce productive authority.

## 7) Closure note

The HH candidate scope line is now final for preparation purposes: exactly one relation candidate is named, the scope is deliberately small, HH is separated from Kuramoto-like and abstract current modes, non-goals are explicit, and any later work must begin from this bounded simulation-only/diagnostic-only scope basis.
