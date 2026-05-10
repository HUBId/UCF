# Blue-Brain Discoverability Findings Map v1

Status: **maintenance-facing findings map** for the current post-BR6/IR1/MD2/MD3/SC1 Blue-Brain repo state. This document records discoverability and reference-clarity findings only; it is not a roadmap, not a new authority source and not a feature-expansion proposal.

## Scope lock

This pass keeps the existing maintenance-only end state intact:

- current bounded anatomical regions remain exactly **Hippocampus, Amygdala, Thalamus, Basal Ganglia, Cerebellum and Hypothalamus**;
- the bounded inter-region architecture remains IR1 and stays advisory/reference/diagnostic;
- active selective model-deepening remains exactly `Amygdala ↔ Thalamus` and `Amygdala ↔ Basal Ganglia`;
- non-deepened regions remain in `abstract functional current mode`;
- no new region, third model-deepening candidate, compute-core work, planner/agent surface, policy/governance platform, retry/orchestration path, retrieval/reasoning platform or memory-persistence authority is introduced.

## Finding classes

| Finding class | Maintenance meaning | Handling rule |
| --- | --- | --- |
| `current_reference_path_unclear` | A reader can find several plausible starting points before seeing the current authority chain. | Point back to `docs/README.md`, `docs/blue_brain_authority_chain_status_map.md` and `docs/blue_brain_maintenance_discoverability_map_v1.md` without creating a second truth source. |
| `historical_pointer_too_prominent` | BB25/BB27/BB29 or earlier transition files are listed before their historical/supporting status is obvious. | Keep the audit trail, but label it as historical/supporting and route current consumption through the authority map. |
| `non_canonical_path_too_visible` | DBM/microcircuit/neuro/helper surfaces or candidate residues can look like current regions, relations or models. | Keep them discoverable only through the shadow-surface inventory and explicit non-canonical/internal-only wording. |
| `duplicate_reference_path` | The same current state is reachable through multiple index paths with different emphasis. | Collapse the reading order to one canonical entry path plus supporting references. |
| `missing_index_hint` | A valid maintenance-facing reference exists but is not named where maintainers look first. | Add a small index hint, not feature prose. |
| `no_change_needed` | The checked area already separates current, supporting, historical and non-canonical meaning. | Leave unchanged and record the reason. |

## Findings from this pass

| Finding class | Repo area checked | Finding | Resolution in this pass |
| --- | --- | --- | --- |
| `current_reference_path_unclear` | `docs/README.md` and authority/discoverability maps | The README already named the authority and discoverability maps, but lacked a compact consumer path that separates current authority, supporting references, historical snapshots and non-canonical surfaces before the long series index. | Added/linked this findings map and tightened the discoverability map with a maintenance quick-start path. |
| `historical_pointer_too_prominent` | `docs/README.md` and `docs/roadmap/REPO_MAP.md` | Long BB25/BB27/BB29 and transition-era lists remained highly visible and could be read before their historical/supporting role was clear. | Added explicit historical/supporting labels and a current-first note in the repo map; kept the preserved audit trail intact. |
| `non_canonical_path_too_visible` | README/index and module-level discoverability surfaces | Additional DBM/microcircuit/neuro surfaces remain searchable and useful, but can be misread as canonical region/platform scope without the inventory. | Re-routed non-canonical interpretation through `docs/blue_brain_non_canonical_shadow_surface_inventory_v1.md`; no crate/module promotion. |
| `duplicate_reference_path` | README, authority map, discoverability map and repo map | The same post-BR6/IR1/MD2/MD3/SC1 end state was referenced by several indexes with different amounts of context. | Clarified that the authority map is the single classification source, the discoverability map is the reading order and this file is only findings evidence. |
| `missing_index_hint` | `docs/roadmap/REPO_MAP.md` | The repo map did not surface the current BR6/IR1/MD2/MD3/SC1 path before older BB25-BB29 transition lines. | Added a compact current Blue-Brain maintenance entrypoint block and corrected the maintenance-default note from post-BR5 to post-SC1/post-MD3. |
| `no_change_needed` | Region role maps, IR1 docs, MD2/MD3 docs and guard semantics | The six-region set, implemented-vs-deferred relation wording, two active model-deepening lines and no-direct-* boundaries are already present in current authority/supporting docs. | No feature or behavior change; only index/discoverability wording was adjusted. |

## Maintenance reading order after cleanup

1. Start at `docs/README.md` for the operational index.
2. Use `docs/blue_brain_authority_chain_status_map.md` as the single source for current/supporting/historical/non-canonical classification.
3. Use `docs/blue_brain_maintenance_discoverability_map_v1.md` for the compact reading order.
4. Use region docs BR1-BR6, IR1 and MD2/MD3 only as scoped supporting/current references according to the authority map.
5. Use `docs/blue_brain_non_canonical_shadow_surface_inventory_v1.md` for any DBM/microcircuit/neuro/helper surface before inferring operational meaning.

## Closure note

The pass found discoverability drift, not behavior drift. Further maintenance may continue shortening older index lists and adding labels where historical transition docs are still prominent, but that work must remain cleanup-only and must not open a new region, relation, model candidate or platform lane.
