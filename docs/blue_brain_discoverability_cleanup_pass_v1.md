# Blue-Brain Discoverability Cleanup Pass v1

Status: **closure note** for a narrow maintenance/bugfix/cleanup pass focused on discoverability, reference clarity, documentation navigation and maintenance-facing readability. This is not a new feature line and not a second authority source.

## Files changed by the pass

- `docs/README.md` — added the discoverability findings map to the supporting reference set and made the maintenance reading order explicit before older series lists.
- `docs/blue_brain_authority_chain_status_map.md` — classified the discoverability findings map as a supporting reference, not a competing source of truth.
- `docs/blue_brain_maintenance_discoverability_map_v1.md` — added a current-first quick-start path, explicit model-pair names and a link to the findings evidence.
- `docs/blue_brain_discoverability_findings_map_v1.md` — introduced the findings map for current-path, historical-pointer, non-canonical, duplicate-path and missing-index findings.
- `docs/roadmap/REPO_MAP.md` — added a current-first Blue-Brain maintenance entrypoint and corrected the maintenance-default note from post-BR5 wording to the post-SC1/MD3 state.
- `docs/blue_brain_discoverability_cleanup_pass_v1.md` — records this closure note.

## Discoverability problems found

- The current reference path was technically present but too easy to miss before long BB25-BB29 and transition-era lists.
- Historical pointers remained highly visible in README/repo-map contexts and needed clearer current-vs-historical framing.
- Non-canonical DBM/microcircuit/neuro/helper surfaces were discoverable but needed an explicit route through the shadow-surface inventory.
- README, authority map, discoverability map and repo map provided overlapping entry paths; the pass re-stated their distinct roles.
- The repo map lacked a compact current-first hint for the post-BR6/IR1/MD2/MD3/SC1 end state.

## Cleanup performed

- Kept `docs/blue_brain_authority_chain_status_map.md` as the single classification source.
- Kept `docs/blue_brain_maintenance_discoverability_map_v1.md` as the compact reading-order reference.
- Added `docs/blue_brain_discoverability_findings_map_v1.md` as findings evidence only.
- Marked older BB25/BB27/BB29 and transition-era paths as audit trail/supporting/historical relative to the current six-region end state.
- Made active model-deepening pairs explicit: `Amygdala ↔ Thalamus` and `Amygdala ↔ Basal Ganglia`.
- Preserved the no-expansion boundary: no new region, relation, model candidate, compute-core lane, planner/agent surface, policy/governance platform, retry/orchestration path, retrieval/reasoning platform or memory-persistence authority.

## Checks run

- Verified all new/changed path references resolve locally with a targeted Python path check.
- Ran strict docs lint to catch documentation drift.
- Ran workspace tests.
- Ran formatting and clippy checks.
- Ran readiness gate for the standard test profile.

## Maturity and remaining maintenance need

Result: **maintenance-ready discoverability cleanup**. The current Blue-Brain structure is easier to find through a single current-first path, while current authority, supporting references, historical snapshots and non-canonical/internal-only surfaces remain distinguishable.

Remaining maintenance need is low and cleanup-only: future passes may continue shortening older index lists or adding local labels where historical transition documents are still too prominent, but must not introduce a new anatomical region, third model-deepening candidate or platform lane.
