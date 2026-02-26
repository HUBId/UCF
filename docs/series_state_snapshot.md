# Prompt Series State Snapshot

## Current milestone state
- Last executed prompt ID: **128** (this wrap-up/indexing prompt).
- Prompt-series index: `docs/prompt_series_index.md`.
- Prompt-to-module coverage map: `docs/module_map.md`.

## Active branches/features (repo state context)
- Current branch: `work`.
- Latest completed prompt-era feature block before this wrap-up: v1.1 readiness/signoff path (`121–127`).

## Mandatory prompt block before enabling real weights
Based on the v1.1 plan and readiness/signoff artifacts, the mandatory prompt chain is:
- **121**: weights promotion/rollback lifecycle tooling.
- **122**: VL-JEPA slot + `WeightSpec` scaffolding.
- **123**: VL-JEPA shadow rollout + drift/probe controls.
- **124**: SAE real slot spec/backend path.
- **125**: SSM optimized kernel lane + parity checks.
- **126**: optional GPU lane scaffolding and parity gating.
- **127**: v1.1 readiness gate extension + signoff flow.

## Resume rule
- To continue the series, start at **`PROMPT 129`** and increment monotonically.
- Follow `docs/prompt_rulebook.md` for structure/invariants.
- Update `docs/prompt_series_index.md` and `docs/module_map.md` in the same change.
