# Blue-Brain Maintenance Consolidation Pass v1

Status: **completed maintenance-facing consolidation pass** over authority, discoverability, shadow surfaces, audit baselines and Cargo maintenance noise. This note records the result only; it does not create new Blue-Brain functionality.

## 1) Files changed

- `Cargo.toml`
- `docs/README.md`
- `docs/blue_brain_authority_chain_status_map.md`
- `docs/blue_brain_audit_baseline_map_v1.md`
- `docs/blue_brain_sc1_prompt2_post_br6_repro_baseline_refresh_v1.md`
- `docs/blue_brain_sc1_prompt3_cross_line_terminology_guard_checklist_consolidation_v1.md`
- `docs/blue_brain_sc1_prompt4_final_system_consolidation_sweep_v1.md`
- `docs/blue_brain_inter_region_architecture_serie_ir1_prompt1_v1.md`
- `docs/blue_brain_first_inter_region_implementation_serie_ir1_prompt2_v1.md`
- `docs/blue_brain_inter_region_diagnostics_contracts_serie_ir1_prompt3_v1.md`
- `docs/blue_brain_first_inter_region_relation_line_serie_bb26_prompt4_v1.md`
- `docs/blue_brain_two_region_guard_contract_consistency_serie_bb26_prompt7_v1.md`
- `docs/blue_brain_maintenance_discoverability_map_v1.md`
- `docs/blue_brain_non_canonical_shadow_surface_inventory_v1.md`
- `docs/blue_brain_maintenance_consolidation_pass_v1.md`

## 2) Discoverability and authority fixes

- README and the authority map now point to the same current authority hierarchy and the same maintenance discoverability map.
- The allowed discoverability classes are explicit: `current operational authority`, `supporting current reference`, `historical snapshot`, `stale discoverability pointer`, and `non-canonical/internal-only shadow surface`.
- Early IR1 and BB26 two-region relation docs now carry a front-matter note that they are supporting/historical implementation-stage references where later BR6/IR1/MD2/MD3/SC1 authority wins.
- SC1 Prompt 2/3/4 now cross-reference the discoverability map, audit map and shadow inventory so their evidence/guard/decision role is clear without becoming a second authority source.

## 3) Baseline/reference consolidation

- `docs/blue_brain_audit_baseline_map_v1.md` now reflects the current 2026-05-08 post-BR6/IR1/MD2/MD3/SC1 evidence bundle.
- Older `out/blue_brain_audit_baseline_2026-05-02/` and `out/blue_brain_audit_baseline_2026-05-04/` references are explicitly historical comparison traces.
- The baseline map no longer describes the current system as a two-region state.

## 4) Shadow-surface consolidation

- `docs/blue_brain_non_canonical_shadow_surface_inventory_v1.md` compactly inventories additional DBM, microcircuit, biophys/neuro and adjacent-domain crates.
- These surfaces are classified as non-canonical/internal-only/deferred unless a current-authority document explicitly promotes them.
- The inventory explicitly rules out implicit seventh-region, IR1, model-platform, planner/agent/policy/retry or compute-core expansion from crate presence.

## 5) Cargo warning handling

- The root `Cargo.toml` no longer contains the unsupported virtual-workspace `[workspace.features]` table.
- The prior `warning: /workspace/UCF/Cargo.toml: unused manifest key: workspace.features` maintenance noise is therefore technically removed rather than merely documented as tolerated.
- No replacement build logic was introduced.

## 6) Remaining caveats

- Historical docs remain intentionally preserved and searchable; maintainers must read them through the authority/discoverability maps.
- Selection-mediated and execution-interface-mediated relation wording remains sensitive and must remain read/diagnostic-only.
- Shadow crates remain present in the workspace for traceability/internal support and require labeling discipline.
- Future expansion still requires an explicit re-scope; none is active after this pass.

## 7) Closure decision

After this consolidation, **maintenance without another consolidation block is credible** for the current Blue-Brain surface. The remaining work class is normal maintenance: bugfixes, cleanup, report refreshes, terminology hardening and test hardening within the existing authority envelope.
