# Serie BB31 Prompt 2: first-anatomical docs/tests/index cleanup (maintenance-facing reference surface)

Status: Die erste anatomische Region (`hippocampus_like_region`) bleibt die **einzige** geöffnete anatomische Region. Diese Referenz schärft ausschließlich Doku-/Test-/Index-Pfade für den Maintenance-Betrieb und öffnet keine neue Capability-Linie.

## Canonical first-anatomical maintenance reference map

1. canonical anatomical-region reference doc
2. canonical anatomical-region test surface
3. maintenance-facing index/reference path
4. non-canonical/internal-only or legacy anatomical-region path

### 1) Canonical anatomical-region reference doc

- `docs/blue_brain_first_anatomical_stabilization_line_serie_bb31_prompt1_v1.md`
- Diese Referenz bleibt die kanonische Stabilisierungskarte der first-anatomical Linie.
- Surface, diagnostics, contract, model boundary und no-direct-* Guard-Semantik werden dort als maintenance-hardened fixiert.

### 2) Canonical anatomical-region test surface

- `runtime/ucf-compute/src/blue_brain_region_first_integration.rs` (gezielte first-anatomical guards/docs/tests).
- Die Testfläche hält explizit sichtbar:
  - advisory-only input/output guard boundaries,
  - diagnostics state separation,
  - contract signal semantics,
  - current model mode (`abstract functional current mode`),
  - no direct action/execution/retry/memory/compute/safety override,
  - keine implizite Öffnung einer zweiten anatomischen Region.

### 3) Maintenance-facing index/reference path

- `docs/README.md` und `docs/roadmap/REPO_MAP.md` führen die BB31-Linie als schnellen operativen Einstieg.
- Canonical order innerhalb BB31:
  1. Prompt 1 Stabilization Line,
  2. Prompt 2 Docs/Tests/Index Cleanup.

### 4) Non-canonical/internal-only or legacy anatomical-region path

- Nicht-kanonische/internal-only/legacy Formulierungen bleiben explizit residual und nicht-operativ.
- Sie dürfen keine zweite Wahrheitsquelle gegenüber der kanonischen BB31-Linie aufbauen.

## Guard-/Freeze-/Maintenance-Hinweise (unverändert bindend)

- BB23 Freeze-/Maintenance-Baseline bleibt aktiv.
- no-direct-* Grenzen bleiben hart:
  - no direct action trigger
  - no direct execution trigger
  - no direct retry trigger
  - no direct memory commit
  - no direct compute invocation
  - no safety override
- Weitere anatomische Regionen sind nicht geöffnet; dafür ist ein expliziter Re-Scope nötig.

## Out of scope (explizit)

- keine zweite anatomische Region,
- keine neue Modellplattform,
- keine Planner-/Agenten-/Policy-/Retry-Orchestrierungsplattform,
- keine Compute-Core-Ausweitung,
- keine neue Dokumentationsplattform.
