# Serie BB25 Prompt 2: Region-1 Docs/Tests/Index Cleanup und Maintenance-Referenzfläche

Status: **kanonische maintenance-facing Referenzfläche für die erste und einzige geöffnete Regionenklasse**.

Diese Referenz konsolidiert BB24 Prompt 5–10 und BB25 Prompt 1 zu einer klaren Region-1
Maintenance-Surface. Es wird **keine zweite Regionenklasse** geöffnet, keine neue Runtime-Autorität
gebaut und kein Scope über advisory-/diagnostic-only hinaus erweitert.

## 1) Canonical Region-1 Maintenance Reference Map

### A) Canonical region-1 reference doc
- `docs/blue_brain_first_region_finalization_serie_bb24_prompt10_v1.md`
- `docs/blue_brain_first_region_stabilization_serie_bb25_prompt1_v1.md`
- `docs/blue_brain_region1_maintenance_reference_surface_serie_bb25_prompt2_v1.md` (diese Datei)

### B) Canonical region-1 test surface
- `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`
  - Region-1 Maps (`*_MAP`), Contract-Signale, Guard-/State-Semantik und Drift-Checks.

### C) Maintenance-facing index/reference path
- `docs/README.md` (operativer Einstieg)
- `docs/blue_brain_bb23_freeze_maintenance_baseline_serie_bb23_prompt1_v1.md`
- `docs/blue_brain_bb23_maintenance_guard_rails_allowed_change_envelope_serie_bb23_prompt2_v1.md`

### D) Non-canonical/internal-only or legacy region-1 path
- `NonCanonicalInternalOnlyRegionPath` und `NonCanonicalInternalOnlyResidualPath` in
  `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`
- historische BB24-Aufbaupfade (Prompt 5–9) bleiben Traceability, aber nicht zweite operative Wahrheit.

## 2) Operative Region-1 Semantik (maintenance-facing)

Region 1 bleibt explizit:
- bounded/advisory-only,
- diagnostics- und contract-gebunden,
- durch input guards auf kanonische Quellen begrenzt,
- gegen direkte Autoritätseskalation geschlossen.

Kanonische Vertrags-/Diagnostikzustände bleiben sichtbar:
- `advisory`, `caveated`, `deferred`, `blocked`, `insufficient`, `diagnostic-only`,
- `non-canonical/internal-only` als explizit ausgeschlossener Residualpfad.

## 3) Guard-/Freeze-Hinweise (bleiben bindend)

Unverändert in Scope-Grenzen:
- no-direct-action,
- no-direct-execution,
- no-direct-retry,
- no-direct-memory,
- no-direct-compute,
- no-safety-override,
- **region-2-not-opened (explicit re-scope required)**.

## 4) Cleanup-Regel für Region-1-Doku

Für Region-1-bezogene Doku gilt:
1. Keine zweite Wahrheitsquelle neben den kanonischen Referenzpfaden in Abschnitt 1.
2. Keine aspirativen Aussagen über weitere Regionen oder Autoritätsausbau.
3. Historische Prompt-Dokus bleiben zulässig, aber als Aufbau-/Traceability-Fläche, nicht als neue operative Baseline.
4. Maintenance-/Freeze-Kontext aus BB23 bleibt übergeordnet verbindlich.

## 5) Targeted Drift-Checks für diese Referenzfläche

Targeted checks für Region-1-Doku-/Testkonsistenz:
- `first_region_stabilization_map_contains_required_classes`
- `first_region_runtime_selection_reference_contract_reads_are_consistent`
- `first_region_rejects_non_canonical_input_sources`
- `first_region_non_canonical_contract_signal_is_explicitly_non_canonical`
- `region1_maintenance_reference_doc_pins_canonical_maps_and_boundaries`

Damit bleibt Region 1 schnell auffindbar, operativ eindeutig und maintenance-fähig – ohne implizite Öffnung einer zweiten Regionenklasse.
