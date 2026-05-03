# Blue Brain Three-Region Docs/Tests/Index Cleanup (Serie BB29 Prompt 2)

Status: maintenance-facing cleanup für die **stabilisierte Drei-Regionen-Basis** (Region 1, Region 2, Region 3 + bounded Relations). Keine vierte Region, keine Plattformausweitung.

## 1) Canonical three-region maintenance reference map

Kanonische Kategorien für die operative Referenzfläche:

1. `canonical three-region reference doc`
2. `canonical region-1 test surface`
3. `canonical region-2 test surface`
4. `canonical region-3 test surface`
5. `canonical bounded relation test surfaces`
6. `maintenance-facing index/reference path`
7. `non-canonical/internal-only or legacy three-region path`

## 2) Canonical reference paths (maintenance-facing)

- canonical three-region reference doc:
  - `docs/blue_brain_three_region_maintenance_stabilization_line_serie_bb29_prompt1_v1.md`
- canonical region-1 test surface:
  - `runtime/ucf-compute/src/blue_brain_region_first_integration.rs` (region-1 tests)
- canonical region-2 test surface:
  - `runtime/ucf-compute/src/blue_brain_region_first_integration.rs` (region-2 tests)
- canonical region-3 test surface:
  - `runtime/ucf-compute/src/blue_brain_region_first_integration.rs` (region-3 tests)
- canonical bounded relation test surfaces:
  - `runtime/ucf-compute/src/blue_brain_region_first_integration.rs` (region-1↔2, region-1↔3, region-2↔3)
- maintenance-facing index/reference path:
  - `docs/README.md`
  - `docs/roadmap/REPO_MAP.md`

## 3) Maintenance guard and boundary visibility (must stay explicit)

- Drei-Regionen-Basis bleibt maintenance-hardened und bounded.
- `no direct action trigger`
- `no direct execution trigger`
- `no direct retry trigger`
- `no direct memory commit`
- `no direct compute invocation`
- `no safety override`
- keine implizite Plattformbildung über Regionenrelationen.
- Region 4 ist **nicht offen** und benötigt expliziten Re-Scope.

## 4) Legacy and non-canonical handling

- Nicht-kanonische, interne oder historisch überholte Verweise bleiben `non-canonical/internal-only or legacy three-region path`.
- Sie haben keine operative Default-Autorität.
- Keine zweite Wahrheitsquelle neben den oben genannten kanonischen Pfaden.
