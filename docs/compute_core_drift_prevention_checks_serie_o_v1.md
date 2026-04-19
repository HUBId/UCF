# Serie O: Compute Core Drift-Prevention Checks v1

Stand: Repo-Zustand am 2026-04-19.

Zweck: eine **schmale, load-bearing Drift-Prevention-Schicht** für den abgeschlossenen Compute-Kern, ohne CI-/Governance-/Policy-Plattform und ohne zweite Wahrheit.

Kanonische Referenzlinie bleibt unverändert:
- `submit -> compute_canonical -> result/fault/status -> execution_snapshot`

Code source of truth:
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_DRIFT_PREVENTION_CHECK_MAP`
  - `CANONICAL_FINAL_REFERENCE_LINE`
- `runtime/ucf-compute/src/contracts.rs`
  - `CROSS_CUTTING_PRODUCTION_INVARIANTS_V1`
- `runtime/ucf-compute/src/service_surface.rs`
  - `CanonicalComputeEntryPoint::status_evidence_export_surface`
  - `CanonicalComputeEntryPoint::integration_hook_view`

## 1) Kanonische minimale Check-Map

Genau vier Check-Klassen gelten als kanonisch:

1. `reference_line_consistency`
   - schützt die finale Referenzlinie gegen schleichende Text-/Code-Entkopplung.
2. `outward_facing_contract_consistency`
   - schützt outward-facing Status-/Evidence-/Hook-Semantik vor Drift in interne/Expert-Semantik.
3. `shared_core_semantics_consistency`
   - hält die shared-core Semantik stabil: `blocked/failed/no_op` sowie `current/partial/stale/caveated/degraded`.
4. `doc_code_alignment`
   - bindet Serie-O-Doku explizit an dieselbe Abschlusslinie und verhindert zweite Wahrheitsquellen.

## 2) Welche Drift-Risiken damit früh sichtbar werden

- **Final-reference drift**: wenn Kernlinie oder Extensions in Doku und code-pinned Konstanten auseinanderlaufen.
- **Outward/inward drift**: wenn outward-facing Hooks internale/expert-only Semantik übernehmen.
- **Semantic collapse drift**: wenn `blocked`, `failed`, `no_op` oder freshness/caveat/degraded Klassen vermischt werden.
- **Maintenance-boundary drift**: wenn Serie-O-Grenzen aus der finalen Referenzlinie herausdriften.

## 3) Bewusst enge Grenzen

Diese Schicht macht **keine** neue Plattform auf:
- keine CI-/Governance-/Policy-Plattform,
- keine große Lint-Suite,
- keine Testreform,
- keine neue Integrations- oder Capability-Welle.

Sie hält nur die load-bearing Linie stabil und macht kleine Divergenzen früh sichtbar.
