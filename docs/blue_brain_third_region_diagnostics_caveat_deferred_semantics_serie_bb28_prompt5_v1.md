# BlueBrain Serie BB28 – Prompt 5: Region-3 Diagnostics/Caveat/Deferred Semantik

Status: Dieser Schritt härtet die **regionsspezifische Diagnostic-Semantik für Region 3**. Die Linie bleibt strikt bounded/advisory-only und erweitert **keine** Runtime-Autorität.

## Kanonische Region-3-Diagnostic-States

Die kanonische Region-3-Diagnostic-Map umfasst:

- `region3_advisory_only_diagnostic`
- `region3_caveated_diagnostic`
- `region3_deferred_diagnostic`
- `region3_blocked_diagnostic`
- `region3_insufficient_diagnostic`
- `region3_diagnostic_only_state`
- `caveated_inter_region_diagnostic_influence`
- `non_canonical_internal_only_region3_diagnostic_path`

## Abgrenzungen

- **advisory-only**: bounded positives Signal ohne direkte Autorität.
- **caveated**: schwaches/partielles Signal; kein stark positives Signal.
- **deferred**: bounded Zurückstellung, nicht failed execution.
- **blocked**: contract-/safety-/reference-bedingte Begrenzung, nicht nur niedrige Priorität.
- **insufficient**: keine tragfähige bounded Basis.
- **diagnostic-only**: sichtbar, aber nicht operative advisory support basis.

## Runtime / Selection / Reference Konsistenz

Die Region-3-Diagnostic-State-Ableitung erfolgt signalbasiert konsistent über dieselbe Klassifikation:

- runtime-nahe contract signals,
- selection-nahe contract signals,
- reference-nahe contract signals.

Damit wird verhindert, dass dieselbe Region-3-Lage in Runtime/Selection/Reference unterschiedlich interpretiert wird.

## Bounded Inter-Region Relation (Region 3 ↔ Region 1/2)

Inter-regionale Wirkung bleibt diagnostisch bounded:

- relationale Caveats können `caveated_inter_region_diagnostic_influence` erzeugen,
- blocked/deferred relationale Zustände bleiben begrenzt diagnostisch,
- shared-reference mediated bleibt diagnostic-only,
- keine direkte Region-zu-Region Autorität.

## No-direct-* Guards (unverändert strikt)

Region-3-Diagnostics aktivieren nicht:

- keinen direct action trigger,
- keinen direct execution trigger,
- keinen direct retry trigger,
- keinen direct memory commit,
- keine direct compute invocation,
- keinen safety override.

Damit bleibt BB28 Prompt 5 eine gehärtete Diagnostics-Linie ohne Plattformausweitung.
