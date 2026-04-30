# Serie BB24 Prompt 6: First-Region Runtime/Selection/Reference Contract Line (gehärtet)

Status: **erste regionsspezifische Runtime-/Selection-/Reference-Schnitt explizit definiert und gehärtet**.

Diese Linie erweitert die in BB24 Prompt 5 minimal integrierte Regionenklasse
`Attention/Selection-related` kontrolliert auf eine belastbare Contract-Semantik,
ohne neue Autoritätskanäle oder Mehrfach-Regionen-Ausbau.

## 1) Kanonische First-Region Contract Map

Die kanonische Schnitt ist in
`runtime/ucf-compute/src/blue_brain_region_first_integration.rs`
über `CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP` explizit als bounded Klassen geführt:

- `RegionToRuntimeAdvisorySignal`
- `RuntimeToRegionBoundedInput`
- `RegionToSelectionAdvisorySignal`
- `SelectionToRegionBoundedStateInput`
- `RegionReferenceSignal`
- `CaveatedDeferredBlockedRegionContractSignal`
- `ReferenceOnlyRegionContractSignal`
- plus die BB24-P5 Basisflächen (`RegionInput/State/Output/ReferenceSurface`, blocked/deferred,
  non-canonical/internal-only)

Damit bleibt die Region nur über kanonische Contract-Pfade sichtbar.

## 2) Runtime-Semantik (bounded + advisory-only)

Runtime darf nur bounded advisory Signale lesen.
`BlueBrainFirstRegionOutputSurface` bleibt explizit mit no-direct-* Rails:

- `direct_action_selection = false`
- `direct_execution_trigger = false`
- `direct_retry_trigger = false`
- `direct_memory_commit = false`
- `direct_compute_invocation = false`
- `safety_override = false`

Zusätzlich ist `contract_signal` explizit typisiert, damit Runtime `caveated/deferred/blocked/reference-only`
getrennt lesen kann statt impliziter String-/Interpretationslogik.

## 3) Selection-Semantik (hinweisgebunden, nicht autoritativ)

Selection liest dieselbe Surface ebenfalls advisory-only. `contract_signal` trennt:

- `Deferred` (verschoben, nicht blockiert),
- `Blocked` (nicht ausführbar im aktuellen Pfad, aber kein failed execution result),
- `Caveated` (schwächere Basis, kein starker Signalpfad),
- `ReferenceOnly` (referenzierbar, aber nicht operative support basis).

Es entsteht **keine** direkte Proposal-/Action-Autorität.

## 4) Reference-/Context-Semantik

Reference-gebundene Fälle sind kanonisch über `RegionReferenceSignal` und
`ReferenceOnlyRegionContractSignal` sichtbar.

`reference_only` wird explizit auf dem Output markiert und bleibt getrennt von
aktueller/starker Basis. Daraus entsteht keine implizite Memory-Persistenz.

## 5) Deferred-/Blocked-/Caveated-Grenzen

Die Contract-Linie schärft explizit:

- `deferred != blocked`
- `blocked != failed execution`
- `caveated != strong signal`
- `reference-only != operative support basis`

## 6) Dynamics-Kopplung

Für diese erste Regionenklasse bleibt die Dynamics-Kopplung unverändert minimal:
keine zusätzliche produktive Kuramoto-/HH-Steuerintegration.

## 7) Scope-/Guard-Grenzen

Verbindlich erhalten:

- kein direct action trigger,
- kein direct execution trigger,
- kein direct retry trigger,
- kein direct memory commit,
- kein direct compute invocation,
- kein safety override,
- keine implizite zweite Regionenexpansion.

