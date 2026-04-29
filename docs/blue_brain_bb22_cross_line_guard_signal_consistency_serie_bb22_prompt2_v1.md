# Blue Brain — Serie BB22 Prompt 2: Cross-line Guard Consistency & Bounded Signal Classification Cleanup

Status: Schmaler BB22-Stabilisierungspass ohne neue Plattform. Ziel ist eine einheitliche Guard- und Signal-Semantik über Runtime, Selection, Execution, References und bounded advisory-only Dynamics.

## Kanonische Cross-line Signal-Klassen

1. `strong operational signal`
2. `bounded advisory-only signal`
3. `weak/reference-only signal`
4. `caveated signal`
5. `blocked/insufficient signal`
6. `non-canonical/internal-only signal path`

Diese Klassen sind absichtlich minimal und trennen starke operative Signale strikt von bounded/weak/blocked/non-canonical Pfaden.

## Einheitliche no-direct-* Guard-Bedeutung (repo-weit)

Bounded advisory-only Dynamics und schwache Referenzpfade behalten unverändert:
- kein direct action trigger,
- kein direct retry trigger,
- kein direct memory effect,
- kein direct compute effect,
- kein direct policy/planner/agent effect.

Zusätzlich bleibt `no_safety_override` unverändert bindend.

## Cross-line Konsistenzpunkte

- Runtime/Selection konsumieren dieselben Klassen für strong/bounded/weak/caveated/blocked/insufficient/non-canonical.
- Execution/Reference-Übergänge bleiben kompatibel mit derselben Klassifikation.
- Dynamics bleibt advisory-only bounded und kann keine direkte Autorität hochstufen.
- `blocked/insufficient` wird nicht als advisory Unterstützung umgedeutet.

## Ausgeschlossene non-canonical/internal-only Pfade

Folgende Pfade bleiben explizit außerhalb kanonischer operativer Nutzung:
- lose interne Signalpfade ohne kanonische Klassifikation,
- implizite Sonderfälle ohne Class-Mapping,
- interne-only Modulationspfade als operative Autoritätsquelle.

## BB22 Prompt-2 Abschlussrahmen

- Guard-Semantik wurde nur präzisiert, nicht erweitert.
- Signal-Klassen wurden nur konsolidiert, keine neue Meta-Plattform.
- Scope bleibt maintenance-first und narrow cross-line stabilization only.
