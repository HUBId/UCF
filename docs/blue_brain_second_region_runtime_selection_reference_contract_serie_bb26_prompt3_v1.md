# Serie BB26 Prompt 3: Region-2 Runtime-/Selection-/Reference-Contract (gehärtet, bounded)

Status: **kanonische second-region contract line** für die bereits in BB26 Prompt 2 geöffnete Region 2 (*Memory/Context-related*), ohne Scope-Erweiterung.

## 1) Canonical second-region contract map

Die Region-2-Schnitt bleibt auf genau diese bounded Signalklassen limitiert:

1. `region-2-to-runtime advisory signal`
2. `runtime-to-region-2 bounded input`
3. `region-2-to-selection advisory signal`
4. `selection-to-region-2 bounded state input`
5. `region-2-reference signal`
6. `caveated/deferred/blocked region-2 contract signal`
7. `reference-only region-2 contract signal`
8. `non-canonical/internal-only region-2 contract path`

Keine weitere inter-region Meta-Plattform wird eingeführt.

## 2) Runtime-semantik (Region 2)

Runtime darf Region 2 nur als bounded advisory-only lesen:
- `RegionToRuntimeAdvisory` für normale bounded Hinweise,
- `Caveated` bei caveated/insufficient Referenzlage,
- `Deferred` bei Deferral,
- `Blocked` bei Blocked,
- `NonCanonicalInternalOnly` als nicht-operativer Pfad.

Explizit ausgeschlossen bleiben: direkte Action-/Execution-/Retry-/Memory-/Compute-Wirkung und Safety-Override.

## 3) Selection-semantik (Region 2)

Selection liest Region 2 ebenfalls nur advisory-only und bounded:
- `RegionToSelectionAdvisory` für kontext-/referenzbezogene Ergänzung,
- `Caveated` und `Deferred` als Prioritäts-/Vorsichtshinweise,
- `Blocked` als blocked-state Hinweis (nicht failed execution),
- `NonCanonicalInternalOnly` außerhalb des kanonischen Pfads.

Region 2 erzeugt keine direkte Action-Selection- oder Proposal-Autorität.

## 4) Reference-/Context-semantik (Region 2)

Kanonische Reference-Signale:
- `RegionReferenceSignal` für current/reference-bounded Hinweise,
- `ReferenceOnly` für reference-only (explizit ohne operative Stützwirkung),
- `Caveated` bei eingeschränkter Referenzqualität,
- `NonCanonicalInternalOnly` für interne nicht-kanonische Pfade.

Aus `reference-only` oder `caveated` entsteht kein impliziter Memory-Commit.

## 5) Deferred/Blocked/Caveated/Reference-only Trennung

Region-2-Zustände bleiben separat:
- `DeferredState` ≠ `BlockedState`
- `BlockedState` ≠ failed execution
- `CaveatedReferenceState` ≠ starker Region-2-Support
- `ReferenceOnlyState` ≠ operative Support-Basis

## 6) Ergänzung zu Region 1

- Region 1 bleibt attention/selection-zentriert.
- Region 2 ergänzt reference/context-quality und caveat/reference-only Lesbarkeit.
- Keine Dublette und keine implizite dritte Regionenklasse.

## 7) Scope-Grenzen (no-direct-*)

Die bestehenden Guard Rails bleiben unverändert:
- kein direct action trigger,
- kein direct execution trigger,
- kein direct retry trigger,
- kein direct memory commit,
- kein direct compute invocation,
- kein safety override,
- keine implizite dritte Regionenklasse,
- keine breite inter-region platform.

## 8) Bounded dynamics

Für diesen Schritt bleibt Region 2 **ungekoppelt** zu zusätzlicher bounded-dynamics-Steuerung; advisory-only bleibt strikt getrennt von jeder direkten Steuerungswirkung.
