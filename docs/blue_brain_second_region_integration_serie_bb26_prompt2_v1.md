# Serie BB26 Prompt 2: Second-Region Integration Line (minimal, bounded, repo-treu)

Status: **genau eine zweite Regionenklasse minimal integriert**.

Die in BB26 Prompt 1 priorisierte zweite Regionenklasse ist weiterhin:

- **Memory/Context-related** (`second_expansion_candidate`)

Diese Integration bleibt **abstract-functional** und **advisory-only**. Sie öffnet keine dritte Region, keine neue Compute-Core-Semantik und keine neue Autoritätskanäle.

## 1) Canonical second-region integration map

Verbindliche Integrationsklassen:

1. `region-2 input surface`
2. `region-2 state surface`
3. `region-2 output/advisory surface`
4. `region-2 reference surface`
5. `blocked/deferred region-2 path`
6. `non-canonical/internal-only region-2 path`

## 2) Region-2 input surface (minimal)

Region 2 darf ausschließlich lesen:

- runtime deferral lifecycle signal,
- canonical reference validity signal,
- context evidence priority signal.

Explizit unzulässig bleiben:

- direkte Tool-/Action-Steuersignale,
- compute-interne Rohzustände,
- direkte Safety-Override-Eingänge,
- implizite Memory-Mutationsinputs.

## 3) Region-2 state surface (bounded)

Region 2 trägt genau folgende bounded Zustände:

- `active_bounded_advisory_only`,
- `caveated_reference_state`,
- `deferred_or_blocked_state`,
- `non_canonical_internal_only`.

Damit bleibt Region 2 komplementär zu Region 1:

- Region 1: attention/selection advisory lane,
- Region 2: context/reference quality + caveat lane.

## 4) Region-2 output/advisory surface (bounded)

Region 2 darf nur advisory-only Signale erzeugen:

- caveat hint,
- deferral hint,
- reference-bounded signal,
- blocked/deferred marker,
- non-canonical/internal-only marker.

Outputs informieren Runtime/Selection/Reference nur bounded. Explizit verboten bleiben:

- direct action selection,
- direct execution trigger,
- direct retry trigger,
- direct memory commit,
- direct compute invocation,
- safety override.

## 5) Andockung an Runtime/Selection/Reference

Region 2 ist nur über bestehende Linien angebunden:

- Runtime sieht advisory-only Hinweise (keine direkte Ausführung),
- Selection sieht advisory-only Hinweise (keine direkte Selektionserzwingung),
- Reference/Context liefert bounded Inputs (keine Persistenzautorität).

Es entsteht keine zweite operative Wirklichkeit und keine breite inter-region platform.

## 6) Dynamics-Einordnung

Für diese Region-2-Öffnung ist **kein** zusätzlicher bounded-dynamics-Anschluss erforderlich.

- Kuramoto/HH bleiben unverändert in ihren bestehenden Rollen,
- keine HH-Produktivintegration,
- region-2 integration bleibt abstract-functional.

## 7) Guard-/Scope-/Safety-Grenzen

Unverändert hart:

- keine direkte Action-/Retry-/Memory-/Compute-Autorität,
- keine Safety-Override-Semantik,
- keine implizite Reaktivierung deferred/non-canonical Pfade,
- keine dritte Regionenklasse,
- keine implizite inter-region platform.
