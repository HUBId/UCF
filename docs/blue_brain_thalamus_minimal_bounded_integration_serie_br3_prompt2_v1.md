# Serie BR3 Prompt 2: Thalamus minimal und bounded in UCF einhängen

Status: `thalamus_like_region` ist als dritte echte anatomische Region minimal integriert. Die Integration bleibt eine kontrollierte UCF-/BlueBrain-Surface für relay/gating/routing, nicht ein biologischer Vollnachbau.

## 1) Kanonische thalamus integration map

Die kanonische Integrationskarte ist bewusst klein:

1. `thalamus input surface`
2. `thalamus state surface`
3. `thalamus output/advisory surface`
4. `thalamus reference surface`
5. `blocked/deferred thalamus path`
6. `non-canonical/internal-only thalamus path`

Diese Map ist die einzige neue thalamische Wahrheitsfläche für BR3 Prompt 2. Sie erzeugt keine Meta-Plattform und keine neue Compute-Core-Arbeit.

## 2) Thalamus input surface

Zulässige bounded Inputs:

- Runtime-relay Signale als lesbare, bounded Basis.
- Selection-gating Signale als selection-support context.
- Routing-/Deferral-Signale aus bestehenden Deferral-Semantiken.
- Context-/Reference-Signale als bounded reference basis.
- Reference-validity Signale für current/caveated/stale/blocked/insufficient/reference-only Entscheidungen.

Explizit verboten:

- direkte Tool-/Action-Steuersignale
- compute-interne Rohzustände
- direkte Safety-Override-Eingänge
- implizite Memory-Mutationsinputs

## 3) Thalamus state surface

Der Thalamus darf nur bounded Zustände tragen:

- `active bounded relay advisory-only`
- `caveated reference routing state`
- `deferred routing state`
- `blocked routing state`
- `reference-only routing state`
- `non-canonical/internal-only`

Er darf keine Action-, Execution-, Retry-, Memory-, Compute- oder Safety-Zustände besitzen.

## 4) Thalamus output/advisory surface

Zulässige thalamische Outputs bleiben advisory-only:

- `relay-hint`
- `routing-hint`
- `gating-hint`
- `caveat-hint`
- `reference-bounded signal`
- `blocked/deferred` diagnostic output
- `non-canonical/internal-only` diagnostic output

Runtime, Selection und Routing dürfen diese Signale nur bounded konsumieren. Reference erhält nur reference-bounded/read-only Kontext. Es entsteht keine direkte Proposal-, Action- oder Execution-Autorität.

Explizit verboten:

- direct action selection
- direct execution trigger
- direct retry trigger
- direct memory commit
- direct compute invocation
- safety override

## 5) Runtime/Selection/Routing bounded Anschluss

Runtime sieht den Thalamus ausschließlich als advisory relay/routing surface. Selection sieht ihn ausschließlich als gating-/selection-support Hinweisgeber. Routing darf nur bounded durch relay-, gating-, routing-, caveat- oder reference-bounded hints beeinflusst werden.

Blocked/deferred Fälle bleiben Contract-/Diagnostic-Zustände und starten keine Retry-Orchestrierung. Non-canonical/internal-only Fälle bleiben sichtbar, aber nicht operativ.

## 6) Reference/Context bounded Anschluss

Kanonische thalamusbezogene Referenzen sind nur:

- context/reference basis für relay/routing hints,
- reference validity für current/caveated/stale/blocked/insufficient/reference-only Einordnung,
- stale/caveated/reference-only Kennzeichnung ohne Autoritätserhöhung.

Es gibt keine zweite Referenzwirklichkeit und keine implizite Memory-Persistenz. Reference-only bleibt diagnostic/read-only und darf keine Action-, Execution- oder Commit-Folge auslösen.

## 7) Modellgrenze

Der aktuelle Thalamus-Modus bleibt der in BR3 Prompt 1 festgelegte Modus:

- `abstract functional current mode`

Explizit nicht geöffnet:

- keine Kuramoto-Produktivaufweitung
- keine Hodgkin-Huxley-Produktivintegration
- keine globale Neurodynamikplattform
- keine Planner-/Agenten-/Policy-/Governance-/Retry-/Orchestration-Plattform

Spätere Modellvertiefung muss separat entschieden werden.

## 8) Abgrenzung zu Hippocampus und Amygdala

- `hippocampus_like_region` bleibt context/reference/episode/indexing-lastig.
- `amygdala_like_region` bleibt salience/valence/caveat/priority-lastig.
- `thalamus_like_region` bleibt relay/gating/routing-lastig.

Die drei Regionen teilen bounded Contract-/Diagnostic-Mechanik, aber keine direkte Autorität und keine semantische Gleichsetzung.

## 9) Guard-/Out-of-scope-Grenzen

Die thalamische Integration hält ausdrücklich:

- no direct action trigger
- no direct execution trigger
- no direct retry trigger
- no direct memory commit
- no direct compute invocation
- no safety override
- keine implizite Öffnung weiterer anatomischer Regionen
- keine allowed-actions-Erweiterung

## 10) BR3-Nächste Schritte

1. Thalamus-Diagnostics gegen Runtime-/Selection-Snapshots härten.
2. Routing-/Deferral-Caveats mit thalamusbezogenen golden checks stabilisieren.
3. Inter-region consistency checks für Hippocampus/Amygdala/Thalamus weiter schärfen.
4. Reference-only/stale thalamus Fälle in readiness diagnostics sichtbar machen.
5. Optionalen Re-Scope für bounded Kuramoto-like Thalamus-Kopplung erst nach Stabilisierung separat entscheiden.
