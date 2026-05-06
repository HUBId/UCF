# Serie BR3 Prompt 3: Thalamus Surface/Diagnostics/Contracts härten

Status: `thalamus_like_region` bleibt die dritte echte anatomische Region in UCF-BlueBrain, jetzt mit gehärteter Surface-/Diagnostics-/Contract-Line. Die Linie ist absichtlich schmal: relay/gating/routing-nahe, bounded, advisory-only und ohne direkte operative Autorität.

## 1) Kanonische thalamus diagnostics/contract map

Die kanonische Map besteht aus genau diesen Klassen:

1. `thalamus advisory-only diagnostic`
2. `thalamus caveated diagnostic`
3. `thalamus deferred diagnostic`
4. `thalamus blocked diagnostic`
5. `thalamus insufficient diagnostic`
6. `thalamus diagnostic-only state`
7. `thalamus bounded contract signal`
8. `non-canonical/internal-only thalamus path`

Diese Map erweitert keine Meta-Plattform. Sie beschreibt nur, wie die bestehende thalamus input surface, thalamus state surface, thalamus output/advisory surface und thalamus reference surface repo-konsistent gelesen werden.

## 2) Surface- und Read-Semantik

Die kanonischen Surfaces bleiben unterscheidbar:

- `thalamus input surface`: bounded Runtime-relay, Selection-gating, Routing-/Deferral-, Context-/Reference- und Reference-validity Inputs.
- `thalamus state surface`: advisory-only, caveated, deferred, blocked, insufficient, diagnostic-only oder non-canonical/internal-only Zustände.
- `thalamus output/advisory surface`: relay-, routing-, gating-, caveat-, reference-bounded-, blocked/deferred- und insufficient-diagnostic Outputs.
- `thalamus reference surface`: read-only/reference-bounded Kontext ohne zweite Referenzwirklichkeit.

Runtime, Selection, Routing und Reference lesen dieselbe kanonische Thalamus-Semantik über denselben bounded contract read. Es gibt keine layer-spezifische Umdeutung desselben thalamischen Zustands.

## 3) Advisory-only vs caveated

`advisory-only != caveated`.

- `thalamus advisory-only diagnostic` ist ein bounded positives Signal. Es kann Runtime/Selection/Routing als relay/gating/routing Hinweis dienen, aber bleibt advisory-only.
- `thalamus caveated diagnostic` ist kein starkes positives Signal. Es entsteht aus caveated Reference-, Selection- oder Routing-Basis oder aus partiellem thalamischem Signal.
- Caveated darf nicht zu advisory-only eskalieren und darf keine direkte operative Autorität erzeugen.

## 4) Deferred vs blocked vs insufficient

`deferred != blocked` und `blocked != insufficient`.

- `thalamus deferred diagnostic` bedeutet bounded Aufschub oder Zurückstellung, etwa wegen stale/deferred Routing-/Reference-Basis.
- `thalamus blocked diagnostic` bedeutet begrenzender Contract-, Safety- oder Reference-Zustand; er bleibt ein Block-/Diagnostic-Read und startet keine Ausführung.
- `thalamus insufficient diagnostic` bedeutet keine tragfähige bounded Basis. Das ist nicht caveated, nicht deferred und nicht blocked.

Diese drei Zustände werden getrennt gehalten, damit Runtime/Selection/Reference keine implizite Retry-, Block- oder Positivdeutung ableiten.

## 5) Diagnostic-only und bounded contract signal

- `thalamus diagnostic-only state` beschreibt reference-only/read-only Thalamus-Sichtbarkeit. Sie ist diagnostisch nützlich, aber nicht ausführungswirksam.
- `thalamus bounded contract signal` ist eine lesbare Contract-Bindung zwischen Runtime, Selection, Routing und Reference. Sie ist kein Action-Kanal und kein Memory-/Compute-Kanal.
- `non-canonical/internal-only thalamus path` bleibt sichtbar, aber operativ unzulässig.

## 6) No-direct-* Grenzen

Ein thalamus contract signal ist ausdrücklich:

- no action request
- no execution trigger
- no retry trigger
- no memory commit
- no compute trigger
- no safety override

Zusätzlich bleibt verboten: Tool-/Action-Steuerung, Retry-Orchestrierung, automatische Memory-Persistenz, Planner-/Agentenlogik, Policy-/Governance-Plattformen, allowed-actions-Erweiterungen und neue Compute-Core-Arbeit.

## 7) Modellgrenze

`current model mode remains unchanged`: Der Thalamus bleibt im `abstract functional current mode`.

Explizit getrennt bleiben:

- bounded Kuramoto-like bleibt späterer Kandidat, nicht Produktivmodus.
- Hodgkin-Huxley simulation-only/diagnostic-only remains deferred.
- HH-later/selective deepening braucht eine explizite Re-Entscheidung.
- Deferred Modellpfade sind keine implizite Runtime- oder Compute-Öffnung.

Damit entsteht keine Modell-Drift und keine globale Neurodynamikplattform.

## 8) Abgrenzung gegen Hippocampus und Amygdala

- hippocampus remains context/reference/episode/indexing-lastig.
- amygdala remains salience/valence/caveat/priority-lastig.
- thalamus remains relay/gating/routing-lastig.

Spätere bounded Kopplung darf diese Rollen ergänzen, aber nicht gleichsetzen. Der Thalamus ist keine semantische Dublette zu Hippocampus oder Amygdala.

## 9) BR3 nächste Schritte

1. Thalamus diagnostics in weitere readiness-/guard-nahe Reports spiegeln, ohne neue operative Autorität.
2. Golden/reference checks für advisory-only/caveated/deferred/blocked/insufficient ausbauen.
3. Runtime-/Selection-/Reference-Snapshots gegen denselben thalamischen canonical read stabilisieren.
4. Non-canonical/internal-only Restpfade weiter auditieren und nur diagnostisch sichtbar halten.
5. Erst nach stabiler Contract-Line einen separaten Re-Scope für bounded Kuramoto-like Kopplung prüfen.
