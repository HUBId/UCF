# Serie BB30 Prompt 2: First Anatomical Region Selection Line (genau eine frühe reale Region)

Status: **selection line established** als kontrollierte Fortsetzung von BB30-P1.

Diese Linie priorisiert **genau eine** frühe echte Hirnregion für die nächste anatomische Expansion.
Keine Mehrfachauswahl, keine globale Neurodynamikplattform, keine HH-Produktivintegration.

## 1) Early-viable Regionenvergleich (Hebel vs. Risiko)

Bewertet wurden die in BB30-P1 als `early viable` markierten Regionen:

| Region | UCF-Hebel jetzt | Integrationsnähe (BB2/4/8/12/17/19/21) | Cross-line Risiko | Modellschwere | Scope-Drift Risiko | Einordnung |
|---|---|---|---|---|---|---|
| `hippocampus_like_region` | hoch (Context/Reference-Hebel, bessere Hand-off-Kohärenz) | sehr hoch (BB8/BB17 + Region-2/3-Basis) | niedrig | niedrig bis mittel (abstract-first) | niedrig | **bestes Hebel/Risiko-Verhältnis** |
| `amygdala_like_region` | mittel bis hoch (Salienz/Caveat) | hoch (BB4/BB19 + BB12 advisory) | mittel (schnell übergewichtete Caveat-Signale) | mittel (bounded-dynamics-gekoppelt) | mittel | viable but not first |
| `thalamus_like_region` | mittel (Relay/Gating-Struktur) | mittel bis hoch (BB19 + BB21) | mittel bis höher (Execution-nahe Fehlinterpretation) | mittel | mittel bis höher | viable but not first |
| `basal_ganglia_like_region` | mittel (Go/No-Go-readiness diagnostics) | mittel (BB13/BB14/BB21) | höher (implizite Freigabeautorität droht) | mittel bis höher | höher | later-phase anatomical candidate |
| `prefrontal_executive_control_like_region` | hoch (Priorisierung/Fokus) | hoch (BB4/BB19) | höher (Planner-/Agenten-Shadow-Risiko) | mittel | höher | later-phase anatomical candidate |

Kurzfazit: **`hippocampus_like_region` ist jetzt zuerst technisch am belastbarsten**, weil sie die bestehende Drei-Regionen-Basis direkt ergänzt, ohne action-nahe Autoritätsrisiken zu öffnen.

## 2) Kanonische first-anatomical-region selection map

Für BB30-P2 gilt folgende minimal notwendige Auswahlkarte:

1. `first_anatomical_expansion_candidate`
2. `viable_but_not_first`
3. `later_phase_anatomical_candidate`
4. `simulation_only_or_deferred_anatomical_candidate`
5. `non_canonical_internal_only_path`

Zuordnung (genau eine First-Region):

| Region | Selection state |
|---|---|
| `hippocampus_like_region` | `first_anatomical_expansion_candidate` |
| `amygdala_like_region` | `viable_but_not_first` |
| `thalamus_like_region` | `viable_but_not_first` |
| `basal_ganglia_like_region` | `later_phase_anatomical_candidate` |
| `prefrontal_executive_control_like_region` | `later_phase_anatomical_candidate` |
| `cerebellum_like_region` | `simulation_only_or_deferred_anatomical_candidate` |
| `hypothalamus_like_region` | `simulation_only_or_deferred_anatomical_candidate` |

Nicht-kanonische/kompatibilitätsinterne Pfade bleiben ausschließlich:

- `non_canonical_internal_only_path`

## 3) Explizite Auswahlkriterien (technisch knapp)

Die Auswahl nutzt nur folgende Kriterien:

- maximaler funktionaler UCF-Hebel ohne neue Autorität,
- hohe Integrationsnähe zu Runtime/Selection/Context/Reference,
- Verträglichkeit mit BB12 bounded advisory-only Grenzen,
- geringe Action-/Retry-/Memory-Commit-/Compute-Autoritätsgefahr,
- überschaubare Implementationstiefe (abstract-first),
- Ergänzung der bestehenden Drei-Regionen-Basis statt Dublette.

## 4) Priorisierte Region (genau eine): `hippocampus_like_region`

`hippocampus_like_region` wird als **erste reale anatomische Expansion** priorisiert.

Warum jetzt zuerst:

- Sie nutzt die stabilsten bestehenden Linien (BB8/BB17 Context/Memory/Reference) direkt aus.
- Sie bleibt klar advisory-/strukturbezogen und vermeidet execution-nahe Fehlautorität.
- Sie stärkt die bereits etablierte Region-2/Region-3-Vertragsbasis, statt neue Plattformen zu eröffnen.
- Sie ist als **abstract-first** anschließbar; kein HH-first, kein globaler Dynamik-Sprung.

Warum schwerere Kandidaten nicht zuerst:

- `basal_ganglia_like_region` und `thalamus_like_region` liegen näher an Eligibility/Gating-Missdeutung.
- `prefrontal_executive_control_like_region` hat erhöhtes Planner-/Agenten-Shadow-Risiko.
- HH-lastigere Vertiefungen bleiben bewusst später und selektiv.

## 5) Bewusste Nachrangigkeit der übrigen Kandidaten

- **Viable but not first:** `amygdala_like_region`, `thalamus_like_region`.
- **Later phase:** `basal_ganglia_like_region`, `prefrontal_executive_control_like_region`.
- **Simulation-only/deferred:** `cerebellum_like_region`, `hypothalamus_like_region`.

`not first` bedeutet **nicht verworfen**, sondern kontrolliert nachgelagert.

## 6) Guard-/Scope-/Safety-Absicherung

Für die priorisierte erste Region bleibt verbindlich:

- keine direkte Action-/Retry-/Queue-/Orchestration-Autorität,
- keine direkte Memory-Commit- oder Compute-Core-Autorität,
- keine Safety-Override-Semantik,
- keine Reaktivierung deferred/non-canonical Pfade,
- keine Öffnung einer zweiten anatomischen Region innerhalb desselben Schritts.

## 7) Minimale Einhängungsrichtung (nächster Schritt, noch ohne Vollimplementierung)

Minimal andocken an:

- BB8/BB17 Context-/Memory-/Reference-Signale,
- BB19 Runtime-/Selection-Contract-Interpretation (advisory/deferred/caveated sauber getrennt),
- BB21 reference interaction boundary (rein referenzierend, nicht autorisierend).

Grundsätzlich zulässige I/O-Richtung:

- Input: bounded context/reference/runtime-diagnostic state,
- Output: advisory-only context-binding/reference-indexing hints.

Explizit nicht berühren:

- action execution authority,
- retry/orchestration semantics,
- policy/governance flows,
- compute-core behavior,
- globale HH/Kuramoto-Plattformausweitung.

## 8) Out-of-scope bleibt bewusst hart

Unverändert außerhalb von BB30-P2:

- Mehrfachauswahl anatomischer First-Regionen,
- Vollhirn-/Vollregionssimulation,
- globale Neurodynamikplattform,
- direkte produktive HH-Integration,
- neue Planner-/Agenten-/Policy-/Queue-Plattformen.

## 9) Ergebnislinie

Diese BB30-P2-Linie stellt belastbar her:

1. exakt eine priorisierte erste anatomische Region,
2. nachvollziehbare technische Abgrenzung gegen andere Kandidaten,
3. klare Scope-Kontrolle ohne implizite Zweitexpansion,
4. vorbereitete minimale Anschlussrichtung für den nächsten BB30-Schritt.
