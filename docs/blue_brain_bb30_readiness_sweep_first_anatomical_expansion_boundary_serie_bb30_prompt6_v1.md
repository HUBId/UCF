# Serie BB30 Prompt 6: BB30-readiness sweep und anatomische Expansionsgrenze

Status: BB30 wird als **harte Abschlusslinie für genau eine erste anatomische Region** konsolidiert. Die operativ aktive Region bleibt `hippocampus_like_region` im Modus `abstract functional current mode`.

## 1) BB30-expansion-readiness map (kanonisch)

| Bereich | Zustand | Festlegung |
|---|---|---|
| First anatomical input surface | **stable first-anatomical operational surface** | Nur bounded Runtime/Selection/Reference-konforme Inputs. |
| First anatomical state surface | **stable first-anatomical operational surface** | Region-interne Zustände bleiben deterministisch und bounded. |
| First anatomical output/advisory surface | **advisory-only** | Keine direkte Action-/Execution-/Retry-/Memory-/Compute-Autorität. |
| First anatomical reference surface | **stable first-anatomical operational surface** | Referenzen bleiben kanonisch, bounded, auditierbar. |
| Diagnostics states | **usable with caveats** | advisory-only/caveated/deferred/blocked/insufficient/diagnostic-only/reference-only bleiben getrennt. |
| Contract signals | **usable with caveats** | Signal-Semantik bleibt explizit und no-direct-* gebunden. |
| Current model mode | **stable current model mode** | `abstract functional current mode` ist festgezogen. |
| Bounded dynamics coupling | **advisory-only** | Nur informierend; keine direkte Steuerautorität. |
| Deferred/blocked/insufficient/diagnostic-only/reference-only lanes | **deferred/blocked/insufficient/diagnostic-only/reference-only** | Nicht-promotet, nicht autoritativ, guard-sichtbar. |
| Non-canonical/internal-only lanes | **non-canonical/internal-only** | Explizit nicht Teil der kanonischen first-anatomical expansion line. |

## 2) First-anatomical expansion line (explizit)

Die erste echte anatomische Regionsexpansion ist ausschließlich:

- `hippocampus_like_region`

Kanonische Surface- und Zustandsgrenze:

- Input: anatomical region input surface (bounded, no direct control signals).
- State: anatomical region state surface (deterministic, bounded).
- Output: anatomical region output/advisory surface (strictly advisory-only).
- Reference: anatomical region reference surface (canonical, bounded, no authority escalation).
- Diagnostics: advisory-only/caveated/deferred/blocked/insufficient/diagnostic-only/reference-only.
- Contract: runtime/selection/reference lesen dieselbe regionsspezifische Diagnostik-Semantik.
- Model: `abstract functional current mode`.

## 3) Finale no-direct-* und out-of-scope Grenzen

Explizit **nicht operativ** in BB30 Prompt 6:

- zweite anatomische Region
- direkte Action-Steuerung
- direkte Execution-Auslösung
- Retry-Orchestrierung
- Planner-/Policy-/Agentenlogik
- automatische Memory-Mutation/Persistenz
- Safety-Override-Semantik
- Compute-Core-Ausweitung
- HH-Produktivintegration
- globale Modellplattform

No-direct-* bleibt bindend:

- no direct action trigger
- no direct execution trigger
- no direct retry trigger
- no direct memory commit
- no direct compute invocation
- no safety override

## 4) BlueBrain-Linien- und Compute-Abschlusskonsistenz

Diese Abschlusslinie hält explizit fest:

- BB2 Runtime/Transition/Feedback bleibt bounded und nicht-autoritativ überschrieben.
- BB4 Selection/Priority/Deferral bleibt getrennt; Priority bleibt advisory-only.
- BB8/BB17 Context/Memory/Reference hardening bleibt intakt.
- BB12 bounded dynamics bleibt advisory-only.
- BB19 Runtime/Selection contract line bleibt unvermischt.
- BB21 Execution/Reference interaction bleibt ohne direkte Anatomie-Autorität.
- Compute bleibt finalisierte maintenance-only Core-Linie mit bestehenden outward-facing contracts.

## 5) Entscheidung: zweite anatomische Region vs Stabilisierung

Repo-basierte Entscheidung für den nächsten Schritt:

- **Priorität: Stabilisierungspass der ersten anatomischen Region**.
- **Nicht priorisiert jetzt: zweite anatomische Region**.

Technischer Grund:

1. Die erste anatomische Region ist kanonisch operativ, aber bewusst bounded/advisory-only.
2. Der größte Hebel liegt in Stabilität der bestehenden Surface-/Diagnostics-/Contract-/Model-Trennungen.
3. Frühe zweite Regionenexpansion würde ohne zusätzlichen operativen Nutzen Grenzverwischung und Maintenance-Risiko erhöhen.
4. HH-lastigere oder schwerere Regionen bleiben deferred, bis die erste anatomische Linie längerfristig stabil bleibt.
