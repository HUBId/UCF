# Serie BB30 Prompt 5: first anatomical region diagnostics/contract semantics

Status: Die erste anatomische Region (`hippocampus_like_region`) bleibt **bounded, advisory-only und maintenance-kompatibel**; diese Stufe härtet ausschließlich die diagnostics-/contract-nahe Semantik.

## Canonical first-anatomical-region diagnostics map

Kanonische Zustände:

1. `anatomical region advisory-only diagnostic`
2. `anatomical region caveated diagnostic`
3. `anatomical region deferred diagnostic`
4. `anatomical region blocked diagnostic`
5. `anatomical region insufficient diagnostic`
6. `anatomical region diagnostic-only state`
7. `non-canonical/internal-only anatomical region diagnostic path`

Diese Map ist absichtlich schmal und regionsspezifisch; es wird keine neue inter-region Meta-Plattform geöffnet.

## Contract-Semantik (advisory-only vs caveated)

- **advisory-only diagnostic** bleibt ein begrenztes positives Signal ohne direkte Autorität.
- **caveated diagnostic** bleibt abgeschwächt und wird nicht zu advisory-only hochgestuft.
- Schwache oder partielle Reference-/Context-Basis bleibt caveated/insufficient statt implizit positiv.

## Contract-Semantik (deferred vs blocked)

- **deferred diagnostic** = bounded Aufschub/Zurückstellung.
- **blocked diagnostic** = begrenzender Contract-/Safety-/Reference-Zustand.
- Deferred ist nicht failed execution; blocked ist nicht bloß niedrige Priority.

## Insufficient und diagnostic-only

- **insufficient diagnostic** = keine tragfähige bounded Basis.
- **diagnostic-only state** = sichtbar für Diagnose/Referenz, aber keine operative advisory support basis.
- Beide bleiben ohne direkte Execution-/Memory-/Compute-Autorität.

## Runtime/Selection/Reference Konsistenz

Runtime/selection/reference lesen dieselbe regionsspezifische Diagnostik via gemeinsame Canonical-Mapping-Funktion; es gibt keine getrennten Semantik-Dialekte pro Schicht.

## Modellmodus-Einbettung und bounded dynamics

Der aktuelle Modus bleibt `abstract functional current mode` (Prompt 4). Deshalb wird hier **keine** zusätzliche bounded-dynamics-Steuerkopplung eingeführt und **keine** HH-Produktivintegration geöffnet.

## No-direct-* Guardrails (weiterhin bindend)

- no direct action trigger
- no direct execution trigger
- no direct retry trigger
- no direct memory commit
- no direct compute invocation
- no safety override

Zusätzlich bleibt die Erweiterung auf weitere anatomische Regionen explizit außerhalb dieses Schritts.

Canonical consumer alignment: runtime/selection/reference read the same diagnostics map.
