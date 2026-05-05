# Serie BR2 Prompt 3: Amygdala Surface/Diagnostics/Contracts härten

Status: Die Amygdala-Schnitt ist als **maintenance-hardened advisory/diagnostic contract line** konsolidiert. Keine direkte operative Autorität.

## Kanonische Amygdala-Surface (bounded)
- amygdala input surface
- amygdala state surface
- amygdala output/advisory surface
- amygdala reference surface

Diese Surface bleibt strikt bounded auf Runtime/Selection/Reference-Konsum und ohne eigene Action-/Execution-Autorität.

## Kanonische diagnostics/contract map
- amygdala advisory-only diagnostic
- amygdala caveated diagnostic
- amygdala deferred diagnostic
- amygdala blocked diagnostic
- amygdala insufficient diagnostic
- amygdala diagnostic-only state
- amygdala bounded contract signal
- non-canonical/internal-only amygdala path

## Semantik-Härtung
- advisory-only != caveated
- deferred != blocked
- blocked != insufficient
- diagnostic-only bleibt diagnostic-only und nicht operativer Trigger

Caveated bleibt bewusst schwächer als advisory-only und kann aus schwacher/partieller Selection-/Priority-/Reference-Basis entstehen.
Deferred ist bounded Aufschub. Blocked ist Contract-/Safety-/Reference-limitierender Zustand. Insufficient markiert fehlende tragfähige bounded Basis.

## Runtime/Selection/Reference Konsistenz
Alle drei Schichten lesen dieselbe amygdaläre Semantik über die gleiche Contract-Signal-Klassifizierung; keine layer-spezifische Uminterpretation derselben Zustände.

## No-direct-* Guard (weiterhin explizit)
Amygdala contract signal ist:
- no action request
- no execution trigger
- no retry trigger
- no memory commit
- no compute trigger
- no safety override

## Modellgrenze
- current model mode remains unchanged
- abstract/Kuramoto-like/HH simulation-only/HH-later/deferred bleiben getrennt
- spätere Vertiefung bleibt explizit re-scope-pflichtig

## Abgrenzung Hippocampus vs Amygdala
- hippocampus remains context/reference/episode/indexing
- amygdala bleibt salience/valence/caveat/priority-lastig
- keine semantische Dublette und keine implizite Gleichsetzung
