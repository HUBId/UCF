# EBM Reasoner

## v1 Architektur

`CandleEbmReasonerV1` nutzt ein kleines MLP (`ebm.w1`, `ebm.b1`, `ebm.w2`, `ebm.b2`) mit begrenzten Shapes:
- `D <= 64`
- `H <= 32`
- CPU-only, offline, hash-locked via `ModelStore`/`WeightSpec`

Forward (deterministisch):
1. i16-Features nach f32: `x = q / 32767.0`
2. `h = tanh(W1*x + b1)`
3. `e = sigmoid(W2*h + b2)`
4. `energy_q = UQ0_16::from_f32_clamped(e)`

Bei Fehlern/Budget/degenerierten Outputs wird `DegradedFallback` genutzt, ohne Governance zu umgehen.

## Feature-Encoding

Kanonischer Vektor:
- Signale: risk/confidence/pressure/surprise/uncertainty/coherence
- Kandidatenmetadaten: kind, tool_class
- Quantisierte Cost-Features aus Candidate (`compute_units`, `bytes_out`, `tool_calls`)

## Bounded Search

Optional über `UCF_EBM_SEARCH=1`:
- diskrete Varianten (max 4 pro Kandidat)
- keine neuen Fähigkeiten
- deterministische Generierung + Sortierung
- Schritte begrenzt (`<=16`) und in ESS recordet

## Rollout/Fallback

- `UCF_SLOT_EBM_MODE=off|shadow|compare|active`
- `UCF_EBM_CANDLE=1` aktiviert Candle-Reasoner
- Timeout/Budget/NaN/Degenerate => Fallback + envelope violation record

Governance/Tool-Gate bleibt final maßgeblich; EBM beeinflusst nur Kandidaten-Rerank.


## FEP/Governor Coupling v1

- EBM liefert ein deterministisches `EbmSignal` mit `energy_min_q`, `energy_mean_topk_q`, `energy_dispersion_q` und `ebm_digest_prefix`.
- FEP nutzt `energy_mean_topk_q` als zusätzlichen Free-Energy-Proxy-Term:
  - `free_energy_proxy_q = clamp(base_free_energy_q + w_ebm_q * energy_mean_topk_q, 0..1)`
  - fixed-point only auf dem Policy-Pfad.
- Governor nutzt EBM **nur tightening-only**:
  - Bei `energy_mean_topk_q > E_HIGH` wird eine additive Penalty berechnet.
  - Unterhalb des Schwellwerts: keine Wirkung.
  - EBM kann Governor niemals permissiver machen.
- Bei `emergency_active` wird EBM im Governor ignoriert und kann im Reasoning als `suppressed_by_emergency=true` markiert werden.


## Constraint Library v1

- EBM addiert harte Safety-Constraints als deterministische Energy-Terme (`E_total = clamp(E_model + Σterm, 0..1)`).
- Term-Definitionen kommen aus `policies/bundle_v1/ebm_constraints.toml` und sind über den Policy-Bundle-Hash gesichert.
- Bei Lade-/Validierungsfehlern greift ein konservativer Fallback (`ToolIntentPenalty`) und wird telemetriert.
- Explain-Tick zeigt `base_energy_q` plus Top-Term-Contributions (id + Label + Beitrag).
