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
