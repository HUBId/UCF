# SAE / Feature Extractor v0

`ucf-compute` enthält einen deterministischen SAE-v0-Extraktor (`ToySaeExtractor`) mit folgenden Regeln:

- Offline-Fixture: `runtime/ucf-compute/fixtures/sae_proj_v1.json`
- Projektion: `y = W*x + b`
- Aktivierung: `a = max(0, y)`
- Sparsify: stabiles Top-K mit Ordnung `(-activation, feature_id)`
- Spike-Ausgabe: `Spike { feature_id, magnitude in [0,1], timestamp }`
- Boundaries: `F=128`, `D=32`, `K=32`, `max_spikes=32`

## Determinismus

- Keine Netz- oder Modell-Downloads.
- Keine RNG für v0-Projektion.
- Kanonischer `spikes_digest` aus sortierten Spikes + fixture digest + `(t, seed, context_digest_prefix)`.

## Degrade / FailFast

Bei Budgetüberschreitung im SAE-Stage:

- `DegradeStages`: leeres Spike-Set, `spike_count=0`, `sparsity=1`, `energy=0`, `quality=DegradedFallback`.
- `FailFast`: `ComputeError::BudgetExceeded`.

## Persistenz

Persistiert werden nur:

- `spike_count`
- `sparsity`
- `energy`
- `spikes_digest`
- `quality`

Einzelspikes sind runtime-only und werden nicht für Persistence benötigt.

## Erweiterungspfad

Später kann `ToySaeExtractor` durch echte SAE-Backends ersetzt werden, solange der Contract (`SaeInput`/`SaeOutput`) und die deterministischen Kanonikalisierungsregeln erhalten bleiben.
