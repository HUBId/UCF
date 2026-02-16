# World Model (JEPA Predictor Adapter v0)

`ucf-compute` nutzt ein deterministisches, offline-fähiges JEPA-Mock-Dynamics-Modell als `WorldModelPredictor`.

## Contract

- `WorldModelInput`
  - `t: u64`
  - `context_digest: [u8;32]`
  - `obs_features: [f32;16]` (deterministisch aus `context_digest`)
  - `seed: u64`
- `WorldModelOutput` (bounded)
  - `prediction_digest: [u8;32]`
  - `state_digest: [u8;32]`
  - `prediction_error: f32` (0..1)
  - `surprise: f32` (0..1)
  - `state_norm: f32` (0..1)
  - `quality: StageQuality`
  - `notes: Vec<String>` (bounded)

Keine Roh-Payloads (weder Pred-Vektor noch State-Vektor) werden persistiert.

## Mock dynamics v0

Fixtures aus `runtime/ucf-compute/fixtures/jepa_dyn_v1.json` werden per `include_bytes!` geladen.

Dynamik:
- `pred = A*state + B*obs + c`
- `new_state = lerp(state, pred, alpha=0.22)` mit Clamp `[-1,1]`
- `prediction_error = mean_abs(pred - obs)`
- `surprise = clamp(prediction_error, 0..1)`

Digests sind kanonisch über Float-Bits + `(seed, t, fixture_digest)`.

## Budget semantics

- WorkUnits skalieren mit `K^2`.
- Budget exceeded:
  - `DegradeStages`: `WorldModelOutput::degraded_budget` mit `surprise=1`, `prediction_error=1`, zero-digests, `quality=degraded_fallback`.
  - `FailFast`: `ComputeSignals::unavailable`.

## Integration

`ComputePipelineBackend` setzt:
- `ComputeSignals.surprise <- world_output.surprise`
- `EvidenceRef.world_digest <- prediction_digest`
- Telemetry:
  - `ucf_world_prediction_error`
  - `ucf_world_surprise`
  - `ucf_world_degraded_total`

## Upgrade-Pfad zu VL-JEPA

Der Adapter hält den Vertrag stabil (`WorldModelInput/Output`, digests, budget semantics), sodass später nur die interne Prädiktor-Implementierung (z. B. VL-JEPA weights + encoder) ersetzt werden muss.
