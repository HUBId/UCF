# World VL-JEPA Scaffold v1.1

`ModelSlot::WorldVljepa` adds an adapter-first world-model slot that is **shadow-only**.

## Encoding contract

`WorldInputEncodingV1` is bounded and deterministic:

- fixed dimension `D=64` (<=128)
- scalar signals mapped to `[-1,1]`
- only digest/prefix-derived bytes mapped to float bridge features
- no raw payloads

## Shadow-only behavior

Enable with:

```bash
UCF_SLOT_WORLD_VLJEPA_MODE=shadow
```

In this mode, outputs are only emitted as shadow telemetry/notes and do not change decision path.

## WeightSpec skeleton

`ModelSlot::WorldVljepa` uses strict tensor requirements:

- `vljepa.w1` `[D,H]`
- `vljepa.b1` `[H]`
- `vljepa.w2` `[H,D]`
- `vljepa.b2` `[D]`

This keeps contract shape stable while allowing later promotion to a real VL-JEPA backend through lifecycle tooling.


Fixture descriptor: `runtime/ucf-compute/fixtures/world_vljepa_mlp_small.fixture.json` (text-only scaffold fixture).

## Shadow Envelope Metrics v1.1

Window-based shadow telemetry (`UCF_WORLD_VLJEPA_WINDOW_TICKS`, default `512`) now tracks:
- `latency_ms` (mean + p95)
- `prediction_error_q` (mean + p95)
- `error_delta_q = max(vljepa_error_q - baseline_error_q, 0)` vs JEPA stub baseline
- `invalid_rate` and `saturation_rate`

Drift watch evaluates each window with policy thresholds:
- `UCF_WORLD_VLJEPA_LAT_P95_MAX_MS`
- `UCF_WORLD_VLJEPA_ERR_MEAN_MAX_Q`
- `UCF_WORLD_VLJEPA_ERR_SPIKE_MAX_Q`
- `UCF_WORLD_VLJEPA_INVALID_OUTPUT_MAX_RATE`

On sustained/severe drift, shadow is tightening-only and can auto-disable (`DisabledShadow`) while staying decision-neutral.

## Shadow report for promotion evidence

Generate bounded shadow evidence artifact:

```bash
ucf-ops world shadow-report --run <run_id> --windows 10 --out ./out/world_shadow_report.json
```

The report includes window summaries, drift alarms, and model/manifest digests and is consumed by `models promote` for `world_vljepa` gating.


## Promotion gate linkage (v1.1)

When WorldVljepa is active (or recently promoted), readiness gate requires shadow evidence:
- `out/world_shadow_report.json` must be PASS.
- promotion provenance must reference a shadow report digest prefix.
- drift alarm rate must stay under configured threshold for a minimum window count.

If WorldVljepa runs shadow-only, gate allows PASS with sufficient windows and no severe alarms; otherwise FAIL with remediation guidance.
