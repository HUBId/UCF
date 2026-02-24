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
