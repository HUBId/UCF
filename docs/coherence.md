# Coherence v0 (SNN/ONN/TCF/IIT Monitor)

This repository now includes a deterministic **coherence runtime layer** in `runtime/ucf-runtime`.

## Scope (v0)

The implementation is intentionally bounded and replayable:

- No neuromorphic hardware.
- No real IIT calculation.
- No tool calls or side effects from coherence modules.
- Deterministic routing, scheduling, and metric outputs for the same input sequence.

## Components

## 1) Spike Bus (event routing)

Implemented in `runtime/ucf-runtime/src/coherence.rs`.

- Converts compute spikes into `SpikeEvent`.
- Uses deterministic ordering and bounded capacity.
- Subscriber interest can be:
  - `TopKFeatures(Vec<u32>)` (bounded)
  - `HashBuckets(Vec<u8>)` (bounded)
- Produces `SpikeRoutingSummary` with digest.

## 2) ONN-like phase windows

Also in `coherence.rs`:

- Each subscriber module has a deterministic `PhaseState` seeded from module ID.
- Per tick update computes alignment against a reference phase.
- Window opens when alignment exceeds threshold.

## 3) TCF-like scheduler

In `coherence.rs`:

- Deterministic score combines alignment, pressure, and coherence penalties.
- Selects up to bounded `K` modules each tick.
- Emits bounded reason codes.

## 4) IIT monitor (phi-proxy)

In `coherence.rs`:

- Maintains rolling bounded history.
- Computes:
  - `coherence` in `[0,1]`
  - `instability` in `[0,1]`
  - `phi_proxy` in `[0,1]`
- Emits canonical digest.

## Policy/FEP gating integration

`RuntimeOrchestrator` now executes coherence tick after compute signals and before final action decision.

- `ComputeSignalsSummary` now includes optional coherence fields:
  - `coherence`
  - `instability`
  - `phi_proxy`
  - `coherence_digest`
- `DecisionFrame` includes optional `gating_reason`.
- If coherence gating trips (`coherence_low` or `instability_high`) and decision is `Allow`, decision is deterministically converted to `Defer` with reason `coherence_gate`.

## How to add a new subscriber

In `RuntimeOrchestrator::new`:

1. Register a `Subscriber` into `coherence_runtime`.
2. Set deterministic `module_id`, stable `name`, and bounded `InterestProfile`.
3. Keep subscriber count under global cap.

Example pattern:

```rust
runtime.register_subscriber(Subscriber {
    module_id: 9,
    name: "my_module",
    interest: InterestProfile::HashBuckets(vec![0, 3, 7]),
});
```

## Determinism and bounds checklist

- Max subscribers: fixed cap.
- Max spikes per tick: fixed cap.
- Max events per subscriber batch: fixed cap.
- Max selected modules per tick: fixed cap.
- Fixed-size rolling monitor window.
- Stable ordering and tie-breakers.
- Coherence outputs persisted through compute summary + decision metadata.
