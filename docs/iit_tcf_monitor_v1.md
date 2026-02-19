# IIT/TCF Monitor v1 (Operational Proxy, No Mysticism)

This document defines the **operational** (not metaphysical) coherence monitor used by runtime v1.

## Scope
- Deterministic, bounded per-tick computation.
- Fixed-point (`UQ0_16`) coherence and incoherence outputs.
- Tightening-only behavior: incoherence can only make gating/governance stricter.

## Metrics

`iitmap_coherence_q` is computed from three bounded proxies and aggregated via `min(...)`:

1. **Signal agreement coherence (A)**
   - Inputs: `risk_q`, `pressure_q`, `surprise_q`, `uncertainty_q`.
   - Computes normalized dispersion via mean absolute deviation (MAD).
   - Coherence falls as dispersion rises.

2. **Phase consistency (B)**
   - Inputs: phase-window alignments from ONN/SNN scheduler hooks.
   - Uses average alignment across phase windows.
   - If phase windows unavailable, fallback stays bounded.

3. **Stability contradiction trend (C)**
   - Penalizes coherence when risk is high while stability is low.

Then:
- `coherence_q = min(A, B, C)`
- `incoherence_q = 1 - coherence_q`

## Outputs

Per tick, monitor emits:
- `contract_version`
- `backend_id = "IitMonitorV1"`
- `coherence_q`, `incoherence_q`
- up to 4 reason codes
- `monitor_digest = H(coherence||incoherence||reasons||t||signals_digest)`

## TCF phase-window hook behavior

If `incoherence_q > TH`:
- allow: `LFM`, `NSR`, `EBM constraints`, `governor`
- deny: `LLM generation`, `tool planning`

A phase-window note is persisted in ESS each tick with allow-mask and reason.

## Governor tightening

When `incoherence_q > INC_HIGH` an extra deterministic penalty is applied and effective tier is tightened (never loosened by incoherence).

## Explain-tick visibility

`ComputeSignalsSummary` now includes:
- `iit_coherence_q`
- `iit_incoherence_q`
- `iit_reason_codes`
- `stage_allow_mask`
- `coherence_digest` bound to monitor digest

This keeps observability scalar-only (no raw timeseries persistence).
