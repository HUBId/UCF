# Delta Spec: ONN/SNN Integration (T100)

## A) Summary

This document is the canonical delta spec for the ONN/SNN integration in UCF before switching from mocked AI compute to real model backends.

- **Goal**: _"Alles wirkt auf alles, zeitlich verzögert, probabilistisch adaptiv"_.
- **Delta introduced**: explicit ONN phase bus wiring, explicit SNN event bus wiring, and explicit feedback signals with bounded gains and cycle delays.

In practice, the architecture now treats temporal phase coherence (ONN), sparse event flow (SNN/SpikeBus), and structural governance (TCF/NSR/CDE/SLE) as one deterministic loop.

## B) Covered already (mapped to modules)

### Perception influences memory

- `ucf-ai-port` / `ucf-ai-runtime` / `domains/features/ucf-feature-spiker` provide feature and spike-facing inputs.
- `ucf-spike-encoder` + `ucf-spikebus` accept/aggregate spikes.
- `ucf-ssm` consumes these signals and updates memory state.

### Memory influences attention

- `ucf-ssm` emits `attention_gain` (plus novelty/salience context).
- ONN/TCF/IIT consume memory-derived attention context for next-cycle regulation.

### Attention influences learning

- Learning control is explicit through learning-relevant signals from:
  - `attention_gain`
  - `novelty`
  - `salience`
  - `surprise`
  - `coherence`/phi proxies
- These are consumed by `ucf-ncde`, `ucf-cde`, and related governance modules.

### Learning changes structure

- `ucf-cde` and `ucf-ncde` outputs feed structural regulation surfaces (directly or via router wiring).
- Structural adaptation is represented by bounded deltas (budget and threshold shifts, coupling hints, gating pressure) rather than unconstrained rewrites.

### Delays are explicit

- ONN lock behavior includes previous-cycle context (`lock_window` state carried between cycles).
- Budget/cooldown style delays exist in regulation modules (TCF/risk/policy paths), preventing immediate runaway gain escalation.

## C) ONN specifics

### Phase bus + PLV

- `ucf-onn` exposes a phase bus contract including:
  - `gamma_bucket`
  - `global_plv`
  - `phase_commit`
- This phase bus is consumed by coherence/safety modules (`ucf-iit`, `ucf-tcf`, `ucf-nsr`, `ucf-sle`) through router sequencing.

### Lock window auto-tune (TCF interaction)

- ONN lock behavior is not a free-running oscillator.
- TCF-mediated constraints and coherence pressure tune effective lock behavior over time, including lock-window pressure/capping behavior.

### IIT monitor coupling points

- ONN phase/coherence outputs are explicit inputs to IIT and IIT-monitor paths (`ucf-iit`, `ucf-iit-monitor`).
- IIT hints (`tighten_sync`, dampening hints, replay hints) feed back into timing/learning/output caps.

### Deterministic commit formulas (what is hashed)

ONN/SNN integration remains deterministic by hashing committed state surfaces (phase, memory, coherence, policy-relevant outputs) inside router/module commit flows.

At minimum, deterministic mode selection and replayability are anchored to commits from:

- ONN phase (`phase_commit`)
- SSM state commit/digest
- IIT-related commit surfaces
- Policy/NSR/risk surfaces used for overrides

## D) SNN specifics

### SpikeBus acceptance rules

Spike acceptance is explicit and bounded in `ucf-spikebus`:

- event admission depends on contract checks (shape/type/limits),
- accepted spikes are summarized via committed roots/counts,
- downstream modules consume accepted summaries instead of opaque mutable state.

### Time-to-first-spike bucket encoding (`gamma_bucket`)

- Temporal sparsity is represented in coarse buckets (`gamma_bucket`) for deterministic routing.
- This gives a stable timing abstraction usable by ONN/IIT/TCF without leaking raw, high-variance event traces into every module.

### ThoughtOnly channel + non-leak enforcement

- Thought-only internal emissions are explicit (`OutputChannel::Thought` / thought-only roots).
- Router and policy gates enforce non-leakage to outward channels by default.
- SNN-like internal event flow can still inform internal memory/attention updates without external disclosure.

## E) Cross-module feedback matrix

| Source module | ONN (`ucf-onn`) | SpikeBus (`ucf-spikebus`) | Memory (`ucf-ssm`) | Learning (`ucf-ncde`/`ucf-cde`) | Governance (`ucf-iit`/`ucf-tcf`/`ucf-nsr`/`ucf-sle`) |
|---|---|---|---|---|---|
| Perception (`ucf-ai-port`, feature spiker) | `gamma_bucket` context, range `[0,1]`, delay `0`, budgeted `N` | spike candidates/counts, bounded by bus limits, delay `0`, budgeted `Y` | percept energy/commit influence, bounded normalized inputs `[0,1]`, delay `0`, budgeted `Y` | novelty pressure precursor `[0,1]`, delay `1`, budgeted `Y` | policy-relevant trace context, bounded digest inputs, delay `1`, budgeted `Y` |
| ONN (`ucf-onn`) | lock-window self-tune signals `[0,1]`, delay `1`, budgeted `Y` | phase-aligned acceptance context `[0,1]`, delay `0`, budgeted `N` | phase/coherence modulation `[0,1]`, delay `0`, budgeted `Y` | timing/coherence priors `[0,1]`, delay `0`, budgeted `Y` | phase bus (`global_plv`, `phase_commit`) `[0,1]`, delay `0`, budgeted `N` |
| SpikeBus (`ucf-spikebus`) | spike timing pressure `[0,1]`, delay `0`, budgeted `N` | acceptance stats/root, bounded by configured limits, delay `0`, budgeted `Y` | spike-derived update strength `[0,1]`, delay `0`, budgeted `Y` | sparse-event novelty/salience cues `[0,1]`, delay `0`, budgeted `Y` | event-pressure hints `[0,1]`, delay `1`, budgeted `Y` |
| Memory (`ucf-ssm`) | `attention_gain` and memory priors `[0,1]`, delay `0`, budgeted `Y` | salience-based readout pressure `[0,1]`, delay `1`, budgeted `Y` | state commit + salience/novelty `[0,1]`, delay `1`, budgeted `Y` | learning signal tuple (`attention_gain`, novelty, salience, surprise, coherence) `[0,1]`, delay `0`, budgeted `Y` | coherence/risk context `[0,1]`, delay `0`, budgeted `Y` |
| Learning (`ucf-ncde`, `ucf-cde`) | coupling hints / structural pressure `[0,1]`, delay `1`, budgeted `Y` | acceptance policy pressure `[0,1]`, delay `1`, budgeted `Y` | structural delta pressure (threshold/budget hints) `[0,1]`, delay `1`, budgeted `Y` | DAG/causal commit surfaces, bounded scores `[0,1]`, delay `1`, budgeted `Y` | NSR/TCF-facing causal evidence `[0,1]`, delay `0`, budgeted `Y` |
| Governance (`ucf-iit`, `ucf-tcf`, `ucf-nsr`, `ucf-sle`) | sync tighten/damp hints `[0,1]`, delay `0`, budgeted `Y` | acceptance throttles/filters, bounded limits, delay `0`, budgeted `Y` | output and learning caps `[0,1]`, delay `0`, budgeted `Y` | cap multipliers and replay requests `[0,1]`, delay `0`, budgeted `Y` | verdicts/locks/risk state `[0,1]`, delay `1`, budgeted `Y` |

## F) Safety / runaway control

### GainBudget model

- Gain and influence are treated as budgeted quantities, not unlimited multipliers.
- TCF/risk/NSR surfaces cap output and learning channels under instability.

### Anti-runaway triggers

Typical triggers wired in the coherence loop:

- coherence drop / drift rise,
- policy or NSR restriction,
- surprise/coherence mismatch,
- repeated stabilization pressure.

Resulting actions include stabilize mode, damped output caps, damped learning caps, and replay-style requests.

### Coherence gate CI tests

The coherence gate asserts deterministic pipeline order, repeatable commits on fixed inputs, and thought-only non-leak behavior.

## G) What remains / not yet implemented

The ONN/SNN delta is architectural and contract-level complete, but these items are still pending for full "real compute" operation:

1. Real LFM/RLM hosts via Candle/Burn backends (`ucf-ai-runtime` currently mock-oriented).
2. SAE training pipelines and Lens extraction wiring (`ucf-sae-port`, `ucf-lens-port` currently port-level).
3. True JEPA world model behavior (current JEPA path remains simplified).
4. Production CDE discovery algorithms (current CDE is deterministic but still scaffolded vs full discovery stack).
5. RSA/OpenEvolve safe sandbox execution at full capability (`ucf-rsa`, `ucf-rsa-hooks`, `ucf-openevolve-port`, `ucf-sandbox` integration depth pending).
6. BlueBrain bridge + richer microcircuit interoperability (`ucf-bluebrain-port`, `ucf-digital-brain`, microcircuit crates).
7. STARK/Firewood archive-chain hardening (if retained as product direction) around `ucf-archive`, `ucf-archive-firewood`, `ucf-archive-store`.

## H) Non-goals of this delta

This T100 delta does **not** introduce:

- new solver dependencies,
- ML training workloads,
- hardware driver integration.

## Workspace path cross-check (for this document)

Referenced crate paths in this document were cross-checked against workspace members in root `Cargo.toml` before publishing this spec. All referenced paths exist.
