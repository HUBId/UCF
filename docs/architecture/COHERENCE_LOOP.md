# Dynamic Coherence Loop (Canonical)

## 0. Related delta specification
For ONN/SNN-specific architecture deltas and remaining implementation work, see `docs/architecture/DELTA_ONN_SNN.md`.

## 1. Purpose & scope (UCF core loop, timing backbone)
This document codifies the canonical dynamic coherence loop for UCF. It defines the deterministic timing backbone that governs how perception, memory, attention, learning, and structural adaptation interact per cycle. The intent is to make the loop auditable and replayable by tying every stage to explicit commitments and update modes rather than emergent or stochastic behavior.【F:core/crates/ucf-router/src/lib.rs†L700-L840】【F:core/crates/ucf-router/src/lib.rs†L3081-L3173】

## 2. Deterministic “probabilistic-adaptive” definition (hash-driven mode selection)
“Probabilistic-adaptive” is defined deterministically: the system selects an update mode by hashing stable commitments (phase, SSM, IIT) and mapping the result to a fixed set of modes. This is pseudo-random in distribution but fully reproducible for the same commitments.

- **Mode seed** = `H("ucf.router.update.mode.v1", phase_commit, ssm_commit, iit_commit)` → `raw % 4` → `UpdateMode` (Conservative, Normal, Exploratory, Stabilize).【F:core/crates/ucf-router/src/lib.rs†L5128-L5170】
- **Override rules** are deterministic: NSR verdict, risk, drift, surprise, and phi thresholds can force Conservative/Exploratory/Stabilize regardless of base seed.【F:core/crates/ucf-router/src/lib.rs†L3081-L3119】

This makes “Alles wirkt auf alles (time-lagged, probabilistic-adaptive)” deterministic: all cross-module effects flow through committed signals and time-lagged buffers, and any adaptive mode shift is a hash-derived function of prior commits plus explicit safety overrides.【F:core/crates/ucf-router/src/lib.rs†L117-L140】【F:core/crates/ucf-router/src/lib.rs†L3081-L3173】

## 3. Cycle pipeline order (authoritative list)
The router executes cycle stages in the **exact order listed in the cycle plan** (`CyclePlan.pulses`). The canonical plan used by the orchestrator test is:

1. **Think**
2. **Sense**
3. **Verify**
4. **Consolidate**
5. **Broadcast**

This order is authoritative for documentation; the router iterates pulses in-order and emits the stage trace in that order.【F:core/crates/ucf-router/tests/wire_path_e2e.rs†L72-L142】【F:core/crates/ucf-router/tests/wire_path_e2e.rs†L290-L307】【F:core/crates/ucf-router/src/lib.rs†L742-L781】

### 3.1 Module order inside the coherence loop (System-1 path)
Within the Verify→Consolidate/Broadcast flow, the coherence loop is wired in the following order:

`ONN → SpikeBus → (Coupling) → JEPA → IIT → TCF → NSR → SLE → NCDE → SSM → CDE → Output → Archive`【F:core/crates/ucf-router/src/lib.rs†L798-L1510】【F:core/crates/ucf-router/src/lib.rs†L1730-L2085】【F:core/crates/ucf-router/src/lib.rs†L3608-L3703】

## 4. Signals and their directions
The loop relies on explicit signals and commits (all hashed) with clear directionality:

- **Perception signals** → Memory/Attention
  - `percept_energy` (control frame energy proxy) → SSM inputs.【F:core/crates/ucf-router/src/lib.rs†L796-L840】【F:core/crates/ucf-ssm/src/lib.rs†L120-L208】
  - `percept_commit` (control frame commitment) → SSM inputs.【F:core/crates/ucf-router/src/lib.rs†L796-L840】【F:core/crates/ucf-ssm/src/lib.rs†L120-L208】

- **Temporal signals** → Coherence/TCF/NSR
  - `gamma_bucket`, `global_plv`, `phase_commit` from ONN PhaseBus → IIT/TCF/NSR/SLE inputs.【F:core/crates/ucf-onn/src/lib.rs†L170-L210】【F:core/crates/ucf-iit/src/lib.rs†L108-L178】【F:core/crates/ucf-tcf/src/lib.rs†L120-L210】【F:core/crates/ucf-nsr/src/lib.rs†L128-L210】【F:core/crates/ucf-sle/src/lib.rs†L110-L190】

- **Sparse events** → Memory/Learning
  - SpikeBus `accepted_root` + `counts` → SSM/NCDE/CDE inputs.【F:core/crates/ucf-spikebus/src/lib.rs†L218-L278】【F:core/crates/ucf-ssm/src/lib.rs†L120-L208】【F:core/crates/ucf-ncde/src/lib.rs†L120-L210】【F:core/crates/ucf-cde/src/lib.rs†L170-L260】

- **Memory signals** → Attention/Learning
  - `ssm_state_digest/commit`, `salience`, `novelty`, `attention_gain` → ONN/NCDE/IIT/SLE/CDE inputs.【F:core/crates/ucf-ssm/src/lib.rs†L170-L238】【F:core/crates/ucf-onn/src/lib.rs†L118-L176】【F:core/crates/ucf-ncde/src/lib.rs†L120-L210】【F:core/crates/ucf-iit/src/lib.rs†L108-L178】【F:core/crates/ucf-sle/src/lib.rs†L110-L190】【F:core/crates/ucf-cde/src/lib.rs†L170-L260】

- **World-model signals** → Causal loop
  - `jepa.world_state` → CDE hypothesis input (observation commit).【F:core/crates/ucf-jepa/src/lib.rs†L1-L210】【F:core/crates/ucf-router/src/lib.rs†L1244-L1285】
  - `jepa.surprise` → ONN/TCF/SLE/SSM surprise inputs.【F:core/crates/ucf-jepa/src/lib.rs†L1-L210】【F:core/crates/ucf-router/src/lib.rs†L798-L2105】

- **Structural signals** → Learning/Policy gating
  - `nsr_verdict` + `nsr_trace_root` → TCF/SLE/ONN inputs.【F:core/crates/ucf-nsr/src/notar.rs†L160-L210】【F:core/crates/ucf-tcf/src/lib.rs†L120-L210】【F:core/crates/ucf-sle/src/lib.rs†L110-L190】【F:core/crates/ucf-onn/src/lib.rs†L118-L176】
  - `cde_commit` + coupling roots → ONN/IIT/TCF/NCDE inputs.【F:core/crates/ucf-cde/src/lib.rs†L260-L320】【F:core/crates/ucf-onn/src/lib.rs†L118-L176】【F:core/crates/ucf-iit/src/lib.rs†L108-L178】【F:core/crates/ucf-tcf/src/lib.rs†L120-L210】【F:core/crates/ucf-ncde/src/lib.rs†L120-L210】

- **Coherence signals** → Regulation
  - `phi_proxy` + IIT hints (`tighten_sync`, `damp_output`, `damp_learning`, `request_replay`) → TCF inputs and downstream caps.【F:core/crates/ucf-iit/src/lib.rs†L170-L246】【F:core/crates/ucf-tcf/src/lib.rs†L120-L210】

- **Self loop** → Thought-only internal feedback
  - `reflection/self_symbol` + `thought_only_root` from SLE → Workspace tracking and internal thought-only event gating.【F:core/crates/ucf-sle/src/lib.rs†L210-L330】【F:core/crates/ucf-workspace/src/lib.rs†L1720-L2004】

## 5. The five-link loop invariants (must statements)
1. **Perception → Memory**: Every cycle **must** map `percept_commit` + `percept_energy` into the SSM input commit, making memory state update traceable to perception input.【F:core/crates/ucf-ssm/src/lib.rs†L120-L208】【F:core/crates/ucf-ssm/src/lib.rs†L547-L591】
2. **Memory → Attention**: SSM outputs (`salience`, `novelty`, `attention_gain`, `ssm_state_commit`) **must** feed ONN/NCDE/IIT inputs, ensuring attention adapts to memory state.【F:core/crates/ucf-ssm/src/lib.rs†L170-L238】【F:core/crates/ucf-onn/src/lib.rs†L118-L176】【F:core/crates/ucf-ncde/src/lib.rs†L120-L210】【F:core/crates/ucf-iit/src/lib.rs†L108-L178】
3. **Attention → Learning**: TCF caps and ONN timing **must** gate learning inputs (NCDE/SSM/CDE) through committed attention/learning caps and phase locks.【F:core/crates/ucf-tcf/src/lib.rs†L340-L420】【F:core/crates/ucf-onn/src/lib.rs†L170-L214】【F:core/crates/ucf-ncde/src/lib.rs†L120-L210】【F:core/crates/ucf-ssm/src/lib.rs†L120-L208】【F:core/crates/ucf-cde/src/lib.rs†L170-L260】
4. **Learning → Structure**: CDE outputs (DAG commit + interventions) **must** summarize causal adjustments and surface them as committed structural signals for subsequent cycles.【F:core/crates/ucf-cde/src/lib.rs†L260-L320】【F:core/crates/ucf-cde/src/lib.rs†L807-L872】
5. **Structure → Perception loop closure**: Structural outcomes (NSR verdicts, coupling influences, TCF caps, SLE reflection) **must** feed back into the next cycle’s phase/attention/perception gating via committed inputs and coherence lag buffers.【F:core/crates/ucf-nsr/src/notar.rs†L160-L210】【F:core/crates/ucf-coupling/src/lib.rs†L196-L236】【F:core/crates/ucf-tcf/src/lib.rs†L120-L210】【F:core/crates/ucf-sle/src/lib.rs†L110-L190】【F:core/crates/ucf-router/src/lib.rs†L293-L340】

## 6. Time-lagged coupling (K=4 ring buffer) and update modes
- **K=4 coherence lag**: The router maintains a fixed 4-slot ring buffer (`COHERENCE_LAG_LEN = 4`) over phase, SSM, IIT, and NSR signals; the buffer rotates each cycle and is committed as a single coherence lag digest.【F:core/crates/ucf-router/src/lib.rs†L117-L140】【F:core/crates/ucf-router/src/lib.rs†L293-L340】
- **Coupling rules**: Lagged influences are computed from committed samples and per-signal lag buffers; the rule set caps max lag but always respects the coherence lag window for update-mode selection.【F:core/crates/ucf-coupling/src/lib.rs†L124-L210】【F:core/crates/ucf-coupling/src/lib.rs†L280-L336】
- **Update modes**: `Conservative`, `Normal`, `Exploratory`, `Stabilize` are chosen deterministically from the hash seed and forced overrides (NSR, risk, drift, phi/surprise thresholds).【F:core/crates/ucf-router/src/lib.rs†L275-L317】【F:core/crates/ucf-router/src/lib.rs†L3081-L3119】【F:core/crates/ucf-router/src/lib.rs†L5128-L5170】

## 7. Safety invariants
- **ThoughtOnly non-leakage**: Thought-only outputs (OutputChannel::Thought) **must** remain internal; SLE emits ThoughtOnly spikes and the router routes them as internal thought-only events without promotion to broadcast by default, enforcing label-based segregation and router gating.【F:core/crates/ucf-sle/src/lib.rs†L210-L330】【F:core/crates/ucf-router/src/lib.rs†L844-L986】【F:core/crates/ucf-router/src/lib.rs†L1558-L1593】
- **Promotion attempt handling**: Any attempt to promote restricted output **must** result in policy/NSR gating and, when coherence degrades, a Stabilize override for update mode to damp outputs and learning caps.【F:core/crates/ucf-router/src/lib.rs†L3081-L3119】【F:core/crates/ucf-tcf/src/lib.rs†L340-L410】
- **Policy ecology read-only assumption**: Policy evaluation is treated as immutable input (policy commit is read-only for the cycle) and NSR decisions are computed against that committed policy state.【F:core/crates/ucf-router/src/lib.rs†L742-L781】【F:core/crates/ucf-nsr/src/notar.rs†L160-L210】

## 8. Coherence Gate (CI guardrails)
The coherence gate is a lightweight CI guardrail that asserts:

- **Pipeline order** is fixed and matches the authoritative `PIPELINE` list used by the runtime orchestrator.
- **Determinism**: running the same fixed input sequence produces identical workspace snapshot commits and archive roots.
- **ThoughtOnly non-leak**: ThoughtOnly emission and a speech escape attempt are rejected by the output router, while stabilization counters advance.

### Extending the pipeline safely
If you add or reorder coherence stages:

1. Update `ucf_router::PIPELINE` to match the new canonical sequence.
2. Update the coherence-gate tests to assert the new order.
3. Keep changes additive and deterministic (no wall-clock or random inputs in tests).

## 9. Failure modes & expected system response
- **Low coherence / high drift** → **Stabilize**: Update mode is forced to Stabilize, lowering learning caps and dampening outputs via TCF and IIT hints.【F:core/crates/ucf-router/src/lib.rs†L3081-L3173】【F:core/crates/ucf-tcf/src/lib.rs†L340-L410】【F:core/crates/ucf-iit/src/lib.rs†L170-L246】
- **High surprise + low phi** → **Replay request**: IIT/SLE can request replay; TCF and router honor replay-active state when thresholds are crossed.【F:core/crates/ucf-iit/src/lib.rs†L170-L246】【F:core/crates/ucf-sle/src/lib.rs†L250-L330】【F:core/crates/ucf-tcf/src/lib.rs†L340-L410】
- **Policy/NSR denial** → **Output suppression**: TCF output caps drop to zero when policy is not OK or NSR verdict is restrictive/deny, and the router blocks outward emissions.【F:core/crates/ucf-tcf/src/lib.rs†L340-L410】【F:core/crates/ucf-router/src/lib.rs†L2058-L2125】

## 10. Glossary (short)
- **Commit**: Blake3 digest binding a structured input/output to deterministic bytes.
- **PhaseBus**: ONN output with `gamma_bucket`, `global_plv`, and `phase_commit` shared across coherence modules.【F:core/crates/ucf-onn/src/lib.rs†L170-L210】
- **Coherence lag**: 4-slot ring buffer of phase/SSM/IIT/NSR state used to seed update modes.【F:core/crates/ucf-router/src/lib.rs†L117-L140】【F:core/crates/ucf-router/src/lib.rs†L293-L340】
- **Update mode**: Deterministic mode (Conservative/Normal/Exploratory/Stabilize) derived from hashed commitments and safety thresholds.【F:core/crates/ucf-router/src/lib.rs†L275-L317】【F:core/crates/ucf-router/src/lib.rs†L3081-L3119】
