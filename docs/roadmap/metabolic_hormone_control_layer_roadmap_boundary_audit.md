# UCF Metabolic/Hormone Control Layer Roadmap and Boundary Audit

## 0. Purpose
- Inventory/roadmap only.
- No runtime/action authority.
- No policy mutation.
- No identity finalization.

## 1. Baseline
- HEAD: `8c02b3e084197b5c2551575b6d7235f821106186`.
- Presence checks:
  - `docs/current_state_architecture_index.md`: yes
  - `docs/module_implementation_depth_registry.md`: yes
  - `domains/ucf-neuromod`: yes (note: required path `core/crates/ucf-neuromod` is absent)
  - `runtime/ucf-replay`: yes
  - `core/crates/ucf-sleep-coordinator`: yes
  - `domains/geist/crates/ucf-geist`: yes
  - `domains/consolidation/crates/ucf-consolidation`: yes

## 2. Metabolic / Hormone Surface Inventory

| Concern | Path | Current behavior | Maturity | Risk |
|---|---|---|---|---|
| Neuromod v0 state/update loop | `domains/ucf-neuromod/src/v0/{field.rs,rules.rs,scheduler.rs,tests.rs}` | Has `NeuromodulatorField`, `NeuromodInputs`, deterministic `compute_delta`, tick scheduler and unit tests; update uses `f32` clamps and time-step scheduler ticks. | partial | Scheduler/timing loop can be misread as runtime authority; `f32` determinism risk across environments. |
| Minimal Spine neuromod envelope | `domains/ucf-neuromod/src/minimal_spine.rs`, `domains/ucf-neuromod/tests/minimal_spine_envelope.rs` | Deterministic metadata-only hints (`salience/stability/risk/noise/learning`) derived from canonical links; explicit `allows_decision_override=false`; bounded hints and deterministic digest tests. | bounded/tested | Overclaim risk if interpreted as full hormone/metabolic control rather than metadata envelope. |
| Neuromod snapshot type | `domains/ucf-frames/src/v1/neuromod.rs`, `domains/ucf-frames/src/v1/mod.rs` | Exposes neuromod snapshot channels used by neuromod crate. | skeleton | Could be interpreted as integrated control contract though no cross-module authority binding. |
| Hormone-adjacent frame fields | `domains/ucf-frames/src/v1/biophys.rs`, `domains/ucf-frames/src/v1/digital_brain.rs`, `domains/ucf-biophys/src/v0/hormone.rs` | Contains hormone/biophys naming and quantization helpers; no bounded HormoneState v1 contract for spine integration. | partial | Biological naming can overstate fidelity/authority. |
| Replay hormone consumption | `runtime/ucf-replay/src/lib.rs` | Reads hormone stress from replayed hormone records as optional bounded input to governance signal digest; replay boundaries keep identity/gateway false in tested paths. | partial | Misread risk that hormone stress drives scheduler/action authority. |
| Sleep integration surfaces | `core/crates/ucf-sleep-coordinator` | Sleep candidate/audit/boundary path exists; no direct hormone-owned scheduler authority. | bounded/tested | Coupling could be overextended without candidate-only constraints. |
| Geist/ISM relation | `domains/geist/crates/ucf-geist` + roadmap docs | Current Geist line is bounded/candidate-oriented with no required hormone write/upsert authority. | bounded/tested | Future coupling risk if hormone layer writes ISM. |
| Consolidation attention/novelty fields | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | Attention/novelty weighting exists in consolidation scoring; not a hormone contract. | functional-prototype | Misinterpretation as hormone modulation authority path. |
| Current-state + registry docs | `docs/current_state_architecture_index.md`, `docs/module_implementation_depth_registry.md`, `docs/minimal_spine_v1_freeze.md` | Explicitly classify neuromod as partial/experimental except bounded minimal envelope; full metabolic/DBM/HPA claims deferred. | bounded/tested | Documentation drift/overclaim if future updates do not preserve these caveats. |

Answers (M1):
- HormoneState/NeuromodState exists as **neuromod field/snapshot forms** (`NeuromodulatorField`, `NeuromodulatorSnapshot`), but no bounded `HormoneStateV1` contract.
- Deterministic update function exists (`compute_delta`) but uses `f32` and a scheduler loop; not yet a spine-bounded hormone contract.
- Modulation output type exists only as bounded minimal metadata hints (`MinimalSpineNeuromodEnvelope`), not full replay/sleep modulation outputs.
- Replay/Sleep/Geist connections are indirect and bounded; no direct hormone authority for scheduler/action/identity/policy.
- Tests exist for neuromod v0 and minimal envelope, plus replay/sleep/geist boundary suites.
- Broad unsafe surface risk: experimental neuromod scheduler/state APIs can be overread as runtime control.
- Policy/gateway/identity risks are currently guarded in spine tests/docs, but coupling must remain explicit candidate/read-only.

## 3. Boundary Decisions

| Boundary | Decision | Reason |
|---|---|---|
| Policy Ecology | read-only only | Hormone layer may observe policy outcomes/pressure but must never mutate policy packs or policy verdict authority. |
| Replay | candidate priority only | Hormone outputs can suggest replay priority multipliers only; scheduler/apply authority remains in replay boundaries. |
| Sleep | candidate pressure only | Hormone outputs can suggest sleep-pressure deltas only; no SleepCompleted/runtime sleep authority. |
| Geist/ISM | no write/upsert | Prevent hormone layer from becoming hidden identity/state authority via ISM writes. |
| Evidence/Archive | no append | Preserve existing append authority boundaries and avoid secondary event-log authority. |
| Gateway | no visibility/action | Hormone internals stay non-gateway-visible and action-disconnected. |
| Identity | no anchor/finalization | Hormone modulation must never set anchor/finalization semantics. |
| Runtime scheduler | out of scope | M1-M7 remain docs/contract/bounded mapping only; no queue/worker/scheduler runtime activation. |
| State update | deterministic only | Only explicit state-object transitions with bounded coefficients/clamps; no hidden mutable globals or nondeterministic inputs. |

## 4. Risk Matrix

| Risk | Severity | Guardrail |
|---|---|---|
| hormone state becomes hidden global mutable authority | high | State must be explicit (`prev,input,config -> next,output`), serialized, and test-covered for deterministic replay. |
| policy mutation through metabolic layer | high | Enforce read-only policy interface and explicit no-policy-mutation tests/docs assertions. |
| replay/sleep priority becomes runtime scheduler authority | high | Restrict outputs to candidate multipliers/deltas; prohibit scheduler enqueue/activation paths. |
| cortisol/stress interpreted as safety policy | high | Treat stress as advisory scalar only; never map directly to allow/deny policy verdicts. |
| sleep pressure interpreted as SleepCompleted | high | Keep sleep-pressure as candidate signal only; completion status remains separate authority path. |
| novelty/reward interpreted as action authority | high | No direct action fields in modulation output; only bounded gains/noise/priority hints. |
| Geist/ISM write via hormone state | high | No ISM write/upsert interfaces in hormone module contract. |
| Evidence/Archive side effects | high | Forbid append calls in hormone layer and keep verify-only audit/candidate records. |
| unbounded feedback loop | medium-high | Use bounded gains, decay, clamps, and explicit max-step deterministic update semantics. |
| nondeterministic update due wall-clock/randomness | high | No wall-clock/random sampling in update function; deterministic numeric contract and reproducible tests. |

## 5. Proposed Architecture Shape

| Proposed component | Purpose | Inputs | Outputs | Non-goals |
|---|---|---|---|---|
| `HormoneStateV1` | Canonical bounded hormone/metabolic state container. | previous state + bounded channel values. | next state fields (`dopamine_like`, `serotonin_like`, `cortisol_like`, `arousal_like`, `sleep_pressure`, `novelty_pressure`, `stability_pressure`). | No policy/gateway/identity/evidence authority. |
| `HormoneInputFrameV1` | Deterministic external signal frame for update. | reward/novelty/threat/fatigue/inconsistency/replay-density/policy-violation-pressure signals. | normalized deterministic input payload for one update step. | No runtime side effects or scheduler writes. |
| `HormoneUpdateConfigV1` | Explicit deterministic tunables. | fixed decay/gain/clamp constants. | validated config used by updater. | No hidden auto-tuning, no random/adaptive mutation. |
| `HormoneModulationOutputV1` | Advisory-only modulation vector for downstream candidate builders. | `next HormoneStateV1` (+ optionally validated context). | attention gain, LR multiplier, replay priority multiplier, noise scale, consolidation gate, sleep-pressure delta, risk damping. | No direct action execution, no final verdict authority. |
| `update_hormone_state_v1(prev,input,config)` | Single deterministic transition primitive. | previous state + input + config. | `(next_state, modulation_output)` deterministic tuple. | No scheduler runtime loop, no policy mutation, no append/write authority. |

## 6. Prompt Series Plan

| Prompt | Title | Goal | Acceptance criteria | Guardrails |
|---|---|---|---|---|
| M2 | Deterministic HormoneState v1 Contract | Add bounded type contract and invariants only. | ✅ `HormoneStateV1` + `NormalizedHormoneLevelV1` fixed-point contract added with deterministic invariants and targeted tests (`domains/ucf-neuromod/tests/hormone_state_v1.rs`). | No runtime loops, no policy/gateway/identity/archive authority. |
| M3 | Hormone Update Rules v1 | Define deterministic update mapping and clamps/decay. | `update_hormone_state_v1` deterministic tests incl. clamp/decay edge cases. | No wall-clock randomness, no scheduler activation. |
| M4 | Hormone Modulation Output Mapping | Map state to advisory modulation outputs. | `HormoneModulationOutputV1` mapping with bounded multipliers + docs/tests. | Advisory-only; no direct action/policy decisions. |
| M5 | Replay/Sleep Priority Candidate Mapping | Connect modulation outputs to candidate builders only. | Candidate-only mapping for replay priority/sleep pressure, with boundary tests. | No replay/sleep scheduler authority, no SleepCompleted semantics. |
| M6 | Metabolic Verify-Only Audit Contract | Add deterministic audit record contract for hormone updates/modulations. | Verify-only audit type + deterministic digest + no append/write authority tests. | No Evidence/Archive append, no ISM upsert, no gateway visibility. |
| M7 | Metabolic Docs Overclaim Guard | Harden docs/index/registry to prevent capability overclaim. | Updated docs with explicit non-goals and boundary assertions. | No behavior change. |
| M8 | Metabolic Readiness Refresh | Targeted readiness/docs checks for metabolic lane artifacts. | Targeted checks pass; docs lint clean; no full runtime claims. | No production-readiness claim, no runtime activation. |

## 7. Current Status
- Metabolic/Hormone control not fully integrated.
- Bounded deterministic minimal-spine neuromod metadata envelope exists, plus experimental neuromod v0 state/rules/scheduler.
- Bounded deterministic `HormoneStateV1` contract is implemented in `domains/ucf-neuromod/src/hormone_state_v1.rs` with targeted tests in `domains/ucf-neuromod/tests/hormone_state_v1.rs`.
- M3 implemented bounded deterministic update rules in `domains/ucf-neuromod/src/hormone_update_v1.rs` with targeted tests in `domains/ucf-neuromod/tests/hormone_update_v1.rs`.
- Replay/Sleep/Geist mapping and runtime scheduler integration remain explicitly deferred.
- M4 completed: extracted deterministic advisory-only mapping `derive_hormone_modulation_output_v1(state)` with targeted modulation semantic/boundary tests in `domains/ucf-neuromod/tests/hormone_modulation_v1.rs`.
- Next step: `UCF Prompt M5 — Replay/Sleep Priority Candidate Mapping`.

## 8. Open Questions
- Which crate owns `HormoneState`?
- Is `ucf-neuromod` the correct owner?
- Should hormone values be scalar bounded floats, fixed-point ints, or typed normalized values?
- How to avoid nondeterminism?
- How to encode decay/gain coefficients?
- How to connect replay/sleep without scheduler authority?
- How to prevent policy mutation?
- How to keep Geist/ISM write out of scope?

## 9. Recommended Next Prompt
UCF Prompt M5 — Replay/Sleep Priority Candidate Mapping
