# Module Commit Dependencies (Canonical)

This document lists the canonical inputs, outputs, and hash dependencies for key modules in the coherence loop. Hash notation uses `H(…)` to denote Blake3-based commitments from the referenced module implementations.

## ONN (Oscillatory Neural Net)
**Inputs**
- `ssm_state_commit`, `ncde_state_digest`, `cde_commit`, `nsr_trace_root`, `iit_hints_commit`
- `lock_window_buckets`, `risk`, `drift`, `surprise`, `cycle_id`

**Outputs**
- `PhaseBus` (`gamma_bucket`, `global_plv`, `phase_commit`, `commit`)
- `PhaseLockDecision` (`accept_center`, `lock_window_buckets`, `commit`)
- `OnnOutputs.commit`

**Hash dependency summary**
- `inputs.commit = H(cycle_id, ssm_state_commit, ncde_state_digest, cde_commit, nsr_trace_root, iit_hints_commit, lock_window_buckets, risk, drift, surprise)`
- `phase_bus.commit = H(cycle_id, gamma_bucket, global_plv, osc_buckets, inputs.commit, params.commit)`
- `lock.commit = H(cycle_id, lock_window_buckets, accept_center, phase_commit)`
- `outputs.commit = H(phase_bus.commit, lock.commit)`

【F:core/crates/ucf-onn/src/lib.rs†L118-L214】【F:core/crates/ucf-onn/src/lib.rs†L504-L582】

## SpikeBus (SNN bridge)
**Inputs**
- `lock` (PhaseLockDecision from ONN)
- `candidates` (spike list)

**Outputs**
- `accepted_root`, `counts`, `max_intensity`, `SpikeOutputs.commit`

**Hash dependency summary**
- `inputs.commit = H(cycle_id, lock.commit, candidates.len, spike.commit*)`
- `outputs.commit = H(cycle_id, accepted_root, max_intensity, counts, params.commit, inputs.commit)`

【F:core/crates/ucf-spikebus/src/lib.rs†L196-L279】【F:core/crates/ucf-spikebus/src/lib.rs†L366-L406】

## IIT (Coherence regulator)
**Inputs**
- `phase_bus_commit`, `gamma_bucket`, `global_plv`
- `ssm_state_commit`, `ncde_state_digest`, `cde_commit`, `nsr_trace_root`
- `coupling_influences_root`, `risk`, `drift`, `surprise`

**Outputs**
- `phi_proxy`
- `tighten_sync`, `damp_output`, `damp_learning`, `request_replay`
- `hints_commit`, `commit`

**Hash dependency summary**
- `inputs.commit = H(cycle_id, phase_bus_commit, gamma_bucket, global_plv, ssm_state_commit, ncde_state_digest, cde_commit, nsr_trace_root, coupling_influences_root, risk, drift, surprise)`
- `hints_commit = H(cycle_id, phi_proxy, tighten_sync, damp_output, damp_learning, request_replay)`
- `outputs.commit = H(cycle_id, phi_proxy, hints_commit, inputs.commit)`

【F:core/crates/ucf-iit/src/lib.rs†L108-L178】【F:core/crates/ucf-iit/src/lib.rs†L429-L479】

## TCF (Coherence + timing controller)
**Inputs**
- `phase_bus_commit`, `gamma_bucket`, `global_plv`, `phi_proxy`
- `risk`, `drift`, `surprise`
- IIT hints: `iit_hints_commit`, `tighten_sync`, `damp_output`, `damp_learning`, `request_replay`
- `coupling_influences_root`, `nsr_verdict`, `policy_ok`

**Outputs**
- `TcfPlan` (`attention_gain_cap`, `learning_gain_cap`, `output_gain_cap`, `sleep_active`, `replay_active`, `lock_window_buckets`, `smoothing_override`, `commit`)

**Hash dependency summary**
- `inputs.commit = H(cycle_id, phase_bus_commit, gamma_bucket, global_plv, phi_proxy, risk, drift, surprise, iit_hints_commit, tighten_sync, damp_output, damp_learning, request_replay, coupling_influences_root, nsr_verdict, policy_ok)`
- `plan.commit = H(cycle_id, attention_gain_cap, learning_gain_cap, output_gain_cap, sleep_active, replay_active, lock_window_buckets, smoothing_override, inputs.commit, params.commit, state.commit)`

【F:core/crates/ucf-tcf/src/lib.rs†L120-L216】【F:core/crates/ucf-tcf/src/lib.rs†L520-L577】

## NSR (Notarized safety reasoning)
**Inputs**
- `phase_bus_commit`, `policy_commit`, `facts` (rooted)

**Outputs**
- `verdict`, `trace_root`, `NsrOutputs.commit`

**Hash dependency summary**
- `inputs.commit = H(cycle_id, phase_bus_commit, policy_commit, facts_root)`
- `trace_root = H(trace_commit)`
- `outputs.commit = H(cycle_id, verdict, trace_root)`

【F:core/crates/ucf-nsr/src/notar.rs†L160-L210】【F:core/crates/ucf-nsr/src/notar.rs†L390-L464】

## SLE (Self-loop engine)
**Inputs**
- `phase_bus_commit`, `gamma_bucket`
- `ssm_state_commit`, `ssm_salience`, `ssm_novelty`
- `ncde_state_digest`, `ncde_energy`
- `cde_commit`, `nsr_verdict`, `nsr_trace_root`
- `phi_proxy`, `global_plv`, `tcf_sleep_active`, `tcf_replay_active`
- `risk`, `drift`, `surprise`

**Outputs**
- `reflection_commit` (class + intensity + self_symbol)
- `ssm_bias`, `cde_bias`, `request_replay`, `thought_only_root`
- `SleOutputs.commit`

**Hash dependency summary**
- `reflection_commit = H(class, intensity, self_symbol, phase_bus_commit)`
- `outputs.commit = H(cycle_id, reflection_commit, ssm_bias, cde_bias, request_replay, thought_only_root)`

【F:core/crates/ucf-sle/src/lib.rs†L110-L190】【F:core/crates/ucf-sle/src/lib.rs†L250-L330】

## NCDE (Neural CDE memory)
**Inputs**
- `phase_bus_commit`, `gamma_bucket`
- `spike_accepted_root`, `spike_counts`
- `attention_gain`, `coupling_influences_root`, `coupling_influences`
- `ssm_state_commit`, `ssm_salience`, `ssm_novelty`
- `risk`, `drift`, `surprise`, `learning_gain_cap`

**Outputs**
- `ncde_state_digest`, `ncde_energy`, `replay_pressure_hint`, `NcdeOutputs.commit`

**Hash dependency summary**
- `inputs.commit = H(cycle_id, phase_bus_commit, gamma_bucket, spike_accepted_root, spike_counts, attention_gain, coupling_influences_root, coupling_influences, ssm_state_commit, ssm_salience, ssm_novelty, risk, drift, surprise, learning_gain_cap)`
- `outputs.commit = H(cycle_id, ncde_state_digest, ncde_energy, replay_pressure_hint, inputs.commit, params.commit, state.commit)`

【F:core/crates/ucf-ncde/src/lib.rs†L120-L210】【F:core/crates/ucf-ncde/src/lib.rs†L346-L428】

## SSM (State-space memory)
**Inputs**
- `phase_bus_commit`, `gamma_bucket`
- `percept_commit`, `percept_energy`
- `spike_accepted_root`, `spike_counts`
- `coupling_influences_root`, `coupling_influences`
- `tcf_attention_cap`, `tcf_learning_cap`, `b_q15_bias`, `sle_ssm_bias`
- `ncde_energy`, `risk`, `drift`, `surprise`

**Outputs**
- `ssm_state_commit`, `ssm_state_digest`, `ssm_salience`, `ssm_novelty`, `ssm_attention_gain`, `SsmOutputs.commit`

**Hash dependency summary**
- `inputs.commit = H(cycle_id, phase_bus_commit, gamma_bucket, percept_commit, percept_energy, spike_accepted_root, spike_counts, coupling_influences_root, coupling_influences, tcf_attention_cap, tcf_learning_cap, b_q15_bias, sle_ssm_bias, ncde_energy, risk, drift, surprise)`
- `ssm_state_digest = H(state_chunks, cycle_id, phase_bus_commit)`
- `ssm_state_commit = H(ssm_state_digest, params.commit, inputs.commit)`
- `outputs.commit = H(cycle_id, ssm_state_commit, ssm_salience, ssm_novelty, ssm_attention_gain)`

【F:core/crates/ucf-ssm/src/lib.rs†L120-L238】【F:core/crates/ucf-ssm/src/lib.rs†L512-L640】

## CDE (Causal discovery engine)
**Inputs**
- `phase_commit`, `phase_bucket`
- `ssm_salience`, `ssm_novelty`, `cde_bias`
- `attention_gain`, `learning_rate`, `replay_pressure`, `sleep_drive`
- `ncde_energy`, `coherence_plv`, `phi_proxy`
- `risk`, `drift`, `surprise`, `sleep_active`, `replay_active`
- `spike_accepted_root`

**Outputs**
- `dag_commit`, `top_edges`, `intervention`, `summary_commit`, `causal_link_spikes`, `CdeOutputs.commit`

**Hash dependency summary**
- `inputs.commit = H(cycle_id, phase_commit, phase_bucket, ssm_salience, ssm_novelty, cde_bias, attention_gain, learning_rate, replay_pressure, sleep_drive, ncde_energy, coherence_plv, phi_proxy, risk, drift, surprise, sleep_active, replay_active, spike_accepted_root)`
- `summary_commit = H(dag_commit, top_edges, intervention)`
- `outputs.commit = H(cycle_id, dag_commit, summary_commit, top_edges, intervention, spikes)`

【F:core/crates/ucf-cde/src/lib.rs†L170-L320】【F:core/crates/ucf-cde/src/lib.rs†L782-L861】

## Workspace (integration ledger)
**Inputs**
- Aggregated module outputs: SpikeBus, NCDE, CDE, SSM, Coupling, TCF, ONN, IIT, NSR, RSA, SLE
- Cycle metadata and broadcast signals

**Outputs**
- `WorkspaceSnapshot.commit` (single digest of the complete cycle state)

**Hash dependency summary**
- `snapshot.commit = H(cycle_id, recursion_used, spike outputs, ncde outputs, cde outputs, ssm outputs, influence/coupling outputs, tcf plan, coherence lag commit, update_mode, onn phase, iit output+hints, nsr trace/verdict, rsa commits, sle outputs, internal utterances, broadcast signals)`

【F:core/crates/ucf-workspace/src/lib.rs†L1720-L2004】

## Archive (record store)
**Inputs**
- Structured `ExperienceRecord` payloads for cycle plan, workspace snapshots, module reports, and output events

**Outputs**
- Canonical record commitment via `commit_experience_record`

**Hash dependency summary**
- `experience_record.commit = H(encode_experience_record(record))` (deterministic field order + length prefixes)

【F:core/crates/ucf-commit/src/lib.rs†L32-L79】

## Text-only dependency graph (canonical dataflow)
```
ControlFrame/perception
  └─> SSM (percept_commit, percept_energy)
        ├─> ONN (ssm_state_commit)
        ├─> NCDE (ssm_state_commit, salience, novelty)
        ├─> CDE (ssm_salience, ssm_novelty)
        └─> SLE (ssm_state_commit, salience, novelty)

ONN (PhaseBus, Lock)
  └─> SpikeBus (lock)
        ├─> SSM (spike_accepted_root, counts)
        ├─> NCDE (spike_accepted_root, counts)
        └─> CDE (spike_accepted_root)

NCDE
  ├─> ONN (ncde_state_digest)
  ├─> SLE (ncde_state_digest, ncde_energy)
  └─> CDE (ncde_energy)

CDE
  ├─> ONN (cde_commit)
  └─> SLE (cde_commit)

NSR (policy + facts)
  ├─> ONN (nsr_trace_root)
  ├─> IIT (nsr_trace_root)
  ├─> TCF (nsr_verdict)
  └─> SLE (nsr_verdict, nsr_trace_root)

IIT (phi_proxy + hints)
  ├─> TCF (hints)
  └─> ONN (iit_hints_commit)

TCF (caps + lock_window_buckets)
  ├─> SSM/NCDE (attention/learning caps)
  └─> ONN (lock_window_buckets)

SLE (self loop)
  └─> Workspace (reflection_commit, thought_only_root, biases)

Workspace
  └─> Archive (ExperienceRecord commitments)
```
