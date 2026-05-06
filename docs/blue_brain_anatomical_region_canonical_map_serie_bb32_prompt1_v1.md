# Serie BB32 Prompt 1: canonical anatomical region map and first-region consolidation

Status: controlled anatomical expansion pass on top of the existing Blue-Brain functional base, without opening a full neuro-simulation platform.

## 1) canonical anatomical region map

The following set is the canonical anatomical region map for this phase:

1. hippocampus
2. amygdala
3. prefrontal cortex
4. anterior cingulate cortex
5. basal ganglia
6. thalamus
7. insula

## 2) mapping from functional base to anatomical regions

- attention/selection-related functional path is mapped to hippocampus as the first operational anatomical region.
- caveat/threat salience lane is mapped to amygdala.
- policy/control consistency lane is mapped to prefrontal cortex.
- conflict-monitoring lane is mapped to anterior cingulate cortex.
- action-gating mediation lane is mapped to basal ganglia.
- relay integration lane is mapped to thalamus.
- interoceptive context lane is mapped to insula.

## 3) model mode per anatomical region

- hippocampus: abstract (current default, first operational region).
- amygdala: bounded kuramoto-like.
- anterior cingulate cortex: bounded kuramoto-like.
- thalamus: abstract functional current mode.
- basal ganglia: abstract functional current mode.
- prefrontal cortex: later selective HH deepening.
- insula: deferred.

This keeps the current architecture bounded and deterministic while preserving explicit deepening options; HH simulation-only/diagnostic-only remains available only as a non-operative diagnostic lane, not as a current default.

## 4) first real anatomical region consolidation

First real anatomical region remains hippocampus (hippocampus_like_region in code lineage), with stabilized:

- input surface
- state surface
- output/advisory surface
- reference surface
- diagnostics + contract semantics
- no-direct authority boundaries

No direct authority is granted from anatomical outputs:

- no direct action trigger
- no direct execution trigger
- no direct retry trigger
- no direct memory commit
- no direct compute invocation
- no safety override

## 5) explicitly deferred in this pass

- no full HH production integration
- no multi-region simultaneous expansion rollout
- no new global neuro-dynamics platform
- no planner/agent/policy authority changes
- no compute-core authority expansion
