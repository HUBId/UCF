# Blue-Brain Hippocampus Surface / Diagnostics / Contracts Hardening (Serie BR1 — Prompt 3)

This pass hardens the existing BR1 Prompt-2 hippocampus integration line without expanding authority or scope.

## Canonical hippocampus surface classes
- hippocampus input surface
- hippocampus state surface
- hippocampus output/advisory surface
- hippocampus reference surface
- blocked/deferred hippocampus path
- non-canonical/internal-only hippocampus path

## Canonical hippocampus diagnostics/contract map
- hippocampus advisory-only diagnostic
- hippocampus caveated diagnostic
- hippocampus deferred diagnostic
- hippocampus blocked diagnostic
- hippocampus insufficient diagnostic
- hippocampus diagnostic-only state
- hippocampus bounded contract signal
- non-canonical/internal-only hippocampus path

## Contract semantics hardening
- advisory-only != caveated
- caveated is not a strong positive support signal
- deferred != blocked
- blocked != insufficient
- diagnostic-only stays non-operative

Runtime, Selection, and Reference consume the same canonical hippocampus semantics through the same contract-to-diagnostic mapping and do not maintain divergent interpretations.

## Bounded integration and explicit exclusions
The hippocampus contract signal remains advisory/diagnostic and bounded to context-memory-reference and runtime-selection surfaces.

Explicitly excluded:
- no action request
- no execution trigger
- no retry trigger
- no memory commit
- no compute trigger
- no safety override

## Model-mode boundary
- abstract functional (current mode)
- bounded Kuramoto-like and HH simulation-only/diagnostic-only remains deferred for this hippocampus line
- HH simulation-only/diagnostic-only remains deferred unless explicitly re-scoped in a future decision

## Out-of-scope guardrail
No new anatomical region, no planner/agent orchestration, no policy-governance expansion, and no compute-core widening is introduced in this hardening pass.
