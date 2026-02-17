# Real Compute Onboarding v0 Scope

## In v0
- Deterministic local compute stubs/toy paths for JEPA, SAE, SSM, and LFM.
- Optional LFM-LNN kernel path under deterministic constraints.
- Governance tiering, throttles, and emergency mode behavior.
- Local hash-locked model slots via manifest + allowlist root.
- Model probes and staged enablement rollout (including shadow patterns).
- Offline readiness gate, adversarial harness, replay verify-only checks.
- ESS snapshot/compaction and run/status operator surfaces.

## Deferred (explicitly not in v0)
- Network/remote compute execution and federation workflows.
- Training or fine-tuning pipelines.
- Adaptive/non-deterministic samplers and dynamic solvers.
- Full WASM+cgroups sandbox with remote attestation proofs.
- Large-scale vector database retrieval/serving infrastructure.
- Any online dependency required for successful sign-off.
