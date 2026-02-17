# Threat Model v1

## System boundary
The protected boundary is the local/offline runtime stack: sandboxed agent loop, governor/issuance, ToolGate authorization, and sandboxed filesystem/tool execution interfaces.

## Assumptions
- Offline execution (no network required by harness runtime).
- Local filesystem allowlists enforced by capability scope and sandbox FS roots.
- Deny-by-default issuance posture.
- No unrestricted subprocess execution in normal policy path.

## Assets
- Policy bundle hash (`bundle_sha256`) and provenance records.
- ESS/audit integrity chain digests.
- Model slot hashes and backend pack metadata.
- Capability tokens and throttle state digests.

## Adversaries
- Prompt injection via user-controlled text.
- Model hallucination causing improper tool request.
- Malicious model weights constrained by sandbox/policy.
- Corrupted/tampered ESS evidence records.

## Attack surfaces
- Tool invocation interface and authorization path.
- Filesystem path traversal and symlink escape.
- Policy bundle tampering / hash mismatch at startup.
- Capability token replay/reuse and rate abuse.
- Compute/governor DoS pressure via repeated requests.

## Mitigations mapped to code
- **ToolGate authorization + deny reasons**: `runtime/ucf-policy/src/gem.rs`.
- **Issuance tiers/governor throttle**: `runtime/ucf-policy/src/gem.rs`.
- **Capability scope checks**: `runtime/ucf-policy/src/capability.rs`.
- **Emergency mode + forced deny-all semantics**: `runtime/ucf-runtime/src/orchestrator.rs`.
- **Sandbox filesystem traversal/escape protection**: `runtime/ucf-runtime/src/sandbox_fs.rs`.
- **Policy bundle verification fail-fast**: `runtime/ucf-policy/src/policy_bundle.rs`, `runtime/ucf-runtime/src/orchestrator.rs`.

## Out of scope
- Physical host compromise.
- Kernel compromise / hypervisor escape.
- Supply-chain compromise outside pinned repository state.

## Residual risks
- In-process isolation limits blast-radius vs process-level or WASM compartmentalization.
- Prompt-only policy bypass attempts still require continual fixture expansion.
- Replay/forensics guarantees depend on persistent storage durability.

## Future work
- Stronger process isolation for all tool invocations.
- WASM-first deterministic tool adapters with fuel accounting.
- Signed policy bundle distribution + key rotation docs.
