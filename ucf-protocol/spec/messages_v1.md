# UCF Protocol v1 Messages

This document defines the v1 message set at a textual level. Field numbers are illustrative and
must follow the field-numbering discipline in `README.md`.

## ControlFrame

Represents a control directive emitted by a policy system.

- `string frame_id` (tag 1): Stable identifier for the frame.
- `uint64 issued_at_ms` (tag 2): Wall-clock timestamp in milliseconds.
- `PolicyDecision decision` (tag 3): Decision payload.
- `repeated string evidence_ids` (tag 4): References to evidence records used for the decision.
- `string policy_id` (tag 5): Stable identifier for the policy version.

## PolicyDecision

Encodes a policy outcome with structured metadata.

- `DecisionKind kind` (tag 1): The high-level decision type.
- `ActionCode action` (tag 2): The specific policy action.
- `string rationale` (tag 3): Human-readable explanation.
- `uint32 confidence_bp` (tag 4): Confidence in basis points (0-10,000).
- `repeated string constraint_ids` (tag 5): References to active constraints.

### DecisionKind (enum)

- `DECISION_KIND_UNSPECIFIED` (0)
- `DECISION_KIND_ALLOW` (1)
- `DECISION_KIND_DENY` (2)
- `DECISION_KIND_ESCALATE` (3)
- `DECISION_KIND_OBSERVE` (4)

### ActionCode (enum)

- `ACTION_CODE_UNSPECIFIED` (0)
- `ACTION_CODE_CONTINUE` (1)
- `ACTION_CODE_PAUSE` (2)
- `ACTION_CODE_TERMINATE` (3)
- `ACTION_CODE_REQUIRE_HUMAN` (4)

## ExperienceRecord

Captures an observed experience that can be used for audit or learning.

- `string record_id` (tag 1): Stable identifier for the record.
- `uint64 observed_at_ms` (tag 2): Observation timestamp in milliseconds.
- `string subject_id` (tag 3): Entity producing the experience.
- `bytes payload` (tag 4): Canonical serialized payload for downstream parsing.
- `Digest digest` (tag 5): Content digest for verification.
- `VRFTag vrf_tag` (tag 6): Proof that the record was sampled/selected.

## Digest

Represents a content digest for integrity checking.

- `string algorithm` (tag 1): Hash algorithm identifier (e.g., "sha256").
- `bytes value` (tag 2): Raw digest bytes.

## VRFTag

Verifiable random function tag for sampling/selection proofs.

- `string algorithm` (tag 1): VRF algorithm identifier.
- `bytes proof` (tag 2): VRF proof bytes.
- `bytes output` (tag 3): VRF output bytes.

## Milestones

Milestones track lifecycle progress at different granularities.

### MicroMilestone

- `string milestone_id` (tag 1): Stable identifier.
- `uint64 achieved_at_ms` (tag 2): Achievement timestamp.
- `string label` (tag 3): Human-readable label.

### MesoMilestone

- `string milestone_id` (tag 1): Stable identifier.
- `uint64 achieved_at_ms` (tag 2): Achievement timestamp.
- `string label` (tag 3): Human-readable label.
- `repeated string micro_milestone_ids` (tag 4): References to micro milestones.

### MacroMilestone

- `string milestone_id` (tag 1): Stable identifier.
- `uint64 achieved_at_ms` (tag 2): Achievement timestamp.
- `string label` (tag 3): Human-readable label.
- `repeated string meso_milestone_ids` (tag 4): References to meso milestones.

## ProofEnvelope

Container for attaching proofs to protocol messages.

- `string envelope_id` (tag 1): Stable identifier for the envelope.
- `bytes payload` (tag 2): Canonical serialized payload of the wrapped message.
- `Digest payload_digest` (tag 3): Digest of the payload.
- `repeated VRFTag vrf_tags` (tag 4): VRF tags associated with the payload.
- `repeated string signature_ids` (tag 5): References to external signatures.
