# UCF Protocol v1 Specification

This directory contains the v1 protocol specification for UCF. The v1 namespace is intended to
remain stable; changes are governed by strict compatibility rules to protect existing producers and
consumers.

## Versioning rules

- **v1 namespace is stable.** New fields can be added to v1 messages, but existing fields must not
  change semantics or wire tags.
- **Backward compatibility is mandatory.** v1 messages must remain readable by older v1 consumers;
  new optional fields must preserve default behavior.
- **Forward compatibility is preferred.** Unknown fields must be ignored by v1 consumers to allow
  future extensions.
- **Breaking changes require v2.** Any change that removes or repurposes a field, or changes
  serialization expectations, must be introduced in a new major namespace (v2).

## Field numbering discipline

- **Never reuse field numbers.** Once a tag is assigned, it is permanently reserved, even if the
  field is deprecated.
- **Reserve ranges explicitly.** If a block of tags is reserved, document it in the message
  definition notes.
- **Monotonic assignment.** Assign new tags in ascending order to avoid accidental reuse.

## Canonical serialization expectations

- **Deterministic serialization.** Producers must emit fields in canonical order (tag order), and
  consumers must treat different but equivalent encodings as the same semantic value.
- **Stable numeric encodings.** Use explicit integer widths and avoid ambiguous encodings that could
  change across implementations.
- **Canonical hashing.** When messages are hashed, the canonical serialization is the input.

## ID model

- **Stable IDs.** Entity IDs (e.g., frames, policies, milestones) are stable across versions and
  must not be recycled.
- **Referential integrity.** References must point to previously or concurrently published IDs.
- **Opaque identifiers.** IDs are opaque strings/bytes with no semantic meaning; consumers must not
  parse meaning from them.
