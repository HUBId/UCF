# ucf-sdk Versioning Policy

`ucf-sdk` is the **Stable Core API** for UCF library consumers.

## SemVer rules

- `ucf-sdk` follows semantic versioning.
- **PATCH/MINOR** updates are additive or non-breaking only.
- **MAJOR** updates are required for breaking API changes.

A breaking change includes (non-exhaustive):
- removing or renaming a public type
- removing or renaming a public field
- changing the type of a public field
- removing a public re-export

## Additive change policy

Allowed without a major bump:
- adding a new public type
- adding new optional capabilities behind new APIs
- extending docs/tests/internal implementation without changing public contracts

For enum/struct extensibility, `ucf-sdk` uses `#[non_exhaustive]` on boundary types.

## Deprecation policy

- Mark APIs as deprecated first.
- Keep deprecated APIs available for at least one minor release line before removal.
- Actual removals require the next major version.

## CI enforcement

Public API checks are run using a repository snapshot:

- Snapshot file: `docs/sdk_public_api_snapshot.txt`
- Generator/checker: `scripts/sdk_api_snapshot.py`

Typical workflow:

```bash
python scripts/sdk_api_snapshot.py generate
python scripts/sdk_api_snapshot.py check --baseline-ref HEAD^
```

CI blocks when:
- snapshot is stale, or
- breaking changes are detected relative to baseline without a major version bump.
