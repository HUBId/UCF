# Policy Packs v1

`policies/packs/base_v1` defines the canonical base policy pack. Environment overlays live under `policies/packs/overlays/{dev,test,prod}`.

## Structure
Each pack contains:
- `pbm_gem_rules.toml`
- `nsr_rules_v1.dl`
- `ebm_constraints.toml`
- `budgets.toml`
- `thresholds.toml`
- `allowlists.toml`
- `pack_manifest.toml` (name, semver version, schema_version, file hashes, `pack_digest`)

## Merge semantics
- Base loads first and is fully hash-verified.
- Overlay (optional) is hash-verified and merged deterministically.
- Rule append is stable and deduplicated by id.
- EBM term id collisions are rejected.
- Budget/threshold/allowlist overlays may only override keys existing in base (strict typo protection).
- Policy graph encoding is bounded (`MAX_GRAPH_BYTES`, max term/rule counts).

The merged graph digest is `policy_graph_digest` and is used as the canonical runtime policy reference.

## CLI tooling
Validate:

```bash
ucf-ops policy validate --pack policies/packs/base_v1 --overlay policies/packs/overlays/test
```

Diff:

```bash
ucf-ops policy diff \
  --a-pack policies/packs/base_v1 --a-overlay policies/packs/overlays/dev \
  --b-pack policies/packs/base_v1 --b-overlay policies/packs/overlays/prod
```

Explain from ESS provenance:

```bash
ucf-ops policy explain --workdir .ucf --digest <policy_graph_digest_prefix>
```

## Runtime binding
Runtime startup computes `policy_graph_digest` from merged packs and can enforce strict expectation via:

```bash
UCF_POLICY_GRAPH_DIGEST=<full_digest>
```

Overlay selection is controlled via:

```bash
UCF_POLICY_OVERLAY=dev|test|prod
```

All decisions, capability issuance, and budget-window audit records bind to the merged graph digest (or digest prefix in bounded records).
