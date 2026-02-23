# Deprecations (post-rc1 cleanup)

## Active deprecations

| Item | Status | Replacement | Last usage | Target removal |
|---|---|---|---|---|
| `ucf-compute` feature `backend-stub` | Deprecated, kept for toy/default bringup | `compute-candle`/`compute-burn` runtime lanes | default workspace profile | v1.2 |
| `ucf-compute` feature `backend-toy` | Deprecated, kept for toy/default bringup | `compute-candle`/`compute-burn` runtime lanes | default workspace profile | v1.2 |
| `ucf-compute` feature `compute-stub` | Deprecated compatibility alias | `compute-candle`/`compute-burn` | tests/default profile | v1.2 |

## Notes

- Removal follows the rule: no behavior change without tests. Before removing any listed flag/module, add/keep matrix coverage and migration notes.
- Truly unused code should be removed immediately once confirmed unreachable.
