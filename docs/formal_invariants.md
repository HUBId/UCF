# Formal Verification Hooks (v1)

This repository provides a bounded SMT harness for policy invariants.

## Invariants

1. `tool_execution_requires_issued_token`
   - Model: `has_exec -> has_issue`
   - Counterexample query: `has_exec && !has_issue`.
2. `governor_tier_monotone_tightening`
   - Model: `risk1 <= risk2 && uncertainty1 <= uncertainty2 -> score1 <= score2`
   - Score uses fixed-point integer weights (`*_weight_q`) and bounded domains `{0, 5000, 10000}`.
3. `sampling_disabled_in_prod`
   - For `prod`, solver checks there is no model with `sampling_allowed=true`.
4. `promoted_only_weights`
   - Pinned/active model hashes must belong to the promoted set.

All witnesses are redaction-safe and only expose bounded integer/enum-ish values and hash strings.

## Determinism

- SMT instances are generated as canonical SMT-LIB text with stable declaration order.
- `smt_instance_digest` is SHA-256 of the full SMT script.
- Same policy graph and environment inputs produce byte-identical SMT scripts.

## Running

Feature lane:

```bash
cargo run -p ucf-ops --features formal-smt -- readiness-gate --profile prod --out ./out/gate_report.json
```

If `z3` is not available, readiness gate marks `formal_invariants_smt` as `SKIP` with a remediation hint.

## Artifacts

- Formal report path:
  - `out/formal/<policy_graph_digest>/invariants_report.json`
- Includes:
  - per-invariant status
  - `smt_instance_digest`
  - optional witness for SAT counterexamples

## Counterexample interpretation

- `status=FAIL` means the solver found a satisfying assignment for the counterexample query.
- For monotonicity failures, inspect `risk1/risk2/score1/score2` witness lines and policy weights.
- For promoted-only failures, inspect `pin_<idx>` witness keys and compare with `models/promoted/*/*`.
