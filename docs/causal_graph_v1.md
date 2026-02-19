# Causal Graph v1

Causal Graph v1 builds a deterministic, bounded graph from ESS records on demand.

## Node model

`EventId` is a SHA-256 digest over `(event_type, run_id, t, primary_record_digest)`.

Supported event types:

- `control`
- `decision`
- `tool_plan`
- `tool_issue`
- `tool_exec`
- `experience`
- `milestone`

Each `EventNode` stores only digest-safe metadata:

- `event_id`
- `event_type`
- `t`
- `record_digest_prefix`
- `policy_graph_digest_prefix` (optional)
- optional fixed-point scalar summaries (`risk_q`, `pressure_q`, `energy_q`)

## Edge model

Edge tuple:

- `src_event_id`
- `dst_event_id`
- `edge_type`
- `evidence_digest_prefix`

Edge types:

- `causes`
- `enables`
- `justifies`
- `consumes`
- `produces`
- `counterfactual_of`

### Construction rules

Explicit links:

1. `Control -> Decision` (`causes`) via correlation ID.
2. `ToolPlan -> ToolIssue` (`enables`) via `plan_digest_prefix`.
3. `ToolIssue -> ToolExec` (`enables`) via strict correlation/tick linkage.
4. `Decision -> Experience` (`produces`) when decision id is present.

Inferred links (strict deterministic heuristics):

1. Same-tick adjacency: `Decision -> ToolPlan` (`enables`).
2. Same candidate id: EBM-linked `Experience -> Decision` (`justifies`).

## CLI

### Build bounded causal slice

```bash
ucf-ops causal slice --run <id> --event <event_id> --radius 2 --out ./out/causal_slice.json
```

Alternative centered by decision id:

```bash
ucf-ops causal slice --run <id> --decision <decision_id> --radius 2 --out ./out/causal_slice.json
```

Output is bounded by BFS radius and max node cap (128).

### Explain why

```bash
ucf-ops explain why --decision <id>
```

This prints top incoming causes and outgoing effects from the decision-centered slice.

### Counterfactual simulation-only

```bash
ucf-ops counterfactual simulate --decision <id> --candidate <candidate_id> --out ./out/counterfactual_result.json
```

Simulation behavior:

1. Uses existing ESS decision context and candidate summaries.
2. Re-evaluates alternative candidate under same policy digest context.
3. If tools would be required, reports simulated issuance outcome only.
4. Never executes tools.

`CounterfactualRecord` persistence is digest-only (`ess/counterfactual_records.json`).

## Limitations and safety

- v1 derives graph on demand; no global graph DB.
- v1 stores digest-only artifacts and bounded slices.
- Counterfactual is non-executing and auditable (evidence digest prefixes).
