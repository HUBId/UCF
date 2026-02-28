# Tool Plan End-to-End Demo (v1)

Diese Demo zeigt den vollständigen sicheren Tool-Pfad mit lokalem Plugin-Tool und Two-Phase-Commit:

1. `ControlFrame` mit Marker `tool_demo_file_read` erzeugt einen `CandidateSet` mit SafeText + ToolIntent.
2. ToolIntent wird durch Governance (EBM/NSR/GEM) bewertet und nur bei erlaubter Policy ausgestellt.
3. 2PC-Kette: `ToolPlanRecord` -> `ToolIssueRecord` -> `ToolExecutionRecord`.
4. Ausführung erfolgt über lokales Plugin (`file_read`) gegen `fixtures/demo_root/hello.txt`.
5. Single-use Token ist erzwungen (`token_replay` wird beim zweiten Intent im selben Tick abgewiesen).
6. Explain/Replay prüfen die Nachvollziehbarkeit der Security- und Evidence-Chain.

## Demo-Artefakte

- Fixture: `fixtures/e2e_tool_plan_demo.json`
- Demo-Datei: `fixtures/demo_root/hello.txt`
- Client fixture: `fixtures/client/controlframe_tool_demo.json`
- Overlay: `policies/packs/overlays/demo_toolread/`
- Script: `scripts/tool_demo_gateway.sh` (PowerShell: `.ps1`)

## Lokal ausführen

```bash
scripts/tool_demo_gateway.sh ./out/tool_demo
```

## Erwartete Record-Kette

In ESS/Explain-Tick sollte mindestens sichtbar sein:

- `CandidateSetRecord`
- `ToolPlanRecord`
- `ToolIssueRecord` (issued)
- `ToolExecutionRecord` (ein `AllowedExecuted` + ein `token_replay` deny)
- `DecisionFrame`/Output mit Experience-Verweisen auf die Tool-Audit-Kette

## Explain-Tick Inspektion

```bash
cargo run -p ucf-client -- --endpoint unix:///tmp/ucf_gateway_v1.sock --auth test-token report explain-tick --t 10
```

## Replay Verify-Only

```bash
cargo run -p ucf-ops -- replay --from 0 --to 999999 --report ./out/tool_demo/replay_verify_report.json
```

