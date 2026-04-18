# Serie L: Final constrained-vs-accepted Exit Boundary (Prompt 2) v1

Stand: Repo-Zustand am 2026-04-18.

Ziel: die finale technische Exit-Grenze explizit ziehen zwischen **stable**, **constrained but accepted** und **not accepted for final exit** – ohne Governance-/Risikorahmen und ohne Ausbauarbeit.

## 1) Verbleibende constrained Bereiche (repo-basiert gesammelt)

Aus den kanonischen Referenz- und Integrationsflächen verbleiben technisch relevante constrained Ränder in:

- Rollout/Replay-Handoff und Comparability-Preflight (`replay_preflight -> replay_with_entry`, fail-closed bei unvollständiger Grundlage).
- Outward Integration als read-only/caveated Exportfläche (`status_evidence_export_surface`, `integration_hook_view`).
- Expert-Runtime-Control (`run_operation_with_entry`, `replay_with_entry`) als high-trust/internal Boundary.
- Compatibility/Legacy-Lanes (`build_backend(kind=stub|candle)`, `build_backend(kind=worker) + domains/ai*`).

Diese Fläche folgt direkt aus `runtime/ucf-compute/src/reference_map.rs`,
`runtime/ucf-compute/src/service_surface.rs` sowie der Serie-J/K/L-Doku.

Kanonische Referenzanker (wörtlich, code-pinned):

- `submit -> compute_canonical -> result/fault/status -> execution_snapshot`
- `replay_preflight -> replay_with_entry -> comparison/evidence on same result/fault/status core`
- `compatibility backends + internal/legacy worker/domain lanes are extension/internal only`

## 2) Finale Klassifikation je Bereich

| Bereich | Finalklasse | Technischer Kurzgrund |
|---|---|---|
| Canonical production line (`submit -> compute_canonical -> result/fault/status -> execution_snapshot`) | **stable** | Eindeutig code-pinned als kanonische Produktionslinie; keine zweite outward Produktionsautorität. |
| Rollout/Replay strictness boundary (`insufficient`/`blocked` statt soft success) | **constrained but accepted** | Constrained durch harte Preconditions, aber bewusst fail-closed und deterministisch; schützt Exit-Semantik statt sie zu verwässern. |
| Outward status/evidence + integration-safe hooks | **stable** (mit expliziten Caveats) | Outward bleibt read-only/caveated und hängt am gleichen Core-Semantikanker; Caveats sind Teil der gewollten Oberfläche, kein Blocker. |
| Expert runtime control (`run_operation_with_entry`, `replay_with_entry`) | **constrained but accepted** | Technisch absichtlich high-trust/internal; kein outward Standard-Contract, aber auf derselben Core-Semantik fixiert. |
| Compatibility/legacy lanes (`stub|candle`, `worker`, `domains/ai*`) | **not accepted for final exit** (als outward authority) | Bewusst nur extension/internal; bleibt außerhalb finaler outward Exit-Akzeptanzgrenze.
|

## 3) Explizite Accept/Not-Accept Boundary (final)

### Accepted für technischen Exit

1. **Eine stabile kanonische Produktionslinie** als alleinige outward Ausführungslinie.
2. **Fail-closed Rollout/Replay-Strictness** als akzeptierte constrained Schutzkante.
3. **Outward status/evidence/hook surfaces** nur in read-only/caveated Semantik.
4. **Expert-/Runtime-Control intern** als constrained, aber akzeptiert solange die shared core invariants unverändert bleiben.

### Nicht accepted für technischen Exit

1. **Jede Nutzung von compatibility/legacy lanes als outward Produktionsautorität.**
2. **Jede Umdeutung von internal/dev lanes zu generischen outward Contracts.**
3. **Jede Aufweichung der fail-closed Rollout/Replay-Preconditions zu soft-success ohne belastbare Grundlage.**

## 4) Minimale Nachhärtung in dieser Runde

Nur eine kleine Konsistenzhärtung wird ergänzt, damit die Exit-Grenze nicht implizit driftet:

- repo-Test koppelt diese Boundary-Doku an die kanonischen Referenzkonstanten und an die
  expliziten Finalklassen (`stable`, `constrained but accepted`, `not accepted for final exit`).

Keine neue Architektur, kein neuer Governance-/Risikorahmen, keine breite Testwelle.

## 5) Direkte Folge für Serie L

Nächster unmittelbarer Schritt nach dieser Boundary-Klärung:

1. verbleibende Serie-L-Runde nur noch auf punktuelle Konsistenz-Driftchecks der akzeptierten Linien,
2. keine Wiedereröffnung bereits als `not accepted` markierter outward-Authority-Restkanten,
3. Abschlussfokus auf reproduzierbare technische Exit-Stabilität der kanonischen Linie.


## 6) Alignment mit finalem Exit-Dossier (Prompt 3)

Diese Prompt-2-Boundary bleibt die detailierte Accept/Not-Accept-Abgrenzung und ist auf
`docs/real_compute_exit_dossier_serie_l_v1.md` als finalen kompakten Abschluss ausgerichtet.
Die kanonische Autorität bleibt die final reference line plus code-pinned invariants/contracts.
