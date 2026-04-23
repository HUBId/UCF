# Serie BB8 Prompt 4: Persisted-memory maintenance / invalidation / caveat refresh (minimal)

Status: Dieser Schritt ergänzt BB8 Prompt 1-3 um eine minimale, kanonische Maintenance-Schicht für persisted Blue-Brain memory records. Ziel ist explizite Gültigkeitspflege (current/stale/caveated/refreshed/invalidated/blocked), **ohne** Consolidation-, Ranking-, Search- oder Reasoning-Buildout.

## Scope und betroffene Flächen

- `runtime/ucf-compute/src/blue_brain_memory.rs`
- `runtime/ucf-compute/README.md`

## Kanonische Maintenance-/Validity-Zustände

Persisted-memory records tragen jetzt eine explizite Maintenance-Posture:

- `current`
- `stale`
- `caveated`
- `caveat_refreshed`
- `invalidated`
- `maintenance_blocked`
- `refresh_unavailable`
- `non_canonical_internal_only_path`

Zusätzlich wird Caveat-Refresh-Zustand separat geführt:

- `preserved`
- `refreshed_from_reference_or_evidence`
- `strengthened`
- `weakened`
- `refresh_unavailable`
- `refresh_blocked`

## Minimale Maintenance-Operationen

`BlueBrainMemoryStore::apply_maintenance` erlaubt ausschließlich minimale, deterministische Pflegeoperationen:

- `mark_current`
- `mark_stale`
- `invalidate { reason }`
- `mark_maintenance_blocked { reason }`
- `mark_refresh_unavailable { reason }`
- `refresh_caveats { caveats, refresh_state }`

Result-Semantik (`BlueBrainMemoryMaintenanceResultState`):

- `applied`
- `no_op`
- `blocked`
- `failed`
- `unavailable`
- `caveated`

## Retrieval-/Context-/Selection-Rückbindung

Read-Surface (`read_reference`) bindet Maintenance zurück:

- `invalidated` wird als `retrieved_invalidated` diagnostiziert und schwächt/blockiert strong candidate/proposal basis.
- `stale` bleibt `retrieved_stale` und führt zu defer/weakening.
- `maintenance_blocked` wird retrieval-seitig als `blocked` mit maintenance diagnostic geführt.
- `caveat_refreshed` trägt aktualisierte Caveats in der Reference Surface.
- `refresh_unavailable` bleibt explizit caveated/unavailable statt stillschweigend current.

No-auto-trigger Grenzen bleiben unverändert:

- kein automatischer Compute Trigger,
- kein automatischer Action/Planning Trigger,
- kein automatischer Memory Commit Trigger.

## Historie/Snapshot/Evidence/Replay Abgrenzung

Die Maintenance-Semantik bleibt strikt auf persisted-memory records begrenzt.

- Invalidating memory invalidiert **nicht** automatisch Evidence.
- Stale memory ist **nicht** stale snapshot/replay state.
- Caveat refresh ist **kein** replay refresh.
- Maintenance result ist **kein** audit/result claim.

## Non-canonical Pfade

Internal-/expert-only locator bleiben non-canonical:

- Maintenance über `internal:*` ohne explizite Freigabe wird blockiert.
- Ergebnis bleibt als non-canonical/internal-only diagnostic sichtbar.
- Kein Umweg über History/Snapshot/Evidence-Missbrauch.

## Weiterhin bewusst nicht implementiert

- Memory Consolidation
- Retrieval Ranking / Semantic Search / Vector Search
- Knowledge Graph
- automatische Truth-Validation / Reasoning Engine
- automatische Compute-/Action-/Tool-Invocation aus Maintenance
- automatische neue Memory Persistence aus Maintenance
