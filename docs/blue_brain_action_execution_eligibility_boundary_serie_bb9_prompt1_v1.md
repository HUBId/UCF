# Serie BB9 Prompt 1: Minimal action execution eligibility boundary und safety precheck semantics

Status: BB9 Prompt 1 zieht eine **minimale execution-eligibility boundary** für Blue-Brain-Handoffs.
Der Scope bleibt bewusst eng: Eligibility + Safety-Precheck-Semantik, **keine Tool-Execution-Engine**,
keine autonome Agentenplattform, keine Policy-Governance-Engine, keine Planner-/RL-Engine.

Repo-Anker (code-pinned):
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_BLUE_BRAIN_ACTION_EXECUTION_ELIGIBILITY_MAP`
  - `CANONICAL_BLUE_BRAIN_SAFETY_PRECHECK_MAP`
  - `CANONICAL_BLUE_BRAIN_FUTURE_ACTION_HANDOFF_MAP`
  - `CANONICAL_BLUE_BRAIN_PLAN_ACTION_READINESS_DIAGNOSTICS_MAP`
  - `CANONICAL_BLUE_BRAIN_CANDIDATE_COMPARISON_MAP`
  - `CANONICAL_BLUE_BRAIN_CONTROL_ATTENTION_SELECTION_MAP`
  - `CANONICAL_BLUE_BRAIN_CONTEXT_EVIDENCE_PRIORITY_MAP`
  - `CANONICAL_BLUE_BRAIN_MEMORY_COMMIT_DIAGNOSTICS_MAP`
- `runtime/ucf-compute/src/blue_brain_memory.rs`

Finale Referenzlinie (unverändert):
- `submit -> compute_canonical -> result/fault/status -> execution_snapshot`

## 1) Kanonische BB9-Klassen

### 1.1 Action execution eligibility boundary
- `future-action-ready handoff`
- `execution-eligible handoff`
- `execution-ineligible handoff`
- `execution-blocked handoff`
- `execution-caveated handoff`
- `execution-insufficient basis`
- `safety-precheck-passed`
- `safety-precheck-failed`
- `safety-precheck-blocked`
- `safety-precheck-caveated`
- `safety-precheck-unavailable`
- `executed action (canonical only if explicit real path exists)`
- `non-canonical/internal-only execution path`

### 1.2 Safety precheck semantics
- `precheck passed`
- `precheck failed`
- `precheck blocked`
- `precheck caveated`
- `precheck insufficient`
- `precheck unavailable`
- `precheck not applicable`

Precheck bleibt ein minimaler Safety-Filter und ist weder Governance-Policy noch Ausführung.

## 2) Minimalbedingungen für execution eligibility

`execution-eligible handoff` gilt nur, wenn alle folgenden Basen explizit vorhanden sind:
1. `future-action-ready handoff` vorhanden.
2. Context-Basis ausreichend oder caveated-allowed.
3. Evidence-/Reference-Basis ausreichend oder caveated-allowed.
4. Proposal/Candidate-Basis selected oder accepted und nicht blocked.
5. Memory-Basis current oder caveated-acceptable; invalidated blockiert, missing kann insufficient machen.
6. Kein blocking Candidate-/Proposal-Diagnostic.
7. Safety precheck `passed` oder explizit `caveated`-allowed.
8. Keine internal/expert-only dependency als kanonische Autorität.

## 3) Explizite Trennung: future-action-ready vs execution-eligible

- `future-action-ready but not execution-eligible` ist ein kanonischer Zustand.
- `future-action-ready becomes execution-eligible after precheck` ist explizit und nachvollziehbar.
- `future-action-ready` kann durch context/evidence/memory/safety als blocked oder insufficient enden.
- `future-action-ready` kann rein diagnostic-only bleiben.

## 4) Explizite Trennung: execution-eligible vs executed action

- `execution-eligible != executed action`.
- Eligibility verursacht **keine automatische Action Execution**.
- Eligibility verursacht **keine Tool Invocation**.
- Eligibility verursacht **keine automatische Compute Invocation**.
- Eligibility verursacht **keine automatische Memory Persistence**.

BB9 Prompt 1 definiert nur Eligibility + Precheck-Semantik. Eine echte Execution-Lane ist bewusst nicht Bestandteil dieses Schritts.

## 5) Memory-, Candidate-, Proposal- und Selection-Rückbindung

Eligibility nutzt BB8-/BB6-/BB4-/BB3-Basis:
- Memory current/stale/caveated/invalidated/missing beeinflusst Eligibility.
- Candidate insufficient / comparison inconclusive / proposal caveated beeinflusst Eligibility.
- Selection blocked oder deferral active blockiert Eligibility-Promotion.
- Context/Evidence bleibt referenz- und diagnostikgebunden, nicht policy-autonom.

## 6) Non-canonical execution/safety paths

Compute-interne Details, expert/internal hooks, legacy compat objects oder unstabile dev/test helper sind
keine kanonische Eligibility-/Safety-Autorität ohne explizites Down-Mapping.

## 7) Harte Grenzen (BB9)

BB9 Prompt 1 enthält bewusst:
- keine Tool-Execution-Engine,
- keine vollständige Policy-/Governance-Plattform,
- keine autonome Planner-/Reasoning-/Agenten-Laufzeit,
- keine automatische Action Execution,
- keine automatische Compute Invocation,
- keine automatische Memory Persistence.

Compute-Kern bleibt maintenance-only. Hodgkin-Huxley/Kuramoto bleiben außerhalb dieser BB9-Phase.
