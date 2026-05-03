# Blue Brain Third-Region Relation Line (Serie BB28 Prompt 4)

Status: erste **bounded** Drei-Regionen-Relation mit Region 3 (runtime-feedback integration lane) gegen Region 1/2.

## Canonical third-region relation map

Die kanonische Map enthält genau:
- `region-3-to-region-1 bounded relation`
- `region-1-to-region-3 bounded relation`
- `region-3-to-region-2 bounded relation`
- `region-2-to-region-3 bounded relation`
- `shared reference-mediated relation`
- `caveated inter-region relation`
- `blocked/deferred inter-region relation`
- `non-canonical/internal-only inter-region path`

## Richtungs- und Semantikgrenzen

- Region 3 informiert Region 1 und Region 2 nur advisory-only.
- Region 1 und Region 2 informieren Region 3 nur advisory-only.
- Shared coupling bleibt reference/context mediated und ist keine direkte Authority.
- Caveated/deferred/blocked bleiben voneinander getrennt:
  - caveated != strong relation authority
  - deferred != blocked
  - blocked != failed execution

## No-direct-* Grenzen (hart)

- no direct action selection
- no direct execution trigger
- no direct retry trigger
- no direct memory commit
- no direct compute invocation
- no safety override
- no broad inter-region platform
- no fourth-region opening

## Runtime / Selection / Reference Rückbindung

- Runtime darf die Relation nur als bounded advisory classification sehen.
- Selection darf die Relation nur als bounded advisory classification sehen.
- Reference/Context darf nur bounded mediated support liefern.
- Es entsteht keine neue autonome inter-region Schicht und keine zweite operative Wirklichkeit.

## Bounded dynamics

- Für diese Relation nicht führend: dynamics bleibt, falls vorhanden, advisory-only Nebenpfad.
- Keine HH-Produktivintegration und keine direkte Steuerung über dynamics.
