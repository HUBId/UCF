# BlueBrain first inter-region relation line (Serie BB26 / Prompt 4)

Status: **first bounded relation line** between the already integrated Region 1 and Region 2 surfaces.  
Scope remains strictly **two regions only** and **advisory-only**.

## Canonical relation map (bounded)

The canonical first inter-region relation map contains exactly:

- `region-1-to-region-2 bounded relation`
- `region-2-to-region-1 bounded relation`
- `shared reference-mediated relation`
- `caveated inter-region relation`
- `blocked/deferred inter-region relation`
- `non-canonical/internal-only inter-region path`

No third region class is introduced.

## Direction and semantics

- Region 1 can inform Region 2 advisory-only via existing runtime/selection/reference contract surfaces.
- Region 2 can inform Region 1 advisory-only via the same bounded surfaces.
- Shared inter-region coupling is reference/context mediated only.
- Relation can be asymmetric by state (for example caveated, deferred, or blocked), without creating full bilateral coupling.

## Caveat/deferred/blocked boundaries

- caveated inter-region relation: quality-limited advisory signal, not execution authority.
- deferred inter-region relation: delayed/awaiting context state, not equivalent to blocked.
- blocked inter-region relation: explicitly unavailable path, not equivalent to failed execution.
- shared reference-mediated relation: bounded context/reference bridge, not direct region-to-region authority.

## No-direct-* and scope guard rails

The two-region relation explicitly enforces:

- no direct action selection
- no direct execution trigger
- no direct retry trigger
- no direct memory commit
- no direct compute invocation
- no safety override
- no general region-to-region decision authority
- no broad inter-region platform layer

## Runtime / Selection / Reference visibility

- Runtime may consume the relation only as bounded advisory classification.
- Selection may consume the relation only as bounded advisory classification.
- Reference/Context may mediate relation state, but does not become an action authority chain.
- The relation is attached to existing canonical BlueBrain lines and does not create a second operational reality.

## Bounded dynamics posture

For this step, bounded dynamics stays non-leading and optional:

- no HH production integration
- no dynamics-led relation authority
- relation semantics remain runtime/selection/reference anchored
