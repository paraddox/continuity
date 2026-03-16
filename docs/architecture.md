# Continuity Architecture Invariants

## Canonical Memory IR

Continuity freezes one canonical memory IR for the engine:

- a `subject graph` of canonical identities with auditable aliases and
  merge/split history
- `immutable observations` as normalized source records
- explicit `admission` decisions before any candidate memory can become durable
- typed claims linked to canonical subjects and memory loci

Typed claims are the only durable derived primitive. Beliefs, profiles, cards,
prompt blocks, answers, and other compiled views are projections over claims
rather than separate durable roots.

## Subject Graph

Subjects answer "who or what is this memory about?" The subject graph keeps:

- a canonical `subject_id`
- typed subject kind
- auditable aliases backed by source observations
- merge history for mistaken identity collapse
- split history for later identity separation

Alias matching is normalized for comparison, but the original alias string and
its source observations stay attached for auditability.

## Immutable Observations

Observations are immutable normalized source records. They record what happened,
when it was observed, which session it came from, and which canonical subject
authored the source content. They are append-only inputs to later admission and
claim derivation.

## Claims And Admission

Candidate memory must pass through explicit admission before it can publish as a
durable claim. Only `durable_claim` admission outcomes may create claim-ledger
entries.

Claims remain:

- append-only
- typed
- scoped for applicability
- separate from audience or disclosure policy
- linked to claim provenance through source observation ids and optional
  derivation-run ids
- timestamped with `observed_at`, `learned_at`, `valid_from`, and `valid_to`
  where applicable

Claim relations encode the core revision edges: support, supersession,
contradiction, and correction.

## Memory Loci

Memory loci answer "which memory slot or evolving state is this claim part of?"
Every claim resolves to a canonical `(subject_id, locus_key)` address. Loci keep
the stable address, applicability scope, default disclosure policy,
`conflict_set_key`, and explicit aggregation mode.

This keeps conflict resolution, retrieval, and later belief projection
locus-scoped instead of forcing the engine to scan undifferentiated claims.

## Belief Revision

Beliefs are projections over claims grouped by subject and locus, not separate
durable roots. Current belief and historical claim history stay separate so the
engine can revise what it currently relies on without deleting evidence-backed
claim history.

Revision rules stay explicit:

- contradictory surviving claims do not silently coexist as equally active truth
- explicit corrections and supersessions remove older claims from current belief
  without erasing the older history
- stale beliefs may decay into qualified output while keeping their underlying
  claim history available
- each locus resolves according to its aggregation mode
- retrieval should prefer active beliefs over merely historical claims

## Epistemic Status

Continuity exposes explicit epistemic status instead of flattening every result
into asserted truth. The v1 status set is:

- `supported`
- `unknown`
- `tentative`
- `conflicted`
- `stale`
- `needs_confirmation`

These statuses may attach to claims, locus resolutions, compiled views, and
answers.

Answer and prompt behavior follows the status:

- `supported` may assert and include normally
- `tentative` and `stale` must qualify rather than overstate certainty
- `unknown` and `conflicted` should abstain instead of asserting unsupported
  memory
- `needs_confirmation` should ask for confirmation before publishing as settled
  memory

`prompt_view` should qualify tentative or stale memory, and suppress unknown,
conflicted, or needs-confirmation state by default. Replay should be able to
inspect whether the engine qualified, abstained, or over-asserted.

## Host-Visible Artifacts

No durable derived memory artifact exists outside the claim ledger. Host-visible
artifacts are compiled views over claims, not independent durable roots.

Every host-visible artifact must carry claim provenance. Profile and card views
are projections over active claims, and dialectic answers may include
supporting claim and observation payloads, but none of those artifacts may
exist without claim provenance back to the ledger.
