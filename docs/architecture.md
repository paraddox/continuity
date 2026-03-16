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

## Typed Memory Ontology

Continuity keeps a typed memory ontology that stays small, explicit, and
Hermes-driven. The v1 memory classes are:

- `preference`
- `biography`
- `relationship`
- `task_state`
- `project_fact`
- `instruction`
- `commitment`
- `open_question`
- `ephemeral_context`
- `assistant_self_model`

Each memory class owns one host-visible contract surface:

- which subject kinds and applicability scopes it may attach to
- which evidence sources may produce the claim
- which locus prefix and aggregation mode it belongs to
- which admission outcome it defaults to before any durable publication
- whether it may later promote into durable belief
- which decay mode applies over time
- which prompt rendering style the host should use

This keeps user memory, shared context, assistant memory, and ephemeral state
explicitly partitioned. `preference` and `biography` stay in user memory,
`task_state` and `project_fact` stay in shared context, `assistant_self_model`
stays partitioned in assistant memory, and `ephemeral_context` remains
prompt-only session state by default.

## Versioned Policy Packs

Continuity also keeps versioned policy packs so memory behavior is explained by
policy version instead of hidden defaults. The first policy pack is
`hermes_v1`.

Each policy pack stamps a concrete policy version on host-facing decisions and
owns:

- the active ontology and enabled memory classes
- admission defaults and write budgets by memory partition
- retrieval ordering and prompt rendering rules
- utility weights for ranking and later replay analysis
- a replay-comparable policy fingerprint for policy version comparisons

That makes prompt rendering, utility weighting, admission choices, and replay
inspection transport-neutral and inspectable. A host can say which policy
version produced a memory decision instead of reverse-engineering behavior from
scattered module defaults.

## Memory Admission Gate

Admission is an explicit gate between candidate memory and durable claim
publication. v1 keeps the outcome set small and closed:

- `discard`
- `session_ephemeral`
- `prompt_only`
- `needs_confirmation`
- `durable_claim`

Admission decisions must stay explainable in terms of explicit thresholds and
budgets rather than opaque salience scoring alone. The gate records:

- evidence threshold
- novelty threshold
- stability threshold
- salience threshold
- per-partition write budgets
- utility signals that can refine a tie but cannot replace hard policy rules

Only `durable_claim` may publish into the claim ledger. `session_ephemeral` and
`prompt_only` keep candidate material useful without creating durable claims.
`needs_confirmation` keeps the candidate out of the claim ledger and hands it
off to the resolution queue for follow-up before durable promotion.

Admission differs from later belief revision, disclosure, forgetting, and
tiering. It decides whether a candidate memory may become durable at all. Every
admission decision should be attributable enough for replay, debugging, and
policy evaluation, including the write budgets and threshold gaps that blocked
durable promotion.

## Resolution Queue

Unresolved memory state becomes explicit work in a resolution queue rather than
remaining a dead-end label on a claim candidate. The v1 queue can be created
from:

- `needs_confirmation`
- `needs_followup`
- `open_question`
- `stale-on-use`
- conflicted loci

Queue priority remains policy-first. Base priority decides the main ordering,
and utility weights only break ties within the same policy band. That keeps the
queue inspectable and prevents utility from replacing explicit follow-up rules.

Queue items may surface through:

- `prompt_queue`
- `host_api`
- `inspection`

Those surfaces must not publish durable memory directly. They expose pending
work without polluting the claim ledger.

Resolution actions are explicit:

- confirm
- correct
- discard
- keep ephemeral
- promote to durable claim

Every resolution records downstream effects for admission, belief revision,
outcome recording, and replay. That makes it possible to explain why a follow-up
was queued, surfaced, deferred, or resolved, and how the final action changed
the memory pipeline.

## Host-Visible Artifacts

No durable derived memory artifact exists outside the claim ledger. Host-visible
artifacts are compiled views over claims, not independent durable roots.

Every host-visible artifact must carry claim provenance. Profile and card views
are projections over active claims, and dialectic answers may include
supporting claim and observation payloads, but none of those artifacts may
exist without claim provenance back to the ledger.
