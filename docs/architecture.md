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

## Disclosure / Audience Layer

Disclosure is explicit and transport-neutral. Scope differs from audience:
`scope` says where a claim may apply, while disclosure says who may see it,
through which host channel, for which purpose, and in what transformed form.

The v1 audience principals are:

- `assistant_internal`
- `current_user`
- `current_peer`
- `shared_session`
- `host_internal`

Viewer kinds stay explicit as well:

- `assistant`
- `user`
- `peer`
- `host`

The v1 disclosure channels and purposes stay small and inspectable:

- `prompt`
- `answer`
- `search`
- `profile`
- `evidence`
- `replay`
- `migration`
- `inspection`

Disclosure actions are also explicit:

- `allow`
- `summarize`
- `redact`
- `withhold`
- `needs_consent`

Claim-level defaults, locus-level defaults, and compiled-view overrides compose
into one effective disclosure policy for a host read. Composition may only
narrow audience reach, channels, purposes, or transform actions; it should not
silently widen disclosure relative to the underlying claim contract.

`current_peer` is cross-peer-sensitive by default. A read compiled for one peer
must not leak to another peer merely because the underlying claim is relevant.
That makes cross-peer leakage a direct contract violation rather than a ranking
bug.

Every host-facing read should be explainable in terms of:

- the effective policy name and version stamp
- the audience principal and viewer kind
- the channel and purpose
- whether the content was exposed directly, summarized, redacted, withheld, or
  gated on consent
- why the final transform happened

Turn artifacts and replay should capture disclosure decisions explicitly, and
redaction or withholding should emit inspectable utility-facing events such as
`redacted` or `withheld`.

## Forgetting / Retraction / Erasure Contract

Forgetting is distinct from both correction and disclosure. The v1 withdrawal
modes are:

- `supersede`
- `suppress`
- `seal`
- `expunge`

Their semantics stay explicit:

- `supersede` is revision, not erasure; historical content remains recoverable
  for audit and replay
- `suppress` withdraws the memory from normal prompt, answer, search, profile,
  and retrieval flows while retaining bounded auditable content
- `seal` removes host-visible payloads, snapshots, and replay inputs while
  keeping only minimal administrative traceability
- `expunge` removes recoverable content from the claim ledger, observation log,
  indexes, snapshots, caches, archives, imports, and future derivations, while
  leaving only minimal tombstones if policy requires

Targets may be claims, loci, subjects, sessions, imported artifacts, or derived
views. The contract must say what survives in the claim ledger, observation
log, vector index, snapshot store, prefetch cache, replay artifacts, archive
tiers, import pipeline, derivation pipeline, and tombstones ledger for each
mode.

`suppress`, `seal`, and `expunge` all withdraw host-facing reads, but they do
not retain the same thing:

- suppress keeps recoverable content for audit
- seal keeps only administrative metadata
- expunge keeps only non-recoverable tombstones

Replay avoid resurrecting expunged content, and the same resurrection guard
must cover imports, replay artifacts, derivation pipelines, caches, archives,
and vector indexes. That is how append-only auditing can coexist with explicit
withdrawal without silently reintroducing removed memory later.

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

## Outcome Ledger and Utility Ledger

Continuity records downstream feedback in an explicit outcome ledger instead of
burying it in ad hoc notes or implicit counters. Outcome records are distinct
from epistemic status: `supported`, `tentative`, `conflicted`, and similar
states describe what the system currently believes, while the outcome ledger
describes what happened after the memory was used. This outcome ledger remains
distinct from epistemic status even when both are attached to the same claim or
compiled view.

The v1 outcome ledger stays small and closed:

- `prompt_included`
- `answer_cited`
- `user_confirmed`
- `user_corrected`
- `stale_on_use`

Each outcome record must keep:

- a concrete target such as a claim, compiled view, answer, or resolution item
- a policy stamp and timestamp
- explicit rationale
- claim provenance whenever the target can affect durable memory interpretation
- replay capture so counterfactual runs can inspect the same downstream facts

The utility ledger is compiled from explicit outcome records rather than from
heuristic guesses. v1 compiles the following utility signals and weights:

- `prompt_inclusion`
- `answer_citation`
- `user_corrected`
- `stale_on_use`

Compiled utility weights are deterministic, attributable, and distinct from raw
outcomes or epistemic status. They give admission, prompting, resolution,
retention, and replay one inspectable value model while keeping policy order
primary and using utility only where the policy pack explicitly allows it.

## Incremental Memory Compiler

The incremental memory compiler makes recompilation explicit instead of hiding
it behind scans or opportunistic cache refreshes. The compiler tracks
fingerprints for four separate categories of nodes:

- `source_input` nodes such as observations, imported artifacts, admission
  rules, policy packs, adapter versions, and other upstream inputs
- `derived_ir` nodes such as canonical subjects, claims, loci, and claim
  relations
- `utility_state` nodes such as compiled utility weights that may reprioritize
  prompt assembly, queue order, retention, or rebuild urgency
- `compiled_artifact` nodes such as `state_view`, `timeline_view`,
  `set_view`, `profile_view`, `prompt_view`, `evidence_view`, `answer_view`,
  and `vector_index_record`

The separation matters. Source inputs are authoritative upstream causes.
Derived IR remains the canonical intermediate representation. Utility state is
tracked independently so `utility_input_changed` can invalidate only the views
or indexes that actually depend on utility. Compiled artifacts remain
downstream outputs rather than a second durable root.

Dependency edges are explicit and typed. Content, projection, membership,
policy, utility, provenance, and index edges all remain inspectable so the
compiler can answer why a node is dirty and which path caused the rebuild.
Dirty reasons stay explicit rather than inferred from broad scans. The v1
reason set includes `source_edited`, `admission_policy_changed`,
`claim_corrected`, `subject_identity_changed`, `locus_membership_changed`,
`forgetting_changed`, `resolution_changed`, `utility_input_changed`,
`policy_upgraded`, and `adapter_changed`.

When a fingerprint changes, the compiler must:

- mark only the affected downstream nodes dirty
- preserve the dependency path that explains each dirty node
- keep rebuild planning deterministic and subject- plus locus-scoped
- rebuild in dependency order instead of broad invalidation waves

That gives Continuity an inspectable answer to questions like "why did this
prompt change?", "which locus was affected?", and "did this policy_upgraded
event require a full rebuild or only prompt-facing views?"

## Replay Artifacts

Continuity keeps one canonical replay artifact format for counterfactual replay.
The v1 replay artifact is versioned as `replay_v1` and captures:

- the stable replay artifact id
- the source transaction and waterline that produced the baseline artifact
- deterministic replay inputs pinned to a snapshot id, journal position,
  arbiter lane position, host surface, viewer context, and stable references
- the baseline policy fingerprint and per-step strategy fingerprints
- the stable output references and metric scores used for comparison

Deterministic replay inputs are explicit and must not depend on hidden runtime
state. The required input bundle includes:

- the frozen snapshot id used for the read or transaction
- the journal cut and arbiter cut that bound the authoritative source history
- the host surface being replayed such as `prompt_view` or `answer_view`
- the disclosure viewer, audience, channel, and purpose context
- stable claim, observation, compiled-view, outcome, and other reference ids
- optional query text when the host surface depends on user wording

Counterfactual replay reuses the same deterministic replay inputs for every
alternate run. Only the policy fingerprint or the retrieval, belief, and
reasoning strategy fingerprints may vary across compared runs.

Policy version comparison is explicit. Each replay run records its own policy
fingerprint so the engine can compare baseline and alternate policy versions
without pretending they were the same runtime decision.

Replay comparison is read-only. Counterfactual replay may inspect alternate
retrieval, belief, and reasoning outcomes, but it must not mutate authoritative
claims, views, snapshots, or utility state. Any derived comparison output is a
new replay artifact or evaluation result, not a publication into the source of
truth memory.

## Host-Visible Artifacts

No durable derived memory artifact exists outside the claim ledger. Host-visible
artifacts are compiled views over claims, not independent durable roots.

Every host-visible artifact must carry claim provenance. Profile and card views
are projections over active claims, and dialectic answers may include
supporting claim and observation payloads, but none of those artifacts may
exist without claim provenance back to the ledger.

## Compiled View Algebra

Continuity exposes a small, named compiled-view algebra instead of treating
reads as one generic materialization path. The v1 view kinds are:

- `state_view(subject, locus)` for current resolved belief state
- `timeline_view(subject, locus)` for ordered claim and revision history
- `set_view(subject, locus)` for loci that intentionally keep multiple active
  items
- `profile_view(subject)` for peer-card and profile projections
- `prompt_view(session, peer, policy)` for prompt-ready memory assembly
- `evidence_view(target)` for provenance behind a claim, belief, or answer
- `answer_view(query, scope)` for dialectic answers assembled from the other
  view kinds

Each view contract stays transport-neutral and explicit about:

- its input boundaries
- which other view kinds it may compose over
- deterministic and cacheable behavior
- snapshot binding
- provenance surface
- disclosure purposes
- default hot, warm, or cold tier inclusion

`prompt_view` and `answer_view` compose over `state_view`, `timeline_view`,
`set_view`, `profile_view`, and `evidence_view` instead of reaching around the
view layer directly. `evidence_view` is the richest provenance surface and must
expose both claim ids and supporting observation ids. Other host-visible views
still require claim provenance even when they do not surface raw observations by
default.

## Budgeted Prompt Planner

Continuity treats `prompt_view` assembly as a deterministic packing problem.
The planner works from:

- a hard token budget
- optional soft sub-budgets for fragment families such as evidence
- candidate fragments produced by prompt-eligible source views
- policy ordering and utility weights
- disclosure transforms and epistemic exposure rules

The planner must expose inclusion and exclusion reasons for every candidate
fragment. It also records disclosure transformation reasons like
`redacted_for_peer` or `withheld_requires_consent`, and it records degradation
ladder outcomes when a fragment is compressed instead of dropped.

The first degradation ladder is intentionally small and explicit:

- prefer full fragments while they fit
- compress eligible fragments when the full form would break a soft or hard
  budget
- drop fragments only after compression fails

Examples of deterministic degradation reasons include `collapsed timeline` and
`dropped low-priority evidence`.

Prompt exposure remains status-aware:

- `supported` fragments may include normally
- `tentative` and `stale` fragments must be qualified
- `unknown`, `conflicted`, and `needs_confirmation` fragments are suppressed by
  default

This keeps prompt packing bounded, explainable, and replayable under one
inspectable contract.

## Snapshot Consistency Layer

Continuity never serves mixed-state memory. Every retrieval, prompt assembly,
answer query, prefetch warm, and replay run pins to one immutable snapshot for
the whole operation.

Each snapshot carries:

- one `snapshot_id`
- one optional parent snapshot lineage pointer
- one policy stamp
- explicit references to compiled view artifacts and vector-index artifacts
- the transaction boundary that produced it

The read contract stays small and explicit:

- retrieval runs against one snapshot
- prompt assembly runs against one snapshot
- replay runs against one snapshot
- hosts read `current`
- compiler work writes a candidate snapshot first

Publication is inspectable because the active head and each candidate snapshot
are distinct references. Promotion from a candidate snapshot to the active head
is explicit, diffable, and reversible through rollback to an earlier snapshot
without mutating the snapshot payloads themselves.

This keeps forgetting, sealing, expunge, and rebuild publication atomic at the
snapshot boundary instead of leaking partially rebuilt or partially withdrawn
artifacts into host reads.

## Generational Memory Tiering Layer

Continuity uses a small, policy-driven generational tier model:

- `hot` for current working context, active commitments, and prompt-adjacent
  artifacts
- `warm` for stable durable memory that should remain available in ordinary host
  reads
- `cold` for recallable long-tail evidence and superseded material that should
  not dominate default reads
- `frozen` for replay records, snapshot history, and audit-heavy archival
  artifacts

Tiering starts only after a candidate memory passes durable admission. It does
not change source-of-truth claim semantics; it only governs retrieval defaults,
snapshot residency, rebuild urgency, and retention pressure.

The tier policy remains inspectable about:

- default claim-type placement
- compiled-view default tiers
- archival artifact placement
- promotion and demotion edges
- utility-driven promotion, demotion, and pruning bias
- expunge reachability for `cold` and `frozen` artifacts

`hot` and `warm` tiers are included in active host-visible reads by default.
`cold` is recallable without bloating normal reads, and `frozen` remains
archival-only unless audit or replay workflows ask for it directly.

## Memory Transaction Pipeline

Continuity treats runtime behavior as a closed set of named transactions rather
than ad hoc helper calls. The v1 transaction set is:

- `ingest_turn`
- `write_conclusion`
- `forget_memory`
- `import_history`
- `compile_views`
- `publish_snapshot`
- `prefetch_next_turn`

Each transaction keeps one deterministic phase order even when the caller only
waits for an earlier waterline. The canonical phase vocabulary is:

- normalize observations
- commit observations
- resolve subjects
- derive candidates
- run admission
- record non-durable context
- assign loci
- commit claims
- resolve forgetting
- revise beliefs
- compile views
- refresh utility
- capture replay
- publish snapshot
- prefetch

The main transaction boundaries stay explicit:

- `ingest_turn` runs the full turn pipeline from normalize observations through
  prefetch, with admission deciding whether candidate memory becomes durable,
  remains prompt-only, or stays session-ephemeral.
- `write_conclusion` records an explicit host write, then continues through
  claim commit, view compilation, snapshot publication, and optional next-turn
  prefetch.
- `forget_memory` resolves the requested forgetting mode first, then applies
  durable visibility changes, recompiles affected views, captures replay, and
  publishes a replacement snapshot.
- `import_history` follows the same canonical ingest spine without inventing a
  migration-only write path.
- `compile_views` and `publish_snapshot` stay available as explicit runtime
  entrypoints instead of side effects hidden inside reads.
- `prefetch_next_turn` is its own transaction path so the engine can warm
  prompt-facing caches without redefining the completion contract of a mutating
  operation.

`writeFrequency` is therefore a timing policy over the `ingest_turn`
transaction, not a second write implementation. `async` runs every turn and may
return at `observation_committed`, `turn` waits through `snapshot_published`,
`session` defers the full flush to session end, and integer `N` means the same
transaction is batched on a turn threshold instead of inventing a new runtime
path.

## Durability Contract

Continuity expresses completion semantics through a small ordered waterline set:

- `observation_committed`
- `claim_committed`
- `views_compiled`
- `snapshot_published`
- `prefetch_warmed`

These waterlines mean:

- `observation_committed`: source observations and transaction metadata are
  durably stored.
- `claim_committed`: durable claim writes, locus assignment, and belief-side
  effects are durably stored.
- `views_compiled`: the required compiled views for the operation have been
  rebuilt.
- `snapshot_published`: the host-visible snapshot head for the operation is
  published.
- `prefetch_warmed`: next-turn caches pinned to the active snapshot are ready.

Host-facing operations stay explicit about the minimum waterline they require:

- `save_turn` depends on `writeFrequency`: `async` awaits
  `observation_committed`, `turn` awaits `snapshot_published`, `session`
  returns after `observation_committed` while the full publish waits for the
  session flush, and integer batching returns after `observation_committed`
  until the threshold turn flushes.
- `write_conclusion` and `import_history` require at least `views_compiled` so
  durable writes do not return before the affected compiled views exist.
- `forget_memory` uses mode-specific guarantees: `supersede` requires
  `views_compiled`, while `suppress`, `seal`, and `expunge` require
  `snapshot_published` so withdrawn material is absent from subsequent host
  reads.
- prompt-facing reads require `snapshot_published`.
- `prefetch_next_turn` is best-effort relative to mutating operations and only
  defines `prefetch_warmed` when the caller explicitly invokes the prefetch
  transaction itself.

This keeps async behavior precise: callers know which waterline was awaited,
which phases may still be running after return, and which guarantees remain the
same for embedded mode now and a daemon wrapper later.

## Continuity Service Facade

Continuity exposes one transport-neutral Continuity service facade. The
canonical request/response contract is a typed envelope around named host
operations such as `initialize`, `save_turn`, `search`, `get_state_view`,
`get_timeline_view`, `get_profile_view`, `get_prompt_view`,
`answer_memory_question`, `forget_memory`, `write_conclusion`,
`list_memory_follow_ups`, `resolve_memory_follow_up`, `import_history`,
`publish_snapshot`, `resolve_subject`, `inspect_evidence`,
`inspect_admission`, `inspect_resolution_queue`, `inspect_disclosure`,
`inspect_forgetting`, `inspect_epistemic_status`, `record_outcome`,
`inspect_outcomes`, `inspect_utility`, `inspect_turn_decision`,
`inspect_policy`, `inspect_compiler`, `inspect_snapshot`, and
`inspect_tiers`.

The request contract stays intentionally small:

- operation name
- transport-neutral payload only
- optional disclosure context for host-facing reads
- optional target snapshot binding
- optional requested durability waterline for mutating operations

The response contract stays equally small:

- operation name
- transport-neutral payload only
- optional active snapshot id
- optional reached durability waterline
- optional replay artifact references

This boundary freezes before host integration work spreads. Python objects such
as repository handles, SQLite connections, threads, vector clients, or other
transport-specific runtime details must not leak across the canonical service
contract.

## Reasoning Adapter Extension Contract

The first shipped reasoning adapter is the Codex-backed `CodexAdapter` using
the `hermes_v1` policy pack, but that adapter is not a privileged engine path.
Future reasoning adapters must extend the same frozen `ReasoningAdapter`
contract rather than creating host-specific or provider-specific side channels.

The extension surface is intentionally closed to four operations:

- `answer_query`
- `generate_structured`
- `summarize_session`
- `derive_claims`

Every future reasoning adapter must consume the same transport-neutral request
DTOs and return the same transport-neutral response DTOs that the current
adapter uses. Provider SDK clients, model names, prompt-shaping details,
retries, and auth concerns stay behind the adapter boundary rather than leaking
into the service facade or compiled-view contracts.

Reasoning writes remain schema-hard even when the implementation changes.
`generate_structured` and `derive_claims` outputs must pass schema validation
before they may publish authoritative mutations. Observation capture may still
commit when an adapter response is malformed, but failed schema validation must
block claim publication, belief updates, and compiled-view mutation.

Adapter-specific identity belongs in replay and compiler metadata, not in the
typed host API contract. Strategy ids, fingerprints, and adapter versions may
change to support future reasoning adapters, but those changes must not widen
the service request/response envelope or add provider-specific payload fields.

## Deployment Modes And Process Boundary

Continuity keeps one engine and two shells:

- `embedded` is the v1 default for the internal Hermes patch
- `daemon` is reserved for a later local wrapper

The shell boundary is explicit. The engine owns transaction entrypoints,
durability waterlines, snapshot consistency, replay artifacts, disclosure
decisions, and the transport-neutral service contract. The embedding host or
daemon shell owns process lifecycle, request transport, and scheduling around
the same engine contract.

Embedded and daemon modes must keep identical semantics for:

- transaction entrypoints
- durability waterlines
- snapshot consistency
- replay artifacts
- disclosure decisions

Transport is the only intended difference:

- `embedded` uses direct in-process calls
- `daemon` uses Unix domain sockets when it is added later
- both modes remain local-only
- hosted service assumptions stay out of scope

SQLite ownership rules are explicit in v1:

- one owning Hermes process per Continuity SQLite store
- one serialized commit lane
- in-process worker threads only
- no multi-process write coordination in v1

The daemon shell must use the same typed request/response contract as embedded
mode. It must not add daemon-only service operations, daemon-only payload
fields, or daemon-only semantic branches. Transport concerns such as Unix
domain socket paths, process supervision, startup ordering, and retries belong
to the shell and must not leak into the transport-neutral engine contract.

The future daemon wrapper keeps the same single-owner semantics, but the owner
process becomes the daemon instead of Hermes. That changes process lifecycle and
transport, not memory semantics.

## Mutation Arbiter

Continuity routes every authoritative publication through one serialized commit
lane instead of letting concurrent workers mutate host-visible state directly.

The v1 boundary between off-lane computation and authoritative publication
stays explicit. Allowed off-lane computation is limited to:

- `embedding_generation`
- `claim_derivation`
- `view_compilation`
- `prefetch_preparation`

Those computations may run on worker threads, but they must publish back
through the mutation arbiter before they change authoritative state.

The v1 authoritative publication set is:

- `observation_commit`
- `claim_commit`
- `belief_revision`
- `forgetting_publication`
- `view_publication`
- `work_status_transition`
- `snapshot_head_promotion`
- `durability_signal`
- `outcome_recording`

Each arbiter publication carries:

- a positive lane position
- the originating transaction and phase
- the affected authoritative object ids
- any reached durability waterline
- replay-visible arbiter order metadata

`snapshot-head promotion` and `durability-waterline completion signaling` stay
on the same serialized commit lane as other authoritative publication. Replay
and debugging should refer to arbiter order rather than inferred
interleavings.

## System Event Journal

The system event journal is append-only and records authoritative publication in
journal order rather than forcing reconstruction from mutable table state
alone.

The v1 event type set is:

- `observation_ingested`
- `claim_committed`
- `belief_revised`
- `memory_forgotten`
- `view_compiled`
- `snapshot_published`
- `outcome_recorded`

Each journal entry links back to the originating arbiter lane position so
journal order and arbiter order can both be inspected during replay, crash
recovery, and debugging.

Payload handling is explicit:

- `inline` stores small control-plane facts directly in the journal entry
- `reference` stores stable identifiers that point at larger artifacts
- `mixed` stores a compact summary inline plus stable references

Every journal entry keeps:

- an ordered journal position
- event type and transaction kind
- the originating arbiter lane position
- the affected authoritative object ids
- durability-waterline context where relevant
- enough payload or references for reconstruction and debugging
