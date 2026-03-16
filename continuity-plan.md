# Plan: Continuity

**Generated**: 2026-03-16
**Estimated Complexity**: High

## Overview

Build `Continuity`, a local-first memory system that reproduces the subset of Honcho functionality Hermes actually uses, without depending on Honcho's hosted or self-hosted server architecture.

The functional target is Hermes compatibility at the behavior layer:
- session-backed memory
- user and AI peer modeling
- prompt-ready context retrieval
- semantic memory search
- profile / peer-card retrieval
- dialectic-style memory answers
- conclusion writes
- migration of local memory files and prior session history
- recall modes and write-frequency controls
- async prefetch for next-turn continuity

The implementation target is not "clone Honcho internally." It is:
- `SQLite` as the durable source of truth
- `zvec` as the local semantic index / retrieval engine
- `Ollama` `nomic-embed-text` for embeddings
- a minimal reasoning adapter abstraction
- `Codex SDK` with `gpt-5.4` low reasoning as the first adapter

The project should be usable as a standalone local memory backend for Hermes and later adaptable to other agent runtimes.

The most important architectural rule is that memory must be auditable. Every durable claim, profile item, summary fragment, and dialectic answer should be traceable back to concrete source messages and derivation runs.

The second core rule is that memory must have one canonical intermediate representation. Raw observations are immutable inputs. Typed claims are the only durable derived primitive. Beliefs, profiles, summaries, prompt blocks, vector documents, and other host-facing outputs are compiled views over claims rather than co-equal durable roots.

The third core rule is that memory must be identity-resolved. Every claim and every locus should anchor to a canonical subject, not just a raw name string. Subject merges, splits, aliases, and mistaken identity resolution must be auditable and replayable.

The fourth core rule is that memory must be locus-addressable. Every claim should belong to a canonical memory locus: the stable address, slot, set, or state machine that claim competes within or contributes to. Revision, retrieval, compiler invalidation, and profile materialization should operate on loci rather than scanning undifferentiated claims.

The fifth core rule is that memory must be audience-aware. Every claim, compiled view, and host read should carry explicit disclosure semantics that answer who may see or use the memory, through which channel, for which purpose, and whether it should be shown directly, summarized, redacted, withheld, or gated on consent.

The sixth core rule is that memory must be revisable over time. Continuity should optimize for current belief state, not just historical accumulation.

The seventh core rule is that memory decisions must be replayable. Continuity should preserve enough turn-time artifacts to re-run retrieval, claim selection, belief selection, and reasoning later under alternate strategies.

The eighth core rule is that memory must be typed. Continuity should explicitly model what kind of memory each observation and claim represents so revision, retrieval, and rendering rules are type-aware rather than generic.

The ninth core rule is that memory behavior must be versioned. Continuity should bundle ontology, revision, retrieval, rendering, derivation, identity-resolution, and disclosure behavior into explicit policy versions that can be inspected, replayed, and evolved safely.

The tenth core rule is that memory must be incrementally compilable. Continuity should treat observations, subjects, claims, loci, audience policies, policy packs, and adapter behavior as inputs to a compiler that rebuilds only the affected compiled views when something changes.

The eleventh core rule is that memory reads must compile to explicit view types. Hosts should consume named compiled views such as state, timeline, set, profile, prompt, evidence, and answer views rather than a generic bag of materialized artifacts.

The twelfth core rule is that memory runtime behavior must follow explicit transaction pipelines. Saving a turn, writing a conclusion, importing history, compiling views, publishing a snapshot, and prefetching next-turn context should each have deterministic phase ordering and replayable boundaries.

The thirteenth core rule is that memory operations must have explicit durability waterlines. Each transaction and host API should declare the strongest commit point it guarantees before returning, such as `observation_committed`, `claim_committed`, `views_compiled`, `snapshot_published`, or `prefetch_warmed`.

The fourteenth core rule is that authoritative mutation must flow through a serialized arbiter. Concurrent ingest, conclusion writes, imports, rebuild publication, and snapshot-head changes should not race directly; they should publish through a single ordered commit lane.

The fifteenth core rule is that authoritative mutation should be journaled append-only. Every published state transition should emit a durable system event so ordering, reconstruction, replay, and forensic debugging do not depend on inferred state alone.

The sixteenth core rule is that memory must surface epistemic status explicitly. Claims, locus resolutions, compiled views, and answers should be able to say `unknown`, `tentative`, `conflicted`, `stale`, or `needs_confirmation` instead of always collapsing to a single asserted state.

The seventeenth core rule is that memory must record downstream outcomes explicitly. Corrections, confirmations, overrides, stale-on-use events, and user acceptance should become first-class outcome records rather than only ad hoc notes or replay scores.

The eighteenth core rule is that memory reads must be snapshot-consistent. Continuity should expose coherent, branchable memory snapshots so hosts never read a mixed partially rebuilt state.

The nineteenth core rule is that prompt assembly must be budgeted and explainable. `prompt_view` should be built under explicit token budgets using deterministic packing, degradation ladders, inclusion/exclusion traces.

The twentieth core rule is that memory must be tiered generationally. Continuity should separate hot, warm, cold, and frozen memory so retrieval quality, rebuild cost, storage growth, and prompt efficiency stay bounded over long runtimes.

## Design Direction

### Core Architecture

Use a split architecture:
- relational state in `SQLite`
- semantic retrieval in `zvec`
- reasoning in a pluggable SDK adapter
- an immutable `Observation Log` for normalized source messages and imports
- a first-class `Subject Graph` that resolves names and references into canonical subjects
- a typed, append-only `Claim Ledger` for provenance-linked memory assertions
- a first-class `Memory Locus Model` that groups claims into stable addresses, conflict sets, and aggregation modes
- a first-class `Disclosure / Audience Layer` that decides which claims and views may surface to which viewer, channel, and purpose, and whether they are exposed directly, summarized, redacted, or withheld
- a `Compiled View Algebra` that defines the host-visible read products built from claims
- a `Budgeted Prompt Planner` that turns candidate memory fragments into bounded `prompt_view` outputs
- an `Epistemic Status Layer` that expresses uncertainty, conflict, staleness, and abstention explicitly
- materializers for current beliefs, profiles/cards, summaries, prompt-ready memory blocks, evidence views, and answer views
- an `Evidence Graph` built over observations, claims, and compiled views
- a `Temporal Belief Revision Engine` that turns claims into current beliefs
- a `Counterfactual Replay Engine` that records and replays turn-time memory decisions
- a `Typed Memory Ontology` that defines memory classes and their lifecycle rules
- a `Versioned Memory Policy Layer` that defines named behavior packs such as `hermes_v1`
- an `Incremental Memory Compiler` that tracks dependencies, invalidation, and selective rebuilds from observations to claims to compiled views
- a `Memory Transaction Pipeline` that defines deterministic runtime phase ordering for ingest, derivation, compilation, publication, and prefetch
- a `Durability Contract` that defines which commit waterline each transaction and API guarantees
- a `Mutation Arbiter` that serializes authoritative state transitions and snapshot-head publication
- a `System Event Journal` that records authoritative published state transitions in append-only order
- an `Outcome Ledger` that records post-hoc confirmations, corrections, overrides, and downstream usefulness
- a `Snapshot Consistency Layer` that exposes atomic read states and candidate rebuild branches
- a `Generational Memory Tiering Layer` that governs admission, promotion, demotion, and archival behavior

This is the right fit because Hermes needs more than vectors:
- peers
- subjects
- subject_aliases
- subject_links
- sessions
- messages
- observations
- claims
- claim_loci
- claim_sources
- claim_relations
- audience / disclosure policy metadata
- claim derivation runs
- belief/locus/view state
- profile/card compiled views
- derivation runs
- provenance links
- claim validity windows
- conflict/supersession/correction metadata
- migration metadata
- artifact dependency metadata
- compilation state and dirty queues
- snapshot metadata and snapshot heads
- retention policies and tier transition state
- queue / prefetch state
- configuration and compatibility state

These belong in `SQLite`.

`zvec` should be treated as the retrieval index, not the canonical store. If the index becomes stale, it should be rebuildable from SQLite.

### Canonical Memory IR

Continuity should have one native memory object: the typed claim.

The model should be:
- `observations`: immutable normalized records of what happened, what was said, or what was imported
- `subjects`: canonical identities for users, assistants, peers, projects, repositories, files, or other addressable entities
- `claims`: append-only, typed assertions derived from observations or explicit host writes
- `compiled views`: current beliefs, peer cards, summaries, timelines, prompt blocks, vector documents, evidence views, answer views, and other host-facing projections over claims

Claims should carry the fields needed to support revision without mutating history:
- identity and claim type
- `subject_id`
- `locus_key`
- scope (`user`, `assistant`, `peer`, `session`, `shared`) describing where the memory may apply
- audience / disclosure policy reference describing who may consume the memory and how
- canonical value payload/schema
- provenance links to source observations and derivation runs
- confidence / support metadata
- `conflict_set_key`
- `aggregation_mode`
- contradiction, supersession, and correction edges
- `observed_at`
- `learned_at`
- `valid_from`
- `valid_to`

This is the subsystem boundary that simplifies everything else:
- belief revision becomes claim selection and projection instead of state mutation across multiple roots
- profiles, cards, summaries, prompts, and answer products become compiled views instead of special storage primitives
- compiler invalidation works on one IR instead of bespoke domain tables
- replay can compare claim sets and compiled views directly

Hermes-visible operations like conclusion writes or profile reads can still exist at the API boundary, but internally they should land in the claim ledger and then flow into compiled views.

### Subject Graph

Continuity should make subject identity resolution first-class, not implicit.

A subject is the canonical identity that answers: "who or what is this memory actually about?" Examples:
- the human user
- the assistant persona
- a specific peer
- a repository or project
- a file or environment target

The subject graph should define:
- canonical `subject_id`s
- `subject_aliases` for names, handles, nicknames, repo shorthands, and imported labels
- typed subject kinds such as `user`, `assistant`, `peer`, `project`, `repo`, `file`
- merge and split provenance when identity resolution changes over time
- subject links for relations like "same as", "part of", or "represents"

This is the foundation under the locus model:
- loci become stable as `(subject_id, locus_key)` rather than loose strings
- retrieval can resolve "who is this about?" before "what is current?"
- migration can import messy names into canonical subjects without corrupting downstream memory
- replay can compare alternate subject-resolution policies on the same turn history

### Memory Loci

Continuity should make memory loci first-class, not implicit.

A locus is the canonical address that answers: "which claims are actually about the same underlying memory slot or evolving state?" Examples:
- a user's current editor preference
- a project's default branch
- a peer relationship status
- an open-question set for a session

Each locus should define:
- `subject_id`: the canonical subject the memory is about
- `locus_key`: the stable address within that subject
- `scope`: where the memory is allowed to apply
- default audience / disclosure policy
- `conflict_set_key`: which claims compete directly
- `aggregation_mode`: how claims combine, such as `latest_wins`, `set_union`, `timeline`, or `state_machine`
- value schema / normalization rules for comparisons and rendering

This is the missing control surface for the claim ledger:
- belief revision can resolve current state per locus instead of globally scoring all claims together
- retrieval can ask for the best current state, a timeline, or a multi-valued set depending on locus semantics
- compiler invalidation can target affected loci and downstream compiled views precisely
- profiles and cards can materialize deterministic locus outputs instead of bespoke special cases

### Disclosure / Audience Layer

Continuity should make disclosure semantics first-class, not implicit.

A subject answers: "who or what is this memory about?" A locus answers: "which memory slot or evolving state is it part of?" A disclosure rule answers: "who may see or use it, through which channel, for which purpose, and in what form?"

This layer should define:
- audience principals such as `assistant_internal`, `current_user`, `current_peer`, `shared_session`, and `host_internal`
- disclosure channels and purposes such as `prompt`, `answer`, `search`, `profile`, `evidence`, `replay`, and `migration`
- disclosure actions such as `allow`, `summarize`, `redact`, `withhold`, and `needs_consent`
- claim-level defaults, locus-level defaults, and compiled-view overrides
- auditable decision traces for why a fragment was exposed, transformed, or withheld

This is the missing second axis of memory:
- retrieval can filter eligibility before it ranks relevance
- `prompt_view` can be safe by construction instead of relying on incidental ranking or post-hoc cleanup
- `answer_view` can distinguish `unknown` from `known_but_withheld`
- cross-peer, cross-session, and assistant-internal leakage become explicit, testable failure modes

### Compiled View Algebra

Continuity should define a small, explicit algebra of host-visible compiled views instead of treating all reads as one generic materialization flow.

The initial view set should be:
- `state_view(subject, locus)` for current resolved belief state
- `timeline_view(subject, locus)` for change history and revisions over time
- `set_view(subject, locus)` for multi-valued loci that intentionally retain more than one active item
- `profile_view(subject)` for peer-card or profile materialization
- `prompt_view(session, peer, policy)` for prompt-ready memory assembly
- `evidence_view(target)` for provenance and supporting claims behind a claim, belief, or answer
- `answer_view(query, scope)` for dialectic answers composed from the other views

Each view type should declare:
- its inputs and dependency boundaries
- its determinism and cacheability expectations
- its snapshot semantics
- its provenance surface
- its disclosure / audience contract
- its tier inclusion defaults

This gives the read side the same rigor as the write side:
- retrieval becomes view selection plus assembly rather than an open-ended ranking pipeline
- compiler invalidation can target explicit read products
- snapshots can define exactly which view artifacts are included
- the host API becomes a thin layer over named view contracts

### Budgeted Prompt Planner

Continuity should treat `prompt_view` assembly as a bounded packing problem, not just a retrieval afterthought.

The planner should work from:
- a hard token budget and optional soft sub-budgets
- candidate fragments from `state_view`, `profile_view`, `timeline_view`, `set_view`, and `evidence_view` that remain eligible after disclosure-policy filtering
- policy-defined fragment priorities and rendering styles
- host context such as active peer, task type, and recall mode

The planner should decide:
- which fragments are included
- which fragments are compressed or summarized
- which fragments are dropped first under pressure
- how evidence is thinned without losing critical current state

The planner should expose:
- final token estimate and actual token usage where available
- inclusion and exclusion reasons
- disclosure transformation reasons such as `redacted_for_peer` or `withheld_requires_consent`
- degradation ladder decisions such as "collapsed timeline" or "dropped low-priority evidence"
- the fragment set that actually reached the model

This is the missing prompt-time optimization layer:
- `prompt_view` becomes a deterministic bounded product instead of an unbounded composition
- replay can compare packing strategies, not just retrieval strategies
- policy packs can change prompt economics without changing the claim model
- Hermes compatibility can be evaluated under the same token constraints the real agent faces

### Memory Transaction Pipeline

Continuity should make runtime behavior explicit through a small set of named transaction pipelines.

The initial transaction set should be:
- `ingest_turn`
- `write_conclusion`
- `import_history`
- `compile_views`
- `publish_snapshot`
- `prefetch_next_turn`

Each transaction should define deterministic phase ordering. For example:
- normalize observations
- resolve subjects
- derive or accept claims
- assign loci
- revise beliefs
- compile affected views
- capture replay artifacts
- publish a candidate or active snapshot
- enqueue or perform prefetch when applicable

This is the runtime spine missing from the plan:
- `writeFrequency` becomes an execution policy over transaction timing rather than an ad hoc save toggle
- replay gains canonical boundaries for what happened in each operation
- compiler, snapshot, and prefetch subsystems stop feeling loosely coupled
- host integration can call stable transaction entrypoints rather than assembling side effects manually

### Durability Contract

Continuity should make completion semantics explicit through a small set of durability waterlines.

The initial waterline set should be:
- `observation_committed`
- `claim_committed`
- `views_compiled`
- `snapshot_published`
- `prefetch_warmed`

These should mean:
- `observation_committed`: source observations and basic transaction metadata are durably stored
- `claim_committed`: claims, subject/locus assignment, and belief-side effects for the operation are durably stored
- `views_compiled`: all required compiled views for the operation are durably updated
- `snapshot_published`: the host-visible snapshot head reflecting the operation is published
- `prefetch_warmed`: optional next-turn caches are prepared

Transactions and APIs should declare the strongest waterline they guarantee before returning. For example:
- `save_turn` in async mode may guarantee only `observation_committed`
- `write_conclusion` may require `claim_committed` or `views_compiled`
- host-facing prompt reads should require `snapshot_published`
- prefetch should remain best-effort and never be mistaken for the completion boundary of the mutating operation

This gives the runtime contract real precision:
- `writeFrequency` becomes about which waterline is awaited
- host integrations know exactly what is durable when an API returns
- replay can compare not only outcomes but commit boundaries
- async behavior stops being ambiguous

### Mutation Arbiter

Continuity should make authoritative mutation serialization explicit.

Not all work needs to happen on the serialized lane:
- embedding generation
- claim derivation
- view compilation
- prefetch preparation

But all authoritative publication should flow back through one arbiter:
- subject/claim/locus mutation
- belief-state publication
- dirty-queue transitions that change authoritative work status
- snapshot-head promotion
- durability-waterline completion signaling

The arbiter should provide:
- one ordered commit lane for authoritative mutations
- stable ordering metadata for replay and debugging
- a clear boundary between off-lane computation and on-lane publication
- a single place to enforce snapshot-head and durability invariants

This is the missing coordination layer:
- concurrent transactions stop competing to publish authoritative state directly
- rebuilds and foreground writes can coexist without corrupting heads or waterlines
- replay can refer to commit-lane order rather than inferring interleavings
- async behavior becomes operationally tractable rather than merely well-specified

### System Event Journal

Continuity should persist authoritative publication as an append-only journal, not just as the latest table state.

The journal should record event types such as:
- `observation_ingested`
- `claim_committed`
- `belief_revised`
- `view_compiled`
- `snapshot_published`
- `outcome_recorded`

Each event should capture:
- ordered journal position
- event type
- the authoritative object or objects affected
- transaction and arbiter metadata
- durability-waterline context where relevant
- enough payload or references to support reconstruction and debugging

This is the missing control-plane substrate:
- replay gains a system-level event history rather than only turn-local artifacts
- arbiter order becomes durable instead of merely observed
- crash recovery and forensic debugging get a stable reconstruction source
- migration and projection rebuilds can use the journal as ground truth for state transitions

### Epistemic Status Layer

Continuity should make uncertainty and non-commitment first-class outputs, not just internal scores.

The system should be able to represent statuses such as:
- `unknown`
- `tentative`
- `conflicted`
- `stale`
- `needs_confirmation`
- `asserted`

These statuses should attach to:
- claims when evidence is weak or contradictory
- locus resolutions when no clean current state exists
- compiled views when the output should surface ambiguity instead of flattening it
- answers when the safest behavior is to abstain, qualify, or ask for clarification

This layer should define:
- when to abstain instead of assert
- when to surface multiple candidates instead of choosing one
- when low-confidence memory should be kept out of `prompt_view`
- when the system should explicitly ask the user to confirm or correct something
- how replay scores calibration and abstention quality, not just correctness

This is the missing judgment layer:
- belief revision can distinguish "best current guess" from "safe to rely on"
- prompt packing can prioritize high-confidence state and exclude risky fragments
- answers can be more trustworthy without being artificially overconfident
- replay can evaluate not only recall quality but calibration quality

### Outcome Ledger

Continuity should record what later turned out to happen, not just what it believed at the time.

The system should be able to store outcome labels such as:
- `confirmed`
- `corrected`
- `contradicted`
- `stale_on_use`
- `helpful`
- `harmful`
- `needs_followup`
- `user_accepted`
- `user_overrode`

Outcome records should be attachable to:
- claims
- locus resolutions
- compiled views such as `prompt_view` or `answer_view`
- turn artifacts and replay runs

The outcome ledger should capture:
- what artifact or decision the outcome refers to
- who or what supplied the outcome signal
- when the outcome was observed
- whether the signal is direct user feedback, inferred downstream behavior, or explicit correction
- how the outcome should feed replay scoring and future policy tuning

This is the missing learning substrate:
- replay can be judged against real downstream outcomes instead of only ex ante heuristics
- policy tuning can optimize for usefulness and correctness over time
- epistemic calibration can be evaluated against later confirmation or override
- memory quality becomes a closed loop instead of just a replayable one

### Evidence Graph

Continuity should treat memory as a compiled artifact graph rooted in observations and claims, not just a bag of retrieved snippets.

That means:
- raw messages become normalized observations
- reasoning adapters derive typed claims from observations
- each claim stores provenance to one or more source observations
- profile/card items are materialized views over current active claims
- summaries and dialectic answers can cite supporting evidence
- claims can supersede, contradict, expire, or be corrected over time without silent corruption

This provides three major benefits:
- trust: the system can explain why it believes something
- maintainability: stale claims can be invalidated or re-derived surgically
- adapter portability: Codex-, Claude-, and OpenCode-derived claims can coexist with the same evidence model

The Evidence Graph should be a first-class storage and API concern, not a debugging afterthought. It should be implemented over observations, subjects, claims, loci, and locus-to-view edges rather than parallel durable fact tables.

### Temporal Belief Revision

Continuity should not treat all remembered claims as equally current.

It should maintain an explicit belief layer on top of the claim ledger:
- observations record what was said or seen
- claims express typed candidate memory assertions
- beliefs represent the system's current best materialized view per locus
- belief updates track promotions, demotions, supersessions, corrections, and expirations

This lets the system distinguish:
- currently believed claims
- stale but historically important claims
- contradicted claims
- corrected preferences
- claims that require re-derivation after new evidence

The revision engine should consider:
- freshness
- evidence strength
- validity windows
- locus aggregation mode
- recency of supporting messages
- contradiction or supersession edges
- source diversity
- explicit user corrections

This provides the highest-value improvement to memory quality:
- fewer stale assumptions
- better handling of changing preferences
- better “what changed?” answers
- safer long-running memory accumulation

Belief revision should resolve one or more current states per locus according to `aggregation_mode`, not globally rank all claims as if they shared the same semantics.

Belief revision must also respect subject identity: claims should compete within the right subject boundary first, then within the right locus.

### Counterfactual Replay

Continuity should not only store memory state. It should store how memory decisions were made on each turn.

For each answered turn, the system should preserve a canonical decision record containing:
- the retrieval query and retrieval candidates
- the subjects resolved for the turn
- the loci resolved for the turn
- the active claims selected for use
- the active beliefs selected for use
- the prompt-ready memory block that was assembled
- the reasoning adapter input/output envelope
- the final answer or structured result
- the evidence references attached to that result

This enables offline replay under alternate strategies:
- different retrieval ranking rules
- different belief revision policies
- different prompt assembly rules
- different disclosure and redaction policies
- different reasoning adapters

This provides the most compounding long-term leverage:
- memory quality becomes measurable instead of anecdotal
- retrieval and belief changes can be regression-tested against real turns
- adapter comparisons become apples-to-apples
- the system can learn from stale recalls, missed claims, and incorrect carry-forward assumptions

The replay engine should be a first-class evaluation and hardening subsystem, not an optional debug log.

### Typed Memory Ontology

Continuity should not treat all memories as the same kind of object.

It should define an explicit ontology for the claim classes Hermes-like agents actually rely on, such as:
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

Each claim type should carry explicit policies for:
- promotion into durable belief
- decay or expiration behavior
- contradiction and supersession handling
- retrieval priority
- prompt rendering style
- default disclosure posture
- acceptable evidence sources

This is the cleanest way to keep the belief engine principled:
- changing user preferences should not be handled like repository claims
- temporary task context should not be rendered like long-term biography
- open questions should not be promoted like settled beliefs
- assistant self-model updates should be partitioned from user memory

The ontology should be a core runtime primitive attached to claims. Without it, belief revision and retrieval risk collapsing into generic scoring heuristics.

### Versioned Memory Policy Layer

Continuity should not scatter memory behavior across modules as unversioned defaults.

It should define explicit policy packs, starting with something like:
- `hermes_v1`

Each policy pack should own:
- the active ontology and allowed memory types
- claim derivation and claim-normalization rules
- subject-resolution and alias-resolution rules
- locus-definition and locus-resolution rules
- disclosure and redaction rules by audience, channel, and purpose
- per-type belief revision rules
- retrieval and ranking profiles
- prompt rendering rules
- prompt budget and fragment-packing rules
- claim derivation settings
- migration compatibility rules where needed

This makes the rest of the architecture governable:
- belief updates can be explained in terms of a specific policy version
- replay can compare `hermes_v1` against future policy revisions directly
- Hermes compatibility can remain stable while new host-specific policies are added later
- heuristic changes stop leaking silently into storage, retrieval, and prompting logic

The policy layer should be thin but authoritative. It is the contract that turns the ontology, revision engine, retrieval logic, and replay system into one coherent memory product.

### Incremental Memory Compiler

Continuity should make recomputation explicit.

Treat these as source inputs:
- observations and imported files
- subject-resolution rules and alias maps where they affect stored artifacts
- policy packs and policy versions
- reasoning adapter versions and output schemas
- prompt-rendering rules where they affect stored artifacts

Treat these as canonical derived IR:
- subjects
- claims
- loci
- claim relations

Treat these as compiled views:
- `state_view` artifacts
- `timeline_view` artifacts
- `set_view` artifacts
- `profile_view` artifacts
- `prompt_view` artifacts
- `evidence_view` artifacts
- `answer_view` artifacts
- vector-index records built from view or claim outputs where needed

The compiler should track dependency edges and fingerprints for the inputs that produced each artifact. When a source changes, the system should:
- mark only affected artifacts dirty
- explain why they are dirty
- rebuild them selectively
- preserve prior versions or replayability where needed

This is the missing operational layer that makes the rest of the plan durable:
- user corrections trigger targeted rebuilds instead of broad recomputation
- policy upgrades can recompile memory under `hermes_v2` without corrupting prior runs
- adapter changes can invalidate only the artifacts that depended on old behavior
- “what changed?” becomes an answerable systems question, not just a semantic one

The compiler should be a first-class subsystem, not an implementation detail hidden inside prefetch or storage code. Dirtying and rebuild planning should be subject- and locus-aware so one corrected preference does not force unrelated memory slots through the same pipeline.

### Snapshot Consistency Layer

Continuity should never answer from a half-rebuilt memory state.

It should expose immutable memory snapshots, each representing one coherent read view over:
- `state_view` artifacts
- `timeline_view` artifacts
- `set_view` artifacts
- `profile_view` artifacts
- `prompt_view` artifacts
- `evidence_view` artifacts
- `answer_view` artifacts
- vector-index records
- any other compiled artifacts surfaced to hosts

Reads should pin to a snapshot:
- retrieval runs against one snapshot
- prefetch warms one snapshot
- `answer_query` and prompt assembly use one snapshot
- replay and policy experiments can branch new candidate snapshots without disturbing the active head

Compiler rebuilds should happen against a candidate snapshot, then promote atomically when complete. That gives the system a clean contract:
- hosts read `current`
- rebuilds write `next`
- diffs explain what changed between snapshots
- failed or partial rebuilds do not corrupt active reads

This is the production-safety layer for the rest of the plan. Without it, async rebuilds, policy upgrades, and prefetch can still leak mixed-state memory into prompts.

### Generational Memory Tiering Layer

Continuity should not treat all retained memory as equally expensive, equally urgent, or equally prompt-worthy.

It should define a small set of generational tiers, for example:
- `hot`: current working context, recent active beliefs, prompt-adjacent artifacts
- `warm`: durable preferences, commitments, stable project knowledge claims, peer cards
- `cold`: older observations, superseded beliefs, long-tail evidence that should remain recallable
- `frozen`: replay records, archival snapshots, historical artifacts retained mainly for audit and evaluation

Each tier should carry explicit rules for:
- admission and initial placement
- promotion and demotion
- retention and compression
- default retrieval priority
- rebuild urgency
- snapshot inclusion strategy

This gives the current architecture a scale model:
- prompt assembly stays sharp because `hot` and selected `warm` memory dominate by default
- compiler work can prioritize high-value artifacts first
- old evidence remains available without bloating active snapshots
- replay and auditing can keep rich history in `frozen` storage without polluting host-facing reads

The tiering layer should remain small and policy-driven. It is the economic control system for long-lived memory quality.

### Reasoning Layer

Do not hardwire Codex SDK calls into the business logic.

Define a minimal adapter interface for only the reasoning work this project needs:
- `answer_query`
- `generate_structured`
- `summarize_session`
- `derive_claims`

The first implementation is:
- `CodexReasoningAdapter`
- model: `gpt-5.4`
- reasoning: `low`

Future adapters:
- `ClaudeReasoningAdapter`
- `OpenCodeReasoningAdapter`

The abstraction should be only as broad as required for Hermes-used features. Do not design a full provider framework up front.

### Embeddings Layer

Use:
- `Ollama`
- model: `nomic-embed-text`

Embeddings should be generated once, stored durably in SQLite metadata as needed, and indexed into `zvec`.

If `zvec` persistence semantics ever prove insufficient for the full workload, the source of truth remains SQLite and the retrieval index can be rebuilt.

### How zvec Fits

`zvec` is useful here as the embedded retrieval engine:
- fast local vector similarity
- no separate vector service
- in-process access
- simpler deploy story than pgvector

It should not own the session / peer / claim domain model.

`zvec` should index evidence-bearing artifacts, not replace the graph that explains where those artifacts came from.

## Scope

### In Scope

- A local memory backend with the same Hermes-visible capabilities currently exercised through Honcho
- Hermes-compatible config concepts:
  - `memoryMode`
  - `recallMode`
  - `writeFrequency`
  - session naming
  - peer naming
- Local reasoning through Codex SDK
- Local embeddings through Ollama
- Semantic search through `zvec`
- File/history migration and AI identity seeding
- A small consumer-facing API for host integrations
- An immutable observation log and typed claim ledger as the canonical memory IR
- Evidence-backed, subject-scoped, locus-addressed claims, with provenance, supersession, correction, and validity tracking
- Explicit current-belief management with freshness and contradiction handling
- Append-only system event journaling for authoritative state transitions
- Explicit epistemic status on claims, resolutions, compiled views, and answers
- Explicit post-hoc outcome labels on claims, views, and decisions
- Token-budget-aware prompt planning with inspectable packing decisions
- Turn-level decision capture and offline replay for retrieval, belief, and reasoning evaluation
- Type-aware memory classification, policy enforcement, and prompt rendering
- Audience-aware disclosure, redaction, withholding, and consent behavior for prompt, answer, profile, search, and evidence surfaces
- Explicit policy-pack versioning for host-facing memory behavior
- Incremental invalidation and selective rebuild of derived memory artifacts
- Snapshot-consistent reads with branchable candidate snapshots
- Tier-aware retention, retrieval, and archival behavior

### Out of Scope

- Reproducing Honcho's full server/API architecture
- Reproducing Honcho's exact database schema
- Multi-tenant hosted deployment concerns
- Recreating every Honcho endpoint or SDK shape
- A broad multi-provider abstraction beyond the minimal reasoning adapter interface

## Proposed Layout

- `src/continuity/config.py`
- `src/continuity/store/sqlite.py`
- `src/continuity/store/schema.py`
- `src/continuity/store/claims.py`
- `src/continuity/store/belief_revision.py`
- `src/continuity/store/replay.py`
- `src/continuity/ontology.py`
- `src/continuity/policy.py`
- `src/continuity/disclosure.py`
- `src/continuity/compiler.py`
- `src/continuity/transactions.py`
- `src/continuity/arbiter.py`
- `src/continuity/events.py`
- `src/continuity/views.py`
- `src/continuity/epistemics.py`
- `src/continuity/outcomes.py`
- `src/continuity/prompt_planner.py`
- `src/continuity/snapshots.py`
- `src/continuity/tiers.py`
- `src/continuity/index/zvec_index.py`
- `src/continuity/embeddings/ollama.py`
- `src/continuity/reasoning/base.py`
- `src/continuity/reasoning/codex_adapter.py`
- `src/continuity/session_manager.py`
- `src/continuity/context_builder.py`
- `src/continuity/prefetch.py`
- `src/continuity/migration.py`
- `src/continuity/api.py`
- `src/continuity/evals/replay_runner.py`
- `tests/`
- `docs/architecture.md`
- `docs/hermes-compatibility.md`

## Prerequisites

- Use Python for v1.
- Collect a Hermes fixture set from `~/.hermes/hermes-agent`:
  - Honcho integration behavior
  - tool descriptions
  - recall-mode behavior
  - config parsing cases
  - migration examples
- Keep local runtime assumptions explicit:
  - SQLite available
  - Ollama available
  - `nomic-embed-text` installed
  - Codex SDK credentials available through Hermes environment/runtime
- Decide whether the first consumer is:
  - an internal Hermes patch
  - an MCP wrapper
  - or both

## Sprint 1: Define Continuity Contracts

**Goal**: Lock the public behavior and data contracts before implementation details spread.

**Demo/Validation**:
- Write a compatibility table mapping Hermes Honcho features to Continuity features.
- Define the core SQLite entities and the reasoning adapter interface.
- Define the subject-graph, observation-log, claim-ledger, and memory-locus invariants.
- Define the disclosure/audience layer and host-read eligibility invariants.
- Define the compiled-view algebra and its invariants.
- Define the memory transaction pipeline and its invariants.
- Define the durability contract and commit waterlines.
- Define the mutation arbiter and commit-lane invariants.
- Define the system event journal and append-only event semantics.
- Define the epistemic-status layer and abstention rules.
- Define the outcome ledger and outcome-label semantics.
- Define the budgeted prompt planner and packing invariants.
- Define the belief-state and revision invariants.
- Define replay-record and replay-run invariants.
- Define typed-memory classes and policy invariants.
- Define policy-pack schema and versioning invariants.
- Define compiler dependency and invalidation invariants.
- Define snapshot consistency and promotion invariants.
- Define generational tiering and retention invariants.
- Run unit tests for config/session-name normalization.

### Task 1.1: Document Hermes-visible parity targets
- **Location**: `docs/hermes-compatibility.md`
- **Description**: Document the exact Hermes surface to preserve:
  - `honcho_context` equivalent behavior
  - `honcho_search` equivalent behavior
  - `honcho_profile` equivalent behavior
  - `honcho_conclude` equivalent behavior
  - prefetch-driven prompt context
  - memory/write modes
- **Dependencies**: None
- **Acceptance Criteria**:
  - Every in-scope Hermes behavior is listed.
  - Anything intentionally omitted is explicit.
- **Validation**:
  - Manual review against `~/.hermes/hermes-agent/honcho_integration/*`, `tools/honcho_tools.py`, and `run_agent.py`.

### Task 1.2: Define config and naming rules
- **Location**: `src/continuity/config.py`, `tests/test_config.py`
- **Description**: Implement config parsing compatible with the Hermes mental model:
  - host/global precedence
  - `memoryMode`
  - `recallMode`
  - `writeFrequency`
  - session resolution strategy
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Config precedence is deterministic.
  - Session naming behavior matches fixture expectations.
- **Validation**:
  - Unit tests covering real Hermes-like cases.

### Task 1.3: Define the minimal reasoning adapter interface
- **Location**: `src/continuity/reasoning/base.py`, `tests/test_reasoning_contract.py`
- **Description**: Define only the methods needed by Continuity:
  - `answer_query`
  - `generate_structured`
  - `summarize_session`
  - `derive_claims`
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Interface is minimal and Hermes-driven.
  - No provider-specific fields leak into core memory logic.
- **Validation**:
  - Static typing and fake adapter tests.

### Task 1.4: Define subject-graph, observation, claim-ledger, and memory-locus invariants
- **Location**: `src/continuity/store/claims.py`, `docs/architecture.md`, `tests/test_claim_model.py`
- **Description**: Define the canonical memory IR:
  - subjects are canonical identities with auditable aliases and merge/split history
  - observations are immutable normalized source records
  - claims are append-only, typed, scoped, and provenance-linked
  - claims and loci distinguish applicability scope from audience/disclosure policy
  - every claim resolves to a canonical subject and locus with stable address semantics
  - claims carry `observed_at`, `learned_at`, `valid_from`, and `valid_to` where applicable
  - loci define `subject_id`, `locus_key`, `conflict_set_key`, and `aggregation_mode`
  - claim relations encode support, supersession, contradiction, and correction
  - profile/card compiled views are projections over active claims
  - dialectic answers may include supporting claim and observation payloads
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Subject, observation, claim, and locus rules are explicit and testable.
  - Subject aliasing and merge/split semantics are explicit and testable.
  - Locus addressing and conflict semantics are explicit and testable.
  - No durable derived memory artifact exists outside the claim ledger.
  - No host-visible memory artifact exists without claim provenance.
- **Validation**:
  - Invariant tests and architecture review.

### Task 1.5: Define belief revision invariants
- **Location**: `docs/architecture.md`, `tests/test_belief_revision_model.py`
- **Description**: Define the required temporal belief model:
  - beliefs are projections over evidence-backed claims grouped by subject and locus
  - contradictory claims do not silently coexist as equally active truth
  - explicit corrections can supersede older beliefs
  - stale beliefs can decay without deleting source evidence
  - each locus resolves according to its aggregation mode
  - retrieval should prefer active beliefs over merely historical claims
- **Dependencies**: Tasks 1.1 and 1.4
- **Acceptance Criteria**:
  - The revision rules are explicit and testable.
  - “Current belief” and “historical claim history” are clearly separated.
- **Validation**:
  - Invariant tests and architecture review.

### Task 1.6: Define replay invariants
- **Location**: `docs/architecture.md`, `tests/test_replay_model.py`
- **Description**: Define the required replay model:
  - every answered turn can persist a canonical decision record
  - replay inputs are deterministic and versioned
  - subject resolution is captured as part of the replayable decision path
  - alternate retrieval, belief, and reasoning paths can be compared without mutating source-of-truth memory
  - replay results can be scored against expected outcomes or later corrections
- **Dependencies**: Tasks 1.1, 1.4, and 1.5
- **Acceptance Criteria**:
  - Replay artifacts and replay runs are explicit and testable.
  - Replay does not require hidden runtime state.
- **Validation**:
  - Invariant tests and architecture review.

### Task 1.7: Define typed memory ontology
- **Location**: `src/continuity/ontology.py`, `docs/architecture.md`, `tests/test_ontology.py`
- **Description**: Define the memory classes and their policy rules:
  - which memory types exist in v1
  - what evidence can produce each claim type
  - which subject kinds each claim type can attach to
  - which locus shapes each claim type can inhabit
  - how each type decays, conflicts, or gets promoted
  - how each type should be rendered into prompt context
  - which types are user-memory, assistant-memory, shared-session context, or ephemeral state
- **Dependencies**: Tasks 1.1, 1.4, and 1.5
- **Acceptance Criteria**:
  - The ontology is small, explicit, and Hermes-driven.
  - Claim derivation, belief revision, and retrieval policies can reference ontology types directly.
- **Validation**:
  - Invariant tests and architecture review.

### Task 1.8: Define versioned memory policy packs
- **Location**: `src/continuity/policy.py`, `docs/architecture.md`, `tests/test_policy.py`
- **Description**: Define how named policy packs govern memory behavior:
  - policy pack identity and versioning
  - which ontology and memory classes a policy enables
  - how a policy supplies subject-resolution, claim-derivation, locus-resolution, per-type revision, and retrieval rules
  - how prompt rendering, budget planning, and compiled-view behavior are attached to a policy
  - how replay compares decisions across policy versions
- **Dependencies**: Tasks 1.5, 1.6, and 1.7
- **Acceptance Criteria**:
  - A policy pack is explicit, small, and host-facing.
  - Core memory behavior can be explained by policy version instead of scattered defaults.
- **Validation**:
  - Invariant tests and architecture review.

### Task 1.9: Define incremental compiler invariants
- **Location**: `src/continuity/compiler.py`, `docs/architecture.md`, `tests/test_compiler_model.py`
- **Description**: Define the compiler model for derived memory artifacts:
  - which artifacts are source inputs, canonical subjects/claims/loci, and downstream compiled views
  - how dependencies and fingerprints are represented
  - how dirtying and selective rebuild work
  - how dirtying resolves to affected subjects and loci before downstream rebuild
  - how policy-version and adapter-version changes trigger invalidation
  - how rebuild reasons are surfaced for inspection and replay
- **Dependencies**: Tasks 1.4, 1.6, 1.7, and 1.8
- **Acceptance Criteria**:
  - Source inputs, claims, and compiled views are explicitly separated.
  - Invalidation and rebuild rules are deterministic and inspectable.
- **Validation**:
  - Invariant tests and architecture review.

### Task 1.10: Define compiled view algebra
- **Location**: `src/continuity/views.py`, `docs/architecture.md`, `tests/test_view_model.py`
- **Description**: Define the host-visible compiled view types:
  - which view kinds exist in v1
  - which claims, loci, and policies feed each view kind
  - which view kinds are single-locus, multi-locus, subject-level, session-level, or query-level
  - how audience/disclosure policy constrains each view kind
  - how provenance is exposed by each view kind
  - how prompt assembly and answer generation compose over lower-level views
- **Dependencies**: Tasks 1.4, 1.5, 1.7, 1.8, and 1.9
- **Acceptance Criteria**:
  - The view set is explicit, small, and host-facing.
  - Retrieval and API behavior can be described in terms of named views instead of generic materialization flows.
- **Validation**:
  - Invariant tests and architecture review.

### Task 1.11: Define memory transaction pipeline
- **Location**: `src/continuity/transactions.py`, `docs/architecture.md`, `tests/test_transaction_model.py`
- **Description**: Define the runtime transaction set and phase ordering:
  - which transaction kinds exist in v1
  - which phases each transaction includes
  - where replay capture boundaries sit
  - which transactions may publish snapshots
  - which transactions may enqueue or execute prefetch
  - how `writeFrequency` maps to transaction timing
- **Dependencies**: Tasks 1.4, 1.5, 1.6, 1.8, 1.9, and 1.10
- **Acceptance Criteria**:
  - Runtime behavior can be described in terms of named transactions with deterministic phase order.
  - Save-turn, conclusion-write, import, compile, publish, and prefetch flows are explicit instead of implicit orchestration.
- **Validation**:
  - Invariant tests and architecture review.

### Task 1.12: Define durability contract
- **Location**: `src/continuity/transactions.py`, `docs/architecture.md`, `tests/test_durability_model.py`
- **Description**: Define the commit waterlines for runtime behavior:
  - which durability waterlines exist in v1
  - which transaction kinds can satisfy which waterlines
  - which host APIs require which minimum waterlines before returning
  - how `writeFrequency` maps to awaited waterlines
  - which side effects are best-effort and excluded from the completion contract
- **Dependencies**: Tasks 1.10 and 1.11
- **Acceptance Criteria**:
  - Every mutating operation has an explicit completion contract.
  - Async behavior is described in terms of waterlines rather than vague timing language.
- **Validation**:
  - Invariant tests and architecture review.

### Task 1.13: Define mutation arbiter
- **Location**: `src/continuity/arbiter.py`, `docs/architecture.md`, `tests/test_arbiter_model.py`
- **Description**: Define the serialized commit-lane model:
  - which state transitions must pass through the arbiter
  - which work may happen off-lane and publish back later
  - how commit ordering is represented
  - how arbiter ordering interacts with durability waterlines and snapshot promotion
  - how replay records arbiter order for debugging and evaluation
- **Dependencies**: Tasks 1.11 and 1.12
- **Acceptance Criteria**:
  - Authoritative mutation boundaries are explicit and testable.
  - The plan distinguishes between concurrent computation and serialized publication.
- **Validation**:
  - Invariant tests and architecture review.

### Task 1.14: Define epistemic-status layer
- **Location**: `src/continuity/epistemics.py`, `docs/architecture.md`, `tests/test_epistemics_model.py`
- **Description**: Define the uncertainty and abstention model:
  - which epistemic statuses exist in v1
  - which statuses can attach to claims, locus resolutions, compiled views, and answers
  - when abstention, qualification, or clarification is required
  - how low-confidence or conflicted state is suppressed or surfaced in `prompt_view`
  - how replay evaluates calibration quality
- **Dependencies**: Tasks 1.5, 1.8, 1.10, and 1.12
- **Acceptance Criteria**:
  - The system can express not only current state but confidence and uncertainty semantics explicitly.
  - Abstention and clarification behavior are explicit and inspectable.
- **Validation**:
  - Invariant tests and architecture review.

### Task 1.15: Define outcome ledger
- **Location**: `src/continuity/outcomes.py`, `docs/architecture.md`, `tests/test_outcomes_model.py`
- **Description**: Define the downstream outcome model:
  - which outcome labels exist in v1
  - which artifacts and decisions can receive outcome annotations
  - which outcome signals come from user feedback, explicit corrections, or inferred downstream use
  - how outcome records attach to replay and policy evaluation
  - how outcome labels differ from epistemic status
- **Dependencies**: Tasks 1.6, 1.12, and 1.14
- **Acceptance Criteria**:
  - The system can record what later happened, not just what it believed.
  - Replay and policy evaluation can consume explicit downstream outcomes.
- **Validation**:
  - Invariant tests and architecture review.

### Task 1.16: Define snapshot consistency invariants
- **Location**: `src/continuity/snapshots.py`, `docs/architecture.md`, `tests/test_snapshot_model.py`
- **Description**: Define the snapshot model for host-visible reads:
  - what belongs to a snapshot
  - how snapshots reference compiled view artifacts
  - how active heads and candidate branches are represented
  - how promotion, rollback, and diffing work
  - how reads pin to a snapshot during retrieval, prompting, and replay
- **Dependencies**: Tasks 1.6, 1.8, 1.9, 1.10, 1.11, 1.12, 1.13, 1.14, and 1.15
- **Acceptance Criteria**:
  - Snapshot reads are coherent and immutable.
  - Promotion from candidate to active is explicit and inspectable.
- **Validation**:
  - Invariant tests and architecture review.

### Task 1.17: Define generational tiering invariants
- **Location**: `src/continuity/tiers.py`, `docs/architecture.md`, `tests/test_tiers_model.py`
- **Description**: Define the tiering model for long-lived memory:
  - which tiers exist in v1
  - which ontology types and compiled view artifacts belong in each tier by default
  - how policy packs govern promotion, demotion, retention, and archival
  - how tiering affects retrieval defaults, compiler urgency, and snapshot inclusion
  - how replay and audit artifacts move into archival tiers without polluting active reads
- **Dependencies**: Tasks 1.7, 1.8, 1.9, 1.10, 1.11, 1.12, 1.13, 1.14, 1.15, and 1.16
- **Acceptance Criteria**:
  - Tier boundaries are explicit, small, and policy-driven.
  - Tiering affects retrieval and rebuild behavior without changing source-of-truth semantics.
- **Validation**:
  - Invariant tests and architecture review.

### Task 1.18: Define budgeted prompt planner
- **Location**: `src/continuity/prompt_planner.py`, `docs/architecture.md`, `tests/test_prompt_planner_model.py`
- **Description**: Define the bounded prompt packing model for `prompt_view`:
  - which fragment classes can compete for prompt space
  - which hard and soft budgets exist in v1
  - how disclosure-ineligible fragments are filtered, summarized, redacted, or withheld
  - how priority, compression, and drop order are defined
  - how planner decisions are exposed for replay, debugging, and token-cost analysis
  - how policy packs tune prompt economics without changing claim semantics
- **Dependencies**: Tasks 1.8, 1.10, 1.11, and 1.12
- **Acceptance Criteria**:
  - Prompt assembly can be described as deterministic packing under explicit budget.
  - Inclusion, exclusion, and degradation reasons are explicit and inspectable.
- **Validation**:
  - Invariant tests and architecture review.

### Task 1.19: Define system event journal
- **Location**: `src/continuity/events.py`, `docs/architecture.md`, `tests/test_event_journal_model.py`
- **Description**: Define the append-only control-plane event model:
  - which authoritative event types exist in v1
  - which transitions must emit journal entries
  - how journal ordering interacts with arbiter ordering and durability waterlines
  - which payloads are stored inline vs by reference
  - how replay and debugging consume journal entries
- **Dependencies**: Tasks 1.11, 1.12, and 1.13
- **Acceptance Criteria**:
  - The plan can reconstruct authoritative state transitions from journal order.
  - Event semantics are explicit enough to support replay, debugging, and rebuilds.
- **Validation**:
  - Invariant tests and architecture review.

### Task 1.20: Define disclosure/audience layer
- **Location**: `src/continuity/disclosure.py`, `docs/architecture.md`, `tests/test_disclosure_model.py`
- **Description**: Define the host-read eligibility and transformation model:
  - which audience principals, viewer kinds, and channels exist in v1
  - how claim-level, locus-level, and compiled-view disclosure policies compose
  - how `scope` differs from audience/disclosure semantics
  - how reads compile under viewer, channel, purpose, and policy version
  - when memory is exposed directly, summarized, redacted, withheld, or gated on consent
  - how turn artifacts and replay capture disclosure decisions
- **Dependencies**: Tasks 1.4, 1.8, 1.10, and 1.18
- **Acceptance Criteria**:
  - Host-facing reads have explicit, inspectable disclosure semantics.
  - The system can explain why memory was exposed, transformed, or withheld for a given audience.
- **Validation**:
  - Invariant tests and architecture review.

## Sprint 2: Build Durable Memory Storage

**Goal**: Build the SQLite-backed source-of-truth layer for subjects, peers, sessions, messages, observations, claims, loci, and the compiled views derived from them.

**Demo/Validation**:
- Create subjects, peers, sessions, and messages.
- Create observations, loci, and provenance-linked claims.
- Create active beliefs and belief updates from claims resolved per locus.
- Persist compiled view metadata and view artifacts for host-facing reads.
- Persist replayable turn artifacts without any vector layer.
- Persist append-only system events for authoritative state transitions.
- Execute a basic `ingest_turn` transaction end to end without retrieval.
- Persist ontology types and policy metadata on observations, claims, and beliefs.
- Persist audience/disclosure policy metadata and effective disclosure decisions on claims, views, and turn artifacts.
- Persist policy versions on derived artifacts and turn decision records.
- Persist compiler dependency metadata, fingerprints, and dirty queues.
- Persist snapshot membership and active/candidate snapshot heads.
- Persist tier assignments, tier transitions, and retention policies.
- Reload the store from disk and recover state.
- Query session history and claims without any vector layer.

### Task 2.1: Define SQLite schema
- **Location**: `src/continuity/store/schema.py`, `tests/test_schema.py`
- **Description**: Create tables for:
  - peers
  - subjects
  - subject_aliases
  - subject_links
  - subject_resolution_runs
  - sessions
  - session_peers
  - messages
  - observations
  - claim_loci
  - claims
  - claim_sources
  - claim_relations
  - beliefs
  - belief_updates
  - derivation_runs
  - turn_artifacts
  - replay_runs
  - policy_versions
  - disclosure_policies
  - system_events
  - outcome_records
  - compiled_views
  - view_disclosure_decisions
  - artifact_dependencies
  - artifact_versions
  - dirty_queue
  - snapshots
  - snapshot_heads
  - snapshot_artifacts
  - artifact_tiers
  - tier_transitions
  - retention_policies
  - peer_cards
  - representations
  - migration_records
  - prefetch_cache
- **Dependencies**: Sprint 1 complete
- **Acceptance Criteria**:
  - Schema covers all in-scope domain objects.
  - Rebuild of vector index is possible from durable state.
  - Canonical subjects and aliases are explicit and queryable.
  - Claim loci and claim membership are explicit and queryable.
  - Compiled views are explicit, typed, and queryable.
  - Derived claims can be traced to source observations and derivation runs.
  - Active beliefs can be derived, superseded, and inspected independently of raw evidence.
  - Replayable turn artifacts can be stored and loaded independently of retrieval engine internals.
  - System event journal entries are append-only and ordered.
  - Outcome records can be attached to claims, views, and decisions explicitly.
  - Memory records can be typed and queried by ontology class.
  - Claim validity, subject, locus, scope, and relation metadata are explicit and queryable.
  - Beliefs, derivation runs, and turn artifacts can be traced to a concrete policy version.
  - Derived artifacts can be invalidated and selectively rebuilt from stored dependency metadata.
  - Host-visible artifacts can be resolved through a coherent snapshot head.
  - Artifacts can be assigned to explicit tiers and transitioned deterministically over time.
- **Validation**:
  - Schema migration and round-trip tests.

### Task 2.2: Implement repository layer
- **Location**: `src/continuity/store/sqlite.py`, `tests/test_store.py`
- **Description**: Build CRUD/query helpers for the schema, including subject and alias resolution primitives plus disclosure-policy lookups.
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Reads/writes are explicit and transactional.
  - No `zvec` dependency is required for core state operations.
- **Validation**:
  - Unit tests with temporary SQLite databases.

### Task 2.3: Implement belief revision engine
- **Location**: `src/continuity/store/belief_revision.py`, `tests/test_belief_revision.py`
- **Description**: Implement the engine that converts evidence-backed claims into current beliefs:
  - create initial beliefs
  - respect canonical subject boundaries before locus resolution
  - resolve claims within each locus according to aggregation mode
  - supersede or demote stale beliefs
  - resolve explicit corrections
  - record belief update history
- **Dependencies**: Task 2.2
- **Acceptance Criteria**:
  - Current belief state is reproducible from stored evidence.
  - Belief updates are durable and inspectable.
  - Belief promotion, decay, and supersession can vary by memory type without hidden heuristics.
  - Belief outputs can carry explicit epistemic status when resolution is uncertain or conflicted.
  - Belief updates are attributable to a specific policy version.
- **Validation**:
  - Unit tests for correction, contradiction, and preference-change cases.

### Task 2.4: Implement transaction runner
- **Location**: `src/continuity/transactions.py`, `tests/test_transactions.py`
- **Description**: Implement the runtime transaction orchestration layer:
  - `ingest_turn`
  - `write_conclusion`
  - `import_history`
  - `compile_views`
  - `publish_snapshot`
  - `prefetch_next_turn`
  - enforce deterministic phase ordering and transaction boundaries
- **Dependencies**: Tasks 2.2 and 2.3
- **Acceptance Criteria**:
  - Transaction phases are explicit and inspectable.
  - Runtime behavior does not depend on hidden orchestration in unrelated modules.
  - `writeFrequency` can be mapped onto transaction timing without redefining core behavior.
  - Each transaction reports which durability waterline it satisfied before returning.
- **Validation**:
  - Transaction-flow tests for ingest, conclude, and import paths.

### Task 2.5: Implement mutation arbiter
- **Location**: `src/continuity/arbiter.py`, `tests/test_arbiter.py`
- **Description**: Implement the serialized commit-lane runtime:
  - accept authoritative publication requests from transaction flows
  - serialize subject/claim/locus mutation and snapshot-head updates
  - separate off-lane computation from on-lane publication
  - record arbiter ordering metadata for replay/debugging
- **Dependencies**: Tasks 2.2, 2.3, and 2.4
- **Acceptance Criteria**:
  - Authoritative mutation does not depend on ad hoc locking scattered across modules.
  - Concurrent publish attempts resolve through one ordered lane.
  - Snapshot-head and durability-waterline signaling pass through the same arbiter.
  - Every authoritative publish emits a corresponding system event.
- **Validation**:
  - Arbiter ordering and publication tests.

### Task 2.6: Implement system event journal
- **Location**: `src/continuity/events.py`, `tests/test_event_journal_store.py`
- **Description**: Persist and query append-only authoritative events:
  - record published subject/claim/locus transitions
  - record snapshot-head publications
  - expose journal ordering for replay and debugging
  - support reconstruction-oriented queries over event order
- **Dependencies**: Tasks 2.4 and 2.5
- **Acceptance Criteria**:
  - System events are append-only, ordered, and attributable.
  - Replay and debugging can consume journal order without inferring from table state alone.
- **Validation**:
  - Journal append and reconstruction tests.

### Task 2.7: Implement replay artifact repository
- **Location**: `src/continuity/store/replay.py`, `tests/test_replay_store.py`
- **Description**: Persist canonical turn decision records and replay runs:
  - save retrieval candidates and selected evidence
  - save subject-resolution decisions used for prompt assembly
  - save resolved loci used for prompt assembly
  - save source compiled views used for prompt assembly or answers
  - save disclosure decisions and withheld/redacted fragments for prompt assembly or answers
  - save active claims used for prompt assembly
  - save active beliefs used for prompt assembly
  - save prompt-view snapshots
  - save reasoning input/output envelopes
  - save replay comparison scores and notes
- **Dependencies**: Tasks 2.2 and 2.3
- **Acceptance Criteria**:
  - Replay artifacts are versioned, queryable, and immutable after capture.
  - Replay runs do not mutate source-of-truth conversation memory.
  - Replay records include the policy version used for the original and replayed decisions.
  - Replay records include the transaction kind and phase boundary that produced them where applicable.
  - Replay records include arbiter ordering metadata when publication occurred.
  - Replay records can link back to authoritative journal positions where relevant.
- **Validation**:
  - Repository round-trip tests.

### Task 2.8: Implement outcome ledger repository
- **Location**: `src/continuity/outcomes.py`, `tests/test_outcomes_store.py`
- **Description**: Persist and query post-hoc outcome records:
  - record confirmations, corrections, overrides, and downstream usefulness labels
  - attach outcomes to claims, compiled views, and turn artifacts
  - query outcomes for replay and policy evaluation
- **Dependencies**: Tasks 2.2, 2.5, and 2.6
- **Acceptance Criteria**:
  - Outcome records are durable, queryable, and attributable.
  - Replay and policy code can consume outcome records without scraping ad hoc notes.
- **Validation**:
  - Repository round-trip tests.

### Task 2.9: Implement compiler state repository
- **Location**: `src/continuity/compiler.py`, `tests/test_compiler_store.py`
- **Description**: Persist and query compiler dependency state:
  - register dependencies between source inputs, subjects, claims, loci, and compiled views
  - persist fingerprints and artifact versions
  - enqueue dirty artifacts with reason codes
  - query affected subjects before affected loci
  - query affected loci before downstream rebuild
  - query rebuild plans
- **Dependencies**: Tasks 2.2, 2.3, and 2.4
- **Acceptance Criteria**:
  - Compiler state is durable and queryable.
  - Dirtying one source input does not require scanning all artifacts heuristically.
- **Validation**:
  - Round-trip and selective-invalidation tests.

### Task 2.10: Implement snapshot repository
- **Location**: `src/continuity/snapshots.py`, `tests/test_snapshots_store.py`
- **Description**: Persist and query snapshot state:
  - create immutable snapshots
  - map artifacts into snapshots
  - manage active and candidate heads
  - diff snapshots and inspect promotion history
- **Dependencies**: Tasks 2.4 and 2.5
- **Acceptance Criteria**:
  - Reads can resolve a stable active snapshot without inspecting mutable compiler state.
  - Candidate snapshots can be built and inspected before promotion.
- **Validation**:
  - Round-trip, promotion, and diff tests.

### Task 2.11: Implement tier state repository
- **Location**: `src/continuity/tiers.py`, `tests/test_tiers_store.py`
- **Description**: Persist and query tiering state:
  - assign initial artifact tiers
  - record promotion and demotion transitions
  - resolve retention policy metadata
  - query tier-bounded retrieval and rebuild candidates
- **Dependencies**: Tasks 2.5 and 2.6
- **Acceptance Criteria**:
  - Tier state is durable and inspectable.
  - Tier transitions do not require ad hoc scans or heuristics outside the policy layer.
- **Validation**:
  - Round-trip and tier-transition tests.

### Task 2.12: Implement session manager on SQLite
- **Location**: `src/continuity/session_manager.py`, `tests/test_session_manager.py`
- **Description**: Implement:
  - peer/session creation
  - local message cache
  - transaction-trigger, write-frequency, and awaited-waterline logic
  - peer-specific memory-mode gating
- **Dependencies**: Tasks 2.4, 2.5, and 2.6
- **Acceptance Criteria**:
  - Session manager works without retrieval/reasoning enabled.
  - Write-frequency rules behave deterministically.
  - The requested waterline for each save mode is explicit and testable.
- **Validation**:
  - Unit tests covering async/turn/session/N-turn modes.

## Sprint 3: Add Retrieval With zvec + Ollama

**Goal**: Build local semantic retrieval and typed compiled-view assembly over claims and their compiled views.

**Demo/Validation**:
- Generate embeddings via Ollama.
- Index messages, observations, durable claims, and current beliefs in `zvec`.
- Build explicit state/profile/prompt/evidence/answer view assembly paths.
- Enforce audience-aware disclosure rules during host-facing reads.
- Build a `prompt_view` under a hard token budget with deterministic degradation.
- Retrieve relevant memory context for a query.
- Register vector-index records as rebuildable compiled artifacts.
- Resolve retrieval against a pinned snapshot.
- Apply tier-aware retrieval defaults.

### Task 3.1: Implement Ollama embedding client
- **Location**: `src/continuity/embeddings/ollama.py`, `tests/test_ollama_embeddings.py`
- **Description**: Build the embedding client for `nomic-embed-text`.
- **Dependencies**: Sprint 2 complete
- **Acceptance Criteria**:
  - Embeddings are generated reproducibly.
  - Dimension handling is explicit.
- **Validation**:
  - Live opt-in test against local Ollama.

### Task 3.2: Implement zvec indexing
- **Location**: `src/continuity/index/zvec_index.py`, `tests/test_zvec_index.py`
- **Description**: Index messages, observations, evidence-bearing claims, compiled views, and current beliefs into `zvec`.
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Index entries map back to SQLite records.
  - Rebuild-from-SQLite path exists.
  - Search hits can be resolved back to observation, claim, and locus nodes.
  - Belief-state indexing is distinguishable from raw-history indexing.
  - Index entries participate in compiler invalidation and rebuild flows.
- **Validation**:
  - Unit tests plus index rebuild test.

### Task 3.3: Implement retrieval APIs
- **Location**: `src/continuity/context_builder.py`, `tests/test_retrieval.py`
- **Description**: Build:
  - semantic search
  - peer/profile lookup
  - context retrieval with peer target/perspective
  - subject resolution from names, aliases, and imported references
  - `state_view`, `timeline_view`, and `set_view` assembly
  - `profile_view` assembly
  - `prompt_view` assembly
  - `evidence_view` assembly
  - budgeted prompt packing with deterministic fragment selection and degradation ladders
  - prompt-ready memory assembly
  - evidence-aware ranking and citation selection
  - disclosure-policy filtering and transformation by viewer, channel, and purpose
  - belief-aware ranking with freshness and confidence
  - epistemic-status-aware suppression and surfacing rules
  - locus-aware resolution for single-value, set, timeline, and state-machine memory slots
  - ontology-aware ranking and rendering by memory type
  - policy-pack-driven retrieval and prompt assembly
- **Dependencies**: Task 3.2
- **Acceptance Criteria**:
  - Retrieval works from local state only.
  - Recall-mode behavior is explicit.
  - Retrieval resolves ambiguous subject references deterministically and inspectably.
  - Each host-visible read path resolves to an explicit compiled view kind.
  - Retrieval compiles reads under explicit viewer, channel, and purpose context.
  - Search results and prompt memory can expose provenance when needed.
  - Retrieval prefers active beliefs and well-supported current claims over stale historical claims by default.
  - Retrieval resolves each locus according to aggregation mode before prompt assembly.
  - Retrieval and prompt assembly can prioritize or suppress memory types based on context.
  - Retrieval can withhold, redact, or summarize memory that is relevant but not disclosable for the active audience.
  - `prompt_view` fits within an explicit token budget.
  - Inclusion, exclusion, and degradation decisions for `prompt_view` are inspectable.
  - Low-confidence or conflicted memory can be qualified or omitted instead of being flattened into asserted prompt state.
  - Cross-peer and assistant-internal leakage are prevented by explicit disclosure rules rather than incidental ranking behavior.
  - Retrieval behavior is selectable and inspectable by policy version.
  - Retrieval can pin to a specific snapshot for coherent reads.
  - Retrieval defaults prefer `hot` and relevant `warm` memory, descending into `cold` only when needed.
- **Validation**:
  - Snapshot-style tests for memory block assembly.

## Sprint 4: Add Codex SDK Reasoning

**Goal**: Implement the first reasoning adapter and use it for Honcho-like chat, summaries, and claim derivation.

**Demo/Validation**:
- Answer a dialectic-style query from stored memory.
- Generate structured claims from conversation history.
- Summarize a session through the adapter.
- Preserve provenance links for all derived outputs.
- Feed the belief revision layer after derivation.
- Capture decision records that can be replayed later.
- Register derived outputs with compiler dependencies and fingerprints.
- Stage new artifacts into candidate snapshots before promotion.
- Assign derived outputs to initial tiers based on ontology and policy.

### Task 4.1: Implement Codex reasoning adapter
- **Location**: `src/continuity/reasoning/codex_adapter.py`, `tests/test_codex_adapter.py`
- **Description**: Implement the reasoning adapter using Codex SDK:
  - model `gpt-5.4`
  - low reasoning
  - structured output for claim derivation
  - text output for dialectic answers
  - structured evidence references in outputs where applicable
- **Dependencies**: Sprint 1 complete
- **Acceptance Criteria**:
  - Adapter satisfies the minimal interface.
  - Config for model/reasoning is centralized.
  - Adapter-facing prompting and structured-output expectations can vary by policy pack without changing core interfaces.
- **Validation**:
  - Fake adapter unit tests and opt-in live tests.

### Task 4.2: Implement claim derivation pipeline
- **Location**: `src/continuity/reasoning/*.py`, `tests/test_claim_derivation.py`
- **Description**: Use Codex adapter to derive durable typed claims and update downstream compiled views.
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Derived claims are persisted in SQLite.
  - Retrieval index updates after new claims.
  - Each derived claim links to the observations and derivation run that produced it.
  - Each derived claim is assigned to a canonical subject before locus assignment.
  - Each derived claim is assigned to a canonical locus before belief revision.
  - Belief revision runs after claim derivation and updates the current belief state.
  - Derived claims are assigned ontology types before entering the belief layer.
  - Derivation runs record the policy version that governed classification and promotion.
  - Derived claims register compiler dependencies on observations, policy version, and adapter version.
  - Affected compiled views can be staged into a candidate snapshot without mutating the active snapshot.
  - Derived outputs and affected views receive tier treatment appropriate to their ontology type and policy.
- **Validation**:
  - End-to-end local tests with fixture conversations.

### Task 4.3: Implement dialectic-style answers
- **Location**: `src/continuity/api.py`, `tests/test_dialectic.py`
- **Description**: Use retrieval plus Codex adapter to produce `answer_view` results for natural-language questions about a peer.
- **Dependencies**: Tasks 3.3 and 4.1
- **Acceptance Criteria**:
  - Equivalent to Hermes’ `honcho_context` behavior at the user level.
  - The implementation can surface supporting evidence for “why do you think that?” style follow-ups.
  - Answers can distinguish current belief from older superseded evidence when relevant.
  - Answers can abstain, qualify, or ask for confirmation when epistemic status requires it.
  - Answers can distinguish `unknown` from `known_but_withheld` when disclosure policy requires it.
- **Validation**:
  - Snapshot/fixture tests and local smoke tests.

### Task 4.4: Capture turn decision records
- **Location**: `src/continuity/api.py`, `tests/test_turn_artifacts.py`
- **Description**: Record the artifacts needed for replay whenever Continuity answers a memory question or assembles a compiled view for host use.
- **Dependencies**: Tasks 3.3, 4.1, and 4.3
- **Acceptance Criteria**:
  - Retrieval inputs, selected claims, selected beliefs, source compiled views, and reasoning envelopes are durably captured.
  - Capture is deterministic and versioned.
  - Capture records the transaction kind and relevant phase boundary.
  - Capture records the durability waterline reached before the host saw completion.
  - Capture records prompt packing decisions for `prompt_view` when applicable.
  - Capture records disclosure decisions, redactions, and withheld fragments when applicable.
  - Capture can be linked to later outcome records.
  - Turn artifacts record the active policy pack.
  - Turn artifacts record the snapshot used for the read.
  - Turn artifacts and replay records are eligible for archival tiers by policy.
- **Validation**:
  - Fixture tests asserting stable serialized turn artifacts.

## Sprint 5: Compilation, Snapshots, Tiering, Prefetch, Migration, And Host Integration

**Goal**: Make the system practical for Hermes integration and keep derived memory fresh without broad recomputation.

**Demo/Validation**:
- Change a source observation and rebuild only affected claims and compiled views.
- Change a policy version and produce a targeted rebuild plan.
- Build a candidate snapshot and promote it atomically.
- Promote and demote artifacts across tiers without breaking host-visible reads.
- Warm next-turn memory context.
- Import local memory files/history.
- Exercise a host-facing integration API.

### Task 5.1: Implement incremental rebuild planner
- **Location**: `src/continuity/compiler.py`, `tests/test_compiler.py`
- **Description**: Build the compiler runtime:
  - detect source changes
  - compute rebuild plans from dependency metadata
  - resolve source changes into affected loci
  - rebuild affected claims, compiled views, caches, and index records
  - expose rebuild reasons for inspection and replay
- **Dependencies**: Sprints 2, 3, and 4 complete
- **Acceptance Criteria**:
  - Source edits, policy upgrades, and adapter upgrades rebuild only affected claims and compiled views.
  - Rebuild plans explain which loci were affected and why.
  - Rebuild plans are deterministic and explainable.
  - Rebuild publication flows back through the mutation arbiter rather than mutating authoritative state directly.
  - Rebuild output can be materialized into a candidate snapshot rather than mutating the active head.
- **Validation**:
  - Fixture tests for corrections, policy upgrades, and adapter upgrades.

### Task 5.2: Implement snapshot builder and promoter
- **Location**: `src/continuity/snapshots.py`, `tests/test_snapshots_runtime.py`
- **Description**: Build the runtime that assembles and promotes coherent read states:
  - materialize compiled view artifacts into candidate snapshots
  - atomically promote candidate snapshots to active heads
  - diff snapshots
  - expose branchable snapshots for replay and policy experiments
- **Dependencies**: Tasks 2.6 and 5.1
- **Acceptance Criteria**:
  - Hosts never read partially rebuilt state.
  - Candidate snapshot promotion is atomic and inspectable.
  - Snapshot-head promotion flows through the mutation arbiter.
  - Snapshot publication is recorded in the system event journal.
- **Validation**:
  - Fixture tests for concurrent rebuild/promotion scenarios.

### Task 5.3: Implement tiering and retention runtime
- **Location**: `src/continuity/tiers.py`, `tests/test_tiers_runtime.py`
- **Description**: Build the runtime that manages memory economics over time:
  - assign initial tiers from ontology and policy
  - promote and demote compiled views across tiers
  - bound `hot` and `warm` working sets
  - archive long-tail and replay-heavy artifacts into colder tiers
  - expose tier-aware retention decisions for inspection
- **Dependencies**: Tasks 2.7, 3.3, 5.1, and 5.2
- **Acceptance Criteria**:
  - Prompt-facing reads stay focused on `hot` and relevant `warm` memory by default.
  - Older artifacts remain recallable or auditable without bloating active snapshots.
- **Validation**:
  - Fixture tests for promotion, demotion, and archive recall.

### Task 5.4: Implement async prefetch
- **Location**: `src/continuity/prefetch.py`, `tests/test_prefetch.py`
- **Description**: Cache next-turn compiled views and optional next-turn synthesis.
- **Dependencies**: Tasks 2.4, 3.3, 4.3, 5.1, 5.2, and 5.3
- **Acceptance Criteria**:
  - First turn may be cold.
  - Next turns avoid unnecessary blocking work.
  - Prefetch never serves stale artifacts that are marked dirty by the compiler.
  - Prefetch reads and caches are pinned to a specific snapshot.
  - Prefetch respects tier-aware retrieval defaults.
  - Prefetch is triggered through an explicit `prefetch_next_turn` transaction path.
  - Prefetch remains outside the required completion waterline of mutating operations unless explicitly requested.
- **Validation**:
  - Prefetch cache behavior tests.

### Task 5.5: Implement migration and identity seeding
- **Location**: `src/continuity/migration.py`, `tests/test_migration.py`
- **Description**: Support:
  - prior session history import
  - `MEMORY.md` and `USER.md` import
  - AI identity seeding from `SOUL.md`/similar files
  - imported-name normalization into canonical subjects and aliases
  - provenance tagging for imported claims
- **Dependencies**: Sprint 2 complete
- **Acceptance Criteria**:
  - Migration is deterministic, identity-stable, and replay-safe.
- **Validation**:
  - Fixture-based migration tests.

### Task 5.6: Expose a minimal host API
- **Location**: `src/continuity/api.py`, `tests/test_api.py`
- **Description**: Expose only the host-needed surface:
  - initialize
  - save turn
  - search
  - get state view
  - get timeline view
  - get profile
  - get prompt memory block
  - answer memory question
  - write conclusion
  - import history
  - publish snapshot
  - resolve subject
  - inspect evidence for a claim or answer
  - inspect disclosure decisions for a view or answer
  - inspect epistemic status
  - record outcome
  - inspect outcomes
  - inspect turn decision record
  - inspect active policy pack and policy version
  - inspect compiler status and rebuild reasons
  - inspect current snapshot and snapshot diffs
  - inspect tier assignments and retention status
- **Dependencies**: Tasks 5.4 and 5.5
- **Acceptance Criteria**:
  - A host runtime does not need internal store/index details and can request named compiled views explicitly.
  - Host-facing reads compile under explicit viewer, channel, purpose, and disclosure policy context.
  - Host mutating operations map to explicit transaction entrypoints.
  - Host mutating operations document the durability waterline they guarantee before returning.
  - Host mutating operations do not publish authoritative state except through the mutation arbiter.
  - The API can accept and inspect explicit downstream outcome signals.
- **Validation**:
  - End-to-end integration tests.

## Sprint 6: Hardening And Adapter Expansion Readiness

**Goal**: Make the architecture stable enough to support future adapters without redesign.

**Demo/Validation**:
- Full automated suite passes.
- Codex adapter can be swapped with a fake adapter in tests.
- Adapter assumptions are documented.
- Replay-driven regression and counterfactual evaluation exists for real fixture turns.
- Policy-vs-policy replay comparisons exist for real fixture turns.
- Compiler-driven rebuilds are validated on real fixture turns.
- Snapshot-stable reads are validated during rebuild and promotion flows.
- Tier-bounded retrieval and retention behavior are validated on real fixture turns.

### Task 6.1: Add adapter conformance tests
- **Location**: `tests/reasoning/test_adapter_contract.py`
- **Description**: Define a shared contract suite for reasoning adapters.
- **Dependencies**: Sprint 4 complete
- **Acceptance Criteria**:
  - Future Claude/OpenCode adapters can be validated against the same suite.
- **Validation**:
  - Contract tests pass for Codex adapter and fake adapter.

### Task 6.2: Implement replay runner and evaluations
- **Location**: `src/continuity/evals/replay_runner.py`, `tests/test_replay_runner.py`
- **Description**: Re-run stored turns under alternate retrieval, belief, or reasoning strategies and compare outcomes.
- **Dependencies**: Tasks 2.4 and 4.4
- **Acceptance Criteria**:
  - Retrieval-only, belief-only, and end-to-end replays are supported.
  - Replay output can be scored for correctness, freshness, and token/cost impact.
  - Replay output can be scored for disclosure safety and over-disclosure impact.
  - Replay can score against explicit downstream outcome records where available.
  - Replay can consume authoritative journal order when event-level reconstruction matters.
  - Replay can compare multiple policy versions on the same stored turns.
- **Validation**:
  - Fixture evaluations on stored turn artifacts.

### Task 6.3: Document future adapter integration
- **Location**: `docs/architecture.md`
- **Description**: Document how Claude Agent SDK and OpenCode SDK can be added later.
- **Dependencies**: Tasks 6.1 and 6.2
- **Acceptance Criteria**:
  - The extension story is clear without changing the core interface.
- **Validation**:
  - Manual review.

## Testing Strategy

- Unit-test config normalization and session naming.
- Unit-test SQLite repositories independently of retrieval and reasoning.
- Add local live tests for:
  - Ollama embeddings
  - zvec indexing
  - Codex adapter
- Add fixture-based Hermes parity tests for:
  - recall behavior
  - prompt memory assembly
  - conclusion writes
  - profile and search results
- Add claim-ledger tests for:
  - append-only claim invariants
  - subject-assignment invariants
  - locus-address invariants
  - provenance completeness
  - supersession/conflict/correction behavior
  - bitemporal validity-window behavior
  - selective re-derivation after source changes
- Add belief revision tests for:
  - explicit correction handling
  - stale preference replacement
  - subject-boundary isolation
  - per-locus aggregation behavior for `latest_wins`, `set_union`, `timeline`, and `state_machine`
  - conflict resolution ordering
  - “what changed?” answer support
- Add ontology tests for:
  - type assignment for derived claims
  - per-type decay and supersession rules
  - prompt rendering differences by memory class
  - partitioning of user, assistant, and ephemeral memory
- Add disclosure tests for:
  - explicit separation between applicability scope and audience/disclosure policy
  - `assistant_internal`, `current_peer`, `shared_session`, and `host_internal` visibility cases
  - redaction, withholding, summarization, and consent-gated behavior for host-facing reads
  - prevention of cross-peer or assistant-internal leakage in prompt and answer assembly
- Add epistemics tests for:
  - explicit `unknown`, `tentative`, `conflicted`, `stale`, and `needs_confirmation` states
  - abstention instead of assertion when evidence is insufficient
  - clarification triggers under conflicted or low-confidence state
  - calibration scoring in replay fixtures
- Add outcome-ledger tests for:
  - explicit `confirmed`, `corrected`, `contradicted`, `stale_on_use`, `user_accepted`, and `user_overrode` labels
  - attachment of outcomes to claims, views, and turn artifacts
  - user-feedback vs inferred-outcome attribution
- Add prompt-planner tests for:
  - hard-budget adherence
  - deterministic fragment selection under equal inputs
  - degradation ladder behavior under budget pressure
  - low-confidence fragment suppression under budget and policy rules
  - inclusion and exclusion reason traces
- Add policy tests for:
  - policy-pack selection and version stamping
  - policy-specific subject-resolution outcomes
  - policy-specific retrieval/rendering differences
  - policy-specific prompt-packing outcomes
  - policy-specific belief revision outcomes
  - policy-vs-policy replay comparison on the same turn set
- Add compiler tests for:
  - source-edit invalidation
  - explicit correction invalidation
  - subject-merge or subject-split invalidation
  - locus-scoped invalidation
  - policy-upgrade rebuild planning
  - adapter-upgrade rebuild planning
  - selective rebuild of only affected claims and compiled views
- Add transaction tests for:
  - deterministic phase ordering for `ingest_turn`
  - `write_conclusion` transaction boundaries
  - `import_history` transaction boundaries
  - `publish_snapshot` transaction behavior
  - `prefetch_next_turn` transaction triggering
- Add arbiter tests for:
  - single ordered publication under concurrent writes
  - off-lane computation publishing back through the arbiter
  - snapshot-head serialization
  - durability-waterline signaling through arbiter order
- Add event-journal tests for:
  - append-only ordering
  - linkage from arbiter publications to journal entries
  - reconstruction of authoritative transition order
  - replay use of journal order for debugging and evaluation
- Add durability tests for:
  - `save_turn` waterline behavior by `writeFrequency`
  - `write_conclusion` minimum commit guarantee
  - `publish_snapshot` as the required boundary for prompt-visible reads
  - prefetch exclusion from mutating-operation completion by default
- Add snapshot tests for:
  - coherent reads during background rebuild
  - candidate snapshot promotion
  - snapshot diffing after policy upgrade
  - prefetch pinned to snapshot
- Add tiering tests for:
  - initial tier assignment by ontology type
  - promotion and demotion rules
  - archive recall from `cold` and `frozen`
  - tier-bounded prompt assembly
- Add replay tests for:
  - deterministic turn artifact capture
  - subject-resolution capture on the same stored turns
  - locus-resolution capture on the same stored turns
  - retrieval-only counterfactual replays
  - belief-policy counterfactual replays
  - prompt-packing counterfactual replays
  - calibration and abstention counterfactual replays
  - outcome-anchored counterfactual replays
  - claim-set comparison on the same stored turns
  - adapter comparison on the same stored turns
- Keep all live tests opt-in.

## Potential Risks And Gotchas

- `zvec` should not become the source of truth. Treat it as rebuildable.
- The claim ledger can sprawl if claim granularity is vague. Keep claim boundaries explicit and Hermes-driven.
- The locus model can either collapse distinct memories together or fragment one memory into too many slots. Lock locus-address semantics early and validate them with fixtures.
- The subject graph can over-merge distinct entities or under-merge aliases for the same entity. Lock subject-resolution semantics early and validate them with migration and replay fixtures.
- The Evidence Graph can become complex quickly if provenance rules are vague. Keep observation-to-subject, subject-to-claim, claim-to-locus, and locus-to-view edges explicit and minimal.
- Belief revision can become heuristic sludge if scoring rules are underdefined. Keep revision rules explicit, deterministic, and inspectable.
- Runtime behavior can drift if transaction ordering is implicit. Lock phase order early and test it directly.
- Completion semantics can drift if APIs and transactions do not declare their waterlines. Lock durability contracts early and test them directly.
- Concurrency can still corrupt authoritative state if publication is not serialized. Lock arbiter semantics early and test them under concurrent publish scenarios.
- State reconstruction can stay brittle if authoritative transitions are not journaled append-only. Lock event semantics early and test them directly.
- Overconfident memory outputs can be worse than missing memory. Make epistemic status explicit and test abstention/calibration directly.
- Replay can become disconnected from reality if no post-hoc outcomes are recorded. Keep an explicit outcome ledger and prefer real downstream signals over purely synthetic scores.
- Prompt packing can become opaque heuristic sludge if budget rules are not explicit. Keep planner decisions deterministic, budgeted, and inspectable.
- A memory system can be correct and still unsafe if disclosure semantics are vague. Lock audience rules early and test for cross-peer, cross-session, and assistant-internal leakage directly.
- Claim validity windows can become ambiguous if `observed_at`, `learned_at`, and `valid_*` semantics are not defined early. Lock those semantics in Sprint 1.
- The ontology can sprawl if too many classes are introduced too early. Keep v1 small and Hermes-driven.
- Policy packs can become a dumping ground for arbitrary flags. Keep them opinionated, versioned, and tightly scoped to host-visible behavior.
- Compiler dependency graphs can become either too coarse or too granular. Keep artifact boundaries explicit and validate invalidation behavior with fixtures.
- Snapshot storage can bloat or promotion can become ambiguous. Keep snapshots immutable, heads explicit, and retention bounded.
- Tiering can become ad hoc if too many levels or transitions are introduced. Keep the tier model small, deterministic, and policy-driven.
- Replay records can become bloated if the artifact format is not bounded. Keep a strict schema and version it.
- Codex SDK behavior may differ from Claude/OpenCode later. Keep the adapter interface narrow.
- Structured-output reliability needs explicit validation and retry strategy.
- Async prefetch should never become a hidden source of stale state.
- Migration and AI identity seeding should remain explicit flows, not implicit magic.
- If Hermes integration starts by mimicking Honcho tool names, keep those aliases at the boundary, not in the core model.

## Rollback Plan

- Keep each sprint independently usable.
- If `zvec` integration becomes problematic, retain SQLite and substitute a simpler retrieval layer.
- If Codex adapter proves brittle for a specific operation, disable only that operation behind the adapter, not the whole system.
- Preserve the host-facing API while iterating on storage/retrieval internals.

## Recommended Execution Order

1. Implement SQLite source-of-truth first.
2. Lock the subject graph before claim-ledger and memory-locus semantics spread.
3. Lock the observation-log, typed claim-ledger, and memory-locus model before belief revision and compiled views spread.
4. Lock the compiled view algebra before retrieval and host API behavior spread.
5. Lock the disclosure/audience layer before retrieval, prompt assembly, and host-facing read behavior spread.
6. Lock the budgeted prompt planner before prompt assembly and prompting logic spread.
7. Lock the epistemic-status layer before retrieval, prompt packing, and answer behavior spread.
8. Lock the outcome ledger before replay scoring and policy tuning spread.
9. Lock the memory transaction pipeline before async write, replay, snapshot publication, and prefetch behavior spread.
10. Lock the mutation arbiter before concurrent publication, snapshot promotion, and waterline signaling spread.
11. Lock the system event journal before reconstruction, replay, and debugging semantics spread.
12. Lock the durability contract before async behavior and host completion semantics spread.
13. Lock the typed memory ontology before retrieval and derivation policies spread.
14. Lock the first policy pack, `hermes_v1`, before retrieval and prompting logic spread.
15. Lock the compiler dependency model after subject, locus, and disclosure rules are explicit and before downstream compiled views spread across the system.
16. Lock the snapshot consistency model before prefetch and host-facing read contracts spread.
17. Lock the generational tiering model before retrieval and retention defaults spread.
18. Add Ollama embeddings and `zvec` retrieval next.
19. Add Codex adapter after retrieval.
20. Add turn artifact capture as part of the first end-to-end reasoning path.
21. Add incremental rebuild, snapshot promotion, and tier transition runtime before prefetch and host integration.
22. Add prefetch and migration after core retrieval/reasoning work.
23. Harden replay and adapter contracts last so Claude/OpenCode can be added later.
