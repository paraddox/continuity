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

The second core rule is that memory must have one canonical intermediate representation. Raw observations are immutable inputs. Typed claims are the only durable derived primitive. Beliefs, profiles, summaries, prompt blocks, and vector documents are materializations over claims rather than co-equal durable roots.

The third core rule is that memory must be revisable over time. Continuity should optimize for current belief state, not just historical accumulation.

The fourth core rule is that memory decisions must be replayable. Continuity should preserve enough turn-time artifacts to re-run retrieval, claim selection, belief selection, and reasoning later under alternate strategies.

The fifth core rule is that memory must be typed. Continuity should explicitly model what kind of memory each observation and claim represents so revision, retrieval, and rendering rules are type-aware rather than generic.

The sixth core rule is that memory behavior must be versioned. Continuity should bundle ontology, revision, retrieval, rendering, and derivation behavior into explicit policy versions that can be inspected, replayed, and evolved safely.

The seventh core rule is that memory must be incrementally compilable. Continuity should treat observations, claims, policy packs, and adapter behavior as inputs to a compiler that rebuilds only the affected materializations when something changes.

The eighth core rule is that memory reads must be snapshot-consistent. Continuity should expose coherent, branchable memory snapshots so hosts never read a mixed partially rebuilt state.

The ninth core rule is that memory must be tiered generationally. Continuity should separate hot, warm, cold, and frozen memory so retrieval quality, rebuild cost, storage growth, and prompt efficiency stay bounded over long runtimes.

## Design Direction

### Core Architecture

Use a split architecture:
- relational state in `SQLite`
- semantic retrieval in `zvec`
- reasoning in a pluggable SDK adapter
- an immutable `Observation Log` for normalized source messages and imports
- a typed, append-only `Claim Ledger` for provenance-linked memory assertions
- materializers for current beliefs, profiles/cards, summaries, and prompt-ready memory blocks
- an `Evidence Graph` built over observations, claims, and materializations
- a `Temporal Belief Revision Engine` that turns claims into current beliefs
- a `Counterfactual Replay Engine` that records and replays turn-time memory decisions
- a `Typed Memory Ontology` that defines memory classes and their lifecycle rules
- a `Versioned Memory Policy Layer` that defines named behavior packs such as `hermes_v1`
- an `Incremental Memory Compiler` that tracks dependencies, invalidation, and selective rebuilds from observations to claims to materializations
- a `Snapshot Consistency Layer` that exposes atomic read states and candidate rebuild branches
- a `Generational Memory Tiering Layer` that governs admission, promotion, demotion, and archival behavior

This is the right fit because Hermes needs more than vectors:
- peers
- sessions
- messages
- observations
- claims
- claim_sources
- claim_relations
- claim derivation runs
- belief/materialization state
- profile/card materializations
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
- `claims`: append-only, typed assertions derived from observations or explicit host writes
- `materializations`: current beliefs, peer cards, summaries, prompt blocks, vector documents, and other host-facing projections over claims

Claims should carry the fields needed to support revision without mutating history:
- identity and claim type
- scope (`user`, `assistant`, `peer`, `session`, `shared`)
- provenance links to source observations and derivation runs
- confidence / support metadata
- contradiction, supersession, and correction edges
- `observed_at`
- `learned_at`
- `valid_from`
- `valid_to`

This is the subsystem boundary that simplifies everything else:
- belief revision becomes claim selection and projection instead of state mutation across multiple roots
- profiles, cards, and summaries become materializations instead of special storage primitives
- compiler invalidation works on one IR instead of bespoke domain tables
- replay can compare claim sets and materializations directly

Hermes-visible operations like conclusion writes or profile reads can still exist at the API boundary, but internally they should land in the claim ledger and then flow into materializations.

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

The Evidence Graph should be a first-class storage and API concern, not a debugging afterthought. It should be implemented over observations, claims, and claim-to-materialization edges rather than parallel durable fact tables.

### Temporal Belief Revision

Continuity should not treat all remembered claims as equally current.

It should maintain an explicit belief layer on top of the claim ledger:
- observations record what was said or seen
- claims express typed candidate memory assertions
- beliefs represent the system's current best materialized view
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
- recency of supporting messages
- contradiction or supersession edges
- source diversity
- explicit user corrections

This provides the highest-value improvement to memory quality:
- fewer stale assumptions
- better handling of changing preferences
- better “what changed?” answers
- safer long-running memory accumulation

### Counterfactual Replay

Continuity should not only store memory state. It should store how memory decisions were made on each turn.

For each answered turn, the system should preserve a canonical decision record containing:
- the retrieval query and retrieval candidates
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
- per-type belief revision rules
- retrieval and ranking profiles
- prompt rendering rules
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
- policy packs and policy versions
- reasoning adapter versions and output schemas
- prompt-rendering rules where they affect stored artifacts

Treat these as canonical derived IR:
- claims
- claim relations

Treat these as compiled materializations:
- beliefs
- peer cards and representations
- summaries
- prompt memory caches
- vector-index records

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

The compiler should be a first-class subsystem, not an implementation detail hidden inside prefetch or storage code.

### Snapshot Consistency Layer

Continuity should never answer from a half-rebuilt memory state.

It should expose immutable memory snapshots, each representing one coherent read view over:
- beliefs
- peer cards and representations
- summaries
- prompt memory caches
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
- Evidence-backed claims, with provenance, supersession, correction, and validity tracking
- Explicit current-belief management with freshness and contradiction handling
- Turn-level decision capture and offline replay for retrieval, belief, and reasoning evaluation
- Type-aware memory classification, policy enforcement, and prompt rendering
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
- `src/continuity/compiler.py`
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
- Define the observation-log and claim-ledger invariants.
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

### Task 1.4: Define observation and claim-ledger invariants
- **Location**: `src/continuity/store/claims.py`, `docs/architecture.md`, `tests/test_claim_model.py`
- **Description**: Define the canonical memory IR:
  - observations are immutable normalized source records
  - claims are append-only, typed, scoped, and provenance-linked
  - claims carry `observed_at`, `learned_at`, `valid_from`, and `valid_to` where applicable
  - claim relations encode support, supersession, contradiction, and correction
  - profile/card materializations are projections over active claims
  - dialectic answers may include supporting claim and observation payloads
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - The observation/claim rules are explicit and testable.
  - No durable derived memory artifact exists outside the claim ledger.
  - No host-visible memory artifact exists without claim provenance.
- **Validation**:
  - Invariant tests and architecture review.

### Task 1.5: Define belief revision invariants
- **Location**: `docs/architecture.md`, `tests/test_belief_revision_model.py`
- **Description**: Define the required temporal belief model:
  - beliefs are projections over evidence-backed claims
  - contradictory claims do not silently coexist as equally active truth
  - explicit corrections can supersede older beliefs
  - stale beliefs can decay without deleting source evidence
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
  - how a policy supplies claim-derivation, per-type revision, and retrieval rules
  - how prompt rendering and materialization behavior are attached to a policy
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
  - which artifacts are source inputs, canonical claims, and downstream materializations
  - how dependencies and fingerprints are represented
  - how dirtying and selective rebuild work
  - how policy-version and adapter-version changes trigger invalidation
  - how rebuild reasons are surfaced for inspection and replay
- **Dependencies**: Tasks 1.4, 1.6, 1.7, and 1.8
- **Acceptance Criteria**:
  - Source inputs, claims, and compiled materializations are explicitly separated.
  - Invalidation and rebuild rules are deterministic and inspectable.
- **Validation**:
  - Invariant tests and architecture review.

### Task 1.10: Define snapshot consistency invariants
- **Location**: `src/continuity/snapshots.py`, `docs/architecture.md`, `tests/test_snapshot_model.py`
- **Description**: Define the snapshot model for host-visible reads:
  - what belongs to a snapshot
  - how snapshots reference compiled artifacts
  - how active heads and candidate branches are represented
  - how promotion, rollback, and diffing work
  - how reads pin to a snapshot during retrieval, prompting, and replay
- **Dependencies**: Tasks 1.6, 1.8, and 1.9
- **Acceptance Criteria**:
  - Snapshot reads are coherent and immutable.
  - Promotion from candidate to active is explicit and inspectable.
- **Validation**:
  - Invariant tests and architecture review.

### Task 1.11: Define generational tiering invariants
- **Location**: `src/continuity/tiers.py`, `docs/architecture.md`, `tests/test_tiers_model.py`
- **Description**: Define the tiering model for long-lived memory:
  - which tiers exist in v1
  - which ontology types and artifacts belong in each tier by default
  - how policy packs govern promotion, demotion, retention, and archival
  - how tiering affects retrieval defaults, compiler urgency, and snapshot inclusion
  - how replay and audit artifacts move into archival tiers without polluting active reads
- **Dependencies**: Tasks 1.7, 1.8, 1.9, and 1.10
- **Acceptance Criteria**:
  - Tier boundaries are explicit, small, and policy-driven.
  - Tiering affects retrieval and rebuild behavior without changing source-of-truth semantics.
- **Validation**:
  - Invariant tests and architecture review.

## Sprint 2: Build Durable Memory Storage

**Goal**: Build the SQLite-backed source-of-truth layer for peers, sessions, messages, observations, claims, and the materializations derived from claims.

**Demo/Validation**:
- Create peers, sessions, and messages.
- Create observations and provenance-linked claims.
- Create active beliefs and belief updates from claims.
- Persist replayable turn artifacts without any vector layer.
- Persist ontology types and policy metadata on observations, claims, and beliefs.
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
  - sessions
  - session_peers
  - messages
  - observations
  - claims
  - claim_sources
  - claim_relations
  - beliefs
  - belief_updates
  - derivation_runs
  - turn_artifacts
  - replay_runs
  - policy_versions
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
  - Derived claims can be traced to source observations and derivation runs.
  - Active beliefs can be derived, superseded, and inspected independently of raw evidence.
  - Replayable turn artifacts can be stored and loaded independently of retrieval engine internals.
  - Memory records can be typed and queried by ontology class.
  - Claim validity, scope, and relation metadata are explicit and queryable.
  - Beliefs, derivation runs, and turn artifacts can be traced to a concrete policy version.
  - Derived artifacts can be invalidated and selectively rebuilt from stored dependency metadata.
  - Host-visible artifacts can be resolved through a coherent snapshot head.
  - Artifacts can be assigned to explicit tiers and transitioned deterministically over time.
- **Validation**:
  - Schema migration and round-trip tests.

### Task 2.2: Implement repository layer
- **Location**: `src/continuity/store/sqlite.py`, `tests/test_store.py`
- **Description**: Build CRUD/query helpers for the schema.
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
  - supersede or demote stale beliefs
  - resolve explicit corrections
  - record belief update history
- **Dependencies**: Task 2.2
- **Acceptance Criteria**:
  - Current belief state is reproducible from stored evidence.
  - Belief updates are durable and inspectable.
  - Belief promotion, decay, and supersession can vary by memory type without hidden heuristics.
  - Belief updates are attributable to a specific policy version.
- **Validation**:
  - Unit tests for correction, contradiction, and preference-change cases.

### Task 2.4: Implement replay artifact repository
- **Location**: `src/continuity/store/replay.py`, `tests/test_replay_store.py`
- **Description**: Persist canonical turn decision records and replay runs:
  - save retrieval candidates and selected evidence
  - save active claims used for prompt assembly
  - save active beliefs used for prompt assembly
  - save memory block snapshots
  - save reasoning input/output envelopes
  - save replay comparison scores and notes
- **Dependencies**: Tasks 2.2 and 2.3
- **Acceptance Criteria**:
  - Replay artifacts are versioned, queryable, and immutable after capture.
  - Replay runs do not mutate source-of-truth conversation memory.
  - Replay records include the policy version used for the original and replayed decisions.
- **Validation**:
  - Repository round-trip tests.

### Task 2.5: Implement compiler state repository
- **Location**: `src/continuity/compiler.py`, `tests/test_compiler_store.py`
- **Description**: Persist and query compiler dependency state:
  - register dependencies between source inputs, claims, and compiled materializations
  - persist fingerprints and artifact versions
  - enqueue dirty artifacts with reason codes
  - query rebuild plans
- **Dependencies**: Tasks 2.2, 2.3, and 2.4
- **Acceptance Criteria**:
  - Compiler state is durable and queryable.
  - Dirtying one source input does not require scanning all artifacts heuristically.
- **Validation**:
  - Round-trip and selective-invalidation tests.

### Task 2.6: Implement snapshot repository
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

### Task 2.7: Implement tier state repository
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

### Task 2.8: Implement session manager on SQLite
- **Location**: `src/continuity/session_manager.py`, `tests/test_session_manager.py`
- **Description**: Implement:
  - peer/session creation
  - local message cache
  - sync/write-frequency logic
  - peer-specific memory-mode gating
- **Dependencies**: Task 2.3
- **Acceptance Criteria**:
  - Session manager works without retrieval/reasoning enabled.
  - Write-frequency rules behave deterministically.
- **Validation**:
  - Unit tests covering async/turn/session/N-turn modes.

## Sprint 3: Add Retrieval With zvec + Ollama

**Goal**: Build local semantic retrieval and belief-aware profile/context lookup over claims and their materializations.

**Demo/Validation**:
- Generate embeddings via Ollama.
- Index messages, observations, durable claims, and current beliefs in `zvec`.
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
- **Description**: Index messages, observations, evidence-bearing claims, and current beliefs into `zvec`.
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Index entries map back to SQLite records.
  - Rebuild-from-SQLite path exists.
  - Search hits can be resolved back to observation and claim nodes.
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
  - prompt-ready memory assembly
  - evidence-aware ranking and citation selection
  - belief-aware ranking with freshness and confidence
  - ontology-aware ranking and rendering by memory type
  - policy-pack-driven retrieval and prompt assembly
- **Dependencies**: Task 3.2
- **Acceptance Criteria**:
  - Retrieval works from local state only.
  - Recall-mode behavior is explicit.
  - Search results and prompt memory can expose provenance when needed.
  - Retrieval prefers active beliefs and well-supported current claims over stale historical claims by default.
  - Retrieval and prompt assembly can prioritize or suppress memory types based on context.
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
- **Description**: Use Codex adapter to derive durable typed claims and update peer cards/representations.
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Derived claims are persisted in SQLite.
  - Retrieval index updates after new claims.
  - Each derived claim links to the observations and derivation run that produced it.
  - Belief revision runs after claim derivation and updates the current belief state.
  - Derived claims are assigned ontology types before entering the belief layer.
  - Derivation runs record the policy version that governed classification and promotion.
  - Derived claims register compiler dependencies on observations, policy version, and adapter version.
  - Derived outputs can be staged into a candidate snapshot without mutating the active snapshot.
  - Derived outputs receive an initial tier assignment appropriate to their ontology type and policy.
- **Validation**:
  - End-to-end local tests with fixture conversations.

### Task 4.3: Implement dialectic-style answers
- **Location**: `src/continuity/api.py`, `tests/test_dialectic.py`
- **Description**: Use retrieval plus Codex adapter to answer natural-language questions about a peer.
- **Dependencies**: Tasks 3.3 and 4.1
- **Acceptance Criteria**:
  - Equivalent to Hermes’ `honcho_context` behavior at the user level.
  - The implementation can surface supporting evidence for “why do you think that?” style follow-ups.
  - Answers can distinguish current belief from older superseded evidence when relevant.
- **Validation**:
  - Snapshot/fixture tests and local smoke tests.

### Task 4.4: Capture turn decision records
- **Location**: `src/continuity/api.py`, `tests/test_turn_artifacts.py`
- **Description**: Record the artifacts needed for replay whenever Continuity answers a memory question or assembles a prompt memory block.
- **Dependencies**: Tasks 3.3, 4.1, and 4.3
- **Acceptance Criteria**:
  - Retrieval inputs, selected claims, selected beliefs, prompt memory block, and reasoning envelopes are durably captured.
  - Capture is deterministic and versioned.
  - Turn artifacts record the active policy pack.
  - Turn artifacts record the snapshot used for the read.
  - Turn artifacts and replay records are eligible for archival tiers by policy.
- **Validation**:
  - Fixture tests asserting stable serialized turn artifacts.

## Sprint 5: Compilation, Snapshots, Tiering, Prefetch, Migration, And Host Integration

**Goal**: Make the system practical for Hermes integration and keep derived memory fresh without broad recomputation.

**Demo/Validation**:
- Change a source observation and rebuild only affected claims and materializations.
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
  - rebuild affected claims, beliefs, cards, summaries, caches, and index records
  - expose rebuild reasons for inspection and replay
- **Dependencies**: Sprints 2, 3, and 4 complete
- **Acceptance Criteria**:
  - Source edits, policy upgrades, and adapter upgrades rebuild only affected claims and materializations.
  - Rebuild plans are deterministic and explainable.
  - Rebuild output can be materialized into a candidate snapshot rather than mutating the active head.
- **Validation**:
  - Fixture tests for corrections, policy upgrades, and adapter upgrades.

### Task 5.2: Implement snapshot builder and promoter
- **Location**: `src/continuity/snapshots.py`, `tests/test_snapshots_runtime.py`
- **Description**: Build the runtime that assembles and promotes coherent read states:
  - materialize compiled artifacts into candidate snapshots
  - atomically promote candidate snapshots to active heads
  - diff snapshots
  - expose branchable snapshots for replay and policy experiments
- **Dependencies**: Tasks 2.6 and 5.1
- **Acceptance Criteria**:
  - Hosts never read partially rebuilt state.
  - Candidate snapshot promotion is atomic and inspectable.
- **Validation**:
  - Fixture tests for concurrent rebuild/promotion scenarios.

### Task 5.3: Implement tiering and retention runtime
- **Location**: `src/continuity/tiers.py`, `tests/test_tiers_runtime.py`
- **Description**: Build the runtime that manages memory economics over time:
  - assign initial tiers from ontology and policy
  - promote and demote artifacts across tiers
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
- **Description**: Cache next-turn context and optional next-turn synthesis.
- **Dependencies**: Tasks 3.3, 4.3, 5.1, 5.2, and 5.3
- **Acceptance Criteria**:
  - First turn may be cold.
  - Next turns avoid unnecessary blocking work.
  - Prefetch never serves stale artifacts that are marked dirty by the compiler.
  - Prefetch reads and caches are pinned to a specific snapshot.
  - Prefetch respects tier-aware retrieval defaults.
- **Validation**:
  - Prefetch cache behavior tests.

### Task 5.5: Implement migration and identity seeding
- **Location**: `src/continuity/migration.py`, `tests/test_migration.py`
- **Description**: Support:
  - prior session history import
  - `MEMORY.md` and `USER.md` import
  - AI identity seeding from `SOUL.md`/similar files
  - provenance tagging for imported claims
- **Dependencies**: Sprint 2 complete
- **Acceptance Criteria**:
  - Migration is deterministic and replay-safe.
- **Validation**:
  - Fixture-based migration tests.

### Task 5.6: Expose a minimal host API
- **Location**: `src/continuity/api.py`, `tests/test_api.py`
- **Description**: Expose only the host-needed surface:
  - initialize
  - save turn
  - search
  - get profile
  - get prompt memory block
  - answer memory question
  - write conclusion
  - inspect evidence for a claim or answer
  - inspect turn decision record
  - inspect active policy pack and policy version
  - inspect compiler status and rebuild reasons
  - inspect current snapshot and snapshot diffs
  - inspect tier assignments and retention status
- **Dependencies**: Tasks 5.4 and 5.5
- **Acceptance Criteria**:
  - A host runtime does not need internal store/index details.
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
  - provenance completeness
  - supersession/conflict/correction behavior
  - bitemporal validity-window behavior
  - selective re-derivation after source changes
- Add belief revision tests for:
  - explicit correction handling
  - stale preference replacement
  - conflict resolution ordering
  - “what changed?” answer support
- Add ontology tests for:
  - type assignment for derived claims
  - per-type decay and supersession rules
  - prompt rendering differences by memory class
  - partitioning of user, assistant, and ephemeral memory
- Add policy tests for:
  - policy-pack selection and version stamping
  - policy-specific retrieval/rendering differences
  - policy-specific belief revision outcomes
  - policy-vs-policy replay comparison on the same turn set
- Add compiler tests for:
  - source-edit invalidation
  - explicit correction invalidation
  - policy-upgrade rebuild planning
  - adapter-upgrade rebuild planning
  - selective rebuild of only affected claims and materializations
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
  - retrieval-only counterfactual replays
  - belief-policy counterfactual replays
  - claim-set comparison on the same stored turns
  - adapter comparison on the same stored turns
- Keep all live tests opt-in.

## Potential Risks And Gotchas

- `zvec` should not become the source of truth. Treat it as rebuildable.
- The claim ledger can sprawl if claim granularity is vague. Keep claim boundaries explicit and Hermes-driven.
- The Evidence Graph can become complex quickly if provenance rules are vague. Keep observation-to-claim and claim-to-materialization edges explicit and minimal.
- Belief revision can become heuristic sludge if scoring rules are underdefined. Keep revision rules explicit, deterministic, and inspectable.
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
2. Lock the observation-log and typed claim-ledger model before belief revision and materializations spread.
3. Lock the typed memory ontology before retrieval and derivation policies spread.
4. Lock the first policy pack, `hermes_v1`, before retrieval and prompting logic spread.
5. Lock the compiler dependency model before downstream materializations spread across the system.
6. Lock the snapshot consistency model before prefetch and host-facing read contracts spread.
7. Lock the generational tiering model before retrieval and retention defaults spread.
8. Add Ollama embeddings and `zvec` retrieval next.
9. Add Codex adapter after retrieval.
10. Add turn artifact capture as part of the first end-to-end reasoning path.
11. Add incremental rebuild, snapshot promotion, and tier transition runtime before prefetch and host integration.
12. Add prefetch and migration after core retrieval/reasoning work.
13. Harden replay and adapter contracts last so Claude/OpenCode can be added later.
