# Beads Backlog Creation Prompt

Use the prompt below verbatim with an LLM to analyze `continuity-plan.md` and create a dependency-aware beads backlog.

```text
You are working in `/home/soso/honcho-clone/continuity`.

Your job is backlog creation only: analyze the plan, then create a dependency-aware beads backlog that is actually implementable.

Read and obey first:
- `/home/soso/honcho-clone/continuity/AGENTS.md`
- `/home/soso/honcho-clone/continuity/continuity-plan.md`

Goal:
Convert `continuity-plan.md` into a clean set of beads epics and tasks that an engineer can implement directly without having to rediscover architecture decisions.

Settled decisions you must preserve:
- v1 first consumer is `internal Hermes patch only`
- implementation language is `Python`
- deployment is `embedded/in-process first`, `daemon later`
- do not create v1 work that assumes daemon delivery as the primary runtime
- embedded-mode ownership model is:
  - one owning Hermes process per Continuity SQLite store
  - one serialized commit lane
  - in-process worker threads only
  - no multi-process write coordination in v1
- the canonical typed host API contract must be frozen before host integration work spreads
- reasoning writes are schema-hard:
  - observation capture may still commit
  - no claims, belief updates, or compiled-view mutations may publish unless reasoning output passes schema validation
- fixture taxonomy must be preserved:
  - `core engine fixtures`
  - `Hermes parity fixtures`
  - `service-contract fixtures`
- the plan is intentionally comprehensive; do not de-scope it
- use beads priorities and dependencies instead of cutting scope

Required workflow:
1. Read `AGENTS.md` and `continuity-plan.md` carefully.
2. Inspect local beads docs / CLI help before creating issues:
   - `bd --help`
   - `bd create --help`
   - `bd update --help`
   - any local README / QUICKSTART / docs needed to confirm exact dependency syntax
   Do not guess beads syntax.
3. Derive a backlog structure:
   - epics = coherent implementation milestones with a clear done condition
   - tasks = implementation-ready units with concrete outcomes, validation, and likely files/modules/tests
   - use the plan’s existing tasks as baseline granularity
   - split only if a plan task is too broad for one focused implementation task
   - merge only if a plan task is too small to be useful as a standalone beads issue
4. Preserve implementation order and dependency structure from the plan, but do not blindly map `1 sprint = 1 epic` if a better dependency grouping exists.
5. Create the beads issues using `bd` in JSON mode:
   - create epics first
   - create tasks second
   - link dependencies after IDs exist
   - do not claim issues
   - do not close issues
   - do not implement code
6. End with a report showing what was created and how it maps back to the plan.

Backlog quality bar:
- Every task must be implementable by an engineer without rereading the full plan.
- Every task must have explicit acceptance criteria and validation.
- Every task should map to concrete modules/files/tests from the plan where possible.
- Avoid issue explosion.
- Avoid vague umbrella tasks like “work on X”.
- Avoid administrative-only micro-tasks unless they are required by the implementation flow.
- Keep host-neutral engine work separate from Hermes-boundary work.
- Keep daemon-later preparation tasks architectural and contract-focused, not premature runtime implementation.
- Keep contract-first work at the front of the backlog.

Epic design rules:
- Each epic must have:
  - title
  - purpose
  - scope
  - done condition
  - sequencing notes if needed
- Epics should represent real implementation milestones, not just document headings.

Task design rules:
- Each task must have:
  - clear outcome
  - likely files/modules/tests involved
  - acceptance criteria
  - validation method
  - dependencies
- Prefer tasks that produce a meaningful verified increment.
- Keep cross-cutting tasks rare and only when the boundary is crisp.

The backlog must cover at least:
- Hermes parity and contract work
- core SQLite / store / transaction / arbiter / event infrastructure
- retrieval / views / prompt planning
- reasoning / claim derivation
- compiler / snapshot / tiering / prefetch / migration / host API
- hardening / replay / evals
- deployment-boundary / transport-neutral API work
- fixture infrastructure

Issue creation rules:
- Use beads only. Do not create markdown TODOs.
- Use non-interactive commands only.
- Use `--json` where appropriate.
- If dependency syntax or epic/task linking workflow is unclear, read local docs/help until it is clear.
- If a part of the plan is genuinely ambiguous, do not invent behavior. Create the smallest safe issue only if it is clearly needed; otherwise report the ambiguity explicitly.

Output format after creation:
1. Short summary of the epic structure
2. Epic-by-epic list of created IDs and task IDs
3. Dependency notes
4. Any plan tasks that were intentionally split or merged, and why
5. Any ambiguity or follow-up decisions still worth resolving before implementation begins

Constraints:
- Do not edit `continuity-plan.md`
- Do not edit source files
- Do not implement anything
- This is backlog construction only

Start now.
```
