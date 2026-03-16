# Continuity Implementation Prompt

Use the prompt below verbatim with an LLM to implement Continuity from the existing beads backlog.

```text
You are working in `/home/soso/honcho-clone/continuity`.

Your job is implementation only: use the existing beads backlog to build Continuity. Do not re-plan the project and do not redesign the backlog unless you discover a real gap while implementing.

Read and obey first:
- `/home/soso/honcho-clone/continuity/AGENTS.md`
- `/home/soso/honcho-clone/continuity/continuity-plan.md`

Project rules you must preserve:
- v1 first consumer is `internal Hermes patch only`
- implementation language is `Python`
- deployment is `embedded/in-process first`, `daemon later`
- do not implement daemon-first runtime behavior in v1
- embedded-mode ownership model is:
  - one owning Hermes process per Continuity SQLite store
  - one serialized commit lane
  - in-process worker threads only
  - no multi-process write coordination in v1
- the canonical typed host API contract is transport-neutral and must remain frozen
- reasoning writes are schema-hard:
  - observation capture may still commit
  - no claims, belief updates, or compiled-view mutations may publish unless reasoning output passes schema validation
- fixture taxonomy must be preserved:
  - `core engine fixtures`
  - `Hermes parity fixtures`
  - `service-contract fixtures`

Backlog source of truth:
- The beads backlog is the implementation source of truth.
- Use `continuity-plan.md` only for deeper context, invariants, and file references when the issue description is not enough.

Important backlog shape:
- Some backlog items are umbrella trackers and should not be implemented directly if narrower child tasks exist.
- In particular, do not pick these umbrellas for direct implementation:
  - `continuity-x9j.8`
  - `continuity-x9j.10`
  - `continuity-9br.6`
  - `continuity-q3f.1`
- Prefer their child tasks.

Required workflow:
1. Inspect the backlog with full visibility:
   - `bd list --json --limit 0`
   - `bd ready --json --limit 0`
2. Pick one ready implementation task at a time.
   - Prefer the highest-priority ready child task.
   - Do not start from epics if a concrete child task is ready.
3. Before coding, inspect the selected issue in detail:
   - `bd show <id> --json`
   - `bd dep list <id> --json`
4. Claim the task:
   - `bd update <id> --claim --json`
5. Implement the code and tests needed for that one task.
6. Verify with real commands before claiming success.
7. Close the issue only after verification succeeds.
8. Finish the session properly:
   - `git pull --rebase`
   - `bd dolt push`
   - `git push`
   - `git status`

Implementation rules:
- Use beads only for task tracking.
- Do not create markdown TODO lists.
- Use non-interactive commands only.
- Use `--json` for beads commands where possible.
- Keep changes tightly scoped to the claimed issue.
- Follow existing architecture and module boundaries from the plan.
- Do not redesign settled decisions.
- Do not implement speculative daemon runtime features unless the claimed issue explicitly calls for daemon-later contract work.
- Keep host-neutral engine work separate from Hermes-boundary work.

When selecting work:
- Start with `bd ready --json --limit 0`.
- Choose a real implementation task, not an epic.
- If multiple tasks are ready, prefer:
  1. contract and fixture groundwork
  2. SQLite/runtime substrate
  3. retrieval
  4. reasoning
  5. incremental runtime / host integration
  6. hardening / replay / evals

When an issue is too broad or wrong:
- First check whether it already has narrower child tasks.
- If yes, work the child task instead.
- If no, and implementation would be unsafe without splitting, create a new beads task linked with `discovered-from:<current-id>`.
- Do not silently widen scope.

Verification standard:
- Never claim completion without fresh verification output.
- Run the exact tests or checks that prove the claimed behavior.
- If tests are not possible, say exactly what was and was not verified.
- Prefer issue-local verification first, then broader regression checks if warranted.

Expected output style during work:
- Concise status updates
- Explicit mention of claimed issue ID
- Explicit verification evidence
- Clear note of any follow-up issue created

Completion report format:
1. Claimed issue ID and title
2. What changed
3. Verification commands run and results
4. Any follow-up beads created
5. Final git/beads sync status

Constraints:
- Do not rewrite `continuity-plan.md` unless the claimed issue is explicitly about the plan or docs
- Do not create new architecture unless required by the claimed issue
- Do not skip tests just because the code “looks right”
- Do not work multiple unrelated tasks in one pass

Start by listing ready work with full visibility and choosing the best ready child task.
```
