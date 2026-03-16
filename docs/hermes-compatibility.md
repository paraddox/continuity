# Hermes Compatibility Notes

## Purpose

This document freezes the Hermes-visible replacement target that Continuity v1
must preserve. It is intentionally narrower than the full engine design in
`continuity-plan.md`: this file defines the shipped Hermes boundary, the
fixture taxonomy, and the normalized fixture bundle schema that later tests and
runtime tasks should consume directly.

Continuity v1 is scoped to the following deployment and ownership boundary:

- v1 first consumer is `internal Hermes patch only`.
- implementation language is `Python`.
- deployment is `embedded/in-process first`.
- deployment is `daemon later`, not daemon-first.
- embedded mode assumes `one owning Hermes process per Continuity SQLite store`.
- embedded mode uses `one serialized commit lane`.
- embedded mode uses `in-process worker threads only`.
- there is `no multi-process write coordination in v1`.

The goal is Hermes behavior compatibility at the boundary, not a hosted Honcho
clone and not a separate daemon product in v1.

## Hermes-visible parity target

The following surface is in scope for the internal Hermes embedded patch:

| Surface | Continuity v1 requirement | Harvested fixture IDs |
| --- | --- | --- |
| Session-backed startup wiring | Hermes creates or attaches one session wrapper for the active session key, attaches that session to the memory boundary, prewarms continuity when recall mode is not `tools`, and flushes pending writes on shutdown. | `async-prefetch-cross-turn-context` |
| Prompt continuity block | Prefetch builds a next-turn continuity block with user representation, AI peer representation, and optional continuity synthesis sections. `tools` mode skips prefetch-backed context injection. | `async-prefetch-cross-turn-context` |
| Honcho tool alias surface | Hermes exposes exactly four boundary aliases: `honcho_profile`, `honcho_search`, `honcho_context`, and `honcho_conclude`. These remain boundary-level aliases over Continuity behavior rather than core model names. | `honcho-tool-surface` |
| Tool behavior split | `honcho_profile` returns a peer card snapshot, `honcho_search` returns semantic excerpts, `honcho_context` is the dialectic answer path, and `honcho_conclude` is the durable user-fact write path. Tools must fail closed when no active Honcho or Continuity session is available. | `honcho-tool-surface` |
| Recall mode contract | Accepted recall modes are `hybrid`, `context`, and `tools`, with legacy alias `auto -> hybrid`. `context` injects memory and hides tools, `tools` shows tools and skips injected context, and `hybrid` does both. Invalid values fall back to `hybrid`. | `recall-mode-contracts` |
| Config precedence and coercion | Hermes-facing config resolution prefers the host block over root config, then falls back to environment values, then built-in defaults. `writeFrequency` accepts Hermes-like values such as `async`, `turn`, `session`, and integer `N` turns. `memoryMode` may be a string or an object with default plus peer overrides. | `host-config-resolution-cases` |
| Session naming and resolution | The session key used at the Hermes boundary remains a host-derived session identifier such as `telegram:123456`; downstream runtime work must preserve deterministic session resolution instead of inventing a new naming scheme. | `async-prefetch-cross-turn-context`, `host-config-resolution-cases` |
| Clearing behavior | The harvested Hermes source guarantees local session clearing only. Continuity must not treat this source slice as evidence of a durable forget contract. | `session-clear-without-durable-forget` |
| Migration behavior | Continuity must import prior transcript history and local memory files in a way that preserves Hermes-visible wrapper formats, upload naming, and peer targeting for migrated content. | `openclaw-migration-artifacts` |

## Intentional omissions

The following are explicitly out of scope for this v1 parity target:

- Any hosted or self-hosted Honcho server deployment model.
- Daemon-first behavior, network transport, or generic multi-host rollout in
  v1.
- Multi-process SQLite write coordination or multiple owning Hermes processes
  for one store.
- A durable forget API inferred from current Hermes source. The harvested slice
  only proves local session clear semantics, not durable deletion semantics.
- Provider-specific reasoning adapter details. The Hermes-visible requirement
  is only that the dialectic answer path stays distinct from durable conclude
  writes; the minimal typed reasoning contract is frozen separately.
- Reproducing every Honcho implementation detail internally. Continuity may
  change internals as long as the Hermes-visible behavior and later transport-
  neutral contract remain stable.

## Fixture taxonomy

- `core_engine`:
  Host-neutral substrate facts extracted from Hermes source that later engine
  fixtures can reuse without depending on Hermes prompt text or Honcho alias
  names.
- `hermes_parity`:
  Hermes-facing behaviors the embedded v1 patch must preserve exactly at the
  boundary.
- `service_contract`:
  Transport-neutral request or response expectations that later host API
  fixtures can validate independently from runtime internals.

Hermes is the first fixture producer, not the sole definition of engine truth.
When a harvested behavior is Hermes-specific, the family split keeps that fact
at the boundary rather than leaking it into host-neutral engine semantics.

## Normalized fixture bundle schema

Later fixture harnesses should assemble each family into one normalized fixture
bundle schema. The canonical harvested source remains
`tests/fixtures/hermes_corpus.json`, while the family manifests partition the
shared corpus:

- `tests/fixtures/core_engine/manifest.json`
- `tests/fixtures/hermes_parity/manifest.json`
- `tests/fixtures/service_contract/manifest.json`

Every assembled bundle must use the same envelope and fixture entry shape:

```json
{
  "bundle_version": "2026-03-16",
  "fixture_family": "hermes_parity",
  "producer": {
    "corpus": "tests/fixtures/hermes_corpus.json",
    "manifest": "tests/fixtures/hermes_parity/manifest.json"
  },
  "parity_target": "internal_hermes_embedded_patch_v1",
  "fixtures": [
    {
      "id": "honcho-tool-surface",
      "category": "tool_descriptions",
      "title": "Honcho tool surface exposes four distinct memory operations",
      "summary": "Short source-backed behavior summary.",
      "fixture_families": ["hermes_parity", "service_contract"],
      "normalized_input": {},
      "expected_behavior": {},
      "provenance": [
        {
          "path": "tools/honcho_tools.py",
          "line_start": 54,
          "line_end": 249
        }
      ]
    }
  ]
}
```

Bundle assembly rules:

1. Load `tests/fixtures/hermes_corpus.json` as the canonical harvested source.
2. Load the family manifest and preserve its `fixture_ids` order.
3. Resolve each ID to exactly one corpus fixture and reject unknown IDs.
4. Require the target `fixture_family` to appear in each fixture's
   `fixture_families` list.
5. Preserve the harvested `normalized_input`, `expected_behavior`, and
   `provenance` fields without adding family-specific ad hoc fields.
6. Keep the bundle host-neutral where possible:
   `core_engine` and `service_contract` fixtures may be harvested from Hermes,
   but their assertions should not depend on Hermes-only prompt phrasing or
   tool alias names unless the fixture is explicitly `hermes_parity`.

This schema is deliberately simple so later tasks can implement loaders,
validators, and harnesses without reopening `continuity-plan.md`.

## Harvested Hermes source corpus

`tests/fixtures/hermes_corpus.json` is the canonical harvested corpus for the
current Hermes Honcho replacement surface. It records source-backed fixture
inputs that later tasks can reuse without re-reading `~/.hermes/hermes-agent`
from scratch.

## Harvested categories

- `honcho_integration_behavior`:
  Session attachment, prewarm, next-turn context assembly, and cross-turn
  continuity behavior.
- `tool_descriptions`:
  The four Honcho tool contracts and the role split between cheap reads,
  dialectic answers, and durable conclusion writes.
- `recall_modes`:
  Recall mode normalization, tool visibility, and context injection behavior.
- `config_parsing_cases`:
  Hermes config parsing covers host-over-root precedence, environment fallback,
  `writeFrequency` coercion, and object-form `memoryMode`.
- `clearing_forgetting`:
  Current source covers session clear semantics; it does not expose a durable
  forget API in this harvested slice.
- `migration_examples`:
  Prior transcript upload, memory file wrapping, and the interactive migration
  walkthrough from OpenClaw files into Honcho peers.

## Inventory

| Fixture ID | Category | Primary sources | Fixture families |
| --- | --- | --- | --- |
| `async-prefetch-cross-turn-context` | `honcho_integration_behavior` | `honcho_integration/session.py`, `run_agent.py` | `core_engine`, `hermes_parity` |
| `honcho-tool-surface` | `tool_descriptions` | `tools/honcho_tools.py`, `honcho_integration/cli.py` | `hermes_parity`, `service_contract` |
| `recall-mode-contracts` | `recall_modes` | `honcho_integration/client.py`, `honcho_integration/cli.py`, `run_agent.py` | `hermes_parity`, `service_contract` |
| `host-config-resolution-cases` | `config_parsing_cases` | `honcho_integration/client.py`, Honcho integration tests | `core_engine`, `service_contract` |
| `session-clear-without-durable-forget` | `clearing_forgetting` | `honcho_integration/session.py`, `tests/honcho_integration/test_session.py` | `core_engine`, `hermes_parity` |
| `openclaw-migration-artifacts` | `migration_examples` | `honcho_integration/session.py`, `honcho_integration/cli.py`, migration transcript tests | `core_engine`, `hermes_parity`, `service_contract` |

## Notes for follow-up tasks

- `continuity-9pd.3` should implement the documented config precedence,
  `memoryMode`, `recallMode`, `writeFrequency`, and session-resolution rules
  rather than redefining them.
- `continuity-9pd.4` should freeze the minimal typed reasoning contract behind
  the already-documented Hermes-visible split between dialectic answers and
  durable conclude writes.
- `continuity-q3f.1.1` should build loaders and validators directly against the
  normalized fixture bundle schema in this document.
