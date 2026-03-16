# Hermes Compatibility Notes

## Harvested Hermes source corpus

`tests/fixtures/hermes_corpus.json` is the canonical harvested corpus for the
current Hermes Honcho replacement surface. It records source-backed fixture
inputs that later tasks can reuse without re-reading `~/.hermes/hermes-agent`
from scratch.

This issue only seeds harvested inputs and provenance. It does not redefine the
typed Continuity contract or the broader parity target.

## Fixture taxonomy

- `core_engine`: host-neutral substrate facts extracted from Hermes source that
  later engine fixtures can reuse without depending on Hermes prompt text.
- `hermes_parity`: Hermes-facing behaviors the embedded v1 patch must preserve.
- `service_contract`: transport-neutral contract expectations that later host
  API fixtures can validate independently from runtime internals.

## Harvested categories

- `honcho_integration_behavior`: session attachment, prewarm, next-turn context
  assembly, and cross-turn continuity behavior.
- `tool_descriptions`: the four Honcho tool contracts and the role split
  between fast reads, dialectic answers, and durable conclusion writes.
- `recall_modes`: recall mode normalization, config parsing, tool visibility,
  and context injection behavior.
- `config_parsing_cases`: host-over-root precedence, environment fallback,
  write frequency coercion, and object-form `memoryMode`.
- `clearing_forgetting`: current source covers session clear semantics; it does
  not expose a durable forget API in this harvested slice.
- `migration_examples`: prior transcript upload, memory file wrapping, and the
  interactive migration walkthrough from OpenClaw files into Honcho peers.

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

- The corpus is intentionally normalized around reusable fixture inputs, not
  around full executable tests yet.
- The `clearing_forgetting` category is narrow on purpose: session clear is
  harvested from current Hermes source, while durable forgetting remains a later
  Continuity concern.
- The migration entries preserve file-to-peer mapping and wrapper formats so
  later import work can reproduce Hermes-compatible behavior without mining the
  CLI again.
