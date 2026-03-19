# Plan: Continuity Hermes Backend Plugin

**Generated**: 2026-03-19
**Estimated Complexity**: Medium

## Overview

Continuity already has most of the internal Hermes-facing compatibility layer:

- `HermesMemoryConfig` for host-facing config parsing
- `create_continuity_backend()` for building a configured manager
- `ContinuityHermesSessionManager` for the Honcho-shaped session contract

What is still missing is the actual external backend provider entrypoint that
Hermes can load through its new abstraction layer. The goal of this work is to
turn the existing Continuity compatibility layer into a real out-of-tree Hermes
backend plugin, with the smallest possible new surface area.

The recommended approach is a **thin plugin entrypoint**:

- add `continuity.hermes_compat.plugin:create_backend`
- import Hermes' `memory_backends.base` types lazily at runtime
- adapt `create_continuity_backend()` output into a real
  `MemoryBackendBundle`
- keep all substantive runtime behavior in the existing
  `hermes_compat` config/factory/manager code

This avoids building a second adapter path, avoids daemon/HTTP indirection for
v1, and aligns directly with the upstream Hermes memory backend contract.

## Approaches Considered

### Option 1: Thin plugin entrypoint over existing `hermes_compat`

**Summary**: Add one new module that translates Continuity's existing Hermes
compatibility objects into Hermes' `MemoryBackendBundle` and
`MemoryBackendManifest`.

**Pros**
- Smallest new surface
- Reuses the manager and config layer that already exist
- Matches the external factory design in Hermes exactly
- Low risk of semantic drift between “local Hermes patch mode” and “plugin mode”

**Cons**
- The plugin module must import Hermes runtime types when loaded inside Hermes
- Continuity tests need optional or fake Hermes type coverage because Hermes is
  not installed in the Continuity repo by default

**Recommendation**: Yes. This is the best v1 path.

### Option 2: Build a second plugin-specific adapter layer

**Summary**: Keep current `hermes_compat` code as one path and build a separate
plugin adapter specifically for Hermes' external backend contract.

**Pros**
- Lets the plugin layer hide or remap differences explicitly

**Cons**
- Duplicates concepts we already have
- Increases maintenance burden immediately
- High risk of the two Hermes-facing paths drifting apart

**Recommendation**: No.

### Option 3: Expose Continuity over HTTP/daemon and make the plugin a client

**Summary**: Have the plugin speak to a local Continuity daemon instead of
embedding Continuity directly in the Hermes process.

**Pros**
- Decouples Hermes process lifecycle from Continuity runtime

**Cons**
- Much larger scope
- Adds deployment and IPC complexity before the embedded plugin path is proven
- Violates the current v1 assumption that embedded mode is the default

**Recommendation**: No for v1. Revisit only after the embedded plugin is stable.

## Prerequisites

- Hermes external backend contract is available upstream in:
  - `memory_backends/base.py`
  - `memory_backends/factory.py`
- Continuity repo remains installable into the same Python environment as
  Hermes
- Local Hermes config uses:
  - `hosts.hermes.experimental.memory_backend_factory`
- Local live validation environment still has:
  - Ollama embeddings available
  - Codex OAuth runtime available through Hermes when needed

## Sprint 1: Provider Entry Point

**Goal**: Make Continuity loadable by Hermes as a real external backend.

**Demo/Validation**
- Import `continuity.hermes_compat.plugin:create_backend` from a plain Python
  shell
- Build a valid Hermes `MemoryBackendBundle`
- Verify the manifest validates through Hermes' `validate_memory_backend_bundle`

### Task 1.1: Add plugin module
- **Location**:
  - `src/continuity/hermes_compat/plugin.py`
- **Description**:
  - Add a single public factory function `create_backend(host="hermes", config_path=None)`
  - Import Hermes backend types lazily inside the function body
  - Convert the output of `create_continuity_backend()` into:
    - `MemoryBackendManifest`
    - `MemoryBackendBundle`
- **Dependencies**: None
- **Acceptance Criteria**:
  - The factory matches Hermes' external loader signature
  - The plugin does not require Hermes imports at Continuity module import time
  - Disabled Continuity returns a disabled config + inactive manager cleanly
- **Validation**:
  - Focused unit test constructing the plugin bundle via fake Hermes type module

### Task 1.2: Define Continuity manifest mapping
- **Location**:
  - `src/continuity/hermes_compat/plugin.py`
- **Description**:
  - Create a fixed manifest mapping for the Continuity backend:
    - `protocol_version=1`
    - `backend_id="continuity"`
    - `display_name="Continuity"`
    - capability set derived from actual manager support
    - `config_source` from resolved config path
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Capabilities reflect real methods already implemented by
    `ContinuityHermesSessionManager`
  - Manifest never advertises unsupported capabilities
- **Validation**:
  - Test asserting exact capability set and required manifest fields

### Task 1.3: Export plugin from package surface
- **Location**:
  - `src/continuity/hermes_compat/__init__.py`
- **Description**:
  - Export the plugin factory if that improves discoverability
  - Keep the import graph safe so importing `continuity.hermes_compat` outside
    Hermes still works
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Optional package exports remain import-safe
  - No import cycle is introduced
- **Validation**:
  - Import test for `continuity.hermes_compat`

## Sprint 2: Contract Hardening

**Goal**: Prove the plugin satisfies the upstream Hermes backend contract.

**Demo/Validation**
- Validate a Continuity bundle against Hermes' backend contract functions
- Ensure bad config or missing runtime dependencies fail clearly

### Task 2.1: Add Hermes contract fixture types for Continuity tests
- **Location**:
  - `tests/test_hermes_plugin.py`
  - optionally `tests/fakes/hermes_memory_backend_types.py`
- **Description**:
  - Add a small fake module that mimics Hermes' `memory_backends.base` classes
    closely enough for Continuity-side tests
  - Use it to validate bundle shape without requiring a full Hermes install in
    the Continuity repo
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Continuity tests can exercise plugin output deterministically
  - Tests do not depend on the local Hermes checkout being on `PYTHONPATH`
- **Validation**:
  - Unit test proving plugin output instantiates fake Hermes manifest/bundle

### Task 2.2: Add plugin success-path tests
- **Location**:
  - `tests/test_hermes_plugin.py`
- **Description**:
  - Cover:
    - continuity backend enabled
    - continuity backend disabled
    - config path propagation
    - manifest fields
    - capability set
    - manager presence/absence
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Plugin output is deterministic and contract-shaped
- **Validation**:
  - `pytest tests/test_hermes_plugin.py`

### Task 2.3: Add failure-path tests
- **Location**:
  - `tests/test_hermes_plugin.py`
- **Description**:
  - Cover:
    - missing embedding dimensions for `zvec`
    - bad continuity config
    - unavailable Hermes Codex runtime fallback path
    - plugin import safety outside Hermes
- **Dependencies**: Task 2.2
- **Acceptance Criteria**:
  - Failures are explicit and actionable
  - Plugin does not silently fake missing runtime support
- **Validation**:
  - Focused negative tests with exact error assertions

## Sprint 3: Hermes Wiring Proof

**Goal**: Prove Hermes can actually load Continuity through the new abstraction.

**Demo/Validation**
- Hermes config points at `continuity.hermes_compat.plugin:create_backend`
- Hermes status shows external backend active
- `honcho_*` tools operate through Continuity using the external loader path

### Task 3.1: Document Hermes config snippet for external loading
- **Location**:
  - `README.md`
  - `docs/hermes-compatibility.md`
- **Description**:
  - Add the exact config shape for Hermes:
    - `hosts.hermes.experimental.memory_backend_factory`
  - Document the Continuity-specific host block fields that the plugin expects
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - A developer can configure Hermes without reading source
  - The docs distinguish built-in Honcho config from external Continuity config
- **Validation**:
  - Doc readback
  - packaging/bootstrap test if appropriate

### Task 3.2: Add Hermes local smoke runbook
- **Location**:
  - `README.md`
  - optionally `docs/hermes-compatibility.md`
- **Description**:
  - Document:
    - install Continuity into Hermes venv
    - set `memory_backend_factory`
    - run `hermes honcho status`
    - run one conclude/search/context/profile smoke
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - The external plugin path is reproducible from docs
- **Validation**:
  - Manual readback

### Task 3.3: Perform live Hermes smoke against plugin path
- **Location**:
  - runtime validation only
- **Description**:
  - In the Hermes checkout:
    - configure `memory_backend_factory`
    - verify external backend loads
    - verify at least one real memory operation works
- **Dependencies**: Sprint 1, Sprint 2
- **Acceptance Criteria**:
  - Hermes recognizes Continuity as an external backend
  - Continuity manager methods are reached through the upstream backend seam
- **Validation**:
  - `hermes honcho status`
  - direct or tool-driven smoke over `honcho_profile`, `honcho_search`,
    `honcho_context`, `honcho_conclude`

## Testing Strategy

- **Continuity unit tests**
  - `tests/test_hermes_compat.py`
  - new `tests/test_hermes_plugin.py`
- **Continuity packaging/docs checks**
  - readback + existing bootstrap/doc tests where relevant
- **Hermes contract proof**
  - local external backend smoke using the actual Hermes checkout
- **No daemon path in v1**
  - keep validation scoped to embedded external backend loading only

## Potential Risks & Gotchas

- **Runtime import dependency on Hermes types**
  - The plugin must not import `memory_backends.base` at module import time or
    Continuity becomes impossible to import outside Hermes
- **Capability drift**
  - The manifest must stay aligned with the actual manager methods
- **Config duplication**
  - Avoid inventing a second config schema if `HermesMemoryConfig` already
    covers the needed fields
- **Reasoning runtime mismatch**
  - Hermes Codex OAuth may not always be available in the environment running
    the plugin; the fallback behavior needs to stay explicit
- **Embedding backend assumptions**
  - `zvec` requires dimensions and environment support; tests should not assume
    it by default
- **Semantic drift between local patch mode and plugin mode**
  - The plugin must delegate to existing `hermes_compat` internals instead of
    re-implementing behavior

## Rollback Plan

- Remove `src/continuity/hermes_compat/plugin.py`
- Remove plugin-specific tests/docs
- Leave `hermes_compat` manager/factory/config in place for local embedded
  experiments if needed
- Remove the `memory_backend_factory` config from Hermes and fall back to
  built-in Honcho

