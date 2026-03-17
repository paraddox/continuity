#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from continuity.transactions import DurabilityWaterline
from tests.fixtures.core_engine_harness import execute_core_engine_bundle
from tests.fixtures.shared import load_fixture_bundle


class CoreEngineFixtureHarnessTests(unittest.TestCase):
    def test_executes_core_engine_bundle_in_manifest_order(self) -> None:
        bundle = load_fixture_bundle("core_engine", root_dir=ROOT_DIR)
        executions = execute_core_engine_bundle(root_dir=ROOT_DIR)

        self.assertEqual(tuple(executions), tuple(fixture.id for fixture in bundle.fixtures))

    def test_async_prefetch_fixture_stays_host_neutral_while_rendering_continuity_sections(self) -> None:
        execution = execute_core_engine_bundle(root_dir=ROOT_DIR)["async-prefetch-cross-turn-context"]

        self.assertEqual(execution["session_id"], "telegram:123456")
        self.assertEqual(execution["session_name"], "telegram:123456")
        self.assertEqual(
            execution["startup_actions"],
            (
                "attach session context to honcho tools",
                "prewarm context cache when recall mode is not tools",
                "register an exit hook that flushes pending Honcho writes",
            ),
        )
        self.assertEqual(
            execution["prefetch_sections"],
            (
                "## User representation",
                "## AI peer representation",
                "## Continuity synthesis",
            ),
        )
        self.assertTrue(execution["prefetch_enabled"])
        self.assertTrue(execution["tools_mode_skips_prefetch"])
        self.assertIn("User prefers concise technical answers.", execution["rendered_prefetch_block"])
        self.assertIn("Hermes keeps continuity via Honcho.", execution["rendered_prefetch_block"])

    def test_host_config_fixture_resolves_core_runtime_inputs_and_write_waterlines(self) -> None:
        execution = execute_core_engine_bundle(root_dir=ROOT_DIR)["host-config-resolution-cases"]

        self.assertEqual(
            execution["resolution_order"],
            (
                "host block",
                "root config",
                "environment fallback",
                "built-in defaults",
            ),
        )
        self.assertEqual(execution["resolved_workspace"], "host-ws")
        self.assertEqual(execution["resolved_ai_peer"], "host-ai")
        self.assertEqual(execution["coerced_write_frequency"], 3)
        self.assertEqual(execution["memory_mode_default"], "hybrid")
        self.assertEqual(execution["memory_mode_for_hermes"], "honcho")
        self.assertEqual(
            execution["write_frequency_waterlines"],
            {
                "async": DurabilityWaterline.OBSERVATION_COMMITTED.value,
                "turn": DurabilityWaterline.SNAPSHOT_PUBLISHED.value,
                "session": DurabilityWaterline.OBSERVATION_COMMITTED.value,
                "5": DurabilityWaterline.OBSERVATION_COMMITTED.value,
            },
        )

    def test_session_clear_fixture_clears_local_cache_without_claiming_durable_forget(self) -> None:
        execution = execute_core_engine_bundle(root_dir=ROOT_DIR)["session-clear-without-durable-forget"]

        self.assertEqual(execution["messages_before"], ("msg1", "msg2"))
        self.assertEqual(execution["messages_after"], ())
        self.assertEqual(execution["cleared_message_count"], 2)
        self.assertTrue(execution["cleared_after_last_message"])
        self.assertFalse(execution["durable_forget_implied_by_fixture"])

    def test_openclaw_migration_fixture_preserves_wrappers_and_import_waterline(self) -> None:
        execution = execute_core_engine_bundle(root_dir=ROOT_DIR)["openclaw-migration-artifacts"]

        self.assertEqual(execution["operation"], "import_history")
        self.assertEqual(execution["default_minimum_waterline"], DurabilityWaterline.VIEWS_COMPILED.value)
        self.assertEqual(
            execution["upload_names"],
            (
                "prior_history.txt",
                "consolidated_memory.md",
                "user_profile.md",
                "agent_soul.md",
            ),
        )
        self.assertEqual(execution["entry_sources"], ("local_jsonl", "local_memory", "local_memory", "local_memory"))
        self.assertEqual(execution["target_peers"], (None, "user", "user", "ai"))
        self.assertTrue(execution["history_entry"].startswith("<prior_conversation_history>"))
        self.assertTrue(execution["memory_entries"][0].startswith("<prior_memory_file>"))


if __name__ == "__main__":
    unittest.main()
