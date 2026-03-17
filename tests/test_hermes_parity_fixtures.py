#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tests.fixtures.hermes_parity_harness import execute_hermes_parity_bundle
from tests.fixtures.shared import load_fixture_bundle


class HermesParityFixtureHarnessTests(unittest.TestCase):
    def test_executes_hermes_parity_bundle_in_manifest_order(self) -> None:
        bundle = load_fixture_bundle("hermes_parity", root_dir=ROOT_DIR)
        executions = execute_hermes_parity_bundle(root_dir=ROOT_DIR)

        self.assertEqual(tuple(executions), tuple(fixture.id for fixture in bundle.fixtures))

    def test_async_prefetch_fixture_preserves_hermes_startup_boundary(self) -> None:
        execution = execute_hermes_parity_bundle(root_dir=ROOT_DIR)["async-prefetch-cross-turn-context"]

        self.assertEqual(execution["session_id"], "telegram:123456")
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
        self.assertTrue(execution["empty_session_triggers_memory_migration"])
        self.assertTrue(execution["tools_mode_skips_prefetch"])
        self.assertIn("User prefers concise technical answers.", execution["continuity_block"])
        self.assertIn("Hermes keeps continuity via Honcho.", execution["continuity_block"])

    def test_honcho_tool_fixture_keeps_aliases_structured_results_and_closed_session_gate(self) -> None:
        execution = execute_hermes_parity_bundle(root_dir=ROOT_DIR)["honcho-tool-surface"]

        self.assertEqual(
            execution["tool_names"],
            (
                "honcho_profile",
                "honcho_search",
                "honcho_context",
                "honcho_conclude",
            ),
        )
        self.assertEqual(
            execution["operation_bindings"],
            {
                "honcho_profile": "get_profile_view",
                "honcho_search": "search",
                "honcho_context": "answer_memory_question",
                "honcho_conclude": "write_conclusion",
            },
        )
        self.assertEqual(execution["cheaper_read_tools"], ("honcho_profile", "honcho_search"))
        self.assertEqual(execution["dialectic_tool"], "honcho_context")
        self.assertTrue(execution["session_bound_results_are_json_strings"])
        self.assertEqual(
            execution["inactive_session_errors"],
            {
                "honcho_profile": "Honcho is not active for this session.",
                "honcho_search": "Honcho is not active for this session.",
                "honcho_context": "Honcho is not active for this session.",
                "honcho_conclude": "Honcho is not active for this session.",
            },
        )

    def test_recall_mode_fixture_hides_tools_only_in_context_mode(self) -> None:
        execution = execute_hermes_parity_bundle(root_dir=ROOT_DIR)["recall-mode-contracts"]

        self.assertEqual(execution["accepted_modes"], ("hybrid", "context", "tools"))
        self.assertEqual(execution["legacy_aliases"], {"auto": "hybrid"})
        self.assertEqual(execution["invalid_values_fall_back_to"], "hybrid")
        self.assertEqual(
            execution["runtime_examples"],
            {
                "context": {"tools_visible": False, "context_injected": True},
                "tools": {"tools_visible": True, "context_injected": False},
                "hybrid": {"tools_visible": True, "context_injected": True},
            },
        )
        self.assertTrue(execution["host_block_overrides_root_config"])
        self.assertTrue(execution["tools_mode_skips_startup_prefetch"])
        self.assertTrue(execution["context_mode_hides_honcho_tools"])

    def test_session_clear_fixture_only_clears_local_state(self) -> None:
        execution = execute_hermes_parity_bundle(root_dir=ROOT_DIR)["session-clear-without-durable-forget"]

        self.assertEqual(execution["messages_before"], ("msg1", "msg2"))
        self.assertEqual(execution["messages_after"], ())
        self.assertTrue(execution["updated_at_advances"])
        self.assertFalse(execution["durable_forget_surface_present"])

    def test_openclaw_migration_fixture_preserves_hermes_wrappers_and_peers(self) -> None:
        execution = execute_hermes_parity_bundle(root_dir=ROOT_DIR)["openclaw-migration-artifacts"]

        self.assertEqual(execution["history_upload_name"], "prior_history.txt")
        self.assertEqual(
            execution["memory_upload_names"],
            (
                "consolidated_memory.md",
                "user_profile.md",
                "agent_soul.md",
            ),
        )
        self.assertEqual(execution["history_metadata_source"], "local_jsonl")
        self.assertEqual(execution["memory_metadata_source"], "local_memory")
        self.assertEqual(execution["target_peers"], ("user", "user", "ai"))
        self.assertTrue(execution["history_entry"].startswith("<prior_conversation_history>"))
        self.assertTrue(all(entry.startswith("<prior_memory_file>") for entry in execution["memory_entries"]))


if __name__ == "__main__":
    unittest.main()
