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
from tests.fixtures.service_contract_harness import execute_service_contract_bundle
from tests.fixtures.shared import load_fixture_bundle


class ServiceContractFixtureHarnessTests(unittest.TestCase):
    def test_executes_service_contract_bundle_in_manifest_order(self) -> None:
        bundle = load_fixture_bundle("service_contract", root_dir=ROOT_DIR)
        executions = execute_service_contract_bundle(root_dir=ROOT_DIR)

        self.assertEqual(tuple(executions), tuple(fixture.id for fixture in bundle.fixtures))

    def test_honcho_tool_surface_maps_fixture_tools_to_canonical_service_operations(self) -> None:
        execution = execute_service_contract_bundle(root_dir=ROOT_DIR)["honcho-tool-surface"]

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
        self.assertEqual(execution["required_parameters"]["honcho_profile"], ())
        self.assertEqual(execution["required_parameters"]["honcho_search"], ("query",))
        self.assertEqual(
            execution["optional_parameters"]["honcho_context"],
            ("peer",),
        )
        self.assertEqual(execution["llm_backed"]["honcho_context"], True)
        self.assertEqual(execution["llm_backed"]["honcho_profile"], False)
        self.assertEqual(execution["structured_response_payloads"], True)

    def test_recall_mode_fixture_executes_against_config_surface(self) -> None:
        execution = execute_service_contract_bundle(root_dir=ROOT_DIR)["recall-mode-contracts"]

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
        self.assertEqual(execution["host_block_overrides_root_config"], True)
        self.assertEqual(execution["tools_mode_skips_startup_prefetch"], True)

    def test_host_config_fixture_validates_resolution_and_write_frequency_examples(self) -> None:
        execution = execute_service_contract_bundle(root_dir=ROOT_DIR)["host-config-resolution-cases"]

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
            execution["write_frequency_examples"],
            ("async", "turn", "session", 5),
        )

    def test_openclaw_migration_fixture_maps_to_import_history_request(self) -> None:
        execution = execute_service_contract_bundle(root_dir=ROOT_DIR)["openclaw-migration-artifacts"]

        self.assertEqual(execution["operation"], "import_history")
        self.assertEqual(execution["default_minimum_waterline"], DurabilityWaterline.VIEWS_COMPILED.value)
        self.assertEqual(execution["source_kind"], "openclaw")

        entries = execution["entries"]
        self.assertEqual(entries[0]["upload_name"], "prior_history.txt")
        self.assertEqual(entries[0]["metadata"]["source"], "local_jsonl")
        self.assertIn("<prior_conversation_history>", entries[0]["content"])
        self.assertEqual(entries[1]["upload_name"], "consolidated_memory.md")
        self.assertEqual(entries[1]["metadata"]["source"], "local_memory")
        self.assertIn("<prior_memory_file>", entries[1]["content"])
        self.assertEqual(entries[3]["metadata"]["target_peer"], "ai")


if __name__ == "__main__":
    unittest.main()
