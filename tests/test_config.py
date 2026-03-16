#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
CORPUS_PATH = ROOT_DIR / "tests" / "fixtures" / "hermes_corpus.json"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from continuity.config import ContinuityConfig, normalize_recall_mode


def load_fixture(fixture_id: str) -> dict[str, object]:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    for fixture in corpus["fixtures"]:
        if fixture["id"] == fixture_id:
            return fixture
    raise AssertionError(f"missing fixture: {fixture_id}")


class ConfigParsingTests(unittest.TestCase):
    def test_fixture_backed_host_resolution_cases(self) -> None:
        fixture = load_fixture("host-config-resolution-cases")
        cases = {
            case["name"]: case["config"]
            for case in fixture["normalized_input"]["config_cases"]
        }

        host_override = ContinuityConfig.from_mapping(cases["host overrides root workspace and aiPeer"])
        self.assertEqual(host_override.workspace_id, "host-ws")
        self.assertEqual(host_override.ai_peer, "host-ai")

        coerced_frequency = ContinuityConfig.from_mapping(cases["writeFrequency string coerces to integer"])
        self.assertEqual(coerced_frequency.write_frequency, 3)

        object_mode = ContinuityConfig.from_mapping(cases["memoryMode object provides default plus peer override"])
        self.assertEqual(object_mode.memory_mode, "hybrid")
        self.assertEqual(object_mode.peer_memory_mode("hermes"), "honcho")
        self.assertEqual(object_mode.peer_memory_mode("other"), "hybrid")

    def test_write_frequency_examples_round_trip(self) -> None:
        fixture = load_fixture("host-config-resolution-cases")
        examples = fixture["expected_behavior"]["write_frequency_examples"]

        for example in examples:
            with self.subTest(example=example):
                config = ContinuityConfig.from_mapping({"writeFrequency": example})
                self.assertEqual(config.write_frequency, example)

    def test_recall_mode_contract_is_closed_and_host_block_wins(self) -> None:
        fixture = load_fixture("recall-mode-contracts")
        accepted_modes = fixture["normalized_input"]["accepted_modes"]
        alias_map = fixture["normalized_input"]["legacy_aliases"]
        invalid_fallback = fixture["expected_behavior"]["invalid_values_fall_back_to"]

        for mode in accepted_modes:
            with self.subTest(mode=mode):
                self.assertEqual(normalize_recall_mode(mode), mode)

        for legacy_name, normalized in alias_map.items():
            with self.subTest(legacy_name=legacy_name):
                self.assertEqual(normalize_recall_mode(legacy_name), normalized)

        self.assertEqual(normalize_recall_mode("unexpected"), invalid_fallback)

        config = ContinuityConfig.from_mapping(
            {
                "recallMode": "tools",
                "hosts": {"hermes": {"recallMode": "context"}},
            }
        )
        self.assertEqual(config.recall_mode, "context")


class SessionResolutionTests(unittest.TestCase):
    def test_manual_override_beats_title_and_strategy(self) -> None:
        config = ContinuityConfig(sessions={"/repo": "manual-name"}, session_strategy="per-repo")

        self.assertEqual(
            config.resolve_session_name("/repo", session_title="Ignored Title", session_id="telegram:123456"),
            "manual-name",
        )

    def test_title_is_sanitized_and_optionally_prefixed(self) -> None:
        config = ContinuityConfig(peer_name="alice", session_peer_prefix=True)

        self.assertEqual(
            config.resolve_session_name("/repo", session_title="Sprint 42 / Memory?"),
            "alice-Sprint-42---Memory",
        )

    def test_per_session_strategy_prefers_host_session_id(self) -> None:
        config = ContinuityConfig(session_strategy="per-session", peer_name="alice", session_peer_prefix=True)

        self.assertEqual(
            config.resolve_session_name("/repo", session_id="telegram:123456"),
            "alice-telegram:123456",
        )

    def test_per_repo_strategy_uses_git_root_name(self) -> None:
        config = ContinuityConfig(session_strategy="per-repo", peer_name="eri", session_peer_prefix=True)

        with patch.object(ContinuityConfig, "_git_repo_name", return_value="continuity"):
            self.assertEqual(config.resolve_session_name("/repo/subdir"), "eri-continuity")

    def test_per_directory_falls_back_to_directory_name(self) -> None:
        config = ContinuityConfig()

        self.assertEqual(config.resolve_session_name("/home/user/my-project"), "my-project")

    def test_global_strategy_uses_workspace_name(self) -> None:
        config = ContinuityConfig(session_strategy="global", workspace_id="shared-hermes")

        self.assertEqual(config.resolve_session_name("/repo"), "shared-hermes")


if __name__ == "__main__":
    unittest.main()
