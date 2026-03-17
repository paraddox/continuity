#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tests.fixtures.shared import FixtureBundleError, load_fixture_bundle, load_fixture_bundles


class FixtureBundleUtilityTests(unittest.TestCase):
    def test_load_fixture_bundle_assembles_family_bundle_in_manifest_order(self) -> None:
        bundle = load_fixture_bundle("hermes_parity", root_dir=ROOT_DIR)
        manifest_path = ROOT_DIR / "tests" / "fixtures" / "hermes_parity" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(bundle.bundle_version, "2026-03-16")
        self.assertEqual(bundle.fixture_family, "hermes_parity")
        self.assertEqual(bundle.producer.corpus, "tests/fixtures/hermes_corpus.json")
        self.assertEqual(bundle.producer.manifest, "tests/fixtures/hermes_parity/manifest.json")
        self.assertEqual(bundle.parity_target, "internal_hermes_embedded_patch_v1")
        self.assertEqual(
            [fixture.id for fixture in bundle.fixtures],
            manifest["fixture_ids"],
        )
        self.assertEqual(
            bundle.fixture("honcho-tool-surface").category,
            "tool_descriptions",
        )

    def test_load_fixture_bundles_exposes_one_shared_schema_for_all_fixture_families(self) -> None:
        bundles = load_fixture_bundles(root_dir=ROOT_DIR)

        self.assertEqual(
            tuple(bundles),
            ("core_engine", "hermes_parity", "service_contract"),
        )
        self.assertEqual(
            {bundle.bundle_version for bundle in bundles.values()},
            {"2026-03-16"},
        )
        self.assertEqual(
            {bundle.producer.corpus for bundle in bundles.values()},
            {"tests/fixtures/hermes_corpus.json"},
        )

    def test_load_fixture_bundle_rejects_unknown_manifest_fixture_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            fixtures_dir = root_dir / "tests" / "fixtures" / "core_engine"
            fixtures_dir.mkdir(parents=True)
            corpus_path = root_dir / "tests" / "fixtures" / "hermes_corpus.json"
            corpus_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-03-16",
                        "fixtures": [],
                    }
                ),
                encoding="utf-8",
            )
            (fixtures_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "fixture_family": "core_engine",
                        "fixture_ids": ["missing-fixture"],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FixtureBundleError, "missing-fixture"):
                load_fixture_bundle("core_engine", root_dir=root_dir)

    def test_load_fixture_bundle_rejects_fixture_family_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            fixtures_root = root_dir / "tests" / "fixtures"
            fixtures_root.mkdir(parents=True)
            (fixtures_root / "service_contract").mkdir()
            (fixtures_root / "hermes_parity").mkdir()
            (fixtures_root / "core_engine").mkdir()

            (fixtures_root / "hermes_corpus.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-03-16",
                        "fixtures": [
                            {
                                "id": "fixture-1",
                                "category": "tool_descriptions",
                                "title": "Fixture title",
                                "summary": "Fixture summary",
                                "fixture_families": ["hermes_parity"],
                                "normalized_input": {},
                                "expected_behavior": {},
                                "provenance": [
                                    {
                                        "path": "tools/honcho_tools.py",
                                        "line_start": 1,
                                        "line_end": 1,
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (fixtures_root / "service_contract" / "manifest.json").write_text(
                json.dumps(
                    {
                        "fixture_family": "service_contract",
                        "fixture_ids": ["fixture-1"],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FixtureBundleError, "service_contract"):
                load_fixture_bundle("service_contract", root_dir=root_dir)


if __name__ == "__main__":
    unittest.main()
