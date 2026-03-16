#!/usr/bin/env python3

import json
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
FIXTURES_DIR = ROOT_DIR / "tests" / "fixtures"
CORPUS_PATH = FIXTURES_DIR / "hermes_corpus.json"
DOC_PATH = ROOT_DIR / "docs" / "hermes-compatibility.md"
HERMES_ROOT = Path.home() / ".hermes" / "hermes-agent"


class HermesFixtureCorpusTests(unittest.TestCase):
    def test_harvested_corpus_covers_required_categories_with_provenance(self) -> None:
        self.assertTrue(CORPUS_PATH.exists(), f"missing harvested corpus: {CORPUS_PATH}")

        corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
        fixtures = corpus.get("fixtures")
        self.assertIsInstance(fixtures, list)
        self.assertGreater(len(fixtures), 0)

        ids = set()
        categories = set()
        families = set()

        for fixture in fixtures:
            fixture_id = fixture.get("id")
            self.assertIsInstance(fixture_id, str)
            self.assertNotIn(fixture_id, ids, f"duplicate fixture id: {fixture_id}")
            ids.add(fixture_id)

            category = fixture.get("category")
            self.assertIsInstance(category, str)
            categories.add(category)

            fixture_families = fixture.get("fixture_families")
            self.assertIsInstance(fixture_families, list)
            self.assertGreater(len(fixture_families), 0)
            families.update(fixture_families)

            self.assertIsInstance(fixture.get("title"), str)
            self.assertIsInstance(fixture.get("summary"), str)
            self.assertTrue(fixture["summary"].strip())
            self.assertIsInstance(fixture.get("normalized_input"), dict)
            self.assertIsInstance(fixture.get("expected_behavior"), dict)

            provenance = fixture.get("provenance")
            self.assertIsInstance(provenance, list)
            self.assertGreater(len(provenance), 0)
            for source in provenance:
                rel_path = source.get("path")
                self.assertIsInstance(rel_path, str)
                source_path = HERMES_ROOT / rel_path
                self.assertTrue(source_path.exists(), f"missing provenance source: {source_path}")

                line_start = source.get("line_start")
                line_end = source.get("line_end")
                self.assertIsInstance(line_start, int)
                self.assertIsInstance(line_end, int)
                self.assertLessEqual(line_start, line_end)

                with source_path.open("r", encoding="utf-8") as handle:
                    line_count = sum(1 for _ in handle)
                self.assertLessEqual(line_end, line_count, f"bad line range for {source_path}")

        self.assertEqual(
            categories,
            {
                "honcho_integration_behavior",
                "tool_descriptions",
                "recall_modes",
                "config_parsing_cases",
                "clearing_forgetting",
                "migration_examples",
            },
        )
        self.assertEqual(
            families,
            {"core_engine", "hermes_parity", "service_contract"},
        )

    def test_fixture_manifests_partition_the_harvested_corpus(self) -> None:
        corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
        fixtures = corpus["fixtures"]
        fixture_ids = {fixture["id"] for fixture in fixtures}

        seen = set()
        for family in ("core_engine", "hermes_parity", "service_contract"):
            manifest_path = FIXTURES_DIR / family / "manifest.json"
            self.assertTrue(manifest_path.exists(), f"missing fixture manifest: {manifest_path}")

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest.get("fixture_family"), family)
            manifest_ids = manifest.get("fixture_ids")
            self.assertIsInstance(manifest_ids, list)
            self.assertGreater(len(manifest_ids), 0)

            for fixture_id in manifest_ids:
                self.assertIn(fixture_id, fixture_ids, f"unknown fixture id in {family}: {fixture_id}")
                seen.add(fixture_id)

        self.assertEqual(seen, fixture_ids)

    def test_compatibility_doc_references_corpus_and_fixture_taxonomy(self) -> None:
        self.assertTrue(DOC_PATH.exists(), f"missing compatibility doc: {DOC_PATH}")

        text = DOC_PATH.read_text(encoding="utf-8")
        self.assertIn("Harvested Hermes source corpus", text)
        self.assertIn("tests/fixtures/hermes_corpus.json", text)
        self.assertIn("core_engine", text)
        self.assertIn("hermes_parity", text)
        self.assertIn("service_contract", text)
        self.assertIn("config parsing", text.lower())
        self.assertIn("recall mode", text.lower())
        self.assertIn("migration", text.lower())


if __name__ == "__main__":
    unittest.main()
