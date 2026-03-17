#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import re
import unittest
from datetime import date, timedelta
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
PLAN_PATH = ROOT_DIR / "continuity-plan.md"
MANIFEST_PATH = ROOT_DIR / "docs" / "plan-coverage.json"

TASK_PATTERN = re.compile(r"^### Task (\d+\.\d+): (.+)$", re.MULTILINE)
LAYOUT_SECTION_PATTERN = re.compile(
    r"^## Proposed Layout\n\n(?P<body>.*?)(?=^## )",
    re.MULTILINE | re.DOTALL,
)
LAYOUT_ENTRY_PATTERN = re.compile(r"^- `([^`]+)`$", re.MULTILINE)


class PlanCoverageManifestTests(unittest.TestCase):
    def test_manifest_exists_and_maps_every_plan_task(self) -> None:
        manifest = self._load_manifest()
        plan_tasks = self._plan_tasks()

        self.assertEqual(
            [(entry["id"], entry["title"]) for entry in manifest["tasks"]],
            plan_tasks,
        )

    def test_manifest_points_only_to_existing_files(self) -> None:
        manifest = self._load_manifest()

        for entry in manifest["tasks"]:
            with self.subTest(task_id=entry["id"]):
                self.assertIn(
                    entry["status"],
                    {"covered", "covered_with_manual_review", "covered_with_live_validation"},
                )
                self.assertTrue(entry["impl_refs"], "each task must point at implementation artifacts")
                self.assertTrue(
                    entry["proof_refs"] or entry["manual_proof"] or entry["live_proof"],
                    "each task must provide an automated, manual, or live proof path",
                )

                for path_text in (*entry["impl_refs"], *entry["proof_refs"]):
                    path = ROOT_DIR / path_text
                    self.assertTrue(path.exists(), f"{entry['id']} references missing file: {path_text}")

    def test_proposed_layout_gaps_are_explicit(self) -> None:
        manifest = self._load_manifest()
        explicit_missing = {
            item["path"]: item["reason"]
            for item in manifest["explicit_missing_paths"]
        }

        for path_text in self._proposed_layout_paths():
            path = ROOT_DIR / path_text
            if path.exists():
                continue
            self.assertIn(
                path_text,
                explicit_missing,
                f"missing proposed-layout path must be declared explicitly: {path_text}",
            )
            self.assertTrue(explicit_missing[path_text].strip())

    def test_manual_and_live_proofs_include_freshness_metadata(self) -> None:
        manifest = self._load_manifest()
        as_of = self._as_of_date()

        for entry in manifest["tasks"]:
            with self.subTest(task_id=entry["id"]):
                if entry["status"] == "covered_with_manual_review":
                    self.assertTrue(entry["manual_proof"], "manual-review tasks must declare manual proof records")
                if entry["status"] == "covered_with_live_validation":
                    self.assertTrue(entry["live_proof"], "live-validation tasks must declare live proof records")

                for proof in entry["manual_proof"]:
                    self._assert_manual_proof_is_fresh(entry_id=entry["id"], proof=proof, as_of=as_of)
                for proof in entry["live_proof"]:
                    self._assert_live_proof_is_fresh(entry_id=entry["id"], proof=proof, as_of=as_of)

    def _load_manifest(self) -> dict[str, object]:
        self.assertTrue(
            MANIFEST_PATH.exists(),
            "docs/plan-coverage.json must exist to validate plan coverage",
        )
        return json.loads(MANIFEST_PATH.read_text())

    def _plan_tasks(self) -> list[tuple[str, str]]:
        plan_text = PLAN_PATH.read_text()
        return [(match.group(1), match.group(2)) for match in TASK_PATTERN.finditer(plan_text)]

    def _proposed_layout_paths(self) -> list[str]:
        match = LAYOUT_SECTION_PATTERN.search(PLAN_PATH.read_text())
        self.assertIsNotNone(match, "continuity-plan.md should contain a Proposed Layout section")
        body = match.group("body")
        return [layout_match.group(1) for layout_match in LAYOUT_ENTRY_PATTERN.finditer(body)]

    def _as_of_date(self) -> date:
        raw = os.environ.get("CONTINUITY_PLAN_COVERAGE_AS_OF")
        if raw:
            return date.fromisoformat(raw)
        return date.today()

    def _assert_manual_proof_is_fresh(self, *, entry_id: str, proof: object, as_of: date) -> None:
        self.assertIsInstance(proof, dict, f"{entry_id} manual proof entries must be structured objects")
        assert isinstance(proof, dict)
        self.assertTrue(str(proof.get("description", "")).strip(), f"{entry_id} manual proof requires description")
        self.assertIn("last_reviewed_on", proof, f"{entry_id} manual proof requires last_reviewed_on")
        self.assertIn("refresh_within_days", proof, f"{entry_id} manual proof requires refresh_within_days")
        reviewed_on = date.fromisoformat(str(proof["last_reviewed_on"]))
        refresh_within_days = int(proof["refresh_within_days"])
        self.assertGreater(refresh_within_days, 0, f"{entry_id} manual proof refresh window must be positive")
        self.assertLessEqual(
            as_of,
            reviewed_on + timedelta(days=refresh_within_days),
            f"{entry_id} manual proof is stale as of {as_of.isoformat()}",
        )

    def _assert_live_proof_is_fresh(self, *, entry_id: str, proof: object, as_of: date) -> None:
        self.assertIsInstance(proof, dict, f"{entry_id} live proof entries must be structured objects")
        assert isinstance(proof, dict)
        self.assertTrue(str(proof.get("description", "")).strip(), f"{entry_id} live proof requires description")
        self.assertTrue(
            str(proof.get("verification_command", "")).strip(),
            f"{entry_id} live proof requires verification_command",
        )
        self.assertIn("last_verified_on", proof, f"{entry_id} live proof requires last_verified_on")
        self.assertIn("refresh_within_days", proof, f"{entry_id} live proof requires refresh_within_days")
        verified_on = date.fromisoformat(str(proof["last_verified_on"]))
        refresh_within_days = int(proof["refresh_within_days"])
        self.assertGreater(refresh_within_days, 0, f"{entry_id} live proof refresh window must be positive")
        self.assertLessEqual(
            as_of,
            verified_on + timedelta(days=refresh_within_days),
            f"{entry_id} live proof is stale as of {as_of.isoformat()}",
        )


if __name__ == "__main__":
    unittest.main()
