#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent


class ImplementPromptTests(unittest.TestCase):
    def test_prompt_handles_completed_implementation_backlog(self) -> None:
        prompt = (ROOT_DIR / "implement-prompt.md").read_text(encoding="utf-8")

        self.assertIn("bd status --json", prompt)
        self.assertIn("If no ready implementation tasks exist", prompt)
        self.assertIn("Do not invent new implementation work", prompt)
        self.assertIn("if there is no ready work at all, stop and report the complete backlog state", prompt)


if __name__ == "__main__":
    unittest.main()
