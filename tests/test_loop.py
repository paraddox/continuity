#!/usr/bin/env python3

import os
import signal
import subprocess
import tempfile
import textwrap
import time
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
LOOP_SCRIPT = ROOT_DIR / "loop.sh"


class LoopSignalHandlingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)
        self.fake_bin = self.tmpdir / "fake-bin"
        self.fake_bin.mkdir()
        self.log_file = self.tmpdir / "codex.log"
        self.out_file = self.tmpdir / "loop.out"
        self.proc: subprocess.Popen[str] | None = None

        fake_codex = self.fake_bin / "codex"
        fake_codex.write_text(
            textwrap.dedent(
                """\
                #!/bin/bash
                set -euo pipefail

                log_file="${FAKE_CODEX_LOG:?}"
                sleep_seconds="${FAKE_CODEX_SLEEP:-1}"

                echo "start $$ $*" >>"$log_file"
                trap 'echo "int $$" >>"$log_file"; exit 130' INT
                trap 'echo "term $$" >>"$log_file"; exit 143' TERM

                sleep "$sleep_seconds"
                echo "done $$" >>"$log_file"
                """
            ),
            encoding="utf-8",
        )
        fake_codex.chmod(0o755)

    def tearDown(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=5)
        self._tmpdir.cleanup()

    def _start_loop(self, *, sleep_seconds: str) -> subprocess.Popen[str]:
        env = os.environ.copy()
        env["PATH"] = f"{self.fake_bin}:{env['PATH']}"
        env["FAKE_CODEX_LOG"] = str(self.log_file)
        env["FAKE_CODEX_SLEEP"] = sleep_seconds

        out_handle = self.out_file.open("w", encoding="utf-8")
        self.addCleanup(out_handle.close)

        self.proc = subprocess.Popen(
            ["bash", str(LOOP_SCRIPT), "prompt"],
            cwd=ROOT_DIR,
            env=env,
            stdout=out_handle,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )
        self._wait_for(lambda: self._log_contains_prefix("start "))
        return self.proc

    def _wait_for(self, predicate, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.05)
        self.fail("timed out waiting for expected test condition")

    def _read_log_lines(self) -> list[str]:
        if not self.log_file.exists():
            return []
        return self.log_file.read_text(encoding="utf-8").splitlines()

    def _log_contains_prefix(self, prefix: str) -> bool:
        return any(line.startswith(prefix) for line in self._read_log_lines())

    def test_first_sigint_stops_after_current_run(self) -> None:
        proc = self._start_loop(sleep_seconds="1")

        os.killpg(proc.pid, signal.SIGINT)

        status = proc.wait(timeout=5)
        self.assertEqual(
            status,
            0,
            "first Ctrl-C should stop after the current run and exit cleanly",
        )

        lines = self._read_log_lines()
        start_count = sum(1 for line in lines if line.startswith("start "))
        self.assertEqual(
            start_count,
            1,
            "first Ctrl-C should prevent a second codex run from starting",
        )
        self.assertTrue(
            any(line.startswith("done ") for line in lines),
            "current codex run should finish after first Ctrl-C",
        )
        self.assertFalse(
            any(line.startswith(("int ", "term ")) for line in lines),
            "first Ctrl-C should not interrupt the active codex run",
        )

    def test_second_sigint_stops_immediately(self) -> None:
        proc = self._start_loop(sleep_seconds="30")

        os.killpg(proc.pid, signal.SIGINT)
        time.sleep(0.2)
        os.killpg(proc.pid, signal.SIGINT)

        status = proc.wait(timeout=5)
        self.assertNotEqual(
            status,
            0,
            "second Ctrl-C should exit non-zero for an immediate stop",
        )

        lines = self._read_log_lines()
        start_count = sum(1 for line in lines if line.startswith("start "))
        self.assertEqual(
            start_count,
            1,
            "second Ctrl-C should still stop before another codex run starts",
        )
        self.assertTrue(
            any(line.startswith(("int ", "term ")) for line in lines),
            "second Ctrl-C should interrupt the active codex run immediately",
        )


if __name__ == "__main__":
    unittest.main()
