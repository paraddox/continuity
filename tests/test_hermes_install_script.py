#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent


class HermesInstallScriptTests(unittest.TestCase):
    def test_script_exists_and_documents_default_target(self) -> None:
        script_path = ROOT_DIR / "scripts" / "install-hermes-plugin.sh"
        self.assertTrue(script_path.exists(), "install script should exist")

        script = script_path.read_text(encoding="utf-8")
        self.assertIn('DEFAULT_HERMES_DIR="$HOME/.hermes/hermes-agent"', script)
        self.assertIn("continuity.hermes_compat.plugin:create_backend", script)
        self.assertIn("SKIP_PIP_INSTALL", script)

    def test_script_updates_honcho_config_for_custom_install_dir(self) -> None:
        script_path = ROOT_DIR / "scripts" / "install-hermes-plugin.sh"

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            hermes_dir = temp_root / "custom-hermes"
            python_bin = hermes_dir / "venv" / "bin" / "python"
            python_bin.parent.mkdir(parents=True)
            python_bin.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            python_bin.chmod(
                python_bin.stat().st_mode
                | stat.S_IXUSR
                | stat.S_IXGRP
                | stat.S_IXOTH
            )

            config_path = temp_root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "apiKey": "local-dev",
                        "baseUrl": "http://127.0.0.1:43845",
                        "hosts": {
                            "hermes": {
                                "peerName": "soso",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            env = os.environ.copy()
            env.update(
                {
                    "HONCHO_CONFIG_PATH": str(config_path),
                    "SKIP_PIP_INSTALL": "1",
                    "RESTART_GATEWAY": "0",
                }
            )

            subprocess.run(
                [str(script_path), str(hermes_dir)],
                check=True,
                cwd=ROOT_DIR,
                env=env,
            )

            updated = json.loads(config_path.read_text(encoding="utf-8"))
            host = updated["hosts"]["hermes"]

            self.assertEqual(
                host["experimental"]["memory_backend_factory"],
                "continuity.hermes_compat.plugin:create_backend",
            )
            self.assertEqual(host["continuity"]["storePath"], str(Path.home() / ".hermes" / "continuity.db"))
            self.assertEqual(
                host["continuity"]["collectionPath"],
                str(Path.home() / ".hermes" / "continuity-zvec"),
            )
            self.assertEqual(host["continuity"]["vectorBackend"], "zvec")
            self.assertEqual(host["continuity"]["embeddingDimensions"], 768)
            self.assertEqual(host["continuity"]["reasoningModel"], "gpt-5.4")
            self.assertEqual(host["continuity"]["reasoningEffort"], "low")
            self.assertEqual(host["peerName"], "soso")


if __name__ == "__main__":
    unittest.main()
