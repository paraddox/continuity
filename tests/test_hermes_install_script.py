#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml


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

    def test_script_writes_reasoning_target_fields_when_requested(self) -> None:
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
            config_path.write_text(json.dumps({"hosts": {"hermes": {}}}), encoding="utf-8")

            env = os.environ.copy()
            env.update(
                {
                    "HONCHO_CONFIG_PATH": str(config_path),
                    "SKIP_PIP_INSTALL": "1",
                    "RESTART_GATEWAY": "0",
                    "CONTINUITY_REASONING_TARGET_NAME": "GLM 5 Turbo",
                    "CONTINUITY_REASONING_PROVIDER": "zai",
                    "CONTINUITY_REASONING_TARGET_MODEL": "glm-5-turbo",
                    "CONTINUITY_REASONING_TARGET_EFFORT": "low",
                }
            )

            subprocess.run(
                [str(script_path), str(hermes_dir)],
                check=True,
                cwd=ROOT_DIR,
                env=env,
            )

            updated = json.loads(config_path.read_text(encoding="utf-8"))
            target = updated["hosts"]["hermes"]["continuity"]["reasoningTarget"]
            self.assertEqual(
                target,
                {
                    "targetName": "GLM 5 Turbo",
                    "provider": "zai",
                    "model": "glm-5-turbo",
                    "reasoningEffort": "low",
                },
            )

    def test_script_can_select_active_hermes_model_as_continuity_target(self) -> None:
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

            honcho_config_path = temp_root / "honcho-config.json"
            honcho_config_path.write_text(json.dumps({"hosts": {"hermes": {}}}), encoding="utf-8")

            hermes_config_path = temp_root / "hermes-config.yaml"
            hermes_config_path.write_text(
                yaml.safe_dump(
                    {
                        "model": {
                            "provider": "openai-codex",
                            "default": "gpt-5.4-mini",
                            "base_url": "https://chatgpt.com/backend-api/codex",
                            "reasoning_effort": "medium",
                        }
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            env = os.environ.copy()
            env.update(
                {
                    "HONCHO_CONFIG_PATH": str(honcho_config_path),
                    "HERMES_CONFIG_PATH": str(hermes_config_path),
                    "SKIP_PIP_INSTALL": "1",
                    "RESTART_GATEWAY": "0",
                    "CONTINUITY_REASONING_SELECTION": "active-model",
                }
            )

            subprocess.run(
                [str(script_path), str(hermes_dir)],
                check=True,
                cwd=ROOT_DIR,
                env=env,
            )

            updated = json.loads(honcho_config_path.read_text(encoding="utf-8"))
            target = updated["hosts"]["hermes"]["continuity"]["reasoningTarget"]
            self.assertEqual(
                target,
                {
                    "provider": "openai-codex",
                    "model": "gpt-5.4-mini",
                    "reasoningEffort": "medium",
                },
            )

    def test_script_can_create_new_hermes_custom_provider_for_continuity(self) -> None:
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

            honcho_config_path = temp_root / "honcho-config.json"
            honcho_config_path.write_text(json.dumps({"hosts": {"hermes": {}}}), encoding="utf-8")

            hermes_config_path = temp_root / "hermes-config.yaml"
            hermes_config_path.write_text(
                yaml.safe_dump({"model": {"provider": "openai-codex", "default": "gpt-5.4"}}, sort_keys=False),
                encoding="utf-8",
            )

            env = os.environ.copy()
            env.update(
                {
                    "HONCHO_CONFIG_PATH": str(honcho_config_path),
                    "HERMES_CONFIG_PATH": str(hermes_config_path),
                    "SKIP_PIP_INSTALL": "1",
                    "RESTART_GATEWAY": "0",
                    "CONTINUITY_REASONING_SELECTION": "create-custom-provider",
                    "CONTINUITY_CREATE_PROVIDER_NAME": "GLM 5 Turbo",
                    "CONTINUITY_CREATE_PROVIDER_BASE_URL": "https://api.z.ai/api/coding/paas/v4",
                    "CONTINUITY_CREATE_PROVIDER_API_KEY": "glm-secret",
                    "CONTINUITY_CREATE_PROVIDER_MODEL": "glm-5-turbo",
                }
            )

            subprocess.run(
                [str(script_path), str(hermes_dir)],
                check=True,
                cwd=ROOT_DIR,
                env=env,
            )

            updated_honcho = json.loads(honcho_config_path.read_text(encoding="utf-8"))
            target = updated_honcho["hosts"]["hermes"]["continuity"]["reasoningTarget"]
            self.assertEqual(target, {"targetName": "GLM 5 Turbo"})

            updated_hermes = yaml.safe_load(hermes_config_path.read_text(encoding="utf-8"))
            self.assertEqual(
                updated_hermes["custom_providers"],
                [
                    {
                        "name": "GLM 5 Turbo",
                        "base_url": "https://api.z.ai/api/coding/paas/v4",
                        "api_key": "glm-secret",
                        "model": "glm-5-turbo",
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
