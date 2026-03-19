#!/usr/bin/env python3

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from continuity.hermes_compat.config import HermesMemoryBackendKind


def _fake_hermes_memory_backend_module() -> tuple[ModuleType, ModuleType]:
    package = ModuleType("memory_backends")
    package.__path__ = []  # type: ignore[attr-defined]

    module = ModuleType("memory_backends.base")

    class MemoryBackendCapability(str, Enum):
        PROFILE = "profile"
        SEARCH = "search"
        ANSWER = "answer"
        CONCLUDE = "conclude"
        PREFETCH = "prefetch"
        MIGRATE = "migrate"
        AI_IDENTITY = "ai_identity"

    @dataclass(frozen=True)
    class MemoryBackendManifest:
        protocol_version: int = 1
        backend_id: str = ""
        display_name: str = ""
        capabilities: frozenset[str] = field(default_factory=frozenset)
        config_source: str = ""

    @dataclass(frozen=True)
    class MemoryBackendBundle:
        manager: object | None
        config: object
        manifest: MemoryBackendManifest

    module.PROTOCOL_VERSION = 1
    module.MemoryBackendCapability = MemoryBackendCapability
    module.MemoryBackendManifest = MemoryBackendManifest
    module.MemoryBackendBundle = MemoryBackendBundle
    package.base = module  # type: ignore[attr-defined]
    return package, module


class HermesPluginTests(unittest.TestCase):
    def test_plugin_module_import_is_safe_without_hermes_types(self) -> None:
        module = importlib.import_module("continuity.hermes_compat.plugin")
        self.assertTrue(callable(module.create_backend))

    def test_plugin_exports_create_backend_from_package_surface(self) -> None:
        package = importlib.import_module("continuity.hermes_compat")
        plugin = importlib.import_module("continuity.hermes_compat.plugin")
        self.assertIs(package.create_backend, plugin.create_backend)

    def test_create_backend_returns_hermes_bundle_and_forces_continuity_backend(self) -> None:
        package, base_module = _fake_hermes_memory_backend_module()
        fake_manager = object()

        captured: dict[str, object] = {}

        def _fake_create_continuity_backend(config=None, **_kwargs):
            captured["config"] = config
            return fake_manager, config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "hosts": {
                            "hermes": {
                                "peerName": "soso",
                                "continuity": {
                                    "storePath": str(Path(tmpdir) / "continuity.db"),
                                    "vectorBackend": "inmemory",
                                },
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch.dict(
                    sys.modules,
                    {
                        "memory_backends": package,
                        "memory_backends.base": base_module,
                    },
                ),
                patch(
                    "continuity.hermes_compat.plugin.create_continuity_backend",
                    side_effect=_fake_create_continuity_backend,
                ),
            ):
                plugin = importlib.import_module("continuity.hermes_compat.plugin")
                bundle = plugin.create_backend(host="hermes", config_path=str(config_path))

        config = captured["config"]
        self.assertEqual(config.backend, HermesMemoryBackendKind.CONTINUITY)
        self.assertTrue(config.enabled)
        self.assertIs(bundle.manager, fake_manager)
        self.assertIs(bundle.config, config)
        self.assertIsInstance(bundle, base_module.MemoryBackendBundle)
        self.assertIsInstance(bundle.manifest, base_module.MemoryBackendManifest)
        self.assertEqual(bundle.manifest.backend_id, "continuity")
        self.assertEqual(bundle.manifest.display_name, "Continuity")
        self.assertEqual(bundle.manifest.config_source, str(config_path))
        self.assertEqual(
            bundle.manifest.capabilities,
            frozenset(
                {
                    "profile",
                    "search",
                    "answer",
                    "conclude",
                    "prefetch",
                    "migrate",
                    "ai_identity",
                }
            ),
        )

    def test_create_backend_preserves_explicit_disabled_state(self) -> None:
        package, base_module = _fake_hermes_memory_backend_module()
        fake_manager = object()

        captured: dict[str, object] = {}

        def _fake_create_continuity_backend(config=None, **_kwargs):
            captured["config"] = config
            return None, config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "hosts": {
                            "hermes": {
                                "enabled": False,
                                "continuity": {
                                    "storePath": str(Path(tmpdir) / "continuity.db"),
                                    "vectorBackend": "inmemory",
                                },
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch.dict(
                    sys.modules,
                    {
                        "memory_backends": package,
                        "memory_backends.base": base_module,
                    },
                ),
                patch(
                    "continuity.hermes_compat.plugin.create_continuity_backend",
                    side_effect=_fake_create_continuity_backend,
                ),
            ):
                plugin = importlib.import_module("continuity.hermes_compat.plugin")
                bundle = plugin.create_backend(host="hermes", config_path=str(config_path))

        config = captured["config"]
        self.assertEqual(config.backend, HermesMemoryBackendKind.CONTINUITY)
        self.assertFalse(config.enabled)
        self.assertIsNone(bundle.manager)
        self.assertIs(bundle.config, config)

    def test_create_backend_fails_clearly_when_hermes_types_are_unavailable(self) -> None:
        plugin = importlib.import_module("continuity.hermes_compat.plugin")

        real_import_module = importlib.import_module

        def _raising_import(name: str, package: str | None = None):
            if name == "memory_backends.base":
                raise ImportError("missing Hermes backend types")
            return real_import_module(name, package)

        with patch("continuity.hermes_compat.plugin.importlib.import_module", side_effect=_raising_import):
            with self.assertRaisesRegex(RuntimeError, "Hermes memory backend contract types are unavailable"):
                plugin.create_backend()


if __name__ == "__main__":
    unittest.main()
