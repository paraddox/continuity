"""External Hermes memory backend entrypoint for Continuity."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from continuity.config import DEFAULT_CONFIG_PATH
from continuity.hermes_compat.config import HermesMemoryBackendKind, HermesMemoryConfig
from continuity.hermes_compat.factory import create_continuity_backend


_CONTINUITY_CAPABILITIES = frozenset(
    {
        "profile",
        "search",
        "answer",
        "conclude",
        "prefetch",
        "migrate",
        "ai_identity",
    }
)


def _load_hermes_backend_types() -> Any:
    try:
        module = importlib.import_module("memory_backends.base")
    except Exception as exc:
        raise RuntimeError(
            "Hermes memory backend contract types are unavailable. "
            "Load this plugin from a Hermes runtime."
        ) from exc

    missing = [
        name
        for name in ("MemoryBackendBundle", "MemoryBackendManifest")
        if not hasattr(module, name)
    ]
    if missing:
        raise RuntimeError(
            "Hermes memory backend contract types are unavailable. "
            f"Missing: {', '.join(missing)}."
        )
    return module


def _resolved_config_path(config_path: str | Path | None) -> Path:
    if config_path is None:
        return DEFAULT_CONFIG_PATH
    return Path(config_path).expanduser()


def _force_continuity_backend(raw: Mapping[str, object], *, host: str) -> dict[str, object]:
    result = dict(raw)
    raw_hosts = raw.get("hosts")
    hosts = dict(raw_hosts) if isinstance(raw_hosts, Mapping) else {}
    raw_host_block = hosts.get(host)
    host_block = dict(raw_host_block) if isinstance(raw_host_block, Mapping) else {}
    host_block["backend"] = HermesMemoryBackendKind.CONTINUITY.value
    hosts[host] = host_block
    result["hosts"] = hosts
    return result


def _env_plugin_config(*, host: str) -> HermesMemoryConfig:
    config = HermesMemoryConfig.from_env(host=host)
    return replace(
        config,
        backend=HermesMemoryBackendKind.CONTINUITY,
        enabled=True,
    )


def _load_plugin_config(*, host: str, config_path: Path) -> HermesMemoryConfig:
    if not config_path.exists():
        return _env_plugin_config(host=host)

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _env_plugin_config(host=host)

    if not isinstance(raw, Mapping):
        return _env_plugin_config(host=host)

    return HermesMemoryConfig.from_mapping(
        _force_continuity_backend(raw, host=host),
        host=host,
    )


def create_backend(
    *,
    host: str = "hermes",
    config_path: str | Path | None = None,
):
    """Return a Hermes external-backend bundle backed by Continuity."""

    backend_types = _load_hermes_backend_types()
    resolved_config_path = _resolved_config_path(config_path)
    resolved_config = _load_plugin_config(host=host, config_path=resolved_config_path)
    manager, effective_config = create_continuity_backend(config=resolved_config)

    manifest = backend_types.MemoryBackendManifest(
        protocol_version=getattr(backend_types, "PROTOCOL_VERSION", 1),
        backend_id="continuity",
        display_name="Continuity",
        capabilities=_CONTINUITY_CAPABILITIES,
        config_source=str(resolved_config_path),
    )
    return backend_types.MemoryBackendBundle(
        manager=manager,
        config=effective_config,
        manifest=manifest,
    )
