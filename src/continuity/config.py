"""Hermes-compatible config parsing and session naming rules for Continuity."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_HOST = "hermes"
DEFAULT_CONFIG_PATH = Path.home() / ".honcho" / "config.json"
RECALL_MODE_ALIASES = {"auto": "hybrid"}
VALID_RECALL_MODES = {"hybrid", "context", "tools"}


def normalize_recall_mode(value: object) -> str:
    """Return the normalized Hermes-compatible recall mode."""
    normalized = str(value or "").strip().lower()
    normalized = RECALL_MODE_ALIASES.get(normalized, normalized)
    if normalized in VALID_RECALL_MODES:
        return normalized
    return "hybrid"


def _coerce_write_frequency(value: object) -> str | int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value

    text = str(value or "async").strip()
    try:
        return int(text)
    except ValueError:
        return text or "async"


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    return {}


def _resolve_memory_mode(global_value: object, host_value: object | None) -> tuple[str, dict[str, str]]:
    selected = host_value if host_value is not None else global_value

    if isinstance(selected, Mapping):
        default = str(selected.get("default") or "hybrid")
        overrides = {
            str(name): str(mode)
            for name, mode in selected.items()
            if name != "default"
        }
        return default, overrides

    default = str(selected or "hybrid")
    return default, {}


@dataclass(frozen=True)
class ContinuityConfig:
    """Resolved host-facing config surface for the Continuity Hermes boundary."""

    host: str = DEFAULT_HOST
    workspace_id: str = DEFAULT_HOST
    ai_peer: str = DEFAULT_HOST
    peer_name: str | None = None
    memory_mode: str = "hybrid"
    peer_memory_modes: dict[str, str] = field(default_factory=dict)
    recall_mode: str = "hybrid"
    write_frequency: str | int = "async"
    session_strategy: str = "per-session"
    session_peer_prefix: bool = False
    sessions: dict[str, str] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        raw_config: Mapping[str, object],
        *,
        host: str = DEFAULT_HOST,
    ) -> ContinuityConfig:
        raw = dict(raw_config)
        hosts = _mapping(raw.get("hosts"))
        host_block = _mapping(hosts.get(host))

        workspace = str(host_block.get("workspace") or raw.get("workspace") or host)
        ai_peer = str(host_block.get("aiPeer") or raw.get("aiPeer") or host)
        peer_name = host_block.get("peerName") or raw.get("peerName")

        memory_mode, peer_memory_modes = _resolve_memory_mode(
            raw.get("memoryMode", "hybrid"),
            host_block.get("memoryMode"),
        )

        recall_mode = normalize_recall_mode(
            host_block.get("recallMode") or raw.get("recallMode") or "hybrid"
        )
        write_frequency = _coerce_write_frequency(
            host_block.get("writeFrequency") or raw.get("writeFrequency") or "async"
        )
        session_strategy = str(
            host_block.get("sessionStrategy") or raw.get("sessionStrategy") or "per-session"
        )

        host_prefix = host_block.get("sessionPeerPrefix")
        session_peer_prefix = (
            bool(host_prefix)
            if host_prefix is not None
            else bool(raw.get("sessionPeerPrefix", False))
        )

        sessions = raw.get("sessions")

        return cls(
            host=host,
            workspace_id=workspace,
            ai_peer=ai_peer,
            peer_name=str(peer_name) if peer_name is not None else None,
            memory_mode=memory_mode,
            peer_memory_modes=peer_memory_modes,
            recall_mode=recall_mode,
            write_frequency=write_frequency,
            session_strategy=session_strategy,
            session_peer_prefix=session_peer_prefix,
            sessions=dict(sessions) if isinstance(sessions, dict) else {},
            raw=raw,
        )

    @classmethod
    def from_global_config(
        cls,
        *,
        host: str = DEFAULT_HOST,
        config_path: Path | None = None,
    ) -> ContinuityConfig:
        path = config_path or DEFAULT_CONFIG_PATH

        try:
            if not path.exists():
                return cls(host=host, workspace_id=host, ai_peer=host)

            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls(host=host, workspace_id=host, ai_peer=host)

        if not isinstance(raw, Mapping):
            return cls(host=host, workspace_id=host, ai_peer=host)

        return cls.from_mapping(raw, host=host)

    def peer_memory_mode(self, peer_name: str) -> str:
        return self.peer_memory_modes.get(peer_name, self.memory_mode)

    @staticmethod
    def _git_repo_name(cwd: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None

        if result.returncode == 0:
            return Path(result.stdout.strip()).name
        return None

    def resolve_session_name(
        self,
        cwd: str | None = None,
        *,
        session_title: str | None = None,
        session_id: str | None = None,
    ) -> str | None:
        base_cwd = cwd or os.getcwd()

        manual = self.sessions.get(base_cwd)
        if manual:
            return manual

        if session_title:
            sanitized = re.sub(r"[^a-zA-Z0-9_-]", "-", session_title).strip("-")
            if sanitized:
                return self._apply_peer_prefix(sanitized)

        if self.session_strategy == "per-session" and session_id:
            return self._apply_peer_prefix(session_id)

        if self.session_strategy == "per-repo":
            return self._apply_peer_prefix(self._git_repo_name(base_cwd) or Path(base_cwd).name)

        if self.session_strategy in {"per-directory", "per-session"}:
            return self._apply_peer_prefix(Path(base_cwd).name)

        return self.workspace_id

    def _apply_peer_prefix(self, value: str) -> str:
        if self.session_peer_prefix and self.peer_name:
            return f"{self.peer_name}-{value}"
        return value
