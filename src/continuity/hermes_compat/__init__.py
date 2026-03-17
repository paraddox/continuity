"""Hermes-facing compatibility layer for embedding Continuity behind Honcho tools."""

from continuity.hermes_compat.config import (
    ContinuityVectorBackendKind,
    HermesMemoryBackendKind,
    HermesMemoryConfig,
)
from continuity.hermes_compat.factory import create_continuity_backend
from continuity.hermes_compat.manager import (
    ContinuityHermesSession,
    ContinuityHermesSessionManager,
)

__all__ = [
    "ContinuityHermesSession",
    "ContinuityHermesSessionManager",
    "ContinuityVectorBackendKind",
    "HermesMemoryBackendKind",
    "HermesMemoryConfig",
    "create_continuity_backend",
]
