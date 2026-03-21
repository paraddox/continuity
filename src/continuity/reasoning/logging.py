"""Reasoning-event logging and 24h inspection helpers."""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from continuity.reasoning.base import (
    AnswerQueryRequest,
    ClaimDerivationRequest,
    RawStructuredOutput,
    ReasoningAdapter,
    SessionSummaryRequest,
    StructuredGenerationRequest,
    TextResponse,
)
from continuity.store.schema import apply_migrations


RETENTION_HOURS = 24


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _retention_cutoff(*, now: datetime | None = None) -> datetime:
    anchor = now or _now_utc()
    return anchor - timedelta(hours=RETENTION_HOURS)


def _clean_optional_text(value: object) -> str | None:
    cleaned = str(value or "").strip()
    return cleaned or None


@dataclass(frozen=True, slots=True)
class ReasoningRuntimeMetadata:
    adapter: str
    provider: str | None = None
    model: str | None = None
    target_name: str | None = None
    base_url: str | None = None


def open_reasoning_log_connection(store_path: Path) -> sqlite3.Connection:
    path = Path(store_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), check_same_thread=False)
    connection.row_factory = sqlite3.Row
    apply_migrations(connection)
    return connection


def prune_reasoning_events(
    connection: sqlite3.Connection,
    *,
    now: datetime | None = None,
) -> None:
    cutoff = _retention_cutoff(now=now).isoformat()
    connection.execute(
        "DELETE FROM reasoning_events WHERE recorded_at < ?",
        (cutoff,),
    )
    connection.commit()


def list_reasoning_events(
    connection: sqlite3.Connection | None,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if connection is None:
        return []
    prune_reasoning_events(connection)
    rows = connection.execute(
        """
        SELECT
            event_id,
            recorded_at,
            operation,
            adapter,
            provider,
            model,
            target_name,
            base_url,
            success,
            latency_ms,
            error_text
        FROM reasoning_events
        ORDER BY recorded_at DESC, event_id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [
        {
            "event_id": row["event_id"],
            "recorded_at": row["recorded_at"],
            "operation": row["operation"],
            "adapter": row["adapter"],
            "provider": row["provider"],
            "model": row["model"],
            "target_name": row["target_name"],
            "base_url": row["base_url"],
            "success": bool(row["success"]),
            "latency_ms": row["latency_ms"],
            "error_text": row["error_text"],
        }
        for row in rows
    ]


def latest_reasoning_event(connection: sqlite3.Connection | None) -> dict[str, Any] | None:
    events = list_reasoning_events(connection, limit=1)
    if not events:
        return None
    return events[0]


class LoggingReasoningAdapter:
    """Wrap a reasoning adapter and persist the last 24h of provider usage."""

    def __init__(
        self,
        *,
        delegate: ReasoningAdapter,
        store_path: Path,
        metadata: ReasoningRuntimeMetadata,
        connection_factory: Callable[[Path], sqlite3.Connection] = open_reasoning_log_connection,
    ) -> None:
        self.delegate = delegate
        self.metadata = metadata
        self._store_path = Path(store_path).expanduser()
        self._connection = connection_factory(self._store_path)
        self._lock = threading.Lock()

    def close(self) -> None:
        self._connection.close()

    def answer_query(self, request: AnswerQueryRequest) -> TextResponse:
        return self._record("answer_query", lambda: self.delegate.answer_query(request))

    def generate_structured(self, request: StructuredGenerationRequest) -> RawStructuredOutput:
        return self._record("generate_structured", lambda: self.delegate.generate_structured(request))

    def summarize_session(self, request: SessionSummaryRequest) -> TextResponse:
        return self._record("summarize_session", lambda: self.delegate.summarize_session(request))

    def derive_claims(self, request: ClaimDerivationRequest) -> RawStructuredOutput:
        return self._record("derive_claims", lambda: self.delegate.derive_claims(request))

    def _record(self, operation: str, invoke: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        recorded_at = _now_utc()
        success = False
        error_text: str | None = None
        try:
            result = invoke()
            success = True
            return result
        except Exception as exc:
            error_text = str(exc)
            raise
        finally:
            latency_ms = int((time.perf_counter() - started) * 1000)
            with self._lock:
                self._connection.execute(
                    """
                    INSERT INTO reasoning_events(
                        event_id,
                        recorded_at,
                        operation,
                        adapter,
                        provider,
                        model,
                        target_name,
                        base_url,
                        success,
                        latency_ms,
                        error_text
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"reasoning:{uuid.uuid4().hex}",
                        recorded_at.isoformat(),
                        operation,
                        self.metadata.adapter,
                        _clean_optional_text(self.metadata.provider),
                        _clean_optional_text(self.metadata.model),
                        _clean_optional_text(self.metadata.target_name),
                        _clean_optional_text(self.metadata.base_url),
                        1 if success else 0,
                        max(0, latency_ms),
                        _clean_optional_text(error_text),
                    ),
                )
                self._connection.commit()
                prune_reasoning_events(self._connection, now=recorded_at)
