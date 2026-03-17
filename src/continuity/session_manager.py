"""Session bootstrap, local cache, and write-frequency coordination for Continuity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from continuity.admission import AdmissionDecisionTrace
from continuity.config import ContinuityConfig
from continuity.store.claims import AdmissionOutcome, CandidateMemory, Subject, SubjectKind
from continuity.store.sqlite import (
    SQLiteRepository,
    SessionBufferRecord,
    SessionMessageRecord,
    SessionRecord,
)
from continuity.transactions import (
    DurabilityWaterline,
    TransactionExecution,
    TransactionKind,
    TransactionRunner,
    WriteFrequencySchedule,
    WriteFrequencyPolicy,
    write_frequency_policy_for,
)


DISABLED_MEMORY_MODES = frozenset({"off", "disabled", "none", "false"})


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


class SessionBufferKind(StrEnum):
    PROMPT_ONLY = "prompt_only"
    SESSION_EPHEMERAL = "session_ephemeral"
    WRITE_FREQUENCY_STATE = "write_frequency_state"


@dataclass(frozen=True, slots=True)
class SessionSaveResult:
    session: SessionRecord
    message: SessionMessageRecord
    policy: WriteFrequencyPolicy
    requested_waterline: DurabilityWaterline
    execution: TransactionExecution
    pending_turns: int


@dataclass(frozen=True, slots=True)
class SessionFlushResult:
    session: SessionRecord
    policy: WriteFrequencyPolicy
    requested_waterline: DurabilityWaterline
    execution: TransactionExecution
    pending_turns: int


class SessionManager:
    """Own one in-process session/cache coordinator per embedded Continuity store."""

    def __init__(
        self,
        repository: SQLiteRepository,
        *,
        config: ContinuityConfig,
        runner: TransactionRunner | None = None,
    ) -> None:
        self._repository = repository
        self._config = config
        self._runner = runner or TransactionRunner()

    def memory_mode_for_peer(self, peer_name: str | None = None) -> str:
        candidate = peer_name or self._config.peer_name or self._config.ai_peer
        return self._config.peer_memory_mode(candidate)

    def memory_enabled_for_peer(self, peer_name: str | None = None) -> bool:
        return self.memory_mode_for_peer(peer_name).strip().lower() not in DISABLED_MEMORY_MODES

    def ensure_session(
        self,
        *,
        session_id: str,
        cwd: str | None = None,
        session_title: str | None = None,
        created_at: datetime,
        metadata: dict[str, Any] | None = None,
    ) -> SessionRecord:
        resolved_id = _clean_text(session_id, field_name="session_id")
        session_name = self._config.resolve_session_name(
            cwd,
            session_title=session_title,
            session_id=resolved_id,
        ) or resolved_id
        session = SessionRecord(
            session_id=resolved_id,
            host_namespace=self._config.host,
            session_name=session_name,
            recall_mode=self._config.recall_mode,
            write_frequency=str(self._config.write_frequency),
            created_at=created_at,
            metadata={} if metadata is None else dict(metadata),
        )
        self._repository.save_session(session)
        return self._repository.read_session(resolved_id) or session

    def save_turn(
        self,
        *,
        session_id: str,
        message_id: str,
        role: str,
        author_subject_id: str,
        content: str,
        observed_at: datetime,
        cwd: str | None = None,
        session_title: str | None = None,
        peer_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        write_frequency: str | int | None = None,
    ) -> SessionSaveResult:
        if not self.memory_enabled_for_peer(peer_name):
            raise ValueError(f"memory mode is disabled for peer {peer_name or self._config.ai_peer}")

        session = self.ensure_session(
            session_id=session_id,
            cwd=cwd,
            session_title=session_title,
            created_at=observed_at,
        )
        self._ensure_subject(author_subject_id=author_subject_id, role=role, created_at=observed_at)
        message = SessionMessageRecord(
            message_id=message_id,
            session_id=session.session_id,
            role=role,
            author_subject_id=author_subject_id,
            content=content,
            observed_at=observed_at,
            metadata={} if metadata is None else dict(metadata),
        )
        self._repository.save_message(message)

        policy = write_frequency_policy_for(write_frequency or session.write_frequency)
        requested_waterline, pending_turns = self._advance_write_frequency_state(
            session.session_id,
            policy,
            updated_at=observed_at,
        )
        execution = self._runner.run(
            TransactionKind.INGEST_TURN,
            payload={
                "session_id": session.session_id,
                "message_id": message.message_id,
                "role": message.role,
                "author_subject_id": message.author_subject_id,
            },
            requested_waterline=requested_waterline,
        )
        return SessionSaveResult(
            session=session,
            message=message,
            policy=policy,
            requested_waterline=requested_waterline,
            execution=execution,
            pending_turns=pending_turns,
        )

    def flush_session(self, session_id: str) -> SessionFlushResult:
        session = self._repository.read_session(session_id)
        if session is None:
            raise ValueError(f"unknown session: {session_id}")

        policy = write_frequency_policy_for(session.write_frequency)
        requested_waterline = DurabilityWaterline.SNAPSHOT_PUBLISHED
        execution = self._runner.run(
            TransactionKind.INGEST_TURN,
            payload={"session_id": session.session_id, "flush_reason": "session_end"},
            requested_waterline=requested_waterline,
        )
        self._write_pending_turns(
            session.session_id,
            pending_turns=0,
            updated_at=session.created_at,
        )
        return SessionFlushResult(
            session=session,
            policy=policy,
            requested_waterline=requested_waterline,
            execution=execution,
            pending_turns=0,
        )

    def record_non_durable_memory(
        self,
        *,
        session_id: str,
        candidate: CandidateMemory,
        trace: AdmissionDecisionTrace,
        updated_at: datetime,
    ) -> SessionBufferRecord:
        if self._repository.read_session(session_id) is None:
            self.ensure_session(session_id=session_id, created_at=updated_at)
        buffer_kind = self._buffer_kind_for_outcome(trace.decision.outcome)
        buffer = SessionBufferRecord(
            buffer_key=f"{buffer_kind.value}:{session_id}:{candidate.candidate_id}",
            session_id=session_id,
            buffer_kind=buffer_kind.value,
            payload={
                "candidate_id": candidate.candidate_id,
                "claim_type": candidate.claim_type,
                "subject_id": candidate.subject_id,
                "scope": candidate.scope.value,
                "value": candidate.value,
                "source_observation_ids": candidate.source_observation_ids,
                "outcome": trace.decision.outcome.value,
                "policy_stamp": trace.policy_stamp,
                "rationale": trace.decision.rationale,
            },
            updated_at=updated_at,
        )
        self._repository.save_session_buffer(buffer)
        return buffer

    def list_buffers(
        self,
        session_id: str,
        *,
        buffer_kind: SessionBufferKind | None = None,
    ) -> tuple[SessionBufferRecord, ...]:
        return self._repository.list_session_buffers(
            session_id=session_id,
            buffer_kind=None if buffer_kind is None else buffer_kind.value,
        )

    def _advance_write_frequency_state(
        self,
        session_id: str,
        policy: WriteFrequencyPolicy,
        *,
        updated_at: datetime,
    ) -> tuple[DurabilityWaterline, int]:
        if policy.schedule is WriteFrequencySchedule.PER_TURN:
            self._write_pending_turns(session_id, pending_turns=0, updated_at=updated_at)
            return policy.awaited_waterline, 0

        pending_turns = self._pending_turns(session_id) + 1

        if policy.schedule is WriteFrequencySchedule.SESSION_END:
            self._write_pending_turns(session_id, pending_turns=pending_turns, updated_at=updated_at)
            return DurabilityWaterline.OBSERVATION_COMMITTED, pending_turns

        assert policy.batch_size is not None
        if pending_turns >= policy.batch_size:
            self._write_pending_turns(session_id, pending_turns=0, updated_at=updated_at)
            return DurabilityWaterline.SNAPSHOT_PUBLISHED, 0

        self._write_pending_turns(session_id, pending_turns=pending_turns, updated_at=updated_at)
        return DurabilityWaterline.OBSERVATION_COMMITTED, pending_turns

    def _pending_turns(self, session_id: str) -> int:
        record = self._repository.read_session_buffer(self._write_frequency_buffer_key(session_id))
        if record is None:
            return 0
        pending_turns = record.payload.get("pending_turns", 0)
        if not isinstance(pending_turns, int) or pending_turns < 0:
            raise ValueError("write_frequency_state pending_turns must be a non-negative integer")
        return pending_turns

    def _write_pending_turns(
        self,
        session_id: str,
        *,
        pending_turns: int,
        updated_at: datetime,
    ) -> None:
        if pending_turns == 0:
            self._repository.delete_session_buffer(self._write_frequency_buffer_key(session_id))
            return

        self._repository.save_session_buffer(
            SessionBufferRecord(
                buffer_key=self._write_frequency_buffer_key(session_id),
                session_id=session_id,
                buffer_kind=SessionBufferKind.WRITE_FREQUENCY_STATE.value,
                payload={"pending_turns": pending_turns},
                updated_at=updated_at,
            )
        )

    def _ensure_subject(
        self,
        *,
        author_subject_id: str,
        role: str,
        created_at: datetime,
    ) -> None:
        if self._repository.read_subject(author_subject_id) is not None:
            return

        kind = {
            "user": SubjectKind.USER,
            "assistant": SubjectKind.ASSISTANT,
        }.get(role, SubjectKind.PEER)
        canonical_name = author_subject_id.rsplit(":", 1)[-1]
        self._repository.save_subject(
            Subject(
                subject_id=author_subject_id,
                kind=kind,
                canonical_name=canonical_name,
            ),
            created_at=created_at,
        )

    def _buffer_kind_for_outcome(self, outcome: AdmissionOutcome) -> SessionBufferKind:
        if outcome is AdmissionOutcome.PROMPT_ONLY:
            return SessionBufferKind.PROMPT_ONLY
        if outcome is AdmissionOutcome.SESSION_EPHEMERAL:
            return SessionBufferKind.SESSION_EPHEMERAL
        raise ValueError(f"{outcome.value} does not map to a non-durable session buffer")

    @staticmethod
    def _write_frequency_buffer_key(session_id: str) -> str:
        return f"{SessionBufferKind.WRITE_FREQUENCY_STATE.value}:{session_id}"
