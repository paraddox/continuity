"""Hermes-facing session manager that embeds Continuity behind Honcho tools."""

from __future__ import annotations

import hashlib
import logging
import queue
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from continuity.context_builder import ContinuityContextBuilder
from continuity.disclosure import (
    DisclosureChannel,
    DisclosureContext,
    disclosure_policy_for,
    DisclosurePrincipal,
    DisclosurePurpose,
    DisclosureViewer,
    ViewerKind,
)
from continuity.hermes_compat.config import HermesMemoryConfig
from continuity.index.zvec_index import EmbeddingClientProtocol, VectorBackendProtocol
from continuity.reasoning.base import ReasoningAdapter
from continuity.reasoning.claim_derivation import ClaimDerivationPipeline
from continuity.session_manager import SessionManager
from continuity.store.claims import Observation, Subject, SubjectKind
from continuity.store.schema import apply_migrations
from continuity.store.sqlite import SQLiteRepository, StoredDisclosurePolicy
from continuity.transactions import TransactionKind


logger = logging.getLogger(__name__)

_ASYNC_SHUTDOWN = object()


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _hash_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return f"{prefix}:{digest.hexdigest()[:16]}"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return _now_utc()
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _sanitize_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "-", value)


def _ensure_default_disclosure_policies(repository: SQLiteRepository) -> None:
    for policy_id in (
        "assistant_internal",
        "current_user",
        "current_peer",
        "shared_session",
        "host_internal",
    ):
        if repository.read_disclosure_policy(policy_id) is not None:
            continue
        policy = disclosure_policy_for(policy_id)
        repository.save_disclosure_policy(
            StoredDisclosurePolicy(
                policy_id=policy.policy_name,
                audience_principal=policy.principal.value,
                channel="|".join(channel.value for channel in policy.allowed_channels),
                purpose="|".join(purpose.value for purpose in policy.allowed_purposes),
                exposure_mode=policy.default_action.value,
                redaction_mode="none",
                capture_for_replay=policy.capture_for_replay,
            )
        )


@dataclass
class ContinuityHermesSession:
    key: str
    user_peer_id: str
    assistant_peer_id: str
    continuity_session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=_now_utc)
    updated_at: datetime = field(default_factory=_now_utc)
    metadata: dict[str, Any] = field(default_factory=dict)
    synced_count: int = 0

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        message = {
            "role": role,
            "content": content,
            "timestamp": _now_utc().isoformat(),
            **kwargs,
        }
        self.messages.append(message)
        self.updated_at = _now_utc()

    def get_history(self, max_messages: int = 50) -> list[dict[str, Any]]:
        recent = (
            self.messages[-max_messages:]
            if len(self.messages) > max_messages
            else self.messages
        )
        return [{"role": item["role"], "content": item["content"]} for item in recent]

    def clear(self) -> None:
        self.messages = []
        self.updated_at = _now_utc()
        self.synced_count = 0


class ContinuityHermesSessionManager:
    """Compatibility manager that preserves Hermes's Honcho-facing session contract."""

    def __init__(
        self,
        *,
        config: HermesMemoryConfig,
        reasoning_adapter: ReasoningAdapter,
        embedding_client: EmbeddingClientProtocol,
        vector_backend: VectorBackendProtocol,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        self._config = config
        self._owns_connection = connection is None
        self._connection = connection or self._open_connection(config.continuity_store_path)
        apply_migrations(self._connection)
        self._repository = SQLiteRepository(self._connection)
        _ensure_default_disclosure_policies(self._repository)

        continuity_config = config.continuity
        self._session_manager = SessionManager(self._repository, config=continuity_config)
        self._builder = ContinuityContextBuilder(
            connection=self._connection,
            embedding_client=embedding_client,
            vector_backend=vector_backend,
            policy_name=config.continuity_policy_name,
            reasoning_adapter=reasoning_adapter,
        )
        self._derivation = ClaimDerivationPipeline(
            connection=self._connection,
            adapter=reasoning_adapter,
            embedding_client=embedding_client,
            vector_backend=vector_backend,
            session_manager=self._session_manager,
            policy_name=config.continuity_policy_name,
        )

        self._cache: dict[str, ContinuityHermesSession] = {}
        self._context_cache: dict[str, dict[str, str]] = {}
        self._dialectic_cache: dict[str, str] = {}
        self._pending_observation_ids: dict[str, list[str]] = {}
        self._turn_counter = 0
        self._prefetch_cache_lock = threading.Lock()
        self._operation_lock = threading.RLock()
        self._async_queue: queue.Queue[tuple[str, tuple[str, ...], datetime] | object] | None = None
        self._async_thread: threading.Thread | None = None

        if config.write_frequency == "async":
            self._async_queue = queue.Queue()
            self._async_thread = threading.Thread(
                target=self._async_derivation_loop,
                name="continuity-hermes-async-derivation",
                daemon=True,
            )
            self._async_thread.start()

    @staticmethod
    def _open_connection(store_path: Path) -> sqlite3.Connection:
        path = Path(store_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(str(path), check_same_thread=False)

    def get_or_create(self, key: str) -> ContinuityHermesSession:
        cleaned_key = _clean_text(key, field_name="key")
        with self._operation_lock:
            cached = self._cache.get(cleaned_key)
            if cached is not None:
                return cached

            session_id = cleaned_key
            user_peer_id = self._user_peer_id(cleaned_key)
            assistant_peer_id = _clean_text(self._config.ai_peer, field_name="ai_peer")
            user_subject_id = self._user_subject_id(user_peer_id)
            assistant_subject_id = self._assistant_subject_id(assistant_peer_id)
            created_at = _now_utc()
            self._ensure_runtime_state(
                session_id=session_id,
                user_subject_id=user_subject_id,
                assistant_subject_id=assistant_subject_id,
                created_at=created_at,
                user_name=self._config.peer_name or user_peer_id,
                assistant_name=assistant_peer_id,
            )

            stored_messages = self._repository.list_messages(session_id=session_id)
            messages = [
                {
                    "role": message.role,
                    "content": message.content,
                    "timestamp": message.observed_at.isoformat(),
                }
                for message in stored_messages
            ]
            session = ContinuityHermesSession(
                key=cleaned_key,
                user_peer_id=user_peer_id,
                assistant_peer_id=assistant_peer_id,
                continuity_session_id=session_id,
                messages=messages,
                synced_count=len(messages),
            )
            self._cache[cleaned_key] = session
            return session

    def save(self, session: ContinuityHermesSession) -> None:
        with self._operation_lock:
            new_messages = session.messages[session.synced_count :]
            if not new_messages:
                return

            user_subject_id = self._user_subject_id(session.user_peer_id)
            assistant_subject_id = self._assistant_subject_id(session.assistant_peer_id)
            self._ensure_runtime_state(
                session_id=session.continuity_session_id,
                user_subject_id=user_subject_id,
                assistant_subject_id=assistant_subject_id,
                created_at=session.created_at,
                user_name=self._config.peer_name or session.user_peer_id,
                assistant_name=session.assistant_peer_id,
            )

            observation_ids: list[str] = []
            for offset, message in enumerate(new_messages, start=session.synced_count):
                observed_at = _parse_timestamp(message.get("timestamp"))
                role = str(message.get("role", "")).strip().lower() or "user"
                content = _clean_text(str(message.get("content", "")), field_name="content")
                author_subject_id = (
                    assistant_subject_id
                    if role == "assistant"
                    else user_subject_id
                )
                message_id = _hash_id(
                    "message",
                    session.continuity_session_id,
                    str(offset),
                    role,
                    content,
                    observed_at.isoformat(),
                )
                observation_id = _hash_id(
                    "observation",
                    session.continuity_session_id,
                    message_id,
                )
                self._session_manager.save_turn(
                    session_id=session.continuity_session_id,
                    message_id=message_id,
                    role=role,
                    author_subject_id=author_subject_id,
                    content=content,
                    observed_at=observed_at,
                    write_frequency=self._config.write_frequency,
                )
                self._repository.save_observation(
                    Observation(
                        observation_id=observation_id,
                        source_kind="session_message",
                        session_id=session.continuity_session_id,
                        author_subject_id=author_subject_id,
                        content=content,
                        observed_at=observed_at,
                        metadata={"origin": "hermes_session_sync"},
                    ),
                    message_id=message_id,
                )
                observation_ids.append(observation_id)

            session.synced_count = len(session.messages)
            session.updated_at = _now_utc()
            self._cache[session.key] = session
            self._schedule_derivation(
                session_id=session.continuity_session_id,
                observation_ids=tuple(observation_ids),
                run_at=session.updated_at,
            )

    def flush_all(self) -> None:
        with self._operation_lock:
            for session in tuple(self._cache.values()):
                self.save(session)
            for session_id, observation_ids in tuple(self._pending_observation_ids.items()):
                if observation_ids:
                    self._derive_observations(
                        session_id=session_id,
                        observation_ids=tuple(observation_ids),
                        run_at=_now_utc(),
                    )
                self._pending_observation_ids[session_id] = []

            if self._async_queue is not None:
                while not self._async_queue.empty():
                    item = self._async_queue.get_nowait()
                    if item is _ASYNC_SHUTDOWN:
                        continue
                    queued_session_id, queued_observation_ids, queued_at = item
                    self._derive_observations(
                        session_id=queued_session_id,
                        observation_ids=queued_observation_ids,
                        run_at=queued_at,
                    )

    def shutdown(self) -> None:
        self.flush_all()
        if self._async_queue is not None and self._async_thread is not None:
            self._async_queue.put(_ASYNC_SHUTDOWN)
            self._async_thread.join(timeout=10)
        if self._owns_connection:
            self._connection.close()

    def delete(self, key: str) -> bool:
        return self._cache.pop(_clean_text(key, field_name="key"), None) is not None

    def new_session(self, key: str) -> ContinuityHermesSession:
        with self._operation_lock:
            base_key = _clean_text(key, field_name="key")
            self._cache.pop(base_key, None)
            fresh = self.get_or_create(f"{base_key}:{int(_now_utc().timestamp())}")
            self._cache[base_key] = fresh
            return fresh

    def list_sessions(self) -> list[dict[str, Any]]:
        return [
            {
                "key": session.key,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "message_count": len(session.messages),
            }
            for session in self._cache.values()
        ]

    def prefetch_context(self, session_key: str, user_message: str | None = None) -> None:
        def _run() -> None:
            result = self.get_prefetch_context(session_key, user_message)
            if result:
                self.set_context_result(session_key, result)

        threading.Thread(
            target=_run,
            name="continuity-hermes-context-prefetch",
            daemon=True,
        ).start()

    def set_context_result(self, session_key: str, result: dict[str, str]) -> None:
        if not result:
            return
        with self._prefetch_cache_lock:
            self._context_cache[_clean_text(session_key, field_name="session_key")] = dict(result)

    def pop_context_result(self, session_key: str) -> dict[str, str]:
        with self._prefetch_cache_lock:
            return self._context_cache.pop(_clean_text(session_key, field_name="session_key"), {})

    def prefetch_dialectic(self, session_key: str, query: str) -> None:
        def _run() -> None:
            result = self.dialectic_query(session_key, query)
            if result:
                self.set_dialectic_result(session_key, result)

        threading.Thread(
            target=_run,
            name="continuity-hermes-dialectic-prefetch",
            daemon=True,
        ).start()

    def set_dialectic_result(self, session_key: str, result: str) -> None:
        if not result:
            return
        with self._prefetch_cache_lock:
            self._dialectic_cache[_clean_text(session_key, field_name="session_key")] = _clean_text(
                result,
                field_name="result",
            )

    def pop_dialectic_result(self, session_key: str) -> str:
        with self._prefetch_cache_lock:
            return self._dialectic_cache.pop(_clean_text(session_key, field_name="session_key"), "")

    def get_prefetch_context(
        self,
        session_key: str,
        user_message: str | None = None,
    ) -> dict[str, str]:
        with self._operation_lock:
            session = self.get_or_create(session_key)
            user_context = self._prompt_representation(
                session=session,
                session_id=session.continuity_session_id,
                subject_id=self._user_subject_id(session.user_peer_id),
                channel=DisclosureChannel.PROMPT,
                purpose=DisclosurePurpose.PROMPT,
                include_prompt=True,
            )
            ai_context = self._profile_representation(
                session=session,
                subject_id=self._assistant_subject_id(session.assistant_peer_id),
                channel=DisclosureChannel.PROMPT,
                purpose=DisclosurePurpose.PROMPT,
            )
            payload = {
                "representation": user_context["representation"],
                "card": user_context["card"],
                "ai_representation": ai_context["representation"],
                "ai_card": ai_context["card"],
            }
            if user_message:
                payload["query_hint"] = _clean_text(user_message, field_name="user_message")
            return payload

    def get_peer_card(self, session_key: str) -> list[str]:
        with self._operation_lock:
            session = self.get_or_create(session_key)
            try:
                profile = self._builder.build_profile_view(
                    subject_id=self._user_subject_id(session.user_peer_id),
                    disclosure_context=self._disclosure_context(
                        session=session,
                        channel=DisclosureChannel.PROMPT,
                        purpose=DisclosurePurpose.PROMPT,
                    ),
                )
            except LookupError:
                return []
            return [str(entry["summary"]) for entry in profile.payload["entries"]]

    def search_context(self, session_key: str, query: str, max_tokens: int = 800) -> str:
        with self._operation_lock:
            session = self.get_or_create(session_key)
            results = self._builder.search(
                query_text=_clean_text(query, field_name="query"),
                disclosure_context=self._disclosure_context(
                    session=session,
                    channel=DisclosureChannel.SEARCH,
                    purpose=DisclosurePurpose.SEARCH,
                ),
                subject_id=self._user_subject_id(session.user_peer_id),
                limit=5,
            )
            if not results:
                return ""
            budget = max(int(max_tokens), 1) * 4
            excerpts: list[str] = []
            current = 0
            for result in results:
                excerpt = result.excerpt.strip()
                if not excerpt:
                    continue
                if excerpts and current + len(excerpt) > budget:
                    break
                excerpts.append(excerpt)
                current += len(excerpt)
            return "\n\n".join(excerpts)

    def dialectic_query(
        self,
        session_key: str,
        query: str,
        reasoning_level: str | None = None,
        peer: str = "user",
    ) -> str:
        del reasoning_level
        with self._operation_lock:
            session = self.get_or_create(session_key)
            subject_id = (
                self._assistant_subject_id(session.assistant_peer_id)
                if str(peer).strip().lower() == "ai"
                else self._user_subject_id(session.user_peer_id)
            )
            view = self._builder.build_answer_view(
                question=_clean_text(query, field_name="query"),
                disclosure_context=self._disclosure_context(
                    session=session,
                    channel=DisclosureChannel.ANSWER,
                    purpose=DisclosurePurpose.ANSWER,
                ),
                subject_id=subject_id,
            )
            return str(view.payload["answer_text"])

    def create_conclusion(self, session_key: str, content: str) -> bool:
        cleaned = _clean_text(content, field_name="content")
        with self._operation_lock:
            session = self.get_or_create(session_key)
            observation_id = self._save_import_observation(
                session_id=session.continuity_session_id,
                author_subject_id=self._user_subject_id(session.user_peer_id),
                content=cleaned,
                metadata={"origin": "honcho_conclude", "target_subject": "user"},
            )
            result = self._derivation.derive_from_observations(
                observation_ids=(observation_id,),
                session_id=session.continuity_session_id,
                source_transaction_kind=TransactionKind.WRITE_CONCLUSION,
                run_at=_now_utc(),
            )
            return bool(result.claim_ids)

    def migrate_local_history(self, session_key: str, messages: list[dict[str, Any]]) -> bool:
        if not messages:
            return False
        with self._operation_lock:
            session = self.get_or_create(session_key)
            content = self._format_migration_transcript(session_key, messages).decode("utf-8")
            observation_id = self._save_import_observation(
                session_id=session.continuity_session_id,
                author_subject_id=self._user_subject_id(session.user_peer_id),
                content=content,
                metadata={
                    "origin": "local_jsonl",
                    "upload_name": "prior_history.txt",
                    "count": len(messages),
                },
            )
            self._derivation.derive_from_observations(
                observation_ids=(observation_id,),
                session_id=session.continuity_session_id,
                source_transaction_kind=TransactionKind.IMPORT_HISTORY,
                run_at=_now_utc(),
            )
            return True

    def migrate_memory_files(self, session_key: str, memory_dir: str) -> bool:
        session = self.get_or_create(session_key)
        root = Path(memory_dir).expanduser()
        if not root.exists():
            return False

        uploaded = False
        files = (
            (
                "MEMORY.md",
                "consolidated_memory.md",
                "Long-term agent notes and preferences",
                self._user_subject_id(session.user_peer_id),
                "user",
            ),
            (
                "USER.md",
                "user_profile.md",
                "User profile and preferences",
                self._user_subject_id(session.user_peer_id),
                "user",
            ),
            (
                "SOUL.md",
                "agent_soul.md",
                "Agent persona and identity configuration",
                self._assistant_subject_id(session.assistant_peer_id),
                "ai",
            ),
        )

        with self._operation_lock:
            for filename, upload_name, description, author_subject_id, target_peer in files:
                path = root / filename
                if not path.exists():
                    continue
                content = path.read_text(encoding="utf-8").strip()
                if not content:
                    continue
                wrapped = (
                    "<prior_memory_file>\n"
                    "<context>\n"
                    "This file was consolidated from local conversations BEFORE Honcho was activated.\n"
                    f"{description}. Treat as foundational context for this user.\n"
                    "</context>\n"
                    "\n"
                    f"{content}\n"
                    "</prior_memory_file>\n"
                )
                observation_id = self._save_import_observation(
                    session_id=session.continuity_session_id,
                    author_subject_id=author_subject_id,
                    content=wrapped,
                    metadata={
                        "origin": "local_memory",
                        "upload_name": upload_name,
                        "original_file": filename,
                        "target_peer": target_peer,
                    },
                )
                self._derivation.derive_from_observations(
                    observation_ids=(observation_id,),
                    session_id=session.continuity_session_id,
                    source_transaction_kind=TransactionKind.IMPORT_HISTORY,
                    run_at=_now_utc(),
                )
                uploaded = True
        return uploaded

    def seed_ai_identity(self, session_key: str, content: str, source: str = "manual") -> bool:
        cleaned = _clean_text(content, field_name="content")
        with self._operation_lock:
            session = self.get_or_create(session_key)
            observation_id = self._save_import_observation(
                session_id=session.continuity_session_id,
                author_subject_id=self._assistant_subject_id(session.assistant_peer_id),
                content=(
                    "<ai_identity_seed>\n"
                    f"<source>{_clean_text(source, field_name='source')}</source>\n\n"
                    f"{cleaned}\n"
                    "</ai_identity_seed>"
                ),
                metadata={"origin": "ai_identity_seed"},
            )
            result = self._derivation.derive_from_observations(
                observation_ids=(observation_id,),
                session_id=session.continuity_session_id,
                source_transaction_kind=TransactionKind.IMPORT_HISTORY,
                run_at=_now_utc(),
            )
            return bool(result.claim_ids or result.buffered_candidate_ids)

    def get_ai_representation(self, session_key: str) -> dict[str, str]:
        with self._operation_lock:
            session = self.get_or_create(session_key)
            return self._profile_representation(
                subject_id=self._assistant_subject_id(session.assistant_peer_id),
                channel=DisclosureChannel.PROMPT,
                purpose=DisclosurePurpose.PROMPT,
            )

    @staticmethod
    def _format_migration_transcript(session_key: str, messages: list[dict[str, Any]]) -> bytes:
        timestamps = [str(message.get("timestamp", "")) for message in messages]
        time_range = f"{timestamps[0]} to {timestamps[-1]}" if timestamps else "unknown"
        lines = [
            "<prior_conversation_history>",
            "<context>",
            "This conversation history occurred BEFORE the Honcho memory system was activated.",
            "These messages are the preceding elements of this conversation session and should",
            "be treated as foundational context for all subsequent interactions. The user and",
            "assistant have already established rapport through these exchanges.",
            "</context>",
            "",
            f'<transcript session_key="{session_key}" message_count="{len(messages)}"',
            f'           time_range="{time_range}">',
            "",
        ]
        for message in messages:
            lines.append(
                f"[{message.get('timestamp', '?')}] {message.get('role', 'unknown')}: "
                f"{message.get('content') or ''}"
            )
        lines.extend(("", "</transcript>", "</prior_conversation_history>"))
        return "\n".join(lines).encode("utf-8")

    def _schedule_derivation(
        self,
        *,
        session_id: str,
        observation_ids: tuple[str, ...],
        run_at: datetime,
    ) -> None:
        if not observation_ids:
            return
        self._turn_counter += 1
        pending = self._pending_observation_ids.setdefault(session_id, [])
        pending.extend(observation_ids)
        schedule = self._config.write_frequency

        if schedule == "async":
            assert self._async_queue is not None
            queued = tuple(pending)
            pending.clear()
            self._async_queue.put((session_id, queued, run_at))
            return

        if schedule == "turn":
            queued = tuple(pending)
            pending.clear()
            self._derive_observations(
                session_id=session_id,
                observation_ids=queued,
                run_at=run_at,
            )
            return

        if schedule == "session":
            return

        if isinstance(schedule, int) and schedule > 0 and self._turn_counter % schedule == 0:
            queued = tuple(pending)
            pending.clear()
            self._derive_observations(
                session_id=session_id,
                observation_ids=queued,
                run_at=run_at,
            )

    def _derive_observations(
        self,
        *,
        session_id: str,
        observation_ids: tuple[str, ...],
        run_at: datetime,
    ) -> None:
        if not observation_ids:
            return
        self._derivation.derive_from_observations(
            observation_ids=observation_ids,
            session_id=session_id,
            source_transaction_kind=TransactionKind.INGEST_TURN,
            run_at=run_at,
        )

    def _async_derivation_loop(self) -> None:
        assert self._async_queue is not None
        while True:
            item = self._async_queue.get()
            if item is _ASYNC_SHUTDOWN:
                return
            session_id, observation_ids, run_at = item
            try:
                with self._operation_lock:
                    self._derive_observations(
                        session_id=session_id,
                        observation_ids=observation_ids,
                        run_at=run_at,
                    )
            except Exception:
                logger.exception("Continuity async derivation failed for %s", session_id)

    def _ensure_runtime_state(
        self,
        *,
        session_id: str,
        user_subject_id: str,
        assistant_subject_id: str,
        created_at: datetime,
        user_name: str,
        assistant_name: str,
    ) -> None:
        self._session_manager.ensure_session(
            session_id=session_id,
            created_at=created_at,
        )
        if self._repository.read_subject(user_subject_id) is None:
            self._repository.save_subject(
                Subject(
                    subject_id=user_subject_id,
                    kind=SubjectKind.USER,
                    canonical_name=_clean_text(user_name, field_name="user_name"),
                ),
                created_at=created_at,
            )
        if self._repository.read_subject(assistant_subject_id) is None:
            self._repository.save_subject(
                Subject(
                    subject_id=assistant_subject_id,
                    kind=SubjectKind.ASSISTANT,
                    canonical_name=_clean_text(assistant_name, field_name="assistant_name"),
                ),
                created_at=created_at,
            )

    def _save_import_observation(
        self,
        *,
        session_id: str,
        author_subject_id: str,
        content: str,
        metadata: dict[str, Any],
    ) -> str:
        observed_at = _now_utc()
        observation_id = _hash_id(
            "observation",
            session_id,
            author_subject_id,
            content,
            observed_at.isoformat(),
        )
        self._repository.save_observation(
            Observation(
                observation_id=observation_id,
                source_kind="host_import",
                session_id=session_id,
                author_subject_id=author_subject_id,
                content=_clean_text(content, field_name="content"),
                observed_at=observed_at,
                metadata=dict(metadata),
            ),
        )
        return observation_id

    def _disclosure_context(
        self,
        *,
        session: ContinuityHermesSession,
        channel: DisclosureChannel,
        purpose: DisclosurePurpose,
        principal: DisclosurePrincipal = DisclosurePrincipal.CURRENT_PEER,
    ) -> DisclosureContext:
        return DisclosureContext(
            viewer=DisclosureViewer(
                viewer_kind=ViewerKind.ASSISTANT,
                viewer_subject_id=self._assistant_subject_id(session.assistant_peer_id),
                active_user_id=self._user_subject_id(session.user_peer_id),
                active_peer_id=self._assistant_subject_id(session.assistant_peer_id),
            ),
            audience_principal=principal,
            channel=channel,
            purpose=purpose,
            policy_stamp="hermes_v1@1.0.0",
        )

    def _prompt_representation(
        self,
        *,
        session: ContinuityHermesSession,
        session_id: str,
        subject_id: str,
        channel: DisclosureChannel,
        purpose: DisclosurePurpose,
        include_prompt: bool,
    ) -> dict[str, str]:
        disclosure_context = self._disclosure_context(session=session, channel=channel, purpose=purpose)
        representation = ""
        if include_prompt:
            try:
                prompt_view = self._builder.build_prompt_view(
                    session_id=session_id,
                    disclosure_context=disclosure_context,
                    recall_mode=self._config.recall_mode,
                )
                representation = str(prompt_view.payload["prompt_plan"]["model_text"]).strip()
            except LookupError:
                representation = ""

        profile = self._profile_representation(
            session=session,
            subject_id=subject_id,
            channel=channel,
            purpose=purpose,
        )
        if not representation:
            representation = profile["representation"]
        return {"representation": representation, "card": profile["card"]}

    def _profile_representation(
        self,
        *,
        session: ContinuityHermesSession,
        subject_id: str,
        channel: DisclosureChannel,
        purpose: DisclosurePurpose,
    ) -> dict[str, str]:
        disclosure_context = self._disclosure_context(
            session=session,
            channel=channel,
            purpose=purpose,
            principal=DisclosurePrincipal.ASSISTANT_INTERNAL,
        )
        try:
            view = self._builder.build_profile_view(
                subject_id=subject_id,
                disclosure_context=disclosure_context,
            )
        except LookupError:
            return {"representation": "", "card": ""}

        summaries = [str(entry["summary"]) for entry in view.payload["entries"]]
        return {
            "representation": "\n".join(summaries),
            "card": "\n".join(summaries),
        }

    def _user_peer_id(self, key: str) -> str:
        if self._config.peer_name:
            return _sanitize_id(self._config.peer_name)
        parts = key.split(":", 1)
        if len(parts) == 2:
            return _sanitize_id(f"user-{parts[0]}-{parts[1]}")
        return _sanitize_id(f"user-{key}")

    @staticmethod
    def _user_subject_id(user_peer_id: str) -> str:
        return f"subject:user:{_sanitize_id(user_peer_id)}"

    @staticmethod
    def _assistant_subject_id(assistant_peer_id: str) -> str:
        return f"subject:assistant:{_sanitize_id(assistant_peer_id)}"
