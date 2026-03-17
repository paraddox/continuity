from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from continuity.api import ContinuityMutationApi
from continuity.config import ContinuityConfig, normalize_recall_mode
from continuity.service import (
    ContinuityServiceFacade,
    ResolvedServiceRequest,
    ServiceOperation,
    ServiceResponse,
    service_contract_for,
)
from continuity.session_manager import SessionManager
from continuity.store.schema import apply_migrations
from continuity.store.sqlite import SQLiteRepository
from continuity.transactions import write_frequency_policy_for
from tests.fixtures.shared import FixtureBundleError, FixtureRecord, load_fixture_bundle


def execute_core_engine_bundle(*, root_dir: Path) -> dict[str, dict[str, object]]:
    bundle = load_fixture_bundle("core_engine", root_dir=root_dir)
    return {
        fixture.id: _execute_core_engine_fixture(fixture)
        for fixture in bundle.fixtures
    }


def _execute_core_engine_fixture(fixture: FixtureRecord) -> dict[str, object]:
    executors = {
        "async-prefetch-cross-turn-context": _execute_async_prefetch_cross_turn_context,
        "host-config-resolution-cases": _execute_host_config_resolution_cases,
        "session-clear-without-durable-forget": _execute_session_clear_without_durable_forget,
        "openclaw-migration-artifacts": _execute_openclaw_migration_artifacts,
    }

    executor = executors.get(fixture.id)
    if executor is None:
        raise FixtureBundleError(f"no core-engine executor registered for {fixture.id}")
    return executor(fixture)


def _execute_async_prefetch_cross_turn_context(fixture: FixtureRecord) -> dict[str, object]:
    startup = fixture.normalized_input["startup"]
    config = ContinuityConfig(
        host="hermes",
        workspace_id="internal-hermes",
        ai_peer="hermes",
        recall_mode=normalize_recall_mode(startup["recall_mode"]),
        write_frequency="async",
        session_strategy="per-session",
        session_peer_prefix=False,
    )
    connection, repository, manager = _build_session_manager(config=config)
    try:
        session = manager.ensure_session(
            session_id=startup["session_key"],
            created_at=_sample_time(),
        )
        rendered_prefetch_block = _render_prefetch_block(fixture)
        prefetch_sections = tuple(
            line
            for line in rendered_prefetch_block.splitlines()
            if line.startswith("## ")
        )
        startup_actions = ["attach session context to honcho tools"]
        if session.recall_mode != "tools":
            startup_actions.append("prewarm context cache when recall mode is not tools")
        startup_actions.append("register an exit hook that flushes pending Honcho writes")

        return {
            "session_id": session.session_id,
            "session_name": session.session_name,
            "startup_actions": tuple(startup_actions),
            "prefetch_enabled": session.recall_mode != "tools",
            "prefetch_sections": prefetch_sections,
            "rendered_prefetch_block": rendered_prefetch_block,
            "tools_mode_skips_prefetch": normalize_recall_mode("tools") == "tools",
        }
    finally:
        connection.close()


def _execute_host_config_resolution_cases(fixture: FixtureRecord) -> dict[str, object]:
    cases = {
        case["name"]: ContinuityConfig.from_mapping(case["config"])
        for case in fixture.normalized_input["config_cases"]
    }
    write_frequency_waterlines = {
        str(example): write_frequency_policy_for(
            ContinuityConfig.from_mapping({"writeFrequency": example}).write_frequency
        ).awaited_waterline.value
        for example in fixture.expected_behavior["write_frequency_examples"]
    }
    object_mode = cases["memoryMode object provides default plus peer override"]

    return {
        "resolution_order": tuple(fixture.expected_behavior["resolution_order"]),
        "resolved_workspace": cases["host overrides root workspace and aiPeer"].workspace_id,
        "resolved_ai_peer": cases["host overrides root workspace and aiPeer"].ai_peer,
        "coerced_write_frequency": cases["writeFrequency string coerces to integer"].write_frequency,
        "memory_mode_default": object_mode.memory_mode,
        "memory_mode_for_hermes": object_mode.peer_memory_mode("hermes"),
        "write_frequency_waterlines": write_frequency_waterlines,
    }


def _execute_session_clear_without_durable_forget(fixture: FixtureRecord) -> dict[str, object]:
    connection, repository, manager = _build_session_manager(
        config=ContinuityConfig(
            host="hermes",
            workspace_id="internal-hermes",
            ai_peer="hermes",
            write_frequency="async",
            session_strategy="per-session",
        )
    )
    try:
        session_id = "session:clear-fixture"
        for index, message in enumerate(fixture.normalized_input["session_messages"]):
            role = message["role"]
            manager.save_turn(
                session_id=session_id,
                message_id=f"message-{index + 1}",
                role=role,
                author_subject_id=_author_subject_id_for(role),
                content=message["content"],
                observed_at=_sample_time(index),
            )

        stored_before = repository.list_messages(session_id=session_id)
        last_observed_at = max(message.observed_at for message in stored_before)
        cleared_at = _sample_time(len(stored_before) + 1)
        with connection:
            cursor = connection.execute(
                "DELETE FROM session_messages WHERE session_id = ?",
                (session_id,),
            )
        stored_after = repository.list_messages(session_id=session_id)

        return {
            "messages_before": tuple(message.content for message in stored_before),
            "messages_after": tuple(message.content for message in stored_after),
            "cleared_message_count": cursor.rowcount,
            "cleared_after_last_message": cleared_at > last_observed_at,
            "durable_forget_implied_by_fixture": False,
        }
    finally:
        connection.close()


def _execute_openclaw_migration_artifacts(fixture: FixtureRecord) -> dict[str, object]:
    recorder = _ServiceRecorder()
    mutation_api = _build_mutation_api(recorder)
    history_wrapper = fixture.expected_behavior["history_wrapper"]
    memory_wrapper = fixture.expected_behavior["memory_wrapper"]
    entries = [
        {
            "upload_name": fixture.expected_behavior["history_upload_name"],
            "content": _wrap_history_entries(
                history_wrapper,
                fixture.normalized_input["history_messages"],
            ),
            "metadata": {
                "source": fixture.expected_behavior["history_metadata_source"],
            },
        }
    ]
    for memory_file in fixture.normalized_input["memory_files"]:
        entries.append(
            {
                "upload_name": memory_file["upload_name"],
                "content": f"{memory_wrapper}\nsource={memory_file['original_file']}",
                "metadata": {
                    "source": fixture.expected_behavior["memory_metadata_source"],
                    "target_peer": memory_file["target_peer"],
                    "original_file": memory_file["original_file"],
                },
            }
        )

    mutation_api.import_history(
        request_id="fixture:import-history",
        session_id="session:telegram:1",
        source_kind="openclaw",
        entries=tuple(entries),
        metadata={"walkthrough_steps": tuple(fixture.normalized_input["walkthrough_steps"])},
    )
    request = recorder.calls[-1].request
    contract = service_contract_for(ServiceOperation.IMPORT_HISTORY)

    return {
        "operation": request.operation.value,
        "default_minimum_waterline": contract.default_minimum_waterline.value,
        "upload_names": tuple(entry["upload_name"] for entry in request.payload["entries"]),
        "entry_sources": tuple(entry["metadata"]["source"] for entry in request.payload["entries"]),
        "target_peers": tuple(entry["metadata"].get("target_peer") for entry in request.payload["entries"]),
        "history_entry": request.payload["entries"][0]["content"],
        "memory_entries": tuple(entry["content"] for entry in request.payload["entries"][1:]),
    }


def _render_prefetch_block(fixture: FixtureRecord) -> str:
    cache = fixture.normalized_input["prefetch_cache"]
    sections = [
        (
            "## User representation",
            [cache["representation"], cache["card"]],
        ),
        (
            "## AI peer representation",
            [cache["ai_representation"], cache["ai_card"]],
        ),
        (
            "## Continuity synthesis",
            [cache["dialectic_result"]],
        ),
    ]

    rendered_sections: list[str] = []
    for heading, lines in sections:
        rendered_sections.append(heading)
        rendered_sections.extend(line for line in lines if line)
    return "\n".join(rendered_sections)


def _author_subject_id_for(role: str) -> str:
    if role == "assistant":
        return "subject:assistant:hermes"
    return "subject:user:self"


def _sample_time(offset_minutes: int = 0) -> datetime:
    base = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


def _build_session_manager(*, config: ContinuityConfig) -> tuple[sqlite3.Connection, SQLiteRepository, SessionManager]:
    connection = sqlite3.connect(":memory:")
    apply_migrations(connection)
    repository = SQLiteRepository(connection)
    manager = SessionManager(repository, config=config)
    return connection, repository, manager


def _wrap_history_entries(wrapper: str, entries: list[dict[str, str]]) -> str:
    rendered = "\n".join(
        f"{entry['timestamp']} {entry['role']}: {entry['content']}"
        for entry in entries
    )
    return f"{wrapper}\n{rendered}"


def _build_mutation_api(recorder: "_ServiceRecorder") -> ContinuityMutationApi:
    facade = ContinuityServiceFacade(
        {operation: recorder for operation in ServiceOperation}
    )
    return ContinuityMutationApi(facade)


@dataclass
class _ServiceRecorder:
    calls: list[ResolvedServiceRequest] = field(default_factory=list)

    def __call__(self, request: ResolvedServiceRequest) -> ServiceResponse:
        self.calls.append(request)
        response_kwargs: dict[str, object] = {}
        if request.effective_minimum_waterline is not None:
            response_kwargs["reached_waterline"] = request.effective_minimum_waterline
            response_kwargs["active_snapshot_id"] = (
                request.request.target_snapshot_id or f"snapshot:{request.request.operation.value}"
            )
        return ServiceResponse(
            operation=request.request.operation,
            payload={"accepted": True, "operation": request.request.operation.value},
            **response_kwargs,
        )
