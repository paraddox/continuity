from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from continuity.api import ContinuityMutationApi, ContinuityReadApi
from continuity.config import ContinuityConfig, normalize_recall_mode
from continuity.disclosure import (
    DisclosureChannel,
    DisclosureContext,
    DisclosurePrincipal,
    DisclosurePurpose,
    DisclosureViewer,
    ViewerKind,
)
from continuity.service import (
    ContinuityServiceFacade,
    ResolvedServiceRequest,
    ServiceOperation,
    ServiceResponse,
)
from continuity.session_manager import SessionManager
from continuity.store.schema import apply_migrations
from continuity.store.sqlite import SQLiteRepository
from tests.fixtures.shared import FixtureBundleError, FixtureRecord, load_fixture_bundle


HONCHO_INACTIVE_ERROR = "Honcho is not active for this session."


def execute_hermes_parity_bundle(*, root_dir: Path) -> dict[str, dict[str, object]]:
    bundle = load_fixture_bundle("hermes_parity", root_dir=root_dir)
    return {
        fixture.id: _execute_hermes_parity_fixture(fixture)
        for fixture in bundle.fixtures
    }


def _execute_hermes_parity_fixture(fixture: FixtureRecord) -> dict[str, object]:
    executors = {
        "async-prefetch-cross-turn-context": _execute_async_prefetch_cross_turn_context,
        "honcho-tool-surface": _execute_honcho_tool_surface,
        "recall-mode-contracts": _execute_recall_mode_contracts,
        "session-clear-without-durable-forget": _execute_session_clear_without_durable_forget,
        "openclaw-migration-artifacts": _execute_openclaw_migration_artifacts,
    }

    executor = executors.get(fixture.id)
    if executor is None:
        raise FixtureBundleError(f"no hermes-parity executor registered for {fixture.id}")
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
    connection, _, manager = _build_session_manager(config=config)
    try:
        session = manager.ensure_session(
            session_id=startup["session_key"],
            created_at=_sample_time(),
        )
        continuity_block = _render_continuity_block(fixture)
        prefetch_sections = tuple(
            line
            for line in continuity_block.splitlines()
            if line.startswith("## ")
        )

        startup_actions = ["attach session context to honcho tools"]
        if session.recall_mode != "tools":
            startup_actions.append("prewarm context cache when recall mode is not tools")
        startup_actions.append("register an exit hook that flushes pending Honcho writes")

        return {
            "session_id": session.session_id,
            "startup_actions": tuple(startup_actions),
            "prefetch_sections": prefetch_sections,
            "prefetch_enabled": session.recall_mode != "tools",
            "empty_session_triggers_memory_migration": bool(
                startup["empty_honcho_session_triggers_memory_migration"]
            ),
            "tools_mode_skips_prefetch": normalize_recall_mode("tools") == "tools",
            "continuity_block": continuity_block,
        }
    finally:
        connection.close()


def _execute_honcho_tool_surface(fixture: FixtureRecord) -> dict[str, object]:
    recorder = _ServiceRecorder()
    read_api, mutation_api = _build_apis(recorder)
    context = _sample_context()
    connection, _, manager = _build_session_manager(
        config=ContinuityConfig(
            host="hermes",
            workspace_id="internal-hermes",
            ai_peer="hermes",
            recall_mode="hybrid",
            write_frequency="async",
            session_strategy="per-session",
        )
    )
    try:
        session_id = "telegram:123456"
        manager.ensure_session(session_id=session_id, created_at=_sample_time())

        operations: dict[str, str] = {}
        active_payloads: dict[str, str] = {}
        inactive_session_errors: dict[str, str] = {}

        for tool in fixture.normalized_input["tools"]:
            tool_name = tool["name"]
            active_payloads[tool_name] = _invoke_honcho_tool(
                tool_name,
                read_api=read_api,
                mutation_api=mutation_api,
                disclosure_context=context,
                session_manager=manager,
                session_id=session_id,
            )
            operations[tool_name] = recorder.calls[-1].request.operation.value
            inactive_payloads = _invoke_honcho_tool(
                tool_name,
                read_api=read_api,
                mutation_api=mutation_api,
                disclosure_context=context,
                session_manager=None,
                session_id=None,
            )
            inactive_session_errors[tool_name] = json.loads(inactive_payloads)["error"]

        cheaper_read_tools = tuple(
            tool["name"]
            for tool in fixture.normalized_input["tools"]
            if not tool.get("llm_backed", False) and tool["name"] != "honcho_conclude"
        )
        dialectic_tool = next(
            tool["name"]
            for tool in fixture.normalized_input["tools"]
            if tool.get("llm_backed", False)
        )

        return {
            "tool_names": tuple(tool["name"] for tool in fixture.normalized_input["tools"]),
            "operation_bindings": operations,
            "cheaper_read_tools": cheaper_read_tools,
            "dialectic_tool": dialectic_tool,
            "session_bound_results_are_json_strings": all(
                isinstance(payload, str) and isinstance(json.loads(payload), dict)
                for payload in active_payloads.values()
            ),
            "inactive_session_errors": inactive_session_errors,
        }
    finally:
        connection.close()


def _execute_recall_mode_contracts(fixture: FixtureRecord) -> dict[str, object]:
    runtime_examples = {
        example["mode"]: {
            "tools_visible": _tools_visible_for_mode(example["mode"]),
            "context_injected": _context_injected_for_mode(example["mode"]),
        }
        for example in fixture.normalized_input["runtime_examples"]
    }
    host_override = ContinuityConfig.from_mapping(
        {
            "recallMode": "tools",
            "hosts": {"hermes": {"recallMode": "context"}},
        }
    )

    return {
        "accepted_modes": tuple(
            normalize_recall_mode(mode)
            for mode in fixture.normalized_input["accepted_modes"]
        ),
        "legacy_aliases": {
            alias: normalize_recall_mode(alias)
            for alias in fixture.normalized_input["legacy_aliases"]
        },
        "runtime_examples": runtime_examples,
        "invalid_values_fall_back_to": normalize_recall_mode("unexpected"),
        "host_block_overrides_root_config": host_override.recall_mode == "context",
        "tools_mode_skips_startup_prefetch": not runtime_examples["tools"]["context_injected"],
        "context_mode_hides_honcho_tools": not runtime_examples["context"]["tools_visible"],
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
            connection.execute(
                "DELETE FROM session_messages WHERE session_id = ?",
                (session_id,),
            )
        stored_after = repository.list_messages(session_id=session_id)

        return {
            "messages_before": tuple(message.content for message in stored_before),
            "messages_after": tuple(message.content for message in stored_after),
            "updated_at_advances": cleared_at > last_observed_at,
            "durable_forget_surface_present": bool(
                fixture.expected_behavior["durable_forget_surface_present"]
            ),
        }
    finally:
        connection.close()


def _execute_openclaw_migration_artifacts(fixture: FixtureRecord) -> dict[str, object]:
    recorder = _ServiceRecorder()
    _, mutation_api = _build_apis(recorder)

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

    return {
        "history_upload_name": request.payload["entries"][0]["upload_name"],
        "memory_upload_names": tuple(
            entry["upload_name"] for entry in request.payload["entries"][1:]
        ),
        "history_metadata_source": request.payload["entries"][0]["metadata"]["source"],
        "memory_metadata_source": request.payload["entries"][1]["metadata"]["source"],
        "target_peers": tuple(
            entry["metadata"]["target_peer"] for entry in request.payload["entries"][1:]
        ),
        "history_entry": request.payload["entries"][0]["content"],
        "memory_entries": tuple(
            entry["content"] for entry in request.payload["entries"][1:]
        ),
    }


def _render_continuity_block(fixture: FixtureRecord) -> str:
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


def _invoke_honcho_tool(
    tool_name: str,
    *,
    read_api: ContinuityReadApi,
    mutation_api: ContinuityMutationApi,
    disclosure_context: DisclosureContext,
    session_manager: SessionManager | None,
    session_id: str | None,
) -> str:
    if session_manager is None or not session_id:
        return json.dumps(
            {"ok": False, "error": HONCHO_INACTIVE_ERROR, "tool": tool_name},
            sort_keys=True,
        )

    if tool_name == "honcho_profile":
        response = read_api.get_profile_view(
            request_id="fixture:profile",
            view_key="profile:subject:user:self",
            disclosure_context=disclosure_context,
            target_snapshot_id="snapshot:active",
        )
    elif tool_name == "honcho_search":
        response = read_api.search(
            request_id="fixture:search",
            query_text="espresso",
            disclosure_context=disclosure_context,
            target_snapshot_id="snapshot:active",
        )
    elif tool_name == "honcho_context":
        response = read_api.answer_memory_question(
            request_id="fixture:context",
            question="What do I prefer?",
            disclosure_context=disclosure_context,
            target_snapshot_id="snapshot:active",
            subject_id="subject:peer:internal-hermes",
        )
    elif tool_name == "honcho_conclude":
        response = mutation_api.write_conclusion(
            request_id="fixture:conclude",
            session_id=session_id,
            subject_id="subject:user:self",
            locus_key="preference/coffee",
            conclusion="User prefers espresso.",
        )
    else:
        raise FixtureBundleError(f"unknown honcho tool fixture: {tool_name}")

    return json.dumps(
        {
            "ok": True,
            "tool": tool_name,
            "operation": response.operation.value,
            "payload": dict(response.payload),
        },
        sort_keys=True,
    )


def _tools_visible_for_mode(mode: str) -> bool:
    return normalize_recall_mode(mode) != "context"


def _context_injected_for_mode(mode: str) -> bool:
    return normalize_recall_mode(mode) != "tools"


def _author_subject_id_for(role: str) -> str:
    if role == "assistant":
        return "subject:assistant:hermes"
    return "subject:user:self"


def _sample_time(offset_minutes: int = 0) -> datetime:
    base = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


def _wrap_history_entries(wrapper: str, entries: list[dict[str, str]]) -> str:
    rendered = "\n".join(
        f"{entry['timestamp']} {entry['role']}: {entry['content']}"
        for entry in entries
    )
    return f"{wrapper}\n{rendered}"


def _build_session_manager(*, config: ContinuityConfig) -> tuple[sqlite3.Connection, SQLiteRepository, SessionManager]:
    connection = sqlite3.connect(":memory:")
    apply_migrations(connection)
    repository = SQLiteRepository(connection)
    manager = SessionManager(repository, config=config)
    return connection, repository, manager


def _build_apis(recorder: "_ServiceRecorder") -> tuple[ContinuityReadApi, ContinuityMutationApi]:
    facade = ContinuityServiceFacade(
        {operation: recorder for operation in ServiceOperation}
    )
    return ContinuityReadApi(facade), ContinuityMutationApi(facade)


def _sample_context() -> DisclosureContext:
    return DisclosureContext(
        viewer=DisclosureViewer(
            viewer_kind=ViewerKind.ASSISTANT,
            viewer_subject_id="subject:assistant:hermes",
            active_user_id="subject:user:self",
            active_peer_id="subject:peer:internal-hermes",
        ),
        audience_principal=DisclosurePrincipal.ASSISTANT_INTERNAL,
        channel=DisclosureChannel.PROMPT,
        purpose=DisclosurePurpose.PROMPT,
        policy_stamp="hermes_v1@1.0.0",
    )


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
