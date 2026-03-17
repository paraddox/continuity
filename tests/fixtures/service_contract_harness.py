from __future__ import annotations

from dataclasses import dataclass, field
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
    service_contract_for,
)
from tests.fixtures.shared import FixtureBundleError, FixtureRecord, load_fixture_bundle


def execute_service_contract_bundle(*, root_dir: Path) -> dict[str, dict[str, object]]:
    bundle = load_fixture_bundle("service_contract", root_dir=root_dir)
    return {
        fixture.id: _execute_service_contract_fixture(fixture)
        for fixture in bundle.fixtures
    }


def _execute_service_contract_fixture(fixture: FixtureRecord) -> dict[str, object]:
    executors = {
        "honcho-tool-surface": _execute_honcho_tool_surface,
        "recall-mode-contracts": _execute_recall_mode_contracts,
        "host-config-resolution-cases": _execute_host_config_resolution_cases,
        "openclaw-migration-artifacts": _execute_openclaw_migration_artifacts,
    }

    executor = executors.get(fixture.id)
    if executor is None:
        raise FixtureBundleError(f"no service-contract executor registered for {fixture.id}")
    return executor(fixture)


def _execute_honcho_tool_surface(fixture: FixtureRecord) -> dict[str, object]:
    recorder = _ServiceRecorder()
    read_api, mutation_api = _build_apis(recorder)
    context = _sample_context()

    operations: dict[str, str] = {}
    responses: list[ServiceResponse] = []
    for tool in fixture.normalized_input["tools"]:
        tool_name = tool["name"]
        if tool_name == "honcho_profile":
            response = read_api.get_profile_view(
                request_id="fixture:profile",
                view_key="profile:subject:user:self",
                disclosure_context=context,
                target_snapshot_id="snapshot:active",
            )
        elif tool_name == "honcho_search":
            response = read_api.search(
                request_id="fixture:search",
                query_text="espresso",
                disclosure_context=context,
                target_snapshot_id="snapshot:active",
            )
        elif tool_name == "honcho_context":
            response = read_api.answer_memory_question(
                request_id="fixture:context",
                question="What do I prefer?",
                disclosure_context=context,
                target_snapshot_id="snapshot:active",
                subject_id="subject:peer:internal-hermes",
            )
        elif tool_name == "honcho_conclude":
            response = mutation_api.write_conclusion(
                request_id="fixture:conclude",
                session_id="session:telegram:1",
                subject_id="subject:user:self",
                locus_key="preference/coffee",
                conclusion="User prefers espresso.",
            )
        else:
            raise FixtureBundleError(f"unknown honcho tool fixture: {tool_name}")

        operations[tool_name] = recorder.calls[-1].request.operation.value
        responses.append(response)

    return {
        "tool_names": tuple(tool["name"] for tool in fixture.normalized_input["tools"]),
        "operation_bindings": operations,
        "required_parameters": {
            tool["name"]: tuple(tool.get("required_parameters", ()))
            for tool in fixture.normalized_input["tools"]
        },
        "optional_parameters": {
            tool["name"]: tuple(tool.get("optional_parameters", ()))
            for tool in fixture.normalized_input["tools"]
        },
        "llm_backed": {
            tool["name"]: bool(tool.get("llm_backed", False))
            for tool in fixture.normalized_input["tools"]
        },
        "availability_gate": fixture.normalized_input["availability_gate"],
        "structured_response_payloads": all(isinstance(response.payload, dict) for response in responses),
    }


def _execute_recall_mode_contracts(fixture: FixtureRecord) -> dict[str, object]:
    runtime_examples = {
        example["mode"]: _recall_mode_runtime(example["mode"])
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
    }


def _execute_host_config_resolution_cases(fixture: FixtureRecord) -> dict[str, object]:
    cases = {
        case["name"]: ContinuityConfig.from_mapping(case["config"])
        for case in fixture.normalized_input["config_cases"]
    }
    write_frequency_examples = tuple(
        ContinuityConfig.from_mapping({"writeFrequency": example}).write_frequency
        for example in fixture.expected_behavior["write_frequency_examples"]
    )

    object_mode = cases["memoryMode object provides default plus peer override"]

    return {
        "resolution_order": tuple(fixture.expected_behavior["resolution_order"]),
        "resolved_workspace": cases["host overrides root workspace and aiPeer"].workspace_id,
        "resolved_ai_peer": cases["host overrides root workspace and aiPeer"].ai_peer,
        "coerced_write_frequency": cases["writeFrequency string coerces to integer"].write_frequency,
        "memory_mode_default": object_mode.memory_mode,
        "memory_mode_for_hermes": object_mode.peer_memory_mode("hermes"),
        "write_frequency_examples": write_frequency_examples,
    }


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
    contract = service_contract_for(ServiceOperation.IMPORT_HISTORY)

    return {
        "operation": request.operation.value,
        "default_minimum_waterline": contract.default_minimum_waterline.value,
        "source_kind": request.payload["source_kind"],
        "entries": request.payload["entries"],
    }


def _recall_mode_runtime(mode: str) -> dict[str, bool]:
    normalized = normalize_recall_mode(mode)
    return {
        "tools_visible": normalized != "context",
        "context_injected": normalized != "tools",
    }


def _wrap_history_entries(wrapper: str, entries: list[dict[str, str]]) -> str:
    rendered = "\n".join(
        f"{entry['timestamp']} {entry['role']}: {entry['content']}"
        for entry in entries
    )
    return f"{wrapper}\n{rendered}"


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
