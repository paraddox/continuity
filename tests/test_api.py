#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from continuity.api import ContinuityReadApi
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
    InspectionTarget,
    ResolvedServiceRequest,
    SERVICE_CONTRACT_VERSION,
    ServiceOperation,
    ServiceRequest,
    ServiceResponse,
    service_contract_for,
)
from continuity.transactions import DurabilityWaterline
from continuity.views import ViewKind


def sample_context() -> DisclosureContext:
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


def request_for(operation: ServiceOperation) -> ServiceRequest:
    contract = service_contract_for(operation)
    kwargs: dict[str, object] = {
        "operation": operation,
        "request_id": f"request:{operation.value}",
        "payload": {"operation": operation.value},
    }
    if contract.requires_disclosure_context:
        kwargs["disclosure_context"] = sample_context()
    if operation in {
        ServiceOperation.GET_PROMPT_VIEW,
        ServiceOperation.ANSWER_MEMORY_QUESTION,
        ServiceOperation.INSPECT_EVIDENCE,
        ServiceOperation.PUBLISH_SNAPSHOT,
    }:
        kwargs["target_snapshot_id"] = "snapshot:active"
    if operation is ServiceOperation.FORGET_MEMORY:
        kwargs["minimum_waterline"] = DurabilityWaterline.SNAPSHOT_PUBLISHED
    return ServiceRequest(**kwargs)


class RecordingExecutor:
    def __init__(self) -> None:
        self.calls: list[ResolvedServiceRequest] = []

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
            payload={
                "operation": request.request.operation.value,
                "family": request.contract.family.value,
                "echo": dict(request.request.payload),
            },
            **response_kwargs,
        )


class ContinuityServiceFacadeTests(unittest.TestCase):
    def build_facade(self, executor: RecordingExecutor | None = None) -> tuple[ContinuityServiceFacade, RecordingExecutor]:
        shared_executor = executor or RecordingExecutor()
        facade = ContinuityServiceFacade(
            {
                operation: shared_executor
                for operation in ServiceOperation
            }
        )
        return facade, shared_executor

    def test_facade_exposes_closed_supported_operation_surface(self) -> None:
        facade, _ = self.build_facade()

        self.assertEqual(facade.supported_operations(), tuple(ServiceOperation))
        self.assertEqual(
            tuple(contract.operation for contract in facade.contracts()),
            tuple(ServiceOperation),
        )

    def test_facade_executes_every_operation_through_transport_neutral_envelopes(self) -> None:
        facade, executor = self.build_facade()

        for operation in ServiceOperation:
            with self.subTest(operation=operation):
                request = request_for(operation)

                response = facade.execute(request)
                resolved = executor.calls[-1]
                contract = service_contract_for(operation)

                self.assertIsInstance(resolved, ResolvedServiceRequest)
                self.assertEqual(resolved.request, request)
                self.assertEqual(resolved.contract, contract)
                self.assertEqual(response.operation, operation)
                self.assertEqual(response.payload["operation"], operation.value)
                self.assertEqual(response.payload["family"], contract.family.value)
                self.assertEqual(response.contract_version, SERVICE_CONTRACT_VERSION)

                expected_waterline = request.minimum_waterline or contract.default_minimum_waterline
                self.assertEqual(resolved.effective_minimum_waterline, expected_waterline)
                self.assertEqual(response.reached_waterline, expected_waterline)

    def test_facade_rejects_underreported_mutation_waterlines(self) -> None:
        def low_waterline_executor(request: ResolvedServiceRequest) -> ServiceResponse:
            return ServiceResponse(
                operation=request.request.operation,
                payload={"accepted": True},
                reached_waterline=DurabilityWaterline.OBSERVATION_COMMITTED,
                active_snapshot_id="snapshot:turn-1",
            )

        facade = ContinuityServiceFacade({ServiceOperation.SAVE_TURN: low_waterline_executor})

        with self.assertRaisesRegex(ValueError, "must reach at least"):
            facade.execute(
                ServiceRequest(
                    operation=ServiceOperation.SAVE_TURN,
                    request_id="request:save-turn",
                    payload={"turn_id": "turn-1"},
                    minimum_waterline=DurabilityWaterline.VIEWS_COMPILED,
                )
            )

    def test_facade_rejects_responses_for_the_wrong_operation(self) -> None:
        def wrong_operation_executor(request: ResolvedServiceRequest) -> ServiceResponse:
            return ServiceResponse(
                operation=ServiceOperation.SEARCH,
                payload={"accepted": True},
            )

        facade = ContinuityServiceFacade({ServiceOperation.INITIALIZE: wrong_operation_executor})

        with self.assertRaisesRegex(ValueError, "must respond with the same operation"):
            facade.execute(
                ServiceRequest(
                    operation=ServiceOperation.INITIALIZE,
                    request_id="request:initialize",
                    payload={"host_namespace": "hermes"},
                )
            )


class ContinuityReadApiTests(unittest.TestCase):
    def build_api(self) -> tuple[ContinuityReadApi, RecordingExecutor]:
        executor = RecordingExecutor()
        facade = ContinuityServiceFacade(
            {
                operation: executor
                for operation in ServiceOperation
            }
        )
        return ContinuityReadApi(facade), executor

    def test_read_api_dispatches_named_read_operations(self) -> None:
        api, executor = self.build_api()
        context = sample_context()

        cases = (
            (
                api.search,
                {
                    "request_id": "request:search",
                    "query_text": "coffee preference",
                    "disclosure_context": context,
                    "target_snapshot_id": "snapshot:active",
                    "limit": 5,
                    "subject_id": "subject:user:self",
                    "view_kinds": (ViewKind.STATE, ViewKind.PROFILE),
                },
                ServiceOperation.SEARCH,
                {
                    "query_text": "coffee preference",
                    "limit": 5,
                    "subject_id": "subject:user:self",
                    "view_kinds": ("state", "profile"),
                },
            ),
            (
                api.get_state_view,
                {
                    "request_id": "request:state",
                    "view_key": "state:subject:user:self:preference/coffee",
                    "disclosure_context": context,
                    "target_snapshot_id": "snapshot:active",
                },
                ServiceOperation.GET_STATE_VIEW,
                {
                    "view_key": "state:subject:user:self:preference/coffee",
                },
            ),
            (
                api.get_timeline_view,
                {
                    "request_id": "request:timeline",
                    "view_key": "timeline:subject:user:self:preference/coffee",
                    "disclosure_context": context,
                },
                ServiceOperation.GET_TIMELINE_VIEW,
                {
                    "view_key": "timeline:subject:user:self:preference/coffee",
                },
            ),
            (
                api.get_profile_view,
                {
                    "request_id": "request:profile",
                    "view_key": "profile:subject:user:self",
                    "disclosure_context": context,
                },
                ServiceOperation.GET_PROFILE_VIEW,
                {
                    "view_key": "profile:subject:user:self",
                },
            ),
            (
                api.get_prompt_view,
                {
                    "request_id": "request:prompt",
                    "view_key": "prompt:session:turn-1",
                    "disclosure_context": context,
                    "target_snapshot_id": "snapshot:active",
                },
                ServiceOperation.GET_PROMPT_VIEW,
                {
                    "view_key": "prompt:session:turn-1",
                },
            ),
            (
                api.answer_memory_question,
                {
                    "request_id": "request:answer",
                    "question": "What do you know about my coffee preferences?",
                    "disclosure_context": context,
                    "target_snapshot_id": "snapshot:active",
                    "subject_id": "subject:user:self",
                },
                ServiceOperation.ANSWER_MEMORY_QUESTION,
                {
                    "question": "What do you know about my coffee preferences?",
                    "subject_id": "subject:user:self",
                },
            ),
            (
                api.list_memory_follow_ups,
                {
                    "request_id": "request:follow-ups",
                    "subject_id": "subject:user:self",
                    "status": "open",
                    "limit": 10,
                },
                ServiceOperation.LIST_MEMORY_FOLLOW_UPS,
                {
                    "subject_id": "subject:user:self",
                    "status": "open",
                    "limit": 10,
                },
            ),
            (
                api.resolve_subject,
                {
                    "request_id": "request:resolve-subject",
                    "reference_text": "Soso",
                    "subject_kind": "user",
                },
                ServiceOperation.RESOLVE_SUBJECT,
                {
                    "reference_text": "Soso",
                    "subject_kind": "user",
                },
            ),
        )

        for method, kwargs, expected_operation, expected_payload in cases:
            with self.subTest(operation=expected_operation):
                response = method(**kwargs)
                request = executor.calls[-1].request

                self.assertEqual(request.operation, expected_operation)
                self.assertEqual(dict(request.payload), expected_payload)
                self.assertEqual(response.operation, expected_operation)

    def test_read_api_dispatches_named_inspection_operations(self) -> None:
        api, executor = self.build_api()

        cases = (
            (
                api.inspect_evidence,
                {
                    "request_id": "request:evidence",
                    "target_id": "answer:1",
                    "target_kind": "answer",
                    "target_snapshot_id": "snapshot:active",
                },
                ServiceOperation.INSPECT_EVIDENCE,
                InspectionTarget.EVIDENCE,
                {
                    "target_id": "answer:1",
                    "target_kind": "answer",
                },
            ),
            (
                api.inspect_admission,
                {
                    "request_id": "request:admission",
                    "candidate_id": "candidate:1",
                    "outcome": "durable_claim",
                    "limit": 3,
                },
                ServiceOperation.INSPECT_ADMISSION,
                InspectionTarget.ADMISSION,
                {
                    "candidate_id": "candidate:1",
                    "outcome": "durable_claim",
                    "limit": 3,
                },
            ),
            (
                api.inspect_resolution_queue,
                {
                    "request_id": "request:resolution",
                    "status": "open",
                    "session_id": "session:1",
                },
                ServiceOperation.INSPECT_RESOLUTION_QUEUE,
                InspectionTarget.RESOLUTION_QUEUE,
                {
                    "status": "open",
                    "session_id": "session:1",
                },
            ),
            (
                api.inspect_disclosure,
                {
                    "request_id": "request:disclosure",
                    "target_id": "view:prompt:1",
                    "target_kind": "compiled_view",
                    "policy_name": "assistant_internal",
                },
                ServiceOperation.INSPECT_DISCLOSURE,
                InspectionTarget.DISCLOSURE,
                {
                    "target_id": "view:prompt:1",
                    "target_kind": "compiled_view",
                    "policy_name": "assistant_internal",
                },
            ),
            (
                api.inspect_forgetting,
                {
                    "request_id": "request:forgetting",
                    "target_id": "claim:1",
                    "target_kind": "claim",
                    "mode": "suppress",
                },
                ServiceOperation.INSPECT_FORGETTING,
                InspectionTarget.FORGETTING,
                {
                    "target_id": "claim:1",
                    "target_kind": "claim",
                    "mode": "suppress",
                },
            ),
            (
                api.inspect_epistemic_status,
                {
                    "request_id": "request:epistemics",
                    "claim_id": "claim:1",
                    "view_key": "state:subject:user:self:preference/coffee",
                },
                ServiceOperation.INSPECT_EPISTEMIC_STATUS,
                InspectionTarget.EPISTEMIC_STATUS,
                {
                    "claim_id": "claim:1",
                    "view_key": "state:subject:user:self:preference/coffee",
                },
            ),
            (
                api.inspect_outcomes,
                {
                    "request_id": "request:outcomes",
                    "target_id": "answer:1",
                    "target_kind": "answer",
                    "label": "answer_cited",
                },
                ServiceOperation.INSPECT_OUTCOMES,
                InspectionTarget.OUTCOMES,
                {
                    "target_id": "answer:1",
                    "target_kind": "answer",
                    "label": "answer_cited",
                },
            ),
            (
                api.inspect_utility,
                {
                    "request_id": "request:utility",
                    "target_id": "answer:1",
                    "target_kind": "answer",
                    "policy_stamp": "hermes_v1@1.0.0",
                },
                ServiceOperation.INSPECT_UTILITY,
                InspectionTarget.UTILITY,
                {
                    "target_id": "answer:1",
                    "target_kind": "answer",
                    "policy_stamp": "hermes_v1@1.0.0",
                },
            ),
            (
                api.inspect_turn_decision,
                {
                    "request_id": "request:turn-decision",
                    "artifact_id": "replay:1",
                    "run_id": "run:1",
                },
                ServiceOperation.INSPECT_TURN_DECISION,
                InspectionTarget.TURN_DECISION,
                {
                    "artifact_id": "replay:1",
                    "run_id": "run:1",
                },
            ),
            (
                api.inspect_policy,
                {
                    "request_id": "request:policy",
                    "policy_stamp": "hermes_v1@1.0.0",
                },
                ServiceOperation.INSPECT_POLICY,
                InspectionTarget.POLICY,
                {
                    "policy_stamp": "hermes_v1@1.0.0",
                },
            ),
            (
                api.inspect_compiler,
                {
                    "request_id": "request:compiler",
                    "node_id": "node:1",
                    "dirty_only": True,
                    "limit": 20,
                },
                ServiceOperation.INSPECT_COMPILER,
                InspectionTarget.COMPILER,
                {
                    "node_id": "node:1",
                    "dirty_only": True,
                    "limit": 20,
                },
            ),
            (
                api.inspect_snapshot,
                {
                    "request_id": "request:snapshot",
                    "snapshot_id": "snapshot:active",
                    "include_diff_from": "snapshot:previous",
                },
                ServiceOperation.INSPECT_SNAPSHOT,
                InspectionTarget.SNAPSHOT,
                {
                    "snapshot_id": "snapshot:active",
                    "include_diff_from": "snapshot:previous",
                },
            ),
            (
                api.inspect_tiers,
                {
                    "request_id": "request:tiers",
                    "target_kind": "compiled_view",
                    "target_id": "view:prompt:1",
                    "policy_stamp": "hermes_v1@1.0.0",
                    "tiers": ("hot", "warm"),
                },
                ServiceOperation.INSPECT_TIERS,
                InspectionTarget.TIERS,
                {
                    "target_kind": "compiled_view",
                    "target_id": "view:prompt:1",
                    "policy_stamp": "hermes_v1@1.0.0",
                    "tiers": ("hot", "warm"),
                },
            ),
        )

        for method, kwargs, expected_operation, expected_target, expected_payload in cases:
            with self.subTest(operation=expected_operation):
                response = method(**kwargs)
                request = executor.calls[-1].request
                contract = executor.calls[-1].contract

                self.assertEqual(request.operation, expected_operation)
                self.assertEqual(contract.inspection_target, expected_target)
                self.assertEqual(dict(request.payload), expected_payload)
                self.assertEqual(response.payload["family"], "inspection")


if __name__ == "__main__":
    unittest.main()
