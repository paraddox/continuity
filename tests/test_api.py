#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
    SERVICE_CONTRACT_VERSION,
    ServiceOperation,
    ServiceRequest,
    ServiceResponse,
    service_contract_for,
)
from continuity.transactions import DurabilityWaterline


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


if __name__ == "__main__":
    unittest.main()
