#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DOC_PATH = ROOT_DIR / "docs" / "architecture.md"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from continuity.api import (
    DeploymentMode,
    TransportAdapter,
    deployment_boundaries,
    deployment_boundary_for,
)
from continuity.disclosure import (
    DisclosureChannel,
    DisclosureContext,
    DisclosurePrincipal,
    DisclosurePurpose,
    DisclosureViewer,
    ViewerKind,
)
from continuity.service import (
    InspectionTarget,
    SERVICE_CONTRACT_VERSION,
    ServiceOperation,
    ServiceRequest,
    ServiceResponse,
    service_contract_for,
    service_contracts,
)
from continuity.transactions import DurabilityWaterline, TransactionKind
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


class ServiceContractTests(unittest.TestCase):
    def test_service_surface_is_closed_and_transport_neutral(self) -> None:
        self.assertEqual(
            {operation.value for operation in ServiceOperation},
            {
                "initialize",
                "save_turn",
                "search",
                "get_state_view",
                "get_timeline_view",
                "get_profile_view",
                "get_prompt_view",
                "answer_memory_question",
                "forget_memory",
                "write_conclusion",
                "list_memory_follow_ups",
                "resolve_memory_follow_up",
                "import_history",
                "publish_snapshot",
                "resolve_subject",
                "inspect_evidence",
                "inspect_admission",
                "inspect_resolution_queue",
                "inspect_disclosure",
                "inspect_forgetting",
                "inspect_epistemic_status",
                "record_outcome",
                "inspect_outcomes",
                "inspect_utility",
                "inspect_turn_decision",
                "inspect_policy",
                "inspect_compiler",
                "inspect_snapshot",
                "inspect_tiers",
            },
        )
        self.assertEqual(
            {target.value for target in InspectionTarget},
            {
                "evidence",
                "admission",
                "resolution_queue",
                "disclosure",
                "forgetting",
                "epistemic_status",
                "outcomes",
                "utility",
                "turn_decision",
                "policy",
                "compiler",
                "snapshot",
                "tiers",
            },
        )

        contracts = service_contracts()
        self.assertEqual(set(contracts), set(ServiceOperation))

        state_view = service_contract_for(ServiceOperation.GET_STATE_VIEW)
        self.assertEqual(state_view.view_kind, ViewKind.STATE)
        self.assertTrue(state_view.requires_disclosure_context)
        self.assertIsNone(state_view.transaction_kind)

        prompt_view = service_contract_for(ServiceOperation.GET_PROMPT_VIEW)
        self.assertEqual(prompt_view.view_kind, ViewKind.PROMPT)
        self.assertTrue(prompt_view.requires_disclosure_context)

        save_turn = service_contract_for(ServiceOperation.SAVE_TURN)
        self.assertEqual(save_turn.transaction_kind, TransactionKind.INGEST_TURN)
        self.assertEqual(
            save_turn.default_minimum_waterline,
            DurabilityWaterline.OBSERVATION_COMMITTED,
        )

        forget_memory = service_contract_for(ServiceOperation.FORGET_MEMORY)
        self.assertEqual(forget_memory.transaction_kind, TransactionKind.FORGET_MEMORY)
        self.assertEqual(
            forget_memory.default_minimum_waterline,
            DurabilityWaterline.VIEWS_COMPILED,
        )

        publish_snapshot = service_contract_for(ServiceOperation.PUBLISH_SNAPSHOT)
        self.assertEqual(
            publish_snapshot.transaction_kind,
            TransactionKind.PUBLISH_SNAPSHOT,
        )
        self.assertEqual(
            publish_snapshot.default_minimum_waterline,
            DurabilityWaterline.SNAPSHOT_PUBLISHED,
        )

        evidence = service_contract_for(ServiceOperation.INSPECT_EVIDENCE)
        self.assertEqual(evidence.inspection_target, InspectionTarget.EVIDENCE)

        compiler = service_contract_for(ServiceOperation.INSPECT_COMPILER)
        self.assertEqual(compiler.inspection_target, InspectionTarget.COMPILER)

    def test_request_and_response_envelopes_reject_python_specific_payloads(self) -> None:
        request = ServiceRequest(
            operation=ServiceOperation.GET_PROMPT_VIEW,
            request_id="request-1",
            disclosure_context=sample_context(),
            target_snapshot_id="snapshot:active",
            payload={
                "session_id": "session-1",
                "peer_id": "subject:peer:internal-hermes",
                "max_fragments": 8,
            },
        )

        self.assertEqual(request.contract_version, SERVICE_CONTRACT_VERSION)
        self.assertEqual(request.target_snapshot_id, "snapshot:active")

        with self.assertRaises(ValueError):
            ServiceRequest(
                operation=ServiceOperation.GET_PROMPT_VIEW,
                request_id="request-2",
                payload={"session_id": "session-1"},
            )

        with self.assertRaises(ValueError):
            ServiceRequest(
                operation=ServiceOperation.SAVE_TURN,
                request_id="request-3",
                minimum_waterline=DurabilityWaterline.PREFETCH_WARMED,
                payload={"turn_id": "turn-1"},
            )

        with self.assertRaises(TypeError):
            ServiceRequest(
                operation=ServiceOperation.SEARCH,
                request_id="request-4",
                disclosure_context=sample_context(),
                payload={"callback": sample_context},
            )

        response = ServiceResponse(
            operation=ServiceOperation.SAVE_TURN,
            payload={"turn_id": "turn-1", "accepted": True},
            reached_waterline=DurabilityWaterline.OBSERVATION_COMMITTED,
            active_snapshot_id="snapshot:turn-1",
            replay_artifact_ids=("replay:turn-1", "replay:turn-1"),
        )

        self.assertEqual(response.replay_artifact_ids, ("replay:turn-1",))

        with self.assertRaises(ValueError):
            ServiceResponse(
                operation=ServiceOperation.SEARCH,
                payload={"results": []},
                reached_waterline=DurabilityWaterline.OBSERVATION_COMMITTED,
            )


class DeploymentBoundaryTests(unittest.TestCase):
    def test_embedded_and_daemon_modes_share_engine_semantics_but_not_transport(self) -> None:
        self.assertEqual(
            {mode.value for mode in DeploymentMode},
            {"embedded", "daemon"},
        )

        boundaries = deployment_boundaries()
        self.assertEqual(set(boundaries), set(DeploymentMode))

        embedded = deployment_boundary_for(DeploymentMode.EMBEDDED)
        daemon = deployment_boundary_for(DeploymentMode.DAEMON)

        self.assertEqual(embedded.transport_adapter, TransportAdapter.IN_PROCESS)
        self.assertEqual(daemon.transport_adapter, TransportAdapter.UNIX_DOMAIN_SOCKET)
        self.assertEqual(embedded.contract_version, SERVICE_CONTRACT_VERSION)
        self.assertEqual(daemon.contract_version, SERVICE_CONTRACT_VERSION)
        self.assertTrue(embedded.sqlite_ownership.one_owner_process_per_store)
        self.assertEqual(embedded.sqlite_ownership.owner_role, "hermes_process")
        self.assertEqual(daemon.sqlite_ownership.owner_role, "daemon_process")
        self.assertTrue(embedded.sqlite_ownership.serialized_commit_lane)
        self.assertTrue(embedded.sqlite_ownership.in_process_worker_threads_only)
        self.assertFalse(embedded.sqlite_ownership.multi_process_write_coordination)
        self.assertTrue(daemon.local_only)
        self.assertFalse(daemon.hosted_service_assumptions)
        self.assertEqual(embedded.shared_transaction_kinds, daemon.shared_transaction_kinds)
        self.assertEqual(embedded.shared_durability_waterlines, daemon.shared_durability_waterlines)
        self.assertEqual(embedded.shared_semantics, daemon.shared_semantics)
        self.assertIn("replay_artifacts", embedded.shared_semantics)
        self.assertIn("snapshot_consistency", embedded.shared_semantics)
        self.assertIn("unix_domain_sockets", daemon.shell_responsibilities)


class ArchitectureDocTests(unittest.TestCase):
    def test_architecture_doc_mentions_service_facade_and_embedded_first_boundary(self) -> None:
        self.assertTrue(DOC_PATH.exists(), f"missing architecture doc: {DOC_PATH}")

        text = DOC_PATH.read_text(encoding="utf-8").lower()

        self.assertIn("continuity service facade", text)
        self.assertIn("transport-neutral", text)
        self.assertIn("embedded", text)
        self.assertIn("daemon", text)
        self.assertIn("unix domain sockets", text)
        self.assertIn("one owning hermes process per continuity sqlite store", text)
        self.assertIn("one serialized commit lane", text)
        self.assertIn("in-process worker threads only", text)
        self.assertIn("no multi-process write coordination in v1", text)


if __name__ == "__main__":
    unittest.main()
