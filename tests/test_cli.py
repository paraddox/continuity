#!/usr/bin/env python3

from __future__ import annotations

import io
import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from continuity.cli import main
from continuity.admission import (
    AdmissionAssessment,
    AdmissionDecisionTrace,
    AdmissionStrength,
    AdmissionThresholds,
    AdmissionWriteBudget,
)
from continuity.ontology import MemoryPartition
from continuity.hermes_compat.config import HermesMemoryBackendKind, HermesMemoryConfig
from continuity.store.claims import (
    AdmissionDecision,
    AdmissionOutcome,
    AggregationMode,
    CandidateMemory,
    Claim,
    ClaimProvenance,
    ClaimScope,
    MemoryLocus,
    Observation,
    Subject,
    SubjectKind,
)
from continuity.store.schema import apply_migrations
from continuity.store.sqlite import (
    SQLiteRepository,
    SessionMessageRecord,
    SessionRecord,
    StoredDisclosurePolicy,
)


def sample_time(offset_minutes: int = 0) -> datetime:
    base = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


def open_memory_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    apply_migrations(connection)
    return connection


def build_admission_trace(*, candidate_id: str, claim_type: str, recorded_at: datetime) -> AdmissionDecisionTrace:
    return AdmissionDecisionTrace(
        decision=AdmissionDecision(
            candidate_id=candidate_id,
            outcome=AdmissionOutcome.DURABLE_CLAIM,
            recorded_at=recorded_at,
            rationale="explicit user statement",
        ),
        claim_type=claim_type,
        policy_stamp="hermes_v1@1.0.0",
        assessment=AdmissionAssessment(
            claim_type=claim_type,
            evidence=AdmissionStrength.HIGH,
            novelty=AdmissionStrength.HIGH,
            stability=AdmissionStrength.HIGH,
            salience=AdmissionStrength.MEDIUM,
            rationale="explicit user statement",
            utility_signals=("prompt_inclusion",),
        ),
        thresholds=AdmissionThresholds(
            evidence=AdmissionStrength.MEDIUM,
            novelty=AdmissionStrength.MEDIUM,
            stability=AdmissionStrength.MEDIUM,
            salience=AdmissionStrength.LOW,
        ),
        budget=AdmissionWriteBudget(
            partition=MemoryPartition.USER_MEMORY,
            window_key="session:telegram:123456",
            limit=8,
            used=0,
        ),
    )


def seed_cli_state(path: Path) -> None:
    connection = open_memory_database(path)
    repository = SQLiteRepository(connection)

    repository.save_disclosure_policy(
        StoredDisclosurePolicy(
            policy_id="current_peer",
            audience_principal="current_peer",
            channel="prompt|answer|search|profile|evidence",
            purpose="prompt|answer|search|profile|evidence",
            exposure_mode="allow",
            redaction_mode="none",
            capture_for_replay=True,
        )
    )
    repository.save_subject(
        Subject(
            subject_id="subject:user:alice",
            kind=SubjectKind.USER,
            canonical_name="Alice Example",
        ),
        created_at=sample_time(),
    )
    repository.save_session(
        SessionRecord(
            session_id="telegram:123456",
            host_namespace="hermes",
            session_name="telegram:123456",
            recall_mode="hybrid",
            write_frequency="async",
            created_at=sample_time(),
        )
    )
    repository.save_message(
        SessionMessageRecord(
            message_id="message-1",
            session_id="telegram:123456",
            role="user",
            author_subject_id="subject:user:alice",
            content="I prefer espresso over tea.",
            observed_at=sample_time(1),
            metadata={"transport": "telegram"},
        )
    )
    repository.save_observation(
        Observation(
            observation_id="obs-1",
            source_kind="session_message",
            session_id="telegram:123456",
            author_subject_id="subject:user:alice",
            content="I prefer espresso over tea.",
            observed_at=sample_time(1),
            metadata={"message_id": "message-1"},
        ),
        message_id="message-1",
    )
    candidate = CandidateMemory(
        candidate_id="candidate-1",
        claim_type="preference",
        subject_id="subject:user:alice",
        scope=ClaimScope.USER,
        value={"preferred": "espresso", "over": "tea"},
        source_observation_ids=("obs-1",),
    )
    repository.save_candidate_memory(candidate, created_at=sample_time(2))
    trace = build_admission_trace(
        candidate_id=candidate.candidate_id,
        claim_type=candidate.claim_type,
        recorded_at=sample_time(2),
    )
    repository.admissions.record_decision(trace)
    claim = Claim.from_candidate(
        claim_id="claim-1",
        candidate=candidate,
        admission=trace.decision,
        locus=MemoryLocus(
            subject_id="subject:user:alice",
            locus_key="preference/beverage",
            scope=ClaimScope.USER,
            default_disclosure_policy="current_peer",
            conflict_set_key="preference.beverage",
            aggregation_mode=AggregationMode.LATEST_WINS,
        ),
        provenance=ClaimProvenance(observation_ids=("obs-1",)),
        disclosure_policy="current_peer",
        observed_at=sample_time(1),
        learned_at=sample_time(3),
        valid_from=sample_time(1),
    )
    repository.save_claim(claim)
    connection.close()


class ContinuityCliTests(unittest.TestCase):
    def build_config(self, db_path: Path) -> HermesMemoryConfig:
        return HermesMemoryConfig(
            backend=HermesMemoryBackendKind.CONTINUITY,
            enabled=True,
            continuity_store_path=db_path,
        )

    def run_cli(self, argv: list[str], *, config: HermesMemoryConfig) -> str:
        buffer = io.StringIO()
        with (
            patch("continuity.cli.HermesMemoryConfig.from_global_config", return_value=config),
            redirect_stdout(buffer),
        ):
            exit_code = main(argv)
        self.assertEqual(exit_code, 0)
        return buffer.getvalue()

    def test_status_json_reports_config_and_table_counts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "continuity.db"
            seed_cli_state(db_path)

            output = self.run_cli(["status", "--json"], config=self.build_config(db_path))
            payload = json.loads(output)

        self.assertEqual(payload["backend"], "continuity")
        self.assertEqual(payload["store_path"], str(db_path))
        self.assertTrue(payload["store_exists"])
        self.assertEqual(payload["counts"]["sessions"], 1)
        self.assertEqual(payload["counts"]["claims"], 1)
        self.assertEqual(payload["counts"]["observations"], 1)

    def test_sessions_json_lists_recent_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "continuity.db"
            seed_cli_state(db_path)

            output = self.run_cli(["sessions", "--json"], config=self.build_config(db_path))
            payload = json.loads(output)

        self.assertEqual(len(payload["sessions"]), 1)
        self.assertEqual(payload["sessions"][0]["session_id"], "telegram:123456")

    def test_claims_json_lists_recent_claims(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "continuity.db"
            seed_cli_state(db_path)

            output = self.run_cli(["claims", "--json"], config=self.build_config(db_path))
            payload = json.loads(output)

        self.assertEqual(len(payload["claims"]), 1)
        self.assertEqual(payload["claims"][0]["claim_id"], "claim-1")
        self.assertEqual(payload["claims"][0]["locus_key"], "preference/beverage")

    def test_status_json_handles_missing_store(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "missing.db"
            output = self.run_cli(["status", "--json"], config=self.build_config(db_path))
            payload = json.loads(output)

        self.assertFalse(payload["store_exists"])
        self.assertEqual(payload["counts"]["sessions"], 0)
        self.assertEqual(payload["counts"]["claims"], 0)


if __name__ == "__main__":
    unittest.main()
