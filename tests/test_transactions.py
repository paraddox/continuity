#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from continuity.transactions import (
    DurabilityWaterline,
    TransactionKind,
    TransactionRunner,
    transaction_contract_for,
)


class TransactionRunnerTests(unittest.TestCase):
    def build_runner(self) -> tuple[TransactionRunner, list[str]]:
        calls: list[str] = []
        handlers = {
            phase: (
                lambda context, phase=phase: (
                    calls.append(phase.value),
                    {"phase": phase.value, "transaction": context.kind.value},
                )[1]
            )
            for phase in transaction_contract_for(TransactionKind.INGEST_TURN).phases
        }
        handlers.update(
            {
                phase: (
                    lambda context, phase=phase: (
                        calls.append(phase.value),
                        {"phase": phase.value, "transaction": context.kind.value},
                    )[1]
                )
                for phase in transaction_contract_for(TransactionKind.WRITE_CONCLUSION).phases
            }
        )
        return TransactionRunner(handlers), calls

    def test_runner_stops_at_requested_observation_waterline(self) -> None:
        runner, calls = self.build_runner()

        execution = runner.run(
            TransactionKind.INGEST_TURN,
            payload={"turn_id": "turn:1"},
            requested_waterline=DurabilityWaterline.OBSERVATION_COMMITTED,
        )

        self.assertEqual(
            execution.executed_phases,
            tuple(transaction_contract_for(TransactionKind.INGEST_TURN).phases[:2]),
        )
        self.assertEqual(
            execution.deferred_phases,
            tuple(transaction_contract_for(TransactionKind.INGEST_TURN).phases[2:]),
        )
        self.assertEqual(execution.reached_waterline, DurabilityWaterline.OBSERVATION_COMMITTED)
        self.assertEqual(calls, [phase.value for phase in execution.executed_phases])
        self.assertEqual(
            execution.phase_execution_for(execution.executed_phases[-1]).reached_waterline,
            DurabilityWaterline.OBSERVATION_COMMITTED,
        )

    def test_runner_stops_at_requested_views_compiled_waterline(self) -> None:
        runner, calls = self.build_runner()

        execution = runner.run(
            TransactionKind.WRITE_CONCLUSION,
            payload={"conclusion_id": "conclusion:1"},
            requested_waterline=DurabilityWaterline.VIEWS_COMPILED,
        )

        expected_executed = (
            transaction_contract_for(TransactionKind.WRITE_CONCLUSION)
            .phases[:9]
        )
        expected_deferred = (
            transaction_contract_for(TransactionKind.WRITE_CONCLUSION)
            .phases[9:]
        )

        self.assertEqual(execution.executed_phases, tuple(expected_executed))
        self.assertEqual(execution.deferred_phases, tuple(expected_deferred))
        self.assertEqual(execution.reached_waterline, DurabilityWaterline.VIEWS_COMPILED)
        self.assertEqual(calls, [phase.value for phase in execution.executed_phases])
        self.assertEqual(
            execution.phase_execution_for(execution.executed_phases[-1]).reached_waterline,
            DurabilityWaterline.VIEWS_COMPILED,
        )

    def test_runner_rejects_unreachable_requested_waterlines(self) -> None:
        runner, _ = self.build_runner()

        with self.assertRaisesRegex(ValueError, "cannot satisfy"):
            runner.run(
                TransactionKind.PUBLISH_SNAPSHOT,
                requested_waterline=DurabilityWaterline.PREFETCH_WARMED,
            )


if __name__ == "__main__":
    unittest.main()
