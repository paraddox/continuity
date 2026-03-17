#!/usr/bin/env python3

from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from continuity.hermes_compat.config import (
    ContinuityVectorBackendKind,
    HermesMemoryBackendKind,
    HermesMemoryConfig,
)
from continuity.hermes_compat.factory import create_continuity_backend
from continuity.hermes_compat.manager import ContinuityHermesSessionManager
from continuity.index.zvec_index import InMemoryZvecBackend
from continuity.reasoning.base import AnswerQueryRequest, ClaimDerivationRequest, RawStructuredOutput, TextResponse


def sample_time(offset_minutes: int = 0) -> datetime:
    base = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


@dataclass(frozen=True, slots=True)
class FakeEmbeddingBatch:
    model: str
    fingerprint: str
    embeddings: tuple[tuple[float, ...], ...]


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.model = "nomic-embed-text"
        self.fingerprint = "embedding:fake_nomic_embed_text@1"

    def embed(self, inputs: str | tuple[str, ...]) -> FakeEmbeddingBatch:
        texts = (inputs,) if isinstance(inputs, str) else tuple(inputs)
        return FakeEmbeddingBatch(
            model=self.model,
            fingerprint=self.fingerprint,
            embeddings=tuple((float(len(text.split())), 1.0, 0.0, 0.0) for text in texts),
        )


class FakeReasoningAdapter:
    def __init__(self) -> None:
        self.answer_requests: list[AnswerQueryRequest] = []
        self.claim_requests: list[ClaimDerivationRequest] = []

    def answer_query(self, request: AnswerQueryRequest) -> TextResponse:
        self.answer_requests.append(request)
        return TextResponse(text="Alice currently prefers espresso.")

    def generate_structured(self, request: object) -> object:
        raise AssertionError("not used in Hermes compat tests")

    def summarize_session(self, request: object) -> object:
        raise AssertionError("not used in Hermes compat tests")

    def derive_claims(self, request: ClaimDerivationRequest) -> RawStructuredOutput:
        self.claim_requests.append(request)
        observations = tuple(message.content for message in request.observations)
        joined = " ".join(observations).casefold()
        candidates: list[dict[str, object]] = []

        if "espresso" in joined:
            subject_ref = "observation:0.author"
            if "user prefers espresso" in joined:
                subject_ref = "subject:user:alice"
            candidates.append(
                {
                    "claim_type": "preference",
                    "subject_ref": subject_ref,
                    "scope": "user",
                    "locus_key": "preference/favorite_drink",
                    "value": {"drink": "espresso"},
                    "evidence_refs": ["observation:0"],
                }
            )

        return RawStructuredOutput(payload={"candidates": candidates})


class HermesMemoryConfigTests(unittest.TestCase):
    def test_host_continuity_backend_auto_enables_without_honcho_api_key(self) -> None:
        config = HermesMemoryConfig.from_mapping(
            {
                "backend": "honcho",
                "apiKey": "unused-key",
                "hosts": {
                    "hermes": {
                        "backend": "continuity",
                        "recallMode": "context",
                        "memoryMode": {"default": "hybrid", "hermes": "honcho"},
                        "continuity": {
                            "storePath": "/tmp/continuity-test.db",
                            "vectorBackend": "zvec",
                            "collectionPath": "/tmp/continuity-zvec",
                            "embeddingDimensions": 768,
                            "reasoningModel": "gpt-5.4-mini",
                            "reasoningEffort": "medium",
                        },
                    }
                },
            }
        )

        self.assertEqual(config.backend, HermesMemoryBackendKind.CONTINUITY)
        self.assertTrue(config.enabled)
        self.assertEqual(config.recall_mode, "context")
        self.assertEqual(config.peer_memory_mode("hermes"), "honcho")
        self.assertEqual(config.continuity_vector_backend, ContinuityVectorBackendKind.ZVEC)
        self.assertEqual(config.continuity_embedding_dimensions, 768)
        self.assertEqual(config.continuity_reasoning_model, "gpt-5.4-mini")
        self.assertEqual(config.continuity_reasoning_effort, "medium")
        self.assertEqual(config.continuity_store_path, Path("/tmp/continuity-test.db"))

    def test_factory_returns_none_for_non_continuity_backend(self) -> None:
        manager, config = create_continuity_backend(
            HermesMemoryConfig.from_mapping({"backend": "honcho"})
        )

        self.assertIsNone(manager)
        self.assertEqual(config.backend, HermesMemoryBackendKind.HONCHO)

    def test_env_backend_override_wins_over_file_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text('{"backend":"honcho"}', encoding="utf-8")
            original = os.environ.get("HERMES_MEMORY_BACKEND")
            os.environ["HERMES_MEMORY_BACKEND"] = "continuity"
            try:
                config = HermesMemoryConfig.from_global_config(config_path=config_path)
            finally:
                if original is None:
                    os.environ.pop("HERMES_MEMORY_BACKEND", None)
                else:
                    os.environ["HERMES_MEMORY_BACKEND"] = original

        self.assertEqual(config.backend, HermesMemoryBackendKind.CONTINUITY)
        self.assertTrue(config.enabled)


class ContinuityHermesSessionManagerTests(unittest.TestCase):
    def build_manager(self) -> tuple[ContinuityHermesSessionManager, FakeReasoningAdapter, tempfile.TemporaryDirectory[str]]:
        tmpdir = tempfile.TemporaryDirectory()
        config = HermesMemoryConfig(
            backend=HermesMemoryBackendKind.CONTINUITY,
            enabled=True,
            continuity_store_path=Path(tmpdir.name) / "continuity.db",
            continuity=HermesMemoryConfig.from_mapping(
                {
                    "backend": "continuity",
                    "peerName": "alice",
                    "aiPeer": "hermes",
                    "writeFrequency": "turn",
                    "recallMode": "hybrid",
                    "sessionPeerPrefix": True,
                }
            ).continuity,
        )
        adapter = FakeReasoningAdapter()
        manager = ContinuityHermesSessionManager(
            config=config,
            reasoning_adapter=adapter,
            embedding_client=FakeEmbeddingClient(),
            vector_backend=InMemoryZvecBackend(),
        )
        self.addCleanup(manager.shutdown)
        self.addCleanup(tmpdir.cleanup)
        return manager, adapter, tmpdir

    def test_save_populates_profile_search_and_answer_paths(self) -> None:
        manager, adapter, _ = self.build_manager()
        session = manager.get_or_create("telegram:123456")
        session.add_message("user", "I prefer espresso over tea.", timestamp=sample_time().isoformat())
        session.add_message("assistant", "Noted.", timestamp=sample_time(1).isoformat())

        manager.save(session)

        self.assertEqual(session.synced_count, 2)
        self.assertEqual(len(adapter.claim_requests), 1)
        self.assertIn("espresso", " ".join(manager.get_peer_card("telegram:123456")).casefold())
        self.assertIn(
            "espresso",
            manager.search_context("telegram:123456", "drink").casefold(),
        )
        self.assertIn(
            "espresso",
            manager.dialectic_query("telegram:123456", "What does Alice prefer?").casefold(),
        )

    def test_prefetch_and_conclude_reuse_same_runtime(self) -> None:
        manager, _, _ = self.build_manager()
        session = manager.get_or_create("telegram:123456")
        session.add_message("user", "I prefer espresso.", timestamp=sample_time().isoformat())
        manager.save(session)

        context = manager.get_prefetch_context("telegram:123456", "What were we working on?")
        self.assertIn("espresso", context["representation"].casefold())
        manager.set_context_result("telegram:123456", context)
        self.assertEqual(manager.pop_context_result("telegram:123456"), context)

        manager.set_dialectic_result("telegram:123456", "You were discussing espresso.")
        self.assertIn("espresso", manager.pop_dialectic_result("telegram:123456").casefold())

        self.assertTrue(manager.create_conclusion("telegram:123456", "User prefers espresso."))

    def test_migrate_memory_files_imports_local_memory_wrappers(self) -> None:
        manager, adapter, tmpdir = self.build_manager()
        session = manager.get_or_create("telegram:123456")
        memory_dir = Path(tmpdir.name) / "memories"
        memory_dir.mkdir(parents=True, exist_ok=True)
        (memory_dir / "MEMORY.md").write_text("User likes espresso.\n", encoding="utf-8")
        (memory_dir / "SOUL.md").write_text("Hermes is methodical.\n", encoding="utf-8")

        migrated = manager.migrate_memory_files(session.key, str(memory_dir))

        self.assertTrue(migrated)
        joined = " ".join(
            message.content for request in adapter.claim_requests for message in request.observations
        )
        self.assertIn("prior_memory_file", joined)
        self.assertIn("User likes espresso.", joined)


if __name__ == "__main__":
    unittest.main()
