#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from continuity.embeddings.ollama import (
    OllamaEmbeddingClient,
    OllamaEmbeddingConfig,
    OllamaEmbeddingError,
)


class RecordingTransport:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def post_json(
        self,
        *,
        url: str,
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> object:
        self.calls.append(
            {
                "url": url,
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        if not self._responses:
            raise AssertionError("no fake responses remaining")
        return self._responses.pop(0)


class OllamaEmbeddingConfigTests(unittest.TestCase):
    def test_default_config_tracks_model_and_embedding_fingerprint(self) -> None:
        config = OllamaEmbeddingConfig()

        self.assertEqual(config.model, "nomic-embed-text")
        self.assertIsNone(config.dimensions)
        self.assertEqual(config.fingerprint, "embedding:ollama_nomic_embed_text_native@1")


class OllamaEmbeddingClientTests(unittest.TestCase):
    def test_embed_posts_explicit_payload_and_parses_embeddings(self) -> None:
        transport = RecordingTransport(
            [
                {
                    "model": "nomic-embed-text",
                    "embeddings": [
                        [0.1, 0.2, 0.3],
                        [0.4, 0.5, 0.6],
                    ],
                    "total_duration": 10,
                    "load_duration": 2,
                    "prompt_eval_count": 7,
                }
            ]
        )
        client = OllamaEmbeddingClient(
            config=OllamaEmbeddingConfig(
                base_url="http://ollama.internal:11434",
                dimensions=3,
                truncate=False,
                keep_alive="5m",
                request_timeout_seconds=12.5,
            ),
            transport=transport,
        )

        batch = client.embed(("alpha", "beta"))

        self.assertEqual(batch.model, "nomic-embed-text")
        self.assertEqual(batch.dimensions, 3)
        self.assertEqual(batch.fingerprint, "embedding:ollama_nomic_embed_text_dim_3@1")
        self.assertEqual(
            batch.embeddings,
            (
                (0.1, 0.2, 0.3),
                (0.4, 0.5, 0.6),
            ),
        )
        self.assertEqual(batch.total_duration, 10)
        self.assertEqual(batch.load_duration, 2)
        self.assertEqual(batch.prompt_eval_count, 7)
        self.assertEqual(
            transport.calls,
            [
                {
                    "url": "http://ollama.internal:11434/api/embed",
                    "payload": {
                        "model": "nomic-embed-text",
                        "input": ["alpha", "beta"],
                        "truncate": False,
                        "dimensions": 3,
                        "keep_alive": "5m",
                    },
                    "timeout_seconds": 12.5,
                }
            ],
        )

    def test_embed_rejects_dimension_mismatch(self) -> None:
        transport = RecordingTransport(
            [
                {
                    "model": "nomic-embed-text",
                    "embeddings": [[0.1, 0.2]],
                }
            ]
        )
        client = OllamaEmbeddingClient(
            config=OllamaEmbeddingConfig(dimensions=3),
            transport=transport,
        )

        with self.assertRaisesRegex(OllamaEmbeddingError, "expected embeddings with 3 dimensions"):
            client.embed("alpha")

    def test_opt_in_live_round_trip_uses_local_ollama(self) -> None:
        if os.environ.get("CONTINUITY_RUN_LIVE_OLLAMA") != "1":
            self.skipTest("set CONTINUITY_RUN_LIVE_OLLAMA=1 to run live Ollama coverage")

        client = OllamaEmbeddingClient()
        batch = client.embed("continuity readiness probe")

        self.assertEqual(batch.model, "nomic-embed-text")
        self.assertEqual(len(batch.embeddings), 1)
        self.assertGreater(batch.dimensions, 0)


if __name__ == "__main__":
    unittest.main()
