#!/usr/bin/env python3

from __future__ import annotations

import io
import json
import sys
import tempfile
import tomllib
import unittest
from contextlib import redirect_stdout
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class PackagingBootstrapTests(unittest.TestCase):
    def test_pyproject_defines_retrieval_backend_extra(self) -> None:
        pyproject_path = ROOT_DIR / "pyproject.toml"
        self.assertTrue(pyproject_path.exists(), "pyproject.toml should define the repository bootstrap contract")

        pyproject = tomllib.loads(pyproject_path.read_text())

        self.assertEqual(pyproject["project"]["name"], "continuity")
        self.assertEqual(pyproject["project"]["requires-python"], ">=3.11")
        self.assertEqual(pyproject["tool"]["setuptools"]["package-dir"][""], "src")
        self.assertIn("retrieval-zvec", pyproject["project"]["optional-dependencies"])
        self.assertIn("zvec>=0.2.1b0,<0.3", pyproject["project"]["optional-dependencies"]["retrieval-zvec"])

    def test_bootstrap_doc_records_safe_zvec_floor_for_non_avx512_hosts(self) -> None:
        bootstrap_doc = (ROOT_DIR / "docs" / "retrieval-backend-bootstrap.md").read_text()

        self.assertIn("zvec>=0.2.1b0,<0.3", bootstrap_doc)
        self.assertIn("AVX-512", bootstrap_doc)


class ZvecSmokeModuleTests(unittest.TestCase):
    def test_package_init_keeps_smoke_module_lazy_for_python_m_execution(self) -> None:
        sys.modules.pop("continuity.index", None)
        sys.modules.pop("continuity.index.zvec_smoke", None)

        import continuity.index  # noqa: F401

        self.assertNotIn(
            "continuity.index.zvec_smoke",
            sys.modules,
            "continuity.index should not pre-import the smoke module before python -m execution",
        )

    def test_run_zvec_backend_smoke_rejects_existing_collection_path(self) -> None:
        smoke_module_path = SRC_DIR / "continuity" / "index" / "zvec_smoke.py"
        self.assertTrue(smoke_module_path.exists(), "zvec smoke module should exist for real-backend verification")

        from continuity.index import InMemoryZvecBackend
        from continuity.index.zvec_smoke import run_zvec_backend_smoke

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                run_zvec_backend_smoke(
                    collection_path=temp_dir,
                    backend_factory=lambda **_: InMemoryZvecBackend(),
                )

    def test_run_zvec_backend_smoke_exercises_backend_contract(self) -> None:
        smoke_module_path = SRC_DIR / "continuity" / "index" / "zvec_smoke.py"
        self.assertTrue(smoke_module_path.exists(), "zvec smoke module should exist for real-backend verification")

        from continuity.index import InMemoryZvecBackend
        from continuity.index.zvec_smoke import run_zvec_backend_smoke

        backend_calls: list[dict[str, object]] = []

        def backend_factory(*, collection_path: str, dimensions: int, collection_name: str = "continuity_index") -> InMemoryZvecBackend:
            backend_calls.append(
                {
                    "collection_path": collection_path,
                    "dimensions": dimensions,
                    "collection_name": collection_name,
                }
            )
            return InMemoryZvecBackend()

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_zvec_backend_smoke(
                collection_path=Path(temp_dir) / "collection",
                backend_factory=backend_factory,
            )

        self.assertEqual(len(backend_calls), 1)
        self.assertEqual(backend_calls[0]["dimensions"], 4)
        self.assertEqual(backend_calls[0]["collection_name"], "continuity_smoke")
        self.assertTrue(str(backend_calls[0]["collection_path"]).endswith("collection"))
        self.assertEqual(
            set(result.indexed_record_ids),
            {"vector:smoke:alpha", "vector:smoke:beta"},
        )
        self.assertEqual(result.top_hit_record_id, "vector:smoke:alpha")
        self.assertEqual(result.remaining_record_ids, ("vector:smoke:alpha",))

    def test_cli_prints_json_summary(self) -> None:
        smoke_module_path = SRC_DIR / "continuity" / "index" / "zvec_smoke.py"
        self.assertTrue(smoke_module_path.exists(), "zvec smoke module should expose a CLI entrypoint")

        from continuity.index import InMemoryZvecBackend
        from continuity.index.zvec_smoke import main

        def backend_factory(*, collection_path: str, dimensions: int, collection_name: str = "continuity_index") -> InMemoryZvecBackend:
            return InMemoryZvecBackend()

        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    ["--collection-path", str(Path(temp_dir) / "collection")],
                    backend_factory=backend_factory,
                )

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["top_hit_record_id"], "vector:smoke:alpha")
        self.assertEqual(payload["remaining_record_ids"], ["vector:smoke:alpha"])


if __name__ == "__main__":
    unittest.main()
