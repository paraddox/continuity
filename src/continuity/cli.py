"""Small operator-facing CLI for inspecting a local Continuity store."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from continuity.hermes_compat.config import HermesMemoryConfig


COUNT_TABLES = (
    "sessions",
    "session_messages",
    "observations",
    "candidate_memories",
    "claims",
    "belief_states",
    "compiled_views",
    "resolution_queue_items",
    "compiled_utility_weights",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a local Continuity memory store.")
    parser.add_argument("--db", help="Override the SQLite store path.")
    subparsers = parser.add_subparsers(dest="command")

    status = subparsers.add_parser("status", help="Show store path, backend config, and table counts.")
    status.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    sessions = subparsers.add_parser("sessions", help="Show recent sessions.")
    sessions.add_argument("--limit", type=int, default=10, help="Maximum number of sessions to show.")
    sessions.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    claims = subparsers.add_parser("claims", help="Show recent claims.")
    claims.add_argument("--limit", type=int, default=20, help="Maximum number of claims to show.")
    claims.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    return parser


def _config(db_override: str | None) -> tuple[HermesMemoryConfig, Path]:
    config = HermesMemoryConfig.from_global_config()
    store_path = Path(db_override).expanduser() if db_override else config.continuity_store_path
    return config, store_path


def _connect(path: Path) -> sqlite3.Connection | None:
    if not path.exists():
        return None
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _counts(connection: sqlite3.Connection | None) -> dict[str, int]:
    if connection is None:
        return {table: 0 for table in COUNT_TABLES}
    result: dict[str, int] = {}
    for table in COUNT_TABLES:
        result[table] = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    return result


def _recent_sessions(connection: sqlite3.Connection | None, *, limit: int) -> list[dict[str, Any]]:
    if connection is None:
        return []
    rows = connection.execute(
        """
        SELECT session_id, session_name, recall_mode, write_frequency, created_at
        FROM sessions
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [
        {
            "session_id": row["session_id"],
            "session_name": row["session_name"],
            "recall_mode": row["recall_mode"],
            "write_frequency": row["write_frequency"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def _recent_claims(connection: sqlite3.Connection | None, *, limit: int) -> list[dict[str, Any]]:
    if connection is None:
        return []
    rows = connection.execute(
        """
        SELECT
            claims.claim_id,
            claims.claim_type,
            claims.subject_id,
            memory_loci.locus_key,
            claims.disclosure_policy_id,
            claims.learned_at
        FROM claims
        JOIN memory_loci ON memory_loci.locus_id = claims.locus_id
        ORDER BY claims.learned_at DESC, claims.claim_id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [
        {
            "claim_id": row["claim_id"],
            "claim_type": row["claim_type"],
            "subject_id": row["subject_id"],
            "locus_key": row["locus_key"],
            "disclosure_policy": row["disclosure_policy_id"],
            "learned_at": row["learned_at"],
        }
        for row in rows
    ]


def _print_json(payload: dict[str, Any]) -> int:
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _status_command(*, config: HermesMemoryConfig, store_path: Path, as_json: bool) -> int:
    connection = _connect(store_path)
    try:
        payload = {
            "backend": config.backend.value,
            "store_path": str(store_path),
            "store_exists": store_path.exists(),
            "vector_backend": config.continuity_vector_backend.value,
            "collection_path": str(config.continuity_collection_path),
            "counts": _counts(connection),
        }
        if as_json:
            return _print_json(payload)
        print(f"Backend: {payload['backend']}")
        print(f"Store:   {payload['store_path']}")
        print(f"Exists:  {'yes' if payload['store_exists'] else 'no'}")
        print(f"Vector:  {payload['vector_backend']}")
        print(f"Index:   {payload['collection_path']}")
        print("")
        print("Counts:")
        for table, count in payload["counts"].items():
            print(f"  {table}: {count}")
        return 0
    finally:
        if connection is not None:
            connection.close()


def _sessions_command(*, store_path: Path, limit: int, as_json: bool) -> int:
    connection = _connect(store_path)
    try:
        sessions = _recent_sessions(connection, limit=limit)
        if as_json:
            return _print_json({"store_path": str(store_path), "sessions": sessions})
        if not sessions:
            print("No sessions found.")
            return 0
        for session in sessions:
            print(
                f"{session['created_at']}  {session['session_id']}  "
                f"{session['recall_mode']}  {session['write_frequency']}  {session['session_name']}"
            )
        return 0
    finally:
        if connection is not None:
            connection.close()


def _claims_command(*, store_path: Path, limit: int, as_json: bool) -> int:
    connection = _connect(store_path)
    try:
        claims = _recent_claims(connection, limit=limit)
        if as_json:
            return _print_json({"store_path": str(store_path), "claims": claims})
        if not claims:
            print("No claims found.")
            return 0
        for claim in claims:
            print(
                f"{claim['learned_at']}  {claim['claim_id']}  {claim['claim_type']}  "
                f"{claim['subject_id']}  {claim['locus_key']}  {claim['disclosure_policy']}"
            )
        return 0
    finally:
        if connection is not None:
            connection.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    command = args.command or "status"
    config, store_path = _config(args.db)

    if command == "status":
        return _status_command(config=config, store_path=store_path, as_json=args.json)
    if command == "sessions":
        return _sessions_command(store_path=store_path, limit=args.limit, as_json=args.json)
    if command == "claims":
        return _claims_command(store_path=store_path, limit=args.limit, as_json=args.json)

    parser.error(f"unknown command: {command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
