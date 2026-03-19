#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_HERMES_DIR="$HOME/.hermes/hermes-agent"
TARGET_HERMES_DIR="${1:-${HERMES_INSTALL_DIR:-$DEFAULT_HERMES_DIR}}"
HONCHO_CONFIG_PATH="${HONCHO_CONFIG_PATH:-$HOME/.honcho/config.json}"
HERMES_HOST="${HERMES_HOST:-hermes}"
PLUGIN_FACTORY="${PLUGIN_FACTORY:-continuity.hermes_compat.plugin:create_backend}"
SKIP_PIP_INSTALL="${SKIP_PIP_INSTALL:-0}"
INSTALL_ZVEC="${INSTALL_ZVEC:-1}"
RESTART_GATEWAY="${RESTART_GATEWAY:-0}"

CONTINUITY_STORE_PATH="${CONTINUITY_STORE_PATH:-$HOME/.hermes/continuity.db}"
CONTINUITY_COLLECTION_PATH="${CONTINUITY_COLLECTION_PATH:-$HOME/.hermes/continuity-zvec}"
CONTINUITY_VECTOR_BACKEND="${CONTINUITY_VECTOR_BACKEND:-zvec}"
CONTINUITY_EMBEDDING_DIMENSIONS="${CONTINUITY_EMBEDDING_DIMENSIONS:-768}"
CONTINUITY_EMBEDDING_MODEL="${CONTINUITY_EMBEDDING_MODEL:-nomic-embed-text}"
CONTINUITY_EMBEDDING_BASE_URL="${CONTINUITY_EMBEDDING_BASE_URL:-http://127.0.0.1:11434}"
CONTINUITY_REASONING_MODEL="${CONTINUITY_REASONING_MODEL:-gpt-5.4}"
CONTINUITY_REASONING_EFFORT="${CONTINUITY_REASONING_EFFORT:-low}"
CONTINUITY_POLICY_NAME="${CONTINUITY_POLICY_NAME:-hermes_v1}"

TARGET_PYTHON="$TARGET_HERMES_DIR/venv/bin/python"
TARGET_HERMES_BIN="$TARGET_HERMES_DIR/venv/bin/hermes"

export HERMES_HOST
export PLUGIN_FACTORY
export CONTINUITY_STORE_PATH
export CONTINUITY_COLLECTION_PATH
export CONTINUITY_VECTOR_BACKEND
export CONTINUITY_EMBEDDING_DIMENSIONS
export CONTINUITY_EMBEDDING_MODEL
export CONTINUITY_EMBEDDING_BASE_URL
export CONTINUITY_REASONING_MODEL
export CONTINUITY_REASONING_EFFORT
export CONTINUITY_POLICY_NAME

if [[ ! -x "$TARGET_PYTHON" ]]; then
  echo "Missing Hermes venv python at $TARGET_PYTHON" >&2
  exit 1
fi

if [[ "$SKIP_PIP_INSTALL" != "1" ]]; then
  "$TARGET_PYTHON" -m pip install --upgrade pip setuptools wheel

  extras="reasoning-openai"
  install_args=(-e "${ROOT_DIR}[${extras}]")
  if [[ "$INSTALL_ZVEC" == "1" ]]; then
    "$TARGET_PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 10) or sys.version_info >= (3, 13):
    raise SystemExit(
        "retrieval-zvec install requires Python 3.10 through 3.12; "
        "rerun with INSTALL_ZVEC=0 or a compatible Hermes venv"
    )
PY
    extras="${extras},retrieval-zvec"
    install_args=(-e "${ROOT_DIR}[${extras}]" --no-build-isolation)
  fi

  "$TARGET_PYTHON" -m pip install "${install_args[@]}"
fi

mkdir -p "$(dirname "$HONCHO_CONFIG_PATH")"

python3 - "$HONCHO_CONFIG_PATH" <<'PY'
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

config_path = Path(sys.argv[1]).expanduser()
backup_path = None
if config_path.exists():
    backup_path = config_path.with_name(
        f"{config_path.name}.backup-install-hermes-plugin"
    )
    shutil.copyfile(config_path, backup_path)

try:
    data = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
except (OSError, json.JSONDecodeError):
    data = {}

if not isinstance(data, dict):
    data = {}

host = os.environ["HERMES_HOST"]
hosts = data.setdefault("hosts", {})
host_block = hosts.get(host)
if not isinstance(host_block, dict):
    host_block = {}
hosts[host] = host_block

host_block.setdefault("enabled", True)
host_block.setdefault("workspace", host)
host_block.setdefault("aiPeer", host)
host_block.setdefault("memoryMode", "hybrid")
host_block.setdefault("recallMode", "hybrid")
host_block.setdefault("writeFrequency", "async")
host_block.setdefault("sessionStrategy", "per-session")

experimental = host_block.get("experimental")
if not isinstance(experimental, dict):
    experimental = {}
host_block["experimental"] = experimental
experimental["memory_backend_factory"] = os.environ["PLUGIN_FACTORY"]

continuity = host_block.get("continuity")
if not isinstance(continuity, dict):
    continuity = {}
host_block["continuity"] = continuity
continuity.update(
    {
        "storePath": os.environ["CONTINUITY_STORE_PATH"],
        "vectorBackend": os.environ["CONTINUITY_VECTOR_BACKEND"],
        "collectionPath": os.environ["CONTINUITY_COLLECTION_PATH"],
        "embeddingDimensions": int(os.environ["CONTINUITY_EMBEDDING_DIMENSIONS"]),
        "embeddingModel": os.environ["CONTINUITY_EMBEDDING_MODEL"],
        "embeddingBaseUrl": os.environ["CONTINUITY_EMBEDDING_BASE_URL"],
        "reasoningModel": os.environ["CONTINUITY_REASONING_MODEL"],
        "reasoningEffort": os.environ["CONTINUITY_REASONING_EFFORT"],
        "policyName": os.environ["CONTINUITY_POLICY_NAME"],
    }
)

config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

print(f"Updated {config_path}")
if backup_path is not None:
    print(f"Backup saved to {backup_path}")
PY

if [[ "$RESTART_GATEWAY" == "1" ]]; then
  if [[ -x "$TARGET_HERMES_BIN" ]]; then
    if sudo -n "$TARGET_HERMES_BIN" gateway restart --system >/dev/null 2>&1; then
      echo "Restarted Hermes gateway system service"
    elif "$TARGET_HERMES_BIN" gateway restart >/dev/null 2>&1; then
      echo "Restarted Hermes gateway user service"
    else
      echo "Gateway restart failed; restart it manually" >&2
      exit 1
    fi
  else
    echo "Missing Hermes launcher at $TARGET_HERMES_BIN" >&2
    exit 1
  fi
fi

cat <<EOF
Continuity Hermes plugin install complete.
Target Hermes install: $TARGET_HERMES_DIR
Honcho config: $HONCHO_CONFIG_PATH
Plugin factory: $PLUGIN_FACTORY

Next steps:
  1. Check status: $TARGET_PYTHON - <<'PY'
from argparse import Namespace
from honcho_integration.cli import cmd_status
cmd_status(Namespace())
PY
  2. If you did not set RESTART_GATEWAY=1, restart the gateway now.
EOF
