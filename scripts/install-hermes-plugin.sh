#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_HERMES_DIR="$HOME/.hermes/hermes-agent"
TARGET_HERMES_DIR="${1:-${HERMES_INSTALL_DIR:-$DEFAULT_HERMES_DIR}}"
HONCHO_CONFIG_PATH="${HONCHO_CONFIG_PATH:-$HOME/.honcho/config.json}"
HERMES_CONFIG_PATH="${HERMES_CONFIG_PATH:-$HOME/.hermes/config.yaml}"
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
CONTINUITY_REASONING_TARGET_NAME="${CONTINUITY_REASONING_TARGET_NAME:-}"
CONTINUITY_REASONING_PROVIDER="${CONTINUITY_REASONING_PROVIDER:-}"
CONTINUITY_REASONING_TARGET_MODEL="${CONTINUITY_REASONING_TARGET_MODEL:-}"
CONTINUITY_REASONING_TARGET_EFFORT="${CONTINUITY_REASONING_TARGET_EFFORT:-}"
CONTINUITY_REASONING_SELECTION="${CONTINUITY_REASONING_SELECTION:-}"
CONTINUITY_CREATE_PROVIDER_NAME="${CONTINUITY_CREATE_PROVIDER_NAME:-}"
CONTINUITY_CREATE_PROVIDER_BASE_URL="${CONTINUITY_CREATE_PROVIDER_BASE_URL:-}"
CONTINUITY_CREATE_PROVIDER_API_KEY="${CONTINUITY_CREATE_PROVIDER_API_KEY:-}"
CONTINUITY_CREATE_PROVIDER_MODEL="${CONTINUITY_CREATE_PROVIDER_MODEL:-}"
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
export CONTINUITY_REASONING_TARGET_NAME
export CONTINUITY_REASONING_PROVIDER
export CONTINUITY_REASONING_TARGET_MODEL
export CONTINUITY_REASONING_TARGET_EFFORT
export CONTINUITY_REASONING_SELECTION
export CONTINUITY_CREATE_PROVIDER_NAME
export CONTINUITY_CREATE_PROVIDER_BASE_URL
export CONTINUITY_CREATE_PROVIDER_API_KEY
export CONTINUITY_CREATE_PROVIDER_MODEL
export CONTINUITY_POLICY_NAME
export HERMES_CONFIG_PATH

if [[ -z "$CONTINUITY_REASONING_SELECTION" && -t 0 && -t 1 ]]; then
  echo "Select Continuity reasoning target:"
  echo "  1. Keep current Continuity reasoningModel/reasoningEffort"
  echo "  2. Use current Hermes active model"
  echo "  3. Use an existing Hermes custom provider"
  echo "  4. Create a new Hermes custom provider for Continuity"
  read -r -p "Choice [1-4]: " _continuity_reasoning_choice
  case "${_continuity_reasoning_choice:-1}" in
    2)
      CONTINUITY_REASONING_SELECTION="active-model"
      export CONTINUITY_REASONING_SELECTION
      ;;
    3)
      CONTINUITY_REASONING_SELECTION="custom-provider"
      export CONTINUITY_REASONING_SELECTION
      if [[ -z "$CONTINUITY_REASONING_TARGET_NAME" ]]; then
        read -r -p "Existing Hermes custom provider name: " CONTINUITY_REASONING_TARGET_NAME
        export CONTINUITY_REASONING_TARGET_NAME
      fi
      ;;
    4)
      CONTINUITY_REASONING_SELECTION="create-custom-provider"
      export CONTINUITY_REASONING_SELECTION
      if [[ -z "$CONTINUITY_CREATE_PROVIDER_NAME" ]]; then
        read -r -p "New provider name: " CONTINUITY_CREATE_PROVIDER_NAME
        export CONTINUITY_CREATE_PROVIDER_NAME
      fi
      if [[ -z "$CONTINUITY_CREATE_PROVIDER_BASE_URL" ]]; then
        read -r -p "Provider base URL: " CONTINUITY_CREATE_PROVIDER_BASE_URL
        export CONTINUITY_CREATE_PROVIDER_BASE_URL
      fi
      if [[ -z "$CONTINUITY_CREATE_PROVIDER_API_KEY" ]]; then
        read -r -p "Provider API key (optional): " CONTINUITY_CREATE_PROVIDER_API_KEY
        export CONTINUITY_CREATE_PROVIDER_API_KEY
      fi
      if [[ -z "$CONTINUITY_CREATE_PROVIDER_MODEL" ]]; then
        read -r -p "Provider model: " CONTINUITY_CREATE_PROVIDER_MODEL
        export CONTINUITY_CREATE_PROVIDER_MODEL
      fi
      ;;
    *)
      ;;
  esac
fi

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
mkdir -p "$(dirname "$HERMES_CONFIG_PATH")"

python3 - "$HONCHO_CONFIG_PATH" "$HERMES_CONFIG_PATH" <<'PY'
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
import yaml


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    return data


def _load_yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, yaml.YAMLError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    return data


def _backup_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    backup_path = path.with_name(f"{path.name}.backup-install-hermes-plugin")
    shutil.copyfile(path, backup_path)
    return backup_path


def _clean(value: object) -> str:
    return str(value or "").strip()


def _normalized_name(value: str) -> str:
    return value.strip().lower().replace(" ", "-")

config_path = Path(sys.argv[1]).expanduser()
hermes_config_path = Path(sys.argv[2]).expanduser()
backup_path = _backup_file(config_path)
hermes_backup_path = None

data = _load_json(config_path)
hermes_config = _load_yaml(hermes_config_path)

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

target_name = os.environ.get("CONTINUITY_REASONING_TARGET_NAME", "").strip()
target_provider = os.environ.get("CONTINUITY_REASONING_PROVIDER", "").strip()
target_model = os.environ.get("CONTINUITY_REASONING_TARGET_MODEL", "").strip()
target_effort = os.environ.get("CONTINUITY_REASONING_TARGET_EFFORT", "").strip()
selection = _clean(os.environ.get("CONTINUITY_REASONING_SELECTION")).lower()

if not any((target_name, target_provider, target_model, target_effort)) and selection:
    if selection == "active-model":
        model_cfg = hermes_config.get("model")
        if not isinstance(model_cfg, dict):
            raise SystemExit("Hermes config does not define an active model to reuse")
        target_provider = _clean(model_cfg.get("provider"))
        target_model = _clean(model_cfg.get("default") or model_cfg.get("name"))
        target_effort = _clean(
            model_cfg.get("reasoning_effort")
            or (hermes_config.get("agent") or {}).get("reasoning_effort")
        )
        if not target_provider or not target_model:
            raise SystemExit("Hermes active model is missing provider or model")
    elif selection == "custom-provider":
        if not target_name:
            raise SystemExit("CONTINUITY_REASONING_TARGET_NAME is required for custom-provider selection")
    elif selection == "create-custom-provider":
        provider_name = _clean(os.environ.get("CONTINUITY_CREATE_PROVIDER_NAME"))
        provider_base_url = _clean(os.environ.get("CONTINUITY_CREATE_PROVIDER_BASE_URL"))
        provider_api_key = _clean(os.environ.get("CONTINUITY_CREATE_PROVIDER_API_KEY"))
        provider_model = _clean(os.environ.get("CONTINUITY_CREATE_PROVIDER_MODEL"))
        if not provider_name or not provider_base_url or not provider_model:
            raise SystemExit(
                "create-custom-provider requires CONTINUITY_CREATE_PROVIDER_NAME, "
                "CONTINUITY_CREATE_PROVIDER_BASE_URL, and CONTINUITY_CREATE_PROVIDER_MODEL"
            )

        custom_providers = hermes_config.get("custom_providers")
        if not isinstance(custom_providers, list):
            custom_providers = []
            hermes_config["custom_providers"] = custom_providers

        normalized = _normalized_name(provider_name)
        updated = False
        for entry in custom_providers:
            if not isinstance(entry, dict):
                continue
            if _normalized_name(_clean(entry.get("name"))) != normalized:
                continue
            entry["name"] = provider_name
            entry["base_url"] = provider_base_url
            entry["model"] = provider_model
            if provider_api_key:
                entry["api_key"] = provider_api_key
            updated = True
            break

        if not updated:
            new_entry = {
                "name": provider_name,
                "base_url": provider_base_url,
                "model": provider_model,
            }
            if provider_api_key:
                new_entry["api_key"] = provider_api_key
            custom_providers.append(new_entry)

        hermes_backup_path = _backup_file(hermes_config_path)
        hermes_config_path.write_text(
            yaml.safe_dump(hermes_config, sort_keys=False),
            encoding="utf-8",
        )
        target_name = provider_name
        target_provider = ""
        target_model = ""
        target_effort = ""
    else:
        raise SystemExit(f"Unknown CONTINUITY_REASONING_SELECTION={selection!r}")

if any((target_name, target_provider, target_model, target_effort)):
    reasoning_target = continuity.get("reasoningTarget")
    if not isinstance(reasoning_target, dict):
        reasoning_target = {}
    continuity["reasoningTarget"] = reasoning_target
    if target_name:
        reasoning_target["targetName"] = target_name
    if target_provider:
        reasoning_target["provider"] = target_provider
    if target_model:
        reasoning_target["model"] = target_model
    if target_effort:
        reasoning_target["reasoningEffort"] = target_effort

config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

print(f"Updated {config_path}")
if backup_path is not None:
    print(f"Backup saved to {backup_path}")
if hermes_backup_path is not None:
    print(f"Hermes config backup saved to {hermes_backup_path}")
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
