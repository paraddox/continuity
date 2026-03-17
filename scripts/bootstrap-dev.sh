#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_ZVEC="${INSTALL_ZVEC:-0}"
RUN_ZVEC_SMOKE="${RUN_ZVEC_SMOKE:-${INSTALL_ZVEC}}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
. "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
EXTRAS="dev,reasoning-openai"
INSTALL_ARGS=(-e ".[${EXTRAS}]")

if [[ "${INSTALL_ZVEC}" == "1" ]]; then
  python - <<'PY'
import sys

if sys.version_info < (3, 10) or sys.version_info >= (3, 13):
    raise SystemExit(
        "retrieval-zvec bootstrap requires Python 3.10 through 3.12; "
        "rerun with INSTALL_ZVEC=1 PYTHON_BIN=python3.12"
    )
PY
  EXTRAS="${EXTRAS},retrieval-zvec"
  INSTALL_ARGS=(-e ".[${EXTRAS}]" --no-build-isolation)
fi

python -m pip install "${INSTALL_ARGS[@]}"
python -m pytest

if [[ "${RUN_ZVEC_SMOKE}" == "1" ]]; then
  python -m continuity.index.zvec_smoke
fi
