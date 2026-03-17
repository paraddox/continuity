# Continuity

Continuity is a local-first memory engine for Hermes continuity.

## Requirements

- Python `3.11+` for the core package
- Python `3.12` if you want the optional real `retrieval-zvec` backend
- local Ollama plus `nomic-embed-text` if you want live embedding checks
- `OPENAI_API_KEY` if you want live Codex/OpenAI adapter checks

## Core install

For a plain editable install of the package:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

## Dev bootstrap

For the normal contributor setup, use the repo bootstrap script:

```bash
./scripts/bootstrap-dev.sh
```

That creates `.venv`, upgrades packaging tooling, installs the package in
editable mode with `dev` and `reasoning-openai` extras, and runs the test suite.

If you want to do the same steps manually:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,reasoning-openai]"
python -m pytest
```

If you also want the optional `retrieval-zvec` backend in a dedicated venv, use
the same script in opt-in mode:

```bash
INSTALL_ZVEC=1 PYTHON_BIN=python3.12 VENV_DIR=.venv-zvec ./scripts/bootstrap-dev.sh
```

That installs the normal dev/bootstrap extras plus `retrieval-zvec`, then runs
the repo test suite and the `continuity.index.zvec_smoke` check.

## Optional runtime extras

### OpenAI / Codex adapter

The OpenAI Python SDK is kept optional. Install it through the
`reasoning-openai` extra:

```bash
python -m pip install -e ".[reasoning-openai]"
```

Live adapter verification is opt-in:

```bash
CONTINUITY_RUN_LIVE_OPENAI=1 OPENAI_API_KEY=your-key \
uv run --isolated -p 3.13 --with pytest python -m pytest tests/test_codex_adapter.py
```

### Ollama embeddings

The Ollama client uses the Python standard library, so there is no extra Python
package to install. You do need a local Ollama server and the
`nomic-embed-text` model available.

Live Ollama verification is opt-in:

```bash
CONTINUITY_RUN_LIVE_OLLAMA=1 \
uv run --isolated -p 3.13 --with pytest python -m pytest tests/test_ollama_embeddings.py
```

### retrieval-zvec backend

The optional real `retrieval-zvec` backend is documented separately in
[`docs/retrieval-backend-bootstrap.md`](docs/retrieval-backend-bootstrap.md).
The short version is:

```bash
INSTALL_ZVEC=1 PYTHON_BIN=python3.12 VENV_DIR=.venv-zvec ./scripts/bootstrap-dev.sh
```

If you want the lower-level manual path instead:

```bash
python3.12 -m venv .venv-zvec
. .venv-zvec/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[retrieval-zvec]" --no-build-isolation
python -m continuity.index.zvec_smoke
```

## Common validation commands

```bash
uv run --isolated -p 3.11 --with pytest python -m pytest
uv run --isolated -p 3.13 --with pytest python -m pytest
uv run --isolated -p 3.11 --with build python -m build
```

## Notes

- `dev` currently provides the repo's basic build/test tools: `pytest` and
  `build`.
- `reasoning-openai` installs the optional `openai` SDK used by the live
  adapter path.
- The repo does not currently have CI, so these commands are run manually in
  the normal beads workflow.
