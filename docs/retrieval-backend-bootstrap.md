# Retrieval Backend Bootstrap

Continuity's optional real `zvec` backend currently needs Python `3.10` through `3.12`.
In this checkout on 2026-03-17, `python3` is `3.13.7`, so use `python3.12` for the bootstrap path below.

## Fresh checkout bootstrap

You can use the repo bootstrap script for the full zvec-capable setup:

```bash
INSTALL_ZVEC=1 PYTHON_BIN=python3.12 VENV_DIR=.venv-zvec ./scripts/bootstrap-dev.sh
```

That creates the venv, installs the dev/bootstrap extras plus
`retrieval-zvec`, runs the repo tests, and then runs the real zvec smoke check.

If you want the lower-level manual path instead, use:

```bash
python3.12 -m venv .venv-zvec
. .venv-zvec/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install ".[retrieval-zvec]" --no-build-isolation
python -m continuity.index.zvec_smoke
```

The smoke command prints a JSON summary after it has:
- created a real `zvec` collection
- indexed two sample documents
- queried for the expected top hit
- deleted one document and verified the deletion

## Host note

Continuity pins `zvec>=0.2.1b0,<0.3` in the `retrieval-zvec` extra because the published `zvec 0.2.0`
Linux `x86_64` wheel can crash with `Illegal instruction` on CPUs that do not expose `AVX-512`.
That failure was reproduced on this host (`AMD Ryzen 9 5900X`), where `zvec 0.2.0` installed cleanly in a
fresh Python `3.12` venv but died on plain `import zvec`, while `zvec 0.2.1b0` imported cleanly and passed
the Continuity smoke helper.

If you need to verify a manually installed wheel or source checkout directly, run:

```bash
PYTHONPATH=src python3.12 -m continuity.index.zvec_smoke
```

If `zvec 0.2.1b0` still crashes on your machine or no compatible wheel exists for your platform, install a
compatible source build of `zvec` before rerunning the smoke command.

## Persistent smoke collection

If you want to inspect the collection directory afterward, pass a path that does not already exist:

```bash
. .venv-zvec/bin/activate
python -m continuity.index.zvec_smoke --collection-path .tmp/zvec-smoke
```
