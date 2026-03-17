# Retrieval Backend Bootstrap

Continuity's optional real `zvec` backend currently needs Python `3.10` through `3.12`.
In this checkout on 2026-03-17, `python3` is `3.13.7`, so use `python3.12` for the bootstrap path below.

## Fresh checkout bootstrap

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

On this host, the published `zvec 0.2.0` wheel installed in a clean Python `3.12` venv but crashed during import with `Illegal instruction`.
The Continuity smoke helper itself was verified against a source-built `zvec 0.2.1.dev4` under system `python3.12`:

```bash
PYTHONPATH=src python3.12 -m continuity.index.zvec_smoke
```

If the wheel crashes on your machine too, use a compatible source build of `zvec` before rerunning the smoke command.

## Persistent smoke collection

If you want to inspect the collection directory afterward, pass a path that does not already exist:

```bash
. .venv-zvec/bin/activate
python -m continuity.index.zvec_smoke --collection-path .tmp/zvec-smoke
```
