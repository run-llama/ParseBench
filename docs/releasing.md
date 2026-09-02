# Releasing parse-bench

Releases are cut from `main` and published to PyPI automatically by
`.github/workflows/publish.yml` when a `v*` tag is pushed.

## Versioning

`parse-bench` follows semantic versioning with one benchmark-specific rule:

- **Patch** (`0.3.x`): provider fixes, new pipelines, docs, tooling. Scores for
  existing pipelines on the public dataset do not change.
- **Minor** (`0.x.0`): anything that can change a score on the public dataset
  (metric fixes, rule matching changes, aggregation changes) or a breaking CLI /
  API change. Call these out under a "Scoring" heading in `CHANGELOG.md`.

Evaluation outputs record the `parse-bench` version so leaderboard rows can be
traced to the code that produced them.

## Steps

1. Update `CHANGELOG.md`: move items from `[Unreleased]` under a new
   `## [X.Y.Z] - YYYY-MM-DD` heading.
2. Bump `__version__` in `src/parse_bench/__init__.py` (the single source of truth;
   `pyproject.toml` reads it at build time).
3. Refresh the lockfile and run the checks:
   ```bash
   uv lock
   uv sync --extra runners --extra fast --extra dev
   uv run ruff check src tests && uv run ruff format --check src tests
   uv run pytest -q
   uv build
   ```
4. Open a PR with those changes and merge it.
5. Tag and push:
   ```bash
   git tag vX.Y.Z && git push origin vX.Y.Z
   ```
   The publish workflow verifies that the tag matches `__version__`, builds the
   sdist and wheel, publishes to PyPI via trusted publishing, and creates a GitHub
   release with auto-generated notes.

## One-time PyPI setup

The publish workflow uses [trusted publishing](https://docs.pypi.org/trusted-publishers/),
so no PyPI token is stored in GitHub.

1. On pypi.org, create the `parse-bench` project (or use "Add a new pending publisher"
   if the project does not exist yet).
2. Under *Publishing*, add a GitHub publisher: owner `run-llama`, repository
   `ParseBench`, workflow `publish.yml`, environment `pypi`.
3. In the GitHub repository settings, create an environment named `pypi`. Optionally
   require reviewers so a tag push cannot publish without approval.

## Consuming the package from another project

```bash
uv add "parse-bench[runners]"          # or pip install "parse-bench[runners]"
uv add "parse-bench[llamaparse]"       # just one provider
```

Pin an exact version in evaluation harnesses so scores are reproducible:

```toml
dependencies = ["parse-bench==0.3.0"]
```
