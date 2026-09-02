# Changelog

All notable changes to `parse-bench` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

Scoring-relevant changes (anything that can move a leaderboard number) are
called out explicitly so that results produced by different versions can be
compared with care.

## [0.3.0] - 2026-09-02

### Added
- `parse-bench` is now published to PyPI. Install with `pip install "parse-bench[runners]"`.
- Per-provider extras (`llamaparse`, `openai`, `anthropic`, `google`, `azure`, `aws`,
  `reducto`, `datalab`, `landingai`, `unstructured`, `chunkr`, `extend`, `docling`, `local`)
  so a runner can be installed without every provider SDK. `runners` remains the union.
- `parse-bench version` command.
- CI (lint, tests, wheel smoke test) and a tag-driven PyPI publish workflow.

### Changed
- Dropped unused core dependencies `datasets` (which pulled in pyarrow, 123 MB) and `tqdm`; a base install is now ~280 MB instead of ~410 MB.
- `markdown2` floor raised to 2.5.5: 2.5.4 renders `*`/`_` runs inside table cells differently, which changed 12 of 2078 LiteParse outputs between environments.
- The package version is single-sourced from `parse_bench.__version__`.
- `.env` discovery now walks up from the current directory instead of assuming a repo checkout.

## [0.2.0] - 2026-09-01

Last version distributed as a source checkout only. See the git history for details.
