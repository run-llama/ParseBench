# Changelog

All notable changes to `parse-bench` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

Scoring-relevant changes (anything that can move a leaderboard number) are
called out explicitly so that results produced by different versions can be
compared with care.

## [0.3.0] - 2026-09-02

This release brings the public evaluator back to parity with the internal
LlamaIndex benchmark harness, which had accumulated five months of fixes since
ParseBench was extracted from it. Existing leaderboard rows were produced by the
internal harness and are therefore already consistent with this code; runs made
with `parse-bench` 0.2.0 or earlier are **not** directly comparable to runs made
with this version.

### Scoring (changes evaluation numbers)
- Text similarity: removed an erroneous `/100` that made `text_similarity` ~100x too small.
- Text normalization: combining marks are preserved for Thai, Indic and related scripts;
  abutting inline tags (`</u><s>`) no longer weld adjacent words; trailing dot-leader
  stripping is linear instead of quadratic; shared dot-leader, boolean-marker (checkmark),
  quote and dash normalizers are applied consistently across table metrics.
- Tables: table cell text now *replaces* the table in the rule-augmented text instead of
  being appended, which previously counted every table word two to three times in
  occurrence-counting rules; `<thead>/<tbody>/<tfoot>`, `scope=`, `<colgroup>` and
  `<caption>` are honoured; a `<th>` section-label row inside `<tbody>` is no longer folded
  into the column headers; row provenance uses identity instead of structural equality.
- Table pairing: GriTS and TEDS pair GT and predicted tables per page (`expected_pages`
  metadata) instead of document-wide, and a `<caption>` on one side versus a title band
  on the other is neutralised symmetrically.
- Table record match and header accuracy compare cells leader-insensitively, so `no.`
  and `no` no longer unmatch a whole column.
- Rules: `order` anchors strip residual HTML; bag-of-words rules use a Unicode word class
  and ignore markdown image alt text; degenerate marker-only rules are excluded from the
  denominator instead of scoring 0; a per-document rule budget and a timeout around
  normalization stop a single pathological page from stalling a run.
- Formatting rules: emphasis is resolved with a delimiter-run pairer instead of regex,
  typographic and ASCII quotes are folded, and `<strong>/<em>/<mark>/<sup>/<sub>` with
  attributes are recognised.
- `ParseTestCase` accepts layout and extract-field rules on `test_rules` and carries `metadata`.
- Rule schemas and classes for `table_marker_cells`, `text_color`, `absent_unless_strikeout`,
  `present_as_strikeout`, `is_not_latex` and the extended `form_field` rule (`label` list,
  `label_max_diffs`, `value_max_diffs`); `is_latex` requires `formula`. Not used by the public dataset.

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
