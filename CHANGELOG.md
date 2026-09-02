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
- Visual grounding: the LlamaParse V2 `code` item type is mapped (code blocks were
  silently dropped); LlamaParse attribution is restricted to layout-aware segments
  rather than falling back to coarse item bboxes; Docling attribution blocks no longer
  fall back to the item bbox.
- Extract accuracy: a missing object or array is weighted by its leaf count instead of 1;
  date normalization handles `MM/DD/YY`, missing spaces after commas and out-of-range
  years; keyless lists are aligned order-invariantly; nullable numeric fields treat `0`
  and `null` as equal when the schema allows null.
- Inference providers: 13 self-hosted VLM/OCR providers (Chandra 2, DeepSeek-OCR 2,
  Falcon OCR, Gemma 4, Granite Vision, Infinity Parser 2, MinerU 2.5/2605/Diffusion,
  Nemotron Omni, PaddleOCR, Surya 2, Unlimited-OCR) rendered only the first page of a
  multi-page PDF; Reducto output was truncated to the first chunk; Gemini empty and
  RECITATION responses are retried instead of scored as empty output;
  `amazon_nova_2_lite_parse_with_layout` now actually enables layout mode; layout
  checkbox predictions are de-duplicated.
- Cost columns: Pulse cost used an account-wide cumulative page counter; LlamaParse now
  bills cost-optimised pages at the cost-effective rate; OpenAI GPT-5 Mini and GPT-5.6
  prices corrected; AWS Textract and Extend now report cost; Anthropic prompt-cache
  tokens are priced. `parse-bench inference renormalize` re-prices saved runs via
  `Provider.recompute_cost` without re-running inference.
- Leaderboard: the *LiteParse (no OCR)* row was re-run with LiteParse 2.14.3 and this release's scoring (overall 32.8 -> 36.9; visual grounding 10.7 -> 31.8 now that layout blocks are scored).
- Aggregation: `micro_*` aggregates are emitted for standalone F1, accuracy and pass-rate
  metrics; fractional `passed` counts are no longer dropped from `total_*` metrics.
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
- `liteparse` extra installs the released `lit` CLI from PyPI; the provider finds it on `PATH`.
- LlamaParse normalisation rewrites `layout_pages[*].items[*].type` to canonical layout classes (splitting items whose segments disagree) and synthesises checkbox mark items, matching the internal harness's output contract. ParseBench's own layout scores are unchanged; it lets other consumers of saved outputs classify without the raw-label adapter.
- LiteParse now requests `--extract-blocks` and emits `layout_pages` from the block kinds and bboxes, with a matching layout adapter and label mapper, so the Visual Grounding column can be scored (set `extract_blocks: false` in the pipeline config to disable).
- Runner: `per_file_timeout` (CLI > pipeline > default 1800s), randomised external
  filenames for third-party providers, a hint when a corpus contains only unsupported
  extensions.
- New pipelines: Reducto change-tracking / agentic table / agentic chart, GPT-5.4
  reasoning-none, Sonnet 5 parse-with-layout, Gemini 3.6 / 3.7 / 3.1 Flash Lite
  layout variants, Mistral OCR 4.1, Nemotron Omni vLLM, Qwen3.8 Flash Next.
- Evaluation worker pool: a hung document is now killed and retried once instead of
  blocking the run forever; `*.images/` artifact directories are skipped.
- Comparison reports pick the primary metric from an ordered candidate chain and read
  predictions from `layout_pages`; run labels keep their dataset suffix.
- `parse_bench_version` is recorded in `_metadata.json` and `_evaluation_report.json`.
- CI (lint, tests, wheel smoke test) and a tag-driven PyPI publish workflow.

### Changed
- Dropped unused core dependencies `datasets` (which pulled in pyarrow, 123 MB) and `tqdm`; a base install is now ~280 MB instead of ~410 MB.
- `markdown2` floor raised to 2.5.5: 2.5.4 renders `*`/`_` runs inside table cells differently, which changed 12 of 2078 LiteParse outputs between environments.
- The package version is single-sourced from `parse_bench.__version__`.
- `.env` discovery now walks up from the current directory instead of assuming a repo checkout.
- The optional LlamaParse job-log HTML renderer is configured with `PARSE_BENCH_LOG_VIEWER`
  instead of a hard-coded sibling-directory path.
- The `liteparse` provider locates the `lit` binary via `LITEPARSE_BIN` or `PATH`.

## [0.2.0] - 2026-09-01

Last version distributed as a source checkout only. See the git history for details.
