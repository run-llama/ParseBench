# Changelog

All notable changes to `parse-bench` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- Dropped the `scikit-learn` dependency. Layout-detection average precision is
  now computed with a small numpy implementation that matches
  `sklearn.metrics.average_precision_score` (including tied-score handling),
  so evaluation numbers are unchanged. `scipy` remains a core dependency.

## [1.0.2] - 2026-09-04

Ports two fixes that landed in the internal harness while 1.0.1 was being cut.

### Scoring (changes evaluation numbers)
- `chart_data_point`: number parsing accepts accounting negatives (`(4)`,
  `$(4)`, `($4)`, trailing minus), the `bn` billion suffix, and no longer
  reads a comma followed by a space as a decimal separator. A numeric rule
  value no longer fuzzy-matches a compound cell (`249` vs `249, 188`); a cell
  that carries one numeric token plus an axis unit (`68.7 days`) does match.
  A repeated column header after a body section label counts as local scope
  only when the complete pair precedes the candidate.

### Added
- Granular pages carry a `cells` layer; the LlamaParse adapter reads the
  provider-neutral `granular_layers` on normalized pages before falling back
  to the grounded-page payload.

## [1.0.1] - 2026-09-03

Harness-facing release: no evaluation number changes. Everything here lets a
downstream benchmark harness build on `parse-bench` without overriding
built-in rule classes or patching package models at import time.

### Fixed
- `diagram_graph` rules with a `reference_image` never received the test case
  or source file path from the evaluator, so the reference render was never
  found. `ParseEvaluator` now forwards `source_file_path` and
  `test_case_file_path` to the rule metric.

### Added
- `RuleBasedMetric._prepare_rule(rule, actual, kwargs)`: one hook, run inside
  the per-rule timeout, that hands a freshly created rule its side inputs.
  Injection is attribute-driven, so any rule (built-in or extension) that
  declares `raw_output`, `source_file_path` or `test_case_path` receives the
  matching `compute` keyword argument. Subclasses extend it for
  harness-specific inputs.
- `ChartTableCache`: chart rules on one document share a single parsed-table
  pass. Pass `chart_table_cache=` to `compute` to read the parse back.
- Page-parallel GriTS pairwise scoring: `GriTSMetric(pair_workers=N)`,
  `ParseEvaluator(grits_pair_workers=N)`, or `BENCH_GRITS_PAIR_WORKERS`.
  The evaluation runner splits the CPU budget so `doc_workers x pair_workers`
  stays within the core count. Scores are identical to the sequential path
  (guarded by `test_grits_perf_equivalence`).
- `py.typed` marker: the package is now typed for downstream mypy.
- Rotated-box geometry: `parse_bench.geometry.rotated_bbox` (`xywh_r` <->
  polygon conversion, containment) and rotated polygon IoU / IoA in
  `evaluation.metrics.layoutdet.iou` for datasets whose layout ground truth
  carries an `r` rotation.

### Layout adapters and schemas (changes layout numbers for the providers named)
- Azure Document Intelligence: checkbox items carry `scope=mark`, so mark-scope
  checkbox datasets score them; Textract and Azure DI adapters build granular
  (line / word) pages for granular layout scoring.
- LlamaParse: the adapter also matches results whose `ParseOutput` carries
  `grounded_pages`, and builds granular pages from that payload before falling
  back to the raw response; merged granular bboxes keep a shared rotation `r`.
- Projection prefers `layout_pages[*].items` over the legacy flat
  `predictions` list when any page has items (canonical labels coerced
  directly), and a missing detector score projects as 0.0 in both paths.
- `LayoutOutput` gains the `ParseOutput`-parity fields (`pages`,
  `grounded_pages`, `job_id`); `LayoutDetectionModel` gains
  `OPENAI_COMPATIBLE_VLM_LAYOUT`, `CHECKBOX_DETECTOR_YOLOV8`, `COHERE_PARSE_LAYOUT`.
- Parse IR: `LineNumberIR`, `LinkIR`, `RevisionIR`, `GranularUnitIR` /
  `GranularLayerIR`, `LayoutSegmentIR.r`, `ParseLayoutPageIR.links /
  revisions / granular_layers` and `ParseOutput.grounded_pages`.
- `register_pipeline_resolver` (in `parse_bench.extensions`): a harness with
  its own pipeline registry can map `pipeline_name` to a provider key, so the
  adapter and mapper registries resolve its pipelines too.
- Adapter aliases: `anthropic_haiku` resolves to the Anthropic adapter.

### Changed
- `MetricValue` and `RunStat` accept extra fields (`extra="allow"`) so
  provenance a harness attaches round-trips losslessly.
- `ParseRuleInput` and `ParseTestRule.__init__` accept any `ParseRuleBase`
  subclass, not only the closed built-in union, so extension rule classes
  need no cast to call `super().__init__`.
- `FieldCitation.bbox` is optional: a page-only citation carries `bbox=None`.

## [1.0.0] - 2026-09-02

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
  fall back to the item bbox; layout checkbox predictions are de-duplicated.
- Extract accuracy: a missing object or array is weighted by its leaf count instead of 1;
  date normalization handles `MM/DD/YY`, missing spaces after commas and out-of-range
  years; keyless lists are aligned order-invariantly; nullable numeric fields treat `0`
  and `null` as equal when the schema allows null.
- Aggregation: `micro_*` aggregates are emitted for standalone F1, accuracy and pass-rate
  metrics; fractional `passed` counts are no longer dropped from `total_*` metrics.
- Inference providers: 13 self-hosted VLM/OCR providers (Chandra 2, DeepSeek-OCR 2,
  Falcon OCR, Gemma 4, Granite Vision, Infinity Parser 2, MinerU 2.5/2605/Diffusion,
  Nemotron Omni, PaddleOCR, Surya 2, Unlimited-OCR) rendered only the first page of a
  multi-page PDF; Reducto output was truncated to the first chunk; Gemini empty and
  RECITATION responses are retried instead of scored as empty output;
  `amazon_nova_2_lite_parse_with_layout` now actually enables layout mode.
- Cost columns: Pulse cost used an account-wide cumulative page counter; LlamaParse now
  bills cost-optimised pages at the cost-effective rate; OpenAI GPT-5 Mini and GPT-5.6
  prices corrected; AWS Textract and Extend now report cost; Anthropic prompt-cache
  tokens are priced. `parse-bench inference renormalize` re-prices saved runs via
  `Provider.recompute_cost` without re-running inference.

### Added
- Rule types from the internal harness: `heading_structure`, `list_level`, `page_decoration`,
  `watermark_removal`, `diagram_graph` / `diagram_edge` / `diagram_count`, `table_marker_cells`,
  `text_color`, `absent_unless_strikeout`, `present_as_strikeout`, `is_not_latex`; metadata-only
  `table_merging` and `cross_page_table_consistency` metrics; extended `form_field` rule
  (`label` list, `label_max_diffs`, `value_max_diffs`). None are used by the public dataset yet.
- `ParseTestCase` accepts layout and extract-field rules on `test_rules` and carries `metadata`.
- Evaluation worker pool: a hung document is now killed and retried once instead of
  blocking the run forever; `*.images/` artifact directories are skipped.
- Runner: `per_file_timeout` (CLI > pipeline > default 1800s), randomised external
  filenames for third-party providers, a hint when a corpus contains only unsupported
  extensions.
- New pipelines: Reducto change-tracking / agentic table / agentic chart, GPT-5.4
  reasoning-none, Sonnet 5 parse-with-layout, Gemini 3.6 / 3.7 / 3.1 Flash Lite
  layout variants, Mistral OCR 4.1, Nemotron Omni vLLM, Qwen3.8 Flash Next.
- `parse_bench_version` is recorded in `_metadata.json` and `_evaluation_report.json`.
- Leaderboard: the *LiteParse (no OCR)* row was re-run with LiteParse 2.14.3 and this release's scoring (overall 32.8 -> 36.9; visual grounding 10.7 -> 31.8 now that layout blocks are scored).
- `parse-bench` is now published to PyPI. Install with `pip install "parse-bench[runners]"`.
- Per-provider extras (`llamaparse`, `openai`, `anthropic`, `google`, `azure`, `aws`,
  `reducto`, `datalab`, `landingai`, `unstructured`, `chunkr`, `extend`, `docling`, `local`)
  so a runner can be installed without every provider SDK. `runners` remains the union.
- `parse-bench version` command.
- `liteparse` extra installs the released `lit` CLI from PyPI; the provider finds it on `PATH`.
- LlamaParse normalisation rewrites `layout_pages[*].items[*].type` to canonical layout classes (splitting items whose segments disagree) and synthesises checkbox mark items, matching the internal harness's output contract. ParseBench's own layout scores are unchanged; it lets other consumers of saved outputs classify without the raw-label adapter.
- LiteParse now requests `--extract-blocks` and emits `layout_pages` from the block kinds and bboxes, with a matching layout adapter and label mapper, so the Visual Grounding column can be scored (set `extract_blocks: false` in the pipeline config to disable).
- `markdown2` floor raised to 2.5.5: 2.5.4 renders `*`/`_` runs inside table cells differently, which changed 12 of 2078 LiteParse outputs between environments.
- CI (lint, tests, wheel smoke test) and a tag-driven PyPI publish workflow.

### Changed
- Dropped unused core dependencies `datasets` (which pulled in pyarrow, 123 MB) and `tqdm`; a base install is now ~280 MB instead of ~410 MB.
- The package version is single-sourced from `parse_bench.__version__`.
- `.env` discovery now walks up from the current directory instead of assuming a repo checkout.
- The optional LlamaParse job-log HTML renderer is configured with `PARSE_BENCH_LOG_VIEWER`
  instead of a hard-coded sibling-directory path.
- The `liteparse` provider locates the `lit` binary via `LITEPARSE_BIN` or `PATH`.

## [0.2.0] - 2026-09-01

Last version distributed as a source checkout only. See the git history for details.
